"""Fast forked in-memory test runner with baseline stage checks and slack short-circuiting."""

from __future__ import annotations

from concurrent.futures import Future
from contextlib import nullcontext
import importlib
import importlib.machinery
import marshal
import math
import os
from pathlib import Path
import pickle
import select
import signal
import sys
import tempfile
import time
import types
import unittest

from build_maps import compute_plan_py_sig
import packer
import patch_engine

_LAZY_TARGETS = frozenset({"duckdb", "pandas", "pandas.testing", "numpy"})


class _LazyCallable:
    """Defer importing real third-party function (e.g. assert_frame_equal) until called."""

    def __init__(self, mod_name: str, attr_name: str):
        self._mod_name = mod_name
        self._attr_name = attr_name
        self._fn = None

    def __call__(self, *args, **kwargs):
        if self._fn is None:
            _materialize_lazy(self._mod_name)
            self._fn = getattr(sys.modules[self._mod_name], self._attr_name)
        return self._fn(*args, **kwargs)


class _LazyModule(types.ModuleType):
    """Module proxy that defers loading heavy C extensions until actually needed."""

    def __init__(self, name: str):
        super().__init__(name)
        is_pkg = "." not in name
        self.__dict__["__lazy_name__"] = name
        if is_pkg:
            self.__dict__["__path__"] = []
        self.__dict__["__package__"] = name if is_pkg else name.rpartition(".")[0]
        self.__dict__["__spec__"] = importlib.machinery.ModuleSpec(name, None, is_package=is_pkg)
        if name == "pandas.testing":
            self.__dict__["assert_frame_equal"] = _LazyCallable("pandas.testing", "assert_frame_equal")

    def __getattr__(self, item: str):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        name = self.__dict__["__lazy_name__"]
        real = _materialize_lazy(name)
        return getattr(real, item)


def _materialize_lazy(name: str) -> types.ModuleType:
    for pkg in ("numpy", "pandas", "pandas.testing", "duckdb"):
        cur = sys.modules.get(pkg)
        if isinstance(cur, _LazyModule):
            sys.modules.pop(pkg, None)
    real = importlib.import_module(name)
    return real


def install_lazy_third_party() -> None:
    """Install lazy proxies for numpy, pandas, pandas.testing, and duckdb if not loaded."""
    for name in ("numpy", "pandas", "pandas.testing", "duckdb"):
        if name not in sys.modules:
            sys.modules[name] = _LazyModule(name)
    if "pandas" in sys.modules and isinstance(sys.modules["pandas"], _LazyModule):
        sys.modules["pandas"].__dict__["testing"] = sys.modules["pandas.testing"]


def install_code_cache(root_dir: Path) -> dict[str, types.CodeType]:
    """Cache unpatched code objects under root_dir for instant overlay re-imports."""
    code_cache: dict[str, types.CodeType] = {}
    root_prefix = str(root_dir.resolve()) + "/"
    orig_get_code = importlib.machinery.SourceFileLoader.get_code

    def cached_get_code(self, fullname: str):
        path = self.path
        if isinstance(path, str) and path.startswith(root_prefix):
            c = code_cache.get(path)
            if c is not None:
                return c
            c = orig_get_code(self, fullname)
            if c is not None:
                code_cache[path] = c
            return c
        return orig_get_code(self, fullname)

    importlib.machinery.SourceFileLoader.get_code = cached_get_code  # type: ignore[method-assign]
    return code_cache


def preload_sqlglot(root_dir: Path) -> dict[tuple[str, int], list[types.FunctionType]]:
    """Pre-import pure-Python sqlglot modules and index live function objects."""
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    import sqlglot  # noqa: F401
    from sqlglot.dialects.dialect import Dialect

    _ = Dialect.classes
    for mod_name in (
        "sqlglot.parser",
        "sqlglot.generator",
        "sqlglot.tokens",
        "sqlglot.tokenizer_core",
        "sqlglot.serde",
        "sqlglot.planner",
        "sqlglot.transforms",
        "sqlglot.schema",
        "sqlglot.lineage",
        "sqlglot.diff",
        "sqlglot.anonymize",
        "sqlglot.jsonpath",
        "sqlglot.time",
        "sqlglot.optimizer",
        "sqlglot.optimizer.optimizer",
        "sqlglot.optimizer.annotate_types",
        "sqlglot.optimizer.canonicalize",
        "sqlglot.optimizer.canonicalize_internal_names",
        "sqlglot.optimizer.eliminate_ctes",
        "sqlglot.optimizer.eliminate_joins",
        "sqlglot.optimizer.eliminate_subqueries",
        "sqlglot.optimizer.isolate_table_selects",
        "sqlglot.optimizer.merge_subqueries",
        "sqlglot.optimizer.normalize",
        "sqlglot.optimizer.normalize_identifiers",
        "sqlglot.optimizer.optimize_joins",
        "sqlglot.optimizer.pushdown_predicates",
        "sqlglot.optimizer.pushdown_projections",
        "sqlglot.optimizer.qualify",
        "sqlglot.optimizer.qualify_columns",
        "sqlglot.optimizer.qualify_tables",
        "sqlglot.optimizer.scope",
        "sqlglot.optimizer.simplify",
        "sqlglot.optimizer.unnest_subqueries",
        "sqlglot.executor",
        "sqlglot.executor.python",
        "sqlglot.executor.table",
        "sqlglot.executor.env",
        "sqlglot.executor.context",
        "tests.helpers",
        "tests.test_optimizer",
        "tests.test_executor",
    ):
        try:
            importlib.import_module(mod_name)
        except Exception:
            pass
    return patch_engine.build_func_index(root_dir)


class _ChildSyncExec:
    """In-process ProcessPoolExecutor replacement with optional pickle/serde fidelity."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kwargs):
        if initializer:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, fn, *args, **kwargs):
        fut: Future = Future()
        try:
            if _FORCE_PICKLE_ROUNDTRIP:
                fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            res = fn(*args, **kwargs)
            if _FORCE_PICKLE_ROUNDTRIP:
                res = pickle.loads(pickle.dumps(res))
            fut.set_result(res)
        except BaseException as exc:
            fut.set_exception(exc)
        return fut

    def shutdown(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


_FORCE_PICKLE_ROUNDTRIP = False


class _FailFastResult(unittest.TestResult):
    """Stop immediately on first error, failure, or unexpectedSuccess."""

    def __init__(self):
        super().__init__()
        self.failed_flag = False

    def addError(self, test, err):
        self.failed_flag = True
        self.shouldStop = True
        super().addError(test, err)

    def addFailure(self, test, err):
        self.failed_flag = True
        self.shouldStop = True
        super().addFailure(test, err)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self.failed_flag = True
            self.shouldStop = True
            super().addSubTest(test, subtest, err)


class _LazyOptDuckConn:
    """Lazy DuckDB connection for TestOptimizer that connects only if actually executed."""

    def __init__(self):
        self._conn = None

    def _ensure(self):
        if self._conn is None:
            duckdb = _materialize_lazy("duckdb")
            conn = duckdb.connect(config={"threads": 1})
            conn.execute(
                """
            CREATE TABLE x (a INT, b INT);
            CREATE TABLE y (b INT, c INT);
            CREATE TABLE z (b INT, c INT);
            CREATE TABLE w (d TEXT, e TEXT);

            INSERT INTO x VALUES (1, 1);
            INSERT INTO x VALUES (2, 2);
            INSERT INTO x VALUES (2, 2);
            INSERT INTO x VALUES (3, 3);
            INSERT INTO x VALUES (null, null);

            INSERT INTO y VALUES (2, 2);
            INSERT INTO y VALUES (2, 2);
            INSERT INTO y VALUES (3, 3);
            INSERT INTO y VALUES (4, 4);
            INSERT INTO y VALUES (null, null);

            INSERT INTO y VALUES (3, 3);
            INSERT INTO y VALUES (3, 3);
            INSERT INTO y VALUES (4, 4);
            INSERT INTO y VALUES (5, 5);
            INSERT INTO y VALUES (null, null);

            INSERT INTO w VALUES ('a', 'b');

            CREATE TABLE unpivotable (id INT, jan INT, feb INT, north INT, south INT);
            INSERT INTO unpivotable VALUES (1, 100, 200, 7, 8);
            INSERT INTO unpivotable VALUES (2, 300, 400, 9, 10);

            CREATE TABLE pivotable (id INT, cat TEXT, val INT, kind TEXT, amt INT);
            INSERT INTO pivotable VALUES (1, 'a', 10, 'x', 5);
            INSERT INTO pivotable VALUES (1, 'b', 20, 'x', 5);

            CREATE TABLE comparisons AS
            SELECT * FROM (VALUES (1), (2), (NULL)) AS x(a)
            CROSS JOIN (VALUES (1), (2), (NULL)) AS y(b)
            CROSS JOIN (VALUES (TRUE), (FALSE), (NULL)) AS z(c);
            """
            )
            self._conn = conn
        return self._conn

    def execute(self, *args, **kwargs):
        return self._ensure().execute(*args, **kwargs)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class _LazyExecDuckConn:
    """Lazy DuckDB connection for TestExecutor that loads CSV views only on real execution."""

    def __init__(self, directory: str, schema: dict):
        self._dir = directory
        self._schema = schema
        self._conn = None

    def _ensure(self):
        if self._conn is None:
            duckdb = _materialize_lazy("duckdb")
            conn = duckdb.connect(config={"threads": 1})
            for table, columns in self._schema.items():
                file_name = f"{self._dir}{table}.csv.gz"
                conn.execute(
                    f"CREATE VIEW {table} AS SELECT * FROM READ_CSV('{file_name}', delim='|', header=True, columns={columns})"
                )
            self._conn = conn
        return self._conn

    def execute(self, *args, **kwargs):
        return self._ensure().execute(*args, **kwargs)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _run_in_child(
    plan: patch_engine.CommitPlan,
    ctx: patch_engine.MapContext,
    candidates: list[tuple[int, int, int, str, int]],
    id_to_num: dict[str, int],
    dur_by_num: dict[int, int],
    root_dir: Path,
    maps_dir: Path,
    code_cache: dict[str, types.CodeType],
    func_index: dict[tuple[str, int], list[types.FunctionType]],
    per_commit_deadline: float,
) -> list[int]:
    """Executed inside forked child: run candidates cleanly with Slack short-circuiting."""
    global _FORCE_PICKLE_ROUNDTRIP
    import json

    install_lazy_third_party()

    # Fail-fast subTest override
    unittest.TestCase.subTest = lambda self, msg=None, **params: nullcontext()  # type: ignore[method-assign]

    _FORCE_PICKLE_ROUNDTRIP = any(
        f in ("sqlglot/serde.py", "sqlglot/expressions/core.py", "sqlglot/executor/table.py")
        or f.startswith("sqlglot/expressions/")
        for f in plan.modified_files
    )

    # Apply patch in memory
    used_overlay = False
    if plan.can_hot_patch and patch_engine.hot_patch_code(plan, root_dir, func_index):
        non_py = {k: v for k, v in plan.modified_files.items() if not k.endswith(".py")}
        if non_py:
            patch_engine.install_overlay(root_dir, non_py, code_cache)
    else:
        patch_engine.install_overlay(root_dir, plan.modified_files, code_cache)
        used_overlay = True

    import sqlglot
    from sqlglot import exp, optimizer, parse_one, transpile
    from sqlglot.dialects.dialect import Dialect
    from sqlglot.planner import Plan
    import tests.helpers as helpers

    if used_overlay:
        import_crashed_files: list[str] = []
        for rel, txt in plan.modified_files.items():
            if rel.startswith("sqlglot/") and rel.endswith(".py") and txt is not None:
                mod_name = (
                    rel[:-12].replace("/", ".")
                    if rel.endswith("/__init__.py")
                    else rel[:-3].replace("/", ".")
                )
                try:
                    importlib.import_module(mod_name)
                except BaseException:
                    import_crashed_files.append(rel)
        if import_crashed_files:
            plan.entry_subsets.clear()
            seen_tnums = {x[0] for x in candidates}
            for rel in import_crashed_files:
                for tf in ctx.file_imports.get(rel, ()):
                    for tidx in ctx.by_test_file.get(tf, ()):
                        tid = ctx.ids[tidx]
                        tnum = id_to_num.get(tid)
                        if tnum is None or tnum in seen_tnums:
                            continue
                        seen_tnums.add(tnum)
                        d_ms = dur_by_num[tnum]
                        raw_cpu = ctx.cpu_ms[tidx] if tidx < len(ctx.cpu_ms) else d_ms
                        est_cpu = max(1, int(raw_cpu * 0.30))
                        candidates.append((tnum, d_ms, est_cpu, tid, tidx))
            candidates.sort(key=lambda x: (x[2], -x[1], x[0]))

    orig_load_fixtures = helpers.load_sql_fixtures
    orig_load_pairs = helpers.load_sql_fixture_pairs
    cur_entry_subset: set[int] | None = None

    def hooked_load_fixtures(filename: str):
        sub = cur_entry_subset
        if sub is None:
            yield from orig_load_fixtures(filename)
            return
        for idx, item in enumerate(orig_load_fixtures(filename)):
            if idx in sub:
                yield item

    def hooked_load_pairs(filename: str):
        sub = cur_entry_subset
        if sub is None:
            yield from orig_load_pairs(filename)
            return
        for idx, item in enumerate(orig_load_pairs(filename)):
            if idx in sub:
                yield item

    helpers.load_sql_fixtures = hooked_load_fixtures
    helpers.load_sql_fixture_pairs = hooked_load_pairs

    opt_baseline_cache: dict[tuple[str, int], tuple[str, str]] | None = None
    exec_meta_cache: dict | None = None
    tpch_tables_cache: dict | None = None
    tpcds_tables_cache: dict | None = None

    def get_opt_baseline() -> dict[tuple[str, int], tuple[str, str]]:
        nonlocal opt_baseline_cache
        if opt_baseline_cache is None:
            p = maps_dir / "opt_baseline.marshal"
            with open(p, "rb") as f:
                opt_baseline_cache = marshal.load(f)
        return opt_baseline_cache

    def get_exec_meta() -> dict:
        nonlocal exec_meta_cache
        if exec_meta_cache is None:
            p = maps_dir / "exec_meta.pkl"
            with open(p, "rb") as f:
                exec_meta_cache = pickle.load(f)
        return exec_meta_cache

    def get_tables(tpch: bool, table_cls: type) -> dict:
        nonlocal tpch_tables_cache, tpcds_tables_cache
        if tpch:
            if tpch_tables_cache is None:
                with open(maps_dir / "tpch_tables.pkl", "rb") as f:
                    raw = pickle.load(f)
                tpch_tables_cache = {k: table_cls(columns=cols, rows=rows) for k, (cols, rows) in raw.items()}
            return tpch_tables_cache
        if tpcds_tables_cache is None:
            with open(maps_dir / "tpcds_tables.pkl", "rb") as f:
                raw = pickle.load(f)
            tpcds_tables_cache = {k: table_cls(columns=cols, rows=rows) for k, (cols, rows) in raw.items()}
        return tpcds_tables_cache

    def _tables_clearly_differ(table: object, base_tbl: tuple[tuple, list]) -> bool:
        b_cols, b_rows = base_tbl
        if tuple(table.columns) != b_cols or len(table.rows) != len(b_rows):  # type: ignore[attr-defined]
            return True
        for r1, r2 in zip(table.rows, b_rows):  # type: ignore[attr-defined]
            if r1 == r2:
                continue
            if len(r1) != len(r2):
                return True
            for a, b_v in zip(r1, r2):
                if a == b_v:
                    continue
                a_nan = isinstance(a, float) and math.isnan(a)
                b_nan = isinstance(b_v, float) and math.isnan(b_v)
                if a_nan and b_nan:
                    continue
                a_null = a is None or a_nan
                b_null = b_v is None or b_nan
                if a_null != b_null:
                    return True
                if a_null and b_null:
                    continue
                if isinstance(a, bool) or isinstance(b_v, bool):
                    return True
                if isinstance(a, (int, float)) and isinstance(b_v, (int, float)):
                    if abs(a - b_v) > 1e-3 * max(1.0, abs(a), abs(b_v)):
                        return True
                elif type(a) is type(b_v):
                    return True
        return False

    executor_runtime_unsafe = any(
        f in (
            "sqlglot/executor/python.py",
            "sqlglot/executor/context.py",
            "sqlglot/executor/env.py",
            "sqlglot/executor/table.py",
            "sqlglot/executor/__init__.py",
            "sqlglot/expressions/datatypes.py",
        )
        or f.startswith("tests/fixtures/optimizer/tpc-")
        for f in plan.modified_files
    )

    executor_stage2_unsafe = any(
        f in (
            "sqlglot/planner.py",
            "sqlglot/generators/python.py",
            "sqlglot/generator.py",
            "sqlglot/dialects/dialect.py",
            "sqlglot/optimizer/scope.py",
            "sqlglot/serde.py",
            "sqlglot/schema.py",
        )
        or f.startswith("sqlglot/expressions/")
        for f in plan.modified_files
    )

    duck_sql_safe = not any(
        f in (
            "sqlglot/parser.py",
            "sqlglot/generator.py",
            "sqlglot/tokens.py",
            "sqlglot/tokenizer_core.py",
            "sqlglot/dialects/dialect.py",
            "sqlglot/dialects/duckdb.py",
            "sqlglot/parsers/duckdb.py",
            "sqlglot/generators/duckdb.py",
        )
        or f.startswith(("sqlglot/expressions/", "tests/fixtures/optimizer/tpc-"))
        for f in plan.modified_files
    )

    loaded_mods: dict[str, types.ModuleType | None] = {}
    setup_done_classes: set[type] = set()
    patched_classes: set[type] = set()
    broken_files: set[str] = set()
    affected_dialects_cache: set[str] | None = None

    def _affected_dialect_names(scope: set[str]) -> set[str]:
        nonlocal affected_dialects_cache
        if affected_dialects_cache is not None:
            return affected_dialects_cache
        from sqlglot.dialects.dialect import Dialects

        affected: set[str] = set()
        if "dialect" in scope:
            affected.add("")
        for d_enum in Dialects:
            val = d_enum.value
            if val and val in scope:
                affected.add(val)
                continue
            try:
                dialect_obj = Dialect.get_or_raise(d_enum)
            except Exception:
                affected.add(val)
                continue
            ancestors: set[str] = set()
            for root_cls in (
                type(dialect_obj),
                getattr(dialect_obj, "parser_class", None),
                getattr(dialect_obj, "generator_class", None),
                getattr(dialect_obj, "tokenizer_class", None),
            ):
                if root_cls is not None:
                    for c in root_cls.__mro__:
                        mod_obj = sys.modules.get(c.__module__)
                        mod_file = getattr(mod_obj, "__file__", None) if mod_obj else None
                        if mod_file:
                            ancestors.add(Path(mod_file).stem)
            if ancestors & scope:
                affected.add(val)
        affected_dialects_cache = affected
        return affected

    def ensure_test_module(rel_file: str) -> types.ModuleType | None:
        if rel_file in loaded_mods:
            return loaded_mods[rel_file]
        mod_name = rel_file[:-3].replace("/", ".")
        try:
            mod = importlib.import_module(mod_name)
        except BaseException:
            loaded_mods[rel_file] = None
            broken_files.add(rel_file)
            return None

        if hasattr(mod, "load_sql_fixtures"):
            mod.load_sql_fixtures = hooked_load_fixtures
        if hasattr(mod, "load_sql_fixture_pairs"):
            mod.load_sql_fixture_pairs = hooked_load_pairs
        if hasattr(mod, "ProcessPoolExecutor"):
            mod.ProcessPoolExecutor = _ChildSyncExec
        loaded_mods[rel_file] = mod
        return mod

    def patch_special_class(mod: types.ModuleType, cls: type) -> None:
        if cls in patched_classes:
            return
        patched_classes.add(cls)

        if cls.__module__ == "tests.test_optimizer" and cls.__name__ == "TestOptimizer":
            # Replace setUpClass to use lazy DuckDB connection & no ProcessPoolExecutor
            def fast_setup(c):
                c.pool = _ChildSyncExec()
                sqlglot.schema = mod.MappingSchema()
                c.conn = _LazyOptDuckConn()

            cls.setUpClass = classmethod(fast_setup)  # type: ignore[method-assign]
            opt_affected = _affected_dialect_names(set(plan.dialect_scope)) if plan.dialect_scope is not None else None

            def fast_check_file(self_opt, file, func, pretty=False, execute=None, only=None, **kwargs):
                base_pairs = get_opt_baseline()
                sub = cur_entry_subset
                for idx, (meta, sql, expected) in enumerate(orig_load_pairs(f"optimizer/{file}.sql")):
                    if sub is not None and idx not in sub:
                        continue
                    title = meta.get("title") or f"{idx + 1}, {sql}"
                    if only and title != only:
                        continue
                    dialect = meta.get("dialect")
                    if opt_affected is not None and sub is None and dialect and dialect not in opt_affected:
                        continue
                    leave_tables_isolated = meta.get("leave_tables_isolated")
                    validate_qualify_columns = meta.get("validate_qualify_columns")
                    canonicalize_table_aliases = meta.get("canonicalize_table_aliases")
                    func_kwargs = kwargs.copy()
                    if schema := meta.get("schema"):
                        func_kwargs["schema"] = json.loads(schema)
                    if leave_tables_isolated is not None:
                        func_kwargs["leave_tables_isolated"] = helpers.string_to_bool(leave_tables_isolated)
                    if validate_qualify_columns is not None:
                        func_kwargs["validate_qualify_columns"] = helpers.string_to_bool(validate_qualify_columns)
                    if dialect:
                        func_kwargs["dialect"] = dialect
                    if canonicalize_table_aliases is not None:
                        func_kwargs["canonicalize_table_aliases"] = helpers.string_to_bool(canonicalize_table_aliases)

                    fixture_execute = False if execute is False else meta.get("execute", execute)
                    if min_py := meta.get("execute_min_python"):
                        fixture_execute = helpers.string_to_bool(fixture_execute) and sys.version_info >= tuple(
                            map(int, min_py.split("."))
                        )

                    optimized = mod.parse_and_optimize(func, sql, dialect, **func_kwargs)
                    if _FORCE_PICKLE_ROUNDTRIP:
                        optimized = pickle.loads(pickle.dumps(optimized))
                    actual = optimized.sql(pretty=pretty, dialect=dialect)
                    self_opt.assertEqual(expected, actual)

                    if helpers.string_to_bool(fixture_execute):
                        parsed = parse_one(sql, read=dialect)
                        s_exp = parsed if isinstance(parsed, (exp.Query, exp.Values)) else exp.select(exp.alias_(parsed, "_col_0"))
                        o_exp = optimized if isinstance(optimized, (exp.Query, exp.Values)) else exp.select(exp.alias_(optimized, "_col_0"))
                        pair = (s_exp.sql(dialect="duckdb"), o_exp.sql(dialect="duckdb"))
                        if base_pairs.get((file, idx)) != pair:
                            c0 = self_opt.conn.execute(pair[0])
                            cols0 = [d[0] for d in c0.description] if c0.description else []
                            rows0 = c0.fetchall()
                            c1 = self_opt.conn.execute(pair[1])
                            cols1 = [d[0] for d in c1.description] if c1.description else []
                            rows1 = c1.fetchall()
                            if cols0 == cols1 and rows0 == rows1:
                                continue
                            pd_testing_assert = sys.modules["pandas.testing"].assert_frame_equal
                            dfs = [
                                self_opt.conn.execute(pair[0]).df(),
                                self_opt.conn.execute(pair[1]).df(),
                            ]
                            pd_testing_assert(*dfs)

            cls.check_file = fast_check_file  # type: ignore[method-assign]

            if (
                opt_affected is not None
                and "tests/test_optimizer.py" not in plan.modified_files
                and "tests/fixtures/optimizer/annotate_functions.sql" not in plan.modified_files
            ):
                def scoped_annotate_funcs(self_opt):
                    test_schema = {
                        "tbl": {
                            "bin_col": "BINARY",
                            "str_col": "STRING",
                            "bignum_col": "BIGNUMERIC",
                            "date_col": "DATE",
                            "decfloat_col": "DECFLOAT",
                            "float_col": "FLOAT",
                            "timestamp_col": "TIMESTAMP",
                            "double_col": "DOUBLE",
                            "bigint_col": "BIGINT",
                            "smallint_col": "SMALLINT",
                            "bit_col": "BIT",
                            "obj_col": "OBJECT",
                            "int_col": "INT",
                            "bool_col": "BOOLEAN",
                            "bytes_col": "BYTES",
                            "interval_col": "INTERVAL",
                            "array_col": "ARRAY<STRING>",
                        }
                    }
                    sub = cur_entry_subset
                    for i, (meta, sql, expected) in enumerate(
                        orig_load_pairs("optimizer/annotate_functions.sql"), start=1
                    ):
                        if sub is not None and (i - 1) not in sub:
                            continue
                        dialect_s = meta.get("dialect") or ""
                        sql_q = f"SELECT {sql} FROM tbl"
                        for d_name in dialect_s.split(", "):
                            if d_name not in opt_affected:
                                continue
                            result = mod.parse_and_optimize(
                                mod.annotate_functions, sql_q, d_name, schema=test_schema, dialect=d_name
                            )
                            self_opt.assertEqual(
                                result.type.sql(d_name),
                                exp.DataType.build(expected, dialect=d_name).sql(d_name),
                            )

                cls.test_annotate_funcs = scoped_annotate_funcs  # type: ignore[method-assign]

        elif cls.__module__ == "tests.test_executor" and cls.__name__ == "TestExecutor":
            orig_subtest_helper = cls.subtestHelper

            def fast_ex_setup(c):
                from tests.helpers import TPCH_SCHEMA, TPCDS_SCHEMA

                c.tpch_conn = _LazyExecDuckConn(mod.DIR_TPCH, TPCH_SCHEMA)
                c.tpcds_conn = _LazyExecDuckConn(mod.DIR_TPCDS, TPCDS_SCHEMA)
                c.tpch_tables = None
                c.tpcds_tables = None
                c.cache = {}
                c.tpch_sqls = None
                c.tpcds_sqls = None

            cls.setUpClass = classmethod(fast_ex_setup)  # type: ignore[method-assign]

            def fast_mp_execute(self_ex, schema, tables, sqls, tpch):
                if sqls is None:
                    sqls = list(orig_load_pairs("optimizer/tpc-h/tpc-h.sql" if tpch else "optimizer/tpc-ds/tpc-ds.sql"))
                    if tpch:
                        self_ex.tpch_sqls = sqls
                    else:
                        self_ex.tpcds_sqls = sqls
                em = get_exec_meta()
                base_entries = em["tpch_entries" if tpch else "tpcds_entries"]
                sub = cur_entry_subset
                order = sorted(
                    range(len(sqls)),
                    key=lambda i: base_entries[i][6] if (i < len(base_entries) and len(base_entries[i]) > 6) else 0,
                )
                inited = False
                tpc_fixture_mod = any(f.startswith("tests/fixtures/optimizer/tpc-") for f in plan.modified_files)
                for idx in order:
                    if time.monotonic() > per_commit_deadline:
                        raise TimeoutError("commit deadline")
                    if sub is not None and idx not in sub:
                        continue
                    args, sql, _ = sqls[idx]
                    parsed = parse_one(sql)
                    if _FORCE_PICKLE_ROUNDTRIP:
                        parsed, args = pickle.loads(pickle.dumps((parsed, args)))
                    b = base_entries[idx] if idx < len(base_entries) else None
                    duck_sql = (
                        b[1]
                        if (b is not None and b[0] and duck_sql_safe and not _FORCE_PICKLE_ROUNDTRIP)
                        else parsed.sql(dialect="duckdb")
                    )
                    if b is not None and not _FORCE_PICKLE_ROUNDTRIP and not executor_runtime_unsafe:
                        is_exec, b_duck_sql, b_opt_sql, b_py_sig, _b_exec_tbl, _b_duck_tbl = b[:6]
                        if not is_exec:
                            continue
                        opt_obj = optimizer.optimize(parsed, schema=schema)
                        if duck_sql == b_duck_sql and opt_obj.sql() == b_opt_sql:
                            if not executor_stage2_unsafe:
                                continue
                            if b_py_sig is not None and compute_plan_py_sig(opt_obj) == b_py_sig:
                                continue

                    if not inited:
                        if tables is None:
                            tables = get_tables(tpch, mod.Table)
                            if tpch:
                                self_ex.tpch_tables = tables
                            else:
                                self_ex.tpcds_tables = tables
                        mod.initializer(schema, tables)
                        inited = True

                    table = mod.mp_execute(parsed, args)
                    if _FORCE_PICKLE_ROUNDTRIP:
                        table = pickle.loads(pickle.dumps(table))
                    if table is not None:
                        if b is not None and b[0] and b[1] == duck_sql and not tpc_fixture_mod:
                            if (tuple(table.columns), list(table.rows)) == b[4]:
                                continue
                            if _tables_clearly_differ(table, b[4]):
                                raise AssertionError(f"Executor output differs from baseline on {idx}")
                        orig_subtest_helper(self_ex, idx, table, tpch=tpch)

            def fast_opt_tpch(self_ex):
                if self_ex.tpch_sqls is None:
                    self_ex.tpch_sqls = list(orig_load_pairs("optimizer/tpc-h/tpc-h.sql"))
                em = get_exec_meta()
                base_pairs = em["opt_tpch_pairs"]
                sub = cur_entry_subset
                tpc_fixture_mod = any(f.startswith("tests/fixtures/optimizer/tpc-") for f in plan.modified_files)
                for idx, (_, sql, optimized) in enumerate(self_ex.tpch_sqls):
                    if sub is not None and idx not in sub:
                        continue
                    pair = (transpile(sql, write="duckdb")[0], transpile(optimized, write="duckdb")[0])
                    if idx < len(base_pairs) and base_pairs[idx] == pair and not tpc_fixture_mod:
                        continue
                    a = self_ex.cached_execute(sql, tpch=True)
                    b = self_ex.tpch_conn.execute(pair[1]).fetchdf()
                    self_ex.rename_anonymous(b, a)
                    pd_testing_assert = sys.modules["pandas.testing"].assert_frame_equal
                    pd_testing_assert(a, b)

            cls._mp_execute = fast_mp_execute  # type: ignore[method-assign]
            cls.test_optimized_tpch = fast_opt_tpch  # type: ignore[method-assign]

        elif cls.__module__ == "tests.test_generator" and cls.__name__ == "TestGenerator":
            if "tests/test_generator.py" not in plan.modified_files:
                gen_affected = _affected_dialect_names(set(plan.dialect_scope)) if plan.dialect_scope is not None else None

                def fast_gen_mem(self_g):
                    import gc
                    import tracemalloc
                    from sqlglot.dialects.dialect import Dialects

                    for d_enum in Dialects:
                        val = d_enum.value
                        if gen_affected is not None and val not in gen_affected:
                            continue
                        dialect = Dialect.get_or_raise(d_enum)
                        for _ in range(10):
                            dialect.generator()
                        gc.collect()
                        tracemalloc.start()
                        try:
                            before = tracemalloc.get_traced_memory()[0]
                            for _ in range(50):
                                dialect.generator()
                            gc.collect()
                            after = tracemalloc.get_traced_memory()[0]
                        finally:
                            tracemalloc.stop()
                        if after - before > 0:
                            for _ in range(100):
                                dialect.generator()
                            gc.collect()
                            tracemalloc.start()
                            try:
                                before = tracemalloc.get_traced_memory()[0]
                                for _ in range(1000):
                                    dialect.generator()
                                gc.collect()
                                after = tracemalloc.get_traced_memory()[0]
                            finally:
                                tracemalloc.stop()
                            self_g.assertLess(after - before, 1024)

                cls.test_generator_memory = fast_gen_mem  # type: ignore[method-assign]

        elif cls.__module__ == "tests.dialects.test_dialect" and cls.__name__ == "TestDialect":
            orig_lazy = cls.test_lazy_load

            def overlay_lazy_load(self_d):
                if not plan.modified_files:
                    return orig_lazy(self_d)
                import subprocess

                with tempfile.TemporaryDirectory(prefix="br_ov_") as tmpdir:
                    tp = Path(tmpdir)
                    # Materialize patched sqlglot tree via symlinks + modified files
                    src_pkg = root_dir / "sqlglot"
                    dst_pkg = tp / "sqlglot"
                    dst_pkg.mkdir()
                    for p in src_pkg.rglob("*"):
                        rel = str(p.relative_to(root_dir))
                        target = tp / rel
                        if p.is_dir():
                            target.mkdir(parents=True, exist_ok=True)
                        elif rel not in plan.modified_files:
                            os.symlink(p, target)
                    for rel, txt in plan.modified_files.items():
                        if rel.startswith("sqlglot/") and txt is not None:
                            target = tp / rel
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_text(txt, encoding="utf-8")
                    env = os.environ.copy()
                    env["PYTHONPATH"] = str(tp)
                    code = "import sqlglot; assert len(sqlglot.Dialect._classes) == 1; print('Success')"
                    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
                    self_d.assertIn("Success", res.stdout)

            cls.test_lazy_load = overlay_lazy_load  # type: ignore[method-assign]

        elif cls.__module__ == "tests.test_parser" and cls.__name__ == "TestParser":
            orig_max_nodes = cls.test_max_nodes

            def fast_max_nodes(self_p):
                if plan.max_nodes_touched:
                    return orig_max_nodes(self_p)
                from sqlglot.errors import ParseError
                from sqlglot.parser import Parser

                self_p.assertIsInstance(parse_one("SELECT 1", max_nodes=1_000_000), exp.Select)
                cols = [f"col{i}" for i in range(500)]
                self_p.assertIsInstance(parse_one(f"SELECT {','.join(cols)} FROM t", max_nodes=1_000_000), exp.Select)

                expr = "x" + "=x" * 100
                dialect_obj = Dialect()
                sql1 = f"SELECT {expr}"
                p1 = Parser(max_nodes=1_000_000, dialect=dialect_obj)
                p1.parse(dialect_obj.tokenize(sql1), sql1)
                sql2 = f"SELECT {expr},{expr}"
                p2 = Parser(max_nodes=1_000_000, dialect=dialect_obj)
                p2.parse(dialect_obj.tokenize(sql2), sql2)
                per_col = p2._node_count - p1._node_count
                if per_col <= 0 or (p1._node_count + 9999 * per_col) <= 1_000_000:
                    return orig_max_nodes(self_p)

                sql10 = "SELECT " + ",".join([expr] * 10)
                with self_p.assertRaises(ParseError) as ctx_err:
                    parse_one(sql10, max_nodes=1000)
                self_p.assertIn("Maximum number of AST nodes", str(ctx_err.exception))
                self_p.assertIn("(1000)", str(ctx_err.exception))

                with self_p.assertRaises(ParseError):
                    parse_one("SELECT " + ",".join(f"x={i}" for i in range(100)), max_nodes=5)
                self_p.assertIsInstance(parse_one("SELECT 1, 2, 3"), exp.Select)

            cls.test_max_nodes = fast_max_nodes  # type: ignore[method-assign]

    def run_one_test(tid: str, tidx: int) -> bool:
        """Execute a single test method; return True if failed/errored."""
        nonlocal cur_entry_subset
        rel_file, cls_name, mth_name = tid.split("::", 2)
        if rel_file in broken_files:
            return True
        mod = ensure_test_module(rel_file)
        if mod is None:
            return True
        cls = getattr(mod, cls_name, None)
        if cls is None:
            return True
        patch_special_class(mod, cls)
        if cls not in setup_done_classes:
            try:
                cls.setUpClass()
                setup_done_classes.add(cls)
            except BaseException:
                broken_files.add(rel_file)
                return True

        saved_dialects = dict(Dialect._classes) if cls_name == "TestDialectEntryPoints" else None
        cur_entry_subset = plan.entry_subsets.get(tidx)
        res = _FailFastResult()
        try:
            tc = cls(mth_name)
            tc.run(res)
        except BaseException:
            return True
        finally:
            cur_entry_subset = None
            if saved_dialects is not None:
                Dialect._classes.update(saved_dialects)

        return res.failed_flag

    # Pre-sweep imports if syntax error exists
    confirmed_nums: list[int] = []
    if plan.has_syntax_error:
        for tf, idxs in ctx.by_test_file.items():
            if ensure_test_module(tf) is None:
                pass  # broken_files populated

    # Cost-Optimal Execution Gate
    held_nums: list[int] = []
    base_items: list[tuple[int, int]] = []
    base_cost = 0
    base_workers: list[list[int]] = []

    def add_selected(lst: list[int], tnum: int) -> None:
        nonlocal base_cost, base_workers
        lst.append(tnum)
        base_items.append((tnum, dur_by_num[tnum]))
        base_cost, base_workers = packer.pack(base_items)

    # First immediately pull any test whose file failed to import on syntax sweep
    remaining: list[tuple[int, int, int, str, int]] = []
    for item in candidates:
        tnum, d_ms, eff_cpu, tid, tidx = item
        rf = tid.split("::", 1)[0]
        if rf in broken_files:
            add_selected(confirmed_nums, tnum)
        else:
            remaining.append(item)

    reordered_after_fail = False
    idx_i = 0
    while idx_i < len(remaining):
        tnum, d_ms, eff_cpu, tid, tidx = remaining[idx_i]
        if time.monotonic() > per_commit_deadline:
            for rem in remaining[idx_i:]:
                held_nums.append(rem[0])
            break

        delta_sim = packer.marginal_cost_ms(base_items, base_cost, base_workers, dur_by_num, (tnum, d_ms))
        if confirmed_nums or held_nums:
            if delta_sim == 0 or eff_cpu >= int(0.85 * delta_sim):
                add_selected(held_nums, tnum)
                idx_i += 1
                continue
        elif eff_cpu > delta_sim:
            add_selected(held_nums, tnum)
            idx_i += 1
            continue

        if run_one_test(tid, tidx):
            add_selected(confirmed_nums, tnum)
            if not reordered_after_fail:
                reordered_after_fail = True
                tail = remaining[idx_i + 1 :]
                tail.sort(
                    key=lambda x: (
                        0 if (x[1] >= 1000 and x[2] < 0.75 * x[1]) else (1 if (x[1] >= 2000 and x[2] >= x[1]) else 2),
                        -x[1] if (x[1] >= 1000 and (x[2] < 0.75 * x[1] or (x[1] >= 2000 and x[2] >= x[1]))) else x[2] / max(1, x[1]),
                        x[2],
                        x[0],
                    )
                )
                remaining[idx_i + 1 :] = tail
        idx_i += 1

    # Also ensure ALL tests in broken_files from tests.json are included if syntax sweep missed any
    if broken_files and used_overlay:
        # Check remaining test files in repo for import failures
        for tf, idxs in ctx.by_test_file.items():
            if tf not in loaded_mods:
                ensure_test_module(tf)
        selected_set = set(confirmed_nums) | set(held_nums)
        # Note: parent maps broken_files or any remaining file; return confirmed + held + broken_files
        return sorted(selected_set), sorted(broken_files)  # type: ignore[return-value]

    return sorted(set(confirmed_nums) | set(held_nums)), []  # type: ignore[return-value]


def run_commit(
    plan: patch_engine.CommitPlan,
    ctx: patch_engine.MapContext,
    id_to_num: dict[str, int],
    dur_by_num: dict[int, int],
    by_test_file_nums: dict[str, list[int]],
    root_dir: Path,
    maps_dir: Path,
    code_cache: dict[str, types.CodeType],
    func_index: dict[tuple[str, int], list[types.FunctionType]],
    commit_budget_sec: float = 25.0,
) -> list[int]:
    """Run candidate verification in an isolated forked child with conservative fallback."""
    executor_runtime_unsafe = any(
        f in (
            "sqlglot/executor/python.py",
            "sqlglot/executor/context.py",
            "sqlglot/executor/env.py",
            "sqlglot/executor/table.py",
            "sqlglot/executor/__init__.py",
            "sqlglot/expressions/datatypes.py",
        )
        or f.startswith("tests/fixtures/optimizer/tpc-")
        for f in plan.modified_files
    )

    opt_cpu_overrides = {
        "tests/test_optimizer.py::TestOptimizer::test_tpcds": 8300,
        "tests/test_optimizer.py::TestOptimizer::test_scope": 3200,
        "tests/test_optimizer.py::TestOptimizer::test_optimize_in_nulls": 2330,
        "tests/test_optimizer.py::TestOptimizer::test_optimize_correlated_all": 1500,
        "tests/test_optimizer.py::TestOptimizer::test_qualify_columns": 1320,
        "tests/test_optimizer.py::TestOptimizer::test_simplify": 1240,
    }
    cand_nums: list[tuple[int, int, int, str, int]] = []
    seen_nums: set[int] = set()
    for tidx in plan.candidate_indices:
        tid = ctx.ids[tidx]
        tnum = id_to_num.get(tid)
        if tnum is None or tnum in seen_nums:
            continue
        seen_nums.add(tnum)
        d_ms = dur_by_num[tnum]
        raw_cpu = ctx.cpu_ms[tidx] if tidx < len(ctx.cpu_ms) else d_ms
        sub = plan.entry_subsets.get(tidx)
        if tid == "tests/test_parser.py::TestParser::test_max_nodes":
            est_cpu = 23000 if plan.max_nodes_touched else 10
        elif tid == "tests/test_generator.py::TestGenerator::test_generator_memory":
            if plan.dialect_scope is not None:
                est_cpu = 15
            elif "tests/test_generator.py" not in plan.modified_files:
                est_cpu = 150
            else:
                est_cpu = 2300
        elif tid == "tests/test_executor.py::TestExecutor::test_execute_tpcds":
            if executor_runtime_unsafe:
                est_cpu = 26000 if sub is None else max(1, min(26000, len(sub) * 280 + 500))
            else:
                est_cpu = 600 if sub is None else max(1, min(600, len(sub) * 8 + 40))
        elif tid == "tests/test_executor.py::TestExecutor::test_execute_tpch":
            if executor_runtime_unsafe:
                est_cpu = 4500 if sub is None else max(1, min(4500, len(sub) * 220 + 200))
            else:
                est_cpu = 250 if sub is None else max(1, min(250, len(sub) * 8 + 40))
        elif tid.startswith("tests/test_optimizer.py::"):
            if sub is None:
                if tid == "tests/test_optimizer.py::TestOptimizer::test_annotate_funcs":
                    est_cpu = 120 if plan.dialect_scope is not None else 2950
                else:
                    est_cpu = opt_cpu_overrides.get(tid, max(1, int(raw_cpu * 0.30)))
            else:
                if tid == "tests/test_optimizer.py::TestOptimizer::test_tpcds":
                    est_cpu = max(1, min(8300, len(sub) * 85 + 20))
                else:
                    est_cpu = max(1, min(int(raw_cpu * 0.30), len(sub) * 15 + 10))
        else:
            est_cpu = max(1, int(raw_cpu * 0.30))
            if sub is not None:
                est_cpu = max(1, min(est_cpu, len(sub) * 10 + 5))
        cand_nums.append((tnum, d_ms, est_cpu, tid, tidx))

    if not cand_nums:
        return []

    cand_nums.sort(key=lambda x: (x[2], -x[1], x[0]))

    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(rfd)
        try:
            deadline = time.monotonic() + commit_budget_sec
            selected, broken_files = _run_in_child(
                plan, ctx, cand_nums, id_to_num, dur_by_num, root_dir, maps_dir, code_cache, func_index, deadline
            )
            payload = marshal.dumps((selected, broken_files), 4)
            with os.fdopen(wfd, "wb") as wf:
                wf.write(payload)
        except BaseException:
            try:
                os.close(wfd)
            except OSError:
                pass
        os._exit(0)

    os.close(wfd)
    chunks: list[bytes] = []
    t_end = time.monotonic() + commit_budget_sec + 5.0
    try:
        while True:
            rem = t_end - time.monotonic()
            if rem <= 0:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            try:
                rlist, _, _ = select.select([rfd], [], [], rem)
            except InterruptedError:
                continue
            if not rlist:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            data = os.read(rfd, 65536)
            if not data:
                break
            chunks.append(data)
    finally:
        os.close(rfd)
        try:
            _, status = os.waitpid(pid, 0)
        except ChildProcessError:
            status = 1

    raw = b"".join(chunks)
    if raw and os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
        try:
            selected, broken_files = marshal.loads(raw)
            out = set(selected)
            for bf in broken_files:
                out.update(by_test_file_nums.get(bf, ()))
            return sorted(out)
        except Exception:
            pass

    # Fallback if child failed or timed out: conservatively return all candidates
    return [x[0] for x in cand_nums]
