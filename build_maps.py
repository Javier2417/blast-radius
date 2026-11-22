#!/usr/bin/env python3
"""Build pinned-commit per-test coverage maps and baseline execution caches.

Runs during Docker build on Python 3.13 using sys.monitoring (PEP 669) across
parallel worker processes with SKIP_INTEGRATION=0. Produces two compact binary
files in MAPS_DIR:
  - cov_map.marshal : line coverage, per-entry fixture coverage, file reads,
                      dialect/expression/dispatch-dict dependencies, import closures,
                      pinned sources, and per-test baseline CPU durations.
  - exec_baseline.pkl : DuckDB/optimizer/executor baseline snapshots so runtime
                        checks can verify equivalence without launching subprocesses
                        or rerunning unchanged DuckDB/Pandas frames.
"""

from __future__ import annotations

import argparse
import array
import ast
import builtins
from collections import defaultdict
from concurrent.futures import Future
import functools
import gzip
import hashlib
import importlib
import marshal
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time
import types
import unittest

DISPATCH_DICT_ATTRS = frozenset(
    {
        "FUNCTIONS",
        "NO_PAREN_FUNCTIONS",
        "EXPRESSION_PARSERS",
        "STATEMENT_PARSERS",
        "UNARY_PARSERS",
        "STRING_PARSERS",
        "NUMERIC_PARSERS",
        "PRIMARY_PARSERS",
        "PLACEHOLDER_PARSERS",
        "RANGE_PARSERS",
        "PIPE_SYNTAX_TRANSFORM_PARSERS",
        "PROPERTY_PARSERS",
        "CONSTRAINT_PARSERS",
        "ALTER_PARSERS",
        "ALTER_ALTER_PARSERS",
        "NO_PAREN_FUNCTION_PARSERS",
        "FUNCTION_PARSERS",
        "QUERY_MODIFIER_PARSERS",
        "SET_PARSERS",
        "SHOW_PARSERS",
        "TYPE_LITERAL_PARSERS",
        "ANALYZE_EXPRESSION_PARSERS",
        "TRANSFORMS",
        "TYPE_MAPPING",
        "PROPERTIES_LOCATION",
        "EXPRESSION_METADATA",
        "TIME_MAPPING",
        "FORMAT_MAPPING",
        "NUMERIC_ESCAPES",
        "CREATABLE_KIND_MAPPING",
        "UNESCAPED_SEQUENCES",
        "RESERVED_TOKENS",
    }
)


def _key_repr(k: object) -> str:
    if isinstance(k, str):
        return f"s:{k}"
    if isinstance(k, int):
        if hasattr(k, "name"):
            return f"e:{type(k).__name__}.{k.name}"
        return f"i:{k}"
    if isinstance(k, type):
        return f"t:{k.__name__}"
    if isinstance(k, tuple):
        return "u:(" + ",".join(_key_repr(x) for x in k) + ")"
    return f"r:{k!r}"


def _rel_path(filename: str | None, root_prefix: str) -> str | None:
    if not filename:
        return None
    if filename.startswith(root_prefix):
        return filename[len(root_prefix) :]
    try:
        resolved = os.path.realpath(filename)
    except OSError:
        return None
    if resolved.startswith(root_prefix):
        return resolved[len(root_prefix) :]
    return None


def _discover_tests(root: Path) -> list[tuple[str, str, str]]:
    """Discover all unittest test methods matching pytest's collection semantics."""
    tests_dir = root / "tests"
    loader = unittest.TestLoader()
    discovered: list[tuple[str, str, str]] = []
    for p in sorted(tests_dir.rglob("test_*.py")):
        rel = str(p.relative_to(root))
        mod_name = rel[:-3].replace("/", ".")
        mod = importlib.import_module(mod_name)
        for obj in vars(mod).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, unittest.TestCase)
                and obj.__module__ == mod.__name__
            ):
                for mth in loader.getTestCaseNames(obj):
                    discovered.append((rel, obj.__name__, mth))
    return discovered


def _build_static_file_imports(root: Path) -> dict[str, list[str]]:
    """Map every source/test .py file -> test files transitively importing it."""
    modules: dict[str, str] = {}
    for sub in ("sqlglot", "tests"):
        for p in sorted((root / sub).rglob("*.py")):
            rel = str(p.relative_to(root))
            if p.name == "__init__.py":
                mod = str(Path(rel).parent).replace("/", ".")
            else:
                mod = rel[:-3].replace("/", ".")
            modules[mod] = rel

    graph: dict[str, set[str]] = {}
    for mod, rel in modules.items():
        try:
            tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        except Exception:
            graph[mod] = set()
            continue
        imps: set[str] = set()
        pkg = mod if rel.endswith("/__init__.py") else mod.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(("sqlglot", "tests")):
                        imps.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base_mod = node.module or ""
                if node.level > 0:
                    parts = pkg.split(".") if pkg else []
                    up = node.level - 1
                    prefix = parts[: len(parts) - up] if up <= len(parts) else []
                    full = ".".join([*prefix, base_mod]) if base_mod else ".".join(prefix)
                else:
                    full = base_mod
                if full.startswith(("sqlglot", "tests")):
                    imps.add(full)
                    for alias in node.names:
                        cand = f"{full}.{alias.name}"
                        if cand in modules:
                            imps.add(cand)
        graph[mod] = imps

    # Dialect imports sqlglot.dialects.<name> dynamically via Dialects / _try_load
    dialect_mods = {
        m for m in modules if m.startswith("sqlglot.dialects.") and m != "sqlglot.dialects.dialect"
    }
    if "sqlglot.dialects.dialect" in graph:
        graph["sqlglot.dialects.dialect"].update(dialect_mods)

    reverse: dict[str, set[str]] = defaultdict(set)
    for mod, deps in graph.items():
        for dep in deps:
            if dep in modules:
                reverse[modules[dep]].add(modules[mod])
            else:
                parts = dep.split(".")
                for i in range(len(parts) - 1, 0, -1):
                    p = ".".join(parts[:i])
                    if p in modules:
                        reverse[modules[p]].add(modules[mod])
                        break

    # BFS forward closure on reverse dependency graph
    file_to_test_files: dict[str, list[str]] = {}
    for src_rel in sorted(modules.values()):
        visited: set[str] = {src_rel}
        queue = [src_rel]
        while queue:
            cur = queue.pop()
            for nxt in reverse.get(cur, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        tfs = sorted(f for f in visited if f.startswith("tests/") and Path(f).name.startswith("test_"))
        if tfs:
            file_to_test_files[src_rel] = tfs
    return file_to_test_files


def _build_attr_refs(
    root: Path,
    file_imports: dict[str, list[str]],
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, list[str]], list[str]]:
    """Statically scan sqlglot/ and tests/ for class-attribute dataflow references."""
    metaclass_attrs: set[str] = set()
    parsed_sqlglot: list[tuple[str, ast.Module]] = []

    for p in sorted((root / "sqlglot").rglob("*.py")):
        rel = str(p.relative_to(root))
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        parsed_sqlglot.append((rel, tree))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ("__new__", "__init_subclass__"):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Attribute):
                        metaclass_attrs.add(sub.attr)
                    elif isinstance(sub, ast.Name):
                        metaclass_attrs.add(sub.id)
                    elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        metaclass_attrs.add(sub.value)

    tracked_attrs: set[str] = set()
    dialect_prefixes = ("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")
    for rel, tree in parsed_sqlglot:
        if not rel.startswith(dialect_prefixes):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        for tgt in stmt.targets:
                            if isinstance(tgt, ast.Name):
                                name = tgt.id
                                if not (name.startswith("__") and name.endswith("__")):
                                    tracked_attrs.add(name)
                    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        name = stmt.target.id
                        if not (name.startswith("__") and name.endswith("__")):
                            tracked_attrs.add(name)

    tracked_attrs -= metaclass_attrs

    attr_refs_sqlglot_sets: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for rel, tree in parsed_sqlglot:
        skip_lhs_ids: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        for tgt in stmt.targets:
                            if isinstance(tgt, ast.Name) and tgt.id in tracked_attrs:
                                skip_lhs_ids.add(id(tgt))
                    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        if stmt.target.id in tracked_attrs:
                            skip_lhs_ids.add(id(stmt.target))

        for node in ast.walk(tree):
            if id(node) in skip_lhs_ids:
                continue
            name: str | None = None
            if isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                name = node.value
            if name is not None and name in tracked_attrs:
                ln = getattr(node, "lineno", None)
                if ln is not None:
                    attr_refs_sqlglot_sets[name].add((rel, ln))

    attr_refs_tests_sets: dict[str, set[str]] = defaultdict(set)
    for p in sorted((root / "tests").rglob("*.py")):
        rel = str(p.relative_to(root))
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        is_test_file = p.name.startswith("test_")
        target_tfs = [rel] if is_test_file else file_imports.get(rel, [])
        if not target_tfs:
            continue
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                name = node.value
            if name is not None and name in tracked_attrs:
                attr_refs_tests_sets[name].update(target_tfs)

    attr_refs_sqlglot = {k: sorted(v) for k, v in attr_refs_sqlglot_sets.items() if v}
    attr_refs_tests = {k: sorted(v) for k, v in attr_refs_tests_sets.items() if v}
    return attr_refs_sqlglot, attr_refs_tests, sorted(metaclass_attrs)


def compute_plan_py_sig(opt_obj: object) -> bytes | None:
    """Compute a deterministic Stage 2+3 Plan+PythonGenerator signature for an optimized query."""
    try:
        from sqlglot import exp
        from sqlglot.executor.python import PythonExecutor
        from sqlglot.planner import Plan

        plan = Plan(opt_obj)  # type: ignore[arg-type]
        pe = PythonExecutor()

        def _dep_sort_key(step_obj: object, seen_keys: dict[int, tuple]) -> tuple:
            sid = id(step_obj)
            if sid in seen_keys:
                return seen_keys[sid]
            dep_keys = tuple(sorted(_dep_sort_key(d, seen_keys) for d in getattr(step_obj, "dependencies", ())))
            src = getattr(step_obj, "source", None)
            src_key = src.sql() if isinstance(src, exp.Expr) else repr(src)
            key = (
                type(step_obj).__name__,
                getattr(step_obj, "name", None) or "",
                src_key,
                tuple(p.sql() if isinstance(p, exp.Expr) else repr(p) for p in getattr(step_obj, "projections", ())),
                repr(getattr(step_obj, "limit", None)),
                getattr(step_obj, "offset", 0),
                dep_keys,
            )
            seen_keys[sid] = key
            return key

        def _render_expr(expr: object) -> str | None:
            if expr is None:
                return None
            if not isinstance(expr, exp.Expr):
                return repr(expr)
            return pe.generator.generate(pe._replace_subqueries(expr))

        def _serialize_plan(p: Plan) -> tuple:
            saved_ctes = pe._ctes
            pe._ctes = p.ctes
            try:
                sort_memo: dict[int, tuple] = {}
                node_ids: dict[int, int] = {}
                steps_out: list[tuple] = []

                def visit(st: object) -> int:
                    sid = id(st)
                    if sid in node_ids:
                        return node_ids[sid]
                    deps_sorted = sorted(
                        getattr(st, "dependencies", ()),
                        key=lambda d: _dep_sort_key(d, sort_memo),
                    )
                    dep_ids = tuple(visit(d) for d in deps_sorted)
                    my_id = len(node_ids)
                    node_ids[sid] = my_id

                    cls_name = type(st).__name__
                    src = getattr(st, "source", None)
                    src_repr = (
                        (src.alias_or_name, src.name, src.sql())
                        if isinstance(src, exp.Expr)
                        else repr(src)
                    )
                    joins = getattr(st, "joins", None)
                    joins_sig = None
                    if joins:
                        joins_sig = tuple(
                            (
                                jname,
                                j.get("side"),
                                _render_expr(j.get("condition")),
                                tuple(_render_expr(x) for x in (j.get("join_key") or ())),
                                tuple(_render_expr(x) for x in (j.get("source_key") or ())),
                            )
                            for jname, j in joins.items()
                        )
                    group = getattr(st, "group", None)
                    group_sig = (
                        tuple((k, _render_expr(v)) for k, v in group.items())
                        if group
                        else None
                    )
                    op = getattr(st, "op", None)
                    steps_out.append(
                        (
                            my_id,
                            cls_name,
                            getattr(st, "name", None),
                            repr(getattr(st, "limit", None)),
                            getattr(st, "offset", 0),
                            dep_ids,
                            src_repr,
                            getattr(st, "source_name", None),
                            op.__name__ if isinstance(op, type) else repr(op),
                            getattr(st, "distinct", None),
                            getattr(st, "left", None),
                            getattr(st, "right", None),
                            _render_expr(getattr(st, "condition", None)),
                            tuple(
                                (e.alias_or_name if isinstance(e, exp.Expr) else repr(e), _render_expr(e))
                                for e in getattr(st, "projections", ())
                            ),
                            tuple(_render_expr(e) for e in getattr(st, "aggregations", ())),
                            tuple(
                                (e.alias_or_name if isinstance(e, exp.Expr) else repr(e), _render_expr(e))
                                for e in getattr(st, "operands", ())
                            ),
                            group_sig,
                            tuple(_render_expr(e) for e in (getattr(st, "key", None) or ())),
                            joins_sig,
                        )
                    )
                    return my_id

                root_id = visit(p.root)
                return (p.ctes.sql() if p.ctes is not None else None, root_id, tuple(steps_out))
            finally:
                pe._ctes = saved_ctes

        main_sig = _serialize_plan(plan)
        sub_sigs: list[tuple[str, tuple]] = []
        sq_done = 0
        while sq_done < len(pe._subquery_plans):
            items = list(pe._subquery_plans.items())
            for sq_name, (sq_plan, _) in items[sq_done:]:
                sub_sigs.append((sq_name, _serialize_plan(sq_plan)))
            sq_done = len(items)

        payload = repr((main_sig, tuple(sub_sigs))).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).digest()
    except Exception:
        return None


def _worker_trace(
    worker_id: int,
    root_str: str,
    assigned_files: list[str],
    id_to_idx: dict[str, int],
) -> dict:
    """Trace all tests in assigned_files using sys.monitoring + runtime hooks."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    orig_stdout_fd = os.dup(1)
    os.dup2(devnull, 1)
    os.close(devnull)
    sys.stdout = open(os.devnull, "w", encoding="utf-8")

    def log(msg: str) -> None:
        sys.stderr.write(f"[w{worker_id}] {msg}\n")
        sys.stderr.flush()

    os.environ["SKIP_INTEGRATION"] = "0"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    root = Path(root_str).resolve()
    os.environ["PYTHONPATH"] = str(root)
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    root_prefix = str(root) + "/"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import sqlglot  # noqa: E402
    from sqlglot import exp, optimizer, parse_one, transpile  # noqa: E402
    from sqlglot.dialects.dialect import Dialect, _Dialect  # noqa: E402
    from sqlglot.planner import Plan  # noqa: E402
    from sqlglot.executor import execute  # noqa: E402
    import tests.helpers as helpers  # noqa: E402

    loader = unittest.TestLoader()
    test_result = unittest.TestResult()

    # Worker-local state updated for each running phase/test
    active_cov: set[tuple[str, int]] = set()
    active_entry_cov: dict[tuple[str, int], set[int]] = defaultdict(set)
    active_reads: set[str] = set()
    active_dialects: set[str] = set()
    active_exprs: set[str] = set()
    active_dict_keys: set[tuple[str, str, str, str]] = set()

    collecting = False
    cur_entry: int | None = None  # None => -1 (not inside a subsettable entry loop)

    # 1. File read hooks
    orig_open = builtins.open
    orig_gzip_open = gzip.open
    orig_path_open = Path.open
    orig_path_read_text = Path.read_text
    orig_path_read_bytes = Path.read_bytes

    def _record_file(path_obj: object) -> None:
        if not collecting or not isinstance(path_obj, (str, os.PathLike)):
            return
        p_str = os.fspath(path_obj)
        rel = _rel_path(p_str, root_prefix)
        if rel is not None and not rel.endswith(".pyc"):
            active_reads.add(rel)

    @functools.wraps(orig_open)
    def hooked_open(file, mode="r", *args, **kwargs):
        _record_file(file)
        return orig_open(file, mode, *args, **kwargs)

    @functools.wraps(orig_gzip_open)
    def hooked_gzip_open(filename, mode="rb", *args, **kwargs):
        _record_file(filename)
        return orig_gzip_open(filename, mode, *args, **kwargs)

    def hooked_path_open(self, *args, **kwargs):
        _record_file(self)
        return orig_path_open(self, *args, **kwargs)

    def hooked_path_read_text(self, *args, **kwargs):
        _record_file(self)
        return orig_path_read_text(self, *args, **kwargs)

    def hooked_path_read_bytes(self, *args, **kwargs):
        _record_file(self)
        return orig_path_read_bytes(self, *args, **kwargs)

    builtins.open = hooked_open  # type: ignore[assignment]
    gzip.open = hooked_gzip_open  # type: ignore[assignment]
    Path.open = hooked_path_open  # type: ignore[method-assign]
    Path.read_text = hooked_path_read_text  # type: ignore[method-assign]
    Path.read_bytes = hooked_path_read_bytes  # type: ignore[method-assign]

    # 2. Dialect & Expr hooks
    orig_dialect_get = _Dialect.get.__func__  # type: ignore[attr-defined]
    orig_dialect_getitem = _Dialect.__getitem__.__func__  # type: ignore[attr-defined]
    orig_dialect_init = Dialect.__init__
    orig_expr_init = exp.Expr.__init__

    def _record_dialect_cls(dcls: object) -> None:
        if not collecting or not isinstance(dcls, type) or not issubclass(dcls, Dialect):
            return
        for c in dcls.__mro__:
            rel = _rel_path(getattr(sys.modules.get(c.__module__), "__file__", None), root_prefix)
            if rel and rel.startswith("sqlglot/dialects/"):
                active_dialects.add(Path(rel).stem)
            for attr in ("parser_class", "generator_class", "tokenizer_class"):
                sub = c.__dict__.get(attr)
                if isinstance(sub, type):
                    for sc in sub.__mro__:
                        srel = _rel_path(getattr(sys.modules.get(sc.__module__), "__file__", None), root_prefix)
                        if srel and srel.startswith(("sqlglot/parsers/", "sqlglot/generators/")):
                            active_dialects.add(Path(srel).stem)

    @classmethod  # type: ignore[misc]
    def hooked_dialect_get(cls, key, default=None):
        res = orig_dialect_get(cls, key, default)
        if res is not None:
            _record_dialect_cls(res)
        return res

    @classmethod  # type: ignore[misc]
    def hooked_dialect_getitem(cls, key):
        res = orig_dialect_getitem(cls, key)
        _record_dialect_cls(res)
        return res

    def hooked_dialect_init(self, *args, **kwargs):
        _record_dialect_cls(type(self))
        return orig_dialect_init(self, *args, **kwargs)

    def hooked_expr_init(self, **args):
        if collecting:
            for c in type(self).__mro__:
                if c is exp.Expr or c is object:
                    break
                active_exprs.add(c.__name__)
        return orig_expr_init(self, **args)

    _Dialect.get = hooked_dialect_get  # type: ignore[method-assign]
    _Dialect.__getitem__ = hooked_dialect_getitem  # type: ignore[method-assign]
    Dialect.__init__ = hooked_dialect_init  # type: ignore[method-assign]
    exp.Expr.__init__ = hooked_expr_init  # type: ignore[method-assign]

    # 3. Dispatch dict wrapper (installed AFTER special meta-tests run!)
    class TrackedDict(dict):
        __slots__ = ("_owners",)

        def __init__(self, initial, owners):
            super().__init__(initial)
            self._owners = owners  # list of (rel, cls_name, attr_name, d_orig)

        def _hit(self, key: object) -> None:
            if not collecting:
                return
            kr = _key_repr(key)
            for rel, cname, attr, d_orig in self._owners:
                if key in d_orig:
                    active_dict_keys.add((rel, cname, attr, kr))

        def _hit_all(self) -> None:
            if not collecting:
                return
            for rel, cname, attr, _ in self._owners:
                active_dict_keys.add((rel, cname, attr, "*"))

        def __getitem__(self, key):
            self._hit(key)
            return super().__getitem__(key)

        def get(self, key, default=None):
            self._hit(key)
            return super().get(key, default)

        def __contains__(self, key):
            self._hit(key)
            return super().__contains__(key)

        def __iter__(self):
            self._hit_all()
            return super().__iter__()

        def items(self):
            self._hit_all()
            return super().items()

        def keys(self):
            self._hit_all()
            return super().keys()

        def values(self):
            self._hit_all()
            return super().values()

        def copy(self):
            self._hit_all()
            return dict(self)

    dicts_wrapped = False

    def install_dispatch_wrappers() -> None:
        nonlocal dicts_wrapped
        if dicts_wrapped:
            return
        # Ensure all dialects and standalone parser/generator/tokenizer modules are loaded
        _ = Dialect.classes
        Dialect._classes["dialect"] = Dialect

        for sub_pkg in ("parsers", "generators", "dialects"):
            sub_dir = root / "sqlglot" / sub_pkg
            if sub_dir.is_dir():
                for py_path in sorted(sub_dir.glob("*.py")):
                    if py_path.name == "__init__.py":
                        continue
                    try:
                        importlib.import_module(f"sqlglot.{sub_pkg}.{py_path.stem}")
                    except Exception:
                        pass

        import sqlglot.generator as gen_mod
        from sqlglot.generator import Generator
        from sqlglot.parser import Parser
        from sqlglot.tokens import Tokenizer

        target_classes: set[type] = set()

        def _add_subclasses(base_cls: type) -> None:
            stack = [base_cls]
            while stack:
                cur = stack.pop()
                for c in cur.__mro__:
                    target_classes.add(c)
                for sub in cur.__subclasses__():
                    if sub not in target_classes:
                        stack.append(sub)

        for root_cls in (Dialect, Parser, Generator, Tokenizer):
            _add_subclasses(root_cls)

        for dcls in list(Dialect._classes.values()):
            if not isinstance(dcls, type):
                continue
            for c in dcls.__mro__:
                target_classes.add(c)
            for comp in ("parser_class", "generator_class", "tokenizer_class"):
                comp_cls = getattr(dcls, comp, None)
                if isinstance(comp_cls, type):
                    for c in comp_cls.__mro__:
                        target_classes.add(c)

        for cls in target_classes:
            cls_rel = _rel_path(getattr(sys.modules.get(cls.__module__), "__file__", None), root_prefix)
            if not cls_rel or not cls_rel.startswith("sqlglot/"):
                continue
            for attr in list(cls.__dict__.keys()):
                if attr not in DISPATCH_DICT_ATTRS:
                    continue
                val = cls.__dict__[attr]
                if not isinstance(val, dict) or isinstance(val, TrackedDict):
                    continue
                owners: list[tuple[str, str, str, dict]] = []
                for base in cls.__mro__:
                    base_rel = _rel_path(
                        getattr(sys.modules.get(base.__module__), "__file__", None),
                        root_prefix,
                    )
                    if not base_rel or not base_rel.startswith("sqlglot/"):
                        continue
                    bd = base.__dict__.get(attr)
                    if isinstance(bd, dict):
                        raw = dict.copy(bd)
                        owners.append((base_rel, base.__name__, attr, raw))
                setattr(cls, attr, TrackedDict(dict.copy(val), owners))

        gen_mod._DISPATCH_CACHE.clear()

        def tracked_build_dispatch(cls):
            transforms = cls.TRANSFORMS
            owners = transforms._owners if isinstance(transforms, TrackedDict) else []
            dispatch = dict.copy(transforms)
            for attr_name in dir(cls):
                if not attr_name.endswith("_sql") or attr_name.startswith("_"):
                    continue
                expr_key = attr_name[:-4]
                expr_cls = exp.EXPR_CLASSES.get(expr_key)
                if expr_cls and expr_cls not in dispatch:
                    dispatch[expr_cls] = getattr(cls, attr_name)
            return TrackedDict(dispatch, owners) if owners else dispatch

        gen_mod._build_dispatch = tracked_build_dispatch
        dicts_wrapped = True

    # 4. Fixture entry hooks for per-entry line coverage
    orig_load_fixtures = helpers.load_sql_fixtures
    orig_load_pairs = helpers.load_sql_fixture_pairs
    entry_track_enabled = False

    def hooked_load_fixtures(filename):
        nonlocal cur_entry
        if not entry_track_enabled:
            yield from orig_load_fixtures(filename)
            return
        for idx, item in enumerate(orig_load_fixtures(filename)):
            cur_entry = idx
            sys.monitoring.restart_events()
            yield item
        cur_entry = None
        sys.monitoring.restart_events()

    def hooked_load_pairs(filename):
        nonlocal cur_entry
        if not entry_track_enabled:
            yield from orig_load_pairs(filename)
            return
        for idx, item in enumerate(orig_load_pairs(filename)):
            cur_entry = idx
            sys.monitoring.restart_events()
            yield item
        cur_entry = None
        sys.monitoring.restart_events()

    helpers.load_sql_fixtures = hooked_load_fixtures
    helpers.load_sql_fixture_pairs = hooked_load_pairs

    # 5. sys.monitoring setup
    tool_id = sys.monitoring.PROFILER_ID
    sys.monitoring.use_tool_id(tool_id, f"br_cov_{worker_id}")
    rel_cache: dict[str, str | None] = {}

    def on_line(code: types.CodeType, lineno: int):
        if not collecting:
            return sys.monitoring.DISABLE
        fn = code.co_filename
        rel = rel_cache.get(fn)
        if rel is None and fn not in rel_cache:
            rel = _rel_path(fn, root_prefix)
            rel_cache[fn] = rel
        if rel is not None and not rel.startswith("<"):
            key = (rel, lineno)
            active_cov.add(key)
            if entry_track_enabled:
                active_entry_cov[key].add(-1 if cur_entry is None else cur_entry)
        return sys.monitoring.DISABLE

    sys.monitoring.register_callback(tool_id, sys.monitoring.events.LINE, on_line)

    # Storage returned from this worker
    line_tests: dict[tuple[str, int], set[int]] = defaultdict(set)
    entry_line_map: dict[tuple[str, int], dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    read_tests: dict[str, set[int]] = defaultdict(set)
    dialect_tests: dict[str, set[int]] = defaultdict(set)
    expr_tests: dict[str, set[int]] = defaultdict(set)
    dict_key_tests: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    test_cpu_ms: dict[int, int] = {}
    opt_duckdb_pairs: dict[tuple[str, int], tuple[str, str]] = {}
    exec_baseline: dict = {}

    # Priority ordering so special reflection/import tests run cleanly BEFORE Dialect.classes/wrappers
    PRIORITY = {
        "tests/test_typing.py": 0,
        "tests/test_dialect_imports.py": 1,
        "tests/test_dialect_entry_points.py": 2,
    }
    ordered_files = sorted(assigned_files, key=lambda f: (PRIORITY.get(f, 10), f))

    class SyncExec:
        def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kwargs):
            if initializer:
                initializer(*pickle.loads(pickle.dumps(initargs)))

        def submit(self, fn, *args, **kwargs):
            fut: Future = Future()
            try:
                fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
                res = fn(*args, **kwargs)
                fut.set_result(pickle.loads(pickle.dumps(res)))
            except BaseException as exc:
                fut.set_exception(exc)
            return fut

        def shutdown(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    SUBSETTABLE_TESTS = frozenset(
        {
            "tests/test_optimizer.py::TestOptimizer::test_annotate_types",
            "tests/test_optimizer.py::TestOptimizer::test_annotate_funcs",
            "tests/test_optimizer.py::TestOptimizer::test_qualify_columns__invalid",
            "tests/test_serde.py::TestSerde::test_serde",
            "tests/test_transpile.py::TestTranspile::test_identity",
            "tests/test_transpile.py::TestTranspile::test_partial",
            "tests/test_transpile.py::TestTranspile::test_pretty",
            "tests/test_executor.py::TestExecutor::test_execute_tpch",
            "tests/test_executor.py::TestExecutor::test_execute_tpcds",
            "tests/test_executor.py::TestExecutor::test_optimized_tpch",
        }
    )

    for rel_file in ordered_files:
        if PRIORITY.get(rel_file, 10) >= 10 and not dicts_wrapped:
            install_dispatch_wrappers()

        t_f0 = time.monotonic()
        mod_name = rel_file[:-3].replace("/", ".")
        mod = importlib.import_module(mod_name)

        # Ensure module-bound helpers / ProcessPoolExecutor use our hooked versions
        if hasattr(mod, "load_sql_fixtures"):
            mod.load_sql_fixtures = hooked_load_fixtures
        if hasattr(mod, "load_sql_fixture_pairs"):
            mod.load_sql_fixture_pairs = hooked_load_pairs
        if hasattr(mod, "ProcessPoolExecutor"):
            mod.ProcessPoolExecutor = SyncExec

        # Hook TestOptimizer.check_file to also capture opt_duckdb_pairs + per-entry tracing
        if rel_file == "tests/test_optimizer.py":
            orig_check_file = mod.TestOptimizer.check_file

            def instrumented_check_file(self_opt, file, func, pretty=False, execute=None, only=None, **kwargs):
                nonlocal collecting
                was_collecting = collecting
                collecting = False
                for idx, (meta, sql, expected) in enumerate(orig_load_pairs(f"optimizer/{file}.sql")):
                    title = meta.get("title") or f"{idx + 1}, {sql}"
                    if only and title != only:
                        continue
                    dialect = meta.get("dialect")
                    fixture_execute = False if execute is False else meta.get("execute", execute)
                    if min_py := meta.get("execute_min_python"):
                        fixture_execute = helpers.string_to_bool(fixture_execute) and sys.version_info >= tuple(
                            map(int, min_py.split("."))
                        )
                    if helpers.string_to_bool(fixture_execute):
                        func_kwargs = kwargs.copy()
                        if schema := meta.get("schema"):
                            func_kwargs["schema"] = __import__("json").loads(schema)
                        for opt_k in ("leave_tables_isolated", "validate_qualify_columns", "canonicalize_table_aliases"):
                            if meta.get(opt_k) is not None:
                                func_kwargs[opt_k] = helpers.string_to_bool(meta[opt_k])
                        if dialect:
                            func_kwargs["dialect"] = dialect
                        s_raw = parse_one(sql, read=dialect)
                        optimized = func(parse_one(sql, read=dialect), **func_kwargs)
                        s_exp = s_raw if isinstance(s_raw, (exp.Query, exp.Values)) else exp.select(exp.alias_(s_raw, "_col_0"))
                        o_exp = optimized if isinstance(optimized, (exp.Query, exp.Values)) else exp.select(exp.alias_(optimized, "_col_0"))
                        opt_duckdb_pairs[(file, idx)] = (
                            s_exp.sql(dialect="duckdb"),
                            o_exp.sql(dialect="duckdb"),
                        )
                collecting = was_collecting
                sys.monitoring.restart_events()
                return orig_check_file(self_opt, file, func, pretty=pretty, execute=execute, only=only, **kwargs)

            mod.TestOptimizer.check_file = instrumented_check_file

        # Hook TestExecutor to capture exec_baseline + per-entry tracing on tpch/tpcds
        if rel_file == "tests/test_executor.py":
            import duckdb
            import pandas as pd

            orig_subtest_helper = mod.TestExecutor.subtestHelper
            orig_mp_exec = mod.TestExecutor._mp_execute
            orig_opt_tpch = mod.TestExecutor.test_optimized_tpch

            def _plain_tables(tables_dict):
                return {k: (tuple(v.columns), list(v.rows)) for k, v in tables_dict.items()}

            def hooked_mp_execute(self_ex, schema, tables, sqls, tpch):
                nonlocal collecting, cur_entry
                key_name = "tpch_entries" if tpch else "tpcds_entries"
                exec_baseline["tpch_tables" if tpch else "tpcds_tables"] = _plain_tables(tables)
                entries: list = []
                mod.initializer(schema, tables)
                for idx, (args, sql, opt_expected) in enumerate(sqls):
                    cur_entry = idx
                    sys.monitoring.restart_events()
                    expr_p, args_p = pickle.loads(pickle.dumps((parse_one(sql), args)))
                    t_q0 = time.monotonic()
                    table = mod.mp_execute(expr_p, args_p)
                    dt_ms = max(1, round((time.monotonic() - t_q0) * 1000))
                    table = pickle.loads(pickle.dumps(table))
                    if table is not None:
                        orig_subtest_helper(self_ex, idx, table, tpch=tpch)
                    # Capture baseline snapshot with tracing paused
                    was_coll = collecting
                    collecting = False
                    duck_sql = transpile(sql, write="duckdb")[0]
                    if table is None:
                        entries.append((False, duck_sql, None, None, None, None, 0))
                    else:
                        df = self_ex.cache[sql]
                        duck_cols = [str(c) for c in df.columns]
                        duck_rows = [
                            tuple(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in row)
                            for row in df.itertuples(index=False, name=None)
                        ]
                        opt_obj = optimizer.optimize(parse_one(sql), schema=schema)
                        entries.append(
                            (
                                True,
                                duck_sql,
                                opt_obj.sql(),
                                compute_plan_py_sig(opt_obj),
                                (tuple(table.columns), list(table.rows)),
                                (duck_cols, duck_rows),
                                dt_ms,
                            )
                        )
                    collecting = was_coll
                    sys.monitoring.restart_events()
                exec_baseline[key_name] = entries
                cur_entry = None
                sys.monitoring.restart_events()

            def hooked_opt_tpch(self_ex):
                nonlocal collecting, cur_entry
                opt_pairs: list[tuple[str, str]] = []
                for idx, (_, sql, optimized) in enumerate(self_ex.tpch_sqls):
                    cur_entry = idx
                    sys.monitoring.restart_events()
                    with self_ex.subTest(f"{idx + 1}, {sql}"):
                        a = self_ex.cached_execute(sql, tpch=True)
                        b = self_ex.tpch_conn.execute(transpile(optimized, write="duckdb")[0]).fetchdf()
                        self_ex.rename_anonymous(b, a)
                        mod.assert_frame_equal(a, b)
                    was_coll = collecting
                    collecting = False
                    opt_pairs.append((transpile(sql, write="duckdb")[0], transpile(optimized, write="duckdb")[0]))
                    collecting = was_coll
                    sys.monitoring.restart_events()
                exec_baseline["opt_tpch_pairs"] = opt_pairs
                cur_entry = None
                sys.monitoring.restart_events()

            mod.TestExecutor._mp_execute = hooked_mp_execute
            mod.TestExecutor.test_optimized_tpch = hooked_opt_tpch

        # Discover TestCase subclasses defined in this module
        test_classes = [
            obj
            for obj in vars(mod).values()
            if isinstance(obj, type)
            and issubclass(obj, unittest.TestCase)
            and obj.__module__ == mod.__name__
        ]

        sys.monitoring.set_events(tool_id, sys.monitoring.events.LINE)

        for cls in test_classes:
            # Trace setUpClass once and attach to all tests of this class
            setup_cov: set[tuple[str, int]] = set()
            setup_reads: set[str] = set()
            setup_dialects: set[str] = set()
            setup_exprs: set[str] = set()
            setup_dkeys: set[tuple[str, str, str, str]] = set()

            active_cov.clear()
            active_reads.clear()
            active_dialects.clear()
            active_exprs.clear()
            active_dict_keys.clear()
            cur_entry = None
            entry_track_enabled = False
            collecting = True
            sys.monitoring.restart_events()
            cls.setUpClass()
            collecting = False

            setup_cov = set(active_cov)
            setup_reads = set(active_reads)
            setup_dialects = set(active_dialects)
            setup_exprs = set(active_exprs)
            setup_dkeys = set(active_dict_keys)

            methods = loader.getTestCaseNames(cls)
            for mth in methods:
                tid = f"{rel_file}::{cls.__name__}::{mth}"
                tidx = id_to_idx.get(tid)
                if tidx is None:
                    continue

                # Snapshot Dialect._classes before test_dialect_entry_points
                saved_dialects = dict(Dialect._classes) if cls.__name__ == "TestDialectEntryPoints" else None

                active_cov.clear()
                active_entry_cov.clear()
                active_reads.clear()
                active_dialects.clear()
                active_exprs.clear()
                active_dict_keys.clear()
                cur_entry = None
                entry_track_enabled = (tid in SUBSETTABLE_TESTS) or (
                    rel_file == "tests/test_optimizer.py" and mth not in (
                        "test_annotate_types",
                        "test_annotate_funcs",
                        "test_qualify_columns__invalid",
                    )
                )

                tc = cls(mth)
                cpu0 = time.process_time()
                collecting = True
                sys.monitoring.restart_events()
                tc.run(test_result)
                collecting = False
                cpu_ms = round((time.process_time() - cpu0) * 1000)

                entry_track_enabled = False
                cur_entry = None

                if saved_dialects is not None:
                    Dialect._classes.update(saved_dialects)

                test_cpu_ms[tidx] = cpu_ms
                cov_all = active_cov | setup_cov
                for key in cov_all:
                    line_tests[key].add(tidx)

                for key, entries in active_entry_cov.items():
                    if -1 in entries:
                        entry_line_map[key][tidx].add(-1)
                    else:
                        entry_line_map[key][tidx].update(entries)

                for rf in active_reads | setup_reads:
                    read_tests[rf].add(tidx)
                for dn in active_dialects | setup_dialects:
                    dialect_tests[dn].add(tidx)
                for en in active_exprs | setup_exprs:
                    expr_tests[en].add(tidx)
                for dk in active_dict_keys | setup_dkeys:
                    dict_key_tests[dk].add(tidx)

            try:
                cls.doClassCleanups()
                cls.tearDownClass()
            except Exception:
                pass

        log(f"Finished {rel_file} in {time.monotonic() - t_f0:.2f}s")

    sys.monitoring.set_events(tool_id, 0)
    os.dup2(orig_stdout_fd, 1)
    os.close(orig_stdout_fd)

    if test_result.errors or test_result.failures:
        log(f"WARNING baseline had {len(test_result.errors)} errors, {len(test_result.failures)} failures")
        for t, tb in test_result.errors[:3] + test_result.failures[:3]:
            log(f"  {t}: {tb.splitlines()[-1]}")

    # Convert sets to sorted lists for clean IPC pickling
    return {
        "line_tests": {k: sorted(v) for k, v in line_tests.items()},
        "entry_line_map": {
            k: {tidx: sorted(es) for tidx, es in tmap.items()}
            for k, tmap in entry_line_map.items()
        },
        "read_tests": {k: sorted(v) for k, v in read_tests.items()},
        "dialect_tests": {k: sorted(v) for k, v in dialect_tests.items()},
        "expr_tests": {k: sorted(v) for k, v in expr_tests.items()},
        "dict_key_tests": {k: sorted(v) for k, v in dict_key_tests.items()},
        "test_cpu_ms": test_cpu_ms,
        "opt_duckdb_pairs": opt_duckdb_pairs,
        "exec_baseline": exec_baseline,
        "errors": len(test_result.errors),
        "failures": len(test_result.failures),
    }


def _partition_files(all_files: list[str], num_workers: int) -> list[list[str]]:
    if num_workers <= 1:
        return [all_files]
    heavy = ["tests/test_executor.py", "tests/test_optimizer.py", "tests/test_parser.py"]
    buckets: list[list[str]] = [[] for _ in range(num_workers)]
    idx = 0
    for hf in heavy:
        if hf in all_files:
            buckets[idx % num_workers].append(hf)
            idx += 1
    rem = [f for f in all_files if f not in set(heavy)]
    if num_workers >= 4:
        # Dedicate workers 0,1,2 to heavy files; give all remaining fast files to worker 3+
        start = 3
        span = num_workers - 3
        for i, rf in enumerate(rem):
            buckets[start + (i % span)].append(rf)
    else:
        for rf in rem:
            buckets[idx % num_workers].append(rf)
            idx += 1
    return [b for b in buckets if b]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build sqlglot coverage maps and baseline caches.")
    parser.add_argument("--sqlglot", type=Path, default=Path(os.environ.get("SQLGLOT_DIR", "/opt/sqlglot")))
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("MAPS_DIR", "/app/maps")))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    t0 = time.monotonic()
    root = args.sqlglot.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["SKIP_INTEGRATION"] = "0"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    print(f"Discovering tests in {root}...", file=sys.stderr)
    # Suppress sqlglot __version__ print during discovery import
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    os.dup2(devnull, 1)
    os.close(devnull)
    try:
        discovered = _discover_tests(root)
    finally:
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)

    test_ids = [f"{r}::{c}::{m}" for r, c, m in discovered]
    id_to_idx = {tid: i for i, tid in enumerate(test_ids)}
    test_files = sorted({r for r, _, _ in discovered})
    print(f"Discovered {len(test_ids)} tests across {len(test_files)} files.", file=sys.stderr)

    print("Building static import closure and attr refs...", file=sys.stderr)
    file_imports = _build_static_file_imports(root)
    attr_refs_sqlglot, attr_refs_tests, metaclass_attrs = _build_attr_refs(root, file_imports)

    buckets = _partition_files(test_files, max(1, args.workers))
    print(f"Tracing across {len(buckets)} parallel worker(s)...", file=sys.stderr)

    if len(buckets) == 1:
        results = [_worker_trace(0, str(root), buckets[0], id_to_idx)]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=len(buckets)) as pool:
            results = pool.starmap(
                _worker_trace,
                [(i, str(root), b, id_to_idx) for i, b in enumerate(buckets)],
            )

    # Merge worker results
    merged_line_tests: dict[tuple[str, int], set[int]] = defaultdict(set)
    merged_entry_map: dict[tuple[str, int], dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    merged_reads: dict[str, set[int]] = defaultdict(set)
    merged_dialects: dict[str, set[int]] = defaultdict(set)
    merged_exprs: dict[str, set[int]] = defaultdict(set)
    merged_dkeys: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    test_cpu_ms = [1] * len(test_ids)
    opt_duckdb_pairs: dict[tuple[str, int], tuple[str, str]] = {}
    exec_baseline: dict = {}
    total_errs = 0
    total_fails = 0

    for res in results:
        total_errs += res["errors"]
        total_fails += res["failures"]
        for k, v in res["line_tests"].items():
            merged_line_tests[k].update(v)
        for k, tmap in res["entry_line_map"].items():
            dest = merged_entry_map[k]
            for tidx, es in tmap.items():
                dest[tidx].update(es)
        for k, v in res["read_tests"].items():
            merged_reads[k].update(v)
        for k, v in res["dialect_tests"].items():
            merged_dialects[k].update(v)
        for k, v in res["expr_tests"].items():
            merged_exprs[k].update(v)
        for k, v in res["dict_key_tests"].items():
            merged_dkeys[k].update(v)
        for tidx, ms in res["test_cpu_ms"].items():
            test_cpu_ms[tidx] = ms
        opt_duckdb_pairs.update(res["opt_duckdb_pairs"])
        exec_baseline.update(res["exec_baseline"])

    # Propagate typing module inheritance (parent typing -> child dialect users)
    typing_dir = root / "sqlglot" / "typing"
    if typing_dir.is_dir():
        typing_children: dict[str, set[str]] = defaultdict(set)
        for tp in typing_dir.glob("*.py"):
            if tp.name == "__init__.py":
                continue
            try:
                tree = ast.parse(tp.read_text(encoding="utf-8"))
            except Exception:
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.startswith("sqlglot.typing")
                    and any(a.name == "EXPRESSION_METADATA" for a in node.names)
                ):
                    parent_stem = node.module.split(".")[-1]
                    if parent_stem != "typing":
                        typing_children[parent_stem].add(tp.stem)
        # Forward close typing inheritance
        changed = True
        while changed:
            changed = False
            for p_stem, kids in list(typing_children.items()):
                extra = set()
                for k in kids:
                    extra |= typing_children.get(k, set())
                if not extra.issubset(kids):
                    kids |= extra
                    changed = True
        for p_stem, kids in typing_children.items():
            for k in kids:
                merged_dialects[p_stem].update(merged_dialects.get(k, ()))

    # Intern test index groups for fast, compact marshal storage
    group_id_by_tuple: dict[tuple[int, ...], int] = {}
    groups: list[list[int]] = []

    def intern_group(indices: set[int] | list[int]) -> int:
        tup = tuple(sorted(indices))
        gid = group_id_by_tuple.get(tup)
        if gid is None:
            gid = len(groups)
            group_id_by_tuple[tup] = gid
            groups.append(list(tup))
        return gid

    file_idx_by_rel: dict[str, int] = {}
    files_list: list[str] = []

    def intern_file(rel: str) -> int:
        fid = file_idx_by_rel.get(rel)
        if fid is None:
            fid = len(files_list)
            file_idx_by_rel[rel] = fid
            files_list.append(rel)
        return fid

    # Pre-intern all source, test, config, and fixture files
    for top_name in ("setup.py", "pyproject.toml", "setup.cfg"):
        if (root / top_name).is_file():
            intern_file(top_name)
    for sub in ("sqlglot", "tests", "sqlglotc"):
        sub_dir = root / sub
        if not sub_dir.exists():
            continue
        for p in sorted(sub_dir.rglob("*")):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts:
                continue
            intern_file(str(p.relative_to(root)))

    # Per-file line coverage blobs (lazy unmarshaled at runtime only for touched files)
    lines_by_file: dict[str, dict[int, int]] = defaultdict(dict)
    all_tests_by_file: dict[str, set[int]] = defaultdict(set)
    for (rel, lineno), tset in merged_line_tests.items():
        lines_by_file[rel][lineno] = intern_group(tset)
        all_tests_by_file[rel].update(tset)

    lines_blobs: dict[str, bytes] = {
        rel: marshal.dumps(lmap, 4) for rel, lmap in lines_by_file.items()
    }
    file_all_tests: dict[str, int] = {
        rel: intern_group(tset) for rel, tset in all_tests_by_file.items() if tset
    }

    # Intern entry sets and (tidx, egid) rows into per-file blobs
    entry_groups: list[list[int]] = []
    entry_group_by_tuple: dict[tuple[int, ...], int] = {}

    def intern_eg(es: set[int]) -> int:
        if -1 in es or len(es) >= 250:
            return -1
        tup = tuple(sorted(es))
        egid = entry_group_by_tuple.get(tup)
        if egid is None:
            egid = len(entry_groups)
            entry_group_by_tuple[tup] = egid
            entry_groups.append(list(tup))
        return egid

    entry_rows: list[list[tuple[int, int]]] = []
    entry_row_by_tuple: dict[tuple[tuple[int, int], ...], int] = {}

    def intern_erow(pairs: tuple[tuple[int, int], ...]) -> int:
        rid = entry_row_by_tuple.get(pairs)
        if rid is None:
            rid = len(entry_rows)
            entry_row_by_tuple[pairs] = rid
            entry_rows.append(list(pairs))
        return rid

    entry_by_file: dict[str, dict[int, int]] = defaultdict(dict)
    for (rel, lineno), tmap in merged_entry_map.items():
        row_pairs = tuple((tidx, intern_eg(es)) for tidx, es in sorted(tmap.items()))
        entry_by_file[rel][lineno] = intern_erow(row_pairs)

    entry_blobs: dict[str, bytes] = {
        rel: marshal.dumps(emap, 4) for rel, emap in entry_by_file.items()
    }

    reads_compact: dict[str, int] = {
        rel: intern_group(tset) for rel, tset in merged_reads.items()
    }
    dialects_compact: dict[str, int] = {
        dn: intern_group(tset) for dn, tset in merged_dialects.items() if tset
    }
    exprs_compact: dict[str, int] = {
        en: intern_group(tset) for en, tset in merged_exprs.items() if tset
    }

    dkeys_by_file: dict[str, dict[tuple[str, str, str], int]] = defaultdict(dict)
    for (rel, cname, attr, kr), tset in merged_dkeys.items():
        if tset:
            dkeys_by_file[rel][(cname, attr, kr)] = intern_group(tset)
    dict_key_blobs: dict[str, bytes] = {
        rel: marshal.dumps(dmap, 4) for rel, dmap in dkeys_by_file.items()
    }

    assert len(test_ids) < 65535
    g_data = array.array("H")
    g_offs = array.array("I", [0])
    for grp in groups:
        g_data.extend(grp)
        g_offs.append(len(g_data))

    eg_data = array.array("H")
    eg_offs = array.array("I", [0])
    for eg in entry_groups:
        eg_data.extend(eg)
        eg_offs.append(len(eg_data))

    er_data = array.array("i")
    er_offs = array.array("I", [0])
    for row in entry_rows:
        for tidx, egid in row:
            er_data.append(tidx)
            er_data.append(egid)
        er_offs.append(len(er_data))

    cov_map = {
        "version": 4,
        "ids": test_ids,
        "cpu_ms": test_cpu_ms,
        "files": files_list,
        "groups_data": g_data.tobytes(),
        "groups_offs": g_offs.tobytes(),
        "lines_blobs": lines_blobs,
        "file_all_tests": file_all_tests,
        "eg_data": eg_data.tobytes(),
        "eg_offs": eg_offs.tobytes(),
        "er_data": er_data.tobytes(),
        "er_offs": er_offs.tobytes(),
        "entry_blobs": entry_blobs,
        "reads": reads_compact,
        "dialects": dialects_compact,
        "exprs": exprs_compact,
        "dict_key_blobs": dict_key_blobs,
        "file_imports": file_imports,
        "attr_refs_sqlglot": attr_refs_sqlglot,
        "attr_refs_tests": attr_refs_tests,
        "metaclass_attrs": metaclass_attrs,
    }

    cov_path = args.out / "cov_map.marshal"
    with open(cov_path, "wb") as f:
        marshal.dump(cov_map, f, 4)

    opt_path = args.out / "opt_baseline.marshal"
    with open(opt_path, "wb") as f:
        marshal.dump(opt_duckdb_pairs, f, 4)

    (args.out / "exec_baseline.pkl").unlink(missing_ok=True)
    exec_meta = {
        "tpch_entries": exec_baseline.get("tpch_entries", []),
        "tpcds_entries": exec_baseline.get("tpcds_entries", []),
        "opt_tpch_pairs": exec_baseline.get("opt_tpch_pairs", []),
    }
    exec_meta_path = args.out / "exec_meta.pkl"
    with open(exec_meta_path, "wb") as f:
        pickle.dump(exec_meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    tpch_tables_path = args.out / "tpch_tables.pkl"
    with open(tpch_tables_path, "wb") as f:
        pickle.dump(exec_baseline.get("tpch_tables", {}), f, protocol=pickle.HIGHEST_PROTOCOL)

    tpcds_tables_path = args.out / "tpcds_tables.pkl"
    with open(tpcds_tables_path, "wb") as f:
        pickle.dump(exec_baseline.get("tpcds_tables", {}), f, protocol=pickle.HIGHEST_PROTOCOL)

    dt = time.monotonic() - t0
    print(
        f"Built maps in {dt:.2f}s | errs={total_errs} fails={total_fails} | "
        f"cov_map={cov_path.stat().st_size / 1e6:.2f}MB ({len(groups)} groups) | "
        f"opt_baseline={opt_path.stat().st_size / 1e6:.2f}MB | "
        f"exec_meta={exec_meta_path.stat().st_size / 1e6:.2f}MB | "
        f"tpch_tables={tpch_tables_path.stat().st_size / 1e6:.2f}MB | "
        f"tpcds_tables={tpcds_tables_path.stat().st_size / 1e6:.2f}MB",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
