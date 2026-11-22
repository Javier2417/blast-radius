#!/usr/bin/env python3
"""Build import and fixture maps from sqlglot source at the pinned commit.

Run during Docker build. Outputs maps.json read by the selector at runtime.
"""

import ast
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

SQLGLOT_ROOT = Path(os.environ.get("SQLGLOT_DIR", "/opt/sqlglot"))


def discover_modules(root: Path) -> dict[str, str]:
    """Map module dotted names to file paths (relative to repo root)."""
    modules = {}
    for dirpath, _, filenames in os.walk(root / "sqlglot"):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            full = Path(dirpath) / fn
            rel = full.relative_to(root)
            if fn == "__init__.py":
                mod = str(rel.parent).replace("/", ".")
            else:
                mod = str(rel.with_suffix("")).replace("/", ".")
            modules[mod] = str(rel)
    return modules


def build_import_graph(modules: dict[str, str], root: Path) -> dict[str, set[str]]:
    """Direct import graph: module -> set of sqlglot modules it imports."""
    graph = {}
    for mod, relpath in modules.items():
        path = root / relpath
        try:
            tree = ast.parse(path.read_text())
        except Exception:
            graph[mod] = set()
            continue
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("sqlglot"):
                        imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("sqlglot"):
                    imports.add(node.module)
        graph[mod] = imports
    return graph


def transitive_closure(graph: dict[str, set[str]]) -> dict[str, set[str]]:
    """For each module, all modules it transitively depends on (including itself)."""
    cache: dict[str, set[str]] = {}

    def _deps(mod: str, stack: frozenset[str] = frozenset()) -> set[str]:
        if mod in cache:
            return cache[mod]
        if mod in stack:
            return {mod}
        result = {mod}
        new_stack = stack | {mod}
        for dep in graph.get(mod, set()):
            result |= _deps(dep, new_stack)
        cache[mod] = result
        return result

    for mod in graph:
        _deps(mod)
    return cache


def build_test_imports(root: Path) -> dict[str, set[str]]:
    """test file -> set of sqlglot modules directly imported."""
    test_imports = {}
    for dirpath, _, filenames in os.walk(root / "tests"):
        for fn in filenames:
            if not fn.startswith("test_") or not fn.endswith(".py"):
                continue
            path = Path(dirpath) / fn
            rel = str(path.relative_to(root))
            try:
                tree = ast.parse(path.read_text())
            except Exception:
                continue
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("sqlglot"):
                            imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("sqlglot"):
                        imports.add(node.module)
            test_imports[rel] = imports
    return test_imports


def build_source_to_test_map(
    modules: dict[str, str],
    test_imports: dict[str, set[str]],
    closure: dict[str, set[str]],
) -> dict[str, list[str]]:
    """source file -> list of test files that (transitively) import it."""
    test_all_deps = {}
    for test_file, direct_imports in test_imports.items():
        all_deps = set()
        for imp in direct_imports:
            if imp in closure:
                all_deps |= closure[imp]
            else:
                parts = imp.split(".")
                for i in range(len(parts), 0, -1):
                    parent = ".".join(parts[:i])
                    if parent in closure:
                        all_deps |= closure[parent]
                        break
                all_deps.add(imp)
        test_all_deps[test_file] = all_deps

    src_to_tests = defaultdict(set)
    for test_file, deps in test_all_deps.items():
        for dep in deps:
            if dep in modules:
                src_to_tests[modules[dep]].add(test_file)

    return {k: sorted(v) for k, v in src_to_tests.items()}


def build_dialect_explicit_map(root: Path) -> dict[str, list[str]]:
    """Explicitly map dialect/generator/parser/typing files to their test files."""
    dialect_map: dict[str, list[str]] = {}
    name_to_test: dict[str, list[str]] = {}

    test_dir = root / "tests" / "dialects"
    if test_dir.is_dir():
        for tf in test_dir.iterdir():
            if tf.name.startswith("test_") and tf.suffix == ".py":
                dialect_name = tf.stem[5:]
                name_to_test.setdefault(dialect_name, []).append(
                    f"tests/dialects/{tf.name}"
                )

    extra_mappings = {
        "spark2": ["tests/dialects/test_spark.py", "tests/dialects/test_databricks.py"],
        "dialect": ["tests/dialects/test_dialect.py"],
        "base": ["tests/dialects/test_dialect.py"],
        "python": [],
    }

    for subdir in ["dialects", "generators", "parsers", "typing"]:
        src_dir = root / "sqlglot" / subdir
        if not src_dir.is_dir():
            continue
        for f in src_dir.iterdir():
            if not f.suffix == ".py" or f.name == "__init__.py":
                continue
            name = f.stem
            rel = f"sqlglot/{subdir}/{f.name}"
            tests = set()
            if name in name_to_test:
                tests.update(name_to_test[name])
            if name in extra_mappings:
                tests.update(extra_mappings[name])
            if tests:
                dialect_map[rel] = sorted(tests)

    return dialect_map


def build_fixture_map(root: Path) -> dict[str, list[str]]:
    """fixture file path -> list of test IDs that use it."""
    fixture_map = defaultdict(set)
    all_fixtures = set()
    for path in (root / "tests" / "fixtures").rglob("*"):
        if path.is_file() and not path.name.endswith(".csv.gz"):
            all_fixtures.add(str(path.relative_to(root)))

    for py_path in (root / "tests").rglob("test_*.py"):
        rel_py = str(py_path.relative_to(root))
        try:
            tree = ast.parse(py_path.read_text())
        except Exception:
            continue
        for class_node in tree.body:
            if isinstance(class_node, ast.ClassDef):
                for func_node in class_node.body:
                    if isinstance(func_node, ast.FunctionDef) and func_node.name.startswith("test_"):
                        tid = f"{rel_py}::{class_node.name}::{func_node.name}"
                        for sub in ast.walk(func_node):
                            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                                val = sub.value
                                for fix in all_fixtures:
                                    fix_base = Path(fix).name
                                    fix_stem = Path(fix).stem
                                    if val == fix or val == fix_base or val == fix_stem:
                                        fixture_map[fix].add(tid)
                                    elif val in fix and len(val) >= 6:
                                        fixture_map[fix].add(tid)

    # Specific known fixes
    fixture_map["tests/fixtures/optimizer/pushdown_projections.sql"].add(
        "tests/test_optimizer.py::TestOptimizer::test_pushdown_projection"
    )

    return {k: sorted(v) for k, v in fixture_map.items()}


def build_granular_dialect_methods(root: Path) -> tuple[dict, dict, dict]:
    """Map dialects to test methods in test_optimizer, test_parser, test_expressions."""
    dialects = [f.stem for f in (root / "sqlglot" / "dialects").glob("*.py") if f.name != "__init__.py"]

    # 1. Optimizer methods
    opt_dialect_methods = defaultdict(list)
    opt_tree = ast.parse((root / "tests" / "test_optimizer.py").read_text())
    for node in ast.walk(opt_tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            code = ast.unparse(node).lower()
            for d in dialects:
                if f'"{d}"' in code or f"'{d}'" in code or f"={d}" in code or f"({d}" in code:
                    opt_dialect_methods[d].append(node.name)

    # Fixture dialect annotations in optimizer .sql
    fixture_dialects = defaultdict(set)
    for f in (root / "tests" / "fixtures" / "optimizer").glob("*.sql"):
        content = f.read_text().lower()
        for d in dialects:
            if f"dialect: {d}" in content or f"read: {d}" in content or f"write: {d}" in content or f"{d}," in content or f", {d}" in content:
                fixture_dialects[f.stem].add(d)

    for node in ast.walk(opt_tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    func_name = ""
                    if isinstance(sub.func, ast.Attribute): func_name = sub.func.attr
                    elif isinstance(sub.func, ast.Name): func_name = sub.func.id
                    if func_name in ("check_file", "load_sql_fixtures", "load_sql_fixture_pairs"):
                        for arg in sub.args:
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                fix_stem = Path(arg.value).stem
                                for d in fixture_dialects.get(fix_stem, set()):
                                    if node.name not in opt_dialect_methods[d]:
                                        opt_dialect_methods[d].append(node.name)

    # 2. Parser methods
    parser_dialect_methods = defaultdict(list)
    ptree = ast.parse((root / "tests" / "test_parser.py").read_text())
    for node in ast.walk(ptree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            code = ast.unparse(node).lower()
            for d in dialects:
                if f'"{d}"' in code or f"'{d}'" in code or f"={d}" in code:
                    parser_dialect_methods[d].append(node.name)

    # 3. Expressions methods
    expr_dialect_methods = defaultdict(list)
    etree = ast.parse((root / "tests" / "test_expressions.py").read_text())
    for node in ast.walk(etree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            code = ast.unparse(node).lower()
            for d in dialects:
                if f'\"{d}\"' in code or f"'{d}'" in code or f"={d}" in code:
                    expr_dialect_methods[d].append(node.name)

    return (
        {k: sorted(v) for k, v in opt_dialect_methods.items()},
        {k: sorted(v) for k, v in parser_dialect_methods.items()},
        {k: sorted(v) for k, v in expr_dialect_methods.items()},
    )


def main():
    root = SQLGLOT_ROOT
    print(f"Building maps from {root}", file=sys.stderr)

    modules = discover_modules(root)
    print(f"  {len(modules)} sqlglot modules", file=sys.stderr)

    graph = build_import_graph(modules, root)
    closure = transitive_closure(graph)
    test_imports = build_test_imports(root)
    print(f"  {len(test_imports)} test files", file=sys.stderr)

    src_to_tests = build_source_to_test_map(modules, test_imports, closure)
    fixture_map = build_fixture_map(root)
    dialect_map = build_dialect_explicit_map(root)
    opt_d_methods, parser_d_methods, expr_d_methods = build_granular_dialect_methods(root)

    maps = {
        "source_to_tests": src_to_tests,
        "fixture_to_tests": fixture_map,
        "dialect_to_tests": dialect_map,
        "opt_dialect_methods": opt_d_methods,
        "parser_dialect_methods": parser_d_methods,
        "expr_dialect_methods": expr_d_methods,
    }

    out = Path(os.environ.get("MAPS_OUT", "/app/maps.json"))
    out.write_text(json.dumps(maps, sort_keys=True))
    print(f"  Wrote {out} ({out.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
