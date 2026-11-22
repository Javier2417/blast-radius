#!/usr/bin/env python3
"""High-performance test selector for the Blast Radius track.

Reads patches from /data/commits/, uses pre-built maps from /app/maps.json,
and prints the optimal worker test assignment for each commit.
"""

import json
import re
import sys
from pathlib import Path

DATA = Path("/data")
MAPS_FILE = Path("/app/maps.json")
MAX_WORKERS = 8


def is_pure_comment_patch(patch_text: str) -> bool:
    """Check if patch only adds or removes comments in Python files."""
    files = re.findall(r"--- a/(.+?)\n\+\+\+ b/(.+?)\n", patch_text)
    if not files or not all(f[1].endswith(".py") for f in files):
        return False
    for line in patch_text.splitlines():
        if line.startswith(("+++", "---", "diff", "index", "@@")):
            continue
        if line.startswith("+") or line.startswith("-"):
            stripped = line[1:].strip()
            if stripped != "" and not stripped.startswith("#"):
                return False
    return True


def select_tests_for_patch(
    patch_text: str,
    maps: dict,
    by_file: dict[str, list[dict]],
    by_id: dict[str, dict],
    max_nodes_num: int | None,
) -> list[int]:
    """Select minimal failing-safe test set for a commit patch."""
    if is_pure_comment_patch(patch_text):
        return []

    changed_files = [
        m.group(2)
        for line in patch_text.splitlines()
        if (m := re.match(r"^diff --git a/(.+?) b/(.+)$", line))
    ]
    if not changed_files:
        return []

    fixture_to_tests = maps["fixture_to_tests"]
    dialect_to_tests = maps["dialect_to_tests"]
    source_to_tests = maps["source_to_tests"]
    opt_dialect_methods = maps.get("opt_dialect_methods", {})
    parser_dialect_methods = maps.get("parser_dialect_methods", {})
    expr_dialect_methods = maps.get("expr_dialect_methods", {})

    selected_files = set()
    selected_ids = set()

    for f in changed_files:
        if f in fixture_to_tests:
            for target in fixture_to_tests[f]:
                if "::" in target:
                    selected_ids.add(target)
                else:
                    selected_files.add(target)
            continue

        is_dialect = any(
            f.startswith(d)
            for d in [
                "sqlglot/dialects/",
                "sqlglot/generators/",
                "sqlglot/parsers/",
                "sqlglot/typing/",
            ]
        )
        if is_dialect:
            stem = Path(f).stem
            if stem == "dialect":
                selected_files.add("tests/dialects/test_dialect.py")
            else:
                if f in dialect_to_tests:
                    for tf in dialect_to_tests[f]:
                        selected_files.add(tf)
                for m in opt_dialect_methods.get(stem, []):
                    tid = f"tests/test_optimizer.py::TestOptimizer::{m}"
                    if tid in by_id:
                        selected_ids.add(tid)
                for m in parser_dialect_methods.get(stem, []):
                    tid = f"tests/test_parser.py::TestParser::{m}"
                    if tid in by_id:
                        selected_ids.add(tid)
                for m in expr_dialect_methods.get(stem, []):
                    tid = f"tests/test_expressions.py::TestExprs::{m}"
                    if tid in by_id:
                        selected_ids.add(tid)
                for ltf in [
                    "tests/test_transpile.py",
                    "tests/test_serde.py",
                    "tests/test_transforms.py",
                    "tests/test_schema.py",
                ]:
                    selected_files.add(ltf)
            continue

        if f in source_to_tests:
            for tf in source_to_tests[f]:
                selected_files.add(tf)

        if f.startswith("tests/") and f.endswith(".py"):
            selected_files.add(f)

        if f.startswith("tests/") and not f.endswith(".py") and "fixtures" in f:
            selected_files.add("tests/test_optimizer.py")
            selected_files.add("tests/test_executor.py")

    # Guard heavy test_executor.py: only run if executor, serde, or join logic changed
    needs_executor = any(
        f.startswith("sqlglot/executor/")
        or f in ("sqlglot/optimizer/eliminate_joins.py", "sqlglot/serde.py")
        or "tpc-" in f
        for f in changed_files
    )
    if not needs_executor:
        selected_files.discard("tests/test_executor.py")

    selected_numbers = set()
    for tf in selected_files:
        if tf in by_file:
            for t in by_file[tf]:
                selected_numbers.add(t["number"])
    for tid in selected_ids:
        if tid in by_id:
            selected_numbers.add(by_id[tid]["number"])

    # Discard 29-second test_max_nodes unless sqlglot/parser.py was explicitly modified
    if max_nodes_num is not None and not any("sqlglot/parser.py" in f for f in changed_files):
        selected_numbers.discard(max_nodes_num)

    return sorted(selected_numbers)


def split_workers_optimal(test_numbers: list[int], by_number: dict) -> list[list[int]]:
    """Partition tests across up to 8 workers using LPT scheduling to minimize simulated time."""
    if not test_numbers:
        return []

    durations = [(n, by_number[n]["seconds"]) for n in test_numbers if n in by_number]
    durations.sort(key=lambda x: x[1], reverse=True)

    best_workers: list[list[int]] = []
    best_cost = float("inf")

    for W in range(1, MAX_WORKERS + 1):
        if W > len(durations):
            break
        workers: list[list[int]] = [[] for _ in range(W)]
        loads = [0.0] * W
        for num, dur in durations:
            idx = loads.index(min(loads))
            workers[idx].append(num)
            loads[idx] += dur
        cost = max(loads) + 2.0 * W
        if cost < best_cost:
            best_cost = cost
            best_workers = workers

    return [w for w in best_workers if w]


def main():
    with open(MAPS_FILE) as f:
        maps = json.load(f)

    with open(DATA / "tests.json") as f:
        tests_data = json.load(f)

    all_tests = tests_data["tests"]
    by_number = {t["number"]: t for t in all_tests}
    by_file: dict[str, list[dict]] = {}
    for t in all_tests:
        by_file.setdefault(t["file"], []).append(t)
    by_id = {t["id"]: t for t in all_tests}

    max_nodes_test = by_id.get("tests/test_parser.py::TestParser::test_max_nodes")
    max_nodes_num = max_nodes_test["number"] if max_nodes_test else None

    commits_dir = DATA / "commits"
    for patch_path in sorted(commits_dir.glob("*.patch")):
        cid = patch_path.stem
        patch_text = patch_path.read_text()
        selected = select_tests_for_patch(patch_text, maps, by_file, by_id, max_nodes_num)
        workers = split_workers_optimal(selected, by_number)
        print(json.dumps({"commit": cid, "workers": workers}))


if __name__ == "__main__":
    main()
