#!/usr/bin/env python3
"""Blast Radius entry point coordinating static AST/coverage narrowing and forked execution."""

from __future__ import annotations

import json
import marshal
import os
from pathlib import Path
import sys
import time
from typing import TextIO

import packer
import patch_engine
import runner


def _resolve_dir(env_keys: tuple[str, ...], docker_default: str, local_default: str) -> Path:
    for key in env_keys:
        val = os.environ.get(key)
        if val:
            return Path(val)
    docker_path = Path(docker_default)
    if docker_path.exists():
        return docker_path
    return Path(local_default)


def main(out_stream: TextIO | None = None) -> None:
    if out_stream is None:
        out_stream = sys.stdout

    debug = bool(os.environ.get("BR_DEBUG"))
    data_dir = _resolve_dir(("BR_DATA", "DATA_DIR"), "/data", "sample")
    root_dir = _resolve_dir(("SQLGLOT_DIR",), "/opt/sqlglot", "/tmp/sqlglot-source").resolve()
    maps_dir = _resolve_dir(("MAPS_DIR",), "/app/maps", "/tmp/br-maps")

    with open(maps_dir / "cov_map.marshal", "rb") as f:
        ctx = patch_engine.MapContext(marshal.load(f))

    tests_data = json.loads((data_dir / "tests.json").read_text(encoding="utf-8"))["tests"]
    id_to_num: dict[str, int] = {t["id"]: t["number"] for t in tests_data}
    dur_by_num: dict[int, int] = {t["number"]: packer.duration_ms(t["seconds"]) for t in tests_data}
    by_test_file_nums: dict[str, list[int]] = {}
    for t in tests_data:
        by_test_file_nums.setdefault(t["file"], []).append(t["number"])
    all_items: list[tuple[int, int]] = [(t["number"], dur_by_num[t["number"]]) for t in tests_data]

    runner.install_lazy_third_party()
    code_cache = runner.install_code_cache(root_dir)
    func_index = runner.preload_sqlglot(root_dir)

    t_start = time.monotonic()
    batch_deadline = t_start + 150.0
    commits_dir = data_dir / "commits"

    for patch_path in sorted(commits_dir.glob("*.patch")):
        cid = patch_path.stem
        c_t0 = time.monotonic() if debug else 0.0
        try:
            patch_text = patch_path.read_text(encoding="utf-8")
            plan = patch_engine.analyze_commit(patch_text, root_dir, ctx)

            now = time.monotonic()
            rem = batch_deadline - now
            if rem <= 0.2:
                seen: set[int] = set()
                selected = []
                for tidx in plan.candidate_indices:
                    tnum = id_to_num.get(ctx.ids[tidx])
                    if tnum is not None and tnum not in seen:
                        seen.add(tnum)
                        selected.append(tnum)
                selected.sort()
            else:
                commit_budget = min(22.0, max(1.0, rem * 0.45))
                selected = runner.run_commit(
                    plan=plan,
                    ctx=ctx,
                    id_to_num=id_to_num,
                    dur_by_num=dur_by_num,
                    by_test_file_nums=by_test_file_nums,
                    root_dir=root_dir,
                    maps_dir=maps_dir,
                    code_cache=code_cache,
                    func_index=func_index,
                    commit_budget_sec=commit_budget,
                )

            cost_ms, workers = packer.pack([(n, dur_by_num[n]) for n in selected])
            if debug:
                elapsed_ms = (time.monotonic() - c_t0) * 1000.0
                print(
                    f"[BR_DEBUG] {cid}: kind={plan.kind} hot={plan.can_hot_patch} "
                    f"cands={len(plan.candidate_indices)} sel={len(selected)} "
                    f"workers={len(workers)} cost={cost_ms}ms wall={elapsed_ms:.1f}ms",
                    file=sys.stderr,
                )
        except Exception as exc:
            if debug:
                print(f"[BR_DEBUG] {cid}: ERROR {exc!r}, falling back to all tests", file=sys.stderr)
            _, workers = packer.pack(all_items)

        out_stream.write(json.dumps({"commit": cid, "workers": workers}, separators=(",", ":")) + "\n")

    out_stream.flush()


if __name__ == "__main__":
    import main as _main

    raise SystemExit(_main._bootstrap())
