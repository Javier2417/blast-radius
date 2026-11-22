#!/usr/bin/env python3
"""Fetch and extract the pinned sqlglot commit from GitHub without requiring git."""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

PINNED_COMMIT = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
ARCHIVE_URL = f"https://github.com/tobymao/sqlglot/archive/{PINNED_COMMIT}.tar.gz"


def fetch_sqlglot(dest: Path, url: str = ARCHIVE_URL) -> None:
    """Download the pinned sqlglot tarball and extract it cleanly to dest."""
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        archive_path = tmp_path / "sqlglot.tar.gz"

        last_err: Exception | None = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "blast-radius-fetch/1.0"},
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    with open(archive_path, "wb") as out:
                        shutil.copyfileobj(resp, out)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                time.sleep(2**attempt)

        if last_err is not None:
            raise last_err

        extract_root = tmp_path / "extracted"
        extract_root.mkdir()
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_root, filter="tar")

        candidates = [
            p for p in extract_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected 1 extracted root dir, found: {candidates}")

        shutil.move(str(candidates[0]), str(dest))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    target = Path(args[0] if args else os.environ.get("SQLGLOT_DIR", "/opt/sqlglot"))
    fetch_sqlglot(target)
    print(f"Fetched pinned sqlglot ({PINNED_COMMIT[:12]}) into {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
