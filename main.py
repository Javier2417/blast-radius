#!/usr/bin/env python3
"""Entry point for the Blast Radius selector container.

Redirects fd 1 / sys.stdout to /dev/null BEFORE importing sqlglot or helpers
(sqlglot/__init__.py prints a version warning to stdout when uneditable) and
hands the preserved original stdout stream strictly to selector.main().
"""

from __future__ import annotations

import os
import sys


def _bootstrap() -> int:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    out_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
    finally:
        os.close(devnull)
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    out_stream = os.fdopen(out_fd, "w", encoding="utf-8")

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    import selector  # noqa: E402

    try:
        selector.main(out_stream)
    finally:
        out_stream.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(_bootstrap())
