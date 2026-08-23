#!/usr/bin/env python3
"""Publish attempt-004 DSW results with the frozen SAME8 validator."""

from __future__ import annotations

import importlib.util
import stat
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "aggregate_results.py"
R3_RUN_ID = "fastwam-gau1-step10k-placefood-same8-r3-20260823"
RUN_ID = "fastwam-gau1-step10k-placefood-same8-dsw4-r4-20260823"


metadata = BASE_PATH.lstat()
if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
    raise RuntimeError(f"unsafe base aggregator: {BASE_PATH}")
spec = importlib.util.spec_from_file_location("fastwam_same8_base_aggregator", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load base aggregator: {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
if base.RUN_ID != R3_RUN_ID:
    raise RuntimeError(f"base aggregator run identity drift: {base.RUN_ID}")
base.RUN_ID = RUN_ID


if __name__ == "__main__":
    base.main()
