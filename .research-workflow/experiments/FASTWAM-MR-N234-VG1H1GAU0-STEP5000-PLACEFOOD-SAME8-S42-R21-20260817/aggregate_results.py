#!/usr/bin/env python3
"""R21 entrypoint for the frozen same-panel GAU0 result validator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


BASE = (
    Path(__file__).resolve().parent.parent
    / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R20-20260815"
    / "aggregate_results.py"
)
spec = importlib.util.spec_from_file_location("fastwam_gau0_placefood_r21_aggregator_impl", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load frozen R20 aggregator: {BASE}")
impl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = impl
spec.loader.exec_module(impl)

ARMS = impl.ARMS
validate_arm = impl.validate_arm
validate_baseline = impl.validate_baseline
comparison = impl.comparison
main = impl.main


if __name__ == "__main__":
    main()
