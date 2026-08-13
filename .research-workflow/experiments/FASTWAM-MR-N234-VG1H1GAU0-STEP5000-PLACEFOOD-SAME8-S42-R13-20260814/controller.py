#!/usr/bin/env python3
"""R13 identity wrapper around the actual R10 GAU0 evaluator controller."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parent / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R10-20260814"
BASE_CONTROLLER = BASE_DIR / "controller.py"


def _load_base():
    spec = importlib.util.spec_from_file_location("fastwam_gau0_placefood_r13_impl", BASE_CONTROLLER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen R10 controller implementation: {BASE_CONTROLLER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


impl = _load_base()

EXPERIMENT_ID = "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R13-20260814"
RUN_ID = "fastwam-gau0-placefood-same8-r13-20260814"
DISPLAY_NAME = "fw-gau0-placefood-same8-r13"
SOURCE_ROOT = Path(
    "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/"
    "fastwam-gau0-placefood-same8-eval-20260814-r22"
)
OUTPUT_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260814-r13")
DURABLE_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260814-r13-controller")
RESERVATION_PATH = DURABLE_ROOT / "prepared-reservation.json"
LATCH_PATH = DURABLE_ROOT / "submission-latch.json"
ACK_PATH = DURABLE_ROOT / "job-acknowledgement.json"
LOCAL_ROOT = Path("/run/fastwam-dlc-submit-state/workspace-270969/gau0-placefood-same8-r13")
STATE_PATH = LOCAL_ROOT / "state.json"
EXPERIMENT_REL = Path(".research-workflow/experiments") / EXPERIMENT_ID

for _name in (
    "EXPERIMENT_ID",
    "RUN_ID",
    "DISPLAY_NAME",
    "SOURCE_ROOT",
    "OUTPUT_ROOT",
    "DURABLE_ROOT",
    "RESERVATION_PATH",
    "LATCH_PATH",
    "ACK_PATH",
    "LOCAL_ROOT",
    "STATE_PATH",
    "EXPERIMENT_REL",
):
    setattr(impl, _name, globals()[_name])


def __getattr__(name: str):
    return getattr(impl, name)


main = impl.main


if __name__ == "__main__":
    main()
