#!/usr/bin/env python3
"""R22 identity for the EGL-guarded GAU0 matched-panel evaluation."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
R21_DIR = THIS_DIR.parent / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R21-20260817"
R21_CONTROLLER = R21_DIR / "controller.py"


def _load_r21():
    spec = importlib.util.spec_from_file_location("fastwam_gau0_placefood_r22_r21", R21_CONTROLLER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen R21 controller: {R21_CONTROLLER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r21 = _load_r21()
r20 = r21.r20
r19 = r21.r19
r18 = r21.r18
r17 = r21.r17
impl = r21.impl

EXPERIMENT_ID = "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R22-20260817"
RUN_ID = "fastwam-gau0-placefood-same8-r22-20260817"
DISPLAY_NAME = "fw-gau0-placefood-same8-r22"
SOURCE_ROOT = Path(
    "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/"
    "fastwam-gau0-placefood-same8-eval-20260817-r36"
)
OUTPUT_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260817-r22")
DURABLE_ROOT = Path(
    "/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260817-r22-controller"
)
RESERVATION_PATH = DURABLE_ROOT / "prepared-reservation.json"
LATCH_PATH = DURABLE_ROOT / "submission-latch.json"
ACK_PATH = DURABLE_ROOT / "job-acknowledgement.json"
LOCAL_ROOT = Path("/run/fastwam-dlc-submit-state/workspace-270969/gau0-placefood-same8-r22")
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
    for _module in (r21, r20, r19, r18, r17, impl):
        setattr(_module, _name, globals()[_name])


def validate_worker_environment(reservation: dict) -> None:
    request_env = reservation["request"]["Envs"]
    leaked = sorted(r21.GRAPHICS_RUNTIME_KEYS.intersection(request_env))
    if leaked:
        impl.fail(f"frozen R22 request must not override provider graphics runtime: {leaked}")
    for key, expected in request_env.items():
        if os.environ.get(key) != expected:
            impl.fail(f"worker environment differs from frozen request: {key}")


def worker_preflight() -> None:
    if os.environ.get("FASTWAM_RESERVATION_PATH") != str(RESERVATION_PATH):
        impl.fail("worker reservation path mismatch")
    reservation = impl.load_reservation()
    impl.validate_live(reservation, output_absent=True)
    validate_worker_environment(reservation)
    r21.validate_worker_dependencies()
    print("GAU0_R22_WORKER_PREFLIGHT_PASS")


for _module in (r21, r20, r19, r18, r17, impl):
    _module.validate_worker_environment = validate_worker_environment
    _module.worker_preflight = worker_preflight


def __getattr__(name: str):
    return getattr(r21, name)


main = impl.main


if __name__ == "__main__":
    raise SystemExit(main())
