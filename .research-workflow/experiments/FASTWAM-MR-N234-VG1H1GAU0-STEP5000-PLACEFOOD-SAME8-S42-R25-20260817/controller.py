#!/usr/bin/env python3
"""R25 identity with exact train-split and legacy stats contracts."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


THIS_DIR = Path(__file__).resolve().parent
R23_DIR = THIS_DIR.parent / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R23-20260817"
R23_CONTROLLER = R23_DIR / "controller.py"
CONTRACT_MODULE = THIS_DIR.parents[2] / "experiments" / "robofactory" / "normalization_stats_contract.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r23 = _load_module("fastwam_gau0_placefood_r25_r23", R23_CONTROLLER)
stats_contract = _load_module("fastwam_gau0_placefood_r25_stats_contract", CONTRACT_MODULE)
r22 = r23.r22
r21 = r23.r21
r20 = r23.r20
r19 = r23.r19
r18 = r23.r18
r17 = r23.r17
impl = r23.impl

EXPERIMENT_ID = "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R25-20260817"
RUN_ID = "fastwam-gau0-placefood-same8-r25-20260817"
DISPLAY_NAME = "fw-gau0-placefood-same8-r25"
SOURCE_ROOT = Path(
    "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/"
    "fastwam-gau0-placefood-same8-eval-20260817-r39"
)
OUTPUT_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260817-r25")
DURABLE_ROOT = Path(
    "/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260817-r25-controller"
)
RESERVATION_PATH = DURABLE_ROOT / "prepared-reservation.json"
LATCH_PATH = DURABLE_ROOT / "submission-latch.json"
ACK_PATH = DURABLE_ROOT / "job-acknowledgement.json"
LOCAL_ROOT = Path("/run/fastwam-dlc-submit-state/workspace-270969/gau0-placefood-same8-r25")
STATE_PATH = LOCAL_ROOT / "state.json"
EXPERIMENT_REL = Path(".research-workflow/experiments") / EXPERIMENT_ID

_modules = (r23, r22, r21, r20, r19, r18, r17, impl)
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
    for _module in _modules:
        setattr(_module, _name, globals()[_name])

_base_input_bindings = impl.input_bindings
_base_validate_inputs = impl.validate_inputs


def _stats_payload(
    bindings: Mapping[str, Any], key: str, expected_path: Path
) -> Mapping[str, Any]:
    binding = bindings.get(key)
    if not isinstance(binding, Mapping):
        impl.fail(f"missing direct stats binding: {key}")
    if binding.get("path") != str(expected_path):
        impl.fail(
            f"stats path mismatch for {key}: "
            f"expected={expected_path} got={binding.get('path')!r}"
        )
    encoded = binding.get("content_b64")
    if not isinstance(encoded, str):
        impl.fail(f"stats binding lacks frozen direct bytes: {key}")
    try:
        payload = json.loads(base64.b64decode(encoded, validate=True).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        impl.fail(f"stats binding is not canonical JSON for {key}: {exc}")
    if not isinstance(payload, Mapping):
        impl.fail(f"stats binding must decode to a JSON object: {key}")
    return payload


def validate_stats_semantics(bindings: Mapping[str, Any]) -> None:
    pairs = (
        ("gau1_stats", impl.GAU1_STATS, stats_contract.TRAIN_SPLIT),
        (
            "gau0_native_stats",
            impl.GAU0_STATS,
            stats_contract.LEGACY_FULL_DATASET,
        ),
    )
    for key, expected_path, mode in pairs:
        payload = _stats_payload(bindings, key, expected_path)
        try:
            stats_contract.validate_normalization_stats_provenance(payload, mode)
        except (TypeError, ValueError) as exc:
            impl.fail(f"normalization stats semantic contract failed for {key}: {exc}")


def input_bindings() -> dict[str, Any]:
    bindings = _base_input_bindings()
    validate_stats_semantics(bindings)
    return bindings


def validate_inputs(bindings: dict[str, Any]) -> None:
    _base_validate_inputs(bindings)
    validate_stats_semantics(bindings)


def validate_worker_environment(reservation: dict) -> None:
    request_env = reservation["request"]["Envs"]
    leaked = sorted(r21.GRAPHICS_RUNTIME_KEYS.intersection(request_env))
    if leaked:
        impl.fail(f"frozen R25 request must not override provider graphics runtime: {leaked}")
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
    print("GAU0_R25_WORKER_PREFLIGHT_PASS")


for _module in _modules:
    _module.input_bindings = input_bindings
    _module.validate_inputs = validate_inputs
    _module.validate_worker_environment = validate_worker_environment
    _module.worker_preflight = worker_preflight


def __getattr__(name: str):
    return getattr(r23, name)


main = impl.main


if __name__ == "__main__":
    raise SystemExit(main())
