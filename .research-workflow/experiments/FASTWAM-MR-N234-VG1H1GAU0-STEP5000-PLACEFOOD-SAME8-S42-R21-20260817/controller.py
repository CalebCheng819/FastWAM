#!/usr/bin/env python3
"""R21 identity and isolated graphics discovery for GAU0 evaluation."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
R20_DIR = THIS_DIR.parent / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R20-20260815"
R20_CONTROLLER = R20_DIR / "controller.py"


def _load_r20():
    spec = importlib.util.spec_from_file_location("fastwam_gau0_placefood_r21_r20", R20_CONTROLLER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen R20 controller: {R20_CONTROLLER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r20 = _load_r20()
r19 = r20.r19
r18 = r20.r18
r17 = r20.r17
impl = r20.impl

EXPERIMENT_ID = "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R21-20260817"
RUN_ID = "fastwam-gau0-placefood-same8-r21-20260817"
DISPLAY_NAME = "fw-gau0-placefood-same8-r21"
SOURCE_ROOT = Path(
    "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/"
    "fastwam-gau0-placefood-same8-eval-20260817-r35"
)
OUTPUT_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260817-r21")
DURABLE_ROOT = Path(
    "/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260817-r21-controller"
)
RESERVATION_PATH = DURABLE_ROOT / "prepared-reservation.json"
LATCH_PATH = DURABLE_ROOT / "submission-latch.json"
ACK_PATH = DURABLE_ROOT / "job-acknowledgement.json"
LOCAL_ROOT = Path("/run/fastwam-dlc-submit-state/workspace-270969/gau0-placefood-same8-r21")
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
    for _module in (r20, r19, r18, r17, impl):
        setattr(_module, _name, globals()[_name])


GRAPHICS_RUNTIME_KEYS = frozenset(
    {
        "LD_LIBRARY_PATH",
        "VK_ICD_FILENAMES",
        "VK_DRIVER_FILES",
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "__EGL_VENDOR_LIBRARY_DIRS",
        "__GLX_VENDOR_LIBRARY_NAME",
        "SAPIEN_VULKAN_LIBRARY_PATH",
        "FASTWAM_GL_SHIM_ROOT",
        "LIBGL_DRIVERS_PATH",
        "GBM_BACKEND",
        "MUJOCO_GL",
        "EGL_PLATFORM",
        "PYOPENGL_PLATFORM",
        "NVIDIA_DRIVER_CAPABILITIES",
    }
)


DEPENDENCY_PROGRAM = r'''
import importlib.util
import os
import stat
from pathlib import Path

import boto3
import git
import torch
import transformers
import diffusers
import accelerate
import deepspeed
import fastwam.runtime as runtime
import fastwam_multi_robot_policy as policy

source_root = Path(os.environ["FASTWAM_SOURCE_ROOT"]).resolve(strict=True)
robofactory_root = Path(os.environ["FASTWAM_ROBOFACTORY_ROOT"]).resolve(strict=True)
expected_src = source_root / "src"

def require_regular_file(label, path):
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"{label} must be a non-symlink ordinary file: {path}")
    return path.resolve(strict=True)

def require_installed_module(label):
    spec = importlib.util.find_spec(label)
    if spec is None or spec.origin in (None, "built-in", "frozen"):
        raise SystemExit(f"installed module spec unavailable: {label}")
    return require_regular_file(label, Path(spec.origin))

expected_modules = {
    "runtime": (expected_src / "fastwam" / "runtime.py").resolve(strict=True),
    "policy": (source_root / "experiments" / "robofactory" / "fastwam_multi_robot_policy.py").resolve(strict=True),
}
actual_modules = {
    "runtime": Path(runtime.__file__).resolve(strict=True),
    "policy": Path(policy.__file__).resolve(strict=True),
}
if actual_modules != expected_modules:
    raise SystemExit(f"worker module provenance mismatch: {actual_modules} != {expected_modules}")
graphics_specs = {
    "mani_skill": require_installed_module("mani_skill"),
    "sapien": require_installed_module("sapien"),
}
robofactory_files = {
    "place_food": require_regular_file("place_food", robofactory_root / "tasks" / "place_food.py"),
    "scenes": require_regular_file("scenes", robofactory_root / "utils" / "scenes" / "__init__.py"),
}
if not callable(getattr(runtime, "create_multi_robot_fastwam", None)):
    raise SystemExit("frozen fastwam.runtime lacks create_multi_robot_fastwam")
print(
    "GAU0_R21_STATIC_DEPENDENCY_PASS "
    f"modules={actual_modules} graphics_specs={graphics_specs} robofactory_files={robofactory_files}"
)
'''


def worker_dependency_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": impl.worker_pythonpath(),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONFAULTHANDLER": "1",
            "FASTWAM_SOURCE_ROOT": str(impl.SOURCE_ROOT),
            "FASTWAM_ROBOFACTORY_ROOT": str(impl.ROBOFACTORY_ROOT),
        }
    )
    return env


def validate_worker_dependencies() -> None:
    impl.validate_python()
    impl.require_dir(impl.PYTHON_EXTRA_ROOT)
    completed = subprocess.run(
        [str(impl.PYTHON), "-B", "-c", DEPENDENCY_PROGRAM],
        check=False,
        env=worker_dependency_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0:
        raise impl.ContractError(
            "worker static dependency validation failed: "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )


_base_runtime_env = r20.runtime_env


def runtime_env(source_commit: str) -> dict[str, str]:
    env = _base_runtime_env(source_commit)
    for key in GRAPHICS_RUNTIME_KEYS:
        env.pop(key, None)
    return env


def validate_worker_environment(reservation: dict) -> None:
    request_env = reservation["request"]["Envs"]
    leaked = sorted(GRAPHICS_RUNTIME_KEYS.intersection(request_env))
    if leaked:
        impl.fail(f"frozen R21 request must not override provider graphics runtime: {leaked}")
    for key, expected in request_env.items():
        if os.environ.get(key) != expected:
            impl.fail(f"worker environment differs from frozen request: {key}")


def worker_preflight() -> None:
    if os.environ.get("FASTWAM_RESERVATION_PATH") != str(RESERVATION_PATH):
        impl.fail("worker reservation path mismatch")
    reservation = impl.load_reservation()
    impl.validate_live(reservation, output_absent=True)
    validate_worker_environment(reservation)
    validate_worker_dependencies()
    print("GAU0_R21_WORKER_PREFLIGHT_PASS")


for _module in (r20, r19, r18, r17, impl):
    _module.runtime_env = runtime_env
    _module.worker_dependency_env = worker_dependency_env
    _module.validate_worker_dependencies = validate_worker_dependencies
    _module.validate_worker_environment = validate_worker_environment
    _module.worker_preflight = worker_preflight


def __getattr__(name: str):
    return getattr(r20, name)


main = impl.main


if __name__ == "__main__":
    raise SystemExit(main())
