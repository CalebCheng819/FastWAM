#!/usr/bin/env python3
"""R18 identity and worker-environment fix for the frozen R17 GAU0 evaluator."""

from __future__ import annotations

import importlib.util
import os
import stat
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
R17_DIR = THIS_DIR.parent / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R17-20260814"
R17_CONTROLLER = R17_DIR / "controller.py"


def _load_r17():
    spec = importlib.util.spec_from_file_location("fastwam_gau0_placefood_r18_r17", R17_CONTROLLER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen R17 controller: {R17_CONTROLLER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r17 = _load_r17()
impl = r17.impl

# R17's dependency-probe failure branch used this legacy exception name. Bind
# it to the frozen controller's fail-closed exception so an unsuccessful probe
# remains a clean contract failure instead of becoming an AttributeError.
impl.ControllerError = impl.ContractError

EXPERIMENT_ID = "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R18-20260814"
RUN_ID = "fastwam-gau0-placefood-same8-r18-20260814"
DISPLAY_NAME = "fw-gau0-placefood-same8-r18"
SOURCE_ROOT = Path(
    "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/"
    "fastwam-gau0-placefood-same8-eval-20260814-r31"
)
OUTPUT_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260814-r18")
DURABLE_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260814-r18-controller")
RESERVATION_PATH = DURABLE_ROOT / "prepared-reservation.json"
LATCH_PATH = DURABLE_ROOT / "submission-latch.json"
ACK_PATH = DURABLE_ROOT / "job-acknowledgement.json"
LOCAL_ROOT = Path("/run/fastwam-dlc-submit-state/workspace-270969/gau0-placefood-same8-r18")
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
    setattr(r17, _name, globals()[_name])
    setattr(impl, _name, globals()[_name])


def validate_worker_environment(reservation: dict) -> None:
    """Validate the runtime-added one-link EGL shim without weakening other env bindings."""

    request_env = reservation["request"]["Envs"]
    for key, expected in request_env.items():
        if key == "LD_LIBRARY_PATH":
            continue
        if os.environ.get(key) != expected:
            impl.fail(f"worker environment differs from frozen request: {key}")

    expected_loader = request_env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    canonical_expected = [
        str((impl.NVIDIA_GRAPHICS_ROOT / "driver-lib").resolve(strict=True)),
        str(Path("/usr/local/cuda-12.8/lib64").resolve(strict=False)),
    ]
    canonical_frozen = [str(Path(item).resolve(strict=False)) for item in expected_loader if item]
    if canonical_frozen != canonical_expected:
        impl.fail(f"frozen worker loader namespace is invalid: {canonical_frozen}")

    if "SAPIEN_VULKAN_LIBRARY_PATH" in os.environ:
        impl.fail("SAPIEN_VULKAN_LIBRARY_PATH must be absent")
    shim_value = os.environ.get("FASTWAM_GL_SHIM_ROOT")
    if not shim_value:
        impl.fail("worker EGL shim root is absent")
    shim_root = Path(shim_value)
    impl.require_dir(shim_root)
    if sorted(item.name for item in shim_root.iterdir()) != ["libEGL.so.1"]:
        impl.fail("worker EGL shim must contain exactly libEGL.so.1")
    egl_link = shim_root / "libEGL.so.1"
    link_info = egl_link.lstat()
    if not stat.S_ISLNK(link_info.st_mode):
        impl.fail("worker EGL shim frontend is not a symlink")
    if egl_link.resolve(strict=True) != impl.EGL_FRONTEND.resolve(strict=True):
        impl.fail("worker EGL shim frontend target mismatch")
    if impl.EGL_FRONTEND.stat().st_size != impl.EGL_FRONTEND_BYTES:
        impl.fail("worker EGL frontend size mismatch")

    current_loader = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    canonical_current = [str(Path(item).resolve(strict=False)) for item in current_loader if item]
    canonical_runtime = [str(shim_root.resolve(strict=True)), *canonical_expected]
    if canonical_current != canonical_runtime:
        impl.fail(
            "worker LD_LIBRARY_PATH must be exactly the private EGL shim plus the frozen loader namespace"
        )


def worker_dependency_env(shim_dir: Path) -> dict[str, str]:
    """Build the dependency-probe environment without inheriting a loader suffix."""

    shim_dir = shim_dir.resolve(strict=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": impl.worker_pythonpath(),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONFAULTHANDLER": "1",
            "MUJOCO_GL": "egl",
            "EGL_PLATFORM": "surfaceless",
            "PYOPENGL_PLATFORM": "egl",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "VK_ICD_FILENAMES": str(impl.NVIDIA_GRAPHICS_ROOT / "nvidia_icd.json"),
            "VK_DRIVER_FILES": str(impl.NVIDIA_GRAPHICS_ROOT / "nvidia_icd.json"),
            "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
            "__EGL_VENDOR_LIBRARY_FILENAMES": str(impl.NVIDIA_GRAPHICS_ROOT / "10_nvidia.json"),
            "FASTWAM_GL_SHIM_ROOT": str(shim_dir),
            "LD_LIBRARY_PATH": os.pathsep.join(
                (
                    str(shim_dir),
                    str(impl.NVIDIA_GRAPHICS_ROOT / "driver-lib"),
                    "/usr/local/cuda-12.8/lib64",
                )
            ),
        }
    )
    env.pop("SAPIEN_VULKAN_LIBRARY_PATH", None)
    return env


def worker_preflight() -> None:
    if os.environ.get("FASTWAM_RESERVATION_PATH") != str(RESERVATION_PATH):
        impl.fail("worker reservation path mismatch")
    reservation = impl.load_reservation()
    impl.validate_live(reservation, output_absent=True)
    validate_worker_environment(reservation)
    r17.validate_worker_dependencies()
    print("GAU0_R18_WORKER_PREFLIGHT_PASS")


r17.validate_worker_environment = validate_worker_environment
r17.worker_dependency_env = worker_dependency_env
r17.worker_preflight = worker_preflight
impl.validate_worker_environment = validate_worker_environment
impl.worker_dependency_env = worker_dependency_env
impl.worker_preflight = worker_preflight


def __getattr__(name: str):
    return getattr(r17, name)


main = impl.main


if __name__ == "__main__":
    raise SystemExit(main())
