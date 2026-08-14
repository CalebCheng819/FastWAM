#!/usr/bin/env python3
"""R17 identity and narrow-EGL wrapper for the frozen R10 GAU0 controller."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parent / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R10-20260814"
BASE_CONTROLLER = BASE_DIR / "controller.py"


def _load_base():
    spec = importlib.util.spec_from_file_location("fastwam_gau0_placefood_r17_impl", BASE_CONTROLLER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen R10 controller implementation: {BASE_CONTROLLER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


impl = _load_base()

EXPERIMENT_ID = "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R17-20260814"
RUN_ID = "fastwam-gau0-placefood-same8-r17-20260814"
DISPLAY_NAME = "fw-gau0-placefood-same8-r17"
SOURCE_ROOT = Path(
    "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/"
    "fastwam-gau0-placefood-same8-eval-20260814-r29"
)
OUTPUT_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260814-r17")
DURABLE_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260814-r17-controller")
RESERVATION_PATH = DURABLE_ROOT / "prepared-reservation.json"
LATCH_PATH = DURABLE_ROOT / "submission-latch.json"
ACK_PATH = DURABLE_ROOT / "job-acknowledgement.json"
LOCAL_ROOT = Path("/run/fastwam-dlc-submit-state/workspace-270969/gau0-placefood-same8-r17")
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


def worker_dependency_env(shim_dir: Path) -> dict[str, str]:
    """Expose only the missing GLVND EGL frontend plus the vendor implementation."""

    shim_dir = shim_dir.resolve(strict=True)
    env = os.environ.copy()
    library_paths = [
        str(shim_dir),
        str(impl.NVIDIA_GRAPHICS_ROOT / "driver-lib"),
        "/usr/local/cuda-12.8/lib64",
    ]
    if env.get("LD_LIBRARY_PATH"):
        library_paths.append(env["LD_LIBRARY_PATH"])
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
            "LD_LIBRARY_PATH": os.pathsep.join(library_paths),
        }
    )
    env.pop("SAPIEN_VULKAN_LIBRARY_PATH", None)
    return env


def validate_worker_dependencies() -> None:
    """Import the evaluator stack with one frozen EGL soname link only."""

    impl.validate_python()
    impl.require_dir(impl.PYTHON_EXTRA_ROOT)
    program = r'''
import os
from pathlib import Path

if "SAPIEN_VULKAN_LIBRARY_PATH" in os.environ:
    raise SystemExit("SAPIEN_VULKAN_LIBRARY_PATH must be absent")
shim_root = Path(os.environ["FASTWAM_GL_SHIM_ROOT"]).resolve(strict=True)
if sorted(item.name for item in shim_root.iterdir()) != ["libEGL.so.1"]:
    raise SystemExit(f"unexpected narrow EGL shim contents: {list(shim_root.iterdir())}")
egl_link = shim_root / "libEGL.so.1"
egl_link.lstat()
if not egl_link.is_symlink():
    raise SystemExit(f"narrow EGL frontend is not a symlink: {egl_link}")
egl_frontend = Path(os.environ["FASTWAM_EGL_FRONTEND"]).resolve(strict=True)
if egl_link.resolve(strict=True) != egl_frontend:
    raise SystemExit(f"narrow EGL frontend target mismatch: {egl_link.resolve(strict=True)} != {egl_frontend}")
if egl_frontend.stat().st_size != int(os.environ["FASTWAM_EGL_FRONTEND_SIZE_BYTES"]):
    raise SystemExit("narrow EGL frontend size mismatch")
expected_loader_prefix = [
    shim_root,
    (Path(os.environ["FASTWAM_NVIDIA_GRAPHICS_ROOT"]) / "driver-lib").resolve(strict=True),
    Path("/usr/local/cuda-12.8/lib64").resolve(strict=False),
]
loader_prefix = [Path(item).resolve(strict=False) for item in os.environ["LD_LIBRARY_PATH"].split(os.pathsep)[:3]]
if loader_prefix != expected_loader_prefix:
    raise SystemExit(f"narrow EGL loader namespace mismatch: {loader_prefix} != {expected_loader_prefix}")

from OpenGL import EGL
if not callable(getattr(EGL, "eglQueryString", None)) or not callable(getattr(EGL, "eglGetDisplay", None)):
    raise SystemExit("PyOpenGL EGL frontend is incomplete")

import boto3
import git
import torch
import transformers
import diffusers
import accelerate
import deepspeed
from eval_robofactory_multi_robot import _preflight_environment_imports
import mani_skill
import sapien
import fastwam.runtime as runtime
import fastwam_multi_robot_policy as policy

source_root = Path(os.environ["FASTWAM_SOURCE_ROOT"]).resolve(strict=True)
robofactory_root = Path(os.environ["FASTWAM_ROBOFACTORY_ROOT"]).resolve(strict=True)
expected_src = source_root / "src"
environment_modules = _preflight_environment_imports(robofactory_root)
import tasks.place_food as place_food
import utils.scenes as scenes
expected_modules = {
    "runtime": (expected_src / "fastwam" / "runtime.py").resolve(strict=True),
    "policy": (source_root / "experiments" / "robofactory" / "fastwam_multi_robot_policy.py").resolve(strict=True),
    "place_food": (robofactory_root / "tasks" / "place_food.py").resolve(strict=True),
    "scenes": (robofactory_root / "utils" / "scenes" / "__init__.py").resolve(strict=True),
}
actual_modules = {
    "runtime": Path(runtime.__file__).resolve(strict=True),
    "policy": Path(policy.__file__).resolve(strict=True),
    "place_food": Path(place_food.__file__).resolve(strict=True),
    "scenes": Path(scenes.__file__).resolve(strict=True),
}
if actual_modules != expected_modules:
    raise SystemExit(f"worker module provenance mismatch: {actual_modules} != {expected_modules}")
if environment_modules["place_food"] != str(expected_modules["place_food"]):
    raise SystemExit(f"environment preflight place_food mismatch: {environment_modules}")
if environment_modules["scenes"] != str(expected_modules["scenes"]):
    raise SystemExit(f"environment preflight scenes mismatch: {environment_modules}")
if not callable(getattr(runtime, "create_multi_robot_fastwam", None)):
    raise SystemExit("frozen fastwam.runtime lacks create_multi_robot_fastwam")
print(f"GAU0_NARROW_EGL_DEPENDENCY_PREFLIGHT_PASS loader={loader_prefix} modules={actual_modules}")
'''
    with tempfile.TemporaryDirectory(prefix="fastwam-gau0-r17-egl-") as raw_shim_dir:
        shim_dir = Path(raw_shim_dir)
        os.symlink(str(impl.EGL_FRONTEND), shim_dir / "libEGL.so.1")
        env = worker_dependency_env(shim_dir)
        env["FASTWAM_SOURCE_ROOT"] = str(impl.SOURCE_ROOT)
        env["FASTWAM_ROBOFACTORY_ROOT"] = str(impl.ROBOFACTORY_ROOT)
        env["FASTWAM_NVIDIA_GRAPHICS_ROOT"] = str(impl.NVIDIA_GRAPHICS_ROOT)
        env["FASTWAM_EGL_FRONTEND"] = str(impl.EGL_FRONTEND)
        env["FASTWAM_EGL_FRONTEND_SIZE_BYTES"] = str(impl.EGL_FRONTEND_BYTES)
        completed = subprocess.run(
            [str(impl.PYTHON), "-B", "-c", program],
            check=False,
            env=env,
            capture_output=True,
            text=True,
        )
    if completed.returncode != 0:
        raise impl.ControllerError(
            "worker dependency validation failed: "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )


_base_runtime_env = impl.runtime_env


def runtime_env(source_commit: str) -> dict[str, str]:
    """Freeze request inputs; the worker creates its private one-link EGL shim."""

    env = _base_runtime_env(source_commit)
    env.pop("SAPIEN_VULKAN_LIBRARY_PATH", None)
    env.pop("FASTWAM_GL_SHIM_ROOT", None)
    env.update(
        {
            "VK_ICD_FILENAMES": str(impl.NVIDIA_GRAPHICS_ROOT / "nvidia_icd.json"),
            "VK_DRIVER_FILES": str(impl.NVIDIA_GRAPHICS_ROOT / "nvidia_icd.json"),
            "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
            "__EGL_VENDOR_LIBRARY_FILENAMES": str(impl.NVIDIA_GRAPHICS_ROOT / "10_nvidia.json"),
            "LD_LIBRARY_PATH": os.pathsep.join(
                (
                    str(impl.NVIDIA_GRAPHICS_ROOT / "driver-lib"),
                    "/usr/local/cuda-12.8/lib64",
                )
            ),
        }
    )
    return env


impl.worker_dependency_env = worker_dependency_env
impl.validate_worker_dependencies = validate_worker_dependencies
impl.runtime_env = runtime_env


def __getattr__(name: str):
    return getattr(impl, name)


main = impl.main


if __name__ == "__main__":
    raise SystemExit(main())
