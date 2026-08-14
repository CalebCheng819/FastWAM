#!/usr/bin/env python3
"""R16 identity and minimal-graphics wrapper for the R10 GAU0 controller."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parent / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R10-20260814"
BASE_CONTROLLER = BASE_DIR / "controller.py"


def _load_base():
    spec = importlib.util.spec_from_file_location("fastwam_gau0_placefood_r16_impl", BASE_CONTROLLER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen R10 controller implementation: {BASE_CONTROLLER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


impl = _load_base()

EXPERIMENT_ID = "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R16-20260814"
RUN_ID = "fastwam-gau0-placefood-same8-r16-20260814"
DISPLAY_NAME = "fw-gau0-placefood-same8-r16"
SOURCE_ROOT = Path(
    "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/"
    "fastwam-gau0-placefood-same8-eval-20260814-r26"
)
OUTPUT_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260814-r16")
DURABLE_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260814-r16-controller")
RESERVATION_PATH = DURABLE_ROOT / "prepared-reservation.json"
LATCH_PATH = DURABLE_ROOT / "submission-latch.json"
ACK_PATH = DURABLE_ROOT / "job-acknowledgement.json"
LOCAL_ROOT = Path("/run/fastwam-dlc-submit-state/workspace-270969/gau0-placefood-same8-r16")
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
    """Use the graphics namespace proven by the successful GAU1 evaluator."""

    del shim_dir
    env = os.environ.copy()
    library_paths = [
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
            "LD_LIBRARY_PATH": os.pathsep.join(library_paths),
        }
    )
    env.pop("SAPIEN_VULKAN_LIBRARY_PATH", None)
    env.pop("FASTWAM_GL_SHIM_ROOT", None)
    return env


def validate_worker_dependencies() -> None:
    """Import the frozen evaluator stack without injecting GLVND frontends."""

    impl.validate_python()
    impl.require_dir(impl.PYTHON_EXTRA_ROOT)
    program = r'''
import os
from pathlib import Path

if "SAPIEN_VULKAN_LIBRARY_PATH" in os.environ:
    raise SystemExit("SAPIEN_VULKAN_LIBRARY_PATH must be absent")
if "FASTWAM_GL_SHIM_ROOT" in os.environ:
    raise SystemExit("FASTWAM_GL_SHIM_ROOT must be absent")
expected_loader_prefix = [
    (Path(os.environ["FASTWAM_NVIDIA_GRAPHICS_ROOT"]) / "driver-lib").resolve(strict=True),
    Path("/usr/local/cuda-12.8/lib64").resolve(strict=False),
]
loader_prefix = [Path(item).resolve(strict=False) for item in os.environ["LD_LIBRARY_PATH"].split(os.pathsep)[:2]]
if loader_prefix != expected_loader_prefix:
    raise SystemExit(f"minimal loader namespace mismatch: {loader_prefix} != {expected_loader_prefix}")

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
print(f"GAU0_MINIMAL_GRAPHICS_DEPENDENCY_PREFLIGHT_PASS loader={loader_prefix} modules={actual_modules}")
'''
    env = worker_dependency_env(Path("unused"))
    env["FASTWAM_SOURCE_ROOT"] = str(impl.SOURCE_ROOT)
    env["FASTWAM_ROBOFACTORY_ROOT"] = str(impl.ROBOFACTORY_ROOT)
    env["FASTWAM_NVIDIA_GRAPHICS_ROOT"] = str(impl.NVIDIA_GRAPHICS_ROOT)
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
    """Freeze all R10 inputs but leave SAPIEN to the proven system loader."""

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
