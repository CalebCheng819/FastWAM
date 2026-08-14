#!/usr/bin/env python3
"""R19 identity and complete-GLVND worker fix for the frozen GAU0 evaluator."""

from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
R18_DIR = THIS_DIR.parent / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R18-20260814"
R18_CONTROLLER = R18_DIR / "controller.py"


def _load_r18():
    spec = importlib.util.spec_from_file_location("fastwam_gau0_placefood_r19_r18", R18_CONTROLLER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen R18 controller: {R18_CONTROLLER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r18 = _load_r18()
r17 = r18.r17
impl = r18.impl

EXPERIMENT_ID = "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R19-20260814"
RUN_ID = "fastwam-gau0-placefood-same8-r19-20260814"
DISPLAY_NAME = "fw-gau0-placefood-same8-r19"
SOURCE_ROOT = Path(
    "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/"
    "fastwam-gau0-placefood-same8-eval-20260814-r32"
)
OUTPUT_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260814-r19")
DURABLE_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260814-r19-controller")
RESERVATION_PATH = DURABLE_ROOT / "prepared-reservation.json"
LATCH_PATH = DURABLE_ROOT / "submission-latch.json"
ACK_PATH = DURABLE_ROOT / "job-acknowledgement.json"
LOCAL_ROOT = Path("/run/fastwam-dlc-submit-state/workspace-270969/gau0-placefood-same8-r19")
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
    for _module in (r18, r17, impl):
        setattr(_module, _name, globals()[_name])


GLVND_LINKS = {
    "libEGL.so": (impl.EGL_FRONTEND, impl.EGL_FRONTEND_BYTES),
    "libEGL.so.1": (impl.EGL_FRONTEND, impl.EGL_FRONTEND_BYTES),
    "libGL.so": (impl.GL_FRONTEND, impl.GL_FRONTEND_BYTES),
    "libGL.so.1": (impl.GL_FRONTEND, impl.GL_FRONTEND_BYTES),
    "libGLESv1_CM.so": (impl.GLES1_FRONTEND, impl.GLES1_FRONTEND_BYTES),
    "libGLESv1_CM.so.1": (impl.GLES1_FRONTEND, impl.GLES1_FRONTEND_BYTES),
    "libGLESv2.so": (impl.GLES2_FRONTEND, impl.GLES2_FRONTEND_BYTES),
    "libGLESv2.so.2": (impl.GLES2_FRONTEND, impl.GLES2_FRONTEND_BYTES),
    "libGLX.so": (impl.GLX_FRONTEND, impl.GLX_FRONTEND_BYTES),
    "libGLX.so.0": (impl.GLX_FRONTEND, impl.GLX_FRONTEND_BYTES),
    "libOpenGL.so": (impl.OPENGL_FRONTEND, impl.OPENGL_FRONTEND_BYTES),
    "libOpenGL.so.0": (impl.OPENGL_FRONTEND, impl.OPENGL_FRONTEND_BYTES),
}


def create_glvnd_links(shim_root: Path) -> None:
    impl.require_dir(shim_root)
    if any(shim_root.iterdir()):
        impl.fail("private GLVND shim must be empty before population")
    for name, (target, expected_bytes) in GLVND_LINKS.items():
        impl.require_file(target, expected_bytes)
        os.symlink(str(target), shim_root / name)


def validate_glvnd_links(shim_root: Path) -> None:
    impl.require_dir(shim_root)
    if sorted(item.name for item in shim_root.iterdir()) != sorted(GLVND_LINKS):
        impl.fail("private GLVND shim allowlist mismatch")
    for name, (target, expected_bytes) in GLVND_LINKS.items():
        link = shim_root / name
        info = link.lstat()
        if not stat.S_ISLNK(info.st_mode):
            impl.fail(f"private GLVND entry is not a symlink: {name}")
        if os.readlink(link) != str(target):
            impl.fail(f"private GLVND link text mismatch: {name}")
        if link.resolve(strict=True) != target.resolve(strict=True):
            impl.fail(f"private GLVND target mismatch: {name}")
        impl.require_file(target, expected_bytes)


def request_loader_namespace() -> list[str]:
    return [
        str((impl.NVIDIA_GRAPHICS_ROOT / "lib").resolve(strict=True)),
        str((impl.NVIDIA_GRAPHICS_ROOT / "driver-lib").resolve(strict=True)),
        str(Path("/usr/local/cuda-12.8/lib64").resolve(strict=False)),
    ]


def runtime_loader_namespace(shim_root: Path) -> list[str]:
    return [str(shim_root.resolve(strict=True)), *request_loader_namespace()]


def worker_dependency_env(shim_root: Path) -> dict[str, str]:
    shim_root = shim_root.resolve(strict=True)
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
            "FASTWAM_GL_SHIM_ROOT": str(shim_root),
            "LD_LIBRARY_PATH": os.pathsep.join(runtime_loader_namespace(shim_root)),
        }
    )
    env.pop("SAPIEN_VULKAN_LIBRARY_PATH", None)
    return env


DEPENDENCY_PROGRAM = r'''
import os
from pathlib import Path

if "SAPIEN_VULKAN_LIBRARY_PATH" in os.environ:
    raise SystemExit("SAPIEN_VULKAN_LIBRARY_PATH must be absent")
shim_root = Path(os.environ["FASTWAM_GL_SHIM_ROOT"]).resolve(strict=True)
frontends = {
    "libEGL.so": ("FASTWAM_EGL_FRONTEND", "FASTWAM_EGL_FRONTEND_SIZE_BYTES"),
    "libEGL.so.1": ("FASTWAM_EGL_FRONTEND", "FASTWAM_EGL_FRONTEND_SIZE_BYTES"),
    "libGL.so": ("FASTWAM_GL_FRONTEND", "FASTWAM_GL_FRONTEND_SIZE_BYTES"),
    "libGL.so.1": ("FASTWAM_GL_FRONTEND", "FASTWAM_GL_FRONTEND_SIZE_BYTES"),
    "libGLESv1_CM.so": ("FASTWAM_GLES1_FRONTEND", "FASTWAM_GLES1_FRONTEND_SIZE_BYTES"),
    "libGLESv1_CM.so.1": ("FASTWAM_GLES1_FRONTEND", "FASTWAM_GLES1_FRONTEND_SIZE_BYTES"),
    "libGLESv2.so": ("FASTWAM_GLES2_FRONTEND", "FASTWAM_GLES2_FRONTEND_SIZE_BYTES"),
    "libGLESv2.so.2": ("FASTWAM_GLES2_FRONTEND", "FASTWAM_GLES2_FRONTEND_SIZE_BYTES"),
    "libGLX.so": ("FASTWAM_GLX_FRONTEND", "FASTWAM_GLX_FRONTEND_SIZE_BYTES"),
    "libGLX.so.0": ("FASTWAM_GLX_FRONTEND", "FASTWAM_GLX_FRONTEND_SIZE_BYTES"),
    "libOpenGL.so": ("FASTWAM_OPENGL_FRONTEND", "FASTWAM_OPENGL_FRONTEND_SIZE_BYTES"),
    "libOpenGL.so.0": ("FASTWAM_OPENGL_FRONTEND", "FASTWAM_OPENGL_FRONTEND_SIZE_BYTES"),
}
if sorted(item.name for item in shim_root.iterdir()) != sorted(frontends):
    raise SystemExit("complete GLVND shim allowlist mismatch")
for name, (path_var, size_var) in frontends.items():
    link = shim_root / name
    if not link.is_symlink():
        raise SystemExit(f"complete GLVND entry is not a symlink: {name}")
    target = Path(os.environ[path_var]).resolve(strict=True)
    if link.resolve(strict=True) != target:
        raise SystemExit(f"complete GLVND target mismatch: {name}")
    if target.stat().st_size != int(os.environ[size_var]):
        raise SystemExit(f"complete GLVND target size mismatch: {name}")
expected_loader = [
    shim_root,
    (Path(os.environ["FASTWAM_NVIDIA_GRAPHICS_ROOT"]) / "lib").resolve(strict=True),
    (Path(os.environ["FASTWAM_NVIDIA_GRAPHICS_ROOT"]) / "driver-lib").resolve(strict=True),
    Path("/usr/local/cuda-12.8/lib64").resolve(strict=False),
]
actual_loader = [Path(item).resolve(strict=False) for item in os.environ["LD_LIBRARY_PATH"].split(os.pathsep)]
if actual_loader != expected_loader:
    raise SystemExit(f"complete GLVND loader mismatch: {actual_loader} != {expected_loader}")

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
import mani_skill
import sapien
import fastwam.runtime as runtime
import fastwam_multi_robot_policy as policy
from eval_robofactory_multi_robot import _build_environment, _preflight_environment_imports

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
environment = _build_environment(robofactory_root, "PlaceFood-rf")
environment.close()
print(f"GAU0_FULL_GLVND_ENVIRONMENT_PASS loader={actual_loader} modules={actual_modules}")
'''


def validate_worker_dependencies() -> None:
    impl.validate_python()
    impl.require_dir(impl.PYTHON_EXTRA_ROOT)
    with tempfile.TemporaryDirectory(prefix="fastwam-gau0-r19-glvnd-") as raw_shim_root:
        shim_root = Path(raw_shim_root)
        create_glvnd_links(shim_root)
        validate_glvnd_links(shim_root)
        env = worker_dependency_env(shim_root)
        env.update(
            {
                "FASTWAM_SOURCE_ROOT": str(impl.SOURCE_ROOT),
                "FASTWAM_ROBOFACTORY_ROOT": str(impl.ROBOFACTORY_ROOT),
                "FASTWAM_NVIDIA_GRAPHICS_ROOT": str(impl.NVIDIA_GRAPHICS_ROOT),
                "FASTWAM_EGL_FRONTEND": str(impl.EGL_FRONTEND),
                "FASTWAM_EGL_FRONTEND_SIZE_BYTES": str(impl.EGL_FRONTEND_BYTES),
                "FASTWAM_GL_FRONTEND": str(impl.GL_FRONTEND),
                "FASTWAM_GL_FRONTEND_SIZE_BYTES": str(impl.GL_FRONTEND_BYTES),
                "FASTWAM_GLES1_FRONTEND": str(impl.GLES1_FRONTEND),
                "FASTWAM_GLES1_FRONTEND_SIZE_BYTES": str(impl.GLES1_FRONTEND_BYTES),
                "FASTWAM_GLES2_FRONTEND": str(impl.GLES2_FRONTEND),
                "FASTWAM_GLES2_FRONTEND_SIZE_BYTES": str(impl.GLES2_FRONTEND_BYTES),
                "FASTWAM_OPENGL_FRONTEND": str(impl.OPENGL_FRONTEND),
                "FASTWAM_OPENGL_FRONTEND_SIZE_BYTES": str(impl.OPENGL_FRONTEND_BYTES),
                "FASTWAM_GLX_FRONTEND": str(impl.GLX_FRONTEND),
                "FASTWAM_GLX_FRONTEND_SIZE_BYTES": str(impl.GLX_FRONTEND_BYTES),
            }
        )
        completed = subprocess.run(
            [str(impl.PYTHON), "-B", "-c", DEPENDENCY_PROGRAM],
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    if completed.returncode != 0:
        raise impl.ContractError(
            "worker dependency validation failed: "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )


_base_runtime_env = r18.runtime_env


def runtime_env(source_commit: str) -> dict[str, str]:
    env = _base_runtime_env(source_commit)
    env.pop("SAPIEN_VULKAN_LIBRARY_PATH", None)
    env.pop("FASTWAM_GL_SHIM_ROOT", None)
    env.update(
        {
            "VK_ICD_FILENAMES": str(impl.NVIDIA_GRAPHICS_ROOT / "nvidia_icd.json"),
            "VK_DRIVER_FILES": str(impl.NVIDIA_GRAPHICS_ROOT / "nvidia_icd.json"),
            "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
            "__EGL_VENDOR_LIBRARY_FILENAMES": str(impl.NVIDIA_GRAPHICS_ROOT / "10_nvidia.json"),
            "LD_LIBRARY_PATH": os.pathsep.join(request_loader_namespace()),
        }
    )
    return env


def validate_worker_environment(reservation: dict) -> None:
    request_env = reservation["request"]["Envs"]
    for key, expected in request_env.items():
        if key == "LD_LIBRARY_PATH":
            continue
        if os.environ.get(key) != expected:
            impl.fail(f"worker environment differs from frozen request: {key}")
    frozen_loader = [str(Path(item).resolve(strict=False)) for item in request_env["LD_LIBRARY_PATH"].split(os.pathsep)]
    if frozen_loader != request_loader_namespace():
        impl.fail(f"frozen worker loader namespace is invalid: {frozen_loader}")
    if "SAPIEN_VULKAN_LIBRARY_PATH" in os.environ:
        impl.fail("SAPIEN_VULKAN_LIBRARY_PATH must be absent")
    shim_value = os.environ.get("FASTWAM_GL_SHIM_ROOT")
    if not shim_value:
        impl.fail("worker GLVND shim root is absent")
    shim_root = Path(shim_value)
    validate_glvnd_links(shim_root)
    current_loader = [str(Path(item).resolve(strict=False)) for item in os.environ["LD_LIBRARY_PATH"].split(os.pathsep)]
    if current_loader != runtime_loader_namespace(shim_root):
        impl.fail("worker LD_LIBRARY_PATH must exactly match the frozen complete GLVND namespace")


def worker_preflight() -> None:
    if os.environ.get("FASTWAM_RESERVATION_PATH") != str(RESERVATION_PATH):
        impl.fail("worker reservation path mismatch")
    reservation = impl.load_reservation()
    impl.validate_live(reservation, output_absent=True)
    validate_worker_environment(reservation)
    validate_worker_dependencies()
    print("GAU0_R19_WORKER_PREFLIGHT_PASS")


for _module in (r18, r17, impl):
    _module.runtime_env = runtime_env
    _module.worker_dependency_env = worker_dependency_env
    _module.validate_worker_dependencies = validate_worker_dependencies
    _module.validate_worker_environment = validate_worker_environment
    _module.worker_preflight = worker_preflight


def __getattr__(name: str):
    return getattr(r18, name)


main = impl.main


if __name__ == "__main__":
    raise SystemExit(main())
