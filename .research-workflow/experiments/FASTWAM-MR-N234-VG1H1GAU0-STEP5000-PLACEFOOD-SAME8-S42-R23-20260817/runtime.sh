#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

export FASTWAM_RUNTIME_EXPERIMENT_REL='.research-workflow/experiments/FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R23-20260817'
export FASTWAM_RUNTIME_GENERATION='R23'

die() {
  printf 'GAU0_R23_EVAL_FATAL: %s\n' "$*" >&2
  exit 1
}

required_env=(
  FASTWAM_SOURCE_ROOT FASTWAM_ROBOFACTORY_ROOT FASTWAM_PYTHON_EXTRA_ROOT FASTWAM_PYTHON
  FASTWAM_EGL_FRONTEND FASTWAM_EGL_FRONTEND_SIZE_BYTES
  FASTWAM_GL_FRONTEND FASTWAM_GL_FRONTEND_SIZE_BYTES
  FASTWAM_GLES1_FRONTEND FASTWAM_GLES1_FRONTEND_SIZE_BYTES
  FASTWAM_GLES2_FRONTEND FASTWAM_GLES2_FRONTEND_SIZE_BYTES
  FASTWAM_OPENGL_FRONTEND FASTWAM_OPENGL_FRONTEND_SIZE_BYTES
  FASTWAM_GLX_FRONTEND FASTWAM_GLX_FRONTEND_SIZE_BYTES
  FASTWAM_EGL_DISPATCH FASTWAM_EGL_DISPATCH_SIZE_BYTES
  FASTWAM_EGL_VENDOR FASTWAM_EGL_VENDOR_SIZE_BYTES
  FASTWAM_VULKAN_LOADER FASTWAM_VULKAN_LOADER_SIZE_BYTES
  FASTWAM_NVIDIA_GRAPHICS_ROOT FASTWAM_RESERVATION_PATH
)
for name in "${required_env[@]}"; do
  [[ -n "${!name:-}" ]] || die "missing environment variable ${name}"
done

controller="${FASTWAM_SOURCE_ROOT}/${FASTWAM_RUNTIME_EXPERIMENT_REL}/controller.py"
shared_runtime="${FASTWAM_SOURCE_ROOT}/.research-workflow/experiments/FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R21-20260817/runtime.sh"
scratch_root="$(mktemp -d /tmp/fastwam-gau0-placefood-r23.XXXXXXXX)"
shim_root="${scratch_root}/glvnd"
cleanup() {
  rm -rf -- "${scratch_root}"
}
trap cleanup EXIT
ulimit -c 0

mkdir -m 0700 -- "${shim_root}" "${scratch_root}/pycache" "${scratch_root}/tmp"

"${FASTWAM_PYTHON}" -B - <<'PY'
import os
import stat
from pathlib import Path

pairs = (
    ("FASTWAM_EGL_FRONTEND", "FASTWAM_EGL_FRONTEND_SIZE_BYTES"),
    ("FASTWAM_GL_FRONTEND", "FASTWAM_GL_FRONTEND_SIZE_BYTES"),
    ("FASTWAM_GLES1_FRONTEND", "FASTWAM_GLES1_FRONTEND_SIZE_BYTES"),
    ("FASTWAM_GLES2_FRONTEND", "FASTWAM_GLES2_FRONTEND_SIZE_BYTES"),
    ("FASTWAM_OPENGL_FRONTEND", "FASTWAM_OPENGL_FRONTEND_SIZE_BYTES"),
    ("FASTWAM_GLX_FRONTEND", "FASTWAM_GLX_FRONTEND_SIZE_BYTES"),
    ("FASTWAM_EGL_DISPATCH", "FASTWAM_EGL_DISPATCH_SIZE_BYTES"),
    ("FASTWAM_EGL_VENDOR", "FASTWAM_EGL_VENDOR_SIZE_BYTES"),
    ("FASTWAM_VULKAN_LOADER", "FASTWAM_VULKAN_LOADER_SIZE_BYTES"),
)
for path_key, size_key in pairs:
    path = Path(os.environ[path_key])
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"graphics dependency is not an ordinary file: {path}")
    expected = int(os.environ[size_key])
    if info.st_size != expected:
        raise SystemExit(f"graphics dependency size mismatch: {path}: {info.st_size} != {expected}")
for manifest_name in ("10_nvidia.json", "nvidia_icd.json"):
    manifest = Path(os.environ["FASTWAM_NVIDIA_GRAPHICS_ROOT"]) / manifest_name
    info = manifest.lstat()
    if manifest.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"graphics manifest is not an ordinary file: {manifest}")
print("GAU0_R23_GRAPHICS_INPUTS_PASS")
PY

declare -A shim_targets=(
  [libEGL.so.1]="${FASTWAM_EGL_FRONTEND}"
  [libGL.so.1]="${FASTWAM_GL_FRONTEND}"
  [libGLESv1_CM.so.1]="${FASTWAM_GLES1_FRONTEND}"
  [libGLESv2.so.2]="${FASTWAM_GLES2_FRONTEND}"
  [libOpenGL.so.0]="${FASTWAM_OPENGL_FRONTEND}"
  [libGLX.so.0]="${FASTWAM_GLX_FRONTEND}"
  [libGLdispatch.so.0]="${FASTWAM_EGL_DISPATCH}"
  [libvulkan.so.1]="${FASTWAM_VULKAN_LOADER}"
)
for soname in "${!shim_targets[@]}"; do
  ln -s -- "${shim_targets[${soname}]}" "${shim_root}/${soname}"
done
declare -A shim_aliases=(
  [libEGL.so]=libEGL.so.1
  [libGL.so]=libGL.so.1
  [libGLESv1_CM.so]=libGLESv1_CM.so.1
  [libGLESv2.so]=libGLESv2.so.2
  [libOpenGL.so]=libOpenGL.so.0
  [libGLX.so]=libGLX.so.0
  [libGLdispatch.so]=libGLdispatch.so.0
  [libvulkan.so]=libvulkan.so.1
)
for alias in "${!shim_aliases[@]}"; do
  ln -s -- "${shim_aliases[${alias}]}" "${shim_root}/${alias}"
done

export LD_LIBRARY_PATH="${shim_root}:${FASTWAM_NVIDIA_GRAPHICS_ROOT}/lib:${FASTWAM_NVIDIA_GRAPHICS_ROOT}/driver-lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VK_ICD_FILENAMES="${FASTWAM_NVIDIA_GRAPHICS_ROOT}/nvidia_icd.json"
export VK_DRIVER_FILES="${VK_ICD_FILENAMES}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${FASTWAM_NVIDIA_GRAPHICS_ROOT}/10_nvidia.json"
unset __EGL_VENDOR_LIBRARY_DIRS
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export SAPIEN_VULKAN_LIBRARY_PATH="${shim_root}/libvulkan.so.1"
export FASTWAM_GL_SHIM_ROOT="${shim_root}"
export MUJOCO_GL=egl
export EGL_PLATFORM=surfaceless
export PYOPENGL_PLATFORM=egl
export NVIDIA_DRIVER_CAPABILITIES=all
export FASTWAM_REQUIRE_PROVIDER_NATIVE_GRAPHICS=1
export PYTHONPATH="${FASTWAM_ROBOFACTORY_ROOT}:${FASTWAM_SOURCE_ROOT}/src:${FASTWAM_PYTHON_EXTRA_ROOT}:${FASTWAM_SOURCE_ROOT}/experiments/robofactory"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1
export PYTHONPYCACHEPREFIX="${scratch_root}/pycache"
export TMPDIR="${scratch_root}/tmp"

"${FASTWAM_PYTHON}" -B "${controller}" worker-preflight

gpu_count="$("${FASTWAM_PYTHON}" -B -c 'import torch; print(torch.cuda.device_count())')"
[[ "${gpu_count}" == '8' ]] || die "expected exactly 8 visible GPUs, observed ${gpu_count}"

"${FASTWAM_PYTHON}" -B - <<'PY'
import ctypes
import os
from pathlib import Path

shim = Path(os.environ["FASTWAM_GL_SHIM_ROOT"])
expected = {
    "libEGL.so.1": Path(os.environ["FASTWAM_EGL_FRONTEND"]),
    "libGL.so.1": Path(os.environ["FASTWAM_GL_FRONTEND"]),
    "libGLESv1_CM.so.1": Path(os.environ["FASTWAM_GLES1_FRONTEND"]),
    "libGLESv2.so.2": Path(os.environ["FASTWAM_GLES2_FRONTEND"]),
    "libOpenGL.so.0": Path(os.environ["FASTWAM_OPENGL_FRONTEND"]),
    "libGLX.so.0": Path(os.environ["FASTWAM_GLX_FRONTEND"]),
    "libGLdispatch.so.0": Path(os.environ["FASTWAM_EGL_DISPATCH"]),
    "libvulkan.so.1": Path(os.environ["FASTWAM_VULKAN_LOADER"]),
}
for name, target in expected.items():
    if (shim / name).resolve(strict=True) != target.resolve(strict=True):
        raise SystemExit(f"shim target mismatch: {name}")

ctypes.CDLL(str(shim / "libGLdispatch.so.0"), mode=ctypes.RTLD_GLOBAL)
egl = ctypes.CDLL(str(shim / "libEGL.so.1"), mode=ctypes.RTLD_GLOBAL)
if not hasattr(egl, "eglQueryString"):
    raise SystemExit("complete EGL frontend lacks eglQueryString")
vendor = ctypes.CDLL(os.environ["FASTWAM_EGL_VENDOR"], mode=ctypes.RTLD_GLOBAL)
if not hasattr(vendor, "__egl_Main"):
    raise SystemExit("NVIDIA EGL vendor lacks __egl_Main")
vulkan = ctypes.CDLL(str(shim / "libvulkan.so.1"), mode=ctypes.RTLD_GLOBAL)
enumerate_version = vulkan.vkEnumerateInstanceVersion
enumerate_version.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
enumerate_version.restype = ctypes.c_int32
version = ctypes.c_uint32()
if enumerate_version(ctypes.byref(version)) != 0:
    raise SystemExit("vkEnumerateInstanceVersion failed")

from OpenGL import EGL
if not callable(EGL.eglQueryString):
    raise SystemExit("PyOpenGL EGL.eglQueryString is unavailable")
from eval_robofactory_multi_robot import _preflight_environment_imports
_preflight_environment_imports(Path(os.environ["FASTWAM_ROBOFACTORY_ROOT"]))
print("GAU0_R23_COMPLETE_GLVND_VULKAN_PREFLIGHT_PASS")
PY

probe_program=$'import os\nfrom pathlib import Path\nfrom eval_robofactory_multi_robot import _build_environment\nroot = Path(os.environ["FASTWAM_ROBOFACTORY_ROOT"])\nenvironment = _build_environment(root, "PlaceFood-rf")\nenvironment.close()\nprint("GAU0_R23_PROVIDER_NATIVE_ENVIRONMENT_PROBE_PASS task=PlaceFood-rf device=0")\n'
timeout --signal=TERM --kill-after=30s 240s env CUDA_VISIBLE_DEVICES=0 \
  "${FASTWAM_PYTHON}" -B -c "${probe_program}" || die 'provider-native PlaceFood-rf construct/close probe failed'

/bin/bash "${shared_runtime}" "$@"
