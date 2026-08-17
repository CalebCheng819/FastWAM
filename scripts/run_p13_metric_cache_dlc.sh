#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

die() {
  printf 'P13 metric cache error: %s\n' "$*" >&2
  exit 1
}

RUN_ID="${RUN_ID:?RUN_ID is required}"
CODE_REVISION="${FASTWAM_P13_CODE_REVISION:?FASTWAM_P13_CODE_REVISION is required}"
SOURCE_BUNDLE="${FASTWAM_P13_SOURCE_BUNDLE:?FASTWAM_P13_SOURCE_BUNDLE is required}"
RUNTIME_ARCHIVE="${FASTWAM_P13_RUNTIME_ARCHIVE:?FASTWAM_P13_RUNTIME_ARCHIVE is required}"
OUTPUT_ROOT="${FASTWAM_P13_CACHE_OUTPUT_ROOT:?FASTWAM_P13_CACHE_OUTPUT_ROOT is required}"
DATASET_ROOT="${FASTWAM_P13_DATASET_ROOT:?FASTWAM_P13_DATASET_ROOT is required}"
ROBOFACTORY_ROOT="${FASTWAM_P13_ROBOFACTORY_ROOT:?FASTWAM_P13_ROBOFACTORY_ROOT is required}"
PYTHON_BIN="${FASTWAM_P13_PYTHON:?FASTWAM_P13_PYTHON is required}"
PYTHON_EXTRA_ROOT="${FASTWAM_P13_PYTHON_EXTRA_ROOT:?FASTWAM_P13_PYTHON_EXTRA_ROOT is required}"
DRIVER_ROOT="${FASTWAM_P13_DRIVER_ROOT:?FASTWAM_P13_DRIVER_ROOT is required}"
VULKAN_LOADER="${FASTWAM_P13_VULKAN_LOADER:?FASTWAM_P13_VULKAN_LOADER is required}"
EGL_FRONTEND="${DRIVER_ROOT}/lib/libEGL.so.1.1.0"
GL_FRONTEND="${DRIVER_ROOT}/lib/libGL.so.1.7.0"
GLES1_FRONTEND="${DRIVER_ROOT}/lib/libGLESv1_CM.so.1.2.0"
GLES2_FRONTEND="${DRIVER_ROOT}/lib/libGLESv2.so.2.1.0"
OPENGL_FRONTEND="${DRIVER_ROOT}/lib/libOpenGL.so.0"
GLX_FRONTEND="${DRIVER_ROOT}/lib/libGLX.so.0"
EGL_DISPATCH="${DRIVER_ROOT}/lib/libGLdispatch.so.0"
EGL_VENDOR="${DRIVER_ROOT}/driver-lib/libEGL_nvidia.so.570.153.02"
LOCAL_ROOT="/tmp/fastwam-p13-metric-cache/${RUN_ID}"
LOCAL_REPO="${LOCAL_ROOT}/source"
PARTIAL_REPO="${LOCAL_REPO}.partial.${BASHPID}"
RUNTIME_ROOT="${LOCAL_ROOT}/runtime"
LOCAL_CACHE="${LOCAL_ROOT}/cache-output"
SCRATCH_ROOT="${LOCAL_ROOT}/scratch"

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || die "unsafe RUN_ID=${RUN_ID}"
[[ "${CODE_REVISION}" =~ ^[0-9a-f]{40}$ ]] || die "invalid Git revision"
[[ -x "${PYTHON_BIN}" ]] || die "Python is missing: ${PYTHON_BIN}"
[[ -f "${SOURCE_BUNDLE}" && ! -L "${SOURCE_BUNDLE}" ]] || die "source bundle is missing"
[[ -f "${RUNTIME_ARCHIVE}" && ! -L "${RUNTIME_ARCHIVE}" ]] || die "runtime archive is missing"
[[ -d "${DATASET_ROOT}" && ! -L "${DATASET_ROOT}" ]] || die "dataset root is missing"
[[ -d "${ROBOFACTORY_ROOT}" && ! -L "${ROBOFACTORY_ROOT}" ]] || die "RoboFactory root is missing"
[[ "$(basename -- "${ROBOFACTORY_ROOT}")" == robofactory ]] || \
  die "RoboFactory root must name the robofactory package directory"
ROBOFACTORY_PACKAGE_PARENT="$(dirname -- "${ROBOFACTORY_ROOT}")"
[[ -d "${ROBOFACTORY_PACKAGE_PARENT}" && ! -L "${ROBOFACTORY_PACKAGE_PARENT}" ]] || \
  die "RoboFactory package parent is missing"
[[ -d "${PYTHON_EXTRA_ROOT}" && ! -L "${PYTHON_EXTRA_ROOT}" ]] || die "Python extra root is missing"
[[ -f "${DRIVER_ROOT}/nvidia_icd.json" ]] || die "Vulkan ICD is missing"
[[ -f "${DRIVER_ROOT}/10_nvidia.json" ]] || die "EGL vendor file is missing"
[[ -f "${VULKAN_LOADER}" && ! -L "${VULKAN_LOADER}" ]] || die "Vulkan loader is missing"
[[ ! -e "${LOCAL_ROOT}" && ! -L "${LOCAL_ROOT}" ]] || die "node-local run root already exists: ${LOCAL_ROOT}"

verify_graphics_file() {
  local label=$1 path=$2 expected_size=$3 observed_size
  [[ -f "${path}" && ! -L "${path}" ]] || die "${label} is not an ordinary file: ${path}"
  observed_size=$(stat -c '%s' -- "${path}")
  [[ "${observed_size}" == "${expected_size}" ]] || \
    die "${label} size mismatch: ${observed_size} != ${expected_size}: ${path}"
}

verify_graphics_file EGL_FRONTEND "${EGL_FRONTEND}" 80328
verify_graphics_file GL_FRONTEND "${GL_FRONTEND}" 649416
verify_graphics_file GLES1_FRONTEND "${GLES1_FRONTEND}" 43208
verify_graphics_file GLES2_FRONTEND "${GLES2_FRONTEND}" 80064
verify_graphics_file OPENGL_FRONTEND "${OPENGL_FRONTEND}" 198848
verify_graphics_file GLX_FRONTEND "${GLX_FRONTEND}" 137616
verify_graphics_file EGL_DISPATCH "${EGL_DISPATCH}" 952576
verify_graphics_file EGL_VENDOR "${EGL_VENDOR}" 1358016
verify_graphics_file VULKAN_LOADER "${VULKAN_LOADER}" 445104

mkdir -m 0700 -p -- \
  "${RUNTIME_ROOT}" "${SCRATCH_ROOT}/xdg-cache" "${SCRATCH_ROOT}/xdg-runtime" \
  "${SCRATCH_ROOT}/torch" "${SCRATCH_ROOT}/matplotlib" "${SCRATCH_ROOT}/tmp" \
  "${SCRATCH_ROOT}/pycache" "${SCRATCH_ROOT}/graphics-lib" \
  "${SCRATCH_ROOT}/graphics-probes"
declare -A graphics_shim_targets=(
  [libEGL.so.1]="${EGL_FRONTEND}"
  [libGL.so.1]="${GL_FRONTEND}"
  [libGLESv1_CM.so.1]="${GLES1_FRONTEND}"
  [libGLESv2.so.2]="${GLES2_FRONTEND}"
  [libOpenGL.so.0]="${OPENGL_FRONTEND}"
  [libGLX.so.0]="${GLX_FRONTEND}"
  [libGLdispatch.so.0]="${EGL_DISPATCH}"
  [libvulkan.so.1]="${VULKAN_LOADER}"
)
for soname in "${!graphics_shim_targets[@]}"; do
  ln -s -- "${graphics_shim_targets[${soname}]}" "${SCRATCH_ROOT}/graphics-lib/${soname}"
done
declare -A graphics_shim_aliases=(
  [libEGL.so]=libEGL.so.1
  [libGL.so]=libGL.so.1
  [libGLESv1_CM.so]=libGLESv1_CM.so.1
  [libGLESv2.so]=libGLESv2.so.2
  [libOpenGL.so]=libOpenGL.so.0
  [libGLX.so]=libGLX.so.0
  [libGLdispatch.so]=libGLdispatch.so.0
  [libvulkan.so]=libvulkan.so.1
)
for alias in "${!graphics_shim_aliases[@]}"; do
  ln -s -- "${graphics_shim_aliases[${alias}]}" "${SCRATCH_ROOT}/graphics-lib/${alias}"
done

# SAPIEN probes both conventional GLVND vendor directories during import.  The
# DLC image may omit either directory, so make the directory contract explicit
# before any graphics-sensitive Python import.  Existing provider files are
# preserved; an absent NVIDIA entry receives the frozen CPFS manifest.
for egl_vendor_dir in /usr/share/glvnd/egl_vendor.d /etc/glvnd/egl_vendor.d; do
  [[ ! -L "${egl_vendor_dir}" ]] || die "GLVND vendor directory is a symlink: ${egl_vendor_dir}"
  if [[ -e "${egl_vendor_dir}" && ! -d "${egl_vendor_dir}" ]]; then
    die "GLVND vendor directory path is not a directory: ${egl_vendor_dir}"
  fi
  install -d -m 0755 -- "${egl_vendor_dir}"
  egl_vendor_target="${egl_vendor_dir}/10_nvidia.json"
  if [[ ! -e "${egl_vendor_target}" && ! -L "${egl_vendor_target}" ]]; then
    install -m 0644 -- "${DRIVER_ROOT}/10_nvidia.json" "${egl_vendor_target}"
  else
    [[ -f "${egl_vendor_target}" && ! -L "${egl_vendor_target}" ]] || \
      die "existing GLVND vendor entry is not an ordinary file: ${egl_vendor_target}"
  fi
done
tar -xzf "${RUNTIME_ARCHIVE}" -C "${RUNTIME_ROOT}"
[[ -d "${RUNTIME_ROOT}/site-packages" ]] || die "runtime archive layout is invalid"

git clone --quiet --no-checkout -- "${SOURCE_BUNDLE}" "${PARTIAL_REPO}"
git -C "${PARTIAL_REPO}" checkout --quiet --detach "${CODE_REVISION}"
[[ "$(git -C "${PARTIAL_REPO}" rev-parse HEAD)" == "${CODE_REVISION}" ]] || die "source revision mismatch"
[[ -z "$(git -C "${PARTIAL_REPO}" status --porcelain --untracked-files=all)" ]] || die "restored source is dirty"
mv -T -- "${PARTIAL_REPO}" "${LOCAL_REPO}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1
# RoboFactory has a mixed import contract: callers import the legacy top-level
# `tasks` package from the package directory, while that package imports
# `robofactory.utils` through the package parent.  Both entries are therefore
# required, and their order is frozen to the already-proven R25 evaluator.
export PYTHONPATH="${ROBOFACTORY_PACKAGE_PARENT}:${ROBOFACTORY_ROOT}:${LOCAL_REPO}/src:${PYTHON_EXTRA_ROOT}:${LOCAL_REPO}/scripts:${RUNTIME_ROOT}/site-packages"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"
export XDG_RUNTIME_DIR="${SCRATCH_ROOT}/xdg-runtime"
export TORCH_HOME="${SCRATCH_ROOT}/torch"
export MPLCONFIGDIR="${SCRATCH_ROOT}/matplotlib"
export TMPDIR="${SCRATCH_ROOT}/tmp"
export PYTHONPYCACHEPREFIX="${SCRATCH_ROOT}/pycache"
export FASTWAM_P13_LOCAL_REPO="${LOCAL_REPO}"
export FASTWAM_P13_ROBOFACTORY_ROOT="${ROBOFACTORY_ROOT}"
export DISPLAY=

graphics_keys=(
  LD_LIBRARY_PATH VK_ICD_FILENAMES VK_DRIVER_FILES
  __EGL_VENDOR_LIBRARY_FILENAMES __EGL_VENDOR_LIBRARY_DIRS
  __GLX_VENDOR_LIBRARY_NAME SAPIEN_VULKAN_LIBRARY_PATH FASTWAM_GL_SHIM_ROOT
  LIBGL_DRIVERS_PATH GBM_BACKEND MUJOCO_GL EGL_PLATFORM PYOPENGL_PLATFORM
  NVIDIA_DRIVER_CAPABILITIES FASTWAM_REQUIRE_PROVIDER_NATIVE_GRAPHICS
)
declare -A provider_graphics_present=()
declare -A provider_graphics_value=()
for name in "${graphics_keys[@]}"; do
  if [[ -v "${name}" ]]; then
    provider_graphics_present["${name}"]=1
    provider_graphics_value["${name}"]="${!name}"
  else
    provider_graphics_present["${name}"]=0
    provider_graphics_value["${name}"]=''
  fi
done

restore_provider_graphics() {
  local key
  for key in "${graphics_keys[@]}"; do
    if [[ "${provider_graphics_present[$key]}" == 1 ]]; then
      export "${key}=${provider_graphics_value[$key]}"
    else
      unset "${key}"
    fi
  done
}

apply_r25_graphics_contract() {
  restore_provider_graphics
  export LD_LIBRARY_PATH="${SCRATCH_ROOT}/graphics-lib:${DRIVER_ROOT}/lib:${DRIVER_ROOT}/driver-lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export VK_ICD_FILENAMES="${DRIVER_ROOT}/nvidia_icd.json"
  export VK_DRIVER_FILES="${VK_ICD_FILENAMES}"
  export __EGL_VENDOR_LIBRARY_FILENAMES="${DRIVER_ROOT}/10_nvidia.json"
  unset __EGL_VENDOR_LIBRARY_DIRS
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  export SAPIEN_VULKAN_LIBRARY_PATH="${SCRATCH_ROOT}/graphics-lib/libvulkan.so.1"
  export FASTWAM_GL_SHIM_ROOT="${SCRATCH_ROOT}/graphics-lib"
  unset LIBGL_DRIVERS_PATH GBM_BACKEND
  export MUJOCO_GL=egl
  export EGL_PLATFORM=surfaceless
  export PYOPENGL_PLATFORM=egl
  export NVIDIA_DRIVER_CAPABILITIES=all
  export FASTWAM_REQUIRE_PROVIDER_NATIVE_GRAPHICS=1
}

validate_r25_graphics_contract() {
  export FASTWAM_P13_GL_SHIM_ROOT="${SCRATCH_ROOT}/graphics-lib"
  export FASTWAM_P13_EGL_FRONTEND="${EGL_FRONTEND}"
  export FASTWAM_P13_GL_FRONTEND="${GL_FRONTEND}"
  export FASTWAM_P13_GLES1_FRONTEND="${GLES1_FRONTEND}"
  export FASTWAM_P13_GLES2_FRONTEND="${GLES2_FRONTEND}"
  export FASTWAM_P13_OPENGL_FRONTEND="${OPENGL_FRONTEND}"
  export FASTWAM_P13_GLX_FRONTEND="${GLX_FRONTEND}"
  export FASTWAM_P13_EGL_DISPATCH="${EGL_DISPATCH}"
  export FASTWAM_P13_EGL_VENDOR="${EGL_VENDOR}"
  "${PYTHON_BIN}" -B - <<'PY'
import ctypes
import os
from pathlib import Path

shim = Path(os.environ["FASTWAM_P13_GL_SHIM_ROOT"])
expected = {
    "libEGL.so.1": Path(os.environ["FASTWAM_P13_EGL_FRONTEND"]),
    "libGL.so.1": Path(os.environ["FASTWAM_P13_GL_FRONTEND"]),
    "libGLESv1_CM.so.1": Path(os.environ["FASTWAM_P13_GLES1_FRONTEND"]),
    "libGLESv2.so.2": Path(os.environ["FASTWAM_P13_GLES2_FRONTEND"]),
    "libOpenGL.so.0": Path(os.environ["FASTWAM_P13_OPENGL_FRONTEND"]),
    "libGLX.so.0": Path(os.environ["FASTWAM_P13_GLX_FRONTEND"]),
    "libGLdispatch.so.0": Path(os.environ["FASTWAM_P13_EGL_DISPATCH"]),
    "libvulkan.so.1": Path(os.environ["FASTWAM_P13_VULKAN_LOADER"]),
}
for name, target in expected.items():
    if (shim / name).resolve(strict=True) != target.resolve(strict=True):
        raise SystemExit(f"graphics shim target mismatch: {name}")

ctypes.CDLL(str(shim / "libGLdispatch.so.0"), mode=ctypes.RTLD_GLOBAL)
egl = ctypes.CDLL(str(shim / "libEGL.so.1"), mode=ctypes.RTLD_GLOBAL)
if not hasattr(egl, "eglQueryString"):
    raise SystemExit("complete EGL frontend lacks eglQueryString")
vendor = ctypes.CDLL(os.environ["FASTWAM_P13_EGL_VENDOR"], mode=ctypes.RTLD_GLOBAL)
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
if not callable(getattr(EGL, "eglQueryString", None)):
    raise SystemExit("PyOpenGL EGL.eglQueryString is unavailable")
import cv2
import mani_skill
import sapien
import robofactory
import tasks.place_food
import robofactory.utils.scenes
print("P13_METRIC_CACHE_R25_GRAPHICS_IMPORT_PREFLIGHT_PASS")
PY
}

nvidia-smi -L
"${PYTHON_BIN}" - <<'PY'
import importlib
import json

import torch

# Graphics-sensitive imports are exercised only after a candidate EGL/Vulkan
# profile is installed below.  Importing SAPIEN (directly or transitively via
# ManiSkill/RoboFactory) here would fail before the profile probe can run.
modules = ["gymnasium", "h5py"]
for name in modules:
    importlib.import_module(name)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible CUDA GPU, got {torch.cuda.device_count()}")
print(json.dumps({
    "cache_preflight": "PASS",
    "cuda_device": torch.cuda.get_device_name(0),
    "visible_gpus": torch.cuda.device_count(),
}, sort_keys=True))
PY

probe_program=$'import os\nfrom pathlib import Path\nfrom build_robofactory_metric_geometry_cache import _build_environment\nroot = Path(os.environ["FASTWAM_P13_ROBOFACTORY_ROOT"])\nenvironment = _build_environment(root, "PlaceFood-rf")\nenvironment.close()\nprint("P13_METRIC_CACHE_ENVIRONMENT_CONSTRUCTION_PROBE_PASS task=PlaceFood-rf device=0")\n'
apply_r25_graphics_contract
validate_r25_graphics_contract
timeout --signal=TERM --kill-after=30s 240s env CUDA_VISIBLE_DEVICES=0 \
  "${PYTHON_BIN}" -B -c "${probe_program}" || \
  die 'R25 graphics contract could not construct and close PlaceFood-rf'
printf 'P13_METRIC_CACHE_GRAPHICS_PREFLIGHT_PASS contract=r25-complete-glvnd\n'

"${PYTHON_BIN}" "${LOCAL_REPO}/scripts/build_robofactory_metric_geometry_cache.py" \
  --dataset-root "${DATASET_ROOT}" \
  --robofactory-root "${ROBOFACTORY_ROOT}" \
  --output-root "${LOCAL_CACHE}" \
  --task-name PlaceFood-rf \
  --required-agent-count 2 \
  --num-frames 33 \
  --train-window-stride 16 \
  --val-window-stride 32 \
  --val-set-proportion 0.1 \
  --split-seed 42 \
  --output-size 60 80 \
  --progress-every 25

for required in frames.f16 manifest.json COMPLETE stat-cmp.allowlist; do
  [[ -f "${LOCAL_CACHE}/${required}" && ! -L "${LOCAL_CACHE}/${required}" ]] || die "cache artifact missing: ${required}"
done

"${PYTHON_BIN}" "${LOCAL_REPO}/scripts/publish_metric_geometry_cache.py" \
  --source-root "${LOCAL_CACHE}" \
  --target-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --timeout-seconds 1800 \
  --poll-seconds 15

printf 'P13_METRIC_CACHE_COMPLETE run_id=%s revision=%s output=%s\n' \
  "${RUN_ID}" "${CODE_REVISION}" "${OUTPUT_ROOT}"
