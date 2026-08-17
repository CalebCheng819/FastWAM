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
DRIVER_ROOT="${FASTWAM_P13_DRIVER_ROOT:?FASTWAM_P13_DRIVER_ROOT is required}"
VULKAN_LOADER="${FASTWAM_P13_VULKAN_LOADER:?FASTWAM_P13_VULKAN_LOADER is required}"
LOCAL_ROOT="/tmp/fastwam-p13-metric-cache/${RUN_ID}"
LOCAL_REPO="${LOCAL_ROOT}/source"
PARTIAL_REPO="${LOCAL_REPO}.partial.${BASHPID}"
RUNTIME_ROOT="${LOCAL_ROOT}/runtime"
LOCAL_CACHE="${LOCAL_ROOT}/cache-output"
SCRATCH_ROOT="${LOCAL_ROOT}/scratch"
selected_profile=''

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || die "unsafe RUN_ID=${RUN_ID}"
[[ "${CODE_REVISION}" =~ ^[0-9a-f]{40}$ ]] || die "invalid Git revision"
[[ -x "${PYTHON_BIN}" ]] || die "Python is missing: ${PYTHON_BIN}"
[[ -f "${SOURCE_BUNDLE}" && ! -L "${SOURCE_BUNDLE}" ]] || die "source bundle is missing"
[[ -f "${RUNTIME_ARCHIVE}" && ! -L "${RUNTIME_ARCHIVE}" ]] || die "runtime archive is missing"
[[ -d "${DATASET_ROOT}" && ! -L "${DATASET_ROOT}" ]] || die "dataset root is missing"
[[ -d "${ROBOFACTORY_ROOT}" && ! -L "${ROBOFACTORY_ROOT}" ]] || die "RoboFactory root is missing"
[[ -f "${DRIVER_ROOT}/nvidia_icd.json" ]] || die "Vulkan ICD is missing"
[[ -f "${DRIVER_ROOT}/10_nvidia.json" ]] || die "EGL vendor file is missing"
[[ -f "${VULKAN_LOADER}" && ! -L "${VULKAN_LOADER}" ]] || die "Vulkan loader is missing"
[[ ! -e "${LOCAL_ROOT}" && ! -L "${LOCAL_ROOT}" ]] || die "node-local run root already exists: ${LOCAL_ROOT}"

mkdir -m 0700 -p -- \
  "${RUNTIME_ROOT}" "${SCRATCH_ROOT}/xdg-cache" "${SCRATCH_ROOT}/xdg-runtime" \
  "${SCRATCH_ROOT}/torch" "${SCRATCH_ROOT}/matplotlib" "${SCRATCH_ROOT}/tmp" \
  "${SCRATCH_ROOT}/pycache" "${SCRATCH_ROOT}/graphics-lib" \
  "${SCRATCH_ROOT}/graphics-probes"
ln -sf -- "${VULKAN_LOADER}" "${SCRATCH_ROOT}/graphics-lib/libvulkan.so.1"
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
export PYTHONPATH="${RUNTIME_ROOT}/site-packages:${LOCAL_REPO}/src:${LOCAL_REPO}/scripts:${ROBOFACTORY_ROOT%/robofactory}:${ROBOFACTORY_ROOT}"
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
  NVIDIA_DRIVER_CAPABILITIES
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

clear_explicit_graphics_selection() {
  unset VK_ICD_FILENAMES VK_DRIVER_FILES
  unset __EGL_VENDOR_LIBRARY_FILENAMES __EGL_VENDOR_LIBRARY_DIRS
  unset __GLX_VENDOR_LIBRARY_NAME SAPIEN_VULKAN_LIBRARY_PATH
  unset FASTWAM_GL_SHIM_ROOT LIBGL_DRIVERS_PATH GBM_BACKEND
}

apply_headless_contract() {
  export MUJOCO_GL=egl
  export EGL_PLATFORM=surfaceless
  export PYOPENGL_PLATFORM=egl
  export NVIDIA_DRIVER_CAPABILITIES=all
}

build_discovered_loader() {
  local candidate resolved entry
  local -a loader=()
  declare -A seen=()
  for candidate in \
    /usr/local/nvidia/lib64 /usr/local/nvidia/lib /usr/local/cuda/compat \
    /usr/local/cuda/lib64 /usr/local/cuda-12.8/lib64 \
    /usr/lib/x86_64-linux-gnu /usr/lib64 /lib/x86_64-linux-gnu; do
    [[ -d "${candidate}" && ! -L "${candidate}" ]] || continue
    resolved=$(readlink -f -- "${candidate}")
    if [[ -z "${seen[$resolved]:-}" ]]; then
      loader+=("${resolved}")
      seen["${resolved}"]=1
    fi
  done
  IFS=:
  entry="${loader[*]}"
  unset IFS
  [[ -n "${entry}" ]] || return 1
  export LD_LIBRARY_PATH="${entry}"
}

prepend_library_paths() {
  local path current
  current="${LD_LIBRARY_PATH:-}"
  for path in "$@"; do
    [[ -d "${path}" ]] || continue
    if [[ -z "${current}" ]]; then
      current="${path}"
    else
      current="${path}:${current}"
    fi
  done
  export LD_LIBRARY_PATH="${current}"
}

apply_cpfs_loader() {
  build_discovered_loader
  prepend_library_paths \
    "${SCRATCH_ROOT}/graphics-lib" \
    "${DRIVER_ROOT}/driver-lib" "${DRIVER_ROOT}/lib64" "${DRIVER_ROOT}/lib"
}

first_regular_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "${candidate}" && ! -L "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

apply_graphics_profile() {
  local profile=$1
  local vk_manifest egl_manifest
  restore_provider_graphics
  case "${profile}" in
    cpfs_manifest_headless)
      clear_explicit_graphics_selection
      apply_cpfs_loader
      export VK_ICD_FILENAMES="${DRIVER_ROOT}/nvidia_icd.json"
      export VK_DRIVER_FILES="${DRIVER_ROOT}/nvidia_icd.json"
      export __EGL_VENDOR_LIBRARY_FILENAMES="${DRIVER_ROOT}/10_nvidia.json"
      export __GLX_VENDOR_LIBRARY_NAME=nvidia
      export SAPIEN_VULKAN_LIBRARY_PATH="${SCRATCH_ROOT}/graphics-lib/libvulkan.so.1"
      ;;
    provider_native_headless)
      ;;
    provider_clean_headless)
      clear_explicit_graphics_selection
      ;;
    system_default_headless)
      clear_explicit_graphics_selection
      unset LD_LIBRARY_PATH
      ;;
    system_discovered_headless)
      clear_explicit_graphics_selection
      build_discovered_loader
      ;;
    system_manifest_headless)
      clear_explicit_graphics_selection
      build_discovered_loader
      vk_manifest=$(first_regular_file /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json) || return 2
      egl_manifest=$(first_regular_file /usr/share/glvnd/egl_vendor.d/10_nvidia.json /etc/glvnd/egl_vendor.d/10_nvidia.json) || return 2
      export VK_ICD_FILENAMES="${vk_manifest}"
      export VK_DRIVER_FILES="${vk_manifest}"
      export __EGL_VENDOR_LIBRARY_FILENAMES="${egl_manifest}"
      export __GLX_VENDOR_LIBRARY_NAME=nvidia
      ;;
    system_discovered_sapien_loader)
      clear_explicit_graphics_selection
      build_discovered_loader
      export SAPIEN_VULKAN_LIBRARY_PATH="${VULKAN_LOADER}"
      ;;
    *)
      return 2
      ;;
  esac
  apply_headless_contract
}

ensure_sapien_egl_contract() {
  if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" && -z "${__EGL_VENDOR_LIBRARY_DIRS:-}" ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="${DRIVER_ROOT}/10_nvidia.json"
  fi
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
profiles=(
  cpfs_manifest_headless
  provider_native_headless provider_clean_headless system_default_headless
  system_discovered_headless system_manifest_headless
  system_discovered_sapien_loader
)
for profile in "${profiles[@]}"; do
  probe_log="${SCRATCH_ROOT}/graphics-probes/${profile}.log"
  if ! apply_graphics_profile "${profile}"; then
    printf 'P13_METRIC_CACHE_GRAPHICS_PROFILE_SKIPPED profile=%s reason=unavailable\n' "${profile}"
    continue
  fi
  ensure_sapien_egl_contract
  set +e
  timeout --signal=TERM --kill-after=30s 180s env CUDA_VISIBLE_DEVICES=0 \
    "${PYTHON_BIN}" -B -c "${probe_program}" >"${probe_log}" 2>&1
  probe_rc=$?
  set -e
  if [[ "${probe_rc}" == 0 ]]; then
    selected_profile="${profile}"
    printf 'P13_METRIC_CACHE_GRAPHICS_PROFILE_SELECTED profile=%s\n' "${profile}"
    break
  fi
  printf 'P13_METRIC_CACHE_GRAPHICS_PROFILE_REJECTED profile=%s rc=%s\n' \
    "${profile}" "${probe_rc}" >&2
  tail -n 80 -- "${probe_log}" >&2 || true
done
[[ -n "${selected_profile}" ]] || die 'no GPU graphics profile could construct and close PlaceFood-rf'
apply_graphics_profile "${selected_profile}" || die 'selected graphics profile became unavailable'
ensure_sapien_egl_contract
printf 'P13_METRIC_CACHE_GRAPHICS_PREFLIGHT_PASS profile=%s\n' "${selected_profile}"

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
