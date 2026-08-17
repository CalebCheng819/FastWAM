#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

EXPERIMENT_REL="${FASTWAM_RUNTIME_EXPERIMENT_REL:-.research-workflow/experiments/FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R21-20260817}"
RUNTIME_GENERATION="${FASTWAM_RUNTIME_GENERATION:-R21}"
[[ "${RUNTIME_GENERATION}" =~ ^R[0-9]+$ ]] || {
  printf 'invalid FASTWAM_RUNTIME_GENERATION: %s\n' "${RUNTIME_GENERATION}" >&2
  exit 1
}
export FASTWAM_RUNTIME_GENERATION

die() {
  printf 'GAU0_%s_EVAL_FATAL: %s\n' "${RUNTIME_GENERATION}" "$*" >&2
  exit 1
}

required_env=(
  FASTWAM_SOURCE_ROOT FASTWAM_SOURCE_COMMIT FASTWAM_OUTPUT_ROOT
  FASTWAM_EXPERIMENT_ID FASTWAM_RUN_ID FASTWAM_CHECKPOINT
  FASTWAM_CHECKPOINT_SIZE_BYTES FASTWAM_PANEL FASTWAM_PANEL_SIZE_BYTES
  FASTWAM_GAU1_STATS FASTWAM_GAU1_STATS_SIZE_BYTES FASTWAM_GAU0_NATIVE_STATS
  FASTWAM_GAU0_NATIVE_STATS_SIZE_BYTES FASTWAM_DATASET_ROOT
  FASTWAM_ROBOFACTORY_ROOT FASTWAM_PYTHON_EXTRA_ROOT FASTWAM_CONTEXT_CACHE_DIR FASTWAM_CONTEXT_SIZE_BYTES
  FASTWAM_MODEL_CACHE_ROOT FASTWAM_NVIDIA_GRAPHICS_ROOT FASTWAM_PYTHON
  FASTWAM_EGL_FRONTEND FASTWAM_EGL_FRONTEND_SIZE_BYTES
  FASTWAM_GL_FRONTEND FASTWAM_GL_FRONTEND_SIZE_BYTES
  FASTWAM_GLES1_FRONTEND FASTWAM_GLES1_FRONTEND_SIZE_BYTES
  FASTWAM_GLES2_FRONTEND FASTWAM_GLES2_FRONTEND_SIZE_BYTES
  FASTWAM_OPENGL_FRONTEND FASTWAM_OPENGL_FRONTEND_SIZE_BYTES
  FASTWAM_GLX_FRONTEND FASTWAM_GLX_FRONTEND_SIZE_BYTES FASTWAM_EGL_DISPATCH
  FASTWAM_EGL_DISPATCH_SIZE_BYTES FASTWAM_EGL_VENDOR FASTWAM_EGL_VENDOR_SIZE_BYTES
  FASTWAM_VULKAN_LOADER FASTWAM_VULKAN_LOADER_SIZE_BYTES
  FASTWAM_PYTHON_TARGET FASTWAM_PYTHON_VERSION FASTWAM_PYTHON_CACHE_TAG FASTWAM_PYTHON_SOABI
  FASTWAM_BASELINE_ROOT FASTWAM_TRAINING_SOURCE_COMMIT
  FASTWAM_TRAINING_JOB_ID FASTWAM_RESERVATION_PATH
)
for name in "${required_env[@]}"; do
  [[ -n "${!name:-}" ]] || die "missing environment variable ${name}"
done

controller="${FASTWAM_SOURCE_ROOT}/${EXPERIMENT_REL}/controller.py"
evaluator="${FASTWAM_SOURCE_ROOT}/experiments/robofactory/eval_robofactory_multi_robot.py"
aggregator="${FASTWAM_SOURCE_ROOT}/${EXPERIMENT_REL}/aggregate_results.py"

graphics_keys=(
  LD_LIBRARY_PATH VK_ICD_FILENAMES VK_DRIVER_FILES
  __EGL_VENDOR_LIBRARY_FILENAMES __EGL_VENDOR_LIBRARY_DIRS __GLX_VENDOR_LIBRARY_NAME
  SAPIEN_VULKAN_LIBRARY_PATH FASTWAM_GL_SHIM_ROOT LIBGL_DRIVERS_PATH GBM_BACKEND
  MUJOCO_GL EGL_PLATFORM PYOPENGL_PLATFORM NVIDIA_DRIVER_CAPABILITIES
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

scratch_root="$(mktemp -d /tmp/fastwam-gau0-placefood-r21.XXXXXXXX)"
cleanup() {
  rm -rf -- "${scratch_root}"
}
trap cleanup EXIT
ulimit -c 0

mkdir -m 0700 -- "${scratch_root}/xdg-cache" "${scratch_root}/xdg-runtime" \
  "${scratch_root}/torch" "${scratch_root}/matplotlib" "${scratch_root}/tmp" \
  "${scratch_root}/pycache" "${scratch_root}/graphics-probes"

scratch_egl_manifest="${scratch_root}/10_nvidia.json"
"${FASTWAM_PYTHON}" -B - "${scratch_egl_manifest}" "${FASTWAM_EGL_VENDOR}" <<'PY'
import json
import os
import stat
import sys
from pathlib import Path

output = Path(sys.argv[1])
vendor = Path(sys.argv[2])
vendor_info = vendor.lstat()
if vendor.is_symlink() or not stat.S_ISREG(vendor_info.st_mode):
    raise SystemExit(f"EGL vendor library must be a non-symlink ordinary file: {vendor}")
with output.open("x", encoding="utf-8") as handle:
    json.dump(
        {"file_format_version": "1.0.0", "ICD": {"library_path": str(vendor)}},
        handle,
        separators=(",", ":"),
    )
    handle.write("\n")
os.chmod(output, 0o600)
PY

export PYTHONPATH="${FASTWAM_ROBOFACTORY_ROOT}:${FASTWAM_SOURCE_ROOT}/src:${FASTWAM_PYTHON_EXTRA_ROOT}:${FASTWAM_SOURCE_ROOT}/experiments/robofactory"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1
export PYTHONPYCACHEPREFIX="${scratch_root}/pycache"
export WANDB_MODE=offline
export XDG_CACHE_HOME="${scratch_root}/xdg-cache"
export XDG_RUNTIME_DIR="${scratch_root}/xdg-runtime"
export TORCH_HOME="${scratch_root}/torch"
export MPLCONFIGDIR="${scratch_root}/matplotlib"
export TMPDIR="${scratch_root}/tmp"

"${FASTWAM_PYTHON}" -B "${controller}" worker-preflight

gpu_count="$("${FASTWAM_PYTHON}" -B -c 'import torch; print(torch.cuda.device_count())')"
[[ "${gpu_count}" == '8' ]] || die "expected exactly 8 visible GPUs, observed ${gpu_count}"

restore_provider_graphics() {
  local key
  for key in "${graphics_keys[@]}"; do
    if [[ "${provider_graphics_present[${key}]}" == 1 ]]; then
      export "${key}=${provider_graphics_value[${key}]}"
    else
      unset "${key}"
    fi
  done
}

clear_explicit_graphics_selection() {
  unset VK_ICD_FILENAMES VK_DRIVER_FILES
  unset __EGL_VENDOR_LIBRARY_FILENAMES __EGL_VENDOR_LIBRARY_DIRS __GLX_VENDOR_LIBRARY_NAME
  unset SAPIEN_VULKAN_LIBRARY_PATH FASTWAM_GL_SHIM_ROOT LIBGL_DRIVERS_PATH GBM_BACKEND
}

apply_headless_contract() {
  export MUJOCO_GL=egl
  export EGL_PLATFORM=surfaceless
  export PYOPENGL_PLATFORM=egl
  export NVIDIA_DRIVER_CAPABILITIES=all
}

apply_sapien_egl_guard() {
  if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" && -z "${__EGL_VENDOR_LIBRARY_DIRS:-}" ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="${scratch_egl_manifest}"
  fi
}

build_discovered_loader() {
  local candidate resolved entry
  local -a loader=()
  declare -A seen=()
  for candidate in \
    /usr/local/nvidia/lib64 \
    /usr/local/nvidia/lib \
    /usr/local/cuda/compat \
    /usr/local/cuda/lib64 \
    /usr/local/cuda-12.8/lib64 \
    /usr/lib/x86_64-linux-gnu \
    /usr/lib64 \
    /lib/x86_64-linux-gnu; do
    [[ -d "${candidate}" && ! -L "${candidate}" ]] || continue
    resolved="$(readlink -f -- "${candidate}")"
    if [[ -z "${seen[${resolved}]:-}" ]]; then
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
  local profile="$1"
  local vk_manifest egl_manifest
  restore_provider_graphics
  case "${profile}" in
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
      vk_manifest="$(first_regular_file /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json)" || return 2
      egl_manifest="$(first_regular_file /usr/share/glvnd/egl_vendor.d/10_nvidia.json /etc/glvnd/egl_vendor.d/10_nvidia.json)" || return 2
      export VK_ICD_FILENAMES="${vk_manifest}"
      export VK_DRIVER_FILES="${vk_manifest}"
      export __EGL_VENDOR_LIBRARY_FILENAMES="${egl_manifest}"
      export __GLX_VENDOR_LIBRARY_NAME=nvidia
      ;;
    system_discovered_sapien_loader)
      clear_explicit_graphics_selection
      build_discovered_loader
      [[ -f "${FASTWAM_VULKAN_LOADER}" && ! -L "${FASTWAM_VULKAN_LOADER}" ]] || return 2
      export SAPIEN_VULKAN_LIBRARY_PATH="${FASTWAM_VULKAN_LOADER}"
      ;;
    *)
      return 2
      ;;
  esac
  apply_sapien_egl_guard
  apply_headless_contract
}

probe_program=$'import os\nfrom pathlib import Path\nfrom eval_robofactory_multi_robot import _build_environment\nroot = Path(os.environ["FASTWAM_ROBOFACTORY_ROOT"])\nenvironment = _build_environment(root, "PlaceFood-rf")\nenvironment.close()\nprint("GAU0_" + os.environ["FASTWAM_RUNTIME_GENERATION"] + "_ENVIRONMENT_CONSTRUCTION_PROBE_PASS task=PlaceFood-rf device=0")\n'

profiles=(
  provider_native_headless
  provider_clean_headless
  system_default_headless
  system_discovered_headless
  system_manifest_headless
  system_discovered_sapien_loader
)
if [[ "${FASTWAM_REQUIRE_PROVIDER_NATIVE_GRAPHICS:-0}" == '1' ]]; then
  profiles=(provider_native_headless)
fi
selected_profile=''
for profile in "${profiles[@]}"; do
  probe_log="${scratch_root}/graphics-probes/${profile}.log"
  if ! apply_graphics_profile "${profile}"; then
    printf 'GAU0_%s_GRAPHICS_PROFILE_SKIPPED profile=%s reason=unavailable\n' "${RUNTIME_GENERATION}" "${profile}"
    continue
  fi
  set +e
  timeout --signal=TERM --kill-after=30s 180s env CUDA_VISIBLE_DEVICES=0 \
    "${FASTWAM_PYTHON}" -B -c "${probe_program}" >"${probe_log}" 2>&1
  probe_rc=$?
  set -e
  if [[ "${probe_rc}" == 0 ]]; then
    selected_profile="${profile}"
    printf 'GAU0_%s_GRAPHICS_PROFILE_SELECTED profile=%s\n' "${RUNTIME_GENERATION}" "${profile}"
    rm -- "${probe_log}"
    break
  fi
  printf 'GAU0_%s_GRAPHICS_PROFILE_REJECTED profile=%s rc=%s\n' "${RUNTIME_GENERATION}" "${profile}" "${probe_rc}" >&2
  tail -n 80 -- "${probe_log}" >&2 || true
done
[[ -n "${selected_profile}" ]] || die 'no graphics profile could construct and close PlaceFood-rf'
apply_graphics_profile "${selected_profile}" || die "selected graphics profile became unavailable: ${selected_profile}"

common_argv=(
  "${evaluator}"
  --mode fastwam
  --task PlaceFood-rf
  --panel "${FASTWAM_PANEL}"
  --integrity-mode metadata_no_hash
  --panel-size-bytes "${FASTWAM_PANEL_SIZE_BYTES}"
  --dataset-root "${FASTWAM_DATASET_ROOT}"
  --robofactory-root "${FASTWAM_ROBOFACTORY_ROOT}"
  --eval-code-commit "${FASTWAM_SOURCE_COMMIT}"
  --num-episodes 2
  --max-steps 300
  --exec-horizon 5
  --policy-seed 10000
  --checkpoint "${FASTWAM_CHECKPOINT}"
  --checkpoint-size-bytes "${FASTWAM_CHECKPOINT_SIZE_BYTES}"
  --context-cache-dir "${FASTWAM_CONTEXT_CACHE_DIR}"
  --context-size-bytes "${FASTWAM_CONTEXT_SIZE_BYTES}"
  --model-cache-root "${FASTWAM_MODEL_CACHE_ROOT}"
  --no-gaussian-conditioning
  --training-source-commit "${FASTWAM_TRAINING_SOURCE_COMMIT}"
  --training-job-id "${FASTWAM_TRAINING_JOB_ID}"
  --device cuda:0
  --action-horizon 32
  --num-inference-steps 20
)

run_arm() {
  local arm="$1"
  local stats="$2"
  local stats_size="$3"
  local stats_provenance_mode="$4"
  local arm_root="${scratch_root}/${arm}"
  local shard start end shard_name shard_runtime

  mkdir -m 0700 -- "${arm_root}"
  for shard in 0 1 2 3; do
    start=$((2 * shard))
    end=$((start + 1))
    shard_name="shard${shard}-episodes${start}-${end}"
    shard_runtime="${scratch_root}/shard-runtime/${arm}/shard-${shard}"
    mkdir -m 0700 -p -- \
      "${shard_runtime}/xdg-cache" "${shard_runtime}/xdg-runtime" \
      "${shard_runtime}/torch" "${shard_runtime}/matplotlib" \
      "${shard_runtime}/tmp" "${shard_runtime}/pycache"
    if ! (
      export CUDA_VISIBLE_DEVICES="${shard}"
      export XDG_CACHE_HOME="${shard_runtime}/xdg-cache"
      export XDG_RUNTIME_DIR="${shard_runtime}/xdg-runtime"
      export TORCH_HOME="${shard_runtime}/torch"
      export MPLCONFIGDIR="${shard_runtime}/matplotlib"
      export TMPDIR="${shard_runtime}/tmp"
      export PYTHONPYCACHEPREFIX="${shard_runtime}/pycache"
      exec "${FASTWAM_PYTHON}" -B "${common_argv[@]}" \
        --episode-start "${start}" \
        --output-dir "${arm_root}/${shard_name}" \
        --stats "${stats}" \
        --stats-size-bytes "${stats_size}" \
        --stats-provenance-mode "${stats_provenance_mode}"
    ) >"${arm_root}/${shard_name}.log" 2>&1; then
      printf '===== %s %s =====\n' "${arm}" "${shard_name}" >&2
      tail -n 120 -- "${arm_root}/${shard_name}.log" >&2 || true
      die "${arm} evaluator failed at ${shard_name}"
    fi
    rm -- "${arm_root}/${shard_name}.log"
  done
}

run_arm gau1_stats "${FASTWAM_GAU1_STATS}" "${FASTWAM_GAU1_STATS_SIZE_BYTES}" train_split
run_arm gau0_native_stats "${FASTWAM_GAU0_NATIVE_STATS}" "${FASTWAM_GAU0_NATIVE_STATS_SIZE_BYTES}" legacy_full_dataset

platform_job_id="$("${FASTWAM_PYTHON}" -B "${controller}" job-id)"
"${FASTWAM_PYTHON}" -B "${aggregator}" \
  --temp-root "${scratch_root}" \
  --baseline-root "${FASTWAM_BASELINE_ROOT}" \
  --output-root "${FASTWAM_OUTPUT_ROOT}" \
  --source-commit "${FASTWAM_SOURCE_COMMIT}" \
  --job-id "${platform_job_id}"

"${FASTWAM_PYTHON}" -B "${controller}" validate-terminal --member gau0
printf 'GAU0_%s_EVAL_SCIENTIFIC_COMPLETE\n' "${RUNTIME_GENERATION}"
