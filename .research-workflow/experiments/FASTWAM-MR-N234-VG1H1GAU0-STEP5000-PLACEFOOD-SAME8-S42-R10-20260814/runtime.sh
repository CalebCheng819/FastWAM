#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

EXPERIMENT_REL="${FASTWAM_EXPERIMENT_REL_OVERRIDE:-.research-workflow/experiments/FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R10-20260814}"

die() {
  printf 'GAU0_EVAL_FATAL: %s\n' "$*" >&2
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

scratch_root="$(mktemp -d "${FASTWAM_SCRATCH_TEMPLATE_OVERRIDE:-/tmp/fastwam-gau0-placefood-r10.XXXXXXXX}")"
cleanup() {
  rm -rf -- "${scratch_root}"
}
trap cleanup EXIT

mkdir -m 0700 -- "${scratch_root}/xdg-cache" "${scratch_root}/xdg-runtime" \
  "${scratch_root}/torch" "${scratch_root}/matplotlib" "${scratch_root}/tmp" \
  "${scratch_root}/glvnd-runtime"

export PYTHONPATH="${FASTWAM_ROBOFACTORY_ROOT}:${FASTWAM_SOURCE_ROOT}/src:${FASTWAM_PYTHON_EXTRA_ROOT}:${FASTWAM_SOURCE_ROOT}/experiments/robofactory${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1
export PYTHONPYCACHEPREFIX="${scratch_root}/pycache"
export WANDB_MODE=offline
export MUJOCO_GL=egl
export EGL_PLATFORM=surfaceless
export PYOPENGL_PLATFORM=egl
export NVIDIA_DRIVER_CAPABILITIES=all
export XDG_CACHE_HOME="${scratch_root}/xdg-cache"
export XDG_RUNTIME_DIR="${scratch_root}/xdg-runtime"
export TORCH_HOME="${scratch_root}/torch"
export MPLCONFIGDIR="${scratch_root}/matplotlib"
export TMPDIR="${scratch_root}/tmp"
export VK_ICD_FILENAMES="${FASTWAM_NVIDIA_GRAPHICS_ROOT}/nvidia_icd.json"
export VK_DRIVER_FILES="${FASTWAM_NVIDIA_GRAPHICS_ROOT}/nvidia_icd.json"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __EGL_VENDOR_LIBRARY_FILENAMES="${FASTWAM_NVIDIA_GRAPHICS_ROOT}/10_nvidia.json"

require_runtime_file() {
  local path="$1"
  local expected_bytes="$2"
  [[ -f "${path}" && ! -L "${path}" ]] || die "GLVND runtime input is not an ordinary non-link file: ${path}"
  [[ "$(stat -c '%s' -- "${path}")" == "${expected_bytes}" ]] || \
    die "GLVND runtime input size mismatch: ${path}"
}

require_runtime_file "${FASTWAM_EGL_FRONTEND}" "${FASTWAM_EGL_FRONTEND_SIZE_BYTES}"
require_runtime_file "${FASTWAM_GL_FRONTEND}" "${FASTWAM_GL_FRONTEND_SIZE_BYTES}"
require_runtime_file "${FASTWAM_GLES1_FRONTEND}" "${FASTWAM_GLES1_FRONTEND_SIZE_BYTES}"
require_runtime_file "${FASTWAM_GLES2_FRONTEND}" "${FASTWAM_GLES2_FRONTEND_SIZE_BYTES}"
require_runtime_file "${FASTWAM_OPENGL_FRONTEND}" "${FASTWAM_OPENGL_FRONTEND_SIZE_BYTES}"
require_runtime_file "${FASTWAM_GLX_FRONTEND}" "${FASTWAM_GLX_FRONTEND_SIZE_BYTES}"
require_runtime_file "${FASTWAM_EGL_DISPATCH}" "${FASTWAM_EGL_DISPATCH_SIZE_BYTES}"
require_runtime_file "${FASTWAM_EGL_VENDOR}" "${FASTWAM_EGL_VENDOR_SIZE_BYTES}"
require_runtime_file "${FASTWAM_VULKAN_LOADER}" "${FASTWAM_VULKAN_LOADER_SIZE_BYTES}"
ln -s -- "${FASTWAM_EGL_FRONTEND}" "${scratch_root}/glvnd-runtime/libEGL.so.1"
ln -s -- "${FASTWAM_GL_FRONTEND}" "${scratch_root}/glvnd-runtime/libGL.so.1"
ln -s -- "${FASTWAM_GLES1_FRONTEND}" "${scratch_root}/glvnd-runtime/libGLESv1_CM.so.1"
ln -s -- "${FASTWAM_GLES2_FRONTEND}" "${scratch_root}/glvnd-runtime/libGLESv2.so.2"
ln -s -- "${FASTWAM_OPENGL_FRONTEND}" "${scratch_root}/glvnd-runtime/libOpenGL.so.0"
ln -s -- "${FASTWAM_GLX_FRONTEND}" "${scratch_root}/glvnd-runtime/libGLX.so.0"
ln -s -- 'libEGL.so.1' "${scratch_root}/glvnd-runtime/libEGL.so"
ln -s -- 'libGL.so.1' "${scratch_root}/glvnd-runtime/libGL.so"
ln -s -- 'libGLESv1_CM.so.1' "${scratch_root}/glvnd-runtime/libGLESv1_CM.so"
ln -s -- 'libGLESv2.so.2' "${scratch_root}/glvnd-runtime/libGLESv2.so"
ln -s -- "${FASTWAM_VULKAN_LOADER}" "${scratch_root}/glvnd-runtime/libvulkan.so.1"
ln -s -- 'libvulkan.so.1' "${scratch_root}/glvnd-runtime/libvulkan.so"
export LD_LIBRARY_PATH="${scratch_root}/glvnd-runtime:${FASTWAM_NVIDIA_GRAPHICS_ROOT}/lib:${FASTWAM_NVIDIA_GRAPHICS_ROOT}/driver-lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export SAPIEN_VULKAN_LIBRARY_PATH="${scratch_root}/glvnd-runtime/libvulkan.so.1"

"${FASTWAM_PYTHON}" -B -c '
import ctypes
import os
from pathlib import Path

from eval_robofactory_multi_robot import _preflight_environment_imports

expected = {
    "libEGL.so.1": Path(os.environ["FASTWAM_EGL_FRONTEND"]).resolve(strict=True),
    "libGL.so.1": Path(os.environ["FASTWAM_GL_FRONTEND"]).resolve(strict=True),
    "libGLESv1_CM.so.1": Path(os.environ["FASTWAM_GLES1_FRONTEND"]).resolve(strict=True),
    "libGLESv2.so.2": Path(os.environ["FASTWAM_GLES2_FRONTEND"]).resolve(strict=True),
    "libOpenGL.so.0": Path(os.environ["FASTWAM_OPENGL_FRONTEND"]).resolve(strict=True),
    "libGLX.so.0": Path(os.environ["FASTWAM_GLX_FRONTEND"]).resolve(strict=True),
    "libvulkan.so.1": Path(os.environ["FASTWAM_VULKAN_LOADER"]).resolve(strict=True),
}
vendor = Path(os.environ["FASTWAM_EGL_VENDOR"]).resolve(strict=True)
shim_root = Path(os.environ["LD_LIBRARY_PATH"].split(os.pathsep)[0]).resolve(strict=True)
for soname, target in expected.items():
    actual = (shim_root / soname).resolve(strict=True)
    if actual != target:
        raise SystemExit(f"GLVND shim mismatch for {soname}: {actual} != {target}")
frontend_handle = ctypes.CDLL(str(shim_root / "libEGL.so.1"), mode=ctypes.RTLD_GLOBAL)
if not callable(getattr(frontend_handle, "eglQueryString", None)):
    raise SystemExit("controlled EGL frontend lacks eglQueryString")
ctypes.CDLL(str(vendor), mode=ctypes.RTLD_GLOBAL)
vulkan = ctypes.CDLL(str(shim_root / "libvulkan.so.1"), mode=ctypes.RTLD_GLOBAL)
enumerate_version = vulkan.vkEnumerateInstanceVersion
enumerate_version.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
enumerate_version.restype = ctypes.c_int32
version = ctypes.c_uint32()
result = enumerate_version(ctypes.byref(version))
if result != 0:
    raise SystemExit(f"vkEnumerateInstanceVersion failed: {result}")
modules = _preflight_environment_imports(Path(os.environ["FASTWAM_ROBOFACTORY_ROOT"]))
print(f"GAU0_GLVND_RUNTIME_PREFLIGHT_PASS shim={shim_root} vendor={vendor} modules={modules}")
'

"${FASTWAM_PYTHON}" -B "${controller}" worker-preflight

"${FASTWAM_PYTHON}" -B -c '
import os
from pathlib import Path

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
from eval_robofactory_multi_robot import _preflight_environment_imports

source_root = Path(os.environ["FASTWAM_SOURCE_ROOT"]).resolve(strict=True)
robofactory_root = Path(os.environ["FASTWAM_ROBOFACTORY_ROOT"]).resolve(strict=True)
expected_src = source_root / "src"
pythonpath = [Path(item).resolve(strict=True) for item in os.environ["PYTHONPATH"].split(os.pathsep)]
expected_path = [
    robofactory_root,
    expected_src,
    Path(os.environ["FASTWAM_PYTHON_EXTRA_ROOT"]).resolve(strict=True),
    (source_root / "experiments" / "robofactory").resolve(strict=True),
]
if pythonpath[:4] != expected_path:
    raise SystemExit(f"worker PYTHONPATH prefix mismatch: {pythonpath[:4]} != {expected_path}")
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
print(f"GAU0_FROZEN_RUNTIME_IMPORT_PASS path={actual_modules}")
'

gpu_count="$(${FASTWAM_PYTHON} -B -c 'import torch; print(torch.cuda.device_count())')"
[[ "${gpu_count}" == '8' ]] || die "expected exactly 8 visible GPUs, observed ${gpu_count}"

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
  local arm_root="${scratch_root}/${arm}"
  local shard start end shard_name shard_runtime pid failure=0
  local -a pids=()

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
    (
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
        --stats-size-bytes "${stats_size}"
    ) >"${arm_root}/${shard_name}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failure=1
    fi
  done
  if (( failure != 0 )); then
    for shard in 0 1 2 3; do
      start=$((2 * shard))
      end=$((start + 1))
      shard_name="shard${shard}-episodes${start}-${end}"
      printf '===== %s %s =====\n' "${arm}" "${shard_name}" >&2
      tail -n 120 -- "${arm_root}/${shard_name}.log" >&2 || true
    done
    die "one or more ${arm} evaluator processes failed"
  fi
  rm -- "${arm_root}"/*.log
}

run_arm gau1_stats "${FASTWAM_GAU1_STATS}" "${FASTWAM_GAU1_STATS_SIZE_BYTES}"
run_arm gau0_native_stats "${FASTWAM_GAU0_NATIVE_STATS}" "${FASTWAM_GAU0_NATIVE_STATS_SIZE_BYTES}"

platform_job_id="$("${FASTWAM_PYTHON}" -B "${controller}" job-id)"
"${FASTWAM_PYTHON}" -B "${aggregator}" \
  --temp-root "${scratch_root}" \
  --baseline-root "${FASTWAM_BASELINE_ROOT}" \
  --output-root "${FASTWAM_OUTPUT_ROOT}" \
  --source-commit "${FASTWAM_SOURCE_COMMIT}" \
  --job-id "${platform_job_id}"

"${FASTWAM_PYTHON}" -B "${controller}" validate-terminal --member gau0
printf 'GAU0_EVAL_SCIENTIFIC_COMPLETE\n'
