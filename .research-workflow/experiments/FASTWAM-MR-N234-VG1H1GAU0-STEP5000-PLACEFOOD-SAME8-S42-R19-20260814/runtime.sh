#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

EXPERIMENT_REL='.research-workflow/experiments/FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R19-20260814'

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

scratch_root="$(mktemp -d /tmp/fastwam-gau0-placefood-r19.XXXXXXXX)"
cleanup() {
  rm -rf -- "${scratch_root}"
}
trap cleanup EXIT

mkdir -m 0700 -- "${scratch_root}/xdg-cache" "${scratch_root}/xdg-runtime" \
  "${scratch_root}/torch" "${scratch_root}/matplotlib" "${scratch_root}/tmp" \
  "${scratch_root}/pycache" "${scratch_root}/glvnd-runtime"

declare -A glvnd_targets=(
  [libEGL.so]="${FASTWAM_EGL_FRONTEND}"
  [libEGL.so.1]="${FASTWAM_EGL_FRONTEND}"
  [libGL.so]="${FASTWAM_GL_FRONTEND}"
  [libGL.so.1]="${FASTWAM_GL_FRONTEND}"
  [libGLESv1_CM.so]="${FASTWAM_GLES1_FRONTEND}"
  [libGLESv1_CM.so.1]="${FASTWAM_GLES1_FRONTEND}"
  [libGLESv2.so]="${FASTWAM_GLES2_FRONTEND}"
  [libGLESv2.so.2]="${FASTWAM_GLES2_FRONTEND}"
  [libGLX.so]="${FASTWAM_GLX_FRONTEND}"
  [libGLX.so.0]="${FASTWAM_GLX_FRONTEND}"
  [libOpenGL.so]="${FASTWAM_OPENGL_FRONTEND}"
  [libOpenGL.so.0]="${FASTWAM_OPENGL_FRONTEND}"
)
for name in "${!glvnd_targets[@]}"; do
  target="${glvnd_targets[${name}]}"
  [[ -f "${target}" ]] || die "frozen GLVND frontend is not a regular file: ${target}"
  [[ ! -L "${target}" ]] || die "frozen GLVND frontend must not be a symlink: ${target}"
  ln -s -- "${target}" "${scratch_root}/glvnd-runtime/${name}"
done

export PYTHONPATH="${FASTWAM_ROBOFACTORY_ROOT}:${FASTWAM_SOURCE_ROOT}/src:${FASTWAM_PYTHON_EXTRA_ROOT}:${FASTWAM_SOURCE_ROOT}/experiments/robofactory"
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
export FASTWAM_GL_SHIM_ROOT="${scratch_root}/glvnd-runtime"
export LD_LIBRARY_PATH="${FASTWAM_GL_SHIM_ROOT}:${FASTWAM_NVIDIA_GRAPHICS_ROOT}/lib:${FASTWAM_NVIDIA_GRAPHICS_ROOT}/driver-lib:/usr/local/cuda-12.8/lib64"
unset SAPIEN_VULKAN_LIBRARY_PATH

"${FASTWAM_PYTHON}" -B -c '
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
loader = [Path(item).resolve(strict=False) for item in os.environ["LD_LIBRARY_PATH"].split(os.pathsep)]
if loader != expected_loader:
    raise SystemExit(f"complete GLVND loader namespace mismatch: {loader} != {expected_loader}")
expected_pythonpath = [
    Path(os.environ["FASTWAM_ROBOFACTORY_ROOT"]).resolve(strict=True),
    (Path(os.environ["FASTWAM_SOURCE_ROOT"]) / "src").resolve(strict=True),
    Path(os.environ["FASTWAM_PYTHON_EXTRA_ROOT"]).resolve(strict=True),
    (Path(os.environ["FASTWAM_SOURCE_ROOT"]) / "experiments" / "robofactory").resolve(strict=True),
]
pythonpath = [Path(item).resolve(strict=True) for item in os.environ["PYTHONPATH"].split(os.pathsep)]
if pythonpath != expected_pythonpath:
    raise SystemExit(f"worker PYTHONPATH mismatch: {pythonpath} != {expected_pythonpath}")

from OpenGL import EGL
if not callable(getattr(EGL, "eglQueryString", None)) or not callable(getattr(EGL, "eglGetDisplay", None)):
    raise SystemExit("PyOpenGL EGL frontend is incomplete")

from eval_robofactory_multi_robot import _preflight_environment_imports
modules = _preflight_environment_imports(Path(os.environ["FASTWAM_ROBOFACTORY_ROOT"]))
print(f"GAU0_COMPLETE_GLVND_RUNTIME_PREFLIGHT_PASS loader={loader} modules={modules}")
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
if pythonpath != expected_path:
    raise SystemExit(f"worker PYTHONPATH mismatch: {pythonpath} != {expected_path}")
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

gpu_count="$("${FASTWAM_PYTHON}" -B -c 'import torch; print(torch.cuda.device_count())')"
[[ "${gpu_count}" == '8' ]] || die "expected exactly 8 visible GPUs, observed ${gpu_count}"

CUDA_VISIBLE_DEVICES=0 "${FASTWAM_PYTHON}" -B -c '
import os
from pathlib import Path

from eval_robofactory_multi_robot import _build_environment

root = Path(os.environ["FASTWAM_ROBOFACTORY_ROOT"])
environment = _build_environment(root, "PlaceFood-rf")
environment.close()
print("GAU0_ENVIRONMENT_CONSTRUCTION_PREFLIGHT_PASS task=PlaceFood-rf device=0")
'

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
        --stats-size-bytes "${stats_size}"
    ) >"${arm_root}/${shard_name}.log" 2>&1; then
      printf '===== %s %s =====\n' "${arm}" "${shard_name}" >&2
      tail -n 120 -- "${arm_root}/${shard_name}.log" >&2 || true
      die "${arm} evaluator failed at ${shard_name}"
    fi
    rm -- "${arm_root}/${shard_name}.log"
  done
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
