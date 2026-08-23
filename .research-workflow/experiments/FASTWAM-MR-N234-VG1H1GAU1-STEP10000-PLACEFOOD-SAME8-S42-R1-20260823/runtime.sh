#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

EXPERIMENT_REL='.research-workflow/experiments/FASTWAM-MR-N234-VG1H1GAU1-STEP10000-PLACEFOOD-SAME8-S42-R1-20260823'

die() {
  printf 'STEP10K_EVAL_FATAL: %s\n' "$*" >&2
  exit 1
}

required_env=(
  FASTWAM_SOURCE_ROOT FASTWAM_SOURCE_COMMIT FASTWAM_OUTPUT_ROOT
  FASTWAM_EXPERIMENT_ID FASTWAM_RUN_ID FASTWAM_ATTEMPT_ID FASTWAM_CHECKPOINT
  FASTWAM_CHECKPOINT_SIZE_BYTES FASTWAM_PANEL FASTWAM_PANEL_SIZE_BYTES
  FASTWAM_STATS FASTWAM_STATS_SIZE_BYTES FASTWAM_DATASET_ROOT
  FASTWAM_ROBOFACTORY_ROOT FASTWAM_CONTEXT_CACHE_DIR FASTWAM_CONTEXT_SIZE_BYTES
  FASTWAM_MODEL_CACHE_ROOT FASTWAM_POLICY_LIGHTNING_ROOT
  FASTWAM_POLICY_LIGHTNING_COMMIT FASTWAM_NOPOSPLAT_CHECKPOINT
  FASTWAM_NOPOSPLAT_CHECKPOINT_SIZE_BYTES FASTWAM_NVIDIA_GRAPHICS_ROOT
  FASTWAM_PYTHON FASTWAM_TRAINING_SOURCE_COMMIT FASTWAM_TRAINING_JOB_ID
)
for name in "${required_env[@]}"; do
  [[ -n "${!name:-}" ]] || die "missing environment variable ${name}"
done

[[ "${FASTWAM_EXPERIMENT_ID}" == 'FASTWAM-MR-N234-VG1H1GAU1-STEP10000-PLACEFOOD-SAME8-S42-R1-20260823' ]] || die 'experiment identity drift'
[[ "${FASTWAM_RUN_ID}" == 'fastwam-gau1-step10k-placefood-same8-r3-20260823' ]] || die 'run identity drift'
[[ "${FASTWAM_ATTEMPT_ID}" == 'attempt-003' ]] || die 'attempt identity drift'
[[ "${FASTWAM_CHECKPOINT}" == */step_010000.pt ]] || die 'checkpoint is not step_010000.pt'
[[ "${FASTWAM_CHECKPOINT_SIZE_BYTES}" == '12047213657' ]] || die 'checkpoint byte-size drift'
[[ "${FASTWAM_OUTPUT_ROOT}" == '/oss-chengjuntao/artifacts/fastwam-gau1-step10k-placefood-same8-eval-20260823-r3' ]] || die 'output root drift'
[[ ! -e "${FASTWAM_OUTPUT_ROOT}" && ! -L "${FASTWAM_OUTPUT_ROOT}" ]] || die 'output root already exists'

evaluator="${FASTWAM_SOURCE_ROOT}/experiments/robofactory/eval_robofactory_multi_robot.py"
aggregator="${FASTWAM_SOURCE_ROOT}/${EXPERIMENT_REL}/aggregate_results.py"
source_src="${FASTWAM_SOURCE_ROOT}/src"
[[ -f "${evaluator}" && ! -L "${evaluator}" ]] || die 'evaluator is missing or unsafe'
[[ -f "${aggregator}" && ! -L "${aggregator}" ]] || die 'aggregator is missing or unsafe'
[[ -d "${source_src}" && ! -L "${source_src}" ]] || die 'bundle source directory is missing or unsafe'
[[ -x "${FASTWAM_PYTHON}" ]] || die 'evaluation Python is not executable'

scratch_root="$(mktemp -d /tmp/fastwam-step10k-placefood-eval.XXXXXXXX)"
cleanup() { rm -rf -- "${scratch_root}"; }
trap cleanup EXIT
mkdir -m 0700 -- "${scratch_root}/shards" "${scratch_root}/logs" \
  "${scratch_root}/xdg-cache" "${scratch_root}/xdg-runtime" \
  "${scratch_root}/torch" "${scratch_root}/matplotlib" "${scratch_root}/tmp"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${source_src}"
export WANDB_MODE=offline
export MUJOCO_GL=egl
export XDG_CACHE_HOME="${scratch_root}/xdg-cache"
export XDG_RUNTIME_DIR="${scratch_root}/xdg-runtime"
export TORCH_HOME="${scratch_root}/torch"
export MPLCONFIGDIR="${scratch_root}/matplotlib"
export TMPDIR="${scratch_root}/tmp"
export VK_ICD_FILENAMES="${FASTWAM_NVIDIA_GRAPHICS_ROOT}/nvidia_icd.json"
export VK_DRIVER_FILES="${FASTWAM_NVIDIA_GRAPHICS_ROOT}/nvidia_icd.json"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __EGL_VENDOR_LIBRARY_FILENAMES="${FASTWAM_NVIDIA_GRAPHICS_ROOT}/10_nvidia.json"
export LD_LIBRARY_PATH="${FASTWAM_NVIDIA_GRAPHICS_ROOT}/lib:${LD_LIBRARY_PATH:-}"

"${FASTWAM_PYTHON}" -B - <<'PY' || die 'bundle Python source identity gate failed'
import os
from pathlib import Path

import fastwam
import fastwam.runtime as runtime

expected = (Path(os.environ["FASTWAM_SOURCE_ROOT"]) / "src").resolve(strict=True)
package_file = Path(fastwam.__file__).resolve(strict=True)
runtime_file = Path(runtime.__file__).resolve(strict=True)
for label, path in (("fastwam", package_file), ("fastwam.runtime", runtime_file)):
    try:
        path.relative_to(expected)
    except ValueError as error:
        raise SystemExit(f"{label} resolved outside bundle src: {path}") from error
factory = getattr(runtime, "create_multi_robot_fastwam", None)
if not callable(factory):
    raise SystemExit(f"bundle runtime lacks callable create_multi_robot_fastwam: {runtime_file}")
print(f"STEP10K_EVAL_SOURCE_GATE=PASS package={package_file} runtime={runtime_file}", flush=True)
PY

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
  --num-episodes 1
  --max-steps 300
  --exec-horizon 5
  --policy-seed 10000
  --checkpoint "${FASTWAM_CHECKPOINT}"
  --checkpoint-size-bytes "${FASTWAM_CHECKPOINT_SIZE_BYTES}"
  --stats "${FASTWAM_STATS}"
  --stats-size-bytes "${FASTWAM_STATS_SIZE_BYTES}"
  --stats-provenance-mode train_split
  --context-cache-dir "${FASTWAM_CONTEXT_CACHE_DIR}"
  --context-size-bytes "${FASTWAM_CONTEXT_SIZE_BYTES}"
  --model-cache-root "${FASTWAM_MODEL_CACHE_ROOT}"
  --policy-lightning-repo "${FASTWAM_POLICY_LIGHTNING_ROOT}"
  --policy-lightning-commit "${FASTWAM_POLICY_LIGHTNING_COMMIT}"
  --noposplat-checkpoint "${FASTWAM_NOPOSPLAT_CHECKPOINT}"
  --noposplat-checkpoint-size-bytes "${FASTWAM_NOPOSPLAT_CHECKPOINT_SIZE_BYTES}"
  --gaussian-conditioning
  --training-source-commit "${FASTWAM_TRAINING_SOURCE_COMMIT}"
  --training-job-id "${FASTWAM_TRAINING_JOB_ID}"
  --device cuda:0
  --action-horizon 32
  --num-inference-steps 20
)

declare -a pids=()
for index in 0 1 2 3 4 5 6 7; do
  shard="${scratch_root}/shards/episode-$(printf '%02d' "${index}")"
  log="${scratch_root}/logs/episode-$(printf '%02d' "${index}").log"
  CUDA_VISIBLE_DEVICES="${index}" PYTHONPATH="${source_src}" PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 "${FASTWAM_PYTHON}" -B "${common_argv[@]}" \
    --episode-start "${index}" --output-dir "${shard}" >"${log}" 2>&1 &
  pids+=("$!")
done

failure=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failure=1; fi
done
if (( failure != 0 )); then
  for index in 0 1 2 3 4 5 6 7; do
    log="${scratch_root}/logs/episode-$(printf '%02d' "${index}").log"
    printf '===== episode %s =====\n' "${index}" >&2
    tail -n 160 -- "${log}" >&2 || true
  done
  die 'one or more evaluator processes failed'
fi

platform_job_id="${PAI_JOB_ID:-${DLC_JOB_ID:-${JOB_ID:-unknown-at-runtime}}}"
"${FASTWAM_PYTHON}" -B "${aggregator}" \
  --temp-root "${scratch_root}/shards" \
  --output-root "${FASTWAM_OUTPUT_ROOT}" \
  --source-commit "${FASTWAM_SOURCE_COMMIT}" \
  --job-id "${platform_job_id}"
printf 'STEP10K_EVAL_SCIENTIFIC_COMPLETE\n'
