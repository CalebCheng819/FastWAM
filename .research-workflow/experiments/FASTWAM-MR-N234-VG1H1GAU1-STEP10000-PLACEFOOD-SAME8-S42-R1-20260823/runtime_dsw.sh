#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

EXPERIMENT_REL='.research-workflow/experiments/FASTWAM-MR-N234-VG1H1GAU1-STEP10000-PLACEFOOD-SAME8-S42-R1-20260823'
EXPERIMENT_ID='FASTWAM-MR-N234-VG1H1GAU1-STEP10000-PLACEFOOD-SAME8-S42-R1-20260823'
RUN_ID='fastwam-gau1-step10k-placefood-same8-dsw4-r4-20260823'

die() {
  printf 'STEP10K_DSW_EVAL_FATAL: %s\n' "$*" >&2
  exit 1
}

required_env=(
  FASTWAM_EVAL_SCOPE FASTWAM_SOURCE_ROOT FASTWAM_SOURCE_COMMIT
  FASTWAM_OUTPUT_ROOT FASTWAM_CONTROL_ROOT FASTWAM_EXPERIMENT_ID FASTWAM_RUN_ID
  FASTWAM_ATTEMPT_ID FASTWAM_CHECKPOINT FASTWAM_CHECKPOINT_SIZE_BYTES
  FASTWAM_PANEL FASTWAM_PANEL_SIZE_BYTES FASTWAM_STATS FASTWAM_STATS_SIZE_BYTES
  FASTWAM_DATASET_ROOT FASTWAM_ROBOFACTORY_ROOT FASTWAM_CONTEXT_CACHE_DIR
  FASTWAM_CONTEXT_SIZE_BYTES FASTWAM_MODEL_CACHE_ROOT FASTWAM_POLICY_LIGHTNING_ROOT
  FASTWAM_POLICY_LIGHTNING_COMMIT FASTWAM_NOPOSPLAT_CHECKPOINT
  FASTWAM_NOPOSPLAT_CHECKPOINT_SIZE_BYTES FASTWAM_NVIDIA_GRAPHICS_ROOT
  FASTWAM_PYTHON FASTWAM_TRAINING_SOURCE_COMMIT FASTWAM_TRAINING_JOB_ID
)
for name in "${required_env[@]}"; do
  [[ -n "${!name:-}" ]] || die "missing environment variable ${name}"
done

[[ "${FASTWAM_EVAL_SCOPE}" == 'smoke' || "${FASTWAM_EVAL_SCOPE}" == 'formal' ]] || die 'scope must be smoke or formal'
[[ "${FASTWAM_EXPERIMENT_ID}" == "${EXPERIMENT_ID}" ]] || die 'experiment identity drift'
[[ "${FASTWAM_RUN_ID}" == "${RUN_ID}" ]] || die 'run identity drift'
[[ "${FASTWAM_ATTEMPT_ID}" == 'attempt-004' ]] || die 'attempt identity drift'
[[ "${FASTWAM_CHECKPOINT}" == '/oss-chengjuntao/artifacts/fastwam-n234-vg1h1gau1-cont50k-s42-24g-r1-20260822/checkpoints/weights/step_010000.pt' ]] || die 'checkpoint path drift'
[[ "${FASTWAM_CHECKPOINT_SIZE_BYTES}" == '12047213657' ]] || die 'checkpoint byte-size drift'

if [[ "${FASTWAM_EVAL_SCOPE}" == 'smoke' ]]; then
  expected_output='/oss-chengjuntao/artifacts/fastwam-gau1-step10k-placefood-same8-eval-dsw4-r4-smoke-episode0-20260823'
  expected_control='/oss-chengjuntao/artifacts/fastwam-gau1-step10k-placefood-same8-eval-dsw4-r4-smoke-control-20260823'
else
  expected_output='/oss-chengjuntao/artifacts/fastwam-gau1-step10k-placefood-same8-eval-dsw4-r4-20260823'
  expected_control='/oss-chengjuntao/artifacts/fastwam-gau1-step10k-placefood-same8-eval-dsw4-r4-control-20260823'
fi
[[ "${FASTWAM_OUTPUT_ROOT}" == "${expected_output}" ]] || die 'output root drift'
[[ "${FASTWAM_CONTROL_ROOT}" == "${expected_control}" ]] || die 'control root drift'
[[ ! -e "${FASTWAM_OUTPUT_ROOT}" && ! -L "${FASTWAM_OUTPUT_ROOT}" ]] || die 'output root already exists'
[[ ! -e "${FASTWAM_CONTROL_ROOT}" && ! -L "${FASTWAM_CONTROL_ROOT}" ]] || die 'control root already exists'

evaluator="${FASTWAM_SOURCE_ROOT}/experiments/robofactory/eval_robofactory_multi_robot.py"
aggregator="${FASTWAM_SOURCE_ROOT}/${EXPERIMENT_REL}/aggregate_results_dsw.py"
source_src="${FASTWAM_SOURCE_ROOT}/src"
[[ -f "${evaluator}" && ! -L "${evaluator}" ]] || die 'evaluator is missing or unsafe'
[[ -f "${aggregator}" && ! -L "${aggregator}" ]] || die 'aggregator is missing or unsafe'
[[ -d "${source_src}" && ! -L "${source_src}" ]] || die 'source directory is missing or unsafe'
[[ -x "${FASTWAM_PYTHON}" ]] || die 'evaluation Python is not executable'
for file in "${FASTWAM_CHECKPOINT}" "${FASTWAM_PANEL}" "${FASTWAM_STATS}" "${FASTWAM_NOPOSPLAT_CHECKPOINT}"; do
  [[ -f "${file}" && ! -L "${file}" ]] || die "required file is missing or unsafe: ${file}"
done
for directory in "${FASTWAM_DATASET_ROOT}" "${FASTWAM_ROBOFACTORY_ROOT}" \
  "${FASTWAM_CONTEXT_CACHE_DIR}" "${FASTWAM_MODEL_CACHE_ROOT}" \
  "${FASTWAM_POLICY_LIGHTNING_ROOT}" "${FASTWAM_NVIDIA_GRAPHICS_ROOT}"; do
  [[ -d "${directory}" && ! -L "${directory}" ]] || die "required directory is missing or unsafe: ${directory}"
done

observed_commit="$(git -C "${FASTWAM_SOURCE_ROOT}" rev-parse HEAD)"
[[ "${observed_commit}" == "${FASTWAM_SOURCE_COMMIT}" ]] || die 'source commit drift'
[[ -z "$(git -C "${FASTWAM_SOURCE_ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || die 'source checkout is not clean'

mkdir -m 0700 -- "${FASTWAM_CONTROL_ROOT}"
mkdir -m 0700 -- "${FASTWAM_CONTROL_ROOT}/logs" "${FASTWAM_CONTROL_ROOT}/shards"
scratch_root="$(mktemp -d /tmp/fastwam-step10k-placefood-dsw.XXXXXXXX)"
cleanup() {
  case "${scratch_root}" in
    /tmp/fastwam-step10k-placefood-dsw.*) rm -rf -- "${scratch_root}" ;;
    *) printf 'STEP10K_DSW_EVAL_WARN: refusing unsafe scratch cleanup %s\n' "${scratch_root}" >&2 ;;
  esac
}
trap cleanup EXIT
mkdir -m 0700 -- "${scratch_root}/xdg-cache" "${scratch_root}/xdg-runtime" \
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

"${FASTWAM_PYTHON}" -B - <<'PY' || die 'source identity gate failed'
import os
from pathlib import Path

import fastwam
import fastwam.runtime as runtime

expected = (Path(os.environ["FASTWAM_SOURCE_ROOT"]) / "src").resolve(strict=True)
for label, module in (("fastwam", fastwam), ("fastwam.runtime", runtime)):
    path = Path(module.__file__).resolve(strict=True)
    try:
        path.relative_to(expected)
    except ValueError as error:
        raise SystemExit(f"{label} resolved outside source checkout: {path}") from error
if not callable(getattr(runtime, "create_multi_robot_fastwam", None)):
    raise SystemExit("source runtime lacks callable create_multi_robot_fastwam")
print("STEP10K_DSW_EVAL_SOURCE_GATE=PASS", flush=True)
PY

gpu_count="$(${FASTWAM_PYTHON} -B -c 'import torch; print(torch.cuda.device_count())')"
[[ "${gpu_count}" == '4' ]] || die "expected exactly 4 visible GPUs, observed ${gpu_count}"

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

run_wave() {
  local first="$1"
  local last="$2"
  local index gpu shard log
  local -a pids=()
  local failure=0
  for ((index=first; index<=last; index++)); do
    gpu=$((index - first))
    shard="${FASTWAM_CONTROL_ROOT}/shards/episode-$(printf '%02d' "${index}")"
    log="${FASTWAM_CONTROL_ROOT}/logs/episode-$(printf '%02d' "${index}").log"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${source_src}" PYTHONNOUSERSITE=1 \
      PYTHONDONTWRITEBYTECODE=1 "${FASTWAM_PYTHON}" -B "${common_argv[@]}" \
      --episode-start "${index}" --output-dir "${shard}" >"${log}" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failure=1; fi
  done
  if (( failure != 0 )); then
    for ((index=first; index<=last; index++)); do
      log="${FASTWAM_CONTROL_ROOT}/logs/episode-$(printf '%02d' "${index}").log"
      printf '===== episode %s =====\n' "${index}" >&2
      tail -n 160 -- "${log}" >&2 || true
    done
    die "one or more evaluator processes failed in wave ${first}-${last}"
  fi
}

if [[ "${FASTWAM_EVAL_SCOPE}" == 'smoke' ]]; then
  run_wave 0 0
  "${FASTWAM_PYTHON}" -B - "${FASTWAM_CONTROL_ROOT}/shards/episode-00" "${FASTWAM_OUTPUT_ROOT}" <<'PY'
import json
import os
import shutil
import stat
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
required = {"episodes.jsonl", "run_manifest.json", "summary.json"}
if {item.name for item in source.iterdir()} != required:
    raise SystemExit("smoke shard file set drift")
manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
lines = [line for line in (source / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
if len(lines) != 1:
    raise SystemExit("smoke must contain exactly one episode")
record = json.loads(lines[0])
if manifest.get("status") != "terminal" or manifest.get("integrity_mode") != "metadata_no_hash":
    raise SystemExit("smoke manifest did not reach the frozen terminal contract")
if summary.get("status") != "PASS" or summary.get("infrastructure_errors") != 0:
    raise SystemExit("smoke contains an infrastructure failure")
if record.get("status") != "completed" or record.get("panel_index") != 0:
    raise SystemExit("smoke episode did not complete panel index zero")
for path in source.iterdir():
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise SystemExit(f"unsafe smoke artifact: {path}")
staging = output.parent / f".{output.name}.staging.{os.getpid()}"
staging.mkdir(mode=0o700)
try:
    for path in source.iterdir():
        with path.open("rb") as reader, (staging / path.name).open("xb") as writer:
            shutil.copyfileobj(reader, writer)
    (staging / "SMOKE_COMPLETE.json").write_text(
        json.dumps({"status": "PASS", "episode": 0, "success": record.get("success")}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.rename(staging, output)
except Exception:
    if staging.exists():
        shutil.rmtree(staging)
    raise
print(json.dumps({"status": "PASS", "episode": 0, "success": record.get("success")}), flush=True)
PY
  printf 'STEP10K_DSW_EVAL_SMOKE_COMPLETE\n'
else
  run_wave 0 3
  run_wave 4 7
  "${FASTWAM_PYTHON}" -B "${aggregator}" \
    --temp-root "${FASTWAM_CONTROL_ROOT}/shards" \
    --output-root "${FASTWAM_OUTPUT_ROOT}" \
    --source-commit "${FASTWAM_SOURCE_COMMIT}" \
    --job-id "dsw:${HOSTNAME:-unknown}"
  printf 'STEP10K_DSW_EVAL_SCIENTIFIC_COMPLETE\n'
fi
