#!/usr/bin/env bash
set -euo pipefail

# Persistent, fail-closed launcher for the POSE_FOCUS 3 Worker x 8 GPU treatment.
# The outer DLC command runs once per worker. It stages the R5 weight before
# spawning the eight local Accelerate ranks, so every child reads node-local
# storage and no rank can accidentally restore the old 32-GPU optimizer state.

die() {
  printf 'POSE_FOCUS launcher error: %s\n' "$*" >&2
  exit 1
}

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "required environment variable ${name} is empty"
}

require_exact_env() {
  local name="$1"
  local expected="$2"
  require_env "${name}"
  [[ "${!name}" == "${expected}" ]] || die "${name} drifted from the verified DLC contract"
}

is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_safe_id() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]
}

is_non_loopback() {
  [[ -n "$1" && "$1" != "localhost" && "$1" != "::1" && "$1" != 127.* ]]
}

TEST_MODE="${FASTWAM_POSE_FOCUS_TEST_MODE:-0}"
[[ "${TEST_MODE}" == "0" || "${TEST_MODE}" == "1" ]] || die "FASTWAM_POSE_FOCUS_TEST_MODE must be 0 or 1"
if [[ "${TEST_MODE}" == "1" && -n "${PAI_DLC_JOB_ID:-}${DLC_JOB_ID:-}${PAI_JOB_ID:-}" ]]; then
  die "FASTWAM_POSE_FOCUS_TEST_MODE is forbidden inside a DLC job"
fi
require_env RUN_ID
require_env FASTWAM_POSE_FOCUS_ATTEMPT_ID

# PAI injects node topology through these names.  Preserve the outer worker
# values before any helper can mutate the environment; they are deliberately
# removed immediately before Accelerate creates its own 24-rank world.
NUM_MACHINES="${WORLD_SIZE:-}"
MACHINE_RANK="${RANK:-}"
GPUS_PER_NODE="${NPROC_PER_NODE:-}"
MASTER_HOST="${MASTER_ADDR:-}"
MASTER_TCP_PORT="${MASTER_PORT:-}"

# Preserve the POSE_FOCUS source identity before the legacy dependency bootstrap is
# sourced.  That bootstrap is allowed to provide only a node-local Python and
# its dependencies; its historical source checkout must never become the POSE_FOCUS
# training source.
POSE_FOCUS_SOURCE_BUNDLE="${FASTWAM_POSE_FOCUS_SOURCE_BUNDLE:-}"
POSE_FOCUS_CODE_COMMIT="${FASTWAM_POSE_FOCUS_CODE_COMMIT:-}"
POSE_FOCUS_LOCAL_SOURCE_ROOT="${FASTWAM_POSE_FOCUS_LOCAL_SOURCE_ROOT:-/tmp/fastwam-pose_focus-source-checkouts}"
OFFLINE_CODE_COMMIT="${FASTWAM_OFFLINE_CODE_COMMIT:-}"

# The DLC image is only a CUDA/OS substrate.  A production worker first enters
# the audited offline dependency environment and then checks out the current
# POSE_FOCUS Git bundle into a separate node-local directory.  There is deliberately no
# `python3` fallback and no execution of a launcher from the legacy snapshot.
if [[ "${FASTWAM_POSE_FOCUS_OFFLINE_ENV_READY:-0}" != "1" ]]; then
  require_env FASTWAM_POSE_FOCUS_BOOTSTRAP_SCRIPT
  require_env FASTWAM_OFFLINE_ENV_BASE_PYTHON
  require_env FASTWAM_POSE_FOCUS_SOURCE_BUNDLE
  require_env FASTWAM_POSE_FOCUS_CODE_COMMIT
  require_env FASTWAM_OFFLINE_CODE_COMMIT
  if [[ "${TEST_MODE}" != "1" ]]; then
    [[ "${FASTWAM_POSE_FOCUS_BOOTSTRAP_SCRIPT}" == /oss-chengjuntao/* ]] || \
      die "bootstrap script must be an absolute path below /oss-chengjuntao"
    [[ "${POSE_FOCUS_SOURCE_BUNDLE}" == /oss-chengjuntao/* ]] || \
      die "POSE_FOCUS source bundle must be an absolute path below /oss-chengjuntao"
    [[ "${POSE_FOCUS_LOCAL_SOURCE_ROOT}" == /tmp/* ]] || \
      die "POSE_FOCUS source checkout root must be node-local below /tmp"
  fi
  [[ "${POSE_FOCUS_CODE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
    die "FASTWAM_POSE_FOCUS_CODE_COMMIT must be an exact lowercase 40-hex Git revision"
  [[ "${OFFLINE_CODE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
    die "FASTWAM_OFFLINE_CODE_COMMIT must be an exact lowercase 40-hex Git revision"
  [[ "${POSE_FOCUS_SOURCE_BUNDLE}" == *.bundle ]] || die "POSE_FOCUS source must be a Git .bundle file"
  [[ -f "${POSE_FOCUS_SOURCE_BUNDLE}" && ! -L "${POSE_FOCUS_SOURCE_BUNDLE}" ]] || \
    die "POSE_FOCUS source bundle must be a regular non-symlink file"
  [[ "${POSE_FOCUS_LOCAL_SOURCE_ROOT}" == /* ]] || die "POSE_FOCUS local source root must be absolute"
  [[ -f "${FASTWAM_POSE_FOCUS_BOOTSTRAP_SCRIPT}" && ! -L "${FASTWAM_POSE_FOCUS_BOOTSTRAP_SCRIPT}" ]] || \
    die "bootstrap script must be a regular non-symlink file"
  [[ "${FASTWAM_OFFLINE_ENV_BASE_PYTHON}" == /* && -x "${FASTWAM_OFFLINE_ENV_BASE_PYTHON}" ]] || \
    die "FASTWAM_OFFLINE_ENV_BASE_PYTHON must be an absolute executable"

  # shellcheck source=/dev/null
  source "${FASTWAM_POSE_FOCUS_BOOTSTRAP_SCRIPT}"
  declare -F fastwam_prepare_offline_training_env >/dev/null || \
    die "offline bootstrap does not define fastwam_prepare_offline_training_env"
  # The legacy helper expects FASTWAM_CODE_COMMIT for its own archived source.
  # It must never see the POSE_FOCUS revision under that ambiguous historical name.
  export FASTWAM_CODE_COMMIT="${OFFLINE_CODE_COMMIT}"
  fastwam_prepare_offline_training_env || die "offline dependency bootstrap failed"
  DEPENDENCY_PYTHON="${FASTWAM_PYTHON:-}"
  [[ "${DEPENDENCY_PYTHON}" == /* && -x "${DEPENDENCY_PYTHON}" ]] || \
    die "offline bootstrap did not export an executable FASTWAM_PYTHON"
  if [[ "${TEST_MODE}" != "1" ]]; then
    [[ "${DEPENDENCY_PYTHON}" == /tmp/* ]] || \
      die "offline dependency Python must be node-local below /tmp"
  fi

  command -v git >/dev/null 2>&1 || die "git is required to restore the POSE_FOCUS source bundle"
  POSE_FOCUS_LOCAL_PARENT="${POSE_FOCUS_LOCAL_SOURCE_ROOT}/${RUN_ID:-missing-run-id}"
  POSE_FOCUS_LOCAL_REPO="${POSE_FOCUS_LOCAL_PARENT}/${FASTWAM_POSE_FOCUS_ATTEMPT_ID:-missing-attempt-id}"
  POSE_FOCUS_PARTIAL_REPO="${POSE_FOCUS_LOCAL_REPO}.partial.${BASHPID}"
  [[ ! -e "${POSE_FOCUS_LOCAL_REPO}" && ! -L "${POSE_FOCUS_LOCAL_REPO}" ]] || \
    die "POSE_FOCUS node-local source checkout already exists: ${POSE_FOCUS_LOCAL_REPO}"
  [[ ! -e "${POSE_FOCUS_PARTIAL_REPO}" && ! -L "${POSE_FOCUS_PARTIAL_REPO}" ]] || \
    die "POSE_FOCUS partial source checkout already exists: ${POSE_FOCUS_PARTIAL_REPO}"
  mkdir -p -- "${POSE_FOCUS_LOCAL_PARENT}"
  git clone --quiet --no-checkout -- "${POSE_FOCUS_SOURCE_BUNDLE}" "${POSE_FOCUS_PARTIAL_REPO}" || \
    die "failed to clone the POSE_FOCUS source bundle"
  git -C "${POSE_FOCUS_PARTIAL_REPO}" checkout --quiet --detach "${POSE_FOCUS_CODE_COMMIT}" || \
    die "POSE_FOCUS commit is not present in the supplied source bundle"
  [[ "$(git -C "${POSE_FOCUS_PARTIAL_REPO}" rev-parse HEAD)" == "${POSE_FOCUS_CODE_COMMIT}" ]] || \
    die "restored POSE_FOCUS Git revision differs from FASTWAM_POSE_FOCUS_CODE_COMMIT"
  [[ -z "$(git -C "${POSE_FOCUS_PARTIAL_REPO}" status --porcelain --untracked-files=all)" ]] || \
    die "restored POSE_FOCUS source checkout is dirty"
  mv -- "${POSE_FOCUS_PARTIAL_REPO}" "${POSE_FOCUS_LOCAL_REPO}"

  # Rebind every source selector after bootstrap.  In particular, discard the
  # legacy FASTWAM_REPO_ROOT and PYTHONPATH values exported by the v9 helper.
  export FASTWAM_POSE_FOCUS_PYTHON="${DEPENDENCY_PYTHON}"
  export FASTWAM_POSE_FOCUS_REPO_ROOT="${POSE_FOCUS_LOCAL_REPO}"
  export FASTWAM_REPO_ROOT="${POSE_FOCUS_LOCAL_REPO}"
  # From this point forward, the unqualified legacy name is rebound to POSE_FOCUS so a
  # downstream helper cannot silently select the archived v9 source checkout.
  export FASTWAM_CODE_COMMIT="${POSE_FOCUS_CODE_COMMIT}"
  export PYTHONPATH="${POSE_FOCUS_LOCAL_REPO}/src"
  export PYTHONNOUSERSITE=1
  export FASTWAM_POSE_FOCUS_OFFLINE_ENV_READY=1
else
  require_env FASTWAM_POSE_FOCUS_REPO_ROOT
  require_env FASTWAM_POSE_FOCUS_PYTHON
  if [[ "${TEST_MODE}" != "1" ]]; then
    require_env FASTWAM_POSE_FOCUS_CODE_COMMIT
    [[ "${POSE_FOCUS_CODE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
      die "FASTWAM_POSE_FOCUS_CODE_COMMIT must be an exact lowercase 40-hex Git revision"
  fi
  export FASTWAM_REPO_ROOT="${FASTWAM_POSE_FOCUS_REPO_ROOT}"
  export FASTWAM_CODE_COMMIT="${POSE_FOCUS_CODE_COMMIT}"
  export PYTHONPATH="${FASTWAM_POSE_FOCUS_REPO_ROOT}/src"
  export PYTHONNOUSERSITE=1
fi

RUN_ID="${RUN_ID:-}"
ATTEMPT_ID="${FASTWAM_POSE_FOCUS_ATTEMPT_ID:-}"
REPO_ROOT="${FASTWAM_POSE_FOCUS_REPO_ROOT:-}"
OUTPUT_DIR="${FASTWAM_POSE_FOCUS_OUTPUT_DIR:-}"
DATASET_ROOT="${FASTWAM_POSE_FOCUS_DATASET_ROOT:-/cpfs/user/chengjuntao/datasets/robofactory_multi_robot}"
STATS_SOURCE_ROOT="${DATASET_ROOT}"
STATS_PATH="${FASTWAM_POSE_FOCUS_STATS_PATH:-${DATASET_ROOT}/fastwam_multi_robot_n234_train_s42_stats_v2.json}"
TEXT_CACHE_DIR="${FASTWAM_POSE_FOCUS_TEXT_CACHE_DIR:-${DATASET_ROOT}/text_embeds_cache_n234}"
GAUSSIAN_CACHE_DIR="${FASTWAM_POSE_FOCUS_GAUSSIAN_CACHE_DIR:-/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/noposplat-c944b498-4a35bc8c/builds/fastwam-8a035024af96-s42-20260801T230944Z/compact-s42-13x28x40-fp16-meanalpha-v2}"
MODEL_CACHE_ROOT="${DIFFSYNTH_MODEL_BASE_PATH:-}"
VAE_PATH="${FASTWAM_LOCAL_VAE_PATH:-}"
PYTHON_BIN="${FASTWAM_POSE_FOCUS_PYTHON:-}"
DRY_RUN="${FASTWAM_POSE_FOCUS_DRY_RUN:-0}"
TASK_PROFILE="${FASTWAM_POSE_FOCUS_TASK_PROFILE:-robofactory_placefood_pose_focus_r5_224_5e-6}"
SCALE_PROFILE="robofactory_multi_robot_24gpu_pose_focus"
case "${TASK_PROFILE}" in
  robofactory_placefood_pose_focus_r5_224_5e-6|robofactory_placefood_pose_phase_x0_r5_224_5e-6) ;;
  robofactory_placefood_gaussian_spatial_p4_224_5e-6|robofactory_placefood_semantic_phase_p5_224_5e-6) ;;
  robofactory_placefood_spatial_semantic_p6_224_5e-6) ;;
  *) die "unsupported FASTWAM_POSE_FOCUS_TASK_PROFILE=${TASK_PROFILE}" ;;
esac
R5_SOURCE="/oss-chengjuntao/artifacts/fastwam-action-n234-formal-r5-20260812/fastwam-act-n2-placefood-1k-s42-r5-20260812/checkpoints/weights/step_001000.pt"
P1_SOURCE="/oss-chengjuntao/artifacts/fastwam-placefood-posefocus-r5-s42-24g-r2-20260813/checkpoints/weights/step_001000.pt"
P2_SOURCE="/oss-chengjuntao/artifacts/fastwam-placefood-phase-x0-r5-s42-24g-r1-20260813/checkpoints/weights/step_001000.pt"
P5_SOURCE="/oss-chengjuntao/artifacts/fastwam-placefood-semantic-phase-p5-s42-24g-r1-20260814/checkpoints/weights/step_001000.pt"
CANONICAL_SOURCE="${R5_SOURCE}"
case "${TASK_PROFILE}" in
  robofactory_placefood_gaussian_spatial_p4_224_5e-6) CANONICAL_SOURCE="${P1_SOURCE}" ;;
  robofactory_placefood_semantic_phase_p5_224_5e-6) CANONICAL_SOURCE="${P2_SOURCE}" ;;
  robofactory_placefood_spatial_semantic_p6_224_5e-6) CANONICAL_SOURCE="${P5_SOURCE}" ;;
esac
SOURCE_WEIGHT="${FASTWAM_POSE_FOCUS_SOURCE_WEIGHT:-${CANONICAL_SOURCE}}"
EXPECTED_WEIGHT_BYTES="${FASTWAM_POSE_FOCUS_SOURCE_WEIGHT_BYTES:-12047407619}"

require_env RUN_ID
require_env FASTWAM_POSE_FOCUS_ATTEMPT_ID
require_env FASTWAM_POSE_FOCUS_OUTPUT_DIR
require_env FASTWAM_POSE_FOCUS_REPO_ROOT
require_env FASTWAM_POSE_FOCUS_PYTHON
is_safe_id "${RUN_ID}" || die "RUN_ID is not a safe identifier: ${RUN_ID}"
is_safe_id "${ATTEMPT_ID}" || die "FASTWAM_POSE_FOCUS_ATTEMPT_ID is not a safe identifier: ${ATTEMPT_ID}"
if [[ -n "${FASTWAM_B4_ATTEMPT_ID:-}" && "${FASTWAM_B4_ATTEMPT_ID}" != "${ATTEMPT_ID}" ]]; then
  die "FASTWAM_B4_ATTEMPT_ID conflicts with FASTWAM_POSE_FOCUS_ATTEMPT_ID"
fi
# stat_cmp config publication is shared with the audited B4 runtime contract.
export FASTWAM_B4_ATTEMPT_ID="${ATTEMPT_ID}"
printf 'POSE_FOCUS runtime provenance binding: FASTWAM_B4_ATTEMPT_ID=%s\n' "${FASTWAM_B4_ATTEMPT_ID}"

[[ "${NUM_MACHINES}" == "3" ]] || die "WORLD_SIZE must be the DLC worker count 3, got ${NUM_MACHINES:-unset}"
[[ "${GPUS_PER_NODE}" == "8" ]] || die "NPROC_PER_NODE must be 8, got ${GPUS_PER_NODE:-unset}"
is_uint "${MACHINE_RANK:-x}" || die "RANK must be an integer in [0,2], got ${MACHINE_RANK:-unset}"
((10#${MACHINE_RANK} < 3)) || die "RANK must be in [0,2], got ${MACHINE_RANK}"
[[ -z "${LOCAL_RANK:-}" || "${LOCAL_RANK}" == "0" ]] || die "outer DLC command must run only once per node (LOCAL_RANK=0)"
is_non_loopback "${MASTER_HOST}" || die "MASTER_ADDR must be a non-loopback address shared by all workers"
is_uint "${MASTER_TCP_PORT:-x}" || die "MASTER_PORT must be an integer"
((10#${MASTER_TCP_PORT} >= 1 && 10#${MASTER_TCP_PORT} <= 65535)) || die "MASTER_PORT must be in [1,65535]"
[[ $((10#${NUM_MACHINES} * 10#${GPUS_PER_NODE})) -eq 24 ]] || die "global world size must be exactly 24"

[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "FASTWAM_POSE_FOCUS_DRY_RUN must be 0 or 1"

[[ "${REPO_ROOT}" == /* && -d "${REPO_ROOT}" ]] || die "repository root is not an existing absolute directory: ${REPO_ROOT}"
if [[ "${TEST_MODE}" != "1" ]]; then
  [[ "${REPO_ROOT}" == /tmp/* ]] || die "production source checkout must be node-local below /tmp"
  [[ "${PYTHON_BIN}" == /tmp/* ]] || die "production Python must come from the node-local offline environment"
fi
[[ -f "${REPO_ROOT}/scripts/train.py" ]] || die "missing scripts/train.py under ${REPO_ROOT}"
[[ -f "${REPO_ROOT}/scripts/b4_stat_cmp_cache.py" && ! -L "${REPO_ROOT}/scripts/b4_stat_cmp_cache.py" ]] || \
  die "missing POSE_FOCUS stat-cmp cache helper"
[[ -f "${REPO_ROOT}/scripts/accelerate_configs/accelerate_zero2_ds.yaml" ]] || die "missing ZeRO-2 Accelerate config"
[[ -f "${REPO_ROOT}/src/fastwam/trainer.py" ]] || die "missing trainer source"
[[ -f "${REPO_ROOT}/configs/task/${TASK_PROFILE}.yaml" ]] || die "missing formal POSE_FOCUS task profile"
[[ -f "${REPO_ROOT}/configs/scale/${SCALE_PROFILE}.yaml" ]] || die "missing formal POSE_FOCUS 24-GPU scale profile"
if [[ "${TEST_MODE}" == "1" ]]; then
  [[ "${OUTPUT_DIR}" == /* ]] || die "test output must be an absolute path"
else
  [[ "${OUTPUT_DIR}" == "/oss-chengjuntao/artifacts/${RUN_ID}" ]] || die "output must be the unique canonical path /oss-chengjuntao/artifacts/${RUN_ID}"
  [[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${POSE_FOCUS_CODE_COMMIT}" ]] || \
    die "active POSE_FOCUS checkout does not match FASTWAM_POSE_FOCUS_CODE_COMMIT"
  [[ -z "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=all)" ]] || \
    die "active POSE_FOCUS source checkout is dirty"
fi

if [[ "${TEST_MODE}" != "1" ]]; then
  [[ "${SOURCE_WEIGHT}" == "${CANONICAL_SOURCE}" ]] || \
    die "production source does not match the audited source for ${TASK_PROFILE}"
  [[ "${EXPECTED_WEIGHT_BYTES}" == "12047407619" ]] || die "production source byte count must remain 12047407619"
fi
[[ "${SOURCE_WEIGHT}" == *.pt ]] || die "source must be a weight .pt file, not a training-state directory"
[[ "${SOURCE_WEIGHT}" != */checkpoints/state/* ]] || die "32-GPU optimizer/training state is forbidden"
[[ -f "${SOURCE_WEIGHT}" && ! -L "${SOURCE_WEIGHT}" ]] || die "source weight must be an existing regular non-symlink file"
is_uint "${EXPECTED_WEIGHT_BYTES}" || die "FASTWAM_POSE_FOCUS_SOURCE_WEIGHT_BYTES must be a positive integer"
((10#${EXPECTED_WEIGHT_BYTES} > 0)) || die "FASTWAM_POSE_FOCUS_SOURCE_WEIGHT_BYTES must be positive"
[[ "$(stat -c '%s' -- "${SOURCE_WEIGHT}")" == "${EXPECTED_WEIGHT_BYTES}" ]] || die "source weight byte count differs from the audited value"

[[ "${PYTHON_BIN}" == /* && -x "${PYTHON_BIN}" ]] || die "Python executable is unavailable: ${PYTHON_BIN}"

# Fail closed if Python still resolves FastWAM from the dependency bootstrap's
# historical source checkout.  PYTHONPATH is intentionally only the new POSE_FOCUS
# source tree, rather than an append to the old environment value.
FASTWAM_EXPECTED_REPO_ROOT="${REPO_ROOT}" \
PYTHONPATH="${REPO_ROOT}/src" \
PYTHONNOUSERSITE=1 \
"${PYTHON_BIN}" - <<'PY' || die "fastwam import did not resolve from the active POSE_FOCUS source checkout"
import os
from pathlib import Path

import fastwam

origin = Path(fastwam.__file__).resolve()
expected = (Path(os.environ["FASTWAM_EXPECTED_REPO_ROOT"]) / "src" / "fastwam").resolve()
try:
    origin.relative_to(expected)
except ValueError as exc:
    raise SystemExit(f"fastwam import origin escaped POSE_FOCUS source: {origin} not below {expected}") from exc
print(f"POSE_FOCUS source import gate: origin={origin} expected_root={expected}")
PY

# Stage the completed 32-GPU job's two published file selections without
# calculating a new POSE_FOCUS digest.  The historical manifest files are used only as
# relative-path allowlists: their first field is ignored.  Every selected file
# must be a regular non-symlink below the declared source root, and the helper
# verifies source/destination stat plus a full byte comparison before atomically
# publishing a READY.stat-cmp.json contract.
if [[ "${TEST_MODE}" != "1" && "${DRY_RUN}" == "0" ]]; then
  require_exact_env FASTWAM_POSE_FOCUS_PROVENANCE_MODE "stat_cmp"
  require_exact_env FASTWAM_POSE_FOCUS_INPUT_CACHE_ROOT "/tmp/fastwam-pose_focus-input-cache"
  require_exact_env FASTWAM_POSE_FOCUS_CPFS_SOURCE_ROOT "/oss-chengjuntao/cpfs-user-chengjuntao"
  require_exact_env FASTWAM_POSE_FOCUS_STATS_SOURCE_ROOT "/cpfs/user/chengjuntao/datasets/robofactory_multi_robot"
  require_exact_env FASTWAM_POSE_FOCUS_CPFS_ALLOWLIST "/oss-chengjuntao/artifacts/fastwam-n234-input-bundles-s42-v1-2023667-20260802T1235Z/cpfs-whole-file-bundle.sha256"
  require_exact_env FASTWAM_POSE_FOCUS_OSS_SOURCE_ROOT "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/noposplat-c944b498-4a35bc8c/builds/fastwam-8a035024af96-s42-20260801T230944Z"
  require_exact_env FASTWAM_POSE_FOCUS_OSS_ALLOWLIST "/oss-chengjuntao/artifacts/fastwam-n234-input-bundles-s42-v1-2023667-20260802T1235Z/oss-compact-whole-file-bundle.sha256"
  require_exact_env FASTWAM_LOCAL_DATASET_RELATIVE_ROOT "datasets/robofactory_multi_robot"
  require_exact_env FASTWAM_LOCAL_STATS_RELATIVE_PATH "datasets/robofactory_multi_robot/fastwam_multi_robot_n234_train_s42_stats_v2.json"
  require_exact_env FASTWAM_LOCAL_TEXT_EMBEDS_RELATIVE_ROOT "datasets/robofactory_multi_robot/text_embeds_cache_n234"
  require_exact_env FASTWAM_LOCAL_MODEL_CACHE_RELATIVE_ROOT "checkpoints/FastWAM/model-cache"
  require_exact_env FASTWAM_LOCAL_VAE_RELATIVE_PATH "checkpoints/FastWAM/model-cache/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
  require_exact_env FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT "compact-s42-13x28x40-fp16-meanalpha-v2"
  require_exact_env FASTWAM_LOCAL_EXPECTED_H5_FILES "24"
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/scripts/dlc_preflight.sh"
  fastwam_prepare_nvidia_host570 || die "NVIDIA host preparation failed"
  fastwam_run_local_cuda_preflight "${GPUS_PER_NODE}" "${MACHINE_RANK}" || \
    die "local CUDA preflight failed"

  POSE_FOCUS_INPUT_ATTEMPT_ROOT="${FASTWAM_POSE_FOCUS_INPUT_CACHE_ROOT}/${RUN_ID}/${ATTEMPT_ID}"
  FASTWAM_LOCAL_CPFS_CACHE_DIR="${POSE_FOCUS_INPUT_ATTEMPT_ROOT}/cpfs"
  FASTWAM_LOCAL_OSS_CACHE_DIR="${POSE_FOCUS_INPUT_ATTEMPT_ROOT}/oss"
  [[ "${POSE_FOCUS_INPUT_ATTEMPT_ROOT}" == /tmp/* ]] || \
    die "POSE_FOCUS input cache must remain on node-local /tmp"
  [[ ! -e "${POSE_FOCUS_INPUT_ATTEMPT_ROOT}" && ! -L "${POSE_FOCUS_INPUT_ATTEMPT_ROOT}" ]] || \
    die "POSE_FOCUS input cache attempt already exists: ${POSE_FOCUS_INPUT_ATTEMPT_ROOT}"
  mkdir -p -- "${POSE_FOCUS_INPUT_ATTEMPT_ROOT}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/b4_stat_cmp_cache.py" \
    --source-root "${FASTWAM_POSE_FOCUS_CPFS_SOURCE_ROOT}" \
    --allowlist "${FASTWAM_POSE_FOCUS_CPFS_ALLOWLIST}" \
    --destination "${FASTWAM_LOCAL_CPFS_CACHE_DIR}" \
    --run-id "${RUN_ID}" --attempt-id "${ATTEMPT_ID}" --source-label cpfs || \
    die "CPFS stat-cmp node-local cache preparation failed"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/b4_stat_cmp_cache.py" \
    --source-root "${FASTWAM_POSE_FOCUS_OSS_SOURCE_ROOT}" \
    --allowlist "${FASTWAM_POSE_FOCUS_OSS_ALLOWLIST}" \
    --destination "${FASTWAM_LOCAL_OSS_CACHE_DIR}" \
    --run-id "${RUN_ID}" --attempt-id "${ATTEMPT_ID}" --source-label oss || \
    die "OSS stat-cmp node-local cache preparation failed"
  [[ -f "${FASTWAM_LOCAL_CPFS_CACHE_DIR}/READY.stat-cmp.json" ]] || \
    die "CPFS stat-cmp READY contract is missing"
  [[ -f "${FASTWAM_LOCAL_OSS_CACHE_DIR}/READY.stat-cmp.json" ]] || \
    die "OSS stat-cmp READY contract is missing"

  # Preserve the logical source root declared by the published stats before
  # rebinding reads to the attempt-owned node-local copy. The physical OSS
  # mirror remains the staging source above; dataset provenance accepts only
  # this explicit logical-source -> staged-read pair.
  STATS_SOURCE_ROOT="${FASTWAM_POSE_FOCUS_STATS_SOURCE_ROOT}"
  DATASET_ROOT="${FASTWAM_LOCAL_CPFS_CACHE_DIR}/${FASTWAM_LOCAL_DATASET_RELATIVE_ROOT}"
  STATS_PATH="${FASTWAM_LOCAL_CPFS_CACHE_DIR}/${FASTWAM_LOCAL_STATS_RELATIVE_PATH}"
  TEXT_CACHE_DIR="${FASTWAM_LOCAL_CPFS_CACHE_DIR}/${FASTWAM_LOCAL_TEXT_EMBEDS_RELATIVE_ROOT}"
  MODEL_CACHE_ROOT="${FASTWAM_LOCAL_CPFS_CACHE_DIR}/${FASTWAM_LOCAL_MODEL_CACHE_RELATIVE_ROOT}"
  VAE_PATH="${FASTWAM_LOCAL_CPFS_CACHE_DIR}/${FASTWAM_LOCAL_VAE_RELATIVE_PATH}"
  GAUSSIAN_CACHE_DIR="${FASTWAM_LOCAL_OSS_CACHE_DIR}/${FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT}"
  export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_CACHE_ROOT}"
  export FASTWAM_LOCAL_VAE_PATH="${VAE_PATH}"
fi

[[ -d "${DATASET_ROOT}" ]] || die "dataset root does not exist: ${DATASET_ROOT}"
[[ -f "${STATS_PATH}" && ! -L "${STATS_PATH}" ]] || \
  die "normalization stats must be an existing regular non-symlink file"
[[ -d "${TEXT_CACHE_DIR}" ]] || die "text embedding cache does not exist: ${TEXT_CACHE_DIR}"
[[ -d "${GAUSSIAN_CACHE_DIR}" ]] || die "Gaussian cache does not exist: ${GAUSSIAN_CACHE_DIR}"
if [[ "${TEST_MODE}" != "1" && "${DRY_RUN}" == "0" ]]; then
  [[ -d "${MODEL_CACHE_ROOT}" ]] || die "DiffSynth model cache is missing: ${MODEL_CACHE_ROOT}"
  [[ -f "${VAE_PATH}" && ! -L "${VAE_PATH}" ]] || \
    die "Wan2.2 VAE must be a node-local regular non-symlink file"
  h5_count="$(find "${DATASET_ROOT}" -type f -name '*.h5' -print | wc -l | tr -d ' ')"
  [[ "${h5_count}" == "${FASTWAM_LOCAL_EXPECTED_H5_FILES}" ]] || \
    die "node-local dataset must contain exactly ${FASTWAM_LOCAL_EXPECTED_H5_FILES} H5 files, observed ${h5_count}"
fi

"${PYTHON_BIN}" - \
  "${REPO_ROOT}/configs/task/${TASK_PROFILE}.yaml" \
  "${REPO_ROOT}/configs/scale/${SCALE_PROFILE}.yaml" <<'PY' || die "formal POSE_FOCUS task/scale contract validation failed"
import pathlib
import sys

import yaml

task_path, scale_path = map(pathlib.Path, sys.argv[1:])
task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
scale = yaml.safe_load(scale_path.read_text(encoding="utf-8"))
is_gaussian_spatial = task_path.stem in {
    "robofactory_placefood_gaussian_spatial_p4_224_5e-6",
    "robofactory_placefood_spatial_semantic_p6_224_5e-6",
}
is_semantic_phase = task_path.stem in {
    "robofactory_placefood_semantic_phase_p5_224_5e-6",
    "robofactory_placefood_spatial_semantic_p6_224_5e-6",
}
contract_task = task
if is_gaussian_spatial:
    parent_name = (
        "robofactory_placefood_semantic_phase_p5_224_5e-6.yaml"
        if is_semantic_phase
        else "robofactory_placefood_pose_focus_r5_224_5e-6.yaml"
    )
    parent_path = task_path.with_name(parent_name)
    contract_task = yaml.safe_load(parent_path.read_text(encoding="utf-8"))

def value_at(mapping, dotted):
    value = mapping
    for part in dotted.split("."):
        value = value[part]
    return value

task_expected = {
    "trainable_scope": "action",
    "learning_rate": 5.0e-6,
    "max_steps": 1000,
    "save_every": 500,
    "eval_every": 500,
    "offline_eval_num_samples": 24,
    "resume": "${oc.env:FASTWAM_POSE_FOCUS_BASE_CHECKPOINT}",
    "weights_only_warm_start.enabled": True,
    "weights_only_warm_start.expected_source_training_mode": "action_only_cache",
    "weights_only_warm_start.expected_source_trainable_scope": "action",
    "weights_only_warm_start.expected_source_state_kind": "full",
    "data.train.load_future_video": False,
    "data.val.load_future_video": False,
    "data.train.required_agent_counts": [2],
    "data.val.required_agent_counts": [2],
    "data.train.required_task_names": ["PlaceFood-rf"],
    "data.val.required_task_names": ["PlaceFood-rf"],
    "data.train.gaussian_cache_verify": "stat_cmp",
    "data.val.gaussian_cache_verify": "stat_cmp",
    "data.train.gaussian_cache_expected_manifest_sha256": None,
    "data.train.gaussian_cache_expected_selection_sha256": None,
    "data.train.gaussian_cache_expected_source_identity_sha256": None,
    "data.val.gaussian_cache_expected_manifest_sha256": None,
    "data.val.gaussian_cache_expected_selection_sha256": None,
    "data.val.gaussian_cache_expected_source_identity_sha256": None,
    "model.training_mode": "action_only_cache",
    "model.loss.lambda_video": 0.0,
    "model.loss.lambda_action": 1.0,
    "model.loss.pose_focus.enabled": True,
    "model.loss.pose_focus.active_agent_id": 0,
    "model.loss.pose_focus.active_arm_weight": 4.0,
    "model.loss.pose_focus.other_arm_weight": 1.0,
    "model.loss.pose_focus.gripper_weight": 1.0,
    "model.loss.pose_focus.first_steps": 5,
    "model.loss.pose_focus.first_steps_weight": 2.0,
    "model.loss.pose_focus.gripper_dim": 7,
    "model.loss.pose_focus.clean_arm_huber_beta": 0.1,
}
if task_path.stem in {
    "robofactory_placefood_pose_phase_x0_r5_224_5e-6",
    "robofactory_placefood_semantic_phase_p5_224_5e-6",
    "robofactory_placefood_spatial_semantic_p6_224_5e-6",
}:
    task_expected.update({
        "phase_balanced_fraction": 0.5,
        "data.train.b4_phase_agent_id": 0,
        "data.val.b4_phase_agent_id": 0,
        "model.loss.pose_focus.lambda_clean_arm_x0": 1.0,
    })
    if is_semantic_phase:
        task_expected.update({
            "data.train.phase_label_source": "placefood_task_state",
            "data.val.phase_label_source": "placefood_task_state",
            "data.train.placefood_lift_threshold": 0.03,
            "data.val.placefood_lift_threshold": 0.03,
            "data.train.placefood_target_xy_threshold": 0.10,
            "data.val.placefood_target_xy_threshold": 0.10,
            "data.train.placefood_release_command_threshold": 0.0,
            "data.val.placefood_release_command_threshold": 0.0,
        })
else:
    task_expected["model.loss.pose_focus.lambda_clean_arm_x0"] = 0.0
scale_expected = {
    "gradient_accumulation_steps": 1,
    "checkpoint_state_kind": "full",
    "save_training_state": True,
    "save_final_checkpoint": True,
    "seal_training_state": False,
    "seal_training_run": False,
    "terminal_rehash_weights": False,
}
for label, mapping, expected in (
    ("task", contract_task, task_expected),
    ("scale", scale, scale_expected),
):
    for key, wanted in expected.items():
        try:
            actual = value_at(mapping, key)
        except (KeyError, TypeError) as exc:
            raise SystemExit(f"{label} profile is missing {key}: {exc}")
        if actual != wanted:
            raise SystemExit(f"{label} profile drift at {key}: expected {wanted!r}, got {actual!r}")
if is_gaussian_spatial:
    gaussian_expected = {
        "weights_only_warm_start.architecture_upgrade": "gaussian_spatial_v2_from_pooled_v1",
        "model.action_dit_config.gaussian_conditioning_mode": "spatial_cross_attention",
        "model.action_dit_config.gaussian_residual_floor": 0.1,
        "model.action_dit_config.gaussian_attention_temperature": 0.1,
    }
    for key, wanted in gaussian_expected.items():
        try:
            actual = value_at(task, key)
        except (KeyError, TypeError) as exc:
            raise SystemExit(f"Gaussian spatial profile is missing {key}: {exc}")
        if actual != wanted:
            raise SystemExit(
                f"Gaussian spatial profile drift at {key}: expected {wanted!r}, got {actual!r}"
            )
    expected_parent = (
        "robofactory_placefood_semantic_phase_p5_224_5e-6"
        if is_semantic_phase
        else "robofactory_placefood_pose_focus_r5_224_5e-6"
    )
else:
    expected_parent = "robofactory_multi_robot_vg1_hub1_gau1_224_1e-4"
defaults = task.get("defaults", [])
if expected_parent not in defaults:
    raise SystemExit(f"POSE_FOCUS task must inherit the audited profile {expected_parent}")

PY

if [[ "${TEST_MODE}" != "1" && "${DRY_RUN}" == "0" ]]; then
  # These are the same operational eRDMA/NCCL settings used by the completed
  # formal 32-GPU run.  The existing helper owns its archived bundle checks;
  # this POSE_FOCUS launcher adds no source/environment digest or hash-chain contract.
  export FASTWAM_ERDMA_BUNDLE_ROOT="${FASTWAM_ERDMA_BUNDLE_ROOT:-/oss-chengjuntao/artifacts/erdma-userspace-56.2-1.0.3}"
  export FASTWAM_ERDMA_EXPECTED_VERSION="${FASTWAM_ERDMA_EXPECTED_VERSION:-56.2-1.0.3}"
  export NCCL_IB_HCA="${NCCL_IB_HCA:-erdma}"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
  export FASTWAM_PREFLIGHT_REQUIRE_ERDMA="${FASTWAM_PREFLIGHT_REQUIRE_ERDMA:-1}"
  export FASTWAM_PREFLIGHT_TIMEOUT="${FASTWAM_PREFLIGHT_TIMEOUT:-7200}"
  export FASTWAM_PREFLIGHT_OUTER_TIMEOUT="${FASTWAM_PREFLIGHT_OUTER_TIMEOUT:-7260}"
  [[ "${FASTWAM_ERDMA_BUNDLE_ROOT}" == "/oss-chengjuntao/artifacts/erdma-userspace-56.2-1.0.3" ]] || \
    die "production eRDMA bundle root drifted from the verified artifact"
  [[ "${FASTWAM_ERDMA_EXPECTED_VERSION}" == "56.2-1.0.3" ]] || die "production eRDMA version drifted"
  [[ "${NCCL_IB_HCA}" == "erdma" ]] || die "NCCL_IB_HCA must be erdma"
  [[ "${NCCL_DEBUG}" == "INFO" ]] || die "NCCL_DEBUG must be INFO"
  [[ "${NCCL_DEBUG_SUBSYS}" == "INIT,NET" ]] || die "NCCL_DEBUG_SUBSYS must be INIT,NET"
  [[ "${FASTWAM_PREFLIGHT_REQUIRE_ERDMA}" == "1" ]] || die "eRDMA preflight must remain required"
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/docker/prepare-erdma-userspace.sh"
  fastwam_prepare_erdma_userspace || die "eRDMA userspace preparation failed"
  fastwam_run_global_allreduce_preflight \
    "${GPUS_PER_NODE}" "${NUM_MACHINES}" "${MACHINE_RANK}" \
    "${MASTER_HOST}" "${MASTER_TCP_PORT}" "${RUN_ID}" || die "global NCCL/eRDMA preflight failed"
fi

LOCAL_CACHE_ROOT="${FASTWAM_POSE_FOCUS_LOCAL_CACHE_ROOT:-/tmp/fastwam-pose_focus-checkpoints}"
LOCAL_ATTEMPT_DIR="${LOCAL_CACHE_ROOT}/${RUN_ID}/${ATTEMPT_ID}"
LOCAL_WEIGHT="${LOCAL_ATTEMPT_DIR}/step_001000.pt"
LOCAL_READY="${LOCAL_ATTEMPT_DIR}/.ready"
RESERVATION="${OUTPUT_DIR}/.pose_focus-run-reservation"
OUTPUT_RESERVATION_TIMEOUT="${FASTWAM_POSE_FOCUS_OUTPUT_RESERVATION_TIMEOUT:-300}"
is_uint "${OUTPUT_RESERVATION_TIMEOUT}" || \
  die "FASTWAM_POSE_FOCUS_OUTPUT_RESERVATION_TIMEOUT must be a positive integer"
((10#${OUTPUT_RESERVATION_TIMEOUT} > 0)) || \
  die "FASTWAM_POSE_FOCUS_OUTPUT_RESERVATION_TIMEOUT must be positive"
if [[ "${TEST_MODE}" != "1" ]]; then
  [[ "${OUTPUT_RESERVATION_TIMEOUT}" == "300" ]] || \
    die "formal output reservation timeout must remain 300 seconds"
fi
RESERVATION_BODY="run_id=${RUN_ID}
attempt_id=${ATTEMPT_ID}
workers=3
gpus_per_worker=8
global_world_size=24
source_weight=${SOURCE_WEIGHT}
initialization=weights-only
optimizer=fresh
provenance_mode=stat_cmp
stage_steps=1000
"

if [[ "${DRY_RUN}" == "0" ]]; then
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  [[ "${gpu_count}" == "8" ]] || die "each worker must expose exactly 8 GPUs, observed ${gpu_count:-0}"

  if [[ "${MACHINE_RANK}" == "0" ]]; then
    [[ ! -e "${OUTPUT_DIR}" ]] || die "output already exists; RUN_ID is not unique: ${OUTPUT_DIR}"
    mkdir -- "${OUTPUT_DIR}" || die "cannot reserve output directory"
    printf '%s' "${RESERVATION_BODY}" >"${RESERVATION}"
  else
    deadline=$((SECONDS + 10#${OUTPUT_RESERVATION_TIMEOUT}))
    while [[ ! -f "${RESERVATION}" && ${SECONDS} -lt ${deadline} ]]; do
      sleep 1
    done
    [[ -f "${RESERVATION}" ]] || die "timed out waiting for rank-0 output reservation"
  fi
  [[ "$(cat -- "${RESERVATION}")" == "${RESERVATION_BODY%$'\n'}" ]] || die "output reservation belongs to a different run contract"

  [[ "${LOCAL_ATTEMPT_DIR}" == /tmp/* || "${TEST_MODE}" == "1" ]] || \
    die "checkpoint cache must remain on node-local /tmp"
  [[ ! -e "${LOCAL_ATTEMPT_DIR}" && ! -L "${LOCAL_ATTEMPT_DIR}" ]] || \
    die "node-local attempt cache already exists: ${LOCAL_ATTEMPT_DIR}"
  LOCAL_ATTEMPT_PARENT="${LOCAL_CACHE_ROOT}/${RUN_ID}"
  PARTIAL_ATTEMPT="${LOCAL_ATTEMPT_DIR}.partial.${BASHPID}"
  [[ ! -e "${PARTIAL_ATTEMPT}" && ! -L "${PARTIAL_ATTEMPT}" ]] || \
    die "node-local partial checkpoint cache already exists: ${PARTIAL_ATTEMPT}"
  mkdir -p -- "${LOCAL_ATTEMPT_PARENT}"
  mkdir -- "${PARTIAL_ATTEMPT}"
  PARTIAL_WEIGHT="${PARTIAL_ATTEMPT}/step_001000.pt"
  SOURCE_STAT_BEFORE="$(stat -Lc '%d:%i:%s:%Y' -- "${SOURCE_WEIGHT}")"
  SOURCE_MTIME="$(stat -Lc '%Y' -- "${SOURCE_WEIGHT}")"
  cp -- "${SOURCE_WEIGHT}" "${PARTIAL_WEIGHT}" || die "failed to copy R5 weight to node-local storage"
  [[ -f "${PARTIAL_WEIGHT}" && ! -L "${PARTIAL_WEIGHT}" ]] || \
    die "node-local checkpoint copy is not a regular non-symlink file"
  [[ "$(stat -c '%s' -- "${PARTIAL_WEIGHT}")" == "${EXPECTED_WEIGHT_BYTES}" ]] || \
    die "node-local checkpoint copy has the wrong byte count"
  [[ "$(stat -Lc '%d:%i:%s:%Y' -- "${SOURCE_WEIGHT}")" == "${SOURCE_STAT_BEFORE}" ]] || \
    die "R5 source weight changed while staging"
  cmp -s -- "${SOURCE_WEIGHT}" "${PARTIAL_WEIGHT}" || \
    die "node-local checkpoint bytes differ from the R5 source"
  [[ "$(stat -Lc '%d:%i:%s:%Y' -- "${SOURCE_WEIGHT}")" == "${SOURCE_STAT_BEFORE}" ]] || \
    die "R5 source weight changed during byte comparison"
  printf 'provenance_mode=stat_cmp\nrun_id=%s\nattempt_id=%s\nsource_path=%s\ndestination_path=%s\nbytes=%s\nsource_mtime_epoch=%s\nfile_count=1\n' \
    "${RUN_ID}" "${ATTEMPT_ID}" "${SOURCE_WEIGHT}" "${LOCAL_WEIGHT}" \
    "${EXPECTED_WEIGHT_BYTES}" "${SOURCE_MTIME}" >"${PARTIAL_ATTEMPT}/.ready"
  mv -T -- "${PARTIAL_ATTEMPT}" "${LOCAL_ATTEMPT_DIR}"
  [[ -f "${LOCAL_WEIGHT}" && -f "${LOCAL_READY}" ]] || die "node-local staging barrier was not published"
  printf 'POSE_FOCUS node staging complete: worker=%s mode=stat_cmp local_weight=%s bytes=%s mtime=%s\n' \
    "${MACHINE_RANK}" "${LOCAL_WEIGHT}" "${EXPECTED_WEIGHT_BYTES}" "${SOURCE_MTIME}"
else
  printf 'POSE_FOCUS dry-run: no output reservation, checkpoint copy, GPU query, or training was performed.\n'
fi

# POSE_FOCUS binds the existing Gaussian cache by its audited path and embedded
# metadata.  The formal POSE_FOCUS task intentionally disables the old SHA env gates.
export FASTWAM_GAUSSIAN_CACHE_DIR="${GAUSSIAN_CACHE_DIR}"
export FASTWAM_POSE_FOCUS_BASE_CHECKPOINT="${LOCAL_WEIGHT}"

COMMAND=(
  "${PYTHON_BIN}" -m accelerate.commands.launch
  --config_file "${REPO_ROOT}/scripts/accelerate_configs/accelerate_zero2_ds.yaml"
  --num_machines 3
  --machine_rank "${MACHINE_RANK}"
  --main_process_ip "${MASTER_HOST}"
  --main_process_port "${MASTER_TCP_PORT}"
  --num_processes 24
  --deepspeed_multinode_launcher standard
  "${REPO_ROOT}/scripts/train.py"
  "task=${TASK_PROFILE}"
  "+scale=${SCALE_PROFILE}"
  "data.train.root_dir=${DATASET_ROOT}"
  "data.val.root_dir=${DATASET_ROOT}"
  "data.train.pretrained_norm_stats=${STATS_PATH}"
  "data.val.pretrained_norm_stats=${STATS_PATH}"
  "data.train.stats_source_root=${STATS_SOURCE_ROOT}"
  "data.val.stats_source_root=${STATS_SOURCE_ROOT}"
  "data.train.text_embedding_cache_dir=${TEXT_CACHE_DIR}"
  "data.val.text_embedding_cache_dir=${TEXT_CACHE_DIR}"
  "data.train.gaussian_cache_dir=${GAUSSIAN_CACHE_DIR}"
  "data.val.gaussian_cache_dir=${GAUSSIAN_CACHE_DIR}"
  "output_dir=${OUTPUT_DIR}"
  wandb.name="${RUN_ID}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'POSE_FOCUS resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

# PAI's outer-worker topology describes three node coordinators, whereas
# Accelerate is about to construct a fresh 24-process topology.  Keeping the
# injected rank variables would make child rank discovery ambiguous.  All
# values needed by Accelerate are already frozen into COMMAND above.
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK NODE_RANK

cd -- "${REPO_ROOT}"
exec "${COMMAND[@]}"
