#!/usr/bin/env bash
set -euo pipefail

# Run on all four DLC worker pods after the immutable offline bootstrap has
# exported FASTWAM_PYTHON and FASTWAM_REPO_ROOT. The two launcher calls create
# separate Accelerate/DeepSpeed process worlds, which is the destruction and
# restoration boundary for the real full-model state.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
NUM_MACHINES="${WORLD_SIZE:?PAI WORLD_SIZE is required}"
MACHINE_RANK="${RANK:?PAI RANK is required}"
NPROC_PER_NODE="${NPROC_PER_NODE:?PAI NPROC_PER_NODE is required}"
OUTPUT_ROOT="${FASTWAM_FORMAL_OUTPUT_DIR:?FASTWAM_FORMAL_OUTPUT_DIR is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
PYTHON_TOOL="${FASTWAM_PYTHON:-python3}"

if [[ "${NUM_MACHINES}" != 4 || "${NPROC_PER_NODE}" != 8 ]] || \
  [[ ! "${MACHINE_RANK}" =~ ^[0-3]$ ]]; then
  echo "Error: N=4 full-model gate requires exactly 4 nodes x 8 GPUs." >&2
  exit 1
fi
if [[ "${OUTPUT_ROOT}" != /oss-chengjuntao/* ]] || \
  [[ "${OUTPUT_ROOT##*/}" != "${RUN_ID}" ]]; then
  echo "Error: gate output must be a unique /oss-chengjuntao path ending in RUN_ID." >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}/COMPLETE" || -L "${OUTPUT_ROOT}/COMPLETE" ]]; then
  echo "Error: gate output is already terminal and is never reused: ${OUTPUT_ROOT}" >&2
  exit 1
fi
if ! PYTHON_TOOL="$(command -v "${PYTHON_TOOL}")"; then
  echo "Error: FASTWAM_PYTHON is not executable: ${FASTWAM_PYTHON:-python3}" >&2
  exit 1
fi
export FASTWAM_PYTHON="${PYTHON_TOOL}"
if [[ ! "${FASTWAM_CODE_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || \
  [[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" != "${FASTWAM_CODE_COMMIT}" ]]; then
  echo "Error: FASTWAM_CODE_COMMIT must equal the exact gate-runner Git HEAD." >&2
  exit 1
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Error: N=4 full-model gate requires a clean immutable Git worktree." >&2
  exit 1
fi
for bypass_name in \
  FASTWAM_LAUNCHER_UNIT_TEST_ALLOW_DIRTY \
  FASTWAM_LAUNCHER_UNIT_TEST_SKIP_ENV_PREFLIGHT; do
  case "${!bypass_name:-0}" in
    0 | false | no | off) ;;
    *)
      echo "Error: N=4 full-model gate forbids ${bypass_name}." >&2
      exit 1
      ;;
  esac
done

# Make the immutable scientific bindings explicit in the final marker. The
# launcher independently verifies the corresponding files and source bundles.
export FASTWAM_OFFICIAL_CHECKPOINT_SHA256="${FASTWAM_OFFICIAL_CHECKPOINT_SHA256:-1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579}"
export FASTWAM_VAE_SHA256="${FASTWAM_VAE_SHA256:-0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996}"
export FASTWAM_STATS_SHA256="${FASTWAM_STATS_SHA256:-350493b685d8db0ea4cfd66f58f49849e8cd1f65cecc269f15aff9101ac8a04d}"
for required_name in \
  FASTWAM_CPFS_BUNDLE_MANIFEST_SHA256 \
  FASTWAM_OSS_BUNDLE_MANIFEST_SHA256 \
  FASTWAM_GAUSSIAN_CACHE_MANIFEST_SHA256 \
  FASTWAM_GAUSSIAN_CACHE_SELECTION_SHA256 \
  FASTWAM_GAUSSIAN_CACHE_SOURCE_IDENTITY_SHA256 \
  FASTWAM_TRAINING_ENV_BUNDLE_MANIFEST_SHA256 \
  FASTWAM_OUTPUT_ZERO_CHECKPOINT_SMOKE_SHA256 \
  FASTWAM_DLC_IMAGE_REFERENCE \
  FASTWAM_DLC_IMAGE_DIGEST; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "Error: missing required N=4 gate binding ${required_name}." >&2
    exit 1
  fi
done

gate_args=(
  task=robofactory_multi_robot_vg1_hub1_gau1_224_1e-4
  +scale=robofactory_multi_robot_32gpu_n4_fullmodel_gate
)

unset FASTWAM_N4_FULLMODEL_GATE_OUTPUT_ROOT \
  FASTWAM_N4_FULLMODEL_GATE_COMPLETE_SHA256
export FASTWAM_N4_FULLMODEL_GATE_PHASE=save
unset FASTWAM_FORMAL_RESUME_STATE_DIR \
  FASTWAM_FORMAL_RESUME_STATE_MANIFEST \
  FASTWAM_FORMAL_RESUME_STATE_MANIFEST_SHA256 \
  FASTWAM_FORMAL_RESUME_TRAINER_STATE_SHA256
bash "${REPO_ROOT}/scripts/train_zero2.sh" 8 "${gate_args[@]}"

state_root="${OUTPUT_ROOT}/checkpoints/state/step_000002"
state_manifest="${OUTPUT_ROOT}/checkpoints/state/step_000002.state-tree.json"
trainer_state="${state_root}/trainer_state.json"
for required_path in "${state_manifest}" "${trainer_state}"; do
  if [[ -L "${required_path}" || ! -f "${required_path}" ]]; then
    echo "Error: save phase did not publish a regular full-state artifact: ${required_path}" >&2
    exit 1
  fi
done
state_manifest_sha256="$(sha256sum -- "${state_manifest}")"
state_manifest_sha256="${state_manifest_sha256%% *}"
trainer_state_sha256="$(sha256sum -- "${trainer_state}")"
trainer_state_sha256="${trainer_state_sha256%% *}"

export FASTWAM_N4_FULLMODEL_GATE_PHASE=load
export FASTWAM_FORMAL_RESUME_STATE_DIR="${state_root}"
export FASTWAM_FORMAL_RESUME_STATE_MANIFEST="${state_manifest}"
export FASTWAM_FORMAL_RESUME_STATE_MANIFEST_SHA256="${state_manifest_sha256}"
export FASTWAM_FORMAL_RESUME_TRAINER_STATE_SHA256="${trainer_state_sha256}"
bash "${REPO_ROOT}/scripts/train_zero2.sh" 8 "${gate_args[@]}"

if ((10#${MACHINE_RANK} == 0)); then
  "${PYTHON_TOOL}" "${REPO_ROOT}/scripts/finalize_n4_fullmodel_gate.py" \
    --phase finalize \
    --output-root "${OUTPUT_ROOT}"
else
  deadline=$((SECONDS + 21600))
  until [[ -f "${OUTPUT_ROOT}/COMPLETE" ]]; do
    if [[ -L "${OUTPUT_ROOT}/GATE.FAILED.json" ]]; then
      echo "Error: gate failure marker must not be a symlink." >&2
      exit 1
    fi
    if [[ -f "${OUTPUT_ROOT}/GATE.FAILED.json" ]]; then
      "${PYTHON_TOOL}" - "${OUTPUT_ROOT}/GATE.FAILED.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
raise SystemExit(
    "N=4 gate finalizer failed on rank zero: "
    f"{payload.get('error_type')}: {payload.get('error_message')}"
)
PY
    fi
    if [[ -L "${OUTPUT_ROOT}/COMPLETE" ]]; then
      echo "Error: gate COMPLETE must not be a symlink." >&2
      exit 1
    fi
    if ((SECONDS >= deadline)); then
      echo "Error: timed out waiting for N=4 gate COMPLETE." >&2
      exit 1
    fi
    sleep 1
  done
fi

"${PYTHON_TOOL}" "${REPO_ROOT}/scripts/finalize_n4_fullmodel_gate.py" \
  --phase validate \
  --output-root "${OUTPUT_ROOT}"
