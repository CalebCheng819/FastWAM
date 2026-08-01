#!/usr/bin/env bash
set -euo pipefail

# Run this command on all four DLC nodes. PAI WORLD_SIZE/RANK are node count
# and node rank; NPROC_PER_NODE must be eight. The two Accelerate invocations
# are separate OS process worlds, which is the checkpoint destruction boundary.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
NUM_MACHINES="${WORLD_SIZE:?PAI WORLD_SIZE is required}"
MACHINE_RANK="${RANK:?PAI RANK is required}"
NPROC_PER_NODE="${NPROC_PER_NODE:?PAI NPROC_PER_NODE is required}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is required}"
MASTER_PORT="${MASTER_PORT:?MASTER_PORT is required}"
OUTPUT_ROOT="${FASTWAM_ZERO_SMOKE_OUTPUT_ROOT:?FASTWAM_ZERO_SMOKE_OUTPUT_ROOT is required}"
PYTHON_TOOL="${FASTWAM_PYTHON:-python3}"

if [[ "${NUM_MACHINES}" != 4 || "${NPROC_PER_NODE}" != 8 ]] || \
  [[ ! "${MACHINE_RANK}" =~ ^[0-3]$ ]]; then
  echo "Error: formal OSS ZeRO-2 smoke requires exactly 4 nodes x 8 GPUs." >&2
  exit 1
fi
if [[ "${OUTPUT_ROOT}" != /oss-chengjuntao/* ]]; then
  echo "Error: FASTWAM_ZERO_SMOKE_OUTPUT_ROOT must be under /oss-chengjuntao." >&2
  exit 1
fi
if ! PYTHON_TOOL="$(command -v "${PYTHON_TOOL}")"; then
  echo "Error: FASTWAM_PYTHON is not executable: ${FASTWAM_PYTHON:-python3}" >&2
  exit 1
fi
export FASTWAM_PYTHON="${PYTHON_TOOL}"
if [[ ! "${FASTWAM_CODE_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || \
  [[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" != "${FASTWAM_CODE_COMMIT}" ]]; then
  echo "Error: FASTWAM_CODE_COMMIT must equal the exact current Git HEAD." >&2
  exit 1
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Error: the real ZeRO-2 smoke requires a clean immutable Git worktree." >&2
  exit 1
fi
: "${FASTWAM_DLC_IMAGE_REFERENCE:?FASTWAM_DLC_IMAGE_REFERENCE is required}"
if [[ ! "${FASTWAM_DLC_IMAGE_DIGEST:-}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Error: the OSS authorization smoke requires an exact OCI image digest." >&2
  exit 1
fi

reservation="${OUTPUT_ROOT}/.SMOKE_RESERVED"
if ((10#${MACHINE_RANK} == 0)); then
  if ! mkdir -m 0750 -- "${OUTPUT_ROOT}"; then
    echo "Error: smoke output root already exists and is never reused: ${OUTPUT_ROOT}" >&2
    exit 1
  fi
  printf '%s\n' "${FASTWAM_CODE_COMMIT}" >"${reservation}"
else
  deadline=$((SECONDS + 600))
  until [[ -f "${reservation}" ]]; do
    if ((SECONDS >= deadline)); then
      echo "Error: timed out waiting for rank-0 smoke reservation ${reservation}" >&2
      exit 1
    fi
    sleep 1
  done
fi
if [[ "$(<"${reservation}")" != "${FASTWAM_CODE_COMMIT}" ]]; then
  echo "Error: smoke reservation code identity mismatch." >&2
  exit 1
fi

source "${REPO_ROOT}/docker/prepare-erdma-userspace.sh"
fastwam_prepare_erdma_userspace
export NCCL_IB_HCA="${NCCL_IB_HCA:-erdma}"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

cd "${REPO_ROOT}"
launch_phase() {
  local phase="$1"
  env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    -u GROUP_RANK -u ROLE_RANK \
    "${PYTHON_TOOL}" -m accelerate.commands.launch \
    --config_file "${REPO_ROOT}/scripts/accelerate_configs/accelerate_zero2_ds.yaml" \
    --num_machines "${NUM_MACHINES}" \
    --machine_rank "${MACHINE_RANK}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --num_processes 32 \
    --deepspeed_multinode_launcher standard \
    "${REPO_ROOT}/scripts/zero2_checkpoint_roundtrip.py" \
    --phase "${phase}" \
    --output-root "${OUTPUT_ROOT}"
}

launch_phase save
launch_phase load

marker="${OUTPUT_ROOT}/zero2-roundtrip-smoke.json"
if ((10#${MACHINE_RANK} == 0)); then
  "${PYTHON_TOOL}" "${REPO_ROOT}/scripts/zero2_checkpoint_roundtrip.py" \
    --phase finalize \
    --output-root "${OUTPUT_ROOT}"
else
  deadline=$((SECONDS + 7200))
  until [[ -f "${marker}" ]]; do
    if ((SECONDS >= deadline)); then
      echo "Error: timed out waiting for rank-0 ZeRO-2 smoke marker ${marker}" >&2
      exit 1
    fi
    sleep 1
  done
fi

marker_sha256="$(sha256sum -- "${marker}")"
marker_sha256="${marker_sha256%% *}"
"${PYTHON_TOOL}" "${REPO_ROOT}/scripts/validate_zero_checkpoint_smoke.py" \
  --marker "${marker}" \
  --expected-sha256 "${marker_sha256}" \
  --output-parent "${OUTPUT_ROOT}"
