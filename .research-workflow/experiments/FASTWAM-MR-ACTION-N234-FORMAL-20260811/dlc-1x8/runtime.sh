#!/usr/bin/env bash
# Trusted worker runtime for action_only_native_agents_1x8_v1.
#
# This script deliberately keeps every trainer write on local POSIX scratch.
# The durable output is created only after train -> resume -> fresh-load and
# all local evidence have passed.  Durable publication is exclusive-create,
# streaming, close/reopen byte comparison, with COMPLETE written last.

set -eEuo pipefail
IFS=$'\n\t'
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() {
  echo "[formal-runtime] ERROR: $*" >&2
  exit 1
}

required_env=(
  FASTWAM_AGENT_COUNT FASTWAM_ARTIFACT_INTEGRITY_MODE FASTWAM_CODE_COMMIT
  FASTWAM_DATASET_ROOT FASTWAM_EXPERIMENT_ID FASTWAM_EXTERNAL_CONTRACT
  FASTWAM_GAUSSIAN_CACHE_DIR FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR
  FASTWAM_INITIAL_CHECKPOINT FASTWAM_MEMBER FASTWAM_MAX_OSS_PUBLISH_BYTES
  FASTWAM_MIN_TMP_FREE_BYTES FASTWAM_OSS_OUTPUT_ROOT
  FASTWAM_PREPARED_RESERVATION_PATH FASTWAM_PYTHON FASTWAM_PYTHON_TARGET
  FASTWAM_RUN_ID
  FASTWAM_SOURCE_ROOT FASTWAM_STATS_SOURCE
  FASTWAM_SUITE_STORAGE_RESERVATION_PATH FASTWAM_TASK_CONFIG
  FASTWAM_TASKS_JSON FASTWAM_TEXT_CACHE_MAP_HYDRA
  FASTWAM_TEXT_CACHE_MAP_JSON FASTWAM_VAE_SOURCE
  NPROC_PER_NODE
)
for name in "${required_env[@]}"; do
  [[ -n "${!name:-}" ]] || die "required environment is absent: ${name}"
done

[[ "${FASTWAM_EXTERNAL_CONTRACT}" == action_only_native_agents_1x8_v1 ]] || die "external contract mismatch"
[[ "${FASTWAM_ARTIFACT_INTEGRITY_MODE}" == metadata_no_hash ]] || die "integrity mode mismatch"
[[ "${NPROC_PER_NODE}" == 8 ]] || die "this runtime requires exactly eight ranks"
[[ "${FASTWAM_MAX_OSS_PUBLISH_BYTES}" == $((62 * 1024 * 1024 * 1024)) ]] || die "per-run publication cap mismatch"
[[ "${FASTWAM_MIN_TMP_FREE_BYTES}" == $((200 * 1024 * 1024 * 1024)) ]] || die "local scratch floor mismatch"
[[ "${FASTWAM_OSS_OUTPUT_ROOT}" == /oss-chengjuntao/artifacts/fastwam-action-n234-formal-r5-20260812/* ]] || die "durable output prefix mismatch"
[[ -d "$(dirname -- "${FASTWAM_OSS_OUTPUT_ROOT}")" && ! -L "$(dirname -- "${FASTWAM_OSS_OUTPUT_ROOT}")" ]] || die "prepared durable output prefix is absent"
[[ "${FASTWAM_SOURCE_ROOT}" == /oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/fastwam-action-n234-formal-r5-20260812-r1 ]] || die "frozen R5 source mismatch"
[[ "${FASTWAM_PREPARED_RESERVATION_PATH}" == /oss-chengjuntao/* ]] || die "reservation must be on OSS"
[[ "${FASTWAM_SUITE_STORAGE_RESERVATION_PATH}" == /oss-chengjuntao/* ]] || die "suite reservation must be on OSS"
[[ "${FASTWAM_DATASET_ROOT}" == /oss-chengjuntao/* ]] || die "dataset must be sourced from OSS"
[[ "${FASTWAM_INITIAL_CHECKPOINT}" == /oss-chengjuntao/* ]] || die "initial checkpoint must be sourced from OSS"
[[ "${FASTWAM_STATS_SOURCE}" == /oss-chengjuntao/* ]] || die "stats must be sourced from OSS"
[[ "${FASTWAM_VAE_SOURCE}" == /oss-chengjuntao/* ]] || die "VAE must be sourced from OSS"
[[ "${FASTWAM_GAUSSIAN_CACHE_DIR}" == /oss-chengjuntao/* ]] || die "Gaussian cache must be sourced from OSS"
[[ "${FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR}" == /oss-chengjuntao/* ]] || die "Gaussian fallback must be sourced from OSS"
[[ "${FASTWAM_PYTHON}" == /cpfs/user/chengjuntao/venvs/fastwam-gaudp-py310-20260802/bin/python ]] || die "pinned Python mismatch"
[[ "${FASTWAM_PYTHON_TARGET}" == /cpfs/user/chengjuntao/runtimes/uv-python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10 ]] || die "pinned Python target contract mismatch"
[[ -L "${FASTWAM_PYTHON}" && -x "${FASTWAM_PYTHON}" ]] || die "pinned venv Python link is unavailable"
resolved_python="$(readlink -f -- "${FASTWAM_PYTHON}")" || die "cannot resolve pinned venv Python"
[[ "${resolved_python}" == "${FASTWAM_PYTHON_TARGET}" ]] || die "pinned venv Python target mismatch"
[[ -f "${FASTWAM_PYTHON_TARGET}" && -x "${FASTWAM_PYTHON_TARGET}" && ! -L "${FASTWAM_PYTHON_TARGET}" ]] || die "resolved pinned Python is not a regular executable"
[[ ! -e "${FASTWAM_OSS_OUTPUT_ROOT}" && ! -L "${FASTWAM_OSS_OUTPUT_ROOT}" ]] || die "unique durable output already exists"

case "${FASTWAM_MEMBER}" in
  n2)
    expected_agents=2
    expected_experiment=FASTWAM-MR-FT-ACT-N2-PLACEFOOD-1K-S42-R5-20260812
    expected_run=fastwam-act-n2-placefood-1k-s42-r5-20260812
    expected_config=robofactory_multi_robot_ft_n2_placefood_vg0_hub1_gau1_224_3e-5
    expected_tasks='["PlaceFood-rf"]'
    ;;
  n3)
    expected_agents=3
    expected_experiment=FASTWAM-MR-FT-ACT-N3-POOL-1K-S42-R5-20260812
    expected_run=fastwam-act-n3-pool-1k-s42-r5-20260812
    expected_config=robofactory_multi_robot_ft_n3_pool_vg0_hub1_gau1_224_3e-5
    expected_tasks='["ThreeRobotsPlaceShoes-rf","ThreeRobotsStackCube-rf"]'
    ;;
  n4)
    expected_agents=4
    expected_experiment=FASTWAM-MR-FT-ACT-N4-STACKCUBE-1K-S42-R5-20260812
    expected_run=fastwam-act-n4-stackcube-1k-s42-r5-20260812
    expected_config=robofactory_multi_robot_ft_n4_stackcube_vg0_hub1_gau1_224_3e-5
    expected_tasks='["FourRobotsStackCube-rf"]'
    ;;
  *) die "unknown suite member" ;;
esac
[[ "${FASTWAM_AGENT_COUNT}" == "${expected_agents}" ]] || die "native agent count mismatch"
[[ "${FASTWAM_EXPERIMENT_ID}" == "${expected_experiment}" ]] || die "experiment identity mismatch"
[[ "${FASTWAM_RUN_ID}" == "${expected_run}" ]] || die "run identity mismatch"
[[ "${FASTWAM_TASK_CONFIG}" == "${expected_config}" ]] || die "task config mismatch"
[[ "${FASTWAM_TASKS_JSON}" == "${expected_tasks}" ]] || die "task scope mismatch"

for input in \
  "${FASTWAM_SOURCE_ROOT}" "${FASTWAM_DATASET_ROOT}" \
  "${FASTWAM_GAUSSIAN_CACHE_DIR}" "${FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR}"; do
  [[ -d "${input}" && ! -L "${input}" ]] || die "required input directory is invalid: ${input}"
done
for input in \
  "${FASTWAM_INITIAL_CHECKPOINT}" "${FASTWAM_STATS_SOURCE}" \
  "${FASTWAM_VAE_SOURCE}" "${FASTWAM_PREPARED_RESERVATION_PATH}" \
  "${FASTWAM_SUITE_STORAGE_RESERVATION_PATH}"; do
  [[ -f "${input}" && ! -L "${input}" ]] || die "required input file is invalid: ${input}"
done

CONTROLLER_REL=.research-workflow/experiments/FASTWAM-MR-ACTION-N234-FORMAL-20260811/dlc-1x8/controller.py
SOURCE_CONTROLLER="${FASTWAM_SOURCE_ROOT}/${CONTROLLER_REL}"
[[ -f "${SOURCE_CONTROLLER}" && ! -L "${SOURCE_CONTROLLER}" ]] || die "frozen controller is absent from source"

# Re-read the immutable suite and all three member reservations plus every
# input metadata descriptor immediately before local execution.
"${FASTWAM_PYTHON}" -B -I -S - "${SOURCE_CONTROLLER}" "${FASTWAM_MEMBER}" "${FASTWAM_PREPARED_RESERVATION_PATH}" <<'PY'
import importlib.util
import sys
from pathlib import Path

controller_path, member, reservation_path = sys.argv[1:]
spec = importlib.util.spec_from_file_location("formal_worker_controller", controller_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen controller")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
reservation, _ = module.read_json(Path(reservation_path))
module.validate_reservation_live(member, reservation, require_output_absent=True)
print("[formal-runtime] reservation and all live inputs: PASS")
PY

visible_count="$("${FASTWAM_PYTHON}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
[[ "${visible_count}" == 8 ]] || die "worker does not expose exactly eight GPUs"

tmp_available="$(df -PB1 /tmp | awk 'NR==2 {print $4}')"
[[ "${tmp_available}" =~ ^[0-9]+$ ]] || die "cannot read local scratch capacity"
(( tmp_available >= FASTWAM_MIN_TMP_FREE_BYTES )) || die "local scratch is below the 200 GiB floor"

SCRATCH="$(mktemp -d /tmp/fastwam-action-n234-formal-r5.XXXXXX)"
LOCAL_SOURCE="${SCRATCH}/source"
TRAIN_OUTPUT="${SCRATCH}/train"
VERIFY_OUTPUT="${SCRATCH}/fresh-load"
PHASE1_LOG="${SCRATCH}/phase1.log"
PHASE2_LOG="${SCRATCH}/phase2.log"
PHASE3_LOG="${SCRATCH}/phase3.log"
cleanup() {
  if [[ "${FASTWAM_KEEP_LOCAL_SCRATCH:-0}" != 1 ]]; then
    rm -rf -- "${SCRATCH}"
  else
    echo "[formal-runtime] preserving local scratch: ${SCRATCH}" >&2
  fi
}
trap cleanup EXIT

mkdir -m 0700 "${LOCAL_SOURCE}"
cp -a -- "${FASTWAM_SOURCE_ROOT}/." "${LOCAL_SOURCE}/"
"${FASTWAM_PYTHON}" -B -I -S - "${SOURCE_CONTROLLER}" "${FASTWAM_MEMBER}" "${FASTWAM_PREPARED_RESERVATION_PATH}" "${FASTWAM_SOURCE_ROOT}" "${LOCAL_SOURCE}" <<'PY'
import importlib.util
import sys
from pathlib import Path

controller_path, member, reservation_literal, source_literal, target_literal = sys.argv[1:]
spec = importlib.util.spec_from_file_location("formal_stage_controller", controller_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen controller for staged source validation")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
reservation, _ = module.read_json(Path(reservation_literal))
if not isinstance(reservation, dict) or reservation.get("member") != member:
    raise RuntimeError("prepared reservation member mismatch during staged source validation")
source = reservation.get("source")
if not isinstance(source, dict) or source.get("root") != source_literal:
    raise RuntimeError("prepared reservation source root mismatch during staged source validation")
expected = source.get("inventory")
observed = module.source_inventory(Path(target_literal))
module.assert_source_inventory_matches(
    expected, observed, label="staged source portable content mismatch"
)
print("[formal-runtime] staged source portable content readback: PASS")
PY

[[ -f "${LOCAL_SOURCE}/scripts/accelerate_configs/accelerate_zero2_ds.yaml" ]] || die "Accelerate config absent"
[[ -f "${LOCAL_SOURCE}/scripts/ds_configs/ds_zero2_config.json" ]] || die "DeepSpeed config absent"
[[ -f "${LOCAL_SOURCE}/configs/task/${FASTWAM_TASK_CONFIG}.yaml" ]] || die "task config absent"

MODEL_CACHE_ROOT="${FASTWAM_VAE_SOURCE%/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors}"
[[ "${MODEL_CACHE_ROOT}" != "${FASTWAM_VAE_SOURCE}" && -d "${MODEL_CACHE_ROOT}" ]] || die "VAE does not identify the model-cache root"

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_CACHE_ROOT}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${LOCAL_SOURCE}/src"
export TMPDIR="${SCRATCH}/tmp"
export HF_HOME="${SCRATCH}/hf"
export TORCH_HOME="${SCRATCH}/torch"
export XDG_CACHE_HOME="${SCRATCH}/cache"
mkdir -m 0700 "${TMPDIR}" "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

# The controller deliberately starts the worker with a clean import
# environment.  Bind imports back to this exact staged source tree and reject
# an installed/editable fastwam from any other checkout before launching ranks.
"${FASTWAM_PYTHON}" - "${LOCAL_SOURCE}/src/fastwam" <<'PY'
import sys
from pathlib import Path

import fastwam

expected = Path(sys.argv[1]).resolve(strict=True)
literal = getattr(fastwam, "__file__", None)
if not isinstance(literal, str):
    raise RuntimeError("fastwam package has no regular module origin")
origin = Path(literal).resolve(strict=True)
if not origin.is_file() or not origin.is_relative_to(expected):
    raise RuntimeError(
        f"fastwam import escaped staged source: origin={origin} expected={expected}"
    )
print(f"[formal-runtime] frozen fastwam import: {origin}")
PY

COMMON_OVERRIDES=(
  "task=${FASTWAM_TASK_CONFIG}"
  "max_steps=1000"
  "save_every=500"
  "eval_every=500"
  "offline_eval_num_samples=32"
  "seed=42"
  "checkpoint_state_kind=full"
  "seal_training_state=false"
  "seal_training_run=false"
  "terminal_rehash_weights=false"
  "training_terminal_contract=null"
  "training_run_profile=null"
  "training_task_scope_receipt=null"
  "wandb.enabled=false"
  "+artifact_integrity_mode=metadata_no_hash"
  "+model.checkpoint_integrity_mode=metadata_no_hash"
  "+recovery_gate_stop_after_checkpoint_step=500"
  "data.train.root_dir=${FASTWAM_DATASET_ROOT}"
  "data.val.root_dir=${FASTWAM_DATASET_ROOT}"
  "data.train.split_seed=42"
  "data.val.split_seed=42"
  "data.train.required_agent_counts=[${FASTWAM_AGENT_COUNT}]"
  "data.val.required_agent_counts=[${FASTWAM_AGENT_COUNT}]"
  "data.train.required_tasks=${FASTWAM_TASKS_JSON}"
  "data.val.required_tasks=${FASTWAM_TASKS_JSON}"
  "data.train.pretrained_norm_stats=${FASTWAM_STATS_SOURCE}"
  "data.val.pretrained_norm_stats=${FASTWAM_STATS_SOURCE}"
  "data.train.text_embedding_cache_dir=null"
  "data.val.text_embedding_cache_dir=null"
  "+data.train.text_embedding_cache_files=${FASTWAM_TEXT_CACHE_MAP_HYDRA}"
  "+data.val.text_embedding_cache_files=${FASTWAM_TEXT_CACHE_MAP_HYDRA}"
  "+data.train.integrity_mode=metadata_no_hash"
  "+data.val.integrity_mode=metadata_no_hash"
  "data.train.gaussian_cache_dir=${FASTWAM_GAUSSIAN_CACHE_DIR}"
  "data.val.gaussian_cache_dir=${FASTWAM_GAUSSIAN_CACHE_DIR}"
  "data.train.gaussian_cache_verify=manifest"
  "data.val.gaussian_cache_verify=manifest"
  "data.train.gaussian_cache_expected_manifest_sha256=null"
  "data.train.gaussian_cache_expected_selection_sha256=null"
  "data.train.gaussian_cache_expected_source_identity_sha256=null"
  "data.val.gaussian_cache_expected_manifest_sha256=null"
  "data.val.gaussian_cache_expected_selection_sha256=null"
  "data.val.gaussian_cache_expected_source_identity_sha256=null"
  "+data.train.gaussian_fallback_cache_dir=${FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR}"
  "+data.val.gaussian_fallback_cache_dir=${FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR}"
  "+data.train.gaussian_fallback_projection=opacity-aware-moment-matching-cell-mean-alpha-v2"
  "+data.val.gaussian_fallback_projection=opacity-aware-moment-matching-cell-mean-alpha-v2"
)

PHASE1_ARGV=(
  "${COMMON_OVERRIDES[@]}"
  "output_dir=../train"
  "resume=null"
  "init_weights=${FASTWAM_INITIAL_CHECKPOINT}"
  "save_training_state=true"
  "save_final_checkpoint=true"
)
PHASE2_ARGV=(
  "${COMMON_OVERRIDES[@]}"
  "output_dir=../train"
  "resume=../train/checkpoints/state/step_000500"
  "init_weights=null"
  "save_training_state=true"
  "save_final_checkpoint=true"
)
PHASE3_ARGV=(
  "${COMMON_OVERRIDES[@]}"
  "output_dir=../fresh-load"
  "resume=../train/checkpoints/state/step_001000"
  "init_weights=null"
  "save_training_state=false"
  "save_final_checkpoint=false"
)
readonly -a COMMON_OVERRIDES PHASE1_ARGV PHASE2_ARGV PHASE3_ARGV

# Bind the worker's literal Bash argv vectors back to the frozen controller
# contract, then ask Hydra itself to compose all three train configs in memory.
# This runs before Accelerate or scripts/train.py and never imports the trainer.
"${FASTWAM_PYTHON}" -B -I -S - \
  "${SOURCE_CONTROLLER}" "${FASTWAM_MEMBER}" \
  "${FASTWAM_PREPARED_RESERVATION_PATH}" "${LOCAL_SOURCE}" \
  "${#PHASE1_ARGV[@]}" "${#PHASE2_ARGV[@]}" "${#PHASE3_ARGV[@]}" \
  "${PHASE1_ARGV[@]}" "${PHASE2_ARGV[@]}" "${PHASE3_ARGV[@]}" <<'PY'
import importlib.util
import sys
from pathlib import Path

(
    controller_path,
    member,
    reservation_path,
    local_source,
    phase1_count,
    phase2_count,
    phase3_count,
    *arguments,
) = sys.argv[1:]
counts = [int(phase1_count), int(phase2_count), int(phase3_count)]
if any(count <= 0 for count in counts) or sum(counts) != len(arguments):
    raise RuntimeError("runtime Hydra argv framing mismatch")
observed = []
offset = 0
for count in counts:
    observed.append(arguments[offset:offset + count])
    offset += count

spec = importlib.util.spec_from_file_location(
    "formal_hydra_preflight_controller", controller_path
)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen controller for Hydra preflight")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
reservation, _ = module.read_json(Path(reservation_path))
request = module.validate_member_reservation_structure(member, reservation)
contract = module.hydra_phase_overrides(member, request["Envs"])
if len(contract) != len(observed):
    raise RuntimeError("runtime Hydra phase count differs from controller")
supplied = [
    {
        "name": expected["name"],
        "overrides": actual,
        "expected": expected["expected"],
    }
    for expected, actual in zip(contract, observed)
]
module.validate_hydra_argv_preflight(
    member,
    request,
    source_root=Path(local_source),
    phase_overrides=supplied,
)
print("[formal-runtime] exact three-phase Hydra argv compose: PASS")
PY

launch_training() {
  local log_path=$1
  local receipt_path=$2
  shift 2
  (
    cd "${LOCAL_SOURCE}"
    [[ "${PYTHONPATH}" == "${LOCAL_SOURCE}/src" ]] || die "staged import binding changed"
    unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
    if [[ -n "${receipt_path}" ]]; then
      export FASTWAM_RECOVERY_LOAD_RECEIPT="${receipt_path}"
    else
      unset FASTWAM_RECOVERY_LOAD_RECEIPT || true
    fi
    "${FASTWAM_PYTHON}" -m accelerate.commands.launch \
      --config_file "${LOCAL_SOURCE}/scripts/accelerate_configs/accelerate_zero2_ds.yaml" \
      --num_machines 1 --machine_rank 0 --main_process_ip 127.0.0.1 \
      --main_process_port 29561 --num_processes 8 \
      --deepspeed_multinode_launcher standard \
      "${LOCAL_SOURCE}/scripts/train.py" \
      "$@"
  ) 2>&1 | tee "${log_path}"
}

mkdir -m 0700 "${TRAIN_OUTPUT}"
launch_training "${PHASE1_LOG}" "" "${PHASE1_ARGV[@]}"

STEP500_WEIGHT="${TRAIN_OUTPUT}/checkpoints/weights/step_000500.pt"
STEP500_STATE="${TRAIN_OUTPUT}/checkpoints/state/step_000500"
[[ -f "${STEP500_WEIGHT}" && ! -L "${STEP500_WEIGHT}" ]] || die "phase1 full weight is absent"
[[ -f "${STEP500_STATE}/trainer_state.json" && ! -L "${STEP500_STATE}/trainer_state.json" ]] || die "phase1 state is absent"

PHASE2_RECEIPT="${TRAIN_OUTPUT}/recovery_load_receipt.json"
[[ ! -e "${PHASE2_RECEIPT}" && ! -L "${PHASE2_RECEIPT}" ]] || die "phase2 receipt target already exists"
launch_training "${PHASE2_LOG}" "${PHASE2_RECEIPT}" "${PHASE2_ARGV[@]}"

STEP1000_WEIGHT="${TRAIN_OUTPUT}/checkpoints/weights/step_001000.pt"
STEP1000_STATE="${TRAIN_OUTPUT}/checkpoints/state/step_001000"
[[ -f "${STEP1000_WEIGHT}" && ! -L "${STEP1000_WEIGHT}" ]] || die "phase2 full weight is absent"
[[ -f "${STEP1000_STATE}/trainer_state.json" && ! -L "${STEP1000_STATE}/trainer_state.json" ]] || die "phase2 final state is absent"

# Prove from the trainer-native local manifests that both checkpoint payloads
# are full, correctly stepped metadata_no_hash weights.  These path/inode-bound
# sidecars are local evidence only and are deliberately excluded from OSS.
"${FASTWAM_PYTHON}" - "${STEP500_WEIGHT}" 500 "${STEP1000_WEIGHT}" 1000 <<'PY'
import sys
from pathlib import Path

from fastwam.nohash_artifacts import read_json, regular_file_metadata

arguments = sys.argv[1:]
if len(arguments) != 4:
    raise RuntimeError("expected exactly two weight/step pairs")
for index in range(0, len(arguments), 2):
    checkpoint = Path(arguments[index]).resolve(strict=True)
    step = int(arguments[index + 1])
    manifest_path = checkpoint.with_name(f"{checkpoint.name}.manifest.json")
    complete_path = checkpoint.with_name(f"{checkpoint.name}.COMPLETE")
    checkpoint_metadata = regular_file_metadata(checkpoint)
    manifest, manifest_metadata = read_json(manifest_path)
    complete, _ = read_json(complete_path)
    expected_manifest = {
        "schema_name": "fastwam-weights-checkpoint-metadata-no-hash",
        "schema_version": 1,
        "integrity_mode": "metadata_no_hash",
        "filename": checkpoint.name,
        "file": checkpoint_metadata,
        "global_step": step,
        "checkpoint_state_kind": "full",
    }
    expected_complete = {
        "schema_name": "fastwam-weights-checkpoint-complete-metadata-no-hash",
        "schema_version": 1,
        "integrity_mode": "metadata_no_hash",
        "manifest_filename": manifest_path.name,
        "manifest_file": manifest_metadata,
        "checkpoint_filename": checkpoint.name,
        "checkpoint_file": checkpoint_metadata,
    }
    if manifest != expected_manifest:
        raise RuntimeError(f"trainer weight manifest mismatch: {manifest_path}")
    if complete != expected_complete:
        raise RuntimeError(f"trainer weight COMPLETE mismatch: {complete_path}")
PY

"${FASTWAM_PYTHON}" - "${PHASE2_RECEIPT}" "${STEP500_STATE}" "${TRAIN_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

receipt_path, source_state, output = map(Path, sys.argv[1:])
value = json.loads(receipt_path.read_text())
expected_keys = {
    "schema_name", "schema_version", "integrity_mode",
    "accelerator_load_state_returned", "source_state_dir",
    "source_trainer_state_file", "output_dir", "restored_global_step",
    "restored_epoch", "restored_batch_in_epoch", "world_size",
}
if set(value) != expected_keys:
    raise RuntimeError("phase2 recovery receipt key set mismatch")
if (value["schema_name"] != "fastwam-recovery-load-receipt" or
        value["schema_version"] != 1 or
        value["integrity_mode"] != "metadata_no_hash" or
        value["accelerator_load_state_returned"] is not True or
        value["source_state_dir"] != str(source_state.resolve()) or
        value["output_dir"] != str(output.resolve()) or
        value["restored_global_step"] != 500 or value["world_size"] != 8):
    raise RuntimeError("phase2 recovery receipt mismatch")
if set(value["source_trainer_state_file"]) != {"path", "bytes", "mtime_ns", "dev", "ino", "mode"}:
    raise RuntimeError("phase2 trainer-state descriptor mismatch")
PY

# Recovery point 500 is local-only.  It is reclaimed only after the native
# receipt proves accelerator.load_state returned and training reached 1000.
rm -rf -- "${STEP500_STATE}"
[[ ! -e "${STEP500_STATE}" && ! -L "${STEP500_STATE}" ]] || die "local step500 state reclamation failed"

mkdir -m 0700 "${VERIFY_OUTPUT}"
PHASE3_RECEIPT="${VERIFY_OUTPUT}/recovery_load_receipt.json"
launch_training "${PHASE3_LOG}" "${PHASE3_RECEIPT}" "${PHASE3_ARGV[@]}"

# Validate terminal state/eval, phase3 zero-update behavior, and both native
# recovery receipts before the durable output directory exists.
LOCAL_EVAL_RECEIPT="${SCRATCH}/offline-eval.json"
"${FASTWAM_PYTHON}" - \
  "${TRAIN_OUTPUT}" "${VERIFY_OUTPUT}" "${STEP1000_STATE}" \
  "${PHASE3_RECEIPT}" "${FASTWAM_AGENT_COUNT}" "${FASTWAM_TASKS_JSON}" \
  "${LOCAL_EVAL_RECEIPT}" <<'PY'
import json
import math
import os
import stat
import sys
from pathlib import Path

train, verify, final_state, phase3_receipt = map(Path, sys.argv[1:5])
agent_count = int(sys.argv[5])
tasks = json.loads(sys.argv[6])
eval_target = Path(sys.argv[7])

state = json.loads((final_state / "trainer_state.json").read_text())
if state.get("global_step") != 1000:
    raise RuntimeError("final trainer state is not step1000")
records = state.get("evaluation_records")
if not isinstance(records, list):
    raise RuntimeError("final trainer state has no evaluation records")
selected = {}
for record in records:
    if not isinstance(record, dict) or record.get("step") not in (500, 1000):
        continue
    step = int(record["step"])
    if (record.get("evaluation_kind") != "multi_robot_offline_loss" or
            record.get("offline_samples") != 32 or
            record.get("offline_agent_counts") != [agent_count] or
            record.get("offline_tasks") != tasks):
        raise RuntimeError(f"offline evaluation scope mismatch at step {step}")
    for key in ("val_loss", "val_loss_action"):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError(f"invalid {key} at step {step}")
    if "val_loss_video" in record:
        raise RuntimeError("action-only evaluation unexpectedly reports video loss")
    selected[step] = record
if set(selected) != {500, 1000}:
    raise RuntimeError("both step500 and step1000 offline evaluations are required")

receipt = json.loads(phase3_receipt.read_text())
expected_keys = {
    "schema_name", "schema_version", "integrity_mode",
    "accelerator_load_state_returned", "source_state_dir",
    "source_trainer_state_file", "output_dir", "restored_global_step",
    "restored_epoch", "restored_batch_in_epoch", "world_size",
}
if set(receipt) != expected_keys:
    raise RuntimeError("phase3 recovery receipt key set mismatch")
if (receipt["schema_name"] != "fastwam-recovery-load-receipt" or
        receipt["schema_version"] != 1 or
        receipt["integrity_mode"] != "metadata_no_hash" or
        receipt["accelerator_load_state_returned"] is not True or
        receipt["source_state_dir"] != str(final_state.resolve()) or
        receipt["output_dir"] != str(verify.resolve()) or
        receipt["restored_global_step"] != 1000 or receipt["world_size"] != 8):
    raise RuntimeError("phase3 fresh-load receipt mismatch")
if set(receipt["source_trainer_state_file"]) != {"path", "bytes", "mtime_ns", "dev", "ino", "mode"}:
    raise RuntimeError("phase3 trainer-state descriptor mismatch")

for root in (verify / "checkpoints" / "weights", verify / "checkpoints" / "state"):
    if root.exists() and any(root.rglob("*")):
        raise RuntimeError("phase3 zero-update process wrote a checkpoint")

payload = {
    "schema": "fastwam-multi-robot-offline-eval-receipt-v1",
    "agent_count": agent_count,
    "tasks": tasks,
    "records": [selected[500], selected[1000]],
}
with eval_target.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

# Build and validate the complete allowlist and its byte budget before the
# first mkdir against FASTWAM_OSS_OUTPUT_ROOT.  Publisher-generated evidence is
# included in the same cap.  No capacity decision is made from OSS FUSE df.
"${FASTWAM_PYTHON}" - \
  "${FASTWAM_OSS_OUTPUT_ROOT}" "${FASTWAM_MAX_OSS_PUBLISH_BYTES}" \
  "${STEP500_WEIGHT}" "${STEP1000_WEIGHT}" "${STEP1000_STATE}" \
  "${PHASE2_RECEIPT}" "${PHASE3_RECEIPT}" "${LOCAL_EVAL_RECEIPT}" \
  "${PHASE1_LOG}" "${PHASE2_LOG}" "${PHASE3_LOG}" \
  "${FASTWAM_MEMBER}" "${FASTWAM_EXPERIMENT_ID}" "${FASTWAM_RUN_ID}" \
  "${FASTWAM_AGENT_COUNT}" "${FASTWAM_TASKS_JSON}" \
  "${FASTWAM_PREPARED_RESERVATION_PATH}" \
  "${FASTWAM_SUITE_STORAGE_RESERVATION_PATH}" <<'PY'
import json
import os
import stat
import sys
from pathlib import Path

(root_literal, cap_literal, w500_literal, w1000_literal, state_literal,
 p2_literal, p3_literal, eval_literal, log1_literal, log2_literal, log3_literal,
 member, experiment_id, run_id, agent_literal, tasks_literal,
 reservation_path, suite_path) = sys.argv[1:]
root = Path(root_literal)
cap = int(cap_literal)
state_root = Path(state_literal)
if root.exists() or root.is_symlink():
    raise FileExistsError(f"unique durable output exists: {root}")

sources = [
    (Path(w500_literal), "checkpoints/weights/step_000500.pt"),
    (Path(w1000_literal), "checkpoints/weights/step_001000.pt"),
    (Path(p2_literal), "receipts/step500-resume.json"),
    (Path(p3_literal), "receipts/step1000-fresh-load.json"),
    (Path(eval_literal), "eval/offline-eval.json"),
    (Path(log1_literal), "logs/phase1-train-to-500.log"),
    (Path(log2_literal), "logs/phase2-resume-to-1000.log"),
    (Path(log3_literal), "logs/phase3-fresh-load-step1000.log"),
]
# STATE_TREE_ENUMERATION_BEGIN
def enumerate_state_tree(root):
    root_info = root.lstat()
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise RuntimeError(f"state tree root is not a real directory: {root}")
    entries = []
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"state tree symlink forbidden: {path}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"state tree entry is not a single-link regular file: {path}")
        entries.append(
            (path, "checkpoints/state/step_001000/" + path.relative_to(root).as_posix())
        )
    return entries
# STATE_TREE_ENUMERATION_END

sources.extend(enumerate_state_tree(state_root))

def descriptor(path):
    if path.is_symlink():
        raise RuntimeError(f"publish source symlink forbidden: {path}")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError(f"publish source is not a single-link regular file: {path}")
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns)

if len({relative for _, relative in sources}) != len(sources):
    raise RuntimeError("publication allowlist has duplicate destinations")
before = {str(path): descriptor(path) for path, _ in sources}
artifact_entries = [{"path": relative, "bytes": before[str(path)][4]} for path, relative in sources]
terminal = {
    "schema": "fastwam-action-native-agents-terminal-receipt-v1",
    "external_contract": "action_only_native_agents_1x8_v1",
    "member": member,
    "experiment_id": experiment_id,
    "run_id": run_id,
    "native_agent_count": int(agent_literal),
    "tasks": json.loads(tasks_literal),
    "masked_agent_set": False,
    "treatment": {"training_mode": "action_only_cache", "video_generation": False,
                  "hub_enabled": True, "gaussian_enabled": True,
                  "trainable_scope": "action"},
    "schedule": {"max_steps": 1000, "save_every": 500, "eval_every": 500,
                 "offline_eval_num_samples": 32, "seed": 42,
                 "train_split_seed": 42, "val_split_seed": 42},
    "hardware": {"workers": 1, "gpus_per_worker": 8, "total_gpus": 8},
    "checkpoint_state_kind": "full",
    "phase3_fresh_world_load": {"world_size": 8, "restored_global_step": 1000,
                                "zero_update": True, "source": "local_final_state"},
    "prepared_reservation_path": reservation_path,
    "suite_storage_reservation_path": suite_path,
    "artifacts": artifact_entries,
    "status": "COMPLETE",
}
terminal_bytes = (json.dumps(terminal, indent=2, sort_keys=True) + "\n").encode()
complete = {
    "schema": "fastwam-action-native-agents-complete-v1",
    "terminal_receipt": "receipts/terminal.json",
    "status": "COMPLETE",
}
complete_bytes = (json.dumps(complete, indent=2, sort_keys=True) + "\n").encode()
total = sum(item["bytes"] for item in artifact_entries) + len(terminal_bytes) + len(complete_bytes)
if total > cap:
    raise RuntimeError(f"publication plan exceeds per-run cap: {total} > {cap}")

def write_all(fd, payload):
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("zero-byte write while publishing durable artifact")
        view = view[written:]

def create_bytes(destination, payload):
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(destination, flags, 0o600)
    try:
        write_all(fd, payload)
    finally:
        os.close(fd)
    with destination.open("rb") as handle:
        if handle.read() != payload:
            raise RuntimeError(f"durable JSON readback differs: {destination}")

def stream_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    before_stat = descriptor(source)
    fd = os.open(destination, flags, 0o600)
    try:
        with source.open("rb") as handle:
            while True:
                block = handle.read(8 * 1024 * 1024)
                if not block:
                    break
                write_all(fd, block)
    finally:
        os.close(fd)
    if descriptor(source) != before_stat:
        raise RuntimeError(f"publish source changed during copy: {source}")
    with source.open("rb") as left, destination.open("rb") as right:
        while True:
            a = left.read(8 * 1024 * 1024)
            b = right.read(8 * 1024 * 1024)
            if a != b:
                raise RuntimeError(f"durable byte readback differs: {destination}")
            if not a:
                break

# This is intentionally the first mutation beneath the unique durable output.
root.mkdir(parents=False, mode=0o700)
for source, relative in sources:
    stream_copy(source, root / relative)
create_bytes(root / "receipts/terminal.json", terminal_bytes)
if {str(path): descriptor(path) for path, _ in sources} != before:
    raise RuntimeError("a local artifact changed after durable publication")

# COMPLETE last: nothing may be created beneath root after this call.
create_bytes(root / "COMPLETE", complete_bytes)
expected = {relative for _, relative in sources} | {"receipts/terminal.json", "COMPLETE"}
observed = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
if observed != expected:
    raise RuntimeError(f"durable allowlist mismatch: expected={sorted(expected)} observed={sorted(observed)}")
print(json.dumps({"status": "COMPLETE", "member": member, "published_bytes": total,
                  "artifact_files": len(expected), "output_root": str(root)}, sort_keys=True))
PY

echo "[formal-runtime] COMPLETE member=${FASTWAM_MEMBER} run_id=${FASTWAM_RUN_ID}"
