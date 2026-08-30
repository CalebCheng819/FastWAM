#!/usr/bin/env bash
set -euo pipefail

# Formal 3-worker x 8-GPU launcher for the joint-safe RoboFactory table11 rerun.
# PAI invokes this script once per worker. The script fails closed on source,
# topology, assets, configuration, inter-node communication, and output reuse.

die() {
  printf 'table11 launcher error: %s\n' "$*" >&2
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
  [[ "${!name}" == "${expected}" ]] || die "${name} drifted from the formal DLC contract"
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

TEST_MODE="${FASTWAM_TABLE11_TEST_MODE:-0}"
[[ "${TEST_MODE}" == "0" || "${TEST_MODE}" == "1" ]] || \
  die "FASTWAM_TABLE11_TEST_MODE must be 0 or 1"
if [[ "${TEST_MODE}" == "1" && -n "${PAI_DLC_JOB_ID:-}${DLC_JOB_ID:-}${PAI_JOB_ID:-}" ]]; then
  die "FASTWAM_TABLE11_TEST_MODE is forbidden inside a DLC job"
fi

require_env RUN_ID
require_env FASTWAM_TABLE11_ATTEMPT_ID
RUN_MODE="${FASTWAM_TABLE11_RUN_MODE:-formal}"
[[ "${RUN_MODE}" == "formal" || "${RUN_MODE}" == "preflight-one-step" ]] || \
  die "FASTWAM_TABLE11_RUN_MODE must be formal or preflight-one-step"
if [[ "${RUN_MODE}" == "preflight-one-step" ]]; then
  EXPECTED_MACHINES=1
  EXPECTED_WORLD=8
  TARGET_STEP=1
  OPTIMIZER_UPDATES=1
else
  EXPECTED_MACHINES=3
  EXPECTED_WORLD=24
  TARGET_STEP=50000
  OPTIMIZER_UPDATES=50000
fi

# Freeze PAI's outer-worker topology before Accelerate creates its 24-rank
# world, then remove the outer rank variables immediately before exec.
NUM_MACHINES="${WORLD_SIZE:-}"
MACHINE_RANK="${RANK:-}"
GPUS_PER_NODE="${NPROC_PER_NODE:-}"
MASTER_HOST="${MASTER_ADDR:-}"
MASTER_TCP_PORT="${MASTER_PORT:-}"

SOURCE_BUNDLE="${FASTWAM_TABLE11_SOURCE_BUNDLE:-}"
CODE_COMMIT="${FASTWAM_TABLE11_CODE_COMMIT:-}"
LOCAL_SOURCE_ROOT="${FASTWAM_TABLE11_LOCAL_SOURCE_ROOT:-/tmp/fastwam-table11-source-checkouts}"
OFFLINE_CODE_COMMIT="${FASTWAM_OFFLINE_CODE_COMMIT:-}"

if [[ "${FASTWAM_TABLE11_OFFLINE_ENV_READY:-0}" != "1" ]]; then
  require_env FASTWAM_TABLE11_BOOTSTRAP_SCRIPT
  require_env FASTWAM_OFFLINE_ENV_BASE_PYTHON
  require_env FASTWAM_TABLE11_SOURCE_BUNDLE
  require_env FASTWAM_TABLE11_CODE_COMMIT
  require_env FASTWAM_OFFLINE_CODE_COMMIT
  if [[ "${TEST_MODE}" != "1" ]]; then
    [[ "${FASTWAM_TABLE11_BOOTSTRAP_SCRIPT}" == /oss-chengjuntao/* ]] || \
      die "bootstrap script must be below /oss-chengjuntao"
    [[ "${SOURCE_BUNDLE}" == /oss-chengjuntao/* ]] || \
      die "source bundle must be below /oss-chengjuntao"
    [[ "${LOCAL_SOURCE_ROOT}" == /tmp/* ]] || \
      die "source checkout root must be node-local below /tmp"
  fi
  [[ "${CODE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
    die "FASTWAM_TABLE11_CODE_COMMIT must be an exact lowercase Git revision"
  [[ "${OFFLINE_CODE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
    die "FASTWAM_OFFLINE_CODE_COMMIT must be an exact lowercase Git revision"
  [[ "${SOURCE_BUNDLE}" == *.bundle ]] || die "source must be a Git .bundle file"
  [[ -f "${SOURCE_BUNDLE}" && ! -L "${SOURCE_BUNDLE}" ]] || \
    die "source bundle must be a regular non-symlink file"
  [[ -f "${FASTWAM_TABLE11_BOOTSTRAP_SCRIPT}" && ! -L "${FASTWAM_TABLE11_BOOTSTRAP_SCRIPT}" ]] || \
    die "bootstrap script must be a regular non-symlink file"
  [[ "${FASTWAM_OFFLINE_ENV_BASE_PYTHON}" == /* && -x "${FASTWAM_OFFLINE_ENV_BASE_PYTHON}" ]] || \
    die "FASTWAM_OFFLINE_ENV_BASE_PYTHON must be an absolute executable"

  # shellcheck source=/dev/null
  source "${FASTWAM_TABLE11_BOOTSTRAP_SCRIPT}"
  declare -F fastwam_prepare_offline_training_env >/dev/null || \
    die "offline bootstrap does not define fastwam_prepare_offline_training_env"
  export FASTWAM_CODE_COMMIT="${OFFLINE_CODE_COMMIT}"
  fastwam_prepare_offline_training_env || die "offline dependency bootstrap failed"
  DEPENDENCY_PYTHON="${FASTWAM_PYTHON:-}"
  [[ "${DEPENDENCY_PYTHON}" == /* && -x "${DEPENDENCY_PYTHON}" ]] || \
    die "offline bootstrap did not export an executable FASTWAM_PYTHON"
  if [[ "${TEST_MODE}" != "1" ]]; then
    [[ "${DEPENDENCY_PYTHON}" == /tmp/* ]] || \
      die "dependency Python must be node-local below /tmp"
  fi

  command -v git >/dev/null 2>&1 || die "git is required to restore the source bundle"
  LOCAL_PARENT="${LOCAL_SOURCE_ROOT}/${RUN_ID}"
  LOCAL_REPO="${LOCAL_PARENT}/${FASTWAM_TABLE11_ATTEMPT_ID}"
  PARTIAL_REPO="${LOCAL_REPO}.partial.${BASHPID}"
  [[ ! -e "${LOCAL_REPO}" && ! -L "${LOCAL_REPO}" ]] || \
    die "node-local source checkout already exists: ${LOCAL_REPO}"
  [[ ! -e "${PARTIAL_REPO}" && ! -L "${PARTIAL_REPO}" ]] || \
    die "partial source checkout already exists: ${PARTIAL_REPO}"
  mkdir -p -- "${LOCAL_PARENT}"
  git clone --quiet --no-checkout -- "${SOURCE_BUNDLE}" "${PARTIAL_REPO}" || \
    die "failed to clone the source bundle"
  git -C "${PARTIAL_REPO}" checkout --quiet --detach "${CODE_COMMIT}" || \
    die "code commit is absent from the source bundle"
  [[ "$(git -C "${PARTIAL_REPO}" rev-parse HEAD)" == "${CODE_COMMIT}" ]] || \
    die "restored source revision differs from FASTWAM_TABLE11_CODE_COMMIT"
  [[ -z "$(git -C "${PARTIAL_REPO}" status --porcelain --untracked-files=all)" ]] || \
    die "restored source checkout is dirty"
  mv -- "${PARTIAL_REPO}" "${LOCAL_REPO}"

  export FASTWAM_TABLE11_PYTHON="${DEPENDENCY_PYTHON}"
  export FASTWAM_TABLE11_REPO_ROOT="${LOCAL_REPO}"
  export FASTWAM_REPO_ROOT="${LOCAL_REPO}"
  export FASTWAM_CODE_COMMIT="${CODE_COMMIT}"
  export PYTHONPATH="${LOCAL_REPO}/src"
  export PYTHONNOUSERSITE=1
  export FASTWAM_TABLE11_OFFLINE_ENV_READY=1
else
  require_env FASTWAM_TABLE11_REPO_ROOT
  require_env FASTWAM_TABLE11_PYTHON
  export FASTWAM_REPO_ROOT="${FASTWAM_TABLE11_REPO_ROOT}"
  export FASTWAM_CODE_COMMIT="${CODE_COMMIT}"
  export PYTHONPATH="${FASTWAM_TABLE11_REPO_ROOT}/src"
  export PYTHONNOUSERSITE=1
fi

RUN_ID="${RUN_ID:-}"
ATTEMPT_ID="${FASTWAM_TABLE11_ATTEMPT_ID:-}"
REPO_ROOT="${FASTWAM_TABLE11_REPO_ROOT:-}"
PYTHON_BIN="${FASTWAM_TABLE11_PYTHON:-}"
OUTPUT_DIR="${FASTWAM_TABLE11_OUTPUT_DIR:-}"
DRY_RUN="${FASTWAM_TABLE11_DRY_RUN:-0}"
DATASET_ROOT="${FASTWAM_TABLE11_DATASET_ROOT:-}"
STATS_PATH="${FASTWAM_TABLE11_STATS_PATH:-}"
TEXT_CACHE_DIR="${FASTWAM_TABLE11_TEXT_CACHE_DIR:-}"
GAUSSIAN_CACHE_DIR="${FASTWAM_TABLE11_GAUSSIAN_CACHE_DIR:-}"
MODEL_CACHE_ROOT="${FASTWAM_TABLE11_MODEL_CACHE_ROOT:-}"
VAE_PATH="${FASTWAM_TABLE11_VAE_PATH:-}"
SOURCE_WEIGHT="${FASTWAM_TABLE11_SOURCE_WEIGHT:-}"
EXPECTED_WEIGHT_BYTES="${FASTWAM_TABLE11_SOURCE_WEIGHT_BYTES:-}"
EXPECTED_H5_FILES="${FASTWAM_TABLE11_EXPECTED_H5_FILES:-}"

TASK_PROFILE="robofactory_table11_vg1_hub1_gau1_scratch50k_224_1e-4"
SCALE_PROFILE="robofactory_multi_robot_24gpu_scratch50k"

is_safe_id "${RUN_ID}" || die "RUN_ID is not a safe identifier: ${RUN_ID}"
is_safe_id "${ATTEMPT_ID}" || die "attempt ID is not a safe identifier: ${ATTEMPT_ID}"
if [[ -n "${FASTWAM_ATTEMPT_ID:-}" && "${FASTWAM_ATTEMPT_ID}" != "${ATTEMPT_ID}" ]]; then
  die "FASTWAM_ATTEMPT_ID conflicts with FASTWAM_TABLE11_ATTEMPT_ID"
fi
export FASTWAM_ATTEMPT_ID="${ATTEMPT_ID}"
[[ "${NUM_MACHINES}" == "${EXPECTED_MACHINES}" ]] || \
  die "WORLD_SIZE must be the DLC worker count ${EXPECTED_MACHINES}, got ${NUM_MACHINES:-unset}"
[[ "${GPUS_PER_NODE}" == "8" ]] || \
  die "NPROC_PER_NODE must be 8, got ${GPUS_PER_NODE:-unset}"
is_uint "${MACHINE_RANK:-x}" || die "RANK must be an integer in [0,2]"
((10#${MACHINE_RANK} < 10#${EXPECTED_MACHINES})) || \
  die "RANK must be below ${EXPECTED_MACHINES}, got ${MACHINE_RANK}"
[[ -z "${LOCAL_RANK:-}" || "${LOCAL_RANK}" == "0" ]] || \
  die "outer DLC command must run once per node with LOCAL_RANK=0"
is_non_loopback "${MASTER_HOST}" || die "MASTER_ADDR must be a shared non-loopback address"
is_uint "${MASTER_TCP_PORT:-x}" || die "MASTER_PORT must be an integer"
((10#${MASTER_TCP_PORT} >= 1 && 10#${MASTER_TCP_PORT} <= 65535)) || \
  die "MASTER_PORT must be in [1,65535]"
[[ $((10#${NUM_MACHINES} * 10#${GPUS_PER_NODE})) -eq 10#${EXPECTED_WORLD} ]] || \
  die "global world size must be exactly ${EXPECTED_WORLD}"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || \
  die "FASTWAM_TABLE11_DRY_RUN must be 0 or 1"

[[ "${REPO_ROOT}" == /* && -d "${REPO_ROOT}" ]] || die "repository root is invalid"
[[ "${PYTHON_BIN}" == /* && -x "${PYTHON_BIN}" ]] || die "Python executable is unavailable"
if [[ "${TEST_MODE}" != "1" ]]; then
  [[ "${REPO_ROOT}" == /tmp/* ]] || die "production source checkout must be below /tmp"
  [[ "${PYTHON_BIN}" == /tmp/* ]] || die "production Python must be below /tmp"
  [[ "${OUTPUT_DIR}" == "/oss-chengjuntao/artifacts/${RUN_ID}" ]] || \
    die "output must be the unique canonical OSS artifact path"
  [[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${CODE_COMMIT}" ]] || \
    die "active source checkout does not match the frozen commit"
  [[ -z "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=all)" ]] || \
    die "active source checkout is dirty"

  require_exact_env FASTWAM_TABLE11_PROVENANCE_MODE "stat_cmp"
  require_exact_env FASTWAM_TABLE11_DATASET_ROOT "/oss-chengjuntao/robofactory/table/robofactory-table-11task-200each-h256-2g-stateful-safe-r3-20260827/tasks"
  require_exact_env FASTWAM_TABLE11_STATS_PATH "/oss-chengjuntao/fastwam-assets/robofactory/table11-200each-h256-stateful-safe-r3-s42/stats/train-stats.json"
  require_exact_env FASTWAM_TABLE11_TEXT_CACHE_DIR "/oss-chengjuntao/fastwam-assets/robofactory/table11-200each-h256-stateful-safe-r3-s42/text-embeds"
  require_exact_env FASTWAM_TABLE11_GAUSSIAN_CACHE_DIR "/oss-chengjuntao/fastwam-assets/robofactory/table11-200each-h256-stateful-safe-r3-s42/gaussian/compact-s42-13x28x40-fp16-meanalpha-direct-v1"
  require_exact_env FASTWAM_TABLE11_MODEL_CACHE_ROOT "/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/model-cache"
  require_exact_env FASTWAM_TABLE11_VAE_PATH "/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/model-cache/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
  require_exact_env FASTWAM_TABLE11_SOURCE_WEIGHT "/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/yuanty-fastwam-139eebb6d90cdd9bdbbe465f72c6edc9ad5a518a/libero_uncond_2cam224.pt"
  require_exact_env FASTWAM_TABLE11_SOURCE_WEIGHT_BYTES "12041735140"
  require_exact_env FASTWAM_TABLE11_EXPECTED_H5_FILES "11"
fi

for path in \
  "${REPO_ROOT}/scripts/train.py" \
  "${REPO_ROOT}/scripts/accelerate_configs/accelerate_zero2_ds.yaml" \
  "${REPO_ROOT}/src/fastwam/trainer.py" \
  "${REPO_ROOT}/configs/task/${TASK_PROFILE}.yaml" \
  "${REPO_ROOT}/configs/scale/${SCALE_PROFILE}.yaml" \
  "${REPO_ROOT}/configs/data/robofactory_table11.yaml"; do
  [[ -f "${path}" && ! -L "${path}" ]] || die "required source file is missing: ${path}"
done

[[ "${SOURCE_WEIGHT}" == *.pt ]] || die "source must be a weight .pt file"
[[ "${SOURCE_WEIGHT}" != */checkpoints/state/* ]] || \
  die "optimizer/training-state directories are forbidden"
[[ -f "${SOURCE_WEIGHT}" && ! -L "${SOURCE_WEIGHT}" ]] || \
  die "source weight must be a regular non-symlink file"
is_uint "${EXPECTED_WEIGHT_BYTES:-x}" || die "source weight byte count must be positive"
((10#${EXPECTED_WEIGHT_BYTES} > 0)) || die "source weight byte count must be positive"
[[ "$(stat -c '%s' -- "${SOURCE_WEIGHT}")" == "${EXPECTED_WEIGHT_BYTES}" ]] || \
  die "source weight byte count differs from the audited value"

FASTWAM_EXPECTED_REPO_ROOT="${REPO_ROOT}" \
PYTHONPATH="${REPO_ROOT}/src" PYTHONNOUSERSITE=1 \
"${PYTHON_BIN}" - <<'PY' || die "fastwam import escaped the active source checkout"
import os
from pathlib import Path
import fastwam

origin = Path(fastwam.__file__).resolve()
expected = (Path(os.environ["FASTWAM_EXPECTED_REPO_ROOT"]) / "src" / "fastwam").resolve()
origin.relative_to(expected)
print(f"table11 source import gate: origin={origin} expected_root={expected}")
PY

[[ -d "${DATASET_ROOT}" ]] || die "dataset root is missing: ${DATASET_ROOT}"
[[ -f "${STATS_PATH}" && ! -L "${STATS_PATH}" ]] || die "stats file is missing"
[[ -d "${TEXT_CACHE_DIR}" ]] || die "text cache is missing"
[[ -d "${GAUSSIAN_CACHE_DIR}" ]] || die "Gaussian cache is missing"
[[ -f "${GAUSSIAN_CACHE_DIR}/COMPLETE" && ! -L "${GAUSSIAN_CACHE_DIR}/COMPLETE" ]] || \
  die "Gaussian COMPLETE marker is missing"
[[ -f "${GAUSSIAN_CACHE_DIR}/manifest.json" && ! -L "${GAUSSIAN_CACHE_DIR}/manifest.json" ]] || \
  die "Gaussian manifest is missing"
[[ -f "${GAUSSIAN_CACHE_DIR}/selection.jsonl" && ! -L "${GAUSSIAN_CACHE_DIR}/selection.jsonl" ]] || \
  die "Gaussian selection is missing"
if [[ "${TEST_MODE}" != "1" ]]; then
  [[ -d "${MODEL_CACHE_ROOT}" ]] || die "model cache is missing"
  [[ -f "${VAE_PATH}" && ! -L "${VAE_PATH}" ]] || die "Wan VAE is missing"
fi

FASTWAM_TABLE11_CHECK_ROOT="${DATASET_ROOT}" \
FASTWAM_TABLE11_CHECK_STATS="${STATS_PATH}" \
FASTWAM_TABLE11_CHECK_TEXT="${TEXT_CACHE_DIR}" \
FASTWAM_TABLE11_CHECK_GAUSSIAN="${GAUSSIAN_CACHE_DIR}" \
FASTWAM_TABLE11_CHECK_H5="${EXPECTED_H5_FILES}" \
"${PYTHON_BIN}" - <<'PY' || die "RoboFactory table11 asset contract validation failed"
import json
import math
import os
from pathlib import Path

root = Path(os.environ["FASTWAM_TABLE11_CHECK_ROOT"])
h5_files = sorted(root.rglob("*.h5"))
if len(h5_files) != int(os.environ["FASTWAM_TABLE11_CHECK_H5"]):
    raise SystemExit(f"expected 11 H5 files, observed {len(h5_files)}")
tasks = {path.parts[len(root.parts)] for path in h5_files}
if len(tasks) != 11:
    raise SystemExit(f"expected 11 task directories, observed {len(tasks)}")

stats = json.loads(Path(os.environ["FASTWAM_TABLE11_CHECK_STATS"]).read_text())
for section, size in (("action", 8), ("state", 18)):
    for key in ("mean", "std"):
        values = stats[section][key]
        if len(values) != size or not all(math.isfinite(float(value)) for value in values):
            raise SystemExit(f"invalid {section}.{key}")
    if not all(float(value) > 0 for value in stats[section]["std"]):
        raise SystemExit(f"non-positive {section}.std")

text_files = [path for path in Path(os.environ["FASTWAM_TABLE11_CHECK_TEXT"]).rglob("*") if path.is_file()]
if len(text_files) < 11:
    raise SystemExit(f"expected at least 11 text cache files, observed {len(text_files)}")

gaussian = Path(os.environ["FASTWAM_TABLE11_CHECK_GAUSSIAN"])
complete = json.loads((gaussian / "COMPLETE").read_text())
manifest = json.loads((gaussian / "manifest.json").read_text())
if complete.get("complete") is not True:
    raise SystemExit("Gaussian completion marker is not terminal")
if int(manifest.get("total_frames", 0)) <= 0:
    raise SystemExit("Gaussian manifest has no frames")
if manifest.get("derivation", {}).get("source") != "direct-teacher-forward-index-v1":
    raise SystemExit("Gaussian cache is not the formal direct-compact derivation")
print(
    "table11 asset gate: "
    f"tasks={len(tasks)} h5={len(h5_files)} text_files={len(text_files)} "
    f"gaussian_frames={manifest['total_frames']}"
)
PY

export FASTWAM_GAUSSIAN_CACHE_DIR="${GAUSSIAN_CACHE_DIR}"
export FASTWAM_PRETRAINED_ROOT="${SOURCE_WEIGHT}"
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_CACHE_ROOT}"
export FASTWAM_LOCAL_VAE_PATH="${VAE_PATH}"

FASTWAM_TABLE11_REPO_FOR_CONFIG="${REPO_ROOT}" \
FASTWAM_TABLE11_CONFIG_DATASET="${DATASET_ROOT}" \
FASTWAM_TABLE11_CONFIG_STATS="${STATS_PATH}" \
FASTWAM_TABLE11_CONFIG_TEXT="${TEXT_CACHE_DIR}" \
FASTWAM_TABLE11_CONFIG_GAUSSIAN="${GAUSSIAN_CACHE_DIR}" \
FASTWAM_TABLE11_CONFIG_SOURCE_WEIGHT="${SOURCE_WEIGHT}" \
FASTWAM_TABLE11_CONFIG_RUN_MODE="${RUN_MODE}" \
FASTWAM_TABLE11_CONFIG_EXPECTED_WORLD="${EXPECTED_WORLD}" \
"${PYTHON_BIN}" - <<'PY' || die "formal Hydra configuration contract validation failed"
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()
repo = Path(os.environ["FASTWAM_TABLE11_REPO_FOR_CONFIG"])
overrides = [
    "task=robofactory_table11_vg1_hub1_gau1_scratch50k_224_1e-4",
    "+scale=robofactory_multi_robot_24gpu_scratch50k",
    f"data.train.root_dir={os.environ['FASTWAM_TABLE11_CONFIG_DATASET']}",
    f"data.val.root_dir={os.environ['FASTWAM_TABLE11_CONFIG_DATASET']}",
    f"data.train.pretrained_norm_stats={os.environ['FASTWAM_TABLE11_CONFIG_STATS']}",
    f"data.val.pretrained_norm_stats={os.environ['FASTWAM_TABLE11_CONFIG_STATS']}",
    f"data.train.text_embedding_cache_dir={os.environ['FASTWAM_TABLE11_CONFIG_TEXT']}",
    f"data.val.text_embedding_cache_dir={os.environ['FASTWAM_TABLE11_CONFIG_TEXT']}",
    f"data.train.gaussian_cache_dir={os.environ['FASTWAM_TABLE11_CONFIG_GAUSSIAN']}",
    f"data.val.gaussian_cache_dir={os.environ['FASTWAM_TABLE11_CONFIG_GAUSSIAN']}",
]
run_mode = os.environ["FASTWAM_TABLE11_CONFIG_RUN_MODE"]
if run_mode == "preflight-one-step":
    overrides.extend(
        [
            "max_steps=1",
            "save_every=0",
            "eval_every=0",
            "log_every=1",
            "save_training_state=false",
            "save_final_checkpoint=false",
            "seal_training_run=false",
        ]
    )
with initialize_config_dir(config_dir=str(repo / "configs"), version_base="1.3"):
    cfg = compose(config_name="train", overrides=overrides)
resolved = OmegaConf.to_container(cfg, resolve=True)
expected_world = int(os.environ["FASTWAM_TABLE11_CONFIG_EXPECTED_WORLD"])

expected = {
    "batch_size": 1,
    "num_workers": 8,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-2,
    "lr_scheduler_type": "cosine",
    "max_steps": 50000,
    "run_initial_global_step": 0,
    "save_every": 5000,
    "eval_every": 5000,
    "checkpoint_state_kind": "full",
    "save_training_state": True,
    "save_final_checkpoint": True,
    "trainable_scope": "dit",
}
if run_mode == "preflight-one-step":
    expected.update(
        {
            "max_steps": 1,
            "save_every": 0,
            "eval_every": 0,
            "log_every": 1,
            "save_training_state": False,
            "save_final_checkpoint": False,
            "seal_training_run": False,
        }
    )
for key, wanted in expected.items():
    if resolved[key] != wanted:
        raise SystemExit(f"config drift at {key}: expected {wanted!r}, got {resolved[key]!r}")
if resolved["weights_only_warm_start"]["enabled"] is not False:
    raise SystemExit("weights-only warm start must remain disabled")
if resolved["resume"] != os.environ["FASTWAM_TABLE11_CONFIG_SOURCE_WEIGHT"]:
    raise SystemExit("generic pretrained initialization path drifted")
if resolved["model"]["training_mode"] != "joint":
    raise SystemExit("model training mode is not joint")
if not resolved["model"]["action_dit_config"]["hub_enabled"]:
    raise SystemExit("HUB is disabled")
if not resolved["model"]["action_dit_config"]["enable_gaussian"]:
    raise SystemExit("Gaussian conditioning is disabled")
for split in ("train", "val"):
    data = resolved["data"][split]
    if data["required_agent_counts"] != [1, 2, 3, 4]:
        raise SystemExit(f"{split} agent-count coverage drifted")
    if data["gaussian_cache_verify"] != "stat_cmp":
        raise SystemExit(f"{split} Gaussian verification is not stat_cmp")
if len(resolved["data"]["train"]["instruction_map"]) != 11:
    raise SystemExit("instruction map does not contain exactly 11 tasks")
if run_mode == "preflight-one-step":
    print(f"table11 safe config gate: preflight world={expected_world} update=0->1")
else:
    print("table11 safe config gate: world=24 global_batch=24 updates=50000 checkpoints=5000..50000/5000")
print("table11 initialization gate: run_initial_global_step=0 weights_only_warm_start.enabled=false")
PY

if [[ "${TEST_MODE}" != "1" && "${DRY_RUN}" == "0" ]]; then
  require_exact_env FASTWAM_ERDMA_BUNDLE_ROOT "/oss-chengjuntao/artifacts/erdma-userspace-56.2-1.0.3"
  require_exact_env FASTWAM_ERDMA_EXPECTED_VERSION "56.2-1.0.3"
  require_exact_env NCCL_IB_HCA "erdma"
  require_exact_env NCCL_DEBUG "INFO"
  require_exact_env NCCL_DEBUG_SUBSYS "INIT,NET"
  require_exact_env FASTWAM_PREFLIGHT_REQUIRE_ERDMA "1"
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/scripts/dlc_preflight.sh"
  fastwam_prepare_nvidia_host570 || die "NVIDIA host preparation failed"
  fastwam_run_local_cuda_preflight "${GPUS_PER_NODE}" "${MACHINE_RANK}" || \
    die "local CUDA preflight failed"
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/docker/prepare-erdma-userspace.sh"
  fastwam_prepare_erdma_userspace || die "eRDMA userspace preparation failed"
  fastwam_run_global_allreduce_preflight \
    "${GPUS_PER_NODE}" "${NUM_MACHINES}" "${MACHINE_RANK}" \
    "${MASTER_HOST}" "${MASTER_TCP_PORT}" "${RUN_ID}" || \
    die "global NCCL/eRDMA preflight failed"
fi

LOCAL_CACHE_ROOT="${FASTWAM_TABLE11_LOCAL_WEIGHT_ROOT:-/tmp/fastwam-table11-checkpoints}"
LOCAL_ATTEMPT_DIR="${LOCAL_CACHE_ROOT}/${RUN_ID}/${ATTEMPT_ID}"
LOCAL_WEIGHT="${LOCAL_ATTEMPT_DIR}/libero_uncond_2cam224.pt"
LOCAL_READY="${LOCAL_ATTEMPT_DIR}/.ready"
RESERVATION="${OUTPUT_DIR}/.table11-run-reservation"
OUTPUT_RESERVATION_TIMEOUT="${FASTWAM_TABLE11_OUTPUT_RESERVATION_TIMEOUT:-300}"
is_uint "${OUTPUT_RESERVATION_TIMEOUT}" || die "output reservation timeout must be positive"
((10#${OUTPUT_RESERVATION_TIMEOUT} > 0)) || die "output reservation timeout must be positive"
if [[ "${TEST_MODE}" != "1" ]]; then
  [[ "${OUTPUT_RESERVATION_TIMEOUT}" == "300" ]] || \
    die "formal output reservation timeout must remain 300 seconds"
fi
RESERVATION_BODY="run_id=${RUN_ID}
attempt_id=${ATTEMPT_ID}
run_mode=${RUN_MODE}
workers=${EXPECTED_MACHINES}
gpus_per_worker=8
global_world_size=${EXPECTED_WORLD}
source_weight=${SOURCE_WEIGHT}
initialization=official-generic-pretrained-model-weights
optimizer=fresh
provenance_mode=stat_cmp
initial_global_step=0
target_global_step=${TARGET_STEP}
optimizer_steps_this_run=${OPTIMIZER_UPDATES}
per_device_batch_size=1
gradient_accumulation_steps=1
reference_global_batch_size=24
global_batch_size=${EXPECTED_WORLD}
sample_budget_equivalent=$([[ "${RUN_MODE}" == "formal" ]] && printf true || printf false)
learning_rate=0.0001
lr_scheduler=cosine
scheduler_warmup_steps=2250
save_every=5000
checkpoint_steps=5000,10000,15000,20000,25000,30000,35000,40000,45000,50000
dataset_root=${DATASET_ROOT}
gaussian_cache_dir=${GAUSSIAN_CACHE_DIR}
"
if [[ "${RUN_MODE}" == "preflight-one-step" ]]; then
  RESERVATION_BODY="${RESERVATION_BODY/save_every=5000/save_every=0}"
  RESERVATION_BODY="${RESERVATION_BODY/checkpoint_steps=5000,10000,15000,20000,25000,30000,35000,40000,45000,50000/checkpoint_steps=none}"
fi

if [[ "${DRY_RUN}" == "0" ]]; then
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  [[ "${gpu_count}" == "8" ]] || die "each worker must expose exactly 8 GPUs"
  if [[ "${MACHINE_RANK}" == "0" ]]; then
    [[ ! -e "${OUTPUT_DIR}" ]] || die "output already exists; RUN_ID is not unique"
    mkdir -- "${OUTPUT_DIR}" || die "cannot reserve output directory"
    printf '%s' "${RESERVATION_BODY}" >"${RESERVATION}"
  else
    deadline=$((SECONDS + 10#${OUTPUT_RESERVATION_TIMEOUT}))
    while [[ ! -f "${RESERVATION}" && ${SECONDS} -lt ${deadline} ]]; do
      sleep 1
    done
    [[ -f "${RESERVATION}" ]] || die "timed out waiting for rank-0 output reservation"
  fi
  [[ "$(cat -- "${RESERVATION}")" == "${RESERVATION_BODY%$'\n'}" ]] || \
    die "output reservation belongs to a different run contract"

  [[ "${LOCAL_ATTEMPT_DIR}" == /tmp/* || "${TEST_MODE}" == "1" ]] || \
    die "weight cache must remain node-local"
  [[ ! -e "${LOCAL_ATTEMPT_DIR}" && ! -L "${LOCAL_ATTEMPT_DIR}" ]] || \
    die "node-local weight cache already exists"
  PARTIAL_ATTEMPT="${LOCAL_ATTEMPT_DIR}.partial.${BASHPID}"
  mkdir -p -- "$(dirname -- "${LOCAL_ATTEMPT_DIR}")"
  mkdir -- "${PARTIAL_ATTEMPT}"
  PARTIAL_WEIGHT="${PARTIAL_ATTEMPT}/libero_uncond_2cam224.pt"
  SOURCE_STAT_BEFORE="$(stat -Lc '%d:%i:%s:%Y' -- "${SOURCE_WEIGHT}")"
  SOURCE_MTIME="$(stat -Lc '%Y' -- "${SOURCE_WEIGHT}")"
  cp -- "${SOURCE_WEIGHT}" "${PARTIAL_WEIGHT}" || die "failed to copy source weight"
  [[ "$(stat -c '%s' -- "${PARTIAL_WEIGHT}")" == "${EXPECTED_WEIGHT_BYTES}" ]] || \
    die "node-local source weight has the wrong byte count"
  [[ "$(stat -Lc '%d:%i:%s:%Y' -- "${SOURCE_WEIGHT}")" == "${SOURCE_STAT_BEFORE}" ]] || \
    die "source weight changed during staging"
  cmp -s -- "${SOURCE_WEIGHT}" "${PARTIAL_WEIGHT}" || \
    die "node-local weight bytes differ from source"
  printf 'provenance_mode=stat_cmp\nrun_id=%s\nattempt_id=%s\nsource_path=%s\ndestination_path=%s\nbytes=%s\nsource_mtime_epoch=%s\nfile_count=1\n' \
    "${RUN_ID}" "${ATTEMPT_ID}" "${SOURCE_WEIGHT}" "${LOCAL_WEIGHT}" \
    "${EXPECTED_WEIGHT_BYTES}" "${SOURCE_MTIME}" >"${PARTIAL_ATTEMPT}/.ready"
  mv -T -- "${PARTIAL_ATTEMPT}" "${LOCAL_ATTEMPT_DIR}"
  [[ -f "${LOCAL_WEIGHT}" && -f "${LOCAL_READY}" ]] || \
    die "node-local weight staging barrier was not published"
  printf 'table11 node staging complete: worker=%s local_weight=%s bytes=%s\n' \
    "${MACHINE_RANK}" "${LOCAL_WEIGHT}" "${EXPECTED_WEIGHT_BYTES}"
else
  LOCAL_WEIGHT="${SOURCE_WEIGHT}"
  printf 'table11 dry-run: no output reservation, GPU query, weight copy, or training was performed.\n'
fi

export FASTWAM_PRETRAINED_ROOT="${LOCAL_WEIGHT}"

COMMAND=(
  "${PYTHON_BIN}" -m accelerate.commands.launch
  --config_file "${REPO_ROOT}/scripts/accelerate_configs/accelerate_zero2_ds.yaml"
  --num_machines "${EXPECTED_MACHINES}"
  --machine_rank "${MACHINE_RANK}"
  --main_process_ip "${MASTER_HOST}"
  --main_process_port "${MASTER_TCP_PORT}"
  --num_processes "${EXPECTED_WORLD}"
  --deepspeed_multinode_launcher standard
  "${REPO_ROOT}/scripts/train.py"
  "task=${TASK_PROFILE}"
  "+scale=${SCALE_PROFILE}"
  "data.train.root_dir=${DATASET_ROOT}"
  "data.val.root_dir=${DATASET_ROOT}"
  "data.train.pretrained_norm_stats=${STATS_PATH}"
  "data.val.pretrained_norm_stats=${STATS_PATH}"
  "data.train.stats_source_root=${DATASET_ROOT}"
  "data.val.stats_source_root=${DATASET_ROOT}"
  "data.train.text_embedding_cache_dir=${TEXT_CACHE_DIR}"
  "data.val.text_embedding_cache_dir=${TEXT_CACHE_DIR}"
  "data.train.gaussian_cache_dir=${GAUSSIAN_CACHE_DIR}"
  "data.val.gaussian_cache_dir=${GAUSSIAN_CACHE_DIR}"
  "output_dir=${OUTPUT_DIR}"
  "wandb.name=${RUN_ID}"
)

if [[ "${RUN_MODE}" == "preflight-one-step" ]]; then
  COMMAND+=(
    "max_steps=1"
    "save_every=0"
    "eval_every=0"
    "log_every=1"
    "save_training_state=false"
    "save_final_checkpoint=false"
    "seal_training_run=false"
  )
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'table11 resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK NODE_RANK
cd -- "${REPO_ROOT}"
if [[ "${RUN_MODE}" == "formal" ]]; then
  exec "${COMMAND[@]}"
fi

[[ "${MACHINE_RANK}" == "0" ]] || die "one-step preflight must have exactly one worker"
PREFLIGHT_LOG="${OUTPUT_DIR}/preflight-train.log"
PREFLIGHT_TERMINAL="${OUTPUT_DIR}/terminal.json"
PREFLIGHT_COMPLETE="${OUTPUT_DIR}/COMPLETE"
for target in "${PREFLIGHT_LOG}" "${PREFLIGHT_TERMINAL}" "${PREFLIGHT_COMPLETE}"; do
  [[ ! -e "${target}" && ! -L "${target}" ]] || die "preflight target already exists: ${target}"
done
set +e
"${COMMAND[@]}" 2>&1 | tee -- "${PREFLIGHT_LOG}"
pipeline_status=("${PIPESTATUS[@]}")
command_status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
set -e
[[ "${command_status}" == "0" ]] || die "one-step training command failed with ${command_status}"
[[ "${tee_status}" == "0" ]] || die "one-step log capture failed with ${tee_status}"
[[ -f "${PREFLIGHT_LOG}" && ! -L "${PREFLIGHT_LOG}" ]] || die "preflight log is not a regular file"
grep -Fq -- "FASTWAM_GENERIC_BASE_LOAD=PASS before_prepare=true" "${PREFLIGHT_LOG}" || \
  die "generic base checkpoint load was not observed"
grep -Fq -- "optimizer/scheduler/step are intentionally not restored." "${PREFLIGHT_LOG}" || \
  die "fresh optimizer/scheduler declaration was not observed"
grep -Fq -- "FASTWAM_TRAINING_START initial_global_step=0 max_steps=1 optimizer_steps_this_run=1" "${PREFLIGHT_LOG}" || \
  die "step-zero training start was not observed"
grep -Eq -- '\[train\].*step=1/1([[:space:]]|$)' "${PREFLIGHT_LOG}" || \
  die "real optimizer step 1/1 was not observed"
! grep -Fq -- "step_005000.pt" "${PREFLIGHT_LOG}" || die "forbidden old checkpoint appeared in log"
! grep -Fq -- "Loaded explicit cross-treatment weights-only warm start" "${PREFLIGHT_LOG}" || \
  die "weights-only continuation appeared in scratch preflight"

FASTWAM_TABLE11_PREFLIGHT_LOG="${PREFLIGHT_LOG}" \
FASTWAM_TABLE11_PREFLIGHT_TERMINAL="${PREFLIGHT_TERMINAL}" \
FASTWAM_TABLE11_PREFLIGHT_COMPLETE="${PREFLIGHT_COMPLETE}" \
FASTWAM_TABLE11_PREFLIGHT_RUN_ID="${RUN_ID}" \
FASTWAM_TABLE11_PREFLIGHT_ATTEMPT_ID="${ATTEMPT_ID}" \
FASTWAM_TABLE11_PREFLIGHT_SOURCE_WEIGHT="${SOURCE_WEIGHT}" \
FASTWAM_TABLE11_PREFLIGHT_OUTPUT_DIR="${OUTPUT_DIR}" \
"${PYTHON_BIN}" - <<'PY' || die "failed to publish one-step preflight terminal receipt"
import datetime as dt
import json
import os
from pathlib import Path


def publish(path: Path, value: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            if written <= 0:
                raise OSError("short write while publishing preflight receipt")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


run_id = os.environ["FASTWAM_TABLE11_PREFLIGHT_RUN_ID"]
attempt_id = os.environ["FASTWAM_TABLE11_PREFLIGHT_ATTEMPT_ID"]
source_weight = os.environ["FASTWAM_TABLE11_PREFLIGHT_SOURCE_WEIGHT"]
output_dir = Path(os.environ["FASTWAM_TABLE11_PREFLIGHT_OUTPUT_DIR"])
log_path = Path(os.environ["FASTWAM_TABLE11_PREFLIGHT_LOG"])
terminal_path = Path(os.environ["FASTWAM_TABLE11_PREFLIGHT_TERMINAL"])
complete_path = Path(os.environ["FASTWAM_TABLE11_PREFLIGHT_COMPLETE"])
terminal = {
    "schema": "fastwam-table11safe-realdata-scratch-preflight-terminal-v1",
    "status": "PASS",
    "run_id": run_id,
    "attempt_id": attempt_id,
    "run_mode": "preflight-one-step",
    "dataset_kind": "joint-safe-table11-real-data",
    "initialization": "official-generic-pretrained-model-weights",
    "source_weight": source_weight,
    "optimizer": "fresh",
    "scheduler": "fresh",
    "initial_global_step": 0,
    "final_global_step": 1,
    "optimizer_steps_this_run": 1,
    "world_size": 8,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "log_path": str(log_path),
    "completed_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
}
publish(terminal_path, terminal)
expected_before_complete = {
    ".table11-run-reservation",
    "preflight-train.log",
    "terminal.json",
}
actual_before_complete = {path.name for path in output_dir.iterdir()}
if actual_before_complete != expected_before_complete:
    raise SystemExit(
        f"preflight pre-COMPLETE allowlist drift: {sorted(actual_before_complete)}"
    )
complete = {
    "schema": "fastwam-table11safe-realdata-scratch-preflight-complete-v1",
    "status": "PASS",
    "run_id": run_id,
    "attempt_id": attempt_id,
    "terminal": str(terminal_path),
}
publish(complete_path, complete)
expected = {".table11-run-reservation", "preflight-train.log", "terminal.json", "COMPLETE"}
actual = {path.name for path in output_dir.iterdir()}
if actual != expected:
    raise SystemExit(f"preflight output allowlist drift: {sorted(actual)}")
PY
printf 'table11 real-data scratch preflight: PASS run=%s step=0->1\n' "${RUN_ID}"
