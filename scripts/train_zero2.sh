#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
FORMAL_SCALE_NAME="robofactory_multi_robot_32gpu"
FORMAL_CPFS_PREFIX="/cpfs/user/chengjuntao"
FORMAL_OSS_PREFIX="/oss-chengjuntao"
FORMAL_LOCAL_CACHE_ROOT="/tmp/fastwam-whole-file-cache"
FORMAL_LOCAL_RUNTIME_ROOT="/tmp/fastwam-local-runtime"
OFFICIAL_FASTWAM_CHECKPOINT_SHA256="1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579"
OFFICIAL_WAN22_VAE_SHA256="0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996"
OFFICIAL_N234_TRAIN_S42_STATS_SHA256="350493b685d8db0ea4cfd66f58f49849e8cd1f65cecc269f15aff9101ac8a04d"
ERDMA_EXPECTED_VERSION="56.2-1.0.3"
ERDMA_EXPECTED_BUNDLE_SHA256="8f2c1c43d64a7745bea19bfe4cd1383344c9cf32779166f4aa67809ebf1f5fab"
ERDMA_EXPECTED_SOURCE_MANIFEST_SHA256="f05443faa27533274ae1b322723e21ac09bd80bd5b2513638dd2619c67552215"
ERDMA_EXPECTED_ENV_SHA256="b581a454249ad2a27ef21dad929a0db6d963a6613340bce10a866ff40017c11c"

PAI_WORLD_SIZE="${WORLD_SIZE:-}"
PAI_RANK="${RANK:-}"
PAI_NPROC_PER_NODE="${NPROC_PER_NODE:-}"
COMPAT_NNODES="${NNODES:-}"
COMPAT_NODE_RANK="${NODE_RANK:-}"
POSITIONAL_NPROC_PER_NODE=""
if (($# > 0)) && [[ "$1" =~ ^[0-9]+$ ]]; then
  POSITIONAL_NPROC_PER_NODE="$1"
  shift
fi

EXTRA_ARGS=("$@")
MAIN_PROCESS_IP="${MASTER_ADDR:-}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"
LAUNCH_DRY_RUN="${FASTWAM_LAUNCH_DRY_RUN:-0}"

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

is_enabled() {
  case "${1,,}" in
    1 | true | yes | on) return 0 ;;
    0 | false | no | off) return 1 ;;
    *)
      echo "Error: expected a boolean value, got '${1}'." >&2
      return 2
      ;;
  esac
}

is_valid_port() {
  is_integer "${1}" && ((1 <= 10#${1} && 10#${1} <= 65535))
}

is_sha256() {
  [[ "${1}" =~ ^[0-9a-f]{64}$ ]]
}

validate_safe_relative_path() {
  local name="$1"
  local value="$2"
  if [[ \
    -z "${value}" || \
    "${value}" == /* || \
    "${value}" == ./* || \
    "${value}" == "." || \
    "${value}" == ".." || \
    "${value}" == ../* || \
    "${value}" == */./* || \
    "${value}" == */. || \
    "${value}" == */../* || \
    "${value}" == */.. || \
    "${value}" == *//* \
  ]]; then
    echo "Error: ${name} must be a canonical non-escaping relative path, got '${value}'." >&2
    return 1
  fi
}

require_sha256_environment() {
  local name="$1"
  local value="${!name:-}"
  value="${value,,}"
  if ! is_sha256 "${value}"; then
    echo "Error: ${name} is required and must be 64 lowercase hex characters." >&2
    return 1
  fi
  printf -v "${name}" '%s' "${value}"
  export "${name}"
}

for value_name in PAI_WORLD_SIZE COMPAT_NNODES PAI_NPROC_PER_NODE POSITIONAL_NPROC_PER_NODE; do
  value="${!value_name}"
  if [[ -n "${value}" ]] && { ! is_integer "${value}" || ((10#${value} < 1)); }; then
    echo "Error: ${value_name} (${value}) must be a positive integer." >&2
    exit 1
  fi
done
for value_name in PAI_RANK COMPAT_NODE_RANK; do
  value="${!value_name}"
  if [[ -n "${value}" ]] && ! is_integer "${value}"; then
    echo "Error: ${value_name} (${value}) must be a non-negative integer." >&2
    exit 1
  fi
done

if [[ -n "${PAI_WORLD_SIZE}" && -n "${COMPAT_NNODES}" ]] && \
  ((10#${PAI_WORLD_SIZE} != 10#${COMPAT_NNODES})); then
  echo "Error: PAI WORLD_SIZE (${PAI_WORLD_SIZE} nodes) conflicts with NNODES (${COMPAT_NNODES})." >&2
  exit 1
fi
if [[ -n "${PAI_RANK}" && -n "${COMPAT_NODE_RANK}" ]] && \
  ((10#${PAI_RANK} != 10#${COMPAT_NODE_RANK})); then
  echo "Error: PAI RANK (${PAI_RANK}) conflicts with NODE_RANK (${COMPAT_NODE_RANK})." >&2
  exit 1
fi
if [[ -n "${PAI_NPROC_PER_NODE}" && -n "${POSITIONAL_NPROC_PER_NODE}" ]] && \
  ((10#${PAI_NPROC_PER_NODE} != 10#${POSITIONAL_NPROC_PER_NODE})); then
  echo "Error: PAI NPROC_PER_NODE (${PAI_NPROC_PER_NODE}) conflicts with positional nproc_per_node (${POSITIONAL_NPROC_PER_NODE})." >&2
  exit 1
fi

NUM_MACHINES="${PAI_WORLD_SIZE:-${COMPAT_NNODES:-1}}"
MACHINE_RANK="${PAI_RANK:-${COMPAT_NODE_RANK:-0}}"
NPROC_PER_NODE="${PAI_NPROC_PER_NODE:-${POSITIONAL_NPROC_PER_NODE:-}}"
if [[ -z "${NPROC_PER_NODE}" ]]; then
  echo "Error: set PAI NPROC_PER_NODE or pass positional <nproc_per_node>." >&2
  exit 1
fi

NUM_MACHINES=$((10#${NUM_MACHINES}))
MACHINE_RANK=$((10#${MACHINE_RANK}))
NPROC_PER_NODE=$((10#${NPROC_PER_NODE}))
GLOBAL_WORLD_SIZE=$((NUM_MACHINES * NPROC_PER_NODE))

if ((NUM_MACHINES > 1)) && [[ -z "${PAI_RANK}" && -z "${COMPAT_NODE_RANK}" ]]; then
  echo "Error: PAI RANK or compatibility NODE_RANK is required when the resolved node count is ${NUM_MACHINES}." >&2
  exit 1
fi
if ((MACHINE_RANK >= NUM_MACHINES)); then
  echo "Error: resolved machine rank (${MACHINE_RANK}) must be smaller than resolved node count (${NUM_MACHINES})." >&2
  exit 1
fi
if [[ -z "${MAIN_PROCESS_IP}" ]]; then
  if ((NUM_MACHINES == 1)); then
    MAIN_PROCESS_IP="127.0.0.1"
  else
    echo "Error: MASTER_ADDR is required when NNODES=${NUM_MACHINES}." >&2
    exit 1
  fi
fi
if ((NUM_MACHINES > 1)) && [[ \
  "${MAIN_PROCESS_IP}" == "localhost" || \
  "${MAIN_PROCESS_IP}" == "::1" || \
  "${MAIN_PROCESS_IP}" == 127.* \
]]; then
  echo "Error: MASTER_ADDR (${MAIN_PROCESS_IP}) must be reachable by every machine when NNODES=${NUM_MACHINES}." >&2
  exit 1
fi
if ! is_valid_port "${MAIN_PROCESS_PORT}"; then
  echo "Error: MASTER_PORT (${MAIN_PROCESS_PORT}) must be an integer in [1, 65535]." >&2
  exit 1
fi
MAIN_PROCESS_PORT=$((10#${MAIN_PROCESS_PORT}))

if is_enabled "${LAUNCH_DRY_RUN}"; then
  LAUNCH_DRY_RUN_ENABLED=1
else
  dry_run_status=$?
  if ((dry_run_status == 2)); then
    exit 1
  fi
  LAUNCH_DRY_RUN_ENABLED=0
fi

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

TASK_BASENAME="train"
SCALE_PROFILE=""
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      ;;
    +scale=* | scale=*)
      cfg="${arg#*=}"
      cfg="${cfg%.yaml}"
      if [[ -n "${SCALE_PROFILE}" && "${SCALE_PROFILE}" != "${cfg}" ]]; then
        echo "Error: multiple conflicting scale profiles were provided: ${SCALE_PROFILE} and ${cfg}." >&2
        exit 1
      fi
      SCALE_PROFILE="${cfg}"
      ;;
  esac
done

FORMAL_32GPU=0
if [[ "${SCALE_PROFILE}" == "${FORMAL_SCALE_NAME}" ]]; then
  FORMAL_32GPU=1
  if ((NUM_MACHINES != 4 || NPROC_PER_NODE != 8 || GLOBAL_WORLD_SIZE != 32)); then
    echo "Error: +scale=${FORMAL_SCALE_NAME} requires DLC WORLD_SIZE=4 nodes, NPROC_PER_NODE=8, and global world size 32; resolved ${NUM_MACHINES}x${NPROC_PER_NODE}=${GLOBAL_WORLD_SIZE}." >&2
    exit 1
  fi
  if ((MACHINE_RANK < 0 || MACHINE_RANK > 3)); then
    echo "Error: +scale=${FORMAL_SCALE_NAME} requires node RANK in [0,3], got ${MACHINE_RANK}." >&2
    exit 1
  fi
fi

if [[ -z "${RUN_ID:-}" ]]; then
  if ((NUM_MACHINES > 1)); then
    echo "Error: an explicit identical RUN_ID is required on every multi-node pod." >&2
    exit 1
  elif ((LAUNCH_DRY_RUN_ENABLED)); then
    RUN_ID="dry-run"
  else
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  fi
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "Error: RUN_ID must be 1-128 safe characters: leading alphanumeric, then alphanumeric/dot/underscore/hyphen." >&2
  exit 1
fi

OUTPUT_DIR="./runs/${TASK_BASENAME}/${RUN_ID}"
FORMAL_HYDRA_OVERRIDES=()
GAUSSIAN_ENABLED=0
[[ "${TASK_BASENAME}" == *_gau1_* ]] && GAUSSIAN_ENABLED=1
CODE_COMMIT=""
PYPROJECT_SHA256=""
IMAGE_DIGEST_STATUS=""
FORMAL_OUTPUT_DIR=""
FORMAL_OUTPUT_PREFIX=""
OUTPUT_STORAGE_KIND=""
OUTPUT_ZERO_SMOKE_SHA256=""
FORMAL_RESUME_STATE_DIR=""
FORMAL_RESUME_STATE_MANIFEST=""
FORMAL_RESUME_STATE_MANIFEST_SHA256=""
FORMAL_RESUME_TRAINER_STATE_SHA256=""
FORMAL_RESUME_MODE="fresh_weights"
ERDMA_BOOTSTRAP_SHA256=""
TRAINING_ENV_BUNDLE_MANIFEST_SHA256=""
LOCAL_CPFS_BUNDLE_DIR=""
LOCAL_OSS_BUNDLE_DIR=""
LOCAL_DERIVED_STATS_PATH=""
PYTHON_TOOL="${FASTWAM_PYTHON:-python3}"
RESERVATION_ARGS=()

if ! resolved_python_tool="$(command -v "${PYTHON_TOOL}")"; then
  echo "Error: Python tool '${PYTHON_TOOL}' is required for preflight and Accelerate." >&2
  exit 1
fi
PYTHON_TOOL="${resolved_python_tool}"
export FASTWAM_PYTHON="${PYTHON_TOOL}"

if ((FORMAL_32GPU)); then
  if [[ "${TASK_BASENAME}" != robofactory_multi_robot_* ]]; then
    echo "Error: +scale=${FORMAL_SCALE_NAME} is reserved for a RoboFactory multi-robot task." >&2
    exit 1
  fi
  for arg in "${EXTRA_ARGS[@]}"; do
    case "${arg}" in
      output_dir=* | wandb.name=* | resume=* | checkpoint_state_kind=* | \
      data.train.root_dir=* | data.val.root_dir=* | \
      data.train.pretrained_norm_stats=* | data.val.pretrained_norm_stats=* | \
      data.train.text_embedding_cache_dir=* | data.val.text_embedding_cache_dir=* | \
      data.train.gaussian_cache_dir=* | data.val.gaussian_cache_dir=*)
        echo "Error: formal 32-GPU launcher owns provenance and node-local path override '${arg%%=*}'; remove the conflicting CLI override." >&2
        exit 1
        ;;
    esac
  done

  # The committed task and scale files are the complete scientific contract
  # for a formal arm. User-provided Hydra overrides are intentionally not a
  # customization surface here: accepting even one unclassified key can make
  # the RUN_ID/task label disagree with the actual treatment or schedule.
  formal_task_selector_count=0
  formal_scale_selector_count=0
  for arg in "${EXTRA_ARGS[@]}"; do
    case "${arg}" in
      task=*)
        formal_task="${arg#task=}"
        formal_task="${formal_task%.yaml}"
        if [[ ! "${formal_task}" =~ ^robofactory_multi_robot_vg[01]_hub[01]_gau[01]_224_1e-4$ ]]; then
          echo "Error: formal 32-GPU CLI task must be one of the eight explicit vg{0,1}_hub{0,1}_gau{0,1} arms, got '${formal_task}'." >&2
          exit 1
        fi
        formal_task_selector_count=$((formal_task_selector_count + 1))
        ;;
      +scale=*)
        formal_scale="${arg#+scale=}"
        formal_scale="${formal_scale%.yaml}"
        if [[ "${formal_scale}" != "${FORMAL_SCALE_NAME}" ]]; then
          echo "Error: formal 32-GPU CLI scale must be +scale=${FORMAL_SCALE_NAME}." >&2
          exit 1
        fi
        formal_scale_selector_count=$((formal_scale_selector_count + 1))
        ;;
      *)
        echo "Error: formal 32-GPU CLI allowlist accepts only one explicit task=<2x2x2-arm> and +scale=${FORMAL_SCALE_NAME}; user override/flag '${arg}' is forbidden because it can change the sealed treatment, data, schedule, eval, or checkpoint contract." >&2
        exit 1
        ;;
    esac
  done
  if ((formal_task_selector_count != 1 || formal_scale_selector_count != 1)); then
    echo "Error: formal 32-GPU CLI requires exactly one task selector and one fixed scale selector; got task=${formal_task_selector_count} scale=${formal_scale_selector_count}." >&2
    exit 1
  fi

  if ((!LAUNCH_DRY_RUN_ENABLED)); then
    for bypass_name in \
      FASTWAM_LAUNCHER_UNIT_TEST_ALLOW_DIRTY \
      FASTWAM_LAUNCHER_UNIT_TEST_SKIP_ENV_PREFLIGHT; do
      if is_enabled "${!bypass_name:-0}"; then
        echo "Error: formal non-dry-run launch forbids unit-test bypass ${bypass_name}; clean Git and the exact Python/pip-check preflight are mandatory." >&2
        exit 1
      else
        bypass_status=$?
        if ((bypass_status == 2)); then
          exit 1
        fi
      fi
    done
  fi

  if is_enabled "${FASTWAM_LOCAL_CACHE_ENABLED:-0}"; then
    :
  else
    formal_cache_status=$?
    if ((formal_cache_status == 2)); then
      exit 1
    fi
    echo "Error: formal 32-GPU training requires FASTWAM_LOCAL_CACHE_ENABLED=1 for checkpoint, dataset, text, stats, and compact-cache prewarm." >&2
    exit 1
  fi
  export FASTWAM_LOCAL_CACHE_REQUIRE_VERIFY_HIT=1
  export FASTWAM_LOCAL_CACHE_VERIFY_HIT=1
  if is_enabled "${FASTWAM_LOCAL_CACHE_ALLOW_SHARED_FS:-0}"; then
    echo "Error: formal 32-GPU training forbids FASTWAM_LOCAL_CACHE_ALLOW_SHARED_FS; use node-local /tmp storage." >&2
    exit 1
  else
    shared_fs_status=$?
    if ((shared_fs_status == 2)); then
      exit 1
    fi
  fi

  FORMAL_OUTPUT_DIR="${FASTWAM_FORMAL_OUTPUT_DIR:?FASTWAM_FORMAL_OUTPUT_DIR is required for formal 32-GPU training}"
  OUTPUT_DIR="${FORMAL_OUTPUT_DIR}"
  if ! command -v "${PYTHON_TOOL}" >/dev/null 2>&1; then
    echo "Error: Python tool '${PYTHON_TOOL}' is required for formal run provenance." >&2
    exit 1
  fi
  case "${FORMAL_OUTPUT_DIR}" in
    "${FORMAL_CPFS_PREFIX}/"*)
      FORMAL_OUTPUT_PREFIX="${FORMAL_CPFS_PREFIX}"
      OUTPUT_STORAGE_KIND=cpfs
      ;;
    "${FORMAL_OSS_PREFIX}/"*)
      FORMAL_OUTPUT_PREFIX="${FORMAL_OSS_PREFIX}"
      OUTPUT_STORAGE_KIND=oss_experimental
      require_sha256_environment FASTWAM_OUTPUT_ZERO_CHECKPOINT_SMOKE_SHA256 || exit 1
      output_smoke_marker="${FASTWAM_OUTPUT_ZERO_CHECKPOINT_SMOKE_MARKER:?FASTWAM_OUTPUT_ZERO_CHECKPOINT_SMOKE_MARKER is required for OSS output}"
      "${PYTHON_TOOL}" "${SCRIPT_DIR}/validate_zero_checkpoint_smoke.py" \
        --marker "${output_smoke_marker}" \
        --expected-sha256 "${FASTWAM_OUTPUT_ZERO_CHECKPOINT_SMOKE_SHA256}" \
        --output-parent "$(dirname "${FORMAL_OUTPUT_DIR}")" >/dev/null
      OUTPUT_ZERO_SMOKE_SHA256="${FASTWAM_OUTPUT_ZERO_CHECKPOINT_SMOKE_SHA256}"
      ;;
    *)
      echo "Error: formal output_dir must be under ${FORMAL_CPFS_PREFIX} or experimental ${FORMAL_OSS_PREFIX}." >&2
      exit 1
      ;;
  esac

  FORMAL_RESUME_STATE_DIR="${FASTWAM_FORMAL_RESUME_STATE_DIR:-}"
  FORMAL_RESUME_STATE_MANIFEST="${FASTWAM_FORMAL_RESUME_STATE_MANIFEST:-}"
  FORMAL_RESUME_STATE_MANIFEST_SHA256="${FASTWAM_FORMAL_RESUME_STATE_MANIFEST_SHA256:-}"
  FORMAL_RESUME_TRAINER_STATE_SHA256="${FASTWAM_FORMAL_RESUME_TRAINER_STATE_SHA256:-}"
  if [[ -n "${FORMAL_RESUME_STATE_DIR}" ]]; then
    FORMAL_RESUME_MODE="full_state"
    if [[ -z "${FORMAL_RESUME_STATE_MANIFEST}" ]]; then
      echo "Error: FASTWAM_FORMAL_RESUME_STATE_MANIFEST is required for full-state resume." >&2
      exit 1
    fi
    require_sha256_environment FASTWAM_FORMAL_RESUME_STATE_MANIFEST_SHA256 || exit 1
    FORMAL_RESUME_STATE_MANIFEST_SHA256="${FASTWAM_FORMAL_RESUME_STATE_MANIFEST_SHA256}"
    require_sha256_environment FASTWAM_FORMAL_RESUME_TRAINER_STATE_SHA256 || exit 1
    FORMAL_RESUME_TRAINER_STATE_SHA256="${FASTWAM_FORMAL_RESUME_TRAINER_STATE_SHA256}"
  elif [[ -n "${FORMAL_RESUME_STATE_MANIFEST}" || \
    -n "${FORMAL_RESUME_STATE_MANIFEST_SHA256}" || \
    -n "${FORMAL_RESUME_TRAINER_STATE_SHA256}" ]]; then
    echo "Error: formal resume manifest/hash variables require FASTWAM_FORMAL_RESUME_STATE_DIR." >&2
    exit 1
  fi

  cpfs_source_root="${FASTWAM_CPFS_BUNDLE_SOURCE_ROOT:?FASTWAM_CPFS_BUNDLE_SOURCE_ROOT is required}"
  cpfs_bundle_manifest="${FASTWAM_CPFS_BUNDLE_MANIFEST:?FASTWAM_CPFS_BUNDLE_MANIFEST is required}"
  oss_source_root="${FASTWAM_OSS_BUNDLE_SOURCE_ROOT:-}"
  oss_bundle_manifest="${FASTWAM_OSS_BUNDLE_MANIFEST:-}"
  if [[ "${cpfs_source_root}" != "${FORMAL_CPFS_PREFIX}" && "${cpfs_source_root}" != "${FORMAL_CPFS_PREFIX}/"* ]]; then
    echo "Error: data/checkpoint/VAE bundle source must be on CPFS under ${FORMAL_CPFS_PREFIX}." >&2
    exit 1
  fi
  if [[ "${cpfs_bundle_manifest}" != "${FORMAL_CPFS_PREFIX}/"* && \
    "${cpfs_bundle_manifest}" != "${FORMAL_OSS_PREFIX}/"* ]]; then
    echo "Error: CPFS source manifest must be an immutable file on CPFS or the mounted OSS root." >&2
    exit 1
  fi
  if ((GAUSSIAN_ENABLED)); then
    if [[ "${oss_source_root}" != "${FORMAL_OSS_PREFIX}" && "${oss_source_root}" != "${FORMAL_OSS_PREFIX}/"* ]] || \
      [[ "${oss_bundle_manifest}" != "${FORMAL_OSS_PREFIX}/"* ]]; then
      echo "Error: Gaussian compact bundle root and manifest must be on OSS under ${FORMAL_OSS_PREFIX}." >&2
      exit 1
    fi
  else
    for gaussian_name in \
      FASTWAM_OSS_BUNDLE_SOURCE_ROOT \
      FASTWAM_OSS_BUNDLE_MANIFEST \
      FASTWAM_OSS_BUNDLE_MANIFEST_SHA256 \
      FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT \
      FASTWAM_GAUSSIAN_CACHE_MANIFEST_SHA256 \
      FASTWAM_GAUSSIAN_CACHE_SELECTION_SHA256 \
      FASTWAM_GAUSSIAN_CACHE_SOURCE_IDENTITY_SHA256; do
      if [[ -n "${!gaussian_name:-}" ]]; then
        echo "Error: GAU0 formal arms forbid irrelevant Gaussian OSS input ${gaussian_name}; leave it unset so the baseline has no Gaussian asset dependency." >&2
        exit 1
      fi
    done
  fi
  if [[ -n "${FASTWAM_LOCAL_CACHE_ROOT:-}" && \
    "${FASTWAM_LOCAL_CACHE_ROOT}" != "${FORMAL_LOCAL_CACHE_ROOT}" ]]; then
    echo "Error: formal FASTWAM_LOCAL_CACHE_ROOT is fixed to ${FORMAL_LOCAL_CACHE_ROOT} so exact full-state resume reproduces dataset paths." >&2
    exit 1
  fi
  export FASTWAM_LOCAL_CACHE_ROOT="${FORMAL_LOCAL_CACHE_ROOT}"
  if [[ -n "${FASTWAM_LOCAL_RUNTIME_ROOT:-}" && \
    "${FASTWAM_LOCAL_RUNTIME_ROOT}" != "${FORMAL_LOCAL_RUNTIME_ROOT}" ]]; then
    echo "Error: formal FASTWAM_LOCAL_RUNTIME_ROOT is fixed to ${FORMAL_LOCAL_RUNTIME_ROOT} so exact full-state resume reproduces normalization paths." >&2
    exit 1
  fi
  export FASTWAM_LOCAL_RUNTIME_ROOT="${FORMAL_LOCAL_RUNTIME_ROOT}"

  require_sha256_environment FASTWAM_CPFS_BUNDLE_MANIFEST_SHA256 || exit 1
  require_sha256_environment FASTWAM_TRAINING_ENV_BUNDLE_MANIFEST_SHA256 || exit 1
  TRAINING_ENV_BUNDLE_MANIFEST_SHA256="${FASTWAM_TRAINING_ENV_BUNDLE_MANIFEST_SHA256}"
  for mapping_name in \
    FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH \
    FASTWAM_LOCAL_DATASET_RELATIVE_ROOT \
    FASTWAM_LOCAL_STATS_RELATIVE_PATH \
    FASTWAM_LOCAL_TEXT_EMBEDS_RELATIVE_ROOT \
    FASTWAM_LOCAL_MODEL_CACHE_RELATIVE_ROOT \
    FASTWAM_LOCAL_VAE_RELATIVE_PATH; do
    validate_safe_relative_path "${mapping_name}" "${!mapping_name:-}" || exit 1
  done
  if ((GAUSSIAN_ENABLED)); then
    require_sha256_environment FASTWAM_OSS_BUNDLE_MANIFEST_SHA256 || exit 1
    validate_safe_relative_path \
      FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT \
      "${FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT:-}" || exit 1
    require_sha256_environment FASTWAM_GAUSSIAN_CACHE_MANIFEST_SHA256 || exit 1
    require_sha256_environment FASTWAM_GAUSSIAN_CACHE_SELECTION_SHA256 || exit 1
    require_sha256_environment FASTWAM_GAUSSIAN_CACHE_SOURCE_IDENTITY_SHA256 || exit 1
  fi

  export FASTWAM_OFFICIAL_CHECKPOINT_SHA256="${FASTWAM_OFFICIAL_CHECKPOINT_SHA256:-${OFFICIAL_FASTWAM_CHECKPOINT_SHA256}}"
  FASTWAM_OFFICIAL_CHECKPOINT_SHA256="${FASTWAM_OFFICIAL_CHECKPOINT_SHA256,,}"
  if [[ "${FASTWAM_OFFICIAL_CHECKPOINT_SHA256}" != "${OFFICIAL_FASTWAM_CHECKPOINT_SHA256}" ]]; then
    echo "Error: formal run requires the official FastWAM checkpoint SHA-256 ${OFFICIAL_FASTWAM_CHECKPOINT_SHA256}." >&2
    exit 1
  fi
  export FASTWAM_VAE_SHA256="${FASTWAM_VAE_SHA256:-${OFFICIAL_WAN22_VAE_SHA256}}"
  FASTWAM_VAE_SHA256="${FASTWAM_VAE_SHA256,,}"
  if [[ "${FASTWAM_VAE_SHA256}" != "${OFFICIAL_WAN22_VAE_SHA256}" ]]; then
    echo "Error: formal run requires the verified Wan2.2 VAE SHA-256 ${OFFICIAL_WAN22_VAE_SHA256}." >&2
    exit 1
  fi
  if [[ "${FASTWAM_LOCAL_EXPECTED_H5_FILES:-24}" != "24" ]]; then
    echo "Error: formal RoboFactory N=2/3/4 bundle requires exactly 24 H5 files." >&2
    exit 1
  fi
  export FASTWAM_LOCAL_EXPECTED_H5_FILES=24
  ERDMA_BOOTSTRAP_SCRIPT="${REPO_ROOT}/docker/prepare-erdma-userspace.sh"
  if [[ ! -f "${ERDMA_BOOTSTRAP_SCRIPT}" || -L "${ERDMA_BOOTSTRAP_SCRIPT}" ]]; then
    echo "Error: formal eRDMA bootstrap must be a regular repo-owned file: ${ERDMA_BOOTSTRAP_SCRIPT}" >&2
    exit 1
  fi
  ERDMA_BOOTSTRAP_SHA256="$(sha256sum -- "${ERDMA_BOOTSTRAP_SCRIPT}")" || exit 1
  ERDMA_BOOTSTRAP_SHA256="${ERDMA_BOOTSTRAP_SHA256%% *}"
  export FASTWAM_ERDMA_BUNDLE_ROOT="/oss-chengjuntao/artifacts/erdma-userspace-${ERDMA_EXPECTED_VERSION}"
  export FASTWAM_ERDMA_EXPECTED_VERSION="${ERDMA_EXPECTED_VERSION}"
  export FASTWAM_ERDMA_EXPECTED_BUNDLE_SHA256="${ERDMA_EXPECTED_BUNDLE_SHA256}"
  export FASTWAM_ERDMA_EXPECTED_SOURCE_MANIFEST_SHA256="${ERDMA_EXPECTED_SOURCE_MANIFEST_SHA256}"
  export FASTWAM_ERDMA_EXPECTED_ENV_SHA256="${ERDMA_EXPECTED_ENV_SHA256}"

  if ! command -v git >/dev/null 2>&1; then
    echo "Error: git is required to seal the formal code identity." >&2
    exit 1
  fi
  CODE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --verify HEAD)" || exit 1
  declared_commit="${FASTWAM_CODE_COMMIT:-}"
  declared_commit="${declared_commit,,}"
  if [[ ! "${declared_commit}" =~ ^[0-9a-f]{40}$ || "${declared_commit}" != "${CODE_COMMIT}" ]]; then
    echo "Error: FASTWAM_CODE_COMMIT must be the exact current 40-hex HEAD ${CODE_COMMIT}." >&2
    exit 1
  fi
  export FASTWAM_CODE_COMMIT="${declared_commit}"
  PYPROJECT_SHA256="$(sha256sum -- "${REPO_ROOT}/pyproject.toml")" || exit 1
  PYPROJECT_SHA256="${PYPROJECT_SHA256%% *}"
  image_reference="${FASTWAM_DLC_IMAGE_REFERENCE:?FASTWAM_DLC_IMAGE_REFERENCE is required for formal image provenance}"
  image_digest="${FASTWAM_DLC_IMAGE_DIGEST:-}"
  image_digest="${image_digest,,}"
  if [[ -n "${image_digest}" ]]; then
    if [[ ! "${image_digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
      echo "Error: FASTWAM_DLC_IMAGE_DIGEST must use sha256:<64 lowercase hex>." >&2
      exit 1
    fi
    IMAGE_DIGEST_STATUS=resolved
  else
    if ((!LAUNCH_DRY_RUN_ENABLED)); then
      echo "Error: formal non-dry-run launch requires FASTWAM_DLC_IMAGE_DIGEST=sha256:<64hex>; mutable image tags cannot be acknowledged for execution." >&2
      exit 1
    elif is_enabled "${FASTWAM_ACK_MUTABLE_IMAGE_TAG_RISK:-0}"; then
      IMAGE_DIGEST_STATUS=unresolved_mutable_tag
      echo "[formal_gate] warning=mutable_image_tag digest=UNRESOLVED reference=${image_reference}" >&2
    else
      image_ack_status=$?
      if ((image_ack_status == 2)); then
        exit 1
      fi
      echo "Error: image digest is unresolved; set FASTWAM_ACK_MUTABLE_IMAGE_TAG_RISK=1 to record and accept the mutable-tag risk." >&2
      exit 1
    fi
  fi
  if ((!LAUNCH_DRY_RUN_ENABLED)) && [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=all)" ]]; then
    echo "Error: formal 32-GPU training requires a clean immutable Git worktree." >&2
    exit 1
  fi

  LOCAL_CPFS_BUNDLE_DIR="${FASTWAM_LOCAL_CACHE_ROOT%/}/cpfs/${FASTWAM_CPFS_BUNDLE_MANIFEST_SHA256}"
  if ((GAUSSIAN_ENABLED)); then
    LOCAL_OSS_BUNDLE_DIR="${FASTWAM_LOCAL_CACHE_ROOT%/}/oss/${FASTWAM_OSS_BUNDLE_MANIFEST_SHA256}"
  fi
  export FASTWAM_LOCAL_CHECKPOINT_PATH="${LOCAL_CPFS_BUNDLE_DIR}/${FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH}"
  export FASTWAM_LOCAL_DATASET_ROOT="${LOCAL_CPFS_BUNDLE_DIR}/${FASTWAM_LOCAL_DATASET_RELATIVE_ROOT}"
  export FASTWAM_LOCAL_STATS_SOURCE_PATH="${LOCAL_CPFS_BUNDLE_DIR}/${FASTWAM_LOCAL_STATS_RELATIVE_PATH}"
  export FASTWAM_LOCAL_TEXT_EMBEDS_ROOT="${LOCAL_CPFS_BUNDLE_DIR}/${FASTWAM_LOCAL_TEXT_EMBEDS_RELATIVE_ROOT}"
  export DIFFSYNTH_MODEL_BASE_PATH="${LOCAL_CPFS_BUNDLE_DIR}/${FASTWAM_LOCAL_MODEL_CACHE_RELATIVE_ROOT}"
  export FASTWAM_LOCAL_VAE_PATH="${LOCAL_CPFS_BUNDLE_DIR}/${FASTWAM_LOCAL_VAE_RELATIVE_PATH}"
  if ((GAUSSIAN_ENABLED)); then
    export FASTWAM_GAUSSIAN_CACHE_DIR="${LOCAL_OSS_BUNDLE_DIR}/${FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT}"
  fi
  LOCAL_DERIVED_STATS_PATH="${FASTWAM_LOCAL_RUNTIME_ROOT%/}/${FASTWAM_CPFS_BUNDLE_MANIFEST_SHA256}/dataset_stats.json"

  if ! command -v "${PYTHON_TOOL}" >/dev/null 2>&1; then
    echo "Error: Python tool '${PYTHON_TOOL}' is required for formal run provenance." >&2
    exit 1
  fi
  RESERVATION_ARGS=(
    --output-dir "${FORMAL_OUTPUT_DIR}"
    --allowed-prefix "${FORMAL_OUTPUT_PREFIX}"
    --source-root "${REPO_ROOT}"
    --run-id "${RUN_ID}"
    --code-commit "${CODE_COMMIT}"
    --task "${TASK_BASENAME}"
    --num-machines "${NUM_MACHINES}"
    --nproc-per-node "${NPROC_PER_NODE}"
    --expected-global-world-size 32
    --cpfs-bundle-manifest-sha256 "${FASTWAM_CPFS_BUNDLE_MANIFEST_SHA256}"
    --oss-bundle-manifest-sha256 "${FASTWAM_OSS_BUNDLE_MANIFEST_SHA256:-}"
    --cache-manifest-sha256 "${FASTWAM_GAUSSIAN_CACHE_MANIFEST_SHA256:-}"
    --cache-selection-sha256 "${FASTWAM_GAUSSIAN_CACHE_SELECTION_SHA256:-}"
    --cache-source-identity-sha256 "${FASTWAM_GAUSSIAN_CACHE_SOURCE_IDENTITY_SHA256:-}"
    --checkpoint-sha256 "${FASTWAM_OFFICIAL_CHECKPOINT_SHA256}"
    --vae-sha256 "${FASTWAM_VAE_SHA256}"
    --stats-sha256 "${OFFICIAL_N234_TRAIN_S42_STATS_SHA256}"
    --erdma-bootstrap-sha256 "${ERDMA_BOOTSTRAP_SHA256}"
    --erdma-bundle-sha256 "${ERDMA_EXPECTED_BUNDLE_SHA256}"
    --erdma-source-manifest-sha256 "${ERDMA_EXPECTED_SOURCE_MANIFEST_SHA256}"
    --erdma-env-sha256 "${ERDMA_EXPECTED_ENV_SHA256}"
    --training-env-bundle-manifest-sha256 "${TRAINING_ENV_BUNDLE_MANIFEST_SHA256}"
    --image-reference "${image_reference}"
    --image-digest-status "${IMAGE_DIGEST_STATUS}"
    --image-digest "${image_digest}"
    --pyproject-sha256 "${PYPROJECT_SHA256}"
    --output-storage "${OUTPUT_STORAGE_KIND}"
    --output-zero-checkpoint-smoke-sha256 "${OUTPUT_ZERO_SMOKE_SHA256}"
    --resume-state-dir "${FORMAL_RESUME_STATE_DIR}"
    --resume-state-manifest "${FORMAL_RESUME_STATE_MANIFEST}"
    --resume-state-manifest-sha256 "${FORMAL_RESUME_STATE_MANIFEST_SHA256}"
    --resume-trainer-state-sha256 "${FORMAL_RESUME_TRAINER_STATE_SHA256}"
    --timeout "${FASTWAM_RUN_RESERVATION_TIMEOUT:-300}"
    --resume-timeout "${FASTWAM_RESUME_VALIDATION_TIMEOUT:-21600}"
  )
  "${PYTHON_TOOL}" "${SCRIPT_DIR}/reserve_dlc_run.py" \
    --mode validate "${RESERVATION_ARGS[@]}" >/dev/null
fi

PREFLIGHT_MODE="${FASTWAM_DLC_PREFLIGHT:-auto}"
case "${PREFLIGHT_MODE,,}" in
  auto)
    if ((NUM_MACHINES > 1)); then
      PREFLIGHT_ENABLED=1
    else
      PREFLIGHT_ENABLED=0
    fi
    ;;
  1 | true | yes | on) PREFLIGHT_ENABLED=1 ;;
  0 | false | no | off) PREFLIGHT_ENABLED=0 ;;
  *)
    echo "Error: FASTWAM_DLC_PREFLIGHT must be auto or a boolean, got '${PREFLIGHT_MODE}'." >&2
    exit 1
    ;;
esac

if ((FORMAL_32GPU)); then
  if ((!PREFLIGHT_ENABLED)); then
    echo "Error: formal 32-GPU training cannot disable FASTWAM_DLC_PREFLIGHT." >&2
    exit 1
  fi
  export FASTWAM_PREFLIGHT_REQUIRE_ERDMA=1
  export FASTWAM_PREFLIGHT_MIN_ALGBW_GBPS="${FASTWAM_PREFLIGHT_MIN_ALGBW_GBPS:-5.0}"
  export FASTWAM_PREFLIGHT_BANDWIDTH_MIB="${FASTWAM_PREFLIGHT_BANDWIDTH_MIB:-256}"
  export FASTWAM_PREFLIGHT_BANDWIDTH_WARMUP="${FASTWAM_PREFLIGHT_BANDWIDTH_WARMUP:-2}"
  export FASTWAM_PREFLIGHT_BANDWIDTH_ITERS="${FASTWAM_PREFLIGHT_BANDWIDTH_ITERS:-5}"
  # The first node can reach rendezvous while another still verifies tens of
  # GiB of node-local cache. Keep this gate at the cache-wait timescale rather
  # than the diagnostic 180-second default.
  export FASTWAM_PREFLIGHT_TIMEOUT="${FASTWAM_PREFLIGHT_TIMEOUT:-7200}"
  export FASTWAM_PREFLIGHT_OUTER_TIMEOUT="${FASTWAM_PREFLIGHT_OUTER_TIMEOUT:-7260}"
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT,NET
fi

if ((!LAUNCH_DRY_RUN_ENABLED)); then
  # Ordered fail-closed startup: exact Python environment first, then the
  # host-driver shim and local CUDA, node-local whole-file cache, and one global
  # all-reduce. The preflight and Accelerate rendezvous reuse MASTER_PORT
  # sequentially, never concurrently.
  source "$(dirname "${BASH_SOURCE[0]}")/dlc_preflight.sh"
  if is_enabled "${FASTWAM_LAUNCHER_UNIT_TEST_SKIP_ENV_PREFLIGHT:-0}"; then
    echo "[preflight] stage=python_environment action=SKIP reason=launcher_unit_test_override" >&2
  else
    env_skip_status=$?
    if ((env_skip_status == 2)); then
      exit 1
    fi
    fastwam_run_python_environment_preflight \
      "$(dirname "${BASH_SOURCE[0]}")/../pyproject.toml"
  fi

  if ((PREFLIGHT_ENABLED)); then
    fastwam_prepare_nvidia_host570
    fastwam_run_local_cuda_preflight "${NPROC_PER_NODE}" "${MACHINE_RANK}"
  fi

  if is_enabled "${FASTWAM_LOCAL_CACHE_ENABLED:-0}"; then
    if ((NUM_MACHINES > 1)); then
      export FASTWAM_LOCAL_CACHE_REQUIRE_VERIFY_HIT=1
    fi
    if ((FORMAL_32GPU)); then
      source "${SCRIPT_DIR}/dlc_multi_source_cache.sh"
      fastwam_prepare_multi_source_cache
    else
      source "${SCRIPT_DIR}/dlc_local_cache.sh"
      fastwam_prepare_local_cache
    fi
  else
    cache_status=$?
    if ((cache_status == 2)); then
      exit 1
    fi
  fi

  if ((FORMAL_32GPU)); then
    if [[ "${FASTWAM_LOCAL_CPFS_CACHE_MANIFEST_SHA256:-}" != "${FASTWAM_CPFS_BUNDLE_MANIFEST_SHA256}" ]]; then
      echo "Error: prepared CPFS node-local bundle identity does not match the formal pin." >&2
      exit 1
    fi
    if ((GAUSSIAN_ENABLED)); then
      if [[ "${FASTWAM_LOCAL_OSS_CACHE_MANIFEST_SHA256:-}" != "${FASTWAM_OSS_BUNDLE_MANIFEST_SHA256}" ]]; then
        echo "Error: prepared Gaussian OSS node-local bundle identity does not match the formal pin." >&2
        exit 1
      fi
    elif [[ -n "${FASTWAM_LOCAL_OSS_CACHE_MANIFEST_SHA256:-}" ]]; then
      echo "Error: GAU0 unexpectedly prepared an OSS Gaussian bundle." >&2
      exit 1
    fi
    if [[ "${FASTWAM_LOCAL_STATS_MANIFEST_SHA256:-}" != "${OFFICIAL_N234_TRAIN_S42_STATS_SHA256}" ]]; then
      echo "Error: prepared N=2/3/4 train-s42 stats do not match the fixed formal SHA-256." >&2
      exit 1
    fi
    # This launcher is a non-login shell. Source the verified helper here so
    # its provider/library exports survive both NCCL preflight and Accelerate.
    source "${ERDMA_BOOTSTRAP_SCRIPT}"
    fastwam_prepare_erdma_userspace
    if [[ "${FASTWAM_ERDMA_BUNDLE_SHA256:-}" != "${ERDMA_EXPECTED_BUNDLE_SHA256}" ]] || \
      [[ "${FASTWAM_ERDMA_SOURCE_MANIFEST_SHA256:-}" != "${ERDMA_EXPECTED_SOURCE_MANIFEST_SHA256}" ]] || \
      [[ "${FASTWAM_ERDMA_ENV_SHA256:-}" != "${ERDMA_EXPECTED_ENV_SHA256}" ]]; then
      echo "Error: prepared eRDMA runtime identity does not match the formal pins." >&2
      exit 1
    fi
    export NCCL_IB_HCA="${NCCL_IB_HCA:-erdma}"
    PREPARE_BUNDLE_ARGS=(
      --bundle-root "${FASTWAM_LOCAL_CPFS_CACHE_DIR}"
      --bundle-manifest-sha256 "${FASTWAM_LOCAL_CPFS_CACHE_MANIFEST_SHA256}"
      --dataset-root "${FASTWAM_LOCAL_DATASET_ROOT}"
      --expected-h5-files "${FASTWAM_LOCAL_EXPECTED_H5_FILES}"
      --stats-source "${FASTWAM_LOCAL_STATS_SOURCE_PATH}"
      --text-embeds-root "${FASTWAM_LOCAL_TEXT_EMBEDS_ROOT}"
      --checkpoint "${FASTWAM_LOCAL_CHECKPOINT_PATH}"
      --checkpoint-manifest-sha256 "${FASTWAM_LOCAL_CHECKPOINT_MANIFEST_SHA256}"
      --expected-checkpoint-sha256 "${FASTWAM_OFFICIAL_CHECKPOINT_SHA256}"
      --model-cache-root "${DIFFSYNTH_MODEL_BASE_PATH}"
      --vae "${FASTWAM_LOCAL_VAE_PATH}"
      --vae-manifest-sha256 "${FASTWAM_LOCAL_VAE_MANIFEST_SHA256}"
      --expected-vae-sha256 "${FASTWAM_VAE_SHA256}"
      --output-stats "${LOCAL_DERIVED_STATS_PATH}"
    )
    if ((GAUSSIAN_ENABLED)); then
      PREPARE_BUNDLE_ARGS+=(
        --gaussian-bundle-root "${FASTWAM_LOCAL_OSS_CACHE_DIR}"
        --gaussian-root "${FASTWAM_GAUSSIAN_CACHE_DIR}"
      )
    fi
    "${PYTHON_TOOL}" "${SCRIPT_DIR}/prepare_local_training_bundle.py" \
      "${PREPARE_BUNDLE_ARGS[@]}"
  fi

  if ((PREFLIGHT_ENABLED)); then
    fastwam_run_global_allreduce_preflight \
      "${NPROC_PER_NODE}" \
      "${NUM_MACHINES}" \
      "${MACHINE_RANK}" \
      "${MAIN_PROCESS_IP}" \
      "${MAIN_PROCESS_PORT}" \
      "${RUN_ID}"
  fi

  if ((FORMAL_32GPU)); then
    if [[ "${FORMAL_RESUME_MODE}" == "full_state" ]]; then
      # Resume is the sole allowed reuse case: it must point into this exact
      # previously reserved run and match that run's immutable identity.
      if ((MACHINE_RANK == 0)); then
        reservation_mode=validate-existing
      else
        reservation_mode=wait-existing
      fi
    elif ((MACHINE_RANK == 0)); then
      reservation_mode=owner
    else
      reservation_mode=wait
    fi
    "${PYTHON_TOOL}" "${SCRIPT_DIR}/reserve_dlc_run.py" \
      --mode "${reservation_mode}" "${RESERVATION_ARGS[@]}"
  fi
fi

if ((FORMAL_32GPU)); then
  if [[ "${FORMAL_RESUME_MODE}" == "full_state" ]]; then
    FORMAL_RESUME_VALUE="${FORMAL_RESUME_STATE_DIR}"
  else
    FORMAL_RESUME_VALUE="${FASTWAM_LOCAL_CHECKPOINT_PATH}"
  fi
  FORMAL_HYDRA_OVERRIDES=(
    "resume=${FORMAL_RESUME_VALUE}"
    "checkpoint_state_kind=full"
    "data.train.root_dir=${FASTWAM_LOCAL_DATASET_ROOT}"
    "data.val.root_dir=${FASTWAM_LOCAL_DATASET_ROOT}"
    "data.train.pretrained_norm_stats=${LOCAL_DERIVED_STATS_PATH}"
    "data.val.pretrained_norm_stats=${LOCAL_DERIVED_STATS_PATH}"
    "data.train.text_embedding_cache_dir=${FASTWAM_LOCAL_TEXT_EMBEDS_ROOT}"
    "data.val.text_embedding_cache_dir=${FASTWAM_LOCAL_TEXT_EMBEDS_ROOT}"
  )
  if ((GAUSSIAN_ENABLED)); then
    FORMAL_HYDRA_OVERRIDES+=(
      "data.train.gaussian_cache_dir=${FASTWAM_GAUSSIAN_CACHE_DIR}"
      "data.val.gaussian_cache_dir=${FASTWAM_GAUSSIAN_CACHE_DIR}"
    )
  fi
fi

echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} global_world_size=${GLOBAL_WORLD_SIZE} master=${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT} run_id=${RUN_ID} output_dir=${OUTPUT_DIR}"

ACCELERATE_COMMAND=(
  "${PYTHON_TOOL}" -m accelerate.commands.launch
  --config_file "${REPO_ROOT}/scripts/accelerate_configs/accelerate_zero2_ds.yaml"
  --num_machines "${NUM_MACHINES}"
  --machine_rank "${MACHINE_RANK}"
  --main_process_ip "${MAIN_PROCESS_IP}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  --num_processes "${GLOBAL_WORLD_SIZE}"
  --deepspeed_multinode_launcher standard
  "${REPO_ROOT}/scripts/train.py"
  "${EXTRA_ARGS[@]}"
  "output_dir=${OUTPUT_DIR}"
  "wandb.name=${TASK_BASENAME}-${RUN_ID}"
  "${FORMAL_HYDRA_OVERRIDES[@]}"
)

if ((LAUNCH_DRY_RUN_ENABLED)); then
  printf '[dry_run]'
  printf ' %q' "${ACCELERATE_COMMAND[@]}"
  printf '\n'
else
  # PAI's WORLD_SIZE/RANK describe nodes, while the training subprocesses need
  # torchrun's global process values. Accelerate recreates them from CLI args.
  unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
  cd "${REPO_ROOT}"
  exec "${ACCELERATE_COMMAND[@]}"
fi
