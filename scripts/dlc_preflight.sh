#!/usr/bin/env bash

# Source this file before launching Accelerate so the repaired host-driver
# library path remains exported in the training process. Direct execution is
# also supported for diagnostics. PAI DLC defines WORLD_SIZE/RANK as node
# count/node rank and NPROC_PER_NODE as the number of local workers.

_fastwam_prepend_library_path() {
  local directory="$1"
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${directory}:"*) ;;
    *) export LD_LIBRARY_PATH="${directory}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

_fastwam_preflight_resolve_python() {
  if [[ -n "${FASTWAM_PYTHON:-}" ]]; then
    printf '%s\n' "${FASTWAM_PYTHON}"
  elif command -v python >/dev/null 2>&1; then
    printf '%s\n' python
  elif command -v python3 >/dev/null 2>&1; then
    printf '%s\n' python3
  else
    echo "Error: neither python nor python3 is available in PATH." >&2
    return 1
  fi
}

_fastwam_preflight_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || ((10#${value} < 1)); then
    echo "Error: ${name} (${value}) must be a positive integer." >&2
    return 1
  fi
}

_fastwam_preflight_nonnegative_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "Error: ${name} (${value}) must be a non-negative integer." >&2
    return 1
  fi
}

_fastwam_preflight_bool() {
  case "${1,,}" in
    1 | true | yes | on) return 0 ;;
    0 | false | no | off) return 1 ;;
    *)
      echo "Error: expected a boolean value, got '${1}'." >&2
      return 2
      ;;
  esac
}

_fastwam_validate_nccl_transport_log() {
  local log_path="$1"
  if [[ ! -f "${log_path}" || -L "${log_path}" ]]; then
    echo "Error: NCCL transport log must be a regular non-symlink file: ${log_path}" >&2
    return 1
  fi
  if grep -Eiq 'NET/IB[[:space:]]*:[[:space:]]*No device found' "${log_path}"; then
    echo "Error: NCCL eRDMA gate rejected 'NET/IB : No device found'." >&2
    return 1
  fi
  if grep -Eiq 'NET/Socket[[:space:]]*:[[:space:]]*Using|Using network Socket' "${log_path}"; then
    echo "Error: NCCL eRDMA gate rejected Socket transport fallback." >&2
    return 1
  fi
  if ! grep -Eiq 'NET/IB.*erdma' "${log_path}"; then
    echo "Error: NCCL eRDMA gate found no NET/IB eRDMA transport evidence." >&2
    return 1
  fi
  echo "[preflight] transport=erdma status=PASS log=${log_path}"
}

fastwam_run_python_environment_preflight() {
  local pyproject_path="${1:-$(dirname "${BASH_SOURCE[0]}")/../pyproject.toml}"
  local pip_check_timeout="${FASTWAM_PIP_CHECK_TIMEOUT:-120}"
  local python_bin
  local status

  _fastwam_preflight_positive_integer FASTWAM_PIP_CHECK_TIMEOUT "${pip_check_timeout}" || return 1
  python_bin="$(_fastwam_preflight_resolve_python)" || return 1
  echo "[preflight] stage=python_environment pyproject=${pyproject_path}"
  if "${python_bin}" "$(dirname "${BASH_SOURCE[0]}")/validate_python_environment.py" \
    --pyproject "${pyproject_path}" \
    --pip-check-timeout "${pip_check_timeout}"; then
    :
  else
    status=$?
    echo "Error: Python environment preflight failed with status=${status}." >&2
    return "${status}"
  fi
}

fastwam_prepare_nvidia_host570() {
  local mode="${FASTWAM_NVIDIA_HOST570_FIX:-auto}"
  local driver_version="${FASTWAM_NVIDIA_DRIVER_VERSION:-570.153.02}"
  local host_lib_dir="${FASTWAM_NVIDIA_HOST_LIB_DIR:-/usr/lib/x86_64-linux-gnu}"
  local shim_dir="${FASTWAM_NVIDIA_SHIM_DIR:-/tmp/fastwam-nvidia-${driver_version}}"
  local cuda_lib_dir="${FASTWAM_CUDA_LIB_DIR:-/usr/local/cuda/lib64}"
  local source_cuda="${host_lib_dir}/libcuda.so.${driver_version}"
  local source_nvml="${host_lib_dir}/libnvidia-ml.so.${driver_version}"

  case "${mode,,}" in
    0 | false | no | off)
      echo "[nvidia_host_fix] mode=disabled"
      return 0
      ;;
    auto)
      if [[ ! -f "${source_cuda}" || ! -f "${source_nvml}" ]]; then
        echo "[nvidia_host_fix] mode=auto action=skip reason=host570_libraries_not_found"
        return 0
      fi
      ;;
    1 | true | yes | on) ;;
    *)
      echo "Error: FASTWAM_NVIDIA_HOST570_FIX must be auto or a boolean, got '${mode}'." >&2
      return 1
      ;;
  esac

  if [[ ! -f "${source_cuda}" || ! -f "${source_nvml}" ]]; then
    echo "Error: requested host-driver fix but missing ${source_cuda} or ${source_nvml}." >&2
    return 1
  fi
  if [[ -z "${shim_dir}" || "${shim_dir}" == "/" ]]; then
    echo "Error: FASTWAM_NVIDIA_SHIM_DIR must be a specific directory." >&2
    return 1
  fi

  mkdir -p -- "${shim_dir}" || return 1
  ln -sfn -- "${source_cuda}" "${shim_dir}/libcuda.so" || return 1
  ln -sfn -- "${source_cuda}" "${shim_dir}/libcuda.so.1" || return 1
  ln -sfn -- "${source_nvml}" "${shim_dir}/libnvidia-ml.so" || return 1
  ln -sfn -- "${source_nvml}" "${shim_dir}/libnvidia-ml.so.1" || return 1

  if [[ -d "${cuda_lib_dir}" ]]; then
    _fastwam_prepend_library_path "${cuda_lib_dir}"
  fi
  # Keep the exact host-driver shim ahead of CUDA toolkit/image libraries.
  _fastwam_prepend_library_path "${shim_dir}"
  export FASTWAM_NVIDIA_SHIM_DIR="${shim_dir}"
  echo "[nvidia_host_fix] mode=${mode} driver=${driver_version} shim_dir=${shim_dir} action=applied"
}

fastwam_run_local_cuda_preflight() {
  local nproc_per_node="${1:?nproc_per_node is required}"
  local machine_rank="${2:?machine_rank is required}"
  local allocation_mib="${FASTWAM_PREFLIGHT_ALLOCATION_MIB:-16}"
  local python_bin
  local status

  _fastwam_preflight_positive_integer nproc_per_node "${nproc_per_node}" || return 1
  if [[ ! "${machine_rank}" =~ ^[0-9]+$ ]]; then
    echo "Error: machine_rank (${machine_rank}) must be a non-negative integer." >&2
    return 1
  fi
  _fastwam_preflight_positive_integer FASTWAM_PREFLIGHT_ALLOCATION_MIB "${allocation_mib}" || return 1
  nproc_per_node=$((10#${nproc_per_node}))
  machine_rank=$((10#${machine_rank}))
  allocation_mib=$((10#${allocation_mib}))
  python_bin="$(_fastwam_preflight_resolve_python)" || return 1

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi is not available in PATH." >&2
    return 1
  fi
  echo "[preflight] stage=nvidia_smi node_rank=${machine_rank}"
  if nvidia-smi; then
    :
  else
    status=$?
    echo "Error: nvidia-smi preflight failed on node_rank=${machine_rank} with status=${status}." >&2
    return "${status}"
  fi

  echo "[preflight] stage=torch_devices node_rank=${machine_rank} expected_local_devices=${nproc_per_node}"
  if "${python_bin}" "$(dirname "${BASH_SOURCE[0]}")/validate_cuda_devices.py" \
    --expected "${nproc_per_node}" \
    --allocation-mib "${allocation_mib}"; then
    :
  else
    status=$?
    echo "Error: local CUDA preflight failed on node_rank=${machine_rank} with status=${status}." >&2
    return "${status}"
  fi
}

fastwam_run_global_allreduce_preflight() {
  local nproc_per_node="${1:?nproc_per_node is required}"
  local num_machines="${2:?num_machines is required}"
  local machine_rank="${3:?machine_rank is required}"
  local master_addr="${4:?master_addr is required}"
  local master_port="${5:?master_port is required}"
  local run_id="${6:?run_id is required}"
  local timeout_s="${FASTWAM_PREFLIGHT_TIMEOUT:-180}"
  local outer_timeout_s="${FASTWAM_PREFLIGHT_OUTER_TIMEOUT:-}"
  local kill_after_s="${FASTWAM_PREFLIGHT_KILL_AFTER:-15}"
  local timeout_bin="${FASTWAM_TIMEOUT_BIN:-timeout}"
  local bandwidth_mib="${FASTWAM_PREFLIGHT_BANDWIDTH_MIB:-256}"
  local bandwidth_warmup="${FASTWAM_PREFLIGHT_BANDWIDTH_WARMUP:-2}"
  local bandwidth_iters="${FASTWAM_PREFLIGHT_BANDWIDTH_ITERS:-5}"
  local min_algbw_gbps="${FASTWAM_PREFLIGHT_MIN_ALGBW_GBPS:-5.0}"
  local require_erdma="${FASTWAM_PREFLIGHT_REQUIRE_ERDMA:-0}"
  local require_erdma_enabled
  local transport_log=""
  local python_bin
  local world_size
  local status
  local -a pipeline_status
  local -a command

  _fastwam_preflight_positive_integer nproc_per_node "${nproc_per_node}" || return 1
  _fastwam_preflight_positive_integer num_machines "${num_machines}" || return 1
  if [[ ! "${machine_rank}" =~ ^[0-9]+$ ]]; then
    echo "Error: machine_rank (${machine_rank}) must be a non-negative integer." >&2
    return 1
  fi
  _fastwam_preflight_positive_integer master_port "${master_port}" || return 1
  _fastwam_preflight_positive_integer FASTWAM_PREFLIGHT_TIMEOUT "${timeout_s}" || return 1
  _fastwam_preflight_positive_integer FASTWAM_PREFLIGHT_KILL_AFTER "${kill_after_s}" || return 1
  _fastwam_preflight_positive_integer FASTWAM_PREFLIGHT_BANDWIDTH_MIB "${bandwidth_mib}" || return 1
  _fastwam_preflight_nonnegative_integer FASTWAM_PREFLIGHT_BANDWIDTH_WARMUP "${bandwidth_warmup}" || return 1
  _fastwam_preflight_positive_integer FASTWAM_PREFLIGHT_BANDWIDTH_ITERS "${bandwidth_iters}" || return 1
  if [[ ! "${min_algbw_gbps}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
    [[ "${min_algbw_gbps}" =~ ^0*([.]0*)?$ ]]; then
    echo "Error: FASTWAM_PREFLIGHT_MIN_ALGBW_GBPS (${min_algbw_gbps}) must be positive." >&2
    return 1
  fi
  if _fastwam_preflight_bool "${require_erdma}"; then
    require_erdma_enabled=1
  else
    status=$?
    if ((status == 2)); then
      return 1
    fi
    require_erdma_enabled=0
  fi
  if [[ -z "${outer_timeout_s}" ]]; then
    outer_timeout_s=$((10#${timeout_s} + 60))
  fi
  _fastwam_preflight_positive_integer FASTWAM_PREFLIGHT_OUTER_TIMEOUT "${outer_timeout_s}" || return 1
  if [[ ! "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    echo "Error: preflight run_id must use the launcher's safe RUN_ID format." >&2
    return 1
  fi

  nproc_per_node=$((10#${nproc_per_node}))
  num_machines=$((10#${num_machines}))
  machine_rank=$((10#${machine_rank}))
  master_port=$((10#${master_port}))
  timeout_s=$((10#${timeout_s}))
  outer_timeout_s=$((10#${outer_timeout_s}))
  kill_after_s=$((10#${kill_after_s}))
  bandwidth_mib=$((10#${bandwidth_mib}))
  bandwidth_warmup=$((10#${bandwidth_warmup}))
  bandwidth_iters=$((10#${bandwidth_iters}))
  if ((machine_rank >= num_machines || master_port > 65535)); then
    echo "Error: invalid DLC preflight rank or port." >&2
    return 1
  fi
  if ((num_machines > 1)) && [[ \
    "${master_addr}" == "localhost" || \
    "${master_addr}" == "::1" || \
    "${master_addr}" == 127.* \
  ]]; then
    echo "Error: multi-machine preflight master_addr must be reachable by every machine." >&2
    return 1
  fi
  if ! command -v "${timeout_bin}" >/dev/null 2>&1; then
    echo "Error: timeout command '${timeout_bin}' is not available." >&2
    return 1
  fi
  if ! command -v tee >/dev/null 2>&1; then
    echo "Error: tee is required to audit NCCL transport logs." >&2
    return 1
  fi
  python_bin="$(_fastwam_preflight_resolve_python)" || return 1
  world_size=$((nproc_per_node * num_machines))

  command=(
    "${timeout_bin}"
    --foreground
    --signal=TERM
    --kill-after="${kill_after_s}s"
    "${outer_timeout_s}s"
    "${python_bin}" -m torch.distributed.run
    --nnodes "${num_machines}"
    --nproc-per-node "${nproc_per_node}"
    --node-rank "${machine_rank}"
    --master-addr "${master_addr}"
    --master-port "${master_port}"
    --rdzv-backend static
    --rdzv-id "${run_id}-preflight"
    --rdzv-conf "timeout=${timeout_s}"
    "$(dirname "${BASH_SOURCE[0]}")/validate_distributed_cuda.py"
    --expected-world-size "${world_size}"
    --expected-local-world-size "${nproc_per_node}"
    --expected-num-nodes "${num_machines}"
    --timeout "${timeout_s}"
    --bandwidth-mib "${bandwidth_mib}"
    --bandwidth-warmup "${bandwidth_warmup}"
    --bandwidth-iters "${bandwidth_iters}"
    --min-algbw-gbps "${min_algbw_gbps}"
  )

  transport_log="$(mktemp "/tmp/fastwam-nccl-${run_id}-node${machine_rank}.XXXXXX.log")" || return 1
  echo "[preflight] stage=distributed_all_reduce node_rank=${machine_rank} world_size=${world_size} rendezvous=${master_addr}:${master_port} rdzv_timeout_s=${timeout_s} outer_timeout_s=${outer_timeout_s} bandwidth_mib=${bandwidth_mib} min_algbw_gbps=${min_algbw_gbps} require_erdma=${require_erdma_enabled}"
  if NCCL_DEBUG="${NCCL_DEBUG:-INFO}" \
    NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}" \
    "${command[@]}" 2>&1 | tee "${transport_log}"; then
    pipeline_status=("${PIPESTATUS[@]}")
  else
    pipeline_status=("${PIPESTATUS[@]}")
  fi
  if ((pipeline_status[0] != 0 || pipeline_status[1] != 0)); then
    if ((pipeline_status[0] != 0)); then
      status="${pipeline_status[0]}"
    else
      status="${pipeline_status[1]}"
    fi
    echo "Error: distributed all-reduce preflight failed on node_rank=${machine_rank} with status=${status}." >&2
    rm -f -- "${transport_log}"
    return "${status}"
  fi
  if ((require_erdma_enabled)); then
    if ! _fastwam_validate_nccl_transport_log "${transport_log}"; then
      rm -f -- "${transport_log}"
      return 1
    fi
  fi
  rm -f -- "${transport_log}"
  echo "[preflight] status=PASS node_rank=${machine_rank} world_size=${world_size}"
}

fastwam_run_dlc_preflight() {
  local nproc_per_node="${1:?nproc_per_node is required}"
  local num_machines="${2:?num_machines is required}"
  local machine_rank="${3:?machine_rank is required}"
  local master_addr="${4:?master_addr is required}"
  local master_port="${5:?master_port is required}"
  local run_id="${6:-${RUN_ID:-preflight}}"
  local pyproject_path="${7:-$(dirname "${BASH_SOURCE[0]}")/../pyproject.toml}"

  fastwam_run_python_environment_preflight "${pyproject_path}" || return $?
  fastwam_prepare_nvidia_host570 || return $?
  fastwam_run_local_cuda_preflight "${nproc_per_node}" "${machine_rank}" || return $?
  fastwam_run_global_allreduce_preflight \
    "${nproc_per_node}" \
    "${num_machines}" \
    "${machine_rank}" \
    "${master_addr}" \
    "${master_port}" \
    "${run_id}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  positional_nproc="${1:-}"
  pai_nproc="${NPROC_PER_NODE:-}"
  if [[ -n "${positional_nproc}" && -n "${pai_nproc}" ]] && \
    [[ "${positional_nproc}" != "${pai_nproc}" ]]; then
    echo "Error: PAI NPROC_PER_NODE (${pai_nproc}) conflicts with positional nproc_per_node (${positional_nproc})." >&2
    exit 1
  fi
  nproc_per_node="${pai_nproc:-${positional_nproc:-}}"
  if [[ -z "${nproc_per_node}" ]]; then
    echo "Usage: NPROC_PER_NODE=<gpus> bash scripts/dlc_preflight.sh [nproc_per_node]" >&2
    exit 1
  fi
  if [[ -n "${WORLD_SIZE:-}" && -n "${NNODES:-}" && "${WORLD_SIZE}" != "${NNODES}" ]]; then
    echo "Error: PAI WORLD_SIZE (${WORLD_SIZE}) conflicts with NNODES (${NNODES})." >&2
    exit 1
  fi
  if [[ -n "${RANK:-}" && -n "${NODE_RANK:-}" && "${RANK}" != "${NODE_RANK}" ]]; then
    echo "Error: PAI RANK (${RANK}) conflicts with NODE_RANK (${NODE_RANK})." >&2
    exit 1
  fi
  num_machines="${WORLD_SIZE:-${NNODES:-1}}"
  machine_rank="${RANK:-${NODE_RANK:-0}}"
  master_addr="${MASTER_ADDR:-127.0.0.1}"
  master_port="${MASTER_PORT:-29500}"
  run_id="${RUN_ID:-preflight}"
  fastwam_run_dlc_preflight \
    "${nproc_per_node}" \
    "${num_machines}" \
    "${machine_rank}" \
    "${master_addr}" \
    "${master_port}" \
    "${run_id}"
fi
