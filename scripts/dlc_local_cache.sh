#!/usr/bin/env bash

# Materialize a sha256sum-style manifest from shared source storage into one atomic, immutable
# node-local directory. The launcher sources this file and calls
# fastwam_prepare_local_cache; direct execution prints the resulting directory.
#
# Required environment:
#   FASTWAM_LOCAL_CACHE_SOURCE_ROOT=/oss-chengjuntao/fastwam-training-bundle-v1
#   FASTWAM_LOCAL_CACHE_MANIFEST=/oss-chengjuntao/.../whole-file-cache.sha256
#
# Optional environment:
#   FASTWAM_LOCAL_CACHE_ROOT=/tmp/fastwam-whole-file-cache
#   FASTWAM_NODE_LOCAL_RANK=0
#   FASTWAM_LOCAL_CACHE_WAIT_TIMEOUT=7200
#   FASTWAM_LOCAL_CACHE_MIN_FREE_BYTES=1073741824
#   FASTWAM_LOCAL_CACHE_VERIFY_HIT=1
#   FASTWAM_LOCAL_CACHE_REQUIRE_VERIFY_HIT=0
#   FASTWAM_LOCAL_CACHE_STALE_LOCK_SECONDS=7200
#   FASTWAM_LOCAL_CACHE_EXPECTED_MANIFEST_SHA256=<64 hex>
#   FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT=relative/compact-cache
#   FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH=relative/checkpoint.pt
#   FASTWAM_LOCAL_DATASET_RELATIVE_ROOT=relative/dataset
#   FASTWAM_LOCAL_STATS_RELATIVE_PATH=relative/dataset/stats.json
#   FASTWAM_LOCAL_TEXT_EMBEDS_RELATIVE_ROOT=relative/dataset/text-embeds
#   FASTWAM_LOCAL_MODEL_CACHE_RELATIVE_ROOT=relative/model-cache
#   FASTWAM_LOCAL_VAE_RELATIVE_PATH=relative/model-cache/.../Wan2.2_VAE.safetensors
#   FASTWAM_LOCAL_ERDMA_RELATIVE_ROOT=relative/erdma-runtime
#
# Manifest lines are: <64-char sha256><whitespace><relative/path>. Absolute
# paths, interior dot components, symlinks, duplicate paths, and source-root
# escapes are rejected; a conventional leading './' is normalized. READY is
# published only after every copied whole file matches SHA.

_fastwam_cache_log() {
  printf '[local_cache] %s\n' "$*" >&2
}

_fastwam_cache_bool() {
  case "${1,,}" in
    1 | true | yes | on) return 0 ;;
    0 | false | no | off) return 1 ;;
    *)
      echo "Error: expected a boolean value, got '${1}'." >&2
      return 2
      ;;
  esac
}

_fastwam_cache_validate_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || ((10#${value} < 1)); then
    echo "Error: ${name} (${value}) must be a positive integer." >&2
    return 1
  fi
}

_fastwam_cache_validate_manifest() {
  local source_root="$1"
  local manifest="$2"
  local line
  local line_number=0
  local expected_sha
  local relative_path
  local source_path
  local real_source_path
  local file_size
  local existing_path

  FASTWAM_CACHE_RELATIVE_PATHS=()
  FASTWAM_CACHE_SOURCE_PATHS=()
  FASTWAM_CACHE_EXPECTED_SHAS=()
  FASTWAM_CACHE_TOTAL_BYTES=0

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line_number=$((line_number + 1))
    if [[ -z "${line}" || "${line}" == \#* ]]; then
      continue
    fi
    if [[ ! "${line}" =~ ^([0-9a-fA-F]{64})[[:space:]]+(.+)$ ]]; then
      echo "Error: invalid cache manifest line ${line_number}; expected '<sha256>  <relative/path>'." >&2
      return 1
    fi

    expected_sha="${BASH_REMATCH[1],,}"
    relative_path="${BASH_REMATCH[2]}"
    while [[ "${relative_path}" == ./* ]]; do
      relative_path="${relative_path#./}"
    done
    if [[ \
      -z "${relative_path}" || \
      "${relative_path}" == /* || \
      "${relative_path}" == "." || \
      "${relative_path}" == ".." || \
      "${relative_path}" == ../* || \
      "${relative_path}" == */./* || \
      "${relative_path}" == */. || \
      "${relative_path}" == */../* || \
      "${relative_path}" == */.. || \
      "${relative_path}" == *//* \
    ]]; then
      echo "Error: unsafe relative path '${relative_path}' on manifest line ${line_number}." >&2
      return 1
    fi
    for existing_path in "${FASTWAM_CACHE_RELATIVE_PATHS[@]}"; do
      if [[ "${existing_path}" == "${relative_path}" ]]; then
        echo "Error: duplicate relative path '${relative_path}' in cache manifest." >&2
        return 1
      fi
    done

    source_path="${source_root}/${relative_path}"
    if [[ ! -f "${source_path}" || -L "${source_path}" ]]; then
      echo "Error: cache source must be a regular non-symlink file: ${source_path}" >&2
      return 1
    fi
    real_source_path="$(realpath -e -- "${source_path}")"
    case "${real_source_path}" in
      "${source_root}"/*) ;;
      *)
        echo "Error: cache source escapes FASTWAM_LOCAL_CACHE_SOURCE_ROOT: ${source_path}" >&2
        return 1
        ;;
    esac

    file_size="$(stat -c '%s' -- "${real_source_path}")"
    FASTWAM_CACHE_RELATIVE_PATHS+=("${relative_path}")
    FASTWAM_CACHE_SOURCE_PATHS+=("${real_source_path}")
    FASTWAM_CACHE_EXPECTED_SHAS+=("${expected_sha}")
    FASTWAM_CACHE_TOTAL_BYTES=$((FASTWAM_CACHE_TOTAL_BYTES + file_size))
  done < "${manifest}"

  if ((${#FASTWAM_CACHE_RELATIVE_PATHS[@]} == 0)); then
    echo "Error: cache manifest has no file entries: ${manifest}" >&2
    return 1
  fi
}

_fastwam_cache_ready_is_valid() {
  local destination="$1"
  local manifest_sha="$2"
  local ready_file="${destination}/.FASTWAM_READY"
  [[ -f "${ready_file}" ]] && grep -Fqx "manifest_sha256=${manifest_sha}" "${ready_file}"
}

_fastwam_cache_verify_destination() {
  local destination="$1"
  local index
  local destination_path
  local actual_sha

  for ((index = 0; index < ${#FASTWAM_CACHE_RELATIVE_PATHS[@]}; index++)); do
    destination_path="${destination}/${FASTWAM_CACHE_RELATIVE_PATHS[index]}"
    if [[ ! -f "${destination_path}" || -L "${destination_path}" ]]; then
      echo "Error: cached file is missing or is a symlink: ${destination_path}" >&2
      return 1
    fi
    actual_sha="$(sha256sum -- "${destination_path}")"
    actual_sha="${actual_sha%% *}"
    if [[ "${actual_sha}" != "${FASTWAM_CACHE_EXPECTED_SHAS[index]}" ]]; then
      echo "Error: cached SHA-256 mismatch for ${destination_path}." >&2
      return 1
    fi
  done
}

_fastwam_cache_normalize_mapping() {
  local value="$1"
  while [[ "${value}" == ./* ]]; do
    value="${value#./}"
  done
  if [[ \
    -z "${value}" || \
    "${value}" == /* || \
    "${value}" == "." || \
    "${value}" == ".." || \
    "${value}" == ../* || \
    "${value}" == */./* || \
    "${value}" == */. || \
    "${value}" == */../* || \
    "${value}" == */.. || \
    "${value}" == *//* \
  ]]; then
    echo "Error: unsafe node-local mapping relative path '${value}'." >&2
    return 1
  fi
  printf '%s\n' "${value}"
}

_fastwam_cache_export_mapping() {
  local destination="$1"
  local source_variable="$2"
  local target_variable="$3"
  local expected_kind="$4"
  local raw_value="${!source_variable:-}"
  local relative_path
  local target_path
  local real_target
  local manifest_path
  local matched=0
  local index

  [[ -n "${raw_value}" ]] || return 0
  relative_path="$(_fastwam_cache_normalize_mapping "${raw_value}")" || return 1
  target_path="${destination}/${relative_path}"
  if [[ -L "${target_path}" ]]; then
    echo "Error: node-local mapping target must not be a symlink: ${target_path}" >&2
    return 1
  fi
  case "${expected_kind}" in
    file)
      [[ -f "${target_path}" ]] || {
        echo "Error: node-local mapping target is not a file: ${target_path}" >&2
        return 1
      }
      ;;
    directory)
      [[ -d "${target_path}" ]] || {
        echo "Error: node-local mapping target is not a directory: ${target_path}" >&2
        return 1
      }
      ;;
    *)
      echo "Error: internal mapping kind is invalid: ${expected_kind}" >&2
      return 1
      ;;
  esac
  real_target="$(realpath -e -- "${target_path}")" || return 1
  case "${real_target}" in
    "${destination}"/*) ;;
    *)
      echo "Error: node-local mapping escapes immutable bundle: ${target_path}" >&2
      return 1
      ;;
  esac

  for ((index = 0; index < ${#FASTWAM_CACHE_RELATIVE_PATHS[@]}; index++)); do
    manifest_path="${FASTWAM_CACHE_RELATIVE_PATHS[index]}"
    if [[ "${expected_kind}" == "file" && "${manifest_path}" == "${relative_path}" ]]; then
      matched=1
      FASTWAM_CACHE_MAPPING_SHA256="${FASTWAM_CACHE_EXPECTED_SHAS[index]}"
      break
    fi
    if [[ "${expected_kind}" == "directory" && "${manifest_path}" == "${relative_path}/"* ]]; then
      matched=1
    fi
  done
  if ((!matched)); then
    echo "Error: node-local mapping is not covered by the verified manifest: ${relative_path}" >&2
    return 1
  fi
  printf -v "${target_variable}" '%s' "${real_target}"
  export "${target_variable}"
  _fastwam_cache_log "mapping=${target_variable} path=${relative_path} target=${real_target}"
}

_fastwam_cache_export_ready() {
  local destination="$1"
  local manifest_sha="$2"
  export FASTWAM_LOCAL_CACHE_DIR="${destination}"
  export FASTWAM_LOCAL_CACHE_MANIFEST_SHA256="${manifest_sha}"
  _fastwam_cache_export_mapping \
    "${destination}" FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT FASTWAM_GAUSSIAN_CACHE_DIR directory || return 1
  _fastwam_cache_export_mapping \
    "${destination}" FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH FASTWAM_LOCAL_CHECKPOINT_PATH file || return 1
  if [[ -n "${FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH:-}" ]]; then
    export FASTWAM_LOCAL_CHECKPOINT_MANIFEST_SHA256="${FASTWAM_CACHE_MAPPING_SHA256}"
  fi
  _fastwam_cache_export_mapping \
    "${destination}" FASTWAM_LOCAL_DATASET_RELATIVE_ROOT FASTWAM_LOCAL_DATASET_ROOT directory || return 1
  _fastwam_cache_export_mapping \
    "${destination}" FASTWAM_LOCAL_STATS_RELATIVE_PATH FASTWAM_LOCAL_STATS_SOURCE_PATH file || return 1
  if [[ -n "${FASTWAM_LOCAL_STATS_RELATIVE_PATH:-}" ]]; then
    export FASTWAM_LOCAL_STATS_MANIFEST_SHA256="${FASTWAM_CACHE_MAPPING_SHA256}"
  fi
  _fastwam_cache_export_mapping \
    "${destination}" FASTWAM_LOCAL_TEXT_EMBEDS_RELATIVE_ROOT FASTWAM_LOCAL_TEXT_EMBEDS_ROOT directory || return 1
  _fastwam_cache_export_mapping \
    "${destination}" FASTWAM_LOCAL_MODEL_CACHE_RELATIVE_ROOT DIFFSYNTH_MODEL_BASE_PATH directory || return 1
  _fastwam_cache_export_mapping \
    "${destination}" FASTWAM_LOCAL_VAE_RELATIVE_PATH FASTWAM_LOCAL_VAE_PATH file || return 1
  if [[ -n "${FASTWAM_LOCAL_VAE_RELATIVE_PATH:-}" ]]; then
    export FASTWAM_LOCAL_VAE_MANIFEST_SHA256="${FASTWAM_CACHE_MAPPING_SHA256}"
  fi
  _fastwam_cache_export_mapping \
    "${destination}" FASTWAM_LOCAL_ERDMA_RELATIVE_ROOT FASTWAM_LOCAL_ERDMA_ROOT directory || return 1
}

_fastwam_cache_build() {
  local staging="$1"
  local ready_file="${staging}/.FASTWAM_READY"
  local manifest_sha="$2"
  local source_root="$3"
  local index
  local relative_path
  local source_path
  local destination_path
  local actual_sha

  mkdir -p -- "${staging}" || return 1
  for ((index = 0; index < ${#FASTWAM_CACHE_RELATIVE_PATHS[@]}; index++)); do
    relative_path="${FASTWAM_CACHE_RELATIVE_PATHS[index]}"
    source_path="${FASTWAM_CACHE_SOURCE_PATHS[index]}"
    destination_path="${staging}/${relative_path}"
    mkdir -p -- "$(dirname -- "${destination_path}")" || return 1
    _fastwam_cache_log "copy file=$((index + 1))/${#FASTWAM_CACHE_RELATIVE_PATHS[@]} path=${relative_path}"
    cp --reflink=auto --sparse=always --preserve=mode,timestamps -- \
      "${source_path}" "${destination_path}" || return 1
    actual_sha="$(sha256sum -- "${destination_path}")" || return 1
    actual_sha="${actual_sha%% *}"
    if [[ "${actual_sha}" != "${FASTWAM_CACHE_EXPECTED_SHAS[index]}" ]]; then
      echo "Error: copied SHA-256 mismatch for ${relative_path}." >&2
      return 1
    fi
  done

  {
    printf 'schema_version=1\n'
    printf 'manifest_sha256=%s\n' "${manifest_sha}"
    printf 'source_root=%s\n' "${source_root}"
    printf 'file_count=%s\n' "${#FASTWAM_CACHE_RELATIVE_PATHS[@]}"
    printf 'total_bytes=%s\n' "${FASTWAM_CACHE_TOTAL_BYTES}"
  } > "${ready_file}" || return 1
}

_fastwam_cache_wait_for_ready() {
  local destination="$1"
  local manifest_sha="$2"
  local lock_directory="$3"
  local failure_file="$4"
  local timeout_s="$5"
  local deadline=$((SECONDS + timeout_s))

  _fastwam_cache_log "action=wait destination=${destination} timeout_s=${timeout_s}"
  while ((SECONDS < deadline)); do
    if _fastwam_cache_ready_is_valid "${destination}" "${manifest_sha}"; then
      return 0
    fi
    if [[ -f "${failure_file}" && ! -d "${lock_directory}" ]]; then
      # Give a simultaneously starting rank zero one polling interval to
      # remove a stale failure marker and acquire the node-local lock.
      sleep 1
      if [[ -f "${failure_file}" && ! -d "${lock_directory}" ]]; then
        echo "Error: node-local cache builder failed; marker: ${failure_file}" >&2
        return 1
      fi
    fi
    sleep 1
  done
  echo "Error: timed out waiting for node-local cache READY at ${destination}." >&2
  return 1
}

_fastwam_cache_reap_stale_lock() {
  local lock_directory="$1"
  local stale_seconds="$2"
  local owner_file="${lock_directory}/owner"
  local owner_pid=""
  local owner_host=""
  local current_host
  local now_epoch
  local lock_epoch
  local lock_age
  local line
  local stale_reason=""

  [[ -e "${lock_directory}" || -L "${lock_directory}" ]] || return 1
  current_host="$(hostname)" || return 2
  if [[ -f "${owner_file}" && ! -L "${owner_file}" ]]; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
      case "${line}" in
        pid=*) owner_pid="${line#pid=}" ;;
        hostname=*) owner_host="${line#hostname=}" ;;
      esac
    done < "${owner_file}"
  fi

  if [[ "${owner_host}" == "${current_host}" && "${owner_pid}" =~ ^[0-9]+$ ]]; then
    if kill -0 "${owner_pid}" 2>/dev/null; then
      _fastwam_cache_log "action=keep_lock reason=live_local_owner pid=${owner_pid}"
      return 1
    fi
    stale_reason="dead_local_owner"
  else
    now_epoch="$(date +%s)" || return 2
    lock_epoch="$(stat -c '%Y' -- "${lock_directory}")" || return 2
    lock_age=$((10#${now_epoch} - 10#${lock_epoch}))
    if ((lock_age >= 10#${stale_seconds})); then
      stale_reason="age_${lock_age}s"
    else
      _fastwam_cache_log "action=keep_lock reason=owner_unconfirmed age_s=${lock_age}"
      return 1
    fi
  fi

  rm -rf -- "${lock_directory}" || return 2
  _fastwam_cache_log "action=reap_stale_lock reason=${stale_reason} lock=${lock_directory}"
  return 0
}

_fastwam_cache_build_owner() (
  local destination="$1"
  local manifest_sha="$2"
  local source_root="$3"
  local cache_root="$4"
  local lock_directory="$5"
  local failure_file="$6"
  local reserve_bytes="$7"
  local staging="${cache_root}/.${manifest_sha}.STAGING.$$"
  local available_bytes
  local required_bytes
  local build_status
  local owner_host

  _fastwam_cache_owner_cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if ((status != 0)) && [[ ! -f "${failure_file}" ]]; then
      printf 'reason=builder_interrupted_or_failed status=%s\n' "${status}" > "${failure_file}" || true
    fi
    if [[ -n "${staging}" && -e "${staging}" ]]; then
      rm -rf -- "${staging}" || true
    fi
    rm -rf -- "${lock_directory}" || true
    exit "${status}"
  }
  trap _fastwam_cache_owner_cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  owner_host="$(hostname)" || exit 1
  {
    printf 'pid=%s\n' "${BASHPID}"
    printf 'hostname=%s\n' "${owner_host}"
    printf 'started_epoch=%s\n' "$(date +%s)"
  } > "${lock_directory}/owner" || exit 1
  rm -f -- "${failure_file}" || exit 1

  if available_bytes="$(df -B1 --output=avail -- "${cache_root}" | tail -n 1 | tr -d '[:space:]')"; then
    :
  else
    printf 'reason=free_space_probe_failed\n' > "${failure_file}" || true
    exit 1
  fi
  required_bytes=$((FASTWAM_CACHE_TOTAL_BYTES + 10#${reserve_bytes}))
  if [[ ! "${available_bytes}" =~ ^[0-9]+$ ]] || ((10#${available_bytes} < required_bytes)); then
    printf 'reason=insufficient_space required_bytes=%s available_bytes=%s\n' \
      "${required_bytes}" "${available_bytes}" > "${failure_file}" || true
    echo "Error: insufficient node-local space; required=${required_bytes}, available=${available_bytes}." >&2
    exit 1
  fi

  if _fastwam_cache_build "${staging}" "${manifest_sha}" "${source_root}"; then
    :
  else
    build_status=$?
    printf 'reason=copy_or_sha_failed status=%s\n' "${build_status}" > "${failure_file}" || true
    exit "${build_status}"
  fi
  if mv -- "${staging}" "${destination}"; then
    staging=""
  else
    build_status=$?
    printf 'reason=atomic_publish_failed status=%s\n' "${build_status}" > "${failure_file}" || true
    exit "${build_status}"
  fi
  _fastwam_cache_log "status=READY action=build directory=${destination} files=${#FASTWAM_CACHE_RELATIVE_PATHS[@]} bytes=${FASTWAM_CACHE_TOTAL_BYTES} manifest_sha256=${manifest_sha}"
)

fastwam_prepare_local_cache() {
  local source_root="${FASTWAM_LOCAL_CACHE_SOURCE_ROOT:?FASTWAM_LOCAL_CACHE_SOURCE_ROOT is required}"
  local manifest="${FASTWAM_LOCAL_CACHE_MANIFEST:?FASTWAM_LOCAL_CACHE_MANIFEST is required}"
  local cache_root="${FASTWAM_LOCAL_CACHE_ROOT:-/tmp/fastwam-whole-file-cache}"
  local local_rank="${FASTWAM_NODE_LOCAL_RANK:-${LOCAL_RANK:-0}}"
  local timeout_s="${FASTWAM_LOCAL_CACHE_WAIT_TIMEOUT:-7200}"
  local reserve_bytes="${FASTWAM_LOCAL_CACHE_MIN_FREE_BYTES:-1073741824}"
  local verify_hit="${FASTWAM_LOCAL_CACHE_VERIFY_HIT:-1}"
  local require_verify_hit="${FASTWAM_LOCAL_CACHE_REQUIRE_VERIFY_HIT:-0}"
  local stale_lock_seconds="${FASTWAM_LOCAL_CACHE_STALE_LOCK_SECONDS:-7200}"
  local allow_shared_fs="${FASTWAM_LOCAL_CACHE_ALLOW_SHARED_FS:-0}"
  local expected_manifest_sha="${FASTWAM_LOCAL_CACHE_EXPECTED_MANIFEST_SHA256:-}"
  local manifest_sha
  local destination
  local lock_directory
  local failure_file
  local source_device
  local cache_device
  local build_status
  local verify_status
  local verify_hit_enabled
  local require_verify_hit_enabled
  local lock_acquired=0

  if [[ -z "${cache_root}" || "${cache_root}" == "/" ]]; then
    echo "Error: FASTWAM_LOCAL_CACHE_ROOT must be a specific directory." >&2
    return 1
  fi
  _fastwam_cache_validate_positive_integer FASTWAM_LOCAL_CACHE_WAIT_TIMEOUT "${timeout_s}" || return 1
  if [[ ! "${reserve_bytes}" =~ ^[0-9]+$ ]]; then
    echo "Error: FASTWAM_LOCAL_CACHE_MIN_FREE_BYTES (${reserve_bytes}) must be a non-negative integer." >&2
    return 1
  fi
  if [[ ! "${stale_lock_seconds}" =~ ^[0-9]+$ ]]; then
    echo "Error: FASTWAM_LOCAL_CACHE_STALE_LOCK_SECONDS (${stale_lock_seconds}) must be a non-negative integer." >&2
    return 1
  fi
  if [[ ! "${local_rank}" =~ ^[0-9]+$ ]]; then
    echo "Error: FASTWAM_NODE_LOCAL_RANK (${local_rank}) must be a non-negative integer." >&2
    return 1
  fi
  if _fastwam_cache_bool "${verify_hit}"; then
    verify_hit_enabled=1
  else
    verify_status=$?
    if ((verify_status == 2)); then
      return 1
    fi
    verify_hit_enabled=0
  fi
  if _fastwam_cache_bool "${require_verify_hit}"; then
    require_verify_hit_enabled=1
  else
    verify_status=$?
    if ((verify_status == 2)); then
      return 1
    fi
    require_verify_hit_enabled=0
  fi
  if ((require_verify_hit_enabled && !verify_hit_enabled)); then
    echo "Error: FASTWAM_LOCAL_CACHE_REQUIRE_VERIFY_HIT=1 forbids disabling FASTWAM_LOCAL_CACHE_VERIFY_HIT." >&2
    return 1
  fi

  source_root="$(realpath -e -- "${source_root}")" || return 1
  if [[ ! -f "${manifest}" || -L "${manifest}" ]]; then
    echo "Error: FASTWAM_LOCAL_CACHE_MANIFEST must be a regular non-symlink file." >&2
    return 1
  fi
  # Test the caller-supplied path before canonicalizing it; otherwise realpath
  # would erase the evidence that the manifest itself was a symlink.
  manifest="$(realpath -e -- "${manifest}")" || return 1
  mkdir -p -- "${cache_root}" || return 1
  cache_root="$(realpath -e -- "${cache_root}")" || return 1

  source_device="$(stat -c '%d' -- "${source_root}")" || return 1
  cache_device="$(stat -c '%d' -- "${cache_root}")" || return 1
  if [[ "${source_device}" == "${cache_device}" ]]; then
    if _fastwam_cache_bool "${allow_shared_fs}"; then
      _fastwam_cache_log "warning=source_and_cache_share_filesystem override=enabled"
    else
      verify_status=$?
      if ((verify_status == 2)); then
        return 1
      fi
      echo "Error: cache root and source root share a filesystem; choose node-local storage or set FASTWAM_LOCAL_CACHE_ALLOW_SHARED_FS=1 for an intentional test." >&2
      return 1
    fi
  else
    if _fastwam_cache_bool "${allow_shared_fs}"; then
      :
    else
      verify_status=$?
      if ((verify_status == 2)); then
        return 1
      fi
    fi
  fi

  _fastwam_cache_validate_manifest "${source_root}" "${manifest}" || return 1
  manifest_sha="$(sha256sum -- "${manifest}")" || return 1
  manifest_sha="${manifest_sha%% *}"
  if [[ -n "${expected_manifest_sha}" ]]; then
    expected_manifest_sha="${expected_manifest_sha,,}"
    if [[ ! "${expected_manifest_sha}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "Error: FASTWAM_LOCAL_CACHE_EXPECTED_MANIFEST_SHA256 must be 64 lowercase hex characters." >&2
      return 1
    fi
    if [[ "${manifest_sha}" != "${expected_manifest_sha}" ]]; then
      echo "Error: node-local bundle manifest SHA-256 mismatch: expected=${expected_manifest_sha} actual=${manifest_sha}." >&2
      return 1
    fi
  fi
  destination="${cache_root}/${manifest_sha}"
  lock_directory="${cache_root}/.${manifest_sha}.LOCK"
  failure_file="${cache_root}/.${manifest_sha}.FAILED"

  if _fastwam_cache_ready_is_valid "${destination}" "${manifest_sha}"; then
    if ((verify_hit_enabled)); then
      _fastwam_cache_verify_destination "${destination}" || return 1
    fi
    _fastwam_cache_export_ready "${destination}" "${manifest_sha}" || return 1
    _fastwam_cache_log "status=READY action=hit directory=${destination} manifest_sha256=${manifest_sha}"
    return 0
  fi
  if [[ -e "${destination}" ]]; then
    echo "Error: cache destination exists without a valid READY marker: ${destination}" >&2
    return 1
  fi

  if ((10#${local_rank} != 0)); then
    _fastwam_cache_wait_for_ready \
      "${destination}" "${manifest_sha}" "${lock_directory}" "${failure_file}" "${timeout_s}" || return 1
  else
    if mkdir -- "${lock_directory}" 2>/dev/null; then
      lock_acquired=1
    elif _fastwam_cache_reap_stale_lock "${lock_directory}" "${stale_lock_seconds}"; then
      if mkdir -- "${lock_directory}" 2>/dev/null; then
        lock_acquired=1
      fi
    else
      verify_status=$?
      if ((verify_status == 2)); then
        return 1
      fi
    fi

    if ((lock_acquired)); then
      if _fastwam_cache_build_owner \
        "${destination}" \
        "${manifest_sha}" \
        "${source_root}" \
        "${cache_root}" \
        "${lock_directory}" \
        "${failure_file}" \
        "${reserve_bytes}"; then
        :
      else
        build_status=$?
        return "${build_status}"
      fi
    else
      _fastwam_cache_wait_for_ready \
        "${destination}" "${manifest_sha}" "${lock_directory}" "${failure_file}" "${timeout_s}" || return 1
    fi
  fi

  if ! _fastwam_cache_ready_is_valid "${destination}" "${manifest_sha}"; then
    echo "Error: node-local cache publish completed without a valid READY marker." >&2
    return 1
  fi
  _fastwam_cache_export_ready "${destination}" "${manifest_sha}" || return 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  fastwam_prepare_local_cache
  printf '%s\n' "${FASTWAM_LOCAL_CACHE_DIR}"
fi
