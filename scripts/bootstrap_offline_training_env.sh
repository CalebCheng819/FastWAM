#!/usr/bin/env bash

# Build a verified, content-addressed FastWAM checkout and Python environment
# from an immutable whole-file bundle. Source this file, call
# fastwam_prepare_offline_training_env, and then launch from the exported clean
# checkout in the *same outer shell*:
#
#   source /verified/source-snapshot/scripts/bootstrap_offline_training_env.sh
#   fastwam_prepare_offline_training_env
#   "$FASTWAM_REPO_ROOT/scripts/train_zero2.sh" 8 ...
#
# Direct execution is diagnostic only: exports cannot modify its parent shell.

_fastwam_offline_env_log() {
  printf '[offline_env] %s\n' "$*" >&2
}

_fastwam_offline_env_normalize_relative_path() {
  local name="$1"
  local value="$2"

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
    "${value}" == *//* || \
    "${value}" == *[$'\001'-$'\037'$'\177']* \
  ]]; then
    echo "Error: ${name} contains an unsafe relative path: '${value}'." >&2
    return 1
  fi
  printf '%s\n' "${value}"
}

_fastwam_offline_env_validate_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || ((10#${value} < 1)); then
    echo "Error: ${name} (${value}) must be a positive integer." >&2
    return 1
  fi
}

_fastwam_offline_env_validate_nonnegative_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "Error: ${name} (${value}) must be a non-negative integer." >&2
    return 1
  fi
}

_fastwam_offline_env_validate_sha256() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Error: ${name} must be exactly 64 lowercase hexadecimal characters." >&2
    return 1
  fi
}

_fastwam_offline_env_validate_commit() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Error: ${name} must be an exact 40-character lowercase Git commit." >&2
    return 1
  fi
}

_fastwam_offline_env_prepare_tmp_root() {
  local name="$1"
  local value="$2"
  local relative
  local component
  local probe
  local canonical
  local prospective
  local -a components=()

  if [[ \
    "${value}" != /tmp/* || \
    "${value}" == /tmp/ || \
    "${value}" == */ || \
    "${value}" == *//* || \
    "${value}" == */./* || \
    "${value}" == */. || \
    "${value}" == */../* || \
    "${value}" == */.. || \
    "${value}" == *[$'\001'-$'\037'$'\177']* \
  ]]; then
    echo "Error: ${name} must be a canonical lexical path below /tmp (got '${value}')." >&2
    return 1
  fi
  if [[ ! -d /tmp || -L /tmp ]] || [[ "$(realpath -e -- /tmp)" != /tmp ]]; then
    echo "Error: /tmp must be a real, non-symlink directory." >&2
    return 1
  fi
  prospective="$(realpath -m -- "${value}")"
  case "${prospective}" in
    /tmp/*) ;;
    *)
      echo "Error: ${name} prospective canonical path escapes /tmp: ${prospective}" >&2
      return 1
      ;;
  esac

  relative="${value#/tmp/}"
  IFS='/' read -r -a components <<<"${relative}"
  probe=/tmp
  for component in "${components[@]}"; do
    if [[ -z "${component}" || "${component}" == "." || "${component}" == ".." ]]; then
      echo "Error: ${name} contains an unsafe path component." >&2
      return 1
    fi
    probe="${probe}/${component}"
    if [[ -L "${probe}" ]]; then
      echo "Error: ${name} rejects symlink root or parent: ${probe}" >&2
      return 1
    fi
    if [[ -e "${probe}" && ! -d "${probe}" ]]; then
      echo "Error: ${name} component is not a directory: ${probe}" >&2
      return 1
    fi
  done

  mkdir -p -- "${value}"
  probe=/tmp
  for component in "${components[@]}"; do
    probe="${probe}/${component}"
    if [[ -L "${probe}" || ! -d "${probe}" ]]; then
      echo "Error: ${name} became a symlink or non-directory: ${probe}" >&2
      return 1
    fi
  done
  canonical="$(realpath -e -- "${value}")"
  if [[ "${canonical}" != "${value}" ]]; then
    echo "Error: ${name} lexical/canonical mismatch: '${value}' -> '${canonical}'." >&2
    return 1
  fi
  printf '%s\n' "${canonical}"
}

_fastwam_offline_env_bind_python_identity() {
  local configured="${FASTWAM_OFFLINE_ENV_BASE_PYTHON:-python3.10}"
  local executable
  local identity_output
  local reported_executable

  executable="$(type -P -- "${configured}" 2>/dev/null || true)"
  if [[ -z "${executable}" || ! -x "${executable}" ]]; then
    echo "Error: configured formal base interpreter '${configured}' is not executable." >&2
    return 1
  fi
  executable="$(realpath -e -- "${executable}")"
  identity_output="$(env -u PYTHONHOME -u PYTHONPATH \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "${executable}" -I -c 'import os,platform,sys,sysconfig; print("\t".join((platform.python_version(), sys.implementation.name, sysconfig.get_config_var("SOABI") or "", sys.implementation.cache_tag or "", sysconfig.get_platform(), os.path.realpath(sys.executable))))')" || {
    echo "Error: failed to inspect fixed python3.10 interpreter ${executable}." >&2
    return 1
  }
  IFS=$'\t' read -r \
    FASTWAM_OFFLINE_ENV_PYTHON_VERSION \
    FASTWAM_OFFLINE_ENV_PYTHON_IMPLEMENTATION \
    FASTWAM_OFFLINE_ENV_PYTHON_ABI \
    FASTWAM_OFFLINE_ENV_PYTHON_CACHE_TAG \
    FASTWAM_OFFLINE_ENV_PYTHON_PLATFORM \
    reported_executable <<<"${identity_output}"

  if [[ ! "${FASTWAM_OFFLINE_ENV_PYTHON_VERSION}" =~ ^3\.10\.[0-9]+$ ]]; then
    echo "Error: python3.10 resolved to incompatible version '${FASTWAM_OFFLINE_ENV_PYTHON_VERSION}'." >&2
    return 1
  fi
  if [[ "${FASTWAM_OFFLINE_ENV_PYTHON_IMPLEMENTATION}" != cpython ]]; then
    echo "Error: formal Python implementation must be CPython, got '${FASTWAM_OFFLINE_ENV_PYTHON_IMPLEMENTATION}'." >&2
    return 1
  fi
  if [[ "${FASTWAM_OFFLINE_ENV_PYTHON_ABI}" != cpython-310* || "${FASTWAM_OFFLINE_ENV_PYTHON_CACHE_TAG}" != cpython-310 || -z "${FASTWAM_OFFLINE_ENV_PYTHON_PLATFORM}" ]]; then
    echo "Error: python3.10 has incompatible ABI/cache tag/platform '${FASTWAM_OFFLINE_ENV_PYTHON_ABI}'/'${FASTWAM_OFFLINE_ENV_PYTHON_CACHE_TAG}'/'${FASTWAM_OFFLINE_ENV_PYTHON_PLATFORM}'." >&2
    return 1
  fi
  if [[ "${reported_executable}" != "${executable}" ]]; then
    echo "Error: python3.10 executable identity changed: '${executable}' vs '${reported_executable}'." >&2
    return 1
  fi

  FASTWAM_OFFLINE_ENV_BASE_PYTHON_EXECUTABLE="${executable}"
  FASTWAM_OFFLINE_ENV_PYTHON_EXECUTABLE_SHA256="$(sha256sum -- "${executable}" | awk '{print $1}')"
  FASTWAM_OFFLINE_ENV_PYTHON_IDENTITY_SHA256="$({
    printf 'version=%s\n' "${FASTWAM_OFFLINE_ENV_PYTHON_VERSION}"
    printf 'implementation=%s\n' "${FASTWAM_OFFLINE_ENV_PYTHON_IMPLEMENTATION}"
    printf 'abi=%s\n' "${FASTWAM_OFFLINE_ENV_PYTHON_ABI}"
    printf 'cache_tag=%s\n' "${FASTWAM_OFFLINE_ENV_PYTHON_CACHE_TAG}"
    printf 'platform=%s\n' "${FASTWAM_OFFLINE_ENV_PYTHON_PLATFORM}"
    printf 'executable=%s\n' "${FASTWAM_OFFLINE_ENV_BASE_PYTHON_EXECUTABLE}"
    printf 'executable_sha256=%s\n' "${FASTWAM_OFFLINE_ENV_PYTHON_EXECUTABLE_SHA256}"
  } | sha256sum | awk '{print $1}')"
}

_fastwam_offline_env_python_runtime_matches() {
  local python="$1"
  local expected_version="$2"
  local expected_implementation="$3"
  local expected_abi="$4"
  local expected_cache_tag="$5"
  local expected_platform="$6"
  local expected_executable_sha="$7"
  local actual_output
  local actual_version
  local actual_implementation
  local actual_abi
  local actual_cache_tag
  local actual_platform
  local actual_executable
  local actual_sha

  if [[ ! -x "${python}" ]]; then
    echo "Error: prepared Python is not executable: ${python}" >&2
    return 1
  fi
  actual_executable="$(realpath -e -- "${python}")"
  if [[ ! -f "${actual_executable}" ]]; then
    echo "Error: prepared Python does not resolve to a regular file: ${python}" >&2
    return 1
  fi
  actual_sha="$(sha256sum -- "${actual_executable}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${expected_executable_sha}" ]]; then
    echo "Error: prepared Python executable SHA-256 mismatch." >&2
    return 1
  fi
  actual_output="$(env -u PYTHONHOME -u PYTHONPATH \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "${python}" -I -c 'import platform,sys,sysconfig; print("\t".join((platform.python_version(), sys.implementation.name, sysconfig.get_config_var("SOABI") or "", sys.implementation.cache_tag or "", sysconfig.get_platform())))')" || return 1
  IFS=$'\t' read -r actual_version actual_implementation actual_abi actual_cache_tag actual_platform <<<"${actual_output}"
  if [[ \
    "${actual_version}" != "${expected_version}" || \
    "${actual_implementation}" != "${expected_implementation}" || \
    "${actual_abi}" != "${expected_abi}" || \
    "${actual_cache_tag}" != "${expected_cache_tag}" || \
    "${actual_platform}" != "${expected_platform}" \
  ]]; then
    echo "Error: prepared Python version/ABI identity mismatch." >&2
    return 1
  fi
}

_fastwam_offline_env_stage_payload() {
  local script_dir="$1"
  local source_root="$2"
  local manifest="$3"
  local manifest_sha="$4"
  local cache_root="$5"
  local minimum_free_bytes="$6"

  (
    set -euo pipefail
    export FASTWAM_LOCAL_CACHE_SOURCE_ROOT="${source_root}"
    export FASTWAM_LOCAL_CACHE_MANIFEST="${manifest}"
    export FASTWAM_LOCAL_CACHE_EXPECTED_MANIFEST_SHA256="${manifest_sha}"
    export FASTWAM_LOCAL_CACHE_ROOT="${cache_root}"
    export FASTWAM_LOCAL_CACHE_MIN_FREE_BYTES="${minimum_free_bytes}"
    export FASTWAM_LOCAL_CACHE_VERIFY_HIT=1
    export FASTWAM_LOCAL_CACHE_REQUIRE_VERIFY_HIT=1
    # This bootstrap is called once per pod. Its cache helper must own staging
    # even if the outer launcher inherited a nonzero worker rank.
    export FASTWAM_NODE_LOCAL_RANK=0
    unset LOCAL_RANK OMPI_COMM_WORLD_LOCAL_RANK MPI_LOCALRANKID SLURM_LOCALID
    unset \
      FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH \
      FASTWAM_LOCAL_DATASET_RELATIVE_ROOT \
      FASTWAM_LOCAL_STATS_RELATIVE_PATH \
      FASTWAM_LOCAL_TEXT_EMBEDS_RELATIVE_ROOT \
      FASTWAM_LOCAL_MODEL_CACHE_RELATIVE_ROOT \
      FASTWAM_LOCAL_VAE_RELATIVE_PATH \
      FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT \
      FASTWAM_LOCAL_ERDMA_RELATIVE_ROOT
    # shellcheck source=dlc_local_cache.sh
    source "${script_dir}/dlc_local_cache.sh" || return 1
    fastwam_prepare_local_cache >&2 || return 1
    printf '%s\n' "${FASTWAM_LOCAL_CACHE_DIR}"
  )
}

_fastwam_offline_env_validate_checkout_tree() {
  local destination="$1"
  local expected_commit="$2"
  local actual_commit
  local dirty
  local ignored
  local required

  if [[ ! -d "${destination}" || -L "${destination}" ]]; then
    return 1
  fi
  actual_commit="$(git -C "${destination}" rev-parse HEAD 2>/dev/null)" || return 1
  [[ "${actual_commit}" == "${expected_commit}" ]] || return 1
  dirty="$(git -C "${destination}" status --porcelain --untracked-files=all 2>/dev/null)" || return 1
  [[ -z "${dirty}" ]] || return 1
  ignored="$(git -C "${destination}" ls-files --others --ignored --exclude-standard 2>/dev/null)" || return 1
  [[ -z "${ignored}" ]] || return 1
  for required in \
    pyproject.toml \
    scripts/validate_python_environment.py \
    scripts/train_zero2.sh \
    src/fastwam/__init__.py; do
    if [[ ! -f "${destination}/${required}" || -L "${destination}/${required}" ]]; then
      return 1
    fi
  done
}

_fastwam_offline_env_validate_checkout() {
  local destination="$1"
  local ready_file="$2"
  local expected_marker="$3"
  local expected_commit="$4"

  _fastwam_offline_env_validate_checkout_tree "${destination}" "${expected_commit}" || return 1
  [[ -f "${ready_file}" && ! -L "${ready_file}" ]] || return 1
  [[ "$(<"${ready_file}")" == "${expected_marker}" ]] || return 1
}

_fastwam_offline_env_validate_venv() {
  local destination="$1"
  local expected_marker="$2"
  local expected_version="$3"
  local expected_implementation="$4"
  local expected_abi="$5"
  local expected_cache_tag="$6"
  local expected_platform="$7"
  local expected_executable_sha="$8"
  local marker_file="${destination}/.FASTWAM_ENV_READY"

  [[ -d "${destination}" && ! -L "${destination}" ]] || return 1
  [[ -f "${marker_file}" && ! -L "${marker_file}" ]] || return 1
  [[ "$(<"${marker_file}")" == "${expected_marker}" ]] || return 1
  _fastwam_offline_env_python_runtime_matches \
    "${destination}/bin/python" \
    "${expected_version}" \
    "${expected_implementation}" \
    "${expected_abi}" \
    "${expected_cache_tag}" \
    "${expected_platform}" \
    "${expected_executable_sha}"
}

_fastwam_offline_env_validate_import_resolution() {
  local python="$1"
  local checkout="$2"
  local expected_src="${checkout}/src"

  env -u PYTHONHOME -u PYTHONPATH \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${expected_src}" \
    FASTWAM_EXPECTED_REPO_SRC="${expected_src}" \
    "${python}" -c '
import os
from pathlib import Path
import fastwam
expected_src = Path(os.environ["FASTWAM_EXPECTED_REPO_SRC"]).resolve(strict=True)
resolved = Path(fastwam.__file__).resolve(strict=True)
expected = (expected_src / "fastwam" / "__init__.py").resolve(strict=True)
if resolved != expected:
    raise SystemExit(f"fastwam resolved outside prepared checkout: {resolved} != {expected}")
'
}

_fastwam_offline_env_validate_training_runtime() {
  local python="$1"
  local checkout="$2"

  env -u PYTHONHOME -u PYTHONPATH \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${checkout}/src" \
    "${python}" "${checkout}/scripts/validate_python_environment.py" \
      --pyproject "${checkout}/pyproject.toml" || return 1
  env -u PYTHONHOME -u PYTHONPATH \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "${python}" -I -m pip check || return 1
  _fastwam_offline_env_validate_import_resolution "${python}" "${checkout}" || return 1
  # The venv is atomically moved after creation, so formal launch uses module
  # entrypoints through this Python and never relocation-sensitive bin scripts.
  env -u PYTHONHOME -u PYTHONPATH \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "${python}" -I -m accelerate.commands.launch --help >/dev/null || return 1
}

_fastwam_offline_env_write_failure() {
  local failure_file="$1"
  local message="$2"
  local temporary="${failure_file}.tmp.$$.${RANDOM}"

  if [[ -L "${failure_file}" ]]; then
    echo "Error: refusing symlink failure marker: ${failure_file}" >&2
    return 1
  fi
  printf 'pid=%s\nhost=%s\ntime=%s\nmessage=%s\n' \
    "$$" "$(hostname)" "$(date +%s)" "${message}" >"${temporary}"
  chmod 0444 "${temporary}"
  mv -T -- "${temporary}" "${failure_file}"
}

_fastwam_offline_env_lock_value() {
  local lock_dir="$1"
  local key="$2"
  local owner_file="${lock_dir}/owner"

  [[ -f "${owner_file}" && ! -L "${owner_file}" ]] || return 1
  sed -n "s/^${key}=//p" "${owner_file}" | head -n1
}

_fastwam_offline_env_lock_snapshot() {
  local lock_dir="$1"
  local owner_file="${lock_dir}/owner"
  local lock_stat
  local owner_sha=missing

  if [[ -L "${lock_dir}" || ! -d "${lock_dir}" ]]; then
    return 1
  fi
  if [[ -L "${owner_file}" ]]; then
    return 1
  fi
  lock_stat="$(stat -c '%d:%i:%Y:%s' -- "${lock_dir}")" || return 1
  if [[ -f "${owner_file}" ]]; then
    owner_sha="$(sha256sum -- "${owner_file}" | awk '{print $1}')" || return 1
  fi
  printf '%s|%s\n' "${lock_stat}" "${owner_sha}"
}

_fastwam_offline_env_lock_is_owned() {
  local lock_dir="$1"
  local expected_token="$2"
  local actual_token

  actual_token="$(_fastwam_offline_env_lock_value "${lock_dir}" token 2>/dev/null)" || return 1
  [[ "${actual_token}" == "${expected_token}" ]]
}

_fastwam_offline_env_write_lock_owner() {
  local lock_dir="$1"
  local token="$2"
  local staging="${3-}"
  local require_existing_token="${4:-1}"
  local temporary="${lock_dir}/.owner.tmp.${BASHPID}.${RANDOM}"

  [[ -d "${lock_dir}" && ! -L "${lock_dir}" ]] || return 1
  if [[ "${require_existing_token}" == 1 ]]; then
    _fastwam_offline_env_lock_is_owned "${lock_dir}" "${token}" || return 1
  elif [[ -e "${lock_dir}/owner" || -L "${lock_dir}/owner" ]]; then
    _fastwam_offline_env_lock_is_owned "${lock_dir}" "${token}" || return 1
  fi
  printf 'token=%s\npid=%s\nhost=%s\ntime=%s\nstaging=%s\n' \
    "${token}" "${BASHPID}" "$(hostname)" "$(date +%s)" "${staging}" >"${temporary}"
  chmod 0444 "${temporary}"
  mv -T -- "${temporary}" "${lock_dir}/owner"
}

_fastwam_offline_env_release_lock() {
  local lock_dir="$1"
  local token="$2"
  local quarantine="${lock_dir}.RELEASE.${BASHPID}.${RANDOM}"

  _fastwam_offline_env_lock_is_owned "${lock_dir}" "${token}" || return 1
  if mv -T -- "${lock_dir}" "${quarantine}" 2>/dev/null; then
    rm -rf -- "${quarantine}"
    return 0
  fi
  return 1
}

_fastwam_offline_env_lock_is_stale() {
  local lock_dir="$1"
  local stale_seconds="$2"
  local owner_file="${lock_dir}/owner"
  local owner_pid=""
  local owner_host=""
  local owner_time=""
  local lock_time
  local now

  if [[ -L "${lock_dir}" || ! -d "${lock_dir}" ]]; then
    echo "Error: lock is a symlink or non-directory: ${lock_dir}" >&2
    return 2
  fi
  if [[ -L "${owner_file}" ]]; then
    echo "Error: lock owner marker is a symlink: ${owner_file}" >&2
    return 2
  fi
  lock_time="$(stat -c '%Y' -- "${lock_dir}")" || return 2
  if [[ -f "${owner_file}" ]]; then
    owner_pid="$(_fastwam_offline_env_lock_value "${lock_dir}" pid || true)"
    owner_host="$(_fastwam_offline_env_lock_value "${lock_dir}" host || true)"
    owner_time="$(_fastwam_offline_env_lock_value "${lock_dir}" time || true)"
  fi
  now="$(date +%s)"
  if [[ ! "${owner_time}" =~ ^[0-9]+$ ]]; then
    owner_time="${lock_time}"
  fi

  if [[ "${owner_host}" == "$(hostname)" && "${owner_pid}" =~ ^[0-9]+$ ]]; then
    if ! kill -0 "${owner_pid}" 2>/dev/null; then
      return 0
    fi
    return 1
  fi
  if ((now - owner_time >= stale_seconds)); then
    return 0
  fi
  return 1
}

_fastwam_offline_env_try_acquire_lock() {
  local lock_dir="$1"
  local stale_seconds="$2"
  local managed_parent="$3"
  local staging_prefix="$4"
  local stale_rc
  local snapshot_before
  local snapshot_after
  local stale_staging=""
  local quarantine
  local token

  token="$(hostname):${BASHPID}:$(date +%s):${RANDOM}:${RANDOM}"
  if mkdir -- "${lock_dir}" 2>/dev/null; then
    if ! _fastwam_offline_env_write_lock_owner "${lock_dir}" "${token}" "" 0; then
      quarantine="${lock_dir}.BROKEN.${BASHPID}.${RANDOM}"
      mv -T -- "${lock_dir}" "${quarantine}" 2>/dev/null || true
      [[ ! -e "${quarantine}" || -L "${quarantine}" ]] || rm -rf -- "${quarantine}"
      return 2
    fi
    FASTWAM_OFFLINE_ENV_ACQUIRED_LOCK_TOKEN="${token}"
    return 0
  fi
  if [[ -L "${lock_dir}" || ! -d "${lock_dir}" ]]; then
    echo "Error: lock path is a symlink or non-directory: ${lock_dir}" >&2
    return 2
  fi
  snapshot_before="$(_fastwam_offline_env_lock_snapshot "${lock_dir}")" || {
    echo "Error: cannot snapshot lock safely: ${lock_dir}" >&2
    return 2
  }
  stale_staging="$(_fastwam_offline_env_lock_value "${lock_dir}" staging 2>/dev/null || true)"
  if _fastwam_offline_env_lock_is_stale "${lock_dir}" "${stale_seconds}"; then
    snapshot_after="$(_fastwam_offline_env_lock_snapshot "${lock_dir}")" || return 1
    [[ "${snapshot_after}" == "${snapshot_before}" ]] || return 1
    quarantine="${lock_dir}.REAP.${BASHPID}.${RANDOM}"
    _fastwam_offline_env_log "action=reap_stale lock=${lock_dir}"
    if ! mv -T -- "${lock_dir}" "${quarantine}" 2>/dev/null; then
      return 1
    fi
    if [[ -n "${stale_staging}" ]]; then
      _fastwam_offline_env_safe_remove \
        "${stale_staging}" "${managed_parent}" "${staging_prefix}" || true
    fi
    rm -rf -- "${quarantine}"
    token="$(hostname):${BASHPID}:$(date +%s):${RANDOM}:${RANDOM}"
    if mkdir -- "${lock_dir}" 2>/dev/null; then
      if ! _fastwam_offline_env_write_lock_owner "${lock_dir}" "${token}" "" 0; then
        quarantine="${lock_dir}.BROKEN.${BASHPID}.${RANDOM}"
        mv -T -- "${lock_dir}" "${quarantine}" 2>/dev/null || true
        [[ ! -e "${quarantine}" || -L "${quarantine}" ]] || rm -rf -- "${quarantine}"
        return 2
      fi
      FASTWAM_OFFLINE_ENV_ACQUIRED_LOCK_TOKEN="${token}"
      return 0
    fi
  else
    stale_rc=$?
    if ((stale_rc == 2)); then
      return 2
    fi
  fi
  return 1
}

_fastwam_offline_env_safe_remove() {
  local target="$1"
  local expected_parent="$2"
  local expected_prefix="$3"
  local target_parent
  local target_name

  target_parent="$(dirname -- "${target}")"
  target_name="$(basename -- "${target}")"
  if [[ \
    "${target_parent}" != "${expected_parent}" || \
    "${target_name}" != "${expected_prefix}"* || \
    "${target}" == "${expected_parent}" || \
    -L "${target}" \
  ]]; then
    echo "Error: refusing unsafe managed-path removal: ${target}" >&2
    return 1
  fi
  if [[ -e "${target}" ]]; then
    rm -rf -- "${target}"
  fi
}

_fastwam_offline_env_wait_for_checkout() {
  local destination="$1"
  local ready_file="$2"
  local expected_marker="$3"
  local expected_commit="$4"
  local lock_dir="$5"
  local failure_file="$6"
  local timeout_seconds="$7"
  local stale_lock_seconds="$8"
  local started
  local now
  local stale_rc

  started="$(date +%s)"
  _fastwam_offline_env_log "action=wait kind=checkout destination=${destination}"
  while true; do
    if [[ ! -e "${lock_dir}" && ! -L "${lock_dir}" ]]; then
      if _fastwam_offline_env_validate_checkout \
        "${destination}" "${ready_file}" "${expected_marker}" "${expected_commit}"; then
        return 0
      fi
      if [[ -L "${failure_file}" ]]; then
        echo "Error: checkout failure marker is a symlink: ${failure_file}" >&2
        return 1
      fi
      if [[ -f "${failure_file}" ]]; then
        echo "Error: checkout builder failed: $(tr '\n' ' ' <"${failure_file}")" >&2
        return 1
      fi
      _fastwam_offline_env_log "action=retry_gap kind=checkout lock=${lock_dir}"
      return 75
    fi
    if [[ -L "${lock_dir}" || ! -d "${lock_dir}" ]]; then
      echo "Error: checkout lock is a symlink or non-directory: ${lock_dir}" >&2
      return 1
    fi
    if _fastwam_offline_env_lock_is_stale "${lock_dir}" "${stale_lock_seconds}"; then
      _fastwam_offline_env_log "action=retry_stale kind=checkout lock=${lock_dir}"
      return 75
    else
      stale_rc=$?
      if ((stale_rc == 2)); then
        return 1
      fi
    fi
    now="$(date +%s)"
    if ((now - started >= timeout_seconds)); then
      echo "Error: timed out waiting for checkout ${destination}." >&2
      return 1
    fi
    sleep 1
  done
}

_fastwam_offline_env_wait_for_venv() {
  local destination="$1"
  local expected_marker="$2"
  local expected_version="$3"
  local expected_implementation="$4"
  local expected_abi="$5"
  local expected_cache_tag="$6"
  local expected_platform="$7"
  local expected_executable_sha="$8"
  local lock_dir="$9"
  local failure_file="${10}"
  local timeout_seconds="${11}"
  local stale_lock_seconds="${12}"
  local started
  local now
  local stale_rc

  started="$(date +%s)"
  _fastwam_offline_env_log "action=wait kind=venv destination=${destination}"
  while true; do
    if [[ ! -e "${lock_dir}" && ! -L "${lock_dir}" ]]; then
      if _fastwam_offline_env_validate_venv \
        "${destination}" "${expected_marker}" "${expected_version}" \
        "${expected_implementation}" "${expected_abi}" "${expected_cache_tag}" \
        "${expected_platform}" "${expected_executable_sha}"; then
        return 0
      fi
      if [[ -L "${failure_file}" ]]; then
        echo "Error: venv failure marker is a symlink: ${failure_file}" >&2
        return 1
      fi
      if [[ -f "${failure_file}" ]]; then
        echo "Error: venv builder failed: $(tr '\n' ' ' <"${failure_file}")" >&2
        return 1
      fi
      _fastwam_offline_env_log "action=retry_gap kind=venv lock=${lock_dir}"
      return 75
    fi
    if [[ -L "${lock_dir}" || ! -d "${lock_dir}" ]]; then
      echo "Error: venv lock is a symlink or non-directory: ${lock_dir}" >&2
      return 1
    fi
    if _fastwam_offline_env_lock_is_stale "${lock_dir}" "${stale_lock_seconds}"; then
      _fastwam_offline_env_log "action=retry_stale kind=venv lock=${lock_dir}"
      return 75
    else
      stale_rc=$?
      if ((stale_rc == 2)); then
        return 1
      fi
    fi
    now="$(date +%s)"
    if ((now - started >= timeout_seconds)); then
      echo "Error: timed out waiting for venv ${destination}." >&2
      return 1
    fi
    sleep 1
  done
}

_fastwam_offline_env_build_checkout() {
  local source_bundle="$1"
  local destination="$2"
  local ready_file="$3"
  local expected_marker="$4"
  local expected_commit="$5"
  local lock_dir="$6"
  local failure_file="$7"
  local checkout_root="$8"
  local checkout_prefix="$9"
  local lock_token="${10}"

  (
    set -Eeuo pipefail
    local staging="${checkout_root}/.${checkout_prefix}.STAGING.${BASHPID}.${RANDOM}"
    local ready_temporary="${ready_file}.tmp.${BASHPID}.${RANDOM}"
    local published=0
    local complete=0

    _fastwam_checkout_cleanup() {
      local rc=$?
      local owns_lock=0
      trap - EXIT INT TERM
      if _fastwam_offline_env_lock_is_owned "${lock_dir}" "${lock_token}"; then
        owns_lock=1
      fi
      if ((complete == 0)); then
        _fastwam_offline_env_safe_remove "${staging}" "${checkout_root}" ".${checkout_prefix}.STAGING." || true
        if ((owns_lock == 1 && published == 1)); then
          _fastwam_offline_env_safe_remove "${destination}" "${checkout_root}" "${checkout_prefix}" || true
        fi
        if [[ -e "${ready_temporary}" && ! -L "${ready_temporary}" ]]; then
          rm -f -- "${ready_temporary}"
        fi
        if ((owns_lock == 1)) && [[ -e "${ready_file}" && ! -L "${ready_file}" ]]; then
          rm -f -- "${ready_file}"
        fi
        if ((owns_lock == 1)); then
          _fastwam_offline_env_write_failure "${failure_file}" "checkout build exited ${rc}" || true
        fi
      fi
      if ((owns_lock == 1)); then
        _fastwam_offline_env_release_lock "${lock_dir}" "${lock_token}" || true
      fi
      exit "${rc}"
    }
    trap _fastwam_checkout_cleanup EXIT
    trap 'exit 130' INT TERM

    _fastwam_offline_env_write_lock_owner "${lock_dir}" "${lock_token}" "${staging}" || {
      echo "Error: checkout builder could not publish lock ownership." >&2
      return 1
    }
    if [[ -L "${failure_file}" || -L "${ready_file}" || -L "${destination}" ]]; then
      echo "Error: checkout destination/marker path must not be a symlink." >&2
      return 1
    fi
    rm -f -- "${failure_file}" "${ready_file}" || return 1
    if [[ -e "${destination}" ]]; then
      _fastwam_offline_env_safe_remove \
        "${destination}" "${checkout_root}" "${checkout_prefix}" || return 1
    fi
    if [[ -e "${staging}" || -L "${staging}" ]]; then
      echo "Error: unique checkout staging path already exists: ${staging}" >&2
      return 1
    fi

    git clone --no-hardlinks -- "${source_bundle}" "${staging}" || {
      echo "Error: failed to clone the verified source bundle." >&2
      return 1
    }
    git -C "${staging}" checkout --detach "${expected_commit}" || {
      echo "Error: bundled Git repository does not contain the exact requested commit." >&2
      return 1
    }
    _fastwam_offline_env_validate_checkout_tree "${staging}" "${expected_commit}" || {
      echo "Error: staged checkout failed exact commit/clean-tree validation." >&2
      return 1
    }
    _fastwam_offline_env_lock_is_owned "${lock_dir}" "${lock_token}" || {
      echo "Error: checkout builder lost its identity lock before publication." >&2
      return 1
    }
    mv -T -- "${staging}" "${destination}" || return 1
    published=1
    _fastwam_offline_env_validate_checkout_tree "${destination}" "${expected_commit}" || {
      echo "Error: published checkout failed validation." >&2
      return 1
    }
    printf '%s\n' "${expected_marker}" >"${ready_temporary}" || return 1
    chmod 0444 "${ready_temporary}" || return 1
    _fastwam_offline_env_lock_is_owned "${lock_dir}" "${lock_token}" || {
      echo "Error: checkout builder lost its identity lock before READY." >&2
      return 1
    }
    mv -T -- "${ready_temporary}" "${ready_file}" || return 1
    _fastwam_offline_env_validate_checkout \
      "${destination}" "${ready_file}" "${expected_marker}" "${expected_commit}" || {
      echo "Error: checkout READY validation failed after publication." >&2
      return 1
    }

    _fastwam_offline_env_release_lock "${lock_dir}" "${lock_token}" || return 1
    complete=1
    trap - EXIT INT TERM
  )
}

_fastwam_offline_env_prepare_checkout() {
  local source_bundle="$1"
  local checkout_root="$2"
  local checkout_identity="$3"
  local expected_marker="$4"
  local expected_commit="$5"
  local wait_timeout="$6"
  local stale_lock_seconds="$7"
  local prefix="source-${checkout_identity}"
  local destination="${checkout_root}/${prefix}"
  local ready_file="${checkout_root}/.${prefix}.READY"
  local lock_dir="${checkout_root}/.${prefix}.LOCK"
  local failure_file="${checkout_root}/.${prefix}.FAILED"
  local acquire_rc
  local wait_rc
  local lock_token

  while true; do
    if [[ ! -e "${lock_dir}" && ! -L "${lock_dir}" ]] && \
      _fastwam_offline_env_validate_checkout \
        "${destination}" "${ready_file}" "${expected_marker}" "${expected_commit}"; then
      _fastwam_offline_env_log "action=hit kind=checkout destination=${destination}"
      printf '%s\n' "${destination}"
      return 0
    fi
    if _fastwam_offline_env_try_acquire_lock \
      "${lock_dir}" "${stale_lock_seconds}" "${checkout_root}" ".${prefix}.STAGING."; then
      lock_token="${FASTWAM_OFFLINE_ENV_ACQUIRED_LOCK_TOKEN}"
      # READY may have appeared between the initial hit check and mkdir, or a
      # dead owner may have published valid state before losing its lock.
      # Recheck while holding our token so a recovery owner never replaces a
      # destination that another caller can already consume.
      if _fastwam_offline_env_validate_checkout \
        "${destination}" "${ready_file}" "${expected_marker}" "${expected_commit}"; then
        if [[ -L "${failure_file}" ]]; then
          echo "Error: checkout failure marker is a symlink: ${failure_file}" >&2
          _fastwam_offline_env_release_lock "${lock_dir}" "${lock_token}" || true
          return 1
        fi
        rm -f -- "${failure_file}" || {
          _fastwam_offline_env_release_lock "${lock_dir}" "${lock_token}" || true
          return 1
        }
        _fastwam_offline_env_release_lock "${lock_dir}" "${lock_token}" || return 1
        _fastwam_offline_env_log "action=recovered_hit kind=checkout destination=${destination}"
        printf '%s\n' "${destination}"
        return 0
      fi
      _fastwam_offline_env_log "action=build kind=checkout destination=${destination}"
      _fastwam_offline_env_build_checkout \
        "${source_bundle}" "${destination}" "${ready_file}" "${expected_marker}" \
        "${expected_commit}" "${lock_dir}" "${failure_file}" "${checkout_root}" \
        "${prefix}" "${lock_token}" >&2 || return 1
      _fastwam_offline_env_validate_checkout \
        "${destination}" "${ready_file}" "${expected_marker}" "${expected_commit}" || {
        echo "Error: final checkout validation failed: ${destination}" >&2
        return 1
      }
      printf '%s\n' "${destination}"
      return 0
    else
      acquire_rc=$?
      if ((acquire_rc != 1)); then
        return "${acquire_rc}"
      fi
    fi
    if _fastwam_offline_env_wait_for_checkout \
      "${destination}" "${ready_file}" "${expected_marker}" "${expected_commit}" \
      "${lock_dir}" "${failure_file}" "${wait_timeout}" "${stale_lock_seconds}"; then
      printf '%s\n' "${destination}"
      return 0
    else
      wait_rc=$?
      if ((wait_rc == 75)); then
        continue
      fi
      return "${wait_rc}"
    fi
  done
}

_fastwam_offline_env_build_venv() {
  local runtime_lock="$1"
  local wheelhouse="$2"
  local checkout="$3"
  local destination="$4"
  local expected_marker="$5"
  local lock_dir="$6"
  local failure_file="$7"
  local venv_root="$8"
  local venv_prefix="$9"
  local expected_version="${10}"
  local expected_implementation="${11}"
  local expected_abi="${12}"
  local expected_cache_tag="${13}"
  local expected_platform="${14}"
  local expected_executable_sha="${15}"
  local base_python="${16}"
  local lock_token="${17}"

  (
    set -Eeuo pipefail
    local staging="${venv_root}/.${venv_prefix}.STAGING.${BASHPID}.${RANDOM}"
    local marker_file="${staging}/.FASTWAM_ENV_READY"
    local marker_temporary="${staging}/.FASTWAM_ENV_READY.tmp.${BASHPID}.${RANDOM}"
    local published=0
    local complete=0

    _fastwam_venv_cleanup() {
      local rc=$?
      local owns_lock=0
      trap - EXIT INT TERM
      if _fastwam_offline_env_lock_is_owned "${lock_dir}" "${lock_token}"; then
        owns_lock=1
      fi
      if ((complete == 0)); then
        _fastwam_offline_env_safe_remove "${staging}" "${venv_root}" ".${venv_prefix}.STAGING." || true
        if ((owns_lock == 1 && published == 1)); then
          _fastwam_offline_env_safe_remove "${destination}" "${venv_root}" "${venv_prefix}" || true
        fi
        if ((owns_lock == 1)); then
          _fastwam_offline_env_write_failure "${failure_file}" "venv build exited ${rc}" || true
        fi
      fi
      if ((owns_lock == 1)); then
        _fastwam_offline_env_release_lock "${lock_dir}" "${lock_token}" || true
      fi
      exit "${rc}"
    }
    trap _fastwam_venv_cleanup EXIT
    trap 'exit 130' INT TERM

    _fastwam_offline_env_write_lock_owner "${lock_dir}" "${lock_token}" "${staging}" || {
      echo "Error: venv builder could not publish lock ownership." >&2
      return 1
    }
    if [[ -L "${failure_file}" || -L "${destination}" ]]; then
      echo "Error: venv destination/failure path must not be a symlink." >&2
      return 1
    fi
    rm -f -- "${failure_file}" || return 1
    if [[ -e "${destination}" ]]; then
      _fastwam_offline_env_safe_remove \
        "${destination}" "${venv_root}" "${venv_prefix}" || return 1
    fi
    if [[ -e "${staging}" || -L "${staging}" ]]; then
      echo "Error: unique venv staging path already exists: ${staging}" >&2
      return 1
    fi

    env -u PYTHONHOME -u PYTHONPATH \
      PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
      "${base_python}" -I -m venv --copies "${staging}" || {
      echo "Error: isolated venv creation failed." >&2
      return 1
    }
    env -u PYTHONHOME -u PYTHONPATH \
      PIP_NO_INDEX=1 PIP_NO_CACHE_DIR=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
      "${staging}/bin/python" -I -m pip install \
        --no-index \
        --find-links "${wheelhouse}" \
        --require-hashes \
        -r "${runtime_lock}" || {
      echo "Error: hash-locked offline package installation failed." >&2
      return 1
    }
    _fastwam_offline_env_python_runtime_matches \
      "${staging}/bin/python" "${expected_version}" "${expected_implementation}" \
      "${expected_abi}" "${expected_cache_tag}" "${expected_platform}" \
      "${expected_executable_sha}" || return 1
    _fastwam_offline_env_validate_training_runtime \
      "${staging}/bin/python" "${checkout}" || return 1

    # The marker is the final staged write and is never visible at the final
    # destination until every runtime and import-origin gate has passed.
    printf '%s\n' "${expected_marker}" >"${marker_temporary}" || return 1
    chmod 0444 "${marker_temporary}" || return 1
    mv -T -- "${marker_temporary}" "${marker_file}" || return 1
    _fastwam_offline_env_validate_venv \
      "${staging}" "${expected_marker}" "${expected_version}" \
      "${expected_implementation}" "${expected_abi}" "${expected_cache_tag}" \
      "${expected_platform}" "${expected_executable_sha}" || return 1
    _fastwam_offline_env_lock_is_owned "${lock_dir}" "${lock_token}" || {
      echo "Error: venv builder lost its identity lock before publication." >&2
      return 1
    }
    mv -T -- "${staging}" "${destination}" || return 1
    published=1
    _fastwam_offline_env_validate_venv \
      "${destination}" "${expected_marker}" "${expected_version}" \
      "${expected_implementation}" "${expected_abi}" "${expected_cache_tag}" \
      "${expected_platform}" "${expected_executable_sha}" || return 1
    _fastwam_offline_env_validate_training_runtime \
      "${destination}/bin/python" "${checkout}" || return 1

    _fastwam_offline_env_lock_is_owned "${lock_dir}" "${lock_token}" || {
      echo "Error: venv builder lost its identity lock after publication." >&2
      return 1
    }
    _fastwam_offline_env_release_lock "${lock_dir}" "${lock_token}" || return 1
    complete=1
    trap - EXIT INT TERM
  )
}

_fastwam_offline_env_prepare_venv() {
  local runtime_lock="$1"
  local wheelhouse="$2"
  local checkout="$3"
  local venv_root="$4"
  local environment_identity="$5"
  local expected_marker="$6"
  local wait_timeout="$7"
  local stale_lock_seconds="$8"
  local expected_version="$9"
  local expected_implementation="${10}"
  local expected_abi="${11}"
  local expected_cache_tag="${12}"
  local expected_platform="${13}"
  local expected_executable_sha="${14}"
  local base_python="${15}"
  local prefix="cpython3.10-${environment_identity}"
  local destination="${venv_root}/${prefix}"
  local lock_dir="${venv_root}/.${prefix}.LOCK"
  local failure_file="${venv_root}/.${prefix}.FAILED"
  local acquire_rc
  local wait_rc
  local lock_token

  while true; do
    if [[ ! -e "${lock_dir}" && ! -L "${lock_dir}" ]] && \
      _fastwam_offline_env_validate_venv \
        "${destination}" "${expected_marker}" "${expected_version}" \
        "${expected_implementation}" "${expected_abi}" "${expected_cache_tag}" \
        "${expected_platform}" "${expected_executable_sha}"; then
      _fastwam_offline_env_validate_training_runtime \
        "${destination}/bin/python" "${checkout}" >&2 || return 1
      _fastwam_offline_env_log "action=hit kind=venv destination=${destination}"
      printf '%s\n' "${destination}"
      return 0
    fi
    if _fastwam_offline_env_try_acquire_lock \
      "${lock_dir}" "${stale_lock_seconds}" "${venv_root}" ".${prefix}.STAGING."; then
      lock_token="${FASTWAM_OFFLINE_ENV_ACQUIRED_LOCK_TOKEN}"
      if _fastwam_offline_env_validate_venv \
        "${destination}" "${expected_marker}" "${expected_version}" \
        "${expected_implementation}" "${expected_abi}" "${expected_cache_tag}" \
        "${expected_platform}" "${expected_executable_sha}" && \
        _fastwam_offline_env_validate_training_runtime \
          "${destination}/bin/python" "${checkout}" >&2; then
        if [[ -L "${failure_file}" ]]; then
          echo "Error: venv failure marker is a symlink: ${failure_file}" >&2
          _fastwam_offline_env_release_lock "${lock_dir}" "${lock_token}" || true
          return 1
        fi
        rm -f -- "${failure_file}" || {
          _fastwam_offline_env_release_lock "${lock_dir}" "${lock_token}" || true
          return 1
        }
        _fastwam_offline_env_release_lock "${lock_dir}" "${lock_token}" || return 1
        _fastwam_offline_env_log "action=recovered_hit kind=venv destination=${destination}"
        printf '%s\n' "${destination}"
        return 0
      fi
      _fastwam_offline_env_log "action=build kind=venv destination=${destination}"
      _fastwam_offline_env_build_venv \
        "${runtime_lock}" "${wheelhouse}" "${checkout}" "${destination}" \
        "${expected_marker}" "${lock_dir}" "${failure_file}" "${venv_root}" "${prefix}" \
        "${expected_version}" "${expected_implementation}" "${expected_abi}" \
        "${expected_cache_tag}" "${expected_platform}" "${expected_executable_sha}" \
        "${base_python}" "${lock_token}" >&2 || return 1
      _fastwam_offline_env_validate_venv \
        "${destination}" "${expected_marker}" "${expected_version}" \
        "${expected_implementation}" "${expected_abi}" "${expected_cache_tag}" \
        "${expected_platform}" "${expected_executable_sha}" || {
        echo "Error: final venv validation failed: ${destination}" >&2
        return 1
      }
      _fastwam_offline_env_validate_training_runtime \
        "${destination}/bin/python" "${checkout}" >&2 || return 1
      printf '%s\n' "${destination}"
      return 0
    else
      acquire_rc=$?
      if ((acquire_rc != 1)); then
        return "${acquire_rc}"
      fi
    fi
    if _fastwam_offline_env_wait_for_venv \
      "${destination}" "${expected_marker}" "${expected_version}" \
      "${expected_implementation}" "${expected_abi}" "${expected_cache_tag}" \
      "${expected_platform}" "${expected_executable_sha}" "${lock_dir}" \
      "${failure_file}" "${wait_timeout}" "${stale_lock_seconds}"; then
      _fastwam_offline_env_validate_training_runtime \
        "${destination}/bin/python" "${checkout}" >&2 || return 1
      printf '%s\n' "${destination}"
      return 0
    else
      wait_rc=$?
      if ((wait_rc == 75)); then
        continue
      fi
      return "${wait_rc}"
    fi
  done
}

fastwam_prepare_offline_training_env() {
  local script_path
  local script_dir
  local source_root_raw="${FASTWAM_OFFLINE_ENV_SOURCE_ROOT-}"
  local manifest_raw="${FASTWAM_OFFLINE_ENV_MANIFEST-}"
  local manifest_sha="${FASTWAM_OFFLINE_ENV_MANIFEST_SHA256-}"
  local runtime_lock_sha="${FASTWAM_OFFLINE_ENV_RUNTIME_LOCK_SHA256-}"
  local code_commit="${FASTWAM_CODE_COMMIT-}"
  local source_bundle_relative_raw="${FASTWAM_OFFLINE_ENV_SOURCE_BUNDLE_RELATIVE_PATH-}"
  local cache_helper_sha="${FASTWAM_OFFLINE_ENV_CACHE_HELPER_SHA256-}"
  local runtime_lock_relative_raw="${FASTWAM_OFFLINE_ENV_RUNTIME_LOCK_RELATIVE_PATH:-locks/requirements-runtime-offline-hashed.txt}"
  local cache_root_raw="${FASTWAM_OFFLINE_ENV_CACHE_ROOT:-/tmp/fastwam-offline-env-cache}"
  local venv_root_raw="${FASTWAM_OFFLINE_ENV_VENV_ROOT:-/tmp/fastwam-offline-env-venvs}"
  local checkout_root_raw="${FASTWAM_SOURCE_CHECKOUT_ROOT:-/tmp/fastwam-source-checkouts}"
  local minimum_free_bytes="${FASTWAM_OFFLINE_ENV_MIN_FREE_BYTES:-17179869184}"
  local wait_timeout="${FASTWAM_OFFLINE_ENV_WAIT_TIMEOUT:-${FASTWAM_LOCAL_CACHE_WAIT_TIMEOUT:-7200}}"
  local stale_lock_seconds="${FASTWAM_OFFLINE_ENV_STALE_LOCK_SECONDS:-${FASTWAM_LOCAL_CACHE_STALE_LOCK_SECONDS:-7200}}"
  local source_root
  local manifest
  local actual_manifest_sha
  local actual_helper_sha
  local source_bundle_relative
  local runtime_lock_relative
  local cache_root
  local venv_root
  local checkout_root
  local payload_dir
  local payload_canonical
  local runtime_lock
  local source_bundle
  local wheelhouse
  local actual_runtime_lock_sha
  local source_bundle_sha
  local checkout_identity
  local environment_identity
  local checkout_marker
  local environment_marker
  local checkout
  local venv

  for required_name in \
    FASTWAM_OFFLINE_ENV_SOURCE_ROOT \
    FASTWAM_OFFLINE_ENV_MANIFEST \
    FASTWAM_OFFLINE_ENV_MANIFEST_SHA256 \
    FASTWAM_OFFLINE_ENV_RUNTIME_LOCK_SHA256 \
    FASTWAM_CODE_COMMIT \
    FASTWAM_OFFLINE_ENV_SOURCE_BUNDLE_RELATIVE_PATH \
    FASTWAM_OFFLINE_ENV_CACHE_HELPER_SHA256; do
    if [[ -z "${!required_name-}" ]]; then
      echo "Error: required environment variable ${required_name} is unset or empty." >&2
      return 1
    fi
  done
  _fastwam_offline_env_validate_sha256 FASTWAM_OFFLINE_ENV_MANIFEST_SHA256 "${manifest_sha}" || return 1
  _fastwam_offline_env_validate_sha256 FASTWAM_OFFLINE_ENV_RUNTIME_LOCK_SHA256 "${runtime_lock_sha}" || return 1
  _fastwam_offline_env_validate_sha256 FASTWAM_OFFLINE_ENV_CACHE_HELPER_SHA256 "${cache_helper_sha}" || return 1
  _fastwam_offline_env_validate_commit FASTWAM_CODE_COMMIT "${code_commit}" || return 1
  _fastwam_offline_env_validate_nonnegative_integer FASTWAM_OFFLINE_ENV_MIN_FREE_BYTES "${minimum_free_bytes}" || return 1
  _fastwam_offline_env_validate_positive_integer FASTWAM_OFFLINE_ENV_WAIT_TIMEOUT "${wait_timeout}" || return 1
  _fastwam_offline_env_validate_positive_integer FASTWAM_OFFLINE_ENV_STALE_LOCK_SECONDS "${stale_lock_seconds}" || return 1

  source_bundle_relative="$(_fastwam_offline_env_normalize_relative_path \
    FASTWAM_OFFLINE_ENV_SOURCE_BUNDLE_RELATIVE_PATH "${source_bundle_relative_raw}")" || return 1
  runtime_lock_relative="$(_fastwam_offline_env_normalize_relative_path \
    FASTWAM_OFFLINE_ENV_RUNTIME_LOCK_RELATIVE_PATH "${runtime_lock_relative_raw}")" || return 1

  cache_root="$(_fastwam_offline_env_prepare_tmp_root FASTWAM_OFFLINE_ENV_CACHE_ROOT "${cache_root_raw}")" || return 1
  venv_root="$(_fastwam_offline_env_prepare_tmp_root FASTWAM_OFFLINE_ENV_VENV_ROOT "${venv_root_raw}")" || return 1
  checkout_root="$(_fastwam_offline_env_prepare_tmp_root FASTWAM_SOURCE_CHECKOUT_ROOT "${checkout_root_raw}")" || return 1
  if [[ \
    "${cache_root}" == "${venv_root}" || \
    "${cache_root}" == "${checkout_root}" || \
    "${venv_root}" == "${checkout_root}" || \
    "${cache_root}" == "${venv_root}"/* || \
    "${venv_root}" == "${cache_root}"/* || \
    "${cache_root}" == "${checkout_root}"/* || \
    "${checkout_root}" == "${cache_root}"/* || \
    "${venv_root}" == "${checkout_root}"/* || \
    "${checkout_root}" == "${venv_root}"/* \
  ]]; then
    echo "Error: cache, venv, and checkout roots must be distinct and non-nested." >&2
    return 1
  fi

  if [[ ! -d "${source_root_raw}" || -L "${source_root_raw}" ]]; then
    echo "Error: FASTWAM_OFFLINE_ENV_SOURCE_ROOT must be a real non-symlink directory." >&2
    return 1
  fi
  source_root="$(realpath -e -- "${source_root_raw}")"
  case "${source_root}" in
    /oss-chengjuntao | /oss-chengjuntao/*) ;;
    *)
      echo "Error: FASTWAM_OFFLINE_ENV_SOURCE_ROOT must resolve below /oss-chengjuntao." >&2
      return 1
      ;;
  esac
  if [[ "${source_root}" != "${source_root_raw}" ]]; then
    echo "Error: FASTWAM_OFFLINE_ENV_SOURCE_ROOT must be supplied canonically: ${source_root}" >&2
    return 1
  fi
  if [[ ! -f "${manifest_raw}" || -L "${manifest_raw}" ]]; then
    echo "Error: FASTWAM_OFFLINE_ENV_MANIFEST must be a regular non-symlink file." >&2
    return 1
  fi
  manifest="$(realpath -e -- "${manifest_raw}")"
  actual_manifest_sha="$(sha256sum -- "${manifest}" | awk '{print $1}')"
  if [[ "${actual_manifest_sha}" != "${manifest_sha}" ]]; then
    echo "Error: offline environment manifest SHA-256 mismatch." >&2
    return 1
  fi

  script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
  script_dir="$(dirname -- "${script_path}")"
  if [[ ! -f "${script_dir}/dlc_local_cache.sh" || -L "${script_dir}/dlc_local_cache.sh" ]]; then
    echo "Error: sibling dlc_local_cache.sh must be a regular non-symlink file." >&2
    return 1
  fi
  actual_helper_sha="$(sha256sum -- "${script_dir}/dlc_local_cache.sh" | awk '{print $1}')"
  if [[ "${actual_helper_sha}" != "${cache_helper_sha}" ]]; then
    echo "Error: dlc_local_cache.sh SHA-256 mismatch." >&2
    return 1
  fi
  command -v git >/dev/null 2>&1 || {
    echo "Error: git is required to restore the immutable source checkout." >&2
    return 1
  }
  _fastwam_offline_env_bind_python_identity || return 1

  payload_dir="$(_fastwam_offline_env_stage_payload \
    "${script_dir}" "${source_root}" "${manifest}" "${manifest_sha}" \
    "${cache_root}" "${minimum_free_bytes}")" || return 1
  if [[ ! -d "${payload_dir}" || -L "${payload_dir}" ]]; then
    echo "Error: cache helper returned a non-directory or symlink payload: ${payload_dir}" >&2
    return 1
  fi
  payload_canonical="$(realpath -e -- "${payload_dir}")"
  case "${payload_canonical}" in
    "${cache_root}"/*) ;;
    *)
      echo "Error: cache helper payload escaped canonical cache root." >&2
      return 1
      ;;
  esac
  if [[ "${payload_canonical}" != "${payload_dir}" ]]; then
    echo "Error: cache helper payload must be returned canonically." >&2
    return 1
  fi

  runtime_lock="${payload_canonical}/${runtime_lock_relative}"
  source_bundle="${payload_canonical}/${source_bundle_relative}"
  wheelhouse="${payload_canonical}/wheelhouse"
  if [[ ! -f "${runtime_lock}" || -L "${runtime_lock}" ]]; then
    echo "Error: runtime lock must be a regular non-symlink file: ${runtime_lock}" >&2
    return 1
  fi
  if [[ ! -f "${source_bundle}" || -L "${source_bundle}" ]]; then
    echo "Error: source Git bundle must be a regular non-symlink file: ${source_bundle}" >&2
    return 1
  fi
  if [[ ! -d "${wheelhouse}" || -L "${wheelhouse}" ]]; then
    echo "Error: wheelhouse must be a real non-symlink directory: ${wheelhouse}" >&2
    return 1
  fi
  for payload_path in "${runtime_lock}" "${source_bundle}" "${wheelhouse}"; do
    case "$(realpath -e -- "${payload_path}")" in
      "${payload_canonical}"/*) ;;
      *)
        echo "Error: offline payload path escaped verified cache: ${payload_path}" >&2
        return 1
        ;;
    esac
  done
  actual_runtime_lock_sha="$(sha256sum -- "${runtime_lock}" | awk '{print $1}')"
  if [[ "${actual_runtime_lock_sha}" != "${runtime_lock_sha}" ]]; then
    echo "Error: FASTWAM_OFFLINE_ENV_RUNTIME_LOCK_SHA256 mismatch." >&2
    return 1
  fi
  source_bundle_sha="$(sha256sum -- "${source_bundle}" | awk '{print $1}')"

  checkout_identity="$({
    printf 'schema=fastwam-offline-checkout-v2\n'
    printf 'commit=%s\n' "${code_commit}"
    printf 'source_bundle_sha256=%s\n' "${source_bundle_sha}"
  } | sha256sum | awk '{print $1}')"
  printf -v checkout_marker '%s\n%s\n%s' \
    'schema=fastwam-offline-checkout-v2' \
    "commit=${code_commit}" \
    "source_bundle_sha256=${source_bundle_sha}"

  checkout="$(_fastwam_offline_env_prepare_checkout \
    "${source_bundle}" "${checkout_root}" "${checkout_identity}" "${checkout_marker}" \
    "${code_commit}" "${wait_timeout}" "${stale_lock_seconds}")" || return 1

  environment_identity="$({
    printf 'schema=fastwam-offline-environment-v2\n'
    printf 'manifest_sha256=%s\n' "${manifest_sha}"
    printf 'commit=%s\n' "${code_commit}"
    printf 'runtime_lock_sha256=%s\n' "${runtime_lock_sha}"
    printf 'cache_helper_sha256=%s\n' "${cache_helper_sha}"
    printf 'source_bundle_sha256=%s\n' "${source_bundle_sha}"
    printf 'python_identity_sha256=%s\n' "${FASTWAM_OFFLINE_ENV_PYTHON_IDENTITY_SHA256}"
  } | sha256sum | awk '{print $1}')"
  printf -v environment_marker '%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s' \
    'schema=fastwam-offline-environment-v2' \
    "manifest_sha256=${manifest_sha}" \
    "commit=${code_commit}" \
    "runtime_lock_sha256=${runtime_lock_sha}" \
    "cache_helper_sha256=${cache_helper_sha}" \
    "source_bundle_sha256=${source_bundle_sha}" \
    "python_version=${FASTWAM_OFFLINE_ENV_PYTHON_VERSION}" \
    "python_implementation=${FASTWAM_OFFLINE_ENV_PYTHON_IMPLEMENTATION}" \
    "python_abi=${FASTWAM_OFFLINE_ENV_PYTHON_ABI}" \
    "python_cache_tag=${FASTWAM_OFFLINE_ENV_PYTHON_CACHE_TAG}" \
    "python_platform=${FASTWAM_OFFLINE_ENV_PYTHON_PLATFORM}" \
    "python_executable=${FASTWAM_OFFLINE_ENV_BASE_PYTHON_EXECUTABLE}" \
    "python_executable_sha256=${FASTWAM_OFFLINE_ENV_PYTHON_EXECUTABLE_SHA256}" \
    "python_identity_sha256=${FASTWAM_OFFLINE_ENV_PYTHON_IDENTITY_SHA256}" \
    "environment_identity_sha256=${environment_identity}"

  venv="$(_fastwam_offline_env_prepare_venv \
    "${runtime_lock}" "${wheelhouse}" "${checkout}" "${venv_root}" \
    "${environment_identity}" "${environment_marker}" "${wait_timeout}" \
    "${stale_lock_seconds}" "${FASTWAM_OFFLINE_ENV_PYTHON_VERSION}" \
    "${FASTWAM_OFFLINE_ENV_PYTHON_IMPLEMENTATION}" \
    "${FASTWAM_OFFLINE_ENV_PYTHON_ABI}" "${FASTWAM_OFFLINE_ENV_PYTHON_CACHE_TAG}" \
    "${FASTWAM_OFFLINE_ENV_PYTHON_PLATFORM}" \
    "${FASTWAM_OFFLINE_ENV_PYTHON_EXECUTABLE_SHA256}" \
    "${FASTWAM_OFFLINE_ENV_BASE_PYTHON_EXECUTABLE}")" || return 1

  _fastwam_offline_env_validate_checkout \
    "${checkout}" "${checkout_root}/.source-${checkout_identity}.READY" \
    "${checkout_marker}" "${code_commit}" || return 1
  _fastwam_offline_env_validate_venv \
    "${venv}" "${environment_marker}" "${FASTWAM_OFFLINE_ENV_PYTHON_VERSION}" \
    "${FASTWAM_OFFLINE_ENV_PYTHON_IMPLEMENTATION}" \
    "${FASTWAM_OFFLINE_ENV_PYTHON_ABI}" "${FASTWAM_OFFLINE_ENV_PYTHON_CACHE_TAG}" \
    "${FASTWAM_OFFLINE_ENV_PYTHON_PLATFORM}" \
    "${FASTWAM_OFFLINE_ENV_PYTHON_EXECUTABLE_SHA256}" || return 1
  _fastwam_offline_env_validate_training_runtime "${venv}/bin/python" "${checkout}" || return 1

  unset PYTHONHOME
  export PYTHONNOUSERSITE=1
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="${checkout}/src"
  export FASTWAM_PYTHON="${venv}/bin/python"
  export FASTWAM_REPO_ROOT="${checkout}"
  export FASTWAM_TRAIN_SCRIPT="${FASTWAM_REPO_ROOT}/scripts/train_zero2.sh"
  export FASTWAM_TRAINING_ENV_BUNDLE_MANIFEST_SHA256="${manifest_sha}"
  export FASTWAM_OFFLINE_ENV_IDENTITY_SHA256="${environment_identity}"

  _fastwam_offline_env_validate_import_resolution "${FASTWAM_PYTHON}" "${FASTWAM_REPO_ROOT}" || return 1
  _fastwam_offline_env_log "status=READY identity=${environment_identity}"
  _fastwam_offline_env_log "outer command must invoke: ${FASTWAM_REPO_ROOT}/scripts/train_zero2.sh"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  fastwam_prepare_offline_training_env
  printf 'FASTWAM_PYTHON=%s\n' "${FASTWAM_PYTHON}"
  printf 'FASTWAM_REPO_ROOT=%s\n' "${FASTWAM_REPO_ROOT}"
  printf 'FASTWAM_TRAIN_SCRIPT=%s\n' "${FASTWAM_TRAIN_SCRIPT}"
  printf 'PYTHONPATH=%s\n' "${PYTHONPATH}"
  printf 'FASTWAM_OFFLINE_ENV_IDENTITY_SHA256=%s\n' "${FASTWAM_OFFLINE_ENV_IDENTITY_SHA256}"
fi
