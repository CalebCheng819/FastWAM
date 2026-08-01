#!/usr/bin/env bash
# Source this file, then call fastwam_prepare_erdma_userspace.  The function
# verifies a pinned OSS/CPFS bundle, atomically unpacks it into content-addressed
# node-local /tmp, and exports the provider paths into the caller's shell.

_fastwam_erdma_fail() {
  echo "ERDMA_PREPARE=FAIL reason=$1" >&2
  return 1
}

_fastwam_erdma_require_command() {
  command -v "$1" >/dev/null 2>&1 || _fastwam_erdma_fail "missing_command:$1"
}

_fastwam_erdma_require_regular_file() {
  [[ -f "$1" && ! -L "$1" ]] || _fastwam_erdma_fail "not_regular_file:$1"
}

_fastwam_erdma_sha256() {
  sha256sum -- "$1" | awk '{print $1}'
}

_fastwam_erdma_prepend_path() {
  local name="$1"
  local directory="$2"
  local current="${!name:-}"
  case ":${current}:" in
    *":${directory}:"*) ;;
    *) printf -v "${name}" '%s' "${directory}${current:+:${current}}" ;;
  esac
  export "${name}"
}

_fastwam_erdma_validate_stage() {
  local stage_root="$1"
  local expected_version="$2"
  local expected_bundle_sha256="$3"
  local expected_manifest_sha256="$4"

  [[ -d "${stage_root}" && ! -L "${stage_root}" ]] || return 1
  [[ -f "${stage_root}/READY" && ! -L "${stage_root}/READY" ]] || return 1
  grep -Fxq "provider_version=${expected_version}" "${stage_root}/READY" || return 1
  grep -Fxq "bundle_sha256=${expected_bundle_sha256}" "${stage_root}/READY" || return 1
  grep -Fxq "source_manifest_sha256=${expected_manifest_sha256}" "${stage_root}/READY" || return 1
  [[ "$(wc -l <"${stage_root}/READY")" -eq 3 ]] || return 1

  [[ -d "${stage_root}/source" && ! -L "${stage_root}/source" ]] || return 1
  _fastwam_erdma_require_regular_file "${stage_root}/source/SHA256SUMS" || return 1
  _fastwam_erdma_require_regular_file "${stage_root}/source/manifest.json" || return 1
  _fastwam_erdma_require_regular_file "${stage_root}/source/COMPLETE" || return 1
  [[ "$(_fastwam_erdma_sha256 "${stage_root}/source/SHA256SUMS")" == \
    "${expected_bundle_sha256}" ]] || return 1
  [[ "$(_fastwam_erdma_sha256 "${stage_root}/source/manifest.json")" == \
    "${expected_manifest_sha256}" ]] || return 1
  [[ "$(_fastwam_erdma_sha256 "${stage_root}/source/COMPLETE")" == \
    275a3885a5ef56e284d29dfe3ad7f21cfa6430bac9dbe2df339b6605d8568240 ]] || return 1
  (
    cd -- "${stage_root}/source"
    sha256sum --check --strict SHA256SUMS >/dev/null
  ) || return 1

  [[ "$(_fastwam_erdma_sha256 "${stage_root}/root/etc/libibverbs.d/erdma.driver")" == \
    c416bd378f751e60c4c54909813368e0cee2aa7fb8a5ada9b0c563ca492f016d ]] || return 1
  [[ "$(_fastwam_erdma_sha256 "${stage_root}/root/usr/lib/x86_64-linux-gnu/liberdma.so.1.0.56.2")" == \
    3ebeb0e7ff25fbe0eefa75d6820ab36ebab6a3ef08ff9bee4da5a01c895d6836 ]] || return 1
  [[ "$(_fastwam_erdma_sha256 "${stage_root}/root/usr/lib/x86_64-linux-gnu/libibverbs.so.1.14.56.2")" == \
    ca367ecfa7eb5553bce845bbaa93049feaead7e40e3de19c15969aecc09c5f92 ]] || return 1
  [[ "$(_fastwam_erdma_sha256 "${stage_root}/root/usr/bin/ibv_devinfo")" == \
    c22b072e3f8bf18cad8d9f5dd1eaa1344f8bb1e003a0a89dad9b7d2205d95560 ]] || return 1
  [[ "$(_fastwam_erdma_sha256 "${stage_root}/root/usr/bin/ibv_devices")" == \
    d4544d1945f24c8271f34034f72de65e3927dceb0158f0cea59f7a4c842250ff ]] || return 1
  [[ "$(_fastwam_erdma_sha256 "${stage_root}/root/usr/lib/x86_64-linux-gnu/librdmacm.so.1.3.56.2")" == \
    3ee9f9a53d5029d302abb4caa5ab96cdd3a5e543de161850a09a3a8d129662a2 ]] || return 1

  [[ "$(readlink "${stage_root}/root/usr/lib/x86_64-linux-gnu/libibverbs/liberdma-rdmav34.so")" == \
    ../liberdma.so.1.0.56.2 ]] || return 1
  [[ "$(readlink "${stage_root}/root/usr/lib/x86_64-linux-gnu/liberdma.so.1")" == \
    liberdma.so.1.0.56.2 ]] || return 1
}

_fastwam_erdma_materialize() (
  set -euo pipefail

  local bundle_root="$1"
  local stage_root="$2"
  local expected_version="$3"
  local expected_bundle_sha256="$4"
  local expected_manifest_sha256="$5"
  local lock_path="${stage_root}.lock"
  local partial_root=""
  local lock_fd
  local package_file
  local package_name
  local source_root
  local observed_name
  local observed_version
  local observed_architecture

  exec {lock_fd}>"${lock_path}"
  flock --exclusive --timeout 120 "${lock_fd}"

  if _fastwam_erdma_validate_stage \
    "${stage_root}" \
    "${expected_version}" \
    "${expected_bundle_sha256}" \
    "${expected_manifest_sha256}"; then
    flock --unlock "${lock_fd}"
    exec {lock_fd}>&-
    return 0
  fi
  if [[ -e "${stage_root}" || -L "${stage_root}" ]]; then
    _fastwam_erdma_fail "invalid_existing_stage:${stage_root}"
    return 1
  fi

  partial_root="$(mktemp -d "/tmp/.fastwam-erdma-${expected_version}.partial.XXXXXX")"
  trap 'rm -rf -- "${partial_root}"' EXIT
  mkdir -p -- "${partial_root}/root" "${partial_root}/source"
  cp -p -- \
    "${bundle_root}/SHA256SUMS" \
    "${bundle_root}/manifest.json" \
    "${bundle_root}/COMPLETE" \
    "${bundle_root}"/*.deb \
    "${partial_root}/source/"
  source_root="${partial_root}/source"
  [[ "$(_fastwam_erdma_sha256 "${source_root}/SHA256SUMS")" == \
    "${expected_bundle_sha256}" ]]
  [[ "$(_fastwam_erdma_sha256 "${source_root}/manifest.json")" == \
    "${expected_manifest_sha256}" ]]
  [[ "$(_fastwam_erdma_sha256 "${source_root}/COMPLETE")" == \
    275a3885a5ef56e284d29dfe3ad7f21cfa6430bac9dbe2df339b6605d8568240 ]]
  (
    cd -- "${source_root}"
    sha256sum --check --strict SHA256SUMS
  )

  while read -r package_name package_file; do
    observed_name="$(dpkg-deb -f "${source_root}/${package_file}" Package)"
    observed_version="$(dpkg-deb -f "${source_root}/${package_file}" Version)"
    observed_architecture="$(dpkg-deb -f "${source_root}/${package_file}" Architecture)"
    [[ "${observed_name}" == "${package_name}" ]] || \
      _fastwam_erdma_fail "package_name:${package_file}:${observed_name}"
    [[ "${observed_version}" == "${expected_version}" ]] || \
      _fastwam_erdma_fail "package_version:${package_file}:${observed_version}"
    [[ "${observed_architecture}" == amd64 ]] || \
      _fastwam_erdma_fail "package_architecture:${package_file}:${observed_architecture}"
    dpkg-deb --extract "${source_root}/${package_file}" "${partial_root}/root"
  done <<'PACKAGES'
ibverbs-providers ibverbs-providers_56.2-1.0.3_amd64.deb
ibverbs-utils ibverbs-utils_56.2-1.0.3_amd64.deb
libibverbs1 libibverbs1_56.2-1.0.3_amd64.deb
librdmacm1 librdmacm1_56.2-1.0.3_amd64.deb
PACKAGES

  printf '%s\n' \
    "provider_version=${expected_version}" \
    "bundle_sha256=${expected_bundle_sha256}" \
    "source_manifest_sha256=${expected_manifest_sha256}" \
    >"${partial_root}/READY"

  _fastwam_erdma_validate_stage \
    "${partial_root}" \
    "${expected_version}" \
    "${expected_bundle_sha256}" \
    "${expected_manifest_sha256}"
  mv -T -- "${partial_root}" "${stage_root}"
  partial_root=""
  trap - EXIT
  flock --unlock "${lock_fd}"
  exec {lock_fd}>&-
)

fastwam_prepare_erdma_userspace() {
  local built_in_version=56.2-1.0.3
  local built_in_bundle_sha256=8f2c1c43d64a7745bea19bfe4cd1383344c9cf32779166f4aa67809ebf1f5fab
  local built_in_manifest_sha256=f05443faa27533274ae1b322723e21ac09bd80bd5b2513638dd2619c67552215
  local expected_version="${FASTWAM_ERDMA_EXPECTED_VERSION:-${built_in_version}}"
  local expected_bundle_sha256="${FASTWAM_ERDMA_EXPECTED_BUNDLE_SHA256:-${built_in_bundle_sha256}}"
  local expected_manifest_sha256="${FASTWAM_ERDMA_EXPECTED_SOURCE_MANIFEST_SHA256:-${built_in_manifest_sha256}}"
  local configured_bundle_root="${FASTWAM_ERDMA_BUNDLE_ROOT:-/oss-chengjuntao/artifacts/erdma-userspace-56.2-1.0.3}"
  local bundle_root
  local sums_path
  local manifest_path
  local complete_path
  local observed_bundle_sha256
  local observed_manifest_sha256
  local stage_root
  local provider_root
  local library_dir
  local provider_dir
  local bin_dir
  local local_source_root
  local env_contract
  local probe_dir
  local device_count

  [[ "${expected_version}" == "${built_in_version}" ]] || \
    _fastwam_erdma_fail "expected_version:${expected_version}" || return 1
  [[ "${expected_bundle_sha256}" == "${built_in_bundle_sha256}" ]] || \
    _fastwam_erdma_fail "expected_bundle_sha256:${expected_bundle_sha256}" || return 1
  [[ "${expected_manifest_sha256}" == "${built_in_manifest_sha256}" ]] || \
    _fastwam_erdma_fail "expected_source_manifest_sha256:${expected_manifest_sha256}" || return 1

  _fastwam_erdma_require_command awk || return 1
  _fastwam_erdma_require_command dpkg-deb || return 1
  _fastwam_erdma_require_command flock || return 1
  _fastwam_erdma_require_command grep || return 1
  _fastwam_erdma_require_command realpath || return 1
  _fastwam_erdma_require_command sha256sum || return 1

  [[ -d "${configured_bundle_root}" && ! -L "${configured_bundle_root}" ]] || \
    _fastwam_erdma_fail "invalid_bundle_root:${configured_bundle_root}" || return 1
  bundle_root="$(realpath -e -- "${configured_bundle_root}")" || return 1
  case "${bundle_root}" in
    /cpfs/user/chengjuntao/artifacts/erdma-userspace-56.2-1.0.3 | \
      /oss-chengjuntao/artifacts/erdma-userspace-56.2-1.0.3) ;;
    *) _fastwam_erdma_fail "unapproved_bundle_root:${bundle_root}"; return 1 ;;
  esac

  sums_path="${bundle_root}/SHA256SUMS"
  manifest_path="${bundle_root}/manifest.json"
  complete_path="${bundle_root}/COMPLETE"
  _fastwam_erdma_require_regular_file "${sums_path}" || return 1
  _fastwam_erdma_require_regular_file "${manifest_path}" || return 1
  _fastwam_erdma_require_regular_file "${complete_path}" || return 1

  observed_bundle_sha256="$(_fastwam_erdma_sha256 "${sums_path}")" || return 1
  observed_manifest_sha256="$(_fastwam_erdma_sha256 "${manifest_path}")" || return 1
  [[ "${observed_bundle_sha256}" == "${expected_bundle_sha256}" ]] || \
    _fastwam_erdma_fail "bundle_sha256:${observed_bundle_sha256}" || return 1
  [[ "${observed_manifest_sha256}" == "${expected_manifest_sha256}" ]] || \
    _fastwam_erdma_fail "source_manifest_sha256:${observed_manifest_sha256}" || return 1
  grep -Fxq "artifact_id=erdma-userspace-${expected_version}" "${complete_path}" || \
    _fastwam_erdma_fail "complete_artifact_id" || return 1
  grep -Fxq "bundle_sha256=${expected_bundle_sha256}" "${complete_path}" || \
    _fastwam_erdma_fail "complete_bundle_sha256" || return 1
  grep -Fxq "source_manifest_sha256=${expected_manifest_sha256}" "${complete_path}" || \
    _fastwam_erdma_fail "complete_source_manifest_sha256" || return 1
  [[ "$(wc -l <"${complete_path}")" -eq 3 ]] || \
    _fastwam_erdma_fail "complete_line_count" || return 1

  (
    cd -- "${bundle_root}"
    sha256sum --check --strict SHA256SUMS
  ) || _fastwam_erdma_fail "package_sha256" || return 1

  while read -r _ package_file; do
    _fastwam_erdma_require_regular_file "${bundle_root}/${package_file}" || return 1
  done <<'PACKAGES'
c20665d5a80773edbf3e31618376249f03342ea76c3eb3d61ca31a8255214548 ibverbs-providers_56.2-1.0.3_amd64.deb
6d2c9a3d5336d9116e74b539f85b972cb00f00ff60c7e647eff2660213a6fc5e ibverbs-utils_56.2-1.0.3_amd64.deb
a91dbb8eca1d911c2eaddd8f801de1a761c2c3cbc6ce5fd1370cba2597a3d5ec libibverbs1_56.2-1.0.3_amd64.deb
a6ebb5ed896acef9df72b68ca77221d30f63d5574fdac4d65e23b03bd93bdd01 librdmacm1_56.2-1.0.3_amd64.deb
PACKAGES

  stage_root="/tmp/fastwam-erdma-userspace-${expected_version}-${expected_bundle_sha256:0:12}"
  _fastwam_erdma_materialize \
    "${bundle_root}" \
    "${stage_root}" \
    "${expected_version}" \
    "${expected_bundle_sha256}" \
    "${expected_manifest_sha256}" || return 1

  _fastwam_erdma_validate_stage \
    "${stage_root}" \
    "${expected_version}" \
    "${expected_bundle_sha256}" \
    "${expected_manifest_sha256}" || \
    _fastwam_erdma_fail "stage_post_validation" || return 1

  provider_root="${stage_root}/root"
  library_dir="${provider_root}/usr/lib/x86_64-linux-gnu"
  provider_dir="${library_dir}/libibverbs"
  bin_dir="${provider_root}/usr/bin"
  local_source_root="${stage_root}/source"

  export FASTWAM_ERDMA_VERSION="${expected_version}"
  export FASTWAM_ERDMA_BUNDLE_ROOT_RESOLVED="${bundle_root}"
  export FASTWAM_ERDMA_BUNDLE_SHA256="${expected_bundle_sha256}"
  export FASTWAM_ERDMA_SOURCE_MANIFEST_SHA256="${expected_manifest_sha256}"
  export FASTWAM_ERDMA_PROVIDER_ROOT="${provider_root}"
  export FASTWAM_ERDMA_LIBRARY_DIR="${library_dir}"
  export FASTWAM_ERDMA_PROVIDER_DIR="${provider_dir}"
  export FASTWAM_ERDMA_BIN_DIR="${bin_dir}"
  export FASTWAM_ERDMA_LOCAL_SOURCE_ROOT="${local_source_root}"
  export IBV_CONFIG_DIR="${provider_root}/etc/libibverbs.d"
  export IBV_DRIVERS=erdma
  export RDMAV_DRIVERS=erdma
  _fastwam_erdma_prepend_path LD_LIBRARY_PATH "${provider_dir}"
  _fastwam_erdma_prepend_path LD_LIBRARY_PATH "${library_dir}"
  _fastwam_erdma_prepend_path PATH "${bin_dir}"

  env_contract="$(printf '%s\n' \
    "FASTWAM_ERDMA_VERSION=${FASTWAM_ERDMA_VERSION}" \
    "FASTWAM_ERDMA_BUNDLE_SHA256=${FASTWAM_ERDMA_BUNDLE_SHA256}" \
    "FASTWAM_ERDMA_SOURCE_MANIFEST_SHA256=${FASTWAM_ERDMA_SOURCE_MANIFEST_SHA256}" \
    "FASTWAM_ERDMA_PROVIDER_ROOT=${FASTWAM_ERDMA_PROVIDER_ROOT}" \
    "FASTWAM_ERDMA_LOCAL_SOURCE_ROOT=${FASTWAM_ERDMA_LOCAL_SOURCE_ROOT}" \
    "IBV_CONFIG_DIR=${IBV_CONFIG_DIR}" \
    "IBV_DRIVERS=${IBV_DRIVERS}" \
    "RDMAV_DRIVERS=${RDMAV_DRIVERS}")"
  FASTWAM_ERDMA_ENV_SHA256="$(printf '%s\n' "${env_contract}" | sha256sum | awk '{print $1}')"
  export FASTWAM_ERDMA_ENV_SHA256

  probe_dir="$(mktemp -d /tmp/fastwam-erdma-provider-probe.XXXXXX)" || return 1
  if ! ibv_devices >"${probe_dir}/ibv_devices.txt" 2>&1; then
    rm -rf -- "${probe_dir}"
    _fastwam_erdma_fail "ibv_devices" || return 1
  fi
  if ! ibv_devinfo >"${probe_dir}/ibv_devinfo.txt" 2>&1; then
    rm -rf -- "${probe_dir}"
    _fastwam_erdma_fail "ibv_devinfo" || return 1
  fi
  device_count="$(awk '$1 ~ /^erdma_[0-9]+$/ {count++} END {print count+0}' "${probe_dir}/ibv_devices.txt")"
  if ((device_count < 1)); then
    rm -rf -- "${probe_dir}"
    _fastwam_erdma_fail "no_erdma_device" || return 1
  fi
  if grep -Eq 'state:[[:space:]]+PORT_ACTIVE' "${probe_dir}/ibv_devinfo.txt"; then
    :
  else
    rm -rf -- "${probe_dir}"
    _fastwam_erdma_fail "no_active_erdma_port" || return 1
  fi
  rm -rf -- "${probe_dir}"

  echo "ERDMA_PREPARE=PASS version=${FASTWAM_ERDMA_VERSION} bundle_sha256=${FASTWAM_ERDMA_BUNDLE_SHA256} source_manifest_sha256=${FASTWAM_ERDMA_SOURCE_MANIFEST_SHA256} env_sha256=${FASTWAM_ERDMA_ENV_SHA256} device_count=${device_count} provider_root=${FASTWAM_ERDMA_PROVIDER_ROOT}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This helper must be sourced so its exports reach NCCL:" >&2
  echo "  source docker/prepare-erdma-userspace.sh" >&2
  echo "  fastwam_prepare_erdma_userspace" >&2
  exit 2
fi
