#!/usr/bin/env bash
set -euo pipefail

docker_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
prepare_helper="${docker_dir}/prepare-erdma-userspace.sh"
verify_helper="${docker_dir}/verify-erdma-userspace.sh"
sums_file="${docker_dir}/erdma-userspace-56.2-1.0.3.SHA256SUMS"
manifest_file="${docker_dir}/erdma-userspace-56.2-1.0.3.manifest.json"
complete_file="${docker_dir}/erdma-userspace-56.2-1.0.3.COMPLETE"

bash -n "${prepare_helper}"
bash -n "${verify_helper}"

[[ "$(sha256sum "${sums_file}" | awk '{print $1}')" == \
  8f2c1c43d64a7745bea19bfe4cd1383344c9cf32779166f4aa67809ebf1f5fab ]]
[[ "$(sha256sum "${manifest_file}" | awk '{print $1}')" == \
  f05443faa27533274ae1b322723e21ac09bd80bd5b2513638dd2619c67552215 ]]
[[ "$(sha256sum "${complete_file}" | awk '{print $1}')" == \
  275a3885a5ef56e284d29dfe3ad7f21cfa6430bac9dbe2df339b6605d8568240 ]]

set +e
direct_output="$(bash "${prepare_helper}" 2>&1)"
direct_status=$?
set -e
[[ "${direct_status}" -eq 2 ]]
grep -Fq 'must be sourced' <<<"${direct_output}"

(
  source "${prepare_helper}"
  PATH=/usr/bin
  _fastwam_erdma_prepend_path PATH /fixed/provider/bin
  _fastwam_erdma_prepend_path PATH /fixed/provider/bin
  [[ "${PATH}" == /fixed/provider/bin:/usr/bin ]]
)

set +e
mismatch_output="$(
  FASTWAM_ERDMA_EXPECTED_VERSION=56.2-bad
  export FASTWAM_ERDMA_EXPECTED_VERSION
  source "${prepare_helper}"
  fastwam_prepare_erdma_userspace 2>&1
)"
mismatch_status=$?
set -e
[[ "${mismatch_status}" -eq 1 ]]
grep -Fq 'ERDMA_PREPARE=FAIL reason=expected_version:56.2-bad' <<<"${mismatch_output}"

for required_name in \
  FASTWAM_ERDMA_BUNDLE_SHA256 \
  FASTWAM_ERDMA_SOURCE_MANIFEST_SHA256 \
  FASTWAM_ERDMA_ENV_SHA256 \
  FASTWAM_ERDMA_PROVIDER_ROOT \
  FASTWAM_ERDMA_PROVIDER_DIR \
  IBV_CONFIG_DIR \
  IBV_DRIVERS \
  RDMAV_DRIVERS \
  LD_LIBRARY_PATH; do
  grep -Fq "${required_name}" "${prepare_helper}"
done

if grep -Eq '\b(curl|wget|apt-get|pip)[[:space:]]' "${prepare_helper}"; then
  echo "Runtime helper must remain offline and consume only the pinned bundle." >&2
  exit 1
fi

echo "ERDMA_HELPER_STATIC_TEST=PASS"
