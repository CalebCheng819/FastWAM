#!/usr/bin/env bash
set -euo pipefail

context="${1:-dlc}"
expected_version="${FASTWAM_ERDMA_USERSPACE_VERSION:-56.2-1.0.3}"
provider_root="${FASTWAM_ERDMA_PROVIDER_ROOT:-}"
provider_mode=installed

case "${context}" in
  dlc) expected_socket_interface=eth0 ;;
  dsw) expected_socket_interface=eth1 ;;
  *)
    echo "Usage: fastwam-verify-erdma-userspace [dlc|dsw]" >&2
    exit 2
    ;;
esac

fail() {
  echo "ERDMA_USERSPACE_GATE=FAIL reason=$1" >&2
  exit 1
}

require_file() {
  [[ -e "$1" ]] || fail "missing:$1"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "missing_command:$1"
}

require_env_equals() {
  local name="$1"
  local expected="$2"
  local observed="${!name:-}"
  [[ "${observed}" == "${expected}" ]] || \
    fail "${name}:expected=${expected}:observed=${observed:-unset}"
}

require_command dpkg-query
require_command ibv_devices
require_command ibv_devinfo
require_command ip
require_command rdma

if [[ -n "${provider_root}" ]]; then
  provider_mode=overlay
  [[ -d "${provider_root}" && ! -L "${provider_root}" ]] || \
    fail "invalid_provider_root:${provider_root}"
  installed_version="${FASTWAM_ERDMA_VERSION:-}"
  [[ "${FASTWAM_ERDMA_BUNDLE_SHA256:-}" == \
    8f2c1c43d64a7745bea19bfe4cd1383344c9cf32779166f4aa67809ebf1f5fab ]] || \
    fail "overlay_bundle_sha256"
  [[ "${FASTWAM_ERDMA_SOURCE_MANIFEST_SHA256:-}" == \
    f05443faa27533274ae1b322723e21ac09bd80bd5b2513638dd2619c67552215 ]] || \
    fail "overlay_source_manifest_sha256"
else
  provider_root=""
  installed_version="$(dpkg-query -W -f='${Version}' ibverbs-providers 2>/dev/null)" || \
    fail "ibverbs-providers:not_installed"
fi
[[ "${installed_version}" == "${expected_version}" ]] || \
  fail "ibverbs-providers:expected=${expected_version}:observed=${installed_version}"

require_file "${provider_root}/etc/libibverbs.d/erdma.driver"
require_file "${provider_root}/usr/lib/x86_64-linux-gnu/libibverbs/liberdma-rdmav34.so"
require_file "${provider_root}/usr/lib/x86_64-linux-gnu/liberdma.so.1"

shopt -s nullglob
devices=(/sys/class/infiniband/erdma_*)
uverbs=(/dev/infiniband/uverbs*)
(( ${#devices[@]} > 0 )) || fail "no_erdma_hca"
(( ${#uverbs[@]} >= ${#devices[@]} )) || \
  fail "uverbs_count=${#uverbs[@]}:hca_count=${#devices[@]}"

ip link show "${expected_socket_interface}" >/dev/null 2>&1 || \
  fail "missing_socket_interface:${expected_socket_interface}"

# PAI injects these defaults for eRDMA-capable DLC resources.  Verify rather
# than overwrite them so a scheduler/image mismatch fails before training.
require_env_equals NCCL_SOCKET_IFNAME "${expected_socket_interface}"
require_env_equals NCCL_IB_GID_INDEX 1
require_env_equals NCCL_IB_HCA erdma
require_env_equals NCCL_IB_QPS_PER_CONNECTION 8
require_env_equals NCCL_MIN_NCHANNELS 16
require_env_equals NCCL_NET_PLUGIN none
[[ "${NCCL_IB_DISABLE:-0}" != 1 ]] || fail "NCCL_IB_DISABLE=1"

probe_dir="$(mktemp -d /tmp/fastwam-erdma-userspace.XXXXXX)"
trap 'rm -rf -- "${probe_dir}"' EXIT
ibv_devices >"${probe_dir}/ibv_devices.txt" 2>&1 || \
  fail "ibv_devices_failed"
ibv_devinfo >"${probe_dir}/ibv_devinfo.txt" 2>&1 || \
  fail "ibv_devinfo_failed"
rdma link show >"${probe_dir}/rdma_link.txt" 2>&1 || \
  fail "rdma_link_failed"

for device_path in "${devices[@]}"; do
  device_name="${device_path##*/}"
  grep -Eq "^[[:space:]]*${device_name}[[:space:]]" "${probe_dir}/ibv_devices.txt" || \
    fail "provider_did_not_open:${device_name}"
  awk -v hca="${device_name}" '
    $1 == "hca_id:" { in_hca = ($2 == hca) }
    in_hca && /state:[[:space:]]+PORT_ACTIVE/ { active = 1 }
    END { exit(active ? 0 : 1) }
  ' "${probe_dir}/ibv_devinfo.txt" || fail "port_not_active:${device_name}"
  grep -Eq "link[[:space:]]+${device_name}/[0-9]+[[:space:]]+state[[:space:]]+ACTIVE" \
    "${probe_dir}/rdma_link.txt" || fail "rdma_link_not_active:${device_name}"
done

python_bin="${FASTWAM_PYTHON:-}"
if [[ -z "${python_bin}" ]]; then
  python_bin="$(command -v python || command -v python3 || true)"
fi
[[ -n "${python_bin}" ]] || fail "python_not_found"

"${python_bin}" - <<'PY' || exit $?
import json
import re
import sys

try:
    import torch
except Exception as error:
    print(f"ERDMA_USERSPACE_GATE=FAIL reason=torch_import:{type(error).__name__}", file=sys.stderr)
    raise SystemExit(1)

cuda_match = re.match(r"^(\d+)\.(\d+)", str(torch.version.cuda or ""))
if not cuda_match or tuple(map(int, cuda_match.groups())) < (12, 1):
    print(f"ERDMA_USERSPACE_GATE=FAIL reason=cuda_version:{torch.version.cuda}", file=sys.stderr)
    raise SystemExit(1)

try:
    nccl = tuple(torch.cuda.nccl.version())
except Exception as error:
    print(f"ERDMA_USERSPACE_GATE=FAIL reason=nccl_probe:{type(error).__name__}", file=sys.stderr)
    raise SystemExit(1)
if nccl < (2, 19):
    print(f"ERDMA_USERSPACE_GATE=FAIL reason=nccl_version:{nccl}", file=sys.stderr)
    raise SystemExit(1)

print(
    "ERDMA_FRAMEWORK_GATE=PASS "
    + json.dumps(
        {
            "cuda": torch.version.cuda,
            "nccl": list(nccl),
            "torch": torch.__version__,
        },
        sort_keys=True,
    )
)
PY

if command -v ib_write_bw >/dev/null 2>&1; then
  perftest_available=true
else
  perftest_available=false
fi
echo "ERDMA_USERSPACE_GATE=PASS context=${context} hca_count=${#devices[@]} provider=${installed_version} provider_mode=${provider_mode} socket_interface=${expected_socket_interface} perftest_available=${perftest_available}"
