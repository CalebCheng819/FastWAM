#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONTROL_PYTHON="/mnt/workspace/tools/pai-control-py312/20260717-credentials1.0.10-dlc1.9.2-aiworkspace8.2.0/bin/python"
CONTROL_PYTHON_TARGET="/usr/local/bin/python3.12"
LOCK_DIR="/tmp/fastwam-dlc-submit-state/workspace-270969"
# This lock is only the SSH970 local collision guard.  The new R5 experiment's
# permanent OSS O_EXCL latch remains the cross-node single-mutation boundary.
LOCK_FILE="${LOCK_DIR}/.gate2-nohash-r5-submit.lock"

if [[ -z "${SSH_CONNECTION:-}" ]]; then
  echo "Error: invoke this entrypoint inside the SSH970 session." >&2
  exit 2
fi
if [[ ! -L "${CONTROL_PYTHON}" || ! -x "${CONTROL_PYTHON}" ]]; then
  echo "Error: pinned SSH970 control Python is unavailable." >&2
  exit 2
fi
CONTROL_PYTHON_REAL="$(realpath -e -- "${CONTROL_PYTHON}")"
if [[ "${CONTROL_PYTHON_REAL}" != "${CONTROL_PYTHON_TARGET}" ]]; then
  echo "Error: pinned SSH970 control Python resolves to an unexpected target." >&2
  exit 2
fi
if [[ ! -f "${CONTROL_PYTHON_TARGET}" || ! -x "${CONTROL_PYTHON_TARGET}" \
   || -L "${CONTROL_PYTHON_TARGET}" ]]; then
  echo "Error: pinned SSH970 control Python target is not a regular executable." >&2
  exit 2
fi
if ! "${CONTROL_PYTHON}" -B -I -c \
  'import alibabacloud_credentials,alibabacloud_pai_dlc20201203,alibabacloud_tea_openapi' \
  >/dev/null 2>&1; then
  echo "Error: pinned SSH970 control Python lacks the required Alibaba Cloud SDK." >&2
  exit 2
fi
install -d -m 0700 -- "${LOCK_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Error: another SSH970 Gate2-R5 controller holds the single-writer lock." >&2
  exit 75
fi

export FASTWAM_CONTROL_NODE=ssh970
export FASTWAM_LOCK_FD=9
export PYTHONDONTWRITEBYTECODE=1
exec "${CONTROL_PYTHON}" -B -I "${SCRIPT_DIR}/submit_gate2_r5.py" "$@"
