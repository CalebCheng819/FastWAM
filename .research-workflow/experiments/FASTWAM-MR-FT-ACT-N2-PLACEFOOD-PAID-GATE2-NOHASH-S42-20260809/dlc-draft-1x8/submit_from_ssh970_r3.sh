#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONTROL_PYTHON="/mnt/workspace/tools/pai-control-py312/20260717-credentials1.0.10-dlc1.9.2-aiworkspace8.2.0/bin/python"
LOCK_DIR="/tmp/fastwam-dlc-submit-state/workspace-270969"
# Node-local collision guard only; it is not the cross-node safety boundary.
# The experiment-scoped permanent OSS O_EXCL submission latch is authoritative.
LOCK_FILE="${LOCK_DIR}/.gate2-nohash-submit.lock"

if [[ -z "${SSH_CONNECTION:-}" ]]; then
  echo "Error: invoke this entrypoint inside the SSH970 session." >&2
  exit 2
fi
if [[ ! -x "${CONTROL_PYTHON}" ]]; then
  echo "Error: pinned SSH970 control Python is unavailable." >&2
  exit 2
fi
install -d -m 0700 -- "${LOCK_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Error: another SSH970 Gate2-R3 controller holds the single-writer lock." >&2
  exit 75
fi

export FASTWAM_CONTROL_NODE=ssh970
export FASTWAM_LOCK_FD=9
exec "${CONTROL_PYTHON}" "${SCRIPT_DIR}/submit_gate2_r3.py" "$@"
