#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
[[ -n "${SSH_CONNECTION:-}" ]] || {
  echo "Error: controller must be invoked from an SSH session on ssh970" >&2
  exit 1
}
read -r ssh_client ssh_client_port ssh_server ssh_server_port ssh_extra <<<"${SSH_CONNECTION}"
[[ -n "${ssh_client}" && "${ssh_client_port}" =~ ^[0-9]+$ && -n "${ssh_server}" \
   && "${ssh_server_port}" =~ ^[0-9]+$ && -z "${ssh_extra:-}" ]] || {
  echo "Error: SSH_CONNECTION has an invalid shape" >&2
  exit 1
}
CONTROL_PYTHON=/mnt/workspace/tools/pai-control-py312/20260717-credentials1.0.10-dlc1.9.2-aiworkspace8.2.0/bin/python
[[ -x "${CONTROL_PYTHON}" && ! -L "${CONTROL_PYTHON}" ]] || {
  echo "Error: pinned PAI control Python is unavailable" >&2
  exit 1
}
[[ "$(realpath -e -- "${CONTROL_PYTHON}")" == "${CONTROL_PYTHON}" ]] || {
  echo "Error: pinned PAI control Python does not resolve to its frozen path" >&2
  exit 1
}
"${CONTROL_PYTHON}" -I -c \
  'import alibabacloud_credentials,alibabacloud_pai_dlc20201203,alibabacloud_tea_openapi,alibabacloud_tea_util' \
  || { echo "Error: pinned PAI SDK environment is incomplete" >&2; exit 1; }
LOCK_ROOT="/tmp/fastwam-dlc-submit-state/workspace-270969"
mkdir -p -m 0700 "${LOCK_ROOT}"
exec 9>"${LOCK_ROOT}/action-n234-formal-controller.lock"
flock -n 9 || { echo "Error: another formal controller is active" >&2; exit 1; }
export FASTWAM_CONTROL_NODE=ssh970
export FASTWAM_LOCK_FD=9
exec "${CONTROL_PYTHON}" -I "${SCRIPT_DIR}/controller.py" "$@"
