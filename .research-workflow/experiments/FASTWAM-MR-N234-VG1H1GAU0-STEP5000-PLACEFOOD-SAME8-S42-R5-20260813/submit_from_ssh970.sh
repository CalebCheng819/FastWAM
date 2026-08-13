#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

EXPERIMENT_REL='.research-workflow/experiments/FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R5-20260813'
CONTROL_PYTHON='/mnt/workspace/tools/pai-control-py312/20260717-credentials1.0.10-dlc1.9.2-aiworkspace8.2.0/bin/python'
CONTROL_PYTHON_LINK_TARGET='python3'
CONTROL_PYTHON_RESOLVED_TARGET='/usr/local/bin/python3.12'
LOCK_ANCHOR='/run'
LOCK_PARENT='fastwam-dlc-submit-state'
LOCK_WORKSPACE='workspace-270969'
LOCK_NAME='gau0-placefood-same8-r5-controller.lock'

die() {
  printf 'GAU0_WRAPPER_FATAL: %s\n' "$*" >&2
  exit 1
}

[[ -n "${SSH_CONNECTION:-}" ]] || die 'must be invoked through SSH'
read -r -a ssh_fields <<<"${SSH_CONNECTION}"
[[ "${#ssh_fields[@]}" == '4' ]] || die 'malformed SSH connection metadata'
[[ "${ssh_fields[1]}" =~ ^[0-9]+$ && "${ssh_fields[3]}" =~ ^[0-9]+$ ]] \
  || die 'malformed SSH connection ports'
[[ "$(id -u)" == '0' ]] || die 'must run as root'
[[ -L "${CONTROL_PYTHON}" ]] || die 'control Python is not a symlink'
[[ "$(readlink -- "${CONTROL_PYTHON}")" == "${CONTROL_PYTHON_LINK_TARGET}" ]] || die 'control Python link target changed'
[[ "$(readlink -f -- "${CONTROL_PYTHON}")" == "${CONTROL_PYTHON_RESOLVED_TARGET}" ]] || die 'control Python resolved target changed'
[[ -x "${CONTROL_PYTHON_RESOLVED_TARGET}" ]] || die 'control Python target is not executable'
"${CONTROL_PYTHON}" -B -I -c 'import alibabacloud_credentials, alibabacloud_pai_dlc20201203, alibabacloud_tea_openapi, alibabacloud_tea_util' \
  || die 'pinned control SDK imports failed'

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_root="$(cd -- "${script_dir}/../../.." && pwd -P)"
[[ "${source_root}/${EXPERIMENT_REL}/submit_from_ssh970.sh" == "$(readlink -f -- "${BASH_SOURCE[0]}")" ]] \
  || die 'wrapper/source-root relation changed'
controller="${source_root}/${EXPERIMENT_REL}/controller.py"
[[ -f "${controller}" && ! -L "${controller}" ]] || die 'controller must be an ordinary file'

# The bootstrap traverses /run by directory descriptor, creates only the two
# frozen private directories when absent, opens the lock without following or
# truncating links, acquires it, and then passes the locked descriptor as fd 9.
exec "${CONTROL_PYTHON}" -B -I -S - \
  "${LOCK_ANCHOR}" "${LOCK_PARENT}" "${LOCK_WORKSPACE}" "${LOCK_NAME}" \
  "${CONTROL_PYTHON}" "${controller}" "$@" <<'PY'
import fcntl
import os
import stat
import sys

anchor, parent_name, workspace_name, lock_name, control_python, controller, *command_args = sys.argv[1:]


def validate_dir(fd: int, *, mode: int, label: str) -> None:
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) != mode:
        raise SystemExit(f"unsafe {label} directory metadata")
    if info.st_mode & 0o022:
        raise SystemExit(f"{label} directory is group/other writable")


def open_private_child(parent_fd: int, name: str, label: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        return os.open(name, flags, dir_fd=parent_fd)


if anchor != "/run" or "/" in parent_name or "/" in workspace_name or "/" in lock_name:
    raise SystemExit("frozen lock namespace changed")
anchor_info = os.lstat(anchor)
if not stat.S_ISDIR(anchor_info.st_mode) or stat.S_ISLNK(anchor_info.st_mode) or anchor_info.st_uid != 0:
    raise SystemExit("unsafe /run anchor")
if stat.S_IMODE(anchor_info.st_mode) != 0o755 or anchor_info.st_mode & 0o022:
    raise SystemExit("unexpected /run permissions")

anchor_fd = os.open(anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
try:
    parent_fd = open_private_child(anchor_fd, parent_name, "control root")
finally:
    os.close(anchor_fd)
try:
    validate_dir(parent_fd, mode=0o700, label="control root")
    workspace_fd = open_private_child(parent_fd, workspace_name, "workspace root")
finally:
    os.close(parent_fd)
try:
    validate_dir(workspace_fd, mode=0o700, label="workspace root")
    lock_fd = os.open(
        lock_name,
        os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=workspace_fd,
    )
finally:
    os.close(workspace_fd)

lock_info = os.fstat(lock_fd)
if (
    not stat.S_ISREG(lock_info.st_mode)
    or lock_info.st_uid != 0
    or stat.S_IMODE(lock_info.st_mode) != 0o600
    or lock_info.st_nlink != 1
    or lock_info.st_size != 0
):
    raise SystemExit("unsafe controller lock metadata")
try:
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError as exc:
    raise SystemExit("another controller invocation holds the frozen lock") from exc

os.dup2(lock_fd, 9)
if lock_fd != 9:
    os.close(lock_fd)
os.set_inheritable(9, True)
environment = dict(os.environ)
environment["FASTWAM_CONTROL_NODE"] = "ssh970"
environment["FASTWAM_LOCK_FD"] = "9"
os.execve(
    control_python,
    [control_python, "-B", "-I", controller, *command_args],
    environment,
)
PY
