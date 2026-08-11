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
CONTROL_PYTHON_TARGET=/usr/local/bin/python3.12
[[ -L "${CONTROL_PYTHON}" && -x "${CONTROL_PYTHON}" ]] || {
  echo "Error: pinned PAI control Python is unavailable" >&2
  exit 1
}
[[ "$(realpath -e -- "${CONTROL_PYTHON}")" == "${CONTROL_PYTHON_TARGET}" ]] || {
  echo "Error: pinned PAI control Python resolves to an unexpected target" >&2
  exit 1
}
[[ -f "${CONTROL_PYTHON_TARGET}" && -x "${CONTROL_PYTHON_TARGET}" \
   && ! -L "${CONTROL_PYTHON_TARGET}" ]] || {
  echo "Error: pinned PAI control Python target is not a regular executable" >&2
  exit 1
}
"${CONTROL_PYTHON}" -I -c \
  'import alibabacloud_credentials,alibabacloud_pai_dlc20201203,alibabacloud_tea_openapi,alibabacloud_tea_util' \
  || { echo "Error: pinned PAI SDK environment is incomplete" >&2; exit 1; }
LOCK_ANCHOR="/run"
LOCK_ROOT="/run/fastwam-dlc-submit-state/workspace-270969"
LOCK_NAME="action-n234-formal-r4-controller.lock"

# The pinned interpreter opens each component relative to an already validated
# directory descriptor.  In particular, the lock is never opened by shell
# redirection: no symlink is followed and no existing file is truncated before
# its identity has been checked.  Descriptor 9 retains the flock across exec.
exec "${CONTROL_PYTHON}" -B -I -S - \
  "${LOCK_ANCHOR}" "${LOCK_ROOT}" "${LOCK_NAME}" "${SCRIPT_DIR}/controller.py" \
  "${CONTROL_PYTHON}" "$@" <<'PY'
import errno
import fcntl
import os
import stat
import sys


lock_anchor, lock_root, lock_name, controller, control_python, *controller_args = sys.argv[1:]
expected_lock_fd = 9


def fail(message):
    raise RuntimeError(message)


def identity(metadata):
    return metadata.st_dev, metadata.st_ino


def validate_anchor(fd):
    opened = os.fstat(fd)
    named = os.lstat(lock_anchor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or identity(opened) != identity(named)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) & 0o022
    ):
        fail("unsafe controller lock anchor")


def validate_private_directory(parent_fd, name, fd):
    opened = os.fstat(fd)
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or identity(opened) != identity(named)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o700
    ):
        fail(f"unsafe controller lock directory component: {name}")


def validate_lock(parent_fd, fd):
    opened = os.fstat(fd)
    named = os.stat(lock_name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or identity(opened) != identity(named)
        or opened.st_nlink != 1
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o600
    ):
        fail("unsafe R4 controller lock file")


normal_anchor = os.path.normpath(lock_anchor)
normal_root = os.path.normpath(lock_root)
if (
    not os.path.isabs(lock_anchor)
    or normal_anchor != lock_anchor
    or not os.path.isabs(lock_root)
    or normal_root != lock_root
    or os.path.commonpath((lock_anchor, lock_root)) != lock_anchor
    or lock_root == lock_anchor
    or "/../" in f"{lock_root}/"
):
    fail("controller lock root must be a canonical child of the lock anchor")
components = os.path.relpath(lock_root, lock_anchor).split(os.sep)
if not components or any(part in ("", ".", "..") for part in components):
    fail("controller lock root contains an invalid component")
if not lock_name or os.sep in lock_name or lock_name in (".", ".."):
    fail("controller lock name is invalid")

directory_flags = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | os.O_NOFOLLOW
    | os.O_CLOEXEC
)
anchor_fd = os.open(lock_anchor, directory_flags)
directory_fds = [anchor_fd]
directory_edges = []
try:
    validate_anchor(anchor_fd)
    parent_fd = anchor_fd
    for component in components:
        try:
            os.mkdir(component, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
        directory_fds.append(child_fd)
        directory_edges.append((parent_fd, component, child_fd))
        validate_private_directory(parent_fd, component, child_fd)
        parent_fd = child_fd

    lock_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
    )
    lock_fd = os.open(lock_name, lock_flags, 0o600, dir_fd=parent_fd)
    try:
        validate_lock(parent_fd, lock_fd)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN):
                fail("another formal R4 controller is active")
            raise

        # LOCK_TEST_REVALIDATION_POINT
        validate_anchor(anchor_fd)
        for edge in directory_edges:
            validate_private_directory(*edge)
        validate_lock(parent_fd, lock_fd)
    except BaseException:
        os.close(lock_fd)
        raise
finally:
    for directory_fd in reversed(directory_fds):
        os.close(directory_fd)

if lock_fd != expected_lock_fd:
    os.dup2(lock_fd, expected_lock_fd, inheritable=True)
    os.close(lock_fd)
else:
    os.set_inheritable(expected_lock_fd, True)

environment = os.environ.copy()
environment["FASTWAM_CONTROL_NODE"] = "ssh970"
environment["FASTWAM_LOCK_FD"] = str(expected_lock_fd)
environment["PYTHONDONTWRITEBYTECODE"] = "1"
os.execve(
    control_python,
    [control_python, "-B", "-I", controller, *controller_args],
    environment,
)
PY
