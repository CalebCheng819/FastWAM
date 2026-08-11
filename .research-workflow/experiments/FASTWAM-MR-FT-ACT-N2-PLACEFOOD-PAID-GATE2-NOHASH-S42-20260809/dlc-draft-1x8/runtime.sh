#!/usr/bin/env bash
set -euo pipefail
umask 077

# One DLC worker only: all large mutable output stays on node-local /tmp until
# both the save world and a new load world succeed.  CPFS is input-only.
: "${FASTWAM_EXPERIMENT_ID:?}"
: "${FASTWAM_SUBMISSION_TAG:?}"
: "${FASTWAM_SOURCE_ROOT:?}"
: "${FASTWAM_PREPARED_BINDING_PATH:?}"
: "${FASTWAM_GATE2_ENTRYPOINT:?}"
: "${FASTWAM_GATE2_TRUSTED_RUNTIME_B64:?}"
: "${FASTWAM_GATE2_TRUSTED_RUNTIME_BYTES:?}"
: "${FASTWAM_OSS_OUTPUT_ROOT:?}"
: "${FASTWAM_TASK_CONFIG:?}"
: "${FASTWAM_ARTIFACT_INTEGRITY_MODE:?}"
: "${FASTWAM_PYTHON:?}"
: "${FASTWAM_PYTHON_TARGET:?}"
: "${FASTWAM_DATASET_ROOT:?}"
: "${FASTWAM_INITIAL_CHECKPOINT:?}"
: "${FASTWAM_N234_NOHASH_STATS_SOURCE:?}"
: "${FASTWAM_TEXT_CACHE_DIR:?}"
: "${FASTWAM_GAUSSIAN_CACHE_DIR:?}"
: "${FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR:?}"
: "${FASTWAM_MIN_TMP_FREE_BYTES:?}"
: "${NPROC_PER_NODE:?}"

EXPECTED_EXPERIMENT="FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-R6-S42-20260811"
SOURCE_EXPERIMENT="FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-S42-20260809"
EXPECTED_TASK="robofactory_multi_robot_ft_n2_placefood_vg0_hub1_gau1_224_3e-5_nohash_gate"
EXPECTED_DATASET_ROOT="/cpfs/user/chengjuntao/datasets/robofactory_multi_robot"
SOURCE_PREFIX="/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/"
EXPECTED_TRUSTED_RUNTIME_PATH="/tmp/fastwam-gate2-trusted-runtime.sh"
EXPECTED_VAE_SOURCE="/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/model-cache/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
EXPECTED_VAE_SOURCE_BYTES=1409401152
OUTPUT_PREFIX="/oss-chengjuntao/artifacts/fastwam-gate2-nohash-results/"
GAUSSIAN_PREFIX="/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"

fail() {
  echo "Error: $*" >&2
  exit 1
}

[[ "${FASTWAM_EXPERIMENT_ID}" == "${EXPECTED_EXPERIMENT}" ]] || fail "experiment identity mismatch"
[[ "${FASTWAM_TASK_CONFIG}" == "${EXPECTED_TASK}" ]] || fail "task config mismatch"
[[ "${FASTWAM_DATASET_ROOT}" == "${EXPECTED_DATASET_ROOT}" ]] || fail "dataset root mismatch"
[[ "${FASTWAM_ARTIFACT_INTEGRITY_MODE}" == "metadata_no_hash" ]] || fail "metadata_no_hash is mandatory"
[[ "${NPROC_PER_NODE}" == "8" ]] || fail "exactly eight local processes are required"
[[ "${WORLD_SIZE:-1}" == "1" && "${RANK:-0}" == "0" ]] || fail "DLC topology must be one worker pod"
[[ "${FASTWAM_SOURCE_ROOT}" == "${SOURCE_PREFIX}"* ]] || fail "source snapshot is outside the approved OSS prefix"
SOURCE_SNAPSHOT_NAME="${FASTWAM_SOURCE_ROOT#"${SOURCE_PREFIX}"}"
[[ -n "${SOURCE_SNAPSHOT_NAME}" && "${SOURCE_SNAPSHOT_NAME}" != */* ]] \
  || fail "source snapshot must be one unique direct child of the approved OSS prefix"
[[ "${SOURCE_SNAPSHOT_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$ ]] \
  || fail "source snapshot name is outside the portable unique-name contract"
[[ "${FASTWAM_PREPARED_BINDING_PATH}" == "/oss-chengjuntao/artifacts/fastwam-dlc-submit-ledger/workspace-270969/${EXPECTED_EXPERIMENT}/prepared-binding.json" ]] \
  || fail "prepared binding path mismatch"
[[ "${FASTWAM_OSS_OUTPUT_ROOT}" == "${OUTPUT_PREFIX}${FASTWAM_SUBMISSION_TAG}" ]] || fail "unique OSS output identity mismatch"
[[ "${FASTWAM_GAUSSIAN_CACHE_DIR}/" == "${GAUSSIAN_PREFIX}"* ]] || fail "primary Gaussian cache is outside its approved OSS prefix"
[[ "${FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR}/" == "${GAUSSIAN_PREFIX}"* ]] || fail "fallback Gaussian cache is outside its approved OSS prefix"
[[ "${FASTWAM_GAUSSIAN_CACHE_DIR}" != "${FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR}" ]] || fail "Gaussian primary and fallback roots must differ"
[[ ! -e "${FASTWAM_OSS_OUTPUT_ROOT}" && ! -L "${FASTWAM_OSS_OUTPUT_ROOT}" ]] || fail "OSS output root already exists"
[[ ! -L "${FASTWAM_SOURCE_ROOT}" && -d "${FASTWAM_SOURCE_ROOT}" ]] || fail "source snapshot is missing or linked"
[[ "$(readlink -f -- "${FASTWAM_SOURCE_ROOT}")" == "${FASTWAM_SOURCE_ROOT}" ]] \
  || fail "source snapshot path is not canonical"
[[ ! -L "${FASTWAM_INITIAL_CHECKPOINT}" && -f "${FASTWAM_INITIAL_CHECKPOINT}" ]] || fail "initial checkpoint is missing or linked"
[[ ! -L "${FASTWAM_N234_NOHASH_STATS_SOURCE}" && -f "${FASTWAM_N234_NOHASH_STATS_SOURCE}" ]] \
  || fail "normalization stats source is missing or linked"
[[ ! -L "${EXPECTED_VAE_SOURCE}" && -f "${EXPECTED_VAE_SOURCE}" ]] \
  || fail "fixed VAE source is missing or linked"
[[ "$(stat -c %s -- "${EXPECTED_VAE_SOURCE}")" == "${EXPECTED_VAE_SOURCE_BYTES}" ]] \
  || fail "fixed VAE source byte count mismatch"
[[ "${FASTWAM_PYTHON}" == "/cpfs/user/chengjuntao/venvs/fastwam-gaudp-py310-20260802/bin/python" ]] \
  || fail "pinned venv Python contract mismatch"
[[ "${FASTWAM_PYTHON_TARGET}" == "/cpfs/user/chengjuntao/runtimes/uv-python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10" ]] \
  || fail "pinned Python target contract mismatch"
[[ -L "${FASTWAM_PYTHON}" && -x "${FASTWAM_PYTHON}" ]] || fail "pinned venv Python is unavailable"
RESOLVED_PYTHON="$(readlink -f -- "${FASTWAM_PYTHON}")"
[[ "${RESOLVED_PYTHON}" == "${FASTWAM_PYTHON_TARGET}" ]] || fail "pinned venv Python target mismatch"
[[ -f "${FASTWAM_PYTHON_TARGET}" && -x "${FASTWAM_PYTHON_TARGET}" && ! -L "${FASTWAM_PYTHON_TARGET}" ]] \
  || fail "resolved pinned Python is not a regular executable"
[[ -d "${FASTWAM_DATASET_ROOT}" && -d "${FASTWAM_TEXT_CACHE_DIR}" ]] || fail "dataset or text cache is unavailable"
[[ -d "${FASTWAM_GAUSSIAN_CACHE_DIR}" && -d "${FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR}" ]] || fail "Gaussian caches are unavailable"

TRUSTED_RUNTIME_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
[[ "${TRUSTED_RUNTIME_PATH}" == "${EXPECTED_TRUSTED_RUNTIME_PATH}" ]] \
  || fail "runtime did not start from the request-carried trusted bootstrap payload"
"${FASTWAM_PYTHON}" -B -I -S - "${TRUSTED_RUNTIME_PATH}" <<'PY'
import base64
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = base64.b64decode(
    os.environ["FASTWAM_GATE2_TRUSTED_RUNTIME_B64"].encode("ascii"),
    validate=True,
)
declared = int(os.environ["FASTWAM_GATE2_TRUSTED_RUNTIME_BYTES"])
if len(payload) != declared:
    raise RuntimeError("trusted runtime byte count mismatch")
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
fd = os.open(path, flags)
try:
    before = os.fstat(fd)
    raw = bytearray()
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        raw.extend(chunk)
    after = os.fstat(fd)
finally:
    os.close(fd)
fields = lambda value: (
    value.st_dev,
    value.st_ino,
    value.st_mode,
    value.st_size,
    value.st_mtime_ns,
)
if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or fields(before) != fields(after):
    raise RuntimeError("trusted runtime is not a stable single-link regular file")
if bytes(raw) != payload:
    raise RuntimeError("executing runtime differs from request-carried trusted bytes")
PY

CPFS_OPTIONS="$(findmnt -n -o OPTIONS -T /cpfs/user/chengjuntao)"
case ",${CPFS_OPTIONS}," in
  *,ro,*) ;;
  *) fail "CPFS datasource must be mounted read-only" ;;
esac
[[ -d /oss-chengjuntao && -w /oss-chengjuntao ]] || fail "OSS datasource must be writable"

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
[[ "${GPU_COUNT}" == "8" ]] || fail "the worker does not expose exactly eight GPUs"
TMP_FREE_BYTES="$(df -PB1 /tmp | awk 'NR==2 {print $4}')"
[[ "${TMP_FREE_BYTES}" =~ ^[0-9]+$ ]] || fail "cannot determine /tmp free space"
(( TMP_FREE_BYTES >= FASTWAM_MIN_TMP_FREE_BYTES )) || fail "/tmp free space is below the declared gate"

LOCAL_ROOT="$(mktemp -d /tmp/fastwam-gate2-nohash.XXXXXXXX)"
LOCAL_SOURCE="${LOCAL_ROOT}/source"
LOCAL_INITIAL="${LOCAL_ROOT}/initial.pt"
LOCAL_STATS="${LOCAL_ROOT}/normalization-stats.json"
LOCAL_MODEL_CACHE="${LOCAL_ROOT}/model-cache"
LOCAL_VAE="${LOCAL_MODEL_CACHE}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
LOCAL_GAUSSIAN_PRIMARY="${LOCAL_ROOT}/gaussian-primary"
PUBLISH_ROOT="${LOCAL_ROOT}/publish"
SAVE_OUTPUT="${PUBLISH_ROOT}/save_world"
LOAD_OUTPUT="${PUBLISH_ROOT}/load_world"
FINAL_VERIFY_OUTPUT="${PUBLISH_ROOT}/final_verify_world"
mkdir -p -m 0700 -- \
  "${LOCAL_SOURCE}" "${LOCAL_MODEL_CACHE}" "$(dirname -- "${LOCAL_VAE}")" \
  "${LOCAL_GAUSSIAN_PRIMARY}" \
  "${PUBLISH_ROOT}" "${SAVE_OUTPUT}" "${LOAD_OUTPUT}" \
  "${FINAL_VERIFY_OUTPUT}"

validate_prepared_source_binding() {
  "${FASTWAM_PYTHON}" - \
    "${FASTWAM_PREPARED_BINDING_PATH}" \
    "${FASTWAM_SOURCE_ROOT}" \
    "${FASTWAM_EXPERIMENT_ID}" \
    "${FASTWAM_SUBMISSION_TAG}" \
    "${FASTWAM_GATE2_ENTRYPOINT}" <<'PY'
import base64
import json
import os
import stat
import sys
from pathlib import Path

binding_path = Path(sys.argv[1])
source_root_literal = sys.argv[2]
experiment_id = sys.argv[3]
submission_tag = sys.argv[4]
entrypoint = sys.argv[5]

binding_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(
    os, "O_CLOEXEC", 0
)
try:
    binding_fd = os.open(binding_path, binding_flags)
except OSError as error:
    raise RuntimeError("prepared binding is missing, linked, or unreadable") from error
try:
    before = os.fstat(binding_fd)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RuntimeError("prepared binding is not a single-link regular file")
    chunks = []
    while True:
        chunk = os.read(binding_fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(binding_fd)
    try:
        binding_path_after = binding_path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError("prepared binding path disappeared while reading") from error
finally:
    os.close(binding_fd)

binding_descriptor = lambda value: (
    value.st_dev,
    value.st_ino,
    value.st_mode,
    value.st_nlink,
    value.st_size,
    value.st_mtime_ns,
    value.st_ctime_ns,
)
if binding_descriptor(before) != binding_descriptor(after):
    raise RuntimeError("prepared binding changed while reading")
if (
    not stat.S_ISREG(binding_path_after.st_mode)
    or binding_path_after.st_nlink != 1
    or binding_path_after.st_dev != after.st_dev
    or binding_path_after.st_ino != after.st_ino
):
    raise RuntimeError("prepared binding path was replaced while reading")
raw = b"".join(chunks)
if len(raw) != after.st_size:
    raise RuntimeError("prepared binding byte count disagrees with file metadata")
binding = json.loads(raw)
request = binding.get("request") or {}
envs = request.get("Envs") or {}
if binding.get("schema") != "fastwam-dlc-prepared-binding-v1":
    raise RuntimeError("prepared binding schema mismatch")
if binding.get("experiment_id") != experiment_id:
    raise RuntimeError("prepared binding experiment mismatch")
if envs.get("FASTWAM_EXPERIMENT_ID") != experiment_id:
    raise RuntimeError("prepared request experiment mismatch")
if envs.get("FASTWAM_SUBMISSION_TAG") != submission_tag:
    raise RuntimeError("prepared request submission tag mismatch")
if envs.get("FASTWAM_SOURCE_ROOT") != source_root_literal:
    raise RuntimeError("prepared request source root mismatch")
if envs.get("FASTWAM_GATE2_ENTRYPOINT") != entrypoint:
    raise RuntimeError("prepared request entrypoint mismatch")
if envs.get("FASTWAM_PREPARED_BINDING_PATH") != str(binding_path):
    raise RuntimeError("prepared request binding path mismatch")
if envs.get("FASTWAM_GATE2_TRUSTED_RUNTIME_B64") != os.environ.get(
    "FASTWAM_GATE2_TRUSTED_RUNTIME_B64"
):
    raise RuntimeError("prepared request trusted runtime payload mismatch")
if envs.get("FASTWAM_GATE2_TRUSTED_RUNTIME_BYTES") != os.environ.get(
    "FASTWAM_GATE2_TRUSTED_RUNTIME_BYTES"
):
    raise RuntimeError("prepared request trusted runtime byte count mismatch")
trusted_runtime = base64.b64decode(
    envs["FASTWAM_GATE2_TRUSTED_RUNTIME_B64"].encode("ascii"), validate=True
)
if len(trusted_runtime) != int(envs["FASTWAM_GATE2_TRUSTED_RUNTIME_BYTES"]):
    raise RuntimeError("prepared trusted runtime byte count is inconsistent")

source_root = Path(source_root_literal)
if source_root.is_symlink() or not source_root.is_dir():
    raise RuntimeError("approved source root is missing or linked")

expected = binding.get("approved_source_metadata")
if not isinstance(expected, dict) or set(expected) != {
    "schema", "approved_source_root", "entries"
}:
    raise RuntimeError("prepared source binding has unexpected top-level fields")
if expected["schema"] != "fastwam-nohash-source-content-binding-v3":
    raise RuntimeError("prepared source binding schema mismatch")
if expected["approved_source_root"] != source_root_literal:
    raise RuntimeError("prepared source binding root mismatch")
expected_entries = expected["entries"]
if not isinstance(expected_entries, list) or not expected_entries:
    raise RuntimeError("prepared source binding entries are missing")
expected_paths = []
for entry in expected_entries:
    if not isinstance(entry, dict):
        raise RuntimeError("prepared source binding entry is not an object")
    relative_path = entry.get("path")
    kind = entry.get("kind")
    if not isinstance(relative_path, str) or not isinstance(kind, str):
        raise RuntimeError("prepared source binding path or kind is invalid")
    expected_paths.append(relative_path)
    if relative_path == ".":
        if kind != "directory":
            raise RuntimeError("prepared source root entry must be a directory")
    else:
        relative = Path(relative_path)
        if (
            not relative_path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != relative_path
        ):
            raise RuntimeError("prepared source binding path is non-canonical")
    if kind == "directory":
        if set(entry) != {"path", "kind"}:
            raise RuntimeError("prepared source directory has non-portable fields")
    elif kind == "file":
        if set(entry) != {"path", "kind", "size", "content_b64"}:
            raise RuntimeError("prepared source file has non-portable fields")
        size = entry["size"]
        encoded = entry["content_b64"]
        if type(size) is not int or size < 0 or not isinstance(encoded, str):
            raise RuntimeError("prepared source file size or content is invalid")
        try:
            payload = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as error:
            raise RuntimeError("prepared source content is not canonical base64") from error
        if len(payload) != size or base64.b64encode(payload).decode("ascii") != encoded:
            raise RuntimeError("prepared source content length or encoding mismatch")
    else:
        raise RuntimeError("prepared source binding kind is unsupported")
if (
    expected_paths != sorted(expected_paths)
    or len(expected_paths) != len(set(expected_paths))
    or expected_paths[0] != "."
):
    raise RuntimeError("prepared source paths are incomplete, duplicated, or unsorted")

if not source_root.is_absolute() or not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
    raise RuntimeError("approved source cannot be opened with the no-follow contract")
source_descriptor = lambda value: (
    value.st_dev,
    value.st_ino,
    value.st_mode,
    value.st_nlink,
    value.st_size,
    value.st_mtime_ns,
    value.st_ctime_ns,
)
directory_flags = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
)
file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)

def open_absolute_directory_nofollow(path):
    current_fd = os.open("/", directory_flags)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise

def collect_directory(directory_fd, relative_path, result):
    directory_before = os.fstat(directory_fd)
    if not stat.S_ISDIR(directory_before.st_mode):
        raise RuntimeError(f"approved source entry is not a directory: {relative_path}")
    result.append({"path": relative_path, "kind": "directory"})
    names_before = sorted(os.listdir(directory_fd))
    for name in names_before:
        if not name or name in {".", ".."} or "/" in name:
            raise RuntimeError(f"approved source returned an invalid name: {name!r}")
        child_relative = name if relative_path == "." else f"{relative_path}/{name}"
        initial = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(initial.st_mode):
            raise RuntimeError(f"approved source contains a symlink: {child_relative}")
        if stat.S_ISDIR(initial.st_mode):
            child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
            try:
                opened = os.fstat(child_fd)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or opened.st_dev != initial.st_dev
                    or opened.st_ino != initial.st_ino
                ):
                    raise RuntimeError(
                        f"approved source directory was replaced while opening: {child_relative}"
                    )
                collect_directory(child_fd, child_relative, result)
                path_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(path_after.st_mode)
                    or path_after.st_dev != opened.st_dev
                    or path_after.st_ino != opened.st_ino
                ):
                    raise RuntimeError(
                        f"approved source directory path was replaced: {child_relative}"
                    )
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
            raise RuntimeError(f"approved source contains an unsupported entry: {child_relative}")
        child_fd = os.open(name, file_flags, dir_fd=directory_fd)
        try:
            before = os.fstat(child_fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_dev != initial.st_dev
                or before.st_ino != initial.st_ino
            ):
                raise RuntimeError(
                    f"approved source file was replaced while opening: {child_relative}"
                )
            chunks = []
            while True:
                chunk = os.read(child_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(child_fd)
            path_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        finally:
            os.close(child_fd)
        payload = b"".join(chunks)
        if (
            source_descriptor(before) != source_descriptor(after)
            or not stat.S_ISREG(path_after.st_mode)
            or path_after.st_nlink != 1
            or path_after.st_dev != after.st_dev
            or path_after.st_ino != after.st_ino
            or len(payload) != after.st_size
        ):
            raise RuntimeError(f"approved source file changed while reading: {child_relative}")
        result.append({
            "path": child_relative,
            "kind": "file",
            "size": len(payload),
            "content_b64": base64.b64encode(payload).decode("ascii"),
        })
    names_after = sorted(os.listdir(directory_fd))
    directory_after = os.fstat(directory_fd)
    if names_before != names_after or source_descriptor(directory_before) != source_descriptor(directory_after):
        raise RuntimeError(f"approved source directory changed while scanning: {relative_path}")

source_fd = open_absolute_directory_nofollow(source_root)
try:
    source_root_before = os.fstat(source_fd)
    entries = []
    collect_directory(source_fd, ".", entries)
finally:
    os.close(source_fd)
source_fd_after = open_absolute_directory_nofollow(source_root)
try:
    source_root_after = os.fstat(source_fd_after)
finally:
    os.close(source_fd_after)
if (
    source_root_before.st_dev != source_root_after.st_dev
    or source_root_before.st_ino != source_root_after.st_ino
    or not stat.S_ISDIR(source_root_after.st_mode)
):
    raise RuntimeError("approved source root path was replaced while scanning")
entries.sort(key=lambda entry: entry["path"])
observed = {
    "schema": "fastwam-nohash-source-content-binding-v3",
    "approved_source_root": source_root_literal,
    "entries": entries,
}
if binding.get("approved_source_root") != source_root_literal:
    raise RuntimeError("prepared approved source root mismatch")
if expected != observed:
    raise RuntimeError("approved source metadata drifted after prepare")
PY
}

compare_trees() {
  "${FASTWAM_PYTHON}" - "$1" "$2" <<'PY'
import os
import stat
import sys
from pathlib import Path

left = Path(sys.argv[1]).resolve(strict=True)
right = Path(sys.argv[2]).resolve(strict=True)

def inventory(root):
    result = {}
    for base, dirs, files in os.walk(root, followlinks=False):
        base_path = Path(base)
        for name in dirs + files:
            path = base_path / name
            mode = path.lstat().st_mode
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"linked entry is forbidden: {path}")
            if stat.S_ISDIR(mode):
                result[relative] = ("dir", 0)
            elif stat.S_ISREG(mode):
                result[relative] = ("file", path.stat().st_size)
            else:
                raise RuntimeError(f"non-regular entry is forbidden: {path}")
    return result

left_items = inventory(left)
right_items = inventory(right)
if left_items != right_items:
    raise RuntimeError("tree path/type/byte-count comparison failed")
for relative, (kind, _) in left_items.items():
    if kind != "file":
        continue
    with (left / relative).open("rb") as a, (right / relative).open("rb") as b:
        while True:
            first = a.read(8 * 1024 * 1024)
            second = b.read(8 * 1024 * 1024)
            if first != second:
                raise RuntimeError(f"tree byte comparison failed: {relative}")
            if not first:
                break
PY
}

validate_prepared_source_binding
cp -a -- "${FASTWAM_SOURCE_ROOT}/." "${LOCAL_SOURCE}/"
validate_prepared_source_binding
compare_trees "${FASTWAM_SOURCE_ROOT}" "${LOCAL_SOURCE}"
cp -a -- "${FASTWAM_GAUSSIAN_CACHE_DIR}/." "${LOCAL_GAUSSIAN_PRIMARY}/"
compare_trees "${FASTWAM_GAUSSIAN_CACHE_DIR}" "${LOCAL_GAUSSIAN_PRIMARY}"
ACTIVE_GAUSSIAN_CACHE_DIR="${LOCAL_GAUSSIAN_PRIMARY}"
cp --reflink=auto -- "${FASTWAM_INITIAL_CHECKPOINT}" "${LOCAL_INITIAL}"
cmp -s -- "${FASTWAM_INITIAL_CHECKPOINT}" "${LOCAL_INITIAL}" || fail "initial checkpoint byte comparison failed"
cp --reflink=auto -- "${FASTWAM_N234_NOHASH_STATS_SOURCE}" "${LOCAL_STATS}"
cmp -s -- "${FASTWAM_N234_NOHASH_STATS_SOURCE}" "${LOCAL_STATS}" \
  || fail "normalization stats byte comparison failed"

"${FASTWAM_PYTHON}" - \
  "${EXPECTED_VAE_SOURCE}" "${LOCAL_VAE}" \
  "${PUBLISH_ROOT}/vae_staging.json" "${EXPECTED_VAE_SOURCE_BYTES}" <<'PY'
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
receipt = Path(sys.argv[3])
expected_bytes = int(sys.argv[4])

if source.is_symlink() or source.resolve(strict=True) != source:
    raise RuntimeError("VAE source must be the exact canonical non-symlink path")
if destination.exists() or destination.is_symlink():
    raise RuntimeError("private VAE destination already exists")


def fields(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )


read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
write_flags = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
)
source_fd = os.open(source, read_flags)
destination_fd = os.open(destination, write_flags, 0o600)
copied_bytes = 0
try:
    source_before = os.fstat(source_fd)
    if (
        not stat.S_ISREG(source_before.st_mode)
        or source_before.st_nlink != 1
        or source_before.st_size != expected_bytes
    ):
        raise RuntimeError("VAE source metadata is outside the fixed contract")
    while True:
        chunk = os.read(source_fd, 8 * 1024 * 1024)
        if not chunk:
            break
        view = memoryview(chunk)
        while view:
            written = os.write(destination_fd, view)
            if written <= 0:
                raise RuntimeError("VAE private staging made no write progress")
            view = view[written:]
            copied_bytes += written
    os.fsync(destination_fd)
    source_after_copy = os.fstat(source_fd)
    destination_after_copy = os.fstat(destination_fd)
finally:
    os.close(destination_fd)
    os.close(source_fd)

if fields(source_before) != fields(source_after_copy):
    raise RuntimeError("VAE source changed while staging")
if (
    copied_bytes != expected_bytes
    or not stat.S_ISREG(destination_after_copy.st_mode)
    or destination_after_copy.st_nlink != 1
    or destination_after_copy.st_size != expected_bytes
):
    raise RuntimeError("private VAE staging metadata is incomplete")

source_fd = os.open(source, read_flags)
destination_fd = os.open(destination, read_flags)
compared_bytes = 0
try:
    source_before_compare = os.fstat(source_fd)
    destination_before_compare = os.fstat(destination_fd)
    while True:
        source_chunk = os.read(source_fd, 8 * 1024 * 1024)
        destination_chunk = os.read(destination_fd, 8 * 1024 * 1024)
        if source_chunk != destination_chunk:
            raise RuntimeError("private VAE direct byte comparison failed")
        compared_bytes += len(source_chunk)
        if not source_chunk:
            break
    source_after_compare = os.fstat(source_fd)
    destination_after_compare = os.fstat(destination_fd)
finally:
    os.close(destination_fd)
    os.close(source_fd)

if (
    fields(source_before) != fields(source_before_compare)
    or fields(source_before_compare) != fields(source_after_compare)
):
    raise RuntimeError("VAE source metadata changed across staging and comparison")
if fields(destination_before_compare) != fields(destination_after_compare):
    raise RuntimeError("private VAE metadata changed during comparison")
if compared_bytes != expected_bytes:
    raise RuntimeError("private VAE compared byte count mismatch")

payload = {
    "schema": "fastwam-gate2-vae-staging-v1",
    "integrity_mode": "metadata_no_hash",
    "source_path": str(source),
    "private_path": str(destination),
    "expected_bytes": expected_bytes,
    "copied_bytes": copied_bytes,
    "compared_bytes": compared_bytes,
    "source_regular_single_link": True,
    "private_regular_single_link": True,
    "source_metadata_stable": True,
    "private_metadata_stable": True,
    "direct_file_byte_comparison": "passed",
    "created_at": datetime.now(timezone.utc)
    .replace(microsecond=0)
    .isoformat()
    .replace("+00:00", "Z"),
}
with receipt.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
directory_fd = os.open(receipt.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY

"${FASTWAM_PYTHON}" - "${LOCAL_STATS}" "${FASTWAM_DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

stats_path = Path(sys.argv[1])
dataset_root_literal = sys.argv[2]
dataset_root = Path(dataset_root_literal).resolve(strict=True)
with stats_path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
if not isinstance(payload, dict):
    raise TypeError("normalization stats must be a JSON object")
source_root = payload.get("source_root")
if source_root != dataset_root_literal or str(dataset_root) != dataset_root_literal:
    raise RuntimeError(
        f"normalization stats source_root mismatch: stats={source_root!r} dataset={dataset_root}"
    )
for field in ("action", "state", "files", "trajectories", "cardinality", "normalization_fit"):
    if field not in payload:
        raise KeyError(f"normalization stats lacks required field: {field}")
PY

"${FASTWAM_PYTHON}" - \
  "${FASTWAM_GAUSSIAN_CACHE_DIR}" "${ACTIVE_GAUSSIAN_CACHE_DIR}" \
  "${PUBLISH_ROOT}/gaussian_primary_staging.json" <<'PY'
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

source = Path(sys.argv[1]).resolve(strict=True)
active = Path(sys.argv[2]).resolve(strict=True)
target = Path(sys.argv[3])

def inventory(root):
    files = 0
    total_bytes = 0
    for base, dirs, names in os.walk(root, followlinks=False):
        base_path = Path(base)
        for name in dirs + names:
            path = base_path / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"linked cache entry is forbidden: {path}")
            if stat.S_ISREG(mode):
                files += 1
                total_bytes += path.stat().st_size
            elif not stat.S_ISDIR(mode):
                raise RuntimeError(f"non-regular cache entry is forbidden: {path}")
    return {"files": files, "bytes": total_bytes}

payload = {
    "schema": "fastwam-gate2-gaussian-primary-staging-v1",
    "integrity_mode": "metadata_no_hash",
    "source_root": str(source),
    "active_root": str(active),
    "source_inventory": inventory(source),
    "active_inventory": inventory(active),
    "direct_path_type_size_and_byte_comparison": "passed",
    "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
}
if payload["source_inventory"] != payload["active_inventory"]:
    raise RuntimeError("staged Gaussian cache inventory mismatch")
with target.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY

[[ -f "${LOCAL_SOURCE}/scripts/train.py" ]] || fail "staged source lacks scripts/train.py"
[[ -f "${LOCAL_SOURCE}/configs/task/${EXPECTED_TASK}.yaml" ]] || fail "staged source lacks Gate2 task config"
[[ -f "${LOCAL_SOURCE}/scripts/accelerate_configs/accelerate_zero2_ds.yaml" ]] || fail "staged source lacks Accelerate config"
PREFLIGHT_SCRIPT="${LOCAL_SOURCE}/.research-workflow/experiments/${SOURCE_EXPERIMENT}/real_data_nohash_preflight.py"
[[ -f "${PREFLIGHT_SCRIPT}" && ! -L "${PREFLIGHT_SCRIPT}" ]] || fail "staged source lacks real-data no-hash preflight"
STRUCTURED_EVIDENCE_SCRIPT="${LOCAL_SOURCE}/.research-workflow/experiments/${SOURCE_EXPERIMENT}/dlc-draft-1x8/gate2_structured_evidence.py"
[[ -f "${STRUCTURED_EVIDENCE_SCRIPT}" && ! -L "${STRUCTURED_EVIDENCE_SCRIPT}" ]] \
  || fail "staged source lacks the structured Gate2 evidence validator"

export PYTHONPATH="${LOCAL_SOURCE}/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export DIFFSYNTH_MODEL_BASE_PATH="${LOCAL_MODEL_CACHE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HOME="${LOCAL_ROOT}/home"
export HF_HOME="${LOCAL_ROOT}/hf-home"
export MODELSCOPE_CACHE="${LOCAL_ROOT}/modelscope-cache"
export XDG_CACHE_HOME="${LOCAL_ROOT}/xdg-cache"
export TORCH_HOME="${LOCAL_ROOT}/torch-cache"
export WANDB_MODE=disabled
export FASTWAM_N234_NOHASH_STATS="${LOCAL_STATS}"
export FASTWAM_WEIGHT_STAGING_DIR="${LOCAL_ROOT}/weight-staging"
mkdir -p -m 0700 -- \
  "${FASTWAM_WEIGHT_STAGING_DIR}" "${HOME}" "${HF_HOME}" \
  "${MODELSCOPE_CACHE}" "${XDG_CACHE_HOME}" "${TORCH_HOME}"

PREFLIGHT_EVIDENCE="${PUBLISH_ROOT}/real_data_nohash_preflight.json"
PREFLIGHT_LOG="${PUBLISH_ROOT}/real_data_nohash_preflight.log"
env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u ROLE_RANK \
  CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "${FASTWAM_PYTHON}" "${PREFLIGHT_SCRIPT}" \
  --data-root "${FASTWAM_DATASET_ROOT}" \
  --stats "${LOCAL_STATS}" \
  --text-cache "${FASTWAM_TEXT_CACHE_DIR}" \
  --gaussian-cache "${ACTIVE_GAUSSIAN_CACHE_DIR}" \
  --gaussian-fallback-cache "${FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR}" \
  --output "${PREFLIGHT_EVIDENCE}" \
  2>&1 | tee "${PREFLIGHT_LOG}"

"${FASTWAM_PYTHON}" - "${PREFLIGHT_EVIDENCE}" \
  "${FASTWAM_DATASET_ROOT}" "${ACTIVE_GAUSSIAN_CACHE_DIR}" \
  "${FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR}" <<'PY'
import json
import sys
from pathlib import Path

evidence_path = Path(sys.argv[1])
with evidence_path.open("r", encoding="utf-8") as handle:
    evidence = json.load(handle)
if evidence.get("status") != "PASS":
    raise RuntimeError("real-data no-hash preflight did not report PASS")
if evidence.get("integrity_mode") != "metadata_no_hash":
    raise RuntimeError("real-data preflight integrity mode mismatch")
if evidence.get("known_python_digest_entrypoints_forbidden") is not True:
    raise RuntimeError("real-data preflight digest-entrypoint guard is incomplete")
if evidence.get("guarded_python_digest_attempts") != 0:
    raise RuntimeError("real-data preflight observed a digest attempt")
if evidence.get("huggingface_datasets_imported") is not False:
    raise RuntimeError("RoboFactory preflight unexpectedly imported Hugging Face datasets")
if evidence.get("train_agent_counts") != [2] or evidence.get("val_agent_counts") != [2]:
    raise RuntimeError("real-data preflight agent-count scope is not exactly N=2")
if int(evidence.get("train_windows") or 0) < 1 or int(evidence.get("val_windows") or 0) < 1:
    raise RuntimeError("real-data preflight found an empty train or validation split")
if evidence.get("data_root") != str(Path(sys.argv[2]).resolve(strict=True)):
    raise RuntimeError("real-data preflight dataset root mismatch")
if evidence.get("gaussian_cache") != str(Path(sys.argv[3]).resolve(strict=True)):
    raise RuntimeError("real-data preflight primary Gaussian root mismatch")
if evidence.get("gaussian_fallback_cache") != str(Path(sys.argv[4]).resolve(strict=True)):
    raise RuntimeError("real-data preflight fallback Gaussian root mismatch")
train_fallback = (evidence.get("fallback_samples") or {}).get("train") or {}
if train_fallback.get("status") != "materialized":
    raise RuntimeError("training preflight did not materialize a canonical fallback frame")
if train_fallback.get("frames_read", 0) < 1:
    raise RuntimeError("training preflight did not record a canonical fallback read")
if len(train_fallback.get("reads") or []) != train_fallback.get("frames_read"):
    raise RuntimeError("training fallback read count is internally inconsistent")
if not train_fallback.get("primary_missing_agents"):
    raise RuntimeError("training fallback candidate lacks a primary-cache miss")
shape = train_fallback.get("agent_gaussian_shape") or []
if shape != [2, 13, 28, 40]:
    raise RuntimeError(f"fallback projection shape mismatch: {shape!r}")
if train_fallback.get("agent_gaussian_dtype") != "torch.float16":
    raise RuntimeError("fallback projection dtype is not torch.float16")
if train_fallback.get("agent_gaussian_finite") is not True:
    raise RuntimeError("fallback projection contains non-finite values")
train_cache = evidence.get("train_gaussian_preflight") or {}
if int(train_cache.get("fallback_keys") or 0) < 1:
    raise RuntimeError("training Gaussian preflight did not declare fallback keys")
if int(train_cache.get("fallback_shards_validated") or 0) < 1:
    raise RuntimeError("training Gaussian preflight did not validate a fallback shard")
PY

launch_training() {
  local phase="$1"
  local output="$2"
  local resume_value="$3"
  local init_value="$4"
  local save_state="$5"
  local recovery_receipt="$6"
  local save_final_checkpoint="$7"
  local log_file="${PUBLISH_ROOT}/${phase}.log"

  env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    -u GROUP_RANK -u ROLE_RANK \
    FASTWAM_RECOVERY_LOAD_RECEIPT="${recovery_receipt}" \
    "${FASTWAM_PYTHON}" -m accelerate.commands.launch \
      --config_file "${LOCAL_SOURCE}/scripts/accelerate_configs/accelerate_zero2_ds.yaml" \
      --num_machines 1 \
      --num_processes 8 \
      "${LOCAL_SOURCE}/scripts/train.py" \
      "task=${EXPECTED_TASK}" \
      "output_dir=${output}" \
      "artifact_integrity_mode=metadata_no_hash" \
      "model.checkpoint_integrity_mode=metadata_no_hash" \
      "data.train.integrity_mode=metadata_no_hash" \
      "data.val.integrity_mode=metadata_no_hash" \
      "data.train.root_dir=${FASTWAM_DATASET_ROOT}" \
      "data.val.root_dir=${FASTWAM_DATASET_ROOT}" \
      "data.train.gaussian_cache_dir=${ACTIVE_GAUSSIAN_CACHE_DIR}" \
      "data.val.gaussian_cache_dir=${ACTIVE_GAUSSIAN_CACHE_DIR}" \
      "data.train.gaussian_fallback_cache_dir=${FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR}" \
      "data.val.gaussian_fallback_cache_dir=${FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR}" \
      "init_weights=${init_value}" \
      "resume=${resume_value}" \
      "max_steps=2" \
      "save_every=1" \
      "recovery_gate_stop_after_checkpoint_step=1" \
      "checkpoint_state_kind=full" \
      "eval_every=0" \
      "offline_eval_num_samples=0" \
      "num_workers=0" \
      "save_training_state=${save_state}" \
      "save_final_checkpoint=${save_final_checkpoint}" \
      "seal_training_state=false" \
      "seal_training_run=false" \
      "terminal_rehash_weights=false" \
      "training_terminal_contract=null" \
      "training_run_profile=null" \
      "training_task_scope_receipt=null" \
      "wandb.enabled=false" \
      2>&1 | tee "${log_file}"
}

cd -- "${LOCAL_SOURCE}"
launch_training save_world "${SAVE_OUTPUT}" null "${LOCAL_INITIAL}" true "" true
SAVE_STATE="${SAVE_OUTPUT}/checkpoints/state/step_000001"
STATE_FILE="${SAVE_STATE}/trainer_state.json"
SAVE_WEIGHTS="${SAVE_OUTPUT}/checkpoints/weights/step_000001.pt"
SAVE_MANIFEST="${SAVE_WEIGHTS}.manifest.json"
SAVE_COMPLETE="${SAVE_WEIGHTS}.COMPLETE"
[[ -d "${SAVE_STATE}" && -f "${STATE_FILE}" ]] || fail "save world did not publish step-1 full state"
[[ -f "${SAVE_WEIGHTS}" && -f "${SAVE_MANIFEST}" && -f "${SAVE_COMPLETE}" ]] \
  || fail "save world did not publish the complete step-1 weights set"

"${FASTWAM_PYTHON}" "${STRUCTURED_EVIDENCE_SCRIPT}" verify-save \
  --publish-root "${PUBLISH_ROOT}"

# This is a separate Accelerate/DeepSpeed process world. It restores step 1,
# performs the optimizer update for step 2, and writes both full step-2 weights
# and a fresh full training state.
LOAD_RECEIPT="${LOAD_OUTPUT}/recovery_load_receipt.json"
launch_training load_world "${LOAD_OUTPUT}" "${SAVE_STATE}" null true \
  "${LOAD_RECEIPT}" true
FINAL_STATE="${LOAD_OUTPUT}/checkpoints/state/step_000002"
FINAL_STATE_FILE="${FINAL_STATE}/trainer_state.json"
FINAL_WEIGHTS="${LOAD_OUTPUT}/checkpoints/weights/step_000002.pt"
FINAL_MANIFEST="${FINAL_WEIGHTS}.manifest.json"
FINAL_COMPLETE="${FINAL_WEIGHTS}.COMPLETE"
[[ -d "${FINAL_STATE}" && -f "${FINAL_STATE_FILE}" ]] \
  || fail "fresh load world did not publish step-2 full training state"
[[ -f "${FINAL_WEIGHTS}" && -f "${FINAL_MANIFEST}" && -f "${FINAL_COMPLETE}" ]] \
  || fail "fresh load world did not publish the complete step-2 full weights set"
[[ -f "${LOAD_RECEIPT}" && ! -L "${LOAD_RECEIPT}" ]] \
  || fail "fresh load world did not publish its native recovery receipt"

# A third, separately launched process world loads the completed step-2 state
# and exits without training or checkpoint publication. Its trainer-native
# receipt proves that the final state is itself a future resume point.
FINAL_VERIFY_RECEIPT="${FINAL_VERIFY_OUTPUT}/recovery_load_receipt.json"
launch_training final_verify_world "${FINAL_VERIFY_OUTPUT}" "${FINAL_STATE}" \
  null false "${FINAL_VERIFY_RECEIPT}" false
[[ -f "${FINAL_VERIFY_RECEIPT}" && ! -L "${FINAL_VERIFY_RECEIPT}" ]] \
  || fail "final verification world did not publish its native recovery receipt"

"${FASTWAM_PYTHON}" "${STRUCTURED_EVIDENCE_SCRIPT}" verify-recovery \
  --publish-root "${PUBLISH_ROOT}"

PUBLISHER_SCRIPT="${LOCAL_SOURCE}/.research-workflow/experiments/FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-S42-20260809/dlc-draft-1x8/publish_gate2.py"
[[ -f "${PUBLISHER_SCRIPT}" && ! -L "${PUBLISHER_SCRIPT}" ]] || fail "Gate2 OSS publisher is missing or linked"
"${FASTWAM_PYTHON}" "${PUBLISHER_SCRIPT}" \
  --local-stage "${PUBLISH_ROOT}" \
  --oss-output-root "${FASTWAM_OSS_OUTPUT_ROOT}" \
  --submission-tag "${FASTWAM_SUBMISSION_TAG}"

echo "Gate2 complete: ${FASTWAM_OSS_OUTPUT_ROOT}/COMPLETE.json"
