#!/usr/bin/env python3
"""Fail-closed one-process-per-GPU manager for the 40-task MF-WAM G0 run.

The manager never invokes a shell, never retries a task, and never replaces an
existing log, status, archive, or terminal receipt.  A successful run moves the
four raw LIBERO suite directories out of the canonical artifact root with
Linux ``renameat2(RENAME_NOREPLACE)`` before publishing the trusted terminal
manifest consumed by ``seal_mf_wam_g0_terminal.py``.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

# Formal manager execution must not dirty a fresh source checkout before the
# source gate can inspect it.  The manager is executed as a script; setting this
# before importing ``fastwam`` prevents its validation imports from creating
# ignored ``__pycache__`` artifacts even if the outer launcher omitted ``-B``.
sys.dont_write_bytecode = True

try:
    import fastwam.validation.g0_contract as contract
except ModuleNotFoundError:
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root / "src"))
    import fastwam.validation.g0_contract as contract  # type: ignore[no-redef]


SUITES = contract.SUITES
TASKS_PER_SUITE = contract.TASKS_PER_SUITE
TRIALS_PER_TASK = contract.TRIALS_PER_TASK
EXPECTED_TASKS = contract.EXPECTED_TASKS
EXPECTED_EPISODES = contract.EXPECTED_EPISODES
EXPECTED_INPUT_FILES = EXPECTED_TASKS * 2 + EXPECTED_EPISODES
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PYTHON_EXECUTABLE_RE = re.compile(r"^python(?:3(?:\.\d+)?)?$")
_RENAME_NOREPLACE = 1
_RENAME_PROBE_PREFIX = ".mf-wam-g0-rename-probe-"
INSTRUMENTATION_MEMFD_FD = 198
_INOTIFY_EVENT = struct.Struct("iIII")
_INOTIFY_MUTATION_MASK = (
    0x00000002  # IN_MODIFY
    | 0x00000004  # IN_ATTRIB
    | 0x00000008  # IN_CLOSE_WRITE
    | 0x00000040  # IN_MOVED_FROM
    | 0x00000080  # IN_MOVED_TO
    | 0x00000100  # IN_CREATE
    | 0x00000200  # IN_DELETE
    | 0x00000400  # IN_DELETE_SELF
    | 0x00000800  # IN_MOVE_SELF
    | 0x00002000  # IN_UNMOUNT
    | 0x00004000  # IN_Q_OVERFLOW
    | 0x00008000  # IN_IGNORED
)
_F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
_MEMFD_SEALS = (
    getattr(fcntl, "F_SEAL_SEAL", 1)
    | getattr(fcntl, "F_SEAL_SHRINK", 2)
    | getattr(fcntl, "F_SEAL_GROW", 4)
    | getattr(fcntl, "F_SEAL_WRITE", 8)
)
_OFFICIAL_CRITICAL_PATHS = (
    "experiments/libero/eval_libero_single.py",
    "experiments/libero/libero_utils.py",
    "experiments/libero/action_ensembler.py",
    "configs/sim_libero.yaml",
    "configs/train.yaml",
    "configs/data/libero_2cam.yaml",
    "configs/model/fastwam.yaml",
    "configs/task/libero_uncond_2cam224_1e-4.yaml",
    "src/fastwam/runtime.py",
    "src/fastwam/models/wan22/fastwam.py",
    "src/fastwam/utils/pytorch_utils.py",
)
_INSTRUMENTATION_CRITICAL_PATHS = (
    "scripts/run_mf_wam_g0_traced.py",
    "scripts/mf_wam_g0_instrumentation.py",
)

WORKER_TERMINAL_KEYS = frozenset(
    (
        "status", "kind", "run_id", "process_receipt", "official_commit",
        "official_result_type", "official_result_receipt",
        "terminal_source_identities", "external_prelaunch_commit_tree_gate_required",
        "environment_sha256",
    )
)

FIXED_WORKER_ENVIRONMENT = {
    "HOME": "/tmp",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONUTF8": "1",
    "TMPDIR": "/tmp",
    "TZ": "UTC",
}
_GIT_ENVIRONMENT = {
    **FIXED_WORKER_ENVIRONMENT,
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
}

STATUS_KEYS = frozenset(
    (
        "schema_version", "kind", "run_id", "process_id", "task_suite",
        "task_id", "gpu_id", "state", "launched_at", "completed_at",
        "exit_code", "complete", "failure_reason", "command_argv",
        "command_sha256", "environment_bindings", "environment_sha256",
        "log", "canonical_result", "trace_receipt", "raw_result",
    )
)


class ManagerError(RuntimeError):
    """Raised when the manager cannot prove a safe, complete run."""


def _reject_constant(value: str) -> None:
    raise ManagerError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManagerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except ManagerError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ManagerError(f"cannot load strict JSON {label}: {exc}") from exc


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ManagerError(f"cannot encode canonical JSON: {exc}") from exc


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value) or value == "0" * 64:
        raise ManagerError(f"{label} must be a non-placeholder lowercase SHA-256")
    return value


def _lexical_absolute(path: Path) -> Path:
    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    if ".." in absolute.parts:
        raise ManagerError(f"parent traversal is forbidden: {path}")
    return Path(os.path.normpath(str(absolute)))


def _safe_relative(value: str, label: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or pure.as_posix() != value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ManagerError(f"unsafe {label}: {value!r}")
    return value


def _open_absolute(path: Path, *, directory: bool) -> int:
    absolute = _lexical_absolute(path)
    parts = absolute.parts[1:]
    current_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for index, component in enumerate(parts):
            final = index == len(parts) - 1
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            if not final or directory:
                flags |= os.O_DIRECTORY
            next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        metadata = os.fstat(current_fd)
        expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
        if not expected:
            raise ManagerError(f"path has the wrong type: {absolute}")
        return current_fd
    except Exception as exc:
        try:
            os.close(current_fd)
        except OSError:
            pass
        if isinstance(exc, ManagerError):
            raise
        raise ManagerError(f"cannot open without following symlinks: {absolute}: {exc}") from exc


def _read_fd(fd: int, *, capture: bool) -> dict[str, Any]:
    before = os.fstat(fd)
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if capture else None
    total = 0
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
        if chunks is not None:
            chunks.append(chunk)
    after = os.fstat(fd)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ManagerError("file changed during readback")
    if total != after.st_size or after.st_nlink != 1:
        raise ManagerError("file is unstable or hardlinked")
    result: dict[str, Any] = {
        "sha256": digest.hexdigest(),
        "size_bytes": total,
        "identity": (after.st_dev, after.st_ino),
        "mode": after.st_mode,
    }
    if chunks is not None:
        result["bytes"] = b"".join(chunks)
    return result


def _read_absolute(path: Path, *, capture: bool = False) -> dict[str, Any]:
    descriptor = _open_absolute(path, directory=False)
    try:
        return _read_fd(descriptor, capture=capture)
    finally:
        os.close(descriptor)


def _open_relative(root: Path, relative: str) -> int:
    pure = PurePosixPath(_safe_relative(relative, "relative path"))
    directory_fd = _open_absolute(root, directory=True)
    try:
        for component in pure.parts[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(
            pure.parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise ManagerError(f"cannot safely open {relative}: {exc}") from exc
    finally:
        os.close(directory_fd)


def _read_relative(root: Path, relative: str, *, capture: bool = False) -> dict[str, Any]:
    descriptor = _open_relative(root, relative)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ManagerError(f"not a regular file: {relative}")
        return _read_fd(descriptor, capture=capture)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class _BoundSourceFile:
    root: Path
    relative: str
    sha256: str
    size_bytes: int
    identity: tuple[int, int]


class _SourceMutationGuard:
    """Local-host source guard; it is not a CPFS cross-node trust primitive.

    The inotify queue detects transient swap/restore operations while pinned
    inode and SHA-256 readbacks prevent a checkpoint from accepting a changed
    critical file.  This boundary deliberately does not claim resistance to
    root, ptrace, or writes performed on another CPFS client.
    """

    def __init__(self, roots: Sequence[Path]) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        init = getattr(libc, "inotify_init1", None)
        add_watch = getattr(libc, "inotify_add_watch", None)
        if init is None or add_watch is None:
            raise ManagerError("Linux inotify is required for the source race guard")
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add_watch.restype = ctypes.c_int
        descriptor = init(os.O_NONBLOCK | os.O_CLOEXEC)
        if descriptor < 0:
            error = ctypes.get_errno()
            raise ManagerError(
                f"cannot initialize source inotify guard: {os.strerror(error)}"
            )
        self._fd = descriptor
        self._add_watch = add_watch
        self._roots: list[tuple[Path, tuple[int, int]]] = []
        self._pinned_fds: list[int] = []
        self._watch_filters: dict[int, set[str] | None] = {}
        self._files: tuple[_BoundSourceFile, ...] = ()
        self._closed = False
        try:
            unique = sorted(
                {_lexical_absolute(root) for root in roots},
                key=lambda value: str(value).encode("utf-8"),
            )
            for root in unique:
                parent_fd = _open_absolute(root.parent, directory=True)
                self._pinned_fds.append(parent_fd)
                self._watch_pinned_directory(parent_fd, filter_name=root.name)
                try:
                    root_fd = os.open(
                        root.name,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_CLOEXEC
                        | os.O_NOFOLLOW,
                        dir_fd=parent_fd,
                    )
                except OSError as exc:
                    raise ManagerError(f"cannot pin source root {root}: {exc}") from exc
                self._pinned_fds.append(root_fd)
                metadata = os.fstat(root_fd)
                self._roots.append((root, (metadata.st_dev, metadata.st_ino)))
                self._watch_pinned_directory(root_fd, filter_name=None)
                pinned_root = Path(f"/proc/self/fd/{root_fd}")

                def fail_walk(error: OSError) -> None:
                    raise ManagerError(
                        f"cannot enumerate pinned source root {root}: {error}"
                    ) from error

                for current, _directories, _files in os.walk(
                    pinned_root,
                    topdown=True,
                    onerror=fail_walk,
                    followlinks=False,
                ):
                    if Path(current) == pinned_root:
                        continue
                    try:
                        current_fd = os.open(
                            current,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
                        )
                    except OSError as exc:
                        raise ManagerError(
                            f"cannot pin recursive source directory {current}: {exc}"
                        ) from exc
                    try:
                        self._watch_pinned_directory(current_fd, filter_name=None)
                    finally:
                        os.close(current_fd)
            self.checkpoint("watch installation")
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "_SourceMutationGuard":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            os.close(self._fd)
            for descriptor in reversed(self._pinned_fds):
                os.close(descriptor)
            self._pinned_fds.clear()

    def _watch_pinned_directory(
        self, descriptor: int, *, filter_name: str | None
    ) -> None:
        before = os.fstat(descriptor)
        watch_mask = (
            _INOTIFY_MUTATION_MASK
            | 0x01000000  # IN_ONLYDIR
            | 0x04000000  # IN_EXCL_UNLINK
        )
        watch = self._add_watch(
            self._fd, os.fsencode(f"/proc/self/fd/{descriptor}"), watch_mask
        )
        if watch < 0:
            error = ctypes.get_errno()
            raise ManagerError(
                "cannot watch pinned source directory: " + os.strerror(error)
            )
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ManagerError("pinned source directory changed while watched")
        existing = self._watch_filters.get(watch, set())
        if existing is None or filter_name is None:
            self._watch_filters[watch] = None
        else:
            existing.add(filter_name)
            self._watch_filters[watch] = existing

    def bind_critical_sources(self, config: "ManagerConfig") -> bytes:
        rows: list[_BoundSourceFile] = []
        instrumentation_bytes: bytes | None = None
        for root, paths in (
            (_lexical_absolute(config.official_root), _OFFICIAL_CRITICAL_PATHS),
            (
                _lexical_absolute(config.working_directory),
                _INSTRUMENTATION_CRITICAL_PATHS,
            ),
        ):
            for relative in paths:
                observed = _read_relative(root, relative, capture=True)
                rows.append(
                    _BoundSourceFile(
                        root=root,
                        relative=relative,
                        sha256=observed["sha256"],
                        size_bytes=observed["size_bytes"],
                        identity=observed["identity"],
                    )
                )
                if relative == "scripts/mf_wam_g0_instrumentation.py":
                    instrumentation_bytes = observed["bytes"]
        if instrumentation_bytes is None or not instrumentation_bytes:
            raise ManagerError("validated instrumentation source is empty")
        self._files = tuple(rows)
        self.checkpoint("critical source binding")
        return instrumentation_bytes

    def _drain_events(self, label: str) -> None:
        while True:
            try:
                raw = os.read(self._fd, 1024 * 1024)
            except BlockingIOError:
                return
            except OSError as exc:
                raise ManagerError(f"cannot read source inotify queue: {exc}") from exc
            if not raw:
                raise ManagerError("source inotify descriptor closed unexpectedly")
            offset = 0
            events: list[str] = []
            while offset < len(raw):
                if len(raw) - offset < _INOTIFY_EVENT.size:
                    raise ManagerError("truncated source inotify event")
                watch, mask, _cookie, length = _INOTIFY_EVENT.unpack_from(raw, offset)
                end = offset + _INOTIFY_EVENT.size + length
                if end > len(raw):
                    raise ManagerError("truncated source inotify name")
                name_raw = raw[offset + _INOTIFY_EVENT.size : end]
                name = name_raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")
                global_failure = bool(
                    mask
                    & (
                        0x00000400  # IN_DELETE_SELF
                        | 0x00000800  # IN_MOVE_SELF
                        | 0x00002000  # IN_UNMOUNT
                        | 0x00004000  # IN_Q_OVERFLOW
                        | 0x00008000  # IN_IGNORED
                    )
                )
                watched_names = self._watch_filters.get(watch)
                if global_failure or watched_names is None or name in watched_names:
                    events.append(f"wd={watch},mask=0x{mask:08x},name={name!r}")
                offset = end
            if events:
                raise ManagerError(
                    f"source mutation observed during {label}: "
                    + "; ".join(events[:8])
                )

    def checkpoint(self, label: str) -> None:
        self._drain_events(label)
        for root, identity in self._roots:
            descriptor = _open_absolute(root, directory=True)
            try:
                observed = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if (observed.st_dev, observed.st_ino) != identity:
                raise ManagerError(f"source root inode changed during {label}: {root}")
        for item in self._files:
            observed = _read_relative(item.root, item.relative)
            if (
                observed["identity"] != item.identity
                or observed["sha256"] != item.sha256
                or observed["size_bytes"] != item.size_bytes
            ):
                raise ManagerError(
                    f"critical source inode/hash changed during {label}: "
                    f"{item.root / item.relative}"
                )
        self._drain_events(label)


def _create_sealed_instrumentation_memfd(source: bytes) -> int:
    if not hasattr(os, "memfd_create"):
        raise ManagerError("Linux memfd_create is required for instrumentation loading")
    try:
        os.fstat(INSTRUMENTATION_MEMFD_FD)
    except OSError as exc:
        if exc.errno != errno.EBADF:
            raise ManagerError(f"cannot inspect fixed instrumentation fd: {exc}") from exc
    else:
        raise ManagerError(
            f"fixed instrumentation fd {INSTRUMENTATION_MEMFD_FD} is already occupied"
        )
    descriptor = os.memfd_create(
        "mf-wam-g0-instrumentation",
        os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
    )
    try:
        view = memoryview(source)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise ManagerError("short write to instrumentation memfd")
            written += count
        os.fchmod(descriptor, 0o400)
        fcntl.fcntl(descriptor, _F_ADD_SEALS, _MEMFD_SEALS)
        if fcntl.fcntl(descriptor, _F_GET_SEALS) != _MEMFD_SEALS:
            raise ManagerError("instrumentation memfd seal readback mismatch")
        os.dup2(descriptor, INSTRUMENTATION_MEMFD_FD, inheritable=True)
    finally:
        os.close(descriptor)
    readback = os.pread(INSTRUMENTATION_MEMFD_FD, len(source) + 1, 0)
    if readback != source:
        os.close(INSTRUMENTATION_MEMFD_FD)
        raise ManagerError("instrumentation memfd byte readback mismatch")
    return INSTRUMENTATION_MEMFD_FD


def _mkdir_absolute(path: Path) -> None:
    absolute = _lexical_absolute(path)
    parent_fd = _open_absolute(absolute.parent, directory=True)
    try:
        try:
            os.mkdir(absolute.name, 0o755, dir_fd=parent_fd)
        except FileExistsError:
            descriptor = os.open(
                absolute.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def _mkdir_relative(root: Path, relative: str) -> None:
    current = _lexical_absolute(root)
    for component in PurePosixPath(_safe_relative(relative, "directory path")).parts:
        current = current / component
        _mkdir_absolute(current)


def _publish_no_replace(
    path: Path,
    payload: bytes,
    *,
    post_link_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    target = _lexical_absolute(path)
    parent_fd = _open_absolute(target.parent, directory=True)
    temporary = f".{target.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o644)
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise ManagerError(f"refusing to overwrite existing artifact: {target}") from exc
        os.fsync(parent_fd)
        if post_link_check is not None:
            try:
                post_link_check()
            except BaseException:
                linked = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
                temporary_metadata = os.stat(
                    temporary, dir_fd=parent_fd, follow_symlinks=False
                )
                if (linked.st_dev, linked.st_ino) != (
                    temporary_metadata.st_dev,
                    temporary_metadata.st_ino,
                ):
                    raise ManagerError(
                        "terminal artifact identity changed before rollback"
                    )
                os.unlink(target.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
                raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _publish_json_no_replace(
    path: Path,
    payload: Mapping[str, Any],
    *,
    post_link_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    return _publish_no_replace(
        path,
        _canonical_bytes(payload),
        post_link_check=post_link_check,
    )


def _open_log_no_replace(path: Path) -> BinaryIO:
    target = _lexical_absolute(path)
    parent_fd = _open_absolute(target.parent, directory=True)
    try:
        try:
            descriptor = os.open(
                target.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o644,
                dir_fd=parent_fd,
            )
        except FileExistsError as exc:
            raise ManagerError(f"refusing to overwrite existing log: {target}") from exc
    finally:
        os.close(parent_fd)
    return os.fdopen(descriptor, "wb", buffering=0)


def _tree_sha(files: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda entry: str(entry["path"]).encode("utf-8")):
        digest.update(f"{item['sha256']}  {item['path']}\n".encode("utf-8"))
    return digest.hexdigest()


def _timestamp(now: Callable[[], dt.datetime]) -> str:
    observed = now()
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ManagerError("manager clock must return a timezone-aware datetime")
    return observed.isoformat()


@dataclass(frozen=True)
class ManagerConfig:
    run_id: str
    artifact_root: Path
    raw_log_root: Path
    working_directory: Path
    official_root: Path
    official_commit: str
    instrumentation_commit: str
    preregistration_path: Path
    preregistration_sha256: str
    runtime_start_path: Path
    runtime_start_sha256: str
    task_map_path: Path
    seed_schedule_path: Path
    seed_schedule_sha256: str
    resolved_config_path: Path
    resolved_config_sha256: str
    checkpoint_path: Path
    dataset_stats_path: Path
    gpu_ids: tuple[int, ...]
    python_executable: str
    runner_path: Path
    task_config: str = "libero_uncond_2cam224_1e-4"
    poll_interval_seconds: float = 0.2


@dataclass
class _RunningTask:
    suite: str
    task_id: int
    gpu_id: int
    process: Any
    log_handle: BinaryIO
    launched_at: str
    command_argv: list[str]
    command_sha256: str
    environment_bindings: dict[str, str]
    environment_sha256: str
    log_path: str


def _drain_started_workers(
    running: dict[str, _RunningTask],
    *,
    sleep: Callable[[float], None],
    poll_interval_seconds: float,
) -> None:
    """Wait for every launched worker and close logs without sending signals."""

    errors: list[str] = []
    for process_id, task in list(running.items()):
        try:
            wait = getattr(task.process, "wait", None)
            if callable(wait):
                wait()
            else:
                while task.process.poll() is None:
                    sleep(poll_interval_seconds)
        except BaseException as exc:
            errors.append(f"{process_id} wait: {type(exc).__name__}: {exc}")
        finally:
            try:
                task.log_handle.close()
            except BaseException as exc:
                errors.append(
                    f"{process_id} log close: {type(exc).__name__}: {exc}"
                )
            running.pop(process_id, None)
    if errors:
        raise ManagerError("worker drain failed: " + "; ".join(errors))


def _load_bound_json(
    path: Path,
    trusted_sha: str,
    label: str,
) -> tuple[dict[str, Any], str, bytes]:
    readback = _read_absolute(path, capture=True)
    if readback["sha256"] != _require_sha(trusted_sha, f"trusted {label} digest"):
        raise ManagerError(f"{label} differs from its trusted file digest")
    payload = _loads_json(readback["bytes"], label)
    if not isinstance(payload, dict):
        raise ManagerError(f"{label} must be a JSON object")
    return payload, str(readback["sha256"]), readback["bytes"]


def _require_normalized_canonical_file(
    raw: bytes,
    normalized: Mapping[str, Any],
    label: str,
) -> None:
    if raw != _canonical_bytes(normalized):
        raise ManagerError(
            f"{label} must be the validator-normalized canonical JSON bytes"
        )


def _assert_no_omegaconf_interpolation(node: Any, *, location: str) -> None:
    """Reject every resolver/interpolation in a supposedly resolved lock."""

    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - caller imports too.
        raise ManagerError("omegaconf is required for resolved config validation") from exc
    if isinstance(node, DictConfig):
        keys: Sequence[str | int] = list(node.keys())
    elif isinstance(node, ListConfig):
        keys = list(range(len(node)))
    else:
        return
    for key in keys:
        child_location = f"{location}.{key}"
        if OmegaConf.is_interpolation(node, key):
            raise ManagerError(
                f"locked resolved config contains interpolation at {child_location}"
            )
        _assert_no_omegaconf_interpolation(
            node._get_node(key),  # noqa: SLF001 - avoids resolving before rejection.
            location=child_location,
        )


def _validate_manager_locked_resolved_config(
    config: ManagerConfig,
    *,
    resolved_raw: bytes,
    seed: int,
) -> None:
    """Re-compose the baseline and every permitted dynamic projection."""

    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except (ImportError, ModuleNotFoundError) as exc:
        raise ManagerError(
            "hydra-core and omegaconf are required for manager config validation"
        ) from exc
    try:
        locked = OmegaConf.create(resolved_raw.decode("utf-8", errors="strict"))
        _assert_no_omegaconf_interpolation(locked, location="locked")
        locked_value = OmegaConf.to_container(
            locked,
            resolve=True,
            throw_on_missing=True,
        )
        if not isinstance(locked_value, dict):
            raise ManagerError("locked resolved config must be a mapping")
        gpu_id = locked_value.get("gpu_id")
        evaluation = locked_value.get("EVALUATION")
        if not isinstance(evaluation, dict):
            raise ManagerError("locked resolved config lacks EVALUATION mapping")
        suite = evaluation.get("task_suite_name")
        task_id = evaluation.get("task_id")
        if type(gpu_id) is not int or not 0 <= gpu_id <= 7:
            raise ManagerError("locked resolved config has an invalid baseline gpu_id")
        if suite not in SUITES:
            raise ManagerError("locked resolved config has an invalid baseline task suite")
        if type(task_id) is not int or not 0 <= task_id < TASKS_PER_SUITE:
            raise ManagerError("locked resolved config has an invalid baseline task_id")
        config_dir = _lexical_absolute(config.official_root) / "configs"
        with initialize_config_dir(
            version_base="1.3",
            config_dir=str(config_dir),
            job_name="mf_wam_g0_manager_preflight",
        ):
            projections = [(gpu_id, suite, task_id)] + [
                (projected_gpu_id, projected_suite, projected_task_id)
                for projected_suite in SUITES
                for projected_task_id in range(TASKS_PER_SUITE)
                for projected_gpu_id in config.gpu_ids
                if (projected_gpu_id, projected_suite, projected_task_id)
                != (gpu_id, suite, task_id)
            ]
            for projected_gpu_id, projected_suite, projected_task_id in projections:
                overrides = build_worker_command(
                    config,
                    suite=projected_suite,
                    task_id=projected_task_id,
                    gpu_id=projected_gpu_id,
                    seed=seed,
                )[2:]
                composed = compose(config_name="sim_libero", overrides=overrides)
                composed_value = OmegaConf.to_container(
                    composed,
                    resolve=True,
                    throw_on_missing=True,
                )
                expected = OmegaConf.create(locked_value)
                OmegaConf.update(
                    expected, "gpu_id", projected_gpu_id, merge=False, force_add=False
                )
                OmegaConf.update(
                    expected,
                    "EVALUATION.task_suite_name",
                    projected_suite,
                    merge=False,
                    force_add=False,
                )
                OmegaConf.update(
                    expected,
                    "EVALUATION.task_id",
                    projected_task_id,
                    merge=False,
                    force_add=False,
                )
                expected_value = OmegaConf.to_container(
                    expected,
                    resolve=True,
                    throw_on_missing=True,
                )
                if composed_value != expected_value:
                    raise ManagerError(
                        "official Hydra config projection differs from locked base; "
                        f"gpu_id={projected_gpu_id}, suite={projected_suite}, "
                        f"task_id={projected_task_id}"
                    )
    except ManagerError:
        raise
    except Exception as exc:
        raise ManagerError(
            f"manager official Hydra config composition failed: {exc}"
        ) from exc


def _validate_config(
    config: ManagerConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes]:
    prereg_raw, _, prereg_bytes = _load_bound_json(
        config.preregistration_path, config.preregistration_sha256, "preregistration"
    )
    start_raw, _, start_bytes = _load_bound_json(
        config.runtime_start_path, config.runtime_start_sha256, "runtime start"
    )
    task_map_readback = _read_absolute(config.task_map_path, capture=True)
    task_map_raw = _loads_json(task_map_readback["bytes"], "task map")
    schedule_raw, _, schedule_bytes = _load_bound_json(
        config.seed_schedule_path, config.seed_schedule_sha256, "seed schedule"
    )
    try:
        prereg = contract.validate_preregistration(prereg_raw)
        task_map = contract.validate_task_map(task_map_raw)
        schedule = contract.validate_seed_schedule(schedule_raw, task_map=task_map)
        start = contract.validate_runtime_start(
            start_raw,
            preregistration=prereg,
            model_cache_root=Path(
                prereg["runtime_environment"]["DIFFSYNTH_MODEL_BASE_PATH"]
            ),
        )
    except contract.ContractError as exc:
        raise ManagerError(f"upstream contract validation failed: {exc}") from exc
    _require_normalized_canonical_file(prereg_bytes, prereg, "preregistration")
    _require_normalized_canonical_file(start_bytes, start, "runtime start")
    _require_normalized_canonical_file(
        task_map_readback["bytes"], task_map, "task map"
    )
    _require_normalized_canonical_file(schedule_bytes, schedule, "seed schedule")
    if (
        contract.task_map_sha256(task_map)
        != prereg["data"]["task_map_canonical_sha256"]
        or schedule["seed"] != prereg["seeds"]["seed"]
        or schedule["python_hash_seed"] != prereg["seeds"]["python_hash_seed"]
        or _canonical_sha(schedule_raw) != prereg["seeds"]["schedule_canonical_sha256"]
    ):
        raise ManagerError("task map or seed schedule differs from preregistration")
    if (
        config.run_id != prereg["run_id"]
        or str(_lexical_absolute(config.artifact_root)) != prereg["output"]["artifact_root"]
        or str(_lexical_absolute(config.working_directory))
        != prereg["launch"]["working_directory"]
        or config.official_commit != prereg["source"]["fastwam"]["commit"]
        or config.instrumentation_commit
        != prereg["source"]["instrumentation"]["commit"]
        or len(config.gpu_ids) != prereg["launch"]["gpu_count"]
    ):
        raise ManagerError("manager configuration differs from preregistration")
    if config.task_config != "libero_uncond_2cam224_1e-4":
        raise ManagerError("task_config differs from the formal G0 task config")
    if (
        not 1 <= len(config.gpu_ids) <= 8
        or list(config.gpu_ids) != sorted(set(config.gpu_ids))
        or any(type(item) is not int or not 0 <= item <= 7 for item in config.gpu_ids)
    ):
        raise ManagerError("gpu_ids must be 1..8 sorted unique integers in [0,7]")
    if not _COMMIT_RE.fullmatch(config.official_commit) or not _COMMIT_RE.fullmatch(
        config.instrumentation_commit
    ):
        raise ManagerError("source commits must be exact 40-hex Git identities")
    resolved = _read_absolute(config.resolved_config_path, capture=True)
    if resolved["sha256"] != _require_sha(
        config.resolved_config_sha256, "resolved config digest"
    ) or (
        resolved["sha256"] != prereg["artifacts"]["resolved_config"]["sha256"]
        or resolved["size_bytes"]
        != prereg["artifacts"]["resolved_config"]["size_bytes"]
    ):
        raise ManagerError("resolved config differs from preregistration")
    for path, artifact, label in (
        (config.checkpoint_path, prereg["artifacts"]["checkpoint"], "checkpoint"),
        (config.dataset_stats_path, prereg["artifacts"]["dataset_stats"], "dataset stats"),
    ):
        observed = _read_absolute(path)
        if observed["sha256"] != artifact["sha256"] or observed["size_bytes"] != artifact["size_bytes"]:
            raise ManagerError(f"{label} differs from preregistration")
    processes = schedule.get("task_processes")
    if not isinstance(processes, list) or len(processes) != EXPECTED_TASKS:
        raise ManagerError("seed schedule does not contain exactly 40 task processes")
    expected_ids = [
        f"{suite}/task{task_id:02d}"
        for suite in SUITES
        for task_id in range(TASKS_PER_SUITE)
    ]
    if [item.get("process_id") for item in processes if isinstance(item, Mapping)] != expected_ids:
        raise ManagerError("seed schedule process order/coverage is not canonical")
    return prereg, start, schedule, resolved["bytes"]


def build_worker_command(
    config: ManagerConfig,
    *,
    suite: str,
    task_id: int,
    gpu_id: int,
    seed: int,
) -> list[str]:
    """Return the exact argv passed directly to ``subprocess.Popen``."""

    return [
        config.python_executable,
        str(_lexical_absolute(config.runner_path)),
        f"task={config.task_config}",
        f"ckpt={_lexical_absolute(config.checkpoint_path)}",
        f"gpu_id={gpu_id}",
        f"seed={seed}",
        f"output_dir={_lexical_absolute(config.artifact_root)}",
        f"EVALUATION.task_suite_name={suite}",
        f"EVALUATION.task_id={task_id}",
        f"EVALUATION.output_dir={_lexical_absolute(config.artifact_root)}",
        f"EVALUATION.dataset_stats_path={_lexical_absolute(config.dataset_stats_path)}",
        "EVALUATION.num_trials=50",
        "EVALUATION.env_num=1",
        "EVALUATION.num_steps_wait=30",
        "EVALUATION.replan_steps=10",
        "EVALUATION.binarize_gripper=true",
        "EVALUATION.use_action_ensembler=false",
        "EVALUATION.visualize_future_video=false",
        "EVALUATION.action_horizon=32",
    ]


def build_worker_environment(
    config: ManagerConfig,
    *,
    preregistration: Mapping[str, Any],
    gpu_id: int,
    python_hash_seed: int,
) -> dict[str, str]:
    runtime = preregistration["runtime_environment"]
    return {
        **FIXED_WORKER_ENVIRONMENT,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "DIFFSYNTH_DOWNLOAD_SOURCE": str(runtime["DIFFSYNTH_DOWNLOAD_SOURCE"]),
        "DIFFSYNTH_MODEL_BASE_PATH": str(runtime["DIFFSYNTH_MODEL_BASE_PATH"]),
        "DIFFSYNTH_SKIP_DOWNLOAD": str(runtime["DIFFSYNTH_SKIP_DOWNLOAD"]),
        "LOCAL_RANK": "0",
        "MF_WAM_G0_PREREG_PATH": str(_lexical_absolute(config.preregistration_path)),
        "MF_WAM_G0_PREREG_SHA256": config.preregistration_sha256,
        "MF_WAM_G0_RESOLVED_CONFIG_PATH": str(
            _lexical_absolute(config.resolved_config_path)
        ),
        "MF_WAM_G0_RESOLVED_CONFIG_SHA256": config.resolved_config_sha256,
        "MF_WAM_G0_RUN_ID": config.run_id,
        "MF_WAM_G0_RUNTIME_START_PATH": str(_lexical_absolute(config.runtime_start_path)),
        "MF_WAM_G0_RUNTIME_START_SHA256": config.runtime_start_sha256,
        "MF_WAM_G0_SEED_SCHEDULE_PATH": str(_lexical_absolute(config.seed_schedule_path)),
        "MF_WAM_G0_SEED_SCHEDULE_SHA256": config.seed_schedule_sha256,
        "MF_WAM_INSTRUMENTATION_COMMIT": config.instrumentation_commit,
        "MF_WAM_OFFICIAL_COMMIT": config.official_commit,
        "MF_WAM_OFFICIAL_ROOT": str(_lexical_absolute(config.official_root)),
        "MUJOCO_GL": str(runtime["MUJOCO_GL"]),
        "PYOPENGL_PLATFORM": str(runtime["PYOPENGL_PLATFORM"]),
        "PYTHONHASHSEED": str(python_hash_seed),
        "RANK": "0",
        "WORLD_SIZE": "1",
    }


def _artifact_reference(root: Path, relative: str, *, role: str | None = None) -> dict[str, Any]:
    observed = _read_relative(root, relative)
    result = {
        "path": relative,
        "sha256": observed["sha256"],
        "size_bytes": observed["size_bytes"],
    }
    if role is not None:
        result["role"] = role
    return result


def _successful_worker_artifacts(
    config: ManagerConfig,
    *,
    suite: str,
    task_id: int,
    gpu_id: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = _artifact_reference(
        config.artifact_root,
        f"results/{suite}/task{task_id:02d}.json",
    )
    receipt_relative = f"trace_receipts/{suite}/task{task_id:02d}.json"
    receipt = _artifact_reference(config.artifact_root, receipt_relative)
    receipt_readback = _read_relative(config.artifact_root, receipt_relative, capture=True)
    payload = _loads_json(receipt_readback["bytes"], receipt_relative)
    if (
        not isinstance(payload, Mapping)
        or payload.get("kind") != "mf_wam_g0_task_trace_receipt"
        or payload.get("run_id") != config.run_id
        or payload.get("process_id") != f"{suite}/task{task_id:02d}"
        or payload.get("episode_count") != TRIALS_PER_TASK
        or not isinstance(payload.get("traces"), list)
        or len(payload["traces"]) != TRIALS_PER_TASK
    ):
        raise ManagerError(f"task receipt is incomplete: {suite}/task{task_id:02d}")
    trace_rows: list[dict[str, Any]] = []
    for trial_idx, raw in enumerate(payload["traces"]):
        expected_path = f"traces/{suite}/task{task_id:02d}/trial{trial_idx:03d}.json"
        if not isinstance(raw, Mapping) or raw.get("trial_idx") != trial_idx or raw.get("path") != expected_path:
            raise ManagerError(f"task receipt trace order/path mismatch: {expected_path}")
        actual = _artifact_reference(config.artifact_root, expected_path)
        expected = {key: raw.get(key) for key in ("path", "sha256", "size_bytes")}
        if actual != expected:
            raise ManagerError(f"task receipt trace content mismatch: {expected_path}")
        trace_rows.append({"trial_idx": trial_idx, **actual})
    trace_tree = _tree_sha(trace_rows)
    if payload.get("tree_sha256") != trace_tree:
        raise ManagerError(f"task receipt trace tree mismatch: {suite}/task{task_id:02d}")
    receipt.update({"tree_sha256": trace_tree, "episode_count": TRIALS_PER_TASK})
    raw_source = f"{suite}/gpu{gpu_id}_task{task_id}_results.json"
    raw = _artifact_reference(config.artifact_root, raw_source)
    if raw["sha256"] != result["sha256"] or raw["size_bytes"] != result["size_bytes"]:
        raise ManagerError(f"raw/canonical task result mismatch: {suite}/task{task_id:02d}")
    raw_result = {
        "source_path": raw_source,
        "archive_path": f"official/{suite}/gpu{gpu_id}_task{task_id}_results.json",
        "sha256": raw["sha256"],
        "size_bytes": raw["size_bytes"],
    }
    return result, receipt, raw_result


def _validate_worker_terminal_log(
    config: ManagerConfig,
    *,
    log_relative: str,
    suite: str,
    task_id: int,
    environment_sha256: str,
) -> dict[str, Any]:
    readback = _read_relative(config.raw_log_root, log_relative, capture=True)
    try:
        lines = [line for line in readback["bytes"].splitlines() if line.strip()]
    except AttributeError as exc:
        raise ManagerError("worker log readback is unavailable") from exc
    if not lines:
        raise ManagerError("worker log is empty")
    payload = _loads_json(lines[-1], f"{log_relative} terminal line")
    if not isinstance(payload, Mapping) or set(payload) != WORKER_TERMINAL_KEYS:
        raise ManagerError("worker terminal line has an invalid schema")
    expected_receipt = str(
        _lexical_absolute(
            config.artifact_root
            / f"trace_receipts/{suite}/task{task_id:02d}.json"
        )
    )
    sources = payload.get("terminal_source_identities")
    official_source = sources.get("official") if isinstance(sources, Mapping) else None
    instrumentation_source = (
        sources.get("instrumentation") if isinstance(sources, Mapping) else None
    )
    if (
        payload.get("status") != "PASS"
        or payload.get("kind") != "mf_wam_g0_traced_worker_terminal"
        or payload.get("run_id") != config.run_id
        or payload.get("official_commit") != config.official_commit
        or payload.get("process_receipt") != expected_receipt
        or payload.get("official_result_type") != "dict"
        or payload.get("environment_sha256") != environment_sha256
        or payload.get("external_prelaunch_commit_tree_gate_required") is not True
        or not isinstance(sources, Mapping)
        or sources.get("status") != "PASS"
        or not isinstance(official_source, Mapping)
        or official_source.get("commit") != config.official_commit
        or not isinstance(instrumentation_source, Mapping)
        or instrumentation_source.get("commit") != config.instrumentation_commit
    ):
        raise ManagerError("worker terminal line does not prove traced completion")
    result_receipt = payload.get("official_result_receipt")
    if not isinstance(result_receipt, Mapping):
        raise ManagerError("worker terminal result receipt is absent")
    return dict(payload)


def _bind_worker_terminal_artifacts(
    terminal: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> None:
    returned = terminal["official_result_receipt"]
    expected = {
        "path": result["path"],
        "sha256": result["sha256"],
        "size_bytes": result["size_bytes"],
        "source_path": raw_result["source_path"],
        "source_sha256": raw_result["sha256"],
        "source_size_bytes": raw_result["size_bytes"],
    }
    if any(returned.get(key) != value for key, value in expected.items()):
        raise ManagerError("worker terminal result receipt differs from task artifacts")


def _status_payload(
    config: ManagerConfig,
    *,
    suite: str,
    task_id: int,
    gpu_id: int | None,
    state: str,
    launched_at: str | None,
    completed_at: str,
    exit_code: int | None,
    failure_reason: str | None,
    command_argv: list[str] | None,
    environment_bindings: dict[str, str] | None,
    log: dict[str, Any] | None,
    canonical_result: dict[str, Any] | None,
    trace_receipt: dict[str, Any] | None,
    raw_result: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": "mf_wam_g0_manager_task_status",
        "run_id": config.run_id,
        "process_id": f"{suite}/task{task_id:02d}",
        "task_suite": suite,
        "task_id": task_id,
        "gpu_id": gpu_id,
        "state": state,
        "launched_at": launched_at,
        "completed_at": completed_at,
        "exit_code": exit_code,
        "complete": state == "SUCCEEDED",
        "failure_reason": failure_reason,
        "command_argv": command_argv,
        "command_sha256": _canonical_sha(command_argv) if command_argv is not None else None,
        "environment_bindings": environment_bindings,
        "environment_sha256": (
            _canonical_sha(environment_bindings)
            if environment_bindings is not None
            else None
        ),
        "log": log,
        "canonical_result": canonical_result,
        "trace_receipt": trace_receipt,
        "raw_result": raw_result,
    }
    if set(payload) != STATUS_KEYS:
        raise AssertionError("internal manager status schema mismatch")
    return payload


def _manifest_task(status: Mapping[str, Any], status_reference: Mapping[str, Any]) -> dict[str, Any]:
    def field(reference: Any, name: str) -> Any:
        return reference.get(name) if isinstance(reference, Mapping) else None

    return {
        "process_id": status["process_id"],
        "task_suite": status["task_suite"],
        "task_id": status["task_id"],
        "gpu_id": status["gpu_id"],
        "state": status["state"],
        "launched_at": status["launched_at"],
        "completed_at": status["completed_at"],
        "exit_code": status["exit_code"],
        "complete": status["complete"],
        "failure_reason": status["failure_reason"],
        "command_sha256": status["command_sha256"],
        "environment_sha256": status["environment_sha256"],
        "log_path": field(status["log"], "path"),
        "log_sha256": field(status["log"], "sha256"),
        "log_size_bytes": field(status["log"], "size_bytes"),
        "status_path": status_reference["path"],
        "status_sha256": status_reference["sha256"],
        "status_size_bytes": status_reference["size_bytes"],
        "result_path": field(status["canonical_result"], "path"),
        "result_sha256": field(status["canonical_result"], "sha256"),
        "result_size_bytes": field(status["canonical_result"], "size_bytes"),
        "trace_receipt_path": field(status["trace_receipt"], "path"),
        "trace_receipt_sha256": field(status["trace_receipt"], "sha256"),
        "trace_receipt_size_bytes": field(status["trace_receipt"], "size_bytes"),
        "trace_tree_sha256": field(status["trace_receipt"], "tree_sha256"),
        "episode_count": field(status["trace_receipt"], "episode_count") or 0,
        "raw_result_source_path": field(status["raw_result"], "source_path"),
        "raw_result_archive_path": field(status["raw_result"], "archive_path"),
        "raw_result_sha256": field(status["raw_result"], "sha256"),
        "raw_result_size_bytes": field(status["raw_result"], "size_bytes"),
    }


def _publish_status(config: ManagerConfig, payload: dict[str, Any]) -> dict[str, Any]:
    relative = f"status/{payload['task_suite']}/task{payload['task_id']:02d}.json"
    receipt = _publish_json_no_replace(config.raw_log_root / relative, payload)
    return {"path": relative, **receipt}


def _renameat2_no_replace_at(
    source_fd: int,
    source_name: str,
    target_fd: int,
    target_name: str,
) -> None:
    """Rename one directory entry atomically without replacing its target."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ManagerError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_fd,
        os.fsencode(source_name),
        target_fd,
        os.fsencode(target_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), target_name)
    raise ManagerError(
        f"renameat2(RENAME_NOREPLACE) failed for {source_name}: "
        f"{os.strerror(error)}"
    )


def _rename_directory_no_replace(
    source_parent: Path,
    source_name: str,
    target_parent: Path,
) -> None:
    source_fd = _open_absolute(source_parent, directory=True)
    target_fd = _open_absolute(target_parent, directory=True)
    try:
        metadata = os.stat(source_name, dir_fd=source_fd, follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ManagerError(f"raw suite source is not a directory: {source_name}")
        try:
            _renameat2_no_replace_at(
                source_fd,
                source_name,
                target_fd,
                source_name,
            )
        except FileExistsError as exc:
            raise ManagerError(
                f"refusing to overwrite raw suite archive: {source_name}"
            ) from exc
        os.fsync(source_fd)
        os.fsync(target_fd)
    finally:
        os.close(source_fd)
        os.close(target_fd)


def _probe_renameat2_no_replace(source_parent: Path, target_parent: Path) -> None:
    """Prove rename success and no-replace semantics on the launch filesystem."""

    source_fd = _open_absolute(source_parent, directory=True)
    target_root_fd = _open_absolute(target_parent, directory=True)
    token = f"{_RENAME_PROBE_PREFIX}{os.getpid()}-{time.monotonic_ns()}"
    moved_name = f"{token}-moved"
    collision_name = f"{token}-collision"
    target_directory_name = f"{token}-target"
    target_fd: int | None = None
    source_entries: set[str] = set()
    target_entries: set[str] = set()
    target_directory_created = False
    failure: BaseException | None = None
    cleanup_errors: list[str] = []
    try:
        os.mkdir(moved_name, 0o700, dir_fd=source_fd)
        source_entries.add(moved_name)
        os.mkdir(collision_name, 0o700, dir_fd=source_fd)
        source_entries.add(collision_name)
        os.mkdir(target_directory_name, 0o700, dir_fd=target_root_fd)
        target_directory_created = True
        target_fd = os.open(
            target_directory_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=target_root_fd,
        )

        moved_identity = os.stat(
            moved_name,
            dir_fd=source_fd,
            follow_symlinks=False,
        )
        _renameat2_no_replace_at(
            source_fd,
            moved_name,
            target_fd,
            moved_name,
        )
        source_entries.remove(moved_name)
        target_entries.add(moved_name)
        moved_target = os.stat(
            moved_name,
            dir_fd=target_fd,
            follow_symlinks=False,
        )
        if (moved_target.st_dev, moved_target.st_ino) != (
            moved_identity.st_dev,
            moved_identity.st_ino,
        ):
            raise ManagerError("renameat2 probe changed the moved directory identity")

        os.mkdir(collision_name, 0o700, dir_fd=target_fd)
        target_entries.add(collision_name)
        source_collision = os.stat(
            collision_name,
            dir_fd=source_fd,
            follow_symlinks=False,
        )
        target_collision = os.stat(
            collision_name,
            dir_fd=target_fd,
            follow_symlinks=False,
        )
        try:
            _renameat2_no_replace_at(
                source_fd,
                collision_name,
                target_fd,
                collision_name,
            )
        except FileExistsError:
            pass
        else:
            source_entries.discard(collision_name)
            raise ManagerError("renameat2 probe replaced an existing target")
        source_after = os.stat(
            collision_name,
            dir_fd=source_fd,
            follow_symlinks=False,
        )
        target_after = os.stat(
            collision_name,
            dir_fd=target_fd,
            follow_symlinks=False,
        )
        if (
            (source_after.st_dev, source_after.st_ino)
            != (source_collision.st_dev, source_collision.st_ino)
            or (target_after.st_dev, target_after.st_ino)
            != (target_collision.st_dev, target_collision.st_ino)
        ):
            raise ManagerError("renameat2 probe did not preserve collision identities")
        os.fsync(source_fd)
        os.fsync(target_fd)
        os.fsync(target_root_fd)
    except BaseException as exc:
        failure = exc
    finally:
        if target_fd is not None:
            for name in sorted(target_entries):
                try:
                    os.rmdir(name, dir_fd=target_fd)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    cleanup_errors.append(f"target entry {name}: {exc}")
            os.close(target_fd)
        for name in sorted(source_entries):
            try:
                os.rmdir(name, dir_fd=source_fd)
            except FileNotFoundError:
                pass
            except OSError as exc:
                cleanup_errors.append(f"source entry {name}: {exc}")
        if target_directory_created:
            try:
                os.rmdir(target_directory_name, dir_fd=target_root_fd)
            except FileNotFoundError:
                pass
            except OSError as exc:
                cleanup_errors.append(
                    f"target directory {target_directory_name}: {exc}"
                )
        for descriptor, label in (
            (source_fd, "source parent"),
            (target_root_fd, "target parent"),
        ):
            try:
                os.fsync(descriptor)
            except OSError as exc:
                cleanup_errors.append(f"{label} fsync: {exc}")
        for descriptor, name in (
            (source_fd, moved_name),
            (source_fd, collision_name),
            (target_root_fd, target_directory_name),
        ):
            try:
                os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            except OSError as exc:
                cleanup_errors.append(f"cleanup readback {name}: {exc}")
            else:
                cleanup_errors.append(f"cleanup readback still finds {name}")
        os.close(source_fd)
        os.close(target_root_fd)
    if cleanup_errors:
        cleanup_error = ManagerError(
            "renameat2 probe cleanup failed: " + "; ".join(cleanup_errors)
        )
        if failure is not None:
            raise cleanup_error from failure
        raise cleanup_error
    if failure is not None:
        raise failure


def _expected_input_paths() -> set[str]:
    result: set[str] = set()
    for suite in SUITES:
        for task_id in range(TASKS_PER_SUITE):
            result.add(f"results/{suite}/task{task_id:02d}.json")
            result.add(f"trace_receipts/{suite}/task{task_id:02d}.json")
            for trial_idx in range(TRIALS_PER_TASK):
                result.add(f"traces/{suite}/task{task_id:02d}/trial{trial_idx:03d}.json")
    return result


def _input_inventory(root: Path) -> list[dict[str, Any]]:
    expected = _expected_input_paths()
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    identities: set[tuple[int, int]] = set()
    root = _lexical_absolute(root)
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        identity = (metadata.st_dev, metadata.st_ino)
        if stat.S_ISLNK(metadata.st_mode):
            raise ManagerError(f"symlink is forbidden in artifact root: {relative}")
        if identity in identities:
            raise ManagerError(f"inode alias is forbidden: {relative}")
        identities.add(identity)
        if stat.S_ISDIR(metadata.st_mode):
            observed_directories.add(relative)
        elif stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise ManagerError(f"hardlink is forbidden: {relative}")
            observed_files.add(relative)
        else:
            raise ManagerError(f"non-regular artifact is forbidden: {relative}")
    if observed_files != expected:
        raise ManagerError(
            "canonical input scope mismatch: "
            f"missing={sorted(expected-observed_files)[:5]}, "
            f"extra={sorted(observed_files-expected)[:5]}"
        )
    allowed_directories = {"results", "trace_receipts", "traces"}
    for suite in SUITES:
        allowed_directories.update(
            {f"results/{suite}", f"trace_receipts/{suite}", f"traces/{suite}"}
        )
        for task_id in range(TASKS_PER_SUITE):
            allowed_directories.add(f"traces/{suite}/task{task_id:02d}")
    if observed_directories != allowed_directories:
        raise ManagerError("canonical input directory scope mismatch")
    inventory = []
    for relative in sorted(expected, key=lambda value: value.encode("utf-8")):
        role = (
            "task_result"
            if relative.startswith("results/")
            else "task_trace_receipt"
            if relative.startswith("trace_receipts/")
            else "episode_trace"
        )
        inventory.append(_artifact_reference(root, relative, role=role))
    return inventory


def _require_empty_directory(path: Path, label: str) -> None:
    descriptor = _open_absolute(path, directory=True)
    try:
        entries = sorted(os.listdir(descriptor))
    finally:
        os.close(descriptor)
    if entries:
        raise ManagerError(f"{label} must be empty before launch: {entries[:5]!r}")


def _run_git_readonly(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                f"safe.directory={root}",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-C",
                str(root),
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=_GIT_ENVIRONMENT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ManagerError(f"cannot inspect Git checkout {root}: {exc}") from exc
    return completed.stdout


def _run_git_readonly_bytes(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                f"safe.directory={root}",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-C",
                str(root),
                *arguments,
            ],
            check=True,
            capture_output=True,
            timeout=30,
            env=_GIT_ENVIRONMENT,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ManagerError(f"cannot read Git object from {root}: {exc}") from exc


def _git_ignored_entries(root: Path, label: str) -> tuple[str, ...]:
    raw = _run_git_readonly_bytes(
        root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
    )
    if raw and not raw.endswith(b"\0"):
        raise ManagerError(f"{label} ignored-file inventory is not NUL terminated")
    fields = raw[:-1].split(b"\0") if raw else []
    entries: list[str] = []
    observed: set[str] = set()
    for field in fields:
        try:
            relative = field.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ManagerError(
                f"{label} ignored-file path is not strict UTF-8"
            ) from exc
        relative = _safe_relative(relative, f"{label} ignored-file path")
        if relative in observed:
            raise ManagerError(f"{label} ignored-file inventory contains duplicates")
        observed.add(relative)
        entries.append(relative)
    return tuple(entries)


def _verify_git_checkout_policy(root: Path, label: str) -> None:
    expected_git_dir = _lexical_absolute(root) / ".git"
    git_dir_fd = _open_absolute(expected_git_dir, directory=True)
    os.close(git_dir_fd)
    local_config_keys = _run_git_readonly(
        root, "config", "--local", "--no-includes", "--name-only", "--list"
    ).splitlines()
    forbidden_config = [
        key
        for key in local_config_keys
        if key.lower().startswith(("filter.", "include.", "includeif."))
        or key.lower() == "core.attributesfile"
    ]
    if forbidden_config:
        raise ManagerError(
            f"{label} repository-local filters/includes are forbidden: "
            f"{forbidden_config[:5]!r}"
        )
    info_fd = _open_absolute(expected_git_dir / "info", directory=True)
    try:
        try:
            os.stat("attributes", dir_fd=info_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ManagerError(f"{label} .git/info/attributes is forbidden")
    finally:
        os.close(info_fd)
    top_level = _run_git_readonly(root, "rev-parse", "--show-toplevel").strip()
    if top_level != str(_lexical_absolute(root)):
        raise ManagerError(f"{label} Git top-level differs from its source root")
    absolute_git_dir = _run_git_readonly(
        root, "rev-parse", "--absolute-git-dir"
    ).strip()
    common_git_dir = _run_git_readonly(
        root, "rev-parse", "--path-format=absolute", "--git-common-dir"
    ).strip()
    if absolute_git_dir != str(expected_git_dir) or common_git_dir != str(
        expected_git_dir
    ):
        raise ManagerError(
            f"{label} linked worktrees or external Git directories are forbidden"
        )
    if _run_git_readonly(root, "rev-parse", "--show-object-format").strip() != "sha1":
        raise ManagerError(f"{label} Git object format must be sha1")
    replacements = _run_git_readonly(
        root, "for-each-ref", "--format=%(refname)", "refs/replace/"
    ).splitlines()
    if replacements:
        raise ManagerError(f"{label} Git replace refs are forbidden: {replacements[:5]!r}")
    ignored = _git_ignored_entries(root, label)
    if ignored:
        raise ManagerError(
            f"{label} source root contains gitignored artifacts: {ignored[:5]!r}"
        )


def _verify_critical_git_sources(
    root: Path,
    expected_commit: str,
    critical_paths: Sequence[str],
    label: str,
) -> None:
    root = _lexical_absolute(root)
    for relative in critical_paths:
        marker = _run_git_readonly(root, "ls-files", "-v", "--", relative).strip()
        if marker != f"H {relative}":
            raise ManagerError(
                f"{label} critical source has assume-unchanged/skip-worktree "
                f"or is untracked: {relative}: {marker!r}"
            )
        worktree = _read_relative(root, relative, capture=True)
        git_blob = _run_git_readonly_bytes(
            root,
            "cat-file",
            "blob",
            f"{expected_commit}:{relative}",
        )
        if worktree["bytes"] != git_blob:
            raise ManagerError(
                f"{label} critical source differs from exact commit blob: {relative}"
            )


def _verify_exact_commit_tree(root: Path, expected_commit: str, label: str) -> None:
    raw_tree = _run_git_readonly_bytes(
        root, "ls-tree", "-r", "-z", "--full-tree", expected_commit
    )
    if raw_tree and not raw_tree.endswith(b"\0"):
        raise ManagerError(f"{label} Git tree inventory is not NUL terminated")
    observed_paths: set[str] = set()
    for record in raw_tree[:-1].split(b"\0") if raw_tree else []:
        try:
            metadata, path_raw = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.split(b" ", 2)
            relative = path_raw.decode("utf-8", errors="strict")
        except (ValueError, UnicodeError) as exc:
            raise ManagerError(f"{label} Git tree record is malformed") from exc
        relative = _safe_relative(relative, f"{label} tracked source path")
        if relative in observed_paths:
            raise ManagerError(f"{label} Git tree contains duplicate paths")
        observed_paths.add(relative)
        if mode not in (b"100644", b"100755") or object_type != b"blob":
            raise ManagerError(
                f"{label} symlink, gitlink, or non-regular tracked source is forbidden: "
                f"{relative}"
            )
        try:
            object_id_text = object_id.decode("ascii", errors="strict")
        except UnicodeError as exc:
            raise ManagerError(f"{label} Git blob identity is malformed") from exc
        if not _COMMIT_RE.fullmatch(object_id_text):
            raise ManagerError(f"{label} Git blob identity is malformed")
        worktree = _read_relative(root, relative, capture=True)
        observed_object_id = hashlib.sha1(
            f"blob {len(worktree['bytes'])}\0".encode("ascii")
            + worktree["bytes"]
        ).hexdigest()
        executable = bool(worktree["mode"] & 0o111)
        if observed_object_id != object_id_text or executable != (mode == b"100755"):
            raise ManagerError(
                f"{label} tracked worktree differs from exact commit tree: {relative}"
            )
    if not observed_paths:
        raise ManagerError(f"{label} exact commit tree is empty")


def _verify_clean_git_root(root: Path, expected_commit: str, label: str) -> None:
    root = _lexical_absolute(root)
    descriptor = _open_absolute(root, directory=True)
    os.close(descriptor)
    _verify_git_checkout_policy(root, label)
    head = _run_git_readonly(root, "rev-parse", "HEAD").strip()
    if head != expected_commit:
        raise ManagerError(
            f"{label} Git HEAD mismatch: expected {expected_commit}, observed {head}"
        )
    tree = _run_git_readonly(root, "rev-parse", "HEAD^{tree}").strip()
    porcelain = _run_git_readonly(
        root, "status", "--porcelain", "--untracked-files=all"
    )
    if porcelain:
        raise ManagerError(f"{label} Git checkout is not clean: {porcelain.splitlines()!r}")
    index_markers = _run_git_readonly(root, "ls-files", "-v").splitlines()
    flagged = [line for line in index_markers if not line.startswith("H ")]
    if flagged:
        raise ManagerError(
            f"{label} Git index contains assume-unchanged/skip-worktree flags: "
            f"{flagged[:5]!r}"
        )
    _verify_exact_commit_tree(root, expected_commit, label)
    _verify_git_checkout_policy(root, label)
    if (
        _run_git_readonly(root, "rev-parse", "HEAD").strip() != head
        or _run_git_readonly(root, "rev-parse", "HEAD^{tree}").strip() != tree
        or _run_git_readonly(
            root, "status", "--porcelain", "--untracked-files=all"
        )
        or any(
            not line.startswith("H ")
            for line in _run_git_readonly(root, "ls-files", "-v").splitlines()
        )
    ):
        raise ManagerError(f"{label} Git identity changed during prelaunch readback")


def _canonical_python_executable(value: str) -> str:
    python_candidate = Path(value)
    if not python_candidate.is_absolute():
        raise ManagerError("python_executable must be an absolute path")
    python_candidate = _lexical_absolute(python_candidate)
    try:
        resolved_python = python_candidate.resolve(strict=True)
    except OSError as exc:
        raise ManagerError(f"python_executable cannot be resolved: {exc}") from exc
    python_name = PurePosixPath(resolved_python).name
    if not _PYTHON_EXECUTABLE_RE.fullmatch(python_name):
        raise ManagerError("python_executable does not satisfy the sealed runner contract")
    python_fd = _open_absolute(resolved_python, directory=False)
    os.close(python_fd)
    if not os.access(resolved_python, os.X_OK):
        raise ManagerError("python_executable is not executable")
    return str(resolved_python)


def _verify_prelaunch_sources(config: ManagerConfig) -> str:
    resolved_python = _canonical_python_executable(config.python_executable)

    runner = _lexical_absolute(config.runner_path)
    if runner.parent.name != "scripts" or runner.name != "run_mf_wam_g0_traced.py":
        raise ManagerError("runner_path must select scripts/run_mf_wam_g0_traced.py")
    runner_readback = _read_absolute(runner)
    if runner_readback["size_bytes"] <= 0:
        raise ManagerError("traced runner is empty")
    instrumentation_root = runner.parent.parent
    if instrumentation_root != _lexical_absolute(config.working_directory):
        raise ManagerError("runner instrumentation root differs from working_directory")
    _verify_clean_git_root(config.official_root, config.official_commit, "official")
    _verify_critical_git_sources(
        config.official_root,
        config.official_commit,
        _OFFICIAL_CRITICAL_PATHS,
        "official",
    )
    _verify_clean_git_root(
        instrumentation_root,
        config.instrumentation_commit,
        "instrumentation",
    )
    _verify_critical_git_sources(
        instrumentation_root,
        config.instrumentation_commit,
        _INSTRUMENTATION_CRITICAL_PATHS,
        "instrumentation",
    )
    return resolved_python


def _preflight_roots(config: ManagerConfig) -> None:
    artifact_root = _lexical_absolute(config.artifact_root)
    raw_log_root = _lexical_absolute(config.raw_log_root)
    if (
        artifact_root == raw_log_root
        or artifact_root in raw_log_root.parents
        or raw_log_root in artifact_root.parents
    ):
        raise ManagerError("artifact_root and raw_log_root must be disjoint directories")
    artifact_fd = _open_absolute(config.artifact_root, directory=True)
    raw_parent_fd = _open_absolute(raw_log_root.parent, directory=True)
    try:
        if os.fstat(artifact_fd).st_dev != os.fstat(raw_parent_fd).st_dev:
            raise ManagerError(
                "artifact_root and raw_log_root parent must be on the same filesystem"
            )
        try:
            raw_metadata = os.stat(
                raw_log_root.name,
                dir_fd=raw_parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            raw_metadata = None
        if raw_metadata is not None:
            if not stat.S_ISDIR(raw_metadata.st_mode):
                raise ManagerError("raw_log_root exists and is not a directory")
            _require_empty_directory(raw_log_root, "raw_log_root")
    finally:
        os.close(artifact_fd)
        os.close(raw_parent_fd)
    _require_empty_directory(artifact_root, "artifact_root")
    _probe_renameat2_no_replace(artifact_root, raw_log_root.parent)
    _require_empty_directory(artifact_root, "artifact_root after renameat2 probe")


def _create_roots(config: ManagerConfig) -> None:
    _mkdir_absolute(config.raw_log_root)
    for relative in ("logs", "status", "official"):
        _mkdir_relative(config.raw_log_root, relative)
    for suite in SUITES:
        _mkdir_relative(config.raw_log_root, f"logs/{suite}")
        _mkdir_relative(config.raw_log_root, f"status/{suite}")


def _publish_failure_receipt(
    config: ManagerConfig,
    *,
    reason: str,
    pending: Sequence[tuple[str, int]],
    task_rows: dict[str, dict[str, Any]],
    now: Callable[[], dt.datetime],
) -> dict[str, Any]:
    for suite, task_id in pending:
        process_id = f"{suite}/task{task_id:02d}"
        payload = _status_payload(
            config,
            suite=suite,
            task_id=task_id,
            gpu_id=None,
            state="NOT_LAUNCHED",
            launched_at=None,
            completed_at=_timestamp(now),
            exit_code=None,
            failure_reason=f"not launched after prior failure: {reason}",
            command_argv=None,
            environment_bindings=None,
            log=None,
            canonical_result=None,
            trace_receipt=None,
            raw_result=None,
        )
        status_reference = _publish_status(config, payload)
        task_rows[process_id] = _manifest_task(payload, status_reference)
    try:
        ordered = [
            task_rows[f"{suite}/task{task_id:02d}"]
            for suite in SUITES
            for task_id in range(TASKS_PER_SUITE)
        ]
    except KeyError as exc:
        raise ManagerError("failure receipt task coverage is incomplete") from exc
    failure = {
        "schema_version": 1,
        "kind": "mf_wam_g0_manager_failure_receipt",
        "run_id": config.run_id,
        "completed_at": _timestamp(now),
        "manager_exit_code": 1,
        "failure_reason": reason,
        "artifact_root": str(_lexical_absolute(config.artifact_root)),
        "raw_log_root": str(_lexical_absolute(config.raw_log_root)),
        "gpu_ids": list(config.gpu_ids),
        "task_processes": ordered,
    }
    failure_path = config.raw_log_root / "manager_failure.json"
    receipt = _publish_json_no_replace(failure_path, failure)
    return {
        "status": "FAILED",
        "failure_reason": reason,
        "failure_receipt_path": str(failure_path),
        "failure_receipt_sha256": receipt["sha256"],
        "terminal_manifest_written": False,
    }


def run_manager(
    config: ManagerConfig,
    *,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run under a source monitor installed before any source validation."""

    with _SourceMutationGuard(
        (config.official_root, config.working_directory)
    ) as source_guard:
        source_guard.checkpoint("before validation")
        prereg, _start, schedule, resolved_raw = _validate_config(config)
        resolved_python = _verify_prelaunch_sources(config)
        config = replace(config, python_executable=resolved_python)
        instrumentation_source = source_guard.bind_critical_sources(config)
        instrumentation_fd = _create_sealed_instrumentation_memfd(
            instrumentation_source
        )
        try:
            _validate_manager_locked_resolved_config(
                config,
                resolved_raw=resolved_raw,
                seed=schedule["seed"],
            )
            source_guard.checkpoint("after validation")
            return _run_manager_monitored(
                config,
                prereg=prereg,
                schedule=schedule,
                source_guard=source_guard,
                instrumentation_fd=instrumentation_fd,
                popen_factory=popen_factory,
                now=now,
                sleep=sleep,
            )
        finally:
            os.close(instrumentation_fd)


def _run_manager_monitored(
    config: ManagerConfig,
    *,
    prereg: Mapping[str, Any],
    schedule: Mapping[str, Any],
    source_guard: _SourceMutationGuard,
    instrumentation_fd: int,
    popen_factory: Callable[..., Any],
    now: Callable[[], dt.datetime],
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    """Execute the schedule while the caller owns the live source guard."""

    _preflight_roots(config)
    _create_roots(config)
    terminal_path = config.raw_log_root / "manager_terminal.json"
    failure_path = config.raw_log_root / "manager_failure.json"
    for output in (terminal_path, failure_path):
        try:
            descriptor = _open_absolute(output, directory=False)
        except ManagerError:
            continue
        else:
            os.close(descriptor)
            raise ManagerError(f"manager output already exists: {output}")

    schedule_by_id = {
        item["process_id"]: item for item in schedule["task_processes"]
    }
    pending = [
        (suite, task_id)
        for suite in SUITES
        for task_id in range(TASKS_PER_SUITE)
    ]
    available = list(config.gpu_ids)
    running: dict[str, _RunningTask] = {}
    task_rows: dict[str, dict[str, Any]] = {}
    failed = False
    manager_failure_reason: str | None = None

    try:
        while pending or running:
            # Reap every observed completion before considering any new dispatch.
            # After a completion, loop once more before refilling a free GPU so a
            # sibling failure that becomes observable on the next poll wins the
            # dispatch race deterministically.
            completed_any = False
            for process_id, task in list(running.items()):
                source_guard.checkpoint(f"before reap {process_id}")
                exit_code = task.process.poll()
                if exit_code is None:
                    continue
                source_guard.checkpoint(f"after reap {process_id}")
                completed_any = True
                task.log_handle.close()
                completed_at = _timestamp(now)
                log = _artifact_reference(config.raw_log_root, task.log_path)
                state = "SUCCEEDED"
                failure_reason = None
                result = receipt = raw_result = None
                if exit_code == 0:
                    try:
                        worker_terminal = _validate_worker_terminal_log(
                            config,
                            log_relative=task.log_path,
                            suite=task.suite,
                            task_id=task.task_id,
                            environment_sha256=task.environment_sha256,
                        )
                        result, receipt, raw_result = _successful_worker_artifacts(
                            config,
                            suite=task.suite,
                            task_id=task.task_id,
                            gpu_id=task.gpu_id,
                        )
                        _bind_worker_terminal_artifacts(
                            worker_terminal,
                            result=result,
                            raw_result=raw_result,
                        )
                    except ManagerError as exc:
                        state = "FAILED"
                        failure_reason = f"successful child has invalid artifacts: {exc}"
                else:
                    state = "FAILED"
                    failure_reason = f"child exited with code {exit_code}"
                if state != "SUCCEEDED":
                    failed = True
                    if manager_failure_reason is None:
                        manager_failure_reason = failure_reason
                payload = _status_payload(
                    config,
                    suite=task.suite,
                    task_id=task.task_id,
                    gpu_id=task.gpu_id,
                    state=state,
                    launched_at=task.launched_at,
                    completed_at=completed_at,
                    exit_code=int(exit_code),
                    failure_reason=failure_reason,
                    command_argv=task.command_argv,
                    environment_bindings=task.environment_bindings,
                    log=log,
                    canonical_result=result,
                    trace_receipt=receipt,
                    raw_result=raw_result,
                )
                status_reference = _publish_status(config, payload)
                task_rows[process_id] = _manifest_task(payload, status_reference)
                del running[process_id]
                available.append(task.gpu_id)
                available.sort()
            if failed and not running:
                break
            if completed_any:
                continue
            if failed:
                if running:
                    sleep(config.poll_interval_seconds)
                continue

            while pending and available:
                suite, task_id = pending.pop(0)
                gpu_id = available.pop(0)
                process_id = f"{suite}/task{task_id:02d}"
                scheduled = schedule_by_id[process_id]
                command = build_worker_command(
                    config,
                    suite=suite,
                    task_id=task_id,
                    gpu_id=gpu_id,
                    seed=scheduled["global_seed"],
                )
                environment_bindings = build_worker_environment(
                    config,
                    preregistration=prereg,
                    gpu_id=gpu_id,
                    python_hash_seed=scheduled["python_hash_seed"],
                )
                # Pass exactly the sealed mapping.  No ambient manager variable is
                # inherited by a worker.
                child_environment = dict(environment_bindings)
                log_relative = f"logs/{suite}/task{task_id:02d}.log"
                log_handle = _open_log_no_replace(config.raw_log_root / log_relative)
                log_ownership_transferred = False
                try:
                    launched_at = _timestamp(now)
                    source_guard.checkpoint(f"before launch {process_id}")
                    try:
                        process = popen_factory(
                            command,
                            cwd=str(_lexical_absolute(config.working_directory)),
                            env=child_environment,
                            stdin=subprocess.DEVNULL,
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                            shell=False,
                            close_fds=True,
                            pass_fds=(instrumentation_fd,),
                        )
                    except Exception as exc:
                        log_handle.close()
                        failure_reason = f"launch failed: {exc}"
                        payload = _status_payload(
                            config,
                            suite=suite,
                            task_id=task_id,
                            gpu_id=gpu_id,
                            state="LAUNCH_FAILED",
                            launched_at=launched_at,
                            completed_at=_timestamp(now),
                            exit_code=None,
                            failure_reason=failure_reason,
                            command_argv=command,
                            environment_bindings=environment_bindings,
                            log=_artifact_reference(config.raw_log_root, log_relative),
                            canonical_result=None,
                            trace_receipt=None,
                            raw_result=None,
                        )
                        status_reference = _publish_status(config, payload)
                        task_rows[process_id] = _manifest_task(payload, status_reference)
                        failed = True
                        manager_failure_reason = failure_reason
                        available.append(gpu_id)
                        break
                    running[process_id] = _RunningTask(
                        suite=suite,
                        task_id=task_id,
                        gpu_id=gpu_id,
                        process=process,
                        log_handle=log_handle,
                        launched_at=launched_at,
                        command_argv=command,
                        command_sha256=_canonical_sha(command),
                        environment_bindings=environment_bindings,
                        environment_sha256=_canonical_sha(environment_bindings),
                        log_path=log_relative,
                    )
                    log_ownership_transferred = True
                    source_guard.checkpoint(f"after launch {process_id}")
                finally:
                    if not log_ownership_transferred:
                        log_handle.close()
            if running:
                sleep(config.poll_interval_seconds)

    except BaseException as scheduler_error:
        try:
            _drain_started_workers(
                running,
                sleep=sleep,
                poll_interval_seconds=config.poll_interval_seconds,
            )
        except BaseException as drain_error:
            raise ManagerError(
                "scheduler failed and worker drain was incomplete: "
                f"{type(drain_error).__name__}: {drain_error}"
            ) from scheduler_error
        raise
    if failed:
        reason = manager_failure_reason or "manager stopped after an unknown failure"
        source_guard.checkpoint("before manager failure publication")
        return _publish_failure_receipt(
            config,
            reason=reason,
            pending=pending,
            task_rows=task_rows,
            now=now,
        )

    try:
        source_guard.checkpoint("before finalization")
        for suite in SUITES:
            _rename_directory_no_replace(
                config.artifact_root,
                suite,
                config.raw_log_root / "official",
            )
        for row in task_rows.values():
            archive = _read_relative(config.raw_log_root, row["raw_result_archive_path"])
            if (
                archive["sha256"] != row["raw_result_sha256"]
                or archive["size_bytes"] != row["raw_result_size_bytes"]
            ):
                raise ManagerError(
                    f"archived raw result mismatch: {row['raw_result_archive_path']}"
                )
        inventory = _input_inventory(config.artifact_root)
    except (ManagerError, OSError) as exc:
        source_guard.checkpoint("before finalization failure publication")
        return _publish_failure_receipt(
            config,
            reason=f"manager finalization failed: {exc}",
            pending=(),
            task_rows=task_rows,
            now=now,
        )
    ordered = [
        task_rows[f"{suite}/task{task_id:02d}"]
        for suite in SUITES
        for task_id in range(TASKS_PER_SUITE)
    ]
    upstream_bindings = {
        "preregistration_file_sha256": config.preregistration_sha256,
        "runtime_start_file_sha256": config.runtime_start_sha256,
        "seed_schedule_file_sha256": config.seed_schedule_sha256,
        "resolved_config_sha256": config.resolved_config_sha256,
        "official_commit": config.official_commit,
        "instrumentation_commit": config.instrumentation_commit,
        "python_hash_seed": schedule["python_hash_seed"],
    }
    manifest = {
        "schema_version": 1,
        "kind": "mf_wam_g0_manager_terminal_manifest",
        "run_id": config.run_id,
        "completed_at": _timestamp(now),
        "manager_exit_code": 0,
        "artifact_root": str(_lexical_absolute(config.artifact_root)),
        "raw_log_root": str(_lexical_absolute(config.raw_log_root)),
        "gpu_ids": list(config.gpu_ids),
        "upstream_bindings": upstream_bindings,
        "canonical_input_file_count": EXPECTED_INPUT_FILES,
        "canonical_input_tree_sha256": _tree_sha(inventory),
        "task_processes": ordered,
    }
    source_guard.checkpoint("before terminal publication")
    terminal_reference = _publish_json_no_replace(
        terminal_path,
        manifest,
        post_link_check=lambda: source_guard.checkpoint(
            "during terminal publication"
        ),
    )
    return {
        "status": "SUCCEEDED",
        "terminal_manifest_path": str(terminal_path),
        "terminal_manifest_sha256": terminal_reference["sha256"],
        "canonical_input_file_count": EXPECTED_INPUT_FILES,
        "canonical_input_tree_sha256": manifest["canonical_input_tree_sha256"],
        "failure_receipt_written": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--raw-log-root", required=True, type=Path)
    parser.add_argument("--working-directory", required=True, type=Path)
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--official-commit", required=True)
    parser.add_argument("--instrumentation-commit", required=True)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--runtime-start", required=True, type=Path)
    parser.add_argument("--runtime-start-sha256", required=True)
    parser.add_argument("--task-map", required=True, type=Path)
    parser.add_argument("--seed-schedule", required=True, type=Path)
    parser.add_argument("--seed-schedule-sha256", required=True)
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument("--resolved-config-sha256", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-stats", required=True, type=Path)
    parser.add_argument("--gpu-ids", required=True, help="comma-separated logical GPU IDs")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).resolve().with_name("run_mf_wam_g0_traced.py"),
    )
    parser.add_argument("--task-config", default="libero_uncond_2cam224_1e-4")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        gpu_ids = tuple(int(item) for item in arguments.gpu_ids.split(","))
        config = ManagerConfig(
            run_id=arguments.run_id,
            artifact_root=arguments.artifact_root,
            raw_log_root=arguments.raw_log_root,
            working_directory=arguments.working_directory,
            official_root=arguments.official_root,
            official_commit=arguments.official_commit,
            instrumentation_commit=arguments.instrumentation_commit,
            preregistration_path=arguments.preregistration,
            preregistration_sha256=arguments.preregistration_sha256,
            runtime_start_path=arguments.runtime_start,
            runtime_start_sha256=arguments.runtime_start_sha256,
            task_map_path=arguments.task_map,
            seed_schedule_path=arguments.seed_schedule,
            seed_schedule_sha256=arguments.seed_schedule_sha256,
            resolved_config_path=arguments.resolved_config,
            resolved_config_sha256=arguments.resolved_config_sha256,
            checkpoint_path=arguments.checkpoint,
            dataset_stats_path=arguments.dataset_stats,
            gpu_ids=gpu_ids,
            python_executable=arguments.python_executable,
            runner_path=arguments.runner,
            task_config=arguments.task_config,
        )
        receipt = run_manager(config)
    except (ManagerError, OSError, ValueError) as exc:
        sys.stderr.buffer.write(
            _canonical_bytes({"status": "FAIL", "error": str(exc)}) + b"\n"
        )
        return 2
    sys.stdout.buffer.write(_canonical_bytes(receipt) + b"\n")
    return 0 if receipt["status"] == "SUCCEEDED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
