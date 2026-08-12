"""Small, dependency-free runtime provenance primitives."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path
from typing import Any


_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_STAT_CMP_MARKER_SCHEMA = "fastwam-runtime-file-barrier-stat-cmp-v2"
_ATTEMPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _temporary_path(path: Path) -> Path:
    return path.parent / (f".{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}")


def _write_fsynced_exclusive(path: Path, payload: bytes, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("runtime provenance write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


_SNAPSHOT_STABLE_FIELDS = (
    "st_dev",
    "st_ino",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


def _read_regular_file_once(path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise RuntimeError(
                f"runtime provenance path must not be a symlink: {path}"
            ) from error
        raise
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"runtime provenance path is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            stat.S_IFMT(before.st_mode) != stat.S_IFMT(after.st_mode)
            or any(
                getattr(before, field) != getattr(after, field)
                for field in _SNAPSHOT_STABLE_FIELDS
            )
        ):
            raise RuntimeError(
                f"runtime provenance file changed while being read: {path}"
            )
        if len(payload) != after.st_size:
            raise RuntimeError(
                f"runtime provenance file changed while being read: {path}"
            )
        try:
            pathname = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise RuntimeError(
                f"runtime provenance pathname changed while being read: {path}"
            ) from error
        if (
            not stat.S_ISREG(pathname.st_mode)
            or stat.S_IFMT(after.st_mode) != stat.S_IFMT(pathname.st_mode)
            or any(
                getattr(after, field) != getattr(pathname, field)
                for field in _SNAPSHOT_STABLE_FIELDS
            )
        ):
            raise RuntimeError(
                f"runtime provenance pathname changed while being read: {path}"
            )
        return payload, after
    finally:
        os.close(descriptor)


def _read_regular_file_snapshot(path: Path) -> tuple[bytes, dict[str, int]]:
    """Read a small provenance file twice and bind bytes to its pathname.

    The second O_NOFOLLOW open is intentional.  Some FUSE clients can expose
    path-stable or coarsely quantized metadata across an atomic replacement,
    so fd/path stat comparisons alone cannot prove that the bytes just read
    still belong to the pathname.  Full byte equality across two independent
    opens supplies that binding without adding a digest.
    """

    first_payload, first = _read_regular_file_once(path)
    second_payload, second = _read_regular_file_once(path)
    if first_payload != second_payload:
        raise RuntimeError(
            f"runtime provenance pathname changed between verification reads: {path}"
        )
    if (
        stat.S_IFMT(first.st_mode) != stat.S_IFMT(second.st_mode)
        or any(
            getattr(first, field) != getattr(second, field)
            for field in _SNAPSHOT_STABLE_FIELDS
        )
    ):
        raise RuntimeError(
            f"runtime provenance pathname changed between verification reads: {path}"
        )
    return second_payload, {
        "bytes": int(second.st_size),
        "mtime_ns": int(second.st_mtime_ns),
    }


def _read_regular_file(path: Path) -> bytes:
    return _read_regular_file_snapshot(path)[0]


def _atomic_publish_noreplace(path: Path, payload: bytes, mode: int) -> None:
    """Publish a fully fsynced file without ever exposing partial contents."""

    temporary = _temporary_path(path)
    try:
        _write_fsynced_exclusive(temporary, payload, mode)
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError(
                "renameat2 is required for no-clobber runtime provenance"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            _AT_FDCWD,
            os.fsencode(temporary),
            _AT_FDCWD,
            os.fsencode(path),
            _RENAME_NOREPLACE,
        )
        if result != 0:
            value = ctypes.get_errno()
            if value != errno.EEXIST:
                raise OSError(value, os.strerror(value), str(path))
            observed = _read_regular_file(path)
            if observed != payload:
                raise RuntimeError(
                    f"runtime provenance no-clobber collision has different content: {path}"
                )
        else:
            _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def rank_and_world_from_environment() -> tuple[int, int]:
    """Return the torchrun rank/world contract before a process group exists."""

    try:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as error:
        raise RuntimeError("RANK and WORLD_SIZE must be integers") from error
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise RuntimeError(
            f"invalid runtime topology: rank={rank} world_size={world_size}"
        )
    return rank, world_size


def _atomic_replace(path: Path, payload: bytes) -> None:
    temporary = _temporary_path(path)
    try:
        _write_fsynced_exclusive(temporary, payload, 0o640)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def publish_rank_zero_file(
    path: Path,
    payload: bytes,
    *,
    rank: int,
    world_size: int,
    timeout_seconds: float = 300.0,
    provenance_mode: str = "sha256",
    attempt_id: str | None = None,
) -> str | None:
    """Publish bytes once and make every rank verify the same file identity.

    The ready marker is the pre-process-group filesystem barrier.  It is
    published only after the target file is fsynced and atomically renamed.
    """

    if world_size < 1 or rank < 0 or rank >= world_size:
        raise RuntimeError(
            f"invalid file-barrier topology: rank={rank} world_size={world_size}"
        )
    if timeout_seconds <= 0:
        raise ValueError("file-barrier timeout must be positive")
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {path.parent}")
    provenance_mode = str(provenance_mode).strip().lower()
    if provenance_mode not in {"sha256", "stat_cmp"}:
        raise ValueError(
            "provenance_mode must be 'sha256' or 'stat_cmp', "
            f"got {provenance_mode!r}"
        )
    if provenance_mode == "stat_cmp":
        return _publish_rank_zero_file_stat_cmp(
            path,
            payload,
            rank=rank,
            world_size=world_size,
            timeout_seconds=timeout_seconds,
            attempt_id=attempt_id,
        )
    digest = hashlib.sha256(payload).hexdigest()
    ready = path.parent / f".{path.name}.ready.{digest}"

    if rank == 0:
        _atomic_replace(path, payload)
        _atomic_publish_noreplace(ready, f"sha256={digest}\n".encode("ascii"), 0o440)

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            marker = _read_regular_file(ready)
        except FileNotFoundError:
            marker = None
        if marker is not None:
            expected_marker = f"sha256={digest}\n".encode("ascii")
            if marker != expected_marker:
                raise RuntimeError(
                    f"runtime ready marker has an invalid payload: {ready}"
                )
            observed = hashlib.sha256(_read_regular_file(path)).hexdigest()
            if observed != digest:
                raise RuntimeError(
                    f"runtime config identity mismatch: expected={digest} observed={observed}"
                )
            return digest
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for rank-0 runtime file barrier: {ready}")


def _stat_cmp_marker_payload(
    path: Path,
    *,
    attempt_id: str,
    world_size: int,
    metadata: dict[str, int],
) -> bytes:
    marker: dict[str, Any] = {
        "schema": _STAT_CMP_MARKER_SCHEMA,
        "attempt_id": attempt_id,
        "world_size": world_size,
        "path": str(path.resolve()),
        "bytes": int(metadata["bytes"]),
        "mtime_ns": int(metadata["mtime_ns"]),
        "count": 1,
    }
    return (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _validate_stat_cmp_marker(
    marker_payload: bytes,
    *,
    ready: Path,
    path: Path,
    payload: bytes,
    attempt_id: str,
    world_size: int,
) -> None:
    try:
        marker = json.loads(marker_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"runtime ready marker has an invalid payload: {ready}"
        ) from error
    expected_keys = {
        "schema",
        "attempt_id",
        "world_size",
        "path",
        "bytes",
        "mtime_ns",
        "count",
    }
    if not isinstance(marker, dict) or set(marker) != expected_keys:
        raise RuntimeError(f"runtime ready marker has an invalid schema: {ready}")
    expected_static = {
        "schema": _STAT_CMP_MARKER_SCHEMA,
        "attempt_id": attempt_id,
        "world_size": world_size,
        "path": str(path.resolve()),
        "bytes": len(payload),
        "count": 1,
    }
    if (
        type(marker.get("world_size")) is not int
        or type(marker.get("bytes")) is not int
        or type(marker.get("count")) is not int
        or any(marker.get(key) != value for key, value in expected_static.items())
    ):
        raise RuntimeError(f"runtime ready marker contract mismatch: {ready}")
    if type(marker.get("mtime_ns")) is not int or int(marker["mtime_ns"]) < 0:
        raise RuntimeError(f"runtime ready marker has an invalid mtime: {ready}")
    observed, metadata = _read_regular_file_snapshot(path)
    if observed != payload:
        raise RuntimeError(f"runtime config byte comparison mismatch: {path}")
    if metadata["bytes"] != int(marker["bytes"]):
        raise RuntimeError(f"runtime config stat comparison mismatch: {path}")
    # OSSFS clients can expose the same immutable object with differently
    # quantized or cached mtimes.  Keep rank zero's mtime in the marker for
    # diagnostics, but make exact bytes plus size the cross-client identity
    # gate.  Snapshot reads above still reject a file changing during a read.


def _publish_rank_zero_file_stat_cmp(
    path: Path,
    payload: bytes,
    *,
    rank: int,
    world_size: int,
    timeout_seconds: float,
    attempt_id: str | None,
) -> None:
    if attempt_id is None or not _ATTEMPT_ID_RE.fullmatch(str(attempt_id)):
        raise ValueError(
            "stat_cmp file barrier requires a safe non-empty attempt_id containing only "
            "letters, digits, '.', '_' or '-'"
        )
    attempt_id = str(attempt_id)
    ready = path.parent / f".{path.name}.ready.stat_cmp.{attempt_id}"

    if rank == 0:
        try:
            existing_marker = _read_regular_file(ready)
        except FileNotFoundError:
            existing_marker = None
        if existing_marker is None:
            _atomic_replace(path, payload)
            observed, metadata = _read_regular_file_snapshot(path)
            if observed != payload:
                raise RuntimeError(f"runtime config byte comparison mismatch: {path}")
            _atomic_publish_noreplace(
                ready,
                _stat_cmp_marker_payload(
                    path,
                    attempt_id=attempt_id,
                    world_size=world_size,
                    metadata=metadata,
                ),
                0o440,
            )

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            marker_payload = _read_regular_file(ready)
        except FileNotFoundError:
            marker_payload = None
        if marker_payload is not None:
            _validate_stat_cmp_marker(
                marker_payload,
                ready=ready,
                path=path,
                payload=payload,
                attempt_id=attempt_id,
                world_size=world_size,
            )
            return None
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for rank-0 runtime file barrier: {ready}")
