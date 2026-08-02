"""Small, dependency-free runtime provenance primitives."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import stat
import time
from pathlib import Path


_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


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


def _read_regular_file(path: Path) -> bytes:
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
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"runtime provenance path is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        if len(payload) != metadata.st_size:
            raise RuntimeError(
                f"runtime provenance file changed while being read: {path}"
            )
        return payload
    finally:
        os.close(descriptor)


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
) -> str:
    """Publish bytes once and make every rank verify the same file identity.

    The hash-qualified ready marker is the pre-process-group filesystem barrier.
    It is published only after the target file is fsynced and atomically renamed.
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
