"""Small, dependency-free runtime provenance primitives."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path


def rank_and_world_from_environment() -> tuple[int, int]:
    """Return the torchrun rank/world contract before a process group exists."""

    try:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as error:
        raise RuntimeError("RANK and WORLD_SIZE must be integers") from error
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise RuntimeError(f"invalid runtime topology: rank={rank} world_size={world_size}")
    return rank, world_size


def _atomic_replace(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
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
        raise RuntimeError(f"invalid file-barrier topology: rank={rank} world_size={world_size}")
    if timeout_seconds <= 0:
        raise ValueError("file-barrier timeout must be positive")
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {path.parent}")
    digest = hashlib.sha256(payload).hexdigest()
    ready = path.parent / f".{path.name}.ready.{digest}"

    if rank == 0:
        _atomic_replace(path, payload)
        if not ready.exists():
            descriptor = os.open(ready, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
            try:
                os.write(descriptor, f"sha256={digest}\n".encode("ascii"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if ready.is_symlink():
            raise RuntimeError(f"runtime ready marker must not be a symlink: {ready}")
        if ready.is_file():
            marker = ready.read_text(encoding="ascii")
            if marker != f"sha256={digest}\n":
                raise RuntimeError(f"runtime ready marker has an invalid payload: {ready}")
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"runtime config is not a regular non-symlink file: {path}")
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed != digest:
                raise RuntimeError(
                    f"runtime config identity mismatch: expected={digest} observed={observed}"
                )
            return digest
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for rank-0 runtime file barrier: {ready}")
