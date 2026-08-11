"""Fail-closed artifact helpers that deliberately do not compute digests.

This module is for the explicit ``metadata_no_hash`` recovery mode.  It uses
no-follow file descriptors, stable ``fstat`` metadata, direct byte comparison,
and exclusive local publication.  It must remain independent of
``formal_artifacts`` and ``runtime_provenance``.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from typing import Any


def _open_regular_no_follow(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(fd)
        raise RuntimeError(f"Expected a regular file: {path}")
    return fd


def _stat_fields(metadata: os.stat_result) -> dict[str, int]:
    return {
        "bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "dev": int(metadata.st_dev),
        "ino": int(metadata.st_ino),
        "mode": int(metadata.st_mode),
    }


def regular_file_metadata(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if Path(path).expanduser().is_symlink():
        raise RuntimeError(f"Symlink is forbidden for metadata_no_hash: {path}")
    fd = _open_regular_no_follow(resolved)
    try:
        before = os.fstat(fd)
        after = os.fstat(fd)
        if _stat_fields(before) != _stat_fields(after):
            raise RuntimeError(f"File metadata changed while inspecting it: {resolved}")
        return {"path": str(resolved), **_stat_fields(after)}
    finally:
        os.close(fd)


def read_regular_bytes(path: str | Path) -> tuple[bytes, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if Path(path).expanduser().is_symlink():
        raise RuntimeError(f"Symlink is forbidden for metadata_no_hash: {path}")
    fd = _open_regular_no_follow(resolved)
    try:
        before = os.fstat(fd)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        if _stat_fields(before) != _stat_fields(after):
            raise RuntimeError(f"File changed while reading it: {resolved}")
        payload = b"".join(chunks)
        if len(payload) != int(after.st_size):
            raise RuntimeError(f"Short read from regular file: {resolved}")
        return payload, {"path": str(resolved), **_stat_fields(after)}
    finally:
        os.close(fd)


def read_json(path: str | Path) -> tuple[Any, dict[str, Any]]:
    payload, metadata = read_regular_bytes(path)
    try:
        return json.loads(payload.decode("utf-8")), metadata
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid UTF-8 JSON artifact: {path}") from error


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path.parent, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_exclusive_bytes(path: str | Path, payload: bytes) -> dict[str, Any]:
    """Publish bytes without replacement using a temporary file plus hard link.

    The destination filesystem must support same-directory hard links.  This is
    intentional: metadata_no_hash Gate2 writes to a local shared filesystem and
    fails closed instead of weakening exclusive publication on object mounts.
    """

    destination = Path(path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise RuntimeError(f"Short write while publishing: {destination}")
            view = view[written:]
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)
    try:
        os.link(temporary, destination, follow_symlinks=False)
        _fsync_parent(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink()
    _fsync_parent(destination)
    observed, metadata = read_regular_bytes(destination)
    if observed != payload:
        raise RuntimeError(f"Published bytes differ from source bytes: {destination}")
    return metadata


def publish_exclusive_json(path: str | Path, value: Any) -> dict[str, Any]:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    return publish_exclusive_bytes(path, payload)


def copy_exclusive_and_compare(source: str | Path, destination: str | Path) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve(strict=True)
    destination_path = Path(destination).expanduser().resolve(strict=False)
    source_fd = _open_regular_no_follow(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    target_fd = os.open(destination_path, flags, 0o600)
    try:
        before = os.fstat(source_fd)
        while True:
            chunk = os.read(source_fd, 8 * 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(target_fd, view)
                if written <= 0:
                    raise RuntimeError(f"Short checkpoint write: {destination_path}")
                view = view[written:]
        os.fsync(target_fd)
        after = os.fstat(source_fd)
        target_stat = os.fstat(target_fd)
        if _stat_fields(before) != _stat_fields(after):
            raise RuntimeError(f"Source changed during checkpoint copy: {source_path}")
        if int(target_stat.st_size) != int(after.st_size):
            raise RuntimeError(f"Checkpoint copy size mismatch: {destination_path}")
    except BaseException:
        os.close(source_fd)
        os.close(target_fd)
        destination_path.unlink(missing_ok=True)
        raise
    os.close(source_fd)
    os.close(target_fd)
    _fsync_parent(destination_path)

    source_fd = _open_regular_no_follow(source_path)
    target_fd = _open_regular_no_follow(destination_path)
    try:
        while True:
            source_chunk = os.read(source_fd, 8 * 1024 * 1024)
            target_chunk = os.read(target_fd, 8 * 1024 * 1024)
            if source_chunk != target_chunk:
                raise RuntimeError(
                    f"Checkpoint byte comparison failed: {destination_path}"
                )
            if not source_chunk:
                break
    except BaseException:
        destination_path.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_fd)
        os.close(target_fd)
    return regular_file_metadata(destination_path)


def publish_rank_zero_payload(
    path: str | Path,
    payload: bytes,
    *,
    rank: int,
    world_size: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    destination = Path(path).expanduser().resolve(strict=False)
    ready = destination.with_name(f"{destination.name}.metadata_no_hash.ready.json")
    if rank == 0:
        metadata = publish_exclusive_bytes(destination, payload)
        publish_exclusive_json(
            ready,
            {
                "schema_name": "fastwam-metadata-no-hash-ready",
                "schema_version": 1,
                "world_size": int(world_size),
                "file": metadata,
            },
        )
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        try:
            marker, _ = read_json(ready)
            observed, metadata = read_regular_bytes(destination)
        except FileNotFoundError:
            time.sleep(0.1)
            continue
        if not isinstance(marker, dict) or marker.get("world_size") != int(world_size):
            raise RuntimeError(f"Invalid metadata_no_hash ready marker: {ready}")
        if marker.get("file") != metadata:
            raise RuntimeError(f"Resolved config metadata changed after publication: {destination}")
        if observed != payload:
            raise RuntimeError(f"Resolved config differs across ranks: {destination}")
        return metadata
    raise TimeoutError(f"Timed out waiting for metadata_no_hash config barrier: {ready}")
