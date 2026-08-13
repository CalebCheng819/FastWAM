"""Metadata-only artifact readers for standalone RoboFactory evaluation."""

from __future__ import annotations

import json
import os
import stat
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
    unresolved = Path(path).expanduser()
    resolved = unresolved.resolve(strict=True)
    if unresolved.is_symlink():
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
    unresolved = Path(path).expanduser()
    resolved = unresolved.resolve(strict=True)
    if unresolved.is_symlink():
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
