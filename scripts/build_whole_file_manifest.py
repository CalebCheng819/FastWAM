#!/usr/bin/env python3
"""Build an immutable sha256sum manifest for selected whole-file bundles.

The source root is never modified.  Includes are explicit relative files or
directories, traversal and symlinks are rejected, entries are bytewise sorted,
and the output is published atomically without replacing an existing manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path, PurePosixPath


def _safe_relative(value: str) -> PurePosixPath:
    if not value or "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"unsafe relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("/"):
        raise ValueError(f"path must be relative: {value!r}")
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"path contains an unsafe component: {value!r}")
    return path


def _walk_selected(root: Path, relative: PurePosixPath) -> list[PurePosixPath]:
    path = root.joinpath(*relative.parts)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"symlinks are forbidden in manifest input: {relative}")
    if stat.S_ISREG(info.st_mode):
        return [relative]
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"manifest input must be a regular file or directory: {relative}")

    files: list[PurePosixPath] = []
    with os.scandir(path) as entries:
        children = sorted(entries, key=lambda entry: os.fsencode(entry.name))
    for entry in children:
        child_name = entry.name
        _safe_relative(child_name)
        child = relative / child_name
        if entry.is_symlink():
            raise ValueError(f"symlinks are forbidden in manifest input: {child}")
        if entry.is_dir(follow_symlinks=False):
            files.extend(_walk_selected(root, child))
        elif entry.is_file(follow_symlinks=False):
            files.append(child)
        else:
            raise ValueError(f"special files are forbidden in manifest input: {child}")
    return files


def _sha256_regular_file(path: Path) -> str:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {path}")
        digest = hashlib.sha256()
        with os.fdopen(fd, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"source changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(fd)


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def build_manifest(source_root: Path, includes: list[str], output: Path) -> dict[str, object]:
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError(f"source root must be a regular directory, not a symlink: {source_root}")
    source_root = source_root.resolve(strict=True)
    output = output.expanduser().resolve(strict=False)
    try:
        output.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("manifest output must be outside the source root")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError(f"manifest output parent must be an existing non-symlink directory: {output.parent}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace existing manifest: {output}")
    if not includes:
        raise ValueError("at least one --include is required; recursive source-root scans are not implicit")

    selected: set[PurePosixPath] = set()
    for raw_include in includes:
        relative = _safe_relative(raw_include)
        for item in _walk_selected(source_root, relative):
            selected.add(item)
    if not selected:
        raise ValueError("selected bundle contains no regular files")
    ordered = sorted(selected, key=lambda item: os.fsencode(item.as_posix()))

    manifest_digest = hashlib.sha256()
    total_bytes = 0
    temporary = output.parent / f".{output.name}.tmp.{os.getpid()}"
    publish_lock = output.parent / f".{output.name}.publish.lock"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(temporary, flags, 0o440)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            for relative in ordered:
                source = source_root.joinpath(*relative.parts)
                digest = _sha256_regular_file(source)
                line = f"{digest}  {relative.as_posix()}\n".encode("utf-8")
                stream.write(line)
                manifest_digest.update(line)
                total_bytes += source.stat().st_size
            stream.flush()
            os.fsync(fd)
        try:
            os.mkdir(publish_lock, 0o700)
        except FileExistsError as error:
            raise FileExistsError(f"manifest publish lock already exists: {publish_lock}") from error
        try:
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"refusing to replace existing manifest: {output}")
            # os.replace is the atomic visibility boundary. The exclusive
            # publish lock provides no-clobber semantics on object/FUSE mounts
            # such as OSS, where POSIX hard links are intentionally unsupported.
            os.replace(temporary, output)
            _fsync_directory(output.parent)
        finally:
            publish_lock.rmdir()
    finally:
        os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    return {
        "file_count": len(ordered),
        "manifest_sha256": manifest_digest.hexdigest(),
        "output": str(output),
        "source_root": str(source_root),
        "total_bytes": total_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = build_manifest(args.source_root, args.include, args.output)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
