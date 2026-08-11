#!/usr/bin/env python3
"""Stage an allowlisted tree with stat + byte comparison, without new hashes."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import stat
from collections.abc import Callable


CopyFile = Callable[[pathlib.Path, pathlib.Path], object]


def _regular_non_symlink(path: pathlib.Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    return metadata


def _safe_relative_path(raw: str) -> pathlib.PurePosixPath:
    candidate = raw.strip()
    # GNU checksum manifests may mark binary-mode paths with a leading '*'.
    # The preceding digest field is intentionally ignored by this B4 tool.
    if candidate.startswith("*"):
        candidate = candidate[1:]
    relative = pathlib.PurePosixPath(candidate)
    if (
        not candidate
        or candidate == "."
        or relative.is_absolute()
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise RuntimeError(f"unsafe allowlist relative path: {raw!r}")
    return relative


def read_allowlist(path: pathlib.Path) -> list[pathlib.PurePosixPath]:
    _regular_non_symlink(path, label="allowlist")
    selected: list[pathlib.PurePosixPath] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(
                f"allowlist line {line_number} must contain an ignored id field and a relative path"
            )
        relative = _safe_relative_path(fields[1])
        key = relative.as_posix()
        if key in seen:
            raise RuntimeError(f"duplicate allowlist path at line {line_number}: {key}")
        seen.add(key)
        selected.append(relative)
    if not selected:
        raise RuntimeError("allowlist contains no files")
    return selected


def _resolve_source(
    source_root: pathlib.Path,
    relative: pathlib.PurePosixPath,
) -> pathlib.Path:
    candidate = source_root.joinpath(*relative.parts)
    cursor = source_root
    for part in relative.parts:
        cursor = cursor / part
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise RuntimeError(f"allowlisted source is unavailable: {relative}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"allowlisted source traverses a symlink: {relative}")
    _regular_non_symlink(candidate, label="allowlisted source")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(source_root)
    except ValueError as exc:
        raise RuntimeError(f"allowlisted source escaped source root: {relative}") from exc
    return resolved


def _files_equal(left: pathlib.Path, right: pathlib.Path) -> bool:
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while True:
            left_chunk = left_stream.read(8 * 1024 * 1024)
            right_chunk = right_stream.read(8 * 1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def stage_allowlisted_tree(
    *,
    source_root: pathlib.Path,
    allowlist: pathlib.Path,
    destination: pathlib.Path,
    run_id: str,
    attempt_id: str,
    source_label: str,
    copy_file: CopyFile = shutil.copyfile,
) -> dict[str, object]:
    if not source_root.is_absolute() or not destination.is_absolute():
        raise RuntimeError("source root and destination must be absolute")
    if source_root.is_symlink() or not source_root.is_dir():
        raise RuntimeError(f"source root must be an existing non-symlink directory: {source_root}")
    source_root = source_root.resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    if partial.exists() or partial.is_symlink():
        raise RuntimeError(f"partial destination already exists: {partial}")

    selected = read_allowlist(allowlist)
    records: list[dict[str, object]] = []
    total_bytes = 0
    newest_mtime_ns = 0
    partial.mkdir(mode=0o700)
    try:
        for relative in selected:
            source = _resolve_source(source_root, relative)
            before = source.stat()
            target = partial.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_file(source, target)
            copied = _regular_non_symlink(target, label="staged destination")
            after = source.stat()
            identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            if identity_before != identity_after:
                raise RuntimeError(f"source changed while staging: {relative.as_posix()}")
            if copied.st_size != before.st_size:
                raise RuntimeError(f"staged byte count differs for {relative.as_posix()}")
            if not _files_equal(source, target):
                raise RuntimeError(f"staged bytes differ for {relative.as_posix()}")
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns), follow_symlinks=False)
            record = {
                "path": relative.as_posix(),
                "bytes": before.st_size,
                "mtime_ns": before.st_mtime_ns,
            }
            records.append(record)
            total_bytes += before.st_size
            newest_mtime_ns = max(newest_mtime_ns, before.st_mtime_ns)

        ready = {
            "provenance_mode": "stat_cmp",
            "run_id": run_id,
            "attempt_id": attempt_id,
            "source_label": source_label,
            "source_root": str(source_root),
            "allowlist_path": str(allowlist.resolve(strict=True)),
            "destination_path": str(destination),
            "file_count": len(records),
            "total_bytes": total_bytes,
            "newest_source_mtime_ns": newest_mtime_ns,
            "files": records,
        }
        ready_path = partial / "READY.stat-cmp.json"
        with ready_path.open("x", encoding="utf-8") as stream:
            json.dump(ready, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(partial, destination)
        return ready
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--allowlist", type=pathlib.Path, required=True)
    parser.add_argument("--destination", type=pathlib.Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--source-label", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        ready = stage_allowlisted_tree(
            source_root=args.source_root,
            allowlist=args.allowlist,
            destination=args.destination,
            run_id=args.run_id,
            attempt_id=args.attempt_id,
            source_label=args.source_label,
        )
    except RuntimeError as exc:
        raise SystemExit(f"B4 stat-cmp cache error: {exc}") from exc
    print(
        "B4 stat-cmp cache ready: "
        f"source={ready['source_label']} path={ready['destination_path']} "
        f"files={ready['file_count']} bytes={ready['total_bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
