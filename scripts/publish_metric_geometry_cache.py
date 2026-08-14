#!/usr/bin/env python3
"""Publish a validated node-local metric cache to a versioned shared path."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from fastwam.datasets.metric_geometry_cache import MetricGeometryCache


ALLOWLIST = (
    "metadata frames.f16\n"
    "metadata manifest.json\n"
    "metadata COMPLETE\n"
)
DATA_FILES = ("frames.f16", "manifest.json", "stat-cmp.allowlist")
TRANSIENT_ERRNOS = {
    errno.EAGAIN,
    errno.EIO,
    errno.ENOTCONN,
    errno.ESTALE,
    errno.ETIMEDOUT,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def regular_file(root: Path, name: str) -> Path:
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"cache artifact is missing or non-regular: {path}")
    return path


def validate_cache(root: Path) -> dict[str, int]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"cache root is missing or non-regular: {root}")
    complete = regular_file(root, "COMPLETE")
    if complete.read_text(encoding="utf-8") != "complete\n":
        raise RuntimeError("cache COMPLETE marker mismatch")
    allowlist = regular_file(root, "stat-cmp.allowlist")
    if allowlist.read_text(encoding="utf-8") != ALLOWLIST:
        raise RuntimeError("cache stat-cmp allowlist mismatch")
    cache = MetricGeometryCache.open(root)
    try:
        counts = cache.manifest.get("counts") or {}
        windows = int(counts.get("windows", 0))
        frames = int(counts.get("frames", 0))
        if windows <= 0 or frames != 2 * windows or frames != cache.frames:
            raise RuntimeError(f"cache count contract mismatch: {counts}")
        return {
            "bytes": int(cache.data_path.stat().st_size),
            "frames": frames,
            "windows": windows,
        }
    finally:
        cache.close()


def streams_equal(left: BinaryIO, right: BinaryIO, chunk_bytes: int = 8 << 20) -> bool:
    while True:
        left_chunk = left.read(chunk_bytes)
        right_chunk = right.read(chunk_bytes)
        if left_chunk != right_chunk:
            return False
        if not left_chunk:
            return True


def files_equal(left: Path, right: Path) -> bool:
    left_stat = left.stat()
    right_stat = right.stat()
    if left_stat.st_size != right_stat.st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        return streams_equal(left_stream, right_stream)


def copy_regular_file(source: Path, destination: Path) -> None:
    source_stat = source.stat()
    if destination.exists() or destination.is_symlink():
        destination_stat = destination.stat()
        if (
            destination.is_file()
            and destination_stat.st_size == source_stat.st_size
            and destination_stat.st_mtime_ns == source_stat.st_mtime_ns
            and files_equal(source, destination)
        ):
            return
        raise FileExistsError(f"published file conflicts with source: {destination}")

    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=8 << 20)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.utime(
            temporary,
            ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
            follow_symlinks=False,
        )
        if not files_equal(source, temporary):
            raise RuntimeError(f"published byte comparison failed: {destination}")
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


class Publisher:
    def __init__(self, source_root: Path, target_root: Path, run_id: str) -> None:
        self.source_root = source_root
        self.target_root = target_root
        self.run_id = run_id
        self.created_target = False

    def publish_once(self) -> dict[str, int]:
        if not self.created_target:
            if self.target_root.exists() or self.target_root.is_symlink():
                raise FileExistsError(
                    f"use a new versioned target path: {self.target_root}"
                )
            self.target_root.parent.mkdir(parents=True, exist_ok=True)
            self.target_root.mkdir()
            self.created_target = True
        elif self.target_root.is_symlink() or not self.target_root.is_dir():
            raise RuntimeError(f"publisher lost target ownership: {self.target_root}")

        for name in DATA_FILES:
            copy_regular_file(
                regular_file(self.source_root, name), self.target_root / name
            )
        for name in DATA_FILES:
            source = regular_file(self.source_root, name)
            target = regular_file(self.target_root, name)
            if source.stat().st_mtime_ns != target.stat().st_mtime_ns:
                raise RuntimeError(f"published mtime comparison failed: {target}")
            if not files_equal(source, target):
                raise RuntimeError(f"published byte comparison failed: {target}")

        # COMPLETE is the commit marker and must be the final published file.
        copy_regular_file(
            regular_file(self.source_root, "COMPLETE"),
            self.target_root / "COMPLETE",
        )
        summary = validate_cache(self.target_root)
        print(
            json.dumps(
                {
                    "event": "metric_cache_published",
                    "observed_at_utc": utc_now(),
                    "run_id": self.run_id,
                    "source_root": str(self.source_root),
                    "target_root": str(self.target_root),
                    **summary,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return summary


def publish_with_retry(
    publisher: Publisher,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, int]:
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        try:
            return publisher.publish_once()
        except OSError as error:
            if error.errno not in TRANSIENT_ERRNOS or time.monotonic() >= deadline:
                raise
            print(
                json.dumps(
                    {
                        "attempt": attempt,
                        "errno": error.errno,
                        "event": "metric_cache_publish_retry",
                        "message": str(error),
                        "observed_at_utc": utc_now(),
                        "run_id": publisher.run_id,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(poll_seconds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0 or args.poll_seconds <= 0:
        parser.error("retry intervals must be positive")
    if not args.run_id or len(args.run_id) > 128:
        parser.error("run-id must contain 1-128 characters")
    return args


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve(strict=True)
    target_root = args.target_root.expanduser()
    if not target_root.is_absolute():
        raise RuntimeError("target root must be absolute")
    if source_root == target_root or source_root in target_root.parents:
        raise RuntimeError("target root must not be inside source root")
    source_summary = validate_cache(source_root)
    print(
        json.dumps(
            {
                "event": "metric_cache_local_validation_pass",
                "observed_at_utc": utc_now(),
                "run_id": args.run_id,
                "source_root": str(source_root),
                **source_summary,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    publish_with_retry(
        Publisher(source_root, target_root, args.run_id),
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
