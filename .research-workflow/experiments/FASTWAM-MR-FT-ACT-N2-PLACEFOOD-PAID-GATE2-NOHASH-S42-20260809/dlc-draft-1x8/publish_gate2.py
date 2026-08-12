#!/usr/bin/env python3
"""Publish the minimal, portable Gate2 result set to the OSS FUSE mount.

The training worlds write to node-local POSIX storage.  This publisher then
creates every destination file exclusively, closes it, and reopens both sides
for a direct byte comparison.  It intentionally does not depend on hard links,
renames, directory fsync, or inode identity on the object-store mount.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path


PER_RUN_LIMIT_BYTES = 50 * 1024**3


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def regular_file(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"expected a non-linked regular file: {path}")
    return metadata


def stable_source_fields(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )


def add_tree(files: dict[str, Path], root: Path, prefix: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"required state directory is missing or linked: {root}")
    for base, dirs, names in os.walk(root, followlinks=False):
        base_path = Path(base)
        for name in dirs:
            path = base_path / name
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(f"linked or non-directory state entry: {path}")
        for name in names:
            path = base_path / name
            regular_file(path)
            relative = path.relative_to(root).as_posix()
            files[f"{prefix}/{relative}"] = path


def selected_files(stage: Path) -> dict[str, Path]:
    fixed = {
        "gate2_trainer_evidence.json": stage / "gate2_trainer_evidence.json",
        "real_data_nohash_preflight.json": stage / "real_data_nohash_preflight.json",
        "real_data_nohash_preflight.log": stage / "real_data_nohash_preflight.log",
        "gaussian_primary_staging.json": stage / "gaussian_primary_staging.json",
        "vae_staging.json": stage / "vae_staging.json",
        "logs/save_world.log": stage / "save_world.log",
        "logs/load_world.log": stage / "load_world.log",
        "logs/final_verify_world.log": stage / "final_verify_world.log",
        "load_world/recovery_load_receipt.json": (
            stage / "load_world" / "recovery_load_receipt.json"
        ),
        "final_verify_world/recovery_load_receipt.json": (
            stage / "final_verify_world" / "recovery_load_receipt.json"
        ),
        "load_world/checkpoints/weights/step_000002.pt": (
            stage / "load_world" / "checkpoints" / "weights" / "step_000002.pt"
        ),
    }
    for path in fixed.values():
        regular_file(path)

    state = stage / "load_world" / "checkpoints" / "state" / "step_000002"
    add_tree(
        fixed,
        state,
        "load_world/checkpoints/state/step_000002",
    )
    if not any(
        relative.endswith("/trainer_state.json") for relative in fixed
    ):
        raise RuntimeError("final state lacks trainer_state.json")
    return fixed


def copy_exclusive(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_fd = -1
    try:
        source_before = os.fstat(source_fd)
        if not stat.S_ISREG(source_before.st_mode):
            raise RuntimeError(f"publication source is not regular: {source}")
        destination_fd = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        copied = 0
        while True:
            chunk = os.read(source_fd, 8 * 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise RuntimeError(f"publication made no progress: {destination}")
                copied += written
                view = view[written:]
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        os.close(source_fd)

    source_after = regular_file(source)
    if stable_source_fields(source_before) != stable_source_fields(source_after):
        raise RuntimeError(f"publication source changed while copying: {source}")
    destination_metadata = regular_file(destination)
    if copied != source_before.st_size or destination_metadata.st_size != copied:
        raise RuntimeError(f"publication byte count differs: {destination}")

    with source.open("rb") as left, destination.open("rb") as right:
        while True:
            left_chunk = left.read(8 * 1024 * 1024)
            right_chunk = right.read(8 * 1024 * 1024)
            if left_chunk != right_chunk:
                raise RuntimeError(f"publication direct comparison failed: {destination}")
            if not left_chunk:
                break
    if stable_source_fields(regular_file(source)) != stable_source_fields(source_after):
        raise RuntimeError(f"publication source changed during readback: {source}")
    if regular_file(destination).st_size != copied:
        raise RuntimeError(f"publication destination changed during readback: {destination}")
    return {
        "relative_path": destination.as_posix(),
        "bytes": copied,
        "direct_byte_comparison": "passed",
    }


def write_exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise RuntimeError(f"exclusive write made no progress: {path}")
            view = view[written:]
    finally:
        os.close(fd)
    if path.read_bytes() != payload:
        raise RuntimeError(f"exclusive object readback differs: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-stage", required=True)
    parser.add_argument("--oss-output-root", required=True)
    parser.add_argument("--submission-tag", required=True)
    args = parser.parse_args()

    stage_literal = Path(args.local_stage)
    output = Path(args.oss_output_root)
    if stage_literal.is_symlink() or not stage_literal.is_dir():
        raise RuntimeError("local Gate2 stage is missing or linked")
    stage = stage_literal.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise RuntimeError("Gate2 OSS output already exists")

    selected = selected_files(stage)
    planned_bytes = sum(regular_file(path).st_size for path in selected.values())
    if planned_bytes > PER_RUN_LIMIT_BYTES:
        raise RuntimeError(
            f"selected Gate2 result exceeds {PER_RUN_LIMIT_BYTES} bytes"
        )

    # The suite-wide prefix may not exist on a fresh OSS mount.  Creating only
    # that shared parent is safe and leaves the run-owned directory creation
    # below as the exclusive, fail-if-present publication boundary.
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.mkdir(mode=0o700)
    artifacts = output / "artifacts"
    artifacts.mkdir(mode=0o700)
    entries = []
    for relative, source in sorted(selected.items()):
        destination = artifacts / relative
        entry = copy_exclusive(source, destination)
        entry["relative_path"] = f"artifacts/{relative}"
        entry["retention"] = (
            "temporary_probe_removed_after_readback"
            if relative == "load_world/checkpoints/weights/step_000002.pt"
            or relative.startswith("load_world/checkpoints/state/step_000002/")
            else "retained"
        )
        entries.append(entry)

    published_bytes = sum(int(entry["bytes"]) for entry in entries)
    temporary_entries = [
        entry
        for entry in entries
        if entry["retention"] == "temporary_probe_removed_after_readback"
    ]
    temporary_bytes = sum(int(entry["bytes"]) for entry in temporary_entries)

    # The large state and full-weight objects are an exact-scoped publication
    # probe for this newly created Gate2 run.  Their close/reopen direct byte
    # comparison has completed, so remove only those enumerated objects before
    # writing the durable receipt.  This preserves quota for the three formal
    # runs while retaining the evidence and logs.
    for entry in temporary_entries:
        path = output / str(entry["relative_path"])
        regular_file(path)
        path.unlink()
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"temporary Gate2 probe object remains: {path}")
    candidate_dirs = {
        parent
        for entry in temporary_entries
        for parent in (output / str(entry["relative_path"])).parents
        if artifacts in parent.parents and parent != artifacts
    }
    for directory in sorted(candidate_dirs, key=lambda item: len(item.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass

    retained_bytes = published_bytes - temporary_bytes
    receipt = {
        "schema": "fastwam-gate2-oss-publication-receipt-v2",
        "submission_tag": args.submission_tag,
        "created_at": utc_now(),
        "integrity_mode": "metadata_no_hash",
        "publication_method": "exclusive_stream_close_and_direct_readback",
        "local_weight_sidecars_published": False,
        "step1_training_state_published": False,
        "step2_training_state_probe_published_and_compared": True,
        "final_full_weights_probe_published_and_compared": True,
        "temporary_large_probe_objects_removed_after_readback": True,
        "temporary_probe_bytes": temporary_bytes,
        "retained_bytes": retained_bytes,
        "planned_bytes": planned_bytes,
        "directly_compared_bytes": published_bytes,
        "files": entries,
    }
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    write_exclusive_bytes(output / "publication_receipt.json", receipt_bytes)

    complete = {
        "schema": "fastwam-gate2-complete-v2",
        "status": "COMPLETE",
        "integrity_mode": "metadata_no_hash",
        "submission_tag": args.submission_tag,
        "completed_at": utc_now(),
        "resumed_from_step": 1,
        "final_global_step": 2,
        "fresh_load_advanced": True,
        "checkpoint_state_kind": "full",
        "final_state_fresh_load_verified": True,
        "temporary_final_weights_probe": (
            "artifacts/load_world/checkpoints/weights/step_000002.pt"
        ),
        "temporary_final_state_probe": (
            "artifacts/load_world/checkpoints/state/step_000002"
        ),
        "temporary_large_probe_objects_removed_after_readback": True,
        "trainer_evidence": "artifacts/gate2_trainer_evidence.json",
        "publication_receipt": "publication_receipt.json",
        "directly_compared_bytes": published_bytes,
        "retained_bytes": retained_bytes,
    }
    complete_bytes = (json.dumps(complete, indent=2, sort_keys=True) + "\n").encode()
    write_exclusive_bytes(output / "COMPLETE.json", complete_bytes)

    # COMPLETE must remain the final object and must still be readable after
    # closing its create handle.
    if (output / "COMPLETE.json").read_bytes() != complete_bytes:
        raise RuntimeError("Gate2 COMPLETE final readback differs")


if __name__ == "__main__":
    main()
