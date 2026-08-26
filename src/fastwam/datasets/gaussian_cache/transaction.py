"""Fail-closed restart transactions for paired canonical/compact micro-parts."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import (
    COMPLETE_FILENAME,
    canonical_json_bytes,
    load_manifest,
    sha256_file,
    write_immutable_file,
)

TRANSACTION_VERSION = 1
_PART_NAME_RE = re.compile(r"^part-\d{5,}$")


class UnsafeCacheRestartError(RuntimeError):
    """Raised when an incomplete root cannot be proven to belong to this task."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _marker_path(root: Path) -> Path:
    return root.with_name(f"{root.name}.BUILDING.json")


def _journal_path(root: Path) -> Path:
    return root.with_name(f"{root.name}.JOURNAL.jsonl")


def _identity_payload(
    root: Path,
    *,
    task_id: str,
    work_plan_sha256: str,
    role: str,
    micro_part_index: int,
    work_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not task_id:
        raise ValueError("task_id must be non-empty")
    if not re.fullmatch(r"[0-9a-f]{64}", str(work_plan_sha256)):
        raise ValueError("work_plan_sha256 must be a lowercase SHA-256")
    if role not in {"canonical", "compact"}:
        raise ValueError("role must be 'canonical' or 'compact'")
    if int(micro_part_index) < 0:
        raise ValueError("micro_part_index must be non-negative")
    if not _PART_NAME_RE.fullmatch(root.name):
        raise ValueError(
            f"Restart-managed cache root must be named part-XXXXX, got {root.name!r}"
        )
    identity = json.loads(canonical_json_bytes(dict(work_identity)))
    return {
        "transaction_version": TRANSACTION_VERSION,
        "task_id": str(task_id),
        "work_plan_sha256": str(work_plan_sha256),
        "role": role,
        "micro_part_index": int(micro_part_index),
        "root_name": root.name,
        "work_identity": identity,
    }


def _read_marker(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsafeCacheRestartError(f"Invalid task-owned BUILDING marker: {path}") from exc
    if not isinstance(value, dict):
        raise UnsafeCacheRestartError(f"BUILDING marker must be a JSON object: {path}")
    return value


def _append_journal(root: Path, identity: Mapping[str, Any], event: str, **details: Any) -> None:
    payload = {
        "at": _utc_now(),
        "event": str(event),
        **dict(identity),
        **details,
    }
    path = _journal_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, canonical_json_bytes(payload))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_complete_cache(
    root: str | Path,
    *,
    verify_shard_checksums: bool = True,
) -> dict[str, Any] | None:
    """Return a verified manifest, ``None`` for incomplete, and reject bad seals."""

    path = Path(root).expanduser()
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_dir():
        raise UnsafeCacheRestartError(f"Cache root is not a real directory: {path}")
    complete_path = path / COMPLETE_FILENAME
    if not complete_path.exists():
        return None
    if not complete_path.is_file():
        raise UnsafeCacheRestartError(f"COMPLETE is not a regular file: {complete_path}")
    try:
        manifest = load_manifest(path, require_complete=True)
        for shard in manifest["shards"]:
            shard_path = path / str(shard["path"])
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            if shard_path.stat().st_size != int(shard["bytes"]):
                raise ValueError(f"Shard size mismatch: {shard_path}")
            if verify_shard_checksums and sha256_file(shard_path) != str(shard["sha256"]):
                raise ValueError(f"Shard SHA-256 mismatch: {shard_path}")
    except Exception as exc:
        # A present COMPLETE is a sealed-looking root.  Never delete it during
        # restart, even when corrupt; human intervention must choose a new root.
        raise UnsafeCacheRestartError(
            f"Refusing to modify cache with an invalid COMPLETE seal: {path}"
        ) from exc
    return manifest


def _safe_clear_incomplete_root(root: Path, marker: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise UnsafeCacheRestartError(f"Incomplete cache root is not a real directory: {root}")
    if (root / COMPLETE_FILENAME).exists():
        raise UnsafeCacheRestartError(f"Never clear a cache root containing COMPLETE: {root}")
    resolved_parent = root.parent.resolve()
    resolved_root = root.resolve()
    if resolved_root.parent != resolved_parent or not _PART_NAME_RE.fullmatch(root.name):
        raise UnsafeCacheRestartError(f"Unsafe restart cleanup target: {root}")
    if marker.parent.resolve() != resolved_parent or not marker.is_file():
        raise UnsafeCacheRestartError(f"Task-owned BUILDING marker is absent: {marker}")

    # Unlink entries without following symlinks, then remove only the exact
    # proven micro-part directory.  The marker itself remains outside the root.
    for directory, child_dirs, filenames in os.walk(root, topdown=False, followlinks=False):
        directory_path = Path(directory)
        for filename in filenames:
            (directory_path / filename).unlink()
        for child_name in child_dirs:
            child = directory_path / child_name
            if child.is_symlink():
                child.unlink()
            else:
                child.rmdir()
    root.rmdir()


def prepare_cache_build(
    root: str | Path,
    *,
    task_id: str,
    work_plan_sha256: str,
    role: str,
    micro_part_index: int,
    work_identity: Mapping[str, Any],
    verify_shard_checksums: bool = True,
) -> str:
    """Prepare an absent/incomplete part and return ``new``, ``restart``, or ``complete``."""

    path = Path(root).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = _identity_payload(
        path,
        task_id=task_id,
        work_plan_sha256=work_plan_sha256,
        role=role,
        micro_part_index=micro_part_index,
        work_identity=work_identity,
    )
    complete = verify_complete_cache(
        path,
        verify_shard_checksums=verify_shard_checksums,
    )
    marker = _marker_path(path)
    if complete is not None:
        _append_journal(path, identity, "verified-complete", total_frames=complete["total_frames"])
        return "complete"

    action = "new"
    if marker.exists():
        if _read_marker(marker) != identity:
            raise UnsafeCacheRestartError(
                f"BUILDING marker identity does not match requested work: {marker}"
            )
        action = "restart"
    else:
        write_immutable_file(marker, canonical_json_bytes(identity))
    if path.exists():
        _safe_clear_incomplete_root(path, marker)
        action = "restart"
    _append_journal(path, identity, f"prepare-{action}")
    return action


def _finish_cache_build(
    root: Path,
    identity: Mapping[str, Any],
    *,
    verify_shard_checksums: bool,
) -> dict[str, Any]:
    manifest = verify_complete_cache(
        root,
        verify_shard_checksums=verify_shard_checksums,
    )
    if manifest is None:
        raise RuntimeError(f"Build callback returned without sealing cache: {root}")
    marker = _marker_path(root)
    if marker.exists():
        if _read_marker(marker) != dict(identity):
            raise UnsafeCacheRestartError(f"BUILDING marker changed during build: {marker}")
        marker.unlink()
    _append_journal(root, identity, "sealed-and-verified", total_frames=manifest["total_frames"])
    return manifest


def run_single_micro_part(
    root: str | Path,
    *,
    task_id: str,
    work_plan_sha256: str,
    role: str,
    micro_part_index: int,
    work_identity: Mapping[str, Any],
    build: Callable[[], None],
    verify_existing_shard_checksums: bool = True,
    verify_new_shard_checksums: bool = False,
) -> dict[str, Any]:
    """Build one restart-managed cache part without a paired sibling cache."""

    path = Path(root).expanduser()
    identity = _identity_payload(
        path,
        task_id=task_id,
        work_plan_sha256=work_plan_sha256,
        role=role,
        micro_part_index=int(micro_part_index),
        work_identity=work_identity,
    )
    complete = verify_complete_cache(
        path,
        verify_shard_checksums=verify_existing_shard_checksums,
    )
    if complete is not None:
        complete = _finish_cache_build(
            path,
            identity,
            verify_shard_checksums=verify_existing_shard_checksums,
        )
        return {"status": "already-complete", role: complete}

    prepare_cache_build(
        path,
        task_id=task_id,
        work_plan_sha256=work_plan_sha256,
        role=role,
        micro_part_index=int(micro_part_index),
        work_identity=work_identity,
        verify_shard_checksums=verify_existing_shard_checksums,
    )
    _append_journal(path, identity, "single-build-start")
    try:
        build()
        manifest = _finish_cache_build(
            path,
            identity,
            verify_shard_checksums=verify_new_shard_checksums,
        )
    except Exception as exc:
        _append_journal(
            path,
            identity,
            "single-build-failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    return {"status": "built", role: manifest}


def run_paired_micro_part(
    canonical_root: str | Path,
    compact_root: str | Path,
    *,
    task_id: str,
    work_plan_sha256: str,
    micro_part_index: int,
    work_identity: Mapping[str, Any],
    build_both: Callable[[], None],
    recover_compact_from_canonical: Callable[[], None],
    verify_existing_shard_checksums: bool = True,
    verify_new_shard_checksums: bool = False,
) -> dict[str, Any]:
    """Run/recover one paired micro-part without recomputing sealed canonical data."""

    canonical_path = Path(canonical_root).expanduser()
    compact_path = Path(compact_root).expanduser()
    if canonical_path.name != compact_path.name:
        raise ValueError("Canonical and compact roots must share the same micro-part name")
    common = {
        "task_id": task_id,
        "work_plan_sha256": work_plan_sha256,
        "micro_part_index": int(micro_part_index),
        "work_identity": work_identity,
        "verify_shard_checksums": verify_existing_shard_checksums,
    }
    canonical_identity = _identity_payload(canonical_path, role="canonical", **{
        key: value for key, value in common.items() if key != "verify_shard_checksums"
    })
    compact_identity = _identity_payload(compact_path, role="compact", **{
        key: value for key, value in common.items() if key != "verify_shard_checksums"
    })

    canonical_complete = verify_complete_cache(
        canonical_path,
        verify_shard_checksums=verify_existing_shard_checksums,
    )
    compact_complete = verify_complete_cache(
        compact_path,
        verify_shard_checksums=verify_existing_shard_checksums,
    )
    if canonical_complete is not None and compact_complete is not None:
        return {
            "status": "already-complete",
            "canonical": canonical_complete,
            "compact": compact_complete,
        }
    if canonical_complete is None and compact_complete is not None:
        raise UnsafeCacheRestartError(
            "Compact micro-part is sealed while canonical micro-part is incomplete; "
            "refusing to delete either root"
        )

    if canonical_complete is not None:
        canonical_complete = _finish_cache_build(
            canonical_path,
            canonical_identity,
            verify_shard_checksums=verify_existing_shard_checksums,
        )
        prepare_cache_build(compact_path, role="compact", **common)
        _append_journal(compact_path, compact_identity, "recover-from-canonical-start")
        try:
            recover_compact_from_canonical()
            compact_manifest = _finish_cache_build(
                compact_path,
                compact_identity,
                verify_shard_checksums=verify_new_shard_checksums,
            )
        except Exception as exc:
            _append_journal(
                compact_path,
                compact_identity,
                "recover-from-canonical-failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        return {
            "status": "compact-recovered",
            "canonical": canonical_complete,
            "compact": compact_manifest,
        }

    prepare_cache_build(canonical_path, role="canonical", **common)
    prepare_cache_build(compact_path, role="compact", **common)
    _append_journal(canonical_path, canonical_identity, "paired-build-start")
    _append_journal(compact_path, compact_identity, "paired-build-start")
    try:
        build_both()
        canonical_manifest = _finish_cache_build(
            canonical_path,
            canonical_identity,
            verify_shard_checksums=verify_new_shard_checksums,
        )
        compact_manifest = _finish_cache_build(
            compact_path,
            compact_identity,
            verify_shard_checksums=verify_new_shard_checksums,
        )
    except Exception as exc:
        _append_journal(
            canonical_path,
            canonical_identity,
            "paired-build-failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        _append_journal(
            compact_path,
            compact_identity,
            "paired-build-failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    return {
        "status": "built",
        "canonical": canonical_manifest,
        "compact": compact_manifest,
    }
