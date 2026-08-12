"""Fail-closed terminal artifacts for formal training and full-model gates."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import ctypes
import errno
import fcntl
import functools
import secrets
import stat
from contextlib import contextmanager
from dataclasses import fields as dataclass_fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


SHA256_HEX = frozenset("0123456789abcdef")
N4_GATE_WORLD_SIZE = 32
N4_GATE_LOCAL_MICRO_BATCH_SIZE = 1
N4_GATE_GRADIENT_ACCUMULATION_STEPS = 1
N4_GATE_GLOBAL_TRAIN_BATCH_SIZE = 32
N4_GATE_TRAIN_STEPS = 2
N4_GATE_MAX_PEAK_ALLOCATED_BYTES = 42 * 2**30
N4_GATE_MAX_PEAK_RESERVED_BYTES = 44 * 2**30

ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT = "action_only_n2_1x8_v1"
ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT_VERSION = 1
ACTION_ONLY_N2_1X8_WORLD_SIZE = 8
ACTION_ONLY_N2_DEEPSPEED_VERSION = "0.18.5"
ACTION_ONLY_N2_1X8_RESERVATION_SCHEMA_VERSION = 2
ACTION_ONLY_N2_TASK_SCOPE_SCHEMA = "fastwam-action-only-n2-task-scope"
ACTION_ONLY_N2_TASK_SCOPE_SCHEMA_VERSION = 1
ACTION_ONLY_N2_RUN_PROFILES = frozenset({"paid_gate_1step", "formal_1k"})
ACTION_ONLY_N2_PAID_GATE_STEP = 1
ACTION_ONLY_N2_RELOAD_PROOF_DIR = "reload-proof"
ACTION_ONLY_N2_RELOAD_PROOF_SCHEMA_VERSION = 1
ACTION_ONLY_N2_RELOAD_LOAD_ATTEMPTS_DIR = "load-attempts"
ACTION_ONLY_N2_RELOAD_COMMITTED_ATTEMPT = "committed-load-attempt.json"
ACTION_ONLY_N2_TERMINAL_CANDIDATE = "paid-gate-terminal-candidate.json"
ACTION_ONLY_N2_RESERVATION_FIELDS = frozenset(
    {
        "base_code_commit",
        "bundle_manifest_sha256",
        "cache_manifest_sha256",
        "cache_selection_sha256",
        "cache_source_identity_sha256",
        "checkpoint_sha256",
        "checkpoint_state_kind",
        "code_commit",
        "cpfs_bundle_manifest_sha256",
        "effective_patched_tree",
        "erdma_bootstrap_sha256",
        "erdma_bundle_sha256",
        "erdma_env_sha256",
        "erdma_source_manifest_sha256",
        "formal_n4_fullmodel_gate",
        "global_world_size",
        "identity_sha256",
        "image_digest",
        "image_digest_status",
        "image_reference",
        "init_checkpoint_sha256",
        "n4_fullmodel_gate_complete_sha256",
        "nproc_per_node",
        "num_machines",
        "oss_bundle_manifest_sha256",
        "output_storage",
        "output_zero_checkpoint_smoke_sha256",
        "pyproject_sha256",
        "request_sha256",
        "run_id",
        "run_profile",
        "schema_version",
        "stats_sha256",
        "task",
        "task_scope_receipt",
        "task_scope_receipt_sha256",
        "trainable_scope",
        "training_env_bundle_manifest_sha256",
        "training_mode",
        "training_terminal_contract",
        "training_terminal_contract_version",
        "vae_sha256",
    }
)


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def require_sha256(value: str, *, label: str) -> str:
    value = str(value).strip().lower()
    if len(value) != 64 or any(character not in SHA256_HEX for character in value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def require_git_object_id(value: str, *, label: str) -> str:
    value = str(value).strip().lower()
    if len(value) != 40 or any(character not in SHA256_HEX for character in value):
        raise ValueError(f"{label} must be 40 lowercase hexadecimal characters")
    return value


def require_proof_attempt_id(value: str, *, label: str) -> str:
    value = str(value).strip()
    if len(value) != 32 or any(character not in SHA256_HEX for character in value):
        raise ValueError(f"{label} must be 32 lowercase hexadecimal characters")
    return value


def safe_relative_path(value: str) -> PurePosixPath:
    if not value or any(character in value for character in ("\x00", "\n", "\r")):
        raise ValueError(f"unsafe relative artifact path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe relative artifact path: {value!r}")
    return relative


def resolved_unaliased_directory(path: str | Path, *, label: str) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise ValueError(f"{label} must be absolute: {supplied}")
    absolute = Path(os.path.abspath(supplied))
    if supplied != absolute:
        raise ValueError(f"{label} must be an existing unaliased directory: {supplied}")
    descriptor = _open_directory_chain(absolute)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{label} must be a directory: {supplied}")
        proc_path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        if proc_path != absolute:
            raise ValueError(
                f"{label} must be an existing unaliased directory: {supplied}"
            )
    finally:
        os.close(descriptor)
    return absolute


def _open_directory_chain(path: str | Path) -> int:
    """Open every absolute path component without following any symlink."""

    absolute = Path(os.path.abspath(Path(path)))
    if not absolute.is_absolute():
        raise ValueError(f"directory path must be absolute: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(os.sep, flags)
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular(path: Path, *, require_single_link: bool = True) -> tuple[int, os.stat_result]:
    path = Path(os.path.abspath(path))
    parent_descriptor = _open_directory_chain(path.parent)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
    finally:
        os.close(parent_descriptor)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise ValueError(f"artifact must be a regular file: {path}")
    if require_single_link and info.st_nlink != 1:
        os.close(descriptor)
        raise ValueError(f"artifact must not be hard-linked: nlink={info.st_nlink} path={path}")
    return descriptor, info


def sha256_regular_file(path: str | Path, *, require_single_link: bool = True) -> tuple[str, int]:
    path = Path(path)
    descriptor, before = _open_regular(path, require_single_link=require_single_link)
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise RuntimeError(f"artifact changed while hashing: {path}")
        return digest.hexdigest(), int(after.st_size)
    finally:
        os.close(descriptor)


def read_canonical_json(path: str | Path) -> tuple[dict[str, Any], str, int]:
    path = Path(path)
    descriptor, before = _open_regular(path)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            encoded = handle.read()
        after = os.fstat(descriptor)
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"artifact changed while reading: {path}")
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    if encoded != canonical_json_bytes(payload):
        raise ValueError(f"JSON artifact is not canonical: {path}")
    return payload, hashlib.sha256(encoded).hexdigest(), len(encoded)


def _validate_reservation_identity(
    reservation: Mapping[str, Any], *, label: str
) -> str:
    reservation_identity_sha256 = require_sha256(
        reservation.get("identity_sha256", ""),
        label=f"{label} identity SHA-256",
    )
    reservation_identity_payload = dict(reservation)
    del reservation_identity_payload["identity_sha256"]
    if canonical_json_sha256(reservation_identity_payload) != reservation_identity_sha256:
        raise RuntimeError(f"{label} identity_sha256 does not match its payload")
    return reservation_identity_sha256


def _normalized_n2_dataset_task_scope(
    dataset_contract: Mapping[str, Any],
) -> tuple[list[int], list[str]]:
    if not isinstance(dataset_contract, Mapping):
        raise TypeError("N=2 terminal dataset contract must be a mapping")
    observed: dict[str, tuple[list[int], list[str]]] = {}
    for split in ("train", "val"):
        split_contract = dataset_contract.get(split)
        if not isinstance(split_contract, Mapping):
            raise TypeError(f"N=2 terminal dataset contract lacks mapping split {split!r}")
        raw_counts = split_contract.get("required_agent_counts")
        raw_tasks = split_contract.get("required_tasks")
        if not isinstance(raw_counts, list) or raw_counts != [2]:
            raise ValueError(
                f"N=2 terminal {split} dataset must require exactly [2], got {raw_counts!r}"
            )
        if (
            not isinstance(raw_tasks, list)
            or not raw_tasks
            or any(not isinstance(value, str) or not value.strip() for value in raw_tasks)
        ):
            raise ValueError(
                f"N=2 terminal {split} dataset task scope must be non-empty strings"
            )
        tasks = [value.strip() for value in raw_tasks]
        if tasks != sorted(set(tasks)):
            raise ValueError(
                f"N=2 terminal {split} dataset tasks must be unique and sorted: {tasks}"
            )
        observed[split] = ([2], tasks)
    if observed["train"] != observed["val"]:
        raise ValueError(
            "N=2 terminal train/val task scopes must match exactly: "
            f"train={observed['train']} val={observed['val']}"
        )
    return observed["train"]


def validate_action_only_n2_terminal_reservation(
    output_root: str | Path,
    *,
    run_id: str,
    base_code_commit: str,
    effective_patched_tree: str,
    request_sha256: str,
    init_checkpoint_sha256: str,
    world_size: int,
    formal_n4_fullmodel_gate: bool,
    checkpoint_state_kind: str,
    trainable_scope: str,
    training_mode: str,
    dataset_contract: Mapping[str, Any],
    task_scope_receipt_relative_path: str,
    run_profile: str,
) -> dict[str, Any]:
    """Validate the immutable caller authorization for the N=2 formal run.

    This is intentionally usable both at Trainer initialization and again by
    the terminal publisher.  A self-consistent marker is insufficient: the
    caller-provided request/tree/checkpoint pins and task-scope receipt must all
    match the runtime values supplied by the launcher.
    """

    output_root = resolved_unaliased_directory(output_root, label="training output root")
    base_code_commit = require_git_object_id(
        base_code_commit, label="N=2 terminal base code commit"
    )
    effective_patched_tree = require_git_object_id(
        effective_patched_tree, label="N=2 terminal effective patched tree"
    )
    request_sha256 = require_sha256(
        request_sha256, label="N=2 terminal request SHA-256"
    )
    init_checkpoint_sha256 = require_sha256(
        init_checkpoint_sha256, label="N=2 terminal initialization checkpoint SHA-256"
    )
    run_profile = str(run_profile).strip()
    if run_profile not in ACTION_ONLY_N2_RUN_PROFILES:
        raise ValueError(
            "N=2 terminal run_profile must be one of "
            f"{sorted(ACTION_ONLY_N2_RUN_PROFILES)}, got {run_profile!r}"
        )
    scalar_contract = {
        "world_size": (int(world_size), ACTION_ONLY_N2_1X8_WORLD_SIZE),
        "formal_n4_fullmodel_gate": (bool(formal_n4_fullmodel_gate), False),
        "checkpoint_state_kind": (str(checkpoint_state_kind), "sparse_delta"),
        "trainable_scope": (str(trainable_scope), "action"),
        "training_mode": (str(training_mode), "action_only_cache"),
    }
    mismatches = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in scalar_contract.items()
        if observed != expected
    }
    if mismatches:
        raise ValueError(f"N=2 action-only terminal scalar contract mismatch: {mismatches}")

    required_agent_counts, required_tasks = _normalized_n2_dataset_task_scope(
        dataset_contract
    )
    receipt_relative = safe_relative_path(task_scope_receipt_relative_path)
    if receipt_relative.as_posix() != task_scope_receipt_relative_path:
        raise ValueError(
            "N=2 task-scope receipt path must already be normalized: "
            f"{task_scope_receipt_relative_path!r}"
        )
    receipt, receipt_sha256, _ = read_canonical_json(output_root / receipt_relative)
    expected_receipt_fields = {
        "pins",
        "required_agent_counts",
        "required_tasks",
        "schema_name",
        "schema_version",
        "run_profile",
        "training_terminal_contract",
    }
    if set(receipt) != expected_receipt_fields:
        raise ValueError(
            "N=2 task-scope receipt fields mismatch: "
            f"expected={sorted(expected_receipt_fields)} observed={sorted(receipt)}"
        )
    pins = receipt.get("pins")
    expected_pin_fields = {
        "effective_patched_tree",
        "init_checkpoint_sha256",
        "request_sha256",
        "task_scope_id",
    }
    if not isinstance(pins, Mapping) or set(pins) != expected_pin_fields:
        raise ValueError(
            "N=2 task-scope receipt pin fields mismatch: "
            f"expected={sorted(expected_pin_fields)} "
            f"observed={sorted(pins) if isinstance(pins, Mapping) else type(pins)}"
        )
    task_scope_id = pins.get("task_scope_id")
    if (
        not isinstance(task_scope_id, str)
        or not task_scope_id.strip()
        or len(task_scope_id) > 128
        or any(character in task_scope_id for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError("N=2 task-scope receipt requires a non-empty safe task_scope_id")
    expected_receipt = {
        "schema_name": ACTION_ONLY_N2_TASK_SCOPE_SCHEMA,
        "schema_version": ACTION_ONLY_N2_TASK_SCOPE_SCHEMA_VERSION,
        "training_terminal_contract": ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT,
        "required_agent_counts": required_agent_counts,
        "required_tasks": required_tasks,
        "run_profile": run_profile,
        "pins": {
            "effective_patched_tree": effective_patched_tree,
            "init_checkpoint_sha256": init_checkpoint_sha256,
            "request_sha256": request_sha256,
            "task_scope_id": task_scope_id.strip(),
        },
    }
    if receipt != expected_receipt:
        raise RuntimeError(
            "N=2 task-scope receipt does not bind the runtime data/request pins: "
            f"expected={expected_receipt} observed={receipt}"
        )

    reservation, reservation_sha256, _ = read_canonical_json(
        output_root / ".RUN_RESERVED"
    )
    if set(reservation) != ACTION_ONLY_N2_RESERVATION_FIELDS:
        raise ValueError(
            "N=2 .RUN_RESERVED fields mismatch: "
            f"expected={sorted(ACTION_ONLY_N2_RESERVATION_FIELDS)} "
            f"observed={sorted(reservation)}"
        )
    reservation_identity_sha256 = _validate_reservation_identity(
        reservation, label="formal .RUN_RESERVED"
    )
    reservation_contract = {
        "base_code_commit": base_code_commit,
        "checkpoint_state_kind": "sparse_delta",
        "code_commit": base_code_commit,
        "effective_patched_tree": effective_patched_tree,
        "formal_n4_fullmodel_gate": False,
        "global_world_size": ACTION_ONLY_N2_1X8_WORLD_SIZE,
        "init_checkpoint_sha256": init_checkpoint_sha256,
        "nproc_per_node": 8,
        "num_machines": 1,
        "request_sha256": request_sha256,
        "run_profile": run_profile,
        "run_id": str(run_id),
        "schema_version": ACTION_ONLY_N2_1X8_RESERVATION_SCHEMA_VERSION,
        "task_scope_receipt": receipt_relative.as_posix(),
        "task_scope_receipt_sha256": receipt_sha256,
        "trainable_scope": "action",
        "training_mode": "action_only_cache",
        "training_terminal_contract": ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT,
        "training_terminal_contract_version": (
            ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT_VERSION
        ),
    }
    reservation_mismatches = {
        key: {"expected": expected, "observed": reservation.get(key)}
        for key, expected in reservation_contract.items()
        if reservation.get(key) != expected
    }
    if reservation_mismatches:
        raise RuntimeError(
            "formal .RUN_RESERVED does not authorize the N=2 action-only terminal run: "
            f"{reservation_mismatches}"
        )
    return {
        "contract": {
            "name": ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT,
            "run_profile": run_profile,
            "version": ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT_VERSION,
        },
        "reservation": {
            "identity_sha256": reservation_identity_sha256,
            "path": ".RUN_RESERVED",
            "sha256": reservation_sha256,
        },
        "task_scope": {
            "path": receipt_relative.as_posix(),
            "sha256": receipt_sha256,
            "required_agent_counts": required_agent_counts,
            "required_tasks": required_tasks,
            "pins": dict(pins),
        },
    }


def publish_exclusive_bytes(path: str | Path, payload: bytes) -> None:
    path = Path(os.path.abspath(Path(path)))
    if path.name in {"", ".", ".."}:
        raise ValueError(f"invalid formal artifact filename: {path}")
    parent_descriptor = _open_directory_chain(path.parent)
    temporary = f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        try:
            existing = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            raise FileExistsError(f"refusing to replace formal artifact: {path}")
        descriptor = os.open(temporary, flags, 0o440, dir_fd=parent_descriptor)
        try:
            written = 0
            while written < len(payload):
                written += os.write(descriptor, payload[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _rename_noreplace_at(
            parent_descriptor,
            temporary,
            parent_descriptor,
            path.name,
            display_path=path,
        )
        os.fsync(parent_descriptor)
    finally:
        try:
            os.unlink(temporary, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish ``source`` without ever replacing ``destination``."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "libc renameat2 is unavailable; refusing unsafe publish")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            f"refusing to replace formal artifact: {destination}",
            str(destination),
        )
    raise OSError(
        error_number,
        f"atomic no-clobber publication failed for {destination}; no fallback is allowed",
        str(destination),
    )


def _rename_noreplace_at(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
    *,
    display_path: Path,
) -> None:
    """dirfd-relative ``renameat2(RENAME_NOREPLACE)`` publication."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "libc renameat2 is unavailable; refusing unsafe publish")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_directory_fd,
        os.fsencode(source_name),
        destination_directory_fd,
        os.fsencode(destination_name),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            f"refusing to replace formal artifact: {display_path}",
            str(display_path),
        )
    raise OSError(
        error_number,
        f"atomic no-clobber publication failed for {display_path}; no fallback is allowed",
        str(display_path),
    )


def publish_exclusive_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    publish_exclusive_bytes(path, canonical_json_bytes(payload))


@contextmanager
def _training_terminal_lock(output_root: Path):
    """Serialize mutually exclusive run-level PASS/FAIL publication."""

    directory_fd = _open_directory_chain(output_root)
    flags = (
        os.O_RDONLY
        | os.O_CREAT
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(
        ".TRAINING.TERMINAL.lock", flags, 0o440, dir_fd=directory_fd
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(
                "training terminal lock must be a single-link regular file: "
                f"{output_root / '.TRAINING.TERMINAL.lock'}"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
            os.close(directory_fd)


def _with_training_terminal_lock(function):
    @functools.wraps(function)
    def wrapped(output_root: str | Path, *args, **kwargs):
        resolved = resolved_unaliased_directory(
            output_root, label="formal output root"
        )
        with _training_terminal_lock(resolved):
            return function(resolved, *args, **kwargs)

    return wrapped


@_with_training_terminal_lock
def publish_action_only_n2_reload_proof_record(
    output_root: str | Path,
    *,
    relative_path: str | PurePosixPath,
    payload: Mapping[str, Any],
) -> dict[str, str]:
    """Publish one N=2 reload record under the shared terminal lock."""

    relative = safe_relative_path(str(relative_path))
    parts = relative.parts
    save_names = {
        f"save-rank-{rank:05d}.json"
        for rank in range(ACTION_ONLY_N2_1X8_WORLD_SIZE)
    }
    load_names = {
        f"load-rank-{rank:05d}.json"
        for rank in range(ACTION_ONLY_N2_1X8_WORLD_SIZE)
    }
    valid_binding = parts == (
        ACTION_ONLY_N2_RELOAD_PROOF_DIR,
        "checkpoint-binding.json",
    )
    valid_save = (
        len(parts) == 2
        and parts[0] == ACTION_ONLY_N2_RELOAD_PROOF_DIR
        and parts[1] in save_names
    )
    valid_load = (
        len(parts) == 4
        and parts[0] == ACTION_ONLY_N2_RELOAD_PROOF_DIR
        and parts[1] == ACTION_ONLY_N2_RELOAD_LOAD_ATTEMPTS_DIR
        and parts[3] in load_names
    )
    if valid_load:
        require_proof_attempt_id(parts[2], label="N=2 reload load attempt id")
    if not (valid_binding or valid_save or valid_load):
        raise ValueError(f"unsupported N=2 reload proof record path: {relative}")
    destination = output_root / relative
    publish_exclusive_json(destination, payload)
    digest, _ = sha256_regular_file(destination)
    return {"path": relative.as_posix(), "sha256": digest}


@_with_training_terminal_lock
def publish_failure_marker(
    output_root: str | Path,
    *,
    marker_name: str,
    schema_name: str,
    error: BaseException,
    success_markers: Sequence[str],
) -> dict[str, Any]:
    """Publish a task-owned terminal failure signal without ever replacing PASS."""

    output_root = resolved_unaliased_directory(output_root, label="formal output root")
    for success_name in success_markers:
        success = output_root / safe_relative_path(success_name)
        if success.exists() or success.is_symlink():
            raise RuntimeError(
                f"refusing failure publication after a success marker exists: {success}"
            )
    marker_relative = safe_relative_path(marker_name)
    message = str(error)
    if len(message) > 4096:
        message = message[:4096] + "...[truncated]"
    payload = {
        "error_message": message,
        "error_type": type(error).__name__,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "run_id": os.environ.get("RUN_ID", ""),
        "schema_name": schema_name,
        "schema_version": 1,
        "status": "FAIL",
    }
    publish_exclusive_json(output_root / marker_relative, payload)
    return payload


def _canonical_fingerprint(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            digest.update(b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
            digest.update(b"\0")
            digest.update(array.tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda candidate: (type(candidate).__name__, repr(candidate))):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            for nested in item:
                update(nested)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            digest.update(repr(item).encode("utf-8"))
            digest.update(b"\0")
        else:
            buffer = io.BytesIO()
            torch.save(item, buffer)
            digest.update(b"torch-save\0")
            digest.update(buffer.getvalue())

    update(value)
    return digest.hexdigest()


def _sample_tensor(tensor: torch.Tensor, *, values_per_edge: int = 8) -> dict[str, Any]:
    flat = tensor.detach().reshape(-1)
    count = min(values_per_edge, int(flat.numel()))
    if count:
        sample = torch.cat((flat[:count], flat[-count:])).cpu().contiguous()
    else:
        sample = torch.empty(0, dtype=tensor.dtype)
    return {
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "sample_sha256": _canonical_fingerprint(sample),
        "shape": list(tensor.shape),
    }


def _tensor_probe(tensor: torch.Tensor, *, full_state: bool) -> dict[str, Any]:
    if not full_state:
        return _sample_tensor(tensor)
    return {
        "bytes": int(tensor.numel()) * int(tensor.element_size()),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "sha256": _canonical_fingerprint(tensor),
        "shape": list(tensor.shape),
    }


def model_probe(
    model: torch.nn.Module, *, limit: int = 8, full_state: bool = False
) -> dict[str, Any]:
    if full_state:
        parameter_names = {name for name, _ in model.named_parameters()}
        buffer_names = {name for name, _ in model.named_buffers()}
        records: list[dict[str, Any]] = []
        for name, value in sorted(model.state_dict().items(), key=lambda item: item[0]):
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    "N=2 full model state proof only accepts tensor state_dict "
                    f"entries, got {type(value).__name__} for {name!r}"
                )
            kind = (
                "parameter"
                if name in parameter_names
                else "buffer"
                if name in buffer_names
                else "extra"
            )
            records.append(
                {
                    "name": name,
                    "kind": kind,
                    **_tensor_probe(value, full_state=True),
                }
            )
        if not records:
            raise RuntimeError("formal state probe found an empty model state_dict")
        inventory = {
            "buffer_count": sum(record["kind"] == "buffer" for record in records),
            "extra_count": sum(record["kind"] == "extra" for record in records),
            "inventory_count": len(records),
            "parameter_count": sum(
                record["kind"] == "parameter" for record in records
            ),
            "total_bytes": sum(int(record["bytes"]) for record in records),
            "total_numel": sum(int(record["numel"]) for record in records),
        }
        body = {
            "coverage": "full_state_dict",
            "inventory": inventory,
            "records": records,
        }
        return {**body, "fingerprint": canonical_json_sha256(body)}

    parameters = sorted(
        ((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad),
        key=lambda item: item[0],
    )
    if not parameters:
        raise RuntimeError("formal state probe found no trainable model parameters")
    selected = (
        parameters
        if full_state
        else parameters[: max(limit // 2, 1)] + parameters[-max(limit // 2, 1) :]
    )
    deduplicated: list[tuple[str, torch.Tensor]] = []
    observed: set[str] = set()
    for name, parameter in selected:
        if name not in observed:
            observed.add(name)
            deduplicated.append((name, parameter))
    records = [
        {"name": name, **_tensor_probe(parameter, full_state=full_state)}
        for name, parameter in deduplicated
    ]
    payload = {
        "fingerprint": canonical_json_sha256({"records": records}),
        "records": records,
        "trainable_parameter_count": len(parameters),
    }
    return payload


def _optimizer_with_state(optimizer: Any) -> Any:
    current = optimizer
    visited: set[int] = set()
    candidates = []
    for _ in range(12):
        if id(current) in visited:
            break
        visited.add(id(current))
        candidates.append(current)
        nested = getattr(current, "optimizer", None)
        if nested is None or nested is current:
            break
        current = nested
    for candidate in reversed(candidates):
        state = getattr(candidate, "state", None)
        groups = getattr(candidate, "param_groups", None)
        if isinstance(state, Mapping) and state and isinstance(groups, Sequence):
            return candidate
    raise RuntimeError("formal state probe found no populated optimizer state")


def _optimizer_chain(optimizer: Any) -> list[Any]:
    current = optimizer
    observed: set[int] = set()
    candidates: list[Any] = []
    for _ in range(12):
        if id(current) in observed:
            break
        observed.add(id(current))
        candidates.append(current)
        nested = getattr(current, "optimizer", None)
        if nested is None or nested is current:
            break
        current = nested
    return candidates


def _optimizer_scalar(value: Any) -> tuple[str, Any] | None:
    if value is None:
        return "none", None
    if isinstance(value, bool):
        return "bool", value
    if isinstance(value, int):
        return "int", value
    if isinstance(value, float):
        return "float_hex", value.hex()
    if isinstance(value, str):
        return "str", value
    if isinstance(value, torch.dtype):
        return "torch_dtype", str(value)
    if isinstance(value, torch.device):
        return "torch_device", str(value)
    return None


def _encode_optimizer_value(
    value: Any,
    *,
    counters: dict[str, int],
    count_value: bool = True,
    tensor_cache: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        if tensor_cache is None:
            tensor_cache = {}
        record = tensor_cache.get(id(value))
        if record is None:
            record = {"kind": "tensor", **_tensor_probe(value, full_state=True)}
            tensor_cache[id(value)] = record
        if count_value:
            counters["tensor_count"] += 1
            counters["total_bytes"] += int(record["bytes"])
        return record
    scalar = _optimizer_scalar(value)
    if scalar is not None:
        scalar_type, normalized = scalar
        if count_value:
            counters["scalar_count"] += 1
        return {"kind": "scalar", "type": scalar_type, "value": normalized}
    if isinstance(value, Mapping):
        entries = [
            {
                "key": _encode_optimizer_value(
                    key,
                    counters=counters,
                    count_value=False,
                    tensor_cache=tensor_cache,
                ),
                "value": _encode_optimizer_value(
                    nested,
                    counters=counters,
                    count_value=count_value,
                    tensor_cache=tensor_cache,
                ),
            }
            for key, nested in value.items()
        ]
        entries.sort(key=lambda entry: canonical_json_bytes(entry["key"]))
        return {"entries": entries, "kind": "mapping"}
    if isinstance(value, (list, tuple)):
        return {
            "items": [
                _encode_optimizer_value(
                    nested,
                    counters=counters,
                    count_value=count_value,
                    tensor_cache=tensor_cache,
                )
                for nested in value
            ],
            "kind": "tuple" if isinstance(value, tuple) else "list",
        }
    concrete_type = f"{type(value).__module__}.{type(value).__qualname__}"
    if is_dataclass(value) and not isinstance(value, type):
        object_state = {
            field.name: getattr(value, field.name) for field in dataclass_fields(value)
        }
        return {
            "concrete_type": concrete_type,
            "kind": "object_state",
            "state": _encode_optimizer_value(
                object_state,
                counters=counters,
                count_value=count_value,
                tensor_cache=tensor_cache,
            ),
        }
    if type(value).__module__.startswith("deepspeed.runtime.fp16.loss_scaler"):
        return {
            "concrete_type": concrete_type,
            "kind": "object_state",
            "state": _encode_optimizer_value(
                dict(vars(value)),
                counters=counters,
                count_value=count_value,
                tensor_cache=tensor_cache,
            ),
        }
    raise TypeError(
        "unsupported optimizer state value for exact N=2 proof: " + concrete_type
    )


def _is_deepspeed_zero2_optimizer(candidate: Any) -> bool:
    return (
        type(candidate).__module__ == "deepspeed.runtime.zero.stage_1_and_2"
        and type(candidate).__qualname__ == "DeepSpeedZeroOptimizer"
        and getattr(candidate, "partition_gradients", None) is True
        and callable(getattr(candidate, "state_dict", None))
    )


def _deepspeed_zero2_optimizer_probe(
    concrete: Any, *, require_populated_state: bool
) -> dict[str, Any]:
    state_dict = concrete.state_dict()
    required_keys = {
        "base_optimizer_state",
        "clip_grad",
        "ds_version",
        "dynamic_loss_scale",
        "group_paddings",
        "loss_scaler",
        "overflow",
        "param_slice_mappings",
        "partition_count",
        "single_partition_of_fp32_groups",
        "zero_stage",
    }
    optional_keys = {"universal_checkpoint_info"}
    if (
        not isinstance(state_dict, Mapping)
        or not all(isinstance(key, str) for key in state_dict)
        or not required_keys.issubset(state_dict)
        or set(state_dict) - required_keys - optional_keys
    ):
        raise ValueError(
            "DeepSpeed ZeRO-2 state_dict fields mismatch: "
            f"observed={sorted(map(str, state_dict)) if isinstance(state_dict, Mapping) else type(state_dict).__name__}"
        )
    zero_stage = state_dict["zero_stage"]
    if isinstance(zero_stage, bool) or not isinstance(zero_stage, int) or int(zero_stage) != 2:
        raise ValueError(f"DeepSpeed optimizer is not ZeRO-2: zero_stage={zero_stage!r}")
    if state_dict["ds_version"] != ACTION_ONLY_N2_DEEPSPEED_VERSION:
        raise ValueError(
            "DeepSpeed version mismatch for the N=2 proof: "
            f"expected={ACTION_ONLY_N2_DEEPSPEED_VERSION!r} "
            f"observed={state_dict['ds_version']!r}"
        )
    for field in ("dynamic_loss_scale", "overflow"):
        if type(state_dict[field]) is not bool:
            raise TypeError(f"DeepSpeed ZeRO-2 {field} must be a boolean")
    clip_grad = state_dict["clip_grad"]
    if (
        isinstance(clip_grad, bool)
        or not isinstance(clip_grad, (int, float))
        or not math.isfinite(float(clip_grad))
        or float(clip_grad) < 0.0
    ):
        raise ValueError("DeepSpeed ZeRO-2 clip_grad must be finite and non-negative")
    if not type(state_dict["loss_scaler"]).__module__.startswith(
        "deepspeed.runtime.fp16.loss_scaler"
    ):
        raise TypeError("DeepSpeed ZeRO-2 loss_scaler has an unexpected concrete type")
    base_optimizer_state = state_dict["base_optimizer_state"]
    if (
        not isinstance(base_optimizer_state, Mapping)
        or set(base_optimizer_state) != {"state", "param_groups"}
        or not isinstance(base_optimizer_state["state"], Mapping)
        or not isinstance(base_optimizer_state["param_groups"], list)
    ):
        raise TypeError("DeepSpeed ZeRO-2 base optimizer state is invalid")
    if require_populated_state and not base_optimizer_state["state"]:
        raise RuntimeError("formal state probe found no populated optimizer state")
    fp32_partitions = state_dict["single_partition_of_fp32_groups"]
    if (
        not isinstance(fp32_partitions, (list, tuple))
        or not fp32_partitions
        or any(
            not isinstance(partition, torch.Tensor)
            or partition.dtype != torch.float32
            or partition.numel() <= 0
            for partition in fp32_partitions
        )
    ):
        raise RuntimeError(
            "DeepSpeed ZeRO-2 proof requires non-empty rank-local FP32 master partitions"
        )
    expected_group_count = len(fp32_partitions)
    if expected_group_count != len(base_optimizer_state["param_groups"]):
        raise ValueError(
            "DeepSpeed ZeRO-2 FP32 master partition count does not match base "
            "optimizer parameter groups"
        )
    partition_count = state_dict["partition_count"]
    if (
        not isinstance(partition_count, list)
        or len(partition_count) != expected_group_count
        or any(
            type(count) is not int
            or count != ACTION_ONLY_N2_1X8_WORLD_SIZE
            for count in partition_count
        )
    ):
        raise ValueError(
            "DeepSpeed ZeRO-2 partition count must contain one exact world-size "
            f"entry per FP32 master group: {partition_count!r}"
        )
    group_paddings = state_dict["group_paddings"]
    if (
        not isinstance(group_paddings, list)
        or len(group_paddings) != expected_group_count
        or any(type(padding) is not int or padding < 0 for padding in group_paddings)
    ):
        raise ValueError(
            "DeepSpeed ZeRO-2 group paddings must contain one non-negative integer "
            "per FP32 master group"
        )
    param_slice_mappings = state_dict["param_slice_mappings"]
    if (
        not isinstance(param_slice_mappings, list)
        or len(param_slice_mappings) != expected_group_count
        or any(not isinstance(mapping, Mapping) for mapping in param_slice_mappings)
    ):
        raise TypeError(
            "DeepSpeed ZeRO-2 parameter slice mappings must contain one mapping "
            "per FP32 master group"
        )

    counters = {"tensor_count": 0, "scalar_count": 0, "total_bytes": 0}
    encoded = _encode_optimizer_value(
        state_dict,
        counters=counters,
        tensor_cache={},
    )
    inventory = {
        "base_optimizer_param_group_count": len(
            base_optimizer_state["param_groups"]
        ),
        "base_optimizer_state_parameter_count": len(base_optimizer_state["state"]),
        "fp32_master_partition_count": len(fp32_partitions),
        "fp32_master_partition_numel": sum(
            int(partition.numel()) for partition in fp32_partitions
        ),
        "fp32_master_partition_total_bytes": sum(
            int(partition.numel() * partition.element_size())
            for partition in fp32_partitions
        ),
        "state_dict_scalar_count": counters["scalar_count"],
        "state_dict_tensor_count": counters["tensor_count"],
        "state_dict_total_bytes": counters["total_bytes"],
    }
    body = {
        "concrete_type": (
            "deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer"
        ),
        "coverage": "rank_local_deepspeed_zero2_state_dict",
        "inventory": inventory,
        "state_dict": encoded,
    }
    return {**body, "fingerprint": canonical_json_sha256(body)}


def _full_optimizer_probe(
    optimizer: Any, *, require_populated_state: bool
) -> dict[str, Any]:
    candidates = _optimizer_chain(optimizer)
    zero2_candidates = [
        candidate for candidate in candidates if _is_deepspeed_zero2_optimizer(candidate)
    ]
    if zero2_candidates:
        if len(zero2_candidates) != 1:
            raise RuntimeError(
                "formal state probe found multiple DeepSpeed ZeRO-2 optimizer wrappers"
            )
        return _deepspeed_zero2_optimizer_probe(
            zero2_candidates[0], require_populated_state=require_populated_state
        )
    concrete = None
    for candidate in reversed(candidates):
        if callable(getattr(candidate, "state_dict", None)) and isinstance(
            getattr(candidate, "param_groups", None), Sequence
        ):
            concrete = candidate
            break
    if concrete is None:
        raise RuntimeError("formal state probe found no concrete optimizer state_dict")
    state_dict = concrete.state_dict()
    if not isinstance(state_dict, Mapping) or set(state_dict) != {"state", "param_groups"}:
        raise ValueError(
            "N=2 optimizer state_dict must contain exactly state and param_groups"
        )
    raw_state = state_dict["state"]
    raw_groups = state_dict["param_groups"]
    if not isinstance(raw_state, Mapping) or not isinstance(raw_groups, list):
        raise TypeError("N=2 optimizer state_dict has invalid state/param_groups")
    if require_populated_state and not raw_state:
        raise RuntimeError("formal state probe found no populated optimizer state")

    state_counters = {"tensor_count": 0, "scalar_count": 0, "total_bytes": 0}
    state_records = []
    for parameter_id, value in raw_state.items():
        if isinstance(parameter_id, bool) or not isinstance(parameter_id, int):
            raise TypeError(
                f"N=2 optimizer parameter id must be an integer: {parameter_id!r}"
            )
        state_records.append(
            {
                "parameter_id": parameter_id,
                "state": _encode_optimizer_value(
                    value, counters=state_counters, count_value=True
                ),
            }
        )
    state_records.sort(key=lambda record: record["parameter_id"])

    group_counters = {"tensor_count": 0, "scalar_count": 0, "total_bytes": 0}
    param_groups = []
    for group_index, group in enumerate(raw_groups):
        if not isinstance(group, Mapping) or "params" not in group:
            raise TypeError(f"N=2 optimizer param group {group_index} is invalid")
        parameter_ids = group["params"]
        if (
            not isinstance(parameter_ids, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in parameter_ids)
        ):
            raise TypeError(
                f"N=2 optimizer param group {group_index} has invalid parameter ids"
            )
        hyperparameters = {key: value for key, value in group.items() if key != "params"}
        param_groups.append(
            {
                "group_index": group_index,
                "hyperparameters": _encode_optimizer_value(
                    hyperparameters, counters=group_counters, count_value=True
                ),
                "parameter_ids": list(parameter_ids),
            }
        )
    inventory = {
        "param_group_count": len(param_groups),
        "param_group_parameter_count": sum(
            len(group["parameter_ids"]) for group in param_groups
        ),
        "param_group_scalar_count": group_counters["scalar_count"],
        "param_group_tensor_count": group_counters["tensor_count"],
        "param_group_total_bytes": group_counters["total_bytes"],
        "state_parameter_count": len(state_records),
        "state_scalar_count": state_counters["scalar_count"],
        "state_tensor_count": state_counters["tensor_count"],
        "state_total_bytes": state_counters["total_bytes"],
    }
    body = {
        "concrete_type": f"{type(concrete).__module__}.{type(concrete).__qualname__}",
        "coverage": "rank_local_full_state_dict",
        "inventory": inventory,
        "param_groups": param_groups,
        "state_records": state_records,
    }
    return {**body, "fingerprint": canonical_json_sha256(body)}


def optimizer_probe(
    optimizer: Any,
    *,
    limit: int = 8,
    require_populated_state: bool = True,
    full_state: bool = False,
) -> dict[str, Any]:
    if full_state:
        return _full_optimizer_probe(
            optimizer, require_populated_state=require_populated_state
        )
    try:
        concrete = _optimizer_with_state(optimizer)
    except RuntimeError:
        if require_populated_state:
            raise
        groups = getattr(optimizer, "param_groups", [])
        summary = [
            {
                str(key): value
                for key, value in sorted(group.items())
                if key != "params" and isinstance(value, (str, int, float, bool, type(None)))
            }
            for group in groups
        ]
        payload = {
            "concrete_type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "fingerprint": canonical_json_sha256({"empty_param_groups": summary}),
            "records": [],
        }
        return payload
    records = []
    for group_index, group in enumerate(concrete.param_groups):
        for parameter_index, parameter in enumerate(group.get("params", [])):
            state = concrete.state.get(parameter)
            if not state:
                continue
            sampled_state = {}
            for key in sorted(state, key=lambda candidate: str(candidate)):
                value = state[key]
                sampled_state[str(key)] = (
                    _tensor_probe(value, full_state=full_state)
                    if isinstance(value, torch.Tensor)
                    else value
                )
            records.append(
                {
                    "group_index": group_index,
                    "parameter_index": parameter_index,
                    "state": sampled_state,
                }
            )
            if not full_state and len(records) >= limit:
                break
        if not full_state and len(records) >= limit:
            break
    if not records:
        raise RuntimeError("formal state probe found no sampleable optimizer tensors")
    payload = {
        "concrete_type": f"{type(concrete).__module__}.{type(concrete).__qualname__}",
        "fingerprint": canonical_json_sha256({"records": records}),
        "records": records,
    }
    return payload


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state(torch.cuda.current_device())
    return state


def next_rng_sample(device: torch.device) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "numpy": np.random.random(4).tolist(),
        "python": [random.random() for _ in range(4)],
        "torch_cpu": torch.rand(4, device="cpu").tolist(),
    }
    if device.type == "cuda":
        sample["torch_cuda"] = torch.rand(4, device=device).cpu().tolist()
    return sample


def state_fingerprints(
    *,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    global_step: int,
    require_optimizer_state: bool = True,
    full_state: bool = False,
) -> dict[str, Any]:
    model_state = model_probe(model, full_state=full_state)
    optimizer_state = optimizer_probe(
        optimizer,
        require_populated_state=require_optimizer_state,
        full_state=full_state,
    )
    scheduler_state = scheduler.state_dict()
    return {
        "global_step": int(global_step),
        "model": model_state["fingerprint"],
        "model_probe": model_state,
        "optimizer": optimizer_state["fingerprint"],
        "optimizer_probe": optimizer_state,
        "rng": _canonical_fingerprint(rng_state()),
        "scheduler": _canonical_fingerprint(scheduler_state),
    }


def _inventory_regular_tree(root: Path) -> dict[PurePosixPath, Path]:
    """Inventory a tree using only dirfd-relative, no-follow traversal."""

    root_fd = _open_directory_chain(root)
    observed: dict[PurePosixPath, Path] = {}
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )

    def walk(directory_fd: int, prefix: PurePosixPath | None) -> None:
        for name in sorted(os.listdir(directory_fd), key=os.fsencode):
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            relative = PurePosixPath(name) if prefix is None else prefix / name
            safe_relative_path(relative.as_posix())
            if stat.S_ISDIR(info.st_mode):
                child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                try:
                    walk(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                observed[relative] = root / relative
            else:
                raise ValueError(
                    "state tree contains an aliased/special entry: "
                    f"{root / relative}"
                )

    try:
        walk(root_fd, None)
    finally:
        os.close(root_fd)
    return observed


def _validate_state_tree_metadata(state_root: Path, manifest_path: Path) -> dict[str, Any]:
    state_root = resolved_unaliased_directory(
        state_root, label="sealed training state root"
    )
    state_parent = resolved_unaliased_directory(
        state_root.parent, label="sealed training state parent"
    )
    if manifest_path.parent != state_parent:
        raise ValueError(
            "state-tree manifest must be adjacent to the unaliased state root: "
            f"root={state_root} manifest={manifest_path}"
        )
    payload, manifest_sha256, _ = read_canonical_json(manifest_path)
    if set(payload) != {"files", "role", "schema_version", "total_bytes"}:
        raise ValueError(f"state-tree manifest fields mismatch: {manifest_path}")
    if payload["schema_version"] != 1 or payload["role"] != "accelerate_zero2_full_state":
        raise ValueError(f"state-tree manifest role/schema mismatch: {manifest_path}")
    records = payload["files"]
    if not isinstance(records, list) or not records:
        raise ValueError(f"state-tree manifest is empty: {manifest_path}")
    expected: dict[PurePosixPath, tuple[int, str]] = {}
    previous: bytes | None = None
    for record in records:
        if not isinstance(record, dict) or set(record) != {"bytes", "path", "sha256"}:
            raise ValueError(f"invalid state-tree record: {manifest_path}")
        relative = safe_relative_path(record["path"])
        key = os.fsencode(relative.as_posix())
        if previous is not None and key <= previous:
            raise ValueError(f"state-tree records are not unique and sorted: {manifest_path}")
        previous = key
        size = record["bytes"]
        digest = require_sha256(record["sha256"], label=f"state file {relative} SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid state file size for {relative}: {size!r}")
        expected[relative] = (size, digest)
    if PurePosixPath("trainer_state.json") not in expected:
        raise ValueError(f"state-tree manifest does not bind trainer_state.json: {manifest_path}")
    observed = _inventory_regular_tree(state_root)
    if set(observed) != set(expected):
        missing = sorted(path.as_posix() for path in set(expected) - set(observed))
        unexpected = sorted(path.as_posix() for path in set(observed) - set(expected))
        raise RuntimeError(
            f"state tree inventory mismatch: missing={missing[:12]} unexpected={unexpected[:12]}"
        )
    total_bytes = 0
    for relative, (expected_size, expected_sha256) in expected.items():
        actual_sha256, actual_size = sha256_regular_file(observed[relative])
        if actual_size != expected_size:
            raise RuntimeError(
                f"state file size mismatch: expected={expected_size} actual={actual_size} "
                f"path={observed[relative]}"
            )
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "state file SHA-256 mismatch during terminal strong readback: "
                f"expected={expected_sha256} actual={actual_sha256} "
                f"path={observed[relative]}"
            )
        total_bytes += actual_size
    if payload["total_bytes"] != total_bytes:
        raise RuntimeError(
            f"state-tree total mismatch: expected={payload['total_bytes']} actual={total_bytes}"
        )
    return {
        "file_count": len(expected),
        "manifest_sha256": manifest_sha256,
        "total_bytes": total_bytes,
    }


def checkpoint_seal_descriptor(
    output_root: str | Path,
    *,
    step: int,
    rehash_weights: bool,
    expected_checkpoint_state_kind: str = "full",
) -> dict[str, Any]:
    expected_checkpoint_state_kind = str(expected_checkpoint_state_kind).strip().lower()
    if expected_checkpoint_state_kind not in {"full", "sparse_delta"}:
        raise ValueError(
            "expected checkpoint state kind must be 'full' or 'sparse_delta', got "
            f"{expected_checkpoint_state_kind!r}"
        )
    output_root = resolved_unaliased_directory(output_root, label="training output root")
    tag = f"step_{int(step):06d}"
    checkpoints_root = resolved_unaliased_directory(
        output_root / "checkpoints", label="sealed checkpoints root"
    )
    weights_root = resolved_unaliased_directory(
        checkpoints_root / "weights", label="sealed weights root"
    )
    state_parent = resolved_unaliased_directory(
        checkpoints_root / "state", label="sealed state parent"
    )
    weights = weights_root / f"{tag}.pt"
    weights_manifest = weights.with_name(f"{weights.name}.manifest.json")
    weights_complete = weights.with_name(f"{weights.name}.COMPLETE")
    state_root = state_parent / tag
    state_manifest = state_root.with_name(f"{tag}.state-tree.json")

    manifest_payload, weights_manifest_sha256, _ = read_canonical_json(weights_manifest)
    if set(manifest_payload) != {
        "bytes",
        "checkpoint_state_kind",
        "filename",
        "global_step",
        "schema_name",
        "schema_version",
        "sha256",
    }:
        raise ValueError(f"weights manifest fields mismatch: {weights_manifest}")
    expected_checkpoint_sha256 = require_sha256(
        manifest_payload["sha256"], label="weights checkpoint SHA-256"
    )
    if (
        manifest_payload["schema_name"] != "fastwam-weights-checkpoint"
        or manifest_payload["schema_version"] != 1
        or manifest_payload["filename"] != weights.name
        or manifest_payload["global_step"] != int(step)
        or manifest_payload["checkpoint_state_kind"]
        != expected_checkpoint_state_kind
    ):
        raise ValueError(f"weights manifest semantic mismatch: {weights_manifest}")
    descriptor, info = _open_regular(weights)
    os.close(descriptor)
    if info.st_size != manifest_payload["bytes"]:
        raise RuntimeError(f"weights checkpoint byte-size mismatch: {weights}")
    if rehash_weights:
        actual_checkpoint_sha256, _ = sha256_regular_file(weights)
        if actual_checkpoint_sha256 != expected_checkpoint_sha256:
            raise RuntimeError(
                "weights checkpoint SHA-256 mismatch: "
                f"expected={expected_checkpoint_sha256} actual={actual_checkpoint_sha256}"
            )

    complete_payload, weights_complete_sha256, _ = read_canonical_json(weights_complete)
    if set(complete_payload) != {
        "checkpoint_sha256",
        "manifest_filename",
        "manifest_sha256",
        "schema_name",
        "schema_version",
    }:
        raise ValueError(f"weights COMPLETE fields mismatch: {weights_complete}")
    if (
        complete_payload["schema_name"] != "fastwam-weights-checkpoint-complete"
        or complete_payload["schema_version"] != 1
        or complete_payload["manifest_filename"] != weights_manifest.name
        or complete_payload["manifest_sha256"] != weights_manifest_sha256
        or complete_payload["checkpoint_sha256"] != expected_checkpoint_sha256
    ):
        raise RuntimeError(f"weights COMPLETE does not bind its manifest/checkpoint: {weights_complete}")

    state_summary = _validate_state_tree_metadata(state_root, state_manifest)
    trainer_state = state_root / "trainer_state.json"
    trainer_payload, trainer_state_sha256, _ = read_canonical_json(trainer_state)
    if trainer_payload.get("global_step") != int(step):
        raise RuntimeError(f"trainer_state global_step mismatch: {trainer_state}")

    def relative(path: Path) -> str:
        return path.relative_to(output_root).as_posix()

    return {
        "global_step": int(step),
        "state": {
            "file_count": state_summary["file_count"],
            "manifest": relative(state_manifest),
            "manifest_sha256": state_summary["manifest_sha256"],
            "root": relative(state_root),
            "total_bytes": state_summary["total_bytes"],
            "trainer_state_sha256": trainer_state_sha256,
        },
        "weights": {
            "bytes": int(info.st_size),
            "checkpoint": relative(weights),
            "checkpoint_sha256": expected_checkpoint_sha256,
            "complete": relative(weights_complete),
            "complete_sha256": weights_complete_sha256,
            "manifest": relative(weights_manifest),
            "manifest_sha256": weights_manifest_sha256,
            "rehash_verified": bool(rehash_weights),
        },
    }


def _validate_n2_tensor_record(
    record: Any, *, label: str, expected_fields: set[str]
) -> tuple[int, int]:
    if not isinstance(record, dict) or set(record) != expected_fields:
        raise ValueError(f"{label} tensor record fields mismatch")
    dtype_name = record.get("dtype")
    shape = record.get("shape")
    numel = record.get("numel")
    byte_count = record.get("bytes")
    if not isinstance(dtype_name, str) or not dtype_name.startswith("torch."):
        raise ValueError(f"{label} tensor dtype is invalid: {dtype_name!r}")
    dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{label} tensor dtype is unsupported: {dtype_name!r}")
    if (
        not isinstance(shape, list)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape)
        or isinstance(numel, bool)
        or not isinstance(numel, int)
        or numel < 0
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise ValueError(f"{label} tensor shape/size is invalid")
    expected_numel = 1
    for dimension in shape:
        expected_numel *= dimension
    expected_bytes = expected_numel * torch.empty((), dtype=dtype).element_size()
    if numel != expected_numel or byte_count != expected_bytes:
        raise RuntimeError(
            f"{label} tensor inventory mismatch: numel={numel}/{expected_numel} "
            f"bytes={byte_count}/{expected_bytes}"
        )
    require_sha256(record.get("sha256", ""), label=f"{label} tensor SHA-256")
    return numel, byte_count


def _validate_optimizer_encoded_value(
    payload: Any,
    *,
    label: str,
    counters: dict[str, int],
    count_value: bool = True,
) -> None:
    if not isinstance(payload, dict) or "kind" not in payload:
        raise ValueError(f"{label} encoded optimizer value is invalid")
    kind = payload["kind"]
    if kind == "tensor":
        _validate_n2_tensor_record(
            payload,
            label=label,
            expected_fields={"bytes", "dtype", "kind", "numel", "sha256", "shape"},
        )
        if count_value:
            counters["tensor_count"] += 1
            counters["total_bytes"] += int(payload["bytes"])
        return
    if kind == "scalar":
        if set(payload) != {"kind", "type", "value"}:
            raise ValueError(f"{label} optimizer scalar fields mismatch")
        scalar_type = payload["type"]
        value = payload["value"]
        valid = {
            "none": value is None,
            "bool": isinstance(value, bool),
            "int": isinstance(value, int) and not isinstance(value, bool),
            "float_hex": isinstance(value, str),
            "str": isinstance(value, str),
            "torch_dtype": isinstance(value, str) and value.startswith("torch."),
            "torch_device": isinstance(value, str) and bool(value),
        }
        if scalar_type not in valid or not valid[scalar_type]:
            raise ValueError(f"{label} optimizer scalar is invalid")
        if scalar_type == "float_hex":
            try:
                float.fromhex(value)
            except ValueError as error:
                raise ValueError(f"{label} optimizer float scalar is invalid") from error
        if count_value:
            counters["scalar_count"] += 1
        return
    if kind == "mapping":
        if set(payload) != {"entries", "kind"} or not isinstance(
            payload["entries"], list
        ):
            raise ValueError(f"{label} optimizer mapping fields mismatch")
        previous: bytes | None = None
        for index, entry in enumerate(payload["entries"]):
            if not isinstance(entry, dict) or set(entry) != {"key", "value"}:
                raise ValueError(f"{label} optimizer mapping entry {index} is invalid")
            key_bytes = canonical_json_bytes(entry["key"])
            if previous is not None and key_bytes <= previous:
                raise ValueError(f"{label} optimizer mapping keys are not unique/sorted")
            previous = key_bytes
            _validate_optimizer_encoded_value(
                entry["key"],
                label=f"{label} key {index}",
                counters=counters,
                count_value=False,
            )
            _validate_optimizer_encoded_value(
                entry["value"],
                label=f"{label} value {index}",
                counters=counters,
                count_value=count_value,
            )
        return
    if kind in {"list", "tuple"}:
        if set(payload) != {"items", "kind"} or not isinstance(payload["items"], list):
            raise ValueError(f"{label} optimizer sequence fields mismatch")
        for index, item in enumerate(payload["items"]):
            _validate_optimizer_encoded_value(
                item,
                label=f"{label} item {index}",
                counters=counters,
                count_value=count_value,
            )
        return
    if kind == "object_state":
        if (
            set(payload) != {"concrete_type", "kind", "state"}
            or not isinstance(payload["concrete_type"], str)
            or not payload["concrete_type"]
        ):
            raise ValueError(f"{label} optimizer object state fields mismatch")
        _validate_optimizer_encoded_value(
            payload["state"],
            label=f"{label} object state",
            counters=counters,
            count_value=count_value,
        )
        return
    raise ValueError(f"{label} optimizer value kind is invalid: {kind!r}")


def _encoded_string_key(payload: Any) -> str | None:
    if (
        isinstance(payload, dict)
        and payload.get("kind") == "scalar"
        and payload.get("type") == "str"
        and set(payload) == {"kind", "type", "value"}
        and isinstance(payload.get("value"), str)
    ):
        return payload["value"]
    return None


def _encoded_mapping_items_by_string_key(
    payload: Any, *, label: str
) -> dict[str, Any]:
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != "mapping"
        or set(payload) != {"entries", "kind"}
        or not isinstance(payload.get("entries"), list)
    ):
        raise ValueError(f"{label} must be an encoded mapping")
    result: dict[str, Any] = {}
    for index, entry in enumerate(payload["entries"]):
        if not isinstance(entry, dict) or set(entry) != {"key", "value"}:
            raise ValueError(f"{label} mapping entry {index} is invalid")
        key = _encoded_string_key(entry["key"])
        if key is None or key in result:
            raise ValueError(f"{label} keys must be unique strings")
        result[key] = entry["value"]
    return result


def _encoded_scalar_value(
    payload: Any, *, scalar_type: str, label: str
) -> Any:
    if (
        not isinstance(payload, dict)
        or set(payload) != {"kind", "type", "value"}
        or payload.get("kind") != "scalar"
        or payload.get("type") != scalar_type
    ):
        raise ValueError(f"{label} must be an encoded {scalar_type} scalar")
    return payload["value"]


def _validate_deepspeed_zero2_optimizer_probe(
    optimizer_probe: Any,
    *,
    label: str,
    require_optimizer_records: bool,
    expected_fingerprint: str,
) -> None:
    if not isinstance(optimizer_probe, dict) or set(optimizer_probe) != {
        "concrete_type",
        "coverage",
        "fingerprint",
        "inventory",
        "state_dict",
    }:
        raise ValueError(f"{label} optimizer probe fields mismatch")
    if (
        optimizer_probe["concrete_type"]
        != "deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer"
        or optimizer_probe["coverage"]
        != "rank_local_deepspeed_zero2_state_dict"
    ):
        raise RuntimeError(
            f"{label} optimizer proof must cover the DeepSpeed ZeRO-2 wrapper"
        )
    counters = {"tensor_count": 0, "scalar_count": 0, "total_bytes": 0}
    encoded_state_dict = optimizer_probe["state_dict"]
    _validate_optimizer_encoded_value(
        encoded_state_dict,
        label=f"{label} DeepSpeed ZeRO-2 state_dict",
        counters=counters,
    )
    state_items = _encoded_mapping_items_by_string_key(
        encoded_state_dict,
        label=f"{label} DeepSpeed ZeRO-2 state_dict",
    )
    required_keys = {
        "base_optimizer_state",
        "clip_grad",
        "ds_version",
        "dynamic_loss_scale",
        "group_paddings",
        "loss_scaler",
        "overflow",
        "param_slice_mappings",
        "partition_count",
        "single_partition_of_fp32_groups",
        "zero_stage",
    }
    optional_keys = {"universal_checkpoint_info"}
    if (
        not required_keys.issubset(state_items)
        or set(state_items) - required_keys - optional_keys
    ):
        raise ValueError(f"{label} DeepSpeed ZeRO-2 state_dict fields mismatch")
    zero_stage = state_items["zero_stage"]
    if (
        not isinstance(zero_stage, dict)
        or zero_stage.get("kind") != "scalar"
        or zero_stage.get("type") != "int"
        or zero_stage.get("value") != 2
    ):
        raise RuntimeError(f"{label} optimizer state does not prove ZeRO stage 2")
    ds_version = _encoded_scalar_value(
        state_items["ds_version"],
        scalar_type="str",
        label=f"{label} DeepSpeed version",
    )
    if ds_version != ACTION_ONLY_N2_DEEPSPEED_VERSION:
        raise RuntimeError(
            f"{label} DeepSpeed version mismatch: "
            f"expected={ACTION_ONLY_N2_DEEPSPEED_VERSION!r} observed={ds_version!r}"
        )
    for field in ("dynamic_loss_scale", "overflow"):
        _encoded_scalar_value(
            state_items[field],
            scalar_type="bool",
            label=f"{label} DeepSpeed {field}",
        )
    clip_grad_payload = state_items["clip_grad"]
    try:
        if isinstance(clip_grad_payload, dict) and clip_grad_payload.get("type") == "int":
            clip_grad = float(
                _encoded_scalar_value(
                    clip_grad_payload,
                    scalar_type="int",
                    label=f"{label} DeepSpeed clip_grad",
                )
            )
        else:
            clip_grad = float.fromhex(
                _encoded_scalar_value(
                    clip_grad_payload,
                    scalar_type="float_hex",
                    label=f"{label} DeepSpeed clip_grad",
                )
            )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label} DeepSpeed clip_grad must be a finite non-negative number"
        ) from error
    if not math.isfinite(clip_grad) or clip_grad < 0.0:
        raise ValueError(
            f"{label} DeepSpeed clip_grad must be a finite non-negative number"
        )
    loss_scaler = state_items["loss_scaler"]
    if (
        not isinstance(loss_scaler, dict)
        or loss_scaler.get("kind") != "object_state"
        or not isinstance(loss_scaler.get("concrete_type"), str)
        or not loss_scaler["concrete_type"].startswith(
            "deepspeed.runtime.fp16.loss_scaler."
        )
    ):
        raise TypeError(f"{label} DeepSpeed loss_scaler concrete type mismatch")

    base_items = _encoded_mapping_items_by_string_key(
        state_items["base_optimizer_state"],
        label=f"{label} DeepSpeed base optimizer state",
    )
    if set(base_items) != {"param_groups", "state"}:
        raise ValueError(f"{label} DeepSpeed base optimizer fields mismatch")
    base_state = base_items["state"]
    base_groups = base_items["param_groups"]
    if (
        not isinstance(base_state, dict)
        or base_state.get("kind") != "mapping"
        or not isinstance(base_state.get("entries"), list)
        or not isinstance(base_groups, dict)
        or base_groups.get("kind") != "list"
        or not isinstance(base_groups.get("items"), list)
    ):
        raise TypeError(f"{label} DeepSpeed base optimizer state is invalid")
    base_state_parameter_count = len(base_state["entries"])
    if require_optimizer_records and base_state_parameter_count == 0:
        raise RuntimeError(f"{label} optimizer proof has no populated state")

    masters = state_items["single_partition_of_fp32_groups"]
    if (
        not isinstance(masters, dict)
        or masters.get("kind") not in {"list", "tuple"}
        or not isinstance(masters.get("items"), list)
        or not masters["items"]
    ):
        raise RuntimeError(f"{label} optimizer proof has no FP32 master partitions")
    master_numel = 0
    master_bytes = 0
    for index, record in enumerate(masters["items"]):
        _validate_n2_tensor_record(
            record,
            label=f"{label} FP32 master partition {index}",
            expected_fields={"bytes", "dtype", "kind", "numel", "sha256", "shape"},
        )
        if record["kind"] != "tensor" or record["dtype"] != "torch.float32":
            raise RuntimeError(
                f"{label} master partition {index} is not an FP32 tensor"
            )
        if int(record["numel"]) <= 0:
            raise RuntimeError(f"{label} master partition {index} is empty")
        master_numel += int(record["numel"])
        master_bytes += int(record["bytes"])
    master_group_count = len(masters["items"])
    if master_group_count != len(base_groups["items"]):
        raise RuntimeError(
            f"{label} FP32 master partitions do not match base optimizer groups"
        )
    partition_count = state_items["partition_count"]
    if (
        not isinstance(partition_count, dict)
        or partition_count.get("kind") != "list"
        or not isinstance(partition_count.get("items"), list)
        or len(partition_count["items"]) != master_group_count
    ):
        raise ValueError(f"{label} DeepSpeed partition count shape mismatch")
    for index, encoded_count in enumerate(partition_count["items"]):
        count = _encoded_scalar_value(
            encoded_count,
            scalar_type="int",
            label=f"{label} DeepSpeed partition count {index}",
        )
        if count != ACTION_ONLY_N2_1X8_WORLD_SIZE:
            raise RuntimeError(
                f"{label} DeepSpeed partition count {index} does not prove "
                f"world size {ACTION_ONLY_N2_1X8_WORLD_SIZE}"
            )
    group_paddings = state_items["group_paddings"]
    if (
        not isinstance(group_paddings, dict)
        or group_paddings.get("kind") != "list"
        or not isinstance(group_paddings.get("items"), list)
        or len(group_paddings["items"]) != master_group_count
    ):
        raise ValueError(f"{label} DeepSpeed group paddings shape mismatch")
    for index, encoded_padding in enumerate(group_paddings["items"]):
        padding = _encoded_scalar_value(
            encoded_padding,
            scalar_type="int",
            label=f"{label} DeepSpeed group paddings {index}",
        )
        if padding < 0:
            raise ValueError(
                f"{label} DeepSpeed group paddings {index} must be non-negative"
            )
    param_slice_mappings = state_items["param_slice_mappings"]
    if (
        not isinstance(param_slice_mappings, dict)
        or param_slice_mappings.get("kind") != "list"
        or not isinstance(param_slice_mappings.get("items"), list)
        or len(param_slice_mappings["items"]) != master_group_count
        or any(
            not isinstance(mapping, dict) or mapping.get("kind") != "mapping"
            for mapping in param_slice_mappings["items"]
        )
    ):
        raise TypeError(
            f"{label} DeepSpeed parameter slice mappings shape/type mismatch"
        )

    expected_inventory = {
        "base_optimizer_param_group_count": len(base_groups["items"]),
        "base_optimizer_state_parameter_count": base_state_parameter_count,
        "fp32_master_partition_count": len(masters["items"]),
        "fp32_master_partition_numel": master_numel,
        "fp32_master_partition_total_bytes": master_bytes,
        "state_dict_scalar_count": counters["scalar_count"],
        "state_dict_tensor_count": counters["tensor_count"],
        "state_dict_total_bytes": counters["total_bytes"],
    }
    if optimizer_probe["inventory"] != expected_inventory:
        raise RuntimeError(f"{label} optimizer inventory mismatch")
    optimizer_body = {
        "concrete_type": optimizer_probe["concrete_type"],
        "coverage": optimizer_probe["coverage"],
        "inventory": optimizer_probe["inventory"],
        "state_dict": encoded_state_dict,
    }
    if optimizer_probe["fingerprint"] != expected_fingerprint:
        raise RuntimeError(f"{label} optimizer probe fingerprint mismatch")
    if canonical_json_sha256(optimizer_body) != expected_fingerprint:
        raise RuntimeError(
            f"{label} optimizer probe does not reproduce its fingerprint"
        )


def _validate_n2_state_fingerprints(
    payload: Any,
    *,
    label: str,
    expected_global_step: int | None,
    require_optimizer_records: bool,
) -> dict[str, Any]:
    expected_fields = {
        "global_step",
        "model",
        "model_probe",
        "optimizer",
        "optimizer_probe",
        "rng",
        "scheduler",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError(f"{label} state fingerprint fields mismatch")
    global_step = payload["global_step"]
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise ValueError(f"{label} global_step is invalid: {global_step!r}")
    if expected_global_step is not None and global_step != expected_global_step:
        raise RuntimeError(
            f"{label} global_step mismatch: expected={expected_global_step} "
            f"observed={global_step}"
        )
    for field in ("model", "optimizer", "rng", "scheduler"):
        require_sha256(payload[field], label=f"{label} {field} fingerprint")

    model_probe = payload["model_probe"]
    if not isinstance(model_probe, dict) or set(model_probe) != {
        "coverage", "fingerprint", "inventory", "records"
    }:
        raise ValueError(f"{label} model probe fields mismatch")
    records = model_probe["records"]
    if not isinstance(records, list) or not records or model_probe["coverage"] != "full_state_dict":
        raise ValueError(f"{label} model probe is empty or inconsistent")
    previous_name: str | None = None
    model_inventory = {
        "buffer_count": 0,
        "extra_count": 0,
        "inventory_count": len(records),
        "parameter_count": 0,
        "total_bytes": 0,
        "total_numel": 0,
    }
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"{label} model record {index} must be a mapping")
        name = record.get("name")
        kind = record.get("kind")
        if (
            not isinstance(name, str)
            or not name
            or (previous_name is not None and name <= previous_name)
            or kind not in {"parameter", "buffer", "extra"}
        ):
            raise ValueError(f"{label} model record ordering/kind is invalid")
        previous_name = name
        numel, byte_count = _validate_n2_tensor_record(
            record,
            label=f"{label} model record {name}",
            expected_fields={
                "bytes", "dtype", "kind", "name", "numel", "sha256", "shape"
            },
        )
        model_inventory[f"{kind}_count"] += 1
        model_inventory["total_numel"] += numel
        model_inventory["total_bytes"] += byte_count
    if model_probe["inventory"] != model_inventory:
        raise RuntimeError(f"{label} model inventory mismatch")
    model_body = {
        "coverage": model_probe["coverage"],
        "inventory": model_probe["inventory"],
        "records": records,
    }
    expected_model = canonical_json_sha256(model_body)
    if model_probe["fingerprint"] != payload["model"]:
        raise RuntimeError(f"{label} model probe fingerprint mismatch")
    if expected_model != payload["model"]:
        raise RuntimeError(f"{label} model probe does not reproduce its fingerprint")

    _validate_deepspeed_zero2_optimizer_probe(
        payload["optimizer_probe"],
        label=label,
        require_optimizer_records=require_optimizer_records,
        expected_fingerprint=payload["optimizer"],
    )
    return payload


def _validate_n2_next_rng_sample(payload: Any, *, label: str) -> dict[str, Any]:
    expected_fields = {"numpy", "python", "torch_cpu", "torch_cuda"}
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError(f"{label} next RNG sample fields mismatch")
    for field in sorted(expected_fields):
        values = payload[field]
        if (
            not isinstance(values, list)
            or len(values) != 4
            or any(isinstance(value, bool) for value in values)
            or not all(np.isfinite(float(value)) for value in values)
        ):
            raise ValueError(f"{label} invalid next RNG sample for {field}")
    return payload


def _validate_n2_sampler_cursor(payload: Any, *, label: str) -> dict[str, Any]:
    expected_fields = {
        "agent_action_token_budget",
        "batch_in_epoch",
        "epoch",
        "global_batch_offset",
        "global_batches_per_epoch",
        "global_step",
        "gradient_accumulation_steps",
        "microbatches_per_process",
        "num_processes",
        "optimizer_steps_per_epoch",
        "schedule_fingerprint",
        "uses_agent_count_batch_sampler",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError(f"{label} sampler cursor fields mismatch")
    expected_scalars = {
        "batch_in_epoch": 4,
        "epoch": 0,
        "global_batch_offset": 32,
        "global_step": ACTION_ONLY_N2_PAID_GATE_STEP,
        "num_processes": ACTION_ONLY_N2_1X8_WORLD_SIZE,
        "gradient_accumulation_steps": 4,
        "agent_action_token_budget": 128,
        "uses_agent_count_batch_sampler": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": payload.get(key)}
        for key, expected in expected_scalars.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"{label} sampler cursor contract mismatch: {mismatches}")
    for field in (
        "epoch",
        "batch_in_epoch",
        "global_batch_offset",
        "global_batches_per_epoch",
        "microbatches_per_process",
        "optimizer_steps_per_epoch",
    ):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} sampler cursor {field} is invalid: {value!r}")
    if payload["batch_in_epoch"] % payload["gradient_accumulation_steps"]:
        raise RuntimeError(f"{label} sampler cursor is not optimizer-step aligned")
    if (
        payload["global_batch_offset"]
        != payload["batch_in_epoch"] * payload["num_processes"]
    ):
        raise RuntimeError(f"{label} sampler global offset is not derived from cursor")
    if (
        payload["global_batches_per_epoch"]
        != payload["microbatches_per_process"] * payload["num_processes"]
    ):
        raise RuntimeError(f"{label} sampler epoch shape is inconsistent")
    if (
        payload["optimizer_steps_per_epoch"]
        * payload["gradient_accumulation_steps"]
        != payload["microbatches_per_process"]
    ):
        raise RuntimeError(f"{label} sampler optimizer-step shape is inconsistent")
    require_sha256(
        payload["schedule_fingerprint"], label=f"{label} schedule fingerprint"
    )
    return payload


def _read_action_only_n2_terminal_candidate(
    output_root: Path,
    *,
    expected_arguments_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    candidate, candidate_sha256, _ = read_canonical_json(
        output_root / ACTION_ONLY_N2_TERMINAL_CANDIDATE
    )
    if set(candidate) != {
        "arguments",
        "arguments_sha256",
        "run_id",
        "schema_name",
        "schema_version",
        "status",
    }:
        raise ValueError("N=2 terminal candidate fields mismatch")
    arguments = candidate.get("arguments")
    if not isinstance(arguments, dict):
        raise TypeError("N=2 terminal candidate arguments must be a mapping")
    arguments_sha256 = canonical_json_sha256(arguments)
    if (
        candidate.get("schema_name")
        != "fastwam-action-only-n2-terminal-candidate"
        or candidate.get("schema_version") != 1
        or candidate.get("status") != "AWAITING_FRESH_RELOAD"
        or candidate.get("run_id") != arguments.get("run_id")
        or candidate.get("arguments_sha256") != arguments_sha256
        or (
            expected_arguments_sha256 is not None
            and arguments_sha256 != expected_arguments_sha256
        )
    ):
        raise RuntimeError("N=2 terminal candidate identity/contract mismatch")
    return candidate, arguments, candidate_sha256


def _validate_action_only_n2_reload_inventory(
    proof_dir: Path,
    *,
    load_attempt_id: str,
    expected_root_files: set[str],
    expected_selected_files: set[str],
) -> None:
    """Validate only root topology and the selected immutable load attempt."""

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    root_fd = _open_directory_chain(proof_dir)
    attempts_fd: int | None = None
    selected_fd: int | None = None
    try:
        expected_root_names = expected_root_files | {
            ACTION_ONLY_N2_RELOAD_LOAD_ATTEMPTS_DIR
        }
        observed_root_names = set(os.listdir(root_fd))
        if observed_root_names != expected_root_names:
            raise RuntimeError(
                "N=2 reload proof root inventory mismatch: "
                f"missing={sorted(expected_root_names - observed_root_names)} "
                f"unexpected={sorted(observed_root_names - expected_root_names)}"
            )
        for name in expected_root_files:
            info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(
                    "N=2 reload proof root contains an aliased/special artifact: "
                    f"{name}"
                )
        attempts_fd = os.open(
            ACTION_ONLY_N2_RELOAD_LOAD_ATTEMPTS_DIR,
            directory_flags,
            dir_fd=root_fd,
        )
        for observed_attempt_id in os.listdir(attempts_fd):
            require_proof_attempt_id(
                observed_attempt_id,
                label="N=2 reload load-attempt directory name",
            )
            attempt_fd = os.open(
                observed_attempt_id,
                directory_flags,
                dir_fd=attempts_fd,
            )
            if observed_attempt_id == load_attempt_id:
                selected_fd = attempt_fd
            else:
                os.close(attempt_fd)
        if selected_fd is None:
            raise RuntimeError(
                "N=2 selected load-attempt inventory mismatch: "
                f"missing directory={load_attempt_id!r}"
            )
        observed_selected_names = set(os.listdir(selected_fd))
        if observed_selected_names != expected_selected_files:
            raise RuntimeError(
                "N=2 selected load-attempt inventory mismatch: "
                f"missing={sorted(expected_selected_files - observed_selected_names)} "
                f"unexpected={sorted(observed_selected_names - expected_selected_files)}"
            )
        for name in expected_selected_files:
            info = os.stat(name, dir_fd=selected_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(
                    "N=2 selected load attempt contains an aliased/special "
                    f"artifact: {name}"
                )
    finally:
        if selected_fd is not None:
            os.close(selected_fd)
        if attempts_fd is not None:
            os.close(attempts_fd)
        os.close(root_fd)


def validate_action_only_n2_reload_proof(
    output_root: str | Path,
    *,
    run_id: str,
    checkpoint: Mapping[str, Any],
    terminal_arguments_sha256: str,
    load_attempt_id: str | None = None,
    require_committed: bool = True,
) -> dict[str, Any]:
    """Strongly validate the paid N=2 fresh-process 8-rank reload proof."""

    output_root = resolved_unaliased_directory(output_root, label="training output root")
    terminal_arguments_sha256 = require_sha256(
        terminal_arguments_sha256,
        label="N=2 paid terminal arguments SHA-256",
    )
    candidate, candidate_arguments, candidate_sha256 = (
        _read_action_only_n2_terminal_candidate(
            output_root,
            expected_arguments_sha256=terminal_arguments_sha256,
        )
    )
    if (
        candidate["run_id"] != str(run_id)
        or candidate_arguments.get("run_profile") != "paid_gate_1step"
        or candidate_arguments.get("max_steps") != ACTION_ONLY_N2_PAID_GATE_STEP
    ):
        raise RuntimeError("N=2 terminal candidate does not identify this paid gate")
    proof_dir = resolved_unaliased_directory(
        output_root / ACTION_ONLY_N2_RELOAD_PROOF_DIR,
        label="N=2 reload proof directory",
    )
    binding_path = proof_dir / "checkpoint-binding.json"
    binding, binding_sha256, _ = read_canonical_json(binding_path)
    if set(binding) != {
        "checkpoint",
        "global_step",
        "proof_attempt_id",
        "run_id",
        "schema_name",
        "schema_version",
        "terminal_arguments_sha256",
        "terminal_candidate_sha256",
        "world_size",
    }:
        raise ValueError("N=2 reload checkpoint binding fields mismatch")
    proof_attempt_id = require_proof_attempt_id(
        binding.get("proof_attempt_id", ""),
        label="N=2 reload proof attempt id",
    )
    if (
        binding["schema_name"] != "fastwam-action-only-n2-reload-checkpoint-binding"
        or binding["schema_version"] != ACTION_ONLY_N2_RELOAD_PROOF_SCHEMA_VERSION
        or binding["run_id"] != str(run_id)
        or binding["global_step"] != ACTION_ONLY_N2_PAID_GATE_STEP
        or binding["world_size"] != ACTION_ONLY_N2_1X8_WORLD_SIZE
        or binding["checkpoint"] != checkpoint
        or binding["terminal_arguments_sha256"] != terminal_arguments_sha256
        or binding["terminal_candidate_sha256"] != candidate_sha256
    ):
        raise RuntimeError("N=2 reload checkpoint binding does not match the live checkpoint")

    commit_path = proof_dir / ACTION_ONLY_N2_RELOAD_COMMITTED_ATTEMPT
    commitment: dict[str, Any] | None = None
    commitment_sha256: str | None = None
    if require_committed:
        commitment, commitment_sha256, _ = read_canonical_json(commit_path)
        if not isinstance(commitment.get("load_attempt_id"), str):
            raise ValueError("N=2 committed load attempt lacks an attempt id")
        committed_attempt_id = require_proof_attempt_id(
            commitment["load_attempt_id"],
            label="N=2 committed load attempt id",
        )
        if load_attempt_id is not None and load_attempt_id != committed_attempt_id:
            raise RuntimeError(
                "requested N=2 load attempt differs from the immutable commitment"
            )
        load_attempt_id = committed_attempt_id
    else:
        if commit_path.exists() or commit_path.is_symlink():
            raise RuntimeError("N=2 reload proof already has a committed load attempt")
        if load_attempt_id is None:
            raise ValueError("N=2 precommit validation requires a load attempt id")
    load_attempt_id = require_proof_attempt_id(
        load_attempt_id or "", label="N=2 reload load attempt id"
    )

    expected_root_names = {"checkpoint-binding.json"}
    expected_root_names.update(
        f"save-rank-{rank:05d}.json"
        for rank in range(ACTION_ONLY_N2_1X8_WORLD_SIZE)
    )
    if require_committed:
        expected_root_names.add(ACTION_ONLY_N2_RELOAD_COMMITTED_ATTEMPT)
    expected_selected_names = {
        f"load-rank-{rank:05d}.json"
        for rank in range(ACTION_ONLY_N2_1X8_WORLD_SIZE)
    }
    _validate_action_only_n2_reload_inventory(
        proof_dir,
        load_attempt_id=load_attempt_id,
        expected_root_files=expected_root_names,
        expected_selected_files=expected_selected_names,
    )

    phase_payloads: dict[str, list[dict[str, Any]]] = {"save": [], "load": []}
    proof_records: dict[str, list[dict[str, str]]] = {"save": [], "load": []}
    base_fields = {
        "checkpoint",
        "checkpoint_binding_sha256",
        "fingerprints",
        "global_step",
        "next_rng_sample",
        "phase",
        "proof_attempt_id",
        "process_nonce",
        "process_pid",
        "process_start_ticks",
        "rank",
        "run_id",
        "sampler_cursor",
        "schema_name",
        "schema_version",
        "terminal_arguments_sha256",
        "terminal_candidate_sha256",
        "world_size",
    }
    for phase in ("save", "load"):
        expected_fields = set(base_fields)
        if phase == "load":
            expected_fields.update(
                {"checks", "load_attempt_id", "pre_load_fingerprints"}
            )
        for rank in range(ACTION_ONLY_N2_1X8_WORLD_SIZE):
            if phase == "save":
                relative = PurePosixPath(ACTION_ONLY_N2_RELOAD_PROOF_DIR) / (
                    f"save-rank-{rank:05d}.json"
                )
            else:
                relative = (
                    PurePosixPath(ACTION_ONLY_N2_RELOAD_PROOF_DIR)
                    / ACTION_ONLY_N2_RELOAD_LOAD_ATTEMPTS_DIR
                    / load_attempt_id
                    / f"load-rank-{rank:05d}.json"
                )
            payload, digest, _ = read_canonical_json(output_root / relative)
            if set(payload) != expected_fields:
                raise ValueError(f"N=2 {phase} proof fields mismatch at rank {rank}")
            expected_phase = (
                "save_after_sealed_checkpoint"
                if phase == "save"
                else "load_fresh_process"
            )
            expected_schema = f"fastwam-action-only-n2-reload-{phase}-proof"
            if (
                payload["schema_name"] != expected_schema
                or payload["schema_version"]
                != ACTION_ONLY_N2_RELOAD_PROOF_SCHEMA_VERSION
                or payload["phase"] != expected_phase
                or payload["rank"] != rank
                or payload["world_size"] != ACTION_ONLY_N2_1X8_WORLD_SIZE
                or payload["global_step"] != ACTION_ONLY_N2_PAID_GATE_STEP
                or payload["proof_attempt_id"] != proof_attempt_id
                or (
                    phase == "load"
                    and payload["load_attempt_id"] != load_attempt_id
                )
                or payload["run_id"] != str(run_id)
                or payload["checkpoint_binding_sha256"] != binding_sha256
                or payload["checkpoint"] != checkpoint
                or payload["terminal_arguments_sha256"]
                != terminal_arguments_sha256
                or payload["terminal_candidate_sha256"] != candidate_sha256
            ):
                raise RuntimeError(f"N=2 {phase} proof identity mismatch at rank {rank}")
            nonce = payload["process_nonce"]
            if (
                not isinstance(nonce, str)
                or len(nonce) != 32
                or any(character not in SHA256_HEX for character in nonce)
            ):
                raise ValueError(f"N=2 {phase} proof process nonce is invalid at rank {rank}")
            for field in ("process_pid", "process_start_ticks"):
                value = payload[field]
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(
                        f"N=2 {phase} proof {field} is invalid at rank {rank}"
                    )
            _validate_n2_state_fingerprints(
                payload["fingerprints"],
                label=f"N=2 {phase} rank {rank}",
                expected_global_step=ACTION_ONLY_N2_PAID_GATE_STEP,
                require_optimizer_records=True,
            )
            _validate_n2_next_rng_sample(
                payload["next_rng_sample"], label=f"N=2 {phase} rank {rank}"
            )
            _validate_n2_sampler_cursor(
                payload["sampler_cursor"], label=f"N=2 {phase} rank {rank}"
            )
            phase_payloads[phase].append(payload)
            proof_records[phase].append(
                {"path": relative.as_posix(), "sha256": digest}
            )

    save_nonces = {payload["process_nonce"] for payload in phase_payloads["save"]}
    load_nonces = {payload["process_nonce"] for payload in phase_payloads["load"]}
    save_processes = {
        (payload["process_pid"], payload["process_start_ticks"])
        for payload in phase_payloads["save"]
    }
    load_processes = {
        (payload["process_pid"], payload["process_start_ticks"])
        for payload in phase_payloads["load"]
    }
    if (
        len(save_nonces) != ACTION_ONLY_N2_1X8_WORLD_SIZE
        or len(load_nonces) != ACTION_ONLY_N2_1X8_WORLD_SIZE
        or save_nonces & load_nonces
        or len(save_processes) != ACTION_ONLY_N2_1X8_WORLD_SIZE
        or len(load_processes) != ACTION_ONLY_N2_1X8_WORLD_SIZE
        or save_processes & load_processes
    ):
        raise RuntimeError("N=2 reload proof does not establish two fresh 8-rank worlds")

    check_fields = {
        "checkpoint_binding",
        "fresh_process",
        "global_step",
        "model",
        "next_rng_sample",
        "optimizer",
        "pre_load_was_distinct",
        "rng",
        "sampler_cursor",
        "scheduler",
        "terminal_candidate",
    }
    for rank, (saved, loaded) in enumerate(
        zip(phase_payloads["save"], phase_payloads["load"], strict=True)
    ):
        pre_load = _validate_n2_state_fingerprints(
            loaded["pre_load_fingerprints"],
            label=f"N=2 pre-load rank {rank}",
            expected_global_step=0,
            require_optimizer_records=False,
        )
        expected_checks = {
            "checkpoint_binding": loaded["checkpoint"] == saved["checkpoint"],
            "fresh_process": (
                loaded["process_nonce"] != saved["process_nonce"]
                and (
                    loaded["process_pid"], loaded["process_start_ticks"]
                )
                != (saved["process_pid"], saved["process_start_ticks"])
            ),
            "global_step": loaded["global_step"] == saved["global_step"],
            "model": loaded["fingerprints"]["model"] == saved["fingerprints"]["model"],
            "next_rng_sample": loaded["next_rng_sample"] == saved["next_rng_sample"],
            "optimizer": loaded["fingerprints"]["optimizer"]
            == saved["fingerprints"]["optimizer"],
            "pre_load_was_distinct": any(
                pre_load[key] != saved["fingerprints"][key]
                for key in ("model", "optimizer")
            ),
            "rng": loaded["fingerprints"]["rng"] == saved["fingerprints"]["rng"],
            "sampler_cursor": loaded["sampler_cursor"] == saved["sampler_cursor"],
            "scheduler": loaded["fingerprints"]["scheduler"]
            == saved["fingerprints"]["scheduler"],
            "terminal_candidate": (
                loaded["terminal_candidate_sha256"]
                == saved["terminal_candidate_sha256"]
                == candidate_sha256
                and loaded["terminal_arguments_sha256"]
                == saved["terminal_arguments_sha256"]
                == terminal_arguments_sha256
            ),
        }
        if loaded["fingerprints"] != saved["fingerprints"]:
            expected_checks["model"] = False
        if (
            not isinstance(loaded["checks"], dict)
            or set(loaded["checks"]) != check_fields
            or loaded["checks"] != expected_checks
            or not all(expected_checks.values())
        ):
            raise RuntimeError(
                f"N=2 fresh reload semantic mismatch on rank {rank}: "
                f"{expected_checks}"
            )

    trainer_state_relative = PurePosixPath(checkpoint["state"]["root"]) / (
        "trainer_state.json"
    )
    trainer_state, trainer_state_sha256, _ = read_canonical_json(
        output_root / trainer_state_relative
    )
    if trainer_state_sha256 != checkpoint["state"]["trainer_state_sha256"]:
        raise RuntimeError("N=2 paid trainer state changed after checkpoint sealing")
    trainer_state_contract = {
        "batch_in_epoch": 4,
        "epoch": 0,
        "evaluation_records": candidate_arguments.get("evaluation_records"),
        "global_step": ACTION_ONLY_N2_PAID_GATE_STEP,
        "last_step_metrics": candidate_arguments.get("last_step_metrics"),
    }
    trainer_state_mismatches = {
        key: {"expected": expected, "observed": trainer_state.get(key)}
        for key, expected in trainer_state_contract.items()
        if trainer_state.get(key) != expected
    }
    if trainer_state_mismatches:
        raise RuntimeError(
            "N=2 paid terminal candidate does not match sealed trainer state: "
            f"{trainer_state_mismatches}"
        )

    rank_state_inventory = [
        {
            "model_sha256": saved["fingerprints"]["model"],
            "optimizer_sha256": saved["fingerprints"]["optimizer"],
            "rank": rank,
            "state_fingerprints_sha256": canonical_json_sha256(
                saved["fingerprints"]
            ),
        }
        for rank, saved in enumerate(phase_payloads["save"])
    ]
    rank_state_aggregate_sha256 = canonical_json_sha256(
        {"rank_state_inventory": rank_state_inventory}
    )
    reload_proof = {
        "checkpoint_binding": {
            "path": (
                f"{ACTION_ONLY_N2_RELOAD_PROOF_DIR}/checkpoint-binding.json"
            ),
            "sha256": binding_sha256,
        },
        "fresh_process_verified": True,
        "global_step": ACTION_ONLY_N2_PAID_GATE_STEP,
        "load_attempt_id": load_attempt_id,
        "load_proofs": proof_records["load"],
        "proof_attempt_id": proof_attempt_id,
        "rank_state_aggregate_sha256": rank_state_aggregate_sha256,
        "rank_state_inventory": rank_state_inventory,
        "run_id": str(run_id),
        "save_proofs": proof_records["save"],
        "schema_name": "fastwam-action-only-n2-reload-proof-summary",
        "schema_version": ACTION_ONLY_N2_RELOAD_PROOF_SCHEMA_VERSION,
        "terminal_candidate": {
            "arguments_sha256": terminal_arguments_sha256,
            "path": ACTION_ONLY_N2_TERMINAL_CANDIDATE,
            "sha256": candidate_sha256,
        },
        "verified_fields": sorted(check_fields),
        "verified_ranks": list(range(ACTION_ONLY_N2_1X8_WORLD_SIZE)),
        "world_size": ACTION_ONLY_N2_1X8_WORLD_SIZE,
    }
    if require_committed:
        assert commitment is not None and commitment_sha256 is not None
        expected_commitment = _action_only_n2_reload_commitment(reload_proof)
        if commitment != expected_commitment:
            raise RuntimeError(
                "N=2 committed load attempt does not match validated proof evidence"
            )
        reload_proof["committed_load_attempt"] = {
            "path": (
                f"{ACTION_ONLY_N2_RELOAD_PROOF_DIR}/"
                f"{ACTION_ONLY_N2_RELOAD_COMMITTED_ATTEMPT}"
            ),
            "sha256": commitment_sha256,
        }
    return reload_proof


def _action_only_n2_reload_commitment(
    reload_proof: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_binding": reload_proof["checkpoint_binding"],
        "global_step": reload_proof["global_step"],
        "load_attempt_id": reload_proof["load_attempt_id"],
        "load_proofs": reload_proof["load_proofs"],
        "proof_attempt_id": reload_proof["proof_attempt_id"],
        "rank_state_aggregate_sha256": reload_proof[
            "rank_state_aggregate_sha256"
        ],
        "run_id": reload_proof["run_id"],
        "save_proofs": reload_proof["save_proofs"],
        "schema_name": "fastwam-action-only-n2-reload-load-attempt-commit",
        "schema_version": ACTION_ONLY_N2_RELOAD_PROOF_SCHEMA_VERSION,
        "status": "COMMITTED",
        "terminal_candidate": reload_proof["terminal_candidate"],
        "world_size": reload_proof["world_size"],
    }


@_with_training_terminal_lock
def publish_action_only_n2_reload_attempt_commit(
    output_root: str | Path,
    *,
    run_id: str,
    checkpoint: Mapping[str, Any],
    terminal_arguments_sha256: str,
    load_attempt_id: str,
) -> dict[str, Any]:
    """Atomically validate and commit one complete live reload attempt."""

    output_root = resolved_unaliased_directory(
        output_root, label="training output root"
    )
    commit_path = (
        output_root
        / ACTION_ONLY_N2_RELOAD_PROOF_DIR
        / ACTION_ONLY_N2_RELOAD_COMMITTED_ATTEMPT
    )
    if commit_path.exists() or commit_path.is_symlink():
        raise FileExistsError(f"N=2 reload proof is already committed: {commit_path}")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("N=2 reload proof commit requires a non-empty run id")
    terminal_arguments_sha256 = require_sha256(
        terminal_arguments_sha256,
        label="N=2 paid terminal arguments SHA-256",
    )
    load_attempt_id = require_proof_attempt_id(
        load_attempt_id, label="N=2 reload load attempt id"
    )
    reload_proof = validate_action_only_n2_reload_proof(
        output_root,
        run_id=run_id,
        checkpoint=checkpoint,
        terminal_arguments_sha256=terminal_arguments_sha256,
        load_attempt_id=load_attempt_id,
        require_committed=False,
    )
    commitment = _action_only_n2_reload_commitment(reload_proof)
    publish_exclusive_json(commit_path, commitment)
    digest, _ = sha256_regular_file(commit_path)
    validate_action_only_n2_reload_proof(
        output_root,
        run_id=run_id,
        checkpoint=checkpoint,
        terminal_arguments_sha256=terminal_arguments_sha256,
        load_attempt_id=load_attempt_id,
        require_committed=True,
    )
    return {"path": commit_path.relative_to(output_root).as_posix(), "sha256": digest}


def _publish_sha256sums(
    output_root: Path,
    relative_paths: Iterable[str],
    *,
    expected_sha256s: Mapping[str, str] | None = None,
) -> tuple[str, list[str]]:
    paths = sorted({safe_relative_path(value) for value in relative_paths}, key=lambda item: os.fsencode(item.as_posix()))
    expected_sha256s = dict(expected_sha256s or {})
    records = []
    lines = []
    for relative in paths:
        digest, size = sha256_regular_file(output_root / relative)
        expected = expected_sha256s.get(relative.as_posix())
        if expected is not None and digest != expected:
            raise RuntimeError(
                "artifact changed after terminal validation: "
                f"path={relative} expected={expected} actual={digest}"
            )
        records.append(relative.as_posix())
        lines.append(f"{digest}  {relative.as_posix()}\n")
    payload = "".join(lines).encode("utf-8")
    publish_exclusive_bytes(output_root / "SHA256SUMS", payload)
    return hashlib.sha256(payload).hexdigest(), records


def normalize_formal_evaluation_records(
    evaluation_records: Sequence[Mapping[str, Any]],
    *,
    expected_steps: Sequence[int],
    training_mode: str,
    expected_offline_samples: int = 12,
    expected_offline_agent_counts: Sequence[int] = (2, 3, 4),
    expected_offline_tasks: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Validate the exact offline-eval evidence required by one treatment."""

    training_mode = str(training_mode).strip().lower()
    if training_mode not in {"joint", "action_only_cache"}:
        raise ValueError(
            f"unsupported formal training mode for eval contract: {training_mode!r}"
        )
    expected_steps = [int(step) for step in expected_steps]
    expected_offline_samples = int(expected_offline_samples)
    if expected_offline_samples <= 0:
        raise ValueError("formal offline-eval sample count must be positive")
    expected_offline_agent_counts = [
        int(value) for value in expected_offline_agent_counts
    ]
    if (
        not expected_offline_agent_counts
        or expected_offline_agent_counts
        != sorted(set(expected_offline_agent_counts))
    ):
        raise ValueError(
            "formal offline-eval agent counts must be non-empty, unique, and sorted: "
            f"{expected_offline_agent_counts}"
        )
    normalized_expected_tasks = None
    if expected_offline_tasks is not None:
        normalized_expected_tasks = [str(value).strip() for value in expected_offline_tasks]
        if (
            not normalized_expected_tasks
            or any(not value for value in normalized_expected_tasks)
            or normalized_expected_tasks != sorted(set(normalized_expected_tasks))
        ):
            raise ValueError(
                "formal offline-eval tasks must be non-empty, unique, and sorted: "
                f"{normalized_expected_tasks}"
            )
    base_fields = {
        "evaluation_kind",
        "offline_agent_counts",
        "offline_samples",
        "offline_tasks",
        "step",
        "val_loss",
        "val_loss_action",
    }
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(evaluation_records):
        if not isinstance(record, Mapping):
            raise TypeError(f"formal evaluation record {index} must be a mapping")
        fields = set(record)
        allowed_fields = base_fields | {"val_loss_video"}
        if fields != base_fields and fields != allowed_fields:
            raise RuntimeError(
                f"formal evaluation record field mismatch at index {index}: {sorted(fields)}"
            )
        tasks = record.get("offline_tasks")
        counts = record.get("offline_agent_counts")
        if (
            record.get("evaluation_kind") != "multi_robot_offline_loss"
            or record.get("offline_samples") != expected_offline_samples
            or not isinstance(counts, list)
            or counts != expected_offline_agent_counts
            or not isinstance(tasks, list)
            or not tasks
            or tasks != sorted(set(str(value) for value in tasks))
            or (
                normalized_expected_tasks is not None
                and tasks != normalized_expected_tasks
            )
        ):
            raise RuntimeError(f"formal offline-eval contract mismatch: {record}")
        for metric_name in ("val_loss", "val_loss_action"):
            if not np.isfinite(float(record[metric_name])):
                raise RuntimeError(
                    f"formal offline evaluation lacks finite {metric_name}: {record}"
                )
        video_value = record.get("val_loss_video")
        if training_mode == "joint":
            if video_value is None or not np.isfinite(float(video_value)):
                raise RuntimeError(
                    f"joint VideoGen evaluation lacks finite val_loss_video: {record}"
                )
        elif video_value is not None:
            raise RuntimeError(
                "action-only evaluation must not claim a video-loss metric: "
                f"{record}"
            )
        normalized_record = dict(record)
        normalized_record["val_loss_video"] = (
            float(video_value) if training_mode == "joint" else None
        )
        normalized.append(normalized_record)
    observed_steps = [int(record["step"]) for record in normalized]
    if observed_steps != expected_steps:
        raise RuntimeError(
            "formal evaluation steps mismatch: "
            f"expected={expected_steps} observed={observed_steps}"
        )
    return normalized


@_with_training_terminal_lock
def publish_training_terminal_seal(
    output_root: str | Path,
    *,
    run_id: str,
    code_commit: str,
    config_relative_path: str,
    config_sha256: str,
    max_steps: int,
    expected_checkpoint_steps: Sequence[int],
    expected_evaluation_steps: Sequence[int],
    world_size: int,
    last_step_metrics: Mapping[str, Any],
    evaluation_records: Sequence[Mapping[str, Any]],
    training_mode: str,
    dataset_contract_sha256: str,
    authorization_gate_complete_sha256: str,
    rehash_weights: bool = True,
    training_terminal_contract: str | None = None,
    formal_n4_fullmodel_gate: bool = False,
    checkpoint_state_kind: str = "full",
    trainable_scope: str = "dit",
    dataset_contract: Mapping[str, Any] | None = None,
    task_scope_receipt_relative_path: str = "",
    effective_patched_tree: str = "",
    request_sha256: str = "",
    init_checkpoint_sha256: str = "",
    offline_eval_num_samples: int | None = None,
    run_profile: str = "",
) -> dict[str, Any]:
    output_root = resolved_unaliased_directory(output_root, label="training output root")
    failure_marker = output_root / "TRAINING.FAILED.json"
    if failure_marker.exists() or failure_marker.is_symlink():
        raise RuntimeError(
            "refusing terminal training success after a failure marker exists: "
            f"{failure_marker}"
        )
    for name in ("training-summary.json", "SHA256SUMS", "TRAINING.COMPLETE"):
        target = output_root / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to replace terminal training artifact: {target}")
    contract_name = "" if training_terminal_contract is None else str(
        training_terminal_contract
    ).strip()
    if contract_name not in {"", ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT}:
        raise ValueError(f"unsupported formal training terminal contract: {contract_name!r}")
    is_action_only_n2 = contract_name == ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT
    code_commit = require_git_object_id(
        code_commit, label="formal training terminal code commit"
    )
    config_sha256 = require_sha256(config_sha256, label="resolved config SHA-256")
    dataset_contract_sha256 = require_sha256(
        dataset_contract_sha256, label="dataset contract SHA-256"
    )
    checkpoint_state_kind = str(checkpoint_state_kind).strip().lower()
    trainable_scope = str(trainable_scope).strip().lower()
    if is_action_only_n2:
        if rehash_weights is not True:
            raise ValueError(
                "N=2 action-only terminal contracts require "
                "terminal_rehash_weights=true"
            )
        if authorization_gate_complete_sha256:
            raise ValueError(
                "N=2 action-only terminal runs must not claim the N=4 full-model gate"
            )
        if dataset_contract is None:
            raise TypeError("N=2 action-only terminal seal requires the dataset contract")
        actual_dataset_contract_sha256 = canonical_json_sha256(dataset_contract)
        if actual_dataset_contract_sha256 != dataset_contract_sha256:
            raise RuntimeError(
                "N=2 terminal dataset contract SHA-256 mismatch: "
                f"expected={dataset_contract_sha256} "
                f"actual={actual_dataset_contract_sha256}"
            )
        n2_evidence = validate_action_only_n2_terminal_reservation(
            output_root,
            run_id=run_id,
            base_code_commit=code_commit,
            effective_patched_tree=effective_patched_tree,
            request_sha256=request_sha256,
            init_checkpoint_sha256=init_checkpoint_sha256,
            world_size=world_size,
            formal_n4_fullmodel_gate=formal_n4_fullmodel_gate,
            checkpoint_state_kind=checkpoint_state_kind,
            trainable_scope=trainable_scope,
            training_mode=training_mode,
            dataset_contract=dataset_contract,
            task_scope_receipt_relative_path=task_scope_receipt_relative_path,
            run_profile=run_profile,
        )
        authorization_gate_complete_sha256 = ""
        reservation_descriptor = n2_evidence["reservation"]
    else:
        if int(world_size) != N4_GATE_WORLD_SIZE:
            raise ValueError(
                "legacy formal training terminal seal requires world_size=32, "
                f"got {world_size}"
            )
        authorization_gate_complete_sha256 = require_sha256(
            authorization_gate_complete_sha256,
            label="N=4 full-model authorization gate COMPLETE SHA-256",
        )
        reservation, reservation_sha256, _ = read_canonical_json(
            output_root / ".RUN_RESERVED"
        )
        reservation_identity_sha256 = _validate_reservation_identity(
            reservation, label="formal .RUN_RESERVED"
        )
        reservation_contract = {
            "code_commit": code_commit,
            "global_world_size": int(world_size),
            "n4_fullmodel_gate_complete_sha256": (
                authorization_gate_complete_sha256
            ),
            "run_id": str(run_id),
            "schema_version": 1,
        }
        reservation_mismatches = {
            key: {"expected": expected, "observed": reservation.get(key)}
            for key, expected in reservation_contract.items()
            if reservation.get(key) != expected
        }
        if reservation_mismatches:
            raise RuntimeError(
                "formal .RUN_RESERVED does not authorize the terminal run: "
                f"{reservation_mismatches}"
            )
        reservation_descriptor = {
            "identity_sha256": reservation_identity_sha256,
            "path": ".RUN_RESERVED",
            "sha256": reservation_sha256,
        }
    config_relative = safe_relative_path(config_relative_path)
    actual_config_sha256, _ = sha256_regular_file(output_root / config_relative)
    if actual_config_sha256 != config_sha256:
        raise RuntimeError(
            f"resolved config SHA-256 mismatch: expected={config_sha256} actual={actual_config_sha256}"
        )
    steps = sorted({int(step) for step in expected_checkpoint_steps})
    if not steps or steps[-1] != int(max_steps):
        raise ValueError(
            f"terminal checkpoint steps must be non-empty and end at max_steps={max_steps}: {steps}"
        )
    checkpoints = [
        checkpoint_seal_descriptor(
            output_root,
            step=step,
            rehash_weights=rehash_weights,
            expected_checkpoint_state_kind=checkpoint_state_kind,
        )
        for step in steps
    ]
    required_last_metric_fields = {
        "grad_norm",
        "learning_rate",
        "loss",
        "loss_components",
        "step",
    }
    if set(last_step_metrics) != required_last_metric_fields:
        raise ValueError(
            "terminal last-step metric fields mismatch: "
            f"expected={sorted(required_last_metric_fields)} "
            f"observed={sorted(last_step_metrics)}"
        )
    if last_step_metrics.get("step") != int(max_steps):
        raise RuntimeError(
            f"terminal metrics must describe max_steps={max_steps}: {last_step_metrics}"
        )
    finite_terminal_values = {
        "loss": last_step_metrics.get("loss"),
        "grad_norm": last_step_metrics.get("grad_norm"),
        "learning_rate": last_step_metrics.get("learning_rate"),
        **{
            f"loss_components.{key}": value
            for key, value in dict(last_step_metrics.get("loss_components", {})).items()
        },
    }
    if not finite_terminal_values or not all(
        np.isfinite(float(value)) for value in finite_terminal_values.values()
    ):
        raise RuntimeError(
            f"terminal metrics contain non-finite values: {finite_terminal_values}"
        )
    expected_eval_steps = [int(step) for step in expected_evaluation_steps]
    if expected_eval_steps != sorted(set(expected_eval_steps)):
        raise ValueError(
            f"expected evaluation steps must be unique and sorted: {expected_eval_steps}"
        )
    if (
        not is_action_only_n2
        and int(max_steps) == 5000
        and expected_eval_steps != [1000, 2000, 3000, 4000, 5000]
    ):
        raise ValueError(
            "the formal 5000-step mixed N=2/3/4 run requires offline eval at "
            "steps [1000,2000,3000,4000,5000]"
        )
    reload_proof: dict[str, Any] | None = None
    if is_action_only_n2 and run_profile == "paid_gate_1step":
        if int(max_steps) != 1 or steps != [1] or expected_eval_steps:
            raise ValueError(
                "paid_gate_1step requires max_steps=1, sealed checkpoint steps=[1], "
                f"and no evaluation steps; got max_steps={max_steps}, "
                f"checkpoint_steps={steps}, evaluation_steps={expected_eval_steps}"
            )
        if evaluation_records:
            raise RuntimeError(
                "paid_gate_1step eval_every=0 requires evaluation_records=[]"
            )
        if offline_eval_num_samples not in (None, 0):
            raise ValueError(
                "paid_gate_1step requires offline_eval_num_samples=0"
            )
        if checkpoints[0]["weights"]["rehash_verified"] is not True:
            raise RuntimeError("N=2 paid-gate weights were not strongly rehashed")
        paid_terminal_arguments = {
            "authorization_gate_complete_sha256": "",
            "checkpoint_state_kind": checkpoint_state_kind,
            "code_commit": code_commit,
            "config_relative_path": config_relative_path,
            "config_sha256": config_sha256,
            "dataset_contract": dict(dataset_contract),
            "dataset_contract_sha256": dataset_contract_sha256,
            "effective_patched_tree": effective_patched_tree,
            "evaluation_records": [dict(record) for record in evaluation_records],
            "expected_checkpoint_steps": [int(step) for step in expected_checkpoint_steps],
            "expected_evaluation_steps": [
                int(step) for step in expected_evaluation_steps
            ],
            "formal_n4_fullmodel_gate": bool(formal_n4_fullmodel_gate),
            "init_checkpoint_sha256": init_checkpoint_sha256,
            "last_step_metrics": dict(last_step_metrics),
            "max_steps": int(max_steps),
            "offline_eval_num_samples": int(offline_eval_num_samples or 0),
            "rehash_weights": True,
            "request_sha256": request_sha256,
            "run_id": str(run_id),
            "run_profile": run_profile,
            "task_scope_receipt_relative_path": task_scope_receipt_relative_path,
            "trainable_scope": trainable_scope,
            "training_mode": training_mode,
            "training_terminal_contract": contract_name,
            "world_size": int(world_size),
        }
        terminal_arguments_sha256 = canonical_json_sha256(paid_terminal_arguments)
        _, candidate_arguments, _ = _read_action_only_n2_terminal_candidate(
            output_root,
            expected_arguments_sha256=terminal_arguments_sha256,
        )
        if candidate_arguments != paid_terminal_arguments:
            raise RuntimeError(
                "N=2 paid terminal invocation differs from the staged candidate"
            )
        reload_proof = validate_action_only_n2_reload_proof(
            output_root,
            run_id=run_id,
            checkpoint=checkpoints[0],
            terminal_arguments_sha256=terminal_arguments_sha256,
        )
        normalized_evaluations: list[dict[str, Any]] = []
    else:
        expected_offline_samples = 12
        expected_offline_counts: Sequence[int] = (2, 3, 4)
        expected_offline_tasks: Sequence[str] | None = None
        if is_action_only_n2:
            if run_profile != "formal_1k":
                raise ValueError(
                    f"unsupported N=2 terminal run_profile: {run_profile!r}"
                )
            formal_eval_steps = [500, 1000]
            if int(max_steps) != 1000 or expected_eval_steps != formal_eval_steps:
                raise ValueError(
                    "N=2 action-only formal held-out eval requires max_steps=1000 "
                    f"and steps={formal_eval_steps}, got max_steps={max_steps} "
                    f"steps={expected_eval_steps}"
                )
            if steps != [500, 1000]:
                raise ValueError(
                    "N=2 action-only formal run requires sealed checkpoints at "
                    f"[500, 1000], got {steps}"
                )
            if int(offline_eval_num_samples or 0) != 32:
                raise ValueError(
                    "N=2 action-only formal held-out eval requires "
                    "offline_eval_num_samples=32"
                )
            expected_offline_samples = 32
            expected_offline_counts = n2_evidence["task_scope"][
                "required_agent_counts"
            ]
            expected_offline_tasks = n2_evidence["task_scope"]["required_tasks"]
        normalized_evaluations = normalize_formal_evaluation_records(
            evaluation_records,
            expected_steps=expected_eval_steps,
            training_mode=training_mode,
            expected_offline_samples=expected_offline_samples,
            expected_offline_agent_counts=expected_offline_counts,
            expected_offline_tasks=expected_offline_tasks,
        )
    observed_eval_steps = [int(record["step"]) for record in normalized_evaluations]
    if observed_eval_steps != expected_eval_steps:
        raise RuntimeError(
            "terminal evaluation steps mismatch: "
            f"expected={expected_eval_steps} observed={observed_eval_steps}"
        )
    summary: dict[str, Any] = {
        "authorization_gate_complete_sha256": authorization_gate_complete_sha256,
        "checkpoints": checkpoints,
        "code_commit": code_commit,
        "config": {"path": config_relative.as_posix(), "sha256": config_sha256},
        "dataset_contract_sha256": dataset_contract_sha256,
        "evaluation_records": normalized_evaluations,
        "last_step_metrics": dict(last_step_metrics),
        "max_steps": int(max_steps),
        "published_at": datetime.now(timezone.utc).isoformat(),
        "reservation": reservation_descriptor,
        "run_id": str(run_id),
        "schema_name": "fastwam-training-summary",
        "schema_version": 1,
        "status": "PASS",
        "treatment": {
            "training_mode": training_mode,
            "video_gen": training_mode == "joint",
        },
        "world_size": int(world_size),
    }
    if is_action_only_n2:
        summary.update(
            {
                "authorization_gate_complete_sha256": None,
                "base_code_commit": code_commit,
                "checkpoint_state_kind": checkpoint_state_kind,
                "effective_patched_tree": require_git_object_id(
                    effective_patched_tree,
                    label="N=2 terminal effective patched tree",
                ),
                "formal_n4_fullmodel_gate": False,
                "init_checkpoint_sha256": require_sha256(
                    init_checkpoint_sha256,
                    label="N=2 terminal initialization checkpoint SHA-256",
                ),
                "offline_eval_num_samples": int(offline_eval_num_samples or 0),
                "request_sha256": require_sha256(
                    request_sha256, label="N=2 terminal request SHA-256"
                ),
                "run_profile": run_profile,
                "schema_version": 2,
                "task_scope": n2_evidence["task_scope"],
                "training_terminal_contract": contract_name,
                "training_terminal_contract_version": (
                    ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT_VERSION
                ),
                "treatment": {
                    "checkpoint_state_kind": checkpoint_state_kind,
                    "formal_n4_fullmodel_gate": False,
                    "trainable_scope": trainable_scope,
                    "training_mode": training_mode,
                    "video_gen": False,
                },
            }
        )
        if reload_proof is not None:
            summary["reload_proof"] = reload_proof
    if reload_proof is not None:
        _read_action_only_n2_terminal_candidate(
            output_root,
            expected_arguments_sha256=reload_proof["terminal_candidate"][
                "arguments_sha256"
            ],
        )
    summary_path = output_root / "training-summary.json"
    publish_exclusive_json(summary_path, summary)
    bound_paths = [
        summary_path.relative_to(output_root).as_posix(),
        config_relative.as_posix(),
        ".RUN_RESERVED",
    ]
    if is_action_only_n2:
        bound_paths.append(n2_evidence["task_scope"]["path"])
        if reload_proof is not None:
            bound_paths.append(ACTION_ONLY_N2_TERMINAL_CANDIDATE)
            bound_paths.append(reload_proof["checkpoint_binding"]["path"])
            bound_paths.append(reload_proof["committed_load_attempt"]["path"])
            for phase in ("save_proofs", "load_proofs"):
                bound_paths.extend(
                    record["path"] for record in reload_proof[phase]
                )
    for checkpoint in checkpoints:
        bound_paths.extend(
            (
                checkpoint["weights"]["manifest"],
                checkpoint["weights"]["complete"],
                checkpoint["state"]["manifest"],
                f"{checkpoint['state']['root']}/trainer_state.json",
            )
        )
    expected_sha256s = {}
    if reload_proof is not None:
        expected_sha256s[ACTION_ONLY_N2_TERMINAL_CANDIDATE] = reload_proof[
            "terminal_candidate"
        ]["sha256"]
        expected_sha256s[reload_proof["checkpoint_binding"]["path"]] = (
            reload_proof["checkpoint_binding"]["sha256"]
        )
        expected_sha256s[reload_proof["committed_load_attempt"]["path"]] = (
            reload_proof["committed_load_attempt"]["sha256"]
        )
        for phase in ("save_proofs", "load_proofs"):
            expected_sha256s.update(
                {record["path"]: record["sha256"] for record in reload_proof[phase]}
            )
    sha256sums_sha256, bound_paths = _publish_sha256sums(
        output_root,
        bound_paths,
        expected_sha256s=expected_sha256s,
    )
    summary_sha256, _ = sha256_regular_file(summary_path)
    complete: dict[str, Any] = {
        "bound_paths": bound_paths,
        "max_steps": int(max_steps),
        "run_id": str(run_id),
        "schema_name": "fastwam-training-complete",
        "schema_version": 1,
        "sha256sums_sha256": sha256sums_sha256,
        "status": "PASS",
        "summary_sha256": summary_sha256,
        "world_size": int(world_size),
    }
    if is_action_only_n2:
        complete.update(
            {
                "checkpoint_state_kind": checkpoint_state_kind,
                "formal_n4_fullmodel_gate": False,
                "run_profile": run_profile,
                "schema_version": 2,
                "task_scope_receipt_sha256": n2_evidence["task_scope"]["sha256"],
                "training_terminal_contract": contract_name,
                "training_terminal_contract_version": (
                    ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT_VERSION
                ),
            }
        )
        if reload_proof is not None:
            complete.update(
                {
                    "fresh_process_reload_verified": True,
                    "load_attempt_id": reload_proof["load_attempt_id"],
                    "proof_attempt_id": reload_proof["proof_attempt_id"],
                    "rank_state_aggregate_sha256": reload_proof[
                        "rank_state_aggregate_sha256"
                    ],
                    "reload_proof_checkpoint_binding_sha256": reload_proof[
                        "checkpoint_binding"
                    ]["sha256"],
                    "reload_proof_committed_load_attempt_sha256": reload_proof[
                        "committed_load_attempt"
                    ]["sha256"],
                    "terminal_arguments_sha256": reload_proof[
                        "terminal_candidate"
                    ]["arguments_sha256"],
                    "terminal_candidate_sha256": reload_proof[
                        "terminal_candidate"
                    ]["sha256"],
                }
            )
    publish_exclusive_json(output_root / "TRAINING.COMPLETE", complete)
    return complete


def publish_action_only_n2_terminal_candidate(
    output_root: str | Path,
    *,
    terminal_arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Stage the save world's terminal inputs without claiming completion."""

    output_root = resolved_unaliased_directory(output_root, label="training output root")
    for terminal_name in ("TRAINING.COMPLETE", "TRAINING.FAILED.json"):
        terminal_path = output_root / terminal_name
        if terminal_path.exists() or terminal_path.is_symlink():
            raise RuntimeError(
                "refusing N=2 terminal candidate after a run-level terminal marker: "
                f"{terminal_path}"
            )
    arguments = dict(terminal_arguments)
    if (
        arguments.get("training_terminal_contract")
        != ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT
        or arguments.get("run_profile") != "paid_gate_1step"
        or arguments.get("world_size") != ACTION_ONLY_N2_1X8_WORLD_SIZE
        or arguments.get("max_steps") != ACTION_ONLY_N2_PAID_GATE_STEP
    ):
        raise ValueError("terminal candidate is not the exact N=2 paid-gate contract")
    payload = {
        "arguments": arguments,
        "arguments_sha256": canonical_json_sha256(arguments),
        "run_id": str(arguments.get("run_id", "")),
        "schema_name": "fastwam-action-only-n2-terminal-candidate",
        "schema_version": 1,
        "status": "AWAITING_FRESH_RELOAD",
    }
    publish_exclusive_json(output_root / ACTION_ONLY_N2_TERMINAL_CANDIDATE, payload)
    return payload


def finalize_action_only_n2_paid_gate(
    output_root: str | Path,
) -> dict[str, Any]:
    """Publish TRAINING.COMPLETE only after the staged 8-rank reload proof passes."""

    output_root = resolved_unaliased_directory(output_root, label="training output root")
    failure_marker = output_root / "TRAINING.FAILED.json"
    if failure_marker.exists() or failure_marker.is_symlink():
        raise RuntimeError(
            "refusing N=2 paid-gate finalization after TRAINING.FAILED.json exists"
        )
    candidate, arguments, _ = _read_action_only_n2_terminal_candidate(output_root)
    if (
        arguments.get("training_terminal_contract")
        != ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT
        or arguments.get("run_profile") != "paid_gate_1step"
        or arguments.get("world_size") != ACTION_ONLY_N2_1X8_WORLD_SIZE
        or arguments.get("max_steps") != ACTION_ONLY_N2_PAID_GATE_STEP
    ):
        raise RuntimeError("N=2 terminal candidate identity/contract mismatch")
    return publish_training_terminal_seal(output_root, **arguments)


def _load_rank_proofs(proof_dir: Path, pattern: str, *, expected: int) -> list[dict[str, Any]]:
    paths = sorted(proof_dir.glob(pattern), key=lambda path: os.fsencode(path.name))
    if len(paths) != expected:
        raise RuntimeError(f"expected exactly {expected} proofs matching {pattern}, got {len(paths)}")
    payloads = []
    for rank, path in enumerate(paths):
        if path.name != pattern.replace("*", f"{rank:05d}"):
            raise RuntimeError(f"rank proof filename mismatch at rank {rank}: {path.name}")
        payload, _, _ = read_canonical_json(path)
        if payload.get("rank") != rank or payload.get("world_size") != expected:
            raise RuntimeError(f"rank/world proof mismatch in {path}")
        payloads.append(payload)
    return payloads


def _summarize_n4_peak_memory(
    step_proofs: Mapping[int, list[dict[str, Any]]],
) -> dict[str, int | str]:
    """Validate per-rank memory evidence and return a conservative summary.

    Alibaba PAI can schedule RTX 4090 workers with different visible memory
    capacities.  Capacity is therefore a per-rank safety input, not part of
    the cross-rank device identity.  Each rank must report a stable
    ``(device_name, total_device_bytes)`` across both optimizer steps, and
    every proof is checked against the limits derived from that rank's own
    capacity.  The sealed run-level summary uses the smallest observed
    capacity so downstream consumers never infer a larger safety margin than
    the least-capable worker actually provided.
    """

    peak_allocated = 0
    peak_reserved = 0
    rank_identities: dict[int, tuple[str, int]] = {}
    device_names: set[str] = set()
    total_device_capacities: set[int] = set()
    expected_memory_fields = {
        "device_name",
        "effective_max_allocated_bytes",
        "effective_max_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "required_max_allocated_bytes",
        "required_max_reserved_bytes",
        "total_device_bytes",
    }
    for step, proofs in step_proofs.items():
        for proof in proofs:
            rank = int(proof.get("rank", -1))
            memory = proof.get("memory", {})
            if set(memory) != expected_memory_fields:
                raise RuntimeError(f"peak-memory proof fields mismatch: rank={rank}")
            if (
                memory.get("required_max_allocated_bytes")
                != N4_GATE_MAX_PEAK_ALLOCATED_BYTES
                or memory.get("required_max_reserved_bytes")
                != N4_GATE_MAX_PEAK_RESERVED_BYTES
            ):
                raise RuntimeError(f"peak-memory threshold mismatch: rank={rank}")
            raw_device_name = memory.get("device_name")
            device_name = raw_device_name.strip() if isinstance(raw_device_name, str) else ""
            total_device_bytes = int(memory.get("total_device_bytes", -1))
            expected_allocated_limit = min(
                N4_GATE_MAX_PEAK_ALLOCATED_BYTES,
                total_device_bytes * 90 // 100,
            )
            expected_reserved_limit = min(
                N4_GATE_MAX_PEAK_RESERVED_BYTES,
                total_device_bytes * 95 // 100,
            )
            if (
                not device_name
                or total_device_bytes <= 0
                or memory.get("effective_max_allocated_bytes")
                != expected_allocated_limit
                or memory.get("effective_max_reserved_bytes")
                != expected_reserved_limit
            ):
                raise RuntimeError(f"relative peak-memory threshold mismatch: rank={rank}")
            allocated = int(memory.get("peak_allocated_bytes", -1))
            reserved = int(memory.get("peak_reserved_bytes", -1))
            if allocated < 0 or reserved < 0:
                raise RuntimeError(f"missing peak-memory evidence in N=4 proof: rank={rank}")
            if allocated > expected_allocated_limit or reserved > expected_reserved_limit:
                raise RuntimeError(
                    f"N=4 proof exceeds memory gate: rank={rank} "
                    f"allocated={allocated} reserved={reserved}"
                )
            identity = (device_name, total_device_bytes)
            previous_identity = rank_identities.setdefault(rank, identity)
            if previous_identity != identity:
                raise RuntimeError(
                    "N=4 gate CUDA device identity changed between optimizer steps: "
                    f"rank={rank} previous={previous_identity} current={identity} step={step}"
                )
            device_names.add(device_name)
            total_device_capacities.add(total_device_bytes)
            peak_allocated = max(peak_allocated, allocated)
            peak_reserved = max(peak_reserved, reserved)
    if set(rank_identities) != set(range(N4_GATE_WORLD_SIZE)):
        raise RuntimeError(
            "N=4 gate memory evidence does not cover exactly all ranks: "
            f"observed={sorted(rank_identities)}"
        )
    if len(device_names) != 1:
        raise RuntimeError(
            f"N=4 gate requires the same non-empty CUDA device name on all 32 ranks: {device_names}"
        )
    minimum_total_device_bytes = min(total_device_capacities)
    conservative_allocated_limit = min(
        N4_GATE_MAX_PEAK_ALLOCATED_BYTES,
        minimum_total_device_bytes * 90 // 100,
    )
    conservative_reserved_limit = min(
        N4_GATE_MAX_PEAK_RESERVED_BYTES,
        minimum_total_device_bytes * 95 // 100,
    )
    if (
        peak_allocated > conservative_allocated_limit
        or peak_reserved > conservative_reserved_limit
    ):
        raise RuntimeError(
            "N=4 proofs exceed the conservative run-level memory gate: "
            f"allocated={peak_allocated}/{conservative_allocated_limit} "
            f"reserved={peak_reserved}/{conservative_reserved_limit}"
        )
    return {
        "device_name": next(iter(device_names)),
        "effective_max_allocated_bytes": conservative_allocated_limit,
        "effective_max_reserved_bytes": conservative_reserved_limit,
        "total_device_bytes": minimum_total_device_bytes,
        "max_allocated_bytes": peak_allocated,
        "max_reserved_bytes": peak_reserved,
        "required_max_allocated_bytes": N4_GATE_MAX_PEAK_ALLOCATED_BYTES,
        "required_max_reserved_bytes": N4_GATE_MAX_PEAK_RESERVED_BYTES,
    }


def finalize_n4_fullmodel_gate(
    output_root: str | Path,
    *,
    run_id: str,
    code_commit: str,
    image_reference: str,
    image_digest: str,
    input_bindings: Mapping[str, str],
) -> dict[str, Any]:
    output_root = resolved_unaliased_directory(output_root, label="N=4 gate output root")
    for name in ("manifest.json", "SHA256SUMS", "COMPLETE"):
        target = output_root / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to replace N=4 gate terminal artifact: {target}")
    code_commit = str(code_commit).lower()
    if len(code_commit) != 40 or any(character not in SHA256_HEX for character in code_commit):
        raise ValueError("N=4 gate requires a 40-hex code commit")
    image_digest = str(image_digest).lower()
    if not image_reference or not image_digest.startswith("sha256:"):
        raise ValueError("N=4 gate requires an exact image reference and OCI digest")
    require_sha256(image_digest.split(":", 1)[1], label="OCI image digest")
    expected_batch = {
        "global_train_batch_size": N4_GATE_GLOBAL_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": N4_GATE_GRADIENT_ACCUMULATION_STEPS,
        "local_micro_batch_size": N4_GATE_LOCAL_MICRO_BATCH_SIZE,
        "world_size": N4_GATE_WORLD_SIZE,
    }
    expected_shapes = {
        "action": [1, 4, 32, 8],
        "agent_gaussian": [1, 4, 13, 28, 40],
        "agent_geometry": [1, 4, 7],
        "agent_state": [1, 4, 18],
        "video": [1, 3, 9, 224, 320],
    }
    proof_dir = output_root / "gate-proofs"
    if proof_dir.is_symlink() or not proof_dir.is_dir():
        raise ValueError(f"N=4 gate proof directory is invalid: {proof_dir}")
    expected_proof_names = {
        *(f"step-{step:06d}-rank-{rank:05d}.json" for step in range(1, N4_GATE_TRAIN_STEPS + 1) for rank in range(N4_GATE_WORLD_SIZE)),
        *(f"save-state-rank-{rank:05d}.json" for rank in range(N4_GATE_WORLD_SIZE)),
        *(f"load-state-rank-{rank:05d}.json" for rank in range(N4_GATE_WORLD_SIZE)),
    }
    observed_proof_names = set()
    for path in proof_dir.iterdir():
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"N=4 gate proof root contains an aliased/special entry: {path}")
        observed_proof_names.add(path.name)
    if observed_proof_names != expected_proof_names:
        raise RuntimeError(
            "N=4 gate proof path set mismatch: "
            f"missing={sorted(expected_proof_names - observed_proof_names)[:12]} "
            f"unexpected={sorted(observed_proof_names - expected_proof_names)[:12]}"
        )
    step_proofs = {
        step: _load_rank_proofs(
            proof_dir, f"step-{step:06d}-rank-*.json", expected=N4_GATE_WORLD_SIZE
        )
        for step in range(1, N4_GATE_TRAIN_STEPS + 1)
    }
    save_proofs = _load_rank_proofs(
        proof_dir, "save-state-rank-*.json", expected=N4_GATE_WORLD_SIZE
    )
    load_proofs = _load_rank_proofs(
        proof_dir, "load-state-rank-*.json", expected=N4_GATE_WORLD_SIZE
    )
    expected_step_fields = {
        "agent_count",
        "batch_accounting",
        "gradients",
        "hub_token_policy",
        "losses",
        "memory",
        "num_hub_tokens",
        "phase",
        "process_nonce",
        "process_pid",
        "process_start_ticks",
        "rank",
        "sample_shapes",
        "schema_name",
        "schema_version",
        "step",
        "world_size",
    }
    for step, proofs in step_proofs.items():
        for proof in proofs:
            if (
                set(proof) != expected_step_fields
                or proof.get("schema_name") != "fastwam-n4-fullmodel-step-proof"
                or proof.get("schema_version") != 1
                or proof.get("phase") != "train_step"
                or proof.get("step") != step
                or proof.get("batch_accounting") != expected_batch
                or proof.get("sample_shapes") != expected_shapes
                or proof.get("agent_count") != 4
                or proof.get("num_hub_tokens") != 8
                or proof.get("hub_token_policy") != "ceil(hub_token_ratio*num_agents)"
            ):
                raise RuntimeError(f"N=4 step proof semantic mismatch: rank={proof.get('rank')} step={step}")
            losses = proof.get("losses", {})
            gradients = proof.get("gradients", {})
            if set(losses) != {"action", "total", "video"} or not all(
                np.isfinite(float(value)) for value in losses.values()
            ):
                raise RuntimeError(f"non-finite/incomplete losses in N=4 proof: rank={proof.get('rank')}")
            gradient_source = gradients.get("source")
            if gradient_source == "deepspeed_global_grad_norm":
                expected_gradient_fields = {"all_finite", "norm", "source"}
                source_valid = True
            elif gradient_source == "parameter_grad_scan":
                expected_gradient_fields = {
                    "all_finite",
                    "norm",
                    "source",
                    "tensor_count",
                }
                source_valid = int(gradients.get("tensor_count", 0)) > 0
            else:
                expected_gradient_fields = set()
                source_valid = False
            gradient_norm = float(gradients.get("norm", float("nan")))
            if (
                set(gradients) != expected_gradient_fields
                or gradients.get("all_finite") is not True
                or not source_valid
                or not np.isfinite(gradient_norm)
                or gradient_norm <= 0.0
            ):
                raise RuntimeError(f"non-finite/missing gradients in N=4 proof: rank={proof.get('rank')}")
    peak_memory = _summarize_n4_peak_memory(step_proofs)
    roundtrip_checks = {
        "global_step": True,
        "model": True,
        "optimizer": True,
        "rng": True,
        "rng_next_sample": True,
        "scheduler": True,
        "separate_process": True,
        "pre_load_was_distinct": True,
    }
    expected_save_fields = {
        "batch_accounting",
        "fingerprints",
        "next_rng_sample",
        "phase",
        "process_nonce",
        "process_pid",
        "process_start_ticks",
        "rank",
        "schema_name",
        "schema_version",
        "world_size",
    }
    expected_load_fields = {
        "batch_accounting",
        "checks",
        "fingerprints",
        "next_rng_sample",
        "phase",
        "pre_load_fingerprints",
        "process_nonce",
        "process_pid",
        "process_start_ticks",
        "rank",
        "schema_name",
        "schema_version",
        "world_size",
    }
    fingerprint_keys = {
        "global_step",
        "model",
        "model_probe",
        "optimizer",
        "optimizer_probe",
        "rng",
        "scheduler",
    }
    check_keys = {
        "global_step",
        "model",
        "optimizer",
        "pre_load_was_distinct",
        "rng",
        "rng_next_sample",
        "scheduler",
    }
    for rank, (saved, loaded) in enumerate(zip(save_proofs, load_proofs, strict=True)):
        if (
            set(saved) != expected_save_fields
            or set(loaded) != expected_load_fields
            or saved.get("schema_name") != "fastwam-n4-fullmodel-save-proof"
            or loaded.get("schema_name") != "fastwam-n4-fullmodel-load-proof"
            or saved.get("schema_version") != 1
            or loaded.get("schema_version") != 1
            or saved.get("phase") != "save_after_full_checkpoint"
            or loaded.get("phase") != "load_fresh_process"
            or saved.get("batch_accounting") != expected_batch
            or loaded.get("batch_accounting") != expected_batch
        ):
            raise RuntimeError(f"N=4 save/load proof schema mismatch at rank {rank}")
        checks = loaded.get("checks", {})
        saved_fingerprints = saved.get("fingerprints", {})
        loaded_fingerprints = loaded.get("fingerprints", {})
        pre_load_fingerprints = loaded.get("pre_load_fingerprints", {})
        if (
            set(checks) != check_keys
            or set(saved_fingerprints) != fingerprint_keys
            or set(loaded_fingerprints) != fingerprint_keys
            or set(pre_load_fingerprints) != fingerprint_keys
        ):
            raise RuntimeError(f"N=4 state fingerprint/check field mismatch at rank {rank}")
        if (
            saved_fingerprints["global_step"] != N4_GATE_TRAIN_STEPS
            or loaded_fingerprints["global_step"] != N4_GATE_TRAIN_STEPS
        ):
            raise RuntimeError(f"N=4 save/load proof did not bind global_step=2 at rank {rank}")
        for key in ("global_step", "model", "optimizer", "rng", "scheduler"):
            direct_match = loaded_fingerprints[key] == saved_fingerprints[key]
            roundtrip_checks[key] &= bool(checks.get(key)) and direct_match
        direct_rng_next = loaded.get("next_rng_sample") == saved.get("next_rng_sample")
        roundtrip_checks["rng_next_sample"] &= (
            bool(checks.get("rng_next_sample")) and direct_rng_next
        )
        direct_pre_load_distinct = any(
            pre_load_fingerprints[key] != saved_fingerprints[key]
            for key in ("global_step", "model", "optimizer", "rng", "scheduler")
        )
        roundtrip_checks["pre_load_was_distinct"] &= (
            bool(checks.get("pre_load_was_distinct")) and direct_pre_load_distinct
        )
        roundtrip_checks["separate_process"] &= (
            saved.get("process_nonce") != loaded.get("process_nonce")
            and (saved.get("process_pid"), saved.get("process_start_ticks"))
            != (loaded.get("process_pid"), loaded.get("process_start_ticks"))
        )
    if not all(roundtrip_checks.values()):
        raise RuntimeError(f"N=4 full-state roundtrip aggregation failed: {roundtrip_checks}")

    checkpoint = checkpoint_seal_descriptor(
        output_root, step=N4_GATE_TRAIN_STEPS, rehash_weights=True
    )
    expected_binding_keys = {
        "cpfs_bundle_manifest",
        "gaussian_cache_manifest",
        "gaussian_cache_selection",
        "gaussian_cache_source_identity",
        "official_checkpoint",
        "oss_bundle_manifest",
        "synthetic_zero2_gate",
        "stats",
        "training_environment_bundle",
        "vae",
    }
    if set(input_bindings) != expected_binding_keys:
        raise ValueError(
            "N=4 gate input binding key set mismatch: "
            f"missing={sorted(expected_binding_keys - set(input_bindings))} "
            f"unexpected={sorted(set(input_bindings) - expected_binding_keys)}"
        )
    normalized_bindings = {
        key: require_sha256(value, label=f"input binding {key}")
        for key, value in sorted(input_bindings.items())
    }
    reservation, reservation_sha256, _ = read_canonical_json(
        output_root / ".RUN_RESERVED"
    )
    reservation_identity_sha256 = require_sha256(
        reservation.get("identity_sha256", ""),
        label="N=4 gate reservation identity SHA-256",
    )
    reservation_identity_payload = dict(reservation)
    del reservation_identity_payload["identity_sha256"]
    if canonical_json_sha256(reservation_identity_payload) != reservation_identity_sha256:
        raise RuntimeError("N=4 gate reservation identity_sha256 does not match its payload")
    expected_reservation = {
        "bundle_manifest_sha256": None,
        "cache_manifest_sha256": normalized_bindings["gaussian_cache_manifest"],
        "cache_selection_sha256": normalized_bindings["gaussian_cache_selection"],
        "cache_source_identity_sha256": normalized_bindings[
            "gaussian_cache_source_identity"
        ],
        "checkpoint_sha256": normalized_bindings["official_checkpoint"],
        "code_commit": code_commit,
        "cpfs_bundle_manifest_sha256": normalized_bindings["cpfs_bundle_manifest"],
        "global_world_size": N4_GATE_WORLD_SIZE,
        "image_digest": image_digest,
        "image_digest_status": "resolved",
        "image_reference": str(image_reference),
        "n4_fullmodel_gate_complete_sha256": None,
        "nproc_per_node": 8,
        "num_machines": 4,
        "oss_bundle_manifest_sha256": normalized_bindings["oss_bundle_manifest"],
        "output_storage": "oss_experimental",
        "output_zero_checkpoint_smoke_sha256": normalized_bindings[
            "synthetic_zero2_gate"
        ],
        "run_id": str(run_id),
        "schema_version": 1,
        "stats_sha256": normalized_bindings["stats"],
        "task": "robofactory_multi_robot_vg1_hub1_gau1_224_1e-4",
        "training_env_bundle_manifest_sha256": normalized_bindings[
            "training_environment_bundle"
        ],
        "vae_sha256": normalized_bindings["vae"],
    }
    reservation_mismatches = {
        key: {"expected": expected, "observed": reservation.get(key)}
        for key, expected in expected_reservation.items()
        if reservation.get(key) != expected
    }
    if reservation_mismatches:
        raise RuntimeError(
            "N=4 gate reservation does not bind the proof/finalizer identity: "
            f"{reservation_mismatches}"
        )
    manifest = {
        "batch_accounting": expected_batch,
        "checkpoint": checkpoint,
        "code_commit": code_commit,
        "image_digest": image_digest,
        "image_reference": str(image_reference),
        "input_bindings": normalized_bindings,
        "peak_memory": peak_memory,
        "proof_counts": {
            "load_state": len(load_proofs),
            "save_state": len(save_proofs),
            "step_1": len(step_proofs[1]),
            "step_2": len(step_proofs[2]),
        },
        "published_at": datetime.now(timezone.utc).isoformat(),
        "reservation": {
            "identity_sha256": reservation_identity_sha256,
            "path": ".RUN_RESERVED",
            "sha256": reservation_sha256,
        },
        "roundtrip": roundtrip_checks,
        "run_id": str(run_id),
        "schema_name": "fastwam-n4-fullmodel-gate",
        "schema_version": 1,
        "status": "PASS",
        "train_steps": N4_GATE_TRAIN_STEPS,
        "world_size": N4_GATE_WORLD_SIZE,
        "zero_stage": 2,
    }
    manifest_path = output_root / "manifest.json"
    publish_exclusive_json(manifest_path, manifest)
    bound_paths = ["manifest.json", ".RUN_RESERVED", "config.save.yaml", "config.load.yaml"]
    bound_paths.extend(path.relative_to(output_root).as_posix() for path in sorted(proof_dir.glob("*.json")))
    bound_paths.extend(
        (
            checkpoint["weights"]["manifest"],
            checkpoint["weights"]["complete"],
            checkpoint["state"]["manifest"],
            f"{checkpoint['state']['root']}/trainer_state.json",
        )
    )
    sha256sums_sha256, bound_paths = _publish_sha256sums(output_root, bound_paths)
    manifest_sha256, _ = sha256_regular_file(manifest_path)
    complete = {
        "bound_paths": bound_paths,
        "manifest_sha256": manifest_sha256,
        "run_id": str(run_id),
        "schema_name": "fastwam-n4-fullmodel-gate-complete",
        "schema_version": 1,
        "sha256sums_sha256": sha256sums_sha256,
        "status": "PASS",
        "world_size": N4_GATE_WORLD_SIZE,
    }
    publish_exclusive_json(output_root / "COMPLETE", complete)
    return complete


def validate_terminal_sha256sums(
    output_root: str | Path,
    *,
    complete_name: str,
    expected_complete_schema: str,
) -> dict[str, Any]:
    output_root = resolved_unaliased_directory(output_root, label="formal output root")
    complete, complete_sha256, _ = read_canonical_json(output_root / complete_name)
    if (
        complete.get("schema_name") != expected_complete_schema
        or complete.get("schema_version") != 1
        or complete.get("status") != "PASS"
    ):
        raise ValueError(f"terminal COMPLETE schema/status mismatch: {output_root / complete_name}")
    sha_path = output_root / "SHA256SUMS"
    sha_digest, _ = sha256_regular_file(sha_path)
    if sha_digest != complete.get("sha256sums_sha256"):
        raise RuntimeError("terminal COMPLETE does not bind SHA256SUMS")
    records = []
    previous_record_key: bytes | None = None
    with sha_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(line) < 67 or line[64:66] != "  ":
                raise ValueError(f"invalid SHA256SUMS line: {line!r}")
            expected = require_sha256(line[:64], label="SHA256SUMS digest")
            relative = safe_relative_path(line[66:].rstrip("\n"))
            record_key = os.fsencode(relative.as_posix())
            if previous_record_key is not None and record_key <= previous_record_key:
                raise ValueError("SHA256SUMS paths must be unique and bytewise sorted")
            previous_record_key = record_key
            actual, _ = sha256_regular_file(output_root / relative)
            if actual != expected:
                raise RuntimeError(
                    f"terminal artifact SHA-256 mismatch: expected={expected} actual={actual} path={relative}"
                )
            records.append(relative.as_posix())
    if records != complete.get("bound_paths"):
        raise RuntimeError("terminal COMPLETE bound_paths do not exactly match SHA256SUMS")
    return {
        "bound_paths": records,
        "complete_sha256": complete_sha256,
        "sha256sums_sha256": sha_digest,
        "status": "PASS",
    }


def validate_n4_fullmodel_gate_binding(
    output_root: str | Path,
    *,
    allowed_prefix: str | Path,
    forbidden_output_root: str | Path,
    expected_complete_sha256: str,
    code_commit: str,
    image_reference: str,
    image_digest: str,
    input_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Validate that a PASS gate authorizes this exact main-run identity."""

    output_root = resolved_unaliased_directory(output_root, label="N=4 gate output root")
    allowed_prefix = resolved_unaliased_directory(
        allowed_prefix, label="N=4 gate allowed storage prefix"
    )
    if output_root == allowed_prefix:
        raise ValueError("N=4 gate output must be a child of its allowed storage prefix")
    try:
        output_root.relative_to(allowed_prefix)
    except ValueError as error:
        raise ValueError(
            f"N=4 gate output {output_root} is outside allowed prefix {allowed_prefix}"
        ) from error
    forbidden_supplied = Path(forbidden_output_root).expanduser()
    if not forbidden_supplied.is_absolute():
        raise ValueError(
            f"main training output root must be absolute: {forbidden_supplied}"
        )
    forbidden_resolved = forbidden_supplied.resolve(strict=False)
    if (
        output_root == forbidden_resolved
        or output_root in forbidden_resolved.parents
        or forbidden_resolved in output_root.parents
    ):
        raise ValueError(
            "N=4 gate output and main training output must be independent: "
            f"gate={output_root} main={forbidden_resolved}"
        )
    expected_complete_sha256 = require_sha256(
        expected_complete_sha256, label="N=4 gate COMPLETE SHA-256"
    )
    complete, actual_complete_sha256, _ = read_canonical_json(output_root / "COMPLETE")
    if actual_complete_sha256 != expected_complete_sha256:
        raise RuntimeError(
            "N=4 gate COMPLETE SHA-256 mismatch: "
            f"expected={expected_complete_sha256} actual={actual_complete_sha256}"
        )
    validate_terminal_sha256sums(
        output_root,
        complete_name="COMPLETE",
        expected_complete_schema="fastwam-n4-fullmodel-gate-complete",
    )
    manifest, manifest_sha256, _ = read_canonical_json(output_root / "manifest.json")
    if complete.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("N=4 gate COMPLETE does not bind manifest.json")
    expected_manifest_fields = {
        "batch_accounting",
        "checkpoint",
        "code_commit",
        "image_digest",
        "image_reference",
        "input_bindings",
        "peak_memory",
        "proof_counts",
        "published_at",
        "reservation",
        "roundtrip",
        "run_id",
        "schema_name",
        "schema_version",
        "status",
        "train_steps",
        "world_size",
        "zero_stage",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("N=4 gate manifest field set mismatch")
    expected_batch = {
        "global_train_batch_size": N4_GATE_GLOBAL_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": N4_GATE_GRADIENT_ACCUMULATION_STEPS,
        "local_micro_batch_size": N4_GATE_LOCAL_MICRO_BATCH_SIZE,
        "world_size": N4_GATE_WORLD_SIZE,
    }
    expected_roundtrip = {
        "global_step": True,
        "model": True,
        "optimizer": True,
        "pre_load_was_distinct": True,
        "rng": True,
        "rng_next_sample": True,
        "scheduler": True,
        "separate_process": True,
    }
    expected_proof_counts = {
        "load_state": N4_GATE_WORLD_SIZE,
        "save_state": N4_GATE_WORLD_SIZE,
        "step_1": N4_GATE_WORLD_SIZE,
        "step_2": N4_GATE_WORLD_SIZE,
    }
    reservation, reservation_sha256, _ = read_canonical_json(
        output_root / ".RUN_RESERVED"
    )
    reservation_identity_sha256 = require_sha256(
        reservation.get("identity_sha256", ""),
        label="N=4 gate reservation identity SHA-256",
    )
    reservation_identity_payload = dict(reservation)
    del reservation_identity_payload["identity_sha256"]
    if canonical_json_sha256(reservation_identity_payload) != reservation_identity_sha256:
        raise RuntimeError("N=4 gate reservation identity SHA-256 mismatch")
    expected_reservation_descriptor = {
        "identity_sha256": reservation_identity_sha256,
        "path": ".RUN_RESERVED",
        "sha256": reservation_sha256,
    }
    if (
        manifest.get("schema_name") != "fastwam-n4-fullmodel-gate"
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "PASS"
        or manifest.get("world_size") != N4_GATE_WORLD_SIZE
        or manifest.get("zero_stage") != 2
        or manifest.get("train_steps") != N4_GATE_TRAIN_STEPS
        or manifest.get("batch_accounting") != expected_batch
        or manifest.get("roundtrip") != expected_roundtrip
        or manifest.get("proof_counts") != expected_proof_counts
        or manifest.get("reservation") != expected_reservation_descriptor
    ):
        raise RuntimeError("N=4 gate manifest does not satisfy the formal PASS contract")
    normalized_bindings = {
        key: require_sha256(value, label=f"current input binding {key}")
        for key, value in sorted(input_bindings.items())
    }
    if manifest.get("input_bindings") != normalized_bindings:
        raise RuntimeError(
            "N=4 gate input bindings do not match the proposed main run: "
            f"gate={manifest.get('input_bindings')} current={normalized_bindings}"
        )
    if (
        manifest.get("code_commit") != str(code_commit).lower()
        or manifest.get("image_reference") != str(image_reference)
        or manifest.get("image_digest") != str(image_digest).lower()
    ):
        raise RuntimeError(
            "N=4 gate code/image identity does not match the proposed main run"
        )
    # The small outer terminal files prove what was observed when the gate was
    # finalized, but they cannot make a mutable OSS/FUSE checkpoint immutable.
    # Re-read and hash the actual full weights plus every ZeRO state shard at
    # authorization time, then require the resulting descriptor to be exactly
    # the one sealed in the gate manifest.  A deleted or modified large file
    # must therefore invalidate the gate before the main run starts.
    observed_checkpoint = checkpoint_seal_descriptor(
        output_root,
        step=N4_GATE_TRAIN_STEPS,
        rehash_weights=True,
    )
    if manifest.get("checkpoint") != observed_checkpoint:
        raise RuntimeError(
            "N=4 gate checkpoint changed after finalization: "
            f"sealed={manifest.get('checkpoint')} observed={observed_checkpoint}"
        )
    return {
        "complete_sha256": actual_complete_sha256,
        "manifest_sha256": manifest_sha256,
        "run_id": manifest["run_id"],
        "status": "PASS",
    }
