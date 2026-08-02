#!/usr/bin/env python3
"""Specialized, fail-closed audit for paired canonical MF-WAM G0 bundles.

Unlike ``audit_mf_wam_g0.py`` (a retained legacy diagnostic), this program
reads the canonical schema-v2 episode traces and task receipts, independently
reconstructs every episode outcome, and recomputes the locked paired bootstrap.
It can emit the exact specialized receipt envelope consumed structurally by
``mf_wam_gates.py``.  The receipt never authorizes training: the independent
external-anchor consumer and runtime authorizer remain separate gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

# Avoid creating ignored Python bytecode in a fresh auditor checkout before its
# own source identity gate runs.
sys.dont_write_bytecode = True

try:
    from fastwam.validation.g0_contract import (
        canonical_json_sha256 as _contract_canonical_sha256,
        validate_contract_chain,
        validate_data_inventory,
        validate_preregistration,
        validate_runtime_start,
        validate_seed_schedule,
        validate_terminal_receipt,
    )
except ModuleNotFoundError:
    _REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))
    from fastwam.validation.g0_contract import (  # type: ignore[no-redef]
        canonical_json_sha256 as _contract_canonical_sha256,
        validate_contract_chain,
        validate_data_inventory,
        validate_preregistration,
        validate_runtime_start,
        validate_seed_schedule,
        validate_terminal_receipt,
    )


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
CANONICAL_JSON_ALGORITHM = "python-json-sort-keys-utf8-v1"
TREE_ALGORITHM = "sha256sum-posix-path-v1"
CI_CONTRACT_ID = "MF-WAM-G0-CI-v1"
EXPECTED_FASTWAM_COMMIT = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
EXPECTED_LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
EXPECTED_APPROVED_ASSETS_RAW_SHA256 = (
    "f4cbb8ce0f518782f7083ce9e0c90dfd0c8f1000a624ea3e46251488b62bc690"
)

TRACE_TOP_KEYS = {"schema_version", "kind", "metadata", "records"}
TRACE_METADATA_KEYS = {
    "run_id",
    "task_suite",
    "task_id",
    "trial_idx",
    "initial_state_index",
    "initial_state_sha256",
    "task_description",
    "warmup_steps",
    "first_replan_env_step",
    "replan_steps",
    "action_horizon",
    "action_dimension",
    "state_dimension",
    "seed_contract",
    "seed_schedule_process",
    "upstream_digests",
    "official_source",
    "instrumentation_source",
    "success",
    "record_count",
    "environment_step_count",
    "observer_rng_unchanged_checks",
    "official_module_origin_inventory_sha256",
}
TRACE_RECORD_KEYS = {
    "episode_idx",
    "replan_idx",
    "env_step",
    "state",
    "pre_state",
    "pre_observation_sha256",
    "policy_seed",
    "policy_seed_scope",
    "proposed_raw_action_chunk",
    "proposed_env_action_chunk",
    "executed_env_actions",
    "executed_count",
    "done_after_execution",
    "executions",
}
TRACE_EXECUTION_KEYS = {"action", "post_state", "post_observation_sha256", "done"}
TRACE_SEED_CONTRACT_KEYS = {
    "task_seed",
    "effective_global_rank",
    "effective_process_seed",
    "task_seed_scope",
    "environment_seed",
    "environment_seed_scope",
    "policy_seed",
    "policy_seed_scope",
    "episode_rng_position",
}
TASK_RECEIPT_KEYS = {
    "schema_version",
    "kind",
    "run_id",
    "process_id",
    "task_suite",
    "task_id",
    "execution_scope",
    "world_size",
    "global_rank",
    "local_rank",
    "bindings",
    "seeds",
    "official_result",
    "episode_count",
    "traces",
    "tree_sha256",
}
TASK_RECEIPT_BINDING_KEYS = {
    "preregistration_canonical_sha256",
    "runtime_start_canonical_sha256",
    "seed_schedule_canonical_sha256",
    "resolved_config_sha256",
    "image_digest",
    "fastwam_commit",
    "instrumentation_commit",
}
TASK_RECEIPT_SEED_KEYS = {
    "global_seed",
    "environment_seed",
    "environment_seed_scope",
    "policy_seed",
    "policy_seed_scope",
    "python_hash_seed",
    "trial_order",
    "initial_state_index_rule",
}
RUN_ANCHOR_KEYS = {
    "schema_version",
    "kind",
    "run_role",
    "run_id",
    "artifact_root",
    "raw_log_root",
    "manager_manifest_sha256",
    "preregistration_canonical_sha256",
    "runtime_start_canonical_sha256",
    "seed_schedule_canonical_sha256",
    "resolved_config_sha256",
    "terminal_canonical_sha256",
    "structural_audit_file_sha256",
    "approved_assets_manifest_sha256",
    "image_digest",
    "fastwam_commit",
    "instrumentation_commit",
}
MANAGER_TOP_KEYS = {
    "schema_version",
    "kind",
    "run_id",
    "completed_at",
    "manager_exit_code",
    "artifact_root",
    "raw_log_root",
    "gpu_ids",
    "upstream_bindings",
    "canonical_input_file_count",
    "canonical_input_tree_sha256",
    "task_processes",
}
MANAGER_PROCESS_KEYS = {
    "process_id",
    "task_suite",
    "task_id",
    "gpu_id",
    "state",
    "launched_at",
    "completed_at",
    "exit_code",
    "complete",
    "failure_reason",
    "command_sha256",
    "environment_sha256",
    "log_path",
    "log_sha256",
    "log_size_bytes",
    "status_path",
    "status_sha256",
    "status_size_bytes",
    "result_path",
    "result_sha256",
    "result_size_bytes",
    "trace_receipt_path",
    "trace_receipt_sha256",
    "trace_receipt_size_bytes",
    "trace_tree_sha256",
    "episode_count",
    "raw_result_source_path",
    "raw_result_archive_path",
    "raw_result_sha256",
    "raw_result_size_bytes",
}
MANAGER_STATUS_KEYS = {
    "schema_version",
    "kind",
    "run_id",
    "process_id",
    "task_suite",
    "task_id",
    "gpu_id",
    "state",
    "launched_at",
    "completed_at",
    "exit_code",
    "complete",
    "failure_reason",
    "command_argv",
    "command_sha256",
    "environment_bindings",
    "environment_sha256",
    "log",
    "canonical_result",
    "trace_receipt",
    "raw_result",
}
FORMAL_HYDRA_OVERRIDE_KEYS = {
    "task",
    "ckpt",
    "gpu_id",
    "seed",
    "output_dir",
    "EVALUATION.task_suite_name",
    "EVALUATION.task_id",
    "EVALUATION.output_dir",
    "EVALUATION.dataset_stats_path",
    "EVALUATION.num_trials",
    "EVALUATION.env_num",
    "EVALUATION.num_steps_wait",
    "EVALUATION.replan_steps",
    "EVALUATION.binarize_gripper",
    "EVALUATION.use_action_ensembler",
    "EVALUATION.visualize_future_video",
    "EVALUATION.action_horizon",
}
DYNAMIC_RUNTIME_OVERLAY_PATHS = (
    ("gpu_id",),
    ("EVALUATION", "task_suite_name"),
    ("EVALUATION", "task_id"),
)
_PAIRED_ARTIFACT_ROOT_SENTINEL = "<MF_WAM_G0_RUN_ARTIFACT_ROOT>"
_FIXED_WORKER_ENVIRONMENT = {
    "HOME": "/tmp",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONUTF8": "1",
    "TMPDIR": "/tmp",
    "TZ": "UTC",
}
_GIT_ENVIRONMENT = {
    **_FIXED_WORKER_ENVIRONMENT,
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
}
_FORMAL_ENVIRONMENT_KEYS = {
    *_FIXED_WORKER_ENVIRONMENT,
    "CUDA_VISIBLE_DEVICES",
    "DIFFSYNTH_DOWNLOAD_SOURCE",
    "DIFFSYNTH_MODEL_BASE_PATH",
    "DIFFSYNTH_SKIP_DOWNLOAD",
    "LOCAL_RANK",
    "MF_WAM_G0_PREREG_PATH",
    "MF_WAM_G0_PREREG_SHA256",
    "MF_WAM_G0_RESOLVED_CONFIG_PATH",
    "MF_WAM_G0_RESOLVED_CONFIG_SHA256",
    "MF_WAM_G0_RUN_ID",
    "MF_WAM_G0_RUNTIME_START_PATH",
    "MF_WAM_G0_RUNTIME_START_SHA256",
    "MF_WAM_G0_SEED_SCHEDULE_PATH",
    "MF_WAM_G0_SEED_SCHEDULE_SHA256",
    "MF_WAM_INSTRUMENTATION_COMMIT",
    "MF_WAM_OFFICIAL_COMMIT",
    "MF_WAM_OFFICIAL_ROOT",
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "PYTHONHASHSEED",
    "RANK",
    "WORLD_SIZE",
}
RECEIPT_ARTIFACT_DIGEST_KEYS = (
    "source_manifest_sha256",
    "data_manifest_sha256",
    "seed_manifest_sha256",
    "resolved_config_sha256",
    "checkpoint_sha256",
    "dataset_stats_sha256",
    "runtime_environment_sha256",
    "identity_inventory_sha256",
    "metric_rows_sha256",
    "trace_tree_sha256",
    "terminal_summary_bundle_sha256",
)
ANCHOR_TYPES = (
    "notion_experiment_page",
    "immutable_artifact_root",
    "source_commit",
    "container_image_digest",
)
_FORMAL_MAX_REPLAN_RECORDS = {
    "libero_spatial": 40,
    "libero_object": 40,
    "libero_goal": 40,
    "libero_10": 70,
}
_OFFICIAL_CRITICAL_PATHS = (
    "experiments/libero/eval_libero_single.py",
    "experiments/libero/libero_utils.py",
    "experiments/libero/action_ensembler.py",
    "configs/sim_libero.yaml",
    "configs/train.yaml",
    "configs/data/libero_2cam.yaml",
    "configs/model/fastwam.yaml",
    "configs/task/libero_uncond_2cam224_1e-4.yaml",
    "src/fastwam/runtime.py",
    "src/fastwam/models/wan22/fastwam.py",
    "src/fastwam/utils/pytorch_utils.py",
)
_INSTRUMENTATION_CRITICAL_PATHS = (
    "scripts/mf_wam_g0_instrumentation.py",
    "scripts/run_mf_wam_g0_traced.py",
)


class SpecializedAuditError(RuntimeError):
    """Raised when any claimed G0 artifact is incomplete or inconsistent."""


@dataclass(frozen=True)
class AuditScope:
    suites: tuple[str, ...] = SUITES
    tasks_per_suite: int = 10
    trials_per_task: int = 50
    minimum_records_per_episode: int = 7
    warmup_steps: int = 30
    replan_steps: int = 10
    action_horizon: int = 32
    action_dimension: int = 7
    state_dimension: int = 8
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 42
    confidence_level: float = 0.95
    overall_margin: float = 0.02
    suite_drop_margin: float = 0.03

    @property
    def episode_count(self) -> int:
        return len(self.suites) * self.tasks_per_suite * self.trials_per_task

    @property
    def is_formal(self) -> bool:
        return self == FORMAL_SCOPE


FORMAL_SCOPE = AuditScope()


def _reject_constant(value: str) -> None:
    raise SpecializedAuditError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SpecializedAuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads_json(raw: bytes, label: str) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
        _assert_finite_tree(value, label)
        return value
    except SpecializedAuditError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SpecializedAuditError(f"cannot parse strict JSON {label}: {exc}") from exc


def _assert_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SpecializedAuditError(f"non-finite number in {label}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{label}[{index}]")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SpecializedAuditError(f"value is not canonical JSON: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value) or value == "0" * 64:
        raise SpecializedAuditError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or not COMMIT_RE.fullmatch(value):
        raise SpecializedAuditError(f"{label} must be a lowercase 40-hex commit")
    return value


def _require_exact_keys(value: Any, keys: Iterable[str], label: str) -> Mapping[str, Any]:
    expected = set(keys)
    if not isinstance(value, Mapping) or set(value) != expected:
        observed = sorted(value, key=repr) if isinstance(value, Mapping) else type(value).__name__
        raise SpecializedAuditError(
            f"{label} fields mismatch: expected={sorted(expected)} observed={observed}"
        )
    return value


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecializedAuditError(f"{label} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in ("", ".", "..") for part in path.parts):
        raise SpecializedAuditError(f"{label} is not canonical or escapes its root: {value!r}")
    return value


def _open_root(path: Path) -> tuple[Path, int]:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        info = lexical.lstat()
    except OSError as exc:
        raise SpecializedAuditError(f"cannot stat root {lexical}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SpecializedAuditError(f"root must be a real directory: {lexical}")
    if lexical.resolve(strict=True) != lexical:
        raise SpecializedAuditError(f"root path contains a symlink: {lexical}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return lexical, os.open(lexical, flags)
    except OSError as exc:
        raise SpecializedAuditError(f"cannot open root {lexical}: {exc}") from exc


def _read_relative(
    root: Path,
    relative: str,
    *,
    maximum_bytes: int = 128 << 20,
    minimum_bytes: int = 1,
    metadata: dict[str, int] | None = None,
) -> tuple[bytes, str, int]:
    relative = _safe_relative_path(relative, "artifact path")
    _, descriptor = _open_root(root)
    current = descriptor
    try:
        parts = PurePosixPath(relative).parts
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        for part in parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=current)
            if current != descriptor:
                os.close(current)
            current = next_fd
        file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(parts[-1], file_flags, dir_fd=current)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise SpecializedAuditError(f"artifact must be a single-link regular file: {relative}")
            if before.st_size < minimum_bytes or before.st_size > maximum_bytes:
                raise SpecializedAuditError(f"artifact size is outside bounds: {relative}")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(fd, min(1 << 20, remaining))
                if not chunk:
                    raise SpecializedAuditError(f"short read: {relative}")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                raise SpecializedAuditError(f"artifact grew during read: {relative}")
            after = os.fstat(fd)
            identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            if identity != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise SpecializedAuditError(f"artifact changed during read: {relative}")
            payload = b"".join(chunks)
            if metadata is not None:
                metadata.update(
                    {
                        "device": after.st_dev,
                        "inode": after.st_ino,
                        "mode": after.st_mode,
                    }
                )
            return payload, hashlib.sha256(payload).hexdigest(), len(payload)
        finally:
            os.close(fd)
    except OSError as exc:
        raise SpecializedAuditError(f"cannot read artifact {relative}: {exc}") from exc
    finally:
        if current != descriptor:
            os.close(current)
        os.close(descriptor)


def _read_absolute(path: Path, *, maximum_bytes: int = 128 << 20) -> tuple[bytes, str, int]:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    if not absolute.is_absolute() or len(absolute.parts) < 2:
        raise SpecializedAuditError(f"invalid absolute artifact path: {path}")
    return _read_relative(Path(absolute.anchor), PurePosixPath(*absolute.parts[1:]).as_posix(), maximum_bytes=maximum_bytes)


def _hash_relative(root: Path, relative: str, *, maximum_bytes: int = 32 << 30) -> tuple[str, int]:
    """Stream-hash a no-follow regular file without retaining its bytes."""

    relative = _safe_relative_path(relative, "artifact path")
    _, descriptor = _open_root(root)
    current = descriptor
    try:
        parts = PurePosixPath(relative).parts
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        for part in parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=current)
            if current != descriptor:
                os.close(current)
            current = next_fd
        fd = os.open(parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise SpecializedAuditError(f"artifact must be a single-link regular file: {relative}")
            if before.st_size < 1 or before.st_size > maximum_bytes:
                raise SpecializedAuditError(f"artifact size is outside bounds: {relative}")
            digest = hashlib.sha256()
            remaining = before.st_size
            while remaining:
                chunk = os.read(fd, min(8 << 20, remaining))
                if not chunk:
                    raise SpecializedAuditError(f"short read: {relative}")
                digest.update(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                raise SpecializedAuditError(f"artifact grew during read: {relative}")
            after = os.fstat(fd)
            identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            if identity != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise SpecializedAuditError(f"artifact changed during read: {relative}")
            return digest.hexdigest(), before.st_size
        finally:
            os.close(fd)
    except OSError as exc:
        raise SpecializedAuditError(f"cannot hash artifact {relative}: {exc}") from exc
    finally:
        if current != descriptor:
            os.close(current)
        os.close(descriptor)


def _hash_absolute(path: Path, *, maximum_bytes: int = 32 << 30) -> tuple[str, int]:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    if not absolute.is_absolute() or len(absolute.parts) < 2:
        raise SpecializedAuditError(f"invalid absolute artifact path: {path}")
    return _hash_relative(
        Path(absolute.anchor),
        PurePosixPath(*absolute.parts[1:]).as_posix(),
        maximum_bytes=maximum_bytes,
    )


def _load_relative_json(root: Path, relative: str) -> tuple[Any, str, int]:
    raw, digest, size = _read_relative(root, relative)
    return _loads_json(raw, f"{root}/{relative}"), digest, size


def _load_absolute_json(path: Path) -> tuple[Any, str, int]:
    raw, digest, size = _read_absolute(path)
    return _loads_json(raw, str(path)), digest, size


def _canonical_absolute_directory(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    opened, descriptor = _open_root(lexical)
    os.close(descriptor)
    if opened != lexical:
        raise SpecializedAuditError(f"{label} is not a canonical absolute directory")
    return lexical


def _git_readonly(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                f"safe.directory={root}",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-C",
                str(root),
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=_GIT_ENVIRONMENT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SpecializedAuditError(
            f"cannot verify official Git checkout {root}: {exc}"
        ) from exc
    return completed.stdout


def _git_readonly_bytes(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                f"safe.directory={root}",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-C",
                str(root),
                *arguments,
            ],
            check=True,
            capture_output=True,
            timeout=30,
            env=_GIT_ENVIRONMENT,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise SpecializedAuditError(
            f"cannot read official Git snapshot {root}: {exc}"
        ) from exc


def _git_ignored_entries(root: Path, role: str) -> tuple[str, ...]:
    raw = _git_readonly_bytes(
        root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
    )
    if raw and not raw.endswith(b"\0"):
        raise SpecializedAuditError(
            f"{role} ignored-file inventory is not NUL terminated"
        )
    entries: list[str] = []
    observed: set[str] = set()
    for field in raw[:-1].split(b"\0") if raw else []:
        try:
            relative = field.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise SpecializedAuditError(
                f"{role} ignored-file path is not strict UTF-8"
            ) from exc
        relative = _safe_relative_path(relative, f"{role} ignored-file path")
        if relative in observed:
            raise SpecializedAuditError(
                f"{role} ignored-file inventory contains duplicates"
            )
        observed.add(relative)
        entries.append(relative)
    return tuple(entries)


def _verify_git_checkout_policy(root: Path, role: str) -> None:
    expected_git_dir = root / ".git"
    opened_git_dir, git_dir_fd = _open_root(expected_git_dir)
    os.close(git_dir_fd)
    if opened_git_dir != expected_git_dir:
        raise SpecializedAuditError(f"{role} Git directory is not canonical")
    local_config_keys = _git_readonly(
        root, "config", "--local", "--no-includes", "--name-only", "--list"
    ).splitlines()
    forbidden_config = [
        key
        for key in local_config_keys
        if key.lower().startswith(("filter.", "include.", "includeif."))
        or key.lower() == "core.attributesfile"
    ]
    if forbidden_config:
        raise SpecializedAuditError(
            f"{role} repository-local filters/includes are forbidden: "
            f"{forbidden_config[:5]!r}"
        )
    opened_info_dir, info_fd = _open_root(expected_git_dir / "info")
    if opened_info_dir != expected_git_dir / "info":
        os.close(info_fd)
        raise SpecializedAuditError(f"{role} Git info directory is not canonical")
    try:
        try:
            os.stat("attributes", dir_fd=info_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise SpecializedAuditError(
                f"{role} .git/info/attributes is forbidden"
            )
    finally:
        os.close(info_fd)
    if _git_readonly(root, "rev-parse", "--show-toplevel").strip() != str(root):
        raise SpecializedAuditError(
            f"{role} Git top-level differs from its bound root"
        )
    absolute_git_dir = _git_readonly(
        root, "rev-parse", "--absolute-git-dir"
    ).strip()
    common_git_dir = _git_readonly(
        root, "rev-parse", "--path-format=absolute", "--git-common-dir"
    ).strip()
    if absolute_git_dir != str(expected_git_dir) or common_git_dir != str(
        expected_git_dir
    ):
        raise SpecializedAuditError(
            f"{role} linked worktrees or external Git directories are forbidden"
        )
    if _git_readonly(root, "rev-parse", "--show-object-format").strip() != "sha1":
        raise SpecializedAuditError(f"{role} Git object format must be sha1")
    replacements = _git_readonly(
        root, "for-each-ref", "--format=%(refname)", "refs/replace/"
    ).splitlines()
    if replacements:
        raise SpecializedAuditError(
            f"{role} Git replace refs are forbidden: {replacements[:5]!r}"
        )
    ignored = _git_ignored_entries(root, role)
    if ignored:
        raise SpecializedAuditError(
            f"{role} source root contains gitignored artifacts: {ignored[:5]!r}"
        )


def _verify_exact_commit_tree(root: Path, expected_commit: str, role: str) -> None:
    raw_tree = _git_readonly_bytes(
        root, "ls-tree", "-r", "-z", "--full-tree", expected_commit
    )
    if raw_tree and not raw_tree.endswith(b"\0"):
        raise SpecializedAuditError(
            f"{role} Git tree inventory is not NUL terminated"
        )
    observed_paths: set[str] = set()
    for record in raw_tree[:-1].split(b"\0") if raw_tree else []:
        try:
            metadata, path_raw = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.split(b" ", 2)
            relative = path_raw.decode("utf-8", errors="strict")
            object_id_text = object_id.decode("ascii", errors="strict")
        except (ValueError, UnicodeError) as exc:
            raise SpecializedAuditError(
                f"{role} Git tree record is malformed"
            ) from exc
        relative = _safe_relative_path(relative, f"{role} tracked source path")
        if relative in observed_paths:
            raise SpecializedAuditError(f"{role} Git tree contains duplicate paths")
        observed_paths.add(relative)
        if (
            mode not in (b"100644", b"100755")
            or object_type != b"blob"
            or not COMMIT_RE.fullmatch(object_id_text)
        ):
            raise SpecializedAuditError(
                f"{role} symlink, gitlink, or non-regular tracked source is forbidden: "
                f"{relative}"
            )
        source_metadata: dict[str, int] = {}
        worktree, _digest, _size = _read_relative(
            root, relative, minimum_bytes=0, metadata=source_metadata
        )
        observed_object_id = hashlib.sha1(
            f"blob {len(worktree)}\0".encode("ascii") + worktree
        ).hexdigest()
        executable = bool(source_metadata["mode"] & 0o111)
        if observed_object_id != object_id_text or executable != (mode == b"100755"):
            raise SpecializedAuditError(
                f"{role} tracked worktree differs from exact commit tree: {relative}"
            )
    if not observed_paths:
        raise SpecializedAuditError(f"{role} exact commit tree is empty")


def _materialize_config_snapshot(
    source_root: Path, commit: str, destination: Path
) -> dict[str, Any]:
    """Extract only regular configs/ members from an exact Git commit archive."""

    archive = _git_readonly_bytes(
        source_root, "archive", "--format=tar", commit, "configs/"
    )
    inventory: list[dict[str, Any]] = []
    observed: set[str] = set()
    try:
        tar = tarfile.open(fileobj=io.BytesIO(archive), mode="r:")
    except tarfile.TarError as exc:
        raise SpecializedAuditError(f"cannot parse official config snapshot: {exc}") from exc
    with tar:
        for member in tar.getmembers():
            pure = PurePosixPath(member.name)
            if (
                pure.is_absolute()
                or not pure.parts
                or pure.parts[0] != "configs"
                or any(part in ("", ".", "..") for part in pure.parts)
                or pure.as_posix() in observed
                or not (member.isdir() or member.isreg())
                or member.issym()
                or member.islnk()
            ):
                raise SpecializedAuditError(
                    f"unsafe member in official config snapshot: {member.name!r}"
                )
            observed.add(pure.as_posix())
            target = destination.joinpath(*pure.parts)
            if member.isdir():
                target.mkdir(mode=0o700, parents=True, exist_ok=False)
                continue
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            stream = tar.extractfile(member)
            if stream is None:
                raise SpecializedAuditError(
                    f"cannot read official config snapshot member: {member.name}"
                )
            raw = stream.read()
            if len(raw) != member.size:
                raise SpecializedAuditError(
                    f"short official config snapshot member: {member.name}"
                )
            if b"${oc.env:" in raw or b"${env:" in raw:
                raise SpecializedAuditError(
                    f"ambient environment interpolation is forbidden: {member.name}"
                )
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            try:
                view = memoryview(raw)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise SpecializedAuditError(
                            f"short config snapshot write: {member.name}"
                        )
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            inventory.append(
                {
                    "path": pure.as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw),
                }
            )
    if not inventory or not (destination / "configs/sim_libero.yaml").is_file():
        raise SpecializedAuditError("official config snapshot is incomplete")
    return {
        "commit": commit,
        "file_count": len(inventory),
        "tree_sha256": _tree_sha256(inventory),
        "files": inventory,
    }


def _verified_source(
    root: Path,
    expected_commit: str,
    *,
    role: str,
    critical_paths: Sequence[str],
) -> dict[str, Any]:
    """Bind a clean checkout and prove critical bytes equal their HEAD blobs."""

    root = _canonical_absolute_directory(root, f"{role} root")
    _verify_git_checkout_policy(root, role)
    expected_commit = _require_commit(expected_commit, "official expected commit")
    top_level = _git_readonly(root, "rev-parse", "--show-toplevel").strip()
    head = _git_readonly(root, "rev-parse", "HEAD").strip()
    tree = _git_readonly(root, "rev-parse", "HEAD^{tree}").strip()
    dirty = _git_readonly(root, "status", "--porcelain", "--untracked-files=all")
    if top_level != str(root):
        raise SpecializedAuditError(f"{role} Git top-level differs from its bound root")
    if head != expected_commit:
        raise SpecializedAuditError(
            f"{role} commit mismatch: expected {expected_commit}, observed {head}"
        )
    _require_commit(tree, f"{role} tree")
    if dirty:
        raise SpecializedAuditError(f"{role} checkout must be exactly clean")
    index_rows = _git_readonly(root, "ls-files", "-v", "-z").split("\0")
    if any(row and not row.startswith("H ") for row in index_rows):
        raise SpecializedAuditError(
            f"{role} index contains assume-unchanged or skip-worktree entries"
        )
    _verify_exact_commit_tree(root, expected_commit, role)
    critical_files: list[dict[str, Any]] = []
    for relative in critical_paths:
        raw, digest, size = _read_relative(root, relative)
        git_blob = _git_readonly(root, "rev-parse", f"HEAD:{relative}").strip()
        _require_commit(git_blob, f"{role} critical Git blob")
        observed_git_blob = hashlib.sha1(
            f"blob {len(raw)}\0".encode("ascii") + raw
        ).hexdigest()
        if (
            _git_readonly(root, "cat-file", "-t", git_blob).strip() != "blob"
            or observed_git_blob != git_blob
        ):
            raise SpecializedAuditError(
                f"{role} critical file differs from its HEAD blob: {relative}"
            )
        critical_files.append(
            {
                "path": relative,
                "git_blob": git_blob,
                "sha256": digest,
                "size_bytes": size,
                "git_blob_content_sha256": digest,
            }
        )
    _verify_git_checkout_policy(root, role)
    _verify_exact_commit_tree(root, expected_commit, role)
    if (
        _git_readonly(root, "rev-parse", "--show-toplevel").strip() != top_level
        or
        _git_readonly(root, "rev-parse", "HEAD").strip() != head
        or _git_readonly(root, "rev-parse", "HEAD^{tree}").strip() != tree
        or _git_readonly(root, "status", "--porcelain", "--untracked-files=all")
        or any(
            row and not row.startswith("H ")
            for row in _git_readonly(root, "ls-files", "-v", "-z").split("\0")
        )
    ):
        raise SpecializedAuditError(f"{role} identity changed during readback")
    return {
        "status": "PASS",
        "role": role,
        "root": str(root),
        "commit": head,
        "tree": tree,
        "clean": True,
        "critical_files": critical_files,
        "critical_file_inventory_sha256": _canonical_sha256(critical_files),
    }


def _verified_official_source(root: Path, expected_commit: str) -> dict[str, Any]:
    return _verified_source(
        root,
        expected_commit,
        role="official_policy_and_evaluator_source",
        critical_paths=_OFFICIAL_CRITICAL_PATHS,
    )


def _verified_instrumentation_source(
    root: Path, expected_commit: str
) -> dict[str, Any]:
    return _verified_source(
        root,
        expected_commit,
        role="external_observer_and_launcher_source",
        critical_paths=_INSTRUMENTATION_CRITICAL_PATHS,
    )


def _formal_command_overrides(
    command_argv: Any,
    *,
    process_id: str,
    suite: str,
    task_id: int,
    gpu_id: int,
    scheduled_process: Mapping[str, Any],
    artifact_root: Path,
    approved: Mapping[str, Any],
    instrumentation_root: Path,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    if (
        not isinstance(command_argv, list)
        or len(command_argv) != len(FORMAL_HYDRA_OVERRIDE_KEYS) + 2
        or any(not isinstance(item, str) or not item for item in command_argv)
    ):
        raise SpecializedAuditError(f"formal manager command is invalid: {process_id}")
    python_path = Path(command_argv[0])
    if not python_path.is_absolute():
        raise SpecializedAuditError(f"formal Python command is not absolute: {process_id}")
    lexical_python = Path(os.path.abspath(os.fspath(python_path.expanduser())))
    try:
        resolved_python = lexical_python.resolve(strict=True)
    except OSError as exc:
        raise SpecializedAuditError(
            f"formal Python command cannot be resolved: {process_id}"
        ) from exc
    python_name = PurePosixPath(resolved_python).name
    if not re.fullmatch(r"python(?:3(?:\.\d+)?)?", python_name):
        raise SpecializedAuditError(f"formal Python command is invalid: {process_id}")
    if str(resolved_python) != command_argv[0] or not os.access(resolved_python, os.X_OK):
        raise SpecializedAuditError(f"formal Python command is noncanonical: {process_id}")
    python_sha, python_size = _hash_absolute(resolved_python)
    expected_runner = instrumentation_root / "scripts/run_mf_wam_g0_traced.py"
    runner = Path(command_argv[1])
    if (
        not runner.is_absolute()
        or Path(os.path.abspath(os.fspath(runner.expanduser()))) != expected_runner
        or command_argv[1] != str(expected_runner)
    ):
        raise SpecializedAuditError(f"formal traced runner is invalid: {process_id}")
    arguments = command_argv[2:]
    overrides: dict[str, str] = {}
    for argument in arguments:
        if (
            "=" not in argument
            or argument.startswith("-")
            or any(character in argument for character in ("\x00", "\r", "\n"))
        ):
            raise SpecializedAuditError(f"formal Hydra override is invalid: {process_id}")
        key, value = argument.split("=", 1)
        if not key or not value or key in overrides:
            raise SpecializedAuditError(
                f"formal Hydra override is empty or duplicated: {process_id}"
            )
        overrides[key] = value
    if set(overrides) != FORMAL_HYDRA_OVERRIDE_KEYS:
        raise SpecializedAuditError(f"formal Hydra override scope is invalid: {process_id}")
    checkpoint = approved.get("checkpoint")
    dataset_stats = approved.get("dataset_stats")
    if not isinstance(checkpoint, Mapping) or not isinstance(dataset_stats, Mapping):
        raise SpecializedAuditError("approved launch artifacts are incomplete")
    expected = {
        "task": "libero_uncond_2cam224_1e-4",
        "ckpt": str(checkpoint["path"]),
        "gpu_id": str(gpu_id),
        "seed": str(scheduled_process["global_seed"]),
        "output_dir": str(artifact_root),
        "EVALUATION.task_suite_name": suite,
        "EVALUATION.task_id": str(task_id),
        "EVALUATION.output_dir": str(artifact_root),
        "EVALUATION.dataset_stats_path": str(dataset_stats["path"]),
        "EVALUATION.num_trials": "50",
        "EVALUATION.env_num": "1",
        "EVALUATION.num_steps_wait": "30",
        "EVALUATION.replan_steps": "10",
        "EVALUATION.binarize_gripper": "true",
        "EVALUATION.use_action_ensembler": "false",
        "EVALUATION.visualize_future_video": "false",
        "EVALUATION.action_horizon": "32",
    }
    if overrides != expected:
        raise SpecializedAuditError(f"formal Hydra override values are invalid: {process_id}")
    return arguments, overrides, {
        "path": str(resolved_python),
        "sha256": python_sha,
        "size_bytes": python_size,
    }


def _manager_launch_evidence(
    *,
    anchor: Mapping[str, Any],
    contract: Mapping[str, Any],
    approved: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-read the anchor-bound manager manifest and all 40 status commands."""

    raw_log_text = anchor.get("raw_log_root")
    artifact_text = anchor.get("artifact_root")
    if not isinstance(raw_log_text, str) or not isinstance(artifact_text, str):
        raise SpecializedAuditError("run anchor launch roots are invalid")
    raw_log_root = _canonical_absolute_directory(Path(raw_log_text), "raw log root")
    artifact_root = Path(os.path.abspath(os.fspath(Path(artifact_text).expanduser())))
    if str(raw_log_root) != raw_log_text or str(artifact_root) != artifact_text:
        raise SpecializedAuditError("run anchor launch roots are noncanonical")
    try:
        raw_log_root.relative_to(artifact_root)
    except ValueError:
        pass
    else:
        raise SpecializedAuditError("raw log root must be outside artifact root")
    try:
        artifact_root.relative_to(raw_log_root)
    except ValueError:
        pass
    else:
        raise SpecializedAuditError("artifact root must be outside raw log root")
    manifest_path = raw_log_root / "manager_terminal.json"
    manifest, manifest_sha, _ = _load_absolute_json(manifest_path)
    if manifest_sha != _require_sha256(
        anchor.get("manager_manifest_sha256"), "run anchor manager manifest SHA"
    ):
        raise SpecializedAuditError("manager manifest differs from the run anchor")
    manifest = _require_exact_keys(manifest, MANAGER_TOP_KEYS, "manager manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "mf_wam_g0_manager_terminal_manifest"
        or manifest.get("run_id") != anchor.get("run_id")
        or manifest.get("manager_exit_code") != 0
        or manifest.get("artifact_root") != artifact_text
        or manifest.get("raw_log_root") != raw_log_text
        or manifest.get("canonical_input_file_count") != 2080
    ):
        raise SpecializedAuditError("manager launch manifest identity is invalid")
    _require_sha256(
        manifest.get("canonical_input_tree_sha256"),
        "manager canonical input tree SHA",
    )
    preregistration = contract.get("preregistration")
    schedule = contract.get("seed_schedule")
    terminal = contract.get("terminal")
    digests = contract.get("digests")
    if not all(
        isinstance(item, Mapping)
        for item in (preregistration, schedule, terminal, digests)
    ):
        raise SpecializedAuditError("validated contract launch documents are incomplete")
    upstream = manifest.get("upstream_bindings")
    expected_upstream = {
        "preregistration_file_sha256": digests["preregistration_file_sha256"],
        "runtime_start_file_sha256": digests["runtime_start_file_sha256"],
        "seed_schedule_file_sha256": digests["seed_schedule_file_sha256"],
        "resolved_config_sha256": anchor["resolved_config_sha256"],
        "official_commit": anchor["fastwam_commit"],
        "instrumentation_commit": anchor["instrumentation_commit"],
        "python_hash_seed": schedule["python_hash_seed"],
    }
    if not isinstance(upstream, Mapping) or dict(upstream) != expected_upstream:
        raise SpecializedAuditError("manager launch upstream bindings are invalid")
    launch = preregistration.get("launch")
    runtime_environment = preregistration.get("runtime_environment")
    if not isinstance(launch, Mapping) or not isinstance(runtime_environment, Mapping):
        raise SpecializedAuditError("preregistered launch environment is incomplete")
    instrumentation_text = launch.get("working_directory")
    if not isinstance(instrumentation_text, str):
        raise SpecializedAuditError("preregistered instrumentation root is invalid")
    instrumentation_root = _canonical_absolute_directory(
        Path(instrumentation_text), "preregistered instrumentation root"
    )
    if str(instrumentation_root) != instrumentation_text:
        raise SpecializedAuditError("preregistered instrumentation root is noncanonical")
    scheduled = schedule.get("task_processes")
    if not isinstance(scheduled, list) or len(scheduled) != 40:
        raise SpecializedAuditError("specialized config gate requires exactly 40 scheduled tasks")
    processes = manifest.get("task_processes")
    gpu_ids = manifest.get("gpu_ids")
    if (
        not isinstance(processes, list)
        or len(processes) != 40
        or not isinstance(gpu_ids, list)
        or gpu_ids != sorted(set(gpu_ids))
        or any(type(item) is not int or not 0 <= item <= 7 for item in gpu_ids)
    ):
        raise SpecializedAuditError("manager launch process/GPU scope is invalid")
    commands: list[dict[str, Any]] = []
    status_inventory: list[dict[str, Any]] = []
    official_root: str | None = None
    resolved_runtime_path: str | None = None
    upstream_path_bindings: dict[str, str] | None = None
    python_identity: dict[str, Any] | None = None
    for index, (raw_process, scheduled_process) in enumerate(zip(processes, scheduled)):
        if not isinstance(scheduled_process, Mapping):
            raise SpecializedAuditError("seed schedule process is invalid")
        item = _require_exact_keys(
            raw_process, MANAGER_PROCESS_KEYS, f"manager process {index}"
        )
        suite = scheduled_process.get("task_suite")
        task_id = scheduled_process.get("task_id")
        process_id = scheduled_process.get("process_id")
        gpu_id = item.get("gpu_id")
        if (
            suite not in SUITES
            or type(task_id) is not int
            or process_id != f"{suite}/task{task_id:02d}"
            or item.get("process_id") != process_id
            or item.get("task_suite") != suite
            or item.get("task_id") != task_id
            or type(gpu_id) is not int
            or gpu_id not in gpu_ids
            or item.get("state") != "SUCCEEDED"
            or item.get("exit_code") != 0
            or item.get("complete") is not True
            or item.get("failure_reason") is not None
            or item.get("episode_count") != 50
        ):
            raise SpecializedAuditError(f"manager process is incomplete: {process_id}")
        status_relative = _safe_relative_path(
            item.get("status_path"), f"manager status path {process_id}"
        )
        if status_relative != f"status/{suite}/task{task_id:02d}.json":
            raise SpecializedAuditError(f"manager status path is noncanonical: {process_id}")
        status, status_sha, status_size = _load_relative_json(
            raw_log_root, status_relative
        )
        if (
            status_sha != _require_sha256(item.get("status_sha256"), "status SHA")
            or status_size != item.get("status_size_bytes")
        ):
            raise SpecializedAuditError(f"manager status readback mismatch: {process_id}")
        status = _require_exact_keys(
            status, MANAGER_STATUS_KEYS, f"manager status {process_id}"
        )
        scalar_fields = (
            "process_id",
            "task_suite",
            "task_id",
            "gpu_id",
            "state",
            "exit_code",
            "complete",
            "failure_reason",
            "command_sha256",
            "environment_sha256",
        )
        if (
            status.get("schema_version") != 1
            or status.get("kind") != "mf_wam_g0_manager_task_status"
            or status.get("run_id") != anchor.get("run_id")
            or any(status.get(field) != item.get(field) for field in scalar_fields)
        ):
            raise SpecializedAuditError(f"manager status identity is invalid: {process_id}")
        command_argv = status.get("command_argv")
        if _canonical_sha256(command_argv) != item.get("command_sha256"):
            raise SpecializedAuditError(f"manager command hash is invalid: {process_id}")
        arguments, overrides, observed_python = _formal_command_overrides(
            command_argv,
            process_id=str(process_id),
            suite=str(suite),
            task_id=int(task_id),
            gpu_id=int(gpu_id),
            scheduled_process=scheduled_process,
            artifact_root=artifact_root,
            approved=approved,
            instrumentation_root=instrumentation_root,
        )
        if python_identity is None:
            python_identity = observed_python
        elif python_identity != observed_python:
            raise SpecializedAuditError(
                "formal Python executable changed across task processes"
            )
        environment = status.get("environment_bindings")
        if (
            not isinstance(environment, Mapping)
            or set(environment) != _FORMAL_ENVIRONMENT_KEYS
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in environment.items()
            )
            or _canonical_sha256(environment) != item.get("environment_sha256")
        ):
            raise SpecializedAuditError(f"manager environment binding is invalid: {process_id}")
        observed_upstream_paths = {
            "preregistration": environment["MF_WAM_G0_PREREG_PATH"],
            "runtime_start": environment["MF_WAM_G0_RUNTIME_START_PATH"],
            "seed_schedule": environment["MF_WAM_G0_SEED_SCHEDULE_PATH"],
            "resolved_config": environment["MF_WAM_G0_RESOLVED_CONFIG_PATH"],
        }
        if upstream_path_bindings is None:
            for name, path_text in observed_upstream_paths.items():
                path = Path(path_text)
                if (
                    not path.is_absolute()
                    or str(Path(os.path.abspath(os.fspath(path.expanduser()))))
                    != path_text
                ):
                    raise SpecializedAuditError(
                        f"manager {name} path is noncanonical: {process_id}"
                    )
                observed_sha, _ = _hash_absolute(path)
                expected_sha = {
                    "preregistration": digests["preregistration_file_sha256"],
                    "runtime_start": digests["runtime_start_file_sha256"],
                    "seed_schedule": digests["seed_schedule_file_sha256"],
                    "resolved_config": anchor["resolved_config_sha256"],
                }[name]
                if observed_sha != expected_sha:
                    raise SpecializedAuditError(
                        f"manager {name} file differs from its contract binding"
                    )
            upstream_path_bindings = observed_upstream_paths
        elif upstream_path_bindings != observed_upstream_paths:
            raise SpecializedAuditError(
                "manager upstream paths changed across task processes"
            )
        expected_environment = {
            **_FIXED_WORKER_ENVIRONMENT,
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "DIFFSYNTH_DOWNLOAD_SOURCE": runtime_environment[
                "DIFFSYNTH_DOWNLOAD_SOURCE"
            ],
            "DIFFSYNTH_MODEL_BASE_PATH": runtime_environment[
                "DIFFSYNTH_MODEL_BASE_PATH"
            ],
            "DIFFSYNTH_SKIP_DOWNLOAD": runtime_environment[
                "DIFFSYNTH_SKIP_DOWNLOAD"
            ],
            "LOCAL_RANK": "0",
            "MF_WAM_G0_PREREG_PATH": observed_upstream_paths["preregistration"],
            "MF_WAM_G0_PREREG_SHA256": digests[
                "preregistration_file_sha256"
            ],
            "MF_WAM_G0_RESOLVED_CONFIG_PATH": observed_upstream_paths[
                "resolved_config"
            ],
            "MF_WAM_G0_RESOLVED_CONFIG_SHA256": anchor[
                "resolved_config_sha256"
            ],
            "MF_WAM_G0_RUN_ID": anchor["run_id"],
            "MF_WAM_G0_RUNTIME_START_PATH": observed_upstream_paths[
                "runtime_start"
            ],
            "MF_WAM_G0_RUNTIME_START_SHA256": digests[
                "runtime_start_file_sha256"
            ],
            "MF_WAM_G0_SEED_SCHEDULE_PATH": observed_upstream_paths[
                "seed_schedule"
            ],
            "MF_WAM_G0_SEED_SCHEDULE_SHA256": digests[
                "seed_schedule_file_sha256"
            ],
            "MF_WAM_INSTRUMENTATION_COMMIT": anchor[
                "instrumentation_commit"
            ],
            "MF_WAM_OFFICIAL_COMMIT": anchor["fastwam_commit"],
            "MF_WAM_OFFICIAL_ROOT": environment["MF_WAM_OFFICIAL_ROOT"],
            "MUJOCO_GL": runtime_environment["MUJOCO_GL"],
            "PYOPENGL_PLATFORM": runtime_environment["PYOPENGL_PLATFORM"],
            "PYTHONHASHSEED": str(scheduled_process["python_hash_seed"]),
            "RANK": "0",
            "WORLD_SIZE": "1",
        }
        if dict(environment) != expected_environment:
            raise SpecializedAuditError(
                f"manager environment differs from formal contract: {process_id}"
            )
        observed_official_root = environment.get("MF_WAM_OFFICIAL_ROOT")
        observed_resolved_path = environment.get("MF_WAM_G0_RESOLVED_CONFIG_PATH")
        if not isinstance(observed_official_root, str) or not isinstance(
            observed_resolved_path, str
        ):
            raise SpecializedAuditError(f"manager launch paths are invalid: {process_id}")
        canonical_official_root = str(
            _canonical_absolute_directory(
                Path(observed_official_root), "manager official source root"
            )
        )
        if canonical_official_root != observed_official_root:
            raise SpecializedAuditError(f"manager official root is noncanonical: {process_id}")
        resolved_absolute = Path(
            os.path.abspath(os.fspath(Path(observed_resolved_path).expanduser()))
        )
        if str(resolved_absolute) != observed_resolved_path:
            raise SpecializedAuditError(f"manager resolved config path is noncanonical: {process_id}")
        if official_root is None:
            official_root = canonical_official_root
            resolved_runtime_path = observed_resolved_path
        elif (
            official_root != canonical_official_root
            or resolved_runtime_path != observed_resolved_path
        ):
            raise SpecializedAuditError("manager launch roots changed across task processes")
        commands.append(
            {
                "process_id": process_id,
                "task_suite": suite,
                "task_id": task_id,
                "gpu_id": gpu_id,
                "arguments": arguments,
                "overrides": overrides,
                "command_sha256": item["command_sha256"],
            }
        )
        status_inventory.append(
            {"path": status_relative, "sha256": status_sha, "size_bytes": status_size}
        )
    if official_root is None or resolved_runtime_path is None or python_identity is None:
        raise SpecializedAuditError("manager launch evidence is empty")
    runtime_resolved_sha, _ = _hash_absolute(Path(resolved_runtime_path))
    if runtime_resolved_sha != anchor.get("resolved_config_sha256"):
        raise SpecializedAuditError("runtime resolved config differs from the run anchor")
    return {
        "raw_log_root": str(raw_log_root),
        "manager_manifest_sha256": manifest_sha,
        "terminal_canonical_sha256": anchor["terminal_canonical_sha256"],
        "official_root": official_root,
        "instrumentation_root": str(instrumentation_root),
        "python_executable": python_identity,
        "resolved_config_runtime_path": resolved_runtime_path,
        "status_file_count": len(status_inventory),
        "status_tree_sha256": _tree_sha256(status_inventory),
        "commands": commands,
    }


def _compose_official_config(official_root: Path, arguments: Sequence[str]) -> Any:
    try:
        from hydra import compose, initialize_config_dir
    except (ImportError, ModuleNotFoundError) as exc:
        raise SpecializedAuditError(
            "hydra-core is required for specialized resolved-config audit"
        ) from exc
    try:
        with initialize_config_dir(
            version_base="1.3",
            config_dir=str(official_root / "configs"),
            job_name="mf_wam_g0_specialized_audit",
        ):
            return compose(config_name="sim_libero", overrides=list(arguments))
    except Exception as exc:
        raise SpecializedAuditError(
            f"specialized Hydra config composition failed: {exc}"
        ) from exc


def _config_runtime_versions() -> dict[str, str]:
    try:
        import hydra
        import omegaconf
    except (ImportError, ModuleNotFoundError) as exc:
        raise SpecializedAuditError(
            "hydra-core and omegaconf are required for specialized config audit"
        ) from exc
    return {
        "hydra": str(hydra.__version__),
        "omegaconf": str(omegaconf.__version__),
    }


def _normalize_resolved_config(value: Any) -> Mapping[str, Any]:
    try:
        from omegaconf import OmegaConf
    except (ImportError, ModuleNotFoundError) as exc:
        raise SpecializedAuditError(
            "omegaconf is required for specialized resolved-config audit"
        ) from exc
    try:
        normalized = OmegaConf.to_container(
            value,
            resolve=True,
            throw_on_missing=True,
            enum_to_str=True,
        )
    except Exception as exc:
        raise SpecializedAuditError(f"cannot normalize resolved Hydra config: {exc}") from exc
    if not isinstance(normalized, Mapping):
        raise SpecializedAuditError("resolved Hydra config must be an object")
    _assert_finite_tree(normalized, "resolved Hydra config")
    return normalized


def _load_locked_resolved_config(path: Path) -> tuple[Any, str, int]:
    raw, digest, size = _read_absolute(path)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SpecializedAuditError("locked resolved config is not UTF-8") from exc
    try:
        from omegaconf import OmegaConf
    except (ImportError, ModuleNotFoundError) as exc:
        raise SpecializedAuditError(
            "omegaconf is required for specialized resolved-config audit"
        ) from exc
    try:
        return OmegaConf.create(text), digest, size
    except Exception as exc:
        raise SpecializedAuditError(f"cannot parse locked resolved config: {exc}") from exc


def _static_config_projection(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    projected = _loads_json(_canonical_bytes(value), label)
    if not isinstance(projected, dict):
        raise SpecializedAuditError(f"{label} must be an object")
    for path in DYNAMIC_RUNTIME_OVERLAY_PATHS:
        node: Any = projected
        for component in path[:-1]:
            if not isinstance(node, dict) or component not in node:
                raise SpecializedAuditError(
                    f"{label} lacks dynamic overlay path: {'.'.join(path)}"
                )
            node = node[component]
        if not isinstance(node, dict) or path[-1] not in node:
            raise SpecializedAuditError(
                f"{label} lacks dynamic overlay path: {'.'.join(path)}"
            )
        del node[path[-1]]
    return projected


def _paired_static_config_projection(
    value: Mapping[str, Any], *, artifact_root: str, label: str
) -> dict[str, Any]:
    """Normalize only the two already-validated run-local output locations."""

    projected = _loads_json(_canonical_bytes(value), label)
    evaluation = projected.get("EVALUATION") if isinstance(projected, dict) else None
    if (
        not isinstance(evaluation, dict)
        or projected.get("output_dir") != artifact_root
        or evaluation.get("output_dir") != artifact_root
    ):
        raise SpecializedAuditError(
            f"{label} does not bind both fixed output directories to its run root"
        )
    projected["output_dir"] = _PAIRED_ARTIFACT_ROOT_SENTINEL
    evaluation["output_dir"] = _PAIRED_ARTIFACT_ROOT_SENTINEL
    return projected


def _dynamic_config_values(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        evaluation = value["EVALUATION"]
        if not isinstance(evaluation, Mapping):
            raise KeyError("EVALUATION")
        gpu_id = value["gpu_id"]
        suite = evaluation["task_suite_name"]
        task_id = evaluation["task_id"]
    except (KeyError, TypeError) as exc:
        raise SpecializedAuditError(f"{label} lacks the exact dynamic overlay") from exc
    if (
        type(gpu_id) is not int
        or not isinstance(suite, str)
        or suite not in SUITES
        or type(task_id) is not int
        or not 0 <= task_id < 10
    ):
        raise SpecializedAuditError(f"{label} dynamic overlay types/values are invalid")
    return {
        "gpu_id": gpu_id,
        "EVALUATION.task_suite_name": suite,
        "EVALUATION.task_id": task_id,
    }


def _audit_resolved_config_gate(
    *,
    anchor: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_dir: Path,
    approved: Mapping[str, Any],
    compose_config: Callable[[Path, Sequence[str]], Any] | None = None,
    load_locked_config: Callable[[Path], tuple[Any, str, int]] | None = None,
    normalize_config: Callable[[Any], Mapping[str, Any]] | None = None,
    runtime_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Third, independent launch/config gate before a scientific G0 verdict."""

    launch = _manager_launch_evidence(anchor=anchor, contract=contract, approved=approved)
    official_source = _verified_official_source(
        Path(launch["official_root"]), str(anchor["fastwam_commit"])
    )
    instrumentation_source = _verified_instrumentation_source(
        Path(launch["instrumentation_root"]), str(anchor["instrumentation_commit"])
    )
    locked_path = contract_dir / "resolved-config.yaml"
    loader = load_locked_config or _load_locked_resolved_config
    normalizer = normalize_config or _normalize_resolved_config
    composer = compose_config or _compose_official_config
    runtime_lock = contract["preregistration"].get("runtime_lock")
    if not isinstance(runtime_lock, Mapping):
        raise SpecializedAuditError("preregistered config runtime lock is absent")
    if runtime_versions is None and compose_config is not None:
        observed_versions = {
            "hydra": str(runtime_lock.get("hydra")),
            "omegaconf": str(runtime_lock.get("omegaconf")),
        }
    else:
        observed_versions = dict(runtime_versions or _config_runtime_versions())
    if (
        set(observed_versions) != {"hydra", "omegaconf"}
        or observed_versions["hydra"] != runtime_lock.get("hydra")
        or observed_versions["omegaconf"] != runtime_lock.get("omegaconf")
    ):
        raise SpecializedAuditError(
            "specialized Hydra/OmegaConf versions differ from the runtime lock"
        )
    locked_raw, locked_sha, locked_size = loader(locked_path)
    if locked_sha != anchor.get("resolved_config_sha256"):
        raise SpecializedAuditError("locked resolved config differs from the run anchor")
    locked = normalizer(locked_raw)
    _dynamic_config_values(locked, "locked resolved config")
    locked_static = _static_config_projection(locked, "locked resolved config")
    static_sha = _canonical_sha256(locked_static)
    paired_static = _paired_static_config_projection(
        locked_static,
        artifact_root=str(anchor["artifact_root"]),
        label="locked static config projection",
    )
    paired_static_sha = _canonical_sha256(paired_static)
    live_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mf-wam-g0-config-snapshot-") as directory:
        snapshot_root = Path(directory)
        snapshot = _materialize_config_snapshot(
            Path(launch["official_root"]), str(anchor["fastwam_commit"]), snapshot_root
        )
        for command in launch["commands"]:
            process_id = str(command["process_id"])
            live_raw = composer(snapshot_root, command["arguments"])
            live = normalizer(live_raw)
            dynamic = _dynamic_config_values(live, f"live config {process_id}")
            expected_dynamic = {
                "gpu_id": command["gpu_id"],
                "EVALUATION.task_suite_name": command["task_suite"],
                "EVALUATION.task_id": command["task_id"],
            }
            if dynamic != expected_dynamic:
                raise SpecializedAuditError(
                    f"live config dynamic overlay differs from the sealed command: {process_id}"
                )
            live_static = _static_config_projection(live, f"live config {process_id}")
            live_static_sha = _canonical_sha256(live_static)
            if live_static != locked_static or live_static_sha != static_sha:
                raise SpecializedAuditError(
                    f"live config static projection differs from locked resolved config: {process_id}"
                )
            live_rows.append(
                {
                    "process_id": process_id,
                    "command_sha256": command["command_sha256"],
                    "live_resolved_config_sha256": _canonical_sha256(live),
                    "static_projection_sha256": live_static_sha,
                }
            )
    if len(live_rows) != 40 or len({row["process_id"] for row in live_rows}) != 40:
        raise SpecializedAuditError("specialized resolved-config coverage is not exactly 40")
    terminal_official_source = _verified_official_source(
        Path(launch["official_root"]), str(anchor["fastwam_commit"])
    )
    terminal_instrumentation_source = _verified_instrumentation_source(
        Path(launch["instrumentation_root"]), str(anchor["instrumentation_commit"])
    )
    if (
        terminal_official_source != official_source
        or terminal_instrumentation_source != instrumentation_source
    ):
        raise SpecializedAuditError("source identity changed during Hydra composition")
    return {
        "run_id": anchor["run_id"],
        "manager_manifest_sha256": launch["manager_manifest_sha256"],
        "status_tree_sha256": launch["status_tree_sha256"],
        "status_file_count": launch["status_file_count"],
        "terminal_canonical_sha256": anchor["terminal_canonical_sha256"],
        "locked_resolved_config_sha256": locked_sha,
        "locked_resolved_config_size_bytes": locked_size,
        "static_projection_sha256": static_sha,
        "paired_static_projection_sha256": paired_static_sha,
        "process_count": len(live_rows),
        "process_config_evidence_sha256": _canonical_sha256(live_rows),
        "config_snapshot": snapshot,
        "config_runtime_versions": observed_versions,
        "python_executable": launch["python_executable"],
        "official_source": official_source,
        "instrumentation_source": instrumentation_source,
    }


def _tree_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["path"]).encode("utf-8")):
        sha = _require_sha256(row.get("sha256"), "tree row sha256")
        path = _safe_relative_path(row.get("path"), "tree row path")
        digest.update(f"{sha}  {path}\n".encode("utf-8"))
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecializedAuditError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SpecializedAuditError(f"{label} is non-finite")
    return result


def _numeric_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise SpecializedAuditError(f"{label} must have length {length}")
    return [_finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _numeric_matrix(value: Any, rows: int | None, columns: int, label: str) -> list[list[float]]:
    if not isinstance(value, list) or (rows is not None and len(value) != rows):
        raise SpecializedAuditError(f"{label} row count mismatch")
    return [_numeric_vector(row, columns, f"{label}[{index}]") for index, row in enumerate(value)]


def expected_paths(scope: AuditScope) -> dict[str, set[str]]:
    results: set[str] = set()
    receipts: set[str] = set()
    traces: set[str] = set()
    for suite in scope.suites:
        for task_id in range(scope.tasks_per_suite):
            results.add(f"results/{suite}/task{task_id:02d}.json")
            receipts.add(f"trace_receipts/{suite}/task{task_id:02d}.json")
            for trial_idx in range(scope.trials_per_task):
                traces.add(f"traces/{suite}/task{task_id:02d}/trial{trial_idx:03d}.json")
    return {"results": results, "trace_receipts": receipts, "traces": traces}


def _walk_relative_files(root: Path, relative: str) -> set[str]:
    directory = root / relative
    lexical, descriptor = _open_root(directory)
    os.close(descriptor)
    observed: set[str] = set()
    for current, directories, files in os.walk(lexical, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            child = current_path / name
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise SpecializedAuditError(f"non-directory or symlink in canonical tree: {child}")
        for name in files:
            child = current_path / name
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise SpecializedAuditError(f"non-regular canonical artifact: {child}")
            observed.add(child.relative_to(root).as_posix())
    return observed


def _validate_exact_layout(root: Path, scope: AuditScope) -> None:
    expected = expected_paths(scope)
    for directory, wanted in expected.items():
        observed = _walk_relative_files(root, directory)
        if observed != wanted:
            missing = sorted(wanted - observed)[:5]
            extra = sorted(observed - wanted)[:5]
            raise SpecializedAuditError(
                f"{root}/{directory} layout mismatch: missing={missing} extra={extra}"
            )


def _validate_inventory_readback(
    root: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    """Re-read a completed inventory so mid-audit substitutions fail closed."""

    for row in rows:
        digest, size = _hash_relative(root, str(row["path"]), maximum_bytes=128 << 20)
        if digest != row["sha256"] or size != row["size_bytes"]:
            raise SpecializedAuditError(
                f"artifact changed after audit read: {row['path']}"
            )


def _validate_trace(
    payload: Any,
    *,
    run_anchor: Mapping[str, Any],
    suite: str,
    task_id: int,
    trial_idx: int,
    scope: AuditScope,
    expected_seed_process: Mapping[str, Any] | None = None,
    contract_file_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    trace = _require_exact_keys(payload, TRACE_TOP_KEYS, "trace")
    if trace["schema_version"] != 2 or trace["kind"] != "mf_wam_g0_structured_trace":
        raise SpecializedAuditError("trace must use canonical schema v2")
    metadata = _require_exact_keys(trace["metadata"], TRACE_METADATA_KEYS, "trace.metadata")
    fixed = {
        "run_id": run_anchor["run_id"],
        "task_suite": suite,
        "task_id": task_id,
        "trial_idx": trial_idx,
        "initial_state_index": trial_idx,
        "warmup_steps": scope.warmup_steps,
        "first_replan_env_step": scope.warmup_steps,
        "replan_steps": scope.replan_steps,
        "action_horizon": scope.action_horizon,
        "action_dimension": scope.action_dimension,
        "state_dimension": scope.state_dimension,
    }
    for key, expected in fixed.items():
        if metadata.get(key) != expected:
            raise SpecializedAuditError(f"trace metadata {key} mismatch")
    _require_sha256(metadata.get("initial_state_sha256"), "initial_state_sha256")
    _require_sha256(
        metadata.get("official_module_origin_inventory_sha256"),
        "official_module_origin_inventory_sha256",
    )
    if not isinstance(metadata.get("task_description"), str) or not metadata[
        "task_description"
    ].strip():
        raise SpecializedAuditError("trace task_description is absent")
    if (
        type(metadata.get("observer_rng_unchanged_checks")) is not int
        or metadata["observer_rng_unchanged_checks"] < 1
    ):
        raise SpecializedAuditError("trace observer RNG checks are absent")
    if type(metadata.get("success")) is not bool:
        raise SpecializedAuditError("trace success must be boolean")
    upstream = metadata.get("upstream_digests")
    expected_upstream_keys = {
        "preregistration_file_sha256",
        "preregistration_canonical_sha256",
        "runtime_start_file_sha256",
        "runtime_start_canonical_sha256",
        "seed_schedule_file_sha256",
        "seed_schedule_canonical_sha256",
        "resolved_config_sha256",
    }
    if not isinstance(upstream, Mapping) or set(upstream) != expected_upstream_keys:
        raise SpecializedAuditError("trace upstream digests are absent")
    for key, value in upstream.items():
        _require_sha256(value, f"trace upstream {key}")
    for key, expected in (
        ("preregistration_canonical_sha256", run_anchor["preregistration_canonical_sha256"]),
        ("runtime_start_canonical_sha256", run_anchor["runtime_start_canonical_sha256"]),
        ("seed_schedule_canonical_sha256", run_anchor["seed_schedule_canonical_sha256"]),
        ("resolved_config_sha256", run_anchor["resolved_config_sha256"]),
    ):
        if upstream.get(key) != expected:
            raise SpecializedAuditError(f"trace upstream digest mismatch: {key}")
    if contract_file_digests is not None:
        expected_file_digests = {
            "preregistration_file_sha256": contract_file_digests[
                "preregistration_file_sha256"
            ],
            "runtime_start_file_sha256": contract_file_digests[
                "runtime_start_file_sha256"
            ],
            "seed_schedule_file_sha256": contract_file_digests[
                "seed_schedule_file_sha256"
            ],
        }
        for key, expected in expected_file_digests.items():
            if upstream.get(key) != expected:
                raise SpecializedAuditError(f"trace upstream file digest mismatch: {key}")
    process = metadata.get("seed_schedule_process")
    process_keys = {
        "process_id",
        "task_suite",
        "task_id",
        "global_rank",
        "global_seed",
        "environment_seed",
        "environment_seed_scope",
        "policy_seed",
        "policy_seed_scope",
        "python_hash_seed",
        "trial_order",
        "initial_state_index_rule",
    }
    if not isinstance(process, Mapping) or set(process) != process_keys:
        raise SpecializedAuditError("trace seed process is absent")
    if (
        process.get("process_id") != f"{suite}/task{task_id:02d}"
        or process.get("task_suite") != suite
        or process.get("task_id") != task_id
        or process.get("global_rank") != 0
        or process.get("trial_order") != list(range(scope.trials_per_task))
        or process.get("initial_state_index_rule") != "trial_idx"
    ):
        raise SpecializedAuditError("trace seed process does not bind the formal task")
    if expected_seed_process is not None and dict(process) != dict(expected_seed_process):
        raise SpecializedAuditError("trace seed process differs from bound seed-schedule.json")
    seed_contract = _require_exact_keys(
        metadata.get("seed_contract"), TRACE_SEED_CONTRACT_KEYS, "trace.seed_contract"
    )
    expected_seed_contract = {
        "task_seed": process["global_seed"],
        "effective_global_rank": 0,
        "effective_process_seed": process["global_seed"],
        "task_seed_scope": (
            "once_per_task_process_before_model_and_benchmark_construction"
        ),
        "environment_seed": process["environment_seed"],
        "environment_seed_scope": "once_per_task_process_before_trial_loop",
        "policy_seed": process["policy_seed"],
        "policy_seed_scope": "fresh_generator_per_replan",
        "episode_rng_position": (
            "ordered_trial_index_in_shared_task_environment_stream"
        ),
    }
    if dict(seed_contract) != expected_seed_contract:
        raise SpecializedAuditError("trace seed_contract differs from bound schedule semantics")

    def source_pair_identity(name: str, expected_commit: str) -> dict[str, Any]:
        identity = metadata.get(name)
        if (
            not isinstance(identity, Mapping)
            or identity.get("commit") != expected_commit
            or identity.get("clean") is not True
        ):
            raise SpecializedAuditError(f"trace {name} is not clean/exact")
        normalized: dict[str, Any] = {
            "commit": expected_commit,
            "clean": True,
        }
        if scope.is_formal:
            expected_role, expected_paths = {
                "official_source": (
                    "official_policy_and_evaluator_source",
                    _OFFICIAL_CRITICAL_PATHS,
                ),
                "instrumentation_source": (
                    "external_observer_and_launcher_source",
                    _INSTRUMENTATION_CRITICAL_PATHS,
                ),
            }[name]
            required = {
                "status",
                "role",
                "root",
                "commit",
                "tree",
                "clean",
                "critical_files",
                "critical_file_inventory_sha256",
            }
            if (
                set(identity) != required
                or identity.get("status") != "PASS"
                or identity.get("role") != expected_role
                or not isinstance(identity.get("root"), str)
                or not PurePosixPath(identity["root"]).is_absolute()
            ):
                raise SpecializedAuditError(f"trace {name} formal identity is incomplete")
            _require_commit(identity.get("tree"), f"trace {name}.tree")
            _require_sha256(
                identity.get("critical_file_inventory_sha256"),
                f"trace {name}.critical_file_inventory_sha256",
            )
            critical_files = identity.get("critical_files")
            if not isinstance(critical_files, list) or len(critical_files) != len(
                expected_paths
            ):
                raise SpecializedAuditError(f"trace {name} critical files are absent")
            normalized_files: list[dict[str, Any]] = []
            for index, (file_identity, expected_path) in enumerate(
                zip(critical_files, expected_paths)
            ):
                item = _require_exact_keys(
                    file_identity,
                    {
                        "path",
                        "git_blob",
                        "sha256",
                        "size_bytes",
                        "git_blob_content_sha256",
                    },
                    f"trace {name}.critical_files[{index}]",
                )
                if item["path"] != expected_path:
                    raise SpecializedAuditError(
                        f"trace {name} critical source path differs from allowlist"
                    )
                _require_commit(item["git_blob"], f"trace {name} critical git blob")
                sha = _require_sha256(
                    item["sha256"], f"trace {name} critical file SHA"
                )
                git_content_sha = _require_sha256(
                    item["git_blob_content_sha256"],
                    f"trace {name} critical Git-content SHA",
                )
                if (
                    type(item["size_bytes"]) is not int
                    or item["size_bytes"] < 1
                    or sha != git_content_sha
                ):
                    raise SpecializedAuditError(
                        f"trace {name} critical file identity is inconsistent"
                    )
                normalized_files.append(dict(item))
            if _canonical_sha256(normalized_files) != identity[
                "critical_file_inventory_sha256"
            ]:
                raise SpecializedAuditError(
                    f"trace {name} critical file inventory digest mismatch"
                )
            normalized.update(
                {
                    "status": identity["status"],
                    "role": identity["role"],
                    "root": identity["root"],
                    "tree": identity["tree"],
                    "critical_files": normalized_files,
                    "critical_file_inventory_sha256": identity[
                        "critical_file_inventory_sha256"
                    ],
                }
            )
        return normalized

    official_pair_identity = source_pair_identity(
        "official_source", str(run_anchor["fastwam_commit"])
    )
    instrumentation_pair_identity = source_pair_identity(
        "instrumentation_source", str(run_anchor["instrumentation_commit"])
    )
    records = trace["records"]
    if not isinstance(records, list) or len(records) < scope.minimum_records_per_episode:
        raise SpecializedAuditError("trace record count is below the locked minimum")
    if metadata.get("record_count") != len(records):
        raise SpecializedAuditError("trace metadata record_count mismatch")
    previous_post_state: list[float] | None = None
    previous_post_observation_sha256: str | None = None
    executed_total = 0
    all_done_values: list[bool] = []
    for replan_idx, raw_record in enumerate(records):
        record = _require_exact_keys(raw_record, TRACE_RECORD_KEYS, f"trace.records[{replan_idx}]")
        if "raw_action_chunk" in record:
            raise SpecializedAuditError("legacy raw_action_chunk is forbidden")
        if (
            record.get("episode_idx") != trial_idx
            or record.get("replan_idx") != replan_idx
            or record.get("env_step") != scope.warmup_steps + replan_idx * scope.replan_steps
            or record.get("policy_seed_scope") != "fresh_generator_per_replan"
            or type(record.get("executed_count")) is not int
            or type(record.get("done_after_execution")) is not bool
        ):
            raise SpecializedAuditError("trace record identity/cadence mismatch")
        state = _numeric_vector(record.get("state"), scope.state_dimension, "record.state")
        pre_state = _numeric_vector(
            record.get("pre_state"), scope.state_dimension, "record.pre_state"
        )
        if state != pre_state:
            raise SpecializedAuditError("record state and pre_state differ")
        pre_observation_sha256 = _require_sha256(
            record.get("pre_observation_sha256"), "pre_observation_sha256"
        )
        if replan_idx and (
            pre_state != previous_post_state
            or pre_observation_sha256 != previous_post_observation_sha256
        ):
            raise SpecializedAuditError("trace state/observation continuity mismatch")
        raw_proposal = _numeric_matrix(
            record.get("proposed_raw_action_chunk"),
            scope.action_horizon,
            scope.action_dimension,
            "proposed_raw_action_chunk",
        )
        proposal = _numeric_matrix(
            record.get("proposed_env_action_chunk"),
            scope.action_horizon,
            scope.action_dimension,
            "proposed_env_action_chunk",
        )
        for action_index, (raw_action, env_action) in enumerate(
            zip(raw_proposal, proposal)
        ):
            expected_gripper_source = -((raw_action[-1] * 2.0) - 1.0)
            expected_gripper = (
                1.0
                if expected_gripper_source > 0.0
                else -1.0
                if expected_gripper_source < 0.0
                else 0.0
            )
            expected_action = [*raw_action[:-1], expected_gripper]
            if env_action != expected_action:
                raise SpecializedAuditError(
                    "formal binarized raw-to-environment action transform mismatch "
                    f"at proposal row {action_index}"
                )
        executed = _numeric_matrix(
            record.get("executed_env_actions"),
            None,
            scope.action_dimension,
            "executed_env_actions",
        )
        if not 1 <= len(executed) <= scope.replan_steps or record["executed_count"] != len(executed):
            raise SpecializedAuditError("executed prefix length mismatch")
        if executed != proposal[: len(executed)]:
            raise SpecializedAuditError("executed actions are not the proposal prefix")
        if record.get("policy_seed") != process["policy_seed"]:
            raise SpecializedAuditError("record policy_seed differs from seed schedule")
        executions = record.get("executions")
        if not isinstance(executions, list) or len(executions) != len(executed):
            raise SpecializedAuditError("execution detail count mismatch")
        for index, raw_execution in enumerate(executions):
            execution = _require_exact_keys(raw_execution, TRACE_EXECUTION_KEYS, "trace execution")
            if type(execution.get("done")) is not bool:
                raise SpecializedAuditError("execution done must be boolean")
            if _numeric_vector(execution.get("action"), scope.action_dimension, "execution.action") != executed[index]:
                raise SpecializedAuditError("execution action differs from executed prefix")
            post_state = _numeric_vector(
                execution.get("post_state"), scope.state_dimension, "execution.post_state"
            )
            post_observation_sha256 = _require_sha256(
                execution.get("post_observation_sha256"), "post_observation_sha256"
            )
            if execution["done"] and not (
                replan_idx == len(records) - 1 and index == len(executions) - 1
            ):
                raise SpecializedAuditError("trace continued after an environment terminal step")
            previous_post_state = post_state
            previous_post_observation_sha256 = post_observation_sha256
            all_done_values.append(execution["done"])
        if record["done_after_execution"] != executions[-1]["done"]:
            raise SpecializedAuditError("done_after_execution differs from final execution")
        executed_total += len(executed)
        if replan_idx < len(records) - 1 and (
            len(executed) != scope.replan_steps or record["done_after_execution"]
        ):
            raise SpecializedAuditError("nonterminal replan did not execute the full prefix")
    success = metadata["success"]
    if metadata.get("environment_step_count") != scope.warmup_steps + executed_total:
        raise SpecializedAuditError("trace environment_step_count mismatch")
    if success and (
        not records[-1]["done_after_execution"] or not all_done_values[-1]
    ):
        raise SpecializedAuditError("successful trace lacks a terminal done")
    if not success and any(all_done_values):
        raise SpecializedAuditError("failed trace contains a terminal done")
    if not success and records[-1]["done_after_execution"]:
        raise SpecializedAuditError("failed trace claims done_after_execution")
    if scope.is_formal:
        maximum_records = _FORMAL_MAX_REPLAN_RECORDS[suite]
        if len(records) > maximum_records:
            raise SpecializedAuditError("trace exceeds the official suite horizon")
        if not success and (
            len(records) != maximum_records
            or any(record["executed_count"] != scope.replan_steps for record in records)
        ):
            raise SpecializedAuditError("failed trace does not exhaust the official horizon")
    return {
        "success": success,
        "initial_state_sha256": metadata["initial_state_sha256"],
        "task_description": metadata["task_description"],
        "seed_process": dict(process),
        "official_source": official_pair_identity,
        "instrumentation_source": instrumentation_pair_identity,
    }


def _validate_result(payload: Any, suite: str, task_id: int, outcomes: Sequence[bool], scope: AuditScope) -> None:
    if not isinstance(payload, Mapping):
        raise SpecializedAuditError("task result must be an object")
    successes = payload.get("success_episodes")
    failures = payload.get("failure_episodes")
    expected_successes = [index for index, success in enumerate(outcomes) if success]
    expected_failures = [index for index, success in enumerate(outcomes) if not success]
    if (
        payload.get("task_suite") != suite
        or payload.get("task_id") != task_id
        or not isinstance(payload.get("task_description"), str)
        or not payload["task_description"].strip()
        or payload.get("total_episodes") != scope.trials_per_task
        or type(payload.get("successes")) is not int
        or payload.get("successes") != len(expected_successes)
        or successes != expected_successes
        or failures != expected_failures
        or type(payload.get("gpu_id")) is not int
        or payload.get("gpu_id") < 0
    ):
        raise SpecializedAuditError(f"task result semantics mismatch: {suite}/task{task_id:02d}")
    expected_rate = len(expected_successes) / scope.trials_per_task
    if "success_rate" in payload and _finite_number(payload["success_rate"], "task result success_rate") != expected_rate:
        raise SpecializedAuditError("task result success_rate mismatch")
    if "duration" in payload:
        if _finite_number(payload["duration"], "task result duration") < 0:
            raise SpecializedAuditError("task result duration must be nonnegative")
    if "future_video_psnr_mean" in payload and payload["future_video_psnr_mean"] is not None:
        _finite_number(payload["future_video_psnr_mean"], "future video PSNR")


def _validate_task_receipt(
    payload: Any,
    *,
    root: Path,
    anchor: Mapping[str, Any],
    suite: str,
    task_id: int,
    result_row: Mapping[str, Any],
    trace_rows: Sequence[Mapping[str, Any]],
    scope: AuditScope,
    expected_seed_process: Mapping[str, Any] | None = None,
) -> None:
    receipt = _require_exact_keys(payload, TASK_RECEIPT_KEYS, "task receipt")
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "mf_wam_g0_task_trace_receipt"
        or receipt["run_id"] != anchor["run_id"]
        or receipt["process_id"] != f"{suite}/task{task_id:02d}"
        or receipt["task_suite"] != suite
        or receipt["task_id"] != task_id
        or receipt["execution_scope"] != "one-process-per-task"
        or (receipt["world_size"], receipt["global_rank"], receipt["local_rank"]) != (1, 0, 0)
        or receipt["episode_count"] != scope.trials_per_task
    ):
        raise SpecializedAuditError("task receipt identity/scope mismatch")
    bindings = _require_exact_keys(receipt["bindings"], TASK_RECEIPT_BINDING_KEYS, "receipt.bindings")
    expected_bindings = {
        "preregistration_canonical_sha256": anchor["preregistration_canonical_sha256"],
        "runtime_start_canonical_sha256": anchor["runtime_start_canonical_sha256"],
        "seed_schedule_canonical_sha256": anchor["seed_schedule_canonical_sha256"],
        "resolved_config_sha256": anchor["resolved_config_sha256"],
        "image_digest": anchor["image_digest"],
        "fastwam_commit": anchor["fastwam_commit"],
        "instrumentation_commit": anchor["instrumentation_commit"],
    }
    if dict(bindings) != expected_bindings:
        raise SpecializedAuditError("task receipt upstream bindings mismatch")
    seeds = _require_exact_keys(receipt["seeds"], TASK_RECEIPT_SEED_KEYS, "receipt.seeds")
    if seeds.get("trial_order") != list(range(scope.trials_per_task)) or seeds.get("initial_state_index_rule") != "trial_idx":
        raise SpecializedAuditError("task receipt seed schedule mismatch")
    if expected_seed_process is not None:
        expected_seeds = {
            key: expected_seed_process[key] for key in TASK_RECEIPT_SEED_KEYS
        }
        if dict(seeds) != expected_seeds:
            raise SpecializedAuditError(
                "task receipt seeds differ from bound seed-schedule.json"
            )
    official_result = _require_exact_keys(receipt["official_result"], {"path", "sha256", "size_bytes"}, "receipt.official_result")
    if dict(official_result) != dict(result_row):
        raise SpecializedAuditError("task receipt result binding mismatch")
    observed_traces = receipt["traces"]
    if not isinstance(observed_traces, list) or observed_traces != list(trace_rows):
        raise SpecializedAuditError("task receipt trace inventory mismatch")
    if receipt["tree_sha256"] != _tree_sha256(trace_rows):
        raise SpecializedAuditError("task receipt trace tree mismatch")
    del root


def _audit_summaries(
    root: Path,
    *,
    run_id: str,
    task_rows: Sequence[Mapping[str, Any]],
    scope: AuditScope,
) -> dict[str, Any]:
    by_task = {
        (str(row["task_suite"]), int(row["task_id"])): row for row in task_rows
    }
    summary, summary_sha, summary_size = _load_relative_json(root, "summary.json")
    if not isinstance(summary, Mapping) or summary.get("run_id") != run_id:
        raise SpecializedAuditError("summary.json run identity mismatch")
    suite_stats = summary.get("suite_stats")
    task_results = summary.get("task_results")
    overall = summary.get("overall")
    if not all(isinstance(item, Mapping) for item in (suite_stats, task_results, overall)):
        raise SpecializedAuditError("summary.json sections are incomplete")
    if set(suite_stats) != set(scope.suites):
        raise SpecializedAuditError("summary.json suite coverage mismatch")
    expected_task_keys = {
        f"{suite}_{task_id}"
        for suite in scope.suites
        for task_id in range(scope.tasks_per_suite)
    }
    if set(task_results) != expected_task_keys:
        raise SpecializedAuditError("summary.json task coverage mismatch")
    suite_rates: dict[str, float] = {}
    total_time = 0.0
    for suite in scope.suites:
        suite_rows = [by_task[(suite, task_id)] for task_id in range(scope.tasks_per_suite)]
        successes = sum(int(row["successes"]) for row in suite_rows)
        rate_percent = successes / (scope.tasks_per_suite * scope.trials_per_task) * 100
        suite_rates[suite] = rate_percent
        stats_payload = suite_stats[suite]
        if not isinstance(stats_payload, Mapping):
            raise SpecializedAuditError("summary.json suite stats entry is invalid")
        if (
            stats_payload.get("total_tasks") != scope.tasks_per_suite
            or stats_payload.get("total_trials") != scope.tasks_per_suite * scope.trials_per_task
            or stats_payload.get("total_successes") != successes
        ):
            raise SpecializedAuditError(f"summary.json suite totals mismatch: {suite}")
        suite_total_time = _finite_number(stats_payload.get("total_time"), "suite total_time")
        if suite_total_time < 0:
            raise SpecializedAuditError("summary suite time must be nonnegative")
        total_time += suite_total_time
        for task_id in range(scope.tasks_per_suite):
            row = by_task[(suite, task_id)]
            task = task_results[f"{suite}_{task_id}"]
            if not isinstance(task, Mapping):
                raise SpecializedAuditError("summary task entry is invalid")
            if (
                task.get("total_episodes") != scope.trials_per_task
                or task.get("successes") != row["successes"]
                or _finite_number(task.get("success_rate"), "task summary success_rate")
                != row["success_rate"] * 100
            ):
                raise SpecializedAuditError("summary task values differ from raw outcomes")
    expected_overall_percent = sum(suite_rates.values()) / len(scope.suites)
    if (
        _finite_number(overall.get("average_success_rate"), "overall success rate")
        != expected_overall_percent
        or _finite_number(overall.get("total_time"), "overall total time") != total_time
    ):
        raise SpecializedAuditError("summary.json overall values mismatch")

    task_csv_raw, task_csv_sha, task_csv_size = _read_relative(root, "task_success_rates.csv")
    try:
        task_csv_text = task_csv_raw.decode("utf-8", errors="strict").splitlines()
        task_csv_rows = list(csv.DictReader(task_csv_text))
    except (UnicodeError, csv.Error) as exc:
        raise SpecializedAuditError(f"invalid task_success_rates.csv: {exc}") from exc
    if len(task_csv_rows) != len(expected_task_keys):
        raise SpecializedAuditError("task_success_rates.csv row count mismatch")
    observed_tasks: set[str] = set()
    for row in task_csv_rows:
        task = row.get("Task")
        if task not in expected_task_keys or task in observed_tasks:
            raise SpecializedAuditError("task_success_rates.csv task identity mismatch")
        observed_tasks.add(str(task))
        suite, task_id_text = str(task).rsplit("_", 1)
        expected = f"{by_task[(suite, int(task_id_text))]['success_rate'] * 100:.2f}"
        if row.get("Success Rate (%)") != expected:
            raise SpecializedAuditError("task_success_rates.csv value mismatch")

    summary_csv_raw, summary_csv_sha, summary_csv_size = _read_relative(root, "summary.csv")
    try:
        lines = summary_csv_raw.decode("utf-8", errors="strict").splitlines()
        if len(lines) < 3:
            raise SpecializedAuditError("summary.csv is truncated")
        table = list(csv.reader(lines[1:]))
    except (UnicodeError, csv.Error) as exc:
        raise SpecializedAuditError(f"invalid summary.csv: {exc}") from exc
    header = table[0]
    expected_columns = ["", *scope.suites, "Overall"]
    if header != expected_columns:
        raise SpecializedAuditError("summary.csv suite columns/order mismatch")
    success_rows = [row for row in table[1:] if row and row[0] == "Success Rate (%)"]
    if len(success_rows) != 1:
        raise SpecializedAuditError("summary.csv success-rate row is absent or duplicated")
    expected_values = [f"{suite_rates[suite]:.2f}" for suite in scope.suites] + [
        f"{expected_overall_percent:.2f}"
    ]
    if success_rows[0][1:] != expected_values:
        raise SpecializedAuditError("summary.csv success rates differ from raw outcomes")
    files = [
        {"path": "summary.json", "sha256": summary_sha, "size_bytes": summary_size},
        {"path": "summary.csv", "sha256": summary_csv_sha, "size_bytes": summary_csv_size},
        {
            "path": "task_success_rates.csv",
            "sha256": task_csv_sha,
            "size_bytes": task_csv_size,
        },
    ]
    return {
        "files": files,
        "tree_sha256": _tree_sha256(files),
        "suite_success_rates_percent": suite_rates,
        "overall_success_rate_percent": expected_overall_percent,
    }


def _audit_run(
    root: Path,
    anchor: Mapping[str, Any],
    scope: AuditScope,
    *,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if scope.is_formal and contract is None:
        raise SpecializedAuditError("formal run audit requires a validated contract chain")
    _validate_exact_layout(root, scope)
    episode_map: dict[tuple[str, int, int], bool] = {}
    episode_pair_identities: dict[tuple[str, int, int], dict[str, Any]] = {}
    trace_inventory: list[dict[str, Any]] = []
    result_inventory: list[dict[str, Any]] = []
    receipt_inventory: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    schedule_processes: dict[tuple[str, int], Mapping[str, Any]] = {}
    contract_file_digests: Mapping[str, str] | None = None
    if contract is not None:
        schedule = contract.get("seed_schedule")
        digests = contract.get("digests")
        if not isinstance(schedule, Mapping) or not isinstance(
            schedule.get("task_processes"), list
        ):
            raise SpecializedAuditError("validated contract lacks a seed schedule")
        if not isinstance(digests, Mapping):
            raise SpecializedAuditError("validated contract lacks document digests")
        contract_file_digests = digests
        for process in schedule["task_processes"]:
            if not isinstance(process, Mapping):
                raise SpecializedAuditError("validated seed process is invalid")
            identity = (process.get("task_suite"), process.get("task_id"))
            if identity in schedule_processes:
                raise SpecializedAuditError("validated seed process is duplicated")
            schedule_processes[identity] = process
    for suite in scope.suites:
        for task_id in range(scope.tasks_per_suite):
            expected_seed_process = schedule_processes.get((suite, task_id))
            if contract is not None and expected_seed_process is None:
                raise SpecializedAuditError("validated seed schedule lacks a formal task")
            trace_rows: list[dict[str, Any]] = []
            outcomes: list[bool] = []
            task_description: str | None = None
            for trial_idx in range(scope.trials_per_task):
                relative = f"traces/{suite}/task{task_id:02d}/trial{trial_idx:03d}.json"
                payload, digest, size = _load_relative_json(root, relative)
                trace_identity = _validate_trace(
                    payload,
                    run_anchor=anchor,
                    suite=suite,
                    task_id=task_id,
                    trial_idx=trial_idx,
                    scope=scope,
                    expected_seed_process=expected_seed_process,
                    contract_file_digests=contract_file_digests,
                )
                success = bool(trace_identity["success"])
                if task_description is None:
                    task_description = str(trace_identity["task_description"])
                elif task_description != trace_identity["task_description"]:
                    raise SpecializedAuditError(
                        f"task description changed within {suite}/task{task_id:02d}"
                    )
                episode_map[(suite, task_id, trial_idx)] = success
                episode_pair_identities[(suite, task_id, trial_idx)] = {
                    key: value
                    for key, value in trace_identity.items()
                    if key != "success"
                }
                outcomes.append(success)
                row = {"trial_idx": trial_idx, "path": relative, "sha256": digest, "size_bytes": size}
                trace_rows.append(row)
                trace_inventory.append({"path": relative, "sha256": digest, "size_bytes": size})
            result_path = f"results/{suite}/task{task_id:02d}.json"
            result, result_sha, result_size = _load_relative_json(root, result_path)
            _validate_result(result, suite, task_id, outcomes, scope)
            if result.get("task_description") != task_description:
                raise SpecializedAuditError(
                    f"official result task_description mismatch: {suite}/task{task_id:02d}"
                )
            result_row = {"path": result_path, "sha256": result_sha, "size_bytes": result_size}
            result_inventory.append(result_row)
            receipt_path = f"trace_receipts/{suite}/task{task_id:02d}.json"
            receipt, receipt_sha, receipt_size = _load_relative_json(root, receipt_path)
            _validate_task_receipt(
                receipt,
                root=root,
                anchor=anchor,
                suite=suite,
                task_id=task_id,
                result_row=result_row,
                trace_rows=trace_rows,
                scope=scope,
                expected_seed_process=expected_seed_process,
            )
            receipt_inventory.append({"path": receipt_path, "sha256": receipt_sha, "size_bytes": receipt_size})
            task_rows.append(
                {
                    "task_suite": suite,
                    "task_id": task_id,
                    "successes": sum(outcomes),
                    "total_episodes": scope.trials_per_task,
                    "success_rate": sum(outcomes) / scope.trials_per_task,
                    "result_sha256": result_sha,
                    "trace_tree_sha256": _tree_sha256(trace_rows),
                    "receipt_sha256": receipt_sha,
                }
            )
    if len(episode_map) != scope.episode_count:
        raise SpecializedAuditError("run episode cardinality mismatch")
    summary_audit = _audit_summaries(
        root,
        run_id=str(anchor["run_id"]),
        task_rows=task_rows,
        scope=scope,
    )
    _validate_inventory_readback(
        root,
        [
            *trace_inventory,
            *result_inventory,
            *receipt_inventory,
            *summary_audit["files"],
        ],
    )
    _validate_exact_layout(root, scope)
    return {
        "run_id": anchor["run_id"],
        "episode_map": episode_map,
        "episode_pair_identities": episode_pair_identities,
        "task_rows": task_rows,
        "trace_inventory": trace_inventory,
        "result_inventory": result_inventory,
        "receipt_inventory": receipt_inventory,
        "trace_tree_sha256": _tree_sha256(trace_inventory),
        "result_tree_sha256": _tree_sha256(result_inventory),
        "receipt_tree_sha256": _tree_sha256(receipt_inventory),
        "summary_audit": summary_audit,
    }


def _percentile_interval(samples: list[float], confidence: float) -> tuple[float, float]:
    samples.sort()
    alpha = 1.0 - confidence
    lower = max(0, math.floor(alpha / 2 * len(samples)))
    upper = min(len(samples) - 1, math.ceil((1 - alpha / 2) * len(samples)) - 1)
    return samples[lower], samples[upper]


def _paired_bootstrap(
    reference: Mapping[tuple[str, int, int], bool],
    candidate: Mapping[tuple[str, int, int], bool],
    scope: AuditScope,
) -> tuple[dict[str, float], dict[str, dict[str, float]], str]:
    if set(reference) != set(candidate):
        raise SpecializedAuditError("candidate/reference episode identities differ")
    task_differences: list[float] = []
    suite_task_differences: dict[str, list[float]] = {}
    for suite in scope.suites:
        values: list[float] = []
        for task_id in range(scope.tasks_per_suite):
            value = sum(
                int(candidate[(suite, task_id, trial)]) - int(reference[(suite, task_id, trial)])
                for trial in range(scope.trials_per_task)
            ) / scope.trials_per_task
            values.append(value)
            task_differences.append(value)
        suite_task_differences[suite] = values
    draws_document: dict[str, Any] = {
        "algorithm": "paired-task-bootstrap-python-random-v1",
        "replicates": scope.bootstrap_replicates,
        "seed": scope.bootstrap_seed,
    }
    if any(task_differences):
        rng = random.Random(scope.bootstrap_seed)
        overall_samples = [
            sum(rng.choice(task_differences) for _ in task_differences) / len(task_differences)
            for _ in range(scope.bootstrap_replicates)
        ]
    else:
        overall_samples = [0.0] * scope.bootstrap_replicates
    overall_lower, overall_upper = _percentile_interval(overall_samples, scope.confidence_level)
    overall = {
        "estimate": sum(task_differences) / len(task_differences),
        "ci_lower": overall_lower,
        "ci_upper": overall_upper,
    }
    suites: dict[str, dict[str, float]] = {}
    suite_draw_hashes: dict[str, str] = {}
    for suite_index, suite in enumerate(scope.suites):
        values = suite_task_differences[suite]
        if any(values):
            rng = random.Random(scope.bootstrap_seed + suite_index)
            samples = [
                sum(rng.choice(values) for _ in values) / len(values)
                for _ in range(scope.bootstrap_replicates)
            ]
        else:
            samples = [0.0] * scope.bootstrap_replicates
        lower, upper = _percentile_interval(samples, scope.confidence_level)
        suites[suite] = {
            "estimate": sum(values) / len(values),
            "ci_lower": lower,
            "ci_upper": upper,
        }
        suite_draw_hashes[suite] = _canonical_sha256(samples)
    draws_document["overall_draws_sha256"] = _canonical_sha256(overall_samples)
    draws_document["suite_draws_sha256"] = suite_draw_hashes
    return overall, suites, _canonical_sha256(draws_document)


def _validate_gate_evidence(
    evidence: Mapping[str, Any],
    *,
    scope: AuditScope,
    overall: Mapping[str, float],
    suites: Mapping[str, Mapping[str, float]],
) -> str:
    expected = {
        "evidence_complete": True,
        "episode_count": scope.episode_count,
        "suite_count": len(scope.suites),
        "tasks_per_suite": scope.tasks_per_suite,
        "trials_per_task": scope.trials_per_task,
        "missing_trace_count": 0,
        "non_finite_count": 0,
        "missing_seed_binding_count": 0,
        "first_replan_env_step": scope.warmup_steps,
        "minimum_trace_records_per_episode": scope.minimum_records_per_episode,
        "artifact_bindings_complete": True,
        "paired_episode_identity_complete": True,
        "overall_success_delta": dict(overall),
        "suite_success_deltas": {key: dict(value) for key, value in suites.items()},
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise SpecializedAuditError(f"gate evidence differs from recomputation: {key}")
    without_receipt = {key: value for key, value in evidence.items() if key != "specialized_g0_audit_receipt"}
    return _canonical_sha256(without_receipt)


def _validate_run_anchor(path: Path, expected_sha: str, role: str) -> tuple[dict[str, Any], str]:
    expected_sha = _require_sha256(expected_sha, f"{role} anchor expected SHA")
    payload, actual_sha, _ = _load_absolute_json(path)
    if actual_sha != expected_sha:
        raise SpecializedAuditError(f"{role} run-anchor file SHA mismatch")
    anchor = dict(_require_exact_keys(payload, RUN_ANCHOR_KEYS, f"{role} run anchor"))
    artifact_root = anchor.get("artifact_root")
    raw_log_root = anchor.get("raw_log_root")
    if not isinstance(artifact_root, str) or not isinstance(raw_log_root, str):
        raise SpecializedAuditError(f"{role} run anchor roots are invalid")
    artifact_path = Path(os.path.abspath(os.fspath(Path(artifact_root).expanduser())))
    raw_log_path = Path(os.path.abspath(os.fspath(Path(raw_log_root).expanduser())))
    if (
        anchor["schema_version"] != 1
        or anchor["kind"] != "mf_wam_g0_run_anchor_manifest"
        or anchor["run_role"] != role
        or not isinstance(anchor["run_id"], str)
        or not RUN_ID_RE.fullmatch(anchor["run_id"])
        or not Path(artifact_root).is_absolute()
        or not Path(raw_log_root).is_absolute()
        or str(artifact_path) != artifact_root
        or str(raw_log_path) != raw_log_root
        or not IMAGE_DIGEST_RE.fullmatch(str(anchor["image_digest"]))
        or anchor["fastwam_commit"] != EXPECTED_FASTWAM_COMMIT
    ):
        raise SpecializedAuditError(f"{role} run anchor identity is invalid")
    try:
        raw_log_path.relative_to(artifact_path)
    except ValueError:
        pass
    else:
        raise SpecializedAuditError(f"{role} raw log root must be outside artifact root")
    try:
        artifact_path.relative_to(raw_log_path)
    except ValueError:
        pass
    else:
        raise SpecializedAuditError(f"{role} artifact root must be outside raw log root")
    for field in (
        "preregistration_canonical_sha256",
        "runtime_start_canonical_sha256",
        "seed_schedule_canonical_sha256",
        "resolved_config_sha256",
        "terminal_canonical_sha256",
        "structural_audit_file_sha256",
        "approved_assets_manifest_sha256",
        "manager_manifest_sha256",
    ):
        _require_sha256(anchor[field], f"{role}.{field}")
    _require_commit(anchor["fastwam_commit"], f"{role}.fastwam_commit")
    _require_commit(anchor["instrumentation_commit"], f"{role}.instrumentation_commit")
    return anchor, actual_sha


def _validate_contract_documents(
    anchor: Mapping[str, Any],
    contract_dir: Path,
    approved: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently replay the complete structural contract and live inventories."""

    documents = {
        "preregistration": (
            "preregistration.json",
            "preregistration_canonical_sha256",
            True,
        ),
        "runtime_start": (
            "runtime-start.json",
            "runtime_start_canonical_sha256",
            True,
        ),
        "seed_schedule": (
            "seed-schedule.json",
            "seed_schedule_canonical_sha256",
            True,
        ),
        "resolved_config": ("resolved-config.yaml", "resolved_config_sha256", False),
        "terminal": ("terminal.json", "terminal_canonical_sha256", True),
        "structural_audit": (
            "structural-audit.json",
            "structural_audit_file_sha256",
            False,
        ),
        "data_inventory": ("data-inventory.json", None, True),
    }
    digests: dict[str, str] = {}
    sizes: dict[str, int] = {}
    parsed: dict[str, Any] = {}
    for name, (relative, anchor_field, canonical) in documents.items():
        raw, file_sha, size = _read_relative(contract_dir, relative)
        payload = (
            _loads_json(raw, f"{contract_dir}/{relative}")
            if relative.endswith(".json")
            else None
        )
        digest = _canonical_sha256(payload) if canonical else file_sha
        if anchor_field is not None and digest != anchor[anchor_field]:
            raise SpecializedAuditError(f"contract document mismatch: {name}")
        digests[f"{name}_file_sha256"] = file_sha
        if canonical:
            digests[f"{name}_canonical_sha256"] = digest
        sizes[f"{name}_size_bytes"] = size
        if payload is not None:
            parsed[name] = payload

    prereg_raw = parsed["preregistration"]
    start_raw = parsed["runtime_start"]
    schedule_raw = parsed["seed_schedule"]
    terminal_raw = parsed["terminal"]
    inventory_raw = parsed["data_inventory"]
    structural = parsed["structural_audit"]
    if not all(
        isinstance(item, Mapping)
        for item in (
            prereg_raw,
            start_raw,
            schedule_raw,
            terminal_raw,
            inventory_raw,
            structural,
        )
    ):
        raise SpecializedAuditError("contract documents must be JSON objects")

    trusted = {
        "preregistration_canonical_sha256": anchor[
            "preregistration_canonical_sha256"
        ],
        "runtime_start_canonical_sha256": anchor[
            "runtime_start_canonical_sha256"
        ],
        "terminal_canonical_sha256": anchor["terminal_canonical_sha256"],
    }
    chain_audit = validate_contract_chain(
        preregistration=prereg_raw,
        runtime_start=start_raw,
        terminal=terminal_raw,
        data_inventory=inventory_raw,
        seed_schedule=schedule_raw,
        trusted_anchors=trusted,
        data_root=approved["dataset"]["root"],
        model_cache_root=approved["model_cache"]["root"],
        artifact_root=anchor["artifact_root"],
    )
    if dict(structural) != chain_audit:
        raise SpecializedAuditError(
            "stored structural-audit.json differs from independent chain replay"
        )
    if (
        chain_audit.get("status") != "STRUCTURAL_PASS_ONLY"
        or chain_audit.get("specialized_g0_status") != "UNCERTAIN"
        or chain_audit.get("formal_training_allowed") is not False
        or chain_audit.get("terminal_success") is not True
    ):
        raise SpecializedAuditError("contract terminal/structural status is invalid")

    prereg = validate_preregistration(prereg_raw)
    # The chain replay above already performed the expensive live readback.
    # These calls only retain its independently normalized documents.
    inventory = validate_data_inventory(inventory_raw)
    task_map = {
        "schema_version": 1,
        "kind": "mf_wam_g0_task_map",
        "tasks": inventory["tasks"],
    }
    schedule = validate_seed_schedule(schedule_raw, task_map=task_map)
    start = validate_runtime_start(start_raw, preregistration=prereg)
    terminal = validate_terminal_receipt(
        terminal_raw,
        preregistration=prereg,
        runtime_start=start,
        seed_schedule=schedule,
        task_map=task_map,
    )
    if prereg["run_id"] != anchor["run_id"]:
        raise SpecializedAuditError("contract run_id differs from run anchor")
    if prereg["output"] != {
        "artifact_root": anchor["artifact_root"],
        "overwrite": False,
    }:
        raise SpecializedAuditError("preregistered artifact root differs from run anchor")
    if (
        prereg["image"]["digest"] != anchor["image_digest"]
        or prereg["source"]["fastwam"]["commit"] != anchor["fastwam_commit"]
        or prereg["source"]["instrumentation"]["commit"]
        != anchor["instrumentation_commit"]
    ):
        raise SpecializedAuditError("preregistered source/image identity mismatch")

    expected_artifacts = {
        "checkpoint": {
            "sha256": approved["checkpoint"]["sha256"],
            "size_bytes": approved["checkpoint"]["size_bytes"],
        },
        "dataset_stats": {
            "sha256": approved["dataset_stats"]["sha256"],
            "size_bytes": approved["dataset_stats"]["size_bytes"],
        },
        "resolved_config": {
            "sha256": digests["resolved_config_file_sha256"],
            "size_bytes": sizes["resolved_config_size_bytes"],
        },
    }
    for name, expected in expected_artifacts.items():
        if prereg["artifacts"][name] != expected:
            raise SpecializedAuditError(
                f"preregistered artifact differs from approved identity: {name}"
            )
    if digests["resolved_config_file_sha256"] != anchor["resolved_config_sha256"]:
        raise SpecializedAuditError("resolved config raw digest differs from run anchor")

    approved_cache_core = {
        "algorithm": approved["model_cache"]["algorithm"],
        "file_count": approved["model_cache"]["file_count"],
        "files": sorted(
            [dict(item) for item in approved["model_cache"]["files"]],
            key=lambda item: item["role"],
        ),
    }
    approved_cache = {
        **approved_cache_core,
        "canonical_sha256": _contract_canonical_sha256(approved_cache_core),
    }
    if prereg["artifacts"]["model_cache"] != approved_cache:
        raise SpecializedAuditError("preregistered model cache differs from approved assets")
    expected_data = {
        "tree_sha256": approved["dataset"]["tree_sha256"],
        "tree_algorithm": approved["dataset"]["tree_algorithm"],
        "dataset_id": approved["dataset"]["dataset_id"],
        "revision": approved["dataset"]["revision"],
        "task_map_canonical_sha256": approved["task_map"]["canonical_sha256"],
        "task_count": approved["dataset"]["task_count"],
        "file_count": approved["dataset"]["file_count"],
        "total_size_bytes": approved["dataset"]["total_size_bytes"],
    }
    for key, expected in expected_data.items():
        if prereg["data"][key] != expected or inventory.get(key) != expected:
            raise SpecializedAuditError(f"live data inventory differs from approved {key}")
    if prereg["data"]["inventory_canonical_sha256"] != _contract_canonical_sha256(
        inventory
    ):
        raise SpecializedAuditError("preregistered data inventory digest mismatch")

    return {
        "digests": digests,
        "sizes": sizes,
        "preregistration": prereg,
        "runtime_start": start,
        "seed_schedule": schedule,
        "terminal": terminal,
        "data_inventory": inventory,
        "structural_audit": chain_audit,
    }


def _validate_approved_assets(
    path: Path,
    expected_sha: str,
    *,
    live_readback: bool,
    source_root: Path | None = None,
) -> tuple[dict[str, Any], str]:
    payload, file_sha, _ = _load_absolute_json(path)
    if (
        file_sha != _require_sha256(expected_sha, "approved-assets expected SHA")
        or file_sha != EXPECTED_APPROVED_ASSETS_RAW_SHA256
    ):
        raise SpecializedAuditError("approved-assets file SHA mismatch")
    manifest = _require_exact_keys(
        payload,
        {
            "schema_version",
            "kind",
            "source",
            "dataset",
            "task_map",
            "checkpoint",
            "dataset_stats",
            "model_cache",
        },
        "approved-assets",
    )
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "mf_wam_g0_approved_assets":
        raise SpecializedAuditError("approved-assets schema/kind mismatch")
    if manifest.get("source") != {"fastwam_commit": EXPECTED_FASTWAM_COMMIT, "libero_commit": EXPECTED_LIBERO_COMMIT}:
        raise SpecializedAuditError("approved source commits mismatch")
    dataset = manifest.get("dataset")
    task_map = manifest.get("task_map")
    checkpoint = manifest.get("checkpoint")
    stats_payload = manifest.get("dataset_stats")
    cache = manifest.get("model_cache")
    if not all(isinstance(item, Mapping) for item in (dataset, task_map, checkpoint, stats_payload, cache)):
        raise SpecializedAuditError("approved-assets sections are incomplete")
    _require_exact_keys(
        dataset,
        {
            "dataset_id",
            "root",
            "revision",
            "tree_algorithm",
            "tree_sha256",
            "task_count",
            "file_count",
            "total_size_bytes",
        },
        "approved-assets.dataset",
    )
    _require_exact_keys(
        task_map,
        {
            "repository_relative_path",
            "file_sha256",
            "canonical_sha256",
            "size_bytes",
        },
        "approved-assets.task_map",
    )
    _require_exact_keys(
        checkpoint,
        {"path", "sha256", "size_bytes"},
        "approved-assets.checkpoint",
    )
    _require_exact_keys(
        stats_payload,
        {"path", "sha256", "size_bytes"},
        "approved-assets.dataset_stats",
    )
    _require_exact_keys(
        cache,
        {"root", "algorithm", "file_count", "files"},
        "approved-assets.model_cache",
    )
    if (
        dataset.get("dataset_id") != "libero-40"
        or dataset.get("root") != "/mnt/workspace/LIBERO/libero/libero"
        or dataset.get("tree_algorithm") != TREE_ALGORITHM
        or dataset.get("tree_sha256") != "839db05f1ab9a26966d95c39bf2d292d586c7de1d0d2cd02fae27a04a1a8a21d"
        or dataset.get("task_count") != 40
        or dataset.get("file_count") != 80
        or dataset.get("total_size_bytes") != 1806808
        or dataset.get("revision") != EXPECTED_LIBERO_COMMIT
        or task_map.get("repository_relative_path")
        != "configs/validation/mf_wam_g0_task_map.json"
        or task_map.get("file_sha256") != "71b79aa08e9f73a1107ccc54fb7070da67b56c147b0f77f73a72201582e4b96b"
        or task_map.get("canonical_sha256") != "b5ec2f546ff7b6386b6af181af1b1dd297b2b37f33274f8f411aab9e003f021f"
        or task_map.get("size_bytes") != 16304
        or checkpoint.get("path")
        != "/mnt/workspace/checkpoints/FastWAM/yuanty-fastwam-139eebb6d90cdd9bdbbe465f72c6edc9ad5a518a/libero_uncond_2cam224.pt"
        or checkpoint.get("sha256") != "1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579"
        or checkpoint.get("size_bytes") != 12041735140
        or stats_payload.get("path")
        != "/mnt/workspace/checkpoints/FastWAM/yuanty-fastwam-139eebb6d90cdd9bdbbe465f72c6edc9ad5a518a/libero_uncond_2cam224_dataset_stats.json"
        or stats_payload.get("sha256") != "30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638"
        or stats_payload.get("size_bytes") != 40939
        or cache.get("root") != "/mnt/workspace/checkpoints/FastWAM/model-cache"
        or cache.get("algorithm") != "model-cache-per-file-sha256-v1"
        or cache.get("file_count") != 6
    ):
        raise SpecializedAuditError("approved-assets immutable values differ from the locked release")
    expected_cache = {
        "text_encoder_weights": (
            "DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors",
            "d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23",
            11361845432,
        ),
        "vae_weights": (
            "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors",
            "0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996",
            1409401152,
        ),
        "tokenizer_config": (
            "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json",
            "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05",
            61728,
        ),
        "tokenizer_json": (
            "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json",
            "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b",
            16837417,
        ),
        "tokenizer_special_tokens_map": (
            "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json",
            "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d",
            6623,
        ),
        "tokenizer_model": (
            "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model",
            "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458",
            4548313,
        ),
    }
    files = cache.get("files")
    if not isinstance(files, list) or len(files) != len(expected_cache):
        raise SpecializedAuditError("approved model-cache inventory mismatch")
    observed_roles: set[str] = set()
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != {"role", "path", "sha256", "size_bytes"}:
            raise SpecializedAuditError("approved model-cache entry schema mismatch")
        role = entry.get("role")
        if role not in expected_cache or role in observed_roles:
            raise SpecializedAuditError("approved model-cache role coverage mismatch")
        observed_roles.add(str(role))
        if (
            entry.get("path"),
            entry.get("sha256"),
            entry.get("size_bytes"),
        ) != expected_cache[str(role)]:
            raise SpecializedAuditError(f"approved model-cache identity mismatch: {role}")
    if live_readback:
        if source_root is None:
            raise SpecializedAuditError(
                "approved live readback requires the verified auditor source root"
            )
        task_map_raw, task_map_sha, task_map_size = _read_relative(
            source_root, str(task_map["repository_relative_path"])
        )
        live_task_map = _loads_json(task_map_raw, "approved task-map live readback")
        if (
            task_map_sha != task_map["file_sha256"]
            or task_map_size != task_map["size_bytes"]
            or _canonical_sha256(live_task_map) != task_map["canonical_sha256"]
        ):
            raise SpecializedAuditError("approved task-map live readback mismatch")
        live_entries = [checkpoint, stats_payload]
        cache_root = Path(str(cache.get("root", "")))
        for entry in live_entries:
            digest, size = _hash_absolute(Path(str(entry.get("path", ""))), maximum_bytes=16 << 30)
            if digest != entry.get("sha256") or size != entry.get("size_bytes"):
                raise SpecializedAuditError("approved live artifact readback mismatch")
        for entry in files:
            if not isinstance(entry, Mapping):
                raise SpecializedAuditError("approved model-cache entry is invalid")
            digest, size = _hash_absolute(cache_root / str(entry.get("path", "")), maximum_bytes=16 << 30)
            if digest != entry.get("sha256") or size != entry.get("size_bytes"):
                raise SpecializedAuditError("model-cache live readback mismatch")
    return dict(manifest), file_sha


def _git_identity(root: Path) -> dict[str, Any]:
    root = _canonical_absolute_directory(root, "auditor source root")
    _verify_git_checkout_policy(root, "auditor source")
    commit = _git_readonly(root, "rev-parse", "HEAD").strip()
    dirty = _git_readonly(
        root, "status", "--porcelain", "--untracked-files=all"
    )
    _require_commit(commit, "auditor source commit")
    if dirty:
        raise SpecializedAuditError("auditor source checkout must be clean")
    index_rows = _git_readonly(root, "ls-files", "-v", "-z").split("\0")
    if any(row and not row.startswith("H ") for row in index_rows):
        raise SpecializedAuditError(
            "auditor source index contains assume-unchanged or skip-worktree entries"
        )
    _verify_git_checkout_policy(root, "auditor source")
    return {"source_commit": commit, "clean": True}


def _validate_external_anchor(
    anchor_type: str,
    anchor_id: str,
    path: Path,
    expected_sha: str,
    *,
    expected_anchor_id: str,
    expected_bindings: Mapping[str, Any],
) -> dict[str, str]:
    if (
        anchor_type not in ANCHOR_TYPES
        or not isinstance(anchor_id, str)
        or anchor_id != expected_anchor_id
    ):
        raise SpecializedAuditError("external anchor type/id is invalid")
    payload, actual, _ = _load_absolute_json(path)
    if actual != _require_sha256(expected_sha, f"{anchor_type} expected SHA"):
        raise SpecializedAuditError(f"external anchor readback mismatch: {anchor_type}")
    anchor = _require_exact_keys(
        payload,
        {"schema_version", "kind", "anchor_type", "anchor_id", "bindings"},
        f"external anchor {anchor_type}",
    )
    if (
        anchor["schema_version"] != 1
        or anchor["kind"] != "mf_wam_g0_external_anchor"
        or anchor["anchor_type"] != anchor_type
        or anchor["anchor_id"] != expected_anchor_id
        or not isinstance(anchor["bindings"], Mapping)
        or dict(anchor["bindings"]) != dict(expected_bindings)
    ):
        raise SpecializedAuditError(
            f"external anchor semantics mismatch: {anchor_type}"
        )
    return {
        "anchor_type": anchor_type,
        "anchor_id": anchor_id,
        "artifact_sha256": actual,
    }


def _contract_pair_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return only identities that must be identical in the paired A/B runs."""

    prereg = contract.get("preregistration")
    start = contract.get("runtime_start")
    schedule = contract.get("seed_schedule")
    if not all(isinstance(value, Mapping) for value in (prereg, start, schedule)):
        raise SpecializedAuditError("validated contract pair identity is incomplete")
    launch = prereg.get("launch")
    gpu = start.get("gpu")
    control_process = start.get("control_process")
    if not all(isinstance(value, Mapping) for value in (launch, gpu, control_process)):
        raise SpecializedAuditError("validated runtime pair identity is incomplete")
    return {
        "preregistration": {
            "iteration_id": prereg["iteration_id"],
            "project_page_id": prereg["project_page_id"],
            "source": prereg["source"],
            "image": prereg["image"],
            "artifacts": {
                key: prereg["artifacts"][key]
                for key in ("checkpoint", "dataset_stats", "model_cache")
            },
            "data": prereg["data"],
            "seeds": prereg["seeds"],
            "runtime_lock": prereg["runtime_lock"],
            "runtime_environment": prereg["runtime_environment"],
            "evaluation": prereg["evaluation"],
        },
        "launch": {
            key: launch[key]
            for key in (
                "provider",
                "gpu_count",
                "gpu_model",
                "gpu_memory_mib",
                "driver_version",
            )
        },
        "runtime_start": {
            "source": start["source"],
            "image": start["image"],
            "bindings": {
                key: start["bindings"][key]
                for key in (
                    "checkpoint_sha256",
                    "dataset_stats_sha256",
                    "data_inventory_canonical_sha256",
                    "data_tree_sha256",
                    "seed_schedule_canonical_sha256",
                    "model_cache_inventory_canonical_sha256",
                )
            },
            "runtime": start["runtime"],
            "runtime_environment": start["runtime_environment"],
            "gpu": {
                key: gpu[key]
                for key in ("count", "model", "memory_mib", "driver_version")
            },
            "control_process": {
                "python_hash_seed": control_process["python_hash_seed"]
            },
            "imports": start["imports"],
            "model_cache_inventory": start["model_cache_inventory"],
        },
        "seed_schedule": schedule,
    }


def audit_pair(
    *,
    reference_root: Path,
    candidate_root: Path,
    reference_anchor: Mapping[str, Any],
    candidate_anchor: Mapping[str, Any],
    scope: AuditScope = FORMAL_SCOPE,
    reference_contract: Mapping[str, Any] | None = None,
    candidate_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if reference_anchor["run_id"] == candidate_anchor["run_id"]:
        raise SpecializedAuditError("reference and candidate run_id must differ")
    if Path(reference_root).resolve() == Path(candidate_root).resolve():
        raise SpecializedAuditError("reference and candidate artifact roots must differ")
    if Path(reference_anchor["raw_log_root"]).resolve() == Path(
        candidate_anchor["raw_log_root"]
    ).resolve():
        raise SpecializedAuditError("reference and candidate raw log roots must differ")
    common_fields = (
        "seed_schedule_canonical_sha256",
        "approved_assets_manifest_sha256",
        "image_digest",
        "fastwam_commit",
        "instrumentation_commit",
    )
    for field in common_fields:
        if reference_anchor[field] != candidate_anchor[field]:
            raise SpecializedAuditError(f"paired runs differ in locked field: {field}")
    if scope.is_formal and (
        reference_contract is None or candidate_contract is None
    ):
        raise SpecializedAuditError("formal paired audit requires both contract chains")
    if reference_contract is not None and candidate_contract is not None:
        if _contract_pair_identity(reference_contract) != _contract_pair_identity(
            candidate_contract
        ):
            raise SpecializedAuditError("paired runs differ in validated runtime identity")
    reference = _audit_run(
        reference_root, reference_anchor, scope, contract=reference_contract
    )
    candidate = _audit_run(
        candidate_root, candidate_anchor, scope, contract=candidate_contract
    )
    if reference["episode_pair_identities"] != candidate["episode_pair_identities"]:
        differing = next(
            (
                identity
                for identity in reference["episode_pair_identities"]
                if reference["episode_pair_identities"].get(identity)
                != candidate["episode_pair_identities"].get(identity)
            ),
            None,
        )
        raise SpecializedAuditError(
            f"paired episode identity mismatch: {differing}"
        )
    overall, suites, bootstrap_draws_sha = _paired_bootstrap(
        reference["episode_map"], candidate["episode_map"], scope
    )
    outcome_pass = (
        overall["ci_lower"] >= -scope.overall_margin
        and overall["ci_upper"] <= scope.overall_margin
        and all(item["ci_lower"] >= -scope.suite_drop_margin for item in suites.values())
    )
    metric_rows = [
        {
            "task_suite": suite,
            "task_id": task_id,
            "trial_idx": trial_idx,
            "reference_success": reference["episode_map"][(suite, task_id, trial_idx)],
            "candidate_success": candidate["episode_map"][(suite, task_id, trial_idx)],
            "paired_identity_sha256": _canonical_sha256(
                reference["episode_pair_identities"][(suite, task_id, trial_idx)]
            ),
        }
        for suite in scope.suites
        for task_id in range(scope.tasks_per_suite)
        for trial_idx in range(scope.trials_per_task)
    ]
    combined_trace_rows = [
        {**row, "path": f"reference/{row['path']}"} for row in reference["trace_inventory"]
    ] + [
        {**row, "path": f"candidate/{row['path']}"} for row in candidate["trace_inventory"]
    ]
    return {
        "scope": scope,
        "reference": reference,
        "candidate": candidate,
        "overall_success_delta": overall,
        "suite_success_deltas": suites,
        "bootstrap_draws_sha256": bootstrap_draws_sha,
        "outcome_parity_pass": outcome_pass,
        "metric_rows_sha256": _canonical_sha256(metric_rows),
        "trace_tree_sha256": _tree_sha256(combined_trace_rows),
    }


def _validate_trace_source_bindings(
    run_report: Mapping[str, Any],
    config_audit: Mapping[str, Any],
    *,
    role: str,
) -> None:
    episodes = run_report.get("episode_pair_identities")
    if not isinstance(episodes, Mapping):
        raise SpecializedAuditError(f"{role} episode source inventory is absent")
    expected_official = config_audit.get("official_source")
    expected_instrumentation = config_audit.get("instrumentation_source")
    if not isinstance(expected_official, Mapping) or not isinstance(
        expected_instrumentation, Mapping
    ):
        raise SpecializedAuditError(f"{role} independently verified sources are absent")
    for episode, identity in episodes.items():
        if (
            not isinstance(identity, Mapping)
            or identity.get("official_source") != expected_official
            or identity.get("instrumentation_source") != expected_instrumentation
        ):
            raise SpecializedAuditError(
                f"{role} trace source differs from manager-bound checkout: {episode}"
            )


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> tuple[str, int]:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    absolute.parent.mkdir(parents=True, exist_ok=True)
    parent, parent_fd = _open_root(absolute.parent)
    del parent
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    temporary_name = f".{absolute.name}.{os.getpid()}.{os.urandom(12).hex()}.tmp"
    fd: int | None = None
    temporary_exists = False
    try:
        fd = os.open(temporary_name, flags, 0o644, dir_fd=parent_fd)
        temporary_exists = True
        try:
            view = memoryview(encoded)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
            fd = None
        try:
            os.link(
                temporary_name,
                absolute.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise SpecializedAuditError(
                f"refusing to replace existing receipt: {absolute}"
            ) from exc
        os.unlink(temporary_name, dir_fd=parent_fd)
        temporary_exists = False
        os.fsync(parent_fd)
    finally:
        if fd is not None:
            os.close(fd)
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)
    readback, digest, size = _read_absolute(absolute)
    if readback != encoded:
        raise SpecializedAuditError("receipt readback differs from published bytes")
    return digest, size


def _policy_identity(policy: Mapping[str, Any]) -> tuple[str, str]:
    if policy.get("policy_id") != "MF-WAM-G0-G3-2026-08-02-v3":
        raise SpecializedAuditError("unexpected G0 gate policy identity")
    try:
        check = next(
            item
            for item in policy["gates"]["G0"]["checks"]
            if item.get("type") == "specialized_audited_receipt"
        )
    except (KeyError, TypeError, StopIteration) as exc:
        raise SpecializedAuditError("gate policy lacks the specialized G0 check") from exc
    if (
        check.get("receipt_kind") != "mf_wam_g0_specialized_audit_receipt"
        or check.get("receipt_schema_version") != 1
        or check.get("required_formal_training_allowed") is not False
        or check.get("ci_contract_id") != CI_CONTRACT_ID
        or check.get("required_scope") != _formal_receipt_scope()
        or tuple(check.get("required_artifact_digests") or ()) != RECEIPT_ARTIFACT_DIGEST_KEYS
        or tuple(check.get("required_external_anchor_types") or ()) != ANCHOR_TYPES
    ):
        raise SpecializedAuditError("specialized G0 policy contract differs from the auditor")
    return policy["policy_id"], _canonical_sha256(policy)


def _formal_receipt_scope() -> dict[str, Any]:
    return {
        "episode_count": 2000,
        "suite_count": 4,
        "tasks_per_suite": 10,
        "trials_per_task": 50,
        "confidence_level": 0.95,
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 42,
        "outcome_parity_classification": "OUTCOME_PARITY_ONLY",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--reference-contract-dir", type=Path, required=True)
    parser.add_argument("--candidate-contract-dir", type=Path, required=True)
    parser.add_argument("--reference-run-anchor", type=Path, required=True)
    parser.add_argument("--reference-run-anchor-sha256", required=True)
    parser.add_argument("--candidate-run-anchor", type=Path, required=True)
    parser.add_argument("--candidate-run-anchor-sha256", required=True)
    parser.add_argument("--approved-assets", type=Path, required=True)
    parser.add_argument("--approved-assets-sha256", required=True)
    parser.add_argument("--gate-policy", type=Path, required=True)
    parser.add_argument("--gate-policy-sha256", required=True)
    parser.add_argument("--gate-evidence", type=Path, required=True)
    parser.add_argument("--auditor-source-root", type=Path, required=True)
    for anchor_type in ANCHOR_TYPES:
        option = anchor_type.replace("_", "-")
        parser.add_argument(f"--{option}-anchor-id", required=True)
        parser.add_argument(f"--{option}-anchor-file", type=Path, required=True)
        parser.add_argument(f"--{option}-anchor-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        approved, approved_sha = _validate_approved_assets(
            args.approved_assets,
            args.approved_assets_sha256,
            live_readback=True,
            source_root=args.auditor_source_root,
        )
        reference_anchor, reference_anchor_sha = _validate_run_anchor(
            args.reference_run_anchor,
            args.reference_run_anchor_sha256,
            "reference",
        )
        candidate_anchor, candidate_anchor_sha = _validate_run_anchor(
            args.candidate_run_anchor,
            args.candidate_run_anchor_sha256,
            "candidate",
        )
        if reference_anchor["approved_assets_manifest_sha256"] != approved_sha or candidate_anchor["approved_assets_manifest_sha256"] != approved_sha:
            raise SpecializedAuditError("run anchors do not bind the approved-assets manifest")
        if Path(reference_anchor["artifact_root"]) != Path(os.path.abspath(os.fspath(args.reference_root.expanduser()))):
            raise SpecializedAuditError("reference CLI root differs from its external anchor")
        if Path(candidate_anchor["artifact_root"]) != Path(os.path.abspath(os.fspath(args.candidate_root.expanduser()))):
            raise SpecializedAuditError("candidate CLI root differs from its external anchor")
        reference_contract = _validate_contract_documents(
            reference_anchor, args.reference_contract_dir, approved
        )
        candidate_contract = _validate_contract_documents(
            candidate_anchor, args.candidate_contract_dir, approved
        )
        reference_config_audit = _audit_resolved_config_gate(
            anchor=reference_anchor,
            contract=reference_contract,
            contract_dir=args.reference_contract_dir,
            approved=approved,
        )
        candidate_config_audit = _audit_resolved_config_gate(
            anchor=candidate_anchor,
            contract=candidate_contract,
            contract_dir=args.candidate_contract_dir,
            approved=approved,
        )
        if (
            reference_config_audit["paired_static_projection_sha256"]
            != candidate_config_audit["paired_static_projection_sha256"]
        ):
            raise SpecializedAuditError(
                "paired runs differ in independently composed static config"
            )
        report = audit_pair(
            reference_root=args.reference_root,
            candidate_root=args.candidate_root,
            reference_anchor=reference_anchor,
            candidate_anchor=candidate_anchor,
            scope=FORMAL_SCOPE,
            reference_contract=reference_contract,
            candidate_contract=candidate_contract,
        )
        _validate_trace_source_bindings(
            report["reference"], reference_config_audit, role="reference"
        )
        _validate_trace_source_bindings(
            report["candidate"], candidate_config_audit, role="candidate"
        )
        if not report["outcome_parity_pass"]:
            raise SpecializedAuditError("recomputed G0 outcome parity does not pass")
        evidence, _, _ = _load_absolute_json(args.gate_evidence)
        if not isinstance(evidence, Mapping):
            raise SpecializedAuditError("gate evidence must be an object")
        evidence_sha = _validate_gate_evidence(
            evidence,
            scope=FORMAL_SCOPE,
            overall=report["overall_success_delta"],
            suites=report["suite_success_deltas"],
        )
        policy, policy_file_sha, _ = _load_absolute_json(args.gate_policy)
        if not isinstance(policy, Mapping):
            raise SpecializedAuditError("gate policy must be an object")
        if policy_file_sha != _require_sha256(args.gate_policy_sha256, "gate policy expected SHA"):
            raise SpecializedAuditError("gate policy file SHA mismatch")
        policy_id, policy_sha = _policy_identity(policy)
        auditor = _git_identity(args.auditor_source_root)
        if auditor["source_commit"] != candidate_anchor["instrumentation_commit"]:
            raise SpecializedAuditError(
                "auditor and instrumentation must be the same verified source commit"
            )
        auditor_root = _canonical_absolute_directory(
            args.auditor_source_root, "auditor source root"
        )
        if any(
            config_audit["instrumentation_source"]["root"] != str(auditor_root)
            for config_audit in (reference_config_audit, candidate_config_audit)
        ):
            raise SpecializedAuditError(
                "manager-bound instrumentation root differs from the auditor checkout"
            )
        source_path = Path(__file__).resolve(strict=True)
        try:
            source_path.relative_to(args.auditor_source_root.resolve(strict=True))
        except ValueError as exc:
            raise SpecializedAuditError("auditor source file is outside its verified Git root") from exc
        auditor_source_sha, _ = _hash_absolute(source_path)
        source_manifest = {
            "fastwam_commit": EXPECTED_FASTWAM_COMMIT,
            "instrumentation_commit": candidate_anchor["instrumentation_commit"],
            "auditor_commit": auditor["source_commit"],
            "auditor_source_sha256": auditor_source_sha,
            "reference_official_source_sha256": _canonical_sha256(
                reference_config_audit["official_source"]
            ),
            "candidate_official_source_sha256": _canonical_sha256(
                candidate_config_audit["official_source"]
            ),
            "reference_static_config_projection_sha256": reference_config_audit[
                "static_projection_sha256"
            ],
            "candidate_static_config_projection_sha256": candidate_config_audit[
                "static_projection_sha256"
            ],
            "paired_static_config_projection_sha256": candidate_config_audit[
                "paired_static_projection_sha256"
            ],
        }
        source_manifest_sha = _canonical_sha256(source_manifest)
        reference_prereg = reference_contract["preregistration"]
        candidate_prereg = candidate_contract["preregistration"]
        reference_structural = reference_contract["structural_audit"]
        candidate_structural = candidate_contract["structural_audit"]
        notion_bindings = {
            "project_page_id": reference_prereg["project_page_id"],
            "iteration_id": reference_prereg["iteration_id"],
            "reference_run_id": reference_anchor["run_id"],
            "candidate_run_id": candidate_anchor["run_id"],
            "reference_preregistration_canonical_sha256": reference_anchor[
                "preregistration_canonical_sha256"
            ],
            "candidate_preregistration_canonical_sha256": candidate_anchor[
                "preregistration_canonical_sha256"
            ],
        }
        immutable_root_bindings = {
            "reference_run_id": reference_anchor["run_id"],
            "reference_artifact_root": reference_anchor["artifact_root"],
            "reference_terminal_canonical_sha256": reference_anchor[
                "terminal_canonical_sha256"
            ],
            "reference_structural_audit_file_sha256": reference_anchor[
                "structural_audit_file_sha256"
            ],
            "reference_artifact_inventory_raw_sha256": reference_structural[
                "artifact_audit"
            ]["artifact_inventory_raw_sha256"],
            "reference_artifact_inventory_tree_sha256": reference_structural[
                "artifact_audit"
            ]["artifact_inventory_tree_sha256"],
            "reference_raw_log_root": reference_anchor["raw_log_root"],
            "reference_manager_manifest_sha256": reference_anchor[
                "manager_manifest_sha256"
            ],
            "candidate_run_id": candidate_anchor["run_id"],
            "candidate_artifact_root": candidate_anchor["artifact_root"],
            "candidate_terminal_canonical_sha256": candidate_anchor[
                "terminal_canonical_sha256"
            ],
            "candidate_structural_audit_file_sha256": candidate_anchor[
                "structural_audit_file_sha256"
            ],
            "candidate_artifact_inventory_raw_sha256": candidate_structural[
                "artifact_audit"
            ]["artifact_inventory_raw_sha256"],
            "candidate_artifact_inventory_tree_sha256": candidate_structural[
                "artifact_audit"
            ]["artifact_inventory_tree_sha256"],
            "candidate_raw_log_root": candidate_anchor["raw_log_root"],
            "candidate_manager_manifest_sha256": candidate_anchor[
                "manager_manifest_sha256"
            ],
        }
        source_bindings = {
            **source_manifest,
            "libero_commit": EXPECTED_LIBERO_COMMIT,
            "reference_source_identity_sha256": _canonical_sha256(
                reference_prereg["source"]
            ),
            "candidate_source_identity_sha256": _canonical_sha256(
                candidate_prereg["source"]
            ),
            "paired_trace_source_identity_sha256": _canonical_sha256(
                [
                    {
                        "episode": list(identity),
                        "official_source": values["official_source"],
                        "instrumentation_source": values[
                            "instrumentation_source"
                        ],
                    }
                    for identity, values in sorted(
                        report["reference"]["episode_pair_identities"].items()
                    )
                ]
            ),
            "reference_resolved_config_audit_sha256": _canonical_sha256(
                reference_config_audit
            ),
            "candidate_resolved_config_audit_sha256": _canonical_sha256(
                candidate_config_audit
            ),
        }
        container_bindings = {
            "image_uri": reference_prereg["image"]["uri"],
            "image_digest": candidate_anchor["image_digest"],
            "reference_runtime_start_canonical_sha256": reference_anchor[
                "runtime_start_canonical_sha256"
            ],
            "candidate_runtime_start_canonical_sha256": candidate_anchor[
                "runtime_start_canonical_sha256"
            ],
        }
        external_requirements = {
            "notion_experiment_page": (
                reference_prereg["project_page_id"],
                notion_bindings,
            ),
            "immutable_artifact_root": (
                _canonical_sha256(immutable_root_bindings),
                immutable_root_bindings,
            ),
            "source_commit": (EXPECTED_FASTWAM_COMMIT, source_bindings),
            "container_image_digest": (
                candidate_anchor["image_digest"],
                container_bindings,
            ),
        }
        anchor_lineage = []
        for anchor_type in ANCHOR_TYPES:
            attribute = anchor_type
            expected_anchor_id, expected_bindings = external_requirements[anchor_type]
            anchor_lineage.append(
                _validate_external_anchor(
                    anchor_type,
                    getattr(args, f"{attribute}_anchor_id"),
                    getattr(args, f"{attribute}_anchor_file"),
                    getattr(args, f"{attribute}_anchor_sha256"),
                    expected_anchor_id=expected_anchor_id,
                    expected_bindings=expected_bindings,
                )
            )
        identity_inventory = {
            "reference_run_anchor_sha256": reference_anchor_sha,
            "candidate_run_anchor_sha256": candidate_anchor_sha,
            "reference_contract": reference_contract,
            "candidate_contract": candidate_contract,
            "reference_resolved_config_audit": reference_config_audit,
            "candidate_resolved_config_audit": candidate_config_audit,
            "bootstrap_draws_sha256": report["bootstrap_draws_sha256"],
        }
        terminal_bundle = {
            "reference_terminal_sha256": reference_anchor["terminal_canonical_sha256"],
            "candidate_terminal_sha256": candidate_anchor["terminal_canonical_sha256"],
            "reference_result_tree_sha256": report["reference"]["result_tree_sha256"],
            "candidate_result_tree_sha256": report["candidate"]["result_tree_sha256"],
            "reference_receipt_tree_sha256": report["reference"]["receipt_tree_sha256"],
            "candidate_receipt_tree_sha256": report["candidate"]["receipt_tree_sha256"],
            "reference_summary_tree_sha256": report["reference"]["summary_audit"]["tree_sha256"],
            "candidate_summary_tree_sha256": report["candidate"]["summary_audit"]["tree_sha256"],
            "reference_manager_manifest_sha256": reference_anchor[
                "manager_manifest_sha256"
            ],
            "candidate_manager_manifest_sha256": candidate_anchor[
                "manager_manifest_sha256"
            ],
            "static_config_projection_sha256": candidate_config_audit[
                "paired_static_projection_sha256"
            ],
            "overall_success_delta": report["overall_success_delta"],
            "suite_success_deltas": report["suite_success_deltas"],
        }
        artifact_digests = {
            "source_manifest_sha256": source_manifest_sha,
            "data_manifest_sha256": _canonical_sha256(approved["dataset"]),
            "seed_manifest_sha256": candidate_anchor["seed_schedule_canonical_sha256"],
            "resolved_config_sha256": candidate_anchor["resolved_config_sha256"],
            "checkpoint_sha256": approved["checkpoint"]["sha256"],
            "dataset_stats_sha256": approved["dataset_stats"]["sha256"],
            "runtime_environment_sha256": _canonical_sha256(
                {
                    "reference": reference_anchor["runtime_start_canonical_sha256"],
                    "candidate": candidate_anchor["runtime_start_canonical_sha256"],
                    "image_digest": candidate_anchor["image_digest"],
                }
            ),
            "identity_inventory_sha256": _canonical_sha256(identity_inventory),
            "metric_rows_sha256": report["metric_rows_sha256"],
            "trace_tree_sha256": report["trace_tree_sha256"],
            "terminal_summary_bundle_sha256": _canonical_sha256(terminal_bundle),
        }
        if tuple(artifact_digests) != RECEIPT_ARTIFACT_DIGEST_KEYS:
            raise SpecializedAuditError("internal specialized artifact digest order/schema drift")
        receipt = {
            "schema_version": 1,
            "kind": "mf_wam_g0_specialized_audit_receipt",
            "gate_id": "G0",
            "policy_id": policy_id,
            "policy_sha256": policy_sha,
            "ci_contract_id": CI_CONTRACT_ID,
            "evidence_sha256": evidence_sha,
            "terminal": True,
            "scientific_status": "SPECIALIZED_G0_PASS",
            "formal_training_allowed": False,
            "source_manifest_sha256": source_manifest_sha,
            "scope": _formal_receipt_scope(),
            "auditor": auditor,
            "artifact_digests": artifact_digests,
            "anchor_lineage": anchor_lineage,
        }
        receipt_sha, receipt_size = _write_exclusive_json(args.output, receipt)
        print(
            json.dumps(
                {
                    "status": "SPECIALIZED_G0_PASS",
                    "formal_training_allowed": False,
                    "output": str(args.output.resolve()),
                    "sha256": receipt_sha,
                    "size_bytes": receipt_size,
                    "overall_success_delta": report["overall_success_delta"],
                    "suite_success_deltas": report["suite_success_deltas"],
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    except (SpecializedAuditError, OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "formal_training_allowed": False,
                    "error": str(exc),
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
