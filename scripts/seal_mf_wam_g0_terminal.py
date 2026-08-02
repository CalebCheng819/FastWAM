#!/usr/bin/env python3
"""Fail-closed terminal bundle sealer for a completed MF-WAM G0 run.

This command proves structural completeness only.  It never turns a sealed run
into a scientific G0 PASS; the comparative specialized auditor remains a
separate mandatory stage.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import itertools
import json
import math
import os
import re
import stat
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

# Preserve the dedicated formal checkout: importing the local contract must not
# create an ignored ``__pycache__`` entry before source identity is audited.
sys.dont_write_bytecode = True


try:
    import fastwam.validation.g0_contract as contract
except ModuleNotFoundError:
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root / "src"))
    import fastwam.validation.g0_contract as contract  # type: ignore[no-redef]


SUITES = contract.SUITES
TASKS_PER_SUITE = contract.TASKS_PER_SUITE
TRIALS_PER_TASK = contract.TRIALS_PER_TASK
EXPECTED_TASKS = contract.EXPECTED_TASKS
EXPECTED_EPISODES = contract.EXPECTED_EPISODES
EXPECTED_INPUT_FILES = EXPECTED_TASKS * 2 + EXPECTED_EPISODES
EXPECTED_INVENTORY_FILES = contract.EXPECTED_TERMINAL_FILES
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PYTHON_EXECUTABLE_RE = re.compile(r"^python(?:3(?:\.\d+)?)?$")
_TEMP_COUNTER = itertools.count()

_TRACED_RUNNER_RELATIVE = PurePosixPath("scripts/run_mf_wam_g0_traced.py")
_REQUIRED_HYDRA_OVERRIDES = frozenset(
    (
        "task", "ckpt", "EVALUATION.dataset_stats_path",
        "EVALUATION.task_suite_name", "EVALUATION.task_id", "gpu_id",
        "output_dir",
        "EVALUATION.num_trials", "EVALUATION.output_dir",
        "EVALUATION.env_num", "EVALUATION.num_steps_wait",
        "EVALUATION.replan_steps", "EVALUATION.action_horizon",
        "EVALUATION.binarize_gripper", "seed",
    )
)
_OPTIONAL_HYDRA_OVERRIDES = frozenset(
    ("EVALUATION.visualize_future_video", "EVALUATION.use_action_ensembler")
)
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
_RUNNER_ENVIRONMENT_KEYS = frozenset(
    (
        *_FIXED_WORKER_ENVIRONMENT,
        "MF_WAM_OFFICIAL_ROOT", "MF_WAM_OFFICIAL_COMMIT", "MF_WAM_G0_RUN_ID",
        "MF_WAM_INSTRUMENTATION_COMMIT", "MF_WAM_G0_PREREG_PATH",
        "MF_WAM_G0_PREREG_SHA256", "MF_WAM_G0_RUNTIME_START_PATH",
        "MF_WAM_G0_RUNTIME_START_SHA256", "MF_WAM_G0_SEED_SCHEDULE_PATH",
        "MF_WAM_G0_SEED_SCHEDULE_SHA256", "MF_WAM_G0_RESOLVED_CONFIG_PATH",
        "MF_WAM_G0_RESOLVED_CONFIG_SHA256", "CUDA_VISIBLE_DEVICES",
        "PYTHONHASHSEED", "WORLD_SIZE", "RANK", "LOCAL_RANK", "MUJOCO_GL",
        "PYOPENGL_PLATFORM", "DIFFSYNTH_DOWNLOAD_SOURCE",
        "DIFFSYNTH_MODEL_BASE_PATH", "DIFFSYNTH_SKIP_DOWNLOAD",
    )
)
WORKER_TERMINAL_KEYS = frozenset(
    (
        "status", "kind", "run_id", "process_receipt", "official_commit",
        "official_result_type", "official_result_receipt",
        "terminal_source_identities", "external_prelaunch_commit_tree_gate_required",
        "environment_sha256",
    )
)

TRACE_TOP_KEYS = frozenset(("schema_version", "kind", "metadata", "records"))
TRACE_METADATA_KEYS = frozenset(
    (
        "run_id", "task_suite", "task_id", "trial_idx", "initial_state_index",
        "initial_state_sha256", "task_description", "warmup_steps",
        "first_replan_env_step", "replan_steps", "action_horizon",
        "action_dimension", "state_dimension", "seed_contract",
        "seed_schedule_process", "upstream_digests", "official_source",
        "instrumentation_source", "success", "record_count",
        "environment_step_count", "observer_rng_unchanged_checks",
        "official_module_origin_inventory_sha256",
    )
)
TRACE_RECORD_KEYS = frozenset(
    (
        "episode_idx", "replan_idx", "env_step", "state", "pre_state",
        "pre_observation_sha256", "policy_seed", "policy_seed_scope",
        "proposed_raw_action_chunk", "proposed_env_action_chunk",
        "executed_env_actions", "executed_count", "done_after_execution",
        "executions",
    )
)
TRACE_EXECUTION_KEYS = frozenset(
    ("action", "post_state", "post_observation_sha256", "done")
)
TRACE_UPSTREAM_KEYS = frozenset(
    (
        "preregistration_file_sha256", "preregistration_canonical_sha256",
        "runtime_start_file_sha256", "runtime_start_canonical_sha256",
        "seed_schedule_file_sha256", "seed_schedule_canonical_sha256",
        "resolved_config_sha256",
    )
)
MANAGER_TOP_KEYS = frozenset(
    (
        "schema_version", "kind", "run_id", "completed_at",
        "manager_exit_code", "artifact_root", "raw_log_root", "gpu_ids",
        "upstream_bindings", "canonical_input_file_count",
        "canonical_input_tree_sha256", "task_processes",
    )
)
MANAGER_UPSTREAM_KEYS = frozenset(
    (
        "preregistration_file_sha256", "runtime_start_file_sha256",
        "seed_schedule_file_sha256", "resolved_config_sha256",
        "official_commit", "instrumentation_commit", "python_hash_seed",
    )
)
MANAGER_PROCESS_KEYS = frozenset(
    (
        "process_id", "task_suite", "task_id", "gpu_id", "state",
        "launched_at", "completed_at", "exit_code", "complete", "failure_reason",
        "command_sha256", "environment_sha256", "log_path", "log_sha256",
        "log_size_bytes", "status_path", "status_sha256", "status_size_bytes",
        "result_path", "result_sha256", "result_size_bytes",
        "trace_receipt_path", "trace_receipt_sha256",
        "trace_receipt_size_bytes", "trace_tree_sha256", "episode_count",
        "raw_result_source_path", "raw_result_archive_path",
        "raw_result_sha256", "raw_result_size_bytes",
    )
)
MANAGER_STATUS_KEYS = frozenset(
    (
        "schema_version", "kind", "run_id", "process_id", "task_suite",
        "task_id", "gpu_id", "state", "launched_at", "completed_at",
        "exit_code", "complete", "failure_reason", "command_argv", "command_sha256",
        "environment_bindings", "environment_sha256", "log",
        "canonical_result", "trace_receipt", "raw_result",
    )
)


class SealError(RuntimeError):
    """Raised when a terminal bundle cannot be proven complete and immutable."""


def _reject_constant(value: str) -> None:
    raise SealError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SealError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except SealError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SealError(f"cannot load strict JSON {label}: {exc}") from exc


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SealError(f"cannot encode canonical JSON: {exc}") from exc


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _lexical_absolute(path: Path) -> Path:
    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    if ".." in absolute.parts:
        raise SealError(f"parent traversal is forbidden: {path}")
    return Path(os.path.normpath(str(absolute)))


def _open_absolute_nofollow(path: Path, *, directory: bool) -> int:
    absolute = _lexical_absolute(path)
    parts = absolute.parts[1:]
    root_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    current_fd = root_fd
    try:
        for index, part in enumerate(parts):
            final = index == len(parts) - 1
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            if not final or directory:
                flags |= os.O_DIRECTORY
            next_fd = os.open(part, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        metadata = os.fstat(current_fd)
        valid = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
        if not valid:
            raise SealError(f"path is not a {'directory' if directory else 'regular file'}: {absolute}")
        return current_fd
    except Exception as exc:
        try:
            os.close(current_fd)
        except OSError:
            pass
        if isinstance(exc, SealError):
            raise
        raise SealError(f"cannot open without following symlinks: {absolute}: {exc}") from exc


def _read_fd(fd: int, *, capture: bool) -> dict[str, Any]:
    before = os.fstat(fd)
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if capture else None
    total = 0
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
        if chunks is not None:
            chunks.append(chunk)
    after = os.fstat(fd)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable) or total != after.st_size:
        raise SealError("file changed during no-follow readback")
    if after.st_nlink != 1:
        raise SealError("hardlinked input artifact is forbidden")
    result: dict[str, Any] = {
        "sha256": digest.hexdigest(),
        "size_bytes": total,
        "identity": (after.st_dev, after.st_ino),
    }
    if chunks is not None:
        result["bytes"] = b"".join(chunks)
    return result


def _read_absolute(path: Path, *, capture: bool = True) -> dict[str, Any]:
    fd = _open_absolute_nofollow(path, directory=False)
    try:
        return _read_fd(fd, capture=capture)
    finally:
        os.close(fd)


def _ensure_absolute_absent(path: Path) -> None:
    """Fail if *path* already names any filesystem object, without following it."""

    target = _lexical_absolute(path)
    if not target.name:
        raise SealError(f"terminal output must name a file: {target}")
    parent_fd = _open_absolute_nofollow(target.parent, directory=True)
    try:
        try:
            os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise SealError(f"cannot prove terminal output is absent: {target}: {exc}") from exc
        raise SealError(f"refusing to overwrite existing artifact: {target}")
    finally:
        os.close(parent_fd)


def _open_relative(root: Path, relative: str) -> int:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise SealError(f"unsafe relative path: {relative}")
    directory_fd = _open_absolute_nofollow(root, directory=True)
    try:
        for part in pure.parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(
            pure.parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise SealError(f"cannot safely open artifact {relative}: {exc}") from exc
    finally:
        os.close(directory_fd)


def _read_relative(root: Path, relative: str, *, capture: bool = True) -> dict[str, Any]:
    fd = _open_relative(root, relative)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise SealError(f"artifact is not a regular file: {relative}")
        return _read_fd(fd, capture=capture)
    finally:
        os.close(fd)


def _atomic_publish_absolute(path: Path, raw: bytes) -> None:
    target = _lexical_absolute(path)
    parent_fd = _open_absolute_nofollow(target.parent, directory=True)
    temporary = f".{target.name}.{os.getpid()}.{next(_TEMP_COUNTER):016x}.tmp"
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fchmod(handle.fileno(), 0o644)
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise SealError(f"refusing to overwrite existing artifact: {target}") from exc
        os.fsync(parent_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _atomic_publish_relative(root: Path, relative: str, raw: bytes) -> None:
    pure = PurePosixPath(relative)
    parent = root.joinpath(*pure.parts[:-1])
    _atomic_publish_absolute(parent / pure.name, raw)


def _expected_input_paths() -> set[str]:
    paths: set[str] = set()
    for suite in SUITES:
        for task_id in range(TASKS_PER_SUITE):
            paths.add(f"results/{suite}/task{task_id:02d}.json")
            paths.add(f"trace_receipts/{suite}/task{task_id:02d}.json")
            for trial_idx in range(TRIALS_PER_TASK):
                paths.add(f"traces/{suite}/task{task_id:02d}/trial{trial_idx:03d}.json")
    if len(paths) != EXPECTED_INPUT_FILES:
        raise AssertionError("internal expected input scope is wrong")
    return paths


def _expected_directories() -> set[str]:
    directories = {"results", "trace_receipts", "traces"}
    for suite in SUITES:
        directories.update(
            {
                f"results/{suite}",
                f"trace_receipts/{suite}",
                f"traces/{suite}",
            }
        )
        for task_id in range(TASKS_PER_SUITE):
            directories.add(f"traces/{suite}/task{task_id:02d}")
    return directories


def _scan_tree(root: Path) -> tuple[set[str], set[str]]:
    root_fd = _open_absolute_nofollow(root, directory=True)
    files: set[str] = set()
    directories: set[str] = set()
    identities: set[tuple[int, int]] = set()

    def visit(directory_fd: int, prefix: PurePosixPath) -> None:
        try:
            names = sorted(os.listdir(directory_fd), key=lambda value: value.encode("utf-8"))
        except OSError as exc:
            raise SealError(f"cannot enumerate artifact directory {prefix}: {exc}") from exc
        for name in names:
            if name in (".", "..") or "/" in name or "\x00" in name:
                raise SealError(f"unsafe artifact name: {name!r}")
            relative = prefix / name
            try:
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise SealError(f"cannot stat artifact {relative}: {exc}") from exc
            identity = (metadata.st_dev, metadata.st_ino)
            if stat.S_ISLNK(metadata.st_mode):
                raise SealError(f"symlink is forbidden in artifact root: {relative}")
            if identity in identities:
                raise SealError(f"hardlink/inode alias is forbidden: {relative}")
            identities.add(identity)
            text = relative.as_posix()
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(text)
                try:
                    child_fd = os.open(
                        name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                except OSError as exc:
                    raise SealError(f"cannot open artifact directory {relative}: {exc}") from exc
                try:
                    visit(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise SealError(f"hardlinked artifact is forbidden: {relative}")
                files.add(text)
            else:
                raise SealError(f"non-regular artifact is forbidden: {relative}")

    try:
        visit(root_fd, PurePosixPath())
    finally:
        os.close(root_fd)
    return files, directories


def _require_exact_tree(root: Path, expected_files: set[str]) -> None:
    files, directories = _scan_tree(root)
    if files != expected_files:
        missing = sorted(expected_files - files)
        extra = sorted(files - expected_files)
        raise SealError(
            f"artifact file scope mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    expected_directories = _expected_directories()
    if directories != expected_directories:
        missing = sorted(expected_directories - directories)
        extra = sorted(directories - expected_directories)
        raise SealError(
            f"artifact directory scope mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )


def _tree_sha(files: list[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda entry: str(entry["path"]).encode("utf-8")):
        digest.update(f"{item['sha256']}  {item['path']}\n".encode("utf-8"))
    return digest.hexdigest()


def _inventory_item(root: Path, relative: str, role: str, *, capture: bool = False) -> tuple[dict[str, Any], bytes | None]:
    readback = _read_relative(root, relative, capture=capture)
    item = {
        "path": relative,
        "role": role,
        "sha256": readback["sha256"],
        "size_bytes": readback["size_bytes"],
    }
    return item, readback.get("bytes")


def _inventory_item_from_bytes(relative: str, role: str, raw: bytes) -> dict[str, Any]:
    """Build the exact inventory row that publishing *raw* will create."""

    return {
        "path": relative,
        "role": role,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value) or value == "0" * 64:
        raise SealError(f"{label} must be a non-placeholder lowercase SHA-256")
    return value


def _require_exact_keys(value: Any, keys: set[str] | frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        observed = set(value) if isinstance(value, Mapping) else set()
        raise SealError(
            f"{label} schema mismatch: missing={sorted(set(keys)-observed)}, "
            f"extra={sorted(observed-set(keys))}"
        )
    return value


def _require_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SealError(f"{label} must be a non-empty relative POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise SealError(f"{label} is not a canonical relative POSIX path: {value!r}")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise SealError(f"{label} must be a positive integer")
    return value


def _parse_timestamp(value: Any, label: str) -> dt.datetime:
    if not isinstance(value, str) or not value:
        raise SealError(f"{label} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SealError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SealError(f"{label} must include a timezone offset")
    return parsed


def _numeric_vector(value: Any, length: int, label: str) -> list[float | int]:
    if not isinstance(value, list) or len(value) != length:
        raise SealError(f"{label} must be a numeric vector of length {length}")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise SealError(f"{label} contains a non-finite/non-numeric value")
    return value


def _numeric_matrix(value: Any, rows: int | None, columns: int, label: str) -> list[list[float | int]]:
    if not isinstance(value, list) or (rows is not None and len(value) != rows):
        raise SealError(f"{label} has an invalid row count")
    for index, row in enumerate(value):
        _numeric_vector(row, columns, f"{label}[{index}]")
    return value


def _load_external_json(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    readback = _read_absolute(path, capture=True)
    payload = _loads_json(readback["bytes"], label)
    if not isinstance(payload, dict):
        raise SealError(f"{label} must be a JSON object")
    return payload, readback


def _bind_launch_artifact(
    value: str,
    *,
    role: str,
    official_root: Path,
    preregistration: Mapping[str, Any],
    shared_launch_bindings: dict[str, str],
) -> None:
    """Hash one command-selected artifact once and require one path for all tasks."""

    if not value or "\x00" in value:
        raise SealError(f"runner {role} path must be non-empty")
    candidate = Path(value)
    absolute = _lexical_absolute(
        candidate if candidate.is_absolute() else official_root / candidate
    )
    path_key = f"{role}_path"
    previous = shared_launch_bindings.get(path_key)
    if previous is not None:
        if previous != str(absolute):
            raise SealError(f"runner {role} path differs across manager tasks")
        return
    readback = _read_absolute(absolute, capture=False)
    expected = preregistration["artifacts"][role]
    if (
        readback["sha256"] != expected["sha256"]
        or readback["size_bytes"] != expected["size_bytes"]
    ):
        raise SealError(f"runner {role} file differs from preregistration")
    shared_launch_bindings[path_key] = str(absolute)


def _validate_runner_command(
    command_argv: Any,
    *,
    manager_item: Mapping[str, Any],
    scheduled_process: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    artifact_root: Path,
    official_root: Path,
    shared_launch_bindings: dict[str, str],
) -> str:
    """Validate the traced Hydra evaluator command by override semantics."""

    process_id = str(manager_item["process_id"])
    if (
        not isinstance(command_argv, list)
        or any(not isinstance(item, str) or not item for item in command_argv)
        or _canonical_sha(command_argv) != manager_item["command_sha256"]
    ):
        raise SealError(f"manager status command binding is invalid: {process_id}")
    if len(command_argv) < 2:
        raise SealError(f"runner command contract is invalid: {process_id}")
    python_name = PurePosixPath(command_argv[0]).name
    if not _PYTHON_EXECUTABLE_RE.fullmatch(python_name):
        raise SealError(f"runner command contract is invalid: {process_id}")

    runner_text = command_argv[1]
    runner = PurePosixPath(runner_text)
    if any(part in ("", ".", "..") for part in runner.parts):
        raise SealError(f"runner command contract is invalid: {process_id}")
    if runner.is_absolute():
        if runner.parts[-2:] != _TRACED_RUNNER_RELATIVE.parts:
            raise SealError(f"runner command contract is invalid: {process_id}")
        instrumentation_root = str(
            _lexical_absolute(Path(runner_text).parent.parent)
        )
    elif runner != _TRACED_RUNNER_RELATIVE:
        raise SealError(f"runner command contract is invalid: {process_id}")
    else:
        instrumentation_root = preregistration["launch"]["working_directory"]

    overrides: dict[str, str] = {}
    for argument in command_argv[2:]:
        if "=" not in argument:
            raise SealError(f"runner Hydra override is invalid: {process_id}")
        key, value = argument.split("=", 1)
        if not key or not value or key in overrides:
            raise SealError(
                f"runner Hydra override is empty or duplicated: {process_id}"
            )
        overrides[key] = value
    observed_keys = set(overrides)
    if (
        not _REQUIRED_HYDRA_OVERRIDES <= observed_keys
        or observed_keys - _REQUIRED_HYDRA_OVERRIDES - _OPTIONAL_HYDRA_OVERRIDES
    ):
        raise SealError(f"runner Hydra override scope is invalid: {process_id}")
    expected_values = {
        "task": "libero_uncond_2cam224_1e-4",
        "EVALUATION.task_suite_name": str(manager_item["task_suite"]),
        "EVALUATION.task_id": str(manager_item["task_id"]),
        "gpu_id": str(manager_item["gpu_id"]),
        "EVALUATION.num_trials": str(TRIALS_PER_TASK),
        "EVALUATION.output_dir": str(artifact_root),
        "output_dir": str(artifact_root),
        "EVALUATION.env_num": "1",
        "EVALUATION.num_steps_wait": "30",
        "EVALUATION.replan_steps": "10",
        "EVALUATION.action_horizon": "32",
        "EVALUATION.binarize_gripper": "true",
        "seed": str(scheduled_process["global_seed"]),
    }
    if any(overrides.get(key) != value for key, value in expected_values.items()):
        raise SealError(f"runner Hydra override values are invalid: {process_id}")
    if any(overrides[key] != "false" for key in observed_keys & _OPTIONAL_HYDRA_OVERRIDES):
        raise SealError(f"runner optional Hydra override is unsafe: {process_id}")
    _bind_launch_artifact(
        overrides["ckpt"],
        role="checkpoint",
        official_root=official_root,
        preregistration=preregistration,
        shared_launch_bindings=shared_launch_bindings,
    )
    _bind_launch_artifact(
        overrides["EVALUATION.dataset_stats_path"],
        role="dataset_stats",
        official_root=official_root,
        preregistration=preregistration,
        shared_launch_bindings=shared_launch_bindings,
    )
    return instrumentation_root


def _validate_worker_terminal_log(
    raw: bytes,
    *,
    manager_item: Mapping[str, Any],
    run_id: str,
    preregistration: Mapping[str, Any],
    artifact_root: Path,
) -> None:
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        raise SealError(f"manager worker log is empty: {manager_item['process_id']}")
    payload = _loads_json(lines[-1], f"worker terminal {manager_item['process_id']}")
    terminal = _require_exact_keys(
        payload,
        WORKER_TERMINAL_KEYS,
        f"worker terminal {manager_item['process_id']}",
    )
    sources = terminal["terminal_source_identities"]
    official_source = sources.get("official") if isinstance(sources, Mapping) else None
    instrumentation_source = (
        sources.get("instrumentation") if isinstance(sources, Mapping) else None
    )
    expected_process_receipt = str(
        artifact_root / str(manager_item["trace_receipt_path"])
    )
    if (
        terminal["status"] != "PASS"
        or terminal["kind"] != "mf_wam_g0_traced_worker_terminal"
        or terminal["run_id"] != run_id
        or terminal["official_commit"]
        != preregistration["source"]["fastwam"]["commit"]
        or terminal["official_result_type"] != "dict"
        or terminal["process_receipt"] != expected_process_receipt
        or terminal["environment_sha256"] != manager_item["environment_sha256"]
        or terminal["external_prelaunch_commit_tree_gate_required"] is not True
        or not isinstance(sources, Mapping)
        or sources.get("status") != "PASS"
        or not isinstance(official_source, Mapping)
        or official_source.get("commit")
        != preregistration["source"]["fastwam"]["commit"]
        or not isinstance(instrumentation_source, Mapping)
        or instrumentation_source.get("commit")
        != preregistration["source"]["instrumentation"]["commit"]
    ):
        raise SealError(
            f"worker terminal completion binding is invalid: {manager_item['process_id']}"
        )
    result = terminal["official_result_receipt"]
    expected_result = {
        "path": manager_item["result_path"],
        "sha256": manager_item["result_sha256"],
        "size_bytes": manager_item["result_size_bytes"],
        "source_path": manager_item["raw_result_source_path"],
        "source_sha256": manager_item["raw_result_sha256"],
        "source_size_bytes": manager_item["raw_result_size_bytes"],
    }
    if not isinstance(result, Mapping) or any(
        result.get(key) != value for key, value in expected_result.items()
    ):
        raise SealError(
            f"worker terminal result binding is invalid: {manager_item['process_id']}"
        )


def _validate_manager_task_status(
    payload: Any,
    *,
    manager_item: Mapping[str, Any],
    run_id: str,
    scheduled_process: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    artifact_root: Path,
    upstream_paths: Mapping[str, Path],
    upstream_file_sha256: Mapping[str, str],
    shared_launch_bindings: dict[str, str],
) -> None:
    process_id = str(manager_item["process_id"])
    status_payload = _require_exact_keys(
        payload, MANAGER_STATUS_KEYS, f"manager status {process_id}"
    )
    scalar_fields = (
        "process_id", "task_suite", "task_id", "gpu_id", "state",
        "launched_at", "completed_at", "exit_code", "complete",
        "failure_reason", "command_sha256", "environment_sha256",
    )
    if (
        status_payload["schema_version"] != 1
        or status_payload["kind"] != "mf_wam_g0_manager_task_status"
        or status_payload["run_id"] != run_id
        or any(status_payload[field] != manager_item[field] for field in scalar_fields)
    ):
        raise SealError(f"manager status identity differs from manifest: {process_id}")
    environment = status_payload["environment_bindings"]
    if (
        not isinstance(environment, Mapping)
        or set(environment) != _RUNNER_ENVIRONMENT_KEYS
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            for key, value in environment.items()
        )
        or _canonical_sha(environment) != manager_item["environment_sha256"]
    ):
        raise SealError(f"manager status environment binding is invalid: {process_id}")
    official_root_text = environment["MF_WAM_OFFICIAL_ROOT"]
    try:
        official_root = _lexical_absolute(Path(official_root_text))
    except (OSError, TypeError, ValueError) as exc:
        raise SealError(f"runner official root is invalid: {process_id}") from exc
    if str(official_root) != official_root_text:
        raise SealError(f"runner official root is noncanonical: {process_id}")
    expected_environment = {
        **_FIXED_WORKER_ENVIRONMENT,
        "MF_WAM_OFFICIAL_ROOT": str(official_root),
        "MF_WAM_OFFICIAL_COMMIT": preregistration["source"]["fastwam"]["commit"],
        "MF_WAM_G0_RUN_ID": run_id,
        "MF_WAM_INSTRUMENTATION_COMMIT": preregistration["source"][
            "instrumentation"
        ]["commit"],
        "MF_WAM_G0_PREREG_PATH": str(upstream_paths["preregistration"]),
        "MF_WAM_G0_PREREG_SHA256": upstream_file_sha256["preregistration"],
        "MF_WAM_G0_RUNTIME_START_PATH": str(upstream_paths["runtime_start"]),
        "MF_WAM_G0_RUNTIME_START_SHA256": upstream_file_sha256["runtime_start"],
        "MF_WAM_G0_SEED_SCHEDULE_PATH": str(upstream_paths["seed_schedule"]),
        "MF_WAM_G0_SEED_SCHEDULE_SHA256": upstream_file_sha256["seed_schedule"],
        "MF_WAM_G0_RESOLVED_CONFIG_PATH": str(upstream_paths["resolved_config"]),
        "MF_WAM_G0_RESOLVED_CONFIG_SHA256": upstream_file_sha256["resolved_config"],
        "CUDA_VISIBLE_DEVICES": str(manager_item["gpu_id"]),
        "PYTHONHASHSEED": str(scheduled_process["python_hash_seed"]),
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "MUJOCO_GL": preregistration["runtime_environment"]["MUJOCO_GL"],
        "PYOPENGL_PLATFORM": preregistration["runtime_environment"][
            "PYOPENGL_PLATFORM"
        ],
        "DIFFSYNTH_DOWNLOAD_SOURCE": preregistration["runtime_environment"][
            "DIFFSYNTH_DOWNLOAD_SOURCE"
        ],
        "DIFFSYNTH_MODEL_BASE_PATH": preregistration["runtime_environment"][
            "DIFFSYNTH_MODEL_BASE_PATH"
        ],
        "DIFFSYNTH_SKIP_DOWNLOAD": preregistration["runtime_environment"][
            "DIFFSYNTH_SKIP_DOWNLOAD"
        ],
    }
    if dict(environment) != expected_environment:
        raise SealError(
            f"manager status environment differs from formal runner contract: {process_id}"
        )
    instrumentation_root = _validate_runner_command(
        status_payload["command_argv"],
        manager_item=manager_item,
        scheduled_process=scheduled_process,
        preregistration=preregistration,
        artifact_root=artifact_root,
        official_root=official_root,
        shared_launch_bindings=shared_launch_bindings,
    )
    for key, value in (
        ("official_root", str(official_root)),
        ("instrumentation_root", instrumentation_root),
    ):
        previous = shared_launch_bindings.get(key)
        if previous is not None and previous != value:
            raise SealError(f"runner {key} differs across manager tasks")
        shared_launch_bindings[key] = value
    expected_objects = {
        "log": {
            "path": manager_item["log_path"],
            "sha256": manager_item["log_sha256"],
            "size_bytes": manager_item["log_size_bytes"],
        },
        "canonical_result": {
            "path": manager_item["result_path"],
            "sha256": manager_item["result_sha256"],
            "size_bytes": manager_item["result_size_bytes"],
        },
        "trace_receipt": {
            "path": manager_item["trace_receipt_path"],
            "sha256": manager_item["trace_receipt_sha256"],
            "size_bytes": manager_item["trace_receipt_size_bytes"],
            "tree_sha256": manager_item["trace_tree_sha256"],
            "episode_count": manager_item["episode_count"],
        },
        "raw_result": {
            "source_path": manager_item["raw_result_source_path"],
            "archive_path": manager_item["raw_result_archive_path"],
            "sha256": manager_item["raw_result_sha256"],
            "size_bytes": manager_item["raw_result_size_bytes"],
        },
    }
    for field, expected in expected_objects.items():
        observed = status_payload[field]
        if not isinstance(observed, Mapping) or dict(observed) != expected:
            raise SealError(
                f"manager status {field} differs from manifest: {process_id}"
            )


def _validate_manager_manifest(
    payload: Any,
    *,
    trusted_file_sha256: str,
    readback: Mapping[str, Any],
    manifest_path: Path,
    run_id: str,
    artifact_root: Path,
    expected_upstream_bindings: Mapping[str, Any],
    gpu_count: int,
    runtime_started_at: dt.datetime,
    scheduled_by_id: Mapping[str, Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    upstream_paths: Mapping[str, Path],
    upstream_file_sha256: Mapping[str, str],
) -> dict[str, Any]:
    manifest = _require_exact_keys(payload, MANAGER_TOP_KEYS, "manager manifest")
    if readback["sha256"] != _require_sha(trusted_file_sha256, "trusted manager manifest digest"):
        raise SealError("manager manifest does not match its trusted file digest")
    if (
        manifest["schema_version"] != 1
        or manifest["kind"] != "mf_wam_g0_manager_terminal_manifest"
        or manifest["run_id"] != run_id
        or manifest["manager_exit_code"] != 0
        or manifest["artifact_root"] != str(artifact_root)
        or manifest["canonical_input_file_count"] != EXPECTED_INPUT_FILES
    ):
        raise SealError("manager manifest identity/success/input scope is invalid")
    completed_at = _parse_timestamp(manifest["completed_at"], "manager completed_at")
    if completed_at < runtime_started_at:
        raise SealError("manager completion predates runtime start")
    _require_sha(
        manifest["canonical_input_tree_sha256"],
        "manager canonical input tree digest",
    )
    upstream = _require_exact_keys(
        manifest["upstream_bindings"],
        MANAGER_UPSTREAM_KEYS,
        "manager upstream bindings",
    )
    if dict(upstream) != dict(expected_upstream_bindings):
        raise SealError("manager upstream bindings differ from sealed inputs")

    raw_log_text = manifest["raw_log_root"]
    if not isinstance(raw_log_text, str):
        raise SealError("manager raw_log_root must be an absolute path")
    raw_log_root = _lexical_absolute(Path(raw_log_text))
    if str(raw_log_root) != raw_log_text:
        raise SealError("manager raw_log_root must be a canonical absolute path")
    try:
        raw_log_root.relative_to(artifact_root)
    except ValueError:
        pass
    else:
        raise SealError("manager raw_log_root must be outside artifact_root")
    raw_log_fd = _open_absolute_nofollow(raw_log_root, directory=True)
    os.close(raw_log_fd)
    if _lexical_absolute(manifest_path) != raw_log_root / "manager_terminal.json":
        raise SealError("manager manifest must be raw_log_root/manager_terminal.json")

    gpu_ids = manifest["gpu_ids"]
    if (
        not isinstance(gpu_ids, list)
        or any(type(item) is not int or not 0 <= item <= 7 for item in gpu_ids)
        or gpu_ids != sorted(set(gpu_ids))
        or not 1 <= len(gpu_ids) <= 8
        or len(gpu_ids) != gpu_count
    ):
        raise SealError("manager gpu_ids must be a sorted unique list matching the GPU count")
    processes = manifest["task_processes"]
    if not isinstance(processes, list) or len(processes) != EXPECTED_TASKS:
        raise SealError("manager manifest must contain exactly 40 task processes")
    by_id: dict[str, dict[str, Any]] = {}
    external_paths: set[str] = set()
    shared_launch_bindings: dict[str, str] = {}
    expected_order = [
        (suite, task_id)
        for suite in SUITES
        for task_id in range(TASKS_PER_SUITE)
    ]
    for index, (raw, expected_identity) in enumerate(zip(processes, expected_order)):
        item = dict(
            _require_exact_keys(raw, MANAGER_PROCESS_KEYS, f"manager task {index}")
        )
        suite = item.get("task_suite")
        task_id = item.get("task_id")
        expected_suite, expected_task_id = expected_identity
        expected_id = f"{expected_suite}/task{expected_task_id:02d}"
        launched_at = _parse_timestamp(
            item.get("launched_at"), f"manager task {expected_id} launched_at"
        )
        task_completed_at = _parse_timestamp(
            item.get("completed_at"), f"manager task {expected_id} completed_at"
        )
        if (
            suite != expected_suite
            or task_id != expected_task_id
            or item.get("process_id") != expected_id
            or expected_id in by_id
            or item.get("gpu_id") not in gpu_ids
            or item.get("state") != "SUCCEEDED"
            or launched_at < runtime_started_at
            or task_completed_at < launched_at
            or task_completed_at > completed_at
            or item.get("exit_code") != 0
            or item.get("complete") is not True
            or item.get("failure_reason") is not None
            or item.get("episode_count") != TRIALS_PER_TASK
        ):
            raise SealError(f"manager task process is incomplete or invalid: {item}")
        for field in (
            "command_sha256", "environment_sha256", "log_sha256",
            "status_sha256", "result_sha256", "trace_receipt_sha256",
            "trace_tree_sha256", "raw_result_sha256",
        ):
            _require_sha(item[field], f"manager task {expected_id} {field}")
        for field in (
            "log_size_bytes", "status_size_bytes", "result_size_bytes",
            "trace_receipt_size_bytes", "raw_result_size_bytes",
        ):
            _require_positive_int(item[field], f"manager task {expected_id} {field}")

        gpu_id = item["gpu_id"]
        expected_paths = {
            "log_path": f"logs/{suite}/task{task_id:02d}.log",
            "status_path": f"status/{suite}/task{task_id:02d}.json",
            "result_path": f"results/{suite}/task{task_id:02d}.json",
            "trace_receipt_path": (
                f"trace_receipts/{suite}/task{task_id:02d}.json"
            ),
            "raw_result_source_path": (
                f"{suite}/gpu{gpu_id}_task{task_id}_results.json"
            ),
            "raw_result_archive_path": (
                f"official/{suite}/gpu{gpu_id}_task{task_id}_results.json"
            ),
        }
        for field, expected in expected_paths.items():
            observed = _require_relative_path(
                item[field], f"manager task {expected_id} {field}"
            )
            if observed != expected:
                raise SealError(
                    f"manager task {expected_id} has noncanonical {field}: {observed}"
                )
        worker_log_bytes: bytes | None = None
        for path_field, sha_field, size_field in (
            ("log_path", "log_sha256", "log_size_bytes"),
            ("status_path", "status_sha256", "status_size_bytes"),
            (
                "raw_result_archive_path", "raw_result_sha256",
                "raw_result_size_bytes",
            ),
        ):
            relative = _require_relative_path(
                item[path_field], f"manager task {expected_id} {path_field}"
            )
            if relative in external_paths:
                raise SealError(f"duplicate manager external artifact path: {relative}")
            external_paths.add(relative)
            external = _read_relative(
                raw_log_root,
                relative,
                capture=path_field in ("log_path", "status_path"),
            )
            if (
                external["sha256"] != item[sha_field]
                or external["size_bytes"] != item[size_field]
            ):
                raise SealError(
                    f"manager task {expected_id} external artifact mismatch: {relative}"
                )
            if path_field == "log_path":
                worker_log_bytes = external.get("bytes", b"")
            elif path_field == "status_path":
                status_payload = _loads_json(
                    external.get("bytes", b""), f"manager status {expected_id}"
                )
                _validate_manager_task_status(
                    status_payload,
                    manager_item=item,
                    run_id=run_id,
                    scheduled_process=scheduled_by_id[expected_id],
                    preregistration=preregistration,
                    artifact_root=artifact_root,
                    upstream_paths=upstream_paths,
                    upstream_file_sha256=upstream_file_sha256,
                    shared_launch_bindings=shared_launch_bindings,
                )
        if worker_log_bytes is None:
            raise SealError(f"worker log was not captured: {expected_id}")
        _validate_worker_terminal_log(
            worker_log_bytes,
            manager_item=item,
            run_id=run_id,
            preregistration=preregistration,
            artifact_root=artifact_root,
        )
        by_id[expected_id] = item
    required_launch_bindings = {
        "official_root", "instrumentation_root", "checkpoint_path",
        "dataset_stats_path",
    }
    if not required_launch_bindings <= set(shared_launch_bindings):
        raise SealError("manager runner launch bindings are incomplete")
    return {
        "completed_at": manifest["completed_at"],
        "canonical_input_tree_sha256": manifest["canonical_input_tree_sha256"],
        "raw_log_root": raw_log_root,
        "gpu_ids": list(gpu_ids),
        "task_processes": by_id,
        "official_root": shared_launch_bindings["official_root"],
        "instrumentation_root": shared_launch_bindings["instrumentation_root"],
    }


def _validate_task_result(payload: Any, *, suite: str, task_id: int) -> dict[int, bool]:
    required = {
        "task_suite", "task_id", "task_description", "successes", "total_episodes",
        "gpu_id", "success_episodes", "failure_episodes", "start_time", "duration",
    }
    optional = {"episode_future_video_psnr", "future_video_psnr_mean"}
    if not isinstance(payload, Mapping) or not required <= set(payload) or set(payload) - required - optional:
        raise SealError(f"task result schema is invalid for {suite}/task{task_id:02d}")
    successes = payload["success_episodes"]
    failures = payload["failure_episodes"]
    if (
        payload["task_suite"] != suite
        or payload["task_id"] != task_id
        or payload["total_episodes"] != TRIALS_PER_TASK
        or type(payload["successes"]) is not int
        or type(payload["gpu_id"]) is not int
        or not isinstance(payload["task_description"], str)
        or not payload["task_description"].strip()
        or not isinstance(payload["start_time"], str)
        or not payload["start_time"].strip()
        or not isinstance(successes, list)
        or not isinstance(failures, list)
        or any(type(item) is not int for item in successes + failures)
        or successes != sorted(successes)
        or failures != sorted(failures)
        or len(set(successes)) != len(successes)
        or len(set(failures)) != len(failures)
        or set(successes) & set(failures)
        or set(successes) | set(failures) != set(range(TRIALS_PER_TASK))
        or payload["successes"] != len(successes)
        or isinstance(payload["duration"], bool)
        or not isinstance(payload["duration"], (int, float))
        or not math.isfinite(float(payload["duration"]))
        or float(payload["duration"]) < 0
    ):
        raise SealError(f"task result outcome partition is invalid for {suite}/task{task_id:02d}")
    return {trial_idx: trial_idx in set(successes) for trial_idx in range(TRIALS_PER_TASK)}


def _validate_trace(
    payload: Any,
    *,
    suite: str,
    task_id: int,
    trial_idx: int,
    expected_success: bool,
    scheduled_process: Mapping[str, Any],
    upstream_digests: Mapping[str, str],
    run_id: str,
    task_description: str,
    official_commit: str,
    instrumentation_commit: str,
    official_root: str,
    instrumentation_root: str,
) -> None:
    trace = _require_exact_keys(payload, TRACE_TOP_KEYS, "episode trace")
    if trace["schema_version"] != 2 or trace["kind"] != "mf_wam_g0_structured_trace":
        raise SealError("episode trace is not schema v2")
    metadata = _require_exact_keys(trace["metadata"], TRACE_METADATA_KEYS, "trace metadata")
    if (
        metadata["run_id"] != run_id
        or metadata["task_suite"] != suite
        or metadata["task_id"] != task_id
        or metadata["trial_idx"] != trial_idx
        or metadata["initial_state_index"] != trial_idx
        or metadata["success"] is not expected_success
        or metadata["warmup_steps"] != 30
        or metadata["first_replan_env_step"] != 30
        or metadata["replan_steps"] != 10
        or metadata["action_horizon"] != 32
        or metadata["action_dimension"] != 7
        or metadata["state_dimension"] != 8
        or metadata["seed_schedule_process"] != scheduled_process
        or metadata["upstream_digests"] != upstream_digests
        or metadata["task_description"] != task_description
        or type(metadata["record_count"]) is not int
        or type(metadata["environment_step_count"]) is not int
        or type(metadata["observer_rng_unchanged_checks"]) is not int
        or metadata["observer_rng_unchanged_checks"] < 1
    ):
        raise SealError(f"trace metadata contract mismatch at {suite}/task{task_id:02d}/trial{trial_idx:03d}")
    _require_sha(metadata["initial_state_sha256"], "trace initial state digest")
    _require_sha(
        metadata["official_module_origin_inventory_sha256"],
        "trace module-origin digest",
    )
    source_expectations = (
        (
            "official_source", "official_policy_and_evaluator_source",
            official_commit, official_root,
        ),
        (
            "instrumentation_source", "external_observer_and_launcher_source",
            instrumentation_commit, instrumentation_root,
        ),
    )
    for field, role, expected_commit, expected_root in source_expectations:
        identity = metadata[field]
        if (
            not isinstance(identity, Mapping)
            or identity.get("status") != "PASS"
            or identity.get("role") != role
            or identity.get("commit") != expected_commit
            or identity.get("clean") is not True
            or identity.get("root") != expected_root
        ):
            raise SealError(f"trace {field} identity is incomplete or mismatched")
    seed_contract = metadata["seed_contract"]
    expected_seed_contract = {
        "task_seed": scheduled_process["global_seed"],
        "effective_global_rank": 0,
        "effective_process_seed": scheduled_process["global_seed"],
        "task_seed_scope": "once_per_task_process_before_model_and_benchmark_construction",
        "environment_seed": scheduled_process["environment_seed"],
        "environment_seed_scope": "once_per_task_process_before_trial_loop",
        "policy_seed": scheduled_process["policy_seed"],
        "policy_seed_scope": "fresh_generator_per_replan",
        "episode_rng_position": "ordered_trial_index_in_shared_task_environment_stream",
    }
    if seed_contract != expected_seed_contract:
        raise SealError("trace seed contract differs from scheduled process")
    records = trace["records"]
    if not isinstance(records, list) or len(records) < 7 or metadata["record_count"] != len(records):
        raise SealError("trace record coverage is invalid")
    total_executed = 0
    any_done = False
    for index, raw_record in enumerate(records):
        record = _require_exact_keys(raw_record, TRACE_RECORD_KEYS, f"trace record {index}")
        if (
            record["episode_idx"] != trial_idx
            or record["replan_idx"] != index
            or record["env_step"] != 30 + index * 10
            or record["policy_seed"] != scheduled_process["policy_seed"]
            or record["policy_seed_scope"] != "fresh_generator_per_replan"
            or type(record["executed_count"]) is not int
            or type(record["done_after_execution"]) is not bool
        ):
            raise SealError("trace record identity/cadence is invalid")
        _numeric_vector(record["state"], 8, "record.state")
        _numeric_vector(record["pre_state"], 8, "record.pre_state")
        _require_sha(record["pre_observation_sha256"], "pre observation digest")
        _numeric_matrix(record["proposed_raw_action_chunk"], 32, 7, "raw proposal")
        proposal = _numeric_matrix(record["proposed_env_action_chunk"], 32, 7, "env proposal")
        executed = _numeric_matrix(record["executed_env_actions"], None, 7, "executed prefix")
        if not 1 <= len(executed) <= 10 or record["executed_count"] != len(executed):
            raise SealError("executed action count is invalid")
        if executed != proposal[: len(executed)]:
            raise SealError("executed actions do not equal the environment proposal prefix")
        executions = record["executions"]
        if not isinstance(executions, list) or len(executions) != len(executed):
            raise SealError("trace execution list length is invalid")
        for execution_idx, raw_execution in enumerate(executions):
            execution = _require_exact_keys(
                raw_execution, TRACE_EXECUTION_KEYS, f"trace execution {execution_idx}"
            )
            _numeric_vector(execution["action"], 7, "execution.action")
            _numeric_vector(execution["post_state"], 8, "execution.post_state")
            _require_sha(execution["post_observation_sha256"], "post observation digest")
            if execution["action"] != executed[execution_idx] or type(execution["done"]) is not bool:
                raise SealError("trace execution content is invalid")
        done_indices = [
            execution_idx
            for execution_idx, execution in enumerate(executions)
            if execution["done"]
        ]
        done = bool(done_indices)
        if record["done_after_execution"] is not executions[-1]["done"]:
            raise SealError("trace done_after_execution mismatch")
        if done and (
            index != len(records) - 1
            or done_indices != [len(executions) - 1]
        ):
            raise SealError("trace continued after an environment terminal step")
        if index < len(records) - 1 and (len(executed) != 10 or done):
            raise SealError("non-final replan is incomplete or terminal")
        total_executed += len(executed)
        any_done = any_done or done
    if metadata["environment_step_count"] != 30 + total_executed:
        raise SealError("trace environment_step_count mismatch")
    if any_done is not expected_success or records[-1]["done_after_execution"] is not expected_success:
        raise SealError("trace success/done identity differs from task result")


def _validate_task_receipt(
    payload: Any,
    *,
    run_id: str,
    suite: str,
    task_id: int,
    scheduled_process: Mapping[str, Any],
    result_item: Mapping[str, Any],
    trace_items: list[Mapping[str, Any]],
    prereg: Mapping[str, Any],
    start: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> str:
    keys = {
        "schema_version", "kind", "run_id", "process_id", "task_suite", "task_id",
        "execution_scope", "world_size", "global_rank", "local_rank", "bindings",
        "seeds", "official_result", "episode_count", "traces", "tree_sha256",
    }
    receipt = _require_exact_keys(payload, keys, "task trace receipt")
    process_id = f"{suite}/task{task_id:02d}"
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "mf_wam_g0_task_trace_receipt"
        or receipt["run_id"] != run_id
        or receipt["process_id"] != process_id
        or receipt["task_suite"] != suite
        or receipt["task_id"] != task_id
        or receipt["execution_scope"] != "one-process-per-task"
        or receipt["world_size"] != 1
        or receipt["global_rank"] != 0
        or receipt["local_rank"] != 0
        or receipt["episode_count"] != TRIALS_PER_TASK
    ):
        raise SealError(f"task trace receipt identity mismatch: {process_id}")
    expected_bindings = {
        "preregistration_canonical_sha256": _canonical_sha(prereg),
        "runtime_start_canonical_sha256": _canonical_sha(start),
        "seed_schedule_canonical_sha256": _canonical_sha(schedule),
        "resolved_config_sha256": prereg["artifacts"]["resolved_config"]["sha256"],
        "image_digest": prereg["image"]["digest"],
        "fastwam_commit": prereg["source"]["fastwam"]["commit"],
        "instrumentation_commit": prereg["source"]["instrumentation"]["commit"],
    }
    if receipt["bindings"] != expected_bindings:
        raise SealError(f"task receipt upstream bindings mismatch: {process_id}")
    seed_fields = (
        "global_seed", "environment_seed", "environment_seed_scope", "policy_seed",
        "policy_seed_scope", "python_hash_seed", "trial_order", "initial_state_index_rule",
    )
    expected_seeds = {field: scheduled_process[field] for field in seed_fields}
    if receipt["seeds"] != expected_seeds:
        raise SealError(f"task receipt seed binding mismatch: {process_id}")
    expected_result = {
        key: result_item[key] for key in ("path", "sha256", "size_bytes")
    }
    if receipt["official_result"] != expected_result:
        raise SealError(f"task receipt result binding mismatch: {process_id}")
    expected_traces = [
        {
            "trial_idx": trial_idx,
            "path": item["path"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for trial_idx, item in enumerate(trace_items)
    ]
    if receipt["traces"] != expected_traces:
        raise SealError(f"task receipt trace inventory mismatch: {process_id}")
    expected_tree = _tree_sha(expected_traces)
    if receipt["tree_sha256"] != expected_tree:
        raise SealError(f"task receipt trace tree mismatch: {process_id}")
    return expected_tree


def _csv_bytes(rows: list[list[str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _rate_text(successes: int, trials: int) -> str:
    return f"{100.0 * successes / trials:.2f}"


def _summary_payloads(
    *,
    run_id: str,
    outcomes: Mapping[tuple[str, int, int], bool],
    task_metadata: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, bytes]:
    task_rows = [["Task", "Description", "Success Rate (%)"]]
    suite_stats: dict[str, dict[str, Any]] = {}
    task_results: dict[str, dict[str, Any]] = {}
    total_successes = 0
    total_time = 0.0
    for suite in SUITES:
        suite_successes = 0
        suite_time = 0.0
        suite_durations: list[float] = []
        for task_id in range(TASKS_PER_SUITE):
            successes = sum(
                int(outcomes[(suite, task_id, trial_idx)])
                for trial_idx in range(TRIALS_PER_TASK)
            )
            metadata = task_metadata[(suite, task_id)]
            description = str(metadata["task_description"])
            duration = float(metadata["duration"])
            suite_successes += successes
            suite_time += duration
            suite_durations.append(duration)
            task_rows.append(
                [
                    f"{suite}_{task_id}", description,
                    _rate_text(successes, TRIALS_PER_TASK),
                ]
            )
            task_results[f"{suite}_{task_id}"] = {
                "task_description": description,
                "total_episodes": TRIALS_PER_TASK,
                "successes": successes,
                "success_rate": 100.0 * successes / TRIALS_PER_TASK,
                "duration": duration,
            }
        total_successes += suite_successes
        total_time += suite_time
        suite_stats[suite] = {
            "total_tasks": TASKS_PER_SUITE,
            "total_trials": TASKS_PER_SUITE * TRIALS_PER_TASK,
            "total_successes": suite_successes,
            "success_rate": suite_successes / (TASKS_PER_SUITE * TRIALS_PER_TASK),
            "total_time": suite_time,
            "max_time": max(suite_durations),
        }
    suite_rate_values = [
        100.0 * suite_stats[suite]["total_successes"]
        / suite_stats[suite]["total_trials"]
        for suite in SUITES
    ]
    suite_percentages = [f"{value:.2f}" for value in suite_rate_values]
    overall_percent = sum(suite_rate_values) / len(SUITES)
    summary_rows = [
        ["MF-WAM G0 Evaluation Summary"],
        ["", *SUITES, "Overall"],
        [
            "Success Rate (%)",
            *suite_percentages,
            f"{overall_percent:.2f}",
        ],
    ]
    summary_json = {
        "schema_version": 1,
        "kind": "mf_wam_g0_summary",
        "run_id": run_id,
        "total_tasks": EXPECTED_TASKS,
        "total_trials": EXPECTED_EPISODES,
        "total_successes": total_successes,
        "overall_success_rate": total_successes / EXPECTED_EPISODES,
        "suite_stats": suite_stats,
        "task_results": task_results,
        "overall": {
            "average_success_rate": overall_percent,
            "total_time": total_time,
            "average_task_time": total_time / EXPECTED_TASKS,
        },
    }
    return {
        "summary.csv": _csv_bytes(summary_rows),
        "task_success_rates.csv": _csv_bytes(task_rows),
        "summary.json": _canonical_bytes(summary_json),
    }


def seal_terminal_bundle(
    *,
    artifact_root: Path,
    preregistration_path: Path,
    runtime_start_path: Path,
    seed_schedule_path: Path,
    resolved_config_path: Path,
    task_map_path: Path,
    manager_manifest_path: Path,
    trusted_manager_manifest_sha256: str,
    terminal_output: Path,
) -> dict[str, Any]:
    """Seal terminal artifacts only; this is not full contract-chain validation."""

    artifact_root = _lexical_absolute(artifact_root)
    terminal_output = _lexical_absolute(terminal_output)
    preregistration_path = _lexical_absolute(preregistration_path)
    runtime_start_path = _lexical_absolute(runtime_start_path)
    seed_schedule_path = _lexical_absolute(seed_schedule_path)
    resolved_config_path = _lexical_absolute(resolved_config_path)
    task_map_path = _lexical_absolute(task_map_path)
    manager_manifest_path = _lexical_absolute(manager_manifest_path)
    try:
        terminal_output.relative_to(artifact_root)
    except ValueError:
        pass
    else:
        raise SealError("terminal output must be outside artifact_root")
    if terminal_output == artifact_root:
        raise SealError("terminal output must be outside artifact_root")
    _ensure_absolute_absent(terminal_output)

    prereg_raw, prereg_file = _load_external_json(preregistration_path, "preregistration")
    start_raw, start_file = _load_external_json(runtime_start_path, "runtime start")
    schedule_raw, schedule_file = _load_external_json(seed_schedule_path, "seed schedule")
    task_map_raw, _ = _load_external_json(task_map_path, "task map")
    manager_raw, manager_file = _load_external_json(manager_manifest_path, "manager manifest")

    try:
        task_map = contract.validate_task_map(task_map_raw)
        prereg = contract.validate_preregistration(prereg_raw)
        start = contract.validate_runtime_start(
            start_raw,
            preregistration=prereg,
            model_cache_root=Path(
                prereg["runtime_environment"]["DIFFSYNTH_MODEL_BASE_PATH"]
            ),
        )
        schedule = contract.validate_seed_schedule(schedule_raw, task_map=task_map)
    except contract.ContractError as exc:
        raise SealError(f"upstream contract validation failed: {exc}") from exc
    if _canonical_sha(schedule) != prereg["seeds"]["schedule_canonical_sha256"]:
        raise SealError("seed schedule differs from preregistration")
    if str(artifact_root) != prereg["output"]["artifact_root"]:
        raise SealError("artifact_root differs from preregistration")
    resolved_config_file = _read_absolute(resolved_config_path, capture=False)
    expected_resolved_config = prereg["artifacts"]["resolved_config"]
    if (
        resolved_config_file["sha256"] != expected_resolved_config["sha256"]
        or resolved_config_file["size_bytes"] != expected_resolved_config["size_bytes"]
    ):
        raise SealError("resolved config differs from preregistration")
    upstream_digests = {
        "preregistration_file_sha256": prereg_file["sha256"],
        "preregistration_canonical_sha256": _canonical_sha(prereg),
        "runtime_start_file_sha256": start_file["sha256"],
        "runtime_start_canonical_sha256": _canonical_sha(start),
        "seed_schedule_file_sha256": schedule_file["sha256"],
        "seed_schedule_canonical_sha256": _canonical_sha(schedule),
        "resolved_config_sha256": resolved_config_file["sha256"],
    }
    if set(upstream_digests) != TRACE_UPSTREAM_KEYS:
        raise AssertionError("internal upstream binding schema mismatch")
    expected_manager_upstream = {
        "preregistration_file_sha256": prereg_file["sha256"],
        "runtime_start_file_sha256": start_file["sha256"],
        "seed_schedule_file_sha256": schedule_file["sha256"],
        "resolved_config_sha256": resolved_config_file["sha256"],
        "official_commit": prereg["source"]["fastwam"]["commit"],
        "instrumentation_commit": prereg["source"]["instrumentation"]["commit"],
        "python_hash_seed": schedule["python_hash_seed"],
    }
    scheduled_by_id = {
        item["process_id"]: item for item in schedule["task_processes"]
    }
    upstream_paths = {
        "preregistration": preregistration_path,
        "runtime_start": runtime_start_path,
        "seed_schedule": seed_schedule_path,
        "resolved_config": resolved_config_path,
    }
    upstream_file_sha256 = {
        "preregistration": prereg_file["sha256"],
        "runtime_start": start_file["sha256"],
        "seed_schedule": schedule_file["sha256"],
        "resolved_config": resolved_config_file["sha256"],
    }
    manager = _validate_manager_manifest(
        manager_raw,
        trusted_file_sha256=trusted_manager_manifest_sha256,
        readback=manager_file,
        manifest_path=manager_manifest_path,
        run_id=prereg["run_id"],
        artifact_root=artifact_root,
        expected_upstream_bindings=expected_manager_upstream,
        gpu_count=prereg["launch"]["gpu_count"],
        runtime_started_at=_parse_timestamp(start["observed_at"], "runtime start"),
        scheduled_by_id=scheduled_by_id,
        preregistration=prereg,
        upstream_paths=upstream_paths,
        upstream_file_sha256=upstream_file_sha256,
    )
    completed_at = manager["completed_at"]
    manager_processes = manager["task_processes"]

    expected_inputs = _expected_input_paths()
    _require_exact_tree(artifact_root, expected_inputs)

    inventory_files: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    task_processes: list[dict[str, Any]] = []
    outcomes: dict[tuple[str, int, int], bool] = {}
    task_metadata: dict[tuple[str, int], dict[str, Any]] = {}
    for suite in SUITES:
        for task_id in range(TASKS_PER_SUITE):
            process_id = f"{suite}/task{task_id:02d}"
            if process_id not in manager_processes or process_id not in scheduled_by_id:
                raise SealError(f"missing manager/seed process: {process_id}")
            scheduled_process = scheduled_by_id[process_id]
            result_path = f"results/{suite}/task{task_id:02d}.json"
            result_item, result_bytes = _inventory_item(
                artifact_root, result_path, "task_result", capture=True
            )
            result_payload = _loads_json(result_bytes or b"", result_path)
            task_outcomes = _validate_task_result(result_payload, suite=suite, task_id=task_id)
            task_metadata[(suite, task_id)] = {
                "task_description": result_payload["task_description"],
                "duration": float(result_payload["duration"]),
            }
            manager_process = manager_processes[process_id]
            expected_manager_result = {
                "result_path": result_item["path"],
                "result_sha256": result_item["sha256"],
                "result_size_bytes": result_item["size_bytes"],
                "raw_result_sha256": result_item["sha256"],
                "raw_result_size_bytes": result_item["size_bytes"],
            }
            if (
                result_payload["gpu_id"] != manager_process["gpu_id"]
                or any(
                    manager_process[field] != expected
                    for field, expected in expected_manager_result.items()
                )
            ):
                raise SealError(f"manager result binding mismatch: {process_id}")
            result_rows.append(result_item)
            inventory_files.append(result_item)
            trace_items: list[dict[str, Any]] = []
            for trial_idx in range(TRIALS_PER_TASK):
                trace_path = f"traces/{suite}/task{task_id:02d}/trial{trial_idx:03d}.json"
                trace_item, trace_bytes = _inventory_item(
                    artifact_root, trace_path, "episode_trace", capture=True
                )
                trace_payload = _loads_json(trace_bytes or b"", trace_path)
                _validate_trace(
                    trace_payload,
                    suite=suite,
                    task_id=task_id,
                    trial_idx=trial_idx,
                    expected_success=task_outcomes[trial_idx],
                    scheduled_process=scheduled_process,
                    upstream_digests=upstream_digests,
                    run_id=prereg["run_id"],
                    task_description=result_payload["task_description"],
                    official_commit=prereg["source"]["fastwam"]["commit"],
                    instrumentation_commit=prereg["source"]["instrumentation"][
                        "commit"
                    ],
                    official_root=manager["official_root"],
                    instrumentation_root=manager["instrumentation_root"],
                )
                outcomes[(suite, task_id, trial_idx)] = task_outcomes[trial_idx]
                trace_items.append(trace_item)
                trace_rows.append(trace_item)
                inventory_files.append(trace_item)
            receipt_path = f"trace_receipts/{suite}/task{task_id:02d}.json"
            receipt_item, receipt_bytes = _inventory_item(
                artifact_root, receipt_path, "task_trace_receipt", capture=True
            )
            receipt_payload = _loads_json(receipt_bytes or b"", receipt_path)
            trace_tree = _validate_task_receipt(
                receipt_payload,
                run_id=prereg["run_id"],
                suite=suite,
                task_id=task_id,
                scheduled_process=scheduled_process,
                result_item=result_item,
                trace_items=trace_items,
                prereg=prereg,
                start=start,
                schedule=schedule,
            )
            expected_manager_receipt = {
                "trace_receipt_path": receipt_item["path"],
                "trace_receipt_sha256": receipt_item["sha256"],
                "trace_receipt_size_bytes": receipt_item["size_bytes"],
                "trace_tree_sha256": trace_tree,
            }
            if any(
                manager_process[field] != expected
                for field, expected in expected_manager_receipt.items()
            ):
                raise SealError(f"manager trace-receipt binding mismatch: {process_id}")
            inventory_files.append(receipt_item)
            task_processes.append(
                {
                    "process_id": process_id,
                    "task_suite": suite,
                    "task_id": task_id,
                    "execution_scope": "one-process-per-task",
                    "world_size": 1,
                    "global_rank": 0,
                    "local_rank": 0,
                    "exit_code": manager_processes[process_id]["exit_code"],
                    "result_path": result_path,
                    "result_sha256": result_item["sha256"],
                    "result_size_bytes": result_item["size_bytes"],
                    "trace_receipt_path": receipt_path,
                    "trace_receipt_sha256": receipt_item["sha256"],
                    "trace_receipt_size_bytes": receipt_item["size_bytes"],
                    "trace_tree_sha256": trace_tree,
                    "episode_count": TRIALS_PER_TASK,
                    "complete": True,
                }
            )
    if len(inventory_files) != EXPECTED_INPUT_FILES or len(outcomes) != EXPECTED_EPISODES:
        raise SealError("recomputed input scope is incomplete")
    if _tree_sha(inventory_files) != manager["canonical_input_tree_sha256"]:
        raise SealError("manager canonical input tree differs from the exact 2080 inputs")

    summary_payloads = _summary_payloads(
        run_id=prereg["run_id"], outcomes=outcomes, task_metadata=task_metadata
    )
    for relative, raw in summary_payloads.items():
        role = {
            "summary.csv": "summary_csv",
            "task_success_rates.csv": "task_success_rates_csv",
            "summary.json": "summary_json",
        }[relative]
        inventory_files.append(_inventory_item_from_bytes(relative, role, raw))

    aggregates = {
        "task_result_tree_sha256": _tree_sha(result_rows),
        "trace_tree_sha256": _tree_sha(trace_rows),
    }
    terminal_without_inventory = {
        "schema_version": 1,
        "kind": "mf_wam_g0_terminal",
        "phase": "TERMINAL",
        "run_id": prereg["run_id"],
        "completed_at": completed_at,
        "preregistration_canonical_sha256": _canonical_sha(prereg),
        "runtime_start_canonical_sha256": _canonical_sha(start),
        "status": "SUCCEEDED",
        "failure_reason": None,
        "manager_exit_code": 0,
        "scope": {
            "task_process_count": EXPECTED_TASKS,
            "episode_count": EXPECTED_EPISODES,
            "complete": True,
        },
        "task_processes": task_processes,
        "artifact_inventory": None,
        "aggregates": aggregates,
    }
    terminal_core = {
        key: value
        for key, value in terminal_without_inventory.items()
        if key != "artifact_inventory"
    }
    completion = {
        "schema_version": 1,
        "kind": "mf_wam_g0_completion_marker",
        "run_id": prereg["run_id"],
        "status": "SUCCEEDED",
        "task_process_count": EXPECTED_TASKS,
        "episode_count": EXPECTED_EPISODES,
        "terminal_core_canonical_sha256": _canonical_sha(terminal_core),
    }
    completion_raw = _canonical_bytes(completion)
    completion_item = _inventory_item_from_bytes(
        "completion.json", "completion_marker", completion_raw
    )
    inventory_files.append(completion_item)
    if len(inventory_files) != EXPECTED_INVENTORY_FILES:
        raise SealError("terminal inventory does not contain exactly 2084 members")
    inventory_files.sort(key=lambda item: item["path"].encode("utf-8"))
    inventory = {
        "schema_version": 1,
        "kind": "mf_wam_g0_terminal_artifact_inventory",
        "algorithm": contract.DATA_TREE_ALGORITHM,
        "file_count": EXPECTED_INVENTORY_FILES,
        "total_size_bytes": sum(item["size_bytes"] for item in inventory_files),
        "files": inventory_files,
        "tree_sha256": _tree_sha(inventory_files),
    }
    inventory_raw = _canonical_bytes(inventory)
    inventory_item = _inventory_item_from_bytes(
        "artifact_inventory.json", "artifact_inventory", inventory_raw
    )
    terminal = {
        **terminal_without_inventory,
        "artifact_inventory": {
            key: inventory_item[key] for key in ("path", "sha256", "size_bytes")
        },
    }
    try:
        normalized_terminal = contract.validate_terminal_receipt(
            terminal,
            preregistration=prereg,
            runtime_start=start,
            seed_schedule=schedule,
            task_map=task_map,
        )
    except contract.ContractError as exc:
        raise SealError(f"pre-publication terminal validation failed: {exc}") from exc

    # All deterministic semantic checks above run before publication.  The
    # completion marker is deliberately the final publication operation.
    for relative, raw in summary_payloads.items():
        _atomic_publish_relative(artifact_root, relative, raw)
    _atomic_publish_relative(artifact_root, "artifact_inventory.json", inventory_raw)
    _atomic_publish_absolute(terminal_output, _canonical_bytes(normalized_terminal))
    terminal_readback = _read_absolute(terminal_output, capture=True)
    terminal_payload = _loads_json(terminal_readback["bytes"], str(terminal_output))
    try:
        second_terminal = contract.validate_terminal_receipt(
            terminal_payload,
            preregistration=prereg,
            runtime_start=start,
            seed_schedule=schedule,
            task_map=task_map,
        )
    except contract.ContractError as exc:
        raise SealError(f"terminal readback validation failed: {exc}") from exc
    if normalized_terminal != second_terminal:
        raise SealError("terminal readback differs from pre-publication validation")

    _atomic_publish_relative(artifact_root, "completion.json", completion_raw)
    expected_final_files = expected_inputs | set(summary_payloads) | {
        "completion.json", "artifact_inventory.json"
    }
    _require_exact_tree(artifact_root, expected_final_files)
    try:
        postwrite_audit = contract._audit_terminal_artifacts(  # noqa: SLF001
            artifact_root=artifact_root,
            terminal=second_terminal,
            preregistration=prereg,
            runtime_start=start,
            seed_schedule=schedule,
        )
    except contract.ContractError as exc:
        raise SealError(f"post-publication structural audit failed: {exc}") from exc
    return {
        "schema_version": 1,
        "kind": "mf_wam_g0_terminal_seal_receipt",
        "status": "STRUCTURAL_PASS_ONLY",
        "specialized_g0_status": "UNCERTAIN",
        "formal_training_allowed": False,
        "run_id": prereg["run_id"],
        "terminal_path": str(terminal_output),
        "terminal_canonical_sha256": _canonical_sha(second_terminal),
        "artifact_inventory_sha256": inventory_item["sha256"],
        "artifact_inventory_tree_sha256": inventory["tree_sha256"],
        "artifact_count": EXPECTED_INVENTORY_FILES,
        "task_process_count": EXPECTED_TASKS,
        "episode_count": EXPECTED_EPISODES,
        "secondary_structural_audit": postwrite_audit,
        "audit_scope": "terminal_artifacts_only",
        "full_contract_chain_validated": False,
        "scientific_audit_required": True,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed MF-WAM G0 terminal sealer. This proves structural "
            "completeness only and never authorizes formal training."
        )
    )
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--runtime-start", required=True, type=Path)
    parser.add_argument("--seed-schedule", required=True, type=Path)
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument("--task-map", required=True, type=Path)
    parser.add_argument("--manager-manifest", required=True, type=Path)
    parser.add_argument("--trusted-manager-manifest-sha256", required=True)
    parser.add_argument("--terminal-output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        receipt = seal_terminal_bundle(
            artifact_root=arguments.artifact_root,
            preregistration_path=arguments.preregistration,
            runtime_start_path=arguments.runtime_start,
            seed_schedule_path=arguments.seed_schedule,
            resolved_config_path=arguments.resolved_config,
            task_map_path=arguments.task_map,
            manager_manifest_path=arguments.manager_manifest,
            trusted_manager_manifest_sha256=(
                arguments.trusted_manager_manifest_sha256
            ),
            terminal_output=arguments.terminal_output,
        )
    except (SealError, contract.ContractError, OSError, ValueError) as exc:
        failure = {
            "schema_version": 1,
            "kind": "mf_wam_g0_terminal_seal_failure",
            "status": "FAIL",
            "error": str(exc),
        }
        sys.stderr.buffer.write(_canonical_bytes(failure) + b"\n")
        return 2
    sys.stdout.buffer.write(_canonical_bytes(receipt) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
