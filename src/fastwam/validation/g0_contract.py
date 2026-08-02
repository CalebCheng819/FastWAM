"""Fail-closed, deterministic contracts for a fresh MF-WAM G0 run.

This module deliberately stops at *structural* verification.  It constructs
and validates the immutable preregistration, runtime-start receipt, terminal
receipt, LIBERO input inventory, and task-process seed schedule.  It does not
turn those receipts into a formal-training authorization: a separate
specialized auditor must still read raw episode artifacts and recompute G0.

Only the Python standard library is used so that the contract can be checked
on a control host before a GPU environment exists.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import stat
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = 1
DATA_TREE_ALGORITHM = "sha256sum-posix-path-v1"
MODEL_CACHE_INVENTORY_ALGORITHM = "model-cache-per-file-sha256-v1"
CANONICAL_JSON_ALGORITHM = "python-json-sort-keys-utf8-v1"
SEED_SCHEDULE_SEMANTICS = "one-process-per-task-sequential-trials-v1"

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASKS_PER_SUITE = 10
TRIALS_PER_TASK = 50
EXPECTED_TASKS = len(SUITES) * TASKS_PER_SUITE
EXPECTED_EPISODES = EXPECTED_TASKS * TRIALS_PER_TASK
EXPECTED_DATA_FILES = EXPECTED_TASKS * 2

OFFICIAL_FASTWAM_COMMIT = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
OFFICIAL_LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
_ITERATION_ID_RE = re.compile(r"^ITER-[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{3}$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_PROCESS_ID_RE = re.compile(r"^(libero_(?:spatial|object|goal|10))/task([0-9]{2})$")
_EPISODE_ID_RE = re.compile(
    r"^(libero_(?:spatial|object|goal|10))/task([0-9]{2})/trial([0-9]{3})$"
)

_REQUIRED_IMPORTS = frozenset(
    ("fastwam", "libero", "torch", "mujoco", "robosuite", "numpy")
)
_RUNTIME_VERSION_FIELDS = (
    "os_release",
    "kernel",
    "python",
    "torch",
    "torchvision",
    "cuda_runtime",
    "cudnn",
    "nccl",
    "triton",
    "mujoco",
    "libero",
    "robosuite",
    "bddl",
    "hydra",
    "omegaconf",
    "numpy",
    "graphics_pack_version",
)
_RUNTIME_ENVIRONMENT_FIELDS = (
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "DIFFSYNTH_DOWNLOAD_SOURCE",
    "DIFFSYNTH_MODEL_BASE_PATH",
    "DIFFSYNTH_SKIP_DOWNLOAD",
)
_MODEL_CACHE_ROLES = frozenset(
    (
        "text_encoder_weights",
        "vae_weights",
        "tokenizer_config",
        "tokenizer_json",
        "tokenizer_special_tokens_map",
        "tokenizer_model",
    )
)
_TERMINAL_ROLE_COUNTS = {
    "task_result": EXPECTED_TASKS,
    "task_trace_receipt": EXPECTED_TASKS,
    "episode_trace": EXPECTED_EPISODES,
    "summary_csv": 1,
    "task_success_rates_csv": 1,
    "summary_json": 1,
    "completion_marker": 1,
}
EXPECTED_TERMINAL_FILES = sum(_TERMINAL_ROLE_COUNTS.values())


class ContractError(ValueError):
    """Raised when an MF-WAM G0 contract is missing or contradictory."""


def _reject_json_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_json_strict(path: str | Path) -> Any:
    """Load strict UTF-8 JSON, rejecting duplicate keys and NaN/Infinity."""

    input_path = Path(path)
    try:
        raw = input_path.read_bytes()
        return _loads_json_strict_bytes(raw, str(input_path))
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load strict JSON {input_path}: {exc}") from exc


def _loads_json_strict_bytes(raw: bytes, location: str) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except ContractError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load strict JSON {location}: {exc}") from exc


def _assert_json_value(value: Any, location: str = "payload") -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"non-finite numeric value at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_value(item, f"{location}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"non-string object key at {location}")
            _assert_json_value(item, f"{location}.{key}")
        return
    raise ContractError(f"unsupported JSON type at {location}: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the versioned canonical JSON byte representation."""

    _assert_json_value(value)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value cannot be canonically encoded: {exc}") from exc


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _lexical_absolute(path: Path) -> Path:
    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    if ".." in absolute.parts:
        raise ContractError(f"parent traversal is forbidden: {path}")
    return Path(os.path.normpath(str(absolute)))


def _open_or_create_directory_nofollow(path: Path) -> int:
    """Open/create a directory path with pinned ``mkdirat``/``openat`` parents."""

    absolute = _lexical_absolute(path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    odirectory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or odirectory is None:
        raise ContractError(
            "platform lacks O_NOFOLLOW/O_DIRECTORY; refusing unsafe directory creation"
        )
    flags = os.O_RDONLY | odirectory | nofollow | getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(
        "/", os.O_RDONLY | odirectory | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        for component in absolute.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o755, dir_fd=directory_fd)
                except FileExistsError:
                    # Accept a concurrent creator only after a no-follow open
                    # proves the new entry is a real directory.
                    pass
                next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except Exception as exc:
        try:
            os.close(directory_fd)
        except OSError:
            pass
        if isinstance(exc, ContractError):
            raise
        raise ContractError(
            f"cannot create/open directory without following symlinks: {absolute}: {exc}"
        ) from exc


def write_canonical_json(path: str | Path, value: Any) -> None:
    """Atomically create canonical JSON and fail if the target already exists.

    Formal preregistration/runtime/terminal artifacts are append-only.  The
    temporary inode is hard-linked into place, which provides atomic
    no-clobber semantics; unlike ``os.replace`` it can never overwrite an
    existing receipt.
    """

    output_path = _lexical_absolute(Path(path))
    if not output_path.name:
        raise ContractError(f"output path must name a file: {output_path}")
    data = canonical_json_bytes(value)
    parent_fd = _open_or_create_directory_nofollow(output_path.parent)
    temporary_name = f".{output_path.name}.{os.getpid()}.{os.urandom(12).hex()}.tmp"
    descriptor: int | None = None
    temporary_created = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        temporary_created = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary_name,
                output_path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise ContractError(f"refusing to overwrite existing artifact: {output_path}") from exc
        os.unlink(temporary_name, dir_fd=parent_fd)
        temporary_created = False
        os.fsync(parent_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _expect_object(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{location} must be an object")
    return value


def _expect_exact_keys(
    value: Mapping[str, Any], expected: Iterable[str], location: str
) -> None:
    expected_set = set(expected)
    observed_set = set(value)
    if observed_set != expected_set:
        missing = sorted(expected_set - observed_set)
        unexpected = sorted(observed_set - expected_set)
        raise ContractError(
            f"{location} keys do not match schema; missing={missing}, "
            f"unexpected={unexpected}"
        )


def _expect_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ContractError(f"{location} must be a non-empty trimmed string")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ContractError(f"{location} contains a forbidden control character")
    return value


def _expect_int(
    value: Any,
    location: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ContractError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{location} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ContractError(f"{location} must be <= {maximum}")
    return value


def _expect_sha256(value: Any, location: str, *, allow_null: bool = False) -> str | None:
    if value is None and allow_null:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ContractError(f"{location} must be a lowercase SHA-256")
    if value == "0" * 64:
        raise ContractError(f"{location} must not be an all-zero placeholder")
    return value


def _expect_commit(value: Any, location: str) -> str:
    if not isinstance(value, str) or not _GIT_COMMIT_RE.fullmatch(value):
        raise ContractError(f"{location} must be a lowercase 40-hex Git commit")
    if value == "0" * 40:
        raise ContractError(f"{location} must not be an all-zero placeholder")
    return value


def _parse_timestamp(value: Any, location: str) -> dt.datetime:
    text = _expect_nonempty_string(value, location)
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"{location} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError(f"{location} must include a timezone offset")
    return parsed


def _validate_relative_posix_path(value: Any, location: str) -> str:
    text = _expect_nonempty_string(value, location)
    if "\\" in text:
        raise ContractError(f"{location} must use POSIX separators")
    if unicodedata.normalize("NFC", text) != text:
        raise ContractError(f"{location} must use NFC Unicode normalization")
    pure = PurePosixPath(text)
    if pure.is_absolute() or str(pure) != text:
        raise ContractError(f"{location} is not a canonical relative POSIX path")
    if any(part in ("", ".", "..") for part in pure.parts):
        raise ContractError(f"{location} contains an unsafe path component")
    return text


def _validate_absolute_path(value: Any, location: str) -> str:
    text = _expect_nonempty_string(value, location)
    path = PurePosixPath(text)
    if not path.is_absolute() or str(path) != text or ".." in path.parts:
        raise ContractError(f"{location} must be a canonical absolute POSIX path")
    return text


def _validate_runtime_environment(value: Any, location: str) -> dict[str, str]:
    environment = _expect_object(value, location)
    _expect_exact_keys(environment, _RUNTIME_ENVIRONMENT_FIELDS, location)
    normalized = {
        field: _expect_nonempty_string(environment[field], f"{location}.{field}")
        for field in _RUNTIME_ENVIRONMENT_FIELDS
    }
    normalized["DIFFSYNTH_MODEL_BASE_PATH"] = _validate_absolute_path(
        environment["DIFFSYNTH_MODEL_BASE_PATH"],
        f"{location}.DIFFSYNTH_MODEL_BASE_PATH",
    )
    if normalized["DIFFSYNTH_SKIP_DOWNLOAD"] != "true":
        raise ContractError(f"{location}.DIFFSYNTH_SKIP_DOWNLOAD must be exactly 'true'")
    return normalized


def _safe_hash_relative_file(
    root: str | Path,
    relative_path: str,
    *,
    capture_bytes: bool = False,
    maximum_capture_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    """Hash one regular file through no-follow directory descriptors.

    The same open file descriptor supplies the bytes and both metadata samples;
    any observed mutation during the read is rejected.  Every path component is
    opened with ``O_NOFOLLOW`` so a manifest path cannot escape through a
    symlink between validation and open.
    """

    relative = _validate_relative_posix_path(relative_path, "relative_path")
    root_path = Path(root)
    try:
        root_lstat = root_path.lstat()
    except OSError as exc:
        raise ContractError(f"cannot stat data root {root_path}: {exc}") from exc
    if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
        raise ContractError("data root must be a real directory, not a symlink")

    nofollow = getattr(os, "O_NOFOLLOW", None)
    odirectory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or odirectory is None:
        raise ContractError("platform lacks O_NOFOLLOW/O_DIRECTORY; refusing unsafe read")

    directory_flags = os.O_RDONLY | odirectory | nofollow
    file_flags = os.O_RDONLY | nofollow
    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        directory_fd = os.open(root_path, directory_flags)
        parts = PurePosixPath(relative).parts
        for component in parts[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(parts[-1], file_flags, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ContractError(f"input is not a regular file: {relative}")

        digest = hashlib.sha256()
        total = 0
        captured: list[bytes] | None = [] if capture_bytes else None
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if captured is not None:
                if total > maximum_capture_bytes:
                    raise ContractError(
                        f"captured artifact exceeds {maximum_capture_bytes} bytes: {relative}"
                    )
                captured.append(chunk)
        after = os.fstat(file_fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise ContractError(f"input changed while being hashed: {relative}")
        if total != after.st_size:
            raise ContractError(f"short or inconsistent read while hashing: {relative}")
        result = {
            "sha256": digest.hexdigest(),
            "size_bytes": total,
            "filesystem_identity": (after.st_dev, after.st_ino),
            "link_count": after.st_nlink,
        }
        if captured is not None:
            result["bytes"] = b"".join(captured)
        return result
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError(f"cannot safely hash {relative}: {exc}") from exc
    finally:
        if file_fd is not None:
            try:
                os.close(file_fd)
            except OSError:
                pass
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                pass


def _task_sort_key(task: Mapping[str, Any]) -> tuple[int, int]:
    return SUITES.index(str(task["task_suite"])), int(task["task_id"])


def _normalized_task_map(task_map: Any) -> dict[str, Any]:
    payload = _expect_object(task_map, "task_map")
    _expect_exact_keys(payload, ("schema_version", "kind", "tasks"), "task_map")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ContractError("task_map.schema_version is unsupported")
    if payload["kind"] != "mf_wam_g0_task_map":
        raise ContractError("task_map.kind is invalid")
    tasks_raw = payload["tasks"]
    if not isinstance(tasks_raw, list) or len(tasks_raw) != EXPECTED_TASKS:
        raise ContractError(f"task_map.tasks must contain exactly {EXPECTED_TASKS} tasks")

    tasks: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    paths: set[str] = set()
    for index, raw in enumerate(tasks_raw):
        task = _expect_object(raw, f"task_map.tasks[{index}]")
        _expect_exact_keys(
            task,
            (
                "task_suite",
                "task_id",
                "task_name",
                "bddl_path",
                "init_state_path",
                "trial_count",
            ),
            f"task_map.tasks[{index}]",
        )
        suite = task["task_suite"]
        if suite not in SUITES:
            raise ContractError(f"task_map.tasks[{index}].task_suite is invalid")
        task_id = _expect_int(
            task["task_id"],
            f"task_map.tasks[{index}].task_id",
            minimum=0,
            maximum=TASKS_PER_SUITE - 1,
        )
        identity = (suite, task_id)
        if identity in identities:
            raise ContractError(f"duplicate task identity: {identity}")
        identities.add(identity)
        task_name = _expect_nonempty_string(
            task["task_name"], f"task_map.tasks[{index}].task_name"
        )
        bddl_path = _validate_relative_posix_path(
            task["bddl_path"], f"task_map.tasks[{index}].bddl_path"
        )
        init_path = _validate_relative_posix_path(
            task["init_state_path"], f"task_map.tasks[{index}].init_state_path"
        )
        expected_bddl = f"bddl_files/{suite}/{task_name}.bddl"
        expected_init = f"init_files/{suite}/{task_name}.pruned_init"
        if bddl_path != expected_bddl or init_path != expected_init:
            raise ContractError(
                f"task {suite}/{task_id} paths do not pair with task_name; "
                f"expected {expected_bddl!r} and {expected_init!r}"
            )
        if bddl_path in paths or init_path in paths or bddl_path == init_path:
            raise ContractError(f"duplicate task input path at {suite}/{task_id}")
        paths.update((bddl_path, init_path))
        if task["trial_count"] != TRIALS_PER_TASK:
            raise ContractError(
                f"task_map.tasks[{index}].trial_count must be {TRIALS_PER_TASK}"
            )
        tasks.append(
            {
                "task_suite": suite,
                "task_id": task_id,
                "task_name": task_name,
                "bddl_path": bddl_path,
                "init_state_path": init_path,
                "trial_count": TRIALS_PER_TASK,
            }
        )

    expected_identities = {
        (suite, task_id)
        for suite in SUITES
        for task_id in range(TASKS_PER_SUITE)
    }
    if identities != expected_identities:
        missing = sorted(expected_identities - identities)
        raise ContractError(f"task_map task coverage is incomplete: {missing[:5]}")
    tasks.sort(key=_task_sort_key)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mf_wam_g0_task_map",
        "tasks": tasks,
    }


def validate_task_map(task_map: Any) -> dict[str, Any]:
    """Return the canonical, exact 40-task LIBERO map or raise."""

    return _normalized_task_map(task_map)


def task_map_sha256(task_map: Any) -> str:
    return canonical_json_sha256(_normalized_task_map(task_map))


def _tree_sha256(files: Sequence[Mapping[str, Any]]) -> str:
    """Apply DATA_TREE_ALGORITHM to already validated file records."""

    digest = hashlib.sha256()
    ordered = sorted(files, key=lambda item: str(item["path"]).encode("utf-8"))
    for item in ordered:
        file_sha = _expect_sha256(item.get("sha256"), "tree file sha256")
        path = _validate_relative_posix_path(item.get("path"), "tree file path")
        digest.update(f"{file_sha}  {path}\n".encode("utf-8"))
    return digest.hexdigest()


def build_data_inventory(
    data_root: str | Path,
    task_map: Any,
    *,
    dataset_id: str,
    revision: str,
) -> dict[str, Any]:
    """Read and hash the exact 40 BDDL + 40 initial-state file allowlist."""

    normalized_map = _normalized_task_map(task_map)
    dataset = _expect_nonempty_string(dataset_id, "dataset_id")
    data_revision = _expect_nonempty_string(revision, "revision")
    files: list[dict[str, Any]] = []
    filesystem_identities: set[tuple[int, int]] = set()
    for task in normalized_map["tasks"]:
        for role, field in (("bddl", "bddl_path"), ("initial_states", "init_state_path")):
            receipt = _safe_hash_relative_file(data_root, task[field])
            filesystem_identity = receipt.pop("filesystem_identity")
            if filesystem_identity in filesystem_identities:
                raise ContractError(
                    f"duplicate filesystem object referenced by {task[field]}"
                )
            filesystem_identities.add(filesystem_identity)
            files.append(
                {
                    "path": task[field],
                    "role": role,
                    "task_suite": task["task_suite"],
                    "task_id": task["task_id"],
                    "sha256": receipt["sha256"],
                    "size_bytes": receipt["size_bytes"],
                }
            )
    files.sort(key=lambda item: item["path"].encode("utf-8"))
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mf_wam_g0_data_inventory",
        "algorithm": DATA_TREE_ALGORITHM,
        "dataset_id": dataset,
        "revision": data_revision,
        "task_count": EXPECTED_TASKS,
        "file_count": EXPECTED_DATA_FILES,
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "task_map_canonical_sha256": canonical_json_sha256(normalized_map),
        "tasks": normalized_map["tasks"],
        "files": files,
        "tree_sha256": _tree_sha256(files),
    }
    return validate_data_inventory(inventory, data_root=data_root)


def validate_data_inventory(
    inventory: Any, *, data_root: str | Path | None = None
) -> dict[str, Any]:
    """Validate an exact inventory, optionally re-reading every input file."""

    payload = _expect_object(inventory, "data_inventory")
    _expect_exact_keys(
        payload,
        (
            "schema_version",
            "kind",
            "algorithm",
            "dataset_id",
            "revision",
            "task_count",
            "file_count",
            "total_size_bytes",
            "task_map_canonical_sha256",
            "tasks",
            "files",
            "tree_sha256",
        ),
        "data_inventory",
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ContractError("data_inventory.schema_version is unsupported")
    if payload["kind"] != "mf_wam_g0_data_inventory":
        raise ContractError("data_inventory.kind is invalid")
    if payload["algorithm"] != DATA_TREE_ALGORITHM:
        raise ContractError("data_inventory.algorithm is invalid")
    dataset_id = _expect_nonempty_string(payload["dataset_id"], "data_inventory.dataset_id")
    revision = _expect_nonempty_string(payload["revision"], "data_inventory.revision")
    if payload["task_count"] != EXPECTED_TASKS:
        raise ContractError(f"data_inventory.task_count must be {EXPECTED_TASKS}")
    if payload["file_count"] != EXPECTED_DATA_FILES:
        raise ContractError(f"data_inventory.file_count must be {EXPECTED_DATA_FILES}")

    normalized_map = _normalized_task_map(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "mf_wam_g0_task_map",
            "tasks": payload["tasks"],
        }
    )
    observed_map_sha = _expect_sha256(
        payload["task_map_canonical_sha256"],
        "data_inventory.task_map_canonical_sha256",
    )
    expected_map_sha = canonical_json_sha256(normalized_map)
    if observed_map_sha != expected_map_sha:
        raise ContractError("data_inventory task-map digest mismatch")

    files_raw = payload["files"]
    if not isinstance(files_raw, list) or len(files_raw) != EXPECTED_DATA_FILES:
        raise ContractError(f"data_inventory.files must contain {EXPECTED_DATA_FILES} entries")
    expected_bindings: dict[str, tuple[str, str, int]] = {}
    for task in normalized_map["tasks"]:
        expected_bindings[task["bddl_path"]] = (
            "bddl",
            task["task_suite"],
            task["task_id"],
        )
        expected_bindings[task["init_state_path"]] = (
            "initial_states",
            task["task_suite"],
            task["task_id"],
        )

    files: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    filesystem_identities: set[tuple[int, int]] = set()
    for index, raw in enumerate(files_raw):
        item = _expect_object(raw, f"data_inventory.files[{index}]")
        _expect_exact_keys(
            item,
            ("path", "role", "task_suite", "task_id", "sha256", "size_bytes"),
            f"data_inventory.files[{index}]",
        )
        path = _validate_relative_posix_path(
            item["path"], f"data_inventory.files[{index}].path"
        )
        if path in seen_paths:
            raise ContractError(f"duplicate inventory path: {path}")
        seen_paths.add(path)
        if path not in expected_bindings:
            raise ContractError(f"inventory path is not in the exact task allowlist: {path}")
        expected_role, expected_suite, expected_task_id = expected_bindings[path]
        if (
            item["role"] != expected_role
            or item["task_suite"] != expected_suite
            or item["task_id"] != expected_task_id
        ):
            raise ContractError(f"inventory role/task binding mismatch: {path}")
        file_sha = _expect_sha256(
            item["sha256"], f"data_inventory.files[{index}].sha256"
        )
        size_bytes = _expect_int(
            item["size_bytes"],
            f"data_inventory.files[{index}].size_bytes",
            minimum=1,
        )
        if data_root is not None:
            actual = _safe_hash_relative_file(data_root, path)
            filesystem_identity = actual.pop("filesystem_identity")
            if filesystem_identity in filesystem_identities:
                raise ContractError(f"duplicate filesystem object in inventory: {path}")
            filesystem_identities.add(filesystem_identity)
            if actual["sha256"] != file_sha or actual["size_bytes"] != size_bytes:
                raise ContractError(f"data file does not match inventory: {path}")
        files.append(
            {
                "path": path,
                "role": expected_role,
                "task_suite": expected_suite,
                "task_id": expected_task_id,
                "sha256": file_sha,
                "size_bytes": size_bytes,
            }
        )

    if seen_paths != set(expected_bindings):
        missing = sorted(set(expected_bindings) - seen_paths)
        raise ContractError(f"inventory file coverage is incomplete: {missing[:5]}")
    files.sort(key=lambda item: item["path"].encode("utf-8"))
    total_size = sum(item["size_bytes"] for item in files)
    if payload["total_size_bytes"] != total_size:
        raise ContractError("data_inventory.total_size_bytes mismatch")
    observed_tree = _expect_sha256(payload["tree_sha256"], "data_inventory.tree_sha256")
    expected_tree = _tree_sha256(files)
    if observed_tree != expected_tree:
        raise ContractError("data_inventory.tree_sha256 mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mf_wam_g0_data_inventory",
        "algorithm": DATA_TREE_ALGORITHM,
        "dataset_id": dataset_id,
        "revision": revision,
        "task_count": EXPECTED_TASKS,
        "file_count": EXPECTED_DATA_FILES,
        "total_size_bytes": total_size,
        "task_map_canonical_sha256": expected_map_sha,
        "tasks": normalized_map["tasks"],
        "files": files,
        "tree_sha256": expected_tree,
    }


def _process_id(suite: str, task_id: int) -> str:
    return f"{suite}/task{task_id:02d}"


def _episode_id(suite: str, task_id: int, trial_idx: int) -> str:
    return f"{_process_id(suite, task_id)}/trial{trial_idx:03d}"


def build_seed_schedule(
    task_map: Any,
    *,
    seed: int,
    python_hash_seed: int,
) -> dict[str, Any]:
    """Build the actual FastWAM evaluation seed semantics.

    There is no invented per-episode ``task_seed``.  One fresh process is used
    per task, global and environment seeding occur once in that process, and
    the same policy seed is passed to every replan call while trials execute in
    the exact order 0..49.
    """

    normalized_map = _normalized_task_map(task_map)
    runtime_seed = _expect_int(seed, "seed", minimum=1, maximum=2**32 - 2)
    hash_seed = _expect_int(
        python_hash_seed, "python_hash_seed", minimum=0, maximum=2**32 - 1
    )
    processes: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for task in normalized_map["tasks"]:
        suite = task["task_suite"]
        task_id = task["task_id"]
        process_id = _process_id(suite, task_id)
        processes.append(
            {
                "process_id": process_id,
                "task_suite": suite,
                "task_id": task_id,
                "global_rank": 0,
                "global_seed": runtime_seed,
                "environment_seed": runtime_seed,
                "environment_seed_scope": "once-before-trial-0",
                "policy_seed": runtime_seed,
                "policy_seed_scope": "constant-each-replan-call",
                "python_hash_seed": hash_seed,
                "trial_order": list(range(TRIALS_PER_TASK)),
                "initial_state_index_rule": "trial_idx",
            }
        )
        for trial_idx in range(TRIALS_PER_TASK):
            episodes.append(
                {
                    "episode_id": _episode_id(suite, task_id, trial_idx),
                    "process_id": process_id,
                    "task_suite": suite,
                    "task_id": task_id,
                    "trial_idx": trial_idx,
                    "episode_ordinal": trial_idx,
                    "initial_state_index": trial_idx,
                }
            )
    schedule = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mf_wam_g0_task_process_seed_schedule",
        "semantics": SEED_SCHEDULE_SEMANTICS,
        "seed": runtime_seed,
        "python_hash_seed": hash_seed,
        "task_map_canonical_sha256": canonical_json_sha256(normalized_map),
        "task_process_count": EXPECTED_TASKS,
        "episode_count": EXPECTED_EPISODES,
        "task_processes": processes,
        "episodes": episodes,
    }
    return validate_seed_schedule(schedule, task_map=normalized_map)


def validate_seed_schedule(schedule: Any, *, task_map: Any) -> dict[str, Any]:
    payload = _expect_object(schedule, "seed_schedule")
    _expect_exact_keys(
        payload,
        (
            "schema_version",
            "kind",
            "semantics",
            "seed",
            "python_hash_seed",
            "task_map_canonical_sha256",
            "task_process_count",
            "episode_count",
            "task_processes",
            "episodes",
        ),
        "seed_schedule",
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ContractError("seed_schedule.schema_version is unsupported")
    if payload["kind"] != "mf_wam_g0_task_process_seed_schedule":
        raise ContractError("seed_schedule.kind is invalid")
    if payload["semantics"] != SEED_SCHEDULE_SEMANTICS:
        raise ContractError("seed_schedule.semantics is invalid")
    seed = _expect_int(payload["seed"], "seed_schedule.seed", minimum=1, maximum=2**32 - 2)
    python_hash_seed = _expect_int(
        payload["python_hash_seed"],
        "seed_schedule.python_hash_seed",
        minimum=0,
        maximum=2**32 - 1,
    )
    if payload["task_process_count"] != EXPECTED_TASKS:
        raise ContractError(f"seed_schedule.task_process_count must be {EXPECTED_TASKS}")
    if payload["episode_count"] != EXPECTED_EPISODES:
        raise ContractError(f"seed_schedule.episode_count must be {EXPECTED_EPISODES}")
    normalized_map = _normalized_task_map(task_map)
    expected_map_sha = canonical_json_sha256(normalized_map)
    observed_map_sha = _expect_sha256(
        payload["task_map_canonical_sha256"],
        "seed_schedule.task_map_canonical_sha256",
    )
    if observed_map_sha != expected_map_sha:
        raise ContractError("seed_schedule task-map digest mismatch")

    processes_raw = payload["task_processes"]
    episodes_raw = payload["episodes"]
    if not isinstance(processes_raw, list) or len(processes_raw) != EXPECTED_TASKS:
        raise ContractError(f"seed_schedule.task_processes must contain {EXPECTED_TASKS} entries")
    if not isinstance(episodes_raw, list) or len(episodes_raw) != EXPECTED_EPISODES:
        raise ContractError(f"seed_schedule.episodes must contain {EXPECTED_EPISODES} entries")

    expected_task_identities = {
        (task["task_suite"], task["task_id"]) for task in normalized_map["tasks"]
    }
    seen_processes: set[tuple[str, int]] = set()
    processes: list[dict[str, Any]] = []
    for index, raw in enumerate(processes_raw):
        process = _expect_object(raw, f"seed_schedule.task_processes[{index}]")
        _expect_exact_keys(
            process,
            (
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
            ),
            f"seed_schedule.task_processes[{index}]",
        )
        suite = process["task_suite"]
        task_id = process["task_id"]
        if suite not in SUITES or type(task_id) is not int:
            raise ContractError(
                f"invalid task process identity fields at index {index}"
            )
        identity = (suite, task_id)
        if identity not in expected_task_identities or identity in seen_processes:
            raise ContractError(f"invalid or duplicate task process identity: {identity}")
        seen_processes.add(identity)
        expected_process_id = _process_id(suite, task_id)
        if process["process_id"] != expected_process_id or not _PROCESS_ID_RE.fullmatch(
            str(process["process_id"])
        ):
            raise ContractError(f"invalid process_id for {identity}")
        exact_values = {
            "global_rank": 0,
            "global_seed": seed,
            "environment_seed": seed,
            "environment_seed_scope": "once-before-trial-0",
            "policy_seed": seed,
            "policy_seed_scope": "constant-each-replan-call",
            "python_hash_seed": python_hash_seed,
            "trial_order": list(range(TRIALS_PER_TASK)),
            "initial_state_index_rule": "trial_idx",
        }
        for field, expected in exact_values.items():
            if process[field] != expected:
                raise ContractError(f"seed process {expected_process_id} has wrong {field}")
        processes.append(dict(process))
    if seen_processes != expected_task_identities:
        raise ContractError("seed_schedule task-process coverage is incomplete")
    processes.sort(key=lambda item: (SUITES.index(item["task_suite"]), item["task_id"]))

    seen_episodes: set[tuple[str, int, int]] = set()
    episodes: list[dict[str, Any]] = []
    for index, raw in enumerate(episodes_raw):
        episode = _expect_object(raw, f"seed_schedule.episodes[{index}]")
        _expect_exact_keys(
            episode,
            (
                "episode_id",
                "process_id",
                "task_suite",
                "task_id",
                "trial_idx",
                "episode_ordinal",
                "initial_state_index",
            ),
            f"seed_schedule.episodes[{index}]",
        )
        suite = episode["task_suite"]
        task_id = episode["task_id"]
        trial_idx = episode["trial_idx"]
        if suite not in SUITES or type(task_id) is not int:
            raise ContractError(f"invalid episode task identity fields at index {index}")
        identity = (suite, task_id, trial_idx)
        if (
            (suite, task_id) not in expected_task_identities
            or type(trial_idx) is not int
            or not 0 <= trial_idx < TRIALS_PER_TASK
            or identity in seen_episodes
        ):
            raise ContractError(f"invalid or duplicate episode identity: {identity}")
        seen_episodes.add(identity)
        expected_values = {
            "episode_id": _episode_id(suite, task_id, trial_idx),
            "process_id": _process_id(suite, task_id),
            "episode_ordinal": trial_idx,
            "initial_state_index": trial_idx,
        }
        if not _EPISODE_ID_RE.fullmatch(str(episode["episode_id"])):
            raise ContractError(f"invalid episode_id at {identity}")
        for field, expected in expected_values.items():
            if episode[field] != expected:
                raise ContractError(f"seed episode {identity} has wrong {field}")
        episodes.append(dict(episode))
    expected_episodes = {
        (suite, task_id, trial_idx)
        for suite, task_id in expected_task_identities
        for trial_idx in range(TRIALS_PER_TASK)
    }
    if seen_episodes != expected_episodes:
        raise ContractError("seed_schedule episode coverage is incomplete")
    episodes.sort(
        key=lambda item: (
            SUITES.index(item["task_suite"]),
            item["task_id"],
            item["trial_idx"],
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mf_wam_g0_task_process_seed_schedule",
        "semantics": SEED_SCHEDULE_SEMANTICS,
        "seed": seed,
        "python_hash_seed": python_hash_seed,
        "task_map_canonical_sha256": expected_map_sha,
        "task_process_count": EXPECTED_TASKS,
        "episode_count": EXPECTED_EPISODES,
        "task_processes": processes,
        "episodes": episodes,
    }


def _validate_source_identity(value: Any, location: str) -> dict[str, Any]:
    source = _expect_object(value, location)
    _expect_exact_keys(source, ("commit", "clean"), location)
    commit = _expect_commit(source["commit"], f"{location}.commit")
    if source["clean"] is not True:
        raise ContractError(f"{location}.clean must be true")
    return {"commit": commit, "clean": True}


def _validate_image(value: Any, location: str) -> dict[str, str]:
    image = _expect_object(value, location)
    _expect_exact_keys(image, ("uri", "digest"), location)
    uri = _expect_nonempty_string(image["uri"], f"{location}.uri")
    digest = image["digest"]
    if not isinstance(digest, str) or not _IMAGE_DIGEST_RE.fullmatch(digest):
        raise ContractError(f"{location}.digest must be sha256:<64 lowercase hex>")
    if digest == f"sha256:{'0' * 64}":
        raise ContractError(f"{location}.digest must not be an unknown placeholder")
    if not uri.endswith(f"@{digest}"):
        raise ContractError(f"{location}.uri must end with @{digest}")
    return {"uri": uri, "digest": digest}


def _validate_file_artifact(value: Any, location: str) -> dict[str, Any]:
    artifact = _expect_object(value, location)
    _expect_exact_keys(artifact, ("sha256", "size_bytes"), location)
    return {
        "sha256": _expect_sha256(artifact["sha256"], f"{location}.sha256"),
        "size_bytes": _expect_int(
            artifact["size_bytes"], f"{location}.size_bytes", minimum=1
        ),
    }


def _validate_model_cache_inventory(
    value: Any,
    location: str,
    *,
    model_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    inventory = _expect_object(value, location)
    _expect_exact_keys(
        inventory,
        ("algorithm", "file_count", "files", "canonical_sha256"),
        location,
    )
    if inventory["algorithm"] != MODEL_CACHE_INVENTORY_ALGORITHM:
        raise ContractError(f"{location}.algorithm is invalid")
    if inventory["file_count"] != len(_MODEL_CACHE_ROLES):
        raise ContractError(
            f"{location}.file_count must be {len(_MODEL_CACHE_ROLES)}"
        )
    files_raw = inventory["files"]
    if not isinstance(files_raw, list) or len(files_raw) != len(_MODEL_CACHE_ROLES):
        raise ContractError(
            f"{location}.files must contain exactly {len(_MODEL_CACHE_ROLES)} entries"
        )
    files: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    filesystem_identities: set[tuple[int, int]] = set()
    for index, raw in enumerate(files_raw):
        item = _expect_object(raw, f"{location}.files[{index}]")
        _expect_exact_keys(
            item,
            ("role", "path", "sha256", "size_bytes"),
            f"{location}.files[{index}]",
        )
        role = item["role"]
        if (
            not isinstance(role, str)
            or role not in _MODEL_CACHE_ROLES
            or role in seen_roles
        ):
            raise ContractError(f"invalid or duplicate model-cache role: {role}")
        seen_roles.add(role)
        path = _validate_relative_posix_path(
            item["path"], f"{location}.files[{index}].path"
        )
        if path in seen_paths:
            raise ContractError(f"duplicate model-cache path: {path}")
        seen_paths.add(path)
        file_sha = _expect_sha256(
            item["sha256"], f"{location}.files[{index}].sha256"
        )
        size_bytes = _expect_int(
            item["size_bytes"],
            f"{location}.files[{index}].size_bytes",
            minimum=1,
        )
        if model_cache_root is not None:
            actual = _safe_hash_relative_file(model_cache_root, path)
            filesystem_identity = actual.pop("filesystem_identity")
            if filesystem_identity in filesystem_identities:
                raise ContractError(f"duplicate model-cache filesystem object: {path}")
            filesystem_identities.add(filesystem_identity)
            if actual["sha256"] != file_sha or actual["size_bytes"] != size_bytes:
                raise ContractError(f"model-cache file does not match inventory: {path}")
        files.append(
            {
                "role": role,
                "path": path,
                "sha256": file_sha,
                "size_bytes": size_bytes,
            }
        )
    if seen_roles != _MODEL_CACHE_ROLES:
        missing = sorted(_MODEL_CACHE_ROLES - seen_roles)
        raise ContractError(f"model-cache role coverage is incomplete: {missing}")
    files.sort(key=lambda item: item["role"])
    core = {
        "algorithm": MODEL_CACHE_INVENTORY_ALGORITHM,
        "file_count": len(_MODEL_CACHE_ROLES),
        "files": files,
    }
    observed_sha = _expect_sha256(
        inventory["canonical_sha256"], f"{location}.canonical_sha256"
    )
    expected_sha = canonical_json_sha256(core)
    if observed_sha != expected_sha:
        raise ContractError(f"{location}.canonical_sha256 mismatch")
    return {**core, "canonical_sha256": expected_sha}


def _expected_evaluation_contract() -> dict[str, Any]:
    return {
        "suites": list(SUITES),
        "tasks_per_suite": TASKS_PER_SUITE,
        "trials_per_task": TRIALS_PER_TASK,
        "task_process_count": EXPECTED_TASKS,
        "episode_count": EXPECTED_EPISODES,
        "task_order_index": 0,
        "num_steps_wait": 30,
        "first_replan_env_step": 30,
        "replan_steps": 10,
        "action_horizon": 32,
        "action_dimension": 7,
        "state_dimension": 8,
        "strict_success_predicate": "libero_task_success",
        "confidence_level": 0.95,
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 42,
        "overall_equivalence_margin": 0.02,
        "suite_drop_margin": 0.03,
    }


def build_preregistration(
    spec: Any,
    *,
    data_inventory: Any,
    seed_schedule: Any,
) -> dict[str, Any]:
    """Build a preregistration from an explicit spec and locked inputs.

    The spec must already contain a real immutable image URI and digest.  This
    function never invents or fills unknown runtime identities.
    """

    inventory = validate_data_inventory(data_inventory)
    task_map = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mf_wam_g0_task_map",
        "tasks": inventory["tasks"],
    }
    schedule = validate_seed_schedule(seed_schedule, task_map=task_map)
    if schedule["task_map_canonical_sha256"] != inventory["task_map_canonical_sha256"]:
        raise ContractError("data inventory and seed schedule use different task maps")
    source = _expect_object(spec, "preregistration_spec")
    _expect_exact_keys(
        source,
        (
            "run_id",
            "created_at",
            "iteration_id",
            "project_page_id",
            "source",
            "image",
            "artifacts",
            "runtime_lock",
            "runtime_environment",
            "launch",
            "output",
        ),
        "preregistration_spec",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mf_wam_g0_preregistration",
        "phase": "PREREGISTERED",
        **dict(source),
        "data": {
            "inventory_canonical_sha256": canonical_json_sha256(inventory),
            "tree_sha256": inventory["tree_sha256"],
            "tree_algorithm": DATA_TREE_ALGORITHM,
            "dataset_id": inventory["dataset_id"],
            "revision": inventory["revision"],
            "task_map_canonical_sha256": inventory["task_map_canonical_sha256"],
            "task_count": EXPECTED_TASKS,
            "file_count": EXPECTED_DATA_FILES,
            "total_size_bytes": inventory["total_size_bytes"],
        },
        "seeds": {
            "schedule_canonical_sha256": canonical_json_sha256(schedule),
            "semantics": SEED_SCHEDULE_SEMANTICS,
            "seed": schedule["seed"],
            "python_hash_seed": schedule["python_hash_seed"],
            "task_process_count": EXPECTED_TASKS,
            "episode_count": EXPECTED_EPISODES,
        },
        "evaluation": _expected_evaluation_contract(),
    }
    return validate_preregistration(payload)


def validate_preregistration(preregistration: Any) -> dict[str, Any]:
    payload = _expect_object(preregistration, "preregistration")
    _expect_exact_keys(
        payload,
        (
            "schema_version",
            "kind",
            "phase",
            "run_id",
            "created_at",
            "iteration_id",
            "project_page_id",
            "source",
            "image",
            "artifacts",
            "data",
            "seeds",
            "runtime_lock",
            "runtime_environment",
            "launch",
            "evaluation",
            "output",
        ),
        "preregistration",
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ContractError("preregistration.schema_version is unsupported")
    if payload["kind"] != "mf_wam_g0_preregistration" or payload["phase"] != "PREREGISTERED":
        raise ContractError("preregistration kind/phase is invalid")
    run_id = _expect_nonempty_string(payload["run_id"], "preregistration.run_id")
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ContractError("preregistration.run_id is invalid")
    _parse_timestamp(payload["created_at"], "preregistration.created_at")
    iteration_id = _expect_nonempty_string(
        payload["iteration_id"], "preregistration.iteration_id"
    )
    if not _ITERATION_ID_RE.fullmatch(iteration_id):
        raise ContractError("preregistration.iteration_id is invalid")
    project_page_id = _expect_nonempty_string(
        payload["project_page_id"], "preregistration.project_page_id"
    )
    if not _UUID_RE.fullmatch(project_page_id):
        raise ContractError("preregistration.project_page_id is invalid")

    sources = _expect_object(payload["source"], "preregistration.source")
    _expect_exact_keys(
        sources, ("fastwam", "instrumentation", "libero", "auditor"), "preregistration.source"
    )
    normalized_sources = {
        name: _validate_source_identity(value, f"preregistration.source.{name}")
        for name, value in sources.items()
    }
    if normalized_sources["fastwam"]["commit"] != OFFICIAL_FASTWAM_COMMIT:
        raise ContractError("preregistration FastWAM commit is not the canonical baseline")
    if normalized_sources["libero"]["commit"] != OFFICIAL_LIBERO_COMMIT:
        raise ContractError("preregistration LIBERO commit is not the canonical baseline")
    image = _validate_image(payload["image"], "preregistration.image")

    artifacts = _expect_object(payload["artifacts"], "preregistration.artifacts")
    _expect_exact_keys(
        artifacts,
        ("checkpoint", "dataset_stats", "resolved_config", "model_cache"),
        "preregistration.artifacts",
    )
    normalized_artifacts = {
        name: _validate_file_artifact(
            artifacts[name], f"preregistration.artifacts.{name}"
        )
        for name in ("checkpoint", "dataset_stats", "resolved_config")
    }
    normalized_artifacts["model_cache"] = _validate_model_cache_inventory(
        artifacts["model_cache"], "preregistration.artifacts.model_cache"
    )

    data = _expect_object(payload["data"], "preregistration.data")
    _expect_exact_keys(
        data,
        (
            "inventory_canonical_sha256",
            "tree_sha256",
            "tree_algorithm",
            "dataset_id",
            "revision",
            "task_map_canonical_sha256",
            "task_count",
            "file_count",
            "total_size_bytes",
        ),
        "preregistration.data",
    )
    normalized_data = {
        "inventory_canonical_sha256": _expect_sha256(
            data["inventory_canonical_sha256"],
            "preregistration.data.inventory_canonical_sha256",
        ),
        "tree_sha256": _expect_sha256(
            data["tree_sha256"], "preregistration.data.tree_sha256"
        ),
        "tree_algorithm": data["tree_algorithm"],
        "dataset_id": _expect_nonempty_string(
            data["dataset_id"], "preregistration.data.dataset_id"
        ),
        "revision": _expect_nonempty_string(
            data["revision"], "preregistration.data.revision"
        ),
        "task_map_canonical_sha256": _expect_sha256(
            data["task_map_canonical_sha256"],
            "preregistration.data.task_map_canonical_sha256",
        ),
        "task_count": data["task_count"],
        "file_count": data["file_count"],
        "total_size_bytes": _expect_int(
            data["total_size_bytes"],
            "preregistration.data.total_size_bytes",
            minimum=1,
        ),
    }
    if normalized_data["tree_algorithm"] != DATA_TREE_ALGORITHM:
        raise ContractError("preregistration.data.tree_algorithm is invalid")
    if normalized_data["task_count"] != EXPECTED_TASKS:
        raise ContractError(f"preregistration.data.task_count must be {EXPECTED_TASKS}")
    if normalized_data["file_count"] != EXPECTED_DATA_FILES:
        raise ContractError(f"preregistration.data.file_count must be {EXPECTED_DATA_FILES}")
    if normalized_data["revision"] != normalized_sources["libero"]["commit"]:
        raise ContractError(
            "preregistration.data.revision must equal the locked LIBERO source commit"
        )

    seeds = _expect_object(payload["seeds"], "preregistration.seeds")
    _expect_exact_keys(
        seeds,
        (
            "schedule_canonical_sha256",
            "semantics",
            "seed",
            "python_hash_seed",
            "task_process_count",
            "episode_count",
        ),
        "preregistration.seeds",
    )
    normalized_seeds = {
        "schedule_canonical_sha256": _expect_sha256(
            seeds["schedule_canonical_sha256"],
            "preregistration.seeds.schedule_canonical_sha256",
        ),
        "semantics": seeds["semantics"],
        "seed": _expect_int(
            seeds["seed"], "preregistration.seeds.seed", minimum=1, maximum=2**32 - 2
        ),
        "python_hash_seed": _expect_int(
            seeds["python_hash_seed"],
            "preregistration.seeds.python_hash_seed",
            minimum=0,
            maximum=2**32 - 1,
        ),
        "task_process_count": seeds["task_process_count"],
        "episode_count": seeds["episode_count"],
    }
    if normalized_seeds["semantics"] != SEED_SCHEDULE_SEMANTICS:
        raise ContractError("preregistration.seeds.semantics is invalid")
    if normalized_seeds["task_process_count"] != EXPECTED_TASKS:
        raise ContractError("preregistration.seeds.task_process_count is invalid")
    if normalized_seeds["episode_count"] != EXPECTED_EPISODES:
        raise ContractError("preregistration.seeds.episode_count is invalid")

    runtime_lock = _expect_object(payload["runtime_lock"], "preregistration.runtime_lock")
    _expect_exact_keys(runtime_lock, _RUNTIME_VERSION_FIELDS, "preregistration.runtime_lock")
    normalized_runtime_lock = {
        field: _expect_nonempty_string(
            runtime_lock[field], f"preregistration.runtime_lock.{field}"
        )
        for field in _RUNTIME_VERSION_FIELDS
    }
    normalized_runtime_environment = _validate_runtime_environment(
        payload["runtime_environment"], "preregistration.runtime_environment"
    )

    launch = _expect_object(payload["launch"], "preregistration.launch")
    _expect_exact_keys(
        launch,
        (
            "provider",
            "worker_count",
            "gpu_count",
            "gpu_model",
            "gpu_memory_mib",
            "driver_version",
            "job_spec_sha256",
            "command_sha256",
            "sanitized_environment_sha256",
            "working_directory",
        ),
        "preregistration.launch",
    )
    if launch["provider"] != "alibaba-pai-dlc":
        raise ContractError("preregistration.launch.provider is invalid")
    if launch["worker_count"] != 1:
        raise ContractError("preregistration.launch.worker_count must be 1")
    gpu_count = _expect_int(
        launch["gpu_count"], "preregistration.launch.gpu_count", minimum=1, maximum=8
    )
    normalized_launch = {
        "provider": "alibaba-pai-dlc",
        "worker_count": 1,
        "gpu_count": gpu_count,
        "gpu_model": _expect_nonempty_string(
            launch["gpu_model"], "preregistration.launch.gpu_model"
        ),
        "gpu_memory_mib": _expect_int(
            launch["gpu_memory_mib"],
            "preregistration.launch.gpu_memory_mib",
            minimum=1,
        ),
        "driver_version": _expect_nonempty_string(
            launch["driver_version"], "preregistration.launch.driver_version"
        ),
        "job_spec_sha256": _expect_sha256(
            launch["job_spec_sha256"], "preregistration.launch.job_spec_sha256"
        ),
        "command_sha256": _expect_sha256(
            launch["command_sha256"], "preregistration.launch.command_sha256"
        ),
        "sanitized_environment_sha256": _expect_sha256(
            launch["sanitized_environment_sha256"],
            "preregistration.launch.sanitized_environment_sha256",
        ),
        "working_directory": _validate_absolute_path(
            launch["working_directory"], "preregistration.launch.working_directory"
        ),
    }
    evaluation = _expect_object(payload["evaluation"], "preregistration.evaluation")
    expected_evaluation = _expected_evaluation_contract()
    if dict(evaluation) != expected_evaluation:
        raise ContractError("preregistration.evaluation does not match the fixed G0 contract")

    output = _expect_object(payload["output"], "preregistration.output")
    _expect_exact_keys(output, ("artifact_root", "overwrite"), "preregistration.output")
    artifact_root = _validate_absolute_path(
        output["artifact_root"], "preregistration.output.artifact_root"
    )
    if PurePosixPath(artifact_root).name != run_id:
        raise ContractError("preregistration.output.artifact_root must end with run_id")
    if output["overwrite"] is not False:
        raise ContractError("preregistration.output.overwrite must be false")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mf_wam_g0_preregistration",
        "phase": "PREREGISTERED",
        "run_id": run_id,
        "created_at": payload["created_at"],
        "iteration_id": iteration_id,
        "project_page_id": project_page_id,
        "source": normalized_sources,
        "image": image,
        "artifacts": normalized_artifacts,
        "data": normalized_data,
        "seeds": normalized_seeds,
        "runtime_lock": normalized_runtime_lock,
        "runtime_environment": normalized_runtime_environment,
        "launch": normalized_launch,
        "evaluation": expected_evaluation,
        "output": {"artifact_root": artifact_root, "overwrite": False},
    }


def _validate_runtime_bindings(value: Any, location: str) -> dict[str, str]:
    bindings = _expect_object(value, location)
    _expect_exact_keys(
        bindings,
        (
            "checkpoint_sha256",
            "dataset_stats_sha256",
            "resolved_config_sha256",
            "data_inventory_canonical_sha256",
            "data_tree_sha256",
            "seed_schedule_canonical_sha256",
            "model_cache_inventory_canonical_sha256",
        ),
        location,
    )
    return {
        field: _expect_sha256(bindings[field], f"{location}.{field}")
        for field in bindings
    }


def validate_runtime_start(
    runtime_start: Any,
    *,
    preregistration: Any,
    model_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    prereg = validate_preregistration(preregistration)
    payload = _expect_object(runtime_start, "runtime_start")
    _expect_exact_keys(
        payload,
        (
            "schema_version",
            "kind",
            "phase",
            "receipt_scope",
            "run_id",
            "observed_at",
            "preregistration_canonical_sha256",
            "job",
            "source",
            "image",
            "bindings",
            "runtime",
            "runtime_environment",
            "gpu",
            "control_process",
            "imports",
            "model_cache_inventory",
        ),
        "runtime_start",
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ContractError("runtime_start.schema_version is unsupported")
    if payload["kind"] != "mf_wam_g0_runtime_start" or payload["phase"] != "STARTED":
        raise ContractError("runtime_start kind/phase is invalid")
    if payload["receipt_scope"] != "job-control-plane":
        raise ContractError("runtime_start.receipt_scope must be job-control-plane")
    if payload["run_id"] != prereg["run_id"]:
        raise ContractError("runtime_start.run_id does not match preregistration")
    observed_at = _parse_timestamp(payload["observed_at"], "runtime_start.observed_at")
    if observed_at < _parse_timestamp(prereg["created_at"], "preregistration.created_at"):
        raise ContractError("runtime_start predates preregistration")
    prereg_sha = _expect_sha256(
        payload["preregistration_canonical_sha256"],
        "runtime_start.preregistration_canonical_sha256",
    )
    if prereg_sha != canonical_json_sha256(prereg):
        raise ContractError("runtime_start preregistration digest mismatch")

    job = _expect_object(payload["job"], "runtime_start.job")
    _expect_exact_keys(
        job,
        ("provider", "job_id", "job_spec_sha256", "pod_uid", "hostname"),
        "runtime_start.job",
    )
    normalized_job = {
        "provider": job["provider"],
        "job_id": _expect_nonempty_string(job["job_id"], "runtime_start.job.job_id"),
        "job_spec_sha256": _expect_sha256(
            job["job_spec_sha256"], "runtime_start.job.job_spec_sha256"
        ),
        "pod_uid": _expect_nonempty_string(job["pod_uid"], "runtime_start.job.pod_uid"),
        "hostname": _expect_nonempty_string(job["hostname"], "runtime_start.job.hostname"),
    }
    if normalized_job["provider"] != prereg["launch"]["provider"]:
        raise ContractError("runtime_start provider does not match preregistration")
    if normalized_job["job_spec_sha256"] != prereg["launch"]["job_spec_sha256"]:
        raise ContractError("runtime_start job spec does not match preregistration")

    sources = _expect_object(payload["source"], "runtime_start.source")
    _expect_exact_keys(
        sources, ("fastwam", "instrumentation", "libero", "auditor"), "runtime_start.source"
    )
    normalized_sources = {
        name: _validate_source_identity(value, f"runtime_start.source.{name}")
        for name, value in sources.items()
    }
    if normalized_sources != prereg["source"]:
        raise ContractError("runtime_start source identities do not match preregistration")
    image = _validate_image(payload["image"], "runtime_start.image")
    if image != prereg["image"]:
        raise ContractError("runtime_start image does not match preregistration")

    bindings = _validate_runtime_bindings(payload["bindings"], "runtime_start.bindings")
    expected_bindings = {
        "checkpoint_sha256": prereg["artifacts"]["checkpoint"]["sha256"],
        "dataset_stats_sha256": prereg["artifacts"]["dataset_stats"]["sha256"],
        "resolved_config_sha256": prereg["artifacts"]["resolved_config"]["sha256"],
        "data_inventory_canonical_sha256": prereg["data"]["inventory_canonical_sha256"],
        "data_tree_sha256": prereg["data"]["tree_sha256"],
        "seed_schedule_canonical_sha256": prereg["seeds"]["schedule_canonical_sha256"],
        "model_cache_inventory_canonical_sha256": prereg["artifacts"]["model_cache"][
            "canonical_sha256"
        ],
    }
    if bindings != expected_bindings:
        raise ContractError("runtime_start bindings do not match preregistration")

    runtime = _expect_object(payload["runtime"], "runtime_start.runtime")
    _expect_exact_keys(runtime, _RUNTIME_VERSION_FIELDS, "runtime_start.runtime")
    normalized_runtime = {
        field: _expect_nonempty_string(runtime[field], f"runtime_start.runtime.{field}")
        for field in _RUNTIME_VERSION_FIELDS
    }
    if normalized_runtime != prereg["runtime_lock"]:
        raise ContractError("runtime_start versions do not match preregistration")

    normalized_runtime_environment = _validate_runtime_environment(
        payload["runtime_environment"], "runtime_start.runtime_environment"
    )
    if normalized_runtime_environment != prereg["runtime_environment"]:
        raise ContractError("runtime_start environment does not match preregistration")
    normalized_model_cache_root: str | None = None
    if model_cache_root is not None:
        normalized_model_cache_root = _validate_absolute_path(
            os.fspath(model_cache_root), "model_cache_root"
        )
        if (
            normalized_model_cache_root
            != normalized_runtime_environment["DIFFSYNTH_MODEL_BASE_PATH"]
        ):
            raise ContractError(
                "live model_cache_root does not match runtime "
                "DIFFSYNTH_MODEL_BASE_PATH"
            )

    gpu = _expect_object(payload["gpu"], "runtime_start.gpu")
    _expect_exact_keys(
        gpu,
        ("count", "model", "memory_mib", "driver_version", "uuids"),
        "runtime_start.gpu",
    )
    gpu_count = _expect_int(gpu["count"], "runtime_start.gpu.count", minimum=1, maximum=8)
    gpu_model = _expect_nonempty_string(gpu["model"], "runtime_start.gpu.model")
    gpu_memory_mib = _expect_int(
        gpu["memory_mib"], "runtime_start.gpu.memory_mib", minimum=1
    )
    gpu_driver_version = _expect_nonempty_string(
        gpu["driver_version"], "runtime_start.gpu.driver_version"
    )
    uuids = gpu["uuids"]
    if (
        not isinstance(uuids, list)
        or len(uuids) != gpu_count
        or any(not isinstance(item, str) or not item.strip() for item in uuids)
        or len(set(uuids)) != len(uuids)
    ):
        raise ContractError("runtime_start.gpu.uuids must exactly cover unique GPUs")
    if (
        gpu_count != prereg["launch"]["gpu_count"]
        or gpu_model != prereg["launch"]["gpu_model"]
        or gpu_memory_mib != prereg["launch"]["gpu_memory_mib"]
        or gpu_driver_version != prereg["launch"]["driver_version"]
    ):
        raise ContractError("runtime_start GPU allocation does not match preregistration")

    process = _expect_object(payload["control_process"], "runtime_start.control_process")
    _expect_exact_keys(
        process,
        (
            "working_directory",
            "command_sha256",
            "sanitized_environment_sha256",
            "python_hash_seed",
        ),
        "runtime_start.control_process",
    )
    normalized_process = {
        "working_directory": _validate_absolute_path(
            process["working_directory"], "runtime_start.control_process.working_directory"
        ),
        "command_sha256": _expect_sha256(
            process["command_sha256"], "runtime_start.control_process.command_sha256"
        ),
        "sanitized_environment_sha256": _expect_sha256(
            process["sanitized_environment_sha256"],
            "runtime_start.control_process.sanitized_environment_sha256",
        ),
        "python_hash_seed": _expect_int(
            process["python_hash_seed"],
            "runtime_start.control_process.python_hash_seed",
            minimum=0,
            maximum=2**32 - 1,
        ),
    }
    expected_process = {
        "working_directory": prereg["launch"]["working_directory"],
        "command_sha256": prereg["launch"]["command_sha256"],
        "sanitized_environment_sha256": prereg["launch"]["sanitized_environment_sha256"],
        "python_hash_seed": prereg["seeds"]["python_hash_seed"],
    }
    for field, expected in expected_process.items():
        if normalized_process[field] != expected:
            raise ContractError(
                f"runtime_start.control_process.{field} does not match preregistration"
            )

    imports_raw = payload["imports"]
    if not isinstance(imports_raw, list) or len(imports_raw) != len(_REQUIRED_IMPORTS):
        raise ContractError("runtime_start.imports must exactly cover required modules")
    imports: list[dict[str, str]] = []
    seen_imports: set[str] = set()
    for index, raw in enumerate(imports_raw):
        item = _expect_object(raw, f"runtime_start.imports[{index}]")
        _expect_exact_keys(item, ("module", "path", "sha256"), f"runtime_start.imports[{index}]")
        module = _expect_nonempty_string(
            item["module"], f"runtime_start.imports[{index}].module"
        )
        if module not in _REQUIRED_IMPORTS or module in seen_imports:
            raise ContractError(f"invalid or duplicate runtime import: {module}")
        seen_imports.add(module)
        imports.append(
            {
                "module": module,
                "path": _validate_absolute_path(
                    item["path"], f"runtime_start.imports[{index}].path"
                ),
                "sha256": _expect_sha256(
                    item["sha256"], f"runtime_start.imports[{index}].sha256"
                ),
            }
        )
    if seen_imports != _REQUIRED_IMPORTS:
        raise ContractError("runtime_start import coverage is incomplete")
    imports.sort(key=lambda item: item["module"])
    model_cache_inventory = _validate_model_cache_inventory(
        payload["model_cache_inventory"],
        "runtime_start.model_cache_inventory",
        model_cache_root=normalized_model_cache_root,
    )
    if model_cache_inventory != prereg["artifacts"]["model_cache"]:
        raise ContractError("runtime_start model-cache inventory does not match preregistration")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mf_wam_g0_runtime_start",
        "phase": "STARTED",
        "receipt_scope": "job-control-plane",
        "run_id": prereg["run_id"],
        "observed_at": payload["observed_at"],
        "preregistration_canonical_sha256": canonical_json_sha256(prereg),
        "job": normalized_job,
        "source": normalized_sources,
        "image": image,
        "bindings": bindings,
        "runtime": normalized_runtime,
        "runtime_environment": normalized_runtime_environment,
        "gpu": {
            "count": gpu_count,
            "model": gpu_model,
            "memory_mib": gpu_memory_mib,
            "driver_version": gpu_driver_version,
            "uuids": list(uuids),
        },
        "control_process": normalized_process,
        "imports": imports,
        "model_cache_inventory": model_cache_inventory,
    }


def _expected_result_path(suite: str, task_id: int) -> str:
    return f"results/{suite}/task{task_id:02d}.json"


def _expected_trace_receipt_path(suite: str, task_id: int) -> str:
    return f"trace_receipts/{suite}/task{task_id:02d}.json"


def _expected_trace_path(suite: str, task_id: int, trial_idx: int) -> str:
    return f"traces/{suite}/task{task_id:02d}/trial{trial_idx:03d}.json"


def _validate_artifact_reference(
    value: Any, location: str, *, allow_null: bool
) -> dict[str, Any] | None:
    if value is None and allow_null:
        return None
    reference = _expect_object(value, location)
    _expect_exact_keys(reference, ("path", "sha256", "size_bytes"), location)
    return {
        "path": _validate_relative_posix_path(reference["path"], f"{location}.path"),
        "sha256": _expect_sha256(reference["sha256"], f"{location}.sha256"),
        "size_bytes": _expect_int(
            reference["size_bytes"], f"{location}.size_bytes", minimum=1
        ),
    }


def _validate_terminal_aggregates(
    value: Any, *, succeeded: bool
) -> dict[str, str | None]:
    aggregates = _expect_object(value, "terminal.aggregates")
    _expect_exact_keys(
        aggregates,
        ("task_result_tree_sha256", "trace_tree_sha256"),
        "terminal.aggregates",
    )
    return {
        field: _expect_sha256(
            aggregates[field], f"terminal.aggregates.{field}", allow_null=not succeeded
        )
        for field in ("task_result_tree_sha256", "trace_tree_sha256")
    }


def validate_terminal_receipt(
    terminal: Any,
    *,
    preregistration: Any,
    runtime_start: Any,
    seed_schedule: Any,
    task_map: Any,
) -> dict[str, Any]:
    prereg = validate_preregistration(preregistration)
    start = validate_runtime_start(runtime_start, preregistration=prereg)
    schedule = validate_seed_schedule(seed_schedule, task_map=task_map)
    if canonical_json_sha256(schedule) != prereg["seeds"]["schedule_canonical_sha256"]:
        raise ContractError("terminal seed schedule is not the preregistered schedule")
    payload = _expect_object(terminal, "terminal")
    _expect_exact_keys(
        payload,
        (
            "schema_version",
            "kind",
            "phase",
            "run_id",
            "completed_at",
            "preregistration_canonical_sha256",
            "runtime_start_canonical_sha256",
            "status",
            "failure_reason",
            "manager_exit_code",
            "scope",
            "task_processes",
            "artifact_inventory",
            "aggregates",
        ),
        "terminal",
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ContractError("terminal.schema_version is unsupported")
    if payload["kind"] != "mf_wam_g0_terminal" or payload["phase"] != "TERMINAL":
        raise ContractError("terminal kind/phase is invalid")
    if payload["run_id"] != prereg["run_id"]:
        raise ContractError("terminal.run_id does not match preregistration")
    completed_at = _parse_timestamp(payload["completed_at"], "terminal.completed_at")
    if completed_at < _parse_timestamp(start["observed_at"], "runtime_start.observed_at"):
        raise ContractError("terminal predates runtime start")
    if payload["preregistration_canonical_sha256"] != canonical_json_sha256(prereg):
        raise ContractError("terminal preregistration digest mismatch")
    if payload["runtime_start_canonical_sha256"] != canonical_json_sha256(start):
        raise ContractError("terminal runtime-start digest mismatch")
    status_value = payload["status"]
    if status_value not in ("SUCCEEDED", "FAILED", "CANCELLED"):
        raise ContractError("terminal.status is invalid")
    succeeded = status_value == "SUCCEEDED"
    failure_reason = payload["failure_reason"]
    if succeeded:
        if failure_reason is not None:
            raise ContractError("successful terminal receipt must have null failure_reason")
    else:
        _expect_nonempty_string(failure_reason, "terminal.failure_reason")
    manager_exit_code = _expect_int(payload["manager_exit_code"], "terminal.manager_exit_code")
    if succeeded and manager_exit_code != 0:
        raise ContractError("successful terminal receipt requires manager_exit_code=0")

    scope = _expect_object(payload["scope"], "terminal.scope")
    _expect_exact_keys(
        scope, ("task_process_count", "episode_count", "complete"), "terminal.scope"
    )
    task_process_count = _expect_int(
        scope["task_process_count"],
        "terminal.scope.task_process_count",
        minimum=0,
        maximum=EXPECTED_TASKS,
    )
    episode_count = _expect_int(
        scope["episode_count"],
        "terminal.scope.episode_count",
        minimum=0,
        maximum=EXPECTED_EPISODES,
    )
    if type(scope["complete"]) is not bool:
        raise ContractError("terminal.scope.complete must be boolean")
    if succeeded and (
        task_process_count != EXPECTED_TASKS
        or episode_count != EXPECTED_EPISODES
        or scope["complete"] is not True
    ):
        raise ContractError("successful terminal receipt has incomplete scope")
    if not succeeded and scope["complete"] is not False:
        raise ContractError("failed/cancelled terminal receipt must not claim complete scope")

    expected_process_ids = {item["process_id"] for item in schedule["task_processes"]}
    tasks_by_process = {
        item["process_id"]: (item["task_suite"], item["task_id"])
        for item in schedule["task_processes"]
    }
    processes_raw = payload["task_processes"]
    if not isinstance(processes_raw, list):
        raise ContractError("terminal.task_processes must be a list")
    if succeeded and len(processes_raw) != EXPECTED_TASKS:
        raise ContractError(f"successful terminal requires {EXPECTED_TASKS} task receipts")
    if not succeeded and len(processes_raw) > EXPECTED_TASKS:
        raise ContractError("terminal has too many task receipts")
    processes: list[dict[str, Any]] = []
    seen_process_ids: set[str] = set()
    counted_episodes = 0
    for index, raw in enumerate(processes_raw):
        item = _expect_object(raw, f"terminal.task_processes[{index}]")
        _expect_exact_keys(
            item,
            (
                "process_id",
                "task_suite",
                "task_id",
                "execution_scope",
                "world_size",
                "global_rank",
                "local_rank",
                "exit_code",
                "result_path",
                "result_sha256",
                "result_size_bytes",
                "trace_receipt_path",
                "trace_receipt_sha256",
                "trace_receipt_size_bytes",
                "trace_tree_sha256",
                "episode_count",
                "complete",
            ),
            f"terminal.task_processes[{index}]",
        )
        process_id = item["process_id"]
        if (
            not isinstance(process_id, str)
            or process_id not in expected_process_ids
            or process_id in seen_process_ids
        ):
            raise ContractError(f"invalid or duplicate terminal process_id: {process_id}")
        seen_process_ids.add(process_id)
        expected_suite, expected_task_id = tasks_by_process[process_id]
        if item["task_suite"] != expected_suite or item["task_id"] != expected_task_id:
            raise ContractError(f"terminal task identity mismatch for {process_id}")
        exact_process_semantics = {
            "execution_scope": "one-process-per-task",
            "world_size": 1,
            "global_rank": 0,
            "local_rank": 0,
        }
        for field, expected in exact_process_semantics.items():
            if item[field] != expected:
                raise ContractError(f"terminal task {process_id} has wrong {field}")
        exit_code = _expect_int(
            item["exit_code"], f"terminal.task_processes[{index}].exit_code"
        )
        process_episode_count = _expect_int(
            item["episode_count"],
            f"terminal.task_processes[{index}].episode_count",
            minimum=0,
            maximum=TRIALS_PER_TASK,
        )
        if type(item["complete"]) is not bool:
            raise ContractError(f"terminal task {process_id} complete must be boolean")
        result_path = _validate_relative_posix_path(
            item["result_path"], f"terminal.task_processes[{index}].result_path"
        )
        trace_receipt_path = _validate_relative_posix_path(
            item["trace_receipt_path"],
            f"terminal.task_processes[{index}].trace_receipt_path",
        )
        if result_path != _expected_result_path(expected_suite, expected_task_id):
            raise ContractError(f"terminal task {process_id} has noncanonical result_path")
        if trace_receipt_path != _expected_trace_receipt_path(
            expected_suite, expected_task_id
        ):
            raise ContractError(
                f"terminal task {process_id} has noncanonical trace_receipt_path"
            )
        result_sha = _expect_sha256(
            item["result_sha256"],
            f"terminal.task_processes[{index}].result_sha256",
            allow_null=not succeeded,
        )
        result_size = (
            None
            if item["result_size_bytes"] is None and not succeeded
            else _expect_int(
                item["result_size_bytes"],
                f"terminal.task_processes[{index}].result_size_bytes",
                minimum=1,
            )
        )
        trace_receipt_sha = _expect_sha256(
            item["trace_receipt_sha256"],
            f"terminal.task_processes[{index}].trace_receipt_sha256",
            allow_null=not succeeded,
        )
        trace_receipt_size = (
            None
            if item["trace_receipt_size_bytes"] is None and not succeeded
            else _expect_int(
                item["trace_receipt_size_bytes"],
                f"terminal.task_processes[{index}].trace_receipt_size_bytes",
                minimum=1,
            )
        )
        trace_tree_sha = _expect_sha256(
            item["trace_tree_sha256"],
            f"terminal.task_processes[{index}].trace_tree_sha256",
            allow_null=not succeeded,
        )
        if succeeded and (
            exit_code != 0
            or result_sha is None
            or result_size is None
            or trace_receipt_sha is None
            or trace_receipt_size is None
            or trace_tree_sha is None
            or process_episode_count != TRIALS_PER_TASK
            or item["complete"] is not True
        ):
            raise ContractError(f"successful terminal task is incomplete: {process_id}")
        counted_episodes += process_episode_count
        processes.append(
            {
                "process_id": process_id,
                "task_suite": expected_suite,
                "task_id": expected_task_id,
                **exact_process_semantics,
                "exit_code": exit_code,
                "result_path": result_path,
                "result_sha256": result_sha,
                "result_size_bytes": result_size,
                "trace_receipt_path": trace_receipt_path,
                "trace_receipt_sha256": trace_receipt_sha,
                "trace_receipt_size_bytes": trace_receipt_size,
                "trace_tree_sha256": trace_tree_sha,
                "episode_count": process_episode_count,
                "complete": item["complete"],
            }
        )
    if task_process_count != len(processes_raw) or episode_count != counted_episodes:
        raise ContractError("terminal scope counts disagree with task receipts")
    if succeeded and seen_process_ids != expected_process_ids:
        raise ContractError("successful terminal task-process coverage is incomplete")
    processes.sort(key=lambda item: (SUITES.index(item["task_suite"]), item["task_id"]))

    inventory_reference = _validate_artifact_reference(
        payload["artifact_inventory"],
        "terminal.artifact_inventory",
        allow_null=not succeeded,
    )
    if succeeded and inventory_reference is not None:
        if inventory_reference["path"] != "artifact_inventory.json":
            raise ContractError("terminal artifact inventory path is not canonical")
    aggregates = _validate_terminal_aggregates(payload["aggregates"], succeeded=succeeded)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mf_wam_g0_terminal",
        "phase": "TERMINAL",
        "run_id": prereg["run_id"],
        "completed_at": payload["completed_at"],
        "preregistration_canonical_sha256": canonical_json_sha256(prereg),
        "runtime_start_canonical_sha256": canonical_json_sha256(start),
        "status": status_value,
        "failure_reason": failure_reason,
        "manager_exit_code": manager_exit_code,
        "scope": {
            "task_process_count": task_process_count,
            "episode_count": episode_count,
            "complete": scope["complete"],
        },
        "task_processes": processes,
        "artifact_inventory": inventory_reference,
        "aggregates": aggregates,
    }


def _validate_task_trace_receipt(
    payload: Any,
    *,
    process: Mapping[str, Any],
    inventory_by_path: Mapping[str, Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    runtime_start: Mapping[str, Any],
    seed_schedule: Mapping[str, Any],
) -> dict[str, Any]:
    location = f"trace_receipt[{process['process_id']}]"
    receipt = _expect_object(payload, location)
    _expect_exact_keys(
        receipt,
        (
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
        ),
        location,
    )
    if receipt["schema_version"] != SCHEMA_VERSION or receipt["kind"] != "mf_wam_g0_task_trace_receipt":
        raise ContractError(f"{location} kind/schema is invalid")
    if receipt["run_id"] != preregistration["run_id"]:
        raise ContractError(f"{location}.run_id does not match preregistration")
    _expect_int(receipt["task_id"], f"{location}.task_id", minimum=0, maximum=9)
    _expect_int(receipt["world_size"], f"{location}.world_size", minimum=1)
    _expect_int(receipt["global_rank"], f"{location}.global_rank", minimum=0)
    _expect_int(receipt["local_rank"], f"{location}.local_rank", minimum=0)
    for field in (
        "process_id",
        "task_suite",
        "task_id",
        "execution_scope",
        "world_size",
        "global_rank",
        "local_rank",
    ):
        if receipt[field] != process[field]:
            raise ContractError(f"{location}.{field} does not match task process")

    bindings = _expect_object(receipt["bindings"], f"{location}.bindings")
    binding_fields = (
        "preregistration_canonical_sha256",
        "runtime_start_canonical_sha256",
        "seed_schedule_canonical_sha256",
        "resolved_config_sha256",
        "image_digest",
        "fastwam_commit",
        "instrumentation_commit",
    )
    _expect_exact_keys(bindings, binding_fields, f"{location}.bindings")
    normalized_bindings = {
        "preregistration_canonical_sha256": _expect_sha256(
            bindings["preregistration_canonical_sha256"],
            f"{location}.bindings.preregistration_canonical_sha256",
        ),
        "runtime_start_canonical_sha256": _expect_sha256(
            bindings["runtime_start_canonical_sha256"],
            f"{location}.bindings.runtime_start_canonical_sha256",
        ),
        "seed_schedule_canonical_sha256": _expect_sha256(
            bindings["seed_schedule_canonical_sha256"],
            f"{location}.bindings.seed_schedule_canonical_sha256",
        ),
        "resolved_config_sha256": _expect_sha256(
            bindings["resolved_config_sha256"],
            f"{location}.bindings.resolved_config_sha256",
        ),
        "image_digest": bindings["image_digest"],
        "fastwam_commit": _expect_commit(
            bindings["fastwam_commit"], f"{location}.bindings.fastwam_commit"
        ),
        "instrumentation_commit": _expect_commit(
            bindings["instrumentation_commit"],
            f"{location}.bindings.instrumentation_commit",
        ),
    }
    if (
        not isinstance(normalized_bindings["image_digest"], str)
        or not _IMAGE_DIGEST_RE.fullmatch(normalized_bindings["image_digest"])
        or normalized_bindings["image_digest"] == f"sha256:{'0' * 64}"
    ):
        raise ContractError(f"{location}.bindings.image_digest is invalid")
    expected_bindings = {
        "preregistration_canonical_sha256": canonical_json_sha256(preregistration),
        "runtime_start_canonical_sha256": canonical_json_sha256(runtime_start),
        "seed_schedule_canonical_sha256": canonical_json_sha256(seed_schedule),
        "resolved_config_sha256": preregistration["artifacts"]["resolved_config"][
            "sha256"
        ],
        "image_digest": preregistration["image"]["digest"],
        "fastwam_commit": preregistration["source"]["fastwam"]["commit"],
        "instrumentation_commit": preregistration["source"]["instrumentation"][
            "commit"
        ],
    }
    if normalized_bindings != expected_bindings:
        raise ContractError(f"{location}.bindings do not match upstream contracts")

    scheduled_processes = {
        item["process_id"]: item for item in seed_schedule["task_processes"]
    }
    scheduled_process = scheduled_processes.get(process["process_id"])
    if scheduled_process is None:
        raise ContractError(f"{location} lacks a matching scheduled process")
    seeds = _expect_object(receipt["seeds"], f"{location}.seeds")
    seed_fields = (
        "global_seed",
        "environment_seed",
        "environment_seed_scope",
        "policy_seed",
        "policy_seed_scope",
        "python_hash_seed",
        "trial_order",
        "initial_state_index_rule",
    )
    _expect_exact_keys(seeds, seed_fields, f"{location}.seeds")
    normalized_seeds = {
        field: _expect_int(
            seeds[field], f"{location}.seeds.{field}", minimum=0, maximum=2**32 - 1
        )
        for field in (
            "global_seed",
            "environment_seed",
            "policy_seed",
            "python_hash_seed",
        )
    }
    normalized_seeds.update(
        {
            "environment_seed_scope": _expect_nonempty_string(
                seeds["environment_seed_scope"],
                f"{location}.seeds.environment_seed_scope",
            ),
            "policy_seed_scope": _expect_nonempty_string(
                seeds["policy_seed_scope"], f"{location}.seeds.policy_seed_scope"
            ),
            "trial_order": seeds["trial_order"],
            "initial_state_index_rule": _expect_nonempty_string(
                seeds["initial_state_index_rule"],
                f"{location}.seeds.initial_state_index_rule",
            ),
        }
    )
    expected_seeds = {
        field: scheduled_process[field] for field in seed_fields
    }
    if normalized_seeds != expected_seeds:
        raise ContractError(f"{location}.seeds do not match the exact seed schedule")

    official_result = _expect_object(
        receipt["official_result"], f"{location}.official_result"
    )
    _expect_exact_keys(
        official_result,
        ("path", "sha256", "size_bytes"),
        f"{location}.official_result",
    )
    normalized_result = {
        "path": _validate_relative_posix_path(
            official_result["path"], f"{location}.official_result.path"
        ),
        "sha256": _expect_sha256(
            official_result["sha256"], f"{location}.official_result.sha256"
        ),
        "size_bytes": _expect_int(
            official_result["size_bytes"],
            f"{location}.official_result.size_bytes",
            minimum=1,
        ),
    }
    expected_result = {
        "path": process["result_path"],
        "sha256": process["result_sha256"],
        "size_bytes": process["result_size_bytes"],
    }
    inventory_result = inventory_by_path.get(normalized_result["path"])
    if (
        normalized_result != expected_result
        or inventory_result is None
        or inventory_result["role"] != "task_result"
        or {
            field: inventory_result[field]
            for field in ("path", "sha256", "size_bytes")
        }
        != normalized_result
    ):
        raise ContractError(f"{location}.official_result is not triply content-bound")
    if receipt["episode_count"] != TRIALS_PER_TASK:
        raise ContractError(f"{location}.episode_count must be {TRIALS_PER_TASK}")
    traces_raw = receipt["traces"]
    if not isinstance(traces_raw, list) or len(traces_raw) != TRIALS_PER_TASK:
        raise ContractError(f"{location}.traces must contain {TRIALS_PER_TASK} entries")
    traces: list[dict[str, Any]] = []
    seen_trials: set[int] = set()
    for index, raw in enumerate(traces_raw):
        trace = _expect_object(raw, f"{location}.traces[{index}]")
        _expect_exact_keys(
            trace,
            ("trial_idx", "path", "sha256", "size_bytes"),
            f"{location}.traces[{index}]",
        )
        trial_idx = _expect_int(
            trace["trial_idx"],
            f"{location}.traces[{index}].trial_idx",
            minimum=0,
            maximum=TRIALS_PER_TASK - 1,
        )
        if trial_idx in seen_trials:
            raise ContractError(f"{location} contains duplicate trial {trial_idx}")
        seen_trials.add(trial_idx)
        path = _validate_relative_posix_path(
            trace["path"], f"{location}.traces[{index}].path"
        )
        expected_path = _expected_trace_path(
            process["task_suite"], process["task_id"], trial_idx
        )
        if path != expected_path:
            raise ContractError(f"{location} trace path is not canonical for trial {trial_idx}")
        sha = _expect_sha256(trace["sha256"], f"{location}.traces[{index}].sha256")
        size = _expect_int(
            trace["size_bytes"],
            f"{location}.traces[{index}].size_bytes",
            minimum=1,
        )
        inventory_item = inventory_by_path.get(path)
        if (
            inventory_item is None
            or inventory_item["role"] != "episode_trace"
            or inventory_item["sha256"] != sha
            or inventory_item["size_bytes"] != size
        ):
            raise ContractError(f"{location} trace is not bound by artifact inventory: {path}")
        traces.append(
            {"trial_idx": trial_idx, "path": path, "sha256": sha, "size_bytes": size}
        )
    if seen_trials != set(range(TRIALS_PER_TASK)):
        raise ContractError(f"{location} trial coverage is incomplete")
    traces.sort(key=lambda item: item["trial_idx"])
    expected_tree = _tree_sha256(traces)
    observed_tree = _expect_sha256(receipt["tree_sha256"], f"{location}.tree_sha256")
    if observed_tree != expected_tree or observed_tree != process["trace_tree_sha256"]:
        raise ContractError(f"{location} trace tree aggregate mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mf_wam_g0_task_trace_receipt",
        "run_id": preregistration["run_id"],
        "process_id": process["process_id"],
        "task_suite": process["task_suite"],
        "task_id": process["task_id"],
        "execution_scope": process["execution_scope"],
        "world_size": process["world_size"],
        "global_rank": process["global_rank"],
        "local_rank": process["local_rank"],
        "bindings": normalized_bindings,
        "seeds": normalized_seeds,
        "official_result": normalized_result,
        "episode_count": TRIALS_PER_TASK,
        "traces": traces,
        "tree_sha256": expected_tree,
    }


def _terminal_core_canonical_sha256(terminal: Mapping[str, Any]) -> str:
    """Hash the normalized terminal fields that do not create an inventory cycle."""

    fields = (
        "schema_version",
        "kind",
        "phase",
        "run_id",
        "completed_at",
        "preregistration_canonical_sha256",
        "runtime_start_canonical_sha256",
        "status",
        "failure_reason",
        "manager_exit_code",
        "scope",
        "task_processes",
        "aggregates",
    )
    return canonical_json_sha256({field: terminal[field] for field in fields})


def _audit_terminal_artifacts(
    *,
    artifact_root: str | Path,
    terminal: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    runtime_start: Mapping[str, Any],
    seed_schedule: Mapping[str, Any],
) -> dict[str, Any]:
    reference = terminal["artifact_inventory"]
    if reference is None:
        raise ContractError("successful terminal receipt lacks artifact inventory")
    inventory_file = _safe_hash_relative_file(
        artifact_root,
        reference["path"],
        capture_bytes=True,
    )
    if inventory_file["link_count"] != 1:
        raise ContractError("terminal artifact inventory must not be hardlinked")
    if (
        inventory_file["sha256"] != reference["sha256"]
        or inventory_file["size_bytes"] != reference["size_bytes"]
    ):
        raise ContractError("live artifact inventory does not match terminal receipt")
    inventory_payload = _loads_json_strict_bytes(
        inventory_file["bytes"], reference["path"]
    )
    inventory = _expect_object(inventory_payload, "artifact_inventory")
    _expect_exact_keys(
        inventory,
        (
            "schema_version",
            "kind",
            "algorithm",
            "file_count",
            "total_size_bytes",
            "files",
            "tree_sha256",
        ),
        "artifact_inventory",
    )
    if inventory["schema_version"] != SCHEMA_VERSION:
        raise ContractError("artifact_inventory.schema_version is unsupported")
    if inventory["kind"] != "mf_wam_g0_terminal_artifact_inventory":
        raise ContractError("artifact_inventory.kind is invalid")
    if inventory["algorithm"] != DATA_TREE_ALGORITHM:
        raise ContractError("artifact_inventory.algorithm is invalid")
    if inventory["file_count"] != EXPECTED_TERMINAL_FILES:
        raise ContractError(
            f"artifact_inventory.file_count must be {EXPECTED_TERMINAL_FILES}"
        )
    files_raw = inventory["files"]
    if not isinstance(files_raw, list) or len(files_raw) != EXPECTED_TERMINAL_FILES:
        raise ContractError(
            f"artifact_inventory.files must contain {EXPECTED_TERMINAL_FILES} entries"
        )

    files: list[dict[str, Any]] = []
    by_path: dict[str, dict[str, Any]] = {}
    role_counts = {role: 0 for role in _TERMINAL_ROLE_COUNTS}
    filesystem_identities = {inventory_file["filesystem_identity"]}
    captured_receipts: dict[str, bytes] = {}
    captured_marker: bytes | None = None
    for index, raw in enumerate(files_raw):
        item = _expect_object(raw, f"artifact_inventory.files[{index}]")
        _expect_exact_keys(
            item,
            ("path", "role", "sha256", "size_bytes"),
            f"artifact_inventory.files[{index}]",
        )
        path = _validate_relative_posix_path(
            item["path"], f"artifact_inventory.files[{index}].path"
        )
        if path == reference["path"] or path in by_path:
            raise ContractError(f"duplicate/self-referential artifact inventory path: {path}")
        role = item["role"]
        if not isinstance(role, str) or role not in _TERMINAL_ROLE_COUNTS:
            raise ContractError(f"invalid terminal artifact role: {role}")
        sha = _expect_sha256(
            item["sha256"], f"artifact_inventory.files[{index}].sha256"
        )
        size = _expect_int(
            item["size_bytes"],
            f"artifact_inventory.files[{index}].size_bytes",
            minimum=1,
        )
        capture = role in ("task_trace_receipt", "completion_marker")
        actual = _safe_hash_relative_file(
            artifact_root,
            path,
            capture_bytes=capture,
        )
        filesystem_identity = actual["filesystem_identity"]
        if actual["link_count"] != 1:
            raise ContractError(f"hardlinked terminal artifact is forbidden: {path}")
        if filesystem_identity in filesystem_identities:
            raise ContractError(f"duplicate/hardlinked terminal artifact: {path}")
        filesystem_identities.add(filesystem_identity)
        if actual["sha256"] != sha or actual["size_bytes"] != size:
            raise ContractError(f"live terminal artifact does not match inventory: {path}")
        normalized = {"path": path, "role": role, "sha256": sha, "size_bytes": size}
        files.append(normalized)
        by_path[path] = normalized
        role_counts[role] += 1
        if role == "task_trace_receipt":
            captured_receipts[path] = actual["bytes"]
        elif role == "completion_marker":
            captured_marker = actual["bytes"]
    if role_counts != _TERMINAL_ROLE_COUNTS:
        raise ContractError(f"terminal artifact role counts mismatch: {role_counts}")
    files.sort(key=lambda item: item["path"].encode("utf-8"))
    total_size = sum(item["size_bytes"] for item in files)
    if inventory["total_size_bytes"] != total_size:
        raise ContractError("artifact_inventory.total_size_bytes mismatch")
    expected_inventory_tree = _tree_sha256(files)
    if inventory["tree_sha256"] != expected_inventory_tree:
        raise ContractError("artifact_inventory.tree_sha256 mismatch")

    fixed_paths = {
        "summary.csv": "summary_csv",
        "task_success_rates.csv": "task_success_rates_csv",
        "summary.json": "summary_json",
        "completion.json": "completion_marker",
    }
    for path, role in fixed_paths.items():
        if path not in by_path or by_path[path]["role"] != role:
            raise ContractError(f"terminal artifact inventory lacks canonical {path}")

    result_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for process in terminal["task_processes"]:
        result = by_path.get(process["result_path"])
        if (
            result is None
            or result["role"] != "task_result"
            or result["sha256"] != process["result_sha256"]
            or result["size_bytes"] != process["result_size_bytes"]
        ):
            raise ContractError(
                f"task result is not content-bound for {process['process_id']}"
            )
        result_rows.append(result)
        trace_receipt_item = by_path.get(process["trace_receipt_path"])
        if (
            trace_receipt_item is None
            or trace_receipt_item["role"] != "task_trace_receipt"
            or trace_receipt_item["sha256"] != process["trace_receipt_sha256"]
            or trace_receipt_item["size_bytes"] != process["trace_receipt_size_bytes"]
        ):
            raise ContractError(
                f"task trace receipt is not content-bound for {process['process_id']}"
            )
        receipt_payload = _loads_json_strict_bytes(
            captured_receipts[process["trace_receipt_path"]],
            process["trace_receipt_path"],
        )
        normalized_receipt = _validate_task_trace_receipt(
            receipt_payload,
            process=process,
            inventory_by_path=by_path,
            preregistration=preregistration,
            runtime_start=runtime_start,
            seed_schedule=seed_schedule,
        )
        trace_rows.extend(normalized_receipt["traces"])
    result_tree = _tree_sha256(result_rows)
    trace_tree = _tree_sha256(trace_rows)
    if terminal["aggregates"]["task_result_tree_sha256"] != result_tree:
        raise ContractError("terminal task-result aggregate mismatch")
    if terminal["aggregates"]["trace_tree_sha256"] != trace_tree:
        raise ContractError("terminal trace aggregate mismatch")

    if captured_marker is None:
        raise ContractError("terminal completion marker was not captured")
    marker = _loads_json_strict_bytes(captured_marker, "completion.json")
    expected_marker = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mf_wam_g0_completion_marker",
        "run_id": terminal["run_id"],
        "status": "SUCCEEDED",
        "task_process_count": EXPECTED_TASKS,
        "episode_count": EXPECTED_EPISODES,
        "terminal_core_canonical_sha256": _terminal_core_canonical_sha256(terminal),
    }
    if marker != expected_marker:
        raise ContractError("terminal completion marker content is invalid")
    return {
        "artifact_inventory_raw_sha256": inventory_file["sha256"],
        "artifact_inventory_tree_sha256": expected_inventory_tree,
        "artifact_count": EXPECTED_TERMINAL_FILES,
        "task_result_tree_sha256": result_tree,
        "trace_tree_sha256": trace_tree,
        "terminal_core_canonical_sha256": expected_marker[
            "terminal_core_canonical_sha256"
        ],
    }


def validate_contract_chain(
    *,
    preregistration: Any,
    runtime_start: Any,
    terminal: Any,
    data_inventory: Any,
    seed_schedule: Any,
    trusted_anchors: Any,
    data_root: str | Path | None = None,
    model_cache_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate all structural links against independently preserved anchors.

    The three trusted digests must come from outside the receipt chain.  This
    prevents an attacker from changing an upstream receipt and then merely
    resealing every downstream digest.  Even a fully successful result remains
    structural evidence only; specialized G0 outcome auditing is deliberately
    outside this function.
    """

    if data_root is None:
        raise ContractError("contract-chain validation requires a live data_root readback")
    if model_cache_root is None:
        raise ContractError(
            "contract-chain validation requires a live model_cache_root readback"
        )
    if artifact_root is None:
        raise ContractError(
            "contract-chain validation requires a live artifact_root readback"
        )

    anchors = _expect_object(trusted_anchors, "trusted_anchors")
    anchor_fields = (
        "preregistration_canonical_sha256",
        "runtime_start_canonical_sha256",
        "terminal_canonical_sha256",
    )
    _expect_exact_keys(anchors, anchor_fields, "trusted_anchors")
    normalized_anchors = {
        field: _expect_sha256(anchors[field], f"trusted_anchors.{field}")
        for field in anchor_fields
    }

    prereg = validate_preregistration(preregistration)
    prereg_sha = canonical_json_sha256(prereg)
    if prereg_sha != normalized_anchors["preregistration_canonical_sha256"]:
        raise ContractError("preregistration does not match the external trusted anchor")
    live_model_cache_root = _validate_absolute_path(
        os.fspath(model_cache_root), "model_cache_root"
    )
    if (
        live_model_cache_root
        != prereg["runtime_environment"]["DIFFSYNTH_MODEL_BASE_PATH"]
    ):
        raise ContractError(
            "live model_cache_root does not match preregistered "
            "DIFFSYNTH_MODEL_BASE_PATH"
        )

    inventory = validate_data_inventory(data_inventory, data_root=data_root)
    task_map = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mf_wam_g0_task_map",
        "tasks": inventory["tasks"],
    }
    schedule = validate_seed_schedule(seed_schedule, task_map=task_map)
    if canonical_json_sha256(inventory) != prereg["data"]["inventory_canonical_sha256"]:
        raise ContractError("contract chain uses a non-preregistered data inventory")
    if inventory["tree_sha256"] != prereg["data"]["tree_sha256"]:
        raise ContractError("contract chain data tree differs from preregistration")
    if canonical_json_sha256(schedule) != prereg["seeds"]["schedule_canonical_sha256"]:
        raise ContractError("contract chain uses a non-preregistered seed schedule")
    start = validate_runtime_start(
        runtime_start,
        preregistration=prereg,
        model_cache_root=live_model_cache_root,
    )
    start_sha = canonical_json_sha256(start)
    if start_sha != normalized_anchors["runtime_start_canonical_sha256"]:
        raise ContractError("runtime start does not match the external trusted anchor")

    terminal_receipt = validate_terminal_receipt(
        terminal,
        preregistration=prereg,
        runtime_start=start,
        seed_schedule=schedule,
        task_map=task_map,
    )
    terminal_sha = canonical_json_sha256(terminal_receipt)
    if terminal_sha != normalized_anchors["terminal_canonical_sha256"]:
        raise ContractError("terminal receipt does not match the external trusted anchor")

    artifact_root_text = _validate_absolute_path(
        os.fspath(artifact_root), "artifact_root"
    )
    if artifact_root_text != prereg["output"]["artifact_root"]:
        raise ContractError("live artifact_root does not match preregistration")
    artifact_root_path = Path(artifact_root_text)
    try:
        artifact_root_lstat = artifact_root_path.lstat()
    except OSError as exc:
        raise ContractError(
            f"cannot stat artifact root {artifact_root_path}: {exc}"
        ) from exc
    if stat.S_ISLNK(artifact_root_lstat.st_mode) or not stat.S_ISDIR(
        artifact_root_lstat.st_mode
    ):
        raise ContractError("artifact root must be a real directory, not a symlink")

    terminal_success = terminal_receipt["status"] == "SUCCEEDED"
    artifact_audit = (
        _audit_terminal_artifacts(
            artifact_root=artifact_root_path,
            terminal=terminal_receipt,
            preregistration=prereg,
            runtime_start=start,
            seed_schedule=schedule,
        )
        if terminal_success
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mf_wam_g0_structural_contract_audit",
        "status": "STRUCTURAL_PASS_ONLY" if terminal_success else "STRUCTURAL_FAIL",
        "specialized_g0_status": "UNCERTAIN",
        "run_id": prereg["run_id"],
        "canonical_json_algorithm": CANONICAL_JSON_ALGORITHM,
        "data_tree_algorithm": DATA_TREE_ALGORITHM,
        "digests": {
            "data_inventory_canonical_sha256": canonical_json_sha256(inventory),
            "seed_schedule_canonical_sha256": canonical_json_sha256(schedule),
            "preregistration_canonical_sha256": prereg_sha,
            "runtime_start_canonical_sha256": start_sha,
            "terminal_canonical_sha256": terminal_sha,
        },
        "trusted_anchors": normalized_anchors,
        "artifact_audit": artifact_audit,
        "terminal_success": terminal_success,
        "formal_training_allowed": False,
        "authorization_reason": (
            "structural contract verification is not specialized G0 outcome/trace/CI "
            "recomputation"
        ),
    }


__all__ = [
    "CANONICAL_JSON_ALGORITHM",
    "ContractError",
    "DATA_TREE_ALGORITHM",
    "EXPECTED_DATA_FILES",
    "EXPECTED_EPISODES",
    "EXPECTED_TERMINAL_FILES",
    "EXPECTED_TASKS",
    "MODEL_CACHE_INVENTORY_ALGORITHM",
    "OFFICIAL_FASTWAM_COMMIT",
    "OFFICIAL_LIBERO_COMMIT",
    "SCHEMA_VERSION",
    "SEED_SCHEDULE_SEMANTICS",
    "SUITES",
    "TASKS_PER_SUITE",
    "TRIALS_PER_TASK",
    "build_data_inventory",
    "build_preregistration",
    "build_seed_schedule",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "load_json_strict",
    "task_map_sha256",
    "validate_contract_chain",
    "validate_data_inventory",
    "validate_preregistration",
    "validate_runtime_start",
    "validate_seed_schedule",
    "validate_task_map",
    "validate_terminal_receipt",
    "write_canonical_json",
]
