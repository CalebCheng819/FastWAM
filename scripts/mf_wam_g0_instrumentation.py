"""Transparent, fail-closed instrumentation for pristine FastWAM G0 runs.

This module intentionally lives outside the ``fastwam`` package.  A traced G0
worker imports policy/evaluation code from a separate, clean official checkout
and then installs observers around that already-imported module.  The observers
copy evidence but return the original prediction tuple, pass the original action
object to ``env.step``, and return the original environment result object.

The official checkout remains the policy source.  This file is instrumentation
source and must be content/commit bound separately by the caller.
"""

from __future__ import annotations

import contextvars
import hashlib
import importlib
import inspect
import itertools
import json
import math
import os
import random
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import numpy as np
import torch


OFFICIAL_FASTWAM_COMMIT = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
EXPECTED_ACTION_HORIZON = 32
EXPECTED_ACTION_DIMENSION = 7
EXPECTED_STATE_DIMENSION = 8
EXPECTED_WARMUP_STEPS = 30
EXPECTED_REPLAN_STEPS = 10
MIN_TRACE_RECORDS_PER_EPISODE = 7

OFFICIAL_CRITICAL_PATHS = (
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

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "HOME": "/tmp",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
}
FORMAL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
_TEMPORARY_COUNTER = itertools.count()
TRACE_TOP_LEVEL_KEYS = frozenset(("schema_version", "kind", "metadata", "records"))
TRACE_METADATA_KEYS = frozenset(
    (
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
    )
)
TRACE_RECORD_KEYS = frozenset(
    (
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
    )
)
TRACE_EXECUTION_KEYS = frozenset(
    ("action", "post_state", "post_observation_sha256", "done")
)


class InstrumentationError(RuntimeError):
    """Raised when the observer cannot prove its execution contract."""


def _lexical_absolute(path: Path) -> Path:
    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    if ".." in absolute.parts:
        raise InstrumentationError(f"parent traversal is forbidden: {path}")
    return Path(os.path.normpath(str(absolute)))


def _open_nofollow(path: Path, *, require_directory: bool = False) -> int:
    """Open every path component with openat/O_NOFOLLOW."""

    absolute = _lexical_absolute(path)
    parts = absolute.parts[1:]
    if not parts:
        if require_directory:
            return os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        raise InstrumentationError("root directory is not a regular file")
    directory_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for index, component in enumerate(parts):
            final = index == len(parts) - 1
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            if not final or require_directory:
                flags |= os.O_DIRECTORY
            try:
                next_fd = os.open(component, flags, dir_fd=directory_fd)
            except OSError as exc:
                raise InstrumentationError(
                    f"cannot open path without following symlinks: {absolute}: {exc}"
                ) from exc
            os.close(directory_fd)
            directory_fd = next_fd
        metadata = os.fstat(directory_fd)
        expected = stat.S_ISDIR(metadata.st_mode) if require_directory else stat.S_ISREG(metadata.st_mode)
        if not expected:
            kind = "directory" if require_directory else "regular file"
            raise InstrumentationError(f"path is not a {kind}: {absolute}")
        return directory_fd
    except Exception:
        os.close(directory_fd)
        raise


def _open_or_create_directory_nofollow(path: Path) -> int:
    """Open/create every directory component through one pinned parent fd.

    ``Path.mkdir(parents=True)`` resolves the path again for every syscall and
    can therefore follow a concurrently inserted symlink before the later
    no-follow open.  This helper keeps every parent directory descriptor open
    until its child has been opened with ``O_NOFOLLOW``.
    """

    absolute = _lexical_absolute(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for component in absolute.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o755, dir_fd=directory_fd)
                except FileExistsError:
                    # A concurrent creator is safe only if the subsequent
                    # no-follow open proves that it created a real directory.
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
        if isinstance(exc, InstrumentationError):
            raise
        raise InstrumentationError(
            f"cannot create/open directory without following symlinks: {absolute}: {exc}"
        ) from exc


def _read_regular_file_nofollow(path: Path, *, include_bytes: bool = False) -> dict[str, Any]:
    fd = _open_nofollow(path, require_directory=False)
    try:
        before = os.fstat(fd)
        digest = hashlib.sha256()
        content = bytearray() if include_bytes else None
        byte_count = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            digest.update(chunk)
            if content is not None:
                content.extend(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    stable_fields_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    stable_fields_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        stable_fields_before != stable_fields_after
        or byte_count != after.st_size
        or after.st_nlink != 1
    ):
        raise InstrumentationError(f"file changed during no-follow readback: {path}")
    receipt: dict[str, Any] = {
        "sha256": digest.hexdigest(),
        "size_bytes": byte_count,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mode": after.st_mode,
    }
    if content is not None:
        receipt["bytes"] = bytes(content)
    return receipt


def sha256_file(path: Path) -> str:
    return str(_read_regular_file_nofollow(path)["sha256"])


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256sum_posix_tree(inventory: list[dict[str, Any]]) -> str:
    """Hash ``<sha256>  <path>\n`` records in UTF-8 path order."""

    digest = hashlib.sha256()
    for item in sorted(inventory, key=lambda entry: entry["path"].encode("utf-8")):
        digest.update(f"{item['sha256']}  {item['path']}\n".encode("utf-8"))
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise InstrumentationError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InstrumentationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _loads_json_strict(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except InstrumentationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise InstrumentationError(f"cannot load strict JSON {label}: {exc}") from exc


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise InstrumentationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_bound_json(path: Path, trusted_sha256: str, *, label: str) -> dict[str, Any]:
    trusted_sha256 = _require_sha256(trusted_sha256, label=f"{label} trusted digest")
    readback = _read_regular_file_nofollow(path, include_bytes=True)
    if readback["sha256"] != trusted_sha256:
        raise InstrumentationError(
            f"{label} file digest mismatch: expected {trusted_sha256}, "
            f"observed {readback['sha256']}"
        )
    payload = _loads_json_strict(readback["bytes"], label=label)
    if not isinstance(payload, dict):
        raise InstrumentationError(f"{label} must be a JSON object")
    return {
        "path": str(_lexical_absolute(path)),
        "file_sha256": readback["sha256"],
        "size_bytes": readback["size_bytes"],
        "canonical_sha256": _canonical_sha256(payload),
        "payload": payload,
    }


def _expect_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InstrumentationError(f"{label} must be an object")
    return value


def load_upstream_artifact_bindings(
    *,
    run_id: str,
    preregistration_path: Path,
    preregistration_sha256: str,
    runtime_start_path: Path,
    runtime_start_sha256: str,
    seed_schedule_path: Path,
    seed_schedule_sha256: str,
    resolved_config_path: Path,
    resolved_config_sha256: str,
) -> dict[str, Any]:
    """Load and cross-bind the four immutable inputs required by a G0 worker.

    Every file is opened component-by-component with ``O_NOFOLLOW``.  JSON
    anchors are parsed strictly and bound by both their trusted file digest and
    canonical JSON digest; the resolved config is opaque and file-hash bound.
    """

    if not _RUN_ID_RE.fullmatch(run_id):
        raise InstrumentationError(
            "run_id must match ^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$"
        )
    prereg = _read_bound_json(
        preregistration_path, preregistration_sha256, label="preregistration"
    )
    start = _read_bound_json(
        runtime_start_path, runtime_start_sha256, label="runtime-start receipt"
    )
    schedule = _read_bound_json(
        seed_schedule_path, seed_schedule_sha256, label="seed schedule"
    )
    resolved_digest = _require_sha256(
        resolved_config_sha256, label="resolved config trusted digest"
    )
    resolved_readback = _read_regular_file_nofollow(resolved_config_path)
    if resolved_readback["sha256"] != resolved_digest:
        raise InstrumentationError(
            "resolved config file digest mismatch: "
            f"expected {resolved_digest}, observed {resolved_readback['sha256']}"
        )

    prereg_payload = prereg["payload"]
    start_payload = start["payload"]
    schedule_payload = schedule["payload"]
    if (
        prereg_payload.get("kind") != "mf_wam_g0_preregistration"
        or prereg_payload.get("phase") != "PREREGISTERED"
    ):
        raise InstrumentationError("invalid preregistration kind/phase")
    if prereg_payload.get("run_id") != run_id:
        raise InstrumentationError("preregistration run_id does not match worker run_id")
    if (
        start_payload.get("kind") != "mf_wam_g0_runtime_start"
        or start_payload.get("phase") != "STARTED"
    ):
        raise InstrumentationError("invalid runtime-start kind/phase")
    if start_payload.get("run_id") != run_id:
        raise InstrumentationError("runtime-start run_id does not match worker run_id")
    if start_payload.get("preregistration_canonical_sha256") != prereg["canonical_sha256"]:
        raise InstrumentationError("runtime-start does not bind this preregistration")
    if schedule_payload.get("kind") != "mf_wam_g0_task_process_seed_schedule":
        raise InstrumentationError("invalid seed-schedule kind")
    if schedule_payload.get("semantics") != "one-process-per-task-sequential-trials-v1":
        raise InstrumentationError("invalid seed-schedule semantics")

    artifacts = _expect_mapping(prereg_payload.get("artifacts"), label="preregistration.artifacts")
    prereg_resolved = _expect_mapping(
        artifacts.get("resolved_config"), label="preregistration.artifacts.resolved_config"
    )
    prereg_seeds = _expect_mapping(prereg_payload.get("seeds"), label="preregistration.seeds")
    start_bindings = _expect_mapping(start_payload.get("bindings"), label="runtime_start.bindings")
    if prereg_resolved.get("sha256") != resolved_digest:
        raise InstrumentationError("resolved config differs from preregistration binding")
    if prereg_resolved.get("size_bytes") != resolved_readback["size_bytes"]:
        raise InstrumentationError("resolved config size differs from preregistration binding")
    if start_bindings.get("resolved_config_sha256") != resolved_digest:
        raise InstrumentationError("resolved config differs from runtime-start binding")
    if prereg_seeds.get("schedule_canonical_sha256") != schedule["canonical_sha256"]:
        raise InstrumentationError("seed schedule differs from preregistration binding")
    if start_bindings.get("seed_schedule_canonical_sha256") != schedule["canonical_sha256"]:
        raise InstrumentationError("seed schedule differs from runtime-start binding")
    if schedule_payload.get("seed") != prereg_seeds.get("seed"):
        raise InstrumentationError("seed schedule root seed differs from preregistration")
    if schedule_payload.get("python_hash_seed") != prereg_seeds.get("python_hash_seed"):
        raise InstrumentationError("seed schedule Python hash seed differs from preregistration")

    task_processes = schedule_payload.get("task_processes")
    episodes = schedule_payload.get("episodes")
    if (
        schedule_payload.get("task_process_count") != 40
        or not isinstance(task_processes, list)
        or len(task_processes) != 40
    ):
        raise InstrumentationError("seed schedule must contain exactly 40 task processes")
    if (
        schedule_payload.get("episode_count") != 2000
        or not isinstance(episodes, list)
        or len(episodes) != 2000
    ):
        raise InstrumentationError("seed schedule must contain exactly 2000 episodes")
    process_keys: list[tuple[str, int]] = []
    for item in task_processes:
        if (
            not isinstance(item, Mapping)
            or item.get("task_suite") not in FORMAL_SUITES
            or type(item.get("task_id")) is not int
        ):
            raise InstrumentationError("seed schedule contains an invalid task process")
        process_keys.append((str(item["task_suite"]), int(item["task_id"])))
    expected_process_keys = [(suite, task_id) for suite in FORMAL_SUITES for task_id in range(10)]
    if len(set(process_keys)) != len(process_keys) or set(process_keys) != set(expected_process_keys):
        raise InstrumentationError("seed schedule task-process identities are incomplete or duplicated")
    episode_keys: list[tuple[str, int, int]] = []
    for item in episodes:
        if (
            not isinstance(item, Mapping)
            or item.get("task_suite") not in FORMAL_SUITES
            or type(item.get("task_id")) is not int
            or type(item.get("trial_idx")) is not int
        ):
            raise InstrumentationError("seed schedule contains an invalid episode identity")
        episode_keys.append(
            (str(item["task_suite"]), int(item["task_id"]), int(item["trial_idx"]))
        )
    expected_episode_keys = [
        (suite, task_id, trial_idx)
        for suite in FORMAL_SUITES
        for task_id in range(10)
        for trial_idx in range(50)
    ]
    if len(set(episode_keys)) != len(episode_keys) or set(episode_keys) != set(expected_episode_keys):
        raise InstrumentationError("seed schedule episode identities are incomplete or duplicated")

    output = _expect_mapping(prereg_payload.get("output"), label="preregistration.output")
    artifact_root_text = output.get("artifact_root")
    if not isinstance(artifact_root_text, str) or not artifact_root_text:
        raise InstrumentationError("preregistration.output.artifact_root is absent")
    artifact_root = _lexical_absolute(Path(artifact_root_text))
    if artifact_root.name != run_id or output.get("overwrite") is not False:
        raise InstrumentationError("invalid preregistered artifact root/overwrite policy")

    public = {
        "preregistration_file_sha256": prereg["file_sha256"],
        "preregistration_canonical_sha256": prereg["canonical_sha256"],
        "runtime_start_file_sha256": start["file_sha256"],
        "runtime_start_canonical_sha256": start["canonical_sha256"],
        "seed_schedule_file_sha256": schedule["file_sha256"],
        "seed_schedule_canonical_sha256": schedule["canonical_sha256"],
        "resolved_config_sha256": resolved_digest,
    }
    identities = {
        "preregistration": {key: prereg[key] for key in ("path", "file_sha256", "size_bytes", "canonical_sha256")},
        "runtime_start": {key: start[key] for key in ("path", "file_sha256", "size_bytes", "canonical_sha256")},
        "seed_schedule": {key: schedule[key] for key in ("path", "file_sha256", "size_bytes", "canonical_sha256")},
        "resolved_config": {
            "path": str(_lexical_absolute(resolved_config_path)),
            "file_sha256": resolved_digest,
            "size_bytes": resolved_readback["size_bytes"],
        },
    }
    return {
        "status": "PASS",
        "run_id": run_id,
        "artifact_root": str(artifact_root),
        "digests": public,
        "identities": identities,
        "documents": {
            "preregistration": prereg_payload,
            "runtime_start": start_payload,
            "seed_schedule": schedule_payload,
        },
    }


def _run_git(root: Path, *args: str) -> str:
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
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=_GIT_ENVIRONMENT,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise InstrumentationError(f"cannot inspect official Git checkout {root}: {exc}") from exc


def _run_git_bytes(root: Path, *args: str) -> bytes:
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
                *args,
            ],
            check=True,
            capture_output=True,
            timeout=30,
            env=_GIT_ENVIRONMENT,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise InstrumentationError(f"cannot read Git object in {root}: {exc}") from exc


def _git_ignored_entries(root: Path, label: str) -> tuple[str, ...]:
    raw = _run_git_bytes(
        root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
    )
    if raw and not raw.endswith(b"\0"):
        raise InstrumentationError(
            f"{label} ignored-file inventory is not NUL terminated"
        )
    entries: list[str] = []
    observed: set[str] = set()
    for field in raw[:-1].split(b"\0") if raw else []:
        try:
            relative = field.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise InstrumentationError(
                f"{label} ignored-file path is not strict UTF-8"
            ) from exc
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or pure.as_posix() != relative
            or any(part in ("", ".", "..") for part in pure.parts)
        ):
            raise InstrumentationError(
                f"unsafe {label} ignored-file path: {relative!r}"
            )
        if relative in observed:
            raise InstrumentationError(
                f"{label} ignored-file inventory contains duplicates"
            )
        observed.add(relative)
        entries.append(relative)
    return tuple(entries)


def _verify_git_checkout_policy(root: Path, label: str) -> None:
    expected_git_dir = root / ".git"
    git_dir_fd = _open_nofollow(expected_git_dir, require_directory=True)
    os.close(git_dir_fd)
    local_config_keys = _run_git(
        root, "config", "--local", "--no-includes", "--name-only", "--list"
    ).splitlines()
    forbidden_config = [
        key
        for key in local_config_keys
        if key.lower().startswith(("filter.", "include.", "includeif."))
        or key.lower() == "core.attributesfile"
    ]
    if forbidden_config:
        raise InstrumentationError(
            f"{label} repository-local filters/includes are forbidden: "
            f"{forbidden_config[:5]!r}"
        )
    info_fd = _open_nofollow(expected_git_dir / "info", require_directory=True)
    try:
        try:
            os.stat("attributes", dir_fd=info_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise InstrumentationError(f"{label} .git/info/attributes is forbidden")
    finally:
        os.close(info_fd)
    if _run_git(root, "rev-parse", "--show-toplevel").strip() != str(root):
        raise InstrumentationError(
            f"{label} Git top-level differs from its source root"
        )
    absolute_git_dir = _run_git(root, "rev-parse", "--absolute-git-dir").strip()
    common_git_dir = _run_git(
        root, "rev-parse", "--path-format=absolute", "--git-common-dir"
    ).strip()
    if absolute_git_dir != str(expected_git_dir) or common_git_dir != str(
        expected_git_dir
    ):
        raise InstrumentationError(
            f"{label} linked worktrees or external Git directories are forbidden"
        )
    if _run_git(root, "rev-parse", "--show-object-format").strip() != "sha1":
        raise InstrumentationError(f"{label} Git object format must be sha1")
    replacements = _run_git(
        root, "for-each-ref", "--format=%(refname)", "refs/replace/"
    ).splitlines()
    if replacements:
        raise InstrumentationError(
            f"{label} Git replace refs are forbidden: {replacements[:5]!r}"
        )
    ignored = _git_ignored_entries(root, label)
    if ignored:
        raise InstrumentationError(
            f"{label} source root contains gitignored artifacts: {ignored[:5]!r}"
        )


def _verify_exact_commit_tree(root: Path, expected_commit: str, label: str) -> None:
    raw_tree = _run_git_bytes(
        root, "ls-tree", "-r", "-z", "--full-tree", expected_commit
    )
    if raw_tree and not raw_tree.endswith(b"\0"):
        raise InstrumentationError(
            f"{label} Git tree inventory is not NUL terminated"
        )
    observed_paths: set[str] = set()
    for record in raw_tree[:-1].split(b"\0") if raw_tree else []:
        try:
            metadata, path_raw = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.split(b" ", 2)
            relative = path_raw.decode("utf-8", errors="strict")
        except (ValueError, UnicodeError) as exc:
            raise InstrumentationError(f"{label} Git tree record is malformed") from exc
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or pure.as_posix() != relative
            or any(part in ("", ".", "..") for part in pure.parts)
        ):
            raise InstrumentationError(
                f"unsafe {label} tracked source path: {relative!r}"
            )
        if relative in observed_paths:
            raise InstrumentationError(f"{label} Git tree contains duplicate paths")
        observed_paths.add(relative)
        if mode not in (b"100644", b"100755") or object_type != b"blob":
            raise InstrumentationError(
                f"{label} symlink, gitlink, or non-regular tracked source is forbidden: "
                f"{relative}"
            )
        try:
            object_id_text = object_id.decode("ascii", errors="strict")
        except UnicodeError as exc:
            raise InstrumentationError(
                f"{label} Git blob identity is malformed"
            ) from exc
        if not _COMMIT_RE.fullmatch(object_id_text):
            raise InstrumentationError(f"{label} Git blob identity is malformed")
        worktree = _read_regular_file_nofollow(root / pure, include_bytes=True)
        observed_object_id = hashlib.sha1(
            f"blob {len(worktree['bytes'])}\0".encode("ascii")
            + worktree["bytes"]
        ).hexdigest()
        executable = bool(worktree["mode"] & 0o111)
        if observed_object_id != object_id_text or executable != (mode == b"100755"):
            raise InstrumentationError(
                f"{label} tracked worktree differs from exact commit tree: {relative}"
            )
    if not observed_paths:
        raise InstrumentationError(f"{label} exact commit tree is empty")


def verify_pristine_official_root(
    root: Path,
    *,
    expected_commit: str = OFFICIAL_FASTWAM_COMMIT,
    critical_paths: tuple[str, ...] = OFFICIAL_CRITICAL_PATHS,
) -> dict[str, Any]:
    """Verify and inventory a clean official checkout before importing it."""

    root = _lexical_absolute(root)
    if not _COMMIT_RE.fullmatch(expected_commit):
        raise InstrumentationError(f"invalid expected official commit: {expected_commit!r}")
    root_fd = _open_nofollow(root, require_directory=True)
    os.close(root_fd)
    _verify_git_checkout_policy(root, "source")

    head = _run_git(root, "rev-parse", "HEAD").strip()
    if head != expected_commit:
        raise InstrumentationError(
            f"official source HEAD mismatch: expected {expected_commit}, observed {head}"
        )
    porcelain = _run_git(root, "status", "--porcelain", "--untracked-files=all")
    if porcelain:
        raise InstrumentationError(
            "official source worktree is not clean: " + repr(porcelain.splitlines())
        )
    index_rows = _run_git(root, "ls-files", "-v", "-z").split("\0")
    if any(row and not row.startswith("H ") for row in index_rows):
        raise InstrumentationError(
            "source Git index contains assume-unchanged or skip-worktree entries"
        )
    _verify_exact_commit_tree(root, expected_commit, "source")

    tree = _run_git(root, "rev-parse", "HEAD^{tree}").strip()
    files: list[dict[str, Any]] = []
    for relative in critical_paths:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise InstrumentationError(f"invalid critical source path: {relative}")
        path = root / relative_path
        worktree = _read_regular_file_nofollow(path, include_bytes=True)
        blob = _run_git(root, "rev-parse", f"HEAD:{relative}").strip()
        git_content = _run_git_bytes(root, "cat-file", "blob", blob)
        if worktree["bytes"] != git_content:
            raise InstrumentationError(
                f"official critical file differs from its Git object: {relative}"
            )
        files.append(
            {
                "path": relative_path.as_posix(),
                "git_blob": blob,
                "sha256": worktree["sha256"],
                "size_bytes": worktree["size_bytes"],
                "git_blob_content_sha256": hashlib.sha256(git_content).hexdigest(),
            }
        )

    terminal_head = _run_git(root, "rev-parse", "HEAD").strip()
    terminal_tree = _run_git(root, "rev-parse", "HEAD^{tree}").strip()
    terminal_porcelain = _run_git(
        root, "status", "--porcelain", "--untracked-files=all"
    )
    _verify_git_checkout_policy(root, "source")
    terminal_index_rows = _run_git(root, "ls-files", "-v", "-z").split("\0")
    if (
        terminal_head != head
        or terminal_tree != tree
        or terminal_porcelain
        or any(row and not row.startswith("H ") for row in terminal_index_rows)
    ):
        raise InstrumentationError("source Git identity changed during critical-file readback")

    return {
        "status": "PASS",
        "role": "official_policy_and_evaluator_source",
        "root": str(root),
        "commit": head,
        "tree": tree,
        "clean": True,
        "critical_files": files,
        "critical_file_inventory_sha256": _canonical_sha256(files),
    }


def verify_pristine_instrumentation_root(
    root: Path,
    *,
    expected_commit: str,
    critical_paths: tuple[str, ...] = (
        "scripts/mf_wam_g0_instrumentation.py",
        "scripts/run_mf_wam_g0_traced.py",
    ),
) -> dict[str, Any]:
    """Bind the external observer/runner to its own clean Git identity."""

    # Reuse the same strict Git/readback checks, then relabel the source role.
    try:
        receipt = verify_pristine_official_root(
            root,
            expected_commit=expected_commit,
            critical_paths=critical_paths,
        )
    except InstrumentationError as exc:
        raise InstrumentationError(
            f"instrumentation source verification failed: {exc}"
        ) from exc
    receipt["role"] = "external_observer_and_launcher_source"
    return receipt


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def audit_module_origins(
    official_root: Path,
    *,
    modules: Mapping[str, ModuleType] | None = None,
) -> dict[str, Any]:
    """Require every loaded FastWAM/LIBERO-eval implementation module to be official."""

    official_root = _lexical_absolute(official_root)
    root_fd = _open_nofollow(official_root, require_directory=True)
    os.close(root_fd)
    module_map = sys.modules if modules is None else modules
    origins: list[dict[str, str]] = []
    offending: list[dict[str, str]] = []
    for name, module in sorted(module_map.items()):
        relevant = (
            name == "fastwam"
            or name.startswith("fastwam.")
            or name == "experiments.libero"
            or name.startswith("experiments.libero.")
            or name == "action_ensembler"
        )
        if not relevant or module is None:
            continue
        origin_text = getattr(module, "__file__", None)
        if not origin_text:
            continue
        origin = _lexical_absolute(Path(origin_text))
        item = {"module": name, "origin": str(origin)}
        origins.append(item)
        if not _is_within(origin, official_root):
            offending.append(item)
            continue
        relative = origin.relative_to(official_root).as_posix()
        try:
            worktree = _read_regular_file_nofollow(origin, include_bytes=True)
            marker = _run_git(
                official_root, "ls-files", "-v", "--", relative
            ).strip()
            blob = _run_git(
                official_root, "rev-parse", f"HEAD:{relative}"
            ).strip()
            git_content = _run_git_bytes(
                official_root, "cat-file", "blob", blob
            )
        except InstrumentationError:
            raise
        except Exception as exc:
            raise InstrumentationError(
                f"cannot bind loaded module origin to Git: {name}: {exc}"
            ) from exc
        if marker != f"H {relative}" or worktree["bytes"] != git_content:
            raise InstrumentationError(
                f"loaded module origin is not an exact tracked HEAD blob: {name}"
            )

    if offending:
        raise InstrumentationError(
            "non-official FastWAM evaluation modules are loaded: " + repr(offending)
        )
    if not any(item["module"] == "experiments.libero.eval_libero_single" for item in origins):
        raise InstrumentationError("official eval_libero_single module is not loaded")
    if not any(item["module"] == "fastwam" or item["module"].startswith("fastwam.") for item in origins):
        raise InstrumentationError("no official fastwam implementation module is loaded")

    return {
        "status": "PASS",
        "official_root": str(official_root),
        "module_count": len(origins),
        "modules": origins,
        "inventory_sha256": _canonical_sha256(origins),
    }


def import_pristine_official_eval(official_root: Path) -> ModuleType:
    """Import the official eval module without importing instrumentation as policy code."""

    official_root = official_root.expanduser().resolve()
    eval_path = official_root / "experiments/libero/eval_libero_single.py"
    if not eval_path.is_file():
        raise InstrumentationError(f"official eval module is absent: {eval_path}")

    # ``eval_libero_single.py`` imports action_ensembler as a top-level module.
    # Put both official locations ahead of the runner directory/current cwd.
    official_entries = [
        str(official_root / "experiments/libero"),
        str(official_root / "src"),
        str(official_root),
    ]
    for entry in reversed(official_entries):
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    importlib.invalidate_caches()

    module = importlib.import_module("experiments.libero.eval_libero_single")
    observed = Path(str(module.__file__)).expanduser().resolve()
    if observed != eval_path.resolve():
        raise InstrumentationError(
            f"eval module origin mismatch: expected {eval_path.resolve()}, observed {observed}"
        )
    audit_module_origins(official_root)
    return module


def _update_array_hash(digest: Any, name: str, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise InstrumentationError(f"object-dtype observation is not auditable: {name}")
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    digest.update(b"\n")


def array_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_array_hash(digest, "array", value)
    return digest.hexdigest()


def observation_sha256(obs: Mapping[str, Any]) -> str:
    if not isinstance(obs, Mapping) or not obs:
        raise InstrumentationError("observation must be a non-empty mapping")
    digest = hashlib.sha256()
    for key in sorted(obs):
        _update_array_hash(digest, str(key), obs[key])
    return digest.hexdigest()


def extract_sim_state_readonly(obs: Mapping[str, Any]) -> np.ndarray:
    """Match official state semantics without mutating the observation quaternion."""

    try:
        position = np.asarray(obs["robot0_eef_pos"]).copy()
        quaternion = np.asarray(obs["robot0_eef_quat"]).copy()
        gripper = np.asarray(obs["robot0_gripper_qpos"]).copy()
    except (KeyError, TypeError, ValueError) as exc:
        raise InstrumentationError(f"cannot extract simulator state: {exc}") from exc
    if position.shape != (3,) or quaternion.shape != (4,) or gripper.shape != (2,):
        raise InstrumentationError(
            "unexpected simulator state shapes: "
            f"position={position.shape}, quaternion={quaternion.shape}, gripper={gripper.shape}"
        )

    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - float(quaternion[3]) ** 2))
    if math.isclose(denominator, 0.0):
        axis_angle = np.zeros(3)
    else:
        axis_angle = (
            quaternion[:3] * 2.0 * math.acos(float(quaternion[3])) / denominator
        )
    state = np.concatenate((position, axis_angle, gripper)).astype(np.float32)
    if state.shape != (EXPECTED_STATE_DIMENSION,) or not np.isfinite(state).all():
        raise InstrumentationError(f"invalid simulator state: shape={state.shape}")
    return state


def _rng_fingerprint() -> str:
    """Digest global Python/NumPy/Torch RNG states without drawing from them."""

    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode("ascii"))
    numpy_state = np.random.get_state()
    digest.update(str(numpy_state[0]).encode("ascii"))
    digest.update(np.ascontiguousarray(numpy_state[1]).tobytes())
    digest.update(str(numpy_state[2:]).encode("ascii"))
    digest.update(torch.get_rng_state().cpu().numpy().tobytes())
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            digest.update(str(index).encode("ascii"))
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


def _effective_global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(
        os.environ.get(
            "RANK",
            os.environ.get("SLURM_PROCID", os.environ.get("LOCAL_RANK", "0")),
        )
    )


def _execution_ranks() -> dict[str, int]:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = int(torch.distributed.get_world_size())
            global_rank = int(torch.distributed.get_rank())
        else:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            global_rank = _effective_global_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except (TypeError, ValueError) as exc:
        raise InstrumentationError("G0 rank environment must contain integers") from exc
    ranks = {
        "world_size": world_size,
        "global_rank": global_rank,
        "local_rank": local_rank,
    }
    if ranks != {"world_size": 1, "global_rank": 0, "local_rank": 0}:
        raise InstrumentationError(f"G0 task worker requires 1/0/0 ranks, observed {ranks}")
    return ranks


def _atomic_publish_bytes_no_replace(path: Path, payload: bytes, *, label: str) -> None:
    """Publish through a pinned no-follow parent fd and never replace a name."""

    path = _lexical_absolute(path)
    if not path.name:
        raise InstrumentationError(f"{label} path must name a file: {path}")
    parent_fd = _open_or_create_directory_nofollow(path.parent)
    temporary_name = (
        f".{path.name}.{os.getpid()}.{next(_TEMPORARY_COUNTER):016x}.tmp"
    )
    descriptor: int | None = None
    temporary_created = False
    try:
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
                dir_fd=parent_fd,
            )
            temporary_created = True
        except FileExistsError as exc:
            raise InstrumentationError(
                f"unexpected temporary artifact collision: {temporary_name}"
            ) from exc
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o644)
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise InstrumentationError(f"refusing to replace existing {label}: {path}") from exc
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


def _atomic_write_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish JSON without ever replacing an existing trace."""

    try:
        encoded = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InstrumentationError(f"cannot encode JSON artifact: {exc}") from exc
    _atomic_publish_bytes_no_replace(path, encoded, label="trace")


def _atomic_write_bytes_no_replace(path: Path, payload: bytes) -> None:
    """Atomically publish already-validated bytes without replacing a target."""

    _atomic_publish_bytes_no_replace(path, payload, label="artifact")


def _assert_resolved_within(path: Path, root: Path, *, label: str) -> tuple[Path, Path]:
    root = _lexical_absolute(root)
    path = _lexical_absolute(path)
    root.mkdir(parents=True, exist_ok=True)
    root_fd = _open_nofollow(root, require_directory=True)
    os.close(root_fd)
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise InstrumentationError(
            f"{label} escapes its locked root: path={resolved_path}, root={resolved_root}"
        ) from exc
    return resolved_path, resolved_root


def verify_process_receipt_trace_inventory(
    receipt_path: Path,
    *,
    output_root: Path | None = None,
    expected_trace_count: int = 50,
) -> dict[str, Any]:
    """Read back every trace bound by a terminal instrumentation receipt."""

    receipt_path = _lexical_absolute(receipt_path)
    try:
        receipt_bytes = _read_regular_file_nofollow(receipt_path, include_bytes=True)["bytes"]
        receipt = _loads_json_strict(receipt_bytes, label=str(receipt_path))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InstrumentationError(f"cannot read process receipt {receipt_path}: {exc}") from exc
    expected_receipt_keys = {
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
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected_receipt_keys
        or receipt.get("kind") != "mf_wam_g0_task_trace_receipt"
    ):
        raise InstrumentationError("invalid instrumentation process receipt")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("execution_scope") != "one-process-per-task"
        or receipt.get("world_size") != 1
        or receipt.get("global_rank") != 0
        or receipt.get("local_rank") != 0
    ):
        raise InstrumentationError("invalid execution scope/ranks in task trace receipt")
    suite = receipt.get("task_suite")
    task_id = receipt.get("task_id")
    if suite not in FORMAL_SUITES or type(task_id) is not int or not 0 <= task_id <= 9:
        raise InstrumentationError("invalid task identity in instrumentation process receipt")
    if receipt.get("process_id") != f"{suite}/task{task_id:02d}":
        raise InstrumentationError("invalid process identity in instrumentation process receipt")
    bindings = receipt.get("bindings")
    expected_binding_keys = {
        "preregistration_canonical_sha256",
        "runtime_start_canonical_sha256",
        "seed_schedule_canonical_sha256",
        "resolved_config_sha256",
        "image_digest",
        "fastwam_commit",
        "instrumentation_commit",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != expected_binding_keys:
        raise InstrumentationError("invalid bindings object in task trace receipt")
    for key in (
        "preregistration_canonical_sha256",
        "runtime_start_canonical_sha256",
        "seed_schedule_canonical_sha256",
        "resolved_config_sha256",
    ):
        _require_sha256(bindings[key], label=f"receipt.bindings.{key}")
    if (
        not isinstance(bindings.get("image_digest"), str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", bindings["image_digest"])
        or not isinstance(bindings.get("fastwam_commit"), str)
        or not _COMMIT_RE.fullmatch(bindings["fastwam_commit"])
        or not isinstance(bindings.get("instrumentation_commit"), str)
        or not _COMMIT_RE.fullmatch(bindings["instrumentation_commit"])
    ):
        raise InstrumentationError("invalid image/source binding in task trace receipt")
    seeds = receipt.get("seeds")
    expected_seed_keys = {
        "global_seed",
        "environment_seed",
        "environment_seed_scope",
        "policy_seed",
        "policy_seed_scope",
        "python_hash_seed",
        "trial_order",
        "initial_state_index_rule",
    }
    if (
        not isinstance(seeds, Mapping)
        or set(seeds) != expected_seed_keys
        or any(type(seeds.get(key)) is not int for key in (
            "global_seed", "environment_seed", "policy_seed", "python_hash_seed"
        ))
        or seeds.get("environment_seed_scope") != "once-before-trial-0"
        or seeds.get("policy_seed_scope") != "constant-each-replan-call"
        or seeds.get("trial_order") != list(range(expected_trace_count))
        or seeds.get("initial_state_index_rule") != "trial_idx"
    ):
        raise InstrumentationError("invalid seeds object in task trace receipt")
    root = (
        receipt_path.parents[2]
        if output_root is None
        else _lexical_absolute(output_root)
    )
    expected_receipt_path = root / "trace_receipts" / suite / f"task{task_id:02d}.json"
    if receipt_path != expected_receipt_path:
        raise InstrumentationError("task trace receipt is not at its canonical path")
    inventory = receipt.get("traces")
    if not isinstance(inventory, list) or not inventory:
        raise InstrumentationError("process receipt has no trace inventory")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(inventory):
        if not isinstance(item, Mapping):
            raise InstrumentationError(f"invalid trace inventory entry {index}")
        trial_idx = item.get("trial_idx")
        relative = item.get("path")
        expected_sha = item.get("sha256")
        expected_size = item.get("size_bytes")
        if (
            type(trial_idx) is not int
            or not 0 <= trial_idx < expected_trace_count
            or not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen
            or not isinstance(expected_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or type(expected_size) is not int
            or expected_size < 0
        ):
            raise InstrumentationError(f"invalid trace inventory entry {index}")
        path = root / relative
        expected_relative = f"traces/{suite}/task{task_id:02d}/trial{trial_idx:03d}.json"
        if relative != expected_relative:
            raise InstrumentationError(f"non-canonical episode trace path: {relative}")
        readback = _read_regular_file_nofollow(path)
        observed = {
            "trial_idx": trial_idx,
            "path": relative,
            "sha256": readback["sha256"],
            "size_bytes": readback["size_bytes"],
        }
        if observed != dict(item):
            raise InstrumentationError(
                f"receipt-bound trace content mismatch: {relative}"
            )
        seen.add(relative)
        normalized.append(observed)
    if normalized != sorted(normalized, key=lambda item: item["trial_idx"]):
        raise InstrumentationError("trace inventory is not sorted by trial_idx")
    if [item["trial_idx"] for item in normalized] != list(range(expected_trace_count)):
        raise InstrumentationError("trace inventory trial scope is incomplete or duplicated")
    observed_tree = _sha256sum_posix_tree(normalized)
    if receipt.get("tree_sha256") != observed_tree:
        raise InstrumentationError("trace inventory tree digest mismatch")
    if receipt.get("episode_count") != len(normalized):
        raise InstrumentationError("trace inventory count mismatch")
    if len(normalized) != expected_trace_count:
        raise InstrumentationError(
            f"trace inventory must contain {expected_trace_count} entries, "
            f"observed {len(normalized)}"
        )
    official_result = receipt.get("official_result")
    if not isinstance(official_result, Mapping) or set(official_result) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise InstrumentationError("invalid official_result binding in process receipt")
    result_path = official_result.get("path")
    result_sha = official_result.get("sha256")
    result_size = official_result.get("size_bytes")
    if (
        not isinstance(result_path, str)
        or not result_path
        or Path(result_path).is_absolute()
        or ".." in Path(result_path).parts
        or not isinstance(result_sha, str)
        or not _SHA256_RE.fullmatch(result_sha)
        or type(result_size) is not int
        or result_size < 1
    ):
        raise InstrumentationError("invalid result binding in process receipt")
    if result_path != f"results/{suite}/task{task_id:02d}.json":
        raise InstrumentationError("non-canonical result path in process receipt")
    result_readback = _read_regular_file_nofollow(root / result_path)
    if (
        result_readback["sha256"] != result_sha
        or result_readback["size_bytes"] != result_size
    ):
        raise InstrumentationError("receipt-bound official result content mismatch")
    return {
        "status": "PASS",
        "trace_count": len(normalized),
        "tree_sha256": observed_tree,
        "result_sha256": result_sha,
    }


def _numeric_matrix(value: Any, *, rows: int | None, columns: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != columns or (rows is not None and array.shape[0] != rows):
        expected_rows = "*" if rows is None else str(rows)
        raise InstrumentationError(
            f"{label} must have shape [{expected_rows},{columns}], observed {array.shape}"
        )
    try:
        finite = np.isfinite(array).all()
    except TypeError as exc:
        raise InstrumentationError(f"{label} is not numeric") from exc
    if not finite:
        raise InstrumentationError(f"{label} contains non-finite values")
    return np.ascontiguousarray(array)


def validate_structured_trace_payload(payload: Any) -> dict[str, Any]:
    """Validate the exact schema-v2 trace shape before publication/readback."""

    if not isinstance(payload, dict) or set(payload) != TRACE_TOP_LEVEL_KEYS:
        raise InstrumentationError("structured trace has unknown or missing top-level fields")
    if payload.get("schema_version") != 2 or payload.get("kind") != "mf_wam_g0_structured_trace":
        raise InstrumentationError("structured trace kind/schema is invalid")
    metadata = payload.get("metadata")
    records = payload.get("records")
    if not isinstance(metadata, dict) or set(metadata) != TRACE_METADATA_KEYS:
        raise InstrumentationError("structured trace metadata has unknown or missing fields")
    if not isinstance(records, list) or not records:
        raise InstrumentationError("structured trace records must be a non-empty list")
    if metadata.get("record_count") != len(records):
        raise InstrumentationError("structured trace record_count mismatch")
    if type(metadata.get("success")) is not bool:
        raise InstrumentationError("structured trace success must be boolean")
    for digest_name in (
        "initial_state_sha256",
        "official_module_origin_inventory_sha256",
    ):
        _require_sha256(metadata.get(digest_name), label=f"trace.metadata.{digest_name}")

    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != TRACE_RECORD_KEYS:
            raise InstrumentationError(
                f"structured trace record {index} has unknown or missing fields"
            )
        if "raw_action_chunk" in record:
            raise InstrumentationError("legacy raw_action_chunk is forbidden in schema v2")
        if (
            record.get("episode_idx") != metadata.get("trial_idx")
            or record.get("replan_idx") != index
            or record.get("env_step") != EXPECTED_WARMUP_STEPS + index * EXPECTED_REPLAN_STEPS
            or record.get("policy_seed_scope") != "fresh_generator_per_replan"
            or type(record.get("executed_count")) is not int
            or type(record.get("done_after_execution")) is not bool
        ):
            raise InstrumentationError(f"structured trace record {index} identity/cadence is invalid")
        _numeric_matrix(
            record.get("proposed_raw_action_chunk"),
            rows=EXPECTED_ACTION_HORIZON,
            columns=EXPECTED_ACTION_DIMENSION,
            label=f"record {index} proposed_raw_action_chunk",
        )
        proposal = _numeric_matrix(
            record.get("proposed_env_action_chunk"),
            rows=EXPECTED_ACTION_HORIZON,
            columns=EXPECTED_ACTION_DIMENSION,
            label=f"record {index} proposed_env_action_chunk",
        )
        executed = _numeric_matrix(
            record.get("executed_env_actions"),
            rows=None,
            columns=EXPECTED_ACTION_DIMENSION,
            label=f"record {index} executed_env_actions",
        )
        if not 1 <= len(executed) <= EXPECTED_REPLAN_STEPS:
            raise InstrumentationError(f"record {index} executed prefix length is invalid")
        if record["executed_count"] != len(executed) or not np.array_equal(
            executed, proposal[: len(executed)]
        ):
            raise InstrumentationError(f"record {index} executed actions are not a proposal prefix")
        for state_name in ("state", "pre_state"):
            state_value = np.asarray(record.get(state_name))
            if state_value.shape != (EXPECTED_STATE_DIMENSION,) or not np.isfinite(state_value).all():
                raise InstrumentationError(f"record {index} {state_name} is invalid")
        _require_sha256(
            record.get("pre_observation_sha256"),
            label=f"record {index} pre_observation_sha256",
        )
        executions = record.get("executions")
        if not isinstance(executions, list) or len(executions) != len(executed):
            raise InstrumentationError(f"record {index} execution list length is invalid")
        for execution_idx, execution in enumerate(executions):
            if not isinstance(execution, dict) or set(execution) != TRACE_EXECUTION_KEYS:
                raise InstrumentationError(
                    f"record {index} execution {execution_idx} has unknown or missing fields"
                )
            action = np.asarray(execution.get("action"))
            state_value = np.asarray(execution.get("post_state"))
            if (
                action.shape != (EXPECTED_ACTION_DIMENSION,)
                or not np.isfinite(action).all()
                or not np.array_equal(action, executed[execution_idx])
                or state_value.shape != (EXPECTED_STATE_DIMENSION,)
                or not np.isfinite(state_value).all()
                or type(execution.get("done")) is not bool
            ):
                raise InstrumentationError(
                    f"record {index} execution {execution_idx} content is invalid"
                )
            _require_sha256(
                execution.get("post_observation_sha256"),
                label=f"record {index} execution {execution_idx} post_observation_sha256",
            )
    return payload


@dataclass
class _EpisodeCapture:
    cfg: Any
    episode_idx: int
    task_description: str
    initial_state_sha256: str
    metadata: dict[str, Any]
    records: list[dict[str, Any]] = field(default_factory=list)
    env_step_count: int = 0
    warmup_count: int = 0
    observer_rng_checks: int = 0
    forwarded_action_object_ids: list[int] = field(default_factory=list)
    current_record: dict[str, Any] | None = None
    pending_raw_proposal: np.ndarray | None = None


class _StepProxy:
    """Observe env.step while preserving the exact argument and return objects."""

    def __init__(self, env: Any, owner: "G0TraceInstrumentation", capture: _EpisodeCapture):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_capture", capture)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def step(self, action: Any) -> Any:
        capture = self._capture
        if capture.current_record is None:
            if capture.records:
                raise InstrumentationError("env.step occurred between closed replan records")
            if capture.warmup_count >= EXPECTED_WARMUP_STEPS:
                raise InstrumentationError("env.step occurred after warmup without a proposal")
            warmup = self._owner._observe_without_rng_draw(
                "copy_warmup_action", lambda: _numeric_matrix([action], rows=1, columns=7, label="warmup action")
            )[0]
            expected_dummy = np.asarray([0, 0, 0, 0, 0, 0, -1])
            if not np.array_equal(warmup, expected_dummy):
                raise InstrumentationError(f"unexpected warmup action: {warmup.tolist()}")
            capture.warmup_count += 1
            result = self._env.step(action)
            capture.env_step_count += 1
            return result

        record = capture.current_record
        executions = record["executions"]
        if len(executions) >= EXPECTED_REPLAN_STEPS:
            raise InstrumentationError("more than 10 actions executed from one proposal")
        action_copy = self._owner._observe_without_rng_draw(
            "copy_executed_action",
            lambda: _numeric_matrix([action], rows=1, columns=7, label="executed action")[0].copy(),
        )
        incoming_object_id = id(action)
        result = self._env.step(action)
        capture.forwarded_action_object_ids.append(incoming_object_id)
        if not isinstance(result, tuple) or len(result) < 4:
            raise InstrumentationError("env.step must return at least (obs, reward, done, info)")
        post_obs = result[0]
        done = bool(result[2])
        post_payload = self._owner._observe_without_rng_draw(
            "capture_post_step",
            lambda: {
                "action": action_copy.tolist(),
                "post_state": extract_sim_state_readonly(post_obs).tolist(),
                "post_observation_sha256": observation_sha256(post_obs),
                "done": done,
            },
        )
        executions.append(post_payload)
        record["executed_env_actions"].append(post_payload["action"])
        record["executed_count"] = len(executions)
        record["done_after_execution"] = done
        capture.env_step_count += 1
        return result


class G0TraceInstrumentation:
    """Install and manage transparent observers around an official eval module."""

    def __init__(
        self,
        official_module: ModuleType,
        *,
        official_root: Path,
        official_identity: Mapping[str, Any],
        trace_root: Path | None = None,
        run_id: str,
        instrumentation_identity: Mapping[str, Any],
        upstream_bindings: Mapping[str, Any],
        enforce_module_origins: bool = True,
        _test_expected_trial_count: int | None = None,
        _test_terminal_source_verifier: Callable[
            [], tuple[Mapping[str, Any], Mapping[str, Any]]
        ]
        | None = None,
        _test_terminal_upstream_verifier: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise InstrumentationError(
                "run_id must match ^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$"
            )
        upstream = dict(upstream_bindings)
        if upstream.get("status") != "PASS" or upstream.get("run_id") != run_id:
            raise InstrumentationError("upstream artifact bindings are absent or use another run_id")
        artifact_root_text = upstream.get("artifact_root")
        digests = upstream.get("digests")
        identities = upstream.get("identities")
        documents = upstream.get("documents")
        if (
            not isinstance(artifact_root_text, str)
            or not isinstance(digests, Mapping)
            or not isinstance(identities, Mapping)
            or not isinstance(documents, Mapping)
        ):
            raise InstrumentationError("upstream artifact bindings are structurally incomplete")
        required_digests = {
            "preregistration_file_sha256",
            "preregistration_canonical_sha256",
            "runtime_start_file_sha256",
            "runtime_start_canonical_sha256",
            "seed_schedule_file_sha256",
            "seed_schedule_canonical_sha256",
            "resolved_config_sha256",
        }
        if set(digests) != required_digests:
            raise InstrumentationError("upstream digest binding set is incomplete")
        for name, digest in digests.items():
            _require_sha256(digest, label=f"upstream digest {name}")
        for name in ("preregistration", "runtime_start", "seed_schedule"):
            if not isinstance(documents.get(name), Mapping):
                raise InstrumentationError(f"upstream {name} document is absent")
        self.module = official_module
        self.official_root = _lexical_absolute(official_root)
        self.official_identity = dict(official_identity)
        self.run_id = run_id
        self.instrumentation_identity = dict(instrumentation_identity)
        self.upstream_bindings = upstream
        self.upstream_digests = dict(digests)
        self.upstream_identities = {str(key): dict(value) for key, value in identities.items()}
        self.upstream_documents = {str(key): dict(value) for key, value in documents.items()}
        self.artifact_root = _lexical_absolute(Path(artifact_root_text))
        canonical_trace_root = self.artifact_root / "traces"
        if trace_root is not None and _lexical_absolute(trace_root) != canonical_trace_root:
            raise InstrumentationError(
                f"trace root must be the canonical artifact path {canonical_trace_root}"
            )
        self.explicit_trace_root = canonical_trace_root
        self.enforce_module_origins = enforce_module_origins
        self.expected_trial_count = 50 if _test_expected_trial_count is None else int(
            _test_expected_trial_count
        )
        if not 1 <= self.expected_trial_count <= 50:
            raise InstrumentationError("expected trial count must be in [1, 50]")
        self._test_terminal_source_verifier = _test_terminal_source_verifier
        self._test_terminal_upstream_verifier = _test_terminal_upstream_verifier
        self._episode_var: contextvars.ContextVar[_EpisodeCapture | None] = contextvars.ContextVar(
            "mf_wam_g0_episode", default=None
        )
        self._installed = False
        self._originals: dict[str, Callable[..., Any]] = {}
        self._global_seed_calls: list[dict[str, int]] = []
        self._environment_seed_calls: list[dict[str, Any]] = []
        self._completed_trace_paths: list[Path] = []
        self._completed_trace_receipts: dict[int, dict[str, Any]] = {}
        self._completed_trial_indices: set[int] = set()
        self._trace_success_by_trial: dict[int, bool] = {}
        self._last_task_identity: tuple[str, int] | None = None
        self._last_output_root: Path | None = None
        self._official_result_source_path: Path | None = None
        self._last_gpu_id: int | None = None
        self._official_result_receipt: dict[str, Any] | None = None
        self._bound_seed_process: dict[str, Any] | None = None
        self._execution_rank_receipt: dict[str, int] | None = None
        self._module_origin_receipt: dict[str, Any] = {
            "status": "SKIPPED_FOR_TEST",
            "inventory_sha256": "0" * 64,
            "modules": [],
        }

        prereg = self.upstream_documents["preregistration"]
        runtime_start = self.upstream_documents["runtime_start"]
        for label, document in (("preregistration", prereg), ("runtime-start", runtime_start)):
            source = _expect_mapping(document.get("source"), label=f"{label}.source")
            fastwam = _expect_mapping(source.get("fastwam"), label=f"{label}.source.fastwam")
            observer = _expect_mapping(
                source.get("instrumentation"), label=f"{label}.source.instrumentation"
            )
            if fastwam.get("commit") != self.official_identity.get("commit"):
                raise InstrumentationError(f"{label} FastWAM source differs from verified checkout")
            if observer.get("commit") != self.instrumentation_identity.get("commit"):
                raise InstrumentationError(
                    f"{label} instrumentation source differs from verified checkout"
                )
        image = _expect_mapping(prereg.get("image"), label="preregistration.image")
        image_digest = image.get("digest")
        if not isinstance(image_digest, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", image_digest
        ):
            raise InstrumentationError("preregistration image digest is absent or invalid")
        self.receipt_bindings = {
            "preregistration_canonical_sha256": self.upstream_digests[
                "preregistration_canonical_sha256"
            ],
            "runtime_start_canonical_sha256": self.upstream_digests[
                "runtime_start_canonical_sha256"
            ],
            "seed_schedule_canonical_sha256": self.upstream_digests[
                "seed_schedule_canonical_sha256"
            ],
            "resolved_config_sha256": self.upstream_digests["resolved_config_sha256"],
            "image_digest": image_digest,
            "fastwam_commit": self.official_identity["commit"],
            "instrumentation_commit": self.instrumentation_identity["commit"],
        }

    def _observe_without_rng_draw(self, label: str, callback: Callable[[], Any]) -> Any:
        before = _rng_fingerprint()
        value = callback()
        after = _rng_fingerprint()
        if before != after:
            raise InstrumentationError(f"observer changed global RNG state during {label}")
        capture = self._episode_var.get()
        if capture is not None:
            capture.observer_rng_checks += 1
        return value

    def _refresh_module_origins(self) -> dict[str, Any]:
        if self.enforce_module_origins:
            self._module_origin_receipt = audit_module_origins(self.official_root)
        return self._module_origin_receipt

    def verify_terminal_source_identities(self) -> dict[str, Any]:
        """Re-read both clean source trees and require exact start/terminal identity."""

        if self._test_terminal_source_verifier is not None:
            observed_official, observed_instrumentation = self._test_terminal_source_verifier()
            terminal_official = dict(observed_official)
            terminal_instrumentation = dict(observed_instrumentation)
        else:
            terminal_official = verify_pristine_official_root(
                Path(str(self.official_identity.get("root", ""))),
                expected_commit=str(self.official_identity.get("commit", "")),
            )
            terminal_instrumentation = verify_pristine_instrumentation_root(
                Path(str(self.instrumentation_identity.get("root", ""))),
                expected_commit=str(self.instrumentation_identity.get("commit", "")),
            )
        if terminal_official != self.official_identity:
            raise InstrumentationError("official source identity drifted after process start")
        if terminal_instrumentation != self.instrumentation_identity:
            raise InstrumentationError(
                "instrumentation source identity drifted after process start"
            )
        return {
            "status": "PASS",
            "official": terminal_official,
            "instrumentation": terminal_instrumentation,
        }

    def verify_terminal_upstream_bindings(self) -> dict[str, Any]:
        """Require all four immutable upstream anchors to equal their start readback."""

        if self._test_terminal_upstream_verifier is not None:
            observed = dict(self._test_terminal_upstream_verifier())
            if observed != self.upstream_identities:
                raise InstrumentationError("upstream artifact identity drifted after process start")
            return {"status": "PASS", "identities": observed}

        observed: dict[str, dict[str, Any]] = {}
        for name, expected in self.upstream_identities.items():
            path = Path(str(expected.get("path", "")))
            include_json = name != "resolved_config"
            readback = _read_regular_file_nofollow(path, include_bytes=include_json)
            identity: dict[str, Any] = {
                "path": str(_lexical_absolute(path)),
                "file_sha256": readback["sha256"],
                "size_bytes": readback["size_bytes"],
            }
            if include_json:
                payload = _loads_json_strict(readback["bytes"], label=name)
                identity["canonical_sha256"] = _canonical_sha256(payload)
            observed[name] = identity
        if observed != self.upstream_identities:
            raise InstrumentationError("upstream artifact identity drifted after process start")
        return {"status": "PASS", "identities": observed}

    def _bind_seed_schedule_for_task(
        self,
        *,
        suite: str,
        task_id: int,
        episode_idx: int,
        seed_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        schedule = self.upstream_documents["seed_schedule"]
        processes = schedule.get("task_processes")
        episodes = schedule.get("episodes")
        if not isinstance(processes, list) or not isinstance(episodes, list):
            raise InstrumentationError("seed schedule task-process/episode lists are absent")
        matching_processes = [
            item
            for item in processes
            if isinstance(item, Mapping)
            and item.get("task_suite") == suite
            and item.get("task_id") == task_id
        ]
        if len(matching_processes) != 1:
            raise InstrumentationError("seed schedule must have one unique matching task process")
        process = dict(matching_processes[0])
        try:
            python_hash_seed = int(os.environ["PYTHONHASHSEED"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InstrumentationError(
                "PYTHONHASHSEED must be present and integer-bound by the seed schedule"
            ) from exc
        expected_process = {
            "process_id": f"{suite}/task{task_id:02d}",
            "task_suite": suite,
            "task_id": task_id,
            "global_rank": 0,
            "global_seed": seed_metadata["task_seed"],
            "environment_seed": seed_metadata["environment_seed"],
            "environment_seed_scope": "once-before-trial-0",
            "policy_seed": seed_metadata["policy_seed"],
            "policy_seed_scope": "constant-each-replan-call",
            "python_hash_seed": python_hash_seed,
            "trial_order": list(range(self.expected_trial_count)),
            "initial_state_index_rule": "trial_idx",
        }
        if process != expected_process:
            raise InstrumentationError("live worker seeds/order differ from matching seed process")
        matching_episodes = [
            dict(item)
            for item in episodes
            if isinstance(item, Mapping)
            and item.get("task_suite") == suite
            and item.get("task_id") == task_id
        ]
        expected_episodes = [
            {
                "episode_id": f"{suite}/task{task_id:02d}/trial{trial_idx:03d}",
                "process_id": f"{suite}/task{task_id:02d}",
                "task_suite": suite,
                "task_id": task_id,
                "trial_idx": trial_idx,
                "episode_ordinal": trial_idx,
                "initial_state_index": trial_idx,
            }
            for trial_idx in range(self.expected_trial_count)
        ]
        if matching_episodes != expected_episodes:
            raise InstrumentationError(
                "seed schedule must bind the exact ordered trial/initial-state identities"
            )
        if episode_idx != expected_episodes[episode_idx]["trial_idx"]:
            raise InstrumentationError("live trial does not match seed-schedule trial order")
        if self._bound_seed_process is not None and self._bound_seed_process != process:
            raise InstrumentationError("seed process binding changed within one worker")
        self._bound_seed_process = process
        return process

    def install(self) -> "G0TraceInstrumentation":
        if self._installed:
            raise InstrumentationError("G0 instrumentation is already installed")
        required = (
            "_predict_action_chunk",
            "_denormalize_action",
            "run_single_episode",
            "set_global_seed",
            "get_libero_env",
        )
        missing = [name for name in required if not callable(getattr(self.module, name, None))]
        if missing:
            raise InstrumentationError(f"official eval module lacks required callables: {missing}")
        self._refresh_module_origins()
        self._originals = {name: getattr(self.module, name) for name in required}
        self.module._predict_action_chunk = self._traced_predict_action_chunk
        self.module._denormalize_action = self._traced_denormalize_action
        self.module.run_single_episode = self._traced_run_single_episode
        self.module.set_global_seed = self._traced_set_global_seed
        self.module.get_libero_env = self._traced_get_libero_env
        self._installed = True
        return self

    def restore(self) -> None:
        if not self._installed:
            return
        for name, original in self._originals.items():
            setattr(self.module, name, original)
        self._installed = False

    def _traced_set_global_seed(self, seed: int, *args: Any, **kwargs: Any) -> Any:
        rank = _effective_global_rank()
        call = {
            "task_seed": int(seed),
            "effective_global_rank": rank,
            "effective_process_seed": int(seed) + rank,
        }
        if self._global_seed_calls and self._global_seed_calls[-1] != call:
            raise InstrumentationError("task-process global seed changed within one worker")
        self._global_seed_calls.append(call)
        return self._originals["set_global_seed"](seed, *args, **kwargs)

    def _traced_get_libero_env(
        self,
        task: Any,
        resolution: Any,
        seed: Any,
        env_num: int = 1,
    ) -> Any:
        task_label = {
            "problem_folder": str(getattr(task, "problem_folder", "")),
            "bddl_file": str(getattr(task, "bddl_file", "")),
            "environment_seed": int(seed),
            "environment_seed_scope": "once_per_task_process_before_trial_loop",
            "env_num": int(env_num),
        }
        self._environment_seed_calls.append(task_label)
        return self._originals["get_libero_env"](task, resolution, seed, env_num=env_num)

    def _traced_denormalize_action(self, *args: Any, **kwargs: Any) -> Any:
        result = self._originals["_denormalize_action"](*args, **kwargs)
        capture = self._episode_var.get()
        if capture is not None and capture.pending_raw_proposal is None:
            capture.pending_raw_proposal = self._observe_without_rng_draw(
                "copy_raw_action_proposal",
                lambda: _numeric_matrix(
                    np.asarray(result)[0],
                    rows=EXPECTED_ACTION_HORIZON,
                    columns=EXPECTED_ACTION_DIMENSION,
                    label="raw action proposal",
                ).copy(),
            )
        return result

    def _traced_predict_action_chunk(self, *args: Any, **kwargs: Any) -> Any:
        capture = self._episode_var.get()
        if capture is None:
            raise InstrumentationError("policy prediction occurred outside an episode capture")
        if capture.warmup_count != EXPECTED_WARMUP_STEPS:
            raise InstrumentationError(
                f"first prediction occurred after {capture.warmup_count} warmup steps, expected 30"
            )
        if capture.current_record is not None:
            previous = capture.current_record
            if previous["executed_count"] != EXPECTED_REPLAN_STEPS:
                raise InstrumentationError("new proposal before 10 prior actions were executed")
            if previous["done_after_execution"]:
                raise InstrumentationError("new proposal requested after terminal env.step")
            capture.current_record = None

        signature = inspect.signature(self._originals["_predict_action_chunk"])
        bound = signature.bind(*args, **kwargs)
        obs = bound.arguments["obs"]
        cfg = bound.arguments["cfg"]
        action_horizon = int(bound.arguments["action_horizon"])
        if action_horizon != EXPECTED_ACTION_HORIZON:
            raise InstrumentationError(f"action horizon changed: {action_horizon}")

        replan_idx = len(capture.records)
        expected_env_step = EXPECTED_WARMUP_STEPS + replan_idx * EXPECTED_REPLAN_STEPS
        if capture.env_step_count != expected_env_step:
            raise InstrumentationError(
                f"replan cadence mismatch: expected env_step {expected_env_step}, "
                f"observed {capture.env_step_count}"
            )
        pre_payload = self._observe_without_rng_draw(
            "capture_pre_prediction",
            lambda: {
                "state": extract_sim_state_readonly(obs).tolist(),
                "pre_state": extract_sim_state_readonly(obs).tolist(),
                "pre_observation_sha256": observation_sha256(obs),
            },
        )
        capture.pending_raw_proposal = None
        result = self._originals["_predict_action_chunk"](*args, **kwargs)
        if capture.pending_raw_proposal is None:
            raise InstrumentationError("official prediction did not pass through _denormalize_action")
        if not isinstance(result, tuple) or not result:
            raise InstrumentationError("official prediction must return a non-empty tuple")
        env_proposal = self._observe_without_rng_draw(
            "copy_env_action_proposal",
            lambda: _numeric_matrix(
                result[0],
                rows=EXPECTED_ACTION_HORIZON,
                columns=EXPECTED_ACTION_DIMENSION,
                label="environment action proposal",
            ).copy(),
        )
        raw_proposal = capture.pending_raw_proposal
        expected_env_proposal = raw_proposal.copy()
        expected_env_proposal[..., -1] = expected_env_proposal[..., -1] * 2 - 1
        expected_env_proposal[..., -1] *= -1
        if bool(cfg.EVALUATION.get("binarize_gripper", False)):
            expected_env_proposal[..., -1] = np.sign(expected_env_proposal[..., -1])
        if not np.array_equal(expected_env_proposal, env_proposal):
            raise InstrumentationError(
                "environment action proposal does not match the official gripper transform"
            )
        policy_seed_value = cfg.get("seed")
        if type(policy_seed_value) is not int:
            raise InstrumentationError(f"policy seed must be int, observed {policy_seed_value!r}")
        record = {
            "episode_idx": capture.episode_idx,
            "replan_idx": replan_idx,
            "env_step": capture.env_step_count,
            **pre_payload,
            "policy_seed": policy_seed_value,
            "policy_seed_scope": "fresh_generator_per_replan",
            "proposed_raw_action_chunk": raw_proposal.tolist(),
            "proposed_env_action_chunk": env_proposal.tolist(),
            "executed_env_actions": [],
            "executed_count": 0,
            "done_after_execution": False,
            "executions": [],
        }
        capture.records.append(record)
        capture.current_record = record
        capture.pending_raw_proposal = None
        return result

    def _trace_root_for_cfg(self, cfg: Any) -> Path:
        output_root = _lexical_absolute(Path(str(cfg.EVALUATION.output_dir)))
        if output_root != self.artifact_root:
            raise InstrumentationError(
                "live EVALUATION.output_dir differs from preregistered artifact root: "
                f"expected {self.artifact_root}, observed {output_root}"
            )
        return self.explicit_trace_root

    def _seed_metadata(self, cfg: Any) -> dict[str, Any]:
        if len(self._global_seed_calls) != 1:
            raise InstrumentationError(
                f"expected one task-process global seed call, observed {len(self._global_seed_calls)}"
            )
        if len(self._environment_seed_calls) != 1:
            raise InstrumentationError(
                f"expected one environment seed call per task process, observed {len(self._environment_seed_calls)}"
            )
        task_seed = self._global_seed_calls[0]
        environment = self._environment_seed_calls[0]
        policy_seed = cfg.get("seed")
        if type(policy_seed) is not int:
            raise InstrumentationError("cfg.seed must be int")
        if environment["environment_seed"] != policy_seed or task_seed["task_seed"] != policy_seed:
            raise InstrumentationError(
                "task/environment/policy seeds differ from the fixed G0 seed contract"
            )
        if task_seed["effective_global_rank"] != 0:
            raise InstrumentationError(
                f"G0 worker effective global rank must be 0, got {task_seed['effective_global_rank']}"
            )
        return {
            **task_seed,
            "task_seed_scope": "once_per_task_process_before_model_and_benchmark_construction",
            "environment_seed": environment["environment_seed"],
            "environment_seed_scope": environment["environment_seed_scope"],
            "policy_seed": policy_seed,
            "policy_seed_scope": "fresh_generator_per_replan",
            "episode_rng_position": "ordered_trial_index_in_shared_task_environment_stream",
        }

    def _validate_and_finalize_capture(self, capture: _EpisodeCapture, success: bool) -> dict[str, Any]:
        if capture.warmup_count != EXPECTED_WARMUP_STEPS:
            raise InstrumentationError(
                f"episode warmup count mismatch: {capture.warmup_count}"
            )
        if len(capture.records) < MIN_TRACE_RECORDS_PER_EPISODE:
            raise InstrumentationError(
                f"episode has only {len(capture.records)} replan records; at least 7 required"
            )
        for index, record in enumerate(capture.records):
            if record["replan_idx"] != index:
                raise InstrumentationError("non-contiguous replan indices")
            if record["env_step"] != EXPECTED_WARMUP_STEPS + index * EXPECTED_REPLAN_STEPS:
                raise InstrumentationError("replan env_step cadence mismatch")
            raw = _numeric_matrix(
                record["proposed_raw_action_chunk"],
                rows=EXPECTED_ACTION_HORIZON,
                columns=EXPECTED_ACTION_DIMENSION,
                label="proposed raw action chunk",
            )
            proposed = _numeric_matrix(
                record["proposed_env_action_chunk"],
                rows=EXPECTED_ACTION_HORIZON,
                columns=EXPECTED_ACTION_DIMENSION,
                label="proposed env action chunk",
            )
            del raw
            executed = _numeric_matrix(
                record["executed_env_actions"],
                rows=None,
                columns=EXPECTED_ACTION_DIMENSION,
                label="executed env actions",
            )
            count = executed.shape[0]
            if not 1 <= count <= EXPECTED_REPLAN_STEPS:
                raise InstrumentationError(f"invalid executed action count at replan {index}: {count}")
            if not np.array_equal(executed, proposed[:count]):
                raise InstrumentationError(
                    f"executed actions differ from proposal prefix at replan {index}"
                )
            if index < len(capture.records) - 1:
                if count != EXPECTED_REPLAN_STEPS or record["done_after_execution"]:
                    raise InstrumentationError("non-terminal replan did not execute exactly 10 actions")

        last = capture.records[-1]
        if success:
            if not last["done_after_execution"] or not last["executions"][-1]["done"]:
                raise InstrumentationError("successful episode lacks a terminal env.step")
        elif any(execution["done"] for record in capture.records for execution in record["executions"]):
            raise InstrumentationError("failed episode contains a terminal-success env.step")

        metadata = {
            **capture.metadata,
            "success": success,
            "record_count": len(capture.records),
            "environment_step_count": capture.env_step_count,
            "observer_rng_unchanged_checks": capture.observer_rng_checks,
            "official_module_origin_inventory_sha256": self._module_origin_receipt[
                "inventory_sha256"
            ],
        }
        payload = {
            "schema_version": 2,
            "kind": "mf_wam_g0_structured_trace",
            "metadata": metadata,
            "records": capture.records,
        }
        return validate_structured_trace_payload(payload)

    def _traced_run_single_episode(
        self,
        env: Any,
        initial_state: Any,
        task_description: str,
        model: Any,
        processor: Any,
        cfg: Any,
        episode_idx: int,
        *,
        action_horizon: int,
        input_w: int,
        input_h: int,
        model_device: str,
    ) -> Any:
        if self._episode_var.get() is not None:
            raise InstrumentationError("nested episode instrumentation is unsupported")
        if int(action_horizon) != EXPECTED_ACTION_HORIZON:
            raise InstrumentationError(f"action horizon must be 32, got {action_horizon}")
        evaluation = cfg.EVALUATION
        fixed = {
            "num_steps_wait": (int(evaluation.get("num_steps_wait", 5)), EXPECTED_WARMUP_STEPS),
            "replan_steps": (int(evaluation.get("replan_steps", 5)), EXPECTED_REPLAN_STEPS),
            "num_trials": (int(evaluation.num_trials), 50),
        }
        mismatches = {name: observed for name, (observed, expected) in fixed.items() if observed != expected}
        if mismatches:
            raise InstrumentationError(f"fixed G0 evaluation contract mismatch: {mismatches}")
        if bool(evaluation.get("use_action_ensembler", False)):
            raise InstrumentationError("G0 trace requires use_action_ensembler=false")

        self._refresh_module_origins()
        execution_ranks = _execution_ranks()
        if (
            self._execution_rank_receipt is not None
            and self._execution_rank_receipt != execution_ranks
        ):
            raise InstrumentationError("execution ranks changed within one task process")
        self._execution_rank_receipt = execution_ranks
        seed_metadata = self._seed_metadata(cfg)
        suite = str(evaluation.task_suite_name)
        if suite not in FORMAL_SUITES:
            raise InstrumentationError(f"invalid formal LIBERO suite: {suite!r}")
        if type(evaluation.task_id) is not int or not 0 <= evaluation.task_id <= 9:
            raise InstrumentationError(f"task_id must be an int in [0, 9]: {evaluation.task_id!r}")
        if type(episode_idx) is not int or not 0 <= episode_idx < self.expected_trial_count:
            raise InstrumentationError(f"trial_idx must be an int in [0, 49]: {episode_idx!r}")
        task_id = evaluation.task_id
        if episode_idx in self._completed_trial_indices:
            raise InstrumentationError(f"duplicate trial trace in one task process: {episode_idx}")
        initial_digest = self._observe_without_rng_draw(
            "hash_initial_state", lambda: array_sha256(initial_state)
        )
        trace_root = self._trace_root_for_cfg(cfg)
        trace_root, _ = _assert_resolved_within(trace_root, trace_root, label="trace root")
        output_root = self.artifact_root
        self._last_output_root = output_root
        if self._last_task_identity is not None and self._last_task_identity != (suite, task_id):
            raise InstrumentationError("one traced worker may cover only one formal task")
        self._last_task_identity = (suite, task_id)
        seed_process = self._bind_seed_schedule_for_task(
            suite=suite,
            task_id=task_id,
            episode_idx=episode_idx,
            seed_metadata=seed_metadata,
        )
        if type(cfg.gpu_id) is not int or cfg.gpu_id < 0:
            raise InstrumentationError(f"gpu_id must be a non-negative int: {cfg.gpu_id!r}")
        if self._last_gpu_id is not None and self._last_gpu_id != cfg.gpu_id:
            raise InstrumentationError("gpu_id changed within one task process")
        self._last_gpu_id = cfg.gpu_id
        official_result_source = (
            self.artifact_root
            / suite
            / f"gpu{cfg.gpu_id}_task{task_id}_results.json"
        )
        if (
            self._official_result_source_path is not None
            and self._official_result_source_path != official_result_source
        ):
            raise InstrumentationError("official result path changed within one task process")
        self._official_result_source_path = official_result_source
        metadata = {
            "run_id": self.run_id,
            "task_suite": suite,
            "task_id": task_id,
            "trial_idx": int(episode_idx),
            "initial_state_index": int(episode_idx),
            "initial_state_sha256": initial_digest,
            "task_description": str(task_description),
            "warmup_steps": EXPECTED_WARMUP_STEPS,
            "first_replan_env_step": EXPECTED_WARMUP_STEPS,
            "replan_steps": EXPECTED_REPLAN_STEPS,
            "action_horizon": EXPECTED_ACTION_HORIZON,
            "action_dimension": EXPECTED_ACTION_DIMENSION,
            "state_dimension": EXPECTED_STATE_DIMENSION,
            "seed_contract": seed_metadata,
            "seed_schedule_process": seed_process,
            "upstream_digests": self.upstream_digests,
            "official_source": self.official_identity,
            "instrumentation_source": self.instrumentation_identity,
        }
        capture = _EpisodeCapture(
            cfg=cfg,
            episode_idx=int(episode_idx),
            task_description=str(task_description),
            initial_state_sha256=initial_digest,
            metadata=metadata,
        )
        proxy = _StepProxy(env, self, capture)
        token = self._episode_var.set(capture)
        try:
            result = self._originals["run_single_episode"](
                env=proxy,
                initial_state=initial_state,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                episode_idx=episode_idx,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            if not isinstance(result, tuple) or not result:
                raise InstrumentationError("run_single_episode returned an invalid result")
            payload = self._validate_and_finalize_capture(capture, bool(result[0]))
            path = (
                trace_root
                / suite
                / f"task{task_id:02d}"
                / f"trial{episode_idx:03d}.json"
            )
            path, _ = _assert_resolved_within(path, trace_root, label="episode trace path")
            self._observe_without_rng_draw(
                "publish_episode_trace", lambda: _atomic_write_json_no_replace(path, payload)
            )
            initial_readback = self._observe_without_rng_draw(
                "read_back_episode_trace", lambda: _read_regular_file_nofollow(path)
            )
            self._completed_trace_paths.append(path)
            self._completed_trial_indices.add(episode_idx)
            self._completed_trace_receipts[episode_idx] = {
                "trial_idx": episode_idx,
                "path": path.relative_to(self.artifact_root).as_posix(),
                "sha256": initial_readback["sha256"],
                "size_bytes": initial_readback["size_bytes"],
            }
            self._trace_success_by_trial[episode_idx] = bool(result[0])
            return result
        finally:
            self._episode_var.reset(token)

    def bind_official_task_result(self, returned_result: Any) -> dict[str, Any]:
        """Validate, canonicalize, and bind the official evaluator task result."""

        if self._last_task_identity is None or self._official_result_source_path is None:
            raise InstrumentationError("cannot bind an official result before traced episodes")
        if self._official_result_receipt is not None:
            raise InstrumentationError("official task result is already bound")
        if not isinstance(returned_result, Mapping):
            raise InstrumentationError("official evaluator return value must be a result object")
        suite, task_id = self._last_task_identity
        source_path, _ = _assert_resolved_within(
            self._official_result_source_path,
            self.artifact_root,
            label="official task result source path",
        )
        source = _read_regular_file_nofollow(source_path, include_bytes=True)
        disk_payload = _loads_json_strict(source["bytes"], label="official task result")
        if not isinstance(disk_payload, Mapping):
            raise InstrumentationError("official task result must be a JSON object")

        def _episode_list(name: str) -> list[int]:
            value = disk_payload.get(name)
            if (
                not isinstance(value, list)
                or any(type(item) is not int for item in value)
                or len(set(value)) != len(value)
            ):
                raise InstrumentationError(f"official result {name} must be unique integer IDs")
            return list(value)

        successes = _episode_list("success_episodes")
        failures = _episode_list("failure_episodes")
        expected_trials = set(range(self.expected_trial_count))
        if (
            disk_payload.get("task_suite") != suite
            or disk_payload.get("task_id") != task_id
            or disk_payload.get("total_episodes") != self.expected_trial_count
            or type(disk_payload.get("gpu_id")) is not int
            or disk_payload.get("gpu_id") != self._last_gpu_id
            or type(disk_payload.get("successes")) is not int
            or disk_payload.get("successes") != len(successes)
            or set(successes) & set(failures)
            or set(successes) | set(failures) != expected_trials
            or len(successes) + len(failures) != self.expected_trial_count
            or successes != sorted(successes)
            or failures != sorted(failures)
        ):
            raise InstrumentationError("official result task/count/success partition is invalid")
        expected_successes = {
            trial_idx for trial_idx, success in self._trace_success_by_trial.items() if success
        }
        expected_failures = expected_trials - expected_successes
        if set(successes) != expected_successes or set(failures) != expected_failures:
            raise InstrumentationError("official result episode identities differ from trace outcomes")

        returned_fields = {
            "task_suite": returned_result.get("task_suite"),
            "task_id": returned_result.get("task_id"),
            "total_episodes": returned_result.get("total_episodes"),
            "successes": returned_result.get("successes"),
            "success_episodes": returned_result.get("success_episodes"),
            "failure_episodes": returned_result.get("failure_episodes"),
        }
        disk_fields = {name: disk_payload.get(name) for name in returned_fields}
        if returned_fields != disk_fields:
            raise InstrumentationError("official returned result differs from its on-disk JSON")

        canonical_path = (
            self.artifact_root / "results" / suite / f"task{task_id:02d}.json"
        )
        canonical_path, _ = _assert_resolved_within(
            canonical_path, self.artifact_root, label="canonical task result path"
        )
        _atomic_write_bytes_no_replace(canonical_path, source["bytes"])
        canonical = _read_regular_file_nofollow(canonical_path)
        if (
            canonical["sha256"] != source["sha256"]
            or canonical["size_bytes"] != source["size_bytes"]
        ):
            raise InstrumentationError("canonical result copy differs from official result bytes")
        receipt = {
            "path": canonical_path.relative_to(self.artifact_root).as_posix(),
            "sha256": canonical["sha256"],
            "size_bytes": canonical["size_bytes"],
            "source_path": source_path.relative_to(self.artifact_root).as_posix(),
            "source_sha256": source["sha256"],
            "source_size_bytes": source["size_bytes"],
            "success_episodes": sorted(successes),
            "failure_episodes": sorted(failures),
        }
        self._official_result_receipt = receipt
        return dict(receipt)

    def finalize_process(self) -> Path:
        """Write a terminal per-task instrumentation/module-origin receipt."""

        if self._last_output_root is None or self._last_task_identity is None:
            raise InstrumentationError("no completed task is available for a process receipt")
        expected_trials = set(range(self.expected_trial_count))
        if self._completed_trial_indices != expected_trials:
            missing = sorted(expected_trials - self._completed_trial_indices)
            extras = sorted(self._completed_trial_indices - expected_trials)
            raise InstrumentationError(
                "terminal task trace scope is incomplete or duplicated: "
                f"expected={self.expected_trial_count}, observed={len(self._completed_trial_indices)}, "
                f"missing={missing[:10]}, extras={extras[:10]}"
            )
        if len(self._completed_trace_paths) != self.expected_trial_count or len(
            set(self._completed_trace_paths)
        ) != self.expected_trial_count:
            raise InstrumentationError("terminal task trace path inventory is not unique and complete")
        if set(self._completed_trace_receipts) != expected_trials:
            raise InstrumentationError("initial trace readback inventory is incomplete")
        if self._official_result_receipt is None:
            raise InstrumentationError("official task result was not safely bound")
        terminal_sources = self.verify_terminal_source_identities()
        terminal_upstream = self.verify_terminal_upstream_bindings()
        self._refresh_module_origins()
        terminal_ranks = _execution_ranks()
        if terminal_ranks != self._execution_rank_receipt:
            raise InstrumentationError("execution ranks drifted after process start")
        suite, task_id = self._last_task_identity
        trace_inventory = []
        paths_by_trial = {
            trial_idx: self.artifact_root / receipt["path"]
            for trial_idx, receipt in self._completed_trace_receipts.items()
        }
        for trial_idx in sorted(expected_trials):
            trace_path = paths_by_trial[trial_idx]
            readback = _read_regular_file_nofollow(trace_path)
            observed = {
                "trial_idx": trial_idx,
                "path": trace_path.relative_to(self.artifact_root).as_posix(),
                "sha256": readback["sha256"],
                "size_bytes": readback["size_bytes"],
            }
            if observed != self._completed_trace_receipts[trial_idx]:
                raise InstrumentationError(
                    f"episode trace drifted after publication: trial {trial_idx}"
                )
            trace_inventory.append(observed)
        trace_inventory.sort(key=lambda item: item["trial_idx"])
        result_receipt = self._official_result_receipt
        terminal_result = _read_regular_file_nofollow(
            self.artifact_root / result_receipt["path"]
        )
        if (
            terminal_result["sha256"] != result_receipt["sha256"]
            or terminal_result["size_bytes"] != result_receipt["size_bytes"]
        ):
            raise InstrumentationError("canonical official result drifted after binding")
        receipt = {
            "schema_version": 1,
            "kind": "mf_wam_g0_task_trace_receipt",
            "run_id": self.run_id,
            "process_id": f"{suite}/task{task_id:02d}",
            "task_suite": suite,
            "task_id": task_id,
            "execution_scope": "one-process-per-task",
            **terminal_ranks,
            "bindings": self.receipt_bindings,
            "seeds": {
                "global_seed": self._bound_seed_process["global_seed"],
                "environment_seed": self._bound_seed_process["environment_seed"],
                "environment_seed_scope": self._bound_seed_process[
                    "environment_seed_scope"
                ],
                "policy_seed": self._bound_seed_process["policy_seed"],
                "policy_seed_scope": self._bound_seed_process["policy_seed_scope"],
                "python_hash_seed": self._bound_seed_process["python_hash_seed"],
                "trial_order": self._bound_seed_process["trial_order"],
                "initial_state_index_rule": self._bound_seed_process[
                    "initial_state_index_rule"
                ],
            },
            "official_result": {
                "path": result_receipt["path"],
                "sha256": result_receipt["sha256"],
                "size_bytes": result_receipt["size_bytes"],
            },
            "episode_count": len(trace_inventory),
            "traces": trace_inventory,
            "tree_sha256": _sha256sum_posix_tree(trace_inventory),
        }
        # The exact receipt schema intentionally records only stable contract
        # bindings.  These terminal checks must nevertheless run successfully
        # immediately before publication.
        del terminal_sources, terminal_upstream
        path = (
            self.artifact_root
            / "trace_receipts"
            / suite
            / f"task{task_id:02d}.json"
        )
        path, _ = _assert_resolved_within(
            path,
            self.artifact_root,
            label="instrumentation process receipt path",
        )
        _atomic_write_json_no_replace(path, receipt)
        return path


def instrumentation_file_identity(*paths: Path) -> dict[str, Any]:
    files = []
    for path in paths:
        resolved = _lexical_absolute(path)
        readback = _read_regular_file_nofollow(resolved)
        files.append(
            {
                "path": str(resolved),
                "sha256": readback["sha256"],
                "size_bytes": readback["size_bytes"],
            }
        )
    return {
        "role": "external_observer_and_launcher_source",
        "files": files,
        "file_inventory_sha256": _canonical_sha256(files),
    }


__all__ = [
    "G0TraceInstrumentation",
    "InstrumentationError",
    "OFFICIAL_FASTWAM_COMMIT",
    "audit_module_origins",
    "import_pristine_official_eval",
    "instrumentation_file_identity",
    "load_upstream_artifact_bindings",
    "validate_structured_trace_payload",
    "verify_process_receipt_trace_inventory",
    "verify_pristine_instrumentation_root",
    "verify_pristine_official_root",
]
