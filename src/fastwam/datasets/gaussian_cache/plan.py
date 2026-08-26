"""Immutable coordinator plans for trajectory-granular Gaussian extraction.

The coordinator is the only process which enumerates the dataset, hashes every
HDF5 source, and discovers trajectory metadata.  Workers consume the sealed
plan and hash only the source files assigned to them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .distributed import (
    TRAJECTORY_PARTITION_ALGORITHM,
    partition_work_units,
)
from .distributed import work_plan_sha256 as distributed_work_plan_sha256
from .manifest import canonical_json_bytes, sha256_file
from .schema import normalize_source_path
from .selection import (
    NORMALIZED_SELECTION_ALGORITHM,
    load_selection_jsonl,
    normalized_selection_identity,
)
from fastwam.datasets.robofactory_layout import agent_names, camera_pair_paths

PLAN_SCHEMA_NAME = "fastwam-gaussian-trajectory-work-plan"
PLAN_SCHEMA_VERSION = 3
PLAN_FILENAME = "work-plan.json"
PLAN_COMPLETE_FILENAME = "COMPLETE"
WORKER_ASSIGNMENT_ALGORITHM = "rank-stride-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA1_RE = re.compile(r"[0-9a-f]{40}")

PRODUCER_SCHEMA_NAME = "fastwam-producer-source-snapshot"
PRODUCER_SCHEMA_VERSION = 1
COMPACT_SELECTION_SCHEMA_NAME = "fastwam-compact-selection-plan"
COMPACT_SELECTION_SCHEMA_VERSION = 1


def _stat_identity(path: Path) -> tuple[int, int]:
    value = path.stat()
    return int(value.st_size), int(value.st_mtime_ns)


def stable_file_identity(
    path: str | Path,
    *,
    relative_to: str | Path | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Hash a file while rejecting concurrent size or mtime changes."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Immutable plan input is missing: {resolved}")
    before = _stat_identity(resolved)
    digest = sha256_file(resolved)
    after = _stat_identity(resolved)
    if before != after:
        raise RuntimeError(f"Plan input changed while hashing: {resolved}")
    expected = None if expected_sha256 is None else str(expected_sha256).lower()
    if expected is not None and digest != expected:
        raise ValueError(
            f"Plan input SHA-256 mismatch for {resolved}: "
            f"expected={expected} actual={digest}"
        )
    if relative_to is None:
        stored_path = str(resolved)
    else:
        root = Path(relative_to).expanduser().resolve()
        try:
            stored_path = normalize_source_path(resolved.relative_to(root).as_posix())
        except ValueError as exc:
            raise ValueError(f"Plan input {resolved} is outside root {root}") from exc
    return {
        "path": stored_path,
        "bytes": before[0],
        "mtime_ns": before[1],
        "sha256": digest,
    }


def _git(repo: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", b"")
        detail = stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(
            f"Could not capture FastWAM producer Git identity at {repo}: {detail}"
        ) from exc


def capture_producer_identity(
    repository_root: str | Path,
    *,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Capture a reproducible FastWAM commit/source snapshot identity.

    Formal coordinators require a clean checkout.  Diagnostic runs may opt in
    to a dirty snapshot; the snapshot hash then covers the binary HEAD diff,
    staged diff, untracked path names, and untracked file bytes.
    """

    requested = Path(repository_root).expanduser().resolve()
    top = Path(
        _git(requested, "rev-parse", "--show-toplevel")
        .decode("utf-8")
        .strip()
    ).resolve()
    commit = _git(top, "rev-parse", "HEAD").decode("ascii").strip().lower()
    tree = _git(top, "rev-parse", "HEAD^{tree}").decode("ascii").strip().lower()
    status = _git(top, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    dirty = bool(status)
    if require_clean and dirty:
        raise ValueError(
            f"Formal Gaussian producer checkout must be clean: {top}"
        )

    snapshot = hashlib.sha256()

    def add(label: bytes, payload: bytes) -> None:
        snapshot.update(label + b"\0")
        snapshot.update(len(payload).to_bytes(8, "big"))
        snapshot.update(payload)

    add(b"commit", commit.encode("ascii"))
    add(b"tree", tree.encode("ascii"))
    add(b"status", status)
    if dirty:
        add(b"worktree-diff", _git(top, "diff", "--binary", "HEAD", "--"))
        add(b"index-diff", _git(top, "diff", "--binary", "--cached", "HEAD", "--"))
        untracked = _git(
            top,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ).split(b"\0")
        for raw_relative in sorted(value for value in untracked if value):
            relative = raw_relative.decode("utf-8", "surrogateescape")
            path = top / relative
            add(b"untracked-path", raw_relative)
            if path.is_file() and not path.is_symlink():
                add(b"untracked-bytes", path.read_bytes())
            elif path.is_symlink():
                add(b"untracked-symlink", os.readlink(path).encode("utf-8"))
            else:
                add(b"untracked-unsupported", b"")

    identity = {
        "schema_name": PRODUCER_SCHEMA_NAME,
        "schema_version": PRODUCER_SCHEMA_VERSION,
        "repository_root": str(top),
        "git_commit": commit,
        "git_tree": tree,
        "dirty": dirty,
        "source_snapshot_sha256": snapshot.hexdigest(),
        "status_sha256": hashlib.sha256(status).hexdigest(),
    }
    validate_producer_identity(identity, require_clean=require_clean)
    return identity


def validate_producer_identity(
    identity: Mapping[str, Any],
    *,
    require_clean: bool = False,
) -> None:
    if identity.get("schema_name") != PRODUCER_SCHEMA_NAME:
        raise ValueError("Unknown FastWAM producer identity schema")
    if int(identity.get("schema_version", -1)) != PRODUCER_SCHEMA_VERSION:
        raise ValueError("Unsupported FastWAM producer identity version")
    if not _GIT_SHA1_RE.fullmatch(str(identity.get("git_commit", ""))):
        raise ValueError("Producer identity lacks a full Git commit")
    if not _GIT_SHA1_RE.fullmatch(str(identity.get("git_tree", ""))):
        raise ValueError("Producer identity lacks a full Git tree")
    if not isinstance(identity.get("dirty"), bool):
        raise TypeError("Producer identity dirty flag must be boolean")
    if require_clean and identity["dirty"] is not False:
        raise ValueError("Formal Gaussian producer identity must be clean")
    for field in ("source_snapshot_sha256", "status_sha256"):
        if not _SHA256_RE.fullmatch(str(identity.get(field, ""))):
            raise ValueError(f"Producer identity has invalid {field}")


def _selection_plan_identity(
    selection_jsonl: str | Path,
    *,
    planned_units: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, str], tuple[Any, ...]]]:
    """Seal raw/full/scoped key identities and exact per-trajectory subsets."""

    raw = stable_file_identity(selection_jsonl)
    full_keys = load_selection_jsonl(selection_jsonl)
    full_identity = normalized_selection_identity(full_keys)
    planned_by_key = {
        (str(unit["source_path"]), str(unit["trajectory"])): unit
        for unit in planned_units
    }
    grouped: dict[tuple[str, str], list[Any]] = {
        key: [] for key in planned_by_key
    }
    for key in full_keys:
        trajectory_key = (key.source_path, key.trajectory)
        if trajectory_key in grouped:
            grouped[trajectory_key].append(key)

    normalized: dict[tuple[str, str], tuple[Any, ...]] = {}
    for trajectory_key, unit in planned_by_key.items():
        keys = tuple(sorted(set(grouped[trajectory_key])))
        if not keys:
            raise ValueError(
                f"Compact selection has no keys for planned trajectory {trajectory_key}"
            )
        expected_agents = {str(name) for name in unit["agent_names"]}
        actual_agents = {key.agent_name for key in keys}
        if actual_agents != expected_agents:
            raise ValueError(
                f"Compact selection agent mismatch for {trajectory_key}: "
                f"expected={sorted(expected_agents)} actual={sorted(actual_agents)}"
            )
        timesteps_by_agent = {
            agent: {key.timestep for key in keys if key.agent_name == agent}
            for agent in expected_agents
        }
        reference = next(iter(timesteps_by_agent.values()))
        if any(value != reference for value in timesteps_by_agent.values()):
            raise ValueError(
                f"Compact selection timesteps differ across agents for {trajectory_key}"
            )
        if not reference or min(reference) < 0 or max(reference) >= int(
            unit["observation_count"]
        ):
            raise ValueError(
                f"Compact selection timestep is outside observation range for {trajectory_key}"
            )
        normalized[trajectory_key] = keys

    planned_keys = [key for keys in normalized.values() for key in keys]
    identity = {
        "schema_name": COMPACT_SELECTION_SCHEMA_NAME,
        "schema_version": COMPACT_SELECTION_SCHEMA_VERSION,
        "mode": "index",
        "raw": raw,
        "normalized": full_identity,
        "planned_normalized": normalized_selection_identity(planned_keys),
    }
    return identity, normalized


def compact_selection_part_identity(
    plan: Mapping[str, Any],
    micro_part: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact selection provenance expected in one compact manifest."""

    selection = plan.get("compact_selection")
    if not isinstance(selection, Mapping) or selection.get("mode") != "index":
        raise ValueError("Work plan does not seal a compact selection")
    part_selection = micro_part.get("compact_selection")
    if not isinstance(part_selection, Mapping):
        raise TypeError("Micro-part lacks a sealed compact selection identity")
    return {
        "schema_name": str(selection["schema_name"]),
        "schema_version": int(selection["schema_version"]),
        "work_plan_sha256": str(plan["partition"]["work_plan_sha256"]),
        "plan_payload_sha256": work_plan_payload_sha256(plan),
        "raw": dict(selection["raw"]),
        "normalized": dict(selection["normalized"]),
        "planned_normalized": dict(selection["planned_normalized"]),
        "part": {
            "part_index": int(micro_part["part_index"]),
            **dict(part_selection),
        },
    }


def _discover_source(
    dataset_root: Path,
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import h5py

    source = stable_file_identity(path, relative_to=dataset_root)
    state_after_hash = _stat_identity(path)
    units: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        for trajectory_name in sorted(handle.keys()):
            trajectory = handle[trajectory_name]
            if "actions" not in trajectory:
                continue
            names = list(agent_names(trajectory))
            if not names:
                continue
            pairs = camera_pair_paths(trajectory, names)
            camera_paths = sorted({path for pair in pairs for path in pair})
            missing = [name for name in camera_paths if name not in trajectory]
            if missing:
                raise KeyError(
                    f"Missing trajectory cameras in {source['path']}:{trajectory_name}: "
                    f"{missing}"
                )
            counts = {int(trajectory[name].shape[0]) for name in camera_paths}
            if len(counts) != 1:
                raise ValueError(
                    "Global/agent observation lengths differ at "
                    f"{source['path']}:{trajectory_name}: {sorted(counts)}"
                )
            observation_count = counts.pop()
            if observation_count <= 0:
                raise ValueError(
                    f"Trajectory has no observations: {source['path']}:{trajectory_name}"
                )
            units.append(
                {
                    "source_path": source["path"],
                    "trajectory": str(trajectory_name),
                    "observation_count": observation_count,
                    "agent_names": names,
                    "weight": observation_count * len(names),
                }
            )
    if _stat_identity(path) != state_after_hash:
        raise RuntimeError(f"Source HDF5 changed while discovering trajectories: {path}")
    if not units:
        raise ValueError(f"Source HDF5 has no extractable trajectories: {path}")
    source["trajectory_count"] = len(units)
    source["observation_agent_weight"] = sum(int(unit["weight"]) for unit in units)
    return source, units


def _micro_part_id(part_index: int, unit: Mapping[str, Any]) -> str:
    identity = canonical_json_bytes(
        {
            "source_path": unit["source_path"],
            "trajectory": unit["trajectory"],
            "observation_count": int(unit["observation_count"]),
            "agent_names": list(unit["agent_names"]),
        }
    )
    suffix = hashlib.sha256(identity).hexdigest()[:16]
    return f"trajectory-{int(part_index):06d}-{suffix}"


def build_work_plan(
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    planned_worker_count: int = 4,
    teacher_identity: Mapping[str, Any] | None = None,
    trajectories: Sequence[tuple[str, str]] | None = None,
    compact_selection_jsonl: str | Path | None = None,
    producer_identity: Mapping[str, Any] | None = None,
    producer_repository_root: str | Path | None = None,
    require_clean_producer: bool = True,
) -> dict[str, Any]:
    """Enumerate and seal all expensive source/checkpoint discovery in memory."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RoboFactory dataset root is missing: {root}")
    worker_count = int(planned_worker_count)
    if worker_count <= 0:
        raise ValueError("planned_worker_count must be positive")
    requested = (
        {
            (normalize_source_path(str(source)), str(trajectory))
            for source, trajectory in trajectories
        }
        if trajectories
        else set()
    )
    if trajectories and len(requested) != len(trajectories):
        raise ValueError("Exact trajectory selectors must be unique")
    if requested:
        paths = sorted({root / source for source, _ in requested})
        missing_sources = [path for path in paths if not path.is_file()]
        if missing_sources:
            raise FileNotFoundError(
                f"Exact trajectory source HDF5 files are absent: {missing_sources}"
            )
    else:
        paths = sorted(root.rglob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No .h5 files found under {root}")

    sources: list[dict[str, Any]] = []
    work_units: list[dict[str, Any]] = []
    for path in paths:
        source, units = _discover_source(root, path)
        sources.append(source)
        work_units.extend(units)

    if requested:
        available = {
            (str(unit["source_path"]), str(unit["trajectory"])) for unit in work_units
        }
        missing = sorted(requested - available)
        if missing:
            raise KeyError(f"Exact trajectory selectors are absent: {missing}")
        work_units = [
            unit
            for unit in work_units
            if (str(unit["source_path"]), str(unit["trajectory"])) in requested
        ]
        selected_by_source: dict[str, list[dict[str, Any]]] = {}
        for unit in work_units:
            selected_by_source.setdefault(str(unit["source_path"]), []).append(unit)
        sources = [
            {
                **source,
                "trajectory_count": len(selected_by_source[str(source["path"])]),
                "observation_agent_weight": sum(
                    int(unit["weight"])
                    for unit in selected_by_source[str(source["path"])]
                ),
            }
            for source in sources
            if str(source["path"]) in selected_by_source
        ]
        scope = {
            "mode": "exact-trajectories",
            "trajectory_selectors": [
                {"source_path": source, "trajectory": trajectory}
                for source, trajectory in sorted(requested)
            ],
        }
    else:
        scope = {"mode": "all", "trajectory_selectors": []}

    partition_count = len(work_units)
    # With one partition per work unit, each immutable part contains exactly
    # one trajectory.  The ordering intentionally matches distributed.py so
    # its zero-copy merge validation remains authoritative.
    partitions = partition_work_units(
        work_units,
        partition_count,
        unit="trajectory",
    )
    if any(len(partition) != 1 for partition in partitions):
        raise AssertionError("One-trajectory micro partition construction failed")
    plan_digest = distributed_work_plan_sha256(
        work_units,
        partition_count,
        unit="trajectory",
    )
    micro_parts = []
    for part_index, partition in enumerate(partitions):
        unit = dict(partition[0])
        micro_parts.append(
            {
                "part_index": part_index,
                "micro_part_id": _micro_part_id(part_index, unit),
                **unit,
            }
        )

    if compact_selection_jsonl is None:
        compact_selection: dict[str, Any] = {"mode": "none"}
    else:
        compact_selection, compact_keys = _selection_plan_identity(
            compact_selection_jsonl,
            planned_units=work_units,
        )
        for micro_part in micro_parts:
            trajectory_key = (
                str(micro_part["source_path"]),
                str(micro_part["trajectory"]),
            )
            micro_part["compact_selection"] = normalized_selection_identity(
                compact_keys[trajectory_key]
            )

    if producer_identity is None:
        repository_root = (
            Path(__file__).resolve().parents[4]
            if producer_repository_root is None
            else producer_repository_root
        )
        producer = capture_producer_identity(
            repository_root,
            require_clean=require_clean_producer,
        )
    else:
        if producer_repository_root is not None:
            raise ValueError(
                "producer_identity and producer_repository_root are mutually exclusive"
            )
        producer = dict(producer_identity)
        validate_producer_identity(
            producer,
            require_clean=require_clean_producer,
        )

    checkpoint = stable_file_identity(
        checkpoint_path,
        expected_sha256=expected_checkpoint_sha256,
    )
    checkpoint["filename"] = Path(checkpoint["path"]).name
    plan: dict[str, Any] = {
        "schema_name": PLAN_SCHEMA_NAME,
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "coordinator_root": str(root),
            "source_count": len(sources),
            "trajectory_count": len(work_units),
            "sources": sorted(sources, key=lambda item: str(item["path"])),
        },
        "checkpoint": checkpoint,
        "teacher": dict(teacher_identity or {}),
        "producer": producer,
        "compact_selection": compact_selection,
        "scope": scope,
        "partition": {
            "algorithm": TRAJECTORY_PARTITION_ALGORITHM,
            "unit": "trajectory",
            "partition_count": partition_count,
            "expected_unit_count": partition_count,
            "expected_unit_weight": sum(int(unit["weight"]) for unit in work_units),
            "work_plan_sha256": plan_digest,
        },
        "worker_assignment": {
            "algorithm": WORKER_ASSIGNMENT_ALGORITHM,
            "planned_worker_count": worker_count,
        },
        "micro_parts": micro_parts,
    }
    validate_work_plan(plan)
    return plan


def work_plan_payload_sha256(plan: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(plan)).hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def write_work_plan(plan_root: str | Path, plan: Mapping[str, Any]) -> str:
    """Write an immutable plan and its checksum marker, with COMPLETE last."""

    validate_work_plan(plan)
    root = Path(plan_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    payload = canonical_json_bytes(plan)
    digest = hashlib.sha256(payload).hexdigest()
    _write_exclusive(root / PLAN_FILENAME, payload)
    complete = canonical_json_bytes(
        {
            "complete": True,
            "schema_name": PLAN_SCHEMA_NAME,
            "schema_version": PLAN_SCHEMA_VERSION,
            "plan_sha256": digest,
            "plan_bytes": len(payload),
            "source_count": int(plan["dataset"]["source_count"]),
            "micro_part_count": len(plan["micro_parts"]),
        }
    )
    _write_exclusive(root / PLAN_COMPLETE_FILENAME, complete)
    root.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )
    return digest


def create_work_plan(
    plan_root: str | Path,
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    planned_worker_count: int = 4,
    teacher_identity: Mapping[str, Any] | None = None,
    trajectories: Sequence[tuple[str, str]] | None = None,
    compact_selection_jsonl: str | Path | None = None,
    producer_identity: Mapping[str, Any] | None = None,
    producer_repository_root: str | Path | None = None,
    require_clean_producer: bool = True,
) -> dict[str, Any]:
    plan = build_work_plan(
        dataset_root,
        checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        planned_worker_count=planned_worker_count,
        teacher_identity=teacher_identity,
        trajectories=trajectories,
        compact_selection_jsonl=compact_selection_jsonl,
        producer_identity=producer_identity,
        producer_repository_root=producer_repository_root,
        require_clean_producer=require_clean_producer,
    )
    write_work_plan(plan_root, plan)
    return plan


def _normalized_work_units(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source_path": normalize_source_path(str(item["source_path"])),
            "trajectory": str(item["trajectory"]),
            "observation_count": int(item["observation_count"]),
            "agent_names": [str(name) for name in item["agent_names"]],
            "weight": int(item["weight"]),
        }
        for item in plan["micro_parts"]
    ]


def _validate_normalized_selection_identity(
    identity: Mapping[str, Any],
    *,
    context: str,
) -> None:
    if identity.get("algorithm") != NORMALIZED_SELECTION_ALGORITHM:
        raise ValueError(f"{context} uses an unsupported normalization algorithm")
    if not _SHA256_RE.fullmatch(str(identity.get("index_sha256", ""))):
        raise ValueError(f"{context} lacks a canonical key-set SHA-256")
    if int(identity.get("selected_key_count", 0)) <= 0:
        raise ValueError(f"{context} selected_key_count must be positive")


def validate_work_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_name") != PLAN_SCHEMA_NAME:
        raise ValueError("Unknown Gaussian work-plan schema")
    if int(plan.get("schema_version", -1)) != PLAN_SCHEMA_VERSION:
        raise ValueError("Unsupported Gaussian work-plan schema version")
    dataset = plan.get("dataset")
    if not isinstance(dataset, Mapping):
        raise TypeError("Work plan lacks dataset metadata")
    sources = dataset.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Work plan must contain source identities")
    source_paths: list[str] = []
    for source in sources:
        relative = normalize_source_path(str(source["path"]))
        source_paths.append(relative)
        if int(source["bytes"]) <= 0 or int(source["mtime_ns"]) <= 0:
            raise ValueError(f"Invalid source stat identity: {source}")
        if not _SHA256_RE.fullmatch(str(source["sha256"])):
            raise ValueError(f"Invalid source SHA-256 identity: {source}")
        if int(source["trajectory_count"]) <= 0:
            raise ValueError(f"Source lacks trajectory accounting: {source}")
    if source_paths != sorted(source_paths) or len(set(source_paths)) != len(source_paths):
        raise ValueError("Work-plan source paths must be sorted and unique")
    if int(dataset.get("source_count", -1)) != len(sources):
        raise ValueError("Work-plan source_count mismatch")

    scope = plan.get("scope")
    if not isinstance(scope, Mapping) or scope.get("mode") not in {
        "all",
        "exact-trajectories",
    }:
        raise ValueError("Work plan lacks an explicit dataset scope")
    selectors = scope.get("trajectory_selectors")
    if not isinstance(selectors, list):
        raise TypeError("Work-plan trajectory_selectors must be a list")

    checkpoint = plan.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Work plan lacks checkpoint identity")
    if int(checkpoint.get("bytes", -1)) <= 0 or int(checkpoint.get("mtime_ns", -1)) <= 0:
        raise ValueError("Invalid checkpoint stat identity")
    if not _SHA256_RE.fullmatch(str(checkpoint.get("sha256", ""))):
        raise ValueError("Invalid checkpoint SHA-256 identity")

    teacher = plan.get("teacher")
    if not isinstance(teacher, Mapping):
        raise TypeError("Work plan lacks teacher provenance")
    if not _GIT_SHA1_RE.fullmatch(str(teacher.get("repository_commit", ""))):
        raise ValueError("Teacher provenance lacks a pinned repository commit")
    if not str(teacher.get("repository_url", "")).strip():
        raise ValueError("Teacher provenance lacks a repository URL")
    config_relative_path = normalize_source_path(
        str(teacher.get("config_relative_path", ""))
    )
    if config_relative_path != teacher.get("config_relative_path"):
        raise ValueError("Teacher config relative path is not normalized")
    if not _SHA256_RE.fullmatch(str(teacher.get("config_sha256", ""))):
        raise ValueError("Teacher provenance lacks a pinned config SHA-256")
    training_provenance = teacher.get("training_data_provenance")
    if not isinstance(training_provenance, Mapping):
        raise TypeError("teacher.training_data_provenance must be a sealed object")
    if training_provenance is not None:
        if int(training_provenance.get("record_bytes", 0)) <= 0:
            raise ValueError("Teacher training provenance record has no bytes")
        if not _SHA256_RE.fullmatch(
            str(training_provenance.get("record_sha256", ""))
        ):
            raise ValueError("Teacher training provenance record SHA-256 is invalid")
        record = training_provenance.get("record")
        if not isinstance(record, Mapping):
            raise TypeError("Teacher training provenance lacks its sealed record")
        if record.get("schema_name") != "fastwam_external_teacher_training_provenance" or int(
            record.get("schema_version", -1)
        ) != 1:
            raise ValueError("Unsupported teacher training provenance schema")
        teacher_checkpoint = record.get("checkpoint")
        if not isinstance(teacher_checkpoint, Mapping) or str(
            teacher_checkpoint.get("sha256", "")
        ).lower() != str(checkpoint["sha256"]).lower():
            raise ValueError(
                "Teacher training provenance checkpoint does not match the work plan"
            )
        declared_datasets = record.get("declared_training_datasets")
        if not isinstance(declared_datasets, list) or not declared_datasets:
            raise ValueError("Teacher training provenance declares no training datasets")
        declaration_source = record.get("declaration_source")
        if not isinstance(declaration_source, Mapping) or not _GIT_SHA1_RE.fullmatch(
            str(declaration_source.get("repository_commit", ""))
        ):
            raise ValueError(
                "Teacher training provenance declaration source lacks a pinned Git commit"
            )
        overlap = record.get("overlap_assessment")
        if not isinstance(overlap, Mapping):
            raise TypeError("Teacher training provenance lacks overlap_assessment")
        if overlap.get("declared_dataset_identity_overlap") is not False:
            raise ValueError(
                "Formal teacher provenance must explicitly declare whether target "
                "dataset identity overlaps its training datasets"
            )
        if overlap.get("file_level_overlap_audit") not in {
            "verified_no_overlap",
            "unavailable_teacher_training_file_inventory",
        }:
            raise ValueError("Unknown teacher file-level overlap audit status")

    producer = plan.get("producer")
    if not isinstance(producer, Mapping):
        raise TypeError("Work plan lacks FastWAM producer provenance")
    validate_producer_identity(producer)

    compact_selection = plan.get("compact_selection")
    if not isinstance(compact_selection, Mapping) or compact_selection.get("mode") not in {
        "none",
        "index",
    }:
        raise ValueError("Work plan must declare compact_selection mode none or index")
    if compact_selection.get("mode") == "index":
        if compact_selection.get("schema_name") != COMPACT_SELECTION_SCHEMA_NAME:
            raise ValueError("Unknown compact-selection work-plan schema")
        if int(compact_selection.get("schema_version", -1)) != (
            COMPACT_SELECTION_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported compact-selection work-plan version")
        raw_selection = compact_selection.get("raw")
        if not isinstance(raw_selection, Mapping):
            raise TypeError("Compact-selection plan lacks raw input identity")
        if not str(raw_selection.get("path", "")):
            raise ValueError("Compact-selection raw path must be non-empty")
        if int(raw_selection.get("bytes", 0)) <= 0:
            raise ValueError("Compact-selection raw byte count must be positive")
        if not _SHA256_RE.fullmatch(str(raw_selection.get("sha256", ""))):
            raise ValueError("Compact-selection raw SHA-256 is invalid")
        for field in ("normalized", "planned_normalized"):
            identity = compact_selection.get(field)
            if not isinstance(identity, Mapping):
                raise TypeError(f"Compact-selection plan lacks {field}")
            _validate_normalized_selection_identity(
                identity,
                context=f"compact_selection.{field}",
            )
        if int(compact_selection["planned_normalized"]["selected_key_count"]) > int(
            compact_selection["normalized"]["selected_key_count"]
        ):
            raise ValueError("Planned compact selection cannot exceed the full key set")

    micro_parts = plan.get("micro_parts")
    if not isinstance(micro_parts, list) or not micro_parts:
        raise ValueError("Work plan must contain trajectory micro-parts")
    indices = [int(item["part_index"]) for item in micro_parts]
    if indices != list(range(len(micro_parts))):
        raise ValueError("Micro-part indices must be ordered and contiguous")
    ids = [str(item["micro_part_id"]) for item in micro_parts]
    if len(set(ids)) != len(ids):
        raise ValueError("Micro-part IDs must be unique")
    keys = [
        (normalize_source_path(str(item["source_path"])), str(item["trajectory"]))
        for item in micro_parts
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Every source trajectory must have exactly one micro-part")
    expected_selectors = [
        {"source_path": source, "trajectory": trajectory}
        for source, trajectory in sorted(keys)
    ]
    if scope["mode"] == "all" and selectors:
        raise ValueError("scope=all must not contain exact trajectory selectors")
    if scope["mode"] == "exact-trajectories" and selectors != expected_selectors:
        raise ValueError("Exact trajectory scope differs from micro-part identities")
    units = _normalized_work_units(plan)
    micro_selection_total = 0
    for item, unit in zip(micro_parts, units):
        if not unit["agent_names"] or len(set(unit["agent_names"])) != len(unit["agent_names"]):
            raise ValueError(f"Invalid micro-part agent_names: {item}")
        if unit["observation_count"] <= 0:
            raise ValueError(f"Invalid micro-part observation_count: {item}")
        if unit["weight"] != unit["observation_count"] * len(unit["agent_names"]):
            raise ValueError(f"Invalid micro-part weight: {item}")
        if unit["source_path"] not in source_paths:
            raise ValueError(f"Micro-part references unknown source: {item}")
        if str(item["micro_part_id"]) != _micro_part_id(int(item["part_index"]), unit):
            raise ValueError(f"Micro-part ID does not match its identity: {item}")
        part_selection = item.get("compact_selection")
        if compact_selection["mode"] == "none":
            if part_selection is not None:
                raise ValueError("Selection-free plan must not seal per-part compact keys")
        else:
            if not isinstance(part_selection, Mapping):
                raise TypeError(f"Micro-part lacks compact selection identity: {item}")
            _validate_normalized_selection_identity(
                part_selection,
                context=f"micro_parts[{item['part_index']}].compact_selection",
            )
            micro_selection_total += int(part_selection["selected_key_count"])
    if compact_selection["mode"] == "index" and micro_selection_total != int(
        compact_selection["planned_normalized"]["selected_key_count"]
    ):
        raise ValueError("Per-part compact selection counts differ from planned key count")

    partition = plan.get("partition")
    if not isinstance(partition, Mapping):
        raise TypeError("Work plan lacks partition metadata")
    if partition.get("algorithm") != TRAJECTORY_PARTITION_ALGORITHM:
        raise ValueError("Work plan has an unsupported partition algorithm")
    if partition.get("unit") != "trajectory":
        raise ValueError("Work plan partition unit must be trajectory")
    if int(partition.get("partition_count", -1)) != len(units):
        raise ValueError("Work-plan partition_count mismatch")
    if int(partition.get("expected_unit_count", -1)) != len(units):
        raise ValueError("Work-plan expected_unit_count mismatch")
    if int(partition.get("expected_unit_weight", -1)) != sum(
        int(unit["weight"]) for unit in units
    ):
        raise ValueError("Work-plan expected_unit_weight mismatch")
    actual_digest = distributed_work_plan_sha256(
        units,
        len(units),
        unit="trajectory",
    )
    if partition.get("work_plan_sha256") != actual_digest:
        raise ValueError("Work-plan partition SHA-256 mismatch")
    expected_partitions = partition_work_units(units, len(units), unit="trajectory")
    for item, expected in zip(micro_parts, expected_partitions):
        if len(expected) != 1 or _normalized_unit(item) != expected[0]:
            raise ValueError("Micro-part ordering differs from distributed LPT ordering")

    assignment = plan.get("worker_assignment")
    if not isinstance(assignment, Mapping):
        raise TypeError("Work plan lacks worker assignment metadata")
    if assignment.get("algorithm") != WORKER_ASSIGNMENT_ALGORITHM:
        raise ValueError("Unsupported worker assignment algorithm")
    if int(assignment.get("planned_worker_count", 0)) <= 0:
        raise ValueError("planned_worker_count must be positive")
    if int(dataset.get("trajectory_count", -1)) != len(micro_parts):
        raise ValueError("Work-plan trajectory_count mismatch")
    if sum(int(source["trajectory_count"]) for source in sources) != len(micro_parts):
        raise ValueError("Source trajectory accounting differs from micro-parts")


def _normalized_unit(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_path": normalize_source_path(str(item["source_path"])),
        "trajectory": str(item["trajectory"]),
        "observation_count": int(item["observation_count"]),
        "agent_names": [str(name) for name in item["agent_names"]],
        "weight": int(item["weight"]),
    }


def load_work_plan(plan_root: str | Path) -> dict[str, Any]:
    root = Path(plan_root).expanduser().resolve()
    plan_path = root / PLAN_FILENAME
    complete_path = root / PLAN_COMPLETE_FILENAME
    if not complete_path.is_file():
        raise FileNotFoundError(f"Gaussian work plan is incomplete: missing {complete_path}")
    payload = plan_path.read_bytes()
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("complete") is not True:
        raise ValueError("Invalid Gaussian work-plan COMPLETE marker")
    if complete.get("schema_name") != PLAN_SCHEMA_NAME:
        raise ValueError("Work-plan COMPLETE marker schema mismatch")
    if int(complete.get("schema_version", -1)) != PLAN_SCHEMA_VERSION:
        raise ValueError("Work-plan COMPLETE marker version mismatch")
    digest = hashlib.sha256(payload).hexdigest()
    if complete.get("plan_sha256") != digest:
        raise ValueError("Work-plan COMPLETE checksum mismatch")
    if int(complete.get("plan_bytes", -1)) != len(payload):
        raise ValueError("Work-plan COMPLETE byte count mismatch")
    plan = json.loads(payload)
    validate_work_plan(plan)
    if int(complete.get("source_count", -1)) != len(plan["dataset"]["sources"]):
        raise ValueError("Work-plan COMPLETE source count mismatch")
    if int(complete.get("micro_part_count", -1)) != len(plan["micro_parts"]):
        raise ValueError("Work-plan COMPLETE micro-part count mismatch")
    return plan


def micro_part_partition_metadata(
    plan: Mapping[str, Any],
    micro_part: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact partition block expected by GaussianCacheBuilder."""

    index = int(micro_part["part_index"])
    if not 0 <= index < len(plan["micro_parts"]):
        raise ValueError("Micro-part index is outside this work plan")
    expected = plan["micro_parts"][index]
    if dict(micro_part) != dict(expected):
        raise ValueError("Micro-part does not belong to this sealed work plan")
    partition = plan["partition"]
    return {
        "algorithm": str(partition["algorithm"]),
        "unit": "trajectory",
        "partition_index": index,
        "partition_count": int(partition["partition_count"]),
        "expected_unit_count": int(partition["expected_unit_count"]),
        "expected_unit_weight": int(partition["expected_unit_weight"]),
        "work_plan_sha256": str(partition["work_plan_sha256"]),
        "assigned_unit_count": 1,
        "assigned_unit_weight": int(micro_part["weight"]),
        "assigned_units": [_normalized_unit(micro_part)],
    }


def source_identity_by_path(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(source["path"]): dict(source)
        for source in plan["dataset"]["sources"]
    }
