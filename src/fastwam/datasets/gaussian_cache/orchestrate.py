"""Standalone coordinator/worker orchestration for Gaussian cache materialization.

This module deliberately separates orchestration from extraction.  Integrators
provide a teacher factory and one trajectory processor; the worker guarantees
sealed-plan consumption, deterministic rank assignment, one CUDA binding and
teacher construction per process/GPU, fail-closed resume, local staging gates,
and durable failure records.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .distributed import MERGE_BUILDING_FILENAME, merge_part_manifests
from .manifest import canonical_json_bytes, sha256_file
from .plan import (
    compact_selection_part_identity,
    create_work_plan,
    load_work_plan,
    micro_part_partition_metadata,
    source_identity_by_path,
    work_plan_payload_sha256,
)
from .provider import GaussianCache
from .schema import FrameKey, normalize_source_path
from .selection import load_selection_jsonl, normalized_selection_identity
from .validate import validate_cache

MIN_STAGING_FREE_BYTES = 25 * (1 << 30)
NODE_LOCAL_FILESYSTEMS = frozenset(
    {
        "btrfs",
        "ext2",
        "ext3",
        "ext4",
        "f2fs",
        "overlay",
        "tmpfs",
        "xfs",
        "zfs",
    }
)


@dataclass(frozen=True)
class WorkerIdentity:
    platform: str
    global_rank: int
    worker_count: int
    local_rank: int
    cuda_index: int
    node_rank: int
    node_count: int


@dataclass(frozen=True)
class BootstrapContext:
    plan_root: Path
    plan: Mapping[str, Any]
    plan_sha256: str
    dataset_root: Path
    checkpoint_path: Path
    worker: WorkerIdentity
    device: Any
    staging_dir: Path
    verified_sources: Mapping[str, tuple[Path, tuple[int, int], Mapping[str, Any]]]
    compact_selection: Mapping[tuple[str, str], tuple[FrameKey, ...]]


@dataclass(frozen=True)
class OfficialBuildSettings:
    teacher_repo: Path
    teacher_config: str
    batch_size: int = 8
    target_shard_bytes: int = 2 << 30
    compact_target_shard_bytes: int = 2 << 30
    verify_uploaded_checksum: bool = True


@dataclass(frozen=True)
class MicroPartContext:
    bootstrap: BootstrapContext
    micro_part: Mapping[str, Any]
    source_identity: Mapping[str, Any]
    canonical_root: Path
    compact_root: Path | None
    need_canonical: bool
    need_compact: bool

    @property
    def partition_metadata(self) -> dict[str, Any]:
        return micro_part_partition_metadata(self.bootstrap.plan, self.micro_part)

    @property
    def work_identity(self) -> dict[str, Any]:
        identity = {
            "micro_part_id": str(self.micro_part["micro_part_id"]),
            "part_index": int(self.micro_part["part_index"]),
            "source_path": str(self.micro_part["source_path"]),
            "source_sha256": str(self.source_identity["sha256"]),
            "trajectory": str(self.micro_part["trajectory"]),
            "observation_count": int(self.micro_part["observation_count"]),
            "agent_names": list(self.micro_part["agent_names"]),
            "checkpoint_sha256": str(self.bootstrap.plan["checkpoint"]["sha256"]),
            "producer_source_snapshot_sha256": str(
                self.bootstrap.plan["producer"]["source_snapshot_sha256"]
            ),
        }
        if self.compact_root is not None:
            identity["compact_selection"] = compact_selection_part_identity(
                self.bootstrap.plan,
                self.micro_part,
            )
        return identity


TeacherFactory = Callable[[BootstrapContext], Any]
MicroPartProcessor = Callable[[MicroPartContext, Any], Any]
PartVerifier = Callable[..., Mapping[str, Any]]
_PLAN_AWARE_ATTRIBUTE = "__fastwam_gaussian_plan_aware_v1__"


def plan_aware_processor(function: MicroPartProcessor) -> MicroPartProcessor:
    """Mark a callback as consuming only ``MicroPartContext`` plan metadata.

    A marked callback must not glob, hash, or rediscover the full dataset.  It
    may open only ``context.source_identity`` at the exact source/trajectory
    given by ``context.micro_part`` and must pass ``context.partition_metadata``
    to any cache builder.  This explicit opt-in prevents accidentally wrapping
    ``extract_canonical_cache``, whose public API intentionally performs a full
    dataset discovery pass.
    """

    setattr(function, _PLAN_AWARE_ATTRIBUTE, True)
    return function


def _require_plan_aware(function: MicroPartProcessor, name: str) -> None:
    if getattr(function, _PLAN_AWARE_ATTRIBUTE, False) is not True:
        raise TypeError(
            f"{name} must be decorated with plan_aware_processor; workers must not "
            "call the dataset-rescanning extract_canonical_cache API"
        )


def _required_nonnegative(value: Any, name: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if integer < 0:
        raise ValueError(f"{name} must be non-negative, got {integer}")
    return integer


def _required_positive(value: Any, name: str) -> int:
    integer = _required_nonnegative(value, name)
    if integer == 0:
        raise ValueError(f"{name} must be positive")
    return integer


def resolve_worker_identity(
    platform: str = "auto",
    *,
    env: Mapping[str, str] | None = None,
    rank: int | None = None,
    worker_count: int | None = None,
    local_rank: int | None = None,
) -> WorkerIdentity:
    """Resolve DSW, PAI DLC node env, or torchrun process env deterministically.

    PAI DLC exposes ``WORLD_SIZE/RANK`` as node count/rank and
    ``NPROC_PER_NODE`` as local GPU workers.  Therefore a DLC worker maps to
    ``global_rank = node_rank * NPROC_PER_NODE + local_rank``.
    """

    values = os.environ if env is None else env
    mode = str(platform).lower()
    if mode == "auto":
        if "LOCAL_RANK" in values and "LOCAL_WORLD_SIZE" in values:
            mode = "torchrun"
        elif "NPROC_PER_NODE" in values and "WORLD_SIZE" in values and "RANK" in values:
            mode = "dlc"
        else:
            mode = "dsw"

    if mode == "torchrun":
        global_rank = _required_nonnegative(
            values.get("RANK") if rank is None else rank,
            "RANK",
        )
        total = _required_positive(
            values.get("WORLD_SIZE") if worker_count is None else worker_count,
            "WORLD_SIZE",
        )
        local = _required_nonnegative(
            values.get("LOCAL_RANK") if local_rank is None else local_rank,
            "LOCAL_RANK",
        )
        local_count = _required_positive(values.get("LOCAL_WORLD_SIZE", 1), "LOCAL_WORLD_SIZE")
        if global_rank >= total or local >= local_count:
            raise ValueError("torchrun rank is outside its declared world")
        return WorkerIdentity(
            platform=mode,
            global_rank=global_rank,
            worker_count=total,
            local_rank=local,
            cuda_index=local,
            node_rank=global_rank // local_count,
            node_count=(total + local_count - 1) // local_count,
        )

    if mode == "dlc":
        node_count = _required_positive(values.get("WORLD_SIZE"), "DLC WORLD_SIZE")
        node_rank = _required_nonnegative(values.get("RANK"), "DLC RANK")
        local_count = _required_positive(values.get("NPROC_PER_NODE"), "DLC NPROC_PER_NODE")
        local = _required_nonnegative(
            values.get("LOCAL_RANK") if local_rank is None else local_rank,
            "DLC local_rank",
        )
        if node_rank >= node_count or local >= local_count:
            raise ValueError("DLC node/local rank is outside its declared world")
        total = node_count * local_count
        global_rank = node_rank * local_count + local
        if rank is not None and int(rank) != global_rank:
            raise ValueError(
                f"Explicit global rank {rank} conflicts with DLC-derived rank {global_rank}"
            )
        if worker_count is not None and int(worker_count) != total:
            raise ValueError(
                f"Explicit worker_count {worker_count} conflicts with DLC-derived count {total}"
            )
        return WorkerIdentity(
            platform=mode,
            global_rank=global_rank,
            worker_count=total,
            local_rank=local,
            cuda_index=local,
            node_rank=node_rank,
            node_count=node_count,
        )

    if mode != "dsw":
        raise ValueError("platform must be auto, dsw, dlc, or torchrun")
    global_rank = _required_nonnegative(
        values.get("LOCAL_RANK", 0) if rank is None else rank,
        "DSW rank",
    )
    total = _required_positive(4 if worker_count is None else worker_count, "DSW worker_count")
    if global_rank >= total:
        raise ValueError("DSW rank is outside worker_count")
    local = global_rank if local_rank is None else _required_nonnegative(local_rank, "local_rank")
    if local >= total:
        raise ValueError("DSW local_rank is outside worker_count")
    return WorkerIdentity(
        platform=mode,
        global_rank=global_rank,
        worker_count=total,
        local_rank=local,
        cuda_index=local,
        node_rank=0,
        node_count=1,
    )


def bind_cuda_device(worker: WorkerIdentity, *, torch_module=None):
    """Bind one worker process to exactly one visible CUDA device."""

    if torch_module is None:
        import torch as torch_module

    if not bool(torch_module.cuda.is_available()):
        raise RuntimeError("CUDA is unavailable for Gaussian extraction worker")
    count = int(torch_module.cuda.device_count())
    if not 0 <= worker.cuda_index < count:
        raise RuntimeError(
            f"CUDA index {worker.cuda_index} is outside visible device count {count}"
        )
    torch_module.cuda.set_device(worker.cuda_index)
    return torch_module.device(f"cuda:{worker.cuda_index}")


def _unescape_mount_path(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def filesystem_type(path: str | Path, *, mountinfo_path: str | Path = "/proc/self/mountinfo") -> str:
    """Return the filesystem type for the longest matching Linux mount point."""

    target = Path(path).expanduser().resolve()
    existing = target
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    matches: list[tuple[int, str]] = []
    for line in Path(mountinfo_path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if len(fields) <= separator + 1:
            continue
        mount_point = Path(_unescape_mount_path(fields[4])).resolve()
        try:
            existing.relative_to(mount_point)
        except ValueError:
            continue
        matches.append((len(mount_point.parts), fields[separator + 1]))
    if not matches:
        raise RuntimeError(f"Could not resolve filesystem type for staging path {target}")
    return max(matches, key=lambda item: item[0])[1]


def ensure_node_local_staging(
    staging_dir: str | Path,
    *,
    min_free_bytes: int = MIN_STAGING_FREE_BYTES,
    allowed_filesystems: Sequence[str] = tuple(sorted(NODE_LOCAL_FILESYSTEMS)),
    mountinfo_path: str | Path = "/proc/self/mountinfo",
) -> Path:
    """Fail closed unless staging is local and has at least 25 GiB by default."""

    path = Path(staging_dir).expanduser().resolve()
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    fs_type = filesystem_type(existing, mountinfo_path=mountinfo_path)
    if fs_type not in set(allowed_filesystems):
        raise RuntimeError(
            f"Staging must be node-local; {path} resolves to filesystem type {fs_type!r}"
        )
    minimum = int(min_free_bytes)
    if minimum < 0:
        raise ValueError("min_free_bytes must be non-negative")
    free = int(shutil.disk_usage(existing).free)
    if free < minimum:
        raise OSError(
            f"Node-local staging has insufficient free space: path={path} "
            f"free_bytes={free} required_bytes={minimum}"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_trajectory_selector(value: str) -> tuple[str, str]:
    source, separator, trajectory = str(value).partition("::")
    if not separator or not source or not trajectory:
        raise ValueError(
            "Trajectory selector must be exact 'relative/source.h5::trajectory_name'"
        )
    return normalize_source_path(source), trajectory


def load_compact_selection_for_plan(
    plan: Mapping[str, Any],
    selection_jsonl: str | Path,
) -> dict[tuple[str, str], tuple[FrameKey, ...]]:
    """Verify the sealed raw/full/scoped identity and return exact part keys."""

    sealed = plan.get("compact_selection")
    if not isinstance(sealed, Mapping) or sealed.get("mode") != "index":
        raise ValueError("Sealed work plan does not authorize compact extraction")
    path = Path(selection_jsonl).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Compact selection input is missing: {path}")
    raw = sealed.get("raw")
    if not isinstance(raw, Mapping):
        raise TypeError("Sealed compact selection lacks raw identity")
    before_state = _file_state(path)
    actual_bytes = before_state[0]
    if actual_bytes != int(raw["bytes"]):
        raise ValueError(
            "Compact selection raw byte count differs from sealed plan: "
            f"expected={raw['bytes']} actual={actual_bytes}"
        )
    actual_raw_sha256 = sha256_file(path)
    if actual_raw_sha256 != str(raw["sha256"]):
        raise ValueError(
            "Compact selection raw SHA-256 differs from sealed plan: "
            f"expected={raw['sha256']} actual={actual_raw_sha256}"
        )

    full_keys = load_selection_jsonl(path)
    if _file_state(path) != before_state:
        raise RuntimeError(f"Compact selection changed during worker verification: {path}")
    actual_full = normalized_selection_identity(full_keys)
    if actual_full != dict(sealed["normalized"]):
        raise ValueError(
            "Compact selection normalized full key identity differs from sealed plan"
        )

    planned = {
        (str(item["source_path"]), str(item["trajectory"])): item
        for item in plan["micro_parts"]
    }
    grouped: dict[tuple[str, str], list[FrameKey]] = {
        key: [] for key in planned
    }
    for key in full_keys:
        trajectory_key = (key.source_path, key.trajectory)
        if trajectory_key in grouped:
            grouped[trajectory_key].append(key)
    normalized: dict[tuple[str, str], tuple[FrameKey, ...]] = {}
    for trajectory_key, micro_part in planned.items():
        keys = sorted(set(grouped[trajectory_key]))
        if not keys:
            raise ValueError(
                f"Compact selection has no keys for planned trajectory {trajectory_key}"
            )
        expected_agents = {str(name) for name in micro_part["agent_names"]}
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
        reference_timesteps = next(iter(timesteps_by_agent.values()))
        if any(value != reference_timesteps for value in timesteps_by_agent.values()):
            raise ValueError(
                f"Compact selection timesteps differ across agents for {trajectory_key}"
            )
        if not reference_timesteps or max(reference_timesteps) >= int(
            micro_part["observation_count"]
        ):
            raise ValueError(
                f"Compact selection timestep is outside observation range for {trajectory_key}"
            )
        actual_part = normalized_selection_identity(keys)
        expected_part = micro_part.get("compact_selection")
        if not isinstance(expected_part, Mapping) or actual_part != dict(expected_part):
            raise ValueError(
                f"Compact selection key identity differs for planned trajectory {trajectory_key}"
            )
        normalized[trajectory_key] = tuple(keys)
    planned_keys = [key for keys in normalized.values() for key in keys]
    if normalized_selection_identity(planned_keys) != dict(
        sealed["planned_normalized"]
    ):
        raise ValueError(
            "Compact selection planned key identity differs from sealed work plan"
        )
    return normalized


def select_micro_parts(
    plan: Mapping[str, Any],
    *,
    part_indices: Sequence[int] | None = None,
    trajectories: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Select exact units before applying rank::worker_count assignment."""

    parts = [dict(item) for item in plan["micro_parts"]]
    if not part_indices and not trajectories:
        return parts
    by_index = {int(item["part_index"]): item for item in parts}
    by_trajectory = {
        (str(item["source_path"]), str(item["trajectory"])): item for item in parts
    }
    selected: dict[int, dict[str, Any]] = {}
    for value in part_indices or ():
        index = int(value)
        if index not in by_index:
            raise KeyError(f"Exact micro-part index is absent from plan: {index}")
        selected[index] = by_index[index]
    for source, trajectory in trajectories or ():
        key = (normalize_source_path(str(source)), str(trajectory))
        if key not in by_trajectory:
            raise KeyError(f"Exact trajectory identity is absent from plan: {key}")
        item = by_trajectory[key]
        selected[int(item["part_index"])] = item
    return [selected[index] for index in sorted(selected)]


def assigned_micro_parts(
    plan: Mapping[str, Any],
    worker: WorkerIdentity,
    *,
    part_indices: Sequence[int] | None = None,
    trajectories: Sequence[tuple[str, str]] | None = None,
    require_planned_worker_count: bool = True,
) -> list[dict[str, Any]]:
    planned = int(plan["worker_assignment"]["planned_worker_count"])
    if require_planned_worker_count and worker.worker_count != planned:
        raise ValueError(
            f"Runtime worker_count={worker.worker_count} differs from sealed "
            f"planned_worker_count={planned}"
        )
    selected = select_micro_parts(
        plan,
        part_indices=part_indices,
        trajectories=trajectories,
    )
    return selected[worker.global_rank :: worker.worker_count]


def micro_part_roots(
    canonical_output_root: str | Path,
    compact_output_root: str | Path | None,
    micro_part: Mapping[str, Any],
) -> tuple[Path, Path | None]:
    index = int(micro_part["part_index"])
    canonical_base = Path(canonical_output_root).expanduser().resolve()
    compact_base = (
        None
        if compact_output_root is None
        else Path(compact_output_root).expanduser().resolve()
    )
    if compact_base is not None and compact_base == canonical_base:
        raise ValueError("Canonical and compact output roots must be distinct")
    canonical = canonical_base / "parts" / f"part-{index:05d}"
    compact = (
        None
        if compact_base is None
        else compact_base / "parts" / f"part-{index:05d}"
    )
    if compact is not None and compact == canonical:
        raise ValueError("Canonical and compact micro-part roots must be unique")
    return canonical, compact


def _file_state(path: Path) -> tuple[int, int]:
    value = path.stat()
    return int(value.st_size), int(value.st_mtime_ns)


def _verify_identity_file(
    path: Path,
    identity: Mapping[str, Any],
    *,
    verify_sha256: bool,
) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Sealed-plan input is missing: {path}")
    state = _file_state(path)
    expected = (int(identity["bytes"]), int(identity["mtime_ns"]))
    if state != expected:
        raise ValueError(
            f"Sealed-plan input stat mismatch for {path}: expected={expected} actual={state}"
        )
    if verify_sha256:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != str(identity["sha256"]):
            raise ValueError(
                f"Sealed-plan input SHA-256 mismatch for {path}: "
                f"expected={identity['sha256']} actual={actual_sha256}"
            )
    return state


def verify_assigned_sources(
    plan: Mapping[str, Any],
    dataset_root: str | Path,
    micro_parts: Sequence[Mapping[str, Any]],
    *,
    verify_sha256: bool = True,
) -> dict[str, tuple[Path, tuple[int, int], dict[str, Any]]]:
    """Verify only assigned source files; never glob or reopen every HDF5."""

    root = Path(dataset_root).expanduser().resolve()
    identities = source_identity_by_path(plan)
    result: dict[str, tuple[Path, tuple[int, int], dict[str, Any]]] = {}
    for relative in sorted({str(item["source_path"]) for item in micro_parts}):
        identity = identities[relative]
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Assigned source escapes dataset root: {relative}") from exc
        state = _verify_identity_file(path, identity, verify_sha256=verify_sha256)
        result[relative] = (path, state, identity)
    return result


def verify_sources_unchanged(
    verified: Mapping[str, tuple[Path, tuple[int, int], Mapping[str, Any]]],
) -> None:
    """End-of-worker TOCTOU gate: stat only; each source was SHA'd at start."""

    for path, state, _identity in verified.values():
        if _file_state(path) != state:
            raise RuntimeError(f"Assigned source changed during worker execution: {path}")


def verify_checkpoint(
    plan: Mapping[str, Any],
    checkpoint_path: str | Path | None = None,
    *,
    verify_sha256: bool = True,
) -> Path:
    identity = plan["checkpoint"]
    path = Path(identity["path"] if checkpoint_path is None else checkpoint_path).expanduser().resolve()
    _verify_identity_file(path, identity, verify_sha256=verify_sha256)
    return path


def _source_manifest_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": normalize_source_path(str(source["path"])),
        "bytes": int(source["bytes"]),
        "sha256": str(source["sha256"]),
    }


def _verify_manifest_teacher_against_plan(
    teacher: Any,
    plan: Mapping[str, Any],
    *,
    context: str,
) -> None:
    if not isinstance(teacher, Mapping) or teacher.get("checkpoint_sha256") != plan[
        "checkpoint"
    ]["sha256"]:
        raise ValueError(f"{context} checkpoint provenance differs")
    planned_teacher = plan.get("teacher")
    if not isinstance(planned_teacher, Mapping):
        raise TypeError("Sealed work plan lacks teacher provenance")
    for key, expected in planned_teacher.items():
        if key not in teacher or teacher[key] != expected:
            raise ValueError(
                f"{context} teacher provenance field {key!r} differs from sealed plan"
            )


def verify_micro_part(
    part_root: str | Path,
    *,
    plan: Mapping[str, Any],
    micro_part: Mapping[str, Any],
    cache_kind: str,
    verify_shard_checksums: bool = True,
) -> Mapping[str, Any]:
    """Verify COMPLETE, every shard, and exact source/work/checkpoint identity."""

    root = Path(part_root).expanduser().resolve()
    with GaussianCache.open(
        root,
        verify="checksums" if verify_shard_checksums else "manifest",
    ) as cache:
        manifest = cache.manifest
        if cache.schema.cache_kind != str(cache_kind):
            raise ValueError(
                f"Micro-part cache kind mismatch at {root}: "
                f"expected={cache_kind} actual={cache.schema.cache_kind}"
            )
    expected_partition = micro_part_partition_metadata(plan, micro_part)
    actual_partition = manifest.get("partition")
    if not isinstance(actual_partition, Mapping):
        raise TypeError(f"Micro-part lacks partition provenance: {root}")
    for key, expected in expected_partition.items():
        if actual_partition.get(key) != expected:
            raise ValueError(f"Micro-part partition field {key!r} differs at {root}")
    expected_source = _source_manifest_identity(
        source_identity_by_path(plan)[str(micro_part["source_path"])]
    )
    actual_sources = [_source_manifest_identity(source) for source in manifest["sources"]]
    if actual_sources != [expected_source]:
        raise ValueError(f"Micro-part source provenance differs at {root}")
    _verify_manifest_teacher_against_plan(
        manifest.get("teacher"),
        plan,
        context=f"Micro-part at {root}",
    )
    if manifest.get("producer") != plan.get("producer"):
        raise ValueError(f"Micro-part FastWAM producer provenance differs at {root}")
    stream_identities = {
        (str(stream["source_path"]), str(stream["trajectory"]))
        for stream in manifest["streams"]
    }
    expected_stream = {
        (str(micro_part["source_path"]), str(micro_part["trajectory"]))
    }
    if stream_identities != expected_stream:
        raise ValueError(f"Micro-part contains the wrong trajectory at {root}")
    selection_keys = None
    if manifest["selection"]["mode"] == "index":
        selection_path = root / str(manifest["selection"]["index_filename"])
        if sha256_file(selection_path) != str(manifest["selection"]["index_sha256"]):
            raise ValueError(f"Micro-part selection checksum differs at {root}")
        selection_keys = load_selection_jsonl(selection_path)
    if cache_kind == "canonical":
        if manifest["selection"]["mode"] != "all":
            raise ValueError(f"Canonical micro-part must use selection=all at {root}")
        if "plan_identity" in manifest["selection"]:
            raise ValueError(
                f"Canonical micro-part must not inherit sparse selection identity at {root}"
            )
    else:
        if manifest["selection"]["mode"] != "index":
            raise ValueError(f"Compact micro-part must use selection=index at {root}")
        expected_selection_identity = compact_selection_part_identity(plan, micro_part)
        if manifest["selection"].get("plan_identity") != expected_selection_identity:
            raise ValueError(f"Compact micro-part selection plan identity differs at {root}")
        expected_part = expected_selection_identity["part"]
        if manifest["selection"].get("index_sha256") != expected_part["index_sha256"]:
            raise ValueError(f"Compact micro-part selected key SHA-256 differs at {root}")
        if int(manifest["selection"].get("selected_key_count", -1)) != int(
            expected_part["selected_key_count"]
        ):
            raise ValueError(f"Compact micro-part selected key count differs at {root}")
    from .distributed import validate_partition_coverage

    validate_partition_coverage(
        manifest,
        selection_keys=selection_keys,
        context=f"micro-part {micro_part['part_index']}",
    )
    return manifest


def append_failure_log(
    log_root: str | Path,
    *,
    worker: WorkerIdentity | None,
    plan_sha256: str,
    phase: str,
    error: BaseException,
    micro_part: Mapping[str, Any] | None = None,
) -> Path:
    root = Path(log_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rank = "coordinator" if worker is None else f"worker-{worker.global_rank:05d}"
    path = root / f"{rank}.failures.jsonl"
    record: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": str(phase),
        "plan_sha256": str(plan_sha256),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }
    if worker is not None:
        record["worker"] = {
            "platform": worker.platform,
            "global_rank": worker.global_rank,
            "worker_count": worker.worker_count,
            "local_rank": worker.local_rank,
            "cuda_index": worker.cuda_index,
            "node_rank": worker.node_rank,
            "node_count": worker.node_count,
        }
    if micro_part is not None:
        record["micro_part"] = {
            "micro_part_id": micro_part["micro_part_id"],
            "part_index": int(micro_part["part_index"]),
            "source_path": micro_part["source_path"],
            "trajectory": micro_part["trajectory"],
        }
    line = canonical_json_bytes(record)
    with path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def make_transactional_processor(
    build_both: MicroPartProcessor,
    recover_compact_from_canonical: MicroPartProcessor,
    *,
    verify_shard_checksums: bool = True,
) -> MicroPartProcessor:
    """Adapt extraction callbacks to transaction.run_paired_micro_part."""

    _require_plan_aware(build_both, "build_both")
    _require_plan_aware(recover_compact_from_canonical, "recover_compact_from_canonical")

    def process(context: MicroPartContext, teacher: Any) -> Any:
        if context.compact_root is None:
            raise ValueError("Transactional paired processing requires compact_output_root")
        from .transaction import run_paired_micro_part

        return run_paired_micro_part(
            context.canonical_root,
            context.compact_root,
            task_id=str(context.micro_part["micro_part_id"]),
            work_plan_sha256=context.bootstrap.plan["partition"]["work_plan_sha256"],
            micro_part_index=int(context.micro_part["part_index"]),
            work_identity=context.work_identity,
            build_both=lambda: build_both(context, teacher),
            recover_compact_from_canonical=lambda: recover_compact_from_canonical(
                context, teacher
            ),
            verify_existing_shard_checksums=verify_shard_checksums,
            verify_new_shard_checksums=False,
        )

    return plan_aware_processor(process)


def make_official_teacher_factory(settings: OfficialBuildSettings) -> TeacherFactory:
    """Create the repository-owned Policy-Lightning teacher factory."""

    def factory(context: BootstrapContext):
        from .teacher import ExternalPolicyLightningTeacher

        repository_commit = str(context.plan["teacher"].get("repository_commit", ""))
        if not repository_commit:
            raise ValueError("Sealed work plan lacks teacher.repository_commit")
        return ExternalPolicyLightningTeacher(
            repo_path=settings.teacher_repo,
            expected_commit=repository_commit,
            checkpoint_path=context.checkpoint_path,
            checkpoint_sha256=str(context.plan["checkpoint"]["sha256"]),
            config_path=settings.teacher_config,
            device=context.device,
        )

    return factory


def make_official_processor(settings: OfficialBuildSettings) -> MicroPartProcessor:
    """Build canonical+compact parts from the sealed plan with no external callbacks."""

    def compact_keys(context: MicroPartContext) -> tuple[FrameKey, ...]:
        key = (
            str(context.micro_part["source_path"]),
            str(context.micro_part["trajectory"]),
        )
        try:
            return context.bootstrap.compact_selection[key]
        except KeyError as exc:
            raise KeyError(f"No preloaded compact selection for {key}") from exc

    @plan_aware_processor
    def build_both(context: MicroPartContext, teacher: Any) -> None:
        if teacher is None or context.compact_root is None:
            raise RuntimeError("Official paired build requires teacher and compact root")
        source_path = str(context.micro_part["source_path"])
        preverified_state = context.bootstrap.verified_sources[source_path][1]
        from .extract import extract_canonical_cache

        extract_canonical_cache(
            context.bootstrap.dataset_root,
            context.canonical_root,
            teacher=teacher,
            selection="all",
            batch_size=int(settings.batch_size),
            target_shard_bytes=int(settings.target_shard_bytes),
            staging_dir=context.bootstrap.staging_dir,
            verify_uploaded_checksum=bool(settings.verify_uploaded_checksum),
            compact_output_root=context.compact_root,
            compact_selection_keys=compact_keys(context),
            compact_target_shard_bytes=int(settings.compact_target_shard_bytes),
            work_plan=context.bootstrap.plan,
            micro_part_index=int(context.micro_part["part_index"]),
            preverified_source_state=preverified_state,
        )

    @plan_aware_processor
    def recover_compact(context: MicroPartContext, teacher: Any) -> None:
        del teacher
        if context.compact_root is None:
            raise RuntimeError("Official compact recovery requires compact root")
        from .compact import (
            COMPACT_HEIGHT,
            COMPACT_WIDTH,
            MOMENT_MATCH_METHOD,
            project_compact_cache,
        )

        project_compact_cache(
            context.canonical_root,
            context.compact_root,
            selection="index",
            selection_keys=compact_keys(context),
            verify="manifest",
            batch_size=int(settings.batch_size),
            target_shard_bytes=int(settings.compact_target_shard_bytes),
            staging_dir=context.bootstrap.staging_dir,
            verify_uploaded_checksum=bool(settings.verify_uploaded_checksum),
            partition=context.partition_metadata,
            preserve_parent_teacher=True,
            producer=context.bootstrap.plan["producer"],
            selection_plan_identity=compact_selection_part_identity(
                context.bootstrap.plan,
                context.micro_part,
            ),
            derivation={
                "method": MOMENT_MATCH_METHOD,
                "output_size": [COMPACT_HEIGHT, COMPACT_WIDTH],
                "source": "same-teacher-forward-canonical-v1",
                "canonical_work_plan_sha256": context.bootstrap.plan["partition"]
                ["work_plan_sha256"],
            },
        )

    return make_transactional_processor(
        build_both,
        recover_compact,
        verify_shard_checksums=True,
    )


def _part_need(
    root: Path | None,
    *,
    plan: Mapping[str, Any],
    micro_part: Mapping[str, Any],
    cache_kind: str,
    verify_shard_checksums: bool,
    part_verifier: PartVerifier,
) -> bool:
    if root is None:
        return False
    if not root.exists():
        return True
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Micro-part root is not a real directory: {root}")
    if not (root / "COMPLETE").is_file():
        # Incomplete task-owned cleanup/recovery is delegated to transaction.py
        # (or to an explicitly plan-aware processor).  It is never skipped.
        return True
    # Existing output is never treated as resumable merely because COMPLETE
    # exists.  It must pass manifest, shard, source/work and checkpoint checks.
    part_verifier(
        root,
        plan=plan,
        micro_part=micro_part,
        cache_kind=cache_kind,
        verify_shard_checksums=verify_shard_checksums,
    )
    return False


def run_worker(
    plan_root: str | Path,
    dataset_root: str | Path,
    canonical_output_root: str | Path,
    *,
    staging_dir: str | Path,
    teacher_factory: TeacherFactory,
    process_micro_part: MicroPartProcessor,
    worker: WorkerIdentity,
    compact_output_root: str | Path | None = None,
    compact_selection_jsonl: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    part_indices: Sequence[int] | None = None,
    trajectories: Sequence[tuple[str, str]] | None = None,
    require_planned_worker_count: bool = True,
    verify_shard_checksums: bool = True,
    verify_source_sha256_before: bool = True,
    min_staging_free_bytes: int = MIN_STAGING_FREE_BYTES,
    failure_log_root: str | Path | None = None,
    cuda_binder: Callable[[WorkerIdentity], Any] = bind_cuda_device,
    part_verifier: PartVerifier = verify_micro_part,
) -> dict[str, Any]:
    """Run rank-strided micro-parts with one teacher instance on one GPU."""

    resolved_plan_root = Path(plan_root).expanduser().resolve()
    plan = load_work_plan(resolved_plan_root)
    plan_sha256 = work_plan_payload_sha256(plan)
    _require_plan_aware(process_micro_part, "process_micro_part")
    if (compact_output_root is None) != (compact_selection_jsonl is None):
        raise ValueError(
            "compact_output_root and compact_selection_jsonl must be supplied together"
        )
    assigned = assigned_micro_parts(
        plan,
        worker,
        part_indices=part_indices,
        trajectories=trajectories,
        require_planned_worker_count=require_planned_worker_count,
    )
    log_root = (
        Path(failure_log_root).expanduser().resolve()
        if failure_log_root is not None
        else Path(canonical_output_root).expanduser().resolve() / "failures"
    )
    pending: list[tuple[dict[str, Any], Path, Path | None, bool, bool]] = []
    skipped = 0
    current: Mapping[str, Any] | None = None
    try:
        compact_selection = (
            {}
            if compact_selection_jsonl is None
            else load_compact_selection_for_plan(plan, compact_selection_jsonl)
        )
        for current in assigned:
            canonical_root, compact_root = micro_part_roots(
                canonical_output_root,
                compact_output_root,
                current,
            )
            need_canonical = _part_need(
                canonical_root,
                plan=plan,
                micro_part=current,
                cache_kind="canonical",
                verify_shard_checksums=verify_shard_checksums,
                part_verifier=part_verifier,
            )
            need_compact = _part_need(
                compact_root,
                plan=plan,
                micro_part=current,
                cache_kind="compact",
                verify_shard_checksums=verify_shard_checksums,
                part_verifier=part_verifier,
            )
            if not need_canonical and not need_compact:
                skipped += 1
                continue
            pending.append(
                (current, canonical_root, compact_root, need_canonical, need_compact)
            )

        if not pending:
            return {
                "worker_rank": worker.global_rank,
                "worker_count": worker.worker_count,
                "assigned": len(assigned),
                "processed": 0,
                "skipped_verified": skipped,
                "teacher_loads": 0,
                "plan_sha256": plan_sha256,
            }

        staging_base = ensure_node_local_staging(
            staging_dir,
            min_free_bytes=min_staging_free_bytes,
        )
        worker_staging = staging_base / plan_sha256[:16] / f"worker-{worker.global_rank:05d}"
        worker_staging.mkdir(parents=True, exist_ok=True)
        device = cuda_binder(worker)
        checkpoint = verify_checkpoint(plan, checkpoint_path, verify_sha256=True)
        pending_parts = [item[0] for item in pending]
        source_states = verify_assigned_sources(
            plan,
            dataset_root,
            pending_parts,
            verify_sha256=verify_source_sha256_before,
        )
        bootstrap = BootstrapContext(
            plan_root=resolved_plan_root,
            plan=plan,
            plan_sha256=plan_sha256,
            dataset_root=Path(dataset_root).expanduser().resolve(),
            checkpoint_path=checkpoint,
            worker=worker,
            device=device,
            staging_dir=worker_staging,
            verified_sources=source_states,
            compact_selection=compact_selection,
        )
        # Deliberately outside the loop: exactly one teacher construction per
        # worker/GPU, irrespective of the number of assigned trajectories.
        needs_teacher = any(item[3] for item in pending)
        teacher = teacher_factory(bootstrap) if needs_teacher else None
        processed = 0
        for current, canonical_root, compact_root, need_canonical, need_compact in pending:
            context = MicroPartContext(
                bootstrap=bootstrap,
                micro_part=current,
                source_identity=source_identity_by_path(plan)[str(current["source_path"])],
                canonical_root=canonical_root,
                compact_root=compact_root,
                need_canonical=need_canonical,
                need_compact=need_compact,
            )
            process_micro_part(context, teacher)
            if need_canonical:
                part_verifier(
                    canonical_root,
                    plan=plan,
                    micro_part=current,
                    cache_kind="canonical",
                    verify_shard_checksums=False,
                )
            if need_compact:
                assert compact_root is not None
                part_verifier(
                    compact_root,
                    plan=plan,
                    micro_part=current,
                    cache_kind="compact",
                    verify_shard_checksums=False,
                )
            processed += 1
        verify_sources_unchanged(source_states)
        return {
            "worker_rank": worker.global_rank,
            "worker_count": worker.worker_count,
            "assigned": len(assigned),
            "processed": processed,
            "skipped_verified": skipped,
            "teacher_loads": int(needs_teacher),
            "plan_sha256": plan_sha256,
            "staging_dir": str(worker_staging),
            "device": str(device),
        }
    except Exception as error:
        append_failure_log(
            log_root,
            worker=worker,
            plan_sha256=plan_sha256,
            phase="worker",
            error=error,
            micro_part=current,
        )
        raise


def verify_merged_cache(
    cache_root: str | Path,
    *,
    plan: Mapping[str, Any],
    cache_kind: str,
    verify_shard_checksums: bool,
) -> Mapping[str, Any]:
    root = Path(cache_root).expanduser().resolve()
    with GaussianCache.open(
        root,
        verify="checksums" if verify_shard_checksums else "manifest",
    ) as cache:
        manifest = cache.manifest
        if cache.schema.cache_kind != cache_kind:
            raise ValueError(f"Merged cache kind mismatch: {root}")
    partition = manifest.get("partition")
    if not isinstance(partition, Mapping) or partition.get("merged") is not True:
        raise ValueError(f"Top-level cache is not a merged cache: {root}")
    expected = plan["partition"]
    for key in (
        "algorithm",
        "unit",
        "partition_count",
        "expected_unit_count",
        "expected_unit_weight",
        "work_plan_sha256",
    ):
        if partition.get(key) != expected.get(key):
            raise ValueError(f"Merged cache partition field {key!r} differs: {root}")
    if len(manifest.get("parts", [])) != len(plan["micro_parts"]):
        raise ValueError(f"Merged cache part count differs from sealed plan: {root}")
    expected_sources = sorted(
        (_source_manifest_identity(source) for source in plan["dataset"]["sources"]),
        key=lambda item: item["path"],
    )
    actual_sources = sorted(
        (_source_manifest_identity(source) for source in manifest["sources"]),
        key=lambda item: item["path"],
    )
    if actual_sources != expected_sources:
        raise ValueError(f"Merged cache source provenance differs from sealed plan: {root}")
    _verify_manifest_teacher_against_plan(
        manifest.get("teacher"),
        plan,
        context=f"Merged cache at {root}",
    )
    if manifest.get("producer") != plan.get("producer"):
        raise ValueError(f"Merged cache FastWAM producer provenance differs: {root}")
    if cache_kind == "canonical":
        if manifest["selection"]["mode"] != "all":
            raise ValueError(f"Merged canonical cache must use selection=all: {root}")
        if "plan_identity" in manifest["selection"]:
            raise ValueError(
                f"Merged canonical cache must not inherit sparse selection identity: {root}"
            )
    else:
        if manifest["selection"]["mode"] != "index":
            raise ValueError(f"Merged compact cache must use selection=index: {root}")
        expected_identity = compact_selection_part_identity(
            plan,
            plan["micro_parts"][0],
        )
        expected_identity.pop("part")
        expected_identity["merged"] = True
        if manifest["selection"].get("plan_identity") != expected_identity:
            raise ValueError(f"Merged compact selection plan identity differs: {root}")
        planned = plan["compact_selection"]["planned_normalized"]
        if manifest["selection"].get("index_sha256") != planned["index_sha256"]:
            raise ValueError(f"Merged compact selected key SHA-256 differs: {root}")
        if int(manifest["selection"].get("selected_key_count", -1)) != int(
            planned["selected_key_count"]
        ):
            raise ValueError(f"Merged compact selected key count differs: {root}")
    return manifest


def merge_and_validate(
    plan_root: str | Path,
    dataset_root: str | Path,
    canonical_output_root: str | Path,
    *,
    compact_output_root: str | Path | None = None,
    verify_part_checksums: bool = True,
    verify_source_checksums: bool = True,
    failure_log_root: str | Path | None = None,
) -> dict[str, Any]:
    """Coordinator-only zero-copy merge followed by fail-closed validation."""

    plan = load_work_plan(plan_root)
    plan_sha256 = work_plan_payload_sha256(plan)
    canonical_root = Path(canonical_output_root).expanduser().resolve()
    compact_root = (
        None
        if compact_output_root is None
        else Path(compact_output_root).expanduser().resolve()
    )
    if compact_root is not None and compact_root == canonical_root:
        raise ValueError("Canonical and compact output roots must be distinct")
    log_root = (
        Path(failure_log_root).expanduser().resolve()
        if failure_log_root is not None
        else canonical_root / "failures"
    )
    try:
        canonical_parts: list[Path] = []
        compact_parts: list[Path] = []
        for micro_part in plan["micro_parts"]:
            canonical_part, compact_part = micro_part_roots(
                canonical_root,
                compact_root,
                micro_part,
            )
            verify_micro_part(
                canonical_part,
                plan=plan,
                micro_part=micro_part,
                cache_kind="canonical",
                verify_shard_checksums=False,
            )
            canonical_parts.append(canonical_part)
            if compact_part is not None:
                verify_micro_part(
                    compact_part,
                    plan=plan,
                    micro_part=micro_part,
                    cache_kind="compact",
                    verify_shard_checksums=False,
                )
                compact_parts.append(compact_part)

        if (canonical_root / "COMPLETE").exists() and not (
            canonical_root / MERGE_BUILDING_FILENAME
        ).exists():
            verify_merged_cache(
                canonical_root,
                plan=plan,
                cache_kind="canonical",
                verify_shard_checksums=False,
            )
        else:
            merge_part_manifests(
                canonical_parts,
                canonical_root,
                verify_part_checksums=False,
            )
        canonical_validation = validate_cache(
            canonical_root,
            verify_shard_checksums=verify_part_checksums,
            source_root=dataset_root,
            verify_source_checksums=verify_source_checksums,
            semantic_mode="coverage",
        )
        result: dict[str, Any] = {
            "plan_sha256": plan_sha256,
            "micro_part_count": len(plan["micro_parts"]),
            "canonical": canonical_validation,
        }
        if compact_root is not None:
            if (compact_root / "COMPLETE").exists() and not (
                compact_root / MERGE_BUILDING_FILENAME
            ).exists():
                verify_merged_cache(
                    compact_root,
                    plan=plan,
                    cache_kind="compact",
                    verify_shard_checksums=False,
                )
            else:
                merge_part_manifests(
                    compact_parts,
                    compact_root,
                    verify_part_checksums=False,
                    canonical_root=canonical_root,
                )
            result["compact"] = validate_cache(
                compact_root,
                verify_shard_checksums=verify_part_checksums,
                source_root=dataset_root,
                # Canonical validation already performed the single full source
                # checksum pass; compact provenance is required to match plan.
                verify_source_checksums=False,
                semantic_mode="coverage",
            )
        return result
    except Exception as error:
        append_failure_log(
            log_root,
            worker=None,
            plan_sha256=plan_sha256,
            phase="merge-validate",
            error=error,
        )
        raise


def _load_callable(specification: str) -> Callable[..., Any]:
    module_name, separator, attribute = str(specification).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Callable must be specified as 'python.module:attribute'")
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"Imported object is not callable: {specification}")
    return value


def _load_teacher_training_provenance(
    path_value: str,
    *,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Load and seal the external teacher's declared training-data lineage."""

    requested_path = Path(path_value).expanduser()
    if requested_path.is_symlink() or not requested_path.is_file():
        raise ValueError(
            "Teacher training provenance must be a regular non-symlink file: "
            f"{requested_path}"
        )
    path = requested_path.resolve(strict=True)
    raw = path.read_bytes()
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid teacher training provenance JSON: {path}") from error
    if not isinstance(record, dict):
        raise TypeError("Teacher training provenance must contain a JSON object")
    checkpoint = record.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or str(checkpoint.get("sha256", "")).lower() != (
        str(expected_checkpoint_sha256).lower()
    ):
        raise ValueError(
            "Teacher training provenance checkpoint SHA-256 does not match the "
            "planned checkpoint"
        )
    return {
        "record": record,
        "record_bytes": len(raw),
        "record_filename": path.name,
        "record_sha256": sha256_file(path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="coordinator: seal source/trajectory/checkpoint plan")
    plan.add_argument("--plan-root", required=True)
    plan.add_argument("--dataset-root", required=True)
    plan.add_argument("--checkpoint", required=True)
    plan.add_argument("--checkpoint-sha256", required=True)
    plan.add_argument("--compact-selection-jsonl", required=True)
    plan.add_argument("--planned-worker-count", type=int, default=4)
    plan.add_argument("--teacher-repository-commit", required=True)
    plan.add_argument("--teacher-repository-url", required=True)
    plan.add_argument("--teacher-config-relative-path", required=True)
    plan.add_argument("--teacher-config-sha256", required=True)
    plan.add_argument(
        "--teacher-training-provenance-json",
        required=True,
        help="sealed declaration of teacher training datasets and target-overlap limits",
    )
    plan.add_argument(
        "--producer-repo",
        default=str(Path(__file__).resolve().parents[4]),
        help="FastWAM Git checkout whose source snapshot is sealed into the plan",
    )
    plan.add_argument(
        "--allow-dirty-producer-snapshot",
        action="store_true",
        help="diagnostic only; formal plans require a clean FastWAM commit",
    )
    plan.add_argument(
        "--trajectory",
        action="append",
        help="optional exact source.h5::trajectory test-plan scope; repeatable",
    )

    worker = commands.add_parser("worker", help="worker: process rank-strided trajectory parts")
    worker.add_argument("--plan-root", required=True)
    worker.add_argument("--dataset-root", required=True)
    worker.add_argument("--canonical-output-root", required=True)
    worker.add_argument("--compact-output-root")
    worker.add_argument("--compact-selection-jsonl")
    worker.add_argument("--staging-dir", required=True)
    worker.add_argument("--checkpoint")
    worker.add_argument("--platform", choices=("auto", "dsw", "dlc", "torchrun"), default="auto")
    worker.add_argument("--rank", type=int)
    worker.add_argument("--worker-count", type=int)
    worker.add_argument("--local-rank", type=int)
    worker.add_argument("--micro-part-index", type=int, action="append")
    worker.add_argument(
        "--trajectory",
        action="append",
        help="exact relative/source.h5::trajectory_name; repeatable",
    )
    worker.add_argument("--teacher-repo", help="pinned Policy-Lightning checkout")
    worker.add_argument(
        "--teacher-config",
        default="config/encoder/noposplat.yaml",
    )
    worker.add_argument("--batch-size", type=int, default=8)
    worker.add_argument("--target-shard-gib", type=float, default=2.0)
    worker.add_argument("--compact-target-shard-gib", type=float, default=2.0)
    worker.add_argument(
        "--teacher-factory",
        help="diagnostic override python.module:callable; formal path omits this",
    )
    processor = worker.add_mutually_exclusive_group()
    processor.add_argument("--processor", help="python.module:callable")
    processor.add_argument("--build-both", help="transactional python.module:callable")
    worker.add_argument("--recover-compact", help="required with --build-both")
    worker.add_argument("--failure-log-root")
    worker.add_argument("--min-staging-free-gib", type=float, default=25.0)
    worker.add_argument("--no-shard-checksums", action="store_true")
    worker.add_argument("--allow-worker-count-mismatch", action="store_true")

    merge = commands.add_parser("merge-validate", help="coordinator: merge and validate all parts")
    merge.add_argument("--plan-root", required=True)
    merge.add_argument("--dataset-root", required=True)
    merge.add_argument("--canonical-output-root", required=True)
    merge.add_argument("--compact-output-root")
    merge.add_argument("--failure-log-root")
    merge.add_argument("--no-shard-checksums", action="store_true")
    merge.add_argument("--no-source-checksums", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "plan":
        teacher_identity = {
            "repository_commit": args.teacher_repository_commit,
            "repository_url": args.teacher_repository_url,
            "config_relative_path": args.teacher_config_relative_path,
            "config_sha256": args.teacher_config_sha256,
        }
        teacher_identity["training_data_provenance"] = (
            _load_teacher_training_provenance(
                args.teacher_training_provenance_json,
                expected_checkpoint_sha256=args.checkpoint_sha256,
            )
        )
        plan = create_work_plan(
            args.plan_root,
            args.dataset_root,
            args.checkpoint,
            expected_checkpoint_sha256=args.checkpoint_sha256,
            planned_worker_count=args.planned_worker_count,
            teacher_identity=teacher_identity,
            trajectories=[
                parse_trajectory_selector(value) for value in args.trajectory or ()
            ],
            compact_selection_jsonl=args.compact_selection_jsonl,
            producer_repository_root=args.producer_repo,
            require_clean_producer=not args.allow_dirty_producer_snapshot,
        )
        result = {
            "plan_root": str(Path(args.plan_root).expanduser().resolve()),
            "plan_sha256": work_plan_payload_sha256(plan),
            "source_count": len(plan["dataset"]["sources"]),
            "micro_part_count": len(plan["micro_parts"]),
            "compact_selection_raw_sha256": plan["compact_selection"]["raw"][
                "sha256"
            ],
            "compact_selection_normalized_sha256": plan["compact_selection"][
                "normalized"
            ]["index_sha256"],
            "compact_selection_normalized_count": plan["compact_selection"][
                "normalized"
            ]["selected_key_count"],
            "producer_source_snapshot_sha256": plan["producer"][
                "source_snapshot_sha256"
            ],
        }
    elif args.command == "worker":
        identity = resolve_worker_identity(
            args.platform,
            rank=args.rank,
            worker_count=args.worker_count,
            local_rank=args.local_rank,
        )
        verify_checksums = not args.no_shard_checksums
        if args.processor:
            if not args.teacher_factory:
                raise ValueError("--processor diagnostic override requires --teacher-factory")
            teacher_factory = _load_callable(args.teacher_factory)
            process = _load_callable(args.processor)
        elif args.build_both:
            if not args.teacher_factory:
                raise ValueError("--build-both diagnostic override requires --teacher-factory")
            if not args.recover_compact:
                raise ValueError("--recover-compact is required with --build-both")
            teacher_factory = _load_callable(args.teacher_factory)
            process = make_transactional_processor(
                _load_callable(args.build_both),
                _load_callable(args.recover_compact),
                verify_shard_checksums=verify_checksums,
            )
        else:
            if args.teacher_factory or args.recover_compact:
                raise ValueError("Incomplete diagnostic callback override")
            if not args.teacher_repo:
                raise ValueError("Formal worker path requires --teacher-repo")
            if not args.compact_output_root or not args.compact_selection_jsonl:
                raise ValueError(
                    "Formal worker requires compact output and preloaded selection JSONL"
                )
            settings = OfficialBuildSettings(
                teacher_repo=Path(args.teacher_repo).expanduser().resolve(),
                teacher_config=args.teacher_config,
                batch_size=args.batch_size,
                target_shard_bytes=int(float(args.target_shard_gib) * (1 << 30)),
                compact_target_shard_bytes=int(
                    float(args.compact_target_shard_gib) * (1 << 30)
                ),
                verify_uploaded_checksum=True,
            )
            teacher_factory = make_official_teacher_factory(settings)
            process = make_official_processor(settings)
        trajectories = [parse_trajectory_selector(value) for value in args.trajectory or ()]
        result = run_worker(
            args.plan_root,
            args.dataset_root,
            args.canonical_output_root,
            staging_dir=args.staging_dir,
            teacher_factory=teacher_factory,
            process_micro_part=process,
            worker=identity,
            compact_output_root=args.compact_output_root,
            compact_selection_jsonl=args.compact_selection_jsonl,
            checkpoint_path=args.checkpoint,
            part_indices=args.micro_part_index,
            trajectories=trajectories,
            require_planned_worker_count=not args.allow_worker_count_mismatch,
            verify_shard_checksums=verify_checksums,
            min_staging_free_bytes=int(float(args.min_staging_free_gib) * (1 << 30)),
            failure_log_root=args.failure_log_root,
        )
    else:
        result = merge_and_validate(
            args.plan_root,
            args.dataset_root,
            args.canonical_output_root,
            compact_output_root=args.compact_output_root,
            verify_part_checksums=not args.no_shard_checksums,
            verify_source_checksums=not args.no_source_checksums,
            failure_log_root=args.failure_log_root,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
