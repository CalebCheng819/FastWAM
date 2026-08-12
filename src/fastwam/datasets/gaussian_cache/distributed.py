"""Deterministic work partitioning and zero-copy part-cache merging."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import (
    COMPLETE_FILENAME,
    MANIFEST_FILENAME,
    canonical_json_bytes,
    load_manifest,
    seal_manifest,
    sha256_file,
    write_immutable_file,
)
from .schema import FrameKey, GaussianCacheSchema, normalize_source_path
from .selection import load_selection_jsonl, write_normalized_selection_index

SOURCE_PARTITION_ALGORITHM = "lpt-source-bytes-v1"
TRAJECTORY_PARTITION_ALGORITHM = "lpt-trajectory-observation-agent-v1"
# Formal extraction uses trajectory work units; retain the generic export name.
PARTITION_ALGORITHM = TRAJECTORY_PARTITION_ALGORITHM
PARTITION_UNITS = {"source", "trajectory"}
MERGE_BUILDING_FILENAME = "MERGE.BUILDING.json"
MERGE_TRANSACTION_SCHEMA = "fastwam-gaussian-top-level-merge"
MERGE_TRANSACTION_VERSION = 1


def _normalized_sources(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [
        {
            "path": normalize_source_path(str(record["path"])),
            "bytes": int(record["bytes"]),
            "sha256": str(record["sha256"]),
        }
        for record in sources
    ]
    paths = [record["path"] for record in normalized]
    if len(set(paths)) != len(paths):
        raise ValueError("Source records contain duplicate paths")
    return sorted(normalized, key=lambda record: record["path"])


def _algorithm(unit: str) -> str:
    unit = str(unit)
    if unit == "source":
        return SOURCE_PARTITION_ALGORITHM
    if unit == "trajectory":
        return TRAJECTORY_PARTITION_ALGORITHM
    raise ValueError(f"partition unit must be one of {sorted(PARTITION_UNITS)}, got {unit!r}")


def _normalize_work_units(
    units: Sequence[Mapping[str, Any]],
    unit: str,
) -> list[dict[str, Any]]:
    unit = str(unit)
    _algorithm(unit)
    normalized: list[dict[str, Any]] = []
    for record in units:
        item: dict[str, Any] = {
            "source_path": normalize_source_path(str(record["source_path"])),
            "weight": int(record["weight"]),
        }
        if item["weight"] <= 0:
            raise ValueError(f"Partition work-unit weight must be positive: {record}")
        if unit == "trajectory":
            trajectory = str(record["trajectory"])
            agent_names = [str(value) for value in record["agent_names"]]
            observation_count = int(record["observation_count"])
            if not trajectory or not agent_names or len(set(agent_names)) != len(agent_names):
                raise ValueError(f"Invalid trajectory partition work unit: {record}")
            if observation_count <= 0 or item["weight"] != observation_count * len(agent_names):
                raise ValueError(
                    "Trajectory work weight must equal observation_count*num_agents: "
                    f"{record}"
                )
            item.update(
                {
                    "trajectory": trajectory,
                    "observation_count": observation_count,
                    "agent_names": agent_names,
                }
            )
        normalized.append(item)
    keys = [_work_unit_key(record, unit) for record in normalized]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Partition plan contains duplicate {unit} work units")
    return sorted(normalized, key=lambda record: _work_unit_key(record, unit))


def _work_unit_key(record: Mapping[str, Any], unit: str) -> tuple[str, ...]:
    if unit == "source":
        return (str(record["source_path"]),)
    return (str(record["source_path"]), str(record["trajectory"]))


def partition_work_units(
    units: Sequence[Mapping[str, Any]],
    partition_count: int,
    *,
    unit: str,
) -> list[list[dict[str, Any]]]:
    """Deterministic LPT balancing with stable partition-index tie breaks."""

    normalized = _normalize_work_units(units, unit)
    count = int(partition_count)
    if count <= 0:
        raise ValueError("partition_count must be positive")
    if count > len(normalized):
        raise ValueError(
            f"partition_count cannot exceed {unit} work-unit count; "
            f"got {count} partitions for {len(normalized)} units"
        )
    partitions: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    totals = [0] * count
    for record in sorted(
        normalized,
        key=lambda item: (-int(item["weight"]), _work_unit_key(item, unit)),
    ):
        index = min(range(count), key=lambda value: (totals[value], value))
        partitions[index].append(record)
        totals[index] += int(record["weight"])
    for index in range(count):
        partitions[index].sort(key=lambda record: _work_unit_key(record, unit))
    return partitions


def work_plan_sha256(
    units: Sequence[Mapping[str, Any]],
    partition_count: int,
    *,
    unit: str,
) -> str:
    payload = canonical_json_bytes(
        {
            "algorithm": _algorithm(unit),
            "unit": unit,
            "partition_count": int(partition_count),
            "work_units": _normalize_work_units(units, unit),
        }
    )
    return hashlib.sha256(payload).hexdigest()


def partition_work_metadata(
    units: Sequence[Mapping[str, Any]],
    *,
    partition_index: int,
    partition_count: int,
    unit: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized = _normalize_work_units(units, unit)
    partitions = partition_work_units(normalized, partition_count, unit=unit)
    index = int(partition_index)
    if not 0 <= index < int(partition_count):
        raise ValueError(
            f"partition_index must be in [0,{int(partition_count)}), got {partition_index}"
        )
    assigned = partitions[index]
    metadata = {
        "algorithm": _algorithm(unit),
        "unit": unit,
        "partition_index": index,
        "partition_count": int(partition_count),
        "expected_unit_count": len(normalized),
        "expected_unit_weight": sum(int(record["weight"]) for record in normalized),
        "work_plan_sha256": work_plan_sha256(
            normalized,
            int(partition_count),
            unit=unit,
        ),
        "assigned_unit_count": len(assigned),
        "assigned_unit_weight": sum(int(record["weight"]) for record in assigned),
        "assigned_units": assigned,
    }
    return assigned, metadata


def source_plan_sha256(
    sources: Sequence[Mapping[str, Any]],
    partition_count: int,
) -> str:
    units = [
        {"source_path": record["path"], "weight": int(record["bytes"])}
        for record in _normalized_sources(sources)
    ]
    return work_plan_sha256(units, partition_count, unit="source")


def partition_source_records(
    sources: Sequence[Mapping[str, Any]],
    partition_count: int,
) -> list[list[dict[str, Any]]]:
    """Compatibility wrapper for deterministic source-byte partitioning."""

    records = _normalized_sources(sources)
    by_path = {record["path"]: record for record in records}
    units = [
        {"source_path": record["path"], "weight": int(record["bytes"])}
        for record in records
    ]
    return [
        [by_path[unit["source_path"]] for unit in partition]
        for partition in partition_work_units(units, partition_count, unit="source")
    ]


def partition_metadata(
    sources: Sequence[Mapping[str, Any]],
    *,
    partition_index: int,
    partition_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compatibility wrapper returning assigned source records and metadata."""

    records = _normalized_sources(sources)
    by_path = {record["path"]: record for record in records}
    units = [
        {"source_path": record["path"], "weight": int(record["bytes"])}
        for record in records
    ]
    assigned_units, metadata = partition_work_metadata(
        units,
        partition_index=partition_index,
        partition_count=partition_count,
        unit="source",
    )
    return [by_path[unit["source_path"]] for unit in assigned_units], metadata


def _part_relative_path(part_root: Path, output_root: Path, partition_index: int) -> str:
    try:
        relative = part_root.resolve().relative_to(output_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Part cache {part_root} must be inside merged output root {output_root}"
        ) from exc
    expected = f"parts/part-{partition_index:05d}"
    if relative != expected:
        raise ValueError(
            f"Part {partition_index} must be stored at {expected}, got {relative}"
        )
    return relative


def _source_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": normalize_source_path(str(record["path"])),
        "bytes": int(record["bytes"]),
        "sha256": str(record["sha256"]),
    }


def validate_partition_coverage(
    manifest: Mapping[str, Any],
    *,
    selection_keys: Sequence[FrameKey] | None = None,
    context: str = "part cache",
) -> dict[str, int]:
    """Validate exact trajectory/agent/count coverage against sealed work metadata.

    This is deliberately stricter than checking that stream work-unit keys are
    a subset of the assignment.  It rejects a cache that silently omits one
    agent, changes an observation count, or reports a stored-frame total that
    differs from the partition's assigned weight/selection index.
    """

    partition = manifest.get("partition")
    if not isinstance(partition, Mapping) or bool(partition.get("merged")):
        raise ValueError(f"{context} lacks unmerged partition assignment metadata")
    unit = str(partition.get("unit"))
    assigned_units = _normalize_work_units(partition.get("assigned_units", []), unit)
    streams = list(manifest.get("streams", []))
    if unit != "trajectory":
        # Source partitioning is a retained compatibility path and cannot
        # encode the exact trajectory/agent universe in its assignment record.
        return {
            "trajectory_count": 0,
            "stream_count": len(streams),
            "stored_count": sum(int(record["stored_count"]) for record in streams),
        }

    expected_by_unit = {
        _work_unit_key(record, unit): record for record in assigned_units
    }
    streams_by_unit: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for stream in streams:
        key = (str(stream["source_path"]), str(stream["trajectory"]))
        streams_by_unit[key].append(stream)
    if set(streams_by_unit) != set(expected_by_unit):
        missing = sorted(set(expected_by_unit) - set(streams_by_unit))
        extra = sorted(set(streams_by_unit) - set(expected_by_unit))
        raise ValueError(
            f"{context} trajectory coverage mismatch: missing={missing[:16]} extra={extra[:16]}"
        )

    selection_mode = str(manifest["selection"]["mode"])
    selected_counts: Counter[tuple[str, str, str]] = Counter()
    if selection_mode == "index":
        if selection_keys is None:
            raise ValueError(f"{context} selection=index requires normalized selection keys")
        selected_counts.update(
            (key.source_path, key.trajectory, key.agent_name) for key in selection_keys
        )
    elif selection_mode != "all":
        raise ValueError(f"{context} has unsupported selection mode {selection_mode!r}")

    expected_stored_total = 0
    for unit_key, work_unit in expected_by_unit.items():
        unit_streams = streams_by_unit[unit_key]
        actual_agents = [str(stream["agent_name"]) for stream in unit_streams]
        expected_agents = [str(name) for name in work_unit["agent_names"]]
        if len(actual_agents) != len(set(actual_agents)) or set(actual_agents) != set(
            expected_agents
        ):
            raise ValueError(
                f"{context} agent completeness mismatch for {unit_key}: "
                f"expected={sorted(expected_agents)} actual={sorted(actual_agents)}"
            )
        observation_count = int(work_unit["observation_count"])
        for stream in unit_streams:
            agent_name = str(stream["agent_name"])
            actual_observations = int(stream["observation_count"])
            if actual_observations != observation_count:
                raise ValueError(
                    f"{context} observation_count mismatch for "
                    f"{(*unit_key, agent_name)}: expected={observation_count} "
                    f"actual={actual_observations}"
                )
            expected_stored = (
                observation_count
                if selection_mode == "all"
                else int(selected_counts[(*unit_key, agent_name)])
            )
            if expected_stored <= 0:
                raise ValueError(
                    f"{context} selection omits assigned agent {(*unit_key, agent_name)}"
                )
            actual_stored = int(stream["stored_count"])
            if actual_stored != expected_stored:
                raise ValueError(
                    f"{context} stored_count mismatch for {(*unit_key, agent_name)}: "
                    f"expected={expected_stored} actual={actual_stored}"
                )
            expected_stored_total += expected_stored

    actual_stored_total = sum(int(stream["stored_count"]) for stream in streams)
    if actual_stored_total != expected_stored_total:
        raise ValueError(
            f"{context} stored-frame total mismatch: expected={expected_stored_total} "
            f"actual={actual_stored_total}"
        )
    if selection_mode == "all":
        assigned_weight = int(partition["assigned_unit_weight"])
        if expected_stored_total != assigned_weight:
            raise ValueError(
                f"{context} assigned weight mismatch: expected={assigned_weight} "
                f"actual={expected_stored_total}"
            )
    elif expected_stored_total != len(selection_keys or ()):
        raise ValueError(
            f"{context} selection contains keys outside assigned agent streams"
        )
    return {
        "trajectory_count": len(expected_by_unit),
        "stream_count": len(streams),
        "stored_count": actual_stored_total,
    }


def _selection_plan_global(value: Mapping[str, Any]) -> dict[str, Any]:
    part = value.get("part")
    if not isinstance(part, Mapping):
        raise TypeError("Compact selection plan identity lacks per-part identity")
    return {key: item for key, item in value.items() if key != "part"}


def _merge_transaction_identity(
    *,
    root: Path,
    schema: Mapping[str, Any],
    partition: Mapping[str, Any],
    loaded: Sequence[tuple[int, Path, Mapping[str, Any]]],
    producer: Mapping[str, Any] | None,
    selection_plan: Mapping[str, Any] | None,
    canonical_root: str | Path | None,
) -> dict[str, Any]:
    canonical_manifest_sha256 = None
    if canonical_root is not None:
        canonical_manifest_sha256 = sha256_file(
            Path(canonical_root).expanduser().resolve() / MANIFEST_FILENAME
        )
    return {
        "schema_name": MERGE_TRANSACTION_SCHEMA,
        "schema_version": MERGE_TRANSACTION_VERSION,
        "output_root": str(root),
        "cache_kind": str(schema["cache_kind"]),
        "work_plan_sha256": str(partition["work_plan_sha256"]),
        "producer": None if producer is None else dict(producer),
        "selection_plan": None if selection_plan is None else dict(selection_plan),
        "canonical_manifest_sha256": canonical_manifest_sha256,
        "parts": [
            {
                "partition_index": int(index),
                "path": _part_relative_path(part_root, root, index),
                "manifest_sha256": sha256_file(part_root / MANIFEST_FILENAME),
            }
            for index, part_root, _manifest in loaded
        ],
    }


def _read_merge_marker(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid top-level merge BUILDING marker: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Top-level merge BUILDING marker must be an object: {path}")
    return value


def _unlink_merge_artifact(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Unsafe top-level merge recovery artifact: {path}")
    path.unlink()


def _prepare_merge_transaction(
    root: Path,
    identity: Mapping[str, Any],
) -> tuple[Path, dict[str, Any] | None]:
    """Recover only coordinator-owned top-level metadata; never traverse parts/."""

    marker = root / MERGE_BUILDING_FILENAME
    top_level = (
        root / MANIFEST_FILENAME,
        root / COMPLETE_FILENAME,
        root / "selection.jsonl",
    )
    if marker.exists():
        if marker.is_symlink() or not marker.is_file():
            raise RuntimeError(f"Unsafe top-level merge marker: {marker}")
        if _read_merge_marker(marker) != dict(identity):
            raise RuntimeError(
                "Top-level merge BUILDING identity differs; refusing recovery"
            )
        if (root / COMPLETE_FILENAME).exists():
            # Crash window: COMPLETE was durably written but the coordinator
            # had not yet removed MERGE.BUILDING.  Verify the immutable seal
            # and every identity bound by the marker, then finalize by removing
            # only the marker.  No cache metadata or part is rewritten.
            manifest = load_manifest(root, require_complete=True)
            if manifest["schema"].get("cache_kind") != identity["cache_kind"]:
                raise RuntimeError("Completed merge cache kind differs from marker")
            if manifest.get("partition", {}).get("work_plan_sha256") != identity[
                "work_plan_sha256"
            ]:
                raise RuntimeError("Completed merge work plan differs from marker")
            if manifest.get("producer") != identity["producer"]:
                raise RuntimeError("Completed merge producer differs from marker")
            actual_parts = [
                {
                    "partition_index": int(part["partition_index"]),
                    "path": str(part["path"]),
                    "manifest_sha256": str(part["manifest_sha256"]),
                }
                for part in manifest.get("parts", [])
            ]
            if actual_parts != list(identity["parts"]):
                raise RuntimeError("Completed merge parts differ from marker")
            actual_selection_plan = manifest["selection"].get("plan_identity")
            if isinstance(actual_selection_plan, Mapping):
                actual_selection_plan = {
                    key: value
                    for key, value in actual_selection_plan.items()
                    if key != "merged"
                }
            if actual_selection_plan != identity["selection_plan"]:
                raise RuntimeError("Completed merge selection plan differs from marker")
            marker.unlink()
            directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return marker, manifest
        for path in top_level:
            _unlink_merge_artifact(path)
        return marker, None

    existing = [str(path) for path in top_level if path.exists()]
    if existing:
        raise RuntimeError(
            "Top-level merge metadata exists without a matching coordinator marker: "
            f"{existing}"
        )
    write_immutable_file(marker, canonical_json_bytes(dict(identity)))
    return marker, None


def merge_part_manifests(
    part_roots: Sequence[str | Path],
    output_root: str | Path,
    *,
    verify_part_checksums: bool = False,
    canonical_root: str | Path | None = None,
    before_complete: Callable[[Path, Mapping[str, Any]], None] | None = None,
    after_complete: Callable[[Path, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Seal one readable top-level cache without copying any shard payload."""

    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Merged output root must already contain parts/: {root}")
    if not part_roots:
        raise ValueError("At least one part cache is required")

    loaded: list[tuple[int, Path, dict[str, Any]]] = []
    for value in part_roots:
        part_root = Path(value).expanduser().resolve()
        manifest = load_manifest(part_root, require_complete=True)
        partition = manifest.get("partition")
        if not isinstance(partition, Mapping):
            raise TypeError(f"Part cache lacks partition metadata: {part_root}")
        index = int(partition.get("partition_index", -1))
        _part_relative_path(part_root, root, index)
        loaded.append((index, part_root, manifest))
    loaded.sort(key=lambda item: item[0])

    indices = [index for index, _, _ in loaded]
    reference_partition = loaded[0][2]["partition"]
    partition_count = int(reference_partition["partition_count"])
    if indices != list(range(partition_count)):
        raise ValueError(
            "Part caches must cover every partition exactly once: "
            f"expected={list(range(partition_count))} actual={indices}"
        )

    reference = loaded[0][2]
    schema = reference["schema"]
    schema_object = GaussianCacheSchema.from_dict(schema)
    teacher_bytes = canonical_json_bytes(reference["teacher"])
    producer = reference.get("producer")
    producer_bytes = (
        None if producer is None else canonical_json_bytes(dict(producer))
    )
    derivation_bytes = (
        None if "derivation" not in reference else canonical_json_bytes(reference["derivation"])
    )
    target_shard_bytes = int(reference["target_shard_bytes"])
    selection_mode = str(reference["selection"]["mode"])
    reference_selection_plan = reference["selection"].get("plan_identity")
    selection_plan_global = (
        None
        if reference_selection_plan is None
        else _selection_plan_global(reference_selection_plan)
    )
    plan_fields = (
        "algorithm",
        "unit",
        "partition_count",
        "expected_unit_count",
        "expected_unit_weight",
        "work_plan_sha256",
    )
    unit = str(reference_partition["unit"])

    source_by_path: dict[str, dict[str, Any]] = {}
    all_assigned_units: list[dict[str, Any]] = []
    selection_keys = []
    shards: list[dict[str, Any]] = []
    streams: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    next_shard_id = 0

    for index, part_root, manifest in loaded:
        partition = manifest["partition"]
        if any(partition.get(field) != reference_partition.get(field) for field in plan_fields):
            raise ValueError(f"Partition plan metadata differs in part {index}")
        if manifest["schema"] != schema:
            raise ValueError(f"Gaussian schema differs in part {index}")
        if canonical_json_bytes(manifest["teacher"]) != teacher_bytes:
            raise ValueError(f"Teacher provenance differs in part {index}")
        part_producer = manifest.get("producer")
        part_producer_bytes = (
            None
            if part_producer is None
            else canonical_json_bytes(dict(part_producer))
        )
        if part_producer_bytes != producer_bytes:
            raise ValueError(f"FastWAM producer provenance differs in part {index}")
        part_derivation = (
            None if "derivation" not in manifest else canonical_json_bytes(manifest["derivation"])
        )
        if part_derivation != derivation_bytes:
            raise ValueError(f"Derivation provenance differs in part {index}")
        if int(manifest["target_shard_bytes"]) != target_shard_bytes:
            raise ValueError(f"target_shard_bytes differs in part {index}")
        if str(manifest["selection"]["mode"]) != selection_mode:
            raise ValueError(f"Selection mode differs in part {index}")
        part_selection_plan = manifest["selection"].get("plan_identity")
        if (part_selection_plan is None) != (selection_plan_global is None):
            raise ValueError(f"Selection plan provenance presence differs in part {index}")
        if selection_plan_global is not None:
            if not isinstance(part_selection_plan, Mapping):
                raise TypeError(f"Selection plan provenance is invalid in part {index}")
            if _selection_plan_global(part_selection_plan) != selection_plan_global:
                raise ValueError(f"Selection plan provenance differs in part {index}")

        assigned_units = _normalize_work_units(partition["assigned_units"], unit)
        if len(assigned_units) != int(partition["assigned_unit_count"]):
            raise ValueError(f"Assigned work-unit count mismatch in part {index}")
        if sum(int(item["weight"]) for item in assigned_units) != int(
            partition["assigned_unit_weight"]
        ):
            raise ValueError(f"Assigned work-unit weight mismatch in part {index}")
        all_assigned_units.extend(assigned_units)
        expected_sources = sorted({item["source_path"] for item in assigned_units})
        actual_sources = sorted(str(record["path"]) for record in manifest["sources"])
        if actual_sources != expected_sources:
            raise ValueError(f"Assigned work-unit sources differ from manifest in part {index}")
        for record in manifest["sources"]:
            identity = _source_identity(record)
            previous = source_by_path.get(identity["path"])
            if previous is not None and previous != identity:
                raise ValueError(
                    f"Shared source HDF5 provenance differs across parts: {identity['path']}"
                )
            source_by_path[identity["path"]] = identity

        part_selection_keys: list[FrameKey] | None = None
        if selection_mode == "index":
            selection = manifest["selection"]
            index_path = part_root / str(selection["index_filename"])
            if sha256_file(index_path) != str(selection["index_sha256"]):
                raise ValueError(f"Part selection index SHA-256 mismatch: {index_path}")
            part_selection_keys = load_selection_jsonl(index_path)
            if len(part_selection_keys) != int(selection["selected_key_count"]):
                raise ValueError(f"Part selection key count mismatch: {index_path}")
            if selection_plan_global is not None:
                plan_part = part_selection_plan["part"]
                if int(plan_part.get("part_index", -1)) != index:
                    raise ValueError(f"Selection plan part index differs in part {index}")
                if plan_part.get("index_sha256") != selection["index_sha256"]:
                    raise ValueError(f"Selection plan key SHA-256 differs in part {index}")
                if int(plan_part.get("selected_key_count", -1)) != int(
                    selection["selected_key_count"]
                ):
                    raise ValueError(f"Selection plan key count differs in part {index}")

        actual_stream_units = {
            (
                (str(stream["source_path"]), str(stream["trajectory"]))
                if unit == "trajectory"
                else (str(stream["source_path"]),)
            )
            for stream in manifest["streams"]
        }
        assigned_unit_keys = {_work_unit_key(item, unit) for item in assigned_units}
        if not actual_stream_units <= assigned_unit_keys:
            raise ValueError(f"Part {index} contains streams outside its assigned work units")
        if selection_mode == "all" and actual_stream_units != assigned_unit_keys:
            raise ValueError(f"Full cache part {index} is missing assigned work-unit streams")
        if unit == "trajectory":
            validate_partition_coverage(
                manifest,
                selection_keys=part_selection_keys,
                context=f"part {index}",
            )

        relative_part = _part_relative_path(part_root, root, index)
        id_map: dict[str, str] = {}
        for shard in manifest["shards"]:
            local_path = normalize_source_path(str(shard["path"]))
            if not local_path.startswith("shards/"):
                raise ValueError(f"Part shard is not local to shards/: {local_path}")
            final_path = part_root / local_path
            if not final_path.is_file() or final_path.stat().st_size != int(shard["bytes"]):
                raise ValueError(f"Part shard missing or byte count mismatch: {final_path}")
            if verify_part_checksums and sha256_file(final_path) != str(shard["sha256"]):
                raise ValueError(f"Part shard SHA-256 mismatch: {final_path}")
            new_id = f"{next_shard_id:06d}"
            next_shard_id += 1
            id_map[str(shard["id"])] = new_id
            shards.append(
                {
                    **shard,
                    "id": new_id,
                    "path": f"{relative_part}/{local_path}",
                    "part_index": index,
                }
            )

        for stream in manifest["streams"]:
            streams.append(
                {
                    **stream,
                    "part_index": index,
                    "segments": [
                        {**segment, "shard": id_map[str(segment["shard"])]}
                        for segment in stream["segments"]
                    ],
                }
            )

        if selection_mode == "index":
            assert part_selection_keys is not None
            selection_keys.extend(part_selection_keys)

        parts.append(
            {
                "partition_index": index,
                "path": relative_part,
                "manifest_sha256": sha256_file(part_root / MANIFEST_FILENAME),
                "source_count": len(manifest["sources"]),
                "shard_count": len(manifest["shards"]),
                "stream_count": len(manifest["streams"]),
                "total_frames": int(manifest["total_frames"]),
            }
        )

    normalized_units = _normalize_work_units(all_assigned_units, unit)
    if len(normalized_units) != int(reference_partition["expected_unit_count"]):
        raise ValueError(
            "Merged work units are incomplete: "
            f"expected={reference_partition['expected_unit_count']} actual={len(normalized_units)}"
        )
    if sum(int(item["weight"]) for item in normalized_units) != int(
        reference_partition["expected_unit_weight"]
    ):
        raise ValueError("Merged work-unit weight differs from the partition plan")
    if work_plan_sha256(normalized_units, partition_count, unit=unit) != str(
        reference_partition["work_plan_sha256"]
    ):
        raise ValueError("Merged work units do not match the sealed partition plan")
    expected_partitions = partition_work_units(normalized_units, partition_count, unit=unit)
    for (index, _, manifest), expected in zip(loaded, expected_partitions):
        actual = _normalize_work_units(manifest["partition"]["assigned_units"], unit)
        if actual != expected:
            raise ValueError(f"Part {index} does not follow deterministic LPT balancing")

    sources = sorted(source_by_path.values(), key=lambda record: record["path"])
    unit_sources = {item["source_path"] for item in normalized_units}
    if set(source_by_path) != unit_sources:
        raise ValueError("Merged source records do not cover the partition work plan")

    transaction_identity = _merge_transaction_identity(
        root=root,
        schema=schema,
        partition=reference_partition,
        loaded=loaded,
        producer=producer,
        selection_plan=selection_plan_global,
        canonical_root=canonical_root,
    )
    merge_marker, recovered_manifest = _prepare_merge_transaction(
        root,
        transaction_identity,
    )
    if recovered_manifest is not None:
        return recovered_manifest

    if selection_mode == "index":
        if len(set(selection_keys)) != len(selection_keys):
            raise ValueError("Sparse selection keys overlap across part caches")
        selection = write_normalized_selection_index(root, selection_keys)
        if selection_plan_global is not None:
            planned_identity = selection_plan_global.get("planned_normalized")
            if not isinstance(planned_identity, Mapping):
                raise ValueError("Selection plan lacks planned_normalized identity")
            if selection["index_sha256"] != planned_identity.get("index_sha256"):
                raise ValueError(
                    "Merged compact selection key set differs from the sealed plan"
                )
            if int(selection["selected_key_count"]) != int(
                planned_identity.get("selected_key_count", -1)
            ):
                raise ValueError(
                    "Merged compact selection key count differs from the sealed plan"
                )
            selection["plan_identity"] = {
                **selection_plan_global,
                "merged": True,
            }
    else:
        if selection_plan_global is not None:
            raise ValueError("selection=all must not carry sparse selection-plan identity")
        selection = {
            "mode": "all",
            "selected_key_count": sum(
                int(manifest["total_frames"]) for _, _, manifest in loaded
            ),
        }

    derivation = reference.get("derivation")
    if schema_object.cache_kind == "compact":
        if canonical_root is None:
            raise ValueError("Merging compact parts requires canonical_root")
        canonical_path = Path(canonical_root).expanduser().resolve()
        canonical = load_manifest(canonical_path, require_complete=True)
        canonical_schema = GaussianCacheSchema.from_dict(canonical["schema"])
        if canonical_schema.cache_kind != "canonical":
            raise ValueError("canonical_root is not a canonical Gaussian cache")
        if canonical_json_bytes(canonical["teacher"]) != teacher_bytes:
            raise ValueError("Compact/canonical teacher provenance differs")
        canonical_producer = canonical.get("producer")
        canonical_producer_bytes = (
            None
            if canonical_producer is None
            else canonical_json_bytes(dict(canonical_producer))
        )
        if canonical_producer_bytes != producer_bytes:
            raise ValueError("Compact/canonical FastWAM producer provenance differs")
        if canonical.get("partition", {}).get("work_plan_sha256") != reference_partition.get(
            "work_plan_sha256"
        ):
            raise ValueError("Compact/canonical partition plans differ")
        if _normalized_sources(canonical["sources"]) != _normalized_sources(sources):
            raise ValueError("Compact/canonical source provenance differs")
        derivation = {
            **(dict(derivation) if isinstance(derivation, Mapping) else {}),
            "parent_manifest_sha256": sha256_file(canonical_path / MANIFEST_FILENAME),
            "parent_cache_kind": "canonical",
            "parent_total_frames": int(canonical["total_frames"]),
            "parent_teacher": canonical["teacher"],
            "parent_selection": canonical["selection"],
        }
    elif canonical_root is not None:
        raise ValueError("canonical_root is only valid while merging compact parts")

    manifest: dict[str, Any] = {
        "manifest_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema": schema,
        "target_shard_bytes": target_shard_bytes,
        "selection": selection,
        "teacher": reference["teacher"],
        "sources": sources,
        "shards": shards,
        "streams": sorted(
            streams,
            key=lambda stream: (
                str(stream["source_path"]),
                str(stream["trajectory"]),
                str(stream["agent_name"]),
            ),
        ),
        "total_frames": sum(int(manifest["total_frames"]) for _, _, manifest in loaded),
        "partition": {
            "algorithm": str(reference_partition["algorithm"]),
            "unit": unit,
            "partition_count": partition_count,
            "expected_unit_count": int(reference_partition["expected_unit_count"]),
            "expected_unit_weight": int(reference_partition["expected_unit_weight"]),
            "work_plan_sha256": str(reference_partition["work_plan_sha256"]),
            "merged": True,
        },
        "parts": parts,
    }
    if producer is not None:
        manifest["producer"] = producer
    if derivation is not None:
        manifest["derivation"] = derivation
    complete = seal_manifest(root, manifest, before_complete=before_complete)
    if after_complete is not None:
        after_complete(root, complete)
    if _read_merge_marker(merge_marker) != transaction_identity:
        raise RuntimeError("Top-level merge BUILDING marker changed before finalization")
    merge_marker.unlink()
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return manifest
