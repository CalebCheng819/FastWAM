"""CLI and library entry points for canonical/compact Gaussian cache creation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from .compact import (
    COMPACT_HEIGHT,
    COMPACT_WIDTH,
    MOMENT_MATCH_METHOD,
    opacity_aware_moment_match,
    project_compact_cache,
)
from .distributed import merge_part_manifests, partition_work_metadata
from .manifest import (
    DEFAULT_TARGET_SHARD_BYTES,
    GaussianCacheBuilder,
    source_record,
)
from .schema import FrameKey, GaussianCacheSchema
from .selection import load_selection_jsonl, write_normalized_selection_index
from .teacher import ExternalPolicyLightningTeacher, GaussianTeacher


def _agent_sort_key(name: str):
    try:
        return int(name.rsplit("-", 1)[-1])
    except ValueError:
        return name


def _camera_name(agent_name: str) -> str:
    _, separator, suffix = str(agent_name).rpartition("-")
    if not separator or not suffix.isdigit():
        raise ValueError(f"Expected RoboFactory agent name ending in an integer, got {agent_name!r}")
    return f"head_camera_agent{int(suffix)}"


def _selected_by_frame(
    keys: Sequence[FrameKey],
) -> dict[tuple[str, str], dict[int, set[str]]]:
    selected: dict[tuple[str, str], dict[int, set[str]]] = {}
    for key in keys:
        selected.setdefault((key.source_path, key.trajectory), {}).setdefault(
            key.timestep, set()
        ).add(key.agent_name)
    return selected


def _discover_trajectory_work_units(
    dataset_root: Path,
    sources: Sequence[Mapping[str, Any]],
    *,
    include: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for source in sources:
        source_path = str(source["path"])
        with h5py.File(dataset_root / source_path, "r") as handle:
            for trajectory_name in sorted(handle.keys()):
                if include is not None and (source_path, trajectory_name) not in include:
                    continue
                trajectory = handle[trajectory_name]
                if "actions" not in trajectory:
                    continue
                agent_names = sorted(trajectory["actions"].keys(), key=_agent_sort_key)
                if not agent_names:
                    continue
                camera_paths = ["obs/sensor_data/head_camera_global/rgb"] + [
                    f"obs/sensor_data/{_camera_name(agent_name)}/rgb"
                    for agent_name in agent_names
                ]
                missing = [path for path in camera_paths if path not in trajectory]
                if missing:
                    raise KeyError(
                        f"Missing trajectory partition cameras in {source_path}:{trajectory_name}: "
                        f"{missing}"
                    )
                counts = {int(trajectory[path].shape[0]) for path in camera_paths}
                if len(counts) != 1:
                    raise ValueError(
                        f"Global/agent observation lengths differ at "
                        f"{source_path}:{trajectory_name}: {sorted(counts)}"
                    )
                observation_count = counts.pop()
                units.append(
                    {
                        "source_path": source_path,
                        "trajectory": trajectory_name,
                        "observation_count": observation_count,
                        "agent_names": list(agent_names),
                        "weight": observation_count * len(agent_names),
                    }
                )
    return units


def _read_global_agent_pairs(
    trajectory: h5py.Group,
    agent_names: Sequence[str],
    timesteps: Sequence[int],
) -> torch.Tensor:
    """Return ``[B*N,2,3,H,W]`` pairs in exact requested agent order."""

    global_path = "obs/sensor_data/head_camera_global/rgb"
    global_rgb = np.asarray(trajectory[global_path][list(timesteps)], dtype=np.uint8)
    agent_views = []
    for agent_name in agent_names:
        path = f"obs/sensor_data/{_camera_name(agent_name)}/rgb"
        rgb = np.asarray(trajectory[path][list(timesteps)], dtype=np.uint8)
        agent_views.append(rgb)
    agents = torch.from_numpy(np.stack(agent_views, axis=1)).permute(0, 1, 4, 2, 3)
    global_view = torch.from_numpy(global_rgb).permute(0, 3, 1, 2)
    batch, agent_count = agents.shape[:2]
    pairs = torch.stack(
        (
            global_view[:, None].expand(-1, agent_count, -1, -1, -1),
            agents,
        ),
        dim=2,
    ).reshape(batch * agent_count, 2, *agents.shape[2:])
    return pairs.float().div(127.5).sub(1.0).contiguous()


def extract_canonical_cache(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    teacher: GaussianTeacher,
    selection: str = "all",
    selection_jsonl: str | Path | None = None,
    selection_keys: Sequence[FrameKey] | None = None,
    batch_size: int = 8,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    staging_dir: str | Path | None = None,
    verify_uploaded_checksum: bool = True,
    partition_index: int = 0,
    partition_count: int = 1,
    partition_unit: str = "trajectory",
    compact_output_root: str | Path | None = None,
    compact_selection_jsonl: str | Path | None = None,
    compact_selection_keys: Sequence[FrameKey] | None = None,
    compact_target_shard_bytes: int | None = None,
    work_plan: Mapping[str, Any] | None = None,
    micro_part_index: int | None = None,
    preverified_source_state: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Extract corrected FP16 ``[means3,cov9,opacity1]`` for every selected view."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"RoboFactory dataset root is missing: {dataset_root}")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    selection = str(selection).lower()
    partition_unit = str(partition_unit).lower()
    if partition_unit not in {"source", "trajectory"}:
        raise ValueError("partition_unit must be 'source' or 'trajectory'")
    if selection_jsonl is not None and selection_keys is not None:
        raise ValueError("Supply selection_jsonl or selection_keys, not both")
    if compact_selection_jsonl is not None and compact_selection_keys is not None:
        raise ValueError(
            "Supply compact_selection_jsonl or compact_selection_keys, not both"
        )
    compact_selector_supplied = (
        compact_selection_jsonl is not None or compact_selection_keys is not None
    )
    dual_compact = compact_output_root is not None or compact_selector_supplied
    if dual_compact and (compact_output_root is None or not compact_selector_supplied):
        raise ValueError(
            "compact_output_root and exactly one compact selection input must be supplied together"
        )
    if dual_compact and selection != "all":
        raise ValueError("Same-forward compact extraction requires canonical selection='all'")
    plan_micro_part: Mapping[str, Any] | None = None
    if (work_plan is None) != (micro_part_index is None):
        raise ValueError("work_plan and micro_part_index must be supplied together")
    if work_plan is not None:
        from .plan import (
            compact_selection_part_identity,
            micro_part_partition_metadata,
            source_identity_by_path,
            stable_file_identity,
            validate_work_plan,
        )

        if partition_unit != "trajectory":
            raise ValueError("Sealed micro-part extraction requires partition_unit='trajectory'")
        if int(partition_index) != 0 or int(partition_count) != 1:
            raise ValueError(
                "partition_index/count are replaced by micro_part_index when work_plan is supplied"
            )
        validate_work_plan(work_plan)
        assert micro_part_index is not None
        index = int(micro_part_index)
        micro_parts = work_plan["micro_parts"]
        if not 0 <= index < len(micro_parts):
            raise ValueError(
                f"micro_part_index must be in [0,{len(micro_parts)}), got {index}"
            )
        plan_micro_part = micro_parts[index]
        source_identities = source_identity_by_path(work_plan)
        planned_source = source_identities[str(plan_micro_part["source_path"])]
        assigned_path = dataset_root / str(planned_source["path"])
        if preverified_source_state is None:
            actual_source = stable_file_identity(
                assigned_path,
                relative_to=dataset_root,
                expected_sha256=str(planned_source["sha256"]),
            )
            actual_state = (
                int(actual_source["bytes"]),
                int(actual_source["mtime_ns"]),
            )
        else:
            actual_stat = assigned_path.stat()
            actual_state = (int(actual_stat.st_size), int(actual_stat.st_mtime_ns))
            if actual_state != tuple(map(int, preverified_source_state)):
                raise RuntimeError(
                    "Assigned source changed after worker-level SHA verification: "
                    f"{planned_source['path']}"
                )
        planned_state = (
            int(planned_source["bytes"]),
            int(planned_source["mtime_ns"]),
        )
        if actual_state != planned_state:
            raise RuntimeError(
                "Assigned source stat identity changed after coordinator sealing: "
                f"{planned_source['path']}"
            )
        all_hdf5_paths = [assigned_path]
        all_source_records = [
            {
                "path": str(planned_source["path"]),
                "bytes": int(planned_source["bytes"]),
                "sha256": str(planned_source["sha256"]),
            }
        ]
        known_plan_sources = set(source_identities)
        work_units = [
            {
                "source_path": str(item["source_path"]),
                "trajectory": str(item["trajectory"]),
                "observation_count": int(item["observation_count"]),
                "agent_names": [str(name) for name in item["agent_names"]],
                "weight": int(item["weight"]),
            }
            for item in work_plan["micro_parts"]
        ]
        assigned_units = [
            {
                "source_path": str(plan_micro_part["source_path"]),
                "trajectory": str(plan_micro_part["trajectory"]),
                "observation_count": int(plan_micro_part["observation_count"]),
                "agent_names": [str(name) for name in plan_micro_part["agent_names"]],
                "weight": int(plan_micro_part["weight"]),
            }
        ]
        partition = micro_part_partition_metadata(work_plan, plan_micro_part)
    else:
        if preverified_source_state is not None:
            raise ValueError("preverified_source_state requires a sealed work_plan")
        all_hdf5_paths = sorted(dataset_root.rglob("*.h5"))
        if not all_hdf5_paths:
            raise FileNotFoundError(f"No .h5 files found under {dataset_root}")
        all_source_records = [
            source_record(path, source_root=dataset_root) for path in all_hdf5_paths
        ]
        known_plan_sources = {str(record["path"]) for record in all_source_records}

    if selection == "all":
        if selection_jsonl is not None or selection_keys is not None:
            raise ValueError("Sparse selection inputs are only valid with selection='index'")
        selected_keys: list[FrameKey] = []
        selected_frames = None
        remaining = None
    elif selection == "index" and (selection_jsonl is not None or selection_keys is not None):
        all_selected_keys = (
            load_selection_jsonl(selection_jsonl)
            if selection_jsonl is not None
            else list(selection_keys or ())
        )
        if not all_selected_keys or len(set(all_selected_keys)) != len(all_selected_keys):
            raise ValueError("selection_keys must be non-empty and duplicate-free")
    else:
        raise ValueError(
            "selection must be 'all', or 'index' with selection_jsonl/selection_keys"
        )

    if selection == "index":
        requested_sources = {key.source_path for key in all_selected_keys}
        missing_sources = sorted(requested_sources - known_plan_sources)
        if missing_sources:
            raise KeyError(
                "Selection references source HDF5 files absent from dataset_root: "
                f"{missing_sources[:16]}"
            )
        plan_sources = [
            record for record in all_source_records if str(record["path"]) in requested_sources
        ]
    else:
        plan_sources = all_source_records
    if work_plan is not None:
        pass
    elif partition_unit == "source":
        work_units = [
            {"source_path": record["path"], "weight": int(record["bytes"])}
            for record in plan_sources
        ]
    else:
        include_trajectories = (
            None
            if selection == "all"
            else {(key.source_path, key.trajectory) for key in all_selected_keys}
        )
        work_units = _discover_trajectory_work_units(
            dataset_root,
            plan_sources,
            include=include_trajectories,
        )
        if include_trajectories is not None:
            discovered = {
                (str(unit["source_path"]), str(unit["trajectory"])) for unit in work_units
            }
            missing_trajectories = sorted(include_trajectories - discovered)
            if missing_trajectories:
                raise KeyError(
                    "Selection references trajectories absent from dataset_root: "
                    f"{missing_trajectories[:16]}"
                )
    if work_plan is None:
        assigned_units, partition = partition_work_metadata(
            work_units,
            partition_index=int(partition_index),
            partition_count=int(partition_count),
            unit=partition_unit,
        )
    assigned_source_paths = {str(unit["source_path"]) for unit in assigned_units}
    source_records = [
        record for record in plan_sources if str(record["path"]) in assigned_source_paths
    ]
    assigned_trajectories = (
        None
        if partition_unit == "source"
        else {
            (str(unit["source_path"]), str(unit["trajectory"])) for unit in assigned_units
        }
    )
    hdf5_paths = [dataset_root / str(record["path"]) for record in source_records]
    if selection == "index":
        selected_keys = [
            key
            for key in all_selected_keys
            if key.source_path in assigned_source_paths
            and (
                assigned_trajectories is None
                or (key.source_path, key.trajectory) in assigned_trajectories
            )
        ]
        selected_frames = _selected_by_frame(selected_keys)
        remaining = set(selected_keys)
    else:
        selected_keys = []
        selected_frames = None
        remaining = None

    compact_keys: list[FrameKey] = []
    compact_frames = None
    compact_remaining: set[FrameKey] | None = None
    if dual_compact:
        all_compact_keys = (
            load_selection_jsonl(compact_selection_jsonl)
            if compact_selection_jsonl is not None
            else list(compact_selection_keys or ())
        )
        if not all_compact_keys or len(set(all_compact_keys)) != len(all_compact_keys):
            raise ValueError("compact_selection_keys must be non-empty and duplicate-free")
        missing_compact_sources = sorted(
            {key.source_path for key in all_compact_keys} - known_plan_sources
        )
        if missing_compact_sources:
            raise KeyError(
                "Compact selection references missing sources: "
                f"{missing_compact_sources[:16]}"
            )
        if partition_unit == "trajectory":
            known_trajectories = {
                (str(unit["source_path"]), str(unit["trajectory"])) for unit in work_units
            }
            requested_trajectories = {
                (key.source_path, key.trajectory) for key in all_compact_keys
            }
            missing_compact_trajectories = sorted(
                requested_trajectories - known_trajectories
            )
            if missing_compact_trajectories:
                raise KeyError(
                    "Compact selection references missing trajectories: "
                    f"{missing_compact_trajectories[:16]}"
                )
        compact_keys = [
            key
            for key in all_compact_keys
            if key.source_path in assigned_source_paths
            and (
                assigned_trajectories is None
                or (key.source_path, key.trajectory) in assigned_trajectories
            )
        ]
        if not compact_keys:
            active_partition_index = (
                int(micro_part_index)
                if micro_part_index is not None
                else int(partition_index)
            )
            raise ValueError(
                f"Partition {active_partition_index} has no compact selection keys"
            )
        compact_frames = _selected_by_frame(compact_keys)
        compact_remaining = set(compact_keys)

    source_state = {
        record["path"]: (
            (dataset_root / record["path"]).stat().st_size,
            (dataset_root / record["path"]).stat().st_mtime_ns,
        )
        for record in source_records
    }
    teacher_provenance = dict(teacher.provenance())
    if work_plan is not None:
        planned_checkpoint_sha = str(work_plan["checkpoint"]["sha256"])
        if str(teacher_provenance.get("checkpoint_sha256", "")) != planned_checkpoint_sha:
            raise ValueError(
                "Teacher checkpoint identity differs from sealed work plan: "
                f"expected={planned_checkpoint_sha} "
                f"actual={teacher_provenance.get('checkpoint_sha256')}"
            )
        planned_teacher = work_plan.get("teacher")
        if not isinstance(planned_teacher, Mapping):
            raise TypeError("Sealed work plan lacks teacher provenance")
        for key, expected in dict(planned_teacher).items():
            if key == "training_data_provenance":
                if key in teacher_provenance and teacher_provenance[key] != expected:
                    raise ValueError(
                        f"Teacher provenance field {key!r} differs from sealed work plan"
                    )
                # The training-set declaration is sealed by the coordinator;
                # unlike runtime Git/config identity it is not discoverable
                # from the instantiated encoder.
                teacher_provenance[key] = expected
                continue
            if key not in teacher_provenance:
                raise ValueError(
                    f"Teacher provenance field {key!r} is absent from runtime teacher"
                )
            if teacher_provenance[key] != expected:
                raise ValueError(
                    f"Teacher provenance field {key!r} differs from sealed work plan"
                )
    config_overrides = dict(teacher_provenance.get("config_overrides", {}))
    config_overrides["coor_type"] = "unify"
    teacher_provenance.update(
        {
            "pairing": "global_agent_unify_v1",
            "input_views_per_pair": 2,
            "cached_pair_view": "agent",
            "config_overrides": config_overrides,
        }
    )
    builder = GaussianCacheBuilder(
        output_root,
        GaussianCacheSchema(height=240, width=320, cache_kind="canonical"),
        sources=source_records,
        teacher=teacher_provenance,
        producer=(None if work_plan is None else work_plan["producer"]),
        selection={
            "mode": selection,
            "selected_key_count": None if selection == "all" else len(selected_keys),
        },
        partition=partition,
        target_shard_bytes=target_shard_bytes,
        staging_dir=staging_dir,
        verify_uploaded_checksum=verify_uploaded_checksum,
    )
    if selection == "index":
        builder.selection = write_normalized_selection_index(output_root, selected_keys)

    compact_builder: GaussianCacheBuilder | None = None
    if dual_compact:
        assert compact_output_root is not None
        compact_builder = GaussianCacheBuilder(
            compact_output_root,
            GaussianCacheSchema(
                height=COMPACT_HEIGHT,
                width=COMPACT_WIDTH,
                cache_kind="compact",
            ),
            sources=source_records,
            teacher=teacher_provenance,
            producer=(None if work_plan is None else work_plan["producer"]),
            selection={"mode": "index", "selected_key_count": len(compact_keys)},
            derivation={
                "method": MOMENT_MATCH_METHOD,
                "output_size": [COMPACT_HEIGHT, COMPACT_WIDTH],
                "source": "same-teacher-forward-canonical-v1",
                "canonical_work_plan_sha256": partition["work_plan_sha256"],
            },
            partition=partition,
            target_shard_bytes=(
                target_shard_bytes
                if compact_target_shard_bytes is None
                else int(compact_target_shard_bytes)
            ),
            staging_dir=staging_dir,
            verify_uploaded_checksum=verify_uploaded_checksum,
        )
        compact_builder.selection = write_normalized_selection_index(
            compact_output_root,
            compact_keys,
        )
        if work_plan is not None:
            assert plan_micro_part is not None
            compact_builder.selection["plan_identity"] = compact_selection_part_identity(
                work_plan,
                plan_micro_part,
            )

    written_count = 0
    compact_written_count = 0
    try:
        for hdf5_path in hdf5_paths:
            source_path = hdf5_path.relative_to(dataset_root).as_posix()
            with h5py.File(hdf5_path, "r") as handle:
                for trajectory_name in sorted(handle.keys()):
                    if (
                        assigned_trajectories is not None
                        and (source_path, trajectory_name) not in assigned_trajectories
                    ):
                        continue
                    trajectory = handle[trajectory_name]
                    if "actions" not in trajectory:
                        continue
                    agent_names = sorted(trajectory["actions"].keys(), key=_agent_sort_key)
                    if not agent_names:
                        continue
                    camera_paths = ["obs/sensor_data/head_camera_global/rgb"] + [
                        f"obs/sensor_data/{_camera_name(agent_name)}/rgb"
                        for agent_name in agent_names
                    ]
                    camera_datasets = []
                    for camera_path in camera_paths:
                        if camera_path not in trajectory:
                            raise KeyError(
                                f"Missing teacher RGB {camera_path!r} in {source_path}:{trajectory_name}"
                            )
                        camera = trajectory[camera_path]
                        if camera.ndim != 4 or tuple(camera.shape[1:]) != (240, 320, 3):
                            raise ValueError(
                                f"Canonical extraction requires RGB [T,240,320,3], got "
                                f"{tuple(camera.shape)} at {source_path}:{trajectory_name}:{camera_path}"
                            )
                        if camera.dtype != np.dtype("uint8"):
                            raise ValueError(f"Canonical extraction requires uint8 RGB at {camera_path}")
                        camera_datasets.append(camera)
                    observation_counts = {int(camera.shape[0]) for camera in camera_datasets}
                    if len(observation_counts) != 1:
                        raise ValueError(
                            f"Global/agent observation lengths differ at {source_path}:{trajectory_name}: "
                            f"{sorted(observation_counts)}"
                        )
                    observation_count = observation_counts.pop()
                    if selection == "all":
                        timesteps = list(range(observation_count))
                    else:
                        assert selected_frames is not None
                        trajectory_selection = selected_frames.get(
                            (source_path, trajectory_name), {}
                        )
                        timesteps = sorted(trajectory_selection)
                    trajectory_compact_selection = (
                        {}
                        if compact_frames is None
                        else compact_frames.get((source_path, trajectory_name), {})
                    )
                    for start in range(0, len(timesteps), int(batch_size)):
                        batch_timesteps = timesteps[start : start + int(batch_size)]
                        images = _read_global_agent_pairs(
                            trajectory,
                            agent_names,
                            batch_timesteps,
                        )
                        pair_gaussian = teacher.encode(images)
                        expected_pair_shape = (
                            len(batch_timesteps) * len(agent_names),
                            2,
                            13,
                            240,
                            320,
                        )
                        if (
                            tuple(pair_gaussian.shape) != expected_pair_shape
                            or pair_gaussian.dtype != torch.float16
                        ):
                            raise ValueError(
                                f"Teacher output must be FP16 {expected_pair_shape}, got "
                                f"shape={tuple(pair_gaussian.shape)} dtype={pair_gaussian.dtype}"
                            )
                        # Each flattened sample is exactly [global, agent_i].
                        # The global prediction is reference-only and is never
                        # stored as an agent cache frame.
                        gaussian = pair_gaussian[:, 1].reshape(
                            len(batch_timesteps),
                            len(agent_names),
                            13,
                            240,
                            320,
                        )
                        for agent_index, agent_name in enumerate(agent_names):
                            if selection == "all":
                                positions = list(range(len(batch_timesteps)))
                            else:
                                assert selected_frames is not None
                                positions = [
                                    position
                                    for position, timestep in enumerate(batch_timesteps)
                                    if agent_name
                                    in trajectory_selection.get(timestep, set())
                                ]
                            if not positions:
                                continue
                            agent_timesteps = [batch_timesteps[position] for position in positions]
                            builder.append_stream(
                                source_path=source_path,
                                trajectory=trajectory_name,
                                agent_name=agent_name,
                                observation_count=observation_count,
                                timesteps=agent_timesteps,
                                frames=gaussian[positions, agent_index],
                            )
                            written_count += len(positions)
                            if remaining is not None:
                                for timestep in agent_timesteps:
                                    remaining.discard(
                                        FrameKey(
                                            source_path,
                                            trajectory_name,
                                            timestep,
                                            agent_name,
                                        )
                                    )
                            if compact_builder is not None:
                                compact_positions = [
                                    position
                                    for position, timestep in enumerate(batch_timesteps)
                                    if agent_name
                                    in trajectory_compact_selection.get(timestep, set())
                                ]
                                if compact_positions:
                                    compact_timesteps = [
                                        batch_timesteps[position]
                                        for position in compact_positions
                                    ]
                                    compact_batch = opacity_aware_moment_match(
                                        gaussian[compact_positions, agent_index]
                                    )
                                    compact_builder.append_stream(
                                        source_path=source_path,
                                        trajectory=trajectory_name,
                                        agent_name=agent_name,
                                        observation_count=observation_count,
                                        timesteps=compact_timesteps,
                                        frames=compact_batch,
                                    )
                                    compact_written_count += len(compact_positions)
                                    assert compact_remaining is not None
                                    for timestep in compact_timesteps:
                                        compact_remaining.discard(
                                            FrameKey(
                                                source_path,
                                                trajectory_name,
                                                timestep,
                                                agent_name,
                                            )
                                        )

        if remaining:
            raise KeyError(
                "Selection contains keys absent from RoboFactory data: "
                f"missing={len(remaining)}, sample={[key.to_dict() for key in sorted(remaining)[:16]]}"
            )
        if compact_remaining:
            raise KeyError(
                "Compact selection contains keys absent from assigned RoboFactory data: "
                f"missing={len(compact_remaining)}, "
                f"sample={[key.to_dict() for key in sorted(compact_remaining)[:16]]}"
            )
        for record in source_records:
            path = dataset_root / record["path"]
            current = (path.stat().st_size, path.stat().st_mtime_ns)
            if current != source_state[record["path"]]:
                raise RuntimeError(f"Source HDF5 changed during extraction: {path}")
        builder.selection["selected_key_count"] = written_count
        manifest = builder.finish()
        if int(manifest["total_frames"]) != written_count:
            raise RuntimeError("Canonical extraction frame accounting mismatch")
        if compact_builder is not None:
            compact_builder.selection["selected_key_count"] = compact_written_count
            compact_manifest = compact_builder.finish()
            if int(compact_manifest["total_frames"]) != compact_written_count:
                raise RuntimeError("Compact extraction frame accounting mismatch")
        return manifest
    except Exception:
        builder.abort()
        if compact_builder is not None:
            compact_builder.abort()
        raise


def _target_bytes(value: float) -> int:
    return int(float(value) * (1 << 30))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    canonical = subparsers.add_parser("canonical", help="extract full-resolution 13ch cache")
    canonical.add_argument("--dataset-root", required=True)
    canonical.add_argument("--output-root", required=True)
    canonical.add_argument("--teacher-repo", required=True)
    canonical.add_argument("--teacher-commit", required=True)
    canonical.add_argument("--teacher-checkpoint", required=True)
    canonical.add_argument("--teacher-checkpoint-sha256", required=True)
    canonical.add_argument(
        "--teacher-config",
        default="config/encoder/noposplat.yaml",
    )
    canonical.add_argument("--device", default="cuda")
    canonical.add_argument("--batch-size", type=int, default=8)
    canonical.add_argument("--target-shard-gib", type=float, default=2.0)
    canonical.add_argument(
        "--staging-dir",
        help="local/CPFS staging base; one shard is uploaded to output at a time",
    )
    canonical.add_argument(
        "--verify-uploaded-checksum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "read back and hash every uploaded shard (formal default); "
            "--no-verify-uploaded-checksum is diagnostic-only"
        ),
    )
    canonical.add_argument("--partition-index", type=int, default=0)
    canonical.add_argument("--partition-count", type=int, default=1)
    canonical.add_argument(
        "--partition-unit",
        choices=("source", "trajectory"),
        default="trajectory",
    )
    canonical.add_argument(
        "--work-plan-root",
        help="sealed coordinator work plan; requires --micro-part-index",
    )
    canonical.add_argument(
        "--micro-part-index",
        type=int,
        help="extract exactly one trajectory from --work-plan-root",
    )
    canonical.add_argument("--compact-output-root")
    canonical.add_argument("--compact-selection-jsonl")
    canonical.add_argument("--compact-target-shard-gib", type=float, default=2.0)
    canonical.add_argument("--selection", choices=("all", "index"), default="all")
    canonical.add_argument("--selection-jsonl")

    compact = subparsers.add_parser("compact", help="derive opacity-aware 28x40 cache")
    compact.add_argument("--canonical-root", required=True)
    compact.add_argument("--output-root", required=True)
    compact.add_argument("--selection", choices=("all", "index"), default="index")
    compact.add_argument("--selection-jsonl")
    compact.add_argument("--verify", choices=("none", "manifest", "checksums"), default="manifest")
    compact.add_argument("--batch-size", type=int, default=8)
    compact.add_argument("--target-shard-gib", type=float, default=2.0)
    compact.add_argument("--staging-dir")
    compact.add_argument(
        "--verify-uploaded-checksum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "read back and hash every uploaded shard (formal default); "
            "--no-verify-uploaded-checksum is diagnostic-only"
        ),
    )

    merge = subparsers.add_parser(
        "merge-part-manifests",
        help="seal a top-level zero-copy cache from parts/part-* caches",
    )
    merge.add_argument("--parts-root", required=True)
    merge.add_argument("--output-root", required=True)
    merge.add_argument("--verify-part-checksums", action="store_true")
    merge.add_argument(
        "--canonical-root",
        help="required when merging compact parts; pins the final canonical manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "canonical":
        if (args.work_plan_root is None) != (args.micro_part_index is None):
            raise ValueError(
                "--work-plan-root and --micro-part-index must be supplied together"
            )
        if args.work_plan_root is None:
            work_plan = None
        else:
            from .plan import load_work_plan

            work_plan = load_work_plan(args.work_plan_root)
        teacher = ExternalPolicyLightningTeacher(
            repo_path=args.teacher_repo,
            expected_commit=args.teacher_commit,
            checkpoint_path=args.teacher_checkpoint,
            checkpoint_sha256=args.teacher_checkpoint_sha256,
            config_path=args.teacher_config,
            device=args.device,
        )
        manifest = extract_canonical_cache(
            args.dataset_root,
            args.output_root,
            teacher=teacher,
            selection=args.selection,
            selection_jsonl=args.selection_jsonl,
            batch_size=args.batch_size,
            target_shard_bytes=_target_bytes(args.target_shard_gib),
            staging_dir=args.staging_dir,
            verify_uploaded_checksum=args.verify_uploaded_checksum,
            partition_index=args.partition_index,
            partition_count=args.partition_count,
            partition_unit=args.partition_unit,
            compact_output_root=args.compact_output_root,
            compact_selection_jsonl=args.compact_selection_jsonl,
            compact_target_shard_bytes=_target_bytes(args.compact_target_shard_gib),
            work_plan=work_plan,
            micro_part_index=args.micro_part_index,
        )
    elif args.command == "compact":
        manifest = project_compact_cache(
            args.canonical_root,
            args.output_root,
            selection=args.selection,
            selection_jsonl=args.selection_jsonl,
            verify=args.verify,
            batch_size=args.batch_size,
            target_shard_bytes=_target_bytes(args.target_shard_gib),
            staging_dir=args.staging_dir,
            verify_uploaded_checksum=args.verify_uploaded_checksum,
        )
    else:
        parts_root = Path(args.parts_root).expanduser().resolve()
        part_roots = sorted(
            path
            for path in parts_root.glob("part-*")
            if path.is_dir()
        )
        manifest = merge_part_manifests(
            part_roots,
            args.output_root,
            verify_part_checksums=args.verify_part_checksums,
            canonical_root=args.canonical_root,
        )
    print(
        json.dumps(
            {
                "output_root": str(Path(args.output_root).resolve()),
                "cache_kind": manifest["schema"]["cache_kind"],
                "total_frames": manifest["total_frames"],
                "shard_count": len(manifest["shards"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
