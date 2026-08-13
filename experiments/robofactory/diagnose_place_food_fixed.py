#!/usr/bin/env python3
"""Run one fixed PlaceFood FastWAM closed-loop rollout or expert diagnostic.

The command is intentionally narrow: it binds one frozen panel episode to
one simulator initial state, records every closed-loop policy query, compares
the persisted and online first-frame Gaussian inputs, and evaluates the same
policy by teacher forcing over the episode's stored expert states.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

try:
    from .fastwam_multi_robot_policy import (
        NOPOSPLAT_CHECKPOINT_SHA256,
        POLICY_LIGHTNING_COMMIT,
        TRAINING_STATS_SHA256,
        FastWAMMultiRobotPolicy,
        encode_compact_agent_gaussian,
        prepare_observation,
    )
except ImportError:
    from fastwam_multi_robot_policy import (  # type: ignore[no-redef]
        NOPOSPLAT_CHECKPOINT_SHA256,
        POLICY_LIGHTNING_COMMIT,
        TRAINING_STATS_SHA256,
        FastWAMMultiRobotPolicy,
        encode_compact_agent_gaussian,
        prepare_observation,
    )


ROBOFACTORY_COMMIT = "2d34fb38c80cb06550a5dbf99abac2c89f4336ed"
LEGACY_HELDOUT_PANEL_SCHEMA = "fastwam-robofactory-heldout-panel-v1"
SPLIT_PANEL_SCHEMA = "fastwam-robofactory-split-panel-nohash-v1"
FORMAL_TEACHER_ACTION_STOP = 268
SPLIT_KEY_SCHEME = "sorted_trajectory_ordinal_splitmix64_v1"
_UINT64_MASK = (1 << 64) - 1
TASK_CONFIGS = {
    "PlaceFood-rf": "configs/table/place_food.yaml",
    "PlaceCubeInCup-rf": "configs/table/place_cube_in_cup.yaml",
    "StrikeCubeHard-rf": "configs/table/strike_cube_hard.yaml",
    "ThreeRobotsPlaceShoes-rf": "configs/table/three_robots_place_shoes.yaml",
    "ThreeRobotsStackCube-rf": "configs/table/three_robots_stack_cube.yaml",
    "FourRobotsStackCube-rf": "configs/table/four_robots_stack_cube.yaml",
}


def _mix_uint64(*values: int) -> int:
    """Mirror the dataset's no-hash integer split mixer exactly."""

    state = 0x9E3779B97F4A7C15
    for value in values:
        state = (
            state
            + (int(value) & _UINT64_MASK)
            + 0x9E3779B97F4A7C15
        ) & _UINT64_MASK
        state = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
        state = ((state ^ (state >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
        state ^= state >> 31
    return state & _UINT64_MASK


def _split_fraction_from_ordinal(ordinal: int, seed: int) -> float:
    return _mix_uint64(seed, ordinal) / float(2**64)


def _selected_episodes(
    panel: Mapping[str, Any],
    task_name: str,
    episode_start: int,
    num_episodes: int,
) -> list[dict[str, Any]]:
    if task_name not in TASK_CONFIGS:
        raise KeyError(
            f"Unsupported task {task_name!r}; expected one of {sorted(TASK_CONFIGS)}"
        )
    records = [
        dict(record)
        for record in panel["episodes"]
        if record.get("task_name") == task_name
    ]
    records.sort(key=lambda record: int(record["task_index"]))
    if episode_start < 0 or num_episodes < 1:
        raise ValueError("episode_start must be >= 0 and num_episodes must be positive")
    selected = records[episode_start : episode_start + num_episodes]
    if len(selected) != num_episodes:
        raise ValueError(
            f"Panel has {len(records)} episodes for {task_name}; requested "
            f"[{episode_start}, {episode_start + num_episodes})"
        )
    return selected


def _source_path(dataset_root: Path, relative: str) -> Path:
    source = (dataset_root / relative).resolve(strict=True)
    try:
        source.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"Source path escapes dataset root: {relative!r}") from error
    if not source.is_file():
        raise ValueError(f"Source H5 must be a regular file: {source}")
    return source


def _as_bool(value: Any, *, label: str) -> bool:
    tensor = torch.as_tensor(value).detach().cpu().reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(
            f"{label} must contain one value, got shape {tuple(tensor.shape)}"
        )
    return bool(tensor.item())


def _flat_action_to_dict(
    action: np.ndarray,
    agent_names: Sequence[str],
) -> dict[str, np.ndarray]:
    flat = np.asarray(action, dtype=np.float32).reshape(-1)
    expected = len(agent_names) * ACTION_DIM
    if flat.shape != (expected,):
        raise ValueError(f"Flat action must have shape ({expected},), got {flat.shape}")
    if not np.isfinite(flat).all():
        raise FloatingPointError("Policy action contains non-finite values")
    return {
        name: np.ascontiguousarray(
            flat[index * ACTION_DIM : (index + 1) * ACTION_DIM]
        )
        for index, name in enumerate(agent_names)
    }


def _expert_action_at(
    actions: Mapping[str, np.ndarray],
    agent_names: Sequence[str],
    timestep: int,
) -> dict[str, np.ndarray]:
    """Return one stored multi-agent action without changing its ordering."""

    if timestep < 0:
        raise ValueError(f"Expert action timestep must be non-negative, got {timestep}")
    result: dict[str, np.ndarray] = {}
    for name in agent_names:
        if name not in actions:
            raise KeyError(f"Expert actions are missing agent {name!r}")
        array = np.asarray(actions[name], dtype=np.float32)
        if array.ndim != 2 or array.shape[1:] != (ACTION_DIM,):
            raise ValueError(f"Invalid expert action array for {name}: {array.shape}")
        if timestep >= len(array):
            raise IndexError(
                f"Expert action timestep {timestep} is outside {name} length {len(array)}"
            )
        vector = np.ascontiguousarray(array[timestep])
        if not np.isfinite(vector).all():
            raise FloatingPointError(
                f"Expert action contains non-finite values for {name} at {timestep}"
            )
        result[name] = vector
    return result


def _build_environment(robofactory_root: Path, task_name: str):
    if str(robofactory_root) not in sys.path:
        sys.path.insert(0, str(robofactory_root))
    os.chdir(robofactory_root)
    __import__("tasks")
    import gymnasium as gym

    config = TASK_CONFIGS[task_name]
    if not (robofactory_root / config).is_file():
        raise FileNotFoundError(f"Task config is missing: {robofactory_root / config}")
    return gym.make(
        task_name,
        config=config,
        obs_mode="rgb",
        reward_mode="dense",
        control_mode="pd_joint_pos",
        render_mode="sensors",
        sensor_configs={"shader_pack": "default"},
        human_render_camera_configs={"shader_pack": "default"},
        viewer_camera_configs={"shader_pack": "default"},
        num_envs=1,
        sim_backend="auto",
        render_backend="gpu",
        enable_shadow=True,
        parallel_in_single_scene=False,
    )


def _reset_environment(env: Any, episode: Mapping[str, Any]) -> None:
    reset_kwargs = dict(episode["reset_kwargs"])
    reset_seed = reset_kwargs.pop("seed", episode["episode_seed"])
    env.reset(seed=reset_seed, **reset_kwargs)


SCHEMA_VERSION = "fastwam-placefood-fixed-diagnostic-v3"
ACTION_DIM = 8
ARM_DIMS = tuple(range(7))
GRIPPER_DIM = 7
FPS = 20


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    temporary.write_text(
        json.dumps(
            _json_value(payload), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            _json_value(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_video(path: Path, *, expected_frames: int) -> dict[str, Any]:
    if path.is_symlink():
        raise RuntimeError(f"Refusing symlink video: {path}")
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    if not resolved.is_file() or stat.st_size <= 0:
        raise RuntimeError(f"Video is not a non-empty regular file: {resolved}")
    reader = imageio.get_reader(str(resolved))
    try:
        actual_frames = int(reader.count_frames())
        if actual_frames != expected_frames:
            raise RuntimeError(
                f"Video frame count mismatch for {resolved}: "
                f"expected={expected_frames} actual={actual_frames}"
            )
        first = np.asarray(reader.get_data(0))
        last = np.asarray(reader.get_data(expected_frames - 1))
    finally:
        reader.close()
    if first.ndim != 3 or last.shape != first.shape or first.shape[-1] < 3:
        raise RuntimeError(
            f"Video decoded frame shape mismatch for {resolved}: "
            f"first={first.shape} last={last.shape}"
        )
    return {
        "bytes": int(stat.st_size),
        "frames": actual_frames,
        "frame_shape": list(first.shape),
        "first_and_last_frame_decoded": True,
    }


def _publish_video(
    source: Path, destination: Path, *, expected_frames: int
) -> dict[str, Any]:
    """Validate a local MP4, atomically publish it, then validate OSS readback."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite video: {destination}")
    local_report = _validate_video(source, expected_frames=expected_frames)
    temporary = destination.parent / (
        f".{destination.name}.publishing.{os.getpid()}.{uuid.uuid4().hex}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o640,
    )
    try:
        with source.open("rb") as source_handle, os.fdopen(
            descriptor, "wb", closefd=False
        ) as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
    finally:
        os.close(descriptor)
    if temporary.stat().st_size != local_report["bytes"]:
        raise RuntimeError(
            f"Published video size mismatch before rename: {temporary}"
        )
    os.replace(temporary, destination)
    published_report = _validate_video(destination, expected_frames=expected_frames)
    if published_report != local_report:
        raise RuntimeError(
            f"Published video readback differs from local staging: {destination}"
        )
    return {
        **published_report,
        "encoding_staged_on_local_disk": True,
        "published_readback_validated": True,
    }


def _publish_staged_file(source: Path, destination: Path) -> dict[str, Any]:
    """Atomically publish a local regular file and verify the OSS readback."""
    if source.is_symlink():
        raise RuntimeError(f"Refusing symlink source: {source}")
    resolved_source = source.resolve(strict=True)
    source_stat = resolved_source.stat()
    if not resolved_source.is_file():
        raise RuntimeError(f"Source is not a regular file: {resolved_source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite file: {destination}")
    temporary = destination.parent / (
        f".{destination.name}.publishing.{os.getpid()}.{uuid.uuid4().hex}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o640,
    )
    try:
        with resolved_source.open("rb") as source_handle, os.fdopen(
            descriptor, "wb", closefd=False
        ) as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
    finally:
        os.close(descriptor)
    if temporary.stat().st_size != source_stat.st_size:
        raise RuntimeError(
            f"Published file size mismatch before rename: {temporary}"
        )
    os.replace(temporary, destination)
    destination_stat = destination.stat()
    if destination_stat.st_size != source_stat.st_size:
        raise RuntimeError(f"Published file size mismatch: {destination}")
    with resolved_source.open("rb") as source_handle, destination.open(
        "rb"
    ) as destination_handle:
        while True:
            source_chunk = source_handle.read(1024 * 1024)
            destination_chunk = destination_handle.read(1024 * 1024)
            if source_chunk != destination_chunk:
                raise RuntimeError(
                    f"Published file readback differs from local staging: {destination}"
                )
            if not source_chunk:
                break
    return {
        "bytes": int(destination_stat.st_size),
        "staged_on_local_disk": True,
        "published_readback_validated": True,
    }


def _load_panel_nohash(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    schema = payload.get("schema_version")
    if schema not in {LEGACY_HELDOUT_PANEL_SCHEMA, SPLIT_PANEL_SCHEMA}:
        raise ValueError(f"Unexpected rollout panel schema: {schema!r}")
    if not isinstance(payload.get("episodes"), list):
        raise TypeError("Rollout panel must contain an episodes list")
    if schema == LEGACY_HELDOUT_PANEL_SCHEMA:
        return payload

    split = payload.get("split")
    if split not in {"train", "val"}:
        raise ValueError(f"Split panel must declare split=train or val, got {split!r}")
    if payload.get("split_key_scheme") != SPLIT_KEY_SCHEME:
        raise ValueError(
            "Split panel key scheme mismatch: "
            f"{payload.get('split_key_scheme')!r}"
        )
    split_seed = int(payload.get("split_seed"))
    val_set_proportion = float(payload.get("val_set_proportion"))
    if not 0.0 < val_set_proportion < 1.0:
        raise ValueError("val_set_proportion must be strictly between zero and one")
    if not payload["episodes"]:
        raise ValueError("Split panel episodes must not be empty")
    paired_policy_seeds = payload.get("paired_policy_seeds")
    if not isinstance(paired_policy_seeds, list):
        raise TypeError("Split panel must contain a paired_policy_seeds list")
    if len(paired_policy_seeds) != len(payload["episodes"]):
        raise ValueError(
            "paired_policy_seeds length must equal the split panel episode count"
        )
    normalized_policy_seeds = [int(value) for value in paired_policy_seeds]
    if len(set(normalized_policy_seeds)) != len(normalized_policy_seeds):
        raise ValueError("paired_policy_seeds must be unique")

    identities: set[tuple[str, str]] = set()
    ordinals: set[int] = set()
    for expected_index, record in enumerate(payload["episodes"]):
        if not isinstance(record, Mapping):
            raise TypeError("Every split panel episode must be an object")
        task_index = int(record["task_index"])
        panel_index = int(record["panel_index"])
        if task_index != expected_index or panel_index != expected_index:
            raise ValueError(
                "Split panel task_index and panel_index must match list order: "
                f"position={expected_index} task_index={task_index} "
                f"panel_index={panel_index}"
            )
        ordinal = int(record["global_ordinal"])
        if ordinal < 0 or ordinal in ordinals:
            raise ValueError(f"Invalid or duplicate global_ordinal: {ordinal}")
        ordinals.add(ordinal)
        recorded_fraction = float(record["split_fraction"])
        expected_fraction = _split_fraction_from_ordinal(ordinal, split_seed)
        if recorded_fraction != expected_fraction:
            raise ValueError(
                f"Split fraction mismatch at global_ordinal={ordinal}: "
                f"recorded={recorded_fraction!r} expected={expected_fraction!r}"
            )
        expected_split = (
            "val" if expected_fraction < val_set_proportion else "train"
        )
        if record.get("split") != split or expected_split != split:
            raise ValueError(
                f"Episode split mismatch at global_ordinal={ordinal}: "
                f"panel={split!r} episode={record.get('split')!r} "
                f"computed={expected_split!r}"
            )
        identity = (str(record["source_path"]), str(record["trajectory"]))
        if identity in identities:
            raise ValueError(f"Duplicate episode identity: {identity!r}")
        identities.add(identity)
    return payload


def _load_episode_data(
    source: Path,
    trajectory: str,
    agent_names: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    from mani_skill.trajectory import utils as trajectory_utils

    with h5py.File(source, "r") as handle:
        group = handle[trajectory]
        states = trajectory_utils.dict_to_list_of_dicts(group["env_states"])
        actions = {
            name: np.asarray(group["actions"][name][:], dtype=np.float32)
            for name in agent_names
        }
        observations: dict[str, np.ndarray] = {}
        for name in agent_names:
            observations[f"agent/{name}/qpos"] = np.asarray(
                group["obs"]["agent"][name]["qpos"][:]
            )
            observations[f"agent/{name}/qvel"] = np.asarray(
                group["obs"]["agent"][name]["qvel"][:]
            )
        for camera in (
            "head_camera_agent0",
            "head_camera_agent1",
            "head_camera_global",
        ):
            observations[f"sensor_data/{camera}/rgb"] = np.asarray(
                group["obs"]["sensor_data"][camera]["rgb"][:]
            )
    action_lengths = {array.shape for array in actions.values()}
    if len(action_lengths) != 1 or next(iter(action_lengths))[1:] != (ACTION_DIM,):
        raise ValueError(f"Invalid expert actions: {action_lengths}")
    action_count = next(iter(actions.values())).shape[0]
    if len(states) != action_count + 1:
        raise ValueError(f"Expected states=actions+1, got {len(states)} and {action_count}")
    for key, array in observations.items():
        if array.shape[0] != len(states):
            raise ValueError(f"Observation length mismatch for {key}: {array.shape}")
    return states, actions, observations


def _stored_observation(
    observations: Mapping[str, np.ndarray],
    agent_names: Sequence[str],
    timestep: int,
) -> dict[str, Any]:
    return {
        "agent": {
            name: {
                "qpos": observations[f"agent/{name}/qpos"][timestep],
                "qvel": observations[f"agent/{name}/qvel"][timestep],
            }
            for name in agent_names
        },
        "sensor_data": {
            camera: {
                "rgb": observations[f"sensor_data/{camera}/rgb"][timestep]
            }
            for camera in (
                "head_camera_agent0",
                "head_camera_agent1",
                "head_camera_global",
            )
        },
    }


def _stored_rgb_on_live_observation(
    live_observation: Mapping[str, Any],
    observations: Mapping[str, np.ndarray],
    agent_names: Sequence[str],
    timestep: int,
) -> dict[str, Any]:
    """Use persisted RGB while keeping live proprioception exactly shared.

    The teacher-forcing A/B is intended to isolate rendering only.  Building
    the stored branch from the just-restored live observation prevents H5
    qpos/qvel serialization differences from becoming a second variable.
    """

    live_agents = live_observation.get("agent")
    if not isinstance(live_agents, Mapping):
        raise KeyError("Live observation is missing the agent mapping")
    return {
        "agent": {
            name: {
                "qpos": live_agents[name]["qpos"],
                "qvel": live_agents[name]["qvel"],
            }
            for name in agent_names
        },
        "sensor_data": {
            camera: {
                "rgb": observations[f"sensor_data/{camera}/rgb"][timestep]
            }
            for camera in (
                "head_camera_agent0",
                "head_camera_agent1",
                "head_camera_global",
            )
        },
    }


def _vector(value: Any, count: int = 3) -> np.ndarray:
    array = torch.as_tensor(value).detach().cpu().numpy().reshape(-1, count)
    return np.asarray(array[0], dtype=np.float64)


def _flat(value: Any) -> np.ndarray:
    return np.asarray(torch.as_tensor(value).detach().cpu().numpy()).reshape(-1)


def _state_leaf_arrays(
    state: Mapping[str, Any],
) -> dict[tuple[Any, ...], np.ndarray]:
    """Copy every numeric state leaf so later simulator writes cannot alias it."""

    leaves: dict[tuple[Any, ...], np.ndarray] = {}

    def visit(value: Any, path: tuple[Any, ...]) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, (*path, key))
            return
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if array.dtype.kind not in "biufc":
            raise TypeError(f"Non-numeric state leaf at {path}: {array.dtype}")
        leaves[path] = np.asarray(array).copy()

    visit(state, ())
    return leaves


def _changed_scalars(
    before: np.ndarray,
    after: np.ndarray,
    *,
    path: tuple[Any, ...],
    atol: float = 1e-7,
) -> list[dict[str, Any]]:
    if before.shape != after.shape:
        raise ValueError(
            f"State shape changed at {path}: {before.shape} -> {after.shape}"
        )
    changed = ~np.isclose(before, after, rtol=0.0, atol=atol, equal_nan=True)
    flat_before = before.reshape(-1)
    flat_after = after.reshape(-1)
    return [
        {
            "state_path": [str(item) for item in path],
            "flat_index": int(index),
            "before": float(flat_before[index]),
            "after": float(flat_after[index]),
        }
        for index in np.flatnonzero(changed)
    ]


def sanitize_initial_pot_lid(env: Any) -> dict[str, Any]:
    """Zero only the one-DoF pot lid qpos/qvel and prove all other state is intact.

    Mutation uses the object API, not RoboFactory's unstable serialized
    ``articulations/None`` key.  The serialized pot leaf is resolved only for a
    fail-closed before/after audit by its one-DoF state shape and current tail.
    """

    unwrapped = env.unwrapped
    pot = unwrapped.pot
    raw_qpos = pot.get_qpos()
    raw_qvel = pot.get_qvel()
    before_qpos = _flat(raw_qpos).astype(np.float64, copy=True)
    before_qvel = _flat(raw_qvel).astype(np.float64, copy=True)
    if before_qpos.shape != (1,) or before_qvel.shape != (1,):
        raise ValueError(
            "PlaceFood pot lid must be exactly one DoF, got "
            f"qpos={before_qpos.shape} qvel={before_qvel.shape}"
        )

    before_leaves = _state_leaf_arrays(unwrapped.get_state_dict())
    expected_leaf_size = 13 + 2 * before_qpos.size
    direct_tail = np.concatenate((before_qpos, before_qvel))
    candidates = [
        path
        for path, array in before_leaves.items()
        if len(path) == 2
        and path[0] == "articulations"
        and array.size == expected_leaf_size
        and np.allclose(
            array.reshape(-1)[-direct_tail.size :],
            direct_tail,
            rtol=0.0,
            atol=1e-6,
        )
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Could not uniquely resolve the one-DoF pot articulation for audit; "
            f"candidate paths={candidates}"
        )
    target_path = candidates[0]

    pot.set_qpos(torch.zeros_like(raw_qpos))
    pot.set_qvel(torch.zeros_like(raw_qvel))
    after_qpos = _flat(pot.get_qpos()).astype(np.float64, copy=True)
    after_qvel = _flat(pot.get_qvel()).astype(np.float64, copy=True)
    if not np.allclose(after_qpos, 0.0, rtol=0.0, atol=1e-7) or not np.allclose(
        after_qvel, 0.0, rtol=0.0, atol=1e-7
    ):
        raise RuntimeError(
            f"Pot lid sanitization did not stick: qpos={after_qpos} qvel={after_qvel}"
        )

    after_leaves = _state_leaf_arrays(unwrapped.get_state_dict())
    if before_leaves.keys() != after_leaves.keys():
        raise RuntimeError("Pot lid sanitization changed the state_dict structure")
    unexpected: list[dict[str, Any]] = []
    observed_changes: list[dict[str, Any]] = []
    for path, before in before_leaves.items():
        after = after_leaves[path]
        observed_changes.extend(_changed_scalars(before, after, path=path))
        expected = before.copy()
        if path == target_path:
            expected.reshape(-1)[-direct_tail.size :] = 0.0
        unexpected.extend(_changed_scalars(expected, after, path=path))
    if unexpected:
        raise RuntimeError(
            "Pot lid sanitization modified state outside the resolved qpos/qvel "
            f"tail: {unexpected[:8]}"
        )

    return {
        "mode": "clean_lid_zero_v1",
        "mutation_api": "env.unwrapped.pot.set_qpos/set_qvel",
        "serialized_target_resolution": "one_dof_shape_and_direct_qpos_qvel_tail",
        "serialized_target_path": [str(item) for item in target_path],
        "before": {"qpos": before_qpos, "qvel": before_qvel},
        "after": {"qpos": after_qpos, "qvel": after_qvel},
        "observed_changed_scalars": observed_changes,
        "other_state_verified_unchanged": True,
        "numeric_state_leaf_count": len(before_leaves),
    }


def teacher_state_schedule(
    *, action_count: int, start_timestep: int, max_states: int, base_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute expert timesteps and source-independent paired seeds."""

    if action_count < 1 or start_timestep < 0 or max_states < 1:
        raise ValueError("Invalid teacher-forcing schedule arguments")
    if start_timestep >= action_count:
        raise ValueError(
            f"Teacher start {start_timestep} is outside {action_count} actions"
        )
    stop = min(action_count, start_timestep + max_states)
    timesteps = np.arange(start_timestep, stop, dtype=np.int64)
    seeds = np.asarray(base_seed + timesteps, dtype=np.int64)
    return timesteps, seeds


def rollout_conditions(
    initial_states: Sequence[str], exec_horizons: Sequence[int]
) -> tuple[tuple[str, int], ...]:
    """Validate and deterministically expand the closed-loop condition matrix."""

    modes = tuple(dict.fromkeys(str(item) for item in initial_states))
    horizons = tuple(dict.fromkeys(int(item) for item in exec_horizons))
    if not modes or any(mode not in {"raw", "clean"} for mode in modes):
        raise ValueError(f"Initial states must be raw/clean, got {modes}")
    if not horizons or any(horizon not in {1, 5} for horizon in horizons):
        raise ValueError(f"Execution horizons must be 1/5, got {horizons}")
    return tuple((mode, horizon) for mode in modes for horizon in horizons)


def validate_formal_rollout_contract(
    *,
    max_steps: int,
    initial_state: str,
    exec_horizon: int,
    initial_state_explicit: bool,
    exec_horizon_explicit: bool,
) -> dict[str, Any]:
    """Fail closed unless one formal rollout cell is fully specified."""

    if max_steps != 300:
        raise ValueError(f"Formal rollout requires max_steps=300, got {max_steps}")
    if not initial_state_explicit or initial_state not in {"raw", "clean"}:
        raise ValueError("Formal rollout requires explicit --initial-state raw|clean")
    if not exec_horizon_explicit or exec_horizon not in {1, 5}:
        raise ValueError("Formal rollout requires explicit --exec-horizon 1|5")
    return {
        "max_steps": max_steps,
        "initial_state": initial_state,
        "exec_horizon": exec_horizon,
        "explicit_cell": True,
    }


def validate_formal_expert_replay_contract(
    *,
    max_steps: int,
    initial_state: str,
    initial_state_explicit: bool,
    evaluation_code_commit: str | None,
) -> dict[str, Any]:
    """Fail closed unless expert replay starts from the untouched H5 state."""

    if max_steps != 300:
        raise ValueError(f"Formal expert replay requires max_steps=300, got {max_steps}")
    if not initial_state_explicit or initial_state != "raw":
        raise ValueError("Formal expert replay requires explicit --initial-state raw")
    if not evaluation_code_commit or evaluation_code_commit.strip() != evaluation_code_commit:
        raise ValueError(
            "Formal expert replay requires a non-empty --evaluation-code-commit"
        )
    return {
        "max_steps": max_steps,
        "initial_state": initial_state,
        "action_source": "stored_h5_expert",
        "policy_initialized": False,
        "evaluation_code_commit": evaluation_code_commit,
    }


def validate_formal_teacher_contract(
    *, timesteps: np.ndarray, valid: np.ndarray, action_horizon: int
) -> dict[str, Any]:
    """Validate the complete fixed-trajectory teacher-forcing artifact."""

    expected = np.arange(5, 268, dtype=np.int64)
    if action_horizon < 5:
        raise ValueError(
            f"Formal teacher forcing requires action_horizon>=5, got {action_horizon}"
        )
    if timesteps.shape != (263,) or not np.array_equal(timesteps, expected):
        raise ValueError(
            "Formal teacher forcing requires exactly timesteps 5..267 "
            f"(263 states), got shape={timesteps.shape}"
        )
    if valid.shape[0] != 263 or valid.shape[1] < 5:
        raise ValueError(f"Formal teacher valid mask has wrong shape: {valid.shape}")
    valid_h1 = int(valid[:, :1].sum())
    valid_h5 = int(valid[:, :5].sum())
    if (valid_h1, valid_h5) != (263, 1305):
        raise ValueError(
            "Formal teacher valid-pair counts must be H1=263/H5=1305, "
            f"got H1={valid_h1}/H5={valid_h5}"
        )
    return {
        "timesteps": "5..267",
        "states": 263,
        "action_stop_exclusive": FORMAL_TEACHER_ACTION_STOP,
        "valid_pairs_h1": valid_h1,
        "valid_pairs_h5": valid_h5,
        "action_horizon": action_horizon,
    }


def teacher_target_length(
    *, action_count: int, timestep: int, horizon: int, formal_contract: bool
) -> int:
    """Return the scored future-action count for one teacher query state."""

    if action_count < 1 or timestep < 0 or horizon < 1:
        raise ValueError("Invalid teacher target-length arguments")
    action_stop = (
        min(action_count, FORMAL_TEACHER_ACTION_STOP)
        if formal_contract
        else action_count
    )
    return max(0, min(horizon, action_stop - timestep))


def target_action_phase_mask(
    phase_mask: np.ndarray, timesteps: np.ndarray, horizon: int
) -> np.ndarray:
    """Map an absolute state phase mask to each prediction target ``t+h``."""

    phase_mask = np.asarray(phase_mask, dtype=bool)
    timesteps = np.asarray(timesteps, dtype=np.int64)
    if phase_mask.ndim != 1 or timesteps.ndim != 1 or horizon < 1:
        raise ValueError("Invalid target-action phase-mask arguments")
    targets = timesteps[:, None] + np.arange(horizon, dtype=np.int64)[None, :]
    result = np.zeros(targets.shape, dtype=bool)
    in_range = (targets >= 0) & (targets < len(phase_mask))
    result[in_range] = phase_mask[targets[in_range]]
    return result


def physical_snapshot(
    env: Any,
    *,
    simulator_step: int,
    last_action: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Capture task geometry and gripper state after a simulator state transition."""

    unwrapped = env.unwrapped
    meat = _vector(unwrapped.meat.pose.p)
    pot = _vector(unwrapped.pot.pose.p)
    lid_qpos = float(_flat(unwrapped.pot.get_qpos())[0])
    lid_qvel = float(_flat(unwrapped.pot.get_qvel())[0])
    robots: dict[str, Any] = {}
    tcp_positions: list[np.ndarray] = []
    for index, wrapper in enumerate(unwrapped.agent.agents):
        name = f"panda-{index}"
        tcp = _vector(wrapper.tcp.pose.p)
        qpos = _flat(wrapper.robot.get_qpos())
        fingers = np.asarray(qpos[-2:], dtype=np.float64)
        aperture = float(fingers.sum())
        tcp_positions.append(tcp)
        robots[name] = {
            "tcp_position": tcp,
            "tcp_to_meat_distance": float(np.linalg.norm(tcp - meat)),
            "tcp_to_pot_distance": float(np.linalg.norm(tcp - pot)),
            "finger_qpos": fingers,
            "gripper_aperture": aperture,
            "gripper_closed": aperture < 0.03,
            "gripper_released": aperture > 0.06,
            "commanded_gripper": (
                None if last_action is None else float(last_action[name][GRIPPER_DIM])
            ),
        }
    robot0_grasping_meat = _as_bool(
        unwrapped.agent.agents[0].is_grasping(unwrapped.meat),
        label="robot0.is_grasping(meat)",
    )
    base_z = float(_vector(unwrapped.agent.agents[0].robot.pose.p)[2])
    meat_pot_delta = meat - pot
    return {
        "simulator_step": int(simulator_step),
        "meat_position": meat,
        "pot_position": pot,
        "meat_pot_xy_distance": float(np.linalg.norm(meat_pot_delta[:2])),
        "meat_pot_xyz_distance": float(np.linalg.norm(meat_pot_delta)),
        "meat_height": float(meat[2]),
        "robot0_grasping_meat": robot0_grasping_meat,
        "pot_lid_qpos": lid_qpos,
        "pot_lid_qvel": lid_qvel,
        "robot0_base_z": base_z,
        "task_success_from_geometry": bool(
            np.linalg.norm(meat_pot_delta[:2]) < 0.1 and meat[2] < base_z + 0.1
        ),
        "tcp_pair_distance": (
            None
            if len(tcp_positions) < 2
            else float(np.linalg.norm(tcp_positions[0] - tcp_positions[1]))
        ),
        "agents": robots,
    }


def rollout_grasp_metrics(snapshots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize physical grasp and lift without changing rollout control."""

    if not snapshots:
        raise ValueError("At least one physical snapshot is required")
    heights = np.asarray(
        [float(item["meat_height"]) for item in snapshots], dtype=np.float64
    )
    grasped = np.asarray(
        [bool(item["robot0_grasping_meat"]) for item in snapshots], dtype=bool
    )
    if not np.isfinite(heights).all():
        raise FloatingPointError("Meat heights contain non-finite values")
    initial_height = float(heights[0])
    max_height = float(heights.max())
    return {
        "robot0_grasp_ever": bool(grasped.any()),
        "robot0_grasp_steps": int(grasped.sum()),
        "robot0_grasp_fraction": float(grasped.mean()),
        "meat_initial_height_m": initial_height,
        "meat_max_height_m": max_height,
        "meat_max_lift_m": max(0.0, max_height - initial_height),
    }


def _video_frame(env: Any) -> np.ndarray:
    frame = torch.as_tensor(env.render()).detach().cpu()
    while frame.ndim > 3 and frame.shape[0] == 1:
        frame = frame[0]
    array = frame.numpy()
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Unexpected rendered frame shape: {array.shape}")
    return np.ascontiguousarray(array[..., :3], dtype=np.uint8)


def gaussian_error(reference: Any, candidate: Any) -> dict[str, Any]:
    reference_array = np.asarray(torch.as_tensor(reference).float().cpu(), dtype=np.float64)
    candidate_array = np.asarray(torch.as_tensor(candidate).float().cpu(), dtype=np.float64)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            f"Gaussian shapes differ: {reference_array.shape} vs {candidate_array.shape}"
        )
    delta = candidate_array - reference_array
    ref_flat = reference_array.reshape(-1)
    candidate_flat = candidate_array.reshape(-1)
    denominator = float(np.linalg.norm(ref_flat) * np.linalg.norm(candidate_flat))
    return {
        "shape": list(reference_array.shape),
        "signed_mean_error": float(delta.mean()),
        "mean_absolute_error": float(np.abs(delta).mean()),
        "max_absolute_error": float(np.abs(delta).max()),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "cosine": None if denominator == 0.0 else float(np.dot(ref_flat, candidate_flat) / denominator),
    }


def gaussian_error_report(reference: Any, candidate: Any) -> dict[str, Any]:
    reference_tensor = torch.as_tensor(reference)
    candidate_tensor = torch.as_tensor(candidate)
    report = {"overall": gaussian_error(reference_tensor, candidate_tensor)}
    report["by_agent"] = {
        f"panda-{index}": gaussian_error(reference_tensor[index], candidate_tensor[index])
        for index in range(reference_tensor.shape[0])
    }
    report["by_channel"] = {
        str(index): gaussian_error(reference_tensor[:, index], candidate_tensor[:, index])
        for index in range(reference_tensor.shape[1])
    }
    return report


def _camera_array(observation: Mapping[str, Any], camera: str) -> np.ndarray:
    value = observation["sensor_data"][camera]["rgb"]
    array = np.asarray(torch.as_tensor(value).detach().cpu())
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Unexpected {camera} RGB shape: {array.shape}")
    return np.asarray(array[..., :3])


def rgb_pair_error_report(
    reference_obs: Mapping[str, Any], candidate_obs: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for camera in (
        "head_camera_agent0",
        "head_camera_agent1",
        "head_camera_global",
    ):
        left = _camera_array(reference_obs, camera).astype(np.float64)
        right = _camera_array(candidate_obs, camera).astype(np.float64)
        result[camera] = gaussian_error(left, right)
    return result


def action_bound_records(
    action: Mapping[str, np.ndarray],
    action_space: Any,
    *,
    simulator_step: int,
    query_index: int,
    chunk_offset: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for agent_name, values in action.items():
        low = np.asarray(action_space[agent_name].low).reshape(-1)
        high = np.asarray(action_space[agent_name].high).reshape(-1)
        vector = np.asarray(values).reshape(-1)
        if vector.shape != (ACTION_DIM,) or low.shape != vector.shape or high.shape != vector.shape:
            raise ValueError(f"Action-space shape mismatch for {agent_name}")
        for dimension, value in enumerate(vector):
            if value < low[dimension] or value > high[dimension]:
                records.append(
                    {
                        "simulator_step": int(simulator_step),
                        "simulator_step_1based": int(simulator_step + 1),
                        "query_index": int(query_index),
                        "chunk_offset": int(chunk_offset),
                        "agent": agent_name,
                        "group": "arm" if dimension in ARM_DIMS else "gripper",
                        "dimension": int(dimension),
                        "value": float(value),
                        "low": float(low[dimension]),
                        "high": float(high[dimension]),
                        "direction": "below_low" if value < low[dimension] else "above_high",
                        "exceedance": float(
                            low[dimension] - value
                            if value < low[dimension]
                            else value - high[dimension]
                        ),
                    }
                )
    return records


def bucket_bound_violations(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(records)

    def counted(key) -> dict[str, int]:
        return dict(sorted(Counter(key(record) for record in records).items()))

    return {
        "total_scalar_violations": len(records),
        "by_agent": counted(lambda record: str(record["agent"])),
        "by_group": counted(lambda record: str(record["group"])),
        "by_agent_and_group": counted(
            lambda record: f"{record['agent']}/{record['group']}"
        ),
        "by_dimension": counted(
            lambda record: f"{record['agent']}/dim{record['dimension']}"
        ),
        "by_agent_group_and_dimension": counted(
            lambda record: (
                f"{record['agent']}/{record['group']}/dim{record['dimension']}"
            )
        ),
        "by_simulator_step": counted(lambda record: str(record["simulator_step"])),
    }


def _first_true(values: Sequence[bool], start: int = 0) -> int | None:
    return next((index for index in range(start, len(values)) if values[index]), None)


def phase_masks(
    snapshots: Sequence[Mapping[str, Any]], *, settle_frames: int = 5
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not snapshots:
        raise ValueError("Cannot derive phases from an empty trajectory")
    lid = np.asarray([float(snapshot["pot_lid_qpos"]) for snapshot in snapshots])
    meat_z = np.asarray([float(snapshot["meat_height"]) for snapshot in snapshots])
    xy = np.asarray([float(snapshot["meat_pot_xy_distance"]) for snapshot in snapshots])
    success = np.asarray(
        [bool(snapshot["task_success_from_geometry"]) for snapshot in snapshots]
    )
    settled = None
    settle_confirmed = None
    for start in range(0, len(lid) - settle_frames + 1):
        if bool(np.all(np.abs(lid[start : start + settle_frames]) < 0.02)):
            settled = start
            settle_confirmed = start + settle_frames - 1
            break
    if settled is None:
        raise ValueError("Pot lid never settled below 0.02 for the required run")
    lid_removed = _first_true((lid >= 0.10).tolist(), int(settle_confirmed) + 1)
    meat_lifted = _first_true((meat_z >= meat_z[0] + 0.05).tolist(), settled)
    in_pot = _first_true((xy <= 0.10).tolist(), settled)
    success_index = _first_true(success.tolist(), settled)
    event_values = {
        "lid_settled": settled,
        "lid_settle_confirmed": settle_confirmed,
        "lid_removed": lid_removed,
        "meat_lifted": meat_lifted,
        "meat_in_pot_xy": in_pot,
        "success": success_index,
    }
    if any(
        event_values[name] is None
        for name in ("lid_removed", "meat_lifted", "meat_in_pot_xy", "success")
    ):
        raise ValueError(f"Expert trajectory is missing required phase event: {event_values}")
    size = len(snapshots)
    index = np.arange(size)
    manipulation_complete = max(int(lid_removed), int(meat_lifted))
    masks = {
        # Lid and grasp are deliberately allowed to overlap for this two-arm expert.
        "lid_move": (index >= settled) & (index <= int(lid_removed)),
        "grasp": (index >= settled) & (index <= int(meat_lifted)),
        "transport": (index >= manipulation_complete) & (index < int(in_pot)),
        "place": (index >= int(in_pot)) & (index <= int(success_index)),
        "post_success": index > int(success_index),
    }
    events = {
        **event_values,
        "settle_rule": f"first frame of a run with pot_lid_qpos<0.02 for {settle_frames} consecutive states",
        "lid_removed_rule": "after_settle and pot_lid_qpos>=0.10",
        "meat_lifted_rule": "meat_z>=initial_meat_z+0.05",
        "in_pot_rule": "meat_pot_xy_distance<=0.10",
        "success_rule": "xy<0.10 and meat_z<robot0_base_z+0.10",
        "phase_semantics": (
            "multi_label; lid_move and grasp may overlap; place ends at first "
            "success and later states are reported separately as post_success"
        ),
        "phase_state_counts": {name: int(mask.sum()) for name, mask in masks.items()},
    }
    return masks, events


def _scalar_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.size == 0:
        raise ValueError(f"Invalid error arrays: {reference.shape} vs {candidate.shape}")
    error = candidate.astype(np.float64) - reference.astype(np.float64)
    return {
        "count": int(error.size),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "max_absolute_error": float(np.abs(error).max()),
    }


def action_error_report(
    expert: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    state_mask: np.ndarray,
    target_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Aggregate ``[T,N,H,8]`` chunks with query/target phase semantics.

    Immediate error is assigned by the query state ``phase[t]``. Multi-step
    errors are assigned by the predicted target action ``phase[t+h]``.
    """

    if expert.shape != prediction.shape or expert.ndim != 4 or expert.shape[-1] != ACTION_DIM:
        raise ValueError(f"Unexpected action tensors: {expert.shape} and {prediction.shape}")
    if valid.shape != expert.shape[:1] + expert.shape[2:3]:
        raise ValueError(f"Unexpected valid mask: {valid.shape}")
    state_mask = np.asarray(state_mask, dtype=bool)
    if state_mask.shape != (expert.shape[0],):
        raise ValueError(f"Unexpected state mask: {state_mask.shape}")
    if target_mask is None:
        target_mask = np.broadcast_to(state_mask[:, None], valid.shape)
    else:
        target_mask = np.asarray(target_mask, dtype=bool)
        if target_mask.shape != valid.shape:
            raise ValueError(f"Unexpected target mask: {target_mask.shape}")
    horizons = {
        "immediate": np.arange(expert.shape[2]) < 1,
        "prediction_horizon_5": np.arange(expert.shape[2]) < 5,
        "full_horizon": np.ones(expert.shape[2], dtype=bool),
    }
    report: dict[str, Any] = {}
    for horizon_name, horizon_mask in horizons.items():
        phase_selection = (
            state_mask[:, None] if horizon_name == "immediate" else target_mask
        )
        selected = valid & phase_selection & horizon_mask[None, :]
        sample_t, sample_h = np.nonzero(selected)
        if not len(sample_t):
            report[horizon_name] = None
            continue
        ref = expert[sample_t, :, sample_h, :]
        pred = prediction[sample_t, :, sample_h, :]
        item: dict[str, Any] = {
            "state_horizon_pairs": int(len(sample_t)),
            "overall": _scalar_error(ref.reshape(-1), pred.reshape(-1)),
            "by_agent": {},
            "by_group": {
                "arm": _scalar_error(ref[..., :7].reshape(-1), pred[..., :7].reshape(-1)),
                "gripper": _scalar_error(ref[..., 7].reshape(-1), pred[..., 7].reshape(-1)),
            },
            "by_agent_and_group": {},
            "by_dimension": {},
        }
        for agent in range(ref.shape[1]):
            agent_name = f"panda-{agent}"
            item["by_agent"][agent_name] = _scalar_error(
                ref[:, agent].reshape(-1), pred[:, agent].reshape(-1)
            )
            item["by_agent_and_group"][f"{agent_name}/arm"] = _scalar_error(
                ref[:, agent, :7].reshape(-1), pred[:, agent, :7].reshape(-1)
            )
            item["by_agent_and_group"][f"{agent_name}/gripper"] = _scalar_error(
                ref[:, agent, 7].reshape(-1), pred[:, agent, 7].reshape(-1)
            )
        for dimension in range(ACTION_DIM):
            item["by_dimension"][str(dimension)] = _scalar_error(
                ref[..., dimension].reshape(-1), pred[..., dimension].reshape(-1)
            )
        arm_ref = ref[..., :7].reshape(-1, 7)
        arm_pred = pred[..., :7].reshape(-1, 7)
        denominator = np.linalg.norm(arm_ref, axis=1) * np.linalg.norm(arm_pred, axis=1)
        usable = denominator > 0
        item["arm_vector_cosine_mean"] = (
            None
            if not usable.any()
            else float(np.mean(np.sum(arm_ref[usable] * arm_pred[usable], axis=1) / denominator[usable]))
        )
        item["gripper_sign_agreement"] = float(
            np.mean(np.sign(ref[..., 7]) == np.sign(pred[..., 7]))
        )
        report[horizon_name] = item
    return report


def run_first_frame_parity(
    *,
    args: argparse.Namespace,
    episode: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    observations: Mapping[str, np.ndarray],
    policy: FastWAMMultiRobotPolicy,
    output_dir: Path,
) -> dict[str, Any]:
    """Compare cache/stored/live extractors on the untouched raw H5 state 0.

    This runs in a dedicated environment and cannot feed persisted expert RGB
    into any closed-loop action query.
    """

    from fastwam.datasets.gaussian_cache.provider import GaussianCache

    env = _build_environment(args.robofactory_root, args.task)
    try:
        _reset_environment(env, episode)
        env.unwrapped.set_state_dict(states[0])
        agent_names = tuple(episode["agent_names"])
        stored_obs = _stored_observation(observations, agent_names, 0)
        live_obs = env.unwrapped.get_obs()
        live_state = env.unwrapped.get_state_dict()
        prepared_stored = prepare_observation(
            stored_obs,
            states[0],
            policy.stats,
            allowed_agent_counts=policy.allowed_agent_counts,
        )
        prepared_live = prepare_observation(
            live_obs,
            live_state,
            policy.stats,
            allowed_agent_counts=policy.allowed_agent_counts,
        )
        online_from_stored = encode_compact_agent_gaussian(
            policy.teacher, prepared_stored
        ).cpu()
        online_from_live = encode_compact_agent_gaussian(
            policy.teacher, prepared_live
        ).cpu()
        with GaussianCache.open(args.gaussian_cache, verify="none") as cache:
            cached = torch.as_tensor(
                cache.get_agents(
                    str(episode["source_path"]),
                    str(episode["trajectory"]),
                    0,
                    agent_names,
                )["agent_gaussian"]
            ).cpu()
        np.savez_compressed(
            output_dir / "first_frame_gaussians.npz",
            cached=cached.numpy(),
            online_from_stored_rgb=online_from_stored.numpy(),
            online_from_live_rerender=online_from_live.numpy(),
        )
        rgb_payload: dict[str, np.ndarray] = {}
        for camera in (
            "head_camera_agent0",
            "head_camera_agent1",
            "head_camera_global",
        ):
            rgb_payload[f"stored_{camera}"] = _camera_array(stored_obs, camera)
            rgb_payload[f"live_{camera}"] = _camera_array(live_obs, camera)
        np.savez_compressed(output_dir / "first_frame_rgb.npz", **rgb_payload)
        report = {
            "reference_state": "raw_h5_t0_unsanitized",
            "closed_loop_policy_input": False,
            "comparison_semantics": {
                "cached_vs_online_stored_rgb": (
                    "pure extractor parity on exactly the persisted H5 RGB"
                ),
                "cached_vs_online_live_rerender": (
                    "deployment/render difference after restoring raw H5 state 0"
                ),
                "online_stored_vs_online_live": (
                    "stored-vs-rerender difference with the same online extractor"
                ),
            },
            "cached_vs_online_stored_rgb": gaussian_error_report(
                cached, online_from_stored
            ),
            "cached_vs_online_live_rerender": gaussian_error_report(
                cached, online_from_live
            ),
            "online_stored_vs_online_live": gaussian_error_report(
                online_from_stored, online_from_live
            ),
            "stored_rgb_vs_live_rerender": rgb_pair_error_report(
                stored_obs, live_obs
            ),
            "artifacts": {
                "gaussians": "first_frame_gaussians.npz",
                "rgb": "first_frame_rgb.npz",
            },
        }
        _atomic_json(output_dir / "first_frame_gaussian_parity.json", report)
        return report
    finally:
        env.close()


def run_rollout(
    *,
    args: argparse.Namespace,
    episode: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    policy: FastWAMMultiRobotPolicy,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=False)
    env = _build_environment(args.robofactory_root, args.task)
    multiview_writer = None
    global_writer = None
    video_stage = tempfile.TemporaryDirectory(prefix="fastwam-rollout-video-")
    video_stage_dir = Path(video_stage.name)
    multiview_stage_path = video_stage_dir / "rollout_multiview.mp4"
    global_stage_path = video_stage_dir / "rollout_global.mp4"
    violations: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    try:
        _reset_environment(env, episode)
        env.unwrapped.set_state_dict(states[0])
        agent_names = tuple(episode["agent_names"])
        initial_audit: dict[str, Any]
        if args.initial_state == "clean":
            initial_audit = sanitize_initial_pot_lid(env)
        else:
            initial_audit = {
                "mode": "raw_h5_t0",
                "mutation_api": None,
                "before": {
                    "qpos": _flat(env.unwrapped.pot.get_qpos()),
                    "qvel": _flat(env.unwrapped.pot.get_qvel()),
                },
                "after": {
                    "qpos": _flat(env.unwrapped.pot.get_qpos()),
                    "qvel": _flat(env.unwrapped.pot.get_qvel()),
                },
                "other_state_verified_unchanged": True,
            }
        _atomic_json(output_dir / "initial_state_audit.json", initial_audit)
        # Keep declared artifacts present even when a successful initial state
        # or zero bound violations would otherwise leave no JSONL records.
        (output_dir / "policy_queries.jsonl").touch(exist_ok=False)
        (output_dir / "action_bound_violations.jsonl").touch(exist_ok=False)
        policy.start_episode(args.policy_seed)
        # Closed loop is live-only.  No persisted expert observation is ever
        # passed to the policy in this function.
        online_obs = env.unwrapped.get_obs()
        policy.update_obs(online_obs, env.unwrapped.get_state_dict())
        multiview_writer = imageio.get_writer(
            multiview_stage_path,
            fps=FPS,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        global_writer = imageio.get_writer(
            global_stage_path,
            fps=FPS,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )

        def record_frame() -> None:
            frame = _video_frame(env)
            multiview_writer.append_data(frame)
            # env.render() tiles agent0, agent1, global in that order.
            global_writer.append_data(frame[:, -320:, :])

        record_frame()
        first = physical_snapshot(env, simulator_step=0)
        snapshots.append(first)
        _append_jsonl(output_dir / "physical_trace.jsonl", first)
        steps = 0
        queries = 0
        success = bool(first["task_success_from_geometry"])
        termination_reason = "initial_success" if success else "max_steps"
        max_steps = min(int(args.max_steps), int(episode["max_episode_steps"]))
        while steps < max_steps and not success:
            trace = policy.get_action_trace()
            query_record = {
                "query_index": trace["query_index"],
                "simulator_step_before_query": steps,
                "inference_seed": trace["inference_seed"],
                "agent_names": trace["agent_names"],
                "normalized_action": trace["normalized_action"],
                "denormalized_action": trace["denormalized_action"],
                "flat_action": trace["flat_action"],
                "planned_exec_horizon": int(args.exec_horizon),
                "observation_source": "live_rerender_only",
            }
            queries += 1
            action_chunk = np.asarray(trace["flat_action"], dtype=np.float32)
            execute = min(args.exec_horizon, len(action_chunk), max_steps - steps)
            executed = 0
            stop = False
            for chunk_offset in range(execute):
                action = _flat_action_to_dict(action_chunk[chunk_offset], agent_names)
                current = action_bound_records(
                    action,
                    env.action_space,
                    simulator_step=steps,
                    query_index=int(trace["query_index"]),
                    chunk_offset=chunk_offset,
                )
                violations.extend(current)
                for record in current:
                    _append_jsonl(output_dir / "action_bound_violations.jsonl", record)
                _, _, terminated, truncated, info = env.step(action)
                policy.record_action(action_chunk[chunk_offset])
                steps += 1
                executed += 1
                record_frame()
                snapshot = physical_snapshot(env, simulator_step=steps, last_action=action)
                snapshot["info_success"] = _as_bool(info["success"], label="info.success")
                snapshots.append(snapshot)
                _append_jsonl(output_dir / "physical_trace.jsonl", snapshot)
                success = bool(snapshot["info_success"])
                online_obs = env.unwrapped.get_obs()
                policy.update_obs(online_obs, env.unwrapped.get_state_dict())
                if success:
                    termination_reason = "success"
                    stop = True
                    break
                if _as_bool(terminated, label="terminated"):
                    termination_reason = "terminated"
                    stop = True
                    break
                if _as_bool(truncated, label="truncated"):
                    termination_reason = "truncated"
                    stop = True
                    break
            query_record["actual_executed_actions"] = executed
            _append_jsonl(output_dir / "policy_queries.jsonl", query_record)
            if stop:
                break
        multiview_writer.close()
        multiview_writer = None
        global_writer.close()
        global_writer = None
        video_integrity = {
            "multiview_mp4": _publish_video(
                multiview_stage_path,
                output_dir / "rollout_multiview.mp4",
                expected_frames=len(snapshots),
            ),
            "global_mp4": _publish_video(
                global_stage_path,
                output_dir / "rollout_global.mp4",
                expected_frames=len(snapshots),
            ),
        }
        buckets = bucket_bound_violations(violations)
        _atomic_json(output_dir / "action_bound_buckets.json", buckets)
        grasp_metrics = rollout_grasp_metrics(snapshots)
        result = {
            "status": "completed",
            "success": success,
            "steps": steps,
            "recorded_video_frames": len(snapshots),
            "policy_queries": queries,
            "termination_reason": termination_reason,
            "initial_state": args.initial_state,
            "exec_horizon": int(args.exec_horizon),
            "observation_source": "live_rerender_only",
            "persisted_expert_rgb_used_for_policy": False,
            **grasp_metrics,
            "initial_state_audit": initial_audit,
            "bound_violations": buckets,
            "video_integrity": video_integrity,
            "elapsed_seconds": time.monotonic() - started,
            "artifacts": {
                "multiview_mp4": "rollout_multiview.mp4",
                "global_mp4": "rollout_global.mp4",
                "queries": "policy_queries.jsonl",
                "physical_trace": "physical_trace.jsonl",
                "violations": "action_bound_violations.jsonl",
                "initial_state_audit": "initial_state_audit.json",
            },
        }
        _atomic_json(output_dir / "rollout_result.json", result)
        return result
    finally:
        if multiview_writer is not None:
            multiview_writer.close()
        if global_writer is not None:
            global_writer.close()
        video_stage.cleanup()
        env.close()


def run_expert_replay(
    *,
    args: argparse.Namespace,
    episode: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    actions: Mapping[str, np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    """Replay the complete stored two-arm action stream without a policy."""

    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=False)
    env = _build_environment(args.robofactory_root, args.task)
    multiview_writer = None
    global_writer = None
    local_stage = tempfile.TemporaryDirectory(prefix="fastwam-expert-replay-")
    local_stage_dir = Path(local_stage.name)
    multiview_stage_path = local_stage_dir / "expert_replay_multiview.mp4"
    global_stage_path = local_stage_dir / "expert_replay_global.mp4"
    actions_stage_path = local_stage_dir / "expert_actions.jsonl"
    physical_trace_stage_path = local_stage_dir / "physical_trace.jsonl"
    violations_stage_path = local_stage_dir / "action_bound_violations.jsonl"
    violations: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    try:
        _reset_environment(env, episode)
        env.unwrapped.set_state_dict(states[0])
        agent_names = tuple(episode["agent_names"])
        action_count = next(iter(actions.values())).shape[0]
        initial_audit = {
            "mode": "raw_h5_t0",
            "mutation_api": None,
            "before": {
                "qpos": _flat(env.unwrapped.pot.get_qpos()),
                "qvel": _flat(env.unwrapped.pot.get_qvel()),
            },
            "after": {
                "qpos": _flat(env.unwrapped.pot.get_qpos()),
                "qvel": _flat(env.unwrapped.pot.get_qvel()),
            },
            "other_state_verified_unchanged": True,
        }
        _atomic_json(output_dir / "initial_state_audit.json", initial_audit)
        actions_stage_path.touch(exist_ok=False)
        physical_trace_stage_path.touch(exist_ok=False)
        violations_stage_path.touch(exist_ok=False)
        multiview_writer = imageio.get_writer(
            multiview_stage_path,
            fps=FPS,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        global_writer = imageio.get_writer(
            global_stage_path,
            fps=FPS,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )

        def record_frame() -> None:
            frame = _video_frame(env)
            multiview_writer.append_data(frame)
            global_writer.append_data(frame[:, -320:, :])

        record_frame()
        first = physical_snapshot(env, simulator_step=0)
        snapshots.append(first)
        _append_jsonl(physical_trace_stage_path, first)
        steps = 0
        success = bool(first["task_success_from_geometry"])
        termination_reason = "initial_success" if success else "expert_actions_exhausted"
        max_steps = min(
            int(args.max_steps),
            int(episode["max_episode_steps"]),
            int(action_count),
        )
        while steps < max_steps and not success:
            action = _expert_action_at(actions, agent_names, steps)
            current = action_bound_records(
                action,
                env.action_space,
                simulator_step=steps,
                query_index=-1,
                chunk_offset=0,
            )
            violations.extend(current)
            for record in current:
                _append_jsonl(violations_stage_path, record)
            _append_jsonl(
                actions_stage_path,
                {
                    "simulator_step": steps,
                    "source_timestep": steps,
                    "action": action,
                },
            )
            _, _, terminated, truncated, info = env.step(action)
            steps += 1
            record_frame()
            snapshot = physical_snapshot(env, simulator_step=steps, last_action=action)
            snapshot["info_success"] = _as_bool(info["success"], label="info.success")
            snapshots.append(snapshot)
            _append_jsonl(physical_trace_stage_path, snapshot)
            success = bool(snapshot["info_success"])
            if success:
                termination_reason = "success"
                break
            if _as_bool(terminated, label="terminated"):
                termination_reason = "terminated"
                break
            if _as_bool(truncated, label="truncated"):
                termination_reason = "truncated"
                break
        multiview_writer.close()
        multiview_writer = None
        global_writer.close()
        global_writer = None
        jsonl_integrity = {
            "actions": _publish_staged_file(
                actions_stage_path, output_dir / "expert_actions.jsonl"
            ),
            "physical_trace": _publish_staged_file(
                physical_trace_stage_path, output_dir / "physical_trace.jsonl"
            ),
            "violations": _publish_staged_file(
                violations_stage_path,
                output_dir / "action_bound_violations.jsonl",
            ),
        }
        video_integrity = {
            "multiview_mp4": _publish_video(
                multiview_stage_path,
                output_dir / "expert_replay_multiview.mp4",
                expected_frames=len(snapshots),
            ),
            "global_mp4": _publish_video(
                global_stage_path,
                output_dir / "expert_replay_global.mp4",
                expected_frames=len(snapshots),
            ),
        }
        buckets = bucket_bound_violations(violations)
        _atomic_json(output_dir / "action_bound_buckets.json", buckets)
        initial_height = float(snapshots[0]["meat_height"])
        max_height = max(float(snapshot["meat_height"]) for snapshot in snapshots)
        result = {
            "status": "completed",
            "success": success,
            "steps": steps,
            "expert_actions_available": int(action_count),
            "expert_actions_executed": steps,
            "action_source": "stored_h5_expert",
            "policy_initialized": False,
            "recorded_video_frames": len(snapshots),
            "termination_reason": termination_reason,
            "initial_state": "raw",
            "initial_state_audit": initial_audit,
            "meat_initial_height_m": initial_height,
            "meat_max_height_m": max_height,
            "meat_max_lift_m": max_height - initial_height,
            "pot_lid_max_qpos": max(
                float(snapshot["pot_lid_qpos"]) for snapshot in snapshots
            ),
            "bound_violations": buckets,
            "jsonl_integrity": jsonl_integrity,
            "video_integrity": video_integrity,
            "elapsed_seconds": time.monotonic() - started,
            "artifacts": {
                "multiview_mp4": "expert_replay_multiview.mp4",
                "global_mp4": "expert_replay_global.mp4",
                "actions": "expert_actions.jsonl",
                "physical_trace": "physical_trace.jsonl",
                "violations": "action_bound_violations.jsonl",
                "initial_state_audit": "initial_state_audit.json",
            },
        }
        _atomic_json(output_dir / "expert_replay_result.json", result)
        return result
    finally:
        if multiview_writer is not None:
            multiview_writer.close()
        if global_writer is not None:
            global_writer.close()
        local_stage.cleanup()
        env.close()


def run_teacher_forcing(
    *,
    args: argparse.Namespace,
    episode: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    actions: Mapping[str, np.ndarray],
    observations: Mapping[str, np.ndarray],
    policy: FastWAMMultiRobotPolicy,
    output_dir: Path,
) -> dict[str, Any]:
    """Paired stored/live teacher forcing over clean expert states t=5..267.

    Each pair restores the same serialized state and restarts the policy with
    the same seed, so the only intended input difference is persisted RGB
    versus a live rerender.  Predictions and Gaussian tensors are retained in
    full for offline aggregation.
    """

    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=False)
    env = _build_environment(args.robofactory_root, args.task)
    local_stage = tempfile.TemporaryDirectory(prefix="fastwam-teacher-forcing-")
    local_stage_dir = Path(local_stage.name)
    physical_trace_stage_path = local_stage_dir / "expert_physical_trace.jsonl"
    states_stage_path = local_stage_dir / "teacher_forcing_states.jsonl"
    first_pair_rgb_stage_path = local_stage_dir / "teacher_first_pair_rgb.npz"
    actions_stage_path = local_stage_dir / "teacher_forcing_actions.npz"
    gaussians_stage_path = local_stage_dir / "teacher_forcing_gaussians.npz"
    phase_summary_stage_path = local_stage_dir / "phase_action_error_summary.json"
    try:
        _reset_environment(env, episode)
        agent_names = tuple(episode["agent_names"])
        action_count = next(iter(actions.values())).shape[0]
        timesteps, inference_seeds = teacher_state_schedule(
            action_count=action_count,
            start_timestep=int(args.teacher_start_timestep),
            max_states=int(args.max_teacher_states),
            base_seed=int(args.policy_seed),
        )
        count = len(timesteps)
        horizon = int(args.action_horizon)
        n_agents = len(agent_names)
        shape = (count, n_agents, horizon, ACTION_DIM)
        stored_norm = np.full(shape, np.nan, np.float32)
        stored_denorm = np.full_like(stored_norm, np.nan)
        live_norm = np.full_like(stored_norm, np.nan)
        live_denorm = np.full_like(stored_norm, np.nan)
        expert_denorm = np.full_like(stored_norm, np.nan)
        expert_norm = np.full_like(stored_norm, np.nan)
        valid = np.zeros((count, horizon), dtype=bool)
        if args.formal_contract:
            expected_valid = np.zeros_like(valid)
            for row, timestep in enumerate(timesteps.tolist()):
                length = teacher_target_length(
                    action_count=action_count,
                    timestep=int(timestep),
                    horizon=horizon,
                    formal_contract=True,
                )
                expected_valid[row, :length] = True
            # Fail before expensive paired inference if the requested matrix is
            # not the complete fixed formal contract.
            validate_formal_teacher_contract(
                timesteps=timesteps,
                valid=expected_valid,
                action_horizon=horizon,
            )
        stored_gaussians: list[np.ndarray] = []
        live_gaussians: list[np.ndarray] = []
        snapshots: list[dict[str, Any]] = []
        physical_trace_stage_path.touch(exist_ok=False)
        states_stage_path.touch(exist_ok=False)
        # Derive phase boundaries from the complete expert state sequence even
        # during a bounded model smoke test.  This keeps the contamination-safe
        # lid-settling rule and phase definitions identical between smoke and
        # formal runs.
        for timestep, state in enumerate(states):
            env.unwrapped.set_state_dict(state)
            snapshot = physical_snapshot(env, simulator_step=timestep)
            snapshots.append(snapshot)
            _append_jsonl(physical_trace_stage_path, snapshot)
        masks, events = phase_masks(snapshots)
        expected_start = int(args.teacher_start_timestep)
        if int(events["lid_settled"]) != expected_start:
            raise RuntimeError(
                "Teacher start must equal the first contamination-safe five-state "
                f"lid settlement: configured={expected_start}, observed={events['lid_settled']}"
            )
        mean = policy.stats.action_mean.detach().cpu().numpy().reshape(1, 1, ACTION_DIM)
        std = policy.stats.action_std.detach().cpu().numpy().reshape(1, 1, ACTION_DIM)
        for row, (timestep_value, seed_value) in enumerate(
            zip(timesteps.tolist(), inference_seeds.tolist(), strict=True)
        ):
            timestep = int(timestep_value)
            inference_seed = int(seed_value)
            env.unwrapped.set_state_dict(states[timestep])
            live_obs = env.unwrapped.get_obs()
            stored_obs = _stored_rgb_on_live_observation(
                live_obs, observations, agent_names, timestep
            )
            policy.start_episode(inference_seed)
            policy.update_obs(stored_obs, states[timestep])
            stored_trace = policy.get_action_trace()

            policy.start_episode(inference_seed)
            # Use the exact same serialized env_state as the stored branch;
            # this isolates stored/live observation rendering.
            policy.update_obs(live_obs, states[timestep])
            live_trace = policy.get_action_trace()
            for source_name, trace in (
                ("stored", stored_trace),
                ("live", live_trace),
            ):
                if int(trace["query_index"]) != 0 or int(trace["inference_seed"]) != inference_seed:
                    raise RuntimeError(
                        f"{source_name} pair seed/query mismatch at t={timestep}: "
                        f"seed={trace['inference_seed']} query={trace['query_index']}"
                    )
            stored_norm[row] = np.asarray(stored_trace["normalized_action"])
            stored_denorm[row] = np.asarray(stored_trace["denormalized_action"])
            live_norm[row] = np.asarray(live_trace["normalized_action"])
            live_denorm[row] = np.asarray(live_trace["denormalized_action"])
            stored_gaussians.append(np.asarray(stored_trace["agent_gaussian"]))
            live_gaussians.append(np.asarray(live_trace["agent_gaussian"]))
            length = teacher_target_length(
                action_count=action_count,
                timestep=timestep,
                horizon=horizon,
                formal_contract=bool(args.formal_contract),
            )
            chunk = np.stack(
                [actions[name][timestep : timestep + length] for name in agent_names],
                axis=0,
            )
            expert_denorm[row, :, :length] = chunk
            expert_norm[row, :, :length] = (chunk - mean) / std
            valid[row, :length] = True
            _append_jsonl(
                states_stage_path,
                {
                    "timestep": timestep,
                    "inference_seed": inference_seed,
                    "pair_contract": "same_serialized_state_same_seed_query0",
                    "valid_future_horizon": length,
                    "rgb_stored_vs_live": rgb_pair_error_report(stored_obs, live_obs),
                    "gaussian_stored_vs_live": gaussian_error_report(
                        stored_trace["agent_gaussian"], live_trace["agent_gaussian"]
                    ),
                    "stored_immediate_denormalized_mae": float(
                        np.abs(stored_denorm[row, :, 0] - chunk[:, 0]).mean()
                    ),
                    "live_immediate_denormalized_mae": float(
                        np.abs(live_denorm[row, :, 0] - chunk[:, 0]).mean()
                    ),
                    "stored_vs_live_immediate_denormalized_mae": float(
                        np.abs(stored_denorm[row, :, 0] - live_denorm[row, :, 0]).mean()
                    ),
                    "expert_denormalized_action": chunk,
                    "expert_normalized_action": expert_norm[row, :, :length],
                },
            )
            if row == 0:
                np.savez_compressed(
                    first_pair_rgb_stage_path,
                    **{
                        f"stored_{camera}": _camera_array(stored_obs, camera)
                        for camera in (
                            "head_camera_agent0",
                            "head_camera_agent1",
                            "head_camera_global",
                        )
                    },
                    **{
                        f"live_{camera}": _camera_array(live_obs, camera)
                        for camera in (
                            "head_camera_agent0",
                            "head_camera_agent1",
                            "head_camera_global",
                        )
                    },
                )
            if (row + 1) % 20 == 0 or row + 1 == count:
                print(
                    json.dumps(
                        {
                            "event": "teacher_forcing_progress",
                            "completed": row + 1,
                            "total": count,
                        }
                    ),
                    flush=True,
                )
        # The trajectory has one final state without a corresponding action.
        phase_summary: dict[str, Any] = {
            "events": events,
            "teacher_start_timestep": expected_start,
            "teacher_end_timestep_inclusive": int(timesteps[-1]),
            "pair_contract": "same_serialized_state_same_seed_query0",
            "seed_rule": "policy_seed + absolute_timestep",
            "phase_mask_semantics": {
                "immediate": "query state phase[t]",
                "prediction_horizon_5": "target action phase[t+h]",
                "full_horizon": "target action phase[t+h]",
            },
            "sources": {},
            "stored_vs_live": {},
        }
        all_mask = np.ones(count, dtype=bool)
        selected_masks = {name: mask[timesteps] for name, mask in masks.items()}
        target_masks = {
            name: target_action_phase_mask(mask, timesteps, horizon)
            for name, mask in masks.items()
        }
        masks_with_all = {
            "all_states": (all_mask, np.ones_like(valid)),
            **{
                name: (selected_masks[name], target_masks[name])
                for name in selected_masks
            },
        }
        for name, (state_mask, target_mask) in masks_with_all.items():
            phase_summary["sources"][name] = {
                "stored_normalized_vs_expert": action_error_report(
                    expert_norm, stored_norm, valid, state_mask, target_mask
                ),
                "stored_denormalized_vs_expert": action_error_report(
                    expert_denorm, stored_denorm, valid, state_mask, target_mask
                ),
                "live_normalized_vs_expert": action_error_report(
                    expert_norm, live_norm, valid, state_mask, target_mask
                ),
                "live_denormalized_vs_expert": action_error_report(
                    expert_denorm, live_denorm, valid, state_mask, target_mask
                ),
            }
            phase_summary["stored_vs_live"][name] = {
                "normalized": action_error_report(
                    stored_norm, live_norm, valid, state_mask, target_mask
                ),
                "denormalized": action_error_report(
                    stored_denorm, live_denorm, valid, state_mask, target_mask
                ),
            }
        formal_contract = None
        if args.formal_contract:
            formal_contract = validate_formal_teacher_contract(
                timesteps=timesteps,
                valid=valid,
                action_horizon=horizon,
            )
            phase_summary["formal_contract"] = formal_contract
        np.savez_compressed(
            actions_stage_path,
            timesteps=timesteps,
            inference_seeds=inference_seeds,
            stored_prediction_normalized=stored_norm,
            stored_prediction_denormalized=stored_denorm,
            live_prediction_normalized=live_norm,
            live_prediction_denormalized=live_denorm,
            expert_normalized=expert_norm,
            expert_denormalized=expert_denorm,
            valid_horizon=valid,
            **{
                f"phase_query_state_{name}": mask
                for name, mask in selected_masks.items()
            },
            **{
                f"phase_target_action_{name}": mask
                for name, mask in target_masks.items()
            },
        )
        np.savez_compressed(
            gaussians_stage_path,
            timesteps=timesteps,
            inference_seeds=inference_seeds,
            stored=np.stack(stored_gaussians),
            live=np.stack(live_gaussians),
        )
        _atomic_json(phase_summary_stage_path, phase_summary)
        artifact_integrity = {
            "actions": _publish_staged_file(
                actions_stage_path, output_dir / "teacher_forcing_actions.npz"
            ),
            "gaussians": _publish_staged_file(
                gaussians_stage_path, output_dir / "teacher_forcing_gaussians.npz"
            ),
            "first_pair_rgb": _publish_staged_file(
                first_pair_rgb_stage_path, output_dir / "teacher_first_pair_rgb.npz"
            ),
            "per_state": _publish_staged_file(
                states_stage_path, output_dir / "teacher_forcing_states.jsonl"
            ),
            "physical_trace": _publish_staged_file(
                physical_trace_stage_path, output_dir / "expert_physical_trace.jsonl"
            ),
            "phase_summary": _publish_staged_file(
                phase_summary_stage_path,
                output_dir / "phase_action_error_summary.json",
            ),
        }
        result = {
            "status": "completed",
            "states_evaluated": count,
            "first_timestep": int(timesteps[0]),
            "last_timestep": int(timesteps[-1]),
            "expert_action_count": action_count,
            "observation_sources": ["stored", "live"],
            "pair_contract": "same_serialized_state_same_seed_query0",
            "phase_mask_semantics": phase_summary["phase_mask_semantics"],
            "formal_contract": formal_contract,
            "phase_events": events,
            "artifact_integrity": artifact_integrity,
            "elapsed_seconds": time.monotonic() - started,
            "artifacts": {
                "actions": "teacher_forcing_actions.npz",
                "gaussians": "teacher_forcing_gaussians.npz",
                "first_pair_rgb": "teacher_first_pair_rgb.npz",
                "per_state": "teacher_forcing_states.jsonl",
                "physical_trace": "expert_physical_trace.jsonl",
                "phase_summary": "phase_action_error_summary.json",
            },
        }
        _atomic_json(output_dir / "teacher_forcing_result.json", result)
        return result
    finally:
        local_stage.cleanup()
        env.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("rollout", "teacher-forcing", "parity", "expert-replay", "all"),
        default="all",
        help=(
            "Use rollout for one independently schedulable raw/clean x h1/h5 cell; "
            "expert-replay executes stored H5 actions without loading FastWAM."
        ),
    )
    parser.add_argument(
        "--formal-contract",
        action="store_true",
        help="Fail closed unless the full fixed diagnostic contract is satisfied.",
    )
    parser.add_argument("--task", default="PlaceFood-rf", choices=tuple(TASK_CONFIGS))
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--robofactory-root", type=Path, required=True)
    parser.add_argument("--gaussian-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-teacher-states", type=int, default=263)
    parser.add_argument("--teacher-start-timestep", type=int, default=5)
    parser.add_argument("--initial-state", choices=("raw", "clean"), default=None)
    parser.add_argument("--exec-horizon", type=int, choices=(1, 5), default=None)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--training-code-commit",
        required=False,
        help="Git commit used to train the selected checkpoint",
    )
    parser.add_argument(
        "--evaluation-code-commit",
        required=False,
        help="Git commit of the evaluator used for a formal expert replay",
    )
    parser.add_argument(
        "--integrity-mode",
        choices=("metadata_no_hash", "sha256"),
        default="metadata_no_hash",
        help="R5 uses ordinary file metadata; sha256 is retained only for legacy evaluation.",
    )
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--stats-sha256")
    parser.add_argument("--context-cache-dir", type=Path)
    parser.add_argument("--context-file", type=Path)
    parser.add_argument("--model-cache-root", type=Path)
    parser.add_argument(
        "--model-project-root",
        type=Path,
        help="FastWAM source/config checkout used to instantiate the checkpoint architecture",
    )
    parser.add_argument(
        "--action-architecture",
        choices=("pooled_v1", "gaussian_spatial_v2"),
        default="pooled_v1",
    )
    parser.add_argument("--policy-lightning-repo", type=Path)
    parser.add_argument("--policy-lightning-commit", default=POLICY_LIGHTNING_COMMIT)
    parser.add_argument("--noposplat-checkpoint", type=Path)
    parser.add_argument("--noposplat-checkpoint-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:0")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--sigma-shift", type=float)
    return parser


def main() -> None:
    args = _parser().parse_args()
    policy_needed = args.mode != "expert-replay"
    args.initial_state_explicit = args.initial_state is not None
    args.exec_horizon_explicit = args.exec_horizon is not None
    if args.initial_state is None:
        args.initial_state = "clean"
    if args.exec_horizon is None:
        args.exec_horizon = 5
    if args.task != "PlaceFood-rf":
        raise ValueError("This diagnostic intentionally supports PlaceFood-rf only")
    if policy_needed:
        required_policy_arguments = {
            "checkpoint": args.checkpoint,
            "training_code_commit": args.training_code_commit,
            "stats": args.stats,
            "gaussian_cache": args.gaussian_cache,
            "model_cache_root": args.model_cache_root,
            "policy_lightning_repo": args.policy_lightning_repo,
            "noposplat_checkpoint": args.noposplat_checkpoint,
        }
        missing_policy_arguments = sorted(
            name for name, value in required_policy_arguments.items() if value is None
        )
        if missing_policy_arguments:
            raise ValueError(
                "Policy mode requires arguments: "
                + ", ".join(missing_policy_arguments)
            )
        if (
            args.action_architecture == "gaussian_spatial_v2"
            and args.model_project_root is None
        ):
            raise ValueError(
                "gaussian_spatial_v2 requires an explicit --model-project-root"
            )
    if policy_needed and args.integrity_mode == "metadata_no_hash":
        if args.context_file is None:
            raise ValueError("metadata_no_hash mode requires --context-file")
        if args.context_cache_dir is not None:
            raise ValueError(
                "metadata_no_hash mode accepts the explicit --context-file, not --context-cache-dir"
            )
        forbidden_hashes = {
            "checkpoint_sha256": args.checkpoint_sha256,
            "stats_sha256": args.stats_sha256,
            "noposplat_checkpoint_sha256": args.noposplat_checkpoint_sha256,
        }
        supplied = sorted(name for name, value in forbidden_hashes.items() if value)
        if supplied:
            raise ValueError(
                "metadata_no_hash mode forbids hash arguments: " + ", ".join(supplied)
            )
    elif policy_needed:
        if args.context_cache_dir is None:
            raise ValueError("sha256 mode requires --context-cache-dir")
        if args.context_file is not None:
            raise ValueError("sha256 mode does not accept --context-file")
        required_hashes = {
            "checkpoint_sha256": args.checkpoint_sha256,
            "stats_sha256": args.stats_sha256,
            "noposplat_checkpoint_sha256": args.noposplat_checkpoint_sha256,
        }
        missing = sorted(name for name, value in required_hashes.items() if not value)
        if missing:
            raise ValueError("sha256 mode requires: " + ", ".join(missing))
    for name in (
        "max_steps",
        "max_teacher_states",
        "exec_horizon",
        "action_horizon",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    formal_rollout_contract = None
    formal_expert_replay_contract = None
    if args.formal_contract:
        if args.mode == "parity":
            raise ValueError("--formal-contract does not apply to parity-only mode")
        if args.mode in ("rollout", "all"):
            formal_rollout_contract = validate_formal_rollout_contract(
                max_steps=int(args.max_steps),
                initial_state=str(args.initial_state),
                exec_horizon=int(args.exec_horizon),
                initial_state_explicit=bool(args.initial_state_explicit),
                exec_horizon_explicit=bool(args.exec_horizon_explicit),
            )
        if args.mode in ("teacher-forcing", "all"):
            if int(args.teacher_start_timestep) != 5:
                raise ValueError("Formal teacher forcing requires start timestep 5")
            if int(args.max_teacher_states) != 263:
                raise ValueError("Formal teacher forcing requires 263 states")
            if int(args.action_horizon) < 5:
                raise ValueError("Formal teacher forcing requires action_horizon>=5")
        if args.mode == "expert-replay":
            formal_expert_replay_contract = validate_formal_expert_replay_contract(
                max_steps=int(args.max_steps),
                initial_state=str(args.initial_state),
                initial_state_explicit=bool(args.initial_state_explicit),
                evaluation_code_commit=args.evaluation_code_commit,
            )
    args.output_dir = args.output_dir.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    args.robofactory_root = args.robofactory_root.expanduser().resolve(strict=True)
    if policy_needed:
        args.gaussian_cache = args.gaussian_cache.expanduser().resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()
    started = time.monotonic()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "initializing",
        "started_at": started_at,
        "provenance_policy": "ordinary Git, run, path, timestamp, size, and version identifiers; no new artifact checksums",
        "base_eval_commit": "f89a7a5b7ca0674c78bca5f329398dfa28fb8758",
        "training_code_commit": (
            str(args.training_code_commit) if policy_needed else None
        ),
        "evaluation_code_commit": args.evaluation_code_commit,
        "robofactory_expected_commit": ROBOFACTORY_COMMIT,
        "runtime_code_path": str(Path(__file__).resolve().parents[2]),
        "python": {"executable": sys.executable, "version": sys.version, "platform": platform.platform()},
        "argv": sys.argv,
        "formal_contract_requested": bool(args.formal_contract),
        "formal_rollout_contract": formal_rollout_contract,
        "formal_expert_replay_contract": formal_expert_replay_contract,
        "policy_initialized": policy_needed,
    }
    _atomic_json(args.output_dir / "run_manifest.json", manifest)
    try:
        panel = _load_panel_nohash(args.panel)
        manifest["panel"] = {
            "path": str(args.panel.expanduser().resolve(strict=True)),
            "schema_version": str(panel["schema_version"]),
            "split": panel.get("split"),
            "split_seed": panel.get("split_seed"),
            "val_set_proportion": panel.get("val_set_proportion"),
            "split_key_scheme": panel.get("split_key_scheme"),
            "episode_count": len(panel["episodes"]),
        }
        episode = _selected_episodes(panel, args.task, args.episode_start, 1)[0]
        if panel["schema_version"] == SPLIT_PANEL_SCHEMA:
            expected_policy_seed = int(
                panel["paired_policy_seeds"][int(episode["panel_index"])]
            )
            if int(args.policy_seed) != expected_policy_seed:
                raise ValueError(
                    "Policy seed does not match the frozen split-panel pairing: "
                    f"got={args.policy_seed} expected={expected_policy_seed}"
                )
        source = _source_path(args.dataset_root, str(episode["source_path"]))
        if source.stat().st_size != int(episode["source_h5_bytes"]):
            raise ValueError(f"Source H5 byte-size mismatch: {source}")
        agent_names = tuple(episode["agent_names"])
        states, actions, observations = _load_episode_data(
            source, str(episode["trajectory"]), agent_names
        )
        action_count = next(iter(actions.values())).shape[0]
        if action_count < 1 or len(states) != action_count + 1:
            raise ValueError(
                "Held-out episode must contain one more state than action; "
                f"got actions={action_count} states={len(states)}"
            )
        manifest["episode"] = {
            "task": args.task,
            "policy_seed": int(args.policy_seed),
            "panel_index": int(episode["panel_index"]),
            "task_index": int(episode["task_index"]),
            "episode_id": int(episode["episode_id"]),
            "environment_seed": int(episode["episode_seed"]),
            "source_relative": str(episode["source_path"]),
            "source_absolute": str(source),
            "source_bytes": source.stat().st_size,
            "trajectory": str(episode["trajectory"]),
            "agent_names": agent_names,
            "expert_actions": next(iter(actions.values())).shape[0],
            "expert_states": len(states),
            "split": episode.get("split"),
            "global_ordinal": episode.get("global_ordinal"),
            "split_fraction": episode.get("split_fraction"),
        }
        policy = None
        if policy_needed:
            init_started = time.monotonic()
            policy = FastWAMMultiRobotPolicy(
                checkpoint_path=args.checkpoint,
                checkpoint_sha256=args.checkpoint_sha256,
                stats_path=args.stats,
                expected_stats_sha256=args.stats_sha256,
                context_cache_dir=args.context_cache_dir,
                context_path=args.context_file,
                task_name=args.task,
                model_cache_root=args.model_cache_root,
                policy_lightning_repo=args.policy_lightning_repo,
                policy_lightning_commit=args.policy_lightning_commit,
                noposplat_checkpoint_path=args.noposplat_checkpoint,
                noposplat_checkpoint_sha256=args.noposplat_checkpoint_sha256,
                integrity_mode=args.integrity_mode,
                allowed_agent_counts=(2,),
                device=args.device,
                teacher_device=args.teacher_device,
                action_horizon=args.action_horizon,
                num_inference_steps=args.num_inference_steps,
                sigma_shift=args.sigma_shift,
                seed=args.policy_seed,
                project_root=(
                    args.model_project_root
                    if args.model_project_root is not None
                    else Path(__file__).resolve().parents[2]
                ),
                action_architecture=args.action_architecture,
            )
            manifest["policy_init_seconds"] = time.monotonic() - init_started
        else:
            manifest["policy_init_seconds"] = 0.0
        manifest["status"] = "running"
        manifest["mode"] = args.mode
        manifest["rollout_cell"] = {
            "initial_state": args.initial_state,
            "exec_horizon": int(args.exec_horizon),
        }
        manifest["teacher_forcing"] = {
            "sources": ["stored", "live"],
            "start_timestep": int(args.teacher_start_timestep),
            "same_state_same_seed": True,
        }
        _atomic_json(args.output_dir / "run_manifest.json", manifest)
        rollout = None
        expert_replay = None
        teacher_forcing = None
        parity = None
        if args.mode in ("rollout", "all"):
            assert policy is not None
            rollout = run_rollout(
                args=args,
                episode=episode,
                states=states,
                policy=policy,
                output_dir=args.output_dir / "rollout",
            )
        if args.mode in ("teacher-forcing", "all"):
            assert policy is not None
            teacher_forcing = run_teacher_forcing(
                args=args,
                episode=episode,
                states=states,
                actions=actions,
                observations=observations,
                policy=policy,
                output_dir=args.output_dir / "teacher_forcing",
            )
        if args.mode in ("parity", "all"):
            assert policy is not None
            parity_dir = args.output_dir / "parity"
            parity_dir.mkdir(parents=True, exist_ok=False)
            parity = run_first_frame_parity(
                args=args,
                episode=episode,
                states=states,
                observations=observations,
                policy=policy,
                output_dir=parity_dir,
            )
        if args.mode == "expert-replay":
            expert_replay = run_expert_replay(
                args=args,
                episode=episode,
                states=states,
                actions=actions,
                output_dir=args.output_dir / "expert_replay",
            )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETED",
            "simulator_success": (
                bool(rollout["success"])
                if rollout is not None
                else (
                    None
                    if expert_replay is None
                    else bool(expert_replay["success"])
                )
            ),
            "rollout": rollout,
            "expert_replay": expert_replay,
            "teacher_forcing": teacher_forcing,
            "first_frame_parity": parity,
            "finished_at": _utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "gpu_memory": (
                None
                if not torch.cuda.is_available()
                else {
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            ),
        }
        _atomic_json(args.output_dir / "summary.json", summary)
        manifest["status"] = "terminal"
        manifest["finished_at"] = summary["finished_at"]
        manifest["summary_path"] = str(args.output_dir / "summary.json")
        _atomic_json(args.output_dir / "run_manifest.json", manifest)
        print(json.dumps(_json_value(summary), sort_keys=True), flush=True)
    except Exception as error:  # noqa: BLE001 - persist a terminal diagnostic record
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "INFRASTRUCTURE_ERROR",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "finished_at": _utc_now(),
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(args.output_dir / "summary.json", failure)
        manifest["status"] = "terminal_error"
        manifest["finished_at"] = failure["finished_at"]
        manifest["error_type"] = failure["error_type"]
        _atomic_json(args.output_dir / "run_manifest.json", manifest)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
