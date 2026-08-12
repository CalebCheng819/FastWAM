#!/usr/bin/env python3
"""Diagnose one fixed PlaceFood FastWAM rollout and expert trajectory.

The command is intentionally narrow: it binds the held-out panel episode to
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
import sys
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
    from .eval_robofactory_multi_robot import (
        ROBOFACTORY_COMMIT,
        STEP5000_CHECKPOINT_SHA256,
        TASK_CONFIGS,
        _as_bool,
        _build_environment,
        _flat_action_to_dict,
        _reset_environment,
        _selected_episodes,
        _source_path,
    )
    from .fastwam_multi_robot_policy import (
        NOPOSPLAT_CHECKPOINT_SHA256,
        POLICY_LIGHTNING_COMMIT,
        TRAINING_CODE_COMMIT,
        TRAINING_STATS_SHA256,
        FastWAMMultiRobotPolicy,
        encode_compact_agent_gaussian,
        prepare_observation,
    )
except ImportError:
    from eval_robofactory_multi_robot import (  # type: ignore[no-redef]
        ROBOFACTORY_COMMIT,
        STEP5000_CHECKPOINT_SHA256,
        TASK_CONFIGS,
        _as_bool,
        _build_environment,
        _flat_action_to_dict,
        _reset_environment,
        _selected_episodes,
        _source_path,
    )
    from fastwam_multi_robot_policy import (  # type: ignore[no-redef]
        NOPOSPLAT_CHECKPOINT_SHA256,
        POLICY_LIGHTNING_COMMIT,
        TRAINING_CODE_COMMIT,
        TRAINING_STATS_SHA256,
        FastWAMMultiRobotPolicy,
        encode_compact_agent_gaussian,
        prepare_observation,
    )


SCHEMA_VERSION = "fastwam-placefood-fixed-diagnostic-v1"
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


def _load_panel_nohash(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "fastwam-robofactory-heldout-panel-v1":
        raise ValueError(f"Unexpected held-out panel schema: {payload.get('schema_version')!r}")
    if not isinstance(payload.get("episodes"), list):
        raise TypeError("Held-out panel must contain an episodes list")
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
        "valid_pairs_h1": valid_h1,
        "valid_pairs_h5": valid_h5,
        "action_horizon": action_horizon,
    }


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
    base_z = float(_vector(unwrapped.agent.agents[0].robot.pose.p)[2])
    meat_pot_delta = meat - pot
    return {
        "simulator_step": int(simulator_step),
        "meat_position": meat,
        "pot_position": pot,
        "meat_pot_xy_distance": float(np.linalg.norm(meat_pot_delta[:2])),
        "meat_pot_xyz_distance": float(np.linalg.norm(meat_pot_delta)),
        "meat_height": float(meat[2]),
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
            output_dir / "rollout_multiview.mp4",
            fps=FPS,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        global_writer = imageio.get_writer(
            output_dir / "rollout_global.mp4",
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
        buckets = bucket_bound_violations(violations)
        _atomic_json(output_dir / "action_bound_buckets.json", buckets)
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
            "initial_state_audit": initial_audit,
            "bound_violations": buckets,
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
                expected_valid[row, : min(horizon, action_count - int(timestep))] = True
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
        # Derive phase boundaries from the complete expert state sequence even
        # during a bounded model smoke test.  This keeps the contamination-safe
        # lid-settling rule and phase definitions identical between smoke and
        # formal runs.
        for timestep, state in enumerate(states):
            env.unwrapped.set_state_dict(state)
            snapshot = physical_snapshot(env, simulator_step=timestep)
            snapshots.append(snapshot)
            _append_jsonl(output_dir / "expert_physical_trace.jsonl", snapshot)
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
            length = min(horizon, action_count - timestep)
            chunk = np.stack(
                [actions[name][timestep : timestep + length] for name in agent_names],
                axis=0,
            )
            expert_denorm[row, :, :length] = chunk
            expert_norm[row, :, :length] = (chunk - mean) / std
            valid[row, :length] = True
            _append_jsonl(
                output_dir / "teacher_forcing_states.jsonl",
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
                    output_dir / "teacher_first_pair_rgb.npz",
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
            output_dir / "teacher_forcing_actions.npz",
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
            output_dir / "teacher_forcing_gaussians.npz",
            timesteps=timesteps,
            inference_seeds=inference_seeds,
            stored=np.stack(stored_gaussians),
            live=np.stack(live_gaussians),
        )
        _atomic_json(output_dir / "phase_action_error_summary.json", phase_summary)
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
        env.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("rollout", "teacher-forcing", "parity", "all"),
        default="all",
        help="Use rollout for one independently schedulable raw/clean x h1/h5 cell.",
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
    parser.add_argument("--gaussian-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-teacher-states", type=int, default=263)
    parser.add_argument("--teacher-start-timestep", type=int, default=5)
    parser.add_argument("--initial-state", choices=("raw", "clean"), default=None)
    parser.add_argument("--exec-horizon", type=int, choices=(1, 5), default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default=STEP5000_CHECKPOINT_SHA256)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--stats-sha256", default=TRAINING_STATS_SHA256)
    parser.add_argument("--context-cache-dir", type=Path, required=True)
    parser.add_argument("--model-cache-root", type=Path, required=True)
    parser.add_argument("--policy-lightning-repo", type=Path, required=True)
    parser.add_argument("--policy-lightning-commit", default=POLICY_LIGHTNING_COMMIT)
    parser.add_argument("--noposplat-checkpoint", type=Path, required=True)
    parser.add_argument("--noposplat-checkpoint-sha256", default=NOPOSPLAT_CHECKPOINT_SHA256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:0")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--sigma-shift", type=float)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.initial_state_explicit = args.initial_state is not None
    args.exec_horizon_explicit = args.exec_horizon is not None
    if args.initial_state is None:
        args.initial_state = "clean"
    if args.exec_horizon is None:
        args.exec_horizon = 5
    if args.task != "PlaceFood-rf":
        raise ValueError("This diagnostic intentionally supports PlaceFood-rf only")
    for name in (
        "max_steps",
        "max_teacher_states",
        "exec_horizon",
        "action_horizon",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    formal_rollout_contract = None
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
    args.output_dir = args.output_dir.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    args.robofactory_root = args.robofactory_root.expanduser().resolve(strict=True)
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
        "training_code_commit": TRAINING_CODE_COMMIT,
        "robofactory_expected_commit": ROBOFACTORY_COMMIT,
        "runtime_code_path": str(Path(__file__).resolve().parents[2]),
        "python": {"executable": sys.executable, "version": sys.version, "platform": platform.platform()},
        "argv": sys.argv,
        "formal_contract_requested": bool(args.formal_contract),
        "formal_rollout_contract": formal_rollout_contract,
    }
    _atomic_json(args.output_dir / "run_manifest.json", manifest)
    try:
        panel = _load_panel_nohash(args.panel)
        episode = _selected_episodes(panel, args.task, args.episode_start, 1)[0]
        if int(episode["panel_index"]) != 0 or str(episode["trajectory"]) != "traj_61":
            raise ValueError(
                "This diagnostic is pinned to held-out panel_index=0, traj_61; "
                f"selected panel_index={episode['panel_index']} trajectory={episode['trajectory']}"
            )
        source = _source_path(args.dataset_root, str(episode["source_path"]))
        if source.stat().st_size != int(episode["source_h5_bytes"]):
            raise ValueError(f"Source H5 byte-size mismatch: {source}")
        agent_names = tuple(episode["agent_names"])
        states, actions, observations = _load_episode_data(
            source, str(episode["trajectory"]), agent_names
        )
        action_count = next(iter(actions.values())).shape[0]
        if action_count != 268 or len(states) != 269:
            raise ValueError(
                "Pinned traj_61 must contain exactly 268 actions and 269 states; "
                f"got actions={action_count} states={len(states)}"
            )
        manifest["episode"] = {
            "task": args.task,
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
        }
        init_started = time.monotonic()
        policy = FastWAMMultiRobotPolicy(
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            stats_path=args.stats,
            expected_stats_sha256=args.stats_sha256,
            context_cache_dir=args.context_cache_dir,
            task_name=args.task,
            model_cache_root=args.model_cache_root,
            policy_lightning_repo=args.policy_lightning_repo,
            policy_lightning_commit=args.policy_lightning_commit,
            noposplat_checkpoint_path=args.noposplat_checkpoint,
            noposplat_checkpoint_sha256=args.noposplat_checkpoint_sha256,
            device=args.device,
            teacher_device=args.teacher_device,
            action_horizon=args.action_horizon,
            num_inference_steps=args.num_inference_steps,
            sigma_shift=args.sigma_shift,
            seed=args.policy_seed,
        )
        manifest["policy_init_seconds"] = time.monotonic() - init_started
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
        teacher_forcing = None
        parity = None
        if args.mode in ("rollout", "all"):
            rollout = run_rollout(
                args=args,
                episode=episode,
                states=states,
                policy=policy,
                output_dir=args.output_dir / "rollout",
            )
        if args.mode in ("teacher-forcing", "all"):
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
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "rollout": rollout,
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
