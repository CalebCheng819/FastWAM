"""Normalize RoboFactory table HDF5 layouts across robot cardinalities.

The original FastWAM corpus stores multi-robot actions and states in groups and
uses one global plus one per-agent camera.  RoboFactory's single-robot table
tasks use flat action/state datasets and a single ``head_camera`` instead.  The
helpers in this module expose one fail-closed logical layout to the dataset,
statistics, selection, and Gaussian-cache pipelines.
"""

from __future__ import annotations

from collections.abc import Sequence

import h5py


SINGLE_AGENT_NAME = "panda-0"


def agent_sort_key(name: str) -> tuple[int, int | str]:
    try:
        return (0, int(str(name).rsplit("-", 1)[-1]))
    except ValueError:
        return (1, str(name))


def agent_names(trajectory: h5py.Group) -> tuple[str, ...]:
    """Return logical agent names for grouped or flat action storage."""

    if "actions" not in trajectory:
        return ()
    actions = trajectory["actions"]
    if isinstance(actions, h5py.Dataset):
        if actions.ndim != 2:
            raise ValueError(
                f"Flat RoboFactory actions must be [T,D], got {tuple(actions.shape)}"
            )
        return (SINGLE_AGENT_NAME,)
    if not isinstance(actions, h5py.Group):
        raise TypeError(f"Unsupported RoboFactory actions object: {type(actions)}")
    return tuple(sorted((str(name) for name in actions), key=agent_sort_key))


def action_dataset(trajectory: h5py.Group, agent_name: str) -> h5py.Dataset:
    actions = trajectory["actions"]
    if isinstance(actions, h5py.Dataset):
        if str(agent_name) != SINGLE_AGENT_NAME:
            raise KeyError(
                f"Flat RoboFactory actions expose only {SINGLE_AGENT_NAME!r}, "
                f"got {agent_name!r}"
            )
        return actions
    value = actions[str(agent_name)]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"actions/{agent_name} must be an HDF5 dataset")
    return value


def state_dataset(
    trajectory: h5py.Group,
    agent_name: str,
    field: str,
) -> h5py.Dataset:
    if field not in {"qpos", "qvel"}:
        raise ValueError(f"Unsupported RoboFactory state field: {field!r}")
    agent_root = trajectory["obs/agent"]
    direct_path = str(field)
    if direct_path in agent_root:
        if str(agent_name) != SINGLE_AGENT_NAME:
            raise KeyError(
                f"Flat RoboFactory state exposes only {SINGLE_AGENT_NAME!r}, "
                f"got {agent_name!r}"
            )
        value = agent_root[direct_path]
    else:
        value = agent_root[f"{agent_name}/{field}"]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"RoboFactory state {agent_name}/{field} must be a dataset")
    return value


def _camera_name(agent_name: str) -> str:
    _, separator, suffix = str(agent_name).rpartition("-")
    if not separator or not suffix.isdigit():
        raise ValueError(
            "Expected RoboFactory agent name ending in an integer, "
            f"got {agent_name!r}"
        )
    return f"head_camera_agent{int(suffix)}"


def camera_pair_paths(
    trajectory: h5py.Group,
    names: Sequence[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return one ``(reference, agent)`` RGB pair per logical agent.

    Multi-robot trajectories retain the original global/agent pairing.  A
    single-robot flat trajectory has only ``head_camera`` and therefore uses a
    deterministic self-pair.  Missing or ambiguous cameras fail closed.
    """

    resolved_names = tuple(agent_names(trajectory) if names is None else names)
    if not resolved_names:
        return ()
    global_path = "obs/sensor_data/head_camera_global/rgb"
    if global_path in trajectory:
        pairs = tuple(
            (
                global_path,
                f"obs/sensor_data/{_camera_name(agent_name)}/rgb",
            )
            for agent_name in resolved_names
        )
    else:
        single_path = "obs/sensor_data/head_camera/rgb"
        if resolved_names != (SINGLE_AGENT_NAME,) or single_path not in trajectory:
            raise KeyError(
                "RoboFactory trajectory must contain either head_camera_global plus "
                "per-agent cameras, or a single-agent head_camera"
            )
        pairs = ((single_path, single_path),)
    missing = sorted(
        {
            path
            for pair in pairs
            for path in pair
            if path not in trajectory
        }
    )
    if missing:
        raise KeyError(f"Missing RoboFactory RGB cameras: {missing}")
    return pairs


def policy_video_path(trajectory: h5py.Group) -> str:
    """Return the policy observation camera without requiring teacher-only views."""

    if not agent_names(trajectory):
        raise KeyError("RoboFactory trajectory has no action agents or policy camera")
    global_path = "obs/sensor_data/head_camera_global/rgb"
    if global_path in trajectory:
        return global_path
    single_path = "obs/sensor_data/head_camera/rgb"
    if agent_names(trajectory) == (SINGLE_AGENT_NAME,) and single_path in trajectory:
        return single_path
    raise KeyError(
        "RoboFactory trajectory must contain head_camera_global or a single-agent "
        "head_camera policy observation"
    )
