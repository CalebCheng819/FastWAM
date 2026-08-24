#!/usr/bin/env python3
"""Run auditable held-out RoboFactory rollouts with FastWAM or expert actions."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch

try:
    from .fastwam_multi_robot_policy import (
        FastWAMMultiRobotPolicy,
        NOPOSPLAT_CHECKPOINT_SHA256,
        NORMALIZATION_STATS_PROVENANCE_MODES,
        POLICY_LIGHTNING_COMMIT,
        TRAINING_STATS_SHA256,
        camera_rgb_uint8,
        require_file_sha256,
        require_regular_file_metadata,
        sha256_file,
    )
except ImportError:
    from fastwam_multi_robot_policy import (  # type: ignore[no-redef]
        FastWAMMultiRobotPolicy,
        NOPOSPLAT_CHECKPOINT_SHA256,
        NORMALIZATION_STATS_PROVENANCE_MODES,
        POLICY_LIGHTNING_COMMIT,
        TRAINING_STATS_SHA256,
        camera_rgb_uint8,
        require_file_sha256,
        require_regular_file_metadata,
        sha256_file,
    )


ROBOFACTORY_COMMIT = "2d34fb38c80cb06550a5dbf99abac2c89f4336ed"
ROBOFACTORY_TREE = "3c59aeed0db5b473c9ec882130210305bc899175"
STEP5000_CHECKPOINT_SHA256 = (
    "ff47c06c3f1761a086084f45c20bbcae17862fcebeced5833433d1eaf6555231"
)
TASK_CONFIGS = {
    "PlaceFood-rf": "configs/table/place_food.yaml",
    "PlaceCubeInCup-rf": "configs/table/place_cube_in_cup.yaml",
    "StrikeCubeHard-rf": "configs/table/strike_cube_hard.yaml",
    "ThreeRobotsPlaceShoes-rf": "configs/table/three_robots_place_shoes.yaml",
    "ThreeRobotsStackCube-rf": "configs/table/three_robots_stack_cube.yaml",
    "FourRobotsStackCube-rf": "configs/table/four_robots_stack_cube.yaml",
}
SCHEMA_VERSION = "fastwam-robofactory-eval-diagnostic-v3"
ACTION_DIM = 8
ARM_DIMS = tuple(range(7))
GRIPPER_DIM = 7
ORACLE_INTERVENTIONS = (
    "none",
    "robot0_pose",
    "robot0_gripper",
    "robot1_action",
)


# SAPIEN, Gymnasium, and OpenCV load overlapping native graphics/runtime
# libraries.  On the DSW evaluation image, letting Gymnasium or OpenCV load
# first makes SAPIEN's first Device/RenderSystem construction segfault inside
# the native extension.  Keep the bootstrap resources alive for the lifetime
# of the evaluator process so that SAPIEN owns native initialization order.
_SAPIEN_NATIVE_BOOTSTRAP_RESOURCES: tuple[Any, Any, Any] | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if torch.is_tensor(value):
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
        json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False)
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
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError(f"Short append while writing {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_robofactory_checkout(
    root: Path, *, integrity_mode: str = "sha256"
) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    actual_commit = _git(root, "rev-parse", "HEAD")
    dirty = _git(root, "status", "--porcelain=v1", "-uall")
    if actual_commit != ROBOFACTORY_COMMIT:
        raise ValueError(
            "RoboFactory source identity mismatch: "
            f"expected_commit={ROBOFACTORY_COMMIT} actual_commit={actual_commit}"
        )
    if dirty:
        raise ValueError(f"RoboFactory checkout is dirty: {root}")
    if integrity_mode not in {"sha256", "metadata_no_hash"}:
        raise ValueError(f"Unsupported integrity mode: {integrity_mode!r}")
    actual_tree = None
    if integrity_mode == "sha256":
        actual_tree = _git(root, "rev-parse", "HEAD^{tree}")
        if actual_tree != ROBOFACTORY_TREE:
            raise ValueError(
                "RoboFactory source tree mismatch: "
                f"expected={ROBOFACTORY_TREE} actual={actual_tree}"
            )
    return {
        "path": str(root),
        "commit": actual_commit,
        "tree": actual_tree,
        "clean": True,
        "integrity_mode": integrity_mode,
        "remote": _git(root, "remote", "get-url", "origin"),
    }


def _load_panel(
    path: Path,
    expected_sha256: str | None,
    *,
    expected_size_bytes: int | None = None,
    integrity_mode: str = "sha256",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if integrity_mode == "sha256":
        if not expected_sha256:
            raise ValueError("panel_sha256 is required in sha256 mode")
        resolved, actual_sha256 = require_file_sha256(
            path,
            expected_sha256,
            label="held-out evaluation panel",
        )
        identity = {
            "path": str(resolved),
            "size_bytes": resolved.stat().st_size,
            "sha256": actual_sha256,
            "integrity_mode": integrity_mode,
        }
    elif integrity_mode == "metadata_no_hash":
        if expected_size_bytes is None:
            raise ValueError("panel_size_bytes is required in metadata_no_hash mode")
        resolved, actual_size_bytes = require_regular_file_metadata(
            path,
            expected_size_bytes=expected_size_bytes,
            label="held-out evaluation panel",
        )
        identity = {
            "path": str(resolved),
            "size_bytes": actual_size_bytes,
            "sha256": None,
            "integrity_mode": integrity_mode,
        }
    else:
        raise ValueError(f"Unsupported integrity mode: {integrity_mode!r}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "fastwam-robofactory-heldout-panel-v1":
        raise ValueError(
            f"Unexpected evaluation panel schema: {payload.get('schema_version')!r}"
        )
    if not isinstance(payload.get("episodes"), list):
        raise TypeError("Evaluation panel must contain an episodes list")
    return payload, identity


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
    requested = dataset_root / relative
    before = requested.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(
            f"Source H5 must be a direct regular non-symlink file: {requested}"
        )
    if before.st_nlink != 1:
        raise ValueError(f"Source H5 must have exactly one hard link: {requested}")
    source = requested.resolve(strict=True)
    try:
        source.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"Source path escapes dataset root: {relative!r}") from error
    after = source.stat()
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise ValueError(f"Source H5 identity changed during resolution: {requested}")
    return source


def _as_bool(value: Any, *, label: str) -> bool:
    tensor = torch.as_tensor(value).detach().cpu().reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(
            f"{label} must contain one value, got shape {tuple(tensor.shape)}"
        )
    return bool(tensor.item())


def _restore_initial_state(env: Any, source: Path, trajectory: str) -> None:
    from mani_skill.trajectory import utils as trajectory_utils

    with h5py.File(source, "r") as handle:
        states = trajectory_utils.dict_to_list_of_dicts(
            handle[trajectory]["env_states"]
        )
    if not states:
        raise ValueError(f"No env states in {source}:{trajectory}")
    env.unwrapped.set_state_dict(states[0])


def _load_expert_actions(
    source: Path,
    trajectory: str,
    agent_names: Sequence[str],
) -> dict[str, np.ndarray]:
    with h5py.File(source, "r") as handle:
        group = handle[trajectory]["actions"]
        actions = {
            name: np.asarray(group[name][:], dtype=np.float32) for name in agent_names
        }
    lengths = {value.shape[0] for value in actions.values()}
    shapes = {value.shape[1:] for value in actions.values()}
    if len(lengths) != 1 or shapes != {(8,)}:
        raise ValueError(f"Invalid expert action contract in {source}:{trajectory}")
    return actions


def _flat_action_to_dict(
    action: np.ndarray,
    agent_names: Sequence[str],
) -> dict[str, np.ndarray]:
    flat = np.asarray(action, dtype=np.float32).reshape(-1)
    expected = len(agent_names) * 8
    if flat.shape != (expected,):
        raise ValueError(f"Flat action must have shape ({expected},), got {flat.shape}")
    if not np.isfinite(flat).all():
        raise FloatingPointError("Policy action contains non-finite values")
    return {
        name: np.ascontiguousarray(flat[index * 8 : (index + 1) * 8])
        for index, name in enumerate(agent_names)
    }


def _dict_action_to_flat(
    action: Mapping[str, np.ndarray], agent_names: Sequence[str]
) -> np.ndarray:
    missing = [name for name in agent_names if name not in action]
    if missing:
        raise KeyError(f"Action is missing agents: {missing}")
    blocks = [np.asarray(action[name], dtype=np.float32).reshape(-1) for name in agent_names]
    if any(block.shape != (ACTION_DIM,) for block in blocks):
        raise ValueError(f"Every agent action must have shape ({ACTION_DIM},)")
    flat = np.concatenate(blocks)
    if not np.isfinite(flat).all():
        raise FloatingPointError("Action contains non-finite values")
    return np.ascontiguousarray(flat)


def apply_oracle_intervention(
    policy_action: Mapping[str, np.ndarray],
    expert_action: Mapping[str, np.ndarray] | None,
    agent_names: Sequence[str],
    intervention: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if intervention not in ORACLE_INTERVENTIONS:
        raise ValueError(f"Unsupported oracle intervention: {intervention!r}")
    executed = {
        name: np.ascontiguousarray(
            np.asarray(policy_action[name], dtype=np.float32)
        ).copy()
        for name in agent_names
    }
    if intervention == "none":
        return executed, {"mode": intervention, "applied": False, "reason": "disabled"}
    if expert_action is None:
        return executed, {
            "mode": intervention,
            "applied": False,
            "reason": "expert_trace_exhausted",
        }
    if len(agent_names) < 2:
        raise ValueError("Oracle interventions require at least two robots")
    if intervention == "robot0_pose":
        executed[agent_names[0]][list(ARM_DIMS)] = np.asarray(
            expert_action[agent_names[0]], dtype=np.float32
        )[list(ARM_DIMS)]
        dimensions = [f"{agent_names[0]}[{index}]" for index in ARM_DIMS]
    elif intervention == "robot0_gripper":
        executed[agent_names[0]][GRIPPER_DIM] = np.asarray(
            expert_action[agent_names[0]], dtype=np.float32
        )[GRIPPER_DIM]
        dimensions = [f"{agent_names[0]}[{GRIPPER_DIM}]"]
    else:
        executed[agent_names[1]][:] = np.asarray(
            expert_action[agent_names[1]], dtype=np.float32
        )
        dimensions = [f"{agent_names[1]}[{index}]" for index in range(ACTION_DIM)]
    return executed, {
        "mode": intervention,
        "applied": True,
        "reason": "expert_action_available",
        "dimensions": dimensions,
    }


def action_bound_records(
    action: Mapping[str, np.ndarray], action_space: Any, *, step: int, source: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for agent_name, value in action.items():
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        space = action_space[agent_name]
        low = np.asarray(space.low, dtype=np.float64).reshape(-1)
        high = np.asarray(space.high, dtype=np.float64).reshape(-1)
        if array.shape != (ACTION_DIM,) or low.shape != array.shape or high.shape != array.shape:
            raise ValueError(
                f"Action-space shape mismatch for {agent_name}: "
                f"action={array.shape} low={low.shape} high={high.shape}"
            )
        for dimension, (actual, minimum, maximum) in enumerate(zip(array, low, high)):
            if not math.isfinite(float(actual)):
                raise FloatingPointError(
                    f"Non-finite action at {agent_name}[{dimension}]: {actual}"
                )
            if actual < minimum or actual > maximum:
                records.append(
                    {
                        "step": int(step),
                        "source": source,
                        "agent": agent_name,
                        "dimension": dimension,
                        "value": float(actual),
                        "low": float(minimum),
                        "high": float(maximum),
                        "underflow": float(max(0.0, minimum - actual)),
                        "overflow": float(max(0.0, actual - maximum)),
                    }
                )
    return records


def summarize_bound_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    maximum_excess: dict[str, float] = {}
    steps: set[int] = set()
    for record in records:
        key = f"{record['agent']}[{int(record['dimension'])}]"
        counts[key] += 1
        steps.add(int(record["step"]))
        excess = max(float(record["underflow"]), float(record["overflow"]))
        maximum_excess[key] = max(maximum_excess.get(key, 0.0), excess)
    return {
        "scalar_violations": len(records),
        "steps_with_violation": len(steps),
        "per_dimension_counts": dict(sorted(counts.items())),
        "per_dimension_max_excess": dict(sorted(maximum_excess.items())),
    }


def _vector(value: Any, size: int, *, label: str) -> list[float]:
    array = torch.as_tensor(value).detach().cpu().numpy().reshape(-1)
    if array.size < size:
        raise ValueError(f"{label} must contain at least {size} values")
    selected = np.asarray(array[:size], dtype=np.float64)
    if not np.isfinite(selected).all():
        raise FloatingPointError(f"{label} contains non-finite values")
    return [float(item) for item in selected]


def physical_snapshot(env: Any, *, step: int, initial_meat_z: float | None) -> dict[str, Any]:
    unwrapped = env.unwrapped
    robot0 = unwrapped.agent.agents[0]
    meat_position = _vector(unwrapped.meat.pose.p, 3, label="meat.pose.p")
    pot_position = _vector(unwrapped.pot.pose.p, 3, label="pot.pose.p")
    qpos = _vector(robot0.robot.get_qpos(), 9, label="robot0.qpos")
    grasping = _as_bool(
        robot0.is_grasping(unwrapped.meat), label="robot0.is_grasping(meat)"
    )
    baseline_z = meat_position[2] if initial_meat_z is None else float(initial_meat_z)
    snapshot = {
        "step": int(step),
        "meat_position": meat_position,
        "pot_position": pot_position,
        "meat_lift_m": max(0.0, meat_position[2] - baseline_z),
        "robot0_grasping_meat": grasping,
        "robot0_qpos": qpos,
        "robot0_gripper_qpos": qpos[-2:],
    }
    tcp = getattr(robot0, "tcp", None)
    if tcp is not None and getattr(tcp, "pose", None) is not None:
        snapshot["robot0_tcp_position"] = _vector(
            tcp.pose.p, 3, label="robot0.tcp.pose.p"
        )
    return snapshot


def summarize_physical_snapshots(
    snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grasp_steps = [
        int(snapshot["step"])
        for snapshot in snapshots
        if bool(snapshot["robot0_grasping_meat"])
    ]
    lifts = [float(snapshot["meat_lift_m"]) for snapshot in snapshots]
    return {
        "grasped_ever": bool(grasp_steps),
        "first_grasp_step": min(grasp_steps) if grasp_steps else None,
        "grasp_steps": len(grasp_steps),
        "max_meat_lift_m": max(lifts) if lifts else 0.0,
        "final_meat_lift_m": lifts[-1] if lifts else 0.0,
    }


def aggregate_bound_summaries(
    results: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    maximum_excess: dict[str, float] = {}
    scalar_violations = 0
    steps_with_violation = 0
    episodes_with_violation = 0
    for result in results:
        summary = result[key]
        episode_violations = int(summary["scalar_violations"])
        scalar_violations += episode_violations
        steps_with_violation += int(summary["steps_with_violation"])
        episodes_with_violation += int(episode_violations > 0)
        counts.update(
            {
                str(name): int(count)
                for name, count in summary["per_dimension_counts"].items()
            }
        )
        for name, excess in summary["per_dimension_max_excess"].items():
            maximum_excess[str(name)] = max(
                maximum_excess.get(str(name), 0.0), float(excess)
            )
    return {
        "episodes": len(results),
        "episodes_with_violation": episodes_with_violation,
        "scalar_violations": scalar_violations,
        "steps_with_violation": steps_with_violation,
        "per_dimension_counts": dict(sorted(counts.items())),
        "per_dimension_max_excess": dict(sorted(maximum_excess.items())),
    }


def aggregate_physical_summaries(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    physical = [result["physical"] for result in results]
    max_lifts = [float(item["max_meat_lift_m"]) for item in physical]
    final_lifts = [float(item["final_meat_lift_m"]) for item in physical]
    grasped = sum(bool(item["grasped_ever"]) for item in physical)
    return {
        "episodes": len(physical),
        "episodes_grasped": grasped,
        "grasp_rate": grasped / len(physical) if physical else None,
        "maximum_meat_lift_m": max(max_lifts) if max_lifts else None,
        "mean_episode_max_meat_lift_m": (
            sum(max_lifts) / len(max_lifts) if max_lifts else None
        ),
        "mean_final_meat_lift_m": (
            sum(final_lifts) / len(final_lifts) if final_lifts else None
        ),
        "per_episode_max_meat_lift_m": max_lifts,
        "per_episode_final_meat_lift_m": final_lifts,
    }


def _video_frame_from_obs(observation: Mapping[str, Any]) -> np.ndarray:
    image = camera_rgb_uint8(observation, "head_camera_global")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Video frame must be CHW RGB, got {tuple(image.shape)}")
    return np.ascontiguousarray(image.permute(1, 2, 0).cpu().numpy())


def _validate_video(path: Path, *, expected_frames: int) -> dict[str, Any]:
    import imageio.v2 as imageio

    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Video is missing or invalid: {path}")
    reader = imageio.get_reader(path)
    try:
        frames = sum(1 for _ in reader)
    finally:
        reader.close()
    if frames != expected_frames:
        raise ValueError(
            f"Video frame count mismatch: expected={expected_frames} actual={frames}"
        )
    return {"path": path.name, "frames": frames, "size_bytes": path.stat().st_size}


def _action_space_contract(action_space: Any, agent_names: Sequence[str]) -> dict[str, Any]:
    agents: dict[str, Any] = {}
    for name in agent_names:
        space = action_space[name]
        agents[name] = {
            "shape": list(space.shape),
            "dtype": str(space.dtype),
            "low": np.asarray(space.low).reshape(-1).tolist(),
            "high": np.asarray(space.high).reshape(-1).tolist(),
        }
    return {
        "environment_action_space": agents,
        "evaluator_clipping": False,
        "policy_denormalization": "normalized_action * training_action_std + training_action_mean",
    }


def _regular_file_inventory(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"Episode artifact contains a symlink: {relative}")
        if path.is_file():
            files.append(relative)
        elif not path.is_dir():
            raise ValueError(f"Episode artifact contains a special file: {relative}")
    return files


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _publish_directory(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Episode artifact already exists: {destination}")
    inventory = _regular_file_inventory(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=False)
    published = _regular_file_inventory(destination)
    if published != inventory:
        raise RuntimeError("Published episode inventory differs from local staging")
    for relative in inventory:
        if not _files_equal(source / relative, destination / relative):
            raise RuntimeError(f"Published episode file differs: {relative}")
    return {
        "path": str(destination),
        "files": len(inventory),
        "bytes": sum((destination / relative).stat().st_size for relative in inventory),
    }


def _module_file(module: Any, label: str) -> Path:
    value = getattr(module, "__file__", None)
    if not value:
        raise RuntimeError(f"{label} has no ordinary module file")
    return Path(value).resolve(strict=True)


def _anchor_robofactory_imports(robofactory_root: Path) -> Path:
    root = robofactory_root.expanduser().resolve(strict=True)
    expected_packages = {
        "utils": (root / "utils").resolve(strict=True),
        "tasks": (root / "tasks").resolve(strict=True),
    }
    root_text = str(root)
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    os.chdir(root)
    importlib.invalidate_caches()
    for name, expected in expected_packages.items():
        loaded = sys.modules.get(name)
        if loaded is None:
            continue
        package_path = getattr(loaded, "__path__", None)
        if package_path is None:
            raise RuntimeError(
                f"loaded {name!r} is not the expected RoboFactory namespace package"
            )
        actual = {Path(entry).resolve(strict=True) for entry in package_path}
        if expected not in actual:
            raise RuntimeError(
                f"loaded {name!r} does not include RoboFactory path: "
                f"expected={expected} actual={sorted(map(str, actual))}"
            )
    return root


def _bootstrap_sapien_native_runtime():
    global _SAPIEN_NATIVE_BOOTSTRAP_RESOURCES

    import sapien

    if _SAPIEN_NATIVE_BOOTSTRAP_RESOURCES is None:
        cpu_device = sapien.Device("cpu")
        render_device = sapien.Device("cuda")
        render_system = sapien.render.RenderSystem(render_device)
        _SAPIEN_NATIVE_BOOTSTRAP_RESOURCES = (
            cpu_device,
            render_device,
            render_system,
        )
    return sapien


def _preflight_environment_imports(robofactory_root: Path) -> dict[str, str]:
    root = _anchor_robofactory_imports(robofactory_root)

    sapien = _bootstrap_sapien_native_runtime()
    import cv2
    import mani_skill
    import tasks.place_food as place_food
    import utils.scenes as scenes
    from OpenGL import EGL

    if not callable(getattr(EGL, "eglQueryString", None)):
        raise RuntimeError("PyOpenGL EGL binding lacks eglQueryString")
    expected = {
        "place_food": (root / "tasks" / "place_food.py").resolve(strict=True),
        "scenes": (root / "utils" / "scenes" / "__init__.py").resolve(strict=True),
    }
    actual = {
        "place_food": _module_file(place_food, "tasks.place_food"),
        "scenes": _module_file(scenes, "utils.scenes"),
    }
    if actual != expected:
        raise RuntimeError(
            f"RoboFactory module provenance mismatch: {actual} != {expected}"
        )
    return {
        **{name: str(path) for name, path in actual.items()},
        "cv2": str(_module_file(cv2, "cv2")),
        "mani_skill": str(_module_file(mani_skill, "mani_skill")),
        "sapien": str(_module_file(sapien, "sapien")),
    }


def _build_environment(robofactory_root: Path, task_name: str):
    robofactory_root = _anchor_robofactory_imports(robofactory_root)
    _preflight_environment_imports(robofactory_root)
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


def _gpu_memory() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _run_episode(
    *,
    args: argparse.Namespace,
    episode: Mapping[str, Any],
    dataset_root: Path,
    robofactory_root: Path,
    policy: FastWAMMultiRobotPolicy | None,
) -> dict[str, Any]:
    started_at = _utc_now()
    start_time = time.monotonic()
    source = _source_path(dataset_root, str(episode["source_path"]))
    source_size_bytes = source.stat().st_size
    if source_size_bytes != int(episode["source_h5_bytes"]):
        raise ValueError(f"Source H5 byte-size mismatch: {source}")
    if args.integrity_mode == "sha256" and args.verify_source_h5:
        actual_source_sha256 = sha256_file(source)
        if actual_source_sha256 != episode["source_h5_sha256"]:
            raise ValueError(
                f"Source H5 SHA-256 mismatch: expected={episode['source_h5_sha256']} "
                f"actual={actual_source_sha256} path={source}"
            )
    else:
        actual_source_sha256 = None

    np.random.seed(args.policy_seed + int(episode["task_index"]))
    torch.manual_seed(args.policy_seed + int(episode["task_index"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.policy_seed + int(episode["task_index"]))
        torch.cuda.reset_peak_memory_stats()

    episode_label = (
        f"panel-{int(episode['panel_index']):04d}-"
        f"task-{int(episode['task_index']):04d}"
    )
    episode_destination = args.output_dir.expanduser().resolve() / "episodes" / episode_label
    if episode_destination.exists() or episode_destination.is_symlink():
        raise FileExistsError(f"Episode output must not exist: {episode_destination}")

    env = None
    video_writer = None
    steps = 0
    policy_queries = 0
    success = False
    termination_reason = "max_steps"
    policy_bound_records: list[dict[str, Any]] = []
    executed_bound_records: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    oracle_attempts = 0
    oracle_applied = 0
    oracle_fallbacks = 0
    initial_meat_z: float | None = None
    action_space_contract: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix=f"fastwam-eval-{episode_label}-") as temporary:
        staging = Path(temporary) / episode_label
        staging.mkdir()
        policy_bounds_path = staging / "policy_action_bound_violations.jsonl"
        executed_bounds_path = staging / "executed_action_bound_violations.jsonl"
        executed_actions_path = staging / "executed_actions.jsonl"
        policy_queries_path = staging / "policy_queries.jsonl"
        physical_path = staging / "physical_snapshots.jsonl"
        for path in (
            policy_bounds_path,
            executed_bounds_path,
            executed_actions_path,
            policy_queries_path,
            physical_path,
        ):
            path.touch(mode=0o640)
        video_path = staging / "rollout.mp4"

        try:
            env = _build_environment(robofactory_root, args.task)
            _reset_environment(env, episode)
            _restore_initial_state(env, source, str(episode["trajectory"]))
            agent_names = tuple(episode["agent_names"])
            actual_agents = tuple(
                f"panda-{index}" for index in range(len(env.unwrapped.agent.agents))
            )
            if actual_agents != agent_names:
                raise ValueError(
                    f"Environment agents {actual_agents} do not match panel {agent_names}"
                )
            action_space_contract = _action_space_contract(
                env.action_space, agent_names
            )
            max_steps = (
                int(episode["max_episode_steps"])
                if args.max_steps is None
                else min(int(args.max_steps), int(episode["max_episode_steps"]))
            )
            expert = None
            if args.mode == "expert-replay" or args.oracle_intervention != "none":
                expert = _load_expert_actions(
                    source, str(episode["trajectory"]), agent_names
                )

            raw_obs = env.unwrapped.get_obs()
            first_snapshot = physical_snapshot(
                env, step=0, initial_meat_z=initial_meat_z
            )
            initial_meat_z = float(first_snapshot["meat_position"][2])
            first_snapshot["meat_lift_m"] = 0.0
            snapshots.append(first_snapshot)
            _append_jsonl(physical_path, first_snapshot)

            if args.record_video:
                import imageio.v2 as imageio

                video_writer = imageio.get_writer(
                    str(video_path),
                    fps=int(args.video_fps),
                    codec="libx264",
                    quality=8,
                    macro_block_size=None,
                )
                video_writer.append_data(_video_frame_from_obs(raw_obs))

            def execute_action(
                policy_action: Mapping[str, np.ndarray],
                *,
                expert_action: Mapping[str, np.ndarray] | None,
                query_index: int | None,
                chunk_index: int,
            ) -> tuple[bool, bool, bool]:
                nonlocal steps, success, oracle_attempts, oracle_applied, oracle_fallbacks
                if args.mode == "expert-replay":
                    executed = {
                        name: np.ascontiguousarray(
                            np.asarray(policy_action[name], dtype=np.float32)
                        )
                        for name in agent_names
                    }
                    oracle = {
                        "mode": "none",
                        "applied": False,
                        "reason": "expert_replay",
                    }
                else:
                    executed, oracle = apply_oracle_intervention(
                        policy_action,
                        expert_action,
                        agent_names,
                        args.oracle_intervention,
                    )
                    if args.oracle_intervention != "none":
                        oracle_attempts += 1
                        if oracle["applied"]:
                            oracle_applied += 1
                        else:
                            oracle_fallbacks += 1
                    if policy is None:
                        raise RuntimeError("FastWAM policy disappeared during rollout")
                    policy.record_action(_dict_action_to_flat(executed, agent_names))

                current_policy_bounds = action_bound_records(
                    policy_action,
                    env.action_space,
                    step=steps,
                    source="policy" if args.mode == "fastwam" else "expert",
                )
                current_executed_bounds = action_bound_records(
                    executed,
                    env.action_space,
                    step=steps,
                    source="executed",
                )
                policy_bound_records.extend(current_policy_bounds)
                executed_bound_records.extend(current_executed_bounds)
                for record in current_policy_bounds:
                    _append_jsonl(policy_bounds_path, record)
                for record in current_executed_bounds:
                    _append_jsonl(executed_bounds_path, record)
                _append_jsonl(
                    executed_actions_path,
                    {
                        "step": steps,
                        "query_index": query_index,
                        "action_chunk_index": chunk_index,
                        "policy_action": policy_action,
                        "expert_action": expert_action,
                        "executed_action": executed,
                        "oracle": oracle,
                    },
                )
                _, _, terminated, truncated, info = env.step(executed)
                steps += 1
                success = _as_bool(info["success"], label="info.success")
                observation = env.unwrapped.get_obs()
                snapshot = physical_snapshot(
                    env, step=steps, initial_meat_z=initial_meat_z
                )
                snapshots.append(snapshot)
                _append_jsonl(physical_path, snapshot)
                if video_writer is not None:
                    video_writer.append_data(_video_frame_from_obs(observation))
                if args.mode == "fastwam":
                    if policy is None:
                        raise RuntimeError("FastWAM policy disappeared during rollout")
                    policy.update_obs(observation, env.unwrapped.get_state_dict())
                return (
                    success,
                    _as_bool(terminated, label="terminated"),
                    _as_bool(truncated, label="truncated"),
                )

            if args.mode == "expert-replay":
                if expert is None:
                    raise RuntimeError("Expert replay is missing its action trace")
                horizon = min(len(next(iter(expert.values()))), max_steps)
                for index in range(horizon):
                    expert_action = {name: expert[name][index] for name in agent_names}
                    succeeded, terminated, truncated = execute_action(
                        expert_action,
                        expert_action=expert_action,
                        query_index=None,
                        chunk_index=0,
                    )
                    if succeeded:
                        termination_reason = "success"
                        break
                    if terminated:
                        termination_reason = "terminated"
                        break
                    if truncated:
                        termination_reason = "truncated"
                        break
                else:
                    if horizon < max_steps:
                        termination_reason = "expert_trace_exhausted"
            else:
                if policy is None:
                    raise RuntimeError("FastWAM mode requires an initialized policy")
                episode_policy_seed = args.policy_seed + int(episode["task_index"])
                policy.start_episode(episode_policy_seed)
                policy.update_obs(raw_obs, env.unwrapped.get_state_dict())
                while steps < max_steps:
                    trace = policy.get_action_trace()
                    policy_queries += 1
                    _append_jsonl(policy_queries_path, trace)
                    action_chunk = np.asarray(trace["flat_action"], dtype=np.float32)
                    execute = min(
                        int(args.exec_horizon), len(action_chunk), max_steps - steps
                    )
                    if execute < 1:
                        raise ValueError("Policy returned an empty executable action chunk")
                    stop_chunk = False
                    for index in range(execute):
                        policy_action = _flat_action_to_dict(
                            action_chunk[index], agent_names
                        )
                        expert_action = None
                        if expert is not None and steps < len(next(iter(expert.values()))):
                            expert_action = {
                                name: expert[name][steps] for name in agent_names
                            }
                        succeeded, terminated, truncated = execute_action(
                            policy_action,
                            expert_action=expert_action,
                            query_index=int(trace["query_index"]),
                            chunk_index=index,
                        )
                        if succeeded:
                            termination_reason = "success"
                            stop_chunk = True
                            break
                        if terminated:
                            termination_reason = "terminated"
                            stop_chunk = True
                            break
                        if truncated:
                            termination_reason = "truncated"
                            stop_chunk = True
                            break
                    if stop_chunk:
                        break
        finally:
            if video_writer is not None:
                video_writer.close()
                video_writer = None
            if env is not None:
                env.close()
                env = None

        video = (
            _validate_video(video_path, expected_frames=steps + 1)
            if args.record_video
            else None
        )
        physical = summarize_physical_snapshots(snapshots)
        if action_space_contract is None:
            raise RuntimeError("Action-space contract was not captured")
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "condition_name": args.condition_name,
            "mode": args.mode,
            "oracle_intervention": args.oracle_intervention,
            "oracle_action_semantics": {
                "robot0_pose": "expert robot0 pd_joint_pos dimensions 0..6",
                "robot0_gripper": "expert robot0 pd_joint_pos dimension 7",
                "robot1_action": "expert robot1 pd_joint_pos dimensions 0..7",
                "time_alignment": "expert action at the same simulator step",
            },
            "oracle_attempts": oracle_attempts,
            "oracle_applied": oracle_applied,
            "oracle_fallbacks": oracle_fallbacks,
            "task_name": args.task,
            "task_index": int(episode["task_index"]),
            "panel_index": int(episode["panel_index"]),
            "source_path": str(episode["source_path"]),
            "source_h5_size_bytes": source_size_bytes,
            "source_h5_sha256": (
                episode["source_h5_sha256"]
                if args.integrity_mode == "sha256"
                else None
            ),
            "source_h5_sha256_verified": actual_source_sha256,
            "trajectory": episode["trajectory"],
            "episode_id": int(episode["episode_id"]),
            "environment_seed": int(episode["episode_seed"]),
            "policy_seed": args.policy_seed + int(episode["task_index"]),
            "success": success,
            "steps": steps,
            "policy_queries": policy_queries,
            "exec_horizon": int(args.exec_horizon),
            "action_space_contract": action_space_contract,
            "policy_action_bounds": summarize_bound_records(policy_bound_records),
            "executed_action_bounds": summarize_bound_records(executed_bound_records),
            "physical": physical,
            "video": video,
            "termination_reason": termination_reason,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_seconds": time.monotonic() - start_time,
            "gpu_memory": _gpu_memory(),
            "artifact_path": str(episode_destination),
        }
        _atomic_json(staging / "episode_result.json", result)
        result["artifact_publication"] = _publish_directory(
            staging, episode_destination
        )
        return result


def _required_fastwam_arguments(args: argparse.Namespace) -> None:
    required = [
        "checkpoint",
        "stats",
        "context_cache_dir",
        "model_cache_root",
    ]
    if args.gaussian_conditioning:
        required.extend(("policy_lightning_repo", "noposplat_checkpoint"))
        if getattr(args, "integrity_mode", "sha256") == "metadata_no_hash":
            required.append("noposplat_checkpoint_size_bytes")
    missing = [
        name
        for name in required
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(f"FastWAM mode is missing required arguments: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("expert-replay", "fastwam"), required=True)
    parser.add_argument("--task", choices=tuple(TASK_CONFIGS), required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument(
        "--integrity-mode",
        choices=("sha256", "metadata_no_hash"),
        default="sha256",
    )
    parser.add_argument("--panel-sha256")
    parser.add_argument("--panel-size-bytes", type=int)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--robofactory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-name", required=True)
    parser.add_argument(
        "--oracle-intervention",
        choices=ORACLE_INTERVENTIONS,
        default="none",
    )
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--eval-code-commit", required=True)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--exec-horizon", type=int, default=5)
    parser.add_argument("--policy-seed", type=int, default=10000)
    parser.add_argument(
        "--verify-source-h5",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256", default=STEP5000_CHECKPOINT_SHA256)
    parser.add_argument("--checkpoint-size-bytes", type=int)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--stats-sha256", default=TRAINING_STATS_SHA256)
    parser.add_argument("--stats-size-bytes", type=int)
    parser.add_argument(
        "--stats-provenance-mode",
        choices=NORMALIZATION_STATS_PROVENANCE_MODES,
        default="train_split",
    )
    parser.add_argument("--context-cache-dir", type=Path)
    parser.add_argument("--context-size-bytes", type=int)
    parser.add_argument("--model-cache-root", type=Path)
    parser.add_argument("--policy-lightning-repo", type=Path)
    parser.add_argument("--policy-lightning-commit", default=POLICY_LIGHTNING_COMMIT)
    parser.add_argument("--noposplat-checkpoint", type=Path)
    parser.add_argument("--noposplat-checkpoint-size-bytes", type=int)
    parser.add_argument(
        "--noposplat-checkpoint-sha256",
        default=NOPOSPLAT_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--gaussian-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--training-source-commit")
    parser.add_argument("--training-job-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-device")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--sigma-shift", type=float)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.condition_name) is None:
        raise ValueError(
            "condition_name must contain only letters, digits, dot, underscore, "
            "or dash and must start with a letter or digit"
        )
    if args.video_fps < 1:
        raise ValueError("video_fps must be positive")
    if args.mode != "fastwam" and args.oracle_intervention != "none":
        raise ValueError("Oracle interventions are only valid in fastwam mode")

    robofactory_root = args.robofactory_root.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Evaluation output must not already exist: {output_dir}")
    output_dir = output_dir.resolve()
    source_identity = _verify_robofactory_checkout(
        robofactory_root, integrity_mode=args.integrity_mode
    )
    panel, panel_identity = _load_panel(
        args.panel,
        args.panel_sha256,
        expected_size_bytes=args.panel_size_bytes,
        integrity_mode=args.integrity_mode,
    )
    selected = _selected_episodes(
        panel,
        args.task,
        args.episode_start,
        args.num_episodes,
    )
    if args.exec_horizon < 1:
        raise ValueError("exec_horizon must be positive")
    for episode in selected:
        source = _source_path(dataset_root, str(episode["source_path"]))
        if source.stat().st_size != int(episode["source_h5_bytes"]):
            raise ValueError(f"Source H5 byte-size mismatch: {source}")

    policy = None
    policy_init_seconds = None
    if args.mode == "fastwam":
        _required_fastwam_arguments(args)
        init_started = time.monotonic()
        policy = FastWAMMultiRobotPolicy(
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=(
                args.checkpoint_sha256
                if args.integrity_mode == "sha256"
                else None
            ),
            checkpoint_size_bytes=args.checkpoint_size_bytes,
            stats_path=args.stats,
            expected_stats_sha256=(
                args.stats_sha256 if args.integrity_mode == "sha256" else None
            ),
            stats_size_bytes=args.stats_size_bytes,
            stats_provenance_mode=args.stats_provenance_mode,
            context_cache_dir=args.context_cache_dir,
            context_size_bytes=args.context_size_bytes,
            task_name=args.task,
            model_cache_root=args.model_cache_root,
            policy_lightning_repo=args.policy_lightning_repo,
            policy_lightning_commit=args.policy_lightning_commit,
            noposplat_checkpoint_path=args.noposplat_checkpoint,
            noposplat_checkpoint_sha256=(
                args.noposplat_checkpoint_sha256
                if args.integrity_mode == "sha256"
                else None
            ),
            noposplat_checkpoint_size_bytes=args.noposplat_checkpoint_size_bytes,
            integrity_mode=args.integrity_mode,
            gaussian_conditioning=args.gaussian_conditioning,
            training_source_commit=args.training_source_commit,
            training_job_id=args.training_job_id,
            device=args.device,
            teacher_device=args.teacher_device,
            action_horizon=args.action_horizon,
            num_inference_steps=args.num_inference_steps,
            sigma_shift=args.sigma_shift,
            seed=args.policy_seed,
        )
        policy_init_seconds = time.monotonic() - init_started

    output_dir.mkdir(parents=True, exist_ok=False)

    run_manifest = {
        "schema_version": "fastwam-robofactory-eval-run-v3",
        "status": "running",
        "started_at": _utc_now(),
        "mode": args.mode,
        "condition_name": args.condition_name,
        "oracle_intervention": args.oracle_intervention,
        "record_video": bool(args.record_video),
        "video_fps": int(args.video_fps),
        "task_name": args.task,
        "episode_start": args.episode_start,
        "num_episodes": args.num_episodes,
        "max_steps_override": args.max_steps,
        "exec_horizon": args.exec_horizon,
        "policy_seed_base": args.policy_seed,
        "policy_seed_schedule": "base_plus_task_index_then_plus_query_index_v1",
        "eval_code_commit": args.eval_code_commit,
        "integrity_mode": args.integrity_mode,
        "panel": panel_identity,
        "dataset_root": str(dataset_root),
        "verify_source_h5": args.verify_source_h5,
        "robofactory": source_identity,
        "policy_init_seconds": policy_init_seconds,
        "policy": None if policy is None else policy.provenance(),
        "argv": sys.argv,
    }
    _atomic_json(output_dir / "run_manifest.json", run_manifest)

    results: list[dict[str, Any]] = []
    infrastructure_errors = 0
    for episode in selected:
        try:
            result = _run_episode(
                args=args,
                episode=episode,
                dataset_root=dataset_root,
                robofactory_root=robofactory_root,
                policy=policy,
            )
        except Exception as error:  # noqa: BLE001 - persisted terminal infrastructure record
            infrastructure_errors += 1
            result = {
                "schema_version": SCHEMA_VERSION,
                "status": "infrastructure_error",
                "mode": args.mode,
                "condition_name": args.condition_name,
                "oracle_intervention": args.oracle_intervention,
                "exec_horizon": int(args.exec_horizon),
                "task_name": args.task,
                "task_index": int(episode["task_index"]),
                "panel_index": int(episode["panel_index"]),
                "source_path": episode["source_path"],
                "trajectory": episode["trajectory"],
                "episode_id": int(episode["episode_id"]),
                "environment_seed": int(episode["episode_seed"]),
                "policy_seed": args.policy_seed + int(episode["task_index"]),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "finished_at": _utc_now(),
            }
        results.append(result)
        _append_jsonl(output_dir / "episodes.jsonl", result)
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["status"] != "completed" and not args.continue_on_error:
            break

    completed = [result for result in results if result["status"] == "completed"]
    successes = sum(bool(result["success"]) for result in completed)
    episodes_path = output_dir / "episodes.jsonl"
    summary = {
        "schema_version": "fastwam-robofactory-eval-summary-v3",
        "status": (
            "PASS"
            if infrastructure_errors == 0 and len(results) == len(selected)
            else "INFRASTRUCTURE_ERROR"
        ),
        "mode": args.mode,
        "condition_name": args.condition_name,
        "exec_horizon": int(args.exec_horizon),
        "oracle_intervention": args.oracle_intervention,
        "record_video": bool(args.record_video),
        "task_name": args.task,
        "episodes_requested": len(selected),
        "episodes_recorded": len(results),
        "episodes_completed": len(completed),
        "infrastructure_errors": infrastructure_errors,
        "successes": successes,
        "strict_success_rate": (
            successes / len(selected)
            if infrastructure_errors == 0 and len(completed) == len(selected)
            else None
        ),
        "diagnostic_success_rate_completed": (
            successes / len(completed) if completed else None
        ),
        "physical": aggregate_physical_summaries(completed),
        "policy_action_bounds": aggregate_bound_summaries(
            completed, "policy_action_bounds"
        ),
        "executed_action_bounds": aggregate_bound_summaries(
            completed, "executed_action_bounds"
        ),
        "oracle": {
            "attempts": sum(int(result["oracle_attempts"]) for result in completed),
            "applied": sum(int(result["oracle_applied"]) for result in completed),
            "fallbacks": sum(int(result["oracle_fallbacks"]) for result in completed),
        },
        "videos_recorded": sum(result["video"] is not None for result in completed),
        "finished_at": _utc_now(),
        "episodes_jsonl_size_bytes": episodes_path.stat().st_size,
        "episodes_jsonl_sha256": (
            sha256_file(episodes_path) if args.integrity_mode == "sha256" else None
        ),
    }
    if (
        args.mode == "expert-replay"
        and summary["status"] == "PASS"
        and successes != len(selected)
    ):
        summary["status"] = "EXPERT_REPLAY_FAILURE"
    _atomic_json(output_dir / "summary.json", summary)
    run_manifest["status"] = "terminal"
    run_manifest["finished_at"] = _utc_now()
    summary_path = output_dir / "summary.json"
    run_manifest["summary_size_bytes"] = summary_path.stat().st_size
    run_manifest["summary_sha256"] = (
        sha256_file(summary_path) if args.integrity_mode == "sha256" else None
    )
    _atomic_json(output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if summary["status"] == "INFRASTRUCTURE_ERROR":
        raise SystemExit(2)
    if summary["status"] == "EXPERT_REPLAY_FAILURE":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
