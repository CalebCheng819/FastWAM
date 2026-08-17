#!/usr/bin/env python3
"""Run auditable held-out RoboFactory rollouts with FastWAM or expert actions."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import stat
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
    try:
        os.write(descriptor, encoded)
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


def _action_bound_violations(
    action: Mapping[str, np.ndarray], action_space: Any
) -> int:
    violations = 0
    for name, value in action.items():
        space = action_space[name]
        violations += int(np.count_nonzero((value < space.low) | (value > space.high)))
    return violations


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


def _preflight_environment_imports(robofactory_root: Path) -> dict[str, str]:
    root = _anchor_robofactory_imports(robofactory_root)

    import cv2
    import mani_skill
    import sapien
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

    env = None
    steps = 0
    policy_queries = 0
    bound_violations = 0
    success = False
    termination_reason = "max_steps"
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
        max_steps = (
            int(episode["max_episode_steps"])
            if args.max_steps is None
            else min(int(args.max_steps), int(episode["max_episode_steps"]))
        )

        if args.mode == "expert-replay":
            expert = _load_expert_actions(
                source, str(episode["trajectory"]), agent_names
            )
            horizon = min(len(next(iter(expert.values()))), max_steps)
            for index in range(horizon):
                env_action = {name: expert[name][index] for name in agent_names}
                bound_violations += _action_bound_violations(
                    env_action, env.action_space
                )
                _, _, terminated, truncated, info = env.step(env_action)
                steps += 1
                success = _as_bool(info["success"], label="info.success")
                if success:
                    termination_reason = "success"
                    break
                if _as_bool(terminated, label="terminated"):
                    termination_reason = "terminated"
                    break
                if _as_bool(truncated, label="truncated"):
                    termination_reason = "truncated"
                    break
        else:
            if policy is None:
                raise RuntimeError("FastWAM mode requires an initialized policy")
            episode_policy_seed = args.policy_seed + int(episode["task_index"])
            policy.start_episode(episode_policy_seed)
            raw_obs = env.unwrapped.get_obs()
            policy.update_obs(raw_obs, env.unwrapped.get_state_dict())
            while steps < max_steps:
                action_chunk = policy.get_action()
                policy_queries += 1
                execute = min(
                    int(args.exec_horizon), len(action_chunk), max_steps - steps
                )
                if execute < 1:
                    raise ValueError("Policy returned an empty executable action chunk")
                stop_chunk = False
                for index in range(execute):
                    env_action = _flat_action_to_dict(action_chunk[index], agent_names)
                    bound_violations += _action_bound_violations(
                        env_action, env.action_space
                    )
                    _, _, terminated, truncated, info = env.step(env_action)
                    steps += 1
                    success = _as_bool(info["success"], label="info.success")
                    raw_obs = env.unwrapped.get_obs()
                    policy.update_obs(raw_obs, env.unwrapped.get_state_dict())
                    if success:
                        termination_reason = "success"
                        stop_chunk = True
                        break
                    if _as_bool(terminated, label="terminated"):
                        termination_reason = "terminated"
                        stop_chunk = True
                        break
                    if _as_bool(truncated, label="truncated"):
                        termination_reason = "truncated"
                        stop_chunk = True
                        break
                if stop_chunk:
                    break

        return {
            "status": "completed",
            "mode": args.mode,
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
            "action_bound_violations": bound_violations,
            "termination_reason": termination_reason,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_seconds": time.monotonic() - start_time,
            "gpu_memory": _gpu_memory(),
        }
    finally:
        if env is not None:
            env.close()


def _required_fastwam_arguments(args: argparse.Namespace) -> None:
    required = [
        "checkpoint",
        "stats",
        "context_cache_dir",
        "model_cache_root",
    ]
    if args.gaussian_conditioning:
        required.extend(("policy_lightning_repo", "noposplat_checkpoint"))
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
        "schema_version": "fastwam-robofactory-eval-run-v2",
        "status": "running",
        "started_at": _utc_now(),
        "mode": args.mode,
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
                "status": "infrastructure_error",
                "mode": args.mode,
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
        "schema_version": "fastwam-robofactory-eval-summary-v2",
        "status": (
            "PASS"
            if infrastructure_errors == 0 and len(results) == len(selected)
            else "INFRASTRUCTURE_ERROR"
        ),
        "mode": args.mode,
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
