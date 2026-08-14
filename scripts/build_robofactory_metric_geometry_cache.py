"""Build exact-window observation-only metric geometry for RoboFactory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from fastwam.datasets.gaussian_cache.schema import FrameKey
from fastwam.datasets.metric_geometry import (
    METRIC_GEOMETRY_MAX_DEPTH_M,
    METRIC_GEOMETRY_SIZE,
    encode_metric_agent_geometry,
)
from fastwam.datasets.metric_geometry_cache import SCHEMA_NAME, SCHEMA_VERSION
from fastwam.datasets.robofactory_multi_robot import _split_fraction


TASK_CONFIGS = {"PlaceFood-rf": "configs/table/place_food.yaml"}
STAT_CMP_ALLOWLIST = "stat-cmp.allowlist"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _agent_sort_key(name: str) -> int:
    return int(name.rsplit("-", 1)[-1])


def _task_name_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if part.endswith("-rf"):
            return part
    return path.parent.name


def _regular_source(dataset_root: Path, relative: str) -> Path:
    source = (dataset_root / relative).resolve(strict=True)
    try:
        source.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"Source path escapes dataset root: {relative!r}") from error
    if not source.is_file() or source.suffix != ".h5":
        raise ValueError(f"Source must be a regular HDF5 file: {source}")
    return source


def _episode_metadata(source: Path) -> dict[int, Mapping[str, Any]]:
    sidecar = source.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"Trajectory sidecar lacks episodes: {sidecar}")
    result: dict[int, Mapping[str, Any]] = {}
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise TypeError(f"Invalid episode record in {sidecar}")
        episode_id = int(episode["episode_id"])
        if episode_id in result:
            raise ValueError(f"Duplicate episode_id={episode_id} in {sidecar}")
        result[episode_id] = episode
    return result


def _episode_id(trajectory: str) -> int:
    prefix, separator, suffix = trajectory.rpartition("_")
    if prefix != "traj" or not separator or not suffix.isdigit():
        raise ValueError(f"Expected trajectory name traj_<id>, got {trajectory!r}")
    return int(suffix)


def _window_timesteps(
    *,
    source_path: str,
    trajectory: str,
    action_count: int,
    action_horizon: int,
    train_window_stride: int,
    val_window_stride: int,
    split_seed: int,
    val_set_proportion: float,
) -> tuple[str, tuple[int, ...]]:
    split_key = f"{source_path}:{trajectory}"
    is_val = _split_fraction(split_key, split_seed) < val_set_proportion
    split = "val" if is_val else "train"
    stride = val_window_stride if is_val else train_window_stride
    timesteps = tuple(range(0, action_count - action_horizon + 1, stride))
    return split, timesteps


def _build_environment(robofactory_root: Path, task_name: str):
    if task_name not in TASK_CONFIGS:
        raise KeyError(f"Unsupported task: {task_name}")
    for import_root in (robofactory_root.parent, robofactory_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    os.chdir(robofactory_root)
    __import__("tasks")
    import gymnasium as gym

    config = robofactory_root / TASK_CONFIGS[task_name]
    if not config.is_file():
        raise FileNotFoundError(f"Task config is missing: {config}")
    return gym.make(
        task_name,
        config=str(config.relative_to(robofactory_root)),
        obs_mode="rgb+depth",
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


def _fsync_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root).expanduser().resolve(strict=True)
    robofactory_root = Path(args.robofactory_root).expanduser().resolve(strict=True)
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Use a new versioned output path: {output_root}")
    if args.num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    if args.train_window_stride < 1 or args.val_window_stride < 1:
        raise ValueError("window strides must be positive")
    if not 0.0 <= args.val_set_proportion < 1.0:
        raise ValueError("val_set_proportion must be in [0,1)")
    if args.limit_trajectories is not None and args.limit_trajectories < 1:
        raise ValueError("limit_trajectories must be positive")

    sources = [
        path
        for path in sorted(dataset_root.rglob("*.h5"))
        if _task_name_from_path(path) == args.task_name
    ]
    if not sources:
        raise FileNotFoundError(
            f"No {args.task_name} HDF5 sources found under {dataset_root}"
        )

    staging = output_root.with_name(f".{output_root.name}.partial-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Staging path already exists: {staging}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    data_path = staging / "frames.f16"
    counters: Counter[str] = Counter()
    entries: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    action_horizon = args.num_frames - 1
    env = None
    started_at = _utc_now()
    try:
        from mani_skill.trajectory import utils as trajectory_utils

        env = _build_environment(robofactory_root, args.task_name)
        with data_path.open("xb") as output:
            trajectory_ordinal = 0
            for discovered_source in sources:
                source_path = discovered_source.relative_to(dataset_root).as_posix()
                source = _regular_source(dataset_root, source_path)
                episodes = _episode_metadata(source)
                source_stat = source.stat()
                sidecar = source.with_suffix(".json")
                sidecar_stat = sidecar.stat()
                source_records.append(
                    {
                        "source_path": source_path,
                        "h5_bytes": int(source_stat.st_size),
                        "h5_mtime_ns": int(source_stat.st_mtime_ns),
                        "sidecar_path": sidecar.relative_to(dataset_root).as_posix(),
                        "sidecar_bytes": int(sidecar_stat.st_size),
                        "sidecar_mtime_ns": int(sidecar_stat.st_mtime_ns),
                    }
                )
                with h5py.File(source, "r") as handle:
                    for trajectory in sorted(handle.keys()):
                        group = handle[trajectory]
                        if "actions" not in group:
                            continue
                        agent_names = tuple(
                            sorted(group["actions"].keys(), key=_agent_sort_key)
                        )
                        if len(agent_names) != args.required_agent_count:
                            continue
                        action_count = int(group["actions"][agent_names[0]].shape[0])
                        split, timesteps = _window_timesteps(
                            source_path=source_path,
                            trajectory=trajectory,
                            action_count=action_count,
                            action_horizon=action_horizon,
                            train_window_stride=args.train_window_stride,
                            val_window_stride=args.val_window_stride,
                            split_seed=args.split_seed,
                            val_set_proportion=args.val_set_proportion,
                        )
                        if not timesteps:
                            continue
                        episode_id = _episode_id(trajectory)
                        if episode_id not in episodes:
                            raise KeyError(
                                f"Missing episode metadata for {source_path}:{trajectory}"
                            )
                        states = trajectory_utils.dict_to_list_of_dicts(
                            group["env_states"]
                        )
                        if len(states) != action_count + 1:
                            raise ValueError(
                                f"Expected states=actions+1 for {source_path}:{trajectory}, "
                                f"got states={len(states)} actions={action_count}"
                            )
                        _reset_environment(env, episodes[episode_id])
                        camera_names = tuple(
                            f"head_camera_agent{_agent_sort_key(name)}"
                            for name in agent_names
                        )
                        for timestep in timesteps:
                            env.unwrapped.set_state_dict(states[timestep])
                            observation = env.unwrapped.get_obs()
                            frames = encode_metric_agent_geometry(
                                observation,
                                camera_names,
                                output_size=tuple(args.output_size),
                            )
                            expected_shape = (
                                len(agent_names),
                                13,
                                args.output_size[0],
                                args.output_size[1],
                            )
                            if tuple(frames.shape) != expected_shape:
                                raise RuntimeError(
                                    f"Metric frame shape mismatch: expected={expected_shape} "
                                    f"observed={tuple(frames.shape)}"
                                )
                            array = np.asarray(frames.numpy(), dtype=np.dtype("<f2"))
                            for agent_index, agent_name in enumerate(agent_names):
                                frame = np.ascontiguousarray(array[agent_index])
                                frame.tofile(output)
                                key = FrameKey(
                                    source_path,
                                    trajectory,
                                    timestep,
                                    agent_name,
                                )
                                entries.append({**key.to_dict(), "offset": len(entries)})
                            counters["windows"] += 1
                            counters[f"{split}_windows"] += 1
                            counters["frames"] += len(agent_names)
                            if counters["windows"] % args.progress_every == 0:
                                print(
                                    json.dumps(
                                        {
                                            "event": "metric_cache_progress",
                                            "windows": counters["windows"],
                                            "frames": counters["frames"],
                                            "source_path": source_path,
                                            "trajectory": trajectory,
                                            "timestep": timestep,
                                        },
                                        sort_keys=True,
                                    ),
                                    flush=True,
                                )
                        counters["trajectories"] += 1
                        trajectory_ordinal += 1
                        if (
                            args.limit_trajectories is not None
                            and trajectory_ordinal >= args.limit_trajectories
                        ):
                            break
                if (
                    args.limit_trajectories is not None
                    and trajectory_ordinal >= args.limit_trajectories
                ):
                    break
            output.flush()
            os.fsync(output.fileno())

        if not entries:
            raise RuntimeError("Metric geometry selection produced no frames")
        data_stat = data_path.stat()
        frame_shape = [13, *map(int, args.output_size)]
        expected_bytes = len(entries) * int(np.prod(frame_shape)) * 2
        if data_stat.st_size != expected_bytes:
            raise RuntimeError(
                f"Metric geometry byte count mismatch: expected={expected_bytes} "
                f"observed={data_stat.st_size}"
            )
        manifest = {
            "schema_name": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "provenance_mode": "stat_cmp",
            "dtype": "float16",
            "byte_order": "little",
            "frame_shape": frame_shape,
            "data": {
                "path": data_path.name,
                "frames": len(entries),
                "bytes": int(data_stat.st_size),
                "mtime_ns": int(data_stat.st_mtime_ns),
            },
            "selection": {
                "task_name": args.task_name,
                "required_agent_count": args.required_agent_count,
                "action_horizon": action_horizon,
                "split_seed": args.split_seed,
                "val_set_proportion": args.val_set_proportion,
                "train_window_stride": args.train_window_stride,
                "val_window_stride": args.val_window_stride,
                "limit_trajectories": args.limit_trajectories,
            },
            "metric_geometry": {
                "source": "maniskill_calibrated_depth",
                "coordinate_frame": "world",
                "input_size": [240, 320],
                "output_size": list(map(int, args.output_size)),
                "depth_scale": 0.001,
                "max_depth_m": METRIC_GEOMETRY_MAX_DEPTH_M,
                "surface_band_m": 0.03,
                "channels": "xyz_mean_covariance_row_major_valid",
            },
            "sources": source_records,
            "counts": dict(sorted(counters.items())),
            "entries": entries,
        }
        _fsync_text(
            staging / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        _fsync_text(staging / "COMPLETE", "complete\n")
        _fsync_text(
            staging / STAT_CMP_ALLOWLIST,
            "metadata frames.f16\nmetadata manifest.json\nmetadata COMPLETE\n",
        )
        os.replace(staging, output_root)
    except BaseException:
        (staging / "FAILED").write_text(_utc_now() + "\n", encoding="utf-8")
        raise
    finally:
        if env is not None:
            env.close()

    return {
        "cache_root": str(output_root),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "frame_shape": [13, *map(int, args.output_size)],
        **dict(sorted(counters.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--robofactory-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task-name", default="PlaceFood-rf", choices=TASK_CONFIGS)
    parser.add_argument("--required-agent-count", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--train-window-stride", type=int, default=16)
    parser.add_argument("--val-window-stride", type=int, default=32)
    parser.add_argument("--val-set-proportion", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--output-size",
        type=int,
        nargs=2,
        default=METRIC_GEOMETRY_SIZE,
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument("--limit-trajectories", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(build_cache(parse_args()), indent=2, sort_keys=True))
