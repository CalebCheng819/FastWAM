"""Direct HDF5 adapter for synchronized RoboFactory multi-robot demos."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from omegaconf import DictConfig, OmegaConf

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


DEFAULT_INSTRUCTIONS = {
    "ThreeRobotsPlaceShoes-rf": "three robots collaboratively place the shoes in their target locations",
    "ThreeRobotsStackCube-rf": "three robots collaboratively stack the cubes",
    "FourRobotsStackCube-rf": "four robots collaboratively stack the cubes",
}


def _plain_mapping(value: Optional[Mapping[str, str] | DictConfig]) -> dict[str, str]:
    if value is None:
        return dict(DEFAULT_INSTRUCTIONS)
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"instruction_map must be mapping-like, got {type(value)}")
    merged = dict(DEFAULT_INSTRUCTIONS)
    merged.update({str(key): str(text) for key, text in value.items()})
    return merged


def _task_name_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if part.endswith("-rf"):
            return part
    return path.parent.name


def _agent_sort_key(name: str):
    try:
        return int(name.rsplit("-", 1)[-1])
    except ValueError:
        return name


def _split_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


class RoboFactoryMultiRobotDataset(torch.utils.data.Dataset):
    """Read fixed-horizon windows without converting or duplicating HDF5 data.

    Returned tensors use a fixed agent axis and are compatible with
    :class:`FastWAMMultiRobot`:

    - ``video``: global camera, ``[3, T_video, H, W]`` in ``[-1, 1]``
    - ``action``: ``[max_agents, horizon, 8]``
    - ``agent_state``: current qpos+qvel, ``[max_agents, 18]``
    - ``agent_mask``: valid robot slots
    """

    def __init__(
        self,
        root_dir: str,
        *,
        num_frames: int = 33,
        action_video_freq_ratio: int = 4,
        video_size: list[int] | tuple[int, int] = (224, 320),
        max_agents: int = 4,
        action_dim: int = 8,
        state_dim: int = 18,
        window_stride: int = 16,
        val_set_proportion: float = 0.1,
        is_training_set: bool = True,
        split_seed: int = 42,
        randomize_agent_order: bool = True,
        pretrained_norm_stats: Optional[str] = None,
        text_embedding_cache_dir: Optional[str] = None,
        context_len: int = 128,
        instruction_map: Optional[Mapping[str, str] | DictConfig] = None,
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        if not self.root_dir.exists():
            raise FileNotFoundError(f"RoboFactory root does not exist: {self.root_dir}")
        self.num_frames = int(num_frames)
        self.action_horizon = self.num_frames - 1
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.max_agents = int(max_agents)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.window_stride = int(window_stride)
        self.val_set_proportion = float(val_set_proportion)
        self.is_training_set = bool(is_training_set)
        self.split_seed = int(split_seed)
        self.randomize_agent_order = bool(randomize_agent_order)
        self.text_embedding_cache_dir = (
            None if text_embedding_cache_dir is None else Path(text_embedding_cache_dir).expanduser()
        )
        self.context_len = int(context_len)
        self.instruction_map = _plain_mapping(instruction_map)
        self._h5_handles: dict[str, h5py.File] = {}

        if self.action_horizon <= 0 or self.action_horizon % self.action_video_freq_ratio:
            raise ValueError(
                "num_frames-1 must be positive and divisible by action_video_freq_ratio, "
                f"got {self.action_horizon} and {self.action_video_freq_ratio}"
            )
        self.video_indices = list(
            range(0, self.num_frames, self.action_video_freq_ratio)
        )
        if (len(self.video_indices) - 1) % 4:
            raise ValueError(
                f"Video transition count must be divisible by 4, got {len(self.video_indices) - 1}"
            )
        if self.video_size[0] % 16 or self.video_size[1] % 16:
            raise ValueError(f"video_size must be divisible by 16, got {self.video_size}")
        if self.max_agents < 1 or self.window_stride < 1:
            raise ValueError("max_agents and window_stride must be positive")
        if not 0.0 <= self.val_set_proportion < 1.0:
            raise ValueError("val_set_proportion must be in [0,1)")

        self.stats = self._load_stats(pretrained_norm_stats)
        self.entries = self._build_index()
        if not self.entries:
            split = "train" if self.is_training_set else "val"
            raise RuntimeError(f"No {split} windows found under {self.root_dir}")
        logger.info(
            "RoboFactory %s dataset: windows=%d root=%s max_agents=%d horizon=%d video_frames=%d",
            "train" if self.is_training_set else "val",
            len(self.entries),
            self.root_dir,
            self.max_agents,
            self.action_horizon,
            len(self.video_indices),
        )

    def _load_stats(self, stats_path: Optional[str]) -> dict[str, torch.Tensor]:
        if not stats_path:
            raise ValueError(
                "`pretrained_norm_stats` is required. Run "
                "`python scripts/compute_robofactory_stats.py --root-dir ... --output ...`."
            )
        path = Path(stats_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = {}
        for kind, expected_dim in (("action", self.action_dim), ("state", self.state_dim)):
            if kind not in payload:
                raise KeyError(f"Stats file {path} is missing {kind!r}")
            mean = torch.as_tensor(payload[kind]["mean"], dtype=torch.float32)
            std = torch.as_tensor(payload[kind]["std"], dtype=torch.float32)
            if mean.shape != (expected_dim,) or std.shape != (expected_dim,):
                raise ValueError(
                    f"{kind} stats must have dim {expected_dim}, got {tuple(mean.shape)}, {tuple(std.shape)}"
                )
            result[f"{kind}_mean"] = mean
            result[f"{kind}_std"] = std.clamp(min=1e-6)
        return result

    def _build_index(self) -> list[dict[str, Any]]:
        h5_paths = sorted(self.root_dir.rglob("*.h5"))
        if not h5_paths:
            raise FileNotFoundError(f"No .h5 files found under {self.root_dir}")
        entries: list[dict[str, Any]] = []
        trajectory_count = 0
        split_trajectory_count = 0
        for h5_path in h5_paths:
            task_name = _task_name_from_path(h5_path)
            with h5py.File(h5_path, "r") as handle:
                for trajectory_name in sorted(handle.keys()):
                    group = handle[trajectory_name]
                    if "actions" not in group:
                        continue
                    agent_names = sorted(group["actions"].keys(), key=_agent_sort_key)
                    if not agent_names:
                        continue
                    length = int(group["actions"][agent_names[0]].shape[0])
                    trajectory_count += 1
                    split_key = f"{h5_path.relative_to(self.root_dir)}:{trajectory_name}"
                    is_val = _split_fraction(split_key, self.split_seed) < self.val_set_proportion
                    if is_val == self.is_training_set:
                        continue
                    if len(agent_names) > self.max_agents:
                        raise ValueError(
                            f"{h5_path}:{trajectory_name} has {len(agent_names)} agents, "
                            f"exceeding max_agents={self.max_agents}"
                        )
                    if length < self.action_horizon:
                        continue
                    split_trajectory_count += 1
                    for start in range(
                        0,
                        length - self.action_horizon + 1,
                        self.window_stride,
                    ):
                        entries.append(
                            {
                                "path": str(h5_path),
                                "trajectory": trajectory_name,
                                "start": start,
                                "task_name": task_name,
                                "agent_names": tuple(agent_names),
                            }
                        )
        logger.info(
            "Indexed %d/%d trajectories for selected split across %d HDF5 files.",
            split_trajectory_count,
            trajectory_count,
            len(h5_paths),
        )
        return entries

    def __len__(self):
        return len(self.entries)

    def _handle(self, path: str) -> h5py.File:
        handle = self._h5_handles.get(path)
        if handle is None or not handle.id.valid:
            handle = h5py.File(path, "r")
            self._h5_handles[path] = handle
        return handle

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set")
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = self.text_embedding_cache_dir / (
            f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing text embedding cache {cache_path}. "
                "Run scripts/precompute_text_embeds.py for each RoboFactory instruction."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"].clone()
        context_mask = payload["mask"].bool().clone()
        if context.shape[0] != self.context_len or context_mask.shape != (self.context_len,):
            raise ValueError(f"Invalid cached text shapes in {cache_path}")
        context[~context_mask] = 0
        # Preserve upstream Wan2.2 behavior: padding is zero-valued but visible.
        context_mask = torch.ones_like(context_mask)
        return context, context_mask

    def __getitem__(self, index: int):
        entry = self.entries[index]
        group = self._handle(entry["path"])[entry["trajectory"]]
        start = int(entry["start"])
        agent_names = list(entry["agent_names"])
        num_agents = len(agent_names)

        rgb_indices = np.asarray([start + offset for offset in self.video_indices], dtype=np.int64)
        rgb = group["obs/sensor_data/head_camera_global/rgb"][rgb_indices]
        video = torch.from_numpy(np.asarray(rgb)).permute(0, 3, 1, 2)
        video = transforms_F.resize(
            video,
            size=list(self.video_size),
            interpolation=transforms_F.InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        video = video.div(127.5).sub(1.0).permute(1, 0, 2, 3).contiguous()

        action = torch.zeros(
            (self.max_agents, self.action_horizon, self.action_dim), dtype=torch.float32
        )
        agent_state = torch.zeros((self.max_agents, self.state_dim), dtype=torch.float32)
        agent_mask = torch.zeros((self.max_agents,), dtype=torch.bool)
        agent_ids = torch.zeros((self.max_agents,), dtype=torch.long)

        order = torch.arange(num_agents)
        if self.is_training_set and self.randomize_agent_order and num_agents > 1:
            order = torch.randperm(num_agents)
        for slot, original_index_tensor in enumerate(order):
            original_index = int(original_index_tensor)
            agent_name = agent_names[original_index]
            raw_action = torch.from_numpy(
                np.asarray(
                    group[f"actions/{agent_name}"][start : start + self.action_horizon],
                    dtype=np.float32,
                )
            )
            qpos = np.asarray(group[f"obs/agent/{agent_name}/qpos"][start], dtype=np.float32)
            qvel = np.asarray(group[f"obs/agent/{agent_name}/qvel"][start], dtype=np.float32)
            raw_state = torch.from_numpy(np.concatenate([qpos, qvel], axis=0))
            action[slot] = (raw_action - self.stats["action_mean"]) / self.stats["action_std"]
            agent_state[slot] = (raw_state - self.stats["state_mean"]) / self.stats["state_std"]
            agent_mask[slot] = True
            agent_ids[slot] = original_index

        task_name = str(entry["task_name"])
        instruction = self.instruction_map.get(task_name)
        if instruction is None:
            raise KeyError(
                f"No instruction for task {task_name!r}; add it to data.instruction_map."
            )
        prompt = DEFAULT_PROMPT.format(task=instruction)
        context, context_mask = self._get_cached_text_context(prompt)

        return {
            "video": video,
            "action": action,
            "agent_state": agent_state,
            "agent_mask": agent_mask,
            "agent_ids": agent_ids,
            "action_is_pad": torch.zeros(
                (self.max_agents, self.action_horizon), dtype=torch.bool
            ),
            "image_is_pad": torch.zeros((len(self.video_indices),), dtype=torch.bool),
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "task_name": task_name,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_handles"] = {}
        return state

    def __del__(self):
        for handle in getattr(self, "_h5_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass


def compute_robofactory_stats(root_dir: str) -> dict[str, Any]:
    """Compute shared per-dimension z-score statistics over all robots."""

    root = Path(root_dir).expanduser().resolve()
    h5_paths = sorted(root.rglob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found under {root}")

    accumulators = {
        "action": {"count": 0, "sum": None, "sum_sq": None, "min": None, "max": None},
        "state": {"count": 0, "sum": None, "sum_sq": None, "min": None, "max": None},
    }

    def update(kind: str, array: np.ndarray):
        array = np.asarray(array, dtype=np.float64).reshape(-1, array.shape[-1])
        acc = accumulators[kind]
        values_sum = array.sum(axis=0)
        values_sum_sq = np.square(array).sum(axis=0)
        values_min = array.min(axis=0)
        values_max = array.max(axis=0)
        if acc["sum"] is None:
            acc["sum"] = values_sum
            acc["sum_sq"] = values_sum_sq
            acc["min"] = values_min
            acc["max"] = values_max
        else:
            acc["sum"] += values_sum
            acc["sum_sq"] += values_sum_sq
            acc["min"] = np.minimum(acc["min"], values_min)
            acc["max"] = np.maximum(acc["max"], values_max)
        acc["count"] += int(array.shape[0])

    trajectory_count = 0
    for path in h5_paths:
        with h5py.File(path, "r") as handle:
            for trajectory_name in sorted(handle.keys()):
                group = handle[trajectory_name]
                if "actions" not in group:
                    continue
                trajectory_count += 1
                for agent_name in sorted(group["actions"].keys(), key=_agent_sort_key):
                    update("action", group[f"actions/{agent_name}"][:])
                    qpos = group[f"obs/agent/{agent_name}/qpos"][:]
                    qvel = group[f"obs/agent/{agent_name}/qvel"][:]
                    update("state", np.concatenate([qpos, qvel], axis=-1))

    result: dict[str, Any] = {
        "source_root": str(root),
        "files": len(h5_paths),
        "trajectories": trajectory_count,
    }
    for kind, acc in accumulators.items():
        if not acc["count"]:
            raise RuntimeError(f"No {kind} values found under {root}")
        mean = acc["sum"] / acc["count"]
        variance = np.maximum(acc["sum_sq"] / acc["count"] - np.square(mean), 0.0)
        result[kind] = {
            "count": acc["count"],
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "min": acc["min"].tolist(),
            "max": acc["max"].tolist(),
        }
    return result
