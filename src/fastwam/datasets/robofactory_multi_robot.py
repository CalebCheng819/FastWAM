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
    "StrikeCubeHard-rf": "two robots collaboratively strike the cube to the target",
    "PlaceFood-rf": "two robots collaboratively place the food in the target location",
    "PlaceCubeInCup-rf": "two robots collaboratively place the cube in the cup",
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

    Returned tensors use the trajectory's native agent count and are compatible
    with :class:`FastWAMMultiRobot`.  Batches must group samples with the same
    agent count (see ``ResumableAgentCountBatchSampler``):

    - ``video``: global camera, ``[3, T_video, H, W]`` in ``[-1, 1]``
    - ``action``: ``[num_agents, horizon, 8]``
    - ``agent_state``: current qpos+qvel, ``[num_agents, 18]``
    - ``agent_geometry``: root pose ``[x,y,z,qw,qx,qy,qz]`` with a normalized,
      sign-canonical quaternion,
      ``[num_agents, 7]``
    - ``agent_ids``: original within-trajectory ordering, used only by optional
      dynamic agent encodings
    """

    def __init__(
        self,
        root_dir: str,
        *,
        num_frames: int = 33,
        action_video_freq_ratio: int = 4,
        load_future_video: bool = True,
        video_size: list[int] | tuple[int, int] = (224, 320),
        action_dim: int = 8,
        state_dim: int = 18,
        agent_geometry_dim: int = 7,
        window_stride: int = 16,
        val_set_proportion: float = 0.1,
        is_training_set: bool = True,
        split_seed: int = 42,
        randomize_agent_order: bool = True,
        required_agent_counts: Optional[list[int] | tuple[int, ...]] = None,
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
        self.load_future_video = bool(load_future_video)
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.agent_geometry_dim = int(agent_geometry_dim)
        self.window_stride = int(window_stride)
        self.val_set_proportion = float(val_set_proportion)
        self.is_training_set = bool(is_training_set)
        self.split_seed = int(split_seed)
        self.randomize_agent_order = bool(randomize_agent_order)
        self.required_agent_counts = (
            None
            if required_agent_counts is None
            else tuple(sorted({int(count) for count in required_agent_counts}))
        )
        if self.required_agent_counts is not None and any(
            count < 1 for count in self.required_agent_counts
        ):
            raise ValueError("required_agent_counts must contain only positive integers")
        self.text_embedding_cache_dir = (
            None if text_embedding_cache_dir is None else Path(text_embedding_cache_dir).expanduser()
        )
        self.context_len = int(context_len)
        self.instruction_map = _plain_mapping(instruction_map)
        self._h5_handles: dict[str, h5py.File] = {}
        self._text_context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        if self.action_horizon <= 0 or self.action_horizon % self.action_video_freq_ratio:
            raise ValueError(
                "num_frames-1 must be positive and divisible by action_video_freq_ratio, "
                f"got {self.action_horizon} and {self.action_video_freq_ratio}"
            )
        full_video_indices = list(
            range(0, self.num_frames, self.action_video_freq_ratio)
        )
        if (len(full_video_indices) - 1) % 4:
            raise ValueError(
                f"Video transition count must be divisible by 4, got {len(full_video_indices) - 1}"
            )
        self.video_indices = full_video_indices if self.load_future_video else [0]
        if self.video_size[0] % 16 or self.video_size[1] % 16:
            raise ValueError(f"video_size must be divisible by 16, got {self.video_size}")
        if self.window_stride < 1:
            raise ValueError("window_stride must be positive")
        if self.agent_geometry_dim != 7:
            raise ValueError(
                "RoboFactory agent geometry currently uses the 7D root pose "
                "[x,y,z,qw,qx,qy,qz]; "
                f"got agent_geometry_dim={self.agent_geometry_dim}"
            )
        if not 0.0 <= self.val_set_proportion < 1.0:
            raise ValueError("val_set_proportion must be in [0,1)")

        self.stats = self._load_stats(pretrained_norm_stats)
        self.entries = self._build_index()
        if not self.entries:
            split = "train" if self.is_training_set else "val"
            raise RuntimeError(f"No {split} windows found under {self.root_dir}")
        self.agent_counts = tuple(int(entry["agent_count"]) for entry in self.entries)
        self.task_ids = tuple(str(entry["task_name"]) for entry in self.entries)
        observed_agent_counts = set(self.agent_counts)
        if self.required_agent_counts is not None:
            declared_agent_counts = set(self.required_agent_counts)
            missing_counts = sorted(declared_agent_counts - observed_agent_counts)
            unexpected_counts = sorted(observed_agent_counts - declared_agent_counts)
            if missing_counts or unexpected_counts:
                raise RuntimeError(
                    "Dataset split does not match the declared real-agent cardinality scope: "
                    f"missing={missing_counts}, unexpected={unexpected_counts}, "
                    f"declared={sorted(declared_agent_counts)}, "
                    f"observed={sorted(observed_agent_counts)} under {self.root_dir}."
                )
        self._validate_stats_provenance()
        self._preflight_text_embedding_cache()
        logger.info(
            "RoboFactory %s dataset: windows=%d root=%s agent_counts=%s horizon=%d "
            "video_frames=%d load_future_video=%s",
            "train" if self.is_training_set else "val",
            len(self.entries),
            self.root_dir,
            sorted(set(self.agent_counts)),
            self.action_horizon,
            len(self.video_indices),
            self.load_future_video,
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
        if not isinstance(payload, Mapping):
            raise TypeError(f"Stats file {path} must contain a JSON object.")
        self._stats_path = path.resolve()
        self._stats_metadata = {
            key: payload[key]
            for key in ("source_root", "files", "trajectories", "cardinality")
            if key in payload
        }
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
        trajectories_by_agent_count: dict[int, int] = {}
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
                    agent_count = len(agent_names)
                    trajectories_by_agent_count[agent_count] = (
                        trajectories_by_agent_count.get(agent_count, 0) + 1
                    )
                    split_key = f"{h5_path.relative_to(self.root_dir)}:{trajectory_name}"
                    is_val = _split_fraction(split_key, self.split_seed) < self.val_set_proportion
                    if is_val == self.is_training_set:
                        continue
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
                                "agent_count": agent_count,
                            }
                        )
        self._source_metadata = {
            "source_root": str(self.root_dir),
            "files": len(h5_paths),
            "trajectories": trajectory_count,
            "cardinality": {
                "agent_counts": sorted(trajectories_by_agent_count),
                "trajectories_by_agent_count": {
                    str(count): trajectories_by_agent_count[count]
                    for count in sorted(trajectories_by_agent_count)
                },
            },
        }
        logger.info(
            "Indexed %d/%d trajectories for selected split across %d HDF5 files.",
            split_trajectory_count,
            trajectory_count,
            len(h5_paths),
        )
        return entries

    @staticmethod
    def _metadata_count(value: Any, *, field: str, stats_path: Path) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"Stats metadata {field!r} in {stats_path} must be a non-negative integer, "
                f"got {value!r}."
            )
        return int(value)

    def _validate_stats_provenance(self) -> None:
        """Reject stale or partial stats before any training sample is read."""

        metadata = self._stats_metadata
        expected = self._source_metadata
        stats_path = self._stats_path

        if self.required_agent_counts is not None and "cardinality" not in metadata:
            raise ValueError(
                f"Stats file {stats_path} has no cardinality metadata and cannot be used "
                f"with required_agent_counts={list(self.required_agent_counts)}. Recompute "
                "unified stats with scripts/compute_robofactory_stats.py."
            )

        if "source_root" in metadata:
            source_root = Path(str(metadata["source_root"])).expanduser().resolve()
            if source_root != self.root_dir:
                raise ValueError(
                    f"Stats source_root mismatch in {stats_path}: "
                    f"stats={source_root} dataset={self.root_dir}."
                )
        for field in ("files", "trajectories"):
            if field not in metadata:
                continue
            actual = self._metadata_count(metadata[field], field=field, stats_path=stats_path)
            if actual != expected[field]:
                raise ValueError(
                    f"Stats {field} mismatch in {stats_path}: "
                    f"stats={actual} dataset={expected[field]}."
                )

        if "cardinality" not in metadata:
            return
        cardinality = metadata["cardinality"]
        if not isinstance(cardinality, Mapping):
            raise TypeError(
                f"Stats cardinality metadata in {stats_path} must be a JSON object."
            )
        required_fields = {"agent_counts", "trajectories_by_agent_count"}
        missing_fields = sorted(required_fields - set(cardinality))
        if missing_fields:
            raise KeyError(
                f"Stats cardinality metadata in {stats_path} is missing {missing_fields}."
            )
        raw_counts = cardinality["agent_counts"]
        if not isinstance(raw_counts, list) or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in raw_counts
        ):
            raise ValueError(
                f"Stats cardinality.agent_counts in {stats_path} must be a list of "
                f"positive integers, got {raw_counts!r}."
            )
        stats_counts = sorted(set(int(count) for count in raw_counts))
        if stats_counts != expected["cardinality"]["agent_counts"]:
            raise ValueError(
                f"Stats agent cardinalities mismatch in {stats_path}: "
                f"stats={stats_counts} dataset={expected['cardinality']['agent_counts']}."
            )
        if self.required_agent_counts is not None:
            missing_required = sorted(set(self.required_agent_counts) - set(stats_counts))
            if missing_required:
                raise ValueError(
                    f"Stats file {stats_path} does not cover required agent cardinalities "
                    f"{missing_required}."
                )

        raw_trajectory_counts = cardinality["trajectories_by_agent_count"]
        if not isinstance(raw_trajectory_counts, Mapping):
            raise TypeError(
                f"Stats cardinality.trajectories_by_agent_count in {stats_path} "
                "must be a JSON object."
            )
        stats_trajectory_counts = {
            str(count): self._metadata_count(
                value,
                field=f"cardinality.trajectories_by_agent_count.{count}",
                stats_path=stats_path,
            )
            for count, value in raw_trajectory_counts.items()
        }
        if stats_trajectory_counts != expected["cardinality"]["trajectories_by_agent_count"]:
            raise ValueError(
                f"Stats trajectory cardinality mismatch in {stats_path}: "
                f"stats={stats_trajectory_counts} "
                f"dataset={expected['cardinality']['trajectories_by_agent_count']}."
            )

    def _text_cache_path(self, prompt: str) -> Path:
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set")
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return self.text_embedding_cache_dir / (
            f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )

    def _load_cached_text_context(
        self, prompt: str, cache_path: Path
    ) -> tuple[torch.Tensor, torch.Tensor]:
        payload = torch.load(cache_path, map_location="cpu")
        if not isinstance(payload, Mapping) or "context" not in payload or "mask" not in payload:
            raise ValueError(f"Invalid cached text payload in {cache_path}")
        context = payload["context"].clone()
        context_mask = payload["mask"].bool().clone()
        if context.shape[0] != self.context_len or context_mask.shape != (self.context_len,):
            raise ValueError(f"Invalid cached text shapes in {cache_path}")
        context[~context_mask] = 0
        # Preserve upstream Wan2.2 behavior: padding is zero-valued but visible.
        context_mask = torch.ones_like(context_mask)
        return context, context_mask

    def _preflight_text_embedding_cache(self) -> None:
        """Validate every prompt referenced by the indexed split at construction."""

        task_names = sorted(set(self.task_ids))
        missing_instructions = [
            task_name for task_name in task_names if task_name not in self.instruction_map
        ]
        if missing_instructions:
            raise KeyError(
                "Missing instructions for indexed RoboFactory tasks: "
                f"{missing_instructions}. Add them to data.instruction_map."
            )
        if self.text_embedding_cache_dir is None:
            raise ValueError(
                "`text_embedding_cache_dir` is required for all indexed RoboFactory prompts."
            )

        prompt_records = []
        for task_name in task_names:
            prompt = DEFAULT_PROMPT.format(task=self.instruction_map[task_name])
            prompt_records.append((task_name, prompt, self._text_cache_path(prompt)))
        missing_cache = [
            (task_name, str(cache_path))
            for task_name, _, cache_path in prompt_records
            if not cache_path.is_file()
        ]
        if missing_cache:
            details = "; ".join(
                f"task={task_name!r} path={cache_path}" for task_name, cache_path in missing_cache
            )
            required_counts = (
                "all observed agent counts"
                if self.required_agent_counts is None
                else "/".join(str(count) for count in self.required_agent_counts)
            )
            raise FileNotFoundError(
                "Missing text embedding caches for indexed RoboFactory prompts: "
                f"{details}. Run scripts/precompute_text_embeds.py for the unified "
                f"N={required_counts} config."
            )

        for _, prompt, cache_path in prompt_records:
            self._text_context_cache[prompt] = self._load_cached_text_context(
                prompt, cache_path
            )

    def __len__(self):
        return len(self.entries)

    def get_agent_count(self, index: int) -> int:
        """Return the real number of agents without opening the HDF5 file."""

        return int(self.entries[index]["agent_count"])

    def get_task_id(self, index: int) -> str:
        """Return the stable task-directory identifier without opening HDF5."""

        return str(self.entries[index]["task_name"])

    @staticmethod
    def _articulation_name(agent_name: str) -> str:
        """Map ``panda-i`` action/state names to RoboFactory articulation names."""

        prefix, separator, suffix = agent_name.rpartition("-")
        if not separator or not suffix.isdigit():
            raise ValueError(
                "Expected RoboFactory agent name ending in a numeric id, "
                f"got {agent_name!r}"
            )
        return f"{prefix}-agent-{suffix}"

    @staticmethod
    def _canonicalize_root_pose(root_pose: torch.Tensor) -> torch.Tensor:
        """Remove quaternion scale/sign ambiguity while preserving physical pose."""

        root_pose = root_pose.clone()
        quaternion = root_pose[3:7]
        norm = torch.linalg.vector_norm(quaternion)
        if not bool(torch.isfinite(norm).item()) or float(norm) < 1e-8:
            raise ValueError(f"Invalid root-pose quaternion: {quaternion.tolist()}")
        quaternion = quaternion / norm
        pivot = int(torch.argmax(quaternion.abs()).item())
        if float(quaternion[pivot]) < 0.0:
            quaternion = -quaternion
        root_pose[3:7] = quaternion
        return root_pose

    def _handle(self, path: str) -> h5py.File:
        handle = self._h5_handles.get(path)
        if handle is None or not handle.id.valid:
            handle = h5py.File(path, "r")
            self._h5_handles[path] = handle
        return handle

    def _get_cached_text_context(self, prompt: str):
        cached = self._text_context_cache.get(prompt)
        if cached is None:
            cache_path = self._text_cache_path(prompt)
            if not cache_path.is_file():
                raise FileNotFoundError(f"Missing text embedding cache {cache_path}.")
            cached = self._load_cached_text_context(prompt, cache_path)
            self._text_context_cache[prompt] = cached
        context, context_mask = cached
        return context.clone(), context_mask.clone()

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

        action = torch.empty(
            (num_agents, self.action_horizon, self.action_dim), dtype=torch.float32
        )
        agent_state = torch.empty((num_agents, self.state_dim), dtype=torch.float32)
        agent_geometry = torch.empty(
            (num_agents, self.agent_geometry_dim), dtype=torch.float32
        )
        agent_ids = torch.empty((num_agents,), dtype=torch.long)

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
            articulation_name = self._articulation_name(agent_name)
            geometry_path = f"env_states/articulations/{articulation_name}"
            if geometry_path not in group:
                raise KeyError(
                    f"Missing agent root-pose geometry {geometry_path!r} in "
                    f"{entry['path']}:{entry['trajectory']}"
                )
            articulation_state = group[geometry_path]
            if articulation_state.ndim != 2 or articulation_state.shape[1] < 7:
                raise ValueError(
                    f"Expected {geometry_path} to be [T,D>=7], got "
                    f"{tuple(articulation_state.shape)}"
                )
            root_pose = self._canonicalize_root_pose(
                torch.from_numpy(
                    np.asarray(articulation_state[start, :7], dtype=np.float32)
                )
            )
            action[slot] = (raw_action - self.stats["action_mean"]) / self.stats["action_std"]
            agent_state[slot] = (raw_state - self.stats["state_mean"]) / self.stats["state_std"]
            agent_geometry[slot] = root_pose
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
            "agent_geometry": agent_geometry,
            "agent_ids": agent_ids,
            "action_is_pad": torch.zeros(
                (num_agents, self.action_horizon), dtype=torch.bool
            ),
            "image_is_pad": torch.zeros((len(self.video_indices),), dtype=torch.bool),
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "task_name": task_name,
            "agent_count": num_agents,
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
    trajectories_by_agent_count: dict[int, int] = {}
    for path in h5_paths:
        with h5py.File(path, "r") as handle:
            for trajectory_name in sorted(handle.keys()):
                group = handle[trajectory_name]
                if "actions" not in group:
                    continue
                agent_names = sorted(group["actions"].keys(), key=_agent_sort_key)
                if not agent_names:
                    continue
                trajectory_count += 1
                agent_count = len(agent_names)
                trajectories_by_agent_count[agent_count] = (
                    trajectories_by_agent_count.get(agent_count, 0) + 1
                )
                for agent_name in agent_names:
                    update("action", group[f"actions/{agent_name}"][:])
                    qpos = group[f"obs/agent/{agent_name}/qpos"][:]
                    qvel = group[f"obs/agent/{agent_name}/qvel"][:]
                    update("state", np.concatenate([qpos, qvel], axis=-1))

    result: dict[str, Any] = {
        "source_root": str(root),
        "files": len(h5_paths),
        "trajectories": trajectory_count,
        "cardinality": {
            "agent_counts": sorted(trajectories_by_agent_count),
            "trajectories_by_agent_count": {
                str(count): trajectories_by_agent_count[count]
                for count in sorted(trajectories_by_agent_count)
            },
        },
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
