"""RoboFactory deployment adapter for the native-agent FastWAM checkpoint.

This module deliberately keeps the simulator boundary explicit.  The caller
must pass both ``env.unwrapped.get_obs()`` and
``env.unwrapped.get_state_dict()`` to :meth:`FastWAMMultiRobotPolicy.update_obs`.
The observation does not contain articulation root poses, and silently
substituting zero geometry would change the model that is being evaluated.

The adapter reproduces the training inputs at commit
``00c0887118e647acf2ec7047dffa26a4231adc9e``:

* a native, variable-length agent axis (no fixed-capacity mask or padding),
* global RGB resized from uint8 240x320 to 224x320 with bicubic antialiasing,
* per-agent qpos+qvel z-score normalization and canonical root pose,
* pinned cached T5 context with the original padding convention,
* online ``[global, agent_i]`` NoPoSplat pairs followed by the exact compact
  opacity-aware moment matcher used to build the training cache, and
* action de-normalization from ``[N,H,8]`` to RoboFactory's ``[H,N*8]`` order.

Loading is fail-closed on checkpoint, normalization, text-context, external
teacher commit, and external teacher checkpoint identities.  No simulator or
formal evaluation is launched by this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from omegaconf import OmegaConf

try:
    from .normalization_stats_contract import (
        LEGACY_FULL_DATASET,
        NORMALIZATION_STATS_PROVENANCE_MODES,
        TRAIN_SPLIT,
        validate_normalization_stats_provenance,
    )
except ImportError:
    from normalization_stats_contract import (  # type: ignore[no-redef]
        LEGACY_FULL_DATASET,
        NORMALIZATION_STATS_PROVENANCE_MODES,
        TRAIN_SPLIT,
        validate_normalization_stats_provenance,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_CODE_COMMIT = "00c0887118e647acf2ec7047dffa26a4231adc9e"
TRAINING_STATS_SHA256 = (
    "92dfdeec62995b625b606d435ffb79ed787c4485348c16c42c3d31875eff64d0"
)
POLICY_LIGHTNING_COMMIT = "c944b4989a89c99c69d2572ea870f6a04680f5e7"
NOPOSPLAT_CHECKPOINT_SHA256 = (
    "4a35bc8c341b20859c0621f5238349b55b19a34a5bbeb3daec8d1f4c4603cd08"
)

# Copied verbatim from the training dataset at TRAINING_CODE_COMMIT.  Defining
# these small immutable strings locally keeps pure deployment transforms
# importable without pulling in the full LeRobot/datasets dependency graph.
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
DEFAULT_INSTRUCTIONS = {
    "StrikeCubeHard-rf": "two robots collaboratively strike the cube to the target",
    "PlaceFood-rf": "two robots collaboratively place the food in the target location",
    "PlaceCubeInCup-rf": "two robots collaboratively place the cube in the cup",
    "ThreeRobotsPlaceShoes-rf": "three robots collaboratively place the shoes in their target locations",
    "ThreeRobotsStackCube-rf": "three robots collaboratively stack the cubes",
    "FourRobotsStackCube-rf": "four robots collaboratively stack the cubes",
}

# Content identities of the six context tensors in the training input bundle.
TRAINING_CONTEXT_SHA256_BY_TASK = {
    "StrikeCubeHard-rf": "58270312488e57438d2c5c1c45ae0f7270bc25077df2b3856cfa4747c52c55c8",
    "PlaceFood-rf": "bc57a2edfa85c0ff8463cc81e54c0f4a88e05cebb86420684307126cc98aa9e7",
    "PlaceCubeInCup-rf": "3f939362fa88164b67aa0c9c95f3ab4d7160f638225eac4d277d6a96490f0635",
    "ThreeRobotsPlaceShoes-rf": "3733ba11bc54899c7b8183fb78e14ab41f1ec987e3003770de86ddf828e73a09",
    "ThreeRobotsStackCube-rf": "13d1cb4aa3f949b300a6907ee7fe7640ae5f2983f737a0c5aa1d85bbe53d65d2",
    "FourRobotsStackCube-rf": "917e1f649d0add0ff92c9996363605bc410e8dcabaf282b6db2be187ac835945",
}

_AGENT_NAME = re.compile(r"^panda-(\d+)$")
_CONTEXT_LENGTH = 128
_TEXT_DIMENSION = 4096
_ACTION_DIMENSION = 8
_STATE_DIMENSION = 18
_NATIVE_RGB_SIZE = (240, 320)
_MODEL_RGB_SIZE = (224, 320)
_COMPACT_GAUSSIAN_SHAPE = (13, 28, 40)


class GaussianTeacher(Protocol):
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Map ``[N,2,3,240,320]`` RGB in [-1,1] to corrected Gaussians."""

    def provenance(self) -> Mapping[str, Any]:
        """Return immutable teacher provenance."""


@dataclass(frozen=True)
class NormalizationStats:
    action_mean: torch.Tensor
    action_std: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor
    path: Path | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    provenance_mode: str = TRAIN_SPLIT


@dataclass(frozen=True)
class TextContext:
    task_name: str
    prompt: str
    context: torch.Tensor
    mask: torch.Tensor
    path: Path
    sha256: str | None
    size_bytes: int


@dataclass(frozen=True)
class PreparedObservation:
    """CPU snapshot of one causal simulator observation."""

    agent_names: tuple[str, ...]
    global_rgb: torch.Tensor
    agent_rgb: torch.Tensor
    agent_states: torch.Tensor
    agent_geometry: torch.Tensor
    agent_ids: torch.Tensor


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_sha256(value: str, *, field: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256")
    return normalized


def require_file_sha256(
    path: str | Path, expected_sha256: str, *, label: str
) -> tuple[Path, str]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {resolved}")
    expected = _normalized_sha256(expected_sha256, field=f"{label} SHA-256")
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected} actual={actual} path={resolved}"
        )
    return resolved, actual


def require_regular_file_metadata(
    path: str | Path,
    expected_size_bytes: int,
    *,
    label: str,
) -> tuple[Path, int]:
    """Bind a file by canonical path/type/identity/size without content hashing."""

    requested = Path(path).expanduser()
    metadata = requested.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FileNotFoundError(f"{label} is not a direct regular file: {requested}")
    if metadata.st_nlink != 1:
        raise ValueError(
            f"{label} must have exactly one hard link: path={requested} "
            f"nlink={metadata.st_nlink}"
        )
    expected = int(expected_size_bytes)
    if expected < 1:
        raise ValueError(f"{label} expected size must be positive, got {expected}")
    if metadata.st_size != expected:
        raise ValueError(
            f"{label} size mismatch: expected={expected} actual={metadata.st_size} "
            f"path={requested}"
        )
    resolved = requested.resolve(strict=True)
    current = resolved.stat()
    if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise ValueError(f"{label} identity changed during validation: {requested}")
    return resolved, int(current.st_size)


def _validate_integrity_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"sha256", "metadata_no_hash"}:
        raise ValueError(
            "integrity_mode must be either 'sha256' or 'metadata_no_hash', "
            f"got {mode!r}"
        )
    return normalized


def canonical_task_name(task_name: str) -> str:
    name = str(task_name).strip()
    if name in DEFAULT_INSTRUCTIONS:
        return name
    candidate = f"{name}-rf"
    if candidate in DEFAULT_INSTRUCTIONS:
        return candidate
    raise KeyError(
        f"Unsupported RoboFactory task {task_name!r}; expected one of "
        f"{sorted(DEFAULT_INSTRUCTIONS)}"
    )


def ordered_agent_names(observation: Mapping[str, Any]) -> tuple[str, ...]:
    agents = observation.get("agent")
    if not isinstance(agents, Mapping) or not agents:
        raise KeyError(
            "RoboFactory observation must contain a non-empty 'agent' mapping"
        )
    indexed: list[tuple[int, str]] = []
    for name in agents:
        match = _AGENT_NAME.fullmatch(str(name))
        if match is None:
            raise ValueError(
                f"Unexpected RoboFactory agent name {name!r}; expected panda-<id>"
            )
        indexed.append((int(match.group(1)), str(name)))
    indexed.sort()
    ids = [index for index, _ in indexed]
    if ids != list(range(len(ids))):
        raise ValueError(
            f"RoboFactory agent ids must be contiguous from zero, got {ids}"
        )
    return tuple(name for _, name in indexed)


def _as_cpu_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(np.asarray(value))


def _flat_vector(
    value: Any, *, label: str, expected_dim: int | None = None
) -> torch.Tensor:
    tensor = _as_cpu_tensor(value)
    while tensor.ndim > 1 and tensor.shape[0] == 1:
        tensor = tensor[0]
    tensor = tensor.reshape(-1).to(torch.float32)
    if expected_dim is not None and tensor.shape != (expected_dim,):
        raise ValueError(
            f"{label} must contain exactly {expected_dim} values, got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{label} contains non-finite values")
    return tensor.contiguous()


def camera_rgb_uint8(observation: Mapping[str, Any], camera_name: str) -> torch.Tensor:
    try:
        value = observation["sensor_data"][camera_name]["rgb"]
    except (KeyError, TypeError) as error:
        raise KeyError(
            f"Missing RoboFactory camera RGB sensor_data/{camera_name}/rgb"
        ) from error
    tensor = _as_cpu_tensor(value)
    while tensor.ndim > 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(
            f"{camera_name}.rgb must be 3D after singleton-batch removal, got {tuple(tensor.shape)}"
        )
    height, width = _NATIVE_RGB_SIZE
    if tuple(tensor.shape[:2]) == (height, width) and tensor.shape[-1] >= 3:
        tensor = tensor[..., :3].permute(2, 0, 1)
    elif tensor.shape[0] >= 3 and tuple(tensor.shape[1:]) == (height, width):
        tensor = tensor[:3]
    else:
        raise ValueError(
            f"{camera_name}.rgb must be HWC/CHW at 240x320, got {tuple(tensor.shape)}"
        )

    if tensor.dtype == torch.uint8:
        return tensor.clone().contiguous()
    values = tensor.to(torch.float32)
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError(f"{camera_name}.rgb contains non-finite values")
    minimum = float(values.min().item())
    maximum = float(values.max().item())
    if minimum >= -1e-6 and maximum <= 1.0 + 1e-6:
        values = values * 255.0
    elif minimum < -1e-6 or maximum > 255.0 + 1e-6:
        raise ValueError(
            f"{camera_name}.rgb must be uint8-like [0,255] or float [0,1], "
            f"got range [{minimum}, {maximum}]"
        )
    return values.round().clamp_(0.0, 255.0).to(torch.uint8).contiguous()


def canonicalize_root_pose(root_pose: Any) -> torch.Tensor:
    pose = _flat_vector(root_pose, label="articulation root pose")
    if pose.numel() < 7:
        raise ValueError(
            f"articulation state must contain at least 7 values, got {pose.numel()}"
        )
    pose = pose[:7].clone()
    quaternion = pose[3:7]
    norm = torch.linalg.vector_norm(quaternion)
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) < 1e-8:
        raise ValueError(f"Invalid root-pose quaternion: {quaternion.tolist()}")
    quaternion = quaternion / norm
    pivot = int(torch.argmax(quaternion.abs()).item())
    if float(quaternion[pivot].item()) < 0.0:
        quaternion = -quaternion
    pose[3:7] = quaternion
    return pose.contiguous()


def _articulation_name(agent_name: str) -> str:
    match = _AGENT_NAME.fullmatch(agent_name)
    if match is None:
        raise ValueError(f"Invalid RoboFactory agent name: {agent_name!r}")
    return f"panda-agent-{int(match.group(1))}"


def load_normalization_stats(
    path: str | Path,
    *,
    expected_sha256: str | None = TRAINING_STATS_SHA256,
    expected_size_bytes: int | None = None,
    integrity_mode: str = "sha256",
    provenance_mode: str = TRAIN_SPLIT,
) -> NormalizationStats:
    mode = _validate_integrity_mode(integrity_mode)
    if mode == "metadata_no_hash":
        if expected_size_bytes is None:
            raise ValueError(
                "RoboFactory normalization stats size is required in metadata_no_hash mode"
            )
        resolved, actual_size_bytes = require_regular_file_metadata(
            path,
            expected_size_bytes,
            label="RoboFactory training normalization stats",
        )
        actual_sha256 = None
    else:
        if expected_sha256 is None:
            raise ValueError("RoboFactory normalization stats SHA-256 is required")
        resolved, actual_sha256 = require_file_sha256(
            path,
            expected_sha256,
            label="RoboFactory training normalization stats",
        )
        actual_size_bytes = resolved.stat().st_size
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Normalization stats must be a JSON object: {resolved}")
    normalized_provenance_mode = validate_normalization_stats_provenance(
        payload, provenance_mode
    )

    tensors: dict[str, torch.Tensor] = {}
    for kind, dimension in (("action", _ACTION_DIMENSION), ("state", _STATE_DIMENSION)):
        record = payload.get(kind)
        if not isinstance(record, Mapping):
            raise KeyError(f"Normalization stats are missing {kind!r}")
        mean = _flat_vector(
            record.get("mean"), label=f"{kind}.mean", expected_dim=dimension
        )
        std = _flat_vector(
            record.get("std"), label=f"{kind}.std", expected_dim=dimension
        )
        tensors[f"{kind}_mean"] = mean
        tensors[f"{kind}_std"] = std.clamp_min(1e-6)
    return NormalizationStats(
        action_mean=tensors["action_mean"],
        action_std=tensors["action_std"],
        state_mean=tensors["state_mean"],
        state_std=tensors["state_std"],
        path=resolved,
        sha256=actual_sha256,
        size_bytes=actual_size_bytes,
        provenance_mode=normalized_provenance_mode,
    )


def load_text_context(
    cache_dir: str | Path,
    task_name: str,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    integrity_mode: str = "sha256",
) -> TextContext:
    canonical_name = canonical_task_name(task_name)
    instruction = DEFAULT_INSTRUCTIONS[canonical_name]
    prompt = DEFAULT_PROMPT.format(task=instruction)
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = Path(cache_dir).expanduser().resolve() / (
        f"{prompt_sha256}.t5_len{_CONTEXT_LENGTH}.wan22ti2v5b.pt"
    )
    mode = _validate_integrity_mode(integrity_mode)
    if mode == "metadata_no_hash":
        if expected_size_bytes is None:
            raise ValueError(
                f"cached T5 context size is required for {canonical_name} "
                "in metadata_no_hash mode"
            )
        resolved, actual_size_bytes = require_regular_file_metadata(
            cache_path,
            expected_size_bytes,
            label=f"cached T5 context for {canonical_name}",
        )
        actual_sha256 = None
    else:
        pinned_sha256 = (
            TRAINING_CONTEXT_SHA256_BY_TASK[canonical_name]
            if expected_sha256 is None
            else expected_sha256
        )
        resolved, actual_sha256 = require_file_sha256(
            cache_path,
            pinned_sha256,
            label=f"cached T5 context for {canonical_name}",
        )
        actual_size_bytes = resolved.stat().st_size
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or "context" not in payload
        or "mask" not in payload
    ):
        raise ValueError(f"Invalid cached text payload: {resolved}")
    context = _as_cpu_tensor(payload["context"]).clone()
    mask = _as_cpu_tensor(payload["mask"]).bool().clone()
    if context.shape != (_CONTEXT_LENGTH, _TEXT_DIMENSION):
        raise ValueError(
            f"Cached context must be [{_CONTEXT_LENGTH},{_TEXT_DIMENSION}], got {tuple(context.shape)}"
        )
    if mask.shape != (_CONTEXT_LENGTH,):
        raise ValueError(
            f"Cached context mask must be [{_CONTEXT_LENGTH}], got {tuple(mask.shape)}"
        )
    if not torch.is_floating_point(context) or not bool(
        torch.isfinite(context).all().item()
    ):
        raise ValueError("Cached context must be a finite floating-point tensor")
    context[~mask] = 0
    # Match RoboFactoryMultiRobotDataset._load_cached_text_context exactly.
    mask = torch.ones_like(mask)
    return TextContext(
        task_name=canonical_name,
        prompt=prompt,
        context=context.contiguous(),
        mask=mask.contiguous(),
        path=resolved,
        sha256=actual_sha256,
        size_bytes=actual_size_bytes,
    )


def extract_agent_state_and_geometry(
    observation: Mapping[str, Any],
    env_state: Mapping[str, Any],
    agent_names: Sequence[str],
    stats: NormalizationStats,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    articulations = env_state.get("articulations")
    if not isinstance(articulations, Mapping):
        raise KeyError("RoboFactory state_dict must contain an 'articulations' mapping")
    raw_states: list[torch.Tensor] = []
    root_poses: list[torch.Tensor] = []
    agent_ids: list[int] = []
    agents = observation.get("agent")
    if not isinstance(agents, Mapping):
        raise KeyError("RoboFactory observation must contain an 'agent' mapping")
    for agent_name in agent_names:
        record = agents.get(agent_name)
        if not isinstance(record, Mapping):
            raise KeyError(f"Missing proprioception for {agent_name}")
        qpos = _flat_vector(
            record.get("qpos"), label=f"{agent_name}.qpos", expected_dim=9
        )
        qvel = _flat_vector(
            record.get("qvel"), label=f"{agent_name}.qvel", expected_dim=9
        )
        raw_states.append(torch.cat((qpos, qvel), dim=0))
        articulation_name = _articulation_name(agent_name)
        if articulation_name not in articulations:
            raise KeyError(f"Missing state_dict articulations/{articulation_name}")
        root_poses.append(canonicalize_root_pose(articulations[articulation_name]))
        agent_ids.append(int(_AGENT_NAME.fullmatch(agent_name).group(1)))  # type: ignore[union-attr]

    raw_state = torch.stack(raw_states, dim=0)
    normalized_state = (raw_state - stats.state_mean) / stats.state_std
    geometry = torch.stack(root_poses, dim=0)
    ids = torch.tensor(agent_ids, dtype=torch.long)
    if not bool(torch.isfinite(normalized_state).all().item()):
        raise ValueError("Normalized agent state contains non-finite values")
    return normalized_state.contiguous(), geometry.contiguous(), ids


def prepare_observation(
    observation: Mapping[str, Any],
    env_state: Mapping[str, Any],
    stats: NormalizationStats,
    *,
    allowed_agent_counts: Sequence[int] = (2, 3, 4),
) -> PreparedObservation:
    agent_names = ordered_agent_names(observation)
    allowed = {int(count) for count in allowed_agent_counts}
    if len(agent_names) not in allowed:
        raise ValueError(
            f"Checkpoint was trained for native N={sorted(allowed)}, env has N={len(agent_names)}"
        )
    global_rgb = camera_rgb_uint8(observation, "head_camera_global")
    agent_rgb = torch.stack(
        [
            camera_rgb_uint8(observation, f"head_camera_agent{agent_id}")
            for agent_id in range(len(agent_names))
        ],
        dim=0,
    )
    states, geometry, ids = extract_agent_state_and_geometry(
        observation,
        env_state,
        agent_names,
        stats,
    )
    return PreparedObservation(
        agent_names=agent_names,
        global_rgb=global_rgb,
        agent_rgb=agent_rgb,
        agent_states=states,
        agent_geometry=geometry,
        agent_ids=ids,
    )


def model_input_image(prepared: PreparedObservation) -> torch.Tensor:
    resized = transforms_F.resize(
        prepared.global_rgb.unsqueeze(0),
        size=list(_MODEL_RGB_SIZE),
        interpolation=transforms_F.InterpolationMode.BICUBIC,
        antialias=True,
    )
    return resized.float().div(127.5).sub(1.0).contiguous()


def teacher_image_pairs(prepared: PreparedObservation) -> torch.Tensor:
    num_agents = len(prepared.agent_names)
    global_views = prepared.global_rgb.unsqueeze(0).expand(num_agents, -1, -1, -1)
    pairs = torch.stack((global_views, prepared.agent_rgb), dim=1)
    return pairs.float().div(127.5).sub(1.0).contiguous()


def encode_compact_agent_gaussian(
    teacher: GaussianTeacher,
    prepared: PreparedObservation,
) -> torch.Tensor:
    from fastwam.datasets.gaussian_cache.compact import opacity_aware_moment_match

    pairs = teacher_image_pairs(prepared)
    pair_gaussian = teacher.encode(pairs)
    expected_pair_shape = (len(prepared.agent_names), 2, 13, *_NATIVE_RGB_SIZE)
    if (
        tuple(pair_gaussian.shape) != expected_pair_shape
        or pair_gaussian.dtype != torch.float16
    ):
        raise ValueError(
            f"Teacher output must be FP16 {expected_pair_shape}, got "
            f"shape={tuple(pair_gaussian.shape)} dtype={pair_gaussian.dtype}"
        )
    # The extraction pipeline stores only the agent element of [global, agent_i].
    compact = opacity_aware_moment_match(pair_gaussian[:, 1])
    expected_compact_shape = (len(prepared.agent_names), *_COMPACT_GAUSSIAN_SHAPE)
    if tuple(compact.shape) != expected_compact_shape or compact.dtype != torch.float16:
        raise ValueError(
            f"Compact Gaussian must be FP16 {expected_compact_shape}, got "
            f"shape={tuple(compact.shape)} dtype={compact.dtype}"
        )
    if not bool(torch.isfinite(compact).all().item()):
        raise ValueError("Compact Gaussian contains non-finite values")
    return compact.contiguous()


def denormalize_actions(
    normalized_action: torch.Tensor,
    stats: NormalizationStats,
) -> torch.Tensor:
    """Apply the training action statistics without implicit clipping."""

    action = _as_cpu_tensor(normalized_action).to(torch.float32)
    if action.ndim != 3 or action.shape[2] != _ACTION_DIMENSION:
        raise ValueError(f"FastWAM action must be [N,H,8], got {tuple(action.shape)}")
    denormalized = action * stats.action_std + stats.action_mean
    if not bool(torch.isfinite(denormalized).all().item()):
        raise FloatingPointError("FastWAM produced non-finite de-normalized actions")
    return denormalized.contiguous()


def action_scaling_contract(stats: NormalizationStats) -> dict[str, Any]:
    """Return the exact normalization contract used by policy inference."""

    return {
        "formula": "normalized_action * action_std + action_mean",
        "clipping_applied": False,
        "action_mean": stats.action_mean.detach().cpu().tolist(),
        "action_std": stats.action_std.detach().cpu().tolist(),
    }


def denormalize_and_flatten_actions(
    normalized_action: torch.Tensor,
    stats: NormalizationStats,
) -> np.ndarray:
    denormalized = denormalize_actions(normalized_action, stats)
    # RoboFactory consumes one flat [agent0(8), agent1(8), ...] vector per step.
    flattened = denormalized.permute(1, 0, 2).reshape(denormalized.shape[1], -1)
    return np.ascontiguousarray(flattened.numpy(), dtype=np.float32)


def compose_step5000_model_config(
    project_root: str | Path = PROJECT_ROOT,
    *,
    enable_gaussian: bool = True,
):
    """Resolve the exact VG1/Hub1 architecture without dataset instantiation."""

    root = Path(project_root).expanduser().resolve()
    data = OmegaConf.load(root / "configs/data/robofactory_multi_robot.yaml")
    model = OmegaConf.load(root / "configs/model/fastwam_multi_robot.yaml")
    config = OmegaConf.create({"data": data, "model": model})
    config.model.load_text_encoder = False
    config.model.skip_dit_load_from_pretrain = True
    config.model.action_dit_pretrained_path = None
    config.model.training_mode = "joint"
    config.model.action_dit_config.hub_enabled = True
    config.model.action_dit_config.enable_gaussian = bool(enable_gaussian)
    config.model.loss.lambda_video = 1.0
    config.model.loss.lambda_action = 1.0
    resolved = OmegaConf.to_container(config.model, resolve=True)
    return OmegaConf.create(resolved)


@contextmanager
def _model_asset_environment(model_cache_root: str | Path):
    root = Path(model_cache_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise FileNotFoundError(f"FastWAM model cache is not a directory: {root}")
    updates = {
        "DIFFSYNTH_MODEL_BASE_PATH": str(root),
        "DIFFSYNTH_SKIP_DOWNLOAD": "true",
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_fastwam_checkpoint(
    model: Any,
    checkpoint_path: str | Path,
    *,
    integrity_mode: str = "sha256",
) -> None:
    """Load through the current API with the matching model provenance mode."""

    mode = _validate_integrity_mode(integrity_mode)
    model._checkpoint_provenance_mode = (
        "stat_cmp" if mode == "metadata_no_hash" else "sha256"
    )
    model.load_checkpoint(checkpoint_path)


class FastWAMMultiRobotPolicy:
    """Native-N RoboFactory policy backed by ``fastwam_multi_robot_v2`` weights."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        checkpoint_sha256: str | None,
        stats_path: str | Path,
        context_cache_dir: str | Path,
        task_name: str,
        model_cache_root: str | Path,
        policy_lightning_repo: str | Path | None,
        noposplat_checkpoint_path: str | Path | None,
        device: str | torch.device = "cuda:0",
        teacher_device: str | torch.device | None = None,
        model_dtype: torch.dtype = torch.bfloat16,
        action_horizon: int = 32,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        expected_stats_sha256: str | None = TRAINING_STATS_SHA256,
        expected_context_sha256: str | None = None,
        integrity_mode: str = "sha256",
        checkpoint_size_bytes: int | None = None,
        stats_size_bytes: int | None = None,
        stats_provenance_mode: str = TRAIN_SPLIT,
        context_size_bytes: int | None = None,
        gaussian_conditioning: bool = True,
        training_source_commit: str | None = None,
        training_job_id: str | None = None,
        policy_lightning_commit: str = POLICY_LIGHTNING_COMMIT,
        noposplat_checkpoint_sha256: str | None = NOPOSPLAT_CHECKPOINT_SHA256,
        noposplat_checkpoint_size_bytes: int | None = None,
        policy_lightning_config_path: str | Path = "config/encoder/noposplat.yaml",
        allowed_agent_counts: Sequence[int] = (2, 3, 4),
        project_root: str | Path = PROJECT_ROOT,
    ) -> None:
        self.integrity_mode = _validate_integrity_mode(integrity_mode)
        if self.integrity_mode == "metadata_no_hash":
            if checkpoint_size_bytes is None:
                raise ValueError(
                    "FastWAM checkpoint size is required in metadata_no_hash mode"
                )
            self.checkpoint_path, self.checkpoint_size_bytes = (
                require_regular_file_metadata(
                    checkpoint_path,
                    checkpoint_size_bytes,
                    label="FastWAM checkpoint",
                )
            )
            self.expected_checkpoint_sha256 = None
        else:
            if checkpoint_sha256 is None:
                raise ValueError("FastWAM checkpoint SHA-256 is required")
            self.checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"FastWAM checkpoint is not a regular file: {self.checkpoint_path}"
                )
            self.expected_checkpoint_sha256 = _normalized_sha256(
                checkpoint_sha256,
                field="FastWAM checkpoint SHA-256",
            )
            self.checkpoint_size_bytes = self.checkpoint_path.stat().st_size
        self.stats = load_normalization_stats(
            stats_path,
            expected_sha256=expected_stats_sha256,
            expected_size_bytes=stats_size_bytes,
            integrity_mode=self.integrity_mode,
            provenance_mode=stats_provenance_mode,
        )
        self.text_context = load_text_context(
            context_cache_dir,
            task_name,
            expected_sha256=expected_context_sha256,
            expected_size_bytes=context_size_bytes,
            integrity_mode=self.integrity_mode,
        )
        self.gaussian_conditioning = bool(gaussian_conditioning)
        self.training_source_commit = (
            None if training_source_commit is None else str(training_source_commit)
        )
        self.training_job_id = None if training_job_id is None else str(training_job_id)
        self.allowed_agent_counts = tuple(
            sorted({int(count) for count in allowed_agent_counts})
        )
        if not self.allowed_agent_counts or any(
            count < 1 for count in self.allowed_agent_counts
        ):
            raise ValueError("allowed_agent_counts must contain positive integers")
        if self.allowed_agent_counts != (2, 3, 4):
            raise ValueError(
                "This checkpoint's declared cardinality scope is exactly (2, 3, 4); "
                f"got {self.allowed_agent_counts}"
            )
        self.device = torch.device(device)
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        self.sigma_shift = None if sigma_shift is None else float(sigma_shift)
        self.seed = None if seed is None else int(seed)
        self._episode_seed = self.seed
        self._query_index = 0
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)

        from hydra.utils import instantiate

        model_config = compose_step5000_model_config(
            project_root,
            enable_gaussian=self.gaussian_conditioning,
        )
        with _model_asset_environment(model_cache_root):
            self.model = instantiate(
                model_config,
                model_dtype=model_dtype,
                device=str(self.device),
            )
        load_fastwam_checkpoint(
            self.model,
            self.checkpoint_path,
            integrity_mode=self.integrity_mode,
        )
        actual_checkpoint_sha256 = getattr(
            self.model,
            "_loaded_base_checkpoint_sha256",
            None,
        )
        if self.integrity_mode == "sha256":
            if actual_checkpoint_sha256 is None:
                actual_checkpoint_sha256 = sha256_file(self.checkpoint_path)
            if actual_checkpoint_sha256 != self.expected_checkpoint_sha256:
                raise ValueError(
                    "FastWAM checkpoint SHA-256 mismatch after strict load: "
                    f"expected={self.expected_checkpoint_sha256} "
                    f"actual={actual_checkpoint_sha256} path={self.checkpoint_path}"
                )
            self.checkpoint_sha256 = actual_checkpoint_sha256
        else:
            if actual_checkpoint_sha256 is not None:
                raise ValueError(
                    "metadata_no_hash checkpoint load unexpectedly computed a SHA-256"
                )
            self.checkpoint_sha256 = None
        self.model.eval()

        self.teacher: GaussianTeacher | None
        if self.gaussian_conditioning:
            if policy_lightning_repo is None or noposplat_checkpoint_path is None:
                raise ValueError(
                    "Gaussian conditioning requires Policy-Lightning and NoPoSplat paths"
                )
            from fastwam.datasets.gaussian_cache.teacher import (
                ExternalPolicyLightningTeacher,
            )

            self.teacher = ExternalPolicyLightningTeacher(
                repo_path=policy_lightning_repo,
                expected_commit=policy_lightning_commit,
                checkpoint_path=noposplat_checkpoint_path,
                checkpoint_sha256=noposplat_checkpoint_sha256,
                checkpoint_size_bytes=noposplat_checkpoint_size_bytes,
                integrity_mode=self.integrity_mode,
                config_path=policy_lightning_config_path,
                device=self.device if teacher_device is None else teacher_device,
                require_clean_repo=True,
            )
        else:
            self.teacher = None
        self._prepared: PreparedObservation | None = None

    def reset(self) -> None:
        self._prepared = None
        self._episode_seed = self.seed
        self._query_index = 0

    def start_episode(self, policy_seed: int | None = None) -> None:
        """Reset causal state and bind an auditable per-query diffusion seed schedule."""

        self._prepared = None
        self._episode_seed = self.seed if policy_seed is None else int(policy_seed)
        self._query_index = 0

    def update_obs(
        self,
        observation: Mapping[str, Any],
        env_state: Mapping[str, Any] | None = None,
    ) -> None:
        if env_state is None:
            embedded = observation.get("__fastwam_env_state__")
            if isinstance(embedded, Mapping):
                env_state = embedded
        if env_state is None:
            raise ValueError(
                "FastWAM needs the current articulation root poses. Call "
                "policy.update_obs(env.unwrapped.get_obs(), "
                "env.unwrapped.get_state_dict()); observation-only deployment is invalid."
            )
        self._prepared = prepare_observation(
            observation,
            env_state,
            self.stats,
            allowed_agent_counts=self.allowed_agent_counts,
        )

    def record_action(self, action: np.ndarray | torch.Tensor) -> None:
        """Validate evaluator plumbing; FastWAM state comes from qpos+qvel, not prior action."""

        if self._prepared is None:
            raise RuntimeError("No observation is recorded; call update_obs first")
        flat = np.asarray(action, dtype=np.float32).reshape(-1)
        expected = len(self._prepared.agent_names) * _ACTION_DIMENSION
        if flat.shape != (expected,):
            raise ValueError(
                f"Executed action must have shape ({expected},), got {flat.shape}"
            )
        if not np.isfinite(flat).all():
            raise ValueError("Executed action contains non-finite values")

    @torch.inference_mode()
    def get_action(self) -> np.ndarray:
        return np.ascontiguousarray(self.get_action_trace()["flat_action"])

    @torch.inference_mode()
    def get_action_trace(self) -> dict[str, Any]:
        """Return normalized and de-normalized actions for rollout diagnostics."""

        if self._prepared is None:
            raise RuntimeError("No observation is recorded; call update_obs first")
        agent_gaussian = (
            None
            if self.teacher is None
            else encode_compact_agent_gaussian(self.teacher, self._prepared)
        )
        inference_seed = (
            None
            if self._episode_seed is None
            else int(self._episode_seed) + int(self._query_index)
        )
        prediction = self.model.infer_action_multi(
            input_image=model_input_image(self._prepared),
            action_horizon=self.action_horizon,
            agent_states=self._prepared.agent_states,
            agent_geometry=self._prepared.agent_geometry,
            agent_ids=self._prepared.agent_ids,
            agent_gaussian=agent_gaussian,
            context=self.text_context.context,
            context_mask=self.text_context.mask,
            num_inference_steps=self.num_inference_steps,
            sigma_shift=self.sigma_shift,
            seed=inference_seed,
            rand_device=self.rand_device,
            tiled=self.tiled,
        )
        if not isinstance(prediction, Mapping) or "action" not in prediction:
            raise ValueError(
                "infer_action_multi must return a mapping containing 'action'"
            )
        normalized_action = prediction["action"]
        expected_shape = (
            len(self._prepared.agent_names),
            self.action_horizon,
            _ACTION_DIMENSION,
        )
        if tuple(normalized_action.shape) != expected_shape:
            raise ValueError(
                f"infer_action_multi action must be {expected_shape}, got {tuple(normalized_action.shape)}"
            )
        normalized_cpu = _as_cpu_tensor(normalized_action).to(torch.float32).contiguous()
        denormalized_cpu = denormalize_actions(normalized_cpu, self.stats)
        flattened = denormalized_cpu.permute(1, 0, 2).reshape(
            denormalized_cpu.shape[1], -1
        )
        flat_action = np.ascontiguousarray(flattened.numpy(), dtype=np.float32)
        query_index = int(self._query_index)
        self._query_index += 1
        return {
            "query_index": query_index,
            "inference_seed": inference_seed,
            "agent_names": list(self._prepared.agent_names),
            "normalized_action": normalized_cpu.numpy(),
            "denormalized_action": denormalized_cpu.numpy(),
            "flat_action": flat_action,
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "adapter_training_code_commit": TRAINING_CODE_COMMIT,
            "integrity_mode": self.integrity_mode,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "training_source_commit": self.training_source_commit,
            "training_job_id": self.training_job_id,
            "normalization_path": str(self.stats.path),
            "normalization_sha256": self.stats.sha256,
            "normalization_size_bytes": self.stats.size_bytes,
            "normalization_provenance_mode": self.stats.provenance_mode,
            "context_path": str(self.text_context.path),
            "context_sha256": self.text_context.sha256,
            "context_size_bytes": self.text_context.size_bytes,
            "task_name": self.text_context.task_name,
            "allowed_agent_counts": list(self.allowed_agent_counts),
            "action_scaling": action_scaling_contract(self.stats),
            "action_horizon": self.action_horizon,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "seed_schedule": "episode_seed_plus_query_index_v1",
            "gaussian_conditioning": self.gaussian_conditioning,
            "teacher": (
                None if self.teacher is None else dict(self.teacher.provenance())
            ),
            "gaussian_pairing": (
                "global_agent_unify_v1" if self.gaussian_conditioning else None
            ),
            "gaussian_compaction": (
                "opacity-aware-moment-matching-cell-mean-alpha-v2"
                if self.gaussian_conditioning
                else None
            ),
        }


__all__ = [
    "FastWAMMultiRobotPolicy",
    "NormalizationStats",
    "PreparedObservation",
    "TextContext",
    "TRAINING_CODE_COMMIT",
    "TRAINING_CONTEXT_SHA256_BY_TASK",
    "TRAINING_STATS_SHA256",
    "TRAIN_SPLIT",
    "LEGACY_FULL_DATASET",
    "NORMALIZATION_STATS_PROVENANCE_MODES",
    "canonical_task_name",
    "canonicalize_root_pose",
    "camera_rgb_uint8",
    "action_scaling_contract",
    "compose_step5000_model_config",
    "denormalize_and_flatten_actions",
    "denormalize_actions",
    "encode_compact_agent_gaussian",
    "extract_agent_state_and_geometry",
    "load_normalization_stats",
    "validate_normalization_stats_provenance",
    "load_text_context",
    "model_input_image",
    "ordered_agent_names",
    "prepare_observation",
    "require_file_sha256",
    "require_regular_file_metadata",
    "sha256_file",
    "teacher_image_pairs",
]
