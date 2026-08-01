"""Canonical tensor schema for immutable per-agent Gaussian caches.

The covariance channels are deliberately explicit.  A legacy Policy-Lightning
implementation flattened ``[H, W, 3, 3]`` directly into ``[9, H, W]``; that
operation mixes spatial and matrix dimensions.  Canonical caches instead store
the covariance matrix in row-major ``(i, j)`` order after moving both matrix
axes in front of the spatial axes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import torch

SCHEMA_NAME = "fastwam.canonical-gaussian-cache"
SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
COMPLETE_FILENAME = "COMPLETE"

CANONICAL_CHANNELS = (
    "mean_x",
    "mean_y",
    "mean_z",
    "cov_xx",
    "cov_xy",
    "cov_xz",
    "cov_yx",
    "cov_yy",
    "cov_yz",
    "cov_zx",
    "cov_zy",
    "cov_zz",
    "opacity",
)
COVARIANCE_ORDER = "row-major-ij"
CANONICAL_DTYPE = "float16"
CANONICAL_CHANNEL_COUNT = len(CANONICAL_CHANNELS)

MIN_SHARD_BYTES = 1 << 30
MAX_SHARD_BYTES = 4 << 30
DEFAULT_TARGET_SHARD_BYTES = 2 << 30


def normalize_source_path(value: str) -> str:
    """Return a portable relative POSIX path, rejecting path traversal."""

    raw = str(value).replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"source_path must be a normalized relative path, got {value!r}")
    return path.as_posix()


@dataclass(frozen=True, order=True)
class FrameKey:
    """Stable lookup key for one agent observation frame."""

    source_path: str
    trajectory: str
    timestep: int
    agent_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", normalize_source_path(self.source_path))
        if not self.trajectory:
            raise ValueError("trajectory must be non-empty")
        if not self.agent_name:
            raise ValueError("agent_name must be non-empty")
        if isinstance(self.timestep, bool) or int(self.timestep) < 0:
            raise ValueError(f"timestep must be a non-negative integer, got {self.timestep!r}")
        object.__setattr__(self, "timestep", int(self.timestep))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FrameKey:
        return cls(
            source_path=str(value["source_path"]),
            trajectory=str(value["trajectory"]),
            timestep=int(value["timestep"]),
            agent_name=str(value["agent_name"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "trajectory": self.trajectory,
            "timestep": self.timestep,
            "agent_name": self.agent_name,
        }


@dataclass(frozen=True)
class GaussianCacheSchema:
    """On-disk tensor contract shared by canonical and compact caches."""

    height: int
    width: int
    cache_kind: str = "canonical"
    dtype: str = CANONICAL_DTYPE

    def __post_init__(self) -> None:
        if self.cache_kind not in {"canonical", "compact"}:
            raise ValueError(f"Unsupported cache_kind={self.cache_kind!r}")
        if self.dtype != CANONICAL_DTYPE:
            raise ValueError(f"Canonical cache dtype must be {CANONICAL_DTYPE}, got {self.dtype}")
        if int(self.height) <= 0 or int(self.width) <= 0:
            raise ValueError(f"height/width must be positive, got {(self.height, self.width)}")
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "width", int(self.width))

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return (CANONICAL_CHANNEL_COUNT, self.height, self.width)

    @property
    def frame_bytes(self) -> int:
        return CANONICAL_CHANNEL_COUNT * self.height * self.width * 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
            "cache_kind": self.cache_kind,
            "dtype": self.dtype,
            "byte_order": "little",
            "channels": list(CANONICAL_CHANNELS),
            "channel_count": CANONICAL_CHANNEL_COUNT,
            "height": self.height,
            "width": self.width,
            "covariance_order": COVARIANCE_ORDER,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GaussianCacheSchema:
        if value.get("name") != SCHEMA_NAME or value.get("version") != SCHEMA_VERSION:
            raise ValueError(
                "Unsupported Gaussian cache schema: "
                f"name={value.get('name')!r} version={value.get('version')!r}"
            )
        if tuple(value.get("channels", ())) != CANONICAL_CHANNELS:
            raise ValueError("Gaussian channel names/order do not match the canonical schema")
        if value.get("covariance_order") != COVARIANCE_ORDER:
            raise ValueError("Gaussian covariance order is not canonical row-major (i,j)")
        if value.get("byte_order") != "little":
            raise ValueError("Only little-endian canonical cache shards are supported")
        if int(value.get("channel_count", -1)) != CANONICAL_CHANNEL_COUNT:
            raise ValueError("Canonical Gaussian cache must contain exactly 13 channels")
        return cls(
            height=int(value["height"]),
            width=int(value["width"]),
            cache_kind=str(value["cache_kind"]),
            dtype=str(value["dtype"]),
        )

    def validate_tensor(self, tensor: torch.Tensor, *, name: str = "gaussian") -> None:
        if tensor.ndim < 3 or tuple(tensor.shape[-3:]) != self.frame_shape:
            raise ValueError(
                f"{name} must end in [13,{self.height},{self.width}], got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.float16:
            raise ValueError(f"{name} must be torch.float16, got {tensor.dtype}")


def unpack_gaussian_channels(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack ``[...,13,H,W]`` into means, ``[...,3,3,H,W]`` covariance, opacity."""

    if tensor.ndim < 3 or tensor.shape[-3] != CANONICAL_CHANNEL_COUNT:
        raise ValueError(f"Expected [...,13,H,W], got {tuple(tensor.shape)}")
    prefix = tuple(tensor.shape[:-3])
    height, width = tensor.shape[-2:]
    means = tensor[..., :3, :, :]
    covariance = tensor[..., 3:12, :, :].reshape(*prefix, 3, 3, height, width)
    opacity = tensor[..., 12:13, :, :]
    return means, covariance, opacity


def pack_gaussian_channels(
    means: torch.Tensor,
    covariance: torch.Tensor,
    opacity: torch.Tensor,
) -> torch.Tensor:
    """Pack canonical channels without mixing covariance and spatial dimensions."""

    if means.ndim < 3 or means.shape[-3] != 3:
        raise ValueError(f"means must be [...,3,H,W], got {tuple(means.shape)}")
    prefix = tuple(means.shape[:-3])
    height, width = means.shape[-2:]
    expected_covariance = (*prefix, 3, 3, height, width)
    expected_opacity = (*prefix, 1, height, width)
    if tuple(covariance.shape) != expected_covariance:
        raise ValueError(
            f"covariance must be {expected_covariance}, got {tuple(covariance.shape)}"
        )
    if tuple(opacity.shape) != expected_opacity:
        raise ValueError(f"opacity must be {expected_opacity}, got {tuple(opacity.shape)}")
    covariance_channels = covariance.reshape(*prefix, 9, height, width)
    return torch.cat((means, covariance_channels, opacity), dim=-3)


def correct_policy_lightning_legacy_covariance_order(tensor: torch.Tensor) -> torch.Tensor:
    """Repair the legacy ``reshape(B,V,9,H,W)`` covariance layout.

    Policy-Lightning commit ``c944b498...`` reshapes a contiguous
    ``[B,V,H,W,3,3]`` tensor directly to ``[B,V,9,H,W]``.  Reconstructing the
    original axes and then permuting them produces canonical row-major channels.
    Means and opacity were already laid out correctly and are preserved.
    """

    if tensor.ndim < 3 or tensor.shape[-3] != CANONICAL_CHANNEL_COUNT:
        raise ValueError(f"Expected [...,13,H,W], got {tuple(tensor.shape)}")
    prefix = tuple(tensor.shape[:-3])
    height, width = tensor.shape[-2:]
    legacy_covariance = tensor[..., 3:12, :, :]
    covariance_hw_ij = legacy_covariance.reshape(*prefix, height, width, 3, 3)
    prefix_ndim = len(prefix)
    permutation = (
        *range(prefix_ndim),
        prefix_ndim + 2,
        prefix_ndim + 3,
        prefix_ndim,
        prefix_ndim + 1,
    )
    covariance = covariance_hw_ij.permute(permutation).contiguous()
    return pack_gaussian_channels(tensor[..., :3, :, :], covariance, tensor[..., 12:13, :, :])
