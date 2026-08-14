"""Observation-only metric geometry features from calibrated depth cameras."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


METRIC_GEOMETRY_CHANNELS = 13
METRIC_GEOMETRY_SIZE = (60, 80)
METRIC_GEOMETRY_MAX_DEPTH_M = 3.0


def _as_float_tensor(value: Any, *, label: str) -> torch.Tensor:
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
    tensor = tensor.to(torch.float32)
    while tensor.ndim > 0 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{label} contains non-finite values")
    return tensor


def _camera_matrix(value: Any, *, label: str, shape: tuple[int, int]) -> torch.Tensor:
    tensor = _as_float_tensor(value, label=label)
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{label} must have shape {shape}, got {tuple(tensor.shape)}")
    return tensor


def encode_metric_geometry(
    depth: Any,
    intrinsic_cv: Any,
    extrinsic_cv: Any,
    *,
    output_size: tuple[int, int] = METRIC_GEOMETRY_SIZE,
    depth_scale: float = 0.001,
    min_depth: float = 0.02,
    max_depth: float = METRIC_GEOMETRY_MAX_DEPTH_M,
    surface_band: float = 0.03,
    covariance_floor: float = 1.0e-6,
) -> torch.Tensor:
    """Encode one calibrated depth image as a metric world-frame Gaussian grid.

    ``extrinsic_cv`` follows ManiSkill's world-to-OpenCV-camera convention.
    Each output cell retains the nearest connected depth surface in its source
    block, which prevents a small foreground object from being averaged into a
    farther table or background surface. Channels are world XYZ mean, row-major
    3x3 covariance, and a binary valid flag.
    """

    depth_scale = float(depth_scale)
    min_depth = float(min_depth)
    max_depth = float(max_depth)
    surface_band = float(surface_band)
    covariance_floor = float(covariance_floor)
    if not (depth_scale > 0.0 and 0.0 <= min_depth < max_depth):
        raise ValueError("depth_scale/depth range is invalid")
    if surface_band < 0.0 or covariance_floor < 0.0:
        raise ValueError("surface_band and covariance_floor must be non-negative")

    depth_tensor = _as_float_tensor(depth, label="depth")
    if depth_tensor.ndim == 3 and depth_tensor.shape[-1] == 1:
        depth_tensor = depth_tensor[..., 0]
    if depth_tensor.ndim != 2:
        raise ValueError(f"depth must have shape [H,W] or [H,W,1], got {tuple(depth_tensor.shape)}")
    input_height, input_width = map(int, depth_tensor.shape)
    output_height, output_width = map(int, output_size)
    if output_height < 1 or output_width < 1:
        raise ValueError(f"output_size must be positive, got {output_size}")
    if input_height % output_height or input_width % output_width:
        raise ValueError(
            "input depth dimensions must be exact multiples of output_size, "
            f"got input={(input_height, input_width)} output={output_size}"
        )

    intrinsic = _camera_matrix(intrinsic_cv, label="intrinsic_cv", shape=(3, 3))
    extrinsic_tensor = _as_float_tensor(extrinsic_cv, label="extrinsic_cv")
    if tuple(extrinsic_tensor.shape) == (4, 4):
        extrinsic = extrinsic_tensor[:3]
    elif tuple(extrinsic_tensor.shape) == (3, 4):
        extrinsic = extrinsic_tensor
    else:
        raise ValueError(
            "extrinsic_cv must have shape (3,4) or (4,4), "
            f"got {tuple(extrinsic_tensor.shape)}"
        )
    if abs(float(torch.det(extrinsic[:, :3]).item())) < 1.0e-8:
        raise ValueError("extrinsic_cv rotation is singular")

    depth_m = depth_tensor * depth_scale
    valid = torch.isfinite(depth_m) & (depth_m >= min_depth) & (depth_m <= max_depth)
    rows = torch.arange(input_height, dtype=torch.float32) + 0.5
    cols = torch.arange(input_width, dtype=torch.float32) + 0.5
    v, u = torch.meshgrid(rows, cols, indexing="ij")
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    if float(fx.item()) <= 0.0 or float(fy.item()) <= 0.0:
        raise ValueError("intrinsic_cv focal lengths must be positive")
    x_camera = (u - intrinsic[0, 2]) * depth_m / fx
    y_camera = (v - intrinsic[1, 2]) * depth_m / fy
    camera_points = torch.stack((x_camera, y_camera, depth_m), dim=-1)

    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    world_points = torch.matmul(
        camera_points - translation.view(1, 1, 3),
        torch.linalg.inv(rotation).T,
    )

    block_height = input_height // output_height
    block_width = input_width // output_width

    def block_view(tensor: torch.Tensor) -> torch.Tensor:
        suffix = tuple(tensor.shape[2:])
        reshaped = tensor.reshape(
            output_height,
            block_height,
            output_width,
            block_width,
            *suffix,
        )
        permutation = (0, 2, 1, 3, *range(4, reshaped.ndim))
        return reshaped.permute(permutation).reshape(
            output_height, output_width, block_height * block_width, *suffix
        )

    block_depth = block_view(depth_m)
    block_valid = block_view(valid)
    block_points = block_view(world_points)
    nearest = torch.where(
        block_valid,
        block_depth,
        torch.full_like(block_depth, float("inf")),
    ).amin(dim=-1, keepdim=True)
    surface = block_valid & (block_depth <= nearest + surface_band)
    weights = surface.to(torch.float32)
    counts = weights.sum(dim=-1, keepdim=True)
    safe_counts = counts.clamp_min(1.0)
    mean = (block_points * weights.unsqueeze(-1)).sum(dim=-2) / safe_counts
    centered = block_points - mean.unsqueeze(-2)
    covariance = torch.einsum(
        "hwpd,hwpe,hwp->hwde", centered, centered, weights
    ) / safe_counts.unsqueeze(-1)
    valid_cell = counts[..., 0] > 0
    if covariance_floor:
        covariance = covariance + torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3) * covariance_floor
    mean = torch.where(valid_cell.unsqueeze(-1), mean, torch.zeros_like(mean))
    covariance = torch.where(
        valid_cell.unsqueeze(-1).unsqueeze(-1),
        covariance,
        torch.zeros_like(covariance),
    )
    packed = torch.cat(
        (
            mean.permute(2, 0, 1),
            covariance.reshape(output_height, output_width, 9).permute(2, 0, 1),
            valid_cell.to(torch.float32).unsqueeze(0),
        ),
        dim=0,
    )
    if tuple(packed.shape) != (METRIC_GEOMETRY_CHANNELS, output_height, output_width):
        raise RuntimeError(f"Internal metric geometry shape error: {tuple(packed.shape)}")
    if not bool(torch.isfinite(packed).all().item()):
        raise FloatingPointError("Metric geometry output contains non-finite values")
    return packed.to(torch.float16).contiguous()


def encode_metric_agent_geometry(
    observation: Mapping[str, Any],
    camera_names: list[str] | tuple[str, ...],
    *,
    output_size: tuple[int, int] = METRIC_GEOMETRY_SIZE,
    depth_scale: float = 0.001,
    max_depth: float = METRIC_GEOMETRY_MAX_DEPTH_M,
) -> torch.Tensor:
    """Encode named ManiSkill cameras from one observation as ``[N,13,H,W]``."""

    sensor_data = observation.get("sensor_data")
    sensor_param = observation.get("sensor_param")
    if not isinstance(sensor_data, Mapping) or not isinstance(sensor_param, Mapping):
        raise KeyError("Metric geometry requires observation sensor_data and sensor_param")
    frames: list[torch.Tensor] = []
    for camera_name in camera_names:
        data = sensor_data.get(camera_name)
        params = sensor_param.get(camera_name)
        if not isinstance(data, Mapping) or not isinstance(params, Mapping):
            raise KeyError(f"Missing calibrated camera observation for {camera_name}")
        if "depth" not in data or "intrinsic_cv" not in params or "extrinsic_cv" not in params:
            raise KeyError(
                f"Camera {camera_name} requires depth, intrinsic_cv, and extrinsic_cv"
            )
        frames.append(
            encode_metric_geometry(
                data["depth"],
                params["intrinsic_cv"],
                params["extrinsic_cv"],
                output_size=output_size,
                depth_scale=depth_scale,
                max_depth=max_depth,
            )
        )
    if not frames:
        raise ValueError("camera_names must be non-empty")
    return torch.stack(frames, dim=0)
