from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fastwam.datasets.gaussian_cache import FrameKey
from fastwam.datasets.metric_geometry import (
    encode_metric_agent_geometry,
    encode_metric_geometry,
)
from fastwam.datasets.metric_geometry_cache import (
    MetricGeometryCache,
    MissingMetricGeometryFramesError,
)
from scripts.build_robofactory_metric_geometry_cache import STAT_CMP_ALLOWLIST


def _intrinsic() -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _extrinsic(translation=(0.0, 0.0, 0.0)) -> torch.Tensor:
    value = torch.eye(4, dtype=torch.float32)
    value[:3, 3] = torch.tensor(translation, dtype=torch.float32)
    return value[:3]


def test_metric_geometry_unprojects_to_world_coordinates() -> None:
    frame = encode_metric_geometry(
        torch.full((4, 4, 1), 1000, dtype=torch.int16),
        _intrinsic().unsqueeze(0),
        _extrinsic((0.25, -0.5, 0.75)).unsqueeze(0),
        output_size=(1, 1),
    )
    assert frame.shape == (13, 1, 1)
    assert frame.dtype == torch.float16
    # Pixel centers span camera x/y=[0,1,2,3], then world=cam-translation.
    torch.testing.assert_close(
        frame[:3, 0, 0].float(),
        torch.tensor([1.25, 2.0, 0.25]),
        atol=2.0e-3,
        rtol=0.0,
    )
    assert float(frame[12, 0, 0]) == 1.0
    covariance = frame[3:12, 0, 0].float().reshape(3, 3)
    torch.testing.assert_close(covariance, covariance.T, atol=1.0e-4, rtol=0.0)
    assert bool(torch.linalg.eigvalsh(covariance).min() >= -1.0e-4)


def test_metric_geometry_preserves_nearest_small_surface() -> None:
    depth = torch.full((4, 4), 2000, dtype=torch.int16)
    depth[1, 2] = 1000
    frame = encode_metric_geometry(
        depth,
        _intrinsic(),
        _extrinsic(),
        output_size=(1, 1),
        surface_band=0.03,
    )
    torch.testing.assert_close(
        frame[:3, 0, 0].float(),
        torch.tensor([2.0, 1.0, 1.0]),
        atol=2.0e-3,
        rtol=0.0,
    )


def test_metric_geometry_invalid_depth_produces_zero_cell() -> None:
    frame = encode_metric_geometry(
        torch.zeros(4, 4, dtype=torch.int16),
        _intrinsic(),
        _extrinsic(),
        output_size=(1, 1),
    )
    assert torch.equal(frame, torch.zeros_like(frame))


def test_metric_geometry_rejects_far_background_depth() -> None:
    frame = encode_metric_geometry(
        torch.full((4, 4), 3001, dtype=torch.int16),
        _intrinsic(),
        _extrinsic(),
        output_size=(1, 1),
    )
    assert torch.equal(frame, torch.zeros_like(frame))


def test_metric_agent_geometry_reads_maniskill_sensor_contract() -> None:
    observation = {
        "sensor_data": {
            "head_camera_agent0": {
                "depth": torch.full((1, 4, 4, 1), 1000, dtype=torch.int16),
            }
        },
        "sensor_param": {
            "head_camera_agent0": {
                "intrinsic_cv": _intrinsic().unsqueeze(0),
                "extrinsic_cv": _extrinsic().unsqueeze(0),
            }
        },
    }
    frames = encode_metric_agent_geometry(
        observation,
        ["head_camera_agent0"],
        output_size=(1, 1),
    )
    assert frames.shape == (1, 13, 1, 1)
    assert frames.dtype == torch.float16


def _write_cache(root: Path) -> FrameKey:
    root.mkdir()
    key = FrameKey("task/demo.h5", "traj_0", 16, "panda-0")
    values = np.arange(13 * 2 * 3, dtype=np.float16).reshape(1, 13, 2, 3)
    data_path = root / "frames.f16"
    values.tofile(data_path)
    stat = data_path.stat()
    manifest = {
        "schema_name": "fastwam.metric-geometry-cache",
        "version": 1,
        "dtype": "float16",
        "byte_order": "little",
        "frame_shape": [13, 2, 3],
        "data": {
            "path": "frames.f16",
            "frames": 1,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "entries": [{**key.to_dict(), "offset": 0}],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    (root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return key


def test_metric_geometry_cache_exact_key_and_metadata(tmp_path: Path) -> None:
    key = _write_cache(tmp_path / "cache")
    cache = MetricGeometryCache.open(tmp_path / "cache")
    assert cache.preflight_keys([key]) == 1
    frame = cache.get_frame(key)
    assert frame.shape == (13, 2, 3)
    assert frame.dtype == torch.float16
    with pytest.raises(MissingMetricGeometryFramesError, match="not present"):
        cache.get_frame(FrameKey(key.source_path, key.trajectory, 17, key.agent_name))
    cache.close()


def test_metric_geometry_cache_detects_changed_data(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    _write_cache(root)
    path = root / "frames.f16"
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="metadata mismatch"):
        MetricGeometryCache.open(root)


def test_metric_cache_allowlist_contains_only_runtime_files() -> None:
    assert STAT_CMP_ALLOWLIST == "stat-cmp.allowlist"
