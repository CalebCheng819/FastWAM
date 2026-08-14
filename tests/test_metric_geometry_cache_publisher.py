from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import numpy as np
import pytest

from fastwam.datasets.gaussian_cache import FrameKey
from scripts.publish_metric_geometry_cache import (
    ALLOWLIST,
    Publisher,
    publish_with_retry,
    validate_cache,
)


def write_cache(root: Path) -> None:
    root.mkdir()
    frames = 2
    shape = [13, 2, 3]
    values = np.arange(frames * np.prod(shape), dtype="<f2")
    data_path = root / "frames.f16"
    values.tofile(data_path)
    data_stat = data_path.stat()
    entries = []
    for index in range(frames):
        key = FrameKey("PlaceFood-rf/demo.h5", "traj_0", 0, f"agent-{index}")
        entries.append({**key.to_dict(), "offset": index})
    manifest = {
        "schema_name": "fastwam.metric-geometry-cache",
        "version": 1,
        "dtype": "float16",
        "byte_order": "little",
        "frame_shape": shape,
        "data": {
            "path": "frames.f16",
            "frames": frames,
            "bytes": data_stat.st_size,
            "mtime_ns": data_stat.st_mtime_ns,
        },
        "counts": {"frames": frames, "windows": 1},
        "entries": entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "stat-cmp.allowlist").write_text(ALLOWLIST, encoding="utf-8")
    (root / "COMPLETE").write_text("complete\n", encoding="utf-8")


def test_publish_preserves_bytes_mtime_and_complete_last(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    write_cache(source)
    summary = Publisher(source, target, "test-run").publish_once()
    assert summary == {"bytes": 312, "frames": 2, "windows": 1}
    assert validate_cache(target) == summary
    for name in ("frames.f16", "manifest.json", "stat-cmp.allowlist", "COMPLETE"):
        assert (source / name).read_bytes() == (target / name).read_bytes()
        assert (source / name).stat().st_mtime_ns == (target / name).stat().st_mtime_ns


def test_publish_rejects_preexisting_versioned_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    write_cache(source)
    target.mkdir()
    with pytest.raises(FileExistsError, match="new versioned target"):
        Publisher(source, target, "test-run").publish_once()


def test_publish_retries_transient_mount_error(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    write_cache(source)
    publisher = Publisher(source, target, "test-run")
    original = publisher.publish_once
    attempts = 0

    def flaky_publish():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(errno.ENOTCONN, os.strerror(errno.ENOTCONN))
        return original()

    monkeypatch.setattr(publisher, "publish_once", flaky_publish)
    monkeypatch.setattr("scripts.publish_metric_geometry_cache.time.sleep", lambda _: None)
    summary = publish_with_retry(publisher, timeout_seconds=1.0, poll_seconds=0.01)
    assert attempts == 2
    assert summary["frames"] == 2
    assert (target / "COMPLETE").read_text(encoding="utf-8") == "complete\n"
