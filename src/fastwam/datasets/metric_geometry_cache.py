"""Metadata-verified exact-key cache for metric geometry observations."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastwam.datasets.gaussian_cache.schema import FrameKey, normalize_source_path


SCHEMA_NAME = "fastwam.metric-geometry-cache"
SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
COMPLETE_FILENAME = "COMPLETE"


class MissingMetricGeometryFramesError(KeyError):
    """Raised when a requested training frame is absent from the cache."""


def _regular_file(root: Path, relative_path: str) -> Path:
    normalized = normalize_source_path(relative_path)
    path = root / normalized
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise RuntimeError(f"Metric geometry cache paths must not contain symlinks: {path}")
    try:
        resolved.relative_to(root)
    except ValueError:
        raise RuntimeError(f"Metric geometry cache path escapes cache root: {path}") from None
    if not path.is_file():
        raise FileNotFoundError(f"Metric geometry cache file is not regular: {path}")
    return path


def _stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {"bytes": int(value.st_size), "mtime_ns": int(value.st_mtime_ns)}


class MetricGeometryCache:
    """Lazy FP16 memmap with exact ``(source, trajectory, timestep, agent)`` keys."""

    def __init__(self, cache_root: str | Path) -> None:
        self._array: np.memmap | None = None
        self.cache_root = Path(cache_root).expanduser().resolve(strict=True)
        if not self.cache_root.is_dir():
            raise NotADirectoryError(f"Metric geometry cache root is not a directory: {self.cache_root}")
        complete = _regular_file(self.cache_root, COMPLETE_FILENAME)
        if complete.read_text(encoding="utf-8").strip() != "complete":
            raise ValueError(f"Invalid metric geometry completion marker: {complete}")
        manifest_path = _regular_file(self.cache_root, MANIFEST_FILENAME)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise TypeError("Metric geometry manifest must contain a JSON object")
        if manifest.get("schema_name") != SCHEMA_NAME or manifest.get("version") != SCHEMA_VERSION:
            raise ValueError(
                "Unsupported metric geometry schema: "
                f"name={manifest.get('schema_name')!r} version={manifest.get('version')!r}"
            )
        if manifest.get("dtype") != "float16" or manifest.get("byte_order") != "little":
            raise ValueError("Metric geometry cache must use little-endian float16")
        shape = tuple(int(value) for value in manifest.get("frame_shape", ()))
        if len(shape) != 3 or shape[0] != 13 or any(value < 1 for value in shape):
            raise ValueError(f"Invalid metric geometry frame_shape: {shape}")
        self.frame_shape = shape
        data = manifest.get("data")
        if not isinstance(data, Mapping):
            raise TypeError("Metric geometry manifest is missing data metadata")
        self.data_path = _regular_file(self.cache_root, str(data["path"]))
        observed = _stat(self.data_path)
        declared = {"bytes": int(data["bytes"]), "mtime_ns": int(data["mtime_ns"])}
        if observed != declared:
            raise ValueError(
                f"Metric geometry data metadata mismatch: declared={declared} observed={observed}"
            )
        self.frames = int(data["frames"])
        expected_bytes = self.frames * int(np.prod(self.frame_shape)) * np.dtype("<f2").itemsize
        if self.frames < 1 or observed["bytes"] != expected_bytes:
            raise ValueError(
                f"Metric geometry data byte count mismatch: expected={expected_bytes} observed={observed['bytes']}"
            )
        entries = manifest.get("entries")
        if not isinstance(entries, list) or len(entries) != self.frames:
            raise ValueError("Metric geometry entries must be a list with one record per frame")
        self._index: dict[FrameKey, int] = {}
        used_offsets: set[int] = set()
        for record in entries:
            if not isinstance(record, Mapping):
                raise TypeError("Metric geometry entry must be a JSON object")
            key = FrameKey.from_mapping(record)
            offset = int(record["offset"])
            if offset < 0 or offset >= self.frames or offset in used_offsets or key in self._index:
                raise ValueError(f"Invalid or duplicate metric geometry entry: {record}")
            self._index[key] = offset
            used_offsets.add(offset)
        if used_offsets != set(range(self.frames)):
            raise ValueError("Metric geometry offsets must cover [0, frames) exactly")
        self.manifest = dict(manifest)
        self.stat_contract = {
            "provenance_mode": "stat_cmp",
            "cache_root": str(self.cache_root),
            "schema_name": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
            "frame_shape": list(self.frame_shape),
            "frames": self.frames,
            "files": [
                {"path": MANIFEST_FILENAME, **_stat(manifest_path)},
                {"path": COMPLETE_FILENAME, **_stat(complete)},
                {"path": str(data["path"]), **observed},
            ],
        }

    @classmethod
    def open(cls, cache_root: str | Path) -> "MetricGeometryCache":
        return cls(cache_root)

    def _memmap(self) -> np.memmap:
        observed = _stat(self.data_path)
        data = self.manifest["data"]
        declared = {"bytes": int(data["bytes"]), "mtime_ns": int(data["mtime_ns"])}
        if observed != declared:
            raise RuntimeError("Metric geometry cache changed after it was opened")
        if self._array is None:
            self._array = np.memmap(
                self.data_path,
                mode="r",
                dtype=np.dtype("<f2"),
                shape=(self.frames, *self.frame_shape),
                order="C",
            )
        return self._array

    def preflight_keys(self, keys: Iterable[FrameKey | Mapping[str, Any]]) -> int:
        checked = 0
        missing: list[dict[str, Any]] = []
        missing_count = 0
        for value in keys:
            key = value if isinstance(value, FrameKey) else FrameKey.from_mapping(value)
            checked += 1
            if key not in self._index:
                missing_count += 1
                if len(missing) < 16:
                    missing.append(key.to_dict())
        if missing_count:
            raise MissingMetricGeometryFramesError(
                f"Metric geometry cache preflight failed: missing={missing_count}/{checked}, sample={missing}"
            )
        return checked

    def get_frame(self, key: FrameKey | Mapping[str, Any]) -> torch.Tensor:
        normalized = key if isinstance(key, FrameKey) else FrameKey.from_mapping(key)
        offset = self._index.get(normalized)
        if offset is None:
            raise MissingMetricGeometryFramesError(
                f"Metric geometry frame is not present: {normalized.to_dict()}"
            )
        frame = np.array(self._memmap()[offset], dtype=np.float16, copy=True, order="C")
        return torch.from_numpy(frame)

    def get_agents(
        self,
        source_path: str,
        trajectory: str,
        timestep: int,
        agent_names: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        if not agent_names:
            raise ValueError("agent_names must be non-empty")
        frames = [
            self.get_frame(FrameKey(source_path, trajectory, timestep, str(agent_name)))
            for agent_name in agent_names
        ]
        return {"agent_gaussian": torch.stack(frames, dim=0)}

    def close(self) -> None:
        if self._array is not None:
            mmap = getattr(self._array, "_mmap", None)
            if mmap is not None and not mmap.closed:
                mmap.close()
            self._array = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_array"] = None
        return state

    def __del__(self) -> None:
        self.close()
