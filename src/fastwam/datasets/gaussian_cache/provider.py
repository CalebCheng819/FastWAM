"""Pickle-safe random access to immutable canonical Gaussian cache shards."""

from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from typing_extensions import Self

from .manifest import (
    load_manifest,
    read_regular_file_snapshot,
    regular_file_stat,
    sha256_file,
)
from .schema import (
    COMPLETE_FILENAME,
    MANIFEST_FILENAME,
    FrameKey,
    GaussianCacheSchema,
    normalize_source_path,
)
from .selection import expand_selection_record


class MissingGaussianFramesError(KeyError):
    """Raised by fail-closed split preflight when one or more keys are absent."""


@dataclass(frozen=True)
class _Segment:
    shard: str
    offset: int
    count: int
    source_start: int
    source_stride: int

    @property
    def source_end(self) -> int:
        return self.source_start + (self.count - 1) * self.source_stride

    def locate(self, timestep: int) -> int | None:
        delta = int(timestep) - self.source_start
        if delta < 0 or delta % self.source_stride:
            return None
        index = delta // self.source_stride
        if index >= self.count:
            return None
        return self.offset + index


class _StreamIndex:
    def __init__(self, record: Mapping[str, Any]) -> None:
        self.observation_count = int(record["observation_count"])
        self.stored_count = int(record["stored_count"])
        self.segments = tuple(
            _Segment(
                shard=str(segment["shard"]),
                offset=int(segment["offset"]),
                count=int(segment["count"]),
                source_start=int(segment["source_start"]),
                source_stride=int(segment["source_stride"]),
            )
            for segment in record["segments"]
        )
        self.starts = tuple(segment.source_start for segment in self.segments)

    def locate(self, timestep: int) -> tuple[str, int] | None:
        index = bisect.bisect_right(self.starts, int(timestep)) - 1
        if index < 0:
            return None
        segment = self.segments[index]
        offset = segment.locate(int(timestep))
        return None if offset is None else (segment.shard, offset)

    def iter_timesteps(self) -> Iterator[int]:
        for segment in self.segments:
            for index in range(segment.count):
                yield segment.source_start + index * segment.source_stride


class GaussianCache:
    """Lazy memmap reader for canonical or compact per-agent Gaussian frames.

    ``GaussianCache`` contains no live file descriptor until the first frame is
    requested.  ``__getstate__`` drops all memmaps, so a dataset may be pickled
    into DataLoader workers and reopen shards lazily in each worker.
    """

    VERIFY_MODES: ClassVar[set[str]] = {
        "none",
        "manifest",
        "checksums",
        "stat_cmp",
    }

    def __init__(
        self,
        cache_root: str | Path,
        *,
        verify: str = "manifest",
        max_open_shards: int = 64,
    ) -> None:
        verify = str(verify).lower()
        if verify not in self.VERIFY_MODES:
            raise ValueError(f"verify must be one of {sorted(self.VERIFY_MODES)}, got {verify!r}")
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.verify = verify
        self.max_open_shards = int(max_open_shards)
        if self.max_open_shards <= 0:
            raise ValueError("max_open_shards must be positive")
        self.manifest = load_manifest(
            self.cache_root,
            require_complete=True,
            provenance_mode="stat_cmp" if verify == "stat_cmp" else "sha256",
        )
        self.schema = GaussianCacheSchema.from_dict(self.manifest["schema"])
        self._shards = {str(record["id"]): dict(record) for record in self.manifest["shards"]}
        self._streams: dict[tuple[str, str, str], _StreamIndex] = {}
        for record in self.manifest["streams"]:
            key = (
                str(record["source_path"]),
                str(record["trajectory"]),
                str(record["agent_name"]),
            )
            self._streams[key] = _StreamIndex(record)
        self._arrays: OrderedDict[str, np.memmap] = OrderedDict()
        self._stat_cmp_shards: dict[str, dict[str, int]] = {}
        self.stat_contract: dict[str, Any] | None = None
        if verify != "none":
            self._verify_shards(
                checksums=verify == "checksums",
                stat_cmp=verify == "stat_cmp",
            )
        if verify == "stat_cmp":
            self._build_stat_cmp_contract()

    @classmethod
    def open(
        cls,
        cache_root: str | Path,
        *,
        verify: str = "manifest",
        max_open_shards: int = 64,
    ) -> GaussianCache:
        return cls(
            cache_root,
            verify=verify,
            max_open_shards=max_open_shards,
        )

    def _cache_file(self, relative_path: str) -> Path:
        normalized = normalize_source_path(relative_path)
        path = self.cache_root / normalized
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing Gaussian cache file: {path}") from None
        if resolved != path:
            raise RuntimeError(f"Gaussian cache path must not contain symlinks: {path}")
        try:
            resolved.relative_to(self.cache_root)
        except ValueError:
            raise RuntimeError(f"Gaussian cache path escapes cache root: {path}") from None
        return path

    def _verify_shards(self, *, checksums: bool, stat_cmp: bool = False) -> None:
        for shard_id, record in self._shards.items():
            path = (
                self._cache_file(str(record["path"]))
                if stat_cmp
                else self.cache_root / str(record["path"])
            )
            if stat_cmp:
                metadata = regular_file_stat(path)
            else:
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing Gaussian cache shard {shard_id}: {path}"
                    )
                metadata = {
                    "bytes": int(path.stat().st_size),
                    "mtime_ns": int(path.stat().st_mtime_ns),
                }
            if metadata["bytes"] != int(record["bytes"]):
                raise ValueError(f"Gaussian cache shard byte count mismatch: {path}")
            if checksums and sha256_file(path) != str(record["sha256"]):
                raise ValueError(f"Gaussian cache shard SHA-256 mismatch: {path}")
            if stat_cmp:
                self._stat_cmp_shards[shard_id] = metadata

    def _load_stat_cmp_selection(self, path: Path) -> tuple[list[FrameKey], dict[str, int]]:
        payload, metadata = read_regular_file_snapshot(path)
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Invalid Gaussian selection JSONL encoding: {path}") from error
        keys: set[FrameKey] = set()
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise TypeError("record is not a JSON object")
                keys.update(expand_selection_record(record))
            except Exception as error:
                raise ValueError(
                    f"Invalid Gaussian selection JSONL at {path}:{line_number}: {error}"
                ) from error
        if not keys:
            raise ValueError(f"Gaussian selection JSONL contains no keys: {path}")
        return sorted(keys), metadata

    def _build_stat_cmp_contract(self) -> None:
        records: list[dict[str, Any]] = []
        for filename in (MANIFEST_FILENAME, COMPLETE_FILENAME):
            path = self._cache_file(filename)
            metadata = regular_file_stat(path)
            records.append({"path": filename, **metadata})
        for shard_id, record in sorted(self._shards.items()):
            records.append(
                {
                    "path": str(record["path"]),
                    **self._stat_cmp_shards[shard_id],
                }
            )

        selection = self.manifest["selection"]
        selected_key_count = int(selection["selected_key_count"])
        if selection["mode"] == "index":
            selection_path = self._cache_file(str(selection["index_filename"]))
            selection_keys, metadata = self._load_stat_cmp_selection(selection_path)
            if len(selection_keys) != selected_key_count:
                raise ValueError(
                    "Gaussian cache selection index count mismatch: "
                    f"declared={selected_key_count} actual={len(selection_keys)} "
                    f"path={selection_path}"
                )
            self.preflight_keys(selection_keys)
            records.append({"path": str(selection["index_filename"]), **metadata})
        elif selected_key_count != int(self.manifest["total_frames"]):
            raise ValueError(
                "Gaussian cache all-selection count does not equal total_frames: "
                f"selected={selected_key_count} total={self.manifest['total_frames']}"
            )

        self.stat_contract = {
            "provenance_mode": "stat_cmp",
            "cache_root": str(self.cache_root),
            "schema": self.schema.to_dict(),
            "selected_key_count": selected_key_count,
            "shard_count": len(self._shards),
            "file_count": len(records),
            "files": records,
        }

    def _coerce_key(self, key: FrameKey | Mapping[str, Any]) -> FrameKey:
        return key if isinstance(key, FrameKey) else FrameKey.from_mapping(key)

    def _locate(self, key: FrameKey) -> tuple[str, int] | None:
        stream = self._streams.get((key.source_path, key.trajectory, key.agent_name))
        return None if stream is None else stream.locate(key.timestep)

    def contains_frame(self, key: FrameKey | Mapping[str, Any]) -> bool:
        return self._locate(self._coerce_key(key)) is not None

    def preflight_keys(
        self,
        keys: Iterable[FrameKey | Mapping[str, Any]],
        *,
        missing_sample_limit: int = 16,
    ) -> int:
        """Validate an entire split before training and return the covered key count."""

        checked = 0
        missing_count = 0
        missing_sample: list[dict[str, Any]] = []
        for value in keys:
            key = self._coerce_key(value)
            checked += 1
            if self._locate(key) is None:
                missing_count += 1
                if len(missing_sample) < missing_sample_limit:
                    missing_sample.append(key.to_dict())
        if missing_count:
            raise MissingGaussianFramesError(
                "Gaussian cache preflight failed: "
                f"missing={missing_count}/{checked}, sample={missing_sample}"
            )
        return checked

    def _array(self, shard_id: str) -> np.memmap:
        array = self._arrays.pop(shard_id, None)
        if array is not None:
            self._arrays[shard_id] = array
            return array
        while len(self._arrays) >= self.max_open_shards:
            _, evicted = self._arrays.popitem(last=False)
            self._close_memmap(evicted)
        record = self._shards[shard_id]
        path = self.cache_root / str(record["path"])
        if self.verify == "stat_cmp":
            observed = regular_file_stat(path)
            if observed != self._stat_cmp_shards[shard_id]:
                raise RuntimeError(
                    f"Gaussian cache shard stat comparison mismatch: {path}"
                )
        array = np.memmap(
            path,
            mode="r",
            dtype=np.dtype("<f2"),
            shape=(int(record["frames"]), *self.schema.frame_shape),
            order="C",
        )
        self._arrays[shard_id] = array
        return array

    @staticmethod
    def _close_memmap(array: np.memmap) -> None:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None and not mmap.closed:
            mmap.close()

    def get_frame(self, key: FrameKey | Mapping[str, Any]) -> torch.Tensor:
        key = self._coerce_key(key)
        location = self._locate(key)
        if location is None:
            raise MissingGaussianFramesError(f"Gaussian frame is not present: {key.to_dict()}")
        shard_id, offset = location
        # Copy out of the read-only memmap so the returned tensor is writable and
        # independent from close()/DataLoader worker lifecycle.
        frame = np.array(self._array(shard_id)[offset], dtype=np.float16, copy=True, order="C")
        return torch.from_numpy(frame)

    def get_agents(
        self,
        source_path: str,
        trajectory: str,
        timestep: int,
        agent_names: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Return the model/dataset contract ``agent_gaussian: [N,13,H,W]``.

        ``agent_names`` order is preserved exactly, including a dataset's random
        agent permutation.
        """

        if not agent_names:
            raise ValueError("agent_names must be non-empty")
        frames = [
            self.get_frame(FrameKey(source_path, trajectory, timestep, str(agent_name)))
            for agent_name in agent_names
        ]
        return {"agent_gaussian": torch.stack(frames, dim=0)}

    def iter_keys(self) -> Iterator[FrameKey]:
        for source_path, trajectory, agent_name in sorted(self._streams):
            stream = self._streams[(source_path, trajectory, agent_name)]
            for timestep in stream.iter_timesteps():
                yield FrameKey(source_path, trajectory, timestep, agent_name)

    def close(self) -> None:
        while self._arrays:
            _, array = self._arrays.popitem(last=False)
            self._close_memmap(array)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = OrderedDict()
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(state)
        self._arrays = OrderedDict()
