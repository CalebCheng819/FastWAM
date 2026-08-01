"""Pickle-safe random access to immutable canonical Gaussian cache shards."""

from __future__ import annotations

import bisect
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from typing_extensions import Self

from .manifest import load_manifest, sha256_file
from .schema import FrameKey, GaussianCacheSchema


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

    VERIFY_MODES: ClassVar[set[str]] = {"none", "manifest", "checksums"}

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
        self.manifest = load_manifest(self.cache_root, require_complete=True)
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
        if verify != "none":
            self._verify_shards(checksums=verify == "checksums")

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

    def _verify_shards(self, *, checksums: bool) -> None:
        for shard_id, record in self._shards.items():
            path = self.cache_root / str(record["path"])
            if not path.is_file():
                raise FileNotFoundError(f"Missing Gaussian cache shard {shard_id}: {path}")
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"Gaussian cache shard byte count mismatch: {path}")
            if checksums and sha256_file(path) != str(record["sha256"]):
                raise ValueError(f"Gaussian cache shard SHA-256 mismatch: {path}")

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
