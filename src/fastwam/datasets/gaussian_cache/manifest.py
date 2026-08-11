"""Immutable shard writer and fail-closed Gaussian cache manifests."""

from __future__ import annotations

import hashlib
import errno
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .schema import (
    COMPLETE_FILENAME,
    DEFAULT_TARGET_SHARD_BYTES,
    MANIFEST_FILENAME,
    MAX_SHARD_BYTES,
    MIN_SHARD_BYTES,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    GaussianCacheSchema,
    normalize_source_path,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_immutable_file(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable cache file: {path}")
    created = False
    try:
        # This is deliberately an exclusive write to the final small-object
        # key.  Object-store FUSE renames are copy+delete rather than atomic;
        # cache validity instead comes from writing COMPLETE strictly last.
        with path.open("xb") as handle:
            created = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, mode)
    except Exception:
        if created and path.exists():
            path.unlink()
        raise


ShardOffloader = Callable[..., Mapping[str, Any] | None]


def copy_staged_shard(
    staged_path: Path,
    final_path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    verify_checksum: bool,
) -> dict[str, Any]:
    """Stream one locally staged shard to its final object key.

    The destination is opened exclusively and never reached through rename.
    This is safe for OSSFS semantics: an interrupted upload cannot make a
    cache readable because manifest/COMPLETE are not emitted until every final
    object has passed readback validation.
    """

    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite shard {final_path}")
    created = False
    try:
        with staged_path.open("rb") as source, final_path.open("xb") as destination:
            created = True
            shutil.copyfileobj(source, destination, length=16 << 20)
            destination.flush()
            os.fsync(destination.fileno())
        if final_path.stat().st_size != int(expected_bytes):
            raise RuntimeError(
                "Uploaded shard byte count mismatch: "
                f"expected={expected_bytes} actual={final_path.stat().st_size}"
            )
        if verify_checksum and sha256_file(final_path) != expected_sha256:
            raise RuntimeError(f"Uploaded shard checksum mismatch: {final_path}")
        os.chmod(final_path, 0o444)
        return {
            "bytes_verified": int(expected_bytes),
            "checksum_verified": bool(verify_checksum),
            "sha256": expected_sha256 if verify_checksum else None,
        }
    except Exception:
        if created and final_path.exists():
            final_path.unlink()
        raise


def source_record(path: str | Path, *, source_root: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    root = Path(source_root).resolve()
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Source HDF5 {source} is outside source_root {root}") from exc
    relative = normalize_source_path(relative)
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": relative,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


@dataclass(frozen=True)
class WriteSpan:
    shard: str
    offset: int
    count: int


class ImmutableShardWriter:
    """Append fixed-shape FP16 frames to content-checked raw binary shards."""

    def __init__(
        self,
        output_root: str | Path,
        schema: GaussianCacheSchema,
        *,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        staging_dir: str | Path | None = None,
        offloader: ShardOffloader | None = None,
        verify_uploaded_checksum: bool = True,
    ) -> None:
        self.output_root = Path(output_root)
        self.schema = schema
        self.target_shard_bytes = int(target_shard_bytes)
        if not MIN_SHARD_BYTES + schema.frame_bytes <= self.target_shard_bytes <= MAX_SHARD_BYTES:
            raise ValueError(
                "target_shard_bytes must leave room for a >=1 GiB non-final shard and be <=4 GiB; "
                f"got {self.target_shard_bytes} for frame_bytes={schema.frame_bytes}"
            )
        self.shards_dir = self.output_root / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=False)
        staging_base = (
            Path(staging_dir).expanduser().resolve()
            if staging_dir is not None
            else Path(tempfile.gettempdir()).resolve()
        )
        staging_base.mkdir(parents=True, exist_ok=True)
        try:
            staging_base.relative_to(self.output_root.expanduser().resolve())
        except ValueError:
            pass
        else:
            raise ValueError(
                "staging_dir must be outside output_root; use local disk or CPFS, not OSSFS"
            )
        self._staging_task_dir = Path(
            tempfile.mkdtemp(prefix="fastwam-gaussian-", dir=staging_base)
        )
        self._offloader = offloader or copy_staged_shard
        self.verify_uploaded_checksum = bool(verify_uploaded_checksum)
        self._handle = None
        self._partial_path: Path | None = None
        self._current_id: str | None = None
        self._current_frames = 0
        self._records: list[dict[str, Any]] = []
        self._finished = False

    def _open(self) -> None:
        if self._handle is not None:
            return
        shard_id = f"{len(self._records):06d}"
        partial = self._staging_task_dir / f"shard-{shard_id}.{uuid.uuid4().hex}.partial"
        self._handle = partial.open("xb")
        self._partial_path = partial
        self._current_id = shard_id
        self._current_frames = 0

    def _finalize_current(self) -> None:
        if self._handle is None:
            return
        if self._current_frames <= 0 or self._partial_path is None or self._current_id is None:
            raise RuntimeError("Cannot finalize an empty Gaussian cache shard")
        handle = self._handle
        partial = self._partial_path
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        self._handle = None
        expected_bytes = self._current_frames * self.schema.frame_bytes
        actual_bytes = partial.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"Shard byte count mismatch before sealing: expected={expected_bytes} actual={actual_bytes}"
            )
        digest = sha256_file(partial)
        filename = f"shard-{self._current_id}-{digest[:16]}.f16"
        final_path = self.shards_dir / filename
        if final_path.exists():
            raise FileExistsError(f"Refusing to overwrite shard {final_path}")
        try:
            receipt = self._offloader(
                partial,
                final_path,
                expected_bytes=actual_bytes,
                expected_sha256=digest,
                verify_checksum=self.verify_uploaded_checksum,
            )
            if not final_path.is_file():
                raise FileNotFoundError(f"Shard offloader did not create final object: {final_path}")
            if final_path.stat().st_size != actual_bytes:
                raise RuntimeError(
                    "Shard final-object readback size mismatch: "
                    f"expected={actual_bytes} actual={final_path.stat().st_size}"
                )
            checksum_receipted = (
                isinstance(receipt, Mapping)
                and receipt.get("checksum_verified") is True
                and int(receipt.get("bytes_verified", -1)) == actual_bytes
                and receipt.get("sha256") == digest
            )
            if (
                self.verify_uploaded_checksum
                and not checksum_receipted
                and sha256_file(final_path) != digest
            ):
                raise RuntimeError(f"Shard final-object readback checksum mismatch: {final_path}")
        except Exception:
            # The path was known absent before this task's offloader call, so a
            # partial object at this exact content-addressed key belongs to us.
            if final_path.exists():
                final_path.unlink()
            raise
        partial.unlink()
        self._records.append(
            {
                "id": self._current_id,
                "path": f"shards/{filename}",
                "sha256": digest,
                "bytes": actual_bytes,
                "frames": self._current_frames,
                "final": False,
                "immutable": True,
            }
        )
        self._partial_path = None
        self._current_id = None
        self._current_frames = 0

    def append(self, frames: torch.Tensor | np.ndarray) -> list[WriteSpan]:
        if self._finished:
            raise RuntimeError("Cannot append after shard writer finish()")
        tensor = frames if isinstance(frames, torch.Tensor) else torch.as_tensor(frames)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        self.schema.validate_tensor(tensor, name="frames")
        tensor = tensor.detach().to(device="cpu").contiguous()
        array = np.asarray(tensor.numpy(), dtype=np.dtype("<f2"), order="C")
        spans: list[WriteSpan] = []
        cursor = 0
        while cursor < array.shape[0]:
            self._open()
            assert self._handle is not None and self._current_id is not None
            remaining_bytes = self.target_shard_bytes - self._current_frames * self.schema.frame_bytes
            capacity = remaining_bytes // self.schema.frame_bytes
            if capacity <= 0:
                self._finalize_current()
                continue
            take = min(int(capacity), int(array.shape[0] - cursor))
            offset = self._current_frames
            chunk = np.ascontiguousarray(array[cursor : cursor + take], dtype=np.dtype("<f2"))
            self._handle.write(chunk.tobytes(order="C"))
            self._current_frames += take
            spans.append(WriteSpan(self._current_id, offset, take))
            cursor += take
            if self.target_shard_bytes - self._current_frames * self.schema.frame_bytes < self.schema.frame_bytes:
                self._finalize_current()
        return spans

    def finish(self) -> list[dict[str, Any]]:
        if self._finished:
            return [dict(record) for record in self._records]
        self._finalize_current()
        if not self._records:
            raise RuntimeError("Cannot finish an empty Gaussian cache")
        self._records[-1]["final"] = True
        for record in self._records:
            size = int(record["bytes"])
            if size > MAX_SHARD_BYTES:
                raise RuntimeError(f"Shard exceeds 4 GiB: {record}")
            if not record["final"] and size < MIN_SHARD_BYTES:
                raise RuntimeError(f"Non-final shard is smaller than 1 GiB: {record}")
        self._finished = True
        self._cleanup_staging_dir()
        return [dict(record) for record in self._records]

    def _cleanup_staging_dir(self) -> None:
        if self._staging_task_dir.exists():
            # rmdir never traverses or deletes caller-owned staging content.
            self._staging_task_dir.rmdir()

    def abort(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._partial_path is not None and self._partial_path.exists():
            self._partial_path.unlink()
        self._partial_path = None
        try:
            self._cleanup_staging_dir()
        except OSError:
            # Preserve unexpected files instead of recursively deleting a
            # staging directory.  Only this task's known partial is removed.
            pass


def _segment_runs(timesteps: Sequence[int]) -> Iterable[tuple[int, int, int, int]]:
    """Yield ``(list_offset, source_start, stride, count)`` arithmetic runs."""

    if not timesteps:
        return
    start_offset = 0
    while start_offset < len(timesteps):
        if start_offset + 1 == len(timesteps):
            yield start_offset, int(timesteps[start_offset]), 1, 1
            return
        stride = int(timesteps[start_offset + 1]) - int(timesteps[start_offset])
        end = start_offset + 2
        while end < len(timesteps) and int(timesteps[end]) - int(timesteps[end - 1]) == stride:
            end += 1
        yield start_offset, int(timesteps[start_offset]), stride, end - start_offset
        start_offset = end


class GaussianCacheBuilder:
    """Build a new cache root and seal manifest/COMPLETE only after all shards."""

    def __init__(
        self,
        output_root: str | Path,
        schema: GaussianCacheSchema,
        *,
        sources: Sequence[Mapping[str, Any]],
        teacher: Mapping[str, Any],
        selection: Mapping[str, Any],
        producer: Mapping[str, Any] | None = None,
        derivation: Mapping[str, Any] | None = None,
        partition: Mapping[str, Any] | None = None,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        staging_dir: str | Path | None = None,
        offloader: ShardOffloader | None = None,
        verify_uploaded_checksum: bool = True,
    ) -> None:
        self.output_root = Path(output_root)
        if self.output_root.exists():
            raise FileExistsError(f"Output cache root already exists: {self.output_root}")
        self.output_root.mkdir(parents=True)
        self.schema = schema
        self.sources = [dict(record) for record in sources]
        self.teacher = dict(teacher)
        self.selection = dict(selection)
        self.producer = None if producer is None else dict(producer)
        self.derivation = None if derivation is None else dict(derivation)
        self.partition = None if partition is None else dict(partition)
        self.target_shard_bytes = int(target_shard_bytes)
        self.writer = ImmutableShardWriter(
            self.output_root,
            schema,
            target_shard_bytes=self.target_shard_bytes,
            staging_dir=staging_dir,
            offloader=offloader,
            verify_uploaded_checksum=verify_uploaded_checksum,
        )
        self._streams: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._finished = False

    @staticmethod
    def _validate_sources(
        sources: Sequence[Mapping[str, Any]],
        *,
        validate_hash_fields: bool = True,
    ) -> None:
        seen: set[str] = set()
        for record in sources:
            path = normalize_source_path(str(record["path"]))
            if path in seen:
                raise ValueError(f"Duplicate source HDF5 record: {path}")
            seen.add(path)
            if int(record["bytes"]) < 0:
                raise ValueError(f"Invalid source record: {record}")
            if validate_hash_fields and not _SHA256_RE.fullmatch(
                str(record["sha256"])
            ):
                raise ValueError(f"Invalid source record: {record}")

    def append_stream(
        self,
        *,
        source_path: str,
        trajectory: str,
        agent_name: str,
        observation_count: int,
        timesteps: Sequence[int],
        frames: torch.Tensor | np.ndarray,
    ) -> None:
        if self._finished:
            raise RuntimeError("Cannot append after GaussianCacheBuilder.finish()")
        source_path = normalize_source_path(source_path)
        times = [int(value) for value in timesteps]
        if not times:
            return
        if times != sorted(set(times)):
            raise ValueError("timesteps must be strictly increasing and unique")
        observation_count = int(observation_count)
        if times[0] < 0 or times[-1] >= observation_count:
            raise ValueError(
                f"timesteps {times[0]}..{times[-1]} exceed observation_count={observation_count}"
            )
        tensor = frames if isinstance(frames, torch.Tensor) else torch.as_tensor(frames)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] != len(times):
            raise ValueError(f"frames/timesteps length mismatch: {tensor.shape[0]} vs {len(times)}")
        key = (source_path, str(trajectory), str(agent_name))
        stream = self._streams.setdefault(
            key,
            {
                "source_path": source_path,
                "trajectory": str(trajectory),
                "agent_name": str(agent_name),
                "observation_count": observation_count,
                "stored_count": 0,
                "segments": [],
                "last_timestep": -1,
            },
        )
        if int(stream["observation_count"]) != observation_count:
            raise ValueError(f"observation_count changed for stream {key}")
        if times[0] <= int(stream["last_timestep"]):
            raise ValueError(f"Stream timesteps must be appended in order for {key}")

        spans = self.writer.append(tensor)
        consumed = 0
        for span in spans:
            span_times = times[consumed : consumed + span.count]
            for local_offset, source_start, stride, count in _segment_runs(span_times):
                segment = {
                    "shard": span.shard,
                    "offset": span.offset + local_offset,
                    "count": count,
                    "source_start": source_start,
                    "source_stride": stride,
                }
                previous = stream["segments"][-1] if stream["segments"] else None
                if (
                    previous is not None
                    and previous["shard"] == segment["shard"]
                    and int(previous["offset"]) + int(previous["count"]) == segment["offset"]
                    and int(previous["source_stride"]) == stride
                    and int(previous["source_start"]) + int(previous["count"]) * stride
                    == source_start
                ):
                    previous["count"] = int(previous["count"]) + count
                else:
                    stream["segments"].append(segment)
            consumed += span.count
        if consumed != len(times):
            raise RuntimeError("Internal shard span accounting mismatch")
        stream["stored_count"] = int(stream["stored_count"]) + len(times)
        stream["last_timestep"] = times[-1]

    def finish(self) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("GaussianCacheBuilder.finish() may only be called once")
        self._validate_sources(self.sources)
        shards = self.writer.finish()
        streams = []
        for key in sorted(self._streams):
            stream = dict(self._streams[key])
            stream.pop("last_timestep", None)
            streams.append(stream)
        total_frames = sum(int(stream["stored_count"]) for stream in streams)
        if total_frames != sum(int(record["frames"]) for record in shards):
            raise RuntimeError("Manifest stream and shard frame totals disagree")
        manifest: dict[str, Any] = {
            "manifest_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "schema": self.schema.to_dict(),
            "target_shard_bytes": self.target_shard_bytes,
            "selection": self.selection,
            "teacher": self.teacher,
            "sources": sorted(self.sources, key=lambda record: str(record["path"])),
            "shards": shards,
            "streams": streams,
            "total_frames": total_frames,
        }
        if self.producer is not None:
            manifest["producer"] = self.producer
        if self.derivation is not None:
            manifest["derivation"] = self.derivation
        if self.partition is not None:
            manifest["partition"] = self.partition
        validate_manifest_structure(manifest)
        seal_manifest(self.output_root, manifest)
        self._finished = True
        return manifest

    def abort(self) -> None:
        self.writer.abort()


def seal_manifest(
    output_root: str | Path,
    manifest: Mapping[str, Any],
    *,
    before_complete: Callable[[Path, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Validate and seal manifest/COMPLETE after every shard object exists."""

    root = Path(output_root)
    validate_manifest_structure(manifest)
    manifest_payload = canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    write_immutable_file(root / MANIFEST_FILENAME, manifest_payload)
    complete = {
        "complete": True,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "manifest_version": int(manifest["manifest_version"]),
        "manifest": MANIFEST_FILENAME,
        "manifest_bytes": len(manifest_payload),
        "manifest_sha256": manifest_sha256,
        "shard_count": len(manifest["shards"]),
        "total_frames": int(manifest["total_frames"]),
    }
    if before_complete is not None:
        before_complete(root, complete)
    write_immutable_file(root / COMPLETE_FILENAME, canonical_json_bytes(complete))
    return complete


def validate_manifest_structure(
    manifest: Mapping[str, Any],
    *,
    validate_hash_fields: bool = True,
) -> GaussianCacheSchema:
    manifest_version = int(manifest.get("manifest_version", -1))
    if manifest_version not in {1, 2}:
        raise ValueError(f"Unsupported manifest_version={manifest.get('manifest_version')!r}")
    schema = GaussianCacheSchema.from_dict(manifest["schema"])
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Manifest must contain at least one source HDF5 record")
    GaussianCacheBuilder._validate_sources(
        sources,
        validate_hash_fields=validate_hash_fields,
    )
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping) or selection.get("mode") not in {"all", "index"}:
        raise ValueError("Manifest selection.mode must be 'all' or 'index'")
    if not validate_hash_fields:
        selected_key_count = int(selection.get("selected_key_count", -1))
        if selected_key_count <= 0:
            raise ValueError("Manifest selection.selected_key_count must be positive")
        if selection.get("mode") == "index":
            normalize_source_path(str(selection.get("index_filename", "")))
    producer = manifest.get("producer")
    if producer is not None and not isinstance(producer, Mapping):
        raise TypeError("Manifest producer provenance must be a mapping")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Manifest must contain at least one shard")
    parts = manifest.get("parts")
    is_merged = parts is not None
    if is_merged != (manifest_version == 2):
        raise ValueError("manifest_version=2 is reserved for merged part caches")
    if is_merged:
        if not isinstance(parts, list) or not parts:
            raise ValueError("Merged manifest parts must be a non-empty list")
        part_indices = [int(record["partition_index"]) for record in parts]
        if part_indices != list(range(len(parts))):
            raise ValueError("Merged manifest parts must cover ordered contiguous partition indices")
        part_by_index = {int(record["partition_index"]): record for record in parts}
    else:
        part_by_index = {}
    shard_by_id: dict[str, Mapping[str, Any]] = {}
    shard_groups: dict[int, list[Mapping[str, Any]]] = {}
    for record in shards:
        shard_id = str(record["id"])
        if shard_id in shard_by_id:
            raise ValueError(f"Duplicate shard id {shard_id}")
        shard_by_id[shard_id] = record
        path = normalize_source_path(str(record["path"]))
        if is_merged:
            part_index = int(record.get("part_index", -1))
            expected_prefix = f"parts/part-{part_index:05d}/shards/"
            if part_index not in part_indices or not path.startswith(expected_prefix):
                raise ValueError(
                    "Merged shard path/part_index mismatch: "
                    f"part_index={part_index} path={path}"
                )
        else:
            part_index = 0
            if not path.startswith("shards/"):
                raise ValueError(f"Shard path must be under shards/: {path}")
        shard_groups.setdefault(part_index, []).append(record)
        frames = int(record["frames"])
        size = int(record["bytes"])
        if frames <= 0 or size != frames * schema.frame_bytes:
            raise ValueError(f"Invalid shard frame/byte accounting: {record}")
        if size > MAX_SHARD_BYTES:
            raise ValueError(f"Shard exceeds 4 GiB: {record}")
        if not bool(record.get("final")) and size < MIN_SHARD_BYTES:
            raise ValueError(f"Non-final shard is smaller than 1 GiB: {record}")
        if record.get("immutable") is not True:
            raise ValueError(f"Shard lacks immutable provenance: {record}")
        if validate_hash_fields and not _SHA256_RE.fullmatch(str(record["sha256"])):
            raise ValueError(f"Shard lacks checksum provenance: {record}")
    for part_index, records in sorted(shard_groups.items()):
        final_positions = [index for index, record in enumerate(records) if bool(record.get("final"))]
        if final_positions != [len(records) - 1]:
            raise ValueError(
                f"Exactly the last shard in part {part_index} must be marked final"
            )
    if is_merged and sorted(shard_groups) != part_indices:
        raise ValueError("Every merged part must contribute at least one shard")
    if is_merged:
        for part_index, records in shard_groups.items():
            if len(records) != int(part_by_index[part_index]["shard_count"]):
                raise ValueError(f"Merged part {part_index} shard_count mismatch")

    streams = manifest.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("Manifest must contain at least one stream")
    stream_keys: set[tuple[str, str, str]] = set()
    stream_total = 0
    stream_counts = {index: 0 for index in part_by_index}
    stream_frames = {index: 0 for index in part_by_index}
    for stream in streams:
        key = (
            normalize_source_path(str(stream["source_path"])),
            str(stream["trajectory"]),
            str(stream["agent_name"]),
        )
        if key in stream_keys:
            raise ValueError(f"Duplicate stream {key}")
        stream_keys.add(key)
        if is_merged:
            stream_part_index = int(stream.get("part_index", -1))
            if stream_part_index not in part_by_index:
                raise ValueError(f"Merged stream lacks a valid part_index: {stream}")
            stream_counts[stream_part_index] += 1
        observation_count = int(stream["observation_count"])
        stored_count = int(stream["stored_count"])
        segments = stream.get("segments")
        if observation_count <= 0 or stored_count <= 0 or not isinstance(segments, list):
            raise ValueError(f"Invalid stream metadata: {stream}")
        covered = 0
        previous_timestep = -1
        for segment in segments:
            shard = shard_by_id.get(str(segment["shard"]))
            if shard is None:
                raise ValueError(f"Segment references unknown shard: {segment}")
            if is_merged and int(shard.get("part_index", -1)) != int(
                stream.get("part_index", -2)
            ):
                raise ValueError(f"Merged stream references a shard from another part: {segment}")
            offset = int(segment["offset"])
            count = int(segment["count"])
            source_start = int(segment["source_start"])
            stride = int(segment["source_stride"])
            if offset < 0 or count <= 0 or stride <= 0 or offset + count > int(shard["frames"]):
                raise ValueError(f"Invalid stream segment bounds: {segment}")
            source_end = source_start + (count - 1) * stride
            if source_start <= previous_timestep or source_end >= observation_count:
                raise ValueError(f"Invalid/non-monotonic source timestep segment: {segment}")
            previous_timestep = source_end
            covered += count
        if covered != stored_count:
            raise ValueError(f"Stream stored_count/segment mismatch for {key}")
        stream_total += stored_count
        if is_merged:
            stream_frames[stream_part_index] += stored_count
    if stream_total != int(manifest.get("total_frames", -1)):
        raise ValueError("Manifest total_frames does not equal stream total")
    if stream_total != sum(int(record["frames"]) for record in shards):
        raise ValueError("Manifest total_frames does not equal shard total")
    if is_merged:
        for part_index, part in part_by_index.items():
            if stream_counts[part_index] != int(part["stream_count"]):
                raise ValueError(f"Merged part {part_index} stream_count mismatch")
            if stream_frames[part_index] != int(part["total_frames"]):
                raise ValueError(f"Merged part {part_index} total_frames mismatch")
    return schema


def _open_regular_file(path: str | Path) -> int:
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise RuntimeError(f"Gaussian cache path must not be a symlink: {source}") from error
        raise
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise RuntimeError(f"Gaussian cache path is not a regular file: {source}")
    return descriptor


def regular_file_stat(path: str | Path) -> dict[str, int]:
    """Return the non-digest contract for one regular, non-symlink file."""

    descriptor = _open_regular_file(path)
    try:
        metadata = os.fstat(descriptor)
        return {
            "bytes": int(metadata.st_size),
            "mtime_ns": int(metadata.st_mtime_ns),
        }
    finally:
        os.close(descriptor)


def read_regular_file_snapshot(path: str | Path) -> tuple[bytes, dict[str, int]]:
    """Read one stable regular file without following a final-component symlink."""

    source = Path(path)
    descriptor = _open_regular_file(source)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise RuntimeError(f"Gaussian cache file changed while being read: {source}")
        if len(payload) != after.st_size:
            raise RuntimeError(f"Gaussian cache file changed while being read: {source}")
        return payload, {
            "bytes": int(after.st_size),
            "mtime_ns": int(after.st_mtime_ns),
        }
    finally:
        os.close(descriptor)


def load_manifest(
    cache_root: str | Path,
    *,
    require_complete: bool = True,
    provenance_mode: str = "sha256",
) -> dict[str, Any]:
    root = Path(cache_root)
    manifest_path = root / MANIFEST_FILENAME
    complete_path = root / COMPLETE_FILENAME
    provenance_mode = str(provenance_mode).strip().lower()
    if provenance_mode not in {"sha256", "stat_cmp"}:
        raise ValueError(
            "provenance_mode must be 'sha256' or 'stat_cmp', "
            f"got {provenance_mode!r}"
        )
    if provenance_mode == "stat_cmp":
        try:
            manifest_payload, _ = read_regular_file_snapshot(manifest_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Gaussian cache manifest is missing: {manifest_path}") from None
    else:
        if require_complete and not complete_path.is_file():
            raise FileNotFoundError(f"Gaussian cache is incomplete: missing {complete_path}")
        manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    validate_manifest_structure(
        manifest,
        validate_hash_fields=provenance_mode == "sha256",
    )
    if require_complete:
        if provenance_mode == "stat_cmp":
            try:
                complete_payload, _ = read_regular_file_snapshot(complete_path)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Gaussian cache is incomplete: missing {complete_path}"
                ) from None
            complete = json.loads(complete_payload)
        else:
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("complete") is not True:
            raise ValueError(f"Invalid Gaussian cache COMPLETE marker: {complete_path}")
        if complete.get("schema_name") != SCHEMA_NAME or complete.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("COMPLETE marker schema does not match canonical cache v1")
        if provenance_mode == "sha256":
            actual_sha256 = hashlib.sha256(manifest_payload).hexdigest()
            if complete.get("manifest_sha256") != actual_sha256:
                raise ValueError("COMPLETE marker manifest checksum mismatch")
        elif complete.get("manifest") != MANIFEST_FILENAME:
            raise ValueError("COMPLETE marker manifest path mismatch")
        if int(complete.get("manifest_bytes", -1)) != len(manifest_payload):
            raise ValueError("COMPLETE marker manifest byte count mismatch")
        if int(complete.get("total_frames", -1)) != int(manifest["total_frames"]):
            raise ValueError("COMPLETE marker total frame count mismatch")
        if int(complete.get("shard_count", -1)) != len(manifest["shards"]):
            raise ValueError("COMPLETE marker shard count mismatch")
        if int(complete.get("manifest_version", -1)) != int(manifest["manifest_version"]):
            raise ValueError("COMPLETE marker manifest version mismatch")
    return manifest
