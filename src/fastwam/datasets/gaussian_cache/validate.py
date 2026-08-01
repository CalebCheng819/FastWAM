"""Fail-closed validation for canonical Gaussian cache manifests and shards."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .distributed import validate_partition_coverage
from .manifest import load_manifest, sha256_file
from .provider import GaussianCache
from .schema import FrameKey, GaussianCacheSchema, unpack_gaussian_channels
from .selection import load_selection_jsonl

SEMANTIC_MODES = {"none", "sample", "coverage"}


def _semantic_coverage_keys(manifest: dict[str, Any]) -> list[FrameKey]:
    """Return deterministic first/middle/last keys for every immutable shard."""

    requested: dict[str, set[int]] = {}
    for shard in manifest["shards"]:
        frame_count = int(shard["frames"])
        requested[str(shard["id"])] = {0, (frame_count - 1) // 2, frame_count - 1}
    matches: dict[tuple[str, int], list[FrameKey]] = {
        (shard_id, offset): []
        for shard_id, offsets in requested.items()
        for offset in offsets
    }
    for stream in manifest["streams"]:
        for segment in stream["segments"]:
            shard_id = str(segment["shard"])
            start = int(segment["offset"])
            count = int(segment["count"])
            for offset in requested.get(shard_id, set()):
                if start <= offset < start + count:
                    matches[(shard_id, offset)].append(
                        FrameKey(
                            str(stream["source_path"]),
                            str(stream["trajectory"]),
                            int(segment["source_start"])
                            + (offset - start) * int(segment["source_stride"]),
                            str(stream["agent_name"]),
                        )
                    )
    invalid = {
        key: values for key, values in matches.items() if len(values) != 1
    }
    if invalid:
        sample = {
            f"{shard_id}:{offset}": [value.to_dict() for value in values]
            for (shard_id, offset), values in sorted(invalid.items())[:16]
        }
        raise ValueError(
            "Semantic coverage anchors must map to exactly one frame; "
            f"sample={sample}"
        )
    return [matches[key][0] for key in sorted(matches)]


def _validate_semantic_frame(
    cache: GaussianCache,
    schema: GaussianCacheSchema,
    key: FrameKey,
) -> None:
    frame = cache.get_frame(key).float()
    if tuple(frame.shape) != schema.frame_shape or not bool(torch.isfinite(frame).all().item()):
        raise ValueError(f"Invalid/non-finite Gaussian frame at {key.to_dict()}")
    _, covariance, opacity = unpack_gaussian_channels(frame)
    if bool(((opacity < 0.0) | (opacity > 1.0)).any().item()):
        raise ValueError(f"Opacity outside [0,1] at {key.to_dict()}")
    asymmetry = (covariance - covariance.transpose(-4, -3)).abs().max()
    if float(asymmetry) > 5e-3:
        raise ValueError(
            f"Covariance is not symmetric at {key.to_dict()}: max_error={float(asymmetry)}"
        )


def validate_cache(
    cache_root: str | Path,
    *,
    verify_shard_checksums: bool = False,
    source_root: str | Path | None = None,
    verify_source_checksums: bool = False,
    semantic_sample_frames: int = 0,
    semantic_mode: str | None = None,
) -> dict[str, Any]:
    root = Path(cache_root).expanduser().resolve()
    manifest = load_manifest(root, require_complete=True)
    schema = GaussianCacheSchema.from_dict(manifest["schema"])
    validated_partition_parts = len(manifest.get("parts", []))
    for part in manifest.get("parts", []):
        part_root = root / str(part["path"])
        part_manifest = load_manifest(part_root, require_complete=True)
        manifest_path = part_root / "manifest.json"
        if sha256_file(manifest_path) != str(part["manifest_sha256"]):
            raise ValueError(f"Merged child manifest checksum mismatch: {manifest_path}")
        if len(part_manifest["sources"]) != int(part["source_count"]):
            raise ValueError(f"Merged child source count mismatch: {part_root}")
        if len(part_manifest["shards"]) != int(part["shard_count"]):
            raise ValueError(f"Merged child shard count mismatch: {part_root}")
        if len(part_manifest["streams"]) != int(part["stream_count"]):
            raise ValueError(f"Merged child stream count mismatch: {part_root}")
        if int(part_manifest["total_frames"]) != int(part["total_frames"]):
            raise ValueError(f"Merged child frame count mismatch: {part_root}")
        part_selection_keys = None
        if part_manifest["selection"]["mode"] == "index":
            part_selection_path = part_root / str(
                part_manifest["selection"]["index_filename"]
            )
            if sha256_file(part_selection_path) != str(
                part_manifest["selection"]["index_sha256"]
            ):
                raise ValueError(
                    f"Merged child selection checksum mismatch: {part_selection_path}"
                )
            part_selection_keys = load_selection_jsonl(part_selection_path)
        validate_partition_coverage(
            part_manifest,
            selection_keys=part_selection_keys,
            context=f"merged child part {part['partition_index']}",
        )
    if not manifest.get("parts") and isinstance(manifest.get("partition"), dict):
        direct_selection_keys = None
        if manifest["selection"]["mode"] == "index":
            direct_selection_keys = load_selection_jsonl(
                root / str(manifest["selection"]["index_filename"])
            )
        validate_partition_coverage(
            manifest,
            selection_keys=direct_selection_keys,
            context="standalone part cache",
        )
        validated_partition_parts = 1
    cache = GaussianCache.open(
        root,
        verify="checksums" if verify_shard_checksums else "manifest",
    )

    source_by_path = {str(record["path"]): record for record in manifest["sources"]}
    stream_sources = {str(record["source_path"]) for record in manifest["streams"]}
    if not stream_sources <= set(source_by_path):
        raise ValueError(
            f"Manifest streams reference unprovenanced source files: {sorted(stream_sources-set(source_by_path))}"
        )
    if verify_source_checksums:
        if source_root is None:
            raise ValueError("verify_source_checksums=True requires source_root")
        source_root_path = Path(source_root).expanduser().resolve()
        for relative, record in source_by_path.items():
            path = source_root_path / relative
            if not path.is_file():
                raise FileNotFoundError(f"Source HDF5 is missing: {path}")
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"Source HDF5 byte count mismatch: {path}")
            if sha256_file(path) != str(record["sha256"]):
                raise ValueError(f"Source HDF5 SHA-256 mismatch: {path}")

    selection = manifest["selection"]
    if selection["mode"] == "all":
        incomplete = [
            (
                record["source_path"],
                record["trajectory"],
                record["agent_name"],
                record["stored_count"],
                record["observation_count"],
            )
            for record in manifest["streams"]
            if int(record["stored_count"]) != int(record["observation_count"])
        ]
        if incomplete:
            raise ValueError(
                "selection=all cache does not cover every observation timestep; "
                f"sample={incomplete[:16]}"
            )
    else:
        index_path = root / str(selection["index_filename"])
        if not index_path.is_file():
            raise FileNotFoundError(f"Sparse cache selection index is missing: {index_path}")
        if sha256_file(index_path) != str(selection["index_sha256"]):
            raise ValueError(f"Sparse cache selection index checksum mismatch: {index_path}")
        keys = load_selection_jsonl(index_path)
        if len(keys) != int(selection["selected_key_count"]):
            raise ValueError("Sparse cache selection index count does not match manifest")
        cache.preflight_keys(keys)

    requested_semantic_mode = semantic_mode
    if requested_semantic_mode is None:
        requested_semantic_mode = (
            "sample" if int(semantic_sample_frames) > 0 else "coverage"
        )
    requested_semantic_mode = str(requested_semantic_mode).lower()
    if requested_semantic_mode not in SEMANTIC_MODES:
        raise ValueError(
            f"semantic_mode must be one of {sorted(SEMANTIC_MODES)}, "
            f"got {requested_semantic_mode!r}"
        )
    if requested_semantic_mode == "sample" and int(semantic_sample_frames) <= 0:
        raise ValueError("semantic_mode='sample' requires semantic_sample_frames > 0")

    if requested_semantic_mode == "none":
        semantic_keys: list[FrameKey] = []
    elif requested_semantic_mode == "sample":
        semantic_keys = []
        for key in cache.iter_keys():
            if len(semantic_keys) >= int(semantic_sample_frames):
                break
            semantic_keys.append(key)
    else:
        semantic_keys = _semantic_coverage_keys(manifest)
    try:
        for key in semantic_keys:
            _validate_semantic_frame(cache, schema, key)
    finally:
        cache.close()

    covered_parts = {
        int(record.get("part_index", 0))
        for record in manifest["shards"]
    } if semantic_keys else set()
    return {
        "cache_root": str(root),
        "cache_kind": schema.cache_kind,
        "shape": list(schema.frame_shape),
        "total_frames": int(manifest["total_frames"]),
        "streams": len(manifest["streams"]),
        "shards": len(manifest["shards"]),
        "parts": len(manifest.get("parts", [])),
        "selection": str(selection["mode"]),
        "shard_checksums_verified": bool(verify_shard_checksums),
        "source_checksums_verified": bool(verify_source_checksums),
        "partition_parts_verified": validated_partition_parts,
        "semantic_mode": requested_semantic_mode,
        "semantic_frames_verified": len(semantic_keys),
        "semantic_shards_covered": (
            len(manifest["shards"]) if requested_semantic_mode == "coverage" else 0
        ),
        "semantic_parts_covered": (
            len(covered_parts) if requested_semantic_mode == "coverage" else 0
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--checksums", action="store_true", help="hash every immutable shard")
    parser.add_argument("--source-root")
    parser.add_argument("--source-checksums", action="store_true")
    parser.add_argument(
        "--semantic-mode",
        choices=sorted(SEMANTIC_MODES),
        default="coverage",
        help="formal default covers first/middle/last frame of every shard",
    )
    parser.add_argument(
        "--semantic-sample-frames",
        type=int,
        default=16,
        help="used only with --semantic-mode sample",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = validate_cache(
        args.cache_root,
        verify_shard_checksums=args.checksums,
        source_root=args.source_root,
        verify_source_checksums=args.source_checksums,
        semantic_sample_frames=args.semantic_sample_frames,
        semantic_mode=args.semantic_mode,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
