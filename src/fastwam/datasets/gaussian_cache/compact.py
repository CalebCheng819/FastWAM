"""Opacity-aware moment matching for the 28x40 active Gaussian cache."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import torch

from .manifest import (
    DEFAULT_TARGET_SHARD_BYTES,
    MANIFEST_FILENAME,
    GaussianCacheBuilder,
    sha256_file,
)
from .provider import GaussianCache
from .schema import (
    FrameKey,
    GaussianCacheSchema,
    pack_gaussian_channels,
    unpack_gaussian_channels,
)
from .selection import load_selection_jsonl, write_normalized_selection_index

COMPACT_HEIGHT = 28
COMPACT_WIDTH = 40
MOMENT_MATCH_METHOD = "opacity-aware-moment-matching-cell-mean-alpha-v2"


def _cell_index(
    height: int,
    width: int,
    output_height: int,
    output_width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    rows = torch.div(
        torch.arange(height, device=device) * output_height,
        height,
        rounding_mode="floor",
    )
    columns = torch.div(
        torch.arange(width, device=device) * output_width,
        width,
        rounding_mode="floor",
    )
    return (rows[:, None] * output_width + columns[None, :]).reshape(-1).long()


def opacity_aware_moment_match(
    gaussian: torch.Tensor,
    *,
    output_size: tuple[int, int] = (COMPACT_HEIGHT, COMPACT_WIDTH),
    eps: float = 1e-8,
) -> torch.Tensor:
    """Merge each spatial cell as an opacity-weighted Gaussian mixture.

    For mixture weights ``w_i = opacity_i`` the compact mean and covariance are
    computed from the first and second moments:

    ``mu = sum(w_i mu_i) / sum(w_i)``

    ``Sigma = sum(w_i (Sigma_i + mu_i mu_i^T))/sum(w_i) - mu mu^T``

    Compact opacity is the area-normalized cell density ``mean(opacity_i)``.
    This deliberately does not use alpha union: union saturates toward one as
    cell area grows and therefore changes merely when the input resolution or
    pooling geometry changes.  All accumulation is float32; the immutable
    cache result is FP16 in canonical channel order.
    """

    if gaussian.ndim < 3 or gaussian.shape[-3] != 13:
        raise ValueError(f"gaussian must be [...,13,H,W], got {tuple(gaussian.shape)}")
    output_height, output_width = map(int, output_size)
    height, width = map(int, gaussian.shape[-2:])
    if not (0 < output_height <= height and 0 < output_width <= width):
        raise ValueError(
            f"output_size must not upsample and must be positive, got {output_size} from {(height, width)}"
        )

    prefix = tuple(gaussian.shape[:-3])
    flat_count = 1
    for dimension in prefix:
        flat_count *= int(dimension)
    values = gaussian.reshape(flat_count, 13, height, width).float()
    means, covariance, opacity = unpack_gaussian_channels(values)
    pixel_count = height * width
    cell_count = output_height * output_width
    cells = _cell_index(
        height,
        width,
        output_height,
        output_width,
        device=values.device,
    )

    means = means.reshape(flat_count, 3, pixel_count)
    covariance = covariance.reshape(flat_count, 3, 3, pixel_count)
    # Numerical asymmetry in a teacher prediction should not survive canonical
    # moment matching.
    covariance = 0.5 * (covariance + covariance.transpose(1, 2))
    alpha = opacity.reshape(flat_count, 1, pixel_count).clamp(0.0, 1.0)
    cell_for_scalar = cells.view(1, 1, pixel_count).expand(flat_count, 1, -1)
    cell_for_mean = cells.view(1, 1, pixel_count).expand(flat_count, 3, -1)
    cell_for_matrix = cells.view(1, 1, pixel_count).expand(flat_count, 9, -1)

    weight = torch.zeros((flat_count, 1, cell_count), device=values.device)
    weight.scatter_add_(2, cell_for_scalar, alpha)
    weighted_mean = torch.zeros((flat_count, 3, cell_count), device=values.device)
    weighted_mean.scatter_add_(2, cell_for_mean, alpha * means)
    safe_weight = weight.clamp_min(float(eps))
    compact_mean = weighted_mean / safe_weight

    outer = means.unsqueeze(2) * means.unsqueeze(1)
    second_moment = covariance + outer
    weighted_second = torch.zeros((flat_count, 9, cell_count), device=values.device)
    weighted_second.scatter_add_(
        2,
        cell_for_matrix,
        (alpha.unsqueeze(2) * second_moment).reshape(flat_count, 9, pixel_count),
    )
    compact_second = (weighted_second / safe_weight).reshape(flat_count, 3, 3, cell_count)
    compact_outer = compact_mean.unsqueeze(2) * compact_mean.unsqueeze(1)
    compact_covariance = compact_second - compact_outer
    compact_covariance = 0.5 * (
        compact_covariance + compact_covariance.transpose(1, 2)
    )

    valid = weight > float(eps)
    compact_mean = torch.where(valid.expand(-1, 3, -1), compact_mean, 0.0)
    compact_covariance = torch.where(
        valid.unsqueeze(2).expand(-1, 3, 3, -1),
        compact_covariance,
        0.0,
    )

    # Cell sizes are not necessarily uniform (240 -> 28 and 320 -> 40), so use
    # an explicit pixel count instead of dividing by a nominal pooling area.
    # Mean alpha is an area-normalized density and cannot saturate simply
    # because a cell contains more pixels.
    cell_pixels = torch.zeros((flat_count, 1, cell_count), device=values.device)
    cell_pixels.scatter_add_(2, cell_for_scalar, torch.ones_like(alpha))
    compact_opacity = (weight / cell_pixels.clamp_min(1.0)).clamp(0.0, 1.0)

    compact_mean = compact_mean.reshape(flat_count, 3, output_height, output_width)
    compact_covariance = compact_covariance.reshape(
        flat_count, 3, 3, output_height, output_width
    )
    compact_opacity = compact_opacity.reshape(flat_count, 1, output_height, output_width)
    packed = pack_gaussian_channels(compact_mean, compact_covariance, compact_opacity)
    if not bool(torch.isfinite(packed).all().item()):
        raise OverflowError("Compact Gaussian moment matching produced non-finite FP32 values")
    max_abs = float(packed.detach().abs().max().item())
    fp16_max = float(torch.finfo(torch.float16).max)
    if max_abs > fp16_max:
        raise OverflowError(
            "Compact Gaussian moments exceed the lossless finite FP16 storage range: "
            f"max_abs_float32={max_abs} fp16_max={fp16_max}"
        )
    compact = packed.reshape(*prefix, 13, output_height, output_width).to(torch.float16)
    if not bool(torch.isfinite(compact).all().item()):
        raise OverflowError("Compact Gaussian FP16 cast produced non-finite values")
    return compact


def _group_keys(keys: Iterable[FrameKey]) -> dict[tuple[str, str, str], list[int]]:
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for key in keys:
        grouped[(key.source_path, key.trajectory, key.agent_name)].append(key.timestep)
    for stream, timesteps in grouped.items():
        ordered = sorted(set(timesteps))
        if len(ordered) != len(timesteps):
            raise ValueError(f"Duplicate compact projection key in stream {stream}")
        grouped[stream] = ordered
    return dict(grouped)


def _manifest_stream_timesteps(record: dict) -> Iterable[int]:
    for segment in record["segments"]:
        start = int(segment["source_start"])
        stride = int(segment["source_stride"])
        for index in range(int(segment["count"])):
            yield start + index * stride


def project_compact_cache(
    canonical_root: str | Path,
    output_root: str | Path,
    *,
    selection: str = "index",
    selection_jsonl: str | Path | None = None,
    selection_keys: Sequence[FrameKey] | None = None,
    verify: str = "manifest",
    batch_size: int = 8,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    staging_dir: str | Path | None = None,
    verify_uploaded_checksum: bool = True,
    partition: Mapping[str, object] | None = None,
    preserve_parent_teacher: bool = False,
    derivation: Mapping[str, object] | None = None,
    producer: Mapping[str, object] | None = None,
    selection_plan_identity: Mapping[str, object] | None = None,
) -> dict:
    """Derive an immutable 28x40 cache, optionally only for selected frame keys."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    canonical = GaussianCache.open(canonical_root, verify=verify)
    if canonical.schema.cache_kind != "canonical":
        raise ValueError("Compact projection requires a canonical parent cache")
    if selection_jsonl is not None and selection_keys is not None:
        raise ValueError("Supply selection_jsonl or selection_keys, not both")
    if selection == "all":
        if selection_jsonl is not None or selection_keys is not None:
            raise ValueError("Sparse selection inputs are only valid with selection='index'")
        keys = None
        grouped = None
        selected_key_count = int(canonical.manifest["total_frames"])
    elif selection == "index" and (selection_jsonl is not None or selection_keys is not None):
        keys = (
            load_selection_jsonl(selection_jsonl)
            if selection_jsonl is not None
            else list(selection_keys or ())
        )
        if not keys:
            raise ValueError("selection_keys must be non-empty")
        if len(set(keys)) != len(keys):
            raise ValueError("selection_keys contains duplicate frame identities")
        canonical.preflight_keys(keys)
        grouped = _group_keys(keys)
        selected_key_count = len(keys)
    else:
        raise ValueError(
            "selection must be 'all', or 'index' with selection_jsonl/selection_keys"
        )
    parent_streams = {
        (
            str(record["source_path"]),
            str(record["trajectory"]),
            str(record["agent_name"]),
        ): record
        for record in canonical.manifest["streams"]
    }
    output = Path(output_root)
    schema = GaussianCacheSchema(
        height=COMPACT_HEIGHT,
        width=COMPACT_WIDTH,
        cache_kind="compact",
    )
    # Builder owns creation of output_root.  Persist the normalized sparse index
    # after it exists, then replace its in-memory selection record before seal.
    if preserve_parent_teacher and partition is None:
        raise ValueError("preserve_parent_teacher requires partition metadata")
    default_derivation = {
        "method": MOMENT_MATCH_METHOD,
        "output_size": [COMPACT_HEIGHT, COMPACT_WIDTH],
        "parent_manifest_sha256": sha256_file(Path(canonical_root) / MANIFEST_FILENAME),
    }
    builder = GaussianCacheBuilder(
        output,
        schema,
        sources=canonical.manifest["sources"],
        teacher=(
            canonical.manifest["teacher"]
            if preserve_parent_teacher
            else {
                "kind": "derived-canonical-cache",
                "parent_teacher": canonical.manifest["teacher"],
            }
        ),
        producer=(
            canonical.manifest.get("producer")
            if producer is None
            else producer
        ),
        selection={"mode": selection, "selected_key_count": selected_key_count},
        derivation=dict(default_derivation if derivation is None else derivation),
        partition=partition,
        target_shard_bytes=target_shard_bytes,
        staging_dir=staging_dir,
        verify_uploaded_checksum=verify_uploaded_checksum,
    )
    if selection == "index":
        assert keys is not None
        builder.selection = write_normalized_selection_index(output, keys)
        if selection_plan_identity is not None:
            builder.selection["plan_identity"] = dict(selection_plan_identity)
    elif selection_plan_identity is not None:
        raise ValueError("selection_plan_identity requires selection='index'")
    try:
        stream_keys = sorted(parent_streams) if grouped is None else sorted(grouped)
        for stream_key in stream_keys:
            record = parent_streams.get(stream_key)
            if record is None:
                raise KeyError(f"Selected stream is absent from canonical cache: {stream_key}")
            source_path, trajectory, agent_name = stream_key
            timesteps = (
                list(_manifest_stream_timesteps(record))
                if grouped is None
                else grouped[stream_key]
            )
            for start in range(0, len(timesteps), int(batch_size)):
                batch_timesteps = timesteps[start : start + int(batch_size)]
                batch = torch.stack(
                    [
                        canonical.get_frame(
                            FrameKey(source_path, trajectory, timestep, agent_name)
                        )
                        for timestep in batch_timesteps
                    ],
                    dim=0,
                )
                compact = opacity_aware_moment_match(batch)
                builder.append_stream(
                    source_path=source_path,
                    trajectory=trajectory,
                    agent_name=agent_name,
                    observation_count=int(record["observation_count"]),
                    timesteps=batch_timesteps,
                    frames=compact,
                )
        return builder.finish()
    except Exception:
        builder.abort()
        raise
    finally:
        canonical.close()
