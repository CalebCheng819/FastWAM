"""Canonical immutable Gaussian cache APIs.

The stable dataset-facing contract is ``GaussianCache.get_agents(...)`` which
returns ``{"agent_gaussian": float16[N,13,H,W]}`` in the requested agent order.
"""

from .compact import (
    COMPACT_HEIGHT,
    COMPACT_WIDTH,
    opacity_aware_moment_match,
    project_compact_cache,
)
from .distributed import (
    PARTITION_ALGORITHM,
    merge_part_manifests,
    partition_metadata,
    partition_source_records,
    validate_partition_coverage,
)
from .manifest import GaussianCacheBuilder, load_manifest, sha256_file, source_record
from .provider import GaussianCache, MissingGaussianFramesError
from .schema import (
    CANONICAL_CHANNELS,
    COMPLETE_FILENAME,
    MANIFEST_FILENAME,
    FrameKey,
    GaussianCacheSchema,
    correct_policy_lightning_legacy_covariance_order,
    pack_gaussian_channels,
    unpack_gaussian_channels,
)
from .selection import load_selection_jsonl
from .transaction import (
    UnsafeCacheRestartError,
    prepare_cache_build,
    run_paired_micro_part,
    verify_complete_cache,
)

__all__ = [
    "CANONICAL_CHANNELS",
    "COMPACT_HEIGHT",
    "COMPACT_WIDTH",
    "COMPLETE_FILENAME",
    "MANIFEST_FILENAME",
    "PARTITION_ALGORITHM",
    "FrameKey",
    "GaussianCache",
    "GaussianCacheBuilder",
    "GaussianCacheSchema",
    "MissingGaussianFramesError",
    "UnsafeCacheRestartError",
    "correct_policy_lightning_legacy_covariance_order",
    "load_manifest",
    "load_selection_jsonl",
    "merge_part_manifests",
    "opacity_aware_moment_match",
    "pack_gaussian_channels",
    "partition_metadata",
    "partition_source_records",
    "prepare_cache_build",
    "project_compact_cache",
    "run_paired_micro_part",
    "sha256_file",
    "source_record",
    "unpack_gaussian_channels",
    "validate_partition_coverage",
    "verify_complete_cache",
]
