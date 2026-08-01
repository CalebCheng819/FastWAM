#!/usr/bin/env bash

# Build two independently pinned node-local bundles and expose one training
# mapping: CPFS supplies dataset/checkpoint/VAE/stats/text; OSS supplies the
# compact Gaussian cache. The separately pinned eRDMA helper stages its own
# small versioned OSS bundle into a content-addressed local runtime.

fastwam_prepare_multi_source_cache() {
  local script_dir
  local common_root="${FASTWAM_LOCAL_CACHE_ROOT:-/tmp/fastwam-whole-file-cache}"
  local cpfs_source="${FASTWAM_CPFS_BUNDLE_SOURCE_ROOT:?FASTWAM_CPFS_BUNDLE_SOURCE_ROOT is required}"
  local cpfs_manifest="${FASTWAM_CPFS_BUNDLE_MANIFEST:?FASTWAM_CPFS_BUNDLE_MANIFEST is required}"
  local cpfs_expected="${FASTWAM_CPFS_BUNDLE_MANIFEST_SHA256:?FASTWAM_CPFS_BUNDLE_MANIFEST_SHA256 is required}"
  local oss_source="${FASTWAM_OSS_BUNDLE_SOURCE_ROOT:?FASTWAM_OSS_BUNDLE_SOURCE_ROOT is required}"
  local oss_manifest="${FASTWAM_OSS_BUNDLE_MANIFEST:?FASTWAM_OSS_BUNDLE_MANIFEST is required}"
  local oss_expected="${FASTWAM_OSS_BUNDLE_MANIFEST_SHA256:?FASTWAM_OSS_BUNDLE_MANIFEST_SHA256 is required}"
  local saved_gaussian="${FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT:-}"
  local saved_checkpoint="${FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH:-}"
  local saved_dataset="${FASTWAM_LOCAL_DATASET_RELATIVE_ROOT:-}"
  local saved_stats="${FASTWAM_LOCAL_STATS_RELATIVE_PATH:-}"
  local saved_text="${FASTWAM_LOCAL_TEXT_EMBEDS_RELATIVE_ROOT:-}"
  local saved_model_cache="${FASTWAM_LOCAL_MODEL_CACHE_RELATIVE_ROOT:-}"
  local saved_vae="${FASTWAM_LOCAL_VAE_RELATIVE_PATH:-}"

  if [[ "${common_root}" != /tmp/* ]]; then
    echo "Error: multi-source node-local cache root must be a specific /tmp path." >&2
    return 1
  fi
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || return 1
  source "${script_dir}/dlc_local_cache.sh"

  export FASTWAM_LOCAL_CACHE_SOURCE_ROOT="${cpfs_source}"
  export FASTWAM_LOCAL_CACHE_MANIFEST="${cpfs_manifest}"
  export FASTWAM_LOCAL_CACHE_EXPECTED_MANIFEST_SHA256="${cpfs_expected}"
  export FASTWAM_LOCAL_CACHE_ROOT="${common_root%/}/cpfs"
  unset FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT FASTWAM_LOCAL_ERDMA_RELATIVE_ROOT
  export FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH="${saved_checkpoint}"
  export FASTWAM_LOCAL_DATASET_RELATIVE_ROOT="${saved_dataset}"
  export FASTWAM_LOCAL_STATS_RELATIVE_PATH="${saved_stats}"
  export FASTWAM_LOCAL_TEXT_EMBEDS_RELATIVE_ROOT="${saved_text}"
  export FASTWAM_LOCAL_MODEL_CACHE_RELATIVE_ROOT="${saved_model_cache}"
  export FASTWAM_LOCAL_VAE_RELATIVE_PATH="${saved_vae}"
  fastwam_prepare_local_cache || return $?
  export FASTWAM_LOCAL_CPFS_CACHE_DIR="${FASTWAM_LOCAL_CACHE_DIR}"
  export FASTWAM_LOCAL_CPFS_CACHE_MANIFEST_SHA256="${FASTWAM_LOCAL_CACHE_MANIFEST_SHA256}"

  export FASTWAM_LOCAL_CACHE_SOURCE_ROOT="${oss_source}"
  export FASTWAM_LOCAL_CACHE_MANIFEST="${oss_manifest}"
  export FASTWAM_LOCAL_CACHE_EXPECTED_MANIFEST_SHA256="${oss_expected}"
  export FASTWAM_LOCAL_CACHE_ROOT="${common_root%/}/oss"
  unset \
    FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH \
    FASTWAM_LOCAL_DATASET_RELATIVE_ROOT \
    FASTWAM_LOCAL_STATS_RELATIVE_PATH \
    FASTWAM_LOCAL_TEXT_EMBEDS_RELATIVE_ROOT \
    FASTWAM_LOCAL_MODEL_CACHE_RELATIVE_ROOT \
    FASTWAM_LOCAL_VAE_RELATIVE_PATH
  export FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT="${saved_gaussian}"
  unset FASTWAM_LOCAL_ERDMA_RELATIVE_ROOT
  fastwam_prepare_local_cache || return $?
  export FASTWAM_LOCAL_OSS_CACHE_DIR="${FASTWAM_LOCAL_CACHE_DIR}"
  export FASTWAM_LOCAL_OSS_CACHE_MANIFEST_SHA256="${FASTWAM_LOCAL_CACHE_MANIFEST_SHA256}"

  # Restore the declarative mapping variables for provenance/logging callers.
  export FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH="${saved_checkpoint}"
  export FASTWAM_LOCAL_DATASET_RELATIVE_ROOT="${saved_dataset}"
  export FASTWAM_LOCAL_STATS_RELATIVE_PATH="${saved_stats}"
  export FASTWAM_LOCAL_TEXT_EMBEDS_RELATIVE_ROOT="${saved_text}"
  export FASTWAM_LOCAL_MODEL_CACHE_RELATIVE_ROOT="${saved_model_cache}"
  export FASTWAM_LOCAL_VAE_RELATIVE_PATH="${saved_vae}"
  export FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT="${saved_gaussian}"
  printf '[local_cache] status=READY mode=multi_source cpfs_sha256=%s oss_sha256=%s\n' \
    "${FASTWAM_LOCAL_CPFS_CACHE_MANIFEST_SHA256}" \
    "${FASTWAM_LOCAL_OSS_CACHE_MANIFEST_SHA256}" >&2
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  fastwam_prepare_multi_source_cache
fi
