#!/usr/bin/env python3
"""Fail-closed real-data preflight for the metadata-no-hash N=2 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_DIGEST_ATTEMPTS: list[str] = []


def _forbid_digest_constructors() -> None:
    def make_fail(name: str):
        def fail(*args, **kwargs):
            _DIGEST_ATTEMPTS.append(name)
            raise RuntimeError(
                "metadata-no-hash preflight observed a digest constructor: "
                f"{name}"
            )

        return fail

    for name in (
        "new",
        "file_digest",
        "md5",
        "sha1",
        "sha224",
        "sha256",
        "sha384",
        "sha512",
        "blake2b",
        "blake2s",
        "shake_128",
        "shake_256",
    ):
        if hasattr(hashlib, name):
            setattr(hashlib, name, make_fail(f"hashlib.{name}"))


# Install the guard before importing torch or any FastWAM data module so the
# exercised import+construction+getitem path is covered too.
_forbid_digest_constructors()

import torch

import fastwam.datasets.gaussian_cache.provider as gaussian_provider_module
import fastwam.datasets.robofactory_multi_robot as robofactory_module
from fastwam.datasets.gaussian_cache import FrameKey
from fastwam.datasets.robofactory_multi_robot import RoboFactoryMultiRobotDataset


def _forbid_aliased_digest(name: str):
    def fail(*args, **kwargs):
        _DIGEST_ATTEMPTS.append(name)
        raise RuntimeError(
            "metadata-no-hash preflight observed an aliased digest helper: "
            f"{name}"
        )

    return fail


robofactory_module.sha256_file = _forbid_aliased_digest(
    "robofactory_multi_robot.sha256_file"
)
robofactory_module.gaussian_source_identity_sha256 = _forbid_aliased_digest(
    "robofactory_multi_robot.gaussian_source_identity_sha256"
)
gaussian_provider_module.sha256_file = _forbid_aliased_digest(
    "gaussian_cache.provider.sha256_file"
)


TEXT_CACHE_FILE = (
    "89bc0bd3ed4a9f6192e149614112915dbd94d0b323d714e2da3c89bb68f6e26a"
    ".t5_len128.wan22ti2v5b.pt"
)
INSTRUCTION_MAP = {
    "PlaceFood-rf": "two robots collaboratively place the food in the target location"
}


def _build_dataset(args: argparse.Namespace, *, training: bool):
    return RoboFactoryMultiRobotDataset(
        root_dir=args.data_root,
        num_frames=33,
        action_video_freq_ratio=4,
        load_future_video=False,
        video_size=(224, 320),
        action_dim=8,
        state_dim=18,
        agent_geometry_dim=7,
        window_stride=16 if training else 32,
        val_set_proportion=0.1,
        is_training_set=training,
        split_seed=42,
        randomize_agent_order=training,
        required_agent_counts=(2,),
        required_tasks=("PlaceFood-rf",),
        integrity_mode="metadata_no_hash",
        pretrained_norm_stats=args.stats,
        text_embedding_cache_dir=args.text_cache,
        text_embedding_cache_files={"PlaceFood-rf": TEXT_CACHE_FILE},
        gaussian_cache_dir=args.gaussian_cache,
        gaussian_cache_verify="manifest",
        gaussian_fallback_cache_dir=args.gaussian_fallback_cache,
        gaussian_fallback_projection=(
            "opacity-aware-moment-matching-cell-mean-alpha-v2"
        ),
        gaussian_cache_expected_manifest_sha256=None,
        gaussian_cache_expected_selection_sha256=None,
        gaussian_cache_expected_source_identity_sha256=None,
        gaussian_channels=13,
        gaussian_size=(28, 40),
        require_train_only_stats=True,
        context_len=128,
        instruction_map=INSTRUCTION_MAP,
    )


def _materialize_fallback_sample(dataset, *, split: str) -> dict:
    primary = dataset._get_gaussian_cache()
    fallback = dataset._get_gaussian_fallback_cache()
    candidate_index = None
    candidate_missing_agents: list[str] = []
    for index, entry in enumerate(dataset.entries):
        missing_agents = []
        for agent_name in entry["agent_names"]:
            key = FrameKey(
                entry["source_path"],
                entry["trajectory"],
                int(entry["start"]),
                agent_name,
            )
            if not primary.contains_frame(key):
                if not fallback.contains_frame(key):
                    raise RuntimeError(
                        f"{split} candidate is absent from both Gaussian caches: "
                        f"{key.to_dict()}"
                    )
                missing_agents.append(agent_name)
        if missing_agents:
            candidate_index = index
            candidate_missing_agents = missing_agents
            break

    declared_fallback_keys = int(dataset._gaussian_preflight["fallback_keys"])
    if candidate_index is None:
        if declared_fallback_keys:
            raise RuntimeError(
                f"{split} declared {declared_fallback_keys} fallback keys but no "
                "materializable candidate was found"
            )
        return {
            "status": "not_required",
            "declared_fallback_keys": 0,
            "frames_read": 0,
        }

    entry = dataset.entries[candidate_index]
    reads: list[dict] = []
    original_get_frame = fallback.get_frame

    def observed_get_frame(key):
        coerced = fallback._coerce_key(key)
        location = fallback._locate(coerced)
        if location is None:
            raise RuntimeError(f"Fallback frame vanished before read: {coerced.to_dict()}")
        frame = original_get_frame(coerced)
        reads.append({"key": coerced.to_dict(), "shard": location[0]})
        return frame

    fallback.get_frame = observed_get_frame
    try:
        sample = dataset[candidate_index]
    finally:
        fallback.get_frame = original_get_frame
    gaussian = sample["agent_gaussian"]
    if not reads:
        raise RuntimeError(f"{split} fallback candidate did not read a fallback frame")
    if tuple(gaussian.shape[1:]) != (13, 28, 40):
        raise RuntimeError(
            f"{split} canonical fallback projection returned an unexpected shape: "
            f"{tuple(gaussian.shape)}"
        )
    if not bool(torch.isfinite(gaussian).all().item()):
        raise RuntimeError(
            f"{split} canonical fallback projection returned non-finite values"
        )
    return {
        "status": "materialized",
        "index": candidate_index,
        "source_path": entry["source_path"],
        "trajectory": entry["trajectory"],
        "start": int(entry["start"]),
        "primary_missing_agents": candidate_missing_agents,
        "frames_read": len(reads),
        "reads": reads,
        "agent_ids": sample["agent_ids"].tolist(),
        "agent_gaussian_shape": list(gaussian.shape),
        "agent_gaussian_dtype": str(gaussian.dtype),
        "agent_gaussian_finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--gaussian-cache", type=Path, required=True)
    parser.add_argument("--gaussian-fallback-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if "datasets" in sys.modules:
        raise RuntimeError("Hugging Face datasets was imported on the RoboFactory path")

    train = _build_dataset(args, training=True)
    val = _build_dataset(args, training=False)
    sample = train[0]
    fallback_samples = {
        "train": _materialize_fallback_sample(train, split="train"),
        "val": _materialize_fallback_sample(val, split="val"),
    }
    if fallback_samples["train"]["status"] != "materialized":
        raise RuntimeError("The training split did not exercise canonical fallback")
    if _DIGEST_ATTEMPTS:
        raise RuntimeError(f"Digest attempts were recorded: {_DIGEST_ATTEMPTS}")
    evidence = {
        "status": "PASS",
        "integrity_mode": "metadata_no_hash",
        "known_python_digest_entrypoints_forbidden": True,
        "guarded_python_digest_attempts": len(_DIGEST_ATTEMPTS),
        "huggingface_datasets_imported": "datasets" in sys.modules,
        "train_windows": len(train),
        "val_windows": len(val),
        "train_agent_counts": sorted(set(train.agent_counts)),
        "val_agent_counts": sorted(set(val.agent_counts)),
        "sample_shapes": {
            key: list(value.shape)
            for key, value in sample.items()
            if isinstance(value, torch.Tensor)
        },
        "fallback_samples": fallback_samples,
        "data_root": str(args.data_root.resolve()),
        "stats": str(args.stats.resolve()),
        "text_cache": str(args.text_cache.resolve()),
        "gaussian_cache": str(args.gaussian_cache.resolve()),
        "gaussian_fallback_cache": str(args.gaussian_fallback_cache.resolve()),
        "train_gaussian_preflight": train._gaussian_preflight,
        "val_gaussian_preflight": val._gaussian_preflight,
    }
    rendered = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        target = args.output
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    print(rendered, end="", flush=True)


if __name__ == "__main__":
    main()
