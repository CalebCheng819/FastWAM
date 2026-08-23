#!/usr/bin/env python3
"""Build a deterministic held-out RoboFactory evaluation panel.

The training split manifest contains one record per selected timestep.  This
script first collapses it to unique trajectories, then ranks trajectories with
a task-qualified SHA-256 key.  It never chooses a training trajectory and it
records the immutable source-H5 identity from the sealed input-bundle
manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import h5py


SCHEMA_VERSION = "fastwam-robofactory-heldout-panel-v1"
SELECTION_RULE = "sha256-ranked-unique-val-trajectory-v1"
TRAJECTORY_PATTERN = re.compile(r"^traj_(\d+)$")
TASK_AGENT_COUNTS = {
    "PlaceFood-rf": 2,
    "PlaceCubeInCup-rf": 2,
    "StrikeCubeHard-rf": 2,
    "ThreeRobotsPlaceShoes-rf": 3,
    "ThreeRobotsStackCube-rf": 3,
    "FourRobotsStackCube-rf": 4,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _agent_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("-", 1)[-1]), name
    except ValueError:
        return 1 << 30, name


def _safe_source(dataset_root: Path, relative_path: str) -> Path:
    candidate = (dataset_root / relative_path).resolve(strict=True)
    try:
        candidate.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(
            f"Source path escapes dataset root: {relative_path!r}"
        ) from error
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"Source H5 must be a regular non-symlink file: {candidate}")
    return candidate


def _load_bundle_h5_hashes(manifest_path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line:
            continue
        digest, separator, relative = raw_line.partition("  ")
        if not separator or len(digest) != 64:
            raise ValueError(
                f"Malformed bundle manifest line {line_number}: {raw_line!r}"
            )
        if relative.startswith(
            "datasets/robofactory_multi_robot/"
        ) and relative.endswith(".h5"):
            hashes[relative.removeprefix("datasets/robofactory_multi_robot/")] = (
                digest.lower()
            )
    if not hashes:
        raise ValueError(f"No RoboFactory H5 records found in {manifest_path}")
    return hashes


def _trajectory_rank(task_name: str, source_path: str, trajectory: str) -> str:
    identity = "\0".join((SCHEMA_VERSION, task_name, source_path, trajectory))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _load_unique_validation_trajectories(
    selection_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    unique: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for line_number, raw_line in enumerate(
        selection_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line:
            continue
        record = json.loads(raw_line)
        if not isinstance(record, Mapping):
            raise TypeError(f"Selection line {line_number} is not a JSON object")
        if record.get("split") != "val":
            continue
        task_name = str(record.get("task_name"))
        if task_name not in TASK_AGENT_COUNTS:
            continue
        source_path = str(record.get("source_path"))
        trajectory = str(record.get("trajectory"))
        key = (source_path, trajectory)
        normalized = {
            "task_name": task_name,
            "source_path": source_path,
            "trajectory": trajectory,
            "agent_count": int(record.get("agent_count")),
            "agent_names": list(map(str, record.get("agent_names", []))),
        }
        previous = unique[task_name].get(key)
        if previous is not None and previous != normalized:
            raise ValueError(
                f"Inconsistent duplicate selection records for {task_name}:{source_path}:{trajectory}"
            )
        unique[task_name][key] = normalized

    missing = sorted(set(TASK_AGENT_COUNTS) - set(unique))
    if missing:
        raise ValueError(
            f"Selection manifest has no held-out trajectories for {missing}"
        )
    counts = {task_name: len(records) for task_name, records in unique.items()}
    ranked = {
        task_name: sorted(
            records.values(),
            key=lambda record: _trajectory_rank(
                task_name,
                record["source_path"],
                record["trajectory"],
            ),
        )
        for task_name, records in unique.items()
    }
    return ranked, counts


def _episode_metadata(sidecar: Mapping[str, Any], episode_id: int) -> Mapping[str, Any]:
    matches = [
        episode
        for episode in sidecar.get("episodes", [])
        if int(episode.get("episode_id", -1)) == episode_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one sidecar episode_id={episode_id}, got {len(matches)}"
        )
    return matches[0]


def build_panel(
    *,
    dataset_root: Path,
    selection_path: Path,
    bundle_manifest_path: Path,
    trajectories_per_task: int,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    selection_path = selection_path.expanduser().resolve(strict=True)
    bundle_manifest_path = bundle_manifest_path.expanduser().resolve(strict=True)
    if trajectories_per_task < 1:
        raise ValueError("trajectories_per_task must be positive")

    ranked, heldout_counts = _load_unique_validation_trajectories(selection_path)
    source_hashes = _load_bundle_h5_hashes(bundle_manifest_path)
    episodes: list[dict[str, Any]] = []
    task_offsets: dict[str, tuple[int, int]] = {}
    for task_name in TASK_AGENT_COUNTS:
        candidates = ranked[task_name]
        if len(candidates) < trajectories_per_task:
            raise ValueError(
                f"Task {task_name} has only {len(candidates)} held-out trajectories; "
                f"requested {trajectories_per_task}"
            )
        task_start = len(episodes)
        for task_index, record in enumerate(candidates[:trajectories_per_task]):
            source_relative = record["source_path"]
            source = _safe_source(dataset_root, source_relative)
            expected_source_sha256 = source_hashes.get(source_relative)
            if expected_source_sha256 is None:
                raise KeyError(
                    f"Source H5 is absent from sealed bundle manifest: {source_relative}"
                )
            sidecar_path = source.with_suffix(".json")
            if sidecar_path.is_symlink() or not sidecar_path.is_file():
                raise FileNotFoundError(f"Source sidecar is missing: {sidecar_path}")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            env_info = sidecar.get("env_info")
            if not isinstance(env_info, Mapping) or env_info.get("env_id") != task_name:
                raise ValueError(f"Sidecar env_id mismatch for {source_relative}")

            trajectory = record["trajectory"]
            match = TRAJECTORY_PATTERN.fullmatch(trajectory)
            if match is None:
                raise ValueError(f"Unsupported trajectory name: {trajectory!r}")
            episode_id = int(match.group(1))
            metadata = _episode_metadata(sidecar, episode_id)
            source_success = metadata.get("success")
            if not isinstance(source_success, (bool, int)) or int(
                source_success
            ) not in (0, 1):
                raise TypeError(
                    f"Source success must be bool-like 0/1: {source}:{trajectory} "
                    f"got={source_success!r}"
                )
            if not bool(source_success):
                raise ValueError(
                    f"Source demonstration is not marked successful: {source}:{trajectory}"
                )

            with h5py.File(source, "r") as handle:
                if trajectory not in handle:
                    raise KeyError(f"Missing {trajectory} in {source}")
                group = handle[trajectory]
                if "env_states" not in group or "actions" not in group:
                    raise KeyError(f"{source}:{trajectory} lacks env_states/actions")
                agent_names = sorted(group["actions"].keys(), key=_agent_sort_key)
                expected_count = TASK_AGENT_COUNTS[task_name]
                if (
                    len(agent_names) != expected_count
                    or agent_names != record["agent_names"]
                ):
                    raise ValueError(
                        f"Agent contract mismatch for {source}:{trajectory}: "
                        f"selection={record['agent_names']} H5={agent_names} expected_N={expected_count}"
                    )
                action_lengths = {
                    int(group["actions"][name].shape[0]) for name in agent_names
                }
                action_dims = {
                    int(group["actions"][name].shape[1]) for name in agent_names
                }
                if len(action_lengths) != 1 or action_dims != {8}:
                    raise ValueError(f"Invalid action arrays for {source}:{trajectory}")
                action_length = action_lengths.pop()

            reset_kwargs = metadata.get("reset_kwargs", {})
            if not isinstance(reset_kwargs, Mapping):
                raise TypeError(
                    f"reset_kwargs must be a mapping for {source}:{trajectory}"
                )
            episodes.append(
                {
                    "panel_index": len(episodes),
                    "task_index": task_index,
                    "task_name": task_name,
                    "agent_count": TASK_AGENT_COUNTS[task_name],
                    "agent_names": agent_names,
                    "source_path": source_relative,
                    "source_h5_bytes": source.stat().st_size,
                    "source_h5_sha256": expected_source_sha256,
                    "source_sidecar_sha256": sha256_file(sidecar_path),
                    "trajectory": trajectory,
                    "episode_id": episode_id,
                    "episode_seed": int(metadata.get("episode_seed")),
                    "reset_kwargs": json.loads(json.dumps(reset_kwargs)),
                    "demonstration_steps": action_length,
                    "source_success": True,
                    "max_episode_steps": int(env_info.get("max_episode_steps")),
                    "rank_sha256": _trajectory_rank(
                        task_name, source_relative, trajectory
                    ),
                }
            )
        task_offsets[task_name] = (task_start, len(episodes))

    return {
        "schema_version": SCHEMA_VERSION,
        "selection_rule": SELECTION_RULE,
        "trajectories_per_task": trajectories_per_task,
        "task_agent_counts": TASK_AGENT_COUNTS,
        "task_offsets": {key: list(value) for key, value in task_offsets.items()},
        "heldout_trajectory_counts": heldout_counts,
        "selection_manifest": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "input_bundle_manifest": {
            "path": str(bundle_manifest_path),
            "sha256": sha256_file(bundle_manifest_path),
        },
        "dataset_root": str(dataset_root),
        "episodes": episodes,
    }


def _write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--input-bundle-manifest", type=Path, required=True)
    parser.add_argument("--trajectories-per-task", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = build_panel(
        dataset_root=args.dataset_root,
        selection_path=args.selection_manifest,
        bundle_manifest_path=args.input_bundle_manifest,
        trajectories_per_task=args.trajectories_per_task,
    )
    _write_atomic_json(args.output.expanduser(), panel)
    print(
        json.dumps(
            {
                "episodes": len(panel["episodes"]),
                "output": str(args.output.expanduser().resolve()),
                "output_sha256": sha256_file(args.output.expanduser()),
                "status": "PASS",
                "tasks": len(TASK_AGENT_COUNTS),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
