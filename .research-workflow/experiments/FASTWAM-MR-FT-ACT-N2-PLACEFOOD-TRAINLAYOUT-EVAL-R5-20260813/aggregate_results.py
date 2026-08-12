#!/usr/bin/env python3
"""Validate and aggregate the frozen 8 train-layout versus 8 val-layout panel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping


SCHEMA_VERSION = "fastwam-placefood-train-layout-eval-aggregate-nohash-v1"
EXPECTED_SPLITS = ("train", "val")
EXPECTED_EPISODES_PER_SPLIT = 8
EXPECTED_PANEL_SCHEMA = "fastwam-robofactory-split-panel-nohash-v1"
EXPECTED_SPLIT_SEED = 42
EXPECTED_VAL_PROPORTION = 0.1
EXPECTED_SPLIT_KEY_SCHEME = "sorted_trajectory_ordinal_splitmix64_v1"
EXPECTED_CHECKPOINT = (
    "/oss-chengjuntao/artifacts/fastwam-action-n234-formal-r5-20260812/"
    "fastwam-act-n2-placefood-1k-s42-r5-20260812/checkpoints/weights/"
    "step_001000.pt"
)
EXPECTED_FORMAL_ROLLOUT_CONTRACT = {
    "max_steps": 300,
    "initial_state": "raw",
    "exec_horizon": 5,
    "explicit_cell": True,
}
EXPECTED_ARGV = {
    "--mode": "rollout",
    "--task": "PlaceFood-rf",
    "--max-steps": "300",
    "--initial-state": "raw",
    "--exec-horizon": "5",
    "--integrity-mode": "metadata_no_hash",
    "--action-horizon": "32",
    "--num-inference-steps": "20",
}


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _single_argv_value(argv: list[Any], flag: str, episode_dir: Path) -> str:
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"argv must contain exactly one {flag}: {episode_dir}")
    return str(argv[positions[0] + 1])


def _episode(
    root: Path,
    split: str,
    index: int,
    *,
    expected_checkpoint: str,
) -> dict[str, Any]:
    episode_dir = root / split / f"episode-{index:02d}"
    summary = _load_json(episode_dir / "summary.json")
    manifest = _load_json(episode_dir / "run_manifest.json")
    if summary.get("status") != "COMPLETED":
        raise ValueError(f"episode is not COMPLETED: {episode_dir}")
    if manifest.get("status") != "terminal":
        raise ValueError(f"run manifest is not terminal: {episode_dir}")
    panel = manifest.get("panel")
    selected = manifest.get("episode")
    rollout = summary.get("rollout")
    if not isinstance(panel, dict) or panel.get("split") != split:
        raise ValueError(f"panel split mismatch: {episode_dir}")
    if panel.get("schema_version") != EXPECTED_PANEL_SCHEMA:
        raise ValueError(f"panel schema mismatch: {episode_dir}")
    if int(panel.get("split_seed", -1)) != EXPECTED_SPLIT_SEED:
        raise ValueError(f"panel split seed mismatch: {episode_dir}")
    if float(panel.get("val_set_proportion", -1.0)) != EXPECTED_VAL_PROPORTION:
        raise ValueError(f"panel val proportion mismatch: {episode_dir}")
    if panel.get("split_key_scheme") != EXPECTED_SPLIT_KEY_SCHEME:
        raise ValueError(f"panel split key scheme mismatch: {episode_dir}")
    if int(panel.get("episode_count", -1)) != EXPECTED_EPISODES_PER_SPLIT:
        raise ValueError(f"panel episode count mismatch: {episode_dir}")
    if manifest.get("formal_contract_requested") is not True:
        raise ValueError(f"formal rollout contract was not requested: {episode_dir}")
    if manifest.get("formal_rollout_contract") != EXPECTED_FORMAL_ROLLOUT_CONTRACT:
        raise ValueError(f"formal rollout contract mismatch: {episode_dir}")
    argv = manifest.get("argv")
    if not isinstance(argv, list) or "--formal-contract" not in argv:
        raise ValueError(f"formal evaluator argv is absent: {episode_dir}")
    for flag, expected in EXPECTED_ARGV.items():
        if _single_argv_value(argv, flag, episode_dir) != expected:
            raise ValueError(f"evaluator argument mismatch for {flag}: {episode_dir}")
    if _single_argv_value(argv, "--checkpoint", episode_dir) != expected_checkpoint:
        raise ValueError(f"evaluator checkpoint mismatch: {episode_dir}")
    if not isinstance(selected, dict):
        raise ValueError(f"selected episode metadata is absent: {episode_dir}")
    if selected.get("split") != split:
        raise ValueError(f"selected episode split mismatch: {episode_dir}")
    if selected.get("task") != "PlaceFood-rf":
        raise ValueError(f"task mismatch: {episode_dir}")
    if int(selected.get("panel_index", -1)) != index:
        raise ValueError(f"panel index mismatch: {episode_dir}")
    if int(selected.get("task_index", -1)) != index:
        raise ValueError(f"task index mismatch: {episode_dir}")
    if int(selected.get("policy_seed", -1)) != 10000 + index:
        raise ValueError(f"policy seed mismatch: {episode_dir}")
    if not isinstance(rollout, dict) or rollout.get("status") != "completed":
        raise ValueError(f"rollout result is not completed: {episode_dir}")
    if not isinstance(summary.get("simulator_success"), bool):
        raise ValueError(f"simulator_success must be boolean: {episode_dir}")
    success = bool(summary["simulator_success"])
    if bool(rollout.get("success")) != success:
        raise ValueError(f"rollout success disagrees with summary: {episode_dir}")
    steps = int(rollout.get("steps", -1))
    queries = int(rollout.get("policy_queries", -1))
    if not 0 < steps <= 300:
        raise ValueError(f"rollout steps outside formal bounds: {episode_dir}")
    if not 0 < queries <= 300:
        raise ValueError(f"policy query count outside formal bounds: {episode_dir}")
    return {
        "split": split,
        "panel_index": index,
        "episode_id": int(selected["episode_id"]),
        "source_relative": str(selected["source_relative"]),
        "trajectory": str(selected["trajectory"]),
        "environment_seed": int(selected["environment_seed"]),
        "policy_seed": int(selected["policy_seed"]),
        "global_ordinal": int(selected["global_ordinal"]),
        "split_fraction": float(selected["split_fraction"]),
        "success": success,
        "steps": steps,
        "policy_queries": queries,
        "termination_reason": str(rollout.get("termination_reason")),
        "elapsed_seconds": float(summary["elapsed_seconds"]),
        "result_dir": str(episode_dir),
    }


def _atomic_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-checkpoint", default=EXPECTED_CHECKPOINT)
    parser.add_argument(
        "--comparison",
        default=(
            "same R5 FastWAM checkpoint and evaluator; only the recorded "
            "initial-state split differs"
        ),
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    rows = [
        _episode(
            root,
            split,
            index,
            expected_checkpoint=args.expected_checkpoint,
        )
        for split in EXPECTED_SPLITS
        for index in range(EXPECTED_EPISODES_PER_SPLIT)
    ]
    identities_by_split: dict[str, set[tuple[str, str]]] = {}
    ordinals_by_split: dict[str, set[int]] = {}
    for split in EXPECTED_SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        identities = {
            (row["source_relative"], row["trajectory"]) for row in split_rows
        }
        ordinals = {row["global_ordinal"] for row in split_rows}
        if len(identities) != EXPECTED_EPISODES_PER_SPLIT:
            raise ValueError(f"duplicate source episode identity in {split} split")
        if len(ordinals) != EXPECTED_EPISODES_PER_SPLIT:
            raise ValueError(f"duplicate global ordinal in {split} split")
        identities_by_split[split] = identities
        ordinals_by_split[split] = ordinals
    if identities_by_split["train"] & identities_by_split["val"]:
        raise ValueError("train and val source episode identities overlap")
    if ordinals_by_split["train"] & ordinals_by_split["val"]:
        raise ValueError("train and val global ordinals overlap")
    grouped: dict[str, dict[str, Any]] = {}
    for split in EXPECTED_SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        successes = sum(int(row["success"]) for row in split_rows)
        grouped[split] = {
            "episodes": len(split_rows),
            "successes": successes,
            "success_rate": successes / len(split_rows),
            "mean_steps": fmean(row["steps"] for row in split_rows),
            "mean_policy_queries": fmean(row["policy_queries"] for row in split_rows),
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "comparison": args.comparison,
        "train": grouped["train"],
        "val": grouped["val"],
        "train_minus_val_success_rate": (
            grouped["train"]["success_rate"] - grouped["val"]["success_rate"]
        ),
        "episodes": rows,
        "provenance_policy": "ordinary Git, run, path, timestamp, size, and version identifiers; no new artifact checksums",
    }
    output = args.output or (root / "aggregate.json")
    _atomic_new_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
