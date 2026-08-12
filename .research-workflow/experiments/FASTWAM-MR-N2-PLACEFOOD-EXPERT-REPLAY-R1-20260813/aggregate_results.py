#!/usr/bin/env python3
"""Validate and aggregate the fixed PlaceFood expert-replay panel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping


SCHEMA_VERSION = "fastwam-placefood-expert-replay-aggregate-nohash-v1"
EXPECTED_EPISODE_SCHEMA = "fastwam-placefood-fixed-diagnostic-v3"
EXPECTED_SPLITS = ("train", "val")
EXPECTED_EPISODES_PER_SPLIT = 8
EXPECTED_PANEL_SCHEMA = "fastwam-robofactory-split-panel-nohash-v1"
EXPECTED_FORMAL_CONTRACT = {
    "max_steps": 300,
    "initial_state": "raw",
    "action_source": "stored_h5_expert",
    "policy_initialized": False,
}
EXPECTED_ARGV = {
    "--mode": "expert-replay",
    "--task": "PlaceFood-rf",
    "--max-steps": "300",
    "--initial-state": "raw",
}
FORBIDDEN_POLICY_FLAGS = {
    "--checkpoint",
    "--gaussian-cache",
    "--stats",
    "--context-file",
    "--model-cache-root",
    "--policy-lightning-repo",
    "--noposplat-checkpoint",
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
    root: Path, split: str, index: int, expected_evaluation_code_commit: str
) -> dict[str, Any]:
    episode_dir = root / split / f"episode-{index:02d}"
    summary = _load_json(episode_dir / "summary.json")
    manifest = _load_json(episode_dir / "run_manifest.json")
    if summary.get("schema_version") != EXPECTED_EPISODE_SCHEMA:
        raise ValueError(f"episode schema mismatch: {episode_dir}")
    if summary.get("status") != "COMPLETED":
        raise ValueError(f"episode is not COMPLETED: {episode_dir}")
    if manifest.get("status") != "terminal":
        raise ValueError(f"run manifest is not terminal: {episode_dir}")
    if manifest.get("policy_initialized") is not False:
        raise ValueError(f"policy was initialized during expert replay: {episode_dir}")
    if manifest.get("training_code_commit") is not None:
        raise ValueError(f"expert replay unexpectedly records training code: {episode_dir}")
    if manifest.get("evaluation_code_commit") != expected_evaluation_code_commit:
        raise ValueError(f"evaluation code commit mismatch: {episode_dir}")
    if manifest.get("formal_contract_requested") is not True:
        raise ValueError(f"formal contract was not requested: {episode_dir}")
    contract = manifest.get("formal_expert_replay_contract")
    expected_contract = dict(EXPECTED_FORMAL_CONTRACT)
    expected_contract["evaluation_code_commit"] = expected_evaluation_code_commit
    if contract != expected_contract:
        raise ValueError(f"formal expert replay contract mismatch: {episode_dir}")
    if manifest.get("formal_rollout_contract") is not None:
        raise ValueError(f"policy rollout contract appeared in expert replay: {episode_dir}")

    argv = manifest.get("argv")
    if not isinstance(argv, list) or "--formal-contract" not in argv:
        raise ValueError(f"formal evaluator argv is absent: {episode_dir}")
    for flag, expected in EXPECTED_ARGV.items():
        if _single_argv_value(argv, flag, episode_dir) != expected:
            raise ValueError(f"evaluator argument mismatch for {flag}: {episode_dir}")
    forbidden = sorted(flag for flag in FORBIDDEN_POLICY_FLAGS if flag in argv)
    if forbidden:
        raise ValueError(f"policy inputs appeared in expert replay argv: {forbidden}")

    panel = manifest.get("panel")
    selected = manifest.get("episode")
    replay = summary.get("expert_replay")
    if not isinstance(panel, dict) or panel.get("split") != split:
        raise ValueError(f"panel split mismatch: {episode_dir}")
    if panel.get("schema_version") != EXPECTED_PANEL_SCHEMA:
        raise ValueError(f"panel schema mismatch: {episode_dir}")
    if int(panel.get("episode_count", -1)) != EXPECTED_EPISODES_PER_SPLIT:
        raise ValueError(f"panel episode count mismatch: {episode_dir}")
    if not isinstance(selected, dict):
        raise ValueError(f"selected episode metadata is absent: {episode_dir}")
    if selected.get("split") != split or selected.get("task") != "PlaceFood-rf":
        raise ValueError(f"selected episode identity mismatch: {episode_dir}")
    if int(selected.get("panel_index", -1)) != index:
        raise ValueError(f"panel index mismatch: {episode_dir}")
    if not isinstance(replay, dict) or replay.get("status") != "completed":
        raise ValueError(f"expert replay result is not completed: {episode_dir}")
    if replay.get("action_source") != "stored_h5_expert":
        raise ValueError(f"expert action source mismatch: {episode_dir}")
    if replay.get("policy_initialized") is not False:
        raise ValueError(f"expert replay initialized policy: {episode_dir}")
    if not isinstance(summary.get("simulator_success"), bool):
        raise ValueError(f"simulator_success must be boolean: {episode_dir}")
    success = bool(summary["simulator_success"])
    if bool(replay.get("success")) != success:
        raise ValueError(f"expert replay success disagrees with summary: {episode_dir}")
    steps = int(replay.get("steps", -1))
    available = int(replay.get("expert_actions_available", -1))
    executed = int(replay.get("expert_actions_executed", -1))
    if not 0 < steps <= min(300, available) or executed != steps:
        raise ValueError(f"expert replay step counts are inconsistent: {episode_dir}")
    buckets = replay.get("bound_violations")
    if not isinstance(buckets, dict):
        raise ValueError(f"bound violation summary is absent: {episode_dir}")
    return {
        "split": split,
        "panel_index": index,
        "episode_id": int(selected["episode_id"]),
        "trajectory": str(selected["trajectory"]),
        "global_ordinal": int(selected["global_ordinal"]),
        "success": success,
        "steps": steps,
        "expert_actions_available": available,
        "termination_reason": str(replay.get("termination_reason")),
        "meat_max_lift_m": float(replay["meat_max_lift_m"]),
        "pot_lid_max_qpos": float(replay["pot_lid_max_qpos"]),
        "bound_scalar_violations": int(buckets.get("total_scalar_violations", -1)),
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
    parser.add_argument("--expected-evaluation-code-commit", required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    rows = [
        _episode(root, split, index, args.expected_evaluation_code_commit)
        for split in EXPECTED_SPLITS
        for index in range(EXPECTED_EPISODES_PER_SPLIT)
    ]
    identities = {(row["split"], row["global_ordinal"]) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("expert replay panel contains duplicate episode identities")
    grouped: dict[str, dict[str, Any]] = {}
    for split in EXPECTED_SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        successes = sum(int(row["success"]) for row in split_rows)
        grouped[split] = {
            "episodes": len(split_rows),
            "successes": successes,
            "success_rate": successes / len(split_rows),
            "mean_steps": fmean(row["steps"] for row in split_rows),
            "mean_meat_max_lift_m": fmean(
                row["meat_max_lift_m"] for row in split_rows
            ),
            "bound_scalar_violations": sum(
                row["bound_scalar_violations"] for row in split_rows
            ),
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "action_source": "stored_h5_expert",
        "policy_initialized": False,
        "evaluation_code_commit": args.expected_evaluation_code_commit,
        "train": grouped["train"],
        "val": grouped["val"],
        "overall_successes": sum(int(row["success"]) for row in rows),
        "overall_episodes": len(rows),
        "overall_success_rate": sum(int(row["success"]) for row in rows) / len(rows),
        "episodes": rows,
        "provenance_policy": (
            "ordinary Git, run, path, timestamp, size, and version identifiers; "
            "no new artifact checksums"
        ),
    }
    output = args.output or (root / "aggregate.json")
    _atomic_new_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
