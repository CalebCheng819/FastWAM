#!/usr/bin/env python3
"""Validate, compare, and marker-atomically publish the GAU0 same-panel eval."""

from __future__ import annotations

import argparse
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "fastwam-gau0-placefood-same8-comparison-v1"
ARMS = ("gau1_stats", "gau0_native_stats")
EXPECTED_ENV_SEEDS = (333183, 333327, 333225, 333180, 333251, 333130, 333167, 333234)
EXPECTED_POLICY_SEEDS = tuple(range(10000, 10008))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_read(path: Path) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RuntimeError(f"unsafe single-link file: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"file descriptor identity mismatch: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(fd)
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_mode", "st_nlink")
    if (
        any(getattr(before, field) != getattr(opened, field) for field in fields)
        or any(getattr(opened, field) != getattr(after, field) for field in fields)
        or len(payload) != after.st_size
    ):
        raise RuntimeError(f"file changed during read: {path}")
    return payload


def load_json(path: Path) -> Any:
    try:
        return json.loads(stable_read(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON file: {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[Any]:
    try:
        lines = stable_read(path).decode("utf-8").splitlines()
        return [json.loads(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSONL file: {path}: {exc}") from exc


def write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise RuntimeError("short write while publishing evaluation artifact")
        offset += written


def write_json_exclusive(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def exact_directory_names(path: Path) -> list[str]:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"not an ordinary directory: {path}")
    names: list[str] = []
    with os.scandir(path) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                raise RuntimeError(f"unsupported shard entry: {path / entry.name}")
            names.append(entry.name)
    return sorted(names)


def validate_arm(temp_root: Path, arm: str) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    for index in range(8):
        shard = temp_root / arm / f"episode-{index:02d}"
        names = exact_directory_names(shard)
        if names != ["episodes.jsonl", "run_manifest.json", "summary.json"]:
            raise RuntimeError(f"unexpected shard allowlist for {arm}/{index}: {names}")
        records = load_jsonl(shard / "episodes.jsonl")
        if len(records) != 1:
            raise RuntimeError(f"{arm}/{index} has {len(records)} episode records")
        episode = records[0]
        summary = load_json(shard / "summary.json")
        manifest = load_json(shard / "run_manifest.json")
        expected = {
            "task_index": index,
            "panel_index": index,
            "environment_seed": EXPECTED_ENV_SEEDS[index],
            "policy_seed": EXPECTED_POLICY_SEEDS[index],
            "status": "completed",
        }
        for key, value in expected.items():
            if episode.get(key) != value:
                raise RuntimeError(f"{arm}/{index} {key} mismatch: {episode.get(key)!r}")
        if episode.get("task_name") != "PlaceFood-rf":
            raise RuntimeError(f"{arm}/{index} task mismatch")
        if summary.get("status") != "PASS" or summary.get("episodes_completed") != 1:
            raise RuntimeError(f"{arm}/{index} evaluator summary is not PASS")
        if summary.get("infrastructure_errors") != 0 or summary.get("episodes_requested") != 1:
            raise RuntimeError(f"{arm}/{index} evaluator accounting mismatch")
        argv = manifest.get("argv")
        if not isinstance(argv, list) or "--no-gaussian-conditioning" not in argv:
            raise RuntimeError(f"{arm}/{index} does not prove GAU0 execution")
        episodes.append(episode)
        shard_summaries.append(summary)

    episodes.sort(key=lambda item: item["task_index"])
    success = sum(bool(item["success"]) for item in episodes)
    return {
        "arm": arm,
        "episodes_requested": 8,
        "episodes_completed": 8,
        "infrastructure_errors": 0,
        "successes": success,
        "success_rate": success / 8.0,
        "total_steps": sum(int(item["steps"]) for item in episodes),
        "policy_queries": sum(int(item["policy_queries"]) for item in episodes),
        "action_bound_violations": sum(int(item["action_bound_violations"]) for item in episodes),
        "episodes": episodes,
        "shard_summaries": shard_summaries,
    }


def validate_baseline(root: Path) -> dict[str, Any]:
    aggregate = load_json(root / "aggregate.json")
    episodes: list[dict[str, Any]] = []
    for shard_index in range(4):
        shard = root / f"shard{shard_index}-episodes{2 * shard_index}-{2 * shard_index + 1}"
        episodes.extend(load_jsonl(shard / "episodes.jsonl"))
    episodes.sort(key=lambda item: item["task_index"])
    if len(episodes) != 8:
        raise RuntimeError("GAU1 baseline does not contain eight episodes")
    for index, episode in enumerate(episodes):
        if (
            episode.get("task_index") != index
            or episode.get("environment_seed") != EXPECTED_ENV_SEEDS[index]
            or episode.get("policy_seed") != EXPECTED_POLICY_SEEDS[index]
            or episode.get("status") != "completed"
        ):
            raise RuntimeError(f"GAU1 baseline episode {index} identity mismatch")
    successes = sum(bool(item["success"]) for item in episodes)
    if successes != 0 or sum(int(item["action_bound_violations"]) for item in episodes) != 2551:
        raise RuntimeError("GAU1 frozen baseline headline metrics changed")
    return {"aggregate": aggregate, "episodes": episodes, "successes": successes}


def comparison(primary: dict[str, Any], native: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for index in range(8):
        gau1 = baseline["episodes"][index]
        gau0 = primary["episodes"][index]
        native_episode = native["episodes"][index]
        rows.append(
            {
                "task_index": index,
                "env_seed": EXPECTED_ENV_SEEDS[index],
                "policy_seed": EXPECTED_POLICY_SEEDS[index],
                "gau1_success": bool(gau1["success"]),
                "gau0_gau1_stats_success": bool(gau0["success"]),
                "gau0_native_stats_success": bool(native_episode["success"]),
                "gau1_steps": int(gau1["steps"]),
                "gau0_gau1_stats_steps": int(gau0["steps"]),
                "gau0_native_stats_steps": int(native_episode["steps"]),
                "gau1_action_bound_violations": int(gau1["action_bound_violations"]),
                "gau0_gau1_stats_action_bound_violations": int(gau0["action_bound_violations"]),
                "gau0_native_stats_action_bound_violations": int(native_episode["action_bound_violations"]),
            }
        )
    return {
        "schema": SCHEMA,
        "comparison_scope": "same PlaceFood-rf panel, env seeds, policy seeds, evaluator, horizons, and inference steps",
        "causal_limit": "GAU0 and GAU1 checkpoints differ in training lineage and trainable scope; this is a matched evaluation, not an isolated Gaussian causal ablation",
        "gau1_baseline": {
            "successes": baseline["successes"],
            "episodes": 8,
            "action_bound_violations": sum(int(item["action_bound_violations"]) for item in baseline["episodes"]),
        },
        "gau0_gau1_stats": {key: primary[key] for key in ("successes", "episodes_completed", "success_rate", "total_steps", "policy_queries", "action_bound_violations")},
        "gau0_native_stats": {key: native[key] for key in ("successes", "episodes_completed", "success_rate", "total_steps", "policy_queries", "action_bound_violations")},
        "paired_episodes": rows,
    }


def copy_tree_no_links(source: Path, destination: Path) -> tuple[int, int]:
    if not stat.S_ISDIR(source.lstat().st_mode):
        raise RuntimeError(f"copy source is not an ordinary directory: {source}")
    files = 0
    bytes_total = 0
    for current, directories, filenames in os.walk(source, followlinks=False):
        current_path = Path(current)
        directories.sort()
        filenames.sort()
        relative = current_path.relative_to(source)
        target_dir = destination / relative
        target_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not stat.S_ISDIR(target_dir.lstat().st_mode):
            raise RuntimeError(f"copy target is not an ordinary directory: {target_dir}")
        for name in directories:
            info = (current_path / name).lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise RuntimeError(f"unsupported directory entry: {current_path / name}")
        for name in filenames:
            src = current_path / name
            payload = stable_read(src)
            dst = target_dir / name
            fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            try:
                write_all(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
            published = dst.lstat()
            if not stat.S_ISREG(published.st_mode) or published.st_nlink != 1 or published.st_size != len(payload):
                raise RuntimeError(f"published file contract failed: {dst}")
            files += 1
            bytes_total += len(payload)
    return files, bytes_total


def regular_file_inventory(root: Path) -> tuple[int, int]:
    files = 0
    bytes_total = 0
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories.sort()
        filenames.sort()
        for name in directories:
            path = current_path / name
            if not stat.S_ISDIR(path.lstat().st_mode):
                raise RuntimeError(f"unsupported published directory entry: {path}")
        for name in filenames:
            path = current_path / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise RuntimeError(f"unsupported published file entry: {path}")
            files += 1
            bytes_total += info.st_size
    return files, bytes_total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise RuntimeError("unique output root already exists")

    arm_results = {arm: validate_arm(args.temp_root, arm) for arm in ARMS}
    baseline = validate_baseline(args.baseline_root)
    compared = comparison(arm_results["gau1_stats"], arm_results["gau0_native_stats"], baseline)

    args.output_root.mkdir(mode=0o700, parents=False)
    if not stat.S_ISDIR(args.output_root.lstat().st_mode):
        raise RuntimeError("output root is not an ordinary directory")
    try:
        for arm in ARMS:
            copy_tree_no_links(args.temp_root / arm, args.output_root / arm)
            write_json_exclusive(args.output_root / f"{arm}-aggregate.json", arm_results[arm])
        write_json_exclusive(args.output_root / "comparison.json", compared)
        artifact_files, artifact_bytes = regular_file_inventory(args.output_root)
        terminal = {
            "schema": "fastwam-gau0-placefood-same8-terminal-v1",
            "status": "SCIENTIFIC_COMPLETE",
            "completed_at": utc_now(),
            "source_commit": args.source_commit,
            "job_id": args.job_id,
            "gaussian_conditioning": False,
            "arms": list(ARMS),
            "episode_invocations": 16,
            "artifact_files_before_terminal": artifact_files,
            "artifact_bytes_before_terminal": artifact_bytes,
            "comparison": compared,
        }
        write_json_exclusive(args.output_root / "terminal-receipt.json", terminal)
        write_json_exclusive(
            args.output_root / "COMPLETE.json",
            {
                "schema": "fastwam-gau0-placefood-same8-complete-v1",
                "status": "SCIENTIFIC_COMPLETE",
                "terminal_receipt": "terminal-receipt.json",
                "completed_at": terminal["completed_at"],
            },
        )
    except BaseException:
        raise
    print(json.dumps({"status": "SCIENTIFIC_COMPLETE", "comparison": compared}, indent=2))


if __name__ == "__main__":
    main()
