#!/usr/bin/env python3
"""Validate and atomically publish the frozen eight-episode evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "FASTWAM-MR-N234-VG1H1GAU1-STEP10000-PLACEFOOD-SAME8-S42-R1-20260823"
RUN_ID = "fastwam-gau1-step10k-placefood-same8-r1-20260823"
CHECKPOINT = "/oss-chengjuntao/artifacts/fastwam-n234-vg1h1gau1-cont50k-s42-24g-r1-20260822/checkpoints/weights/step_010000.pt"
CHECKPOINT_BYTES = 12_047_213_657
ENVIRONMENT_SEEDS = (333183, 333327, 333225, 333180, 333251, 333130, 333167, 333234)
POLICY_SEEDS = tuple(range(10000, 10008))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_json(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    _require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"unsafe JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON document is not an object: {path}")
    return value


def _read_one_jsonl(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    _require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"unsafe JSONL file: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(len(lines) == 1, f"expected exactly one episode record in {path}, got {len(lines)}")
    value = json.loads(lines[0])
    _require(isinstance(value, dict), f"episode record is not an object: {path}")
    return value


def _copy_regular_tree(source: Path, destination: Path) -> None:
    destination.mkdir(mode=0o700)
    for entry in sorted(source.iterdir(), key=lambda item: item.name):
        metadata = entry.lstat()
        target = destination / entry.name
        if stat.S_ISDIR(metadata.st_mode):
            _copy_regular_tree(entry, target)
        elif stat.S_ISREG(metadata.st_mode):
            with entry.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            os.chmod(target, 0o600)
        else:
            raise RuntimeError(f"refusing non-regular evaluation artifact: {entry}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    temp_root = args.temp_root.resolve(strict=True)
    output_root = args.output_root
    _require(not output_root.exists() and not output_root.is_symlink(), f"output already exists: {output_root}")
    parent = output_root.parent.resolve(strict=True)
    _require(parent.is_dir() and not parent.is_symlink(), f"unsafe output parent: {parent}")
    _require(output_root.parent.resolve(strict=True) == parent, "output parent identity drift")

    records: list[dict[str, Any]] = []
    for index in range(8):
        shard = temp_root / f"episode-{index:02d}"
        _require(shard.is_dir() and not shard.is_symlink(), f"missing shard {index}: {shard}")
        _require({item.name for item in shard.iterdir()} == {"episodes.jsonl", "run_manifest.json", "summary.json"}, f"unexpected files in shard {index}")
        record = _read_one_jsonl(shard / "episodes.jsonl")
        manifest = _read_json(shard / "run_manifest.json")
        summary = _read_json(shard / "summary.json")

        _require(manifest.get("schema_version") == "fastwam-robofactory-eval-run-v2", f"shard {index} manifest schema drift")
        _require(manifest.get("status") == "terminal", f"shard {index} is non-terminal")
        _require(manifest.get("mode") == "fastwam" and manifest.get("task_name") == "PlaceFood-rf", f"shard {index} task/mode drift")
        _require(manifest.get("episode_start") == index and manifest.get("num_episodes") == 1, f"shard {index} selection drift")
        _require(manifest.get("max_steps_override") == 300 and manifest.get("exec_horizon") == 5, f"shard {index} closed-loop contract drift")
        _require(manifest.get("policy_seed_base") == 10000, f"shard {index} policy seed base drift")
        _require(manifest.get("eval_code_commit") == args.source_commit, f"shard {index} source commit drift")
        _require(manifest.get("integrity_mode") == "metadata_no_hash", f"shard {index} integrity mode drift")
        policy = manifest.get("policy")
        _require(isinstance(policy, dict), f"shard {index} lacks policy provenance")
        _require(policy.get("checkpoint_path") == CHECKPOINT and policy.get("checkpoint_size_bytes") == CHECKPOINT_BYTES, f"shard {index} checkpoint drift")
        _require(policy.get("integrity_mode") == "metadata_no_hash" and policy.get("checkpoint_sha256") is None, f"shard {index} checkpoint integrity drift")
        _require(policy.get("action_horizon") == 32 and policy.get("num_inference_steps") == 20, f"shard {index} policy horizon drift")
        _require(policy.get("gaussian_conditioning") is True and isinstance(policy.get("teacher"), dict), f"shard {index} GAU1 conditioning missing")

        _require(summary.get("schema_version") == "fastwam-robofactory-eval-summary-v2", f"shard {index} summary schema drift")
        _require(summary.get("status") == "PASS" and summary.get("infrastructure_errors") == 0, f"shard {index} infrastructure failure")
        _require(summary.get("episodes_requested") == summary.get("episodes_recorded") == summary.get("episodes_completed") == 1, f"shard {index} incomplete")
        _require(record.get("status") == "completed" and record.get("task_name") == "PlaceFood-rf", f"shard {index} episode incomplete")
        _require(record.get("task_index") == index and record.get("panel_index") == index, f"shard {index} panel index drift")
        _require(record.get("environment_seed") == ENVIRONMENT_SEEDS[index], f"shard {index} environment seed drift")
        _require(record.get("policy_seed") == POLICY_SEEDS[index], f"shard {index} policy seed drift")
        _require(isinstance(record.get("success"), bool), f"shard {index} success is not boolean")
        for field in ("steps", "policy_queries", "action_bound_violations"):
            _require(isinstance(record.get(field), int) and record[field] >= 0, f"shard {index} invalid {field}")
        records.append(record)

    _require({item.name for item in temp_root.iterdir()} == {f"episode-{i:02d}" for i in range(8)}, "unexpected shard set")
    successes = sum(int(record["success"]) for record in records)
    aggregate = {
        "schema_version": "fastwam-placefood-same8-aggregate-v1",
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "job_id": args.job_id,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "integrity_mode": "metadata_no_hash",
        "checkpoint_path": CHECKPOINT,
        "checkpoint_size_bytes": CHECKPOINT_BYTES,
        "task_name": "PlaceFood-rf",
        "episodes": 8,
        "successes": successes,
        "closed_loop_success_rate": successes / 8,
        "grasp_metric_available": False,
        "max_lift_metric_available": False,
        "total_steps": sum(record["steps"] for record in records),
        "total_policy_queries": sum(record["policy_queries"] for record in records),
        "action_bound_violations": sum(record["action_bound_violations"] for record in records),
        "environment_seeds": list(ENVIRONMENT_SEEDS),
        "policy_seeds": list(POLICY_SEEDS),
        "episode_results": records,
    }

    staging = parent / f".{output_root.name}.staging.{os.getpid()}"
    _require(not staging.exists() and not staging.is_symlink(), f"staging path exists: {staging}")
    try:
        staging.mkdir(mode=0o700)
        shards_out = staging / "shards"
        shards_out.mkdir(mode=0o700)
        for index in range(8):
            _copy_regular_tree(temp_root / f"episode-{index:02d}", shards_out / f"episode-{index:02d}")
        _atomic_json(staging / "aggregate.json", aggregate)
        _atomic_json(staging / "terminal-receipt.json", {"status": "SCIENTIFIC_COMPLETE", "job_id": args.job_id, "source_commit": args.source_commit, "successes": successes, "episodes": 8})
        _atomic_json(staging / "COMPLETE.json", {"status": "PASS", "successes": successes, "episodes": 8})
        os.rename(staging, output_root)
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    print(json.dumps(aggregate, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
