#!/usr/bin/env python3
"""Run and aggregate the fixed PlaceFood R5 closed-loop ablation panel.

The matrix separates replanning, temporal expert-action interventions, and
checkpoint progression.  Every simulator rollout is an independent process on
one GPU; partial output directories fail closed instead of being overwritten.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "fastwam-r5-closedloop-ablations-v1"
REQUESTED_CHECKPOINT_STEPS = (250, 500, 750, 1000, 2500, 5000)


@dataclass(frozen=True)
class Cell:
    name: str
    checkpoint_step: int
    exec_horizon: int
    oracle_intervention: str


CELLS = (
    Cell("step1000_h5_policy", 1000, 5, "none"),
    Cell("step1000_h1_policy", 1000, 1, "none"),
    Cell("step1000_h5_oracle_robot0_pose", 1000, 5, "robot0_pose"),
    Cell("step1000_h5_oracle_robot0_gripper", 1000, 5, "robot0_gripper"),
    Cell("step1000_h5_oracle_robot1_action", 1000, 5, "robot1_action"),
    Cell("step0500_h5_policy", 500, 5, "none"),
    Cell("step0500_h1_policy", 500, 1, "none"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _panel_rows(panel: Mapping[str, Any]) -> list[dict[str, int]]:
    episodes = [
        item
        for item in panel.get("episodes", [])
        if item.get("task") == "PlaceFood-rf"
    ]
    if len(episodes) < 8:
        raise ValueError(f"PlaceFood panel requires at least 8 episodes, got {len(episodes)}")
    rows: list[dict[str, int]] = []
    for episode_start, item in enumerate(episodes[:8]):
        rows.append(
            {
                "episode_start": episode_start,
                "environment_seed": int(item["episode_seed"]),
                "policy_seed": 10000 + episode_start,
            }
        )
    if len({row["environment_seed"] for row in rows}) != 8:
        raise ValueError("PlaceFood panel environment seeds must be unique")
    return rows


def checkpoint_availability(checkpoint_dir: Path) -> list[dict[str, Any]]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve(strict=True)
    records = []
    for step in REQUESTED_CHECKPOINT_STEPS:
        path = checkpoint_dir / f"step_{step:06d}.pt"
        record: dict[str, Any] = {
            "step": step,
            "path": str(path),
            "available": path.is_file(),
        }
        if path.is_file():
            stat = path.stat()
            record.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        records.append(record)
    return records


def _checkpoint_record(
    contract: Mapping[str, Any], checkpoint_step: int
) -> dict[str, Any]:
    matches = [
        record
        for record in contract["checkpoint_availability"]
        if int(record["step"]) == checkpoint_step
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one checkpoint record for step {checkpoint_step}, got {len(matches)}"
        )
    record = matches[0]
    if not record.get("available"):
        raise FileNotFoundError(f"Checkpoint step {checkpoint_step} is unavailable")
    required = {"path", "bytes", "mtime_ns"}
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(
            f"Checkpoint step {checkpoint_step} lacks frozen metadata: {missing}"
        )
    return record


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    panel = _read_json(args.panel)
    seeds = _panel_rows(panel)
    availability = checkpoint_availability(args.checkpoint_dir)
    available_steps = {item["step"] for item in availability if item["available"]}
    needed = {cell.checkpoint_step for cell in CELLS}
    missing = sorted(needed - available_steps)
    if missing:
        raise FileNotFoundError(f"Runnable checkpoint steps are missing: {missing}")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "source_root": str(args.source_root.resolve(strict=True)),
        "diagnostic": str(
            (args.source_root / "experiments/robofactory/diagnose_place_food_fixed.py").resolve(
                strict=True
            )
        ),
        "python": str(args.python.resolve(strict=True)),
        "panel": str(args.panel.resolve(strict=True)),
        "dataset_root": str(args.dataset_root.resolve(strict=True)),
        "robofactory_root": str(args.robofactory_root.resolve(strict=True)),
        "gaussian_cache": str(args.gaussian_cache.resolve(strict=True)),
        "checkpoint_dir": str(args.checkpoint_dir.resolve(strict=True)),
        "stats": str(args.stats.resolve(strict=True)),
        "context_file": str(args.context_file.resolve(strict=True)),
        "model_cache_root": str(args.model_cache_root.resolve(strict=True)),
        "policy_lightning_repo": str(args.policy_lightning_repo.resolve(strict=True)),
        "noposplat_checkpoint": str(args.noposplat_checkpoint.resolve(strict=True)),
        "task": "PlaceFood-rf",
        "initial_state": "raw",
        "max_steps": 300,
        "action_horizon": 32,
        "num_inference_steps": 20,
        "cells": [asdict(cell) for cell in CELLS],
        "seeds": seeds,
        "checkpoint_availability": availability,
        "metrics": [
            "closed_loop_success",
            "robot0_true_grasp",
            "meat_max_lift_m",
        ],
        "oracle_semantics": (
            "same-timestep held-out expert temporal replay intervention, not an "
            "omniscient state-conditioned oracle; requested component only; explicit "
            "policy fallback after expert trace exhaustion"
        ),
        "provenance_policy": (
            "Git revision, paths, timestamps, byte sizes, experiment/run IDs; "
            "no new artifact hashes"
        ),
    }


def ensure_frozen_plan(args: argparse.Namespace) -> dict[str, Any]:
    contract = build_plan(args)
    plan_path = args.output_root / "experiment_plan.json"
    if plan_path.exists():
        existing = _read_json(plan_path)
        if existing.get("contract") != contract:
            raise RuntimeError(f"Existing frozen plan differs: {plan_path}")
        return contract
    args.output_root.mkdir(parents=True, exist_ok=False)
    _atomic_json(
        plan_path,
        {
            "created_at": _utc_now(),
            "contract": contract,
        },
    )
    return contract


def run_id(cell: Cell, seed: Mapping[str, int]) -> str:
    return (
        f"{cell.name}-env{seed['environment_seed']}-"
        f"policy{seed['policy_seed']}"
    )


def _run_command(
    contract: Mapping[str, Any], cell: Cell, seed: Mapping[str, int], output: Path
) -> list[str]:
    checkpoint = _checkpoint_record(contract, cell.checkpoint_step)
    return [
        str(contract["python"]),
        str(contract["diagnostic"]),
        "--task",
        "PlaceFood-rf",
        "--panel",
        str(contract["panel"]),
        "--dataset-root",
        str(contract["dataset_root"]),
        "--robofactory-root",
        str(contract["robofactory_root"]),
        "--gaussian-cache",
        str(contract["gaussian_cache"]),
        "--episode-start",
        str(seed["episode_start"]),
        "--policy-seed",
        str(seed["policy_seed"]),
        "--checkpoint",
        str(checkpoint["path"]),
        "--integrity-mode",
        "metadata_no_hash",
        "--stats",
        str(contract["stats"]),
        "--context-file",
        str(contract["context_file"]),
        "--model-cache-root",
        str(contract["model_cache_root"]),
        "--policy-lightning-repo",
        str(contract["policy_lightning_repo"]),
        "--noposplat-checkpoint",
        str(contract["noposplat_checkpoint"]),
        "--device",
        "cuda:0",
        "--teacher-device",
        "cuda:0",
        "--action-horizon",
        str(contract["action_horizon"]),
        "--num-inference-steps",
        str(contract["num_inference_steps"]),
        "--output-dir",
        str(output),
        "--formal-contract",
        "--mode",
        "rollout",
        "--initial-state",
        str(contract["initial_state"]),
        "--exec-horizon",
        str(cell.exec_horizon),
        "--oracle-intervention",
        cell.oracle_intervention,
        "--max-steps",
        str(contract["max_steps"]),
    ]


def _completed_output(
    path: Path,
    cell: Cell,
    seed: Mapping[str, int],
    contract: Mapping[str, Any],
) -> bool:
    summary_path = path / "summary.json"
    manifest_path = path / "run_manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        return False
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    rollout = summary.get("rollout")
    expected_cell = {
        "initial_state": "raw",
        "exec_horizon": cell.exec_horizon,
        "oracle_intervention": cell.oracle_intervention,
    }
    checkpoint = _checkpoint_record(contract, cell.checkpoint_step)
    expected_policy = {
        "checkpoint_path": checkpoint["path"],
        "checkpoint_bytes": checkpoint["bytes"],
        "checkpoint_mtime_ns": checkpoint["mtime_ns"],
        "policy_seed": seed["policy_seed"],
    }
    return bool(
        summary.get("status") == "COMPLETED"
        and isinstance(rollout, dict)
        and rollout.get("status") == "completed"
        and manifest.get("status") == "terminal"
        and manifest.get("rollout_cell") == expected_cell
        and manifest.get("policy_request") == expected_policy
        and manifest.get("episode", {}).get("environment_seed")
        == seed["environment_seed"]
    )


def _one_run(
    *,
    contract: Mapping[str, Any],
    cell: Cell,
    seed: Mapping[str, int],
    output_root: Path,
    gpu_pool: "queue.Queue[int]",
) -> dict[str, Any]:
    identity = run_id(cell, seed)
    output = output_root / "runs" / cell.name / identity
    receipt = output_root / "receipts" / f"{identity}.json"
    if output.exists():
        if _completed_output(output, cell, seed, contract):
            return {"run_id": identity, "status": "skipped_completed"}
        raise RuntimeError(f"Partial or invalid run output refuses overwrite: {output}")
    if receipt.exists():
        raise RuntimeError(f"Receipt exists without valid completed output: {receipt}")
    gpu = gpu_pool.get()
    started = time.monotonic()
    started_at = _utc_now()
    log_path = output_root / "logs" / f"{identity}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _run_command(contract, cell, seed, output)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=contract["robofactory_root"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": identity,
            "cell": asdict(cell),
            "seed": dict(seed),
            "gpu": gpu,
            "command": command,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "returncode": process.returncode,
            "status": "completed" if process.returncode == 0 else "failed",
            "output": str(output),
            "log": str(log_path),
        }
        _atomic_json(receipt, record)
        if process.returncode != 0 or not _completed_output(
            output, cell, seed, contract
        ):
            raise RuntimeError(f"Rollout failed contract: {identity}; see {log_path}")
        return record
    finally:
        gpu_pool.put(gpu)


def run_matrix(
    args: argparse.Namespace, contract: Mapping[str, Any], *, pilot: bool
) -> list[dict[str, Any]]:
    seeds = list(contract["seeds"][:1] if pilot else contract["seeds"])
    cells = [Cell(**item) for item in contract["cells"]]
    jobs = [(cell, seed) for cell in cells for seed in seeds]
    gpu_pool: "queue.Queue[int]" = queue.Queue()
    for gpu in args.gpus:
        gpu_pool.put(gpu)
    results = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {
            executor.submit(
                _one_run,
                contract=contract,
                cell=cell,
                seed=seed,
                output_root=args.output_root,
                gpu_pool=gpu_pool,
            ): run_id(cell, seed)
            for cell, seed in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            identity = futures[future]
            try:
                record = future.result()
                results.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
            except Exception as error:  # noqa: BLE001 - persist all matrix failures
                errors.append(f"{identity}: {type(error).__name__}: {error}")
                print(errors[-1], file=sys.stderr, flush=True)
    if errors:
        raise RuntimeError("Matrix had failed cells:\n" + "\n".join(errors))
    return results


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def aggregate(args: argparse.Namespace, contract: Mapping[str, Any]) -> dict[str, Any]:
    cells = [Cell(**item) for item in contract["cells"]]
    cell_reports = []
    all_complete = True
    for cell in cells:
        rows = []
        for seed in contract["seeds"]:
            identity = run_id(cell, seed)
            path = args.output_root / "runs" / cell.name / identity
            if not _completed_output(path, cell, seed, contract):
                all_complete = False
                rows.append({"run_id": identity, "operational": False})
                continue
            summary = _read_json(path / "summary.json")
            rollout = summary["rollout"]
            rows.append(
                {
                    "run_id": identity,
                    "operational": True,
                    "environment_seed": seed["environment_seed"],
                    "policy_seed": seed["policy_seed"],
                    "success": bool(rollout["success"]),
                    "robot0_grasp_ever": bool(rollout["robot0_grasp_ever"]),
                    "robot0_grasp_steps": int(rollout["robot0_grasp_steps"]),
                    "meat_max_lift_m": float(rollout["meat_max_lift_m"]),
                    "oracle_coverage_fraction": rollout["oracle_coverage_fraction"],
                    "steps": int(rollout["steps"]),
                    "policy_queries": int(rollout["policy_queries"]),
                    "bound_violations": int(
                        rollout["bound_violations"]["total_scalar_violations"]
                    ),
                    "elapsed_seconds": float(rollout["elapsed_seconds"]),
                    "output": str(path),
                }
            )
        operational = [row for row in rows if row["operational"]]
        lifts = [float(row["meat_max_lift_m"]) for row in operational]
        oracle_coverage = [
            float(row["oracle_coverage_fraction"])
            for row in operational
            if row["oracle_coverage_fraction"] is not None
        ]
        cell_reports.append(
            {
                "cell": asdict(cell),
                "expected_runs": len(rows),
                "operational_runs": len(operational),
                "success_count": sum(bool(row["success"]) for row in operational),
                "closed_loop_success_rate": (
                    _mean([float(bool(row["success"])) for row in operational])
                ),
                "true_grasp_count": sum(
                    bool(row["robot0_grasp_ever"]) for row in operational
                ),
                "true_grasp_rate": _mean(
                    [float(bool(row["robot0_grasp_ever"])) for row in operational]
                ),
                "meat_max_lift_m_mean": _mean(lifts),
                "meat_max_lift_m_median": _median(lifts),
                "meat_max_lift_m_max": None if not lifts else max(lifts),
                "oracle_coverage_fraction_mean": _mean(oracle_coverage),
                "rows": rows,
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": contract["experiment_id"],
        "generated_at": _utc_now(),
        "status": "COMPLETE" if all_complete else "INCOMPLETE",
        "checkpoint_availability": contract["checkpoint_availability"],
        "cells": cell_reports,
    }
    _atomic_json(args.output_root / "aggregate.json", report)
    return report


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "pilot", "formal", "summarize"))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source-root", type=_path, required=True)
    parser.add_argument("--output-root", type=_path, required=True)
    parser.add_argument("--python", type=_path, required=True)
    parser.add_argument("--panel", type=_path, required=True)
    parser.add_argument("--dataset-root", type=_path, required=True)
    parser.add_argument("--robofactory-root", type=_path, required=True)
    parser.add_argument("--gaussian-cache", type=_path, required=True)
    parser.add_argument("--checkpoint-dir", type=_path, required=True)
    parser.add_argument("--stats", type=_path, required=True)
    parser.add_argument("--context-file", type=_path, required=True)
    parser.add_argument("--model-cache-root", type=_path, required=True)
    parser.add_argument("--policy-lightning-repo", type=_path, required=True)
    parser.add_argument("--noposplat-checkpoint", type=_path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    return parser


def main() -> None:
    args = _parser().parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        raise ValueError(f"GPU IDs must be distinct non-negative integers: {args.gpus}")
    contract = ensure_frozen_plan(args)
    if args.command == "plan":
        print(json.dumps({"status": "PLANNED", "contract": contract}, sort_keys=True))
        return
    if args.command in {"pilot", "formal"}:
        run_matrix(args, contract, pilot=args.command == "pilot")
    report = aggregate(args, contract)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
