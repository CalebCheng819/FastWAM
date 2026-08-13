#!/usr/bin/env python3
"""Run one FastWAM checkpoint on the frozen PlaceFood closed-loop val8 panel."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "fastwam-fixed-policy-closedloop-panel-v3"
EXPECTED_RUNS = 8
CONTROL_REQUIREMENTS = {
    "direct": {
        "query_budget_by_horizon": {1: 300, 5: 300},
        "sim_budget": 300,
    },
    "official_topp": {
        "query_budget_by_horizon": {5: 384, 16: 120, 20: 96, 24: 80, 32: 60},
        "sim_budget": 30000,
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _regular_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def _python_executable(path: Path) -> Path:
    lexical = path.expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    resolved = lexical.resolve(strict=True)
    if not resolved.is_file() or not os.access(lexical, os.X_OK):
        raise ValueError(f"Python executable is not executable: {lexical}")
    # Invoking the venv symlink is what selects its pyvenv.cfg and site-packages.
    return lexical


def _directory(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def _panel_rows(panel: Mapping[str, Any]) -> list[dict[str, int]]:
    episodes = [
        item
        for item in panel.get("episodes", [])
        if item.get("task_name", item.get("task")) == "PlaceFood-rf"
    ]
    if len(episodes) < EXPECTED_RUNS:
        raise ValueError(
            f"PlaceFood panel requires {EXPECTED_RUNS} episodes, got {len(episodes)}"
        )
    paired = panel.get("paired_policy_seeds")
    if paired is not None and len(paired) < EXPECTED_RUNS:
        raise ValueError("paired_policy_seeds does not cover the fixed val8 panel")
    rows = []
    for episode_start, item in enumerate(episodes[:EXPECTED_RUNS]):
        panel_index = int(item.get("panel_index", episode_start))
        if panel_index != episode_start:
            raise ValueError(
                f"Panel index/order mismatch: {panel_index} != {episode_start}"
            )
        rows.append(
            {
                "episode_start": episode_start,
                "environment_seed": int(item["episode_seed"]),
                "policy_seed": int(
                    paired[episode_start]
                    if paired is not None
                    else 10000 + episode_start
                ),
            }
        )
    if len({item["environment_seed"] for item in rows}) != EXPECTED_RUNS:
        raise ValueError("Fixed val8 environment seeds must be unique")
    if len({item["policy_seed"] for item in rows}) != EXPECTED_RUNS:
        raise ValueError("Fixed val8 policy seeds must be unique")
    return rows


def _validate_control_contract(
    adapter: str, exec_horizon: int, query_budget: int, sim_budget: int
) -> int:
    controls = CONTROL_REQUIREMENTS[adapter]
    expected_queries = controls["query_budget_by_horizon"].get(exec_horizon)
    if expected_queries is None:
        raise ValueError(
            f"{adapter} requires exec_horizon in "
            f"{sorted(controls['query_budget_by_horizon'])}, got {exec_horizon}"
        )
    if (query_budget, sim_budget) != (
        expected_queries,
        controls["sim_budget"],
    ):
        raise ValueError(
            f"{adapter} requires query/simulator budgets "
            f"{(expected_queries, controls['sim_budget'])} at horizon {exec_horizon}"
        )
    return query_budget * exec_horizon


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    target_action_budget = _validate_control_contract(
        args.control_adapter,
        int(args.exec_horizon),
        int(args.max_policy_queries),
        int(args.max_simulator_steps),
    )
    source_root = _directory(args.source_root, label="source root")
    diagnostic = _regular_file(
        source_root / "experiments/robofactory/diagnose_place_food_fixed.py",
        label="diagnostic",
    )
    python = _python_executable(args.python)
    checkpoint = _regular_file(args.checkpoint, label="checkpoint")
    checkpoint_stat = checkpoint.stat()
    panel_path = _regular_file(args.panel, label="panel")
    rows = _panel_rows(_read_json(panel_path))
    model_project_root = _directory(
        args.model_project_root, label="model project root"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "candidate": args.candidate,
        "source_root": str(source_root),
        "diagnostic": str(diagnostic),
        "python": str(python),
        "python_realpath": str(python.resolve(strict=True)),
        "panel": str(panel_path),
        "dataset_root": str(_directory(args.dataset_root, label="dataset root")),
        "robofactory_root": str(
            _directory(args.robofactory_root, label="RoboFactory root")
        ),
        "gaussian_cache": str(
            _directory(args.gaussian_cache, label="Gaussian cache")
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": int(checkpoint_stat.st_size),
            "mtime_ns": int(checkpoint_stat.st_mtime_ns),
        },
        "training_code_commit": args.training_code_commit,
        "evaluation_code_commit": args.evaluation_code_commit,
        "model_project_root": str(model_project_root),
        "action_architecture": args.action_architecture,
        "stats": str(_regular_file(args.stats, label="normalization stats")),
        "context_file": str(_regular_file(args.context_file, label="task context")),
        "model_cache_root": str(
            _directory(args.model_cache_root, label="model cache root")
        ),
        "policy_lightning_repo": str(
            _directory(args.policy_lightning_repo, label="Policy-Lightning root")
        ),
        "noposplat_checkpoint": str(
            _regular_file(args.noposplat_checkpoint, label="NoPoSplat checkpoint")
        ),
        "nvidia_driver_lib_dir": str(
            _directory(args.nvidia_driver_lib_dir, label="NVIDIA driver libraries")
        ),
        "nvidia_vulkan_icd": str(
            _regular_file(args.nvidia_vulkan_icd, label="NVIDIA Vulkan ICD")
        ),
        "nvidia_egl_vendor_json": str(
            _regular_file(args.nvidia_egl_vendor_json, label="NVIDIA EGL manifest")
        ),
        "task": "PlaceFood-rf",
        "initial_state": "raw",
        "exec_horizon": int(args.exec_horizon),
        "control_adapter": args.control_adapter,
        "topp_step": float(args.topp_step),
        "max_policy_queries": int(args.max_policy_queries),
        "target_action_budget": target_action_budget,
        "max_simulator_steps": int(args.max_simulator_steps),
        "max_steps": 300,
        "action_horizon": 32,
        "num_inference_steps": 20,
        "seeds": rows,
        "provenance_policy": (
            "Git revision, paths, timestamps, byte sizes, experiment and run IDs; "
            "no newly computed artifact hashes"
        ),
    }


def ensure_frozen_plan(args: argparse.Namespace) -> dict[str, Any]:
    contract = build_contract(args)
    plan = args.output_root / "experiment_plan.json"
    if args.output_root.exists():
        if not plan.is_file():
            raise RuntimeError(f"Existing output lacks a frozen plan: {args.output_root}")
        existing = _read_json(plan)
        if existing.get("contract") != contract:
            raise RuntimeError(f"Existing frozen plan differs: {plan}")
        return contract
    args.output_root.mkdir(parents=True, exist_ok=False)
    _atomic_json(plan, {"created_at": _utc_now(), "contract": contract})
    return contract


def run_id(contract: Mapping[str, Any], seed: Mapping[str, int]) -> str:
    adapter = str(contract["control_adapter"]).replace("_", "")
    return (
        f"{contract['candidate']}-{adapter}-h{contract['exec_horizon']}-"
        f"env{seed['environment_seed']}-"
        f"policy{seed['policy_seed']}"
    )


def _run_command(
    contract: Mapping[str, Any], seed: Mapping[str, int], output: Path
) -> list[str]:
    return [
        str(contract["python"]),
        str(contract["diagnostic"]),
        "--mode",
        "rollout",
        "--formal-contract",
        "--task",
        str(contract["task"]),
        "--panel",
        str(contract["panel"]),
        "--dataset-root",
        str(contract["dataset_root"]),
        "--robofactory-root",
        str(contract["robofactory_root"]),
        "--gaussian-cache",
        str(contract["gaussian_cache"]),
        "--output-dir",
        str(output),
        "--episode-start",
        str(seed["episode_start"]),
        "--policy-seed",
        str(seed["policy_seed"]),
        "--max-steps",
        str(contract["max_steps"]),
        "--initial-state",
        str(contract["initial_state"]),
        "--exec-horizon",
        str(contract["exec_horizon"]),
        "--control-adapter",
        str(contract["control_adapter"]),
        "--topp-step",
        str(contract["topp_step"]),
        "--max-policy-queries",
        str(contract["max_policy_queries"]),
        "--max-simulator-steps",
        str(contract["max_simulator_steps"]),
        "--checkpoint",
        str(contract["checkpoint"]["path"]),
        "--training-code-commit",
        str(contract["training_code_commit"]),
        "--evaluation-code-commit",
        str(contract["evaluation_code_commit"]),
        "--integrity-mode",
        "metadata_no_hash",
        "--model-project-root",
        str(contract["model_project_root"]),
        "--action-architecture",
        str(contract["action_architecture"]),
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
    ]


def _python_path(contract: Mapping[str, Any], inherited: str | None = None) -> str:
    paths = [
        str(Path(contract["source_root"]) / "src"),
        str(Path(contract["source_root"]) / "experiments/robofactory"),
        str(Path(contract["source_root"])),
        str(contract["policy_lightning_repo"]),
        str(contract["robofactory_root"]),
    ]
    if inherited:
        paths.append(inherited)
    return os.pathsep.join(paths)


def _argv_value(argv: Sequence[str], flag: str) -> str | None:
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    return None if index + 1 >= len(argv) else str(argv[index + 1])


def _completed_output(
    output: Path, seed: Mapping[str, int], contract: Mapping[str, Any]
) -> bool:
    summary_path = output / "summary.json"
    manifest_path = output / "run_manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        return False
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    rollout = summary.get("rollout")
    argv = manifest.get("argv", [])
    return bool(
        summary.get("status") == "COMPLETED"
        and isinstance(rollout, dict)
        and rollout.get("status") == "completed"
        and manifest.get("status") == "terminal"
        and manifest.get("rollout_cell")
        == {
            "initial_state": "raw",
            "exec_horizon": int(contract["exec_horizon"]),
            "control_adapter": contract["control_adapter"],
            "topp_step": float(contract["topp_step"]),
            "max_policy_queries": int(contract["max_policy_queries"]),
            "max_simulator_steps": int(contract["max_simulator_steps"]),
        }
        and manifest.get("training_code_commit")
        == contract["training_code_commit"]
        and manifest.get("evaluation_code_commit")
        == contract["evaluation_code_commit"]
        and manifest.get("episode", {}).get("environment_seed")
        == seed["environment_seed"]
        and manifest.get("episode", {}).get("policy_seed") == seed["policy_seed"]
        and _argv_value(argv, "--checkpoint") == contract["checkpoint"]["path"]
        and _argv_value(argv, "--action-architecture")
        == contract["action_architecture"]
        and _argv_value(argv, "--model-project-root")
        == contract["model_project_root"]
        and "--oracle-intervention" not in argv
    )


def _one_run(
    *,
    output_root: Path,
    contract: Mapping[str, Any],
    seed: Mapping[str, int],
    gpu_pool: "queue.Queue[int]",
) -> dict[str, Any]:
    identity = run_id(contract, seed)
    output = output_root / "runs" / identity
    receipt = output_root / "receipts" / f"{identity}.json"
    if output.exists():
        if _completed_output(output, seed, contract):
            return {"run_id": identity, "status": "skipped_completed"}
        raise RuntimeError(f"Partial or invalid output refuses overwrite: {output}")
    if receipt.exists():
        raise RuntimeError(f"Receipt exists without valid output: {receipt}")
    gpu = gpu_pool.get()
    started = time.monotonic()
    started_at = _utc_now()
    log_path = output_root / "logs" / f"{identity}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _run_command(contract, seed, output)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = _python_path(contract, env.get("PYTHONPATH"))
    driver_paths = [str(contract["nvidia_driver_lib_dir"])]
    if env.get("LD_LIBRARY_PATH"):
        driver_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(driver_paths)
    env["VK_ICD_FILENAMES"] = str(contract["nvidia_vulkan_icd"])
    env["VK_DRIVER_FILES"] = str(contract["nvidia_vulkan_icd"])
    env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(
        contract["nvidia_egl_vendor_json"]
    )
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
        if process.returncode != 0 or not _completed_output(output, seed, contract):
            raise RuntimeError(f"Rollout failed contract: {identity}; see {log_path}")
        return record
    finally:
        gpu_pool.put(gpu)


def run_panel(
    args: argparse.Namespace, contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    gpu_pool: "queue.Queue[int]" = queue.Queue()
    for gpu in args.gpus:
        gpu_pool.put(gpu)
    results = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {
            executor.submit(
                _one_run,
                output_root=args.output_root,
                contract=contract,
                seed=seed,
                gpu_pool=gpu_pool,
            ): run_id(contract, seed)
            for seed in contract["seeds"]
        }
        for future in concurrent.futures.as_completed(futures):
            identity = futures[future]
            try:
                record = future.result()
                results.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
            except Exception as error:  # noqa: BLE001 - preserve every failed cell
                message = f"{identity}: {type(error).__name__}: {error}"
                errors.append(message)
                print(message, file=sys.stderr, flush=True)
    if errors:
        raise RuntimeError("Fixed panel had failed cells:\n" + "\n".join(errors))
    return results


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def aggregate(
    args: argparse.Namespace, contract: Mapping[str, Any]
) -> dict[str, Any]:
    rows = []
    all_complete = True
    for seed in contract["seeds"]:
        identity = run_id(contract, seed)
        output = args.output_root / "runs" / identity
        if not _completed_output(output, seed, contract):
            all_complete = False
            rows.append({"run_id": identity, "operational": False})
            continue
        summary = _read_json(output / "summary.json")
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
                "steps": int(rollout["steps"]),
                "predicted_target_actions": int(
                    rollout["predicted_target_actions"]
                ),
                "planner_fallbacks": int(rollout["planner_fallbacks"]),
                "policy_queries": int(rollout["policy_queries"]),
                "bound_violations": int(
                    rollout["bound_violations"]["total_scalar_violations"]
                ),
                "elapsed_seconds": float(rollout["elapsed_seconds"]),
                "output": str(output),
            }
        )
    operational = [row for row in rows if row["operational"]]
    lifts = [float(row["meat_max_lift_m"]) for row in operational]
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": contract["experiment_id"],
        "candidate": contract["candidate"],
        "control_adapter": contract["control_adapter"],
        "generated_at": _utc_now(),
        "status": "COMPLETE" if all_complete else "INCOMPLETE",
        "expected_runs": EXPECTED_RUNS,
        "operational_runs": len(operational),
        "success_count": sum(bool(row["success"]) for row in operational),
        "closed_loop_success_rate": _mean(
            [float(bool(row["success"])) for row in operational]
        ),
        "true_grasp_count": sum(
            bool(row["robot0_grasp_ever"]) for row in operational
        ),
        "true_grasp_rate": _mean(
            [float(bool(row["robot0_grasp_ever"])) for row in operational]
        ),
        "meat_max_lift_m_mean": _mean(lifts),
        "meat_max_lift_m_max": None if not lifts else max(lifts),
        "rows": rows,
    }
    _atomic_json(args.output_root / "aggregate.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "formal", "summarize"))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--robofactory-root", type=Path, required=True)
    parser.add_argument("--gaussian-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-code-commit", required=True)
    parser.add_argument("--evaluation-code-commit", required=True)
    parser.add_argument("--model-project-root", type=Path, required=True)
    parser.add_argument(
        "--action-architecture",
        choices=("pooled_v1", "gaussian_spatial_v2"),
        required=True,
    )
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--context-file", type=Path, required=True)
    parser.add_argument("--model-cache-root", type=Path, required=True)
    parser.add_argument("--policy-lightning-repo", type=Path, required=True)
    parser.add_argument("--noposplat-checkpoint", type=Path, required=True)
    parser.add_argument("--nvidia-driver-lib-dir", type=Path, required=True)
    parser.add_argument("--nvidia-vulkan-icd", type=Path, required=True)
    parser.add_argument("--nvidia-egl-vendor-json", type=Path, required=True)
    parser.add_argument(
        "--exec-horizon", type=int, choices=(1, 5, 16, 20, 24, 32), default=5
    )
    parser.add_argument(
        "--control-adapter",
        choices=("direct", "official_topp"),
        default="direct",
    )
    parser.add_argument("--topp-step", type=float, default=0.05)
    parser.add_argument("--max-policy-queries", type=int, default=300)
    parser.add_argument("--max-simulator-steps", type=int, default=300)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        raise ValueError(f"GPU IDs must be distinct non-negative integers: {args.gpus}")
    contract = ensure_frozen_plan(args)
    if args.command == "plan":
        print(json.dumps({"status": "PLANNED", "contract": contract}, sort_keys=True))
        return
    if args.command == "formal":
        run_panel(args, contract)
    report = aggregate(args, contract)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
