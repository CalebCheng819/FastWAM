#!/usr/bin/env python3
"""Run the fixed SAME8 diagnostic matrix across a bounded GPU device pool."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "fastwam-same8-diagnostic-matrix-v1"


@dataclass(frozen=True)
class Condition:
    name: str
    exec_horizon: int
    oracle_intervention: str


CONDITIONS = (
    Condition("h1_policy", 1, "none"),
    Condition("h1_oracle_robot0_pose", 1, "robot0_pose"),
    Condition("h1_oracle_robot0_gripper", 1, "robot0_gripper"),
    Condition("h1_oracle_robot1_action", 1, "robot1_action"),
    Condition("h5_policy", 5, "none"),
)

MANAGED_FLAGS = (
    "--condition-name",
    "--device",
    "--exec-horizon",
    "--oracle-intervention",
    "--output-dir",
    "--record-video",
    "--no-record-video",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_base_arguments(path: Path) -> list[str]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"Base argument document must be a regular file: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "fastwam-same8-diagnostic-base-argv-v1":
        raise ValueError("Unexpected base argument schema")
    arguments = payload.get("arguments")
    if not isinstance(arguments, list) or not all(
        isinstance(value, str) and value for value in arguments
    ):
        raise ValueError("Base arguments must be a non-empty string list")
    for argument in arguments:
        if any(argument == flag or argument.startswith(f"{flag}=") for flag in MANAGED_FLAGS):
            raise ValueError(f"Base arguments must not set matrix-managed flag: {argument}")
    required = ("--mode", "--panel", "--num-episodes", "--integrity-mode")
    for flag in required:
        if flag not in arguments and not any(
            argument.startswith(f"{flag}=") for argument in arguments
        ):
            raise ValueError(f"Base arguments are missing required flag: {flag}")
    return list(arguments)


def configured_python_executable(path: Path) -> Path:
    """Validate a configured Python without resolving a virtualenv symlink."""

    configured = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if not configured.is_file() or not os.access(configured, os.X_OK):
        raise ValueError(f"Python must be an executable file: {configured}")
    return configured


def condition_command(
    condition: Condition,
    *,
    python: Path,
    evaluator: Path,
    output_root: Path,
    base_arguments: Sequence[str],
) -> list[str]:
    return [
        str(python),
        str(evaluator),
        *base_arguments,
        "--condition-name",
        condition.name,
        "--exec-horizon",
        str(condition.exec_horizon),
        "--oracle-intervention",
        condition.oracle_intervention,
        "--record-video",
        "--device",
        "cuda:0",
        "--output-dir",
        str(output_root / condition.name),
    ]


def _run_condition(
    condition: Condition,
    device: str,
    *,
    python: Path,
    evaluator: Path,
    output_root: Path,
    base_arguments: Sequence[str],
) -> dict[str, Any]:
    command = condition_command(
        condition,
        python=python,
        evaluator=evaluator,
        output_root=output_root,
        base_arguments=base_arguments,
    )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = device
    stdout_path = output_root / f"{condition.name}.stdout.log"
    stderr_path = output_root / f"{condition.name}.stderr.log"
    started_at = _utc_now()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        completed = subprocess.run(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    summary_path = output_root / condition.name / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file() and not summary_path.is_symlink()
        else None
    )
    return {
        "condition_name": condition.name,
        "exec_horizon": condition.exec_horizon,
        "oracle_intervention": condition.oracle_intervention,
        "cuda_visible_devices": device,
        "command": command,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "returncode": completed.returncode,
        "stdout": stdout_path.name,
        "stderr": stderr_path.name,
        "summary": summary,
    }


def aggregate_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: str(item["condition_name"])):
        summary = result["summary"] or {}
        physical = summary.get("physical") or {}
        rows.append(
            {
                "condition_name": result["condition_name"],
                "exec_horizon": result["exec_horizon"],
                "oracle_intervention": result["oracle_intervention"],
                "returncode": result["returncode"],
                "status": summary.get("status"),
                "episodes_completed": summary.get("episodes_completed"),
                "successes": summary.get("successes"),
                "strict_success_rate": summary.get("strict_success_rate"),
                "episodes_grasped": physical.get("episodes_grasped"),
                "grasp_rate": physical.get("grasp_rate"),
                "maximum_meat_lift_m": physical.get("maximum_meat_lift_m"),
                "mean_episode_max_meat_lift_m": physical.get(
                    "mean_episode_max_meat_lift_m"
                ),
                "policy_action_bounds": summary.get("policy_action_bounds"),
                "executed_action_bounds": summary.get("executed_action_bounds"),
                "oracle": summary.get("oracle"),
                "videos_recorded": summary.get("videos_recorded"),
            }
        )
    terminal_pass = all(
        result["returncode"] == 0
        and isinstance(result["summary"], dict)
        and result["summary"].get("status") == "PASS"
        for result in results
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if terminal_pass and len(results) == len(CONDITIONS) else "FAILED",
        "conditions_expected": len(CONDITIONS),
        "conditions_recorded": len(results),
        "finished_at": _utc_now(),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-argv-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path(__file__).with_name("eval_robofactory_multi_robot.py"),
    )
    parser.add_argument("--devices", default="0,1,2,3")
    args = parser.parse_args()

    output_root = args.output_root.expanduser()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Matrix output must not already exist: {output_root}")
    output_root = output_root.resolve()
    python = configured_python_executable(args.python)
    evaluator = args.evaluator.expanduser().resolve(strict=True)
    if evaluator.is_symlink() or not evaluator.is_file():
        raise ValueError(f"Evaluator must be a regular file: {evaluator}")
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("Devices must be a non-empty list without duplicates")
    base_arguments = load_base_arguments(args.base_argv_json)

    output_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": _utc_now(),
        "base_argv_json": str(args.base_argv_json.expanduser().resolve(strict=True)),
        "evaluator": str(evaluator),
        "python": str(python),
        "devices": devices,
        "conditions": [condition.__dict__ for condition in CONDITIONS],
        "integrity_mode": "metadata_no_hash",
    }
    _atomic_json(output_root / "matrix_manifest.json", manifest)

    results: list[dict[str, Any]] = []
    # Run full device-sized waves so the fifth condition cannot start on cuda:0
    # while the first cuda:0 condition is still alive.  A modulo assignment in a
    # single executor does not preserve that exclusivity when a different device
    # finishes first.
    for start in range(0, len(CONDITIONS), len(devices)):
        wave = CONDITIONS[start : start + len(devices)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = {
                executor.submit(
                    _run_condition,
                    condition,
                    devices[index],
                    python=python,
                    evaluator=evaluator,
                    output_root=output_root,
                    base_arguments=base_arguments,
                ): condition
                for index, condition in enumerate(wave)
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                _atomic_json(
                    output_root / "matrix_progress.json", {"results": results}
                )
                print(json.dumps(result, sort_keys=True), flush=True)

    summary = aggregate_results(results)
    _atomic_json(output_root / "matrix_summary.json", summary)
    manifest.update(
        {
            "status": "terminal",
            "finished_at": _utc_now(),
            "matrix_status": summary["status"],
        }
    )
    _atomic_json(output_root / "matrix_manifest.json", manifest)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
