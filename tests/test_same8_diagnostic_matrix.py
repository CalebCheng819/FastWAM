from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from experiments.robofactory.run_same8_diagnostic_matrix import (
    CONDITIONS,
    aggregate_results,
    condition_command,
    configured_python_executable,
    load_base_arguments,
)


def _base_document(path: Path, arguments: list[str]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "fastwam-same8-diagnostic-base-argv-v1",
                "arguments": arguments,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_base_arguments_reject_every_matrix_managed_override(tmp_path: Path):
    required = [
        "--mode",
        "fastwam",
        "--panel",
        "panel.json",
        "--num-episodes",
        "8",
        "--integrity-mode",
        "metadata_no_hash",
    ]
    path = _base_document(tmp_path / "base.json", required + ["--exec-horizon=7"])

    with pytest.raises(ValueError, match="matrix-managed flag"):
        load_base_arguments(path)


def test_fixed_matrix_contains_same8_horizons_and_three_oracles():
    assert [(item.name, item.exec_horizon, item.oracle_intervention) for item in CONDITIONS] == [
        ("h1_policy", 1, "none"),
        ("h1_oracle_robot0_pose", 1, "robot0_pose"),
        ("h1_oracle_robot0_gripper", 1, "robot0_gripper"),
        ("h1_oracle_robot1_action", 1, "robot1_action"),
        ("h5_policy", 5, "none"),
    ]


def test_condition_command_pins_video_and_condition_managed_contract(tmp_path: Path):
    command = condition_command(
        CONDITIONS[1],
        python=Path("/usr/bin/python3"),
        evaluator=Path("/repo/eval.py"),
        output_root=tmp_path,
        base_arguments=["--mode", "fastwam"],
    )

    assert "--record-video" in command
    assert command[command.index("--exec-horizon") + 1] == "1"
    assert command[command.index("--oracle-intervention") + 1] == "robot0_pose"
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--condition-name") + 1] == CONDITIONS[1].name


def test_configured_python_preserves_virtualenv_symlink(tmp_path: Path):
    virtualenv_python = tmp_path / "venv" / "bin" / "python"
    virtualenv_python.parent.mkdir(parents=True)
    virtualenv_python.symlink_to(Path(sys.executable).resolve())

    configured = configured_python_executable(virtualenv_python)
    command = condition_command(
        CONDITIONS[0],
        python=configured,
        evaluator=Path("/repo/eval.py"),
        output_root=tmp_path / "output",
        base_arguments=["--mode", "fastwam"],
    )

    assert configured == virtualenv_python
    assert os.path.islink(configured)
    assert command[0] == str(virtualenv_python)


def test_configured_python_rejects_non_executable(tmp_path: Path):
    candidate = tmp_path / "python"
    candidate.write_text("#!/bin/sh\n", encoding="utf-8")
    candidate.chmod(0o644)

    with pytest.raises(ValueError, match="executable file"):
        configured_python_executable(candidate)


def test_matrix_aggregate_exposes_closed_loop_and_diagnostic_metrics():
    results = []
    for condition in CONDITIONS:
        results.append(
            {
                "condition_name": condition.name,
                "exec_horizon": condition.exec_horizon,
                "oracle_intervention": condition.oracle_intervention,
                "returncode": 0,
                "summary": {
                    "status": "PASS",
                    "episodes_completed": 8,
                    "successes": 1,
                    "strict_success_rate": 0.125,
                    "physical": {
                        "episodes_grasped": 2,
                        "grasp_rate": 0.25,
                        "maximum_meat_lift_m": 0.08,
                        "mean_episode_max_meat_lift_m": 0.03,
                    },
                    "policy_action_bounds": {"scalar_violations": 4},
                    "executed_action_bounds": {"scalar_violations": 2},
                    "oracle": {"coverage_rate": 1.0},
                    "videos_recorded": 8,
                },
            }
        )

    summary = aggregate_results(results)

    assert summary["status"] == "PASS"
    assert len(summary["rows"]) == 5
    assert summary["rows"][0]["episodes_grasped"] == 2
    assert summary["rows"][0]["maximum_meat_lift_m"] == pytest.approx(0.08)
