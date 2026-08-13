from __future__ import annotations

import errno
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.robofactory import diagnose_place_food_fixed as diagnostic
from experiments.robofactory import run_fixed_policy_closedloop_panel as panel_runner


class FixedPolicyClosedLoopPanelTests(unittest.TestCase):
    def test_jsonl_append_falls_back_when_append_write_is_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            diagnostic._append_jsonl(path, {"step": 1})

            with mock.patch.object(
                diagnostic.os,
                "write",
                side_effect=OSError(errno.EINVAL, "unsupported append"),
            ):
                diagnostic._append_jsonl(path, {"step": 2})

            self.assertEqual(
                [json.loads(line) for line in path.read_text().splitlines()],
                [{"step": 1}, {"step": 2}],
            )

    def test_jsonl_append_accepts_unsupported_fsync_without_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"

            with mock.patch.object(
                diagnostic.os,
                "fsync",
                side_effect=OSError(errno.EINVAL, "unsupported fsync"),
            ):
                diagnostic._append_jsonl(path, {"step": 1})

            self.assertEqual(
                [json.loads(line) for line in path.read_text().splitlines()],
                [{"step": 1}],
            )

    def test_python_executable_preserves_virtualenv_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime-python"
            runtime.write_text("#!/bin/sh\n", encoding="utf-8")
            runtime.chmod(0o755)
            venv_python = root / "venv-python"
            venv_python.symlink_to(runtime)

            selected = panel_runner._python_executable(venv_python)

            self.assertEqual(selected, venv_python)
            self.assertNotEqual(selected, Path(os.path.realpath(venv_python)))

    def test_panel_rows_preserve_paired_policy_seeds(self) -> None:
        panel = {
            "paired_policy_seeds": [20000 + index for index in range(8)],
            "episodes": [
                {
                    "task_name": "PlaceFood-rf",
                    "panel_index": index,
                    "episode_seed": 30000 + index,
                }
                for index in range(8)
            ],
        }

        rows = panel_runner._panel_rows(panel)

        self.assertEqual([row["policy_seed"] for row in rows], list(range(20000, 20008)))
        self.assertEqual(
            [row["environment_seed"] for row in rows], list(range(30000, 30008))
        )

    def test_rollout_command_uses_new_policy_contract_without_oracle(self) -> None:
        contract = {
            "python": "/python",
            "diagnostic": "/source/diagnostic.py",
            "task": "PlaceFood-rf",
            "panel": "/panel.json",
            "dataset_root": "/dataset",
            "robofactory_root": "/robofactory",
            "gaussian_cache": "/gaussian",
            "checkpoint": {"path": "/checkpoint.pt"},
            "training_code_commit": "train-commit",
            "evaluation_code_commit": "eval-commit",
            "model_project_root": "/model-project",
            "action_architecture": "gaussian_spatial_v2",
            "stats": "/stats.json",
            "context_file": "/context.pt",
            "model_cache_root": "/model-cache",
            "policy_lightning_repo": "/policy-lightning",
            "noposplat_checkpoint": "/noposplat.ckpt",
            "max_steps": 300,
            "initial_state": "raw",
            "exec_horizon": 5,
            "control_adapter": "direct",
            "topp_step": 0.05,
            "max_policy_queries": 300,
            "max_simulator_steps": 300,
            "action_horizon": 32,
            "num_inference_steps": 20,
        }
        command = panel_runner._run_command(
            contract,
            {"episode_start": 3, "policy_seed": 10003},
            Path("/output"),
        )

        self.assertNotIn("--oracle-intervention", command)
        self.assertEqual(
            panel_runner._argv_value(command, "--action-architecture"),
            "gaussian_spatial_v2",
        )
        self.assertEqual(
            panel_runner._argv_value(command, "--training-code-commit"),
            "train-commit",
        )
        self.assertEqual(
            panel_runner._argv_value(command, "--model-project-root"),
            "/model-project",
        )
        self.assertIn("--formal-contract", command)
        self.assertEqual(command.count("metadata_no_hash"), 1)

    def test_run_id_tracks_exec_horizon(self) -> None:
        identity = panel_runner.run_id(
            {
                "candidate": "p2-step1000",
                "control_adapter": "direct",
                "exec_horizon": 1,
            },
            {"environment_seed": 333183, "policy_seed": 10000},
        )

        self.assertEqual(identity, "p2-step1000-direct-h1-env333183-policy10000")

    def test_python_path_contains_explicit_runtime_dependencies(self) -> None:
        contract = {
            "source_root": "/source",
            "policy_lightning_repo": "/policy-lightning",
            "robofactory_root": "/robofactory",
        }

        paths = panel_runner._python_path(contract, "/inherited").split(os.pathsep)

        self.assertEqual(
            paths,
            [
                "/source/src",
                "/source/experiments/robofactory",
                "/source",
                "/policy-lightning",
                "/robofactory",
                "/inherited",
            ],
        )

    def test_completed_output_checks_frozen_checkpoint_and_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = "/checkpoint.pt"
            architecture = "pooled_v1"
            model_project = "/model-project"
            manifest = {
                "status": "terminal",
                "training_code_commit": "train",
                "evaluation_code_commit": "eval",
                "episode": {"environment_seed": 1, "policy_seed": 2},
                "rollout_cell": {
                    "initial_state": "raw",
                    "exec_horizon": 5,
                    "control_adapter": "direct",
                    "topp_step": 0.05,
                    "max_policy_queries": 300,
                    "max_simulator_steps": 300,
                },
                "argv": [
                    "python",
                    "diagnostic.py",
                    "--checkpoint",
                    checkpoint,
                    "--action-architecture",
                    architecture,
                    "--model-project-root",
                    model_project,
                ],
            }
            summary = {
                "status": "COMPLETED",
                "rollout": {"status": "completed"},
            }
            (output / "run_manifest.json").write_text(json.dumps(manifest))
            (output / "summary.json").write_text(json.dumps(summary))
            contract = {
                "training_code_commit": "train",
                "evaluation_code_commit": "eval",
                "checkpoint": {"path": checkpoint},
                "action_architecture": architecture,
                "model_project_root": model_project,
                "exec_horizon": 5,
                "control_adapter": "direct",
                "topp_step": 0.05,
                "max_policy_queries": 300,
                "max_simulator_steps": 300,
            }

            self.assertTrue(
                panel_runner._completed_output(
                    output,
                    {"environment_seed": 1, "policy_seed": 2},
                    contract,
                )
            )
            manifest["argv"].extend(["--oracle-intervention", "none"])
            (output / "run_manifest.json").write_text(json.dumps(manifest))
            self.assertFalse(
                panel_runner._completed_output(
                    output,
                    {"environment_seed": 1, "policy_seed": 2},
                    contract,
                )
            )

            manifest["argv"] = manifest["argv"][:-2]
            (output / "run_manifest.json").write_text(json.dumps(manifest))
            contract["exec_horizon"] = 1
            self.assertFalse(
                panel_runner._completed_output(
                    output,
                    {"environment_seed": 1, "policy_seed": 2},
                    contract,
                )
            )

    def test_grasp_metrics_reports_true_grasp_and_lift(self) -> None:
        snapshots = [
            {"meat_height": 0.10, "robot0_grasping_meat": False},
            {"meat_height": 0.12, "robot0_grasping_meat": True},
            {"meat_height": 0.11, "robot0_grasping_meat": False},
        ]

        metrics = diagnostic.rollout_grasp_metrics(snapshots)

        self.assertTrue(metrics["robot0_grasp_ever"])
        self.assertEqual(metrics["robot0_grasp_steps"], 1)
        self.assertAlmostEqual(metrics["meat_max_lift_m"], 0.02)

    def test_official_topp_synchronizes_paths_and_preserves_grippers(self) -> None:
        class Planner:
            def __init__(self, length: int) -> None:
                self.length = length

            def TOPP(self, path, step, verbose):  # noqa: N802 - upstream API
                positions = np.linspace(path[0], path[1], self.length)
                return None, positions, None, None, self.length * step

        agents = ("panda-0", "panda-1")
        current = {name: np.zeros(7) for name in agents}
        target = {
            "panda-0": np.r_[np.ones(7), -1.0],
            "panda-1": np.r_[np.full(7, 2.0), 1.0],
        }

        actions, record = diagnostic.official_topp_actions(
            planners=[Planner(2), Planner(4)],
            current_qpos=current,
            target_action=target,
            agent_names=agents,
            step=0.05,
        )

        self.assertEqual(len(actions), 4)
        np.testing.assert_allclose(actions[-1]["panda-0"][:7], 1.0)
        np.testing.assert_allclose(actions[-1]["panda-1"][:7], 2.0)
        self.assertEqual(actions[-1]["panda-0"][-1], -1.0)
        self.assertEqual(actions[-1]["panda-1"][-1], 1.0)
        self.assertEqual(record["agents"]["panda-0"]["path_steps"], 2)
        self.assertEqual(record["agents"]["panda-1"]["path_steps"], 4)

    def test_official_topp_falls_back_to_current_arm_qpos(self) -> None:
        class BrokenPlanner:
            def TOPP(self, path, step, verbose):  # noqa: N802 - upstream API
                raise RuntimeError("planner failure")

        actions, record = diagnostic.official_topp_actions(
            planners=[BrokenPlanner()],
            current_qpos={"panda-0": np.arange(7, dtype=np.float64)},
            target_action={"panda-0": np.r_[np.full(7, 9.0), -1.0]},
            agent_names=("panda-0",),
            step=0.05,
        )

        self.assertEqual(len(actions), 1)
        np.testing.assert_allclose(actions[0]["panda-0"][:7], np.arange(7))
        self.assertEqual(actions[0]["panda-0"][-1], -1.0)
        self.assertTrue(record["agents"]["panda-0"]["fallback"])

    def test_formal_official_topp_contract_is_explicit(self) -> None:
        contract = diagnostic.validate_formal_rollout_contract(
            max_steps=300,
            max_policy_queries=60,
            max_simulator_steps=30000,
            initial_state="raw",
            exec_horizon=32,
            control_adapter="official_topp",
            topp_step=0.05,
            initial_state_explicit=True,
            exec_horizon_explicit=True,
            control_adapter_explicit=True,
            max_policy_queries_explicit=True,
            max_simulator_steps_explicit=True,
        )

        self.assertEqual(contract["control_adapter"], "official_topp")
        self.assertEqual(contract["max_policy_queries"], 60)


if __name__ == "__main__":
    unittest.main()
