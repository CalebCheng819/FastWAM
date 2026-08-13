from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.robofactory import diagnose_place_food_fixed as diagnostic
from experiments.robofactory import run_fixed_policy_closedloop_panel as panel_runner


class FixedPolicyClosedLoopPanelTests(unittest.TestCase):
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
                "rollout_cell": {"initial_state": "raw", "exec_horizon": 5},
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


if __name__ == "__main__":
    unittest.main()
