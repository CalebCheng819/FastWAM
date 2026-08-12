from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.robofactory import run_r5_closedloop_ablations as ablations


class R5ClosedLoopAblationTests(unittest.TestCase):
    def test_matrix_separates_replanning_oracles_and_checkpoints(self) -> None:
        cells = {cell.name: cell for cell in ablations.CELLS}

        self.assertEqual(cells["step1000_h5_policy"].exec_horizon, 5)
        self.assertEqual(cells["step1000_h1_policy"].exec_horizon, 1)
        self.assertEqual(
            {
                cells["step1000_h5_oracle_robot0_pose"].oracle_intervention,
                cells["step1000_h5_oracle_robot0_gripper"].oracle_intervention,
                cells["step1000_h5_oracle_robot1_action"].oracle_intervention,
            },
            {"robot0_pose", "robot0_gripper", "robot1_action"},
        )
        self.assertEqual(cells["step0500_h5_policy"].exec_horizon, 5)
        self.assertEqual(cells["step0500_h1_policy"].checkpoint_step, 500)

    def test_checkpoint_availability_never_substitutes_missing_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "step_000500.pt").write_bytes(b"500")
            (root / "step_001000.pt").write_bytes(b"1000")

            rows = ablations.checkpoint_availability(root)

        availability = {row["step"]: row["available"] for row in rows}
        self.assertEqual(
            availability,
            {250: False, 500: True, 750: False, 1000: True, 2500: False, 5000: False},
        )

    def test_panel_rows_freeze_eight_environment_and_policy_seeds(self) -> None:
        panel = {
            "episodes": [
                {"task": "PlaceFood-rf", "episode_seed": seed}
                for seed in (333183, 333327, 333225, 333180, 333251, 333130, 333167, 333234)
            ]
        }

        rows = ablations._panel_rows(panel)

        self.assertEqual([row["policy_seed"] for row in rows], list(range(10000, 10008)))
        self.assertEqual(rows[0]["episode_start"], 0)
        self.assertEqual(rows[-1]["episode_start"], 7)

    def test_completed_output_requires_terminal_cell_and_seed_identity(self) -> None:
        cell = ablations.Cell("cell", 1000, 1, "none")
        seed = {"environment_seed": 333183, "policy_seed": 10000, "episode_start": 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "step_001000.pt"
            checkpoint.write_bytes(b"checkpoint")
            checkpoint_stat = checkpoint.stat()
            contract = {
                "checkpoint_availability": [
                    {
                        "step": 1000,
                        "path": str(checkpoint),
                        "available": True,
                        "bytes": checkpoint_stat.st_size,
                        "mtime_ns": checkpoint_stat.st_mtime_ns,
                    }
                ]
            }
            (root / "summary.json").write_text(
                json.dumps({"status": "COMPLETED", "rollout": {"status": "completed"}})
            )
            (root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "terminal",
                        "rollout_cell": {
                            "initial_state": "raw",
                            "exec_horizon": 1,
                            "oracle_intervention": "none",
                        },
                        "policy_request": {
                            "checkpoint_path": str(checkpoint),
                            "checkpoint_bytes": checkpoint_stat.st_size,
                            "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
                            "policy_seed": 10000,
                        },
                        "episode": {"environment_seed": 333183},
                    }
                )
            )

            self.assertTrue(ablations._completed_output(root, cell, seed, contract))

            manifest_path = root / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["policy_request"]["policy_seed"] = 10001
            manifest_path.write_text(json.dumps(manifest))
            self.assertFalse(ablations._completed_output(root, cell, seed, contract))

            manifest["policy_request"]["policy_seed"] = 10000
            manifest["policy_request"]["checkpoint_mtime_ns"] += 1
            manifest_path.write_text(json.dumps(manifest))
            self.assertFalse(ablations._completed_output(root, cell, seed, contract))

    def test_command_passes_explicit_formal_oracle_and_horizon(self) -> None:
        contract = {
            "python": "/python",
            "diagnostic": "/diagnostic.py",
            "panel": "/panel.json",
            "dataset_root": "/dataset",
            "robofactory_root": "/robofactory",
            "gaussian_cache": "/gaussian",
            "checkpoint_dir": "/checkpoints",
            "checkpoint_availability": [
                {
                    "step": 1000,
                    "path": "/checkpoints/step_001000.pt",
                    "available": True,
                    "bytes": 123,
                    "mtime_ns": 456,
                }
            ],
            "stats": "/stats.json",
            "context_file": "/context.pt",
            "model_cache_root": "/models",
            "policy_lightning_repo": "/policy-lightning",
            "noposplat_checkpoint": "/noposplat.ckpt",
            "action_horizon": 32,
            "num_inference_steps": 20,
            "initial_state": "raw",
            "max_steps": 300,
        }
        cell = ablations.Cell("oracle", 1000, 5, "robot0_gripper")
        seed = {"environment_seed": 333183, "policy_seed": 10000, "episode_start": 0}

        command = ablations._run_command(contract, cell, seed, Path("/output"))

        self.assertEqual(command[command.index("--exec-horizon") + 1], "5")
        self.assertEqual(
            command[command.index("--oracle-intervention") + 1], "robot0_gripper"
        )
        self.assertIn("--formal-contract", command)


if __name__ == "__main__":
    unittest.main()
