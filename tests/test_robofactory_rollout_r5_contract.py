from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import numpy as np

from experiments.robofactory import diagnose_place_food_fixed as diagnostic
from experiments.robofactory import fastwam_multi_robot_policy as policy


class R5RolloutContractTests(unittest.TestCase):
    def test_oracle_interventions_replace_exact_requested_components(self) -> None:
        names = ("panda-0", "panda-1")
        policy_action = {
            "panda-0": np.arange(8, dtype=np.float32),
            "panda-1": np.arange(8, 16, dtype=np.float32),
        }
        expert_action = {
            "panda-0": np.arange(100, 108, dtype=np.float32),
            "panda-1": np.arange(108, 116, dtype=np.float32),
        }

        pose = diagnostic.apply_oracle_intervention(
            policy_action, expert_action, names, "robot0_pose"
        )
        np.testing.assert_array_equal(pose["panda-0"][:7], expert_action["panda-0"][:7])
        self.assertEqual(pose["panda-0"][7], policy_action["panda-0"][7])
        np.testing.assert_array_equal(pose["panda-1"], policy_action["panda-1"])

        gripper = diagnostic.apply_oracle_intervention(
            policy_action, expert_action, names, "robot0_gripper"
        )
        np.testing.assert_array_equal(gripper["panda-0"][:7], policy_action["panda-0"][:7])
        self.assertEqual(gripper["panda-0"][7], expert_action["panda-0"][7])
        np.testing.assert_array_equal(gripper["panda-1"], policy_action["panda-1"])

        robot1 = diagnostic.apply_oracle_intervention(
            policy_action, expert_action, names, "robot1_action"
        )
        np.testing.assert_array_equal(robot1["panda-0"], policy_action["panda-0"])
        np.testing.assert_array_equal(robot1["panda-1"], expert_action["panda-1"])
        np.testing.assert_array_equal(policy_action["panda-0"], np.arange(8))

    def test_rollout_grasp_metrics_use_true_grasp_and_meat_lift(self) -> None:
        metrics = diagnostic.rollout_grasp_metrics(
            [
                {"meat_height": 0.12, "robot0_grasping_meat": False},
                {"meat_height": 0.14, "robot0_grasping_meat": True},
                {"meat_height": 0.13, "robot0_grasping_meat": False},
            ]
        )

        self.assertTrue(metrics["robot0_grasp_ever"])
        self.assertEqual(metrics["robot0_grasp_steps"], 1)
        self.assertAlmostEqual(metrics["robot0_grasp_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["meat_max_lift_m"], 0.02)

    def test_oracle_temporal_trace_exhaustion_falls_back_to_policy(self) -> None:
        names = ("panda-0", "panda-1")
        policy_action = {
            "panda-0": np.arange(8, dtype=np.float32),
            "panda-1": np.arange(8, 16, dtype=np.float32),
        }

        executed, applied = diagnostic.select_executed_action(
            policy_action,
            None,
            names,
            "robot0_gripper",
        )

        self.assertFalse(applied)
        np.testing.assert_array_equal(executed["panda-0"], policy_action["panda-0"])
        np.testing.assert_array_equal(executed["panda-1"], policy_action["panda-1"])

    def test_formal_rollout_requires_explicit_oracle_cell(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit --oracle-intervention"):
            diagnostic.validate_formal_rollout_contract(
                max_steps=300,
                initial_state="clean",
                exec_horizon=1,
                oracle_intervention="none",
                initial_state_explicit=True,
                exec_horizon_explicit=True,
                oracle_intervention_explicit=False,
            )

    def test_video_is_staged_locally_and_validated_after_publication(self) -> None:
        import imageio.v2 as imageio
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "local.mp4"
            destination = root / "published.mp4"
            writer = imageio.get_writer(
                source,
                fps=20,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=None,
            )
            try:
                for value in (0, 64, 128):
                    writer.append_data(
                        np.full((32, 64, 3), value, dtype=np.uint8)
                    )
            finally:
                writer.close()

            report = diagnostic._publish_video(
                source, destination, expected_frames=3
            )

            self.assertEqual(report["frames"], 3)
            self.assertEqual(report["frame_shape"], [32, 64, 3])
            self.assertTrue(report["encoding_staged_on_local_disk"])
            self.assertTrue(report["published_readback_validated"])
            self.assertTrue(destination.is_file())
            with self.assertRaises(FileExistsError):
                diagnostic._publish_video(
                    source, destination, expected_frames=3
                )

    def test_artifact_directory_is_published_once_with_exact_readback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "local" / "rollout"
            destination = root / "published" / "rollout"
            source.mkdir(parents=True)
            destination.parent.mkdir()
            (source / "empty.jsonl").touch()
            nested = source / "nested"
            nested.mkdir()
            (nested / "result.json").write_text(
                '{"status":"completed"}\n', encoding="utf-8"
            )

            report = diagnostic._publish_directory(source, destination)

            self.assertEqual(report["files"], 2)
            self.assertEqual(
                report["bytes"],
                (source / "nested/result.json").stat().st_size,
            )
            self.assertFalse(report["high_frequency_writes_on_destination"])
            self.assertTrue(report["staged_on_local_disk"])
            self.assertTrue(report["published_readback_validated"])
            self.assertEqual(
                (destination / "nested/result.json").read_bytes(),
                (source / "nested/result.json").read_bytes(),
            )
            with self.assertRaises(FileExistsError):
                diagnostic._publish_directory(source, destination)

    def test_artifact_directory_rejects_source_symlink_without_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "local"
            destination = root / "published"
            source.mkdir()
            real = root / "real.txt"
            real.write_text("evidence\n", encoding="utf-8")
            (source / "escape.txt").symlink_to(real)

            with self.assertRaisesRegex(RuntimeError, "contains a symlink"):
                diagnostic._publish_directory(source, destination)

            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".published.publishing.*")), [])

    def test_diagnostic_cli_imports_from_a_clean_script_entrypoint(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(root / "experiments/robofactory/diagnose_place_food_fixed.py"),
                "--help",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--formal-contract", result.stdout)

    def test_r5_action_only_model_contract(self) -> None:
        config = policy.compose_r5_action_model_config()

        self.assertEqual(config.training_mode, "action_only_cache")
        self.assertEqual(config.checkpoint_integrity_mode, "metadata_no_hash")
        self.assertFalse(config.load_text_encoder)
        self.assertTrue(config.skip_dit_load_from_pretrain)
        self.assertIsNone(config.action_dit_pretrained_path)
        self.assertTrue(config.action_dit_config.hub_enabled)
        self.assertTrue(config.action_dit_config.enable_gaussian)
        self.assertEqual(float(config.loss.lambda_video), 0.0)
        self.assertEqual(float(config.loss.lambda_action), 1.0)

    def test_metadata_no_hash_stats_and_context_never_call_sha256(self) -> None:
        stats_payload = {
            "normalization_fit": {
                "split": "train",
                "split_seed": 42,
                "val_set_proportion": 0.1,
            },
            "cardinality": {"agent_counts": [2, 3, 4]},
            "action": {"mean": [0.0] * 8, "std": [1.0] * 8},
            "state": {"mean": [0.0] * 18, "std": [1.0] * 18},
        }
        context_payload = {
            "context": torch.zeros(128, 4096, dtype=torch.float32),
            "mask": torch.ones(128, dtype=torch.bool),
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats_path = root / "stats.json"
            context_path = root / "PlaceFood-rf.pt"
            stats_path.write_text(json.dumps(stats_payload), encoding="utf-8")
            torch.save(context_payload, context_path)

            with mock.patch.object(
                policy.hashlib,
                "sha256",
                side_effect=AssertionError("metadata_no_hash must not compute SHA-256"),
            ):
                stats = policy.load_normalization_stats(
                    stats_path,
                    expected_sha256=None,
                    integrity_mode="metadata_no_hash",
                )
                context = policy.load_text_context(
                    None,
                    "PlaceFood-rf",
                    expected_sha256=None,
                    integrity_mode="metadata_no_hash",
                    context_path=context_path,
                )

        self.assertIsNone(stats.sha256)
        self.assertEqual(stats.metadata["path"], str(stats_path.resolve()))
        self.assertEqual(tuple(stats.action_mean.shape), (8,))
        self.assertEqual(tuple(stats.state_mean.shape), (18,))
        self.assertIsNone(context.sha256)
        self.assertEqual(context.metadata["path"], str(context_path.resolve()))
        self.assertEqual(tuple(context.context.shape), (128, 4096))
        self.assertTrue(bool(context.mask.all().item()))

    def test_metadata_no_hash_rejects_hash_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stats_path = Path(directory) / "stats.json"
            stats_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be omitted"):
                policy.load_normalization_stats(
                    stats_path,
                    expected_sha256="0" * 64,
                    integrity_mode="metadata_no_hash",
                )


if __name__ == "__main__":
    unittest.main()
