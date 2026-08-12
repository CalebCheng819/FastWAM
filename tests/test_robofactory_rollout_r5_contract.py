from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.robofactory import fastwam_multi_robot_policy as policy


class R5RolloutContractTests(unittest.TestCase):
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
