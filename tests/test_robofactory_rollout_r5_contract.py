from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.robofactory import diagnose_place_food_fixed as diagnostic
from experiments.robofactory import fastwam_multi_robot_policy as policy


class B4RolloutContractTests(unittest.TestCase):
    @staticmethod
    def _split_panel(*, split: str, ordinal: int) -> dict[str, object]:
        fraction = diagnostic._split_fraction_from_ordinal(ordinal, 42)
        return {
            "schema_version": diagnostic.SPLIT_PANEL_SCHEMA,
            "split": split,
            "split_seed": 42,
            "val_set_proportion": 0.1,
            "split_key_scheme": diagnostic.SPLIT_KEY_SCHEME,
            "paired_policy_seeds": [10000],
            "episodes": [
                {
                    "task_name": "PlaceFood-rf",
                    "task_index": 0,
                    "panel_index": 0,
                    "episode_id": 100,
                    "episode_seed": 333219,
                    "source_path": "demos/PlaceFood-rf/example.h5",
                    "source_h5_bytes": 123,
                    "trajectory": "traj_100",
                    "agent_names": ["panda-0", "panda-1"],
                    "global_ordinal": ordinal,
                    "split_fraction": fraction,
                    "split": split,
                }
            ],
        }

    def test_split_panel_recomputes_and_accepts_val_membership(self) -> None:
        panel = self._split_panel(split="val", ordinal=1190)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.json"
            path.write_text(json.dumps(panel), encoding="utf-8")
            loaded = diagnostic._load_panel_nohash(path)

        self.assertEqual(loaded["split"], "val")
        self.assertEqual(loaded["episodes"][0]["global_ordinal"], 1190)

    def test_split_panel_rejects_mislabeled_membership(self) -> None:
        panel = self._split_panel(split="train", ordinal=1190)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.json"
            path.write_text(json.dumps(panel), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Episode split mismatch"):
                diagnostic._load_panel_nohash(path)

    def test_split_panel_rejects_fraction_drift(self) -> None:
        panel = self._split_panel(split="val", ordinal=1190)
        panel["episodes"][0]["split_fraction"] = 0.09
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.json"
            path.write_text(json.dumps(panel), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Split fraction mismatch"):
                diagnostic._load_panel_nohash(path)

    def test_split_panel_rejects_policy_seed_cardinality_drift(self) -> None:
        panel = self._split_panel(split="val", ordinal=1190)
        panel["paired_policy_seeds"] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.json"
            path.write_text(json.dumps(panel), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "length must equal"):
                diagnostic._load_panel_nohash(path)

    def test_split_panel_rejects_index_order_drift(self) -> None:
        panel = self._split_panel(split="val", ordinal=1190)
        panel["episodes"][0]["panel_index"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.json"
            path.write_text(json.dumps(panel), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must match list order"):
                diagnostic._load_panel_nohash(path)

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

    def test_b4_action_only_model_contract(self) -> None:
        config = policy.compose_b4_action_model_config()

        self.assertEqual(config.training_mode, "action_only_cache")
        self.assertEqual(config.checkpoint_integrity_mode, "metadata_no_hash")
        self.assertFalse(config.load_text_encoder)
        self.assertTrue(config.skip_dit_load_from_pretrain)
        self.assertIsNone(config.action_dit_pretrained_path)
        self.assertTrue(config.action_dit_config.hub_enabled)
        self.assertTrue(config.action_dit_config.enable_gaussian)
        self.assertEqual(float(config.loss.lambda_video), 0.0)
        self.assertEqual(float(config.loss.lambda_action), 1.0)
        self.assertEqual(
            policy.B4_TRAINING_CODE_COMMIT,
            "6ad834248f0fbc1d070c9be97627364174af143c",
        )

    def test_r5_config_alias_matches_b4_contract(self) -> None:
        self.assertEqual(
            policy.compose_r5_action_model_config(),
            policy.compose_b4_action_model_config(),
        )

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
