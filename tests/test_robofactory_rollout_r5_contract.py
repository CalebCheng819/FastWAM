from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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

    def test_regular_file_is_staged_and_validated_after_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (("trace.jsonl", b'{"step":0}\n'), ("empty.jsonl", b""))
            for name, payload in cases:
                source = root / f"local-{name}"
                destination = root / f"published-{name}"
                source.write_bytes(payload)

                report = diagnostic._publish_staged_file(source, destination)

                self.assertEqual(report["bytes"], len(payload))
                self.assertTrue(report["staged_on_local_disk"])
                self.assertTrue(report["published_readback_validated"])
                self.assertEqual(destination.read_bytes(), payload)
                with self.assertRaises(FileExistsError):
                    diagnostic._publish_staged_file(source, destination)

    def test_formal_teacher_targets_stop_at_action_268(self) -> None:
        lengths = [
            diagnostic.teacher_target_length(
                action_count=283,
                timestep=timestep,
                horizon=5,
                formal_contract=True,
            )
            for timestep in range(5, 268)
        ]

        self.assertEqual(sum(lengths), 1305)
        self.assertEqual(lengths[-5:], [5, 4, 3, 2, 1])
        self.assertEqual(
            diagnostic.teacher_target_length(
                action_count=283,
                timestep=267,
                horizon=5,
                formal_contract=False,
            ),
            5,
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
        self.assertIn("--training-code-commit", result.stdout)
        self.assertIn("--evaluation-code-commit", result.stdout)
        self.assertIn("--model-project-root", result.stdout)
        self.assertIn("--action-architecture", result.stdout)
        self.assertIn("expert-replay", result.stdout)

    def test_expert_replay_cli_does_not_require_model_arguments(self) -> None:
        parsed = diagnostic._parser().parse_args(
            [
                "--mode",
                "expert-replay",
                "--panel",
                "/tmp/panel.json",
                "--dataset-root",
                "/tmp/dataset",
                "--robofactory-root",
                "/tmp/robofactory",
                "--output-dir",
                "/tmp/output",
                "--initial-state",
                "raw",
            ]
        )

        self.assertEqual(parsed.mode, "expert-replay")
        self.assertIsNone(parsed.checkpoint)
        self.assertIsNone(parsed.gaussian_cache)
        self.assertIsNone(parsed.noposplat_checkpoint)

    def test_gaussian_spatial_config_preserves_action_only_contract(self) -> None:
        config = policy.compose_gaussian_spatial_action_model_config()

        self.assertEqual(config.training_mode, "action_only_cache")
        self.assertEqual(config.checkpoint_integrity_mode, "metadata_no_hash")
        self.assertTrue(config.action_dit_config.enable_gaussian)
        self.assertEqual(
            config.action_dit_config.gaussian_conditioning_mode,
            "spatial_cross_attention",
        )
        self.assertEqual(config.action_dit_config.gaussian_residual_floor, 0.1)
        self.assertEqual(config.action_dit_config.gaussian_attention_temperature, 0.1)
        self.assertEqual(config.loss.lambda_video, 0.0)

    def test_task_conditioned_relation_config_preserves_p6_contract(self) -> None:
        config = policy.compose_task_conditioned_relation_action_model_config()

        self.assertEqual(config.training_mode, "action_only_cache")
        self.assertEqual(config.checkpoint_integrity_mode, "metadata_no_hash")
        self.assertTrue(config.action_dit_config.enable_gaussian)
        self.assertEqual(
            config.action_dit_config.gaussian_conditioning_mode,
            "task_conditioned_relation_attention",
        )
        self.assertEqual(config.action_dit_config.gaussian_residual_floor, 0.1)
        self.assertEqual(config.action_dit_config.gaussian_attention_temperature, 0.1)
        self.assertEqual(config.action_dit_config.gaussian_relation_num_heads, 8)
        self.assertEqual(config.loss.lambda_video, 0.0)

    def test_task_conditioned_relation_runtime_contract_is_fail_closed(self) -> None:
        valid = SimpleNamespace(
            gaussian_conditioning_mode="task_conditioned_relation_attention",
            gaussian_residual_floor=0.1,
            gaussian_attention_temperature=0.1,
            gaussian_relation_num_heads=8,
            gaussian_relation_attention=object(),
            gaussian_relation_gate=object(),
            gaussian_query_norm=object(),
            gaussian_key_norm=object(),
        )

        policy._validate_gaussian_action_contract(
            valid,
            "task_conditioned_relation_v3",
        )
        valid.gaussian_relation_attention = None
        with self.assertRaisesRegex(RuntimeError, "P7 task-conditioned"):
            policy._validate_gaussian_action_contract(
                valid,
                "task_conditioned_relation_v3",
            )

    def test_expert_replay_launcher_omits_policy_inputs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher = (
            root
            / ".research-workflow"
            / "experiments"
            / "FASTWAM-MR-N2-PLACEFOOD-EXPERT-REPLAY-R1-20260813"
            / "run_one.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("--mode expert-replay", launcher)
        self.assertIn("--initial-state raw", launcher)
        self.assertIn("--evaluation-code-commit", launcher)
        self.assertNotIn("--checkpoint", launcher)
        self.assertNotIn("--gaussian-cache", launcher)
        self.assertNotIn("--noposplat-checkpoint", launcher)

    def test_expert_action_preserves_agent_order_and_values(self) -> None:
        import numpy as np

        actions = {
            "panda-0": np.arange(24, dtype=np.float32).reshape(3, 8),
            "panda-1": np.arange(100, 124, dtype=np.float32).reshape(3, 8),
        }

        selected = diagnostic._expert_action_at(
            actions, ("panda-1", "panda-0"), timestep=2
        )

        self.assertEqual(tuple(selected), ("panda-1", "panda-0"))
        np.testing.assert_array_equal(selected["panda-1"], actions["panda-1"][2])
        np.testing.assert_array_equal(selected["panda-0"], actions["panda-0"][2])
        with self.assertRaisesRegex(IndexError, "outside"):
            diagnostic._expert_action_at(actions, ("panda-0", "panda-1"), 3)

    def test_formal_expert_replay_requires_explicit_raw_state(self) -> None:
        contract = diagnostic.validate_formal_expert_replay_contract(
            max_steps=300,
            initial_state="raw",
            initial_state_explicit=True,
            evaluation_code_commit="0123456789abcdef0123456789abcdef01234567",
        )

        self.assertEqual(contract["action_source"], "stored_h5_expert")
        self.assertFalse(contract["policy_initialized"])
        with self.assertRaisesRegex(ValueError, "explicit --initial-state raw"):
            diagnostic.validate_formal_expert_replay_contract(
                max_steps=300,
                initial_state="clean",
                initial_state_explicit=True,
                evaluation_code_commit="0123456789abcdef0123456789abcdef01234567",
            )
        with self.assertRaisesRegex(ValueError, "evaluation-code-commit"):
            diagnostic.validate_formal_expert_replay_contract(
                max_steps=300,
                initial_state="raw",
                initial_state_explicit=True,
                evaluation_code_commit=None,
            )

    def test_b4_launcher_pins_and_validates_vulkan_runtime(self) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher = (
            root
            / ".research-workflow/experiments"
            / "FASTWAM-MR-B4-N2-PLACEFOOD-CLOSEDLOOP-EVAL-20260813"
            / "run_one.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("FASTWAM_NVIDIA_GRAPHICS_ROOT", launcher)
        self.assertIn("nvidia-graphics-570.153.02", launcher)
        self.assertIn('"$vulkan_icd"', launcher)
        self.assertIn('"$egl_vendor"', launcher)
        self.assertIn('"$graphics_driver_lib"', launcher)
        self.assertIn('export VK_ICD_FILENAMES="$vulkan_icd"', launcher)
        self.assertIn('export VK_DRIVER_FILES="$vulkan_icd"', launcher)
        self.assertIn('export __GLX_VENDOR_LIBRARY_NAME=nvidia', launcher)
        self.assertIn(
            'export __EGL_VENDOR_LIBRARY_FILENAMES="$egl_vendor"', launcher
        )
        self.assertIn(policy.B4_TRAINING_CODE_COMMIT, launcher)

    def test_b4_batch_runner_supports_actual_gpu_count(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner = (
            root
            / ".research-workflow/experiments"
            / "FASTWAM-MR-B4-N2-PLACEFOOD-CLOSEDLOOP-EVAL-20260813"
            / "run_remaining.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("FASTWAM_EVAL_GPU_COUNT", runner)
        self.assertIn("nvidia-smi --list-gpus", runner)
        self.assertIn('for ((gpu = 0; gpu < gpu_count; gpu++))', runner)
        self.assertIn("--expected-checkpoint", runner)
        self.assertIn("--expected-training-code-commit", runner)
        self.assertIn("step_002500.pt", runner)

    def test_r5_launcher_pins_vulkan_and_training_commit(self) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher = (
            root
            / ".research-workflow/experiments"
            / "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-TRAINLAYOUT-EVAL-R5-20260813"
            / "run_one.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("FASTWAM_NVIDIA_GRAPHICS_ROOT", launcher)
        self.assertIn('export VK_ICD_FILENAMES="$vulkan_icd"', launcher)
        self.assertIn("1a690ab49246cbeb841618a86b5bd546f93ddd40", launcher)
        self.assertIn("/oss-chengjuntao/*", launcher)

    def test_r5_batch_runner_supports_actual_gpu_count(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner = (
            root
            / ".research-workflow/experiments"
            / "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-TRAINLAYOUT-EVAL-R5-20260813"
            / "run_remaining.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("FASTWAM_EVAL_GPU_COUNT", runner)
        self.assertIn("FASTWAM_EVAL_GPU_IDS", runner)
        self.assertIn("nvidia-smi --list-gpus", runner)
        self.assertIn('for ((worker = 0; worker < gpu_count; worker++))', runner)
        self.assertIn('gpu=${gpu_ids[$worker]}', runner)
        self.assertIn('index = worker', runner)
        self.assertIn("--expected-training-code-commit", runner)

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

    def test_metadata_no_hash_adapts_legacy_runtime_without_silent_fallback(self) -> None:
        def legacy_factory(model_id):
            return model_id

        config = policy.compose_gaussian_spatial_action_model_config()
        adapted, legacy_mode = policy._adapt_metadata_no_hash_config_for_runtime(
            config,
            legacy_factory,
        )

        self.assertTrue(legacy_mode)
        self.assertNotIn("checkpoint_integrity_mode", adapted)
        self.assertEqual(config.checkpoint_integrity_mode, "metadata_no_hash")
        self.assertEqual(
            adapted.action_dit_config.gaussian_conditioning_mode,
            "spatial_cross_attention",
        )

        class LegacyModel:
            _checkpoint_provenance_mode = "sha256"

            def _checkpoint_provenance_mode_value(self):
                return self._checkpoint_provenance_mode

        model = LegacyModel()
        policy._activate_legacy_metadata_no_hash_mode(model)
        self.assertEqual(model._checkpoint_provenance_mode, "stat_cmp")

    def test_metadata_no_hash_keeps_modern_runtime_config(self) -> None:
        def modern_factory(model_id, checkpoint_integrity_mode="sha256"):
            return model_id, checkpoint_integrity_mode

        config = policy.compose_gaussian_spatial_action_model_config()
        adapted, legacy_mode = policy._adapt_metadata_no_hash_config_for_runtime(
            config,
            modern_factory,
        )

        self.assertFalse(legacy_mode)
        self.assertIs(adapted, config)
        self.assertEqual(adapted.checkpoint_integrity_mode, "metadata_no_hash")

    def test_metadata_no_hash_keeps_runtime_teacher_with_modern_interface(self) -> None:
        class ModernTeacher:
            def __init__(self, *, integrity_mode="sha256"):
                self.integrity_mode = integrity_mode

        selected, compatibility = policy._select_external_teacher_class(
            ModernTeacher,
            "metadata_no_hash",
        )

        self.assertIs(selected, ModernTeacher)
        self.assertEqual(compatibility, "runtime_native")

    def test_metadata_no_hash_isolates_evaluator_teacher_from_legacy_runtime(self) -> None:
        class LegacyTeacher:
            def __init__(self, *, checkpoint_sha256):
                self.checkpoint_sha256 = checkpoint_sha256

        class EvaluatorTeacher:
            def __init__(self, *, integrity_mode="sha256"):
                self.integrity_mode = integrity_mode

        with mock.patch.object(
            policy,
            "_load_evaluator_metadata_no_hash_teacher",
            return_value=EvaluatorTeacher,
        ) as loader:
            selected, compatibility = policy._select_external_teacher_class(
                LegacyTeacher,
                "metadata_no_hash",
            )

        loader.assert_called_once_with()
        self.assertIs(selected, EvaluatorTeacher)
        self.assertEqual(
            compatibility,
            "evaluator_isolated_metadata_no_hash",
        )

    def test_sha256_mode_preserves_legacy_runtime_teacher(self) -> None:
        class LegacyTeacher:
            def __init__(self, *, checkpoint_sha256):
                self.checkpoint_sha256 = checkpoint_sha256

        selected, compatibility = policy._select_external_teacher_class(
            LegacyTeacher,
            "sha256",
        )

        self.assertIs(selected, LegacyTeacher)
        self.assertEqual(compatibility, "runtime_legacy_sha256")

    def test_evaluator_teacher_loader_is_source_pinned_and_no_hash_capable(self) -> None:
        teacher = policy._load_evaluator_metadata_no_hash_teacher()

        self.assertTrue(policy._supports_keyword(teacher, "integrity_mode"))
        self.assertEqual(
            Path(policy.inspect.getfile(teacher)).resolve(),
            (
                Path(policy.PROJECT_ROOT)
                / "src/fastwam/datasets/gaussian_cache/teacher.py"
            ).resolve(),
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
