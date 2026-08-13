from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "launch_pose_focus_3x8_dlc.sh"
RENDERER = REPO / "scripts" / "render_pose_focus_3x8_dlc_job.py"
STAT_CMP_HELPER = REPO / "scripts" / "b4_stat_cmp_cache.py"
TASK_NAME = "robofactory_placefood_pose_focus_r5_224_5e-6.yaml"
P2_TASK_NAME = "robofactory_placefood_pose_phase_x0_r5_224_5e-6.yaml"
P4_TASK_NAME = "robofactory_placefood_gaussian_spatial_p4_224_5e-6.yaml"
SCALE_NAME = "robofactory_multi_robot_24gpu_pose_focus.yaml"
R5_SOURCE_WEIGHT = (
    "/oss-chengjuntao/artifacts/fastwam-action-n234-formal-r5-20260812/"
    "fastwam-act-n2-placefood-1k-s42-r5-20260812/checkpoints/weights/step_001000.pt"
)
P1_SOURCE_WEIGHT = (
    "/oss-chengjuntao/artifacts/fastwam-placefood-posefocus-r5-s42-24g-r2-20260813/"
    "checkpoints/weights/step_001000.pt"
)


class PoseFocusLauncherTests(unittest.TestCase):
    def fixture(self, root: Path) -> dict[str, str]:
        repo = root / "repo"
        (repo / "scripts" / "accelerate_configs").mkdir(parents=True)
        (repo / "scripts" / "train.py").write_text("pass\n", encoding="utf-8")
        (repo / "scripts" / "b4_stat_cmp_cache.py").write_bytes(
            STAT_CMP_HELPER.read_bytes()
        )
        (repo / "scripts" / "accelerate_configs" / "accelerate_zero2_ds.yaml").write_text(
            "compute_environment: LOCAL_MACHINE\n", encoding="utf-8"
        )
        (repo / "configs" / "task").mkdir(parents=True)
        (repo / "configs" / "scale").mkdir(parents=True)
        (repo / "configs" / "task" / TASK_NAME).write_bytes(
            (REPO / "configs" / "task" / TASK_NAME).read_bytes()
        )
        (repo / "configs" / "task" / P2_TASK_NAME).write_bytes(
            (REPO / "configs" / "task" / P2_TASK_NAME).read_bytes()
        )
        (repo / "configs" / "task" / P4_TASK_NAME).write_bytes(
            (REPO / "configs" / "task" / P4_TASK_NAME).read_bytes()
        )
        (repo / "configs" / "scale" / SCALE_NAME).write_bytes(
            (REPO / "configs" / "scale" / SCALE_NAME).read_bytes()
        )
        (repo / "src" / "fastwam").mkdir(parents=True)
        (repo / "src" / "fastwam" / "__init__.py").write_text(
            '"""Pose-focus launcher fixture."""\n', encoding="utf-8"
        )
        (repo / "src" / "fastwam" / "trainer.py").write_text(
            "class Trainer:\n    pass\n", encoding="utf-8"
        )

        dataset = root / "dataset"
        text_cache = dataset / "text"
        gaussian = root / "gaussian"
        text_cache.mkdir(parents=True)
        gaussian.mkdir()
        stats = dataset / "stats.json"
        stats.write_text("{}\n", encoding="utf-8")
        weight = root / "step_001000.pt"
        weight.write_bytes(b"fixture-r5-weight")
        run_id = "pose-focus-launcher-unit"
        env = os.environ.copy()
        env.update(
            {
                "RUN_ID": run_id,
                "FASTWAM_POSE_FOCUS_ATTEMPT_ID": "attempt-1",
                "FASTWAM_POSE_FOCUS_REPO_ROOT": str(repo),
                "FASTWAM_POSE_FOCUS_OUTPUT_DIR": str(root / "output"),
                "FASTWAM_POSE_FOCUS_SOURCE_WEIGHT": str(weight),
                "FASTWAM_POSE_FOCUS_SOURCE_WEIGHT_BYTES": str(weight.stat().st_size),
                "FASTWAM_POSE_FOCUS_DATASET_ROOT": str(dataset),
                "FASTWAM_POSE_FOCUS_STATS_PATH": str(stats),
                "FASTWAM_POSE_FOCUS_TEXT_CACHE_DIR": str(text_cache),
                "FASTWAM_POSE_FOCUS_GAUSSIAN_CACHE_DIR": str(gaussian),
                "FASTWAM_POSE_FOCUS_PYTHON": sys.executable,
                "FASTWAM_POSE_FOCUS_OFFLINE_ENV_READY": "1",
                "FASTWAM_POSE_FOCUS_TEST_MODE": "1",
                "FASTWAM_POSE_FOCUS_DRY_RUN": "1",
                "WORLD_SIZE": "3",
                "RANK": "0",
                "NPROC_PER_NODE": "8",
                "LOCAL_RANK": "0",
                "MASTER_ADDR": "10.20.30.40",
                "MASTER_PORT": "29500",
            }
        )
        return env

    @staticmethod
    def committed_launcher_bundle(root: Path) -> tuple[Path, str, bytes]:
        source = root / "pose-focus-bundle-source"
        (source / "scripts").mkdir(parents=True)
        launcher_bytes = LAUNCHER.read_bytes() + b"\n# bundle-test-marker\n"
        (source / "scripts" / LAUNCHER.name).write_bytes(launcher_bytes)
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(
            ["git", "-C", str(source), "config", "user.name", "Launcher Test"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "config",
                "user.email",
                "launcher@example.invalid",
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(source), "commit", "-qm", "launcher fixture"],
            check=True,
        )
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        bundle = root / "pose-focus-source.bundle"
        subprocess.run(
            ["git", "-C", str(source), "bundle", "create", str(bundle), "HEAD"],
            check=True,
        )
        return bundle, commit, launcher_bytes

    def test_dry_run_resolves_r5_pose_focus_world24_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            result = subprocess.run(
                ["bash", str(LAUNCHER)],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = result.stdout
            self.assertIn("--num_machines 3", output)
            self.assertIn("--num_processes 24", output)
            self.assertIn("--machine_rank 0", output)
            self.assertIn("task=robofactory_placefood_pose_focus_r5_224_5e-6", output)
            self.assertIn("+scale=robofactory_multi_robot_24gpu_pose_focus", output)
            self.assertIn("POSE_FOCUS source import gate:", output)
            self.assertIn(
                "POSE_FOCUS runtime provenance binding: "
                "FASTWAM_B4_ATTEMPT_ID=attempt-1",
                output,
            )
            self.assertNotIn("/checkpoints/state/", output)

    def test_dry_run_resolves_p2_phase_clean_x0_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env["FASTWAM_POSE_FOCUS_TASK_PROFILE"] = P2_TASK_NAME.removesuffix(".yaml")
            result = subprocess.run(
                ["bash", str(LAUNCHER)],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "task=robofactory_placefood_pose_phase_x0_r5_224_5e-6",
                result.stdout,
            )

    def test_dry_run_resolves_p4_gaussian_spatial_upgrade_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env["FASTWAM_POSE_FOCUS_TASK_PROFILE"] = P4_TASK_NAME.removesuffix(".yaml")
            result = subprocess.run(
                ["bash", str(LAUNCHER)],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "task=robofactory_placefood_gaussian_spatial_p4_224_5e-6",
                result.stdout,
            )

    def test_rejects_conflicting_runtime_attempt_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env["FASTWAM_B4_ATTEMPT_ID"] = "different-attempt"
            result = subprocess.run(
                ["bash", str(LAUNCHER)],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "FASTWAM_B4_ATTEMPT_ID conflicts with "
                "FASTWAM_POSE_FOCUS_ATTEMPT_ID",
                result.stderr,
            )

    def test_renderer_pins_priority7_3x8_and_oss_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job.json"
            bundle, commit, launcher_bytes = self.committed_launcher_bundle(root)
            run_id = "fastwam-placefood-pose-focus-test"
            command = [
                sys.executable,
                str(RENDERER),
                "--run-id",
                run_id,
                "--attempt-id",
                "attempt-1",
                "--output",
                str(output),
                "--bootstrap-script",
                "/oss-chengjuntao/source/bootstrap.sh",
                "--offline-env-source-root",
                "/oss-chengjuntao/offline-env",
                "--offline-env-manifest",
                "/oss-chengjuntao/offline-env/manifest.json",
                "--offline-code-commit",
                "4" * 40,
                "--offline-source-bundle-relative-path",
                "source/FastWAM.bundle",
                "--base-python",
                "/opt/conda/bin/python3.10",
                "--pose-focus-source-bundle",
                str(bundle),
                "--pose-focus-code-commit",
                commit,
                "--task-profile",
                P2_TASK_NAME.removesuffix(".yaml"),
                "--allow-local-bundle-for-tests",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(output.read_text(encoding="utf-8"))
            request = manifest["request"]
            spec = request["JobSpecs"][0]
            self.assertTrue(manifest["dry_run"])
            self.assertTrue(manifest["submission_not_performed"])
            self.assertEqual(request["Priority"], 7)
            self.assertEqual(spec["PodCount"], 3)
            self.assertEqual(spec["ResourceConfig"]["GPU"], "8")
            self.assertEqual(request["SuccessPolicy"], "AllWorkers")
            self.assertEqual(
                request["Envs"]["FASTWAM_POSE_FOCUS_OUTPUT_DIR"],
                f"/oss-chengjuntao/artifacts/{run_id}",
            )
            self.assertEqual(request["Envs"]["FASTWAM_POSE_FOCUS_CODE_COMMIT"], commit)
            self.assertEqual(
                request["Envs"]["FASTWAM_POSE_FOCUS_TASK_PROFILE"],
                P2_TASK_NAME.removesuffix(".yaml"),
            )
            self.assertEqual(
                request["Settings"]["Tags"]["objective"],
                "robot0-phase-clean-x0",
            )
            self.assertEqual(
                request["Envs"]["FASTWAM_POSE_FOCUS_SOURCE_WEIGHT"],
                R5_SOURCE_WEIGHT,
            )
            self.assertEqual(
                request["Envs"]["FASTWAM_POSE_FOCUS_SOURCE_WEIGHT_BYTES"],
                "12047407619",
            )
            self.assertEqual(
                request["Settings"]["Tags"]["initialization"],
                "R5-action-step1000-weights-only",
            )
            self.assertNotIn("FASTWAM_POSE_FOCUS_TEST_MODE", request["Envs"])
            self.assertEqual(
                {(item["MountPath"], item["MountAccess"]) for item in request["DataSources"]},
                {("/cpfs/user/chengjuntao", "RO"), ("/oss-chengjuntao", "RW")},
            )
            self.assertEqual(base64.b64decode(manifest["launcher_payload_base64"]), launcher_bytes)

    def test_renderer_selects_audited_p1_continuation_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job.json"
            bundle, commit, _ = self.committed_launcher_bundle(root)
            command = [
                sys.executable,
                str(RENDERER),
                "--run-id",
                "fastwam-placefood-p1-continuation-test",
                "--attempt-id",
                "attempt-1",
                "--output",
                str(output),
                "--bootstrap-script",
                "/oss-chengjuntao/source/bootstrap.sh",
                "--offline-env-source-root",
                "/oss-chengjuntao/offline-env",
                "--offline-env-manifest",
                "/oss-chengjuntao/offline-env/manifest.json",
                "--offline-code-commit",
                "4" * 40,
                "--offline-source-bundle-relative-path",
                "source/FastWAM.bundle",
                "--base-python",
                "/opt/conda/bin/python3.10",
                "--pose-focus-source-bundle",
                str(bundle),
                "--pose-focus-code-commit",
                commit,
                "--source-weight",
                P1_SOURCE_WEIGHT,
                "--allow-local-bundle-for-tests",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(output.read_text(encoding="utf-8"))
            request = manifest["request"]
            self.assertEqual(
                request["Envs"]["FASTWAM_POSE_FOCUS_SOURCE_WEIGHT"],
                P1_SOURCE_WEIGHT,
            )
            self.assertEqual(
                request["Settings"]["Tags"]["initialization"],
                "P1-pose-focus-step1000-weights-only",
            )
            self.assertEqual(manifest["source_weight"]["path"], P1_SOURCE_WEIGHT)
            self.assertEqual(manifest["source_weight"]["bytes"], 12047407619)

    def test_renderer_selects_p4_gaussian_spatial_objective(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job.json"
            bundle, commit, _ = self.committed_launcher_bundle(root)
            command = [
                sys.executable,
                str(RENDERER),
                "--run-id",
                "fastwam-placefood-gaussian-spatial-p4-test",
                "--attempt-id",
                "attempt-1",
                "--output",
                str(output),
                "--bootstrap-script",
                "/oss-chengjuntao/source/bootstrap.sh",
                "--offline-env-source-root",
                "/oss-chengjuntao/offline-env",
                "--offline-env-manifest",
                "/oss-chengjuntao/offline-env/manifest.json",
                "--offline-code-commit",
                "4" * 40,
                "--offline-source-bundle-relative-path",
                "source/FastWAM.bundle",
                "--base-python",
                "/opt/conda/bin/python3.10",
                "--pose-focus-source-bundle",
                str(bundle),
                "--pose-focus-code-commit",
                commit,
                "--task-profile",
                P4_TASK_NAME.removesuffix(".yaml"),
                "--source-weight",
                P1_SOURCE_WEIGHT,
                "--allow-local-bundle-for-tests",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            request = json.loads(output.read_text(encoding="utf-8"))["request"]
            self.assertEqual(
                request["Settings"]["Tags"]["objective"],
                "robot0-gaussian-spatial-cross-attention",
            )
            self.assertEqual(
                request["Envs"]["FASTWAM_POSE_FOCUS_SOURCE_WEIGHT"],
                P1_SOURCE_WEIGHT,
            )


if __name__ == "__main__":
    unittest.main()
