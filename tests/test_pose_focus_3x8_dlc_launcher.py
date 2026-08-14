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
P5_TASK_NAME = "robofactory_placefood_semantic_phase_p5_224_5e-6.yaml"
P4_TASK_NAME = "robofactory_placefood_gaussian_spatial_p4_224_5e-6.yaml"
P6_TASK_NAME = "robofactory_placefood_spatial_semantic_p6_224_5e-6.yaml"
P7_TASK_NAME = "robofactory_placefood_task_gaussian_relation_p7_224_5e-6.yaml"
P8_TASK_NAME = "robofactory_placefood_relation_gripcontact_p8_224_5e-6.yaml"
P9_TASK_NAME = "robofactory_placefood_spatial_gripcontact_p9_224_5e-6.yaml"
P10_TASK_NAME = "robofactory_placefood_spatial_gripcontact_p10_lowaux_224_5e-6.yaml"
SCALE_NAME = "robofactory_multi_robot_24gpu_pose_focus.yaml"
SCALE_8GPU_NAME = "robofactory_multi_robot_8gpu_eff24_pose_focus.yaml"
SCALE_4GPU_NAME = "robofactory_multi_robot_4gpu_eff24_pose_focus.yaml"
P1_SOURCE_WEIGHT = (
    "/oss-chengjuntao/artifacts/fastwam-placefood-posefocus-r5-s42-24g-r2-20260813/"
    "checkpoints/weights/step_001000.pt"
)
P5_SOURCE_WEIGHT = (
    "/oss-chengjuntao/artifacts/fastwam-placefood-semantic-phase-p5-s42-24g-r1-20260814/"
    "checkpoints/weights/step_001000.pt"
)
P6_SOURCE_WEIGHT = (
    "/oss-chengjuntao/artifacts/fastwam-placefood-spatial-semantic-p6-s42-24g-r1-20260814/"
    "checkpoints/weights/step_001000.pt"
)
P7_SOURCE_WEIGHT = (
    "/oss-chengjuntao/artifacts/fastwam-placefood-task-gaussian-relation-p7-s42-24g-r1-20260814/"
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
        (repo / "configs" / "task" / P5_TASK_NAME).write_bytes(
            (REPO / "configs" / "task" / P5_TASK_NAME).read_bytes()
        )
        (repo / "configs" / "task" / P4_TASK_NAME).write_bytes(
            (REPO / "configs" / "task" / P4_TASK_NAME).read_bytes()
        )
        (repo / "configs" / "task" / P6_TASK_NAME).write_bytes(
            (REPO / "configs" / "task" / P6_TASK_NAME).read_bytes()
        )
        (repo / "configs" / "task" / P7_TASK_NAME).write_bytes(
            (REPO / "configs" / "task" / P7_TASK_NAME).read_bytes()
        )
        (repo / "configs" / "task" / P8_TASK_NAME).write_bytes(
            (REPO / "configs" / "task" / P8_TASK_NAME).read_bytes()
        )
        (repo / "configs" / "task" / P9_TASK_NAME).write_bytes(
            (REPO / "configs" / "task" / P9_TASK_NAME).read_bytes()
        )
        (repo / "configs" / "task" / P10_TASK_NAME).write_bytes(
            (REPO / "configs" / "task" / P10_TASK_NAME).read_bytes()
        )
        (repo / "configs" / "scale" / SCALE_NAME).write_bytes(
            (REPO / "configs" / "scale" / SCALE_NAME).read_bytes()
        )
        (repo / "configs" / "scale" / SCALE_8GPU_NAME).write_bytes(
            (REPO / "configs" / "scale" / SCALE_8GPU_NAME).read_bytes()
        )
        (repo / "configs" / "scale" / SCALE_4GPU_NAME).write_bytes(
            (REPO / "configs" / "scale" / SCALE_4GPU_NAME).read_bytes()
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

    def test_dry_run_resolves_p5_semantic_phase_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env["FASTWAM_POSE_FOCUS_TASK_PROFILE"] = P5_TASK_NAME.removesuffix(".yaml")
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
                "task=robofactory_placefood_semantic_phase_p5_224_5e-6",
                result.stdout,
            )

    def test_dry_run_resolves_p6_spatial_semantic_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env["FASTWAM_POSE_FOCUS_TASK_PROFILE"] = P6_TASK_NAME.removesuffix(".yaml")
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
                "task=robofactory_placefood_spatial_semantic_p6_224_5e-6",
                result.stdout,
            )

    def test_dry_run_resolves_p7_task_conditioned_relation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env["FASTWAM_POSE_FOCUS_TASK_PROFILE"] = P7_TASK_NAME.removesuffix(".yaml")
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
                "task=robofactory_placefood_task_gaussian_relation_p7_224_5e-6",
                result.stdout,
            )

    def test_dry_run_resolves_p8_relation_gripcontact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env["FASTWAM_POSE_FOCUS_TASK_PROFILE"] = P8_TASK_NAME.removesuffix(".yaml")
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
                "task=robofactory_placefood_relation_gripcontact_p8_224_5e-6",
                result.stdout,
            )

    def test_dry_run_resolves_p9_spatial_gripcontact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env["FASTWAM_POSE_FOCUS_TASK_PROFILE"] = P9_TASK_NAME.removesuffix(".yaml")
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
                "task=robofactory_placefood_spatial_gripcontact_p9_224_5e-6",
                result.stdout,
            )

    def test_dry_run_resolves_p10_lowaux_spatial_gripcontact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env["FASTWAM_POSE_FOCUS_TASK_PROFILE"] = P10_TASK_NAME.removesuffix(".yaml")
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
                "task=robofactory_placefood_spatial_gripcontact_p10_lowaux_224_5e-6",
                result.stdout,
            )

    def test_dry_run_resolves_single_node_effective_batch24_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env.update({
                "FASTWAM_POSE_FOCUS_EXPECTED_WORKERS": "1",
                "WORLD_SIZE": "1",
                "RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
            })
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
            self.assertIn("--num_machines 1", result.stdout)
            self.assertIn("--num_processes 8", result.stdout)
            self.assertIn(
                "+scale=robofactory_multi_robot_8gpu_eff24_pose_focus",
                result.stdout,
            )

    def test_dry_run_resolves_four_gpu_effective_batch24_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env.update({
                "FASTWAM_POSE_FOCUS_EXPECTED_WORKERS": "1",
                "FASTWAM_POSE_FOCUS_EXPECTED_GPUS_PER_WORKER": "4",
                "WORLD_SIZE": "1",
                "RANK": "0",
                "NPROC_PER_NODE": "4",
                "MASTER_ADDR": "127.0.0.1",
            })
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
            self.assertIn("--num_machines 1", result.stdout)
            self.assertIn("--num_processes 4", result.stdout)
            self.assertIn(
                "+scale=robofactory_multi_robot_4gpu_eff24_pose_focus",
                result.stdout,
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
            self.assertNotIn("FASTWAM_POSE_FOCUS_TEST_MODE", request["Envs"])
            self.assertEqual(
                {(item["MountPath"], item["MountAccess"]) for item in request["DataSources"]},
                {("/cpfs/user/chengjuntao", "RO"), ("/oss-chengjuntao", "RW")},
            )
            self.assertEqual(base64.b64decode(manifest["launcher_payload_base64"]), launcher_bytes)

    def test_renderer_supports_single_node_without_rdma(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job.json"
            bundle, commit, _ = self.committed_launcher_bundle(root)
            command = [
                sys.executable,
                str(RENDERER),
                "--run-id", "fastwam-placefood-pose-focus-1x8-test",
                "--attempt-id", "attempt-1",
                "--output", str(output),
                "--bootstrap-script", "/oss-chengjuntao/source/bootstrap.sh",
                "--offline-env-source-root", "/oss-chengjuntao/offline-env",
                "--offline-env-manifest", "/oss-chengjuntao/offline-env/manifest.json",
                "--offline-code-commit", "4" * 40,
                "--offline-source-bundle-relative-path", "source/FastWAM.bundle",
                "--base-python", "/opt/conda/bin/python3.10",
                "--pose-focus-source-bundle", str(bundle),
                "--pose-focus-code-commit", commit,
                "--task-profile", P8_TASK_NAME.removesuffix(".yaml"),
                "--worker-count", "1",
                "--allow-local-bundle-for-tests",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            request = json.loads(output.read_text(encoding="utf-8"))["request"]
            self.assertEqual(request["JobSpecs"][0]["PodCount"], 1)
            self.assertEqual(request["Envs"]["FASTWAM_POSE_FOCUS_EXPECTED_WORKERS"], "1")
            self.assertEqual(request["Envs"]["FASTWAM_PREFLIGHT_REQUIRE_ERDMA"], "0")
            self.assertNotIn("NCCL_IB_HCA", request["Envs"])
            self.assertNotIn("FASTWAM_ERDMA_BUNDLE_ROOT", request["Envs"])
            self.assertFalse(request["Settings"]["EnableRDMA"])
            self.assertFalse(request["Settings"]["AllocateAllRDMADevices"])
            self.assertEqual(request["Settings"]["Tags"]["topology"], "1x8-world8")

    def test_renderer_supports_single_node_four_gpu_without_rdma(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job.json"
            bundle, commit, _ = self.committed_launcher_bundle(root)
            command = [
                sys.executable,
                str(RENDERER),
                "--run-id", "fastwam-placefood-pose-focus-1x4-test",
                "--attempt-id", "attempt-1",
                "--output", str(output),
                "--bootstrap-script", "/oss-chengjuntao/source/bootstrap.sh",
                "--offline-env-source-root", "/oss-chengjuntao/offline-env",
                "--offline-env-manifest", "/oss-chengjuntao/offline-env/manifest.json",
                "--offline-code-commit", "4" * 40,
                "--offline-source-bundle-relative-path", "source/FastWAM.bundle",
                "--base-python", "/opt/conda/bin/python3.10",
                "--pose-focus-source-bundle", str(bundle),
                "--pose-focus-code-commit", commit,
                "--task-profile", P8_TASK_NAME.removesuffix(".yaml"),
                "--worker-count", "1",
                "--gpus-per-worker", "4",
                "--allow-local-bundle-for-tests",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            request = json.loads(output.read_text(encoding="utf-8"))["request"]
            self.assertEqual(request["JobSpecs"][0]["PodCount"], 1)
            self.assertEqual(request["JobSpecs"][0]["ResourceConfig"]["GPU"], "4")
            self.assertEqual(request["JobSpecs"][0]["ResourceConfig"]["CPU"], "63")
            self.assertEqual(
                request["JobSpecs"][0]["ResourceConfig"]["Memory"],
                "480Gi",
            )
            self.assertEqual(request["Envs"]["NPROC_PER_NODE"], "4")
            self.assertEqual(
                request["Envs"]["FASTWAM_POSE_FOCUS_EXPECTED_GPUS_PER_WORKER"],
                "4",
            )
            self.assertFalse(request["Settings"]["EnableRDMA"])
            self.assertEqual(request["Settings"]["Tags"]["topology"], "1x4-world4")

    def test_renderer_selects_audited_p1_weight_for_p4(self) -> None:
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
                "--task-profile",
                P4_TASK_NAME.removesuffix(".yaml"),
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

    def test_renderer_selects_p6_source_and_objective(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job.json"
            bundle, commit, _ = self.committed_launcher_bundle(root)
            command = [
                sys.executable,
                str(RENDERER),
                "--run-id",
                "fastwam-placefood-spatial-semantic-p6-test",
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
                P6_TASK_NAME.removesuffix(".yaml"),
                "--source-weight",
                P5_SOURCE_WEIGHT,
                "--allow-local-bundle-for-tests",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(output.read_text(encoding="utf-8"))
            request = manifest["request"]
            self.assertEqual(
                request["Settings"]["Tags"]["objective"],
                "placefood-spatial-gaussian-semantic-phase",
            )
            self.assertEqual(
                request["Settings"]["Tags"]["initialization"],
                "P5-action-step1000-weights-only",
            )
            self.assertEqual(
                request["Envs"]["FASTWAM_POSE_FOCUS_SOURCE_WEIGHT"],
                P5_SOURCE_WEIGHT,
            )

    def test_renderer_selects_p7_source_and_objective(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job.json"
            bundle, commit, _ = self.committed_launcher_bundle(root)
            command = [
                sys.executable,
                str(RENDERER),
                "--run-id",
                "fastwam-placefood-task-gaussian-relation-p7-test",
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
                P7_TASK_NAME.removesuffix(".yaml"),
                "--source-weight",
                P6_SOURCE_WEIGHT,
                "--allow-local-bundle-for-tests",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(output.read_text(encoding="utf-8"))
            request = manifest["request"]
            self.assertEqual(
                request["Settings"]["Tags"]["objective"],
                "placefood-task-conditioned-gaussian-relation",
            )
            self.assertEqual(
                request["Settings"]["Tags"]["initialization"],
                "P6-action-step1000-weights-only",
            )
            self.assertEqual(
                request["Envs"]["FASTWAM_POSE_FOCUS_SOURCE_WEIGHT"],
                P6_SOURCE_WEIGHT,
            )
            self.assertEqual(
                request["Envs"]["FASTWAM_POSE_FOCUS_SOURCE_WEIGHT_BYTES"],
                "12047407747",
            )

    def test_renderer_selects_p8_source_and_objective(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job.json"
            bundle, commit, _ = self.committed_launcher_bundle(root)
            command = [
                sys.executable,
                str(RENDERER),
                "--run-id",
                "fastwam-placefood-relation-gripcontact-p8-test",
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
                P8_TASK_NAME.removesuffix(".yaml"),
                "--source-weight",
                P7_SOURCE_WEIGHT,
                "--allow-local-bundle-for-tests",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            request = json.loads(output.read_text(encoding="utf-8"))["request"]
            self.assertEqual(
                request["Settings"]["Tags"]["objective"],
                "placefood-relation-gripper-contact-proxy",
            )
            self.assertEqual(
                request["Settings"]["Tags"]["initialization"],
                "P7-action-step1000-weights-only",
            )
            self.assertEqual(
                request["Envs"]["FASTWAM_POSE_FOCUS_SOURCE_WEIGHT"],
                P7_SOURCE_WEIGHT,
            )
            self.assertEqual(
                request["Envs"]["FASTWAM_POSE_FOCUS_SOURCE_WEIGHT_BYTES"],
                "12055814467",
            )

    def test_renderer_selects_p9_source_and_objective(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job.json"
            bundle, commit, _ = self.committed_launcher_bundle(root)
            command = [
                sys.executable,
                str(RENDERER),
                "--run-id",
                "fastwam-placefood-spatial-gripcontact-p9-test",
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
                P9_TASK_NAME.removesuffix(".yaml"),
                "--source-weight",
                P6_SOURCE_WEIGHT,
                "--allow-local-bundle-for-tests",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            request = json.loads(output.read_text(encoding="utf-8"))["request"]
            self.assertEqual(
                request["Settings"]["Tags"]["objective"],
                "placefood-spatial-gripper-contact-proxy",
            )
            self.assertEqual(
                request["Settings"]["Tags"]["initialization"],
                "P6-action-step1000-weights-only",
            )
            self.assertEqual(
                request["Envs"]["FASTWAM_POSE_FOCUS_SOURCE_WEIGHT"],
                P6_SOURCE_WEIGHT,
            )
            self.assertEqual(
                request["Envs"]["FASTWAM_POSE_FOCUS_SOURCE_WEIGHT_BYTES"],
                "12047407747",
            )

    def test_renderer_selects_p10_source_and_objective(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job.json"
            bundle, commit, _ = self.committed_launcher_bundle(root)
            command = [
                sys.executable,
                str(RENDERER),
                "--run-id",
                "fastwam-placefood-spatial-gripcontact-p10-lowaux-test",
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
                P10_TASK_NAME.removesuffix(".yaml"),
                "--source-weight",
                P6_SOURCE_WEIGHT,
                "--allow-local-bundle-for-tests",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            request = json.loads(output.read_text(encoding="utf-8"))["request"]
            self.assertEqual(
                request["Settings"]["Tags"]["objective"],
                "placefood-spatial-gripper-contact-lowaux",
            )
            self.assertEqual(
                request["Settings"]["Tags"]["initialization"],
                "P6-action-step1000-weights-only",
            )
            self.assertEqual(
                request["Envs"]["FASTWAM_POSE_FOCUS_SOURCE_WEIGHT"],
                P6_SOURCE_WEIGHT,
            )
            self.assertEqual(
                request["Envs"]["FASTWAM_POSE_FOCUS_SOURCE_WEIGHT_BYTES"],
                "12047407747",
            )


if __name__ == "__main__":
    unittest.main()
