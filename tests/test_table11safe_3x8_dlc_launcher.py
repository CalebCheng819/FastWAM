from __future__ import annotations

import ast
import base64
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "launch_table11safe_3x8_dlc.sh"
RENDERER = REPO / "scripts" / "render_table11safe_3x8_dlc_job.py"


class Table11LauncherTests(unittest.TestCase):
    def fixture(self, root: Path) -> dict[str, str]:
        dataset = root / "tasks"
        for index in range(11):
            task = dataset / f"Task{index}-rf" / f"Task{index}-rf" / "motionplanning"
            task.mkdir(parents=True)
            (task / f"episode_{index}.h5").write_bytes(b"")

        stats = root / "stats.json"
        stats.write_text(
            json.dumps(
                {
                    "action": {"mean": [0.0] * 8, "std": [1.0] * 8},
                    "state": {"mean": [0.0] * 18, "std": [1.0] * 18},
                }
            ),
            encoding="utf-8",
        )
        text_cache = root / "text"
        text_cache.mkdir()
        for index in range(11):
            (text_cache / f"instruction-{index}.pt").write_bytes(b"embedding")

        gaussian = root / "gaussian"
        gaussian.mkdir()
        (gaussian / "COMPLETE").write_text(
            json.dumps({"complete": True}), encoding="utf-8"
        )
        (gaussian / "manifest.json").write_text(
            json.dumps(
                {
                    "total_frames": 1,
                    "derivation": {"source": "direct-teacher-forward-index-v1"},
                }
            ),
            encoding="utf-8",
        )
        (gaussian / "selection.jsonl").write_text("{}\n", encoding="utf-8")

        weight = root / "step_005000.pt"
        weight.write_bytes(b"fixture-weight")
        env = os.environ.copy()
        env.update(
            {
                "RUN_ID": "fastwam-table11-launcher-test",
                "FASTWAM_TABLE11_ATTEMPT_ID": "attempt-1",
                "FASTWAM_TABLE11_REPO_ROOT": str(REPO),
                "FASTWAM_TABLE11_PYTHON": sys.executable,
                "FASTWAM_TABLE11_OFFLINE_ENV_READY": "1",
                "FASTWAM_TABLE11_TEST_MODE": "1",
                "FASTWAM_TABLE11_DRY_RUN": "1",
                "FASTWAM_TABLE11_OUTPUT_DIR": str(root / "output"),
                "FASTWAM_TABLE11_SOURCE_WEIGHT": str(weight),
                "FASTWAM_TABLE11_SOURCE_WEIGHT_BYTES": str(weight.stat().st_size),
                "FASTWAM_TABLE11_DATASET_ROOT": str(dataset),
                "FASTWAM_TABLE11_STATS_PATH": str(stats),
                "FASTWAM_TABLE11_TEXT_CACHE_DIR": str(text_cache),
                "FASTWAM_TABLE11_GAUSSIAN_CACHE_DIR": str(gaussian),
                "FASTWAM_TABLE11_EXPECTED_H5_FILES": "11",
                "FASTWAM_TABLE11_CODE_COMMIT": "1" * 40,
                "WORLD_SIZE": "3",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "NPROC_PER_NODE": "8",
                "MASTER_ADDR": "10.20.30.40",
                "MASTER_PORT": "29500",
            }
        )
        return env

    @staticmethod
    def committed_launcher_bundle(root: Path) -> tuple[Path, str, bytes]:
        source = root / "bundle-source"
        (source / "scripts").mkdir(parents=True)
        launcher_bytes = LAUNCHER.read_bytes() + b"\n# bundle-test-marker\n"
        (source / "scripts" / LAUNCHER.name).write_bytes(launcher_bytes)
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(
            ["git", "-C", str(source), "config", "user.name", "Table11 Test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "config", "user.email", "table11@example.invalid"],
            check=True,
        )
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        bundle = root / "table11-source.bundle"
        subprocess.run(
            ["git", "-C", str(source), "bundle", "create", str(bundle), "HEAD"],
            check=True,
        )
        return bundle, commit, launcher_bytes

    def run_launcher(self, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPO,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_valid_contract_resolves_world24_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_launcher(self.fixture(Path(directory)))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--num_machines 3", result.stdout)
            self.assertIn("--num_processes 24", result.stdout)
            self.assertIn(
                "task=robofactory_table11_vg1_hub1_gau1_cont50k_224_1e-4",
                result.stdout,
            )
            self.assertIn("+scale=robofactory_multi_robot_24gpu_cont50k", result.stdout)
            self.assertIn("table11 safe config gate: world=24 global_batch=24", result.stdout)

    def test_topology_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env["WORLD_SIZE"] = "2"
            result = self.run_launcher(env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("WORLD_SIZE must be the DLC worker count 3", result.stderr)

    def test_launcher_exports_generic_runtime_attempt_contract(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('export FASTWAM_ATTEMPT_ID="${ATTEMPT_ID}"', source)

    def test_conflicting_generic_attempt_id_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env["FASTWAM_ATTEMPT_ID"] = "different-attempt"
            result = self.run_launcher(env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "FASTWAM_ATTEMPT_ID conflicts with FASTWAM_TABLE11_ATTEMPT_ID",
                result.stderr,
            )

    def test_renderer_is_pure_and_pins_priority7_world24_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job.json"
            bundle, commit, launcher_bytes = self.committed_launcher_bundle(root)
            command = [
                sys.executable,
                str(RENDERER),
                "--run-id",
                "fastwam-table11-render-test",
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
                "2" * 40,
                "--offline-source-bundle-relative-path",
                "source/FastWAM.bundle",
                "--base-python",
                "/opt/conda/bin/python3.10",
                "--source-bundle",
                str(bundle),
                "--code-commit",
                commit,
                "--allow-local-bundle-for-tests",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(output.read_text(encoding="utf-8"))
            request = manifest["request"]
            self.assertTrue(manifest["dry_run"])
            self.assertTrue(manifest["submission_not_performed"])
            self.assertEqual(request["Priority"], 7)
            self.assertEqual(request["JobSpecs"][0]["PodCount"], 3)
            self.assertEqual(request["JobSpecs"][0]["ResourceConfig"]["GPU"], "8")
            self.assertEqual(
                {(item["MountPath"], item["MountAccess"]) for item in request["DataSources"]},
                {("/oss-chengjuntao", "RW")},
            )
            self.assertEqual(manifest["batch_contract"]["reference_global_batch"], 24)
            self.assertEqual(manifest["batch_contract"]["replica_global_batch"], 24)
            self.assertTrue(manifest["batch_contract"]["sample_budget_equivalent"])
            self.assertEqual(
                base64.b64decode(manifest["launcher_payload_base64"]), launcher_bytes
            )

    def test_renderer_contains_no_cloud_sdk_call(self) -> None:
        source = RENDERER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse(any(name.startswith("alibabacloud") for name in imported))
        self.assertNotIn("create_job(", source.lower())
        self.assertIn(
            "unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK NODE_RANK",
            LAUNCHER.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
