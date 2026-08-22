from __future__ import annotations

import ast
import base64
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "launch_b4_3x8_dlc.sh"
RENDERER = REPO / "scripts" / "render_b4_3x8_dlc_job.py"
STAT_CMP_HELPER = REPO / "scripts" / "b4_stat_cmp_cache.py"


def load_stat_cmp_helper():
    spec = importlib.util.spec_from_file_location("b4_stat_cmp_cache", STAT_CMP_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load B4 stat-cmp helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class B4LauncherTests(unittest.TestCase):
    def fixture(self, root: Path, *, wired: bool = True) -> dict[str, str]:
        repo = root / "repo"
        (repo / "scripts" / "accelerate_configs").mkdir(parents=True)
        (repo / "scripts" / "train.py").write_text("pass\n", encoding="utf-8")
        (repo / "scripts" / "b4_stat_cmp_cache.py").write_bytes(STAT_CMP_HELPER.read_bytes())
        (repo / "scripts" / "accelerate_configs" / "accelerate_zero2_ds.yaml").write_text(
            "compute_environment: LOCAL_MACHINE\n", encoding="utf-8"
        )
        (repo / "configs" / "task").mkdir(parents=True)
        (repo / "configs" / "scale").mkdir(parents=True)
        for task_name in (
            "robofactory_multi_robot_b4_phase_gripcontact_actft_224_1e-5.yaml",
            "robofactory_multi_robot_vg1_hub1_gau1_cont50k_224_1e-4.yaml",
            "robofactory_multi_robot_vg1_hub1_gau0_cont50k_224_1e-4.yaml",
        ):
            (repo / "configs" / "task" / task_name).write_text(
                (REPO / "configs" / "task" / task_name).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        for scale_name in (
            "robofactory_multi_robot_24gpu_b4.yaml",
            "robofactory_multi_robot_24gpu_cont50k.yaml",
            "robofactory_multi_robot_32gpu_cont50k.yaml",
        ):
            (repo / "configs" / "scale" / scale_name).write_text(
                (REPO / "configs" / "scale" / scale_name).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        (repo / "src" / "fastwam").mkdir(parents=True)
        (repo / "src" / "fastwam" / "__init__.py").write_text(
            '"""B4 launcher fixture."""\n', encoding="utf-8"
        )
        trainer = """
class T:
    def __init__(self, cfg):
        self.phase_balanced_fraction = cfg.get('phase_balanced_fraction', 0.0)

    def _build_loader(self, dataset):
        return ResumableAgentCountBatchSampler(
            dataset=dataset,
            phase_balanced_fraction=self.phase_balanced_fraction,
        )
""" if wired else "class T:\n    pass\n"
        (repo / "src" / "fastwam" / "trainer.py").write_text(trainer, encoding="utf-8")
        dataset = root / "dataset"
        text_cache = dataset / "text"
        gaussian = root / "gaussian"
        text_cache.mkdir(parents=True)
        gaussian.mkdir()
        stats = dataset / "stats.json"
        stats.write_text(
            json.dumps(
                {
                    "action": {
                        "mean": [0] * 7 + [0.24164481092854787],
                        "std": [1] * 7 + [0.9469631616807775],
                    }
                }
            ),
            encoding="utf-8",
        )
        weight = root / "step_005000.pt"
        weight.write_bytes(b"fixture-weight")
        run_id = "b4-launcher-unit"
        env = os.environ.copy()
        env.update(
            {
                "RUN_ID": run_id,
                "FASTWAM_B4_ATTEMPT_ID": "attempt-1",
                "FASTWAM_B4_REPO_ROOT": str(repo),
                "FASTWAM_B4_OUTPUT_DIR": f"/oss-chengjuntao/artifacts/{run_id}",
                "FASTWAM_B4_SOURCE_WEIGHT": str(weight),
                "FASTWAM_B4_SOURCE_WEIGHT_BYTES": str(weight.stat().st_size),
                "FASTWAM_B4_DATASET_ROOT": str(dataset),
                "FASTWAM_B4_STATS_PATH": str(stats),
                "FASTWAM_B4_TEXT_CACHE_DIR": str(text_cache),
                "FASTWAM_B4_GAUSSIAN_CACHE_DIR": str(gaussian),
                "FASTWAM_B4_PYTHON": sys.executable,
                "FASTWAM_B4_OFFLINE_ENV_READY": "1",
                "FASTWAM_B4_TEST_MODE": "1",
                "FASTWAM_B4_DRY_RUN": "1",
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
    def executable(path: Path, body: str) -> None:
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)

    @staticmethod
    def committed_launcher_bundle(root: Path) -> tuple[Path, str, bytes]:
        source = root / "b4-bundle-source"
        (source / "scripts").mkdir(parents=True)
        launcher_bytes = LAUNCHER.read_bytes() + b"\n# explicit-bundle-test-marker\n"
        (source / "scripts" / LAUNCHER.name).write_bytes(launcher_bytes)
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.name", "Launcher Test"], check=True)
        subprocess.run(
            ["git", "-C", str(source), "config", "user.email", "launcher@example.invalid"],
            check=True,
        )
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(source), "commit", "-qm", "launcher fixture"], check=True)
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        bundle = root / "b4-source.bundle"
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

    def test_valid_contract_resolves_world24_weight_only_fresh_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            result = self.run_launcher(env)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = result.stdout
            self.assertIn("--num_machines 3", output)
            self.assertIn("--num_processes 24", output)
            self.assertIn("--machine_rank 0", output)
            self.assertIn(
                "task=robofactory_multi_robot_b4_phase_gripcontact_actft_224_1e-5",
                output,
            )
            self.assertIn("+scale=robofactory_multi_robot_24gpu_b4", output)
            self.assertNotIn("arm_huber_beta=", output)
            self.assertNotIn("phase_balanced_fraction=", output)
            self.assertNotIn("/checkpoints/state/", output)
            self.assertIn("B4 source import gate:", output)

    def test_cont50k_contract_resolves_cumulative_world24_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env["FASTWAM_TRAINING_TREATMENT"] = "n234_vg1h1gau1_cont50k"
            result = self.run_launcher(env)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = result.stdout
            self.assertIn("--num_machines 3", output)
            self.assertIn("--num_processes 24", output)
            self.assertIn(
                "task=robofactory_multi_robot_vg1_hub1_gau1_cont50k_224_1e-4",
                output,
            )
            self.assertIn("+scale=robofactory_multi_robot_24gpu_cont50k", output)
            self.assertNotIn("phase_balanced_fraction=", output)
            self.assertNotIn("/checkpoints/state/", output)
            self.assertIn("B4 source import gate:", output)

    def test_gau0_cont50k_contract_resolves_cumulative_world32_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env["FASTWAM_TRAINING_TREATMENT"] = "n234_vg1h1gau0_cont50k"
            env["WORLD_SIZE"] = "4"
            result = self.run_launcher(env)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = result.stdout
            self.assertIn("--num_machines 4", output)
            self.assertIn("--num_processes 32", output)
            self.assertIn(
                "task=robofactory_multi_robot_vg1_hub1_gau0_cont50k_224_1e-4",
                output,
            )
            self.assertIn("+scale=robofactory_multi_robot_32gpu_cont50k", output)
            self.assertNotIn("data.train.gaussian_cache_dir=", output)
            self.assertNotIn("data.val.gaussian_cache_dir=", output)
            self.assertNotIn("/checkpoints/state/", output)
            self.assertIn("B4 source import gate:", output)

    def test_topology_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env["WORLD_SIZE"] = "4"
            result = self.run_launcher(env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("WORLD_SIZE must be", result.stderr)

    def test_training_state_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = self.fixture(root)
            state_dir = root / "checkpoints" / "state" / "step_005000"
            state_dir.mkdir(parents=True)
            env["FASTWAM_B4_SOURCE_WEIGHT"] = str(state_dir)
            result = self.run_launcher(env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("weight .pt file", result.stderr)

    def test_missing_sampler_wiring_stops_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory), wired=False)
            result = self.run_launcher(env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("phase-balanced sampling", result.stderr)

    def test_node_coordinator_stages_weight_then_launches_from_local_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = self.fixture(root)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            command_log = root / "accelerate-command.txt"
            self.executable(
                fake_bin / "nvidia-smi",
                "#!/usr/bin/env bash\n"
                "for index in 0 1 2 3 4 5 6 7; do printf '%s\\n' \"$index\"; done\n",
            )
            self.executable(
                fake_bin / "python",
                "#!/usr/bin/env bash\n"
                "if [[ \"${1:-}\" == '-m' ]]; then\n"
                "  if [[ -n \"${WORLD_SIZE+x}${RANK+x}${LOCAL_RANK+x}${LOCAL_WORLD_SIZE+x}\" ]]; then\n"
                "    printf 'inherited outer rank environment\\n' >&2\n"
                "    exit 93\n"
                "  fi\n"
                "  printf '%s\\n' \"$@\" > \"$FASTWAM_B4_TEST_COMMAND_LOG\"\n"
                "  exit 0\n"
                "fi\n"
                f"exec {sys.executable} \"$@\"\n",
            )
            env.update(
                {
                    "FASTWAM_B4_DRY_RUN": "0",
                    "FASTWAM_B4_OUTPUT_DIR": str(root / "output"),
                    "FASTWAM_B4_LOCAL_CACHE_ROOT": str(root / "local-cache"),
                    "FASTWAM_B4_PYTHON": str(fake_bin / "python"),
                    "FASTWAM_B4_TEST_COMMAND_LOG": str(command_log),
                    "PATH": str(fake_bin) + os.pathsep + env.get("PATH", ""),
                }
            )
            result = self.run_launcher(env)
            self.assertEqual(result.returncode, 0, result.stderr)
            local_weight = (
                root / "local-cache" / env["RUN_ID"] / env["FASTWAM_B4_ATTEMPT_ID"] / "step_005000.pt"
            )
            self.assertEqual(local_weight.read_bytes(), (root / "step_005000.pt").read_bytes())
            self.assertTrue(local_weight.with_name(".ready").is_file())
            ready = local_weight.with_name(".ready").read_text(encoding="utf-8")
            self.assertIn("provenance_mode=stat_cmp", ready)
            self.assertIn(f"run_id={env['RUN_ID']}", ready)
            self.assertIn(f"attempt_id={env['FASTWAM_B4_ATTEMPT_ID']}", ready)
            self.assertIn("file_count=1", ready)
            self.assertTrue((root / "output" / ".b4-run-reservation").is_file())
            launched = command_log.read_text(encoding="utf-8")
            self.assertIn("--num_processes\n24", launched)
            self.assertIn(
                f"data.train.stats_source_root={root / 'dataset'}",
                launched,
            )
            self.assertIn(
                f"data.val.stats_source_root={root / 'dataset'}",
                launched,
            )
            self.assertEqual(env["FASTWAM_B4_PYTHON"], str(fake_bin / "python"))
            self.assertNotIn("/checkpoints/state/", launched)

    def test_checkpoint_stage_rejects_truncated_and_corrupted_copies_before_promotion(self) -> None:
        for mutation in ("truncate", "corrupt"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env = self.fixture(root)
                fake_bin = root / "bin"
                fake_bin.mkdir()
                self.executable(
                    fake_bin / "nvidia-smi",
                    "#!/usr/bin/env bash\n"
                    "for index in 0 1 2 3 4 5 6 7; do printf '%s\\n' \"$index\"; done\n",
                )
                self.executable(
                    fake_bin / "cp",
                    "#!/usr/bin/env bash\n"
                    "/bin/cp \"$@\"\n"
                    "target=\"${@: -1}\"\n"
                    "if [[ \"$FASTWAM_B4_TEST_COPY_MUTATION\" == truncate ]]; then\n"
                    "  bytes=$(stat -c '%s' -- \"$target\")\n"
                    "  truncate -s \"$((bytes - 1))\" -- \"$target\"\n"
                    "else\n"
                    "  printf 'X' | dd of=\"$target\" bs=1 seek=0 conv=notrunc status=none\n"
                    "fi\n",
                )
                local_root = root / "local-cache"
                env.update(
                    {
                        "FASTWAM_B4_DRY_RUN": "0",
                        "FASTWAM_B4_OUTPUT_DIR": str(root / "output"),
                        "FASTWAM_B4_LOCAL_CACHE_ROOT": str(local_root),
                        "FASTWAM_B4_TEST_COPY_MUTATION": mutation,
                        "PATH": str(fake_bin) + os.pathsep + env.get("PATH", ""),
                    }
                )
                result = self.run_launcher(env)
                self.assertNotEqual(result.returncode, 0)
                expected = "wrong byte count" if mutation == "truncate" else "bytes differ"
                self.assertIn(expected, result.stderr)
                promoted = local_root / env["RUN_ID"] / env["FASTWAM_B4_ATTEMPT_ID"]
                self.assertFalse(promoted.exists())

    def test_formal_task_drift_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = self.fixture(root)
            task = (
                root
                / "repo"
                / "configs"
                / "task"
                / "robofactory_multi_robot_b4_phase_gripcontact_actft_224_1e-5.yaml"
            )
            task.write_text(
                task.read_text(encoding="utf-8").replace("arm_huber_beta: 0.1", "arm_huber_beta: 1.0"),
                encoding="utf-8",
            )
            result = self.run_launcher(env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("formal task/scale contract", result.stderr)

    def test_dependency_bootstrap_cannot_replace_b4_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = self.fixture(root)
            b4_repo = root / "repo"
            subprocess.run(["git", "init", "-q", str(b4_repo)], check=True)
            subprocess.run(["git", "-C", str(b4_repo), "config", "user.name", "Launcher Test"], check=True)
            subprocess.run(
                ["git", "-C", str(b4_repo), "config", "user.email", "launcher@example.invalid"],
                check=True,
            )
            subprocess.run(["git", "-C", str(b4_repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(b4_repo), "commit", "-qm", "fixture"], check=True)
            commit = subprocess.check_output(
                ["git", "-C", str(b4_repo), "rev-parse", "HEAD"], text=True
            ).strip()
            bundle = root / "b4-source.bundle"
            subprocess.run(
                ["git", "-C", str(b4_repo), "bundle", "create", str(bundle), "HEAD"], check=True
            )

            old_repo = root / "old-v9-repo"
            (old_repo / "src" / "fastwam").mkdir(parents=True)
            (old_repo / "src" / "fastwam" / "__init__.py").write_text(
                '"""Legacy dependency snapshot; must not be imported."""\n', encoding="utf-8"
            )
            bootstrap = root / "legacy-bootstrap.sh"
            bootstrap.write_text(
                "fastwam_prepare_offline_training_env() {\n"
                "  [[ \"${FASTWAM_CODE_COMMIT:-}\" == \"${FASTWAM_OFFLINE_CODE_COMMIT:-}\" ]] || return 91\n"
                f"  export FASTWAM_PYTHON={sys.executable!s}\n"
                f"  export FASTWAM_REPO_ROOT={old_repo!s}\n"
                f"  export PYTHONPATH={old_repo!s}/src\n"
                "}\n",
                encoding="utf-8",
            )
            env.update(
                {
                    "FASTWAM_B4_OFFLINE_ENV_READY": "0",
                    "FASTWAM_B4_BOOTSTRAP_SCRIPT": str(bootstrap),
                    "FASTWAM_OFFLINE_ENV_BASE_PYTHON": sys.executable,
                    "FASTWAM_B4_SOURCE_BUNDLE": str(bundle),
                    "FASTWAM_B4_CODE_COMMIT": commit,
                    "FASTWAM_OFFLINE_CODE_COMMIT": "4" * 40,
                    "FASTWAM_B4_LOCAL_SOURCE_ROOT": str(root / "local-b4-source"),
                }
            )
            env.pop("FASTWAM_B4_REPO_ROOT", None)
            env.pop("FASTWAM_B4_PYTHON", None)
            result = self.run_launcher(env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(root / "local-b4-source"), result.stdout)
            self.assertNotIn(str(old_repo / "src" / "fastwam" / "__init__.py"), result.stdout)
            self.assertNotIn(
                'exec /bin/bash "${FASTWAM_REPO_ROOT}/scripts/launch_b4_3x8_dlc.sh"',
                LAUNCHER.read_text(encoding="utf-8"),
            )

    def test_stat_cmp_cache_ignores_fake_digest_and_publishes_ready_contract(self) -> None:
        helper = load_stat_cmp_helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            selected = source / "nested" / "payload.bin"
            selected.parent.mkdir(parents=True)
            selected.write_bytes(b"B4-stat-cmp-payload")
            allowlist = root / "selection.sha256"
            allowlist.write_text(
                "this-is-deliberately-not-a-digest  nested/payload.bin\n",
                encoding="utf-8",
            )
            destination = root / "cache" / "cpfs"
            ready = helper.stage_allowlisted_tree(
                source_root=source,
                allowlist=allowlist,
                destination=destination,
                run_id="run-b4",
                attempt_id="attempt-1",
                source_label="cpfs",
            )
            self.assertEqual((destination / "nested" / "payload.bin").read_bytes(), selected.read_bytes())
            self.assertEqual(ready["provenance_mode"], "stat_cmp")
            self.assertEqual(ready["file_count"], 1)
            self.assertEqual(ready["total_bytes"], selected.stat().st_size)
            persisted = json.loads(
                (destination / "READY.stat-cmp.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["run_id"], "run-b4")
            self.assertEqual(persisted["attempt_id"], "attempt-1")
            self.assertEqual(persisted["destination_path"], str(destination))
            self.assertIn("newest_source_mtime_ns", persisted)

    def test_stat_cmp_cache_accepts_regular_lock_file_but_not_symlinked_roots(self) -> None:
        helper = load_stat_cmp_helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            lock_file = (
                source
                / "checkpoints/FastWAM/model-cache/.lock"
                / "DiffSynth-Studio___Wan-Series-Converted-Safetensors"
            )
            dataset_file = source / "datasets/robofactory_multi_robot/sample.h5"
            lock_file.parent.mkdir(parents=True)
            dataset_file.parent.mkdir(parents=True)
            lock_file.write_bytes(b"")
            dataset_file.write_bytes(b"fixture")
            allowlist = root / "selection.sha256"
            allowlist.write_text(
                "opaque checkpoints/FastWAM/model-cache/.lock/"
                "DiffSynth-Studio___Wan-Series-Converted-Safetensors\n"
                "opaque datasets/robofactory_multi_robot/sample.h5\n",
                encoding="utf-8",
            )
            destination = root / "cache"
            ready = helper.stage_allowlisted_tree(
                source_root=source,
                allowlist=allowlist,
                destination=destination,
                run_id="run-b4",
                attempt_id="attempt-3",
                source_label="oss_cpfs_mirror",
            )
            self.assertEqual(ready["file_count"], 2)
            self.assertTrue((destination / lock_file.relative_to(source)).is_file())
            self.assertEqual(
                (destination / dataset_file.relative_to(source)).read_bytes(), b"fixture"
            )

            linked_source = root / "linked-source"
            linked_source.mkdir()
            (linked_source / "checkpoints").symlink_to(source / "checkpoints", target_is_directory=True)
            linked_allowlist = root / "linked-selection.sha256"
            linked_allowlist.write_text(
                "opaque checkpoints/FastWAM/model-cache/.lock/"
                "DiffSynth-Studio___Wan-Series-Converted-Safetensors\n",
                encoding="utf-8",
            )
            linked_destination = root / "linked-cache"
            with self.assertRaises(RuntimeError):
                helper.stage_allowlisted_tree(
                    source_root=linked_source,
                    allowlist=linked_allowlist,
                    destination=linked_destination,
                    run_id="run-b4",
                    attempt_id="attempt-3",
                    source_label="fixture",
                )
            self.assertFalse(linked_destination.exists())

    def test_stat_cmp_cache_rejects_traversal_and_symlinks(self) -> None:
        helper = load_stat_cmp_helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            outside = root / "outside.bin"
            outside.write_bytes(b"outside")
            (source / "linked.bin").symlink_to(outside)
            cases = ("ignored ../outside.bin\n", "ignored linked.bin\n")
            for index, content in enumerate(cases):
                allowlist = root / f"selection-{index}.sha256"
                allowlist.write_text(content, encoding="utf-8")
                destination = root / f"destination-{index}"
                with self.assertRaises(RuntimeError):
                    helper.stage_allowlisted_tree(
                        source_root=source,
                        allowlist=allowlist,
                        destination=destination,
                        run_id="run-b4",
                        attempt_id="attempt-1",
                        source_label="fixture",
                    )
                self.assertFalse(destination.exists())

    def test_stat_cmp_cache_detects_truncated_and_corrupted_copies_before_promotion(self) -> None:
        helper = load_stat_cmp_helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            payload = source / "payload.bin"
            payload.write_bytes(b"abcdefgh")
            allowlist = root / "selection.sha256"
            allowlist.write_text("ignored payload.bin\n", encoding="utf-8")

            def truncate(left: Path, right: Path) -> None:
                right.write_bytes(left.read_bytes()[:-1])

            def corrupt(left: Path, right: Path) -> None:
                data = bytearray(left.read_bytes())
                data[0] ^= 0xFF
                right.write_bytes(data)

            for index, copier in enumerate((truncate, corrupt)):
                destination = root / f"destination-{index}"
                with self.assertRaises(RuntimeError):
                    helper.stage_allowlisted_tree(
                        source_root=source,
                        allowlist=allowlist,
                        destination=destination,
                        run_id="run-b4",
                        attempt_id="attempt-1",
                        source_label="fixture",
                        copy_file=copier,
                    )
                self.assertFalse(destination.exists())
                self.assertFalse(any(destination.parent.glob(f".{destination.name}.partial.*")))

    def test_renderer_is_pure_dry_run_and_pins_dlc_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job.json"
            bundle, b4_commit, committed_launcher = self.committed_launcher_bundle(root)
            run_id = "fastwam-b4-24g-test"
            command = [
                sys.executable,
                str(RENDERER),
                "--run-id", run_id,
                "--attempt-id", "attempt-20260811-01",
                "--output", str(output),
                "--bootstrap-script", "/oss-chengjuntao/source/scripts/bootstrap_offline_training_env.sh",
                "--offline-env-source-root", "/oss-chengjuntao/offline-env",
                "--offline-env-manifest", "/oss-chengjuntao/offline-env/manifest.json",
                "--offline-code-commit", "4" * 40,
                "--offline-source-bundle-relative-path", "source/FastWAM.bundle",
                "--base-python", "/opt/conda/bin/python3.10",
                "--b4-source-bundle", str(bundle),
                "--b4-code-commit", b4_commit,
                "--allow-local-bundle-for-tests",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(manifest["dry_run"])
            self.assertTrue(manifest["submission_not_performed"])
            request = manifest["request"]
            spec = request["JobSpecs"][0]
            self.assertEqual(spec["PodCount"], 3)
            self.assertEqual(spec["ResourceConfig"]["GPU"], "8")
            self.assertEqual(request["Priority"], 7)
            self.assertEqual(request["JobMaxRunningTimeMinutes"], 10080)
            self.assertEqual(request["SuccessPolicy"], "AllWorkers")
            self.assertEqual(request["Envs"]["FASTWAM_TRAINING_TREATMENT"], "b4")
            self.assertEqual(
                request["Envs"]["FASTWAM_B4_SOURCE_WEIGHT"],
                "/oss-chengjuntao/artifacts/"
                "fastwam-n234-vg1hub1gau1-s42-5000-r2a2-"
                "beg0t5rle97qepyw8u-a57915104bff-20260802t1820z/"
                "checkpoints/weights/step_005000.pt",
            )
            self.assertEqual(manifest["training_treatment"], "b4")
            self.assertEqual(request["Envs"]["FASTWAM_B4_OUTPUT_DIR"], f"/oss-chengjuntao/artifacts/{run_id}")
            self.assertNotIn("FASTWAM_B4_TEST_MODE", request["Envs"])
            self.assertNotIn("FASTWAM_B4_REPO_ROOT", request["Envs"])
            self.assertNotIn("FASTWAM_GAUSSIAN_CACHE_MANIFEST_SHA256", request["Envs"])
            self.assertNotIn("FASTWAM_GAUSSIAN_CACHE_SELECTION_SHA256", request["Envs"])
            self.assertNotIn("FASTWAM_GAUSSIAN_CACHE_SOURCE_IDENTITY_SHA256", request["Envs"])
            self.assertEqual(request["Envs"]["FASTWAM_OFFLINE_ENV_BASE_PYTHON"], "/opt/conda/bin/python3.10")
            self.assertEqual(request["Envs"]["FASTWAM_OFFLINE_CODE_COMMIT"], "4" * 40)
            self.assertNotIn("FASTWAM_CODE_COMMIT", request["Envs"])
            self.assertEqual(request["Envs"]["FASTWAM_B4_CODE_COMMIT"], b4_commit)
            self.assertEqual(
                request["Envs"]["FASTWAM_B4_SOURCE_BUNDLE"],
                str(bundle),
            )
            self.assertEqual(request["Envs"]["FASTWAM_B4_LOCAL_SOURCE_ROOT"], "/tmp/fastwam-b4-source-checkouts")
            self.assertEqual(request["Envs"]["FASTWAM_OFFLINE_ENV_CACHE_ROOT"], "/tmp/fastwam-offline-env-cache")
            self.assertEqual(request["Envs"]["FASTWAM_OFFLINE_ENV_VENV_ROOT"], "/tmp/fastwam-offline-env-venvs")
            self.assertEqual(request["Envs"]["FASTWAM_SOURCE_CHECKOUT_ROOT"], "/tmp/fastwam-source-checkouts")
            self.assertEqual(request["Envs"]["FASTWAM_B4_PROVENANCE_MODE"], "stat_cmp")
            self.assertEqual(
                request["Envs"]["FASTWAM_B4_CPFS_SOURCE_ROOT"],
                "/oss-chengjuntao/cpfs-user-chengjuntao",
            )
            self.assertEqual(
                request["Envs"]["FASTWAM_B4_STATS_SOURCE_ROOT"],
                "/cpfs/user/chengjuntao/datasets/robofactory_multi_robot",
            )
            self.assertEqual(
                request["Envs"]["FASTWAM_B4_INPUT_CACHE_ROOT"],
                "/tmp/fastwam-b4-input-cache",
            )
            self.assertNotIn("FASTWAM_CPFS_BUNDLE_MANIFEST_SHA256", request["Envs"])
            self.assertNotIn("FASTWAM_OSS_BUNDLE_MANIFEST_SHA256", request["Envs"])
            self.assertEqual(
                manifest["b4_provenance_contract"],
                {
                    "mode": "stat_cmp",
                    "new_hashes": False,
                    "records": [
                        "path",
                        "bytes",
                        "mtime",
                        "count",
                        "run_id",
                        "attempt_id",
                        "world_size",
                    ],
                },
            )
            self.assertEqual(request["Envs"]["FASTWAM_LOCAL_EXPECTED_H5_FILES"], "24")
            self.assertEqual(
                request["Envs"]["FASTWAM_LOCAL_MODEL_CACHE_RELATIVE_ROOT"],
                "checkpoints/FastWAM/model-cache",
            )
            self.assertTrue(
                request["Envs"]["FASTWAM_LOCAL_VAE_RELATIVE_PATH"].endswith(
                    "/Wan2.2_VAE.safetensors"
                )
            )
            self.assertEqual(request["Envs"]["FASTWAM_B4_OUTPUT_RESERVATION_TIMEOUT"], "300")
            self.assertEqual(request["Envs"]["NPROC_PER_NODE"], "8")
            self.assertEqual(request["Envs"]["NCCL_IB_HCA"], "erdma")
            self.assertEqual(request["Envs"]["NCCL_DEBUG"], "INFO")
            self.assertEqual(request["Envs"]["NCCL_DEBUG_SUBSYS"], "INIT,NET")
            self.assertEqual(
                {(item["MountPath"], item["MountAccess"]) for item in request["DataSources"]},
                {("/cpfs/user/chengjuntao", "RO"), ("/oss-chengjuntao", "RW")},
            )
            decoded = base64.b64decode(manifest["launcher_payload_base64"])
            self.assertEqual(decoded, committed_launcher)
            self.assertEqual(
                manifest["launcher_source"],
                {
                    "bundle": str(bundle),
                    "code_commit": b4_commit,
                    "path": "scripts/launch_b4_3x8_dlc.sh",
                },
            )
            self.assertIn("base64 --decode", request["UserCommand"])
            self.assertNotIn("python3", request["UserCommand"])

    def test_renderer_pins_cont50k_cumulative_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job.json"
            bundle, code_commit, committed_launcher = self.committed_launcher_bundle(root)
            run_id = "fastwam-n234-vg1h1gau1-cont50k-test"
            command = [
                sys.executable,
                str(RENDERER),
                "--run-id", run_id,
                "--attempt-id", "attempt-001",
                "--treatment", "n234_vg1h1gau1_cont50k",
                "--output", str(output),
                "--bootstrap-script", "/oss-chengjuntao/source/scripts/bootstrap_offline_training_env.sh",
                "--offline-env-source-root", "/oss-chengjuntao/offline-env",
                "--offline-env-manifest", "/oss-chengjuntao/offline-env/manifest.json",
                "--offline-code-commit", "4" * 40,
                "--offline-source-bundle-relative-path", "source/FastWAM.bundle",
                "--base-python", "/usr/local/bin/python3.10",
                "--b4-source-bundle", str(bundle),
                "--b4-code-commit", code_commit,
                "--allow-local-bundle-for-tests",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(output.read_text(encoding="utf-8"))
            request = manifest["request"]
            spec = request["JobSpecs"][0]
            self.assertEqual(
                manifest["sdk_python"],
                "/mnt/workspace/tools/pai-control-py311/"
                "20260817-credentials1.0.10-dlc1.9.2/bin/python",
            )
            self.assertEqual(request["Priority"], 7)
            self.assertEqual(request["JobMaxRunningTimeMinutes"], 20160)
            self.assertEqual(spec["PodCount"], 3)
            self.assertEqual(spec["ResourceConfig"]["GPU"], "8")
            self.assertEqual(
                request["Envs"]["FASTWAM_TRAINING_TREATMENT"],
                "n234_vg1h1gau1_cont50k",
            )
            self.assertEqual(
                manifest["training_treatment"],
                "n234_vg1h1gau1_cont50k",
            )
            self.assertEqual(
                request["Envs"]["FASTWAM_B4_SOURCE_WEIGHT"],
                "/oss-chengjuntao/artifacts/"
                "fastwam-n234-vg1hub1gau1-s42-5000-r2a2-"
                "beg0t5rle97qepyw8u-a57915104bff-20260802t1820z/"
                "checkpoints/weights/step_005000.pt",
            )
            self.assertIn("cumulative 5000 to 50000", request["Description"])
            self.assertEqual(request["Settings"]["Tags"]["optimizer"], "fresh")
            self.assertEqual(
                request["Settings"]["Tags"]["schedule"],
                "cumulative-5000-to-50000-save-5000",
            )
            self.assertEqual(
                base64.b64decode(manifest["launcher_payload_base64"]),
                committed_launcher,
            )

    def test_renderer_pins_gau0_cont50k_world32_cumulative_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job.json"
            bundle, code_commit, committed_launcher = self.committed_launcher_bundle(root)
            run_id = "fastwam-n234-vg1h1gau0-cont50k-test"
            command = [
                sys.executable,
                str(RENDERER),
                "--run-id", run_id,
                "--attempt-id", "attempt-001",
                "--treatment", "n234_vg1h1gau0_cont50k",
                "--output", str(output),
                "--bootstrap-script", "/oss-chengjuntao/source/scripts/bootstrap_offline_training_env.sh",
                "--offline-env-source-root", "/oss-chengjuntao/offline-env",
                "--offline-env-manifest", "/oss-chengjuntao/offline-env/manifest.json",
                "--offline-code-commit", "4" * 40,
                "--offline-source-bundle-relative-path", "source/FastWAM.bundle",
                "--base-python", "/usr/local/bin/python3.10",
                "--b4-source-bundle", str(bundle),
                "--b4-code-commit", code_commit,
                "--allow-local-bundle-for-tests",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(output.read_text(encoding="utf-8"))
            request = manifest["request"]
            spec = request["JobSpecs"][0]
            self.assertEqual(request["Priority"], 7)
            self.assertEqual(request["JobMaxRunningTimeMinutes"], 20160)
            self.assertEqual(spec["PodCount"], 4)
            self.assertEqual(spec["ResourceConfig"]["GPU"], "8")
            self.assertEqual(
                request["Envs"]["FASTWAM_TRAINING_TREATMENT"],
                "n234_vg1h1gau0_cont50k",
            )
            self.assertEqual(manifest["training_treatment"], "n234_vg1h1gau0_cont50k")
            self.assertEqual(
                request["Envs"]["FASTWAM_B4_SOURCE_WEIGHT"],
                "/oss-chengjuntao/artifacts/fastwam-checkpoint-archives-v1/"
                "FASTWAM-MR-N234-VG1H1-S42-20260801/dlc1hqocuisxxdkb/"
                "step_005000/checkpoints/weights/step_005000.pt",
            )
            self.assertEqual(
                request["Envs"]["FASTWAM_B4_SOURCE_WEIGHT_BYTES"],
                "12045923769",
            )
            self.assertEqual(request["Envs"]["NPROC_PER_NODE"], "8")
            self.assertNotIn("FASTWAM_B4_OSS_SOURCE_ROOT", request["Envs"])
            self.assertNotIn("FASTWAM_B4_OSS_ALLOWLIST", request["Envs"])
            self.assertNotIn("FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT", request["Envs"])
            self.assertIn("cumulative 5000 to 50000", request["Description"])
            self.assertEqual(request["Settings"]["Tags"]["optimizer"], "fresh")
            self.assertEqual(request["Settings"]["Tags"]["topology"], "4x8-world32")
            self.assertEqual(
                request["Settings"]["Tags"]["initialization"],
                "GAU0-step5000-weights-only",
            )
            self.assertEqual(
                request["Settings"]["Tags"]["schedule"],
                "cumulative-5000-to-50000-save-5000",
            )
            self.assertEqual(
                base64.b64decode(manifest["launcher_payload_base64"]),
                committed_launcher,
            )

    def test_renderer_rejects_a_commit_absent_from_the_explicit_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job.json"
            bundle, _, _ = self.committed_launcher_bundle(root)
            command = [
                sys.executable,
                str(RENDERER),
                "--run-id", "fastwam-b4-missing-commit",
                "--attempt-id", "attempt-1",
                "--output", str(output),
                "--bootstrap-script", "/oss-chengjuntao/source/bootstrap.sh",
                "--offline-env-source-root", "/oss-chengjuntao/offline-env",
                "--offline-env-manifest", "/oss-chengjuntao/offline-env/manifest.json",
                "--offline-code-commit", "4" * 40,
                "--offline-source-bundle-relative-path", "source/FastWAM.bundle",
                "--base-python", "/opt/conda/bin/python3.10",
                "--b4-source-bundle", str(bundle),
                "--b4-code-commit", "5" * 40,
                "--allow-local-bundle-for-tests",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

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
        launcher_source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            'require_exact_env FASTWAM_B4_CPFS_SOURCE_ROOT '
            '"/oss-chengjuntao/cpfs-user-chengjuntao"',
            launcher_source,
        )
        self.assertIn(
            'require_exact_env FASTWAM_B4_STATS_SOURCE_ROOT '
            '"/cpfs/user/chengjuntao/datasets/robofactory_multi_robot"',
            launcher_source,
        )
        self.assertIn(
            'STATS_SOURCE_ROOT="${FASTWAM_B4_STATS_SOURCE_ROOT}"',
            launcher_source,
        )
        self.assertNotIn(
            'STATS_SOURCE_ROOT="${FASTWAM_B4_CPFS_SOURCE_ROOT}/'
            '${FASTWAM_LOCAL_DATASET_RELATIVE_ROOT}"',
            launcher_source,
        )
        self.assertNotIn(
            'require_exact_env FASTWAM_B4_CPFS_SOURCE_ROOT "/cpfs/user/chengjuntao"',
            launcher_source,
        )


if __name__ == "__main__":
    unittest.main()
