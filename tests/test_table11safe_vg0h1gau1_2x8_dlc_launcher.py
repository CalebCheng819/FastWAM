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
LAUNCHER = REPO / "scripts" / "launch_table11safe_vg0h1gau1_2x8_dlc.sh"
RENDERER = REPO / "scripts" / "render_table11safe_vg0h1gau1_2x8_dlc_job.py"
SUBMITTER = REPO / "scripts" / "submit_table11safe_vg0h1gau1_formal_once.py"
PREFLIGHT_SUBMITTER = (
    REPO / "scripts" / "submit_table11safe_vg0h1gau1_preflight_once.py"
)
TRAINER = REPO / "src" / "fastwam" / "trainer.py"


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

        weight = root / "libero_uncond_2cam224.pt"
        weight.write_bytes(b"fixture-weight")
        env = os.environ.copy()
        env.update(
            {
                "RUN_ID": "fastwam-table11-launcher-test",
                "FASTWAM_TABLE11_ATTEMPT_ID": "attempt-1",
                "FASTWAM_TABLE11_REPO_ROOT": str(REPO),
                "FASTWAM_TABLE11_PYTHON": os.environ.get(
                    "FASTWAM_TEST_HYDRA_PYTHON", sys.executable
                ),
                "FASTWAM_TABLE11_OFFLINE_ENV_READY": "1",
                "FASTWAM_TABLE11_TEST_MODE": "1",
                "FASTWAM_TABLE11_DRY_RUN": "1",
                "FASTWAM_TABLE11_RUN_MODE": "formal",
                "FASTWAM_TABLE11_OUTPUT_DIR": str(root / "output"),
                "FASTWAM_TABLE11_SOURCE_WEIGHT": str(weight),
                "FASTWAM_TABLE11_SOURCE_WEIGHT_BYTES": str(weight.stat().st_size),
                "FASTWAM_TABLE11_DATASET_ROOT": str(dataset),
                "FASTWAM_TABLE11_STATS_PATH": str(stats),
                "FASTWAM_TABLE11_TEXT_CACHE_DIR": str(text_cache),
                "FASTWAM_TABLE11_GAUSSIAN_CACHE_DIR": str(gaussian),
                "FASTWAM_TABLE11_EXPECTED_H5_FILES": "11",
                "FASTWAM_TABLE11_CODE_COMMIT": "1" * 40,
                "WORLD_SIZE": "2",
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

    def test_valid_contract_resolves_world16_action_only_scratch_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_launcher(self.fixture(Path(directory)))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--num_machines 2", result.stdout)
            self.assertIn("--num_processes 16", result.stdout)
            self.assertIn(
                "task=robofactory_table11_vg0_hub1_gau1_scratch50k_224_1e-4",
                result.stdout,
            )
            self.assertIn("+scale=robofactory_multi_robot_16gpu_scratch50k", result.stdout)
            self.assertIn("run_initial_global_step=0", result.stdout)
            self.assertIn("weights_only_warm_start.enabled=false", result.stdout)
            self.assertIn("table11 safe config gate: world=16 global_batch=16", result.stdout)
            self.assertIn("save_every=1000 rolling_keep_last=2", result.stdout)

    def test_topology_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env["WORLD_SIZE"] = "3"
            result = self.run_launcher(env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("WORLD_SIZE must be the DLC worker count 2", result.stderr)

    def test_one_step_preflight_resolves_world8_and_disables_final_save(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            env.update(
                {
                    "FASTWAM_TABLE11_RUN_MODE": "preflight-one-step",
                    "WORLD_SIZE": "1",
                }
            )
            result = self.run_launcher(env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--num_machines 1", result.stdout)
            self.assertIn("--num_processes 8", result.stdout)
            self.assertIn("max_steps=1", result.stdout)
            self.assertIn("run_initial_global_step=0", result.stdout)
            self.assertIn("save_every=0", result.stdout)
            self.assertIn("checkpoint_keep_last=0", result.stdout)
            self.assertIn("eval_every=0", result.stdout)
            self.assertIn("log_every=1", result.stdout)
            self.assertIn("save_training_state=false", result.stdout)
            self.assertIn("save_final_checkpoint=false", result.stdout)
            self.assertIn("seal_training_run=false", result.stdout)

    def test_preflight_publication_is_fail_closed_and_complete_last(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('pipeline_status=("${PIPESTATUS[@]}")', source)
        self.assertIn(
            'grep -Fq -- "FASTWAM_GENERIC_BASE_LOAD=PASS before_prepare=true"',
            source,
        )
        self.assertIn(
            'grep -Fq -- "FASTWAM_TRAINING_START initial_global_step=0 '
            'max_steps=1 optimizer_steps_this_run=1"',
            source,
        )
        self.assertIn(
            'grep -Fq -- "FASTWAM_OPTIMIZER_STEP global_step=1 max_steps=1"',
            source,
        )
        self.assertNotIn(
            'grep -Fq -- "Loading weight checkpoint before optimizer/DeepSpeed '
            'initialization: ${LOCAL_WEIGHT}"',
            source,
        )
        terminal_index = source.index("publish(terminal_path, terminal)")
        allowlist_index = source.index("validate_layout(include_complete=False)")
        complete_index = source.index("publish(complete_path, complete)")
        final_allowlist_index = source.index("validate_layout(include_complete=True)")
        self.assertLess(terminal_index, allowlist_index)
        self.assertLess(allowlist_index, complete_index)
        self.assertLess(complete_index, final_allowlist_index)
        self.assertIn('ready_name = f".config.yaml.ready.stat_cmp.{attempt_id}"', source)
        self.assertIn('checkpoint_children != {"state", "weights"}', source)
        self.assertIn('raise SystemExit(f"preflight directory is not empty:', source)
        self.assertIn('"schema": "fastwam-runtime-file-barrier-stat-cmp-v2"', source)
        self.assertIn(
            '"schema": "fastwam-table11safe-realdata-scratch-preflight-terminal-v1"',
            source,
        )
        self.assertIn('"initial_global_step": 0', source)
        self.assertIn('"final_global_step": 1', source)
        self.assertIn('"optimizer": "fresh"', source)
        self.assertIn('"scheduler": "fresh"', source)

    def test_generic_base_receipt_is_emitted_after_successful_weight_load(self) -> None:
        source = TRAINER.read_text(encoding="utf-8")
        load_index = source.index(
            "self.model.load_checkpoint(str(resume_path), optimizer=None)"
        )
        receipt_index = source.index(
            'logger.warning("FASTWAM_GENERIC_BASE_LOAD=PASS before_prepare=true")'
        )
        fresh_optimizer_index = source.index(
            '"optimizer/scheduler/step are intentionally not restored."'
        )
        training_start_index = source.index(
            '"FASTWAM_TRAINING_START initial_global_step=%d max_steps=%d '
            'optimizer_steps_this_run=%d"'
        )
        optimizer_step_index = source.index('"FASTWAM_OPTIMIZER_STEP "')
        self.assertIn("os.write(1, receipt)", source)
        self.assertLess(load_index, receipt_index)
        self.assertLess(receipt_index, fresh_optimizer_index)
        self.assertLess(fresh_optimizer_index, training_start_index)
        self.assertLess(training_start_index, optimizer_step_index)

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

    def test_renderer_is_pure_and_pins_priority7_world16_contract(self) -> None:
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
            self.assertEqual(request["JobSpecs"][0]["PodCount"], 2)
            self.assertEqual(request["JobSpecs"][0]["ResourceConfig"]["GPU"], "8")
            self.assertEqual(
                {(item["MountPath"], item["MountAccess"]) for item in request["DataSources"]},
                {("/oss-chengjuntao", "RW")},
            )
            self.assertEqual(manifest["batch_contract"]["reference_global_batch"], 24)
            self.assertEqual(manifest["batch_contract"]["replica_global_batch"], 16)
            self.assertFalse(manifest["batch_contract"]["sample_budget_equivalent"])
            self.assertEqual(manifest["batch_contract"]["optimizer_updates"], 50000)
            self.assertEqual(
                request["Envs"]["FASTWAM_TABLE11_SOURCE_WEIGHT"],
                "/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/"
                "yuanty-fastwam-139eebb6d90cdd9bdbbe465f72c6edc9ad5a518a/"
                "libero_uncond_2cam224.pt",
            )
            self.assertEqual(request["Envs"]["FASTWAM_TABLE11_SOURCE_WEIGHT_BYTES"], "12041735140")
            self.assertEqual(
                request["Settings"]["Tags"]["initialization"],
                "official-generic-pretrained-model-weights",
            )
            self.assertEqual(request["Settings"]["Tags"]["optimizer"], "fresh")
            self.assertIn("optimizer steps 0 to 50000", request["Description"])
            self.assertEqual(
                base64.b64decode(manifest["launcher_payload_base64"]), launcher_bytes
            )

    def test_renderer_can_render_separate_world8_one_step_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "preflight.json"
            bundle, commit, _ = self.committed_launcher_bundle(root)
            command = [
                sys.executable,
                str(RENDERER),
                "--run-id", "fastwam-table11-preflight-test",
                "--attempt-id", "preflight-1",
                "--output", str(output),
                "--bootstrap-script", "/oss-chengjuntao/source/bootstrap.sh",
                "--offline-env-source-root", "/oss-chengjuntao/offline-env",
                "--offline-env-manifest", "/oss-chengjuntao/offline-env/manifest.json",
                "--offline-code-commit", "2" * 40,
                "--offline-source-bundle-relative-path", "source/FastWAM.bundle",
                "--base-python", "/opt/conda/bin/python3.10",
                "--source-bundle", str(bundle),
                "--code-commit", commit,
                "--allow-local-bundle-for-tests",
                "--preflight-one-step",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            request = json.loads(output.read_text(encoding="utf-8"))["request"]
            self.assertEqual(request["Priority"], 7)
            self.assertEqual(request["JobSpecs"][0]["PodCount"], 1)
            self.assertEqual(
                request["Settings"]["Tags"]["schedule"],
                "optimizer-0-to-1-no-checkpoint",
            )
            self.assertEqual(request["Envs"]["FASTWAM_TABLE11_RUN_MODE"], "preflight-one-step")
            self.assertIn("optimizer step 0 to 1", request["Description"])

    def test_operational_contract_forbids_old_n234_checkpoint(self) -> None:
        paths = [
            RENDERER,
            SUBMITTER,
            REPO
            / "configs"
            / "task"
            / "robofactory_multi_robot_vg0_hub1_gau1_scratch50k_224_1e-4.yaml",
            REPO
            / "configs"
            / "task"
            / "robofactory_table11_vg0_hub1_gau1_scratch50k_224_1e-4.yaml",
            REPO / "configs" / "scale" / "robofactory_multi_robot_16gpu_scratch50k.yaml",
        ]
        source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        self.assertNotIn("step_005000.pt", source)
        self.assertNotIn("FASTWAM_B4_BASE_CHECKPOINT", source)
        launcher = LAUNCHER.read_text(encoding="utf-8")
        preflight_submitter = PREFLIGHT_SUBMITTER.read_text(encoding="utf-8")
        self.assertIn(
            '! grep -Fq -- "step_005000.pt" "${PREFLIGHT_LOG}"', launcher
        )
        self.assertIn(
            'common.require("step_005000.pt" not in log', preflight_submitter
        )
        self.assertIn("libero_uncond_2cam224.pt", source)
        self.assertIn("num_workers: 2", source)
        self.assertIn("prefetch_factor: 1", source)
        self.assertIn("persistent_workers: false", source)
        self.assertIn("save_every: 1000", source)
        self.assertIn("checkpoint_keep_last: 2", source)
        self.assertIn("checkpoint_keep_last=2", source)
        self.assertIn("checkpoint_retention=rolling-complete-resumable-tuples", source)
        self.assertIn("weights_only_warm_start:\n  enabled: false", source)
        self.assertIn("trainable_scope: action", source)
        self.assertIn("training_mode: action_only_cache", source)
        self.assertIn("load_future_video: false", source)
        self.assertIn("hub_enabled: true", source)
        self.assertIn("enable_gaussian: true", source)
        self.assertIn("lambda_video: 0.0", source)
        self.assertIn("lambda_action: 1.0", source)
        self.assertIn("optimizer-0-to-50000-save-1000", source)
        self.assertNotIn("optimizer-0-to-50000-save-5000", source)

    def test_formal_controller_accepts_only_frozen_bundle_ref_and_commit(self) -> None:
        source = SUBMITTER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "validate_source_bundle_heads"
        )
        namespace = {
            "BUNDLE_REF": "refs/bundles/fastwam-table11safe-vg0h1gau1-scratch50k-r1",
            "require": lambda condition, message: (
                None
                if condition
                else (_ for _ in ()).throw(RuntimeError(message))
            ),
        }
        module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
        exec(compile(module, str(SUBMITTER), "exec"), namespace)
        validate = namespace["validate_source_bundle_heads"]
        commit = "1" * 40
        frozen_ref = "refs/bundles/fastwam-table11safe-vg0h1gau1-scratch50k-r1"

        validate([f"{commit} {frozen_ref}"], commit)
        rejected = (
            [f"{commit} HEAD"],
            [f"{'0' * 40} {frozen_ref}"],
            [f"{commit} refs/bundles/unexpected"],
            [f"{commit} {frozen_ref}", f"{commit} HEAD"],
        )
        for heads in rejected:
            with self.subTest(heads=heads), self.assertRaises(RuntimeError):
                validate(heads, commit)

    def test_dataloader_runtime_diagnostics_are_enabled(self) -> None:
        source = (REPO / "src" / "fastwam" / "trainer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("DataLoader runtime:", source)
        self.assertIn("DataLoader worker started:", source)
        self.assertIn('os.statvfs("/dev/shm")', source)
        self.assertIn("resource.RLIMIT_NOFILE", source)
        self.assertIn("resource.RLIMIT_MEMLOCK", source)

    def test_preflight_controller_latches_before_exactly_one_create(self) -> None:
        source = PREFLIGHT_SUBMITTER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        create_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_job"
        ]
        self.assertEqual(len(create_calls), 1)
        self.assertLess(
            source.index("common.write_exclusive(LATCH, latch)"),
            source.index("response = dlc.create_job(request)"),
        )
        self.assertIn("LATCHED_CREATEJOB_ONCE_NEVER_RETRY", source)
        self.assertIn("CREATE_JOB_EXCEPTION_AMBIGUOUS_DO_NOT_RETRY", source)
        self.assertNotIn("stop_job(", source)
        self.assertNotIn("update_job(", source)

    def test_submission_is_blocked_until_the_new_commit_is_frozen(self) -> None:
        formal = SUBMITTER.read_text(encoding="utf-8")
        preflight = PREFLIGHT_SUBMITTER.read_text(encoding="utf-8")
        self.assertIn("COMMIT != UNPUBLISHED_COMMIT", formal)
        self.assertIn("COMMIT != common.UNPUBLISHED_COMMIT", preflight)

    def test_preflight_controller_requires_provider_and_output_pass(self) -> None:
        source = PREFLIGHT_SUBMITTER.read_text(encoding="utf-8")
        self.assertIn('if status == "Succeeded":', source)
        self.assertIn('if not os.path.lexists(pathlib.Path(OUTPUT_DIR) / "COMPLETE"):', source)
        validation_index = source.index("output = validate_output()")
        pass_index = source.index(
            'conclusion = {\n            "status": "PASS"', validation_index
        )
        self.assertLess(validation_index, pass_index)
        self.assertIn('"initial_global_step": 0', source)
        self.assertIn('"final_global_step": 1', source)
        self.assertIn('"optimizer_steps_this_run": 1', source)
        self.assertIn('"optimizer": "fresh"', source)
        self.assertIn('"scheduler": "fresh"', source)
        self.assertIn('"sample_budget_equivalent": "false"', source)
        self.assertIn(
            '"gaussian_cache_dir": common.GAUSSIAN_CACHE_DIR,', source
        )
        self.assertNotIn('"gaussian_cache_dir": common.GAUSSIAN_CACHE,', source)
        self.assertIn('"Loading weight checkpoint before",', source)
        self.assertIn('"optimizer/DeepSpeed initialization:",', source)
        self.assertIn(
            '"FASTWAM_GENERIC_BASE_LOAD=PASS before_prepare=true",', source
        )
        self.assertNotIn(
            'f"Loading weight checkpoint before optimizer/DeepSpeed initialization:',
            source,
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
