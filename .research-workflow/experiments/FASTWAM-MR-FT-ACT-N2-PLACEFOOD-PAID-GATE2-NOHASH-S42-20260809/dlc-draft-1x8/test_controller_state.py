#!/usr/bin/env python3
"""Dependency-free tests for the split local/OSS Gate2 control state."""

from __future__ import annotations

import ast
import base64
import copy
import importlib.util
import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "submit_gate2.py"
R3_LAUNCHER = HERE / "submit_gate2_r3.py"
RUNTIME = HERE / "runtime.sh"
PUBLISHER = HERE / "publish_gate2.py"
R3_WRAPPER = HERE / "submit_from_ssh970_r3.sh"


def load_launcher():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("gate2_launcher_under_test", LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load Gate2 launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_r3_controller():
    previous = sys.modules.pop("submit_gate2", None)
    sys.path.insert(0, str(HERE))
    try:
        namespace = runpy.run_path(str(R3_LAUNCHER))
        return namespace["controller"]
    finally:
        sys.path.remove(str(HERE))
        sys.modules.pop("submit_gate2", None)
        if previous is not None:
            sys.modules["submit_gate2"] = previous


class ControllerStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_launcher()
        self.temporary = tempfile.TemporaryDirectory(prefix="fastwam-gate2-controller-test-")
        root = Path(self.temporary.name)
        self.module.LOCAL_CONTROL_ROOT = root / "local-a"
        self.module.DURABLE_CONTROL_ROOT = root / "durable"
        self.attempt = "12345678-1234-4123-8123-123456789abc"
        self.body = self.module.build_request(
            Path(
                "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/"
                "controller-test-source"
            ),
            Path("/oss-chengjuntao/artifacts/controller-test-stats.json"),
            Path(
                "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"
                "controller-test-primary"
            ),
            Path(
                "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"
                "controller-test-fallback"
            ),
            self.attempt,
            trusted_runtime_bytes=b"controller-test-trusted-runtime\n",
        )
        self.binding = {
            "schema": "fastwam-dlc-prepared-binding-v1",
            "experiment_id": self.module.EXPERIMENT_ID,
            "attempt": self.attempt,
            "created_at": "2026-08-09T00:00:00Z",
            "request": self.body,
            "approved_source_root": self.body["Envs"]["FASTWAM_SOURCE_ROOT"],
            "approved_source_metadata": {"test": "stable-nohash-metadata"},
            "semantics": "immutable_request_binding_before_any_CreateJob_call",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_r3_wrapper_cannot_mutate_oss_source_with_bytecode(self) -> None:
        wrapper = R3_WRAPPER.read_text(encoding="utf-8")
        self.assertIn("export PYTHONDONTWRITEBYTECODE=1", wrapper)
        self.assertIn(
            'exec "${CONTROL_PYTHON}" -B "${SCRIPT_DIR}/submit_gate2_r3.py" "$@"',
            wrapper,
        )

    def test_durable_record_is_immutable_and_exact(self) -> None:
        path = self.module.prepared_binding_path()
        self.module.durable_exclusive_write(path, self.binding)
        observed, _ = self.module.read_json(path)
        self.assertEqual(observed, self.binding)
        with self.assertRaises(FileExistsError):
            self.module.durable_exclusive_write(path, self.binding)

    def test_local_state_restores_from_durable_ledger_and_latch(self) -> None:
        self.module.durable_exclusive_write(
            self.module.prepared_binding_path(), self.binding
        )
        state, _, _ = self.module.restore_local_state(self.attempt)
        self.assertEqual(state["phase"], "PREPARED")
        self.assertEqual(state["cloud_mutations"], 0)

        latch = self.module.acquire_submission_latch(self.attempt)
        self.assertEqual(latch["create_call_disposition"], "MAY_HAVE_BEEN_SENT")
        self.module.LOCAL_CONTROL_ROOT = Path(self.temporary.name) / "local-after-restart"
        restored, _, _ = self.module.restore_local_state(self.attempt)
        self.assertEqual(restored["phase"], "AMBIGUOUS")
        self.assertEqual(restored["cloud_mutations"], 1)

    def test_same_attempt_cannot_reuse_existing_submission_latch(self) -> None:
        self.module.acquire_submission_latch(self.attempt)
        with self.assertRaises(FileExistsError):
            self.module.acquire_submission_latch(self.attempt)

    def test_snapshot_has_no_workspace_ceiling_or_e38_dependency(self) -> None:
        module = self.module
        module.list_jobs = lambda *args: [{"JobId": "dlc-unrelated-64gpu"}]
        module.get_job = lambda *args: {
            "JobId": "dlc-unrelated-64gpu",
            "DisplayName": "unrelated",
            "Status": "Running",
            "JobSpecs": [
                {"PodCount": 1, "ResourceConfig": {"GPU": 64}}
            ],
        }

        observed = module.snapshot(None, None, None, self.body, 1)
        self.assertEqual(observed["active_gpu_count"], 64)
        self.assertEqual(observed["requested_gpu_count"], 8)
        self.assertEqual(observed["post_submit_gpu_count"], 72)
        self.assertEqual(
            observed["resource_policy"],
            "exactly_8_gpus_per_job_no_artificial_workspace_ceiling",
        )
        self.assertNotIn("ceiling", observed)
        self.assertNotIn("expected_existing_job_id", observed)

    def test_exact_identity_covers_critical_request_fields(self) -> None:
        observed = copy.deepcopy(self.body)
        observed["ServerManagedStatus"] = "Creating"
        observed["Settings"]["ServerManagedSetting"] = True
        observed["JobSpecs"][0]["ServerManagedSpecField"] = "accepted"
        self.assertTrue(self.module.exact_identity(observed, self.body))
        mutations = (
            ("Envs", "FASTWAM_INITIAL_CHECKPOINT"),
            ("Envs", self.module.TRUSTED_RUNTIME_B64_ENV),
            ("DataSources", None),
            ("JobSpecs", None),
            ("Settings", None),
            ("UserCommand", None),
        )
        for top, nested in mutations:
            changed = copy.deepcopy(observed)
            if nested is None:
                changed[top] = "changed"
            else:
                changed[top][nested] = "changed"
            with self.subTest(field=f"{top}.{nested}" if nested else top):
                self.assertFalse(self.module.exact_identity(changed, self.body))

        critical_spec_fields = (
            "Image", "Type", "PodCount", "ResourceConfig", "RestartPolicy",
            "LocalMountSpecs", "StartupDependencies", "ElasticSpotSpecs",
        )
        for field in critical_spec_fields:
            changed = copy.deepcopy(observed)
            changed["JobSpecs"][0][field] = "changed"
            with self.subTest(field=f"JobSpecs.{field}"):
                self.assertFalse(self.module.exact_identity(changed, self.body))

    def test_request_carries_exact_trusted_runtime_bytes(self) -> None:
        envs = self.body["Envs"]
        payload = base64.b64decode(
            envs[self.module.TRUSTED_RUNTIME_B64_ENV].encode("ascii"),
            validate=True,
        )
        self.assertEqual(payload, b"controller-test-trusted-runtime\n")
        self.assertEqual(
            envs[self.module.TRUSTED_RUNTIME_BYTES_ENV], str(len(payload))
        )
        self.assertEqual(
            self.body["UserCommand"], self.module.TRUSTED_BOOTSTRAP_COMMAND
        )
        self.module.validate_request(self.body, validate_live_inputs=False)

        changed = copy.deepcopy(self.body)
        changed["Envs"][self.module.TRUSTED_RUNTIME_BYTES_ENV] = str(len(payload) + 1)
        with self.assertRaisesRegex(RuntimeError, "byte count mismatch"):
            self.module.validate_request(changed, validate_live_inputs=False)

    def test_request_respects_env_limit_and_bootstrap_uses_allowlist(self) -> None:
        self.assertLessEqual(len(self.body["Envs"]), 20)
        command = self.body["UserCommand"]
        self.assertIn("/usr/bin/python3 -I -S", command)
        self.assertEqual(command.count("'"), 2)
        self.assertTrue(command.startswith("unset BASH_ENV ENV PYTHONHOME"))
        self.assertIn("LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT", command)
        self.assertIn("clean={key:os.environ[key] for key in allowed", command)
        self.assertNotIn("clean=dict(os.environ)", command)
        for forbidden in (
            "LD_AUDIT",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "PYTHONHOME",
            "PYTHONSTARTUP",
        ):
            self.assertNotIn(forbidden, self.module.BOOTSTRAP_ALLOWED_ENV)

    def test_runtime_stages_only_the_fixed_vae_and_forces_offline_resolution(self) -> None:
        text = RUNTIME.read_text()
        expected = (
            "/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/model-cache/"
            "DiffSynth-Studio/Wan-Series-Converted-Safetensors/"
            "Wan2.2_VAE.safetensors"
        )
        self.assertEqual(str(self.module.VAE_SOURCE), expected)
        self.assertEqual(self.module.VAE_SOURCE_BYTES, 1_409_401_152)
        self.assertIn(f'EXPECTED_VAE_SOURCE="{expected}"', text)
        self.assertIn("EXPECTED_VAE_SOURCE_BYTES=1409401152", text)
        self.assertIn('export PYTHONPATH="${LOCAL_SOURCE}/src"', text)
        self.assertIn('export DIFFSYNTH_MODEL_BASE_PATH="${LOCAL_MODEL_CACHE}"', text)
        self.assertIn("export DIFFSYNTH_SKIP_DOWNLOAD=true", text)
        self.assertIn("export HF_HUB_OFFLINE=1", text)
        self.assertIn('"direct_file_byte_comparison": "passed"', text)
        self.assertIn(
            '"vae_staging.json": stage / "vae_staging.json"',
            PUBLISHER.read_text(encoding="utf-8"),
        )
        self.assertIn('/usr/bin/python3 -I -S - "${TRUSTED_RUNTIME_PATH}"', text)

    def test_source_content_binding_detects_same_size_same_mtime_replacement(self) -> None:
        source = Path(self.temporary.name) / "source-binding"
        source.mkdir()
        target = source / "payload.bin"
        target.write_bytes(b"first")
        original = target.stat()
        first = self.module.source_snapshot_metadata(source)

        target.write_bytes(b"other")
        os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
        second = self.module.source_snapshot_metadata(source)

        first_entry = next(item for item in first["entries"] if item["path"] == "payload.bin")
        second_entry = next(item for item in second["entries"] if item["path"] == "payload.bin")
        self.assertEqual(first_entry["size"], second_entry["size"])
        self.assertEqual(first_entry["mtime_ns"], second_entry["mtime_ns"])
        self.assertNotEqual(first_entry["content_b64"], second_entry["content_b64"])
        self.assertEqual(
            base64.b64decode(first_entry["content_b64"], validate=True), b"first"
        )

    def test_launcher_has_one_mutating_sdk_call(self) -> None:
        tree = ast.parse(LAUNCHER.read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_job_with_options"
        ]
        self.assertEqual(len(calls), 1)

    def test_runtime_rechecks_prepared_source_binding_around_copy(self) -> None:
        text = RUNTIME.read_text()
        copy_position = text.index(
            'cp -a -- "${FASTWAM_SOURCE_ROOT}/." "${LOCAL_SOURCE}/"'
        )
        first_check = text.index("validate_prepared_source_binding")
        second_check = text.index("validate_prepared_source_binding", first_check + 1)
        third_check = text.index("validate_prepared_source_binding", second_check + 1)

        self.assertLess(second_check, copy_position)
        self.assertGreater(third_check, copy_position)
        self.assertIn('"content_b64": base64.b64encode', text)
        self.assertIn("executing runtime differs from request-carried trusted bytes", text)
        self.assertIn("FASTWAM_PREPARED_BINDING_PATH", text)

    def test_r3_variant_binds_explicit_unique_oss_snapshot_and_mounts_inputs(self) -> None:
        module = load_r3_controller()
        expected_source = Path(
            "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/"
            "fastwam-mr-gate2-explicit-snapshot-20260811"
        )
        body = module.build_request(
            expected_source,
            Path("/oss-chengjuntao/artifacts/controller-test-stats.json"),
            Path(
                "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"
                "controller-test-primary"
            ),
            Path(
                "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"
                "controller-test-fallback"
            ),
            self.attempt,
            trusted_runtime_bytes=b"r3-controller-test-trusted-runtime\n",
        )

        module.validate_request(body, validate_live_inputs=False)
        self.assertIsNone(module.APPROVED_SOURCE_ROOT)
        self.assertEqual(body["Envs"]["FASTWAM_SOURCE_ROOT"], str(expected_source))
        self.assertEqual(
            body["DataSources"][0],
            {
                "DataSourceId": module.CPFS_SOURCE,
                "MountAccess": "RO",
                "MountPath": "/cpfs/user/chengjuntao",
            },
        )
        self.assertEqual(
            body["DataSources"][1],
            {
                "DataSourceId": module.OSS_SOURCE,
                "MountAccess": "RW",
                "MountPath": "/oss-chengjuntao",
            },
        )

        nested_source = expected_source / "nested"
        body["Envs"]["FASTWAM_SOURCE_ROOT"] = str(nested_source)
        body["Envs"]["FASTWAM_GATE2_ENTRYPOINT"] = str(
            nested_source / module.ENTRYPOINT_REL
        )
        with self.assertRaisesRegex(RuntimeError, "unique direct child"):
            module.validate_request(body, validate_live_inputs=False)

    def test_prepare_freezes_the_explicit_direct_child_snapshot(self) -> None:
        module = self.module
        fixture_root = Path(self.temporary.name) / "prepare-fixture"
        module.SOURCE_PREFIX = fixture_root / "source-snapshots"
        module.STATS_SOURCE_PREFIX = fixture_root / "oss"
        module.GAUSSIAN_CACHE_PREFIX = fixture_root / "oss" / "gaussian"
        module.OUTPUT_PREFIX = fixture_root / "oss" / "outputs"
        module.VAE_SOURCE = fixture_root / "oss" / "model-cache" / "vae.safetensors"
        module.VAE_SOURCE_BYTES = 16
        module.APPROVED_SOURCE_ROOT = None

        source = module.SOURCE_PREFIX / "explicit-unique-snapshot-20260811"
        required_files = (
            source / module.ENTRYPOINT_REL,
            source / module.REAL_PREFLIGHT_REL,
            source / module.STRUCTURED_EVIDENCE_REL,
            source / "scripts/train.py",
            source / "scripts/accelerate_configs/accelerate_zero2_ds.yaml",
            source / "configs/task" / f"{module.TASK}.yaml",
        )
        for path in required_files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture for {path.name}\n", encoding="utf-8")

        module.VAE_SOURCE.parent.mkdir(parents=True, exist_ok=True)
        module.VAE_SOURCE.write_bytes(b"fixed-vae-source")
        stats = module.STATS_SOURCE_PREFIX / "normalization-stats.json"
        stats.parent.mkdir(parents=True, exist_ok=True)
        stats.write_text("{}\n", encoding="utf-8")

        def make_cache(name, kind, height, width, selection):
            root = module.GAUSSIAN_CACHE_PREFIX / name
            root.mkdir(parents=True)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": {
                            "cache_kind": kind,
                            "channel_count": 13,
                            "height": height,
                            "width": width,
                        },
                        "selection": {"mode": selection},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "COMPLETE").write_text("complete\n", encoding="utf-8")
            return root

        primary = make_cache("primary", "compact", 28, 40, "index")
        fallback = make_cache("fallback", "canonical", 240, 320, "all")
        module.prepare(str(source), str(stats), str(primary), str(fallback))

        binding, _ = module.read_json(module.prepared_binding_path())
        self.assertEqual(binding["approved_source_root"], str(source))
        self.assertEqual(
            binding["request"]["Envs"]["FASTWAM_SOURCE_ROOT"], str(source)
        )
        self.assertEqual(
            Path(binding["request"]["Envs"]["FASTWAM_SOURCE_ROOT"]).parent,
            module.SOURCE_PREFIX,
        )

    def test_execute_persists_response_and_ack_before_terminal_local_state(self) -> None:
        module = self.module
        module.durable_exclusive_write(module.prepared_binding_path(), self.binding)
        module.restore_local_state(self.attempt)

        class FakeRequest:
            def from_map(self, body):
                self.body = body
                return self

            def validate(self):
                return None

            def to_map(self):
                return self.body

        class FakeModels:
            CreateJobRequest = FakeRequest

        class FakeRuntime:
            def __init__(self, **values):
                self.values = values

        class FakeBody:
            def to_map(self):
                return {"JobId": "dlc-gate2-test", "RequestId": "request-gate2-test"}

        class FakeResponse:
            body = FakeBody()

        class FakeClient:
            def __init__(self):
                self.create_calls = 0

            def create_job_with_options(self, request, headers, runtime):
                self.create_calls += 1
                return FakeResponse()

        client = FakeClient()
        observed = copy.deepcopy(self.body)
        observed.update({"JobId": "dlc-gate2-test", "Status": "Creating"})
        snapshot = {
            "active_jobs": [
                {
                    "job_id": "dlc-unrelated-running-job",
                    "display_name": "unrelated",
                    "status": "Running",
                    "gpus": 24,
                }
            ],
            "active_gpu_count": 24,
        }
        module.load_sdk = lambda: (client, FakeModels, FakeRuntime)
        module.snapshot = lambda *args: dict(snapshot)
        module.get_job = lambda *args: dict(observed)
        module.time.sleep = lambda seconds: None
        module.assert_source_root = lambda value: Path(value)
        module.source_snapshot_metadata = lambda source: {"test": "stable-nohash-metadata"}
        real_validate_request = module.validate_request
        module.validate_request = lambda body, models=None, **kwargs: real_validate_request(
            body, models, validate_live_inputs=False
        )

        module.execute(self.attempt)
        state, _, _ = module.load_state(self.attempt)
        response, _ = module.read_json(module.create_response_path())
        acknowledgement, _ = module.read_json(module.acknowledgement_path())
        self.assertEqual(client.create_calls, 1)
        self.assertEqual(state["phase"], "ACK")
        self.assertEqual(response["job_id"], "dlc-gate2-test")
        self.assertEqual(acknowledgement["identity_check"], "exact_request_identity_passed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
