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
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "submit_gate2.py"
R3_LAUNCHER = HERE / "submit_gate2_r3.py"
R4_LAUNCHER = HERE / "submit_gate2_r4.py"
R5_LAUNCHER = HERE / "submit_gate2_r5.py"
R6_LAUNCHER = HERE / "submit_gate2_r6.py"
R7_LAUNCHER = HERE / "submit_gate2_r7.py"
R8_LAUNCHER = HERE / "submit_gate2_r8.py"
READONLY_MONITOR = HERE / "monitor_gate2_readonly.py"
RUNTIME = HERE / "runtime.sh"
PUBLISHER = HERE / "publish_gate2.py"
R3_WRAPPER = HERE / "submit_from_ssh970_r3.sh"
R4_WRAPPER = HERE / "submit_from_ssh970_r4.sh"
R5_WRAPPER = HERE / "submit_from_ssh970_r5.sh"
R6_WRAPPER = HERE / "submit_from_ssh970_r6.sh"
R7_WRAPPER = HERE / "submit_from_ssh970_r7.sh"
R8_WRAPPER = HERE / "submit_from_ssh970_r8.sh"


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


def load_r4_controller():
    previous = sys.modules.pop("submit_gate2", None)
    sys.path.insert(0, str(HERE))
    try:
        namespace = runpy.run_path(str(R4_LAUNCHER))
        return namespace["controller"]
    finally:
        sys.path.remove(str(HERE))
        sys.modules.pop("submit_gate2", None)
        if previous is not None:
            sys.modules["submit_gate2"] = previous


def load_r5_controller():
    previous = sys.modules.pop("submit_gate2", None)
    sys.path.insert(0, str(HERE))
    try:
        namespace = runpy.run_path(str(R5_LAUNCHER))
        return namespace["controller"]
    finally:
        sys.path.remove(str(HERE))
        sys.modules.pop("submit_gate2", None)
        if previous is not None:
            sys.modules["submit_gate2"] = previous


def load_r6_controller():
    previous = sys.modules.pop("submit_gate2", None)
    sys.path.insert(0, str(HERE))
    try:
        namespace = runpy.run_path(str(R6_LAUNCHER))
        return namespace["controller"]
    finally:
        sys.path.remove(str(HERE))
        sys.modules.pop("submit_gate2", None)
        if previous is not None:
            sys.modules["submit_gate2"] = previous


def load_r7_controller():
    previous = sys.modules.pop("submit_gate2", None)
    sys.path.insert(0, str(HERE))
    try:
        namespace = runpy.run_path(str(R7_LAUNCHER))
        return namespace["controller"]
    finally:
        sys.path.remove(str(HERE))
        sys.modules.pop("submit_gate2", None)
        if previous is not None:
            sys.modules["submit_gate2"] = previous


def load_r8_controller():
    previous = sys.modules.pop("submit_gate2", None)
    sys.path.insert(0, str(HERE))
    try:
        namespace = runpy.run_path(str(R8_LAUNCHER))
        return namespace["controller"]
    finally:
        sys.path.remove(str(HERE))
        sys.modules.pop("submit_gate2", None)
        if previous is not None:
            sys.modules["submit_gate2"] = previous


def observed_getjob_shape(request: dict) -> dict:
    """Identity-relevant shape observed for failed job dlchdvsayhmjbn2w."""

    observed = copy.deepcopy(request)
    observed["JobId"] = "dlchdvsayhmjbn2w"
    observed["Status"] = "Failed"
    observed.pop("JobMaxRunningTimeMinutes")
    observed.pop("SuccessPolicy")
    observed["CustomEnvs"] = [
        {"Key": key, "Value": value, "Visible": "public"}
        for key, value in reversed(list(request["Envs"].items()))
    ]
    observed["DataSources"] = [
        {
            "DataSourceId": item["DataSourceId"],
            "MountPath": item["MountPath"],
            "Uri": "",
        }
        for item in request["DataSources"]
    ]
    observed["Settings"]["ServerManagedSetting"] = True
    observed["JobSpecs"][0].update(
        {"AssignNodeSpec": {}, "EcsSpec": {}, "ImageConfig": {}}
    )
    observed["JobSpecs"][0]["ResourceConfig"]["GPUType"] = ""
    return observed


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
            "approved_source_metadata": {
                "schema": "fastwam-nohash-source-content-binding-v3",
                "approved_source_root": self.body["Envs"]["FASTWAM_SOURCE_ROOT"],
                "entries": [{"path": ".", "kind": "directory"}],
            },
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

    def test_r4_has_new_identity_wrapper_and_no_r3_attempt_reuse(self) -> None:
        module = load_r4_controller()
        expected = (
            "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-"
            "R4-S42-20260811"
        )
        retired_attempt = "7bcd3b16-d73d-4538-add9-394276b9f15f"
        retired_job = "dlchdvsayhmjbn2w"
        wrapper = R4_WRAPPER.read_text(encoding="utf-8")
        launcher = R4_LAUNCHER.read_text(encoding="utf-8")
        runtime = RUNTIME.read_text(encoding="utf-8")
        self.assertEqual(module.EXPERIMENT_ID, expected)
        self.assertEqual(module.SUBMISSION_TAG_PREFIX, "fastwam-gate2-nohash-r4-s42")
        self.assertEqual(module.DISPLAY_NAME_PREFIX, "fw-g2-nh-r4-s42")
        self.assertEqual(module.CONTROL_ENTRYPOINT, "submit_from_ssh970_r4.sh")
        self.assertIn('[[ -z "${SSH_CONNECTION:-}" ]]', wrapper)
        self.assertIn('CONTROL_PYTHON_TARGET="/usr/local/bin/python3.12"', wrapper)
        self.assertIn('[[ ! -L "${CONTROL_PYTHON}"', wrapper)
        self.assertIn('realpath -e -- "${CONTROL_PYTHON}"', wrapper)
        self.assertIn(
            '"${CONTROL_PYTHON_REAL}" != "${CONTROL_PYTHON_TARGET}"', wrapper
        )
        self.assertIn('-L "${CONTROL_PYTHON_TARGET}"', wrapper)
        self.assertIn(
            "import alibabacloud_credentials,alibabacloud_pai_dlc20201203,"
            "alibabacloud_tea_openapi",
            wrapper,
        )
        self.assertIn(
            'exec "${CONTROL_PYTHON}" -B -I "${SCRIPT_DIR}/submit_gate2_r4.py" "$@"',
            wrapper,
        )
        for retired in (retired_attempt, retired_job):
            self.assertNotIn(retired, wrapper)
            self.assertNotIn(retired, launcher)
            self.assertNotIn(retired, runtime)

        expected_source = Path(
            "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/"
            "fastwam-action-n234-formal-20260811-r3"
        )
        self.assertEqual(module.APPROVED_SOURCE_ROOT, expected_source)
        body = module.build_request(
            expected_source,
            Path("/oss-chengjuntao/artifacts/r4-controller-test-stats.json"),
            Path(
                "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"
                "r4-controller-test-primary"
            ),
            Path(
                "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"
                "r4-controller-test-fallback"
            ),
            "87654321-4321-4123-8123-cba987654321",
            trusted_runtime_bytes=b"r4-controller-test-runtime\n",
        )
        module.validate_request(body, validate_live_inputs=False)
        self.assertEqual(body["Envs"]["FASTWAM_EXPERIMENT_ID"], expected)
        self.assertIn("fastwam-gate2-nohash-r4-s42-", body["Envs"]["FASTWAM_SUBMISSION_TAG"])
        self.assertNotIn("r3", body["Envs"]["FASTWAM_OSS_OUTPUT_ROOT"].lower())

        body["Envs"]["FASTWAM_SOURCE_ROOT"] = str(expected_source.with_name("retired-r2"))
        with self.assertRaisesRegex(RuntimeError, "exact approved snapshot"):
            module.validate_request(body, validate_live_inputs=False)

    def test_r4_launcher_imports_exact_sibling_under_isolated_python(self) -> None:
        launcher = R4_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('Path(__file__).resolve(strict=True).parent', launcher)
        self.assertIn('_HERE / "submit_gate2.py"', launcher)
        self.assertIn("spec_from_file_location", launcher)
        self.assertNotIn("import submit_gate2 as controller", launcher)
        with tempfile.TemporaryDirectory(prefix="gate2-r4-isolated-cwd-") as cwd:
            completed = subprocess.run(
                [sys.executable, "-B", "-I", str(R4_LAUNCHER), "--help"],
                cwd=cwd,
                env={"PATH": os.environ.get("PATH", "")},
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage:", completed.stdout.lower())

    def test_r5_has_new_identity_portable_binding_and_no_r4_reuse(self) -> None:
        module = load_r5_controller()
        expected = (
            "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-"
            "R5-S42-20260811"
        )
        retired_attempt = "4c16eab2-c310-41e9-9d41-367cfc038acd"
        retired_job = "dlcr9fkau7hrwj0r"
        wrapper = R5_WRAPPER.read_text(encoding="utf-8")
        launcher = R5_LAUNCHER.read_text(encoding="utf-8")
        runtime = RUNTIME.read_text(encoding="utf-8")
        self.assertEqual(module.EXPERIMENT_ID, expected)
        self.assertEqual(module.SUBMISSION_TAG_PREFIX, "fastwam-gate2-nohash-r5-s42")
        self.assertEqual(module.DISPLAY_NAME_PREFIX, "fw-g2-nh-r5-s42")
        self.assertEqual(module.CONTROL_ENTRYPOINT, "submit_from_ssh970_r5.sh")
        self.assertIn(".gate2-nohash-r5-submit.lock", wrapper)
        self.assertIn(
            'exec "${CONTROL_PYTHON}" -B -I "${SCRIPT_DIR}/submit_gate2_r5.py" "$@"',
            wrapper,
        )
        expected_source = Path(
            "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/"
            "fastwam-action-n234-formal-20260811-r4"
        )
        self.assertEqual(module.APPROVED_SOURCE_ROOT, expected_source)
        self.assertIn("fastwam-nohash-source-content-binding-v3", runtime)
        for retired in (retired_attempt, retired_job):
            self.assertNotIn(retired, wrapper)
            self.assertNotIn(retired, launcher)
            self.assertNotIn(retired, runtime)

        body = module.build_request(
            expected_source,
            Path("/oss-chengjuntao/artifacts/r5-controller-test-stats.json"),
            Path(
                "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"
                "r5-controller-test-primary"
            ),
            Path(
                "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"
                "r5-controller-test-fallback"
            ),
            "11223344-5566-4778-899a-bbccddeeff00",
            trusted_runtime_bytes=b"r5-controller-test-runtime\n",
        )
        module.validate_request(body, validate_live_inputs=False)
        self.assertEqual(body["Envs"]["FASTWAM_EXPERIMENT_ID"], expected)
        self.assertIn("fastwam-gate2-nohash-r5-s42-", body["Envs"]["FASTWAM_SUBMISSION_TAG"])
        self.assertNotIn("r4", body["Envs"]["FASTWAM_OSS_OUTPUT_ROOT"].lower())

        body["Envs"]["FASTWAM_SOURCE_ROOT"] = str(expected_source.with_name("retired-r3"))
        with self.assertRaisesRegex(RuntimeError, "exact approved snapshot"):
            module.validate_request(body, validate_live_inputs=False)

    def test_r5_launcher_imports_exact_sibling_under_isolated_python(self) -> None:
        launcher = R5_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('Path(__file__).resolve(strict=True).parent', launcher)
        self.assertIn('_HERE / "submit_gate2.py"', launcher)
        self.assertIn("spec_from_file_location", launcher)
        self.assertNotIn("import submit_gate2 as controller", launcher)
        with tempfile.TemporaryDirectory(prefix="gate2-r5-isolated-cwd-") as cwd:
            completed = subprocess.run(
                [sys.executable, "-B", "-I", str(R5_LAUNCHER), "--help"],
                cwd=cwd,
                env={"PATH": os.environ.get("PATH", "")},
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage:", completed.stdout.lower())

    def test_r6_has_new_identity_source_and_no_r5_reuse(self) -> None:
        module = load_r6_controller()
        expected = (
            "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-"
            "R6-S42-20260811"
        )
        retired_attempt = "6f33975b-e52c-4625-9edd-1f8bbf5b281e"
        retired_job = "dlcenr216elrf64o"
        wrapper = R6_WRAPPER.read_text(encoding="utf-8")
        launcher = R6_LAUNCHER.read_text(encoding="utf-8")
        runtime = RUNTIME.read_text(encoding="utf-8")
        self.assertEqual(module.EXPERIMENT_ID, expected)
        self.assertEqual(module.SUBMISSION_TAG_PREFIX, "fastwam-gate2-nohash-r6-s42")
        self.assertEqual(module.DISPLAY_NAME_PREFIX, "fw-g2-nh-r6-s42")
        self.assertEqual(module.CONTROL_ENTRYPOINT, "submit_from_ssh970_r6.sh")
        self.assertIn(".gate2-nohash-r6-submit.lock", wrapper)
        self.assertIn("export PYTHONDONTWRITEBYTECODE=1", wrapper)
        self.assertIn(
            'exec "${CONTROL_PYTHON}" -B -I "${SCRIPT_DIR}/submit_gate2_r6.py" "$@"',
            wrapper,
        )
        expected_source = Path(
            "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/"
            "fastwam-action-n234-formal-20260811-r5"
        )
        self.assertEqual(module.APPROVED_SOURCE_ROOT, expected_source)
        for retired in (retired_attempt, retired_job):
            self.assertNotIn(retired, wrapper)
            self.assertNotIn(retired, launcher)
            self.assertNotIn(retired, runtime)

        body = module.build_request(
            expected_source,
            Path("/oss-chengjuntao/artifacts/r6-controller-test-stats.json"),
            Path(
                "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"
                "r6-controller-test-primary"
            ),
            Path(
                "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"
                "r6-controller-test-fallback"
            ),
            "22334455-6677-4889-8aab-ccddeeff0011",
            trusted_runtime_bytes=b"r6-controller-test-runtime\n",
        )
        module.validate_request(body, validate_live_inputs=False)
        self.assertEqual(body["Envs"]["FASTWAM_EXPERIMENT_ID"], expected)
        self.assertIn(
            "fastwam-gate2-nohash-r6-s42-",
            body["Envs"]["FASTWAM_SUBMISSION_TAG"],
        )
        self.assertNotIn("r5", body["Envs"]["FASTWAM_OSS_OUTPUT_ROOT"].lower())

        body["Envs"]["FASTWAM_SOURCE_ROOT"] = str(
            expected_source.with_name("retired-r4")
        )
        with self.assertRaisesRegex(RuntimeError, "exact approved snapshot"):
            module.validate_request(body, validate_live_inputs=False)

    def test_r7_has_fresh_identity_source_and_no_r6_reuse(self) -> None:
        module = load_r7_controller()
        expected = (
            "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-"
            "R7-S42-20260811"
        )
        retired_attempt = "48e33268-a111-4115-bb45-06862d3c97c7"
        retired_job = "dlct3jzm2aiw4xit"
        wrapper = R7_WRAPPER.read_text(encoding="utf-8")
        launcher = R7_LAUNCHER.read_text(encoding="utf-8")
        runtime = RUNTIME.read_text(encoding="utf-8")
        self.assertEqual(module.EXPERIMENT_ID, expected)
        self.assertEqual(module.SUBMISSION_TAG_PREFIX, "fastwam-gate2-nohash-r7-s42")
        self.assertEqual(module.DISPLAY_NAME_PREFIX, "fw-g2-nh-r7-s42")
        self.assertEqual(module.CONTROL_ENTRYPOINT, "submit_from_ssh970_r7.sh")
        self.assertIn(".gate2-nohash-r7-submit.lock", wrapper)
        self.assertIn("export PYTHONDONTWRITEBYTECODE=1", wrapper)
        self.assertIn(
            'exec "${CONTROL_PYTHON}" -B -I "${SCRIPT_DIR}/submit_gate2_r7.py" "$@"',
            wrapper,
        )
        expected_source = Path(
            "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/"
            "fastwam-action-n234-formal-20260811-r7"
        )
        self.assertEqual(module.APPROVED_SOURCE_ROOT, expected_source)
        for retired in (retired_attempt, retired_job):
            self.assertNotIn(retired, wrapper)
            self.assertNotIn(retired, launcher)

        body = module.build_request(
            expected_source,
            Path("/oss-chengjuntao/artifacts/r7-controller-test-stats.json"),
            Path(
                "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"
                "r7-controller-test-primary"
            ),
            Path(
                "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"
                "r7-controller-test-fallback"
            ),
            "33445566-7788-499a-8bbc-ddeeff001122",
            trusted_runtime_bytes=b"r7-controller-test-runtime\n",
        )
        module.validate_request(body, validate_live_inputs=False)
        self.assertEqual(body["Envs"]["FASTWAM_EXPERIMENT_ID"], expected)
        self.assertIn(
            "fastwam-gate2-nohash-r7-s42-",
            body["Envs"]["FASTWAM_SUBMISSION_TAG"],
        )
        self.assertNotIn("r6", body["Envs"]["FASTWAM_OSS_OUTPUT_ROOT"].lower())
        self.assertIn(expected, body["Envs"]["FASTWAM_PREPARED_BINDING_PATH"])

        body["Envs"]["FASTWAM_SOURCE_ROOT"] = str(
            expected_source.with_name("retired-r5")
        )
        with self.assertRaisesRegex(RuntimeError, "exact approved snapshot"):
            module.validate_request(body, validate_live_inputs=False)

    def test_r8_has_fresh_identity_source_and_no_r7_reuse(self) -> None:
        module = load_r8_controller()
        expected = (
            "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-"
            "R8-S42-20260811"
        )
        retired_attempt = "48e33268-a111-4115-bb45-06862d3c97c7"
        retired_job = "dlct3jzm2aiw4xit"
        wrapper = R8_WRAPPER.read_text(encoding="utf-8")
        launcher = R8_LAUNCHER.read_text(encoding="utf-8")
        runtime = RUNTIME.read_text(encoding="utf-8")
        self.assertEqual(module.EXPERIMENT_ID, expected)
        self.assertEqual(module.SUBMISSION_TAG_PREFIX, "fastwam-gate2-nohash-r8-s42")
        self.assertEqual(module.DISPLAY_NAME_PREFIX, "fw-g2-nh-r8-s42")
        self.assertEqual(module.CONTROL_ENTRYPOINT, "submit_from_ssh970_r8.sh")
        self.assertIn(f'EXPECTED_EXPERIMENT="{expected}"', runtime)
        self.assertIn(".gate2-nohash-r8-submit.lock", wrapper)
        self.assertIn("export PYTHONDONTWRITEBYTECODE=1", wrapper)
        self.assertIn(
            'exec "${CONTROL_PYTHON}" -B -I "${SCRIPT_DIR}/submit_gate2_r8.py" "$@"',
            wrapper,
        )
        expected_source = Path(
            "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/"
            "fastwam-action-n234-formal-20260811-r8"
        )
        self.assertEqual(module.APPROVED_SOURCE_ROOT, expected_source)
        for retired in (retired_attempt, retired_job):
            self.assertNotIn(retired, wrapper)
            self.assertNotIn(retired, launcher)
            self.assertNotIn(retired, runtime)

        body = module.build_request(
            expected_source,
            Path("/oss-chengjuntao/artifacts/r8-controller-test-stats.json"),
            Path(
                "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"
                "r8-controller-test-primary"
            ),
            Path(
                "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"
                "r8-controller-test-fallback"
            ),
            "44556677-8899-4aab-9ccd-eeff00112233",
            trusted_runtime_bytes=b"r8-controller-test-runtime\n",
        )
        module.validate_request(body, validate_live_inputs=False)
        self.assertEqual(body["Envs"]["FASTWAM_EXPERIMENT_ID"], expected)
        self.assertIn(
            "fastwam-gate2-nohash-r8-s42-",
            body["Envs"]["FASTWAM_SUBMISSION_TAG"],
        )
        self.assertNotIn("r7", body["Envs"]["FASTWAM_OSS_OUTPUT_ROOT"].lower())
        self.assertIn(expected, body["Envs"]["FASTWAM_PREPARED_BINDING_PATH"])

        body["Envs"]["FASTWAM_SOURCE_ROOT"] = str(
            expected_source.with_name("retired-r7")
        )
        with self.assertRaisesRegex(RuntimeError, "exact approved snapshot"):
            module.validate_request(body, validate_live_inputs=False)

    def test_r8_launcher_and_readonly_monitor_are_isolated(self) -> None:
        launcher = R8_LAUNCHER.read_text(encoding="utf-8")
        monitor = READONLY_MONITOR.read_text(encoding="utf-8")
        monitor_tree = ast.parse(monitor, filename=str(READONLY_MONITOR))
        client_methods = {
            node.func.attr
            for node in ast.walk(monitor_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "client"
        }
        self.assertIn('Path(__file__).resolve(strict=True).parent', launcher)
        self.assertIn('_HERE / "submit_gate2.py"', launcher)
        self.assertIn("spec_from_file_location", launcher)
        self.assertNotIn("import submit_gate2 as controller", launcher)
        self.assertNotIn("importlib", monitor)
        self.assertNotIn("submit_gate2", monitor)
        self.assertNotIn("create_job", monitor.lower())
        self.assertNotIn("stop_job", monitor.lower())
        self.assertEqual(
            client_methods,
            {"get_job_with_options", "get_pod_logs_with_options"},
        )
        self.assertIn("sys.flags.isolated", monitor)
        self.assertIn("sys.flags.dont_write_bytecode", monitor)
        with tempfile.TemporaryDirectory(prefix="gate2-r8-isolated-cwd-") as cwd:
            for command in (
                [sys.executable, "-B", "-I", str(R8_LAUNCHER), "--help"],
                [sys.executable, "-B", "-I", str(READONLY_MONITOR), "--help"],
            ):
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    env={
                        "PATH": os.environ.get("PATH", ""),
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage:", completed.stdout.lower())

    def test_runtime_stats_root_path_equivalence(self) -> None:
        text = RUNTIME.read_text(encoding="utf-8")
        marker = '"${FASTWAM_PYTHON}" - "${LOCAL_STATS}" "${FASTWAM_DATASET_ROOT}" <<\'PY\'\n'
        script_start = text.index(marker) + len(marker)
        script_end = text.index("\nPY\n", script_start)
        script = text[script_start:script_end]
        self.assertIn(
            "stats_source_root = Path(source_root).resolve(strict=True)",
            script,
        )
        self.assertIn("if stats_source_root != dataset_root:", script)
        root = Path(self.temporary.name) / "stats-root-equivalence"
        physical = root / "oss-dataset"
        different = root / "different-dataset"
        logical = root / "cpfs-dataset"
        physical.mkdir(parents=True)
        different.mkdir()
        logical.symlink_to(physical, target_is_directory=True)

        required = {
            "action": {},
            "state": {},
            "files": 1,
            "trajectories": 1,
            "cardinality": {},
            "normalization_fit": {},
        }
        stats = root / "stats.json"
        stats.write_text(
            json.dumps({"source_root": str(logical), **required}),
            encoding="utf-8",
        )
        accepted = subprocess.run(
            [sys.executable, "-B", "-I", "-", str(stats), str(logical)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)

        stats.write_text(
            json.dumps({"source_root": str(different), **required}),
            encoding="utf-8",
        )
        rejected = subprocess.run(
            [sys.executable, "-B", "-I", "-", str(stats), str(logical)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("source_root literal mismatch", rejected.stderr)

    def test_r6_launcher_and_readonly_monitor_are_isolated(self) -> None:
        launcher = R6_LAUNCHER.read_text(encoding="utf-8")
        monitor = READONLY_MONITOR.read_text(encoding="utf-8")
        monitor_tree = ast.parse(monitor, filename=str(READONLY_MONITOR))
        client_methods = {
            node.func.attr
            for node in ast.walk(monitor_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "client"
        }
        self.assertIn('Path(__file__).resolve(strict=True).parent', launcher)
        self.assertIn('_HERE / "submit_gate2.py"', launcher)
        self.assertIn("spec_from_file_location", launcher)
        self.assertNotIn("import submit_gate2 as controller", launcher)
        self.assertNotIn("importlib", monitor)
        self.assertNotIn("submit_gate2", monitor)
        self.assertNotIn("create_job", monitor.lower())
        self.assertNotIn("stop_job", monitor.lower())
        self.assertEqual(
            client_methods,
            {"get_job_with_options", "get_pod_logs_with_options"},
        )
        self.assertIn("sys.flags.isolated", monitor)
        self.assertIn("sys.flags.dont_write_bytecode", monitor)
        with tempfile.TemporaryDirectory(prefix="gate2-r6-isolated-cwd-") as cwd:
            for command in (
                [sys.executable, "-B", "-I", str(R6_LAUNCHER), "--help"],
                [
                    sys.executable,
                    "-B",
                    "-I",
                    str(READONLY_MONITOR),
                    "--help",
                ],
            ):
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    env={
                        "PATH": os.environ.get("PATH", ""),
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage:", completed.stdout.lower())

    def test_durable_record_is_immutable_and_exact(self) -> None:
        path = self.module.prepared_binding_path()
        self.module.durable_exclusive_write(path, self.binding)
        observed, _ = self.module.read_json(path)
        self.assertEqual(observed, self.binding)
        with self.assertRaises(FileExistsError):
            self.module.durable_exclusive_write(path, self.binding)

    def test_record_writers_reject_zero_progress(self) -> None:
        root = Path(self.temporary.name)
        with mock.patch.object(self.module.os, "write", return_value=0):
            with self.assertRaisesRegex(RuntimeError, "made no progress"):
                self.module.atomic_write(root / "atomic-zero.json", {"value": 1})

        with mock.patch.object(self.module.os, "write", return_value=0):
            with self.assertRaisesRegex(RuntimeError, "made no progress"):
                self.module.durable_exclusive_write(
                    root / "durable-zero.json", {"value": 1}
                )

    def test_stable_read_rejects_short_read_and_symlink(self) -> None:
        root = Path(self.temporary.name)
        target = root / "stable-read.json"
        target.write_bytes(b"abc")
        with mock.patch.object(self.module.os, "read", return_value=b""):
            with self.assertRaisesRegex(RuntimeError, "byte count"):
                self.module.stable_read(target)

        alias = root / "stable-read-link.json"
        alias.symlink_to(target)
        with self.assertRaises(OSError):
            self.module.stable_read(alias)

    def test_require_prepared_binding_rejects_non_integer_v3_size(self) -> None:
        binding = copy.deepcopy(self.binding)
        binding["approved_source_metadata"]["entries"].append(
            {
                "path": "payload.bin",
                "kind": "file",
                "size": 3.0,
                "content_b64": base64.b64encode(b"abc").decode("ascii"),
            }
        )
        self.module.durable_exclusive_write(
            self.module.prepared_binding_path(), binding
        )
        with (
            mock.patch.object(self.module, "validate_request", return_value=None),
            mock.patch.object(
                self.module,
                "source_snapshot_literal",
                return_value=Path(binding["approved_source_root"]),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "size or content"):
                self.module.require_prepared_binding(self.attempt)

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
        observed = observed_getjob_shape(self.body)
        self.assertTrue(self.module.exact_identity(observed, self.body))

        for nested in (
            "FASTWAM_INITIAL_CHECKPOINT",
            self.module.TRUSTED_RUNTIME_B64_ENV,
        ):
            changed = copy.deepcopy(observed)
            changed["Envs"][nested] = "changed"
            with self.subTest(field=f"Envs.{nested}"):
                self.assertFalse(self.module.exact_identity(changed, self.body))

        direct_mutations = (
            ("UserCommand", "changed"),
            ("JobType", "changed"),
            ("Priority", 99),
            ("Accessibility", "PUBLIC"),
            ("WorkspaceId", int(self.body["WorkspaceId"])),
        )
        for field, value in direct_mutations:
            changed = copy.deepcopy(observed)
            changed[field] = value
            with self.subTest(field=field):
                self.assertFalse(self.module.exact_identity(changed, self.body))

        changed = copy.deepcopy(observed)
        changed["Settings"]["Tags"]["submission_tag"] = "changed"
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

        for field, value in (("DataSourceId", "changed"), ("MountPath", "/changed"), ("Uri", "oss://unexpected")):
            changed = copy.deepcopy(observed)
            changed["DataSources"][0][field] = value
            with self.subTest(field=f"DataSources.{field}"):
                self.assertFalse(self.module.exact_identity(changed, self.body))
        changed = copy.deepcopy(observed)
        changed["DataSources"].reverse()
        self.assertFalse(self.module.exact_identity(changed, self.body))

        custom_env_mutations = []
        missing = copy.deepcopy(observed)
        missing["CustomEnvs"].pop()
        custom_env_mutations.append(missing)
        duplicate = copy.deepcopy(observed)
        duplicate["CustomEnvs"][-1] = copy.deepcopy(duplicate["CustomEnvs"][0])
        custom_env_mutations.append(duplicate)
        wrong_value = copy.deepcopy(observed)
        wrong_value["CustomEnvs"][0]["Value"] = "changed"
        custom_env_mutations.append(wrong_value)
        private = copy.deepcopy(observed)
        private["CustomEnvs"][0]["Visible"] = "private"
        custom_env_mutations.append(private)
        extra_key = copy.deepcopy(observed)
        extra_key["CustomEnvs"][0]["ServerField"] = "unexpected"
        custom_env_mutations.append(extra_key)
        for index, changed in enumerate(custom_env_mutations):
            with self.subTest(custom_env_mutation=index):
                self.assertFalse(self.module.exact_identity(changed, self.body))

        # The service may omit only these two fields.  If returned, the exact
        # frozen value is still required; no response-side default is inferred.
        for field in ("JobMaxRunningTimeMinutes", "SuccessPolicy"):
            returned = copy.deepcopy(observed)
            returned[field] = self.body[field]
            self.assertTrue(self.module.exact_identity(returned, self.body))
            returned[field] = "changed"
            self.assertFalse(self.module.exact_identity(returned, self.body))

        future_request = copy.deepcopy(self.body)
        future_request["FutureFrozenField"] = {"Required": True}
        self.assertFalse(self.module.exact_identity(observed, future_request))
        future_observed = copy.deepcopy(observed)
        future_observed["FutureFrozenField"] = {"Required": True, "ServerAdded": 1}
        self.assertTrue(self.module.exact_identity(future_observed, future_request))
        future_observed["FutureFrozenField"]["Required"] = False
        self.assertFalse(self.module.exact_identity(future_observed, future_request))

        strict_scalar_request = copy.deepcopy(self.body)
        strict_scalar_request["FutureFrozenField"] = {"Required": 1}
        strict_scalar_observed = copy.deepcopy(observed)
        strict_scalar_observed["FutureFrozenField"] = {"Required": True}
        self.assertFalse(
            self.module.exact_identity(strict_scalar_observed, strict_scalar_request)
        )

    def test_validate_request_freezes_service_omitted_fields(self) -> None:
        for field, changed_value in (
            ("JobMaxRunningTimeMinutes", 721),
            ("SuccessPolicy", "ChiefWorker"),
            ("CustomEnvs", [{"Key": "unexpected"}]),
        ):
            changed = copy.deepcopy(self.body)
            changed[field] = changed_value
            with self.subTest(field=field, mutation="changed"):
                with self.assertRaisesRegex(RuntimeError, "job-level execution contract"):
                    self.module.validate_request(changed, validate_live_inputs=False)
            changed = copy.deepcopy(self.body)
            changed.pop(field)
            with self.subTest(field=field, mutation="removed"):
                with self.assertRaisesRegex(RuntimeError, "job-level execution contract"):
                    self.module.validate_request(changed, validate_live_inputs=False)

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
        self.assertEqual(len(self.body["Envs"]), 20)
        command = self.body["UserCommand"]
        self.assertIn(f"exec {self.module.PINNED_PYTHON} -B -I -S", command)
        self.assertIn(f"readlink -f -- {self.module.PINNED_PYTHON}", command)
        self.assertIn(str(self.module.PINNED_PYTHON_TARGET), command)
        self.assertNotIn("/usr/bin/python3", command)
        self.assertEqual(command.count("'"), 2)
        self.assertTrue(command.startswith("unset BASH_ENV ENV PYTHONHOME"))
        self.assertIn("LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT", command)
        self.assertIn("clean={key:os.environ[key] for key in allowed", command)
        self.assertNotIn("clean=dict(os.environ)", command)
        self.assertIn("FASTWAM_PYTHON_TARGET", self.module.BOOTSTRAP_ALLOWED_ENV)
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
        self.assertIn('"${FASTWAM_PYTHON}" -B -I -S - "${TRUSTED_RUNTIME_PATH}"', text)
        self.assertIn('FASTWAM_PYTHON_TARGET', text)
        self.assertIn('RESOLVED_PYTHON="$(readlink -f -- "${FASTWAM_PYTHON}")"', text)
        self.assertNotIn('/usr/bin/python3', text)

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
        self.assertEqual(first["schema"], "fastwam-nohash-source-content-binding-v3")
        self.assertEqual(first_entry["size"], second_entry["size"])
        self.assertNotEqual(first_entry["content_b64"], second_entry["content_b64"])
        self.assertEqual(
            set(first_entry), {"path", "kind", "size", "content_b64"}
        )
        for forbidden in ("mode", "mtime_ns", "device", "inode", "ctime_ns"):
            self.assertNotIn(forbidden, first_entry)
        self.assertEqual(
            base64.b64decode(first_entry["content_b64"], validate=True), b"first"
        )

    def test_source_binding_is_portable_across_mode_mtime_and_inode_changes(self) -> None:
        source = Path(self.temporary.name) / "portable-source-binding"
        source.mkdir()
        target = source / "payload.bin"
        target.write_bytes(b"portable")
        os.chmod(target, 0o600)
        first = self.module.source_snapshot_metadata(source)

        replacement = source / "replacement.bin"
        replacement.write_bytes(b"portable")
        os.chmod(replacement, 0o644)
        os.utime(replacement, ns=(1_700_000_000_000_000_000,) * 2)
        os.replace(replacement, target)
        second = self.module.source_snapshot_metadata(source)

        self.assertEqual(first, second)
        for entry in second["entries"]:
            if entry["kind"] == "directory":
                self.assertEqual(set(entry), {"path", "kind"})
            else:
                self.assertEqual(
                    set(entry), {"path", "kind", "size", "content_b64"}
                )

    def test_source_binding_rejects_noncanonical_or_malformed_entries(self) -> None:
        source = Path(self.temporary.name) / "source-binding-validation"
        source.mkdir()
        valid = {
            "schema": "fastwam-nohash-source-content-binding-v3",
            "approved_source_root": str(source),
            "entries": [
                {"path": ".", "kind": "directory"},
                {
                    "path": "payload.bin",
                    "kind": "file",
                    "size": 3,
                    "content_b64": base64.b64encode(b"abc").decode("ascii"),
                },
            ],
        }
        self.module.validate_source_snapshot_metadata(valid, source)

        mutations = []
        wrong_size = copy.deepcopy(valid)
        wrong_size["entries"][1]["size"] = 4
        mutations.append(wrong_size)
        duplicate = copy.deepcopy(valid)
        duplicate["entries"].append(copy.deepcopy(duplicate["entries"][1]))
        mutations.append(duplicate)
        traversal = copy.deepcopy(valid)
        traversal["entries"][1]["path"] = "../payload.bin"
        mutations.append(traversal)
        metadata_field = copy.deepcopy(valid)
        metadata_field["entries"][1]["mtime_ns"] = 1
        mutations.append(metadata_field)
        bool_size = copy.deepcopy(valid)
        bool_size["entries"][1]["size"] = True
        mutations.append(bool_size)
        float_size = copy.deepcopy(valid)
        float_size["entries"][1]["size"] = 3.0
        mutations.append(float_size)
        invalid_base64 = copy.deepcopy(valid)
        invalid_base64["entries"][1]["content_b64"] = "not/base64!"
        mutations.append(invalid_base64)
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                with self.assertRaises(RuntimeError):
                    self.module.validate_source_snapshot_metadata(mutation, source)

    def test_source_binding_rejects_ancestor_directory_replacement(self) -> None:
        source = Path(self.temporary.name) / "source-directory-race"
        child = source / "nested"
        moved = source / "nested-moved"
        child.mkdir(parents=True)
        (child / "payload.bin").write_bytes(b"payload")

        real_open = os.open
        replacement_triggered = False

        def replacing_open(path, flags, mode=0o777, *, dir_fd=None):
            nonlocal replacement_triggered
            if dir_fd is None:
                descriptor = real_open(path, flags, mode)
            else:
                descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            if path == "nested" and dir_fd is not None and not replacement_triggered:
                replacement_triggered = True
                child.rename(moved)
                child.symlink_to(moved.name, target_is_directory=True)
            return descriptor

        with mock.patch.object(self.module.os, "open", side_effect=replacing_open):
            with self.assertRaises(RuntimeError):
                self.module.source_snapshot_metadata(source)
        self.assertTrue(replacement_triggered)

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
        source_validation = text[text.index("expected = binding.get") : text.index("compare_trees()")]
        self.assertIn("fastwam-nohash-source-content-binding-v3", source_validation)
        self.assertIn("path_after.st_dev != after.st_dev", source_validation)
        self.assertIn("path_after.st_ino != after.st_ino", source_validation)
        self.assertIn("binding_fd = os.open(binding_path", text)
        self.assertIn("binding_path_after.st_dev != after.st_dev", text)
        self.assertIn("open_absolute_directory_nofollow", source_validation)
        self.assertIn("dir_fd=directory_fd", source_validation)
        self.assertNotIn(".rglob(", source_validation)
        self.assertNotIn('"mode":', source_validation)
        self.assertNotIn('"mtime_ns":', source_validation)
        self.assertIn("executing runtime differs from request-carried trusted bytes", text)
        self.assertIn("FASTWAM_PREPARED_BINDING_PATH", text)

    def test_runtime_prepared_binding_reader_rejects_symlink(self) -> None:
        text = RUNTIME.read_text(encoding="utf-8")
        function_start = text.index("validate_prepared_source_binding()")
        heredoc_marker = "<<'PY'\n"
        script_start = text.index(heredoc_marker, function_start) + len(heredoc_marker)
        script_end = text.index("\nPY\n}", script_start)
        script = text[script_start:script_end]

        root = Path(self.temporary.name)
        source = root / "runtime-source"
        source.mkdir()
        target = root / "prepared-binding.json"
        alias = root / "prepared-binding-link.json"
        experiment = "runtime-binding-test"
        tag = "runtime-binding-tag"
        entrypoint = str(source / "runtime.sh")
        trusted_runtime = b"runtime-binding-test-bytes\n"
        encoded_runtime = base64.b64encode(trusted_runtime).decode("ascii")
        target.write_text(
            json.dumps(
                {
                    "schema": "fastwam-dlc-prepared-binding-v1",
                    "experiment_id": experiment,
                    "request": {
                        "Envs": {
                            "FASTWAM_EXPERIMENT_ID": experiment,
                            "FASTWAM_SUBMISSION_TAG": tag,
                            "FASTWAM_SOURCE_ROOT": str(source),
                            "FASTWAM_GATE2_ENTRYPOINT": entrypoint,
                            "FASTWAM_PREPARED_BINDING_PATH": str(alias),
                            "FASTWAM_GATE2_TRUSTED_RUNTIME_B64": encoded_runtime,
                            "FASTWAM_GATE2_TRUSTED_RUNTIME_BYTES": str(
                                len(trusted_runtime)
                            ),
                        }
                    },
                    "approved_source_metadata": {
                        "schema": "fastwam-nohash-source-content-binding-v3",
                        "approved_source_root": str(source),
                        "entries": [{"path": ".", "kind": "directory"}],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        alias.symlink_to(target)
        completed = subprocess.run(
            [
                sys.executable,
                "-",
                str(alias),
                str(source),
                experiment,
                tag,
                entrypoint,
            ],
            input=script,
            text=True,
            capture_output=True,
            check=False,
            env={
                "FASTWAM_GATE2_TRUSTED_RUNTIME_B64": encoded_runtime,
                "FASTWAM_GATE2_TRUSTED_RUNTIME_BYTES": str(len(trusted_runtime)),
            },
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "prepared binding is missing, linked, or unreadable",
            completed.stderr,
        )

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
        module.assert_pinned_python = lambda: module.PINNED_PYTHON_TARGET

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
        observed = observed_getjob_shape(self.body)
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
        module.source_snapshot_metadata = lambda source: copy.deepcopy(
            self.binding["approved_source_metadata"]
        )
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
