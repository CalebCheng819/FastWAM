from __future__ import annotations

import datetime as dt
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

try:
    import hydra  # noqa: F401 - availability gate for real composition tests.
    from omegaconf import OmegaConf
except ModuleNotFoundError:
    HYDRA_AVAILABLE = False
else:
    HYDRA_AVAILABLE = True

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from fastwam.validation.g0_contract import (
    SUITES,
    canonical_json_bytes,
    canonical_json_sha256,
    validate_runtime_start,
)
from scripts import run_mf_wam_g0_manager as manager
from scripts import run_mf_wam_g0_traced as traced_runner
from scripts import seal_mf_wam_g0_terminal as sealer
from tests import test_g0_contract as contract_tests
from tests import test_mf_wam_g0_terminal_sealer as sealer_tests


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tree_sha(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["path"].encode("utf-8")):
        digest.update(f"{row['sha256']}  {row['path']}\n".encode("utf-8"))
    return digest.hexdigest()


class G0ManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture = contract_tests.G0ContractTest(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        self.fixture = fixture
        base = fixture.artifact_root.parent
        self.control_root = base / "control"
        self.control_root.mkdir()
        self.raw_log_root = base / "raw-logs"
        self.working_directory = base / "instrumentation"
        (self.working_directory / "scripts").mkdir(parents=True)
        self.runner_path = self.working_directory / "scripts/run_mf_wam_g0_traced.py"
        self.runner_path.write_text("# synthetic runner\n", encoding="utf-8")
        self.instrumentation_path = (
            self.working_directory / "scripts/mf_wam_g0_instrumentation.py"
        )
        self.instrumentation_path.write_bytes(
            (REPOSITORY_ROOT / "scripts/mf_wam_g0_instrumentation.py").read_bytes()
        )
        self.official_root = REPOSITORY_ROOT

        launch_files = {
            "checkpoint": ("checkpoint.pt", b"synthetic checkpoint\n"),
            "dataset_stats": ("dataset-stats.json", b'{"synthetic":true}\n'),
        }
        for role, (name, raw) in launch_files.items():
            path = self.control_root / name
            path.write_bytes(raw)
            setattr(self, f"{role}_path", path)
            fixture.prereg["artifacts"][role] = {
                "sha256": _sha(raw),
                "size_bytes": len(raw),
            }
        if HYDRA_AVAILABLE:
            overrides = [
                "task=libero_uncond_2cam224_1e-4",
                f"ckpt={self.checkpoint_path}",
                "gpu_id=0",
                "seed=42",
                f"output_dir={fixture.artifact_root}",
                "EVALUATION.task_suite_name=libero_spatial",
                "EVALUATION.task_id=0",
                f"EVALUATION.output_dir={fixture.artifact_root}",
                f"EVALUATION.dataset_stats_path={self.dataset_stats_path}",
                "EVALUATION.num_trials=50",
                "EVALUATION.env_num=1",
                "EVALUATION.num_steps_wait=30",
                "EVALUATION.replan_steps=10",
                "EVALUATION.binarize_gripper=true",
                "EVALUATION.use_action_ensembler=false",
                "EVALUATION.visualize_future_video=false",
                "EVALUATION.action_horizon=32",
            ]
            composed = traced_runner._compose_official_config(
                self.official_root,
                overrides,
            )
            resolved_raw = canonical_json_bytes(
                OmegaConf.to_container(
                    composed,
                    resolve=True,
                    throw_on_missing=True,
                )
            )
        else:
            # Non-Hydra unit tests patch only the independent semantic gate.
            # The real manager success test is explicitly skipped in this mode.
            resolved_raw = b'{"hydra_dependency_unavailable_in_unit_test":true}'
        self.resolved_config_path = self.control_root / "resolved-config.yaml"
        self.resolved_config_path.write_bytes(resolved_raw)
        fixture.prereg["artifacts"]["resolved_config"] = {
            "sha256": _sha(resolved_raw),
            "size_bytes": len(resolved_raw),
        }
        fixture.prereg["launch"]["working_directory"] = str(self.working_directory)
        self.start = validate_runtime_start(
            fixture._runtime_start(),
            preregistration=fixture.prereg,
        )
        self.prereg_path = self._write_json("preregistration.json", fixture.prereg)
        self.start_path = self._write_json("runtime-start.json", self.start)
        self.task_map_path = self._write_json("task-map.json", fixture.task_map)
        self.schedule_path = self._write_json("seed-schedule.json", fixture.schedule)
        self.config = manager.ManagerConfig(
            run_id=fixture.prereg["run_id"],
            artifact_root=fixture.artifact_root,
            raw_log_root=self.raw_log_root,
            working_directory=self.working_directory,
            official_root=self.official_root,
            official_commit=fixture.prereg["source"]["fastwam"]["commit"],
            instrumentation_commit=fixture.prereg["source"]["instrumentation"]["commit"],
            preregistration_path=self.prereg_path,
            preregistration_sha256=_sha(self.prereg_path.read_bytes()),
            runtime_start_path=self.start_path,
            runtime_start_sha256=_sha(self.start_path.read_bytes()),
            task_map_path=self.task_map_path,
            seed_schedule_path=self.schedule_path,
            seed_schedule_sha256=_sha(self.schedule_path.read_bytes()),
            resolved_config_path=self.resolved_config_path,
            resolved_config_sha256=_sha(self.resolved_config_path.read_bytes()),
            checkpoint_path=self.checkpoint_path,
            dataset_stats_path=self.dataset_stats_path,
            gpu_ids=(0, 1),
            python_executable="/usr/bin/python3",
            runner_path=self.runner_path,
            poll_interval_seconds=0.0,
        )
        self.upstream_digests = {
            "preregistration_file_sha256": self.config.preregistration_sha256,
            "preregistration_canonical_sha256": canonical_json_sha256(fixture.prereg),
            "runtime_start_file_sha256": self.config.runtime_start_sha256,
            "runtime_start_canonical_sha256": canonical_json_sha256(self.start),
            "seed_schedule_file_sha256": self.config.seed_schedule_sha256,
            "seed_schedule_canonical_sha256": canonical_json_sha256(fixture.schedule),
            "resolved_config_sha256": self.config.resolved_config_sha256,
        }
        self.trace_builder = sealer_tests.G0TerminalSealerTest(methodName="runTest")
        self.trace_builder.fixture = fixture
        self.trace_builder.upstream_digests = self.upstream_digests

    def _write_json(self, name: str, payload: object) -> Path:
        path = self.control_root / name
        path.write_bytes(canonical_json_bytes(payload))
        return path

    def _write_success_artifacts(self, suite: str, task_id: int, gpu_id: int) -> None:
        process_id = f"{suite}/task{task_id:02d}"
        process = next(
            item
            for item in self.fixture.schedule["task_processes"]
            if item["process_id"] == process_id
        )
        success_trials = [0]
        task_description = f"synthetic task {process_id}"
        result_raw = canonical_json_bytes(
            {
                "task_suite": suite,
                "task_id": task_id,
                "task_description": task_description,
                "successes": 1,
                "total_episodes": 50,
                "gpu_id": gpu_id,
                "success_episodes": success_trials,
                "failure_episodes": list(range(1, 50)),
                "start_time": "2026-08-03T12:00:00+00:00",
                "duration": 60.0,
            }
        )
        result_path = self.config.artifact_root / f"results/{suite}/task{task_id:02d}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_bytes(result_raw)

        trace_rows = []
        official_source = self._source_identity(official=True)
        instrumentation_source = self._source_identity(official=False)
        for trial_idx in range(50):
            relative = f"traces/{suite}/task{task_id:02d}/trial{trial_idx:03d}.json"
            raw = canonical_json_bytes(
                self.trace_builder._trace_payload(
                    process=process,
                    trial_idx=trial_idx,
                    success=trial_idx in success_trials,
                    official_source=official_source,
                    instrumentation_source=instrumentation_source,
                )
            )
            path = self.config.artifact_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            trace_rows.append(
                {
                    "trial_idx": trial_idx,
                    "path": relative,
                    "sha256": _sha(raw),
                    "size_bytes": len(raw),
                }
            )
        receipt = {
            "schema_version": 1,
            "kind": "mf_wam_g0_task_trace_receipt",
            "run_id": self.config.run_id,
            "process_id": process_id,
            "task_suite": suite,
            "task_id": task_id,
            "execution_scope": "one-process-per-task",
            "world_size": 1,
            "global_rank": 0,
            "local_rank": 0,
            "bindings": {
                "preregistration_canonical_sha256": canonical_json_sha256(
                    self.fixture.prereg
                ),
                "runtime_start_canonical_sha256": canonical_json_sha256(self.start),
                "seed_schedule_canonical_sha256": canonical_json_sha256(
                    self.fixture.schedule
                ),
                "resolved_config_sha256": self.config.resolved_config_sha256,
                "image_digest": self.fixture.prereg["image"]["digest"],
                "fastwam_commit": self.config.official_commit,
                "instrumentation_commit": self.config.instrumentation_commit,
            },
            "seeds": {
                field: copy.deepcopy(process[field])
                for field in (
                    "global_seed", "environment_seed", "environment_seed_scope",
                    "policy_seed", "policy_seed_scope", "python_hash_seed",
                    "trial_order", "initial_state_index_rule",
                )
            },
            "official_result": {
                "path": f"results/{suite}/task{task_id:02d}.json",
                "sha256": _sha(result_raw),
                "size_bytes": len(result_raw),
            },
            "episode_count": 50,
            "traces": trace_rows,
            "tree_sha256": _tree_sha(trace_rows),
        }
        receipt_path = (
            self.config.artifact_root
            / f"trace_receipts/{suite}/task{task_id:02d}.json"
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_bytes(canonical_json_bytes(receipt))

        raw_path = (
            self.config.artifact_root / f"{suite}/gpu{gpu_id}_task{task_id}_results.json"
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(result_raw)
        video = self.config.artifact_root / f"{suite}/videos/task{task_id:02d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"synthetic video\n")

    def _source_identity(self, *, official: bool) -> dict:
        role = (
            "official_policy_and_evaluator_source"
            if official
            else "external_observer_and_launcher_source"
        )
        root = self.config.official_root if official else self.config.working_directory
        commit = self.config.official_commit if official else self.config.instrumentation_commit
        return {
            "status": "PASS",
            "role": role,
            "root": str(root),
            "commit": commit,
            "tree": _sha(f"tree:{role}".encode("utf-8"))[:40],
            "clean": True,
            "critical_files": [],
            "critical_file_inventory_sha256": _sha(
                f"files:{role}".encode("utf-8")
            ),
        }

    def _worker_terminal(
        self,
        suite: str,
        task_id: int,
        gpu_id: int,
        environment_sha256: str,
    ) -> dict:
        result_path = self.config.artifact_root / f"results/{suite}/task{task_id:02d}.json"
        result_raw = result_path.read_bytes()
        raw_relative = f"{suite}/gpu{gpu_id}_task{task_id}_results.json"
        return {
            "status": "PASS",
            "kind": "mf_wam_g0_traced_worker_terminal",
            "run_id": self.config.run_id,
            "process_receipt": str(
                self.config.artifact_root
                / f"trace_receipts/{suite}/task{task_id:02d}.json"
            ),
            "official_commit": self.config.official_commit,
            "official_result_type": "dict",
            "official_result_receipt": {
                "path": f"results/{suite}/task{task_id:02d}.json",
                "sha256": _sha(result_raw),
                "size_bytes": len(result_raw),
                "source_path": raw_relative,
                "source_sha256": _sha(result_raw),
                "source_size_bytes": len(result_raw),
                "success_episodes": [0],
                "failure_episodes": list(range(1, 50)),
            },
            "terminal_source_identities": {
                "status": "PASS",
                "official": {"commit": self.config.official_commit},
                "instrumentation": {"commit": self.config.instrumentation_commit},
            },
            "external_prelaunch_commit_tree_gate_required": True,
            "environment_sha256": environment_sha256,
        }

    @staticmethod
    def _overrides(argv: list[str]) -> dict[str, str]:
        return dict(argument.split("=", 1) for argument in argv[2:])

    @unittest.skipUnless(HYDRA_AVAILABLE, "hydra-core is required for manager semantics")
    def test_manager_rejects_static_drift_and_unresolved_locked_config(self) -> None:
        resolved = json.loads(self.resolved_config_path.read_text(encoding="utf-8"))
        drifted = copy.deepcopy(resolved)
        drifted["mixed_precision"] = "fp16"
        with self.assertRaisesRegex(manager.ManagerError, "projection differs"):
            manager._validate_manager_locked_resolved_config(
                self.config,
                resolved_raw=canonical_json_bytes(drifted),
                seed=42,
            )

        interpolated = copy.deepcopy(resolved)
        interpolated["output_dir"] = "${oc.env:HOME}"
        with self.assertRaisesRegex(manager.ManagerError, "contains interpolation"):
            manager._validate_manager_locked_resolved_config(
                self.config,
                resolved_raw=canonical_json_bytes(interpolated),
                seed=42,
            )

        with self.assertRaisesRegex(manager.ManagerError, "lacks EVALUATION"):
            manager._validate_manager_locked_resolved_config(
                self.config,
                resolved_raw=b'{"synthetic":true}',
                seed=42,
            )

    @unittest.skipUnless(HYDRA_AVAILABLE, "hydra-core is required for full manager success")
    def test_success_uses_one_process_per_gpu_and_publishes_exact_manifest(self) -> None:
        launches: list[dict] = []
        active: set[int] = set()
        maximum_active = 0

        class ImmediateProcess:
            def __init__(self, gpu_id: int) -> None:
                self.gpu_id = gpu_id
                self.finished = False

            def poll(self):
                if not self.finished:
                    self.finished = True
                    active.remove(self.gpu_id)
                return 0

            def terminate(self):  # pragma: no cover - must never be called.
                raise AssertionError("manager must not terminate workers")

            def kill(self):  # pragma: no cover - must never be called.
                raise AssertionError("manager must not kill workers")

        def fake_popen(argv, **kwargs):
            nonlocal maximum_active
            overrides = self._overrides(argv)
            gpu_id = int(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
            self.assertNotIn(gpu_id, active)
            self.assertEqual(overrides["gpu_id"], str(gpu_id))
            self.assertEqual(kwargs["shell"], False)
            self.assertEqual(
                kwargs["pass_fds"], (manager.INSTRUMENTATION_MEMFD_FD,)
            )
            self.assertNotIn("MF_WAM_G0_TRACE_ROOT", kwargs["env"])
            self.assertNotIn("UNSEALED_POISON", kwargs["env"])
            self.assertNotIn("PYTHONPATH", kwargs["env"])
            self.assertNotIn("LD_PRELOAD", kwargs["env"])
            self.assertEqual(set(kwargs["env"]), traced_runner.FORMAL_ENVIRONMENT_KEYS)
            self.assertEqual(kwargs["env"]["DIFFSYNTH_SKIP_DOWNLOAD"], "true")
            self.assertEqual(
                kwargs["env"]["DIFFSYNTH_MODEL_BASE_PATH"],
                self.fixture.prereg["runtime_environment"]["DIFFSYNTH_MODEL_BASE_PATH"],
            )
            self.assertNotIn("EVALUATION.device", overrides)
            active.add(gpu_id)
            maximum_active = max(maximum_active, len(active))
            suite = overrides["EVALUATION.task_suite_name"]
            task_id = int(overrides["EVALUATION.task_id"])
            self._write_success_artifacts(suite, task_id, gpu_id)
            kwargs["stdout"].write(b"synthetic worker log\n")
            kwargs["stdout"].write(
                canonical_json_bytes(
                    self._worker_terminal(
                        suite,
                        task_id,
                        gpu_id,
                        manager._canonical_sha(kwargs["env"]),
                    )
                )
                + b"\n"
            )
            launches.append({"argv": list(argv), "gpu_id": gpu_id, **kwargs})
            return ImmediateProcess(gpu_id)

        with (
            mock.patch.dict(
                os.environ,
                {
                    "UNSEALED_POISON": "forbidden",
                    "PYTHONPATH": "/tmp/forbidden-pythonpath",
                    "LD_PRELOAD": "/tmp/forbidden-preload.so",
                },
                clear=False,
            ),
            mock.patch.object(
                manager,
                "_verify_prelaunch_sources",
                return_value=manager._canonical_python_executable(
                    self.config.python_executable
                ),
            ),
        ):
            result = manager.run_manager(
                self.config,
                popen_factory=fake_popen,
                now=lambda: dt.datetime(2026, 8, 3, 12, 0, tzinfo=dt.timezone.utc),
                sleep=lambda _seconds: None,
            )

        self.assertEqual(result["status"], "SUCCEEDED")
        self.assertEqual(len(launches), 40)
        self.assertEqual(len({tuple(item["argv"]) for item in launches}), 40)
        self.assertLessEqual(maximum_active, 2)
        self.assertEqual(active, set())
        self.assertTrue((self.raw_log_root / "manager_terminal.json").is_file())
        self.assertFalse((self.raw_log_root / "manager_failure.json").exists())
        terminal_path = self.raw_log_root / "manager_terminal.json"
        terminal_raw = terminal_path.read_bytes()
        terminal = json.loads(terminal_raw)
        self.assertEqual(set(terminal), sealer.MANAGER_TOP_KEYS)
        self.assertTrue(
            all(set(row) == sealer.MANAGER_PROCESS_KEYS for row in terminal["task_processes"])
        )
        self.assertEqual(terminal["canonical_input_file_count"], 2080)
        self.assertEqual(len(terminal["task_processes"]), 40)
        self.assertEqual({row["state"] for row in terminal["task_processes"]}, {"SUCCEEDED"})
        self.assertEqual(len(list(self.config.artifact_root.rglob("*.json"))), 2080)
        for suite in SUITES:
            self.assertFalse((self.config.artifact_root / suite).exists())
            self.assertTrue((self.raw_log_root / f"official/{suite}/videos").is_dir())
        upstream_file_sha256 = {
            "preregistration": self.config.preregistration_sha256,
            "runtime_start": self.config.runtime_start_sha256,
            "seed_schedule": self.config.seed_schedule_sha256,
            "resolved_config": self.config.resolved_config_sha256,
        }
        validated = sealer._validate_manager_manifest(
            terminal,
            trusted_file_sha256=_sha(terminal_raw),
            readback={"sha256": _sha(terminal_raw), "size_bytes": len(terminal_raw)},
            manifest_path=terminal_path,
            run_id=self.config.run_id,
            artifact_root=self.config.artifact_root,
            expected_upstream_bindings=terminal["upstream_bindings"],
            gpu_count=2,
            runtime_started_at=dt.datetime.fromisoformat(self.start["observed_at"]),
            scheduled_by_id={
                item["process_id"]: item
                for item in self.fixture.schedule["task_processes"]
            },
            preregistration=self.fixture.prereg,
            upstream_paths={
                "preregistration": self.prereg_path,
                "runtime_start": self.start_path,
                "seed_schedule": self.schedule_path,
                "resolved_config": self.resolved_config_path,
            },
            upstream_file_sha256=upstream_file_sha256,
        )
        self.assertEqual(len(validated["task_processes"]), 40)
        terminal_output = self.control_root.parent / "manager-sealed-terminal.json"
        sealed = sealer.seal_terminal_bundle(
            artifact_root=self.config.artifact_root,
            preregistration_path=self.prereg_path,
            runtime_start_path=self.start_path,
            seed_schedule_path=self.schedule_path,
            resolved_config_path=self.resolved_config_path,
            task_map_path=self.task_map_path,
            manager_manifest_path=terminal_path,
            trusted_manager_manifest_sha256=_sha(terminal_raw),
            terminal_output=terminal_output,
        )
        self.assertEqual(sealed["status"], "STRUCTURAL_PASS_ONLY")
        self.assertEqual(sealed["artifact_count"], 2084)
        self.assertEqual(sealed["task_process_count"], 40)
        self.assertEqual(sealed["episode_count"], 2000)
        self.assertTrue(terminal_output.is_file())
        self.assertTrue((self.config.artifact_root / "completion.json").is_file())

    def test_first_failure_stops_new_launches_and_waits_without_killing(self) -> None:
        launched: list[types.SimpleNamespace] = []

        class FailingProcess:
            def __init__(self, polls: list[int | None]) -> None:
                self.polls = list(polls)
                self.terminate_calls = 0
                self.kill_calls = 0

            def poll(self):
                return self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]

            def terminate(self):
                self.terminate_calls += 1

            def kill(self):
                self.kill_calls += 1

        def fake_popen(argv, **kwargs):
            index = len(launched)
            # The first worker exits zero but lacks the required traced terminal
            # JSON.  It must still fail the run before artifact acceptance.
            process = FailingProcess([0] if index == 0 else [None, 1])
            kwargs["stdout"].write(b"worker FAIL\n")
            launched.append(types.SimpleNamespace(argv=list(argv), process=process))
            return process

        with (
            mock.patch.object(
                manager,
                "_verify_prelaunch_sources",
                return_value=manager._canonical_python_executable(
                    self.config.python_executable
                ),
            ),
            mock.patch.object(
                manager,
                "_validate_manager_locked_resolved_config",
                return_value=None,
            ),
        ):
            result = manager.run_manager(
                self.config,
                popen_factory=fake_popen,
                sleep=lambda _seconds: None,
            )

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(len(launched), 2)
        self.assertTrue(all(item.process.terminate_calls == 0 for item in launched))
        self.assertTrue(all(item.process.kill_calls == 0 for item in launched))
        self.assertFalse((self.raw_log_root / "manager_terminal.json").exists())
        self.assertTrue((self.raw_log_root / "manager_failure.json").is_file())
        statuses = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (self.raw_log_root / "status").glob("*/*.json")
        ]
        self.assertEqual(len(statuses), 40)
        self.assertEqual(sum(item["state"] == "FAILED" for item in statuses), 2)
        self.assertEqual(sum(item["state"] == "NOT_LAUNCHED" for item in statuses), 38)
        first = next(
            item for item in statuses
            if item["state"] == "FAILED" and item["exit_code"] == 0
        )
        self.assertIn("invalid artifacts", first["failure_reason"])
        self.assertIn("terminal line", first["failure_reason"])

    def _assert_swap_restore_rejected(self, target: Path) -> None:
        replacement = target.with_name(target.name + ".malicious")
        original = target.with_name(target.name + ".original")
        replacement.write_bytes(b"raise RuntimeError('malicious swap executed')\n")
        launched: list[types.SimpleNamespace] = []

        class DrainableProcess:
            def __init__(self) -> None:
                self.wait_calls = 0

            def poll(self):
                return None

            def wait(self):
                self.wait_calls += 1
                return 1

        def attacking_popen(_argv, **kwargs):
            os.replace(target, original)
            os.replace(replacement, target)
            os.replace(target, replacement)
            os.replace(original, target)
            process = DrainableProcess()
            launched.append(
                types.SimpleNamespace(process=process, log_handle=kwargs["stdout"])
            )
            return process

        with (
            mock.patch.object(
                manager,
                "_verify_prelaunch_sources",
                return_value=manager._canonical_python_executable(
                    self.config.python_executable
                ),
            ),
            mock.patch.object(
                manager,
                "_validate_manager_locked_resolved_config",
                return_value=None,
            ),
        ):
            with self.assertRaisesRegex(manager.ManagerError, "source mutation"):
                manager.run_manager(
                    self.config,
                    popen_factory=attacking_popen,
                    sleep=lambda _seconds: None,
                )
        self.assertEqual(len(launched), 1)
        self.assertEqual(launched[0].process.wait_calls, 1)
        self.assertTrue(launched[0].log_handle.closed)
        self.assertFalse((self.raw_log_root / "manager_terminal.json").exists())
        self.assertEqual(target.read_bytes(), (
            b"# synthetic runner\n"
            if target == self.runner_path
            else (REPOSITORY_ROOT / "scripts/mf_wam_g0_instrumentation.py").read_bytes()
        ))

    def test_runner_rename_swap_restore_is_rejected(self) -> None:
        self._assert_swap_restore_rejected(self.runner_path)

    def test_instrumentation_replace_swap_restore_is_rejected(self) -> None:
        self._assert_swap_restore_rejected(self.instrumentation_path)

    def test_traced_loader_accepts_only_exact_sealed_commit_blob_and_closes_fd(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "instrumentation-checkout"
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            source = (
                b"class InstrumentationError(RuntimeError):\n    pass\n"
                b"SEALED_SENTINEL = 'exact-commit-blob'\n"
            )
            (scripts / "mf_wam_g0_instrumentation.py").write_bytes(source)
            runner = scripts / "run_mf_wam_g0_traced.py"
            runner.write_text("# synthetic traced runner\n", encoding="utf-8")
            for command in (
                ["/usr/bin/git", "init", "-q"],
                ["/usr/bin/git", "config", "user.email", "g0@example.invalid"],
                ["/usr/bin/git", "config", "user.name", "G0 Test"],
                ["/usr/bin/git", "add", "scripts"],
                ["/usr/bin/git", "commit", "-q", "-m", "sealed fixture"],
            ):
                subprocess.run(command, cwd=root, check=True)
            commit = subprocess.run(
                ["/usr/bin/git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            descriptor = manager._create_sealed_instrumentation_memfd(source)
            with self.assertRaises(PermissionError):
                os.write(descriptor, b"forbidden")
            with (
                mock.patch.object(traced_runner, "__file__", str(runner)),
                mock.patch.dict(
                    os.environ,
                    {"MF_WAM_INSTRUMENTATION_COMMIT": commit},
                    clear=False,
                ),
            ):
                module = traced_runner._load_instrumentation_api()
            self.addCleanup(sys.modules.pop, "mf_wam_g0_instrumentation", None)
            self.assertEqual(module.SEALED_SENTINEL, "exact-commit-blob")
            self.assertEqual(module.__sealed_source_sha256__, _sha(source))
            self.assertEqual(module.__sealed_source_size_bytes__, len(source))
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_source_guard_observes_git_head_swap_restore(self) -> None:
        git_directory = self.working_directory / ".git"
        git_directory.mkdir()
        head = git_directory / "HEAD"
        original = git_directory / "HEAD.original"
        replacement = git_directory / "HEAD.replacement"
        head.write_text("ref: refs/heads/formal\n", encoding="utf-8")
        replacement.write_text("ref: refs/heads/attacker\n", encoding="utf-8")
        with manager._SourceMutationGuard((self.working_directory,)) as guard:
            os.replace(head, original)
            os.replace(replacement, head)
            os.replace(head, replacement)
            os.replace(original, head)
            with self.assertRaisesRegex(manager.ManagerError, "source mutation"):
                guard.checkpoint("git HEAD attack")

    def test_source_guard_observes_root_rename_restore_from_pinned_parent(self) -> None:
        moved = self.working_directory.with_name(
            self.working_directory.name + ".moved"
        )
        with manager._SourceMutationGuard((self.working_directory,)) as guard:
            os.replace(self.working_directory, moved)
            os.replace(moved, self.working_directory)
            with self.assertRaisesRegex(manager.ManagerError, "source mutation"):
                guard.checkpoint("root rename attack")

    def test_gitignored_artifacts_and_replace_refs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "formal-source"
            package = root / "package"
            package.mkdir(parents=True)
            (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
            for command in (
                ["/usr/bin/git", "init", "-q"],
                ["/usr/bin/git", "config", "user.email", "g0@example.invalid"],
                ["/usr/bin/git", "config", "user.name", "G0 Test"],
                ["/usr/bin/git", "add", "."],
                ["/usr/bin/git", "commit", "-q", "-m", "formal source"],
            ):
                subprocess.run(command, cwd=root, check=True)
            commit = subprocess.run(
                ["/usr/bin/git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            manager._verify_clean_git_root(root, commit, "fixture")

            cache = package / "__pycache__"
            cache.mkdir()
            bytecode = cache / "module.cpython-310.pyc"
            bytecode.write_bytes(b"ignored bytecode\n")
            with self.assertRaisesRegex(manager.ManagerError, "gitignored artifacts"):
                manager._verify_clean_git_root(root, commit, "fixture")
            bytecode.unlink()
            cache.rmdir()

            (root / ".git/info/exclude").write_text("*.so\n", encoding="utf-8")
            native = package / "module.cpython-310-x86_64-linux-gnu.so"
            native.write_bytes(b"ignored native extension\n")
            with self.assertRaisesRegex(manager.ManagerError, "gitignored artifacts"):
                manager._verify_clean_git_root(root, commit, "fixture")
            native.unlink()

            subprocess.run(
                [
                    "/usr/bin/git",
                    "update-ref",
                    f"refs/replace/{commit}",
                    commit,
                ],
                cwd=root,
                check=True,
            )
            with self.assertRaisesRegex(manager.ManagerError, "replace refs"):
                manager._verify_clean_git_root(root, commit, "fixture")

    def test_manager_script_does_not_create_bytecode_before_source_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fresh-checkout"
            scripts = root / "scripts"
            validation = root / "src/fastwam/validation"
            scripts.mkdir(parents=True)
            validation.mkdir(parents=True)
            (scripts / "run_mf_wam_g0_manager.py").write_bytes(
                (REPOSITORY_ROOT / "scripts/run_mf_wam_g0_manager.py").read_bytes()
            )
            (validation / "g0_contract.py").write_bytes(
                (REPOSITORY_ROOT / "src/fastwam/validation/g0_contract.py").read_bytes()
            )
            environment = dict(os.environ)
            environment.pop("PYTHONDONTWRITEBYTECODE", None)
            environment.pop("PYTHONPATH", None)
            subprocess.run(
                [sys.executable, str(scripts / "run_mf_wam_g0_manager.py"), "--help"],
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
            )
            self.assertEqual(list(root.rglob("*.pyc")), [])
            self.assertEqual(list(root.rglob("__pycache__")), [])

    def test_linked_worktree_and_external_gitdir_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            primary = base / "primary"
            primary.mkdir()
            (primary / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
            for command in (
                ["/usr/bin/git", "init", "-q"],
                ["/usr/bin/git", "config", "user.email", "g0@example.invalid"],
                ["/usr/bin/git", "config", "user.name", "G0 Test"],
                ["/usr/bin/git", "add", "."],
                ["/usr/bin/git", "commit", "-q", "-m", "primary"],
            ):
                subprocess.run(command, cwd=primary, check=True)
            commit = subprocess.run(
                ["/usr/bin/git", "rev-parse", "HEAD"],
                cwd=primary,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            linked = base / "linked"
            subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(primary),
                    "worktree",
                    "add",
                    "-q",
                    "--detach",
                    str(linked),
                    commit,
                ],
                check=True,
            )
            with self.assertRaises(manager.ManagerError):
                manager._verify_clean_git_root(linked, commit, "linked")

            external_git = base / "external.git"
            indirect = base / "indirect"
            subprocess.run(
                [
                    "/usr/bin/git",
                    "init",
                    "-q",
                    "--separate-git-dir",
                    str(external_git),
                    str(indirect),
                ],
                check=True,
            )
            with self.assertRaises(manager.ManagerError):
                manager._verify_git_checkout_policy(indirect, "indirect")

    def test_reaps_again_before_refilling_gpu_after_a_success(self) -> None:
        launched: list[types.SimpleNamespace] = []

        class SequencedProcess:
            def __init__(self, polls: list[int | None]) -> None:
                self.polls = list(polls)

            def poll(self):
                return self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]

        def fake_popen(argv, **kwargs):
            index = len(launched)
            if index >= 2:
                raise AssertionError("manager dispatched after sibling failure was observable")
            overrides = self._overrides(argv)
            suite = overrides["EVALUATION.task_suite_name"]
            task_id = int(overrides["EVALUATION.task_id"])
            gpu_id = int(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
            if index == 0:
                self._write_success_artifacts(suite, task_id, gpu_id)
                kwargs["stdout"].write(
                    canonical_json_bytes(
                        self._worker_terminal(
                            suite,
                            task_id,
                            gpu_id,
                            manager._canonical_sha(kwargs["env"]),
                        )
                    )
                    + b"\n"
                )
                process = SequencedProcess([0])
            else:
                kwargs["stdout"].write(b"worker failed\n")
                process = SequencedProcess([None, 1])
            launched.append(types.SimpleNamespace(process=process))
            return process

        with (
            mock.patch.object(
                manager,
                "_verify_prelaunch_sources",
                return_value=manager._canonical_python_executable(
                    self.config.python_executable
                ),
            ),
            mock.patch.object(
                manager,
                "_validate_manager_locked_resolved_config",
                return_value=None,
            ),
        ):
            result = manager.run_manager(
                self.config,
                popen_factory=fake_popen,
                sleep=lambda _seconds: None,
            )

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(len(launched), 2)

    def test_scheduler_internal_failure_drains_workers_before_propagating(self) -> None:
        launched: list[types.SimpleNamespace] = []

        class DrainableProcess:
            def __init__(self, poll_code: int | None, wait_code: int) -> None:
                self.poll_code = poll_code
                self.wait_code = wait_code
                self.poll_calls = 0
                self.wait_calls = 0
                self.terminate_calls = 0
                self.kill_calls = 0

            def poll(self):
                self.poll_calls += 1
                return self.poll_code

            def wait(self):
                self.wait_calls += 1
                return self.wait_code

            def terminate(self):
                self.terminate_calls += 1

            def kill(self):
                self.kill_calls += 1

        def fake_popen(argv, **kwargs):
            index = len(launched)
            if index >= 2:
                raise AssertionError("scheduler dispatched after its internal failure")
            process = DrainableProcess(
                poll_code=0 if index == 0 else None,
                wait_code=0 if index == 0 else 1,
            )
            kwargs["stdout"].write(b"incomplete worker log\n")
            launched.append(
                types.SimpleNamespace(
                    process=process,
                    log_handle=kwargs["stdout"],
                )
            )
            return process

        with (
            mock.patch.object(
                manager,
                "_verify_prelaunch_sources",
                return_value=manager._canonical_python_executable(
                    self.config.python_executable
                ),
            ),
            mock.patch.object(
                manager,
                "_validate_manager_locked_resolved_config",
                return_value=None,
            ),
            mock.patch.object(
                manager,
                "_publish_status",
                side_effect=manager.ManagerError("injected status publish failure"),
            ),
        ):
            with self.assertRaisesRegex(
                manager.ManagerError,
                "injected status publish failure",
            ):
                manager.run_manager(
                    self.config,
                    popen_factory=fake_popen,
                    sleep=lambda _seconds: None,
                )

        self.assertEqual(len(launched), 2)
        self.assertTrue(all(item.process.wait_calls == 1 for item in launched))
        self.assertTrue(all(item.log_handle.closed for item in launched))
        self.assertTrue(
            all(item.process.terminate_calls == 0 for item in launched)
        )
        self.assertTrue(all(item.process.kill_calls == 0 for item in launched))
        self.assertFalse((self.raw_log_root / "manager_terminal.json").exists())

    def test_popen_baseexception_closes_untransferred_log(self) -> None:
        opened_logs: list[object] = []

        def interrupting_popen(_argv, **kwargs):
            opened_logs.append(kwargs["stdout"])
            raise KeyboardInterrupt("injected launch interrupt")

        with (
            mock.patch.object(
                manager,
                "_verify_prelaunch_sources",
                return_value=manager._canonical_python_executable(
                    self.config.python_executable
                ),
            ),
            mock.patch.object(
                manager,
                "_validate_manager_locked_resolved_config",
                return_value=None,
            ),
        ):
            with self.assertRaisesRegex(KeyboardInterrupt, "injected launch interrupt"):
                manager.run_manager(
                    self.config,
                    popen_factory=interrupting_popen,
                    sleep=lambda _seconds: None,
                )

        self.assertEqual(len(opened_logs), 1)
        self.assertTrue(opened_logs[0].closed)
        self.assertFalse((self.raw_log_root / "manager_terminal.json").exists())

    def test_renameat2_preflight_cleans_and_missing_capability_starts_nothing(
        self,
    ) -> None:
        manager._probe_renameat2_no_replace(
            self.config.artifact_root,
            self.raw_log_root.parent,
        )
        self.assertEqual(list(self.config.artifact_root.iterdir()), [])
        self.assertFalse(
            any(
                path.name.startswith(manager._RENAME_PROBE_PREFIX)
                for path in self.raw_log_root.parent.iterdir()
            )
        )

        popen = mock.Mock()
        with (
            mock.patch.object(
                manager,
                "_verify_prelaunch_sources",
                return_value=manager._canonical_python_executable(
                    self.config.python_executable
                ),
            ),
            mock.patch.object(
                manager,
                "_validate_manager_locked_resolved_config",
                return_value=None,
            ),
            mock.patch.object(
                manager,
                "_renameat2_no_replace_at",
                side_effect=manager.ManagerError("injected renameat2 unavailable"),
            ),
        ):
            with self.assertRaisesRegex(
                manager.ManagerError,
                "injected renameat2 unavailable",
            ):
                manager.run_manager(self.config, popen_factory=popen)

        popen.assert_not_called()
        self.assertFalse(self.raw_log_root.exists())
        self.assertEqual(list(self.config.artifact_root.iterdir()), [])
        self.assertFalse(
            any(
                path.name.startswith(manager._RENAME_PROBE_PREFIX)
                for path in self.raw_log_root.parent.iterdir()
            )
        )

    def test_read_only_preflight_failure_creates_no_run_directories(self) -> None:
        with mock.patch.object(
            manager,
            "_verify_prelaunch_sources",
            side_effect=manager.ManagerError("dirty instrumentation checkout"),
        ):
            with self.assertRaisesRegex(manager.ManagerError, "dirty instrumentation"):
                manager.run_manager(self.config)
        self.assertFalse(self.raw_log_root.exists())

    def test_non_normalized_runtime_start_fails_before_first_write(self) -> None:
        non_normalized = copy.deepcopy(self.start)
        non_normalized["imports"] = list(reversed(non_normalized["imports"]))
        raw = canonical_json_bytes(non_normalized)
        self.start_path.write_bytes(raw)
        changed = replace(self.config, runtime_start_sha256=_sha(raw))
        with self.assertRaisesRegex(
            manager.ManagerError,
            "validator-normalized canonical JSON bytes",
        ):
            manager.run_manager(changed)
        self.assertFalse(self.raw_log_root.exists())

    def test_task_config_and_python_executable_fail_before_first_write(self) -> None:
        wrong_task = replace(self.config, task_config="libero_joint_2cam224_1e-4")
        with self.assertRaisesRegex(manager.ManagerError, "formal G0 task config"):
            manager.run_manager(wrong_task)
        self.assertFalse(self.raw_log_root.exists())

        wrong_python = replace(self.config, python_executable="/bin/sh")
        with self.assertRaisesRegex(manager.ManagerError, "python_executable"):
            manager.run_manager(wrong_python)
        self.assertFalse(self.raw_log_root.exists())

        relative_python = replace(self.config, python_executable="python3")
        with self.assertRaisesRegex(manager.ManagerError, "absolute path"):
            manager.run_manager(relative_python)
        self.assertFalse(self.raw_log_root.exists())

        changed_task_map = json.loads(self.task_map_path.read_text(encoding="utf-8"))
        changed = changed_task_map["tasks"][0]
        changed["task_name"] += "_changed"
        changed["bddl_path"] = (
            f"bddl_files/{changed['task_suite']}/{changed['task_name']}.bddl"
        )
        changed["init_state_path"] = (
            f"init_files/{changed['task_suite']}/{changed['task_name']}.pruned_init"
        )
        self.task_map_path.write_bytes(canonical_json_bytes(changed_task_map))
        with self.assertRaisesRegex(manager.ManagerError, "task-map digest mismatch"):
            manager.run_manager(self.config)
        self.assertFalse(self.raw_log_root.exists())

    def test_minimal_environment_equals_actual_python_environment(self) -> None:
        environment = manager.build_worker_environment(
            self.config,
            preregistration=self.fixture.prereg,
            gpu_id=0,
            python_hash_seed=self.fixture.schedule["python_hash_seed"],
        )
        code = (
            "import json, os; "
            "from scripts.run_mf_wam_g0_traced import "
            "_validate_formal_process_environment; "
            "print(json.dumps(dict(os.environ), sort_keys=True)); "
            "print(_validate_formal_process_environment())"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPOSITORY_ROOT),
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        lines = completed.stdout.splitlines()
        self.assertEqual(json.loads(lines[-2]), environment)
        self.assertEqual(lines[-1], manager._canonical_sha(environment))

        poisoned = {**environment, "PYTHONPATH": "/tmp/unsealed"}
        with mock.patch.dict(os.environ, poisoned, clear=True):
            with self.assertRaisesRegex(
                traced_runner.TracedRunnerError,
                "unexpected=.*PYTHONPATH",
            ):
                traced_runner._validate_formal_process_environment()


if __name__ == "__main__":
    unittest.main()
