from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from fastwam.validation.g0_contract import (  # noqa: E402
    EXPECTED_EPISODES,
    EXPECTED_TASKS,
    SUITES,
    canonical_json_bytes,
    canonical_json_sha256,
    validate_runtime_start,
)
from scripts.audit_mf_wam_g0_bundle import (  # noqa: E402
    FORMAL_SCOPE,
    _audit_summaries,
)
from scripts import seal_mf_wam_g0_terminal as sealer  # noqa: E402
from scripts.seal_mf_wam_g0_terminal import (  # noqa: E402
    SealError,
    seal_terminal_bundle,
)
from tests import test_g0_contract as contract_tests  # noqa: E402


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _raw_sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tree_sha(files: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda entry: entry["path"].encode("utf-8")):
        digest.update(f"{item['sha256']}  {item['path']}\n".encode("utf-8"))
    return digest.hexdigest()


class G0TerminalSealerTest(unittest.TestCase):
    """Exercise one exact 2,080-input fixture and mutate it fail-closed."""

    def setUp(self) -> None:
        fixture = contract_tests.G0ContractTest(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        self.fixture = fixture
        self.artifact_root = fixture.artifact_root
        self.control_root = self.artifact_root.parent / "control"
        self.raw_log_root = self.artifact_root.parent / "raw-logs"
        self.control_root.mkdir()
        self.raw_log_root.mkdir()
        self.terminal_path = self.artifact_root.parent / "terminal.json"
        launch_artifacts = {
            "checkpoint": ("libero_uncond_2cam224.pt", b"synthetic checkpoint\n"),
            "dataset_stats": ("dataset-stats.json", b'{"synthetic":true}\n'),
            "resolved_config": ("resolved-config.yaml", b"synthetic: true\n"),
        }
        for role, (name, raw) in launch_artifacts.items():
            path = self.control_root / name
            path.write_bytes(raw)
            setattr(self, f"{role}_path", path)
            fixture.prereg["artifacts"][role] = {
                "sha256": _raw_sha(raw),
                "size_bytes": len(raw),
            }
        self.start = validate_runtime_start(
            fixture._runtime_start(), preregistration=fixture.prereg
        )

        self.prereg_path = self._write_control("preregistration.json", fixture.prereg)
        self.start_path = self._write_control("runtime-start.json", self.start)
        self.schedule_path = self._write_control("seed-schedule.json", fixture.schedule)
        self.task_map_path = self._write_control("task-map.json", fixture.task_map)
        self.upstream_digests = {
            "preregistration_file_sha256": _raw_sha(self.prereg_path.read_bytes()),
            "preregistration_canonical_sha256": canonical_json_sha256(fixture.prereg),
            "runtime_start_file_sha256": _raw_sha(self.start_path.read_bytes()),
            "runtime_start_canonical_sha256": canonical_json_sha256(self.start),
            "seed_schedule_file_sha256": _raw_sha(self.schedule_path.read_bytes()),
            "seed_schedule_canonical_sha256": canonical_json_sha256(fixture.schedule),
            "resolved_config_sha256": fixture.prereg["artifacts"]["resolved_config"][
                "sha256"
            ],
        }
        self.manager, self.input_inventory = self._build_exact_inputs_and_manager()
        self.manager_path = self.raw_log_root / "manager_terminal.json"
        self.manager_raw = canonical_json_bytes(self.manager)
        self.manager_path.write_bytes(self.manager_raw)
        self.manager_sha = _raw_sha(self.manager_raw)
        self.first_trace = (
            self.artifact_root / "traces/libero_spatial/task00/trial000.json"
        )
        self.second_trace = (
            self.artifact_root / "traces/libero_spatial/task00/trial001.json"
        )

    def _write_control(self, name: str, payload: dict) -> Path:
        path = self.control_root / name
        path.write_bytes(canonical_json_bytes(payload))
        return path

    def _write_artifact(self, relative: str, role: str, payload: object) -> dict:
        path = self.artifact_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = payload if isinstance(payload, bytes) else canonical_json_bytes(payload)
        path.write_bytes(raw)
        return {
            "path": relative,
            "role": role,
            "sha256": _raw_sha(raw),
            "size_bytes": len(raw),
        }

    def _write_raw(self, relative: str, raw: bytes) -> dict:
        path = self.raw_log_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {
            "path": relative,
            "sha256": _raw_sha(raw),
            "size_bytes": len(raw),
        }

    def _source_identity(self, *, official: bool) -> dict:
        role = (
            "official_policy_and_evaluator_source"
            if official
            else "external_observer_and_launcher_source"
        )
        commit = self.fixture.prereg["source"][
            "fastwam" if official else "instrumentation"
        ]["commit"]
        return {
            "status": "PASS",
            "role": role,
            "root": "/opt/official" if official else "/opt/instrumentation",
            "commit": commit,
            "tree": _digest(f"tree:{role}")[:40],
            "clean": True,
            "critical_files": [],
            "critical_file_inventory_sha256": _digest(f"files:{role}"),
        }

    def _trace_payload(
        self,
        *,
        process: dict,
        trial_idx: int,
        success: bool,
        official_source: dict,
        instrumentation_source: dict,
    ) -> dict:
        records = []
        for replan_idx in range(7):
            proposal = [[0.0] * 7 for _ in range(32)]
            executed_count = 1 if success and replan_idx == 6 else 10
            executed = [row[:] for row in proposal[:executed_count]]
            executions = []
            for execution_idx, action in enumerate(executed):
                done = (
                    success
                    and replan_idx == 6
                    and execution_idx == executed_count - 1
                )
                executions.append(
                    {
                        "action": action[:],
                        "post_state": [0.0] * 8,
                        "post_observation_sha256": _digest(
                            f"post:{process['process_id']}:{trial_idx}:"
                            f"{replan_idx}:{execution_idx}"
                        ),
                        "done": done,
                    }
                )
            records.append(
                {
                    "episode_idx": trial_idx,
                    "replan_idx": replan_idx,
                    "env_step": 30 + replan_idx * 10,
                    "state": [0.0] * 8,
                    "pre_state": [0.0] * 8,
                    "pre_observation_sha256": _digest(
                        f"pre:{process['process_id']}:{trial_idx}:{replan_idx}"
                    ),
                    "policy_seed": process["policy_seed"],
                    "policy_seed_scope": "fresh_generator_per_replan",
                    "proposed_raw_action_chunk": proposal,
                    "proposed_env_action_chunk": proposal,
                    "executed_env_actions": executed,
                    "executed_count": executed_count,
                    "done_after_execution": executions[-1]["done"],
                    "executions": executions,
                }
            )
        total_executed = sum(record["executed_count"] for record in records)
        return {
            "schema_version": 2,
            "kind": "mf_wam_g0_structured_trace",
            "metadata": {
                "run_id": self.fixture.prereg["run_id"],
                "task_suite": process["task_suite"],
                "task_id": process["task_id"],
                "trial_idx": trial_idx,
                "initial_state_index": trial_idx,
                "initial_state_sha256": _digest(
                    f"initial:{process['process_id']}:{trial_idx}"
                ),
                "task_description": f"synthetic task {process['process_id']}",
                "warmup_steps": 30,
                "first_replan_env_step": 30,
                "replan_steps": 10,
                "action_horizon": 32,
                "action_dimension": 7,
                "state_dimension": 8,
                "seed_contract": {
                    "task_seed": process["global_seed"],
                    "effective_global_rank": 0,
                    "effective_process_seed": process["global_seed"],
                    "task_seed_scope": (
                        "once_per_task_process_before_model_and_benchmark_construction"
                    ),
                    "environment_seed": process["environment_seed"],
                    "environment_seed_scope": (
                        "once_per_task_process_before_trial_loop"
                    ),
                    "policy_seed": process["policy_seed"],
                    "policy_seed_scope": "fresh_generator_per_replan",
                    "episode_rng_position": (
                        "ordered_trial_index_in_shared_task_environment_stream"
                    ),
                },
                "seed_schedule_process": copy.deepcopy(process),
                "upstream_digests": copy.deepcopy(self.upstream_digests),
                "official_source": copy.deepcopy(official_source),
                "instrumentation_source": copy.deepcopy(instrumentation_source),
                "success": success,
                "record_count": len(records),
                "environment_step_count": 30 + total_executed,
                "observer_rng_unchanged_checks": 1,
                "official_module_origin_inventory_sha256": _digest(
                    f"modules:{process['process_id']}:{trial_idx}"
                ),
            },
            "records": records,
        }

    def _build_exact_inputs_and_manager(self) -> tuple[dict, list[dict]]:
        inventory: list[dict] = []
        manager_processes: list[dict] = []
        official_source = self._source_identity(official=True)
        instrumentation_source = self._source_identity(official=False)
        for index, process in enumerate(self.fixture.schedule["task_processes"]):
            suite = process["task_suite"]
            task_id = process["task_id"]
            process_id = process["process_id"]
            gpu_id = index % self.fixture.prereg["launch"]["gpu_count"]
            success_trials = [0]
            result_payload = {
                "task_suite": suite,
                "task_id": task_id,
                "task_description": f"synthetic task {process_id}",
                "successes": len(success_trials),
                "total_episodes": 50,
                "gpu_id": gpu_id,
                "success_episodes": success_trials,
                "failure_episodes": list(range(1, 50)),
                "start_time": "2026-08-02T18:06:00+08:00",
                "duration": 60.0,
            }
            result = self._write_artifact(
                f"results/{suite}/task{task_id:02d}.json",
                "task_result",
                result_payload,
            )
            inventory.append(result)
            traces = []
            for trial_idx in range(50):
                trace = self._write_artifact(
                    f"traces/{suite}/task{task_id:02d}/trial{trial_idx:03d}.json",
                    "episode_trace",
                    self._trace_payload(
                        process=process,
                        trial_idx=trial_idx,
                        success=trial_idx in success_trials,
                        official_source=official_source,
                        instrumentation_source=instrumentation_source,
                    ),
                )
                inventory.append(trace)
                traces.append(
                    {
                        "trial_idx": trial_idx,
                        "path": trace["path"],
                        "sha256": trace["sha256"],
                        "size_bytes": trace["size_bytes"],
                    }
                )
            trace_tree = _tree_sha(traces)
            receipt_payload = {
                "schema_version": 1,
                "kind": "mf_wam_g0_task_trace_receipt",
                "run_id": self.fixture.prereg["run_id"],
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
                    "runtime_start_canonical_sha256": canonical_json_sha256(
                        self.start
                    ),
                    "seed_schedule_canonical_sha256": canonical_json_sha256(
                        self.fixture.schedule
                    ),
                    "resolved_config_sha256": self.fixture.prereg["artifacts"][
                        "resolved_config"
                    ]["sha256"],
                    "image_digest": self.fixture.prereg["image"]["digest"],
                    "fastwam_commit": self.fixture.prereg["source"]["fastwam"][
                        "commit"
                    ],
                    "instrumentation_commit": self.fixture.prereg["source"][
                        "instrumentation"
                    ]["commit"],
                },
                "seeds": {
                    field: process[field]
                    for field in (
                        "global_seed", "environment_seed", "environment_seed_scope",
                        "policy_seed", "policy_seed_scope", "python_hash_seed",
                        "trial_order", "initial_state_index_rule",
                    )
                },
                "official_result": {
                    field: result[field] for field in ("path", "sha256", "size_bytes")
                },
                "episode_count": 50,
                "traces": traces,
                "tree_sha256": trace_tree,
            }
            receipt = self._write_artifact(
                f"trace_receipts/{suite}/task{task_id:02d}.json",
                "task_trace_receipt",
                receipt_payload,
            )
            inventory.append(receipt)

            archive = self._write_raw(
                f"official/{suite}/gpu{gpu_id}_task{task_id}_results.json",
                (self.artifact_root / result["path"]).read_bytes(),
            )
            command_argv = [
                "python3", "/opt/instrumentation/scripts/run_mf_wam_g0_traced.py",
                "task=libero_uncond_2cam224_1e-4",
                f"ckpt={self.checkpoint_path}",
                f"EVALUATION.dataset_stats_path={self.dataset_stats_path}",
                f"EVALUATION.task_suite_name={suite}",
                f"EVALUATION.task_id={task_id}",
                f"gpu_id={gpu_id}",
                f"output_dir={self.artifact_root}",
                "EVALUATION.num_trials=50",
                f"EVALUATION.output_dir={self.artifact_root}",
                "EVALUATION.env_num=1",
                "EVALUATION.num_steps_wait=30",
                "EVALUATION.replan_steps=10",
                "EVALUATION.action_horizon=32",
                "EVALUATION.binarize_gripper=true",
                f"seed={process['global_seed']}",
                "EVALUATION.visualize_future_video=false",
                "EVALUATION.use_action_ensembler=false",
            ]
            environment_bindings = {
                **sealer._FIXED_WORKER_ENVIRONMENT,
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "DIFFSYNTH_DOWNLOAD_SOURCE": self.fixture.prereg[
                    "runtime_environment"
                ]["DIFFSYNTH_DOWNLOAD_SOURCE"],
                "DIFFSYNTH_MODEL_BASE_PATH": self.fixture.prereg[
                    "runtime_environment"
                ]["DIFFSYNTH_MODEL_BASE_PATH"],
                "DIFFSYNTH_SKIP_DOWNLOAD": self.fixture.prereg[
                    "runtime_environment"
                ]["DIFFSYNTH_SKIP_DOWNLOAD"],
                "LOCAL_RANK": "0",
                "MF_WAM_G0_PREREG_PATH": str(self.prereg_path),
                "MF_WAM_G0_PREREG_SHA256": self.upstream_digests[
                    "preregistration_file_sha256"
                ],
                "MF_WAM_G0_RESOLVED_CONFIG_PATH": str(self.resolved_config_path),
                "MF_WAM_G0_RESOLVED_CONFIG_SHA256": self.upstream_digests[
                    "resolved_config_sha256"
                ],
                "MF_WAM_G0_RUN_ID": self.fixture.prereg["run_id"],
                "MF_WAM_G0_RUNTIME_START_PATH": str(self.start_path),
                "MF_WAM_G0_RUNTIME_START_SHA256": self.upstream_digests[
                    "runtime_start_file_sha256"
                ],
                "MF_WAM_G0_SEED_SCHEDULE_PATH": str(self.schedule_path),
                "MF_WAM_G0_SEED_SCHEDULE_SHA256": self.upstream_digests[
                    "seed_schedule_file_sha256"
                ],
                "MF_WAM_INSTRUMENTATION_COMMIT": self.fixture.prereg["source"][
                    "instrumentation"
                ]["commit"],
                "MF_WAM_OFFICIAL_COMMIT": self.fixture.prereg["source"]["fastwam"][
                    "commit"
                ],
                "MF_WAM_OFFICIAL_ROOT": "/opt/official",
                "MUJOCO_GL": self.fixture.prereg["runtime_environment"]["MUJOCO_GL"],
                "PYTHONHASHSEED": str(self.fixture.schedule["python_hash_seed"]),
                "PYOPENGL_PLATFORM": self.fixture.prereg["runtime_environment"][
                    "PYOPENGL_PLATFORM"
                ],
                "RANK": "0",
                "WORLD_SIZE": "1",
            }
            command_sha = canonical_json_sha256(command_argv)
            environment_sha = canonical_json_sha256(environment_bindings)
            worker_terminal = {
                "status": "PASS",
                "kind": "mf_wam_g0_traced_worker_terminal",
                "run_id": self.fixture.prereg["run_id"],
                "process_receipt": str(self.artifact_root / receipt["path"]),
                "official_commit": self.fixture.prereg["source"]["fastwam"][
                    "commit"
                ],
                "official_result_type": "dict",
                "official_result_receipt": {
                    "path": result["path"],
                    "sha256": result["sha256"],
                    "size_bytes": result["size_bytes"],
                    "source_path": (
                        f"{suite}/gpu{gpu_id}_task{task_id}_results.json"
                    ),
                    "source_sha256": archive["sha256"],
                    "source_size_bytes": archive["size_bytes"],
                    "success_episodes": success_trials,
                    "failure_episodes": list(range(1, 50)),
                },
                "terminal_source_identities": {
                    "status": "PASS",
                    "official": copy.deepcopy(official_source),
                    "instrumentation": copy.deepcopy(instrumentation_source),
                },
                "external_prelaunch_commit_tree_gate_required": True,
                "environment_sha256": environment_sha,
            }
            log = self._write_raw(
                f"logs/{suite}/task{task_id:02d}.log",
                (
                    f"{process_id} completed successfully\n".encode("utf-8")
                    + canonical_json_bytes(worker_terminal)
                    + b"\n"
                ),
            )
            launched_at = "2026-08-02T18:06:00+08:00"
            task_completed_at = "2026-08-02T18:20:00+08:00"
            status_payload = {
                "schema_version": 1,
                "kind": "mf_wam_g0_manager_task_status",
                "run_id": self.fixture.prereg["run_id"],
                "process_id": process_id,
                "task_suite": suite,
                "task_id": task_id,
                "gpu_id": gpu_id,
                "state": "SUCCEEDED",
                "launched_at": launched_at,
                "completed_at": task_completed_at,
                "exit_code": 0,
                "complete": True,
                "failure_reason": None,
                "command_argv": command_argv,
                "command_sha256": command_sha,
                "environment_bindings": environment_bindings,
                "environment_sha256": environment_sha,
                "log": log,
                "canonical_result": {
                    field: result[field] for field in ("path", "sha256", "size_bytes")
                },
                "trace_receipt": {
                    **{
                        field: receipt[field]
                        for field in ("path", "sha256", "size_bytes")
                    },
                    "tree_sha256": trace_tree,
                    "episode_count": 50,
                },
                "raw_result": {
                    "source_path": f"{suite}/gpu{gpu_id}_task{task_id}_results.json",
                    "archive_path": archive["path"],
                    "sha256": archive["sha256"],
                    "size_bytes": archive["size_bytes"],
                },
            }
            status = self._write_raw(
                f"status/{suite}/task{task_id:02d}.json",
                canonical_json_bytes(status_payload),
            )
            manager_processes.append(
                {
                    "process_id": process_id,
                    "task_suite": suite,
                    "task_id": task_id,
                    "gpu_id": gpu_id,
                    "state": "SUCCEEDED",
                    "launched_at": launched_at,
                    "completed_at": task_completed_at,
                    "exit_code": 0,
                    "complete": True,
                    "failure_reason": None,
                    "command_sha256": command_sha,
                    "environment_sha256": environment_sha,
                    "log_path": log["path"],
                    "log_sha256": log["sha256"],
                    "log_size_bytes": log["size_bytes"],
                    "status_path": status["path"],
                    "status_sha256": status["sha256"],
                    "status_size_bytes": status["size_bytes"],
                    "result_path": result["path"],
                    "result_sha256": result["sha256"],
                    "result_size_bytes": result["size_bytes"],
                    "trace_receipt_path": receipt["path"],
                    "trace_receipt_sha256": receipt["sha256"],
                    "trace_receipt_size_bytes": receipt["size_bytes"],
                    "trace_tree_sha256": trace_tree,
                    "episode_count": 50,
                    "raw_result_source_path": status_payload["raw_result"][
                        "source_path"
                    ],
                    "raw_result_archive_path": archive["path"],
                    "raw_result_sha256": archive["sha256"],
                    "raw_result_size_bytes": archive["size_bytes"],
                }
            )
        self.assertEqual(len(inventory), 2080)
        manager = {
            "schema_version": 1,
            "kind": "mf_wam_g0_manager_terminal_manifest",
            "run_id": self.fixture.prereg["run_id"],
            "completed_at": "2026-08-02T18:30:00+08:00",
            "manager_exit_code": 0,
            "artifact_root": str(self.artifact_root),
            "raw_log_root": str(self.raw_log_root),
            "gpu_ids": list(range(self.fixture.prereg["launch"]["gpu_count"])),
            "upstream_bindings": {
                "preregistration_file_sha256": self.upstream_digests[
                    "preregistration_file_sha256"
                ],
                "runtime_start_file_sha256": self.upstream_digests[
                    "runtime_start_file_sha256"
                ],
                "seed_schedule_file_sha256": self.upstream_digests[
                    "seed_schedule_file_sha256"
                ],
                "resolved_config_sha256": self.upstream_digests[
                    "resolved_config_sha256"
                ],
                "official_commit": self.fixture.prereg["source"]["fastwam"][
                    "commit"
                ],
                "instrumentation_commit": self.fixture.prereg["source"][
                    "instrumentation"
                ]["commit"],
                "python_hash_seed": self.fixture.schedule["python_hash_seed"],
            },
            "canonical_input_file_count": 2080,
            "canonical_input_tree_sha256": _tree_sha(inventory),
            "task_processes": manager_processes,
        }
        return manager, inventory

    def _seal(self, *, manager_sha: str | None = None) -> dict:
        return seal_terminal_bundle(
            artifact_root=self.artifact_root,
            preregistration_path=self.prereg_path,
            runtime_start_path=self.start_path,
            seed_schedule_path=self.schedule_path,
            resolved_config_path=self.resolved_config_path,
            task_map_path=self.task_map_path,
            manager_manifest_path=self.manager_path,
            trusted_manager_manifest_sha256=manager_sha or self.manager_sha,
            terminal_output=self.terminal_path,
        )

    def _assert_unsealed(self) -> None:
        self.assertFalse(self.terminal_path.exists())
        for name in (
            "summary.csv", "task_success_rates.csv", "summary.json",
            "completion.json", "artifact_inventory.json",
        ):
            self.assertFalse((self.artifact_root / name).exists())

    def _install_resigned_first_status(self, status_payload: dict) -> str:
        """Rewrite status + manifest hashes exactly like a compromised manager."""

        bad_manager = copy.deepcopy(self.manager)
        manager_item = bad_manager["task_processes"][0]
        status_payload["command_sha256"] = canonical_json_sha256(
            status_payload["command_argv"]
        )
        status_payload["environment_sha256"] = canonical_json_sha256(
            status_payload["environment_bindings"]
        )
        manager_item["command_sha256"] = status_payload["command_sha256"]
        manager_item["environment_sha256"] = status_payload["environment_sha256"]
        status_raw = canonical_json_bytes(status_payload)
        status_path = self.raw_log_root / manager_item["status_path"]
        status_path.write_bytes(status_raw)
        manager_item["status_sha256"] = _raw_sha(status_raw)
        manager_item["status_size_bytes"] = len(status_raw)
        manager_raw = canonical_json_bytes(bad_manager)
        self.manager_path.write_bytes(manager_raw)
        return _raw_sha(manager_raw)

    def test_fail_closed_attacks_then_exact_positive_seal(self) -> None:
        first_raw = self.first_trace.read_bytes()

        with self.subTest("missing"):
            self.first_trace.unlink()
            with self.assertRaisesRegex(SealError, "file scope mismatch"):
                self._seal()
            self.first_trace.write_bytes(first_raw)
            self._assert_unsealed()

        with self.subTest("extra"):
            extra = self.artifact_root / "unexpected.json"
            extra.write_bytes(b"{}")
            with self.assertRaisesRegex(SealError, "file scope mismatch"):
                self._seal()
            extra.unlink()
            self._assert_unsealed()

        with self.subTest("symlink"):
            outside = self.artifact_root.parent / "outside-trace.json"
            outside.write_bytes(first_raw)
            self.first_trace.unlink()
            self.first_trace.symlink_to(outside)
            with self.assertRaisesRegex(SealError, "symlink is forbidden"):
                self._seal()
            self.first_trace.unlink()
            self.first_trace.write_bytes(first_raw)
            self._assert_unsealed()

        with self.subTest("hardlink"):
            self.first_trace.unlink()
            os.link(self.second_trace, self.first_trace)
            with self.assertRaisesRegex(SealError, "hardlink"):
                self._seal()
            self.first_trace.unlink()
            self.first_trace.write_bytes(first_raw)
            self._assert_unsealed()

        with self.subTest("trace tamper"):
            tampered = json.loads(first_raw)
            tampered["metadata"]["success"] = False
            self.first_trace.write_bytes(canonical_json_bytes(tampered))
            with self.assertRaisesRegex(SealError, "trace metadata contract mismatch"):
                self._seal()
            self.first_trace.write_bytes(first_raw)
            self._assert_unsealed()

        with self.subTest("untrusted manager"):
            with self.assertRaisesRegex(SealError, "trusted file digest"):
                self._seal(manager_sha=_digest("wrong manager"))
            self._assert_unsealed()

        with self.subTest("manager content mismatch"):
            bad_manager = copy.deepcopy(self.manager)
            bad_manager["task_processes"][0]["result_sha256"] = _digest(
                "wrong result"
            )
            bad_raw = canonical_json_bytes(bad_manager)
            self.manager_path.write_bytes(bad_raw)
            with self.assertRaisesRegex(SealError, "manager status canonical_result"):
                self._seal(manager_sha=_raw_sha(bad_raw))
            self.manager_path.write_bytes(self.manager_raw)
            self._assert_unsealed()

        first_status_path = (
            self.raw_log_root / self.manager["task_processes"][0]["status_path"]
        )
        first_status_raw = first_status_path.read_bytes()
        with self.subTest("resigned /bin/true runner substitution"):
            bad_status = json.loads(first_status_raw)
            bad_status["command_argv"] = ["/bin/true"]
            try:
                manager_sha = self._install_resigned_first_status(bad_status)
                with self.assertRaisesRegex(SealError, "runner command contract"):
                    self._seal(manager_sha=manager_sha)
            finally:
                first_status_path.write_bytes(first_status_raw)
                self.manager_path.write_bytes(self.manager_raw)
            self._assert_unsealed()

        with self.subTest("resigned forbidden environment injection"):
            bad_status = json.loads(first_status_raw)
            bad_status["environment_bindings"]["MF_WAM_G0_TRACE_ROOT"] = "/tmp/evil"
            try:
                manager_sha = self._install_resigned_first_status(bad_status)
                with self.assertRaisesRegex(SealError, "environment binding"):
                    self._seal(manager_sha=manager_sha)
            finally:
                first_status_path.write_bytes(first_status_raw)
                self.manager_path.write_bytes(self.manager_raw)
            self._assert_unsealed()

        for attack in ("missing model base", "model base drift", "downloads enabled"):
            with self.subTest(f"resigned {attack} environment"):
                bad_status = json.loads(first_status_raw)
                environment = bad_status["environment_bindings"]
                if attack == "missing model base":
                    del environment["DIFFSYNTH_MODEL_BASE_PATH"]
                    expected_error = "environment binding"
                elif attack == "model base drift":
                    environment["DIFFSYNTH_MODEL_BASE_PATH"] = "/tmp/other-model-cache"
                    expected_error = "formal runner contract"
                else:
                    environment["DIFFSYNTH_SKIP_DOWNLOAD"] = "false"
                    expected_error = "formal runner contract"
                try:
                    manager_sha = self._install_resigned_first_status(bad_status)
                    with self.assertRaisesRegex(SealError, expected_error):
                        self._seal(manager_sha=manager_sha)
                finally:
                    first_status_path.write_bytes(first_status_raw)
                    self.manager_path.write_bytes(self.manager_raw)
                self._assert_unsealed()

        hydra_attacks = {
            "duplicate": lambda argv: argv + ["task=conflicting_task"],
            "missing": lambda argv: [
                item
                for item in argv
                if not item.startswith("EVALUATION.action_horizon=")
            ],
            "conflicting": lambda argv: [
                "EVALUATION.task_suite_name=libero_goal"
                if item.startswith("EVALUATION.task_suite_name=")
                else item
                for item in argv
            ],
        }
        for attack, mutate in hydra_attacks.items():
            with self.subTest(f"resigned Hydra {attack} override"):
                bad_status = json.loads(first_status_raw)
                bad_status["command_argv"] = mutate(bad_status["command_argv"])
                try:
                    manager_sha = self._install_resigned_first_status(bad_status)
                    with self.assertRaisesRegex(SealError, "Hydra override"):
                        self._seal(manager_sha=manager_sha)
                finally:
                    first_status_path.write_bytes(first_status_raw)
                    self.manager_path.write_bytes(self.manager_raw)
                self._assert_unsealed()

        with self.subTest("resigned upstream environment substitution"):
            bad_status = json.loads(first_status_raw)
            bad_status["environment_bindings"]["MF_WAM_OFFICIAL_COMMIT"] = "d" * 40
            try:
                manager_sha = self._install_resigned_first_status(bad_status)
                with self.assertRaisesRegex(SealError, "formal runner contract"):
                    self._seal(manager_sha=manager_sha)
            finally:
                first_status_path.write_bytes(first_status_raw)
                self.manager_path.write_bytes(self.manager_raw)
            self._assert_unsealed()

        with self.subTest("semantic terminal failure publishes nothing"):
            with mock.patch.object(
                sealer.contract,
                "validate_terminal_receipt",
                side_effect=sealer.contract.ContractError("forced semantic failure"),
            ):
                with self.assertRaisesRegex(
                    SealError, "pre-publication terminal validation"
                ):
                    self._seal()
            self._assert_unsealed()

        published: list[Path] = []
        original_publish = sealer._atomic_publish_absolute

        def recording_publish(path: Path, raw: bytes) -> None:
            published.append(Path(path))
            original_publish(path, raw)

        with mock.patch.object(
            sealer, "_atomic_publish_absolute", side_effect=recording_publish
        ):
            receipt = self._seal()
        self.assertEqual(published[-1], self.artifact_root / "completion.json")
        self.assertEqual(receipt["status"], "STRUCTURAL_PASS_ONLY")
        self.assertEqual(receipt["specialized_g0_status"], "UNCERTAIN")
        self.assertFalse(receipt["formal_training_allowed"])
        self.assertEqual(receipt["audit_scope"], "terminal_artifacts_only")
        self.assertFalse(receipt["full_contract_chain_validated"])
        self.assertEqual(receipt["artifact_count"], 2084)
        self.assertEqual(receipt["task_process_count"], EXPECTED_TASKS)
        self.assertEqual(receipt["episode_count"], EXPECTED_EPISODES)

        terminal = json.loads(self.terminal_path.read_text(encoding="utf-8"))
        inventory = json.loads(
            (self.artifact_root / "artifact_inventory.json").read_text(
                encoding="utf-8"
            )
        )
        summary = json.loads(
            (self.artifact_root / "summary.json").read_text(encoding="utf-8")
        )
        completion = json.loads(
            (self.artifact_root / "completion.json").read_text(encoding="utf-8")
        )
        self.assertEqual(inventory["file_count"], 2084)
        self.assertEqual(len(inventory["files"]), 2084)
        self.assertNotIn(
            "artifact_inventory.json", {item["path"] for item in inventory["files"]}
        )
        self.assertEqual(summary["total_successes"], EXPECTED_TASKS)
        self.assertEqual(len(summary["task_results"]), EXPECTED_TASKS)
        self.assertEqual(summary["overall"]["average_success_rate"], 2.0)
        task_rows = []
        for suite in SUITES:
            for task_id in range(10):
                task = json.loads(
                    (
                        self.artifact_root
                        / f"results/{suite}/task{task_id:02d}.json"
                    ).read_text(encoding="utf-8")
                )
                task_rows.append(
                    {
                        "task_suite": suite,
                        "task_id": task_id,
                        "successes": task["successes"],
                        "success_rate": task["successes"] / task["total_episodes"],
                    }
                )
        _audit_summaries(
            self.artifact_root,
            run_id=self.fixture.prereg["run_id"],
            task_rows=task_rows,
            scope=FORMAL_SCOPE,
        )
        terminal_core = {
            key: value for key, value in terminal.items() if key != "artifact_inventory"
        }
        self.assertEqual(
            completion["terminal_core_canonical_sha256"],
            canonical_json_sha256(terminal_core),
        )
        actual_files = [path for path in self.artifact_root.rglob("*") if path.is_file()]
        self.assertEqual(len(actual_files), 2085)

        with self.subTest("overwrite"):
            with self.assertRaisesRegex(SealError, "refusing to overwrite"):
                self._seal()


if __name__ == "__main__":
    unittest.main()
