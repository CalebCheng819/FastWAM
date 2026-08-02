from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_mf_wam_g0_bundle.py"
SPEC = importlib.util.spec_from_file_location("audit_mf_wam_g0_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _write_json(path: Path, payload: object) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest(), len(data)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SpecializedBundleAuditTest(unittest.TestCase):
    scope = AUDIT.AuditScope(
        suites=("libero_spatial",),
        tasks_per_suite=1,
        trials_per_task=2,
        minimum_records_per_episode=1,
        bootstrap_replicates=100,
    )

    def _anchor(self, root: Path, role: str, run_id: str) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "mf_wam_g0_run_anchor_manifest",
            "run_role": role,
            "run_id": run_id,
            "artifact_root": str(root),
            "raw_log_root": str(root.parent / f"{role}-raw-logs"),
            "manager_manifest_sha256": "0" * 64,
            "preregistration_canonical_sha256": "1" * 64,
            "runtime_start_canonical_sha256": "2" * 64,
            "seed_schedule_canonical_sha256": "3" * 64,
            "resolved_config_sha256": "4" * 64,
            "terminal_canonical_sha256": "5" * 64,
            "structural_audit_file_sha256": "6" * 64,
            "approved_assets_manifest_sha256": "7" * 64,
            "image_digest": "sha256:" + "8" * 64,
            "fastwam_commit": AUDIT.EXPECTED_FASTWAM_COMMIT,
            "instrumentation_commit": "9" * 40,
        }

    def _trace(
        self,
        anchor: dict[str, object],
        trial_idx: int,
        success: bool,
    ) -> dict[str, object]:
        raw = [
            [float(index + column) / 100.0 for column in range(7)]
            for index in range(32)
        ]
        env = [
            [
                *row[:-1],
                1.0
                if -((row[-1] * 2.0) - 1.0) > 0.0
                else -1.0
                if -((row[-1] * 2.0) - 1.0) < 0.0
                else 0.0,
            ]
            for row in raw
        ]
        action = env[0]
        record = {
            "episode_idx": trial_idx,
            "replan_idx": 0,
            "env_step": 30,
            "state": [0.0] * 8,
            "pre_state": [0.0] * 8,
            "pre_observation_sha256": "a" * 64,
            "policy_seed": 42,
            "policy_seed_scope": "fresh_generator_per_replan",
            "proposed_raw_action_chunk": raw,
            "proposed_env_action_chunk": env,
            "executed_env_actions": [action],
            "executed_count": 1,
            "done_after_execution": success,
            "executions": [
                {
                    "action": action,
                    "post_state": [1.0] * 8,
                    "post_observation_sha256": "b" * 64,
                    "done": success,
                }
            ],
        }
        upstream = {
            "preregistration_file_sha256": "c" * 64,
            "preregistration_canonical_sha256": anchor[
                "preregistration_canonical_sha256"
            ],
            "runtime_start_file_sha256": "d" * 64,
            "runtime_start_canonical_sha256": anchor[
                "runtime_start_canonical_sha256"
            ],
            "seed_schedule_file_sha256": "e" * 64,
            "seed_schedule_canonical_sha256": anchor[
                "seed_schedule_canonical_sha256"
            ],
            "resolved_config_sha256": anchor["resolved_config_sha256"],
        }
        process = {
            "process_id": "libero_spatial/task00",
            "task_suite": "libero_spatial",
            "task_id": 0,
            "global_rank": 0,
            "global_seed": 42,
            "environment_seed": 42,
            "environment_seed_scope": "once-before-trial-0",
            "policy_seed": 42,
            "policy_seed_scope": "constant-each-replan-call",
            "python_hash_seed": 42,
            "trial_order": [0, 1],
            "initial_state_index_rule": "trial_idx",
        }
        return {
            "schema_version": 2,
            "kind": "mf_wam_g0_structured_trace",
            "metadata": {
                "run_id": anchor["run_id"],
                "task_suite": "libero_spatial",
                "task_id": 0,
                "trial_idx": trial_idx,
                "initial_state_index": trial_idx,
                "initial_state_sha256": "f" * 64,
                "task_description": "synthetic",
                "warmup_steps": 30,
                "first_replan_env_step": 30,
                "replan_steps": 10,
                "action_horizon": 32,
                "action_dimension": 7,
                "state_dimension": 8,
                "seed_contract": {
                    "task_seed": 42,
                    "effective_global_rank": 0,
                    "effective_process_seed": 42,
                    "task_seed_scope": "once_per_task_process_before_model_and_benchmark_construction",
                    "environment_seed": 42,
                    "environment_seed_scope": "once_per_task_process_before_trial_loop",
                    "policy_seed": 42,
                    "policy_seed_scope": "fresh_generator_per_replan",
                    "episode_rng_position": "ordered_trial_index_in_shared_task_environment_stream",
                },
                "seed_schedule_process": process,
                "upstream_digests": upstream,
                "official_source": {
                    "commit": anchor["fastwam_commit"],
                    "clean": True,
                },
                "instrumentation_source": {
                    "commit": anchor["instrumentation_commit"],
                    "clean": True,
                },
                "success": success,
                "record_count": 1,
                "environment_step_count": 31,
                "observer_rng_unchanged_checks": 1,
                "official_module_origin_inventory_sha256": "1" * 64,
            },
            "records": [record],
        }

    def _write_run(
        self,
        root: Path,
        anchor: dict[str, object],
        outcomes: list[bool] | None = None,
    ) -> None:
        outcomes = [True, False] if outcomes is None else outcomes
        trace_rows = []
        for trial_idx, success in enumerate(outcomes):
            relative = f"traces/libero_spatial/task00/trial{trial_idx:03d}.json"
            digest, size = _write_json(root / relative, self._trace(anchor, trial_idx, success))
            trace_rows.append(
                {
                    "trial_idx": trial_idx,
                    "path": relative,
                    "sha256": digest,
                    "size_bytes": size,
                }
            )
        successes = [index for index, value in enumerate(outcomes) if value]
        failures = [index for index, value in enumerate(outcomes) if not value]
        result = {
            "task_suite": "libero_spatial",
            "task_id": 0,
            "gpu_id": 0,
            "total_episodes": 2,
            "successes": len(successes),
            "success_rate": len(successes) / 2,
            "success_episodes": successes,
            "failure_episodes": failures,
            "duration": 1.0,
            "task_description": "synthetic",
        }
        result_path = "results/libero_spatial/task00.json"
        result_sha, result_size = _write_json(root / result_path, result)
        receipt = {
            "schema_version": 1,
            "kind": "mf_wam_g0_task_trace_receipt",
            "run_id": anchor["run_id"],
            "process_id": "libero_spatial/task00",
            "task_suite": "libero_spatial",
            "task_id": 0,
            "execution_scope": "one-process-per-task",
            "world_size": 1,
            "global_rank": 0,
            "local_rank": 0,
            "bindings": {
                "preregistration_canonical_sha256": anchor[
                    "preregistration_canonical_sha256"
                ],
                "runtime_start_canonical_sha256": anchor[
                    "runtime_start_canonical_sha256"
                ],
                "seed_schedule_canonical_sha256": anchor[
                    "seed_schedule_canonical_sha256"
                ],
                "resolved_config_sha256": anchor["resolved_config_sha256"],
                "image_digest": anchor["image_digest"],
                "fastwam_commit": anchor["fastwam_commit"],
                "instrumentation_commit": anchor["instrumentation_commit"],
            },
            "seeds": {
                "global_seed": 42,
                "environment_seed": 42,
                "environment_seed_scope": "once-before-trial-0",
                "policy_seed": 42,
                "policy_seed_scope": "constant-each-replan-call",
                "python_hash_seed": 42,
                "trial_order": [0, 1],
                "initial_state_index_rule": "trial_idx",
            },
            "official_result": {
                "path": result_path,
                "sha256": result_sha,
                "size_bytes": result_size,
            },
            "episode_count": 2,
            "traces": trace_rows,
            "tree_sha256": AUDIT._tree_sha256(trace_rows),
        }
        _write_json(root / "trace_receipts/libero_spatial/task00.json", receipt)
        percent = len(successes) / 2 * 100
        summary = {
            "run_id": anchor["run_id"],
            "ckpt": "synthetic",
            "config": "synthetic",
            "suite_stats": {
                "libero_spatial": {
                    "total_tasks": 1,
                    "total_trials": 2,
                    "total_successes": len(successes),
                    "total_time": 1.0,
                    "max_time": 1.0,
                }
            },
            "task_results": {
                "libero_spatial_0": {
                    "success_rate": percent,
                    "duration": 1.0,
                    "total_episodes": 2,
                    "successes": len(successes),
                    "task_description": "synthetic",
                }
            },
            "overall": {
                "average_success_rate": percent,
                "total_time": 1.0,
                "average_task_time": 1.0,
            },
        }
        _write_json(root / "summary.json", summary)
        (root / "task_success_rates.csv").write_text(
            "Task,Description,Success Rate (%)\n"
            f"libero_spatial_0,synthetic,{percent:.2f}\n",
            encoding="utf-8",
        )
        (root / "summary.csv").write_text(
            "synthetic\n"
            ",libero_spatial,Overall\n"
            f"Success Rate (%),{percent:.2f},{percent:.2f}\n"
            "Average Time (s),1.00,1.00\n"
            "Max Time (s),1.00,1.00\n",
            encoding="utf-8",
        )

    def _fixture(self, base: Path):
        reference_root = base / "reference"
        candidate_root = base / "candidate"
        reference_anchor = self._anchor(reference_root, "reference", "reference-run")
        candidate_anchor = self._anchor(candidate_root, "candidate", "candidate-run")
        self._write_run(reference_root, reference_anchor)
        self._write_run(candidate_root, candidate_anchor)
        return reference_root, candidate_root, reference_anchor, candidate_anchor

    def _reseal_trace_receipt(self, root: Path) -> None:
        receipt_path = root / "trace_receipts/libero_spatial/task00.json"
        receipt = json.loads(receipt_path.read_text())
        rows = []
        for trial_idx in range(self.scope.trials_per_task):
            relative = f"traces/libero_spatial/task00/trial{trial_idx:03d}.json"
            path = root / relative
            rows.append(
                {
                    "trial_idx": trial_idx,
                    "path": relative,
                    "sha256": _sha(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        receipt["traces"] = rows
        receipt["tree_sha256"] = AUDIT._tree_sha256(rows)
        _write_json(receipt_path, receipt)

    def _config_gate_fixture(
        self,
        base: Path,
        *,
        locked_mutation=None,
        asset_root: Path | None = None,
    ):
        def make_source(root, paths):
            for relative in paths:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    f"synthetic source file: {relative}\n", encoding="utf-8"
                )
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "G0 Test"],
                check=True,
            )
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-q", "-m", "fixture"],
                check=True,
            )
            return subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        official_root = base / "official"
        instrumentation_root = base / "instrumentation"
        official_commit = make_source(official_root, AUDIT._OFFICIAL_CRITICAL_PATHS)
        instrumentation_commit = make_source(
            instrumentation_root, AUDIT._INSTRUMENTATION_CRITICAL_PATHS
        )

        artifact_root = base / "artifacts"
        raw_log_root = base / "raw-logs"
        contract_dir = base / "contract"
        control_dir = base / "control"
        artifact_root.mkdir()
        raw_log_root.mkdir()
        contract_dir.mkdir()
        control_dir.mkdir()
        asset_root = base if asset_root is None else asset_root
        asset_root.mkdir(parents=True, exist_ok=True)
        checkpoint = asset_root / "checkpoint.pt"
        dataset_stats = asset_root / "dataset-stats.json"
        checkpoint.write_bytes(b"checkpoint\n")
        dataset_stats.write_bytes(b"{}\n")
        approved = {
            "checkpoint": {"path": str(checkpoint)},
            "dataset_stats": {"path": str(dataset_stats)},
        }
        canonical_live = {
            "gpu_id": 0,
            "seed": 42,
            "output_dir": str(artifact_root),
            "ckpt": str(checkpoint),
            "mixed_precision": "bf16",
            "model": {"variant": "fastwam", "static_width": 32},
            "EVALUATION": {
                "task_suite_name": "libero_spatial",
                "task_id": 0,
                "output_dir": str(artifact_root),
                "dataset_stats_path": str(dataset_stats),
                "num_trials": 50,
                "env_num": 1,
                "num_steps_wait": 30,
                "replan_steps": 10,
                "binarize_gripper": True,
                "use_action_ensembler": False,
                "visualize_future_video": False,
                "action_horizon": 32,
            },
        }
        locked = copy.deepcopy(canonical_live)
        if locked_mutation is not None:
            locked = locked_mutation(locked)
        locked_path = contract_dir / "resolved-config.yaml"
        runtime_locked_path = control_dir / "resolved-config.yaml"
        _write_json(locked_path, locked)
        runtime_locked_path.write_bytes(locked_path.read_bytes())
        locked_sha = _sha(locked_path)

        gpu_ids = [0, 1]
        scheduled = [
            {
                "process_id": f"{suite}/task{task_id:02d}",
                "task_suite": suite,
                "task_id": task_id,
                "global_seed": 42,
                "python_hash_seed": 42,
            }
            for suite in AUDIT.SUITES
            for task_id in range(10)
        ]
        runtime_environment = {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "DIFFSYNTH_DOWNLOAD_SOURCE": "modelscope",
            "DIFFSYNTH_MODEL_BASE_PATH": str(base / "model-cache"),
            "DIFFSYNTH_SKIP_DOWNLOAD": "true",
        }
        preregistration = {
            "launch": {"working_directory": str(instrumentation_root)},
            "runtime_environment": runtime_environment,
            "runtime_lock": {"hydra": "1.3.2", "omegaconf": "2.3.0"},
        }
        schedule_document = {
            "python_hash_seed": 42,
            "task_processes": scheduled,
        }
        prereg_path = control_dir / "preregistration.json"
        start_path = control_dir / "runtime-start.json"
        schedule_path = control_dir / "seed-schedule.json"
        prereg_sha, _ = _write_json(prereg_path, preregistration)
        start_sha, _ = _write_json(start_path, {})
        schedule_sha, _ = _write_json(schedule_path, schedule_document)

        manager_rows = []
        for index, suite in enumerate(AUDIT.SUITES):
            for task_id in range(10):
                process_id = f"{suite}/task{task_id:02d}"
                gpu_id = index % len(gpu_ids) if task_id == 0 else (index * 10 + task_id) % len(gpu_ids)
                command = [
                    str(Path(sys.executable).resolve()),
                    str(instrumentation_root / "scripts/run_mf_wam_g0_traced.py"),
                    "task=libero_uncond_2cam224_1e-4",
                    f"ckpt={checkpoint}",
                    f"gpu_id={gpu_id}",
                    "seed=42",
                    f"output_dir={artifact_root}",
                    f"EVALUATION.task_suite_name={suite}",
                    f"EVALUATION.task_id={task_id}",
                    f"EVALUATION.output_dir={artifact_root}",
                    f"EVALUATION.dataset_stats_path={dataset_stats}",
                    "EVALUATION.num_trials=50",
                    "EVALUATION.env_num=1",
                    "EVALUATION.num_steps_wait=30",
                    "EVALUATION.replan_steps=10",
                    "EVALUATION.binarize_gripper=true",
                    "EVALUATION.use_action_ensembler=false",
                    "EVALUATION.visualize_future_video=false",
                    "EVALUATION.action_horizon=32",
                ]
                environment = {
                    **AUDIT._FIXED_WORKER_ENVIRONMENT,
                    "CUDA_VISIBLE_DEVICES": str(gpu_id),
                    "DIFFSYNTH_DOWNLOAD_SOURCE": "modelscope",
                    "DIFFSYNTH_MODEL_BASE_PATH": str(base / "model-cache"),
                    "DIFFSYNTH_SKIP_DOWNLOAD": "true",
                    "LOCAL_RANK": "0",
                    "MF_WAM_G0_PREREG_PATH": str(prereg_path),
                    "MF_WAM_G0_PREREG_SHA256": prereg_sha,
                    "MF_WAM_G0_RESOLVED_CONFIG_PATH": str(runtime_locked_path),
                    "MF_WAM_G0_RESOLVED_CONFIG_SHA256": locked_sha,
                    "MF_WAM_G0_RUN_ID": "formal-run",
                    "MF_WAM_G0_RUNTIME_START_PATH": str(start_path),
                    "MF_WAM_G0_RUNTIME_START_SHA256": start_sha,
                    "MF_WAM_G0_SEED_SCHEDULE_PATH": str(schedule_path),
                    "MF_WAM_G0_SEED_SCHEDULE_SHA256": schedule_sha,
                    "MF_WAM_INSTRUMENTATION_COMMIT": instrumentation_commit,
                    "MF_WAM_OFFICIAL_ROOT": str(official_root),
                    "MF_WAM_OFFICIAL_COMMIT": official_commit,
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
                    "PYTHONHASHSEED": "42",
                    "RANK": "0",
                    "WORLD_SIZE": "1",
                }
                command_sha = AUDIT._canonical_sha256(command)
                environment_sha = AUDIT._canonical_sha256(environment)
                status = {
                    "schema_version": 1,
                    "kind": "mf_wam_g0_manager_task_status",
                    "run_id": "formal-run",
                    "process_id": process_id,
                    "task_suite": suite,
                    "task_id": task_id,
                    "gpu_id": gpu_id,
                    "state": "SUCCEEDED",
                    "launched_at": "2026-08-02T00:00:00+00:00",
                    "completed_at": "2026-08-02T00:01:00+00:00",
                    "exit_code": 0,
                    "complete": True,
                    "failure_reason": None,
                    "command_argv": command,
                    "command_sha256": command_sha,
                    "environment_bindings": environment,
                    "environment_sha256": environment_sha,
                    "log": {},
                    "canonical_result": {},
                    "trace_receipt": {},
                    "raw_result": {},
                }
                status_relative = f"status/{suite}/task{task_id:02d}.json"
                status_sha, status_size = _write_json(raw_log_root / status_relative, status)
                manager_rows.append(
                    {
                        "process_id": process_id,
                        "task_suite": suite,
                        "task_id": task_id,
                        "gpu_id": gpu_id,
                        "state": "SUCCEEDED",
                        "launched_at": "2026-08-02T00:00:00+00:00",
                        "completed_at": "2026-08-02T00:01:00+00:00",
                        "exit_code": 0,
                        "complete": True,
                        "failure_reason": None,
                        "command_sha256": command_sha,
                        "environment_sha256": environment_sha,
                        "log_path": f"logs/{suite}/task{task_id:02d}.log",
                        "log_sha256": "1" * 64,
                        "log_size_bytes": 1,
                        "status_path": status_relative,
                        "status_sha256": status_sha,
                        "status_size_bytes": status_size,
                        "result_path": f"results/{suite}/task{task_id:02d}.json",
                        "result_sha256": "2" * 64,
                        "result_size_bytes": 1,
                        "trace_receipt_path": f"trace_receipts/{suite}/task{task_id:02d}.json",
                        "trace_receipt_sha256": "3" * 64,
                        "trace_receipt_size_bytes": 1,
                        "trace_tree_sha256": "4" * 64,
                        "episode_count": 50,
                        "raw_result_source_path": f"{suite}/gpu{gpu_id}_task{task_id}_results.json",
                        "raw_result_archive_path": f"official/{suite}/gpu{gpu_id}_task{task_id}_results.json",
                        "raw_result_sha256": "5" * 64,
                        "raw_result_size_bytes": 1,
                    }
                )
        manager = {
            "schema_version": 1,
            "kind": "mf_wam_g0_manager_terminal_manifest",
            "run_id": "formal-run",
            "completed_at": "2026-08-02T00:02:00+00:00",
            "manager_exit_code": 0,
            "artifact_root": str(artifact_root),
            "raw_log_root": str(raw_log_root),
            "gpu_ids": gpu_ids,
            "upstream_bindings": {
                "preregistration_file_sha256": prereg_sha,
                "runtime_start_file_sha256": start_sha,
                "seed_schedule_file_sha256": schedule_sha,
                "resolved_config_sha256": locked_sha,
                "official_commit": official_commit,
                "instrumentation_commit": instrumentation_commit,
                "python_hash_seed": 42,
            },
            "canonical_input_file_count": 2080,
            "canonical_input_tree_sha256": "6" * 64,
            "task_processes": manager_rows,
        }
        manager_sha, _ = _write_json(raw_log_root / "manager_terminal.json", manager)
        anchor = {
            "run_id": "formal-run",
            "artifact_root": str(artifact_root),
            "raw_log_root": str(raw_log_root),
            "manager_manifest_sha256": manager_sha,
            "resolved_config_sha256": locked_sha,
            "terminal_canonical_sha256": "7" * 64,
            "fastwam_commit": official_commit,
            "instrumentation_commit": instrumentation_commit,
        }
        contract = {
            "preregistration": preregistration,
            "seed_schedule": schedule_document,
            "terminal": {},
            "digests": {
                "preregistration_file_sha256": prereg_sha,
                "runtime_start_file_sha256": start_sha,
                "seed_schedule_file_sha256": schedule_sha,
            },
        }

        compose_roots = []

        def compose_config(_root, arguments):
            compose_roots.append(Path(_root))
            overrides = dict(argument.split("=", 1) for argument in arguments)
            live = copy.deepcopy(canonical_live)
            live["gpu_id"] = int(overrides["gpu_id"])
            live["seed"] = int(overrides["seed"])
            live["output_dir"] = overrides["output_dir"]
            live["ckpt"] = overrides["ckpt"]
            evaluation = live["EVALUATION"]
            evaluation["task_suite_name"] = overrides["EVALUATION.task_suite_name"]
            evaluation["task_id"] = int(overrides["EVALUATION.task_id"])
            evaluation["output_dir"] = overrides["EVALUATION.output_dir"]
            evaluation["dataset_stats_path"] = overrides[
                "EVALUATION.dataset_stats_path"
            ]
            for key in (
                "num_trials",
                "env_num",
                "num_steps_wait",
                "replan_steps",
                "action_horizon",
            ):
                evaluation[key] = int(overrides[f"EVALUATION.{key}"])
            for key in (
                "binarize_gripper",
                "use_action_ensembler",
                "visualize_future_video",
            ):
                evaluation[key] = overrides[f"EVALUATION.{key}"] == "true"
            return live

        def load_locked(path):
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload, _sha(path), path.stat().st_size

        return {
            "anchor": anchor,
            "contract": contract,
            "contract_dir": contract_dir,
            "approved": approved,
            "compose": compose_config,
            "load_locked": load_locked,
            "normalize": lambda value: copy.deepcopy(value),
            "official_root": official_root,
            "compose_roots": compose_roots,
        }

    def test_specialized_config_gate_accepts_only_three_dynamic_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._config_gate_fixture(Path(directory))
            audit = AUDIT._audit_resolved_config_gate(
                anchor=fixture["anchor"],
                contract=fixture["contract"],
                contract_dir=fixture["contract_dir"],
                approved=fixture["approved"],
                compose_config=fixture["compose"],
                load_locked_config=fixture["load_locked"],
                normalize_config=fixture["normalize"],
            )
        self.assertEqual(audit["process_count"], 40)
        self.assertEqual(audit["status_file_count"], 40)
        self.assertRegex(audit["static_projection_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(audit["official_source"]["clean"], True)
        self.assertEqual(
            len(audit["official_source"]["critical_files"]),
            len(AUDIT._OFFICIAL_CRITICAL_PATHS),
        )
        self.assertEqual(len(set(fixture["compose_roots"])), 1)
        self.assertNotEqual(fixture["compose_roots"][0], fixture["official_root"])
        self.assertFalse(fixture["compose_roots"][0].exists())
        episode_identity = {
            "official_source": audit["official_source"],
            "instrumentation_source": audit["instrumentation_source"],
        }
        report = {"episode_pair_identities": {("libero_spatial", 0, 0): episode_identity}}
        AUDIT._validate_trace_source_bindings(report, audit, role="reference")
        tampered = copy.deepcopy(report)
        tampered["episode_pair_identities"][("libero_spatial", 0, 0)][
            "official_source"
        ]["root"] = "/attacker/official"
        with self.assertRaisesRegex(
            AUDIT.SpecializedAuditError, "manager-bound checkout"
        ):
            AUDIT._validate_trace_source_bindings(tampered, audit, role="reference")

    def test_distinct_run_roots_share_only_the_paired_static_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset_root = root / "approved-assets"
            reference = self._config_gate_fixture(
                root / "reference", asset_root=asset_root
            )
            candidate = self._config_gate_fixture(
                root / "candidate", asset_root=asset_root
            )
            audits = [
                AUDIT._audit_resolved_config_gate(
                    anchor=fixture["anchor"],
                    contract=fixture["contract"],
                    contract_dir=fixture["contract_dir"],
                    approved=fixture["approved"],
                    compose_config=fixture["compose"],
                    load_locked_config=fixture["load_locked"],
                    normalize_config=fixture["normalize"],
                )
                for fixture in (reference, candidate)
            ]
        self.assertNotEqual(
            audits[0]["locked_resolved_config_sha256"],
            audits[1]["locked_resolved_config_sha256"],
        )
        self.assertNotEqual(
            audits[0]["static_projection_sha256"],
            audits[1]["static_projection_sha256"],
        )
        self.assertEqual(
            audits[0]["paired_static_projection_sha256"],
            audits[1]["paired_static_projection_sha256"],
        )

    def test_synthetic_structural_config_is_rejected_by_specialized_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._config_gate_fixture(
                Path(directory), locked_mutation=lambda _locked: {"synthetic": True}
            )
            with self.assertRaisesRegex(
                AUDIT.SpecializedAuditError, "dynamic overlay"
            ):
                AUDIT._audit_resolved_config_gate(
                    anchor=fixture["anchor"],
                    contract=fixture["contract"],
                    contract_dir=fixture["contract_dir"],
                    approved=fixture["approved"],
                    compose_config=fixture["compose"],
                    load_locked_config=fixture["load_locked"],
                    normalize_config=fixture["normalize"],
                )

    def test_resealed_static_config_drift_is_rejected(self) -> None:
        def drift(locked):
            locked["mixed_precision"] = "fp16"
            return locked

        with tempfile.TemporaryDirectory() as directory:
            fixture = self._config_gate_fixture(Path(directory), locked_mutation=drift)
            with self.assertRaisesRegex(
                AUDIT.SpecializedAuditError, "static projection differs"
            ):
                AUDIT._audit_resolved_config_gate(
                    anchor=fixture["anchor"],
                    contract=fixture["contract"],
                    contract_dir=fixture["contract_dir"],
                    approved=fixture["approved"],
                    compose_config=fixture["compose"],
                    load_locked_config=fixture["load_locked"],
                    normalize_config=fixture["normalize"],
                )

    def test_fully_resealed_static_launch_override_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._config_gate_fixture(Path(directory))
            raw_log_root = Path(fixture["anchor"]["raw_log_root"])
            status_path = raw_log_root / "status/libero_spatial/task00.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            command = list(status["command_argv"])
            command[command.index("EVALUATION.replan_steps=10")] = (
                "EVALUATION.replan_steps=11"
            )
            command_sha = AUDIT._canonical_sha256(command)
            status["command_argv"] = command
            status["command_sha256"] = command_sha
            status_sha, status_size = _write_json(status_path, status)

            manager_path = raw_log_root / "manager_terminal.json"
            manager = json.loads(manager_path.read_text(encoding="utf-8"))
            process = manager["task_processes"][0]
            self.assertEqual(process["process_id"], "libero_spatial/task00")
            process["command_sha256"] = command_sha
            process["status_sha256"] = status_sha
            process["status_size_bytes"] = status_size
            manager_sha, _ = _write_json(manager_path, manager)
            fixture["anchor"]["manager_manifest_sha256"] = manager_sha

            with self.assertRaisesRegex(
                AUDIT.SpecializedAuditError, "formal Hydra override values"
            ):
                AUDIT._audit_resolved_config_gate(
                    anchor=fixture["anchor"],
                    contract=fixture["contract"],
                    contract_dir=fixture["contract_dir"],
                    approved=fixture["approved"],
                    compose_config=fixture["compose"],
                    load_locked_config=fixture["load_locked"],
                    normalize_config=fixture["normalize"],
                )

    def test_official_source_drift_is_rejected_before_compose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._config_gate_fixture(Path(directory))
            source = (
                fixture["official_root"]
                / "configs/task/libero_uncond_2cam224_1e-4.yaml"
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(fixture["official_root"]),
                    "update-index",
                    "--assume-unchanged",
                    "configs/task/libero_uncond_2cam224_1e-4.yaml",
                ],
                check=True,
            )
            source.write_text("drifted\n", encoding="utf-8")
            with self.assertRaisesRegex(
                AUDIT.SpecializedAuditError, "index contains|HEAD blob"
            ):
                AUDIT._audit_resolved_config_gate(
                    anchor=fixture["anchor"],
                    contract=fixture["contract"],
                    contract_dir=fixture["contract_dir"],
                    approved=fixture["approved"],
                    compose_config=fixture["compose"],
                    load_locked_config=fixture["load_locked"],
                    normalize_config=fixture["normalize"],
                )

    def test_specialized_gate_rejects_info_exclude_ignored_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._config_gate_fixture(Path(directory))
            git_info = fixture["official_root"] / ".git/info"
            (git_info / "exclude").write_text("*.pyd\n", encoding="utf-8")
            ignored = (
                fixture["official_root"]
                / "experiments/libero/eval_libero_single.cp310-win_amd64.pyd"
            )
            ignored.write_bytes(b"ignored extension\n")
            with self.assertRaisesRegex(
                AUDIT.SpecializedAuditError, "gitignored artifacts"
            ):
                AUDIT._audit_resolved_config_gate(
                    anchor=fixture["anchor"],
                    contract=fixture["contract"],
                    contract_dir=fixture["contract_dir"],
                    approved=fixture["approved"],
                    compose_config=fixture["compose"],
                    load_locked_config=fixture["load_locked"],
                    normalize_config=fixture["normalize"],
                )

    def test_specialized_hydra_dependency_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"hydra": None}
        ):
            with self.assertRaisesRegex(
                AUDIT.SpecializedAuditError, "hydra-core is required"
            ):
                AUDIT._compose_official_config(Path(directory), [])

    def test_formal_layout_cardinality_is_exact(self) -> None:
        paths = AUDIT.expected_paths(AUDIT.FORMAL_SCOPE)
        self.assertEqual(len(paths["results"]), 40)
        self.assertEqual(len(paths["trace_receipts"]), 40)
        self.assertEqual(len(paths["traces"]), 2000)
        self.assertEqual(AUDIT.FORMAL_SCOPE.episode_count, 2000)

    def test_small_pair_recomputes_exact_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            report = AUDIT.audit_pair(
                reference_root=fixture[0],
                candidate_root=fixture[1],
                reference_anchor=fixture[2],
                candidate_anchor=fixture[3],
                scope=self.scope,
            )
        self.assertTrue(report["outcome_parity_pass"])
        self.assertEqual(report["overall_success_delta"], {
            "estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0
        })

    def test_same_run_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            fixture[3]["run_id"] = fixture[2]["run_id"]
            with self.assertRaisesRegex(AUDIT.SpecializedAuditError, "run_id"):
                AUDIT.audit_pair(
                    reference_root=fixture[0], candidate_root=fixture[1],
                    reference_anchor=fixture[2], candidate_anchor=fixture[3],
                    scope=self.scope,
                )

    def test_legacy_action_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            path = fixture[1] / "traces/libero_spatial/task00/trial000.json"
            payload = json.loads(path.read_text())
            payload["records"][0]["raw_action_chunk"] = [[0.0] * 7] * 10
            _write_json(path, payload)
            with self.assertRaisesRegex(AUDIT.SpecializedAuditError, "fields mismatch"):
                AUDIT.audit_pair(
                    reference_root=fixture[0], candidate_root=fixture[1],
                    reference_anchor=fixture[2], candidate_anchor=fixture[3],
                    scope=self.scope,
                )

    def test_single_trace_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            path = fixture[1] / "traces/libero_spatial/task00/trial001.json"
            payload = json.loads(path.read_text())
            payload["records"][0]["proposed_raw_action_chunk"][0][0] = 999.0
            _write_json(path, payload)
            with self.assertRaisesRegex(AUDIT.SpecializedAuditError, "action transform"):
                AUDIT.audit_pair(
                    reference_root=fixture[0], candidate_root=fixture[1],
                    reference_anchor=fixture[2], candidate_anchor=fixture[3],
                    scope=self.scope,
                )

    def test_result_semantic_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            path = fixture[1] / "results/libero_spatial/task00.json"
            payload = json.loads(path.read_text())
            payload["successes"] = 2
            _write_json(path, payload)
            with self.assertRaisesRegex(AUDIT.SpecializedAuditError, "result semantics"):
                AUDIT.audit_pair(
                    reference_root=fixture[0], candidate_root=fixture[1],
                    reference_anchor=fixture[2], candidate_anchor=fixture[3],
                    scope=self.scope,
                )

    def test_missing_and_extra_canonical_files_are_rejected(self) -> None:
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = self._fixture(Path(directory))
                if mutation == "missing":
                    os.unlink(fixture[1] / "traces/libero_spatial/task00/trial001.json")
                else:
                    _write_json(fixture[1] / "traces/libero_spatial/task00/extra.json", {})
                with self.assertRaisesRegex(AUDIT.SpecializedAuditError, "layout mismatch"):
                    AUDIT.audit_pair(
                        reference_root=fixture[0], candidate_root=fixture[1],
                        reference_anchor=fixture[2], candidate_anchor=fixture[3],
                        scope=self.scope,
                    )

    def test_symlink_and_hardlink_are_rejected(self) -> None:
        for mutation in ("symlink", "hardlink"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = self._fixture(Path(directory))
                path = fixture[1] / "traces/libero_spatial/task00/trial001.json"
                backup = fixture[1] / "backup.json"
                backup.write_bytes(path.read_bytes())
                os.unlink(path)
                if mutation == "symlink":
                    path.symlink_to(backup)
                else:
                    os.link(backup, path)
                with self.assertRaises(AUDIT.SpecializedAuditError):
                    AUDIT.audit_pair(
                        reference_root=fixture[0], candidate_root=fixture[1],
                        reference_anchor=fixture[2], candidate_anchor=fixture[3],
                        scope=self.scope,
                    )

    def test_nonfinite_exponent_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            path = fixture[1] / "traces/libero_spatial/task00/trial000.json"
            text = path.read_text().replace("0.0", "1e999", 1)
            path.write_text(text)
            with self.assertRaisesRegex(AUDIT.SpecializedAuditError, "non-finite"):
                AUDIT.audit_pair(
                    reference_root=fixture[0], candidate_root=fixture[1],
                    reference_anchor=fixture[2], candidate_anchor=fixture[3],
                    scope=self.scope,
                )

    def test_common_anchor_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            fixture[3]["resolved_config_sha256"] = "a" * 64
            with self.assertRaisesRegex(
                AUDIT.SpecializedAuditError,
                "locked field|trace upstream digest",
            ):
                AUDIT.audit_pair(
                    reference_root=fixture[0], candidate_root=fixture[1],
                    reference_anchor=fixture[2], candidate_anchor=fixture[3],
                    scope=self.scope,
                )

    def test_summary_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            path = fixture[1] / "summary.json"
            payload = json.loads(path.read_text())
            payload["overall"]["average_success_rate"] = 100.0
            _write_json(path, payload)
            with self.assertRaisesRegex(AUDIT.SpecializedAuditError, "overall"):
                AUDIT.audit_pair(
                    reference_root=fixture[0], candidate_root=fixture[1],
                    reference_anchor=fixture[2], candidate_anchor=fixture[3],
                    scope=self.scope,
                )

    def test_exclusive_receipt_publish_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            digest, size = AUDIT._write_exclusive_json(
                path, {"formal_training_allowed": False, "status": "test"}
            )
            self.assertEqual(digest, _sha(path))
            self.assertEqual(size, path.stat().st_size)
            with self.assertRaisesRegex(AUDIT.SpecializedAuditError, "replace"):
                AUDIT._write_exclusive_json(path, {"different": True})

    def test_approved_assets_manifest_is_locked(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs/validation/mf_wam_g0_approved_assets.json"
        payload, digest = AUDIT._validate_approved_assets(
            path, _sha(path), live_readback=False
        )
        self.assertEqual(payload["source"]["fastwam_commit"], AUDIT.EXPECTED_FASTWAM_COMMIT)
        self.assertEqual(digest, _sha(path))

    def test_gate_policy_specialized_contract_is_exact(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs/validation/mf_wam_gates.json"
        policy = json.loads(path.read_text())
        policy_id, digest = AUDIT._policy_identity(policy)
        self.assertEqual(policy_id, "MF-WAM-G0-G3-2026-08-02-v3")
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_gate_evidence_must_equal_recomputation(self) -> None:
        evidence = {
            "evidence_complete": True,
            "episode_count": 2,
            "suite_count": 1,
            "tasks_per_suite": 1,
            "trials_per_task": 2,
            "missing_trace_count": 0,
            "non_finite_count": 0,
            "missing_seed_binding_count": 0,
            "first_replan_env_step": 30,
            "minimum_trace_records_per_episode": 1,
            "artifact_bindings_complete": True,
            "paired_episode_identity_complete": True,
            "overall_success_delta": {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
            "suite_success_deltas": {
                "libero_spatial": {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
            },
        }
        digest = AUDIT._validate_gate_evidence(
            evidence,
            scope=self.scope,
            overall=evidence["overall_success_delta"],
            suites=evidence["suite_success_deltas"],
        )
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        evidence["episode_count"] = 3
        with self.assertRaisesRegex(AUDIT.SpecializedAuditError, "episode_count"):
            AUDIT._validate_gate_evidence(
                evidence,
                scope=self.scope,
                overall={"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
                suites={
                    "libero_spatial": {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
                },
            )

    def test_resealed_raw_action_transform_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            path = fixture[1] / "traces/libero_spatial/task00/trial000.json"
            payload = json.loads(path.read_text())
            payload["records"][0]["proposed_raw_action_chunk"][0][0] = 999.0
            _write_json(path, payload)
            self._reseal_trace_receipt(fixture[1])
            with self.assertRaisesRegex(AUDIT.SpecializedAuditError, "action transform"):
                AUDIT.audit_pair(
                    reference_root=fixture[0],
                    candidate_root=fixture[1],
                    reference_anchor=fixture[2],
                    candidate_anchor=fixture[3],
                    scope=self.scope,
                )

    def test_resealed_environment_gripper_transform_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            path = fixture[1] / "traces/libero_spatial/task00/trial000.json"
            payload = json.loads(path.read_text())
            payload["records"][0]["proposed_env_action_chunk"][7][-1] *= -1
            _write_json(path, payload)
            self._reseal_trace_receipt(fixture[1])
            with self.assertRaisesRegex(AUDIT.SpecializedAuditError, "action transform"):
                AUDIT.audit_pair(
                    reference_root=fixture[0],
                    candidate_root=fixture[1],
                    reference_anchor=fixture[2],
                    candidate_anchor=fixture[3],
                    scope=self.scope,
                )

    def test_fully_resealed_seed_substitution_differs_from_schedule(self) -> None:
        anchor = self._anchor(Path("/tmp/synthetic"), "candidate", "candidate-run")
        trace = self._trace(anchor, 0, True)
        expected_process = dict(trace["metadata"]["seed_schedule_process"])
        process = trace["metadata"]["seed_schedule_process"]
        for key in ("global_seed", "environment_seed", "policy_seed", "python_hash_seed"):
            process[key] = 999
        contract = trace["metadata"]["seed_contract"]
        for key in (
            "task_seed",
            "effective_process_seed",
            "environment_seed",
            "policy_seed",
        ):
            contract[key] = 999
        trace["records"][0]["policy_seed"] = 999
        with self.assertRaisesRegex(AUDIT.SpecializedAuditError, "seed-schedule"):
            AUDIT._validate_trace(
                trace,
                run_anchor=anchor,
                suite="libero_spatial",
                task_id=0,
                trial_idx=0,
                scope=self.scope,
                expected_seed_process=expected_process,
            )

    def test_resealed_paired_initial_state_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            path = fixture[1] / "traces/libero_spatial/task00/trial001.json"
            payload = json.loads(path.read_text())
            payload["metadata"]["initial_state_sha256"] = "0" * 63 + "1"
            _write_json(path, payload)
            self._reseal_trace_receipt(fixture[1])
            with self.assertRaisesRegex(AUDIT.SpecializedAuditError, "paired episode identity"):
                AUDIT.audit_pair(
                    reference_root=fixture[0],
                    candidate_root=fixture[1],
                    reference_anchor=fixture[2],
                    candidate_anchor=fixture[3],
                    scope=self.scope,
                )

    def test_resealed_paired_task_description_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            for trial_idx in range(self.scope.trials_per_task):
                path = fixture[1] / f"traces/libero_spatial/task00/trial{trial_idx:03d}.json"
                payload = json.loads(path.read_text())
                payload["metadata"]["task_description"] = "forged task"
                _write_json(path, payload)
            result_path = fixture[1] / "results/libero_spatial/task00.json"
            result = json.loads(result_path.read_text())
            result["task_description"] = "forged task"
            result_sha, result_size = _write_json(result_path, result)
            receipt_path = fixture[1] / "trace_receipts/libero_spatial/task00.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["official_result"] = {
                "path": "results/libero_spatial/task00.json",
                "sha256": result_sha,
                "size_bytes": result_size,
            }
            _write_json(receipt_path, receipt)
            self._reseal_trace_receipt(fixture[1])
            with self.assertRaisesRegex(AUDIT.SpecializedAuditError, "paired episode identity"):
                AUDIT.audit_pair(
                    reference_root=fixture[0],
                    candidate_root=fixture[1],
                    reference_anchor=fixture[2],
                    candidate_anchor=fixture[3],
                    scope=self.scope,
                )

    def test_external_anchor_requires_exact_semantic_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchor.json"
            bindings = {"run_id": "candidate-run", "terminal_sha256": "a" * 64}
            anchor = {
                "schema_version": 1,
                "kind": "mf_wam_g0_external_anchor",
                "anchor_type": "source_commit",
                "anchor_id": AUDIT.EXPECTED_FASTWAM_COMMIT,
                "bindings": bindings,
            }
            digest, _ = _write_json(path, anchor)
            observed = AUDIT._validate_external_anchor(
                "source_commit",
                AUDIT.EXPECTED_FASTWAM_COMMIT,
                path,
                digest,
                expected_anchor_id=AUDIT.EXPECTED_FASTWAM_COMMIT,
                expected_bindings=bindings,
            )
            self.assertEqual(set(observed), {"anchor_type", "anchor_id", "artifact_sha256"})

            forged = dict(anchor)
            forged["bindings"] = {"run_id": "other", "terminal_sha256": "a" * 64}
            forged_sha, _ = _write_json(Path(directory) / "forged.json", forged)
            with self.assertRaisesRegex(AUDIT.SpecializedAuditError, "semantics"):
                AUDIT._validate_external_anchor(
                    "source_commit",
                    AUDIT.EXPECTED_FASTWAM_COMMIT,
                    Path(directory) / "forged.json",
                    forged_sha,
                    expected_anchor_id=AUDIT.EXPECTED_FASTWAM_COMMIT,
                    expected_bindings=bindings,
                )

    def test_gate_evidence_incomplete_is_rejected(self) -> None:
        evidence = {
            "evidence_complete": False,
            "episode_count": 2,
            "suite_count": 1,
            "tasks_per_suite": 1,
            "trials_per_task": 2,
            "missing_trace_count": 0,
            "non_finite_count": 0,
            "missing_seed_binding_count": 0,
            "first_replan_env_step": 30,
            "minimum_trace_records_per_episode": 1,
            "artifact_bindings_complete": True,
            "paired_episode_identity_complete": True,
            "overall_success_delta": {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
            "suite_success_deltas": {
                "libero_spatial": {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
            },
        }
        with self.assertRaisesRegex(AUDIT.SpecializedAuditError, "evidence_complete"):
            AUDIT._validate_gate_evidence(
                evidence,
                scope=self.scope,
                overall=evidence["overall_success_delta"],
                suites=evidence["suite_success_deltas"],
            )


if __name__ == "__main__":
    unittest.main()
