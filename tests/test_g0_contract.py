from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.validation.g0_contract import (
    DATA_TREE_ALGORITHM,
    EXPECTED_DATA_FILES,
    EXPECTED_EPISODES,
    EXPECTED_TASKS,
    OFFICIAL_FASTWAM_COMMIT,
    OFFICIAL_LIBERO_COMMIT,
    SUITES,
    ContractError,
    build_data_inventory,
    build_preregistration,
    build_seed_schedule,
    canonical_json_bytes,
    canonical_json_sha256,
    load_json_strict,
    validate_contract_chain,
    validate_data_inventory,
    validate_preregistration,
    validate_runtime_start,
    validate_seed_schedule,
    validate_terminal_receipt,
    write_canonical_json,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _tree_digest(files: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value["path"].encode("utf-8")):
        digest.update(f"{item['sha256']}  {item['path']}\n".encode("utf-8"))
    return digest.hexdigest()


class G0ContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "libero"
        self.root.mkdir()
        self.model_cache_root = Path(self.temporary.name) / "model-cache"
        self.model_cache_root.mkdir()
        self.artifact_root = Path(self.temporary.name) / "MFWAM-G0-TEST-001"
        self.artifact_root.mkdir()
        tasks = []
        for suite in SUITES:
            for task_id in range(10):
                name = f"{suite}_task_{task_id:02d}"
                bddl = f"bddl_files/{suite}/{name}.bddl"
                init = f"init_files/{suite}/{name}.pruned_init"
                for relative, content in (
                    (bddl, f"bddl:{suite}:{task_id}\n".encode("utf-8")),
                    (init, f"init:{suite}:{task_id}\n".encode("utf-8")),
                ):
                    path = self.root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                tasks.append(
                    {
                        "task_suite": suite,
                        "task_id": task_id,
                        "task_name": name,
                        "bddl_path": bddl,
                        "init_state_path": init,
                        "trial_count": 50,
                    }
                )
        self.task_map = {
            "schema_version": 1,
            "kind": "mf_wam_g0_task_map",
            "tasks": tasks,
        }
        self.inventory = build_data_inventory(
            self.root,
            self.task_map,
            dataset_id="libero-40",
            revision=OFFICIAL_LIBERO_COMMIT,
        )
        self.schedule = build_seed_schedule(
            self.task_map,
            seed=42,
            python_hash_seed=42,
        )
        self.prereg = build_preregistration(
            self._prereg_spec(),
            data_inventory=self.inventory,
            seed_schedule=self.schedule,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _prereg_spec(self) -> dict:
        image_digest = f"sha256:{'a' * 64}"
        model_cache_files = []
        for index, role in enumerate(
            sorted(
                (
                    "text_encoder_weights",
                    "vae_weights",
                    "tokenizer_config",
                    "tokenizer_json",
                    "tokenizer_special_tokens_map",
                    "tokenizer_model",
                )
            )
        ):
            relative = f"cache/{role}.bin"
            path = self.model_cache_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            content = bytes([index + 1]) * (index + 1)
            path.write_bytes(content)
            model_cache_files.append(
                {
                    "role": role,
                    "path": relative,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size_bytes": len(content),
                }
            )
        model_cache_core = {
            "algorithm": "model-cache-per-file-sha256-v1",
            "file_count": 6,
            "files": model_cache_files,
        }
        model_cache = {
            **model_cache_core,
            "canonical_sha256": canonical_json_sha256(model_cache_core),
        }
        return {
            "run_id": "MFWAM-G0-TEST-001",
            "created_at": "2026-08-02T18:00:00+08:00",
            "iteration_id": "ITER-2026-08-02-003",
            "project_page_id": "3b021e77-89cc-81b9-8238-e3cdbf44dda2",
            "source": {
                "fastwam": {"commit": OFFICIAL_FASTWAM_COMMIT, "clean": True},
                "instrumentation": {"commit": "b" * 40, "clean": True},
                "libero": {"commit": OFFICIAL_LIBERO_COMMIT, "clean": True},
                "auditor": {"commit": "c" * 40, "clean": True},
            },
            "image": {
                "uri": f"registry.example/mfwam@{image_digest}",
                "digest": image_digest,
            },
            "artifacts": {
                "checkpoint": {"sha256": _digest("checkpoint"), "size_bytes": 123},
                "dataset_stats": {"sha256": _digest("stats"), "size_bytes": 45},
                "resolved_config": {"sha256": _digest("config"), "size_bytes": 67},
                "model_cache": model_cache,
            },
            "runtime_lock": {
                "os_release": "Ubuntu 22.04.5 LTS",
                "kernel": "6.8.0-test",
                "python": "3.10.20",
                "torch": "2.7.1+cu128",
                "torchvision": "0.22.1+cu128",
                "cuda_runtime": "12.8",
                "cudnn": "9.7.1",
                "nccl": "2.26.2",
                "triton": "3.3.1",
                "mujoco": "3.3.2",
                "libero": OFFICIAL_LIBERO_COMMIT,
                "robosuite": "1.4.0",
                "bddl": "3.6.0",
                "hydra": "1.3.2",
                "omegaconf": "2.3.0",
                "numpy": "1.26.4",
                "graphics_pack_version": "g0-graphics-pack-v1",
            },
            "runtime_environment": {
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "DIFFSYNTH_DOWNLOAD_SOURCE": "modelscope",
                "DIFFSYNTH_MODEL_BASE_PATH": str(self.model_cache_root),
                "DIFFSYNTH_SKIP_DOWNLOAD": "true",
            },
            "launch": {
                "provider": "alibaba-pai-dlc",
                "worker_count": 1,
                "gpu_count": 2,
                "gpu_model": "NVIDIA-GeForce-RTX-4090",
                "gpu_memory_mib": 24564,
                "driver_version": "570.00",
                "job_spec_sha256": _digest("job-spec"),
                "command_sha256": _digest("command"),
                "sanitized_environment_sha256": _digest("environment"),
                "working_directory": "/opt/mfwam",
            },
            "output": {
                "artifact_root": str(self.artifact_root),
                "overwrite": False,
            },
        }

    def _runtime_start(self) -> dict:
        prereg = self.prereg
        return {
            "schema_version": 1,
            "kind": "mf_wam_g0_runtime_start",
            "phase": "STARTED",
            "receipt_scope": "job-control-plane",
            "run_id": prereg["run_id"],
            "observed_at": "2026-08-02T18:05:00+08:00",
            "preregistration_canonical_sha256": canonical_json_sha256(prereg),
            "job": {
                "provider": prereg["launch"]["provider"],
                "job_id": "dlc-test-001",
                "job_spec_sha256": prereg["launch"]["job_spec_sha256"],
                "pod_uid": "pod-test-001",
                "hostname": "worker-0",
            },
            "source": copy.deepcopy(prereg["source"]),
            "image": copy.deepcopy(prereg["image"]),
            "bindings": {
                "checkpoint_sha256": prereg["artifacts"]["checkpoint"]["sha256"],
                "dataset_stats_sha256": prereg["artifacts"]["dataset_stats"]["sha256"],
                "resolved_config_sha256": prereg["artifacts"]["resolved_config"]["sha256"],
                "data_inventory_canonical_sha256": prereg["data"][
                    "inventory_canonical_sha256"
                ],
                "data_tree_sha256": prereg["data"]["tree_sha256"],
                "seed_schedule_canonical_sha256": prereg["seeds"][
                    "schedule_canonical_sha256"
                ],
                "model_cache_inventory_canonical_sha256": prereg["artifacts"][
                    "model_cache"
                ]["canonical_sha256"],
            },
            "runtime": copy.deepcopy(prereg["runtime_lock"]),
            "runtime_environment": copy.deepcopy(prereg["runtime_environment"]),
            "gpu": {
                "count": 2,
                "model": prereg["launch"]["gpu_model"],
                "memory_mib": prereg["launch"]["gpu_memory_mib"],
                "driver_version": prereg["launch"]["driver_version"],
                "uuids": ["GPU-test-0", "GPU-test-1"],
            },
            "control_process": {
                "working_directory": prereg["launch"]["working_directory"],
                "command_sha256": prereg["launch"]["command_sha256"],
                "sanitized_environment_sha256": prereg["launch"][
                    "sanitized_environment_sha256"
                ],
                "python_hash_seed": prereg["seeds"]["python_hash_seed"],
            },
            "imports": [
                {
                    "module": module,
                    "path": f"/opt/python/{module}.py",
                    "sha256": _digest(f"import:{module}"),
                }
                for module in ("fastwam", "libero", "torch", "mujoco", "robosuite", "numpy")
            ],
            "model_cache_inventory": copy.deepcopy(prereg["artifacts"]["model_cache"]),
        }

    def _terminal(self, start: dict | None = None, *, status: str = "SUCCEEDED") -> dict:
        if start is None:
            start = validate_runtime_start(self._runtime_start(), preregistration=self.prereg)
        succeeded = status == "SUCCEEDED"
        processes: list[dict] = []
        artifact_inventory_reference = None
        aggregates = {
            "task_result_tree_sha256": None,
            "trace_tree_sha256": None,
        }

        def make_terminal(artifact_inventory: dict | None) -> dict:
            return {
                "schema_version": 1,
                "kind": "mf_wam_g0_terminal",
                "phase": "TERMINAL",
                "run_id": self.prereg["run_id"],
                "completed_at": "2026-08-03T18:05:00+08:00",
                "preregistration_canonical_sha256": canonical_json_sha256(self.prereg),
                "runtime_start_canonical_sha256": canonical_json_sha256(start),
                "status": status,
                "failure_reason": None if succeeded else "job failed before task completion",
                "manager_exit_code": 0 if succeeded else 1,
                "scope": {
                    "task_process_count": EXPECTED_TASKS if succeeded else 0,
                    "episode_count": EXPECTED_EPISODES if succeeded else 0,
                    "complete": succeeded,
                },
                "task_processes": processes,
                "artifact_inventory": artifact_inventory,
                "aggregates": aggregates,
            }

        if succeeded:
            inventory_files: list[dict] = []
            result_rows: list[dict] = []
            trace_rows: list[dict] = []

            def write_artifact(relative: str, role: str, value: object) -> dict:
                path = self.artifact_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = value if isinstance(value, bytes) else canonical_json_bytes(value)
                path.write_bytes(raw)
                item = {
                    "path": relative,
                    "role": role,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw),
                }
                inventory_files.append(item)
                return item

            for process in self.schedule["task_processes"]:
                process_id = process["process_id"]
                suite = process["task_suite"]
                task_id = process["task_id"]
                result = write_artifact(
                    f"results/{suite}/task{task_id:02d}.json",
                    "task_result",
                    {
                        "schema_version": 1,
                        "kind": "mf_wam_g0_task_result",
                        "process_id": process_id,
                        "episode_count": 50,
                    },
                )
                result_rows.append(result)
                traces = []
                for trial_idx in range(50):
                    trace = write_artifact(
                        f"traces/{suite}/task{task_id:02d}/trial{trial_idx:03d}.json",
                        "episode_trace",
                        # Deliberately identical bytes make the hardlink-alias test
                        # exercise inode detection instead of a digest mismatch.
                        {"schema_version": 1, "kind": "mf_wam_g0_episode_trace"},
                    )
                    trace_rows.append(trace)
                    traces.append(
                        {
                            "trial_idx": trial_idx,
                            "path": trace["path"],
                            "sha256": trace["sha256"],
                            "size_bytes": trace["size_bytes"],
                        }
                    )
                trace_tree = _tree_digest(traces)
                trace_receipt = write_artifact(
                    f"trace_receipts/{suite}/task{task_id:02d}.json",
                    "task_trace_receipt",
                    {
                        "schema_version": 1,
                        "kind": "mf_wam_g0_task_trace_receipt",
                        "run_id": self.prereg["run_id"],
                        "process_id": process_id,
                        "task_suite": suite,
                        "task_id": task_id,
                        "execution_scope": "one-process-per-task",
                        "world_size": 1,
                        "global_rank": 0,
                        "local_rank": 0,
                        "bindings": {
                            "preregistration_canonical_sha256": canonical_json_sha256(
                                self.prereg
                            ),
                            "runtime_start_canonical_sha256": canonical_json_sha256(start),
                            "seed_schedule_canonical_sha256": canonical_json_sha256(
                                self.schedule
                            ),
                            "resolved_config_sha256": self.prereg["artifacts"][
                                "resolved_config"
                            ]["sha256"],
                            "image_digest": self.prereg["image"]["digest"],
                            "fastwam_commit": self.prereg["source"]["fastwam"][
                                "commit"
                            ],
                            "instrumentation_commit": self.prereg["source"][
                                "instrumentation"
                            ]["commit"],
                        },
                        "seeds": {
                            field: process[field]
                            for field in (
                                "global_seed",
                                "environment_seed",
                                "environment_seed_scope",
                                "policy_seed",
                                "policy_seed_scope",
                                "python_hash_seed",
                                "trial_order",
                                "initial_state_index_rule",
                            )
                        },
                        "official_result": {
                            field: result[field]
                            for field in ("path", "sha256", "size_bytes")
                        },
                        "episode_count": 50,
                        "traces": traces,
                        "tree_sha256": trace_tree,
                    },
                )
                processes.append(
                    {
                        "process_id": process_id,
                        "task_suite": suite,
                        "task_id": task_id,
                        "execution_scope": "one-process-per-task",
                        "world_size": 1,
                        "global_rank": 0,
                        "local_rank": 0,
                        "exit_code": 0,
                        "result_path": result["path"],
                        "result_sha256": result["sha256"],
                        "result_size_bytes": result["size_bytes"],
                        "trace_receipt_path": trace_receipt["path"],
                        "trace_receipt_sha256": trace_receipt["sha256"],
                        "trace_receipt_size_bytes": trace_receipt["size_bytes"],
                        "trace_tree_sha256": trace_tree,
                        "episode_count": 50,
                        "complete": True,
                    }
                )
            aggregates = {
                "task_result_tree_sha256": _tree_digest(result_rows),
                "trace_tree_sha256": _tree_digest(trace_rows),
            }
            write_artifact("summary.csv", "summary_csv", b"metric,value\nstatus,structural\n")
            write_artifact(
                "task_success_rates.csv",
                "task_success_rates_csv",
                b"task,success_rate\nall,0.0\n",
            )
            write_artifact(
                "summary.json",
                "summary_json",
                {"schema_version": 1, "kind": "mf_wam_g0_summary"},
            )
            write_artifact(
                "completion.json",
                "completion_marker",
                {
                    "schema_version": 1,
                    "kind": "mf_wam_g0_completion_marker",
                    "run_id": self.prereg["run_id"],
                    "status": "SUCCEEDED",
                    "task_process_count": EXPECTED_TASKS,
                    "episode_count": EXPECTED_EPISODES,
                    "terminal_core_canonical_sha256": canonical_json_sha256(
                        {
                            key: value
                            for key, value in make_terminal(None).items()
                            if key != "artifact_inventory"
                        }
                    ),
                },
            )
            inventory_files.sort(key=lambda item: item["path"].encode("utf-8"))
            inventory_payload = {
                "schema_version": 1,
                "kind": "mf_wam_g0_terminal_artifact_inventory",
                "algorithm": DATA_TREE_ALGORITHM,
                "file_count": len(inventory_files),
                "total_size_bytes": sum(item["size_bytes"] for item in inventory_files),
                "files": inventory_files,
                "tree_sha256": _tree_digest(inventory_files),
            }
            inventory_raw = canonical_json_bytes(inventory_payload)
            inventory_path = self.artifact_root / "artifact_inventory.json"
            inventory_path.write_bytes(inventory_raw)
            artifact_inventory_reference = {
                "path": "artifact_inventory.json",
                "sha256": hashlib.sha256(inventory_raw).hexdigest(),
                "size_bytes": len(inventory_raw),
            }
        return make_terminal(artifact_inventory_reference)

    def _anchors(self, start: dict, terminal: dict, *, prereg: dict | None = None) -> dict:
        prereg = self.prereg if prereg is None else prereg
        normalized_start = validate_runtime_start(start, preregistration=prereg)
        normalized_terminal = validate_terminal_receipt(
            terminal,
            preregistration=prereg,
            runtime_start=normalized_start,
            seed_schedule=self.schedule,
            task_map=self.task_map,
        )
        return {
            "preregistration_canonical_sha256": canonical_json_sha256(prereg),
            "runtime_start_canonical_sha256": canonical_json_sha256(normalized_start),
            "terminal_canonical_sha256": canonical_json_sha256(normalized_terminal),
        }

    def _validate_chain(
        self,
        start: dict,
        terminal: dict,
        *,
        prereg: dict | None = None,
        anchors: dict | None = None,
        model_cache_root: Path | None = None,
    ) -> dict:
        prereg = self.prereg if prereg is None else prereg
        if anchors is None:
            anchors = self._anchors(start, terminal, prereg=prereg)
        return validate_contract_chain(
            preregistration=prereg,
            runtime_start=start,
            terminal=terminal,
            data_inventory=self.inventory,
            seed_schedule=self.schedule,
            trusted_anchors=anchors,
            data_root=self.root,
            model_cache_root=(
                self.model_cache_root
                if model_cache_root is None
                else model_cache_root
            ),
            artifact_root=self.artifact_root,
        )

    def _reseal_inventory_entry(self, terminal: dict, relative_path: str) -> dict:
        artifact = self.artifact_root / relative_path
        raw = artifact.read_bytes()
        inventory_path = self.artifact_root / "artifact_inventory.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        matches = [item for item in inventory["files"] if item["path"] == relative_path]
        self.assertEqual(len(matches), 1)
        matches[0]["sha256"] = hashlib.sha256(raw).hexdigest()
        matches[0]["size_bytes"] = len(raw)
        inventory["total_size_bytes"] = sum(
            item["size_bytes"] for item in inventory["files"]
        )
        inventory["tree_sha256"] = _tree_digest(inventory["files"])
        inventory_raw = canonical_json_bytes(inventory)
        inventory_path.write_bytes(inventory_raw)
        terminal["artifact_inventory"] = {
            "path": "artifact_inventory.json",
            "sha256": hashlib.sha256(inventory_raw).hexdigest(),
            "size_bytes": len(inventory_raw),
        }
        return matches[0]

    def test_canonical_json_is_order_independent_and_duplicate_keys_are_rejected(self) -> None:
        self.assertEqual(
            canonical_json_sha256({"b": 1, "a": [2, 3]}),
            canonical_json_sha256({"a": [2, 3], "b": 1}),
        )
        duplicate = Path(self.temporary.name) / "duplicate.json"
        duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "duplicate JSON object key"):
            load_json_strict(duplicate)

    def test_canonical_json_write_is_atomic_no_clobber(self) -> None:
        output = Path(self.temporary.name) / "immutable.json"
        write_canonical_json(output, {"version": 1})
        original = output.read_bytes()
        with self.assertRaisesRegex(ContractError, "refusing to overwrite"):
            write_canonical_json(output, {"version": 2})
        self.assertEqual(output.read_bytes(), original)

    def test_canonical_json_write_creates_nested_directories_nofollow(self) -> None:
        output = Path(self.temporary.name) / "contracts" / "nested" / "receipt.json"
        write_canonical_json(output, {"version": 1})
        self.assertEqual(output.read_bytes(), canonical_json_bytes({"version": 1}))

    def test_canonical_json_write_does_not_create_through_parent_symlink(self) -> None:
        base = Path(self.temporary.name) / "contract-root"
        outside = Path(self.temporary.name) / "outside"
        base.mkdir()
        outside.mkdir()
        (base / "linked").symlink_to(outside, target_is_directory=True)
        output = base / "linked" / "nested" / "receipt.json"

        with self.assertRaisesRegex(ContractError, "without following symlinks"):
            write_canonical_json(output, {"version": 1})

        self.assertFalse((outside / "nested").exists())
        self.assertFalse(output.exists())

    def test_inventory_matches_sha256sum_path_algorithm_and_exact_roles(self) -> None:
        digest = hashlib.sha256()
        for item in sorted(self.inventory["files"], key=lambda value: value["path"].encode()):
            digest.update(f"{item['sha256']}  {item['path']}\n".encode("utf-8"))
        self.assertEqual(self.inventory["algorithm"], DATA_TREE_ALGORITHM)
        self.assertEqual(self.inventory["tree_sha256"], digest.hexdigest())
        self.assertEqual(self.inventory["file_count"], EXPECTED_DATA_FILES)
        self.assertEqual(
            {item["role"] for item in self.inventory["files"]},
            {"bddl", "initial_states"},
        )

    def test_inventory_and_schedule_are_deterministic_under_task_reordering(self) -> None:
        reversed_map = copy.deepcopy(self.task_map)
        reversed_map["tasks"].reverse()
        other_inventory = build_data_inventory(
            self.root,
            reversed_map,
            dataset_id="libero-40",
            revision=OFFICIAL_LIBERO_COMMIT,
        )
        other_schedule = build_seed_schedule(reversed_map, seed=42, python_hash_seed=42)
        self.assertEqual(other_inventory, self.inventory)
        self.assertEqual(other_schedule, self.schedule)

    def test_inventory_rejects_symlink_and_hardlink_aliases(self) -> None:
        first = self.task_map["tasks"][0]
        bddl = self.root / first["bddl_path"]
        target = bddl.with_suffix(".target")
        bddl.rename(target)
        bddl.symlink_to(target.name)
        with self.assertRaises(ContractError):
            build_data_inventory(
                self.root,
                self.task_map,
                dataset_id="libero-40",
                revision=OFFICIAL_LIBERO_COMMIT,
            )
        bddl.unlink()
        target.rename(bddl)
        second = self.task_map["tasks"][1]
        second_path = self.root / second["bddl_path"]
        second_path.unlink()
        os.link(bddl, second_path)
        with self.assertRaisesRegex(ContractError, "duplicate filesystem object"):
            build_data_inventory(
                self.root,
                self.task_map,
                dataset_id="libero-40",
                revision=OFFICIAL_LIBERO_COMMIT,
            )

    def test_task_map_rejects_path_escape_and_wrong_pairing(self) -> None:
        bad = copy.deepcopy(self.task_map)
        bad["tasks"][0]["bddl_path"] = "../outside.bddl"
        with self.assertRaises(ContractError):
            build_seed_schedule(bad, seed=42, python_hash_seed=42)
        bad = copy.deepcopy(self.task_map)
        bad["tasks"][0]["init_state_path"] = bad["tasks"][1]["init_state_path"]
        with self.assertRaises(ContractError):
            build_seed_schedule(bad, seed=42, python_hash_seed=42)

    def test_inventory_revalidation_reads_back_and_detects_tamper(self) -> None:
        validate_data_inventory(self.inventory, data_root=self.root)
        path = self.root / self.task_map["tasks"][0]["bddl_path"]
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ContractError, "does not match inventory"):
            validate_data_inventory(self.inventory, data_root=self.root)

    def test_seed_schedule_models_task_process_not_invented_episode_seeds(self) -> None:
        self.assertEqual(self.schedule["task_process_count"], EXPECTED_TASKS)
        self.assertEqual(self.schedule["episode_count"], EXPECTED_EPISODES)
        process = self.schedule["task_processes"][0]
        self.assertEqual(process["environment_seed_scope"], "once-before-trial-0")
        self.assertEqual(process["policy_seed_scope"], "constant-each-replan-call")
        self.assertEqual(process["trial_order"], list(range(50)))
        self.assertNotIn("task_seed", self.schedule["episodes"][0])
        self.assertNotIn("environment_seed", self.schedule["episodes"][0])

    def test_seed_schedule_rejects_wrong_or_ambiguous_binding(self) -> None:
        wrong = copy.deepcopy(self.schedule)
        wrong["task_processes"][0]["policy_seed"] = 99
        with self.assertRaisesRegex(ContractError, "wrong policy_seed"):
            validate_seed_schedule(wrong, task_map=self.task_map)
        ambiguous = copy.deepcopy(self.schedule)
        ambiguous["episodes"][0]["task_seed"] = 42
        with self.assertRaisesRegex(ContractError, "keys do not match schema"):
            validate_seed_schedule(ambiguous, task_map=self.task_map)

    def test_preregistration_requires_real_immutable_image(self) -> None:
        missing = self._prereg_spec()
        del missing["image"]
        with self.assertRaisesRegex(ContractError, "missing=.*image"):
            build_preregistration(
                missing,
                data_inventory=self.inventory,
                seed_schedule=self.schedule,
            )
        placeholder = self._prereg_spec()
        placeholder["image"] = {
            "uri": f"registry.example/mfwam@sha256:{'0' * 64}",
            "digest": f"sha256:{'0' * 64}",
        }
        with self.assertRaisesRegex(ContractError, "unknown placeholder"):
            build_preregistration(
                placeholder,
                data_inventory=self.inventory,
                seed_schedule=self.schedule,
            )

    def test_preregistration_rejects_runtime_or_terminal_fields(self) -> None:
        polluted = copy.deepcopy(self.prereg)
        polluted["completed_at"] = "2026-08-03T00:00:00+08:00"
        with self.assertRaisesRegex(ContractError, "unexpected=.*completed_at"):
            validate_preregistration(polluted)

    def test_preregistration_data_revision_must_equal_libero_source_commit(self) -> None:
        mismatched_inventory = copy.deepcopy(self.inventory)
        mismatched_inventory["revision"] = "d" * 40
        with self.assertRaisesRegex(ContractError, r"data\.revision.*LIBERO source commit"):
            build_preregistration(
                self._prereg_spec(),
                data_inventory=mismatched_inventory,
                seed_schedule=self.schedule,
            )

    def test_model_cache_inventory_requires_all_six_files_and_runtime_rebinding(self) -> None:
        missing = self._prereg_spec()
        missing["artifacts"]["model_cache"]["files"].pop()
        with self.assertRaisesRegex(ContractError, "exactly 6 entries"):
            build_preregistration(
                missing,
                data_inventory=self.inventory,
                seed_schedule=self.schedule,
            )
        start = self._runtime_start()
        start["model_cache_inventory"]["files"][0]["sha256"] = _digest(
            "tampered-model-cache"
        )
        core = {
            key: start["model_cache_inventory"][key]
            for key in ("algorithm", "file_count", "files")
        }
        start["model_cache_inventory"]["canonical_sha256"] = canonical_json_sha256(core)
        start["bindings"][
            "model_cache_inventory_canonical_sha256"
        ] = start["model_cache_inventory"]["canonical_sha256"]
        with self.assertRaisesRegex(ContractError, "bindings|model-cache inventory"):
            validate_runtime_start(start, preregistration=self.prereg)
        start = self._runtime_start()
        first_path = self.model_cache_root / start["model_cache_inventory"]["files"][0][
            "path"
        ]
        first_path.write_bytes(b"tampered-on-disk")
        with self.assertRaisesRegex(ContractError, "does not match inventory"):
            validate_runtime_start(
                start,
                preregistration=self.prereg,
                model_cache_root=self.model_cache_root,
            )

    def test_model_cache_environment_is_required_immutable_and_live_bound(self) -> None:
        for field in ("DIFFSYNTH_MODEL_BASE_PATH", "DIFFSYNTH_SKIP_DOWNLOAD"):
            with self.subTest(preregistration_missing=field):
                missing = self._prereg_spec()
                del missing["runtime_environment"][field]
                with self.assertRaisesRegex(ContractError, f"missing=.*{field}"):
                    build_preregistration(
                        missing,
                        data_inventory=self.inventory,
                        seed_schedule=self.schedule,
                    )

        noncanonical = self._prereg_spec()
        noncanonical["runtime_environment"]["DIFFSYNTH_MODEL_BASE_PATH"] = (
            f"{self.model_cache_root}/../model-cache"
        )
        with self.assertRaisesRegex(ContractError, "canonical absolute POSIX path"):
            build_preregistration(
                noncanonical,
                data_inventory=self.inventory,
                seed_schedule=self.schedule,
            )

        downloadable = self._prereg_spec()
        downloadable["runtime_environment"]["DIFFSYNTH_SKIP_DOWNLOAD"] = "false"
        with self.assertRaisesRegex(ContractError, "must be exactly 'true'"):
            build_preregistration(
                downloadable,
                data_inventory=self.inventory,
                seed_schedule=self.schedule,
            )

        missing_start = self._runtime_start()
        del missing_start["runtime_environment"]["DIFFSYNTH_MODEL_BASE_PATH"]
        with self.assertRaisesRegex(
            ContractError, "missing=.*DIFFSYNTH_MODEL_BASE_PATH"
        ):
            validate_runtime_start(missing_start, preregistration=self.prereg)

        drifted_start = self._runtime_start()
        drifted_start["runtime_environment"]["DIFFSYNTH_MODEL_BASE_PATH"] = (
            str(self.model_cache_root.parent / "other-model-cache")
        )
        with self.assertRaisesRegex(ContractError, "environment does not match"):
            validate_runtime_start(drifted_start, preregistration=self.prereg)

        downloadable_start = self._runtime_start()
        downloadable_start["runtime_environment"]["DIFFSYNTH_SKIP_DOWNLOAD"] = "false"
        with self.assertRaisesRegex(ContractError, "must be exactly 'true'"):
            validate_runtime_start(downloadable_start, preregistration=self.prereg)

        with self.assertRaisesRegex(
            ContractError, "live model_cache_root.*DIFFSYNTH_MODEL_BASE_PATH"
        ):
            validate_runtime_start(
                self._runtime_start(),
                preregistration=self.prereg,
                model_cache_root=self.model_cache_root.parent / "other-model-cache",
            )

        start = validate_runtime_start(
            self._runtime_start(), preregistration=self.prereg
        )
        terminal = self._terminal(start)
        with self.assertRaisesRegex(
            ContractError, "live model_cache_root.*DIFFSYNTH_MODEL_BASE_PATH"
        ):
            self._validate_chain(
                start,
                terminal,
                model_cache_root=self.model_cache_root.parent / "other-model-cache",
            )

    def test_runtime_start_requires_actual_gpu_import_and_control_plane_fields(self) -> None:
        start = self._runtime_start()
        del start["gpu"]["uuids"]
        with self.assertRaisesRegex(ContractError, "missing=.*uuids"):
            validate_runtime_start(start, preregistration=self.prereg)
        start = self._runtime_start()
        start["imports"] = start["imports"][:-1]
        with self.assertRaisesRegex(ContractError, "exactly cover"):
            validate_runtime_start(start, preregistration=self.prereg)
        start = self._runtime_start()
        start["job"]["job_spec_sha256"] = _digest("different-job")
        with self.assertRaisesRegex(ContractError, "job spec"):
            validate_runtime_start(start, preregistration=self.prereg)
        start = self._runtime_start()
        del start["runtime"]["cudnn"]
        with self.assertRaisesRegex(ContractError, "missing=.*cudnn"):
            validate_runtime_start(start, preregistration=self.prereg)
        start = self._runtime_start()
        del start["runtime_environment"]["MUJOCO_GL"]
        with self.assertRaisesRegex(ContractError, "missing=.*MUJOCO_GL"):
            validate_runtime_start(start, preregistration=self.prereg)
        start = self._runtime_start()
        start["control_process"]["world_size"] = 1
        with self.assertRaisesRegex(ContractError, "unexpected=.*world_size"):
            validate_runtime_start(start, preregistration=self.prereg)

    def test_runtime_start_rejects_image_source_and_config_substitution(self) -> None:
        start = self._runtime_start()
        replacement = f"sha256:{'d' * 64}"
        start["image"] = {
            "uri": f"registry.example/mfwam@{replacement}",
            "digest": replacement,
        }
        with self.assertRaisesRegex(ContractError, "image does not match"):
            validate_runtime_start(start, preregistration=self.prereg)
        start = self._runtime_start()
        start["source"]["instrumentation"]["commit"] = "d" * 40
        with self.assertRaisesRegex(ContractError, "source identities"):
            validate_runtime_start(start, preregistration=self.prereg)
        start = self._runtime_start()
        start["bindings"]["resolved_config_sha256"] = _digest("other-config")
        with self.assertRaisesRegex(ContractError, "bindings"):
            validate_runtime_start(start, preregistration=self.prereg)

    def test_successful_terminal_requires_all_40_tasks_and_2000_episodes(self) -> None:
        start = validate_runtime_start(self._runtime_start(), preregistration=self.prereg)
        terminal = self._terminal(start)
        receipt = validate_terminal_receipt(
            terminal,
            preregistration=self.prereg,
            runtime_start=start,
            seed_schedule=self.schedule,
            task_map=self.task_map,
        )
        self.assertEqual(receipt["status"], "SUCCEEDED")
        incomplete = copy.deepcopy(terminal)
        incomplete["task_processes"].pop()
        incomplete["scope"]["task_process_count"] -= 1
        incomplete["scope"]["episode_count"] -= 50
        with self.assertRaisesRegex(ContractError, "incomplete scope|requires 40 task receipts"):
            validate_terminal_receipt(
                incomplete,
                preregistration=self.prereg,
                runtime_start=start,
                seed_schedule=self.schedule,
                task_map=self.task_map,
            )

    def test_failed_terminal_is_recordable_but_never_successful(self) -> None:
        start = validate_runtime_start(self._runtime_start(), preregistration=self.prereg)
        terminal = self._terminal(start, status="FAILED")
        receipt = validate_terminal_receipt(
            terminal,
            preregistration=self.prereg,
            runtime_start=start,
            seed_schedule=self.schedule,
            task_map=self.task_map,
        )
        self.assertEqual(receipt["status"], "FAILED")
        chain = self._validate_chain(start, terminal)
        self.assertEqual(chain["status"], "STRUCTURAL_FAIL")
        self.assertEqual(chain["specialized_g0_status"], "UNCERTAIN")
        self.assertFalse(chain["formal_training_allowed"])

    def test_complete_structural_chain_still_cannot_authorize_training(self) -> None:
        start = validate_runtime_start(self._runtime_start(), preregistration=self.prereg)
        terminal = self._terminal(start)
        chain = self._validate_chain(start, terminal)
        self.assertEqual(chain["status"], "STRUCTURAL_PASS_ONLY")
        self.assertEqual(chain["specialized_g0_status"], "UNCERTAIN")
        self.assertEqual(chain["artifact_audit"]["artifact_count"], 2084)
        self.assertTrue(chain["terminal_success"])
        self.assertFalse(chain["formal_training_allowed"])

    def test_missing_external_anchor_fails_closed(self) -> None:
        start = validate_runtime_start(self._runtime_start(), preregistration=self.prereg)
        terminal = self._terminal(start)
        fields = tuple(self._anchors(start, terminal))
        for field in fields:
            missing = self._anchors(start, terminal)
            del missing[field]
            with self.assertRaisesRegex(ContractError, "keys do not match schema"):
                self._validate_chain(start, terminal, anchors=missing)
            for invalid in ("0" * 64, "a" * 63, "A" * 64):
                bad = self._anchors(start, terminal)
                bad[field] = invalid
                with self.assertRaisesRegex(ContractError, "lowercase SHA-256|all-zero"):
                    self._validate_chain(start, terminal, anchors=bad)

    def test_full_downstream_reseal_cannot_cross_old_external_anchor(self) -> None:
        old_start = validate_runtime_start(self._runtime_start(), preregistration=self.prereg)
        old_terminal = self._terminal(old_start)
        old_anchors = self._anchors(old_start, old_terminal)

        resealed_prereg = copy.deepcopy(self.prereg)
        resealed_prereg["runtime_lock"]["numpy"] = "9.9.9"
        resealed_start = copy.deepcopy(old_start)
        resealed_start["preregistration_canonical_sha256"] = canonical_json_sha256(
            resealed_prereg
        )
        resealed_start["runtime"]["numpy"] = "9.9.9"
        resealed_start = validate_runtime_start(
            resealed_start, preregistration=resealed_prereg
        )
        resealed_terminal = copy.deepcopy(old_terminal)
        resealed_terminal["preregistration_canonical_sha256"] = canonical_json_sha256(
            resealed_prereg
        )
        resealed_terminal["runtime_start_canonical_sha256"] = canonical_json_sha256(
            resealed_start
        )
        validate_terminal_receipt(
            resealed_terminal,
            preregistration=resealed_prereg,
            runtime_start=resealed_start,
            seed_schedule=self.schedule,
            task_map=self.task_map,
        )
        with self.assertRaisesRegex(ContractError, "external trusted anchor"):
            self._validate_chain(
                resealed_start,
                resealed_terminal,
                prereg=resealed_prereg,
                anchors=old_anchors,
            )

    def test_live_terminal_artifact_tamper_and_missing_file_fail(self) -> None:
        start = validate_runtime_start(self._runtime_start(), preregistration=self.prereg)
        terminal = self._terminal(start)
        anchors = self._anchors(start, terminal)
        trace = self.artifact_root / terminal["task_processes"][0]["result_path"]
        trace.write_bytes(trace.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ContractError, "does not match inventory"):
            self._validate_chain(start, terminal, anchors=anchors)

        terminal = self._terminal(start)
        anchors = self._anchors(start, terminal)
        missing = self.artifact_root / terminal["task_processes"][0]["trace_receipt_path"]
        missing.unlink()
        with self.assertRaisesRegex(ContractError, "cannot safely hash"):
            self._validate_chain(start, terminal, anchors=anchors)

    def test_live_terminal_symlink_and_hardlink_aliases_fail(self) -> None:
        start = validate_runtime_start(self._runtime_start(), preregistration=self.prereg)
        terminal = self._terminal(start)
        anchors = self._anchors(start, terminal)
        trace_path = self.artifact_root / "traces/libero_spatial/task00/trial000.json"
        target = trace_path.with_suffix(".target")
        trace_path.rename(target)
        trace_path.symlink_to(target.name)
        with self.assertRaisesRegex(ContractError, "cannot safely hash"):
            self._validate_chain(start, terminal, anchors=anchors)
        trace_path.unlink()
        target.rename(trace_path)

        terminal = self._terminal(start)
        anchors = self._anchors(start, terminal)
        first = self.artifact_root / "traces/libero_spatial/task00/trial000.json"
        second = self.artifact_root / "traces/libero_spatial/task00/trial001.json"
        second.unlink()
        os.link(first, second)
        with self.assertRaisesRegex(ContractError, "hardlinked"):
            self._validate_chain(start, terminal, anchors=anchors)

    def test_live_aggregate_mismatch_fails_after_matching_terminal_anchor(self) -> None:
        start = validate_runtime_start(self._runtime_start(), preregistration=self.prereg)
        terminal = self._terminal(start)
        terminal["aggregates"]["trace_tree_sha256"] = _digest("false-aggregate")
        # The terminal anchor is deliberately recomputed: live artifacts must
        # still defeat a self-consistent but false digest label.
        anchors = self._anchors(start, terminal)
        with self.assertRaisesRegex(ContractError, "trace aggregate mismatch"):
            self._validate_chain(start, terminal, anchors=anchors)

    def test_task_receipt_upstream_binding_survives_downstream_reseal(self) -> None:
        start = validate_runtime_start(self._runtime_start(), preregistration=self.prereg)
        terminal = self._terminal(start)
        process = terminal["task_processes"][0]
        receipt_path = self.artifact_root / process["trace_receipt_path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["bindings"]["resolved_config_sha256"] = _digest("wrong-config")
        receipt_path.write_bytes(canonical_json_bytes(receipt))
        inventory_item = self._reseal_inventory_entry(
            terminal, process["trace_receipt_path"]
        )
        process["trace_receipt_sha256"] = inventory_item["sha256"]
        process["trace_receipt_size_bytes"] = inventory_item["size_bytes"]
        anchors = self._anchors(start, terminal)
        with self.assertRaisesRegex(ContractError, "bindings do not match upstream"):
            self._validate_chain(start, terminal, anchors=anchors)

    def test_completion_marker_binds_terminal_core_after_full_reseal(self) -> None:
        start = validate_runtime_start(self._runtime_start(), preregistration=self.prereg)
        terminal = self._terminal(start)
        completion_path = self.artifact_root / "completion.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["terminal_core_canonical_sha256"] = _digest("wrong-terminal-core")
        completion_path.write_bytes(canonical_json_bytes(completion))
        self._reseal_inventory_entry(terminal, "completion.json")
        anchors = self._anchors(start, terminal)
        with self.assertRaisesRegex(ContractError, "completion marker content is invalid"):
            self._validate_chain(start, terminal, anchors=anchors)


if __name__ == "__main__":
    unittest.main()
