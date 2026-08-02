from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.audit_mf_wam_g0 import (
    AuditFailure,
    SUITES,
    _sha256,
    _task_result_tree_sha256,
    audit,
)


class G0AuditTest(unittest.TestCase):
    def _write_tree(self, root: Path, *, flip: tuple[str, int, int] | None = None) -> None:
        task_rows = []
        suite_stats = {}
        for suite in SUITES:
            suite_successes = 0
            for task_id in range(10):
                successes = list(range(50))
                if flip == (suite, task_id, 0):
                    successes.remove(0)
                failures = sorted(set(range(50)) - set(successes))
                suite_successes += len(successes)
                result_dir = root / suite
                result_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "task_suite": suite,
                    "task_id": task_id,
                    "total_episodes": 50,
                    "successes": len(successes),
                    "success_episodes": successes,
                    "failure_episodes": failures,
                }
                (result_dir / f"gpu0_task{task_id}_results.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                task_rows.append(
                    {
                        "Task": f"{suite}_{task_id}",
                        "Description": "synthetic",
                        "Success Rate (%)": f"{len(successes) / 50 * 100:.2f}",
                    }
                )
                for trial in range(50):
                    trace_dir = root / "faildetect_traces" / suite
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    trace = {
                        "metadata": {
                            "task_suite": suite,
                            "task_id": task_id,
                            "trial_idx": trial,
                            "success": trial in successes,
                            "replan_steps": 10,
                            "action_horizon": 32,
                            "task_seed": trial,
                            "environment_seed": trial,
                            "policy_seed": trial,
                        },
                        "records": [
                            {
                                "env_step": 30 + replan_idx * 10,
                                "episode_idx": trial,
                                "replan_idx": replan_idx,
                                "raw_action_chunk": [[0.0] * 7 for _ in range(10)],
                                "state": [0.0] * 8,
                            }
                            for replan_idx in range(7)
                        ],
                    }
                    (trace_dir / f"task{task_id}_trial{trial}.json").write_text(
                        json.dumps(trace), encoding="utf-8"
                    )
            suite_stats[suite] = {
                "total_tasks": 10,
                "total_trials": 500,
                "total_successes": suite_successes,
                "total_time": 1.0,
                "max_time": 1.0,
            }
        with (root / "task_success_rates.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Task", "Description", "Success Rate (%)"])
            writer.writeheader()
            writer.writerows(task_rows)
        rates = [suite_stats[suite]["total_successes"] / 5 for suite in SUITES]
        with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["checkpoint.pt"])
            writer.writerow(["", *SUITES, "Overall"])
            writer.writerow(["Success Rate (%)", *[f"{rate:.2f}" for rate in rates], f"{sum(rates) / 4:.2f}"])
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": (
                        "synthetic-reference"
                        if root.name == "reference"
                        else "synthetic-candidate"
                    ),
                    "ckpt": "synthetic-checkpoint.pt",
                    "config": "synthetic-config",
                    "suite_stats": suite_stats,
                }
            ),
            encoding="utf-8",
        )

    def _audit(self, args: argparse.Namespace) -> dict:
        reference = args.reference_dir.resolve()
        baseline = {
            "baseline_id": "SYNTHETIC-TEST-BASELINE",
            "source_commit": getattr(args, "test_source_commit", "0" * 40),
            "reference_run": {
                "run_id": "synthetic-reference",
                "config": "synthetic-config",
                "checkpoint_label": "synthetic-checkpoint.pt",
                "summary_csv_sha256": _sha256(reference / "summary.csv"),
                "task_success_rates_csv_sha256": _sha256(
                    reference / "task_success_rates.csv"
                ),
                "summary_json_sha256": _sha256(reference / "summary.json"),
                "task_result_tree_sha256": _task_result_tree_sha256(reference),
            },
            "model_artifacts": {
                "checkpoint_sha256": args.checkpoint_sha256 or "0" * 64,
                "dataset_stats_sha256": args.dataset_stats_sha256 or "0" * 64,
            },
            "candidate_contract": getattr(
                args,
                "test_candidate_contract",
                {
                    "status": "UNREGISTERED",
                    "reason": "synthetic candidate contract not locked",
                },
            ),
        }
        with mock.patch(
            "scripts.audit_mf_wam_g0._load_baseline_policy",
            return_value=(baseline, "f" * 64, Path("synthetic-baseline.json")),
        ):
            return audit(args)

    @staticmethod
    def _args(reference: Path, candidate: Path, **overrides: object) -> argparse.Namespace:
        defaults: dict[str, object] = {
            "reference_dir": reference,
            "candidate_dir": candidate,
            "trace_dir": candidate / "faildetect_traces",
            "checkpoint": None,
            "checkpoint_sha256": None,
            "dataset_stats": None,
            "dataset_stats_sha256": None,
            "resolved_config": None,
            "resolved_config_sha256": None,
            "source_root": None,
            "data_manifest": None,
            "data_root": None,
            "seed_manifest": None,
            "run_manifest": None,
            "expected_replan_steps": 10,
            "expected_action_horizon": 32,
            "expected_action_dimension": 7,
            "expected_state_dimension": 8,
            "expected_first_replan_env_step": 30,
            "minimum_trace_records_per_episode": 7,
            "overall_equivalence_margin": 0.02,
            "suite_drop_margin": 0.03,
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 42,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _complete_provenance(
        self,
        base: Path,
        reference: Path,
        candidate: Path,
    ) -> tuple[dict[str, object], Path, Path]:
        overrides: dict[str, object] = {}
        for arg_name in ("checkpoint", "dataset_stats", "resolved_config"):
            path = base / arg_name
            path.write_bytes(arg_name.encode("ascii"))
            overrides[arg_name] = str(path)
            overrides[f"{arg_name}_sha256"] = _sha256(path)

        source_root = base / "source"
        source_root.mkdir()
        (source_root / "README.md").write_text("synthetic source\n", encoding="utf-8")
        subprocess.run(
            ["git", "init", "-q", str(source_root)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(source_root), "add", "README.md"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(source_root),
                "-c",
                "user.name=MF-WAM Test",
                "-c",
                "user.email=mf-wam-test@example.invalid",
                "commit",
                "-q",
                "-m",
                "synthetic",
            ],
            check=True,
            capture_output=True,
        )
        source_commit = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        overrides["source_root"] = str(source_root)
        overrides["test_source_commit"] = source_commit

        data_file = base / "dataset.bin"
        data_file.write_bytes(b"synthetic dataset")
        data_manifest_path = base / "data-manifest.json"
        data_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "mf_wam_g0_data_manifest",
                    "dataset_id": "synthetic-libero",
                    "revision": "synthetic-revision-1",
                    "files": [
                        {
                            "path": data_file.name,
                            "sha256": _sha256(data_file),
                            "size_bytes": data_file.stat().st_size,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        overrides["data_manifest"] = str(data_manifest_path)
        overrides["data_root"] = str(base)

        seed_manifest_path = base / "seed-manifest.json"
        seed_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "mf_wam_g0_seed_manifest",
                    "episodes": [
                        {
                            "task_suite": suite,
                            "task_id": task_id,
                            "trial_idx": trial_idx,
                            "task_seed": trial_idx,
                            "environment_seed": trial_idx,
                            "policy_seed": trial_idx,
                        }
                        for suite in SUITES
                        for task_id in range(10)
                        for trial_idx in range(50)
                    ],
                }
            ),
            encoding="utf-8",
        )
        overrides["seed_manifest"] = str(seed_manifest_path)

        preliminary = self._audit(self._args(reference, candidate, **overrides))
        self.assertEqual(preliminary["status"], "UNCERTAIN")
        bindings = preliminary["artifact_binding"]
        run_manifest_path = base / "run-manifest.json"
        run_manifest = {
            "schema_version": 1,
            "kind": "mf_wam_g0_run_manifest",
            "run_id": preliminary["candidate"]["run_id"],
            "git": {"commit": source_commit, "clean": True},
            "data": {
                "manifest_sha256": bindings["data_manifest"]["sha256"],
                "dataset_id": bindings["data_manifest"]["dataset_id"],
                "revision": bindings["data_manifest"]["revision"],
                "inventory_sha256": bindings["data_manifest"][
                    "inventory_sha256"
                ],
            },
            "seeds": {
                "manifest_sha256": bindings["seed_manifest"]["sha256"],
                "episode_count": bindings["seed_manifest"]["episode_count"],
                "seed_tree_sha256": bindings["seed_manifest"][
                    "seed_tree_sha256"
                ],
            },
            "artifacts": {
                "checkpoint_sha256": bindings["checkpoint"]["sha256"],
                "dataset_stats_sha256": bindings["dataset_stats"]["sha256"],
                "resolved_config_sha256": bindings["resolved_config"]["sha256"],
            },
            "environment": {
                "image_digest": f"sha256:{'a' * 64}",
                "hostname": "synthetic-host",
                "python": "3.10.20",
                "cuda": "12.8",
                "pytorch": "2.7.1+cu128",
                "mujoco": "3.3.2",
                "libero": "synthetic-revision",
            },
            "terminal": {
                "reference_summary_csv_sha256": preliminary["reference"][
                    "summary_csv_sha256"
                ],
                "reference_task_success_rates_csv_sha256": preliminary[
                    "reference"
                ]["task_success_rates_csv_sha256"],
                "reference_summary_json_sha256": preliminary["reference"][
                    "summary_json_sha256"
                ],
                "reference_task_result_tree_sha256": preliminary["reference"][
                    "task_result_tree_sha256"
                ],
                "candidate_summary_csv_sha256": preliminary["candidate"][
                    "summary_csv_sha256"
                ],
                "candidate_task_success_rates_csv_sha256": preliminary[
                    "candidate"
                ]["task_success_rates_csv_sha256"],
                "candidate_summary_json_sha256": preliminary["candidate"][
                    "summary_json_sha256"
                ],
                "candidate_task_result_tree_sha256": preliminary["candidate"][
                    "task_result_tree_sha256"
                ],
                "candidate_trace_tree_sha256": preliminary["traces"][
                    "tree_sha256"
                ],
                "candidate_run_id": preliminary["candidate"]["run_id"],
            },
            "baseline": {
                "policy_sha256": "f" * 64,
                "baseline_id": "SYNTHETIC-TEST-BASELINE",
                "reference_run_id": "synthetic-reference",
            },
            "evaluation": {
                "suites": list(SUITES),
                "tasks_per_suite": 10,
                "trials_per_task": 50,
                "replan_steps": 10,
                "action_horizon": 32,
                "action_dimension": 7,
                "state_dimension": 8,
                "first_replan_env_step": 30,
                "minimum_trace_records_per_episode": 7,
                "strict_success_predicate": "libero_task_success",
                "confidence_level": 0.95,
                "bootstrap_seed": 42,
                "minimum_bootstrap_replicates": 10_000,
            },
        }
        run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
        overrides["run_manifest"] = str(run_manifest_path)
        overrides["test_candidate_contract"] = {
            "status": "LOCKED",
            "dataset_id": bindings["data_manifest"]["dataset_id"],
            "data_revision": bindings["data_manifest"]["revision"],
            "data_inventory_sha256": bindings["data_manifest"][
                "inventory_sha256"
            ],
            "seed_tree_sha256": bindings["seed_manifest"]["seed_tree_sha256"],
            "resolved_config_sha256": bindings["resolved_config"]["sha256"],
            "image_digest": f"sha256:{'a' * 64}",
            "python": "3.10.20",
            "cuda": "12.8",
            "pytorch": "2.7.1+cu128",
            "mujoco": "3.3.2",
            "libero": "synthetic-revision",
        }
        return overrides, source_root, run_manifest_path

    def test_exact_outcomes_without_artifacts_are_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            result = self._audit(self._args(reference, candidate))
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertEqual(result["outcome_reproduction"]["status"], "PASS")
        self.assertTrue(result["outcome_reproduction"]["exact_episode_outcome_match"])
        self.assertEqual(result["traces"]["episode_count"], 2000)
        self.assertEqual(result["gate_evidence"]["overall_success_delta"]["ci_lower"], 0.0)
        self.assertFalse(result["gate_evidence"]["evidence_complete"])

    def test_artifacts_without_run_provenance_are_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            overrides: dict[str, object] = {}
            for arg_name in ("checkpoint", "dataset_stats", "resolved_config"):
                path = base / arg_name
                path.write_bytes(arg_name.encode("ascii"))
                overrides[arg_name] = str(path)
                overrides[f"{arg_name}_sha256"] = _sha256(path)
            result = self._audit(self._args(reference, candidate, **overrides))
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertEqual(result["artifact_binding"]["run_manifest"]["status"], "UNCERTAIN")

    def test_complete_reverse_bound_provenance_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            overrides, _, _ = self._complete_provenance(
                base, reference, candidate
            )
            result = self._audit(self._args(reference, candidate, **overrides))
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["gate_evidence"]["artifact_bindings_complete"])
        self.assertEqual(result["artifact_binding"]["run_manifest"]["status"], "PASS")

    def test_complete_artifacts_without_locked_candidate_contract_are_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            overrides, _, _ = self._complete_provenance(
                base, reference, candidate
            )
            overrides["test_candidate_contract"] = {
                "status": "UNREGISTERED",
                "reason": "candidate contract not frozen before execution",
            }
            result = self._audit(self._args(reference, candidate, **overrides))
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertEqual(
            result["artifact_binding"]["candidate_contract"]["status"],
            "UNCERTAIN",
        )

    def test_run_manifest_terminal_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            overrides, _, run_manifest_path = self._complete_provenance(
                base, reference, candidate
            )
            payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            payload["terminal"]["candidate_trace_tree_sha256"] = "0" * 64
            run_manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            result = self._audit(self._args(reference, candidate, **overrides))
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["artifact_binding"]["run_manifest"]["status"], "FAIL")

    def test_invalid_run_manifest_digest_fails_before_missing_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            overrides, _, run_manifest_path = self._complete_provenance(
                base, reference, candidate
            )
            payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            payload["artifacts"]["checkpoint_sha256"] = "not-a-sha256"
            run_manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            overrides["checkpoint"] = None
            result = self._audit(self._args(reference, candidate, **overrides))
        self.assertEqual(result["status"], "FAIL")
        self.assertIn(
            "invalid SHA-256",
            result["artifact_binding"]["run_manifest"]["reason"],
        )

    def test_dirty_source_worktree_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            overrides, source_root, _ = self._complete_provenance(
                base, reference, candidate
            )
            (source_root / "dirty.txt").write_text("untracked\n", encoding="utf-8")
            result = self._audit(self._args(reference, candidate, **overrides))
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["artifact_binding"]["source_git"]["status"], "FAIL")

    def test_clean_wrong_source_commit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            overrides, source_root, _ = self._complete_provenance(
                base, reference, candidate
            )
            (source_root / "second.txt").write_text("second\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(source_root), "add", "second.txt"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_root),
                    "-c",
                    "user.name=MF-WAM Test",
                    "-c",
                    "user.email=mf-wam-test@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "wrong clean commit",
                ],
                check=True,
                capture_output=True,
            )
            result = self._audit(self._args(reference, candidate, **overrides))
        self.assertEqual(result["status"], "FAIL")
        self.assertIn(
            "does not match",
            result["artifact_binding"]["source_git"]["reason"],
        )

    def test_missing_manifest_data_file_is_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            overrides, _, _ = self._complete_provenance(
                base, reference, candidate
            )
            (base / "dataset.bin").unlink()
            result = self._audit(self._args(reference, candidate, **overrides))
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertEqual(
            result["artifact_binding"]["data_manifest"]["status"],
            "UNCERTAIN",
        )

    def test_tampered_manifest_data_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            overrides, _, _ = self._complete_provenance(
                base, reference, candidate
            )
            (base / "dataset.bin").write_bytes(b"tampered dataset")
            result = self._audit(self._args(reference, candidate, **overrides))
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            result["artifact_binding"]["data_manifest"]["status"],
            "FAIL",
        )

    def test_malformed_run_manifest_fails_even_with_missing_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            run_manifest_path = base / "run-manifest.json"
            run_manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "mf_wam_g0_run_manifest",
                        "run_id": "synthetic-candidate",
                        "environment": {
                            "image_digest": f"sha256:{'a' * 64}",
                            "hostname": "synthetic-host",
                            "python": "3.10.20",
                            "cuda": "12.8",
                            "pytorch": "2.7.1+cu128",
                            "mujoco": "3.3.2",
                            "libero": "synthetic-revision",
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = self._audit(
                self._args(
                    reference,
                    candidate,
                    run_manifest=str(run_manifest_path),
                )
            )
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            result["artifact_binding"]["run_manifest"]["status"],
            "FAIL",
        )

    def test_missing_trace_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            (candidate / "faildetect_traces" / SUITES[0] / "task0_trial0.json").unlink()
            with self.assertRaisesRegex(AuditFailure, "expected 2000 trace"):
                self._audit(self._args(reference, candidate))

    def test_non_finite_trace_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            trace_path = candidate / "faildetect_traces" / SUITES[0] / "task0_trial0.json"
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            payload["records"][0]["state"][0] = float("nan")
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(AuditFailure, "non-finite"):
                self._audit(self._args(reference, candidate))

    def test_wrong_trace_episode_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            trace_path = candidate / "faildetect_traces" / SUITES[0] / "task0_trial0.json"
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            payload["records"][0]["episode_idx"] = 1
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(AuditFailure, "episode identity mismatch"):
                self._audit(self._args(reference, candidate))

    def test_non_numeric_action_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            trace_path = candidate / "faildetect_traces" / SUITES[0] / "task0_trial0.json"
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            payload["records"][0]["raw_action_chunk"][0][0] = "not-a-number"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(AuditFailure, "non-numeric"):
                self._audit(self._args(reference, candidate))

    def test_short_trace_cannot_claim_complete_temporal_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            trace_path = (
                candidate
                / "faildetect_traces"
                / SUITES[0]
                / "task0_trial0.json"
            )
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            payload["records"] = payload["records"][:1]
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(AuditFailure, "temporal coverage"):
                self._audit(self._args(reference, candidate))

    def test_wrong_first_replan_step_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            trace_path = (
                candidate
                / "faildetect_traces"
                / SUITES[0]
                / "task0_trial0.json"
            )
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            payload["records"][0]["env_step"] = 20
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(AuditFailure, "first replan"):
                self._audit(self._args(reference, candidate))

    def test_trace_seed_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            overrides, _, _ = self._complete_provenance(
                base, reference, candidate
            )
            trace_path = (
                candidate
                / "faildetect_traces"
                / SUITES[0]
                / "task0_trial0.json"
            )
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            payload["metadata"]["policy_seed"] = 999
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(AuditFailure, "trace seed mismatch"):
                self._audit(self._args(reference, candidate, **overrides))

    def test_present_wrong_seed_fails_even_when_another_seed_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            overrides, _, _ = self._complete_provenance(
                base, reference, candidate
            )
            trace_path = (
                candidate
                / "faildetect_traces"
                / SUITES[0]
                / "task0_trial0.json"
            )
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            del payload["metadata"]["task_seed"]
            payload["metadata"]["policy_seed"] = 999
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(AuditFailure, "trace seed mismatch"):
                self._audit(self._args(reference, candidate, **overrides))

    def test_preregistered_margins_cannot_be_relaxed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "reference"
            candidate = base / "candidate"
            self._write_tree(reference)
            self._write_tree(candidate)
            with self.assertRaisesRegex(AuditFailure, "margin is fixed"):
                self._audit(
                    self._args(
                        reference,
                        candidate,
                        overall_equivalence_margin=1.0,
                    )
                )

    def test_reference_and_candidate_must_be_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "same-root"
            self._write_tree(root)
            with self.assertRaisesRegex(AuditFailure, "must be distinct"):
                self._audit(self._args(root, root))


if __name__ == "__main__":
    unittest.main()
