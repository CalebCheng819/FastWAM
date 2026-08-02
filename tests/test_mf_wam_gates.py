"""Focused standard-library tests for the MF-WAM gate evaluator."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.validation.mf_wam_gates import (
    FAIL,
    PASS,
    UNCERTAIN,
    default_policy_path,
    evaluate_gate,
    evaluate_policy,
    load_policy,
)


def interval(estimate: float, lower: float, upper: float) -> dict[str, float]:
    return {"estimate": estimate, "ci_lower": lower, "ci_upper": upper}


def passing_evidence() -> dict[str, dict]:
    return {
        "G0": {
            "evidence_complete": True,
            "episode_count": 2000,
            "suite_count": 4,
            "tasks_per_suite": 10,
            "trials_per_task": 50,
            "missing_trace_count": 0,
            "non_finite_count": 0,
            "missing_seed_binding_count": 0,
            "first_replan_env_step": 30,
            "minimum_trace_records_per_episode": 7,
            "artifact_bindings_complete": True,
            "paired_episode_identity_complete": True,
            "overall_success_delta": interval(0.0, -0.01, 0.01),
            "suite_success_deltas": {
                suite: interval(0.0, -0.02, 0.02)
                for suite in (
                    "libero_spatial",
                    "libero_object",
                    "libero_goal",
                    "libero_10",
                )
            },
        },
        "G1": {
            "evidence_complete": True,
            "conformal_alpha": 0.05,
            "split_unit": "base_episode",
            "split_fractions": [0.6, 0.15, 0.1, 0.15],
            "base_episode_grouping_complete": True,
            "test_split_locked": True,
            "tpr_operating_point_fpr_max": 0.05,
            "missed_failures_in_delay": True,
            "action_shuffled_control_included": True,
            "fastwam_backbone_frozen": True,
            "ci_contract_id": "MF-WAM-G1-CI-v1",
            "confidence_level": 0.95,
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 42,
            "false_positive_rate": interval(0.05, 0.04, 0.07),
            "true_positive_rate": interval(0.86, 0.81, 0.9),
            "auroc": interval(0.91, 0.86, 0.95),
            "median_detection_delay_replans": interval(0.6, 0.3, 0.9),
            "action_conditioned_minus_observation_only": interval(
                0.04, 0.01, 0.07
            ),
        },
        "G2": {
            "evidence_complete": True,
            "expert_count": 4,
            "top_k": 2,
            "fastwam_backbone_frozen": True,
            "active_compute_matched_dense_control": True,
            "shuffled_routing_control": True,
            "task_id_routing_control": True,
            "oracle_semantic_mode_control": True,
            "semantic_labels_from_simulator_predicates": True,
            "training_seed_count": 3,
            "ci_contract_id": "MF-WAM-G2-CI-v1",
            "confidence_level": 0.95,
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 42,
            "relative_improvement": interval(0.09, 0.05, 0.13),
            "expert_dispatch_loads": {
                "expert_0": interval(0.25, 0.2, 0.3),
                "expert_1": interval(0.25, 0.2, 0.3),
                "expert_2": interval(0.25, 0.2, 0.3),
                "expert_3": interval(0.25, 0.2, 0.3),
            },
            "effective_expert_count": interval(3.1, 2.4, 3.6),
            "adjusted_mutual_information": interval(0.65, 0.5, 0.78),
        },
        "G3": {
            "evidence_complete": True,
            "fastwam_backbone_frozen": True,
            "controller_identity_matched": True,
            "paired_disturbance_draws": True,
            "horizon_matched": True,
            "replan_policy_matched": True,
            "recovery_attempt_cap_matched": True,
            "controller_trace_complete": True,
            "strict_success_predicate_bound": True,
            "conditional_recovery_eligibility": "paired_saved_intervention_states",
            "eligibility_count_matched": True,
            "reach_exposure_counts_reported": True,
            "ci_contract_id": "MF-WAM-G3-CI-v1",
            "confidence_level": 0.95,
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 42,
            "training_seed_count": 3,
            "scope_unit": "per_model_per_training_seed",
            "nominal_task_count_per_seed": 40,
            "nominal_trials_per_task": 50,
            "nominal_episode_count_per_seed": 2000,
            "nominal_episode_count_total_per_model": 6000,
            "disturbed_task_count_per_seed": 10,
            "disturbed_trials_per_task": 50,
            "disturbed_episode_count_per_seed": 500,
            "disturbed_episode_count_total_per_model": 1500,
            "disturbed_success_improvement": interval(0.09, 0.05, 0.13),
            "conditional_recovery_improvement": interval(0.16, 0.1, 0.22),
            "nominal_success_delta": interval(0.0, -0.019, 0.018),
            "harmful_trigger_rate": interval(0.01, 0.0, 0.02),
        },
    }


class PolicyTests(unittest.TestCase):
    def test_default_policy_is_strict_json_and_loads(self) -> None:
        policy_path = default_policy_path()
        with policy_path.open("r", encoding="utf-8") as handle:
            raw_policy = json.load(handle, parse_constant=self._reject_constant)
        self.assertEqual(raw_policy["policy_id"], load_policy()["policy_id"])
        self.assertEqual(list(raw_policy["gates"]), ["G0", "G1", "G2", "G3"])

    @staticmethod
    def _reject_constant(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant: {value}")


class GateDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = passing_evidence()

    @staticmethod
    def _canonical_sha256(value: object) -> str:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _attach_audited_receipts(self, policy: dict | None = None) -> None:
        policy = load_policy() if policy is None else policy
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        policy_sha256 = self._canonical_sha256(policy)
        for gate_id in ("G1", "G2", "G3"):
            receipt_check = next(
                check
                for check in policy["gates"][gate_id]["checks"]
                if check["type"] == "audited_receipt"
            )
            gate_evidence = self.evidence[gate_id]
            evidence_sha256 = self._canonical_sha256(gate_evidence)
            receipt = {
                "schema_version": 1,
                "kind": "mf_wam_gate_audit_receipt",
                "gate_id": gate_id,
                "policy_id": policy["policy_id"],
                "policy_sha256": policy_sha256,
                "ci_contract_id": receipt_check["ci_contract_id"],
                "evidence_sha256": evidence_sha256,
                "terminal": True,
                "source_manifest_sha256": hashlib.sha256(
                    f"{gate_id}-source".encode("ascii")
                ).hexdigest(),
                "scope": receipt_check["required_scope"],
                "auditor": {"source_commit": "a" * 40, "clean": True},
                "artifact_digests": {
                    name: hashlib.sha256(
                        f"{gate_id}-{name}".encode("ascii")
                    ).hexdigest()
                    for name in receipt_check["required_artifact_digests"]
                },
            }
            path = root / f"{gate_id.lower()}-receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            gate_evidence["audit_receipt"] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

    def test_safe_intervals_and_receipt_envelopes_do_not_authorize(self) -> None:
        self._attach_audited_receipts()
        result = evaluate_policy(self.evidence)
        self.assertEqual(result["status"], UNCERTAIN)
        self.assertEqual(result["gates"]["G0"]["status"], PASS)
        self.assertTrue(
            all(
                result["gates"][gate_id]["status"] == UNCERTAIN
                for gate_id in ("G1", "G2", "G3")
            )
        )
        self.assertFalse(result["formal_training_allowed"])
        self.assertFalse(result["gate_thresholds_passed"])

    def test_raw_passing_dictionary_cannot_authorize_training(self) -> None:
        result = evaluate_policy(self.evidence)
        self.assertEqual(result["status"], UNCERTAIN)
        self.assertFalse(result["formal_training_allowed"])
        self.assertEqual(result["gates"]["G1"]["status"], UNCERTAIN)

    def test_missing_gate_and_missing_metric_are_uncertain(self) -> None:
        del self.evidence["G1"]["auroc"]
        self.assertEqual(evaluate_gate("G1", self.evidence["G1"])["status"], UNCERTAIN)

        del self.evidence["G3"]
        result = evaluate_policy(self.evidence)
        self.assertEqual(result["gates"]["G3"]["status"], UNCERTAIN)
        self.assertFalse(result["formal_training_allowed"])

    def test_non_finite_and_invalid_intervals_are_uncertain(self) -> None:
        self.evidence["G1"]["auroc"]["ci_lower"] = math.nan
        self.assertEqual(evaluate_gate("G1", self.evidence["G1"])["status"], UNCERTAIN)

        self.evidence = passing_evidence()
        self.evidence["G3"]["nominal_success_delta"] = interval(0.0, 0.1, 0.2)
        self.assertEqual(evaluate_gate("G3", self.evidence["G3"])["status"], UNCERTAIN)

    def test_non_finite_extra_metadata_is_uncertain(self) -> None:
        self.evidence["G1"]["diagnostic_metadata"] = {"unused_loss": math.inf}
        result = evaluate_gate("G1", self.evidence["G1"])
        self.assertEqual(result["status"], UNCERTAIN)

    def test_metric_values_outside_physical_domain_are_uncertain(self) -> None:
        self.evidence["G1"]["false_positive_rate"] = interval(-0.1, -0.2, 0.0)
        result = evaluate_gate("G1", self.evidence["G1"])
        self.assertEqual(result["status"], UNCERTAIN)

        self.evidence = passing_evidence()
        self.evidence["G1"]["auroc"] = interval(1.1, 1.05, 1.2)
        result = evaluate_gate("G1", self.evidence["G1"])
        self.assertEqual(result["status"], UNCERTAIN)

    def test_invalid_metric_domain_policy_is_rejected(self) -> None:
        policy = load_policy()
        policy["gates"]["G1"]["checks"][0]["valid_range"] = {
            "minimum": 1.0,
            "maximum": 0.0,
        }
        with self.assertRaises(ValueError):
            evaluate_gate("G1", self.evidence["G1"], policy)

    def test_ci_crossing_boundary_is_uncertain(self) -> None:
        self.evidence["G1"]["false_positive_rate"] = interval(0.07, 0.06, 0.08)
        result = evaluate_gate("G1", self.evidence["G1"])
        self.assertEqual(result["status"], UNCERTAIN)

    def test_ci_wholly_in_failure_direction_is_fail(self) -> None:
        self.evidence["G1"]["true_positive_rate"] = interval(0.72, 0.68, 0.77)
        result = evaluate_gate("G1", self.evidence["G1"])
        self.assertEqual(result["status"], FAIL)

    def test_explicit_failure_precedes_unrelated_uncertainty(self) -> None:
        del self.evidence["G1"]["auroc"]
        self.evidence["G1"]["true_positive_rate"] = interval(0.72, 0.68, 0.77)
        self.assertEqual(evaluate_gate("G1", self.evidence["G1"])["status"], FAIL)

    def test_g0_requires_all_four_preregistered_suites(self) -> None:
        del self.evidence["G0"]["suite_success_deltas"]["libero_goal"]
        result = evaluate_gate("G0", self.evidence["G0"])
        self.assertEqual(result["status"], UNCERTAIN)

    def test_g0_detects_clear_suite_regression(self) -> None:
        self.evidence["G0"]["suite_success_deltas"]["libero_goal"] = interval(
            -0.05, -0.06, -0.04
        )
        result = evaluate_gate("G0", self.evidence["G0"])
        self.assertEqual(result["status"], FAIL)

    def test_g2_is_fixed_to_four_experts_and_top_two(self) -> None:
        self.evidence["G2"]["top_k"] = 1
        self.assertEqual(evaluate_gate("G2", self.evidence["G2"])["status"], FAIL)

        self.evidence = passing_evidence()
        del self.evidence["G2"]["expert_dispatch_loads"]["expert_3"]
        self.assertEqual(
            evaluate_gate("G2", self.evidence["G2"])["status"], UNCERTAIN
        )

    def test_g2_rejects_dispatch_loads_that_do_not_sum_to_one(self) -> None:
        for load in self.evidence["G2"]["expert_dispatch_loads"].values():
            load["estimate"] = 0.2
        result = evaluate_gate("G2", self.evidence["G2"])
        self.assertEqual(result["status"], FAIL)

    def test_g2_clear_router_collapse_fails(self) -> None:
        self.evidence["G2"]["expert_dispatch_loads"] = {
            "expert_0": interval(0.82, 0.78, 0.86),
            "expert_1": interval(0.06, 0.04, 0.08),
            "expert_2": interval(0.06, 0.04, 0.08),
            "expert_3": interval(0.06, 0.04, 0.08),
        }
        self.evidence["G2"]["effective_expert_count"] = interval(1.7, 1.5, 1.9)
        self.assertEqual(evaluate_gate("G2", self.evidence["G2"])["status"], FAIL)

    def test_g3_clear_harmful_trigger_rate_fails(self) -> None:
        self.evidence["G3"]["harmful_trigger_rate"] = interval(0.04, 0.03, 0.05)
        self.assertEqual(evaluate_gate("G3", self.evidence["G3"])["status"], FAIL)

    def test_strict_lower_boundary_does_not_pass(self) -> None:
        self.evidence["G1"]["action_conditioned_minus_observation_only"] = interval(
            0.01, 0.0, 0.02
        )
        self.assertEqual(evaluate_gate("G1", self.evidence["G1"])["status"], UNCERTAIN)

        self.evidence = passing_evidence()
        self.evidence["G3"]["nominal_success_delta"] = interval(
            -0.01, -0.02, 0.0
        )
        self.assertEqual(evaluate_gate("G3", self.evidence["G3"])["status"], UNCERTAIN)

    def test_incomplete_scope_cannot_pass(self) -> None:
        self.evidence["G0"]["evidence_complete"] = False
        result = evaluate_policy(self.evidence)
        self.assertEqual(result["gates"]["G0"]["status"], UNCERTAIN)
        self.assertFalse(result["formal_training_allowed"])

    def test_noncanonical_policy_never_authorizes_formal_training(self) -> None:
        policy = load_policy()
        policy["gates"]["G1"]["checks"][2]["pass"]["threshold"] = 0.84
        policy["gates"]["G1"]["checks"][2]["fail"]["threshold"] = 0.84
        self._attach_audited_receipts(policy)
        result = evaluate_policy(self.evidence, policy)
        self.assertEqual(result["status"], UNCERTAIN)
        self.assertFalse(result["policy_is_canonical"])
        self.assertFalse(result["formal_training_allowed"])

    def test_tampered_audited_receipt_fails(self) -> None:
        self._attach_audited_receipts()
        receipt_path = Path(self.evidence["G1"]["audit_receipt"]["path"])
        receipt_path.write_text("{}", encoding="utf-8")
        result = evaluate_gate("G1", self.evidence["G1"])
        self.assertEqual(result["status"], FAIL)

    def test_resealed_self_signed_receipt_never_authorizes(self) -> None:
        self._attach_audited_receipts()
        receipt_reference = self.evidence["G1"]["audit_receipt"]
        receipt_path = Path(receipt_reference["path"])
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload["artifact_digests"]["metric_rows_sha256"] = "0" * 64
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        receipt_reference["sha256"] = hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest()
        result = evaluate_policy(self.evidence)
        self.assertEqual(result["gates"]["G1"]["status"], UNCERTAIN)
        self.assertFalse(result["formal_training_allowed"])

    def test_receipt_detects_post_seal_evidence_change(self) -> None:
        self._attach_audited_receipts()
        self.evidence["G1"]["auroc"] = interval(0.92, 0.87, 0.96)
        result = evaluate_gate("G1", self.evidence["G1"])
        self.assertEqual(result["status"], FAIL)

    def test_statistical_contract_and_g3_scope_are_frozen(self) -> None:
        self.evidence["G1"]["bootstrap_replicates"] = 9_999
        self.assertEqual(evaluate_gate("G1", self.evidence["G1"])["status"], FAIL)

        self.evidence = passing_evidence()
        self.evidence["G3"]["nominal_episode_count_per_seed"] = 1_999
        self.assertEqual(evaluate_gate("G3", self.evidence["G3"])["status"], FAIL)

    def test_g1_binds_operating_point_and_split_contract(self) -> None:
        self.evidence["G1"]["tpr_operating_point_fpr_max"] = 0.075
        result = evaluate_gate("G1", self.evidence["G1"])
        self.assertEqual(result["status"], FAIL)

    def test_input_is_not_mutated(self) -> None:
        before = copy.deepcopy(self.evidence)
        evaluate_policy(self.evidence)
        self.assertEqual(self.evidence, before)


if __name__ == "__main__":
    unittest.main()
