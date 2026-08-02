"""Evaluate the preregistered MF-WAM G0--G3 gates without third-party code.

The evaluator deliberately distinguishes lack of evidence from evidence of
failure.  A confidence interval must be wholly inside the accepted region to
pass.  A confidence interval wholly inside the rejected region fails.  An
interval that crosses a boundary, or malformed/incomplete evidence, is
``UNCERTAIN`` and therefore never authorizes formal training.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PASS = "PASS"
FAIL = "FAIL"
UNCERTAIN = "UNCERTAIN"

_STATUSES = frozenset((PASS, FAIL, UNCERTAIN))
_INTERVAL_FIELDS = ("estimate", "ci_lower", "ci_upper")
_OPERATORS = {
    "<": lambda left, right: left < right,
    "<=": lambda left, right: left <= right,
    ">": lambda left, right: left > right,
    ">=": lambda left, right: left >= right,
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _policy_sha256(policy: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        policy,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _policy_identity(policy: Mapping[str, Any]) -> tuple[str, bool]:
    digest = _policy_sha256(policy)
    try:
        canonical = load_policy(default_policy_path())
    except (FileNotFoundError, OSError, ValueError):
        return digest, False
    return digest, digest == _policy_sha256(canonical)


def default_policy_path() -> Path:
    """Return the source-checkout path of the versioned MF-WAM gate policy."""

    checkout_path = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "validation"
        / "mf_wam_gates.json"
    )
    if checkout_path.is_file():
        return checkout_path

    cwd_path = Path.cwd() / "configs" / "validation" / "mf_wam_gates.json"
    if cwd_path.is_file():
        return cwd_path
    raise FileNotFoundError(
        "MF-WAM gate policy not found; pass an explicit policy path or run "
        "from the FastWAM source checkout"
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden in policy: {value}")


def load_policy(path: str | Path | None = None) -> dict[str, Any]:
    """Load and structurally validate a JSON gate policy."""

    policy_path = Path(path) if path is not None else default_policy_path()
    with policy_path.open("r", encoding="utf-8") as handle:
        policy = json.load(handle, parse_constant=_reject_json_constant)
    _validate_policy(policy)
    return policy


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _first_non_finite(value: Any, location: str = "evidence") -> str | None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return None
    if isinstance(value, (int, float)):
        return None if math.isfinite(float(value)) else location
    if isinstance(value, Mapping):
        for key, item in value.items():
            found = _first_non_finite(item, f"{location}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _first_non_finite(item, f"{location}[{index}]")
            if found is not None:
                return found
    return None


def _validate_interval(value: Any) -> tuple[dict[str, float] | None, str | None]:
    if not isinstance(value, Mapping):
        return None, "metric must be an interval object"

    interval: dict[str, float] = {}
    for field in _INTERVAL_FIELDS:
        if field not in value:
            return None, f"missing interval field: {field}"
        number = _finite_number(value[field])
        if number is None:
            return None, f"interval field is not finite: {field}"
        interval[field] = number

    if not interval["ci_lower"] <= interval["estimate"] <= interval["ci_upper"]:
        return None, "interval invariant violated: ci_lower <= estimate <= ci_upper"
    return interval, None


def _validate_interval_range(
    interval: Mapping[str, float], valid_range: Any
) -> str | None:
    """Return an error when an interval leaves its metric's physical domain."""

    if valid_range is None:
        return None
    minimum = valid_range.get("minimum")
    maximum = valid_range.get("maximum")
    if minimum is not None and interval["ci_lower"] < float(minimum):
        return f"confidence interval is below the valid minimum: {minimum}"
    if maximum is not None and interval["ci_upper"] > float(maximum):
        return f"confidence interval is above the valid maximum: {maximum}"
    return None


def _criterion_result(
    check_id: str,
    metric_name: str,
    interval: dict[str, float],
    pass_spec: Mapping[str, Any],
    fail_spec: Mapping[str, Any],
) -> dict[str, Any]:
    pass_observed = interval[str(pass_spec["bound"])]
    fail_observed = interval[str(fail_spec["bound"])]
    pass_threshold = float(pass_spec["threshold"])
    fail_threshold = float(fail_spec["threshold"])
    passes = _OPERATORS[str(pass_spec["operator"])](
        pass_observed, pass_threshold
    )
    fails = _OPERATORS[str(fail_spec["operator"])](
        fail_observed, fail_threshold
    )

    if passes:
        status = PASS
        reason = "confidence interval is wholly in the accepted region"
    elif fails:
        status = FAIL
        reason = "confidence interval is wholly in the failure region"
    else:
        status = UNCERTAIN
        reason = "confidence interval crosses the decision boundary"

    return {
        "id": check_id,
        "type": "interval",
        "metric": metric_name,
        "status": status,
        "observed": interval,
        "pass_criterion": dict(pass_spec),
        "fail_criterion": dict(fail_spec),
        "reason": reason,
    }


def _invalid_check(
    check_id: str, check_type: str, subject: str, reason: str
) -> dict[str, Any]:
    return {
        "id": check_id,
        "type": check_type,
        "metric": subject,
        "status": UNCERTAIN,
        "reason": reason,
    }


def _evaluate_interval(
    check: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    check_id = str(check["id"])
    metric_name = str(check["metric"])
    if metric_name not in evidence:
        return _invalid_check(
            check_id, "interval", metric_name, "missing metric evidence"
        )

    interval, error = _validate_interval(evidence[metric_name])
    if interval is None:
        return _invalid_check(check_id, "interval", metric_name, str(error))
    range_error = _validate_interval_range(interval, check.get("valid_range"))
    if range_error is not None:
        return _invalid_check(check_id, "interval", metric_name, range_error)
    return _criterion_result(
        check_id,
        metric_name,
        interval,
        check["pass"],
        check["fail"],
    )


def _collection(
    check: Mapping[str, Any], evidence: Mapping[str, Any]
) -> tuple[Mapping[str, Any] | None, list[str], str | None]:
    collection_name = str(check["collection"])
    expected_ids = [str(item) for item in check["expected_ids"]]
    value = evidence.get(collection_name)
    if not isinstance(value, Mapping):
        return None, expected_ids, "missing or invalid collection evidence"

    observed_ids = set(value)
    expected_id_set = set(expected_ids)
    if observed_ids != expected_id_set:
        missing = sorted(expected_id_set - observed_ids, key=repr)
        extra = sorted(observed_ids - expected_id_set, key=repr)
        return (
            None,
            expected_ids,
            f"collection coverage mismatch; missing={missing}, extra={extra}",
        )
    return value, expected_ids, None


def _combine_statuses(statuses: list[str]) -> str:
    if FAIL in statuses:
        return FAIL
    if not statuses or UNCERTAIN in statuses:
        return UNCERTAIN
    return PASS


def _evaluate_interval_each(
    check: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    check_id = str(check["id"])
    collection_name = str(check["collection"])
    values, expected_ids, error = _collection(check, evidence)
    if values is None:
        return _invalid_check(
            check_id, "interval_each", collection_name, str(error)
        )

    per_item: list[dict[str, Any]] = []
    for item_id in expected_ids:
        interval, interval_error = _validate_interval(values[item_id])
        if interval is None:
            per_item.append(
                _invalid_check(
                    item_id,
                    "interval",
                    f"{collection_name}.{item_id}",
                    str(interval_error),
                )
            )
            continue
        range_error = _validate_interval_range(
            interval, check.get("valid_range")
        )
        if range_error is not None:
            per_item.append(
                _invalid_check(
                    item_id,
                    "interval",
                    f"{collection_name}.{item_id}",
                    range_error,
                )
            )
            continue
        per_item.append(
            _criterion_result(
                item_id,
                f"{collection_name}.{item_id}",
                interval,
                check["pass"],
                check["fail"],
            )
        )

    status = _combine_statuses([item["status"] for item in per_item])
    return {
        "id": check_id,
        "type": "interval_each",
        "metric": collection_name,
        "status": status,
        "per_item": per_item,
        "reason": "all required items must pass" if status == PASS else (
            "at least one item is wholly in the failure region"
            if status == FAIL
            else "at least one required item is uncertain"
        ),
    }


def _evaluate_estimate_sum(
    check: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    check_id = str(check["id"])
    collection_name = str(check["collection"])
    values, expected_ids, error = _collection(check, evidence)
    if values is None:
        return _invalid_check(
            check_id, "estimate_sum", collection_name, str(error)
        )

    estimates: dict[str, float] = {}
    for item_id in expected_ids:
        interval, interval_error = _validate_interval(values[item_id])
        if interval is None:
            return _invalid_check(
                check_id,
                "estimate_sum",
                collection_name,
                f"{item_id}: {interval_error}",
            )
        estimates[item_id] = interval["estimate"]

    observed_sum = math.fsum(estimates.values())
    target = float(check["target"])
    tolerance = float(check["absolute_tolerance"])
    passes = abs(observed_sum - target) <= tolerance
    return {
        "id": check_id,
        "type": "estimate_sum",
        "metric": collection_name,
        "status": PASS if passes else FAIL,
        "estimates": estimates,
        "observed_sum": observed_sum,
        "target": target,
        "absolute_tolerance": tolerance,
        "reason": (
            "dispatch estimates sum to target within tolerance"
            if passes
            else "dispatch estimates do not sum to target within tolerance"
        ),
    }


def _evaluate_audited_receipt(
    check: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    gate_id: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    check_id = str(check["id"])
    field = str(check["field"])
    reference = evidence.get(field)
    if not isinstance(reference, Mapping):
        return _invalid_check(
            check_id,
            "audited_receipt",
            field,
            "missing audited receipt reference",
        )
    path_text = reference.get("path")
    expected_sha256 = reference.get("sha256")
    if (
        not isinstance(path_text, str)
        or not path_text.strip()
        or not isinstance(expected_sha256, str)
        or not _SHA256_PATTERN.fullmatch(expected_sha256)
    ):
        return _invalid_check(
            check_id,
            "audited_receipt",
            field,
            "invalid audited receipt path or SHA-256",
        )
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        return _invalid_check(
            check_id,
            "audited_receipt",
            field,
            "audited receipt file is absent",
        )
    try:
        receipt_bytes = path.read_bytes()
        actual_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    except OSError as exc:
        return _invalid_check(
            check_id,
            "audited_receipt",
            field,
            f"cannot hash audited receipt: {exc}",
        )
    if actual_sha256 != expected_sha256:
        return {
            "id": check_id,
            "type": "audited_receipt",
            "metric": field,
            "status": FAIL,
            "path": str(path),
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "reason": "audited receipt SHA-256 mismatch",
        }
    try:
        receipt = json.loads(
            receipt_bytes.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "id": check_id,
            "type": "audited_receipt",
            "metric": field,
            "status": FAIL,
            "path": str(path),
            "sha256": actual_sha256,
            "reason": f"invalid audited receipt JSON: {exc}",
        }
    if not isinstance(receipt, Mapping):
        return {
            "id": check_id,
            "type": "audited_receipt",
            "metric": field,
            "status": FAIL,
            "path": str(path),
            "sha256": actual_sha256,
            "reason": "audited receipt must be an object",
        }
    non_finite = _first_non_finite(receipt, "receipt")
    if non_finite is not None:
        return {
            "id": check_id,
            "type": "audited_receipt",
            "metric": field,
            "status": FAIL,
            "path": str(path),
            "sha256": actual_sha256,
            "reason": f"non-finite audited receipt value at {non_finite}",
        }

    evidence_without_receipt = {
        key: value for key, value in evidence.items() if key != field
    }
    try:
        evidence_sha256 = _policy_sha256(evidence_without_receipt)
    except (TypeError, ValueError) as exc:
        return _invalid_check(
            check_id,
            "audited_receipt",
            field,
            f"evidence cannot be canonically hashed: {exc}",
        )
    expected_values = {
        "schema_version": 1,
        "kind": "mf_wam_gate_audit_receipt",
        "gate_id": gate_id,
        "policy_id": policy["policy_id"],
        "policy_sha256": _policy_sha256(policy),
        "ci_contract_id": check["ci_contract_id"],
        "evidence_sha256": evidence_sha256,
        "terminal": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": receipt.get(key)}
        for key, expected in expected_values.items()
        if receipt.get(key) != expected
    }
    scope = receipt.get("scope")
    if not isinstance(scope, Mapping):
        mismatches["scope"] = {
            "expected": dict(check.get("required_scope", {})),
            "observed": scope,
        }
    else:
        for key, expected in check.get("required_scope", {}).items():
            if scope.get(key) != expected:
                mismatches[f"scope.{key}"] = {
                    "expected": expected,
                    "observed": scope.get(key),
                }

    source_manifest_sha256 = receipt.get("source_manifest_sha256")
    if (
        not isinstance(source_manifest_sha256, str)
        or not _SHA256_PATTERN.fullmatch(source_manifest_sha256)
    ):
        mismatches["source_manifest_sha256"] = {
            "expected": "64 lowercase hexadecimal characters",
            "observed": source_manifest_sha256,
        }
    auditor = receipt.get("auditor")
    if (
        not isinstance(auditor, Mapping)
        or not isinstance(auditor.get("source_commit"), str)
        or not _GIT_COMMIT_PATTERN.fullmatch(auditor["source_commit"])
        or auditor.get("clean") is not True
    ):
        mismatches["auditor"] = {
            "expected": "clean source receipt with a 40-hex commit",
            "observed": auditor,
        }
    artifact_digests = receipt.get("artifact_digests")
    required_digests = check.get("required_artifact_digests", [])
    if not isinstance(artifact_digests, Mapping):
        mismatches["artifact_digests"] = {
            "expected": list(required_digests),
            "observed": artifact_digests,
        }
    else:
        for digest_name in required_digests:
            digest_value = artifact_digests.get(digest_name)
            if (
                not isinstance(digest_value, str)
                or not _SHA256_PATTERN.fullmatch(digest_value)
            ):
                mismatches[f"artifact_digests.{digest_name}"] = {
                    "expected": "64 lowercase hexadecimal characters",
                    "observed": digest_value,
                }
    if mismatches:
        return {
            "id": check_id,
            "type": "audited_receipt",
            "metric": field,
            "status": FAIL,
            "path": str(path),
            "sha256": actual_sha256,
            "reason": "audited receipt does not match the locked contract",
            "mismatches": mismatches,
        }
    return {
        "id": check_id,
        "type": "audited_receipt",
        "metric": field,
        "status": UNCERTAIN,
        "path": str(path),
        "sha256": actual_sha256,
        "envelope_verified": True,
        "reason": (
            "receipt envelope is content-bound, but specialized artifact "
            "recomputation is not implemented"
        ),
    }


def _evaluate_literal(
    literal: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    literal_id = str(literal["id"])
    field = str(literal["field"])
    if field not in evidence:
        return _invalid_check(
            literal_id, "required_literal", field, "missing required field"
        )

    observed = evidence[field]
    expected = literal["expected"]
    if isinstance(observed, float) and not math.isfinite(observed):
        return _invalid_check(
            literal_id, "required_literal", field, "required field is non-finite"
        )
    matches = type(observed) is type(expected) and observed == expected
    status = PASS if matches else str(literal["mismatch_status"])
    return {
        "id": literal_id,
        "type": "required_literal",
        "metric": field,
        "status": status,
        "observed": observed,
        "expected": expected,
        "reason": "required value matches" if matches else "required value mismatch",
    }


def _coerce_policy(policy: Mapping[str, Any] | str | Path | None) -> Mapping[str, Any]:
    if policy is None or isinstance(policy, (str, Path)):
        return load_policy(policy)
    _validate_policy(policy)
    return policy


def evaluate_gate(
    gate_id: str,
    evidence: Mapping[str, Any] | None,
    policy: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate one gate and return a JSON-serializable decision receipt."""

    policy_data = _coerce_policy(policy)
    gates = policy_data["gates"]
    if gate_id not in gates:
        raise KeyError(f"unknown MF-WAM gate: {gate_id}")

    if not isinstance(evidence, Mapping):
        evidence = {}
    gate = gates[gate_id]
    results: list[dict[str, Any]] = []

    non_finite_location = _first_non_finite(evidence)
    if non_finite_location is not None:
        results.append(
            _invalid_check(
                "all_numbers_finite",
                "evidence_contract",
                non_finite_location,
                "non-finite numeric value in evidence receipt",
            )
        )

    literal_specs = list(policy_data["evidence_contract"]["required_literals"])
    literal_specs.extend(gate.get("required_literals", []))
    for literal in literal_specs:
        results.append(_evaluate_literal(literal, evidence))

    evaluators = {
        "interval": _evaluate_interval,
        "interval_each": _evaluate_interval_each,
        "estimate_sum": _evaluate_estimate_sum,
    }
    for check in gate["checks"]:
        if check["type"] == "audited_receipt":
            results.append(
                _evaluate_audited_receipt(
                    check,
                    evidence,
                    gate_id=gate_id,
                    policy=policy_data,
                )
            )
        else:
            results.append(evaluators[check["type"]](check, evidence))

    status = _combine_statuses([result["status"] for result in results])
    policy_digest, policy_is_canonical = _policy_identity(policy_data)
    return {
        "policy_id": policy_data["policy_id"],
        "policy_sha256": policy_digest,
        "policy_is_canonical": policy_is_canonical,
        "gate": gate_id,
        "status": status,
        "checks": results,
        "formal_training_allowed": False,
    }


def evaluate_policy(
    evidence_by_gate: Mapping[str, Mapping[str, Any]] | None,
    policy: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate G0--G3; only four PASS decisions authorize formal training."""

    policy_data = _coerce_policy(policy)
    if not isinstance(evidence_by_gate, Mapping):
        evidence_by_gate = {}

    gate_results = {
        gate_id: evaluate_gate(
            gate_id,
            evidence_by_gate.get(gate_id),
            policy_data,
        )
        for gate_id in policy_data["gates"]
    }
    status = _combine_statuses(
        [gate_result["status"] for gate_result in gate_results.values()]
    )
    gate_thresholds_passed = bool(gate_results) and all(
        result["status"] == PASS for result in gate_results.values()
    )
    policy_digest, policy_is_canonical = _policy_identity(policy_data)
    return {
        "policy_id": policy_data["policy_id"],
        "policy_sha256": policy_digest,
        "policy_is_canonical": policy_is_canonical,
        "status": status,
        "gates": gate_results,
        "gate_thresholds_passed": gate_thresholds_passed,
        "formal_training_allowed": False,
        "authorization_reason": (
            "formal authorization requires specialized audited-bundle "
            "recomputation, which is not implemented"
        ),
    }


def _validate_criterion(spec: Any, location: str) -> None:
    if not isinstance(spec, Mapping):
        raise ValueError(f"{location} must be an object")
    if spec.get("bound") not in ("ci_lower", "ci_upper"):
        raise ValueError(f"{location}.bound is invalid")
    if spec.get("operator") not in _OPERATORS:
        raise ValueError(f"{location}.operator is invalid")
    if _finite_number(spec.get("threshold")) is None:
        raise ValueError(f"{location}.threshold must be finite")


def _validate_valid_range(value: Any, location: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    if not value or any(field not in ("minimum", "maximum") for field in value):
        raise ValueError(f"{location} must contain only minimum/maximum")
    minimum = value.get("minimum")
    maximum = value.get("maximum")
    if minimum is not None and _finite_number(minimum) is None:
        raise ValueError(f"{location}.minimum must be finite")
    if maximum is not None and _finite_number(maximum) is None:
        raise ValueError(f"{location}.maximum must be finite")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{location}.minimum must not exceed maximum")


def _validate_literals(value: Any, location: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    for index, literal in enumerate(value):
        item_location = f"{location}[{index}]"
        if not isinstance(literal, Mapping):
            raise ValueError(f"{item_location} must be an object")
        for field in ("id", "field", "expected", "mismatch_status"):
            if field not in literal:
                raise ValueError(f"{item_location}.{field} is required")
        if literal["mismatch_status"] not in (FAIL, UNCERTAIN):
            raise ValueError(f"{item_location}.mismatch_status is invalid")


def _validate_policy(policy: Any) -> None:
    if not isinstance(policy, Mapping):
        raise ValueError("gate policy must be an object")
    if policy.get("schema_version") != 1:
        raise ValueError("unsupported gate policy schema_version")
    if policy.get("statuses") != [PASS, FAIL, UNCERTAIN]:
        raise ValueError("gate policy statuses must be PASS, FAIL, UNCERTAIN")
    if not isinstance(policy.get("policy_id"), str) or not policy["policy_id"]:
        raise ValueError("gate policy requires a non-empty policy_id")

    contract = policy.get("evidence_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("gate policy requires an evidence_contract")
    if contract.get("interval_fields") != list(_INTERVAL_FIELDS):
        raise ValueError("gate policy interval_fields do not match evaluator contract")
    if contract.get("all_numbers_must_be_finite") is not True:
        raise ValueError("gate policy must require all numbers to be finite")
    _validate_literals(contract.get("required_literals"), "evidence_contract.required_literals")

    gates = policy.get("gates")
    if not isinstance(gates, Mapping) or list(gates) != ["G0", "G1", "G2", "G3"]:
        raise ValueError("gate policy must define G0, G1, G2, G3 in order")
    for gate_id, gate in gates.items():
        if not isinstance(gate, Mapping):
            raise ValueError(f"{gate_id} must be an object")
        if "required_literals" in gate:
            _validate_literals(gate["required_literals"], f"{gate_id}.required_literals")
        checks = gate.get("checks")
        if not isinstance(checks, list) or not checks:
            raise ValueError(f"{gate_id}.checks must be a non-empty list")
        for index, check in enumerate(checks):
            location = f"{gate_id}.checks[{index}]"
            if not isinstance(check, Mapping):
                raise ValueError(f"{location} must be an object")
            if not isinstance(check.get("id"), str) or not check["id"]:
                raise ValueError(f"{location}.id is required")
            check_type = check.get("type")
            if check_type == "interval":
                if not isinstance(check.get("metric"), str):
                    raise ValueError(f"{location}.metric is required")
                if "valid_range" in check:
                    _validate_valid_range(
                        check["valid_range"], f"{location}.valid_range"
                    )
                _validate_criterion(check.get("pass"), f"{location}.pass")
                _validate_criterion(check.get("fail"), f"{location}.fail")
            elif check_type in ("interval_each", "estimate_sum"):
                if not isinstance(check.get("collection"), str):
                    raise ValueError(f"{location}.collection is required")
                expected_ids = check.get("expected_ids")
                if (
                    not isinstance(expected_ids, list)
                    or not expected_ids
                    or any(not isinstance(item, str) or not item for item in expected_ids)
                    or len(set(expected_ids)) != len(expected_ids)
                ):
                    raise ValueError(f"{location}.expected_ids is invalid")
                if check_type == "interval_each":
                    if "valid_range" in check:
                        _validate_valid_range(
                            check["valid_range"], f"{location}.valid_range"
                        )
                    _validate_criterion(check.get("pass"), f"{location}.pass")
                    _validate_criterion(check.get("fail"), f"{location}.fail")
                else:
                    if _finite_number(check.get("target")) is None:
                        raise ValueError(f"{location}.target must be finite")
                    tolerance = _finite_number(check.get("absolute_tolerance"))
                    if tolerance is None or tolerance < 0:
                        raise ValueError(
                            f"{location}.absolute_tolerance must be finite and non-negative"
                        )
            elif check_type == "audited_receipt":
                if not isinstance(check.get("field"), str) or not check["field"]:
                    raise ValueError(f"{location}.field is required")
                if (
                    not isinstance(check.get("ci_contract_id"), str)
                    or not check["ci_contract_id"]
                ):
                    raise ValueError(f"{location}.ci_contract_id is required")
                required_scope = check.get("required_scope", {})
                if not isinstance(required_scope, Mapping):
                    raise ValueError(f"{location}.required_scope must be an object")
                required_digests = check.get("required_artifact_digests")
                if (
                    not isinstance(required_digests, list)
                    or not required_digests
                    or any(
                        not isinstance(item, str) or not item
                        for item in required_digests
                    )
                    or len(set(required_digests)) != len(required_digests)
                ):
                    raise ValueError(
                        f"{location}.required_artifact_digests is invalid"
                    )
            else:
                raise ValueError(f"{location}.type is unsupported")
