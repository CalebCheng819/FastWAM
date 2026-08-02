"""Evaluate the preregistered MF-WAM G0--G3 gates without third-party code.

The evaluator deliberately distinguishes lack of evidence from evidence of
failure.  A confidence interval must be wholly inside the accepted region to
pass.  A confidence interval wholly inside the rejected region fails.  An
interval that crosses a boundary, or malformed/incomplete evidence, is
``UNCERTAIN`` and therefore never authorizes formal training.
"""

from __future__ import annotations

import hashlib
import json
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
_STAGE_ORDER = (
    "S1_PROBE_SINGLE_SEED_PILOT",
    "S2_PAIRED_CONFIRMATORY_TRAINING",
    "S3_PAPER_LEVEL_CONFIRMATORY_CONCLUSION",
)
_OUTCOME_PARITY_ONLY = "OUTCOME_PARITY_ONLY"
_STRUCTURAL_PASS_ONLY = "STRUCTURAL_PASS_ONLY"
_SPECIALIZED_G0_PASS = "SPECIALIZED_G0_PASS"


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
        "evidence_classification": _STRUCTURAL_PASS_ONLY,
        "structural_status": _STRUCTURAL_PASS_ONLY,
        "specialized_artifact_recomputation_verified": False,
        "external_anchor_lineage_verified": False,
        "reason": (
            "receipt envelope is content-bound, but this generic evaluator "
            "does not consume specialized recomputation or trusted external anchors"
        ),
    }


def _evaluate_specialized_g0_receipt(
    check: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    gate_id: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a future G0 receipt envelope without accepting its claim.

    A local JSON file can prove only that its own envelope is internally
    consistent.  This generic evaluator deliberately does not consume the
    specialized auditor's artifact recomputation, and the separately trusted
    external-anchor consumer does not exist yet.  Therefore an exactly shaped,
    self-resealed receipt remains ``UNCERTAIN`` and is classified
    ``STRUCTURAL_PASS_ONLY``.
    """

    check_id = str(check["id"])
    check_type = "specialized_audited_receipt"
    field = str(check["field"])
    reference = evidence.get(field)
    if not isinstance(reference, Mapping):
        return _invalid_check(
            check_id,
            check_type,
            field,
            "missing specialized G0 audited receipt reference",
        )
    if set(reference) != {"path", "sha256"}:
        return {
            "id": check_id,
            "type": check_type,
            "metric": field,
            "status": FAIL,
            "reason": "specialized G0 receipt reference fields are not exact",
            "expected_fields": ["path", "sha256"],
            "observed_fields": sorted(reference, key=repr),
        }
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
            check_type,
            field,
            "invalid specialized G0 receipt path or SHA-256",
        )

    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        return _invalid_check(
            check_id,
            check_type,
            field,
            "specialized G0 receipt file is absent",
        )
    try:
        receipt_bytes = path.read_bytes()
        actual_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    except OSError as exc:
        return _invalid_check(
            check_id,
            check_type,
            field,
            f"cannot hash specialized G0 receipt: {exc}",
        )
    if actual_sha256 != expected_sha256:
        return {
            "id": check_id,
            "type": check_type,
            "metric": field,
            "status": FAIL,
            "path": str(path),
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "reason": "specialized G0 receipt SHA-256 mismatch",
        }
    try:
        receipt = json.loads(
            receipt_bytes.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "id": check_id,
            "type": check_type,
            "metric": field,
            "status": FAIL,
            "path": str(path),
            "sha256": actual_sha256,
            "reason": f"invalid specialized G0 receipt JSON: {exc}",
        }
    if not isinstance(receipt, Mapping):
        return {
            "id": check_id,
            "type": check_type,
            "metric": field,
            "status": FAIL,
            "path": str(path),
            "sha256": actual_sha256,
            "reason": "specialized G0 receipt must be an object",
        }
    non_finite = _first_non_finite(receipt, "receipt")
    if non_finite is not None:
        return {
            "id": check_id,
            "type": check_type,
            "metric": field,
            "status": FAIL,
            "path": str(path),
            "sha256": actual_sha256,
            "reason": f"non-finite specialized G0 receipt value at {non_finite}",
        }

    evidence_without_receipt = {
        key: value for key, value in evidence.items() if key != field
    }
    try:
        evidence_sha256 = _policy_sha256(evidence_without_receipt)
    except (TypeError, ValueError) as exc:
        return _invalid_check(
            check_id,
            check_type,
            field,
            f"G0 evidence cannot be canonically hashed: {exc}",
        )

    required_receipt_fields = {
        "schema_version",
        "kind",
        "gate_id",
        "policy_id",
        "policy_sha256",
        "ci_contract_id",
        "evidence_sha256",
        "terminal",
        "scientific_status",
        "formal_training_allowed",
        "source_manifest_sha256",
        "scope",
        "auditor",
        "artifact_digests",
        "anchor_lineage",
    }
    mismatches: dict[str, Any] = {}
    receipt_fields = set(receipt)
    if receipt_fields != required_receipt_fields:
        mismatches["receipt_fields"] = {
            "expected": sorted(required_receipt_fields),
            "observed": sorted(receipt_fields, key=repr),
        }
    expected_values = {
        "schema_version": check["receipt_schema_version"],
        "kind": check["receipt_kind"],
        "gate_id": gate_id,
        "policy_id": policy["policy_id"],
        "policy_sha256": _policy_sha256(policy),
        "ci_contract_id": check["ci_contract_id"],
        "evidence_sha256": evidence_sha256,
        "terminal": True,
        "scientific_status": _SPECIALIZED_G0_PASS,
        "formal_training_allowed": check["required_formal_training_allowed"],
    }
    mismatches.update(
        {
            key: {"expected": expected, "observed": receipt.get(key)}
            for key, expected in expected_values.items()
            if receipt.get(key) != expected
        }
    )

    required_scope = dict(check["required_scope"])
    scope = receipt.get("scope")
    if not isinstance(scope, Mapping) or dict(scope) != required_scope:
        mismatches["scope"] = {
            "expected": required_scope,
            "observed": scope,
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
        or set(auditor) != {"source_commit", "clean"}
        or not isinstance(auditor.get("source_commit"), str)
        or not _GIT_COMMIT_PATTERN.fullmatch(auditor["source_commit"])
        or auditor.get("clean") is not True
    ):
        mismatches["auditor"] = {
            "expected": {
                "source_commit": "40 lowercase hexadecimal characters",
                "clean": True,
            },
            "observed": auditor,
        }

    required_digests = list(check["required_artifact_digests"])
    artifact_digests = receipt.get("artifact_digests")
    if not isinstance(artifact_digests, Mapping):
        mismatches["artifact_digests"] = {
            "expected": required_digests,
            "observed": artifact_digests,
        }
    else:
        if set(artifact_digests) != set(required_digests):
            mismatches["artifact_digest_fields"] = {
                "expected": required_digests,
                "observed": sorted(artifact_digests, key=repr),
            }
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
        if artifact_digests.get("source_manifest_sha256") != source_manifest_sha256:
            mismatches["source_manifest_binding"] = {
                "expected": source_manifest_sha256,
                "observed": artifact_digests.get("source_manifest_sha256"),
            }

    required_anchor_types = list(check["required_external_anchor_types"])
    anchor_lineage = receipt.get("anchor_lineage")
    if not isinstance(anchor_lineage, list):
        mismatches["anchor_lineage"] = {
            "expected": required_anchor_types,
            "observed": anchor_lineage,
        }
    else:
        observed_anchor_types: list[Any] = []
        for index, anchor in enumerate(anchor_lineage):
            anchor_location = f"anchor_lineage[{index}]"
            if not isinstance(anchor, Mapping):
                mismatches[anchor_location] = {
                    "expected": ["anchor_type", "anchor_id", "artifact_sha256"],
                    "observed": anchor,
                }
                continue
            observed_anchor_types.append(anchor.get("anchor_type"))
            if set(anchor) != {"anchor_type", "anchor_id", "artifact_sha256"}:
                mismatches[f"{anchor_location}.fields"] = {
                    "expected": ["anchor_type", "anchor_id", "artifact_sha256"],
                    "observed": sorted(anchor, key=repr),
                }
            if not isinstance(anchor.get("anchor_id"), str) or not anchor[
                "anchor_id"
            ].strip():
                mismatches[f"{anchor_location}.anchor_id"] = {
                    "expected": "non-empty string",
                    "observed": anchor.get("anchor_id"),
                }
            artifact_sha256 = anchor.get("artifact_sha256")
            if (
                not isinstance(artifact_sha256, str)
                or not _SHA256_PATTERN.fullmatch(artifact_sha256)
            ):
                mismatches[f"{anchor_location}.artifact_sha256"] = {
                    "expected": "64 lowercase hexadecimal characters",
                    "observed": artifact_sha256,
                }
        if observed_anchor_types != required_anchor_types:
            mismatches["anchor_lineage.anchor_types"] = {
                "expected": required_anchor_types,
                "observed": observed_anchor_types,
            }

    if mismatches:
        return {
            "id": check_id,
            "type": check_type,
            "metric": field,
            "status": FAIL,
            "path": str(path),
            "sha256": actual_sha256,
            "reason": "specialized G0 receipt does not match the locked contract",
            "mismatches": mismatches,
        }
    return {
        "id": check_id,
        "type": check_type,
        "metric": field,
        "status": UNCERTAIN,
        "path": str(path),
        "sha256": actual_sha256,
        "envelope_verified": True,
        "declared_anchor_lineage_structurally_valid": True,
        "evidence_classification": _STRUCTURAL_PASS_ONLY,
        "structural_status": _STRUCTURAL_PASS_ONLY,
        "claimed_scientific_status": _SPECIALIZED_G0_PASS,
        "specialized_artifact_recomputation_verified": False,
        "external_anchor_lineage_verified": False,
        "reason": (
            "specialized G0 receipt is structurally content-bound, but this "
            "generic evaluator does not consume independently recomputed "
            "artifacts or trusted external anchors"
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
        elif check["type"] == "specialized_audited_receipt":
            results.append(
                _evaluate_specialized_g0_receipt(
                    check,
                    evidence,
                    gate_id=gate_id,
                    policy=policy_data,
                )
            )
        else:
            results.append(evaluators[check["type"]](check, evidence))

    status = _combine_statuses([result["status"] for result in results])
    if gate_id == "G0" and status == PASS:
        # The specialized auditor is a separate CLI, while this generic gate
        # evaluator has no trusted path for consuming its recomputation or the
        # external-anchor verdict. Preserve this fail-closed invariant even if
        # a future receipt parser accidentally returns PASS before that boundary
        # is deliberately implemented and reviewed.
        status = UNCERTAIN
    policy_digest, policy_is_canonical = _policy_identity(policy_data)
    receipt = {
        "policy_id": policy_data["policy_id"],
        "policy_sha256": policy_digest,
        "policy_is_canonical": policy_is_canonical,
        "gate": gate_id,
        "status": status,
        "checks": results,
        "stage_eligibility": False,
        "formal_training_allowed": False,
    }
    if gate_id == "G0":
        specialized_results = [
            result
            for result in results
            if result.get("type") == "specialized_audited_receipt"
        ]
        outcome_results = [
            result
            for result in results
            if result.get("type") != "specialized_audited_receipt"
        ]
        outcome_parity_status = _combine_statuses(
            [result["status"] for result in outcome_results]
        )
        specialized_result = specialized_results[0]
        if (
            outcome_parity_status == PASS
            and specialized_result.get("evidence_classification")
            == _STRUCTURAL_PASS_ONLY
        ):
            evidence_classification = _STRUCTURAL_PASS_ONLY
        elif outcome_parity_status == PASS:
            evidence_classification = _OUTCOME_PARITY_ONLY
        else:
            evidence_classification = "INCOMPLETE_OR_FAILED"
        receipt.update(
            {
                "outcome_parity_status": outcome_parity_status,
                "scientific_gate_status": status,
                "evidence_classification": evidence_classification,
                "specialized_g0_auditor_implemented": False,
                "external_anchor_consumer_implemented": False,
            }
        )
    return receipt


def evaluate_policy(
    evidence_by_gate: Mapping[str, Mapping[str, Any]] | None,
    policy: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate structural evidence; never grant stage or runtime authority."""

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
    stage_eligibility = {
        stage_id: False for stage_id in policy_data["decision_semantics"]["stage_order"]
    }
    policy_digest, policy_is_canonical = _policy_identity(policy_data)
    return {
        "policy_id": policy_data["policy_id"],
        "policy_sha256": policy_digest,
        "policy_is_canonical": policy_is_canonical,
        "status": status,
        "gates": gate_results,
        "gate_thresholds_passed": gate_thresholds_passed,
        "stage_eligibility": stage_eligibility,
        "formal_training_allowed": False,
        "authorization_reason": (
            "all protocol stages and runtime authorization are hard-disabled; "
            "this generic evaluator cannot consume specialized recomputation "
            "or trusted external-anchor verdicts"
        ),
    }


def _require_exact_keys(
    value: Any, expected: set[str], location: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected, key=repr)
        raise ValueError(
            f"{location} fields mismatch; missing={missing}, unknown={unknown}"
        )
    return value


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _validate_string_list(value: Any, location: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{location} must be a non-empty unique string list")
    return value


def _validate_criterion(spec: Any, location: str) -> None:
    spec = _require_exact_keys(
        spec, {"bound", "operator", "threshold"}, location
    )
    if spec["bound"] not in ("ci_lower", "ci_upper"):
        raise ValueError(f"{location}.bound is invalid")
    if spec["operator"] not in _OPERATORS:
        raise ValueError(f"{location}.operator is invalid")
    if _finite_number(spec["threshold"]) is None:
        raise ValueError(f"{location}.threshold must be finite")


def _validate_valid_range(value: Any, location: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{location} must be a non-empty object")
    unknown = set(value) - {"minimum", "maximum"}
    if unknown:
        raise ValueError(f"{location} has unknown fields: {sorted(unknown)}")
    minimum = (
        _finite_number(value["minimum"]) if "minimum" in value else None
    )
    maximum = (
        _finite_number(value["maximum"]) if "maximum" in value else None
    )
    if "minimum" in value and minimum is None:
        raise ValueError(f"{location}.minimum must be finite")
    if "maximum" in value and maximum is None:
        raise ValueError(f"{location}.maximum must be finite")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{location}.minimum must not exceed maximum")


def _validate_literals(value: Any, location: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    literal_ids: list[str] = []
    for index, literal in enumerate(value):
        item_location = f"{location}[{index}]"
        literal = _require_exact_keys(
            literal,
            {"id", "field", "expected", "mismatch_status"},
            item_location,
        )
        literal_ids.append(_require_nonempty_string(literal["id"], f"{item_location}.id"))
        _require_nonempty_string(literal["field"], f"{item_location}.field")
        if literal["mismatch_status"] not in (FAIL, UNCERTAIN):
            raise ValueError(f"{item_location}.mismatch_status is invalid")
    if len(set(literal_ids)) != len(literal_ids):
        raise ValueError(f"{location} contains duplicate ids")


def _validate_gpu_budget(value: Any, stage_id: str) -> None:
    common_fields = {
        "maximum_gpus_per_program",
        "maximum_concurrent_mf_wam_gpus",
        "maximum_concurrent_eight_gpu_programs",
        "live_capacity_and_quota_read_immediately_before_launch",
        "unrelated_jobs_must_not_be_stopped",
    }
    stage_fields = {
        _STAGE_ORDER[0]: set(),
        _STAGE_ORDER[1]: {
            "one_seed_pair_at_a_time",
            "third_program_forbidden_during_eight_plus_eight_wave",
        },
        _STAGE_ORDER[2]: {"evaluation_runs_in_at_most_two_program_waves"},
    }
    budget = _require_exact_keys(
        value, common_fields | stage_fields[stage_id], f"{stage_id}.gpu_budget"
    )
    if (
        type(budget["maximum_gpus_per_program"]) is not int
        or budget["maximum_gpus_per_program"] != 8
    ):
        raise ValueError(f"{stage_id}.gpu_budget maximum per program must be 8")
    if (
        type(budget["maximum_concurrent_mf_wam_gpus"]) is not int
        or budget["maximum_concurrent_mf_wam_gpus"] != 16
    ):
        raise ValueError(f"{stage_id}.gpu_budget concurrent maximum must be 16")
    if (
        type(budget["maximum_concurrent_eight_gpu_programs"]) is not int
        or budget["maximum_concurrent_eight_gpu_programs"] != 2
    ):
        raise ValueError(
            f"{stage_id}.gpu_budget must permit at most two eight-GPU programs"
        )
    boolean_fields = {
        "live_capacity_and_quota_read_immediately_before_launch",
        "unrelated_jobs_must_not_be_stopped",
    } | stage_fields[stage_id]
    if any(budget[field] is not True for field in boolean_fields):
        raise ValueError(f"{stage_id}.gpu_budget safety flags must all be true")


def _validate_stage(stage_id: str, value: Any) -> None:
    fields_by_stage = {
        _STAGE_ORDER[0]: {
            "title",
            "protocol_permission",
            "grants_runtime_authority",
            "entry_rule",
            "required_entry_evidence",
            "allowed_work",
            "pilot_scope",
            "required_exit_evidence",
            "exit_rule",
            "gpu_budget",
            "stop_conditions",
        },
        _STAGE_ORDER[1]: {
            "title",
            "protocol_permission",
            "grants_runtime_authority",
            "entry_rule",
            "required_entry_evidence",
            "training_scope",
            "required_exit_evidence",
            "exit_rule",
            "gpu_budget",
            "stop_conditions",
            "paired_failure_rule",
        },
        _STAGE_ORDER[2]: {
            "title",
            "protocol_permission",
            "grants_runtime_authority",
            "entry_rule",
            "required_evidence",
            "conclusion_rule",
            "gpu_budget",
            "stop_conditions",
            "failure_semantics",
        },
    }
    stage = _require_exact_keys(value, fields_by_stage[stage_id], stage_id)
    if stage["grants_runtime_authority"] is not False:
        raise ValueError(f"{stage_id}.grants_runtime_authority must be false")
    for field in ("title", "protocol_permission", "entry_rule"):
        _require_nonempty_string(stage[field], f"{stage_id}.{field}")
    _validate_string_list(stage["stop_conditions"], f"{stage_id}.stop_conditions")
    _validate_gpu_budget(stage["gpu_budget"], stage_id)

    if stage_id == _STAGE_ORDER[0]:
        required_entry = _validate_string_list(
            stage["required_entry_evidence"],
            f"{stage_id}.required_entry_evidence",
        )
        if "specialized_g0_audit_receipt" not in required_entry:
            raise ValueError(
                f"{stage_id}.required_entry_evidence must include "
                "specialized_g0_audit_receipt"
            )
        _validate_string_list(stage["allowed_work"], f"{stage_id}.allowed_work")
        _validate_string_list(
            stage["required_exit_evidence"],
            f"{stage_id}.required_exit_evidence",
        )
        _require_nonempty_string(stage["exit_rule"], f"{stage_id}.exit_rule")
        pilot_scope = _require_exact_keys(
            stage["pilot_scope"],
            {
                "training_seed_count",
                "maximum_training_steps",
                "hard_task_count",
                "matched_base_seed_count_minimum",
                "matched_base_seed_count_maximum",
                "pilot_contract_frozen_before_outcomes",
                "pilot_go_is_formal_gate_pass",
            },
            f"{stage_id}.pilot_scope",
        )
        expected_pilot_scope = {
            "training_seed_count": 1,
            "maximum_training_steps": 5000,
            "hard_task_count": 5,
            "matched_base_seed_count_minimum": 20,
            "matched_base_seed_count_maximum": 25,
            "pilot_contract_frozen_before_outcomes": True,
            "pilot_go_is_formal_gate_pass": False,
        }
        if dict(pilot_scope) != expected_pilot_scope:
            raise ValueError(f"{stage_id}.pilot_scope does not match locked scope")
        return

    if stage_id == _STAGE_ORDER[1]:
        required_entry = _validate_string_list(
            stage["required_entry_evidence"],
            f"{stage_id}.required_entry_evidence",
        )
        required_receipts = {
            "specialized_g0_audit_receipt",
            "specialized_g1_audit_receipt",
            "specialized_g2_single_seed_pilot_receipt",
            "specialized_g3_single_seed_pilot_receipt",
        }
        if not required_receipts.issubset(required_entry):
            raise ValueError(
                f"{stage_id}.required_entry_evidence lacks specialized receipts"
            )
        _validate_string_list(
            stage["required_exit_evidence"],
            f"{stage_id}.required_exit_evidence",
        )
        for field in ("exit_rule", "paired_failure_rule"):
            _require_nonempty_string(stage[field], f"{stage_id}.{field}")
        training_scope = _require_exact_keys(
            stage["training_scope"],
            {
                "arm_a",
                "arm_b",
                "training_seed_count",
                "paired_wave_order",
                "maximum_confirmatory_steps_per_arm",
                "checkpoint_selection_source",
                "matched_dimensions",
            },
            f"{stage_id}.training_scope",
        )
        if training_scope["arm_a"] != "full_dual_router_mf_wam":
            raise ValueError(f"{stage_id}.training_scope.arm_a is not locked")
        if training_scope["arm_b"] != "state_only_matched_control":
            raise ValueError(f"{stage_id}.training_scope.arm_b is not locked")
        if training_scope["training_seed_count"] != 3:
            raise ValueError(f"{stage_id}.training_scope requires three seeds")
        if training_scope["paired_wave_order"] != ["A0+B0", "A1+B1", "A2+B2"]:
            raise ValueError(f"{stage_id}.training_scope paired wave order is invalid")
        if training_scope["maximum_confirmatory_steps_per_arm"] != 20000:
            raise ValueError(f"{stage_id}.training_scope step cap is invalid")
        if training_scope["checkpoint_selection_source"] != "validation_only":
            raise ValueError(f"{stage_id}.training_scope selection source is invalid")
        _validate_string_list(
            training_scope["matched_dimensions"],
            f"{stage_id}.training_scope.matched_dimensions",
        )
        return

    required_evidence = _validate_string_list(
        stage["required_evidence"], f"{stage_id}.required_evidence"
    )
    required_receipts = {
        "specialized_g0_audit_receipt",
        "specialized_g1_audit_receipt",
        "specialized_g2_audit_receipt",
        "specialized_g3_audit_receipt",
    }
    if not required_receipts.issubset(required_evidence):
        raise ValueError(f"{stage_id}.required_evidence lacks specialized receipts")
    _require_nonempty_string(stage["conclusion_rule"], f"{stage_id}.conclusion_rule")
    failure_semantics = _require_exact_keys(
        stage["failure_semantics"],
        {"technical_missingness", "threshold_failure"},
        f"{stage_id}.failure_semantics",
    )
    for field in failure_semantics:
        _require_nonempty_string(
            failure_semantics[field], f"{stage_id}.failure_semantics.{field}"
        )


def _validate_decision_semantics(value: Any) -> None:
    semantics = _require_exact_keys(
        value,
        {
            "runtime_authorization",
            "stage_order",
            "g0_evidence_classifications",
            "paper_level_conclusion_rule",
            "gate_precedence",
            "missing_or_non_finite",
            "confidence_interval_crosses_boundary",
            "confidence_interval_entirely_in_failure_region",
        },
        "decision_semantics",
    )
    runtime = _require_exact_keys(
        semantics["runtime_authorization"],
        {"state", "implemented", "formal_training_allowed", "reason"},
        "decision_semantics.runtime_authorization",
    )
    if runtime["state"] != "HARD_DISABLED":
        raise ValueError("runtime authorization state must be HARD_DISABLED")
    if runtime["implemented"] is not False:
        raise ValueError("runtime authorization must remain unimplemented")
    if runtime["formal_training_allowed"] is not False:
        raise ValueError("runtime authorization must remain hard-false")
    _require_nonempty_string(
        runtime["reason"], "decision_semantics.runtime_authorization.reason"
    )
    if semantics["stage_order"] != list(_STAGE_ORDER):
        raise ValueError("decision_semantics.stage_order must be the exact S1/S2/S3 order")
    classifications = _require_exact_keys(
        semantics["g0_evidence_classifications"],
        {
            _OUTCOME_PARITY_ONLY,
            _STRUCTURAL_PASS_ONLY,
            _SPECIALIZED_G0_PASS,
            "interchangeable",
        },
        "decision_semantics.g0_evidence_classifications",
    )
    for classification in (
        _OUTCOME_PARITY_ONLY,
        _STRUCTURAL_PASS_ONLY,
        _SPECIALIZED_G0_PASS,
    ):
        _require_nonempty_string(
            classifications[classification],
            f"decision_semantics.g0_evidence_classifications.{classification}",
        )
    if classifications["interchangeable"] is not False:
        raise ValueError("G0 evidence classifications must not be interchangeable")
    _require_nonempty_string(
        semantics["paper_level_conclusion_rule"],
        "decision_semantics.paper_level_conclusion_rule",
    )
    if semantics["gate_precedence"] != [FAIL, UNCERTAIN, PASS]:
        raise ValueError("decision_semantics.gate_precedence is invalid")
    expected_status_fields = {
        "missing_or_non_finite": UNCERTAIN,
        "confidence_interval_crosses_boundary": UNCERTAIN,
        "confidence_interval_entirely_in_failure_region": FAIL,
    }
    for field, expected in expected_status_fields.items():
        if semantics[field] != expected:
            raise ValueError(f"decision_semantics.{field} must be {expected}")


def _validate_receipt_check(check: Mapping[str, Any], location: str) -> None:
    _require_nonempty_string(check["field"], f"{location}.field")
    _require_nonempty_string(check["ci_contract_id"], f"{location}.ci_contract_id")
    if not isinstance(check["required_scope"], Mapping):
        raise ValueError(f"{location}.required_scope must be an object")
    if _first_non_finite(check["required_scope"], f"{location}.required_scope"):
        raise ValueError(f"{location}.required_scope must contain finite values")
    _validate_string_list(
        check["required_artifact_digests"],
        f"{location}.required_artifact_digests",
    )


def _validate_policy(policy: Any) -> None:
    policy = _require_exact_keys(
        policy,
        {
            "schema_version",
            "policy_id",
            "iteration_id",
            "statuses",
            "decision_semantics",
            "authorization_stages",
            "evidence_contract",
            "gates",
        },
        "gate policy",
    )
    if policy["schema_version"] != 1:
        raise ValueError("unsupported gate policy schema_version")
    if policy["statuses"] != [PASS, FAIL, UNCERTAIN]:
        raise ValueError("gate policy statuses must be PASS, FAIL, UNCERTAIN")
    _require_nonempty_string(policy["policy_id"], "gate policy.policy_id")
    _require_nonempty_string(policy["iteration_id"], "gate policy.iteration_id")
    _validate_decision_semantics(policy["decision_semantics"])

    stages = _require_exact_keys(
        policy["authorization_stages"],
        set(_STAGE_ORDER),
        "authorization_stages",
    )
    if list(stages) != list(_STAGE_ORDER):
        raise ValueError("authorization_stages must be in the exact S1/S2/S3 order")
    for stage_id in _STAGE_ORDER:
        _validate_stage(stage_id, stages[stage_id])

    contract = _require_exact_keys(
        policy["evidence_contract"],
        {
            "interval_fields",
            "interval_invariant",
            "all_numbers_must_be_finite",
            "required_literals",
        },
        "evidence_contract",
    )
    if contract["interval_fields"] != list(_INTERVAL_FIELDS):
        raise ValueError("gate policy interval_fields do not match evaluator contract")
    if contract["interval_invariant"] != "ci_lower <= estimate <= ci_upper":
        raise ValueError("gate policy interval invariant is invalid")
    if contract["all_numbers_must_be_finite"] is not True:
        raise ValueError("gate policy must require all numbers to be finite")
    _validate_literals(
        contract["required_literals"], "evidence_contract.required_literals"
    )

    gates = _require_exact_keys(
        policy["gates"], {"G0", "G1", "G2", "G3"}, "gates"
    )
    if list(gates) != ["G0", "G1", "G2", "G3"]:
        raise ValueError("gate policy must define G0, G1, G2, G3 in order")
    for gate_id, gate_value in gates.items():
        gate = _require_exact_keys(
            gate_value, {"title", "required_literals", "checks"}, gate_id
        )
        _require_nonempty_string(gate["title"], f"{gate_id}.title")
        _validate_literals(gate["required_literals"], f"{gate_id}.required_literals")
        checks = gate["checks"]
        if not isinstance(checks, list) or not checks:
            raise ValueError(f"{gate_id}.checks must be a non-empty list")
        check_ids: list[str] = []
        specialized_count = 0
        generic_receipt_count = 0
        for index, check_value in enumerate(checks):
            location = f"{gate_id}.checks[{index}]"
            if not isinstance(check_value, Mapping):
                raise ValueError(f"{location} must be an object")
            check_type = check_value.get("type")
            fields_by_type = {
                "interval": {"id", "type", "metric", "valid_range", "pass", "fail"},
                "interval_each": {
                    "id",
                    "type",
                    "collection",
                    "expected_ids",
                    "valid_range",
                    "pass",
                    "fail",
                },
                "estimate_sum": {
                    "id",
                    "type",
                    "collection",
                    "expected_ids",
                    "target",
                    "absolute_tolerance",
                },
                "audited_receipt": {
                    "id",
                    "type",
                    "field",
                    "ci_contract_id",
                    "required_scope",
                    "required_artifact_digests",
                },
                "specialized_audited_receipt": {
                    "id",
                    "type",
                    "field",
                    "receipt_kind",
                    "receipt_schema_version",
                    "required_formal_training_allowed",
                    "ci_contract_id",
                    "required_scope",
                    "required_artifact_digests",
                    "required_external_anchor_types",
                },
            }
            if check_type not in fields_by_type:
                raise ValueError(f"{location}.type is unsupported")
            check = _require_exact_keys(
                check_value, fields_by_type[check_type], location
            )
            check_ids.append(_require_nonempty_string(check["id"], f"{location}.id"))

            if check_type == "interval":
                _require_nonempty_string(check["metric"], f"{location}.metric")
                _validate_valid_range(check["valid_range"], f"{location}.valid_range")
                _validate_criterion(check["pass"], f"{location}.pass")
                _validate_criterion(check["fail"], f"{location}.fail")
            elif check_type == "interval_each":
                _require_nonempty_string(
                    check["collection"], f"{location}.collection"
                )
                _validate_string_list(check["expected_ids"], f"{location}.expected_ids")
                _validate_valid_range(check["valid_range"], f"{location}.valid_range")
                _validate_criterion(check["pass"], f"{location}.pass")
                _validate_criterion(check["fail"], f"{location}.fail")
            elif check_type == "estimate_sum":
                _require_nonempty_string(
                    check["collection"], f"{location}.collection"
                )
                _validate_string_list(check["expected_ids"], f"{location}.expected_ids")
                if _finite_number(check["target"]) is None:
                    raise ValueError(f"{location}.target must be finite")
                tolerance = _finite_number(check["absolute_tolerance"])
                if tolerance is None or tolerance < 0:
                    raise ValueError(
                        f"{location}.absolute_tolerance must be finite and non-negative"
                    )
            elif check_type == "audited_receipt":
                generic_receipt_count += 1
                _validate_receipt_check(check, location)
            else:
                specialized_count += 1
                _validate_receipt_check(check, location)
                if gate_id != "G0":
                    raise ValueError("specialized G0 receipt check may only appear in G0")
                if check["field"] != "specialized_g0_audit_receipt":
                    raise ValueError(f"{location}.field must bind specialized G0 receipt")
                if check["receipt_kind"] != "mf_wam_g0_specialized_audit_receipt":
                    raise ValueError(f"{location}.receipt_kind is invalid")
                if check["receipt_schema_version"] != 1:
                    raise ValueError(f"{location}.receipt_schema_version is invalid")
                if check["required_formal_training_allowed"] is not False:
                    raise ValueError(
                        f"{location}.required_formal_training_allowed must be false"
                    )
                if check["ci_contract_id"] != "MF-WAM-G0-CI-v1":
                    raise ValueError(f"{location}.ci_contract_id is invalid")
                expected_scope = {
                    "episode_count": 2000,
                    "suite_count": 4,
                    "tasks_per_suite": 10,
                    "trials_per_task": 50,
                    "confidence_level": 0.95,
                    "bootstrap_replicates": 10000,
                    "bootstrap_seed": 42,
                    "outcome_parity_classification": _OUTCOME_PARITY_ONLY,
                }
                if dict(check["required_scope"]) != expected_scope:
                    raise ValueError(f"{location}.required_scope is not locked")
                expected_digests = [
                    "source_manifest_sha256",
                    "data_manifest_sha256",
                    "seed_manifest_sha256",
                    "resolved_config_sha256",
                    "checkpoint_sha256",
                    "dataset_stats_sha256",
                    "runtime_environment_sha256",
                    "identity_inventory_sha256",
                    "metric_rows_sha256",
                    "trace_tree_sha256",
                    "terminal_summary_bundle_sha256",
                ]
                if check["required_artifact_digests"] != expected_digests:
                    raise ValueError(f"{location}.required_artifact_digests is not locked")
                expected_anchor_types = [
                    "notion_experiment_page",
                    "immutable_artifact_root",
                    "source_commit",
                    "container_image_digest",
                ]
                if check["required_external_anchor_types"] != expected_anchor_types:
                    raise ValueError(
                        f"{location}.required_external_anchor_types is not locked"
                    )
        if len(set(check_ids)) != len(check_ids):
            raise ValueError(f"{gate_id}.checks contains duplicate ids")
        if gate_id == "G0":
            if specialized_count != 1 or generic_receipt_count != 0:
                raise ValueError("G0 requires exactly one specialized receipt check")
            if checks[-1]["type"] != "specialized_audited_receipt":
                raise ValueError("G0 specialized receipt check must be last")
        elif specialized_count != 0 or generic_receipt_count != 1:
            raise ValueError(f"{gate_id} requires exactly one generic audited receipt")
