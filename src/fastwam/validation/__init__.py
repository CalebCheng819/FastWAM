"""Fail-closed validation contracts for MF-WAM experiments."""

from .mf_wam_gates import (
    FAIL,
    PASS,
    UNCERTAIN,
    default_policy_path,
    evaluate_gate,
    evaluate_policy,
    load_policy,
)

__all__ = [
    "FAIL",
    "PASS",
    "UNCERTAIN",
    "default_policy_path",
    "evaluate_gate",
    "evaluate_policy",
    "load_policy",
]
