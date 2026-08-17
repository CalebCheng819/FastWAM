"""Explicit provenance contracts for RoboFactory normalization statistics."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping

TRAIN_SPLIT = "train_split"
LEGACY_FULL_DATASET = "legacy_full_dataset"
NORMALIZATION_STATS_PROVENANCE_MODES = (TRAIN_SPLIT, LEGACY_FULL_DATASET)


def validate_normalization_stats_provenance(
    payload: Mapping[str, Any], provenance_mode: str
) -> str:
    """Validate one of the two frozen statistics-generation contracts."""

    mode = str(provenance_mode).strip().lower()
    if mode not in NORMALIZATION_STATS_PROVENANCE_MODES:
        raise ValueError(
            "Unsupported normalization-stats provenance mode: "
            f"{provenance_mode!r}"
        )
    cardinality = payload.get("cardinality")
    if not isinstance(cardinality, Mapping) or sorted(
        cardinality.get("agent_counts", [])
    ) != [2, 3, 4]:
        raise ValueError(
            "Normalization stats must cover the exact trained cardinalities [2, 3, 4]"
        )

    if mode == TRAIN_SPLIT:
        fit = payload.get("normalization_fit")
        if not isinstance(fit, Mapping):
            raise ValueError("Normalization stats lack normalization_fit provenance")
        expected_fit = {
            "split": "train",
            "split_seed": 42,
            "val_set_proportion": 0.1,
        }
        for key, expected in expected_fit.items():
            if fit.get(key) != expected:
                raise ValueError(
                    f"Normalization stats {key} mismatch: "
                    f"expected={expected!r} got={fit.get(key)!r}"
                )
        return mode

    if "normalization_fit" in payload:
        raise ValueError(
            "legacy_full_dataset normalization stats must not declare normalization_fit"
        )
    expected_top_level = {
        "source_root": "/cpfs/user/chengjuntao/datasets/robofactory_multi_robot",
        "files": 24,
        "trajectories": 1587,
    }
    for key, expected in expected_top_level.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Legacy normalization stats {key} mismatch: "
                f"expected={expected!r} got={payload.get(key)!r}"
            )
    expected_cardinality = {"2": 562, "3": 802, "4": 223}
    if cardinality.get("trajectories_by_agent_count") != expected_cardinality:
        raise ValueError(
            "Legacy normalization stats cardinality histogram mismatch: "
            f"expected={expected_cardinality!r} "
            f"got={cardinality.get('trajectories_by_agent_count')!r}"
        )
    for kind, expected_count, expected_dim in (
        ("action", 2572601, 8),
        ("state", 2577023, 18),
    ):
        record = payload.get(kind)
        if not isinstance(record, Mapping):
            raise ValueError(f"Legacy normalization stats lack {kind!r} record")
        expected_record_keys = {"count", "max", "mean", "min", "std"}
        if set(record) != expected_record_keys:
            raise ValueError(
                f"Legacy normalization stats {kind} record keys mismatch: "
                f"expected={sorted(expected_record_keys)!r} "
                f"got={sorted(record)!r}"
            )
        count = record.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count != expected_count:
            raise ValueError(
                f"Legacy normalization stats {kind} population mismatch: "
                f"expected_count={expected_count} got_count={record.get('count')!r}"
            )
        for metric in ("max", "mean", "min", "std"):
            values = record.get(metric)
            if not isinstance(values, list) or len(values) != expected_dim:
                raise ValueError(
                    f"Legacy normalization stats {kind}.{metric} dimension mismatch: "
                    f"expected_dim={expected_dim} "
                    f"got_dim={len(values) if isinstance(values, list) else None!r}"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in values
            ):
                raise ValueError(
                    f"Legacy normalization stats {kind}.{metric} must contain "
                    "only finite real numbers"
                )
    return mode


__all__ = [
    "LEGACY_FULL_DATASET",
    "NORMALIZATION_STATS_PROVENANCE_MODES",
    "TRAIN_SPLIT",
    "validate_normalization_stats_provenance",
]
