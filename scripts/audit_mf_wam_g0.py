#!/usr/bin/env python3
"""Legacy diagnostic audit of a paired FastWAM LIBERO reproduction.

The audit separates outcome reproduction from artifact binding.  A candidate
can reproduce all episode outcomes while the full G0 gate remains UNCERTAIN if
the checkpoint, statistics, or resolved configuration cannot be hashed in the
current audit.  This module predates the canonical schema-v2 trace/receipt
contract and MUST NOT be used for scientific G0 authorization.  Its historical
``status`` field is retained only for compatibility; every result explicitly
keeps ``scientific_gate_status=UNCERTAIN`` and
``formal_training_allowed=false``.  Use ``audit_mf_wam_g0_bundle.py`` for the
canonical specialized audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASKS_PER_SUITE = 10
TRIALS_PER_TASK = 50
EXPECTED_EPISODES = len(SUITES) * TASKS_PER_SUITE * TRIALS_PER_TASK
OVERALL_EQUIVALENCE_MARGIN = 0.02
SUITE_DROP_MARGIN = 0.03
MIN_BOOTSTRAP_REPLICATES = 10_000
PREREGISTERED_BOOTSTRAP_SEED = 42
EXPECTED_REPLAN_STEPS = 10
EXPECTED_ACTION_HORIZON = 32
EXPECTED_ACTION_DIMENSION = 7
EXPECTED_STATE_DIMENSION = 8
EXPECTED_FIRST_REPLAN_ENV_STEP = 30
MIN_TRACE_RECORDS_PER_EPISODE = 7
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class AuditFailure(ValueError):
    """Raised when an artifact violates the declared G0 contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_baseline_policy_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "validation"
        / "mf_wam_g0_baseline.json"
    )


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"cannot read JSON {path}: {exc}") from exc


def _load_baseline_policy() -> tuple[dict[str, Any], str, Path]:
    path = _default_baseline_policy_path()
    payload = _load_json(path)
    _assert_finite(payload, str(path))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise AuditFailure("invalid G0 baseline policy schema")
    if payload.get("source_commit") != "45d8e1458921d83f8ad6cf9ce993d371208dabd0":
        raise AuditFailure("unexpected G0 baseline source commit")
    reference = payload.get("reference_run")
    model_artifacts = payload.get("model_artifacts")
    candidate_contract = payload.get("candidate_contract")
    scope = payload.get("evaluation_scope")
    if not all(
        isinstance(item, dict)
        for item in (reference, model_artifacts, candidate_contract, scope)
    ):
        raise AuditFailure("incomplete G0 baseline policy")
    contract_status = candidate_contract.get("status")
    if contract_status == "UNREGISTERED":
        if (
            not isinstance(candidate_contract.get("reason"), str)
            or not candidate_contract["reason"].strip()
        ):
            raise AuditFailure("unregistered G0 candidate contract requires a reason")
    elif contract_status == "LOCKED":
        required_contract_fields = (
            "dataset_id",
            "data_revision",
            "data_inventory_sha256",
            "seed_tree_sha256",
            "resolved_config_sha256",
            "image_digest",
            "python",
            "cuda",
            "pytorch",
            "mujoco",
            "libero",
        )
        if any(field not in candidate_contract for field in required_contract_fields):
            raise AuditFailure("locked G0 candidate contract is incomplete")
        for field in (
            "data_inventory_sha256",
            "seed_tree_sha256",
            "resolved_config_sha256",
        ):
            value = candidate_contract[field]
            if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
                raise AuditFailure(f"invalid candidate contract digest: {field}")
        if (
            not isinstance(candidate_contract["image_digest"], str)
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", candidate_contract["image_digest"]
            )
        ):
            raise AuditFailure("invalid candidate contract image digest")
        for field in (
            "dataset_id",
            "data_revision",
            "python",
            "cuda",
            "pytorch",
            "mujoco",
            "libero",
        ):
            value = candidate_contract[field]
            if not isinstance(value, str) or not value.strip():
                raise AuditFailure(f"invalid candidate contract field: {field}")
    else:
        raise AuditFailure("G0 candidate contract status must be LOCKED or UNREGISTERED")
    digest_fields = (
        reference.get("summary_csv_sha256"),
        reference.get("task_success_rates_csv_sha256"),
        reference.get("summary_json_sha256"),
        reference.get("task_result_tree_sha256"),
        model_artifacts.get("checkpoint_sha256"),
        model_artifacts.get("dataset_stats_sha256"),
    )
    if any(not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value) for value in digest_fields):
        raise AuditFailure("invalid SHA-256 identity in G0 baseline policy")
    expected_scope = {
        "suites": list(SUITES),
        "tasks_per_suite": TASKS_PER_SUITE,
        "trials_per_task": TRIALS_PER_TASK,
        "overall_equivalence_margin": OVERALL_EQUIVALENCE_MARGIN,
        "suite_drop_margin": SUITE_DROP_MARGIN,
        "minimum_bootstrap_replicates": MIN_BOOTSTRAP_REPLICATES,
        "bootstrap_seed": PREREGISTERED_BOOTSTRAP_SEED,
        "replan_steps": EXPECTED_REPLAN_STEPS,
        "action_horizon": EXPECTED_ACTION_HORIZON,
        "action_dimension": EXPECTED_ACTION_DIMENSION,
        "state_dimension": EXPECTED_STATE_DIMENSION,
        "first_replan_env_step": EXPECTED_FIRST_REPLAN_ENV_STEP,
        "minimum_trace_records_per_episode": MIN_TRACE_RECORDS_PER_EPISODE,
    }
    if scope != expected_scope:
        raise AuditFailure("G0 baseline evaluation scope does not match code contract")
    return payload, _sha256(path), path


def _assert_finite(value: Any, location: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise AuditFailure(f"non-finite numeric value at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{location}.{key}")
        return
    raise AuditFailure(f"unsupported value type at {location}: {type(value).__name__}")


def _assert_numeric_vector(value: Any, location: str, expected_size: int) -> None:
    if not isinstance(value, list) or len(value) != expected_size:
        raise AuditFailure(
            f"expected numeric vector of size {expected_size} at {location}"
        )
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise AuditFailure(f"non-numeric value at {location}[{index}]")
        if not math.isfinite(float(item)):
            raise AuditFailure(f"non-finite numeric value at {location}[{index}]")


def _find_task_result(root: Path, suite: str, task_id: int) -> Path:
    matches = sorted((root / suite).glob(f"gpu*_task{task_id}_results.json"))
    if len(matches) != 1:
        raise AuditFailure(
            f"expected exactly one result for {suite}/task{task_id}, found {len(matches)}"
        )
    return matches[0]


def _task_result_tree_sha256(root: Path) -> str:
    paths = sorted(
        _find_task_result(root, suite, task_id)
        for suite in SUITES
        for task_id in range(TASKS_PER_SUITE)
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(f"{_sha256(path)}  {relative}\n".encode("utf-8"))
    return digest.hexdigest()


def _load_episode_map(root: Path) -> tuple[dict[tuple[str, int, int], bool], dict[str, float]]:
    episodes: dict[tuple[str, int, int], bool] = {}
    suite_rates: dict[str, float] = {}
    for suite in SUITES:
        suite_successes = 0
        for task_id in range(TASKS_PER_SUITE):
            path = _find_task_result(root, suite, task_id)
            payload = _load_json(path)
            _assert_finite(payload, str(path))
            if payload.get("task_suite") != suite or payload.get("task_id") != task_id:
                raise AuditFailure(f"task identity mismatch in {path}")
            if payload.get("total_episodes") != TRIALS_PER_TASK:
                raise AuditFailure(f"wrong episode count in {path}")

            successes_raw = payload.get("success_episodes")
            failures_raw = payload.get("failure_episodes")
            if not isinstance(successes_raw, list) or not isinstance(failures_raw, list):
                raise AuditFailure(f"missing success/failure episode lists in {path}")
            if any(type(item) is not int for item in successes_raw + failures_raw):
                raise AuditFailure(f"non-integer episode identity in {path}")
            successes = set(successes_raw)
            failures = set(failures_raw)
            expected = set(range(TRIALS_PER_TASK))
            if len(successes) != len(successes_raw) or len(failures) != len(failures_raw):
                raise AuditFailure(f"duplicate episode identity in {path}")
            if successes & failures or successes | failures != expected:
                raise AuditFailure(f"episode partition is not exact in {path}")
            if payload.get("successes") != len(successes):
                raise AuditFailure(f"success count mismatch in {path}")

            suite_successes += len(successes)
            for trial in range(TRIALS_PER_TASK):
                key = (suite, task_id, trial)
                if key in episodes:
                    raise AuditFailure(f"duplicate episode key {key}")
                episodes[key] = trial in successes
        suite_rates[suite] = suite_successes / (TASKS_PER_SUITE * TRIALS_PER_TASK)

    if len(episodes) != EXPECTED_EPISODES:
        raise AuditFailure(f"expected {EXPECTED_EPISODES} episodes, found {len(episodes)}")
    return episodes, suite_rates


def _audit_task_success_csv(root: Path, episodes: dict[tuple[str, int, int], bool]) -> str:
    path = root / "task_success_rates.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(SUITES) * TASKS_PER_SUITE:
        raise AuditFailure(f"expected 40 task CSV rows in {path}, found {len(rows)}")
    seen: set[tuple[str, int]] = set()
    for row in rows:
        task_name = row.get("Task", "")
        matches = [suite for suite in SUITES if task_name.startswith(f"{suite}_")]
        if len(matches) != 1:
            raise AuditFailure(f"invalid task name {task_name!r} in {path}")
        suite = matches[0]
        try:
            task_id = int(task_name.removeprefix(f"{suite}_"))
            reported = float(row["Success Rate (%)"]) / 100.0
        except (KeyError, ValueError) as exc:
            raise AuditFailure(f"invalid task row {row!r} in {path}") from exc
        key = (suite, task_id)
        if key in seen or not 0 <= task_id < TASKS_PER_SUITE:
            raise AuditFailure(f"duplicate or out-of-range task {key} in {path}")
        seen.add(key)
        actual = sum(episodes[(suite, task_id, trial)] for trial in range(TRIALS_PER_TASK))
        if not math.isclose(reported, actual / TRIALS_PER_TASK, abs_tol=1e-12):
            raise AuditFailure(f"task CSV success mismatch for {key}")
    return _sha256(path)


def _audit_summary_json(root: Path, suite_rates: dict[str, float]) -> str:
    path = root / "summary.json"
    payload = _load_json(path)
    _assert_finite(payload, str(path))
    stats = payload.get("suite_stats")
    if not isinstance(stats, dict):
        raise AuditFailure(f"missing suite_stats in {path}")
    for suite in SUITES:
        suite_stat = stats.get(suite)
        if not isinstance(suite_stat, dict):
            raise AuditFailure(f"missing {suite} suite stats in {path}")
        if suite_stat.get("total_tasks") != TASKS_PER_SUITE:
            raise AuditFailure(f"wrong task count for {suite} in {path}")
        if suite_stat.get("total_trials") != TASKS_PER_SUITE * TRIALS_PER_TASK:
            raise AuditFailure(f"wrong trial count for {suite} in {path}")
        successes = suite_stat.get("total_successes")
        if successes != round(suite_rates[suite] * TASKS_PER_SUITE * TRIALS_PER_TASK):
            raise AuditFailure(f"suite success mismatch for {suite} in {path}")
    return _sha256(path)


def _audit_summary_csv(root: Path, suite_rates: dict[str, float]) -> str:
    path = root / "summary.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    try:
        header_index = next(index for index, row in enumerate(rows) if row and row[0] == "")
    except StopIteration as exc:
        raise AuditFailure(f"missing summary header in {path}") from exc
    header = rows[header_index]
    if header[1:] != [*SUITES, "Overall"]:
        raise AuditFailure(f"unexpected suite order in {path}: {header[1:]}")
    if header_index + 1 >= len(rows) or rows[header_index + 1][0] != "Success Rate (%)":
        raise AuditFailure(f"missing success-rate row in {path}")
    values = rows[header_index + 1][1:]
    if len(values) != len(SUITES) + 1:
        raise AuditFailure(f"wrong success-rate width in {path}")
    reported = [float(value) / 100.0 for value in values]
    expected = [suite_rates[suite] for suite in SUITES]
    expected.append(sum(expected) / len(expected))
    if any(not math.isclose(left, right, abs_tol=1e-12) for left, right in zip(reported, expected)):
        raise AuditFailure(f"summary CSV success mismatch in {path}")
    return _sha256(path)


def _audit_traces(
    trace_root: Path,
    episodes: dict[tuple[str, int, int], bool],
    expected_seed_map: dict[
        tuple[str, int, int], dict[str, int]
    ] | None,
    expected_replan_steps: int,
    expected_action_horizon: int,
    expected_action_dimension: int,
    expected_state_dimension: int,
    expected_first_replan_env_step: int,
    minimum_trace_records_per_episode: int,
) -> dict[str, Any]:
    observed_paths = sorted(trace_root.glob("**/*.json"))
    if len(observed_paths) != EXPECTED_EPISODES:
        raise AuditFailure(
            f"expected {EXPECTED_EPISODES} trace JSON files, found {len(observed_paths)}"
        )
    expected_paths: set[Path] = set()
    total_records = 0
    missing_seed_binding_count = 0
    digest = hashlib.sha256()
    for key, success in sorted(episodes.items()):
        suite, task_id, trial = key
        path = trace_root / suite / f"task{task_id}_trial{trial}.json"
        expected_paths.add(path)
        if not path.is_file():
            raise AuditFailure(f"missing trace {path}")
        payload = _load_json(path)
        _assert_finite(payload, str(path))
        metadata = payload.get("metadata")
        records = payload.get("records")
        if not isinstance(metadata, dict) or not isinstance(records, list) or not records:
            raise AuditFailure(f"invalid trace payload in {path}")
        if len(records) < minimum_trace_records_per_episode:
            raise AuditFailure(
                "insufficient temporal coverage in "
                f"{path}: expected at least {minimum_trace_records_per_episode} "
                f"records, found {len(records)}"
            )
        expected_metadata = {
            "task_suite": suite,
            "task_id": task_id,
            "trial_idx": trial,
            "success": success,
            "replan_steps": expected_replan_steps,
            "action_horizon": expected_action_horizon,
        }
        for field, expected_value in expected_metadata.items():
            if metadata.get(field) != expected_value:
                raise AuditFailure(f"trace metadata mismatch {field} in {path}")
        if expected_seed_map is None:
            missing_seed_binding_count += 1
        else:
            expected_seeds = expected_seed_map[key]
            missing_seed_fields = [
                field for field in expected_seeds if field not in metadata
            ]
            for field, expected_seed in expected_seeds.items():
                if field not in metadata:
                    continue
                observed_seed = metadata[field]
                if type(observed_seed) is not int or observed_seed != expected_seed:
                    raise AuditFailure(
                        f"trace seed mismatch {field} in {path}"
                    )
            if missing_seed_fields:
                missing_seed_binding_count += 1
        previous_env_step = -1
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise AuditFailure(f"non-object record {index} in {path}")
            required = {"env_step", "episode_idx", "replan_idx", "raw_action_chunk", "state"}
            if not required.issubset(record):
                raise AuditFailure(f"missing record fields at {path}:{index}")
            if record["episode_idx"] != trial or type(record["episode_idx"]) is not int:
                raise AuditFailure(f"episode identity mismatch at {path}:{index}")
            if record["replan_idx"] != index or type(record["replan_idx"]) is not int:
                raise AuditFailure(f"replan index mismatch at {path}:{index}")
            env_step = record["env_step"]
            if type(env_step) is not int or env_step < 0 or env_step <= previous_env_step:
                raise AuditFailure(f"invalid or non-increasing env_step at {path}:{index}")
            if index == 0 and env_step != expected_first_replan_env_step:
                raise AuditFailure(
                    f"first replan env_step mismatch at {path}:{index}"
                )
            if index > 0 and env_step - previous_env_step != expected_replan_steps:
                raise AuditFailure(f"replan cadence mismatch at {path}:{index}")
            previous_env_step = env_step
            if "task_suite" in record and record["task_suite"] != suite:
                raise AuditFailure(f"record suite mismatch at {path}:{index}")
            if "task_id" in record and record["task_id"] != task_id:
                raise AuditFailure(f"record task mismatch at {path}:{index}")
            action_chunk = record["raw_action_chunk"]
            if not isinstance(action_chunk, list) or not action_chunk:
                raise AuditFailure(f"empty action chunk at {path}:{index}")
            if len(action_chunk) != expected_replan_steps:
                raise AuditFailure(f"action chunk length mismatch at {path}:{index}")
            for action_index, action in enumerate(action_chunk):
                _assert_numeric_vector(
                    action,
                    f"{path}:{index}.raw_action_chunk[{action_index}]",
                    expected_action_dimension,
                )
            _assert_numeric_vector(
                record["state"],
                f"{path}:{index}.state",
                expected_state_dimension,
            )
        total_records += len(records)
        relative = path.relative_to(trace_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
        digest.update(b"\n")
    if set(observed_paths) != expected_paths:
        extras = sorted(str(path) for path in set(observed_paths) - expected_paths)
        raise AuditFailure(f"unexpected trace paths: {extras[:5]}")
    return {
        "episode_count": len(observed_paths),
        "record_count": total_records,
        "tree_sha256": digest.hexdigest(),
        "missing_trace_count": 0,
        "non_finite_count": 0,
        "first_replan_env_step": expected_first_replan_env_step,
        "minimum_records_per_episode": minimum_trace_records_per_episode,
        "seed_binding_status": (
            "PASS" if missing_seed_binding_count == 0 else "UNCERTAIN"
        ),
        "missing_seed_binding_count": missing_seed_binding_count,
    }


def _task_bootstrap_ci(
    reference: dict[tuple[str, int, int], bool],
    candidate: dict[tuple[str, int, int], bool],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    task_differences = []
    for suite in SUITES:
        for task_id in range(TASKS_PER_SUITE):
            difference = sum(
                int(candidate[(suite, task_id, trial)]) - int(reference[(suite, task_id, trial)])
                for trial in range(TRIALS_PER_TASK)
            ) / TRIALS_PER_TASK
            task_differences.append(difference)
    if not any(task_differences):
        return 0.0, 0.0
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choice(task_differences) for _ in task_differences) / len(task_differences)
        for _ in range(replicates)
    )
    lower_index = max(0, math.floor(0.025 * replicates))
    upper_index = min(replicates - 1, math.ceil(0.975 * replicates) - 1)
    return samples[lower_index], samples[upper_index]


def _suite_bootstrap_intervals(
    reference: dict[tuple[str, int, int], bool],
    candidate: dict[tuple[str, int, int], bool],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    intervals: dict[str, dict[str, float]] = {}
    for suite_index, suite in enumerate(SUITES):
        task_differences = [
            sum(
                int(candidate[(suite, task_id, trial)])
                - int(reference[(suite, task_id, trial)])
                for trial in range(TRIALS_PER_TASK)
            )
            / TRIALS_PER_TASK
            for task_id in range(TASKS_PER_SUITE)
        ]
        estimate = sum(task_differences) / len(task_differences)
        if not any(task_differences):
            lower, upper = 0.0, 0.0
        else:
            rng = random.Random(seed + suite_index)
            samples = sorted(
                sum(rng.choice(task_differences) for _ in task_differences)
                / len(task_differences)
                for _ in range(replicates)
            )
            lower_index = max(0, math.floor(0.025 * replicates))
            upper_index = min(
                replicates - 1, math.ceil(0.975 * replicates) - 1
            )
            lower, upper = samples[lower_index], samples[upper_index]
        intervals[suite] = {
            "estimate": estimate,
            "ci_lower": lower,
            "ci_upper": upper,
        }
    return intervals


def _bind_file(path_text: str | None, expected_sha256: str | None, label: str) -> dict[str, Any]:
    if path_text is None or expected_sha256 is None:
        return {"status": "UNCERTAIN", "reason": f"{label} file/hash not both supplied"}
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        return {"status": "UNCERTAIN", "path": str(path), "reason": f"{label} file is absent"}
    actual = _sha256(path)
    if actual != expected_sha256.lower():
        return {
            "status": "FAIL",
            "path": str(path),
            "expected_sha256": expected_sha256.lower(),
            "actual_sha256": actual,
        }
    return {"status": "PASS", "path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}


def _bind_existing_file(path_text: str | None, label: str) -> dict[str, Any]:
    if path_text is None:
        return {"status": "UNCERTAIN", "reason": f"{label} file not supplied"}
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        return {
            "status": "UNCERTAIN",
            "path": str(path),
            "reason": f"{label} file is absent",
        }
    return {
        "status": "PASS",
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_bound_manifest(
    path_text: str | None,
    *,
    label: str,
    expected_kind: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    binding = _bind_existing_file(path_text, label)
    if binding["status"] != "PASS":
        return binding, None
    path = Path(binding["path"])
    try:
        payload = _load_json(path)
        _assert_finite(payload, str(path))
    except AuditFailure as exc:
        return {**binding, "status": "FAIL", "reason": str(exc)}, None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("kind") != expected_kind
    ):
        return {
            **binding,
            "status": "FAIL",
            "reason": f"invalid {label} schema",
        }, None
    return binding, payload


def _audit_data_manifest(
    path_text: str | None,
    root_text: str | None,
) -> dict[str, Any]:
    binding, payload = _load_bound_manifest(
        path_text,
        label="data manifest",
        expected_kind="mf_wam_g0_data_manifest",
    )
    if payload is None:
        return binding
    dataset_id = payload.get("dataset_id")
    revision = payload.get("revision")
    files = payload.get("files")
    if (
        not isinstance(dataset_id, str)
        or not dataset_id.strip()
        or not isinstance(revision, str)
        or not revision.strip()
        or not isinstance(files, list)
        or not files
    ):
        return {
            **binding,
            "status": "FAIL",
            "reason": "data manifest requires dataset_id, revision, and files",
        }

    if root_text is None:
        return {
            **binding,
            "status": "UNCERTAIN",
            "reason": "data root not supplied",
        }
    data_root = Path(root_text).expanduser().resolve()
    if not data_root.is_dir():
        return {
            **binding,
            "status": "UNCERTAIN",
            "root": str(data_root),
            "reason": "data root is absent",
        }

    normalized_files: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, item in enumerate(files):
        if not isinstance(item, Mapping):
            return {
                **binding,
                "status": "FAIL",
                "reason": f"data manifest file {index} is not an object",
            }
        relative_path = item.get("path")
        sha256 = item.get("sha256")
        size_bytes = item.get("size_bytes")
        if (
            not isinstance(relative_path, str)
            or not relative_path.strip()
            or relative_path.startswith("/")
            or ".." in Path(relative_path).parts
            or relative_path in seen_paths
            or not isinstance(sha256, str)
            or not SHA256_PATTERN.fullmatch(sha256)
            or type(size_bytes) is not int
            or size_bytes < 0
        ):
            return {
                **binding,
                "status": "FAIL",
                "reason": f"invalid data manifest file entry {index}",
            }
        artifact_path = data_root / relative_path
        if not artifact_path.is_file():
            return {
                **binding,
                "status": "UNCERTAIN",
                "root": str(data_root),
                "reason": f"manifest data file is absent: {relative_path}",
            }
        actual_size = artifact_path.stat().st_size
        actual_sha256 = _sha256(artifact_path)
        if actual_size != size_bytes or actual_sha256 != sha256:
            return {
                **binding,
                "status": "FAIL",
                "root": str(data_root),
                "reason": f"manifest data file mismatch: {relative_path}",
                "expected": {"sha256": sha256, "size_bytes": size_bytes},
                "observed": {
                    "sha256": actual_sha256,
                    "size_bytes": actual_size,
                },
            }
        seen_paths.add(relative_path)
        normalized_files.append(
            {
                "path": relative_path,
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
    inventory_sha256 = hashlib.sha256(
        json.dumps(
            normalized_files,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        **binding,
        "root": str(data_root),
        "dataset_id": dataset_id,
        "revision": revision,
        "file_count": len(normalized_files),
        "inventory_sha256": inventory_sha256,
    }


def _audit_seed_manifest(
    path_text: str | None,
) -> tuple[
    dict[str, Any],
    dict[tuple[str, int, int], dict[str, int]] | None,
]:
    binding, payload = _load_bound_manifest(
        path_text,
        label="seed manifest",
        expected_kind="mf_wam_g0_seed_manifest",
    )
    if payload is None:
        return binding, None
    entries = payload.get("episodes")
    if not isinstance(entries, list) or len(entries) != EXPECTED_EPISODES:
        return (
            {
                **binding,
                "status": "FAIL",
                "reason": (
                    f"seed manifest requires {EXPECTED_EPISODES} episode entries"
                ),
            },
            None,
        )
    normalized_entries: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            return (
                {
                    **binding,
                    "status": "FAIL",
                    "reason": f"seed manifest entry {index} is not an object",
                },
                None,
            )
        suite = item.get("task_suite")
        task_id = item.get("task_id")
        trial_idx = item.get("trial_idx")
        seed_values = {
            field: item.get(field)
            for field in ("task_seed", "environment_seed", "policy_seed")
        }
        identity = (suite, task_id, trial_idx)
        if (
            suite not in SUITES
            or type(task_id) is not int
            or not 0 <= task_id < TASKS_PER_SUITE
            or type(trial_idx) is not int
            or not 0 <= trial_idx < TRIALS_PER_TASK
            or identity in seen
            or any(type(value) is not int for value in seed_values.values())
        ):
            return (
                {
                    **binding,
                    "status": "FAIL",
                    "reason": f"invalid seed manifest entry {index}",
                },
                None,
            )
        seen.add(identity)
        normalized_entries.append(
            {
                "task_suite": suite,
                "task_id": task_id,
                "trial_idx": trial_idx,
                **seed_values,
            }
        )
    expected_identities = {
        (suite, task_id, trial_idx)
        for suite in SUITES
        for task_id in range(TASKS_PER_SUITE)
        for trial_idx in range(TRIALS_PER_TASK)
    }
    if seen != expected_identities:
        return (
            {
                **binding,
                "status": "FAIL",
                "reason": "seed manifest episode coverage is not exact",
            },
            None,
        )
    seed_tree_sha256 = hashlib.sha256(
        json.dumps(
            normalized_entries,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    seed_map = {
        (
            str(item["task_suite"]),
            int(item["task_id"]),
            int(item["trial_idx"]),
        ): {
            field: int(item[field])
            for field in ("task_seed", "environment_seed", "policy_seed")
        }
        for item in normalized_entries
    }
    return (
        {
            **binding,
            "episode_count": len(normalized_entries),
            "seed_tree_sha256": seed_tree_sha256,
        },
        seed_map,
    )


def _audit_source_git(
    path_text: str | None,
    expected_commit: str,
) -> dict[str, Any]:
    if path_text is None:
        return {"status": "UNCERTAIN", "reason": "source root not supplied"}
    root = Path(path_text).expanduser().resolve()
    if not root.is_dir():
        return {
            "status": "UNCERTAIN",
            "path": str(root),
            "reason": "source root is absent",
        }
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": "UNCERTAIN",
            "path": str(root),
            "reason": f"cannot audit source Git identity: {exc}",
        }
    if not GIT_COMMIT_PATTERN.fullmatch(head):
        return {"status": "FAIL", "path": str(root), "reason": "invalid Git HEAD"}
    if head != expected_commit:
        return {
            "status": "FAIL",
            "path": str(root),
            "commit": head,
            "expected_commit": expected_commit,
            "reason": "source Git commit does not match canonical baseline",
        }
    if status:
        return {
            "status": "FAIL",
            "path": str(root),
            "commit": head,
            "reason": "source worktree is not clean",
            "porcelain": status.splitlines(),
        }
    return {"status": "PASS", "path": str(root), "commit": head, "clean": True}


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _audit_run_manifest(
    path_text: str | None,
    *,
    source_git: dict[str, Any],
    data_manifest: dict[str, Any],
    seed_manifest: dict[str, Any],
    artifact_bindings: dict[str, dict[str, Any]],
    baseline: dict[str, Any],
    baseline_policy_sha256: str,
    reference_receipt: dict[str, Any],
    candidate_receipt: dict[str, Any],
    trace_receipt: dict[str, Any],
    candidate_run_id: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    binding = _bind_existing_file(path_text, "run manifest")
    if binding["status"] != "PASS":
        return binding
    path = Path(binding["path"])
    try:
        payload = _load_json(path)
        _assert_finite(payload, str(path))
    except AuditFailure as exc:
        return {**binding, "status": "FAIL", "reason": str(exc)}
    if not isinstance(payload, dict):
        return {**binding, "status": "FAIL", "reason": "run manifest is not an object"}

    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "mf_wam_g0_run_manifest"
    ):
        return {**binding, "status": "FAIL", "reason": "invalid run manifest schema"}

    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        return {**binding, "status": "FAIL", "reason": "missing environment receipt"}
    image_digest = environment.get("image_digest")
    version_fields = ("hostname", "python", "cuda", "pytorch", "mujoco", "libero")
    if (
        not isinstance(image_digest, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest)
        or any(
            not isinstance(environment.get(field), str)
            or not environment[field].strip()
            for field in version_fields
        )
    ):
        return {**binding, "status": "FAIL", "reason": "invalid environment receipt"}

    candidate_run_id_valid = (
        isinstance(candidate_run_id, str) and bool(candidate_run_id.strip())
    )
    manifest_run_id = payload.get("run_id")
    if not candidate_run_id_valid:
        return {
            **binding,
            "status": "FAIL",
            "reason": "candidate summary requires a non-empty run ID",
        }
    if not isinstance(manifest_run_id, str) or not manifest_run_id.strip():
        return {
            **binding,
            "status": "FAIL",
            "reason": "run manifest requires a non-empty run ID",
        }
    if manifest_run_id == baseline["reference_run"]["run_id"]:
        return {
            **binding,
            "status": "FAIL",
            "reason": "candidate run ID must differ from reference run ID",
        }

    required_paths = (
        "git.commit",
        "git.clean",
        "data.manifest_sha256",
        "data.dataset_id",
        "data.revision",
        "data.inventory_sha256",
        "seeds.manifest_sha256",
        "seeds.episode_count",
        "seeds.seed_tree_sha256",
        "artifacts.checkpoint_sha256",
        "artifacts.dataset_stats_sha256",
        "artifacts.resolved_config_sha256",
        "terminal.reference_summary_csv_sha256",
        "terminal.reference_task_success_rates_csv_sha256",
        "terminal.reference_summary_json_sha256",
        "terminal.reference_task_result_tree_sha256",
        "terminal.candidate_summary_csv_sha256",
        "terminal.candidate_task_success_rates_csv_sha256",
        "terminal.candidate_summary_json_sha256",
        "terminal.candidate_task_result_tree_sha256",
        "terminal.candidate_trace_tree_sha256",
        "terminal.candidate_run_id",
        "baseline.policy_sha256",
        "baseline.baseline_id",
        "baseline.reference_run_id",
        "evaluation.suites",
        "evaluation.tasks_per_suite",
        "evaluation.trials_per_task",
        "evaluation.replan_steps",
        "evaluation.action_horizon",
        "evaluation.action_dimension",
        "evaluation.state_dimension",
        "evaluation.first_replan_env_step",
        "evaluation.minimum_trace_records_per_episode",
        "evaluation.strict_success_predicate",
        "evaluation.confidence_level",
        "evaluation.bootstrap_seed",
        "evaluation.minimum_bootstrap_replicates",
    )
    missing_paths = [
        field for field in required_paths if _nested(payload, field) is None
    ]
    if missing_paths:
        return {
            **binding,
            "status": "FAIL",
            "reason": "run manifest is structurally incomplete",
            "missing_fields": missing_paths,
        }

    digest_paths = (
        "data.manifest_sha256",
        "data.inventory_sha256",
        "seeds.manifest_sha256",
        "seeds.seed_tree_sha256",
        "artifacts.checkpoint_sha256",
        "artifacts.dataset_stats_sha256",
        "artifacts.resolved_config_sha256",
        "terminal.reference_summary_csv_sha256",
        "terminal.reference_task_success_rates_csv_sha256",
        "terminal.reference_summary_json_sha256",
        "terminal.reference_task_result_tree_sha256",
        "terminal.candidate_summary_csv_sha256",
        "terminal.candidate_task_success_rates_csv_sha256",
        "terminal.candidate_summary_json_sha256",
        "terminal.candidate_task_result_tree_sha256",
        "terminal.candidate_trace_tree_sha256",
        "baseline.policy_sha256",
    )
    invalid_digest_paths = [
        field
        for field in digest_paths
        if not isinstance(_nested(payload, field), str)
        or not SHA256_PATTERN.fullmatch(_nested(payload, field))
    ]
    if invalid_digest_paths:
        return {
            **binding,
            "status": "FAIL",
            "reason": "run manifest contains invalid SHA-256 fields",
            "invalid_fields": invalid_digest_paths,
        }
    static_expected_values = {
        "run_id": candidate_run_id,
        "git.commit": baseline["source_commit"],
        "git.clean": True,
        "seeds.episode_count": EXPECTED_EPISODES,
        "terminal.candidate_run_id": candidate_run_id,
        "baseline.policy_sha256": baseline_policy_sha256,
        "baseline.baseline_id": baseline["baseline_id"],
        "baseline.reference_run_id": baseline["reference_run"]["run_id"],
        "evaluation.suites": list(SUITES),
        "evaluation.tasks_per_suite": TASKS_PER_SUITE,
        "evaluation.trials_per_task": TRIALS_PER_TASK,
        "evaluation.replan_steps": args.expected_replan_steps,
        "evaluation.action_horizon": args.expected_action_horizon,
        "evaluation.action_dimension": args.expected_action_dimension,
        "evaluation.state_dimension": args.expected_state_dimension,
        "evaluation.first_replan_env_step": args.expected_first_replan_env_step,
        "evaluation.minimum_trace_records_per_episode": (
            args.minimum_trace_records_per_episode
        ),
        "evaluation.strict_success_predicate": "libero_task_success",
        "evaluation.confidence_level": 0.95,
        "evaluation.bootstrap_seed": args.bootstrap_seed,
        "evaluation.minimum_bootstrap_replicates": MIN_BOOTSTRAP_REPLICATES,
    }
    static_mismatches = {
        field: {"expected": expected, "observed": _nested(payload, field)}
        for field, expected in static_expected_values.items()
        if _nested(payload, field) != expected
    }
    for field in ("data.dataset_id", "data.revision"):
        observed = _nested(payload, field)
        if not isinstance(observed, str) or not observed.strip():
            static_mismatches[field] = {
                "expected": "non-empty string",
                "observed": observed,
            }
    if static_mismatches:
        return {
            **binding,
            "status": "FAIL",
            "reason": "run manifest violates its static contract",
            "mismatches": static_mismatches,
        }

    dependencies = {
        "source_git": source_git,
        "data_manifest": data_manifest,
        "seed_manifest": seed_manifest,
        **artifact_bindings,
    }
    dependency_statuses = {item["status"] for item in dependencies.values()}
    if "FAIL" in dependency_statuses:
        return {
            **binding,
            "status": "FAIL",
            "reason": "run-manifest dependency failed",
        }
    if dependency_statuses != {"PASS"}:
        return {
            **binding,
            "status": "UNCERTAIN",
            "reason": "run-manifest dependencies are incomplete",
        }
    expected_values = {
        "schema_version": 1,
        "kind": "mf_wam_g0_run_manifest",
        "run_id": candidate_run_id,
        "git.commit": source_git["commit"],
        "git.clean": True,
        "data.manifest_sha256": data_manifest["sha256"],
        "data.dataset_id": data_manifest["dataset_id"],
        "data.revision": data_manifest["revision"],
        "data.inventory_sha256": data_manifest["inventory_sha256"],
        "seeds.manifest_sha256": seed_manifest["sha256"],
        "seeds.episode_count": seed_manifest["episode_count"],
        "seeds.seed_tree_sha256": seed_manifest["seed_tree_sha256"],
        "artifacts.checkpoint_sha256": artifact_bindings["checkpoint"]["sha256"],
        "artifacts.dataset_stats_sha256": artifact_bindings["dataset_stats"]["sha256"],
        "artifacts.resolved_config_sha256": artifact_bindings["resolved_config"]["sha256"],
        "terminal.reference_summary_csv_sha256": reference_receipt["summary_csv_sha256"],
        "terminal.reference_task_success_rates_csv_sha256": reference_receipt[
            "task_success_rates_csv_sha256"
        ],
        "terminal.reference_summary_json_sha256": reference_receipt[
            "summary_json_sha256"
        ],
        "terminal.reference_task_result_tree_sha256": reference_receipt[
            "task_result_tree_sha256"
        ],
        "terminal.candidate_summary_csv_sha256": candidate_receipt["summary_csv_sha256"],
        "terminal.candidate_task_success_rates_csv_sha256": candidate_receipt[
            "task_success_rates_csv_sha256"
        ],
        "terminal.candidate_summary_json_sha256": candidate_receipt["summary_json_sha256"],
        "terminal.candidate_task_result_tree_sha256": candidate_receipt[
            "task_result_tree_sha256"
        ],
        "terminal.candidate_trace_tree_sha256": trace_receipt["tree_sha256"],
        "baseline.policy_sha256": baseline_policy_sha256,
        "baseline.baseline_id": baseline["baseline_id"],
        "baseline.reference_run_id": baseline["reference_run"]["run_id"],
        "evaluation.suites": list(SUITES),
        "evaluation.tasks_per_suite": TASKS_PER_SUITE,
        "evaluation.trials_per_task": TRIALS_PER_TASK,
        "evaluation.replan_steps": args.expected_replan_steps,
        "evaluation.action_horizon": args.expected_action_horizon,
        "evaluation.action_dimension": args.expected_action_dimension,
        "evaluation.state_dimension": args.expected_state_dimension,
        "evaluation.first_replan_env_step": args.expected_first_replan_env_step,
        "evaluation.minimum_trace_records_per_episode": (
            args.minimum_trace_records_per_episode
        ),
        "evaluation.strict_success_predicate": "libero_task_success",
        "evaluation.confidence_level": 0.95,
        "evaluation.bootstrap_seed": args.bootstrap_seed,
        "evaluation.minimum_bootstrap_replicates": MIN_BOOTSTRAP_REPLICATES,
        "terminal.candidate_run_id": candidate_run_id,
    }
    mismatches = {
        field: {"expected": expected, "observed": _nested(payload, field)}
        for field, expected in expected_values.items()
        if _nested(payload, field) != expected
    }
    if mismatches:
        return {
            **binding,
            "status": "FAIL",
            "reason": "run manifest does not match audited artifacts",
            "mismatches": mismatches,
        }
    return {
        **binding,
        "status": "PASS",
        "run_id": manifest_run_id,
        "environment": dict(environment),
    }


def _audit_candidate_contract(
    contract: dict[str, Any],
    *,
    data_manifest: dict[str, Any],
    seed_manifest: dict[str, Any],
    artifact_bindings: dict[str, dict[str, Any]],
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    if contract["status"] != "LOCKED":
        return {
            "status": "UNCERTAIN",
            "reason": contract["reason"],
        }
    dependencies = {
        "data_manifest": data_manifest,
        "seed_manifest": seed_manifest,
        "resolved_config": artifact_bindings["resolved_config"],
        "run_manifest": run_manifest,
    }
    dependency_statuses = {value["status"] for value in dependencies.values()}
    if "FAIL" in dependency_statuses:
        return {
            "status": "FAIL",
            "reason": "locked candidate-contract dependency failed",
        }
    if dependency_statuses != {"PASS"}:
        return {
            "status": "UNCERTAIN",
            "reason": "locked candidate-contract dependencies are incomplete",
        }
    environment = run_manifest["environment"]
    expected_values = {
        "dataset_id": data_manifest["dataset_id"],
        "data_revision": data_manifest["revision"],
        "data_inventory_sha256": data_manifest["inventory_sha256"],
        "seed_tree_sha256": seed_manifest["seed_tree_sha256"],
        "resolved_config_sha256": artifact_bindings["resolved_config"]["sha256"],
        "image_digest": environment["image_digest"],
        "python": environment["python"],
        "cuda": environment["cuda"],
        "pytorch": environment["pytorch"],
        "mujoco": environment["mujoco"],
        "libero": environment["libero"],
    }
    mismatches = {
        field: {"expected": contract[field], "observed": observed}
        for field, observed in expected_values.items()
        if contract[field] != observed
    }
    if mismatches:
        return {
            "status": "FAIL",
            "reason": "candidate artifacts do not match locked preregistration",
            "mismatches": mismatches,
        }
    return {
        "status": "PASS",
        "contract": {
            field: contract[field] for field in expected_values
        },
    }


def _audit_reference_identity(
    root: Path,
    receipt: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    expected = baseline["reference_run"]
    observed_hashes = {
        "summary_csv_sha256": receipt["summary_csv_sha256"],
        "task_success_rates_csv_sha256": receipt[
            "task_success_rates_csv_sha256"
        ],
        "summary_json_sha256": receipt["summary_json_sha256"],
        "task_result_tree_sha256": receipt["task_result_tree_sha256"],
    }
    expected_hashes = {field: expected[field] for field in observed_hashes}
    if observed_hashes != expected_hashes:
        raise AuditFailure(
            "reference artifacts do not match canonical baseline identity"
        )
    summary = _load_json(root / "summary.json")
    expected_labels = {
        "run_id": expected["run_id"],
        "config": expected["config"],
        "ckpt": expected["checkpoint_label"],
    }
    observed_labels = {field: summary.get(field) for field in expected_labels}
    if observed_labels != expected_labels:
        raise AuditFailure("reference summary labels do not match canonical baseline")
    return {
        "status": "PASS",
        "baseline_id": baseline["baseline_id"],
        "source_commit": baseline["source_commit"],
        "artifact_sha256": observed_hashes,
        "labels": observed_labels,
    }


def _validate_audit_policy(args: argparse.Namespace) -> None:
    if args.overall_equivalence_margin != OVERALL_EQUIVALENCE_MARGIN:
        raise AuditFailure(
            f"overall equivalence margin is fixed at {OVERALL_EQUIVALENCE_MARGIN}"
        )
    if args.suite_drop_margin != SUITE_DROP_MARGIN:
        raise AuditFailure(f"suite drop margin is fixed at {SUITE_DROP_MARGIN}")
    if args.bootstrap_replicates < MIN_BOOTSTRAP_REPLICATES:
        raise AuditFailure(
            f"bootstrap replicates must be >= {MIN_BOOTSTRAP_REPLICATES}"
        )
    if args.bootstrap_seed != PREREGISTERED_BOOTSTRAP_SEED:
        raise AuditFailure(
            f"bootstrap seed is fixed at {PREREGISTERED_BOOTSTRAP_SEED}"
        )
    fixed_fields = {
        "expected_replan_steps": (args.expected_replan_steps, EXPECTED_REPLAN_STEPS),
        "expected_action_horizon": (
            args.expected_action_horizon,
            EXPECTED_ACTION_HORIZON,
        ),
        "expected_action_dimension": (
            args.expected_action_dimension,
            EXPECTED_ACTION_DIMENSION,
        ),
        "expected_state_dimension": (
            args.expected_state_dimension,
            EXPECTED_STATE_DIMENSION,
        ),
        "expected_first_replan_env_step": (
            args.expected_first_replan_env_step,
            EXPECTED_FIRST_REPLAN_ENV_STEP,
        ),
        "minimum_trace_records_per_episode": (
            args.minimum_trace_records_per_episode,
            MIN_TRACE_RECORDS_PER_EPISODE,
        ),
    }
    mismatches = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in fixed_fields.items()
        if observed != expected or type(observed) is not int
    }
    if mismatches:
        raise AuditFailure(f"fixed evaluation contract mismatch: {mismatches}")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    _validate_audit_policy(args)
    baseline, baseline_policy_sha256, baseline_policy_path = _load_baseline_policy()
    reference_root = args.reference_dir.expanduser().resolve()
    candidate_root = args.candidate_dir.expanduser().resolve()
    if reference_root == candidate_root:
        raise AuditFailure("reference and candidate roots must be distinct")
    reference, reference_rates = _load_episode_map(reference_root)
    candidate, candidate_rates = _load_episode_map(candidate_root)

    reference_receipt = {
        "summary_csv_sha256": _audit_summary_csv(reference_root, reference_rates),
        "task_success_rates_csv_sha256": _audit_task_success_csv(reference_root, reference),
        "summary_json_sha256": _audit_summary_json(reference_root, reference_rates),
        "task_result_tree_sha256": _task_result_tree_sha256(reference_root),
    }
    candidate_receipt = {
        "summary_csv_sha256": _audit_summary_csv(candidate_root, candidate_rates),
        "task_success_rates_csv_sha256": _audit_task_success_csv(candidate_root, candidate),
        "summary_json_sha256": _audit_summary_json(candidate_root, candidate_rates),
        "task_result_tree_sha256": _task_result_tree_sha256(candidate_root),
    }
    candidate_summary = _load_json(candidate_root / "summary.json")
    candidate_run_id = (
        candidate_summary.get("run_id")
        if isinstance(candidate_summary, Mapping)
        else None
    )
    reference_identity = _audit_reference_identity(
        reference_root,
        reference_receipt,
        baseline,
    )
    source_git = _audit_source_git(args.source_root, baseline["source_commit"])
    data_manifest = _audit_data_manifest(args.data_manifest, args.data_root)
    seed_manifest, seed_map = _audit_seed_manifest(args.seed_manifest)
    trace_root = (
        args.trace_dir.expanduser().resolve()
        if args.trace_dir
        else candidate_root / "faildetect_traces"
    )
    trace_receipt = _audit_traces(
        trace_root,
        candidate,
        seed_map,
        expected_replan_steps=args.expected_replan_steps,
        expected_action_horizon=args.expected_action_horizon,
        expected_action_dimension=args.expected_action_dimension,
        expected_state_dimension=args.expected_state_dimension,
        expected_first_replan_env_step=args.expected_first_replan_env_step,
        minimum_trace_records_per_episode=(
            args.minimum_trace_records_per_episode
        ),
    )

    differences = [int(candidate[key]) - int(reference[key]) for key in sorted(reference)]
    point_difference = sum(differences) / len(differences)
    ci_lower, ci_upper = _task_bootstrap_ci(
        reference,
        candidate,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    suite_differences = {
        suite: candidate_rates[suite] - reference_rates[suite] for suite in SUITES
    }
    suite_difference_intervals = _suite_bootstrap_intervals(
        reference,
        candidate,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    outcome_pass = (
        ci_lower >= -args.overall_equivalence_margin
        and ci_upper <= args.overall_equivalence_margin
        and all(
            value["ci_lower"] >= -args.suite_drop_margin
            for value in suite_difference_intervals.values()
        )
    )
    outcome = {
        "status": "PASS" if outcome_pass else "FAIL",
        "episode_count": len(candidate),
        "reference_successes": sum(reference.values()),
        "candidate_successes": sum(candidate.values()),
        "point_difference": point_difference,
        "task_bootstrap_95ci": [ci_lower, ci_upper],
        "suite_differences": suite_differences,
        "suite_difference_intervals": suite_difference_intervals,
        "exact_episode_outcome_match": not any(differences),
        "candidate_better_count": sum(value > 0 for value in differences),
        "candidate_worse_count": sum(value < 0 for value in differences),
    }

    canonical_checkpoint_sha256 = baseline["model_artifacts"]["checkpoint_sha256"]
    canonical_stats_sha256 = baseline["model_artifacts"]["dataset_stats_sha256"]
    if args.checkpoint_sha256 is not None and args.checkpoint_sha256.lower() != canonical_checkpoint_sha256:
        raise AuditFailure("caller checkpoint SHA-256 disagrees with canonical baseline")
    if args.dataset_stats_sha256 is not None and args.dataset_stats_sha256.lower() != canonical_stats_sha256:
        raise AuditFailure("caller dataset-stats SHA-256 disagrees with canonical baseline")
    artifact_bindings = {
        "checkpoint": _bind_file(
            args.checkpoint,
            canonical_checkpoint_sha256,
            "checkpoint",
        ),
        "dataset_stats": _bind_file(
            args.dataset_stats,
            canonical_stats_sha256,
            "dataset stats",
        ),
        "resolved_config": _bind_file(
            args.resolved_config,
            args.resolved_config_sha256,
            "resolved config",
        ),
    }
    run_manifest = _audit_run_manifest(
        args.run_manifest,
        source_git=source_git,
        data_manifest=data_manifest,
        seed_manifest=seed_manifest,
        artifact_bindings=artifact_bindings,
        baseline=baseline,
        baseline_policy_sha256=baseline_policy_sha256,
        reference_receipt=reference_receipt,
        candidate_receipt=candidate_receipt,
        trace_receipt=trace_receipt,
        candidate_run_id=candidate_run_id,
        args=args,
    )
    candidate_contract = _audit_candidate_contract(
        baseline["candidate_contract"],
        data_manifest=data_manifest,
        seed_manifest=seed_manifest,
        artifact_bindings=artifact_bindings,
        run_manifest=run_manifest,
    )
    bindings = {
        "source_git": source_git,
        "data_manifest": data_manifest,
        "seed_manifest": seed_manifest,
        "trace_seed_binding": {
            "status": trace_receipt["seed_binding_status"],
            "missing_seed_binding_count": trace_receipt[
                "missing_seed_binding_count"
            ],
        },
        **artifact_bindings,
        "run_manifest": run_manifest,
        "candidate_contract": candidate_contract,
    }
    binding_statuses = {value["status"] for value in bindings.values()}
    if outcome["status"] == "FAIL" or "FAIL" in binding_statuses:
        status = "FAIL"
    elif outcome["status"] == "PASS" and binding_statuses == {"PASS"}:
        status = "PASS"
    else:
        status = "UNCERTAIN"

    gate_evidence = {
        "evidence_complete": status == "PASS",
        "episode_count": len(candidate),
        "suite_count": len(SUITES),
        "tasks_per_suite": TASKS_PER_SUITE,
        "trials_per_task": TRIALS_PER_TASK,
        "missing_trace_count": trace_receipt["missing_trace_count"],
        "non_finite_count": trace_receipt["non_finite_count"],
        "missing_seed_binding_count": trace_receipt[
            "missing_seed_binding_count"
        ],
        "artifact_bindings_complete": binding_statuses == {"PASS"},
        "source_git_clean": source_git.get("status") == "PASS",
        "data_manifest_bound": data_manifest.get("status") == "PASS",
        "seed_manifest_bound": seed_manifest.get("status") == "PASS",
        "trace_seeds_bound": trace_receipt["seed_binding_status"] == "PASS",
        "run_manifest_bound": run_manifest.get("status") == "PASS",
        "candidate_contract_locked": candidate_contract.get("status") == "PASS",
        "minimum_trace_records_per_episode": trace_receipt[
            "minimum_records_per_episode"
        ],
        "first_replan_env_step": trace_receipt["first_replan_env_step"],
        "paired_episode_identity_complete": set(reference) == set(candidate),
        "overall_success_delta": {
            "estimate": point_difference,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        },
        "suite_success_deltas": suite_difference_intervals,
    }
    return {
        "schema_version": 1,
        "kind": "mf_wam_g0_reproduction_audit",
        "status": status,
        "scientific_gate_status": "UNCERTAIN",
        "evidence_classification": "LEGACY_DIAGNOSTIC_ONLY",
        "formal_training_allowed": False,
        "superseded_by": "scripts/audit_mf_wam_g0_bundle.py",
        "outcome_reproduction": outcome,
        "artifact_binding": bindings,
        "reference": {"root": str(reference_root), **reference_receipt},
        "candidate": {
            "root": str(candidate_root),
            "run_id": candidate_run_id,
            **candidate_receipt,
        },
        "traces": {"root": str(trace_root), **trace_receipt},
        "gate_evidence": gate_evidence,
        "policy": {
            "baseline_policy_path": str(baseline_policy_path),
            "baseline_policy_sha256": baseline_policy_sha256,
            "reference_identity": reference_identity,
            "suites": list(SUITES),
            "tasks_per_suite": TASKS_PER_SUITE,
            "trials_per_task": TRIALS_PER_TASK,
            "overall_equivalence_margin": args.overall_equivalence_margin,
            "suite_drop_margin": args.suite_drop_margin,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "expected_action_dimension": args.expected_action_dimension,
            "expected_state_dimension": args.expected_state_dimension,
            "expected_first_replan_env_step": args.expected_first_replan_env_step,
            "minimum_trace_records_per_episode": (
                args.minimum_trace_records_per_episode
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--dataset-stats")
    parser.add_argument("--dataset-stats-sha256")
    parser.add_argument("--resolved-config")
    parser.add_argument("--resolved-config-sha256")
    parser.add_argument("--source-root")
    parser.add_argument("--data-manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--seed-manifest")
    parser.add_argument("--run-manifest")
    parser.add_argument(
        "--expected-replan-steps", type=int, default=EXPECTED_REPLAN_STEPS
    )
    parser.add_argument(
        "--expected-action-horizon", type=int, default=EXPECTED_ACTION_HORIZON
    )
    parser.add_argument(
        "--expected-action-dimension", type=int, default=EXPECTED_ACTION_DIMENSION
    )
    parser.add_argument(
        "--expected-state-dimension", type=int, default=EXPECTED_STATE_DIMENSION
    )
    parser.add_argument(
        "--expected-first-replan-env-step",
        type=int,
        default=EXPECTED_FIRST_REPLAN_ENV_STEP,
    )
    parser.add_argument(
        "--minimum-trace-records-per-episode",
        type=int,
        default=MIN_TRACE_RECORDS_PER_EPISODE,
    )
    parser.add_argument(
        "--overall-equivalence-margin",
        type=float,
        default=OVERALL_EQUIVALENCE_MARGIN,
    )
    parser.add_argument("--suite-drop-margin", type=float, default=SUITE_DROP_MARGIN)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=MIN_BOOTSTRAP_REPLICATES
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=PREREGISTERED_BOOTSTRAP_SEED
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    positive_fields = {
        "--bootstrap-replicates": args.bootstrap_replicates,
        "--expected-replan-steps": args.expected_replan_steps,
        "--expected-action-horizon": args.expected_action_horizon,
        "--expected-action-dimension": args.expected_action_dimension,
        "--expected-state-dimension": args.expected_state_dimension,
        "--minimum-trace-records-per-episode": (
            args.minimum_trace_records_per_episode
        ),
    }
    for field, value in positive_fields.items():
        if value < 1:
            parser.error(f"{field} must be >= 1")
    if args.overall_equivalence_margin < 0 or args.suite_drop_margin < 0:
        parser.error("gate margins must be non-negative")
    if args.expected_first_replan_env_step < 0:
        parser.error("--expected-first-replan-env-step must be >= 0")
    try:
        result = audit(args)
    except (AuditFailure, OSError, UnicodeError, csv.Error, ValueError, OverflowError) as exc:
        result = {
            "schema_version": 1,
            "kind": "mf_wam_g0_reproduction_audit",
            "status": "FAIL",
            "scientific_gate_status": "UNCERTAIN",
            "evidence_classification": "LEGACY_DIAGNOSTIC_ONLY",
            "formal_training_allowed": False,
            "superseded_by": "scripts/audit_mf_wam_g0_bundle.py",
            "error": str(exc),
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return {"PASS": 0, "FAIL": 1, "UNCERTAIN": 2}.get(result["status"], 3)


if __name__ == "__main__":
    raise SystemExit(main())
