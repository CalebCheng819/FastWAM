#!/usr/bin/env python3
"""Validate the three Gate2 process worlds from structured artifacts only.

The training logs are retained as auxiliary evidence, but their rendered text is
never used to decide whether the recovery gate passed.  Wrapper control flow is
kept distinct from trainer-native recovery-load receipts, which prove that
``accelerator.load_state`` returned in each fresh process world.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEP_DIRECTORY_RE = re.compile(r"^step_(\d{6})$")
STEP_WEIGHT_RE = re.compile(r"^step_(\d{6})\.pt(?:\.manifest\.json|\.COMPLETE)?$")
WEIGHTS_MANIFEST_SCHEMA = "fastwam-weights-checkpoint-metadata-no-hash"
WEIGHTS_COMPLETE_SCHEMA = "fastwam-weights-checkpoint-complete-metadata-no-hash"
RECOVERY_LOAD_RECEIPT_SCHEMA = "fastwam-recovery-load-receipt"
GATE2_WORLD_SIZE = 8
GATE2_AGENT_ACTION_TOKEN_BUDGET = 128
GATE2_GRADIENT_ACCUMULATION_STEPS = 1
GATE2_GLOBAL_BATCHES_PER_EPOCH = 1352
GATE2_OPTIMIZER_STEPS_PER_EPOCH = 169
DATA_SCHEDULE_FIELDS = {
    "integrity_mode",
    "epoch",
    "seed",
    "batches",
    "agent_action_token_budget",
    "gradient_accumulation_steps",
    "num_processes",
    "global_batches_per_epoch",
    "optimizer_steps_per_epoch",
}
RECOVERY_LOAD_RECEIPT_FIELDS = {
    "schema_name",
    "schema_version",
    "integrity_mode",
    "accelerator_load_state_returned",
    "source_state_dir",
    "source_trainer_state_file",
    "output_dir",
    "restored_global_step",
    "restored_epoch",
    "restored_batch_in_epoch",
    "world_size",
}
RANDOM_STATE_RE = re.compile(r"^random_states_(\d+)\.pkl$")
OPTIMIZER_STATE_RE = re.compile(
    r"^bf16_zero_pp_rank_(\d+)_mp_rank_00_optim_states\.pt$"
)
MODEL_STATE_RE = re.compile(r"^mp_rank_(\d+)_model_states\.pt$")


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _metadata(path: Path, value: os.stat_result) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=True)),
        "bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "dev": int(value.st_dev),
        "ino": int(value.st_ino),
        "mode": int(value.st_mode),
    }


def _open_regular(path: Path) -> int:
    if path.is_symlink():
        raise RuntimeError(f"linked evidence file is forbidden: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode):
        os.close(descriptor)
        raise RuntimeError(f"evidence path is not a regular file: {path}")
    return descriptor


def file_metadata(path: Path) -> dict[str, Any]:
    descriptor = _open_regular(path)
    try:
        before = os.fstat(descriptor)
        after = os.fstat(descriptor)
        if _metadata(path, before) != _metadata(path, after):
            raise RuntimeError(f"evidence file changed while inspecting: {path}")
        return _metadata(path, after)
    finally:
        os.close(descriptor)


def read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = _open_regular(path)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        before_metadata = _metadata(path, before)
        after_metadata = _metadata(path, after)
        if before_metadata != after_metadata:
            raise RuntimeError(f"evidence file changed while reading: {path}")
        raw = b"".join(chunks)
        if len(raw) != after.st_size:
            raise RuntimeError(f"short evidence-file read: {path}")
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid UTF-8 JSON evidence: {path}") from error
    finally:
        os.close(descriptor)
    if not isinstance(value, dict):
        raise TypeError(f"evidence JSON must be an object: {path}")
    return value, after_metadata


def read_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    descriptor = _open_regular(path)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        before_metadata = _metadata(path, before)
        after_metadata = _metadata(path, after)
        if before_metadata != after_metadata:
            raise RuntimeError(f"evidence file changed while reading: {path}")
        raw = b"".join(chunks)
        if len(raw) != after.st_size:
            raise RuntimeError(f"short evidence-file read: {path}")
        return raw, after_metadata
    finally:
        os.close(descriptor)


def directory_inventory(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"training state must be a non-linked directory: {path}")
    files: list[dict[str, Any]] = []
    directories: list[str] = []
    total_bytes = 0
    for base, names, filenames in os.walk(path, followlinks=False):
        base_path = Path(base)
        for name in names:
            child = base_path / name
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise RuntimeError(f"invalid training-state directory entry: {child}")
            directories.append(child.relative_to(path).as_posix())
        for name in filenames:
            child = base_path / name
            metadata = file_metadata(child)
            relative = child.relative_to(path).as_posix()
            files.append(
                {
                    "path": relative,
                    "bytes": metadata["bytes"],
                    "mtime_ns": metadata["mtime_ns"],
                    "dev": metadata["dev"],
                    "ino": metadata["ino"],
                    "mode": metadata["mode"],
                }
            )
            total_bytes += metadata["bytes"]
    files.sort(key=lambda item: item["path"])
    directories.sort()
    if not files:
        raise RuntimeError(f"training-state directory is empty: {path}")
    if "trainer_state.json" not in {item["path"] for item in files}:
        raise RuntimeError(f"training-state inventory lacks trainer_state.json: {path}")
    return {
        "root": str(path.resolve(strict=True)),
        "file_count": len(files),
        "directory_count": len(directories),
        "total_bytes": total_bytes,
        "directories": directories,
        "files": files,
    }


def _typed_equal(observed: Any, expected: Any) -> bool:
    """Compare JSON values without allowing bool/int or int/float coercion."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _typed_equal(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _typed_equal(left, right) for left, right in zip(observed, expected)
        )
    return observed == expected


def _require_nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{label} must be a non-negative integer")
    return value


def _validate_data_schedule(
    payload: dict[str, Any], expected_step: int,
) -> dict[str, Any]:
    epoch = _require_nonnegative_integer(
        payload.get("epoch"), label="trainer-state epoch"
    )
    batch_in_epoch = _require_nonnegative_integer(
        payload.get("batch_in_epoch"), label="trainer-state batch cursor"
    )
    schedule = payload.get("data_schedule")
    if not isinstance(schedule, dict) or set(schedule) != DATA_SCHEDULE_FIELDS:
        observed = sorted(schedule) if isinstance(schedule, dict) else type(schedule).__name__
        raise RuntimeError(
            "trainer state lacks the exact Gate2 data_schedule fields: "
            f"observed={observed}"
        )
    if schedule["integrity_mode"] != "metadata_no_hash":
        raise RuntimeError("data_schedule integrity_mode is not metadata_no_hash")
    schedule_epoch = _require_nonnegative_integer(
        schedule["epoch"], label="data_schedule epoch"
    )
    if schedule_epoch != epoch:
        raise RuntimeError("data_schedule epoch differs from trainer-state epoch")
    if epoch != 0 or batch_in_epoch != expected_step:
        raise RuntimeError(
            "Gate2 trainer-state cursor must bind epoch=0 and "
            f"batch_in_epoch={expected_step}"
        )
    seed = schedule["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != 42:
        raise RuntimeError("data_schedule seed must be the Gate2 integer seed 42")

    expected_scalars = {
        "agent_action_token_budget": GATE2_AGENT_ACTION_TOKEN_BUDGET,
        "gradient_accumulation_steps": GATE2_GRADIENT_ACCUMULATION_STEPS,
        "num_processes": GATE2_WORLD_SIZE,
    }
    for field, expected in expected_scalars.items():
        observed = schedule[field]
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed != expected
        ):
            raise RuntimeError(
                f"data_schedule {field} must be the Gate2 integer {expected}"
            )

    batches = schedule["batches"]
    if not isinstance(batches, list) or not batches:
        raise RuntimeError("data_schedule batches must be a non-empty list of batches")
    for batch_index, batch in enumerate(batches):
        if not isinstance(batch, list) or not batch:
            raise RuntimeError(
                f"data_schedule batch {batch_index} must be a non-empty list"
            )
        for sample_index in batch:
            if (
                isinstance(sample_index, bool)
                or not isinstance(sample_index, int)
                or sample_index < 0
            ):
                raise RuntimeError(
                    "data_schedule batch entries must be non-negative integers"
                )

    global_batches = _require_nonnegative_integer(
        schedule["global_batches_per_epoch"],
        label="data_schedule global_batches_per_epoch",
    )
    if global_batches == 0 or global_batches != len(batches):
        raise RuntimeError(
            "data_schedule global_batches_per_epoch does not match its non-empty batches"
        )
    if global_batches != GATE2_GLOBAL_BATCHES_PER_EPOCH:
        raise RuntimeError(
            "data_schedule global_batches_per_epoch is not the Gate2 value 1352"
        )
    if global_batches % GATE2_WORLD_SIZE:
        raise RuntimeError(
            "data_schedule global_batches_per_epoch is not divisible by world size 8"
        )
    optimizer_steps = _require_nonnegative_integer(
        schedule["optimizer_steps_per_epoch"],
        label="data_schedule optimizer_steps_per_epoch",
    )
    expected_optimizer_steps = global_batches // GATE2_WORLD_SIZE
    if (
        optimizer_steps != expected_optimizer_steps
        or optimizer_steps != GATE2_OPTIMIZER_STEPS_PER_EPOCH
    ):
        raise RuntimeError(
            "data_schedule optimizer_steps_per_epoch is not the Gate2 value 169"
        )
    microbatches_per_process = global_batches // GATE2_WORLD_SIZE
    if batch_in_epoch > microbatches_per_process:
        raise RuntimeError(
            "trainer-state batch cursor exceeds microbatches_per_process"
        )
    if batch_in_epoch % GATE2_GRADIENT_ACCUMULATION_STEPS:
        raise RuntimeError(
            "trainer-state batch cursor is not gradient-accumulation aligned"
        )
    return {
        "integrity_mode": "metadata_no_hash",
        "epoch": epoch,
        "seed": seed,
        "agent_action_token_budget": GATE2_AGENT_ACTION_TOKEN_BUDGET,
        "gradient_accumulation_steps": GATE2_GRADIENT_ACCUMULATION_STEPS,
        "num_processes": GATE2_WORLD_SIZE,
        "global_batches_per_epoch": global_batches,
        "microbatches_per_process": microbatches_per_process,
        "optimizer_steps_per_epoch": optimizer_steps,
        "batch_count": len(batches),
    }


def _validate_full_state_components(
    state_dir: Path, inventory: dict[str, Any]
) -> dict[str, Any]:
    directories = inventory["directories"]
    if directories != ["pytorch_model"]:
        raise RuntimeError(
            "Gate2 ZeRO-2 state must contain exactly the pytorch_model directory: "
            f"observed={directories}"
        )
    files_by_path = {item["path"]: item for item in inventory["files"]}
    required_root_files = {
        "trainer_state.json",
        "latest",
        "scheduler.bin",
        "zero_to_fp32.py",
        *(f"random_states_{rank}.pkl" for rank in range(GATE2_WORLD_SIZE)),
    }
    expected_optimizer_files = {
        f"pytorch_model/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        for rank in range(GATE2_WORLD_SIZE)
    }
    expected_model_file = "pytorch_model/mp_rank_00_model_states.pt"
    required_files = required_root_files | expected_optimizer_files | {
        expected_model_file
    }
    missing = sorted(required_files - set(files_by_path))
    if missing:
        raise RuntimeError(f"Gate2 ZeRO-2 full state lacks required components: {missing}")
    empty_required = sorted(
        relative
        for relative in required_files
        if files_by_path[relative]["bytes"] <= 0
    )
    if empty_required:
        raise RuntimeError(
            f"Gate2 ZeRO-2 required state components are empty: {empty_required}"
        )
    latest_content, latest_metadata = read_bytes(state_dir / "latest")
    if latest_content != b"pytorch_model":
        raise RuntimeError(
            "Gate2 ZeRO-2 latest file must contain exactly 'pytorch_model'"
        )
    inventory_latest = files_by_path["latest"]
    if any(
        inventory_latest[field] != latest_metadata[field]
        for field in ("bytes", "mtime_ns", "dev", "ino", "mode")
    ):
        raise RuntimeError("Gate2 ZeRO-2 latest file changed during validation")

    root_names = {
        relative for relative in files_by_path if "/" not in relative
    }
    random_state_names = sorted(
        name for name in root_names if name.startswith("random_states_")
    )
    random_state_matches = [RANDOM_STATE_RE.fullmatch(name) for name in random_state_names]
    if any(match is None for match in random_state_matches):
        raise RuntimeError("Gate2 full state contains a malformed random-state rank file")
    random_state_ranks = sorted(int(match.group(1)) for match in random_state_matches)
    if random_state_ranks != list(range(GATE2_WORLD_SIZE)):
        raise RuntimeError(
            "Gate2 full state random-state ranks are incomplete or contain extra ranks: "
            f"observed={random_state_ranks}"
        )

    model_names = {
        Path(relative).name
        for relative in files_by_path
        if relative.startswith("pytorch_model/")
    }
    optimizer_names = sorted(
        name
        for name in model_names
        if "rank_" in name and name.endswith("_optim_states.pt")
    )
    optimizer_matches = [OPTIMIZER_STATE_RE.fullmatch(name) for name in optimizer_names]
    if any(match is None for match in optimizer_matches):
        raise RuntimeError("Gate2 full state contains an unexpected optimizer rank file")
    optimizer_ranks = sorted(int(match.group(1)) for match in optimizer_matches)
    if optimizer_ranks != list(range(GATE2_WORLD_SIZE)):
        raise RuntimeError(
            "Gate2 full state optimizer ranks are incomplete or contain extra ranks: "
            f"observed={optimizer_ranks}"
        )
    model_state_names = sorted(
        name
        for name in model_names
        if "rank_" in name and name.endswith("_model_states.pt")
    )
    if any(MODEL_STATE_RE.fullmatch(name) is None for name in model_state_names):
        raise RuntimeError("Gate2 full state contains an unexpected model-state rank file")
    if model_state_names != [Path(expected_model_file).name]:
        raise RuntimeError(
            "Gate2 full state model-state ranks are incomplete or contain extra ranks: "
            f"observed={model_state_names}"
        )
    return {
        "layout": "accelerate_deepspeed_bf16_zero2_1x8",
        "world_size": GATE2_WORLD_SIZE,
        "root_required_files": sorted(required_root_files),
        "random_state_ranks": random_state_ranks,
        "optimizer_state_ranks": optimizer_ranks,
        "model_state_file": expected_model_file,
        "additional_regular_files": sorted(set(files_by_path) - required_files),
    }


def _checkpoint_step_tags(output: Path) -> dict[str, Any]:
    state_root = output / "checkpoints" / "state"
    weights_root = output / "checkpoints" / "weights"
    if state_root.is_symlink() or not state_root.is_dir():
        raise RuntimeError(f"state checkpoint root is invalid: {state_root}")
    if weights_root.is_symlink() or not weights_root.is_dir():
        raise RuntimeError(f"weights checkpoint root is invalid: {weights_root}")
    state_entries = sorted(state_root.iterdir(), key=lambda item: item.name)
    weight_entries = sorted(weights_root.iterdir(), key=lambda item: item.name)
    state_steps: list[int] = []
    weight_steps: set[int] = set()
    for child in state_entries:
        match = STEP_DIRECTORY_RE.fullmatch(child.name)
        if match is None or child.is_symlink() or not child.is_dir():
            raise RuntimeError(f"unexpected state-checkpoint entry: {child}")
        state_steps.append(int(match.group(1)))
    for child in weight_entries:
        match = STEP_WEIGHT_RE.fullmatch(child.name)
        if match is None or child.is_symlink() or not child.is_file():
            raise RuntimeError(f"unexpected weights-checkpoint entry: {child}")
        weight_steps.add(int(match.group(1)))
    return {
        "state_steps": sorted(state_steps),
        "weight_steps": sorted(weight_steps),
        "state_entries": [child.name for child in state_entries],
        "weight_entries": [child.name for child in weight_entries],
    }


def _validate_checkpoint_layout(output: Path, expected_step: int) -> dict[str, Any]:
    observed = _checkpoint_step_tags(output)
    expected = [expected_step]
    step_tag = f"step_{expected_step:06d}"
    if (
        observed["state_steps"] != expected
        or observed["weight_steps"] != expected
        or observed["state_entries"] != [step_tag]
        or observed["weight_entries"]
        != [f"{step_tag}.pt", f"{step_tag}.pt.COMPLETE", f"{step_tag}.pt.manifest.json"]
    ):
        raise RuntimeError(
            "checkpoint layout does not contain exactly the expected completed step: "
            f"output={output} expected={expected_step} observed={observed}"
        )
    return observed


def _validate_last_step_metrics(payload: dict[str, Any], expected_step: int) -> dict[str, Any]:
    metrics = payload.get("last_step_metrics")
    if (
        not isinstance(metrics, dict)
        or not _typed_equal(metrics.get("step"), expected_step)
    ):
        raise RuntimeError(
            f"trainer state last_step_metrics does not prove step {expected_step}"
        )
    if set(metrics) != {
        "grad_norm",
        "learning_rate",
        "loss",
        "loss_components",
        "step",
    }:
        raise RuntimeError(f"trainer metrics have an unexpected schema: {sorted(metrics)}")
    components = metrics.get("loss_components")
    if not isinstance(components, dict) or not components:
        raise RuntimeError("trainer metrics lack loss components")
    numeric_metrics = {
        "grad_norm": metrics["grad_norm"],
        "learning_rate": metrics["learning_rate"],
        "loss": metrics["loss"],
        **{f"loss_components.{key}": value for key, value in components.items()},
    }
    for key, value in numeric_metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"trainer metric {key!r} is not numeric")
        if not math.isfinite(float(value)):
            raise RuntimeError(f"non-finite trainer metric {key!r} at step {expected_step}")
    return metrics


def validate_trainer_state(
    state_dir: Path,
    expected_step: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_file = state_dir / "trainer_state.json"
    payload, state_metadata = read_json(state_file)
    if (
        isinstance(payload.get("global_step"), bool)
        or not isinstance(payload.get("global_step"), int)
        or payload.get("global_step") != expected_step
    ):
        raise RuntimeError(f"trainer state is not step {expected_step}: {state_file}")
    schedule_summary = _validate_data_schedule(payload, expected_step)
    if payload.get("evaluation_records") != []:
        raise RuntimeError("Gate2 trainer state must have no evaluation records")
    metrics = _validate_last_step_metrics(payload, expected_step)
    contract = payload.get("run_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("trainer state lacks a run contract")
    if not _typed_equal(contract.get("contract_version"), 2):
        raise RuntimeError("trainer state contract version is not 2")
    if contract.get("integrity_mode") != "metadata_no_hash":
        raise RuntimeError("trainer state does not declare metadata_no_hash")
    if contract.get("state_kind") != "accelerate_full_state":
        raise RuntimeError("trainer state is not an Accelerate full state")
    expected_treatment = {
        "training_mode": "action_only_cache",
        "trainable_scope": "action",
        "checkpoint_state_kind": "full",
        "video_gen": False,
        "hub": True,
        "gaussian": True,
    }
    treatment = contract.get("treatment")
    if not _typed_equal(treatment, expected_treatment):
        raise RuntimeError(
            "trainer treatment does not match the exact Gate2 scientific configuration"
        )
    expected_optimization = {
        "optimizer": "torch.optim.AdamW",
        "learning_rate": 3.0e-5,
        "weight_decay": 1.0e-2,
        "betas": [0.9, 0.95],
        "lr_scheduler_type": "cosine",
        "max_steps": 2,
        "warmup_steps": 0,
        "batch_size": 1,
        "agent_action_token_budget": GATE2_AGENT_ACTION_TOKEN_BUDGET,
        "gradient_accumulation_steps": GATE2_GRADIENT_ACCUMULATION_STEPS,
        "world_size": GATE2_WORLD_SIZE,
        "mixed_precision": "bf16",
        "max_grad_norm": 1.0,
        "seed": 42,
    }
    optimization = contract.get("optimization") or {}
    if not _typed_equal(optimization, expected_optimization):
        raise RuntimeError(
            "trainer optimization does not match the exact Gate2 scientific configuration"
        )
    resolved = contract.get("resolved_config") or {}
    stop_step = resolved.get("recovery_gate_stop_after_checkpoint_step")
    if isinstance(stop_step, bool) or not isinstance(stop_step, int) or stop_step != 1:
        raise RuntimeError("trainer contract does not bind the step-1 recovery pause")
    inventory = directory_inventory(state_dir)
    full_state_components = _validate_full_state_components(state_dir, inventory)
    state_inventory_entry = next(
        item for item in inventory["files"] if item["path"] == "trainer_state.json"
    )
    for field in ("bytes", "mtime_ns", "dev", "ino", "mode"):
        if state_inventory_entry[field] != state_metadata[field]:
            raise RuntimeError("trainer_state.json changed during state inventory")
    return payload, {
        "global_step": expected_step,
        "epoch": payload["epoch"],
        "batch_in_epoch": payload["batch_in_epoch"],
        "data_schedule": schedule_summary,
        "last_step_metrics": metrics,
        "trainer_state_file": state_metadata,
        "full_state_components": full_state_components,
        "state_inventory": inventory,
    }


def validate_weights(
    weights: Path,
    expected_step: int,
) -> dict[str, Any]:
    manifest_path = weights.with_name(f"{weights.name}.manifest.json")
    complete_path = weights.with_name(f"{weights.name}.COMPLETE")
    manifest, manifest_metadata = read_json(manifest_path)
    complete, complete_metadata = read_json(complete_path)
    weights_metadata = file_metadata(weights)
    expected_manifest = {
        "schema_name": WEIGHTS_MANIFEST_SCHEMA,
        "schema_version": 1,
        "integrity_mode": "metadata_no_hash",
        "filename": weights.name,
        "file": weights_metadata,
        "global_step": expected_step,
        "checkpoint_state_kind": "full",
    }
    if not _typed_equal(manifest, expected_manifest):
        raise RuntimeError(f"step-{expected_step} weights manifest is inconsistent")
    expected_complete = {
        "schema_name": WEIGHTS_COMPLETE_SCHEMA,
        "schema_version": 1,
        "integrity_mode": "metadata_no_hash",
        "manifest_filename": manifest_path.name,
        "manifest_file": manifest_metadata,
        "checkpoint_filename": weights.name,
        "checkpoint_file": weights_metadata,
    }
    if not _typed_equal(complete, expected_complete):
        raise RuntimeError(f"step-{expected_step} weights COMPLETE metadata is inconsistent")
    return {
        "weights_file": weights_metadata,
        "weights_manifest_file": manifest_metadata,
        "weights_complete_file": complete_metadata,
    }


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    rendered = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != rendered:
        raise RuntimeError(f"structured evidence readback comparison failed: {path}")


def _validate_recovery_load_receipt(
    *,
    receipt_path: Path,
    source_state_dir: Path,
    output_dir: Path,
    source_state: dict[str, Any],
    source_summary: dict[str, Any],
    expected_step: int,
) -> dict[str, Any]:
    try:
        receipt, receipt_metadata = read_json(receipt_path)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"trainer-native recovery load receipt is missing: {receipt_path}"
        ) from error
    if set(receipt) != RECOVERY_LOAD_RECEIPT_FIELDS:
        raise RuntimeError(
            "trainer-native recovery load receipt fields differ from the exact schema"
        )

    resolved_source = str(source_state_dir.resolve(strict=True))
    resolved_output = str(output_dir.resolve(strict=True))
    current_source_metadata = file_metadata(source_state_dir / "trainer_state.json")
    if not _typed_equal(current_source_metadata, source_summary["trainer_state_file"]):
        raise RuntimeError(
            "recovery receipt source trainer-state file changed after validation"
        )
    if not _typed_equal(receipt["source_trainer_state_file"], current_source_metadata):
        raise RuntimeError(
            "trainer-native recovery receipt source trainer-state metadata drift"
        )

    expected_scalars = {
        "schema_name": RECOVERY_LOAD_RECEIPT_SCHEMA,
        "schema_version": 1,
        "integrity_mode": "metadata_no_hash",
        "accelerator_load_state_returned": True,
        "source_state_dir": resolved_source,
        "output_dir": resolved_output,
        "restored_global_step": expected_step,
        "restored_epoch": source_state["epoch"],
        "restored_batch_in_epoch": source_state["batch_in_epoch"],
        "world_size": GATE2_WORLD_SIZE,
    }
    for field, expected in expected_scalars.items():
        if not _typed_equal(receipt[field], expected):
            raise RuntimeError(
                "trainer-native recovery load receipt binding mismatch: "
                f"field={field} expected={expected!r} observed={receipt[field]!r}"
            )
    return {
        "receipt_file": receipt_metadata,
        "receipt": receipt,
        "proof_semantics": "trainer_native_accelerator_load_state_returned",
    }


def _validate_final_verify_has_no_checkpoints(output_dir: Path) -> dict[str, Any]:
    checkpoints = output_dir / "checkpoints"
    if checkpoints.is_symlink():
        raise RuntimeError("final verification checkpoints root must not be linked")
    if not checkpoints.exists():
        return {"checkpoints_root_present": False, "empty_directories": []}
    if not checkpoints.is_dir():
        raise RuntimeError("final verification checkpoints root is not a directory")
    empty_directories: list[str] = []
    for child in sorted(checkpoints.iterdir(), key=lambda item: item.name):
        if child.name not in {"state", "weights"}:
            raise RuntimeError(
                f"final verification world produced an unexpected checkpoint entry: {child}"
            )
        if child.is_symlink() or not child.is_dir():
            raise RuntimeError(
                f"final verification checkpoint container is invalid: {child}"
            )
        entries = list(child.iterdir())
        if entries:
            raise RuntimeError(
                "final verification world must not produce checkpoint entries: "
                f"container={child} entries={[entry.name for entry in entries]}"
            )
        empty_directories.append(child.name)
    return {
        "checkpoints_root_present": True,
        "empty_directories": empty_directories,
    }


def verify_save_world(publish_root: Path) -> dict[str, Any]:
    root = publish_root.resolve(strict=True)
    output = root / "save_world"
    state_dir = output / "checkpoints" / "state" / "step_000001"
    weights = output / "checkpoints" / "weights" / "step_000001.pt"
    layout = _validate_checkpoint_layout(output, 1)
    state, state_summary = validate_trainer_state(state_dir, 1)
    weight_summary = validate_weights(weights, 1)
    log_metadata = file_metadata(root / "save_world.log")
    evidence = {
        "schema": "fastwam-gate2-save-world-process-receipt-v1",
        "integrity_mode": "metadata_no_hash",
        "created_at": _now(),
        "phase": "save_world",
        "launch_pipeline_exit_status": 0,
        "launch_pipeline": "set_euo_pipefail_returned_before_structured_validation",
        "launch_pipeline_exit_status_scope": "wrapper_control_flow_only",
        "trainer_native_exit_status_proof": False,
        "expected_pause_step": 1,
        "checkpoint_layout": layout,
        "state": state_summary,
        "weights": weight_summary,
        "run_contract": state["run_contract"],
        "log_file_auxiliary_only": log_metadata,
        "log_text_used_for_acceptance": False,
    }
    _exclusive_json(root / "save_world_process_receipt.json", evidence)
    return evidence


def verify_recovery_worlds(publish_root: Path) -> dict[str, Any]:
    root = publish_root.resolve(strict=True)
    receipt, receipt_metadata = read_json(root / "save_world_process_receipt.json")
    if (
        receipt.get("schema") != "fastwam-gate2-save-world-process-receipt-v1"
        or receipt.get("phase") != "save_world"
        or receipt.get("launch_pipeline_exit_status") != 0
        or receipt.get("launch_pipeline_exit_status_scope")
        != "wrapper_control_flow_only"
        or receipt.get("trainer_native_exit_status_proof") is not False
        or receipt.get("expected_pause_step") != 1
        or receipt.get("log_text_used_for_acceptance") is not False
    ):
        raise RuntimeError("save-world process receipt is inconsistent")

    save_output = root / "save_world"
    load_output = root / "load_world"
    final_verify_output = root / "final_verify_world"
    save_layout = _validate_checkpoint_layout(save_output, 1)
    load_layout = _validate_checkpoint_layout(load_output, 2)
    save_state_dir = save_output / "checkpoints" / "state" / "step_000001"
    final_state_dir = load_output / "checkpoints" / "state" / "step_000002"
    save_state, save_summary = validate_trainer_state(save_state_dir, 1)
    final_state, final_summary = validate_trainer_state(final_state_dir, 2)
    if final_state["run_contract"] != save_state["run_contract"]:
        raise RuntimeError("fresh load-world run contract differs from save world")
    save_weights = save_output / "checkpoints" / "weights" / "step_000001.pt"
    save_weight_summary = validate_weights(save_weights, 1)
    final_weights = load_output / "checkpoints" / "weights" / "step_000002.pt"
    weight_summary = validate_weights(final_weights, 2)
    save_log_metadata = file_metadata(root / "save_world.log")
    load_log_metadata = file_metadata(root / "load_world.log")
    final_verify_log_metadata = file_metadata(root / "final_verify_world.log")
    if (
        receipt.get("checkpoint_layout") != save_layout
        or receipt.get("state") != save_summary
        or receipt.get("weights") != save_weight_summary
        or receipt.get("run_contract") != save_state["run_contract"]
        or receipt.get("log_file_auxiliary_only") != save_log_metadata
    ):
        raise RuntimeError("save-world process receipt no longer matches its artifacts")

    load_receipt = _validate_recovery_load_receipt(
        receipt_path=load_output / "recovery_load_receipt.json",
        source_state_dir=save_state_dir,
        output_dir=load_output,
        source_state=save_state,
        source_summary=save_summary,
        expected_step=1,
    )
    final_verify_receipt = _validate_recovery_load_receipt(
        receipt_path=final_verify_output / "recovery_load_receipt.json",
        source_state_dir=final_state_dir,
        output_dir=final_verify_output,
        source_state=final_state,
        source_summary=final_summary,
        expected_step=2,
    )
    final_verify_checkpoint_state = _validate_final_verify_has_no_checkpoints(
        final_verify_output
    )

    evidence = {
        "schema": "fastwam-gate2-recovery-evidence-v3",
        "integrity_mode": "metadata_no_hash",
        "created_at": _now(),
        "wrapper_control_flow": {
            "save_world_pipeline_returned": True,
            "load_world_pipeline_returned": True,
            "final_verify_world_pipeline_returned": True,
            "trainer_native_exit_status_proof": False,
        },
        "process_exit_semantics": (
            "wrapper_set_euo_pipefail_control_flow_only_not_trainer_native_proof"
        ),
        "resumed_from_step": 1,
        "final_global_step": 2,
        "fresh_load_advanced": True,
        "checkpoint_state_kind": "full",
        "save_training_state_in_load_world": True,
        "save_world_checkpoint_layout": save_layout,
        "load_world_checkpoint_layout": load_layout,
        "run_contract_exact_match": True,
        "save_state": save_summary,
        "final_state": final_summary,
        "final_weights": weight_summary,
        "trainer_native_recovery_load_receipts": {
            "load_world": load_receipt,
            "final_verify_world": final_verify_receipt,
        },
        "final_verify_world_checkpoint_state": final_verify_checkpoint_state,
        "save_world_process_receipt": receipt_metadata,
        "auxiliary_logs": {
            "save_world": save_log_metadata,
            "load_world": load_log_metadata,
            "final_verify_world": final_verify_log_metadata,
        },
        "log_text_used_for_acceptance": False,
        "acceptance_basis": [
            "trainer_native_step1_and_step2_recovery_load_receipts",
            "exact_step_checkpoint_layouts",
            "exact_accelerate_deepspeed_zero2_full_state_components",
            "exact_dynamic_data_schedule_and_cursor_contract",
            "trainer_state_global_step_and_last_step_metrics",
            "exact_save_and_load_run_contract_equality",
            "weights_manifest_and_complete_metadata",
            "final_verify_world_has_no_checkpoint_entries",
        ],
    }
    _exclusive_json(root / "gate2_trainer_evidence.json", evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("verify-save", "verify-recovery"))
    parser.add_argument("--publish-root", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "verify-save":
        verify_save_world(args.publish_root)
    else:
        verify_recovery_worlds(args.publish_root)


if __name__ == "__main__":
    main()
