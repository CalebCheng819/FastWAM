from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE2_DIR = (
    REPO_ROOT
    / ".research-workflow"
    / "experiments"
    / "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-S42-20260809"
    / "dlc-draft-1x8"
)
STRUCTURED_EVIDENCE = GATE2_DIR / "gate2_structured_evidence.py"
RUNTIME = GATE2_DIR / "runtime.sh"
R3_EXPERIMENT_ID = (
    "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-R3-S42-20260809"
)


def _load_structured_evidence() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "fastwam_gate2_structured_evidence_test", STRUCTURED_EVIDENCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _metadata(path: Path) -> dict[str, Any]:
    value = path.stat()
    return {
        "path": str(path.resolve(strict=True)),
        "bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "dev": int(value.st_dev),
        "ino": int(value.st_ino),
        "mode": int(value.st_mode),
    }


def _run_contract() -> dict[str, Any]:
    return {
        "contract_version": 2,
        "integrity_mode": "metadata_no_hash",
        "state_kind": "accelerate_full_state",
        "treatment": {
            "training_mode": "action_only_cache",
            "trainable_scope": "action",
            "checkpoint_state_kind": "full",
            "video_gen": False,
            "hub": True,
            "gaussian": True,
        },
        "optimization": {
            "optimizer": "torch.optim.AdamW",
            "learning_rate": 3.0e-5,
            "weight_decay": 1.0e-2,
            "betas": [0.9, 0.95],
            "lr_scheduler_type": "cosine",
            "max_steps": 2,
            "warmup_steps": 0,
            "batch_size": 1,
            "agent_action_token_budget": 128,
            "gradient_accumulation_steps": 1,
            "world_size": 8,
            "mixed_precision": "bf16",
            "max_grad_norm": 1.0,
            "seed": 42,
        },
        "resolved_config": {"recovery_gate_stop_after_checkpoint_step": 1},
        "dataset": {"identity": "minimal-real-gate2-fixture"},
    }


def _write_state(
    publish_root: Path,
    world: str,
    step: int,
    contract: dict[str, Any],
) -> Path:
    state_dir = (
        publish_root
        / world
        / "checkpoints"
        / "state"
        / f"step_{step:06d}"
    )
    state = {
        "global_step": step,
        "epoch": 0,
        "batch_in_epoch": step,
        "data_schedule": {
            "integrity_mode": "metadata_no_hash",
            "epoch": 0,
            "seed": 42,
            "batches": [[index] for index in range(1352)],
            "agent_action_token_budget": 128,
            "gradient_accumulation_steps": 1,
            "num_processes": 8,
            "global_batches_per_epoch": 1352,
            "optimizer_steps_per_epoch": 169,
        },
        "evaluation_records": [],
        "last_step_metrics": {
            "grad_norm": 1.25,
            "learning_rate": 3.0e-5,
            "loss": 0.5 / step,
            "loss_components": {"action": 0.25 / step, "hub": 0.125 / step},
            "step": step,
        },
        "run_contract": copy.deepcopy(contract),
    }
    _write_json(state_dir / "trainer_state.json", state)
    (state_dir / "latest").write_bytes(b"pytorch_model")
    for filename in ("scheduler.bin", "zero_to_fp32.py"):
        (state_dir / filename).write_bytes(f"fixture {filename}\n".encode())
    for rank in range(8):
        (state_dir / f"random_states_{rank}.pkl").write_bytes(
            f"fixture random state rank {rank}\n".encode()
        )
    model_dir = state_dir / "pytorch_model"
    model_dir.mkdir()
    for rank in range(8):
        (
            model_dir
            / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        ).write_bytes(f"fixture optimizer rank {rank}\n".encode())
    (model_dir / "mp_rank_00_model_states.pt").write_bytes(
        b"fixture model state\n"
    )
    # Ordinary non-rank files are permitted in addition to the required state.
    (model_dir / "version").write_bytes(b"fixture version\n")
    return state_dir


def _write_weights(publish_root: Path, world: str, step: int) -> Path:
    weights = (
        publish_root
        / world
        / "checkpoints"
        / "weights"
        / f"step_{step:06d}.pt"
    )
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(f"minimal full-state weights for step {step}\n".encode())
    manifest = weights.with_name(f"{weights.name}.manifest.json")
    complete = weights.with_name(f"{weights.name}.COMPLETE")
    _write_json(
        manifest,
        {
            "schema_name": "fastwam-weights-checkpoint-metadata-no-hash",
            "schema_version": 1,
            "integrity_mode": "metadata_no_hash",
            "filename": weights.name,
            "global_step": step,
            "checkpoint_state_kind": "full",
            "file": _metadata(weights),
        },
    )
    _write_json(
        complete,
        {
            "schema_name": "fastwam-weights-checkpoint-complete-metadata-no-hash",
            "schema_version": 1,
            "integrity_mode": "metadata_no_hash",
            "checkpoint_filename": weights.name,
            "checkpoint_file": _metadata(weights),
            "manifest_filename": manifest.name,
            "manifest_file": _metadata(manifest),
        },
    )
    return weights


def _write_world(
    publish_root: Path,
    world: str,
    step: int,
    contract: dict[str, Any],
) -> Path:
    state_dir = _write_state(publish_root, world, step, contract)
    _write_weights(publish_root, world, step)
    return state_dir


def _write_recovery_load_receipt(
    output_dir: Path,
    source_state_dir: Path,
    restored_step: int,
) -> Path:
    source_state = json.loads(
        (source_state_dir / "trainer_state.json").read_text(encoding="utf-8")
    )
    receipt = output_dir / "recovery_load_receipt.json"
    _write_json(
        receipt,
        {
            "schema_name": "fastwam-recovery-load-receipt",
            "schema_version": 1,
            "integrity_mode": "metadata_no_hash",
            "accelerator_load_state_returned": True,
            "source_state_dir": str(source_state_dir.resolve(strict=True)),
            "source_trainer_state_file": _metadata(
                source_state_dir / "trainer_state.json"
            ),
            "output_dir": str(output_dir.resolve(strict=True)),
            "restored_global_step": restored_step,
            "restored_epoch": source_state["epoch"],
            "restored_batch_in_epoch": source_state["batch_in_epoch"],
            "world_size": 8,
        },
    )
    return receipt


def _write_valid_recovery_worlds(
    publish_root: Path,
    contract: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    contract = _run_contract() if contract is None else contract
    save_state = _write_world(publish_root, "save_world", 1, contract)
    final_state = _write_world(publish_root, "load_world", 2, contract)
    load_output = publish_root / "load_world"
    final_verify_output = publish_root / "final_verify_world"
    (final_verify_output / "checkpoints" / "state").mkdir(parents=True)
    (final_verify_output / "checkpoints" / "weights").mkdir(parents=True)
    _write_recovery_load_receipt(load_output, save_state, 1)
    _write_recovery_load_receipt(final_verify_output, final_state, 2)
    return save_state, final_state, final_verify_output


@pytest.fixture
def structured_evidence() -> ModuleType:
    return _load_structured_evidence()


@pytest.fixture
def publish_root(tmp_path: Path) -> Path:
    root = tmp_path / "publish"
    root.mkdir()
    # These deliberately resemble Rich-rendered output, including ANSI escapes
    # and line wrapping inside phrases that the old substring gates expected.
    (root / "save_world.log").write_bytes(
        b"\x1b[36m[recovery-\n  gate]\x1b[0m saved check-\n  point one\n"
    )
    (root / "load_world.log").write_bytes(
        b"\x1b[32m[re-\n  sume]\x1b[0m step one ->\n  step two\n"
    )
    (root / "final_verify_world.log").write_bytes(
        b"\x1b[35mtrainer-native reload returned; rendered text is auxiliary\x1b[0m\n"
    )
    return root


def test_rich_ansi_wrapped_logs_do_not_affect_step_one_to_two_success(
    structured_evidence: ModuleType,
    publish_root: Path,
) -> None:
    _write_valid_recovery_worlds(publish_root)

    save_receipt = structured_evidence.verify_save_world(publish_root)
    recovery = structured_evidence.verify_recovery_worlds(publish_root)

    assert save_receipt["launch_pipeline_exit_status"] == 0
    assert save_receipt["checkpoint_layout"] == {
        "state_steps": [1],
        "weight_steps": [1],
        "state_entries": ["step_000001"],
        "weight_entries": [
            "step_000001.pt",
            "step_000001.pt.COMPLETE",
            "step_000001.pt.manifest.json",
        ],
    }
    assert save_receipt["log_text_used_for_acceptance"] is False
    assert recovery["resumed_from_step"] == 1
    assert recovery["final_global_step"] == 2
    assert recovery["fresh_load_advanced"] is True
    assert recovery["run_contract_exact_match"] is True
    assert "save_world_exit_status" not in recovery
    assert recovery["wrapper_control_flow"]["trainer_native_exit_status_proof"] is False
    native_receipts = recovery["trainer_native_recovery_load_receipts"]
    assert native_receipts["load_world"]["receipt"]["restored_global_step"] == 1
    assert (
        native_receipts["final_verify_world"]["receipt"]["restored_global_step"]
        == 2
    )
    assert recovery["final_verify_world_checkpoint_state"] == {
        "checkpoints_root_present": True,
        "empty_directories": ["state", "weights"],
    }
    assert recovery["load_world_checkpoint_layout"] == {
        "state_steps": [2],
        "weight_steps": [2],
        "state_entries": ["step_000002"],
        "weight_entries": [
            "step_000002.pt",
            "step_000002.pt.COMPLETE",
            "step_000002.pt.manifest.json",
        ],
    }
    assert recovery["log_text_used_for_acceptance"] is False


@pytest.mark.parametrize("artifact_kind", ["state", "weights"])
def test_save_world_rejects_any_premature_step_two_artifact(
    structured_evidence: ModuleType,
    publish_root: Path,
    artifact_kind: str,
) -> None:
    _write_world(publish_root, "save_world", 1, _run_contract())
    if artifact_kind == "state":
        (
            publish_root
            / "save_world"
            / "checkpoints"
            / "state"
            / "step_000002"
        ).mkdir(parents=True)
    else:
        unexpected = (
            publish_root
            / "save_world"
            / "checkpoints"
            / "weights"
            / "step_000002.pt"
        )
        unexpected.write_bytes(b"premature step two\n")

    with pytest.raises(RuntimeError, match="exactly the expected completed step"):
        structured_evidence.verify_save_world(publish_root)


@pytest.mark.parametrize("artifact_kind", ["state", "weights"])
def test_load_world_rejects_any_replayed_step_one_artifact(
    structured_evidence: ModuleType,
    publish_root: Path,
    artifact_kind: str,
) -> None:
    _write_valid_recovery_worlds(publish_root)
    structured_evidence.verify_save_world(publish_root)
    if artifact_kind == "state":
        (
            publish_root
            / "load_world"
            / "checkpoints"
            / "state"
            / "step_000001"
        ).mkdir(parents=True)
    else:
        unexpected = (
            publish_root
            / "load_world"
            / "checkpoints"
            / "weights"
            / "step_000001.pt"
        )
        unexpected.write_bytes(b"replayed step one\n")

    with pytest.raises(RuntimeError, match="exactly the expected completed step"):
        structured_evidence.verify_recovery_worlds(publish_root)


def test_recovery_rejects_run_contract_drift(
    structured_evidence: ModuleType,
    publish_root: Path,
) -> None:
    contract = _run_contract()
    _write_valid_recovery_worlds(publish_root, contract)
    structured_evidence.verify_save_world(publish_root)
    final_state = (
        publish_root
        / "load_world"
        / "checkpoints"
        / "state"
        / "step_000002"
        / "trainer_state.json"
    )
    payload = json.loads(final_state.read_text(encoding="utf-8"))
    payload["run_contract"]["dataset"]["identity"] = "drifted-fixture"
    _write_json(final_state, payload)

    with pytest.raises(RuntimeError, match="run contract differs"):
        structured_evidence.verify_recovery_worlds(publish_root)


@pytest.mark.parametrize("bad_metric", [float("nan"), float("inf")])
def test_save_world_rejects_non_finite_last_step_metrics(
    structured_evidence: ModuleType,
    publish_root: Path,
    bad_metric: float,
) -> None:
    state_dir = _write_state(publish_root, "save_world", 1, _run_contract())
    _write_weights(publish_root, "save_world", 1)
    state_file = state_dir / "trainer_state.json"
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    payload["last_step_metrics"]["loss"] = bad_metric
    _write_json(state_file, payload)

    with pytest.raises(RuntimeError, match="non-finite trainer metric"):
        structured_evidence.verify_save_world(publish_root)


@pytest.mark.parametrize("evidence_kind", ["manifest", "complete"])
def test_save_world_rejects_manifest_or_complete_metadata_mismatch(
    structured_evidence: ModuleType,
    publish_root: Path,
    evidence_kind: str,
) -> None:
    _write_state(publish_root, "save_world", 1, _run_contract())
    weights = _write_weights(publish_root, "save_world", 1)
    manifest = weights.with_name(f"{weights.name}.manifest.json")
    complete = weights.with_name(f"{weights.name}.COMPLETE")
    if evidence_kind == "manifest":
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["file"]["bytes"] += 1
        _write_json(manifest, payload)
        expected_message = "weights manifest is inconsistent"
    else:
        payload = json.loads(complete.read_text(encoding="utf-8"))
        payload["manifest_file"]["mtime_ns"] += 1
        _write_json(complete, payload)
        expected_message = "weights COMPLETE metadata is inconsistent"

    with pytest.raises(RuntimeError, match=expected_message):
        structured_evidence.verify_save_world(publish_root)


@pytest.mark.parametrize(
    "missing_relative",
    [
        "pytorch_model/bf16_zero_pp_rank_3_mp_rank_00_optim_states.pt",
        "random_states_4.pkl",
        "scheduler.bin",
    ],
    ids=("optimizer-shard", "random-state", "scheduler"),
)
def test_save_world_rejects_missing_zero2_full_state_component(
    structured_evidence: ModuleType,
    publish_root: Path,
    missing_relative: str,
) -> None:
    state_dir = _write_world(publish_root, "save_world", 1, _run_contract())
    (state_dir / missing_relative).unlink()

    with pytest.raises(RuntimeError, match="lacks required components"):
        structured_evidence.verify_save_world(publish_root)


@pytest.mark.parametrize(
    "extra_relative",
    [
        "random_states_8.pkl",
        "pytorch_model/bf16_zero_pp_rank_8_mp_rank_00_optim_states.pt",
        "pytorch_model/mp_rank_01_model_states.pt",
    ],
    ids=("extra-random-rank", "extra-optimizer-rank", "extra-model-rank"),
)
def test_save_world_rejects_extra_zero2_rank_component(
    structured_evidence: ModuleType,
    publish_root: Path,
    extra_relative: str,
) -> None:
    state_dir = _write_world(publish_root, "save_world", 1, _run_contract())
    (state_dir / extra_relative).write_bytes(b"unexpected rank\n")

    with pytest.raises(RuntimeError, match="extra ranks"):
        structured_evidence.verify_save_world(publish_root)


@pytest.mark.parametrize(
    "empty_relative",
    ["scheduler.bin", "random_states_0.pkl"],
    ids=("empty-scheduler", "empty-random-state"),
)
def test_save_world_rejects_empty_required_zero2_component(
    structured_evidence: ModuleType,
    publish_root: Path,
    empty_relative: str,
) -> None:
    state_dir = _write_world(publish_root, "save_world", 1, _run_contract())
    (state_dir / empty_relative).write_bytes(b"")

    with pytest.raises(RuntimeError, match="required state components are empty"):
        structured_evidence.verify_save_world(publish_root)


def test_save_world_rejects_latest_target_drift(
    structured_evidence: ModuleType,
    publish_root: Path,
) -> None:
    state_dir = _write_world(publish_root, "save_world", 1, _run_contract())
    (state_dir / "latest").write_bytes(b"not_pytorch_model")

    with pytest.raises(RuntimeError, match="must contain exactly 'pytorch_model'"):
        structured_evidence.verify_save_world(publish_root)


@pytest.mark.parametrize(
    "drift_kind",
    ["missing", "token-budget", "epoch", "global-batches", "negative-index"],
)
def test_save_world_rejects_missing_or_drifted_data_schedule(
    structured_evidence: ModuleType,
    publish_root: Path,
    drift_kind: str,
) -> None:
    state_dir = _write_world(publish_root, "save_world", 1, _run_contract())
    state_file = state_dir / "trainer_state.json"
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    if drift_kind == "missing":
        payload.pop("data_schedule")
    elif drift_kind == "token-budget":
        payload["data_schedule"]["agent_action_token_budget"] = 64
    elif drift_kind == "epoch":
        payload["data_schedule"]["epoch"] = 1
    elif drift_kind == "global-batches":
        payload["data_schedule"]["global_batches_per_epoch"] = 1344
    else:
        payload["data_schedule"]["batches"][0][0] = -1
    _write_json(state_file, payload)

    with pytest.raises(RuntimeError, match="data_schedule"):
        structured_evidence.verify_save_world(publish_root)


def test_save_world_rejects_cursor_beyond_microbatches_per_process(
    structured_evidence: ModuleType,
    publish_root: Path,
) -> None:
    state_dir = _write_world(publish_root, "save_world", 1, _run_contract())
    state_file = state_dir / "trainer_state.json"
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    payload["batch_in_epoch"] = 170
    _write_json(state_file, payload)

    with pytest.raises(RuntimeError, match="cursor"):
        structured_evidence.verify_save_world(publish_root)


def test_save_world_rejects_scientific_configuration_drift(
    structured_evidence: ModuleType,
    publish_root: Path,
) -> None:
    state_dir = _write_world(publish_root, "save_world", 1, _run_contract())
    state_file = state_dir / "trainer_state.json"
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    payload["run_contract"]["optimization"]["learning_rate"] = 1.0e-4
    _write_json(state_file, payload)

    with pytest.raises(RuntimeError, match="scientific configuration"):
        structured_evidence.verify_save_world(publish_root)


def test_recovery_rejects_native_receipt_source_metadata_drift(
    structured_evidence: ModuleType,
    publish_root: Path,
) -> None:
    _write_valid_recovery_worlds(publish_root)
    structured_evidence.verify_save_world(publish_root)
    receipt_path = publish_root / "load_world" / "recovery_load_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_trainer_state_file"]["mtime_ns"] += 1
    _write_json(receipt_path, receipt)

    with pytest.raises(RuntimeError, match="source trainer-state metadata drift"):
        structured_evidence.verify_recovery_worlds(publish_root)


def test_recovery_rejects_native_receipt_source_directory_drift(
    structured_evidence: ModuleType,
    publish_root: Path,
) -> None:
    _, final_state, _ = _write_valid_recovery_worlds(publish_root)
    structured_evidence.verify_save_world(publish_root)
    receipt_path = publish_root / "load_world" / "recovery_load_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_state_dir"] = str(final_state.resolve(strict=True))
    _write_json(receipt_path, receipt)

    with pytest.raises(RuntimeError, match="source_state_dir"):
        structured_evidence.verify_recovery_worlds(publish_root)


def test_recovery_rejects_missing_final_step_two_reload_receipt(
    structured_evidence: ModuleType,
    publish_root: Path,
) -> None:
    _, _, final_verify_output = _write_valid_recovery_worlds(publish_root)
    structured_evidence.verify_save_world(publish_root)
    (final_verify_output / "recovery_load_receipt.json").unlink()

    with pytest.raises(RuntimeError, match="recovery load receipt is missing"):
        structured_evidence.verify_recovery_worlds(publish_root)


def test_recovery_rejects_extra_native_receipt_field(
    structured_evidence: ModuleType,
    publish_root: Path,
) -> None:
    _write_valid_recovery_worlds(publish_root)
    structured_evidence.verify_save_world(publish_root)
    receipt_path = publish_root / "load_world" / "recovery_load_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["unexpected"] = "not trainer-native schema"
    _write_json(receipt_path, receipt)

    with pytest.raises(RuntimeError, match="fields differ from the exact schema"):
        structured_evidence.verify_recovery_worlds(publish_root)


def test_recovery_rejects_final_verify_checkpoint_entry(
    structured_evidence: ModuleType,
    publish_root: Path,
) -> None:
    _, _, final_verify_output = _write_valid_recovery_worlds(publish_root)
    structured_evidence.verify_save_world(publish_root)
    (final_verify_output / "checkpoints" / "weights" / "step_000002.pt").write_bytes(
        b"forbidden checkpoint\n"
    )

    with pytest.raises(RuntimeError, match="must not produce checkpoint entries"):
        structured_evidence.verify_recovery_worlds(publish_root)


def test_runtime_is_r3_and_forbids_the_old_rendered_log_fragment_gates() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")

    assert runtime.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert f'EXPECTED_EXPERIMENT="{R3_EXPERIMENT_ID}"' in runtime
    assert "GATE2-NOHASH-R2-" not in runtime
    assert '"${STRUCTURED_EVIDENCE_SCRIPT}" verify-save' in runtime
    assert '"${STRUCTURED_EVIDENCE_SCRIPT}" verify-recovery' in runtime
    assert (
        'launch_training load_world "${LOAD_OUTPUT}" "${SAVE_STATE}" null true'
        in runtime
    )
    assert '--num_processes 8' in runtime
    assert 'LOAD_RECEIPT="${LOAD_OUTPUT}/recovery_load_receipt.json"' in runtime
    assert (
        'FINAL_VERIFY_RECEIPT="${FINAL_VERIFY_OUTPUT}/recovery_load_receipt.json"'
        in runtime
    )
    assert (
        'launch_training final_verify_world "${FINAL_VERIFY_OUTPUT}" "${FINAL_STATE}"'
        in runtime
    )
    assert 'null false "${FINAL_VERIFY_RECEIPT}" false' in runtime
    for forbidden in (
        "pause_fragment",
        "resume_fragment",
        "checkpoint_fragment",
        "done_fragment",
        "save_log_text",
        "load_log_text",
        "last_train_line",
        "[recovery-gate]",
        "[resume]",
        "[train]",
        "[ckpt]",
        "[done]",
        ".read_text(",
    ):
        assert forbidden not in runtime
