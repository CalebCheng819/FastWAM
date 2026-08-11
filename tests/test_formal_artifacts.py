from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from fastwam.formal_artifacts import (
    ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT,
    ACTION_ONLY_N2_RELOAD_COMMITTED_ATTEMPT,
    ACTION_ONLY_N2_RELOAD_LOAD_ATTEMPTS_DIR,
    ACTION_ONLY_N2_RELOAD_PROOF_DIR,
    ACTION_ONLY_N2_RESERVATION_FIELDS,
    ACTION_ONLY_N2_TASK_SCOPE_SCHEMA,
    ACTION_ONLY_N2_TERMINAL_CANDIDATE,
    N4_GATE_MAX_PEAK_ALLOCATED_BYTES,
    N4_GATE_MAX_PEAK_RESERVED_BYTES,
    _summarize_n4_peak_memory,
    canonical_json_bytes,
    canonical_json_sha256,
    checkpoint_seal_descriptor,
    finalize_action_only_n2_paid_gate,
    model_probe,
    optimizer_probe,
    publish_action_only_n2_reload_attempt_commit,
    publish_action_only_n2_terminal_candidate,
    publish_exclusive_bytes,
    publish_failure_marker,
    publish_training_terminal_seal,
    read_canonical_json,
    resolved_unaliased_directory,
    validate_action_only_n2_reload_proof,
    validate_action_only_n2_terminal_reservation,
    validate_n4_fullmodel_gate_binding,
)
from fastwam.trainer import Wan22Trainer


def test_publish_exclusive_bytes_never_overwrites_existing_target(tmp_path):
    destination = tmp_path / "COMPLETE"
    destination.write_bytes(b"immutable-original")

    with pytest.raises(FileExistsError):
        publish_exclusive_bytes(destination, b"replacement")

    assert destination.read_bytes() == b"immutable-original"
    assert not list(tmp_path.glob(".COMPLETE.tmp.*"))


def test_publish_exclusive_bytes_concurrent_writers_have_one_winner(tmp_path):
    destination = tmp_path / "manifest.json"
    payloads = (b"first\n", b"second\n")

    def publish(payload):
        try:
            publish_exclusive_bytes(destination, payload)
            return "published"
        except FileExistsError:
            return "exists"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, payloads))

    assert sorted(outcomes) == ["exists", "published"]
    assert destination.read_bytes() in payloads
    assert not list(tmp_path.glob(".manifest.json.tmp.*"))


def test_publish_exclusive_bytes_rejects_symlink_target_without_touching_victim(tmp_path):
    victim = tmp_path / "victim"
    victim.write_bytes(b"victim-data")
    destination = tmp_path / "TRAINING.COMPLETE"
    destination.symlink_to(victim)

    with pytest.raises(FileExistsError):
        publish_exclusive_bytes(destination, b"replacement")

    assert destination.is_symlink()
    assert victim.read_bytes() == b"victim-data"


def test_critical_artifact_io_rejects_symlinked_ancestor_components(tmp_path):
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    payload_path = real_root / "payload.json"
    publish_exclusive_bytes(payload_path, canonical_json_bytes({"value": 1}))
    alias_root = tmp_path / "alias-root"
    alias_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises((OSError, RuntimeError, ValueError)):
        resolved_unaliased_directory(alias_root, label="aliased root")
    with pytest.raises((OSError, RuntimeError, ValueError)):
        read_canonical_json(alias_root / "payload.json")
    with pytest.raises((OSError, RuntimeError, ValueError)):
        publish_exclusive_bytes(alias_root / "new.json", b"protected\n")

    assert payload_path.read_bytes() == canonical_json_bytes({"value": 1})
    assert not (real_root / "new.json").exists()


def test_training_complete_and_failed_are_bidirectionally_exclusive(tmp_path):
    failed_first = tmp_path / "failed-first"
    failed_first.mkdir()
    publish_failure_marker(
        failed_first,
        marker_name="TRAINING.FAILED.json",
        schema_name="fastwam-training-failure",
        error=RuntimeError("attempt failed"),
        success_markers=("TRAINING.COMPLETE",),
    )
    with pytest.raises(RuntimeError, match="success after a failure marker"):
        publish_training_terminal_seal(
            failed_first,
            run_id="unused",
            code_commit="a" * 40,
            config_relative_path="resolved-config.yaml",
            config_sha256="b" * 64,
            max_steps=1,
            expected_checkpoint_steps=[1],
            expected_evaluation_steps=[],
            world_size=8,
            last_step_metrics={},
            evaluation_records=[],
            training_mode="action_only_cache",
            dataset_contract_sha256="c" * 64,
            authorization_gate_complete_sha256="",
        )

    complete_first = tmp_path / "complete-first"
    complete_first.mkdir()
    publish_exclusive_bytes(
        complete_first / "TRAINING.COMPLETE", canonical_json_bytes({"status": "PASS"})
    )
    with pytest.raises(RuntimeError, match="after a success marker exists"):
        publish_failure_marker(
            complete_first,
            marker_name="TRAINING.FAILED.json",
            schema_name="fastwam-training-failure",
            error=RuntimeError("late failure"),
            success_markers=("TRAINING.COMPLETE",),
        )
    assert not (complete_first / "TRAINING.FAILED.json").exists()


def test_full_state_probes_cover_more_than_eight_parameters_buffers_and_adamw_state():
    class ManyParameters(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2) for _ in range(6)]
            )
            self.register_buffer("running_scale", torch.ones(3))

        def forward(self, value):
            for layer in self.layers:
                value = layer(value)
            return value

    model = ManyParameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    before = model_probe(model, full_state=True)
    optimizer_state = optimizer_probe(optimizer, full_state=True)

    assert before["coverage"] == "full_state_dict"
    assert before["inventory"]["inventory_count"] == len(model.state_dict())
    assert before["inventory"]["parameter_count"] > 8
    assert before["inventory"]["buffer_count"] == 1
    assert len(before["records"]) > 8
    assert optimizer_state["coverage"] == "rank_local_full_state_dict"
    assert optimizer_state["inventory"]["param_group_parameter_count"] == len(
        list(model.parameters())
    )
    assert optimizer_state["inventory"]["state_parameter_count"] == len(
        list(model.parameters())
    )
    assert optimizer_state["inventory"]["state_tensor_count"] >= 3 * len(
        list(model.parameters())
    )
    assert optimizer_state["state_records"]
    assert optimizer_state["param_groups"]

    model.running_scale.add_(1)
    after = model_probe(model, full_state=True)
    assert after["fingerprint"] != before["fingerprint"]


def _fake_deepspeed_zero2_optimizer(
    label,
    *,
    zero_stage=2,
    populated=True,
    ds_version="0.18.5",
    partition_count=None,
    param_slice_mappings=None,
):
    digest = hashlib.sha256(str(label).encode()).digest()
    base = torch.tensor(
        [float(digest[index]) for index in range(4)], dtype=torch.float32
    )
    loss_scaler_type = type(
        "DynamicLossScaler",
        (),
        {"__module__": "deepspeed.runtime.fp16.loss_scaler"},
    )
    loss_scaler = loss_scaler_type()
    loss_scaler.cur_scale = 65536.0
    loss_scaler.cur_iter = 1
    loss_scaler.dynamic = True
    optimizer_state = (
        {
            0: {
                "exp_avg": base.clone(),
                "exp_avg_sq": base.square(),
                "step": torch.tensor(1.0, dtype=torch.float32),
            }
        }
        if populated
        else {}
    )
    state_dict = {
        "base_optimizer_state": {
            "param_groups": [{"lr": 3e-5, "params": [0]}],
            "state": optimizer_state,
        },
        "clip_grad": 1.0,
        "ds_version": ds_version,
        "dynamic_loss_scale": True,
        "group_paddings": [0],
        "loss_scaler": loss_scaler,
        "overflow": False,
        "param_slice_mappings": (
            [{}] if param_slice_mappings is None else param_slice_mappings
        ),
        "partition_count": [8] if partition_count is None else partition_count,
        "single_partition_of_fp32_groups": [base.clone().add_(0.5)],
        "zero_stage": zero_stage,
    }
    optimizer_type = type(
        "DeepSpeedZeroOptimizer",
        (),
        {
            "__module__": "deepspeed.runtime.zero.stage_1_and_2",
            "partition_gradients": True,
            "state_dict": lambda self: self._state_dict,
        },
    )
    optimizer = optimizer_type()
    optimizer._state_dict = state_dict
    return optimizer


def test_full_state_probe_binds_deepspeed_zero2_wrapper_and_fp32_masters():
    zero2 = _fake_deepspeed_zero2_optimizer("rank-0")
    wrapped = SimpleNamespace(optimizer=zero2)

    before = optimizer_probe(wrapped, full_state=True)

    assert before["concrete_type"] == (
        "deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer"
    )
    assert before["coverage"] == "rank_local_deepspeed_zero2_state_dict"
    assert before["inventory"]["base_optimizer_state_parameter_count"] == 1
    assert before["inventory"]["fp32_master_partition_count"] == 1
    assert before["inventory"]["fp32_master_partition_numel"] == 4

    zero2._state_dict["single_partition_of_fp32_groups"][0].add_(1.0)
    after = optimizer_probe(wrapped, full_state=True)
    assert after["fingerprint"] != before["fingerprint"]

    with pytest.raises(ValueError, match="not ZeRO-2"):
        optimizer_probe(
            _fake_deepspeed_zero2_optimizer("wrong-stage", zero_stage=1),
            full_state=True,
        )
    with pytest.raises(ValueError, match="version mismatch"):
        optimizer_probe(
            _fake_deepspeed_zero2_optimizer(
                "wrong-version", ds_version="0.18.4"
            ),
            full_state=True,
        )
    with pytest.raises(ValueError, match="partition count"):
        optimizer_probe(
            _fake_deepspeed_zero2_optimizer(
                "scalar-partition-count", partition_count=8
            ),
            full_state=True,
        )
    with pytest.raises(TypeError, match="slice mappings"):
        optimizer_probe(
            _fake_deepspeed_zero2_optimizer(
                "mapping-shape", param_slice_mappings={}
            ),
            full_state=True,
        )


def test_deepspeed_gradient_evidence_uses_engine_norm_after_grads_are_cleared():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = SimpleNamespace(
        distributed_type=SimpleNamespace(name="DEEPSPEED")
    )
    trainer.model = torch.nn.Linear(2, 1)
    assert all(parameter.grad is None for parameter in trainer.model.parameters())

    evidence = trainer._n4_gate_gradient_evidence(torch.tensor(1.25))

    assert evidence == {
        "all_finite": True,
        "norm": 1.25,
        "source": "deepspeed_global_grad_norm",
    }
    with pytest.raises(RuntimeError, match="DeepSpeed global gradient norm"):
        trainer._n4_gate_gradient_evidence(torch.tensor(0.0))


def test_fresh_reload_config_contract_preserves_and_normalizes_terminal_authority():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.cfg = OmegaConf.create(
        {
            "learning_rate": 3e-5,
            "training_terminal_contract": ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT,
            "training_run_profile": "paid_gate_1step",
            "training_task_scope_receipt": "authorization/task-scope.json",
        }
    )
    trainer.n2_reload_proof_phase = None

    saved = trainer._resolved_config_contract()

    assert saved == {
        "learning_rate": 3e-5,
        "training_run_profile": "paid_gate_1step",
        "training_task_scope_receipt": "authorization/task-scope.json",
        "training_terminal_contract": ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT,
    }
    trainer.cfg.training_terminal_contract = None
    trainer.cfg.training_run_profile = None
    trainer.cfg.training_task_scope_receipt = None
    assert trainer._resolved_config_contract() != saved

    trainer.n2_reload_proof_phase = "load"
    trainer._read_n2_terminal_candidate = lambda: (
        {},
        {
            "run_profile": "paid_gate_1step",
            "task_scope_receipt_relative_path": "authorization/task-scope.json",
            "training_terminal_contract": ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT,
        },
        "a" * 64,
    )
    assert trainer._resolved_config_contract() == saved


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _n4_memory_step_proofs(
    *,
    smaller_capacity: int = 47_673_901_056,
    larger_capacity: int = 50_875_924_480,
):
    proofs = {}
    for step in (1, 2):
        step_payloads = []
        for rank in range(32):
            capacity = smaller_capacity if rank < 24 else larger_capacity
            step_payloads.append(
                {
                    "rank": rank,
                    "memory": {
                        "device_name": "NVIDIA GeForce RTX 4090",
                        "effective_max_allocated_bytes": min(
                            N4_GATE_MAX_PEAK_ALLOCATED_BYTES,
                            capacity * 90 // 100,
                        ),
                        "effective_max_reserved_bytes": min(
                            N4_GATE_MAX_PEAK_RESERVED_BYTES,
                            capacity * 95 // 100,
                        ),
                        "peak_allocated_bytes": 18 * 2**30 + rank,
                        "peak_reserved_bytes": 20 * 2**30 + rank,
                        "required_max_allocated_bytes": N4_GATE_MAX_PEAK_ALLOCATED_BYTES,
                        "required_max_reserved_bytes": N4_GATE_MAX_PEAK_RESERVED_BYTES,
                        "total_device_bytes": capacity,
                    },
                }
            )
        proofs[step] = step_payloads
    return proofs


def test_n4_memory_summary_accepts_stable_mixed_capacities_conservatively():
    proofs = _n4_memory_step_proofs()

    summary = _summarize_n4_peak_memory(proofs)

    minimum_capacity = 47_673_901_056
    assert summary == {
        "device_name": "NVIDIA GeForce RTX 4090",
        "effective_max_allocated_bytes": min(
            N4_GATE_MAX_PEAK_ALLOCATED_BYTES,
            minimum_capacity * 90 // 100,
        ),
        "effective_max_reserved_bytes": min(
            N4_GATE_MAX_PEAK_RESERVED_BYTES,
            minimum_capacity * 95 // 100,
        ),
        "total_device_bytes": minimum_capacity,
        "max_allocated_bytes": 18 * 2**30 + 31,
        "max_reserved_bytes": 20 * 2**30 + 31,
        "required_max_allocated_bytes": N4_GATE_MAX_PEAK_ALLOCATED_BYTES,
        "required_max_reserved_bytes": N4_GATE_MAX_PEAK_RESERVED_BYTES,
    }


def test_n4_memory_summary_rejects_different_device_names():
    proofs = _n4_memory_step_proofs()
    for step in (1, 2):
        proofs[step][31]["memory"]["device_name"] = "Different GPU"

    with pytest.raises(RuntimeError, match="same non-empty CUDA device name"):
        _summarize_n4_peak_memory(proofs)


def test_n4_memory_summary_rejects_per_rank_capacity_drift():
    proofs = _n4_memory_step_proofs()
    changed_capacity = 47_000_000_000
    memory = proofs[2][0]["memory"]
    memory["total_device_bytes"] = changed_capacity
    memory["effective_max_allocated_bytes"] = min(
        N4_GATE_MAX_PEAK_ALLOCATED_BYTES,
        changed_capacity * 90 // 100,
    )
    memory["effective_max_reserved_bytes"] = min(
        N4_GATE_MAX_PEAK_RESERVED_BYTES,
        changed_capacity * 95 // 100,
    )

    with pytest.raises(RuntimeError, match="changed between optimizer steps"):
        _summarize_n4_peak_memory(proofs)


def test_n4_memory_summary_rejects_rank_over_its_own_capacity_limit():
    proofs = _n4_memory_step_proofs()
    memory = proofs[1][0]["memory"]
    memory["peak_allocated_bytes"] = memory["effective_max_allocated_bytes"] + 1

    with pytest.raises(RuntimeError, match="exceeds memory gate"):
        _summarize_n4_peak_memory(proofs)


@pytest.mark.parametrize("peak_field", ["peak_allocated_bytes", "peak_reserved_bytes"])
def test_n4_memory_summary_rejects_peak_over_minimum_capacity_limit(peak_field):
    smaller_capacity = 40 * 2**30
    proofs = _n4_memory_step_proofs(
        smaller_capacity=smaller_capacity,
        larger_capacity=50 * 2**30,
    )
    minimum_limit = (
        smaller_capacity * (90 if peak_field == "peak_allocated_bytes" else 95) // 100
    )
    proofs[1][31]["memory"][peak_field] = minimum_limit + 1

    with pytest.raises(RuntimeError, match="conservative run-level memory gate"):
        _summarize_n4_peak_memory(proofs)


def test_n4_memory_summary_rejects_empty_device_name():
    proofs = _n4_memory_step_proofs()
    for step in (1, 2):
        proofs[step][0]["memory"]["device_name"] = ""

    with pytest.raises(RuntimeError, match="relative peak-memory threshold mismatch"):
        _summarize_n4_peak_memory(proofs)


def test_n4_memory_summary_rejects_non_string_device_name():
    proofs = _n4_memory_step_proofs()
    for step in (1, 2):
        proofs[step][0]["memory"]["device_name"] = None

    with pytest.raises(RuntimeError, match="relative peak-memory threshold mismatch"):
        _summarize_n4_peak_memory(proofs)


def _publish_json(path, payload) -> str:
    encoded = canonical_json_bytes(payload)
    publish_exclusive_bytes(path, encoded)
    return _sha256_bytes(encoded)


def _build_sealed_training_checkpoint(
    output, *, step, state_kind, trainer_state=None
):
    tag = f"step_{step:06d}"
    weights_dir = output / "checkpoints" / "weights"
    state_root = output / "checkpoints" / "state" / tag
    weights_dir.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)

    weights = weights_dir / f"{tag}.pt"
    weights_payload = f"weights-{state_kind}-{step}".encode()
    weights.write_bytes(weights_payload)
    weights_sha256 = _sha256_bytes(weights_payload)
    weights_manifest = {
        "bytes": len(weights_payload),
        "checkpoint_state_kind": state_kind,
        "filename": weights.name,
        "global_step": step,
        "schema_name": "fastwam-weights-checkpoint",
        "schema_version": 1,
        "sha256": weights_sha256,
    }
    weights_manifest_path = weights.with_name(f"{weights.name}.manifest.json")
    weights_manifest_sha256 = _publish_json(weights_manifest_path, weights_manifest)
    _publish_json(
        weights.with_name(f"{weights.name}.COMPLETE"),
        {
            "checkpoint_sha256": weights_sha256,
            "manifest_filename": weights_manifest_path.name,
            "manifest_sha256": weights_manifest_sha256,
            "schema_name": "fastwam-weights-checkpoint-complete",
            "schema_version": 1,
        },
    )

    state_payloads = {
        "trainer_state.json": canonical_json_bytes(
            {"global_step": step} if trainer_state is None else trainer_state
        ),
        "zero-state.bin": f"zero-state-{step}".encode(),
    }
    for relative, payload in state_payloads.items():
        (state_root / relative).write_bytes(payload)
    state_records = [
        {
            "bytes": len(payload),
            "path": relative,
            "sha256": _sha256_bytes(payload),
        }
        for relative, payload in sorted(state_payloads.items())
    ]
    _publish_json(
        state_root.with_name(f"{tag}.state-tree.json"),
        {
            "files": state_records,
            "role": "accelerate_zero2_full_state",
            "schema_version": 1,
            "total_bytes": sum(record["bytes"] for record in state_records),
        },
    )


def _n2_terminal_fixture(tmp_path, *, run_profile, checkpoint_steps):
    output = tmp_path / f"n2-{run_profile}"
    output.mkdir()
    code_commit = "a" * 40
    effective_patched_tree = "b" * 40
    request_sha256 = "c" * 64
    init_checkpoint_sha256 = "d" * 64
    task = "PlaceFood-rf"
    receipt_relative_path = "authorization/task-scope.json"
    receipt_path = output / receipt_relative_path
    receipt_path.parent.mkdir()
    receipt = {
        "pins": {
            "effective_patched_tree": effective_patched_tree,
            "init_checkpoint_sha256": init_checkpoint_sha256,
            "request_sha256": request_sha256,
            "task_scope_id": "placefood-n2-only",
        },
        "required_agent_counts": [2],
        "required_tasks": [task],
        "run_profile": run_profile,
        "schema_name": ACTION_ONLY_N2_TASK_SCOPE_SCHEMA,
        "schema_version": 1,
        "training_terminal_contract": ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT,
    }
    receipt_sha256 = _publish_json(receipt_path, receipt)
    reservation_payload = {
        "base_code_commit": code_commit,
        "bundle_manifest_sha256": None,
        "cache_manifest_sha256": None,
        "cache_selection_sha256": None,
        "cache_source_identity_sha256": None,
        "checkpoint_sha256": init_checkpoint_sha256,
        "checkpoint_state_kind": "sparse_delta",
        "code_commit": code_commit,
        "cpfs_bundle_manifest_sha256": None,
        "effective_patched_tree": effective_patched_tree,
        "erdma_bootstrap_sha256": None,
        "erdma_bundle_sha256": None,
        "erdma_env_sha256": None,
        "erdma_source_manifest_sha256": None,
        "formal_n4_fullmodel_gate": False,
        "global_world_size": 8,
        "image_digest": "sha256:" + "e" * 64,
        "image_digest_status": "resolved",
        "image_reference": "registry.example/fastwam@sha256:" + "e" * 64,
        "init_checkpoint_sha256": init_checkpoint_sha256,
        "n4_fullmodel_gate_complete_sha256": None,
        "nproc_per_node": 8,
        "num_machines": 1,
        "oss_bundle_manifest_sha256": None,
        "output_storage": "cpfs",
        "output_zero_checkpoint_smoke_sha256": None,
        "pyproject_sha256": "f" * 64,
        "request_sha256": request_sha256,
        "run_id": f"n2-{run_profile}",
        "run_profile": run_profile,
        "schema_version": 2,
        "stats_sha256": None,
        "task": "robofactory_multi_robot_ft_n2_placefood_vg0_hub1_gau1_224_3e-5",
        "task_scope_receipt": receipt_relative_path,
        "task_scope_receipt_sha256": receipt_sha256,
        "trainable_scope": "action",
        "training_env_bundle_manifest_sha256": None,
        "training_mode": "action_only_cache",
        "training_terminal_contract": ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT,
        "training_terminal_contract_version": 1,
        "vae_sha256": None,
    }
    reservation = {
        **reservation_payload,
        "identity_sha256": canonical_json_sha256(reservation_payload),
    }
    assert set(reservation) == ACTION_ONLY_N2_RESERVATION_FIELDS
    _publish_json(output / ".RUN_RESERVED", reservation)
    config = output / "resolved-config.yaml"
    config.write_text("training_terminal_contract: action_only_n2_1x8_v1\n")
    for step in checkpoint_steps:
        trainer_state = None
        if run_profile == "paid_gate_1step" and step == 1:
            trainer_state = {
                "batch_in_epoch": 4,
                "epoch": 0,
                "evaluation_records": [],
                "global_step": 1,
                "last_step_metrics": _terminal_metrics(1),
            }
        _build_sealed_training_checkpoint(
            output,
            step=step,
            state_kind="sparse_delta",
            trainer_state=trainer_state,
        )
    dataset_contract = {
        "train": {"required_agent_counts": [2], "required_tasks": [task]},
        "val": {"required_agent_counts": [2], "required_tasks": [task]},
    }
    return {
        "output": output,
        "run_id": f"n2-{run_profile}",
        "code_commit": code_commit,
        "effective_patched_tree": effective_patched_tree,
        "request_sha256": request_sha256,
        "init_checkpoint_sha256": init_checkpoint_sha256,
        "receipt_relative_path": receipt_relative_path,
        "receipt_sha256": receipt_sha256,
        "config_sha256": _sha256_bytes(config.read_bytes()),
        "dataset_contract": dataset_contract,
    }


def _terminal_metrics(step):
    return {
        "grad_norm": 0.5,
        "learning_rate": 3e-5,
        "loss": 1.25,
        "loss_components": {"loss_action": 1.25},
        "step": step,
    }


def _offline_record(step, *, samples, counts, tasks, training_mode):
    record = {
        "evaluation_kind": "multi_robot_offline_loss",
        "offline_agent_counts": counts,
        "offline_samples": samples,
        "offline_tasks": tasks,
        "step": step,
        "val_loss": 1.0,
        "val_loss_action": 1.0,
    }
    if training_mode == "joint":
        record["val_loss_video"] = 1.0
    return record


def _n2_terminal_arguments(
    fixture, *, run_profile, max_steps, checkpoint_steps, eval_steps
):
    evaluations = [
        _offline_record(
            step,
            samples=32,
            counts=[2],
            tasks=["PlaceFood-rf"],
            training_mode="action_only_cache",
        )
        for step in eval_steps
    ]
    return {
        "run_id": fixture["run_id"],
        "code_commit": fixture["code_commit"],
        "config_relative_path": "resolved-config.yaml",
        "config_sha256": fixture["config_sha256"],
        "max_steps": max_steps,
        "expected_checkpoint_steps": checkpoint_steps,
        "expected_evaluation_steps": eval_steps,
        "world_size": 8,
        "last_step_metrics": _terminal_metrics(max_steps),
        "evaluation_records": evaluations,
        "training_mode": "action_only_cache",
        "dataset_contract_sha256": canonical_json_sha256(
            fixture["dataset_contract"]
        ),
        "authorization_gate_complete_sha256": "",
        "training_terminal_contract": ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT,
        "formal_n4_fullmodel_gate": False,
        "checkpoint_state_kind": "sparse_delta",
        "trainable_scope": "action",
        "dataset_contract": fixture["dataset_contract"],
        "task_scope_receipt_relative_path": fixture["receipt_relative_path"],
        "effective_patched_tree": fixture["effective_patched_tree"],
        "request_sha256": fixture["request_sha256"],
        "init_checkpoint_sha256": fixture["init_checkpoint_sha256"],
        "offline_eval_num_samples": 0 if run_profile == "paid_gate_1step" else 32,
        "rehash_weights": True,
        "run_profile": run_profile,
    }


def _publish_n2_terminal(
    fixture, *, run_profile, max_steps, checkpoint_steps, eval_steps
):
    return publish_training_terminal_seal(
        fixture["output"],
        **_n2_terminal_arguments(
            fixture,
            run_profile=run_profile,
            max_steps=max_steps,
            checkpoint_steps=checkpoint_steps,
            eval_steps=eval_steps,
        ),
    )


def _n2_state_fingerprints(label, *, global_step):
    model_records = [
        {
            "bytes": 16,
            "dtype": "torch.float32",
            "kind": "parameter",
            "name": "action_head.weight",
            "numel": 4,
            "sha256": hashlib.sha256(f"model-{label}".encode()).hexdigest(),
            "shape": [2, 2],
        },
        {
            "bytes": 4,
            "dtype": "torch.float32",
            "kind": "buffer",
            "name": "running_scale",
            "numel": 1,
            "sha256": hashlib.sha256(f"buffer-{label}".encode()).hexdigest(),
            "shape": [1],
        },
    ]
    model_body = {
        "coverage": "full_state_dict",
        "inventory": {
            "buffer_count": 1,
            "extra_count": 0,
            "inventory_count": 2,
            "parameter_count": 1,
            "total_bytes": 20,
            "total_numel": 5,
        },
        "records": model_records,
    }
    optimizer_state = optimizer_probe(
        _fake_deepspeed_zero2_optimizer(f"optimizer-{label}"), full_state=True
    )
    model = canonical_json_sha256(model_body)
    optimizer = optimizer_state["fingerprint"]
    return {
        "global_step": global_step,
        "model": model,
        "model_probe": {
            **model_body,
            "fingerprint": model,
        },
        "optimizer": optimizer,
        "optimizer_probe": optimizer_state,
        "rng": hashlib.sha256(f"rng-{label}".encode()).hexdigest(),
        "scheduler": hashlib.sha256(f"scheduler-{label}".encode()).hexdigest(),
    }


def _publish_n2_reload_proof(
    fixture,
    *,
    proof_attempt_id="a" * 32,
    load_attempt_id="b" * 32,
    commit=True,
    extra_partial_attempt_id=None,
):
    output = fixture["output"]
    terminal_arguments = _n2_terminal_arguments(
        fixture,
        run_profile="paid_gate_1step",
        max_steps=1,
        checkpoint_steps=[1],
        eval_steps=[],
    )
    candidate = publish_action_only_n2_terminal_candidate(
        output,
        terminal_arguments=terminal_arguments,
    )
    terminal_arguments_sha256 = candidate["arguments_sha256"]
    terminal_candidate_sha256 = _sha256_bytes(
        (output / ACTION_ONLY_N2_TERMINAL_CANDIDATE).read_bytes()
    )
    checkpoint = checkpoint_seal_descriptor(
        output,
        step=1,
        rehash_weights=True,
        expected_checkpoint_state_kind="sparse_delta",
    )
    proof_dir = output / ACTION_ONLY_N2_RELOAD_PROOF_DIR
    proof_dir.mkdir()
    attempts_root = proof_dir / ACTION_ONLY_N2_RELOAD_LOAD_ATTEMPTS_DIR
    attempts_root.mkdir()
    load_attempt_dir = attempts_root / load_attempt_id
    load_attempt_dir.mkdir()
    binding_sha256 = _publish_json(
        proof_dir / "checkpoint-binding.json",
        {
            "checkpoint": checkpoint,
            "global_step": 1,
            "proof_attempt_id": proof_attempt_id,
            "run_id": fixture["run_id"],
            "schema_name": "fastwam-action-only-n2-reload-checkpoint-binding",
            "schema_version": 1,
            "terminal_arguments_sha256": terminal_arguments_sha256,
            "terminal_candidate_sha256": terminal_candidate_sha256,
            "world_size": 8,
        },
    )
    sampler_cursor = {
        "agent_action_token_budget": 128,
        "batch_in_epoch": 4,
        "epoch": 0,
        "global_batch_offset": 32,
        "global_batches_per_epoch": 64,
        "global_step": 1,
        "gradient_accumulation_steps": 4,
        "microbatches_per_process": 8,
        "num_processes": 8,
        "optimizer_steps_per_epoch": 2,
        "schedule_fingerprint": "e" * 64,
        "uses_agent_count_batch_sampler": True,
    }
    for rank in range(8):
        fingerprints = _n2_state_fingerprints(f"restored-rank-{rank}", global_step=1)
        next_sample = {
            "numpy": [rank + 0.1, rank + 0.2, rank + 0.3, rank + 0.4],
            "python": [rank + 0.5, rank + 0.6, rank + 0.7, rank + 0.8],
            "torch_cpu": [rank + 0.9, rank + 1.0, rank + 1.1, rank + 1.2],
            "torch_cuda": [rank + 1.3, rank + 1.4, rank + 1.5, rank + 1.6],
        }
        common = {
            "checkpoint": checkpoint,
            "checkpoint_binding_sha256": binding_sha256,
            "fingerprints": fingerprints,
            "global_step": 1,
            "next_rng_sample": next_sample,
            "proof_attempt_id": proof_attempt_id,
            "rank": rank,
            "run_id": fixture["run_id"],
            "sampler_cursor": sampler_cursor,
            "schema_version": 1,
            "terminal_arguments_sha256": terminal_arguments_sha256,
            "terminal_candidate_sha256": terminal_candidate_sha256,
            "world_size": 8,
        }
        _publish_json(
            proof_dir / f"save-rank-{rank:05d}.json",
            {
                **common,
                "phase": "save_after_sealed_checkpoint",
                "process_nonce": hashlib.sha256(
                    f"save-{rank}".encode()
                ).hexdigest()[:32],
                "process_pid": 1000 + rank,
                "process_start_ticks": 2000 + rank,
                "schema_name": "fastwam-action-only-n2-reload-save-proof",
            },
        )
        pre_load = _n2_state_fingerprints(f"pre-load-rank-{rank}", global_step=0)
        _publish_json(
            load_attempt_dir / f"load-rank-{rank:05d}.json",
            {
                **common,
                "checks": {
                    "checkpoint_binding": True,
                    "fresh_process": True,
                    "global_step": True,
                    "model": True,
                    "next_rng_sample": True,
                    "optimizer": True,
                    "pre_load_was_distinct": True,
                    "rng": True,
                    "sampler_cursor": True,
                    "scheduler": True,
                    "terminal_candidate": True,
                },
                "load_attempt_id": load_attempt_id,
                "phase": "load_fresh_process",
                "pre_load_fingerprints": pre_load,
                "process_nonce": hashlib.sha256(
                    f"load-{rank}".encode()
                ).hexdigest()[:32],
                "process_pid": 3000 + rank,
                "process_start_ticks": 4000 + rank,
                "schema_name": "fastwam-action-only-n2-reload-load-proof",
            },
        )

    if extra_partial_attempt_id is not None:
        partial_dir = attempts_root / extra_partial_attempt_id
        partial_dir.mkdir()
        partial = json.loads(
            (load_attempt_dir / "load-rank-00000.json").read_text(encoding="utf-8")
        )
        partial["load_attempt_id"] = extra_partial_attempt_id
        partial["process_nonce"] = hashlib.sha256(
            f"partial-{extra_partial_attempt_id}".encode()
        ).hexdigest()[:32]
        _publish_json(partial_dir / "load-rank-00000.json", partial)
        (partial_dir / "stale.tmp").write_bytes(b"interrupted-attempt")
        (partial_dir / "nested-staging").mkdir()

    reload_proof = validate_action_only_n2_reload_proof(
        output,
        run_id=fixture["run_id"],
        checkpoint=checkpoint,
        terminal_arguments_sha256=terminal_arguments_sha256,
        load_attempt_id=load_attempt_id,
        require_committed=False,
    )
    committed = None
    if commit:
        committed = publish_action_only_n2_reload_attempt_commit(
            output,
            run_id=fixture["run_id"],
            checkpoint=checkpoint,
            terminal_arguments_sha256=terminal_arguments_sha256,
            load_attempt_id=load_attempt_id,
        )
    return {
        "checkpoint": checkpoint,
        "committed": committed,
        "load_attempt_id": load_attempt_id,
        "reload_proof": reload_proof,
        "terminal_arguments_sha256": terminal_arguments_sha256,
    }


def test_n2_paid_gate_synthetic_schema_proof_finalizer_publishes_schema_v2(tmp_path):
    # This validates the artifact schema/finalizer only.  It is deliberately
    # not evidence that two real, independent 8-rank CUDA worlds executed.
    fixture = _n2_terminal_fixture(
        tmp_path,
        run_profile="paid_gate_1step",
        checkpoint_steps=[1],
    )
    _publish_n2_reload_proof(fixture)

    complete = finalize_action_only_n2_paid_gate(fixture["output"])

    summary = json.loads(
        (fixture["output"] / "training-summary.json").read_text(encoding="utf-8")
    )
    assert complete["schema_version"] == 2
    assert complete["run_profile"] == "paid_gate_1step"
    assert complete["checkpoint_state_kind"] == "sparse_delta"
    assert complete["task_scope_receipt_sha256"] == fixture["receipt_sha256"]
    assert complete["fresh_process_reload_verified"] is True
    assert complete["proof_attempt_id"] == "a" * 32
    assert complete["load_attempt_id"] == "b" * 32
    assert complete["terminal_arguments_sha256"]
    assert complete["terminal_candidate_sha256"]
    assert complete["rank_state_aggregate_sha256"]
    assert fixture["receipt_relative_path"] in complete["bound_paths"]
    assert ACTION_ONLY_N2_TERMINAL_CANDIDATE in complete["bound_paths"]
    assert (
        f"{ACTION_ONLY_N2_RELOAD_PROOF_DIR}/"
        f"{ACTION_ONLY_N2_RELOAD_COMMITTED_ATTEMPT}"
    ) in complete["bound_paths"]
    assert summary["evaluation_records"] == []
    assert summary["offline_eval_num_samples"] == 0
    assert summary["request_sha256"] == fixture["request_sha256"]
    assert summary["reload_proof"]["fresh_process_verified"] is True
    assert summary["reload_proof"]["proof_attempt_id"] == "a" * 32
    assert summary["reload_proof"]["load_attempt_id"] == "b" * 32
    assert summary["reload_proof"]["verified_ranks"] == list(range(8))
    assert [
        record["rank"]
        for record in summary["reload_proof"]["rank_state_inventory"]
    ] == list(range(8))
    assert summary["reload_proof"]["rank_state_aggregate_sha256"] == (
        canonical_json_sha256(
            {
                "rank_state_inventory": summary["reload_proof"][
                    "rank_state_inventory"
                ]
            }
        )
    )
    assert summary["treatment"] == {
        "checkpoint_state_kind": "sparse_delta",
        "formal_n4_fullmodel_gate": False,
        "trainable_scope": "action",
        "training_mode": "action_only_cache",
        "video_gen": False,
    }


def test_n2_reload_ignores_partial_attempt_then_commits_complete_retry(tmp_path):
    fixture = _n2_terminal_fixture(
        tmp_path,
        run_profile="paid_gate_1step",
        checkpoint_steps=[1],
    )
    failed_attempt_id = "c" * 32
    _publish_n2_reload_proof(
        fixture, extra_partial_attempt_id=failed_attempt_id
    )

    complete = finalize_action_only_n2_paid_gate(fixture["output"])

    partial = (
        fixture["output"]
        / ACTION_ONLY_N2_RELOAD_PROOF_DIR
        / ACTION_ONLY_N2_RELOAD_LOAD_ATTEMPTS_DIR
        / failed_attempt_id
        / "load-rank-00000.json"
    )
    assert partial.is_file()
    assert complete["load_attempt_id"] == "b" * 32
    assert complete["fresh_process_reload_verified"] is True


def test_n2_reload_requires_immutable_committed_attempt(tmp_path):
    fixture = _n2_terminal_fixture(
        tmp_path,
        run_profile="paid_gate_1step",
        checkpoint_steps=[1],
    )
    _publish_n2_reload_proof(fixture, commit=False)

    with pytest.raises(FileNotFoundError):
        finalize_action_only_n2_paid_gate(fixture["output"])

    assert not (fixture["output"] / "TRAINING.FAILED.json").exists()
    assert not (fixture["output"] / "TRAINING.COMPLETE").exists()


def test_n2_reload_rejects_incomplete_committed_attempt(tmp_path):
    fixture = _n2_terminal_fixture(
        tmp_path,
        run_profile="paid_gate_1step",
        checkpoint_steps=[1],
    )
    incomplete_attempt_id = "c" * 32
    _publish_n2_reload_proof(
        fixture,
        commit=False,
        extra_partial_attempt_id=incomplete_attempt_id,
    )
    commit_path = (
        fixture["output"]
        / ACTION_ONLY_N2_RELOAD_PROOF_DIR
        / ACTION_ONLY_N2_RELOAD_COMMITTED_ATTEMPT
    )
    publish_exclusive_bytes(
        commit_path,
        canonical_json_bytes({"load_attempt_id": incomplete_attempt_id}),
    )

    with pytest.raises(RuntimeError, match="selected load-attempt inventory mismatch"):
        finalize_action_only_n2_paid_gate(fixture["output"])

    assert not (fixture["output"] / "TRAINING.COMPLETE").exists()


def test_n2_reload_commit_is_exclusive_and_never_overwritten(tmp_path):
    fixture = _n2_terminal_fixture(
        tmp_path,
        run_profile="paid_gate_1step",
        checkpoint_steps=[1],
    )
    result = _publish_n2_reload_proof(fixture, commit=False)
    first = publish_action_only_n2_reload_attempt_commit(
        fixture["output"],
        run_id=fixture["run_id"],
        checkpoint=result["checkpoint"],
        terminal_arguments_sha256=result["terminal_arguments_sha256"],
        load_attempt_id=result["load_attempt_id"],
    )
    commit_path = fixture["output"] / first["path"]
    original = commit_path.read_bytes()

    with pytest.raises(FileExistsError):
        publish_action_only_n2_reload_attempt_commit(
            fixture["output"],
            run_id=fixture["run_id"],
            checkpoint=result["checkpoint"],
            terminal_arguments_sha256=result["terminal_arguments_sha256"],
            load_attempt_id=result["load_attempt_id"],
        )

    assert commit_path.read_bytes() == original


def test_n2_reload_commit_binds_selected_proof_bytes(tmp_path):
    fixture = _n2_terminal_fixture(
        tmp_path,
        run_profile="paid_gate_1step",
        checkpoint_steps=[1],
    )
    _publish_n2_reload_proof(fixture)
    load_path = (
        fixture["output"]
        / ACTION_ONLY_N2_RELOAD_PROOF_DIR
        / ACTION_ONLY_N2_RELOAD_LOAD_ATTEMPTS_DIR
        / ("b" * 32)
        / "load-rank-00000.json"
    )
    payload = json.loads(load_path.read_text(encoding="utf-8"))
    payload["process_pid"] += 100
    load_path.chmod(0o600)
    load_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(
        RuntimeError,
        match="committed load attempt does not match validated proof evidence",
    ):
        finalize_action_only_n2_paid_gate(fixture["output"])

    assert not (fixture["output"] / "TRAINING.COMPLETE").exists()


def test_n2_reload_rejects_unsafe_load_attempt_id(tmp_path):
    fixture = _n2_terminal_fixture(
        tmp_path,
        run_profile="paid_gate_1step",
        checkpoint_steps=[1],
    )
    result = _publish_n2_reload_proof(fixture, commit=False)

    with pytest.raises(ValueError, match="32 lowercase hexadecimal"):
        validate_action_only_n2_reload_proof(
            fixture["output"],
            run_id=fixture["run_id"],
            checkpoint=result["checkpoint"],
            terminal_arguments_sha256=result["terminal_arguments_sha256"],
            load_attempt_id="../unsafe",
            require_committed=False,
        )


def test_n2_paid_gate_rejects_missing_fresh_process_reload_proof(tmp_path):
    fixture = _n2_terminal_fixture(
        tmp_path,
        run_profile="paid_gate_1step",
        checkpoint_steps=[1],
    )
    publish_action_only_n2_terminal_candidate(
        fixture["output"],
        terminal_arguments=_n2_terminal_arguments(
            fixture,
            run_profile="paid_gate_1step",
            max_steps=1,
            checkpoint_steps=[1],
            eval_steps=[],
        ),
    )

    with pytest.raises(FileNotFoundError):
        _publish_n2_terminal(
            fixture,
            run_profile="paid_gate_1step",
            max_steps=1,
            checkpoint_steps=[1],
            eval_steps=[],
        )

    assert not (fixture["output"] / "training-summary.json").exists()
    assert not (fixture["output"] / "SHA256SUMS").exists()
    assert not (fixture["output"] / "TRAINING.COMPLETE").exists()


def test_n2_paid_gate_rejects_tampered_reload_semantics(tmp_path):
    fixture = _n2_terminal_fixture(
        tmp_path,
        run_profile="paid_gate_1step",
        checkpoint_steps=[1],
    )
    _publish_n2_reload_proof(fixture)
    proof_path = (
        fixture["output"]
        / ACTION_ONLY_N2_RELOAD_PROOF_DIR
        / ACTION_ONLY_N2_RELOAD_LOAD_ATTEMPTS_DIR
        / ("b" * 32)
        / "load-rank-00000.json"
    )
    payload = json.loads(proof_path.read_text(encoding="utf-8"))
    payload["next_rng_sample"]["torch_cuda"][0] += 1.0
    proof_path.chmod(0o600)
    proof_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(RuntimeError, match="fresh reload semantic mismatch"):
        _publish_n2_terminal(
            fixture,
            run_profile="paid_gate_1step",
            max_steps=1,
            checkpoint_steps=[1],
            eval_steps=[],
        )

    assert not (fixture["output"] / "training-summary.json").exists()
    assert not (fixture["output"] / "SHA256SUMS").exists()
    assert not (fixture["output"] / "TRAINING.COMPLETE").exists()


def test_n2_paid_gate_rejects_missing_proof_attempt_id(tmp_path):
    fixture = _n2_terminal_fixture(
        tmp_path,
        run_profile="paid_gate_1step",
        checkpoint_steps=[1],
    )
    _publish_n2_reload_proof(fixture)
    binding_path = (
        fixture["output"]
        / ACTION_ONLY_N2_RELOAD_PROOF_DIR
        / "checkpoint-binding.json"
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    del binding["proof_attempt_id"]
    binding_path.chmod(0o600)
    binding_path.write_bytes(canonical_json_bytes(binding))

    with pytest.raises(ValueError, match="checkpoint binding fields mismatch"):
        _publish_n2_terminal(
            fixture,
            run_profile="paid_gate_1step",
            max_steps=1,
            checkpoint_steps=[1],
            eval_steps=[],
        )


def test_n2_paid_gate_rejects_mixed_proof_attempt_ids(tmp_path):
    fixture = _n2_terminal_fixture(
        tmp_path,
        run_profile="paid_gate_1step",
        checkpoint_steps=[1],
    )
    _publish_n2_reload_proof(fixture)
    proof_path = (
        fixture["output"]
        / ACTION_ONLY_N2_RELOAD_PROOF_DIR
        / ACTION_ONLY_N2_RELOAD_LOAD_ATTEMPTS_DIR
        / ("b" * 32)
        / "load-rank-00007.json"
    )
    payload = json.loads(proof_path.read_text(encoding="utf-8"))
    payload["proof_attempt_id"] = "b" * 32
    proof_path.chmod(0o600)
    proof_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(RuntimeError, match="proof identity mismatch at rank 7"):
        _publish_n2_terminal(
            fixture,
            run_profile="paid_gate_1step",
            max_steps=1,
            checkpoint_steps=[1],
            eval_steps=[],
        )


def test_n2_paid_gate_rejects_preload_distinct_only_by_global_step(tmp_path):
    fixture = _n2_terminal_fixture(
        tmp_path,
        run_profile="paid_gate_1step",
        checkpoint_steps=[1],
    )
    _publish_n2_reload_proof(fixture)
    proof_dir = fixture["output"] / ACTION_ONLY_N2_RELOAD_PROOF_DIR
    saved = json.loads(
        (proof_dir / "save-rank-00000.json").read_text(encoding="utf-8")
    )
    load_path = (
        proof_dir
        / ACTION_ONLY_N2_RELOAD_LOAD_ATTEMPTS_DIR
        / ("b" * 32)
        / "load-rank-00000.json"
    )
    loaded = json.loads(load_path.read_text(encoding="utf-8"))
    pre_load = deepcopy(saved["fingerprints"])
    pre_load["global_step"] = 0
    loaded["pre_load_fingerprints"] = pre_load
    load_path.chmod(0o600)
    load_path.write_bytes(canonical_json_bytes(loaded))

    with pytest.raises(RuntimeError, match="fresh reload semantic mismatch on rank 0"):
        _publish_n2_terminal(
            fixture,
            run_profile="paid_gate_1step",
            max_steps=1,
            checkpoint_steps=[1],
            eval_steps=[],
        )


def test_n2_formal_1k_terminal_contract_requires_exact_schedule_and_eval_scope(tmp_path):
    fixture = _n2_terminal_fixture(
        tmp_path,
        run_profile="formal_1k",
        checkpoint_steps=[500, 1000],
    )

    complete = _publish_n2_terminal(
        fixture,
        run_profile="formal_1k",
        max_steps=1000,
        checkpoint_steps=[500, 1000],
        eval_steps=[500, 1000],
    )

    summary = json.loads(
        (fixture["output"] / "training-summary.json").read_text(encoding="utf-8")
    )
    assert complete["schema_version"] == 2
    assert complete["run_profile"] == "formal_1k"
    assert [record["step"] for record in summary["evaluation_records"]] == [500, 1000]
    assert all(
        record["offline_agent_counts"] == [2]
        and record["offline_samples"] == 32
        and record["offline_tasks"] == ["PlaceFood-rf"]
        for record in summary["evaluation_records"]
    )


def test_n2_terminal_profile_is_bound_across_receipt_and_reservation(tmp_path):
    fixture = _n2_terminal_fixture(
        tmp_path,
        run_profile="paid_gate_1step",
        checkpoint_steps=[1],
    )

    with pytest.raises(RuntimeError, match="task-scope receipt"):
        validate_action_only_n2_terminal_reservation(
            fixture["output"],
            run_id=fixture["run_id"],
            base_code_commit=fixture["code_commit"],
            effective_patched_tree=fixture["effective_patched_tree"],
            request_sha256=fixture["request_sha256"],
            init_checkpoint_sha256=fixture["init_checkpoint_sha256"],
            world_size=8,
            formal_n4_fullmodel_gate=False,
            checkpoint_state_kind="sparse_delta",
            trainable_scope="action",
            training_mode="action_only_cache",
            dataset_contract=fixture["dataset_contract"],
            task_scope_receipt_relative_path=fixture["receipt_relative_path"],
            run_profile="formal_1k",
        )


def test_legacy_n4_terminal_publisher_keeps_schema_v1_shape(tmp_path):
    output = tmp_path / "legacy-n4"
    output.mkdir()
    code_commit = "7" * 40
    gate_sha256 = "8" * 64
    run_id = "legacy-n4"
    reservation_payload = {
        "code_commit": code_commit,
        "global_world_size": 32,
        "n4_fullmodel_gate_complete_sha256": gate_sha256,
        "run_id": run_id,
        "schema_version": 1,
    }
    _publish_json(
        output / ".RUN_RESERVED",
        {
            **reservation_payload,
            "identity_sha256": canonical_json_sha256(reservation_payload),
        },
    )
    config = output / "resolved-config.yaml"
    config.write_text("max_steps: 2\n", encoding="utf-8")
    _build_sealed_training_checkpoint(output, step=2, state_kind="full")

    complete = publish_training_terminal_seal(
        output,
        run_id=run_id,
        code_commit=code_commit,
        config_relative_path=config.name,
        config_sha256=_sha256_bytes(config.read_bytes()),
        max_steps=2,
        expected_checkpoint_steps=[2],
        expected_evaluation_steps=[2],
        world_size=32,
        last_step_metrics=_terminal_metrics(2),
        evaluation_records=[
            _offline_record(
                2,
                samples=12,
                counts=[2, 3, 4],
                tasks=[
                    "FourRobotsStackCube-rf",
                    "PlaceFood-rf",
                    "ThreeRobotsStackCube-rf",
                ],
                training_mode="joint",
            )
        ],
        training_mode="joint",
        dataset_contract_sha256="9" * 64,
        authorization_gate_complete_sha256=gate_sha256,
    )

    summary = json.loads(
        (output / "training-summary.json").read_text(encoding="utf-8")
    )
    assert set(complete) == {
        "bound_paths",
        "max_steps",
        "run_id",
        "schema_name",
        "schema_version",
        "sha256sums_sha256",
        "status",
        "summary_sha256",
        "world_size",
    }
    assert complete["schema_version"] == 1
    assert summary["schema_version"] == 1
    assert summary["authorization_gate_complete_sha256"] == gate_sha256
    assert summary["treatment"] == {"training_mode": "joint", "video_gen": True}


def _build_minimal_sealed_gate(tmp_path):
    allowed = tmp_path / "oss"
    output = allowed / "gate"
    output.mkdir(parents=True)
    weights_dir = output / "checkpoints" / "weights"
    state_root = output / "checkpoints" / "state" / "step_000002"
    weights_dir.mkdir(parents=True)
    state_root.mkdir(parents=True)

    weights = weights_dir / "step_000002.pt"
    weights_payload = b"weight-data"
    weights.write_bytes(weights_payload)
    weights_sha256 = _sha256_bytes(weights_payload)
    weights_manifest = {
        "bytes": len(weights_payload),
        "checkpoint_state_kind": "full",
        "filename": weights.name,
        "global_step": 2,
        "schema_name": "fastwam-weights-checkpoint",
        "schema_version": 1,
        "sha256": weights_sha256,
    }
    weights_manifest_path = weights.with_name(f"{weights.name}.manifest.json")
    weights_manifest_sha256 = _publish_json(weights_manifest_path, weights_manifest)
    _publish_json(
        weights.with_name(f"{weights.name}.COMPLETE"),
        {
            "checkpoint_sha256": weights_sha256,
            "manifest_filename": weights_manifest_path.name,
            "manifest_sha256": weights_manifest_sha256,
            "schema_name": "fastwam-weights-checkpoint-complete",
            "schema_version": 1,
        },
    )

    trainer_state = canonical_json_bytes({"global_step": 2})
    shard_payload = b"zero-state-shard"
    (state_root / "trainer_state.json").write_bytes(trainer_state)
    shard_path = state_root / "zero-state.bin"
    shard_path.write_bytes(shard_payload)
    state_records = [
        {
            "bytes": len(payload),
            "path": name,
            "sha256": _sha256_bytes(payload),
        }
        for name, payload in sorted(
            (
                ("trainer_state.json", trainer_state),
                ("zero-state.bin", shard_payload),
            )
        )
    ]
    _publish_json(
        state_root.with_name("step_000002.state-tree.json"),
        {
            "files": state_records,
            "role": "accelerate_zero2_full_state",
            "schema_version": 1,
            "total_bytes": sum(record["bytes"] for record in state_records),
        },
    )

    reservation_payload = {"schema_version": 1}
    reservation = {
        **reservation_payload,
        "identity_sha256": canonical_json_sha256(reservation_payload),
    }
    reservation_sha256 = _publish_json(output / ".RUN_RESERVED", reservation)
    checkpoint = checkpoint_seal_descriptor(output, step=2, rehash_weights=True)
    input_bindings = {
        "cpfs_bundle_manifest": "1" * 64,
        "gaussian_cache_manifest": "2" * 64,
        "gaussian_cache_selection": "3" * 64,
        "gaussian_cache_source_identity": "4" * 64,
        "official_checkpoint": "5" * 64,
        "oss_bundle_manifest": "6" * 64,
        "stats": "7" * 64,
        "synthetic_zero2_gate": "8" * 64,
        "training_environment_bundle": "9" * 64,
        "vae": "a" * 64,
    }
    code_commit = "b" * 40
    image_reference = "registry.example/fastwam:test"
    image_digest = "sha256:" + "c" * 64
    manifest = {
        "batch_accounting": {
            "global_train_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "local_micro_batch_size": 1,
            "world_size": 32,
        },
        "checkpoint": checkpoint,
        "code_commit": code_commit,
        "image_digest": image_digest,
        "image_reference": image_reference,
        "input_bindings": input_bindings,
        "peak_memory": {},
        "proof_counts": {
            "load_state": 32,
            "save_state": 32,
            "step_1": 32,
            "step_2": 32,
        },
        "published_at": "2026-08-02T00:00:00+00:00",
        "reservation": {
            "identity_sha256": reservation["identity_sha256"],
            "path": ".RUN_RESERVED",
            "sha256": reservation_sha256,
        },
        "roundtrip": {
            "global_step": True,
            "model": True,
            "optimizer": True,
            "pre_load_was_distinct": True,
            "rng": True,
            "rng_next_sample": True,
            "scheduler": True,
            "separate_process": True,
        },
        "run_id": "gate-test",
        "schema_name": "fastwam-n4-fullmodel-gate",
        "schema_version": 1,
        "status": "PASS",
        "train_steps": 2,
        "world_size": 32,
        "zero_stage": 2,
    }
    manifest_sha256 = _publish_json(output / "manifest.json", manifest)
    bound_paths = [".RUN_RESERVED", "manifest.json"]
    sha_payload = b"".join(
        f"{hashlib.sha256((output / relative).read_bytes()).hexdigest()}  {relative}\n".encode()
        for relative in bound_paths
    )
    publish_exclusive_bytes(output / "SHA256SUMS", sha_payload)
    complete_sha256 = _publish_json(
        output / "COMPLETE",
        {
            "bound_paths": bound_paths,
            "manifest_sha256": manifest_sha256,
            "run_id": "gate-test",
            "schema_name": "fastwam-n4-fullmodel-gate-complete",
            "schema_version": 1,
            "sha256sums_sha256": _sha256_bytes(sha_payload),
            "status": "PASS",
            "world_size": 32,
        },
    )
    arguments = {
        "output_root": output,
        "allowed_prefix": allowed,
        "forbidden_output_root": allowed / "main",
        "expected_complete_sha256": complete_sha256,
        "code_commit": code_commit,
        "image_reference": image_reference,
        "image_digest": image_digest,
        "input_bindings": input_bindings,
    }
    return arguments, weights, shard_path


@pytest.mark.parametrize(
    "damage",
    [
        "delete_weights",
        "mutate_weights",
        "delete_state_shard",
        "mutate_state_shard",
    ],
)
def test_gate_binding_strongly_rechecks_large_checkpoint_files(tmp_path, damage):
    arguments, weights, shard_path = _build_minimal_sealed_gate(tmp_path)
    assert validate_n4_fullmodel_gate_binding(**arguments)["status"] == "PASS"

    if damage == "delete_weights":
        weights.unlink()
    elif damage == "mutate_weights":
        weights.chmod(0o600)
        weights.write_bytes(b"WEIGHT-DATA")
    elif damage == "delete_state_shard":
        shard_path.unlink()
    else:
        shard_path.chmod(0o600)
        shard_path.write_bytes(b"ZERO-state-shard")

    with pytest.raises((FileNotFoundError, RuntimeError)):
        validate_n4_fullmodel_gate_binding(**arguments)


@pytest.mark.parametrize("terminal_name", ["manifest.json", "COMPLETE"])
def test_gate_binding_rejects_outer_terminal_mutation(tmp_path, terminal_name):
    arguments, _, _ = _build_minimal_sealed_gate(tmp_path)
    assert validate_n4_fullmodel_gate_binding(**arguments)["status"] == "PASS"

    terminal = arguments["output_root"] / terminal_name
    terminal.chmod(0o600)
    terminal.write_bytes(terminal.read_bytes() + b" ")

    with pytest.raises((RuntimeError, ValueError)):
        validate_n4_fullmodel_gate_binding(**arguments)


def test_gate_binding_rejects_current_input_identity_drift(tmp_path):
    arguments, _, _ = _build_minimal_sealed_gate(tmp_path)
    drifted = dict(arguments["input_bindings"])
    drifted["stats"] = "f" * 64
    arguments["input_bindings"] = drifted

    with pytest.raises(RuntimeError, match="input bindings"):
        validate_n4_fullmodel_gate_binding(**arguments)
