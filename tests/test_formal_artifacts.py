from concurrent.futures import ThreadPoolExecutor
import hashlib
from types import SimpleNamespace

import pytest
import torch

from fastwam.formal_artifacts import (
    canonical_json_bytes,
    canonical_json_sha256,
    checkpoint_seal_descriptor,
    publish_exclusive_bytes,
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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _publish_json(path, payload) -> str:
    encoded = canonical_json_bytes(payload)
    publish_exclusive_bytes(path, encoded)
    return _sha256_bytes(encoded)


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
        weights.write_bytes(b"WEIGHT-DATA")
    elif damage == "delete_state_shard":
        shard_path.unlink()
    else:
        shard_path.write_bytes(b"ZERO-state-shard")

    with pytest.raises((FileNotFoundError, RuntimeError)):
        validate_n4_fullmodel_gate_binding(**arguments)


@pytest.mark.parametrize("terminal_name", ["manifest.json", "COMPLETE"])
def test_gate_binding_rejects_outer_terminal_mutation(tmp_path, terminal_name):
    arguments, _, _ = _build_minimal_sealed_gate(tmp_path)
    assert validate_n4_fullmodel_gate_binding(**arguments)["status"] == "PASS"

    terminal = arguments["output_root"] / terminal_name
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
