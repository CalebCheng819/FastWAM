import os
import stat
from pathlib import Path

import pytest
import torch

import fastwam.models.wan22.fastwam_multi_robot as model_module
import fastwam.models.wan22.helpers.loader as loader_module
from fastwam.models.wan22.fastwam_multi_robot import FastWAMMultiRobot
from fastwam.trainer import Wan22Trainer


class _TinyExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.action_dim = 1
        self.state_dim = 1
        self.hidden_dim = 4
        self.ffn_dim = 8
        self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
        self.num_heads = 1
        self.attn_head_dim = 4
        self.text_dim = 4
        self.freq_dim = 2
        self.agent_encoding_mode = "geometry"
        self.agent_geometry_dim = 7
        self.agent_rope_dim = 2
        self.agent_phase_scale = 1.0
        self.hub_enabled = True
        self.hub_token_ratio = 1.0
        self.hub_position_scale = 1.0
        self.enable_gaussian = False

    @staticmethod
    def backbone_key_set(keys):
        return set(keys)


class _TinyMoT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mixtures = torch.nn.ModuleDict(
            {"video": _TinyExpert(), "action": _TinyExpert()}
        )


class _TrainerAccelerator:
    is_main_process = True

    def __init__(self, model):
        self.model = model

    def unwrap_model(self, model):
        assert model is self.model
        return model


def _checkpoint_model(*, trainable_scope="action", integrity_mode="metadata_no_hash"):
    model = FastWAMMultiRobot.__new__(FastWAMMultiRobot)
    torch.nn.Module.__init__(model)
    model.mot = _TinyMoT()
    model.video_expert = model.mot.mixtures["video"]
    model.action_expert = model.mot.mixtures["action"]
    model.training_mode = "action_only_cache"
    model._trainable_scope = trainable_scope
    model.torch_dtype = torch.float32
    model.checkpoint_integrity_mode = integrity_mode
    model._loaded_base_checkpoint = None
    model._loaded_base_checkpoint_sha256 = None
    model._loaded_base_checkpoint_descriptor = None
    model._loaded_base_checkpoint_can_restore_sparse = False
    model.requires_grad_(False)
    if trainable_scope == "dit":
        model.mot.requires_grad_(True)
    elif trainable_scope == "action":
        model.action_expert.requires_grad_(True)
    else:
        raise ValueError(trainable_scope)
    return model


def _forbid_sha256(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("metadata_no_hash must not call hashlib.sha256")

    monkeypatch.setattr(model_module.hashlib, "sha256", forbidden)


def test_metadata_no_hash_initialization_sparse_save_and_fresh_load(
    tmp_path,
    monkeypatch,
):
    source = _checkpoint_model(trainable_scope="dit")
    base_path = tmp_path / "base.pt"
    source.save_checkpoint(base_path, checkpoint_state_kind="full", step=0)

    _forbid_sha256(monkeypatch)
    trained = _checkpoint_model(trainable_scope="action")
    trained.load_initialization_checkpoint(
        base_path,
        checkpoint_integrity_mode="metadata_no_hash",
    )
    descriptor = trained._loaded_base_checkpoint_descriptor
    assert descriptor == {
        "path": str(base_path),
        "role": "base_dependency",
        "integrity_mode": "metadata_no_hash",
        "stat": {
            "bytes": base_path.stat().st_size,
            "mtime_ns": base_path.stat().st_mtime_ns,
            "dev": base_path.stat().st_dev,
            "ino": base_path.stat().st_ino,
            "mode": base_path.stat().st_mode,
        },
    }
    assert stat.S_ISREG(descriptor["stat"]["mode"])
    assert trained._loaded_base_checkpoint_sha256 is None

    with torch.no_grad():
        trained.action_expert.weight.add_(7.0)
    expected_action = trained.action_expert.weight.detach().clone()
    sparse_path = tmp_path / "sparse.pt"
    trained.save_checkpoint(
        sparse_path,
        checkpoint_state_kind="sparse_delta",
        step=1,
    )
    sparse_payload = torch.load(sparse_path, map_location="cpu", weights_only=True)
    assert sparse_payload["base_checkpoint"] == descriptor
    assert "sha256" not in sparse_payload["base_checkpoint"]

    restored = _checkpoint_model(trainable_scope="action")
    restored.load_checkpoint(sparse_path)
    assert torch.equal(restored.action_expert.weight, expected_action)
    assert restored._loaded_base_checkpoint_descriptor == descriptor
    assert restored._loaded_base_checkpoint_sha256 is None


def test_checkpoint_publication_integrity_mode_is_explicit_and_must_match(tmp_path):
    model = _checkpoint_model(trainable_scope="dit")
    checkpoint = tmp_path / "explicit.pt"
    model.save_checkpoint(
        checkpoint,
        checkpoint_state_kind="full",
        checkpoint_integrity_mode="metadata_no_hash",
    )
    assert checkpoint.is_file()

    with pytest.raises(ValueError, match="publication integrity mode must match"):
        model.save_checkpoint(
            tmp_path / "mismatch.pt",
            checkpoint_state_kind="full",
            checkpoint_integrity_mode="sha256",
        )


def test_trainer_metadata_no_hash_publication_uses_real_multi_robot_model(
    tmp_path,
    monkeypatch,
):
    base_model = _checkpoint_model(trainable_scope="dit")
    base_path = tmp_path / "base.pt"
    base_model.save_checkpoint(base_path, checkpoint_state_kind="full", step=0)

    model = _checkpoint_model(trainable_scope="action")
    model.load_initialization_checkpoint(
        base_path,
        checkpoint_integrity_mode="metadata_no_hash",
    )
    with torch.no_grad():
        model.action_expert.weight.add_(3.0)
    expected_action = model.action_expert.weight.detach().clone()

    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.accelerator = _TrainerAccelerator(model)
    trainer.weights_dir = str(tmp_path / "weights")
    Path(trainer.weights_dir).mkdir()
    trainer.checkpoint_state_kind = "sparse_delta"
    trainer.artifact_integrity_mode = "metadata_no_hash"
    trainer.global_step = 1
    monkeypatch.setenv("FASTWAM_WEIGHT_STAGING_DIR", str(tmp_path / "staging"))
    _forbid_sha256(monkeypatch)

    checkpoint = Path(trainer._save_weights_checkpoint("step_000001"))
    assert checkpoint.is_file()
    assert checkpoint.with_name(f"{checkpoint.name}.manifest.json").is_file()
    assert checkpoint.with_name(f"{checkpoint.name}.COMPLETE").is_file()

    restored = _checkpoint_model(trainable_scope="action")
    restored.load_checkpoint(checkpoint)
    assert torch.equal(restored.action_expert.weight, expected_action)


def test_metadata_no_hash_rejects_changed_base_stat_before_sparse_restore(
    tmp_path,
    monkeypatch,
):
    source = _checkpoint_model(trainable_scope="dit")
    base_path = tmp_path / "base.pt"
    source.save_checkpoint(base_path, checkpoint_state_kind="full")
    trained = _checkpoint_model(trainable_scope="action")
    trained.load_initialization_checkpoint(
        base_path,
        checkpoint_integrity_mode="metadata_no_hash",
    )
    sparse_path = tmp_path / "sparse.pt"
    trained.save_checkpoint(sparse_path, checkpoint_state_kind="sparse_delta")

    _forbid_sha256(monkeypatch)
    before = base_path.stat()
    os.utime(
        base_path,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000),
    )
    with pytest.raises(ValueError, match="metadata mismatch before deserialization"):
        _checkpoint_model(trainable_scope="action").load_checkpoint(sparse_path)


def test_metadata_no_hash_is_explicit_and_rejects_sha_expectation(
    tmp_path,
    monkeypatch,
):
    source = _checkpoint_model(trainable_scope="dit")
    checkpoint = tmp_path / "full.pt"
    source.save_checkpoint(checkpoint, checkpoint_state_kind="full")
    target = _checkpoint_model(trainable_scope="action")
    _forbid_sha256(monkeypatch)

    with pytest.raises(ValueError, match="must match the model"):
        target.load_initialization_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="expected_sha256 is incompatible"):
        target.load_initialization_checkpoint(
            checkpoint,
            expected_sha256="not-consumed",
            checkpoint_integrity_mode="metadata_no_hash",
        )


def test_registered_component_metadata_mode_uses_explicit_unique_name_and_strict_load(
    monkeypatch,
):
    calls = []

    class TinyRegisteredModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

        def load_state_dict(self, state_dict, strict=True, assign=False):
            calls.append(strict)
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

    monkeypatch.setattr(
        loader_module,
        "WAN22_MODEL_REGISTRY",
        [
            {
                "model_hash": "historical-only-value",
                "model_name": "tiny_explicit_model",
                "model_class": TinyRegisteredModel,
            }
        ],
    )

    def forbidden_model_hash(*args, **kwargs):
        raise AssertionError("metadata_no_hash must not call hash_model_file")

    monkeypatch.setattr(loader_module, "hash_model_file", forbidden_model_hash)
    monkeypatch.setattr(
        loader_module,
        "load_state_dict",
        lambda *args, **kwargs: {"weight": torch.ones(2)},
    )
    loaded = loader_module._load_registered_model(
        "/explicit/model/path",
        "tiny_explicit_model",
        torch_dtype=torch.float32,
        device="cpu",
        checkpoint_integrity_mode="metadata_no_hash",
    )
    assert calls == [True]
    assert torch.equal(loaded.weight, torch.ones(2))


def test_registered_component_metadata_mode_rejects_ambiguous_name(monkeypatch):
    entry = {
        "model_hash": "historical-only-value",
        "model_name": "ambiguous",
        "model_class": _TinyExpert,
    }
    monkeypatch.setattr(loader_module, "WAN22_MODEL_REGISTRY", [entry, dict(entry)])

    def forbidden_model_hash(*args, **kwargs):
        raise AssertionError("metadata_no_hash must not call hash_model_file")

    monkeypatch.setattr(loader_module, "hash_model_file", forbidden_model_hash)
    with pytest.raises(ValueError, match="exactly one registry entry"):
        loader_module._load_registered_model(
            "/explicit/model/path",
            "ambiguous",
            torch_dtype=torch.float32,
            device="cpu",
            checkpoint_integrity_mode="metadata_no_hash",
        )
