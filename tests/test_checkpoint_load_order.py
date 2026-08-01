import hashlib
import json
from pathlib import Path

import pytest

from fastwam.trainer import Wan22Trainer


class _WeightModel:
    def __init__(self):
        self.loads = []

    def load_checkpoint(self, path, optimizer=None):
        self.loads.append((str(path), optimizer))


def _trainer(resume: Path, model=None):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.resume = str(resume)
    trainer.model = _WeightModel() if model is None else model
    trainer._weight_checkpoint_loaded_before_prepare = False
    return trainer


def test_weight_checkpoint_is_loaded_once_before_prepare(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    trainer = _trainer(checkpoint)

    trainer._load_weight_checkpoint_before_prepare()

    assert trainer.model.loads == [(str(checkpoint), None)]
    assert trainer._weight_checkpoint_loaded_before_prepare is True

    # This represents the post-Accelerator.prepare phase.  It must never load
    # the module a second time after ZeRO has already created FP32 masters.
    trainer._resume_training_state_after_prepare()
    assert trainer.model.loads == [(str(checkpoint), None)]


def test_full_state_directory_is_deferred_until_after_prepare(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    trainer = _trainer(state_dir)
    restored = []
    trainer.load_training_state = restored.append

    trainer._load_weight_checkpoint_before_prepare()

    assert trainer.model.loads == []
    assert trainer._weight_checkpoint_loaded_before_prepare is False

    trainer._resume_training_state_after_prepare()
    assert restored == [str(state_dir)]


def test_post_prepare_file_resume_fails_if_preload_was_skipped(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    trainer = _trainer(checkpoint)

    with pytest.raises(RuntimeError, match="without being loaded before optimizer"):
        trainer._resume_training_state_after_prepare()


class _StateAccelerator:
    def __init__(self, model=None):
        self.loads = []
        self.is_main_process = True
        self.model = model

    def load_state(self, input_dir):
        self.loads.append(str(input_dir))

    def unwrap_model(self, model):
        assert model is self.model
        return model


def test_full_state_contract_mismatch_fails_before_accelerator_mutation(tmp_path):
    state_dir = tmp_path / "step_000123"
    state_dir.mkdir()
    state_file = state_dir / "trainer_state.json"
    state_file.write_text(
        json.dumps(
            {
                "global_step": 123,
                "epoch": 0,
                "batch_in_epoch": 0,
                "run_contract": {"contract_version": 1, "treatment": "hub0"},
            }
        ),
        encoding="utf-8",
    )
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.allow_legacy_resume = False
    trainer.accelerator = _StateAccelerator()
    trainer._training_state_contract = lambda: {
        "contract_version": 1,
        "treatment": "hub1",
    }

    with pytest.raises(RuntimeError, match="run contract mismatch"):
        trainer.load_training_state(str(state_dir))

    assert trainer.accelerator.loads == []


def test_missing_full_state_contract_fails_before_accelerator_mutation(tmp_path):
    state_dir = tmp_path / "step_000123"
    state_dir.mkdir()
    (state_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": 123, "epoch": 0, "batch_in_epoch": 0}),
        encoding="utf-8",
    )
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.allow_legacy_resume = False
    trainer.accelerator = _StateAccelerator()

    with pytest.raises(RuntimeError, match="lacks the required run_contract"):
        trainer.load_training_state(str(state_dir))

    assert trainer.accelerator.loads == []


def test_saved_trainer_metadata_contains_run_contract(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.global_step = 17
    trainer.epoch = 2
    trainer.batch_in_epoch = 9
    trainer._uses_agent_count_batch_sampler = False
    trainer._training_state_contract = lambda: {
        "contract_version": 1,
        "treatment": {"video_gen": True, "hub": True, "gaussian": True},
    }

    trainer._save_trainer_state(str(tmp_path))

    payload = json.loads((tmp_path / "trainer_state.json").read_text(encoding="utf-8"))
    assert payload["run_contract"]["contract_version"] == 1
    assert payload["run_contract"]["treatment"] == {
        "video_gen": True,
        "hub": True,
        "gaussian": True,
    }


def test_weights_checkpoint_is_strongly_read_back_and_complete_last(
    tmp_path,
    monkeypatch,
):
    class _SavingModel:
        def save_checkpoint(
            self,
            path,
            optimizer=None,
            step=None,
            checkpoint_state_kind=None,
        ):
            assert optimizer is None
            assert step == 31
            assert checkpoint_state_kind == "full"
            Path(path).write_bytes(b"self-contained-weight-checkpoint")

    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    staging_dir = tmp_path / "staging"
    monkeypatch.setenv("FASTWAM_WEIGHT_STAGING_DIR", str(staging_dir))
    model = _SavingModel()
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.accelerator = _StateAccelerator(model)
    trainer.weights_dir = str(weights_dir)
    trainer.checkpoint_state_kind = "full"
    trainer.global_step = 31

    checkpoint = Path(trainer._save_weights_checkpoint("step_000031"))

    manifest_path = weights_dir / "step_000031.pt.manifest.json"
    complete_path = weights_dir / "step_000031.pt.COMPLETE"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    expected_checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert manifest == {
        "schema_name": "fastwam-weights-checkpoint",
        "schema_version": 1,
        "filename": checkpoint.name,
        "bytes": checkpoint.stat().st_size,
        "sha256": expected_checkpoint_sha,
        "global_step": 31,
        "checkpoint_state_kind": "full",
    }
    assert complete["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert complete["checkpoint_sha256"] == expected_checkpoint_sha
    assert list(staging_dir.iterdir()) == []

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        trainer._save_weights_checkpoint("step_000031")


def test_full_state_resume_restores_saved_base_provenance(tmp_path):
    base = tmp_path / "official-base.pt"
    base.write_bytes(b"immutable-official-base")
    descriptor = {
        "path": str(base.resolve()),
        "sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "role": "base_dependency",
    }

    class _Model:
        _loaded_base_checkpoint = None
        _loaded_base_checkpoint_sha256 = None
        _loaded_base_checkpoint_descriptor = None
        _loaded_base_checkpoint_can_restore_sparse = False

    model = _Model()
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.allow_legacy_resume = False
    trainer.accelerator = _StateAccelerator(model)
    trainer._training_state_contract = lambda: {
        "contract_version": 1,
        "base_checkpoint": None,
        "treatment": "same",
    }
    payload = {
        "run_contract": {
            "contract_version": 1,
            "base_checkpoint": descriptor,
            "treatment": "same",
        }
    }

    trainer._validate_training_state_contract(
        payload,
        state_file=tmp_path / "trainer_state.json",
    )

    assert model._loaded_base_checkpoint == str(base.resolve())
    assert model._loaded_base_checkpoint_sha256 == descriptor["sha256"]
    assert model._loaded_base_checkpoint_descriptor == descriptor
    assert model._loaded_base_checkpoint_can_restore_sparse is True
