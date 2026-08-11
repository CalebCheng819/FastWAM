import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastwam.nohash_artifacts import (
    copy_exclusive_and_compare,
    publish_exclusive_bytes,
    publish_rank_zero_payload,
    read_json,
    regular_file_metadata,
)
from fastwam.trainer import Wan22Trainer


def _forbid_digest_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("digest function was called in metadata_no_hash mode")

    monkeypatch.setattr(hashlib, "sha256", forbidden)
    monkeypatch.setattr(hashlib, "md5", forbidden)
    monkeypatch.setattr(hashlib, "sha1", forbidden)


def test_nohash_artifacts_publish_copy_and_barrier_without_digest(tmp_path, monkeypatch):
    _forbid_digest_calls(monkeypatch)
    payload = b"resolved: config\n"
    config = tmp_path / "config.yaml"
    metadata = publish_rank_zero_payload(
        config,
        payload,
        rank=0,
        world_size=1,
        timeout_seconds=1,
    )
    assert metadata == regular_file_metadata(config)
    assert config.read_bytes() == payload

    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint-payload" * 128)
    destination = tmp_path / "destination.pt"
    copied = copy_exclusive_and_compare(source, destination)
    assert copied["bytes"] == source.stat().st_size
    assert destination.read_bytes() == source.read_bytes()

    with pytest.raises(FileExistsError):
        publish_exclusive_bytes(config, b"replacement")
    assert config.read_bytes() == payload


class _FakeDataset:
    num_frames = 1
    action_horizon = 32
    required_agent_counts = (2,)
    required_tasks = ("PlaceFood",)
    video_size = (224, 320)

    def __init__(self, root: Path, stats: Path):
        self.root_dir = root
        self._stats_path = stats
        self._stats_metadata = {"schema": 1}
        self.entries = [
            {
                "path": str(root / "PlaceFood" / "trajectory_0.h5"),
                "source_path": "PlaceFood/trajectory_0.h5",
                "start": 0,
                "agent_count": 2,
            }
        ]

    def __len__(self):
        return 1


def test_dataset_contract_contains_direct_inventory_without_digest(tmp_path, monkeypatch):
    _forbid_digest_calls(monkeypatch)
    root = tmp_path / "dataset"
    trajectory_dir = root / "PlaceFood"
    trajectory_dir.mkdir(parents=True)
    trajectory = trajectory_dir / "trajectory_0.h5"
    trajectory.write_bytes(b"hdf5-placeholder")
    stats = root / "stats.json"
    stats.write_text("{}\n", encoding="utf-8")

    contract = Wan22Trainer._dataset_contract_metadata_no_hash(
        _FakeDataset(root, stats)
    )

    assert contract["integrity_mode"] == "metadata_no_hash"
    assert contract["source_inventory"][0]["path"] == "PlaceFood/trajectory_0.h5"
    assert contract["window_index"][0]["source_path"] == "PlaceFood/trajectory_0.h5"
    serialized = json.dumps(contract, sort_keys=True)
    assert "source_inventory_sha" not in serialized
    assert "window_index_sha" not in serialized


class _FakeSaveModel:
    def save_checkpoint(
        self,
        path,
        optimizer=None,
        step=None,
        checkpoint_state_kind=None,
        checkpoint_integrity_mode=None,
    ):
        assert checkpoint_integrity_mode == "metadata_no_hash"
        Path(path).write_bytes(b"native-model-checkpoint")


class _FakeAccelerator:
    is_main_process = True

    def __init__(self, model):
        self.model = model

    def unwrap_model(self, model):
        assert model is self.model
        return model


def test_weights_publication_uses_metadata_and_direct_bytes(tmp_path, monkeypatch):
    _forbid_digest_calls(monkeypatch)
    model = _FakeSaveModel()
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.accelerator = _FakeAccelerator(model)
    trainer.weights_dir = str(tmp_path / "weights")
    Path(trainer.weights_dir).mkdir()
    trainer.checkpoint_state_kind = "sparse_delta"
    trainer.artifact_integrity_mode = "metadata_no_hash"
    trainer.global_step = 1
    monkeypatch.setenv("FASTWAM_WEIGHT_STAGING_DIR", str(tmp_path / "staging"))

    checkpoint = trainer._save_weights_checkpoint("step_000001")

    assert Path(checkpoint).read_bytes() == b"native-model-checkpoint"
    complete, _ = read_json(f"{checkpoint}.COMPLETE")
    manifest, _ = read_json(f"{checkpoint}.manifest.json")
    assert complete["integrity_mode"] == "metadata_no_hash"
    assert manifest["integrity_mode"] == "metadata_no_hash"
    assert not any(
        token in json.dumps({"complete": complete, "manifest": manifest}).lower()
        for token in ("sha256", "checksum", "digest", "md5")
    )


class _FakeSchedule:
    agent_action_token_budget = 2048
    gradient_accumulation_steps = 4
    num_processes = 8
    global_batches_per_epoch = 16
    optimizer_steps_per_epoch = 1
    seed = 42

    def global_epoch_batches(self, epoch):
        return [[epoch, 1], [epoch, 2]]

    def schedule_fingerprint(self, epoch):
        raise AssertionError("legacy schedule fingerprint was called")


def test_schedule_contract_embeds_exact_batches_without_digest(monkeypatch):
    _forbid_digest_calls(monkeypatch)
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.artifact_integrity_mode = "metadata_no_hash"
    trainer.train_sampler = _FakeSchedule()

    contract = trainer._data_schedule_contract(3)

    assert contract["integrity_mode"] == "metadata_no_hash"
    assert contract["batches"] == [[3, 1], [3, 2]]
    assert "fingerprint" not in contract


def test_trainer_state_json_round_trip_without_digest(tmp_path, monkeypatch):
    _forbid_digest_calls(monkeypatch)
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.artifact_integrity_mode = "metadata_no_hash"
    trainer.global_step = 1
    trainer.epoch = 0
    trainer.batch_in_epoch = 4
    trainer._evaluation_records = []
    trainer._last_step_metrics = {"step": 1}
    trainer._uses_agent_count_batch_sampler = False
    trainer._training_state_contract = lambda: {
        "contract_version": 2,
        "integrity_mode": "metadata_no_hash",
    }
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    trainer._save_trainer_state(str(state_dir))

    payload, _ = read_json(state_dir / "trainer_state.json")
    assert payload["global_step"] == 1
    assert payload["batch_in_epoch"] == 4
    assert payload["run_contract"]["integrity_mode"] == "metadata_no_hash"


def test_base_metadata_resume_validation_is_fail_closed(tmp_path, monkeypatch):
    _forbid_digest_calls(monkeypatch)
    base = tmp_path / "base.pt"
    base.write_bytes(b"base-checkpoint")
    metadata = regular_file_metadata(base)
    descriptor = {
        "path": str(base.resolve()),
        "role": "base_dependency",
        "integrity_mode": "metadata_no_hash",
        "stat": {
            key: metadata[key]
            for key in ("bytes", "mtime_ns", "dev", "ino", "mode")
        },
    }
    model = SimpleNamespace(
        _loaded_base_checkpoint=None,
        _loaded_base_checkpoint_descriptor=None,
        _loaded_base_checkpoint_can_restore_sparse=False,
    )
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.artifact_integrity_mode = "metadata_no_hash"
    trainer.model = model
    trainer.accelerator = _FakeAccelerator(model)

    trainer._restore_base_checkpoint_metadata_no_hash(
        descriptor, state_file=tmp_path / "trainer_state.json"
    )
    assert model._loaded_base_checkpoint_descriptor == descriptor

    base.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="verification failed"):
        trainer._restore_base_checkpoint_metadata_no_hash(
            descriptor, state_file=tmp_path / "trainer_state.json"
        )
