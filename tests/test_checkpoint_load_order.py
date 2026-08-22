import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastwam.trainer import Wan22Trainer


class _WeightModel:
    def __init__(self):
        self.loads = []

    def load_checkpoint(self, path, optimizer=None):
        self.loads.append((str(path), optimizer))


def _trainer(resume: Path, model=None):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.resume = str(resume)
    trainer.run_initial_global_step = 0
    trainer.model = _WeightModel() if model is None else model
    trainer._weight_checkpoint_loaded_before_prepare = False
    trainer.formal_n4_fullmodel_gate = False
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


class _WarmStartModel:
    def __init__(self):
        self.loads = []

    def _load_checkpoint_with_role(self, path, **kwargs):
        self.loads.append((str(path), kwargs))


def _write_native_v2_source_checkpoint(
    path: Path,
    *,
    training_mode: str = "joint",
    trainable_scope: str = "dit",
    include_state_kind: bool = True,
    base_checkpoint=None,
) -> None:
    payload = {
        "format": "fastwam_multi_robot_v2",
        "training_mode": training_mode,
        "trainable_scope": trainable_scope,
        "base_checkpoint": base_checkpoint,
        "mot": {"tiny_weight": torch.ones(1)},
    }
    if include_state_kind:
        payload["state_kind"] = "full"
    torch.save(payload, path)


def _explicit_warm_start_trainer(checkpoint: Path):
    trainer = _trainer(checkpoint, model=_WarmStartModel())
    trainer.checkpoint_state_kind = "full"
    trainer.weights_only_warm_start_enabled = True
    trainer.weights_only_warm_start_expected_source_training_mode = "joint"
    trainer.weights_only_warm_start_expected_source_trainable_scope = "dit"
    trainer.weights_only_warm_start_expected_source_state_kind = "full"
    trainer.weights_only_warm_start_allow_legacy_full_source_metadata = False
    trainer.weights_only_warm_start_expected_legacy_source_base_checkpoint = None
    return trainer


def test_explicit_cross_treatment_warm_start_uses_strict_native_loader(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "joint-full.pt"
    _write_native_v2_source_checkpoint(checkpoint)
    trainer = _explicit_warm_start_trainer(checkpoint)
    original_torch_load = torch.load
    load_calls = []

    def _recording_torch_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _recording_torch_load)

    trainer._load_weight_checkpoint_before_prepare()

    assert trainer._weight_checkpoint_loaded_before_prepare is True
    assert len(load_calls) == 1
    assert Path(load_calls[0][0][0]) == checkpoint.resolve()
    assert load_calls[0][1] == {
        "map_location": "meta",
        "weights_only": True,
        "mmap": True,
    }
    assert trainer.model.loads == [
        (
            str(checkpoint.resolve()),
            {
                "optimizer": None,
                "load_role": "base_dependency",
                "active_paths": set(),
                "validate_trainable_scope": False,
                "allow_legacy_full_source_metadata": False,
                "expected_legacy_base_checkpoint": None,
            },
        )
    ]
    trainer._resume_training_state_after_prepare()
    assert len(trainer.model.loads) == 1


def test_explicit_warm_start_rejects_wrong_source_treatment_before_load(tmp_path):
    checkpoint = tmp_path / "action-full.pt"
    _write_native_v2_source_checkpoint(
        checkpoint,
        training_mode="action_only_cache",
        trainable_scope="action",
    )
    trainer = _explicit_warm_start_trainer(checkpoint)

    with pytest.raises(ValueError, match="source metadata mismatch"):
        trainer._load_weight_checkpoint_before_prepare()

    assert trainer.model.loads == []
    assert trainer._weight_checkpoint_loaded_before_prepare is False


def test_explicit_warm_start_accepts_only_pinned_legacy_full_metadata(tmp_path):
    legacy_base = "/cpfs/user/example/legacy-base.pt"
    checkpoint = tmp_path / "legacy-joint-full.pt"
    _write_native_v2_source_checkpoint(
        checkpoint,
        include_state_kind=False,
        base_checkpoint=legacy_base,
    )
    trainer = _explicit_warm_start_trainer(checkpoint)
    trainer.weights_only_warm_start_allow_legacy_full_source_metadata = True
    trainer.weights_only_warm_start_expected_legacy_source_base_checkpoint = (
        legacy_base
    )

    trainer._load_weight_checkpoint_before_prepare()

    assert trainer._weight_checkpoint_loaded_before_prepare is True
    assert trainer.model.loads == [
        (
            str(checkpoint.resolve()),
            {
                "optimizer": None,
                "load_role": "base_dependency",
                "active_paths": set(),
                "validate_trainable_scope": False,
                "allow_legacy_full_source_metadata": True,
                "expected_legacy_base_checkpoint": legacy_base,
            },
        )
    ]


def test_explicit_warm_start_rejects_legacy_metadata_without_opt_in(tmp_path):
    checkpoint = tmp_path / "legacy-without-opt-in.pt"
    _write_native_v2_source_checkpoint(
        checkpoint,
        include_state_kind=False,
        base_checkpoint="/cpfs/user/example/legacy-base.pt",
    )
    trainer = _explicit_warm_start_trainer(checkpoint)

    with pytest.raises(ValueError, match="source metadata mismatch"):
        trainer._load_weight_checkpoint_before_prepare()

    assert trainer.model.loads == []


def test_explicit_warm_start_rejects_wrong_legacy_base_pointer(tmp_path):
    checkpoint = tmp_path / "legacy-wrong-base.pt"
    _write_native_v2_source_checkpoint(
        checkpoint,
        include_state_kind=False,
        base_checkpoint="/cpfs/user/example/observed.pt",
    )
    trainer = _explicit_warm_start_trainer(checkpoint)
    trainer.weights_only_warm_start_allow_legacy_full_source_metadata = True
    trainer.weights_only_warm_start_expected_legacy_source_base_checkpoint = (
        "/cpfs/user/example/expected.pt"
    )

    with pytest.raises(ValueError, match="base_checkpoint mismatch"):
        trainer._load_weight_checkpoint_before_prepare()

    assert trainer.model.loads == []


def test_explicit_warm_start_rejects_explicit_null_legacy_state_kind(tmp_path):
    checkpoint = tmp_path / "legacy-explicit-null-state-kind.pt"
    _write_native_v2_source_checkpoint(
        checkpoint,
        base_checkpoint="/cpfs/user/example/legacy-base.pt",
    )
    payload = torch.load(checkpoint, weights_only=True)
    payload["state_kind"] = None
    torch.save(payload, checkpoint)
    trainer = _explicit_warm_start_trainer(checkpoint)
    trainer.weights_only_warm_start_allow_legacy_full_source_metadata = True
    trainer.weights_only_warm_start_expected_legacy_source_base_checkpoint = (
        "/cpfs/user/example/legacy-base.pt"
    )

    with pytest.raises(ValueError, match="state_kind key to be absent"):
        trainer._load_weight_checkpoint_before_prepare()

    assert trainer.model.loads == []


def test_explicit_warm_start_rejects_full_state_directory(tmp_path):
    state_dir = tmp_path / "step_001250"
    state_dir.mkdir()
    trainer = _explicit_warm_start_trainer(state_dir)

    with pytest.raises(ValueError, match="not a full-state resume directory"):
        trainer._load_weight_checkpoint_before_prepare()

    assert trainer.model.loads == []
    assert trainer._weight_checkpoint_loaded_before_prepare is False


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


def test_data_schedule_mismatch_fails_before_accelerator_mutation(tmp_path):
    state_dir = tmp_path / "step_000123"
    state_dir.mkdir()
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.allow_legacy_resume = False
    trainer.accelerator = _StateAccelerator()
    trainer._uses_agent_count_batch_sampler = True
    trainer.phase_balanced_fraction = 0.5
    trainer.train_sampler = SimpleNamespace(
        schedule_fingerprint=lambda epoch: f"schedule-epoch-{epoch}",
        agent_action_token_budget=128,
        gradient_accumulation_steps=1,
        num_processes=24,
        global_batches_per_epoch=240,
        optimizer_steps_per_epoch=10,
        phase_balanced_fraction=0.5,
        original_global_batches_per_epoch=120,
        phase_balanced_global_batches_per_epoch=120,
    )
    trainer._validate_training_state_contract = lambda payload, state_file: None
    trainer._validate_resumable_terminal_evidence = (
        lambda payload, state_file: ({}, [])
    )
    saved_schedule = trainer._data_schedule_contract(epoch=0)
    saved_schedule["phase_balanced_global_batches_per_epoch"] = 119
    (state_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 123,
                "epoch": 0,
                "batch_in_epoch": 0,
                "data_schedule": saved_schedule,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="before accelerator.load_state"):
        trainer.load_training_state(str(state_dir))

    assert trainer.accelerator.loads == []


def test_saved_trainer_metadata_contains_run_contract(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.global_step = 17
    trainer.epoch = 2
    trainer.batch_in_epoch = 9
    trainer._uses_agent_count_batch_sampler = False
    trainer._evaluation_records = [{"step": 10, "val_loss": 1.5}]
    trainer._last_step_metrics = {"step": 17, "loss": 0.5}
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
    assert payload["evaluation_records"] == [{"step": 10, "val_loss": 1.5}]
    assert payload["last_step_metrics"] == {"step": 17, "loss": 0.5}


def test_resume_at_max_steps_does_not_republish_existing_checkpoint():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer._set_dit_only_train_mode = lambda: None
    trainer._set_train_data_epoch = lambda epoch: None
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    trainer.model = object()
    trainer.max_steps = 5000
    trainer.global_step = 5000
    trainer.optimizer_steps_this_run = 0
    trainer.epoch = 3
    trainer.train_loader = []
    trainer.formal_n4_fullmodel_gate = False
    trainer.save_final_checkpoint_enabled = True
    trainer.save_checkpoint = lambda: pytest.fail(
        "resume at max_steps must not recreate an exclusive checkpoint"
    )

    trainer.train()


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


def test_stat_cmp_checkpoint_publication_uses_bytewise_readback_without_hash(
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
            Path(path).write_bytes(b"b4-stat-cmp-self-contained-weights")

    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    monkeypatch.setenv("FASTWAM_WEIGHT_STAGING_DIR", str(tmp_path / "staging"))
    model = _SavingModel()
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.accelerator = _StateAccelerator(model)
    trainer.weights_dir = str(weights_dir)
    trainer.checkpoint_state_kind = "full"
    trainer.global_step = 31
    trainer.provenance_mode = "stat_cmp"

    def unexpected_hash(*_args, **_kwargs):
        pytest.fail("stat_cmp publication must not compute a checkpoint hash")

    trainer._sha256_regular_file = unexpected_hash
    checkpoint = Path(trainer._save_weights_checkpoint("step_000031"))

    manifest = json.loads(
        (weights_dir / "step_000031.pt.manifest.json").read_text(encoding="utf-8")
    )
    complete = json.loads(
        (weights_dir / "step_000031.pt.COMPLETE").read_text(encoding="utf-8")
    )
    assert checkpoint.read_bytes() == b"b4-stat-cmp-self-contained-weights"
    assert manifest["schema_version"] == 2
    assert manifest["path"] == str(checkpoint.resolve())
    assert manifest["bytes"] == checkpoint.stat().st_size
    assert manifest["mtime_ns"] == checkpoint.stat().st_mtime_ns
    assert manifest["file_count"] == 1
    assert manifest["verification"] == "stat+bytewise-cmp"
    assert complete["schema_version"] == 2
    assert complete["checkpoint_bytes"] == checkpoint.stat().st_size
    assert complete["checkpoint_mtime_ns"] == checkpoint.stat().st_mtime_ns
    assert complete["file_count"] == 1
    assert complete["verification"] == "stat+bytewise-cmp"
    assert "sha256" not in json.dumps({"manifest": manifest, "complete": complete})


def test_stat_cmp_dataset_and_run_contracts_do_not_compute_hashes(
    tmp_path,
    monkeypatch,
):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    source = dataset_root / "n2" / "episode.h5"
    source.parent.mkdir()
    source.write_bytes(b"h5-payload")
    stats = dataset_root / "stats.json"
    stats.write_text('{"schema":"test"}\n', encoding="utf-8")

    class _Dataset:
        root_dir = dataset_root
        _stats_path = stats
        _stats_metadata = {"schema_name": "test-stats"}
        entries = [
            {"path": str(source), "source_path": "n2/episode.h5", "start_step": 0},
            {"path": str(source), "source_path": "n2/episode.h5", "start_step": 1},
        ]

        def __len__(self):
            return 2

    def unexpected_hash(*_args, **_kwargs):
        pytest.fail("stat_cmp run contracts must not compute provenance hashes")

    monkeypatch.setattr(
        Wan22Trainer,
        "_canonical_json_sha256",
        staticmethod(unexpected_hash),
    )
    monkeypatch.setattr(Wan22Trainer, "_sha256_file", staticmethod(unexpected_hash))

    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.provenance_mode = "stat_cmp"
    dataset_contract = trainer._dataset_contract(_Dataset())
    assert dataset_contract["source_inventory"] == [
        {
            "path": "n2/episode.h5",
            "bytes": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
        }
    ]
    assert dataset_contract["source_inventory_count"] == 1
    assert dataset_contract["source_inventory_total_bytes"] == source.stat().st_size
    assert dataset_contract["window_index_count"] == 2
    assert dataset_contract["window_index_counts_by_source"] == {"n2/episode.h5": 2}
    assert dataset_contract["normalization"]["bytes"] == stats.stat().st_size
    assert dataset_contract["normalization"]["mtime_ns"] == stats.stat().st_mtime_ns
    assert "sha256" not in json.dumps(dataset_contract)

    model = torch.nn.Linear(2, 2)
    model.training_mode = "action_only_cache"
    model._multi_robot_architecture_metadata = lambda: {
        "hub_enabled": True,
        "enable_gaussian": True,
    }
    trainer.model = model
    trainer.accelerator = SimpleNamespace(
        unwrap_model=lambda wrapped: wrapped,
        num_processes=24,
    )
    trainer.trainable_scope = "action"
    trainer.checkpoint_state_kind = "full"
    trainer._dataset_run_contract = {"train": dataset_contract, "val": dataset_contract}
    trainer.learning_rate = 1.0e-5
    trainer.weight_decay = 0.0
    trainer.max_steps = 2500
    trainer.run_initial_global_step = 0
    trainer.optimizer_steps_this_run = 2500
    trainer.scheduler_warmup_steps = 125
    trainer.batch_size = 1
    trainer.agent_action_token_budget = 128
    trainer.phase_balanced_fraction = 0.5
    trainer.gradient_accumulation_steps = 1
    trainer.mixed_precision = "bf16"
    trainer.max_grad_norm = 1.0
    trainer.seed = 42
    trainer.cfg = SimpleNamespace(lr_scheduler_type="cosine")
    trainer._resolved_config_contract = lambda: {
        "learning_rate": 1.0e-5,
        "max_steps": 2500,
    }
    trainer._git_commit = lambda: "a" * 40

    run_contract = trainer._training_state_contract()
    assert run_contract["contract_version"] == 2
    assert run_contract["provenance_mode"] == "stat_cmp"
    assert run_contract["resolved_config"] == {
        "learning_rate": 1.0e-5,
        "max_steps": 2500,
    }
    assert "trainable_parameters_sha256" not in run_contract
    assert "resolved_config_sha256" not in run_contract


def test_periodic_steps_after_start_uses_cumulative_schedule():
    assert Wan22Trainer._periodic_steps_after_start(5000, 50000, 5000) == list(
        range(10000, 50001, 5000)
    )
    assert Wan22Trainer._periodic_steps_after_start(1250, 2500, 0) == [1250, 2500]


def test_checkpoint_publication_wait_uses_regular_complete_file(tmp_path):
    marker = tmp_path / "step_000031.pt.COMPLETE"
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.checkpoint_io_timeout_seconds = 2

    def publish():
        time.sleep(0.05)
        marker.write_text("{}\n", encoding="utf-8")

    publisher = threading.Thread(target=publish)
    publisher.start()
    trainer._wait_for_published_regular_file(marker, label="test marker")
    publisher.join(timeout=1)
    assert not publisher.is_alive()

    marker.unlink()
    target = tmp_path / "target"
    target.write_text("payload", encoding="utf-8")
    marker.symlink_to(target)
    with pytest.raises(RuntimeError, match="must not be a symlink"):
        trainer._wait_for_published_regular_file(marker, label="test marker")


def test_non_main_checkpoint_waits_outside_collectives(tmp_path):
    events = []

    class _NonMainAccelerator:
        is_main_process = False
        device = "cpu"

        def reduce(self, tensor, *, reduction):
            assert reduction == "sum"
            events.append("checkpoint_target_reduce")
            return tensor

        def wait_for_everyone(self):
            events.append("barrier")

        def save_state(self, output_dir):
            events.append("save_state")
            Path(output_dir, "rank-1.bin").write_bytes(b"state")

    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.global_step = 31
    trainer.weights_dir = str(tmp_path / "weights")
    trainer.state_dir = str(tmp_path / "state")
    trainer.save_training_state_enabled = True
    trainer.seal_training_state = True
    trainer.accelerator = _NonMainAccelerator()
    trainer._wait_for_published_regular_file = (
        lambda path, *, label: events.append(f"poll:{Path(path).name}")
    )

    result = trainer.save_checkpoint()

    assert result == {
        "weights_path": None,
        "state_path": str(tmp_path / "state" / "step_000031"),
        "state_manifest": None,
    }
    assert events == [
        "checkpoint_target_reduce",
        "barrier",
        "poll:step_000031.pt.COMPLETE",
        "barrier",
        "save_state",
        "barrier",
        "poll:step_000031.state-tree.json",
        "barrier",
    ]


def test_checkpoint_target_preflight_rejects_stale_complete_collectively(tmp_path):
    class _CollectiveAccelerator:
        device = "cpu"

        def __init__(self, global_conflicts):
            self.global_conflicts = global_conflicts

        def reduce(self, tensor, *, reduction):
            assert reduction == "sum"
            return tensor.new_tensor([self.global_conflicts])

    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.weights_dir = str(weights_dir)
    trainer.state_dir = str(tmp_path / "state")
    trainer.save_training_state_enabled = True
    trainer.seal_training_state = True

    stale = weights_dir / "step_000031.pt.COMPLETE"
    stale.write_text("{}\n", encoding="utf-8")
    trainer.accelerator = _CollectiveAccelerator(global_conflicts=1)
    with pytest.raises(FileExistsError, match="pre-existing targets") as observed:
        trainer._assert_checkpoint_targets_absent(step_tag="step_000031")
    assert str(stale) in str(observed.value)

    stale.unlink()
    # A rank that does not yet see the shared stale file must still fail when
    # another rank reports it through the collective.
    trainer.accelerator = _CollectiveAccelerator(global_conflicts=1)
    with pytest.raises(FileExistsError, match="local_conflicts=none-local"):
        trainer._assert_checkpoint_targets_absent(step_tag="step_000031")


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
