import ast
import importlib.util
import json
import os
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_NOHASH_ARTIFACTS_PATH = _REPO_ROOT / "src/fastwam/nohash_artifacts.py"
_TRAINER_PATH = _REPO_ROOT / "src/fastwam/trainer.py"


def _load_nohash_artifacts_module():
    spec = importlib.util.spec_from_file_location(
        "_fastwam_nohash_artifacts_for_recovery_test",
        _NOHASH_ARTIFACTS_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {_NOHASH_ARTIFACTS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_trainer_methods(nohash_artifacts):
    """Load the exact target methods without importing trainer's GPU stack."""

    source = _TRAINER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_TRAINER_PATH))
    trainer_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Wan22Trainer"
        ),
        None,
    )
    if trainer_class is None:
        raise RuntimeError(f"Wan22Trainer is missing from {_TRAINER_PATH}")

    method_names = (
        "_validate_recovery_gate_stop_after_checkpoint_step",
        "_should_pause_after_recovery_gate_checkpoint",
        "_recovery_load_receipt_target",
        "_publish_recovery_load_receipt",
    )
    methods = {
        node.name: node
        for node in trainer_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    }
    missing = set(method_names) - methods.keys()
    if missing:
        raise RuntimeError(f"Trainer methods are missing: {sorted(missing)}")

    source_lines = source.splitlines(keepends=True)
    method_blocks = []
    for name in method_names:
        method = methods[name]
        first_line = min(
            [method.lineno, *(decorator.lineno for decorator in method.decorator_list)]
        )
        method_blocks.append(
            "".join(source_lines[first_line - 1 : method.end_lineno])
        )
    selected_source = "class Wan22Trainer:\n" + "\n".join(method_blocks)
    namespace = {
        "os": os,
        "Path": Path,
        "nohash_publish_exclusive_json": nohash_artifacts.publish_exclusive_json,
        "nohash_read_json": nohash_artifacts.read_json,
    }
    exec(compile(selected_source, str(_TRAINER_PATH), "exec"), namespace)
    return namespace["Wan22Trainer"]


_nohash_artifacts = _load_nohash_artifacts_module()
nohash_regular_file_metadata = _nohash_artifacts.regular_file_metadata
Wan22Trainer = _load_trainer_methods(_nohash_artifacts)


def _validate(value, *, mode="metadata_no_hash", max_steps=2, save_every=1):
    return Wan22Trainer._validate_recovery_gate_stop_after_checkpoint_step(
        value,
        artifact_integrity_mode=mode,
        max_steps=max_steps,
        save_every=save_every,
    )


def test_recovery_gate_accepts_exact_intermediate_checkpoint_step():
    assert _validate(1) == 1
    assert _validate(None, mode="sha256") is None


@pytest.mark.parametrize(
    ("value", "kwargs", "message"),
    [
        (True, {}, "positive integer"),
        ("1", {}, "positive integer"),
        (0, {}, "must be positive"),
        (1, {"mode": "sha256"}, "restricted to"),
        (1, {"max_steps": None}, "explicit max_steps"),
        (2, {}, "smaller than max_steps"),
        (1, {"save_every": 0}, "save_every>0"),
        (3, {"max_steps": 5, "save_every": 2}, "checkpoint step"),
    ],
)
def test_recovery_gate_validation_is_fail_closed(value, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _validate(value, **kwargs)


def test_recovery_gate_pauses_once_only_after_the_completed_step_one_checkpoint():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.recovery_gate_stop_after_checkpoint_step = 1

    trainer.global_step = 1
    assert trainer._should_pause_after_recovery_gate_checkpoint(
        checkpoint_saved_this_step=True
    )
    assert not trainer._should_pause_after_recovery_gate_checkpoint(
        checkpoint_saved_this_step=False
    )

    # A fresh process restored at step 1 advances to step 2 before this decision.
    trainer.global_step = 2
    assert not trainer._should_pause_after_recovery_gate_checkpoint(
        checkpoint_saved_this_step=True
    )


def test_nohash_gate_profile_enables_the_step_one_pause():
    config = Path(
        "configs/task/"
        "robofactory_multi_robot_ft_n2_placefood_vg0_hub1_gau1_224_3e-5_nohash_gate.yaml"
    ).read_text(encoding="utf-8")

    assert "recovery_gate_stop_after_checkpoint_step: 1" in config
    assert "checkpoint_state_kind: full" in config


def _receipt_trainer(output_dir, *, mode="metadata_no_hash"):
    output_dir.mkdir()
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.artifact_integrity_mode = mode
    trainer.output_dir = str(output_dir)
    return trainer


def test_recovery_load_receipt_target_is_disabled_without_env(monkeypatch, tmp_path):
    trainer = _receipt_trainer(tmp_path / "output")
    monkeypatch.delenv("FASTWAM_RECOVERY_LOAD_RECEIPT", raising=False)

    assert trainer._recovery_load_receipt_target() is None


def test_recovery_load_receipt_target_rejects_other_integrity_modes(
    monkeypatch, tmp_path
):
    output = tmp_path / "output"
    trainer = _receipt_trainer(output, mode="sha256")
    monkeypatch.setenv(
        "FASTWAM_RECOVERY_LOAD_RECEIPT",
        str(output / "recovery_load_receipt.json"),
    )

    with pytest.raises(RuntimeError, match="restricted to metadata_no_hash"):
        trainer._recovery_load_receipt_target()


def test_recovery_load_receipt_target_rejects_relative_path(monkeypatch, tmp_path):
    trainer = _receipt_trainer(tmp_path / "output")
    monkeypatch.setenv(
        "FASTWAM_RECOVERY_LOAD_RECEIPT", "recovery_load_receipt.json"
    )

    with pytest.raises(ValueError, match="must be an absolute path"):
        trainer._recovery_load_receipt_target()


def test_recovery_load_receipt_target_rejects_path_outside_output(
    monkeypatch, tmp_path
):
    trainer = _receipt_trainer(tmp_path / "output")
    outside = tmp_path / "outside" / "recovery_load_receipt.json"
    monkeypatch.setenv("FASTWAM_RECOVERY_LOAD_RECEIPT", str(outside))

    with pytest.raises(ValueError, match="must be the direct output receipt"):
        trainer._recovery_load_receipt_target()


def test_recovery_load_receipt_target_rejects_preexisting_target(
    monkeypatch, tmp_path
):
    output = tmp_path / "output"
    trainer = _receipt_trainer(output)
    target = output / "recovery_load_receipt.json"
    target.write_text("pre-existing\n", encoding="utf-8")
    monkeypatch.setenv("FASTWAM_RECOVERY_LOAD_RECEIPT", str(target))

    with pytest.raises(FileExistsError, match="pre-existing recovery load receipt"):
        trainer._recovery_load_receipt_target()


def test_recovery_load_receipt_target_accepts_absolute_direct_target(
    monkeypatch, tmp_path
):
    output = tmp_path / "output"
    trainer = _receipt_trainer(output)
    target = output / "recovery_load_receipt.json"
    monkeypatch.setenv("FASTWAM_RECOVERY_LOAD_RECEIPT", str(target))

    assert trainer._recovery_load_receipt_target() == target


class _FakeMainAccelerator:
    is_main_process = True
    process_index = 0
    num_processes = 8

    def __init__(self):
        self.wait_count = 0

    def wait_for_everyone(self):
        self.wait_count += 1


def test_publish_recovery_load_receipt_has_exact_local_schema(monkeypatch, tmp_path):
    output = tmp_path / "output"
    trainer = _receipt_trainer(output)
    trainer.global_step = 1
    trainer.epoch = 2
    trainer.batch_in_epoch = 3
    trainer.accelerator = _FakeMainAccelerator()

    source_state_dir = tmp_path / "state" / "step_000001"
    source_state_dir.mkdir(parents=True)
    source_trainer_state = source_state_dir / "trainer_state.json"
    source_trainer_state.write_text('{"global_step": 1}\n', encoding="utf-8")
    source_metadata = nohash_regular_file_metadata(source_trainer_state)

    configured_target = output / "recovery_load_receipt.json"
    monkeypatch.setenv(
        "FASTWAM_RECOVERY_LOAD_RECEIPT", str(configured_target)
    )
    target = trainer._recovery_load_receipt_target()
    trainer._publish_recovery_load_receipt(
        target=target,
        source_state_dir=source_state_dir,
        source_trainer_state_file=source_metadata,
    )

    receipt = json.loads(target.read_text(encoding="utf-8"))
    assert receipt == {
        "schema_name": "fastwam-recovery-load-receipt",
        "schema_version": 1,
        "integrity_mode": "metadata_no_hash",
        "accelerator_load_state_returned": True,
        "source_state_dir": str(source_state_dir.resolve(strict=True)),
        "source_trainer_state_file": source_metadata,
        "output_dir": str(output.resolve(strict=True)),
        "restored_global_step": 1,
        "restored_epoch": 2,
        "restored_batch_in_epoch": 3,
        "world_size": 8,
    }
    assert trainer.accelerator.wait_count == 1
