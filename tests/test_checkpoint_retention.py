from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastwam.trainer import Wan22Trainer


def _trainer(root: Path, *, keep_last: int) -> Wan22Trainer:
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.state_dir = str(root / "state")
    trainer.weights_dir = str(root / "weights")
    trainer.checkpoint_keep_last = keep_last
    trainer.seal_training_state = False
    Path(trainer.state_dir).mkdir(parents=True)
    Path(trainer.weights_dir).mkdir(parents=True)
    return trainer


def _complete_tuple(trainer: Wan22Trainer, step: int) -> dict[str, Path]:
    step_tag = f"step_{step:06d}"
    state = Path(trainer.state_dir) / step_tag
    nested = state / "optimizer"
    nested.mkdir(parents=True)
    (nested / "state.bin").write_bytes(f"optimizer-{step}".encode())
    (state / "trainer_state.json").write_text(
        json.dumps({"global_step": step}),
        encoding="utf-8",
    )

    weights = Path(trainer.weights_dir) / f"{step_tag}.pt"
    manifest = weights.with_name(f"{weights.name}.manifest.json")
    complete = weights.with_name(f"{weights.name}.COMPLETE")
    weights.write_bytes(f"weights-{step}".encode())
    manifest.write_text(json.dumps({"global_step": step}), encoding="utf-8")
    complete.write_text("complete\n", encoding="utf-8")
    return {
        "state": state,
        "weights": weights,
        "manifest": manifest,
        "complete": complete,
    }


def test_retention_keeps_only_newest_complete_resumable_tuples(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, keep_last=2)
    tuples = {step: _complete_tuple(trainer, step) for step in (1000, 2000, 3000)}

    assert trainer._prune_completed_checkpoints() == [1000]

    assert all(not path.exists() for path in tuples[1000].values())
    for step in (2000, 3000):
        assert all(path.exists() for path in tuples[step].values())


def test_retention_ignores_incomplete_state_directories(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, keep_last=1)
    old_tuple = _complete_tuple(trainer, 1000)
    incomplete = Path(trainer.state_dir) / "step_002000"
    incomplete.mkdir()
    (incomplete / "partial.bin").write_bytes(b"partial")
    new_tuple = _complete_tuple(trainer, 3000)

    assert trainer._prune_completed_checkpoints() == [1000]

    assert all(not path.exists() for path in old_tuple.values())
    assert incomplete.is_dir()
    assert (incomplete / "partial.bin").is_file()
    assert all(path.exists() for path in new_tuple.values())


def test_retention_fails_before_delete_on_symlink_in_victim(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, keep_last=1)
    old_tuple = _complete_tuple(trainer, 1000)
    new_tuple = _complete_tuple(trainer, 2000)
    (old_tuple["state"] / "unsafe-link").symlink_to(
        old_tuple["state"] / "trainer_state.json"
    )

    with pytest.raises(RuntimeError, match="refuses symlink"):
        trainer._prune_completed_checkpoints()

    assert all(path.exists() for path in old_tuple.values())
    assert all(path.exists() for path in new_tuple.values())


def test_retention_fails_before_delete_on_hardlink_in_victim(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, keep_last=1)
    old_tuple = _complete_tuple(trainer, 1000)
    new_tuple = _complete_tuple(trainer, 2000)
    source = old_tuple["state"] / "trainer_state.json"
    (old_tuple["state"] / "trainer_state-hardlink.json").hardlink_to(source)

    with pytest.raises(RuntimeError, match="multiply-linked regular file"):
        trainer._prune_completed_checkpoints()

    assert all(path.exists() for path in old_tuple.values())
    assert all(path.exists() for path in new_tuple.values())
