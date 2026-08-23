from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import pytest

from experiments.robofactory.eval_robofactory_multi_robot import (
    _anchor_robofactory_imports,
    _load_panel,
    _required_fastwam_arguments,
)


def test_robofactory_import_anchor_rejects_shadowed_utils(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "RoboFactory"
    (root / "utils" / "scenes").mkdir(parents=True)
    (root / "tasks").mkdir()
    shadow = tmp_path / "site-packages" / "utils"
    shadow.mkdir(parents=True)
    module = types.ModuleType("utils")
    module.__path__ = [str(shadow)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils", module)
    previous = Path.cwd()
    try:
        with pytest.raises(RuntimeError, match="does not include RoboFactory path"):
            _anchor_robofactory_imports(root)
    finally:
        os.chdir(previous)


def test_panel_loader_supports_no_hash_metadata_binding(tmp_path: Path):
    path = tmp_path / "panel.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "fastwam-robofactory-heldout-panel-v1",
                "episodes": [],
            }
        ),
        encoding="utf-8",
    )

    payload, identity = _load_panel(
        path,
        None,
        expected_size_bytes=path.stat().st_size,
        integrity_mode="metadata_no_hash",
    )

    assert payload["episodes"] == []
    assert identity == {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": None,
        "integrity_mode": "metadata_no_hash",
    }


def _arguments(*, gaussian_conditioning: bool) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=Path("checkpoint.pt"),
        stats=Path("stats.json"),
        context_cache_dir=Path("text"),
        model_cache_root=Path("model-cache"),
        policy_lightning_repo=Path("teacher"),
        noposplat_checkpoint=Path("teacher.ckpt"),
        noposplat_checkpoint_size_bytes=123,
        integrity_mode="metadata_no_hash",
        gaussian_conditioning=gaussian_conditioning,
    )


def test_gau0_mode_does_not_require_teacher_inputs():
    arguments = _arguments(gaussian_conditioning=False)
    arguments.policy_lightning_repo = None
    arguments.noposplat_checkpoint = None

    _required_fastwam_arguments(arguments)


def test_gau1_mode_still_requires_teacher_inputs():
    arguments = _arguments(gaussian_conditioning=True)
    arguments.policy_lightning_repo = None

    with pytest.raises(ValueError, match="policy_lightning_repo"):
        _required_fastwam_arguments(arguments)


def test_gau1_metadata_mode_requires_teacher_checkpoint_size():
    arguments = _arguments(gaussian_conditioning=True)
    arguments.noposplat_checkpoint_size_bytes = None

    with pytest.raises(ValueError, match="noposplat_checkpoint_size_bytes"):
        _required_fastwam_arguments(arguments)
