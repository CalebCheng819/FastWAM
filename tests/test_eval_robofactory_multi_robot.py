from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import experiments.robofactory.eval_robofactory_multi_robot as eval_module
from experiments.robofactory.eval_robofactory_multi_robot import (
    _anchor_robofactory_imports,
    _action_space_contract,
    _bootstrap_sapien_native_runtime,
    _load_panel,
    _required_fastwam_arguments,
    action_bound_records,
    aggregate_bound_summaries,
    aggregate_physical_summaries,
    apply_oracle_intervention,
    summarize_bound_records,
)


def test_sapien_native_bootstrap_is_ordered_once_and_retained(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, object]] = []

    class FakeSapien(types.ModuleType):
        def Device(self, name: str):  # noqa: N802 - matches SAPIEN API
            device = object()
            calls.append(("device", name))
            return device

    fake_sapien = FakeSapien("sapien")

    def render_system(device: object):
        renderer = object()
        calls.append(("render_system", device))
        return renderer

    fake_sapien.render = types.SimpleNamespace(RenderSystem=render_system)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sapien", fake_sapien)
    monkeypatch.setattr(eval_module, "_SAPIEN_NATIVE_BOOTSTRAP_RESOURCES", None)

    first = _bootstrap_sapien_native_runtime()
    retained = eval_module._SAPIEN_NATIVE_BOOTSTRAP_RESOURCES
    second = _bootstrap_sapien_native_runtime()

    assert first is fake_sapien
    assert second is fake_sapien
    assert retained is eval_module._SAPIEN_NATIVE_BOOTSTRAP_RESOURCES
    assert retained is not None
    assert calls == [
        ("device", "cpu"),
        ("device", "cuda"),
        ("render_system", retained[1]),
    ]


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


class _Box:
    def __init__(self, low: list[float], high: list[float]):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape
        self.dtype = self.low.dtype


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("robot0_pose", {("panda-0", index) for index in range(7)}),
        ("robot0_gripper", {("panda-0", 7)}),
        ("robot1_action", {("panda-1", index) for index in range(8)}),
    ],
)
def test_oracle_intervention_replaces_only_declared_dimensions_without_mutation(
    mode: str, expected: set[tuple[str, int]]
):
    names = ("panda-0", "panda-1")
    policy = {
        "panda-0": np.arange(8, dtype=np.float32),
        "panda-1": np.arange(8, dtype=np.float32) + 10,
    }
    expert = {
        "panda-0": np.arange(8, dtype=np.float32) + 100,
        "panda-1": np.arange(8, dtype=np.float32) + 200,
    }
    original = {name: value.copy() for name, value in policy.items()}

    executed, record = apply_oracle_intervention(policy, expert, names, mode)

    changed = {
        (name, index)
        for name in names
        for index in range(8)
        if executed[name][index] != policy[name][index]
    }
    assert changed == expected
    assert record["applied"] is True
    for name in names:
        np.testing.assert_array_equal(policy[name], original[name])
        assert not np.shares_memory(executed[name], policy[name])


def test_action_bound_records_and_aggregate_preserve_per_dimension_excess():
    action_space = {
        "panda-0": _Box([-1.0] * 8, [1.0] * 8),
        "panda-1": _Box([-2.0] * 8, [2.0] * 8),
    }
    action = {
        "panda-0": np.asarray([1.5, -1.25, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "panda-1": np.asarray([0, 0, 3.0, 0, 0, 0, 0, 0], dtype=np.float32),
    }

    first = summarize_bound_records(
        action_bound_records(action, action_space, step=4, source="policy")
    )
    second = summarize_bound_records(
        action_bound_records(action, action_space, step=5, source="policy")
    )
    aggregate = aggregate_bound_summaries(
        [{"bounds": first}, {"bounds": second}], "bounds"
    )

    assert first["scalar_violations"] == 3
    assert first["per_dimension_counts"] == {
        "panda-0[0]": 1,
        "panda-0[1]": 1,
        "panda-1[2]": 1,
    }
    assert first["per_dimension_max_excess"]["panda-0[0]"] == pytest.approx(0.5)
    assert aggregate["scalar_violations"] == 6
    assert aggregate["steps_with_violation"] == 2
    assert aggregate["episodes_with_violation"] == 2
    assert aggregate["per_dimension_counts"]["panda-1[2]"] == 2


def test_physical_aggregate_reports_grasp_rate_and_maximum_lift():
    aggregate = aggregate_physical_summaries(
        [
            {
                "physical": {
                    "grasped_ever": True,
                    "max_meat_lift_m": 0.06,
                    "final_meat_lift_m": 0.01,
                }
            },
            {
                "physical": {
                    "grasped_ever": False,
                    "max_meat_lift_m": 0.02,
                    "final_meat_lift_m": 0.0,
                }
            },
        ]
    )

    assert aggregate["episodes_grasped"] == 1
    assert aggregate["grasp_rate"] == pytest.approx(0.5)
    assert aggregate["maximum_meat_lift_m"] == pytest.approx(0.06)
    assert aggregate["mean_episode_max_meat_lift_m"] == pytest.approx(0.04)


def test_action_space_contract_records_exact_bounds_without_evaluator_clipping():
    contract = _action_space_contract(
        {"panda-0": _Box([-1.0] * 8, [1.0] * 8)}, ("panda-0",)
    )

    assert contract["evaluator_clipping"] is False
    assert contract["environment_action_space"]["panda-0"]["shape"] == [8]
    assert contract["environment_action_space"]["panda-0"]["low"] == [-1.0] * 8
    assert contract["environment_action_space"]["panda-0"]["high"] == [1.0] * 8
