from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
R21 = ROOT / ".research-workflow" / "experiments" / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R21-20260817"
R25 = ROOT / ".research-workflow" / "experiments" / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R25-20260817"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = _load("gau0_placefood_r25_controller", R25 / "controller.py")


def _binding(path: Path, payload: dict) -> dict:
    raw = json.dumps(payload).encode("utf-8")
    encoded = base64.b64encode(raw).decode("ascii")
    return {"path": str(path), "bytes": len(raw), "content_b64": encoded}


def _train_payload() -> dict:
    return {
        "normalization_fit": {"split": "train", "split_seed": 42, "val_set_proportion": 0.1},
        "cardinality": {"agent_counts": [2, 3, 4]},
        "action": {"mean": [0.0] * 8, "std": [1.0] * 8},
        "state": {"mean": [0.0] * 18, "std": [1.0] * 18},
    }


def _legacy_payload() -> dict:
    return {
        "source_root": "/cpfs/user/chengjuntao/datasets/robofactory_multi_robot",
        "files": 24,
        "trajectories": 1587,
        "cardinality": {
            "agent_counts": [2, 3, 4],
            "trajectories_by_agent_count": {"2": 562, "3": 802, "4": 223},
        },
        "action": {
            "count": 2572601,
            "max": [1.0] * 8,
            "mean": [0.0] * 8,
            "min": [-1.0] * 8,
            "std": [1.0] * 8,
        },
        "state": {
            "count": 2577023,
            "max": [1.0] * 18,
            "mean": [0.0] * 18,
            "min": [-1.0] * 18,
            "std": [1.0] * 18,
        },
    }


def _semantic_bindings() -> dict:
    return {
        "gau1_stats": _binding(controller.impl.GAU1_STATS, _train_payload()),
        "gau0_native_stats": _binding(controller.impl.GAU0_STATS, _legacy_payload()),
    }


def test_r25_identity_priority_and_execution_namespace(monkeypatch):
    monkeypatch.setattr(
        controller.r19,
        "request_loader_namespace",
        lambda: ["/test/nvidia/lib", "/test/nvidia/driver-lib", "/test/cuda/lib64"],
    )
    assert controller.EXPERIMENT_ID.endswith("R25-20260817")
    assert controller.RUN_ID == "fastwam-gau0-placefood-same8-r25-20260817"
    assert controller.DISPLAY_NAME == "fw-gau0-placefood-same8-r25"
    assert str(controller.SOURCE_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260817-r39")
    assert str(controller.OUTPUT_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260817-r25")
    assert controller.request_body("a" * 40)["Priority"] == 7
    assert controller.main.__globals__ is controller.impl.__dict__
    assert controller.impl.input_bindings is controller.input_bindings
    assert controller.impl.validate_inputs is controller.validate_inputs


def test_r25_stats_semantics_accept_only_exact_frozen_pair():
    controller.validate_stats_semantics(_semantic_bindings())

    crossed = _semantic_bindings()
    crossed["gau0_native_stats"] = _binding(controller.impl.GAU0_STATS, _train_payload())
    with pytest.raises(controller.ContractError, match="must not declare normalization_fit"):
        controller.validate_stats_semantics(crossed)

    drifted = _semantic_bindings()
    payload = _legacy_payload()
    payload["action"]["mean"].pop()
    drifted["gau0_native_stats"] = _binding(controller.impl.GAU0_STATS, payload)
    with pytest.raises(controller.ContractError, match="action.mean dimension mismatch"):
        controller.validate_stats_semantics(drifted)


def test_r25_semantic_gate_runs_during_capture_and_live_validation(monkeypatch):
    bindings = _semantic_bindings()
    monkeypatch.setattr(controller, "_base_input_bindings", lambda: bindings)
    assert controller.input_bindings() is bindings

    calls = []
    monkeypatch.setattr(controller, "_base_validate_inputs", lambda value: calls.append(value))
    controller.validate_inputs(bindings)
    assert calls == [bindings]


def test_r25_runtime_binds_each_arm_to_its_stats_contract():
    shared = (R21 / "runtime.sh").read_text(encoding="utf-8")
    assert "run_arm gau1_stats" in shared and "train_split" in shared
    assert "run_arm gau0_native_stats" in shared and "legacy_full_dataset" in shared
    runtime = (R25 / "runtime.sh").read_text(encoding="utf-8")
    assert "FASTWAM_RUNTIME_GENERATION='R25'" in runtime
    assert "R23-20260817/runtime.sh" in runtime


def test_r25_entrypoints_parse_and_fail_closed():
    completed = subprocess.run(
        [sys.executable, "-B", str(R25 / "controller.py"), "worker-preflight"],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 2
    assert "GAU0_CONTROLLER_FATAL" in completed.stderr
    shell = subprocess.run(
        ["bash", "-n", str(R25 / "runtime.sh"), str(R25 / "submit_from_ssh970.sh")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert shell.returncode == 0, shell.stderr


def test_r25_readme_preserves_scientific_boundary():
    normalized = " ".join((R25 / "README.md").read_text(encoding="utf-8").split())
    assert "before episode 0" in normalized
    assert "16 complete episodes" in normalized
    assert "not a train-time causal ablation" in normalized
    assert "historical `gau1_baseline`" in normalized
    assert "different checkpoint/training scope" in normalized
