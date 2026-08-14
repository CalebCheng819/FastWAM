from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
R20 = (
    ROOT
    / ".research-workflow"
    / "experiments"
    / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R20-20260815"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = _load("gau0_placefood_r20_controller", R20 / "controller.py")


def test_r20_identity_isolated_and_priority_frozen(monkeypatch):
    monkeypatch.setattr(
        controller.r19,
        "request_loader_namespace",
        lambda: ["/test/nvidia/lib", "/test/nvidia/driver-lib", "/test/cuda/lib64"],
    )
    assert controller.EXPERIMENT_ID.endswith("R20-20260815")
    assert controller.RUN_ID == "fastwam-gau0-placefood-same8-r20-20260815"
    assert controller.DISPLAY_NAME == "fw-gau0-placefood-same8-r20"
    assert str(controller.SOURCE_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260815-r33")
    assert str(controller.OUTPUT_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260815-r20")
    assert str(controller.DURABLE_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260815-r20-controller")
    assert str(controller.LOCAL_ROOT).endswith("gau0-placefood-same8-r20")
    request = controller.request_body("a" * 40)
    assert request["Priority"] == 7
    assert controller.GRAPHICS_RUNTIME_KEYS.isdisjoint(request["Envs"])


def test_actual_execution_namespace_is_r20():
    assert controller.main.__globals__ is controller.impl.__dict__
    assert controller.impl.worker_preflight is controller.worker_preflight
    assert controller.impl.runtime_env is controller.runtime_env
    for name in (
        "EXPERIMENT_ID",
        "RUN_ID",
        "DISPLAY_NAME",
        "SOURCE_ROOT",
        "OUTPUT_ROOT",
        "DURABLE_ROOT",
        "RESERVATION_PATH",
        "LATCH_PATH",
        "ACK_PATH",
        "LOCAL_ROOT",
        "STATE_PATH",
        "EXPERIMENT_REL",
    ):
        assert getattr(controller.r19, name) == getattr(controller, name)
        assert getattr(controller.impl, name) == getattr(controller, name)


def test_runtime_env_removes_every_graphics_override_but_keeps_asset_bindings(monkeypatch):
    monkeypatch.setattr(
        controller.r19,
        "request_loader_namespace",
        lambda: ["/test/nvidia/lib", "/test/nvidia/driver-lib", "/test/cuda/lib64"],
    )
    inherited = controller._base_runtime_env("a" * 40)
    actual = controller.runtime_env("a" * 40)
    assert controller.GRAPHICS_RUNTIME_KEYS.intersection(inherited)
    assert controller.GRAPHICS_RUNTIME_KEYS.isdisjoint(actual)
    for key in (
        "FASTWAM_NVIDIA_GRAPHICS_ROOT",
        "FASTWAM_VULKAN_LOADER",
        "FASTWAM_EGL_FRONTEND",
        "FASTWAM_SOURCE_ROOT",
        "FASTWAM_OUTPUT_ROOT",
    ):
        assert actual[key] == inherited[key]


def test_worker_environment_accepts_provider_runtime_and_rejects_request_leaks(monkeypatch):
    reservation = {"request": {"Envs": {"FROZEN_SENTINEL": "exact"}}}
    monkeypatch.setenv("FROZEN_SENTINEL", "exact")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/provider/runtime")
    controller.validate_worker_environment(reservation)

    leaked = {"request": {"Envs": {"FROZEN_SENTINEL": "exact", "VK_DRIVER_FILES": "/bad"}}}
    with pytest.raises(controller.ContractError, match="must not override provider graphics runtime"):
        controller.validate_worker_environment(leaked)


def test_worker_dependency_env_preserves_provider_graphics_runtime(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/provider/driver")
    monkeypatch.setenv("VK_DRIVER_FILES", "/provider/nvidia_icd.json")
    monkeypatch.setattr(controller.impl, "worker_pythonpath", lambda: "frozen-pythonpath")
    env = controller.worker_dependency_env()
    assert env["LD_LIBRARY_PATH"] == "/provider/driver"
    assert env["VK_DRIVER_FILES"] == "/provider/nvidia_icd.json"
    assert env["PYTHONPATH"] == "frozen-pythonpath"


def test_runtime_profiles_are_isolated_real_environment_probes_and_no_cpu_fallback():
    runtime = (R20 / "runtime.sh").read_text(encoding="utf-8")
    expected_profiles = (
        "provider_native_headless",
        "provider_clean_headless",
        "system_default_headless",
        "system_discovered_headless",
        "system_manifest_headless",
        "system_discovered_sapien_loader",
    )
    for profile in expected_profiles:
        assert profile in runtime
    assert 'timeout --signal=TERM --kill-after=30s 180s env CUDA_VISIBLE_DEVICES=0' in runtime
    assert 'environment = _build_environment(root, "PlaceFood-rf")' in runtime
    assert "environment.close()" in runtime
    assert "GAU0_R20_GRAPHICS_PROFILE_REJECTED" in runtime
    assert "GAU0_R20_GRAPHICS_PROFILE_SELECTED" in runtime
    assert 'apply_graphics_profile "${selected_profile}"' in runtime
    assert "CPU rendering is not an allowed fallback" not in runtime
    assert "CUDA_VISIBLE_DEVICES=''" not in runtime
    assert "--no-gaussian-conditioning" in runtime
    assert runtime.index("worker-preflight") < runtime.index("profiles=(")
    assert runtime.index("profiles=(") < runtime.index("run_arm gau1_stats")


def test_runtime_preserves_exact_matched_panel_and_serial_shards():
    runtime = (R20 / "runtime.sh").read_text(encoding="utf-8")
    run_arm = runtime[runtime.index("run_arm() {") : runtime.index("run_arm gau1_stats")]
    assert "for shard in 0 1 2 3" in run_arm
    assert "--num-episodes 2" in runtime
    assert "--max-steps 300" in runtime
    assert "--exec-horizon 5" in runtime
    assert "--policy-seed 10000" in runtime
    assert "--action-horizon 32" in runtime
    assert "--num-inference-steps 20" in runtime
    assert "run_arm gau1_stats" in runtime
    assert "run_arm gau0_native_stats" in runtime
    assert runtime.index("run_arm gau1_stats") < runtime.index("run_arm gau0_native_stats")
    assert ") &" not in run_arm
    assert "wait " not in run_arm


def test_entrypoints_bind_r20_identity_and_frozen_aggregator():
    wrapper = (R20 / "submit_from_ssh970.sh").read_text(encoding="utf-8")
    assert "R20-20260815" in wrapper
    assert "gau0-placefood-same8-r20-controller.lock" in wrapper
    aggregator = _load("gau0_placefood_r20_aggregator", R20 / "aggregate_results.py")
    assert aggregator.ARMS == ("gau1_stats", "gau0_native_stats")
    assert aggregator.validate_arm is aggregator.impl.validate_arm
    assert aggregator.validate_baseline is aggregator.impl.validate_baseline
    assert aggregator.main is aggregator.impl.main


def test_controller_and_shell_entrypoints_fail_closed_or_parse():
    completed = subprocess.run(
        [sys.executable, "-B", str(R20 / "controller.py"), "worker-preflight"],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 2
    assert "GAU0_CONTROLLER_FATAL" in completed.stderr
    shell = subprocess.run(
        ["bash", "-n", str(R20 / "runtime.sh"), str(R20 / "submit_from_ssh970.sh")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert shell.returncode == 0, shell.stderr


def test_readme_states_the_scientific_and_publication_boundary():
    readme = (R20 / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())
    assert "same eight frozen episodes twice" in readme
    assert "eight evaluator subprocesses" in readme
    assert "16 episode evaluations" in readme
    assert "CPU rendering is not an allowed fallback" in readme
    assert "expected 29 artifact files" in normalized
    assert "COMPLETE" in readme
