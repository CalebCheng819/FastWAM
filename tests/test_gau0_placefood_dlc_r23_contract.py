from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
R21 = ROOT / ".research-workflow" / "experiments" / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R21-20260817"
R23 = ROOT / ".research-workflow" / "experiments" / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R23-20260817"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = _load("gau0_placefood_r23_controller", R23 / "controller.py")


def test_r23_identity_isolated_and_priority_seven(monkeypatch):
    monkeypatch.setattr(
        controller.r19,
        "request_loader_namespace",
        lambda: ["/test/nvidia/lib", "/test/nvidia/driver-lib", "/test/cuda/lib64"],
    )
    assert controller.EXPERIMENT_ID.endswith("R23-20260817")
    assert controller.RUN_ID == "fastwam-gau0-placefood-same8-r23-20260817"
    assert controller.DISPLAY_NAME == "fw-gau0-placefood-same8-r23"
    assert str(controller.SOURCE_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260817-r37")
    assert str(controller.OUTPUT_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260817-r23")
    assert str(controller.DURABLE_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260817-r23-controller")
    assert str(controller.LOCAL_ROOT).endswith("gau0-placefood-same8-r23")
    request = controller.request_body("a" * 40)
    assert request["Priority"] == 7
    assert controller.r21.GRAPHICS_RUNTIME_KEYS.isdisjoint(request["Envs"])


def test_actual_execution_namespace_is_r23():
    assert controller.main.__globals__ is controller.impl.__dict__
    assert controller.impl.worker_preflight is controller.worker_preflight
    assert controller.impl.validate_worker_environment is controller.validate_worker_environment
    for name in (
        "EXPERIMENT_ID", "RUN_ID", "DISPLAY_NAME", "SOURCE_ROOT", "OUTPUT_ROOT",
        "DURABLE_ROOT", "RESERVATION_PATH", "LATCH_PATH", "ACK_PATH", "LOCAL_ROOT",
        "STATE_PATH", "EXPERIMENT_REL",
    ):
        expected = getattr(controller, name)
        for module in (
            controller.r22, controller.r21, controller.r20, controller.r19,
            controller.r18, controller.r17, controller.impl,
        ):
            assert getattr(module, name) == expected


def test_worker_environment_accepts_private_runtime_and_rejects_request_leaks(monkeypatch):
    reservation = {"request": {"Envs": {"FROZEN_SENTINEL": "exact"}}}
    monkeypatch.setenv("FROZEN_SENTINEL", "exact")
    monkeypatch.setenv("FASTWAM_GL_SHIM_ROOT", "/tmp/r23/glvnd")
    controller.validate_worker_environment(reservation)
    leaked = {"request": {"Envs": {"FROZEN_SENTINEL": "exact", "FASTWAM_GL_SHIM_ROOT": "/bad"}}}
    with pytest.raises(controller.ContractError, match="frozen R23 request must not override"):
        controller.validate_worker_environment(leaked)


def test_r23_complete_provider_native_graphics_contract():
    runtime = (R23 / "runtime.sh").read_text(encoding="utf-8")
    assert "FASTWAM_RUNTIME_GENERATION='R23'" in runtime
    for soname in (
        "libEGL.so.1", "libGL.so.1", "libGLESv1_CM.so.1", "libGLESv2.so.2",
        "libOpenGL.so.0", "libGLX.so.0", "libGLdispatch.so.0", "libvulkan.so.1",
    ):
        assert soname in runtime
    assert 'VK_ICD_FILENAMES="${FASTWAM_NVIDIA_GRAPHICS_ROOT}/nvidia_icd.json"' in runtime
    assert '__EGL_VENDOR_LIBRARY_FILENAMES="${FASTWAM_NVIDIA_GRAPHICS_ROOT}/10_nvidia.json"' in runtime
    assert 'FASTWAM_REQUIRE_PROVIDER_NATIVE_GRAPHICS=1' in runtime
    assert 'hasattr(egl, "eglQueryString")' in runtime
    assert 'hasattr(vendor, "__egl_Main")' in runtime
    assert "NVIDIA EGL vendor lacks __egl_Main" in runtime
    assert "vkEnumerateInstanceVersion" in runtime
    assert 'environment = _build_environment(root, "PlaceFood-rf")' in runtime
    assert "environment.close()" in runtime
    assert 'env CUDA_VISIBLE_DEVICES=0' in runtime


def test_r23_shared_runtime_fails_closed_to_provider_native():
    shared = (R21 / "runtime.sh").read_text(encoding="utf-8")
    gate = 'if [[ "${FASTWAM_REQUIRE_PROVIDER_NATIVE_GRAPHICS:-0}" == \'1\' ]]'
    assert gate in shared
    assert "profiles=(provider_native_headless)" in shared
    assert shared.index(gate) < shared.index("selected_profile=''")


def test_r23_preserves_exact_two_arm_panel_and_terminal_contract():
    shared = (R21 / "runtime.sh").read_text(encoding="utf-8")
    assert "--no-gaussian-conditioning" in shared
    assert "--num-episodes 2" in shared
    assert "for shard in 0 1 2 3" in shared
    assert "run_arm gau1_stats" in shared
    assert "run_arm gau0_native_stats" in shared
    assert shared.index("run_arm gau1_stats") < shared.index("run_arm gau0_native_stats")
    aggregator = _load("gau0_placefood_r23_aggregator", R23 / "aggregate_results.py")
    assert aggregator.ARMS == ("gau1_stats", "gau0_native_stats")


def test_entrypoints_parse_and_fail_closed():
    completed = subprocess.run(
        [sys.executable, "-B", str(R23 / "controller.py"), "worker-preflight"],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 2
    assert "GAU0_CONTROLLER_FATAL" in completed.stderr
    shell = subprocess.run(
        ["bash", "-n", str(R21 / "runtime.sh"), str(R23 / "runtime.sh"), str(R23 / "submit_from_ssh970.sh")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert shell.returncode == 0, shell.stderr


def test_r23_readme_records_non_result_failures_and_scope():
    readme = (R23 / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())
    assert "R19" in readme and "segfaulted" in normalized
    assert "R22" in readme and "episode 0" in normalized and "eglQueryString" in readme
    assert "__egl_Main" in readme
    assert "EGL, GL, GLES1/2, OpenGL, GLX, GLdispatch" in normalized
    assert "provider-native profile" in normalized
    assert "16 complete episodes" in normalized
    assert "not a train-time causal ablation" in normalized
