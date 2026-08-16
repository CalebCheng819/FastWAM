from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
R21 = (
    ROOT
    / ".research-workflow"
    / "experiments"
    / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R21-20260817"
)
R22 = (
    ROOT
    / ".research-workflow"
    / "experiments"
    / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R22-20260817"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = _load("gau0_placefood_r22_controller", R22 / "controller.py")


def test_r22_identity_isolated_and_priority_frozen(monkeypatch):
    monkeypatch.setattr(
        controller.r19,
        "request_loader_namespace",
        lambda: ["/test/nvidia/lib", "/test/nvidia/driver-lib", "/test/cuda/lib64"],
    )
    assert controller.EXPERIMENT_ID.endswith("R22-20260817")
    assert controller.RUN_ID == "fastwam-gau0-placefood-same8-r22-20260817"
    assert controller.DISPLAY_NAME == "fw-gau0-placefood-same8-r22"
    assert str(controller.SOURCE_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260817-r36")
    assert str(controller.OUTPUT_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260817-r22")
    assert str(controller.DURABLE_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260817-r22-controller")
    assert str(controller.LOCAL_ROOT).endswith("gau0-placefood-same8-r22")
    request = controller.request_body("a" * 40)
    assert request["Priority"] == 7
    assert controller.r21.GRAPHICS_RUNTIME_KEYS.isdisjoint(request["Envs"])


def test_actual_execution_namespace_is_r22():
    assert controller.main.__globals__ is controller.impl.__dict__
    assert controller.impl.worker_preflight is controller.worker_preflight
    assert controller.impl.validate_worker_environment is controller.validate_worker_environment
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
        expected = getattr(controller, name)
        for module in (controller.r21, controller.r20, controller.r19, controller.r18, controller.r17, controller.impl):
            assert getattr(module, name) == expected


def test_worker_environment_accepts_provider_runtime_and_rejects_request_leaks(monkeypatch):
    reservation = {"request": {"Envs": {"FROZEN_SENTINEL": "exact"}}}
    monkeypatch.setenv("FROZEN_SENTINEL", "exact")
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", "/worker/scratch/10_nvidia.json")
    controller.validate_worker_environment(reservation)

    leaked = {"request": {"Envs": {"FROZEN_SENTINEL": "exact", "__EGL_VENDOR_LIBRARY_DIRS": "/bad"}}}
    with pytest.raises(controller.ContractError, match="frozen R22 request must not override"):
        controller.validate_worker_environment(leaked)


def test_r22_runtime_delegates_to_shared_egl_guarded_r21_runtime():
    runtime = (R22 / "runtime.sh").read_text(encoding="utf-8")
    assert "FASTWAM_RUNTIME_GENERATION='R22'" in runtime
    assert "R22-20260817" in runtime
    assert "R21-20260817/runtime.sh" in runtime
    assert 'exec /bin/bash "${FASTWAM_SOURCE_ROOT}/' in runtime

    shared = (R21 / "runtime.sh").read_text(encoding="utf-8")
    assert 'scratch_egl_manifest="${scratch_root}/10_nvidia.json"' in shared
    assert 'with output.open("x", encoding="utf-8")' in shared
    assert 'vendor.is_symlink() or not stat.S_ISREG' in shared
    assert '"file_format_version": "1.0.0"' in shared
    assert '"ICD": {"library_path": str(vendor)}' in shared
    assert 'os.chmod(output, 0o600)' in shared
    assert '__EGL_VENDOR_LIBRARY_FILENAMES="${scratch_egl_manifest}"' in shared
    assert shared.index('apply_sapien_egl_guard') < shared.index('probe_program=')


def test_r22_keeps_real_gpu_probe_and_no_cpu_fallback():
    shared = (R21 / "runtime.sh").read_text(encoding="utf-8")
    assert 'timeout --signal=TERM --kill-after=30s 180s env CUDA_VISIBLE_DEVICES=0' in shared
    assert 'environment = _build_environment(root, "PlaceFood-rf")' in shared
    assert "environment.close()" in shared
    assert 'apply_graphics_profile "${selected_profile}"' in shared
    assert "CUDA_VISIBLE_DEVICES=''" not in shared
    assert "CPU rendering is not an allowed fallback" not in shared
    assert "--no-gaussian-conditioning" in shared


def test_r22_preserves_exact_matched_panel_and_serial_shards():
    shared = (R21 / "runtime.sh").read_text(encoding="utf-8")
    run_arm = shared[shared.index("run_arm() {") : shared.index("run_arm gau1_stats")]
    assert "for shard in 0 1 2 3" in run_arm
    assert "--num-episodes 2" in shared
    assert "--max-steps 300" in shared
    assert "--exec-horizon 5" in shared
    assert "--policy-seed 10000" in shared
    assert "--action-horizon 32" in shared
    assert "--num-inference-steps 20" in shared
    assert "run_arm gau1_stats" in shared
    assert "run_arm gau0_native_stats" in shared
    assert shared.index("run_arm gau1_stats") < shared.index("run_arm gau0_native_stats")
    assert ") &" not in run_arm
    assert "wait " not in run_arm


def test_entrypoints_bind_r22_identity_and_frozen_aggregator():
    wrapper = (R22 / "submit_from_ssh970.sh").read_text(encoding="utf-8")
    assert "R22-20260817" in wrapper
    assert "gau0-placefood-same8-r22-controller.lock" in wrapper
    aggregator = _load("gau0_placefood_r22_aggregator", R22 / "aggregate_results.py")
    assert aggregator.ARMS == ("gau1_stats", "gau0_native_stats")
    assert aggregator.validate_arm is aggregator.impl.validate_arm
    assert aggregator.validate_baseline is aggregator.impl.validate_baseline
    assert aggregator.main is aggregator.impl.main


def test_controller_and_shell_entrypoints_fail_closed_or_parse():
    completed = subprocess.run(
        [sys.executable, "-B", str(R22 / "controller.py"), "worker-preflight"],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 2
    assert "GAU0_CONTROLLER_FATAL" in completed.stderr
    shell = subprocess.run(
        ["bash", "-n", str(R21 / "runtime.sh"), str(R22 / "runtime.sh"), str(R22 / "submit_from_ssh970.sh")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert shell.returncode == 0, shell.stderr


def test_r22_readme_records_failure_fix_and_terminal_boundary():
    readme = (R22 / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())
    assert "R21 failed before evaluator startup" in normalized
    assert "/usr/share/glvnd/egl_vendor.d" in readme
    assert "private, ordinary-file EGL vendor manifest" in normalized
    assert "does not modify the container filesystem" in normalized
    assert "does not" in normalized and "permit CPU rendering" in normalized
    assert "real frozen `PlaceFood-rf` environment on CUDA device 0" in normalized
    assert "16 formal episodes" in normalized
    assert "29 declared artifacts" in normalized
    assert "COMPLETE" in readme
