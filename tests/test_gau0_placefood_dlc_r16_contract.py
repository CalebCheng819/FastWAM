from __future__ import annotations

import importlib.util
import inspect
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R16 = ROOT / ".research-workflow" / "experiments" / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R16-20260814"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = _load("gau0_placefood_r16_controller", R16 / "controller.py")


def test_r16_identity_isolated_and_priority_frozen():
    assert controller.EXPERIMENT_ID.endswith("R16-20260814")
    assert controller.RUN_ID == "fastwam-gau0-placefood-same8-r16-20260814"
    assert controller.DISPLAY_NAME == "fw-gau0-placefood-same8-r16"
    assert str(controller.SOURCE_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260814-r25")
    assert str(controller.OUTPUT_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260814-r16")
    assert str(controller.DURABLE_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260814-r16-controller")
    assert str(controller.LOCAL_ROOT).endswith("gau0-placefood-same8-r16")
    assert controller.request_body("a" * 40)["Priority"] == 7


def test_actual_execution_namespace_is_r16_not_an_intermediate_wrapper():
    assert controller.main.__globals__ is controller.impl.__dict__
    assert controller.impl.prepare.__globals__ is controller.impl.__dict__
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
        assert controller.impl.prepare.__globals__[name] == getattr(controller, name)


def test_worker_and_request_use_minimal_vendor_loader_namespace(tmp_path):
    worker = controller.worker_dependency_env(tmp_path)
    request = controller.runtime_env("a" * 40)
    expected_prefix = [
        str(controller.NVIDIA_GRAPHICS_ROOT / "driver-lib"),
        "/usr/local/cuda-12.8/lib64",
    ]
    for env in (worker, request):
        assert "FASTWAM_GL_SHIM_ROOT" not in env
        assert "SAPIEN_VULKAN_LIBRARY_PATH" not in env
        assert env["LD_LIBRARY_PATH"].split(os.pathsep)[:2] == expected_prefix
        assert env["VK_ICD_FILENAMES"] == str(controller.NVIDIA_GRAPHICS_ROOT / "nvidia_icd.json")
        assert env["__EGL_VENDOR_LIBRARY_FILENAMES"] == str(controller.NVIDIA_GRAPHICS_ROOT / "10_nvidia.json")
    assert controller.impl.worker_dependency_env is controller.worker_dependency_env
    assert controller.impl.runtime_env is controller.runtime_env


def test_controller_cli_propagates_fail_closed_status():
    completed = subprocess.run(
        [sys.executable, "-B", str(R16 / "controller.py"), "worker-preflight"],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 2
    assert "GAU0_CONTROLLER_FATAL" in completed.stderr
    source = (R16 / "controller.py").read_text(encoding="utf-8")
    assert "raise SystemExit(main())" in source


def test_dependency_preflight_rejects_manual_frontends_before_imports():
    source = inspect.getsource(controller.impl.validate_worker_dependencies)
    absent_sapien = 'if "SAPIEN_VULKAN_LIBRARY_PATH" in os.environ:'
    absent_shim = 'if "FASTWAM_GL_SHIM_ROOT" in os.environ:'
    assert source.index(absent_sapien) < source.index("import boto3")
    assert source.index(absent_shim) < source.index("import boto3")
    assert "minimal loader namespace mismatch" in source
    assert "_preflight_environment_imports" in source
    assert "ctypes" not in source
    assert "vkEnumerateInstanceVersion" not in source
    assert "glvnd-runtime" not in source


def test_runtime_preflights_real_environment_and_serializes_shards():
    runtime = (R16 / "runtime.sh").read_text(encoding="utf-8")
    assert "unset SAPIEN_VULKAN_LIBRARY_PATH FASTWAM_GL_SHIM_ROOT" in runtime
    assert "ctypes.CDLL" not in runtime
    assert "glvnd-runtime" not in runtime
    assert "GAU0_MINIMAL_GRAPHICS_RUNTIME_PREFLIGHT_PASS" in runtime
    assert "worker-preflight" in runtime
    assert "_build_environment" in runtime
    assert 'environment = _build_environment(root, "PlaceFood-rf")' in runtime
    assert "environment.close()" in runtime
    assert "GAU0_ENVIRONMENT_CONSTRUCTION_PREFLIGHT_PASS" in runtime
    assert runtime.index("worker-preflight") < runtime.index("GAU0_ENVIRONMENT_CONSTRUCTION_PREFLIGHT_PASS")

    run_arm = runtime[runtime.index("run_arm() {") : runtime.index("run_arm gau1_stats")]
    assert "for shard in 0 1 2 3" in run_arm
    assert "pids" not in run_arm
    assert "wait " not in run_arm
    assert ") &" not in run_arm
    assert "if ! (" in run_arm
    assert 'die "${arm} evaluator failed at ${shard_name}"' in run_arm
    assert "run_arm gau1_stats" in runtime
    assert "run_arm gau0_native_stats" in runtime
    assert runtime.index("run_arm gau1_stats") < runtime.index("run_arm gau0_native_stats")
    for frozen_arg in (
        "--no-gaussian-conditioning",
        "--num-episodes 2",
        "--max-steps 300",
        "--exec-horizon 5",
        "--policy-seed 10000",
        "--action-horizon 32",
        "--num-inference-steps 20",
    ):
        assert frozen_arg in runtime


def test_r16_entrypoints_bind_r16_identity_and_lock():
    wrapper = (R16 / "submit_from_ssh970.sh").read_text(encoding="utf-8")
    assert "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R16-20260814" in wrapper
    assert "FASTWAM_LOCK_NAME_OVERRIDE='gau0-placefood-same8-r16-controller.lock'" in wrapper
    assert "export PYTHONDONTWRITEBYTECODE=1" in wrapper
    assert "FASTWAM_WRAPPER_ENTRYPOINT" in wrapper


def test_r16_aggregator_exposes_frozen_terminal_contract():
    aggregator = _load("gau0_placefood_r16_aggregator", R16 / "aggregate_results.py")
    assert aggregator.ARMS == ("gau1_stats", "gau0_native_stats")
    assert aggregator.validate_arm is aggregator.impl.validate_arm
    assert aggregator.validate_baseline is aggregator.impl.validate_baseline
    assert aggregator.comparison is aggregator.impl.comparison
    assert aggregator.main is aggregator.impl.main


def test_r16_readme_records_failure_fix_and_scientific_boundary():
    readme = (R16 / "README.md").read_text(encoding="utf-8")
    assert "R15 created exactly one Priority-7 DLC job" in readme
    assert "completed zero episodes" in readme
    assert "exit code 139" in readme
    assert "manual GLVND frontend shim" in readme
    assert "minimal NVIDIA vendor-driver namespace" in readme
    assert "sequentially" in readme
    assert "exact historical eight" in readme
    assert "not a pure causal" in readme
