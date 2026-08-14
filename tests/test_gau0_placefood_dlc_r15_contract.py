from __future__ import annotations

import importlib.util
import inspect
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R15 = ROOT / ".research-workflow" / "experiments" / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R15-20260814"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = _load("gau0_placefood_r15_controller", R15 / "controller.py")


def test_r15_identity_isolated_and_priority_frozen():
    assert controller.EXPERIMENT_ID.endswith("R15-20260814")
    assert controller.RUN_ID == "fastwam-gau0-placefood-same8-r15-20260814"
    assert controller.DISPLAY_NAME == "fw-gau0-placefood-same8-r15"
    assert str(controller.SOURCE_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260814-r24")
    assert str(controller.OUTPUT_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260814-r15")
    assert controller.request_body("a" * 40)["Priority"] == 7


def test_actual_execution_namespace_is_r15_not_an_intermediate_wrapper():
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


def test_worker_environment_matches_request_and_retains_glvnd_root(tmp_path):
    worker = controller.worker_dependency_env(tmp_path)
    request = controller.runtime_env("a" * 40)
    assert worker["FASTWAM_GL_SHIM_ROOT"] == str(tmp_path)
    assert worker["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(tmp_path)
    assert worker["SAPIEN_VULKAN_LIBRARY_PATH"] == str(controller.VULKAN_LOADER)
    assert worker["SAPIEN_VULKAN_LIBRARY_PATH"] == request["SAPIEN_VULKAN_LIBRARY_PATH"]
    assert controller.impl.worker_dependency_env is controller.worker_dependency_env


def test_controller_cli_propagates_fail_closed_status():
    completed = subprocess.run(
        [sys.executable, "-B", str(R15 / "controller.py"), "worker-preflight"],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 2
    assert "GAU0_CONTROLLER_FATAL" in completed.stderr
    source = (R15 / "controller.py").read_text(encoding="utf-8")
    assert "raise SystemExit(main())" in source


def test_dependency_preflight_freezes_shim_before_heavy_imports():
    source = inspect.getsource(controller.impl.validate_worker_dependencies)
    frozen = 'shim_root = Path(os.environ["FASTWAM_GL_SHIM_ROOT"]).resolve(strict=True)'
    assert frozen in source
    assert source.index(frozen) < source.index("import boto3")
    assert source.index(frozen) < source.index("import deepspeed")
    assert source.count(frozen) == 1
    assert "initial loader namespace mismatch" in source
    assert source.index("vkEnumerateInstanceVersion") < source.index("import mani_skill")


def test_runtime_preflights_real_environment_and_serializes_shards():
    runtime = (R15 / "runtime.sh").read_text(encoding="utf-8")
    assert 'export SAPIEN_VULKAN_LIBRARY_PATH="${FASTWAM_VULKAN_LOADER}"' in runtime
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


def test_r15_entrypoints_bind_r15_identity_and_lock():
    wrapper = (R15 / "submit_from_ssh970.sh").read_text(encoding="utf-8")
    assert "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R15-20260814" in wrapper
    assert "FASTWAM_LOCK_NAME_OVERRIDE='gau0-placefood-same8-r15-controller.lock'" in wrapper
    assert "export PYTHONDONTWRITEBYTECODE=1" in wrapper
    assert "FASTWAM_WRAPPER_ENTRYPOINT" in wrapper


def test_r15_aggregator_exposes_frozen_terminal_contract():
    aggregator = _load("gau0_placefood_r15_aggregator", R15 / "aggregate_results.py")
    assert aggregator.ARMS == ("gau1_stats", "gau0_native_stats")
    assert aggregator.validate_arm is aggregator.impl.validate_arm
    assert aggregator.validate_baseline is aggregator.impl.validate_baseline
    assert aggregator.comparison is aggregator.impl.comparison
    assert aggregator.main is aggregator.impl.main


def test_r15_readme_records_failure_and_scientific_boundary():
    readme = (R15 / "README.md").read_text(encoding="utf-8")
    assert "R13 created one DLC job" in readme
    assert "produced zero episodes" in readme
    assert "R14 was stopped before submission" in readme
    assert "zero cloud create calls" in readme
    assert "pytest cache" in readme
    assert "failed closed" in readme
    assert "sequentially" in readme
    assert "exact historical eight" in readme
    assert "not a pure causal" in readme
