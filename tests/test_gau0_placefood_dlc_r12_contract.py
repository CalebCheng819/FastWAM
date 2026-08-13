from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R10 = ROOT / ".research-workflow" / "experiments" / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R10-20260814"
R12 = ROOT / ".research-workflow" / "experiments" / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R12-20260814"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = _load("gau0_placefood_r12_controller", R12 / "controller.py")


def test_r12_identity_isolated_and_priority_frozen():
    assert controller.EXPERIMENT_ID.endswith("R12-20260814")
    assert controller.RUN_ID == "fastwam-gau0-placefood-same8-r12-20260814"
    assert controller.DISPLAY_NAME == "fw-gau0-placefood-same8-r12"
    assert str(controller.SOURCE_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260814-r21")
    assert str(controller.OUTPUT_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260814-r12")
    assert controller.request_body("a" * 40)["Priority"] == 7


def test_worker_environment_freezes_shim_root_separately_from_mutable_loader_path(tmp_path):
    env = controller.worker_dependency_env(tmp_path)
    assert env["FASTWAM_GL_SHIM_ROOT"] == str(tmp_path)
    assert env["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(tmp_path)
    assert env["SAPIEN_VULKAN_LIBRARY_PATH"] == str(tmp_path / "libvulkan.so.1")


def test_dependency_preflight_freezes_shim_before_heavy_imports():
    source = inspect.getsource(controller.impl.impl.validate_worker_dependencies)
    frozen = 'shim_root = Path(os.environ["FASTWAM_GL_SHIM_ROOT"]).resolve(strict=True)'
    assert frozen in source
    assert source.index(frozen) < source.index("import boto3")
    assert source.index(frozen) < source.index("import deepspeed")
    assert source.count(frozen) == 1
    assert 'shim_root = Path(os.environ["LD_LIBRARY_PATH"].split(os.pathsep)[0])' not in source
    assert "initial loader namespace mismatch" in source
    assert source.index("vkEnumerateInstanceVersion") < source.index("import mani_skill")


def test_r12_thin_entrypoints_rebind_all_namespaces():
    runtime = (R12 / "runtime.sh").read_text(encoding="utf-8")
    wrapper = (R12 / "submit_from_ssh970.sh").read_text(encoding="utf-8")
    assert "FASTWAM_EXPERIMENT_REL_OVERRIDE" in runtime
    assert "fastwam-gau0-placefood-r12" in runtime
    assert "FASTWAM_LOCK_NAME_OVERRIDE='gau0-placefood-same8-r12-controller.lock'" in wrapper
    assert "FASTWAM_WRAPPER_ENTRYPOINT" in wrapper


def test_r12_aggregator_exposes_frozen_terminal_contract():
    aggregator = _load("gau0_placefood_r12_aggregator", R12 / "aggregate_results.py")
    assert aggregator.ARMS == ("gau1_stats", "gau0_native_stats")
    assert aggregator.validate_arm is aggregator.impl.validate_arm
    assert aggregator.validate_baseline is aggregator.impl.validate_baseline
    assert aggregator.comparison is aggregator.impl.comparison
    assert aggregator.main is aggregator.impl.main


def test_r12_readme_preserves_r11_and_scientific_boundary():
    readme = (R12 / "README.md").read_text(encoding="utf-8")
    assert "R11 stopped during" in readme
    assert "before\nprepare, SDK loading, or CreateJob" in readme
    assert "FASTWAM_GL_SHIM_ROOT" in readme
    assert "exact historical eight" in readme
    assert "not a pure causal" in readme
