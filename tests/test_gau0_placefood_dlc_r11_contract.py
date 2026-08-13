from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R10 = ROOT / ".research-workflow" / "experiments" / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R10-20260814"
R11 = ROOT / ".research-workflow" / "experiments" / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R11-20260814"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = _load("gau0_placefood_r11_controller", R11 / "controller.py")


def test_r11_identity_isolated_and_priority_frozen():
    assert controller.EXPERIMENT_ID.endswith("R11-20260814")
    assert controller.RUN_ID == "fastwam-gau0-placefood-same8-r11-20260814"
    assert controller.DISPLAY_NAME == "fw-gau0-placefood-same8-r11"
    assert str(controller.SOURCE_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260814-r20")
    assert str(controller.OUTPUT_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260814-r11")
    assert controller.request_body("a" * 40)["Priority"] == 7


def test_vulkan_loader_is_bound_into_request_and_preflight():
    env = controller.runtime_env("a" * 40)
    assert env["FASTWAM_VULKAN_LOADER"] == str(controller.VULKAN_LOADER)
    assert env["FASTWAM_VULKAN_LOADER_SIZE_BYTES"] == str(controller.VULKAN_LOADER_BYTES)
    assert env["SAPIEN_VULKAN_LIBRARY_PATH"] == str(controller.VULKAN_LOADER)
    assert env["PYTHONFAULTHANDLER"] == "1"
    source = inspect.getsource(controller.impl.validate_worker_dependencies)
    assert "vkEnumerateInstanceVersion" in source
    assert source.index("vkEnumerateInstanceVersion") < source.index("import mani_skill")


def test_r10_runtime_builds_vulkan_soname_namespace_before_evaluator():
    runtime = (R10 / "runtime.sh").read_text(encoding="utf-8")
    assert 'ln -s -- "${FASTWAM_VULKAN_LOADER}" "${scratch_root}/glvnd-runtime/libvulkan.so.1"' in runtime
    assert "SAPIEN_VULKAN_LIBRARY_PATH" in runtime
    assert "vkEnumerateInstanceVersion" in runtime
    assert runtime.index("vkEnumerateInstanceVersion") < runtime.index('"${FASTWAM_PYTHON}" -B "${controller}" worker-preflight')


def test_r11_thin_entrypoints_rebind_all_namespaces():
    runtime = (R11 / "runtime.sh").read_text(encoding="utf-8")
    wrapper = (R11 / "submit_from_ssh970.sh").read_text(encoding="utf-8")
    assert "FASTWAM_EXPERIMENT_REL_OVERRIDE" in runtime
    assert "fastwam-gau0-placefood-r11" in runtime
    assert "FASTWAM_LOCK_NAME_OVERRIDE='gau0-placefood-same8-r11-controller.lock'" in wrapper
    assert "FASTWAM_WRAPPER_ENTRYPOINT" in wrapper


def test_r11_aggregator_exposes_the_frozen_terminal_contract():
    aggregator = _load("gau0_placefood_r11_aggregator", R11 / "aggregate_results.py")
    assert aggregator.ARMS == ("gau1_stats", "gau0_native_stats")
    assert aggregator.validate_arm is aggregator.impl.validate_arm
    assert aggregator.validate_baseline is aggregator.impl.validate_baseline
    assert aggregator.comparison is aggregator.impl.comparison
    assert aggregator.main is aggregator.impl.main


def test_r11_readme_preserves_r10_and_comparison_boundary():
    readme = (R11 / "README.md").read_text(encoding="utf-8")
    assert "R11 is a new isolated run after R10" in readme
    assert "bundled loader" in readme
    assert "exact historical eight PlaceFood" in readme
    assert "not a pure causal" in readme
