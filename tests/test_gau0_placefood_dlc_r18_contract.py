from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
R18 = ROOT / ".research-workflow" / "experiments" / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R18-20260814"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = _load("gau0_placefood_r18_controller", R18 / "controller.py")


def test_r18_identity_isolated_and_priority_frozen():
    assert controller.EXPERIMENT_ID.endswith("R18-20260814")
    assert controller.RUN_ID == "fastwam-gau0-placefood-same8-r18-20260814"
    assert controller.DISPLAY_NAME == "fw-gau0-placefood-same8-r18"
    assert str(controller.SOURCE_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260814-r31")
    assert str(controller.OUTPUT_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260814-r18")
    assert str(controller.DURABLE_ROOT).endswith("fastwam-gau0-placefood-same8-eval-20260814-r18-controller")
    assert str(controller.LOCAL_ROOT).endswith("gau0-placefood-same8-r18")
    assert controller.request_body("a" * 40)["Priority"] == 7


def test_actual_execution_namespace_is_r18():
    assert controller.main.__globals__ is controller.impl.__dict__
    assert controller.impl.ControllerError is controller.impl.ContractError
    assert controller.impl.worker_preflight is controller.worker_preflight
    for name in (
        "EXPERIMENT_ID", "RUN_ID", "DISPLAY_NAME", "SOURCE_ROOT", "OUTPUT_ROOT",
        "DURABLE_ROOT", "RESERVATION_PATH", "LATCH_PATH", "ACK_PATH", "LOCAL_ROOT",
        "STATE_PATH", "EXPERIMENT_REL",
    ):
        assert getattr(controller.r17, name) == getattr(controller, name)
        assert getattr(controller.impl, name) == getattr(controller, name)


def _worker_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path]:
    graphics = tmp_path / "graphics"
    driver = graphics / "driver-lib"
    library = graphics / "lib"
    driver.mkdir(parents=True)
    library.mkdir()
    egl = library / "libEGL.so.1.1.0"
    egl.write_bytes(b"frozen-egl")
    shim = tmp_path / "egl-runtime"
    shim.mkdir()
    (shim / "libEGL.so.1").symlink_to(egl)
    monkeypatch.setattr(controller.impl, "NVIDIA_GRAPHICS_ROOT", graphics)
    monkeypatch.setattr(controller.impl, "EGL_FRONTEND", egl)
    monkeypatch.setattr(controller.impl, "EGL_FRONTEND_BYTES", egl.stat().st_size)
    frozen_ld = os.pathsep.join((str(driver), "/usr/local/cuda-12.8/lib64"))
    reservation = {"request": {"Envs": {"FROZEN_SENTINEL": "exact", "LD_LIBRARY_PATH": frozen_ld}}}
    monkeypatch.setenv("FROZEN_SENTINEL", "exact")
    monkeypatch.setenv("FASTWAM_GL_SHIM_ROOT", str(shim))
    monkeypatch.setenv("LD_LIBRARY_PATH", os.pathsep.join((str(shim), frozen_ld)))
    monkeypatch.delenv("SAPIEN_VULKAN_LIBRARY_PATH", raising=False)
    return reservation, shim


def test_worker_environment_accepts_only_one_private_egl_prefix(tmp_path, monkeypatch):
    reservation, _ = _worker_fixture(tmp_path, monkeypatch)
    controller.validate_worker_environment(reservation)


def test_dependency_probe_does_not_repeat_runtime_loader_namespace(tmp_path, monkeypatch):
    _, shim = _worker_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(controller.impl, "worker_pythonpath", lambda: "frozen-pythonpath")
    env = controller.worker_dependency_env(shim)
    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [
        str(shim.resolve(strict=True)),
        str((controller.impl.NVIDIA_GRAPHICS_ROOT / "driver-lib").resolve(strict=True)),
        "/usr/local/cuda-12.8/lib64",
    ]
    assert env["PYTHONPATH"] == "frozen-pythonpath"
    assert "SAPIEN_VULKAN_LIBRARY_PATH" not in env


def test_worker_environment_rejects_loader_suffix(tmp_path, monkeypatch):
    reservation, _ = _worker_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("LD_LIBRARY_PATH", os.environ["LD_LIBRARY_PATH"] + ":/unexpected")
    with pytest.raises(controller.ContractError, match="must be exactly"):
        controller.validate_worker_environment(reservation)


def test_worker_environment_keeps_all_non_loader_fields_exact(tmp_path, monkeypatch):
    reservation, _ = _worker_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("FROZEN_SENTINEL", "drifted")
    with pytest.raises(controller.ContractError, match="FROZEN_SENTINEL"):
        controller.validate_worker_environment(reservation)


def test_runtime_clears_inherited_namespaces_before_frozen_r17_runtime():
    runtime = (R18 / "runtime.sh").read_text(encoding="utf-8")
    assert "unset PYTHONPATH LD_LIBRARY_PATH SAPIEN_VULKAN_LIBRARY_PATH FASTWAM_GL_SHIM_ROOT" in runtime
    assert "R18-20260814" in runtime
    assert "fastwam-gau0-placefood-r18.XXXXXXXX" in runtime
    assert "R17-20260814/runtime.sh" in runtime


def test_r18_entrypoints_bind_identity_lock_and_terminal_aggregator():
    wrapper = (R18 / "submit_from_ssh970.sh").read_text(encoding="utf-8")
    assert "R18-20260814" in wrapper
    assert "gau0-placefood-same8-r18-controller.lock" in wrapper
    aggregator = _load("gau0_placefood_r18_aggregator", R18 / "aggregate_results.py")
    assert aggregator.ARMS == ("gau1_stats", "gau0_native_stats")
    assert aggregator.validate_arm is aggregator.impl.validate_arm
    assert aggregator.main is aggregator.impl.main


def test_controller_cli_fails_closed_without_reservation():
    completed = subprocess.run(
        [sys.executable, "-B", str(R18 / "controller.py"), "worker-preflight"],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 2
    assert "GAU0_CONTROLLER_FATAL" in completed.stderr


def test_readme_preserves_r17_failure_and_scientific_boundary():
    readme = (R18 / "README.md").read_text(encoding="utf-8")
    assert "dlc7lmi2y16cjuuk" in readme
    assert "failed before" in readme
    assert "episode 0" in readme
    assert "any suffix" in readme
    assert "exact same eight" in readme
    assert "not by itself a pure causal" in readme
