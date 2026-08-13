from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = (
    ROOT
    / ".research-workflow"
    / "experiments"
    / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R9-20260814"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


controller = _load("gau0_placefood_r9_controller", EXPERIMENT / "controller.py")


def test_r9_identity_and_runtime_env_freeze_complete_glvnd_frontends():
    assert controller.EXPERIMENT_ID.endswith("R9-20260814")
    assert controller.RUN_ID == "fastwam-gau0-placefood-same8-r9-20260814"
    assert controller.DISPLAY_NAME == "fw-gau0-placefood-same8-r9"
    assert controller.request_body("a" * 40)["Priority"] == 7

    env = controller.runtime_env("a" * 40)
    bindings = {
        "FASTWAM_EGL_FRONTEND": (controller.EGL_FRONTEND, controller.EGL_FRONTEND_BYTES),
        "FASTWAM_GL_FRONTEND": (controller.GL_FRONTEND, controller.GL_FRONTEND_BYTES),
        "FASTWAM_GLES1_FRONTEND": (controller.GLES1_FRONTEND, controller.GLES1_FRONTEND_BYTES),
        "FASTWAM_GLES2_FRONTEND": (controller.GLES2_FRONTEND, controller.GLES2_FRONTEND_BYTES),
        "FASTWAM_OPENGL_FRONTEND": (controller.OPENGL_FRONTEND, controller.OPENGL_FRONTEND_BYTES),
        "FASTWAM_GLX_FRONTEND": (controller.GLX_FRONTEND, controller.GLX_FRONTEND_BYTES),
    }
    for name, (path, size) in bindings.items():
        assert env[name] == str(path)
        assert env[f"{name}_SIZE_BYTES"] == str(size)


def test_create_glvnd_shim_has_exact_soname_and_alias_mapping(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    targets = {}
    for index, soname in enumerate(
        ("libEGL.so.1", "libGL.so.1", "libGLESv1_CM.so.1", "libGLESv2.so.2", "libOpenGL.so.0", "libGLX.so.0")
    ):
        target = source / f"target-{index}"
        target.write_bytes(bytes([index]))
        targets[soname] = target
    aliases = {
        "libEGL.so": "libEGL.so.1",
        "libGL.so": "libGL.so.1",
        "libGLESv1_CM.so": "libGLESv1_CM.so.1",
        "libGLESv2.so": "libGLESv2.so.2",
    }
    monkeypatch.setattr(controller, "GLVND_SHIM_TARGETS", targets)
    monkeypatch.setattr(controller, "GLVND_SHIM_ALIASES", aliases)

    shim = tmp_path / "shim"
    shim.mkdir()
    controller.create_glvnd_shim(shim)

    assert {item.name for item in shim.iterdir()} == set(targets) | set(aliases)
    for soname, target in targets.items():
        assert (shim / soname).is_symlink()
        assert os.readlink(shim / soname) == str(target)
        assert (shim / soname).resolve(strict=True) == target.resolve(strict=True)
    for alias, soname in aliases.items():
        assert os.readlink(shim / alias) == soname
        assert (shim / alias).resolve(strict=True) == targets[soname].resolve(strict=True)


def test_worker_dependency_preflight_uses_private_shim_and_environment_import_gate(monkeypatch):
    captured = {}

    def fake_create(shim_dir: Path) -> None:
        captured["shim_dir"] = shim_dir

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)

        class Result:
            returncode = 0
            stdout = "GAU0_WORKER_DEPENDENCY_PREFLIGHT_PASS\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(controller, "validate_python", lambda: None)
    monkeypatch.setattr(controller, "require_dir", lambda path: None)
    monkeypatch.setattr(controller, "create_glvnd_shim", fake_create)
    monkeypatch.setattr(controller.subprocess, "run", fake_run)
    controller.validate_worker_dependencies()

    program = captured["argv"][3]
    assert program.index("_preflight_environment_imports(robofactory)") < program.index("import mani_skill")
    assert program.index("_preflight_environment_imports(robofactory)") < program.index("import tasks.place_food")
    assert "worker module provenance mismatch" in program
    shim_dir = captured["shim_dir"]
    assert Path(captured["env"]["LD_LIBRARY_PATH"].split(os.pathsep)[0]) == shim_dir
    assert captured["env"]["LD_LIBRARY_PATH"].split(os.pathsep)[1:3] == [
        str(controller.NVIDIA_GRAPHICS_ROOT / "lib"),
        str(controller.NVIDIA_GRAPHICS_ROOT / "driver-lib"),
    ]


def test_runtime_freezes_complete_glvnd_shim_and_namespace_preflight():
    runtime = (EXPERIMENT / "runtime.sh").read_text(encoding="utf-8")
    expected_links = {
        "FASTWAM_EGL_FRONTEND": "libEGL.so.1",
        "FASTWAM_GL_FRONTEND": "libGL.so.1",
        "FASTWAM_GLES1_FRONTEND": "libGLESv1_CM.so.1",
        "FASTWAM_GLES2_FRONTEND": "libGLESv2.so.2",
        "FASTWAM_OPENGL_FRONTEND": "libOpenGL.so.0",
        "FASTWAM_GLX_FRONTEND": "libGLX.so.0",
    }
    for variable, soname in expected_links.items():
        assert f'ln -s -- "${{{variable}}}" "${{scratch_root}}/glvnd-runtime/{soname}"' in runtime
        assert f'"{soname}": Path(os.environ["{variable}"]).resolve(strict=True)' in runtime
    assert 'export LD_LIBRARY_PATH="${scratch_root}/glvnd-runtime:${FASTWAM_NVIDIA_GRAPHICS_ROOT}/lib:${FASTWAM_NVIDIA_GRAPHICS_ROOT}/driver-lib' in runtime
    assert "GAU0_GLVND_RUNTIME_PREFLIGHT_PASS" in runtime
    assert runtime.index("_preflight_environment_imports(Path(os.environ[\"FASTWAM_ROBOFACTORY_ROOT\"]))") < runtime.index('"${FASTWAM_PYTHON}" -B "${controller}" worker-preflight')
    assert "environment preflight place_food mismatch" in runtime
    assert "environment preflight scenes mismatch" in runtime


def test_r9_readme_preserves_r8_and_records_exact_failure_and_fix():
    readme = (EXPERIMENT / "README.md").read_text(encoding="utf-8")
    assert "R9 is a new, isolated identity after R8" in readme
    assert "`_glapi_tls_Current`" in readme
    assert "complete GLVND front-end namespace" in readme
    assert "all six front-end SONAMEs" in readme
    assert "RoboFactory `tasks` and `utils`" in readme
    assert "rejects already-loaded foreign modules" in readme
    assert "R1 through R8" in readme


def test_dependency_preflight_precedes_prepare_submit_and_worker_boundaries():
    prepare_source = inspect.getsource(controller.prepare)
    assert prepare_source.index("validate_worker_dependencies()") < prepare_source.index("write_json_exclusive(")

    submit_source = inspect.getsource(controller.submit)
    assert submit_source.index("validate_worker_dependencies()") < submit_source.index("load_sdk()")
    assert submit_source.index("validate_worker_dependencies()") < submit_source.index("write_json_exclusive(LATCH_PATH")

    worker_source = inspect.getsource(controller.worker_preflight)
    assert worker_source.index("validate_worker_dependencies()") < worker_source.index('print("GAU0_WORKER_PREFLIGHT_PASS")')
