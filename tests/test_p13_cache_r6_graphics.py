import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "run_p13_metric_cache_dlc.sh"


class P13CacheGraphicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = WORKER.read_text(encoding="utf-8")

    def test_worker_has_valid_bash_syntax(self):
        subprocess.run(["bash", "-n", str(WORKER)], check=True)

    def test_r25_contract_replaces_profile_lottery(self):
        self.assertIn("FASTWAM_P13_VULKAN_LOADER", self.source)
        self.assertIn("apply_r25_graphics_contract", self.source)
        self.assertIn("contract=r25-complete-glvnd", self.source)
        for retired in (
            "profiles=(",
            "build_discovered_loader",
            "cpfs_manifest_headless",
            "provider_native_headless",
            "system_default_headless",
            "system_discovered_sapien_loader",
        ):
            self.assertNotIn(retired, self.source)

    def test_probe_constructs_and_closes_the_real_environment(self):
        self.assertIn(
            'environment = _build_environment(root, "PlaceFood-rf")', self.source
        )
        self.assertIn("environment.close()", self.source)
        self.assertIn("timeout --signal=TERM --kill-after=30s 240s", self.source)
        self.assertIn("env CUDA_VISIBLE_DEVICES=0", self.source)

    def test_complete_glvnd_soname_contract_is_installed(self):
        for soname in (
            "libEGL.so.1",
            "libGL.so.1",
            "libGLESv1_CM.so.1",
            "libGLESv2.so.2",
            "libOpenGL.so.0",
            "libGLX.so.0",
            "libGLdispatch.so.0",
            "libvulkan.so.1",
        ):
            self.assertIn(f"[{soname}]", self.source)
        self.assertIn("/usr/share/glvnd/egl_vendor.d", self.source)
        self.assertIn("/etc/glvnd/egl_vendor.d", self.source)

    def test_r25_contract_checks_native_egl_vulkan_and_pyopengl_before_sapien(self):
        contract = self.source.index("apply_r25_graphics_contract")
        abi_gate = self.source.index("validate_r25_graphics_contract", contract)
        environment_probe = self.source.index(
            "timeout --signal=TERM --kill-after=30s 240s", abi_gate
        )
        self.assertLess(abi_gate, environment_probe)
        self.assertIn('hasattr(egl, "eglQueryString")', self.source)
        self.assertIn('hasattr(vendor, "__egl_Main")', self.source)
        self.assertIn("vkEnumerateInstanceVersion", self.source)
        self.assertIn("from OpenGL import EGL", self.source)
        self.assertIn('getattr(EGL, "eglQueryString", None)', self.source)
        for module in (
            "cv2",
            "mani_skill",
            "sapien",
            "robofactory",
            "tasks.place_food",
            "robofactory.utils.scenes",
        ):
            self.assertIn(f"import {module}", self.source)

    def test_complete_shim_precedes_driver_paths(self):
        self.assertIn(
            'export LD_LIBRARY_PATH="${SCRATCH_ROOT}/graphics-lib:'
            '${DRIVER_ROOT}/lib:${DRIVER_ROOT}/driver-lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"',
            self.source,
        )
        self.assertIn('export FASTWAM_REQUIRE_PROVIDER_NATIVE_GRAPHICS=1', self.source)
        self.assertIn('export PYOPENGL_PLATFORM=egl', self.source)
        self.assertIn('export EGL_PLATFORM=surfaceless', self.source)

    def test_r25_python_extra_precedes_local_runtime_site_packages(self):
        self.assertIn(
            'export PYTHONPATH="${ROBOFACTORY_PACKAGE_PARENT}:${ROBOFACTORY_ROOT}:'
            '${LOCAL_REPO}/src:'
            '${PYTHON_EXTRA_ROOT}:${LOCAL_REPO}/scripts:${RUNTIME_ROOT}/site-packages"',
            self.source,
        )
        self.assertIn(
            'ROBOFACTORY_PACKAGE_PARENT="$(dirname -- "${ROBOFACTORY_ROOT}")"',
            self.source,
        )
        self.assertIn("FASTWAM_P13_PYTHON_EXTRA_ROOT", self.source)

    def test_graphics_sensitive_imports_wait_for_contract(self):
        preflight_start = self.source.index('"${PYTHON_BIN}" - <<\'PY\'')
        preflight_end = self.source.index("\nPY\n", preflight_start)
        preflight = self.source[preflight_start:preflight_end]
        for module in ("mani_skill", "robofactory", "sapien"):
            self.assertNotIn(f'"{module}"', preflight)

        applied = self.source.rindex("apply_r25_graphics_contract")
        egl_gate = self.source.index("validate_r25_graphics_contract", applied)
        probe = self.source.index(
            "timeout --signal=TERM --kill-after=30s 240s", egl_gate
        )
        self.assertLess(applied, egl_gate)
        self.assertLess(egl_gate, probe)

    def test_contract_import_gate_and_environment_probe_precede_builder(self):
        applied = self.source.rindex("apply_r25_graphics_contract")
        gate = self.source.index("validate_r25_graphics_contract", applied)
        probe = self.source.index("timeout --signal=TERM --kill-after=30s 240s", gate)
        builder = self.source.index(
            '"${PYTHON_BIN}" "${LOCAL_REPO}/scripts/build_robofactory_metric_geometry_cache.py"'
        )
        self.assertLess(applied, gate)
        self.assertLess(gate, probe)
        self.assertLess(probe, builder)

    def test_probe_failure_closes_before_cache_mutation(self):
        failure = self.source.index(
            "die 'R25 graphics contract could not construct and close PlaceFood-rf'"
        )
        builder = self.source.index(
            '"${PYTHON_BIN}" "${LOCAL_REPO}/scripts/build_robofactory_metric_geometry_cache.py"'
        )
        publisher = self.source.index(
            '"${LOCAL_REPO}/scripts/publish_metric_geometry_cache.py"'
        )
        self.assertLess(failure, builder)
        self.assertLess(builder, publisher)


if __name__ == "__main__":
    unittest.main()
