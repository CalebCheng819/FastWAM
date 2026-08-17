import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "run_p13_metric_cache_dlc.sh"


class P13CacheR6GraphicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = WORKER.read_text(encoding="utf-8")

    def test_worker_has_valid_bash_syntax(self):
        subprocess.run(["bash", "-n", str(WORKER)], check=True)

    def test_loader_and_all_graphics_profiles_are_frozen(self):
        self.assertIn("FASTWAM_P13_VULKAN_LOADER", self.source)
        for profile in (
            "cpfs_manifest_headless",
            "provider_native_headless",
            "provider_clean_headless",
            "system_default_headless",
            "system_discovered_headless",
            "system_manifest_headless",
            "system_discovered_sapien_loader",
        ):
            self.assertIn(profile, self.source)

    def test_probe_constructs_and_closes_the_real_environment(self):
        self.assertIn(
            'environment = _build_environment(root, "PlaceFood-rf")', self.source
        )
        self.assertIn("environment.close()", self.source)
        self.assertIn("timeout --signal=TERM --kill-after=30s 180s", self.source)
        self.assertIn("env CUDA_VISIBLE_DEVICES=0", self.source)

    def test_graphics_sensitive_imports_wait_for_profile_contract(self):
        preflight_start = self.source.index('"${PYTHON_BIN}" - <<\'PY\'')
        preflight_end = self.source.index("\nPY\n", preflight_start)
        preflight = self.source[preflight_start:preflight_end]
        for module in ("mani_skill", "robofactory", "sapien"):
            self.assertNotIn(f'"{module}"', preflight)

        loop = self.source.index('for profile in "${profiles[@]}"; do')
        applied = self.source.index('apply_graphics_profile "${profile}"', loop)
        egl_gate = self.source.index("ensure_sapien_egl_contract", applied)
        probe = self.source.index(
            "timeout --signal=TERM --kill-after=30s 180s", egl_gate
        )
        self.assertLess(applied, egl_gate)
        self.assertLess(egl_gate, probe)

    def test_selected_profile_and_egl_gate_precede_builder(self):
        selected = self.source.index('[[ -n "${selected_profile}" ]]')
        reapplied = self.source.index('apply_graphics_profile "${selected_profile}"')
        egl_gate = self.source.index("ensure_sapien_egl_contract", reapplied)
        builder = self.source.index(
            '"${PYTHON_BIN}" "${LOCAL_REPO}/scripts/build_robofactory_metric_geometry_cache.py"'
        )
        self.assertLess(selected, reapplied)
        self.assertLess(reapplied, egl_gate)
        self.assertLess(egl_gate, builder)

    def test_all_probe_failures_close_before_cache_mutation(self):
        failure = self.source.index(
            "die 'no GPU graphics profile could construct and close PlaceFood-rf'"
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
