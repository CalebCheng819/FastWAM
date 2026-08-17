import importlib.util
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = (
    ROOT
    / ".research-workflow"
    / "experiments"
    / "FASTWAM-MR-N2-PLACEFOOD-METRIC-P13-S42-8G-R1-20260815"
)
SUBMITTER = EXPERIMENT / "submit-cache-r11-r25-egl-package-root-dedicated.py"
REQUEST = EXPERIMENT / "cache-create-job-r11-r25-egl-package-root-dedicated.json"
WORKER = EXPERIMENT / "cache-worker-r11-r25-egl-package-root-dedicated.sh"

SPEC = importlib.util.spec_from_file_location("p13_cache_r11", SUBMITTER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class P13CacheR11SubmissionTest(unittest.TestCase):
    def test_request_is_priority_seven_and_uses_new_identity(self):
        body = json.loads(REQUEST.read_text(encoding="utf-8"))
        MODULE.validate_request_map(body)
        self.assertEqual(body["Priority"], 7)
        self.assertEqual(body["DisplayName"], MODULE.DISPLAY_NAME)
        self.assertEqual(body["Envs"]["RUN_ID"], MODULE.RUN_ID)
        self.assertEqual(body["Envs"]["FASTWAM_P13_CACHE_OUTPUT_ROOT"], MODULE.OUTPUT_ROOT)
        self.assertNotIn("r10-r25-egl-dedicated-20260817", MODULE.DISPLAY_NAME)

    def test_source_revision_contains_package_parent_fix(self):
        body = json.loads(REQUEST.read_text(encoding="utf-8"))
        self.assertEqual(body["Envs"]["FASTWAM_P13_CODE_REVISION"], MODULE.SOURCE_REVISION)
        self.assertEqual(MODULE.SOURCE_REVISION, "70aabc72654da6c3a1ac88e8780b58805f75443a")
        subprocess.run(
            ["git", "cat-file", "-e", f"{MODULE.SOURCE_REVISION}^{{commit}}"],
            cwd=ROOT,
            check=True,
        )
        shared = (ROOT / "scripts" / "run_p13_metric_cache_dlc.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("ROBOFACTORY_PACKAGE_PARENT", shared)
        self.assertIn("import robofactory.utils.scenes", shared)

    def test_worker_binds_gpu_zero_and_r11_runtime(self):
        source = WORKER.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(WORKER)], check=True)
        self.assertIn("export CUDA_VISIBLE_DEVICES=0", source)
        self.assertIn("fastwam-p13-runtime-20260817-r11-r25-egl-package-root", source)
        self.assertNotIn("runtime-20260817-r10-r25-egl/", source)

    def test_permanent_latch_precedes_single_create_job(self):
        source = SUBMITTER.read_text(encoding="utf-8")
        self.assertLess(
            source.index("write_exclusive(args.latch, latch)"),
            source.index("client.create_job(request)"),
        )
        self.assertEqual(source.count("client.create_job(request)"), 1)

    def test_duplicate_detection_covers_r11_frozen_identity(self):
        matches = MODULE.duplicate_jobs(
            [
                {"JobId": "a", "DisplayName": MODULE.DISPLAY_NAME},
                {"JobId": "b", "DisplayName": "x", "Envs": {"RUN_ID": MODULE.RUN_ID}},
                {
                    "JobId": "c",
                    "DisplayName": "y",
                    "Envs": {"FASTWAM_P13_CACHE_OUTPUT_ROOT": MODULE.OUTPUT_ROOT},
                },
                {"JobId": "d", "DisplayName": "z"},
            ]
        )
        self.assertEqual([item["job_id"] for item in matches], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
