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
SUBMITTER = EXPERIMENT / "submit-cache-r9-r25-egl-dedicated.py"
REQUEST = EXPERIMENT / "cache-create-job-r9-r25-egl-dedicated.json"
WORKER = EXPERIMENT / "cache-worker-r9-r25-egl-dedicated.sh"

SPEC = importlib.util.spec_from_file_location("p13_cache_r9", SUBMITTER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class P13CacheR9SubmissionTest(unittest.TestCase):
    def test_request_is_priority_seven_and_uses_r25_graphics_contract(self):
        body = json.loads(REQUEST.read_text(encoding="utf-8"))
        MODULE.validate_request_map(body)
        self.assertEqual(body["Priority"], 7)
        self.assertNotIn("OversoldType", body["Settings"])
        self.assertEqual(body["JobSpecs"][0]["ElasticSpotSpecs"], [])
        self.assertEqual(
            body["Settings"]["Tags"]["preflight_order"],
            "r25-egl-imports-before-real-environment",
        )
        self.assertEqual(
            body["Envs"]["FASTWAM_P13_PYTHON_EXTRA_ROOT"],
            "/cpfs/user/chengjuntao/venvs/fastwam-gau0-eval-r7-py310-extra-20260813",
        )

    def test_worker_binds_gpu_zero_and_new_r9_runtime(self):
        source = WORKER.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(WORKER)], check=True)
        self.assertIn("export CUDA_VISIBLE_DEVICES=0", source)
        self.assertIn("fastwam-p13-runtime-20260817-r9-r25-egl", source)
        self.assertNotIn("r8-complete-glvnd", source)

    def test_new_identity_and_fix_commit_are_consistent(self):
        body = json.loads(REQUEST.read_text(encoding="utf-8"))
        self.assertEqual(body["DisplayName"], MODULE.DISPLAY_NAME)
        self.assertEqual(body["Envs"]["RUN_ID"], MODULE.RUN_ID)
        self.assertEqual(body["Envs"]["FASTWAM_P13_CACHE_OUTPUT_ROOT"], MODULE.OUTPUT_ROOT)
        self.assertEqual(body["Envs"]["FASTWAM_P13_CODE_REVISION"], MODULE.SOURCE_REVISION)
        self.assertEqual(
            MODULE.SOURCE_REVISION,
            "f8ac674b27efcb2e1b937c2a8e2b121045321409",
        )

    def test_permanent_latch_precedes_single_create_job(self):
        source = SUBMITTER.read_text(encoding="utf-8")
        self.assertLess(
            source.index("write_exclusive(args.latch, latch)"),
            source.index("client.create_job(request)"),
        )
        self.assertEqual(source.count("client.create_job(request)"), 1)
        self.assertIn("create_job_call_count", source)

    def test_duplicate_detection_covers_new_frozen_identity(self):
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
