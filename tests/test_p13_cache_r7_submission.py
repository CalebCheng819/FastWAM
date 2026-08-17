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
SUBMITTER = EXPERIMENT / "submit-cache-r7-profile-ordered-dedicated.py"
REQUEST = EXPERIMENT / "cache-create-job-r7-profile-ordered-dedicated.json"
WORKER = EXPERIMENT / "cache-worker-r7-profile-ordered-dedicated.sh"

SPEC = importlib.util.spec_from_file_location("p13_cache_r7", SUBMITTER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class P13CacheR7SubmissionTest(unittest.TestCase):
    def test_request_is_priority_seven_dedicated_and_profile_ordered(self):
        body = json.loads(REQUEST.read_text(encoding="utf-8"))
        MODULE.validate_request_map(body)
        self.assertEqual(body["Priority"], 7)
        self.assertNotIn("OversoldType", body["Settings"])
        self.assertNotIn("SpotStrategy", body)
        self.assertEqual(body["JobSpecs"][0]["ElasticSpotSpecs"], [])
        self.assertEqual(
            body["Settings"]["Tags"]["preflight_order"],
            "graphics-profile-before-sensitive-import",
        )

    def test_worker_binds_gpu_zero_and_r7_runtime(self):
        source = WORKER.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(WORKER)], check=True)
        self.assertIn("export CUDA_VISIBLE_DEVICES=0", source)
        self.assertIn("fastwam-p13-runtime-20260817-r7", source)

    def test_permanent_latch_precedes_create_job(self):
        source = SUBMITTER.read_text(encoding="utf-8")
        self.assertLess(
            source.index("write_exclusive(args.latch, latch)"),
            source.index("client.create_job(request)"),
        )
        self.assertIn("create_job_call_count", source)

    def test_duplicate_detection_covers_frozen_identity(self):
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

    def test_r7_source_revision_is_profile_order_fix(self):
        body = json.loads(REQUEST.read_text(encoding="utf-8"))
        self.assertEqual(body["Envs"]["FASTWAM_P13_CODE_REVISION"], MODULE.SOURCE_REVISION)
        self.assertEqual(
            MODULE.SOURCE_REVISION,
            "6b02e6518e0d7d718d5c5a060accd6386fea548e",
        )


if __name__ == "__main__":
    unittest.main()
