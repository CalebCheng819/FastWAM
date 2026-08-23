#!/usr/bin/env python3
"""Static and integration checks for the current four-GPU DSW attempt."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime_dsw.sh"
AGGREGATOR = HERE / "aggregate_results_dsw.py"


class DswAttempt005ContractTests(unittest.TestCase):
    def test_runtime_pins_four_gpu_two_wave_same8_contract(self) -> None:
        runtime = RUNTIME.read_text(encoding="utf-8")
        self.assertIn("expected exactly 4 visible GPUs", runtime)
        self.assertIn("run_wave 0 3", runtime)
        self.assertIn("run_wave 4 7", runtime)
        self.assertIn("--exec-horizon 5", runtime)
        self.assertIn("--action-horizon 32", runtime)
        self.assertIn("--integrity-mode metadata_no_hash", runtime)
        self.assertIn("--gaussian-conditioning", runtime)
        self.assertIn("aggregate_results_dsw.py", runtime)
        self.assertIn("output root already exists", runtime)
        self.assertIn("control root already exists", runtime)
        self.assertIn("status --porcelain=v1 --untracked-files=all", runtime)
        self.assertIn("source checkout is not clean", runtime)
        self.assertIn("[[ \"${FASTWAM_ATTEMPT_ID}\" == 'attempt-005' ]]", runtime)
        self.assertIn("jaxtyping-0.3.7-wadler-0.1.7-py310-attempt005-20260823", runtime)
        self.assertIn('"jaxtyping": "0.3.7"', runtime)
        self.assertIn('"wadler-lindig": "0.1.7"', runtime)
        self.assertIn("STEP10K_DSW_EVAL_DEPENDENCY_GATE=PASS", runtime)
        self.assertIn("from model.noposplat.encoder import get_encoder", runtime)
        self.assertIn("from fastwam.datasets.gaussian_cache.teacher import ExternalPolicyLightningTeacher", runtime)
        self.assertLess(runtime.index("STEP10K_DSW_EVAL_DEPENDENCY_GATE=PASS"), runtime.index('mkdir -m 0700 -- "${FASTWAM_CONTROL_ROOT}"'))
        self.assertNotIn("fastwam-gau1-step10k-placefood-same8-r3-20260823' ]]", runtime)

    def test_dsw_aggregator_changes_only_run_identity(self) -> None:
        spec = importlib.util.spec_from_file_location("attempt005_aggregator", AGGREGATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.base.RUN_ID, "fastwam-gau1-step10k-placefood-same8-dsw4-r5-20260823")
        self.assertEqual(module.base.EXPERIMENT_ID, "FASTWAM-MR-N234-VG1H1GAU1-STEP10000-PLACEFOOD-SAME8-S42-R1-20260823")
        self.assertEqual(module.base.ENVIRONMENT_SEEDS, (333183, 333327, 333225, 333180, 333251, 333130, 333167, 333234))
        self.assertEqual(module.base.POLICY_SEEDS, tuple(range(10000, 10008)))


if __name__ == "__main__":
    unittest.main()
