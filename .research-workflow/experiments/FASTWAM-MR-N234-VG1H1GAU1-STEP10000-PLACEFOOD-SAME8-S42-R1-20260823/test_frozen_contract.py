#!/usr/bin/env python3
"""Regression tests for the frozen step-10k DLC evaluation contract."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
RENDERER = HERE / "render_job.py"
AGGREGATOR = HERE / "aggregate_results.py"
SOURCE_COMMIT = "1" * 40
CHECKPOINT = (
    "/oss-chengjuntao/artifacts/"
    "fastwam-n234-vg1h1gau1-cont50k-s42-24g-r1-20260822/"
    "checkpoints/weights/step_010000.pt"
)
ENVIRONMENT_SEEDS = (333183, 333327, 333225, 333180, 333251, 333130, 333167, 333234)


class FrozenContractTests(unittest.TestCase):
    def test_renderer_pins_one_worker_eight_gpu_priority_seven(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dry-run.json"
            subprocess.run(
                [
                    sys.executable,
                    str(RENDERER),
                    "--source-commit",
                    SOURCE_COMMIT,
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            document = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(document["dry_run"])
        self.assertTrue(document["submission_not_performed"])
        request = document["request"]
        self.assertEqual(request["Priority"], 7)
        self.assertEqual(request["SuccessPolicy"], "AllWorkers")
        self.assertEqual(len(request["JobSpecs"]), 1)
        worker = request["JobSpecs"][0]
        self.assertEqual(worker["Type"], "Worker")
        self.assertEqual(worker["PodCount"], 1)
        self.assertEqual(worker["ResourceConfig"]["GPU"], "8")
        self.assertEqual(worker["RestartPolicy"], "Never")
        self.assertEqual(
            request["DataSources"],
            [
                {
                    "DataSourceId": "d-a5mu77ymwjio71dkmw",
                    "MountPath": "/cpfs/user/chengjuntao",
                    "MountAccess": "RO",
                },
                {
                    "DataSourceId": "d-n7rly4fll0q2z6v91h",
                    "MountPath": "/oss-chengjuntao",
                    "MountAccess": "RW",
                },
            ],
        )
        envs = request["Envs"]
        self.assertEqual(envs["FASTWAM_CHECKPOINT"], CHECKPOINT)
        self.assertEqual(envs["FASTWAM_CHECKPOINT_SIZE_BYTES"], "12047213657")
        self.assertEqual(envs["FASTWAM_SOURCE_COMMIT"], SOURCE_COMMIT)
        self.assertEqual(envs["FASTWAM_ATTEMPT_ID"], "attempt-002")
        self.assertEqual(document["run_id"], "fastwam-gau1-step10k-placefood-same8-r2-20260823")
        self.assertEqual(
            envs["FASTWAM_OUTPUT_ROOT"],
            "/oss-chengjuntao/artifacts/fastwam-gau1-step10k-placefood-same8-eval-20260823-r2",
        )
        launcher = base64.b64decode(envs["FASTWAM_LAUNCHER_B64"]).decode("utf-8")
        self.assertIn("runtime.sh", launcher)
        self.assertIn("output root already exists", (HERE / "runtime.sh").read_text())
        runtime = (HERE / "runtime.sh").read_text()
        self.assertIn('export PYTHONPATH="${source_src}"', runtime)
        self.assertNotIn('PYTHONPATH="${FASTWAM_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"', runtime)
        self.assertIn("create_multi_robot_fastwam", runtime)
        self.assertIn("STEP10K_EVAL_SOURCE_GATE=PASS", runtime)
        self.assertEqual(document["launcher_payload_base64"], envs["FASTWAM_LAUNCHER_B64"])
        self.assertEqual(request["UserCommand"].count("FASTWAM_LAUNCHER_B64"), 1)

    def _write_shards(self, root: Path) -> None:
        for index, environment_seed in enumerate(ENVIRONMENT_SEEDS):
            shard = root / f"episode-{index:02d}"
            shard.mkdir()
            record = {
                "status": "completed",
                "mode": "fastwam",
                "task_name": "PlaceFood-rf",
                "task_index": index,
                "panel_index": index,
                "environment_seed": environment_seed,
                "policy_seed": 10000 + index,
                "success": index in {1, 6},
                "steps": 300,
                "policy_queries": 60,
                "action_bound_violations": 0,
            }
            manifest = {
                "schema_version": "fastwam-robofactory-eval-run-v2",
                "status": "terminal",
                "mode": "fastwam",
                "task_name": "PlaceFood-rf",
                "episode_start": index,
                "num_episodes": 1,
                "max_steps_override": 300,
                "exec_horizon": 5,
                "policy_seed_base": 10000,
                "eval_code_commit": SOURCE_COMMIT,
                "integrity_mode": "metadata_no_hash",
                "policy": {
                    "checkpoint_path": CHECKPOINT,
                    "checkpoint_size_bytes": 12_047_213_657,
                    "checkpoint_sha256": None,
                    "integrity_mode": "metadata_no_hash",
                    "action_horizon": 32,
                    "num_inference_steps": 20,
                    "gaussian_conditioning": True,
                    "teacher": {"checkpoint_integrity_mode": "metadata_no_hash"},
                },
            }
            summary = {
                "schema_version": "fastwam-robofactory-eval-summary-v2",
                "status": "PASS",
                "infrastructure_errors": 0,
                "episodes_requested": 1,
                "episodes_recorded": 1,
                "episodes_completed": 1,
            }
            (shard / "episodes.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            (shard / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (shard / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    def test_aggregator_publishes_only_complete_frozen_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shards = root / "shards"
            shards.mkdir()
            self._write_shards(shards)
            output = root / "published"
            subprocess.run(
                [
                    sys.executable,
                    str(AGGREGATOR),
                    "--temp-root",
                    str(shards),
                    "--output-root",
                    str(output),
                    "--source-commit",
                    SOURCE_COMMIT,
                    "--job-id",
                    "dlc-test",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            aggregate = json.loads((output / "aggregate.json").read_text(encoding="utf-8"))
            complete = json.loads((output / "COMPLETE.json").read_text(encoding="utf-8"))
            self.assertEqual(aggregate["successes"], 2)
            self.assertEqual(aggregate["closed_loop_success_rate"], 0.25)
            self.assertEqual(aggregate["job_id"], "dlc-test")
            self.assertEqual(complete, {"status": "PASS", "successes": 2, "episodes": 8})

    def test_aggregator_rejects_seed_drift_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shards = root / "shards"
            shards.mkdir()
            self._write_shards(shards)
            episode = shards / "episode-03" / "episodes.jsonl"
            record = json.loads(episode.read_text(encoding="utf-8"))
            record["environment_seed"] += 1
            episode.write_text(json.dumps(record) + "\n", encoding="utf-8")
            output = root / "published"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(AGGREGATOR),
                    "--temp-root",
                    str(shards),
                    "--output-root",
                    str(output),
                    "--source-commit",
                    SOURCE_COMMIT,
                    "--job-id",
                    "dlc-test",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
