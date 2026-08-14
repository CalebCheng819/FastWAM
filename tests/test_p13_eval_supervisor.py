from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/supervise_p13_training_and_eval.py"
SPEC = importlib.util.spec_from_file_location("p13_eval_supervisor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class P13EvalSupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_cache(self) -> Path:
        root = self.root / "cache"
        root.mkdir()
        frames = 4
        frame_bytes = frames * 13 * 60 * 80 * 2
        frame_path = root / "frames.f16"
        frame_path.write_bytes(b"\0" * frame_bytes)
        manifest = {
            "schema_name": "fastwam.metric-geometry-cache",
            "version": 1,
            "provenance_mode": "stat_cmp",
            "dtype": "float16",
            "byte_order": "little",
            "frame_shape": [13, 60, 80],
            "data": {
                "path": "frames.f16",
                "frames": frames,
                "bytes": frame_bytes,
                "mtime_ns": frame_path.stat().st_mtime_ns,
            },
            "selection": dict(MODULE.EXPECTED_SELECTION),
            "metric_geometry": dict(MODULE.EXPECTED_GEOMETRY),
            "counts": {
                "frames": frames,
                "windows": 2,
                "train_windows": 1,
                "val_windows": 1,
            },
            "entries": [
                {
                    "offset": index,
                    "source_path": "PlaceFood-rf/trajectory.h5",
                    "trajectory": "traj_0",
                    "timestep": index,
                    "agent_name": f"agent-{index % 2}",
                }
                for index in range(frames)
            ],
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "COMPLETE").write_text("complete\n", encoding="utf-8")
        (root / "stat-cmp.allowlist").write_text(
            MODULE.EXPECTED_ALLOWLIST, encoding="utf-8"
        )
        return root

    def test_metric_cache_contract(self) -> None:
        root = self.make_cache()
        self.assertEqual(MODULE.validate_metric_cache(root)["frames"], 4)
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["metric_geometry"]["coordinate_frame"] = "camera"
        (root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, "geometry contract"):
            MODULE.validate_metric_cache(root)

    def test_training_gate_waits_then_accepts_complete_checkpoint(self) -> None:
        output = self.root / "training"
        checkpoint = output / "checkpoints/weights/step_001000.pt"
        receipt = {
            "job_id": "dlc-example",
            "selected_cache_root": "/cache",
        }
        job = {
            "JobId": "dlc-example",
            "Status": "Running",
            "Envs": {
                "FASTWAM_POSE_FOCUS_OUTPUT_DIR": str(output),
                "FASTWAM_POSE_FOCUS_METRIC_SOURCE_ROOT": "/cache",
            },
        }
        state, _ = MODULE.validate_training_gate(job, receipt, output, checkpoint)
        self.assertEqual(state, "WAITING_FOR_TRAINING")

        job["Status"] = "Succeeded"
        state, _ = MODULE.validate_training_gate(job, receipt, output, checkpoint)
        self.assertEqual(state, "WAITING_FOR_CHECKPOINT")
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"weights")
        Path(f"{checkpoint}.COMPLETE").write_text("complete\n")
        state, summary = MODULE.validate_training_gate(job, receipt, output, checkpoint)
        self.assertEqual(state, "TRAINING_READY")
        self.assertEqual(summary["checkpoint"]["bytes"], 7)

    def test_training_gate_rejects_failed_job(self) -> None:
        output = self.root / "training"
        receipt = {"job_id": "dlc-example", "selected_cache_root": "/cache"}
        job = {
            "JobId": "dlc-example",
            "Status": "Failed",
            "Envs": {
                "FASTWAM_POSE_FOCUS_OUTPUT_DIR": str(output),
                "FASTWAM_POSE_FOCUS_METRIC_SOURCE_ROOT": "/cache",
            },
        }
        with self.assertRaisesRegex(RuntimeError, "terminated unsuccessfully"):
            MODULE.validate_training_gate(
                job, receipt, output, output / "checkpoints/weights/step_001000.pt"
            )

    def test_gpu_selection_requires_no_process_and_low_memory(self) -> None:
        inventory = MODULE.parse_gpu_inventory(
            "0, GPU-a, 512, 0\n1, GPU-b, 1500, 0\n2, GPU-c, 100, 99\n3, GPU-d, 200, 0\n"
        )
        apps = MODULE.parse_compute_apps("GPU-c, 1234, 900\n")
        for row in inventory:
            row["compute_apps"] = apps.get(row["uuid"], [])
        self.assertEqual(MODULE.select_free_gpus(inventory, 2, 1024), [0, 3])
        self.assertEqual(MODULE.select_free_gpus(inventory, 3, 1024), [])

    def test_teacher_and_closedloop_terminal_contracts(self) -> None:
        cache = self.make_cache()
        teacher = self.root / "teacher"
        teacher.mkdir()
        (teacher / "terminal.status").write_text("SUCCEEDED\n")
        (teacher / "TERMINAL_STATUS.json").write_text(
            json.dumps({"status": "SUCCEEDED", "return_code": 0})
        )
        (teacher / "comparison.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETED",
                    "metric_cache_root": str(cache),
                    "states": 263,
                    "valid_pairs_h1": 263,
                    "valid_pairs_h5": 1305,
                }
            )
        )
        self.assertEqual(
            MODULE.validate_teacher_output(teacher, cache)["states"], 263
        )

        closedloop = self.root / "closedloop"
        closedloop.mkdir()
        (closedloop / "aggregate.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "expected_runs": 8,
                    "operational_runs": 8,
                    "success_count": 3,
                }
            )
        )
        self.assertEqual(
            MODULE.validate_closedloop_output(closedloop)["success_count"], 3
        )


if __name__ == "__main__":
    unittest.main()
