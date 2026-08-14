import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "supervise_p13_cache_and_submit.py"
)
SPEC = importlib.util.spec_from_file_location("p13_cache_supervisor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MetricCacheValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "cache"
        self.root.mkdir()
        frames = 4
        frame_shape = [13, 60, 80]
        frame_bytes = frames * 13 * 60 * 80 * 2
        frames_path = self.root / "frames.f16"
        frames_path.write_bytes(b"\0" * frame_bytes)
        manifest = {
            "schema_name": "fastwam.metric-geometry-cache",
            "version": 1,
            "created_at": "2026-08-15T00:00:00Z",
            "provenance_mode": "stat_cmp",
            "dtype": "float16",
            "byte_order": "little",
            "frame_shape": frame_shape,
            "data": {
                "path": "frames.f16",
                "frames": frames,
                "bytes": frame_bytes,
                "mtime_ns": frames_path.stat().st_mtime_ns,
            },
            "selection": {
                "task_name": "PlaceFood-rf",
                "required_agent_count": 2,
                "action_horizon": 32,
                "split_seed": 42,
                "val_set_proportion": 0.1,
                "train_window_stride": 16,
                "val_window_stride": 32,
                "limit_trajectories": None,
            },
            "metric_geometry": {
                "source": "maniskill_calibrated_depth",
                "coordinate_frame": "world",
                "output_size": [60, 80],
                "channels": "xyz_mean_covariance_row_major_valid",
                "render_backend": "gpu",
            },
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
        (self.root / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (self.root / "COMPLETE").write_text("complete\n", encoding="utf-8")
        (self.root / "stat-cmp.allowlist").write_text(
            MODULE.EXPECTED_ALLOWLIST, encoding="utf-8"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_valid_cache(self):
        summary = MODULE.validate_metric_cache(self.root)
        self.assertEqual(summary["frames"], 4)
        self.assertEqual(summary["frame_shape"], [13, 60, 80])

    def test_rejects_truncated_frame_file(self):
        with (self.root / "frames.f16").open("r+b") as stream:
            stream.truncate(10)
        with self.assertRaisesRegex(RuntimeError, "byte count mismatch"):
            MODULE.validate_metric_cache(self.root)

    def test_rejects_wrong_window_contract(self):
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["selection"]["train_window_stride"] = 32
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "selection contract mismatch"):
            MODULE.validate_metric_cache(self.root)

    def test_rejects_symlinked_complete_marker(self):
        complete = self.root / "COMPLETE"
        target = self.root / "complete.target"
        complete.rename(target)
        complete.symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, "missing or non-regular"):
            MODULE.validate_metric_cache(self.root)


class DuplicateDetectionTest(unittest.TestCase):
    def test_matches_each_frozen_identity(self):
        request = {
            "DisplayName": "display",
            "Envs": {
                "RUN_ID": "run",
                "FASTWAM_POSE_FOCUS_OUTPUT_DIR": "/output",
            },
        }
        jobs = [
            {"JobId": "a", "DisplayName": "display", "Envs": {}},
            {"JobId": "b", "DisplayName": "x", "Envs": {"RUN_ID": "run"}},
            {
                "JobId": "c",
                "DisplayName": "y",
                "Envs": {"FASTWAM_POSE_FOCUS_OUTPUT_DIR": "/output"},
            },
            {"JobId": "d", "DisplayName": "z", "Envs": {}},
        ]
        matches = MODULE.duplicate_jobs(jobs, request)
        self.assertEqual([item["JobId"] for item in matches], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
