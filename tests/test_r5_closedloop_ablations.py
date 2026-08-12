from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.robofactory import run_r5_closedloop_ablations as ablations


class R5ClosedLoopAblationTests(unittest.TestCase):
    def test_rollout_subprocess_gets_absolute_checkout_pythonpath(self) -> None:
        contract = {
            "source_root": "/frozen/fastwam",
            "robofactory_root": "/robofactory",
            "nvidia_driver_lib_dir": "/nvidia/driver-lib",
            "nvidia_vulkan_icd": "/nvidia/nvidia_icd.json",
            "nvidia_egl_vendor_json": "/nvidia/10_nvidia.json",
        }
        cell = ablations.Cell("cell", 1000, 1, "none")
        seed = {
            "environment_seed": 333183,
            "policy_seed": 10000,
            "episode_start": 0,
        }
        output_root = Path("/output")
        gpu_pool = __import__("queue").Queue()
        gpu_pool.put(2)

        with (
            mock.patch.dict(
                os.environ,
                {"PYTHONPATH": "src", "LD_LIBRARY_PATH": "/cuda/lib64"},
                clear=False,
            ),
            mock.patch.object(ablations, "_run_command", return_value=["/python"]),
            mock.patch.object(Path, "exists", return_value=False),
            mock.patch.object(Path, "mkdir"),
            mock.patch.object(Path, "open", mock.mock_open()),
            mock.patch.object(ablations.subprocess, "run") as run,
            mock.patch.object(ablations, "_atomic_json"),
            mock.patch.object(ablations, "_completed_output", return_value=True),
        ):
            run.return_value.returncode = 0
            ablations._one_run(
                contract=contract,
                cell=cell,
                seed=seed,
                output_root=output_root,
                gpu_pool=gpu_pool,
            )

        subprocess_env = run.call_args.kwargs["env"]
        self.assertEqual(subprocess_env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(
            subprocess_env["PYTHONPATH"],
            os.pathsep.join(("/frozen/fastwam/src", "src")),
        )
        self.assertEqual(
            subprocess_env["LD_LIBRARY_PATH"],
            os.pathsep.join(("/nvidia/driver-lib", "/cuda/lib64")),
        )
        self.assertEqual(
            subprocess_env["VK_ICD_FILENAMES"], "/nvidia/nvidia_icd.json"
        )
        self.assertEqual(
            subprocess_env["VK_DRIVER_FILES"], "/nvidia/nvidia_icd.json"
        )
        self.assertEqual(subprocess_env["__GLX_VENDOR_LIBRARY_NAME"], "nvidia")
        self.assertEqual(
            subprocess_env["__EGL_VENDOR_LIBRARY_FILENAMES"],
            "/nvidia/10_nvidia.json",
        )

    def test_nvidia_driver_library_dir_requires_absolute_complete_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            ablations._nvidia_driver_library_dir(Path("driver-lib"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "lacks required libraries"):
                ablations._nvidia_driver_library_dir(root)
            for pattern in ablations.REQUIRED_NVIDIA_DRIVER_LIBRARIES:
                (root / pattern.replace("*", "570.153.02")).write_bytes(b"driver")

            resolved = ablations._nvidia_driver_library_dir(root)

        self.assertEqual(resolved, str(root))

    def test_nvidia_graphics_manifest_pins_regular_expected_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            driver = root / "libGLX_nvidia.so.570.153.02"
            driver.write_bytes(b"driver")
            manifest = root / "nvidia_icd.json"
            manifest.write_text(
                json.dumps({"ICD": {"library_path": str(driver)}}),
                encoding="utf-8",
            )

            resolved = ablations._nvidia_graphics_manifest(
                manifest,
                label="NVIDIA Vulkan ICD",
                expected_library_prefix="libGLX_nvidia.so.",
            )

            self.assertEqual(resolved, str(manifest))

    def test_nvidia_graphics_manifest_rejects_symlink_and_wrong_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong = root / "libEGL_mesa.so.0"
            wrong.write_bytes(b"mesa")
            manifest = root / "nvidia_icd.json"
            manifest.write_text(
                json.dumps({"ICD": {"library_path": str(wrong)}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unexpected library identity"):
                ablations._nvidia_graphics_manifest(
                    manifest,
                    label="NVIDIA Vulkan ICD",
                    expected_library_prefix="libGLX_nvidia.so.",
                )
            linked = root / "linked.json"
            linked.symlink_to(manifest)
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                ablations._nvidia_graphics_manifest(
                    linked,
                    label="NVIDIA Vulkan ICD",
                    expected_library_prefix="libGLX_nvidia.so.",
                )

    def test_matrix_separates_replanning_oracles_and_checkpoints(self) -> None:
        cells = {cell.name: cell for cell in ablations.CELLS}

        self.assertEqual(cells["step1000_h5_policy"].exec_horizon, 5)
        self.assertEqual(cells["step1000_h1_policy"].exec_horizon, 1)
        self.assertEqual(
            {
                cells["step1000_h5_oracle_robot0_pose"].oracle_intervention,
                cells["step1000_h5_oracle_robot0_gripper"].oracle_intervention,
                cells["step1000_h5_oracle_robot1_action"].oracle_intervention,
            },
            {"robot0_pose", "robot0_gripper", "robot1_action"},
        )
        self.assertEqual(cells["step0500_h5_policy"].exec_horizon, 5)
        self.assertEqual(cells["step0500_h1_policy"].checkpoint_step, 500)

    def test_checkpoint_availability_never_substitutes_missing_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "step_000500.pt").write_bytes(b"500")
            (root / "step_001000.pt").write_bytes(b"1000")

            rows = ablations.checkpoint_availability(root)

        availability = {row["step"]: row["available"] for row in rows}
        self.assertEqual(
            availability,
            {250: False, 500: True, 750: False, 1000: True, 2500: False, 5000: False},
        )

    def test_python_paths_preserve_venv_launcher_and_record_real_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "python3.10"
            base.write_bytes(b"#!/bin/sh\nexit 0\n")
            base.chmod(0o755)
            venv = root / "venv" / "bin"
            venv.mkdir(parents=True)
            launcher = venv / "python"
            launcher.symlink_to(base)

            executable, realpath = ablations._python_paths(launcher)

        self.assertEqual(executable, str(launcher))
        self.assertEqual(realpath, str(base))

    def test_python_paths_reject_relative_or_non_executable_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute path"):
            ablations._python_paths(Path("venv/bin/python"))
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "python"
            candidate.write_text("#!/bin/sh\n")
            os.chmod(candidate, 0o644)
            with self.assertRaisesRegex(ValueError, "not executable"):
                ablations._python_paths(candidate)

    def test_panel_rows_freeze_eight_environment_and_policy_seeds(self) -> None:
        panel = {
            "episodes": [
                {"task_name": "PlaceFood-rf", "episode_seed": seed}
                for seed in (333183, 333327, 333225, 333180, 333251, 333130, 333167, 333234)
            ]
        }

        rows = ablations._panel_rows(panel)

        self.assertEqual([row["policy_seed"] for row in rows], list(range(10000, 10008)))
        self.assertEqual(rows[0]["episode_start"], 0)
        self.assertEqual(rows[-1]["episode_start"], 7)

    def test_panel_rows_accept_legacy_task_key_but_ignore_other_tasks(self) -> None:
        panel = {
            "episodes": [
                {"task_name": "PlaceCubeInCup-rf", "episode_seed": 1},
                *[
                    {"task": "PlaceFood-rf", "episode_seed": seed}
                    for seed in range(10, 18)
                ],
            ]
        }

        rows = ablations._panel_rows(panel)

        self.assertEqual([row["environment_seed"] for row in rows], list(range(10, 18)))

    def test_completed_output_requires_terminal_cell_and_seed_identity(self) -> None:
        cell = ablations.Cell("cell", 1000, 1, "none")
        seed = {"environment_seed": 333183, "policy_seed": 10000, "episode_start": 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "step_001000.pt"
            checkpoint.write_bytes(b"checkpoint")
            checkpoint_stat = checkpoint.stat()
            contract = {
                "checkpoint_availability": [
                    {
                        "step": 1000,
                        "path": str(checkpoint),
                        "available": True,
                        "bytes": checkpoint_stat.st_size,
                        "mtime_ns": checkpoint_stat.st_mtime_ns,
                    }
                ]
            }
            (root / "summary.json").write_text(
                json.dumps({"status": "COMPLETED", "rollout": {"status": "completed"}})
            )
            (root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "terminal",
                        "rollout_cell": {
                            "initial_state": "raw",
                            "exec_horizon": 1,
                            "oracle_intervention": "none",
                        },
                        "policy_request": {
                            "checkpoint_path": str(checkpoint),
                            "checkpoint_bytes": checkpoint_stat.st_size,
                            "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
                            "policy_seed": 10000,
                        },
                        "episode": {"environment_seed": 333183},
                    }
                )
            )

            self.assertTrue(ablations._completed_output(root, cell, seed, contract))

            manifest_path = root / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["policy_request"]["policy_seed"] = 10001
            manifest_path.write_text(json.dumps(manifest))
            self.assertFalse(ablations._completed_output(root, cell, seed, contract))

            manifest["policy_request"]["policy_seed"] = 10000
            manifest["policy_request"]["checkpoint_mtime_ns"] += 1
            manifest_path.write_text(json.dumps(manifest))
            self.assertFalse(ablations._completed_output(root, cell, seed, contract))

    def test_command_passes_explicit_formal_oracle_and_horizon(self) -> None:
        contract = {
            "python": "/python",
            "diagnostic": "/diagnostic.py",
            "panel": "/panel.json",
            "dataset_root": "/dataset",
            "robofactory_root": "/robofactory",
            "gaussian_cache": "/gaussian",
            "checkpoint_dir": "/checkpoints",
            "checkpoint_availability": [
                {
                    "step": 1000,
                    "path": "/checkpoints/step_001000.pt",
                    "available": True,
                    "bytes": 123,
                    "mtime_ns": 456,
                }
            ],
            "stats": "/stats.json",
            "context_file": "/context.pt",
            "model_cache_root": "/models",
            "policy_lightning_repo": "/policy-lightning",
            "noposplat_checkpoint": "/noposplat.ckpt",
            "action_horizon": 32,
            "num_inference_steps": 20,
            "initial_state": "raw",
            "max_steps": 300,
        }
        cell = ablations.Cell("oracle", 1000, 5, "robot0_gripper")
        seed = {"environment_seed": 333183, "policy_seed": 10000, "episode_start": 0}

        command = ablations._run_command(contract, cell, seed, Path("/output"))

        self.assertEqual(command[command.index("--exec-horizon") + 1], "5")
        self.assertEqual(
            command[command.index("--oracle-intervention") + 1], "robot0_gripper"
        )
        self.assertIn("--formal-contract", command)


if __name__ == "__main__":
    unittest.main()
