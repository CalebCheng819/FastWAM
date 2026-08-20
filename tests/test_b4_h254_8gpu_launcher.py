import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class B4H254PreparationTests(unittest.TestCase):
    def test_scale_preserves_effective_global_batch_24(self):
        text = (ROOT / "configs/scale/robofactory_multi_robot_8gpu_h254_b4.yaml").read_text()
        self.assertIn("gradient_accumulation_steps: 3", text)
        self.assertIn("provenance_mode: stat_cmp", text)
        self.assertNotIn("gradient_accumulation_steps: 1", text)

    def test_raw_manifest_is_exactly_24_unique_h5_paths(self):
        paths = [
            line.strip()
            for line in (ROOT / "configs/transfer/b4_h254_raw_h5_paths.txt").read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(len(paths), 24)
        self.assertEqual(len(set(paths)), 24)
        self.assertTrue(all(path.endswith(".h5") and not path.startswith("/") for path in paths))

    def test_cdk_manifest_has_five_credential_free_tasks(self):
        path = ROOT / "configs/transfer/b4_h254_cdk_tasks.json"
        doc = json.loads(path.read_text())
        self.assertEqual(len(doc["tasks"]), 5)
        self.assertEqual(doc["source"]["bucket"], "pjlab-bjpai-manip")
        self.assertEqual(doc["destination"]["bucket"], "eailab")
        lowered = path.read_text().lower()
        for forbidden in ("access_key", "secret_key", "password", "checksum", "sha256"):
            self.assertNotIn(forbidden, lowered)

    def test_launcher_is_single_node_eight_gpu_and_no_manual_rdma(self):
        text = (ROOT / "scripts/launch_b4_h254_8gpu.sh").read_text()
        self.assertIn("--num_machines 1 --machine_rank 0 --num_processes 8", text)
        self.assertIn("+scale=robofactory_multi_robot_8gpu_h254_b4", text)
        self.assertIn("stats_source_root=${LOGICAL_STATS_ROOT}", text)
        self.assertIn("FASTWAM_H_DRY_RUN", text)
        self.assertNotIn("NCCL_IB_HCA", text)
        self.assertNotIn("NCCL_SOCKET_IFNAME", text)

    def test_environment_bootstrap_is_exact_and_fail_closed(self):
        text = (ROOT / "scripts/bootstrap_b4_h254_env.sh").read_text()
        self.assertIn("wheelhouse/torch-2.7.1-cu128-cp310", text)
        self.assertIn("validated Torch wheelhouse must contain exactly 18 wheels", text)
        self.assertIn('"torchcodec": "0.5+cu128"', text)
        self.assertIn('"torchvision": "0.22.1+cu128"', text)
        self.assertIn('"triton": "3.3.1"', text)
        self.assertIn("--no-index", text)
        self.assertIn("--no-deps", text)
        self.assertIn('TMPDIR="$B4_PIP_TMP_DIR"', text)
        self.assertIn("refusing to overwrite target", text)
        self.assertNotIn("--system-site-packages", text)

    def test_torch_wheelhouse_is_exact_and_atomically_published(self):
        text = (ROOT / "scripts/prepare_b4_h254_torch_wheelhouse.sh").read_text()
        self.assertIn("torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl", text)
        self.assertIn("torchcodec==0.5+cu128", text)
        self.assertIn("torchvision==0.22.1+cu128", text)
        self.assertIn("triton==3.3.1", text)
        self.assertIn("expected exactly one regular cached wheel", text)
        self.assertIn("wheelhouse contract mismatch", text)
        self.assertIn('mv -- "$BUILD_DIR" "$TARGET"', text)
        self.assertNotIn("--system-site-packages", text)

    def test_renderer_can_only_predict_h254_job(self):
        commit = "1" * 40
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/render_b4_h254_rjob.py"), "--commit", commit],
            text=True,
            capture_output=True,
            check=True,
        )
        doc = json.loads(completed.stdout)
        command = doc["command"]
        self.assertFalse(doc["formal_submission_performed"])
        self.assertEqual(command[command.index("--predict-only") + 1], "true")
        self.assertEqual(command[command.index("--gpu") + 1], "8")
        self.assertEqual(command[command.index("--restart-policy") + 1], "never")
        self.assertEqual(command[command.index("--store-host-nvme") + 1], "--mount")
        self.assertEqual(command[command.index("--priority") + 1], "9")
        self.assertEqual(
            command[command.index("--positive-tags") + 1],
            "node/gpu-l-lg-cmc-h-h200-0254.host.h.pjlab.org.cn",
        )
        self.assertNotIn("--group", command)
        self.assertEqual(command[command.index("--charged-group") + 1], "eailabagent_gpu")
        self.assertEqual(command[command.index("--private-machine") + 1], "group")
        self.assertIn("--store-host-nvme", command)
        self.assertIn(
            "FASTWAM_PYTHON=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/envs/fastwam-b4-h254-py310-20260820/bin/python3.10",
            command,
        )
        self.assertEqual(command.count("submit"), 1)


if __name__ == "__main__":
    unittest.main()
