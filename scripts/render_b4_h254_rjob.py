#!/usr/bin/env python3
"""Render or execute only the H-cluster scheduler prediction for B4 H254.

This program deliberately has no formal-submit mode.  The emitted rjob command
always contains ``--predict-only true``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


IMAGE = (
    "registry.h.pjlab.org.cn/ailab/vulkan@"
    "sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
)
NODE = "gpu-l-lg-cmc-h-h200-0254.host.h.pjlab.org.cn"
GROUP = "eailabagent_gpu"
GPFS_ROOT = "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao"
PUBLISH_ROOT = f"{GPFS_ROOT}/fastwam-b4-h254-8g-20260820"
PYTHON = f"{GPFS_ROOT}/FastWAM_yuner/.conda/fastwam/bin/python3.10"
RCLONE_CONFIG = f"{GPFS_ROOT}/.fastwam_runtime/rclone.conf"
RAW_REMOTE = (
    "eailab-hdd2:fkp-migrate/ailab-eailabagent-gpfs/chengjuntao/"
    "placefood_wam/Policy-Lightning/data/baai_tasks"
)
INPUT_REMOTE = (
    "eailab-hdd2:eailab/chengjuntao/fastwam/robofactory-multirobot/"
    "b4-h254-s42-20260820/inputs"
)


def build_command(args: argparse.Namespace) -> list[str]:
    bundle = args.bundle or f"{PUBLISH_ROOT}/FastWAM-{args.commit}.bundle"
    launcher = args.launcher or f"{PUBLISH_ROOT}/launch_b4_h254_8gpu.sh"
    output_remote = (
        "eailab-hdd2:eailab/chengjuntao/fastwam/robofactory-multirobot/"
        f"b4-h254-s42-20260820/outputs/{args.run_id}"
    )
    env = [
        f"FASTWAM_RUN_ID={args.run_id}",
        f"FASTWAM_ATTEMPT_ID={args.attempt_id}",
        f"FASTWAM_GIT_COMMIT={args.commit}",
        f"FASTWAM_GIT_BUNDLE={bundle}",
        f"FASTWAM_PYTHON={args.python}",
        f"FASTWAM_RCLONE_CONFIG={args.rclone_config}",
        f"FASTWAM_H_RAW_H5_REMOTE={RAW_REMOTE}",
        f"FASTWAM_H_INPUT_REMOTE={INPUT_REMOTE}",
        f"FASTWAM_H_OUTPUT_REMOTE={output_remote}",
        "FASTWAM_H_DRY_RUN=0",
    ]
    return [
        "/usr/local/bin/rjob",
        "submit",
        "--predict-only",
        "true",
        "--name",
        args.run_id,
        "--group",
        GROUP,
        "--charged-group",
        GROUP,
        "--priority",
        "9",
        "--task-type",
        "normal",
        "--restart-policy",
        "never",
        "--termination-grace-period-seconds",
        "60",
        "--backoff_limit",
        "1",
        "--image",
        IMAGE,
        "--cpu",
        "40",
        "--gpu",
        "8",
        "--memory",
        "950000",
        "--positive-tags",
        NODE,
        "--store-host-nvme",
        "True",
        "--mount",
        "gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs",
        "gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public",
        "--",
        "/usr/bin/env",
        *env,
        "/bin/bash",
        launcher,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--run-id", default="fastwam-b4-h254-8g-s42-r1-20260820"
    )
    parser.add_argument("--attempt-id", default="attempt-001")
    parser.add_argument("--bundle")
    parser.add_argument("--launcher")
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--rclone-config", default=RCLONE_CONFIG)
    parser.add_argument("--execute-predict", action="store_true")
    parser.add_argument("--receipt")
    args = parser.parse_args()
    if len(args.commit) != 40 or any(c not in "0123456789abcdef" for c in args.commit):
        parser.error("--commit must be a full lowercase Git commit")
    if not args.run_id.replace("-", "").isalnum() or args.run_id.lower() != args.run_id:
        parser.error("--run-id must contain lowercase letters, digits, and hyphens")
    if not args.attempt_id.startswith("attempt-"):
        parser.error("--attempt-id must use attempt-NNN")
    return args


def main() -> int:
    args = parse_args()
    command = build_command(args)
    if "--predict-only" not in command or command[command.index("--predict-only") + 1] != "true":
        raise RuntimeError("renderer must remain predict-only")
    result = None
    if args.execute_predict:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        result = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    payload = {
        "schema_version": 1,
        "operation": "rjob scheduler prediction only",
        "formal_submission_performed": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "prediction": result,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.receipt:
        receipt = Path(args.receipt)
        if receipt.exists() or receipt.is_symlink():
            raise RuntimeError(f"refusing to overwrite receipt: {receipt}")
        receipt.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result is None or result["returncode"] == 0 else result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
