#!/usr/bin/env python3
"""Render the immutable one-pod/eight-GPU DLC evaluation request."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path


EXPERIMENT_ID = "FASTWAM-MR-N234-VG1H1GAU1-STEP10000-PLACEFOOD-SAME8-S42-R1-20260823"
RUN_ID = "fastwam-gau1-step10k-placefood-same8-r1-20260823"
DISPLAY_NAME = "fw-gau1-s10k-placefood-same8-r1"
OUTPUT_ROOT = "/oss-chengjuntao/artifacts/fastwam-gau1-step10k-placefood-same8-eval-20260823-r1"
CHECKPOINT = "/oss-chengjuntao/artifacts/fastwam-n234-vg1h1gau1-cont50k-s42-24g-r1-20260822/checkpoints/weights/step_010000.pt"
SOURCE_BUNDLE = "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/fastwam-gau1-step10k-placefood-same8-eval-20260823-r1.bundle"
IMAGE = "dsw-registry-vpc.cn-beijing.cr.aliyuncs.com/pai/pytorch:2.7.1-gpu-py310-cu128-ubuntu22.04-3995b779-1764350887"


def _exclusive_json(path: Path, payload: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.source_commit) != 40 or any(ch not in "0123456789abcdef" for ch in args.source_commit):
        raise SystemExit("source commit must be a full lowercase Git object id")

    bootstrap = r'''set -Eeuo pipefail
umask 077
die() { printf 'STEP10K_EVAL_BOOTSTRAP_FATAL: %s\n' "$*" >&2; exit 1; }
[[ "${FASTWAM_EXPERIMENT_ID:-}" == "FASTWAM-MR-N234-VG1H1GAU1-STEP10000-PLACEFOOD-SAME8-S42-R1-20260823" ]] || die "experiment drift"
[[ "${FASTWAM_RUN_ID:-}" == "fastwam-gau1-step10k-placefood-same8-r1-20260823" ]] || die "run drift"
[[ "${FASTWAM_SOURCE_BUNDLE:-}" == "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/fastwam-gau1-step10k-placefood-same8-eval-20260823-r1.bundle" ]] || die "bundle drift"
[[ -f "${FASTWAM_SOURCE_BUNDLE}" && ! -L "${FASTWAM_SOURCE_BUNDLE}" ]] || die "bundle unsafe"
root="$(mktemp -d /tmp/fastwam-step10k-source.XXXXXXXX)"
cleanup() { rm -rf -- "${root}"; }
trap cleanup EXIT
git clone --quiet "${FASTWAM_SOURCE_BUNDLE}" "${root}/source" || die "bundle clone failed"
actual="$(git -C "${root}/source" rev-parse HEAD)"
[[ "${actual}" == "${FASTWAM_SOURCE_COMMIT}" ]] || die "source commit mismatch"
[[ -z "$(git -C "${root}/source" status --porcelain --untracked-files=all)" ]] || die "source checkout dirty"
export FASTWAM_SOURCE_ROOT="${root}/source"
exec bash "${root}/source/.research-workflow/experiments/FASTWAM-MR-N234-VG1H1GAU1-STEP10000-PLACEFOOD-SAME8-S42-R1-20260823/runtime.sh"
'''
    user_command = "python3 - <<'PY'\nimport base64,os\np=base64.b64decode(os.environ['FASTWAM_LAUNCHER_B64'])\nos.execv('/bin/bash',['bash','-c',p.decode('utf-8')])\nPY"
    envs = {
        "FASTWAM_EXPERIMENT_ID": EXPERIMENT_ID,
        "FASTWAM_RUN_ID": RUN_ID,
        "FASTWAM_OUTPUT_ROOT": OUTPUT_ROOT,
        "FASTWAM_SOURCE_BUNDLE": SOURCE_BUNDLE,
        "FASTWAM_SOURCE_COMMIT": args.source_commit,
        "FASTWAM_CHECKPOINT": CHECKPOINT,
        "FASTWAM_CHECKPOINT_SIZE_BYTES": "12047213657",
        "FASTWAM_PANEL": "/cpfs/user/chengjuntao/fastwam_eval_runtime/panels/robofactory_n234_s42_val8_v1.json",
        "FASTWAM_PANEL_SIZE_BYTES": "44584",
        "FASTWAM_STATS": "/oss-chengjuntao/cpfs-user-chengjuntao/datasets/robofactory_multi_robot/fastwam_multi_robot_n234_train_s42_stats_v2.json",
        "FASTWAM_STATS_SIZE_BYTES": "3604",
        "FASTWAM_DATASET_ROOT": "/oss-chengjuntao/cpfs-user-chengjuntao/datasets/robofactory_multi_robot",
        "FASTWAM_ROBOFACTORY_ROOT": "/cpfs/user/chengjuntao/fastwam_eval_runtime/RoboFactory-challenge-2d34fb3",
        "FASTWAM_CONTEXT_CACHE_DIR": "/oss-chengjuntao/cpfs-user-chengjuntao/datasets/robofactory_multi_robot/text_embeds_cache_n234",
        "FASTWAM_CONTEXT_SIZE_BYTES": "1051869",
        "FASTWAM_MODEL_CACHE_ROOT": "/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/model-cache",
        "FASTWAM_POLICY_LIGHTNING_ROOT": "/cpfs/user/chengjuntao/Policy-Lightning",
        "FASTWAM_POLICY_LIGHTNING_COMMIT": "c944b4989a89c99c69d2572ea870f6a04680f5e7",
        "FASTWAM_NOPOSPLAT_CHECKPOINT": "/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/noposplat/664ba9156f10a6203f0a0fad2f02c069c6894f4f/mixRe10kDl3dv_512x512.ckpt",
        "FASTWAM_NOPOSPLAT_CHECKPOINT_SIZE_BYTES": "2448478423",
        "FASTWAM_NVIDIA_GRAPHICS_ROOT": "/cpfs/user/chengjuntao/fastwam-deploy/nvidia-graphics-570.153.02",
        "FASTWAM_PYTHON": "/cpfs/user/chengjuntao/venvs/fastwam-gaudp-py310-20260802/bin/python",
        "FASTWAM_TRAINING_SOURCE_COMMIT": "92b62430aebbb1ddfb30ff8e4c362ad7b71fbc86",
        "FASTWAM_TRAINING_JOB_ID": "dlc1bjyosqteai2f",
        "FASTWAM_LAUNCHER_B64": base64.b64encode(bootstrap.encode()).decode(),
    }
    body = {
        "Accessibility": "PRIVATE",
        "DisplayName": DISPLAY_NAME,
        "Description": f"Formal PlaceFood same8 closed-loop evaluation of GAU1 step_010000.pt; experiment={EXPERIMENT_ID}",
        "WorkspaceId": "270969",
        "ResourceId": "quotaksvqq2oh2pg",
        "JobType": "PyTorchJob",
        "Priority": 7,
        "JobMaxRunningTimeMinutes": 2160,
        "SuccessPolicy": "AllWorkers",
        "Envs": envs,
        "DataSources": [
            {"DataSourceId": "d-a5mu77ymwjio71dkmw", "MountPath": "/cpfs/user/chengjuntao", "MountAccess": "RO"},
            {"DataSourceId": "d-n7rly4fll0q2z6v91h", "MountPath": "/oss-chengjuntao", "MountAccess": "RW"},
        ],
        "JobSpecs": [{
            "Type": "Worker",
            "PodCount": 1,
            "Image": IMAGE,
            "RestartPolicy": "Never",
            "ResourceConfig": {"GPU": "8", "CPU": "126", "Memory": "960Gi", "SharedMemory": "960Gi"},
        }],
        "UserCommand": user_command,
    }
    document = {
        "schema_version": "fastwam-dlc-create-job-dry-run-v1",
        "dry_run": True,
        "submission_not_performed": True,
        "operation": "CreateJob",
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "source_commit": args.source_commit,
        "source_bundle": SOURCE_BUNDLE,
        "output_root": OUTPUT_ROOT,
        "launcher_payload_base64": envs["FASTWAM_LAUNCHER_B64"],
        "request": body,
    }
    _exclusive_json(args.output, document)
    print(json.dumps({"dry_run": True, "output": str(args.output), "display_name": DISPLAY_NAME, "priority": 7, "gpus": 8}, sort_keys=True))


if __name__ == "__main__":
    main()
