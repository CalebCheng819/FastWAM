#!/usr/bin/env python3
"""Wait for P12 checkpoints and submit one immutable DLC evaluation job."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = (
    "FASTWAM-MR-N2-PLACEFOOD-CROSSAGENT-GAUSSIAN-P12-EVAL-R4-DLC-20260815"
)
DISPLAY_NAME = "fastwam-p12-gaussian-eval-r4-dlc-20260815"
WORKSPACE_ID = "270969"
RESOURCE_ID = "quotaksvqq2oh2pg"
IMAGE = (
    "dsw-registry-vpc.cn-beijing.cr.aliyuncs.com/pai/pytorch:"
    "2.7.1-gpu-py310-cu128-ubuntu22.04-3995b779-1764350887"
)
CPFS_DATASOURCE_ID = "d-a5mu77ymwjio71dkmw"
OSS_DATASOURCE_ID = "d-n7rly4fll0q2z6v91h"
PROFILE_PATH = Path("/root/.aliyun/config.json")

SOURCE_ROOT = Path(
    "/cpfs/user/chengjuntao/experiments/FastWAM-p12-eval-r4-dlc-20260815"
)
MODEL_ROOT = Path(
    "/cpfs/user/chengjuntao/experiments/FastWAM-p12-render-1181a37-20260814"
)
TRAIN_ROOT = Path(
    "/oss-chengjuntao/artifacts/"
    "fastwam-placefood-crossagent-gaussian-p12-s42-8g-r2-20260814"
)
TF_OUTPUT = Path(
    "/oss-chengjuntao/artifacts/"
    "fastwam-placefood-crossagent-gaussian-p12-paired-tf-20260815-r4-dlc"
)
STEP500_OUTPUT = Path(
    "/oss-chengjuntao/artifacts/"
    "fastwam-p12-step000500-official-topp-h32-val8-20260815-r4-dlc"
)
STEP1000_OUTPUT = Path(
    "/oss-chengjuntao/artifacts/"
    "fastwam-p12-step001000-official-topp-h32-val8-20260815-r4-dlc"
)
RECORD_ROOT = Path(
    "/oss-chengjuntao/artifacts/"
    "fastwam-placefood-crossagent-gaussian-p12-eval-controller-20260815-r4-dlc"
)
LOCAL_ROOT = Path(
    "/mnt/workspace/experiments/FASTWAM-P12-EVAL-R4-DLC-20260815/runtime"
)
EXPERIMENT_REL = Path(".research-workflow/experiments") / EXPERIMENT_ID
TRAINING_COMMIT = "1181a375c880a4a51df2ae78d533e16dde757465"
POLICY_LIGHTNING_COMMIT = "c944b4989a89c99c69d2572ea870f6a04680f5e7"

PYTHON = Path("/cpfs/user/chengjuntao/venvs/fastwam-gaudp-py310-20260802/bin/python")
PYTHON_EXTRA = Path(
    "/cpfs/user/chengjuntao/venvs/fastwam-gau0-eval-r7-py310-extra-20260813"
)
PANEL = Path(
    "/cpfs/user/chengjuntao/fastwam_eval_runtime/panels/"
    "robofactory_n234_s42_val8_v1.json"
)
ROBOFACTORY = Path(
    "/cpfs/user/chengjuntao/fastwam_eval_runtime/RoboFactory-challenge-2d34fb3"
)
POLICY_LIGHTNING = Path("/cpfs/user/chengjuntao/Policy-Lightning")
NOPOSPLAT_CHECKPOINT = Path(
    "/cpfs/user/chengjuntao/checkpoints/noposplat/"
    "664ba9156f10a6203f0a0fad2f02c069c6894f4f/"
    "mixRe10kDl3dv_512x512.ckpt"
)
GRAPHICS_ROOT = Path(
    "/cpfs/user/chengjuntao/fastwam-deploy/nvidia-graphics-570.153.02"
)
VULKAN_LOADER = Path(
    "/cpfs/user/chengjuntao/fastwam-deploy/vulkan-loader-1.3.204/"
    "libvulkan.so.1.3.204"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git_revision(root: Path) -> str:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"worktree must be clean: {root}: {dirty}")
    if len(revision) != 40:
        raise RuntimeError(f"expected full Git revision for {root}: {revision}")
    return revision


def checkpoint_paths() -> list[Path]:
    return [
        TRAIN_ROOT / "checkpoints/weights/step_000500.pt",
        TRAIN_ROOT / "checkpoints/weights/step_001000.pt",
    ]


def checkpoints_ready() -> bool:
    return all(
        path.is_file()
        and path.stat().st_size > 0
        and Path(f"{path}.COMPLETE").is_file()
        and Path(f"{path}.COMPLETE").stat().st_size > 0
        for path in checkpoint_paths()
    )


def validate_inputs() -> str:
    eval_commit = git_revision(SOURCE_ROOT)
    if git_revision(MODEL_ROOT) != TRAINING_COMMIT:
        raise RuntimeError("training source revision mismatch")
    if git_revision(POLICY_LIGHTNING) != POLICY_LIGHTNING_COMMIT:
        raise RuntimeError("Policy-Lightning revision mismatch")
    required = [
        PYTHON,
        PYTHON_EXTRA,
        PANEL,
        ROBOFACTORY,
        NOPOSPLAT_CHECKPOINT,
        GRAPHICS_ROOT / "driver-lib",
        GRAPHICS_ROOT / "nvidia_icd.json",
        GRAPHICS_ROOT / "10_nvidia.json",
        VULKAN_LOADER,
        SOURCE_ROOT / EXPERIMENT_REL / "runtime.sh",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"missing DLC evaluation inputs: {missing}")
    if not os.access(PYTHON, os.X_OK):
        raise RuntimeError(f"evaluation Python is not executable: {PYTHON}")
    if any(path.exists() or path.is_symlink() for path in outputs()):
        raise RuntimeError("fresh R4-DLC evaluation output already exists")
    return eval_commit


def outputs() -> tuple[Path, ...]:
    return TF_OUTPUT, STEP500_OUTPUT, STEP1000_OUTPUT


def runtime_env(eval_commit: str) -> dict[str, str]:
    return {
        "P12_EXPERIMENT_ID": EXPERIMENT_ID,
        "P12_DISPLAY_NAME": DISPLAY_NAME,
        "P12_EVAL_ROOT": str(SOURCE_ROOT),
        "P12_EVALUATION_COMMIT": eval_commit,
        "P12_MODEL_ROOT": str(MODEL_ROOT),
        "P12_TRAIN_ROOT": str(TRAIN_ROOT),
        "P12_TRAINING_COMMIT": TRAINING_COMMIT,
        "P12_TRAINING_JOB_ID": "dlc19rgpvuxr56b7",
        "P12_TF_OUTPUT_ROOT": str(TF_OUTPUT),
        "P12_STEP500_OUTPUT_ROOT": str(STEP500_OUTPUT),
        "P12_STEP1000_OUTPUT_ROOT": str(STEP1000_OUTPUT),
        "P12_RECORD_ROOT": str(RECORD_ROOT),
        "P12_EVAL_PYTHON": str(PYTHON),
        "P12_PYTHON_EXTRA": str(PYTHON_EXTRA),
        "P12_EVAL_PANEL": str(PANEL),
        "P12_ROBOFACTORY_ROOT": str(ROBOFACTORY),
        "P12_POLICY_LIGHTNING_ROOT": str(POLICY_LIGHTNING),
        "P12_POLICY_LIGHTNING_COMMIT": POLICY_LIGHTNING_COMMIT,
        "P12_NOPOSPLAT_CHECKPOINT": str(NOPOSPLAT_CHECKPOINT),
        "P12_GRAPHICS_ROOT": str(GRAPHICS_ROOT),
        "P12_VULKAN_LOADER": str(VULKAN_LOADER),
        "P12_RUNTIME_SCRIPT": str(SOURCE_ROOT / EXPERIMENT_REL / "runtime.sh"),
        "P12_INTEGRITY_MODE": "metadata_no_hash",
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
    }


def request_body(eval_commit: str) -> dict[str, Any]:
    return {
        "Accessibility": "PRIVATE",
        "CustomEnvs": [],
        "DataSources": [
            {
                "DataSourceId": CPFS_DATASOURCE_ID,
                "MountAccess": "RO",
                "MountPath": "/cpfs/user/chengjuntao",
            },
            {
                "DataSourceId": OSS_DATASOURCE_ID,
                "MountAccess": "RW",
                "MountPath": "/oss-chengjuntao",
            },
        ],
        "Description": (
            "P12 paired teacher-forcing and PlaceFood val8 closed-loop evaluation; "
            f"experiment={EXPERIMENT_ID}; eval_commit={eval_commit}"
        ),
        "DisplayName": DISPLAY_NAME,
        "Envs": runtime_env(eval_commit),
        "JobMaxRunningTimeMinutes": 720,
        "JobSpecs": [
            {
                "ElasticSpotSpecs": [],
                "Image": IMAGE,
                "LocalMountSpecs": [],
                "PodCount": 1,
                "ResourceConfig": {
                    "CPU": "126",
                    "GPU": "8",
                    "Memory": "960Gi",
                    "SharedMemory": "960Gi",
                },
                "RestartPolicy": "Never",
                "StartupDependencies": [],
                "Type": "Worker",
            }
        ],
        "JobType": "PyTorchJob",
        "Priority": 7,
        "ResourceId": RESOURCE_ID,
        "Settings": {
            "AllocateAllRDMADevices": False,
            "EnableCPUAffinity": False,
            "EnableErrorMonitoringInAIMaster": False,
            "EnableOssAppend": False,
            "EnableRDMA": False,
            "EnableSanityCheck": False,
            "Tags": {
                "experiment_id": EXPERIMENT_ID,
                "objective": "p12-offline-and-closedloop-eval",
                "protocol": "official-topp-h32-val8",
                "provenance": "git-path-time-job-metadata-no-new-hash",
                "topology": "1x8",
            },
        },
        "SuccessPolicy": "AllWorkers",
        "UserCommand": f"exec /bin/bash {SOURCE_ROOT / EXPERIMENT_REL / 'runtime.sh'}",
        "WorkspaceId": WORKSPACE_ID,
    }


def load_client():
    from alibabacloud_credentials.client import Client as CredentialClient
    from alibabacloud_credentials.models import Config as CredentialConfig
    from alibabacloud_pai_dlc20201203.client import Client
    from alibabacloud_tea_openapi.models import Config

    document = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    current = document.get("current")
    profile = next(
        (item for item in document.get("profiles", []) if item.get("name") == current),
        None,
    )
    if profile is None or profile.get("mode") != "CredentialsURI":
        raise RuntimeError("active Alibaba Cloud profile is not CredentialsURI")
    uri = profile.get("credentials_uri")
    if not uri:
        raise RuntimeError("active CredentialsURI profile has no URI")
    credential = CredentialClient(
        CredentialConfig(type="credentials_uri", credentials_uri=uri)
    )
    return Client(
        Config(
            credential=credential,
            region_id="cn-beijing",
            endpoint="pai-dlc.cn-beijing.aliyuncs.com",
        )
    )


def create_job(eval_commit: str) -> dict[str, Any]:
    from alibabacloud_pai_dlc20201203 import models

    client = load_client()
    listed = client.list_jobs(
        models.ListJobsRequest(
            workspace_id=WORKSPACE_ID,
            resource_id=RESOURCE_ID,
            display_name=DISPLAY_NAME,
            page_number=1,
            page_size=100,
            order="desc",
            sort_by="GmtCreateTime",
        )
    ).body.to_map()
    matches = [
        item
        for item in listed.get("Jobs") or []
        if item.get("DisplayName") == DISPLAY_NAME
    ]
    if matches:
        raise RuntimeError(
            "refusing duplicate submit: "
            + json.dumps(
                [
                    {"JobId": item.get("JobId"), "Status": item.get("Status")}
                    for item in matches
                ]
            )
        )
    request_map = request_body(eval_commit)
    request = models.CreateJobRequest().from_map(request_map)
    request.validate()
    if request.to_map() != request_map:
        raise RuntimeError("DLC request model round-trip changed the frozen request")
    created = client.create_job(request).body.to_map()
    job_id = str(created.get("JobId") or "")
    if not job_id:
        raise RuntimeError("CreateJob returned no JobId")
    observed = client.get_job(
        job_id, models.GetJobRequest(need_detail=True)
    ).body.to_map()
    return {
        "schema_version": "fastwam-p12-dlc-eval-submission-v1",
        "cloud_mutations_called": ["CreateJob"],
        "display_name": DISPLAY_NAME,
        "experiment_id": EXPERIMENT_ID,
        "evaluation_code_commit": eval_commit,
        "job_id": job_id,
        "observed_status": observed.get("Status"),
        "outputs": [str(path) for path in outputs()],
        "priority": 7,
        "request_id": created.get("RequestId"),
        "requested_topology": "1x8",
        "submitted_at_utc": utc_now(),
        "training_job_id": "dlc19rgpvuxr56b7",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        raise ValueError("poll and timeout must be positive")
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    lock = (LOCAL_ROOT / "submit.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another P12 R4 DLC submit supervisor is active") from error
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while not checkpoints_ready():
        if time.monotonic() - started >= args.timeout_seconds:
            atomic_json(
                RECORD_ROOT / "submission-state.json",
                {
                    "state": "FAILED",
                    "detail": "timed out waiting for both checkpoint markers",
                    "observed_at_utc": utc_now(),
                },
            )
            raise TimeoutError("timed out waiting for both checkpoint markers")
        state = {
            "state": "WAITING_FOR_CHECKPOINTS",
            "detail": str(TRAIN_ROOT),
            "observed_at_utc": utc_now(),
        }
        atomic_json(RECORD_ROOT / "submission-state.json", state)
        atomic_json(LOCAL_ROOT / "submission-state.json", state)
        time.sleep(args.poll_seconds)
    eval_commit = validate_inputs()
    request = request_body(eval_commit)
    atomic_json(RECORD_ROOT / "create-job-request.json", request)
    if not args.submit:
        print(json.dumps({"validated": True, "request": request}, sort_keys=True))
        return
    receipt = create_job(eval_commit)
    atomic_json(RECORD_ROOT / "submission-receipt.json", receipt)
    state = {
        "state": "SUBMITTED",
        "detail": receipt["job_id"],
        "observed_at_utc": utc_now(),
    }
    atomic_json(RECORD_ROOT / "submission-state.json", state)
    atomic_json(LOCAL_ROOT / "submission-state.json", state)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
