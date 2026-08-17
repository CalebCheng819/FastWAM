#!/usr/bin/env python3
"""Wait for P12 checkpoints and submit one immutable R6 DLC evaluation job."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = (
    "FASTWAM-MR-N2-PLACEFOOD-CROSSAGENT-GAUSSIAN-P12-EVAL-R6-DLC-20260817"
)
DISPLAY_NAME = "fastwam-p12-gaussian-eval-r6-dlc-full-20260817"
PROBE_DISPLAY_NAME = "fastwam-p12-gaussian-eval-r6-dlc-graphics-probe-20260817"
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
    "/cpfs/user/chengjuntao/experiments/FastWAM-p12-eval-r6-dlc-20260817"
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
    "fastwam-placefood-crossagent-gaussian-p12-paired-tf-20260817-r6-dlc"
)
STEP500_OUTPUT = Path(
    "/oss-chengjuntao/artifacts/"
    "fastwam-p12-step000500-official-topp-h32-val8-20260817-r6-dlc"
)
STEP1000_OUTPUT = Path(
    "/oss-chengjuntao/artifacts/"
    "fastwam-p12-step001000-official-topp-h32-val8-20260817-r6-dlc"
)
RECORD_ROOT = Path(
    "/oss-chengjuntao/artifacts/"
    "fastwam-placefood-crossagent-gaussian-p12-eval-controller-20260817-r6-dlc"
)
PROBE_RECORD_ROOT = Path(
    "/oss-chengjuntao/artifacts/"
    "fastwam-placefood-crossagent-gaussian-p12-eval-controller-20260817-r6-dlc-graphics-probe"
)
LOCAL_ROOT = Path(
    "/mnt/workspace/experiments/FASTWAM-P12-EVAL-R6-DLC-20260817/runtime"
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


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def validate_inputs(run_mode: str = "full_eval") -> str:
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
    if not checkpoints_ready():
        raise RuntimeError("both frozen P12 checkpoint COMPLETE markers are required")
    if run_mode == "full_eval":
        validate_probe_terminal()
        if any(path.exists() or path.is_symlink() for path in outputs()):
            raise RuntimeError("fresh R6-DLC evaluation output already exists")
    elif run_mode != "graphics_probe":
        raise RuntimeError(f"unknown run mode: {run_mode}")
    return eval_commit


def outputs() -> tuple[Path, ...]:
    return TF_OUTPUT, STEP500_OUTPUT, STEP1000_OUTPUT


def read_regular_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required ordinary JSON file is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON object is invalid: {path}")
    return value


def validate_probe_terminal(root: Path = PROBE_RECORD_ROOT) -> dict[str, Any]:
    summary_path = root / "graphics-probe-summary.json"
    terminal_path = root / "worker-terminal.json"
    summary = read_regular_json(summary_path)
    terminal = read_regular_json(terminal_path)
    if summary.get("schema_version") != "fastwam-p12-dlc-gpu-probe-v1":
        raise RuntimeError("graphics probe summary schema mismatch")
    if summary.get("status") != "SUCCEEDED":
        raise RuntimeError("graphics probe summary is not SUCCEEDED")
    if terminal.get("schema_version") != "fastwam-p12-dlc-eval-worker-terminal-v1":
        raise RuntimeError("graphics probe terminal schema mismatch")
    if terminal.get("status") != "SUCCEEDED" or terminal.get("return_code") != 0:
        raise RuntimeError("graphics probe worker terminal is not successful")
    profile = summary.get("graphics_profile")
    if not isinstance(profile, str) or not profile:
        raise RuntimeError("graphics probe summary has no selected profile")
    if terminal.get("graphics_profile") != profile:
        raise RuntimeError("graphics probe profile differs from worker terminal")
    return {
        "summary_path": str(summary_path),
        "terminal_path": str(terminal_path),
        "graphics_profile": profile,
        "summary_status": summary["status"],
        "terminal_status": terminal["status"],
        "return_code": terminal["return_code"],
    }


def runtime_env(eval_commit: str, run_mode: str = "full_eval") -> dict[str, str]:
    if run_mode not in {"full_eval", "graphics_probe"}:
        raise RuntimeError(f"unknown run mode: {run_mode}")
    display_name = (
        PROBE_DISPLAY_NAME if run_mode == "graphics_probe" else DISPLAY_NAME
    )
    record_root = (
        PROBE_RECORD_ROOT if run_mode == "graphics_probe" else RECORD_ROOT
    )
    return {
        "P12_EXPERIMENT_ID": EXPERIMENT_ID,
        "P12_DISPLAY_NAME": display_name,
        "P12_RUN_MODE": run_mode,
        "P12_EVAL_ROOT": str(SOURCE_ROOT),
        "P12_EVALUATION_COMMIT": eval_commit,
        "P12_MODEL_ROOT": str(MODEL_ROOT),
        "P12_TRAIN_ROOT": str(TRAIN_ROOT),
        "P12_TRAINING_COMMIT": TRAINING_COMMIT,
        "P12_TRAINING_JOB_ID": "dlc19rgpvuxr56b7",
        "P12_TF_OUTPUT_ROOT": str(TF_OUTPUT),
        "P12_STEP500_OUTPUT_ROOT": str(STEP500_OUTPUT),
        "P12_STEP1000_OUTPUT_ROOT": str(STEP1000_OUTPUT),
        "P12_RECORD_ROOT": str(record_root),
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


def request_body(eval_commit: str, run_mode: str = "full_eval") -> dict[str, Any]:
    return request_body_mode(eval_commit, run_mode=run_mode)


def request_body_mode(eval_commit: str, run_mode: str = "full_eval") -> dict[str, Any]:
    if run_mode not in {"full_eval", "graphics_probe"}:
        raise RuntimeError(f"unknown run mode: {run_mode}")
    if run_mode == "graphics_probe":
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
                "P12 graphics probe using PlaceFood-rf environment construction test; "
                f"experiment={EXPERIMENT_ID}; eval_commit={eval_commit}"
            ),
            "DisplayName": PROBE_DISPLAY_NAME,
            "Envs": runtime_env(eval_commit, run_mode=run_mode),
            "JobMaxRunningTimeMinutes": 240,
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
                    "protocol": "graphics-probe",
                    "provenance": "git-path-time-job-metadata-no-new-hash",
                    "topology": "1x8",
                },
            },
            "SuccessPolicy": "AllWorkers",
            "UserCommand": f"exec /bin/bash {SOURCE_ROOT / EXPERIMENT_REL / 'runtime.sh'}",
            "WorkspaceId": WORKSPACE_ID,
        }
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
        "Envs": runtime_env(eval_commit, run_mode=run_mode),
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


def list_jobs(client: Any, models: Any) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    expected: int | None = None
    page = 1
    while expected is None or len(jobs) < expected:
        body = client.list_jobs(
            models.ListJobsRequest(
                workspace_id=WORKSPACE_ID,
                resource_id=RESOURCE_ID,
                page_number=page,
                page_size=100,
                order="desc",
                sort_by="GmtCreateTime",
            )
        ).body.to_map()
        if expected is None:
            expected = int(body.get("TotalCount") or 0)
        page_jobs = body.get("Jobs") or []
        jobs.extend(page_jobs)
        if not page_jobs:
            break
        page += 1
    if expected is None or len(jobs) != expected:
        raise RuntimeError(f"ListJobs pagination mismatch: {len(jobs)} != {expected}")
    return jobs


def validate_request_map(body: dict[str, Any], run_mode: str) -> None:
    display_name = PROBE_DISPLAY_NAME if run_mode == "graphics_probe" else DISPLAY_NAME
    expected_gpu = "8"
    expected_topology = "1x8"
    if body.get("WorkspaceId") != WORKSPACE_ID or body.get("ResourceId") != RESOURCE_ID:
        raise RuntimeError("workspace or resource mismatch")
    if body.get("DisplayName") != display_name or body.get("Priority") != 7:
        raise RuntimeError("display name or priority mismatch")
    if body.get("JobType") != "PyTorchJob":
        raise RuntimeError("job type mismatch")
    specs = body.get("JobSpecs") or []
    if len(specs) != 1 or specs[0].get("PodCount") != 1:
        raise RuntimeError("request must contain exactly one worker pod")
    if (specs[0].get("ResourceConfig") or {}).get("GPU") != expected_gpu:
        raise RuntimeError("GPU topology mismatch")
    if specs[0].get("RestartPolicy") != "Never":
        raise RuntimeError("worker restart policy mismatch")
    settings = body.get("Settings") or {}
    if "OversoldType" in settings or "OversoldType" in specs[0]:
        raise RuntimeError("oversold scheduling is forbidden")
    if body.get("SpotStrategy") or specs[0].get("ElasticSpotSpecs"):
        raise RuntimeError("spot scheduling is forbidden")
    if settings.get("EnableRDMA") is not False:
        raise RuntimeError("evaluation must not allocate RDMA")
    if (settings.get("Tags") or {}).get("topology") != expected_topology:
        raise RuntimeError("topology tag mismatch")
    envs = body.get("Envs") or {}
    if envs.get("P12_EXPERIMENT_ID") != EXPERIMENT_ID:
        raise RuntimeError("experiment binding mismatch")
    if envs.get("P12_DISPLAY_NAME") != display_name:
        raise RuntimeError("runtime display binding mismatch")
    if envs.get("P12_RUN_MODE") != run_mode:
        raise RuntimeError("runtime mode binding mismatch")


def build_request(body: dict[str, Any], run_mode: str, models: Any) -> Any:
    validate_request_map(body, run_mode)
    request = models.CreateJobRequest().from_map(body)
    request.validate()
    if request.to_map() != body:
        raise RuntimeError("DLC request model round-trip changed the frozen request")
    return request


def duplicate_jobs(jobs: list[dict[str, Any]], request: dict[str, Any]) -> list[dict[str, Any]]:
    expected_envs = request.get("Envs") or {}
    display_name = request["DisplayName"]
    record_root = expected_envs["P12_RECORD_ROOT"]
    run_mode = expected_envs["P12_RUN_MODE"]
    matches: list[dict[str, Any]] = []
    for job in jobs:
        envs = job.get("Envs") or {}
        if (
            job.get("DisplayName") == display_name
            or (
                envs.get("P12_EXPERIMENT_ID") == EXPERIMENT_ID
                and envs.get("P12_RUN_MODE") == run_mode
            )
            or envs.get("P12_RECORD_ROOT") == record_root
        ):
            matches.append(
                {
                    "job_id": job.get("JobId"),
                    "display_name": job.get("DisplayName"),
                    "status": job.get("Status"),
                }
            )
    return matches


def preflight(run_mode: str) -> tuple[str, dict[str, Any], Any, Any, Any, int]:
    eval_commit = validate_inputs(run_mode=run_mode)
    body = request_body(eval_commit, run_mode=run_mode)
    client = load_client()
    from alibabacloud_pai_dlc20201203 import models

    request = build_request(body, run_mode, models)
    jobs = list_jobs(client, models)
    duplicates = duplicate_jobs(jobs, body)
    if duplicates:
        raise RuntimeError(f"refusing duplicate P12 R6 submit: {duplicates}")
    return eval_commit, body, request, client, models, len(jobs)


def submission_paths(run_mode: str) -> tuple[Path, Path, Path, Path]:
    root = PROBE_RECORD_ROOT if run_mode == "graphics_probe" else RECORD_ROOT
    return (
        root / "submission-latch.json",
        root / "create-job-response.json",
        root / "submission-receipt.json",
        root / "submission-state.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    run_mode = "graphics_probe" if args.probe_only else "full_eval"
    target_display = PROBE_DISPLAY_NAME if run_mode == "graphics_probe" else DISPLAY_NAME
    eval_commit, body, request, client, models, listed_jobs = preflight(run_mode)
    latch_path, response_path, receipt_path, state_path = submission_paths(run_mode)
    existing = [
        str(path)
        for path in (latch_path, response_path, receipt_path)
        if path.exists() or path.is_symlink()
    ]
    if existing:
        raise RuntimeError(f"permanent P12 R6 submission evidence already exists: {existing}")
    if args.audit_only:
        result: dict[str, Any] = {
            "schema_version": "fastwam-p12-dlc-eval-audit-v2",
            "mode": "audit-only",
            "run_mode": run_mode,
            "display_name": target_display,
            "evaluation_code_commit": eval_commit,
            "priority": 7,
            "listed_jobs": listed_jobs,
            "duplicate_count": 0,
            "latch_written": False,
            "create_job_called": False,
            "checked_at_utc": utc_now(),
        }
        if run_mode == "full_eval":
            result["graphics_probe_gate"] = validate_probe_terminal()
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    lock_name = "submit-probe.lock" if run_mode == "graphics_probe" else "submit-eval.lock"
    lock = (LOCAL_ROOT / lock_name).open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another P12 R6 DLC submit supervisor is active") from error
    # Re-run every cloud and filesystem gate while holding the local lock.
    eval_commit, body, request, client, models, listed_jobs = preflight(run_mode)
    latch = {
        "schema_version": "fastwam-p12-dlc-eval-permanent-submission-latch-v1",
        "experiment_id": EXPERIMENT_ID,
        "display_name": target_display,
        "run_mode": run_mode,
        "evaluation_code_commit": eval_commit,
        "priority": 7,
        "listed_jobs": listed_jobs,
        "duplicate_count": 0,
        "latched_at_utc": utc_now(),
        "create_job_call_permitted_once": True,
    }
    write_exclusive(latch_path, latch)
    created = client.create_job(request).body.to_map()
    job_id = str(created.get("JobId") or "")
    if not job_id:
        raise RuntimeError("CreateJob returned no JobId")
    response_record = {
        **latch,
        "schema_version": "fastwam-p12-dlc-eval-create-job-response-v1",
        "job_id": job_id,
        "request_id": created.get("RequestId"),
        "recorded_at_utc": utc_now(),
    }
    write_exclusive(response_path, response_record)
    observed = client.get_job(job_id, models.GetJobRequest(need_detail=True)).body.to_map()
    if observed.get("DisplayName") != target_display or observed.get("Priority") != 7:
        raise RuntimeError("submitted P12 R6 job identity mismatch")
    receipt = {
        **response_record,
        "schema_version": "fastwam-p12-dlc-eval-submission-v2",
        "cloud_mutations_called": ["CreateJob"],
        "observed_status": observed.get("Status"),
        "outputs": [str(path) for path in outputs()],
        "requested_topology": "1x8",
        "submitted_at_utc": utc_now(),
        "training_job_id": "dlc19rgpvuxr56b7",
        "create_job_called": True,
        "create_job_call_count": 1,
    }
    if run_mode == "full_eval":
        receipt["graphics_probe_gate"] = validate_probe_terminal()
    write_exclusive(receipt_path, receipt)
    state = {
        "state": "SUBMITTED",
        "detail": job_id,
        "run_mode": run_mode,
        "display_name": target_display,
        "observed_at_utc": utc_now(),
    }
    atomic_json(state_path, state)
    atomic_json(LOCAL_ROOT / "submission-state.json", state)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
