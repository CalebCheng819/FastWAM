#!/usr/bin/env python3
"""Audit or exactly-once submit the dedicated-quota P13 R6 cache job."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGION = "cn-beijing"
WORKSPACE_ID = "270969"
RESOURCE_ID = "quotaksvqq2oh2pg"
PROFILE_PATH = Path("/root/.aliyun/config.json")
DISPLAY_NAME = "fastwam-p13-metric-cache-s42-8g-r6-graphics-probed-dedicated-20260817"
RUN_ID = DISPLAY_NAME
OUTPUT_ROOT = (
    "/oss-chengjuntao/artifacts/"
    "fastwam-placefood-metric-geometry-60x80-s42-v1-"
    "r6-graphics-probed-dedicated-20260817"
)
SOURCE_REVISION = "60de16ef0628d70f58c3349a182c2fe8be3ade2c"
SOURCE_BUNDLE = (
    "/oss-chengjuntao/artifacts/fastwam-p13-runtime-20260817-r6/"
    "FastWAM-p13-r6-source-60de16e.bundle"
)
VULKAN_LOADER = (
    "/cpfs/user/chengjuntao/fastwam-deploy/vulkan-loader-1.3.204/"
    "libvulkan.so.1.3.204"
)
WORKER_COMMAND = (
    "exec /bin/bash /oss-chengjuntao/artifacts/fastwam-p13-runtime-20260817-r6/"
    "cache-worker-r6-graphics-probed-dedicated.sh"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


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


def sdk() -> tuple[Any, Any, Any, Any, Any]:
    from alibabacloud_credentials.client import Client as CredentialClient
    from alibabacloud_credentials.models import Config as CredentialConfig
    from alibabacloud_pai_dlc20201203 import models
    from alibabacloud_pai_dlc20201203.client import Client
    from alibabacloud_tea_openapi.models import Config

    return CredentialClient, CredentialConfig, models, Client, Config


def load_client() -> tuple[Any, Any]:
    CredentialClient, CredentialConfig, models, Client, Config = sdk()
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
    client = Client(
        Config(
            credential=credential,
            region_id=REGION,
            endpoint="pai-dlc.cn-beijing.aliyuncs.com",
        )
    )
    return client, models


def list_jobs(client: Any, models: Any) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    expected: int | None = None
    page = 1
    while expected is None or len(jobs) < expected:
        response = client.list_jobs(
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
            expected = int(response.get("TotalCount") or 0)
        page_jobs = response.get("Jobs") or []
        jobs.extend(page_jobs)
        if not page_jobs:
            break
        page += 1
    if expected is None or len(jobs) != expected:
        raise RuntimeError(f"ListJobs pagination mismatch: {len(jobs)} != {expected}")
    return jobs


def validate_request_map(body: dict[str, Any]) -> None:
    if body.get("WorkspaceId") != WORKSPACE_ID or body.get("ResourceId") != RESOURCE_ID:
        raise RuntimeError("workspace or resource mismatch")
    if body.get("DisplayName") != DISPLAY_NAME:
        raise RuntimeError("display name mismatch")
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", DISPLAY_NAME) is None:
        raise RuntimeError("invalid display name")
    if body.get("JobType") != "PyTorchJob" or body.get("Priority") != 7:
        raise RuntimeError("job protocol mismatch")
    specs = body.get("JobSpecs") or []
    resources = {
        "CPU": "126",
        "GPU": "8",
        "Memory": "960Gi",
        "SharedMemory": "960Gi",
    }
    if len(specs) != 1 or specs[0].get("ResourceConfig") != resources:
        raise RuntimeError("request is not exactly one eight-GPU worker")
    if specs[0].get("PodCount") != 1 or specs[0].get("RestartPolicy") != "Never":
        raise RuntimeError("worker topology mismatch")
    settings = body.get("Settings") or {}
    if "OversoldType" in settings or "OversoldType" in specs[0]:
        raise RuntimeError("oversold scheduling is forbidden for R6")
    if body.get("SpotStrategy") or specs[0].get("ElasticSpotSpecs"):
        raise RuntimeError("spot scheduling is forbidden for R6")
    if settings.get("EnableRDMA") is not False:
        raise RuntimeError("cache job must not enable RDMA")
    if settings.get("AllocateAllRDMADevices") is not False:
        raise RuntimeError("cache job must not allocate RDMA")
    tags = settings.get("Tags") or {}
    if tags.get("scheduler") != "dedicated-quota":
        raise RuntimeError("dedicated scheduler tag missing")
    if tags.get("graphics_gate") != "real-environment-multi-profile":
        raise RuntimeError("R6 graphics gate tag missing")
    envs = body.get("Envs") or {}
    expected_envs = {
        "RUN_ID": RUN_ID,
        "FASTWAM_P13_CACHE_OUTPUT_ROOT": OUTPUT_ROOT,
        "FASTWAM_P13_CODE_REVISION": SOURCE_REVISION,
        "FASTWAM_P13_SOURCE_BUNDLE": SOURCE_BUNDLE,
        "FASTWAM_P13_VULKAN_LOADER": VULKAN_LOADER,
    }
    if {key: envs.get(key) for key in expected_envs} != expected_envs:
        raise RuntimeError("run, output, source, or graphics binding mismatch")
    if body.get("UserCommand") != WORKER_COMMAND:
        raise RuntimeError("R6 worker command mismatch")


def build_request(body: dict[str, Any], models: Any) -> Any:
    validate_request_map(body)
    request = models.CreateJobRequest().from_map(body)
    request.validate()
    if request.to_map() != body:
        raise RuntimeError("DLC request serialization roundtrip mismatch")
    return request


def duplicate_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = []
    for job in jobs:
        envs = job.get("Envs") or {}
        if (
            job.get("DisplayName") == DISPLAY_NAME
            or envs.get("RUN_ID") == RUN_ID
            or envs.get("FASTWAM_P13_CACHE_OUTPUT_ROOT") == OUTPUT_ROOT
        ):
            matches.append(
                {
                    "job_id": job.get("JobId"),
                    "display_name": job.get("DisplayName"),
                    "status": job.get("Status"),
                }
            )
    return matches


def preflight(body: dict[str, Any], client: Any, models: Any) -> tuple[Any, int]:
    request = build_request(body, models)
    output = Path(OUTPUT_ROOT)
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"R6 versioned output already exists: {output}")
    jobs = list_jobs(client, models)
    duplicates = duplicate_jobs(jobs)
    if duplicates:
        raise RuntimeError(f"duplicate P13 R6 launch target: {duplicates}")
    return request, len(jobs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--latch", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    body = json.loads(args.request.read_text(encoding="utf-8"))
    client, models = load_client()
    request, listed_jobs = preflight(body, client, models)
    base = {
        "schema": "fastwam-p13-dlc-submit-r6-v1",
        "display_name": DISPLAY_NAME,
        "run_id": RUN_ID,
        "output_root": OUTPUT_ROOT,
        "source_revision": SOURCE_REVISION,
        "priority": 7,
        "scheduler": "dedicated-quota",
        "graphics_gate": "real-environment-multi-profile",
        "listed_jobs": listed_jobs,
        "duplicate_count": 0,
    }
    if args.audit_only:
        print(
            json.dumps(
                {
                    **base,
                    "mode": "audit-only",
                    "checked_at_utc": utc_now(),
                    "latch_written": False,
                    "create_job_called": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    latch = {
        **base,
        "schema": "fastwam-p13-permanent-submission-latch-v1",
        "latched_at_utc": utc_now(),
        "create_job_call_permitted_once": True,
    }
    write_exclusive(args.latch, latch)
    response = client.create_job(request)
    job_id = str(response.body.job_id)
    request_id = str(response.body.request_id)
    response_record = {
        **base,
        "schema": "fastwam-p13-create-job-response-v1",
        "recorded_at_utc": utc_now(),
        "job_id": job_id,
        "request_id": request_id,
    }
    write_exclusive(args.response, response_record)
    job = client.get_job(job_id, models.GetJobRequest(need_detail=True)).body.to_map()
    if job.get("DisplayName") != DISPLAY_NAME or job.get("Priority") != 7:
        raise RuntimeError("submitted DLC job identity mismatch")
    result = {
        **response_record,
        "schema": "fastwam-p13-submission-receipt-v1",
        "observed_at_utc": utc_now(),
        "observed_status": job.get("Status"),
        "create_job_called": True,
        "create_job_call_count": 1,
    }
    write_exclusive(args.receipt, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "credentials_printed": False,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
