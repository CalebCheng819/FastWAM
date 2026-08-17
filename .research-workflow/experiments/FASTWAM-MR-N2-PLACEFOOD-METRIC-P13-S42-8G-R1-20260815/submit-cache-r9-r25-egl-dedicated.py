#!/usr/bin/env python3
"""Audit or exactly-once submit the R25-EGL P13 R9 cache job."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


BASE_PATH = Path(__file__).with_name("submit-cache-r6-graphics-probed-dedicated.py")
SPEC = importlib.util.spec_from_file_location("p13_cache_r6_base", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

DISPLAY_NAME = "fastwam-p13-metric-cache-s42-8g-r9-r25-egl-dedicated-20260817"
RUN_ID = DISPLAY_NAME
OUTPUT_ROOT = (
    "/oss-chengjuntao/artifacts/"
    "fastwam-placefood-metric-geometry-60x80-s42-v1-"
    "r9-r25-egl-dedicated-20260817"
)
SOURCE_REVISION = "f8ac674b27efcb2e1b937c2a8e2b121045321409"
SOURCE_BUNDLE = (
    "/oss-chengjuntao/artifacts/fastwam-p13-runtime-20260817-r9-r25-egl/"
    "FastWAM-p13-r9-source-f8ac674.bundle"
)
VULKAN_LOADER = BASE.VULKAN_LOADER
GRAPHICS_GATE_TAG = "r25-egl-pyopengl-before-real-environment"
WORKER_COMMAND = (
    "exec /bin/bash "
    "/oss-chengjuntao/artifacts/fastwam-p13-runtime-20260817-r9-r25-egl/"
    "cache-worker-r9-r25-egl-dedicated.sh"
)

for name in (
    "DISPLAY_NAME",
    "RUN_ID",
    "OUTPUT_ROOT",
    "SOURCE_REVISION",
    "SOURCE_BUNDLE",
    "VULKAN_LOADER",
    "GRAPHICS_GATE_TAG",
    "WORKER_COMMAND",
):
    setattr(BASE, name, globals()[name])

validate_request_map = BASE.validate_request_map
duplicate_jobs = BASE.duplicate_jobs
preflight = BASE.preflight
load_client = BASE.load_client
utc_now = BASE.utc_now
write_exclusive = BASE.write_exclusive


def main() -> None:
    args = BASE.parse_args()
    body = json.loads(args.request.read_text(encoding="utf-8"))
    client, models = load_client()
    request, listed_jobs = preflight(body, client, models)
    base = {
        "schema": "fastwam-p13-dlc-submit-r9-v1",
        "display_name": DISPLAY_NAME,
        "run_id": RUN_ID,
        "output_root": OUTPUT_ROOT,
        "source_revision": SOURCE_REVISION,
        "priority": 7,
        "scheduler": "dedicated-quota",
        "graphics_gate": "r25-egl-pyopengl-before-real-environment",
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
