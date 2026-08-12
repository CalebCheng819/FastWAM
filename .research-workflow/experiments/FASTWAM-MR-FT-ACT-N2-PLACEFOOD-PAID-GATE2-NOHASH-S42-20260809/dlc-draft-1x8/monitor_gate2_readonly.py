#!/usr/bin/env python3
"""Read-only PAI DLC inspection without importing any frozen FastWAM source."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROFILE = Path("/root/.aliyun/config.json")
REGION = "cn-beijing"
ENDPOINT = "pai-dlc.cn-beijing.aliyuncs.com"
JOB_ID_RE = re.compile(r"^dlc[a-z0-9]+$")


def require_isolated_no_bytecode_runtime() -> None:
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode:
        raise RuntimeError("read-only monitor must be invoked with python -B -I")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Read one DLC job or pod log through the installed SDK; never mutate cloud state."
    )
    result.add_argument("--job-id", required=True)
    result.add_argument("--pod-id")
    result.add_argument("--pod-uid")
    result.add_argument("--max-lines", type=int, default=200)
    return result


def read_profile() -> dict[str, Any]:
    info = PROFILE.lstat()
    if PROFILE.is_symlink() or not PROFILE.is_file() or info.st_nlink != 1:
        raise RuntimeError("Alibaba Cloud profile must be one regular non-linked file")
    payload = json.loads(PROFILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Alibaba Cloud profile is not a JSON object")
    return payload


def load_client() -> tuple[Any, Any, Any]:
    from alibabacloud_credentials.client import Client as CredentialClient
    from alibabacloud_credentials.models import Config as CredentialConfig
    from alibabacloud_pai_dlc20201203 import models
    from alibabacloud_pai_dlc20201203.client import Client
    from alibabacloud_tea_openapi.models import Config
    from alibabacloud_tea_util.models import RuntimeOptions

    profile = read_profile()
    current = profile.get("current")
    selected = next(
        (
            item
            for item in profile.get("profiles", [])
            if isinstance(item, dict) and item.get("name") == current
        ),
        None,
    )
    if (
        not isinstance(selected, dict)
        or selected.get("mode") != "CredentialsURI"
        or not isinstance(selected.get("credentials_uri"), str)
        or not selected["credentials_uri"]
    ):
        raise RuntimeError("active Alibaba Cloud profile must use CredentialsURI")
    credential = CredentialClient(
        CredentialConfig(
            type="credentials_uri", credentials_uri=selected["credentials_uri"]
        )
    )
    client = Client(
        Config(credential=credential, region_id=REGION, endpoint=ENDPOINT)
    )
    runtime = RuntimeOptions(
        autoretry=False,
        max_attempts=1,
        connect_timeout=10000,
        read_timeout=30000,
    )
    return client, models, runtime


def main() -> None:
    require_isolated_no_bytecode_runtime()
    args = parser().parse_args()
    if JOB_ID_RE.fullmatch(args.job_id) is None:
        raise RuntimeError("invalid DLC job id")
    if not 1 <= args.max_lines <= 5000:
        raise RuntimeError("max-lines must be between 1 and 5000")
    if (args.pod_id is None) != (args.pod_uid is None):
        raise RuntimeError("pod-id and pod-uid must be provided together")

    client, models, runtime = load_client()
    if args.pod_id is None:
        response = client.get_job_with_options(
            args.job_id,
            models.GetJobRequest(need_detail=True),
            {},
            runtime,
        )
    else:
        response = client.get_pod_logs_with_options(
            args.job_id,
            args.pod_id,
            models.GetPodLogsRequest(
                max_lines=args.max_lines,
                pod_uid=args.pod_uid,
            ),
            {},
            runtime,
        )
    print(json.dumps(response.body.to_map(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
