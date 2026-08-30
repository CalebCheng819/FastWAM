#!/usr/bin/env python3
"""Audit and submit the frozen joint-safe table11 3x8 DLC request exactly once."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import pathlib
import shlex
import stat
import subprocess
import sys
import tempfile

from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_credentials.models import Config as CredentialConfig
from alibabacloud_pai_dlc20201203 import models
from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_tea_openapi.models import Config


RUN_ID = "fastwam-table11safe-vg1h1gau1-scratch50k-s42-24g-r1-20260829"
ATTEMPT_ID = "attempt-r1-20260829"
WORKSPACE_ID = "270969"
RESOURCE_ID = "quotaksvqq2oh2pg"
COMMIT = "8ac5a8dc707084db406f99874d4123749401819a"
IMAGE = (
    "dsw-registry-vpc.cn-beijing.cr.aliyuncs.com/pai/"
    "pytorch:2.7.1-gpu-py310-cu128-ubuntu22.04-3995b779-1764350887"
)
LAUNCH_ROOT = pathlib.Path(f"/oss-chengjuntao/artifacts/{RUN_ID}-launch-control")
DRY_REQUEST = LAUNCH_ROOT / "rendered-request-r1.json"
PRELAUNCH = LAUNCH_ROOT / "prelaunch-formal-3x8-r1.json"
AUDIT_RECORD = LAUNCH_ROOT / "submission-formal-3x8-r1-audit.json"
RECEIPT = LAUNCH_ROOT / "submission-formal-3x8-r1-receipt.json"
REAL_DATA_PREFLIGHT = LAUNCH_ROOT / "preflight-terminal-reconciliation-r6.json"
OUTPUT_DIR = f"/oss-chengjuntao/artifacts/{RUN_ID}"
REAL_DATA_PREFLIGHT_RUN_ID = (
    "fastwam-table11safe-vg1h1gau1-scratch-preflight-s42-8g-r6-20260830"
)
REAL_DATA_PREFLIGHT_ATTEMPT_ID = "attempt-r6-20260830"
REAL_DATA_PREFLIGHT_OUTPUT_DIR = (
    f"/oss-chengjuntao/artifacts/{REAL_DATA_PREFLIGHT_RUN_ID}"
)
BUNDLE = str(LAUNCH_ROOT / "fastwam-table11safe-scratch50k-source-r7-20260830.bundle")
DATASET_ROOT = (
    "/oss-chengjuntao/robofactory/table/"
    "robofactory-table-11task-200each-h256-2g-stateful-safe-r3-20260827/tasks"
)
ASSET_ROOT = (
    "/oss-chengjuntao/fastwam-assets/robofactory/"
    "table11-200each-h256-stateful-safe-r3-s42"
)
STATS_PATH = f"{ASSET_ROOT}/stats/train-stats.json"
TEXT_CACHE_DIR = f"{ASSET_ROOT}/text-embeds"
GAUSSIAN_CACHE_DIR = (
    f"{ASSET_ROOT}/gaussian/"
    "compact-s42-13x28x40-fp16-meanalpha-direct-v1"
)
MODEL_CACHE_ROOT = (
    "/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/model-cache"
)
VAE_PATH = (
    f"{MODEL_CACHE_ROOT}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/"
    "Wan2.2_VAE.safetensors"
)
SOURCE_WEIGHT = (
    "/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/"
    "yuanty-fastwam-139eebb6d90cdd9bdbbe465f72c6edc9ad5a518a/"
    "libero_uncond_2cam224.pt"
)
OFFLINE_ROOT = (
    "/oss-chengjuntao/artifacts/"
    "fastwam-offline-env-v9-00c0887-cp310-cu128-20260803T0005Z"
)
OFFLINE_COMMIT = "00c0887118e647acf2ec7047dffa26a4231adc9e"
SDK_PYTHON = (
    "/mnt/workspace/tools/pai-control-py311/"
    "20260817-credentials1.0.10-dlc1.9.2/bin/python"
)
EXPECTED_ENVS = {
    "FASTWAM_ERDMA_BUNDLE_ROOT": "/oss-chengjuntao/artifacts/erdma-userspace-56.2-1.0.3",
    "FASTWAM_ERDMA_EXPECTED_VERSION": "56.2-1.0.3",
    "FASTWAM_OFFLINE_CODE_COMMIT": OFFLINE_COMMIT,
    "FASTWAM_OFFLINE_ENV_BASE_PYTHON": "/usr/local/bin/python3.10",
    "FASTWAM_OFFLINE_ENV_CACHE_HELPER_SHA256": (
        "89dc9d7302f2edc1320b5f08f0516d5d2e9c6a176705642cf2f57756a1ae22ae"
    ),
    "FASTWAM_OFFLINE_ENV_CACHE_ROOT": "/tmp/fastwam-offline-env-cache",
    "FASTWAM_OFFLINE_ENV_MANIFEST": f"{OFFLINE_ROOT}/SHA256SUMS",
    "FASTWAM_OFFLINE_ENV_MANIFEST_SHA256": (
        "b740e7224ad38628c12347ff0d36cb85dea45095f335ec032a52f07fcade7ee5"
    ),
    "FASTWAM_OFFLINE_ENV_RUNTIME_LOCK_SHA256": (
        "d495f1a1192ced91edd7df2794a94fe0ffb67526a279570d5cf3649d59c0d360"
    ),
    "FASTWAM_OFFLINE_ENV_SOURCE_BUNDLE_RELATIVE_PATH": (
        "fastwam-00c0887118e647acf2ec7047dffa26a4231adc9e.bundle"
    ),
    "FASTWAM_OFFLINE_ENV_SOURCE_ROOT": OFFLINE_ROOT,
    "FASTWAM_OFFLINE_ENV_STALE_LOCK_SECONDS": "7200",
    "FASTWAM_OFFLINE_ENV_VENV_ROOT": "/tmp/fastwam-offline-env-venvs",
    "FASTWAM_OFFLINE_ENV_WAIT_TIMEOUT": "7200",
    "FASTWAM_PREFLIGHT_OUTER_TIMEOUT": "7260",
    "FASTWAM_PREFLIGHT_REQUIRE_ERDMA": "1",
    "FASTWAM_PREFLIGHT_TIMEOUT": "7200",
    "FASTWAM_SOURCE_CHECKOUT_ROOT": "/tmp/fastwam-source-checkouts",
    "FASTWAM_TABLE11_ATTEMPT_ID": ATTEMPT_ID,
    "FASTWAM_TABLE11_RUN_MODE": "formal",
    "FASTWAM_TABLE11_BOOTSTRAP_SCRIPT": (
        f"{OFFLINE_ROOT}/source-snapshot/scripts/bootstrap_offline_training_env.sh"
    ),
    "FASTWAM_TABLE11_CODE_COMMIT": COMMIT,
    "FASTWAM_TABLE11_DATASET_ROOT": DATASET_ROOT,
    "FASTWAM_TABLE11_EXPECTED_H5_FILES": "11",
    "FASTWAM_TABLE11_GAUSSIAN_CACHE_DIR": GAUSSIAN_CACHE_DIR,
    "FASTWAM_TABLE11_LOCAL_SOURCE_ROOT": "/tmp/fastwam-table11-source-checkouts",
    "FASTWAM_TABLE11_MODEL_CACHE_ROOT": MODEL_CACHE_ROOT,
    "FASTWAM_TABLE11_OUTPUT_DIR": OUTPUT_DIR,
    "FASTWAM_TABLE11_OUTPUT_RESERVATION_TIMEOUT": "300",
    "FASTWAM_TABLE11_PROVENANCE_MODE": "stat_cmp",
    "FASTWAM_TABLE11_SOURCE_BUNDLE": BUNDLE,
    "FASTWAM_TABLE11_SOURCE_WEIGHT": SOURCE_WEIGHT,
    "FASTWAM_TABLE11_SOURCE_WEIGHT_BYTES": "12041735140",
    "FASTWAM_TABLE11_STATS_PATH": STATS_PATH,
    "FASTWAM_TABLE11_TEXT_CACHE_DIR": TEXT_CACHE_DIR,
    "FASTWAM_TABLE11_VAE_PATH": VAE_PATH,
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,NET",
    "NCCL_IB_HCA": "erdma",
    "NPROC_PER_NODE": "8",
    "RUN_ID": RUN_ID,
}
EXPECTED_SETTINGS = {
    "AllocateAllRDMADevices": True,
    "EnableCPUAffinity": False,
    "EnableErrorMonitoringInAIMaster": False,
    "EnableOssAppend": False,
    "EnableRDMA": True,
    "EnableSanityCheck": False,
    "Tags": {
        "experiment": "TABLE11SAFE-VG1H1GAU1-SCRATCH50K",
        "initialization": "official-generic-pretrained-model-weights",
        "optimizer": "fresh",
        "provenance": "stat-cmp-no-new-hash",
        "schedule": "optimizer-0-to-50000-save-5000",
        "topology": "3x8-world24",
    },
}
EXPECTED_DOCUMENT_KEYS = {
    "batch_contract",
    "dry_run",
    "endpoint",
    "launcher_payload_base64",
    "launcher_source",
    "operation",
    "provenance_contract",
    "region",
    "request",
    "sdk_python",
    "submission_not_performed",
}
EXPECTED_BODY_KEYS = {
    "Accessibility",
    "CustomEnvs",
    "DataSources",
    "Description",
    "DisplayName",
    "Envs",
    "JobMaxRunningTimeMinutes",
    "JobSpecs",
    "JobType",
    "Priority",
    "ResourceId",
    "Settings",
    "SuccessPolicy",
    "UserCommand",
    "WorkspaceId",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_regular_bytes(path: pathlib.Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        current = os.stat(path, follow_symlinks=False)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        require(
            identity
            == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ),
            f"file changed while reading: {path}",
        )
        require(stat.S_ISREG(current.st_mode), f"pathname is not regular: {path}")
        require(
            identity
            == (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            ),
            f"pathname changed while reading: {path}",
        )
        return b"".join(chunks)
    finally:
        os.close(fd)


def read_json(path: pathlib.Path) -> dict:
    try:
        value = json.loads(read_regular_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"not canonical UTF-8 JSON: {path}") from exc
    require(isinstance(value, dict), f"JSON root is not a mapping: {path}")
    return value


def fsync_parent(path: pathlib.Path) -> None:
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_exclusive(path: pathlib.Path, payload: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    fsync_parent(path)


def replace_receipt(path: pathlib.Path, payload: dict) -> None:
    current = os.stat(path, follow_symlinks=False)
    require(stat.S_ISREG(current.st_mode), "receipt pathname is not regular")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = pathlib.Path(stream.name)
            os.chmod(temporary, 0o600)
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        fsync_parent(path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def credentials_uri() -> str:
    document = json.loads(pathlib.Path("/root/.aliyun/config.json").read_text())
    profiles = document["profiles"]
    current = document["current"]
    if isinstance(profiles, dict):
        profile = profiles[current]
    else:
        profile = next(item for item in profiles if item.get("name") == current)
    uri = (
        profile.get("credentials_uri")
        or profile.get("CredentialsURI")
        or profile.get("credentialsUri")
    )
    require(bool(uri), "current Alibaba Cloud profile has no credentials URI")
    return str(uri)


def client() -> Client:
    credential = CredentialClient(
        CredentialConfig(type="credentials_uri", credentials_uri=credentials_uri())
    )
    return Client(
        Config(
            credential=credential,
            region_id="cn-beijing",
            endpoint="pai-dlc.cn-beijing.aliyuncs.com",
        )
    )


def job_id(job: dict) -> str:
    value = job.get("JobId") or job.get("job_id")
    require(bool(value), f"ListJobs entry has no JobId: {job}")
    return str(value)


def display_name(job: dict) -> str:
    return str(job.get("DisplayName") or job.get("display_name") or "")


def all_jobs_snapshot(dlc: Client) -> list[dict]:
    jobs = []
    seen = set()
    expected_total = None
    page = 1
    page_size = 100
    while True:
        require(page <= 1000, "ListJobs pagination exceeded 1000 pages")
        response = dlc.list_jobs(
            models.ListJobsRequest(
                workspace_id=WORKSPACE_ID,
                resource_id=RESOURCE_ID,
                page_number=page,
                page_size=page_size,
            )
        ).body.to_map()
        require("TotalCount" in response, "ListJobs omitted TotalCount")
        total = int(response["TotalCount"])
        require(total >= 0, "ListJobs returned negative TotalCount")
        if expected_total is None:
            expected_total = total
        else:
            require(total == expected_total, "ListJobs TotalCount drifted")
        batch = response.get("Jobs") or []
        require(isinstance(batch, list), "ListJobs Jobs is not a list")
        if len(jobs) < expected_total:
            require(bool(batch), "ListJobs ended before frozen TotalCount")
        for job in batch:
            require(isinstance(job, dict), "ListJobs entry is not a mapping")
            identifier = job_id(job)
            require(identifier not in seen, f"ListJobs repeated JobId {identifier}")
            seen.add(identifier)
            jobs.append(job)
        require(len(jobs) <= expected_total, "ListJobs exceeded frozen TotalCount")
        if len(jobs) == expected_total:
            return jobs
        page += 1


def verify_no_duplicate_job(dlc: Client, run_id: str = RUN_ID) -> list[dict]:
    first = all_jobs_snapshot(dlc)
    second = all_jobs_snapshot(dlc)
    identity = lambda values: sorted((job_id(job), display_name(job)) for job in values)
    require(identity(first) == identity(second), "ListJobs identity snapshot drifted")
    matches = [job for job in second if display_name(job) == run_id]
    require(not matches, f"refuse duplicate CreateJob: {matches}")
    return second


def git_stdout(*arguments: str, text: bool = False) -> bytes | str:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    ).stdout


def validate_embedded_launcher(
    document: dict,
    body: dict,
    *,
    expected_machines: int = 3,
    expected_world: int = 24,
    bundle: str = BUNDLE,
    commit: str = COMMIT,
    require_formal_schedule: bool = True,
) -> None:
    encoded = document["launcher_payload_base64"]
    require(isinstance(encoded, str) and encoded, "missing launcher payload")
    try:
        launcher = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise RuntimeError("launcher payload is not canonical base64") from exc
    require(base64.b64encode(launcher).decode("ascii") == encoded, "noncanonical base64")
    launcher.decode("utf-8")
    wrapper = (
        "set -euo pipefail; embedded=/tmp/fastwam-table11-outer-launcher.$$; "
        "umask 077; printf '%s' "
        f"{encoded} | base64 --decode > \"$embedded\"; "
        "exec /bin/bash \"$embedded\""
    )
    require(
        shlex.split(body["UserCommand"]) == ["/bin/bash", "-c", wrapper],
        "UserCommand differs from the frozen embedded launcher wrapper",
    )
    require(body["UserCommand"].count(encoded) == 1, "payload occurrence count drift")
    with tempfile.TemporaryDirectory(prefix="table11-submit-audit.") as temp:
        checkout = pathlib.Path(temp) / "source"
        git_stdout("clone", "--quiet", "--no-checkout", bundle, str(checkout))
        git_stdout("-C", str(checkout), "bundle", "verify", bundle)
        expected = git_stdout(
            "-C", str(checkout), "show", f"{commit}:scripts/launch_table11safe_3x8_dlc.sh"
        )
    require(launcher == expected, "embedded launcher differs from frozen commit")
    fragments = [
        b'[[ "${GPUS_PER_NODE}" == "8" ]]',
        f"EXPECTED_MACHINES={expected_machines}".encode("ascii"),
        f"EXPECTED_WORLD={expected_world}".encode("ascii"),
        b'--num_machines "${EXPECTED_MACHINES}"',
        b'--num_processes "${EXPECTED_WORLD}"',
        b"--deepspeed_multinode_launcher standard",
        b"fastwam_run_global_allreduce_preflight",
        b'export FASTWAM_ATTEMPT_ID="${ATTEMPT_ID}"',
    ]
    if require_formal_schedule:
        fragments.append(b"checkpoints=5000..50000/5000")
    for fragment in fragments:
        require(fragment in launcher, f"launcher omitted contract fragment: {fragment!r}")


def validate_assets(
    output_dir: str = OUTPUT_DIR,
    *,
    bundle: str = BUNDLE,
    commit: str = COMMIT,
) -> dict:
    require(not os.path.lexists(output_dir), "canonical output already exists")
    bundle_stat = os.stat(bundle, follow_symlinks=False)
    require(stat.S_ISREG(bundle_stat.st_mode), "source bundle is not regular")
    heads = str(git_stdout("bundle", "list-heads", bundle, text=True)).splitlines()
    require(
        heads == [f"{commit} HEAD"],
        f"source bundle heads drift: {heads}",
    )
    h5_files = list(pathlib.Path(DATASET_ROOT).rglob("*.h5"))
    require(len(h5_files) == 11, f"expected 11 H5 files, got {len(h5_files)}")
    require(pathlib.Path(STATS_PATH).is_file(), "stats file is absent")
    text_files = [path for path in pathlib.Path(TEXT_CACHE_DIR).rglob("*") if path.is_file()]
    require(len(text_files) >= 11, "text cache is incomplete")
    gaussian = pathlib.Path(GAUSSIAN_CACHE_DIR)
    complete = read_json(gaussian / "COMPLETE")
    manifest = read_json(gaussian / "manifest.json")
    require((gaussian / "selection.jsonl").is_file(), "Gaussian selection is absent")
    require(complete.get("complete") is True, "Gaussian cache is not terminal")
    require(int(manifest.get("total_frames", -1)) == 89977, "Gaussian frame count drift")
    require(
        manifest.get("derivation", {}).get("source") == "direct-teacher-forward-index-v1",
        "Gaussian derivation drift",
    )
    weight = os.stat(SOURCE_WEIGHT, follow_symlinks=False)
    require(stat.S_ISREG(weight.st_mode) and weight.st_size == 12041735140, "weight drift")
    for path in (
        pathlib.Path(VAE_PATH),
        pathlib.Path(f"{OFFLINE_ROOT}/SHA256SUMS"),
        pathlib.Path(
            f"{OFFLINE_ROOT}/source-snapshot/scripts/bootstrap_offline_training_env.sh"
        ),
        pathlib.Path(
            f"{OFFLINE_ROOT}/fastwam-{OFFLINE_COMMIT}.bundle"
        ),
        pathlib.Path("/oss-chengjuntao/artifacts/erdma-userspace-56.2-1.0.3"),
    ):
        require(path.exists(), f"required runtime asset is absent: {path}")
    return {
        "bundle_bytes": bundle_stat.st_size,
        "dataset_h5_files": len(h5_files),
        "text_cache_files": len(text_files),
        "gaussian_total_frames": int(manifest["total_frames"]),
        "source_weight_bytes": weight.st_size,
    }


def validate_prelaunch() -> dict:
    prelaunch = read_json(PRELAUNCH)
    require(prelaunch.get("schema_version") == 1, "prelaunch schema drift")
    require(prelaunch.get("status") == "READY_TO_SUBMIT", "prelaunch is not ready")
    require(prelaunch.get("ready_to_submit") is True, "ready_to_submit is not true")
    require(prelaunch.get("create_job_called") is False, "prelaunch says CreateJob was called")
    require(prelaunch.get("run_id") == RUN_ID, "prelaunch run_id drift")
    require(prelaunch.get("attempt_id") == ATTEMPT_ID, "prelaunch attempt drift")
    require(prelaunch.get("code_commit") == COMMIT, "prelaunch commit drift")
    require(prelaunch.get("source_bundle") == BUNDLE, "prelaunch bundle drift")
    require(prelaunch.get("request_path") == str(DRY_REQUEST), "prelaunch request drift")
    require(prelaunch.get("output_dir") == OUTPUT_DIR, "prelaunch output drift")
    real = prelaunch.get("real_data_preflight") or {}
    require(real.get("status") == "PASS", "real-data preflight did not pass")
    require(real.get("path") == str(REAL_DATA_PREFLIGHT), "real-data preflight path drift")
    terminal = read_json(REAL_DATA_PREFLIGHT)
    require(
        terminal.get("schema")
        == "fastwam-table11safe-preflight-terminal-reconciliation-v1",
        "preflight reconciliation schema drift",
    )
    require(
        terminal.get("run_id") == REAL_DATA_PREFLIGHT_RUN_ID,
        "preflight reconciliation run_id drift",
    )
    require(
        terminal.get("attempt_id") == REAL_DATA_PREFLIGHT_ATTEMPT_ID,
        "preflight reconciliation attempt drift",
    )
    require(terminal.get("code_commit") == COMMIT, "preflight reconciliation commit drift")
    require(terminal.get("source_bundle") == BUNDLE, "preflight reconciliation bundle drift")
    require(
        terminal.get("output_dir") == REAL_DATA_PREFLIGHT_OUTPUT_DIR,
        "preflight reconciliation output drift",
    )
    require(
        terminal.get("conclusion")
        == {
            "status": "PASS",
            "initialization": "official-generic-pretrained-model-weights",
            "optimizer": "fresh",
            "scheduler": "fresh",
            "initial_global_step": 0,
            "final_global_step": 1,
            "optimizer_steps_this_run": 1,
        },
        "preflight terminal reconciliation drift",
    )
    scheduler = terminal.get("scheduler_terminal") or {}
    require(
        scheduler.get("status") == "Succeeded",
        "preflight scheduler terminal status drift",
    )
    require(bool(scheduler.get("job_id")), "preflight scheduler job_id absent")
    output = terminal.get("output_validation") or {}
    output_terminal = output.get("terminal") or {}
    require(
        output_terminal.get("status") == "PASS"
        and output_terminal.get("run_id") == REAL_DATA_PREFLIGHT_RUN_ID
        and output_terminal.get("attempt_id") == REAL_DATA_PREFLIGHT_ATTEMPT_ID
        and output_terminal.get("initialization")
        == "official-generic-pretrained-model-weights"
        and output_terminal.get("optimizer") == "fresh"
        and output_terminal.get("scheduler") == "fresh"
        and output_terminal.get("initial_global_step") == 0
        and output_terminal.get("final_global_step") == 1
        and output_terminal.get("optimizer_steps_this_run") == 1
        and output_terminal.get("world_size") == 8,
        "preflight output validation drift",
    )
    return prelaunch


def validate_document(document: dict) -> dict:
    require(set(document) == EXPECTED_DOCUMENT_KEYS, "dry request document keys drift")
    require(document["dry_run"] is True, "dry_run must be true")
    require(document["submission_not_performed"] is True, "submission marker drift")
    require(document["operation"] == "CreateJob", "operation drift")
    require(document["endpoint"] == "pai-dlc.cn-beijing.aliyuncs.com", "endpoint drift")
    require(document["region"] == "cn-beijing", "region drift")
    require(document["sdk_python"] == SDK_PYTHON, "SDK Python drift")
    require(
        document["launcher_source"]
        == {
            "bundle": BUNDLE,
            "code_commit": COMMIT,
            "path": "scripts/launch_table11safe_3x8_dlc.sh",
        },
        "launcher source drift",
    )
    require(
        document["provenance_contract"]
        == {
            "mode": "stat_cmp",
            "new_hashes": False,
            "records": [
                "path",
                "bytes",
                "mtime",
                "count",
                "run_id",
                "attempt_id",
                "world_size",
            ],
        },
        "provenance contract drift",
    )
    require(
        document["batch_contract"]
        == {
            "reference_global_batch": 24,
            "replica_global_batch": 24,
            "micro_batch_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "optimizer_updates": 50000,
            "sample_budget_equivalent": True,
        },
        "batch contract drift",
    )
    body = document["request"]
    require(isinstance(body, dict) and set(body) == EXPECTED_BODY_KEYS, "body keys drift")
    require(body["DisplayName"] == RUN_ID, "DisplayName drift")
    require(body["WorkspaceId"] == WORKSPACE_ID, "WorkspaceId drift")
    require(body["ResourceId"] == RESOURCE_ID, "ResourceId drift")
    require(body["JobType"] == "PyTorchJob", "JobType drift")
    require(body["SuccessPolicy"] == "AllWorkers", "SuccessPolicy drift")
    require(body["Accessibility"] == "PRIVATE", "Accessibility drift")
    require(body["Priority"] == 7, "Priority must be 7")
    require(body["JobMaxRunningTimeMinutes"] == 20160, "max runtime drift")
    require(body["CustomEnvs"] == [], "CustomEnvs drift")
    require(
        body["DataSources"]
        == [
            {
                "DataSourceId": "d-n7rly4fll0q2z6v91h",
                "MountAccess": "RW",
                "MountPath": "/oss-chengjuntao",
            }
        ],
        "datasource contract drift",
    )
    require(body["Settings"] == EXPECTED_SETTINGS, "RDMA/settings drift")
    require(body["Envs"] == EXPECTED_ENVS, "environment contract drift")
    require(
        body["Description"]
        == (
            "Joint-safe RoboFactory table11 VG1H1GAU1 scratch-from-generic-base training: "
            "optimizer steps 0 to 50000, 50000 fresh-optimizer updates, "
            "3 workers x 8 GPUs, world-24 global batch"
        ),
        "Description drift",
    )
    require(
        body["JobSpecs"]
        == [
            {
                "ElasticSpotSpecs": [],
                "Image": IMAGE,
                "LocalMountSpecs": [],
                "PodCount": 3,
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
        "3x8 JobSpec drift",
    )
    validate_embedded_launcher(document, body)
    return body


def base_record(status: str) -> dict:
    return {
        "schema_version": 1,
        "status": status,
        "run_id": RUN_ID,
        "display_name": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "workspace_id": WORKSPACE_ID,
        "resource_id": RESOURCE_ID,
        "request_path": str(DRY_REQUEST),
        "prelaunch_manifest": str(PRELAUNCH),
        "output_dir": OUTPUT_DIR,
        "source_bundle": BUNDLE,
        "code_commit": COMMIT,
    }


def record_failure(payload: dict) -> None:
    try:
        replace_receipt(RECEIPT, payload)
    except BaseException as receipt_exc:
        payload["receipt_update_error_type"] = type(receipt_exc).__name__
        payload["receipt_update_error"] = str(receipt_exc)
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dry_request", type=pathlib.Path)
    parser.add_argument("prelaunch", type=pathlib.Path)
    parser.add_argument("receipt", type=pathlib.Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    require(sys.flags.optimize == 0, "optimized Python is forbidden")
    require(os.path.realpath(sys.executable) == os.path.realpath(SDK_PYTHON), "wrong SDK Python")
    require(args.dry_request == DRY_REQUEST, "unexpected dry request path")
    require(args.prelaunch == PRELAUNCH, "unexpected prelaunch path")
    require(args.receipt == RECEIPT, "unexpected receipt path")
    require(not os.path.lexists(RECEIPT), f"receipt already exists: {RECEIPT}")
    if args.audit_only:
        require(not os.path.lexists(AUDIT_RECORD), f"audit record already exists: {AUDIT_RECORD}")
    else:
        audit = read_json(AUDIT_RECORD)
        require(audit.get("status") == "AUDIT_ONLY_PASS_CREATE_JOB_NOT_CALLED", "audit did not pass")
        require(audit.get("create_job_called") is False, "audit claims CreateJob was called")

    document = read_json(DRY_REQUEST)
    body = validate_document(document)
    prelaunch = validate_prelaunch()
    assets = validate_assets()
    dlc = client()
    jobs = verify_no_duplicate_job(dlc)
    request = models.CreateJobRequest()
    request.from_map(body)
    request.validate()
    require(request.to_map() == body, "SDK CreateJobRequest roundtrip drift")
    require(not os.path.lexists(OUTPUT_DIR), "canonical output appeared before submission")

    if args.audit_only:
        record = base_record("AUDIT_ONLY_PASS_CREATE_JOB_NOT_CALLED")
        record.update(
            {
                "create_job_called": False,
                "checked_at": utc_now(),
                "list_jobs_count": len(jobs),
                "assets": assets,
                "prelaunch_checked_at": prelaunch.get("checked_at"),
            }
        )
        write_exclusive(AUDIT_RECORD, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return 0

    initial = base_record("CREATE_JOB_CALL_IN_FLIGHT_DO_NOT_RETRY")
    initial.update(
        {
            "create_job_called": True,
            "call_started_at": utc_now(),
            "list_jobs_count": len(jobs),
            "instruction": (
                "Do not rerun this submitter. Resolve the outcome by exact DisplayName "
                "ListJobs and durable receipt inspection before any mutation."
            ),
        }
    )
    write_exclusive(RECEIPT, initial)
    try:
        response = dlc.create_job(request)
    except BaseException as exc:
        failure = dict(initial)
        failure.update(
            {
                "status": "CREATE_JOB_EXCEPTION_AMBIGUOUS_DO_NOT_RETRY",
                "call_finished_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        record_failure(failure)
        return 2

    response_body = getattr(response, "body", None)
    request_id = str(getattr(response_body, "request_id", "") or "")
    identifier = str(getattr(response_body, "job_id", "") or "")
    acknowledged = dict(initial)
    acknowledged.update(
        {
            "status": "CREATE_JOB_ACKNOWLEDGED_LIVE_READBACK_PENDING",
            "acknowledged_at": utc_now(),
            "request_id": request_id or None,
            "job_id": identifier or None,
        }
    )
    replace_receipt(RECEIPT, acknowledged)
    if not request_id or not identifier:
        acknowledged.update(
            {
                "status": "CREATE_JOB_ACK_IDENTITY_INCOMPLETE_AMBIGUOUS_DO_NOT_RETRY",
                "call_finished_at": utc_now(),
            }
        )
        record_failure(acknowledged)
        return 2
    try:
        live = dlc.get_job(identifier, models.GetJobRequest(need_detail=True)).body.to_map()
        require(job_id(live) == identifier, "GetJob JobId mismatch")
        require(display_name(live) == RUN_ID, "GetJob DisplayName mismatch")
        require(str(live.get("WorkspaceId")) == WORKSPACE_ID, "GetJob workspace mismatch")
        require(str(live.get("ResourceId")) == RESOURCE_ID, "GetJob resource mismatch")
        require(int(live.get("Priority")) == 7, "GetJob priority mismatch")
    except BaseException as exc:
        acknowledged.update(
            {
                "status": "CREATE_JOB_ACKNOWLEDGED_READBACK_FAILED_DO_NOT_RETRY",
                "call_finished_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        record_failure(acknowledged)
        return 2

    receipt = dict(acknowledged)
    receipt.update(
        {
            "status": "CREATE_JOB_ACKNOWLEDGED",
            "call_finished_at": utc_now(),
            "live_status": live.get("Status"),
            "live_reason_code": live.get("ReasonCode"),
        }
    )
    replace_receipt(RECEIPT, receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
