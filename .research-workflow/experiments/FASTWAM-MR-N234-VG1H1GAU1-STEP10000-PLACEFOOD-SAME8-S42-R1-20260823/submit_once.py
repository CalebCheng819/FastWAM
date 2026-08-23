#!/usr/bin/env python3
"""Fail-closed, exactly-once submission for the frozen step-10k eval."""

from __future__ import annotations

import argparse
import base64
import copy
import datetime as dt
import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_credentials.models import Config as CredentialConfig
from alibabacloud_pai_dlc20201203 import models
from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_tea_openapi.models import Config


EXPERIMENT_ID = "FASTWAM-MR-N234-VG1H1GAU1-STEP10000-PLACEFOOD-SAME8-S42-R1-20260823"
RUN_ID = "fastwam-gau1-step10k-placefood-same8-r3-20260823"
ATTEMPT_ID = "attempt-003"
DISPLAY_NAME = "fw-gau1-s10k-placefood-same8-r3"
WORKSPACE_ID = "270969"
RESOURCE_ID = "quotaksvqq2oh2pg"
OUTPUT_ROOT = "/oss-chengjuntao/artifacts/fastwam-gau1-step10k-placefood-same8-eval-20260823-r3"
CHECKPOINT = "/oss-chengjuntao/artifacts/fastwam-n234-vg1h1gau1-cont50k-s42-24g-r1-20260822/checkpoints/weights/step_010000.pt"
SOURCE_BUNDLE = "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/fastwam-gau1-step10k-placefood-same8-eval-20260823-r3.bundle"
IMAGE = "dsw-registry-vpc.cn-beijing.cr.aliyuncs.com/pai/pytorch:2.7.1-gpu-py310-cu128-ubuntu22.04-3995b779-1764350887"
SDK_PYTHON = pathlib.Path("/usr/bin/python3")
SDK_PYTHONPATH = pathlib.Path("/tmp/gau1-sdk-target-c8Pn5S")
DATA_SOURCES = [
    {"DataSourceId": "d-a5mu77ymwjio71dkmw", "MountPath": "/cpfs/user/chengjuntao", "MountAccess": "RO"},
    {"DataSourceId": "d-n7rly4fll0q2z6v91h", "MountPath": "/oss-chengjuntao", "MountAccess": "RW"},
]
JOB_SPEC = {
    "Type": "Worker",
    "PodCount": 1,
    "Image": IMAGE,
    "RestartPolicy": "Never",
    "ResourceConfig": {"GPU": "8", "CPU": "126", "Memory": "960Gi", "SharedMemory": "960Gi"},
}
EXPECTED_BODY_KEYS = {
    "Accessibility", "DataSources", "Description", "DisplayName", "Envs",
    "JobMaxRunningTimeMinutes", "JobSpecs", "JobType", "Priority", "ResourceId",
    "SuccessPolicy", "UserCommand", "WorkspaceId",
}
EXPECTED_DOCUMENT_KEYS = {
    "schema_version", "dry_run", "submission_not_performed", "operation",
    "experiment_id", "run_id", "source_commit", "source_bundle", "output_root",
    "launcher_payload_base64", "request",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _path_is_within(path: pathlib.Path, root: pathlib.Path) -> bool:
    """Return whether path is under root without requiring Python 3.9."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _expected_sdk_roundtrip(body: dict) -> dict:
    """Return the exact map emitted by DLC SDK 1.9.2 for this request.

    The SDK materializes four omitted optional list fields as empty lists.  Keep
    that normalization explicit and narrow so every other request field still
    has to round-trip byte-for-byte at the JSON object level.
    """
    expected = copy.deepcopy(body)
    expected["CustomEnvs"] = []
    _require(len(expected["JobSpecs"]) == 1, "unexpected JobSpecs cardinality")
    worker = expected["JobSpecs"][0]
    worker["ElasticSpotSpecs"] = []
    worker["LocalMountSpecs"] = []
    worker["StartupDependencies"] = []
    return expected


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_regular_bytes(path: pathlib.Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        current = os.stat(path, follow_symlinks=False)
        frozen = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        _require(frozen == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), f"file changed while being read: {path}")
        _require(stat.S_ISREG(current.st_mode), f"pathname is not regular: {path}")
        _require(frozen == (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns), f"pathname changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _fsync_parent(path: pathlib.Path) -> None:
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_initial_receipt(path: pathlib.Path, payload: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_parent(path)


def _replace_receipt(path: pathlib.Path, payload: dict) -> None:
    current = os.stat(path, follow_symlinks=False)
    _require(stat.S_ISREG(current.st_mode), f"receipt is not regular: {path}")
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = pathlib.Path(handle.name)
            os.chmod(temporary, 0o600)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_parent(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _credentials_uri() -> str:
    document = json.loads(pathlib.Path("/root/.aliyun/config.json").read_text())
    profiles = document["profiles"]
    current = document["current"]
    profile = profiles[current] if isinstance(profiles, dict) else next(item for item in profiles if item.get("name") == current)
    uri = profile.get("credentials_uri") or profile.get("CredentialsURI") or profile.get("credentialsUri")
    _require(bool(uri), "current Alibaba Cloud profile has no credentials URI")
    return str(uri)


def _client() -> Client:
    credential = CredentialClient(CredentialConfig(type="credentials_uri", credentials_uri=_credentials_uri()))
    client = Client(Config(credential=credential, region_id="cn-beijing", endpoint="pai-dlc.cn-beijing.aliyuncs.com"))
    _require(getattr(client, "_retry_options", None) is None, "DLC client unexpectedly enables implicit retries")
    return client


def _job_id(job: dict) -> str:
    value = job.get("JobId") or job.get("job_id")
    _require(bool(value), f"ListJobs entry omitted JobId: {job}")
    return str(value)


def _display_name(job: dict) -> str:
    return str(job.get("DisplayName") or job.get("display_name") or "")


def _all_jobs_snapshot(client: Client) -> list[dict]:
    jobs: list[dict] = []
    seen: set[str] = set()
    expected_total: int | None = None
    page = 1
    page_size = 100
    while True:
        _require(page <= 1000, "ListJobs pagination exceeded 1000 pages")
        response = client.list_jobs(models.ListJobsRequest(workspace_id=WORKSPACE_ID, resource_id=RESOURCE_ID, page_number=page, page_size=page_size)).body.to_map()
        _require("TotalCount" in response, "ListJobs omitted TotalCount")
        total = int(response["TotalCount"])
        _require(total >= 0, "ListJobs returned negative TotalCount")
        if expected_total is None:
            expected_total = total
        else:
            _require(total == expected_total, f"ListJobs TotalCount drifted: {expected_total} -> {total}")
        batch = response.get("Jobs") or []
        _require(isinstance(batch, list) and len(batch) <= page_size, "ListJobs page is malformed")
        if len(jobs) < expected_total:
            _require(bool(batch), "ListJobs ended before frozen TotalCount")
        for job in batch:
            _require(isinstance(job, dict), "ListJobs entry is not a mapping")
            identifier = _job_id(job)
            _require(identifier not in seen, f"ListJobs repeated JobId: {identifier}")
            seen.add(identifier)
            jobs.append(job)
        _require(len(jobs) <= expected_total, "ListJobs exceeded frozen TotalCount")
        if len(jobs) == expected_total:
            return jobs
        page += 1


def _verify_no_duplicate_job(client: Client) -> list[dict]:
    first = _all_jobs_snapshot(client)
    second = _all_jobs_snapshot(client)
    identity = lambda jobs: sorted((_job_id(job), _display_name(job)) for job in jobs)
    _require(identity(first) == identity(second), "ListJobs identity snapshot drifted")
    matches = [job for job in second if _display_name(job) == DISPLAY_NAME]
    _require(not matches, f"refuse duplicate CreateJob: {matches}")
    return second


def _git_stdout(*arguments: str, text: bool = False) -> bytes | str:
    return subprocess.run(["git", *arguments], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text).stdout


def _validate(document: dict) -> dict:
    _require(isinstance(document, dict) and set(document) == EXPECTED_DOCUMENT_KEYS, "dry-run document keys drifted")
    _require(document["schema_version"] == "fastwam-dlc-create-job-dry-run-v1", "schema version drift")
    _require(document["dry_run"] is True and document["submission_not_performed"] is True, "dry-run flags drifted")
    _require(document["operation"] == "CreateJob", "operation drift")
    _require(document["experiment_id"] == EXPERIMENT_ID and document["run_id"] == RUN_ID, "experiment identity drift")
    _require(document["source_bundle"] == SOURCE_BUNDLE and document["output_root"] == OUTPUT_ROOT, "bundle/output drift")
    commit = document["source_commit"]
    _require(isinstance(commit, str) and len(commit) == 40 and all(ch in "0123456789abcdef" for ch in commit), "source commit is not a full lowercase Git object id")

    bundle_stat = os.stat(SOURCE_BUNDLE, follow_symlinks=False)
    _require(stat.S_ISREG(bundle_stat.st_mode), "source bundle is not regular")
    heads = str(_git_stdout("bundle", "list-heads", SOURCE_BUNDLE, text=True)).splitlines()
    _require(heads == [f"{commit} HEAD"], f"source bundle heads drifted: {heads}")
    with tempfile.TemporaryDirectory(prefix="fastwam-gau1-step10k-bundle-audit.") as temporary:
        checkout = pathlib.Path(temporary) / "source"
        _git_stdout("clone", "--quiet", "--no-checkout", SOURCE_BUNDLE, str(checkout))
        _git_stdout("-C", str(checkout), "bundle", "verify", SOURCE_BUNDLE)
        cloned_head = str(_git_stdout("-C", str(checkout), "rev-parse", "HEAD", text=True)).strip()
        _require(cloned_head == commit, f"cloned source commit drifted: {cloned_head}")
        _git_stdout("-C", str(checkout), "show", f"{commit}:.research-workflow/experiments/{EXPERIMENT_ID}/runtime.sh")

    body = document["request"]
    _require(isinstance(body, dict) and set(body) == EXPECTED_BODY_KEYS, "CreateJob body keys drifted")
    _require(body["DisplayName"] == DISPLAY_NAME, "DisplayName drift")
    _require(body["WorkspaceId"] == WORKSPACE_ID and body["ResourceId"] == RESOURCE_ID, "workspace/resource drift")
    _require(body["JobType"] == "PyTorchJob" and body["SuccessPolicy"] == "AllWorkers", "job type/success policy drift")
    _require(body["Accessibility"] == "PRIVATE" and body["Priority"] == 7, "accessibility/priority drift")
    _require(body["JobMaxRunningTimeMinutes"] == 2160, "max runtime drift")
    _require(body["DataSources"] == DATA_SOURCES and body["JobSpecs"] == [JOB_SPEC], "mount or 1x8 Worker contract drift")
    _require(body["Description"] == f"Formal PlaceFood same8 closed-loop evaluation of GAU1 step_010000.pt; experiment={EXPERIMENT_ID}", "description drift")
    expected_command = "python3 - <<'PY'\nimport base64,os\np=base64.b64decode(os.environ['FASTWAM_LAUNCHER_B64'])\nos.execv('/bin/bash',['bash','-c',p.decode('utf-8')])\nPY"
    _require(body["UserCommand"] == expected_command, "UserCommand drift")

    envs = body["Envs"]
    expected_envs = {
        "FASTWAM_EXPERIMENT_ID": EXPERIMENT_ID,
        "FASTWAM_RUN_ID": RUN_ID,
        "FASTWAM_ATTEMPT_ID": ATTEMPT_ID,
        "FASTWAM_OUTPUT_ROOT": OUTPUT_ROOT,
        "FASTWAM_SOURCE_BUNDLE": SOURCE_BUNDLE,
        "FASTWAM_SOURCE_COMMIT": commit,
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
        "FASTWAM_LAUNCHER_B64": document["launcher_payload_base64"],
    }
    _require(envs == expected_envs, "frozen environment contract drift")
    encoded = document["launcher_payload_base64"]
    _require(isinstance(encoded, str) and encoded == envs["FASTWAM_LAUNCHER_B64"], "launcher identity drift")
    launcher = base64.b64decode(encoded, validate=True)
    _require(base64.b64encode(launcher).decode("ascii") == encoded, "launcher base64 is non-canonical")
    for fragment in (
        EXPERIMENT_ID.encode(), RUN_ID.encode(), ATTEMPT_ID.encode(), SOURCE_BUNDLE.encode(),
        b'actual="$(git -C "${root}/source" rev-parse HEAD)"',
        f".research-workflow/experiments/{EXPERIMENT_ID}/runtime.sh".encode(),
    ):
        _require(fragment in launcher, f"launcher omitted contract fragment: {fragment!r}")
    _require(not os.path.lexists(OUTPUT_ROOT), "canonical output already exists")
    return body


def _receipt_base(status: str, dry_request: pathlib.Path) -> dict:
    return {
        "schema_version": 1, "status": status, "experiment_id": EXPERIMENT_ID,
        "display_name": DISPLAY_NAME, "run_id": RUN_ID, "attempt_id": ATTEMPT_ID,
        "workspace_id": WORKSPACE_ID,
        "resource_id": RESOURCE_ID, "dry_request": str(dry_request),
        "output_root": OUTPUT_ROOT, "source_bundle": SOURCE_BUNDLE,
    }


def _record_failure(path: pathlib.Path, payload: dict) -> None:
    try:
        _replace_receipt(path, payload)
    except BaseException as exc:
        payload["receipt_update_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dry_request", type=pathlib.Path)
    parser.add_argument("receipt", type=pathlib.Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    _require(sys.flags.optimize == 0, "optimized Python execution is forbidden")
    _require(os.path.realpath(sys.executable) == os.path.realpath(SDK_PYTHON), f"wrong SDK Python: {sys.executable}")
    pythonpath_entries = [pathlib.Path(item).resolve() for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    _require(SDK_PYTHONPATH.resolve() in pythonpath_entries, f"required SDK PYTHONPATH missing: {SDK_PYTHONPATH}")
    models_path = pathlib.Path(models.__file__).resolve()
    _require(_path_is_within(models_path, SDK_PYTHONPATH.resolve()), f"DLC SDK loaded outside frozen target: {models_path}")
    _require(not os.path.lexists(args.receipt), f"receipt already exists: {args.receipt}")
    document = json.loads(_read_regular_bytes(args.dry_request).decode("utf-8"))
    body = _validate(document)
    client = _client()
    jobs = _verify_no_duplicate_job(client)
    request = models.CreateJobRequest()
    request.from_map(body)
    request.validate()
    _require(request.to_map() == _expected_sdk_roundtrip(body), "SDK CreateJobRequest roundtrip drift")
    _require(not os.path.lexists(OUTPUT_ROOT), "canonical output appeared before submit")
    if args.audit_only:
        result = _receipt_base("AUDIT_ONLY_PASS_CREATE_JOB_NOT_CALLED", args.dry_request)
        result.update({"create_job_called": False, "list_jobs_count": len(jobs), "checked_at": _utc_now()})
        print(json.dumps(result, ensure_ascii=False))
        return 0

    initial = _receipt_base("CREATE_JOB_CALL_IN_FLIGHT_DO_NOT_RETRY", args.dry_request)
    initial.update({"create_job_called": True, "call_started_at": _utc_now(), "list_jobs_count": len(jobs), "instruction": "Do not rerun. Resolve outcome by exact DisplayName before any mutation."})
    _write_initial_receipt(args.receipt, initial)
    try:
        response = client.create_job(request)
    except BaseException as exc:
        failure = dict(initial)
        failure.update({"status": "CREATE_JOB_EXCEPTION_AMBIGUOUS_DO_NOT_RETRY", "call_finished_at": _utc_now(), "error_type": type(exc).__name__, "error": str(exc)})
        _record_failure(args.receipt, failure)
        return 2
    response_body = getattr(response, "body", None)
    request_id = str(getattr(response_body, "request_id", "") or "")
    job_id = str(getattr(response_body, "job_id", "") or "")
    ack = dict(initial)
    ack.update({"acknowledged_at": _utc_now(), "request_id": request_id or None, "job_id": job_id or None})
    if not request_id or not job_id:
        ack["status"] = "CREATE_JOB_ACK_INCOMPLETE_AMBIGUOUS_DO_NOT_RETRY"
        _record_failure(args.receipt, ack)
        return 2
    ack["status"] = "CREATE_JOB_ACKNOWLEDGED_LIVE_READBACK_PENDING"
    _replace_receipt(args.receipt, ack)
    try:
        live = client.get_job(job_id, models.GetJobRequest(need_detail=True)).body.to_map()
        _require(_job_id(live) == job_id and _display_name(live) == DISPLAY_NAME, "GetJob identity mismatch")
        _require(str(live.get("WorkspaceId")) == WORKSPACE_ID and str(live.get("ResourceId")) == RESOURCE_ID, "GetJob workspace/resource mismatch")
    except BaseException as exc:
        ack.update({"status": "CREATE_JOB_ACKNOWLEDGED_READBACK_FAILURE_DO_NOT_RETRY", "call_finished_at": _utc_now(), "error_type": type(exc).__name__, "error": str(exc)})
        _record_failure(args.receipt, ack)
        return 2
    ack.update({"status": "CREATE_JOB_ACKNOWLEDGED", "call_finished_at": _utc_now(), "live_status": live.get("Status"), "live_reason_code": live.get("ReasonCode")})
    _replace_receipt(args.receipt, ack)
    print(json.dumps(ack, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
