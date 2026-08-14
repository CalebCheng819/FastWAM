#!/usr/bin/env python3
"""Wait for a valid P13 metric cache and submit its frozen DLC training job once."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REGION = "cn-beijing"
ENDPOINT = "pai-dlc.cn-beijing.aliyuncs.com"
WORKSPACE_ID = "270969"
RESOURCE_ID = "quotaksvqq2oh2pg"
PROFILE_PATH = Path("/root/.aliyun/config.json")
EXPECTED_SCHEMA = "fastwam.metric-geometry-cache"
EXPECTED_VERSION = 1
EXPECTED_FRAME_SHAPE = [13, 60, 80]
EXPECTED_ALLOWLIST = (
    "metadata frames.f16\n"
    "metadata manifest.json\n"
    "metadata COMPLETE\n"
)
TERMINAL_FAILURE_STATES = {"Failed", "Stopped", "Cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
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


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"observed_at_utc": utc_now(), **event}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_client(profile_path: Path = PROFILE_PATH) -> Any:
    from alibabacloud_credentials.client import Client as CredentialClient
    from alibabacloud_credentials.models import Config as CredentialConfig
    from alibabacloud_pai_dlc20201203.client import Client
    from alibabacloud_tea_openapi.models import Config

    document = json.loads(profile_path.read_text(encoding="utf-8"))
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
        Config(credential=credential, region_id=REGION, endpoint=ENDPOINT)
    )


def list_jobs(client: Any) -> list[dict[str, Any]]:
    from alibabacloud_pai_dlc20201203 import models

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


def parse_candidate(value: str) -> tuple[str, Path]:
    job_id, separator, root = value.partition("=")
    if not separator or not job_id.startswith("dlc") or not root.startswith("/"):
        raise argparse.ArgumentTypeError("candidate must be JOB_ID=/absolute/cache/root")
    return job_id, Path(root)


def validate_metric_cache(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"metric cache root is not a regular directory: {root}")
    complete = root / "COMPLETE"
    manifest_path = root / "manifest.json"
    frames_path = root / "frames.f16"
    allowlist_path = root / "stat-cmp.allowlist"
    for path in (complete, manifest_path, frames_path, allowlist_path):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"metric cache file is missing or non-regular: {path}")
    if complete.read_text(encoding="utf-8") != "complete\n":
        raise RuntimeError("metric cache COMPLETE marker mismatch")
    if allowlist_path.read_text(encoding="utf-8") != EXPECTED_ALLOWLIST:
        raise RuntimeError("metric cache stat-cmp allowlist mismatch")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_name": EXPECTED_SCHEMA,
        "version": EXPECTED_VERSION,
        "provenance_mode": "stat_cmp",
        "dtype": "float16",
        "byte_order": "little",
        "frame_shape": EXPECTED_FRAME_SHAPE,
    }
    observed = {key: manifest.get(key) for key in required}
    if observed != required:
        raise RuntimeError(
            f"metric cache schema mismatch: expected={required!r} observed={observed!r}"
        )

    selection = manifest.get("selection") or {}
    expected_selection = {
        "task_name": "PlaceFood-rf",
        "required_agent_count": 2,
        "action_horizon": 32,
        "split_seed": 42,
        "val_set_proportion": 0.1,
        "train_window_stride": 16,
        "val_window_stride": 32,
        "limit_trajectories": None,
    }
    if {key: selection.get(key) for key in expected_selection} != expected_selection:
        raise RuntimeError("metric cache selection contract mismatch")

    geometry = manifest.get("metric_geometry") or {}
    expected_geometry = {
        "source": "maniskill_calibrated_depth",
        "coordinate_frame": "world",
        "output_size": [60, 80],
        "channels": "xyz_mean_covariance_row_major_valid",
        "render_backend": "gpu",
    }
    if {key: geometry.get(key) for key in expected_geometry} != expected_geometry:
        raise RuntimeError("metric cache geometry contract mismatch")

    data = manifest.get("data") or {}
    frames = data.get("frames")
    declared_bytes = data.get("bytes")
    if not isinstance(frames, int) or frames <= 0:
        raise RuntimeError("metric cache has no positive frame count")
    expected_bytes = frames * 13 * 60 * 80 * 2
    frame_stat = frames_path.stat()
    observed_bytes = frame_stat.st_size
    if data.get("path") != "frames.f16":
        raise RuntimeError("metric cache data path mismatch")
    if declared_bytes != expected_bytes or observed_bytes != expected_bytes:
        raise RuntimeError(
            "metric cache byte count mismatch: "
            f"declared={declared_bytes} observed={observed_bytes} expected={expected_bytes}"
        )
    if data.get("mtime_ns") != frame_stat.st_mtime_ns:
        raise RuntimeError("metric cache frame mtime contract mismatch")

    entries = manifest.get("entries")
    counts = manifest.get("counts") or {}
    if not isinstance(entries, list) or len(entries) != frames:
        raise RuntimeError("metric cache entry count mismatch")
    if counts.get("frames") != frames or counts.get("windows", 0) * 2 != frames:
        raise RuntimeError("metric cache aggregate count mismatch")
    if counts.get("train_windows", 0) + counts.get("val_windows", 0) != counts.get(
        "windows"
    ):
        raise RuntimeError("metric cache split count mismatch")
    offsets: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("metric cache entry is not an object")
        offset = entry.get("offset")
        if not isinstance(offset, int) or offset < 0 or offset >= frames:
            raise RuntimeError("metric cache entry offset is invalid")
        if offset in offsets:
            raise RuntimeError("metric cache entry offset is duplicated")
        offsets.add(offset)
        if (
            not isinstance(entry.get("source_path"), str)
            or not isinstance(entry.get("trajectory"), str)
            or not isinstance(entry.get("timestep"), int)
            or not isinstance(entry.get("agent_name"), str)
        ):
            raise RuntimeError("metric cache frame key is invalid")
    if offsets != set(range(frames)):
        raise RuntimeError("metric cache entry offsets are not contiguous")

    return {
        "cache_root": str(root),
        "created_at": manifest.get("created_at"),
        "frames": frames,
        "windows": counts.get("windows"),
        "train_windows": counts.get("train_windows", 0),
        "val_windows": counts.get("val_windows", 0),
        "bytes": observed_bytes,
        "frame_shape": EXPECTED_FRAME_SHAPE,
    }


def duplicate_jobs(
    jobs: Iterable[dict[str, Any]], request_body: dict[str, Any]
) -> list[dict[str, Any]]:
    envs = request_body["Envs"]
    display_name = request_body["DisplayName"]
    run_id = envs["RUN_ID"]
    output_root = envs["FASTWAM_POSE_FOCUS_OUTPUT_DIR"]
    return [
        job
        for job in jobs
        if job.get("DisplayName") == display_name
        or (job.get("Envs") or {}).get("RUN_ID") == run_id
        or (job.get("Envs") or {}).get("FASTWAM_POSE_FOCUS_OUTPUT_DIR")
        == output_root
    ]


def validate_training_request(
    body: dict[str, Any], selected_cache_root: Path
) -> Any:
    from alibabacloud_pai_dlc20201203 import models

    if body.get("WorkspaceId") != WORKSPACE_ID or body.get("ResourceId") != RESOURCE_ID:
        raise RuntimeError("training workspace or resource mismatch")
    if body.get("DisplayName") != (
        "fastwam-placefood-metric-gaussian-p13-s42-8g-r1-20260815"
    ):
        raise RuntimeError("training display name mismatch")
    if body.get("JobType") != "PyTorchJob" or body.get("Priority") != 7:
        raise RuntimeError("training job protocol mismatch")
    specs = body.get("JobSpecs") or []
    expected_resources = {
        "CPU": "126",
        "GPU": "8",
        "Memory": "960Gi",
        "SharedMemory": "960Gi",
    }
    if len(specs) != 1 or specs[0].get("PodCount") != 1:
        raise RuntimeError("training topology is not one worker")
    if specs[0].get("ResourceConfig") != expected_resources:
        raise RuntimeError("training resource shape is not eight GPUs")
    if specs[0].get("RestartPolicy") != "Never":
        raise RuntimeError("training restart policy mismatch")

    envs = body.get("Envs") or {}
    expected_envs = {
        "RUN_ID": "fastwam-placefood-metric-gaussian-p13-s42-8g-r1-20260815",
        "FASTWAM_POSE_FOCUS_ATTEMPT_ID": "attempt-001",
        "FASTWAM_POSE_FOCUS_CODE_COMMIT": (
            "e5f20bbf91477b82990e5c571d54305c639705c6"
        ),
        "FASTWAM_POSE_FOCUS_TASK_PROFILE": (
            "robofactory_placefood_metric_gaussian_p13_224_5e-6"
        ),
        "FASTWAM_POSE_FOCUS_SOURCE_WEIGHT": (
            "/oss-chengjuntao/artifacts/"
            "fastwam-placefood-spatial-gripcontact-p10-lowaux-s42-8g-r1-20260814/"
            "checkpoints/weights/step_001000.pt"
        ),
        "FASTWAM_POSE_FOCUS_SOURCE_WEIGHT_BYTES": "12047407747",
        "FASTWAM_POSE_FOCUS_SOURCE_BUNDLE": (
            "/oss-chengjuntao/artifacts/fastwam-p13-runtime-20260815/"
            "FastWAM-p13-e5f20bb.bundle"
        ),
        "FASTWAM_POSE_FOCUS_OUTPUT_DIR": (
            "/oss-chengjuntao/artifacts/"
            "fastwam-placefood-metric-gaussian-p13-s42-8g-r1-20260815"
        ),
        "FASTWAM_POSE_FOCUS_EXPECTED_WORKERS": "1",
        "FASTWAM_POSE_FOCUS_EXPECTED_GPUS_PER_WORKER": "8",
        "FASTWAM_POSE_FOCUS_PROVENANCE_MODE": "stat_cmp",
        "FASTWAM_PREFLIGHT_REQUIRE_ERDMA": "0",
    }
    observed_envs = {key: envs.get(key) for key in expected_envs}
    if observed_envs != expected_envs:
        raise RuntimeError(
            f"training environment mismatch: expected={expected_envs!r} "
            f"observed={observed_envs!r}"
        )
    if envs.get("FASTWAM_POSE_FOCUS_METRIC_SOURCE_ROOT") != str(
        selected_cache_root
    ):
        raise RuntimeError("training metric cache root mismatch")
    if envs.get("FASTWAM_POSE_FOCUS_METRIC_ALLOWLIST") != str(
        selected_cache_root / "stat-cmp.allowlist"
    ):
        raise RuntimeError("training metric cache allowlist mismatch")

    source_weight = Path(envs["FASTWAM_POSE_FOCUS_SOURCE_WEIGHT"])
    source_bundle = Path(envs["FASTWAM_POSE_FOCUS_SOURCE_BUNDLE"])
    if not source_weight.is_file() or source_weight.stat().st_size != int(
        envs["FASTWAM_POSE_FOCUS_SOURCE_WEIGHT_BYTES"]
    ):
        raise RuntimeError("training source weight is absent or has the wrong size")
    if not source_bundle.is_file():
        raise RuntimeError("training source bundle is absent")

    request = models.CreateJobRequest().from_map(body)
    request.validate()
    if request.to_map() != body:
        raise RuntimeError("training request serialization roundtrip mismatch")
    return request


def selected_cache(
    client: Any, candidates: list[tuple[str, Path]]
) -> tuple[str, Path, dict[str, Any], dict[str, Any]] | None:
    from alibabacloud_pai_dlc20201203 import models

    succeeded: list[tuple[str, Path, dict[str, Any], dict[str, Any]]] = []
    statuses: list[dict[str, Any]] = []
    for job_id, root in candidates:
        job = client.get_job(
            job_id, models.GetJobRequest(need_detail=True)
        ).body.to_map()
        status = str(job.get("Status") or "Unknown")
        statuses.append({"job_id": job_id, "status": status, "cache_root": str(root)})
        if (job.get("Envs") or {}).get("FASTWAM_P13_CACHE_OUTPUT_ROOT") != str(root):
            raise RuntimeError(f"cache job {job_id} output identity mismatch")
        if status != "Succeeded":
            continue
        try:
            summary = validate_metric_cache(root)
        except Exception as error:
            statuses[-1]["validation_error"] = f"{type(error).__name__}: {error}"
            continue
        succeeded.append((job_id, root, job, summary))

    if succeeded:
        def ordering(
            item: tuple[str, Path, dict[str, Any], dict[str, Any]]
        ) -> tuple[str, str]:
            job_id, _root, job, summary = item
            finished = str(job.get("GmtFinishedTime") or summary.get("created_at") or "9999")
            return finished, job_id

        return min(succeeded, key=ordering)
    terminal_without_valid_cache = all(
        item["status"] in TERMINAL_FAILURE_STATES
        or (item["status"] == "Succeeded" and "validation_error" in item)
        for item in statuses
    )
    if statuses and terminal_without_valid_cache:
        raise RuntimeError(
            f"all cache candidates terminated without a valid cache: {statuses}"
        )
    return None


def adopt_existing_job(
    client: Any,
    matches: list[dict[str, Any]],
    request_body: dict[str, Any],
    selected_job_id: str | None,
    selected_root: Path | None,
) -> dict[str, Any] | None:
    from alibabacloud_pai_dlc20201203 import models

    if not matches:
        return None
    unique = {str(job.get("JobId")) for job in matches}
    if len(unique) != 1:
        raise RuntimeError(f"multiple duplicate training jobs found: {sorted(unique)}")
    job_id = next(iter(unique))
    job = client.get_job(
        job_id, models.GetJobRequest(need_detail=True)
    ).body.to_map()
    envs = job.get("Envs") or {}
    expected_envs = request_body["Envs"]
    identity_keys = (
        "RUN_ID",
        "FASTWAM_POSE_FOCUS_OUTPUT_DIR",
        "FASTWAM_POSE_FOCUS_CODE_COMMIT",
        "FASTWAM_POSE_FOCUS_SOURCE_WEIGHT",
        "FASTWAM_POSE_FOCUS_TASK_PROFILE",
    )
    if job.get("DisplayName") != request_body["DisplayName"] or any(
        envs.get(key) != expected_envs.get(key) for key in identity_keys
    ):
        raise RuntimeError(f"conflicting training job uses frozen identity: {job_id}")
    return {
        "mode": "adopt-existing",
        "job_id": job_id,
        "status": job.get("Status"),
        "display_name": job.get("DisplayName"),
        "selected_cache_job_id": selected_job_id,
        "selected_cache_root": str(selected_root) if selected_root else None,
        "submitted_at_utc": job.get("GmtSubmittedTime"),
        "observed_at_utc": utc_now(),
        "cloud_mutations_called": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-request", required=True, type=Path)
    parser.add_argument("--candidate", required=True, action="append", type=parse_candidate)
    parser.add_argument("--record-root", required=True, type=Path)
    parser.add_argument("--lock-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--timeout-seconds", type=float, default=604800.0)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("poll and timeout seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.record_root.mkdir(parents=True, exist_ok=True)
    args.lock_root.mkdir(parents=True, exist_ok=True)
    receipt_path = args.record_root / "submission-receipt.json"
    state_path = args.record_root / "state.json"
    event_path = args.record_root / "events.jsonl"
    lock_path = args.lock_root / "supervisor.lock"
    lock_stream = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another supervisor owns {lock_path}") from error

    if receipt_path.is_file():
        print(receipt_path.read_text(encoding="utf-8"), end="")
        return

    dry_run = json.loads(args.dry_run_request.read_text(encoding="utf-8"))
    if dry_run.get("dry_run") is not True or dry_run.get(
        "submission_not_performed"
    ) is not True:
        raise RuntimeError("training request is not a frozen non-submitted dry run")
    original_request = dry_run["request"]
    client = load_client()
    started = time.monotonic()
    append_event(
        event_path,
        {
            "event": "supervisor_started",
            "submit_enabled": args.submit,
            "candidate_job_ids": [job_id for job_id, _root in args.candidate],
        },
    )

    while True:
        selected = selected_cache(client, args.candidate)
        request_body = json.loads(json.dumps(original_request))
        selected_job_id: str | None = None
        selected_root: Path | None = None
        cache_summary: dict[str, Any] | None = None
        if selected is not None:
            selected_job_id, selected_root, _cache_job, cache_summary = selected
            request_body["Envs"]["FASTWAM_POSE_FOCUS_METRIC_SOURCE_ROOT"] = str(
                selected_root
            )
            request_body["Envs"]["FASTWAM_POSE_FOCUS_METRIC_ALLOWLIST"] = str(
                selected_root / "stat-cmp.allowlist"
            )

        jobs = list_jobs(client)
        matches = duplicate_jobs(jobs, request_body)
        adopted = adopt_existing_job(
            client, matches, request_body, selected_job_id, selected_root
        )
        if adopted is not None:
            write_exclusive_json(receipt_path, adopted)
            atomic_json(state_path, {"state": "ADOPTED", **adopted})
            append_event(event_path, {"event": "training_job_adopted", **adopted})
            print(json.dumps(adopted, indent=2, sort_keys=True))
            return

        if selected is None:
            state = {
                "state": "WAITING_FOR_VALID_CACHE",
                "observed_at_utc": utc_now(),
                "candidate_job_ids": [job_id for job_id, _root in args.candidate],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            atomic_json(state_path, state)
            print(json.dumps(state, sort_keys=True), flush=True)
            if args.once:
                return
            if time.monotonic() - started >= args.timeout_seconds:
                raise TimeoutError("timed out waiting for a valid metric cache")
            time.sleep(args.poll_seconds)
            continue

        request = validate_training_request(request_body, selected_root)
        training_output = Path(
            request_body["Envs"]["FASTWAM_POSE_FOCUS_OUTPUT_DIR"]
        )
        if training_output.exists():
            raise RuntimeError(
                f"training output already exists without a matching DLC job: {training_output}"
            )
        ready = {
            "state": "READY_TO_SUBMIT",
            "observed_at_utc": utc_now(),
            "selected_cache_job_id": selected_job_id,
            "selected_cache_root": str(selected_root),
            "cache_summary": cache_summary,
            "display_name": request_body["DisplayName"],
            "training_output": str(training_output),
            "submit_enabled": args.submit,
        }
        atomic_json(state_path, ready)
        append_event(event_path, {"event": "cache_validated", **ready})
        if not args.submit:
            print(json.dumps(ready, indent=2, sort_keys=True))
            return

        time.sleep(3.0)
        second_matches = duplicate_jobs(list_jobs(client), request_body)
        adopted = adopt_existing_job(
            client, second_matches, request_body, selected_job_id, selected_root
        )
        if adopted is not None:
            write_exclusive_json(receipt_path, adopted)
            atomic_json(state_path, {"state": "ADOPTED", **adopted})
            print(json.dumps(adopted, indent=2, sort_keys=True))
            return
        if training_output.exists():
            raise RuntimeError(
                f"training output appeared during duplicate guard: {training_output}"
            )

        response = client.create_job(request)
        job_id = str(response.body.job_id)
        from alibabacloud_pai_dlc20201203 import models

        job = client.get_job(
            job_id, models.GetJobRequest(need_detail=True)
        ).body.to_map()
        if job.get("DisplayName") != request_body["DisplayName"]:
            raise RuntimeError("submitted training identity mismatch")
        receipt = {
            "mode": "submit",
            "job_id": job_id,
            "request_id": response.body.request_id,
            "status": job.get("Status"),
            "display_name": job.get("DisplayName"),
            "selected_cache_job_id": selected_job_id,
            "selected_cache_root": str(selected_root),
            "cache_summary": cache_summary,
            "observed_at_utc": utc_now(),
            "cloud_mutations_called": ["CreateJob"],
        }
        write_exclusive_json(receipt_path, receipt)
        atomic_json(state_path, {"state": "SUBMITTED", **receipt})
        append_event(event_path, {"event": "training_job_submitted", **receipt})
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return


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
