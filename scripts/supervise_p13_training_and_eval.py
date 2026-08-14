#!/usr/bin/env python3
"""Wait for P13 training, then run the frozen offline and closed-loop evals."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGION = "cn-beijing"
ENDPOINT = "pai-dlc.cn-beijing.aliyuncs.com"
PROFILE_PATH = Path("/root/.aliyun/config.json")
EXPECTED_ALLOWLIST = (
    "metadata frames.f16\n"
    "metadata manifest.json\n"
    "metadata COMPLETE\n"
)
EXPECTED_SELECTION = {
    "task_name": "PlaceFood-rf",
    "required_agent_count": 2,
    "action_horizon": 32,
    "split_seed": 42,
    "val_set_proportion": 0.1,
    "train_window_stride": 16,
    "val_window_stride": 32,
    "limit_trajectories": None,
}
EXPECTED_GEOMETRY = {
    "source": "maniskill_calibrated_depth",
    "coordinate_frame": "world",
    "output_size": [60, 80],
    "channels": "xyz_mean_covariance_row_major_valid",
    "render_backend": "gpu",
}
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


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"observed_at_utc": utc_now(), **event}, sort_keys=True))
        stream.write("\n")
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
    return Client(Config(credential=credential, region_id=REGION, endpoint=ENDPOINT))


def get_job(client: Any, job_id: str) -> dict[str, Any]:
    from alibabacloud_pai_dlc20201203 import models

    return client.get_job(
        job_id, models.GetJobRequest(need_detail=True)
    ).body.to_map()


def read_training_receipt(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"training receipt is not a regular file: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    job_id = receipt.get("job_id")
    cache_root = receipt.get("selected_cache_root")
    if not isinstance(job_id, str) or not job_id.startswith("dlc"):
        raise RuntimeError("training receipt has an invalid DLC job id")
    if not isinstance(cache_root, str) or not cache_root.startswith("/"):
        raise RuntimeError("training receipt has no absolute selected cache root")
    if receipt.get("mode") not in {"submit", "adopt-existing"}:
        raise RuntimeError("training receipt mode is not recognized")
    return receipt


def validate_metric_cache(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"metric cache root is not a regular directory: {root}")
    files = {
        name: root / name
        for name in ("COMPLETE", "manifest.json", "frames.f16", "stat-cmp.allowlist")
    }
    for path in files.values():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"metric cache file is missing or non-regular: {path}")
    if files["COMPLETE"].read_text(encoding="utf-8") != "complete\n":
        raise RuntimeError("metric cache COMPLETE marker mismatch")
    if files["stat-cmp.allowlist"].read_text(encoding="utf-8") != EXPECTED_ALLOWLIST:
        raise RuntimeError("metric cache stat-cmp allowlist mismatch")

    manifest = json.loads(files["manifest.json"].read_text(encoding="utf-8"))
    expected_schema = {
        "schema_name": "fastwam.metric-geometry-cache",
        "version": 1,
        "provenance_mode": "stat_cmp",
        "dtype": "float16",
        "byte_order": "little",
        "frame_shape": [13, 60, 80],
    }
    if {key: manifest.get(key) for key in expected_schema} != expected_schema:
        raise RuntimeError("metric cache schema mismatch")
    selection = manifest.get("selection") or {}
    if {key: selection.get(key) for key in EXPECTED_SELECTION} != EXPECTED_SELECTION:
        raise RuntimeError("metric cache selection contract mismatch")
    geometry = manifest.get("metric_geometry") or {}
    if {key: geometry.get(key) for key in EXPECTED_GEOMETRY} != EXPECTED_GEOMETRY:
        raise RuntimeError("metric cache geometry contract mismatch")

    data = manifest.get("data") or {}
    frames = data.get("frames")
    if not isinstance(frames, int) or frames <= 0:
        raise RuntimeError("metric cache has no positive frame count")
    expected_bytes = frames * 13 * 60 * 80 * 2
    frame_stat = files["frames.f16"].stat()
    if (
        data.get("path") != "frames.f16"
        or data.get("bytes") != expected_bytes
        or frame_stat.st_size != expected_bytes
    ):
        raise RuntimeError("metric cache byte count mismatch")
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
    offsets = {entry.get("offset") for entry in entries if isinstance(entry, dict)}
    if offsets != set(range(frames)):
        raise RuntimeError("metric cache entry offsets are not contiguous")
    for entry in entries:
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(key), expected_type)
            for key, expected_type in (
                ("source_path", str),
                ("trajectory", str),
                ("timestep", int),
                ("agent_name", str),
            )
        ):
            raise RuntimeError("metric cache frame key is invalid")
    return {
        "cache_root": str(root),
        "frames": frames,
        "windows": counts.get("windows"),
        "bytes": expected_bytes,
        "frame_shape": [13, 60, 80],
    }


def validate_training_gate(
    job: dict[str, Any],
    receipt: dict[str, Any],
    expected_training_output: Path,
    checkpoint: Path,
) -> tuple[str, dict[str, Any]]:
    if str(job.get("JobId")) != receipt["job_id"]:
        raise RuntimeError("DLC job id differs from the training receipt")
    envs = job.get("Envs") or {}
    if envs.get("FASTWAM_POSE_FOCUS_OUTPUT_DIR") != str(expected_training_output):
        raise RuntimeError("DLC training output differs from the frozen output root")
    if envs.get("FASTWAM_POSE_FOCUS_METRIC_SOURCE_ROOT") != receipt[
        "selected_cache_root"
    ]:
        raise RuntimeError("DLC metric cache differs from the submitted cache")
    status = str(job.get("Status") or "Unknown")
    summary = {
        "job_id": receipt["job_id"],
        "display_name": job.get("DisplayName"),
        "status": status,
        "reason_code": job.get("ReasonCode"),
        "reason_message": job.get("ReasonMessage"),
    }
    if status in TERMINAL_FAILURE_STATES:
        raise RuntimeError(f"P13 training terminated unsuccessfully: {summary}")
    if status != "Succeeded":
        return "WAITING_FOR_TRAINING", summary

    expected_checkpoint = expected_training_output / "checkpoints/weights/step_001000.pt"
    if checkpoint != expected_checkpoint:
        raise RuntimeError(
            f"checkpoint path is not the frozen P13 step: {checkpoint} != {expected_checkpoint}"
        )
    marker = Path(f"{checkpoint}.COMPLETE")
    if not checkpoint.is_file() or checkpoint.is_symlink() or checkpoint.stat().st_size <= 0:
        return "WAITING_FOR_CHECKPOINT", summary
    if not marker.is_file() or marker.is_symlink() or marker.stat().st_size <= 0:
        return "WAITING_FOR_CHECKPOINT", summary
    summary["checkpoint"] = {
        "path": str(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "complete_marker": str(marker),
    }
    return "TRAINING_READY", summary


def parse_gpu_inventory(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line}")
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "memory_used_mib": int(fields[2]),
                "utilization_percent": int(fields[3]),
            }
        )
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return rows


def parse_compute_apps(text: str) -> dict[str, list[dict[str, int]]]:
    applications: dict[str, list[dict[str, int]]] = {}
    for line in text.splitlines():
        if not line.strip() or "No running processes" in line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise RuntimeError(f"unexpected nvidia-smi process row: {line}")
        applications.setdefault(fields[0], []).append(
            {"pid": int(fields[1]), "used_gpu_memory_mib": int(fields[2])}
        )
    return applications


def query_gpu_state() -> list[dict[str, Any]]:
    inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = parse_gpu_inventory(inventory.stdout)
    apps = parse_compute_apps(processes.stdout)
    for row in rows:
        row["compute_apps"] = apps.get(row["uuid"], [])
    return rows


def select_free_gpus(
    rows: list[dict[str, Any]], count: int, memory_threshold_mib: int
) -> list[int]:
    free = [
        int(row["index"])
        for row in sorted(rows, key=lambda item: int(item["index"]))
        if not row.get("compute_apps")
        and int(row["memory_used_mib"]) <= memory_threshold_mib
    ]
    return free[:count] if len(free) >= count else []


def validate_teacher_output(root: Path, expected_cache_root: Path) -> dict[str, Any]:
    status = (root / "terminal.status").read_text(encoding="utf-8").strip()
    terminal = json.loads((root / "TERMINAL_STATUS.json").read_text(encoding="utf-8"))
    comparison = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    if status != "SUCCEEDED" or terminal.get("status") != "SUCCEEDED":
        raise RuntimeError("teacher-forcing output is not terminally successful")
    if terminal.get("return_code") != 0 or comparison.get("status") != "COMPLETED":
        raise RuntimeError("teacher-forcing output has an unsuccessful contract")
    if comparison.get("metric_cache_root") != str(expected_cache_root):
        raise RuntimeError("teacher-forcing output used a different metric cache")
    if (
        comparison.get("states") != 263
        or comparison.get("valid_pairs_h1") != 263
        or comparison.get("valid_pairs_h5") != 1305
    ):
        raise RuntimeError("teacher-forcing output has incomplete paired coverage")
    return comparison


def validate_closedloop_output(root: Path) -> dict[str, Any]:
    report = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
    if (
        report.get("status") != "COMPLETE"
        or report.get("expected_runs") != 8
        or report.get("operational_runs") != 8
        or not isinstance(report.get("success_count"), int)
    ):
        raise RuntimeError("closed-loop output is not a complete eight-run panel")
    return report


def existing_output(root: Path, validator: Any, *args: Any) -> dict[str, Any] | None:
    if not root.exists():
        return None
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"evaluation output is not a regular directory: {root}")
    try:
        return validator(root, *args)
    except Exception as error:
        raise RuntimeError(
            f"existing output must be preserved and cannot be adopted: {root}: {error}"
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-receipt", required=True, type=Path)
    parser.add_argument("--expected-training-output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--record-root", required=True, type=Path)
    parser.add_argument("--lock-root", required=True, type=Path)
    parser.add_argument("--teacher-script", required=True, type=Path)
    parser.add_argument("--closedloop-script", required=True, type=Path)
    parser.add_argument("--teacher-output", required=True, type=Path)
    parser.add_argument("--closedloop-output", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--timeout-seconds", type=float, default=1209600.0)
    parser.add_argument("--gpu-memory-threshold-mib", type=int, default=1024)
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("poll and timeout seconds must be positive")
    if args.gpu_memory_threshold_mib < 0:
        parser.error("GPU memory threshold must be non-negative")
    return args


def write_state(path: Path, state: str, **details: Any) -> dict[str, Any]:
    payload = {"state": state, "observed_at_utc": utc_now(), **details}
    atomic_json(path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def run_script(script: Path, environment: dict[str, str]) -> None:
    if script.is_symlink() or not script.is_file():
        raise RuntimeError(f"evaluation script is missing or non-regular: {script}")
    result = subprocess.run(["bash", str(script)], env=environment, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"evaluation script failed with return code {result.returncode}: {script}")


def supervise(args: argparse.Namespace) -> None:
    args.record_root.mkdir(parents=True, exist_ok=True)
    args.lock_root.mkdir(parents=True, exist_ok=True)
    state_path = args.record_root / "state.json"
    event_path = args.record_root / "events.jsonl"
    lock_path = args.lock_root / "eval-supervisor.lock"
    lock_stream = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another evaluation supervisor owns {lock_path}") from error

    client = load_client()
    started = time.monotonic()
    append_event(event_path, {"event": "eval_supervisor_started", "run_eval": args.run_eval})
    while True:
        receipt = read_training_receipt(args.training_receipt)
        if receipt is None:
            write_state(state_path, "WAITING_FOR_TRAINING_SUBMISSION")
        else:
            job = get_job(client, receipt["job_id"])
            gate, training = validate_training_gate(
                job, receipt, args.expected_training_output, args.checkpoint
            )
            if gate != "TRAINING_READY":
                write_state(state_path, gate, training=training)
            else:
                cache_root = Path(receipt["selected_cache_root"])
                cache = validate_metric_cache(cache_root)
                teacher = existing_output(
                    args.teacher_output, validate_teacher_output, cache_root
                )
                if teacher is None:
                    rows = query_gpu_state()
                    selected = select_free_gpus(
                        rows, 1, args.gpu_memory_threshold_mib
                    )
                    if not selected:
                        write_state(
                            state_path,
                            "WAITING_FOR_TEACHER_GPU",
                            training=training,
                            cache=cache,
                            gpus=rows,
                        )
                    elif not args.run_eval:
                        write_state(
                            state_path,
                            "READY_FOR_TEACHER",
                            training=training,
                            cache=cache,
                            selected_gpus=selected,
                        )
                        return
                    else:
                        append_event(
                            event_path,
                            {"event": "teacher_started", "gpus": selected},
                        )
                        environment = os.environ.copy()
                        environment.update(
                            {
                                "P13_METRIC_CACHE_ROOT": str(cache_root),
                                "P13_TRAIN_ROOT": str(args.expected_training_output),
                                "P13_TF_OUTPUT_ROOT": str(args.teacher_output),
                                "P13_TF_GPU": str(selected[0]),
                            }
                        )
                        run_script(args.teacher_script, environment)
                        teacher = validate_teacher_output(args.teacher_output, cache_root)
                        append_event(event_path, {"event": "teacher_succeeded"})

                if teacher is not None:
                    closedloop = existing_output(
                        args.closedloop_output, validate_closedloop_output
                    )
                    if closedloop is not None:
                        write_state(
                            state_path,
                            "COMPLETED",
                            training=training,
                            cache=cache,
                            teacher=teacher,
                            closedloop=closedloop,
                        )
                        append_event(event_path, {"event": "eval_completed"})
                        return
                    rows = query_gpu_state()
                    selected = select_free_gpus(
                        rows, 4, args.gpu_memory_threshold_mib
                    )
                    if not selected:
                        write_state(
                            state_path,
                            "WAITING_FOR_CLOSEDLOOP_GPUS",
                            training=training,
                            cache=cache,
                            teacher=teacher,
                            gpus=rows,
                        )
                    elif not args.run_eval:
                        write_state(
                            state_path,
                            "READY_FOR_CLOSEDLOOP",
                            training=training,
                            cache=cache,
                            teacher=teacher,
                            selected_gpus=selected,
                        )
                        return
                    else:
                        append_event(
                            event_path,
                            {"event": "closedloop_started", "gpus": selected},
                        )
                        environment = os.environ.copy()
                        environment.update(
                            {
                                "P13_METRIC_CACHE_ROOT": str(cache_root),
                                "P13_TRAIN_ROOT": str(args.expected_training_output),
                                "P13_TF_OUTPUT_ROOT": str(args.teacher_output),
                                "P13_CLOSEDLOOP_OUTPUT_ROOT": str(args.closedloop_output),
                                "P13_EVAL_GPUS": " ".join(map(str, selected)),
                            }
                        )
                        run_script(args.closedloop_script, environment)
                        closedloop = validate_closedloop_output(args.closedloop_output)
                        write_state(
                            state_path,
                            "COMPLETED",
                            training=training,
                            cache=cache,
                            teacher=teacher,
                            closedloop=closedloop,
                        )
                        append_event(event_path, {"event": "eval_completed"})
                        return

        if args.once:
            return
        if time.monotonic() - started >= args.timeout_seconds:
            raise TimeoutError("timed out waiting for P13 training or evaluation resources")
        time.sleep(args.poll_seconds)


def main() -> None:
    args = parse_args()
    try:
        supervise(args)
    except Exception as error:
        args.record_root.mkdir(parents=True, exist_ok=True)
        failure = {
            "state": "FAILED",
            "observed_at_utc": utc_now(),
            "error_type": type(error).__name__,
            "message": str(error),
            "credentials_printed": False,
        }
        atomic_json(args.record_root / "state.json", failure)
        append_event(args.record_root / "events.jsonl", {"event": "supervisor_failed", **failure})
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
