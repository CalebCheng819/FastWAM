#!/usr/bin/env python3
"""Exactly-once controller for the real-data scratch-training preflight."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import stat
import sys

import submit_table11safe_formal_once as common


RUN_ID = common.REAL_DATA_PREFLIGHT_RUN_ID
ATTEMPT_ID = common.REAL_DATA_PREFLIGHT_ATTEMPT_ID
COMMIT = "7a99d93dcc14cd8b8afeb962b589b67c79ea89e1"
LAUNCH_ROOT = common.LAUNCH_ROOT
DRY_REQUEST = LAUNCH_ROOT / "rendered-request-preflight-r8.json"
AUDIT_RECORD = LAUNCH_ROOT / "submission-preflight-r8-audit.json"
LATCH = LAUNCH_ROOT / "submission-preflight-r8-latch.json"
ACK = LAUNCH_ROOT / "submission-preflight-r8-job-ack.json"
RECONCILIATION = common.REAL_DATA_PREFLIGHT
OUTPUT_DIR = common.REAL_DATA_PREFLIGHT_OUTPUT_DIR
BUNDLE = common.BUNDLE
TERMINAL_STATUSES = {"Succeeded", "Failed", "Stopped"}


def expected_envs() -> dict:
    envs = copy.deepcopy(common.EXPECTED_ENVS)
    envs.update(
        {
            "FASTWAM_TABLE11_ATTEMPT_ID": ATTEMPT_ID,
            "FASTWAM_TABLE11_RUN_MODE": "preflight-one-step",
            "FASTWAM_TABLE11_CODE_COMMIT": COMMIT,
            "FASTWAM_TABLE11_OUTPUT_DIR": OUTPUT_DIR,
            "RUN_ID": RUN_ID,
        }
    )
    return envs


def expected_settings() -> dict:
    settings = copy.deepcopy(common.EXPECTED_SETTINGS)
    settings["Tags"].update(
        {
            "schedule": "optimizer-0-to-1-no-checkpoint",
            "topology": "1x8-world8",
        }
    )
    return settings


def validate_document(document: dict) -> dict:
    common.require(
        set(document) == common.EXPECTED_DOCUMENT_KEYS,
        "preflight dry-request keys drift",
    )
    common.require(document["dry_run"] is True, "dry_run must be true")
    common.require(
        document["submission_not_performed"] is True,
        "submission marker drift",
    )
    common.require(document["operation"] == "CreateJob", "operation drift")
    common.require(document["region"] == "cn-beijing", "region drift")
    common.require(
        document["endpoint"] == "pai-dlc.cn-beijing.aliyuncs.com",
        "endpoint drift",
    )
    common.require(document["sdk_python"] == common.SDK_PYTHON, "SDK Python drift")
    common.require(
        document["launcher_source"]
        == {
            "bundle": BUNDLE,
            "code_commit": COMMIT,
            "path": "scripts/launch_table11safe_3x8_dlc.sh",
        },
        "launcher source drift",
    )
    common.require(
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
    common.require(
        document["batch_contract"]
        == {
            "reference_global_batch": 24,
            "replica_global_batch": 8,
            "micro_batch_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "optimizer_updates": 1,
            "sample_budget_equivalent": False,
        },
        "preflight batch contract drift",
    )
    body = document["request"]
    common.require(
        isinstance(body, dict) and set(body) == common.EXPECTED_BODY_KEYS,
        "preflight request keys drift",
    )
    common.require(body["DisplayName"] == RUN_ID, "DisplayName drift")
    common.require(body["WorkspaceId"] == common.WORKSPACE_ID, "workspace drift")
    common.require(body["ResourceId"] == common.RESOURCE_ID, "resource drift")
    common.require(body["JobType"] == "PyTorchJob", "job type drift")
    common.require(body["SuccessPolicy"] == "AllWorkers", "success policy drift")
    common.require(body["Accessibility"] == "PRIVATE", "accessibility drift")
    common.require(body["Priority"] == 7, "preflight priority must be 7")
    common.require(body["JobMaxRunningTimeMinutes"] == 20160, "max runtime drift")
    common.require(body["CustomEnvs"] == [], "CustomEnvs drift")
    common.require(
        body["DataSources"]
        == [
            {
                "DataSourceId": "d-n7rly4fll0q2z6v91h",
                "MountAccess": "RW",
                "MountPath": "/oss-chengjuntao",
            }
        ],
        "datasource drift",
    )
    common.require(body["Settings"] == expected_settings(), "settings drift")
    common.require(body["Envs"] == expected_envs(), "environment drift")
    common.require(
        body["Description"]
        == (
            "Joint-safe RoboFactory table11 VG1H1GAU1 scratch-from-generic-base "
            "runtime preflight: optimizer step 0 to 1, one fresh-optimizer update, "
            "1 worker x 8 GPUs"
        ),
        "Description drift",
    )
    common.require(
        body["JobSpecs"]
        == [
            {
                "ElasticSpotSpecs": [],
                "Image": common.IMAGE,
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
        "1x8 JobSpec drift",
    )
    common.validate_embedded_launcher(
        document,
        body,
        expected_machines=1,
        expected_world=8,
        bundle=BUNDLE,
        commit=COMMIT,
        require_formal_schedule=False,
    )
    return body


def validate_assets() -> dict:
    return common.validate_assets(output_dir=OUTPUT_DIR, bundle=BUNDLE, commit=COMMIT)


def request_model(body: dict):
    request = common.models.CreateJobRequest()
    request.from_map(body)
    request.validate()
    common.require(request.to_map() == body, "SDK CreateJobRequest roundtrip drift")
    return request


def read_optional(path: pathlib.Path) -> dict | None:
    return common.read_json(path) if os.path.lexists(path) else None


def exact_live_identity(live: dict, body: dict) -> None:
    common.require(common.display_name(live) == RUN_ID, "live DisplayName drift")
    common.require(str(live.get("WorkspaceId")) == common.WORKSPACE_ID, "live workspace drift")
    common.require(str(live.get("ResourceId")) == common.RESOURCE_ID, "live resource drift")
    common.require(int(live.get("Priority")) == 7, "live priority drift")
    for key in ("JobType", "SuccessPolicy", "Accessibility"):
        if key in live and live[key] is not None:
            common.require(live[key] == body[key], f"live {key} drift")


def exact_candidates(dlc, body: dict) -> tuple[list[dict], int]:
    summaries = common.all_jobs_snapshot(dlc)
    matches = [item for item in summaries if common.display_name(item) == RUN_ID]
    details = []
    for item in matches:
        identifier = common.job_id(item)
        live = dlc.get_job(
            identifier, common.models.GetJobRequest(need_detail=True)
        ).body.to_map()
        common.require(common.job_id(live) == identifier, "GetJob JobId drift")
        exact_live_identity(live, body)
        details.append(live)
    common.require(len(details) <= 1, f"duplicate preflight jobs: {matches}")
    return details, len(summaries)


def immutable_file(path: pathlib.Path) -> os.stat_result:
    current = os.stat(path, follow_symlinks=False)
    common.require(stat.S_ISREG(current.st_mode), f"not regular: {path}")
    common.require(current.st_nlink == 1, f"not single-link: {path}")
    return current


def immutable_dir(path: pathlib.Path) -> os.stat_result:
    current = os.stat(path, follow_symlinks=False)
    common.require(stat.S_ISDIR(current.st_mode), f"not directory: {path}")
    return current


def validate_output() -> dict:
    root = pathlib.Path(OUTPUT_DIR)
    immutable_dir(root)
    ready_name = f".config.yaml.ready.stat_cmp.{ATTEMPT_ID}"
    expected = {
        ".table11-run-reservation",
        ready_name,
        "checkpoints",
        "config.yaml",
        "eval",
        "preflight-train.log",
        "terminal.json",
        "COMPLETE",
    }
    common.require({item.name for item in root.iterdir()} == expected, "output allowlist drift")
    for name in expected - {"checkpoints", "eval"}:
        immutable_file(root / name)
    immutable_dir(root / "checkpoints")
    immutable_dir(root / "checkpoints" / "state")
    immutable_dir(root / "checkpoints" / "weights")
    immutable_dir(root / "eval")
    common.require(
        {item.name for item in (root / "checkpoints").iterdir()}
        == {"state", "weights"},
        "checkpoint allowlist drift",
    )
    for empty_dir in (
        root / "checkpoints" / "state",
        root / "checkpoints" / "weights",
        root / "eval",
    ):
        common.require(not any(empty_dir.iterdir()), f"directory not empty: {empty_dir}")
    config_stat = immutable_file(root / "config.yaml")
    common.require(config_stat.st_size > 0, "runtime config is empty")
    ready = common.read_json(root / ready_name)
    common.require(
        set(ready)
        == {
            "schema",
            "attempt_id",
            "world_size",
            "path",
            "bytes",
            "mtime_ns",
            "count",
        },
        "runtime config marker schema drift",
    )
    common.require(
        {
            key: ready.get(key)
            for key in (
                "schema",
                "attempt_id",
                "world_size",
                "path",
                "bytes",
                "mtime_ns",
                "count",
            )
        }
        == {
            "schema": "fastwam-runtime-file-barrier-stat-cmp-v2",
            "attempt_id": ATTEMPT_ID,
            "world_size": 8,
            "path": str((root / "config.yaml").resolve()),
            "bytes": config_stat.st_size,
            "mtime_ns": config_stat.st_mtime_ns,
            "count": 1,
        },
        "runtime config marker contract drift",
    )
    for key in ("world_size", "bytes", "count", "mtime_ns"):
        common.require(type(ready.get(key)) is int, f"runtime marker {key} type drift")
    reservation_text = common.read_regular_bytes(
        root / ".table11-run-reservation"
    ).decode("utf-8")
    reservation: dict[str, str] = {}
    for line in reservation_text.splitlines():
        common.require(bool(line) and "=" in line, "malformed reservation line")
        key, value = line.split("=", 1)
        common.require(bool(key), "empty reservation key")
        common.require(key not in reservation, f"duplicate reservation key: {key}")
        reservation[key] = value
    common.require(
        reservation
        == {
            "run_id": RUN_ID,
            "attempt_id": ATTEMPT_ID,
            "run_mode": "preflight-one-step",
            "workers": "1",
            "gpus_per_worker": "8",
            "global_world_size": "8",
            "source_weight": common.SOURCE_WEIGHT,
            "initialization": "official-generic-pretrained-model-weights",
            "optimizer": "fresh",
            "provenance_mode": "stat_cmp",
            "initial_global_step": "0",
            "target_global_step": "1",
            "optimizer_steps_this_run": "1",
            "per_device_batch_size": "1",
            "gradient_accumulation_steps": "1",
            "reference_global_batch_size": "24",
            "global_batch_size": "8",
            "sample_budget_equivalent": "false",
            "learning_rate": "0.0001",
            "lr_scheduler": "cosine",
            "scheduler_warmup_steps": "2250",
            "save_every": "0",
            "checkpoint_keep_last": "0",
            "checkpoint_retention": "disabled",
            "checkpoint_steps": "none",
            "dataset_root": common.DATASET_ROOT,
            "gaussian_cache_dir": common.GAUSSIAN_CACHE_DIR,
        },
        "preflight reservation drift",
    )
    terminal = common.read_json(root / "terminal.json")
    complete = common.read_json(root / "COMPLETE")
    expected_terminal = {
        "schema": "fastwam-table11safe-realdata-scratch-preflight-terminal-v1",
        "status": "PASS",
        "run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "run_mode": "preflight-one-step",
        "dataset_kind": "joint-safe-table11-real-data",
        "initialization": "official-generic-pretrained-model-weights",
        "source_weight": common.SOURCE_WEIGHT,
        "optimizer": "fresh",
        "scheduler": "fresh",
        "initial_global_step": 0,
        "final_global_step": 1,
        "optimizer_steps_this_run": 1,
        "world_size": 8,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "log_path": str(root / "preflight-train.log"),
    }
    for key, value in expected_terminal.items():
        common.require(terminal.get(key) == value, f"terminal {key} drift")
    common.require(isinstance(terminal.get("completed_at"), str), "terminal time absent")
    common.require(
        complete
        == {
            "schema": "fastwam-table11safe-realdata-scratch-preflight-complete-v1",
            "status": "PASS",
            "run_id": RUN_ID,
            "attempt_id": ATTEMPT_ID,
            "terminal": str(root / "terminal.json"),
        },
        "COMPLETE drift",
    )
    log = common.read_regular_bytes(root / "preflight-train.log").decode("utf-8")
    required = (
        "Loading weight checkpoint before",
        "optimizer/DeepSpeed initialization:",
        "FASTWAM_GENERIC_BASE_LOAD=PASS before_prepare=true",
        "optimizer/scheduler/step are intentionally not restored.",
        "FASTWAM_TRAINING_START initial_global_step=0 max_steps=1 optimizer_steps_this_run=1",
        "FASTWAM_OPTIMIZER_STEP global_step=1 max_steps=1",
    )
    for marker in required:
        common.require(marker in log, f"preflight log omitted marker: {marker}")
    common.require("step_005000.pt" not in log, "old N234 checkpoint loaded")
    common.require(
        "Loaded explicit cross-treatment weights-only warm start" not in log,
        "cross-treatment warm start occurred",
    )
    return {
        "terminal": terminal,
        "complete": complete,
        "output_entries": sorted(expected),
        "empty_directories": ["checkpoints/state", "checkpoints/weights", "eval"],
        "runtime_config_marker": ready,
        "reservation": reservation,
        "log_bytes": immutable_file(root / "preflight-train.log").st_size,
    }


def audit(dry_request: pathlib.Path) -> int:
    common.require(dry_request == DRY_REQUEST, "unexpected dry-request path")
    for path in (AUDIT_RECORD, LATCH, ACK, RECONCILIATION):
        common.require(not os.path.lexists(path), f"preflight control path exists: {path}")
    document = common.read_json(DRY_REQUEST)
    body = validate_document(document)
    assets = validate_assets()
    request_model(body)
    dlc = common.client()
    details, count = exact_candidates(dlc, body)
    common.require(not details, "preflight job already exists")
    record = {
        "schema": "fastwam-table11safe-preflight-submit-audit-v1",
        "status": "PASS_CREATE_JOB_NOT_CALLED",
        "run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "request_path": str(DRY_REQUEST),
        "source_bundle": BUNDLE,
        "code_commit": COMMIT,
        "output_dir": OUTPUT_DIR,
        "create_job_called": False,
        "list_jobs_count": count,
        "assets": assets,
        "checked_at": common.utc_now(),
    }
    common.write_exclusive(AUDIT_RECORD, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0


def submit(dry_request: pathlib.Path, confirm_run_id: str) -> int:
    common.require(dry_request == DRY_REQUEST, "unexpected dry-request path")
    common.require(confirm_run_id == RUN_ID, "confirmation run ID drift")
    audit_record = common.read_json(AUDIT_RECORD)
    common.require(audit_record.get("status") == "PASS_CREATE_JOB_NOT_CALLED", "audit absent")
    for path in (LATCH, ACK, RECONCILIATION):
        common.require(not os.path.lexists(path), f"preflight control path exists: {path}")
    document = common.read_json(DRY_REQUEST)
    body = validate_document(document)
    validate_assets()
    request = request_model(body)
    dlc = common.client()
    details, count = exact_candidates(dlc, body)
    common.require(not details, "preflight job already exists")
    latch = {
        "schema": "fastwam-table11safe-permanent-create-latch-v1",
        "status": "LATCHED_CREATEJOB_ONCE_NEVER_RETRY",
        "run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "request_path": str(DRY_REQUEST),
        "code_commit": COMMIT,
        "create_job_called": True,
        "cloud_create_call_limit": 1,
        "list_jobs_count": count,
        "latched_at": common.utc_now(),
    }
    common.write_exclusive(LATCH, latch)
    try:
        response = dlc.create_job(request)
    except BaseException as exc:
        print(
            json.dumps(
                {
                    **latch,
                    "status": "CREATE_JOB_EXCEPTION_AMBIGUOUS_DO_NOT_RETRY",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    response_body = getattr(response, "body", None)
    identifier = str(getattr(response_body, "job_id", "") or "")
    request_id = str(getattr(response_body, "request_id", "") or "")
    common.require(identifier, "CreateJob ACK omitted JobId; latch forbids retry")
    ack = {
        "schema": "fastwam-table11safe-preflight-job-ack-v1",
        "status": "CREATE_JOB_ACKNOWLEDGED",
        "run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "job_id": identifier,
        "request_id": request_id or None,
        "source": "CreateJob",
        "recorded_at": common.utc_now(),
    }
    common.write_exclusive(ACK, ack)
    print(json.dumps(ack, ensure_ascii=False), flush=True)
    return 0


def monitor(dry_request: pathlib.Path) -> int:
    common.require(dry_request == DRY_REQUEST, "unexpected dry-request path")
    common.require(os.path.lexists(LATCH), "permanent preflight latch is absent")
    body = validate_document(common.read_json(DRY_REQUEST))
    dlc = common.client()
    details, count = exact_candidates(dlc, body)
    ack = read_optional(ACK)
    if ack is not None:
        identifier = str(ack.get("job_id") or "")
        common.require(identifier, "persisted ACK omitted JobId")
        if details:
            common.require(common.job_id(details[0]) == identifier, "ACK/candidate mismatch")
        else:
            live = dlc.get_job(
                identifier, common.models.GetJobRequest(need_detail=True)
            ).body.to_map()
            common.require(common.job_id(live) == identifier, "ACK GetJob mismatch")
            exact_live_identity(live, body)
            details = [live]
    if not details:
        print(json.dumps({"status": "PENDING_IDENTITY", "list_jobs_count": count}))
        return 3
    live = details[0]
    identifier = common.job_id(live)
    if ack is None:
        ack = {
            "schema": "fastwam-table11safe-preflight-job-ack-v1",
            "status": "CREATE_JOB_RECONCILED",
            "run_id": RUN_ID,
            "attempt_id": ATTEMPT_ID,
            "job_id": identifier,
            "request_id": None,
            "source": "ListJobs_then_GetJob",
            "recorded_at": common.utc_now(),
        }
        common.write_exclusive(ACK, ack)
    status = str(live.get("Status") or "")
    observed = {
        "status": status,
        "reason_code": live.get("ReasonCode"),
        "reason_message": live.get("ReasonMessage"),
        "job_id": identifier,
        "observed_at": common.utc_now(),
    }
    if status not in TERMINAL_STATUSES:
        print(json.dumps({"status": "PENDING_PROVIDER", "scheduler": observed}))
        return 3
    conclusion = {"status": "FAIL", "reason": f"provider terminal {status}"}
    output = None
    if status == "Succeeded":
        if not os.path.lexists(OUTPUT_DIR):
            print(json.dumps({"status": "PENDING_OUTPUT", "scheduler": observed}))
            return 3
        if not os.path.lexists(pathlib.Path(OUTPUT_DIR) / "COMPLETE"):
            print(json.dumps({"status": "PENDING_COMPLETE", "scheduler": observed}))
            return 3
        output = validate_output()
        conclusion = {
            "status": "PASS",
            "initialization": "official-generic-pretrained-model-weights",
            "optimizer": "fresh",
            "scheduler": "fresh",
            "initial_global_step": 0,
            "final_global_step": 1,
            "optimizer_steps_this_run": 1,
        }
    record = {
        "schema": "fastwam-table11safe-preflight-terminal-reconciliation-v1",
        "run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "job_id": identifier,
        "code_commit": COMMIT,
        "source_bundle": BUNDLE,
        "output_dir": OUTPUT_DIR,
        "scheduler_terminal": observed,
        "conclusion": conclusion,
        "output_validation": output,
        "reconciled_at": common.utc_now(),
    }
    common.require(not os.path.lexists(RECONCILIATION), "reconciliation already exists")
    common.write_exclusive(RECONCILIATION, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if conclusion["status"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "monitor"):
        item = subparsers.add_parser(name)
        item.add_argument("dry_request", type=pathlib.Path)
    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("dry_request", type=pathlib.Path)
    submit_parser.add_argument("--confirm-run-id", required=True)
    args = parser.parse_args()
    common.require(sys.flags.optimize == 0, "optimized Python is forbidden")
    common.require(
        os.path.realpath(sys.executable) == os.path.realpath(common.SDK_PYTHON),
        "wrong SDK Python",
    )
    if args.command == "audit":
        return audit(args.dry_request)
    if args.command == "submit":
        return submit(args.dry_request, args.confirm_run_id)
    return monitor(args.dry_request)


if __name__ == "__main__":
    raise SystemExit(main())
