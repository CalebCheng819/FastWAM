#!/usr/bin/env python3
"""Render, but never submit, the formal RoboFactory table11 2x8 DLC job."""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import re
import shlex
import subprocess
import tempfile


IMAGE = (
    "dsw-registry-vpc.cn-beijing.cr.aliyuncs.com/pai/"
    "pytorch:2.7.1-gpu-py310-cu128-ubuntu22.04-3995b779-1764350887"
)
RESOURCE_ID = "quotaksvqq2oh2pg"
WORKSPACE_ID = "270969"
LAUNCHER_PATH = "scripts/launch_table11_2x8_dlc.sh"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")

# The 24-GPU reference task used this archived dependency environment. These
# values are existing infrastructure gates owned by its bootstrap helper; this
# renderer deliberately introduces no new data/checkpoint hash-chain gates.
LEGACY_DEPENDENCY_MANIFEST_ID = "b740e7224ad38628c12347ff0d36cb85dea45095f335ec032a52f07fcade7ee5"
LEGACY_DEPENDENCY_RUNTIME_LOCK_ID = "d495f1a1192ced91edd7df2794a94fe0ffb67526a279570d5cf3649d59c0d360"
LEGACY_DEPENDENCY_CACHE_HELPER_ID = "89dc9d7302f2edc1320b5f08f0516d5d2e9c6a176705642cf2f57756a1ae22ae"

DATASET_ROOT = (
    "/oss-chengjuntao/robofactory/table/"
    "robofactory-table-11task-200each-h299-2g-r1-20260825/tasks"
)
ASSET_ROOT = (
    "/oss-chengjuntao/fastwam-assets/robofactory/"
    "table11-200each-h299-r1-s42"
)
STATS_PATH = f"{ASSET_ROOT}/stats/train-stats.json"
TEXT_CACHE_DIR = f"{ASSET_ROOT}/text-embeds"
GAUSSIAN_CACHE_DIR = (
    f"{ASSET_ROOT}/gaussian/"
    "compact-s42-13x28x40-fp16-meanalpha-direct-v1"
)
MODEL_CACHE_ROOT = (
    "/oss-chengjuntao/cpfs-user-chengjuntao/"
    "checkpoints/FastWAM/model-cache"
)
VAE_PATH = (
    f"{MODEL_CACHE_ROOT}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/"
    "Wan2.2_VAE.safetensors"
)
SOURCE_WEIGHT = (
    "/oss-chengjuntao/artifacts/"
    "fastwam-n234-vg1hub1gau1-s42-5000-r2a2-beg0t5rle97qepyw8u-"
    "a57915104bff-20260802t1820z/checkpoints/weights/step_005000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--bootstrap-script", required=True)
    parser.add_argument("--offline-env-source-root", required=True)
    parser.add_argument("--offline-env-manifest", required=True)
    parser.add_argument("--offline-code-commit", required=True)
    parser.add_argument("--offline-source-bundle-relative-path", required=True)
    parser.add_argument("--base-python", required=True)
    parser.add_argument("--source-bundle", required=True, type=pathlib.Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--max-running-minutes", type=int, default=20160)
    parser.add_argument("--allow-local-bundle-for-tests", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def launcher_from_bundle(bundle: pathlib.Path, commit: str) -> bytes:
    if not bundle.is_file() or bundle.is_symlink():
        raise SystemExit("source-bundle must be an existing regular non-symlink file")
    with tempfile.TemporaryDirectory(prefix="fastwam-table11-render-") as directory:
        repository = pathlib.Path(directory) / "repository"
        for command in (
            ["git", "init", "--quiet", str(repository)],
            ["git", "-C", str(repository), "fetch", "--quiet", "--no-tags", "--", str(bundle), commit],
        ):
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            if result.returncode:
                detail = result.stderr.strip() or result.stdout.strip()
                raise SystemExit(f"failed to read source bundle: {detail}")
        resolved = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "FETCH_HEAD^{commit}"],
            text=True,
            capture_output=True,
            check=False,
        )
        if resolved.returncode or resolved.stdout.strip() != commit:
            raise SystemExit("code-commit did not resolve exactly from source-bundle")
        extracted = subprocess.run(
            ["git", "-C", str(repository), "show", f"{commit}:{LAUNCHER_PATH}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if extracted.returncode:
            detail = extracted.stderr.decode("utf-8", errors="replace").strip()
            raise SystemExit(f"bundled commit is missing {LAUNCHER_PATH}: {detail}")
        if not extracted.stdout.startswith(b"#!/usr/bin/env bash\n"):
            raise SystemExit("bundled launcher has an unexpected interpreter header")
        return extracted.stdout


def main() -> int:
    args = parse_args()
    if not SAFE_ID.fullmatch(args.run_id) or not SAFE_ID.fullmatch(args.attempt_id):
        raise SystemExit("run-id and attempt-id must be safe unique identifiers")
    if args.max_running_minutes <= 0:
        raise SystemExit("max-running-minutes must be positive")
    for name in ("bootstrap_script", "offline_env_source_root", "offline_env_manifest"):
        if not getattr(args, name).startswith("/oss-chengjuntao/"):
            raise SystemExit(f"{name.replace('_', '-')} must be below /oss-chengjuntao/")
    bundle_text = str(args.source_bundle)
    if not args.allow_local_bundle_for_tests and not bundle_text.startswith("/oss-chengjuntao/"):
        raise SystemExit("source-bundle must be below /oss-chengjuntao/")
    if not bundle_text.endswith(".bundle"):
        raise SystemExit("source-bundle must name a Git .bundle file")
    if not args.base_python.startswith("/"):
        raise SystemExit("base-python must be an absolute interpreter path")
    relative_bundle = args.offline_source_bundle_relative_path
    if (
        relative_bundle.startswith("/")
        or ".." in pathlib.PurePosixPath(relative_bundle).parts
        or relative_bundle in ("", ".")
    ):
        raise SystemExit("offline-source-bundle-relative-path must be a safe relative path")
    if not HEX40.fullmatch(args.offline_code_commit):
        raise SystemExit("offline-code-commit must be an exact lowercase Git revision")
    if not HEX40.fullmatch(args.code_commit):
        raise SystemExit("code-commit must be an exact lowercase Git revision")

    launcher_bytes = launcher_from_bundle(args.source_bundle, args.code_commit)
    payload = base64.b64encode(launcher_bytes).decode("ascii")
    outer_shell = (
        "set -euo pipefail; "
        "embedded=/tmp/fastwam-table11-outer-launcher.$$; "
        "umask 077; "
        f"printf '%s' {shlex.quote(payload)} | base64 --decode > \"$embedded\"; "
        "exec /bin/bash \"$embedded\""
    )
    user_command = "/bin/bash -c " + shlex.quote(outer_shell)
    output_dir = f"/oss-chengjuntao/artifacts/{args.run_id}"
    envs = {
        "RUN_ID": args.run_id,
        "FASTWAM_TABLE11_ATTEMPT_ID": args.attempt_id,
        "FASTWAM_TABLE11_OUTPUT_DIR": output_dir,
        "FASTWAM_TABLE11_OUTPUT_RESERVATION_TIMEOUT": "300",
        "FASTWAM_TABLE11_BOOTSTRAP_SCRIPT": args.bootstrap_script,
        "FASTWAM_OFFLINE_ENV_SOURCE_ROOT": args.offline_env_source_root,
        "FASTWAM_OFFLINE_ENV_MANIFEST": args.offline_env_manifest,
        "FASTWAM_OFFLINE_ENV_MANIFEST_SHA256": LEGACY_DEPENDENCY_MANIFEST_ID,
        "FASTWAM_OFFLINE_ENV_RUNTIME_LOCK_SHA256": LEGACY_DEPENDENCY_RUNTIME_LOCK_ID,
        "FASTWAM_OFFLINE_ENV_CACHE_HELPER_SHA256": LEGACY_DEPENDENCY_CACHE_HELPER_ID,
        "FASTWAM_OFFLINE_CODE_COMMIT": args.offline_code_commit,
        "FASTWAM_OFFLINE_ENV_SOURCE_BUNDLE_RELATIVE_PATH": relative_bundle,
        "FASTWAM_OFFLINE_ENV_BASE_PYTHON": args.base_python,
        "FASTWAM_OFFLINE_ENV_CACHE_ROOT": "/tmp/fastwam-offline-env-cache",
        "FASTWAM_OFFLINE_ENV_VENV_ROOT": "/tmp/fastwam-offline-env-venvs",
        "FASTWAM_SOURCE_CHECKOUT_ROOT": "/tmp/fastwam-source-checkouts",
        "FASTWAM_OFFLINE_ENV_WAIT_TIMEOUT": "7200",
        "FASTWAM_OFFLINE_ENV_STALE_LOCK_SECONDS": "7200",
        "FASTWAM_TABLE11_SOURCE_BUNDLE": bundle_text,
        "FASTWAM_TABLE11_CODE_COMMIT": args.code_commit,
        "FASTWAM_TABLE11_LOCAL_SOURCE_ROOT": "/tmp/fastwam-table11-source-checkouts",
        "FASTWAM_TABLE11_PROVENANCE_MODE": "stat_cmp",
        "FASTWAM_TABLE11_DATASET_ROOT": DATASET_ROOT,
        "FASTWAM_TABLE11_STATS_PATH": STATS_PATH,
        "FASTWAM_TABLE11_TEXT_CACHE_DIR": TEXT_CACHE_DIR,
        "FASTWAM_TABLE11_GAUSSIAN_CACHE_DIR": GAUSSIAN_CACHE_DIR,
        "FASTWAM_TABLE11_MODEL_CACHE_ROOT": MODEL_CACHE_ROOT,
        "FASTWAM_TABLE11_VAE_PATH": VAE_PATH,
        "FASTWAM_TABLE11_SOURCE_WEIGHT": SOURCE_WEIGHT,
        "FASTWAM_TABLE11_SOURCE_WEIGHT_BYTES": "12047213728",
        "FASTWAM_TABLE11_EXPECTED_H5_FILES": "11",
        "FASTWAM_ERDMA_BUNDLE_ROOT": "/oss-chengjuntao/artifacts/erdma-userspace-56.2-1.0.3",
        "FASTWAM_ERDMA_EXPECTED_VERSION": "56.2-1.0.3",
        "FASTWAM_PREFLIGHT_REQUIRE_ERDMA": "1",
        "FASTWAM_PREFLIGHT_TIMEOUT": "7200",
        "FASTWAM_PREFLIGHT_OUTER_TIMEOUT": "7260",
        "NCCL_IB_HCA": "erdma",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET",
        "NPROC_PER_NODE": "8",
    }
    request = {
        "Accessibility": "PRIVATE",
        "CustomEnvs": [],
        "DataSources": [
            {
                "DataSourceId": "d-n7rly4fll0q2z6v91h",
                "MountAccess": "RW",
                "MountPath": "/oss-chengjuntao",
            }
        ],
        "Description": (
            "RoboFactory table11 VG1H1GAU1 weights-only continuation: cumulative "
            "5000 to 50000, 45000 fresh-optimizer updates, 2 workers x 8 GPUs; "
            "world-16 global batch differs from the world-24 reference"
        ),
        "DisplayName": args.run_id,
        "Envs": envs,
        "JobMaxRunningTimeMinutes": args.max_running_minutes,
        "JobSpecs": [
            {
                "ElasticSpotSpecs": [],
                "Image": IMAGE,
                "LocalMountSpecs": [],
                "PodCount": 2,
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
            "AllocateAllRDMADevices": True,
            "EnableCPUAffinity": False,
            "EnableErrorMonitoringInAIMaster": False,
            "EnableOssAppend": False,
            "EnableRDMA": True,
            "EnableSanityCheck": False,
            "Tags": {
                "experiment": "TABLE11-VG1H1GAU1-CONT50K",
                "initialization": "GAU1-step5000-weights-only",
                "optimizer": "fresh",
                "provenance": "stat-cmp-no-new-hash",
                "topology": "2x8-world16",
                "schedule": "cumulative-5000-to-50000-save-5000",
            },
        },
        "SuccessPolicy": "AllWorkers",
        "UserCommand": user_command,
        "WorkspaceId": WORKSPACE_ID,
    }
    document = {
        "dry_run": True,
        "submission_not_performed": True,
        "operation": "CreateJob",
        "sdk_python": (
            "/mnt/workspace/tools/pai-control-py311/"
            "20260817-credentials1.0.10-dlc1.9.2/bin/python"
        ),
        "region": "cn-beijing",
        "endpoint": "pai-dlc.cn-beijing.aliyuncs.com",
        "launcher_source": {
            "bundle": bundle_text,
            "code_commit": args.code_commit,
            "path": LAUNCHER_PATH,
        },
        "launcher_payload_base64": payload,
        "provenance_contract": {
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
        "batch_contract": {
            "reference_global_batch": 24,
            "replica_global_batch": 16,
            "micro_batch_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "optimizer_updates": 45000,
            "sample_budget_equivalent": False,
        },
        "request": request,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
