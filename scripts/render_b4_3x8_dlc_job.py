#!/usr/bin/env python3
"""Render, but never submit, a PAI DLC CreateJob manifest for FastWAM."""

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
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")

# These values belong to the already-published dependency environment used by
# the completed formal run. They remain a legacy infrastructure integrity gate;
# B4 data/checkpoint provenance uses path, stat, and byte comparison instead.
LEGACY_DEPENDENCY_MANIFEST_ID = "b740e7224ad38628c12347ff0d36cb85dea45095f335ec032a52f07fcade7ee5"
LEGACY_DEPENDENCY_RUNTIME_LOCK_ID = "d495f1a1192ced91edd7df2794a94fe0ffb67526a279570d5cf3649d59c0d360"
LEGACY_DEPENDENCY_CACHE_HELPER_ID = "89dc9d7302f2edc1320b5f08f0516d5d2e9c6a176705642cf2f57756a1ae22ae"
LAUNCHER_PATH = "scripts/launch_b4_3x8_dlc.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap-script", required=True)
    parser.add_argument("--offline-env-source-root", required=True)
    parser.add_argument("--offline-env-manifest", required=True)
    parser.add_argument("--offline-code-commit", required=True)
    parser.add_argument("--offline-source-bundle-relative-path", required=True)
    parser.add_argument("--base-python", required=True)
    parser.add_argument("--b4-source-bundle", type=pathlib.Path, required=True)
    parser.add_argument("--b4-code-commit", required=True)
    parser.add_argument(
        "--treatment",
        choices=("b4", "n234_vg1h1gau1_cont50k", "n234_vg1h1gau0_cont50k"),
        default="b4",
    )
    parser.add_argument("--max-running-minutes", type=int)
    parser.add_argument(
        "--allow-local-bundle-for-tests",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def launcher_from_bundle(bundle: pathlib.Path, commit: str) -> bytes:
    """Read the launcher at an exact bundled commit without using this worktree."""
    if not bundle.is_file() or bundle.is_symlink():
        raise SystemExit("b4-source-bundle must be an existing regular non-symlink file")
    with tempfile.TemporaryDirectory(prefix="fastwam-b4-render-") as directory:
        repository = pathlib.Path(directory) / "repository"
        commands = (
            ["git", "init", "--quiet", str(repository)],
            [
                "git",
                "-C",
                str(repository),
                "fetch",
                "--quiet",
                "--no-tags",
                "--",
                str(bundle),
                commit,
            ],
        )
        for command in commands:
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            if result.returncode:
                detail = result.stderr.strip() or result.stdout.strip()
                raise SystemExit(f"failed to read B4 source bundle: {detail}")
        resolved = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "FETCH_HEAD^{commit}"],
            text=True,
            capture_output=True,
            check=False,
        )
        if resolved.returncode or resolved.stdout.strip() != commit:
            raise SystemExit("b4-code-commit did not resolve exactly from b4-source-bundle")
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
            raise SystemExit("bundled B4 launcher has an unexpected interpreter header")
        return extracted.stdout


def main() -> int:
    args = parse_args()
    contracts = {
        "b4": {
            "nodes": 3,
            "world_size": 24,
            "source_weight": (
                "/oss-chengjuntao/artifacts/"
                "fastwam-n234-vg1hub1gau1-s42-5000-r2a2-"
                "beg0t5rle97qepyw8u-a57915104bff-20260802t1820z/"
                "checkpoints/weights/step_005000.pt"
            ),
            "source_weight_bytes": "12047213728",
            "use_gaussian": True,
        },
        "n234_vg1h1gau1_cont50k": {
            "nodes": 3,
            "world_size": 24,
            "source_weight": (
                "/oss-chengjuntao/artifacts/"
                "fastwam-n234-vg1hub1gau1-s42-5000-r2a2-"
                "beg0t5rle97qepyw8u-a57915104bff-20260802t1820z/"
                "checkpoints/weights/step_005000.pt"
            ),
            "source_weight_bytes": "12047213728",
            "use_gaussian": True,
        },
        "n234_vg1h1gau0_cont50k": {
            "nodes": 3,
            "world_size": 24,
            "source_weight": (
                "/oss-chengjuntao/artifacts/fastwam-checkpoint-archives-v1/"
                "FASTWAM-MR-N234-VG1H1-S42-20260801/dlc1hqocuisxxdkb/"
                "step_005000/checkpoints/weights/step_005000.pt"
            ),
            "source_weight_bytes": "12045923769",
            "use_gaussian": False,
        },
    }
    contract = contracts[args.treatment]
    max_running_minutes = args.max_running_minutes
    if max_running_minutes is None:
        max_running_minutes = (
            20160 if args.treatment != "b4" else 10080
        )
    if not SAFE_ID.fullmatch(args.run_id) or not SAFE_ID.fullmatch(args.attempt_id):
        raise SystemExit("run-id and attempt-id must be safe unique identifiers")
    if max_running_minutes <= 0:
        raise SystemExit("max-running-minutes must be positive")
    for name in ("bootstrap_script", "offline_env_source_root", "offline_env_manifest"):
        if not getattr(args, name).startswith("/oss-chengjuntao/"):
            raise SystemExit(f"{name.replace('_', '-')} must be below /oss-chengjuntao/")
    bundle_text = str(args.b4_source_bundle)
    if not args.allow_local_bundle_for_tests and not bundle_text.startswith("/oss-chengjuntao/"):
        raise SystemExit("b4-source-bundle must be below /oss-chengjuntao/")
    if not args.base_python.startswith("/"):
        raise SystemExit("base-python must be an absolute interpreter path")
    relative_bundle = args.offline_source_bundle_relative_path
    if (
        relative_bundle.startswith("/")
        or ".." in pathlib.PurePosixPath(relative_bundle).parts
        or relative_bundle in ("", ".")
    ):
        raise SystemExit("offline-source-bundle-relative-path must be a safe relative path")
    if not bundle_text.endswith(".bundle"):
        raise SystemExit("b4-source-bundle must name a Git .bundle file")
    if not HEX40.fullmatch(args.offline_code_commit):
        raise SystemExit("offline-code-commit must be an exact lowercase Git revision")
    if not HEX40.fullmatch(args.b4_code_commit):
        raise SystemExit("b4-code-commit must be an exact lowercase Git revision")

    launcher_bytes = launcher_from_bundle(args.b4_source_bundle, args.b4_code_commit)
    payload = base64.b64encode(launcher_bytes).decode("ascii")
    outer_shell = (
        "set -euo pipefail; "
        "embedded=/tmp/fastwam-b4-outer-launcher.$$; "
        "umask 077; "
        f"printf '%s' {shlex.quote(payload)} | base64 --decode > \"$embedded\"; "
        "exec /bin/bash \"$embedded\""
    )
    user_command = "/bin/bash -c " + shlex.quote(outer_shell)
    output_dir = f"/oss-chengjuntao/artifacts/{args.run_id}"
    envs = {
        "RUN_ID": args.run_id,
        "FASTWAM_TRAINING_TREATMENT": args.treatment,
        "FASTWAM_B4_ATTEMPT_ID": args.attempt_id,
        "FASTWAM_B4_OUTPUT_DIR": output_dir,
        "FASTWAM_B4_OUTPUT_RESERVATION_TIMEOUT": "300",
        "FASTWAM_B4_BOOTSTRAP_SCRIPT": args.bootstrap_script,
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
        "FASTWAM_B4_SOURCE_BUNDLE": bundle_text,
        "FASTWAM_B4_CODE_COMMIT": args.b4_code_commit,
        "FASTWAM_B4_LOCAL_SOURCE_ROOT": "/tmp/fastwam-b4-source-checkouts",
        "FASTWAM_B4_PROVENANCE_MODE": "stat_cmp",
        "FASTWAM_B4_INPUT_CACHE_ROOT": "/tmp/fastwam-b4-input-cache",
        "FASTWAM_B4_SOURCE_WEIGHT": contract["source_weight"],
        "FASTWAM_B4_SOURCE_WEIGHT_BYTES": contract["source_weight_bytes"],
        "FASTWAM_B4_CPFS_SOURCE_ROOT": "/oss-chengjuntao/cpfs-user-chengjuntao",
        "FASTWAM_B4_STATS_SOURCE_ROOT": (
            "/cpfs/user/chengjuntao/datasets/robofactory_multi_robot"
        ),
        "FASTWAM_B4_CPFS_ALLOWLIST": (
            "/oss-chengjuntao/artifacts/fastwam-n234-input-bundles-s42-v1-2023667-"
            "20260802T1235Z/cpfs-whole-file-bundle.sha256"
        ),
        "FASTWAM_LOCAL_DATASET_RELATIVE_ROOT": "datasets/robofactory_multi_robot",
        "FASTWAM_LOCAL_STATS_RELATIVE_PATH": (
            "datasets/robofactory_multi_robot/fastwam_multi_robot_n234_train_s42_stats_v2.json"
        ),
        "FASTWAM_LOCAL_TEXT_EMBEDS_RELATIVE_ROOT": (
            "datasets/robofactory_multi_robot/text_embeds_cache_n234"
        ),
        "FASTWAM_LOCAL_MODEL_CACHE_RELATIVE_ROOT": "checkpoints/FastWAM/model-cache",
        "FASTWAM_LOCAL_VAE_RELATIVE_PATH": (
            "checkpoints/FastWAM/model-cache/DiffSynth-Studio/"
            "Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
        ),
        "FASTWAM_LOCAL_EXPECTED_H5_FILES": "24",
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
    if contract["use_gaussian"]:
        envs.update({
            "FASTWAM_B4_OSS_SOURCE_ROOT": (
                "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"
                "noposplat-c944b498-4a35bc8c/builds/"
                "fastwam-8a035024af96-s42-20260801T230944Z"
            ),
            "FASTWAM_B4_OSS_ALLOWLIST": (
                "/oss-chengjuntao/artifacts/fastwam-n234-input-bundles-s42-v1-"
                "2023667-20260802T1235Z/oss-compact-whole-file-bundle.sha256"
            ),
            "FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT": (
                "compact-s42-13x28x40-fp16-meanalpha-v2"
            ),
        })
    if args.treatment == "n234_vg1h1gau1_cont50k":
        description = (
            "N234 VG1H1GAU1 weights-only continuation: cumulative 5000 to "
            "50000, 45000 fresh-optimizer updates, 3 workers x 8 GPUs"
        )
        tags = {
            "experiment": "N234-VG1H1GAU1-CONT50K",
            "initialization": "GAU1-step5000-weights-only",
            "optimizer": "fresh",
            "provenance": "stat-cmp-no-new-hash",
            "topology": "3x8-world24",
            "schedule": "cumulative-5000-to-50000-save-5000",
        }
    elif args.treatment == "n234_vg1h1gau0_cont50k":
        description = (
            "N234 VG1H1GAU0 weights-only continuation: cumulative 5000 to "
            "50000, 45000 fresh-optimizer updates, 3 workers x 8 GPUs"
        )
        tags = {
            "experiment": "N234-VG1H1GAU0-CONT50K",
            "initialization": "GAU0-step5000-weights-only",
            "optimizer": "fresh",
            "provenance": "stat-cmp-no-new-hash",
            "topology": "3x8-world24",
            "schedule": "cumulative-5000-to-50000-save-5000",
        }
    else:
        description = (
            "B4 weights-only continuation: 3 workers x 8 GPUs, fresh "
            "optimizer, 2500 steps"
        )
        tags = {
            "experiment": "B4",
            "initialization": "GAU1-step5000-weights-only",
            "optimizer": "fresh",
            "provenance": "stat-cmp-no-new-hash",
            "topology": "3x8-world24",
        }

    request = {
        "Accessibility": "PRIVATE",
        "CustomEnvs": [],
        "DataSources": [
            {
                "DataSourceId": "d-a5mu77ymwjio71dkmw",
                "MountAccess": "RO",
                "MountPath": "/cpfs/user/chengjuntao",
            },
            {
                "DataSourceId": "d-n7rly4fll0q2z6v91h",
                "MountAccess": "RW",
                "MountPath": "/oss-chengjuntao",
            },
        ],
        "Description": description,
        "DisplayName": args.run_id,
        "Envs": envs,
        "JobMaxRunningTimeMinutes": max_running_minutes,
        "JobSpecs": [
            {
                "ElasticSpotSpecs": [],
                "Image": IMAGE,
                "LocalMountSpecs": [],
                "PodCount": contract["nodes"],
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
            "Tags": tags,
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
        "training_treatment": args.treatment,
        "launcher_source": {
            "bundle": bundle_text,
            "code_commit": args.b4_code_commit,
            "path": LAUNCHER_PATH,
        },
        "launcher_payload_base64": payload,
        "b4_provenance_contract": {
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
