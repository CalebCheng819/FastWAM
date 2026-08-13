#!/usr/bin/env python3
"""Render, but never submit, a PAI DLC job for POSE_FOCUS 1x4, 1x8, or 3x8."""

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
# POSE_FOCUS data/checkpoint provenance uses path, stat, and byte comparison instead.
LEGACY_DEPENDENCY_MANIFEST_ID = "b740e7224ad38628c12347ff0d36cb85dea45095f335ec032a52f07fcade7ee5"
LEGACY_DEPENDENCY_RUNTIME_LOCK_ID = "d495f1a1192ced91edd7df2794a94fe0ffb67526a279570d5cf3649d59c0d360"
LEGACY_DEPENDENCY_CACHE_HELPER_ID = "89dc9d7302f2edc1320b5f08f0516d5d2e9c6a176705642cf2f57756a1ae22ae"
LAUNCHER_PATH = "scripts/launch_pose_focus_3x8_dlc.sh"
TASK_PROFILES = (
    "robofactory_placefood_pose_focus_r5_224_5e-6",
    "robofactory_placefood_pose_phase_x0_r5_224_5e-6",
    "robofactory_placefood_gaussian_spatial_p4_224_5e-6",
)
R5_SOURCE_WEIGHT = (
    "/oss-chengjuntao/artifacts/fastwam-action-n234-formal-r5-20260812/"
    "fastwam-act-n2-placefood-1k-s42-r5-20260812/checkpoints/weights/step_001000.pt"
)
P1_SOURCE_WEIGHT = (
    "/oss-chengjuntao/artifacts/fastwam-placefood-posefocus-r5-s42-24g-r2-20260813/"
    "checkpoints/weights/step_001000.pt"
)
AUDITED_SOURCE_WEIGHTS = {
    R5_SOURCE_WEIGHT: {
        "bytes": 12047407619,
        "initialization": "R5-action-step1000-weights-only",
    },
    P1_SOURCE_WEIGHT: {
        "bytes": 12047407619,
        "initialization": "P1-pose-focus-step1000-weights-only",
    },
}


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
    parser.add_argument("--pose-focus-source-bundle", type=pathlib.Path, required=True)
    parser.add_argument("--pose-focus-code-commit", required=True)
    parser.add_argument("--task-profile", choices=TASK_PROFILES, default=TASK_PROFILES[0])
    parser.add_argument(
        "--source-weight",
        choices=tuple(AUDITED_SOURCE_WEIGHTS),
        default=R5_SOURCE_WEIGHT,
    )
    parser.add_argument("--max-running-minutes", type=int, default=10080)
    parser.add_argument("--pod-count", type=int, choices=(1, 3), default=3)
    parser.add_argument("--gpus-per-pod", type=int, choices=(4, 8), default=8)
    parser.add_argument(
        "--allow-local-bundle-for-tests",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def launcher_from_bundle(bundle: pathlib.Path, commit: str) -> bytes:
    """Read the launcher at an exact bundled commit without using this worktree."""
    if not bundle.is_file() or bundle.is_symlink():
        raise SystemExit("pose_focus-source-bundle must be an existing regular non-symlink file")
    with tempfile.TemporaryDirectory(prefix="fastwam-pose_focus-render-") as directory:
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
                raise SystemExit(f"failed to read POSE_FOCUS source bundle: {detail}")
        resolved = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "FETCH_HEAD^{commit}"],
            text=True,
            capture_output=True,
            check=False,
        )
        if resolved.returncode or resolved.stdout.strip() != commit:
            raise SystemExit("pose_focus-code-commit did not resolve exactly from pose_focus-source-bundle")
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
            raise SystemExit("bundled POSE_FOCUS launcher has an unexpected interpreter header")
        return extracted.stdout


def main() -> int:
    args = parse_args()
    if not SAFE_ID.fullmatch(args.run_id) or not SAFE_ID.fullmatch(args.attempt_id):
        raise SystemExit("run-id and attempt-id must be safe unique identifiers")
    if args.max_running_minutes <= 0:
        raise SystemExit("max-running-minutes must be positive")
    if args.pod_count == 3 and args.gpus_per_pod != 8:
        raise SystemExit("3x4 is not an audited POSE_FOCUS topology")
    for name in ("bootstrap_script", "offline_env_source_root", "offline_env_manifest"):
        if not getattr(args, name).startswith("/oss-chengjuntao/"):
            raise SystemExit(f"{name.replace('_', '-')} must be below /oss-chengjuntao/")
    bundle_text = str(args.pose_focus_source_bundle)
    if not args.allow_local_bundle_for_tests and not bundle_text.startswith("/oss-chengjuntao/"):
        raise SystemExit("pose_focus-source-bundle must be below /oss-chengjuntao/")
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
        raise SystemExit("pose_focus-source-bundle must name a Git .bundle file")
    if not HEX40.fullmatch(args.offline_code_commit):
        raise SystemExit("offline-code-commit must be an exact lowercase Git revision")
    if not HEX40.fullmatch(args.pose_focus_code_commit):
        raise SystemExit("pose_focus-code-commit must be an exact lowercase Git revision")

    launcher_bytes = launcher_from_bundle(args.pose_focus_source_bundle, args.pose_focus_code_commit)
    source_weight = AUDITED_SOURCE_WEIGHTS[args.source_weight]
    payload = base64.b64encode(launcher_bytes).decode("ascii")
    outer_shell = (
        "set -euo pipefail; "
        "embedded=/tmp/fastwam-pose_focus-outer-launcher.$$; "
        "umask 077; "
        f"printf '%s' {shlex.quote(payload)} | base64 --decode > \"$embedded\"; "
        "exec /bin/bash \"$embedded\""
    )
    user_command = "/bin/bash -c " + shlex.quote(outer_shell)
    output_dir = f"/oss-chengjuntao/artifacts/{args.run_id}"
    envs = {
        "RUN_ID": args.run_id,
        "FASTWAM_POSE_FOCUS_ATTEMPT_ID": args.attempt_id,
        "FASTWAM_POSE_FOCUS_OUTPUT_DIR": output_dir,
        "FASTWAM_POSE_FOCUS_OUTPUT_RESERVATION_TIMEOUT": "300",
        "FASTWAM_POSE_FOCUS_BOOTSTRAP_SCRIPT": args.bootstrap_script,
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
        "FASTWAM_POSE_FOCUS_SOURCE_BUNDLE": bundle_text,
        "FASTWAM_POSE_FOCUS_CODE_COMMIT": args.pose_focus_code_commit,
        "FASTWAM_POSE_FOCUS_TASK_PROFILE": args.task_profile,
        "FASTWAM_POSE_FOCUS_EXPECTED_POD_COUNT": str(args.pod_count),
        "FASTWAM_POSE_FOCUS_EXPECTED_GPUS_PER_NODE": str(args.gpus_per_pod),
        "FASTWAM_POSE_FOCUS_SOURCE_WEIGHT": args.source_weight,
        "FASTWAM_POSE_FOCUS_SOURCE_WEIGHT_BYTES": str(source_weight["bytes"]),
        "FASTWAM_POSE_FOCUS_LOCAL_SOURCE_ROOT": "/tmp/fastwam-pose_focus-source-checkouts",
        "FASTWAM_POSE_FOCUS_PROVENANCE_MODE": "stat_cmp",
        "FASTWAM_POSE_FOCUS_INPUT_CACHE_ROOT": "/tmp/fastwam-pose_focus-input-cache",
        "FASTWAM_POSE_FOCUS_CPFS_SOURCE_ROOT": "/oss-chengjuntao/cpfs-user-chengjuntao",
        "FASTWAM_POSE_FOCUS_STATS_SOURCE_ROOT": (
            "/cpfs/user/chengjuntao/datasets/robofactory_multi_robot"
        ),
        "FASTWAM_POSE_FOCUS_CPFS_ALLOWLIST": (
            "/oss-chengjuntao/artifacts/fastwam-n234-input-bundles-s42-v1-2023667-"
            "20260802T1235Z/cpfs-whole-file-bundle.sha256"
        ),
        "FASTWAM_POSE_FOCUS_OSS_SOURCE_ROOT": (
            "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/"
            "noposplat-c944b498-4a35bc8c/builds/fastwam-8a035024af96-s42-20260801T230944Z"
        ),
        "FASTWAM_POSE_FOCUS_OSS_ALLOWLIST": (
            "/oss-chengjuntao/artifacts/fastwam-n234-input-bundles-s42-v1-2023667-"
            "20260802T1235Z/oss-compact-whole-file-bundle.sha256"
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
        "FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT": "compact-s42-13x28x40-fp16-meanalpha-v2",
        "FASTWAM_LOCAL_EXPECTED_H5_FILES": "24",
        "FASTWAM_ERDMA_BUNDLE_ROOT": "/oss-chengjuntao/artifacts/erdma-userspace-56.2-1.0.3",
        "FASTWAM_ERDMA_EXPECTED_VERSION": "56.2-1.0.3",
        "FASTWAM_PREFLIGHT_REQUIRE_ERDMA": "1",
        "FASTWAM_PREFLIGHT_TIMEOUT": "7200",
        "FASTWAM_PREFLIGHT_OUTER_TIMEOUT": "7260",
        "NCCL_IB_HCA": "erdma",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET",
        "NPROC_PER_NODE": str(args.gpus_per_pod),
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
        "Description": (
            "PlaceFood R5 action-only treatment: "
            f"{args.task_profile}, {args.pod_count} workers x "
            f"{args.gpus_per_pod} GPUs, 1000 steps"
        ),
        "DisplayName": args.run_id,
        "Envs": envs,
        "JobMaxRunningTimeMinutes": args.max_running_minutes,
        "JobSpecs": [
            {
                "ElasticSpotSpecs": [],
                "Image": IMAGE,
                "LocalMountSpecs": [],
                "PodCount": args.pod_count,
                "ResourceConfig": {
                    "CPU": "63" if args.gpus_per_pod == 4 else "126",
                    "GPU": str(args.gpus_per_pod),
                    "Memory": "480Gi" if args.gpus_per_pod == 4 else "960Gi",
                    "SharedMemory": "480Gi" if args.gpus_per_pod == 4 else "960Gi",
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
                "experiment": "POSE_FOCUS",
                "initialization": source_weight["initialization"],
                "optimizer": "fresh",
                "task": "PlaceFood-rf",
                "objective": (
                    "robot0-phase-clean-x0"
                    if args.task_profile == TASK_PROFILES[1]
                    else (
                        "robot0-gaussian-spatial-cross-attention"
                        if args.task_profile == TASK_PROFILES[2]
                        else "active-agent-continuous-pose"
                    )
                ),
                "provenance": "stat-cmp-no-new-hash",
                "topology": (
                    "1x4-world4-accum6"
                    if args.gpus_per_pod == 4
                    else (
                        "1x8-world8-accum3"
                        if args.pod_count == 1
                        else "3x8-world24-accum1"
                    )
                ),
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
            "/mnt/workspace/tools/pai-control-py312/"
            "20260717-credentials1.0.10-dlc1.9.2-aiworkspace8.2.0/bin/python"
        ),
        "region": "cn-beijing",
        "endpoint": "pai-dlc.cn-beijing.aliyuncs.com",
        "launcher_source": {
            "bundle": bundle_text,
            "code_commit": args.pose_focus_code_commit,
            "path": LAUNCHER_PATH,
        },
        "source_weight": {
            "path": args.source_weight,
            "bytes": source_weight["bytes"],
            "initialization": source_weight["initialization"],
        },
        "launcher_payload_base64": payload,
        "pose_focus_provenance_contract": {
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
