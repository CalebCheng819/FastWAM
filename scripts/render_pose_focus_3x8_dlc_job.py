#!/usr/bin/env python3
"""Render, but never submit, a PAI DLC CreateJob manifest for POSE_FOCUS."""

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
    "robofactory_placefood_semantic_phase_p5_224_5e-6",
    "robofactory_placefood_spatial_semantic_p6_224_5e-6",
    "robofactory_placefood_task_gaussian_relation_p7_224_5e-6",
    "robofactory_placefood_relation_gripcontact_p8_224_5e-6",
    "robofactory_placefood_spatial_gripcontact_p9_224_5e-6",
    "robofactory_placefood_spatial_gripcontact_p10_lowaux_224_5e-6",
    "robofactory_placefood_crossagent_gaussian_p12_224_5e-6",
    "robofactory_placefood_metric_gaussian_p13_224_5e-6",
)
P13_TASK_PROFILE = "robofactory_placefood_metric_gaussian_p13_224_5e-6"
DEFAULT_METRIC_SOURCE_ROOT = (
    "/oss-chengjuntao/artifacts/"
    "fastwam-placefood-metric-geometry-60x80-s42-v1-20260815"
)
R5_SOURCE_WEIGHT = (
    "/oss-chengjuntao/artifacts/fastwam-action-n234-formal-r5-20260812/"
    "fastwam-act-n2-placefood-1k-s42-r5-20260812/checkpoints/weights/step_001000.pt"
)
P1_SOURCE_WEIGHT = (
    "/oss-chengjuntao/artifacts/fastwam-placefood-posefocus-r5-s42-24g-r2-20260813/"
    "checkpoints/weights/step_001000.pt"
)
P2_SOURCE_WEIGHT = (
    "/oss-chengjuntao/artifacts/fastwam-placefood-phase-x0-r5-s42-24g-r1-20260813/"
    "checkpoints/weights/step_001000.pt"
)
P5_SOURCE_WEIGHT = (
    "/oss-chengjuntao/artifacts/fastwam-placefood-semantic-phase-p5-s42-24g-r1-20260814/"
    "checkpoints/weights/step_001000.pt"
)
P6_SOURCE_WEIGHT = (
    "/oss-chengjuntao/artifacts/fastwam-placefood-spatial-semantic-p6-s42-24g-r1-20260814/"
    "checkpoints/weights/step_001000.pt"
)
P7_SOURCE_WEIGHT = (
    "/oss-chengjuntao/artifacts/fastwam-placefood-task-gaussian-relation-p7-s42-24g-r1-20260814/"
    "checkpoints/weights/step_001000.pt"
)
P10_SOURCE_WEIGHT = (
    "/oss-chengjuntao/artifacts/fastwam-placefood-spatial-gripcontact-"
    "p10-lowaux-s42-8g-r1-20260814/checkpoints/weights/step_001000.pt"
)
SOURCE_WEIGHTS = {
    TASK_PROFILES[0]: (R5_SOURCE_WEIGHT, 12047407619, "R5-action-step1000-weights-only"),
    TASK_PROFILES[1]: (R5_SOURCE_WEIGHT, 12047407619, "R5-action-step1000-weights-only"),
    TASK_PROFILES[2]: (P1_SOURCE_WEIGHT, 12047407619, "P1-pose-focus-step1000-weights-only"),
    TASK_PROFILES[3]: (P2_SOURCE_WEIGHT, 12047407619, "P2-action-step1000-weights-only"),
    TASK_PROFILES[4]: (P5_SOURCE_WEIGHT, 12047407619, "P5-action-step1000-weights-only"),
    TASK_PROFILES[5]: (P6_SOURCE_WEIGHT, 12047407747, "P6-action-step1000-weights-only"),
    TASK_PROFILES[6]: (P7_SOURCE_WEIGHT, 12055814467, "P7-action-step1000-weights-only"),
    TASK_PROFILES[7]: (P6_SOURCE_WEIGHT, 12047407747, "P6-action-step1000-weights-only"),
    TASK_PROFILES[8]: (P6_SOURCE_WEIGHT, 12047407747, "P6-action-step1000-weights-only"),
    TASK_PROFILES[9]: (P10_SOURCE_WEIGHT, 12047407747, "P10-action-step1000-weights-only"),
    TASK_PROFILES[10]: (P10_SOURCE_WEIGHT, 12047407747, "P10-action-step1000-weights-only"),
}
OBJECTIVES = {
    TASK_PROFILES[0]: "active-agent-continuous-pose",
    TASK_PROFILES[1]: "robot0-phase-clean-x0",
    TASK_PROFILES[2]: "robot0-gaussian-spatial-cross-attention",
    TASK_PROFILES[3]: "placefood-task-semantic-phase-sampling",
    TASK_PROFILES[4]: "placefood-spatial-gaussian-semantic-phase",
    TASK_PROFILES[5]: "placefood-task-conditioned-gaussian-relation",
    TASK_PROFILES[6]: "placefood-relation-gripper-contact-proxy",
    TASK_PROFILES[7]: "placefood-spatial-gripper-contact-proxy",
    TASK_PROFILES[8]: "placefood-spatial-gripper-contact-lowaux",
    TASK_PROFILES[9]: "placefood-crossagent-gaussian-lowaux",
    TASK_PROFILES[10]: "placefood-metric-depth-spatial-lowaux",
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
        "--metric-source-root",
        default=DEFAULT_METRIC_SOURCE_ROOT,
        help="Completed metric-geometry cache root used by the P13 task profile.",
    )
    parser.add_argument("--worker-count", type=int, choices=(1, 3), default=3)
    parser.add_argument("--gpus-per-worker", type=int, choices=(4, 8), default=8)
    parser.add_argument(
        "--source-weight",
        choices=tuple(sorted({item[0] for item in SOURCE_WEIGHTS.values()})),
        help="Optional explicit source; it must match the audited task-profile source.",
    )
    parser.add_argument("--max-running-minutes", type=int, default=10080)
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
    if args.worker_count != 1 and args.gpus_per_worker != 8:
        raise SystemExit("four-GPU topology is supported only with one worker")

    metric_source_root = args.metric_source_root
    metric_prefix = "/oss-chengjuntao/artifacts/"
    metric_name = metric_source_root.removeprefix(metric_prefix)
    if (
        not metric_source_root.startswith(metric_prefix)
        or not SAFE_ID.fullmatch(metric_name)
        or str(pathlib.PurePosixPath(metric_source_root)) != metric_source_root
    ):
        raise SystemExit(
            "metric-source-root must be one direct safe child below "
            "/oss-chengjuntao/artifacts"
        )
    if args.task_profile != P13_TASK_PROFILE and metric_source_root != DEFAULT_METRIC_SOURCE_ROOT:
        raise SystemExit("metric-source-root override is supported only by the P13 task profile")

    source_path, source_bytes, initialization = SOURCE_WEIGHTS[args.task_profile]
    if args.source_weight is not None and args.source_weight != source_path:
        raise SystemExit("source-weight does not match the audited task-profile source")

    launcher_bytes = launcher_from_bundle(args.pose_focus_source_bundle, args.pose_focus_code_commit)
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
    require_erdma = args.worker_count > 1
    world_size = args.worker_count * args.gpus_per_worker
    cpu_per_worker = "63" if args.gpus_per_worker == 4 else "126"
    memory_per_worker = "480Gi" if args.gpus_per_worker == 4 else "960Gi"
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
        "FASTWAM_POSE_FOCUS_SOURCE_WEIGHT": source_path,
        "FASTWAM_POSE_FOCUS_SOURCE_WEIGHT_BYTES": str(source_bytes),
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
        "FASTWAM_POSE_FOCUS_METRIC_SOURCE_ROOT": metric_source_root,
        "FASTWAM_POSE_FOCUS_METRIC_ALLOWLIST": f"{metric_source_root}/stat-cmp.allowlist",
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
        "FASTWAM_PREFLIGHT_REQUIRE_ERDMA": "1" if require_erdma else "0",
        "FASTWAM_PREFLIGHT_TIMEOUT": "7200",
        "FASTWAM_PREFLIGHT_OUTER_TIMEOUT": "7260",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET",
        "NPROC_PER_NODE": str(args.gpus_per_worker),
        "FASTWAM_POSE_FOCUS_EXPECTED_WORKERS": str(args.worker_count),
        "FASTWAM_POSE_FOCUS_EXPECTED_GPUS_PER_WORKER": str(args.gpus_per_worker),
    }
    if require_erdma:
        envs.update({
            "FASTWAM_ERDMA_BUNDLE_ROOT": (
                "/oss-chengjuntao/artifacts/erdma-userspace-56.2-1.0.3"
            ),
            "FASTWAM_ERDMA_EXPECTED_VERSION": "56.2-1.0.3",
            "NCCL_IB_HCA": "erdma",
        })
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
            "PlaceFood action-only continuation: "
            f"{args.task_profile}, {args.worker_count} workers x "
            f"{args.gpus_per_worker} GPUs, 1000 steps"
        ),
        "DisplayName": args.run_id,
        "Envs": envs,
        "JobMaxRunningTimeMinutes": args.max_running_minutes,
        "JobSpecs": [
            {
                "ElasticSpotSpecs": [],
                "Image": IMAGE,
                "LocalMountSpecs": [],
                "PodCount": args.worker_count,
                "ResourceConfig": {
                    "CPU": cpu_per_worker,
                    "GPU": str(args.gpus_per_worker),
                    "Memory": memory_per_worker,
                    "SharedMemory": memory_per_worker,
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
            "AllocateAllRDMADevices": require_erdma,
            "EnableCPUAffinity": False,
            "EnableErrorMonitoringInAIMaster": False,
            "EnableOssAppend": False,
            "EnableRDMA": require_erdma,
            "EnableSanityCheck": False,
            "Tags": {
                "experiment": "POSE_FOCUS",
                "initialization": initialization,
                "optimizer": "fresh",
                "task": "PlaceFood-rf",
                "objective": OBJECTIVES[args.task_profile],
                "provenance": "stat-cmp-no-new-hash",
                "topology": (
                    f"{args.worker_count}x{args.gpus_per_worker}-world{world_size}"
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
        "launcher_payload_base64": payload,
        "source_weight": {
            "path": source_path,
            "bytes": source_bytes,
            "initialization": initialization,
        },
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
