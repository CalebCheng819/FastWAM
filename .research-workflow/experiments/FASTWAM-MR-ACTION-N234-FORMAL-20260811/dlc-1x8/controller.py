#!/usr/bin/env python3
"""Prepare, submit, and reconcile the three native-agent 1x8 DLC runs.

The default CLI action is ``prepare``.  Preparation performs no DLC API
mutation.  Submission is guarded by an immutable, durable latch written before
the one permitted CreateJob call.  This controller deliberately records paths,
ordinary Git/run identifiers, and filesystem metadata only; it never computes
content digests or chains records together.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import json
import os
import re
import stat
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


CONTRACT = "action_only_native_agents_1x8_v1"
PER_RUN_OSS_BUDGET_BYTES = 62 * 1024**3
SUITE_OSS_BUDGET_BYTES = 190 * 1024**3
PLATFORM_QUOTA_MAX_AGE = timedelta(hours=6)
SUITE_ID = "FASTWAM-MR-ACTION-N234-FORMAL-20260811"
WORKSPACE_ID = "270969"
RESOURCE_ID = "quotaksvqq2oh2pg"
REGION = "cn-beijing"
PROFILE = Path("/root/.aliyun/config.json")
CONTROL_ENTRYPOINT = "submit_from_ssh970.sh"
CONTROL_NODE = "ssh970"

LOCAL_CONTROL_ROOT = Path("/tmp/fastwam-dlc-submit-state/workspace-270969")
DURABLE_CONTROL_ROOT = Path(
    "/oss-chengjuntao/artifacts/fastwam-dlc-submit-ledger/workspace-270969"
)
SUITE_STORAGE_RESERVATION_PATH = (
    DURABLE_CONTROL_ROOT / SUITE_ID / "suite-storage-reservation.json"
)
SOURCE_PREFIX = Path("/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots")
OUTPUT_PREFIX = Path(
    "/oss-chengjuntao/artifacts/fastwam-action-n234-formal-20260811"
)
OSS_ROOT = Path("/oss-chengjuntao")
GAUSSIAN_PREFIX = Path(
    "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2"
)
TEXT_CACHE_ROOT = Path(
    "/oss-chengjuntao/cpfs-user-chengjuntao/datasets/robofactory_multi_robot/"
    "text_embeds_cache_n234_named_20260811"
)
PINNED_PYTHON = Path(
    "/cpfs/user/chengjuntao/venvs/fastwam-gaudp-py310-20260802/bin/python"
)
PINNED_PYTHON_TARGET = Path(
    "/cpfs/user/chengjuntao/runtimes/uv-python/"
    "cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"
)
VAE_SOURCE = Path(
    "/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/model-cache/"
    "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
)
CPFS_DATA_SOURCE = "d-a5mu77ymwjio71dkmw"
OSS_DATA_SOURCE = "d-n7rly4fll0q2z6v91h"
IMAGE = (
    "dsw-registry-vpc.cn-beijing.cr.aliyuncs.com/pai/"
    "pytorch:2.7.1-gpu-py310-cu128-ubuntu22.04-3995b779-1764350887"
)
RUNTIME_REL = Path(
    ".research-workflow/experiments/FASTWAM-MR-ACTION-N234-FORMAL-20260811/"
    "dlc-1x8/runtime.sh"
)
SOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
TERMINAL_STATUSES = {
    "Succeeded",
    "Failed",
    "Stopped",
    "Cancelled",
    "Canceled",
    "Terminated",
}


MEMBERS: dict[str, dict[str, Any]] = {
    "n2": {
        "experiment_id": "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-1K-S42-R1-20260811",
        "run_id": "fastwam-act-n2-placefood-1k-s42-r1-20260811",
        "display_name": "fw-act-n2-placefood-1k-s42-r1",
        "config": "robofactory_multi_robot_ft_n2_placefood_vg0_hub1_gau1_224_3e-5",
        "config_file": "configs/task/robofactory_multi_robot_ft_n2_placefood_vg0_hub1_gau1_224_3e-5.yaml",
        "agent_count": 2,
        "tasks": ["PlaceFood-rf"],
    },
    "n3": {
        "experiment_id": "FASTWAM-MR-FT-ACT-N3-POOL-1K-S42-R1-20260811",
        "run_id": "fastwam-act-n3-pool-1k-s42-r1-20260811",
        "display_name": "fw-act-n3-pool-1k-s42-r1",
        "config": "robofactory_multi_robot_ft_n3_pool_vg0_hub1_gau1_224_3e-5",
        "config_file": "configs/task/robofactory_multi_robot_ft_n3_pool_vg0_hub1_gau1_224_3e-5.yaml",
        "agent_count": 3,
        "tasks": ["ThreeRobotsPlaceShoes-rf", "ThreeRobotsStackCube-rf"],
    },
    "n4": {
        "experiment_id": "FASTWAM-MR-FT-ACT-N4-STACKCUBE-1K-S42-R1-20260811",
        "run_id": "fastwam-act-n4-stackcube-1k-s42-r1-20260811",
        "display_name": "fw-act-n4-stackcube-1k-s42-r1",
        "config": "robofactory_multi_robot_ft_n4_stackcube_vg0_hub1_gau1_224_3e-5",
        "config_file": "configs/task/robofactory_multi_robot_ft_n4_stackcube_vg0_hub1_gau1_224_3e-5.yaml",
        "agent_count": 4,
        "tasks": ["FourRobotsStackCube-rf"],
    },
}

DEFAULT_TEXT_CACHES = {
    task: str(TEXT_CACHE_ROOT / f"{task}.t5_len128.wan22ti2v5b.pt")
    for task in (
        "PlaceFood-rf",
        "ThreeRobotsPlaceShoes-rf",
        "ThreeRobotsStackCube-rf",
        "FourRobotsStackCube-rf",
    )
}

TRUSTED_RUNTIME_B64_ENV = "FASTWAM_TRUSTED_RUNTIME_B64"
TRUSTED_RUNTIME_BYTES_ENV = "FASTWAM_TRUSTED_RUNTIME_BYTES"
TRUSTED_RUNTIME_LOCAL_PATH = "/tmp/fastwam-action-native-agents-runtime.sh"
BOOTSTRAP_PATH = (
    "/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:"
    "/usr/sbin:/usr/bin:/sbin:/bin"
)
BOOTSTRAP_ALLOWED_ENV = (
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "FASTWAM_AGENT_COUNT",
    "FASTWAM_ARTIFACT_INTEGRITY_MODE",
    "FASTWAM_CODE_COMMIT",
    "FASTWAM_DATASET_ROOT",
    "FASTWAM_EXPERIMENT_ID",
    "FASTWAM_EXTERNAL_CONTRACT",
    "FASTWAM_GAUSSIAN_CACHE_DIR",
    "FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR",
    "FASTWAM_INITIAL_CHECKPOINT",
    "FASTWAM_MEMBER",
    "FASTWAM_MAX_OSS_PUBLISH_BYTES",
    "FASTWAM_MIN_TMP_FREE_BYTES",
    "FASTWAM_OSS_OUTPUT_ROOT",
    "FASTWAM_PREPARED_RESERVATION_PATH",
    "FASTWAM_PYTHON",
    "FASTWAM_PYTHON_TARGET",
    "FASTWAM_RUN_ID",
    "FASTWAM_SOURCE_ROOT",
    "FASTWAM_STATS_SOURCE",
    "FASTWAM_SUITE_STORAGE_RESERVATION_PATH",
    "FASTWAM_TASK_CONFIG",
    "FASTWAM_TASKS_JSON",
    "FASTWAM_TEXT_CACHE_MAP_JSON",
    "FASTWAM_TRUSTED_RUNTIME_B64",
    "FASTWAM_TRUSTED_RUNTIME_BYTES",
    "FASTWAM_VAE_SOURCE",
    "GROUP_RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "NPROC_PER_NODE",
    "NVIDIA_VISIBLE_DEVICES",
    "RANK",
    "ROLE_RANK",
    "WORLD_SIZE",
)
TRUSTED_BOOTSTRAP_COMMAND = (
    "unset BASH_ENV ENV PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT "
    "LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT GCONV_PATH LOCPATH NLSPATH;"
    f"PATH={BOOTSTRAP_PATH};export PATH;"
    f"test -L {PINNED_PYTHON} && test -x {PINNED_PYTHON} && "
    f"test \"$(readlink -f -- {PINNED_PYTHON})\" = {PINNED_PYTHON_TARGET} && "
    f"test -f {PINNED_PYTHON_TARGET} && test -x {PINNED_PYTHON_TARGET} && "
    f"test ! -L {PINNED_PYTHON_TARGET} || exit 126;"
    f"exec {PINNED_PYTHON} -B -I -S -c 'import base64,os;"
    f"payload=base64.b64decode(os.environ[\"{TRUSTED_RUNTIME_B64_ENV}\"],validate=True);"
    f"expected=int(os.environ[\"{TRUSTED_RUNTIME_BYTES_ENV}\"]);"
    "\nif len(payload)!=expected:\n raise RuntimeError(\"trusted runtime byte count mismatch\")\n"
    f"path=\"{TRUSTED_RUNTIME_LOCAL_PATH}\";"
    "flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,\"O_NOFOLLOW\",0);"
    "fd=os.open(path,flags,0o500);handle=os.fdopen(fd,\"wb\");"
    "written=handle.write(payload);"
    "\nif written!=len(payload):\n raise RuntimeError(\"trusted runtime short write\")\n"
    "handle.flush();os.fsync(handle.fileno());handle.close();"
    f"allowed={json.dumps(BOOTSTRAP_ALLOWED_ENV)};"
    "clean={key:os.environ[key] for key in allowed if key in os.environ};"
    f"clean[\"PATH\"]=\"{BOOTSTRAP_PATH}\";"
    "clean[\"PYTHONDONTWRITEBYTECODE\"]=\"1\";"
    "os.execve(\"/bin/bash\",[\"bash\",\"--noprofile\",\"--norc\",path],clean)'"
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def stable_read(path: Path) -> tuple[bytes, dict[str, int]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"not a single-link regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fields = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )
    if fields(before) != fields(after):
        raise RuntimeError(f"file changed while reading: {path}")
    return b"".join(chunks), portable_file_stat(after)


def read_json(path: Path) -> tuple[Any, dict[str, int]]:
    payload, metadata = stable_read(path)
    return json.loads(payload), metadata


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(json_bytes(value))
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("zero-byte write while updating local controller state")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def exclusive_write(path: Path, value: Any) -> None:
    """Create one immutable record and verify exclusive-create semantics."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("zero-byte write while creating durable controller record")
            view = view[written:]
    finally:
        os.close(descriptor)
    observed, _ = stable_read(path)
    if observed != payload:
        raise RuntimeError(f"durable readback differs from written record: {path}")
    try:
        duplicate = os.open(path, flags, 0o600)
    except FileExistsError:
        return
    os.close(duplicate)
    raise RuntimeError(f"durable storage did not enforce exclusive creation: {path}")


def require_controller_lock() -> None:
    if os.environ.get("FASTWAM_CONTROL_NODE") != CONTROL_NODE:
        raise RuntimeError(
            f"run through {CONTROL_ENTRYPOINT} on the {CONTROL_NODE} control node"
        )
    if os.environ.get("FASTWAM_LOCK_FD") != "9":
        raise RuntimeError("controller wrapper did not declare file descriptor 9")
    metadata = os.fstat(9)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError("controller lock must be a single-link regular file")
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)


def canonical_direct_child(value: str, *, prefix: Path, label: str) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute() or supplied.parent != prefix:
        raise ValueError(f"{label} must be one direct child of {prefix}")
    if SOURCE_NAME_RE.fullmatch(supplied.name) is None:
        raise ValueError(f"{label} has a non-portable name")
    if supplied.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = supplied.resolve(strict=True)
    if not resolved.is_dir() or str(resolved) != value:
        raise ValueError(f"{label} must be an existing canonical directory")
    return resolved


def canonical_oss_path(value: str, *, kind: str, label: str) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ValueError(f"{label} must be an absolute, non-linked OSS {kind}")
    resolved = supplied.resolve(strict=True)
    if not resolved.is_relative_to(OSS_ROOT) or str(resolved) != value:
        raise ValueError(f"{label} must be canonical beneath {OSS_ROOT}")
    predicate = resolved.is_file if kind == "file" else resolved.is_dir
    if not predicate():
        raise ValueError(f"{label} must be an existing {kind}")
    return resolved


def portable_file_stat(metadata: os.stat_result) -> dict[str, int]:
    """Cross-mount file identity; mount-local device/inode stay transient."""

    return {
        "mode": stat.S_IMODE(metadata.st_mode),
        "bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
    }


def regular_file_metadata(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fields = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or fields(before) != fields(after)
    ):
        raise RuntimeError(f"not a stable single-link regular file: {path}")
    return {"path": str(path), "kind": "file", **portable_file_stat(after)}


def directory_metadata(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"not a non-linked directory: {path}")
    return {
        "path": str(path),
        "kind": "directory",
        "mode": stat.S_IMODE(metadata.st_mode),
        "mtime_ns": metadata.st_mtime_ns,
    }


def source_inventory(root: Path) -> list[dict[str, Any]]:
    """Record sorted path/type/mode/size/time metadata, never source bytes."""

    entries: list[dict[str, Any]] = []
    for path in sorted([root, *root.rglob("*")], key=lambda item: str(item)):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"source snapshot contains a symlink: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
            size = 0
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            kind = "file"
            size = metadata.st_size
        else:
            raise RuntimeError(f"source snapshot contains unsupported entry: {path}")
        entries.append(
            {
                "path": "." if path == root else path.relative_to(root).as_posix(),
                "kind": kind,
                "mode": stat.S_IMODE(metadata.st_mode),
                "bytes": size,
                "mtime_ns": metadata.st_mtime_ns,
            }
        )
    return entries


def validate_source(root: Path, member_names: Iterable[str]) -> list[dict[str, Any]]:
    required = [
        root / "scripts/train.py",
        root / "scripts/accelerate_configs/accelerate_zero2_ds.yaml",
        root / "scripts/ds_configs/ds_zero2_config.json",
        root / RUNTIME_REL,
    ]
    required.extend(root / MEMBERS[name]["config_file"] for name in member_names)
    for path in required:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source snapshot lacks required regular file: {path}")
    return source_inventory(root)


def validate_stats(path: Path, dataset_root: Path) -> dict[str, Any]:
    value, metadata = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("normalization stats JSON must contain an object")
    declared_root = value.get("source_root")
    if not isinstance(declared_root, str) or not Path(declared_root).is_absolute():
        raise ValueError("normalization stats source_root must be an absolute path")
    resolved_root = Path(declared_root).resolve(strict=True)
    if resolved_root != dataset_root:
        raise ValueError(
            "normalization stats source_root must resolve to the selected OSS dataset root: "
            f"declared={declared_root!r} resolved={str(resolved_root)!r} "
            f"expected={str(dataset_root)!r}"
        )
    return {
        "path": str(path),
        "kind": "file",
        "declared_source_root": declared_root,
        "resolved_source_root": str(resolved_root),
        **metadata,
    }


def validate_gaussian_root(path: Path, *, expected_kind: str) -> dict[str, Any]:
    if not path.is_relative_to(GAUSSIAN_PREFIX):
        raise ValueError(f"Gaussian cache must be beneath {GAUSSIAN_PREFIX}")
    manifest_path = path / "manifest.json"
    complete_path = path / "COMPLETE"
    manifest, manifest_metadata = read_json(manifest_path)
    complete_metadata = regular_file_metadata(complete_path)
    schema = manifest.get("schema") if isinstance(manifest, dict) else None
    if not isinstance(schema, dict) or schema.get("cache_kind") != expected_kind:
        raise ValueError(
            f"Gaussian cache kind must be {expected_kind!r}: {path}"
        )
    if int(schema.get("channel_count") or 0) != 13:
        raise ValueError("Gaussian cache must declare 13 channels")
    dimensions = (int(schema.get("height") or 0), int(schema.get("width") or 0))
    selection = manifest.get("selection") or {}
    if expected_kind == "compact":
        if dimensions != (28, 40) or selection.get("mode") != "index":
            raise ValueError("compact Gaussian cache must be 13x28x40 index-selected")
    elif dimensions[0] < 28 or dimensions[1] < 40 or selection.get("mode") != "all":
        raise ValueError("canonical Gaussian fallback must be all-selected and >=28x40")
    return {
        **directory_metadata(path),
        "manifest": {"path": str(manifest_path), **manifest_metadata},
        "completion_marker": complete_metadata,
        "cache_kind": expected_kind,
        "dimensions": [13, dimensions[0], dimensions[1]],
        "selection_mode": selection.get("mode"),
    }


def expected_text_map(args: argparse.Namespace) -> dict[str, str]:
    mapping = {
        "PlaceFood-rf": args.text_cache_placefood,
        "ThreeRobotsPlaceShoes-rf": args.text_cache_three_shoes,
        "ThreeRobotsStackCube-rf": args.text_cache_three_stack,
        "FourRobotsStackCube-rf": args.text_cache_four_stack,
    }
    if any(not value for value in mapping.values()):
        raise ValueError("all four task-to-text-cache paths are required")
    return mapping


def reservation_path(member: str) -> Path:
    return (
        DURABLE_CONTROL_ROOT
        / MEMBERS[member]["experiment_id"]
        / "prepared-reservation.json"
    )


def latch_path(member: str) -> Path:
    return DURABLE_CONTROL_ROOT / MEMBERS[member]["experiment_id"] / "submission-latch.json"


def acknowledgement_path(member: str) -> Path:
    return DURABLE_CONTROL_ROOT / MEMBERS[member]["experiment_id"] / "job-acknowledgement.json"


def local_state_path(member: str) -> Path:
    return LOCAL_CONTROL_ROOT / MEMBERS[member]["experiment_id"] / "state.json"


def output_root(member: str) -> Path:
    return OUTPUT_PREFIX / MEMBERS[member]["run_id"]


def task_text_map(member: str, all_caches: dict[str, str]) -> dict[str, str]:
    return {task: all_caches[task] for task in MEMBERS[member]["tasks"]}


def build_request(
    member: str,
    *,
    source_root: Path,
    source_commit: str,
    dataset_root: Path,
    stats_source: Path,
    initial_checkpoint: Path,
    vae_source: Path,
    gaussian_cache: Path,
    gaussian_fallback_cache: Path,
    text_caches: dict[str, str],
    trusted_runtime: bytes,
) -> dict[str, Any]:
    spec = MEMBERS[member]
    tasks = spec["tasks"]
    selected_text = task_text_map(member, text_caches)
    envs = {
        "FASTWAM_AGENT_COUNT": str(spec["agent_count"]),
        "FASTWAM_ARTIFACT_INTEGRITY_MODE": "metadata_no_hash",
        "FASTWAM_CODE_COMMIT": source_commit,
        "FASTWAM_DATASET_ROOT": str(dataset_root),
        "FASTWAM_EXPERIMENT_ID": spec["experiment_id"],
        "FASTWAM_EXTERNAL_CONTRACT": CONTRACT,
        "FASTWAM_GAUSSIAN_CACHE_DIR": str(gaussian_cache),
        "FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR": str(gaussian_fallback_cache),
        "FASTWAM_INITIAL_CHECKPOINT": str(initial_checkpoint),
        "FASTWAM_MEMBER": member,
        "FASTWAM_MAX_OSS_PUBLISH_BYTES": str(PER_RUN_OSS_BUDGET_BYTES),
        "FASTWAM_MIN_TMP_FREE_BYTES": str(200 * 1024**3),
        "FASTWAM_OSS_OUTPUT_ROOT": str(output_root(member)),
        "FASTWAM_PREPARED_RESERVATION_PATH": str(reservation_path(member)),
        "FASTWAM_PYTHON": str(PINNED_PYTHON),
        "FASTWAM_PYTHON_TARGET": str(PINNED_PYTHON_TARGET),
        "FASTWAM_RUN_ID": spec["run_id"],
        "FASTWAM_SOURCE_ROOT": str(source_root),
        "FASTWAM_STATS_SOURCE": str(stats_source),
        "FASTWAM_SUITE_STORAGE_RESERVATION_PATH": str(
            SUITE_STORAGE_RESERVATION_PATH
        ),
        "FASTWAM_TASK_CONFIG": spec["config"],
        "FASTWAM_TASKS_JSON": json.dumps(tasks, separators=(",", ":")),
        "FASTWAM_TEXT_CACHE_MAP_JSON": json.dumps(
            selected_text, sort_keys=True, separators=(",", ":")
        ),
        TRUSTED_RUNTIME_B64_ENV: base64.b64encode(trusted_runtime).decode("ascii"),
        TRUSTED_RUNTIME_BYTES_ENV: str(len(trusted_runtime)),
        "FASTWAM_VAE_SOURCE": str(vae_source),
        "NPROC_PER_NODE": "8",
    }
    return {
        "Accessibility": "PRIVATE",
        "CustomEnvs": [],
        "DataSources": [
            {
                "DataSourceId": CPFS_DATA_SOURCE,
                "MountAccess": "RO",
                "MountPath": "/cpfs/user/chengjuntao",
            },
            {
                "DataSourceId": OSS_DATA_SOURCE,
                "MountAccess": "RW",
                "MountPath": "/oss-chengjuntao",
            },
        ],
        "Description": (
            f"{spec['experiment_id']}; run_id={spec['run_id']}; "
            f"external_contract={CONTRACT}; native_agents={spec['agent_count']}; "
            "one worker, eight GPUs, 1000 steps, seed/split 42"
        ),
        "DisplayName": spec["display_name"],
        "Envs": envs,
        "JobMaxRunningTimeMinutes": 2160,
        "JobSpecs": [
            {
                "ElasticSpotSpecs": [],
                "Image": IMAGE,
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
        "JobType": "PyTorchJob",
        "Priority": 1,
        "ResourceId": RESOURCE_ID,
        "Settings": {
            "AllocateAllRDMADevices": True,
            "EnableCPUAffinity": False,
            "EnableErrorMonitoringInAIMaster": False,
            "EnableOssAppend": False,
            "EnableRDMA": True,
            "EnableSanityCheck": False,
            "Tags": {
                "experiment_id": spec["experiment_id"],
                "project": "fastwam-multirobot",
                "purpose": "action-native-agent-formal",
                "run_id": spec["run_id"],
            },
        },
        "SuccessPolicy": "AllWorkers",
        "UserCommand": TRUSTED_BOOTSTRAP_COMMAND,
        "WorkspaceId": WORKSPACE_ID,
    }


def validate_request(
    member: str,
    body: dict[str, Any],
    *,
    sdk_models: Any | None = None,
    live: bool = False,
) -> Any:
    spec = MEMBERS[member]
    if body.get("WorkspaceId") != WORKSPACE_ID or body.get("ResourceId") != RESOURCE_ID:
        raise RuntimeError("workspace/resource mismatch")
    if (
        body.get("Accessibility") != "PRIVATE"
        or body.get("CustomEnvs") != []
        or body.get("JobType") != "PyTorchJob"
        or body.get("Priority") != 1
        or body.get("SuccessPolicy") != "AllWorkers"
        or body.get("JobMaxRunningTimeMinutes") != 2160
    ):
        raise RuntimeError("job-level execution contract mismatch")
    job_specs = body.get("JobSpecs") or []
    if len(job_specs) != 1:
        raise RuntimeError("request must contain exactly one job spec")
    job_spec = job_specs[0]
    if job_spec.get("Type") != "Worker" or job_spec.get("PodCount") != 1:
        raise RuntimeError("request must contain exactly one Worker pod")
    resources = job_spec.get("ResourceConfig") or {}
    if resources != {
        "CPU": "126", "GPU": "8", "Memory": "960Gi", "SharedMemory": "960Gi"
    } or job_spec.get("RestartPolicy") != "Never" or job_spec.get("Image") != IMAGE:
        raise RuntimeError("worker resource, image, or restart contract mismatch")
    if any(job_spec.get(name) != [] for name in ("ElasticSpotSpecs", "LocalMountSpecs", "StartupDependencies")):
        raise RuntimeError("worker request contains an unexpected scheduling extension")
    mounts = [
        (item.get("DataSourceId"), item.get("MountPath"), item.get("MountAccess"))
        for item in body.get("DataSources") or []
    ]
    if mounts != [
        (CPFS_DATA_SOURCE, "/cpfs/user/chengjuntao", "RO"),
        (OSS_DATA_SOURCE, "/oss-chengjuntao", "RW"),
    ]:
        raise RuntimeError("datasource mount contract mismatch")
    envs = body.get("Envs") or {}
    fixed = {
        "FASTWAM_AGENT_COUNT": str(spec["agent_count"]),
        "FASTWAM_ARTIFACT_INTEGRITY_MODE": "metadata_no_hash",
        "FASTWAM_EXPERIMENT_ID": spec["experiment_id"],
        "FASTWAM_EXTERNAL_CONTRACT": CONTRACT,
        "FASTWAM_MEMBER": member,
        "FASTWAM_MAX_OSS_PUBLISH_BYTES": str(PER_RUN_OSS_BUDGET_BYTES),
        "FASTWAM_OSS_OUTPUT_ROOT": str(output_root(member)),
        "FASTWAM_PREPARED_RESERVATION_PATH": str(reservation_path(member)),
        "FASTWAM_PYTHON": str(PINNED_PYTHON),
        "FASTWAM_PYTHON_TARGET": str(PINNED_PYTHON_TARGET),
        "FASTWAM_RUN_ID": spec["run_id"],
        "FASTWAM_SUITE_STORAGE_RESERVATION_PATH": str(
            SUITE_STORAGE_RESERVATION_PATH
        ),
        "FASTWAM_TASK_CONFIG": spec["config"],
        "FASTWAM_TASKS_JSON": json.dumps(spec["tasks"], separators=(",", ":")),
        "NPROC_PER_NODE": "8",
    }
    for name, expected in fixed.items():
        if envs.get(name) != expected:
            raise RuntimeError(f"frozen environment mismatch for {name}")
    if COMMIT_RE.fullmatch(str(envs.get("FASTWAM_CODE_COMMIT") or "")) is None:
        raise RuntimeError("source Git commit must be one lowercase full object ID")
    source_literal = str(envs.get("FASTWAM_SOURCE_ROOT") or "")
    if Path(source_literal).parent != SOURCE_PREFIX:
        raise RuntimeError("source root is outside the unique snapshot prefix")
    output_literal = str(envs.get("FASTWAM_OSS_OUTPUT_ROOT") or "")
    if Path(output_literal).parent != OUTPUT_PREFIX:
        raise RuntimeError("output root is outside the suite output prefix")
    try:
        tasks = json.loads(envs["FASTWAM_TASKS_JSON"])
        text_map = json.loads(envs["FASTWAM_TEXT_CACHE_MAP_JSON"])
    except (KeyError, json.JSONDecodeError) as error:
        raise RuntimeError("task or text-cache mapping is not valid JSON") from error
    if tasks != spec["tasks"] or set(text_map) != set(tasks):
        raise RuntimeError("text-cache mapping must exactly match the member task scope")
    for value in text_map.values():
        path = Path(value)
        if not path.is_absolute() or not path.is_relative_to(OSS_ROOT):
            raise RuntimeError("every text-cache mapping must be an absolute OSS path")
    encoded = envs.get(TRUSTED_RUNTIME_B64_ENV)
    declared_bytes = str(envs.get(TRUSTED_RUNTIME_BYTES_ENV) or "")
    if not isinstance(encoded, str) or not declared_bytes.isdecimal():
        raise RuntimeError("trusted runtime payload is missing")
    try:
        runtime = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise RuntimeError("trusted runtime payload is not canonical base64") from error
    if not runtime or len(runtime) != int(declared_bytes):
        raise RuntimeError("trusted runtime byte count mismatch")
    if body.get("UserCommand") != TRUSTED_BOOTSTRAP_COMMAND:
        raise RuntimeError("trusted bootstrap command mismatch")
    expected_settings = {
        "AllocateAllRDMADevices": True,
        "EnableCPUAffinity": False,
        "EnableErrorMonitoringInAIMaster": False,
        "EnableOssAppend": False,
        "EnableRDMA": True,
        "EnableSanityCheck": False,
    }
    settings = body.get("Settings") or {}
    if set(settings) != {*expected_settings, "Tags"} or any(
        settings.get(key) != value for key, value in expected_settings.items()
    ):
        raise RuntimeError("DLC RDMA/settings contract mismatch")
    if (body.get("Settings") or {}).get("Tags", {}).get("run_id") != spec["run_id"]:
        raise RuntimeError("request tag mismatch")
    if live:
        source = canonical_direct_child(source_literal, prefix=SOURCE_PREFIX, label="source root")
        current_runtime, _ = stable_read(source / RUNTIME_REL)
        if current_runtime != runtime:
            raise RuntimeError("request-carried runtime differs from source runtime")
        canonical_oss_path(envs["FASTWAM_DATASET_ROOT"], kind="directory", label="dataset")
        canonical_oss_path(envs["FASTWAM_STATS_SOURCE"], kind="file", label="stats")
        canonical_oss_path(
            envs["FASTWAM_INITIAL_CHECKPOINT"], kind="file", label="initial checkpoint"
        )
        canonical_oss_path(envs["FASTWAM_VAE_SOURCE"], kind="file", label="VAE")
        canonical_oss_path(
            envs["FASTWAM_GAUSSIAN_CACHE_DIR"], kind="directory", label="Gaussian cache"
        )
        canonical_oss_path(
            envs["FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR"],
            kind="directory",
            label="Gaussian fallback cache",
        )
        for task, value in text_map.items():
            canonical_oss_path(value, kind="file", label=f"text cache for {task}")
    if sdk_models is None:
        return None
    request = sdk_models.CreateJobRequest().from_map(body)
    request.validate()
    if request.to_map() != body:
        raise RuntimeError("SDK request roundtrip mismatch")
    return request


def canonical_reservation_intent(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("prepared_at", None)
    return result


def parse_utc_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError(f"{label} must be an explicit UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise RuntimeError(f"{label} is not a valid UTC timestamp") from error
    if parsed.tzinfo != timezone.utc:
        raise RuntimeError(f"{label} must use UTC")
    return parsed


def validate_suite_storage_reservation(value: dict[str, Any]) -> None:
    expected_runs = {member: MEMBERS[member]["run_id"] for member in MEMBERS}
    expected_outputs = {member: str(output_root(member)) for member in MEMBERS}
    if (
        set(value)
        != {
            "schema",
            "suite_id",
            "external_contract",
            "members",
            "member_run_ids",
            "per_run_publish_limit_bytes",
            "suite_reserved_bytes",
            "suite_cap_bytes",
            "platform_quota_snapshot",
            "output_roots",
            "prepared_at",
            "semantics",
        }
        or value.get("schema")
        != "fastwam-action-native-agents-suite-storage-reservation-v1"
        or value.get("suite_id") != SUITE_ID
        or value.get("external_contract") != CONTRACT
        or value.get("members") != list(MEMBERS)
        or value.get("member_run_ids") != expected_runs
        or value.get("per_run_publish_limit_bytes") != PER_RUN_OSS_BUDGET_BYTES
        or value.get("suite_reserved_bytes")
        != len(MEMBERS) * PER_RUN_OSS_BUDGET_BYTES
        or value.get("suite_cap_bytes") != SUITE_OSS_BUDGET_BYTES
        or value.get("output_roots") != expected_outputs
    ):
        raise RuntimeError("suite storage reservation contract mismatch")
    snapshot = value.get("platform_quota_snapshot") or {}
    if set(snapshot) != {
        "quota_bytes",
        "free_bytes",
        "evidence",
        "observed_at",
        "authority",
    }:
        raise RuntimeError("suite platform-quota snapshot schema is invalid")
    quota = snapshot.get("quota_bytes")
    free = snapshot.get("free_bytes")
    evidence = snapshot.get("evidence")
    if (
        snapshot.get("authority") != "platform_quota_not_fuse_df"
        or isinstance(quota, bool)
        or not isinstance(quota, int)
        or isinstance(free, bool)
        or not isinstance(free, int)
        or quota < free
        or free < SUITE_OSS_BUDGET_BYTES
        or not isinstance(evidence, str)
        or len(evidence.strip()) < 8
        or "pai" not in evidence.lower()
        or "console" not in evidence.lower()
        or "df" in evidence.lower()
    ):
        raise RuntimeError("suite platform-quota snapshot is invalid")
    observed_at = parse_utc_timestamp(
        snapshot.get("observed_at"), label="platform quota observed_at"
    )
    prepared_at = parse_utc_timestamp(value.get("prepared_at"), label="suite prepared_at")
    if observed_at > prepared_at:
        raise RuntimeError("platform quota observation cannot postdate preparation")
    if prepared_at - observed_at > PLATFORM_QUOTA_MAX_AGE:
        raise RuntimeError("platform quota observation is too old for formal preparation")


def expected_member_storage_contract() -> dict[str, Any]:
    return {
        "per_run_publish_limit_bytes": PER_RUN_OSS_BUDGET_BYTES,
        "suite_prepare_free_space_floor_bytes": SUITE_OSS_BUDGET_BYTES,
        "suite_reserved_bytes": len(MEMBERS) * PER_RUN_OSS_BUDGET_BYTES,
        "suite_storage_reservation_path": str(SUITE_STORAGE_RESERVATION_PATH),
        "checkpoint_state_kind": "full",
        "publish_step500_state": False,
        "publish_final_step1000_state": True,
        "publish_full_weights_steps": [500, 1000],
        "publish_trainer_local_weight_sidecars": False,
    }


def validate_member_reservation_structure(
    member: str, reservation: dict[str, Any]
) -> dict[str, Any]:
    spec = MEMBERS[member]
    if (
        reservation.get("schema") != "fastwam-action-native-agents-reservation-v1"
        or reservation.get("suite_id") != SUITE_ID
        or reservation.get("external_contract") != CONTRACT
        or reservation.get("member") != member
        or reservation.get("experiment_id") != spec["experiment_id"]
        or reservation.get("run_id") != spec["run_id"]
        or reservation.get("native_agent_count") != spec["agent_count"]
        or reservation.get("tasks") != spec["tasks"]
        or reservation.get("masked_agent_set") is not False
        or reservation.get("output_root") != str(output_root(member))
    ):
        raise RuntimeError(f"prepared reservation identity mismatch: {member}")
    if reservation.get("treatment") != {
        "training_mode": "action_only_cache",
        "video_generation": False,
        "hub_enabled": True,
        "gaussian_enabled": True,
        "trainable_scope": "action",
    }:
        raise RuntimeError(f"prepared treatment mismatch: {member}")
    if reservation.get("schedule") != {
        "max_steps": 1000,
        "save_every": 500,
        "eval_every": 500,
        "offline_eval_num_samples": 32,
        "seed": 42,
        "train_split_seed": 42,
        "val_split_seed": 42,
    }:
        raise RuntimeError(f"prepared schedule mismatch: {member}")
    if reservation.get("hardware") != {
        "workers": 1,
        "gpus_per_worker": 8,
        "total_gpus": 8,
    }:
        raise RuntimeError(f"prepared hardware mismatch: {member}")
    if reservation.get("storage_contract") != expected_member_storage_contract():
        raise RuntimeError(f"prepared storage contract mismatch: {member}")
    request = reservation.get("request")
    if not isinstance(request, dict):
        raise RuntimeError(f"prepared reservation lacks the exact request: {member}")
    validate_request(member, request, live=True)
    return request


def validate_complete_suite_members(
    suite_reservation: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Refuse submission until all three immutable member intents exist."""

    validate_suite_storage_reservation(suite_reservation)
    records: dict[str, dict[str, Any]] = {}
    common_source: dict[str, Any] | None = None
    common_inputs: dict[str, Any] | None = None
    for member in MEMBERS:
        record, _ = read_json(reservation_path(member))
        request = validate_member_reservation_structure(member, record)
        if request["Envs"].get("FASTWAM_SUITE_STORAGE_RESERVATION_PATH") != str(
            SUITE_STORAGE_RESERVATION_PATH
        ):
            raise RuntimeError(f"member does not bind this suite reservation: {member}")
        if suite_reservation["output_roots"].get(member) != record.get("output_root"):
            raise RuntimeError(f"member output differs from suite reservation: {member}")
        source = record.get("source")
        inputs = dict(record.get("inputs") or {})
        inputs.pop("text_caches", None)
        if common_source is None:
            common_source = source
            common_inputs = inputs
        elif source != common_source or inputs != common_inputs:
            raise RuntimeError("suite members do not bind the same source and common inputs")
        selected_text = (record.get("inputs") or {}).get("text_caches") or {}
        if set(selected_text) != set(MEMBERS[member]["tasks"]):
            raise RuntimeError(f"member text-cache evidence differs from task scope: {member}")
        records[member] = record
    return records


def prepare_one(
    member: str,
    *,
    source: Path,
    source_commit: str,
    source_entries: list[dict[str, Any]],
    dataset: Path,
    stats: Path,
    stats_metadata: dict[str, Any],
    checkpoint: Path,
    vae: Path,
    primary: Path,
    primary_metadata: dict[str, Any],
    fallback: Path,
    fallback_metadata: dict[str, Any],
    text_paths: dict[str, str],
    input_file_metadata: dict[str, dict[str, Any]],
    trusted_runtime: bytes,
) -> dict[str, Any]:
    request = build_request(
        member,
        source_root=source,
        source_commit=source_commit,
        dataset_root=dataset,
        stats_source=stats,
        initial_checkpoint=checkpoint,
        vae_source=vae,
        gaussian_cache=primary,
        gaussian_fallback_cache=fallback,
        text_caches=text_paths,
        trusted_runtime=trusted_runtime,
    )
    validate_request(member, request, live=True)
    if output_root(member).exists() or output_root(member).is_symlink():
        raise RuntimeError(f"unique output root already exists: {output_root(member)}")
    selected_text = task_text_map(member, text_paths)
    reservation = {
        "schema": "fastwam-action-native-agents-reservation-v1",
        "suite_id": SUITE_ID,
        "external_contract": CONTRACT,
        "member": member,
        "experiment_id": MEMBERS[member]["experiment_id"],
        "run_id": MEMBERS[member]["run_id"],
        "native_agent_count": MEMBERS[member]["agent_count"],
        "tasks": MEMBERS[member]["tasks"],
        "masked_agent_set": False,
        "treatment": {
            "training_mode": "action_only_cache",
            "video_generation": False,
            "hub_enabled": True,
            "gaussian_enabled": True,
            "trainable_scope": "action",
        },
        "schedule": {
            "max_steps": 1000,
            "save_every": 500,
            "eval_every": 500,
            "offline_eval_num_samples": 32,
            "seed": 42,
            "train_split_seed": 42,
            "val_split_seed": 42,
        },
        "hardware": {"workers": 1, "gpus_per_worker": 8, "total_gpus": 8},
        "storage_contract": {
            "per_run_publish_limit_bytes": PER_RUN_OSS_BUDGET_BYTES,
            "suite_prepare_free_space_floor_bytes": SUITE_OSS_BUDGET_BYTES,
            "suite_reserved_bytes": len(MEMBERS) * PER_RUN_OSS_BUDGET_BYTES,
            "suite_storage_reservation_path": str(SUITE_STORAGE_RESERVATION_PATH),
            "checkpoint_state_kind": "full",
            "publish_step500_state": False,
            "publish_final_step1000_state": True,
            "publish_full_weights_steps": [500, 1000],
            "publish_trainer_local_weight_sidecars": False,
        },
        "source": {
            "root": str(source),
            "git_commit": source_commit,
            "inventory": source_entries,
        },
        "inputs": {
            "dataset": directory_metadata(dataset),
            "normalization_stats": stats_metadata,
            "initial_checkpoint": input_file_metadata["initial_checkpoint"],
            "vae": input_file_metadata["vae"],
            "gaussian_primary": primary_metadata,
            "gaussian_fallback": fallback_metadata,
            "text_caches": {
                task: input_file_metadata[f"text:{task}"] for task in selected_text
            },
        },
        "output_root": str(output_root(member)),
        "request": request,
        "prepared_at": utc_now(),
        "semantics": (
            "external generic reservation; trainer terminal contract fields remain null; "
            "terminal success is granted only by the runtime receipt"
        ),
    }
    destination = reservation_path(member)
    if destination.exists():
        observed, _ = read_json(destination)
        if canonical_reservation_intent(observed) != canonical_reservation_intent(reservation):
            raise RuntimeError(f"existing immutable reservation differs: {destination}")
        return {"member": member, "status": "ALREADY_PREPARED", "path": str(destination)}
    if latch_path(member).exists() or acknowledgement_path(member).exists():
        raise RuntimeError(f"member has submission state without a reservation: {member}")
    exclusive_write(destination, reservation)
    state = {
        "schema": "fastwam-dlc-local-controller-state-v1",
        "phase": "PREPARED",
        "member": member,
        "experiment_id": MEMBERS[member]["experiment_id"],
        "run_id": MEMBERS[member]["run_id"],
        "reservation_path": str(destination),
        "cloud_create_calls": 0,
        "updated_at": utc_now(),
    }
    atomic_write(local_state_path(member), state)
    return {"member": member, "status": "PREPARED", "path": str(destination)}


def prepare(args: argparse.Namespace) -> None:
    if args.member:
        raise ValueError("formal prepare is suite-atomic; do not select a member")
    names = list(MEMBERS)
    suite_reserved_bytes = len(MEMBERS) * PER_RUN_OSS_BUDGET_BYTES
    if suite_reserved_bytes > SUITE_OSS_BUDGET_BYTES:
        raise RuntimeError("three per-run caps exceed the frozen suite budget")
    if args.platform_oss_quota_bytes < args.platform_oss_free_bytes:
        raise ValueError("platform OSS quota bytes cannot be below free bytes")
    if args.platform_oss_free_bytes < SUITE_OSS_BUDGET_BYTES:
        raise RuntimeError(
            "the authoritative platform snapshot is below the suite floor: "
            f"free={args.platform_oss_free_bytes} required={SUITE_OSS_BUDGET_BYTES}"
        )
    if not args.platform_oss_quota_evidence.strip():
        raise ValueError("--platform-oss-quota-evidence is required")
    if not args.platform_oss_observed_at.strip():
        raise ValueError("--platform-oss-observed-at is required")
    # The venv entry point is intentionally a symlink.  Bind both the logical
    # venv path (which selects that environment) and its exact regular CPFS
    # interpreter target; merely accepting an arbitrary executable symlink
    # would make the prepared runtime mutable by retargeting it.
    try:
        resolved_python = PINNED_PYTHON.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"pinned CPFS Python cannot be resolved: {PINNED_PYTHON}") from exc
    if (
        not PINNED_PYTHON.is_symlink()
        or resolved_python != PINNED_PYTHON_TARGET
        or PINNED_PYTHON_TARGET.is_symlink()
        or not PINNED_PYTHON_TARGET.is_file()
        or not os.access(PINNED_PYTHON_TARGET, os.X_OK)
    ):
        raise RuntimeError(
            "pinned CPFS Python logical path or resolved executable target mismatch: "
            f"{PINNED_PYTHON} -> {resolved_python}"
        )
    source_commit = args.source_commit.strip().lower()
    if COMMIT_RE.fullmatch(source_commit) is None:
        raise ValueError("--source-commit must be a lowercase full 40-character Git object ID")
    source = canonical_direct_child(args.source_root, prefix=SOURCE_PREFIX, label="source root")
    source_entries = validate_source(source, names)
    trusted_runtime, _ = stable_read(source / RUNTIME_REL)
    dataset = canonical_oss_path(args.dataset_root, kind="directory", label="dataset")
    stats = canonical_oss_path(args.stats_source, kind="file", label="normalization stats")
    stats_metadata = validate_stats(stats, dataset)
    checkpoint = canonical_oss_path(
        args.initial_checkpoint, kind="file", label="initial checkpoint"
    )
    vae = canonical_oss_path(args.vae_source, kind="file", label="VAE")
    primary = canonical_oss_path(args.gaussian_cache, kind="directory", label="Gaussian cache")
    fallback = canonical_oss_path(
        args.gaussian_fallback_cache, kind="directory", label="Gaussian fallback cache"
    )
    if primary == fallback:
        raise ValueError("Gaussian primary and fallback roots must differ")
    primary_metadata = validate_gaussian_root(primary, expected_kind="compact")
    fallback_metadata = validate_gaussian_root(fallback, expected_kind="canonical")
    text_paths = expected_text_map(args)
    input_file_metadata = {
        "initial_checkpoint": regular_file_metadata(checkpoint),
        "vae": regular_file_metadata(vae),
    }
    for task, value in text_paths.items():
        path = canonical_oss_path(value, kind="file", label=f"text cache for {task}")
        input_file_metadata[f"text:{task}"] = regular_file_metadata(path)
        text_paths[task] = str(path)
    suite_reservation = {
        "schema": "fastwam-action-native-agents-suite-storage-reservation-v1",
        "suite_id": SUITE_ID,
        "external_contract": CONTRACT,
        "members": list(MEMBERS),
        "member_run_ids": {
            member: MEMBERS[member]["run_id"] for member in MEMBERS
        },
        "per_run_publish_limit_bytes": PER_RUN_OSS_BUDGET_BYTES,
        "suite_reserved_bytes": suite_reserved_bytes,
        "suite_cap_bytes": SUITE_OSS_BUDGET_BYTES,
        "platform_quota_snapshot": {
            "quota_bytes": args.platform_oss_quota_bytes,
            "free_bytes": args.platform_oss_free_bytes,
            "evidence": args.platform_oss_quota_evidence.strip(),
            "observed_at": args.platform_oss_observed_at.strip(),
            "authority": "platform_quota_not_fuse_df",
        },
        "output_roots": {
            member: str(output_root(member)) for member in MEMBERS
        },
        "prepared_at": utc_now(),
        "semantics": (
            "one suite-wide reservation prevents the three formal workers from "
            "independently claiming the same platform quota headroom"
        ),
    }
    validate_suite_storage_reservation(suite_reservation)
    # Prepare owns the shared, otherwise empty parent prefix.  Member output
    # roots remain absent until their worker has completed all three worlds.
    try:
        os.mkdir(OUTPUT_PREFIX, 0o700)
    except FileExistsError:
        pass
    if canonical_oss_path(
        str(OUTPUT_PREFIX), kind="directory", label="formal output prefix"
    ) != OUTPUT_PREFIX:
        raise RuntimeError("formal output prefix canonicalization mismatch")
    # All three unique destinations are checked before the first suite record
    # is created, so preparation cannot reserve shared capacity for a member
    # whose durable output already exists.
    for member in names:
        destination = output_root(member)
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"unique output root already exists: {destination}")
        if not reservation_path(member).exists() and (
            latch_path(member).exists() or acknowledgement_path(member).exists()
        ):
            raise RuntimeError(
                f"member has submission state without a reservation: {member}"
            )
    suite_already_published = SUITE_STORAGE_RESERVATION_PATH.exists()
    if suite_already_published:
        observed_suite, _ = read_json(SUITE_STORAGE_RESERVATION_PATH)
        validate_suite_storage_reservation(observed_suite)
        if canonical_reservation_intent(observed_suite) != canonical_reservation_intent(
            suite_reservation
        ):
            raise RuntimeError("existing immutable suite storage reservation differs")
    outcomes = [
        prepare_one(
            member,
            source=source,
            source_commit=source_commit,
            source_entries=source_entries,
            dataset=dataset,
            stats=stats,
            stats_metadata=stats_metadata,
            checkpoint=checkpoint,
            vae=vae,
            primary=primary,
            primary_metadata=primary_metadata,
            fallback=fallback,
            fallback_metadata=fallback_metadata,
            text_paths=text_paths,
            input_file_metadata=input_file_metadata,
            trusted_runtime=trusted_runtime,
        )
        for member in names
    ]
    # The suite marker is the commit record.  Publish it only after every
    # member reservation exists and all three have been read back as one
    # coherent 186-GiB intent.  A crash before this point cannot authorize a
    # partial suite for submission.
    validate_complete_suite_members(suite_reservation)
    if not suite_already_published:
        exclusive_write(SUITE_STORAGE_RESERVATION_PATH, suite_reservation)
    observed_suite, _ = read_json(SUITE_STORAGE_RESERVATION_PATH)
    validate_complete_suite_members(observed_suite)
    print(json.dumps({"action": "prepare", "cloud_mutations": 0, "members": outcomes}, indent=2))


def load_sdk() -> tuple[Any, Any, Any]:
    from alibabacloud_credentials.client import Client as CredentialClient
    from alibabacloud_credentials.models import Config as CredentialConfig
    from alibabacloud_pai_dlc20201203 import models
    from alibabacloud_pai_dlc20201203.client import Client
    from alibabacloud_tea_openapi.models import Config
    from alibabacloud_tea_util.models import RuntimeOptions

    profile, _ = read_json(PROFILE)
    current = profile.get("current")
    selected = next(
        (item for item in profile.get("profiles", []) if item.get("name") == current),
        None,
    )
    if (
        not selected
        or selected.get("mode") != "CredentialsURI"
        or not selected.get("credentials_uri")
    ):
        raise RuntimeError("active Alibaba Cloud profile must use CredentialsURI")
    credential = CredentialClient(
        CredentialConfig(
            type="credentials_uri", credentials_uri=selected["credentials_uri"]
        )
    )
    client = Client(
        Config(
            credential=credential,
            region_id=REGION,
            endpoint="pai-dlc.cn-beijing.aliyuncs.com",
        )
    )
    return client, models, RuntimeOptions


def runtime_options(runtime_cls: Any) -> Any:
    return runtime_cls(
        autoretry=False,
        max_attempts=1,
        connect_timeout=10000,
        read_timeout=30000,
    )


def list_jobs(client: Any, models: Any, runtime_cls: Any) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    total: int | None = None
    page = 1
    while total is None or len(jobs) < total:
        response = client.list_jobs_with_options(
            models.ListJobsRequest(
                workspace_id=WORKSPACE_ID,
                page_number=page,
                page_size=100,
                order="desc",
                sort_by="GmtCreateTime",
            ),
            {},
            runtime_options(runtime_cls),
        )
        body = response.body.to_map()
        page_jobs = body.get("Jobs") or []
        if total is None:
            total = int(body.get("TotalCount") or 0)
        jobs.extend(page_jobs)
        if not page_jobs:
            break
        page += 1
    if total is None or len(jobs) != total:
        raise RuntimeError(f"ListJobs pagination mismatch: observed={len(jobs)} total={total}")
    identifiers = [str(job.get("JobId") or "") for job in jobs]
    if "" in identifiers or len(identifiers) != len(set(identifiers)):
        raise RuntimeError("ListJobs returned missing or duplicate job identifiers")
    return jobs


def get_job(client: Any, models: Any, runtime_cls: Any, job_id: str) -> dict[str, Any]:
    response = client.get_job_with_options(
        job_id,
        models.GetJobRequest(need_detail=True),
        {},
        runtime_options(runtime_cls),
    )
    body = response.body.to_map()
    if str(body.get("JobId") or "") != job_id:
        raise RuntimeError("GetJob identity mismatch")
    return body


def requested_subset(observed: Any, requested: Any) -> bool:
    if isinstance(requested, dict):
        return isinstance(observed, dict) and all(
            key in observed and requested_subset(observed[key], value)
            for key, value in requested.items()
        )
    if isinstance(requested, list):
        return (
            isinstance(observed, list)
            and len(observed) == len(requested)
            and all(requested_subset(left, right) for left, right in zip(observed, requested))
        )
    return type(observed) is type(requested) and observed == requested


def custom_env_projection_matches(job: dict[str, Any], request: dict[str, Any]) -> bool:
    """Accept only PAI's observed public projection of the requested Envs map."""

    requested_envs = request.get("Envs")
    observed_custom = job.get("CustomEnvs")
    if request.get("CustomEnvs") != [] or not isinstance(requested_envs, dict):
        return False
    if not isinstance(observed_custom, list) or len(observed_custom) != len(requested_envs):
        return False
    projection: dict[str, Any] = {}
    for item in observed_custom:
        if not isinstance(item, dict) or set(item) != {"Key", "Value", "Visible"}:
            return False
        key = item.get("Key")
        if not isinstance(key, str) or key in projection or item.get("Visible") != "public":
            return False
        projection[key] = item.get("Value")
    return projection == requested_envs


def datasource_projection_matches(job: dict[str, Any], request: dict[str, Any]) -> bool:
    """Accept only the exact GetJob datasource projection observed from PAI."""

    requested_sources = request.get("DataSources")
    observed_sources = job.get("DataSources")
    if not isinstance(requested_sources, list) or not isinstance(observed_sources, list):
        return False
    if len(requested_sources) != len(observed_sources):
        return False
    for observed, requested in zip(observed_sources, requested_sources):
        if (
            not isinstance(requested, dict)
            or set(requested) != {"DataSourceId", "MountAccess", "MountPath"}
            or requested.get("MountAccess") not in {"RO", "RW"}
            or not isinstance(observed, dict)
            or set(observed) != {"DataSourceId", "MountPath", "Uri"}
            or observed.get("DataSourceId") != requested.get("DataSourceId")
            or observed.get("MountPath") != requested.get("MountPath")
            or observed.get("Uri") != ""
        ):
            return False
    return True


def exact_job(job: dict[str, Any], request: dict[str, Any]) -> bool:
    """Match one frozen request under the closed, observed PAI GetJob projection."""

    if not isinstance(job, dict) or not isinstance(request, dict):
        return False
    if (
        not requested_subset(job.get("WorkspaceId"), request.get("WorkspaceId"))
        or not requested_subset(job.get("ResourceId"), request.get("ResourceId"))
        or not custom_env_projection_matches(job, request)
        or not datasource_projection_matches(job, request)
    ):
        return False
    omitted_by_service = {"JobMaxRunningTimeMinutes", "SuccessPolicy"}
    if not omitted_by_service.issubset(request):
        return False
    for key in omitted_by_service:
        if key in job and not requested_subset(job[key], request[key]):
            return False
    special = {
        "WorkspaceId", "ResourceId", "CustomEnvs", "DataSources", *omitted_by_service
    }
    return all(
        key in job and requested_subset(job[key], value)
        for key, value in request.items()
        if key not in special
    )


def publish_acknowledgement(member: str, job: dict[str, Any], *, source: str) -> dict[str, Any]:
    payload = {
        "schema": "fastwam-dlc-job-acknowledgement-v1",
        "member": member,
        "experiment_id": MEMBERS[member]["experiment_id"],
        "run_id": MEMBERS[member]["run_id"],
        "job_id": str(job["JobId"]),
        "status": str(job.get("Status") or "Unknown"),
        "source": source,
        "recorded_at": utc_now(),
    }
    destination = acknowledgement_path(member)
    if destination.exists():
        observed, _ = read_json(destination)
        if observed.get("job_id") != payload["job_id"]:
            raise RuntimeError("durable acknowledgement binds a different job")
        return observed
    exclusive_write(destination, payload)
    return payload


def validate_formal_terminal_output(member: str) -> dict[str, Any]:
    """Validate scientific completion independently of DLC cloud status."""

    spec = MEMBERS[member]
    root = canonical_oss_path(
        str(output_root(member)), kind="directory", label="formal member output"
    )
    terminal, terminal_metadata = read_json(root / "receipts/terminal.json")
    complete, complete_metadata = read_json(root / "COMPLETE")
    if complete != {
        "schema": "fastwam-action-native-agents-complete-v1",
        "terminal_receipt": "receipts/terminal.json",
        "status": "COMPLETE",
    }:
        raise RuntimeError("formal COMPLETE marker mismatch")
    terminal_keys = {
        "schema", "external_contract", "member", "experiment_id", "run_id",
        "native_agent_count", "tasks", "masked_agent_set", "treatment",
        "schedule", "hardware", "checkpoint_state_kind",
        "phase3_fresh_world_load", "prepared_reservation_path",
        "suite_storage_reservation_path", "artifacts", "status",
    }
    if not isinstance(terminal, dict) or set(terminal) != terminal_keys:
        raise RuntimeError("formal terminal receipt key set mismatch")
    fixed = {
        "schema": "fastwam-action-native-agents-terminal-receipt-v1",
        "external_contract": CONTRACT,
        "member": member,
        "experiment_id": spec["experiment_id"],
        "run_id": spec["run_id"],
        "native_agent_count": spec["agent_count"],
        "tasks": spec["tasks"],
        "masked_agent_set": False,
        "treatment": {
            "training_mode": "action_only_cache", "video_generation": False,
            "hub_enabled": True, "gaussian_enabled": True,
            "trainable_scope": "action",
        },
        "schedule": {
            "max_steps": 1000, "save_every": 500, "eval_every": 500,
            "offline_eval_num_samples": 32, "seed": 42,
            "train_split_seed": 42, "val_split_seed": 42,
        },
        "hardware": {"workers": 1, "gpus_per_worker": 8, "total_gpus": 8},
        "checkpoint_state_kind": "full",
        "phase3_fresh_world_load": {
            "world_size": 8, "restored_global_step": 1000,
            "zero_update": True, "source": "local_final_state",
        },
        "prepared_reservation_path": str(reservation_path(member)),
        "suite_storage_reservation_path": str(SUITE_STORAGE_RESERVATION_PATH),
        "status": "COMPLETE",
    }
    for key, expected in fixed.items():
        if terminal.get(key) != expected:
            raise RuntimeError(f"formal terminal identity mismatch: {key}")

    artifacts = terminal.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("formal terminal artifact list is absent")
    declared: dict[str, int] = {}
    required = {
        "checkpoints/weights/step_000500.pt",
        "checkpoints/weights/step_001000.pt",
        "checkpoints/state/step_001000/trainer_state.json",
        "receipts/step500-resume.json",
        "receipts/step1000-fresh-load.json",
        "eval/offline-eval.json",
        "logs/phase1-train-to-500.log",
        "logs/phase2-resume-to-1000.log",
        "logs/phase3-fresh-load-step1000.log",
    }
    exact_non_state = required - {"checkpoints/state/step_001000/trainer_state.json"}
    for item in artifacts:
        if not isinstance(item, dict) or set(item) != {"path", "bytes"}:
            raise RuntimeError("formal artifact descriptor mismatch")
        relative = item.get("path")
        size = item.get("bytes")
        if (
            not isinstance(relative, str) or not relative
            or Path(relative).is_absolute() or Path(relative).as_posix() != relative
            or any(part in ("", ".", "..") for part in Path(relative).parts)
            or isinstance(size, bool) or not isinstance(size, int) or size < 0
            or relative in declared
        ):
            raise RuntimeError("unsafe or duplicate formal artifact descriptor")
        if (
            relative not in exact_non_state
            and not relative.startswith("checkpoints/state/step_001000/")
        ):
            raise RuntimeError(f"formal artifact is outside the allowlist: {relative}")
        if relative.endswith(".pt.manifest.json") or relative.endswith(".pt.COMPLETE"):
            raise RuntimeError("local trainer weight sidecar leaked into OSS")
        path = root / relative
        metadata = regular_file_metadata(path)
        if metadata["path"] != str(path) or metadata["bytes"] != size:
            raise RuntimeError(f"formal artifact metadata mismatch: {relative}")
        declared[relative] = size
    if not required.issubset(declared):
        raise RuntimeError("formal output lacks a required terminal artifact")

    observed: set[str] = set()
    for path in root.rglob("*"):
        info = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"formal output contains a symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"formal output contains an unsupported entry: {relative}")
        observed.add(relative)
    expected_files = set(declared) | {"receipts/terminal.json", "COMPLETE"}
    if observed != expected_files:
        raise RuntimeError("formal output filesystem differs from terminal allowlist")
    published_bytes = (
        sum(declared.values()) + terminal_metadata["bytes"] + complete_metadata["bytes"]
    )
    if published_bytes > PER_RUN_OSS_BUDGET_BYTES:
        raise RuntimeError("formal output exceeds its immutable per-run budget")

    phase2, _ = read_json(root / "receipts/step500-resume.json")
    phase3, _ = read_json(root / "receipts/step1000-fresh-load.json")
    for receipt, step in ((phase2, 500), (phase3, 1000)):
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_name") != "fastwam-recovery-load-receipt"
            or receipt.get("schema_version") != 1
            or receipt.get("integrity_mode") != "metadata_no_hash"
            or receipt.get("accelerator_load_state_returned") is not True
            or receipt.get("restored_global_step") != step
            or receipt.get("world_size") != 8
        ):
            raise RuntimeError(f"native recovery evidence mismatch at step {step}")
    eval_receipt, _ = read_json(root / "eval/offline-eval.json")
    records = eval_receipt.get("records") if isinstance(eval_receipt, dict) else None
    if (
        not isinstance(eval_receipt, dict)
        or eval_receipt.get("schema") != "fastwam-multi-robot-offline-eval-receipt-v1"
        or eval_receipt.get("agent_count") != spec["agent_count"]
        or eval_receipt.get("tasks") != spec["tasks"]
        or not isinstance(records, list)
        or [record.get("step") for record in records if isinstance(record, dict)] != [500, 1000]
    ):
        raise RuntimeError("offline evaluation terminal evidence mismatch")
    final_state, _ = read_json(root / "checkpoints/state/step_001000/trainer_state.json")
    if not isinstance(final_state, dict) or final_state.get("global_step") != 1000:
        raise RuntimeError("published trainer state is not terminal step1000")
    state_records = final_state.get("evaluation_records")
    selected = [
        record for record in state_records if isinstance(record, dict)
        and record.get("step") in (500, 1000)
    ] if isinstance(state_records, list) else []
    if selected != records:
        raise RuntimeError("offline eval receipt differs from final trainer state")
    return {
        "status": "SCIENTIFIC_COMPLETE", "output_root": str(root),
        "published_bytes": published_bytes, "artifact_files": len(declared),
    }


def validate_reservation_live(
    member: str,
    reservation: dict[str, Any],
    *,
    require_output_absent: bool = True,
) -> dict[str, Any]:
    suite_reservation, _ = read_json(SUITE_STORAGE_RESERVATION_PATH)
    suite_members = validate_complete_suite_members(suite_reservation)
    if canonical_reservation_intent(suite_members[member]) != canonical_reservation_intent(
        reservation
    ):
        raise RuntimeError("passed reservation differs from immutable suite member")
    request = validate_member_reservation_structure(member, reservation)
    canonical_oss_path(
        str(OUTPUT_PREFIX), kind="directory", label="formal output prefix"
    )
    source = canonical_direct_child(
        reservation["source"]["root"], prefix=SOURCE_PREFIX, label="source root"
    )
    if source_inventory(source) != reservation["source"]["inventory"]:
        raise RuntimeError("source snapshot metadata changed after preparation")
    envs = request["Envs"]
    inputs = reservation.get("inputs") or {}
    dataset = canonical_oss_path(
        envs["FASTWAM_DATASET_ROOT"], kind="directory", label="dataset"
    )
    if directory_metadata(dataset) != inputs.get("dataset"):
        raise RuntimeError("dataset metadata changed after preparation")
    stats = canonical_oss_path(
        envs["FASTWAM_STATS_SOURCE"], kind="file", label="normalization stats"
    )
    if validate_stats(stats, dataset) != inputs.get("normalization_stats"):
        raise RuntimeError("normalization stats changed after preparation")
    for key, env_name, label in (
        ("initial_checkpoint", "FASTWAM_INITIAL_CHECKPOINT", "initial checkpoint"),
        ("vae", "FASTWAM_VAE_SOURCE", "VAE"),
    ):
        path = canonical_oss_path(envs[env_name], kind="file", label=label)
        if regular_file_metadata(path) != inputs.get(key):
            raise RuntimeError(f"{label} metadata changed after preparation")
    for key, env_name, kind in (
        ("gaussian_primary", "FASTWAM_GAUSSIAN_CACHE_DIR", "compact"),
        ("gaussian_fallback", "FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR", "canonical"),
    ):
        path = canonical_oss_path(envs[env_name], kind="directory", label=key)
        if validate_gaussian_root(path, expected_kind=kind) != inputs.get(key):
            raise RuntimeError(f"{key} metadata changed after preparation")
    text_map = json.loads(envs["FASTWAM_TEXT_CACHE_MAP_JSON"])
    for task, literal in text_map.items():
        path = canonical_oss_path(literal, kind="file", label=f"text cache for {task}")
        if regular_file_metadata(path) != (inputs.get("text_caches") or {}).get(task):
            raise RuntimeError(f"text cache metadata changed after preparation: {task}")
    if require_output_absent and (output_root(member).exists() or output_root(member).is_symlink()):
        raise RuntimeError("unique output root exists before the DLC worker starts")
    return request


def reconcile_member(
    member: str,
    *,
    client: Any,
    models: Any,
    runtime_cls: Any,
    jobs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reservation, _ = read_json(reservation_path(member))
    request = reservation.get("request")
    if not isinstance(request, dict):
        raise RuntimeError("reservation has no request")
    if acknowledgement_path(member).exists():
        ack, _ = read_json(acknowledgement_path(member))
        job = get_job(client, models, runtime_cls, str(ack["job_id"]))
        if not exact_job(job, request):
            raise RuntimeError("acknowledged job no longer matches the prepared request")
        platform_status = job.get("Status")
        formal = (
            validate_formal_terminal_output(member)
            if platform_status == "Succeeded"
            else {"status": "NOT_COMPLETE"}
        )
        return {
            "member": member,
            "job_id": ack["job_id"],
            "platform_status": platform_status,
            "formal": formal,
        }
    observed_jobs = jobs if jobs is not None else list_jobs(client, models, runtime_cls)
    matches = [job for job in observed_jobs if exact_job(job, request)]
    if len(matches) > 1:
        raise RuntimeError(f"multiple exact jobs match member {member}")
    if not matches:
        if latch_path(member).exists():
            return {"member": member, "status": "LATCHED_JOB_NOT_YET_OBSERVED"}
        return {"member": member, "status": "PREPARED_NOT_SUBMITTED"}
    job_id = str(matches[0]["JobId"])
    detail = get_job(client, models, runtime_cls, job_id)
    if not exact_job(detail, request):
        raise RuntimeError("GetJob detail differs from the prepared request")
    ack = publish_acknowledgement(member, detail, source="reconcile")
    platform_status = detail.get("Status")
    formal = (
        validate_formal_terminal_output(member)
        if platform_status == "Succeeded"
        else {"status": "NOT_COMPLETE"}
    )
    return {
        "member": member,
        "job_id": ack["job_id"],
        "platform_status": platform_status,
        "formal": formal,
    }


def submit(args: argparse.Namespace) -> None:
    member = args.member
    if args.confirm_experiment_id != MEMBERS[member]["experiment_id"]:
        raise ValueError("--confirm-experiment-id must exactly match the selected member")
    reservation, _ = read_json(reservation_path(member))
    already_latched = acknowledgement_path(member).exists() or latch_path(member).exists()
    request_body = validate_reservation_live(
        member, reservation, require_output_absent=not already_latched
    )
    client, models, runtime_cls = load_sdk()
    request = validate_request(member, request_body, sdk_models=models, live=True)
    if already_latched:
        result = reconcile_member(
            member, client=client, models=models, runtime_cls=runtime_cls
        )
        print(json.dumps({"action": "submit", "create_calls": 0, **result}, indent=2))
        return
    jobs = list_jobs(client, models, runtime_cls)
    matches = [job for job in jobs if exact_job(job, request_body)]
    if len(matches) > 1:
        raise RuntimeError("multiple exact pre-existing jobs match the request")
    if matches:
        detail = get_job(client, models, runtime_cls, str(matches[0]["JobId"]))
        ack = publish_acknowledgement(member, detail, source="pre_submit_scan")
        print(
            json.dumps(
                {"action": "submit", "create_calls": 0, "member": member, **ack},
                indent=2,
            )
        )
        return
    latch = {
        "schema": "fastwam-dlc-permanent-submission-latch-v1",
        "member": member,
        "experiment_id": MEMBERS[member]["experiment_id"],
        "run_id": MEMBERS[member]["run_id"],
        "reservation_path": str(reservation_path(member)),
        "latched_at": utc_now(),
        "semantics": "one CreateJob call permitted; never retry after this record exists",
    }
    exclusive_write(latch_path(member), latch)
    state = {
        "schema": "fastwam-dlc-local-controller-state-v1",
        "phase": "CREATE_CALL_IN_PROGRESS",
        "member": member,
        "cloud_create_calls": 1,
        "updated_at": utc_now(),
    }
    atomic_write(local_state_path(member), state)
    # This is the only CreateJob mutation call in the controller.  It is never
    # placed inside a retry loop and SDK autoretry is disabled above.
    response = client.create_job_with_options(
        request, {}, runtime_options(runtime_cls)
    )
    response_body = response.body.to_map()
    job_id = str(response_body.get("JobId") or "")
    if not job_id:
        raise RuntimeError("CreateJob returned no JobId; permanent latch forbids retry")
    detail = get_job(client, models, runtime_cls, job_id)
    if not exact_job(detail, request_body):
        raise RuntimeError("created job detail does not match the prepared request")
    ack = publish_acknowledgement(member, detail, source="CreateJob_then_GetJob")
    state.update(
        {
            "phase": "ACKNOWLEDGED",
            "job_id": job_id,
            "job_status": detail.get("Status"),
            "updated_at": utc_now(),
        }
    )
    atomic_write(local_state_path(member), state)
    print(json.dumps({"action": "submit", "create_calls": 1, **ack}, indent=2))


def reconcile(args: argparse.Namespace) -> None:
    names = args.member or list(MEMBERS)
    client, models, runtime_cls = load_sdk()
    jobs = list_jobs(client, models, runtime_cls)
    results = [
        reconcile_member(
            member, client=client, models=models, runtime_cls=runtime_cls, jobs=jobs
        )
        for member in names
    ]
    print(json.dumps({"action": "reconcile", "cloud_mutations": 0, "members": results}, indent=2))


def show(args: argparse.Namespace) -> None:
    names = args.member or list(MEMBERS)
    records = []
    for member in names:
        record: dict[str, Any] = {
            "member": member,
            "experiment_id": MEMBERS[member]["experiment_id"],
            "run_id": MEMBERS[member]["run_id"],
        }
        for label, path in (
            ("reservation", reservation_path(member)),
            ("latch", latch_path(member)),
            ("acknowledgement", acknowledgement_path(member)),
            ("local_state", local_state_path(member)),
        ):
            if path.exists():
                value, _ = read_json(path)
                record[label] = value
            else:
                record[label] = None
        records.append(record)
    print(json.dumps({"action": "show", "cloud_mutations": 0, "members": records}, indent=2))


def add_member_selection(parser: argparse.ArgumentParser, *, single: bool = False) -> None:
    parser.add_argument(
        "--member",
        choices=tuple(MEMBERS),
        action=None if single else "append",
        required=single,
        help="one suite member; repeat for a subset when supported",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="validate and durably reserve; no DLC API mutation")
    add_member_selection(prepare_parser)
    prepare_parser.add_argument("--source-root", required=True)
    prepare_parser.add_argument("--source-commit", required=True)
    prepare_parser.add_argument("--dataset-root", required=True)
    prepare_parser.add_argument("--stats-source", required=True)
    prepare_parser.add_argument("--initial-checkpoint", required=True)
    prepare_parser.add_argument("--vae-source", default=str(VAE_SOURCE))
    prepare_parser.add_argument("--gaussian-cache", required=True)
    prepare_parser.add_argument("--gaussian-fallback-cache", required=True)
    prepare_parser.add_argument("--platform-oss-quota-bytes", required=True, type=int)
    prepare_parser.add_argument("--platform-oss-free-bytes", required=True, type=int)
    prepare_parser.add_argument("--platform-oss-quota-evidence", required=True)
    prepare_parser.add_argument("--platform-oss-observed-at", required=True)
    prepare_parser.add_argument(
        "--text-cache-placefood", default=DEFAULT_TEXT_CACHES["PlaceFood-rf"]
    )
    prepare_parser.add_argument(
        "--text-cache-three-shoes",
        default=DEFAULT_TEXT_CACHES["ThreeRobotsPlaceShoes-rf"],
    )
    prepare_parser.add_argument(
        "--text-cache-three-stack",
        default=DEFAULT_TEXT_CACHES["ThreeRobotsStackCube-rf"],
    )
    prepare_parser.add_argument(
        "--text-cache-four-stack",
        default=DEFAULT_TEXT_CACHES["FourRobotsStackCube-rf"],
    )
    submit_parser = commands.add_parser("submit", help="one latched CreateJob call")
    add_member_selection(submit_parser, single=True)
    submit_parser.add_argument("--confirm-experiment-id", required=True)
    reconcile_parser = commands.add_parser("reconcile", help="read-only cloud reconciliation")
    add_member_selection(reconcile_parser)
    show_parser = commands.add_parser("show", help="show local and durable controller records")
    add_member_selection(show_parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] not in {"prepare", "submit", "reconcile", "show", "-h", "--help"}:
        values.insert(0, "prepare")
    args = build_parser().parse_args(values)
    require_controller_lock()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "submit":
        submit(args)
    elif args.command == "reconcile":
        reconcile(args)
    elif args.command == "show":
        show(args)
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
