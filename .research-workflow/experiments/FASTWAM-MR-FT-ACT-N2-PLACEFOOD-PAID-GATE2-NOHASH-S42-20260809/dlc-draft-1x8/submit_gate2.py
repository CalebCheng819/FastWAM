#!/usr/bin/env python3
"""Single-writer DLC Gate2 launcher.  No cloud mutation happens in prepare/reconcile."""

from __future__ import annotations

import argparse
import base64
import fcntl
import json
import os
import re
import stat
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-S42-20260809"
WORKSPACE_ID = "270969"
RESOURCE_ID = "quotaksvqq2oh2pg"
REQUESTED_GPUS = 8
REGION = "cn-beijing"
PROFILE = Path("/root/.aliyun/config.json")
LOCAL_CONTROL_ROOT = Path("/tmp/fastwam-dlc-submit-state/workspace-270969")
DURABLE_CONTROL_ROOT = Path(
    "/oss-chengjuntao/artifacts/fastwam-dlc-submit-ledger/workspace-270969"
)
SOURCE_PREFIX = Path("/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots")
SOURCE_SNAPSHOT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
OUTPUT_PREFIX = Path("/oss-chengjuntao/artifacts/fastwam-gate2-nohash-results")
STATS_SOURCE_PREFIX = Path("/oss-chengjuntao")
GAUSSIAN_CACHE_PREFIX = Path(
    "/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2"
)
TASK = "robofactory_multi_robot_ft_n2_placefood_vg0_hub1_gau1_224_3e-5_nohash_gate"
SUBMISSION_TAG_PREFIX = "fastwam-gate2-nohash-s42"
DISPLAY_NAME_PREFIX = "fw-g2-nh-s42"
CONTROL_ENTRYPOINT = "submit_from_ssh970.sh"
# Variants may set this to require one exact, canonical source snapshot.
APPROVED_SOURCE_ROOT: Path | None = None
REAL_PREFLIGHT_REL = Path(
    ".research-workflow/experiments/"
    f"{EXPERIMENT_ID}/real_data_nohash_preflight.py"
)
ENTRYPOINT_REL = Path(
    ".research-workflow/experiments/"
    f"{EXPERIMENT_ID}/dlc-draft-1x8/runtime.sh"
)
TRUSTED_RUNTIME_B64_ENV = "FASTWAM_GATE2_TRUSTED_RUNTIME_B64"
TRUSTED_RUNTIME_BYTES_ENV = "FASTWAM_GATE2_TRUSTED_RUNTIME_BYTES"
TRUSTED_RUNTIME_LOCAL_PATH = "/tmp/fastwam-gate2-trusted-runtime.sh"
VAE_SOURCE = Path(
    "/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/model-cache/"
    "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
)
VAE_SOURCE_BYTES = 1_409_401_152
PINNED_PYTHON = Path(
    "/cpfs/user/chengjuntao/venvs/fastwam-gaudp-py310-20260802/bin/python"
)
PINNED_PYTHON_TARGET = Path(
    "/cpfs/user/chengjuntao/runtimes/uv-python/"
    "cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"
)
BOOTSTRAP_PATH = (
    "/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:"
    "/usr/sbin:/usr/bin:/sbin:/bin"
)
BOOTSTRAP_ALLOWED_ENV = (
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "FASTWAM_ARTIFACT_INTEGRITY_MODE",
    "FASTWAM_DATASET_ROOT",
    "FASTWAM_EXPERIMENT_ID",
    "FASTWAM_GATE2_ENTRYPOINT",
    "FASTWAM_GATE2_TRUSTED_RUNTIME_B64",
    "FASTWAM_GATE2_TRUSTED_RUNTIME_BYTES",
    "FASTWAM_GAUSSIAN_CACHE_DIR",
    "FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR",
    "FASTWAM_INITIAL_CHECKPOINT",
    "FASTWAM_MIN_TMP_FREE_BYTES",
    "FASTWAM_N234_NOHASH_STATS_SOURCE",
    "FASTWAM_OSS_OUTPUT_ROOT",
    "FASTWAM_PREPARED_BINDING_PATH",
    "FASTWAM_PYTHON",
    "FASTWAM_PYTHON_TARGET",
    "FASTWAM_SOURCE_ROOT",
    "FASTWAM_SUBMISSION_TAG",
    "FASTWAM_TASK_CONFIG",
    "FASTWAM_TEXT_CACHE_DIR",
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
STRUCTURED_EVIDENCE_REL = Path(
    ".research-workflow/experiments/"
    f"{EXPERIMENT_ID}/dlc-draft-1x8/gate2_structured_evidence.py"
)
IMAGE = (
    "dsw-registry-vpc.cn-beijing.cr.aliyuncs.com/pai/"
    "pytorch:2.7.1-gpu-py310-cu128-ubuntu22.04-3995b779-1764350887"
)
CPFS_SOURCE = "d-a5mu77ymwjio71dkmw"
OSS_SOURCE = "d-n7rly4fll0q2z6v91h"
TERMINAL = {"Succeeded", "Failed", "Stopped", "Cancelled", "Canceled", "Terminated"}
ATTEMPT_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_data(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(json_data(value))
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def durable_exclusive_write(path: Path, value: Any) -> dict[str, int]:
    """Publish one immutable OSS record without rename or directory fsync."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = json_data(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)
    observed, metadata = stable_read(path)
    if observed != payload:
        raise RuntimeError(f"durable record readback mismatch: {path}")
    try:
        duplicate_fd = os.open(path, flags, 0o600)
    except FileExistsError:
        pass
    else:
        os.close(duplicate_fd)
        raise RuntimeError(f"durable store does not enforce exclusive create: {path}")
    return metadata


def descriptor(st: os.stat_result) -> dict[str, int]:
    return {
        "device": st.st_dev,
        "inode": st.st_ino,
        "mode": st.st_mode,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def stable_read(path: Path, expected: dict[str, int] | None = None) -> tuple[bytes, dict[str, int]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"not a single-link regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    observed = descriptor(after)
    if descriptor(before) != observed:
        raise RuntimeError(f"file changed while reading: {path}")
    if expected is not None and observed != expected:
        raise RuntimeError(f"prepared file metadata changed: {path}")
    return b"".join(chunks), observed


def read_json(path: Path, expected: dict[str, int] | None = None) -> tuple[Any, dict[str, int]]:
    raw, observed = stable_read(path, expected)
    return json.loads(raw), observed


def require_single_writer() -> None:
    if os.environ.get("FASTWAM_CONTROL_NODE") != "ssh970":
        raise RuntimeError(f"run only through {CONTROL_ENTRYPOINT} on the SSH970 control node")
    if os.environ.get("FASTWAM_LOCK_FD") != "9":
        raise RuntimeError("the SSH970 local flock is not declared")
    lock_stat = os.fstat(9)
    if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
        raise RuntimeError("SSH970 controller lock must be a single-link regular file")
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    LOCAL_CONTROL_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)


def experiment_root() -> Path:
    return LOCAL_CONTROL_ROOT / EXPERIMENT_ID


def durable_experiment_root() -> Path:
    return DURABLE_CONTROL_ROOT / EXPERIMENT_ID


def prepared_binding_path() -> Path:
    return durable_experiment_root() / "prepared-binding.json"


def submission_latch_path() -> Path:
    return durable_experiment_root() / "submission-latch.json"


def create_response_path() -> Path:
    return durable_experiment_root() / "create-response.json"


def acknowledgement_path() -> Path:
    return durable_experiment_root() / "acknowledgement.json"


def acquire_submission_latch(attempt: str) -> dict[str, Any]:
    """Bind the experiment permanently to one attempt before CreateJob is possible."""

    path = submission_latch_path()
    payload = {
        "schema": "fastwam-dlc-submission-latch-v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt": attempt,
        "created_at": now(),
        "create_call_disposition": "MAY_HAVE_BEEN_SENT",
        "semantics": "at_most_one_CreateJob_call_for_this_experiment_id",
    }
    # FileExistsError is deliberately not recovered, including for the same
    # attempt: only the process that just created the O_EXCL record may submit.
    durable_exclusive_write(path, payload)
    return payload


def require_submission_latch(attempt: str) -> dict[str, Any]:
    payload, _ = read_json(submission_latch_path())
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("attempt") != attempt
    ):
        raise RuntimeError("submission latch identity mismatch")
    return payload


def attempt_paths(attempt: str) -> tuple[Path, Path]:
    if ATTEMPT_RE.fullmatch(attempt) is None:
        raise ValueError("invalid attempt UUID")
    root = LOCAL_CONTROL_ROOT / EXPERIMENT_ID / attempt
    return root / "request.json", root / "state.json"


def require_prepared_binding(attempt: str) -> dict[str, Any]:
    binding, _ = read_json(prepared_binding_path())
    if (
        binding.get("schema") != "fastwam-dlc-prepared-binding-v1"
        or binding.get("experiment_id") != EXPERIMENT_ID
        or binding.get("attempt") != attempt
        or not isinstance(binding.get("request"), dict)
    ):
        raise RuntimeError("durable prepared binding identity mismatch")
    validate_request(binding["request"], validate_live_inputs=False)
    return binding


def optional_durable_record(path: Path) -> dict[str, Any] | None:
    try:
        record, _ = read_json(path)
    except FileNotFoundError:
        return None
    if not isinstance(record, dict):
        raise RuntimeError(f"durable record must be an object: {path}")
    return record


def restore_local_state(attempt: str) -> tuple[dict[str, Any], Path, Path]:
    """Recreate ephemeral local state from the immutable OSS ledger."""

    binding = require_prepared_binding(attempt)
    request_path, state_path = attempt_paths(attempt)
    body = binding["request"]
    if request_path.exists():
        local_request, _ = read_json(request_path)
        if local_request != body:
            raise RuntimeError("local request disagrees with durable prepared binding")
    else:
        atomic_write(request_path, body)
    _, request_meta = stable_read(request_path)
    latch = optional_durable_record(submission_latch_path())
    response = optional_durable_record(create_response_path())
    acknowledgement = optional_durable_record(acknowledgement_path())
    if latch is not None and (
        latch.get("experiment_id") != EXPERIMENT_ID
        or latch.get("attempt") != attempt
    ):
        raise RuntimeError("durable submission latch identity mismatch")
    if response is not None and (
        response.get("experiment_id") != EXPERIMENT_ID
        or response.get("attempt") != attempt
    ):
        raise RuntimeError("durable CreateJob response identity mismatch")
    if acknowledgement is not None and (
        acknowledgement.get("experiment_id") != EXPERIMENT_ID
        or acknowledgement.get("attempt") != attempt
    ):
        raise RuntimeError("durable acknowledgement identity mismatch")
    if acknowledgement is not None:
        phase = "ACK"
    elif latch is not None:
        phase = "AMBIGUOUS"
    else:
        phase = "PREPARED"
    created = str(binding.get("created_at") or now())
    restored_at = now()
    state = {
        "attempt": attempt,
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "created_at": created,
        "updated_at": restored_at,
        "request_path": str(request_path),
        "request_metadata": request_meta,
        "durable_prepared_path": str(prepared_binding_path()),
        "display_name": body["DisplayName"],
        "description": body["Description"],
        "submission_tag": body["Envs"]["FASTWAM_SUBMISSION_TAG"],
        "requested_gpus": REQUESTED_GPUS,
        "cloud_mutations": 1 if latch is not None else 0,
        "history": [{"at": restored_at, "from": None, "to": phase, "reason": "restored_from_durable_ledger"}],
    }
    if latch is not None:
        state["submission_latch"] = latch
        state["create_call_disposition"] = "MAY_HAVE_BEEN_SENT"
    if response is not None:
        state["job_id"] = response.get("job_id")
        state["request_id"] = response.get("request_id")
        state["create_call_disposition"] = "RESPONSE_RECEIVED"
    if acknowledgement is not None:
        state["job_id"] = acknowledgement.get("job_id")
        state["request_id"] = acknowledgement.get("request_id")
        state["create_call_disposition"] = "ACKNOWLEDGED"
    atomic_write(state_path, state)
    return state, request_path, state_path


def load_state(attempt: str) -> tuple[dict[str, Any], Path, Path]:
    request_path, state_path = attempt_paths(attempt)
    if not state_path.exists():
        return restore_local_state(attempt)
    state, _ = read_json(state_path)
    if state.get("attempt") != attempt or state.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("state identity mismatch")
    binding = require_prepared_binding(attempt)
    request, _ = read_json(request_path, state.get("request_metadata"))
    if request != binding["request"]:
        raise RuntimeError("local request disagrees with durable prepared binding")
    return state, request_path, state_path


def transition(
    state_path: Path,
    state: dict[str, Any],
    allowed: set[str],
    target: str,
    **updates: Any,
) -> dict[str, Any]:
    current = str(state.get("phase"))
    if current not in allowed:
        raise RuntimeError(f"refusing transition {current} -> {target}")
    result = dict(state)
    result.update(updates)
    result["phase"] = target
    result["updated_at"] = now()
    history = list(result.get("history") or [])
    history.append({"at": result["updated_at"], "from": current, "to": target})
    result["history"] = history
    atomic_write(state_path, result)
    return result


def source_snapshot_literal(value: str) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute():
        raise ValueError("source root must be absolute")
    if supplied.parent != SOURCE_PREFIX:
        raise ValueError(
            f"source root must be one unique direct child of {SOURCE_PREFIX}"
        )
    if SOURCE_SNAPSHOT_NAME_RE.fullmatch(supplied.name) is None:
        raise ValueError("source snapshot name is outside the portable unique-name contract")
    return supplied


def assert_source_root(value: str) -> Path:
    supplied = source_snapshot_literal(value)
    if supplied.is_symlink():
        raise ValueError("source root must be a non-symlink directory")
    resolved = supplied.resolve(strict=True)
    if not resolved.is_dir() or str(resolved) != value:
        raise ValueError("source root must be an exact canonical directory")
    if APPROVED_SOURCE_ROOT is not None and resolved != APPROVED_SOURCE_ROOT:
        raise ValueError(f"source root must exactly equal approved snapshot {APPROVED_SOURCE_ROOT}")
    required = [
        resolved / ENTRYPOINT_REL,
        resolved / REAL_PREFLIGHT_REL,
        resolved / STRUCTURED_EVIDENCE_REL,
        resolved / "scripts/train.py",
        resolved / "scripts/accelerate_configs/accelerate_zero2_ds.yaml",
        resolved / "configs/task" / f"{TASK}.yaml",
    ]
    for path in required:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source snapshot lacks regular required file: {path}")
    return resolved


def source_snapshot_metadata(source_root: Path) -> dict[str, Any]:
    """Record cross-node metadata and direct bytes for every source entry."""

    entries: list[dict[str, Any]] = []
    for path in sorted([source_root, *source_root.rglob("*")], key=lambda item: str(item)):
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode):
            raise RuntimeError(f"source snapshot contains a symlink: {path}")
        if stat.S_ISDIR(st.st_mode):
            kind = "directory"
        elif stat.S_ISREG(st.st_mode) and st.st_nlink == 1:
            kind = "file"
        else:
            raise RuntimeError(f"source snapshot contains unsupported entry: {path}")
        entry = {
            "path": "." if path == source_root else str(path.relative_to(source_root)),
            "kind": kind,
            "mode": stat.S_IMODE(st.st_mode),
            "size": st.st_size if kind == "file" else 0,
            "mtime_ns": st.st_mtime_ns,
        }
        if kind == "file":
            payload, observed = stable_read(path)
            entry.update(
                {
                    "mode": stat.S_IMODE(observed["mode"]),
                    "size": observed["size"],
                    "mtime_ns": observed["mtime_ns"],
                    "content_b64": base64.b64encode(payload).decode("ascii"),
                }
            )
        entries.append(entry)
    return {
        "schema": "fastwam-nohash-source-content-binding-v2",
        "approved_source_root": str(source_root),
        "entries": entries,
    }


def assert_stats_source(value: str) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ValueError("stats source must be an absolute, non-symlink regular file")
    resolved = supplied.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(STATS_SOURCE_PREFIX):
        raise ValueError(f"stats source must be a regular file beneath {STATS_SOURCE_PREFIX}")
    if resolved.suffix.lower() != ".json":
        raise ValueError("stats source must be a JSON file")
    return resolved


def assert_vae_source(value: str) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ValueError("VAE source must be an absolute, non-symlink regular file")
    resolved = supplied.resolve(strict=True)
    if resolved != VAE_SOURCE or str(resolved) != value:
        raise ValueError(f"VAE source must exactly equal {VAE_SOURCE}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size != VAE_SOURCE_BYTES
        or stable_fields(before) != stable_fields(after)
    ):
        raise ValueError("VAE source is not the expected stable regular file")
    return resolved


def assert_pinned_python() -> Path:
    """Bind the logical venv entry point to one regular CPFS interpreter."""

    try:
        resolved = PINNED_PYTHON.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"pinned CPFS Python cannot be resolved: {PINNED_PYTHON}") from error
    if (
        not PINNED_PYTHON.is_symlink()
        or resolved != PINNED_PYTHON_TARGET
        or PINNED_PYTHON_TARGET.is_symlink()
        or not PINNED_PYTHON_TARGET.is_file()
        or not os.access(PINNED_PYTHON_TARGET, os.X_OK)
    ):
        raise RuntimeError(
            "pinned CPFS Python logical path or resolved executable target mismatch: "
            f"{PINNED_PYTHON} -> {resolved}"
        )
    return resolved


def assert_gaussian_cache_root(value: str, *, expected_kind: str) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ValueError("Gaussian cache root must be an absolute, non-symlink directory")
    resolved = supplied.resolve(strict=True)
    if (
        not resolved.is_dir()
        or not resolved.is_relative_to(GAUSSIAN_CACHE_PREFIX)
        or str(resolved) != value
    ):
        raise ValueError(
            f"Gaussian cache root must be canonical and beneath {GAUSSIAN_CACHE_PREFIX}"
        )
    manifest_path = resolved / "manifest.json"
    complete_path = resolved / "COMPLETE"
    manifest, _ = read_json(manifest_path)
    stable_read(complete_path)
    schema = manifest.get("schema") or {}
    if schema.get("cache_kind") != expected_kind:
        raise ValueError(
            f"Gaussian cache kind mismatch: expected={expected_kind!r} "
            f"actual={schema.get('cache_kind')!r} root={resolved}"
        )
    if int(schema.get("channel_count") or 0) != 13:
        raise ValueError("Gaussian cache must declare exactly 13 channels")
    height = int(schema.get("height") or 0)
    width = int(schema.get("width") or 0)
    if expected_kind == "compact" and (height, width) != (28, 40):
        raise ValueError("compact Gaussian cache must declare spatial size 28x40")
    if expected_kind == "canonical" and (height < 28 or width < 40):
        raise ValueError("canonical Gaussian cache is too small for 28x40 projection")
    selection = manifest.get("selection") or {}
    if expected_kind == "compact" and selection.get("mode") != "index":
        raise ValueError("compact primary must declare selection.mode='index'")
    if expected_kind == "canonical":
        if selection.get("mode") != "all":
            raise ValueError("canonical fallback must declare selection.mode='all'")
    return resolved


def build_request(
    source_root: Path,
    stats_source: Path,
    gaussian_cache: Path,
    gaussian_fallback_cache: Path,
    attempt: str,
    *,
    trusted_runtime_bytes: bytes | None = None,
) -> dict[str, Any]:
    short = attempt.replace("-", "")
    tag = f"{SUBMISSION_TAG_PREFIX}-{short}"
    display = f"{DISPLAY_NAME_PREFIX}-{short[:20]}"
    output = OUTPUT_PREFIX / tag
    entrypoint = source_root / ENTRYPOINT_REL
    if trusted_runtime_bytes is None:
        trusted_runtime_bytes, _ = stable_read(entrypoint)
    description = f"{EXPERIMENT_ID}; submission_tag={tag}; exactly one worker and eight GPUs"
    envs = {
        "FASTWAM_EXPERIMENT_ID": EXPERIMENT_ID,
        "FASTWAM_SUBMISSION_TAG": tag,
        "FASTWAM_SOURCE_ROOT": str(source_root),
        "FASTWAM_PREPARED_BINDING_PATH": str(prepared_binding_path()),
        "FASTWAM_GATE2_ENTRYPOINT": str(entrypoint),
        TRUSTED_RUNTIME_B64_ENV: base64.b64encode(trusted_runtime_bytes).decode("ascii"),
        TRUSTED_RUNTIME_BYTES_ENV: str(len(trusted_runtime_bytes)),
        "FASTWAM_OSS_OUTPUT_ROOT": str(output),
        "FASTWAM_TASK_CONFIG": TASK,
        "FASTWAM_ARTIFACT_INTEGRITY_MODE": "metadata_no_hash",
        "FASTWAM_PYTHON": str(PINNED_PYTHON),
        "FASTWAM_PYTHON_TARGET": str(PINNED_PYTHON_TARGET),
        "FASTWAM_DATASET_ROOT": "/cpfs/user/chengjuntao/datasets/robofactory_multi_robot",
        "FASTWAM_INITIAL_CHECKPOINT": (
            "/oss-chengjuntao/artifacts/fastwam-n234-vg1hub1gau1-s42-5000-"
            "r2a2-beg0t5rle97qepyw8u-a57915104bff-20260802t1820z/"
            "checkpoints/weights/step_005000.pt"
        ),
        "FASTWAM_N234_NOHASH_STATS_SOURCE": str(stats_source),
        "FASTWAM_TEXT_CACHE_DIR": (
            "/cpfs/user/chengjuntao/datasets/robofactory_multi_robot/"
            "text_embeds_cache_n234"
        ),
        "FASTWAM_GAUSSIAN_CACHE_DIR": str(gaussian_cache),
        "FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR": str(gaussian_fallback_cache),
        "FASTWAM_MIN_TMP_FREE_BYTES": "214748364800",
        "NPROC_PER_NODE": "8",
    }
    return {
        "Accessibility": "PRIVATE",
        "CustomEnvs": [],
        "DataSources": [
            {"DataSourceId": CPFS_SOURCE, "MountAccess": "RO", "MountPath": "/cpfs/user/chengjuntao"},
            {"DataSourceId": OSS_SOURCE, "MountAccess": "RW", "MountPath": "/oss-chengjuntao"},
        ],
        "Description": description,
        "DisplayName": display,
        "Envs": envs,
        "JobMaxRunningTimeMinutes": 720,
        "JobSpecs": [
            {
                "ElasticSpotSpecs": [],
                "Image": IMAGE,
                "LocalMountSpecs": [],
                "PodCount": 1,
                "ResourceConfig": {"CPU": "126", "GPU": "8", "Memory": "960Gi", "SharedMemory": "960Gi"},
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
                "experiment_id": EXPERIMENT_ID,
                "project": "fastwam-multirobot",
                "purpose": "paid-gate2-nohash",
                "submission_tag": tag,
            },
        },
        "SuccessPolicy": "AllWorkers",
        "UserCommand": TRUSTED_BOOTSTRAP_COMMAND,
        "WorkspaceId": WORKSPACE_ID,
    }


def validate_request(
    body: dict[str, Any],
    sdk_models: Any | None = None,
    *,
    validate_live_inputs: bool = True,
) -> Any:
    if body.get("WorkspaceId") != WORKSPACE_ID or body.get("ResourceId") != RESOURCE_ID:
        raise RuntimeError("workspace/resource mismatch")
    if (
        body.get("Accessibility") != "PRIVATE"
        or body.get("CustomEnvs") != []
        or body.get("JobType") != "PyTorchJob"
        or body.get("Priority") != 1
        or body.get("JobMaxRunningTimeMinutes") != 720
        or body.get("SuccessPolicy") != "AllWorkers"
    ):
        raise RuntimeError("job-level execution contract mismatch")
    specs = body.get("JobSpecs") or []
    if len(specs) != 1:
        raise RuntimeError("request must have exactly one job spec")
    spec = specs[0]
    if spec.get("Type") != "Worker" or spec.get("PodCount") != 1:
        raise RuntimeError("request must have exactly one worker pod")
    if spec.get("ResourceConfig", {}).get("GPU") != "8" or spec.get("RestartPolicy") != "Never":
        raise RuntimeError("request must allocate exactly eight GPUs with RestartPolicy Never")
    mounts = [(x.get("DataSourceId"), x.get("MountPath"), x.get("MountAccess")) for x in body.get("DataSources") or []]
    if mounts != [
        (CPFS_SOURCE, "/cpfs/user/chengjuntao", "RO"),
        (OSS_SOURCE, "/oss-chengjuntao", "RW"),
    ]:
        raise RuntimeError("mount contract mismatch")
    envs = body.get("Envs") or {}
    tag = str(envs.get("FASTWAM_SUBMISSION_TAG") or "")
    exact_env = {
        "FASTWAM_EXPERIMENT_ID": EXPERIMENT_ID,
        "FASTWAM_PREPARED_BINDING_PATH": str(prepared_binding_path()),
        "FASTWAM_TASK_CONFIG": TASK,
        "FASTWAM_ARTIFACT_INTEGRITY_MODE": "metadata_no_hash",
        "FASTWAM_DATASET_ROOT": "/cpfs/user/chengjuntao/datasets/robofactory_multi_robot",
        "FASTWAM_PYTHON": str(PINNED_PYTHON),
        "FASTWAM_PYTHON_TARGET": str(PINNED_PYTHON_TARGET),
        "NPROC_PER_NODE": "8",
    }
    for name, expected in exact_env.items():
        if envs.get(name) != expected:
            raise RuntimeError(f"frozen environment mismatch for {name}")
    trusted_runtime_literal = envs.get(TRUSTED_RUNTIME_B64_ENV)
    trusted_runtime_size_literal = str(envs.get(TRUSTED_RUNTIME_BYTES_ENV) or "")
    if not isinstance(trusted_runtime_literal, str) or not trusted_runtime_literal:
        raise RuntimeError("trusted runtime payload is missing")
    if not trusted_runtime_size_literal.isdecimal():
        raise RuntimeError("trusted runtime byte count is invalid")
    try:
        trusted_runtime_bytes = base64.b64decode(
            trusted_runtime_literal.encode("ascii"), validate=True
        )
    except (UnicodeEncodeError, ValueError) as error:
        raise RuntimeError("trusted runtime payload is not canonical base64") from error
    if len(trusted_runtime_bytes) != int(trusted_runtime_size_literal):
        raise RuntimeError("trusted runtime byte count mismatch")
    source_literal = str(envs.get("FASTWAM_SOURCE_ROOT") or "")
    entrypoint_literal = str(envs.get("FASTWAM_GATE2_ENTRYPOINT") or "")
    stats_literal = str(envs.get("FASTWAM_N234_NOHASH_STATS_SOURCE") or "")
    primary_literal = str(envs.get("FASTWAM_GAUSSIAN_CACHE_DIR") or "")
    fallback_literal = str(envs.get("FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR") or "")
    try:
        source_snapshot_literal(source_literal)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if APPROVED_SOURCE_ROOT is not None and source_literal != str(APPROVED_SOURCE_ROOT):
        raise RuntimeError("request source root is not the exact approved snapshot")
    if entrypoint_literal != str(Path(source_literal) / ENTRYPOINT_REL):
        raise RuntimeError("Gate2 entrypoint is not bound to the source snapshot")
    if not stats_literal.startswith(str(STATS_SOURCE_PREFIX) + "/"):
        raise RuntimeError("normalization stats are outside the approved source prefix")
    if not primary_literal.startswith(str(GAUSSIAN_CACHE_PREFIX) + "/"):
        raise RuntimeError("primary Gaussian cache is outside the approved prefix")
    if not fallback_literal.startswith(str(GAUSSIAN_CACHE_PREFIX) + "/"):
        raise RuntimeError("fallback Gaussian cache is outside the approved prefix")
    if primary_literal == fallback_literal:
        raise RuntimeError("Gaussian primary and fallback cache roots must differ")
    expected_output = str(OUTPUT_PREFIX / tag)
    if not tag or envs.get("FASTWAM_OSS_OUTPUT_ROOT") != expected_output:
        raise RuntimeError("submission tag and output identity mismatch")
    if validate_live_inputs:
        assert_pinned_python()
        source_root = assert_source_root(str(envs.get("FASTWAM_SOURCE_ROOT") or ""))
        assert_vae_source(str(VAE_SOURCE))
        stats_source = assert_stats_source(
            str(envs.get("FASTWAM_N234_NOHASH_STATS_SOURCE") or "")
        )
        primary = assert_gaussian_cache_root(
            str(envs.get("FASTWAM_GAUSSIAN_CACHE_DIR") or ""),
            expected_kind="compact",
        )
        fallback = assert_gaussian_cache_root(
            str(envs.get("FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR") or ""),
            expected_kind="canonical",
        )
        if primary == fallback:
            raise RuntimeError("Gaussian primary and fallback cache roots must differ")
        canonical_inputs = {
            "FASTWAM_SOURCE_ROOT": source_root,
            "FASTWAM_N234_NOHASH_STATS_SOURCE": stats_source,
            "FASTWAM_GAUSSIAN_CACHE_DIR": primary,
            "FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR": fallback,
        }
        for name, resolved in canonical_inputs.items():
            if str(resolved) != envs.get(name):
                raise RuntimeError(f"{name} path must be canonical")
        current_runtime_bytes, _ = stable_read(source_root / ENTRYPOINT_REL)
        if current_runtime_bytes != trusted_runtime_bytes:
            raise RuntimeError("trusted request runtime differs from approved source runtime")
    if tag not in str(body.get("Description")):
        raise RuntimeError("description lacks unique submission tag")
    if (body.get("Settings") or {}).get("Tags", {}).get("submission_tag") != tag:
        raise RuntimeError("settings tag mismatch")
    if body.get("UserCommand") != TRUSTED_BOOTSTRAP_COMMAND:
        raise RuntimeError("trusted bootstrap command mismatch")
    if sdk_models is None:
        return None
    request = sdk_models.CreateJobRequest().from_map(body)
    request.validate()
    if request.to_map() != body:
        raise RuntimeError("SDK request roundtrip mismatch")
    return request


def prepare(
    source_root: str,
    stats_source: str,
    gaussian_cache: str,
    gaussian_fallback_cache: str,
) -> None:
    root = experiment_root()
    if prepared_binding_path().exists():
        raise RuntimeError(
            "experiment already has an immutable durable prepared binding"
        )
    if submission_latch_path().exists():
        raise RuntimeError("experiment already has an irreversible submission latch")
    if root.exists() and any(root.glob("*/state.json")):
        raise RuntimeError(
            "experiment already has an attempt state; use that attempt or a new experiment ID"
        )
    source = assert_source_root(source_root)
    approved_source_metadata = source_snapshot_metadata(source)
    stats = assert_stats_source(stats_source)
    primary = assert_gaussian_cache_root(gaussian_cache, expected_kind="compact")
    fallback = assert_gaussian_cache_root(
        gaussian_fallback_cache, expected_kind="canonical"
    )
    if primary == fallback:
        raise RuntimeError("Gaussian primary and fallback cache roots must differ")
    attempt = str(uuid.uuid4())
    request_path, state_path = attempt_paths(attempt)
    if request_path.parent.exists():
        raise RuntimeError("attempt directory unexpectedly exists")
    body = build_request(source, stats, primary, fallback, attempt)
    validate_request(body)
    created = now()
    durable_binding = {
        "schema": "fastwam-dlc-prepared-binding-v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt": attempt,
        "created_at": created,
        "request": body,
        "approved_source_root": str(source),
        "approved_source_metadata": approved_source_metadata,
        "semantics": "immutable_request_binding_before_any_CreateJob_call",
    }
    durable_exclusive_write(prepared_binding_path(), durable_binding)
    atomic_write(request_path, body)
    _, request_meta = stable_read(request_path)
    state = {
        "attempt": attempt,
        "experiment_id": EXPERIMENT_ID,
        "phase": "PREPARED",
        "created_at": created,
        "updated_at": created,
        "request_path": str(request_path),
        "request_metadata": request_meta,
        "durable_prepared_path": str(prepared_binding_path()),
        "display_name": body["DisplayName"],
        "description": body["Description"],
        "submission_tag": body["Envs"]["FASTWAM_SUBMISSION_TAG"],
        "requested_gpus": REQUESTED_GPUS,
        "cloud_mutations": 0,
        "history": [{"at": created, "from": None, "to": "PREPARED"}],
    }
    atomic_write(state_path, state)
    print(json.dumps({"attempt": attempt, "phase": "PREPARED", "state_path": str(state_path)}, indent=2))


def load_sdk() -> tuple[Any, Any, Any]:
    from alibabacloud_credentials.client import Client as CredentialClient
    from alibabacloud_credentials.models import Config as CredentialConfig
    from alibabacloud_pai_dlc20201203 import models
    from alibabacloud_pai_dlc20201203.client import Client
    from alibabacloud_tea_openapi.models import Config
    from alibabacloud_tea_util.models import RuntimeOptions

    profile, _ = read_json(PROFILE)
    current = profile.get("current")
    selected = next((item for item in profile.get("profiles", []) if item.get("name") == current), None)
    if not selected or selected.get("mode") != "CredentialsURI" or not selected.get("credentials_uri"):
        raise RuntimeError("active Alibaba Cloud profile must use CredentialsURI")
    credential = CredentialClient(
        CredentialConfig(type="credentials_uri", credentials_uri=selected["credentials_uri"])
    )
    client = Client(Config(credential=credential, region_id=REGION, endpoint="pai-dlc.cn-beijing.aliyuncs.com"))
    return client, models, RuntimeOptions


def runtime_options(runtime_cls: Any) -> Any:
    return runtime_cls(autoretry=False, max_attempts=1, connect_timeout=10000, read_timeout=30000)


def list_jobs(client: Any, models: Any, runtime_cls: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    total: int | None = None
    page = 1
    while total is None or len(result) < total:
        request = models.ListJobsRequest(
            workspace_id=WORKSPACE_ID,
            page_number=page,
            page_size=100,
            order="desc",
            sort_by="GmtCreateTime",
        )
        response = client.list_jobs_with_options(request, {}, runtime_options(runtime_cls))
        body = response.body.to_map()
        page_jobs = body.get("Jobs") or []
        if total is None:
            total = int(body.get("TotalCount") or 0)
        result.extend(page_jobs)
        if not page_jobs:
            break
        page += 1
    if total is None or len(result) != total:
        raise RuntimeError(f"ListJobs pagination mismatch: {len(result)} != {total}")
    ids = [str(item.get("JobId") or "") for item in result]
    if "" in ids or len(ids) != len(set(ids)):
        raise RuntimeError("ListJobs returned missing or duplicate job IDs")
    return result


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


def gpu_count(job: dict[str, Any]) -> int:
    specs = job.get("JobSpecs")
    if not isinstance(specs, list) or not specs:
        raise RuntimeError(f"active job lacks JobSpecs: {job.get('JobId')}")
    total = 0
    for spec in specs:
        pods = int(spec.get("PodCount") or 0)
        gpus = int((spec.get("ResourceConfig") or {}).get("GPU") or 0)
        if pods < 1 or gpus < 0:
            raise RuntimeError(f"invalid active job topology: {job.get('JobId')}")
        total += pods * gpus
    return total


def requested_identity_subset(observed: Any, requested: Any) -> bool:
    """Match every requested field while permitting server-added response fields."""

    if isinstance(requested, dict):
        return isinstance(observed, dict) and all(
            key in observed and requested_identity_subset(observed[key], value)
            for key, value in requested.items()
        )
    if isinstance(requested, list):
        return isinstance(observed, list) and len(observed) == len(requested) and all(
            requested_identity_subset(actual, wanted)
            for actual, wanted in zip(observed, requested)
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


def exact_identity(job: dict[str, Any], request: dict[str, Any]) -> bool:
    """Match one frozen request under the closed, observed PAI GetJob projection."""

    if not isinstance(job, dict) or not isinstance(request, dict):
        return False
    if (
        not requested_identity_subset(job.get("WorkspaceId"), request.get("WorkspaceId"))
        or not requested_identity_subset(job.get("ResourceId"), request.get("ResourceId"))
        or not custom_env_projection_matches(job, request)
        or not datasource_projection_matches(job, request)
    ):
        return False

    # The service omitted exactly these two frozen request fields in all three
    # observed Gate2 responses.  If it ever returns either field, its value must
    # still match; no default is synthesized into the observed response.
    omitted_by_service = {"JobMaxRunningTimeMinutes", "SuccessPolicy"}
    if not omitted_by_service.issubset(request):
        return False
    for key in omitted_by_service:
        if key in job and not requested_identity_subset(job[key], request[key]):
            return False

    special = {
        "WorkspaceId", "ResourceId", "CustomEnvs", "DataSources", *omitted_by_service
    }
    return all(
        key in job and requested_identity_subset(job[key], value)
        for key, value in request.items()
        if key not in special
    )


def belongs_to_experiment(job: dict[str, Any]) -> bool:
    envs = job.get("Envs") or {}
    settings_tags = (job.get("Settings") or {}).get("Tags") or {}
    return (
        envs.get("FASTWAM_EXPERIMENT_ID") == EXPERIMENT_ID
        or settings_tags.get("experiment_id") == EXPERIMENT_ID
        or EXPERIMENT_ID in str(job.get("Description") or "")
    )


def snapshot(client: Any, models: Any, runtime_cls: Any, request: dict[str, Any], number: int) -> dict[str, Any]:
    listed = list_jobs(client, models, runtime_cls)
    active: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for summary in listed:
        detail = get_job(client, models, runtime_cls, str(summary["JobId"]))
        status_value = str(detail.get("Status") or "")
        if status_value not in TERMINAL:
            active.append(
                {
                    "job_id": str(detail["JobId"]),
                    "display_name": str(detail.get("DisplayName") or ""),
                    "status": status_value or "UNKNOWN_ACTIVE",
                    "gpus": gpu_count(detail),
                }
            )
        if belongs_to_experiment(detail):
            candidates.append({"job_id": str(detail["JobId"]), "status": status_value})
    active.sort(key=lambda item: item["job_id"])
    total_gpus = sum(item["gpus"] for item in active)
    if candidates:
        raise RuntimeError(f"an exact Gate2 candidate already exists: {candidates}")
    return {
        "number": number,
        "observed_at": now(),
        "workspace_id": WORKSPACE_ID,
        "listed_jobs": len(listed),
        "active_jobs": active,
        "active_gpu_count": total_gpus,
        "requested_gpu_count": REQUESTED_GPUS,
        "post_submit_gpu_count": total_gpus + REQUESTED_GPUS,
        "resource_policy": "exactly_8_gpus_per_job_no_artificial_workspace_ceiling",
    }


def execute(attempt: str) -> None:
    state, request_path, state_path = load_state(attempt)
    if state.get("phase") != "PREPARED":
        raise RuntimeError("execute is one-shot and accepts PREPARED only; reconcile ambiguity")
    if submission_latch_path().exists():
        raise RuntimeError("submission latch already exists; reconcile and never call CreateJob again")
    body, _ = read_json(request_path, state.get("request_metadata"))
    client, models, runtime_cls = load_sdk()
    request = validate_request(body, models)
    first = snapshot(client, models, runtime_cls, body, 1)
    time.sleep(2)
    second = snapshot(client, models, runtime_cls, body, 2)
    first_shape = [(x["job_id"], x["status"], x["gpus"]) for x in first["active_jobs"]]
    second_shape = [(x["job_id"], x["status"], x["gpus"]) for x in second["active_jobs"]]
    if first_shape != second_shape or first["active_gpu_count"] != second["active_gpu_count"]:
        raise RuntimeError("active workspace allocation changed between the two ListJobs snapshots")
    binding = require_prepared_binding(attempt)
    source = assert_source_root(body["Envs"]["FASTWAM_SOURCE_ROOT"])
    if binding.get("approved_source_root") != str(source):
        raise RuntimeError("prepared approved source-root identity mismatch")
    if binding.get("approved_source_metadata") != source_snapshot_metadata(source):
        raise RuntimeError("approved no-hash source metadata changed since prepare")
    latch = acquire_submission_latch(attempt)
    state = transition(
        state_path,
        state,
        {"PREPARED"},
        "SUBMITTING",
        preflight_snapshots=[first, second],
        submission_latch=latch,
        create_call_disposition="MAY_HAVE_BEEN_SENT",
        submitting_semantics="MAY_HAVE_BEEN_SENT until exact ACK or reconciliation",
    )
    try:
        # The only mutating SDK call in this file.  It is deliberately not wrapped in a retry.
        response = client.create_job_with_options(request, {}, runtime_options(runtime_cls))
        response_body = response.body.to_map()
        job_id = str(response_body.get("JobId") or "")
        request_id = str(response_body.get("RequestId") or "")
        durable_exclusive_write(
            create_response_path(),
            {
                "schema": "fastwam-dlc-create-response-v1",
                "experiment_id": EXPERIMENT_ID,
                "attempt": attempt,
                "observed_at": now(),
                "job_id": job_id or None,
                "request_id": request_id or None,
                "create_call_disposition": "RESPONSE_RECEIVED",
            },
        )
        state = transition(
            state_path,
            state,
            {"SUBMITTING"},
            "SUBMITTING",
            cloud_mutations=1,
            job_id=job_id or None,
            request_id=request_id or None,
            create_call_disposition="RESPONSE_RECEIVED",
            create_response_observed_at=now(),
        )
        if not job_id or not request_id:
            raise RuntimeError("CreateJob response lacks job/request identity")
        observed = get_job(client, models, runtime_cls, job_id)
        if not exact_identity(observed, body):
            raise RuntimeError("GetJob ACK identity mismatch")
        state = transition(
            state_path,
            state,
            {"SUBMITTING"},
            "SENT",
            cloud_mutations=1,
            job_id=job_id,
            request_id=request_id,
            create_call_disposition="RESPONSE_RECEIVED",
            create_job_returned_at=now(),
            observed_status=str(observed.get("Status") or ""),
        )
    except BaseException as error:
        try:
            latest, _, _ = load_state(attempt)
            transition(
                state_path,
                latest,
                {"SUBMITTING", "SENT"},
                "AMBIGUOUS",
                ambiguity_type=type(error).__name__,
                ambiguity_message=str(error),
                cloud_mutations=1,
            )
        finally:
            raise
    acknowledgement = {
        "schema": "fastwam-dlc-acknowledgement-v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt": attempt,
        "acknowledged_at": now(),
        "job_id": job_id,
        "request_id": request_id,
        "observed_status": str(observed.get("Status") or ""),
        "identity_check": "exact_request_identity_passed",
        "source": "execute_GetJob",
    }
    durable_exclusive_write(acknowledgement_path(), acknowledgement)
    acknowledged = transition(
        state_path,
        state,
        {"SENT"},
        "ACK",
        cloud_mutations=1,
        job_id=job_id,
        request_id=request_id,
        create_call_disposition="ACKNOWLEDGED",
        acknowledged_at=acknowledgement["acknowledged_at"],
        durable_acknowledgement_path=str(acknowledgement_path()),
        observed_status=str(observed.get("Status") or ""),
    )
    print(json.dumps(acknowledged, indent=2, sort_keys=True))


def reconcile(attempt: str) -> None:
    state, request_path, state_path = load_state(attempt)
    if state.get("phase") not in {"PREPARED", "SUBMITTING", "SENT", "AMBIGUOUS"}:
        raise RuntimeError("reconcile accepts only latched PREPARED, SUBMITTING, SENT, or AMBIGUOUS")
    require_submission_latch(attempt)
    if state.get("phase") == "PREPARED":
        state = transition(
            state_path,
            state,
            {"PREPARED"},
            "AMBIGUOUS",
            ambiguity_type="LATCHED_BEFORE_STATE_TRANSITION",
            ambiguity_message="CreateJob disposition is unknown; automatic resubmission is forbidden",
        )
    body, _ = read_json(request_path, state.get("request_metadata"))
    client, models, runtime_cls = load_sdk()
    validate_request(body, models, validate_live_inputs=False)
    candidates: list[dict[str, Any]] = []
    durable_response = optional_durable_record(create_response_path())
    if durable_response is not None and (
        durable_response.get("experiment_id") != EXPERIMENT_ID
        or durable_response.get("attempt") != attempt
    ):
        raise RuntimeError("durable CreateJob response identity mismatch")
    known_job_id = str(
        state.get("job_id")
        or (durable_response or {}).get("job_id")
        or ""
    )
    if known_job_id:
        detail = get_job(client, models, runtime_cls, known_job_id)
        if not exact_identity(detail, body):
            raise RuntimeError("persisted job_id does not match the frozen request identity")
        candidates.append(detail)
    else:
        for item in list_jobs(client, models, runtime_cls):
            detail = get_job(client, models, runtime_cls, str(item["JobId"]))
            if exact_identity(detail, body):
                candidates.append(detail)
    if len(candidates) == 1:
        job = candidates[0]
        job_id = str(job["JobId"])
        existing_ack = optional_durable_record(acknowledgement_path())
        if existing_ack is None:
            reconciliation_ack = {
                "schema": "fastwam-dlc-acknowledgement-v1",
                "experiment_id": EXPERIMENT_ID,
                "attempt": attempt,
                "acknowledged_at": now(),
                "job_id": job_id,
                "request_id": (durable_response or {}).get("request_id"),
                "observed_status": str(job.get("Status") or ""),
                "identity_check": "exact_request_identity_passed",
                "source": "reconcile_GetJob_or_ListJobs",
            }
            durable_exclusive_write(acknowledgement_path(), reconciliation_ack)
        elif (
            existing_ack.get("experiment_id") != EXPERIMENT_ID
            or existing_ack.get("attempt") != attempt
            or str(existing_ack.get("job_id") or "") != job_id
            or existing_ack.get("identity_check") != "exact_request_identity_passed"
        ):
            raise RuntimeError("durable acknowledgement identity mismatch")
        result = transition(
            state_path,
            state,
            {"SUBMITTING", "SENT", "AMBIGUOUS"},
            "RECONCILED",
            cloud_mutations=1,
            job_id=job_id,
            observed_status=str(job.get("Status") or ""),
            reconciled_at=now(),
            durable_acknowledgement_path=str(acknowledgement_path()),
        )
    else:
        result = transition(
            state_path,
            state,
            {"SUBMITTING", "SENT", "AMBIGUOUS"},
            "AMBIGUOUS",
            cloud_mutations=1,
            reconciliation_candidate_count=len(candidates),
            reconciliation_observed_at=now(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def show(attempt: str) -> None:
    state, _, _ = load_state(attempt)
    print(json.dumps(state, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("prepare", help="write PREPARED request/state; no cloud call")
    make.add_argument("--source-root", required=True)
    make.add_argument("--stats-source", required=True)
    make.add_argument("--gaussian-cache", required=True)
    make.add_argument("--gaussian-fallback-cache", required=True)
    for name in ("execute", "reconcile", "show"):
        command = sub.add_parser(name)
        command.add_argument("--attempt", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_single_writer()
    if args.command == "prepare":
        prepare(
            args.source_root,
            args.stats_source,
            args.gaussian_cache,
            args.gaussian_fallback_cache,
        )
    elif args.command == "execute":
        execute(args.attempt)
    elif args.command == "reconcile":
        reconcile(args.attempt)
    else:
        show(args.attempt)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "automatic_resubmit": False,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
