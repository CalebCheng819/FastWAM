#!/usr/bin/env python3
"""Fail-closed preparation, exactly-once submission, and terminal validation."""

from __future__ import annotations

import argparse
import base64
import copy
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R9-20260814"
RUN_ID = "fastwam-gau0-placefood-same8-r9-20260814"
DISPLAY_NAME = "fw-gau0-placefood-same8-r9"
WORKSPACE_ID = "270969"
RESOURCE_ID = "quotaksvqq2oh2pg"
SOURCE_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/fastwam-gau0-placefood-same8-eval-20260814-r18")
OUTPUT_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260814-r9")
DURABLE_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-gau0-placefood-same8-eval-20260814-r9-controller")
RESERVATION_PATH = DURABLE_ROOT / "prepared-reservation.json"
LATCH_PATH = DURABLE_ROOT / "submission-latch.json"
ACK_PATH = DURABLE_ROOT / "job-acknowledgement.json"
LOCAL_ROOT = Path("/run/fastwam-dlc-submit-state/workspace-270969/gau0-placefood-same8-r9")
STATE_PATH = LOCAL_ROOT / "state.json"
EXPERIMENT_REL = Path(".research-workflow/experiments") / EXPERIMENT_ID

CHECKPOINT = Path("/oss-chengjuntao/artifacts/fastwam-checkpoint-archives-v1/FASTWAM-MR-N234-VG1H1-S42-20260801/dlc1hqocuisxxdkb/step_005000/checkpoints/weights/step_005000.pt")
CHECKPOINT_BYTES = 12045923769
PANEL = Path("/cpfs/user/chengjuntao/fastwam_eval_runtime/panels/robofactory_n234_s42_val8_v1.json")
PANEL_BYTES = 44584
GAU1_STATS = Path("/cpfs/user/chengjuntao/fastwam_eval_runtime/inputs/dataset_stats.step5000.92dfdeec.json")
GAU1_STATS_BYTES = 2828
GAU0_STATS = Path("/oss-chengjuntao/cpfs-user-chengjuntao/datasets/robofactory_multi_robot/fastwam_multi_robot_n234_stats.json")
GAU0_STATS_BYTES = 3226
DATASET_ROOT = Path("/oss-chengjuntao/cpfs-user-chengjuntao/datasets/robofactory_multi_robot")
ROBOFACTORY_ROOT = Path("/cpfs/user/chengjuntao/fastwam_eval_runtime/RoboFactory-challenge-2d34fb3")
CONTEXT_CACHE_DIR = DATASET_ROOT / "text_embeds_cache_n234"
CONTEXT_FILE = CONTEXT_CACHE_DIR / "89bc0bd3ed4a9f6192e149614112915dbd94d0b323d714e2da3c89bb68f6e26a.t5_len128.wan22ti2v5b.pt"
CONTEXT_BYTES = 1051869
MODEL_CACHE_ROOT = Path("/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/model-cache")
NVIDIA_GRAPHICS_ROOT = Path("/cpfs/user/chengjuntao/fastwam-deploy/nvidia-graphics-570.153.02")
EGL_FRONTEND = NVIDIA_GRAPHICS_ROOT / "lib" / "libEGL.so.1.1.0"
EGL_FRONTEND_BYTES = 80328
GL_FRONTEND = NVIDIA_GRAPHICS_ROOT / "lib" / "libGL.so.1.7.0"
GL_FRONTEND_BYTES = 649416
GLES1_FRONTEND = NVIDIA_GRAPHICS_ROOT / "lib" / "libGLESv1_CM.so.1.2.0"
GLES1_FRONTEND_BYTES = 43208
GLES2_FRONTEND = NVIDIA_GRAPHICS_ROOT / "lib" / "libGLESv2.so.2.1.0"
GLES2_FRONTEND_BYTES = 80064
OPENGL_FRONTEND = NVIDIA_GRAPHICS_ROOT / "lib" / "libOpenGL.so.0"
OPENGL_FRONTEND_BYTES = 198848
GLX_FRONTEND = NVIDIA_GRAPHICS_ROOT / "lib" / "libGLX.so.0"
GLX_FRONTEND_BYTES = 137616
EGL_DISPATCH = NVIDIA_GRAPHICS_ROOT / "lib" / "libGLdispatch.so.0"
EGL_DISPATCH_BYTES = 952576
EGL_VENDOR = NVIDIA_GRAPHICS_ROOT / "driver-lib" / "libEGL_nvidia.so.570.153.02"
EGL_VENDOR_BYTES = 1358016
GLVND_SHIM_TARGETS = {
    "libEGL.so.1": EGL_FRONTEND,
    "libGL.so.1": GL_FRONTEND,
    "libGLESv1_CM.so.1": GLES1_FRONTEND,
    "libGLESv2.so.2": GLES2_FRONTEND,
    "libOpenGL.so.0": OPENGL_FRONTEND,
    "libGLX.so.0": GLX_FRONTEND,
}
GLVND_SHIM_ALIASES = {
    "libEGL.so": "libEGL.so.1",
    "libGL.so": "libGL.so.1",
    "libGLESv1_CM.so": "libGLESv1_CM.so.1",
    "libGLESv2.so": "libGLESv2.so.2",
}
PYTHON = Path("/cpfs/user/chengjuntao/venvs/fastwam-gaudp-py310-20260802/bin/python")
PYTHON_TARGET = Path("/cpfs/user/chengjuntao/runtimes/uv-python/cpython-3.10-linux-x86_64-gnu/bin/python3.10")
PYTHON_VERSION = (3, 10, 20)
PYTHON_CACHE_TAG = "cpython-310"
PYTHON_SOABI = "cpython-310-x86_64-linux-gnu"
PYTHON_EXTRA_ROOT = Path("/cpfs/user/chengjuntao/venvs/fastwam-gau0-eval-r7-py310-extra-20260813")
BASELINE_ROOT = Path("/oss-chengjuntao/artifacts/fastwam-multirobot-eval-f89a7a5/PlaceFood-rf/smoke8-f89a7a5-attempt2")
TRAINING_SOURCE_COMMIT = "dd64664c0a97f1c24c3824159dd8a267120bdd5e"
TRAINING_JOB_ID = "dlc1hqocuisxxdkb"

IMAGE = "dsw-registry-vpc.cn-beijing.cr.aliyuncs.com/pai/pytorch:2.7.1-gpu-py310-cu128-ubuntu22.04-3995b779-1764350887"
CPFS_DATASOURCE_ID = "d-a5mu77ymwjio71dkmw"
OSS_DATASOURCE_ID = "d-n7rly4fll0q2z6v91h"


class ContractError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fail(message: str) -> None:
    raise ContractError(message)


def canonical(path: Path) -> Path:
    result = path.resolve(strict=True)
    if str(result) != str(path):
        fail(f"path is not canonical: {path} -> {result}")
    return result


def require_dir(path: Path) -> os.stat_result:
    canonical(path)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        fail(f"not an ordinary directory: {path}")
    return info


def require_file(path: Path, expected_bytes: int | None = None) -> os.stat_result:
    canonical(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        fail(f"not a single-link ordinary file: {path}")
    if expected_bytes is not None and info.st_size != expected_bytes:
        fail(f"file size mismatch for {path}: {info.st_size} != {expected_bytes}")
    return info


def stable_read(path: Path, expected_bytes: int | None = None) -> bytes:
    before = require_file(path, expected_bytes)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            fail(f"file descriptor identity mismatch: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(fd)
    after = require_file(path, expected_bytes)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_mode", "st_nlink")
    if (
        any(getattr(before, field) != getattr(opened, field) for field in fields)
        or any(getattr(opened, field) != getattr(after, field) for field in fields)
        or len(payload) != after.st_size
    ):
        fail(f"file changed during read: {path}")
    return payload


def load_json(path: Path) -> Any:
    try:
        return json.loads(stable_read(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON at {path}: {exc}") from exc


def write_json_exclusive(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                fail(f"short write while publishing JSON: {path}")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)


def safe_mkdir(path: Path, mode: int = 0o700) -> None:
    path.mkdir(mode=mode)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o077:
        fail(f"new directory does not satisfy private-root contract: {path}")


def validate_empty_oss_durable_root(path: Path) -> None:
    """Validate an exact empty OSS control directory without trusting its mode bits.

    The OSS FUSE mount projects newly created directories as 0777 even after a
    0700 mkdir.  Integrity therefore comes from the frozen path, an ordinary
    root-owned non-link directory, an empty/closed child set, exclusive
    O_NOFOLLOW file creation, and stable single-link file reads.  This is an
    object-integrity contract, not a confidentiality claim.
    """

    canonical(path)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != 0
        or info.st_gid != 0
        or stat.S_IMODE(info.st_mode) not in (0o700, 0o777)
    ):
        fail(f"OSS durable root contract failed: {path}")
    children = sorted(child.name for child in path.iterdir())
    if children:
        fail(f"OSS durable root must be empty before prepare: {path}: {children}")


def ensure_empty_oss_durable_root(path: Path) -> None:
    canonical(path.parent)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    validate_empty_oss_durable_root(path)


def require_exact_children(path: Path, expected: set[str]) -> None:
    require_dir(path)
    actual = {child.name for child in path.iterdir()}
    if actual != expected:
        fail(f"directory allowlist mismatch for {path}: {sorted(actual)} != {sorted(expected)}")


def file_binding(path: Path, expected_bytes: int, direct: bool) -> dict[str, Any]:
    payload = stable_read(path, expected_bytes) if direct else None
    return {
        "path": str(path),
        "bytes": expected_bytes,
        "content_b64": base64.b64encode(payload).decode("ascii") if payload is not None else None,
    }


def validate_file_binding(binding: dict[str, Any]) -> None:
    path = Path(binding["path"])
    expected = int(binding["bytes"])
    direct = binding.get("content_b64")
    if direct is None:
        require_file(path, expected)
        return
    if stable_read(path, expected) != base64.b64decode(direct, validate=True):
        fail(f"direct-byte binding changed: {path}")


def capture_tree(root: Path, *, include_contents: bool) -> dict[str, Any]:
    require_dir(root)
    entries: list[dict[str, Any]] = [{"path": ".", "kind": "dir"}]
    regular_bytes = 0
    for current, directories, filenames in os.walk(root, followlinks=False):
        directories.sort()
        filenames.sort()
        current_path = Path(current)
        relative_current = current_path.relative_to(root)
        for name in list(directories):
            path = current_path / name
            relative = (relative_current / name).as_posix()
            info = path.lstat()
            if name in {".git", "__pycache__"}:
                directories.remove(name)
                continue
            if not stat.S_ISDIR(info.st_mode):
                fail(f"unsupported source tree entry: {path}")
            entries.append({"path": relative, "kind": "dir"})
        for name in filenames:
            path = current_path / name
            relative = (relative_current / name).as_posix()
            if name.endswith((".pyc", ".pyo")):
                fail(f"bytecode is forbidden in frozen tree: {path}")
            info = require_file(path)
            payload = stable_read(path, info.st_size) if include_contents else None
            regular_bytes += info.st_size
            entry: dict[str, Any] = {
                "path": relative,
                "kind": "file",
                "bytes": info.st_size,
                "mode": stat.S_IMODE(info.st_mode),
            }
            if payload is not None:
                entry["content_b64"] = base64.b64encode(payload).decode("ascii")
            entries.append(entry)
    entries.sort(key=lambda item: (item["path"], item["kind"]))
    return {
        "schema": "fastwam-portable-direct-byte-tree-v1" if include_contents else "fastwam-portable-metadata-tree-v1",
        "root": str(root),
        "entries": entries,
        "entry_count": len(entries),
        "regular_bytes": regular_bytes,
    }


def validate_tree(binding: dict[str, Any]) -> None:
    root = Path(binding["root"])
    observed = capture_tree(root, include_contents=False)
    expected_entries = binding["entries"]
    if observed["entry_count"] != binding["entry_count"] or observed["regular_bytes"] != binding["regular_bytes"]:
        fail(f"tree inventory totals changed: {root}")
    observed_by_path = {(item["path"], item["kind"]): item for item in observed["entries"]}
    if set(observed_by_path) != {(item["path"], item["kind"]) for item in expected_entries}:
        fail(f"tree path/type inventory changed: {root}")
    for item in expected_entries:
        actual = observed_by_path[(item["path"], item["kind"])]
        if item["kind"] == "file":
            if actual["bytes"] != item["bytes"] or actual["mode"] != item["mode"]:
                fail(f"tree file metadata changed: {root / item['path']}")
            encoded = item.get("content_b64")
            if encoded is not None and stable_read(root / item["path"], item["bytes"]) != base64.b64decode(encoded, validate=True):
                fail(f"tree direct bytes changed: {root / item['path']}")


def load_aggregator(source_root: Path):
    path = source_root / EXPERIMENT_REL / "aggregate_results.py"
    require_file(path)
    spec = importlib.util.spec_from_file_location("fastwam_gau0_aggregate", path)
    if spec is None or spec.loader is None:
        fail("cannot load frozen aggregator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_python() -> None:
    info = PYTHON.lstat()
    if not stat.S_ISLNK(info.st_mode) or os.readlink(PYTHON) != str(PYTHON_TARGET):
        fail("pinned Python symlink target changed")
    resolved = PYTHON.resolve(strict=True)
    target = resolved.lstat()
    if not stat.S_ISREG(target.st_mode) or not target.st_mode & stat.S_IXUSR:
        fail("pinned Python target is not executable")
    probe = subprocess.run(
        [
            str(PYTHON),
            "-B",
            "-I",
            "-c",
            (
                "import json,sys,sysconfig;"
                "print(json.dumps({'version':list(sys.version_info[:3]),"
                "'cache_tag':sys.implementation.cache_tag,"
                "'soabi':sysconfig.get_config_var('SOABI')},sort_keys=True))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode != 0:
        fail(f"pinned Python semantic probe failed: {probe.stderr.strip()}")
    try:
        payload = json.loads(probe.stdout)
    except json.JSONDecodeError as exc:
        fail(f"pinned Python semantic probe was not JSON: {exc}")
    expected = {
        "version": list(PYTHON_VERSION),
        "cache_tag": PYTHON_CACHE_TAG,
        "soabi": PYTHON_SOABI,
    }
    if payload != expected:
        fail(f"pinned Python ABI changed: {payload!r} != {expected!r}")


def worker_pythonpath() -> str:
    return os.pathsep.join((
        str(ROBOFACTORY_ROOT),
        str(SOURCE_ROOT / "src"),
        str(PYTHON_EXTRA_ROOT),
        str(SOURCE_ROOT / "experiments" / "robofactory"),
    ))


def create_glvnd_shim(shim_dir: Path) -> None:
    require_dir(shim_dir)
    for soname, target in GLVND_SHIM_TARGETS.items():
        require_file(target)
        link = shim_dir / soname
        if os.path.lexists(link):
            fail(f"GLVND shim target already exists: {link}")
        os.symlink(target, link)
    for alias, target_name in GLVND_SHIM_ALIASES.items():
        link = shim_dir / alias
        if os.path.lexists(link):
            fail(f"GLVND shim alias already exists: {link}")
        os.symlink(target_name, link)


def worker_dependency_env(shim_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    library_paths = [
        str(shim_dir),
        str(NVIDIA_GRAPHICS_ROOT / "lib"),
        str(NVIDIA_GRAPHICS_ROOT / "driver-lib"),
    ]
    if os.environ.get("LD_LIBRARY_PATH"):
        library_paths.append(os.environ["LD_LIBRARY_PATH"])
    env.update({
        "PYTHONPATH": worker_pythonpath(),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MUJOCO_GL": "egl",
        "EGL_PLATFORM": "surfaceless",
        "PYOPENGL_PLATFORM": "egl",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "VK_ICD_FILENAMES": str(NVIDIA_GRAPHICS_ROOT / "nvidia_icd.json"),
        "VK_DRIVER_FILES": str(NVIDIA_GRAPHICS_ROOT / "nvidia_icd.json"),
        "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
        "__EGL_VENDOR_LIBRARY_FILENAMES": str(NVIDIA_GRAPHICS_ROOT / "10_nvidia.json"),
        "LD_LIBRARY_PATH": os.pathsep.join(library_paths),
    })
    return env


def validate_worker_dependencies() -> None:
    validate_python()
    require_dir(PYTHON_EXTRA_ROOT)
    program = r'''
import os
from pathlib import Path

import boto3
import git
import torch
import transformers
import diffusers
import accelerate
import deepspeed
from eval_robofactory_multi_robot import _preflight_environment_imports

source = Path(os.environ["FASTWAM_DEP_SOURCE_ROOT"])
robofactory = Path(os.environ["FASTWAM_DEP_ROBOFACTORY_ROOT"])
environment_modules = _preflight_environment_imports(robofactory)

import mani_skill
import sapien
import fastwam.runtime as runtime
import fastwam_multi_robot_policy as policy
import tasks.place_food as place_food
import utils.scenes as scenes

expected = {
    "runtime": (source / "src" / "fastwam" / "runtime.py").resolve(strict=True),
    "policy": (source / "experiments" / "robofactory" / "fastwam_multi_robot_policy.py").resolve(strict=True),
    "place_food": (robofactory / "tasks" / "place_food.py").resolve(strict=True),
    "scenes": (robofactory / "utils" / "scenes" / "__init__.py").resolve(strict=True),
}
actual = {
    "runtime": Path(runtime.__file__).resolve(strict=True),
    "policy": Path(policy.__file__).resolve(strict=True),
    "place_food": Path(place_food.__file__).resolve(strict=True),
    "scenes": Path(scenes.__file__).resolve(strict=True),
}
if actual != expected:
    raise SystemExit(f"worker module provenance mismatch: {actual} != {expected}")
if not callable(getattr(runtime, "create_multi_robot_fastwam", None)):
    raise SystemExit("frozen fastwam.runtime lacks create_multi_robot_fastwam")
print(f"GAU0_WORKER_DEPENDENCY_PREFLIGHT_PASS environment_modules={environment_modules}")
'''
    with tempfile.TemporaryDirectory(prefix="fastwam-gau0-glvnd-r9-") as temporary:
        shim_dir = Path(temporary)
        create_glvnd_shim(shim_dir)
        env = worker_dependency_env(shim_dir)
        env["FASTWAM_DEP_SOURCE_ROOT"] = str(SOURCE_ROOT)
        env["FASTWAM_DEP_ROBOFACTORY_ROOT"] = str(ROBOFACTORY_ROOT)
        result = subprocess.run(
            [str(PYTHON), "-B", "-c", program],
            cwd=SOURCE_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    if result.returncode != 0:
        details = (result.stdout + "\n" + result.stderr)[-8000:]
        fail(f"worker dependency preflight failed with exit {result.returncode}:\n{details}")


def input_bindings() -> dict[str, Any]:
    validate_python()
    for directory in (DATASET_ROOT, ROBOFACTORY_ROOT, CONTEXT_CACHE_DIR, MODEL_CACHE_ROOT, NVIDIA_GRAPHICS_ROOT, PYTHON_EXTRA_ROOT, BASELINE_ROOT):
        require_dir(directory)
    bindings = {
        "checkpoint": file_binding(CHECKPOINT, CHECKPOINT_BYTES, False),
        "panel": file_binding(PANEL, PANEL_BYTES, True),
        "gau1_stats": file_binding(GAU1_STATS, GAU1_STATS_BYTES, True),
        "gau0_native_stats": file_binding(GAU0_STATS, GAU0_STATS_BYTES, True),
        "context": file_binding(CONTEXT_FILE, CONTEXT_BYTES, True),
        "egl_frontend": file_binding(EGL_FRONTEND, EGL_FRONTEND_BYTES, False),
        "gl_frontend": file_binding(GL_FRONTEND, GL_FRONTEND_BYTES, False),
        "gles1_frontend": file_binding(GLES1_FRONTEND, GLES1_FRONTEND_BYTES, False),
        "gles2_frontend": file_binding(GLES2_FRONTEND, GLES2_FRONTEND_BYTES, False),
        "opengl_frontend": file_binding(OPENGL_FRONTEND, OPENGL_FRONTEND_BYTES, False),
        "glx_frontend": file_binding(GLX_FRONTEND, GLX_FRONTEND_BYTES, False),
        "egl_dispatch": file_binding(EGL_DISPATCH, EGL_DISPATCH_BYTES, False),
        "egl_vendor": file_binding(EGL_VENDOR, EGL_VENDOR_BYTES, False),
        "baseline": capture_tree(BASELINE_ROOT, include_contents=True),
    }
    load_aggregator(SOURCE_ROOT).validate_baseline(BASELINE_ROOT)
    return bindings


def validate_inputs(bindings: dict[str, Any]) -> None:
    validate_python()
    for directory in (DATASET_ROOT, ROBOFACTORY_ROOT, CONTEXT_CACHE_DIR, MODEL_CACHE_ROOT, NVIDIA_GRAPHICS_ROOT, PYTHON_EXTRA_ROOT, BASELINE_ROOT):
        require_dir(directory)
    for key in (
        "checkpoint",
        "panel",
        "gau1_stats",
        "gau0_native_stats",
        "context",
        "egl_frontend",
        "gl_frontend",
        "gles1_frontend",
        "gles2_frontend",
        "opengl_frontend",
        "glx_frontend",
        "egl_dispatch",
        "egl_vendor",
    ):
        validate_file_binding(bindings[key])
    validate_tree(bindings["baseline"])
    load_aggregator(SOURCE_ROOT).validate_baseline(BASELINE_ROOT)


def runtime_env(source_commit: str) -> dict[str, str]:
    return {
        "FASTWAM_SOURCE_ROOT": str(SOURCE_ROOT),
        "FASTWAM_SOURCE_COMMIT": source_commit,
        "FASTWAM_OUTPUT_ROOT": str(OUTPUT_ROOT),
        "FASTWAM_EXPERIMENT_ID": EXPERIMENT_ID,
        "FASTWAM_RUN_ID": RUN_ID,
        "FASTWAM_CHECKPOINT": str(CHECKPOINT),
        "FASTWAM_CHECKPOINT_SIZE_BYTES": str(CHECKPOINT_BYTES),
        "FASTWAM_PANEL": str(PANEL),
        "FASTWAM_PANEL_SIZE_BYTES": str(PANEL_BYTES),
        "FASTWAM_GAU1_STATS": str(GAU1_STATS),
        "FASTWAM_GAU1_STATS_SIZE_BYTES": str(GAU1_STATS_BYTES),
        "FASTWAM_GAU0_NATIVE_STATS": str(GAU0_STATS),
        "FASTWAM_GAU0_NATIVE_STATS_SIZE_BYTES": str(GAU0_STATS_BYTES),
        "FASTWAM_DATASET_ROOT": str(DATASET_ROOT),
        "FASTWAM_ROBOFACTORY_ROOT": str(ROBOFACTORY_ROOT),
        "FASTWAM_CONTEXT_CACHE_DIR": str(CONTEXT_CACHE_DIR),
        "FASTWAM_CONTEXT_SIZE_BYTES": str(CONTEXT_BYTES),
        "FASTWAM_MODEL_CACHE_ROOT": str(MODEL_CACHE_ROOT),
        "FASTWAM_NVIDIA_GRAPHICS_ROOT": str(NVIDIA_GRAPHICS_ROOT),
        "FASTWAM_EGL_FRONTEND": str(EGL_FRONTEND),
        "FASTWAM_EGL_FRONTEND_SIZE_BYTES": str(EGL_FRONTEND_BYTES),
        "FASTWAM_GL_FRONTEND": str(GL_FRONTEND),
        "FASTWAM_GL_FRONTEND_SIZE_BYTES": str(GL_FRONTEND_BYTES),
        "FASTWAM_GLES1_FRONTEND": str(GLES1_FRONTEND),
        "FASTWAM_GLES1_FRONTEND_SIZE_BYTES": str(GLES1_FRONTEND_BYTES),
        "FASTWAM_GLES2_FRONTEND": str(GLES2_FRONTEND),
        "FASTWAM_GLES2_FRONTEND_SIZE_BYTES": str(GLES2_FRONTEND_BYTES),
        "FASTWAM_OPENGL_FRONTEND": str(OPENGL_FRONTEND),
        "FASTWAM_OPENGL_FRONTEND_SIZE_BYTES": str(OPENGL_FRONTEND_BYTES),
        "FASTWAM_GLX_FRONTEND": str(GLX_FRONTEND),
        "FASTWAM_GLX_FRONTEND_SIZE_BYTES": str(GLX_FRONTEND_BYTES),
        "FASTWAM_EGL_DISPATCH": str(EGL_DISPATCH),
        "FASTWAM_EGL_DISPATCH_SIZE_BYTES": str(EGL_DISPATCH_BYTES),
        "FASTWAM_EGL_VENDOR": str(EGL_VENDOR),
        "FASTWAM_EGL_VENDOR_SIZE_BYTES": str(EGL_VENDOR_BYTES),
        "FASTWAM_PYTHON": str(PYTHON),
        "FASTWAM_PYTHON_TARGET": str(PYTHON_TARGET),
        "FASTWAM_PYTHON_VERSION": ".".join(str(item) for item in PYTHON_VERSION),
        "FASTWAM_PYTHON_CACHE_TAG": PYTHON_CACHE_TAG,
        "FASTWAM_PYTHON_SOABI": PYTHON_SOABI,
        "FASTWAM_PYTHON_EXTRA_ROOT": str(PYTHON_EXTRA_ROOT),
        "FASTWAM_BASELINE_ROOT": str(BASELINE_ROOT),
        "FASTWAM_TRAINING_SOURCE_COMMIT": TRAINING_SOURCE_COMMIT,
        "FASTWAM_TRAINING_JOB_ID": TRAINING_JOB_ID,
        "FASTWAM_RESERVATION_PATH": str(RESERVATION_PATH),
        "PYTHONUNBUFFERED": "1",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
    }


def request_body(source_commit: str) -> dict[str, Any]:
    command = f"exec /bin/bash {SOURCE_ROOT / EXPERIMENT_REL / 'runtime.sh'}"
    return {
        "Accessibility": "PRIVATE",
        "CustomEnvs": [],
        "DataSources": [
            {"DataSourceId": CPFS_DATASOURCE_ID, "MountAccess": "RO", "MountPath": "/cpfs/user/chengjuntao"},
            {"DataSourceId": OSS_DATASOURCE_ID, "MountAccess": "RW", "MountPath": "/oss-chengjuntao"},
        ],
        "Description": f"GAU0 no-Gaussian PlaceFood same-panel eval; experiment={EXPERIMENT_ID}; run={RUN_ID}; source={source_commit}",
        "DisplayName": DISPLAY_NAME,
        "Envs": runtime_env(source_commit),
        "JobMaxRunningTimeMinutes": 2160,
        "JobSpecs": [{
            "ElasticSpotSpecs": [],
            "Image": IMAGE,
            "LocalMountSpecs": [],
            "PodCount": 1,
            "ResourceConfig": {"CPU": "126", "GPU": "8", "Memory": "960Gi", "SharedMemory": "960Gi"},
            "RestartPolicy": "Never",
            "StartupDependencies": [],
            "Type": "Worker",
        }],
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
            "Tags": {"experiment_id": EXPERIMENT_ID, "run_id": RUN_ID},
        },
        "SuccessPolicy": "AllWorkers",
        "UserCommand": command,
        "WorkspaceId": WORKSPACE_ID,
    }


def load_reservation() -> dict[str, Any]:
    value = load_json(RESERVATION_PATH)
    if value.get("schema") != "fastwam-gau0-placefood-same8-reservation-v1":
        fail("reservation schema mismatch")
    if value.get("experiment_id") != EXPERIMENT_ID or value.get("run_id") != RUN_ID:
        fail("reservation identity mismatch")
    if value.get("source_root") != str(SOURCE_ROOT) or value.get("output_root") != str(OUTPUT_ROOT):
        fail("reservation path binding mismatch")
    if value.get("request") != request_body(value["source_commit"]):
        fail("reservation request changed from frozen constructor")
    return value


def validate_live(reservation: dict[str, Any], *, output_absent: bool) -> None:
    validate_tree(reservation["source_binding"])
    validate_inputs(reservation["input_bindings"])
    if output_absent and (OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink()):
        fail("unique output root already exists")


def require_controller_lock() -> None:
    if os.environ.get("FASTWAM_CONTROL_NODE") != "ssh970":
        fail("controller mutations are restricted to SSH970 wrapper")
    if os.environ.get("FASTWAM_LOCK_FD") != "9":
        fail("controller lock fd is not frozen")
    try:
        info = os.fstat(9)
    except OSError as exc:
        raise ContractError("controller lock fd is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600:
        fail("controller lock file contract failed")


def prepare(args: argparse.Namespace) -> None:
    require_controller_lock()
    if Path(args.source_root) != SOURCE_ROOT:
        fail(f"source root must equal {SOURCE_ROOT}")
    if not args.source_commit or len(args.source_commit) != 40:
        fail("source commit must be a full Git revision")
    if args.platform_oss_quota_bytes <= 0 or args.platform_oss_free_bytes <= 0:
        fail("platform capacity evidence must be positive")
    if args.platform_oss_free_bytes < 128 * 1024**3:
        fail("platform free space is below the frozen 128 GiB floor")
    if any(path.exists() or path.is_symlink() for path in (LOCAL_ROOT, OUTPUT_ROOT)):
        fail("local prepared/output namespace must be wholly absent")
    if DURABLE_ROOT.exists() or DURABLE_ROOT.is_symlink():
        validate_empty_oss_durable_root(DURABLE_ROOT)
    source_binding = capture_tree(SOURCE_ROOT, include_contents=True)
    bindings = input_bindings()
    validate_worker_dependencies()
    request = request_body(args.source_commit)
    prepared_at = utc_now()
    reservation = {
        "schema": "fastwam-gau0-placefood-same8-reservation-v1",
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "prepared_at": prepared_at,
        "source_root": str(SOURCE_ROOT),
        "source_commit": args.source_commit,
        "output_root": str(OUTPUT_ROOT),
        "source_binding": source_binding,
        "input_bindings": bindings,
        "evaluation": {
            "task": "PlaceFood-rf",
            "gaussian_conditioning": False,
            "arms": ["gau1_stats", "gau0_native_stats"],
            "episodes_per_arm": 8,
            "episode_invocations": 16,
            "primary_comparison": "GAU0 with GAU1 evaluator stats versus historical GAU1",
            "causal_claim": False,
        },
        "platform_capacity": {
            "quota_bytes": args.platform_oss_quota_bytes,
            "free_bytes": args.platform_oss_free_bytes,
            "observed_at": args.platform_oss_observed_at,
            "authority": args.platform_oss_quota_evidence,
            "floor_bytes": 128 * 1024**3,
        },
        "request": request,
    }
    ensure_empty_oss_durable_root(DURABLE_ROOT)
    write_json_exclusive(RESERVATION_PATH, reservation)
    if load_json(RESERVATION_PATH) != reservation:
        fail("prepared reservation direct readback mismatch")
    require_exact_children(DURABLE_ROOT, {RESERVATION_PATH.name})
    safe_mkdir(LOCAL_ROOT)
    write_json_exclusive(STATE_PATH, {
        "schema": "fastwam-gau0-placefood-same8-local-state-v1",
        "phase": "PREPARED",
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "cloud_create_calls": 0,
        "job_id": None,
        "job_status": None,
        "updated_at": utc_now(),
    })
    print(json.dumps({"status": "PREPARED", "reservation": str(RESERVATION_PATH)}, indent=2))


def load_sdk():
    from alibabacloud_credentials.client import Client as CredentialClient
    from alibabacloud_credentials.models import Config as CredentialConfig
    from alibabacloud_pai_dlc20201203 import models
    from alibabacloud_pai_dlc20201203.client import Client
    from alibabacloud_tea_openapi.models import Config
    from alibabacloud_tea_util.models import RuntimeOptions

    profile = json.loads(Path("/root/.aliyun/config.json").read_text(encoding="utf-8"))
    active = profile.get("current")
    profiles = {item.get("name"): item for item in profile.get("profiles", [])}
    if active not in profiles or profiles[active].get("mode") != "CredentialsURI":
        fail("active Alibaba profile is not CredentialsURI")
    credential = CredentialClient(CredentialConfig(type="credentials_uri"))
    client = Client(Config(credential=credential, region_id="cn-beijing", endpoint="pai-dlc.cn-beijing.aliyuncs.com"))
    return client, models, RuntimeOptions


def runtime_options(cls):
    return cls(autoretry=False, max_attempts=1, connect_timeout=10000, read_timeout=30000)


def body_map(response: Any) -> dict[str, Any]:
    body = getattr(response, "body", response)
    value = body.to_map() if hasattr(body, "to_map") else body
    if not isinstance(value, dict):
        fail("SDK response body is not a map")
    return value


def list_jobs(client: Any, models: Any, runtime_cls: Any) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    page = 1
    while True:
        request = models.ListJobsRequest(workspace_id=WORKSPACE_ID, page_number=page, page_size=100, order="desc", sort_by="GmtCreateTime")
        response = client.list_jobs_with_options(request, {}, runtime_options(runtime_cls))
        value = body_map(response)
        page_jobs = value.get("Jobs") or []
        if not isinstance(page_jobs, list):
            fail("ListJobs Jobs is not a list")
        jobs.extend(page_jobs)
        total = int(value.get("TotalCount", len(jobs)))
        if len(jobs) >= total or len(page_jobs) == 0:
            break
        page += 1
    ids = [item.get("JobId") for item in jobs]
    if len(ids) != len(set(ids)) or any(not isinstance(item, str) or not item for item in ids):
        fail("ListJobs returned invalid or duplicate job IDs")
    return jobs


def get_job(client: Any, models: Any, runtime_cls: Any, job_id: str) -> dict[str, Any]:
    request = models.GetJobRequest(need_detail=True)
    response = client.get_job_with_options(job_id, request, {}, runtime_options(runtime_cls))
    value = body_map(response)
    job = value.get("Job", value)
    if not isinstance(job, dict) or job.get("JobId") != job_id:
        fail(f"GetJob identity mismatch for {job_id}")
    return job


def requested_subset(expected: Any, observed: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(observed, dict) and all(key in observed and requested_subset(value, observed[key]) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(observed, list) and len(expected) == len(observed) and all(requested_subset(a, b) for a, b in zip(expected, observed))
    return type(expected) is type(observed) and expected == observed


def custom_env_projection_matches(expected: dict[str, str], job: dict[str, Any]) -> bool:
    observed = job.get("CustomEnvs")
    if not isinstance(observed, list) or len(observed) != len(expected):
        return False
    projection: dict[str, str] = {}
    for item in observed:
        if not isinstance(item, dict) or set(item) != {"Key", "Value", "Visible"}:
            return False
        if item["Visible"] != "public" or not isinstance(item["Key"], str) or not isinstance(item["Value"], str):
            return False
        if item["Key"] in projection:
            return False
        projection[item["Key"]] = item["Value"]
    return projection == expected


def datasource_projection_matches(expected: list[dict[str, Any]], job: dict[str, Any]) -> bool:
    observed = job.get("DataSources")
    if not isinstance(observed, list) or len(observed) != len(expected):
        return False
    requested_projection: list[dict[str, str]] = []
    observed_projection: list[dict[str, str]] = []
    for want, got in zip(expected, observed):
        if (
            not isinstance(want, dict)
            or set(want) != {"DataSourceId", "MountAccess", "MountPath"}
            or want["MountAccess"] not in {"RO", "RW"}
            or not isinstance(got, dict)
            or set(got) != {"DataSourceId", "MountPath", "Uri"}
            or got["Uri"] != ""
        ):
            return False
        requested_projection.append({"DataSourceId": want["DataSourceId"], "MountPath": want["MountPath"]})
        observed_projection.append({"DataSourceId": got["DataSourceId"], "MountPath": got["MountPath"]})
    return requested_projection == observed_projection


def exact_job(request: dict[str, Any], job: dict[str, Any]) -> bool:
    if not requested_subset(request["WorkspaceId"], job.get("WorkspaceId")):
        return False
    if not requested_subset(request["ResourceId"], job.get("ResourceId")):
        return False
    if job.get("DisplayName") != request["DisplayName"]:
        return False
    if request.get("CustomEnvs") != []:
        return False
    if not custom_env_projection_matches(request["Envs"], job):
        return False
    if not datasource_projection_matches(request["DataSources"], job):
        return False
    omitted_or_exact = ("JobMaxRunningTimeMinutes", "SuccessPolicy")
    for key in omitted_or_exact:
        if key not in request or (key in job and not requested_subset(request[key], job[key])):
            return False
    ignored = {"CustomEnvs", "DataSources", "Envs", *omitted_or_exact}
    return all(key in job and requested_subset(value, job[key]) for key, value in request.items() if key not in ignored)


def sdk_request(models: Any, body: dict[str, Any]):
    expected_tags = {"experiment_id": EXPERIMENT_ID, "run_id": RUN_ID}
    settings = body.get("Settings")
    if not isinstance(settings, dict) or not isinstance(settings.get("Tags"), dict):
        fail("DLC Settings.Tags must be the frozen string map")
    if settings["Tags"] != expected_tags:
        fail("DLC Settings.Tags changed from the frozen identity map")
    request = models.CreateJobRequest().from_map(body)
    request.validate()
    if request.to_map() != body:
        fail("SDK CreateJobRequest round-trip changed the frozen request")
    return request


def display_candidates(client: Any, models: Any, runtime_cls: Any) -> list[dict[str, Any]]:
    results = []
    for summary in list_jobs(client, models, runtime_cls):
        if summary.get("DisplayName") != DISPLAY_NAME:
            continue
        results.append(get_job(client, models, runtime_cls, summary["JobId"]))
    return results


def exact_candidates(client: Any, models: Any, runtime_cls: Any, request: dict[str, Any]) -> list[dict[str, Any]]:
    return [job for job in display_candidates(client, models, runtime_cls) if exact_job(request, job)]


def persist_ack(job: dict[str, Any], source: str) -> None:
    if ACK_PATH.exists() or ACK_PATH.is_symlink():
        existing = load_json(ACK_PATH)
        if existing.get("job_id") != job["JobId"]:
            fail("durable ACK conflicts with observed job")
        return
    write_json_exclusive(ACK_PATH, {
        "schema": "fastwam-dlc-job-acknowledgement-v1",
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "job_id": job["JobId"],
        "status": job.get("Status"),
        "source": source,
        "recorded_at": utc_now(),
    })
    state = load_json(STATE_PATH)
    state.update({"phase": "ACKNOWLEDGED", "cloud_create_calls": 1, "job_id": job["JobId"], "job_status": job.get("Status"), "updated_at": utc_now()})
    replacement = STATE_PATH.with_name("state.json.next")
    write_json_exclusive(replacement, state)
    os.replace(replacement, STATE_PATH)


def submit(args: argparse.Namespace) -> None:
    require_controller_lock()
    if args.confirm_experiment_id != EXPERIMENT_ID:
        fail("experiment confirmation mismatch")
    reservation = load_reservation()
    validate_live(reservation, output_absent=True)
    validate_worker_dependencies()
    client, models, runtime_cls = load_sdk()
    request = sdk_request(models, reservation["request"])
    display_matches = display_candidates(client, models, runtime_cls)
    candidates = [job for job in display_matches if exact_job(reservation["request"], job)]
    if ACK_PATH.exists() or ACK_PATH.is_symlink():
        ack = load_json(ACK_PATH)
        if len(display_matches) != 1 or len(candidates) != 1 or candidates[0]["JobId"] != ack.get("job_id"):
            fail("ACK reconciliation did not find exactly one matching job")
        print(json.dumps({"status": "ALREADY_ACKNOWLEDGED", "job_id": ack["job_id"]}, indent=2))
        return
    if LATCH_PATH.exists() or LATCH_PATH.is_symlink():
        if len(display_matches) != 1 or len(candidates) != 1:
            fail("permanent latch exists and reconciliation is not uniquely provable")
        persist_ack(candidates[0], "permanent_latch_reconciliation")
        print(json.dumps({"status": "RECONCILED", "job_id": candidates[0]["JobId"]}, indent=2))
        return
    if display_matches:
        fail("cloud job with the frozen display name exists before permanent latch")
    write_json_exclusive(LATCH_PATH, {
        "schema": "fastwam-dlc-permanent-submission-latch-v1",
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "reservation": str(RESERVATION_PATH),
        "latched_at": utc_now(),
    })
    response = client.create_job_with_options(request, {}, runtime_options(runtime_cls))
    response_value = body_map(response)
    job_id = response_value.get("JobId")
    if not isinstance(job_id, str) or not job_id:
        fail("CreateJob returned no JobId; permanent latch forbids retry")
    job = get_job(client, models, runtime_cls, job_id)
    if not exact_job(reservation["request"], job):
        fail("created job does not match frozen request; permanent latch forbids retry")
    persist_ack(job, "CreateJob_then_GetJob")
    print(json.dumps({"status": "ACKNOWLEDGED", "job_id": job_id, "provider_status": job.get("Status")}, indent=2))


def worker_preflight() -> None:
    if os.environ.get("FASTWAM_RESERVATION_PATH") != str(RESERVATION_PATH):
        fail("worker reservation path mismatch")
    reservation = load_reservation()
    validate_live(reservation, output_absent=True)
    for key, value in reservation["request"]["Envs"].items():
        if os.environ.get(key) != value:
            fail(f"worker environment differs from frozen request: {key}")
    validate_worker_dependencies()
    print("GAU0_WORKER_PREFLIGHT_PASS")


def job_id() -> None:
    ack = load_json(ACK_PATH)
    value = ack.get("job_id")
    if not isinstance(value, str) or not value:
        fail("durable ACK contains no job ID")
    print(value)


def tree_inventory(root: Path) -> tuple[set[str], set[str], int]:
    require_dir(root)
    files: set[str] = set()
    directories: set[str] = {"."}
    total_bytes = 0
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        current_path = Path(current)
        relative_current = current_path.relative_to(root)
        for name in dirnames:
            path = current_path / name
            if not stat.S_ISDIR(path.lstat().st_mode):
                fail(f"unsupported terminal directory entry: {path}")
            directories.add((relative_current / name).as_posix())
        for name in filenames:
            path = current_path / name
            info = require_file(path)
            relative = (relative_current / name).as_posix()
            if name.endswith((".pyc", ".pyo")):
                fail(f"bytecode in terminal output: {path}")
            files.add(relative)
            total_bytes += info.st_size
    return files, directories, total_bytes


def validate_terminal() -> dict[str, Any]:
    reservation = load_reservation()
    validate_live(reservation, output_absent=False)
    ack = load_json(ACK_PATH)
    module = load_aggregator(SOURCE_ROOT)
    arms = {arm: module.validate_arm(OUTPUT_ROOT, arm) for arm in module.ARMS}
    baseline = module.validate_baseline(BASELINE_ROOT)
    compared = module.comparison(arms["gau1_stats"], arms["gau0_native_stats"], baseline)
    expected_files = {f"{arm}/episode-{index:02d}/{name}" for arm in module.ARMS for index in range(8) for name in ("episodes.jsonl", "run_manifest.json", "summary.json")}
    expected_files |= {"gau1_stats-aggregate.json", "gau0_native_stats-aggregate.json", "comparison.json", "terminal-receipt.json", "COMPLETE.json"}
    expected_dirs = {"."} | {arm for arm in module.ARMS} | {f"{arm}/episode-{index:02d}" for arm in module.ARMS for index in range(8)}
    files, directories, total_bytes = tree_inventory(OUTPUT_ROOT)
    if files != expected_files or directories != expected_dirs or len(files) != 53 or len(directories) != 19:
        fail("terminal output closed allowlist mismatch")
    if load_json(OUTPUT_ROOT / "gau1_stats-aggregate.json") != arms["gau1_stats"]:
        fail("gau1_stats aggregate mismatch")
    if load_json(OUTPUT_ROOT / "gau0_native_stats-aggregate.json") != arms["gau0_native_stats"]:
        fail("gau0_native_stats aggregate mismatch")
    if load_json(OUTPUT_ROOT / "comparison.json") != compared:
        fail("comparison artifact mismatch")
    terminal = load_json(OUTPUT_ROOT / "terminal-receipt.json")
    complete = load_json(OUTPUT_ROOT / "COMPLETE.json")
    if terminal.get("schema") != "fastwam-gau0-placefood-same8-terminal-v1" or terminal.get("status") != "SCIENTIFIC_COMPLETE":
        fail("terminal receipt status/schema mismatch")
    if terminal.get("source_commit") != reservation["source_commit"] or terminal.get("job_id") != ack.get("job_id"):
        fail("terminal source/job binding mismatch")
    if terminal.get("gaussian_conditioning") is not False or terminal.get("episode_invocations") != 16:
        fail("terminal GAU0/evaluator accounting mismatch")
    if terminal.get("arms") != list(module.ARMS):
        fail("terminal arm declaration mismatch")
    completed_at = terminal.get("completed_at")
    if not isinstance(completed_at, str) or not completed_at.endswith("Z"):
        fail("terminal completion time is missing or malformed")
    try:
        parsed_completed_at = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("terminal completion time is malformed") from exc
    if parsed_completed_at.tzinfo is None or parsed_completed_at.utcoffset() != timezone.utc.utcoffset(parsed_completed_at):
        fail("terminal completion time is not UTC")
    artifact_files = expected_files - {"terminal-receipt.json", "COMPLETE.json"}
    artifact_bytes = sum(require_file(OUTPUT_ROOT / relative).st_size for relative in sorted(artifact_files))
    if (
        terminal.get("artifact_files_before_terminal") != len(artifact_files)
        or terminal.get("artifact_bytes_before_terminal") != artifact_bytes
        or terminal.get("comparison") != compared
    ):
        fail("terminal artifact/comparison declaration mismatch")
    if complete != {
        "schema": "fastwam-gau0-placefood-same8-complete-v1",
        "status": "SCIENTIFIC_COMPLETE",
        "terminal_receipt": "terminal-receipt.json",
        "completed_at": terminal.get("completed_at"),
    }:
        fail("COMPLETE marker mismatch")
    complete_info = require_file(OUTPUT_ROOT / "COMPLETE.json")
    terminal_info = require_file(OUTPUT_ROOT / "terminal-receipt.json")
    if complete_info.st_mtime_ns < terminal_info.st_mtime_ns:
        fail("COMPLETE marker does not follow terminal receipt")
    return {
        "status": "SCIENTIFIC_COMPLETE",
        "job_id": ack["job_id"],
        "artifact_files": len(files),
        "directories": len(directories),
        "published_bytes": total_bytes,
        "comparison": compared,
    }


def validate_terminal_command(args: argparse.Namespace) -> None:
    if args.member != "gau0":
        fail("only frozen member gau0 is valid")
    print(json.dumps(validate_terminal(), indent=2, sort_keys=True))


def show() -> None:
    result: dict[str, Any] = {"experiment_id": EXPERIMENT_ID, "run_id": RUN_ID}
    for name, path in (("reservation", RESERVATION_PATH), ("latch", LATCH_PATH), ("ack", ACK_PATH), ("state", STATE_PATH)):
        result[name] = load_json(path) if path.exists() and not path.is_symlink() else None
    result["output_exists"] = OUTPUT_ROOT.exists()
    print(json.dumps(result, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--source-root", required=True)
    prepare_parser.add_argument("--source-commit", required=True)
    prepare_parser.add_argument("--platform-oss-quota-bytes", type=int, required=True)
    prepare_parser.add_argument("--platform-oss-free-bytes", type=int, required=True)
    prepare_parser.add_argument("--platform-oss-quota-evidence", required=True)
    prepare_parser.add_argument("--platform-oss-observed-at", required=True)
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--confirm-experiment-id", required=True)
    sub.add_parser("worker-preflight")
    sub.add_parser("job-id")
    terminal_parser = sub.add_parser("validate-terminal")
    terminal_parser.add_argument("--member", required=True)
    sub.add_parser("show")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command == "submit":
            submit(args)
        elif args.command == "worker-preflight":
            worker_preflight()
        elif args.command == "job-id":
            job_id()
        elif args.command == "validate-terminal":
            validate_terminal_command(args)
        elif args.command == "show":
            show()
        else:
            fail(f"unsupported command: {args.command}")
        return 0
    except ContractError as exc:
        print(f"GAU0_CONTROLLER_FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
