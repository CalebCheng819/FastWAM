"""Fail-closed terminal artifacts for formal training and full-model gates."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import ctypes
import errno
import secrets
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


SHA256_HEX = frozenset("0123456789abcdef")
N4_GATE_WORLD_SIZE = 32
N4_GATE_LOCAL_MICRO_BATCH_SIZE = 1
N4_GATE_GRADIENT_ACCUMULATION_STEPS = 1
N4_GATE_GLOBAL_TRAIN_BATCH_SIZE = 32
N4_GATE_TRAIN_STEPS = 2
N4_GATE_MAX_PEAK_ALLOCATED_BYTES = 42 * 2**30
N4_GATE_MAX_PEAK_RESERVED_BYTES = 44 * 2**30


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def require_sha256(value: str, *, label: str) -> str:
    value = str(value).strip().lower()
    if len(value) != 64 or any(character not in SHA256_HEX for character in value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def safe_relative_path(value: str) -> PurePosixPath:
    if not value or any(character in value for character in ("\x00", "\n", "\r")):
        raise ValueError(f"unsafe relative artifact path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe relative artifact path: {value!r}")
    return relative


def resolved_unaliased_directory(path: str | Path, *, label: str) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise ValueError(f"{label} must be absolute: {supplied}")
    absolute = Path(os.path.abspath(supplied))
    resolved = supplied.resolve(strict=True)
    if absolute != resolved or resolved.is_symlink() or not resolved.is_dir():
        raise ValueError(f"{label} must be an existing unaliased directory: {supplied}")
    return resolved


def _open_regular(path: Path, *, require_single_link: bool = True) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise ValueError(f"artifact must be a regular file: {path}")
    if require_single_link and info.st_nlink != 1:
        os.close(descriptor)
        raise ValueError(f"artifact must not be hard-linked: nlink={info.st_nlink} path={path}")
    return descriptor, info


def sha256_regular_file(path: str | Path, *, require_single_link: bool = True) -> tuple[str, int]:
    path = Path(path)
    descriptor, before = _open_regular(path, require_single_link=require_single_link)
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise RuntimeError(f"artifact changed while hashing: {path}")
        return digest.hexdigest(), int(after.st_size)
    finally:
        os.close(descriptor)


def read_canonical_json(path: str | Path) -> tuple[dict[str, Any], str, int]:
    path = Path(path)
    descriptor, before = _open_regular(path)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            encoded = handle.read()
        after = os.fstat(descriptor)
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"artifact changed while reading: {path}")
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    if encoded != canonical_json_bytes(payload):
        raise ValueError(f"JSON artifact is not canonical: {path}")
    return payload, hashlib.sha256(encoded).hexdigest(), len(encoded)


def publish_exclusive_bytes(path: str | Path, payload: bytes) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace formal artifact: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError(f"formal artifact parent must be a non-symlink directory: {path.parent}")
    temporary = path.parent / (
        f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        _rename_noreplace(temporary, path)
        parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish ``source`` without ever replacing ``destination``."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "libc renameat2 is unavailable; refusing unsafe publish")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            f"refusing to replace formal artifact: {destination}",
            str(destination),
        )
    raise OSError(
        error_number,
        f"atomic no-clobber publication failed for {destination}; no fallback is allowed",
        str(destination),
    )


def publish_exclusive_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    publish_exclusive_bytes(path, canonical_json_bytes(payload))


def publish_failure_marker(
    output_root: str | Path,
    *,
    marker_name: str,
    schema_name: str,
    error: BaseException,
    success_markers: Sequence[str],
) -> dict[str, Any]:
    """Publish a task-owned terminal failure signal without ever replacing PASS."""

    output_root = resolved_unaliased_directory(output_root, label="formal output root")
    for success_name in success_markers:
        success = output_root / safe_relative_path(success_name)
        if success.exists() or success.is_symlink():
            raise RuntimeError(
                f"refusing failure publication after a success marker exists: {success}"
            )
    marker_relative = safe_relative_path(marker_name)
    message = str(error)
    if len(message) > 4096:
        message = message[:4096] + "...[truncated]"
    payload = {
        "error_message": message,
        "error_type": type(error).__name__,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "run_id": os.environ.get("RUN_ID", ""),
        "schema_name": schema_name,
        "schema_version": 1,
        "status": "FAIL",
    }
    publish_exclusive_json(output_root / marker_relative, payload)
    return payload


def _canonical_fingerprint(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            digest.update(b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
            digest.update(b"\0")
            digest.update(array.tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda candidate: (type(candidate).__name__, repr(candidate))):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            for nested in item:
                update(nested)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            digest.update(repr(item).encode("utf-8"))
            digest.update(b"\0")
        else:
            buffer = io.BytesIO()
            torch.save(item, buffer)
            digest.update(b"torch-save\0")
            digest.update(buffer.getvalue())

    update(value)
    return digest.hexdigest()


def _sample_tensor(tensor: torch.Tensor, *, values_per_edge: int = 8) -> dict[str, Any]:
    flat = tensor.detach().reshape(-1)
    count = min(values_per_edge, int(flat.numel()))
    if count:
        sample = torch.cat((flat[:count], flat[-count:])).cpu().contiguous()
    else:
        sample = torch.empty(0, dtype=tensor.dtype)
    return {
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "sample_sha256": _canonical_fingerprint(sample),
        "shape": list(tensor.shape),
    }


def model_probe(model: torch.nn.Module, *, limit: int = 8) -> dict[str, Any]:
    parameters = sorted(
        ((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad),
        key=lambda item: item[0],
    )
    if not parameters:
        raise RuntimeError("formal state probe found no trainable model parameters")
    selected = parameters[: max(limit // 2, 1)] + parameters[-max(limit // 2, 1) :]
    deduplicated: list[tuple[str, torch.Tensor]] = []
    observed: set[str] = set()
    for name, parameter in selected:
        if name not in observed:
            observed.add(name)
            deduplicated.append((name, parameter))
    records = [{"name": name, **_sample_tensor(parameter)} for name, parameter in deduplicated]
    return {
        "fingerprint": canonical_json_sha256({"records": records}),
        "records": records,
        "trainable_parameter_count": len(parameters),
    }


def _optimizer_with_state(optimizer: Any) -> Any:
    current = optimizer
    visited: set[int] = set()
    candidates = []
    for _ in range(12):
        if id(current) in visited:
            break
        visited.add(id(current))
        candidates.append(current)
        nested = getattr(current, "optimizer", None)
        if nested is None or nested is current:
            break
        current = nested
    for candidate in reversed(candidates):
        state = getattr(candidate, "state", None)
        groups = getattr(candidate, "param_groups", None)
        if isinstance(state, Mapping) and state and isinstance(groups, Sequence):
            return candidate
    raise RuntimeError("formal state probe found no populated optimizer state")


def optimizer_probe(
    optimizer: Any,
    *,
    limit: int = 8,
    require_populated_state: bool = True,
) -> dict[str, Any]:
    try:
        concrete = _optimizer_with_state(optimizer)
    except RuntimeError:
        if require_populated_state:
            raise
        groups = getattr(optimizer, "param_groups", [])
        summary = [
            {
                str(key): value
                for key, value in sorted(group.items())
                if key != "params" and isinstance(value, (str, int, float, bool, type(None)))
            }
            for group in groups
        ]
        return {
            "concrete_type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "fingerprint": canonical_json_sha256({"empty_param_groups": summary}),
            "records": [],
        }
    records = []
    for group_index, group in enumerate(concrete.param_groups):
        for parameter_index, parameter in enumerate(group.get("params", [])):
            state = concrete.state.get(parameter)
            if not state:
                continue
            sampled_state = {}
            for key in sorted(state, key=lambda candidate: str(candidate)):
                value = state[key]
                sampled_state[str(key)] = (
                    _sample_tensor(value) if isinstance(value, torch.Tensor) else value
                )
            records.append(
                {
                    "group_index": group_index,
                    "parameter_index": parameter_index,
                    "state": sampled_state,
                }
            )
            if len(records) >= limit:
                break
        if len(records) >= limit:
            break
    if not records:
        raise RuntimeError("formal state probe found no sampleable optimizer tensors")
    return {
        "concrete_type": f"{type(concrete).__module__}.{type(concrete).__qualname__}",
        "fingerprint": canonical_json_sha256({"records": records}),
        "records": records,
    }


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state(torch.cuda.current_device())
    return state


def next_rng_sample(device: torch.device) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "numpy": np.random.random(4).tolist(),
        "python": [random.random() for _ in range(4)],
        "torch_cpu": torch.rand(4, device="cpu").tolist(),
    }
    if device.type == "cuda":
        sample["torch_cuda"] = torch.rand(4, device=device).cpu().tolist()
    return sample


def state_fingerprints(
    *,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    global_step: int,
    require_optimizer_state: bool = True,
) -> dict[str, Any]:
    model_state = model_probe(model)
    optimizer_state = optimizer_probe(
        optimizer, require_populated_state=require_optimizer_state
    )
    scheduler_state = scheduler.state_dict()
    return {
        "global_step": int(global_step),
        "model": model_state["fingerprint"],
        "model_probe": model_state,
        "optimizer": optimizer_state["fingerprint"],
        "optimizer_probe": optimizer_state,
        "rng": _canonical_fingerprint(rng_state()),
        "scheduler": _canonical_fingerprint(scheduler_state),
    }


def _validate_state_tree_metadata(state_root: Path, manifest_path: Path) -> dict[str, Any]:
    payload, manifest_sha256, _ = read_canonical_json(manifest_path)
    if set(payload) != {"files", "role", "schema_version", "total_bytes"}:
        raise ValueError(f"state-tree manifest fields mismatch: {manifest_path}")
    if payload["schema_version"] != 1 or payload["role"] != "accelerate_zero2_full_state":
        raise ValueError(f"state-tree manifest role/schema mismatch: {manifest_path}")
    records = payload["files"]
    if not isinstance(records, list) or not records:
        raise ValueError(f"state-tree manifest is empty: {manifest_path}")
    expected: dict[PurePosixPath, tuple[int, str]] = {}
    previous: bytes | None = None
    for record in records:
        if not isinstance(record, dict) or set(record) != {"bytes", "path", "sha256"}:
            raise ValueError(f"invalid state-tree record: {manifest_path}")
        relative = safe_relative_path(record["path"])
        key = os.fsencode(relative.as_posix())
        if previous is not None and key <= previous:
            raise ValueError(f"state-tree records are not unique and sorted: {manifest_path}")
        previous = key
        size = record["bytes"]
        digest = require_sha256(record["sha256"], label=f"state file {relative} SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid state file size for {relative}: {size!r}")
        expected[relative] = (size, digest)
    if PurePosixPath("trainer_state.json") not in expected:
        raise ValueError(f"state-tree manifest does not bind trainer_state.json: {manifest_path}")
    observed: dict[PurePosixPath, Path] = {}
    for current, directories, files in os.walk(state_root, topdown=True, followlinks=False):
        directories.sort(key=os.fsencode)
        files.sort(key=os.fsencode)
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"state tree contains an aliased/special directory: {path}")
        for name in files:
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(f"state tree contains an aliased/special file: {path}")
            observed[safe_relative_path(path.relative_to(state_root).as_posix())] = path
    if set(observed) != set(expected):
        missing = sorted(path.as_posix() for path in set(expected) - set(observed))
        unexpected = sorted(path.as_posix() for path in set(observed) - set(expected))
        raise RuntimeError(
            f"state tree inventory mismatch: missing={missing[:12]} unexpected={unexpected[:12]}"
        )
    total_bytes = 0
    for relative, (expected_size, expected_sha256) in expected.items():
        actual_sha256, actual_size = sha256_regular_file(observed[relative])
        if actual_size != expected_size:
            raise RuntimeError(
                f"state file size mismatch: expected={expected_size} actual={actual_size} "
                f"path={observed[relative]}"
            )
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "state file SHA-256 mismatch during terminal strong readback: "
                f"expected={expected_sha256} actual={actual_sha256} "
                f"path={observed[relative]}"
            )
        total_bytes += actual_size
    if payload["total_bytes"] != total_bytes:
        raise RuntimeError(
            f"state-tree total mismatch: expected={payload['total_bytes']} actual={total_bytes}"
        )
    return {
        "file_count": len(expected),
        "manifest_sha256": manifest_sha256,
        "total_bytes": total_bytes,
    }


def checkpoint_seal_descriptor(
    output_root: str | Path,
    *,
    step: int,
    rehash_weights: bool,
) -> dict[str, Any]:
    output_root = resolved_unaliased_directory(output_root, label="training output root")
    tag = f"step_{int(step):06d}"
    weights = output_root / "checkpoints" / "weights" / f"{tag}.pt"
    weights_manifest = weights.with_name(f"{weights.name}.manifest.json")
    weights_complete = weights.with_name(f"{weights.name}.COMPLETE")
    state_root = output_root / "checkpoints" / "state" / tag
    state_manifest = state_root.with_name(f"{tag}.state-tree.json")

    manifest_payload, weights_manifest_sha256, _ = read_canonical_json(weights_manifest)
    if set(manifest_payload) != {
        "bytes",
        "checkpoint_state_kind",
        "filename",
        "global_step",
        "schema_name",
        "schema_version",
        "sha256",
    }:
        raise ValueError(f"weights manifest fields mismatch: {weights_manifest}")
    expected_checkpoint_sha256 = require_sha256(
        manifest_payload["sha256"], label="weights checkpoint SHA-256"
    )
    if (
        manifest_payload["schema_name"] != "fastwam-weights-checkpoint"
        or manifest_payload["schema_version"] != 1
        or manifest_payload["filename"] != weights.name
        or manifest_payload["global_step"] != int(step)
        or manifest_payload["checkpoint_state_kind"] != "full"
    ):
        raise ValueError(f"weights manifest semantic mismatch: {weights_manifest}")
    descriptor, info = _open_regular(weights)
    os.close(descriptor)
    if info.st_size != manifest_payload["bytes"]:
        raise RuntimeError(f"weights checkpoint byte-size mismatch: {weights}")
    if rehash_weights:
        actual_checkpoint_sha256, _ = sha256_regular_file(weights)
        if actual_checkpoint_sha256 != expected_checkpoint_sha256:
            raise RuntimeError(
                "weights checkpoint SHA-256 mismatch: "
                f"expected={expected_checkpoint_sha256} actual={actual_checkpoint_sha256}"
            )

    complete_payload, weights_complete_sha256, _ = read_canonical_json(weights_complete)
    if set(complete_payload) != {
        "checkpoint_sha256",
        "manifest_filename",
        "manifest_sha256",
        "schema_name",
        "schema_version",
    }:
        raise ValueError(f"weights COMPLETE fields mismatch: {weights_complete}")
    if (
        complete_payload["schema_name"] != "fastwam-weights-checkpoint-complete"
        or complete_payload["schema_version"] != 1
        or complete_payload["manifest_filename"] != weights_manifest.name
        or complete_payload["manifest_sha256"] != weights_manifest_sha256
        or complete_payload["checkpoint_sha256"] != expected_checkpoint_sha256
    ):
        raise RuntimeError(f"weights COMPLETE does not bind its manifest/checkpoint: {weights_complete}")

    state_summary = _validate_state_tree_metadata(state_root, state_manifest)
    trainer_state = state_root / "trainer_state.json"
    trainer_payload, trainer_state_sha256, _ = read_canonical_json(trainer_state)
    if trainer_payload.get("global_step") != int(step):
        raise RuntimeError(f"trainer_state global_step mismatch: {trainer_state}")

    def relative(path: Path) -> str:
        return path.relative_to(output_root).as_posix()

    return {
        "global_step": int(step),
        "state": {
            "file_count": state_summary["file_count"],
            "manifest": relative(state_manifest),
            "manifest_sha256": state_summary["manifest_sha256"],
            "root": relative(state_root),
            "total_bytes": state_summary["total_bytes"],
            "trainer_state_sha256": trainer_state_sha256,
        },
        "weights": {
            "bytes": int(info.st_size),
            "checkpoint": relative(weights),
            "checkpoint_sha256": expected_checkpoint_sha256,
            "complete": relative(weights_complete),
            "complete_sha256": weights_complete_sha256,
            "manifest": relative(weights_manifest),
            "manifest_sha256": weights_manifest_sha256,
            "rehash_verified": bool(rehash_weights),
        },
    }


def _publish_sha256sums(output_root: Path, relative_paths: Iterable[str]) -> tuple[str, list[str]]:
    paths = sorted({safe_relative_path(value) for value in relative_paths}, key=lambda item: os.fsencode(item.as_posix()))
    records = []
    lines = []
    for relative in paths:
        digest, size = sha256_regular_file(output_root / relative)
        records.append(relative.as_posix())
        lines.append(f"{digest}  {relative.as_posix()}\n")
    payload = "".join(lines).encode("utf-8")
    publish_exclusive_bytes(output_root / "SHA256SUMS", payload)
    return hashlib.sha256(payload).hexdigest(), records


def normalize_formal_evaluation_records(
    evaluation_records: Sequence[Mapping[str, Any]],
    *,
    expected_steps: Sequence[int],
    training_mode: str,
) -> list[dict[str, Any]]:
    """Validate the exact offline-eval evidence required by one treatment."""

    training_mode = str(training_mode).strip().lower()
    if training_mode not in {"joint", "action_only_cache"}:
        raise ValueError(
            f"unsupported formal training mode for eval contract: {training_mode!r}"
        )
    expected_steps = [int(step) for step in expected_steps]
    base_fields = {
        "evaluation_kind",
        "offline_agent_counts",
        "offline_samples",
        "offline_tasks",
        "step",
        "val_loss",
        "val_loss_action",
    }
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(evaluation_records):
        if not isinstance(record, Mapping):
            raise TypeError(f"formal evaluation record {index} must be a mapping")
        fields = set(record)
        allowed_fields = base_fields | {"val_loss_video"}
        if fields != base_fields and fields != allowed_fields:
            raise RuntimeError(
                f"formal evaluation record field mismatch at index {index}: {sorted(fields)}"
            )
        tasks = record.get("offline_tasks")
        counts = record.get("offline_agent_counts")
        if (
            record.get("evaluation_kind") != "multi_robot_offline_loss"
            or record.get("offline_samples") != 12
            or not isinstance(counts, list)
            or counts != [2, 3, 4]
            or not isinstance(tasks, list)
            or not tasks
            or tasks != sorted(set(str(value) for value in tasks))
        ):
            raise RuntimeError(f"formal offline-eval contract mismatch: {record}")
        for metric_name in ("val_loss", "val_loss_action"):
            if not np.isfinite(float(record[metric_name])):
                raise RuntimeError(
                    f"formal offline evaluation lacks finite {metric_name}: {record}"
                )
        video_value = record.get("val_loss_video")
        if training_mode == "joint":
            if video_value is None or not np.isfinite(float(video_value)):
                raise RuntimeError(
                    f"joint VideoGen evaluation lacks finite val_loss_video: {record}"
                )
        elif video_value is not None:
            raise RuntimeError(
                "action-only evaluation must not claim a video-loss metric: "
                f"{record}"
            )
        normalized_record = dict(record)
        normalized_record["val_loss_video"] = (
            float(video_value) if training_mode == "joint" else None
        )
        normalized.append(normalized_record)
    observed_steps = [int(record["step"]) for record in normalized]
    if observed_steps != expected_steps:
        raise RuntimeError(
            "formal evaluation steps mismatch: "
            f"expected={expected_steps} observed={observed_steps}"
        )
    return normalized


def publish_training_terminal_seal(
    output_root: str | Path,
    *,
    run_id: str,
    code_commit: str,
    config_relative_path: str,
    config_sha256: str,
    max_steps: int,
    expected_checkpoint_steps: Sequence[int],
    expected_evaluation_steps: Sequence[int],
    world_size: int,
    last_step_metrics: Mapping[str, Any],
    evaluation_records: Sequence[Mapping[str, Any]],
    training_mode: str,
    dataset_contract_sha256: str,
    authorization_gate_complete_sha256: str,
    rehash_weights: bool = True,
) -> dict[str, Any]:
    output_root = resolved_unaliased_directory(output_root, label="training output root")
    for name in ("training-summary.json", "SHA256SUMS", "TRAINING.COMPLETE"):
        target = output_root / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to replace terminal training artifact: {target}")
    if int(world_size) != N4_GATE_WORLD_SIZE:
        raise ValueError(f"formal training terminal seal requires world_size=32, got {world_size}")
    code_commit = str(code_commit).lower()
    if len(code_commit) != 40 or any(character not in SHA256_HEX for character in code_commit):
        raise ValueError("formal training terminal seal requires a 40-hex code commit")
    config_sha256 = require_sha256(config_sha256, label="resolved config SHA-256")
    dataset_contract_sha256 = require_sha256(
        dataset_contract_sha256, label="dataset contract SHA-256"
    )
    authorization_gate_complete_sha256 = require_sha256(
        authorization_gate_complete_sha256,
        label="N=4 full-model authorization gate COMPLETE SHA-256",
    )
    reservation_path = output_root / ".RUN_RESERVED"
    reservation, reservation_sha256, _ = read_canonical_json(reservation_path)
    reservation_identity_sha256 = require_sha256(
        reservation.get("identity_sha256", ""),
        label="formal run reservation identity SHA-256",
    )
    reservation_identity_payload = dict(reservation)
    del reservation_identity_payload["identity_sha256"]
    if canonical_json_sha256(reservation_identity_payload) != reservation_identity_sha256:
        raise RuntimeError("formal .RUN_RESERVED identity_sha256 does not match its payload")
    reservation_contract = {
        "code_commit": code_commit,
        "global_world_size": int(world_size),
        "n4_fullmodel_gate_complete_sha256": authorization_gate_complete_sha256,
        "run_id": str(run_id),
        "schema_version": 1,
    }
    reservation_mismatches = {
        key: {"expected": expected, "observed": reservation.get(key)}
        for key, expected in reservation_contract.items()
        if reservation.get(key) != expected
    }
    if reservation_mismatches:
        raise RuntimeError(
            "formal .RUN_RESERVED does not authorize the terminal run: "
            f"{reservation_mismatches}"
        )
    config_relative = safe_relative_path(config_relative_path)
    actual_config_sha256, _ = sha256_regular_file(output_root / config_relative)
    if actual_config_sha256 != config_sha256:
        raise RuntimeError(
            f"resolved config SHA-256 mismatch: expected={config_sha256} actual={actual_config_sha256}"
        )
    steps = sorted({int(step) for step in expected_checkpoint_steps})
    if not steps or steps[-1] != int(max_steps):
        raise ValueError(
            f"terminal checkpoint steps must be non-empty and end at max_steps={max_steps}: {steps}"
        )
    checkpoints = [
        checkpoint_seal_descriptor(output_root, step=step, rehash_weights=rehash_weights)
        for step in steps
    ]
    required_last_metric_fields = {
        "grad_norm",
        "learning_rate",
        "loss",
        "loss_components",
        "step",
    }
    if set(last_step_metrics) != required_last_metric_fields:
        raise ValueError(
            "terminal last-step metric fields mismatch: "
            f"expected={sorted(required_last_metric_fields)} "
            f"observed={sorted(last_step_metrics)}"
        )
    if last_step_metrics.get("step") != int(max_steps):
        raise RuntimeError(
            f"terminal metrics must describe max_steps={max_steps}: {last_step_metrics}"
        )
    finite_terminal_values = {
        "loss": last_step_metrics.get("loss"),
        "grad_norm": last_step_metrics.get("grad_norm"),
        "learning_rate": last_step_metrics.get("learning_rate"),
        **{
            f"loss_components.{key}": value
            for key, value in dict(last_step_metrics.get("loss_components", {})).items()
        },
    }
    if not finite_terminal_values or not all(
        np.isfinite(float(value)) for value in finite_terminal_values.values()
    ):
        raise RuntimeError(
            f"terminal metrics contain non-finite values: {finite_terminal_values}"
        )
    expected_eval_steps = [int(step) for step in expected_evaluation_steps]
    if expected_eval_steps != sorted(set(expected_eval_steps)):
        raise ValueError(
            f"expected evaluation steps must be unique and sorted: {expected_eval_steps}"
        )
    if int(max_steps) == 5000 and expected_eval_steps != [1000, 2000, 3000, 4000, 5000]:
        raise ValueError(
            "the formal 5000-step mixed N=2/3/4 run requires offline eval at "
            "steps [1000,2000,3000,4000,5000]"
        )
    normalized_evaluations = normalize_formal_evaluation_records(
        evaluation_records,
        expected_steps=expected_eval_steps,
        training_mode=training_mode,
    )
    observed_eval_steps = [int(record["step"]) for record in normalized_evaluations]
    if observed_eval_steps != expected_eval_steps:
        raise RuntimeError(
            "terminal evaluation steps mismatch: "
            f"expected={expected_eval_steps} observed={observed_eval_steps}"
        )
    summary = {
        "authorization_gate_complete_sha256": authorization_gate_complete_sha256,
        "checkpoints": checkpoints,
        "code_commit": code_commit,
        "config": {"path": config_relative.as_posix(), "sha256": config_sha256},
        "dataset_contract_sha256": dataset_contract_sha256,
        "evaluation_records": normalized_evaluations,
        "last_step_metrics": dict(last_step_metrics),
        "max_steps": int(max_steps),
        "published_at": datetime.now(timezone.utc).isoformat(),
        "reservation": {
            "identity_sha256": reservation_identity_sha256,
            "path": ".RUN_RESERVED",
            "sha256": reservation_sha256,
        },
        "run_id": str(run_id),
        "schema_name": "fastwam-training-summary",
        "schema_version": 1,
        "status": "PASS",
        "treatment": {
            "training_mode": training_mode,
            "video_gen": training_mode == "joint",
        },
        "world_size": int(world_size),
    }
    summary_path = output_root / "training-summary.json"
    publish_exclusive_json(summary_path, summary)
    bound_paths = [
        summary_path.relative_to(output_root).as_posix(),
        config_relative.as_posix(),
        ".RUN_RESERVED",
    ]
    for checkpoint in checkpoints:
        bound_paths.extend(
            (
                checkpoint["weights"]["manifest"],
                checkpoint["weights"]["complete"],
                checkpoint["state"]["manifest"],
                f"{checkpoint['state']['root']}/trainer_state.json",
            )
        )
    sha256sums_sha256, bound_paths = _publish_sha256sums(output_root, bound_paths)
    summary_sha256, _ = sha256_regular_file(summary_path)
    complete = {
        "bound_paths": bound_paths,
        "max_steps": int(max_steps),
        "run_id": str(run_id),
        "schema_name": "fastwam-training-complete",
        "schema_version": 1,
        "sha256sums_sha256": sha256sums_sha256,
        "status": "PASS",
        "summary_sha256": summary_sha256,
        "world_size": int(world_size),
    }
    publish_exclusive_json(output_root / "TRAINING.COMPLETE", complete)
    return complete


def _load_rank_proofs(proof_dir: Path, pattern: str, *, expected: int) -> list[dict[str, Any]]:
    paths = sorted(proof_dir.glob(pattern), key=lambda path: os.fsencode(path.name))
    if len(paths) != expected:
        raise RuntimeError(f"expected exactly {expected} proofs matching {pattern}, got {len(paths)}")
    payloads = []
    for rank, path in enumerate(paths):
        if path.name != pattern.replace("*", f"{rank:05d}"):
            raise RuntimeError(f"rank proof filename mismatch at rank {rank}: {path.name}")
        payload, _, _ = read_canonical_json(path)
        if payload.get("rank") != rank or payload.get("world_size") != expected:
            raise RuntimeError(f"rank/world proof mismatch in {path}")
        payloads.append(payload)
    return payloads


def _summarize_n4_peak_memory(
    step_proofs: Mapping[int, list[dict[str, Any]]],
) -> dict[str, int | str]:
    """Validate per-rank memory evidence and return a conservative summary.

    Alibaba PAI can schedule RTX 4090 workers with different visible memory
    capacities.  Capacity is therefore a per-rank safety input, not part of
    the cross-rank device identity.  Each rank must report a stable
    ``(device_name, total_device_bytes)`` across both optimizer steps, and
    every proof is checked against the limits derived from that rank's own
    capacity.  The sealed run-level summary uses the smallest observed
    capacity so downstream consumers never infer a larger safety margin than
    the least-capable worker actually provided.
    """

    peak_allocated = 0
    peak_reserved = 0
    rank_identities: dict[int, tuple[str, int]] = {}
    device_names: set[str] = set()
    total_device_capacities: set[int] = set()
    expected_memory_fields = {
        "device_name",
        "effective_max_allocated_bytes",
        "effective_max_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "required_max_allocated_bytes",
        "required_max_reserved_bytes",
        "total_device_bytes",
    }
    for step, proofs in step_proofs.items():
        for proof in proofs:
            rank = int(proof.get("rank", -1))
            memory = proof.get("memory", {})
            if set(memory) != expected_memory_fields:
                raise RuntimeError(f"peak-memory proof fields mismatch: rank={rank}")
            if (
                memory.get("required_max_allocated_bytes")
                != N4_GATE_MAX_PEAK_ALLOCATED_BYTES
                or memory.get("required_max_reserved_bytes")
                != N4_GATE_MAX_PEAK_RESERVED_BYTES
            ):
                raise RuntimeError(f"peak-memory threshold mismatch: rank={rank}")
            raw_device_name = memory.get("device_name")
            device_name = raw_device_name.strip() if isinstance(raw_device_name, str) else ""
            total_device_bytes = int(memory.get("total_device_bytes", -1))
            expected_allocated_limit = min(
                N4_GATE_MAX_PEAK_ALLOCATED_BYTES,
                total_device_bytes * 90 // 100,
            )
            expected_reserved_limit = min(
                N4_GATE_MAX_PEAK_RESERVED_BYTES,
                total_device_bytes * 95 // 100,
            )
            if (
                not device_name
                or total_device_bytes <= 0
                or memory.get("effective_max_allocated_bytes")
                != expected_allocated_limit
                or memory.get("effective_max_reserved_bytes")
                != expected_reserved_limit
            ):
                raise RuntimeError(f"relative peak-memory threshold mismatch: rank={rank}")
            allocated = int(memory.get("peak_allocated_bytes", -1))
            reserved = int(memory.get("peak_reserved_bytes", -1))
            if allocated < 0 or reserved < 0:
                raise RuntimeError(f"missing peak-memory evidence in N=4 proof: rank={rank}")
            if allocated > expected_allocated_limit or reserved > expected_reserved_limit:
                raise RuntimeError(
                    f"N=4 proof exceeds memory gate: rank={rank} "
                    f"allocated={allocated} reserved={reserved}"
                )
            identity = (device_name, total_device_bytes)
            previous_identity = rank_identities.setdefault(rank, identity)
            if previous_identity != identity:
                raise RuntimeError(
                    "N=4 gate CUDA device identity changed between optimizer steps: "
                    f"rank={rank} previous={previous_identity} current={identity} step={step}"
                )
            device_names.add(device_name)
            total_device_capacities.add(total_device_bytes)
            peak_allocated = max(peak_allocated, allocated)
            peak_reserved = max(peak_reserved, reserved)
    if set(rank_identities) != set(range(N4_GATE_WORLD_SIZE)):
        raise RuntimeError(
            "N=4 gate memory evidence does not cover exactly all ranks: "
            f"observed={sorted(rank_identities)}"
        )
    if len(device_names) != 1:
        raise RuntimeError(
            f"N=4 gate requires the same non-empty CUDA device name on all 32 ranks: {device_names}"
        )
    minimum_total_device_bytes = min(total_device_capacities)
    conservative_allocated_limit = min(
        N4_GATE_MAX_PEAK_ALLOCATED_BYTES,
        minimum_total_device_bytes * 90 // 100,
    )
    conservative_reserved_limit = min(
        N4_GATE_MAX_PEAK_RESERVED_BYTES,
        minimum_total_device_bytes * 95 // 100,
    )
    if (
        peak_allocated > conservative_allocated_limit
        or peak_reserved > conservative_reserved_limit
    ):
        raise RuntimeError(
            "N=4 proofs exceed the conservative run-level memory gate: "
            f"allocated={peak_allocated}/{conservative_allocated_limit} "
            f"reserved={peak_reserved}/{conservative_reserved_limit}"
        )
    return {
        "device_name": next(iter(device_names)),
        "effective_max_allocated_bytes": conservative_allocated_limit,
        "effective_max_reserved_bytes": conservative_reserved_limit,
        "total_device_bytes": minimum_total_device_bytes,
        "max_allocated_bytes": peak_allocated,
        "max_reserved_bytes": peak_reserved,
        "required_max_allocated_bytes": N4_GATE_MAX_PEAK_ALLOCATED_BYTES,
        "required_max_reserved_bytes": N4_GATE_MAX_PEAK_RESERVED_BYTES,
    }


def finalize_n4_fullmodel_gate(
    output_root: str | Path,
    *,
    run_id: str,
    code_commit: str,
    image_reference: str,
    image_digest: str,
    input_bindings: Mapping[str, str],
) -> dict[str, Any]:
    output_root = resolved_unaliased_directory(output_root, label="N=4 gate output root")
    for name in ("manifest.json", "SHA256SUMS", "COMPLETE"):
        target = output_root / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to replace N=4 gate terminal artifact: {target}")
    code_commit = str(code_commit).lower()
    if len(code_commit) != 40 or any(character not in SHA256_HEX for character in code_commit):
        raise ValueError("N=4 gate requires a 40-hex code commit")
    image_digest = str(image_digest).lower()
    if not image_reference or not image_digest.startswith("sha256:"):
        raise ValueError("N=4 gate requires an exact image reference and OCI digest")
    require_sha256(image_digest.split(":", 1)[1], label="OCI image digest")
    expected_batch = {
        "global_train_batch_size": N4_GATE_GLOBAL_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": N4_GATE_GRADIENT_ACCUMULATION_STEPS,
        "local_micro_batch_size": N4_GATE_LOCAL_MICRO_BATCH_SIZE,
        "world_size": N4_GATE_WORLD_SIZE,
    }
    expected_shapes = {
        "action": [1, 4, 32, 8],
        "agent_gaussian": [1, 4, 13, 28, 40],
        "agent_geometry": [1, 4, 7],
        "agent_state": [1, 4, 18],
        "video": [1, 3, 9, 224, 320],
    }
    proof_dir = output_root / "gate-proofs"
    if proof_dir.is_symlink() or not proof_dir.is_dir():
        raise ValueError(f"N=4 gate proof directory is invalid: {proof_dir}")
    expected_proof_names = {
        *(f"step-{step:06d}-rank-{rank:05d}.json" for step in range(1, N4_GATE_TRAIN_STEPS + 1) for rank in range(N4_GATE_WORLD_SIZE)),
        *(f"save-state-rank-{rank:05d}.json" for rank in range(N4_GATE_WORLD_SIZE)),
        *(f"load-state-rank-{rank:05d}.json" for rank in range(N4_GATE_WORLD_SIZE)),
    }
    observed_proof_names = set()
    for path in proof_dir.iterdir():
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"N=4 gate proof root contains an aliased/special entry: {path}")
        observed_proof_names.add(path.name)
    if observed_proof_names != expected_proof_names:
        raise RuntimeError(
            "N=4 gate proof path set mismatch: "
            f"missing={sorted(expected_proof_names - observed_proof_names)[:12]} "
            f"unexpected={sorted(observed_proof_names - expected_proof_names)[:12]}"
        )
    step_proofs = {
        step: _load_rank_proofs(
            proof_dir, f"step-{step:06d}-rank-*.json", expected=N4_GATE_WORLD_SIZE
        )
        for step in range(1, N4_GATE_TRAIN_STEPS + 1)
    }
    save_proofs = _load_rank_proofs(
        proof_dir, "save-state-rank-*.json", expected=N4_GATE_WORLD_SIZE
    )
    load_proofs = _load_rank_proofs(
        proof_dir, "load-state-rank-*.json", expected=N4_GATE_WORLD_SIZE
    )
    expected_step_fields = {
        "agent_count",
        "batch_accounting",
        "gradients",
        "hub_token_policy",
        "losses",
        "memory",
        "num_hub_tokens",
        "phase",
        "process_nonce",
        "process_pid",
        "process_start_ticks",
        "rank",
        "sample_shapes",
        "schema_name",
        "schema_version",
        "step",
        "world_size",
    }
    for step, proofs in step_proofs.items():
        for proof in proofs:
            if (
                set(proof) != expected_step_fields
                or proof.get("schema_name") != "fastwam-n4-fullmodel-step-proof"
                or proof.get("schema_version") != 1
                or proof.get("phase") != "train_step"
                or proof.get("step") != step
                or proof.get("batch_accounting") != expected_batch
                or proof.get("sample_shapes") != expected_shapes
                or proof.get("agent_count") != 4
                or proof.get("num_hub_tokens") != 8
                or proof.get("hub_token_policy") != "ceil(hub_token_ratio*num_agents)"
            ):
                raise RuntimeError(f"N=4 step proof semantic mismatch: rank={proof.get('rank')} step={step}")
            losses = proof.get("losses", {})
            gradients = proof.get("gradients", {})
            if set(losses) != {"action", "total", "video"} or not all(
                np.isfinite(float(value)) for value in losses.values()
            ):
                raise RuntimeError(f"non-finite/incomplete losses in N=4 proof: rank={proof.get('rank')}")
            gradient_source = gradients.get("source")
            if gradient_source == "deepspeed_global_grad_norm":
                expected_gradient_fields = {"all_finite", "norm", "source"}
                source_valid = True
            elif gradient_source == "parameter_grad_scan":
                expected_gradient_fields = {
                    "all_finite",
                    "norm",
                    "source",
                    "tensor_count",
                }
                source_valid = int(gradients.get("tensor_count", 0)) > 0
            else:
                expected_gradient_fields = set()
                source_valid = False
            gradient_norm = float(gradients.get("norm", float("nan")))
            if (
                set(gradients) != expected_gradient_fields
                or gradients.get("all_finite") is not True
                or not source_valid
                or not np.isfinite(gradient_norm)
                or gradient_norm <= 0.0
            ):
                raise RuntimeError(f"non-finite/missing gradients in N=4 proof: rank={proof.get('rank')}")
    peak_memory = _summarize_n4_peak_memory(step_proofs)
    roundtrip_checks = {
        "global_step": True,
        "model": True,
        "optimizer": True,
        "rng": True,
        "rng_next_sample": True,
        "scheduler": True,
        "separate_process": True,
        "pre_load_was_distinct": True,
    }
    expected_save_fields = {
        "batch_accounting",
        "fingerprints",
        "next_rng_sample",
        "phase",
        "process_nonce",
        "process_pid",
        "process_start_ticks",
        "rank",
        "schema_name",
        "schema_version",
        "world_size",
    }
    expected_load_fields = {
        "batch_accounting",
        "checks",
        "fingerprints",
        "next_rng_sample",
        "phase",
        "pre_load_fingerprints",
        "process_nonce",
        "process_pid",
        "process_start_ticks",
        "rank",
        "schema_name",
        "schema_version",
        "world_size",
    }
    fingerprint_keys = {
        "global_step",
        "model",
        "model_probe",
        "optimizer",
        "optimizer_probe",
        "rng",
        "scheduler",
    }
    check_keys = {
        "global_step",
        "model",
        "optimizer",
        "pre_load_was_distinct",
        "rng",
        "rng_next_sample",
        "scheduler",
    }
    for rank, (saved, loaded) in enumerate(zip(save_proofs, load_proofs, strict=True)):
        if (
            set(saved) != expected_save_fields
            or set(loaded) != expected_load_fields
            or saved.get("schema_name") != "fastwam-n4-fullmodel-save-proof"
            or loaded.get("schema_name") != "fastwam-n4-fullmodel-load-proof"
            or saved.get("schema_version") != 1
            or loaded.get("schema_version") != 1
            or saved.get("phase") != "save_after_full_checkpoint"
            or loaded.get("phase") != "load_fresh_process"
            or saved.get("batch_accounting") != expected_batch
            or loaded.get("batch_accounting") != expected_batch
        ):
            raise RuntimeError(f"N=4 save/load proof schema mismatch at rank {rank}")
        checks = loaded.get("checks", {})
        saved_fingerprints = saved.get("fingerprints", {})
        loaded_fingerprints = loaded.get("fingerprints", {})
        pre_load_fingerprints = loaded.get("pre_load_fingerprints", {})
        if (
            set(checks) != check_keys
            or set(saved_fingerprints) != fingerprint_keys
            or set(loaded_fingerprints) != fingerprint_keys
            or set(pre_load_fingerprints) != fingerprint_keys
        ):
            raise RuntimeError(f"N=4 state fingerprint/check field mismatch at rank {rank}")
        if (
            saved_fingerprints["global_step"] != N4_GATE_TRAIN_STEPS
            or loaded_fingerprints["global_step"] != N4_GATE_TRAIN_STEPS
        ):
            raise RuntimeError(f"N=4 save/load proof did not bind global_step=2 at rank {rank}")
        for key in ("global_step", "model", "optimizer", "rng", "scheduler"):
            direct_match = loaded_fingerprints[key] == saved_fingerprints[key]
            roundtrip_checks[key] &= bool(checks.get(key)) and direct_match
        direct_rng_next = loaded.get("next_rng_sample") == saved.get("next_rng_sample")
        roundtrip_checks["rng_next_sample"] &= (
            bool(checks.get("rng_next_sample")) and direct_rng_next
        )
        direct_pre_load_distinct = any(
            pre_load_fingerprints[key] != saved_fingerprints[key]
            for key in ("global_step", "model", "optimizer", "rng", "scheduler")
        )
        roundtrip_checks["pre_load_was_distinct"] &= (
            bool(checks.get("pre_load_was_distinct")) and direct_pre_load_distinct
        )
        roundtrip_checks["separate_process"] &= (
            saved.get("process_nonce") != loaded.get("process_nonce")
            and (saved.get("process_pid"), saved.get("process_start_ticks"))
            != (loaded.get("process_pid"), loaded.get("process_start_ticks"))
        )
    if not all(roundtrip_checks.values()):
        raise RuntimeError(f"N=4 full-state roundtrip aggregation failed: {roundtrip_checks}")

    checkpoint = checkpoint_seal_descriptor(
        output_root, step=N4_GATE_TRAIN_STEPS, rehash_weights=True
    )
    expected_binding_keys = {
        "cpfs_bundle_manifest",
        "gaussian_cache_manifest",
        "gaussian_cache_selection",
        "gaussian_cache_source_identity",
        "official_checkpoint",
        "oss_bundle_manifest",
        "synthetic_zero2_gate",
        "stats",
        "training_environment_bundle",
        "vae",
    }
    if set(input_bindings) != expected_binding_keys:
        raise ValueError(
            "N=4 gate input binding key set mismatch: "
            f"missing={sorted(expected_binding_keys - set(input_bindings))} "
            f"unexpected={sorted(set(input_bindings) - expected_binding_keys)}"
        )
    normalized_bindings = {
        key: require_sha256(value, label=f"input binding {key}")
        for key, value in sorted(input_bindings.items())
    }
    reservation, reservation_sha256, _ = read_canonical_json(
        output_root / ".RUN_RESERVED"
    )
    reservation_identity_sha256 = require_sha256(
        reservation.get("identity_sha256", ""),
        label="N=4 gate reservation identity SHA-256",
    )
    reservation_identity_payload = dict(reservation)
    del reservation_identity_payload["identity_sha256"]
    if canonical_json_sha256(reservation_identity_payload) != reservation_identity_sha256:
        raise RuntimeError("N=4 gate reservation identity_sha256 does not match its payload")
    expected_reservation = {
        "bundle_manifest_sha256": None,
        "cache_manifest_sha256": normalized_bindings["gaussian_cache_manifest"],
        "cache_selection_sha256": normalized_bindings["gaussian_cache_selection"],
        "cache_source_identity_sha256": normalized_bindings[
            "gaussian_cache_source_identity"
        ],
        "checkpoint_sha256": normalized_bindings["official_checkpoint"],
        "code_commit": code_commit,
        "cpfs_bundle_manifest_sha256": normalized_bindings["cpfs_bundle_manifest"],
        "global_world_size": N4_GATE_WORLD_SIZE,
        "image_digest": image_digest,
        "image_digest_status": "resolved",
        "image_reference": str(image_reference),
        "n4_fullmodel_gate_complete_sha256": None,
        "nproc_per_node": 8,
        "num_machines": 4,
        "oss_bundle_manifest_sha256": normalized_bindings["oss_bundle_manifest"],
        "output_storage": "oss_experimental",
        "output_zero_checkpoint_smoke_sha256": normalized_bindings[
            "synthetic_zero2_gate"
        ],
        "run_id": str(run_id),
        "schema_version": 1,
        "stats_sha256": normalized_bindings["stats"],
        "task": "robofactory_multi_robot_vg1_hub1_gau1_224_1e-4",
        "training_env_bundle_manifest_sha256": normalized_bindings[
            "training_environment_bundle"
        ],
        "vae_sha256": normalized_bindings["vae"],
    }
    reservation_mismatches = {
        key: {"expected": expected, "observed": reservation.get(key)}
        for key, expected in expected_reservation.items()
        if reservation.get(key) != expected
    }
    if reservation_mismatches:
        raise RuntimeError(
            "N=4 gate reservation does not bind the proof/finalizer identity: "
            f"{reservation_mismatches}"
        )
    manifest = {
        "batch_accounting": expected_batch,
        "checkpoint": checkpoint,
        "code_commit": code_commit,
        "image_digest": image_digest,
        "image_reference": str(image_reference),
        "input_bindings": normalized_bindings,
        "peak_memory": peak_memory,
        "proof_counts": {
            "load_state": len(load_proofs),
            "save_state": len(save_proofs),
            "step_1": len(step_proofs[1]),
            "step_2": len(step_proofs[2]),
        },
        "published_at": datetime.now(timezone.utc).isoformat(),
        "reservation": {
            "identity_sha256": reservation_identity_sha256,
            "path": ".RUN_RESERVED",
            "sha256": reservation_sha256,
        },
        "roundtrip": roundtrip_checks,
        "run_id": str(run_id),
        "schema_name": "fastwam-n4-fullmodel-gate",
        "schema_version": 1,
        "status": "PASS",
        "train_steps": N4_GATE_TRAIN_STEPS,
        "world_size": N4_GATE_WORLD_SIZE,
        "zero_stage": 2,
    }
    manifest_path = output_root / "manifest.json"
    publish_exclusive_json(manifest_path, manifest)
    bound_paths = ["manifest.json", ".RUN_RESERVED", "config.save.yaml", "config.load.yaml"]
    bound_paths.extend(path.relative_to(output_root).as_posix() for path in sorted(proof_dir.glob("*.json")))
    bound_paths.extend(
        (
            checkpoint["weights"]["manifest"],
            checkpoint["weights"]["complete"],
            checkpoint["state"]["manifest"],
            f"{checkpoint['state']['root']}/trainer_state.json",
        )
    )
    sha256sums_sha256, bound_paths = _publish_sha256sums(output_root, bound_paths)
    manifest_sha256, _ = sha256_regular_file(manifest_path)
    complete = {
        "bound_paths": bound_paths,
        "manifest_sha256": manifest_sha256,
        "run_id": str(run_id),
        "schema_name": "fastwam-n4-fullmodel-gate-complete",
        "schema_version": 1,
        "sha256sums_sha256": sha256sums_sha256,
        "status": "PASS",
        "world_size": N4_GATE_WORLD_SIZE,
    }
    publish_exclusive_json(output_root / "COMPLETE", complete)
    return complete


def validate_terminal_sha256sums(
    output_root: str | Path,
    *,
    complete_name: str,
    expected_complete_schema: str,
) -> dict[str, Any]:
    output_root = resolved_unaliased_directory(output_root, label="formal output root")
    complete, complete_sha256, _ = read_canonical_json(output_root / complete_name)
    if (
        complete.get("schema_name") != expected_complete_schema
        or complete.get("schema_version") != 1
        or complete.get("status") != "PASS"
    ):
        raise ValueError(f"terminal COMPLETE schema/status mismatch: {output_root / complete_name}")
    sha_path = output_root / "SHA256SUMS"
    sha_digest, _ = sha256_regular_file(sha_path)
    if sha_digest != complete.get("sha256sums_sha256"):
        raise RuntimeError("terminal COMPLETE does not bind SHA256SUMS")
    records = []
    previous_record_key: bytes | None = None
    with sha_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(line) < 67 or line[64:66] != "  ":
                raise ValueError(f"invalid SHA256SUMS line: {line!r}")
            expected = require_sha256(line[:64], label="SHA256SUMS digest")
            relative = safe_relative_path(line[66:].rstrip("\n"))
            record_key = os.fsencode(relative.as_posix())
            if previous_record_key is not None and record_key <= previous_record_key:
                raise ValueError("SHA256SUMS paths must be unique and bytewise sorted")
            previous_record_key = record_key
            actual, _ = sha256_regular_file(output_root / relative)
            if actual != expected:
                raise RuntimeError(
                    f"terminal artifact SHA-256 mismatch: expected={expected} actual={actual} path={relative}"
                )
            records.append(relative.as_posix())
    if records != complete.get("bound_paths"):
        raise RuntimeError("terminal COMPLETE bound_paths do not exactly match SHA256SUMS")
    return {
        "bound_paths": records,
        "complete_sha256": complete_sha256,
        "sha256sums_sha256": sha_digest,
        "status": "PASS",
    }


def validate_n4_fullmodel_gate_binding(
    output_root: str | Path,
    *,
    allowed_prefix: str | Path,
    forbidden_output_root: str | Path,
    expected_complete_sha256: str,
    code_commit: str,
    image_reference: str,
    image_digest: str,
    input_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Validate that a PASS gate authorizes this exact main-run identity."""

    output_root = resolved_unaliased_directory(output_root, label="N=4 gate output root")
    allowed_prefix = resolved_unaliased_directory(
        allowed_prefix, label="N=4 gate allowed storage prefix"
    )
    if output_root == allowed_prefix:
        raise ValueError("N=4 gate output must be a child of its allowed storage prefix")
    try:
        output_root.relative_to(allowed_prefix)
    except ValueError as error:
        raise ValueError(
            f"N=4 gate output {output_root} is outside allowed prefix {allowed_prefix}"
        ) from error
    forbidden_supplied = Path(forbidden_output_root).expanduser()
    if not forbidden_supplied.is_absolute():
        raise ValueError(
            f"main training output root must be absolute: {forbidden_supplied}"
        )
    forbidden_resolved = forbidden_supplied.resolve(strict=False)
    if (
        output_root == forbidden_resolved
        or output_root in forbidden_resolved.parents
        or forbidden_resolved in output_root.parents
    ):
        raise ValueError(
            "N=4 gate output and main training output must be independent: "
            f"gate={output_root} main={forbidden_resolved}"
        )
    expected_complete_sha256 = require_sha256(
        expected_complete_sha256, label="N=4 gate COMPLETE SHA-256"
    )
    complete, actual_complete_sha256, _ = read_canonical_json(output_root / "COMPLETE")
    if actual_complete_sha256 != expected_complete_sha256:
        raise RuntimeError(
            "N=4 gate COMPLETE SHA-256 mismatch: "
            f"expected={expected_complete_sha256} actual={actual_complete_sha256}"
        )
    validate_terminal_sha256sums(
        output_root,
        complete_name="COMPLETE",
        expected_complete_schema="fastwam-n4-fullmodel-gate-complete",
    )
    manifest, manifest_sha256, _ = read_canonical_json(output_root / "manifest.json")
    if complete.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("N=4 gate COMPLETE does not bind manifest.json")
    expected_manifest_fields = {
        "batch_accounting",
        "checkpoint",
        "code_commit",
        "image_digest",
        "image_reference",
        "input_bindings",
        "peak_memory",
        "proof_counts",
        "published_at",
        "reservation",
        "roundtrip",
        "run_id",
        "schema_name",
        "schema_version",
        "status",
        "train_steps",
        "world_size",
        "zero_stage",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("N=4 gate manifest field set mismatch")
    expected_batch = {
        "global_train_batch_size": N4_GATE_GLOBAL_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": N4_GATE_GRADIENT_ACCUMULATION_STEPS,
        "local_micro_batch_size": N4_GATE_LOCAL_MICRO_BATCH_SIZE,
        "world_size": N4_GATE_WORLD_SIZE,
    }
    expected_roundtrip = {
        "global_step": True,
        "model": True,
        "optimizer": True,
        "pre_load_was_distinct": True,
        "rng": True,
        "rng_next_sample": True,
        "scheduler": True,
        "separate_process": True,
    }
    expected_proof_counts = {
        "load_state": N4_GATE_WORLD_SIZE,
        "save_state": N4_GATE_WORLD_SIZE,
        "step_1": N4_GATE_WORLD_SIZE,
        "step_2": N4_GATE_WORLD_SIZE,
    }
    reservation, reservation_sha256, _ = read_canonical_json(
        output_root / ".RUN_RESERVED"
    )
    reservation_identity_sha256 = require_sha256(
        reservation.get("identity_sha256", ""),
        label="N=4 gate reservation identity SHA-256",
    )
    reservation_identity_payload = dict(reservation)
    del reservation_identity_payload["identity_sha256"]
    if canonical_json_sha256(reservation_identity_payload) != reservation_identity_sha256:
        raise RuntimeError("N=4 gate reservation identity SHA-256 mismatch")
    expected_reservation_descriptor = {
        "identity_sha256": reservation_identity_sha256,
        "path": ".RUN_RESERVED",
        "sha256": reservation_sha256,
    }
    if (
        manifest.get("schema_name") != "fastwam-n4-fullmodel-gate"
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "PASS"
        or manifest.get("world_size") != N4_GATE_WORLD_SIZE
        or manifest.get("zero_stage") != 2
        or manifest.get("train_steps") != N4_GATE_TRAIN_STEPS
        or manifest.get("batch_accounting") != expected_batch
        or manifest.get("roundtrip") != expected_roundtrip
        or manifest.get("proof_counts") != expected_proof_counts
        or manifest.get("reservation") != expected_reservation_descriptor
    ):
        raise RuntimeError("N=4 gate manifest does not satisfy the formal PASS contract")
    normalized_bindings = {
        key: require_sha256(value, label=f"current input binding {key}")
        for key, value in sorted(input_bindings.items())
    }
    if manifest.get("input_bindings") != normalized_bindings:
        raise RuntimeError(
            "N=4 gate input bindings do not match the proposed main run: "
            f"gate={manifest.get('input_bindings')} current={normalized_bindings}"
        )
    if (
        manifest.get("code_commit") != str(code_commit).lower()
        or manifest.get("image_reference") != str(image_reference)
        or manifest.get("image_digest") != str(image_digest).lower()
    ):
        raise RuntimeError(
            "N=4 gate code/image identity does not match the proposed main run"
        )
    # The small outer terminal files prove what was observed when the gate was
    # finalized, but they cannot make a mutable OSS/FUSE checkpoint immutable.
    # Re-read and hash the actual full weights plus every ZeRO state shard at
    # authorization time, then require the resulting descriptor to be exactly
    # the one sealed in the gate manifest.  A deleted or modified large file
    # must therefore invalidate the gate before the main run starts.
    observed_checkpoint = checkpoint_seal_descriptor(
        output_root,
        step=N4_GATE_TRAIN_STEPS,
        rehash_weights=True,
    )
    if manifest.get("checkpoint") != observed_checkpoint:
        raise RuntimeError(
            "N=4 gate checkpoint changed after finalization: "
            f"sealed={manifest.get('checkpoint')} observed={observed_checkpoint}"
        )
    return {
        "complete_sha256": actual_complete_sha256,
        "manifest_sha256": manifest_sha256,
        "run_id": manifest["run_id"],
        "status": "PASS",
    }
