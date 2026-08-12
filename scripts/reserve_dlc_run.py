#!/usr/bin/env python3
"""Reserve or validate one immutable formal DLC output directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path, PurePosixPath

from state_tree_manifest import validate_state_tree_manifest

SCHEMA_VERSION = 1
ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT = "action_only_n2_1x8_v1"
ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT_VERSION = 1
ACTION_ONLY_N2_1X8_RESERVATION_SCHEMA_VERSION = 2
ACTION_ONLY_N2_RUN_PROFILES = frozenset({"paid_gate_1step", "formal_1k"})
ACTION_ONLY_NATIVE_AGENTS_1X8_TERMINAL_CONTRACT = (
    "action_only_native_agents_1x8_v1"
)
ACTION_ONLY_NATIVE_AGENTS_1X8_TERMINAL_CONTRACT_VERSION = 1
ACTION_ONLY_NATIVE_AGENTS_1X8_RESERVATION_SCHEMA_VERSION = 1
ACTION_ONLY_NATIVE_AGENTS_RUN_PROFILES = frozenset(
    {"paid_gate_1step", "formal_1k"}
)
ACTION_ONLY_NATIVE_AGENT_COUNTS = frozenset({2, 3, 4})
MARKER_NAME = ".RUN_RESERVED"
RESUME_MARKER_PREFIX = ".RESUME_VALIDATED."
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def _optional_sha256(value: str) -> str | None:
    if not value:
        return None
    normalized = value.lower()
    if not HEX_64.fullmatch(normalized):
        raise ValueError(f"expected a 64-character SHA-256, got {value!r}")
    return normalized


def _required_sha256(value: str, *, label: str) -> str:
    normalized = _optional_sha256(value)
    if normalized is None:
        raise ValueError(f"{label} is required")
    return normalized


def _required_git_object_id(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if not HEX_40.fullmatch(normalized):
        raise ValueError(f"{label} must be a lowercase 40-character Git object ID")
    return normalized


def _safe_relative_path(value: str, *, label: str) -> str:
    if not value or any(character in value for character in ("\x00", "\n", "\r")):
        raise ValueError(f"{label} must be a non-empty safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a normalized relative path: {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError(f"{label} must be a normalized relative path: {value!r}")
    return normalized


def _safe_oss_path(value: str, *, label: str) -> str:
    if not value or any(character in value for character in ("\x00", "\n", "\r")):
        raise ValueError(f"{label} must be a non-empty absolute OSS path")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or len(path.parts) < 3
        or path.parts[1] != "oss-chengjuntao"
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise ValueError(
            f"{label} must be a normalized child of /oss-chengjuntao: {value!r}"
        )
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError(f"{label} must be normalized: {value!r}")
    return normalized


def _native_agents_identity(
    args: argparse.Namespace,
    *,
    code_commit: str,
    global_world_size: int,
) -> dict[str, object]:
    contract = ACTION_ONLY_NATIVE_AGENTS_1X8_TERMINAL_CONTRACT
    if (args.num_machines, args.nproc_per_node, global_world_size) != (1, 8, 8):
        raise ValueError(f"{contract} requires exactly 1 machine x 8 processes")
    if args.output_storage != "oss":
        raise ValueError(f"{contract} requires --output-storage=oss")
    if args.image_digest_status != "reference_only" or args.image_digest:
        raise ValueError(
            f"{contract} requires --image-digest-status=reference_only and no digest"
        )
    if str(args.artifact_integrity_mode).strip() != "metadata_no_hash":
        raise ValueError(
            f"{contract} requires --artifact-integrity-mode=metadata_no_hash"
        )

    forbidden_digest_bindings = {
        name: getattr(args, name)
        for name in (
            "bundle_manifest_sha256",
            "cpfs_bundle_manifest_sha256",
            "oss_bundle_manifest_sha256",
            "cache_manifest_sha256",
            "cache_selection_sha256",
            "cache_source_identity_sha256",
            "checkpoint_sha256",
            "vae_sha256",
            "stats_sha256",
            "erdma_bootstrap_sha256",
            "erdma_bundle_sha256",
            "erdma_source_manifest_sha256",
            "erdma_env_sha256",
            "training_env_bundle_manifest_sha256",
            "pyproject_sha256",
            "output_zero_checkpoint_smoke_sha256",
            "n4_fullmodel_gate_complete_sha256",
            "request_sha256",
            "init_checkpoint_sha256",
            "task_scope_receipt_sha256",
            "resume_state_manifest_sha256",
            "resume_trainer_state_sha256",
        )
        if str(getattr(args, name)).strip()
    }
    if forbidden_digest_bindings:
        raise ValueError(
            f"{contract} forbids digest/checksum bindings: "
            f"{sorted(forbidden_digest_bindings)}"
        )
    legacy_bindings = {
        name: getattr(args, name)
        for name in (
            "effective_patched_tree",
            "task_scope_receipt",
            "resume_state_dir",
            "resume_state_manifest",
        )
        if str(getattr(args, name)).strip()
    }
    if legacy_bindings:
        raise ValueError(
            f"{contract} forbids legacy receipt/resume bindings: "
            f"{sorted(legacy_bindings)}"
        )

    run_profile = str(args.run_profile).strip()
    if run_profile not in ACTION_ONLY_NATIVE_AGENTS_RUN_PROFILES:
        raise ValueError(
            f"{contract} requires --run-profile in "
            f"{sorted(ACTION_ONLY_NATIVE_AGENTS_RUN_PROFILES)}, got {run_profile!r}"
        )
    scalar_contract = {
        "checkpoint_state_kind": (
            str(args.checkpoint_state_kind).strip(),
            "full",
        ),
        "trainable_scope": (str(args.trainable_scope).strip(), "action"),
        "training_mode": (str(args.training_mode).strip(), "action_only_cache"),
    }
    mismatches = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in scalar_contract.items()
        if observed != expected
    }
    if mismatches:
        raise ValueError(f"native-agent reservation treatment mismatch: {mismatches}")

    required_agent_count = int(args.required_agent_count)
    if required_agent_count not in ACTION_ONLY_NATIVE_AGENT_COUNTS:
        raise ValueError(
            f"{contract} requires --required-agent-count in "
            f"{sorted(ACTION_ONLY_NATIVE_AGENT_COUNTS)}, got {required_agent_count}"
        )
    required_tasks = tuple(str(task).strip() for task in args.required_task)
    if not required_tasks:
        raise ValueError(f"{contract} requires at least one --required-task")
    if len(set(required_tasks)) != len(required_tasks):
        raise ValueError("--required-task values must be unique")
    invalid_tasks = [task for task in required_tasks if not SAFE_RUN_ID.fullmatch(task)]
    if invalid_tasks:
        raise ValueError(f"invalid --required-task values: {invalid_tasks}")

    experiment_id = str(args.experiment_id).strip()
    if not SAFE_RUN_ID.fullmatch(experiment_id):
        raise ValueError("--experiment-id does not match the launcher safe-ID contract")
    if not SAFE_RUN_ID.fullmatch(args.task):
        raise ValueError("--task does not match the launcher safe-ID contract")

    return {
        "agent_cardinality_mode": "native",
        "artifact_integrity_mode": "metadata_no_hash",
        "checkpoint_state_kind": "full",
        "code_commit": code_commit,
        "experiment_id": experiment_id,
        "global_world_size": global_world_size,
        "image_reference": args.image_reference,
        "init_checkpoint_path": _safe_oss_path(
            str(args.init_checkpoint_path).strip(),
            label="initialization checkpoint path",
        ),
        "masked_agent_set": False,
        "nproc_per_node": args.nproc_per_node,
        "num_machines": args.num_machines,
        "output_storage": "oss",
        "required_agent_count": required_agent_count,
        "required_tasks": list(required_tasks),
        "run_id": args.run_id,
        "run_profile": run_profile,
        "schema_name": "fastwam-action-only-native-agents-1x8-reservation",
        "schema_version": ACTION_ONLY_NATIVE_AGENTS_1X8_RESERVATION_SCHEMA_VERSION,
        "source_snapshot_path": _safe_oss_path(
            str(args.source_snapshot_path).strip(),
            label="source snapshot path",
        ),
        "task": args.task,
        "trainable_scope": "action",
        "training_mode": "action_only_cache",
        "training_terminal_contract": contract,
        "training_terminal_contract_version": (
            ACTION_ONLY_NATIVE_AGENTS_1X8_TERMINAL_CONTRACT_VERSION
        ),
    }


def _canonical_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_output(output: Path, allowed_prefix: Path, source_root: Path, run_id: str) -> Path:
    if not output.is_absolute():
        raise ValueError(f"formal output_dir must be absolute: {output}")
    output = output.resolve(strict=False)
    allowed_prefix = allowed_prefix.resolve(strict=False)
    source_root = source_root.resolve(strict=True)
    if output == allowed_prefix:
        raise ValueError("formal output_dir must be a child of the allowed storage prefix")
    try:
        output.relative_to(allowed_prefix)
    except ValueError as error:
        raise ValueError(
            f"formal output_dir {output} is outside allowed storage prefix {allowed_prefix}"
        ) from error
    try:
        output.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError(f"formal output_dir must not be inside source tree {source_root}")
    if output.name != run_id:
        raise ValueError(
            f"formal output_dir basename must equal RUN_ID ({run_id!r}), got {output.name!r}"
        )
    return output


def _identity(args: argparse.Namespace) -> dict[str, object]:
    if not SAFE_RUN_ID.fullmatch(args.run_id):
        raise ValueError("RUN_ID does not match the launcher safe-ID contract")
    code_commit = args.code_commit.lower()
    if not HEX_40.fullmatch(code_commit):
        raise ValueError("code commit must be a lowercase 40-character Git object ID")
    if args.num_machines < 1 or args.nproc_per_node < 1:
        raise ValueError("num machines and processes per node must be positive")
    global_world_size = args.num_machines * args.nproc_per_node
    if args.expected_global_world_size != global_world_size:
        raise ValueError(
            "topology mismatch: "
            f"{args.num_machines}x{args.nproc_per_node}={global_world_size}, "
            f"expected {args.expected_global_world_size}"
        )
    if not args.image_reference.strip():
        raise ValueError("DLC image reference must not be empty")
    terminal_contract = str(args.training_terminal_contract).strip()
    if terminal_contract == ACTION_ONLY_NATIVE_AGENTS_1X8_TERMINAL_CONTRACT:
        return _native_agents_identity(
            args,
            code_commit=code_commit,
            global_world_size=global_world_size,
        )
    image_digest = args.image_digest.lower()
    if args.image_digest_status == "resolved":
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest):
            raise ValueError("resolved DLC image digest must use sha256:<64 lowercase hex>")
    elif args.image_digest_status == "unresolved_mutable_tag":
        if image_digest:
            raise ValueError(
                "unresolved mutable image status must not carry a claimed digest"
            )
    else:
        raise ValueError(
            "image-digest-status=reference_only is restricted to "
            f"{ACTION_ONLY_NATIVE_AGENTS_1X8_TERMINAL_CONTRACT}"
        )
    payload: dict[str, object] = {
        "bundle_manifest_sha256": _optional_sha256(args.bundle_manifest_sha256),
        "cpfs_bundle_manifest_sha256": _optional_sha256(args.cpfs_bundle_manifest_sha256),
        "cache_manifest_sha256": _optional_sha256(args.cache_manifest_sha256),
        "cache_selection_sha256": _optional_sha256(args.cache_selection_sha256),
        "cache_source_identity_sha256": _optional_sha256(args.cache_source_identity_sha256),
        "checkpoint_sha256": _optional_sha256(args.checkpoint_sha256),
        "code_commit": code_commit,
        "erdma_bootstrap_sha256": _optional_sha256(args.erdma_bootstrap_sha256),
        "erdma_bundle_sha256": _optional_sha256(args.erdma_bundle_sha256),
        "erdma_env_sha256": _optional_sha256(args.erdma_env_sha256),
        "erdma_source_manifest_sha256": _optional_sha256(
            args.erdma_source_manifest_sha256
        ),
        "global_world_size": global_world_size,
        "nproc_per_node": args.nproc_per_node,
        "num_machines": args.num_machines,
        "image_digest": image_digest or None,
        "image_digest_status": args.image_digest_status,
        "image_reference": args.image_reference,
        "n4_fullmodel_gate_complete_sha256": _optional_sha256(
            args.n4_fullmodel_gate_complete_sha256
        ),
        "oss_bundle_manifest_sha256": _optional_sha256(args.oss_bundle_manifest_sha256),
        "output_storage": args.output_storage,
        "output_zero_checkpoint_smoke_sha256": _optional_sha256(
            args.output_zero_checkpoint_smoke_sha256
        ),
        "pyproject_sha256": _required_sha256(
            args.pyproject_sha256,
            label="pyproject SHA-256",
        ),
        "run_id": args.run_id,
        "schema_version": SCHEMA_VERSION,
        "task": args.task,
        "stats_sha256": _optional_sha256(args.stats_sha256),
        "training_env_bundle_manifest_sha256": _optional_sha256(
            args.training_env_bundle_manifest_sha256
        ),
        "vae_sha256": _optional_sha256(args.vae_sha256),
    }
    n2_extension_values = {
        "checkpoint_state_kind": str(args.checkpoint_state_kind).strip(),
        "effective_patched_tree": str(args.effective_patched_tree).strip(),
        "init_checkpoint_sha256": str(args.init_checkpoint_sha256).strip(),
        "request_sha256": str(args.request_sha256).strip(),
        "run_profile": str(args.run_profile).strip(),
        "task_scope_receipt": str(args.task_scope_receipt).strip(),
        "task_scope_receipt_sha256": str(args.task_scope_receipt_sha256).strip(),
        "trainable_scope": str(args.trainable_scope).strip(),
        "training_mode": str(args.training_mode).strip(),
    }
    if not terminal_contract:
        native_extension_values = {
            "artifact_integrity_mode": str(args.artifact_integrity_mode).strip(),
            "experiment_id": str(args.experiment_id).strip(),
            "init_checkpoint_path": str(args.init_checkpoint_path).strip(),
            "required_agent_count": args.required_agent_count,
            "required_task": args.required_task,
            "source_snapshot_path": str(args.source_snapshot_path).strip(),
        }
        provided = sorted(
            key
            for key, value in {**n2_extension_values, **native_extension_values}.items()
            if value
        )
        if provided:
            raise ValueError(
                "terminal reservation fields require --training-terminal-contract: "
                f"{provided}"
            )
    else:
        if terminal_contract != ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT:
            raise ValueError(
                f"unsupported training terminal contract: {terminal_contract!r}"
            )
        if (args.num_machines, args.nproc_per_node, global_world_size) != (1, 8, 8):
            raise ValueError(
                "action_only_n2_1x8_v1 requires exactly 1 machine x 8 processes"
            )
        if payload["n4_fullmodel_gate_complete_sha256"] is not None:
            raise ValueError(
                "action_only_n2_1x8_v1 forbids an N=4 full-model gate binding"
            )
        run_profile = n2_extension_values["run_profile"]
        if run_profile not in ACTION_ONLY_N2_RUN_PROFILES:
            raise ValueError(
                "action_only_n2_1x8_v1 requires --run-profile in "
                f"{sorted(ACTION_ONLY_N2_RUN_PROFILES)}, got {run_profile!r}"
            )
        scalar_contract = {
            "checkpoint_state_kind": (
                n2_extension_values["checkpoint_state_kind"],
                "sparse_delta",
            ),
            "trainable_scope": (n2_extension_values["trainable_scope"], "action"),
            "training_mode": (
                n2_extension_values["training_mode"],
                "action_only_cache",
            ),
        }
        mismatches = {
            key: {"observed": observed, "expected": expected}
            for key, (observed, expected) in scalar_contract.items()
            if observed != expected
        }
        if mismatches:
            raise ValueError(f"N=2 reservation treatment mismatch: {mismatches}")
        init_checkpoint_sha256 = _required_sha256(
            n2_extension_values["init_checkpoint_sha256"],
            label="initialization checkpoint SHA-256",
        )
        if payload["checkpoint_sha256"] not in {None, init_checkpoint_sha256}:
            raise ValueError(
                "--checkpoint-sha256 must be empty or equal "
                "--init-checkpoint-sha256 for the N=2 terminal contract"
            )
        payload.update(
            {
                "base_code_commit": code_commit,
                "checkpoint_state_kind": "sparse_delta",
                "effective_patched_tree": _required_git_object_id(
                    n2_extension_values["effective_patched_tree"],
                    label="effective patched tree",
                ),
                "formal_n4_fullmodel_gate": False,
                "init_checkpoint_sha256": init_checkpoint_sha256,
                "request_sha256": _required_sha256(
                    n2_extension_values["request_sha256"],
                    label="external submission request SHA-256",
                ),
                "run_profile": run_profile,
                "schema_version": ACTION_ONLY_N2_1X8_RESERVATION_SCHEMA_VERSION,
                "task_scope_receipt": _safe_relative_path(
                    n2_extension_values["task_scope_receipt"],
                    label="task-scope receipt",
                ),
                "task_scope_receipt_sha256": _required_sha256(
                    n2_extension_values["task_scope_receipt_sha256"],
                    label="task-scope receipt SHA-256",
                ),
                "trainable_scope": "action",
                "training_mode": "action_only_cache",
                "training_terminal_contract": terminal_contract,
                "training_terminal_contract_version": (
                    ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT_VERSION
                ),
            }
        )
    payload["identity_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def _read_exact_marker(marker: Path, expected: dict[str, object]) -> None:
    if marker.is_symlink() or not marker.is_file():
        raise RuntimeError(f"reservation marker is not a regular non-symlink file: {marker}")
    try:
        observed = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot parse reservation marker {marker}: {error}") from error
    if observed != expected:
        raise RuntimeError(
            "reservation identity mismatch: "
            f"expected={json.dumps(expected, sort_keys=True)} "
            f"observed={json.dumps(observed, sort_keys=True)}"
        )


def reserve_owner(output: Path, expected: dict[str, object]) -> None:
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError(
            f"formal output parent must already exist as a non-symlink directory: {output.parent}"
        )
    probe = output.parent / f".fastwam-output-probe.{expected['run_id']}.{os.getpid()}"
    probe_owned = False
    try:
        os.mkdir(probe, 0o700)
        probe_owned = True
        partial = probe / "payload.part"
        ready = probe / "payload.ready"
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            probe_payload = b"FASTWAM_OUTPUT_CAPABILITY_PROBE\n" * 32768
            written = 0
            while written < len(probe_payload):
                written += os.write(descriptor, probe_payload[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(partial, ready)
        if expected.get("artifact_integrity_mode") == "metadata_no_hash":
            probe_matches = ready.read_bytes() == probe_payload
        else:
            probe_matches = (
                hashlib.sha256(ready.read_bytes()).digest()
                == hashlib.sha256(probe_payload).digest()
            )
        if not probe_matches:
            raise RuntimeError("formal output filesystem capability probe readback mismatch")
        _fsync_directory(probe)
        ready.unlink()
        probe.rmdir()
        _fsync_directory(output.parent)
    except Exception:
        # Clean only this process-owned probe; never touch an existing run path.
        if probe_owned:
            for child in (probe / "payload.part", probe / "payload.ready"):
                try:
                    child.unlink()
                except FileNotFoundError:
                    pass
            try:
                probe.rmdir()
            except FileNotFoundError:
                pass
        raise
    try:
        os.mkdir(output, 0o750)
    except FileExistsError as error:
        raise FileExistsError(
            f"formal output_dir already exists; run directories are never reused: {output}"
        ) from error
    marker = output / MARKER_NAME
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        payload = _canonical_bytes(expected)
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(output)
    _fsync_directory(output.parent)
    _read_exact_marker(marker, expected)


def wait_for_reservation(output: Path, expected: dict[str, object], timeout: float) -> None:
    if timeout <= 0:
        raise ValueError("reservation timeout must be positive")
    marker = output / MARKER_NAME
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists() or marker.is_symlink():
            _read_exact_marker(marker, expected)
            return
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for rank-0 reservation marker: {marker}")


def validate_existing_reservation(output: Path, expected: dict[str, object]) -> None:
    """Validate an existing run without granting permission to reuse another run."""

    if output.is_symlink() or not output.is_dir():
        raise RuntimeError(
            f"existing formal output is not a non-symlink directory: {output}"
        )
    _read_exact_marker(output / MARKER_NAME, expected)


def validate_resume_state(
    output: Path,
    state_dir_value: str,
    state_manifest_value: str,
    state_manifest_sha256: str,
    trainer_state_sha256: str,
    *,
    verify_tree: bool,
) -> tuple[dict[str, object], Path] | None:
    if not state_dir_value:
        if state_manifest_value or state_manifest_sha256 or trainer_state_sha256:
            raise ValueError(
                "resume state identity was provided without a resume state directory"
            )
        return None
    expected_trainer_sha256 = _optional_sha256(trainer_state_sha256)
    expected_manifest_sha256 = _optional_sha256(state_manifest_sha256)
    if expected_trainer_sha256 is None or expected_manifest_sha256 is None:
        raise ValueError(
            "full-state resume requires exact manifest and trainer_state.json SHA-256 values"
        )
    if not state_manifest_value:
        raise ValueError("full-state resume requires a sealed state-tree manifest")
    state_dir = Path(state_dir_value).expanduser()
    if not state_dir.is_absolute():
        raise ValueError(f"full-state resume directory must be absolute: {state_dir}")
    state_dir_absolute = Path(os.path.abspath(state_dir))
    state_dir = state_dir.resolve(strict=False)
    if state_dir_absolute != state_dir:
        raise ValueError(
            f"full-state resume directory must not traverse symlinks or aliases: {state_dir_value}"
        )
    try:
        state_relative = state_dir.relative_to(output)
    except ValueError as error:
        raise ValueError(
            "full-state resume directory must be inside the same reserved output: "
            f"state={state_dir} output={output}"
        ) from error
    if state_dir == output:
        raise ValueError("full-state resume directory must be a child of the run output")
    manifest = Path(state_manifest_value).expanduser()
    if not manifest.is_absolute():
        raise ValueError(f"resume state manifest must be absolute: {manifest}")
    manifest_absolute = Path(os.path.abspath(manifest))
    manifest = manifest.resolve(strict=False)
    if manifest_absolute != manifest:
        raise ValueError(
            f"resume state manifest must not traverse symlinks or aliases: {state_manifest_value}"
        )
    try:
        manifest_relative = manifest.relative_to(output)
    except ValueError as error:
        raise ValueError(
            f"resume state manifest must be inside the same reserved output: {manifest}"
        ) from error
    try:
        manifest.relative_to(state_dir)
    except ValueError:
        pass
    else:
        raise ValueError(
            "resume state manifest must be outside the sealed state directory"
        )

    resume_payload: dict[str, object] = {
        "reservation_identity_sha256": None,
        "resume_state_manifest": manifest_relative.as_posix(),
        "resume_state_manifest_sha256": expected_manifest_sha256,
        "resume_state_path": state_relative.as_posix(),
        "resume_trainer_state_sha256": expected_trainer_sha256,
        "schema_version": 1,
    }
    resume_identity_sha256 = hashlib.sha256(_canonical_bytes(resume_payload)).hexdigest()
    resume_payload["resume_identity_sha256"] = resume_identity_sha256
    resume_marker = output / f"{RESUME_MARKER_PREFIX}{resume_identity_sha256}.json"
    if not verify_tree:
        return resume_payload, resume_marker
    if state_dir.is_symlink() or not state_dir.is_dir():
        raise RuntimeError(
            f"full-state resume path is not a non-symlink directory: {state_dir}"
        )
    if manifest.is_symlink() or not manifest.is_file():
        raise RuntimeError(
            f"resume state manifest is not a regular non-symlink file: {manifest}"
        )
    summary = validate_state_tree_manifest(
        state_dir,
        manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_role="accelerate_zero2_full_state",
    )
    trainer_state = state_dir / "trainer_state.json"
    if trainer_state.is_symlink() or not trainer_state.is_file():
        raise RuntimeError(
            "full-state resume requires a regular non-symlink trainer_state.json: "
            f"{trainer_state}"
        )
    trainer_records = [
        record for record in summary["files"] if record["path"] == "trainer_state.json"
    ]
    if len(trainer_records) != 1:
        raise RuntimeError("sealed full-state tree must contain trainer_state.json exactly once")
    actual_sha256 = trainer_records[0]["sha256"]
    if actual_sha256 != expected_trainer_sha256:
        raise RuntimeError(
            "full-state trainer_state.json SHA-256 mismatch: "
            f"expected={expected_trainer_sha256} actual={actual_sha256} path={trainer_state}"
        )
    return resume_payload, resume_marker


def publish_resume_validation(
    marker: Path,
    payload: dict[str, object],
    reservation_identity_sha256: str,
) -> None:
    payload = dict(payload)
    payload["reservation_identity_sha256"] = reservation_identity_sha256
    # Re-key after binding the resume evidence to the exact run reservation.
    payload_without_identity = dict(payload)
    payload_without_identity.pop("resume_identity_sha256", None)
    identity = hashlib.sha256(_canonical_bytes(payload_without_identity)).hexdigest()
    payload["resume_identity_sha256"] = identity
    marker = marker.with_name(f"{RESUME_MARKER_PREFIX}{identity}.json")
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    except FileExistsError:
        _read_exact_marker(marker, payload)
        return
    try:
        encoded = _canonical_bytes(payload)
        written = 0
        while written < len(encoded):
            written += os.write(descriptor, encoded[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(marker.parent)
    _read_exact_marker(marker, payload)


def wait_for_resume_validation(
    output: Path,
    payload: dict[str, object],
    reservation_identity_sha256: str,
    timeout: float,
) -> None:
    payload = dict(payload)
    payload["reservation_identity_sha256"] = reservation_identity_sha256
    payload_without_identity = dict(payload)
    payload_without_identity.pop("resume_identity_sha256", None)
    identity = hashlib.sha256(_canonical_bytes(payload_without_identity)).hexdigest()
    payload["resume_identity_sha256"] = identity
    marker = output / f"{RESUME_MARKER_PREFIX}{identity}.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists() or marker.is_symlink():
            _read_exact_marker(marker, payload)
            return
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for full-state validation marker: {marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "validate",
            "owner",
            "wait",
            "validate-existing",
            "wait-existing",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allowed-prefix", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-machines", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--expected-global-world-size", type=int, required=True)
    parser.add_argument("--bundle-manifest-sha256", default="")
    parser.add_argument("--cpfs-bundle-manifest-sha256", default="")
    parser.add_argument("--oss-bundle-manifest-sha256", default="")
    parser.add_argument("--cache-manifest-sha256", default="")
    parser.add_argument("--cache-selection-sha256", default="")
    parser.add_argument("--cache-source-identity-sha256", default="")
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--vae-sha256", default="")
    parser.add_argument("--stats-sha256", default="")
    parser.add_argument("--erdma-bootstrap-sha256", default="")
    parser.add_argument("--erdma-bundle-sha256", default="")
    parser.add_argument("--erdma-source-manifest-sha256", default="")
    parser.add_argument("--erdma-env-sha256", default="")
    parser.add_argument("--training-env-bundle-manifest-sha256", default="")
    parser.add_argument("--image-reference", required=True)
    parser.add_argument(
        "--image-digest-status",
        choices=("resolved", "unresolved_mutable_tag", "reference_only"),
        required=True,
    )
    parser.add_argument("--image-digest", default="")
    parser.add_argument("--pyproject-sha256", default="")
    parser.add_argument(
        "--output-storage",
        choices=("cpfs", "oss_experimental", "oss"),
        required=True,
    )
    parser.add_argument("--output-zero-checkpoint-smoke-sha256", default="")
    parser.add_argument("--n4-fullmodel-gate-complete-sha256", default="")
    parser.add_argument("--training-terminal-contract", default="")
    parser.add_argument("--effective-patched-tree", default="")
    parser.add_argument("--request-sha256", default="")
    parser.add_argument("--run-profile", default="")
    parser.add_argument("--init-checkpoint-sha256", default="")
    parser.add_argument("--task-scope-receipt", default="")
    parser.add_argument("--task-scope-receipt-sha256", default="")
    parser.add_argument("--checkpoint-state-kind", default="")
    parser.add_argument("--trainable-scope", default="")
    parser.add_argument("--training-mode", default="")
    parser.add_argument("--artifact-integrity-mode", default="")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--required-agent-count", type=int, default=0)
    parser.add_argument("--required-task", action="append", default=[])
    parser.add_argument("--source-snapshot-path", default="")
    parser.add_argument("--init-checkpoint-path", default="")
    parser.add_argument("--resume-state-dir", default="")
    parser.add_argument("--resume-state-manifest", default="")
    parser.add_argument("--resume-state-manifest-sha256", default="")
    parser.add_argument("--resume-trainer-state-sha256", default="")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--resume-timeout", type=float, default=21600.0)
    args = parser.parse_args()
    try:
        if args.timeout <= 0:
            raise ValueError("reservation timeout must be positive")
        if args.resume_timeout <= 0:
            raise ValueError("resume validation timeout must be positive")
        output = _validate_output(
            args.output_dir,
            args.allowed_prefix,
            args.source_root,
            args.run_id,
        )
        expected = _identity(args)
        resume_validation = validate_resume_state(
            output,
            args.resume_state_dir,
            args.resume_state_manifest,
            args.resume_state_manifest_sha256,
            args.resume_trainer_state_sha256,
            verify_tree=False,
        )
        if args.mode == "owner":
            reserve_owner(output, expected)
        elif args.mode == "wait":
            wait_for_reservation(output, expected, args.timeout)
        elif args.mode == "validate-existing":
            validate_existing_reservation(output, expected)
            verified_resume = validate_resume_state(
                output,
                args.resume_state_dir,
                args.resume_state_manifest,
                args.resume_state_manifest_sha256,
                args.resume_trainer_state_sha256,
                verify_tree=True,
            )
            if verified_resume is None:
                raise ValueError("validate-existing requires a full-state resume identity")
            resume_payload, resume_marker = verified_resume
            publish_resume_validation(
                resume_marker,
                resume_payload,
                str(expected["identity_sha256"]),
            )
        elif args.mode == "wait-existing":
            validate_existing_reservation(output, expected)
            if resume_validation is None:
                raise ValueError("wait-existing requires a full-state resume identity")
            resume_payload, _ = resume_validation
            wait_for_resume_validation(
                output,
                resume_payload,
                str(expected["identity_sha256"]),
                args.resume_timeout,
            )
        print(json.dumps(expected, sort_keys=True, separators=(",", ":")))
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
