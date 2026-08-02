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
from pathlib import Path

from state_tree_manifest import validate_state_tree_manifest


SCHEMA_VERSION = 1
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
        raise ValueError("formal output_dir must be a child of the allowed CPFS prefix")
    try:
        output.relative_to(allowed_prefix)
    except ValueError as error:
        raise ValueError(
            f"formal output_dir {output} is outside allowed CPFS prefix {allowed_prefix}"
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
    image_digest = args.image_digest.lower()
    if args.image_digest_status == "resolved":
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest):
            raise ValueError("resolved DLC image digest must use sha256:<64 lowercase hex>")
    elif image_digest:
        raise ValueError("unresolved mutable image status must not carry a claimed digest")
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
        "oss_bundle_manifest_sha256": _optional_sha256(args.oss_bundle_manifest_sha256),
        "output_storage": args.output_storage,
        "output_zero_checkpoint_smoke_sha256": _optional_sha256(
            args.output_zero_checkpoint_smoke_sha256
        ),
        "pyproject_sha256": _optional_sha256(args.pyproject_sha256),
        "run_id": args.run_id,
        "schema_version": SCHEMA_VERSION,
        "task": args.task,
        "stats_sha256": _optional_sha256(args.stats_sha256),
        "vae_sha256": _optional_sha256(args.vae_sha256),
    }
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
        if hashlib.sha256(ready.read_bytes()).digest() != hashlib.sha256(probe_payload).digest():
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
    parser.add_argument("--image-reference", required=True)
    parser.add_argument(
        "--image-digest-status",
        choices=("resolved", "unresolved_mutable_tag"),
        required=True,
    )
    parser.add_argument("--image-digest", default="")
    parser.add_argument("--pyproject-sha256", required=True)
    parser.add_argument("--output-storage", choices=("cpfs", "oss_experimental"), required=True)
    parser.add_argument("--output-zero-checkpoint-smoke-sha256", default="")
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
