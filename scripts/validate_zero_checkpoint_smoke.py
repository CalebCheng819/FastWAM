#!/usr/bin/env python3
"""Validate a sealed real 32-rank DeepSpeed ZeRO-2 roundtrip smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from state_tree_manifest import canonical_bytes, validate_state_tree_manifest


HEX_64 = re.compile(r"^[0-9a-f]{64}$")
OCI_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
ROUNDTRIP_KEYS = {
    "global_step",
    "model",
    "optimizer",
    "rng",
    "rng_next_sample",
    "scheduler",
    "separate_process",
}
PINNED_PACKAGES = ("torch", "accelerate", "deepspeed")
EXPECTED_BATCH_ACCOUNTING = {
    "global_train_batch_size": 128,
    "gradient_accumulation_steps": 1,
    "local_micro_batch_size": 4,
    "world_size": 32,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().lower()


def _resolved_unaliased(path: Path, *, label: str, kind: str) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    absolute = Path(os.path.abspath(path))
    resolved = path.resolve(strict=True)
    if absolute != resolved:
        raise ValueError(f"{label} must not traverse symlinks or aliases: {path}")
    if kind == "directory":
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"{label} must be a non-symlink directory: {path}")
    elif kind == "file":
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    else:  # pragma: no cover - internal caller contract
        raise ValueError(f"unsupported path kind: {kind!r}")
    return resolved


def validate(marker: Path, expected_sha256: str, output_parent: Path) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    unit_test_override = os.environ.get(
        "FASTWAM_ZERO_SMOKE_UNIT_TEST_ALLOW_DIRTY", ""
    ) == "1"
    marker = _resolved_unaliased(marker, label="smoke marker", kind="file")
    expected_sha256 = expected_sha256.lower()
    if not HEX_64.fullmatch(expected_sha256):
        raise ValueError("smoke marker SHA-256 must be 64 lowercase hex characters")
    actual_marker_sha256 = _sha256(marker)
    if actual_marker_sha256 != expected_sha256:
        raise ValueError(
            "smoke marker SHA-256 mismatch: "
            f"expected={expected_sha256} actual={actual_marker_sha256}"
        )
    output_parent = _resolved_unaliased(
        output_parent, label="formal output parent", kind="directory"
    )
    encoded = marker.read_bytes()
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid ZeRO-2 smoke marker JSON: {error}") from error
    required_fields = {
        "batch_accounting",
        "code_commit",
        "filesystem_device",
        "image_digest",
        "image_reference",
        "output_root",
        "package_versions",
        "pyproject_sha256",
        "roundtrip",
        "schema_version",
        "state_tree_manifest",
        "state_tree_manifest_sha256",
        "state_tree_root",
        "status",
        "world_size",
        "zero_stage",
    }
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise ValueError("ZeRO-2 smoke marker fields do not match schema v2")
    if encoded != canonical_bytes(payload):
        raise ValueError("ZeRO-2 smoke marker is not canonical JSON")
    fixed = {
        "schema_version": 2,
        "status": "PASS",
        "world_size": 32,
        "zero_stage": 2,
    }
    for key, expected_value in fixed.items():
        if payload[key] != expected_value:
            raise ValueError(
                f"ZeRO-2 smoke marker field {key!r} must be {expected_value!r}, "
                f"got {payload[key]!r}"
            )
    if payload["batch_accounting"] != EXPECTED_BATCH_ACCOUNTING:
        raise ValueError(
            "ZeRO-2 smoke marker batch accounting mismatch: "
            f"expected={EXPECTED_BATCH_ACCOUNTING} "
            f"observed={payload['batch_accounting']!r}"
        )
    roundtrip = payload["roundtrip"]
    if not isinstance(roundtrip, dict) or set(roundtrip) != ROUNDTRIP_KEYS or not all(
        value is True for value in roundtrip.values()
    ):
        raise ValueError(
            "ZeRO-2 smoke must prove model/optimizer/scheduler/RNG/global-step "
            "roundtrip across a separate process"
        )

    declared_commit = os.environ.get("FASTWAM_CODE_COMMIT", "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", declared_commit):
        raise ValueError("FASTWAM_CODE_COMMIT is required to validate OSS smoke evidence")
    if payload["code_commit"] != declared_commit or _git_head(repository) != declared_commit:
        raise ValueError("ZeRO-2 smoke code commit does not match the current formal code")
    dirty = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty and not unit_test_override:
        raise ValueError("ZeRO-2 smoke validation requires a clean immutable Git worktree")
    image_reference = os.environ.get("FASTWAM_DLC_IMAGE_REFERENCE", "").strip()
    image_digest = os.environ.get("FASTWAM_DLC_IMAGE_DIGEST", "").strip().lower()
    if not image_reference or not OCI_DIGEST.fullmatch(image_digest):
        raise ValueError("OSS smoke validation requires an exact image reference and OCI digest")
    if payload["image_reference"] != image_reference or payload["image_digest"] != image_digest:
        raise ValueError("ZeRO-2 smoke image identity does not match the formal launch image")
    pyproject_sha256 = _sha256(repository / "pyproject.toml")
    if payload["pyproject_sha256"] != pyproject_sha256:
        raise ValueError("ZeRO-2 smoke pyproject SHA-256 mismatch")
    if unit_test_override:
        try:
            actual_versions = json.loads(
                os.environ["FASTWAM_ZERO_SMOKE_UNIT_TEST_PACKAGE_VERSIONS"]
            )
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError("unit-test package versions must be explicit JSON") from error
    else:
        actual_versions = {
            package: importlib.metadata.version(package) for package in PINNED_PACKAGES
        }
    if payload["package_versions"] != actual_versions:
        raise ValueError(
            "ZeRO-2 smoke package identity mismatch: "
            f"expected={actual_versions} observed={payload['package_versions']}"
        )

    output_root = _resolved_unaliased(
        Path(str(payload["output_root"])),
        label="ZeRO-2 smoke output_root",
        kind="directory",
    )
    state_root = _resolved_unaliased(
        Path(str(payload["state_tree_root"])),
        label="ZeRO-2 smoke state root",
        kind="directory",
    )
    state_manifest = _resolved_unaliased(
        Path(str(payload["state_tree_manifest"])),
        label="ZeRO-2 smoke state manifest",
        kind="file",
    )
    expected_paths = {
        "marker": output_root / "zero2-roundtrip-smoke.json",
        "state root": output_root / "zero2-state",
        "state manifest": output_root / "zero2-state-tree.json",
    }
    observed_paths = {
        "marker": marker,
        "state root": state_root,
        "state manifest": state_manifest,
    }
    for label, expected_path in expected_paths.items():
        if observed_paths[label] != expected_path:
            raise ValueError(
                f"ZeRO-2 smoke {label} path mismatch: "
                f"expected={expected_path} observed={observed_paths[label]}"
            )
    # st_dev is scoped to a mount namespace. Preserve the producer's value as
    # a typed diagnostic, but never compare it with another DLC pod's value.
    producer_filesystem_device = payload["filesystem_device"]
    if (
        isinstance(producer_filesystem_device, bool)
        or not isinstance(producer_filesystem_device, int)
        or producer_filesystem_device < 0
    ):
        raise ValueError(
            "ZeRO-2 smoke producer filesystem_device must be a non-negative integer"
        )
    filesystem_device = os.stat(output_root).st_dev
    if (
        os.stat(marker).st_dev != filesystem_device
        or os.stat(output_parent).st_dev != filesystem_device
        or os.stat(state_root).st_dev != filesystem_device
        or os.stat(state_manifest).st_dev != filesystem_device
    ):
        raise ValueError(
            "smoke marker, state tree, manifest, and formal output are not on one "
            "filesystem in the current mount namespace"
        )
    manifest_sha256 = payload["state_tree_manifest_sha256"]
    if not isinstance(manifest_sha256, str) or not HEX_64.fullmatch(manifest_sha256):
        raise ValueError("invalid ZeRO-2 state-tree manifest SHA-256")
    summary = validate_state_tree_manifest(
        state_root,
        state_manifest,
        expected_manifest_sha256=manifest_sha256,
        expected_role="zero2_roundtrip_smoke_state",
    )
    proof_paths = [record["path"] for record in summary["files"]]
    for rank in range(32):
        for prefix in ("save", "mutated", "load"):
            expected_proof = f"smoke-proof/{prefix}-rank-{rank:05d}.json"
            if expected_proof not in proof_paths:
                raise ValueError(f"sealed ZeRO-2 state tree is missing {expected_proof}")
            proof = json.loads((state_root / expected_proof).read_text(encoding="utf-8"))
            if proof.get("rank") != rank or proof.get("world_size") != 32:
                raise ValueError(
                    f"sealed ZeRO-2 {prefix} proof rank/world mismatch at rank {rank}"
                )
            if proof.get("batch_accounting") != EXPECTED_BATCH_ACCOUNTING:
                raise ValueError(
                    f"sealed ZeRO-2 {prefix} proof batch accounting mismatch "
                    f"at rank {rank}"
                )
    return {
        "batch_accounting": EXPECTED_BATCH_ACCOUNTING,
        "marker_sha256": actual_marker_sha256,
        "state_tree_manifest_sha256": manifest_sha256,
        "state_tree_files": summary["file_count"],
        "state_tree_total_bytes": summary["total_bytes"],
        "status": "PASS",
        "world_size": 32,
        "zero_stage": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(args.marker, args.expected_sha256, args.output_parent)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
