#!/usr/bin/env python3
"""Finalize or independently validate the real 32-rank N=4 full-model gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from fastwam.formal_artifacts import (
    finalize_n4_fullmodel_gate,
    publish_failure_marker,
    validate_n4_fullmodel_gate_binding,
    validate_terminal_sha256sums,
)


INPUT_BINDING_ENV = {
    "cpfs_bundle_manifest": "FASTWAM_CPFS_BUNDLE_MANIFEST_SHA256",
    "gaussian_cache_manifest": "FASTWAM_GAUSSIAN_CACHE_MANIFEST_SHA256",
    "gaussian_cache_selection": "FASTWAM_GAUSSIAN_CACHE_SELECTION_SHA256",
    "gaussian_cache_source_identity": "FASTWAM_GAUSSIAN_CACHE_SOURCE_IDENTITY_SHA256",
    "official_checkpoint": "FASTWAM_OFFICIAL_CHECKPOINT_SHA256",
    "oss_bundle_manifest": "FASTWAM_OSS_BUNDLE_MANIFEST_SHA256",
    "synthetic_zero2_gate": "FASTWAM_OUTPUT_ZERO_CHECKPOINT_SMOKE_SHA256",
    "stats": "FASTWAM_STATS_SHA256",
    "training_environment_bundle": "FASTWAM_TRAINING_ENV_BUNDLE_MANIFEST_SHA256",
    "vae": "FASTWAM_VAE_SHA256",
}


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required gate environment is missing: {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("finalize", "validate", "validate-binding"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-complete-sha256")
    parser.add_argument("--allowed-prefix", type=Path)
    parser.add_argument("--forbidden-output-root", type=Path)
    args = parser.parse_args()
    output_root = args.output_root.expanduser()
    try:
        if args.phase == "finalize":
            result = finalize_n4_fullmodel_gate(
                output_root,
                run_id=_required_environment("RUN_ID"),
                code_commit=_required_environment("FASTWAM_CODE_COMMIT"),
                image_reference=_required_environment("FASTWAM_DLC_IMAGE_REFERENCE"),
                image_digest=_required_environment("FASTWAM_DLC_IMAGE_DIGEST"),
                input_bindings={
                    key: _required_environment(environment_name)
                    for key, environment_name in INPUT_BINDING_ENV.items()
                },
            )
        elif args.phase == "validate":
            result = validate_terminal_sha256sums(
                output_root,
                complete_name="COMPLETE",
                expected_complete_schema="fastwam-n4-fullmodel-gate-complete",
            )
        else:
            if not args.expected_complete_sha256:
                raise ValueError(
                    "--expected-complete-sha256 is required for validate-binding"
                )
            if args.allowed_prefix is None or args.forbidden_output_root is None:
                raise ValueError(
                    "--allowed-prefix and --forbidden-output-root are required "
                    "for validate-binding"
                )
            result = validate_n4_fullmodel_gate_binding(
                output_root,
                allowed_prefix=args.allowed_prefix,
                forbidden_output_root=args.forbidden_output_root,
                expected_complete_sha256=args.expected_complete_sha256,
                code_commit=_required_environment("FASTWAM_CODE_COMMIT"),
                image_reference=_required_environment("FASTWAM_DLC_IMAGE_REFERENCE"),
                image_digest=_required_environment("FASTWAM_DLC_IMAGE_DIGEST"),
                input_bindings={
                    key: _required_environment(environment_name)
                    for key, environment_name in INPUT_BINDING_ENV.items()
                },
            )
    except BaseException as error:
        if args.phase == "finalize":
            complete = output_root / "COMPLETE"
            if not complete.exists() and not complete.is_symlink():
                publish_failure_marker(
                    output_root,
                    marker_name="GATE.FAILED.json",
                    schema_name="fastwam-n4-fullmodel-gate-failure",
                    error=error,
                    success_markers=["COMPLETE"],
                )
        raise
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
        raise
