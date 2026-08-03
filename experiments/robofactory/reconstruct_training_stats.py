#!/usr/bin/env python3
"""Reconstruct the byte-exact node-local stats file used by the 32-GPU run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path


SOURCE_STATS_SHA256 = "350493b685d8db0ea4cfd66f58f49849e8cd1f65cecc269f15aff9101ac8a04d"
BUNDLE_MANIFEST_SHA256 = (
    "bd3e034e8dcaca342a1776ea4ed3980e42ff82dd279380a13e38ba1593e38660"
)
DERIVED_STATS_SHA256 = (
    "92dfdeec62995b625b606d435ffb79ed787c4485348c16c42c3d31875eff64d0"
)
TRAINING_DATASET_ROOT = (
    "/tmp/fastwam-whole-file-cache/cpfs/"
    f"{BUNDLE_MANIFEST_SHA256}/datasets/robofactory_multi_robot"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def derive_payload(source: Path) -> bytes:
    source = source.expanduser().resolve(strict=True)
    raw = source.read_bytes()
    actual_source_sha256 = sha256_bytes(raw)
    if actual_source_sha256 != SOURCE_STATS_SHA256:
        raise ValueError(
            "Source stats SHA-256 mismatch: "
            f"expected={SOURCE_STATS_SHA256} actual={actual_source_sha256} path={source}"
        )
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("Source stats must be a JSON object")
    payload["source_root"] = TRAINING_DATASET_ROOT
    payload["fastwam_local_derivation"] = {
        "bundle_manifest_sha256": BUNDLE_MANIFEST_SHA256,
        "source_stats_sha256": SOURCE_STATS_SHA256,
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    actual_derived_sha256 = sha256_bytes(encoded)
    if actual_derived_sha256 != DERIVED_STATS_SHA256:
        raise ValueError(
            "Derived stats SHA-256 mismatch: "
            f"expected={DERIVED_STATS_SHA256} actual={actual_derived_sha256}"
        )
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to replace existing derived stats: {output}")
    encoded = derive_payload(args.source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "bytes": len(encoded),
                "output": str(output),
                "sha256": sha256_bytes(encoded),
                "status": "PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
