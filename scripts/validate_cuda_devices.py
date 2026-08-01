#!/usr/bin/env python3
"""Fail-closed CUDA device validation for DLC preflight jobs."""

from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--allocation-mib", type=int, default=16)
    args = parser.parse_args()

    if args.expected < 1:
        raise ValueError(f"--expected must be positive, got {args.expected}")
    if args.allocation_mib < 1:
        raise ValueError(
            f"--allocation-mib must be positive, got {args.allocation_mib}"
        )

    device_count = torch.cuda.device_count()
    inventory = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        inventory.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        )
    print(
        "CUDA_INVENTORY "
        + json.dumps(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device_count": device_count,
                "devices": inventory,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if device_count != args.expected:
        raise RuntimeError(
            f"Expected {args.expected} CUDA devices, observed {device_count}"
        )

    elements = args.allocation_mib * 1024 * 1024 // 4
    checksums = []
    for index in range(device_count):
        with torch.cuda.device(index):
            tensor = torch.ones(elements, device=f"cuda:{index}", dtype=torch.float32)
            checksum = float(tensor.sum().item())
            torch.cuda.synchronize(index)
        if checksum != float(elements):
            raise RuntimeError(
                f"CUDA checksum mismatch on device {index}: {checksum} != {elements}"
            )
        checksums.append(checksum)

    print(
        "CUDA_DEVICE_VALIDATION=PASS "
        + json.dumps(
            {
                "expected": args.expected,
                "observed": device_count,
                "allocation_mib_per_device": args.allocation_mib,
                "checksums": checksums,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
