#!/usr/bin/env python3
"""Fail-closed NCCL all-reduce validation for a fixed DLC world."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import statistics
import time
from collections import Counter
from datetime import timedelta

import torch
import torch.distributed as dist


def _required_int_env(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"torchrun did not set {name}")
    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--expected-local-world-size", type=int, required=True)
    parser.add_argument("--expected-num-nodes", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--bandwidth-mib", type=int, default=256)
    parser.add_argument("--bandwidth-warmup", type=int, default=2)
    parser.add_argument("--bandwidth-iters", type=int, default=5)
    parser.add_argument("--min-algbw-gbps", type=float, default=5.0)
    args = parser.parse_args()

    if args.expected_world_size < 1:
        raise ValueError("--expected-world-size must be positive")
    if args.expected_local_world_size < 1:
        raise ValueError("--expected-local-world-size must be positive")
    if args.expected_num_nodes < 1:
        raise ValueError("--expected-num-nodes must be positive")
    if (
        args.expected_num_nodes * args.expected_local_world_size
        != args.expected_world_size
    ):
        raise ValueError(
            "expected nodes x local world size must equal expected world size"
        )
    if args.timeout < 1:
        raise ValueError("--timeout must be positive")
    if args.bandwidth_mib < 1:
        raise ValueError("--bandwidth-mib must be positive")
    if args.bandwidth_warmup < 0:
        raise ValueError("--bandwidth-warmup must be non-negative")
    if args.bandwidth_iters < 1:
        raise ValueError("--bandwidth-iters must be positive")
    if args.min_algbw_gbps <= 0:
        raise ValueError("--min-algbw-gbps must be positive")

    rank = _required_int_env("RANK")
    local_rank = _required_int_env("LOCAL_RANK")
    world_size = _required_int_env("WORLD_SIZE")
    local_world_size = _required_int_env("LOCAL_WORLD_SIZE")

    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"Expected WORLD_SIZE={args.expected_world_size}, observed {world_size}"
        )
    if local_world_size != args.expected_local_world_size:
        raise RuntimeError(
            "Expected LOCAL_WORLD_SIZE="
            f"{args.expected_local_world_size}, observed {local_world_size}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != args.expected_local_world_size:
        raise RuntimeError(
            f"Expected {args.expected_local_world_size} visible CUDA devices, "
            f"observed {torch.cuda.device_count()}"
        )
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside the visible CUDA device range"
        )

    torch.cuda.set_device(local_rank)
    initialized = False
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=args.timeout),
        )
        initialized = True

        reduced = torch.tensor(
            [float(rank + 1), 1.0],
            dtype=torch.float64,
            device=f"cuda:{local_rank}",
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(local_rank)

        expected_rank_sum = world_size * (world_size + 1) / 2
        observed_rank_sum, observed_count = reduced.cpu().tolist()
        if observed_rank_sum != expected_rank_sum or observed_count != world_size:
            raise RuntimeError(
                "All-reduce mismatch: "
                f"sum={observed_rank_sum} count={observed_count}, "
                f"expected_sum={expected_rank_sum} expected_count={world_size}"
            )

        hostname = socket.gethostname()
        host_id = int.from_bytes(
            hashlib.sha256(hostname.encode("utf-8")).digest()[:8], "big"
        ) & ((1 << 63) - 1)
        host_tensor = torch.tensor(
            [host_id], dtype=torch.int64, device=f"cuda:{local_rank}"
        )
        gathered_hosts = [torch.empty_like(host_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_hosts, host_tensor)
        host_counts = Counter(int(item.item()) for item in gathered_hosts)
        if len(host_counts) != args.expected_num_nodes:
            raise RuntimeError(
                f"Expected {args.expected_num_nodes} distinct nodes, "
                f"observed {len(host_counts)}"
            )
        if set(host_counts.values()) != {args.expected_local_world_size}:
            raise RuntimeError(
                "Unexpected ranks per node: "
                f"observed={sorted(host_counts.values())}, "
                f"expected_each={args.expected_local_world_size}"
            )

        payload_bytes = args.bandwidth_mib * 1024 * 1024
        element_count = payload_bytes // torch.tensor([], dtype=torch.float32).element_size()
        bandwidth_tensor = torch.ones(
            element_count,
            dtype=torch.float32,
            device=f"cuda:{local_rank}",
        )
        for _ in range(args.bandwidth_warmup):
            bandwidth_tensor.fill_(1.0)
            dist.all_reduce(bandwidth_tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(local_rank)

        algbw_samples: list[float] = []
        for _ in range(args.bandwidth_iters):
            bandwidth_tensor.fill_(1.0)
            dist.barrier()
            torch.cuda.synchronize(local_rank)
            started = time.perf_counter()
            dist.all_reduce(bandwidth_tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(local_rank)
            elapsed = time.perf_counter() - started
            elapsed_tensor = torch.tensor(
                [elapsed], dtype=torch.float64, device=f"cuda:{local_rank}"
            )
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            global_elapsed = float(elapsed_tensor.item())
            algbw_samples.append(payload_bytes / global_elapsed / 1_000_000_000)

        median_algbw = float(statistics.median(algbw_samples))
        minimum_algbw = float(min(algbw_samples))
        median_busbw = median_algbw * 2.0 * (world_size - 1) / world_size
        if median_algbw < args.min_algbw_gbps:
            raise RuntimeError(
                "Distributed bandwidth gate failed: "
                f"median_algbw_gbps={median_algbw:.3f} "
                f"required={args.min_algbw_gbps:.3f}"
            )

        dist.barrier()
        if rank == 0:
            print(
                "DISTRIBUTED_CUDA_VALIDATION=PASS "
                + json.dumps(
                    {
                        "backend": dist.get_backend(),
                        "hostname": hostname,
                        "num_nodes": len(host_counts),
                        "world_size": world_size,
                        "local_world_size": local_world_size,
                        "rank_sum": observed_rank_sum,
                        "bandwidth_mib": args.bandwidth_mib,
                        "bandwidth_iters": args.bandwidth_iters,
                        "median_algbw_gbps": round(median_algbw, 6),
                        "minimum_algbw_gbps": round(minimum_algbw, 6),
                        "median_busbw_gbps": round(median_busbw, 6),
                        "required_algbw_gbps": args.min_algbw_gbps,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if initialized:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
