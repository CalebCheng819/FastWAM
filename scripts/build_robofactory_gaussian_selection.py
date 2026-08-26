"""Build the exact current-frame key set used by FastWAM multi-robot training.

The full-resolution Gaussian cache remains all-timestep canonical data.  This
selection file drives a much smaller compact projection containing only the
current observation at each train/validation action window.  Its split and
window rules intentionally match ``RoboFactoryMultiRobotDataset``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterator

import h5py

from fastwam.datasets.robofactory_layout import action_dataset, agent_names


def _task_name_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if part.endswith("-rf"):
            return part
    return path.parent.name


def _split_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def iter_selection_records(
    root_dir: Path,
    *,
    action_horizon: int,
    train_window_stride: int,
    val_window_stride: int,
    val_set_proportion: float,
    split_seed: int,
    required_agent_counts: set[int] | None,
) -> Iterator[dict]:
    """Yield deterministic cache keys in source/trajectory/timestep order."""

    for h5_path in sorted(root_dir.rglob("*.h5")):
        source_path = h5_path.relative_to(root_dir).as_posix()
        task_name = _task_name_from_path(h5_path)
        with h5py.File(h5_path, "r") as handle:
            for trajectory_name in sorted(handle.keys()):
                group = handle[trajectory_name]
                if "actions" not in group:
                    continue
                names = list(agent_names(group))
                if not names:
                    continue
                agent_count = len(names)
                if (
                    required_agent_counts is not None
                    and agent_count not in required_agent_counts
                ):
                    continue
                length = int(action_dataset(group, names[0]).shape[0])
                if length < action_horizon:
                    continue
                split_key = f"{source_path}:{trajectory_name}"
                is_val = (
                    _split_fraction(split_key, split_seed) < val_set_proportion
                )
                split = "val" if is_val else "train"
                stride = val_window_stride if is_val else train_window_stride
                for timestep in range(0, length - action_horizon + 1, stride):
                    yield {
                        "source_path": source_path,
                        "trajectory": trajectory_name,
                        "timestep": timestep,
                        "agent_names": names,
                        "agent_count": agent_count,
                        "task_name": task_name,
                        "split": split,
                    }


def build_selection(args: argparse.Namespace) -> dict:
    root_dir = Path(args.root_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"RoboFactory root does not exist: {root_dir}")
    if output.exists():
        raise FileExistsError(
            f"Selection already exists: {output}. Use a new versioned output path."
        )
    if args.num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    if args.train_window_stride < 1 or args.val_window_stride < 1:
        raise ValueError("window strides must be positive")
    if not 0.0 <= args.val_set_proportion < 1.0:
        raise ValueError("val_set_proportion must be in [0,1)")

    required_counts = (
        None
        if args.required_agent_counts is None
        else {int(value) for value in args.required_agent_counts}
    )
    if required_counts is not None and any(value < 1 for value in required_counts):
        raise ValueError("required_agent_counts must contain positive integers")

    output.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    digest = hashlib.sha256()
    action_horizon = args.num_frames - 1
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".partial",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for record in iter_selection_records(
                root_dir,
                action_horizon=action_horizon,
                train_window_stride=args.train_window_stride,
                val_window_stride=args.val_window_stride,
                val_set_proportion=args.val_set_proportion,
                split_seed=args.split_seed,
                required_agent_counts=required_counts,
            ):
                encoded = (
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                temporary.write(encoded)
                digest.update(encoded)
                counters["windows"] += 1
                counters["selected_keys"] += int(record["agent_count"])
                counters[f"{record['split']}_windows"] += 1
                counters[f"n{record['agent_count']}_windows"] += 1
            temporary.flush()
            os.fsync(temporary.fileno())
        if counters["windows"] == 0:
            raise RuntimeError(f"No eligible windows found under {root_dir}")
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return {
        "output": str(output),
        "sha256": digest.hexdigest(),
        "source_root": str(root_dir),
        "action_horizon": action_horizon,
        "split_seed": args.split_seed,
        "val_set_proportion": args.val_set_proportion,
        "train_window_stride": args.train_window_stride,
        "val_window_stride": args.val_window_stride,
        "required_agent_counts": (
            None if required_counts is None else sorted(required_counts)
        ),
        **dict(sorted(counters.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--train-window-stride", type=int, default=16)
    parser.add_argument("--val-window-stride", type=int, default=32)
    parser.add_argument("--val-set-proportion", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--required-agent-counts", type=int, nargs="+")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build_selection(parse_args()), indent=2, sort_keys=True))
