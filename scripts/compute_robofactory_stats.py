#!/usr/bin/env python3
"""Compute shared RoboFactory action/state normalization statistics."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from fastwam.datasets.robofactory_multi_robot import compute_robofactory_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = compute_robofactory_stats(args.root_dir)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp.{uuid.uuid4().hex}"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, output)
    print(
        f"wrote {output} files={payload['files']} trajectories={payload['trajectories']} "
        f"action_count={payload['action']['count']} state_count={payload['state']['count']}"
    )


if __name__ == "__main__":
    main()
