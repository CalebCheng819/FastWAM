#!/usr/bin/env python3
"""Build and validate deterministic MF-WAM G0 structural contracts.

Examples:

  python scripts/build_mf_wam_g0_contract.py inventory \
    --data-root /path/to/libero/libero \
    --task-map task-map.json --dataset-id libero-40 \
    --revision 8f1084e3132a39270c3a13ebe37270a43ece2a01 \
    --output data-inventory.json

  python scripts/build_mf_wam_g0_contract.py seeds \
    --task-map task-map.json --seed 42 --python-hash-seed 42 \
    --output seed-schedule.json

  python scripts/build_mf_wam_g0_contract.py prereg \
    --spec prereg-spec.json --inventory data-inventory.json \
    --seed-schedule seed-schedule.json --output preregistration.json

The ``prereg`` command refuses missing or placeholder image identities.  The
``validate-chain`` command always reports ``formal_training_allowed=false``;
raw G0 traces and statistics still require specialized recomputation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# A formal source checkout must remain free of ignored bytecode even when this
# entry point is invoked without the interpreter's ``-B`` flag.
sys.dont_write_bytecode = True


try:
    from fastwam.validation.g0_contract import (
        CANONICAL_JSON_ALGORITHM,
        ContractError,
        build_data_inventory,
        build_preregistration,
        build_seed_schedule,
        canonical_json_sha256,
        load_json_strict,
        validate_contract_chain,
        write_canonical_json,
    )
except ModuleNotFoundError:
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root / "src"))
    from fastwam.validation.g0_contract import (  # type: ignore[no-redef]
        CANONICAL_JSON_ALGORITHM,
        ContractError,
        build_data_inventory,
        build_preregistration,
        build_seed_schedule,
        canonical_json_sha256,
        load_json_strict,
        validate_contract_chain,
        write_canonical_json,
    )


def _emit(value: Any, output: Path | None) -> None:
    if output is None:
        print(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        write_canonical_json(output, value)
        print(
            json.dumps(
                {
                    "output": str(output.resolve()),
                    "canonical_json_algorithm": CANONICAL_JSON_ALGORITHM,
                    "canonical_sha256": canonical_json_sha256(value),
                },
                allow_nan=False,
                sort_keys=True,
            )
        )


def _inventory(args: argparse.Namespace) -> dict[str, Any]:
    task_map = load_json_strict(args.task_map)
    return build_data_inventory(
        args.data_root,
        task_map,
        dataset_id=args.dataset_id,
        revision=args.revision,
    )


def _seeds(args: argparse.Namespace) -> dict[str, Any]:
    task_map = load_json_strict(args.task_map)
    return build_seed_schedule(
        task_map,
        seed=args.seed,
        python_hash_seed=args.python_hash_seed,
    )


def _prereg(args: argparse.Namespace) -> dict[str, Any]:
    return build_preregistration(
        load_json_strict(args.spec),
        data_inventory=load_json_strict(args.inventory),
        seed_schedule=load_json_strict(args.seed_schedule),
    )


def _digest(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json_strict(args.input)
    return {
        "schema_version": 1,
        "kind": "mf_wam_canonical_json_digest",
        "canonical_json_algorithm": CANONICAL_JSON_ALGORITHM,
        "canonical_sha256": canonical_json_sha256(payload),
    }


def _validate_chain(args: argparse.Namespace) -> dict[str, Any]:
    return validate_contract_chain(
        preregistration=load_json_strict(args.preregistration),
        runtime_start=load_json_strict(args.runtime_start),
        terminal=load_json_strict(args.terminal),
        data_inventory=load_json_strict(args.inventory),
        seed_schedule=load_json_strict(args.seed_schedule),
        trusted_anchors={
            "preregistration_canonical_sha256": args.trusted_preregistration_sha256,
            "runtime_start_canonical_sha256": args.trusted_runtime_start_sha256,
            "terminal_canonical_sha256": args.trusted_terminal_sha256,
        },
        data_root=args.data_root,
        model_cache_root=args.model_cache_root,
        artifact_root=args.artifact_root,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser(
        "inventory", help="hash the exact 40 BDDL + 40 initial-state allowlist"
    )
    inventory.add_argument("--data-root", required=True, type=Path)
    inventory.add_argument("--task-map", required=True, type=Path)
    inventory.add_argument("--dataset-id", required=True)
    inventory.add_argument("--revision", required=True)
    inventory.add_argument("--output", type=Path)
    inventory.set_defaults(handler=_inventory)

    seeds = subparsers.add_parser(
        "seeds", help="build the one-process-per-task deterministic seed schedule"
    )
    seeds.add_argument("--task-map", required=True, type=Path)
    seeds.add_argument("--seed", required=True, type=int)
    seeds.add_argument("--python-hash-seed", required=True, type=int)
    seeds.add_argument("--output", type=Path)
    seeds.set_defaults(handler=_seeds)

    prereg = subparsers.add_parser(
        "prereg", help="bind an explicit preregistration spec to inventory/seeds"
    )
    prereg.add_argument("--spec", required=True, type=Path)
    prereg.add_argument("--inventory", required=True, type=Path)
    prereg.add_argument("--seed-schedule", required=True, type=Path)
    prereg.add_argument("--output", type=Path)
    prereg.set_defaults(handler=_prereg)

    digest = subparsers.add_parser("digest", help="compute a strict canonical JSON digest")
    digest.add_argument("--input", required=True, type=Path)
    digest.add_argument("--output", type=Path)
    digest.set_defaults(handler=_digest)

    chain = subparsers.add_parser(
        "validate-chain", help="validate preregistration -> start -> terminal links"
    )
    chain.add_argument("--preregistration", required=True, type=Path)
    chain.add_argument("--runtime-start", required=True, type=Path)
    chain.add_argument("--terminal", required=True, type=Path)
    chain.add_argument("--inventory", required=True, type=Path)
    chain.add_argument("--seed-schedule", required=True, type=Path)
    chain.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="safely re-read all 80 LIBERO files during validation",
    )
    chain.add_argument(
        "--model-cache-root",
        required=True,
        type=Path,
        help="safely re-read all six model-cache files during validation",
    )
    chain.add_argument(
        "--artifact-root",
        required=True,
        type=Path,
        help="safely re-read the exact terminal artifact allowlist",
    )
    chain.add_argument(
        "--trusted-preregistration-sha256",
        required=True,
        help="independently preserved canonical preregistration digest",
    )
    chain.add_argument(
        "--trusted-runtime-start-sha256",
        required=True,
        help="independently preserved canonical runtime-start digest",
    )
    chain.add_argument(
        "--trusted-terminal-sha256",
        required=True,
        help="independently preserved canonical terminal digest",
    )
    chain.add_argument("--output", type=Path)
    chain.set_defaults(handler=_validate_chain)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = args.handler(args)
        _emit(result, args.output)
        return 0
    except (ContractError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "mf_wam_g0_contract_error",
                    "status": "STRUCTURAL_FAIL",
                    "specialized_g0_status": "UNCERTAIN",
                    "error": str(exc),
                    "formal_training_allowed": False,
                },
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
