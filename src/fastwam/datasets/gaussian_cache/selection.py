"""Selection JSONL parsing for sparse current-frame cache projections."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .manifest import sha256_file, write_immutable_file
from .schema import FrameKey

NORMALIZED_SELECTION_ALGORITHM = "sorted-deduplicated-frame-key-jsonl-v1"


def expand_selection_record(record: Mapping[str, Any]) -> Iterable[FrameKey]:
    required = {"source_path", "trajectory", "timestep"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Selection record is missing fields {missing}: {record}")
    has_one = "agent_name" in record
    has_many = "agent_names" in record
    if has_one == has_many:
        raise ValueError("Selection record must contain exactly one of agent_name or agent_names")
    names = [record["agent_name"]] if has_one else record["agent_names"]
    if not isinstance(names, list) or not names:
        raise ValueError(f"agent_names must be a non-empty list: {record}")
    for name in names:
        yield FrameKey(
            source_path=str(record["source_path"]),
            trajectory=str(record["trajectory"]),
            timestep=int(record["timestep"]),
            agent_name=str(name),
        )


def load_selection_jsonl(path: str | Path) -> list[FrameKey]:
    selection_path = Path(path)
    keys: set[FrameKey] = set()
    with selection_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise TypeError("record is not a JSON object")
                keys.update(expand_selection_record(record))
            except Exception as exc:
                raise ValueError(
                    f"Invalid Gaussian selection JSONL at {selection_path}:{line_number}: {exc}"
                ) from exc
    if not keys:
        raise ValueError(f"Gaussian selection JSONL contains no keys: {selection_path}")
    return sorted(keys)


def normalized_selection_payload(keys: Iterable[FrameKey]) -> tuple[list[FrameKey], bytes]:
    """Return the one canonical byte representation used by plans and caches."""

    ordered = sorted(set(keys))
    if not ordered:
        raise ValueError("Cannot normalize an empty Gaussian cache selection")
    payload = b"".join(
        (json.dumps(key.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for key in ordered
    )
    return ordered, payload


def normalized_selection_identity(keys: Iterable[FrameKey]) -> dict[str, Any]:
    """Hash the canonical deduplicated key set without materializing a cache."""

    ordered, payload = normalized_selection_payload(keys)
    return {
        "algorithm": NORMALIZED_SELECTION_ALGORITHM,
        "index_sha256": hashlib.sha256(payload).hexdigest(),
        "selected_key_count": len(ordered),
    }


def selection_manifest(
    mode: str,
    *,
    selection_jsonl: str | Path | None = None,
    selected_key_count: int | None = None,
) -> dict[str, Any]:
    mode = str(mode).lower()
    if mode == "all":
        if selection_jsonl is not None:
            raise ValueError("selection_jsonl is only valid for mode='index'")
        return {"mode": "all", "selected_key_count": selected_key_count}
    if mode != "index" or selection_jsonl is None:
        raise ValueError("mode='index' requires selection_jsonl")
    path = Path(selection_jsonl)
    return {
        "mode": "index",
        "index_filename": path.name,
        "index_sha256": sha256_file(path),
        "selected_key_count": selected_key_count,
    }


def write_normalized_selection_index(
    cache_root: str | Path,
    keys: Iterable[FrameKey],
) -> dict[str, Any]:
    """Persist the exact deduplicated projection keys inside an immutable cache."""

    ordered, payload = normalized_selection_payload(keys)
    path = Path(cache_root) / "selection.jsonl"
    write_immutable_file(path, payload)
    return {
        "mode": "index",
        "index_filename": path.name,
        "index_sha256": hashlib.sha256(payload).hexdigest(),
        "selected_key_count": len(ordered),
    }
