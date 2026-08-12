import hashlib
import json
import sys

import h5py
import numpy as np
import pytest

from fastwam.datasets import robofactory_multi_robot as robofactory
from scripts import compute_robofactory_stats as stats_cli


_SOURCE_LAYOUT = {
    "a-task-rf/motionplanning/z-source.h5": ("traj_a", "traj_z"),
    "m-task-rf/motionplanning/a-source.h5": ("traj_b", "traj_y"),
    "z-task-rf/motionplanning/m-source.h5": ("traj_c", "traj_x"),
}


def _write_ordinal_coded_sources(root):
    ordered_keys = [
        (relative_path, trajectory_name)
        for relative_path in sorted(_SOURCE_LAYOUT)
        for trajectory_name in sorted(_SOURCE_LAYOUT[relative_path])
    ]
    ordinal_by_key = {key: ordinal for ordinal, key in enumerate(ordered_keys)}

    # Deliberately create both files and groups in reverse order. The split must
    # depend on sorted paths/names, not filesystem or HDF5 insertion order.
    for relative_path in reversed(tuple(_SOURCE_LAYOUT)):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.create_group("000-no-actions")
            handle.create_group("001-empty-actions").create_group("actions")
            for trajectory_name in reversed(_SOURCE_LAYOUT[relative_path]):
                ordinal = ordinal_by_key[(relative_path, trajectory_name)]
                value = float(ordinal + 1)
                trajectory = handle.create_group(trajectory_name)
                actions = trajectory.create_group("actions")
                actions.create_dataset(
                    "panda-0",
                    data=np.full((16, 8), value, dtype=np.float32),
                )
                agent = trajectory.create_group("obs").create_group("agent")
                panda = agent.create_group("panda-0")
                panda.create_dataset(
                    "qpos",
                    data=np.full((17, 9), value * 10.0, dtype=np.float32),
                )
                panda.create_dataset(
                    "qvel",
                    data=np.full((17, 9), -value, dtype=np.float32),
                )
    return ordinal_by_key


def _split_case(root, *, seed=73):
    ordinal_by_key = _write_ordinal_coded_sources(root)
    fractions = {
        ordinal: robofactory._split_fraction_from_ordinal(ordinal, seed)
        for ordinal in ordinal_by_key.values()
    }
    sorted_fractions = sorted(fractions.values())
    val_proportion = (sorted_fractions[2] + sorted_fractions[3]) / 2.0
    train_ordinals = sorted(
        ordinal
        for ordinal, fraction in fractions.items()
        if fraction >= val_proportion
    )
    return seed, val_proportion, train_ordinals


def _forbid_digest_and_legacy_split(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("metadata_no_hash stats must not call a digest")

    for constructor in (
        "new",
        "md5",
        "sha1",
        "sha224",
        "sha256",
        "sha384",
        "sha512",
        "blake2b",
        "blake2s",
    ):
        monkeypatch.setattr(hashlib, constructor, fail)
    monkeypatch.setattr(robofactory, "_split_fraction", fail)


def _assert_expected_payload(payload, *, seed, val_proportion, train_ordinals):
    expected_values = np.asarray(
        [float(ordinal + 1) for ordinal in train_ordinals], dtype=np.float64
    )
    assert payload["trajectories"] == 6
    assert payload["normalization_fit"] == {
        "key_scheme": "sorted_trajectory_ordinal_splitmix64_v1",
        "split": "train",
        "split_seed": seed,
        "val_set_proportion": val_proportion,
        "trajectories": len(train_ordinals),
        "cardinality": {
            "agent_counts": [1],
            "trajectories_by_agent_count": {"1": len(train_ordinals)},
        },
    }
    assert payload["action"]["count"] == len(train_ordinals) * 16
    assert payload["action"]["mean"] == pytest.approx(
        [float(expected_values.mean())] * 8
    )
    assert payload["state"]["count"] == len(train_ordinals) * 17
    assert payload["state"]["mean"] == pytest.approx(
        [float(expected_values.mean() * 10.0)] * 9
        + [float(-expected_values.mean())] * 9
    )


def test_compute_stats_metadata_no_hash_uses_sorted_dataset_ordinals(
    tmp_path, monkeypatch
):
    seed, val_proportion, train_ordinals = _split_case(tmp_path)
    _forbid_digest_and_legacy_split(monkeypatch)

    payload = robofactory.compute_robofactory_stats(
        str(tmp_path),
        split_seed=seed,
        val_set_proportion=val_proportion,
        integrity_mode="metadata_no_hash",
    )

    _assert_expected_payload(
        payload,
        seed=seed,
        val_proportion=val_proportion,
        train_ordinals=train_ordinals,
    )


def test_compute_stats_cli_forwards_metadata_no_hash(tmp_path, monkeypatch):
    seed, val_proportion, train_ordinals = _split_case(tmp_path)
    output = tmp_path / "stats.json"
    _forbid_digest_and_legacy_split(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_robofactory_stats.py",
            "--root-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--split-seed",
            str(seed),
            "--val-set-proportion",
            str(val_proportion),
            "--integrity-mode",
            "metadata_no_hash",
        ],
    )

    stats_cli.main()

    _assert_expected_payload(
        json.loads(output.read_text(encoding="utf-8")),
        seed=seed,
        val_proportion=val_proportion,
        train_ordinals=train_ordinals,
    )


def test_compute_stats_integrity_mode_defaults_to_legacy_hash():
    assert (
        robofactory.compute_robofactory_stats.__kwdefaults__["integrity_mode"]
        == "legacy_hash"
    )
