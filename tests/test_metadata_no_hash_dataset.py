import hashlib
import json
import multiprocessing
import os
import pickle
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

import fastwam.datasets.gaussian_cache.provider as gaussian_provider_module
import fastwam.datasets.robofactory_multi_robot as robofactory_module
from fastwam.datasets.gaussian_cache import (
    FrameKey,
    MOMENT_MATCH_METHOD,
    GaussianCache,
    GaussianCacheSchema,
    load_manifest,
)
from fastwam.datasets.robofactory_multi_robot import RoboFactoryMultiRobotDataset


TASK_NAME = "Synthetic2RobotTask-rf"
INSTRUCTION = "two robots complete the synthetic task"
SOURCE_PATH = f"{TASK_NAME}/motionplanning/demo.h5"


def _prefetched_dataset_fork_worker(dataset, connection) -> None:
    """Exercise inherited HDF5/memmap state without pickling the dataset."""

    try:
        cache = dataset._gaussian_cache
        inherited_handles = list(dataset._h5_handles.values())
        inherited_arrays = list(cache._arrays.values())
        stat_paths = []
        stable_regular_file_stat = gaussian_provider_module.stable_regular_file_stat

        def recording_stable_regular_file_stat(path, *, expected_bytes=None):
            stat_paths.append(str(Path(path)))
            return stable_regular_file_stat(path, expected_bytes=expected_bytes)

        gaussian_provider_module.stable_regular_file_stat = (
            recording_stable_regular_file_stat
        )
        before = {
            "dataset_owner_pid": dataset._h5_owner_pid,
            "cache_owner_pid": cache._owner_pid,
            "h5_handles": len(inherited_handles),
            "arrays": len(inherited_arrays),
            "validated_shards": len(cache._validated_shards),
        }
        sample = dataset[0]
        connection.send(
            {
                "pid": os.getpid(),
                "before": before,
                "dataset_owner_pid": dataset._h5_owner_pid,
                "cache_owner_pid": cache._owner_pid,
                "h5_handles": len(dataset._h5_handles),
                "arrays": len(cache._arrays),
                "validated_shards": len(cache._validated_shards),
                "inherited_h5_closed": all(
                    not handle.id.valid for handle in inherited_handles
                ),
                "inherited_memmaps_closed": all(
                    getattr(array, "_mmap", None) is None or array._mmap.closed
                    for array in inherited_arrays
                ),
                "stat_paths": stat_paths,
                "gaussian_shape": tuple(sample["agent_gaussian"].shape),
            }
        )
    except BaseException as exc:
        connection.send({"error": repr(exc)})
    finally:
        connection.close()


def _write_dataset_inputs(root: Path) -> tuple[Path, Path]:
    source = root / SOURCE_PATH
    source.parent.mkdir(parents=True)
    length = 40
    with h5py.File(source, "w") as handle:
        trajectory = handle.create_group("traj_0")
        actions = trajectory.create_group("actions")
        obs = trajectory.create_group("obs")
        agents = obs.create_group("agent")
        sensors = obs.create_group("sensor_data")
        global_camera = sensors.create_group("head_camera_global")
        global_camera.create_dataset(
            "rgb",
            data=np.zeros((length + 1, 16, 16, 3), dtype=np.uint8),
        )
        articulations = trajectory.create_group("env_states").create_group(
            "articulations"
        )
        for agent_index in range(2):
            agent_name = f"panda-{agent_index}"
            actions.create_dataset(
                agent_name,
                data=np.full(
                    (length, 8),
                    float(agent_index + 1),
                    dtype=np.float32,
                ),
            )
            agent = agents.create_group(agent_name)
            agent.create_dataset(
                "qpos", data=np.ones((length + 1, 9), dtype=np.float32)
            )
            agent.create_dataset(
                "qvel", data=np.zeros((length + 1, 9), dtype=np.float32)
            )
            articulation = np.zeros((length + 1, 7), dtype=np.float32)
            articulation[:, 0] = float(agent_index)
            articulation[:, 3] = 1.0
            articulations.create_dataset(
                f"panda-agent-{agent_index}", data=articulation
            )

    stats_path = root / "stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "action": {"mean": [0.0] * 8, "std": [1.0] * 8},
                "state": {"mean": [0.0] * 18, "std": [1.0] * 18},
            }
        ),
        encoding="utf-8",
    )
    text_cache_dir = root / "text-cache"
    text_cache_dir.mkdir()
    (text_cache_dir / "synthetic.pt").write_bytes(b"explicit metadata-no-hash cache")
    return stats_path, text_cache_dir


def _write_gaussian_cache(root: Path, *, selected_key_count: int = 2) -> Path:
    cache_root = root / f"gaussian-cache-{selected_key_count}"
    shards_dir = cache_root / "shards"
    shards_dir.mkdir(parents=True)
    schema = GaussianCacheSchema(height=2, width=2, cache_kind="compact")
    frames = np.stack(
        [
            np.full(schema.frame_shape, float(agent_index + 1), dtype=np.dtype("<f2"))
            for agent_index in range(2)
        ]
    )
    shard_payload = frames.tobytes(order="C")
    shard_path = shards_dir / "shard-000000-existing-digest.f16"
    shard_path.write_bytes(shard_payload)

    selection_records = [
        {
            "source_path": SOURCE_PATH,
            "trajectory": "traj_0",
            "timestep": 0,
            "agent_name": f"panda-{agent_index}",
        }
        for agent_index in range(2)
    ]
    (cache_root / "selection.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in selection_records),
        encoding="utf-8",
    )
    streams = [
        {
            "source_path": SOURCE_PATH,
            "trajectory": "traj_0",
            "agent_name": f"panda-{agent_index}",
            "observation_count": 40,
            "stored_count": 1,
            "segments": [
                {
                    "shard": "000000",
                    "offset": agent_index,
                    "count": 1,
                    "source_start": 0,
                    "source_stride": 1,
                }
            ],
        }
        for agent_index in range(2)
    ]
    manifest = {
        "manifest_version": 1,
        "created_at": "2026-08-09T00:00:00+00:00",
        "schema": schema.to_dict(),
        "target_shard_bytes": 1 << 30,
        "selection": {
            "mode": "index",
            "index_filename": "selection.jsonl",
            "index_sha256": "c" * 64,
            "selected_key_count": selected_key_count,
        },
        "teacher": {"kind": "synthetic"},
        "derivation": {
            "method": MOMENT_MATCH_METHOD,
            "output_size": [2, 2],
            "parent_cache_kind": "canonical",
            "parent_total_frames": 2,
            "parent_teacher": {"kind": "synthetic"},
            "parent_selection": {"mode": "all", "selected_key_count": 2},
        },
        "sources": [
            {
                "path": SOURCE_PATH,
                "bytes": (root / SOURCE_PATH).stat().st_size,
                "sha256": "a" * 64,
            }
        ],
        "shards": [
            {
                "id": "000000",
                "path": "shards/shard-000000-existing-digest.f16",
                "sha256": "b" * 64,
                "bytes": len(shard_payload),
                "frames": 2,
                "final": True,
                "immutable": True,
            }
        ],
        "streams": streams,
        "total_frames": 2,
    }
    manifest_payload = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    (cache_root / "manifest.json").write_bytes(manifest_payload)
    complete = {
        "complete": True,
        "schema_name": schema.to_dict()["name"],
        "schema_version": schema.to_dict()["version"],
        "manifest_version": 1,
        "manifest": "manifest.json",
        "manifest_bytes": len(manifest_payload),
        "manifest_sha256": "d" * 64,
        "shard_count": 1,
        "total_frames": 2,
    }
    (cache_root / "COMPLETE").write_text(
        json.dumps(complete, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return cache_root


def _write_canonical_fallback_cache(root: Path) -> tuple[Path, Path]:
    """Write two canonical streams in separate shards and remove the unused one."""

    cache_root = root / "gaussian-canonical-fallback"
    schema = GaussianCacheSchema(height=4, width=4, cache_kind="canonical")
    shard_records = []
    streams = []
    for agent_index in range(2):
        shards_dir = (
            cache_root / "parts" / f"part-{agent_index:05d}" / "shards"
        )
        shards_dir.mkdir(parents=True)
        frame = np.zeros(schema.frame_shape, dtype=np.dtype("<f2"))
        frame[0:3] = float(agent_index + 1)
        frame[3] = 1.0
        frame[7] = 1.0
        frame[11] = 1.0
        frame[12] = 0.5
        payload = frame[None].tobytes(order="C")
        shard_id = f"{agent_index:06d}"
        relative_path = (
            f"parts/part-{agent_index:05d}/shards/"
            f"shard-{shard_id}-existing-digest.f16"
        )
        shard_path = cache_root / relative_path
        shard_path.write_bytes(payload)
        shard_records.append(
            {
                "id": shard_id,
                "path": relative_path,
                "sha256": chr(ord("a") + agent_index) * 64,
                "bytes": len(payload),
                "frames": 1,
                "final": True,
                "immutable": True,
                "part_index": agent_index,
            }
        )
        streams.append(
            {
                "source_path": SOURCE_PATH,
                "trajectory": "traj_0",
                "agent_name": f"panda-{agent_index}",
                "observation_count": 40,
                "stored_count": 1,
                "part_index": agent_index,
                "segments": [
                    {
                        "shard": shard_id,
                        "offset": 0,
                        "count": 1,
                        "source_start": 0,
                        "source_stride": 1,
                    }
                ],
            }
        )
    manifest = {
        "manifest_version": 2,
        "created_at": "2026-08-09T00:00:00+00:00",
        "schema": schema.to_dict(),
        "target_shard_bytes": 1 << 30,
        "selection": {"mode": "all", "selected_key_count": 2},
        "teacher": {"kind": "synthetic"},
        "sources": [
            {
                "path": SOURCE_PATH,
                "bytes": (root / SOURCE_PATH).stat().st_size,
                "sha256": "c" * 64,
            }
        ],
        "shards": shard_records,
        "streams": streams,
        "total_frames": 2,
        "parts": [
            {
                "partition_index": agent_index,
                "shard_count": 1,
                "stream_count": 1,
                "total_frames": 1,
            }
            for agent_index in range(2)
        ],
    }
    manifest_payload = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    (cache_root / "manifest.json").write_bytes(manifest_payload)
    complete = {
        "complete": True,
        "schema_name": schema.to_dict()["name"],
        "schema_version": schema.to_dict()["version"],
        "manifest_version": 2,
        "manifest": "manifest.json",
        "manifest_bytes": len(manifest_payload),
        "manifest_sha256": "d" * 64,
        "shard_count": 2,
        "total_frames": 2,
    }
    (cache_root / "COMPLETE").write_text(
        json.dumps(complete, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    unused_shard = cache_root / shard_records[0]["path"]
    return cache_root, unused_shard


def _forbid_digest_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("metadata_no_hash must not call a digest constructor")

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
        "shake_128",
        "shake_256",
    ):
        if hasattr(hashlib, constructor):
            monkeypatch.setattr(hashlib, constructor, fail)
    monkeypatch.setattr(robofactory_module, "sha256_file", fail)
    monkeypatch.setattr(
        robofactory_module, "gaussian_source_identity_sha256", fail
    )


def test_robofactory_import_does_not_pull_huggingface_datasets():
    code = (
        "import sys; "
        "import fastwam.datasets.robofactory_multi_robot; "
        "assert 'datasets' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_metadata_no_hash_dataset_manifest_provider_and_getitem(
    tmp_path, monkeypatch
):
    stats_path, text_cache_dir = _write_dataset_inputs(tmp_path)
    gaussian_cache_dir = _write_gaussian_cache(tmp_path)

    def fake_torch_load(path, *, map_location, weights_only):
        assert Path(path) == text_cache_dir / "synthetic.pt"
        assert map_location == "cpu"
        assert weights_only is True
        return {
            "context": torch.zeros(128, 16),
            "mask": torch.ones(128, dtype=torch.bool),
        }

    monkeypatch.setattr(torch, "load", fake_torch_load)
    monkeypatch.setattr(
        RoboFactoryMultiRobotDataset,
        "_text_cache_path",
        lambda *args, **kwargs: pytest.fail(
            "metadata_no_hash must not derive a hashed text-cache filename"
        ),
    )
    _forbid_digest_calls(monkeypatch)

    manifest = load_manifest(
        gaussian_cache_dir,
        integrity_mode="metadata_no_hash",
    )
    assert manifest["total_frames"] == 2
    with GaussianCache.open(
        gaussian_cache_dir,
        verify="manifest",
        integrity_mode="metadata_no_hash",
    ) as cache:
        assert cache.schema.frame_shape == (13, 2, 2)

    dataset = RoboFactoryMultiRobotDataset(
        str(tmp_path),
        load_future_video=False,
        video_size=(16, 16),
        val_set_proportion=0.0,
        is_training_set=True,
        randomize_agent_order=True,
        integrity_mode="metadata_no_hash",
        pretrained_norm_stats=str(stats_path),
        text_embedding_cache_dir=str(text_cache_dir),
        text_embedding_cache_files={TASK_NAME: "synthetic.pt"},
        gaussian_cache_dir=str(gaussian_cache_dir),
        gaussian_cache_verify="manifest",
        gaussian_size=(2, 2),
        instruction_map={TASK_NAME: INSTRUCTION},
    )
    assert dataset.entries[0]["trajectory_ordinal"] == 0
    dataset._text_context_cache.clear()
    first = dataset[0]
    second = dataset[0]
    assert first["agent_ids"].tolist() == second["agent_ids"].tolist()
    for slot, original_agent in enumerate(first["agent_ids"].tolist()):
        assert torch.all(first["agent_gaussian"][slot] == float(original_agent + 1))
        assert torch.all(first["action"][slot] == float(original_agent + 1))


def test_metadata_no_hash_provider_pickle_revalidates_actual_shard(
    tmp_path, monkeypatch
):
    _write_dataset_inputs(tmp_path)
    gaussian_cache_dir = _write_gaussian_cache(tmp_path)
    _forbid_digest_calls(monkeypatch)
    key = FrameKey(SOURCE_PATH, "traj_0", 0, "panda-0")

    cache = GaussianCache.open(
        gaussian_cache_dir,
        verify="manifest",
        integrity_mode="metadata_no_hash",
        shard_validation="on_access",
    )
    restored = None
    try:
        assert tuple(cache.get_frame(key).shape) == (13, 2, 2)
        assert cache.validation_report == {
            "declared_shards": 1,
            "validated_shards": 1,
        }

        restored = pickle.loads(pickle.dumps(cache))
        assert restored._arrays == {}
        assert restored.validation_report == {
            "declared_shards": 1,
            "validated_shards": 0,
        }

        stat_paths = []
        stable_regular_file_stat = gaussian_provider_module.stable_regular_file_stat

        def recording_stable_regular_file_stat(path, *, expected_bytes=None):
            stat_paths.append(Path(path))
            return stable_regular_file_stat(path, expected_bytes=expected_bytes)

        monkeypatch.setattr(
            gaussian_provider_module,
            "stable_regular_file_stat",
            recording_stable_regular_file_stat,
        )
        assert tuple(restored.get_frame(key).shape) == (13, 2, 2)
        expected_shard = gaussian_cache_dir / restored.manifest["shards"][0]["path"]
        assert stat_paths == [expected_shard]
        assert restored.validation_report == {
            "declared_shards": 1,
            "validated_shards": 1,
        }
    finally:
        cache.close()
        if restored is not None:
            restored.close()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="requires the fork multiprocessing start method",
)
def test_metadata_no_hash_prefetched_dataset_resets_file_state_after_fork(
    tmp_path, monkeypatch
):
    stats_path, text_cache_dir = _write_dataset_inputs(tmp_path)
    gaussian_cache_dir = _write_gaussian_cache(tmp_path)

    def fake_torch_load(path, *, map_location, weights_only):
        assert Path(path) == text_cache_dir / "synthetic.pt"
        assert map_location == "cpu"
        assert weights_only is True
        return {
            "context": torch.zeros(128, 16),
            "mask": torch.ones(128, dtype=torch.bool),
        }

    monkeypatch.setattr(torch, "load", fake_torch_load)
    _forbid_digest_calls(monkeypatch)
    dataset = RoboFactoryMultiRobotDataset(
        str(tmp_path),
        load_future_video=False,
        video_size=(16, 16),
        val_set_proportion=0.0,
        is_training_set=True,
        randomize_agent_order=False,
        integrity_mode="metadata_no_hash",
        pretrained_norm_stats=str(stats_path),
        text_embedding_cache_dir=str(text_cache_dir),
        text_embedding_cache_files={TASK_NAME: "synthetic.pt"},
        gaussian_cache_dir=str(gaussian_cache_dir),
        gaussian_cache_verify="manifest",
        gaussian_size=(2, 2),
        instruction_map={TASK_NAME: INSTRUCTION},
    )

    assert tuple(dataset[0]["agent_gaussian"].shape) == (2, 13, 2, 2)
    cache = dataset._gaussian_cache
    assert cache is not None
    parent_pid = os.getpid()
    assert dataset._h5_owner_pid == parent_pid
    assert cache._owner_pid == parent_pid
    assert len(dataset._h5_handles) == 1
    assert len(cache._arrays) == 1
    assert len(cache._validated_shards) == 1

    context = multiprocessing.get_context("fork")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_prefetched_dataset_fork_worker,
        args=(dataset, child_connection),
    )
    result = None
    process.start()
    child_connection.close()
    try:
        if parent_connection.poll(15):
            result = parent_connection.recv()
    finally:
        parent_connection.close()
        process.join(15)
        if process.is_alive():
            process.terminate()
            process.join(5)

    assert result is not None, "forked worker did not return a result"
    assert process.exitcode == 0
    assert "error" not in result
    assert result["pid"] != parent_pid
    assert result["before"] == {
        "dataset_owner_pid": parent_pid,
        "cache_owner_pid": parent_pid,
        "h5_handles": 1,
        "arrays": 1,
        "validated_shards": 1,
    }
    assert result["dataset_owner_pid"] == result["pid"]
    assert result["cache_owner_pid"] == result["pid"]
    assert result["h5_handles"] == 1
    assert result["arrays"] == 1
    assert result["validated_shards"] == 1
    assert result["inherited_h5_closed"] is True
    assert result["inherited_memmaps_closed"] is True
    expected_shard = gaussian_cache_dir / cache.manifest["shards"][0]["path"]
    assert result["stat_paths"] == [str(expected_shard)]
    assert result["gaussian_shape"] == (2, 13, 2, 2)


def test_metadata_no_hash_rejects_sha_pins_without_calling_hashlib(
    tmp_path, monkeypatch
):
    tmp_path.mkdir(exist_ok=True)
    _forbid_digest_calls(monkeypatch)
    with pytest.raises(ValueError, match="expected SHA-256 pins must be empty"):
        RoboFactoryMultiRobotDataset(
            str(tmp_path),
            integrity_mode="metadata_no_hash",
            gaussian_cache_dir=str(tmp_path / "cache"),
            gaussian_cache_expected_manifest_sha256="e" * 64,
        )


def test_metadata_no_hash_rejects_selection_record_count_mismatch(
    tmp_path, monkeypatch
):
    _write_dataset_inputs(tmp_path)
    gaussian_cache_dir = _write_gaussian_cache(tmp_path, selected_key_count=3)
    _forbid_digest_calls(monkeypatch)
    with pytest.raises(ValueError, match="selected_key_count mismatch"):
        load_manifest(
            gaussian_cache_dir,
            integrity_mode="metadata_no_hash",
        )


def test_metadata_no_hash_primary_cache_uses_on_demand_canonical_fallback(
    tmp_path, monkeypatch
):
    stats_path, text_cache_dir = _write_dataset_inputs(tmp_path)
    primary = _write_gaussian_cache(tmp_path)
    fallback, unused_fallback_shard = _write_canonical_fallback_cache(tmp_path)

    # Make panda-1 absent from the primary manifest while preserving a valid
    # two-frame compact cache.  The second frame belongs to an unrelated key,
    # so the dataset split must neither select nor read it.
    primary_manifest_path = primary / "manifest.json"
    primary_manifest = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
    # Keep a second, unrelated primary stream in the same valid shard.  Panda-1
    # is supplied by the canonical fallback, while the unrelated primary key is
    # never selected by this dataset split.
    unused_stream = dict(primary_manifest["streams"][1])
    unused_stream["trajectory"] = "unused_traj"
    unused_stream["agent_name"] = "panda-9"
    unused_stream["segments"] = [dict(unused_stream["segments"][0])]
    primary_manifest["streams"] = [primary_manifest["streams"][0], unused_stream]
    primary_payload = (
        json.dumps(primary_manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    primary_manifest_path.write_bytes(primary_payload)
    primary_complete_path = primary / "COMPLETE"
    primary_complete = json.loads(primary_complete_path.read_text(encoding="utf-8"))
    primary_complete["manifest_bytes"] = len(primary_payload)
    primary_complete_path.write_text(
        json.dumps(primary_complete, sort_keys=True) + "\n", encoding="utf-8"
    )
    primary_selection = primary / "selection.jsonl"
    first_selection_line = primary_selection.read_text(encoding="utf-8").splitlines()[0]
    unused_selection = {
        "source_path": SOURCE_PATH,
        "trajectory": "unused_traj",
        "timestep": 0,
        "agent_name": "panda-9",
    }
    primary_selection.write_text(
        first_selection_line + "\n" + json.dumps(unused_selection, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # The unused canonical panda-0 shard is deliberately absent. On-access
    # validation must not touch it because the compact primary has that key.
    unused_fallback_shard.unlink()

    def fake_torch_load(path, *, map_location, weights_only):
        assert Path(path) == text_cache_dir / "synthetic.pt"
        return {
            "context": torch.zeros(128, 16),
            "mask": torch.ones(128, dtype=torch.bool),
        }

    monkeypatch.setattr(torch, "load", fake_torch_load)
    _forbid_digest_calls(monkeypatch)

    dataset = RoboFactoryMultiRobotDataset(
        str(tmp_path),
        load_future_video=False,
        video_size=(16, 16),
        val_set_proportion=0.0,
        is_training_set=True,
        randomize_agent_order=False,
        integrity_mode="metadata_no_hash",
        pretrained_norm_stats=str(stats_path),
        text_embedding_cache_dir=str(text_cache_dir),
        text_embedding_cache_files={TASK_NAME: "synthetic.pt"},
        gaussian_cache_dir=str(primary),
        gaussian_cache_verify="manifest",
        gaussian_fallback_cache_dir=str(fallback),
        gaussian_fallback_projection=MOMENT_MATCH_METHOD,
        gaussian_size=(2, 2),
        instruction_map={TASK_NAME: INSTRUCTION},
    )
    assert dataset._gaussian_preflight == {
        "checked_keys": 2,
        "primary_keys": 1,
        "fallback_keys": 1,
        "projection": MOMENT_MATCH_METHOD,
        "primary_shards_validated": 1,
        "primary_shards_declared": 1,
        "fallback_shards_validated": 1,
        "fallback_shards_declared": 2,
    }
    sample = dataset[0]
    assert sample["agent_gaussian"].shape == (2, 13, 2, 2)
    assert sample["agent_ids"].tolist() == [0, 1]
    assert torch.all(sample["agent_gaussian"][0] == 1.0)
    assert torch.all(sample["agent_gaussian"][1, 0:3] == 2.0)
    assert torch.all(sample["agent_gaussian"][1, 12] == 0.5)


def test_canonical_fallback_rejects_legacy_integrity_mode(tmp_path):
    with pytest.raises(ValueError, match="restricted to integrity_mode"):
        RoboFactoryMultiRobotDataset(
            str(tmp_path),
            integrity_mode="legacy_hash",
            gaussian_cache_dir=str(tmp_path / "primary"),
            gaussian_fallback_cache_dir=str(tmp_path / "fallback"),
            gaussian_fallback_projection=MOMENT_MATCH_METHOD,
        )
