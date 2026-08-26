import json
import os
import pickle
import stat
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from fastwam.datasets.gaussian_cache import (
    FrameKey,
    GaussianCache,
    GaussianCacheBuilder,
    GaussianCacheSchema,
    MissingGaussianFramesError,
    correct_policy_lightning_legacy_covariance_order,
    merge_part_manifests,
    opacity_aware_moment_match,
    pack_gaussian_channels,
    project_compact_cache,
    sha256_file,
    source_record,
    unpack_gaussian_channels,
)
from fastwam.datasets.gaussian_cache.distributed import partition_work_metadata
from fastwam.datasets.gaussian_cache.extract import (
    _read_global_agent_pairs,
    extract_canonical_cache,
)
from fastwam.datasets.gaussian_cache.plan import build_work_plan
from fastwam.datasets.gaussian_cache.teacher import (
    ExternalPolicyLightningTeacher,
    _compose_encoder_config,
)
from fastwam.datasets.gaussian_cache.validate import validate_cache

_PRODUCER = {
    "schema_name": "fastwam-producer-source-snapshot",
    "schema_version": 1,
    "repository_root": "/synthetic/FastWAM",
    "git_commit": "2" * 40,
    "git_tree": "3" * 40,
    "dirty": False,
    "source_snapshot_sha256": "4" * 64,
    "status_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}


def _planned_teacher_identity(checkpoint_sha256):
    return {
        "repository_commit": "1" * 40,
        "repository_url": "https://example.invalid/policy-lightning.git",
        "config_relative_path": "config/encoder/noposplat.yaml",
        "config_sha256": "6" * 64,
        "training_data_provenance": {
            "record": {
                "schema_name": "fastwam_external_teacher_training_provenance",
                "schema_version": 1,
                "checkpoint": {"sha256": checkpoint_sha256},
                "declared_training_datasets": [
                    {"name": "external", "kind": "video"}
                ],
                "declaration_source": {"repository_commit": "3" * 40},
                "overlap_assessment": {
                    "declared_dataset_identity_overlap": False,
                    "file_level_overlap_audit": (
                        "unavailable_teacher_training_file_inventory"
                    ),
                },
            },
            "record_bytes": 123,
            "record_filename": "teacher-training-provenance.json",
            "record_sha256": "4" * 64,
        },
    }


def _frame(value, height, width, *, opacity=0.5):
    means = torch.full((1, 3, height, width), float(value))
    covariance = torch.zeros(1, 3, 3, height, width)
    covariance[:, 0, 0] = 1.0
    covariance[:, 1, 1] = 2.0
    covariance[:, 2, 2] = 3.0
    alpha = torch.full((1, 1, height, width), float(opacity))
    return pack_gaussian_channels(means, covariance, alpha)[0].half()


def _source(tmp_path):
    root = tmp_path / "source"
    root.mkdir(parents=True)
    path = root / "demo.h5"
    path.write_bytes(b"immutable-source")
    return root, path, source_record(path, source_root=root)


def _build_cache(tmp_path, *, height=28, width=40, cache_kind="compact"):
    source_root, _, record = _source(tmp_path)
    cache_root = tmp_path / "cache"
    builder = GaussianCacheBuilder(
        cache_root,
        GaussianCacheSchema(height, width, cache_kind),
        sources=[record],
        teacher={"kind": "synthetic-test", "checkpoint_sha256": "0" * 64},
        selection={"mode": "all", "selected_key_count": 6},
        staging_dir=tmp_path / "staging",
        verify_uploaded_checksum=True,
    )
    for agent_index in range(2):
        frames = torch.stack(
            [_frame(10 * agent_index + timestep, height, width) for timestep in range(3)]
        )
        builder.append_stream(
            source_path="demo.h5",
            trajectory="traj_0",
            agent_name=f"panda-{agent_index}",
            observation_count=3,
            timesteps=[0, 1, 2],
            frames=frames,
        )
    manifest = builder.finish()
    assert list((tmp_path / "staging").iterdir()) == []
    return source_root, cache_root, manifest


def test_policy_lightning_covariance_layout_is_repaired_without_spatial_mixing():
    height, width = 2, 3
    covariance_hw_ij = torch.arange(2 * height * width * 3 * 3).reshape(
        1, 2, height, width, 3, 3
    )
    legacy_covariance = covariance_hw_ij.reshape(1, 2, 9, height, width)
    raw = torch.cat(
        (
            torch.zeros(1, 2, 3, height, width),
            legacy_covariance,
            torch.ones(1, 2, 1, height, width),
        ),
        dim=2,
    )

    corrected = correct_policy_lightning_legacy_covariance_order(raw)
    _, covariance, _ = unpack_gaussian_channels(corrected)
    expected = covariance_hw_ij.permute(0, 1, 4, 5, 2, 3)
    assert torch.equal(covariance, expected)
    assert not torch.equal(legacy_covariance, expected.reshape(1, 2, 9, height, width))


def test_external_teacher_rejects_fp16_overflow_without_clamping():
    teacher = object.__new__(ExternalPolicyLightningTeacher)
    teacher.device = torch.device("cpu")
    raw = torch.zeros(1, 2, 13, 2, 2, dtype=torch.float32)
    raw[:, :, :3] = 70_000.0
    raw[:, :, 12] = 0.5
    teacher._encoder = lambda _: raw
    with pytest.raises(OverflowError, match="max_abs_float32=70000.0"):
        teacher.encode(torch.zeros(1, 2, 3, 2, 2))


def test_external_teacher_hydra_composes_encoder_backbone_defaults(tmp_path):
    repo = tmp_path / "teacher-repo"
    backbone = repo / "config" / "encoder" / "backbone"
    backbone.mkdir(parents=True)
    (backbone / "croco.yaml").write_text("name: croco\nmodel: tiny\n", encoding="utf-8")
    encoder_path = repo / "config" / "encoder" / "noposplat.yaml"
    encoder_path.write_text(
        "defaults:\n  - backbone: croco\nname: noposplat\ncoor_type: self\n",
        encoding="utf-8",
    )

    encoder, provenance = _compose_encoder_config(repo, encoder_path)
    assert encoder.backbone.name == "croco"
    assert encoder.coor_type == "self"
    assert provenance["method"] == "hydra-compose"
    assert len(provenance["composed_encoder_sha256"]) == 64

    unresolved_outside_repo = tmp_path / "unresolved.yaml"
    unresolved_outside_repo.write_text(
        "defaults:\n  - backbone: croco\nname: noposplat\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be inside the pinned repo"):
        _compose_encoder_config(repo, unresolved_outside_repo)


def test_opacity_aware_moment_matching_preserves_mixture_moments():
    means = torch.zeros(1, 3, 1, 2)
    means[:, :, :, 1] = 2.0
    covariance = torch.zeros(1, 3, 3, 1, 2)
    covariance[:, 0, 0] = 1.0
    covariance[:, 1, 1] = 1.0
    covariance[:, 2, 2] = 1.0
    opacity = torch.tensor([[[[0.25, 0.75]]]])
    gaussian = pack_gaussian_channels(means, covariance, opacity)

    compact = opacity_aware_moment_match(gaussian, output_size=(1, 1))
    compact_mean, compact_covariance, compact_opacity = unpack_gaussian_channels(
        compact.float()
    )
    assert compact.dtype == torch.float16
    assert torch.allclose(compact_mean[..., 0, 0], torch.full((1, 3), 1.5), atol=2e-3)
    assert torch.allclose(
        torch.diagonal(compact_covariance[..., 0, 0], dim1=-2, dim2=-1),
        torch.full((1, 3), 1.75),
        atol=2e-3,
    )
    assert float(compact_opacity[0, 0, 0, 0]) == pytest.approx(0.5, abs=2e-3)


def test_compact_moment_matching_rejects_fp16_covariance_overflow():
    means = torch.zeros(1, 3, 1, 2)
    means[:, 0, 0, 0] = -60_000.0
    means[:, 0, 0, 1] = 60_000.0
    covariance = torch.zeros(1, 3, 3, 1, 2)
    opacity = torch.ones(1, 1, 1, 2)
    gaussian = pack_gaussian_channels(means, covariance, opacity)

    with pytest.raises(OverflowError, match="exceed.*FP16 storage range"):
        opacity_aware_moment_match(gaussian, output_size=(1, 1))


def test_real_resolution_cell_mean_alpha_does_not_saturate_with_pool_area():
    height, width = 240, 320
    means = torch.zeros(1, 3, height, width)
    covariance = torch.zeros(1, 3, 3, height, width)
    covariance[:, 0, 0] = 1.0
    covariance[:, 1, 1] = 1.0
    covariance[:, 2, 2] = 1.0
    opacity = torch.full((1, 1, height, width), 0.02)
    gaussian = pack_gaussian_channels(means, covariance, opacity)

    compact = opacity_aware_moment_match(gaussian, output_size=(28, 40))
    _, _, compact_opacity = unpack_gaussian_channels(compact.float())
    # Alpha union over an 8x8-ish cell would exceed 0.7.  Area-normalized
    # density must remain 0.02 even though the cells have non-uniform areas.
    assert tuple(compact_opacity.shape) == (1, 1, 28, 40)
    assert float(compact_opacity.max()) < 0.021
    assert torch.allclose(
        compact_opacity,
        torch.full_like(compact_opacity, 0.02),
        atol=2e-4,
    )


def test_immutable_manifest_reader_preflight_and_pickle(tmp_path):
    source_root, cache_root, manifest = _build_cache(tmp_path)
    assert manifest["total_frames"] == 6
    assert len(manifest["shards"]) == 1
    assert manifest["shards"][0]["final"] is True
    assert (cache_root / "COMPLETE").is_file()
    shard_mode = (cache_root / manifest["shards"][0]["path"]).stat().st_mode
    assert not shard_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)

    cache = GaussianCache.open(cache_root, verify="checksums")
    key = FrameKey("demo.h5", "traj_0", 2, "panda-1")
    assert cache.contains_frame(key)
    assert not cache.contains_frame(FrameKey("demo.h5", "traj_0", 3, "panda-1"))
    assert cache.preflight_keys([key]) == 1
    with pytest.raises(MissingGaussianFramesError, match="missing=1/2"):
        cache.preflight_keys(
            [key, FrameKey("demo.h5", "traj_0", 3, "panda-1")]
        )

    sample = cache.get_agents("demo.h5", "traj_0", 1, ["panda-1", "panda-0"])
    assert set(sample) == {"agent_gaussian"}
    assert sample["agent_gaussian"].shape == (2, 13, 28, 40)
    assert float(sample["agent_gaussian"][0, 0, 0, 0]) == 11.0
    assert float(sample["agent_gaussian"][1, 0, 0, 0]) == 1.0

    restored = pickle.loads(pickle.dumps(cache))
    assert restored._arrays == {}
    assert torch.equal(restored.get_frame(key), cache.get_frame(key))
    result = validate_cache(
        cache_root,
        verify_shard_checksums=True,
        source_root=source_root,
        verify_source_checksums=True,
        semantic_sample_frames=2,
    )
    assert result["total_frames"] == 6
    assert result["source_checksums_verified"] is True


def test_stat_cmp_cache_treats_historical_hashes_as_opaque_and_never_hashes(
    tmp_path, monkeypatch
):
    _, cache_root, _ = _build_cache(tmp_path)
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["sha256"] = "not-a-current-source-digest"
    manifest["shards"][0]["sha256"] = "not-a-current-shard-digest"
    payload = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    complete_path = cache_root / "COMPLETE"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["manifest_bytes"] = len(payload)

    manifest_path.chmod(0o640)
    manifest_path.write_bytes(payload)
    complete_path.chmod(0o640)
    complete_path.write_bytes(
        (json.dumps(complete, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    )

    import fastwam.datasets.gaussian_cache.manifest as manifest_module
    import fastwam.datasets.gaussian_cache.provider as provider_module

    class ForbiddenHashlib:
        @staticmethod
        def sha256(*args, **kwargs):
            del args, kwargs
            raise AssertionError("stat_cmp cache open must not calculate a digest")

    def _forbid_sha256_file(*args, **kwargs):
        del args, kwargs
        raise AssertionError("stat_cmp cache open must not hash cache files")

    monkeypatch.setattr(manifest_module, "hashlib", ForbiddenHashlib)
    monkeypatch.setattr(provider_module, "sha256_file", _forbid_sha256_file)

    cache = GaussianCache.open(cache_root, verify="stat_cmp")
    key = FrameKey("demo.h5", "traj_0", 2, "panda-1")
    assert float(cache.get_frame(key)[0, 0, 0]) == 12.0
    assert cache.manifest["sources"][0]["sha256"] == "not-a-current-source-digest"
    assert cache.manifest["shards"][0]["sha256"] == "not-a-current-shard-digest"
    contract_text = json.dumps(cache.stat_contract, sort_keys=True)
    assert "sha256" not in contract_text.lower()
    assert "digest" not in contract_text.lower()
    assert cache.stat_contract["selected_key_count"] == 6
    assert cache.stat_contract["shard_count"] == 1
    assert cache.stat_contract["file_count"] == 3
    assert all(
        set(record) == {"path", "bytes", "mtime_ns"}
        for record in cache.stat_contract["files"]
    )


def test_stat_cmp_cache_fails_closed_on_shard_byte_count_change(tmp_path):
    _, cache_root, manifest = _build_cache(tmp_path)
    shard_path = cache_root / manifest["shards"][0]["path"]
    shard_path.chmod(0o640)
    with shard_path.open("ab") as shard_file:
        shard_file.write(b"x")

    with pytest.raises(ValueError, match="shard byte count mismatch"):
        GaussianCache.open(cache_root, verify="stat_cmp")


def test_uploaded_shard_gets_exactly_one_strong_readback(tmp_path, monkeypatch):
    import fastwam.datasets.gaussian_cache.manifest as manifest_module

    _, _, source = _source(tmp_path)
    original_sha256_file = manifest_module.sha256_file
    final_readbacks = []

    def counted_sha256(path, **kwargs):
        value = original_sha256_file(path, **kwargs)
        if Path(path).parent.name == "shards":
            final_readbacks.append(str(path))
        return value

    monkeypatch.setattr(manifest_module, "sha256_file", counted_sha256)
    builder = GaussianCacheBuilder(
        tmp_path / "single-readback",
        GaussianCacheSchema(28, 40, "compact"),
        sources=[source],
        teacher={"kind": "synthetic-test", "checkpoint_sha256": "0" * 64},
        selection={"mode": "all", "selected_key_count": 1},
        staging_dir=tmp_path / "staging",
    )
    builder.append_stream(
        source_path="demo.h5",
        trajectory="traj_0",
        agent_name="panda-0",
        observation_count=1,
        timesteps=[0],
        frames=_frame(0, 28, 40).unsqueeze(0),
    )
    builder.finish()
    assert len(final_readbacks) == 1


def test_validation_rejects_shard_or_source_checksum_changes(tmp_path):
    _, cache_root, manifest = _build_cache(tmp_path)
    shard = cache_root / manifest["shards"][0]["path"]
    os.chmod(shard, 0o644)
    with shard.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"\x01\x00")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_cache(cache_root, verify_shard_checksums=True)

    _, cache_root_two, _ = _build_cache(tmp_path / "second")
    source_path = tmp_path / "second" / "source" / "demo.h5"
    source_path.write_bytes(b"changed-source")
    with pytest.raises(ValueError, match="Source HDF5"):
        validate_cache(
            cache_root_two,
            source_root=tmp_path / "second" / "source",
            verify_source_checksums=True,
        )


def test_sparse_compact_projection_stores_only_selected_current_frames(tmp_path):
    _, canonical_root, _ = _build_cache(
        tmp_path,
        height=56,
        width=80,
        cache_kind="canonical",
    )
    selection_path = tmp_path / "selected.jsonl"
    selection_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_path": "demo.h5",
                        "trajectory": "traj_0",
                        "timestep": 1,
                        "agent_names": ["panda-0", "panda-1"],
                    }
                ),
                json.dumps(
                    {
                        "source_path": "demo.h5",
                        "trajectory": "traj_0",
                        "timestep": 2,
                        "agent_name": "panda-1",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    compact_root = tmp_path / "compact"
    manifest = project_compact_cache(
        canonical_root,
        compact_root,
        selection="index",
        selection_jsonl=selection_path,
        batch_size=2,
    )
    assert manifest["schema"]["cache_kind"] == "compact"
    assert manifest["schema"]["height"] == 28
    assert manifest["schema"]["width"] == 40
    assert manifest["total_frames"] == 3
    assert manifest["selection"]["selected_key_count"] == 3
    assert (
        manifest["derivation"]["method"]
        == "opacity-aware-moment-matching-cell-mean-alpha-v2"
    )

    cache = GaussianCache.open(compact_root)
    selected = cache.get_agents(
        "demo.h5", "traj_0", 1, ["panda-1", "panda-0"]
    )["agent_gaussian"]
    assert selected.shape == (2, 13, 28, 40)
    assert not cache.contains_frame(FrameKey("demo.h5", "traj_0", 0, "panda-0"))
    validate_cache(compact_root, semantic_sample_frames=3)


class _SyntheticTeacher:
    def __init__(self):
        self.calls = []

    def provenance(self):
        return {
            "kind": "synthetic-test",
            "repository_commit": "1" * 40,
            "repository_url": "https://example.invalid/policy-lightning.git",
            "config_relative_path": "config/encoder/noposplat.yaml",
            "config_sha256": "6" * 64,
            "checkpoint_sha256": "2" * 64,
            "legacy_covariance_layout_corrected": True,
        }

    def encode(self, images):
        self.calls.append(tuple(images.shape))
        batch, views, _, height, width = images.shape
        frames = []
        for view in range(views):
            frame = _frame(view + 1, height, width)
            frames.append(frame.expand(batch, -1, -1, -1))
        return torch.stack(frames, dim=1)


class _PlannedSyntheticTeacher(_SyntheticTeacher):
    def __init__(self, checkpoint_sha256):
        super().__init__()
        self.checkpoint_sha256 = checkpoint_sha256

    def provenance(self):
        return {
            **super().provenance(),
            "checkpoint_sha256": self.checkpoint_sha256,
        }


@pytest.mark.parametrize(
    "agent_names",
    [
        ["panda-0"],
        ["panda-1", "panda-0"],
        ["panda-2", "panda-0", "panda-1"],
        ["panda-3", "panda-1", "panda-0", "panda-2"],
    ],
)
def test_teacher_pairs_are_global_agent_only_and_preserve_agent_order(
    tmp_path, agent_names
):
    path = tmp_path / f"pairs-{len(agent_names)}.h5"
    with h5py.File(path, "w") as handle:
        trajectory = handle.create_group("traj_0")
        sensors = trajectory.create_group("obs").create_group("sensor_data")
        global_camera = sensors.create_group("head_camera_global")
        global_camera.create_dataset(
            "rgb", data=np.full((2, 240, 320, 3), 101, dtype=np.uint8)
        )
        for index in range(4):
            camera = sensors.create_group(f"head_camera_agent{index}")
            camera.create_dataset(
                "rgb",
                data=np.full((2, 240, 320, 3), 10 + index, dtype=np.uint8),
            )
        pairs = _read_global_agent_pairs(trajectory, agent_names, [0, 1])

    restored = ((pairs + 1.0) * 127.5).round().to(torch.uint8)
    restored = restored.reshape(2, len(agent_names), 2, 3, 240, 320)
    assert restored.shape[:3] == (2, len(agent_names), 2)
    assert torch.all(restored[:, :, 0] == 101)
    for position, name in enumerate(agent_names):
        index = int(name.rsplit("-", 1)[-1])
        assert torch.all(restored[:, position, 1] == 10 + index)


def test_teacher_pair_uses_single_robot_head_camera_as_deterministic_self_pair(tmp_path):
    path = tmp_path / "single-table.h5"
    with h5py.File(path, "w") as handle:
        trajectory = handle.create_group("traj_0")
        trajectory.create_dataset("actions", data=np.zeros((1, 8), dtype=np.float32))
        camera = (
            trajectory.create_group("obs")
            .create_group("sensor_data")
            .create_group("head_camera")
        )
        camera.create_dataset(
            "rgb", data=np.full((2, 240, 320, 3), 73, dtype=np.uint8)
        )
        pairs = _read_global_agent_pairs(trajectory, ["panda-0"], [0, 1])

    restored = ((pairs + 1.0) * 127.5).round().to(torch.uint8)
    assert restored.shape == (2, 2, 3, 240, 320)
    assert torch.all(restored == 73)


def test_canonical_extraction_keeps_every_agent_observation_timestep(tmp_path):
    dataset_root = tmp_path / "dataset"
    task = dataset_root / "SyntheticTwoRobot-rf" / "motionplanning"
    task.mkdir(parents=True)
    hdf5_path = task / "demo.h5"
    with h5py.File(hdf5_path, "w") as handle:
        trajectory = handle.create_group("traj_0")
        actions = trajectory.create_group("actions")
        sensors = trajectory.create_group("obs").create_group("sensor_data")
        global_camera = sensors.create_group("head_camera_global")
        global_camera.create_dataset(
            "rgb",
            data=np.full((3, 240, 320, 3), 127, dtype=np.uint8),
        )
        for index in range(2):
            actions.create_dataset(f"panda-{index}", data=np.zeros((2, 8), dtype=np.float32))
            camera = sensors.create_group(f"head_camera_agent{index}")
            camera.create_dataset(
                "rgb",
                data=np.full((3, 240, 320, 3), index, dtype=np.uint8),
            )

    cache_root = tmp_path / "canonical"
    manifest = extract_canonical_cache(
        dataset_root,
        cache_root,
        teacher=_SyntheticTeacher(),
        selection="all",
        batch_size=2,
    )
    assert manifest["total_frames"] == 6
    assert (
        manifest["teacher"]["pairing"]
        == "robofactory_reference_agent_or_self_unify_v1"
    )
    assert manifest["teacher"]["config_overrides"]["coor_type"] == "unify"
    assert {record["stored_count"] for record in manifest["streams"]} == {3}
    assert {record["observation_count"] for record in manifest["streams"]} == {3}
    cache = GaussianCache.open(cache_root)
    assert cache.get_agents(
        "SyntheticTwoRobot-rf/motionplanning/demo.h5",
        "traj_0",
        2,
        ["panda-0", "panda-1"],
    )["agent_gaussian"].shape == (2, 13, 240, 320)
    validate_cache(
        cache_root,
        source_root=dataset_root,
        verify_source_checksums=True,
        semantic_sample_frames=2,
    )


def _write_single_agent_hdf5(path, *, trajectory_names=("traj_0",), pixel_value=20):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for trajectory_index, trajectory_name in enumerate(trajectory_names):
            trajectory = handle.create_group(trajectory_name)
            actions = trajectory.create_group("actions")
            actions.create_dataset("panda-0", data=np.zeros((1, 8), dtype=np.float32))
            sensors = trajectory.create_group("obs").create_group("sensor_data")
            for camera_name, value in (
                ("head_camera_global", 127),
                ("head_camera_agent0", pixel_value + trajectory_index),
            ):
                camera = sensors.create_group(camera_name)
                camera.create_dataset(
                    "rgb",
                    data=np.full((2, 240, 320, 3), value, dtype=np.uint8),
                )


def test_sealed_work_plan_extracts_exact_micro_part_without_dataset_rediscovery(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "dataset"
    first_source = "TaskA-rf/motionplanning/a.h5"
    second_source = "TaskB-rf/motionplanning/b.h5"
    _write_single_agent_hdf5(dataset_root / first_source, trajectory_names=("traj_a",))
    _write_single_agent_hdf5(dataset_root / second_source, trajectory_names=("traj_b",))
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.write_bytes(b"pinned-checkpoint")
    checkpoint_sha = sha256_file(checkpoint)
    plan = build_work_plan(
        dataset_root,
        checkpoint,
        expected_checkpoint_sha256=checkpoint_sha,
        planned_worker_count=1,
        teacher_identity=_planned_teacher_identity(checkpoint_sha),
        producer_identity=_PRODUCER,
    )
    micro_part = next(
        item
        for item in plan["micro_parts"]
        if (item["source_path"], item["trajectory"]) == (second_source, "traj_b")
    )

    # Plan-aware extraction must use the coordinator-sealed source record and
    # must never fall back to source_record() over every HDF5 in dataset_root.
    monkeypatch.setattr(
        "fastwam.datasets.gaussian_cache.extract.source_record",
        lambda *args, **kwargs: pytest.fail("unexpected dataset rediscovery"),
    )
    monkeypatch.setattr(
        "fastwam.datasets.gaussian_cache.plan.stable_file_identity",
        lambda *args, **kwargs: pytest.fail("unexpected per-micro-part SHA-256"),
    )
    assigned_stat = (dataset_root / second_source).stat()
    output_root = tmp_path / "parts" / f"part-{micro_part['part_index']:05d}"
    manifest = extract_canonical_cache(
        dataset_root,
        output_root,
        teacher=_PlannedSyntheticTeacher(checkpoint_sha),
        selection="all",
        batch_size=1,
        work_plan=plan,
        micro_part_index=int(micro_part["part_index"]),
        preverified_source_state=(
            int(assigned_stat.st_size),
            int(assigned_stat.st_mtime_ns),
        ),
    )
    assert manifest["partition"]["partition_count"] == 2
    assert manifest["partition"]["assigned_units"] == [
        {
            "source_path": second_source,
            "trajectory": "traj_b",
            "observation_count": 2,
            "agent_names": ["panda-0"],
            "weight": 2,
        }
    ]
    assert {stream["trajectory"] for stream in manifest["streams"]} == {"traj_b"}
    assert [record["path"] for record in manifest["sources"]] == [second_source]


def test_two_part_trajectory_merge_and_same_forward_compact_are_zero_copy(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_path = "TaskA-rf/motionplanning/demo.h5"
    trajectory_names = ["traj_0", "traj_1"]
    _write_single_agent_hdf5(
        dataset_root / source_path,
        trajectory_names=trajectory_names,
    )
    compact_selection = tmp_path / "compact-selection.jsonl"
    compact_selection.write_text(
        "".join(
            json.dumps(
                {
                    "source_path": source_path,
                    "trajectory": trajectory_name,
                    "timestep": 1,
                    "agent_name": "panda-0",
                }
            )
            + "\n"
            for trajectory_name in trajectory_names
        ),
        encoding="utf-8",
    )

    merged_root = tmp_path / "merged"
    parts_root = merged_root / "parts"
    parts_root.mkdir(parents=True)
    compact_root = tmp_path / "compact-merged"
    compact_parts_root = compact_root / "parts"
    compact_parts_root.mkdir(parents=True)
    part_roots = []
    compact_part_roots = []
    for index in range(2):
        part_root = parts_root / f"part-{index:05d}"
        compact_part_root = compact_parts_root / f"part-{index:05d}"
        teacher = _SyntheticTeacher()
        extract_canonical_cache(
            dataset_root,
            part_root,
            teacher=teacher,
            selection="all",
            batch_size=1,
            staging_dir=tmp_path / "staging",
            verify_uploaded_checksum=True,
            partition_index=index,
            partition_count=2,
            partition_unit="trajectory",
            compact_output_root=compact_part_root,
            compact_selection_jsonl=compact_selection,
        )
        # Two observations are encoded once each; compact uses those same
        # in-memory outputs and must not trigger additional teacher forwards.
        assert len(teacher.calls) == 2
        part_roots.append(part_root)
        compact_part_roots.append(compact_part_root)

    part_manifest_sha256 = {
        str(path): sha256_file(path / "manifest.json") for path in part_roots
    }

    def fail_before_complete(_root, _complete):
        raise RuntimeError("injected failure before COMPLETE")

    with pytest.raises(RuntimeError, match="injected failure before COMPLETE"):
        merge_part_manifests(
            part_roots,
            merged_root,
            verify_part_checksums=True,
            before_complete=fail_before_complete,
        )
    assert (merged_root / "MERGE.BUILDING.json").is_file()
    assert (merged_root / "manifest.json").is_file()
    assert not (merged_root / "COMPLETE").exists()
    assert all(path.is_dir() for path in part_roots)
    assert {
        str(path): sha256_file(path / "manifest.json") for path in part_roots
    } == part_manifest_sha256

    def fail_after_complete(_root, _complete):
        raise RuntimeError("injected failure after COMPLETE")

    with pytest.raises(RuntimeError, match="injected failure after COMPLETE"):
        merge_part_manifests(
            part_roots,
            merged_root,
            verify_part_checksums=True,
            after_complete=fail_after_complete,
        )
    assert (merged_root / "MERGE.BUILDING.json").is_file()
    assert (merged_root / "manifest.json").is_file()
    assert (merged_root / "COMPLETE").is_file()
    assert {
        str(path): sha256_file(path / "manifest.json") for path in part_roots
    } == part_manifest_sha256

    manifest_sha256 = sha256_file(merged_root / "manifest.json")
    complete_sha256 = sha256_file(merged_root / "COMPLETE")
    manifest = merge_part_manifests(
        part_roots,
        merged_root,
        verify_part_checksums=True,
    )
    assert not (merged_root / "MERGE.BUILDING.json").exists()
    assert sha256_file(merged_root / "manifest.json") == manifest_sha256
    assert sha256_file(merged_root / "COMPLETE") == complete_sha256
    assert manifest["manifest_version"] == 2
    assert manifest["partition"]["unit"] == "trajectory"
    assert len(manifest["parts"]) == 2
    assert len(manifest["shards"]) == 2
    assert len(manifest["sources"]) == 1
    assert all(record["final"] for record in manifest["shards"])
    assert all(record["path"].startswith("parts/part-") for record in manifest["shards"])
    assert not (merged_root / "shards").exists()
    assert manifest["total_frames"] == 4

    cache = GaussianCache.open(merged_root, verify="checksums")
    for trajectory_name in trajectory_names:
        key = FrameKey(source_path, trajectory_name, 1, "panda-0")
        assert cache.contains_frame(key)
        assert cache.get_frame(key).shape == (13, 240, 320)
    cache.preflight_keys(
        FrameKey(source_path, trajectory_name, timestep, "panda-0")
        for trajectory_name in trajectory_names
        for timestep in (0, 1)
    )
    result = validate_cache(
        merged_root,
        verify_shard_checksums=True,
        source_root=dataset_root,
        verify_source_checksums=True,
        semantic_mode="coverage",
    )
    assert result["parts"] == 2
    assert result["total_frames"] == 4
    assert result["semantic_shards_covered"] == 2
    assert result["semantic_parts_covered"] == 2

    lru_cache = GaussianCache.open(merged_root, max_open_shards=1)
    lru_cache.get_frame(FrameKey(source_path, "traj_0", 0, "panda-0"))
    assert len(lru_cache._arrays) == 1
    first_open = next(iter(lru_cache._arrays))
    lru_cache.get_frame(FrameKey(source_path, "traj_1", 0, "panda-0"))
    assert len(lru_cache._arrays) == 1
    assert next(iter(lru_cache._arrays)) != first_open
    lru_cache.close()

    compact_manifest = merge_part_manifests(
        compact_part_roots,
        compact_root,
        verify_part_checksums=True,
        canonical_root=merged_root,
    )
    assert compact_manifest["total_frames"] == 2
    assert compact_manifest["schema"]["cache_kind"] == "compact"
    assert compact_manifest["derivation"]["parent_manifest_sha256"]
    assert compact_manifest["derivation"]["parent_selection"]["mode"] == "all"
    assert compact_manifest["selection"]["mode"] == "index"
    assert not (compact_root / "shards").exists()
    compact_cache = GaussianCache.open(compact_root, verify="checksums")
    for trajectory_name in trajectory_names:
        key = FrameKey(source_path, trajectory_name, 1, "panda-0")
        assert compact_cache.get_frame(key).shape == (13, 28, 40)
        assert not compact_cache.contains_frame(
            FrameKey(source_path, trajectory_name, 0, "panda-0")
        )
    validate_cache(compact_root, semantic_sample_frames=2)


def test_direct_compact_parts_merge_without_canonical_cache(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_path = "Task-rf/motionplanning/demo.h5"
    trajectory_names = ("traj_0", "traj_1")
    _write_single_agent_hdf5(
        dataset_root / source_path,
        trajectory_names=trajectory_names,
    )
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        "".join(
            json.dumps(
                {
                    "source_path": source_path,
                    "trajectory": trajectory_name,
                    "timestep": 1,
                    "agent_name": "panda-0",
                },
                sort_keys=True,
            )
            + "\n"
            for trajectory_name in trajectory_names
        ),
        encoding="utf-8",
    )

    merged_root = tmp_path / "direct-compact"
    parts_root = merged_root / "parts"
    parts_root.mkdir(parents=True)
    part_roots = []
    for index in range(2):
        part_root = parts_root / f"part-{index:05d}"
        manifest = extract_canonical_cache(
            dataset_root,
            part_root,
            teacher=_SyntheticTeacher(),
            selection="index",
            selection_jsonl=selection,
            direct_compact=True,
            batch_size=1,
            partition_index=index,
            partition_count=2,
            partition_unit="trajectory",
        )
        assert manifest["schema"]["cache_kind"] == "compact"
        assert manifest["schema"]["height"] == 28
        assert manifest["schema"]["width"] == 40
        assert manifest["derivation"]["source"] == "direct-teacher-forward-index-v1"
        assert manifest["total_frames"] == 1
        part_roots.append(part_root)

    manifest = merge_part_manifests(
        part_roots,
        merged_root,
        verify_part_checksums=True,
    )
    assert manifest["total_frames"] == 2
    assert manifest["selection"]["mode"] == "index"
    assert manifest["derivation"] == {
        "method": "opacity-aware-moment-matching-cell-mean-alpha-v2",
        "output_size": [28, 40],
        "source": "direct-teacher-forward-index-v1",
    }
    assert "parent_manifest_sha256" not in manifest["derivation"]

    cache = GaussianCache.open(merged_root, verify="checksums")
    for trajectory_name in trajectory_names:
        key = FrameKey(source_path, trajectory_name, 1, "panda-0")
        assert cache.get_frame(key).shape == (13, 28, 40)
        assert not cache.contains_frame(
            FrameKey(source_path, trajectory_name, 0, "panda-0")
        )
    validate_cache(
        merged_root,
        source_root=dataset_root,
        verify_source_checksums=True,
        semantic_mode="coverage",
    )


def test_merge_rejects_sealed_part_missing_one_assigned_agent(tmp_path):
    _, _, source = _source(tmp_path)
    merged_root = tmp_path / "missing-agent-merged"
    part_root = merged_root / "parts" / "part-00000"
    part_root.parent.mkdir(parents=True)
    units = [
        {
            "source_path": "demo.h5",
            "trajectory": "traj_0",
            "observation_count": 3,
            "agent_names": ["panda-0", "panda-1"],
            "weight": 6,
        }
    ]
    _, partition = partition_work_metadata(
        units,
        partition_index=0,
        partition_count=1,
        unit="trajectory",
    )
    builder = GaussianCacheBuilder(
        part_root,
        GaussianCacheSchema(28, 40, "compact"),
        sources=[source],
        teacher={"kind": "synthetic-test", "checkpoint_sha256": "0" * 64},
        selection={"mode": "all", "selected_key_count": 6},
        partition=partition,
        staging_dir=tmp_path / "staging",
    )
    builder.append_stream(
        source_path="demo.h5",
        trajectory="traj_0",
        agent_name="panda-0",
        observation_count=3,
        timesteps=[0, 1, 2],
        frames=torch.stack([_frame(index, 28, 40) for index in range(3)]),
    )
    builder.finish()

    with pytest.raises(ValueError, match="agent completeness mismatch"):
        merge_part_manifests([part_root], merged_root)
    with pytest.raises(ValueError, match="agent completeness mismatch"):
        validate_cache(part_root, semantic_mode="none")
