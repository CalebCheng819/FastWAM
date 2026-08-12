import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from fastwam.datasets.gaussian_cache import (
    FrameKey,
    load_manifest,
    merge_part_manifests,
    pack_gaussian_channels,
    project_compact_cache,
    sha256_file,
)
from fastwam.datasets.gaussian_cache.orchestrate import (
    OfficialBuildSettings,
    WorkerIdentity,
    _load_teacher_training_provenance,
    make_official_processor,
    merge_and_validate,
    micro_part_roots,
    run_worker,
    verify_micro_part,
)
from fastwam.datasets.gaussian_cache.plan import (
    compact_selection_part_identity,
    create_work_plan,
    micro_part_partition_metadata,
)
from fastwam.datasets.gaussian_cache.selection import normalized_selection_identity

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


def test_teacher_training_provenance_is_checkpoint_bound(tmp_path):
    record = {
        "schema_name": "fastwam_external_teacher_training_provenance",
        "schema_version": 1,
        "checkpoint": {"sha256": "a" * 64},
        "declared_training_datasets": [{"name": "external", "kind": "video"}],
        "overlap_assessment": {
            "declared_dataset_identity_overlap": False,
            "file_level_overlap_audit": "unavailable_teacher_training_file_inventory",
        },
    }
    path = tmp_path / "teacher-provenance.json"
    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")

    sealed = _load_teacher_training_provenance(
        str(path),
        expected_checkpoint_sha256="a" * 64,
    )
    assert sealed["record"] == record
    assert sealed["record_bytes"] == path.stat().st_size
    assert sealed["record_sha256"] == sha256_file(path)

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        _load_teacher_training_provenance(
            str(path),
            expected_checkpoint_sha256="b" * 64,
        )


def _write_source(path: Path, trajectory: str, *, agent_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group(trajectory)
        actions = group.create_group("actions")
        sensors = group.create_group("obs").create_group("sensor_data")
        global_camera = sensors.create_group("head_camera_global")
        global_camera.create_dataset(
            "rgb", data=np.full((2, 240, 320, 3), 127, dtype=np.uint8)
        )
        for index in range(agent_count):
            actions.create_dataset(
                f"panda-{index}", data=np.zeros((1, 8), dtype=np.float32)
            )
            camera = sensors.create_group(f"head_camera_agent{index}")
            camera.create_dataset(
                "rgb", data=np.full((2, 240, 320, 3), index, dtype=np.uint8)
            )


def _write_multi_trajectory_source(
    path: Path,
    trajectories: tuple[str, ...],
    *,
    agent_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for trajectory_name in trajectories:
            group = handle.create_group(trajectory_name)
            actions = group.create_group("actions")
            sensors = group.create_group("obs").create_group("sensor_data")
            sensors.create_group("head_camera_global").create_dataset(
                "rgb", data=np.full((2, 240, 320, 3), 127, dtype=np.uint8)
            )
            for index in range(agent_count):
                actions.create_dataset(
                    f"panda-{index}", data=np.zeros((1, 8), dtype=np.float32)
                )
                sensors.create_group(f"head_camera_agent{index}").create_dataset(
                    "rgb", data=np.full((2, 240, 320, 3), index, dtype=np.uint8)
                )


class _Teacher:
    def __init__(
        self,
        checkpoint_sha256,
        *,
        training_data_provenance=None,
        provenance_overrides=None,
    ):
        self.checkpoint_sha256 = checkpoint_sha256
        self.training_data_provenance = training_data_provenance
        self.provenance_overrides = dict(provenance_overrides or {})
        self.calls = 0

    def provenance(self):
        provenance = {
            "kind": "synthetic-test",
            "repository_commit": "1" * 40,
            "repository_url": "https://example.invalid/policy-lightning.git",
            "config_relative_path": "config/encoder/noposplat.yaml",
            "config_sha256": "6" * 64,
            "checkpoint_sha256": self.checkpoint_sha256,
        }
        if self.training_data_provenance is not None:
            provenance["training_data_provenance"] = self.training_data_provenance
        provenance.update(self.provenance_overrides)
        return provenance

    def encode(self, images):
        self.calls += 1
        batch, views, _, height, width = images.shape
        means = torch.zeros(batch, views, 3, height, width)
        covariance = torch.zeros(batch, views, 3, 3, height, width)
        covariance[:, :, 0, 0] = 1.0
        covariance[:, :, 1, 1] = 1.0
        covariance[:, :, 2, 2] = 1.0
        opacity = torch.full((batch, views, 1, height, width), 0.02)
        return pack_gaussian_channels(means, covariance, opacity).half()


def test_repository_owned_exact_plan_worker_and_coverage_merge(tmp_path):
    dataset_root = tmp_path / "dataset"
    source_a = "N2-rf/motionplanning/a.h5"
    source_b = "N4-rf/motionplanning/b.h5"
    _write_source(dataset_root / source_a, "traj_141", agent_count=2)
    _write_source(dataset_root / source_b, "traj_2", agent_count=4)
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.write_bytes(b"official-checkpoint-fixture")
    checkpoint_sha = sha256_file(checkpoint)
    selection = tmp_path / "compact-selection.jsonl"
    selection.write_text(
        "".join(
            json.dumps(
                {
                    "source_path": source,
                    "trajectory": trajectory,
                    "timestep": timestep,
                    "agent_name": f"panda-{agent}",
                }
            )
            + "\n"
            for source, trajectory, count in (
                (source_a, "traj_141", 2),
                (source_b, "traj_2", 4),
            )
            for timestep in (0, 1)
            for agent in range(count)
        ),
        encoding="utf-8",
    )
    plan_root = tmp_path / "plan"
    teacher_identity = _planned_teacher_identity(checkpoint_sha)
    training_data_provenance = teacher_identity["training_data_provenance"]
    plan = create_work_plan(
        plan_root,
        dataset_root,
        checkpoint,
        expected_checkpoint_sha256=checkpoint_sha,
        planned_worker_count=1,
        teacher_identity=teacher_identity,
        trajectories=[(source_a, "traj_141"), (source_b, "traj_2")],
        compact_selection_jsonl=selection,
        producer_identity=_PRODUCER,
    )
    assert plan["scope"]["mode"] == "exact-trajectories"
    assert len(plan["micro_parts"]) == 2
    assert {source["path"] for source in plan["dataset"]["sources"]} == {
        source_a,
        source_b,
    }
    assert plan["compact_selection"]["raw"]["sha256"] == sha256_file(selection)
    assert plan["compact_selection"]["normalized"]["selected_key_count"] == 12
    assert plan["producer"] == _PRODUCER

    same_keys_different_raw = tmp_path / "same-keys-different-raw.jsonl"
    same_keys_different_raw.write_text(
        "\n" + selection.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="raw byte count differs"):
        run_worker(
            plan_root,
            dataset_root,
            tmp_path / "raw-mismatch-canonical",
            compact_output_root=tmp_path / "raw-mismatch-compact",
            compact_selection_jsonl=same_keys_different_raw,
            staging_dir=tmp_path / "raw-mismatch-staging",
            teacher_factory=lambda _: pytest.fail("raw mismatch must fail before teacher"),
            process_micro_part=make_official_processor(
                OfficialBuildSettings(teacher_repo=tmp_path, teacher_config="unused.yaml")
            ),
            worker=WorkerIdentity("dsw", 0, 1, 0, 0, 0, 1),
            checkpoint_path=checkpoint,
            min_staging_free_bytes=0,
            cuda_binder=lambda _: pytest.fail("raw mismatch must fail before CUDA"),
        )

    missing_agent_selection = tmp_path / "missing-agent-selection.jsonl"
    missing_agent_selection.write_text(
        "".join(
            line
            for line in selection.read_text(encoding="utf-8").splitlines(keepends=True)
            if not (
                json.loads(line)["source_path"] == source_b
                and json.loads(line)["agent_name"] == "panda-3"
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="agent mismatch"):
        create_work_plan(
            tmp_path / "missing-agent-plan",
            dataset_root,
            checkpoint,
            expected_checkpoint_sha256=checkpoint_sha,
            planned_worker_count=1,
            teacher_identity=_planned_teacher_identity(checkpoint_sha),
            trajectories=[(source_a, "traj_141"), (source_b, "traj_2")],
            compact_selection_jsonl=missing_agent_selection,
            producer_identity=_PRODUCER,
        )

    settings = OfficialBuildSettings(
        teacher_repo=tmp_path,
        teacher_config="unused.yaml",
        batch_size=1,
    )
    teachers = []

    def teacher_factory(_):
        teacher = _Teacher(checkpoint_sha)
        teachers.append(teacher)
        return teacher

    worker = WorkerIdentity("dsw", 0, 1, 0, 0, 0, 1)
    with pytest.raises(ValueError, match="config_sha256.*differs from sealed work plan"):
        run_worker(
            plan_root,
            dataset_root,
            tmp_path / "wrong-runtime-config-canonical",
            compact_output_root=tmp_path / "wrong-runtime-config-compact",
            compact_selection_jsonl=selection,
            staging_dir=tmp_path / "wrong-runtime-config-staging",
            teacher_factory=lambda _: _Teacher(
                checkpoint_sha,
                provenance_overrides={"config_sha256": "5" * 64},
            ),
            process_micro_part=make_official_processor(settings),
            worker=worker,
            checkpoint_path=checkpoint,
            min_staging_free_bytes=0,
            cuda_binder=lambda _: torch.device("cpu"),
        )
    with pytest.raises(
        ValueError,
        match="training_data_provenance.*differs from sealed work plan",
    ):
        run_worker(
            plan_root,
            dataset_root,
            tmp_path / "wrong-provenance-canonical",
            compact_output_root=tmp_path / "wrong-provenance-compact",
            compact_selection_jsonl=selection,
            staging_dir=tmp_path / "wrong-provenance-staging",
            teacher_factory=lambda _: _Teacher(
                checkpoint_sha,
                training_data_provenance={"record_sha256": "5" * 64},
            ),
            process_micro_part=make_official_processor(settings),
            worker=worker,
            checkpoint_path=checkpoint,
            min_staging_free_bytes=0,
            cuda_binder=lambda _: torch.device("cpu"),
        )
    canonical_root = tmp_path / "canonical"
    compact_root = tmp_path / "compact"
    result = run_worker(
        plan_root,
        dataset_root,
        canonical_root,
        compact_output_root=compact_root,
        compact_selection_jsonl=selection,
        staging_dir=tmp_path / "staging",
        teacher_factory=teacher_factory,
        process_micro_part=make_official_processor(settings),
        worker=worker,
        checkpoint_path=checkpoint,
        min_staging_free_bytes=0,
        cuda_binder=lambda _: torch.device("cpu"),
    )
    assert result["processed"] == 2
    assert result["teacher_loads"] == 1
    assert len(teachers) == 1
    assert teachers[0].calls == 4

    repeated = run_worker(
        plan_root,
        dataset_root,
        canonical_root,
        compact_output_root=compact_root,
        compact_selection_jsonl=selection,
        staging_dir=tmp_path / "staging",
        teacher_factory=lambda _: (_ for _ in ()).throw(
            AssertionError("verified COMPLETE must skip teacher load")
        ),
        process_micro_part=make_official_processor(settings),
        worker=worker,
        checkpoint_path=checkpoint,
        min_staging_free_bytes=0,
        cuda_binder=lambda _: torch.device("cpu"),
    )
    assert repeated["processed"] == 0
    assert repeated["skipped_verified"] == 2
    assert repeated["teacher_loads"] == 0

    merged = merge_and_validate(
        plan_root,
        dataset_root,
        canonical_root,
        compact_output_root=compact_root,
    )
    assert merged["canonical"]["semantic_mode"] == "coverage"
    assert merged["canonical"]["semantic_parts_covered"] == 2
    assert merged["compact"]["semantic_mode"] == "coverage"
    assert merged["compact"]["semantic_parts_covered"] == 2
    assert (
        load_manifest(canonical_root)["teacher"]["training_data_provenance"]
        == training_data_provenance
    )
    assert (
        load_manifest(compact_root)["teacher"]["training_data_provenance"]
        == training_data_provenance
    )
    assert "plan_identity" not in load_manifest(canonical_root)["selection"]

    # A malicious producer could make every sparse part internally coherent
    # while omitting one selected timestep for all agents.  Agent-completeness
    # coverage alone would pass; the sealed per-part and merged key identities
    # must both reject it.
    malicious_root = tmp_path / "compact-malicious"
    (malicious_root / "parts").mkdir(parents=True)
    malicious_parts = []
    for micro_part in plan["micro_parts"]:
        canonical_part, _ = micro_part_roots(
            canonical_root,
            compact_root,
            micro_part,
        )
        part_index = int(micro_part["part_index"])
        keys = [
            FrameKey(
                str(micro_part["source_path"]),
                str(micro_part["trajectory"]),
                timestep,
                str(agent),
            )
            for timestep in ((0,) if part_index == 0 else (0, 1))
            for agent in micro_part["agent_names"]
        ]
        identity = compact_selection_part_identity(plan, micro_part)
        identity["part"] = {
            "part_index": part_index,
            **normalized_selection_identity(keys),
        }
        output = malicious_root / "parts" / f"part-{part_index:05d}"
        project_compact_cache(
            canonical_part,
            output,
            selection="index",
            selection_keys=keys,
            batch_size=1,
            staging_dir=tmp_path / "staging-malicious",
            partition=micro_part_partition_metadata(plan, micro_part),
            preserve_parent_teacher=True,
            producer=plan["producer"],
            selection_plan_identity=identity,
            derivation={
                "method": "opacity-aware-moment-matching-cell-mean-alpha-v2",
                "output_size": [28, 40],
                "source": "malicious-p0-test",
            },
        )
        malicious_parts.append(output)

    with pytest.raises(ValueError, match="selection plan identity differs"):
        verify_micro_part(
            malicious_parts[0],
            plan=plan,
            micro_part=plan["micro_parts"][0],
            cache_kind="compact",
            verify_shard_checksums=False,
        )
    with pytest.raises(ValueError, match="Merged compact selection key set differs"):
        merge_part_manifests(
            malicious_parts,
            malicious_root,
            canonical_root=canonical_root,
        )


def test_shared_source_is_hashed_once_per_worker_and_once_by_coordinator(
    tmp_path,
    monkeypatch,
):
    dataset_root = tmp_path / "dataset"
    source = "N2-rf/motionplanning/shared.h5"
    trajectories = ("traj_0", "traj_1")
    _write_multi_trajectory_source(
        dataset_root / source,
        trajectories,
        agent_count=2,
    )
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.write_bytes(b"official-checkpoint-fixture")
    checkpoint_sha = sha256_file(checkpoint)
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        "".join(
            json.dumps(
                {
                    "source_path": source,
                    "trajectory": trajectory,
                    "timestep": 1,
                    "agent_name": f"panda-{agent}",
                }
            )
            + "\n"
            for trajectory in trajectories
            for agent in range(2)
        ),
        encoding="utf-8",
    )
    plan_root = tmp_path / "plan"
    create_work_plan(
        plan_root,
        dataset_root,
        checkpoint,
        expected_checkpoint_sha256=checkpoint_sha,
        planned_worker_count=1,
        teacher_identity=_planned_teacher_identity(checkpoint_sha),
        compact_selection_jsonl=selection,
        producer_identity=_PRODUCER,
    )

    import fastwam.datasets.gaussian_cache.orchestrate as orchestrate_module
    import fastwam.datasets.gaussian_cache.validate as validate_module

    worker_source_hashes = []
    original_worker_sha256 = orchestrate_module.sha256_file

    def count_worker_sha256(path, **kwargs):
        if Path(path).suffix == ".h5":
            worker_source_hashes.append(str(Path(path).resolve()))
        return original_worker_sha256(path, **kwargs)

    monkeypatch.setattr(orchestrate_module, "sha256_file", count_worker_sha256)
    canonical_root = tmp_path / "canonical"
    compact_root = tmp_path / "compact"
    result = run_worker(
        plan_root,
        dataset_root,
        canonical_root,
        compact_output_root=compact_root,
        compact_selection_jsonl=selection,
        staging_dir=tmp_path / "staging",
        teacher_factory=lambda _: _Teacher(checkpoint_sha),
        process_micro_part=make_official_processor(
            OfficialBuildSettings(
                teacher_repo=tmp_path,
                teacher_config="unused.yaml",
                batch_size=1,
            )
        ),
        worker=WorkerIdentity("dsw", 0, 1, 0, 0, 0, 1),
        checkpoint_path=checkpoint,
        min_staging_free_bytes=0,
        cuda_binder=lambda _: torch.device("cpu"),
    )
    assert result["processed"] == 2
    assert worker_source_hashes == [str((dataset_root / source).resolve())]

    coordinator_source_hashes = []
    original_validate_sha256 = validate_module.sha256_file

    def count_coordinator_sha256(path, **kwargs):
        if Path(path).suffix == ".h5":
            coordinator_source_hashes.append(str(Path(path).resolve()))
        return original_validate_sha256(path, **kwargs)

    monkeypatch.setattr(validate_module, "sha256_file", count_coordinator_sha256)
    merge_and_validate(
        plan_root,
        dataset_root,
        canonical_root,
        compact_output_root=compact_root,
    )
    assert coordinator_source_hashes == [str((dataset_root / source).resolve())]
