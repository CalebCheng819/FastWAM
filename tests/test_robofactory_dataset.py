import hashlib
import json

import h5py
import numpy as np
import pytest
import torch

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.gaussian_cache import (
    FrameKey,
    GaussianCacheBuilder,
    GaussianCacheSchema,
    load_manifest,
    sha256_file,
    source_record,
)
from fastwam.datasets.gaussian_cache.selection import write_normalized_selection_index
from fastwam.datasets.robofactory_multi_robot import (
    RoboFactoryMultiRobotDataset,
    compute_robofactory_stats,
    gaussian_source_identity_sha256,
)


def _write_compact_gaussian_cache(
    root,
    *,
    task_name: str,
    num_agents: int,
    included_agents: list[int] | None = None,
    nonfinite_agent: int | None = None,
    selection_mode: str = "all",
):
    h5_path = root / task_name / "motionplanning" / "demo.h5"
    schema = GaussianCacheSchema(height=28, width=40, cache_kind="compact")
    cache_root = root / "gaussian-cache"
    builder = GaussianCacheBuilder(
        cache_root,
        schema,
        sources=[source_record(h5_path, source_root=root)],
        teacher={"kind": "synthetic-test-teacher"},
        selection={"mode": "all", "selected_key_count": num_agents},
        target_shard_bytes=(1 << 30) + schema.frame_bytes,
    )
    source_path = h5_path.relative_to(root).as_posix()
    agents = range(num_agents) if included_agents is None else included_agents
    agents = list(agents)
    if selection_mode == "index":
        builder.selection = write_normalized_selection_index(
            cache_root,
            [
                FrameKey(source_path, "traj_0", 0, f"panda-{agent_idx}")
                for agent_idx in agents
            ],
        )
    elif selection_mode != "all":
        raise ValueError(f"Unsupported test selection_mode={selection_mode!r}")
    for agent_idx in agents:
        frame = torch.full(
            (1, 13, 28, 40),
            float(agent_idx + 1),
            dtype=torch.float16,
        )
        if agent_idx == nonfinite_agent:
            frame[0, 0, 0, 0] = float("nan")
        builder.append_stream(
            source_path=source_path,
            trajectory="traj_0",
            agent_name=f"panda-{agent_idx}",
            observation_count=41,
            timesteps=[0],
            frames=frame,
        )
    builder.finish()
    return cache_root


def _write_demo(
    root,
    *,
    task_name: str,
    num_agents: int,
    instruction: str,
    write_cache: bool = True,
):
    task_dir = root / task_name / "motionplanning"
    task_dir.mkdir(parents=True)
    h5_path = task_dir / "demo.h5"
    length = 40
    with h5py.File(h5_path, "w") as handle:
        trajectory = handle.create_group("traj_0")
        actions = trajectory.create_group("actions")
        obs = trajectory.create_group("obs")
        agents = obs.create_group("agent")
        sensors = obs.create_group("sensor_data")
        articulations = trajectory.create_group("env_states").create_group("articulations")
        global_camera = sensors.create_group("head_camera_global")
        global_camera.create_dataset(
            "rgb",
            data=np.random.default_rng(1).integers(
                0, 256, size=(length + 1, 24, 32, 3), dtype=np.uint8
            ),
        )
        for agent_idx in range(num_agents):
            name = f"panda-{agent_idx}"
            actions.create_dataset(
                name,
                data=np.full((length, 8), float(agent_idx + 1), dtype=np.float32),
            )
            agent = agents.create_group(name)
            agent.create_dataset("qpos", data=np.ones((length + 1, 9), dtype=np.float32))
            agent.create_dataset("qvel", data=np.zeros((length + 1, 9), dtype=np.float32))
            articulation = np.zeros((length + 1, 31), dtype=np.float32)
            articulation[:, 0] = float(agent_idx)
            if agent_idx % 2:
                articulation[:, 6] = -2.0
            else:
                articulation[:, 3] = 1.0
            articulation[:, 13:22] = 1.0
            articulations.create_dataset(f"panda-agent-{agent_idx}", data=articulation)

    stats_path = root / "stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "action": {"mean": [0.0] * 8, "std": [1.0] * 8},
                "state": {"mean": [0.0] * 18, "std": [1.0] * 18},
            }
        )
    )
    cache_dir = root / "cache"
    cache_dir.mkdir(exist_ok=True)
    prompt = DEFAULT_PROMPT.format(task=instruction)
    cache_path = cache_dir / (
        hashlib.sha256(prompt.encode()).hexdigest() + ".t5_len128.wan22ti2v5b.pt"
    )
    if write_cache:
        torch.save(
            {"context": torch.zeros(128, 16), "mask": torch.ones(128, dtype=torch.bool)},
            cache_path,
        )
    return stats_path, cache_dir


@pytest.mark.parametrize("num_agents", [1, 2, 3, 4])
def test_robofactory_hdf5_adapter_returns_native_agent_axis(tmp_path, num_agents):
    task_name = f"Synthetic{num_agents}RobotTask-rf"
    instruction = f"{num_agents} robots complete the synthetic task"
    stats_path, cache_dir = _write_demo(
        tmp_path,
        task_name=task_name,
        num_agents=num_agents,
        instruction=instruction,
    )

    dataset = RoboFactoryMultiRobotDataset(
        str(tmp_path),
        video_size=(32, 32),
        window_stride=16,
        val_set_proportion=0.0,
        is_training_set=True,
        randomize_agent_order=False,
        pretrained_norm_stats=str(stats_path),
        text_embedding_cache_dir=str(cache_dir),
        instruction_map={task_name: instruction},
    )
    sample = dataset[0]
    assert dataset.agent_counts == (num_agents,)
    assert dataset.get_agent_count(0) == num_agents
    assert dataset.task_ids == (task_name,)
    assert dataset.get_task_id(0) == task_name
    assert sample["video"].shape == (3, 9, 32, 32)
    assert sample["action"].shape == (num_agents, 32, 8)
    assert sample["agent_state"].shape == (num_agents, 18)
    assert sample["agent_geometry"].shape == (num_agents, 7)
    assert sample["agent_geometry"][:, 0].tolist() == list(map(float, range(num_agents)))
    assert sample["agent_geometry"][0, 3] == 1
    if num_agents > 1:
        assert sample["agent_geometry"][1, 6] == 1
    assert torch.allclose(
        torch.linalg.vector_norm(sample["agent_geometry"][:, 3:7], dim=-1),
        torch.ones(num_agents),
    )
    assert sample["action_is_pad"].shape == (num_agents, 32)
    assert sample["agent_ids"].tolist() == list(range(num_agents))
    assert sample["agent_count"] == num_agents
    assert "agent_mask" not in sample
    if num_agents > 1:
        assert torch.allclose(sample["action"][1], torch.full((32, 8), 2.0))
    assert sample["context"].shape == (128, 16)


def test_action_only_dataset_reads_only_observation_frame(tmp_path):
    task_name = "Synthetic2RobotTask-rf"
    instruction = "two robots complete the synthetic task"
    stats_path, cache_dir = _write_demo(
        tmp_path,
        task_name=task_name,
        num_agents=2,
        instruction=instruction,
    )
    dataset = RoboFactoryMultiRobotDataset(
        str(tmp_path),
        load_future_video=False,
        video_size=(32, 32),
        val_set_proportion=0.0,
        is_training_set=True,
        randomize_agent_order=False,
        pretrained_norm_stats=str(stats_path),
        text_embedding_cache_dir=str(cache_dir),
        instruction_map={task_name: instruction},
    )

    sample = dataset[0]
    assert sample["video"].shape == (3, 1, 32, 32)
    assert sample["action"].shape == (2, 32, 8)
    assert sample["image_is_pad"].shape == (1,)


@pytest.mark.parametrize("num_agents", [1, 2, 3, 4])
def test_compact_gaussian_cache_preserves_native_agent_axis(
    tmp_path,
    num_agents,
):
    task_name = f"Synthetic{num_agents}RobotTask-rf"
    instruction = f"{num_agents} robots complete the synthetic task"
    stats_path, text_cache_dir = _write_demo(
        tmp_path,
        task_name=task_name,
        num_agents=num_agents,
        instruction=instruction,
    )
    gaussian_cache_dir = _write_compact_gaussian_cache(
        tmp_path,
        task_name=task_name,
        num_agents=num_agents,
    )
    dataset = RoboFactoryMultiRobotDataset(
        str(tmp_path),
        load_future_video=False,
        video_size=(32, 32),
        val_set_proportion=0.0,
        is_training_set=True,
        randomize_agent_order=False,
        pretrained_norm_stats=str(stats_path),
        text_embedding_cache_dir=str(text_cache_dir),
        gaussian_cache_dir=str(gaussian_cache_dir),
        instruction_map={task_name: instruction},
    )

    sample = dataset[0]
    assert sample["agent_gaussian"].shape == (num_agents, 13, 28, 40)
    assert sample["agent_gaussian"].dtype == torch.float16
    assert sample["agent_gaussian"][:, 0, 0, 0].tolist() == list(
        map(float, range(1, num_agents + 1))
    )


def test_compact_gaussian_cache_follows_deterministic_epoch_agent_order(tmp_path):
    task_name = "Synthetic3RobotTask-rf"
    instruction = "three robots complete the synthetic task"
    stats_path, text_cache_dir = _write_demo(
        tmp_path,
        task_name=task_name,
        num_agents=3,
        instruction=instruction,
    )
    gaussian_cache_dir = _write_compact_gaussian_cache(
        tmp_path,
        task_name=task_name,
        num_agents=3,
    )
    dataset = RoboFactoryMultiRobotDataset(
        str(tmp_path),
        load_future_video=False,
        video_size=(32, 32),
        val_set_proportion=0.0,
        is_training_set=True,
        randomize_agent_order=True,
        pretrained_norm_stats=str(stats_path),
        text_embedding_cache_dir=str(text_cache_dir),
        gaussian_cache_dir=str(gaussian_cache_dir),
        instruction_map={task_name: instruction},
    )

    dataset.set_epoch(7)
    torch.manual_seed(1)
    first = dataset[0]
    torch.manual_seed(999999)
    resumed = dataset[0]
    assert torch.equal(first["agent_ids"], resumed["agent_ids"])
    assert torch.equal(first["action"], resumed["action"])
    assert torch.equal(first["agent_gaussian"], resumed["agent_gaussian"])
    expected_gaussian_ids = (first["agent_ids"] + 1).tolist()
    assert first["agent_gaussian"][:, 0, 0, 0].tolist() == expected_gaussian_ids

    # Reconstructing the dataset and restoring only the epoch reproduces the
    # same sample permutation, which is the exact-resume contract.
    reconstructed = RoboFactoryMultiRobotDataset(
        str(tmp_path),
        load_future_video=False,
        video_size=(32, 32),
        val_set_proportion=0.0,
        is_training_set=True,
        randomize_agent_order=True,
        pretrained_norm_stats=str(stats_path),
        text_embedding_cache_dir=str(text_cache_dir),
        gaussian_cache_dir=str(gaussian_cache_dir),
        instruction_map={task_name: instruction},
    )
    reconstructed.set_epoch(7)
    assert torch.equal(first["agent_ids"], reconstructed[0]["agent_ids"])


def test_gaussian_cache_manifest_and_source_identities_are_pinned(tmp_path):
    task_name = "Synthetic2RobotTask-rf"
    instruction = "two robots complete the synthetic task"
    stats_path, text_cache_dir = _write_demo(
        tmp_path,
        task_name=task_name,
        num_agents=2,
        instruction=instruction,
    )
    gaussian_cache_dir = _write_compact_gaussian_cache(
        tmp_path,
        task_name=task_name,
        num_agents=2,
        selection_mode="index",
    )
    manifest = load_manifest(gaussian_cache_dir)
    manifest_sha256 = sha256_file(gaussian_cache_dir / "manifest.json")
    selection_sha256 = manifest["selection"]["index_sha256"]
    source_identity_sha256 = gaussian_source_identity_sha256(manifest["sources"])

    dataset = RoboFactoryMultiRobotDataset(
        str(tmp_path),
        load_future_video=False,
        video_size=(32, 32),
        val_set_proportion=0.0,
        is_training_set=True,
        randomize_agent_order=False,
        pretrained_norm_stats=str(stats_path),
        text_embedding_cache_dir=str(text_cache_dir),
        gaussian_cache_dir=str(gaussian_cache_dir),
        gaussian_cache_expected_manifest_sha256=manifest_sha256,
        gaussian_cache_expected_selection_sha256=selection_sha256,
        gaussian_cache_expected_source_identity_sha256=source_identity_sha256,
        instruction_map={task_name: instruction},
    )
    assert dataset[0]["agent_gaussian"].shape == (2, 13, 28, 40)

    with pytest.raises(ValueError, match="manifest identity mismatch"):
        RoboFactoryMultiRobotDataset(
            str(tmp_path),
            load_future_video=False,
            video_size=(32, 32),
            val_set_proportion=0.0,
            is_training_set=True,
            randomize_agent_order=False,
            pretrained_norm_stats=str(stats_path),
            text_embedding_cache_dir=str(text_cache_dir),
            gaussian_cache_dir=str(gaussian_cache_dir),
            gaussian_cache_expected_manifest_sha256="0" * 64,
            instruction_map={task_name: instruction},
        )
    with pytest.raises(ValueError, match="selection identity mismatch"):
        RoboFactoryMultiRobotDataset(
            str(tmp_path),
            load_future_video=False,
            video_size=(32, 32),
            val_set_proportion=0.0,
            is_training_set=True,
            randomize_agent_order=False,
            pretrained_norm_stats=str(stats_path),
            text_embedding_cache_dir=str(text_cache_dir),
            gaussian_cache_dir=str(gaussian_cache_dir),
            gaussian_cache_expected_selection_sha256="0" * 64,
            instruction_map={task_name: instruction},
        )
    with pytest.raises(ValueError, match="source identity mismatch"):
        RoboFactoryMultiRobotDataset(
            str(tmp_path),
            load_future_video=False,
            video_size=(32, 32),
            val_set_proportion=0.0,
            is_training_set=True,
            randomize_agent_order=False,
            pretrained_norm_stats=str(stats_path),
            text_embedding_cache_dir=str(text_cache_dir),
            gaussian_cache_dir=str(gaussian_cache_dir),
            gaussian_cache_expected_source_identity_sha256="0" * 64,
            instruction_map={task_name: instruction},
        )


def test_gaussian_cache_rejects_nonfinite_frame_values(tmp_path):
    task_name = "Synthetic2RobotTask-rf"
    instruction = "two robots complete the synthetic task"
    stats_path, text_cache_dir = _write_demo(
        tmp_path,
        task_name=task_name,
        num_agents=2,
        instruction=instruction,
    )
    gaussian_cache_dir = _write_compact_gaussian_cache(
        tmp_path,
        task_name=task_name,
        num_agents=2,
        nonfinite_agent=1,
    )
    dataset = RoboFactoryMultiRobotDataset(
        str(tmp_path),
        load_future_video=False,
        video_size=(32, 32),
        val_set_proportion=0.0,
        is_training_set=True,
        randomize_agent_order=False,
        pretrained_norm_stats=str(stats_path),
        text_embedding_cache_dir=str(text_cache_dir),
        gaussian_cache_dir=str(gaussian_cache_dir),
        instruction_map={task_name: instruction},
    )
    with pytest.raises(ValueError, match="non-finite values"):
        dataset[0]


def test_compact_gaussian_cache_preflight_fails_before_training(tmp_path):
    task_name = "Synthetic2RobotTask-rf"
    instruction = "two robots complete the synthetic task"
    stats_path, text_cache_dir = _write_demo(
        tmp_path,
        task_name=task_name,
        num_agents=2,
        instruction=instruction,
    )
    gaussian_cache_dir = _write_compact_gaussian_cache(
        tmp_path,
        task_name=task_name,
        num_agents=2,
        included_agents=[0],
    )

    with pytest.raises(KeyError, match="Gaussian cache preflight failed"):
        RoboFactoryMultiRobotDataset(
            str(tmp_path),
            load_future_video=False,
            video_size=(32, 32),
            val_set_proportion=0.0,
            is_training_set=True,
            randomize_agent_order=False,
            pretrained_norm_stats=str(stats_path),
            text_embedding_cache_dir=str(text_cache_dir),
            gaussian_cache_dir=str(gaussian_cache_dir),
            instruction_map={task_name: instruction},
        )


def test_required_agent_counts_reject_legacy_stats_without_cardinality(tmp_path):
    task_name = "Synthetic1RobotTask-rf"
    instruction = "one robot completes the synthetic task"
    stats_path, cache_dir = _write_demo(
        tmp_path,
        task_name=task_name,
        num_agents=1,
        instruction=instruction,
    )

    with pytest.raises(ValueError, match="no cardinality metadata"):
        RoboFactoryMultiRobotDataset(
            str(tmp_path),
            video_size=(32, 32),
            val_set_proportion=0.0,
            is_training_set=True,
            required_agent_counts=[1],
            pretrained_norm_stats=str(stats_path),
            text_embedding_cache_dir=str(cache_dir),
            instruction_map={task_name: instruction},
        )


def test_unified_stats_provenance_and_cardinality_are_validated(tmp_path):
    task_name = "Synthetic2RobotTask-rf"
    instruction = "two robots complete the synthetic task"
    _, cache_dir = _write_demo(
        tmp_path,
        task_name=task_name,
        num_agents=2,
        instruction=instruction,
    )
    stats_path = tmp_path / "unified_stats.json"
    payload = compute_robofactory_stats(str(tmp_path))
    stats_path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = RoboFactoryMultiRobotDataset(
        str(tmp_path),
        video_size=(32, 32),
        val_set_proportion=0.0,
        is_training_set=True,
        required_agent_counts=[2],
        pretrained_norm_stats=str(stats_path),
        text_embedding_cache_dir=str(cache_dir),
        instruction_map={task_name: instruction},
    )
    assert payload["cardinality"] == {
        "agent_counts": [2],
        "trajectories_by_agent_count": {"2": 1},
    }
    assert dataset.agent_counts == (2,)

    payload["files"] = 99
    stats_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Stats files mismatch"):
        RoboFactoryMultiRobotDataset(
            str(tmp_path),
            video_size=(32, 32),
            val_set_proportion=0.0,
            is_training_set=True,
            required_agent_counts=[2],
            pretrained_norm_stats=str(stats_path),
            text_embedding_cache_dir=str(cache_dir),
            instruction_map={task_name: instruction},
        )


def test_formal_stats_require_exact_train_only_split_provenance(tmp_path):
    task_name = "Synthetic2RobotTask-rf"
    instruction = "two robots complete the synthetic task"
    _, cache_dir = _write_demo(
        tmp_path,
        task_name=task_name,
        num_agents=2,
        instruction=instruction,
    )
    stats_path = tmp_path / "train_stats.json"
    payload = compute_robofactory_stats(
        str(tmp_path),
        split_seed=42,
        val_set_proportion=0.0,
    )
    stats_path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = RoboFactoryMultiRobotDataset(
        str(tmp_path),
        video_size=(32, 32),
        val_set_proportion=0.0,
        split_seed=42,
        is_training_set=True,
        required_agent_counts=[2],
        require_train_only_stats=True,
        pretrained_norm_stats=str(stats_path),
        text_embedding_cache_dir=str(cache_dir),
        instruction_map={task_name: instruction},
    )
    assert dataset.agent_counts == (2,)

    payload["normalization_fit"]["split_seed"] = 7
    stats_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="normalization_fit mismatch"):
        RoboFactoryMultiRobotDataset(
            str(tmp_path),
            video_size=(32, 32),
            val_set_proportion=0.0,
            split_seed=42,
            is_training_set=True,
            required_agent_counts=[2],
            require_train_only_stats=True,
            pretrained_norm_stats=str(stats_path),
            text_embedding_cache_dir=str(cache_dir),
            instruction_map={task_name: instruction},
        )


def test_train_only_stats_exclude_validation_trajectory_values(tmp_path):
    task_name = "Synthetic2RobotTask-rf"
    instruction = "two robots complete the synthetic task"
    _write_demo(
        tmp_path,
        task_name=task_name,
        num_agents=2,
        instruction=instruction,
    )
    relative_source = f"{task_name}/motionplanning/demo.h5"

    def split_fraction(trajectory: str) -> float:
        digest = hashlib.sha256(
            f"42:{relative_source}:{trajectory}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    train_name = next(
        name for index in range(100) if split_fraction(name := f"train_{index}") >= 0.5
    )
    val_name = next(
        name for index in range(100) if split_fraction(name := f"val_{index}") < 0.5
    )
    h5_path = tmp_path / relative_source
    with h5py.File(h5_path, "r+") as handle:
        handle.copy("traj_0", train_name)
        handle.copy("traj_0", val_name)
        del handle["traj_0"]
        for agent_name in handle[train_name]["actions"]:
            handle[train_name][f"actions/{agent_name}"][...] = 3.0
        for agent_name in handle[val_name]["actions"]:
            handle[val_name][f"actions/{agent_name}"][...] = 99.0

    payload = compute_robofactory_stats(
        str(tmp_path),
        split_seed=42,
        val_set_proportion=0.5,
    )
    assert payload["normalization_fit"]["trajectories"] == 1
    assert payload["normalization_fit"]["cardinality"] == {
        "agent_counts": [2],
        "trajectories_by_agent_count": {"2": 1},
    }
    assert payload["action"]["mean"] == pytest.approx([3.0] * 8)
    assert payload["action"]["count"] == 80


def test_declared_agent_count_scope_rejects_unexpected_cardinality(tmp_path):
    one_task = "Synthetic1RobotTask-rf"
    two_task = "Synthetic2RobotTask-rf"
    one_instruction = "one robot completes the synthetic task"
    two_instruction = "two robots complete the synthetic task"
    _, cache_dir = _write_demo(
        tmp_path,
        task_name=one_task,
        num_agents=1,
        instruction=one_instruction,
    )
    _write_demo(
        tmp_path,
        task_name=two_task,
        num_agents=2,
        instruction=two_instruction,
    )
    stats_path = tmp_path / "unified_stats.json"
    stats_path.write_text(
        json.dumps(compute_robofactory_stats(str(tmp_path))),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=r"unexpected=\[1\]"):
        RoboFactoryMultiRobotDataset(
            str(tmp_path),
            video_size=(32, 32),
            val_set_proportion=0.0,
            is_training_set=True,
            required_agent_counts=[2],
            pretrained_norm_stats=str(stats_path),
            text_embedding_cache_dir=str(cache_dir),
            instruction_map={
                one_task: one_instruction,
                two_task: two_instruction,
            },
        )


def test_all_indexed_prompt_caches_are_preflighted_at_init(tmp_path):
    first_task = "Synthetic1RobotTask-rf"
    second_task = "Synthetic2RobotTask-rf"
    first_instruction = "one robot completes the first synthetic task"
    second_instruction = "two robots complete the second synthetic task"
    stats_path, cache_dir = _write_demo(
        tmp_path,
        task_name=first_task,
        num_agents=1,
        instruction=first_instruction,
    )
    _write_demo(
        tmp_path,
        task_name=second_task,
        num_agents=2,
        instruction=second_instruction,
        write_cache=False,
    )

    with pytest.raises(FileNotFoundError, match="indexed RoboFactory prompts") as exc_info:
        RoboFactoryMultiRobotDataset(
            str(tmp_path),
            video_size=(32, 32),
            val_set_proportion=0.0,
            is_training_set=True,
            pretrained_norm_stats=str(stats_path),
            text_embedding_cache_dir=str(cache_dir),
            instruction_map={
                first_task: first_instruction,
                second_task: second_instruction,
            },
        )
    assert second_task in str(exc_info.value)
