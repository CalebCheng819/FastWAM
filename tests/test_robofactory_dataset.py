import hashlib
import json

import h5py
import numpy as np
import pytest
import torch

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.robofactory_multi_robot import (
    RoboFactoryMultiRobotDataset,
    compute_robofactory_stats,
)


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
