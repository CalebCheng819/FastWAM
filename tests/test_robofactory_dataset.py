import hashlib
import json

import h5py
import numpy as np
import torch

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.robofactory_multi_robot import RoboFactoryMultiRobotDataset


def test_robofactory_hdf5_adapter(tmp_path):
    task_dir = tmp_path / "ThreeRobotsStackCube-rf" / "motionplanning"
    task_dir.mkdir(parents=True)
    h5_path = task_dir / "demo.h5"
    length = 40
    with h5py.File(h5_path, "w") as handle:
        trajectory = handle.create_group("traj_0")
        actions = trajectory.create_group("actions")
        obs = trajectory.create_group("obs")
        agents = obs.create_group("agent")
        sensors = obs.create_group("sensor_data")
        global_camera = sensors.create_group("head_camera_global")
        global_camera.create_dataset(
            "rgb",
            data=np.random.default_rng(1).integers(
                0, 256, size=(length + 1, 24, 32, 3), dtype=np.uint8
            ),
        )
        for agent_idx in range(3):
            name = f"panda-{agent_idx}"
            actions.create_dataset(
                name,
                data=np.full((length, 8), float(agent_idx + 1), dtype=np.float32),
            )
            agent = agents.create_group(name)
            agent.create_dataset("qpos", data=np.ones((length + 1, 9), dtype=np.float32))
            agent.create_dataset("qvel", data=np.zeros((length + 1, 9), dtype=np.float32))

    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "action": {"mean": [0.0] * 8, "std": [1.0] * 8},
                "state": {"mean": [0.0] * 18, "std": [1.0] * 18},
            }
        )
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    prompt = DEFAULT_PROMPT.format(task="three robots collaboratively stack the cubes")
    cache_path = cache_dir / (
        hashlib.sha256(prompt.encode()).hexdigest() + ".t5_len128.wan22ti2v5b.pt"
    )
    torch.save(
        {"context": torch.zeros(128, 16), "mask": torch.ones(128, dtype=torch.bool)},
        cache_path,
    )

    dataset = RoboFactoryMultiRobotDataset(
        str(tmp_path),
        video_size=(32, 32),
        max_agents=4,
        window_stride=16,
        val_set_proportion=0.0,
        is_training_set=True,
        randomize_agent_order=False,
        pretrained_norm_stats=str(stats_path),
        text_embedding_cache_dir=str(cache_dir),
    )
    sample = dataset[0]
    assert sample["video"].shape == (3, 9, 32, 32)
    assert sample["action"].shape == (4, 32, 8)
    assert sample["agent_state"].shape == (4, 18)
    assert sample["agent_mask"].tolist() == [True, True, True, False]
    assert sample["agent_ids"].tolist() == [0, 1, 2, 0]
    assert torch.allclose(sample["action"][1], torch.full((32, 8), 2.0))
    assert sample["context"].shape == (128, 16)
