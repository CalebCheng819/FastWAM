from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.robofactory.fastwam_multi_robot_policy import (
    DEFAULT_INSTRUCTIONS,
    DEFAULT_PROMPT,
    NOPOSPLAT_CHECKPOINT_SHA256,
    NormalizationStats,
    camera_rgb_uint8,
    canonicalize_root_pose,
    compose_step5000_model_config,
    denormalize_and_flatten_actions,
    load_normalization_stats,
    load_text_context,
    model_input_image,
    prepare_observation,
    sha256_file,
    teacher_image_pairs,
)


def _stats() -> NormalizationStats:
    return NormalizationStats(
        action_mean=torch.arange(8, dtype=torch.float32),
        action_std=torch.full((8,), 2.0),
        state_mean=torch.zeros(18),
        state_std=torch.full((18,), 2.0),
    )


def _observation(agent_count: int):
    height, width = 240, 320
    sensor_data = {
        "head_camera_global": {
            "rgb": torch.zeros((1, height, width, 4), dtype=torch.uint8)
        }
    }
    agents = {}
    state = {"articulations": {}}
    # Reverse insertion order to prove that the adapter uses native numeric ids.
    for agent_id in reversed(range(agent_count)):
        agents[f"panda-{agent_id}"] = {
            "qpos": torch.full((1, 9), float(agent_id + 1)),
            "qvel": torch.full((1, 9), float(10 + agent_id)),
        }
        sensor_data[f"head_camera_agent{agent_id}"] = {
            "rgb": torch.full(
                (1, height, width, 3),
                fill_value=50 + agent_id,
                dtype=torch.uint8,
            )
        }
        articulation = torch.zeros((1, 20), dtype=torch.float32)
        articulation[0, :3] = torch.tensor([agent_id, agent_id + 1, agent_id + 2])
        articulation[0, 3] = -2.0
        state["articulations"][f"panda-agent-{agent_id}"] = articulation
    return {"agent": agents, "sensor_data": sensor_data}, state


@pytest.mark.parametrize("agent_count", [2, 3, 4])
def test_prepare_observation_preserves_dynamic_native_agent_axis(agent_count: int):
    observation, env_state = _observation(agent_count)

    prepared = prepare_observation(observation, env_state, _stats())

    assert prepared.agent_names == tuple(
        f"panda-{index}" for index in range(agent_count)
    )
    assert prepared.agent_rgb.shape == (agent_count, 3, 240, 320)
    assert prepared.agent_states.shape == (agent_count, 18)
    assert prepared.agent_geometry.shape == (agent_count, 7)
    assert prepared.agent_ids.tolist() == list(range(agent_count))
    for agent_id in range(agent_count):
        assert torch.allclose(
            prepared.agent_states[agent_id],
            torch.tensor([agent_id + 1] * 9 + [10 + agent_id] * 9) / 2.0,
        )
        assert prepared.agent_geometry[agent_id, :3].tolist() == [
            agent_id,
            agent_id + 1,
            agent_id + 2,
        ]
        assert prepared.agent_geometry[agent_id, 3:].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_camera_and_teacher_pairing_match_training_ranges():
    observation, env_state = _observation(2)
    # Exercise float [0,1] conversion as returned by some ManiSkill wrappers.
    observation["sensor_data"]["head_camera_agent1"]["rgb"] = torch.full(
        (1, 3, 240, 320),
        1.0,
        dtype=torch.float32,
    )
    prepared = prepare_observation(observation, env_state, _stats())

    assert camera_rgb_uint8(observation, "head_camera_agent1").unique().item() == 255
    pairs = teacher_image_pairs(prepared)
    assert pairs.shape == (2, 2, 3, 240, 320)
    assert pairs.dtype == torch.float32
    assert torch.equal(pairs[0, 0], pairs[1, 0])
    assert pairs[0, 0].unique().item() == -1.0
    assert pairs[1, 1].unique().item() == 1.0

    image = model_input_image(prepared)
    assert image.shape == (1, 3, 224, 320)
    assert image.dtype == torch.float32
    assert image.unique().item() == -1.0


def test_root_pose_uses_maximum_component_for_sign_canonicalization():
    pose = canonicalize_root_pose([1, 2, 3, 0.1, -0.9, 0.2, 0.3, 99])
    expected_quaternion = -torch.tensor([0.1, -0.9, 0.2, 0.3])
    expected_quaternion /= torch.linalg.vector_norm(expected_quaternion)
    assert pose.shape == (7,)
    assert torch.allclose(pose[3:], expected_quaternion)


def test_denormalize_and_flatten_actions_preserves_agent_blocks():
    normalized = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)

    flattened = denormalize_and_flatten_actions(normalized, _stats())

    assert flattened.shape == (3, 16)
    expected_first_step = torch.cat(
        (
            normalized[0, 0] * 2.0 + torch.arange(8),
            normalized[1, 0] * 2.0 + torch.arange(8),
        )
    ).numpy()
    np.testing.assert_array_equal(flattened[0], expected_first_step)


def test_stats_loader_is_hash_pinned_and_clamps_std(tmp_path: Path):
    payload = {
        "normalization_fit": {
            "split": "train",
            "split_seed": 42,
            "val_set_proportion": 0.1,
        },
        "cardinality": {"agent_counts": [4, 2, 3]},
        "action": {"mean": [0.0] * 8, "std": [0.0] + [2.0] * 7},
        "state": {"mean": [1.0] * 18, "std": [3.0] * 18},
    }
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = sha256_file(path)

    stats = load_normalization_stats(path, expected_sha256=digest)

    assert stats.sha256 == digest
    assert stats.action_std[0].item() == pytest.approx(1e-6)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_normalization_stats(path, expected_sha256="0" * 64)


def test_text_context_loader_uses_prompt_hash_and_padding_convention(tmp_path: Path):
    task_name = "PlaceFood-rf"
    prompt = DEFAULT_PROMPT.format(task=DEFAULT_INSTRUCTIONS[task_name])
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    path = tmp_path / f"{prompt_digest}.t5_len128.wan22ti2v5b.pt"
    context = torch.ones((128, 4096), dtype=torch.bfloat16)
    mask = torch.ones((128,), dtype=torch.bool)
    mask[-3:] = False
    torch.save({"context": context, "mask": mask}, path)
    digest = sha256_file(path)

    loaded = load_text_context(tmp_path, task_name, expected_sha256=digest)

    assert loaded.prompt == prompt
    assert loaded.context.dtype == torch.bfloat16
    assert torch.count_nonzero(loaded.context[-3:]).item() == 0
    assert loaded.mask.all().item()
    assert loaded.sha256 == digest


def test_step5000_model_config_is_joint_hub_gaussian_without_text_encoder():
    config = compose_step5000_model_config()

    assert config._target_ == "fastwam.runtime.create_multi_robot_fastwam"
    assert config.training_mode == "joint"
    assert config.load_text_encoder is False
    assert config.skip_dit_load_from_pretrain is True
    assert config.action_dit_config.hub_enabled is True
    assert config.action_dit_config.enable_gaussian is True
    assert config.action_dit_config.action_dim == 8
    assert config.action_dit_config.state_dim == 18
    assert config.loss.lambda_video == 1.0
    assert config.loss.lambda_action == 1.0
    assert NOPOSPLAT_CHECKPOINT_SHA256 == (
        "4a35bc8c341b20859c0621f5238349b55b19a34a5bbeb3daec8d1f4c4603cd08"
    )
