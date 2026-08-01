from copy import deepcopy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ARMS = {
    "robofactory_multi_robot_vg0_hub0_224_1e-4": {
        "training_mode": "action_only_cache",
        "hub_enabled": False,
        "lambda_video": 0.0,
        "trainable_scope": "action",
        "load_future_video": False,
    },
    "robofactory_multi_robot_vg0_hub1_224_1e-4": {
        "training_mode": "action_only_cache",
        "hub_enabled": True,
        "lambda_video": 0.0,
        "trainable_scope": "action",
        "load_future_video": False,
    },
    "robofactory_multi_robot_vg1_hub0_224_1e-4": {
        "training_mode": "joint",
        "hub_enabled": False,
        "lambda_video": 1.0,
        "trainable_scope": "dit",
        "load_future_video": True,
    },
    "robofactory_multi_robot_vg1_hub1_224_1e-4": {
        "training_mode": "joint",
        "hub_enabled": True,
        "lambda_video": 1.0,
        "trainable_scope": "dit",
        "load_future_video": True,
    },
}

LEGACY_ALIASES = {
    "robofactory_multi_robot_hub_224_1e-4": "robofactory_multi_robot_vg0_hub1_224_1e-4",
    "robofactory_multi_robot_nohub_224_1e-4": "robofactory_multi_robot_vg0_hub0_224_1e-4",
}


def _compose_arm(task_name):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={task_name}"])
    return OmegaConf.to_container(cfg, resolve=True)


@pytest.mark.parametrize(("task_name", "expected"), ARMS.items())
def test_multirobot_2x2_arm_invariants(task_name, expected):
    cfg = _compose_arm(task_name)
    action_cfg = cfg["model"]["action_dit_config"]

    assert cfg["batch_size"] == 1
    assert cfg["agent_action_token_budget"] == 128
    assert cfg["gradient_accumulation_steps"] == 4
    assert cfg["seed"] == 42
    assert cfg["max_steps"] == 5000
    assert cfg["save_every"] == 2500
    assert cfg["save_training_state"] is True
    assert cfg["save_final_checkpoint"] is True
    assert cfg["trainable_scope"] == expected["trainable_scope"]
    assert cfg["model"]["training_mode"] == expected["training_mode"]
    assert cfg["model"]["loss"] == {
        "lambda_video": expected["lambda_video"],
        "lambda_action": 1.0,
    }
    assert action_cfg["hub_enabled"] is expected["hub_enabled"]
    assert action_cfg["hub_token_ratio"] == 2.0
    assert action_cfg["agent_encoding_mode"] == "geometry"
    assert action_cfg["agent_geometry_dim"] == 7
    assert "max_agents" not in action_cfg
    assert "num_hub_tokens" not in action_cfg

    for split in ("train", "val"):
        data_cfg = cfg["data"][split]
        assert data_cfg["required_agent_counts"] == [2, 3, 4]
        assert data_cfg["agent_geometry_dim"] == 7
        assert data_cfg["load_future_video"] is expected["load_future_video"]
        assert data_cfg["pretrained_norm_stats"].endswith(
            "/fastwam_multi_robot_n234_stats.json"
        )
        assert data_cfg["text_embedding_cache_dir"].endswith(
            "/text_embeds_cache_n234"
        )
        assert "max_agents" not in data_cfg


def test_multirobot_2x2_uses_one_model_structure_and_data_source():
    configs = {name: _compose_arm(name) for name in ARMS}
    reference = None
    reference_root = None
    reference_checkpoint = None
    reference_schedule = None

    for cfg in configs.values():
        normalized_model = deepcopy(cfg["model"])
        normalized_model["training_mode"] = "<treatment>"
        normalized_model["loss"]["lambda_video"] = "<treatment>"
        normalized_model["action_dit_config"]["hub_enabled"] = "<treatment>"

        if reference is None:
            reference = normalized_model
            reference_root = cfg["data"]["train"]["root_dir"]
            reference_checkpoint = cfg["resume"]
            reference_schedule = {
                key: cfg[key]
                for key in (
                    "seed",
                    "batch_size",
                    "agent_action_token_budget",
                    "gradient_accumulation_steps",
                    "num_epochs",
                    "max_steps",
                )
            }
        else:
            assert normalized_model == reference
            assert cfg["data"]["train"]["root_dir"] == reference_root
            assert cfg["resume"] == reference_checkpoint
            assert {
                key: cfg[key]
                for key in reference_schedule
            } == reference_schedule


@pytest.mark.parametrize(("alias", "canonical"), LEGACY_ALIASES.items())
def test_legacy_multirobot_task_names_resolve_to_vg0_arms(alias, canonical):
    alias_cfg = _compose_arm(alias)
    canonical_cfg = _compose_arm(canonical)

    for key in ("data", "model", "trainable_scope", "resume"):
        assert alias_cfg[key] == canonical_cfg[key]
