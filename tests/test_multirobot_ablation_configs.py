from copy import deepcopy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError


TEST_GAUSSIAN_CACHE_DIR = "/cpfs/test/robofactory/compact-gaussian-v1"
TEST_GAUSSIAN_MANIFEST_SHA256 = "1" * 64
TEST_GAUSSIAN_SELECTION_SHA256 = "2" * 64
TEST_GAUSSIAN_SOURCE_IDENTITY_SHA256 = "3" * 64

ARMS = {
    "robofactory_multi_robot_vg0_hub0_gau0_224_1e-4": {
        "training_mode": "action_only_cache",
        "hub_enabled": False,
        "enable_gaussian": False,
        "lambda_video": 0.0,
        "trainable_scope": "action",
        "load_future_video": False,
    },
    "robofactory_multi_robot_vg0_hub0_gau1_224_1e-4": {
        "training_mode": "action_only_cache",
        "hub_enabled": False,
        "enable_gaussian": True,
        "lambda_video": 0.0,
        "trainable_scope": "action",
        "load_future_video": False,
    },
    "robofactory_multi_robot_vg0_hub1_gau0_224_1e-4": {
        "training_mode": "action_only_cache",
        "hub_enabled": True,
        "enable_gaussian": False,
        "lambda_video": 0.0,
        "trainable_scope": "action",
        "load_future_video": False,
    },
    "robofactory_multi_robot_vg0_hub1_gau1_224_1e-4": {
        "training_mode": "action_only_cache",
        "hub_enabled": True,
        "enable_gaussian": True,
        "lambda_video": 0.0,
        "trainable_scope": "action",
        "load_future_video": False,
    },
    "robofactory_multi_robot_vg1_hub0_gau0_224_1e-4": {
        "training_mode": "joint",
        "hub_enabled": False,
        "enable_gaussian": False,
        "lambda_video": 1.0,
        "trainable_scope": "dit",
        "load_future_video": True,
    },
    "robofactory_multi_robot_vg1_hub0_gau1_224_1e-4": {
        "training_mode": "joint",
        "hub_enabled": False,
        "enable_gaussian": True,
        "lambda_video": 1.0,
        "trainable_scope": "dit",
        "load_future_video": True,
    },
    "robofactory_multi_robot_vg1_hub1_gau0_224_1e-4": {
        "training_mode": "joint",
        "hub_enabled": True,
        "enable_gaussian": False,
        "lambda_video": 1.0,
        "trainable_scope": "dit",
        "load_future_video": True,
    },
    "robofactory_multi_robot_vg1_hub1_gau1_224_1e-4": {
        "training_mode": "joint",
        "hub_enabled": True,
        "enable_gaussian": True,
        "lambda_video": 1.0,
        "trainable_scope": "dit",
        "load_future_video": True,
    },
}

LEGACY_ALIASES = {
    "robofactory_multi_robot_hub_224_1e-4": "robofactory_multi_robot_vg0_hub1_224_1e-4",
    "robofactory_multi_robot_nohub_224_1e-4": "robofactory_multi_robot_vg0_hub0_224_1e-4",
}


def _compose_arm(task_name, *extra_overrides):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[f"task={task_name}", *extra_overrides],
        )
    return OmegaConf.to_container(cfg, resolve=True)


def _set_gaussian_env(monkeypatch):
    monkeypatch.setenv("FASTWAM_GAUSSIAN_CACHE_DIR", TEST_GAUSSIAN_CACHE_DIR)
    monkeypatch.setenv(
        "FASTWAM_GAUSSIAN_CACHE_MANIFEST_SHA256",
        TEST_GAUSSIAN_MANIFEST_SHA256,
    )
    monkeypatch.setenv(
        "FASTWAM_GAUSSIAN_CACHE_SELECTION_SHA256",
        TEST_GAUSSIAN_SELECTION_SHA256,
    )
    monkeypatch.setenv(
        "FASTWAM_GAUSSIAN_CACHE_SOURCE_IDENTITY_SHA256",
        TEST_GAUSSIAN_SOURCE_IDENTITY_SHA256,
    )


@pytest.mark.parametrize(("task_name", "expected"), ARMS.items())
def test_multirobot_2x2x2_arm_invariants(task_name, expected, monkeypatch):
    _set_gaussian_env(monkeypatch)
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
    assert action_cfg["enable_gaussian"] is expected["enable_gaussian"]
    assert action_cfg["gaussian_channels"] == 13
    assert action_cfg["gaussian_height"] == 28
    assert action_cfg["gaussian_width"] == 40
    assert "max_agents" not in action_cfg
    assert "num_hub_tokens" not in action_cfg

    for split in ("train", "val"):
        data_cfg = cfg["data"][split]
        assert data_cfg["required_agent_counts"] == [2, 3, 4]
        assert data_cfg["agent_geometry_dim"] == 7
        assert data_cfg["load_future_video"] is expected["load_future_video"]
        expected_cache = (
            TEST_GAUSSIAN_CACHE_DIR if expected["enable_gaussian"] else None
        )
        assert data_cfg["gaussian_cache_dir"] == expected_cache
        assert data_cfg["gaussian_cache_verify"] == "manifest"
        assert data_cfg["gaussian_cache_expected_manifest_sha256"] == (
            TEST_GAUSSIAN_MANIFEST_SHA256
            if expected["enable_gaussian"]
            else None
        )
        assert data_cfg["gaussian_cache_expected_selection_sha256"] == (
            TEST_GAUSSIAN_SELECTION_SHA256
            if expected["enable_gaussian"]
            else None
        )
        assert data_cfg["gaussian_cache_expected_source_identity_sha256"] == (
            TEST_GAUSSIAN_SOURCE_IDENTITY_SHA256
            if expected["enable_gaussian"]
            else None
        )
        assert data_cfg["gaussian_channels"] == 13
        assert data_cfg["gaussian_size"] == [28, 40]
        assert data_cfg["pretrained_norm_stats"].endswith(
            "/fastwam_multi_robot_n234_train_s42_stats_v2.json"
        )
        assert data_cfg["text_embedding_cache_dir"].endswith(
            "/text_embeds_cache_n234"
        )
        assert "max_agents" not in data_cfg

    assert cfg["wandb"]["group"] == (
        "robofactory-multirobot-videogen-hub-gaussian-2x2x2"
    )
    assert cfg["wandb"]["name"].endswith("-s42")


def test_gaussian_conditioning_axis_is_explicit_and_fail_closed(monkeypatch):
    for name in (
        "FASTWAM_GAUSSIAN_CACHE_DIR",
        "FASTWAM_GAUSSIAN_CACHE_MANIFEST_SHA256",
        "FASTWAM_GAUSSIAN_CACHE_SELECTION_SHA256",
        "FASTWAM_GAUSSIAN_CACHE_SOURCE_IDENTITY_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)
    disabled = _compose_arm("robofactory_multi_robot_vg1_hub1_gau0_224_1e-4")
    assert disabled["model"]["action_dit_config"]["enable_gaussian"] is False
    assert disabled["data"]["train"]["gaussian_cache_dir"] is None
    assert disabled["data"]["val"]["gaussian_cache_dir"] is None

    with pytest.raises(
        InterpolationResolutionError, match="FASTWAM_GAUSSIAN_CACHE_DIR"
    ):
        _compose_arm("robofactory_multi_robot_vg1_hub1_gau1_224_1e-4")

    monkeypatch.setenv("FASTWAM_GAUSSIAN_CACHE_DIR", TEST_GAUSSIAN_CACHE_DIR)
    with pytest.raises(
        InterpolationResolutionError,
        match="FASTWAM_GAUSSIAN_CACHE_MANIFEST_SHA256",
    ):
        _compose_arm("robofactory_multi_robot_vg1_hub1_gau1_224_1e-4")

    _set_gaussian_env(monkeypatch)
    enabled = _compose_arm("robofactory_multi_robot_vg1_hub1_gau1_224_1e-4")
    assert enabled["model"]["action_dit_config"]["enable_gaussian"] is True
    assert enabled["data"]["train"]["gaussian_cache_dir"] == TEST_GAUSSIAN_CACHE_DIR
    assert enabled["data"]["val"]["gaussian_cache_dir"] == TEST_GAUSSIAN_CACHE_DIR
    assert enabled["data"]["train"]["gaussian_cache_expected_manifest_sha256"] == (
        TEST_GAUSSIAN_MANIFEST_SHA256
    )

    disabled_model = deepcopy(disabled["model"])
    enabled_model = deepcopy(enabled["model"])
    disabled_model["action_dit_config"]["enable_gaussian"] = "<treatment>"
    enabled_model["action_dit_config"]["enable_gaussian"] = "<treatment>"
    assert disabled_model == enabled_model


def test_multirobot_2x2x2_uses_one_model_structure_and_data_source(monkeypatch):
    _set_gaussian_env(monkeypatch)
    configs = {name: _compose_arm(name) for name in ARMS}
    reference = None
    reference_data = None
    reference_root = None
    reference_checkpoint = None
    reference_schedule = None

    for cfg in configs.values():
        normalized_model = deepcopy(cfg["model"])
        normalized_model["training_mode"] = "<treatment>"
        normalized_model["loss"]["lambda_video"] = "<treatment>"
        normalized_model["action_dit_config"]["hub_enabled"] = "<treatment>"
        normalized_model["action_dit_config"]["enable_gaussian"] = "<treatment>"

        normalized_data = deepcopy(cfg["data"])
        for split in ("train", "val"):
            normalized_data[split]["load_future_video"] = "<treatment>"
            normalized_data[split]["gaussian_cache_dir"] = "<treatment>"
            normalized_data[split]["gaussian_cache_expected_manifest_sha256"] = (
                "<treatment>"
            )
            normalized_data[split]["gaussian_cache_expected_selection_sha256"] = (
                "<treatment>"
            )
            normalized_data[split]["gaussian_cache_expected_source_identity_sha256"] = (
                "<treatment>"
            )

        if reference is None:
            reference = normalized_model
            reference_data = normalized_data
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
            assert normalized_data == reference_data
            assert cfg["data"]["train"]["root_dir"] == reference_root
            assert cfg["resume"] == reference_checkpoint
            assert {
                key: cfg[key]
                for key in reference_schedule
            } == reference_schedule

    assert len({cfg["wandb"]["name"] for cfg in configs.values()}) == 8


def test_sensor_control_registry_does_not_claim_agent_rgb_is_runnable():
    repo_root = Path(__file__).resolve().parents[1]
    registry = OmegaConf.to_container(
        OmegaConf.load(
            repo_root
            / "configs"
            / "controls"
            / "robofactory_multi_robot_sensor_controls.yaml"
        ),
        resolve=True,
    )["controls"]

    assert set(registry) == {
        "global_only",
        "global_agent_rgb",
        "global_agent_gaussian",
    }
    assert registry["global_only"]["status"] == "runnable"
    assert registry["global_only"]["task_axis"] == "GAU0"
    assert registry["global_agent_rgb"]["status"] == "planned_not_runnable"
    assert registry["global_agent_rgb"]["runnable"] is False
    assert registry["global_agent_rgb"]["task_config"] is None
    assert registry["global_agent_gaussian"]["status"] == (
        "runnable_with_versioned_cache"
    )
    assert registry["global_agent_gaussian"]["required_env"] == [
        "FASTWAM_GAUSSIAN_CACHE_DIR",
        "FASTWAM_GAUSSIAN_CACHE_MANIFEST_SHA256",
        "FASTWAM_GAUSSIAN_CACHE_SELECTION_SHA256",
        "FASTWAM_GAUSSIAN_CACHE_SOURCE_IDENTITY_SHA256",
    ]
    assert not list((repo_root / "configs" / "task").glob("*agent_rgb*.yaml"))


@pytest.mark.parametrize(("alias", "canonical"), LEGACY_ALIASES.items())
def test_legacy_multirobot_task_names_resolve_to_vg0_arms(alias, canonical):
    alias_cfg = _compose_arm(alias)
    canonical_cfg = _compose_arm(canonical)

    for key in ("data", "model", "trainable_scope", "resume"):
        assert alias_cfg[key] == canonical_cfg[key]


def test_formal_32gpu_profile_changes_only_scale_controls(monkeypatch):
    _set_gaussian_env(monkeypatch)
    baseline = _compose_arm(
        "robofactory_multi_robot_vg1_hub1_gau1_224_1e-4"
    )
    scaled = _compose_arm(
        "robofactory_multi_robot_vg1_hub1_gau1_224_1e-4",
        "+scale=robofactory_multi_robot_32gpu",
    )

    assert baseline["gradient_accumulation_steps"] == 4
    assert baseline["eval_every"] == 0
    assert baseline["offline_eval_num_samples"] == 0
    assert baseline["checkpoint_state_kind"] == "auto"
    assert baseline["seal_training_state"] is False
    assert baseline["process_group_timeout_seconds"] == 1800
    assert baseline["checkpoint_io_timeout_seconds"] == 1800
    assert scaled["gradient_accumulation_steps"] == 1
    assert scaled["eval_every"] == 1000
    assert scaled["offline_eval_num_samples"] == 12
    assert scaled["checkpoint_state_kind"] == "full"
    assert scaled["seal_training_state"] is True
    assert scaled["process_group_timeout_seconds"] == 21600
    assert scaled["checkpoint_io_timeout_seconds"] == 21600

    for key in ("data", "model", "seed", "max_steps", "resume"):
        assert scaled[key] == baseline[key]
