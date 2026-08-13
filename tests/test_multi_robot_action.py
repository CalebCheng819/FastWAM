import pytest
import torch

from fastwam.models.wan22.fastwam_multi_robot import FastWAMMultiRobot
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.multi_agent_action_dit import (
    GaussianAgentAdapter,
    MultiAgentActionDiT,
    regular_simplex_vertices,
)


def _tiny_action_expert(
    *,
    hub_enabled=True,
    agent_encoding_mode="geometry",
    enable_gaussian=False,
):
    return MultiAgentActionDiT(
        action_dim=3,
        state_dim=4,
        agent_geometry_dim=7 if agent_encoding_mode == "geometry" else 0,
        agent_encoding_mode=agent_encoding_mode,
        hub_enabled=hub_enabled,
        hub_token_ratio=2.0,
        agent_rope_dim=4,
        enable_gaussian=enable_gaussian,
        gaussian_channels=13,
        gaussian_height=28,
        gaussian_width=40,
        gaussian_stem_dim=8,
        hidden_dim=32,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=8,
        num_layers=2,
        use_gradient_checkpointing=False,
    )


def _inputs(num_agents):
    return {
        "action": torch.randn(1, num_agents, 5, 3),
        "state": torch.randn(1, num_agents, 4),
        "geometry": torch.randn(1, num_agents, 7),
        "gaussian": torch.randn(1, num_agents, 13, 28, 40).half(),
        "ids": torch.arange(num_agents).unsqueeze(0),
        "timestep": torch.tensor([500.0]),
        "context": torch.randn(1, 6, 16),
        "context_mask": torch.ones(1, 6, dtype=torch.bool),
    }


@pytest.mark.parametrize("num_agents", [1, 2, 3, 4])
def test_regular_simplex_is_symmetric(num_agents):
    vertices = regular_simplex_vertices(num_agents)
    if num_agents == 1:
        assert vertices.shape == (1, 1)
        assert torch.equal(vertices, torch.zeros_like(vertices))
        return
    gram = vertices @ vertices.T
    expected = torch.full((num_agents, num_agents), -1.0 / (num_agents - 1))
    expected.fill_diagonal_(1.0)
    assert torch.allclose(gram, expected, atol=1e-6)


@pytest.mark.parametrize("num_agents", [1, 2, 3, 4])
def test_one_model_accepts_native_agent_cardinalities(num_agents):
    torch.manual_seed(3)
    model = _tiny_action_expert().eval()
    data = _inputs(num_agents)
    pre = model.pre_dit(
        data["action"],
        data["timestep"],
        data["context"],
        data["context_mask"],
        agent_states=data["state"],
        agent_geometry=data["geometry"],
        agent_ids=data["ids"],
    )
    assert pre["meta"]["num_agents"] == num_agents
    assert pre["meta"]["num_hub_tokens"] == 2 * num_agents
    assert pre["tokens"].shape == (1, num_agents * 5 + 2 * num_agents, 32)
    assert not hasattr(model, "max_agents")
    assert not hasattr(model, "hub_tokens")


@pytest.mark.parametrize("num_agents", [1, 2, 3, 4])
def test_gaussian_agent_adapter_accepts_fp16_native_cardinalities(num_agents):
    torch.manual_seed(5 + num_agents)
    adapter = GaussianAgentAdapter(
        hidden_dim=32,
        in_channels=13,
        input_height=28,
        input_width=40,
        stem_dim=8,
    ).eval()
    gaussian = torch.randn(2, num_agents, 13, 28, 40).half()
    embedding = adapter(gaussian)
    assert embedding.shape == (2, num_agents, 32)
    assert embedding.dtype == adapter.stem[0].weight.dtype
    assert torch.isfinite(embedding).all()


@pytest.mark.parametrize("num_agents", [1, 2, 3, 4])
def test_gaussian_zero_gate_is_strictly_baseline_equivalent(num_agents):
    torch.manual_seed(11 + num_agents)
    baseline = _tiny_action_expert(enable_gaussian=False).eval()
    conditioned = _tiny_action_expert(enable_gaussian=True).eval()
    load_result = conditioned.load_state_dict(baseline.state_dict(), strict=False)
    assert not load_result.unexpected_keys
    assert load_result.missing_keys
    assert conditioned.gaussian_gate is not None
    assert torch.equal(conditioned.gaussian_gate, torch.zeros_like(conditioned.gaussian_gate))

    data = _inputs(num_agents)
    shared = dict(
        action_tokens=data["action"],
        timestep=data["timestep"],
        context=data["context"],
        context_mask=data["context_mask"],
        agent_states=data["state"],
        agent_geometry=data["geometry"],
        agent_ids=data["ids"],
    )
    baseline_pre = baseline.pre_dit(**shared)
    conditioned_pre = conditioned.pre_dit(
        **shared,
        agent_gaussian=data["gaussian"],
    )
    assert torch.equal(baseline_pre["tokens"], conditioned_pre["tokens"])
    assert torch.equal(baseline_pre["freqs"], conditioned_pre["freqs"])


def test_gaussian_ablation_preserves_all_common_initial_tensors_under_run_seed():
    torch.manual_seed(20260802)
    baseline = _tiny_action_expert(enable_gaussian=False)
    torch.manual_seed(20260802)
    conditioned = _tiny_action_expert(enable_gaussian=True)

    baseline_state = baseline.state_dict()
    conditioned_state = conditioned.state_dict()
    common_keys = sorted(set(baseline_state) & set(conditioned_state))
    assert common_keys
    for key in common_keys:
        assert torch.equal(baseline_state[key], conditioned_state[key]), key
    assert set(conditioned_state) - set(baseline_state) == {
        key
        for key in conditioned_state
        if key.startswith("gaussian_adapter.") or key == "gaussian_gate"
    }


def test_gaussian_conditioning_is_agent_local_and_permutation_equivariant():
    torch.manual_seed(19)
    model = _tiny_action_expert(enable_gaussian=True).eval()
    assert model.gaussian_gate is not None
    model.gaussian_gate.data.fill_(1.0)
    data = _inputs(4)
    pre = model.pre_dit(
        data["action"],
        data["timestep"],
        data["context"],
        data["context_mask"],
        agent_states=data["state"],
        agent_geometry=data["geometry"],
        agent_ids=data["ids"],
        agent_gaussian=data["gaussian"],
    )
    permutation = torch.tensor([3, 1, 0, 2])
    permuted = model.pre_dit(
        data["action"][:, permutation],
        data["timestep"],
        data["context"],
        data["context_mask"],
        agent_states=data["state"][:, permutation],
        agent_geometry=data["geometry"][:, permutation],
        agent_ids=data["ids"][:, permutation],
        agent_gaussian=data["gaussian"][:, permutation],
    )
    horizon = data["action"].shape[2]
    inverse = torch.argsort(permutation)
    action_tokens = pre["tokens"][:, : 4 * horizon].reshape(1, 4, horizon, -1)
    permuted_action_tokens = permuted["tokens"][:, : 4 * horizon].reshape(
        1, 4, horizon, -1
    )
    assert torch.allclose(action_tokens, permuted_action_tokens[:, inverse])
    assert torch.equal(pre["tokens"][:, 4 * horizon :], permuted["tokens"][:, 4 * horizon :])

    perturbed = data["gaussian"].clone()
    perturbed[:, 1].add_(8.0)
    original_embedding = model.gaussian_adapter(data["gaussian"])
    perturbed_embedding = model.gaussian_adapter(perturbed)
    assert torch.equal(original_embedding[:, 0], perturbed_embedding[:, 0])
    assert not torch.allclose(original_embedding[:, 1], perturbed_embedding[:, 1])


def test_gaussian_gate_and_adapter_gradients_are_finite():
    torch.manual_seed(23)
    model = _tiny_action_expert(enable_gaussian=True).train()
    assert model.gaussian_gate is not None
    model.gaussian_gate.data.fill_(1.0)
    data = _inputs(3)
    pre = model.pre_dit(
        data["action"],
        data["timestep"],
        data["context"],
        data["context_mask"],
        agent_states=data["state"],
        agent_geometry=data["geometry"],
        agent_ids=data["ids"],
        agent_gaussian=data["gaussian"],
    )
    pre["tokens"].square().mean().backward()
    assert model.gaussian_gate.grad is not None
    assert torch.isfinite(model.gaussian_gate.grad).all()
    assert model.gaussian_adapter is not None
    adapter_gradients = [
        parameter.grad for parameter in model.gaussian_adapter.parameters()
    ]
    assert adapter_gradients
    assert all(gradient is not None for gradient in adapter_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in adapter_gradients)


def test_gaussian_enabled_requires_exact_shape_and_disabled_ignores_field():
    data = _inputs(2)
    enabled = _tiny_action_expert(enable_gaussian=True).eval()
    with pytest.raises(ValueError, match="enable_gaussian=true requires"):
        enabled.pre_dit(
            data["action"],
            data["timestep"],
            data["context"],
            data["context_mask"],
            agent_states=data["state"],
            agent_geometry=data["geometry"],
            agent_ids=data["ids"],
        )
    with pytest.raises(ValueError, match="`agent_gaussian` must be"):
        enabled.pre_dit(
            data["action"],
            data["timestep"],
            data["context"],
            data["context_mask"],
            agent_states=data["state"],
            agent_geometry=data["geometry"],
            agent_ids=data["ids"],
            agent_gaussian=torch.randn(1, 2, 13, 28, 39),
        )

    disabled = _tiny_action_expert(enable_gaussian=False).eval()
    disabled_without = disabled.pre_dit(
        data["action"],
        data["timestep"],
        data["context"],
        data["context_mask"],
        agent_states=data["state"],
        agent_geometry=data["geometry"],
        agent_ids=data["ids"],
    )
    disabled_with = disabled.pre_dit(
        data["action"],
        data["timestep"],
        data["context"],
        data["context_mask"],
        agent_states=data["state"],
        agent_geometry=data["geometry"],
        agent_ids=data["ids"],
        agent_gaussian=torch.ones(1, 2, 1, 1, 1, dtype=torch.int64),
    )
    assert torch.equal(disabled_without["tokens"], disabled_with["tokens"])


def _bare_multi_robot_model_for_input_validation(*, enable_gaussian):
    model = FastWAMMultiRobot.__new__(FastWAMMultiRobot)
    torch.nn.Module.__init__(model)
    model.action_expert = _tiny_action_expert(enable_gaussian=enable_gaussian)
    model.video_expert = torch.nn.Identity()
    model.video_expert.fuse_vae_embedding_in_latents = False
    model.training_mode = "action_only_cache"
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model._encode_video_latents = lambda video, tiled=False: video
    return model


def _multi_robot_sample_for_input_validation(*, include_gaussian):
    num_agents = 2
    sample = {
        "video": torch.randn(1, 3, 1, 16, 16),
        "action": torch.randn(1, num_agents, 5, 3),
        "agent_state": torch.randn(1, num_agents, 4),
        "agent_geometry": torch.randn(1, num_agents, 7),
        "agent_ids": torch.arange(num_agents).unsqueeze(0),
        "context": torch.randn(1, 6, 16),
        "context_mask": torch.ones(1, 6, dtype=torch.bool),
    }
    if include_gaussian:
        sample["agent_gaussian"] = torch.randn(
            1, num_agents, 13, 28, 40, dtype=torch.float16
        )
    return sample


def test_build_inputs_requires_gaussian_only_for_enabled_ablation():
    enabled = _bare_multi_robot_model_for_input_validation(enable_gaussian=True)
    with pytest.raises(ValueError, match="Missing multi-robot sample fields.*agent_gaussian"):
        enabled.build_inputs(
            _multi_robot_sample_for_input_validation(include_gaussian=False)
        )
    enabled_inputs = enabled.build_inputs(
        _multi_robot_sample_for_input_validation(include_gaussian=True)
    )
    assert enabled_inputs["agent_gaussian"].shape == (1, 2, 13, 28, 40)
    assert enabled_inputs["agent_gaussian"].dtype == torch.float16

    disabled = _bare_multi_robot_model_for_input_validation(enable_gaussian=False)
    disabled_inputs = disabled.build_inputs(
        _multi_robot_sample_for_input_validation(include_gaussian=False)
    )
    assert disabled_inputs["agent_gaussian"] is None


def test_geometry_action_preprocessing_is_permutation_equivariant():
    torch.manual_seed(7)
    model = _tiny_action_expert().eval()
    data = _inputs(3)
    pre = model.pre_dit(
        data["action"],
        data["timestep"],
        data["context"],
        data["context_mask"],
        agent_states=data["state"],
        agent_geometry=data["geometry"],
        agent_ids=data["ids"],
    )
    permutation = torch.tensor([2, 0, 1])
    permuted = model.pre_dit(
        data["action"][:, permutation],
        data["timestep"],
        data["context"],
        data["context_mask"],
        agent_states=data["state"][:, permutation],
        agent_geometry=data["geometry"][:, permutation],
        agent_ids=data["ids"][:, permutation],
    )
    inverse = torch.argsort(permutation)
    horizon = data["action"].shape[2]
    action_tokens = pre["tokens"][:, : 3 * horizon].reshape(1, 3, horizon, -1)
    permuted_action_tokens = permuted["tokens"][:, : 3 * horizon].reshape(
        1, 3, horizon, -1
    )
    assert torch.allclose(action_tokens, permuted_action_tokens[:, inverse])
    assert torch.allclose(pre["tokens"][:, 3 * horizon :], permuted["tokens"][:, 3 * horizon :])


def test_direct_multi_agent_action_forward_is_rejected():
    model = _tiny_action_expert().eval()
    data = _inputs(2)
    with pytest.raises(RuntimeError, match="factorized agent-local/Hub"):
        model(
            data["action"],
            data["timestep"],
            data["context"],
            data["context_mask"],
            agent_states=data["state"],
            agent_geometry=data["geometry"],
            agent_ids=data["ids"],
        )


def test_geometry_mode_fails_when_geometry_is_missing():
    model = _tiny_action_expert().eval()
    data = _inputs(2)
    with pytest.raises(ValueError, match="requires `agent_geometry`"):
        model.pre_dit(
            data["action"],
            data["timestep"],
            data["context"],
            data["context_mask"],
            agent_states=data["state"],
            agent_ids=data["ids"],
        )


def test_dynamic_simplex_requires_permutation_ids():
    model = _tiny_action_expert(agent_encoding_mode="dynamic_simplex").eval()
    data = _inputs(3)
    with pytest.raises(ValueError, match="permutation"):
        model.pre_dit(
            data["action"],
            data["timestep"],
            data["context"],
            data["context_mask"],
            agent_states=data["state"],
            agent_ids=torch.tensor([[0, 0, 2]]),
        )


def test_hub_ablation_preserves_state_dict_schema():
    enabled = _tiny_action_expert(hub_enabled=True)
    disabled = _tiny_action_expert(hub_enabled=False)
    enabled_state = enabled.state_dict()
    disabled_state = disabled.state_dict()
    assert enabled_state.keys() == disabled_state.keys()
    assert {
        key: tuple(value.shape) for key, value in enabled_state.items()
    } == {
        key: tuple(value.shape) for key, value in disabled_state.items()
    }
    assert enabled.num_hub_tokens_for(4) == 8
    assert disabled.num_hub_tokens_for(4) == 0


def _bare_mot():
    mot = MoT.__new__(MoT)
    torch.nn.Module.__init__(mot)
    mot.num_heads = 2
    mot.mot_checkpoint_mixed_attn = False
    return mot


def _factorized_tensors(
    *,
    num_agents,
    horizon=3,
    num_hubs=0,
    batch_size=2,
    video_len=5,
    hidden_dim=16,
    requires_grad=False,
):
    action_len = num_agents * horizon + num_hubs

    def make(*shape):
        return torch.randn(*shape, requires_grad=requires_grad)

    return {
        "q_action": make(batch_size, action_len, hidden_dim),
        "k_action": make(batch_size, action_len, hidden_dim),
        "v_action": make(batch_size, action_len, hidden_dim),
        "k_video": make(batch_size, video_len, hidden_dim),
        "v_video": make(batch_size, video_len, hidden_dim),
    }


def _dense_sparse_reference(
    mot,
    tensors,
    *,
    num_agents,
    horizon,
    num_hubs,
    first_frame_tokens,
):
    q_action = tensors["q_action"]
    k_action = tensors["k_action"]
    v_action = tensors["v_action"]
    k_video = tensors["k_video"]
    v_video = tensors["v_video"]
    num_action_tokens = num_agents * horizon
    action_len = num_action_tokens + num_hubs
    video_len = k_video.shape[1]
    mask = torch.zeros(
        (action_len, video_len + action_len),
        dtype=torch.bool,
        device=q_action.device,
    )
    for agent_idx in range(num_agents):
        query = slice(agent_idx * horizon, (agent_idx + 1) * horizon)
        own_keys = slice(
            video_len + agent_idx * horizon,
            video_len + (agent_idx + 1) * horizon,
        )
        mask[query, :first_frame_tokens] = True
        mask[query, own_keys] = True
        if num_hubs:
            mask[query, video_len + num_action_tokens :] = True
    if num_hubs:
        hub_queries = slice(num_action_tokens, action_len)
        mask[hub_queries, :first_frame_tokens] = True
        mask[hub_queries, video_len:] = True
    return mot._mixed_attention(
        q_cat=q_action,
        k_cat=torch.cat([k_video, k_action], dim=1),
        v_cat=torch.cat([v_video, v_action], dim=1),
        attention_mask=mask,
    )


@pytest.mark.parametrize("num_agents", [1, 2, 3, 4])
@pytest.mark.parametrize("hub_enabled", [False, True])
def test_factorized_attention_matches_dense_sparse_reference(num_agents, hub_enabled):
    torch.manual_seed(100 + num_agents + int(hub_enabled))
    mot = _bare_mot()
    horizon = 3
    num_hubs = 2 * num_agents if hub_enabled else 0
    first_frame_tokens = 2
    tensors = _factorized_tensors(
        num_agents=num_agents,
        horizon=horizon,
        num_hubs=num_hubs,
    )
    factorized = mot._factorized_multi_agent_attention(
        **tensors,
        num_agents=num_agents,
        horizon=horizon,
        num_hub_tokens=num_hubs,
        first_frame_tokens=first_frame_tokens,
    )
    reference = _dense_sparse_reference(
        mot,
        tensors,
        num_agents=num_agents,
        horizon=horizon,
        num_hubs=num_hubs,
        first_frame_tokens=first_frame_tokens,
    )
    assert torch.allclose(factorized, reference, atol=2e-5, rtol=2e-5)


def test_factorized_attention_gradients_match_dense_sparse_reference():
    torch.manual_seed(211)
    mot = _bare_mot()
    layout = {
        "num_agents": 3,
        "horizon": 2,
        "num_hubs": 6,
        "first_frame_tokens": 3,
    }
    factorized_tensors = _factorized_tensors(
        num_agents=layout["num_agents"],
        horizon=layout["horizon"],
        num_hubs=layout["num_hubs"],
        requires_grad=True,
    )
    reference_tensors = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in factorized_tensors.items()
    }
    factorized = mot._factorized_multi_agent_attention(
        **factorized_tensors,
        num_agents=layout["num_agents"],
        horizon=layout["horizon"],
        num_hub_tokens=layout["num_hubs"],
        first_frame_tokens=layout["first_frame_tokens"],
    )
    reference = _dense_sparse_reference(mot, reference_tensors, **layout)
    probe = torch.randn_like(factorized)
    factorized_grads = torch.autograd.grad(
        (factorized * probe).sum(), tuple(factorized_tensors.values())
    )
    reference_grads = torch.autograd.grad(
        (reference * probe).sum(), tuple(reference_tensors.values())
    )
    for factorized_grad, reference_grad in zip(factorized_grads, reference_grads):
        assert torch.allclose(
            factorized_grad, reference_grad, atol=3e-5, rtol=3e-5
        )


def test_factorized_attention_uses_only_sparse_unmasked_blocks():
    torch.manual_seed(307)
    mot = _bare_mot()
    num_agents = 4
    horizon = 3
    num_hubs = 8
    first_frame_tokens = 2
    tensors = _factorized_tensors(
        num_agents=num_agents,
        horizon=horizon,
        num_hubs=num_hubs,
        batch_size=2,
    )
    calls = []
    original_attention = mot._mixed_attention

    def recording_attention(*, q_cat, k_cat, v_cat, attention_mask=None):
        calls.append((tuple(q_cat.shape), tuple(k_cat.shape), attention_mask))
        return original_attention(
            q_cat=q_cat,
            k_cat=k_cat,
            v_cat=v_cat,
            attention_mask=attention_mask,
        )

    mot._mixed_attention = recording_attention
    output = mot._factorized_multi_agent_attention(
        **tensors,
        num_agents=num_agents,
        horizon=horizon,
        num_hub_tokens=num_hubs,
        first_frame_tokens=first_frame_tokens,
    )
    assert output.shape == tensors["q_action"].shape
    assert not hasattr(FastWAMMultiRobot, "_build_multi_robot_attention_mask")
    assert len(calls) == 2
    assert all(mask is None for _, _, mask in calls)
    assert calls[0][:2] == (
        (2 * num_agents, horizon, 16),
        (2 * num_agents, first_frame_tokens + horizon + num_hubs, 16),
    )
    assert calls[1][:2] == (
        (2, num_hubs, 16),
        (2, first_frame_tokens + num_agents * horizon + num_hubs, 16),
    )


def _tiny_mot_pair(*, enable_gaussian=False):
    return MoT(
        mixtures={
            "video": _tiny_action_expert(
                hub_enabled=False, agent_encoding_mode="none"
            ),
            "action": _tiny_action_expert(enable_gaussian=enable_gaussian),
        },
        mot_checkpoint_mixed_attn=False,
    )


def _unit_freqs(seq_len):
    return torch.ones((seq_len, 1, 4), dtype=torch.complex128)


def test_joint_public_path_has_only_video_mask_and_factorized_action_calls():
    torch.manual_seed(349)
    mot = _tiny_mot_pair().train()
    batch_size = 1
    video_len = 3
    num_agents = 2
    horizon = 2
    num_hubs = 4
    action_len = num_agents * horizon + num_hubs
    video_tokens = torch.randn(batch_size, video_len, 32, requires_grad=True)
    action_tokens = torch.randn(batch_size, action_len, 32, requires_grad=True)
    calls = []
    original_attention = mot._mixed_attention

    def recording_attention(*, q_cat, k_cat, v_cat, attention_mask=None):
        calls.append(
            (
                tuple(q_cat.shape),
                tuple(k_cat.shape),
                None if attention_mask is None else tuple(attention_mask.shape),
            )
        )
        return original_attention(
            q_cat=q_cat,
            k_cat=k_cat,
            v_cat=v_cat,
            attention_mask=attention_mask,
        )

    mot._mixed_attention = recording_attention
    output = mot.forward_multi_agent_joint(
        video_tokens=video_tokens,
        action_tokens=action_tokens,
        video_freqs=_unit_freqs(video_len),
        action_freqs=_unit_freqs(action_len),
        video_t_mod=torch.zeros(batch_size, 6, 32),
        action_t_mod=torch.zeros(batch_size, 6, 32),
        video_context_payload=None,
        action_context_payload=None,
        video_attention_mask=torch.ones(video_len, video_len, dtype=torch.bool),
        num_agents=num_agents,
        horizon=horizon,
        num_hub_tokens=num_hubs,
        first_frame_tokens=2,
    )
    assert output["video"].shape == video_tokens.shape
    assert output["action"].shape == action_tokens.shape
    assert len(calls) == 3 * mot.num_layers
    assert [mask for _, _, mask in calls].count((video_len, video_len)) == mot.num_layers
    assert [mask for _, _, mask in calls].count(None) == 2 * mot.num_layers
    assert all(
        mask in {None, (video_len, video_len)} for _, _, mask in calls
    )
    (output["video"].square().mean() + output["action"].square().mean()).backward()
    assert torch.isfinite(video_tokens.grad).all()
    assert torch.isfinite(action_tokens.grad).all()


def test_cached_video_public_path_uses_factorized_action_calls():
    torch.manual_seed(367)
    mot = _tiny_mot_pair().train()
    batch_size = 1
    video_len = 3
    num_agents = 3
    horizon = 2
    num_hubs = 6
    action_len = num_agents * horizon + num_hubs
    action_tokens = torch.randn(batch_size, action_len, 32, requires_grad=True)
    video_kv_cache = [
        {
            "k": torch.randn(batch_size, video_len, 16),
            "v": torch.randn(batch_size, video_len, 16),
        }
        for _ in range(mot.num_layers)
    ]
    calls = []
    original_attention = mot._mixed_attention

    def recording_attention(*, q_cat, k_cat, v_cat, attention_mask=None):
        calls.append(attention_mask)
        return original_attention(
            q_cat=q_cat,
            k_cat=k_cat,
            v_cat=v_cat,
            attention_mask=attention_mask,
        )

    mot._mixed_attention = recording_attention
    output = mot.forward_multi_agent_action_with_video_cache(
        action_tokens=action_tokens,
        action_freqs=_unit_freqs(action_len),
        action_t_mod=torch.zeros(batch_size, 6, 32),
        action_context_payload=None,
        video_kv_cache=video_kv_cache,
        num_agents=num_agents,
        horizon=horizon,
        num_hub_tokens=num_hubs,
        first_frame_tokens=2,
    )
    assert output.shape == action_tokens.shape
    assert len(calls) == 2 * mot.num_layers
    assert all(mask is None for mask in calls)
    output.square().mean().backward()
    assert torch.isfinite(action_tokens.grad).all()


def test_hub0_agent0_output_and_gradient_ignore_agent1():
    torch.manual_seed(401)
    mot = _bare_mot()
    num_agents = 2
    horizon = 3
    tensors = _factorized_tensors(
        num_agents=num_agents,
        horizon=horizon,
        num_hubs=0,
        batch_size=1,
        requires_grad=True,
    )
    perturbed = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in tensors.items()
    }
    with torch.no_grad():
        for name in ("q_action", "k_action", "v_action"):
            perturbed[name][:, horizon : 2 * horizon].add_(7.0)

    output = mot._factorized_multi_agent_attention(
        **tensors,
        num_agents=num_agents,
        horizon=horizon,
        num_hub_tokens=0,
        first_frame_tokens=2,
    )
    perturbed_output = mot._factorized_multi_agent_attention(
        **perturbed,
        num_agents=num_agents,
        horizon=horizon,
        num_hub_tokens=0,
        first_frame_tokens=2,
    )
    assert torch.equal(output[:, :horizon], perturbed_output[:, :horizon])

    action_names = ("q_action", "k_action", "v_action")
    gradients = torch.autograd.grad(
        output[:, :horizon].square().sum(),
        tuple(tensors[name] for name in action_names),
    )
    perturbed_gradients = torch.autograd.grad(
        perturbed_output[:, :horizon].square().sum(),
        tuple(perturbed[name] for name in action_names),
    )
    for gradient, perturbed_gradient in zip(gradients, perturbed_gradients):
        assert torch.equal(
            gradient[:, :horizon], perturbed_gradient[:, :horizon]
        )
        assert torch.count_nonzero(gradient[:, horizon:]) == 0
        assert torch.count_nonzero(perturbed_gradient[:, horizon:]) == 0


def test_hub_gather_then_broadcast_crosses_agents_on_next_layer():
    torch.manual_seed(509)
    mot = _bare_mot()
    num_agents = 2
    horizon = 2
    num_hubs = 2
    tensors = _factorized_tensors(
        num_agents=num_agents,
        horizon=horizon,
        num_hubs=num_hubs,
        batch_size=1,
    )
    perturbed = {name: value.clone() for name, value in tensors.items()}
    perturbed["v_action"][:, horizon : 2 * horizon].add_(9.0)

    first = mot._factorized_multi_agent_attention(
        **tensors,
        num_agents=num_agents,
        horizon=horizon,
        num_hub_tokens=num_hubs,
        first_frame_tokens=2,
    )
    perturbed_first = mot._factorized_multi_agent_attention(
        **perturbed,
        num_agents=num_agents,
        horizon=horizon,
        num_hub_tokens=num_hubs,
        first_frame_tokens=2,
    )
    num_action_tokens = num_agents * horizon
    assert torch.equal(first[:, :horizon], perturbed_first[:, :horizon])
    assert not torch.allclose(
        first[:, num_action_tokens:], perturbed_first[:, num_action_tokens:]
    )

    second = mot._factorized_multi_agent_attention(
        q_action=first,
        k_action=first,
        v_action=first,
        k_video=tensors["k_video"],
        v_video=tensors["v_video"],
        num_agents=num_agents,
        horizon=horizon,
        num_hub_tokens=num_hubs,
        first_frame_tokens=2,
    )
    perturbed_second = mot._factorized_multi_agent_attention(
        q_action=perturbed_first,
        k_action=perturbed_first,
        v_action=perturbed_first,
        k_video=tensors["k_video"],
        v_video=tensors["v_video"],
        num_agents=num_agents,
        horizon=horizon,
        num_hub_tokens=num_hubs,
        first_frame_tokens=2,
    )
    assert not torch.allclose(second[:, :horizon], perturbed_second[:, :horizon])


@pytest.mark.parametrize("num_agents", [1, 2, 3, 4])
@pytest.mark.parametrize("hub_enabled", [False, True])
def test_factorized_dynamic_agent_gradients_are_finite(num_agents, hub_enabled):
    torch.manual_seed(601 + num_agents + int(hub_enabled))
    mot = _bare_mot()
    num_hubs = 2 * num_agents if hub_enabled else 0
    tensors = _factorized_tensors(
        num_agents=num_agents,
        horizon=2,
        num_hubs=num_hubs,
        requires_grad=True,
    )
    output = mot._factorized_multi_agent_attention(
        **tensors,
        num_agents=num_agents,
        horizon=2,
        num_hub_tokens=num_hubs,
        first_frame_tokens=2,
    )
    gradients = torch.autograd.grad(output.square().mean(), tuple(tensors.values()))
    for gradient in gradients:
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_legacy_fixed_hub_bank_converts_to_shared_seed():
    bank = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    upgraded = FastWAMMultiRobot._upgrade_legacy_hub_state(
        {"mixtures.action.hub_tokens": bank}
    )
    assert "mixtures.action.hub_tokens" not in upgraded
    assert torch.equal(
        upgraded["mixtures.action.hub_seed"], bank.mean(dim=0, keepdim=True)
    )


def _bare_checkpoint_model(
    *,
    enable_gaussian=False,
    training_mode="action_only_cache",
    trainable_scope="action",
):
    model = FastWAMMultiRobot.__new__(FastWAMMultiRobot)
    torch.nn.Module.__init__(model)
    model.mot = _tiny_mot_pair(enable_gaussian=enable_gaussian)
    model.action_expert = model.mot.mixtures["action"]
    model.video_expert = model.mot.mixtures["video"]
    model.training_mode = training_mode
    model._trainable_scope = trainable_scope
    model.torch_dtype = torch.float32
    model._loaded_base_checkpoint = None
    model._loaded_base_checkpoint_sha256 = None
    model._loaded_base_checkpoint_descriptor = None
    model._loaded_base_checkpoint_can_restore_sparse = False
    model.requires_grad_(False)
    if trainable_scope == "dit":
        model.mot.requires_grad_(True)
    elif trainable_scope == "action":
        model.action_expert.requires_grad_(True)
    elif trainable_scope == "hub_io":
        model.action_expert.action_encoder.requires_grad_(True)
        model.action_expert.agent_state_encoder.requires_grad_(True)
        if model.action_expert.agent_geometry_encoder is not None:
            model.action_expert.agent_geometry_encoder.requires_grad_(True)
        if model.action_expert.gaussian_adapter is not None:
            model.action_expert.gaussian_adapter.requires_grad_(True)
        if model.action_expert.gaussian_gate is not None:
            model.action_expert.gaussian_gate.requires_grad_(True)
        model.action_expert.head.requires_grad_(True)
        model.action_expert.hub_seed.requires_grad_(
            model.action_expert.hub_enabled
        )
    else:
        raise ValueError(trainable_scope)
    return model


def _bare_gaussian_checkpoint_model():
    return _bare_checkpoint_model(enable_gaussian=True)


def _cloned_state(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _native_full_payload(
    model,
    *,
    training_mode=None,
    trainable_scope=None,
):
    return {
        "format": "fastwam_multi_robot_v2",
        "state_kind": "full",
        "training_mode": training_mode or model.training_mode,
        "trainable_scope": trainable_scope or model._trainable_scope,
        "multi_robot_architecture": model._multi_robot_architecture_metadata(),
        "base_checkpoint": None,
        "mot": _cloned_state(model.mot),
    }


def _legacy_native_v2_full_payload(model):
    metadata = model._multi_robot_architecture_metadata()
    legacy_metadata = {
        key: metadata[key]
        for key in model._LEGACY_NATIVE_V2_ARCHITECTURE_KEYS
    }
    return {
        "format": "fastwam_multi_robot_v2",
        "step": 5000,
        "torch_dtype": str(model.torch_dtype),
        "trainable_scope": "dit",
        "training_mode": "joint",
        "base_checkpoint": "/frozen/official/base.pt",
        "multi_robot_architecture": legacy_metadata,
        "mot": _cloned_state(model.mot),
    }


def _legacy_mot_payload(model):
    state = _cloned_state(model.mot)
    action_prefix = "mixtures.action."
    for key in list(state):
        if not key.startswith(action_prefix):
            continue
        relative_key = key[len(action_prefix) :]
        if any(
            relative_key.startswith(prefix)
            for prefix in model.action_expert.ACTION_BACKBONE_SKIP_PREFIXES
        ):
            state.pop(key)
    return {"mot": state}


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("training_mode", "joint"),
        ("trainable_scope", "dit"),
    ],
)
def test_native_v2_top_level_rejects_treatment_mismatch(
    tmp_path, field, wrong_value
):
    model = _bare_checkpoint_model()
    payload = _native_full_payload(model)
    payload[field] = wrong_value
    checkpoint = tmp_path / f"wrong-{field}.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=f"treatment mismatch for {field}"):
        model.load_checkpoint(checkpoint)


def test_native_v2_top_level_rejects_architecture_extra_key(tmp_path):
    model = _bare_checkpoint_model()
    payload = _native_full_payload(model)
    payload["multi_robot_architecture"]["untracked_treatment"] = True
    checkpoint = tmp_path / "wrong-architecture.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="unexpected metadata keys"):
        model.load_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("hub_position_scale", 0.5),
        ("agent_rope_dim", 2),
        ("agent_phase_scale", 3.0),
    ],
)
def test_native_v2_top_level_pins_parameter_free_architecture(
    tmp_path, field, wrong_value
):
    model = _bare_checkpoint_model()
    payload = _native_full_payload(model)
    payload["multi_robot_architecture"][field] = wrong_value
    checkpoint = tmp_path / f"wrong-{field}.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=f"architecture mismatch for {field}"):
        model.load_checkpoint(checkpoint)


def test_native_v2_full_load_caches_its_sha256(tmp_path):
    source = _bare_checkpoint_model(
        training_mode="joint", trainable_scope="dit"
    )
    checkpoint = tmp_path / "full.pt"
    torch.save(_native_full_payload(source), checkpoint)
    target = _bare_checkpoint_model(
        training_mode="joint", trainable_scope="dit"
    )

    target.load_checkpoint(checkpoint)

    assert target._loaded_base_checkpoint == str(checkpoint.resolve())
    assert target._loaded_base_checkpoint_sha256 == target._checkpoint_sha256(
        checkpoint
    )


@pytest.mark.parametrize("corruption", ["missing", "shape", "dtype"])
def test_native_v2_full_state_is_exact(tmp_path, corruption):
    source = _bare_checkpoint_model(
        training_mode="joint", trainable_scope="dit"
    )
    payload = _native_full_payload(source)
    key = next(
        name for name, value in payload["mot"].items() if value.numel() > 2
    )
    if corruption == "missing":
        payload["mot"].pop(key)
    elif corruption == "shape":
        payload["mot"][key] = payload["mot"][key].reshape(-1)[:1]
    else:
        payload["mot"][key] = payload["mot"][key].to(torch.float64)
    checkpoint = tmp_path / f"full-{corruption}.pt"
    torch.save(payload, checkpoint)
    target = _bare_checkpoint_model(
        training_mode="joint", trainable_scope="dit"
    )

    with pytest.raises(ValueError, match=f"{corruption}_mismatches|{corruption}="):
        target.load_checkpoint(checkpoint)


def test_exact_legacy_native_v2_gau0_full_checkpoint_loads_without_hash(tmp_path):
    source = _bare_checkpoint_model(
        training_mode="joint", trainable_scope="dit"
    )
    checkpoint = tmp_path / "legacy-native-v2-full.pt"
    torch.save(_legacy_native_v2_full_payload(source), checkpoint)
    target = _bare_checkpoint_model(
        training_mode="joint", trainable_scope="dit"
    )

    target.load_checkpoint(checkpoint, record_checkpoint_sha256=False)

    assert target._loaded_base_checkpoint == str(checkpoint.resolve())
    assert target._loaded_base_checkpoint_sha256 is None
    for key, value in source.mot.state_dict().items():
        assert torch.equal(value, target.mot.state_dict()[key]), key


def test_legacy_native_v2_full_checkpoint_is_rejected_for_gau1(tmp_path):
    source = _bare_checkpoint_model(
        training_mode="joint", trainable_scope="dit"
    )
    checkpoint = tmp_path / "legacy-native-v2-gau1.pt"
    torch.save(_legacy_native_v2_full_payload(source), checkpoint)
    target = _bare_checkpoint_model(
        enable_gaussian=True,
        training_mode="joint",
        trainable_scope="dit",
    )

    with pytest.raises(ValueError, match="missing 'action_attention_topology'"):
        target.load_checkpoint(checkpoint, record_checkpoint_sha256=False)


@pytest.mark.parametrize(
    "corruption", ["top_level_extra", "metadata_missing", "state_missing"]
)
def test_legacy_native_v2_full_checkpoint_remains_fail_closed(
    tmp_path, corruption
):
    source = _bare_checkpoint_model(
        training_mode="joint", trainable_scope="dit"
    )
    payload = _legacy_native_v2_full_payload(source)
    if corruption == "top_level_extra":
        payload["untracked"] = True
        error = "missing 'action_attention_topology'"
    elif corruption == "metadata_missing":
        payload["multi_robot_architecture"].pop("hub_token_ratio")
        error = "missing 'action_attention_topology'"
    else:
        payload["mot"].pop(next(iter(payload["mot"])))
        error = "missing="
    checkpoint = tmp_path / f"legacy-native-v2-{corruption}.pt"
    torch.save(payload, checkpoint)
    target = _bare_checkpoint_model(
        training_mode="joint", trainable_scope="dit"
    )

    with pytest.raises(ValueError, match=error):
        target.load_checkpoint(checkpoint, record_checkpoint_sha256=False)


def test_sparse_v2_round_trip_pins_base_and_exact_trainable_names(tmp_path):
    torch.manual_seed(7001)
    source = _bare_checkpoint_model()
    base_checkpoint = tmp_path / "official-base.pt"
    torch.save(_legacy_mot_payload(source), base_checkpoint)
    source.load_checkpoint(base_checkpoint)
    with torch.no_grad():
        for parameter in source.action_expert.parameters():
            parameter.add_(0.25)

    sparse_checkpoint = tmp_path / "sparse.pt"
    source.save_checkpoint(sparse_checkpoint, step=9)
    payload = torch.load(sparse_checkpoint, map_location="cpu")
    expected_names = source._expected_trainable_parameter_names()
    assert payload["state_kind"] == "sparse_delta"
    assert payload["trainable_parameter_names"] == expected_names
    assert sorted(payload["mot_trainable"]) == expected_names
    assert payload["base_checkpoint"] == {
        "path": str(base_checkpoint.resolve()),
        "sha256": source._checkpoint_sha256(base_checkpoint),
        "role": "base_dependency",
    }

    torch.manual_seed(8123)
    target = _bare_checkpoint_model()
    target.load_checkpoint(sparse_checkpoint)
    assert target._loaded_base_checkpoint_sha256 == payload["base_checkpoint"][
        "sha256"
    ]
    source_state = source.mot.state_dict()
    target_state = target.mot.state_dict()
    assert source_state.keys() == target_state.keys()
    for key in source_state:
        assert torch.equal(source_state[key], target_state[key]), key


def test_action_only_can_publish_self_contained_full_checkpoint(tmp_path):
    source = _bare_checkpoint_model(trainable_scope="action")
    checkpoint = tmp_path / "formal-full.pt"

    source.save_checkpoint(checkpoint, checkpoint_state_kind="full", step=31)

    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["state_kind"] == "full"
    assert payload["base_checkpoint"] is None
    assert "mot" in payload
    assert "mot_trainable" not in payload
    target = _bare_checkpoint_model(trainable_scope="action")
    target.load_checkpoint(checkpoint)
    for key, value in source.mot.state_dict().items():
        assert torch.equal(value, target.mot.state_dict()[key]), key


def test_action_scope_full_checkpoint_loads_for_inference_without_scope_mutation(
    tmp_path,
):
    source = _bare_checkpoint_model(trainable_scope="action")
    checkpoint = tmp_path / "action-full.pt"
    source.save_checkpoint(checkpoint, checkpoint_state_kind="full", step=31)
    target = _bare_checkpoint_model(trainable_scope="dit")

    target.load_checkpoint(checkpoint, validate_trainable_scope=False)

    assert target._trainable_scope == "dit"
    for key, value in source.mot.state_dict().items():
        assert torch.equal(value, target.mot.state_dict()[key]), key


def test_fully_trainable_checkpoint_rejects_sparse_override(tmp_path):
    model = _bare_checkpoint_model(training_mode="joint", trainable_scope="dit")
    with pytest.raises(ValueError, match="invalid when trainable_scope='dit'"):
        model.save_checkpoint(
            tmp_path / "invalid-sparse.pt",
            checkpoint_state_kind="sparse_delta",
        )


@pytest.mark.parametrize("corruption", ["state_key", "declared_name", "dtype"])
def test_sparse_v2_rejects_incomplete_trainable_delta(tmp_path, corruption):
    source = _bare_checkpoint_model()
    base_checkpoint = tmp_path / "official-base.pt"
    torch.save(_legacy_mot_payload(source), base_checkpoint)
    source.load_checkpoint(base_checkpoint)
    checkpoint = tmp_path / "valid-sparse.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    key = payload["trainable_parameter_names"][0]
    if corruption == "state_key":
        payload["mot_trainable"].pop(key)
    elif corruption == "declared_name":
        payload["trainable_parameter_names"].remove(key)
    else:
        payload["mot_trainable"][key] = payload["mot_trainable"][key].to(
            torch.float64
        )
    corrupt_checkpoint = tmp_path / f"sparse-{corruption}.pt"
    torch.save(payload, corrupt_checkpoint)

    target = _bare_checkpoint_model()
    expected_message = (
        "mot_trainable state mismatch"
        if corruption != "declared_name"
        else "trainable contract mismatch"
    )
    with pytest.raises(ValueError, match=expected_message):
        target.load_checkpoint(corrupt_checkpoint)


def test_sparse_v2_rejects_base_hash_mismatch(tmp_path):
    source = _bare_checkpoint_model()
    base_checkpoint = tmp_path / "official-base.pt"
    torch.save(_legacy_mot_payload(source), base_checkpoint)
    source.load_checkpoint(base_checkpoint)
    checkpoint = tmp_path / "valid-sparse.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    payload["base_checkpoint"]["sha256"] = "0" * 64
    corrupt_checkpoint = tmp_path / "bad-base-hash.pt"
    torch.save(payload, corrupt_checkpoint)

    with pytest.raises(ValueError, match="Base checkpoint SHA-256 mismatch"):
        _bare_checkpoint_model().load_checkpoint(corrupt_checkpoint)


def test_sparse_v2_native_full_base_allows_different_mode_and_scope(tmp_path):
    source = _bare_checkpoint_model()
    base_checkpoint = tmp_path / "native-full-base.pt"
    torch.save(
        _native_full_payload(
            source,
            training_mode="joint",
            trainable_scope="dit",
        ),
        base_checkpoint,
    )
    source._loaded_base_checkpoint = str(base_checkpoint)
    source._loaded_base_checkpoint_can_restore_sparse = True
    sparse_checkpoint = tmp_path / "sparse-over-native.pt"
    source.save_checkpoint(sparse_checkpoint)

    target = _bare_checkpoint_model()
    target.load_checkpoint(sparse_checkpoint)
    assert target._loaded_base_checkpoint == str(base_checkpoint.resolve())
    assert target._loaded_base_checkpoint_sha256 == target._checkpoint_sha256(
        base_checkpoint
    )


def test_sparse_v2_rejects_nested_sparse_dependency(tmp_path):
    source = _bare_checkpoint_model()
    legacy_base = tmp_path / "official-base.pt"
    torch.save(_legacy_mot_payload(source), legacy_base)
    source.load_checkpoint(legacy_base)
    nested_sparse = tmp_path / "nested-sparse.pt"
    source.save_checkpoint(nested_sparse)

    source._loaded_base_checkpoint = str(nested_sparse)
    source._loaded_base_checkpoint_descriptor = None
    source._loaded_base_checkpoint_can_restore_sparse = True
    outer_sparse = tmp_path / "outer-sparse.pt"
    source.save_checkpoint(outer_sparse)

    with pytest.raises(ValueError, match="Nested sparse"):
        _bare_checkpoint_model().load_checkpoint(outer_sparse)


def test_checkpoint_dependency_cycle_is_rejected_before_load(tmp_path):
    model = _bare_checkpoint_model(
        training_mode="joint", trainable_scope="dit"
    )
    checkpoint = tmp_path / "full.pt"
    torch.save(_native_full_payload(model), checkpoint)

    with pytest.raises(ValueError, match="cycle detected"):
        model._load_checkpoint_with_role(
            checkpoint,
            load_role="base_dependency",
            active_paths={checkpoint.resolve()},
        )


def test_legacy_official_init_is_permissive_but_requires_backbone(tmp_path):
    source = _bare_checkpoint_model()
    legacy_checkpoint = tmp_path / "official.pt"
    legacy_payload = _legacy_mot_payload(source)
    assert len(legacy_payload["mot"]) < len(source.mot.state_dict())
    torch.save(legacy_payload, legacy_checkpoint)

    target = _bare_checkpoint_model()
    target.load_checkpoint(legacy_checkpoint)
    assert target._loaded_base_checkpoint_can_restore_sparse is True
    assert target._loaded_base_checkpoint_sha256 == target._checkpoint_sha256(
        legacy_checkpoint
    )

    garbage_checkpoint = tmp_path / "garbage.pt"
    one_key = next(iter(source.mot.state_dict()))
    torch.save(
        {"mot": {one_key: source.mot.state_dict()[one_key]}},
        garbage_checkpoint,
    )
    with pytest.raises(ValueError, match="minimum safe backbone coverage"):
        _bare_checkpoint_model().load_checkpoint(garbage_checkpoint)


def test_gaussian_v2_metadata_pins_adapter_architecture():
    model = _bare_gaussian_checkpoint_model()
    metadata = model._multi_robot_architecture_metadata()
    assert metadata["gaussian_stem_dim"] == 8
    assert metadata["gaussian_adapter_version"] == "conv_gn_silu_pool_v1"

    stale_metadata = dict(metadata)
    stale_metadata.pop("gaussian_stem_dim")
    with pytest.raises(ValueError, match="missing 'gaussian_stem_dim'"):
        model._validate_multi_robot_checkpoint_metadata(
            {
                "format": "fastwam_multi_robot_v2",
                "training_mode": model.training_mode,
                "trainable_scope": model._trainable_scope,
                "multi_robot_architecture": stale_metadata,
            },
            "stale-v2.pt",
        )


def test_gaussian_v2_checkpoint_requires_every_adapter_and_gate_tensor():
    model = _bare_gaussian_checkpoint_model()
    payload = {
        "format": "fastwam_multi_robot_v2",
        "multi_robot_architecture": model._multi_robot_architecture_metadata(),
    }
    state = model.mot.state_dict()
    model._validate_gaussian_v2_state(
        payload, state, path="valid.pt", label="mot"
    )

    gaussian_keys = sorted(
        key
        for key in state
        if ".gaussian_adapter." in key or key.endswith(".gaussian_gate")
    )
    assert gaussian_keys
    missing_state = dict(state)
    missing_state.pop(gaussian_keys[0])
    with pytest.raises(ValueError, match="Strict GAU1.*missing="):
        model._validate_gaussian_v2_state(
            payload, missing_state, path="missing.pt", label="mot"
        )

    wrong_shape_state = dict(state)
    wrong_shape_state[gaussian_keys[-1]] = torch.zeros(2)
    with pytest.raises(ValueError, match="shape_mismatches"):
        model._validate_gaussian_v2_state(
            payload, wrong_shape_state, path="shape.pt", label="mot_trainable"
        )


def test_official_legacy_checkpoint_keeps_permissive_gaussian_loading():
    model = _bare_gaussian_checkpoint_model()
    # Official checkpoints have no native multi-robot v2 envelope. Their
    # shared backbone is intentionally loaded shape-tolerantly, while the new
    # treatment parameters remain deterministic from the pre-instantiation seed.
    model._validate_gaussian_v2_state(
        {"mot": {}}, {}, path="official-legacy.pt", label="mot"
    )


def test_joint_loss_draws_action_corruption_before_video_noise(monkeypatch):
    """VG0/VG1 validation must use the same seeded action corruption."""

    events = []

    class _Sentinel(Exception):
        pass

    class _BareModel:
        @staticmethod
        def _prepare_noisy_action(inputs):
            events.append("action")
            action = inputs["action"]
            timestep = torch.zeros(action.shape[0])
            return action, action, timestep

    def stop_at_video_noise(value):
        events.append("video")
        raise _Sentinel

    monkeypatch.setattr(torch, "randn_like", stop_at_video_noise)
    inputs = {
        "input_latents": torch.zeros(1, 2, 2, 2, 2),
        "action": torch.zeros(1, 2, 4, 3),
    }

    with pytest.raises(_Sentinel):
        FastWAMMultiRobot._training_loss_joint(_BareModel(), inputs)

    assert events == ["action", "video"]


def test_mot_mixed_attention_accepts_batch_specific_mask():
    mot = MoT.__new__(MoT)
    torch.nn.Module.__init__(mot)
    mot.num_heads = 2
    mot.mot_checkpoint_mixed_attn = False
    q = torch.randn(2, 4, 16)
    k = torch.randn(2, 4, 16)
    v = torch.randn(2, 4, 16)
    mask = torch.ones(2, 4, 4, dtype=torch.bool)
    mask[1, :, 3] = False
    output = mot._mixed_attention(q, k, v, mask)
    assert output.shape == q.shape
    assert torch.isfinite(output).all()
