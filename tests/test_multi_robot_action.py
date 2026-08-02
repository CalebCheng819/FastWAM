import pytest
import torch
from types import SimpleNamespace

from fastwam.models.wan22.fastwam_multi_robot import FastWAMMultiRobot
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.multi_agent_action_dit import (
    MultiAgentActionDiT,
    regular_simplex_vertices,
)


def _tiny_action_expert(*, hub_enabled=True, agent_encoding_mode="geometry"):
    return MultiAgentActionDiT(
        action_dim=3,
        state_dim=4,
        agent_geometry_dim=7 if agent_encoding_mode == "geometry" else 0,
        agent_encoding_mode=agent_encoding_mode,
        hub_enabled=hub_enabled,
        hub_token_ratio=2.0,
        agent_rope_dim=4,
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


def _tiny_mot_pair():
    return MoT(
        mixtures={
            "video": _tiny_action_expert(
                hub_enabled=False, agent_encoding_mode="none"
            ),
            "action": _tiny_action_expert(),
        },
        mot_checkpoint_mixed_attn=False,
    )


def _unit_freqs(seq_len):
    return torch.ones((seq_len, 1, 4), dtype=torch.complex128)


def _first_frame_causal_mask(video_len, first_frame_tokens):
    mask = torch.ones(video_len, video_len, dtype=torch.bool)
    mask[:first_frame_tokens, first_frame_tokens:] = False
    return mask


def _run_tiny_joint(
    mot,
    *,
    video_tokens,
    action_tokens,
    num_agents,
    horizon,
    num_hubs,
    first_frame_tokens,
    conditioning_mode="none",
    detach_hub_kv=True,
):
    batch_size, video_len, hidden_dim = video_tokens.shape
    return mot.forward_multi_agent_joint(
        video_tokens=video_tokens,
        action_tokens=action_tokens,
        video_freqs=_unit_freqs(video_len),
        action_freqs=_unit_freqs(action_tokens.shape[1]),
        video_t_mod=torch.zeros(batch_size, 6, hidden_dim),
        action_t_mod=torch.zeros(batch_size, 6, hidden_dim),
        video_context_payload=None,
        action_context_payload=None,
        video_attention_mask=_first_frame_causal_mask(
            video_len, first_frame_tokens
        ),
        num_agents=num_agents,
        horizon=horizon,
        num_hub_tokens=num_hubs,
        first_frame_tokens=first_frame_tokens,
        video_conditioning_mode=conditioning_mode,
        video_conditioning_start_layer=1,
        video_conditioning_detach_hub_kv=detach_hub_kv,
    )


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


def test_hub_conditioned_video_attention_is_lagged_sparse_and_observed_safe():
    torch.manual_seed(353)
    mot = _tiny_mot_pair().eval()
    batch_size = 1
    video_len = 3
    first_frame_tokens = 1
    num_agents = 2
    horizon = 3
    num_hubs = 4
    action_len = num_agents * horizon + num_hubs
    video_tokens = torch.randn(batch_size, video_len, 32)
    action_tokens = torch.randn(batch_size, action_len, 32)
    calls = []
    action_layer_kv = []
    original_attention = mot._mixed_attention
    original_build_attention_io = mot._build_expert_attention_io

    def recording_build_attention_io(**kwargs):
        result = original_build_attention_io(**kwargs)
        if kwargs["expert"] is mot.mixtures["action"]:
            action_layer_kv.append((result[1].detach().clone(), result[2].detach().clone()))
        return result

    def recording_attention(*, q_cat, k_cat, v_cat, attention_mask=None):
        calls.append(
            {
                "q": tuple(q_cat.shape),
                "k": tuple(k_cat.shape),
                "k_tensor": k_cat.detach().clone(),
                "v_tensor": v_cat.detach().clone(),
                "mask": None if attention_mask is None else attention_mask.clone(),
            }
        )
        return original_attention(
            q_cat=q_cat,
            k_cat=k_cat,
            v_cat=v_cat,
            attention_mask=attention_mask,
        )

    mot._build_expert_attention_io = recording_build_attention_io
    mot._mixed_attention = recording_attention
    output = _run_tiny_joint(
        mot,
        video_tokens=video_tokens,
        action_tokens=action_tokens,
        num_agents=num_agents,
        horizon=horizon,
        num_hubs=num_hubs,
        first_frame_tokens=first_frame_tokens,
        conditioning_mode="hub_kv_lagged",
    )
    assert output["video"].shape == video_tokens.shape
    assert output["action"].shape == action_tokens.shape
    assert len(calls) == 3 * mot.num_layers
    assert len(action_layer_kv) == mot.num_layers

    layer0_video = calls[0]
    layer1_video = calls[3]
    assert layer0_video["k"] == (batch_size, video_len, 16)
    assert layer0_video["mask"].shape == (video_len, video_len)
    assert layer1_video["k"] == (
        batch_size,
        video_len + num_hubs,
        16,
    )
    assert layer1_video["mask"].shape == (
        video_len,
        video_len + num_hubs,
    )
    assert torch.equal(
        layer1_video["mask"][:, :video_len],
        _first_frame_causal_mask(video_len, first_frame_tokens),
    )
    assert not layer1_video["mask"][:first_frame_tokens, -num_hubs:].any()
    assert layer1_video["mask"][first_frame_tokens:, -num_hubs:].all()
    assert torch.equal(
        layer1_video["k_tensor"][:, video_len:],
        action_layer_kv[1][0][:, -num_hubs:],
    )
    assert torch.equal(
        layer1_video["v_tensor"][:, video_len:],
        action_layer_kv[1][1][:, -num_hubs:],
    )


def test_hub_conditioning_changes_only_future_video_not_action_or_observation():
    torch.manual_seed(359)
    mot = _tiny_mot_pair().eval()
    batch_size = 1
    video_len = 3
    first_frame_tokens = 1
    num_agents = 2
    horizon = 3
    num_hubs = 4
    action_len = num_agents * horizon + num_hubs
    video_tokens = torch.randn(batch_size, video_len, 32)
    action_tokens = torch.randn(batch_size, action_len, 32)

    baseline = _run_tiny_joint(
        mot,
        video_tokens=video_tokens,
        action_tokens=action_tokens,
        num_agents=num_agents,
        horizon=horizon,
        num_hubs=num_hubs,
        first_frame_tokens=first_frame_tokens,
    )
    conditioned = _run_tiny_joint(
        mot,
        video_tokens=video_tokens,
        action_tokens=action_tokens,
        num_agents=num_agents,
        horizon=horizon,
        num_hubs=num_hubs,
        first_frame_tokens=first_frame_tokens,
        conditioning_mode="hub_kv_lagged",
    )

    assert torch.equal(
        baseline["video"][:, :first_frame_tokens],
        conditioned["video"][:, :first_frame_tokens],
    )
    assert torch.equal(baseline["action"], conditioned["action"])
    assert not torch.allclose(
        baseline["video"][:, first_frame_tokens:],
        conditioned["video"][:, first_frame_tokens:],
    )


def test_detached_hub_condition_has_value_dependency_without_video_gradient_leak():
    torch.manual_seed(361)
    layout = {
        "num_agents": 2,
        "horizon": 3,
        "num_hubs": 4,
        "first_frame_tokens": 1,
    }

    def video_gradient(detach_hub_kv):
        mot = _tiny_mot_pair().train()
        video_tokens = torch.randn(1, 3, 32, requires_grad=True)
        action_tokens = torch.randn(1, 10, 32, requires_grad=True)
        output = _run_tiny_joint(
            mot,
            video_tokens=video_tokens,
            action_tokens=action_tokens,
            conditioning_mode="hub_kv_lagged",
            detach_hub_kv=detach_hub_kv,
            **layout,
        )
        return torch.autograd.grad(
            output["video"][:, 1:].square().mean(),
            action_tokens,
            allow_unused=True,
        )[0]

    detached_grad = video_gradient(True)
    coupled_grad = video_gradient(False)
    assert detached_grad is None or torch.count_nonzero(detached_grad) == 0
    assert coupled_grad is not None
    assert torch.isfinite(coupled_grad).all()
    assert torch.count_nonzero(coupled_grad) > 0
    assert torch.count_nonzero(coupled_grad[:, : 2 * 3]) > 0


def test_detached_hub_value_still_depends_on_agent_tokens():
    torch.manual_seed(362)
    mot = _tiny_mot_pair().eval()
    layout = {
        "num_agents": 2,
        "horizon": 3,
        "num_hubs": 4,
        "first_frame_tokens": 1,
    }
    video_tokens = torch.randn(1, 3, 32)
    action_tokens = torch.randn(1, 10, 32)
    perturbed_action_tokens = action_tokens.clone()
    perturbed_action_tokens[:, : 2 * 3] += torch.randn(1, 2 * 3, 32)

    reference = _run_tiny_joint(
        mot,
        video_tokens=video_tokens,
        action_tokens=action_tokens,
        conditioning_mode="hub_kv_lagged",
        detach_hub_kv=True,
        **layout,
    )
    perturbed = _run_tiny_joint(
        mot,
        video_tokens=video_tokens,
        action_tokens=perturbed_action_tokens,
        conditioning_mode="hub_kv_lagged",
        detach_hub_kv=True,
        **layout,
    )

    assert torch.equal(
        reference["video"][:, :1], perturbed["video"][:, :1]
    )
    assert not torch.allclose(
        reference["video"][:, 1:], perturbed["video"][:, 1:]
    )


def test_hub_conditioned_video_is_invariant_to_agent_permutation():
    torch.manual_seed(363)
    mot = _tiny_mot_pair().eval()
    num_agents = 3
    horizon = 3
    num_hubs = 6
    first_frame_tokens = 1
    video_tokens = torch.randn(1, 3, 32)
    action_tokens = torch.randn(1, num_agents * horizon + num_hubs, 32)
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.argsort(permutation)
    permuted_agents = action_tokens[:, : num_agents * horizon].reshape(
        1, num_agents, horizon, 32
    )[:, permutation]
    permuted_action_tokens = torch.cat(
        [
            permuted_agents.reshape(1, num_agents * horizon, 32),
            action_tokens[:, num_agents * horizon :],
        ],
        dim=1,
    )

    reference = _run_tiny_joint(
        mot,
        video_tokens=video_tokens,
        action_tokens=action_tokens,
        num_agents=num_agents,
        horizon=horizon,
        num_hubs=num_hubs,
        first_frame_tokens=first_frame_tokens,
        conditioning_mode="hub_kv_lagged",
    )
    permuted = _run_tiny_joint(
        mot,
        video_tokens=video_tokens,
        action_tokens=permuted_action_tokens,
        num_agents=num_agents,
        horizon=horizon,
        num_hubs=num_hubs,
        first_frame_tokens=first_frame_tokens,
        conditioning_mode="hub_kv_lagged",
    )

    assert torch.allclose(reference["video"], permuted["video"], atol=2e-5, rtol=2e-5)
    reference_agents = reference["action"][:, : num_agents * horizon].reshape(
        1, num_agents, horizon, 32
    )
    permuted_agents = permuted["action"][:, : num_agents * horizon].reshape(
        1, num_agents, horizon, 32
    )
    assert torch.allclose(
        reference_agents, permuted_agents[:, inverse], atol=2e-5, rtol=2e-5
    )
    assert torch.allclose(
        reference["action"][:, num_agents * horizon :],
        permuted["action"][:, num_agents * horizon :],
        atol=2e-5,
        rtol=2e-5,
    )


def test_hub_conditioning_fails_closed_without_lag_or_hubs():
    mot = _tiny_mot_pair().eval()
    common = {
        "video_tokens": torch.randn(1, 3, 32),
        "action_tokens": torch.randn(1, 6, 32),
        "video_freqs": _unit_freqs(3),
        "action_freqs": _unit_freqs(6),
        "video_t_mod": torch.zeros(1, 6, 32),
        "action_t_mod": torch.zeros(1, 6, 32),
        "video_context_payload": None,
        "action_context_payload": None,
        "video_attention_mask": _first_frame_causal_mask(3, 1),
        "num_agents": 2,
        "horizon": 3,
        "num_hub_tokens": 0,
        "first_frame_tokens": 1,
        "video_conditioning_mode": "hub_kv_lagged",
    }
    with pytest.raises(ValueError, match="requires HubToken"):
        mot.forward_multi_agent_joint(**common)

    common["num_hub_tokens"] = 4
    common["action_tokens"] = torch.randn(1, 10, 32)
    common["action_freqs"] = _unit_freqs(10)
    common["video_conditioning_start_layer"] = 0
    with pytest.raises(ValueError, match="1 <= start_layer"):
        mot.forward_multi_agent_joint(**common)

    common["video_conditioning_start_layer"] = 1
    common["video_attention_mask"] = torch.ones(3, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="causally isolated"):
        mot.forward_multi_agent_joint(**common)


def _checkpoint_metadata_probe(conditioning_mode):
    probe = object.__new__(FastWAMMultiRobot)
    torch.nn.Module.__init__(probe)
    object.__setattr__(
        probe,
        "action_expert",
        SimpleNamespace(
            agent_encoding_mode="geometry",
            agent_geometry_dim=7,
            hub_enabled=True,
            hub_token_ratio=2.0,
        ),
    )
    object.__setattr__(probe, "video_conditioning_mode", conditioning_mode)
    object.__setattr__(probe, "video_conditioning_start_layer", 1)
    object.__setattr__(probe, "video_conditioning_detach_hub_kv", True)
    object.__setattr__(probe, "training_mode", "joint")
    object.__setattr__(probe, "_trainable_scope", "dit")
    object.__setattr__(probe, "torch_dtype", torch.float32)
    object.__setattr__(probe, "_loaded_base_checkpoint", "official-base.pt")
    probe.mot = torch.nn.Linear(3, 2)
    return probe


def test_v2_checkpoint_is_explicitly_acv0_and_v3_preserves_treatment():
    acv0 = _checkpoint_metadata_probe("none")
    acv1 = _checkpoint_metadata_probe("hub_kv_lagged")
    v2_metadata = {
        key: value
        for key, value in acv0._multi_robot_architecture_metadata().items()
        if not key.startswith("video_action_conditioning_")
    }
    v2_payload = {
        "format": "fastwam_multi_robot_v2",
        "multi_robot_architecture": v2_metadata,
    }
    acv0._validate_multi_robot_checkpoint_metadata(v2_payload, "legacy-v2.pt")
    with pytest.raises(ValueError, match="video_action_conditioning_mode"):
        acv1._validate_multi_robot_checkpoint_metadata(v2_payload, "legacy-v2.pt")

    v3_payload = {
        "format": "fastwam_multi_robot_v3",
        "multi_robot_architecture": acv1._multi_robot_architecture_metadata(),
    }
    acv1._validate_multi_robot_checkpoint_metadata(v3_payload, "acv1-v3.pt")


def test_v3_checkpoint_real_roundtrip_and_cross_arm_rejection(tmp_path):
    torch.manual_seed(365)
    source = _checkpoint_metadata_probe("hub_kv_lagged")
    checkpoint_path = tmp_path / "acv1-v3.pt"
    source.save_checkpoint(checkpoint_path, step=17)

    payload = torch.load(checkpoint_path, map_location="cpu")
    assert payload["format"] == "fastwam_multi_robot_v3"
    assert payload["step"] == 17
    assert payload["multi_robot_architecture"] == (
        source._multi_robot_architecture_metadata()
    )

    target = _checkpoint_metadata_probe("hub_kv_lagged")
    target.load_checkpoint(checkpoint_path)
    for key, value in source.mot.state_dict().items():
        assert torch.equal(value, target.mot.state_dict()[key])

    acv0 = _checkpoint_metadata_probe("none")
    with pytest.raises(ValueError, match="video_action_conditioning_mode"):
        acv0.load_checkpoint(checkpoint_path)


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
