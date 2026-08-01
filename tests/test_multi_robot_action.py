import torch

from fastwam.models.wan22.fastwam_multi_robot import FastWAMMultiRobot
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.multi_agent_action_dit import (
    MultiAgentActionDiT,
    regular_simplex_vertices,
)


def _tiny_action_expert(num_hubs=2):
    return MultiAgentActionDiT(
        action_dim=3,
        state_dim=4,
        max_agents=4,
        num_hub_tokens=num_hubs,
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


def test_regular_simplex_is_symmetric():
    vertices = regular_simplex_vertices(4)
    gram = vertices @ vertices.T
    assert torch.allclose(torch.diag(gram), torch.ones(4), atol=1e-6)
    expected = torch.full((4, 4), -1.0 / 3.0)
    expected.fill_diagonal_(1.0)
    assert torch.allclose(gram, expected, atol=1e-6)


def test_action_expert_permutation_equivariance():
    torch.manual_seed(7)
    model = _tiny_action_expert().eval()
    action = torch.randn(1, 3, 5, 3)
    state = torch.randn(1, 3, 4)
    mask = torch.tensor([[True, True, True]])
    ids = torch.tensor([[0, 1, 2]])
    timestep = torch.tensor([500.0])
    context = torch.randn(1, 6, 16)
    context_mask = torch.ones(1, 6, dtype=torch.bool)

    output = model(
        action,
        timestep,
        context,
        context_mask,
        agent_states=state,
        agent_mask=mask,
        agent_ids=ids,
    )
    permutation = torch.tensor([2, 0, 1])
    permuted = model(
        action[:, permutation],
        timestep,
        context,
        context_mask,
        agent_states=state[:, permutation],
        agent_mask=mask[:, permutation],
        agent_ids=ids[:, permutation],
    )
    inverse = torch.argsort(permutation)
    assert torch.allclose(output, permuted[:, inverse], atol=2e-5, rtol=2e-5)


class _VideoMaskStub:
    @staticmethod
    def build_video_to_video_mask(video_seq_len, video_tokens_per_frame, device):
        del video_tokens_per_frame
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)


def test_sparse_hub_mask_blocks_direct_cross_agent_paths():
    model = FastWAMMultiRobot.__new__(FastWAMMultiRobot)
    torch.nn.Module.__init__(model)
    model.video_expert = _VideoMaskStub()
    action_pre = {
        "meta": {
            "agent_mask": torch.tensor([[True, True, True, False]]),
            "horizon": 2,
            "num_hub_tokens": 2,
        }
    }
    mask = model._build_multi_robot_attention_mask(
        video_seq_len=3,
        video_tokens_per_frame=3,
        action_pre=action_pre,
        device=torch.device("cpu"),
    )[0]
    video_len = 3
    agent0 = slice(video_len, video_len + 2)
    agent1 = slice(video_len + 2, video_len + 4)
    invalid_agent = slice(video_len + 6, video_len + 8)
    hubs = slice(video_len + 8, video_len + 10)

    assert not mask[agent0, agent1].any()
    assert mask[agent0, agent0].all()
    assert mask[agent0, hubs].all()
    assert mask[hubs, agent0].all()
    assert mask[hubs, agent1].all()
    assert not mask[hubs, invalid_agent].any()
    assert not mask[:video_len, video_len:].any()


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
