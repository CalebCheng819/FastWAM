"""Multi-agent action expert with sparse HubToken communication.

The collaboration pattern is inspired by Gamma-World's Sparse Hub Attention
and Simplex Rotary Agent Encoding (Apache-2.0):
https://github.com/nv-tlabs/Gamma-World

This is a FastWAM-native adaptation, not a copy of Gamma-World's Cosmos
implementation.  FastWAM keeps a single shared video expert and represents
robot actions explicitly as [batch, agent, time, action_dim].
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .action_dit import ActionDiT
from .wan_video_dit import sinusoidal_embedding_1d


def regular_simplex_vertices(
    num_vertices: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return unit vertices of a centered regular simplex.

    The result has shape ``[num_vertices, num_vertices - 1]``.  Every row has
    unit norm and every off-diagonal dot product is ``-1/(N-1)``.  A Helmert
    basis is used so the construction is deterministic and parameter-free.
    """

    if num_vertices < 2:
        raise ValueError(f"`num_vertices` must be >= 2, got {num_vertices}")

    basis = torch.zeros(
        (num_vertices, num_vertices - 1),
        device=device,
        dtype=torch.float64,
    )
    for column in range(num_vertices - 1):
        denom = math.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / denom
        basis[column + 1, column] = -(column + 1) / denom

    basis = basis * math.sqrt(num_vertices / (num_vertices - 1))
    return basis.to(dtype=dtype)


class MultiAgentActionDiT(ActionDiT):
    """Shared action expert for a padded set of cooperating robots.

    Agent action tokens use the same encoder and transformer weights.  Current
    per-agent state is added to each action token, and a parameter-free simplex
    phase rotates a reserved part of RoPE.  Learnable hub tokens are appended
    to the action sequence; the owning FastWAM model supplies the sparse mask
    that makes hubs the only cross-agent communication route.
    """

    ACTION_BACKBONE_SKIP_PREFIXES = (
        "action_encoder.",
        "head.",
        "agent_state_encoder.",
        "hub_tokens",
    )

    def __init__(
        self,
        *,
        state_dim: int,
        max_agents: int = 4,
        num_hub_tokens: int = 8,
        agent_rope_dim: int = 48,
        agent_phase_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if state_dim <= 0:
            raise ValueError(f"`state_dim` must be > 0, got {state_dim}")
        if max_agents < 2:
            raise ValueError(f"`max_agents` must be >= 2, got {max_agents}")
        if num_hub_tokens < 0:
            raise ValueError(f"`num_hub_tokens` must be >= 0, got {num_hub_tokens}")
        if agent_rope_dim <= 0 or agent_rope_dim % 2 != 0:
            raise ValueError(f"`agent_rope_dim` must be a positive even integer, got {agent_rope_dim}")
        if agent_rope_dim > self.attn_head_dim:
            raise ValueError(
                f"`agent_rope_dim` ({agent_rope_dim}) exceeds attention head dim ({self.attn_head_dim})"
            )

        self.state_dim = int(state_dim)
        self.max_agents = int(max_agents)
        self.num_hub_tokens = int(num_hub_tokens)
        self.agent_rope_dim = int(agent_rope_dim)
        self.agent_phase_scale = float(agent_phase_scale)

        self.agent_state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.hub_tokens = nn.Parameter(
            torch.randn(self.num_hub_tokens, self.hidden_dim) / math.sqrt(self.hidden_dim)
        )
        self.register_buffer(
            "simplex_vertices",
            regular_simplex_vertices(self.max_agents),
            persistent=False,
        )

    def _validate_agent_inputs(
        self,
        action_tokens: torch.Tensor,
        agent_states: torch.Tensor,
        agent_mask: Optional[torch.Tensor],
        agent_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action_tokens.ndim != 4:
            raise ValueError(
                "`action_tokens` must be [B,N,T,A], "
                f"got shape {tuple(action_tokens.shape)}"
            )
        batch_size, num_agents, _, action_dim = action_tokens.shape
        if action_dim != self.action_dim:
            raise ValueError(f"Action dim must be {self.action_dim}, got {action_dim}")
        if num_agents > self.max_agents:
            raise ValueError(f"Number of agents {num_agents} exceeds max_agents={self.max_agents}")
        if agent_states.shape != (batch_size, num_agents, self.state_dim):
            raise ValueError(
                "`agent_states` must be [B,N,state_dim], "
                f"got {tuple(agent_states.shape)}; expected {(batch_size, num_agents, self.state_dim)}"
            )

        if agent_mask is None:
            agent_mask = torch.ones(
                (batch_size, num_agents),
                dtype=torch.bool,
                device=action_tokens.device,
            )
        elif agent_mask.shape != (batch_size, num_agents):
            raise ValueError(
                f"`agent_mask` must have shape {(batch_size, num_agents)}, got {tuple(agent_mask.shape)}"
            )
        agent_mask = agent_mask.to(device=action_tokens.device, dtype=torch.bool)
        if not bool(agent_mask.any(dim=1).all().item()):
            raise ValueError("Every sample must contain at least one valid agent.")

        if agent_ids is None:
            agent_ids = torch.arange(num_agents, device=action_tokens.device).expand(batch_size, -1)
        elif agent_ids.shape != (batch_size, num_agents):
            raise ValueError(
                f"`agent_ids` must have shape {(batch_size, num_agents)}, got {tuple(agent_ids.shape)}"
            )
        agent_ids = agent_ids.to(device=action_tokens.device, dtype=torch.long)
        valid_ids = agent_ids[agent_mask]
        if valid_ids.numel() and (valid_ids.min() < 0 or valid_ids.max() >= self.max_agents):
            raise ValueError(
                f"Valid `agent_ids` must be in [0,{self.max_agents - 1}], "
                f"got min={int(valid_ids.min())}, max={int(valid_ids.max())}"
            )
        return agent_mask, agent_ids

    def _build_agent_freqs(
        self,
        *,
        batch_size: int,
        num_agents: int,
        horizon: int,
        agent_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        total_action_tokens = num_agents * horizon
        total_tokens = total_action_tokens + self.num_hub_tokens
        if horizon > self.freqs.shape[0]:
            raise ValueError(f"Action horizon {horizon} exceeds RoPE cache {self.freqs.shape[0]}")

        # Each robot shares the same temporal positions.  Hubs use a neutral
        # temporal/agent phase because they summarize the whole action horizon.
        temporal = self.freqs[:horizon].repeat(num_agents, 1)
        if self.num_hub_tokens:
            hub_freqs = torch.ones(
                (self.num_hub_tokens, temporal.shape[1]),
                dtype=temporal.dtype,
                device=temporal.device,
            )
            temporal = torch.cat([temporal, hub_freqs], dim=0)
        freqs = temporal.to(device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()

        agent_complex_dim = self.agent_rope_dim // 2
        simplex = self.simplex_vertices.to(device=device, dtype=torch.float32)[agent_ids]
        repeat_count = math.ceil(agent_complex_dim / simplex.shape[-1])
        phases = simplex.repeat(1, 1, repeat_count)[..., :agent_complex_dim]
        phases = torch.polar(
            torch.ones_like(phases),
            phases * self.agent_phase_scale,
        )
        phases = phases.unsqueeze(2).expand(-1, -1, horizon, -1).reshape(
            batch_size, total_action_tokens, agent_complex_dim
        )
        freqs[:, :total_action_tokens, -agent_complex_dim:] *= phases.to(freqs.dtype)
        return freqs.unsqueeze(2)  # [B, S, 1, Dh/2]

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        *,
        agent_states: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        agent_mask, agent_ids = self._validate_agent_inputs(
            action_tokens=action_tokens,
            agent_states=agent_states,
            agent_mask=agent_mask,
            agent_ids=agent_ids,
        )
        batch_size, num_agents, horizon, _ = action_tokens.shape
        if timestep.ndim != 1 or timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` must be [1] or [B], got shape {tuple(timestep.shape)} for B={batch_size}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch size.")
            timestep = timestep.expand(batch_size)
        if context.ndim != 3 or context.shape[0] != batch_size:
            raise ValueError(
                f"`context` must be [B,L,D] with B={batch_size}, got {tuple(context.shape)}"
            )
        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]),
                dtype=torch.bool,
                device=context.device,
            )
        elif context_mask.shape != (batch_size, context.shape[1]):
            raise ValueError(
                "`context_mask` must match context [B,L], "
                f"got {tuple(context_mask.shape)} vs {tuple(context.shape[:2])}"
            )

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        action_emb = self.action_encoder(action_tokens)
        state_emb = self.agent_state_encoder(
            agent_states.to(device=action_emb.device, dtype=action_emb.dtype)
        )
        action_emb = action_emb + state_emb.unsqueeze(2)
        action_emb = action_emb.reshape(batch_size, num_agents * horizon, self.hidden_dim)
        if self.num_hub_tokens:
            hubs = self.hub_tokens.to(dtype=action_emb.dtype).unsqueeze(0).expand(batch_size, -1, -1)
            tokens = torch.cat([action_emb, hubs], dim=1)
        else:
            tokens = action_emb

        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.to(dtype=torch.bool).unsqueeze(1).expand(
            -1, tokens.shape[1], -1
        )
        freqs = self._build_agent_freqs(
            batch_size=batch_size,
            num_agents=num_agents,
            horizon=horizon,
            agent_ids=agent_ids,
            device=tokens.device,
        )

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "num_agents": num_agents,
                "horizon": horizon,
                "num_action_tokens": num_agents * horizon,
                "num_hub_tokens": self.num_hub_tokens,
                "agent_mask": agent_mask,
                "agent_ids": agent_ids,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        meta = pre_state["meta"]
        num_action_tokens = int(meta["num_action_tokens"])
        action_tokens = tokens[:, :num_action_tokens]
        action = self.head(action_tokens)
        return action.reshape(
            int(meta["batch_size"]),
            int(meta["num_agents"]),
            int(meta["horizon"]),
            self.action_dim,
        )

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        *,
        agent_states: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            agent_states=agent_states,
            agent_mask=agent_mask,
            agent_ids=agent_ids,
        )
        x = pre_state["tokens"]
        for block in self.blocks:
            x = block(
                x,
                pre_state["context"],
                pre_state["t_mod"],
                pre_state["freqs"],
                context_mask=pre_state["context_mask"],
            )
        return self.post_dit(x, pre_state)
