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
import torch.nn.functional as F

from .action_dit import ActionDiT
from .wan_video_dit import sinusoidal_embedding_1d


class GaussianAgentAdapter(nn.Module):
    """Encode one Gaussian feature map per real agent without mixing agents.

    The adapter accepts canonical FP16 cache tensors with shape
    ``[B, N, 13, 28, 40]``.  Agents are folded into the batch axis only while
    applying a shared convolutional encoder, so the output remains a native
    variable-length set ``[B, N, hidden_dim]``.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        in_channels: int = 13,
        input_height: int = 28,
        input_width: int = 40,
        stem_dim: int = 64,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be positive, got {hidden_dim}")
        if in_channels <= 0:
            raise ValueError(f"`in_channels` must be positive, got {in_channels}")
        if input_height <= 0 or input_width <= 0:
            raise ValueError(
                "`input_height` and `input_width` must be positive, "
                f"got {(input_height, input_width)}"
            )
        if stem_dim <= 0 or stem_dim % 8:
            raise ValueError(f"`stem_dim` must be a positive multiple of 8, got {stem_dim}")

        self.hidden_dim = int(hidden_dim)
        self.in_channels = int(in_channels)
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.stem_dim = int(stem_dim)
        widths = (self.stem_dim, self.stem_dim * 2, self.stem_dim * 4)
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, widths[0], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, widths[0]),
            nn.SiLU(),
            nn.Conv2d(widths[0], widths[1], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, widths[1]),
            nn.SiLU(),
            nn.Conv2d(widths[1], widths[2], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, widths[2]),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(widths[2], self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

    def _canonical_input(self, agent_gaussian: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if agent_gaussian.ndim != 5:
            raise ValueError(
                "`agent_gaussian` must be [B,N,C,H,W], "
                f"got {tuple(agent_gaussian.shape)}"
            )
        batch_size, num_agents, channels, height, width = agent_gaussian.shape
        expected = (self.in_channels, self.input_height, self.input_width)
        if (channels, height, width) != expected:
            raise ValueError(
                "`agent_gaussian` channel/spatial shape mismatch: "
                f"expected {expected}, got {(channels, height, width)}"
            )
        if batch_size < 1 or num_agents < 1:
            raise ValueError(
                "`agent_gaussian` must contain at least one batch item and one real agent, "
                f"got {(batch_size, num_agents)}"
            )
        if not torch.is_floating_point(agent_gaussian):
            raise TypeError(
                "`agent_gaussian` must be floating point (the canonical cache is FP16), "
                f"got {agent_gaussian.dtype}"
            )

        # The cache is FP16, while model weights may be FP32/BF16.  Cast only at
        # the module boundary and never combine different agents spatially.
        reference = self.stem[0].weight
        gaussian = agent_gaussian.reshape(
            batch_size * num_agents,
            channels,
            height,
            width,
        ).to(device=reference.device, dtype=reference.dtype)
        return gaussian, batch_size, num_agents

    def forward(self, agent_gaussian: torch.Tensor) -> torch.Tensor:
        gaussian, batch_size, num_agents = self._canonical_input(agent_gaussian)
        embedding = self.projection(self.stem(gaussian))
        return embedding.reshape(batch_size, num_agents, self.hidden_dim)

    def forward_spatial(self, agent_gaussian: torch.Tensor) -> torch.Tensor:
        """Return the stride-8 Gaussian grid as coordinate-aware spatial tokens."""

        if self.hidden_dim % 4:
            raise ValueError(
                "Spatial Gaussian conditioning requires hidden_dim divisible by 4, "
                f"got {self.hidden_dim}"
            )
        gaussian, batch_size, num_agents = self._canonical_input(agent_gaussian)
        feature_map = self.stem[:-1](gaussian)
        _, _, height, width = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = self.projection[2](self.projection[1](tokens))

        rows = torch.arange(height, device=tokens.device, dtype=tokens.dtype)
        rows = rows.repeat_interleave(width)
        columns = torch.arange(width, device=tokens.device, dtype=tokens.dtype)
        columns = columns.repeat(height)
        position = torch.cat(
            [
                sinusoidal_embedding_1d(self.hidden_dim // 2, rows),
                sinusoidal_embedding_1d(self.hidden_dim // 2, columns),
            ],
            dim=-1,
        )
        tokens = tokens + position.unsqueeze(0)
        return tokens.reshape(batch_size, num_agents, height * width, self.hidden_dim)


def regular_simplex_vertices(
    num_vertices: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return unit vertices of a centered regular simplex.

    For ``N>=2`` the result has shape ``[N, N-1]``.  Every row has unit norm
    and every off-diagonal dot product is ``-1/(N-1)``.  For ``N=1`` a single
    neutral zero phase is returned.  A Helmert basis makes the construction
    deterministic and parameter-free.
    """

    if num_vertices < 1:
        raise ValueError(f"`num_vertices` must be >= 1, got {num_vertices}")
    if num_vertices == 1:
        return torch.zeros((1, 1), device=device, dtype=dtype)

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
    """Shared action expert for a native variable-length set of robots.

    Agent action tokens use the same encoder and transformer weights.  Current
    per-agent state (and optional physical geometry) is added to each action
    token.  Agent-order embeddings are disabled by default; a runtime Dynamic-N
    simplex is available as an ablation.  Hub queries are generated at runtime
    from one shared seed, so neither agents nor hubs require a fixed-capacity
    padded bank.
    """

    ACTION_BACKBONE_SKIP_PREFIXES = (
        "action_encoder.",
        "head.",
        "agent_state_encoder.",
        "agent_geometry_encoder.",
        "gaussian_adapter.",
        "gaussian_gate",
        "hub_seed",
    )

    def __init__(
        self,
        *,
        state_dim: int,
        agent_geometry_dim: int = 0,
        agent_encoding_mode: str = "none",
        hub_enabled: bool = True,
        hub_token_ratio: float = 0.5,
        hub_position_scale: float = 0.01,
        agent_rope_dim: int = 48,
        agent_phase_scale: float = 1.0,
        enable_gaussian: bool = False,
        gaussian_channels: int = 13,
        gaussian_height: int = 28,
        gaussian_width: int = 40,
        gaussian_stem_dim: int = 64,
        gaussian_conditioning_mode: str = "pooled_residual",
        gaussian_residual_floor: float = 0.0,
        gaussian_attention_temperature: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if state_dim <= 0:
            raise ValueError(f"`state_dim` must be > 0, got {state_dim}")
        agent_encoding_mode = str(agent_encoding_mode).strip().lower()
        if agent_encoding_mode not in {"none", "geometry", "dynamic_simplex"}:
            raise ValueError(
                "`agent_encoding_mode` must be one of none, geometry, dynamic_simplex; "
                f"got {agent_encoding_mode!r}"
            )
        if agent_geometry_dim < 0:
            raise ValueError(f"`agent_geometry_dim` must be >= 0, got {agent_geometry_dim}")
        if agent_encoding_mode == "geometry" and agent_geometry_dim <= 0:
            raise ValueError("agent_encoding_mode='geometry' requires agent_geometry_dim > 0")
        if hub_token_ratio < 0:
            raise ValueError(f"`hub_token_ratio` must be >= 0, got {hub_token_ratio}")
        if hub_enabled and hub_token_ratio <= 0:
            raise ValueError("hub_enabled=true requires hub_token_ratio > 0")
        if hub_position_scale < 0:
            raise ValueError(f"`hub_position_scale` must be >= 0, got {hub_position_scale}")
        if self.hidden_dim % 2:
            raise ValueError(
                f"`hidden_dim` must be even for sinusoidal hub positions, got {self.hidden_dim}"
            )
        if agent_rope_dim <= 0 or agent_rope_dim % 2 != 0:
            raise ValueError(f"`agent_rope_dim` must be a positive even integer, got {agent_rope_dim}")
        if agent_rope_dim > self.attn_head_dim:
            raise ValueError(
                f"`agent_rope_dim` ({agent_rope_dim}) exceeds attention head dim ({self.attn_head_dim})"
            )

        self.state_dim = int(state_dim)
        self.agent_geometry_dim = int(agent_geometry_dim)
        self.agent_encoding_mode = agent_encoding_mode
        self.hub_enabled = bool(hub_enabled)
        self.hub_token_ratio = float(hub_token_ratio)
        self.hub_position_scale = float(hub_position_scale)
        self.agent_rope_dim = int(agent_rope_dim)
        self.agent_phase_scale = float(agent_phase_scale)
        self.enable_gaussian = bool(enable_gaussian)
        self.gaussian_channels = int(gaussian_channels)
        self.gaussian_height = int(gaussian_height)
        self.gaussian_width = int(gaussian_width)
        self.gaussian_stem_dim = int(gaussian_stem_dim)
        self.gaussian_conditioning_mode = str(gaussian_conditioning_mode).strip().lower()
        self.gaussian_residual_floor = float(gaussian_residual_floor)
        self.gaussian_attention_temperature = float(gaussian_attention_temperature)
        if self.gaussian_conditioning_mode not in {
            "pooled_residual",
            "spatial_cross_attention",
        }:
            raise ValueError(
                "`gaussian_conditioning_mode` must be pooled_residual or "
                f"spatial_cross_attention, got {self.gaussian_conditioning_mode!r}"
            )
        if self.gaussian_residual_floor < 0:
            raise ValueError(
                "`gaussian_residual_floor` must be non-negative, got "
                f"{self.gaussian_residual_floor}"
            )
        if self.gaussian_attention_temperature <= 0:
            raise ValueError(
                "`gaussian_attention_temperature` must be positive, got "
                f"{self.gaussian_attention_temperature}"
            )
        if (
            self.gaussian_conditioning_mode == "pooled_residual"
            and self.gaussian_residual_floor != 0.0
        ):
            raise ValueError(
                "pooled_residual preserves the GAU1 baseline contract and requires "
                "gaussian_residual_floor=0"
            )
        if (
            self.gaussian_conditioning_mode == "spatial_cross_attention"
            and self.hidden_dim % 4
        ):
            raise ValueError(
                "spatial_cross_attention requires hidden_dim divisible by 4, "
                f"got {self.hidden_dim}"
            )

        self.agent_state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.agent_geometry_encoder = (
            nn.Sequential(
                nn.Linear(self.agent_geometry_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            if self.agent_encoding_mode == "geometry"
            else None
        )
        # A single shared seed keeps the parameterization independent of K(N).
        # Parameter-free sinusoidal positions distinguish runtime hub queries.
        self.hub_seed = nn.Parameter(
            torch.randn(1, self.hidden_dim) / math.sqrt(self.hidden_dim)
        )
        self.gaussian_adapter = (
            GaussianAgentAdapter(
                hidden_dim=self.hidden_dim,
                in_channels=self.gaussian_channels,
                input_height=self.gaussian_height,
                input_width=self.gaussian_width,
                stem_dim=self.gaussian_stem_dim,
            )
            if self.enable_gaussian
            else None
        )
        # A zero residual gate makes a newly enabled Gaussian branch exactly
        # equivalent to the pre-Gaussian baseline before its first update.
        self.gaussian_gate = (
            nn.Parameter(torch.zeros(1)) if self.enable_gaussian else None
        )

    def _validate_agent_inputs(
        self,
        action_tokens: torch.Tensor,
        agent_states: torch.Tensor,
        agent_geometry: Optional[torch.Tensor],
        agent_ids: Optional[torch.Tensor],
        agent_gaussian: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if action_tokens.ndim != 4:
            raise ValueError(
                "`action_tokens` must be [B,N,T,A], "
                f"got shape {tuple(action_tokens.shape)}"
            )
        batch_size, num_agents, _, action_dim = action_tokens.shape
        if num_agents < 1:
            raise ValueError("`action_tokens` must contain at least one real agent")
        if action_dim != self.action_dim:
            raise ValueError(f"Action dim must be {self.action_dim}, got {action_dim}")
        if agent_states.shape != (batch_size, num_agents, self.state_dim):
            raise ValueError(
                "`agent_states` must be [B,N,state_dim], "
                f"got {tuple(agent_states.shape)}; expected {(batch_size, num_agents, self.state_dim)}"
            )

        if self.agent_encoding_mode == "geometry":
            expected_geometry_shape = (
                batch_size,
                num_agents,
                self.agent_geometry_dim,
            )
            if agent_geometry is None:
                raise ValueError(
                    "agent_encoding_mode='geometry' requires `agent_geometry` "
                    f"with shape {expected_geometry_shape}"
                )
            if agent_geometry.shape != expected_geometry_shape:
                raise ValueError(
                    f"`agent_geometry` must be {expected_geometry_shape}, "
                    f"got {tuple(agent_geometry.shape)}"
                )
        elif agent_geometry is not None and agent_geometry.shape[:2] != (
            batch_size,
            num_agents,
        ):
            raise ValueError(
                "The first two `agent_geometry` dimensions must match [B,N], "
                f"got {tuple(agent_geometry.shape)}"
            )

        if agent_ids is not None and agent_ids.shape != (batch_size, num_agents):
            raise ValueError(
                f"`agent_ids` must have shape {(batch_size, num_agents)}, got {tuple(agent_ids.shape)}"
            )
        if self.agent_encoding_mode == "dynamic_simplex":
            if agent_ids is None:
                agent_ids = torch.arange(
                    num_agents, device=action_tokens.device
                ).expand(batch_size, -1)
            agent_ids = agent_ids.to(device=action_tokens.device, dtype=torch.long)
            expected_ids = torch.arange(
                num_agents, device=action_tokens.device
            ).expand(batch_size, -1)
            if not torch.equal(torch.sort(agent_ids, dim=1).values, expected_ids):
                raise ValueError(
                    "For dynamic_simplex, every sample's `agent_ids` must be a "
                    f"permutation of [0,{num_agents - 1}]"
                )
        elif agent_ids is not None:
            agent_ids = agent_ids.to(device=action_tokens.device, dtype=torch.long)

        if self.enable_gaussian:
            if agent_gaussian is None:
                raise ValueError(
                    "enable_gaussian=true requires `agent_gaussian` with shape "
                    f"{(batch_size, num_agents, self.gaussian_channels, self.gaussian_height, self.gaussian_width)}"
                )
            expected_gaussian_shape = (
                batch_size,
                num_agents,
                self.gaussian_channels,
                self.gaussian_height,
                self.gaussian_width,
            )
            if agent_gaussian.shape != expected_gaussian_shape:
                raise ValueError(
                    f"`agent_gaussian` must be {expected_gaussian_shape}, "
                    f"got {tuple(agent_gaussian.shape)}"
                )
            if not torch.is_floating_point(agent_gaussian):
                raise TypeError(
                    "`agent_gaussian` must be floating point (the canonical cache is FP16), "
                    f"got {agent_gaussian.dtype}"
                )
        else:
            # Gaussian-off is a true ablation: a loader may still provide the
            # field, but the model deliberately ignores it.
            agent_gaussian = None
        return agent_geometry, agent_ids, agent_gaussian

    def num_hub_tokens_for(self, num_agents: int) -> int:
        """Return runtime K(N) without allocating any fixed-capacity hub bank."""

        if num_agents < 1:
            raise ValueError(f"`num_agents` must be positive, got {num_agents}")
        if not self.hub_enabled:
            return 0
        return max(1, math.ceil(self.hub_token_ratio * num_agents))

    def _build_hub_tokens(
        self,
        *,
        batch_size: int,
        num_hub_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if num_hub_tokens < 1:
            return torch.empty(
                (batch_size, 0, self.hidden_dim), device=device, dtype=dtype
            )
        positions = (
            torch.arange(num_hub_tokens, device=device, dtype=torch.float32) + 0.5
        ) / float(num_hub_tokens)
        position_embedding = sinusoidal_embedding_1d(
            self.hidden_dim, positions
        ).to(dtype=dtype)
        hubs = self.hub_seed.to(device=device, dtype=dtype) + (
            self.hub_position_scale * position_embedding
        )
        return hubs.unsqueeze(0).expand(batch_size, -1, -1)

    def _build_agent_freqs(
        self,
        *,
        batch_size: int,
        num_agents: int,
        horizon: int,
        num_hub_tokens: int,
        agent_ids: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        total_action_tokens = num_agents * horizon
        if horizon > self.freqs.shape[0]:
            raise ValueError(f"Action horizon {horizon} exceeds RoPE cache {self.freqs.shape[0]}")

        # Each robot shares the same temporal positions.  Hubs use a neutral
        # temporal/agent phase because they summarize the whole action horizon.
        temporal = self.freqs[:horizon].repeat(num_agents, 1)
        if num_hub_tokens:
            hub_freqs = torch.ones(
                (num_hub_tokens, temporal.shape[1]),
                dtype=temporal.dtype,
                device=temporal.device,
            )
            temporal = torch.cat([temporal, hub_freqs], dim=0)
        freqs = temporal.to(device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()

        if self.agent_encoding_mode == "dynamic_simplex":
            if agent_ids is None:
                raise RuntimeError("Validated dynamic_simplex inputs are missing agent_ids")
            agent_complex_dim = self.agent_rope_dim // 2
            simplex = regular_simplex_vertices(
                num_agents, device=device, dtype=torch.float32
            )[agent_ids]
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
        agent_geometry: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.Tensor] = None,
        agent_gaussian: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        agent_geometry, agent_ids, agent_gaussian = self._validate_agent_inputs(
            action_tokens=action_tokens,
            agent_states=agent_states,
            agent_geometry=agent_geometry,
            agent_ids=agent_ids,
            agent_gaussian=agent_gaussian,
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
        if self.agent_encoding_mode == "geometry":
            if self.agent_geometry_encoder is None or agent_geometry is None:
                raise RuntimeError("Geometry mode was initialized without geometry inputs")
            geometry_emb = self.agent_geometry_encoder(
                agent_geometry.to(device=action_emb.device, dtype=action_emb.dtype)
            )
            action_emb = action_emb + geometry_emb.unsqueeze(2)
        if self.enable_gaussian:
            if (
                self.gaussian_adapter is None
                or self.gaussian_gate is None
                or agent_gaussian is None
            ):
                raise RuntimeError("Gaussian conditioning was enabled without initialized inputs/modules")
            gaussian_gate = self.gaussian_gate.to(
                device=action_emb.device,
                dtype=action_emb.dtype,
            )
            if self.gaussian_conditioning_mode == "pooled_residual":
                gaussian_emb = self.gaussian_adapter(agent_gaussian).to(
                    device=action_emb.device,
                    dtype=action_emb.dtype,
                )
                action_emb = action_emb + gaussian_gate * gaussian_emb.unsqueeze(2)
            else:
                spatial_tokens = self.gaussian_adapter.forward_spatial(agent_gaussian).to(
                    device=action_emb.device,
                    dtype=action_emb.dtype,
                )
                temporal_position = sinusoidal_embedding_1d(
                    self.hidden_dim,
                    torch.arange(horizon, device=action_emb.device, dtype=action_emb.dtype),
                )
                query = action_emb + temporal_position.view(1, 1, horizon, self.hidden_dim)
                scores = torch.einsum(
                    "bnth,bnph->bntp",
                    F.normalize(query.float(), dim=-1),
                    F.normalize(spatial_tokens.float(), dim=-1),
                )
                attention = torch.softmax(
                    scores / self.gaussian_attention_temperature,
                    dim=-1,
                )
                gaussian_emb = torch.einsum(
                    "bntp,bnph->bnth",
                    attention,
                    spatial_tokens.float(),
                ).to(dtype=action_emb.dtype)
                effective_gate = gaussian_gate + self.gaussian_residual_floor
                action_emb = action_emb + effective_gate * gaussian_emb
        action_emb = action_emb.reshape(batch_size, num_agents * horizon, self.hidden_dim)
        num_hub_tokens = self.num_hub_tokens_for(num_agents)
        if num_hub_tokens:
            hubs = self._build_hub_tokens(
                batch_size=batch_size,
                num_hub_tokens=num_hub_tokens,
                device=action_emb.device,
                dtype=action_emb.dtype,
            )
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
            num_hub_tokens=num_hub_tokens,
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
                "num_hub_tokens": num_hub_tokens,
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
        agent_geometry: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.Tensor] = None,
        agent_gaussian: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del (
            action_tokens,
            timestep,
            context,
            context_mask,
            agent_states,
            agent_geometry,
            agent_ids,
            agent_gaussian,
        )
        raise RuntimeError(
            "MultiAgentActionDiT cannot be called directly: an unstructured "
            "transformer forward would create direct cross-agent attention. "
            "Use FastWAMMultiRobot, which routes action tokens through MoT's "
            "factorized agent-local/Hub attention path."
        )
