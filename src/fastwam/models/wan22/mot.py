from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .wan_video_dit import flash_attention, modulate, rope_apply
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention. This will save memory but use more computation.")

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )
        
        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.to(device=q_cat.device)
            if attn_mask.ndim == 3:
                # SDPA expects a batch-specific mask to broadcast over the head
                # dimension as [B, 1, Q, K].  A raw [B, Q, K] tensor would align
                # B with the number of heads instead.
                attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.ndim != 2:
                raise ValueError(
                    "`attention_mask` must be [Q,K] or [B,Q,K], "
                    f"got shape {tuple(attention_mask.shape)}"
                )

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    def _factorized_multi_agent_attention(
        self,
        *,
        q_action: torch.Tensor,
        k_action: torch.Tensor,
        v_action: torch.Tensor,
        k_video: torch.Tensor,
        v_video: torch.Tensor,
        num_agents: int,
        horizon: int,
        num_hub_tokens: int,
        first_frame_tokens: int,
    ) -> torch.Tensor:
        """Apply the multi-agent Hub graph without a global masked QK product.

        For every transformer layer, each agent query attends only to the
        observed-video prefix, its own temporal tokens, and the current hub
        tokens. Hub queries attend to the observed-video prefix, all agents,
        and all hubs. Consequently, direct agent-to-agent edges are never
        materialized. Information gathered into a hub at layer ``l`` can be
        broadcast to another agent at layer ``l+1``, exactly matching the
        synchronous sparse-mask semantics while retaining the original block
        parameters and checkpoint format.

        The two SDPA calls have sizes ``[B*N,H,V0+H+K]`` and
        ``[B,K,V0+N*H+K]``. Neither call constructs or evaluates a dense
        ``[V+N*H+K, V+N*H+K]`` attention matrix.
        """

        if num_agents < 1 or horizon < 1 or num_hub_tokens < 0:
            raise ValueError(
                "Invalid multi-agent layout: "
                f"N={num_agents}, H={horizon}, K={num_hub_tokens}"
            )
        if q_action.ndim != 3 or k_action.ndim != 3 or v_action.ndim != 3:
            raise ValueError("Action Q/K/V must be 3D [B,S,D].")
        if q_action.shape != k_action.shape or q_action.shape != v_action.shape:
            raise ValueError(
                "Action Q/K/V shapes must match, got "
                f"{tuple(q_action.shape)}, {tuple(k_action.shape)}, {tuple(v_action.shape)}"
            )
        if k_video.ndim != 3 or v_video.ndim != 3 or k_video.shape != v_video.shape:
            raise ValueError(
                "Video K/V must have matching [B,S,D] shapes, got "
                f"{tuple(k_video.shape)} and {tuple(v_video.shape)}"
            )

        batch_size, action_seq_len, hidden_dim = q_action.shape
        num_action_tokens = num_agents * horizon
        expected_action_seq_len = num_action_tokens + num_hub_tokens
        if action_seq_len != expected_action_seq_len:
            raise ValueError(
                "Action sequence/layout mismatch: "
                f"S={action_seq_len}, expected N*H+K={expected_action_seq_len}"
            )
        if k_video.shape[0] != batch_size or k_video.shape[2] != hidden_dim:
            raise ValueError(
                "Video/action K/V batch or hidden dimension mismatch: "
                f"video={tuple(k_video.shape)}, action={tuple(q_action.shape)}"
            )
        if not 1 <= first_frame_tokens <= k_video.shape[1]:
            raise ValueError(
                "`first_frame_tokens` must be within the video K/V sequence, "
                f"got {first_frame_tokens} for length {k_video.shape[1]}"
            )

        video_k = k_video[:, :first_frame_tokens]
        video_v = v_video[:, :first_frame_tokens]
        q_agents = q_action[:, :num_action_tokens].reshape(
            batch_size, num_agents, horizon, hidden_dim
        )
        k_agents = k_action[:, :num_action_tokens].reshape(
            batch_size, num_agents, horizon, hidden_dim
        )
        v_agents = v_action[:, :num_action_tokens].reshape(
            batch_size, num_agents, horizon, hidden_dim
        )

        # Agent-local attention plus broadcast from the previous-layer hubs.
        # Expanding video/hub K/V over N changes only the batch view; the QK
        # products are computed solely for the permitted sparse blocks.
        local_k_parts = [
            video_k.unsqueeze(1).expand(-1, num_agents, -1, -1),
            k_agents,
        ]
        local_v_parts = [
            video_v.unsqueeze(1).expand(-1, num_agents, -1, -1),
            v_agents,
        ]
        if num_hub_tokens:
            k_hubs = k_action[:, num_action_tokens:]
            v_hubs = v_action[:, num_action_tokens:]
            local_k_parts.append(k_hubs.unsqueeze(1).expand(-1, num_agents, -1, -1))
            local_v_parts.append(v_hubs.unsqueeze(1).expand(-1, num_agents, -1, -1))

        local_k = torch.cat(local_k_parts, dim=2).reshape(
            batch_size * num_agents, -1, hidden_dim
        )
        local_v = torch.cat(local_v_parts, dim=2).reshape(
            batch_size * num_agents, -1, hidden_dim
        )
        local_q = q_agents.reshape(batch_size * num_agents, horizon, hidden_dim)
        local_out = self._mixed_attention(
            q_cat=local_q,
            k_cat=local_k,
            v_cat=local_v,
            attention_mask=None,
        ).reshape(batch_size, num_action_tokens, hidden_dim)

        if not num_hub_tokens:
            return local_out

        # Hub gather + hub self. All permitted keys are concatenated directly;
        # there is no masked dense key space and no direct agent-agent edge.
        hub_out = self._mixed_attention(
            q_cat=q_action[:, num_action_tokens:],
            k_cat=torch.cat([video_k, k_action], dim=1),
            v_cat=torch.cat([video_v, v_action], dim=1),
            attention_mask=None,
        )
        return torch.cat([local_out, hub_out], dim=1)

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def forward_video_with_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Run the video branch and return its output plus per-layer K/V.

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE frequencies, shape [Sv, 1, rope_dim].
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Video self-attention mask, shape [Sv, Sv].

        Returns:
            Final video tokens after all MoT layers.
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: video key tensor [B, Sv, H*Dh]
                - `v`: video value tensor [B, Sv, H*Dh]
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `forward_video_with_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            # Video prefill uses only video self-attention mask.
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            # Update video tokens for the next layer and persist current layer K/V.
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": k, "v": v})
        return x, kv_cache

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Prefill video K/V while preserving the established cache-only API."""

        _, kv_cache = self.forward_video_with_cache(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context_payload=video_context_payload,
            video_attention_mask=video_attention_mask,
        )
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim not in (2, 3):
            raise ValueError(
                f"`attention_mask` must be [S,S] or [B,S,S], got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[-2] != attention_mask.shape[-1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")
        if attention_mask.ndim == 3 and attention_mask.shape[0] != action_tokens.shape[0]:
            raise ValueError(
                "`attention_mask` batch mismatch: "
                f"mask={attention_mask.shape[0]} vs tokens={action_tokens.shape[0]}"
            )

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[-1] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[-1]} vs expected_total={total_seq_len}"
            )
        # Use the action query rows from the joint [video+action] mask.
        if attention_mask.ndim == 2:
            action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]
        else:
            action_attention_mask = attention_mask[:, video_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )

            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            # Mixed attention: action queries attend to cached video K/V plus current action K/V.
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x

    def forward_multi_agent_action_with_video_cache(
        self,
        *,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        num_agents: int,
        horizon: int,
        num_hub_tokens: int,
        first_frame_tokens: int,
    ) -> torch.Tensor:
        """Run cached-video action denoising through factorized Hub attention."""

        if "action" not in self.mixtures:
            raise ValueError(
                "MoT requires `action` expert for factorized multi-agent attention."
            )
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, "
                f"got {len(video_kv_cache)}."
            )

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )
            factorized = self._factorized_multi_agent_attention(
                q_action=q_action,
                k_action=k_action,
                v_action=v_action,
                k_video=layer_cache["k"],
                v_video=layer_cache["v"],
                num_agents=num_agents,
                horizon=horizon,
                num_hub_tokens=num_hub_tokens,
                first_frame_tokens=first_frame_tokens,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=factorized,
                context_payload=action_context_payload,
            )
        return x

    def forward_multi_agent_joint(
        self,
        *,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        action_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
        num_agents: int,
        horizon: int,
        num_hub_tokens: int,
        first_frame_tokens: int,
    ) -> Dict[str, torch.Tensor]:
        """Joint VideoGen/action forward without global video-action attention."""

        if video_attention_mask.ndim != 2:
            raise ValueError(
                "`video_attention_mask` must be a 2D video-only mask, got "
                f"{tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape != (
            video_tokens.shape[1],
            video_tokens.shape[1],
        ):
            raise ValueError(
                "Video mask/token mismatch: "
                f"mask={tuple(video_attention_mask.shape)}, "
                f"tokens={tuple(video_tokens.shape)}"
            )

        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        x_video = video_tokens
        x_action = action_tokens
        for layer_idx in range(self.num_layers):
            video_block = video_expert.blocks[layer_idx]
            action_block = action_expert.blocks[layer_idx]
            (
                q_video,
                k_video,
                v_video,
                residual_video,
                gate_msa_video,
                shift_mlp_video,
                scale_mlp_video,
                gate_mlp_video,
                checkpoint_video,
            ) = self._build_expert_attention_io(
                expert=video_expert,
                block=video_block,
                x=x_video,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            (
                q_action,
                k_action,
                v_action,
                residual_action,
                gate_msa_action,
                shift_mlp_action,
                scale_mlp_action,
                gate_mlp_action,
                checkpoint_action,
            ) = self._build_expert_attention_io(
                expert=action_expert,
                block=action_block,
                x=x_action,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )

            video_attention = self._mixed_attention(
                q_cat=q_video,
                k_cat=k_video,
                v_cat=v_video,
                attention_mask=video_attention_mask,
            )
            action_attention = self._factorized_multi_agent_attention(
                q_action=q_action,
                k_action=k_action,
                v_action=v_action,
                k_video=k_video,
                v_video=v_video,
                num_agents=num_agents,
                horizon=horizon,
                num_hub_tokens=num_hub_tokens,
                first_frame_tokens=first_frame_tokens,
            )

            x_video = self._apply_post_with_optional_checkpoint(
                block=video_block,
                residual_x=residual_video,
                gate_msa=gate_msa_video,
                shift_mlp=shift_mlp_video,
                scale_mlp=scale_mlp_video,
                gate_mlp=gate_mlp_video,
                use_gradient_checkpointing=checkpoint_video,
                mixed_slice=video_attention,
                context_payload=video_context_payload,
            )
            x_action = self._apply_post_with_optional_checkpoint(
                block=action_block,
                residual_x=residual_action,
                gate_msa=gate_msa_action,
                shift_mlp=shift_mlp_action,
                scale_mlp=scale_mlp_action,
                gate_mlp=gate_mlp_action,
                use_gradient_checkpointing=checkpoint_action,
                mixed_slice=action_attention,
                context_payload=action_context_payload,
            )

        return {"video": x_video, "action": x_action}

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask.ndim not in (2, 3):
            raise ValueError(
                f"`attention_mask` must be [S,S] or [B,S,S], got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[-2] != attention_mask.shape[-1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        batch_size = next(iter(embeds_all.values())).shape[0]
        if attention_mask.ndim == 3 and attention_mask.shape[0] != batch_size:
            raise ValueError(
                "`attention_mask` batch mismatch: "
                f"mask={attention_mask.shape[0]} vs tokens={batch_size}"
            )

        tokens_all = {k: v for k, v in embeds_all.items()}

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            # 3. concat all tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if attention_mask.shape[-1] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[-1]} vs tokens={total_seq}"
                )

            mixed = self._mixed_attention(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask)

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        return tokens_all
