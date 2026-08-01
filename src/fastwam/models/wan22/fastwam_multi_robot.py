"""FastWAM adaptation for synchronized multi-robot collaboration."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .multi_agent_action_dit import MultiAgentActionDiT

logger = get_logger(__name__)


class FastWAMMultiRobot(FastWAM):
    """One shared world stream plus an explicit multi-agent action stream.

    ``training_mode='action_only_cache'`` freezes the world backbone in the
    usual staged-training setup: only the observed global frame is encoded,
    its K/V tensors are cached without gradients, and the multi-agent action
    expert is optimized.  ``training_mode='joint'`` retains FastWAM's joint
    video/action flow-matching objective.
    """

    def __init__(self, *args, training_mode: str = "action_only_cache", **kwargs):
        super().__init__(*args, **kwargs)
        if training_mode not in {"action_only_cache", "joint"}:
            raise ValueError(
                f"Unsupported training_mode={training_mode!r}; expected 'action_only_cache' or 'joint'."
            )
        self.training_mode = training_mode
        self._trainable_scope = "dit"
        self._loaded_base_checkpoint: Optional[str] = None

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        training_mode: str = "action_only_cache",
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 0.0,
        loss_lambda_action: float = 1.0,
    ) -> "FastWAMMultiRobot":
        if video_dit_config is None or "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config` with `text_dim` is required.")
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        video_expert = components.dit
        action_expert = MultiAgentActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("Action and video experts must use the same number of attention heads.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("Action and video experts must use the same attention head dimension.")
        if len(action_expert.blocks) != len(video_expert.blocks):
            raise ValueError("Action and video experts must use the same number of layers.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=None,
            device=device,
            torch_dtype=torch_dtype,
            training_mode=training_mode,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def configure_trainable_parameters(self, scope: str) -> list[torch.nn.Parameter]:
        """Freeze the model and enable a documented staged-training scope."""

        scope = str(scope).strip().lower()
        if scope not in {"hub_io", "action", "dit"}:
            raise ValueError(f"Unsupported trainable_scope={scope!r}; expected hub_io, action, or dit.")

        self.eval()
        self.requires_grad_(False)
        if scope == "hub_io":
            self.mot.train()
            self.video_expert.eval()
            self.action_expert.action_encoder.requires_grad_(True)
            self.action_expert.agent_state_encoder.requires_grad_(True)
            self.action_expert.head.requires_grad_(True)
            self.action_expert.hub_tokens.requires_grad_(True)
            self.action_expert.train()
        elif scope == "action":
            self.mot.train()
            self.video_expert.eval()
            self.action_expert.requires_grad_(True)
            self.action_expert.train()
        else:
            self.mot.requires_grad_(True)
            self.mot.train()

        self._trainable_scope = scope
        params = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not params:
            raise RuntimeError(f"trainable_scope={scope!r} selected no parameters")
        return params

    def build_inputs(self, sample, tiled: bool = False):
        required = {"video", "action", "agent_state", "agent_mask", "context", "context_mask"}
        missing = sorted(required - set(sample))
        if missing:
            raise ValueError(f"Missing multi-robot sample fields: {missing}")

        video = sample["video"]
        action = sample["action"]
        agent_state = sample["agent_state"]
        agent_mask = sample["agent_mask"]
        context = sample["context"]
        context_mask = sample["context_mask"]

        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"`video` must be [B,3,T,H,W], got {tuple(video.shape)}")
        batch_size, _, num_frames, height, width = video.shape
        if height % 16 or width % 16:
            raise ValueError(f"Video H/W must be multiples of 16, got {(height, width)}")
        if num_frames <= 1 or num_frames % 4 != 1:
            raise ValueError(f"Video T must be >1 and satisfy T % 4 == 1, got {num_frames}")
        if action.ndim != 4:
            raise ValueError(f"`action` must be [B,N,H,A], got {tuple(action.shape)}")
        if action.shape[0] != batch_size or action.shape[-1] != self.action_expert.action_dim:
            raise ValueError(
                f"Action shape mismatch, got {tuple(action.shape)} for B={batch_size}, "
                f"A={self.action_expert.action_dim}"
            )
        num_agents, horizon = int(action.shape[1]), int(action.shape[2])
        if horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"Action horizon {horizon} must be divisible by video transitions {num_frames - 1}."
            )
        if agent_state.shape != (batch_size, num_agents, self.action_expert.state_dim):
            raise ValueError(
                f"`agent_state` must be {(batch_size, num_agents, self.action_expert.state_dim)}, "
                f"got {tuple(agent_state.shape)}"
            )
        if agent_mask.shape != (batch_size, num_agents):
            raise ValueError(f"`agent_mask` must be {(batch_size, num_agents)}, got {tuple(agent_mask.shape)}")

        agent_ids = sample.get("agent_ids")
        if agent_ids is None:
            agent_ids = torch.arange(num_agents).expand(batch_size, -1)
        if agent_ids.shape != (batch_size, num_agents):
            raise ValueError(f"`agent_ids` must be {(batch_size, num_agents)}, got {tuple(agent_ids.shape)}")

        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None and action_is_pad.shape != (batch_size, num_agents, horizon):
            raise ValueError(
                f"`action_is_pad` must be {(batch_size, num_agents, horizon)}, "
                f"got {tuple(action_is_pad.shape)}"
            )
        image_is_pad = sample.get("image_is_pad")
        if image_is_pad is not None and image_is_pad.shape != (batch_size, num_frames):
            raise ValueError(
                f"`image_is_pad` must be {(batch_size, num_frames)}, got {tuple(image_is_pad.shape)}"
            )
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)}, "
                f"{tuple(context_mask.shape)}"
            )

        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if self.training_mode == "action_only_cache":
            input_video = input_video[:, :, :1]
        input_latents = self._encode_video_latents(input_video, tiled=tiled)
        first_frame_latents = input_latents[:, :, :1]

        return {
            "context": context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            "context_mask": context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": bool(
                getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
            "action": action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            "agent_state": agent_state.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            ),
            "agent_mask": agent_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
            "agent_ids": agent_ids.to(device=self.device, dtype=torch.long, non_blocking=True),
            "action_is_pad": (
                None
                if action_is_pad is None
                else action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
            ),
            "image_is_pad": (
                None
                if image_is_pad is None
                else image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
            ),
        }

    def _build_multi_robot_attention_mask(
        self,
        *,
        video_seq_len: int,
        video_tokens_per_frame: int,
        action_pre: dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        meta = action_pre["meta"]
        agent_mask = meta["agent_mask"].to(device=device, dtype=torch.bool)
        batch_size, num_agents = agent_mask.shape
        horizon = int(meta["horizon"])
        num_hubs = int(meta["num_hub_tokens"])
        action_seq_len = num_agents * horizon + num_hubs
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros(
            (batch_size, total_seq_len, total_seq_len),
            dtype=torch.bool,
            device=device,
        )

        video_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[:, :video_seq_len, :video_seq_len] = video_mask
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        hub_start = video_seq_len + num_agents * horizon

        for agent_idx in range(num_agents):
            start = video_seq_len + agent_idx * horizon
            end = start + horizon
            # Even padded query rows retain a non-empty local attention set,
            # preventing all-masked SDPA rows.  They are excluded from loss and
            # are never visible to hubs.
            mask[:, start:end, start:end] = True
            valid = agent_mask[:, agent_idx]
            mask[valid, start:end, :first_frame_tokens] = True
            if num_hubs:
                mask[valid, start:end, hub_start:] = True

        if num_hubs:
            mask[:, hub_start:, :first_frame_tokens] = True
            mask[:, hub_start:, hub_start:] = True
            for agent_idx in range(num_agents):
                start = video_seq_len + agent_idx * horizon
                end = start + horizon
                valid = agent_mask[:, agent_idx]
                mask[valid, hub_start:, start:end] = True
        return mask

    def _multi_action_loss(
        self,
        *,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        timestep_action: torch.Tensor,
        agent_mask: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        token_loss = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=-1)
        valid = agent_mask.unsqueeze(-1).expand_as(token_loss)
        if action_is_pad is not None:
            valid = valid & (~action_is_pad)
        valid_f = valid.to(dtype=token_loss.dtype)
        per_sample = (token_loss * valid_f).sum(dim=(1, 2)) / valid_f.sum(dim=(1, 2)).clamp(min=1.0)
        weight = self.train_action_scheduler.training_weight(timestep_action).to(
            device=per_sample.device, dtype=per_sample.dtype
        )
        return (per_sample * weight).mean()

    def _prepare_noisy_action(self, inputs: dict[str, Any]):
        action = inputs["action"]
        batch_size = action.shape[0]
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )
        return noisy_action, target_action, timestep_action

    def _action_pre(
        self,
        noisy_action: torch.Tensor,
        timestep_action: torch.Tensor,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        return self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            agent_states=inputs["agent_state"],
            agent_mask=inputs["agent_mask"],
            agent_ids=inputs["agent_ids"],
        )

    def _training_loss_action_only(self, inputs: dict[str, Any]):
        noisy_action, target_action, timestep_action = self._prepare_noisy_action(inputs)
        action_pre = self._action_pre(noisy_action, timestep_action, inputs)

        timestep_video = torch.zeros(
            (inputs["first_frame_latents"].shape[0],),
            dtype=inputs["first_frame_latents"].dtype,
            device=self.device,
        )
        with torch.no_grad():
            video_pre = self.video_expert.pre_dit(
                x=inputs["first_frame_latents"],
                timestep=timestep_video,
                context=inputs["context"],
                context_mask=inputs["context_mask"],
                action=None,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            )
            attention_mask = self._build_multi_robot_attention_mask(
                video_seq_len=video_pre["tokens"].shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                action_pre=action_pre,
                device=video_pre["tokens"].device,
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=attention_mask[0, :video_seq_len, :video_seq_len],
            )

        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        pred_action = self.action_expert.post_dit(action_tokens, action_pre)
        loss_action = self._multi_action_loss(
            pred_action=pred_action,
            target_action=target_action,
            timestep_action=timestep_action,
            agent_mask=inputs["agent_mask"],
            action_is_pad=inputs["action_is_pad"],
        )
        loss_total = self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    def _training_loss_joint(self, inputs: dict[str, Any]):
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        noisy_video = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )
        noisy_video[:, :, :1] = inputs["first_frame_latents"]

        noisy_action, target_action, timestep_action = self._prepare_noisy_action(inputs)
        video_pre = self.video_expert.pre_dit(
            x=noisy_video,
            timestep=timestep_video,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self._action_pre(noisy_action, timestep_action, inputs)
        attention_mask = self._build_multi_robot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            action_pre=action_pre,
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)[:, :, 1:]
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=inputs["image_is_pad"],
            include_initial_video_step=False,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()
        loss_action = self._multi_action_loss(
            pred_action=pred_action,
            target_action=target_action,
            timestep_action=timestep_action,
            agent_mask=inputs["agent_mask"],
            action_is_pad=inputs["action_is_pad"],
        )
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        if self.training_mode == "action_only_cache":
            return self._training_loss_action_only(inputs)
        return self._training_loss_joint(inputs)

    @torch.no_grad()
    def infer_action_multi(
        self,
        *,
        input_image: torch.Tensor,
        action_horizon: int,
        agent_states: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.Tensor] = None,
        prompt: Optional[Union[str, Sequence[str]]] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, torch.Tensor]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[:2] != (1, 3):
            raise ValueError(f"`input_image` must be [1,3,H,W], got {tuple(input_image.shape)}")
        if agent_states.ndim == 2:
            agent_states = agent_states.unsqueeze(0)
        if agent_states.ndim != 3 or agent_states.shape[0] != 1:
            raise ValueError(f"`agent_states` must be [N,D] or [1,N,D], got {tuple(agent_states.shape)}")
        num_agents = int(agent_states.shape[1])
        if agent_mask is None:
            agent_mask = torch.ones((1, num_agents), dtype=torch.bool)
        elif agent_mask.ndim == 1:
            agent_mask = agent_mask.unsqueeze(0)
        if agent_ids is None:
            agent_ids = torch.arange(num_agents).unsqueeze(0)
        elif agent_ids.ndim == 1:
            agent_ids = agent_ids.unsqueeze(0)

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt == use_context:
            raise ValueError("Provide exactly one of `prompt` or `context/context_mask`.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("Both `context` and `context_mask` are required.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, num_agents, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        agent_states = agent_states.to(device=self.device, dtype=self.torch_dtype)
        agent_mask = agent_mask.to(device=self.device, dtype=torch.bool)
        agent_ids = agent_ids.to(device=self.device, dtype=torch.long)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image, tiled=tiled)
        timestep_video = torch.zeros((1,), device=self.device, dtype=first_frame_latents.dtype)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
        )
        video_seq_len = int(video_pre["tokens"].shape[1])

        # Mask structure is timestep-independent, so build it from an initial
        # pre-state and reuse it throughout flow-matching denoising.
        initial_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=torch.zeros((1,), device=self.device, dtype=latents_action.dtype),
            context=context,
            context_mask=context_mask,
            agent_states=agent_states,
            agent_mask=agent_mask,
            agent_ids=agent_ids,
        )
        attention_mask = self._build_multi_robot_attention_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            action_pre=initial_pre,
            device=self.device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[0, :video_seq_len, :video_seq_len],
        )

        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t, delta in zip(timesteps, deltas):
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=step_t.unsqueeze(0),
                context=context,
                context_mask=context_mask,
                agent_states=agent_states,
                agent_mask=agent_mask,
                agent_ids=agent_ids,
            )
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            latents_action = self.infer_action_scheduler.step(pred_action, delta, latents_action)

        return {"action": latents_action[0].detach().float().cpu()}

    @staticmethod
    def _load_matching_state(module, state: dict[str, Any], *, label: str):
        current = module.state_dict()
        compatible = {}
        skipped = []
        unexpected = []
        for key, value in state.items():
            if key not in current:
                unexpected.append(key)
                continue
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(current[key].shape):
                skipped.append(key)
                continue
            compatible[key] = value
        result = module.load_state_dict(compatible, strict=False)
        logger.info(
            "Loaded %s: compatible=%d shape_skipped=%d unexpected=%d missing=%d",
            label,
            len(compatible),
            len(skipped),
            len(unexpected),
            len(result.missing_keys),
        )
        if skipped:
            logger.warning("Shape-skipped %s keys (first 12): %s", label, skipped[:12])
        return compatible

    def save_checkpoint(self, path, optimizer=None, step=None):
        del optimizer
        payload: dict[str, Any] = {
            "format": "fastwam_multi_robot_v1",
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "trainable_scope": self._trainable_scope,
            "training_mode": self.training_mode,
            "base_checkpoint": self._loaded_base_checkpoint,
        }
        if self._trainable_scope == "dit":
            payload["mot"] = self.mot.state_dict()
        else:
            trainable_keys = {
                name for name, parameter in self.mot.named_parameters() if parameter.requires_grad
            }
            payload["mot_trainable"] = {
                name: value.detach().cpu()
                for name, value in self.mot.state_dict().items()
                if name in trainable_keys
            }
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Checkpoint payload must be a dict: {path}")

        base_checkpoint = payload.get("base_checkpoint")
        if "mot_trainable" in payload and base_checkpoint and not self._loaded_base_checkpoint:
            if str(base_checkpoint) != str(path):
                try:
                    self.load_checkpoint(str(base_checkpoint), optimizer=None)
                except FileNotFoundError:
                    logger.warning(
                        "Sparse checkpoint refers to missing base checkpoint %s; using current base weights.",
                        base_checkpoint,
                    )

        if "mot" in payload:
            self._load_matching_state(self.mot, payload["mot"], label="mot")
            self._loaded_base_checkpoint = str(path)
        elif "mot_trainable" in payload:
            self._load_matching_state(self.mot, payload["mot_trainable"], label="mot_trainable")
        elif "dit" in payload:
            self._load_matching_state(self.video_expert, payload["dit"], label="legacy video dit")
            self._loaded_base_checkpoint = str(path)
        else:
            raise ValueError(f"Checkpoint missing mot/mot_trainable/dit: {path}")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
