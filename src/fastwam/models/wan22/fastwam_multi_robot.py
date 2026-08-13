"""FastWAM adaptation for synchronized multi-robot collaboration."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from pathlib import Path
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

    CHECKPOINT_INTEGRITY_MODES = {"sha256", "metadata_no_hash"}

    def __init__(
        self,
        *args,
        training_mode: str = "action_only_cache",
        checkpoint_integrity_mode: str = "sha256",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if training_mode not in {"action_only_cache", "joint"}:
            raise ValueError(
                f"Unsupported training_mode={training_mode!r}; expected 'action_only_cache' or 'joint'."
            )
        self.training_mode = training_mode
        if self.training_mode == "action_only_cache" and self.loss_lambda_video != 0.0:
            raise ValueError(
                "training_mode='action_only_cache' requires loss_lambda_video=0"
            )
        if self.training_mode == "joint" and self.loss_lambda_video <= 0.0:
            raise ValueError("training_mode='joint' requires loss_lambda_video > 0")
        self.checkpoint_integrity_mode = self._validated_checkpoint_integrity_mode(
            checkpoint_integrity_mode
        )
        self._trainable_scope = "dit"
        self._loaded_base_checkpoint: Optional[str] = None
        self._loaded_base_checkpoint_sha256: Optional[str] = None
        self._loaded_base_checkpoint_descriptor: Optional[dict[str, Any]] = None
        self._loaded_base_checkpoint_can_restore_sparse = False

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
        checkpoint_integrity_mode: str = "sha256",
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
            checkpoint_integrity_mode=checkpoint_integrity_mode,
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
            checkpoint_integrity_mode=checkpoint_integrity_mode,
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
        if self.training_mode == "joint" and scope != "dit":
            raise ValueError(
                "Joint VideoGen training requires trainable_scope='dit' so the video "
                "loss has a trainable path."
            )
        if self.training_mode == "action_only_cache" and scope == "dit":
            raise ValueError(
                "Action-only training must use trainable_scope='action' or 'hub_io'; "
                "scope='dit' would leave unused trainable video parameters."
            )

        self.eval()
        self.requires_grad_(False)
        if scope == "hub_io":
            self.mot.train()
            self.video_expert.eval()
            self.action_expert.action_encoder.requires_grad_(True)
            self.action_expert.agent_state_encoder.requires_grad_(True)
            if self.action_expert.agent_geometry_encoder is not None:
                self.action_expert.agent_geometry_encoder.requires_grad_(True)
            if self.action_expert.gaussian_adapter is not None:
                self.action_expert.gaussian_adapter.requires_grad_(True)
            if self.action_expert.gaussian_gate is not None:
                self.action_expert.gaussian_gate.requires_grad_(True)
            self.action_expert.head.requires_grad_(True)
            self.action_expert.hub_seed.requires_grad_(self.action_expert.hub_enabled)
            self.action_expert.train()
        elif scope == "action":
            self.mot.train()
            self.video_expert.eval()
            self.action_expert.requires_grad_(True)
            self.action_expert.train()
        else:
            self.mot.requires_grad_(True)
            self.mot.train()

        # HUB0 retains the same model/state-dict schema as HUB1, but its shared
        # seed is inactive in forward and must not be a trainable DDP parameter.
        if not self.action_expert.hub_enabled:
            self.action_expert.hub_seed.requires_grad_(False)

        self._trainable_scope = scope
        params = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not params:
            raise RuntimeError(f"trainable_scope={scope!r} selected no parameters")
        return params

    def build_inputs(self, sample, tiled: bool = False):
        required = {"video", "action", "agent_state", "context", "context_mask"}
        if self.action_expert.agent_encoding_mode == "geometry":
            required.add("agent_geometry")
        if self.action_expert.enable_gaussian:
            required.add("agent_gaussian")
        missing = sorted(required - set(sample))
        if missing:
            raise ValueError(f"Missing multi-robot sample fields: {missing}")

        video = sample["video"]
        action = sample["action"]
        agent_state = sample["agent_state"]
        agent_geometry = sample.get("agent_geometry")
        agent_gaussian = sample.get("agent_gaussian")
        context = sample["context"]
        context_mask = sample["context_mask"]

        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"`video` must be [B,3,T,H,W], got {tuple(video.shape)}")
        batch_size, _, num_frames, height, width = video.shape
        if height % 16 or width % 16:
            raise ValueError(f"Video H/W must be multiples of 16, got {(height, width)}")
        if self.training_mode == "joint":
            if num_frames <= 1 or num_frames % 4 != 1:
                raise ValueError(
                    "Joint VideoGen input T must be >1 and satisfy T % 4 == 1, "
                    f"got {num_frames}"
                )
        elif num_frames < 1:
            raise ValueError("Action-only input must contain its observation frame")
        if action.ndim != 4:
            raise ValueError(f"`action` must be [B,N,H,A], got {tuple(action.shape)}")
        if action.shape[0] != batch_size or action.shape[-1] != self.action_expert.action_dim:
            raise ValueError(
                f"Action shape mismatch, got {tuple(action.shape)} for B={batch_size}, "
                f"A={self.action_expert.action_dim}"
            )
        num_agents, horizon = int(action.shape[1]), int(action.shape[2])
        if self.training_mode == "joint" and horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"Action horizon {horizon} must be divisible by video transitions {num_frames - 1}."
            )
        if agent_state.shape != (batch_size, num_agents, self.action_expert.state_dim):
            raise ValueError(
                f"`agent_state` must be {(batch_size, num_agents, self.action_expert.state_dim)}, "
                f"got {tuple(agent_state.shape)}"
            )
        if self.action_expert.agent_encoding_mode == "geometry":
            expected_geometry_shape = (
                batch_size,
                num_agents,
                self.action_expert.agent_geometry_dim,
            )
            if agent_geometry is None or agent_geometry.shape != expected_geometry_shape:
                got = None if agent_geometry is None else tuple(agent_geometry.shape)
                raise ValueError(
                    f"`agent_geometry` must be {expected_geometry_shape}, got {got}"
                )
        if self.action_expert.enable_gaussian:
            expected_gaussian_shape = (
                batch_size,
                num_agents,
                self.action_expert.gaussian_channels,
                self.action_expert.gaussian_height,
                self.action_expert.gaussian_width,
            )
            if agent_gaussian is None or agent_gaussian.shape != expected_gaussian_shape:
                got = None if agent_gaussian is None else tuple(agent_gaussian.shape)
                raise ValueError(
                    f"`agent_gaussian` must be {expected_gaussian_shape}, got {got}"
                )
            if not torch.is_floating_point(agent_gaussian):
                raise TypeError(
                    "`agent_gaussian` must be floating point (the canonical cache is FP16), "
                    f"got {agent_gaussian.dtype}"
                )
        else:
            # Gaussian-off remains compatible with loaders that either omit or
            # unconditionally provide the ablated field.
            agent_gaussian = None

        agent_count = sample.get("agent_count")
        if agent_count is not None:
            counts = torch.as_tensor(agent_count).reshape(-1)
            if counts.numel() != batch_size or not bool((counts == num_agents).all().item()):
                raise ValueError(
                    "A native variable-length batch must contain one cardinality; "
                    f"tensor N={num_agents}, metadata={counts.tolist()}"
                )

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
            "agent_geometry": (
                None
                if agent_geometry is None
                else agent_geometry.to(
                    device=self.device, dtype=self.torch_dtype, non_blocking=True
                )
            ),
            "agent_gaussian": (
                None
                if agent_gaussian is None
                else agent_gaussian.to(
                    device=self.device, non_blocking=True
                )
            ),
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

    @staticmethod
    def _multi_robot_attention_layout(
        *,
        video_seq_len: int,
        video_tokens_per_frame: int,
        action_pre: dict[str, Any],
    ) -> dict[str, int]:
        meta = action_pre["meta"]
        num_agents = int(meta["num_agents"])
        horizon = int(meta["horizon"])
        num_hubs = int(meta["num_hub_tokens"])
        expected_action_seq_len = num_agents * horizon + num_hubs
        actual_action_seq_len = int(action_pre["tokens"].shape[1])
        if actual_action_seq_len != expected_action_seq_len:
            raise ValueError(
                "Action pre-state/layout mismatch: "
                f"tokens={actual_action_seq_len}, expected={expected_action_seq_len}"
            )
        return {
            "num_agents": num_agents,
            "horizon": horizon,
            "num_hub_tokens": num_hubs,
            "first_frame_tokens": min(video_tokens_per_frame, video_seq_len),
        }

    def _build_video_attention_mask(
        self,
        *,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build only the world-stream mask, never a global agent mask."""

        return self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

    def _multi_action_loss(
        self,
        *,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        timestep_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        token_loss = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=-1)
        valid = torch.ones_like(token_loss, dtype=torch.bool)
        if action_is_pad is not None:
            valid = ~action_is_pad
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
            agent_geometry=inputs["agent_geometry"],
            agent_ids=inputs["agent_ids"],
            agent_gaussian=inputs["agent_gaussian"],
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
            video_seq_len = int(video_pre["tokens"].shape[1])
            video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
            attention_layout = self._multi_robot_attention_layout(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                action_pre=action_pre,
            )
            video_attention_mask = self._build_video_attention_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                device=video_pre["tokens"].device,
            )
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=video_attention_mask,
            )

        action_tokens = self.mot.forward_multi_agent_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            **attention_layout,
        )
        pred_action = self.action_expert.post_dit(action_tokens, action_pre)
        loss_action = self._multi_action_loss(
            pred_action=pred_action,
            target_action=target_action,
            timestep_action=timestep_action,
            action_is_pad=inputs["action_is_pad"],
        )
        loss_total = self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    def _training_loss_joint(self, inputs: dict[str, Any]):
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]

        # Keep action corruption on the same RNG substream as the VG0
        # action-only arm.  Offline evaluation forks/seeds per sample; drawing
        # video noise first here would shift both action noise and its timestep
        # in VG1, making val_loss_action a different Monte Carlo measurement
        # even when the two arms share initialization and seed.
        noisy_action, target_action, timestep_action = self._prepare_noisy_action(inputs)

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

        video_pre = self.video_expert.pre_dit(
            x=noisy_video,
            timestep=timestep_video,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self._action_pre(noisy_action, timestep_action, inputs)
        video_seq_len = int(video_pre["tokens"].shape[1])
        video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        attention_layout = self._multi_robot_attention_layout(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            action_pre=action_pre,
        )
        video_attention_mask = self._build_video_attention_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot.forward_multi_agent_joint(
            video_tokens=video_pre["tokens"],
            action_tokens=action_pre["tokens"],
            video_freqs=video_pre["freqs"],
            action_freqs=action_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            action_t_mod=action_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_attention_mask=video_attention_mask,
            **attention_layout,
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
        agent_geometry: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.Tensor] = None,
        agent_gaussian: Optional[torch.Tensor] = None,
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
        if agent_geometry is not None and agent_geometry.ndim == 2:
            agent_geometry = agent_geometry.unsqueeze(0)
        if self.action_expert.agent_encoding_mode == "geometry":
            expected_geometry_shape = (
                1,
                num_agents,
                self.action_expert.agent_geometry_dim,
            )
            if agent_geometry is None or agent_geometry.shape != expected_geometry_shape:
                got = None if agent_geometry is None else tuple(agent_geometry.shape)
                raise ValueError(
                    f"`agent_geometry` must be [N,G] or {expected_geometry_shape}, got {got}"
                )
        if agent_gaussian is not None and agent_gaussian.ndim == 4:
            agent_gaussian = agent_gaussian.unsqueeze(0)
        if self.action_expert.enable_gaussian:
            expected_gaussian_shape = (
                1,
                num_agents,
                self.action_expert.gaussian_channels,
                self.action_expert.gaussian_height,
                self.action_expert.gaussian_width,
            )
            if agent_gaussian is None or agent_gaussian.shape != expected_gaussian_shape:
                got = None if agent_gaussian is None else tuple(agent_gaussian.shape)
                raise ValueError(
                    "enable_gaussian=true requires `agent_gaussian` as [N,C,H,W] "
                    f"or {expected_gaussian_shape}, got {got}"
                )
            if not torch.is_floating_point(agent_gaussian):
                raise TypeError(
                    "`agent_gaussian` must be floating point (the canonical cache is FP16), "
                    f"got {agent_gaussian.dtype}"
                )
        else:
            agent_gaussian = None
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
        if agent_geometry is not None:
            agent_geometry = agent_geometry.to(device=self.device, dtype=self.torch_dtype)
        if agent_gaussian is not None:
            agent_gaussian = agent_gaussian.to(device=self.device)
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

        # Native N/H/K layout is timestep-independent and is reused throughout
        # denoising. Only the video stream needs an attention mask; the action
        # stream uses factorized local/hub SDPA calls.
        initial_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=torch.zeros((1,), device=self.device, dtype=latents_action.dtype),
            context=context,
            context_mask=context_mask,
            agent_states=agent_states,
            agent_geometry=agent_geometry,
            agent_ids=agent_ids,
            agent_gaussian=agent_gaussian,
        )
        video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        attention_layout = self._multi_robot_attention_layout(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            action_pre=initial_pre,
        )
        video_attention_mask = self._build_video_attention_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
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
            video_attention_mask=video_attention_mask,
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
                agent_geometry=agent_geometry,
                agent_ids=agent_ids,
                agent_gaussian=agent_gaussian,
            )
            action_tokens = self.mot.forward_multi_agent_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                video_kv_cache=video_kv_cache,
                **attention_layout,
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

    @staticmethod
    def _upgrade_legacy_hub_state(state: dict[str, Any]) -> dict[str, Any]:
        """Convert a fixed ``hub_tokens[K,D]`` bank to the v2 shared seed."""

        upgraded = dict(state)
        for key in list(upgraded):
            if key != "hub_tokens" and not key.endswith(".hub_tokens"):
                continue
            value = upgraded.pop(key)
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] < 1:
                logger.warning("Cannot convert legacy hub tensor %s with value %s", key, type(value))
                continue
            seed_key = f"{key[:-len('hub_tokens')]}hub_seed"
            upgraded.setdefault(seed_key, value.mean(dim=0, keepdim=True))
            logger.info("Converted legacy %s to cardinality-independent %s", key, seed_key)
        return upgraded

    @staticmethod
    def _absolute_checkpoint_path(path: str | Path) -> Path:
        """Normalize syntax without resolving any filesystem alias."""

        checkpoint_path = Path(path).expanduser()
        return Path(os.path.abspath(checkpoint_path))

    @classmethod
    def _validated_checkpoint_integrity_mode(cls, value: str) -> str:
        mode = str(value).strip().lower()
        if mode not in cls.CHECKPOINT_INTEGRITY_MODES:
            raise ValueError(
                "checkpoint_integrity_mode must be one of "
                f"{sorted(cls.CHECKPOINT_INTEGRITY_MODES)}, got {value!r}"
            )
        return mode

    def _active_checkpoint_integrity_mode(self) -> str:
        # The fallback keeps old lightweight tests and manually constructed
        # objects on the historical path unless they opt in explicitly.
        return self._validated_checkpoint_integrity_mode(
            getattr(self, "checkpoint_integrity_mode", "sha256")
        )

    @classmethod
    def _open_checkpoint_descriptor(
        cls, path: str | Path
    ) -> tuple[Path, int, os.stat_result]:
        """Open a unique regular checkpoint without following path aliases.

        Every ancestor is opened relative to the already verified parent with
        ``O_NOFOLLOW``.  The leaf is opened through the final parent descriptor,
        so neither a leaf symlink nor a symlinked ancestor can be hidden by a
        prior ``Path.resolve``.  A single hard link is required because the
        opened inode must have one unambiguous provenance path.
        """

        checkpoint_path = cls._absolute_checkpoint_path(path)
        if checkpoint_path == Path(os.sep) or not checkpoint_path.name:
            raise ValueError(f"Checkpoint path must name a file: {checkpoint_path}")

        directory_flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        file_flags = (
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        parent_descriptor = os.open(os.sep, directory_flags)
        file_descriptor: int | None = None
        try:
            for component in checkpoint_path.parts[1:-1]:
                try:
                    next_descriptor = os.open(
                        component,
                        directory_flags,
                        dir_fd=parent_descriptor,
                    )
                except OSError as error:
                    if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise ValueError(
                            "Checkpoint path must not traverse symlinks or "
                            f"non-directory ancestors: {checkpoint_path}"
                        ) from error
                    raise
                os.close(parent_descriptor)
                parent_descriptor = next_descriptor
            try:
                file_descriptor = os.open(
                    checkpoint_path.name,
                    file_flags,
                    dir_fd=parent_descriptor,
                )
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(
                        "Checkpoint path must not be a symlink or traverse "
                        f"symlinked ancestors: {checkpoint_path}"
                    ) from error
                raise
        finally:
            os.close(parent_descriptor)

        try:
            before = os.fstat(file_descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(
                    f"Checkpoint is not a regular file: {checkpoint_path}"
                )
            if before.st_nlink != 1:
                raise ValueError(
                    "Checkpoint must not be hard-linked: "
                    f"nlink={before.st_nlink} path={checkpoint_path}"
                )
            return checkpoint_path, file_descriptor, before
        except BaseException:
            os.close(file_descriptor)
            raise

    @classmethod
    def _checkpoint_sha256(cls, path: str | Path) -> str:
        checkpoint_path, descriptor, before = cls._open_checkpoint_descriptor(path)
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=True) as checkpoint_file:
            while chunk := checkpoint_file.read(8 * 1024 * 1024):
                digest.update(chunk)
            after = os.fstat(checkpoint_file.fileno())
            if cls._checkpoint_file_identity(before) != cls._checkpoint_file_identity(
                after
            ):
                raise RuntimeError(
                    "Checkpoint changed while being hashed: "
                    f"{checkpoint_path}"
                )
        return digest.hexdigest()

    @staticmethod
    def _checkpoint_file_identity(file_stat: os.stat_result) -> tuple[int, ...]:
        """Return the metadata that must stay stable across a checkpoint read."""

        return (
            int(file_stat.st_dev),
            int(file_stat.st_ino),
            int(file_stat.st_mode),
            int(file_stat.st_nlink),
            int(file_stat.st_size),
            int(file_stat.st_mtime_ns),
            int(file_stat.st_ctime_ns),
        )

    @staticmethod
    def _checkpoint_metadata_stat(file_stat: os.stat_result) -> dict[str, int]:
        return {
            "bytes": int(file_stat.st_size),
            "mtime_ns": int(file_stat.st_mtime_ns),
            "dev": int(file_stat.st_dev),
            "ino": int(file_stat.st_ino),
            "mode": int(file_stat.st_mode),
        }

    @classmethod
    def _metadata_base_descriptor(
        cls,
        checkpoint_path: Path,
        file_stat: os.stat_result,
    ) -> dict[str, Any]:
        return {
            "path": str(checkpoint_path),
            "role": "base_dependency",
            "integrity_mode": "metadata_no_hash",
            "stat": cls._checkpoint_metadata_stat(file_stat),
        }

    @classmethod
    def _load_pinned_checkpoint_payload(
        cls,
        path: Path,
        *,
        expected_sha256: str | None,
        digest_label: str,
        integrity_mode: str = "sha256",
        expected_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str | dict[str, Any]]:
        """Validate and deserialize exactly one opened checkpoint object.

        ``metadata_no_hash`` pins the canonical path and opened inode metadata
        without reading bytes for a checksum.  Reusing one descriptor makes an
        atomic pathname replacement incapable of changing the bytes that are
        deserialized.  Before/after ``fstat`` checks reject mutation of that
        opened inode during the read.  The historical SHA-256 path remains the
        default for callers that do not opt in.
        """

        integrity_mode = cls._validated_checkpoint_integrity_mode(integrity_mode)
        if integrity_mode == "metadata_no_hash":
            if expected_sha256 is not None:
                raise ValueError(
                    f"{digest_label} expected_sha256 is incompatible with "
                    "checkpoint_integrity_mode='metadata_no_hash'"
                )
            checkpoint_path, descriptor, before = cls._open_checkpoint_descriptor(path)
            observed_metadata = cls._metadata_base_descriptor(
                checkpoint_path,
                before,
            )
            if expected_metadata is not None and observed_metadata != expected_metadata:
                raise ValueError(
                    f"{digest_label} metadata mismatch before deserialization: "
                    f"expected={expected_metadata!r} actual={observed_metadata!r} "
                    f"path={checkpoint_path}"
                )
            with os.fdopen(descriptor, "rb", closefd=True) as checkpoint_file:
                payload = torch.load(
                    checkpoint_file,
                    map_location="cpu",
                    weights_only=True,
                )
                after = os.fstat(checkpoint_file.fileno())
                if cls._checkpoint_file_identity(
                    before
                ) != cls._checkpoint_file_identity(after):
                    raise RuntimeError(
                        "Checkpoint changed while being deserialized; refusing "
                        f"to mutate the model: {checkpoint_path}"
                    )
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Checkpoint payload must be a dict: {checkpoint_path}"
                )
            return payload, observed_metadata

        normalized_expected = None
        if expected_sha256 is not None:
            normalized_expected = str(expected_sha256).strip().lower()
            if (
                len(normalized_expected) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in normalized_expected
                )
            ):
                raise ValueError(
                    f"Invalid expected {digest_label} SHA-256: {expected_sha256!r}"
                )

        checkpoint_path, descriptor, before = cls._open_checkpoint_descriptor(path)
        with os.fdopen(descriptor, "rb", closefd=True) as checkpoint_file:
            digest = hashlib.sha256()
            while chunk := checkpoint_file.read(8 * 1024 * 1024):
                digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if (
                normalized_expected is not None
                and actual_sha256 != normalized_expected
            ):
                raise ValueError(
                    f"{digest_label} SHA-256 mismatch: "
                    "expected="
                    f"{normalized_expected} actual={actual_sha256} "
                    f"path={checkpoint_path}"
                )

            checkpoint_file.seek(0)
            payload = torch.load(
                checkpoint_file,
                map_location="cpu",
                weights_only=True,
            )
            after = os.fstat(checkpoint_file.fileno())
            if cls._checkpoint_file_identity(before) != cls._checkpoint_file_identity(
                after
            ):
                raise RuntimeError(
                    "Checkpoint changed while being hashed/deserialized; refusing "
                    f"to mutate the model: {checkpoint_path}"
                )

        if not isinstance(payload, dict):
            raise ValueError(f"Checkpoint payload must be a dict: {checkpoint_path}")
        return payload, actual_sha256

    @staticmethod
    def _validate_exact_tensor_state(
        expected_state: dict[str, Any],
        received_state: dict[str, Any],
        *,
        expected_keys: Sequence[str],
        path,
        label: str,
    ) -> None:
        """Validate a native checkpoint before mutating the live module."""

        if not isinstance(received_state, dict):
            raise TypeError(f"Checkpoint {label} state must be a dict: {path}")
        expected_names = set(expected_keys)
        missing = sorted(expected_names - set(received_state))
        unexpected = sorted(set(received_state) - expected_names)
        unknown_expected = sorted(expected_names - set(expected_state))
        if unknown_expected:
            raise RuntimeError(
                f"Internal {label} contract contains unknown model keys: "
                f"{unknown_expected[:12]}"
            )

        type_mismatches = []
        shape_mismatches = []
        dtype_mismatches = []
        for key in sorted(expected_names & set(received_state)):
            value = received_state[key]
            target = expected_state[key]
            if not isinstance(value, torch.Tensor):
                type_mismatches.append((key, type(value).__name__))
                continue
            if tuple(value.shape) != tuple(target.shape):
                shape_mismatches.append(
                    (key, tuple(value.shape), tuple(target.shape))
                )
            if value.dtype != target.dtype:
                dtype_mismatches.append((key, str(value.dtype), str(target.dtype)))

        if (
            missing
            or unexpected
            or type_mismatches
            or shape_mismatches
            or dtype_mismatches
        ):
            raise ValueError(
                f"Strict native v2 {label} state mismatch in {path}: "
                f"missing={missing[:12]} (count={len(missing)}), "
                f"unexpected={unexpected[:12]} (count={len(unexpected)}), "
                f"type_mismatches={type_mismatches[:12]} "
                f"(count={len(type_mismatches)}), "
                f"shape_mismatches={shape_mismatches[:12]} "
                f"(count={len(shape_mismatches)}), "
                f"dtype_mismatches={dtype_mismatches[:12]} "
                f"(count={len(dtype_mismatches)})"
            )

    @staticmethod
    def _native_checkpoint_state_kind(payload: dict[str, Any], path) -> str:
        state_kind = payload.get("state_kind")
        if state_kind not in {"full", "sparse_delta"}:
            raise ValueError(
                "Native v2 checkpoint must declare state_kind='full' or "
                f"'sparse_delta': {path}"
            )
        has_full = "mot" in payload
        has_sparse = "mot_trainable" in payload
        if state_kind == "full" and (not has_full or has_sparse):
            raise ValueError(
                "Native v2 full checkpoint must contain only `mot` state: "
                f"mot={has_full} mot_trainable={has_sparse} in {path}"
            )
        if state_kind == "sparse_delta" and (not has_sparse or has_full):
            raise ValueError(
                "Native v2 sparse_delta checkpoint must contain only "
                f"`mot_trainable` state: mot={has_full} "
                f"mot_trainable={has_sparse} in {path}"
            )
        return str(state_kind)

    def _expected_trainable_parameter_names(self) -> list[str]:
        names = sorted(
            name for name, parameter in self.mot.named_parameters() if parameter.requires_grad
        )
        if not names:
            raise RuntimeError(
                "Native sparse checkpoint requires trainable parameters to be configured "
                "before save/load."
            )
        return names

    def _validate_sparse_trainable_contract(
        self,
        payload: dict[str, Any],
        state: dict[str, Any],
        *,
        path,
    ) -> list[str]:
        declared = payload.get("trainable_parameter_names")
        if not isinstance(declared, list) or not all(
            isinstance(name, str) and name for name in declared
        ):
            raise ValueError(
                "Native v2 sparse_delta must declare a non-empty string list in "
                f"trainable_parameter_names: {path}"
            )
        canonical_declared = sorted(set(declared))
        if declared != canonical_declared:
            raise ValueError(
                "Native v2 trainable_parameter_names must be sorted and unique: "
                f"{path}"
            )
        expected_names = self._expected_trainable_parameter_names()
        if declared != expected_names:
            missing = sorted(set(expected_names) - set(declared))
            unexpected = sorted(set(declared) - set(expected_names))
            raise ValueError(
                "Native v2 sparse trainable contract mismatch: "
                f"missing={missing[:12]} (count={len(missing)}), "
                f"unexpected={unexpected[:12]} (count={len(unexpected)}) in {path}"
            )
        self._validate_exact_tensor_state(
            self.mot.state_dict(),
            state,
            expected_keys=expected_names,
            path=path,
            label="mot_trainable",
        )
        return expected_names

    def _legacy_required_mot_keys(self) -> set[str]:
        current = self.mot.state_dict()
        video_keys = {
            key for key in current if key.startswith("mixtures.video.")
        }
        action_backbone_keys = {
            f"mixtures.action.{key}"
            for key in self.action_expert.backbone_key_set(
                self.action_expert.state_dict().keys()
            )
        }
        required = video_keys | action_backbone_keys
        if not video_keys or not action_backbone_keys:
            raise RuntimeError(
                "Cannot establish minimum legacy FastWAM coverage for video/action backbones."
            )
        return required

    @staticmethod
    def _shape_compatible_keys(module, state: dict[str, Any]) -> set[str]:
        if not isinstance(state, dict):
            return set()
        current = module.state_dict()
        return {
            key
            for key, value in state.items()
            if key in current
            and isinstance(value, torch.Tensor)
            and tuple(value.shape) == tuple(current[key].shape)
        }

    def _validate_legacy_minimum_coverage(
        self,
        state: dict[str, Any],
        *,
        path,
        label: str,
        load_role: str,
    ) -> None:
        if not isinstance(state, dict):
            raise TypeError(f"Legacy checkpoint {label} state must be a dict: {path}")
        if label == "mot":
            required = self._legacy_required_mot_keys()
            compatible = self._shape_compatible_keys(self.mot, state)
        elif label == "dit":
            if load_role == "base_dependency":
                raise ValueError(
                    "A sparse native v2 checkpoint cannot depend on a legacy "
                    f"video-only `dit` checkpoint: {path}"
                )
            required = set(self.video_expert.state_dict())
            compatible = self._shape_compatible_keys(self.video_expert, state)
        else:
            raise RuntimeError(f"Unsupported legacy checkpoint label: {label}")
        missing = sorted(required - compatible)
        if missing:
            raise ValueError(
                f"Legacy {label} checkpoint lacks minimum safe backbone coverage: "
                f"missing={missing[:12]} (count={len(missing)}) in {path}"
            )

    def _validated_base_dependency_descriptor(
        self,
        descriptor: Any,
        *,
        owner_path: str | Path,
        active_paths: set[Path],
    ) -> dict[str, Any]:
        integrity_mode = self._active_checkpoint_integrity_mode()
        if integrity_mode == "metadata_no_hash":
            required_fields = {"path", "role", "integrity_mode", "stat"}
            if not isinstance(descriptor, dict):
                raise ValueError(
                    "Native sparse checkpoint base_checkpoint must be a "
                    "metadata_no_hash descriptor: "
                    f"{owner_path}"
                )
            if set(descriptor) != required_fields:
                raise ValueError(
                    "Native sparse checkpoint metadata base descriptor fields "
                    f"mismatch: expected={sorted(required_fields)} "
                    f"got={sorted(descriptor)} in {owner_path}"
                )
            if descriptor["role"] != "base_dependency":
                raise ValueError(
                    "Native sparse checkpoint base descriptor role must be "
                    f"'base_dependency', got {descriptor['role']!r} in {owner_path}"
                )
            if descriptor["integrity_mode"] != "metadata_no_hash":
                raise ValueError(
                    "Native sparse checkpoint integrity mode mismatch: expected "
                    f"'metadata_no_hash', got {descriptor['integrity_mode']!r} "
                    f"in {owner_path}"
                )
            raw_path = descriptor["path"]
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError(f"Invalid base dependency path in {owner_path}")
            dependency_path = Path(raw_path).expanduser()
            if not dependency_path.is_absolute():
                raise ValueError(
                    "metadata_no_hash base dependency path must be absolute: "
                    f"{raw_path!r} in {owner_path}"
                )
            dependency_path = self._absolute_checkpoint_path(dependency_path)
            if raw_path != str(dependency_path):
                raise ValueError(
                    "metadata_no_hash base dependency path must be canonical: "
                    f"{raw_path!r} in {owner_path}"
                )
            if dependency_path in active_paths:
                raise ValueError(
                    "Checkpoint base dependency cycle detected: "
                    f"{dependency_path} is already active"
                )
            received_stat = descriptor["stat"]
            required_stat_fields = {"bytes", "mtime_ns", "dev", "ino", "mode"}
            if not isinstance(received_stat, dict) or set(received_stat) != required_stat_fields:
                got = sorted(received_stat) if isinstance(received_stat, dict) else type(received_stat)
                raise ValueError(
                    "metadata_no_hash base stat fields mismatch: "
                    f"expected={sorted(required_stat_fields)} got={got} "
                    f"in {owner_path}"
                )
            if not all(
                isinstance(received_stat[field], int)
                and not isinstance(received_stat[field], bool)
                for field in required_stat_fields
            ):
                raise ValueError(
                    f"metadata_no_hash base stat values must be integers in {owner_path}"
                )
            if received_stat["bytes"] < 0 or not stat.S_ISREG(received_stat["mode"]):
                raise ValueError(
                    f"metadata_no_hash base stat must describe a regular file in {owner_path}"
                )
            return {
                "path": str(dependency_path),
                "role": "base_dependency",
                "integrity_mode": "metadata_no_hash",
                "stat": {
                    field: int(received_stat[field])
                    for field in ("bytes", "mtime_ns", "dev", "ino", "mode")
                },
            }

        if not isinstance(descriptor, dict):
            raise ValueError(
                "Native sparse checkpoint base_checkpoint must be a "
                f"{{path, sha256, role}} object: {owner_path}"
            )
        required_fields = {"path", "sha256", "role"}
        if set(descriptor) != required_fields:
            raise ValueError(
                "Native sparse checkpoint base descriptor fields mismatch: "
                f"expected={sorted(required_fields)} got={sorted(descriptor)} "
                f"in {owner_path}"
            )
        if descriptor["role"] != "base_dependency":
            raise ValueError(
                "Native sparse checkpoint base descriptor role must be "
                f"'base_dependency', got {descriptor['role']!r} in {owner_path}"
            )
        raw_path = descriptor["path"]
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"Invalid base dependency path in {owner_path}")
        dependency_path = Path(raw_path).expanduser()
        if not dependency_path.is_absolute():
            dependency_path = Path(owner_path).parent / dependency_path
        dependency_path = self._absolute_checkpoint_path(dependency_path)
        if dependency_path in active_paths:
            raise ValueError(
                "Checkpoint base dependency cycle detected: "
                f"{dependency_path} is already active"
            )
        expected_sha256 = descriptor["sha256"]
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected_sha256)
        ):
            raise ValueError(
                f"Invalid base checkpoint SHA-256 in {owner_path}: {expected_sha256!r}"
            )
        expected_sha256 = expected_sha256.lower()
        return {
            "path": str(dependency_path),
            "sha256": expected_sha256,
            "role": "base_dependency",
        }

    def _base_dependency_descriptor_for_save(self, output_path) -> dict[str, Any]:
        if not getattr(self, "_loaded_base_checkpoint_can_restore_sparse", False):
            raise RuntimeError(
                "Cannot save sparse_delta: no loaded full `mot` checkpoint can "
                "reconstruct frozen parameters."
            )
        base_path_value = getattr(self, "_loaded_base_checkpoint", None)
        if not base_path_value:
            raise RuntimeError("Cannot save sparse_delta without a base checkpoint path.")
        base_path = self._absolute_checkpoint_path(str(base_path_value))
        if self._active_checkpoint_integrity_mode() == "metadata_no_hash":
            output_resolved = self._absolute_checkpoint_path(output_path)
        else:
            output_resolved = Path(output_path).expanduser().resolve(strict=False)
        if output_resolved == base_path:
            raise ValueError(
                "A sparse checkpoint cannot overwrite its own base dependency: "
                f"{base_path}"
            )

        cached = getattr(self, "_loaded_base_checkpoint_descriptor", None)
        if self._active_checkpoint_integrity_mode() == "metadata_no_hash":
            checkpoint_path, descriptor_fd, before = self._open_checkpoint_descriptor(
                base_path
            )
            with os.fdopen(descriptor_fd, "rb", closefd=True) as checkpoint_file:
                after = os.fstat(checkpoint_file.fileno())
            if self._checkpoint_file_identity(before) != self._checkpoint_file_identity(
                after
            ):
                raise RuntimeError(
                    "Loaded base checkpoint changed while validating sparse save: "
                    f"{checkpoint_path}"
                )
            current_descriptor = self._metadata_base_descriptor(
                checkpoint_path,
                before,
            )
            if not isinstance(cached, dict) or cached != current_descriptor:
                raise RuntimeError(
                    "Loaded base checkpoint metadata changed before sparse save: "
                    f"expected={cached!r} actual={current_descriptor!r} "
                    f"path={base_path}"
                )
            self._loaded_base_checkpoint_sha256 = None
            self._loaded_base_checkpoint_descriptor = current_descriptor
            return dict(current_descriptor)

        current_sha256 = self._checkpoint_sha256(base_path)
        if isinstance(cached, dict) and cached.get("path") == str(base_path):
            cached_sha256 = str(cached.get("sha256", "")).strip().lower()
            if current_sha256 != cached_sha256:
                raise RuntimeError(
                    "Loaded base checkpoint changed before sparse save: "
                    f"expected={cached_sha256} actual={current_sha256} path={base_path}"
                )
            return dict(cached)
        descriptor = {
            "path": str(base_path),
            "sha256": current_sha256,
            "role": "base_dependency",
        }
        self._loaded_base_checkpoint_sha256 = current_sha256
        self._loaded_base_checkpoint_descriptor = descriptor
        return dict(descriptor)

    def _remember_loaded_base_checkpoint(
        self,
        checkpoint_path: Path,
        checkpoint_evidence: str | dict[str, Any],
    ) -> None:
        self._loaded_base_checkpoint = str(checkpoint_path)
        if self._active_checkpoint_integrity_mode() == "metadata_no_hash":
            if not isinstance(checkpoint_evidence, dict):
                raise RuntimeError(
                    "metadata_no_hash checkpoint load did not return metadata evidence"
                )
            self._loaded_base_checkpoint_sha256 = None
            self._loaded_base_checkpoint_descriptor = dict(checkpoint_evidence)
        else:
            if not isinstance(checkpoint_evidence, str):
                raise RuntimeError("SHA-256 checkpoint load did not return a digest")
            self._loaded_base_checkpoint_sha256 = checkpoint_evidence
            self._loaded_base_checkpoint_descriptor = {
                "path": str(checkpoint_path),
                "sha256": checkpoint_evidence,
                "role": "base_dependency",
            }
        self._loaded_base_checkpoint_can_restore_sparse = True

    def _multi_robot_architecture_metadata(self) -> dict[str, Any]:
        metadata = {
            "agent_set_representation": "native_variable_length_v1",
            "action_attention_topology": "factorized_agent_local_hub_v1",
            "action_dim": self.action_expert.action_dim,
            "state_dim": self.action_expert.state_dim,
            "hidden_dim": self.action_expert.hidden_dim,
            "ffn_dim": self.action_expert.ffn_dim,
            "num_layers": len(self.action_expert.blocks),
            "num_heads": self.action_expert.num_heads,
            "attn_head_dim": self.action_expert.attn_head_dim,
            "text_dim": self.action_expert.text_dim,
            "freq_dim": self.action_expert.freq_dim,
            "agent_encoding_mode": self.action_expert.agent_encoding_mode,
            "agent_geometry_dim": self.action_expert.agent_geometry_dim,
            "agent_geometry_schema": "robofactory_root_pose_xyz_canonical_qwqxqyqz_v1",
            "agent_rope_dim": self.action_expert.agent_rope_dim,
            "agent_phase_scale": self.action_expert.agent_phase_scale,
            "hub_enabled": self.action_expert.hub_enabled,
            "hub_token_policy": "ceil(hub_token_ratio*num_agents)",
            "hub_token_ratio": self.action_expert.hub_token_ratio,
            "hub_position_scale": self.action_expert.hub_position_scale,
            "enable_gaussian": self.action_expert.enable_gaussian,
        }
        if self.action_expert.enable_gaussian:
            gaussian_metadata = {
                "gaussian_shape": [
                    self.action_expert.gaussian_channels,
                    self.action_expert.gaussian_height,
                    self.action_expert.gaussian_width,
                ],
                "gaussian_hidden_dim": self.action_expert.hidden_dim,
                "gaussian_stem_dim": self.action_expert.gaussian_stem_dim,
                "gaussian_gate_init": 0.0,
            }
            if self.action_expert.gaussian_conditioning_mode == "pooled_residual":
                gaussian_metadata.update(
                    {
                        "gaussian_conditioning": "agent_local_residual_v1",
                        "gaussian_adapter_version": "conv_gn_silu_pool_v1",
                    }
                )
            else:
                gaussian_metadata.update(
                    {
                        "gaussian_conditioning": "agent_local_spatial_cross_attention_v2",
                        "gaussian_adapter_version": "conv_gn_silu_spatial_reuse_v2",
                        "gaussian_spatial_position_encoding": "sincos_2d_v1",
                        "gaussian_residual_floor": self.action_expert.gaussian_residual_floor,
                        "gaussian_attention_temperature": (
                            self.action_expert.gaussian_attention_temperature
                        ),
                    }
                )
            metadata.update(gaussian_metadata)
        return metadata

    def _validate_gaussian_v2_state(
        self,
        payload: dict[str, Any],
        state: dict[str, Any],
        *,
        path,
        label: str,
    ) -> None:
        """Require every GAU1 adapter/gate tensor in native v2 checkpoints.

        Official FastWAM checkpoints predate the multi-robot v2 envelope and
        remain permissive.  Once a checkpoint declares v2 GAU1 metadata,
        silently retaining newly initialized Gaussian parameters would make a
        resumed treatment scientifically different, so those tensors are
        strict even though legacy backbone loading remains shape-tolerant.
        """

        if (
            payload.get("format") != "fastwam_multi_robot_v2"
            or not self.action_expert.enable_gaussian
        ):
            return
        if not isinstance(state, dict):
            raise TypeError(f"Checkpoint {label} state must be a dict: {path}")

        def is_gaussian_key(key: str) -> bool:
            return (
                ".gaussian_adapter." in key
                or key.endswith(".gaussian_gate")
            )

        current = self.mot.state_dict()
        required = {key for key in current if is_gaussian_key(key)}
        received = {key for key in state if is_gaussian_key(key)}
        missing = sorted(required - received)
        unexpected = sorted(received - required)
        shape_mismatches = []
        for key in sorted(required & received):
            value = state[key]
            if not isinstance(value, torch.Tensor):
                shape_mismatches.append(
                    (key, type(value).__name__, tuple(current[key].shape))
                )
            elif tuple(value.shape) != tuple(current[key].shape):
                shape_mismatches.append(
                    (key, tuple(value.shape), tuple(current[key].shape))
                )
        if missing or unexpected or shape_mismatches:
            raise ValueError(
                f"Strict GAU1 {label} state mismatch in {path}: "
                f"missing={missing}, unexpected={unexpected}, "
                f"shape_mismatches={shape_mismatches}"
            )

    def _validate_multi_robot_checkpoint_metadata(
        self,
        payload: dict[str, Any],
        path,
        *,
        validate_treatment: bool = True,
        validate_trainable_scope: bool = True,
    ) -> None:
        if payload.get("format") != "fastwam_multi_robot_v2":
            return
        for key, allowed_values in (
            ("training_mode", {"action_only_cache", "joint"}),
            ("trainable_scope", {"hub_io", "action", "dit"}),
        ):
            if key not in payload:
                raise ValueError(f"Native v2 checkpoint is missing {key!r}: {path}")
            if payload[key] not in allowed_values:
                raise ValueError(
                    f"Native v2 checkpoint has invalid {key}={payload[key]!r}: {path}"
                )
        if validate_treatment:
            treatment_expected = {"training_mode": self.training_mode}
            if validate_trainable_scope:
                treatment_expected["trainable_scope"] = self._trainable_scope
            for key, expected_value in treatment_expected.items():
                if payload[key] != expected_value:
                    raise ValueError(
                        f"Checkpoint treatment mismatch for {key}: "
                        f"expected {expected_value!r}, got {payload[key]!r} in {path}"
                    )

        received = payload.get("multi_robot_architecture")
        if not isinstance(received, dict):
            raise ValueError(f"v2 checkpoint is missing multi_robot_architecture: {path}")
        expected = self._multi_robot_architecture_metadata()
        for key, expected_value in expected.items():
            if key not in received:
                raise ValueError(f"Checkpoint metadata is missing {key!r}: {path}")
            received_value = received[key]
            if isinstance(expected_value, float):
                matches = abs(float(received_value) - expected_value) <= 1e-12
            else:
                matches = received_value == expected_value
            if not matches:
                raise ValueError(
                    f"Checkpoint architecture mismatch for {key}: "
                    f"expected {expected_value!r}, got {received_value!r} in {path}"
                )
        unexpected_metadata = sorted(set(received) - set(expected))
        if unexpected_metadata:
            raise ValueError(
                "Checkpoint architecture contains unexpected metadata keys: "
                f"{unexpected_metadata} in {path}"
            )

    def save_checkpoint(
        self,
        path,
        optimizer=None,
        step=None,
        checkpoint_state_kind: str | None = None,
        checkpoint_integrity_mode: str | None = None,
    ):
        del optimizer
        active_integrity_mode = self._active_checkpoint_integrity_mode()
        if checkpoint_integrity_mode is not None:
            requested_integrity_mode = self._validated_checkpoint_integrity_mode(
                checkpoint_integrity_mode
            )
            if requested_integrity_mode != active_integrity_mode:
                raise ValueError(
                    "Checkpoint publication integrity mode must match the model: "
                    f"requested={requested_integrity_mode!r} "
                    f"model={active_integrity_mode!r}"
                )
        if checkpoint_state_kind is None or checkpoint_state_kind == "auto":
            checkpoint_state_kind = (
                "full" if self._trainable_scope == "dit" else "sparse_delta"
            )
        checkpoint_state_kind = str(checkpoint_state_kind).strip().lower()
        if checkpoint_state_kind not in {"full", "sparse_delta"}:
            raise ValueError(
                "checkpoint_state_kind must be 'full' or 'sparse_delta', got "
                f"{checkpoint_state_kind!r}"
            )
        if checkpoint_state_kind == "sparse_delta" and self._trainable_scope == "dit":
            raise ValueError(
                "checkpoint_state_kind='sparse_delta' is invalid when "
                "trainable_scope='dit'; save a self-contained full checkpoint."
            )
        payload: dict[str, Any] = {
            "format": "fastwam_multi_robot_v2",
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "trainable_scope": self._trainable_scope,
            "training_mode": self.training_mode,
            "multi_robot_architecture": self._multi_robot_architecture_metadata(),
        }
        if checkpoint_state_kind == "full":
            payload["state_kind"] = "full"
            payload["base_checkpoint"] = None
            payload["mot"] = self.mot.state_dict()
        else:
            trainable_keys = self._expected_trainable_parameter_names()
            payload["state_kind"] = "sparse_delta"
            payload["base_checkpoint"] = self._base_dependency_descriptor_for_save(path)
            payload["trainable_parameter_names"] = trainable_keys
            payload["mot_trainable"] = {
                name: value.detach().cpu()
                for name, value in self.mot.state_dict().items()
                if name in trainable_keys
            }
        torch.save(payload, path)

    def load_checkpoint(
        self,
        path,
        optimizer=None,
        *,
        validate_trainable_scope: bool = True,
    ):
        """Load a checkpoint with strict architecture/treatment validation.

        ``trainable_scope`` is training provenance, not an inference-time
        architecture choice.  Training/resume keeps the strict default;
        inference may disable only that comparison for self-contained full
        checkpoints while all tensor, architecture and training-mode checks
        remain active.
        """
        return self._load_checkpoint_with_role(
            path,
            optimizer=optimizer,
            load_role="top_level",
            active_paths=set(),
            validate_treatment=True,
            validate_trainable_scope=validate_trainable_scope,
            require_full_top_level=False,
        )

    def load_initialization_checkpoint(
        self,
        path,
        *,
        expected_sha256: str | None = None,
        checkpoint_integrity_mode: str = "sha256",
    ):
        """Warm-start from an exact native full checkpoint across treatments.

        Architecture and tensor identity stay strict. Only training-mode and
        trainable-scope provenance may differ; optimizer/scheduler/step state is
        deliberately outside this weights-only operation.
        """

        requested_integrity_mode = self._validated_checkpoint_integrity_mode(
            checkpoint_integrity_mode
        )
        active_integrity_mode = self._active_checkpoint_integrity_mode()
        if requested_integrity_mode != active_integrity_mode:
            raise ValueError(
                "Initialization checkpoint integrity mode must match the model: "
                f"requested={requested_integrity_mode!r} "
                f"model={active_integrity_mode!r}"
            )
        return self._load_checkpoint_with_role(
            path,
            optimizer=None,
            load_role="top_level",
            active_paths=set(),
            validate_treatment=False,
            validate_trainable_scope=False,
            require_full_top_level=True,
            expected_sha256=expected_sha256,
        )

    def _load_checkpoint_with_role(
        self,
        path,
        *,
        optimizer=None,
        load_role: str,
        active_paths: set[Path],
        validate_treatment: bool = True,
        validate_trainable_scope: bool = True,
        require_full_top_level: bool = False,
        expected_sha256: str | None = None,
        expected_metadata: dict[str, Any] | None = None,
    ):
        checkpoint_path = self._absolute_checkpoint_path(path)
        if checkpoint_path in active_paths:
            raise ValueError(
                f"Checkpoint base dependency cycle detected at {checkpoint_path}"
            )
        active_paths = set(active_paths)
        active_paths.add(checkpoint_path)

        digest_label = (
            "Base checkpoint"
            if load_role == "base_dependency"
            else "Initialization checkpoint"
            if require_full_top_level
            else "Checkpoint"
        )
        payload, checkpoint_evidence = self._load_pinned_checkpoint_payload(
            checkpoint_path,
            expected_sha256=expected_sha256,
            digest_label=digest_label,
            integrity_mode=self._active_checkpoint_integrity_mode(),
            expected_metadata=expected_metadata,
        )

        checkpoint_format = payload.get("format")
        if require_full_top_level and checkpoint_format != "fastwam_multi_robot_v2":
            raise ValueError(
                "Initialization requires a self-contained native v2 full checkpoint: "
                f"{checkpoint_path}"
            )
        if checkpoint_format == "fastwam_multi_robot_v2":
            self._validate_multi_robot_checkpoint_metadata(
                payload,
                checkpoint_path,
                validate_treatment=validate_treatment and load_role == "top_level",
                validate_trainable_scope=validate_trainable_scope,
            )
            state_kind = self._native_checkpoint_state_kind(payload, checkpoint_path)
            if require_full_top_level and state_kind != "full":
                raise ValueError(
                    "Initialization requires state_kind='full', got "
                    f"{state_kind!r} in {checkpoint_path}"
                )
            if load_role == "base_dependency" and state_kind != "full":
                raise ValueError(
                    "Nested sparse native v2 checkpoints are forbidden; a "
                    f"base_dependency must be state_kind='full': {checkpoint_path}"
                )

            if state_kind == "full":
                mot_state = payload["mot"]
                expected_state = self.mot.state_dict()
                self._validate_exact_tensor_state(
                    expected_state,
                    mot_state,
                    expected_keys=sorted(expected_state),
                    path=checkpoint_path,
                    label="mot",
                )
                self.mot.load_state_dict(mot_state, strict=True)
                if load_role == "top_level":
                    self._remember_loaded_base_checkpoint(
                        checkpoint_path,
                        checkpoint_evidence,
                    )
            else:
                if load_role != "top_level":
                    raise ValueError(
                        f"sparse_delta is only valid as a top-level checkpoint: {checkpoint_path}"
                    )
                trainable_state = payload["mot_trainable"]
                self._validate_sparse_trainable_contract(
                    payload,
                    trainable_state,
                    path=checkpoint_path,
                )
                descriptor = self._validated_base_dependency_descriptor(
                    payload.get("base_checkpoint"),
                    owner_path=checkpoint_path,
                    active_paths=active_paths,
                )
                self._load_checkpoint_with_role(
                    descriptor["path"],
                    optimizer=None,
                    load_role="base_dependency",
                    active_paths=active_paths,
                    validate_treatment=False,
                    validate_trainable_scope=validate_trainable_scope,
                    require_full_top_level=False,
                    expected_sha256=descriptor.get("sha256"),
                    expected_metadata=(
                        descriptor
                        if descriptor.get("integrity_mode") == "metadata_no_hash"
                        else None
                    ),
                )
                result = self.mot.load_state_dict(trainable_state, strict=False)
                if result.unexpected_keys:
                    raise RuntimeError(
                        "Validated sparse state produced unexpected keys during load: "
                        f"{result.unexpected_keys}"
                    )
                self._loaded_base_checkpoint = descriptor["path"]
                self._loaded_base_checkpoint_sha256 = descriptor.get("sha256")
                self._loaded_base_checkpoint_descriptor = descriptor
                self._loaded_base_checkpoint_can_restore_sparse = True
        elif checkpoint_format is not None:
            raise ValueError(
                f"Unsupported checkpoint format {checkpoint_format!r}: {checkpoint_path}"
            )
        elif "mot" in payload:
            mot_state = self._upgrade_legacy_hub_state(payload["mot"])
            self._validate_legacy_minimum_coverage(
                mot_state,
                path=checkpoint_path,
                label="mot",
                load_role=load_role,
            )
            self._load_matching_state(self.mot, mot_state, label="legacy mot")
            if load_role == "top_level":
                self._remember_loaded_base_checkpoint(
                    checkpoint_path,
                    checkpoint_evidence,
                )
        elif "dit" in payload:
            self._validate_legacy_minimum_coverage(
                payload["dit"],
                path=checkpoint_path,
                label="dit",
                load_role=load_role,
            )
            self._load_matching_state(
                self.video_expert,
                payload["dit"],
                label="legacy video dit",
            )
            if load_role == "top_level":
                self._loaded_base_checkpoint = str(checkpoint_path)
                self._loaded_base_checkpoint_sha256 = None
                self._loaded_base_checkpoint_descriptor = None
                self._loaded_base_checkpoint_can_restore_sparse = False
        else:
            raise ValueError(
                f"Legacy checkpoint missing both `mot` and `dit`: {checkpoint_path}"
            )

        if load_role == "top_level" and optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
