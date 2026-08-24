"""Inverse-dynamics training for synchronized multi-robot FastWAM."""

from __future__ import annotations

from typing import Any

import torch

from .fastwam_multi_robot import FastWAMMultiRobot


class FastWAMMultiRobotIDM(FastWAMMultiRobot):
    """Joint world loss plus action denoising conditioned on teacher video.

    The noisy-video branch retains the normal video flow-matching objective.
    A second, independently teacher-forced video branch supplies the K/V cache
    used by the factorized multi-agent action expert.  That makes the action
    objective an inverse-dynamics objective instead of merely relabeling the
    standard joint FastWAM path.
    """

    def __init__(
        self,
        *args,
        video_cond_noise_prob: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.training_mode != "joint":
            raise ValueError("FastWAMMultiRobotIDM requires training_mode='joint'")
        probability = float(video_cond_noise_prob)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("video_cond_noise_prob must be in [0, 1]")
        self.video_cond_noise_prob = probability

    def _training_loss_joint(self, inputs: dict[str, Any]):
        input_latents = inputs["input_latents"]
        batch_size = int(input_latents.shape[0])

        # Preserve the multi-robot A/B evaluator's action RNG substream.
        noisy_action, target_action, timestep_action = self._prepare_noisy_action(inputs)

        # Branch A: ordinary noisy video for the world-model objective.
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

        # Branch B: per-sample clean/noised teacher video for inverse dynamics.
        cond_noise_mask = (
            torch.rand((batch_size,), device=self.device)
            < self.video_cond_noise_prob
        )
        timestep_video_cond = torch.zeros_like(
            timestep_video,
            dtype=input_latents.dtype,
            device=self.device,
        )
        cond_video = input_latents
        if bool(cond_noise_mask.any()):
            sampled_cond_timestep = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=input_latents.dtype,
            )
            cond_noise = torch.randn_like(input_latents)
            noised_cond_video = self.train_video_scheduler.add_noise(
                input_latents, cond_noise, sampled_cond_timestep
            )
            selector = cond_noise_mask.view(batch_size, 1, 1, 1, 1)
            cond_video = torch.where(selector, noised_cond_video, input_latents)
            timestep_video_cond = torch.where(
                cond_noise_mask,
                sampled_cond_timestep,
                timestep_video_cond,
            )
        cond_video = cond_video.clone()
        cond_video[:, :, :1] = inputs["first_frame_latents"]

        video_pre_noisy = self.video_expert.pre_dit(
            x=noisy_video,
            timestep=timestep_video,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        video_pre_cond = self.video_expert.pre_dit(
            x=cond_video,
            timestep=timestep_video_cond,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self._action_pre(noisy_action, timestep_action, inputs)

        noisy_video_seq_len = int(video_pre_noisy["tokens"].shape[1])
        cond_video_seq_len = int(video_pre_cond["tokens"].shape[1])
        noisy_tokens_per_frame = int(video_pre_noisy["meta"]["tokens_per_frame"])
        cond_tokens_per_frame = int(video_pre_cond["meta"]["tokens_per_frame"])
        if noisy_video_seq_len != cond_video_seq_len:
            raise ValueError("IDM noisy and conditioning video sequence lengths must match")
        if noisy_tokens_per_frame != cond_tokens_per_frame:
            raise ValueError("IDM noisy and conditioning video frame layouts must match")

        attention_layout = self._multi_robot_attention_layout(
            video_seq_len=cond_video_seq_len,
            video_tokens_per_frame=cond_tokens_per_frame,
            action_pre=action_pre,
        )
        noisy_video_attention_mask = self._build_video_attention_mask(
            video_seq_len=noisy_video_seq_len,
            video_tokens_per_frame=noisy_tokens_per_frame,
            device=video_pre_noisy["tokens"].device,
        )
        cond_video_attention_mask = self._build_video_attention_mask(
            video_seq_len=cond_video_seq_len,
            video_tokens_per_frame=cond_tokens_per_frame,
            device=video_pre_cond["tokens"].device,
        )

        noisy_video_tokens, _ = self.mot.forward_video_with_cache(
            video_tokens=video_pre_noisy["tokens"],
            video_freqs=video_pre_noisy["freqs"],
            video_t_mod=video_pre_noisy["t_mod"],
            video_context_payload={
                "context": video_pre_noisy["context"],
                "mask": video_pre_noisy["context_mask"],
            },
            video_attention_mask=noisy_video_attention_mask,
        )
        _, cond_video_kv_cache = self.mot.forward_video_with_cache(
            video_tokens=video_pre_cond["tokens"],
            video_freqs=video_pre_cond["freqs"],
            video_t_mod=video_pre_cond["t_mod"],
            video_context_payload={
                "context": video_pre_cond["context"],
                "mask": video_pre_cond["context_mask"],
            },
            video_attention_mask=cond_video_attention_mask,
        )
        action_tokens = self.mot.forward_multi_agent_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=cond_video_kv_cache,
            **attention_layout,
        )

        pred_video = self.video_expert.post_dit(
            noisy_video_tokens, video_pre_noisy
        )[:, :, 1:]
        pred_action = self.action_expert.post_dit(action_tokens, action_pre)
        target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=inputs["image_is_pad"],
            include_initial_video_step=False,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        loss_video = (loss_video_per_sample * video_weight).mean()
        loss_action, b4_metrics = self._multi_action_objective(
            inputs=inputs,
            noisy_action=noisy_action,
            pred_action=pred_action,
            target_action=target_action,
            timestep_action=timestep_action,
        )
        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
        )
        metrics = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "idm_cond_noised_fraction": float(
                cond_noise_mask.float().mean().detach().item()
            ),
        }
        if b4_metrics:
            metrics.update(
                {
                    "loss_action_flow": self.loss_lambda_action
                    * float(b4_metrics["flow"].detach().item()),
                    "loss_b4_arm_huber": self.loss_lambda_action
                    * float(b4_metrics["arm_huber"].detach().item()),
                    "loss_b4_gripper_event": self.loss_lambda_action
                    * float(b4_metrics["gripper_event"].detach().item()),
                    "loss_b4_contact_intent_proxy": self.loss_lambda_action
                    * float(b4_metrics["contact_intent_proxy"].detach().item()),
                }
            )
        return loss_total, metrics
