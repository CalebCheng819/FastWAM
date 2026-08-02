import hashlib
import json
import logging
import inspect
import os
import re
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import (
    ResumableAgentCountBatchSampler,
    ResumableEpochSampler,
    resolve_agent_counts,
    resolve_task_ids,
)
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        configured_token_budget = cfg.get("agent_action_token_budget", None)
        self.agent_action_token_budget = (
            None
            if configured_token_budget in (None, "", "null")
            else int(configured_token_budget)
        )
        if self.agent_action_token_budget is not None and self.agent_action_token_budget <= 0:
            raise ValueError(
                "`agent_action_token_budget` must be positive when enabled, "
                f"got {self.agent_action_token_budget}"
            )
        
        self.resume = cfg.resume
        self.trainable_scope = str(cfg.get("trainable_scope", "dit")).strip().lower()
        self.save_training_state_enabled = bool(cfg.get("save_training_state", True))
        self.save_final_checkpoint_enabled = bool(cfg.get("save_final_checkpoint", True))
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
            dataloader_config=DataLoaderConfiguration(
                # Dynamic token-budget batches intentionally have different
                # sample counts. Their source schedule is explicitly aligned
                # across ranks, so Accelerate must shard without tail filling.
                even_batches=self.agent_action_token_budget is None,
            ),
        )
        self._configure_dynamic_deepspeed_batch_accounting()
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        trainable_params = self._apply_dit_only_train_mode(
            self.model,
            trainable_scope=self.trainable_scope,
        )
        trainable_count = sum(parameter.numel() for parameter in trainable_params)
        logger.info(
            "Selected trainable_scope=%s with %.3f M parameters.",
            self.trainable_scope,
            trainable_count / 1e6,
        )
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _configure_dynamic_deepspeed_batch_accounting(self) -> None:
        """Bridge variable-size batch samplers to DeepSpeed's fixed metadata.

        Accelerate cannot infer a DeepSpeed micro-batch size from a DataLoader
        backed by ``batch_sampler`` because its public ``batch_size`` is None.
        The integer below is only used for DeepSpeed's global-batch accounting;
        the sampler continues to emit native N-dependent batches.
        """

        if self.agent_action_token_budget is None:
            return
        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if plugin is None:
            return
        deepspeed_config = plugin.deepspeed_config
        configured = deepspeed_config.get("train_micro_batch_size_per_gpu", "auto")
        if configured == "auto":
            configured = int(self.batch_size)
            deepspeed_config["train_micro_batch_size_per_gpu"] = configured
        if isinstance(configured, bool) or not isinstance(configured, int) or configured < 1:
            raise ValueError(
                "Dynamic token-budget batching requires DeepSpeed "
                "train_micro_batch_size_per_gpu to be a positive integer after "
                f"resolution, got {configured!r}."
            )
        logger.info(
            "DeepSpeed dynamic-batch accounting: nominal_micro_batch=%d "
            "(real sampler batch sizes vary by agent count).",
            configured,
        )

    def _forward_training_loss(self, sample):
        """Run loss computation through the prepared model wrapper.

        In DeepSpeed mode ``self.model`` is a ``DeepSpeedEngine``. Calling a
        method exposed through ``__getattr__`` would bypass the engine's
        ``forward`` hooks, including its gradient-accumulation loss scaling.
        FastWAM's ``forward`` delegates to ``training_loss``, so invoking the
        wrapper preserves the result while keeping runtime hooks active.
        """

        return self.model(sample)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        agent_counts = resolve_agent_counts(dataset)
        if agent_counts is not None:
            task_ids = resolve_task_ids(dataset)
            if task_ids is None:
                raise TypeError(
                    "Variable-agent datasets must expose stable `task_ids` or "
                    "`get_task_id(index)` for hierarchical balanced sampling."
                )
            self._uses_agent_count_batch_sampler = True
            self.train_sampler = ResumableAgentCountBatchSampler(
                dataset=dataset,
                seed=self.seed,
                batch_size=self.batch_size,
                num_processes=self.accelerator.num_processes,
                agent_counts=agent_counts,
                task_ids=task_ids,
                action_horizon=getattr(dataset, "action_horizon", None),
                agent_action_token_budget=self.agent_action_token_budget,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
            )
            logger.info(
                "Using hierarchical task/count-balanced batching: counts=%s tasks_by_count=%s "
                "batch_sizes=%s token_budget=%s global_batches=%d local_microbatches=%d "
                "optimizer_steps=%d schedule_sha256=%s",
                self.train_sampler.observed_agent_counts,
                self.train_sampler.tasks_by_agent_count,
                self.train_sampler.batch_size_by_agent_count,
                self.train_sampler.agent_action_token_budget,
                self.train_sampler.global_batches_per_epoch,
                self.train_sampler.microbatches_per_process,
                self.train_sampler.optimizer_steps_per_epoch,
                self.train_sampler.schedule_fingerprint(),
            )
            return DataLoader(
                dataset,
                batch_sampler=self.train_sampler,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=worker_init_fn,
            )

        if self.agent_action_token_budget is not None:
            raise ValueError(
                "`agent_action_token_budget` is enabled, but the dataset does not expose "
                "agent-count metadata."
            )
        self._uses_agent_count_batch_sampler = False
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        if self._uses_agent_count_batch_sampler:
            return max(self.train_sampler.optimizer_steps_per_epoch * self.num_epochs, 1)
        else:
            global_batch_size = max(self.batch_size * num_processes, 1)
            micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _set_train_data_epoch(self, epoch: int):
        """Synchronize the source sampler and Accelerate's prepared loader."""

        epoch = int(epoch)
        self.train_sampler.set_epoch(epoch)
        train_loader = getattr(self, "train_loader", None)
        if train_loader is not None and hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    @staticmethod
    def _sample_work_counts(sample) -> tuple[int, int]:
        """Return real samples and agent-action tokens in one local micro-batch."""

        action = sample.get("action") if isinstance(sample, dict) else None
        if isinstance(action, torch.Tensor):
            if action.ndim == 4:
                batch_size, num_agents, horizon = action.shape[:3]
                return int(batch_size), int(batch_size * num_agents * horizon)
            if action.ndim == 3:
                batch_size, horizon = action.shape[:2]
                return int(batch_size), int(batch_size * horizon)

        if isinstance(sample, dict):
            for value in sample.values():
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    return int(value.shape[0]), 0
        return 0, 0

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self):
        logger.info("Applying trainable scope %s and freezing all other components.", self.trainable_scope)
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model, trainable_scope=self.trainable_scope)

    @staticmethod
    def _apply_dit_only_train_mode(model, trainable_scope: str = "dit"):
        if hasattr(model, "configure_trainable_parameters"):
            return list(model.configure_trainable_parameters(trainable_scope))
        if trainable_scope != "dit":
            raise ValueError(
                f"Model {type(model).__name__} does not implement trainable_scope={trainable_scope!r}."
            )
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        trainable_params = list(model.dit.parameters())
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
            trainable_params.extend(list(proprio_encoder.parameters()))
        return trainable_params

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "resume_compatibility": self._resume_compatibility_metadata(),
        }
        if self._uses_agent_count_batch_sampler:
            payload["data_schedule"] = {
                "fingerprint": self.train_sampler.schedule_fingerprint(self.epoch),
                "agent_action_token_budget": self.train_sampler.agent_action_token_budget,
                "gradient_accumulation_steps": self.train_sampler.gradient_accumulation_steps,
                "num_processes": self.train_sampler.num_processes,
                "global_batches_per_epoch": self.train_sampler.global_batches_per_epoch,
                "optimizer_steps_per_epoch": self.train_sampler.optimizer_steps_per_epoch,
            }
        temporary_state_file = f"{state_file}.tmp"
        with open(temporary_state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_state_file, state_file)

    @staticmethod
    def _resolved_resume_config_sha256(cfg) -> str:
        """Hash the resolved experiment config, excluding relocation fields.

        ``resume`` necessarily changes from the initial weight checkpoint to
        the saved ZeRO state directory, and ``output_dir`` may be relocated for
        recovery.  Every other resolved field remains part of the fail-closed
        resume identity.
        """

        if OmegaConf.is_config(cfg):
            payload = OmegaConf.to_container(cfg, resolve=True)
        elif isinstance(cfg, dict):
            payload = dict(cfg)
        else:
            raise TypeError(
                "Trainer resume compatibility requires a DictConfig or dict, "
                f"got {type(cfg).__name__}."
            )
        if not isinstance(payload, dict):
            raise TypeError("Resolved trainer config must be a mapping.")
        payload = dict(payload)
        payload.pop("resume", None)
        payload.pop("output_dir", None)
        canonical = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _resume_compatibility_metadata(self) -> dict:
        model = self.accelerator.unwrap_model(self.model)
        architecture_getter = getattr(
            model, "_multi_robot_architecture_metadata", None
        )
        architecture = (
            architecture_getter() if callable(architecture_getter) else None
        )
        return {
            "schema_version": 1,
            "resolved_config_hash_scheme": (
                "omegaconf_resolved_without_resume_or_output_dir_v1"
            ),
            "resolved_config_sha256": self._resolved_resume_config_sha256(
                self.cfg
            ),
            "multi_robot_architecture": architecture,
        }

    def _validate_training_state_resume_compatibility(
        self, payload: dict, state_file: Path
    ) -> None:
        current = self._resume_compatibility_metadata()
        saved = payload.get("resume_compatibility")
        multi_robot_resume = current["multi_robot_architecture"] is not None
        if not isinstance(saved, dict):
            if multi_robot_resume:
                raise RuntimeError(
                    "Refusing multi-robot full-state resume without "
                    f"resume_compatibility metadata: {state_file}"
                )
            logger.warning(
                "Trainer state predates resume-compatibility metadata; "
                "model/config identity cannot be verified: %s",
                state_file,
            )
            return

        if saved.get("schema_version") != current["schema_version"]:
            raise RuntimeError(
                "Unsupported trainer resume-compatibility schema: "
                f"expected {current['schema_version']!r}, got "
                f"{saved.get('schema_version')!r} in {state_file}"
            )
        saved_architecture = saved.get("multi_robot_architecture")
        current_architecture = current["multi_robot_architecture"]
        if saved_architecture != current_architecture:
            all_keys = sorted(
                set(saved_architecture or {}) | set(current_architecture or {})
            )
            mismatches = {
                key: (
                    (saved_architecture or {}).get(key),
                    (current_architecture or {}).get(key),
                )
                for key in all_keys
                if (saved_architecture or {}).get(key)
                != (current_architecture or {}).get(key)
            }
            raise RuntimeError(
                "Cannot resume training state with a different multi-robot "
                f"architecture/treatment: {mismatches}"
            )
        if (
            saved.get("resolved_config_hash_scheme")
            != current["resolved_config_hash_scheme"]
            or saved.get("resolved_config_sha256")
            != current["resolved_config_sha256"]
        ):
            raise RuntimeError(
                "Cannot resume training state with a different resolved config: "
                f"saved_sha256={saved.get('resolved_config_sha256')!r}, "
                f"current_sha256={current['resolved_config_sha256']!r}"
            )

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = None
        if self.save_training_state_enabled:
            state_path = os.path.join(self.state_dir, step_tag)
            ensure_dir(state_path)
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def _should_save_final_checkpoint(self, *, checkpoint_saved_this_step: bool) -> bool:
        """Return whether the terminal step still needs a checkpoint write."""
        return self.save_final_checkpoint_enabled and not checkpoint_saved_this_step

    def load_training_state(self, state_dir: str):
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._validate_training_state_resume_compatibility(
                payload, state_file
            )
            # Validation must precede Accelerate/DeepSpeed mutation: ACV0 and
            # ACV1 have shape-compatible state_dicts but different semantics.
            self.accelerator.load_state(input_dir=state_dir)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self._set_train_data_epoch(self.epoch)
                if self._uses_agent_count_batch_sampler:
                    saved_schedule = payload.get("data_schedule")
                    if saved_schedule is None:
                        logger.warning(
                            "Trainer state predates data-schedule fingerprints; "
                            "resume compatibility cannot be verified."
                        )
                    else:
                        current_schedule = {
                            "fingerprint": self.train_sampler.schedule_fingerprint(self.epoch),
                            "agent_action_token_budget": self.train_sampler.agent_action_token_budget,
                            "gradient_accumulation_steps": self.train_sampler.gradient_accumulation_steps,
                            "num_processes": self.train_sampler.num_processes,
                            "global_batches_per_epoch": self.train_sampler.global_batches_per_epoch,
                            "optimizer_steps_per_epoch": self.train_sampler.optimizer_steps_per_epoch,
                        }
                        mismatches = {
                            key: (saved_schedule.get(key), current_value)
                            for key, current_value in current_schedule.items()
                            if saved_schedule.get(key) != current_value
                        }
                        if mismatches:
                            raise RuntimeError(
                                "Cannot resume with a different deterministic data schedule: "
                                f"{mismatches}"
                            )
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                if self._uses_agent_count_batch_sampler:
                    logger.info(
                        "Restored bucket-dataloader progress: epoch=%d batch_in_epoch=%d "
                        "global_batch_offset=%d",
                        self.epoch,
                        self.batch_in_epoch,
                        self.train_sampler.resume_global_batch_offset,
                    )
                else:
                    logger.info(
                        "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                        self.epoch,
                        self.batch_in_epoch,
                        self.train_sampler.resume_sample_offset,
                    )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self._set_train_data_epoch(self.epoch)
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        current_compatibility = self._resume_compatibility_metadata()
        if current_compatibility["multi_robot_architecture"] is not None:
            raise RuntimeError(
                "Refusing multi-robot full-state resume because trainer_state.json "
                f"is missing: {state_dir}"
            )
        self.accelerator.load_state(input_dir=state_dir)

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self._set_train_data_epoch(self.epoch)
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        self._set_train_data_epoch(self.epoch)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        self.run_local_samples = 0
        self.run_local_agent_action_tokens = 0

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
                local_samples, local_agent_action_tokens = self._sample_work_counts(sample)
                self.run_local_samples += local_samples
                self.run_local_agent_action_tokens += local_agent_action_tokens
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                self._set_train_data_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                with self.accelerator.autocast():
                    loss, loss_dict = self._forward_training_loss(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    should_log = self.log_every > 0 and self.global_step % self.log_every == 0
                    if should_log:
                        work_counts = torch.tensor(
                            [self.run_local_samples, self.run_local_agent_action_tokens],
                            device=loss.device,
                            dtype=torch.float64,
                        )
                        global_work_counts = self.accelerator.gather(work_counts).reshape(-1, 2).sum(dim=0)
                        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
                        samples_per_sec = float(global_work_counts[0].item() / elapsed)
                        agent_action_tokens_per_sec = float(global_work_counts[1].item() / elapsed)

                    if should_log and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += (
                            "grad_norm=%.4f lr=%.2e speed=%.2f step/s, %.2f samples/s, "
                            "%.2f real_agent_action_tokens/s eta=%s"
                        ) % (
                            global_grad_norm,
                            current_lr,
                            steps_per_sec,
                            samples_per_sec,
                            agent_action_tokens_per_sec,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": samples_per_sec,
                            "performance/real_agent_action_tokens_per_sec": agent_action_tokens_per_sec,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    checkpoint_saved_this_step = False
                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        checkpoint_saved_this_step = True
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        if self._should_save_final_checkpoint(
                            checkpoint_saved_this_step=checkpoint_saved_this_step
                        ):
                            ckpt_info = self.save_checkpoint()
                            checkpoint_saved_this_step = True
                        if self.accelerator.is_main_process and checkpoint_saved_this_step:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        if not self.save_final_checkpoint_enabled:
            return
        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
