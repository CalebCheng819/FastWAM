import hashlib
import inspect
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from datetime import timedelta
from math import ceil
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration, InitProcessGroupKwargs
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .formal_artifacts import (
    N4_GATE_GLOBAL_TRAIN_BATCH_SIZE,
    N4_GATE_GRADIENT_ACCUMULATION_STEPS,
    N4_GATE_LOCAL_MICRO_BATCH_SIZE,
    N4_GATE_MAX_PEAK_ALLOCATED_BYTES,
    N4_GATE_MAX_PEAK_RESERVED_BYTES,
    N4_GATE_TRAIN_STEPS,
    N4_GATE_WORLD_SIZE,
    canonical_json_sha256,
    next_rng_sample,
    normalize_formal_evaluation_records,
    publish_exclusive_json,
    publish_failure_marker,
    publish_training_terminal_seal,
    read_canonical_json,
    state_fingerprints,
)
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
        self.offline_eval_num_samples = int(cfg.get("offline_eval_num_samples", 0))
        if self.offline_eval_num_samples < 0:
            raise ValueError(
                "`offline_eval_num_samples` must be non-negative, got "
                f"{self.offline_eval_num_samples}"
            )
        if (
            self.eval_every > 0
            and hasattr(model, "infer_action_multi")
            and self.offline_eval_num_samples == 0
        ):
            raise ValueError(
                "Multi-robot eval_every>0 requires offline_eval_num_samples>0. "
                "This trainer reports held-out training loss only; simulator success "
                "must be evaluated by a separate rollout job."
            )
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
        self.phase_balanced_fraction = float(
            cfg.get("phase_balanced_fraction", 0.0)
        )
        if self.phase_balanced_fraction not in (0.0, 0.5):
            raise ValueError(
                "`phase_balanced_fraction` currently supports only 0.0 or "
                "the audited B4 50/50 mixture (0.5), got "
                f"{self.phase_balanced_fraction}"
            )

        self.resume = cfg.resume
        self.run_initial_global_step = int(cfg.get("run_initial_global_step", 0))
        if self.run_initial_global_step < 0:
            raise ValueError(
                "`run_initial_global_step` must be non-negative, got "
                f"{self.run_initial_global_step}"
            )
        self.trainable_scope = str(cfg.get("trainable_scope", "dit")).strip().lower()
        requested_checkpoint_state_kind = str(
            cfg.get("checkpoint_state_kind", "auto")
        ).strip().lower()
        if requested_checkpoint_state_kind not in {"auto", "full", "sparse_delta"}:
            raise ValueError(
                "`checkpoint_state_kind` must be one of "
                "['auto', 'full', 'sparse_delta'], got "
                f"{requested_checkpoint_state_kind!r}"
            )
        self.checkpoint_state_kind = (
            "full"
            if requested_checkpoint_state_kind == "auto" and self.trainable_scope == "dit"
            else "sparse_delta"
            if requested_checkpoint_state_kind == "auto"
            else requested_checkpoint_state_kind
        )
        if self.checkpoint_state_kind == "sparse_delta" and self.trainable_scope == "dit":
            raise ValueError(
                "checkpoint_state_kind='sparse_delta' is invalid when "
                "trainable_scope='dit'"
            )
        warm_start_cfg = cfg.get("weights_only_warm_start", {})
        self.weights_only_warm_start_enabled = bool(
            warm_start_cfg.get("enabled", False)
        )
        self.weights_only_warm_start_expected_source_training_mode = (
            None
            if warm_start_cfg.get("expected_source_training_mode") is None
            else str(
                warm_start_cfg.get("expected_source_training_mode")
            ).strip().lower()
        )
        self.weights_only_warm_start_expected_source_trainable_scope = (
            None
            if warm_start_cfg.get("expected_source_trainable_scope") is None
            else str(
                warm_start_cfg.get("expected_source_trainable_scope")
            ).strip().lower()
        )
        self.weights_only_warm_start_expected_source_state_kind = str(
            warm_start_cfg.get("expected_source_state_kind", "full")
        ).strip().lower()
        if self.weights_only_warm_start_enabled:
            if self.weights_only_warm_start_expected_source_training_mode not in {
                "action_only_cache",
                "joint",
            }:
                raise ValueError(
                    "Enabled weights_only_warm_start requires an exact "
                    "expected_source_training_mode of 'joint' or "
                    "'action_only_cache', got "
                    f"{self.weights_only_warm_start_expected_source_training_mode!r}"
                )
            if self.weights_only_warm_start_expected_source_trainable_scope not in {
                "hub_io",
                "action",
                "dit",
            }:
                raise ValueError(
                    "Enabled weights_only_warm_start requires an exact "
                    "expected_source_trainable_scope of 'hub_io', 'action', or "
                    "'dit', got "
                    f"{self.weights_only_warm_start_expected_source_trainable_scope!r}"
                )
            if self.weights_only_warm_start_expected_source_state_kind != "full":
                raise ValueError(
                    "weights_only_warm_start only accepts a self-contained native-v2 "
                    "source with expected_source_state_kind='full', got "
                    f"{self.weights_only_warm_start_expected_source_state_kind!r}"
                )
            if self.checkpoint_state_kind != "full":
                raise ValueError(
                    "weights_only_warm_start requires checkpoint_state_kind='full' "
                    "so the new treatment never publishes a base-dependent delta"
                )
        if self.run_initial_global_step > 0:
            if not self.weights_only_warm_start_enabled:
                raise ValueError(
                    "run_initial_global_step>0 is only supported for an explicit "
                    "weights_only_warm_start continuation"
                )
            if not self.resume:
                raise ValueError(
                    "run_initial_global_step>0 requires a weights-only resume checkpoint"
                )
        self.allow_legacy_resume = bool(cfg.get("allow_legacy_resume", False))
        self.save_training_state_enabled = bool(cfg.get("save_training_state", True))
        self.seal_training_state = bool(cfg.get("seal_training_state", False))
        self.save_final_checkpoint_enabled = bool(cfg.get("save_final_checkpoint", True))
        self.seal_training_run = bool(cfg.get("seal_training_run", False))
        self.terminal_rehash_weights = bool(cfg.get("terminal_rehash_weights", True))
        self.provenance_mode = str(
            cfg.get("provenance_mode", "sha256")
        ).strip().lower()
        if self.provenance_mode not in {"sha256", "stat_cmp"}:
            raise ValueError(
                "`provenance_mode` must be one of ['sha256', 'stat_cmp'], got "
                f"{self.provenance_mode!r}"
            )
        if self.provenance_mode == "stat_cmp":
            incompatible_seals = {
                "seal_training_state": self.seal_training_state,
                "seal_training_run": self.seal_training_run,
                "terminal_rehash_weights": self.terminal_rehash_weights,
            }
            enabled_seals = [
                name for name, enabled in incompatible_seals.items() if enabled
            ]
            if enabled_seals:
                raise ValueError(
                    "provenance_mode='stat_cmp' cannot enable hash-based seals: "
                    f"{enabled_seals}"
                )
            if self.checkpoint_state_kind != "full":
                raise ValueError(
                    "provenance_mode='stat_cmp' requires checkpoint_state_kind='full' "
                    "so resume never depends on a hash-addressed sparse base"
                )
        # The native checkpoint loader runs before Accelerator/DeepSpeed
        # wrapping, so pass the already validated publication policy directly
        # to the model without adding it to the scientific architecture config.
        self.model._checkpoint_provenance_mode = self.provenance_mode
        self.formal_n4_fullmodel_gate = bool(
            cfg.get("formal_n4_fullmodel_gate", False)
        )
        gate_phase = os.environ.get("FASTWAM_N4_FULLMODEL_GATE_PHASE", "").strip().lower()
        if self.formal_n4_fullmodel_gate:
            if gate_phase not in {"save", "load"}:
                raise ValueError(
                    "formal_n4_fullmodel_gate requires "
                    "FASTWAM_N4_FULLMODEL_GATE_PHASE=save or load"
                )
            self.n4_fullmodel_gate_phase = gate_phase
            self._gate_process_nonce = secrets.token_hex(16)
            self._gate_process_pid = os.getpid()
            self._gate_process_start_ticks = self._process_start_ticks()
        else:
            if gate_phase:
                raise ValueError(
                    "FASTWAM_N4_FULLMODEL_GATE_PHASE is forbidden outside the "
                    "committed N=4 gate scale"
                )
            self.n4_fullmodel_gate_phase = None
        self._gate_pre_load_fingerprints = None
        self._last_step_metrics: dict[str, object] = {}
        self._evaluation_records: list[dict[str, object]] = []
        self.process_group_timeout_seconds = int(
            cfg.get("process_group_timeout_seconds", 1800)
        )
        self.checkpoint_io_timeout_seconds = int(
            cfg.get("checkpoint_io_timeout_seconds", 1800)
        )
        if self.process_group_timeout_seconds <= 0:
            raise ValueError(
                "`process_group_timeout_seconds` must be positive, got "
                f"{self.process_group_timeout_seconds}"
            )
        if self.checkpoint_io_timeout_seconds <= 0:
            raise ValueError(
                "`checkpoint_io_timeout_seconds` must be positive, got "
                f"{self.checkpoint_io_timeout_seconds}"
            )
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
            kwargs_handlers=[
                InitProcessGroupKwargs(
                    timeout=timedelta(seconds=self.process_group_timeout_seconds)
                )
            ],
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

        # Freeze the scientific data identity once.  Recomputing the indexed
        # window hash at every checkpoint is unnecessary and would add a
        # sizeable synchronous metadata pass to formal runs.
        train_data_contract = self._dataset_contract(self.train_dataset)
        val_data_contract = (
            train_data_contract
            if self.val_dataset is self.train_dataset
            else self._dataset_contract(self.val_dataset)
        )
        self._dataset_run_contract = {
            "train": train_data_contract,
            "val": val_data_contract,
        }

        # Resolve the exact trainable set before loading a native sparse v2
        # checkpoint.  The checkpoint loader can then require an exact delta
        # key set for this treatment/scope.  This operation does not consume
        # RNG and remains before optimizer/DeepSpeed construction.
        trainable_params = self._apply_dit_only_train_mode(
            self.model,
            trainable_scope=self.trainable_scope,
        )

        # A weight-only checkpoint must be loaded before the optimizer and
        # DeepSpeed engine are constructed.  ZeRO keeps FP32 master weights;
        # loading only the wrapped BF16 module after ``prepare`` leaves those
        # masters at their random initialization, and the first optimizer step
        # silently copies the stale values back into the model.  Full-state
        # resume remains a post-prepare operation below because Accelerate
        # needs the prepared optimizer/scheduler objects to restore it.
        self._weight_checkpoint_loaded_before_prepare = False
        self._load_weight_checkpoint_before_prepare()

        # Non-trainable modules were frozen before loading so ZeRO sees only
        # the treatment's documented parameter scope.
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
        if self.run_initial_global_step >= total_train_steps:
            raise ValueError(
                "run_initial_global_step must be smaller than max_steps, got "
                f"{self.run_initial_global_step}>={total_train_steps}"
            )
        self.optimizer_steps_this_run = (
            total_train_steps - self.run_initial_global_step
        )
        self.scheduler_warmup_steps = int(self.optimizer_steps_this_run * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=self.optimizer_steps_this_run,
            warmup_steps=self.scheduler_warmup_steps,
        )
        self.global_step = self.run_initial_global_step
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
        self._resume_training_state_after_prepare()

        if self.formal_n4_fullmodel_gate:
            self._validate_n4_fullmodel_gate_contract()

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

    @staticmethod
    def _process_start_ticks() -> int:
        """Disambiguate fresh process worlds even when Linux reuses a PID."""

        fields = Path("/proc/self/stat").read_text(encoding="utf-8").split()
        if len(fields) < 22:
            raise RuntimeError("/proc/self/stat does not contain process start ticks")
        return int(fields[21])

    def _resolved_n4_gate_batch_accounting(self) -> dict[str, int]:
        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if plugin is None:
            raise RuntimeError("N=4 full-model gate requires the DeepSpeed plugin")
        config = plugin.deepspeed_config
        values = {
            "global_train_batch_size": config.get("train_batch_size"),
            "gradient_accumulation_steps": config.get(
                "gradient_accumulation_steps"
            ),
            "local_micro_batch_size": config.get(
                "train_micro_batch_size_per_gpu"
            ),
            "world_size": int(self.accelerator.num_processes),
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"N=4 gate requires a resolved positive integer {name}, got {value!r}"
                )
        expected = {
            "global_train_batch_size": N4_GATE_GLOBAL_TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": N4_GATE_GRADIENT_ACCUMULATION_STEPS,
            "local_micro_batch_size": N4_GATE_LOCAL_MICRO_BATCH_SIZE,
            "world_size": N4_GATE_WORLD_SIZE,
        }
        if values != expected:
            raise ValueError(
                f"N=4 gate DeepSpeed batch accounting mismatch: expected={expected} "
                f"observed={values}"
            )
        zero_stage = config.get("zero_optimization", {}).get("stage")
        if zero_stage != 2:
            raise ValueError(f"N=4 gate requires DeepSpeed ZeRO stage 2, got {zero_stage!r}")
        return values

    def _validate_n4_fullmodel_gate_contract(self) -> None:
        """Refuse any treatment, data, schedule, or topology drift in the gate."""

        if self.seal_training_run:
            raise ValueError(
                "N=4 gate must defer run-level sealing until the fresh load process"
            )
        scalar_contract = {
            "batch_size": (self.batch_size, 1),
            "gradient_accumulation_steps": (self.gradient_accumulation_steps, 1),
            "max_steps": (self.max_steps, 2),
            "save_every": (self.save_every, 2),
            "eval_every": (self.eval_every, 0),
            "offline_eval_num_samples": (self.offline_eval_num_samples, 0),
            "mixed_precision": (self.mixed_precision, "bf16"),
            "checkpoint_state_kind": (self.checkpoint_state_kind, "full"),
            "trainable_scope": (self.trainable_scope, "dit"),
        }
        mismatches = {
            name: {"observed": observed, "expected": expected}
            for name, (observed, expected) in scalar_contract.items()
            if observed != expected
        }
        if mismatches:
            raise ValueError(f"N=4 gate scalar contract mismatch: {mismatches}")
        if not (
            self.save_training_state_enabled
            and self.seal_training_state
            and self.save_final_checkpoint_enabled
        ):
            raise ValueError(
                "N=4 gate requires final full weights, full state, and a sealed state tree"
            )
        if self.agent_action_token_budget != 128:
            raise ValueError(
                "N=4 gate requires agent_action_token_budget=128, got "
                f"{self.agent_action_token_budget!r}"
            )
        if not self._uses_agent_count_batch_sampler:
            raise ValueError("N=4 gate requires the native variable-agent batch sampler")
        observed_counts = [int(value) for value in self.train_sampler.observed_agent_counts]
        if observed_counts != [4]:
            raise ValueError(f"N=4 gate sampler must observe only cardinality 4: {observed_counts}")
        batch_sizes = {
            int(key): int(value)
            for key, value in self.train_sampler.batch_size_by_agent_count.items()
        }
        if batch_sizes != {4: 1}:
            raise ValueError(f"N=4 gate sampler batch-size map mismatch: {batch_sizes}")
        for label, dataset in (("train", self.train_dataset), ("val", self.val_dataset)):
            counts = [int(value) for value in getattr(dataset, "required_agent_counts", [])]
            if counts != [4]:
                raise ValueError(
                    f"N=4 gate {label} dataset must require exactly [4], got {counts}"
                )
            if not getattr(dataset, "load_future_video", False):
                raise ValueError(f"N=4 gate {label} dataset must load future video")
            if getattr(dataset, "gaussian_cache_dir", None) is None:
                raise ValueError(f"N=4 gate {label} dataset must bind compact Gaussian cache")
        model = self.accelerator.unwrap_model(self.model)
        architecture_builder = getattr(model, "_multi_robot_architecture_metadata", None)
        if not callable(architecture_builder):
            raise TypeError("N=4 gate requires FastWAM multi-robot architecture metadata")
        architecture = architecture_builder()
        expected_architecture = {
            "agent_set_representation": "native_variable_length_v1",
            "hub_enabled": True,
            "hub_token_policy": "ceil(hub_token_ratio*num_agents)",
            "hub_token_ratio": 2.0,
            "enable_gaussian": True,
            "gaussian_shape": [13, 28, 40],
        }
        architecture_mismatches = {
            key: {"observed": architecture.get(key), "expected": value}
            for key, value in expected_architecture.items()
            if architecture.get(key) != value
        }
        if architecture_mismatches:
            raise ValueError(
                f"N=4 gate architecture contract mismatch: {architecture_mismatches}"
            )
        action_expert = getattr(model, "action_expert", None)
        if action_expert is None or action_expert.num_hub_tokens_for(4) != 8:
            raise ValueError("N=4 gate requires dynamic K(4)=8 HubTokens")
        if getattr(model, "training_mode", None) != "joint":
            raise ValueError("N=4 gate requires joint VideoGen+action training")
        if self.n4_fullmodel_gate_phase == "save":
            expected_checkpoint_sha256 = os.environ.get(
                "FASTWAM_OFFICIAL_CHECKPOINT_SHA256", ""
            ).strip().lower()
            loaded_checkpoint_sha256 = str(
                getattr(model, "_loaded_base_checkpoint_sha256", "")
            ).strip().lower()
            if (
                len(expected_checkpoint_sha256) != 64
                or loaded_checkpoint_sha256 != expected_checkpoint_sha256
            ):
                raise RuntimeError(
                    "N=4 gate save phase did not load the exact official checkpoint "
                    f"before DeepSpeed prepare: expected={expected_checkpoint_sha256!r} "
                    f"observed={loaded_checkpoint_sha256!r}"
                )
        self._resolved_n4_gate_batch_accounting()

    def _n4_gate_sample_shapes(self, sample) -> dict[str, list[int]]:
        required = {
            "video": [1, 3, 9, 224, 320],
            "action": [1, 4, 32, 8],
            "agent_state": [1, 4, 18],
            "agent_geometry": [1, 4, 7],
            "agent_gaussian": [1, 4, 13, 28, 40],
        }
        observed = {}
        for name, expected in required.items():
            value = sample.get(name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"N=4 gate sample is missing tensor field {name!r}")
            observed[name] = list(value.shape)
            if observed[name] != expected:
                raise ValueError(
                    f"N=4 gate sample shape mismatch for {name}: "
                    f"expected={expected} observed={observed[name]}"
                )
        counts = torch.as_tensor(sample.get("agent_count")).reshape(-1)
        if counts.tolist() != [4]:
            raise ValueError(f"N=4 gate sample must contain one real N=4 item: {counts.tolist()}")
        if sample["agent_gaussian"].dtype != torch.float16:
            raise TypeError(
                "N=4 gate requires FP16 compact Gaussian input, got "
                f"{sample['agent_gaussian'].dtype}"
            )
        return observed

    @staticmethod
    def _n4_gate_losses(loss, loss_dict) -> dict[str, float]:
        if "loss_action" not in loss_dict or "loss_video" not in loss_dict:
            raise RuntimeError(
                "N=4 joint gate requires both loss_action and loss_video metrics"
            )
        losses = {
            "total": float(loss.detach().float().item()),
            "action": float(loss_dict["loss_action"]),
            "video": float(loss_dict["loss_video"]),
        }
        if not all(np.isfinite(value) for value in losses.values()):
            raise RuntimeError(f"N=4 gate produced non-finite losses: {losses}")
        return losses

    def _n4_gate_gradient_evidence(self, grad_norm) -> dict[str, object]:
        norm = float(torch.as_tensor(grad_norm).detach().float().item())
        distributed_type = getattr(self.accelerator, "distributed_type", "")
        distributed_name = str(getattr(distributed_type, "name", distributed_type))
        if distributed_name.rsplit(".", 1)[-1].upper() == "DEEPSPEED":
            # Accelerate 1.12 calls DeepSpeedEngineWrapper.backward(), whose
            # implementation performs engine.step() (including clipping and
            # zero_grad) before returning.  Parameter .grad tensors are
            # therefore intentionally unavailable here.  clip_grad_norm_ in
            # Accelerate's DeepSpeed branch returns engine.get_global_grad_norm(),
            # which is the supported post-step evidence for this path.
            if not np.isfinite(norm) or norm <= 0.0:
                raise RuntimeError(
                    "N=4 gate produced missing/non-positive DeepSpeed global "
                    f"gradient norm: {norm}"
                )
            return {
                "all_finite": True,
                "norm": norm,
                "source": "deepspeed_global_grad_norm",
            }

        tensor_count = 0
        all_finite = True
        for parameter in self.model.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            tensor_count += 1
            if not bool(torch.isfinite(gradient).all().item()):
                all_finite = False
                break
        if tensor_count <= 0 or not all_finite or not np.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(
                "N=4 gate produced missing/non-finite gradients: "
                f"tensor_count={tensor_count} all_finite={all_finite} norm={norm}"
            )
        return {
            "all_finite": True,
            "norm": norm,
            "source": "parameter_grad_scan",
            "tensor_count": tensor_count,
        }

    def _n4_gate_proof_dir(self) -> Path:
        proof_dir = Path(self.output_dir) / "gate-proofs"
        proof_dir.mkdir(parents=True, exist_ok=True)
        if proof_dir.is_symlink() or not proof_dir.is_dir():
            raise ValueError(f"N=4 gate proof root must be a non-symlink directory: {proof_dir}")
        return proof_dir

    def _write_n4_gate_step_proof(
        self,
        *,
        step: int,
        sample_shapes: dict[str, list[int]],
        losses: dict[str, float],
        gradients: dict[str, object],
    ) -> None:
        if not torch.cuda.is_available() or self.accelerator.device.type != "cuda":
            raise RuntimeError("N=4 full-model gate requires CUDA on every rank")
        allocated = int(torch.cuda.max_memory_allocated(self.accelerator.device))
        reserved = int(torch.cuda.max_memory_reserved(self.accelerator.device))
        device_properties = torch.cuda.get_device_properties(self.accelerator.device)
        total_device_bytes = int(device_properties.total_memory)
        allocated_limit = min(
            N4_GATE_MAX_PEAK_ALLOCATED_BYTES,
            total_device_bytes * 90 // 100,
        )
        reserved_limit = min(
            N4_GATE_MAX_PEAK_RESERVED_BYTES,
            total_device_bytes * 95 // 100,
        )
        if allocated > allocated_limit:
            raise RuntimeError(
                f"N=4 gate peak allocated memory exceeded: {allocated} > {allocated_limit} "
                f"for total_device_bytes={total_device_bytes}"
            )
        if reserved > reserved_limit:
            raise RuntimeError(
                f"N=4 gate peak reserved memory exceeded: {reserved} > {reserved_limit} "
                f"for total_device_bytes={total_device_bytes}"
            )
        model = self.accelerator.unwrap_model(self.model)
        architecture = model._multi_robot_architecture_metadata()
        payload = {
            "agent_count": 4,
            "batch_accounting": self._resolved_n4_gate_batch_accounting(),
            "gradients": gradients,
            "hub_token_policy": architecture["hub_token_policy"],
            "losses": losses,
            "memory": {
                "device_name": str(device_properties.name),
                "effective_max_allocated_bytes": allocated_limit,
                "effective_max_reserved_bytes": reserved_limit,
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
                "required_max_allocated_bytes": N4_GATE_MAX_PEAK_ALLOCATED_BYTES,
                "required_max_reserved_bytes": N4_GATE_MAX_PEAK_RESERVED_BYTES,
                "total_device_bytes": total_device_bytes,
            },
            "num_hub_tokens": int(model.action_expert.num_hub_tokens_for(4)),
            "phase": "train_step",
            "process_nonce": self._gate_process_nonce,
            "process_pid": self._gate_process_pid,
            "process_start_ticks": self._gate_process_start_ticks,
            "rank": int(self.accelerator.process_index),
            "sample_shapes": sample_shapes,
            "schema_name": "fastwam-n4-fullmodel-step-proof",
            "schema_version": 1,
            "step": int(step),
            "world_size": int(self.accelerator.num_processes),
        }
        destination = self._n4_gate_proof_dir() / (
            f"step-{int(step):06d}-rank-{self.accelerator.process_index:05d}.json"
        )
        publish_exclusive_json(destination, payload)

    def _n4_gate_state_fingerprints(
        self, *, require_optimizer_state: bool = True
    ) -> dict[str, object]:
        return state_fingerprints(
            model=self.accelerator.unwrap_model(self.model),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            require_optimizer_state=require_optimizer_state,
        )

    def publish_n4_gate_save_proof(self) -> None:
        if not self.formal_n4_fullmodel_gate or self.n4_fullmodel_gate_phase != "save":
            raise RuntimeError("N=4 save proof is only valid in the committed gate save phase")
        if self.global_step != N4_GATE_TRAIN_STEPS:
            raise RuntimeError(
                f"N=4 save proof requires step {N4_GATE_TRAIN_STEPS}, got {self.global_step}"
            )
        fingerprints = self._n4_gate_state_fingerprints()
        payload = {
            "batch_accounting": self._resolved_n4_gate_batch_accounting(),
            "fingerprints": fingerprints,
            "next_rng_sample": next_rng_sample(self.accelerator.device),
            "phase": "save_after_full_checkpoint",
            "process_nonce": self._gate_process_nonce,
            "process_pid": self._gate_process_pid,
            "process_start_ticks": self._gate_process_start_ticks,
            "rank": int(self.accelerator.process_index),
            "schema_name": "fastwam-n4-fullmodel-save-proof",
            "schema_version": 1,
            "world_size": int(self.accelerator.num_processes),
        }
        destination = self._n4_gate_proof_dir() / (
            f"save-state-rank-{self.accelerator.process_index:05d}.json"
        )
        publish_exclusive_json(destination, payload)
        self.accelerator.wait_for_everyone()

    def publish_n4_gate_load_proof(self) -> None:
        if not self.formal_n4_fullmodel_gate or self.n4_fullmodel_gate_phase != "load":
            raise RuntimeError("N=4 load proof is only valid in the committed gate load phase")
        if self.global_step != N4_GATE_TRAIN_STEPS:
            raise RuntimeError(
                f"N=4 load proof requires restored step {N4_GATE_TRAIN_STEPS}, got {self.global_step}"
            )
        if self._gate_pre_load_fingerprints is None:
            raise RuntimeError("N=4 load phase did not capture pre-load state")
        rank = int(self.accelerator.process_index)
        expected_path = self._n4_gate_proof_dir() / f"save-state-rank-{rank:05d}.json"
        expected, _, _ = read_canonical_json(expected_path)
        restored = self._n4_gate_state_fingerprints()
        expected_fingerprints = expected.get("fingerprints", {})
        checks = {
            key: restored.get(key) == expected_fingerprints.get(key)
            for key in ("global_step", "model", "optimizer", "rng", "scheduler")
        }
        observed_next_rng_sample = next_rng_sample(self.accelerator.device)
        checks["rng_next_sample"] = (
            observed_next_rng_sample == expected.get("next_rng_sample")
        )
        checks["pre_load_was_distinct"] = any(
            self._gate_pre_load_fingerprints.get(key) != expected_fingerprints.get(key)
            for key in ("global_step", "model", "optimizer", "rng", "scheduler")
        )
        if not all(checks.values()):
            raise RuntimeError(f"N=4 full-state reload mismatch on rank {rank}: {checks}")
        payload = {
            "batch_accounting": self._resolved_n4_gate_batch_accounting(),
            "checks": checks,
            "fingerprints": restored,
            "next_rng_sample": observed_next_rng_sample,
            "phase": "load_fresh_process",
            "pre_load_fingerprints": self._gate_pre_load_fingerprints,
            "process_nonce": self._gate_process_nonce,
            "process_pid": self._gate_process_pid,
            "process_start_ticks": self._gate_process_start_ticks,
            "rank": rank,
            "schema_name": "fastwam-n4-fullmodel-load-proof",
            "schema_version": 1,
            "world_size": int(self.accelerator.num_processes),
        }
        publish_exclusive_json(
            self._n4_gate_proof_dir() / f"load-state-rank-{rank:05d}.json",
            payload,
        )
        self.accelerator.wait_for_everyone()

    def publish_training_terminal_artifacts(
        self,
        *,
        config_relative_path: str,
        config_sha256: str,
    ) -> None:
        if not self.seal_training_run:
            return
        if self.formal_n4_fullmodel_gate:
            raise RuntimeError("N=4 gate terminal seal is owned by its fresh-load finalizer")
        if self.global_step != self.max_steps:
            raise RuntimeError(
                f"refusing terminal seal before max_steps: {self.global_step}/{self.max_steps}"
            )
        if self._last_step_metrics.get("step") != self.max_steps:
            raise RuntimeError("refusing terminal seal without finite final-step metrics")
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        training_mode = str(getattr(unwrapped_model, "training_mode", "")).strip().lower()
        steps = []
        if self.save_every > 0:
            steps.extend(
                self._periodic_steps_after_start(
                    self.save_every,
                    self.max_steps,
                    self.run_initial_global_step,
                )
            )
        if self.save_final_checkpoint_enabled and self.max_steps not in steps:
            steps.append(self.max_steps)
        self.accelerator.wait_for_everyone()
        complete_path = Path(self.output_dir) / "TRAINING.COMPLETE"
        failure_path = Path(self.output_dir) / "TRAINING.FAILED.json"
        if self.accelerator.is_main_process:
            try:
                publish_training_terminal_seal(
                    self.output_dir,
                    run_id=os.environ.get("RUN_ID", ""),
                    code_commit=self._git_commit() or "",
                    config_relative_path=config_relative_path,
                    config_sha256=config_sha256,
                    max_steps=self.max_steps,
                    expected_checkpoint_steps=steps,
                    expected_evaluation_steps=(
                        self._periodic_steps_after_start(
                            self.eval_every,
                            self.max_steps,
                            self.run_initial_global_step,
                        )
                    ),
                    world_size=int(self.accelerator.num_processes),
                    last_step_metrics=self._last_step_metrics,
                    evaluation_records=self._evaluation_records,
                    training_mode=training_mode,
                    dataset_contract_sha256=canonical_json_sha256(
                        self._dataset_run_contract
                    ),
                    authorization_gate_complete_sha256=os.environ.get(
                        "FASTWAM_N4_FULLMODEL_GATE_COMPLETE_SHA256", ""
                    ),
                    rehash_weights=self.terminal_rehash_weights,
                )
            except BaseException as error:
                if not complete_path.exists() and not complete_path.is_symlink():
                    publish_failure_marker(
                        self.output_dir,
                        marker_name=failure_path.name,
                        schema_name="fastwam-training-terminal-failure",
                        error=error,
                        success_markers=[complete_path.name],
                    )
                raise
        else:
            self._wait_for_terminal_or_failure(
                success_path=complete_path,
                failure_path=failure_path,
                label="run-level training terminal seal",
            )
        self.accelerator.wait_for_everyone()

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
                phase_balanced_fraction=self.phase_balanced_fraction,
            )
            if getattr(self, "provenance_mode", "sha256") == "stat_cmp":
                schedule_evidence_name = "schedule_exact_record_count"
                schedule_evidence = self.train_sampler.global_batches_per_epoch
            else:
                schedule_evidence_name = "schedule_sha256"
                schedule_evidence = self.train_sampler.schedule_fingerprint()
            logger.info(
                "Using hierarchical task/count-balanced batching: counts=%s tasks_by_count=%s "
                "batch_sizes=%s token_budget=%s global_batches=%d local_microbatches=%d "
                "optimizer_steps=%d phase_balanced_fraction=%.1f "
                "original_global_batches=%d phase_balanced_global_batches=%d "
                "%s=%s",
                self.train_sampler.observed_agent_counts,
                self.train_sampler.tasks_by_agent_count,
                self.train_sampler.batch_size_by_agent_count,
                self.train_sampler.agent_action_token_budget,
                self.train_sampler.global_batches_per_epoch,
                self.train_sampler.microbatches_per_process,
                self.train_sampler.optimizer_steps_per_epoch,
                self.train_sampler.phase_balanced_fraction,
                self.train_sampler.original_global_batches_per_epoch,
                self.train_sampler.phase_balanced_global_batches_per_epoch,
                schedule_evidence_name,
                schedule_evidence,
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
        if self.phase_balanced_fraction:
            raise ValueError(
                "`phase_balanced_fraction` is enabled, but the dataset does not expose "
                "the variable-agent task/count and B4 phase metadata it requires."
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

    @staticmethod
    def _periodic_steps_after_start(
        period: int,
        max_steps: int,
        initial_step: int,
    ) -> list[int]:
        period = int(period)
        max_steps = int(max_steps)
        initial_step = int(initial_step)
        if period <= 0 or max_steps <= initial_step:
            return []
        first_step = ((initial_step // period) + 1) * period
        return list(range(first_step, max_steps + 1, period))

    def _set_train_data_epoch(self, epoch: int):
        """Synchronize the source sampler and Accelerate's prepared loader."""

        epoch = int(epoch)
        if hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(epoch)
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

    @staticmethod
    def _canonical_json_sha256(value) -> str:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _sha256_file(path: str | Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                block = handle.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    def _dataset_contract(self, dataset):
        """Build the configured scientific data-identity contract."""

        if dataset is None:
            return None
        provenance_mode = getattr(self, "provenance_mode", "sha256")
        contract = {
            "class": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
            "length": int(len(dataset)),
        }
        if provenance_mode == "stat_cmp":
            contract["provenance_mode"] = provenance_mode
        scalar_attributes = (
            "num_frames",
            "action_horizon",
            "action_video_freq_ratio",
            "load_future_video",
            "action_dim",
            "state_dim",
            "agent_geometry_dim",
            "window_stride",
            "val_set_proportion",
            "is_training_set",
            "split_seed",
            "randomize_agent_order",
            "require_train_only_stats",
            "gaussian_cache_verify",
            "gaussian_cache_expected_manifest_sha256",
            "gaussian_cache_expected_selection_sha256",
            "gaussian_cache_expected_source_identity_sha256",
            "gaussian_channels",
            "context_len",
        )
        for name in scalar_attributes:
            if hasattr(dataset, name):
                value = getattr(dataset, name)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    contract[name] = value
        sequence_attributes = (
            "video_size",
            "video_indices",
            "required_agent_counts",
            "gaussian_size",
        )
        for name in sequence_attributes:
            if hasattr(dataset, name):
                value = getattr(dataset, name)
                contract[name] = None if value is None else list(value)

        root = getattr(dataset, "root_dir", None)
        if root is not None:
            root_path = Path(root).expanduser().resolve()
            contract["root_dir"] = str(root_path)
            source_inventory = []
            if root_path.is_dir():
                for source_path in sorted(root_path.rglob("*.h5")):
                    stat_result = source_path.stat()
                    source_inventory.append(
                        {
                            "path": source_path.relative_to(root_path).as_posix(),
                            "bytes": int(stat_result.st_size),
                            "mtime_ns": int(stat_result.st_mtime_ns),
                        }
                    )
            contract["source_inventory_count"] = len(source_inventory)
            if provenance_mode == "stat_cmp":
                # The directly inspectable inventory is intentionally retained
                # rather than replaced with a digest.  It is small (one row per
                # source HDF5 file) and makes path/size/mtime drift explicit.
                contract["source_inventory"] = source_inventory
                contract["source_inventory_total_bytes"] = sum(
                    item["bytes"] for item in source_inventory
                )
            else:
                contract["source_inventory_sha256"] = self._canonical_json_sha256(
                    source_inventory
                )

        entries = getattr(dataset, "entries", None)
        if entries is not None:
            # Do not include the mount-specific absolute `path`; source_path is
            # the immutable dataset-relative identity used by the cache too.
            normalized_entries = []
            for entry in entries:
                normalized_entries.append(
                    {
                        key: (list(value) if isinstance(value, tuple) else value)
                        for key, value in entry.items()
                        if key != "path"
                    }
                )
            if provenance_mode == "stat_cmp":
                entry_counts_by_source = {}
                for entry in normalized_entries:
                    source_path = str(entry.get("source_path", "<unknown>"))
                    entry_counts_by_source[source_path] = (
                        entry_counts_by_source.get(source_path, 0) + 1
                    )
                contract["window_index_count"] = len(normalized_entries)
                contract["window_index_counts_by_source"] = dict(
                    sorted(entry_counts_by_source.items())
                )
            else:
                contract["window_index_sha256"] = self._canonical_json_sha256(
                    normalized_entries
                )

        stats_path = getattr(dataset, "_stats_path", None)
        if stats_path is not None:
            stats_path = Path(stats_path).expanduser().resolve()
            normalization = {
                "path": str(stats_path),
                "schema": getattr(dataset, "_stats_metadata", None),
            }
            if provenance_mode == "stat_cmp":
                stat_result = stats_path.stat()
                normalization.update(
                    {
                        "bytes": int(stat_result.st_size),
                        "mtime_ns": int(stat_result.st_mtime_ns),
                    }
                )
            else:
                normalization["sha256"] = self._sha256_file(stats_path)
            contract["normalization"] = normalization
        return contract

    @staticmethod
    def _git_commit() -> str | None:
        declared = os.environ.get("FASTWAM_CODE_COMMIT", "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{40}", declared):
            return declared
        repository = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        value = completed.stdout.strip().lower()
        if completed.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value):
            return value
        return None

    def _resolved_config_contract(self) -> dict:
        resolved = OmegaConf.to_container(self.cfg, resolve=True)
        if not isinstance(resolved, dict):
            raise TypeError("Resolved training config must be a mapping")
        # These fields may change without altering training state semantics.
        for key in (
            "output_dir",
            "resume",
            "weights_only_warm_start",
            "num_workers",
            "log_every",
            "save_every",
            "eval_every",
            "eval_num_inference_steps",
            "offline_eval_num_samples",
            "save_training_state",
            "seal_training_state",
            "save_final_checkpoint",
            "seal_training_run",
            "terminal_rehash_weights",
            "provenance_mode",
            "allow_legacy_resume",
            "process_group_timeout_seconds",
            "checkpoint_io_timeout_seconds",
            "wandb",
            "hydra",
        ):
            resolved.pop(key, None)
        return resolved

    def _training_state_contract(self) -> dict:
        model = self.accelerator.unwrap_model(self.model)
        architecture = None
        architecture_builder = getattr(model, "_multi_robot_architecture_metadata", None)
        if callable(architecture_builder):
            architecture = architecture_builder()
        trainable = [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
            }
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        config_contract = self._resolved_config_contract()
        treatment = {
            "training_mode": getattr(model, "training_mode", None),
            "trainable_scope": self.trainable_scope,
            "checkpoint_state_kind": self.checkpoint_state_kind,
            "video_gen": getattr(model, "training_mode", None) == "joint",
            "hub": None if architecture is None else architecture.get("hub_enabled"),
            "gaussian": None if architecture is None else architecture.get("enable_gaussian"),
        }
        # A full MoT weights artifact and an Accelerate full-state directory are
        # self-contained.  Do not leak the node-local staging path of their
        # initialization checkpoint into a future resume contract.  Sparse
        # diagnostics retain the exact, loadable base dependency descriptor.
        base_checkpoint = None
        if self.checkpoint_state_kind == "sparse_delta":
            base_checkpoint = getattr(model, "_loaded_base_checkpoint_descriptor", None)
            if not isinstance(base_checkpoint, dict):
                base_path = getattr(model, "_loaded_base_checkpoint", None)
                base_sha256 = getattr(model, "_loaded_base_checkpoint_sha256", None)
                if base_path and base_sha256:
                    base_checkpoint = {
                        "path": str(Path(base_path).expanduser().resolve()),
                        "sha256": str(base_sha256).lower(),
                        "role": "base_dependency",
                    }
                else:
                    base_checkpoint = getattr(
                        self,
                        "_resume_base_checkpoint_provenance",
                        None,
                    )
        provenance_mode = getattr(self, "provenance_mode", "sha256")
        contract = {
            "contract_version": 2 if provenance_mode == "stat_cmp" else 1,
            "state_kind": "accelerate_full_state",
            "treatment": treatment,
            "multi_robot_architecture": architecture,
            "trainable_parameters": trainable,
            "base_checkpoint": base_checkpoint,
            "dataset": self._dataset_run_contract,
            "optimization": {
                "optimizer": "torch.optim.AdamW",
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "betas": [0.9, 0.95],
                "lr_scheduler_type": str(self.cfg.lr_scheduler_type),
                "max_steps": int(self.max_steps),
                "run_initial_global_step": self.run_initial_global_step,
                "optimizer_steps_this_run": self.optimizer_steps_this_run,
                "warmup_steps": self.scheduler_warmup_steps,
                "batch_size": self.batch_size,
                "agent_action_token_budget": self.agent_action_token_budget,
                "phase_balanced_fraction": self.phase_balanced_fraction,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "world_size": int(self.accelerator.num_processes),
                "mixed_precision": self.mixed_precision,
                "max_grad_norm": self.max_grad_norm,
                "seed": self.seed,
            },
            "code_commit": self._git_commit(),
        }
        if provenance_mode == "stat_cmp":
            # Keep the exact JSON-compatible structures in the trainer state;
            # resume compares them recursively without computing a digest.
            contract["provenance_mode"] = provenance_mode
            contract["resolved_config"] = config_contract
        else:
            contract["trainable_parameters_sha256"] = self._canonical_json_sha256(
                trainable
            )
            contract["resolved_config_sha256"] = self._canonical_json_sha256(
                config_contract
            )
        return contract

    @staticmethod
    def _contract_mismatches(saved, current, *, prefix: str = "", limit: int = 32):
        mismatches = []
        if isinstance(saved, dict) and isinstance(current, dict):
            for key in sorted(set(saved) | set(current)):
                path = f"{prefix}.{key}" if prefix else str(key)
                if key not in saved:
                    mismatches.append((path, "<missing>", current[key]))
                elif key not in current:
                    mismatches.append((path, saved[key], "<missing>"))
                else:
                    mismatches.extend(
                        Wan22Trainer._contract_mismatches(
                            saved[key], current[key], prefix=path, limit=limit
                        )
                    )
                if len(mismatches) >= limit:
                    break
            return mismatches[:limit]
        if saved != current:
            mismatches.append((prefix, saved, current))
        return mismatches[:limit]

    def _validate_training_state_contract(self, payload: dict, *, state_file: Path) -> None:
        saved = payload.get("run_contract")
        if not isinstance(saved, dict):
            if self.allow_legacy_resume:
                logger.warning(
                    "Legacy full-state resume is explicitly enabled without a run contract: %s",
                    state_file,
                )
                return
            raise RuntimeError(
                "Full-state resume lacks the required run_contract and is refused before "
                f"model/optimizer mutation: {state_file}. Set allow_legacy_resume=true only "
                "for an explicitly audited legacy recovery."
            )
        current = self._training_state_contract()
        saved_base = saved.get("base_checkpoint")
        current_base = current.get("base_checkpoint")
        if current_base is None and saved_base is not None:
            # An Accelerate full-state directory is self-contained, so the
            # freshly instantiated model has not traversed its original
            # weight-only base dependency.  Validate and restore that
            # provenance separately instead of falsely rejecting every exact
            # full-state resume merely because the runtime attribute is empty.
            current["base_checkpoint"] = saved_base
        mismatches = self._contract_mismatches(saved, current)
        if mismatches:
            raise RuntimeError(
                "Full-state run contract mismatch before accelerator.load_state: "
                f"{mismatches}"
            )
        if saved_base is not None:
            self._restore_base_checkpoint_provenance(saved_base, state_file=state_file)

    def _validate_resumable_terminal_evidence(
        self, payload: dict, *, state_file: Path
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        last_metrics = payload.get("last_step_metrics", {})
        evaluations = payload.get("evaluation_records", [])
        if not isinstance(last_metrics, dict) or not isinstance(evaluations, list):
            raise TypeError(
                f"Invalid resumable terminal evidence in trainer state: {state_file}"
            )
        if last_metrics:
            required_fields = {
                "grad_norm",
                "learning_rate",
                "loss",
                "loss_components",
                "step",
            }
            components = last_metrics.get("loss_components")
            values = {
                "grad_norm": last_metrics.get("grad_norm"),
                "learning_rate": last_metrics.get("learning_rate"),
                "loss": last_metrics.get("loss"),
                **(
                    {f"loss_components.{key}": value for key, value in components.items()}
                    if isinstance(components, dict)
                    else {"loss_components": float("nan")}
                ),
            }
            if (
                set(last_metrics) != required_fields
                or last_metrics.get("step") != int(payload.get("global_step", -1))
                or not all(np.isfinite(float(value)) for value in values.values())
            ):
                raise RuntimeError(
                    f"Invalid last-step metrics in trainer state: {state_file}"
                )
        elif self.seal_training_run and int(payload.get("global_step", 0)) > 0:
            raise RuntimeError(
                f"Formal resume lacks last-step terminal evidence: {state_file}"
            )

        if self.seal_training_run:
            model = self.accelerator.unwrap_model(self.model)
            training_mode = str(getattr(model, "training_mode", "")).strip().lower()
            saved_step = int(payload.get("global_step", -1))
            expected_steps = (
                self._periodic_steps_after_start(
                    self.eval_every,
                    saved_step,
                    self.run_initial_global_step,
                )
            )
            evaluations = normalize_formal_evaluation_records(
                evaluations,
                expected_steps=expected_steps,
                training_mode=training_mode,
            )
        else:
            evaluations = [dict(record) for record in evaluations]
        return dict(last_metrics), evaluations

    def _restore_base_checkpoint_provenance(self, descriptor, *, state_file: Path) -> None:
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "path",
            "sha256",
            "role",
        }:
            raise RuntimeError(
                f"Invalid base checkpoint provenance in full-state contract: {state_file}"
            )
        raw_path = descriptor.get("path")
        expected_sha256 = str(descriptor.get("sha256", "")).lower()
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or descriptor.get("role") != "base_dependency"
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            raise RuntimeError(
                f"Invalid base checkpoint provenance in full-state contract: {state_file}"
            )
        path = Path(raw_path).expanduser().resolve()

        # Read a multi-GiB base exactly once (rank zero), then share the result
        # before any Accelerator state mutation.  This keeps exact resume
        # fail-closed without multiplying CPFS traffic by the world size.
        verification = None
        if self.accelerator.is_main_process:
            try:
                if not path.is_file():
                    raise FileNotFoundError(f"Base checkpoint is missing: {path}")
                actual_sha256 = self._sha256_file(path)
                if actual_sha256 != expected_sha256:
                    raise ValueError(
                        "Base checkpoint SHA-256 mismatch: "
                        f"expected={expected_sha256} actual={actual_sha256} path={path}"
                    )
                verification = {"ok": True, "error": None}
            except Exception as exc:  # propagated verbatim to every rank below
                verification = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            shared = [verification]
            torch.distributed.broadcast_object_list(shared, src=0)
            verification = shared[0]
        if not isinstance(verification, dict) or verification.get("ok") is not True:
            error = None if not isinstance(verification, dict) else verification.get("error")
            raise RuntimeError(
                "Full-state base checkpoint provenance verification failed before "
                f"accelerator.load_state: {error}"
            )

        normalized = {
            "path": str(path),
            "sha256": expected_sha256,
            "role": "base_dependency",
        }
        self._resume_base_checkpoint_provenance = normalized
        model = self.accelerator.unwrap_model(self.model)
        for name, value in (
            ("_loaded_base_checkpoint", str(path)),
            ("_loaded_base_checkpoint_sha256", expected_sha256),
            ("_loaded_base_checkpoint_descriptor", normalized),
            ("_loaded_base_checkpoint_can_restore_sparse", True),
        ):
            if hasattr(model, name):
                setattr(model, name, value)

    def _load_weight_checkpoint_before_prepare(self):
        """Load a file checkpoint before optimizer/ZeRO master construction."""

        resume = self.resume
        if not resume:
            if getattr(self, "weights_only_warm_start_enabled", False):
                raise ValueError(
                    "weights_only_warm_start.enabled=true requires a resume checkpoint file"
                )
            return
        resume_path = Path(str(resume)).expanduser()
        if resume_path.is_dir():
            if getattr(self, "weights_only_warm_start_enabled", False):
                raise ValueError(
                    "weights_only_warm_start requires a .pt weight file, not a "
                    f"full-state resume directory: {resume_path}"
                )
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        if not resume_path.is_file():
            raise ValueError(
                f"Resume path must be a checkpoint file or state directory: {resume}"
            )
        if getattr(self, "weights_only_warm_start_enabled", False):
            if resume_path.suffix.lower() != ".pt":
                raise ValueError(
                    "weights_only_warm_start requires a native-v2 .pt weight file, got "
                    f"{resume_path}"
                )
            self._load_explicit_weights_only_warm_start(resume_path)
            self._weight_checkpoint_loaded_before_prepare = True
            logger.warning(
                "Loaded explicit cross-treatment weights-only warm start before "
                "ZeRO master construction; optimizer, scheduler, and epoch start "
                "fresh while the cumulative step starts at %d.",
                self.run_initial_global_step,
            )
            return
        logger.info(
            "Loading weight checkpoint before optimizer/DeepSpeed initialization: %s",
            resume,
        )
        self.model.load_checkpoint(str(resume_path), optimizer=None)
        self._weight_checkpoint_loaded_before_prepare = True
        # Keep this marker deliberately short and path-free. Rich may wrap the
        # preceding human-readable path across physical log lines, while the
        # formal scratch preflight needs one stable machine-readable receipt
        # emitted only after the checkpoint loader has returned successfully.
        logger.warning("FASTWAM_GENERIC_BASE_LOAD=PASS before_prepare=true")
        logger.warning(
            "Loaded .pt weights before ZeRO master construction; "
            "optimizer/scheduler/step are intentionally not restored."
        )

    def _load_explicit_weights_only_warm_start(self, checkpoint_path: Path) -> None:
        """Initialize from an explicitly declared native-v2 full checkpoint.

        This is the only supported cross-treatment checkpoint path.  A cheap
        meta-device read validates the source treatment before the model's
        strict architecture/tensor loader performs the real load.  Calling the
        native loader as a ``base_dependency`` intentionally skips only target
        treatment equality; native format, architecture, state kind, tensor
        keys, shapes, and dtypes remain strict.
        """

        resolved_path = checkpoint_path.expanduser().resolve(strict=True)
        payload = torch.load(
            resolved_path,
            map_location="meta",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(payload, dict):
            raise TypeError(
                "weights_only_warm_start checkpoint payload must be a mapping: "
                f"{resolved_path}"
            )
        expected = {
            "format": "fastwam_multi_robot_v2",
            "state_kind": getattr(
                self,
                "weights_only_warm_start_expected_source_state_kind",
                None,
            ),
            "training_mode": getattr(
                self,
                "weights_only_warm_start_expected_source_training_mode",
                None,
            ),
            "trainable_scope": getattr(
                self,
                "weights_only_warm_start_expected_source_trainable_scope",
                None,
            ),
        }
        mismatches = {
            key: {"expected": value, "observed": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "weights_only_warm_start source metadata mismatch: "
                f"{mismatches} in {resolved_path}"
            )
        if "mot" not in payload or "mot_trainable" in payload:
            raise ValueError(
                "weights_only_warm_start requires exactly a self-contained native-v2 "
                f"full `mot` state: {resolved_path}"
            )
        if payload.get("base_checkpoint") is not None:
            raise ValueError(
                "weights_only_warm_start full source must declare "
                f"base_checkpoint=null: {resolved_path}"
            )
        del payload

        loader = getattr(self.model, "_load_checkpoint_with_role", None)
        if not callable(loader):
            raise TypeError(
                "weights_only_warm_start requires the native FastWAM multi-robot "
                "strict checkpoint loader"
            )
        loader(
            str(resolved_path),
            optimizer=None,
            load_role="base_dependency",
            active_paths=set(),
            validate_trainable_scope=False,
        )
        logger.info(
            "Validated explicit weights-only warm start: path=%s "
            "source_training_mode=%s source_trainable_scope=%s source_state_kind=%s",
            resolved_path,
            self.weights_only_warm_start_expected_source_training_mode,
            self.weights_only_warm_start_expected_source_trainable_scope,
            self.weights_only_warm_start_expected_source_state_kind,
        )

    def _resume_training_state_after_prepare(self):
        """Restore prepared full state, or confirm an earlier file preload."""

        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            if (
                self.formal_n4_fullmodel_gate
                and self.n4_fullmodel_gate_phase == "load"
            ):
                self._gate_pre_load_fingerprints = (
                    self._n4_gate_state_fingerprints(
                        require_optimizer_state=False
                    )
                )
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        if not self._weight_checkpoint_loaded_before_prepare:
            raise RuntimeError(
                "Weight checkpoint reached post-prepare resume without being loaded "
                f"before optimizer construction: {resume}"
            )
        logger.info(
            "Weight-only checkpoint was loaded before prepare; no post-prepare reload: %s",
            resume,
        )

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

    @staticmethod
    def _to_batched_multi_robot_eval_sample(sample):
        """Batch one native-agent dataset item without dropping conditioning fields."""

        if not isinstance(sample, dict):
            raise TypeError(f"Multi-robot eval sample must be a dict, got {type(sample)}")
        required_tensor_ranks = {
            "video": 4,
            "action": 3,
            "agent_state": 2,
            "context": 2,
            "context_mask": 1,
        }
        optional_tensor_ranks = {
            "agent_geometry": 2,
            "agent_ids": 1,
            "agent_gaussian": 4,
            "action_is_pad": 2,
            "image_is_pad": 1,
            "b4_target_action_phase": 2,
            "b4_gripper_closed_target": 2,
            "b4_gripper_event_target": 2,
            "b4_stable_contact_proxy": 2,
        }
        missing = sorted(set(required_tensor_ranks) - set(sample))
        if missing:
            raise ValueError(f"Missing multi-robot eval fields: {missing}")

        batched = dict(sample)
        for key, unbatched_rank in {
            **required_tensor_ranks,
            **optional_tensor_ranks,
        }.items():
            if key not in sample:
                continue
            value = sample[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Multi-robot eval field {key!r} must be a Tensor")
            if value.ndim == unbatched_rank:
                value = value.unsqueeze(0)
            if value.ndim != unbatched_rank + 1 or value.shape[0] != 1:
                raise ValueError(
                    f"Multi-robot eval field {key!r} must be an unbatched rank-"
                    f"{unbatched_rank} tensor or a singleton batch, got {tuple(value.shape)}"
                )
            batched[key] = value

        action = batched["action"]
        num_agents = int(action.shape[1])
        horizon = int(action.shape[2])
        if num_agents < 1 or horizon < 1:
            raise ValueError(f"Invalid multi-robot action shape: {tuple(action.shape)}")
        if batched["agent_state"].shape[1] != num_agents:
            raise ValueError("agent_state and action agent axes differ in eval sample")
        for key in (
            "agent_geometry",
            "agent_ids",
            "agent_gaussian",
            "action_is_pad",
            "b4_target_action_phase",
            "b4_gripper_closed_target",
            "b4_gripper_event_target",
            "b4_stable_contact_proxy",
        ):
            if key in batched and batched[key].shape[1] != num_agents:
                raise ValueError(f"{key} and action agent axes differ in eval sample")
        for key in (
            "action_is_pad",
            "b4_target_action_phase",
            "b4_gripper_closed_target",
            "b4_gripper_event_target",
            "b4_stable_contact_proxy",
        ):
            if key in batched and batched[key].shape[2] != horizon:
                raise ValueError(f"{key} and action horizon axes differ in eval sample")

        agent_count = sample.get("agent_count", num_agents)
        if isinstance(agent_count, torch.Tensor):
            agent_count = agent_count.reshape(-1)
            if agent_count.numel() != 1:
                raise ValueError("A single multi-robot eval sample must have one agent_count")
            agent_count = agent_count.to(dtype=torch.long)
        else:
            agent_count = torch.tensor([int(agent_count)], dtype=torch.long)
        if int(agent_count.item()) != num_agents:
            raise ValueError(
                f"agent_count={int(agent_count.item())} does not match action N={num_agents}"
            )
        batched["agent_count"] = agent_count

        for key in ("prompt", "task_name"):
            if key not in batched:
                continue
            value = batched[key]
            if isinstance(value, str):
                batched[key] = [value]
            elif isinstance(value, tuple):
                batched[key] = list(value)
            elif not isinstance(value, list) or len(value) != 1:
                raise TypeError(f"Multi-robot eval field {key!r} must be str or singleton list")
        return batched

    @staticmethod
    def _select_multi_robot_eval_indices(dataset, *, limit: int, seed: int) -> list[int]:
        """Choose a fixed validation subset stratified by count and task.

        Counts remain the outer balancing axis.  Within each count, task
        strata are round-robined before a task contributes a second example,
        so a small formal subset does not accidentally omit a low-frequency
        task.  Datasets without stable task metadata retain count-only
        behavior through a single synthetic task stratum.
        """

        limit = min(max(int(limit), 0), len(dataset))
        if limit == 0:
            return []
        counts = resolve_agent_counts(dataset)
        if counts is None or len(counts) != len(dataset):
            raise TypeError(
                "Multi-robot offline validation requires stable agent_counts metadata."
            )
        task_ids = resolve_task_ids(dataset)
        if task_ids is None:
            task_ids = tuple("__all__" for _ in counts)
        if len(task_ids) != len(counts):
            raise TypeError(
                "Multi-robot offline validation task metadata length is unstable."
            )
        strata: dict[int, dict[str, list[int]]] = {}
        for index, (count, task_id) in enumerate(zip(counts, task_ids)):
            strata.setdefault(int(count), {}).setdefault(str(task_id), []).append(index)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        for count in sorted(strata):
            for task_id in sorted(strata[count]):
                bucket = strata[count][task_id]
                permutation = torch.randperm(
                    len(bucket), generator=generator
                ).tolist()
                strata[count][task_id] = [
                    bucket[position] for position in permutation
                ]

        # Materialize one task-balanced queue per cardinality, then balance
        # those queues against each other.  No replacement is needed because
        # limit is bounded by dataset length.
        buckets: dict[int, list[int]] = {}
        for count in sorted(strata):
            task_cursors = {task_id: 0 for task_id in strata[count]}
            ordered: list[int] = []
            while True:
                made_progress = False
                for task_id in sorted(strata[count]):
                    position = task_cursors[task_id]
                    task_bucket = strata[count][task_id]
                    if position >= len(task_bucket):
                        continue
                    ordered.append(task_bucket[position])
                    task_cursors[task_id] += 1
                    made_progress = True
                if not made_progress:
                    break
            buckets[count] = ordered

        selected = []
        cursor = {count: 0 for count in buckets}
        while len(selected) < limit:
            made_progress = False
            for count in sorted(buckets):
                position = cursor[count]
                if position >= len(buckets[count]):
                    continue
                selected.append(buckets[count][position])
                cursor[count] += 1
                made_progress = True
                if len(selected) == limit:
                    break
            if not made_progress:
                break
        return selected

    def _evaluate_multi_robot_offline(self, model):
        indices = self._select_multi_robot_eval_indices(
            self.val_dataset,
            limit=self.offline_eval_num_samples,
            seed=self.seed + 73_991,
        )
        if not indices:
            raise ValueError("Multi-robot offline validation selected no samples.")
        local_indices = indices[
            self.accelerator.process_index :: self.accelerator.num_processes
        ]
        # Stable fixed slots make every rank participate in the same reduction,
        # including ranks with no local sample when the validation subset is
        # smaller than the world size. Each metric has [sum, count].
        metric_names = ("total", "action", "video")
        local_metric_stats = torch.zeros(
            (len(metric_names), 2),
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        cuda_devices = []
        if self.accelerator.device.type == "cuda":
            cuda_devices = [
                self.accelerator.device.index
                if self.accelerator.device.index is not None
                else torch.cuda.current_device()
            ]

        was_training = bool(
            model.training
            or getattr(getattr(model, "action_expert", None), "training", False)
            or getattr(getattr(model, "video_expert", None), "training", False)
        )
        model.eval()
        try:
            for index in local_indices:
                sample = self._to_batched_multi_robot_eval_sample(
                    self.val_dataset[index]
                )
                # Validation noise must be identical at every eval point and
                # must not advance the training RNG stream.
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(self.seed + 1_000_003 * int(index))
                    with self.accelerator.autocast():
                        loss, loss_metrics = model.training_loss(sample)
                if loss.numel() != 1:
                    raise ValueError(
                        "Multi-robot offline validation loss must be scalar, "
                        f"got shape {tuple(loss.shape)} at index {index}"
                    )
                if not bool(torch.isfinite(loss.detach()).all().item()):
                    raise FloatingPointError(
                        f"Non-finite multi-robot offline validation loss at index {index}"
                    )
                if not isinstance(loss_metrics, dict):
                    raise TypeError(
                        "Multi-robot training_loss metrics must be a dict, "
                        f"got {type(loss_metrics)} at index {index}"
                    )
                component_values = {
                    "total": loss,
                    "action": loss_metrics.get("loss_action"),
                    "video": loss_metrics.get("loss_video"),
                }
                if component_values["action"] is None:
                    raise KeyError(
                        "Multi-robot offline validation requires loss_metrics['loss_action']"
                    )
                for metric_index, metric_name in enumerate(metric_names):
                    value = component_values[metric_name]
                    if value is None:
                        continue
                    value = torch.as_tensor(value).detach()
                    if value.numel() != 1:
                        raise ValueError(
                            f"Offline validation metric {metric_name!r} must be scalar, "
                            f"got {tuple(value.shape)} at index {index}"
                        )
                    if not bool(torch.isfinite(value).all().item()):
                        raise FloatingPointError(
                            f"Non-finite offline validation metric {metric_name!r} "
                            f"at index {index}"
                        )
                    local_metric_stats[metric_index, 0] += value.float().to(
                        device=self.accelerator.device, dtype=torch.float64
                    )
                    local_metric_stats[metric_index, 1] += 1
        finally:
            if was_training:
                self._set_dit_only_train_mode()

        global_metric_stats = self.accelerator.reduce(
            local_metric_stats,
            reduction="sum",
        )
        global_count = int(global_metric_stats[0, 1].item())
        if global_count != len(indices):
            raise RuntimeError(
                "Distributed offline validation sample count mismatch: "
                f"expected={len(indices)} reduced={global_count}"
            )
        for metric_index, metric_name in enumerate(metric_names[1:], start=1):
            metric_count = int(global_metric_stats[metric_index, 1].item())
            if metric_count not in {0, global_count}:
                raise RuntimeError(
                    "Distributed offline validation metric presence mismatch: "
                    f"metric={metric_name} count={metric_count}/{global_count}"
                )
        selected_counts = sorted(
            {int(resolve_agent_counts(self.val_dataset)[index]) for index in indices}
        )
        all_task_ids = resolve_task_ids(self.val_dataset)
        selected_tasks = (
            []
            if all_task_ids is None
            else sorted({str(all_task_ids[index]) for index in indices})
        )
        result = {
            "evaluation_kind": "multi_robot_offline_loss",
            "val_loss": float(
                (global_metric_stats[0, 0] / global_metric_stats[0, 1]).item()
            ),
            "val_loss_action": float(
                (global_metric_stats[1, 0] / global_metric_stats[1, 1]).item()
            ),
            "offline_samples": global_count,
            "offline_agent_counts": selected_counts,
            "offline_tasks": selected_tasks,
        }
        if int(global_metric_stats[2, 1].item()) == global_count:
            result["val_loss_video"] = float(
                (global_metric_stats[2, 0] / global_metric_stats[2, 1]).item()
            )
        return result

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        if hasattr(model, "infer_action_multi"):
            return self._evaluate_multi_robot_offline(model)
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
        destination = Path(self.weights_dir) / f"{step_tag}.pt"
        manifest_path = destination.with_name(f"{destination.name}.manifest.json")
        complete_path = destination.with_name(f"{destination.name}.COMPLETE")
        for output in (destination, manifest_path, complete_path):
            if output.exists() or output.is_symlink():
                raise FileExistsError(
                    f"Refusing to overwrite an existing weights artifact: {output}"
                )

        staging_root = Path(
            os.environ.get(
                "FASTWAM_WEIGHT_STAGING_DIR",
                "/tmp/fastwam-weight-staging",
            )
        ).expanduser().resolve()
        staging_root.mkdir(parents=True, exist_ok=True)
        staged = staging_root / (
            f".{step_tag}.rank0.{os.getpid()}.{time.time_ns()}.pt"
        )
        try:
            save_parameters = inspect.signature(model.save_checkpoint).parameters
            if "checkpoint_state_kind" not in save_parameters:
                if self.checkpoint_state_kind != "full":
                    raise RuntimeError(
                        f"Model {type(model).__name__} does not support explicit "
                        f"checkpoint_state_kind={self.checkpoint_state_kind!r}"
                    )
                model.save_checkpoint(staged, optimizer=None, step=self.global_step)
            else:
                model.save_checkpoint(
                    staged,
                    optimizer=None,
                    step=self.global_step,
                    checkpoint_state_kind=self.checkpoint_state_kind,
                )
            if staged.is_symlink() or not staged.is_file():
                raise RuntimeError(f"Model did not produce a regular checkpoint: {staged}")
            with staged.open("rb") as handle:
                os.fsync(handle.fileno())
            expected_bytes = staged.stat().st_size
            provenance_mode = getattr(self, "provenance_mode", "sha256")
            expected_sha256 = (
                self._sha256_regular_file(staged)
                if provenance_mode == "sha256"
                else None
            )

            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with staged.open("rb") as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
            except BaseException:
                # This process created the exact destination in exclusive mode;
                # removing only that known partial is safe. A process crash may
                # still leave an unsealed file, which a retry refuses to replace.
                destination.unlink(missing_ok=True)
                raise
            destination_stat = destination.stat()
            actual_bytes = destination_stat.st_size
            if provenance_mode == "stat_cmp":
                actual_sha256 = None
                strong_readback_ok = (
                    actual_bytes == expected_bytes
                    and self._regular_files_bytewise_equal(staged, destination)
                )
            else:
                actual_sha256 = self._sha256_regular_file(destination)
                strong_readback_ok = (actual_bytes, actual_sha256) == (
                    expected_bytes,
                    expected_sha256,
                )
            if not strong_readback_ok:
                if provenance_mode == "sha256":
                    raise RuntimeError(
                        "Published weights checkpoint failed strong readback: "
                        f"expected=({expected_bytes},{expected_sha256}) "
                        f"actual=({actual_bytes},{actual_sha256}) path={destination}"
                    )
                raise RuntimeError(
                    "Published weights checkpoint failed bytewise readback: "
                    f"expected_bytes={expected_bytes} actual_bytes={actual_bytes} "
                    f"path={destination}"
                )

            manifest = {
                "schema_name": "fastwam-weights-checkpoint",
                "schema_version": 2 if provenance_mode == "stat_cmp" else 1,
                "filename": destination.name,
                "bytes": actual_bytes,
                "global_step": int(self.global_step),
                "checkpoint_state_kind": self.checkpoint_state_kind,
            }
            if provenance_mode == "stat_cmp":
                manifest.update(
                    {
                        "path": str(destination.resolve()),
                        "mtime_ns": int(destination_stat.st_mtime_ns),
                        "file_count": 1,
                        "verification": "stat+bytewise-cmp",
                    }
                )
            else:
                manifest["sha256"] = actual_sha256
            manifest_bytes = (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            self._publish_exclusive_bytes(manifest_path, manifest_bytes)
            if provenance_mode == "stat_cmp":
                manifest_stat = manifest_path.stat()
                complete = {
                    "schema_name": "fastwam-weights-checkpoint-complete",
                    "schema_version": 2,
                    "manifest_filename": manifest_path.name,
                    "manifest_bytes": int(manifest_stat.st_size),
                    "manifest_mtime_ns": int(manifest_stat.st_mtime_ns),
                    "checkpoint_filename": destination.name,
                    "checkpoint_bytes": int(actual_bytes),
                    "checkpoint_mtime_ns": int(destination_stat.st_mtime_ns),
                    "file_count": 1,
                    "verification": "stat+bytewise-cmp",
                }
            else:
                manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
                complete = {
                    "schema_name": "fastwam-weights-checkpoint-complete",
                    "schema_version": 1,
                    "manifest_filename": manifest_path.name,
                    "manifest_sha256": manifest_sha256,
                    "checkpoint_sha256": actual_sha256,
                }
            complete_bytes = (
                json.dumps(complete, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            self._publish_exclusive_bytes(complete_path, complete_bytes)
            if provenance_mode == "stat_cmp":
                logger.info(
                    "Published weights checkpoint: path=%s bytes=%d "
                    "verification=stat+bytewise-cmp manifest=%s",
                    destination,
                    actual_bytes,
                    manifest_path,
                )
            else:
                logger.info(
                    "Sealed weights checkpoint: path=%s bytes=%d sha256=%s manifest=%s",
                    destination,
                    actual_bytes,
                    actual_sha256,
                    manifest_path,
                )
            return str(destination)
        except BaseException:
            # The three paths were verified absent and are created exclusively
            # by this call. Clean only these exact task-owned outputs when a
            # normal exception occurs before a COMPLETE seal.
            if not complete_path.exists():
                manifest_path.unlink(missing_ok=True)
                destination.unlink(missing_ok=True)
            raise
        finally:
            staged.unlink(missing_ok=True)

    @staticmethod
    def _sha256_regular_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _regular_files_bytewise_equal(
        source: Path,
        destination: Path,
        *,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> bool:
        """Compare two regular files without producing a persistent digest."""

        if (
            source.is_symlink()
            or destination.is_symlink()
            or not source.is_file()
            or not destination.is_file()
        ):
            return False
        if source.stat().st_size != destination.stat().st_size:
            return False
        with (
            source.open("rb") as source_handle,
            destination.open("rb") as destination_handle,
        ):
            while True:
                source_block = source_handle.read(chunk_bytes)
                destination_block = destination_handle.read(chunk_bytes)
                if source_block != destination_block:
                    return False
                if not source_block:
                    return True

    @staticmethod
    def _publish_exclusive_bytes(path: Path, payload: bytes) -> None:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def _wait_for_published_regular_file(self, path: Path, *, label: str) -> None:
        """Wait outside a collective for a rank-zero COMPLETE artifact.

        Large weight copies and full ZeRO state-tree hashes can take much longer
        than a normal collective.  Non-main ranks poll the shared filesystem
        while rank zero publishes the artifact and only enter the next barrier
        after the atomic COMPLETE/manifest file is visible.
        """

        deadline = time.monotonic() + self.checkpoint_io_timeout_seconds
        while time.monotonic() < deadline:
            if path.is_symlink():
                raise RuntimeError(f"{label} must not be a symlink: {path}")
            if path.exists():
                if not path.is_file():
                    raise RuntimeError(
                        f"{label} must be a regular file when published: {path}"
                    )
                return
            time.sleep(0.2)
        raise TimeoutError(
            f"timed out after {self.checkpoint_io_timeout_seconds}s waiting for "
            f"{label}: {path}"
        )

    def _wait_for_terminal_or_failure(
        self,
        *,
        success_path: Path,
        failure_path: Path,
        label: str,
    ) -> None:
        deadline = time.monotonic() + self.checkpoint_io_timeout_seconds
        while time.monotonic() < deadline:
            for path, kind in ((failure_path, "failure"), (success_path, "success")):
                if path.is_symlink():
                    raise RuntimeError(f"{label} {kind} marker must not be a symlink: {path}")
                if not path.exists():
                    continue
                if not path.is_file():
                    raise RuntimeError(f"{label} {kind} marker must be regular: {path}")
                if kind == "failure":
                    payload, _, _ = read_canonical_json(path)
                    raise RuntimeError(
                        f"{label} failed on rank zero: "
                        f"{payload.get('error_type')}: {payload.get('error_message')}"
                    )
                return
            time.sleep(0.2)
        raise TimeoutError(
            f"timed out after {self.checkpoint_io_timeout_seconds}s waiting for "
            f"{label}: success={success_path} failure={failure_path}"
        )

    def _assert_checkpoint_targets_absent(self, *, step_tag: str) -> None:
        """Collectively refuse stale or colliding outputs before checkpoint I/O.

        A non-main rank must never accept an old COMPLETE marker while rank zero
        is discovering that the same step already exists.  Every rank probes the
        shared targets, then a small all-reduce makes any observed conflict fatal
        on the whole world before rank zero starts the expensive weight copy.
        """

        targets = [
            Path(self.weights_dir) / f"{step_tag}.pt",
            Path(self.weights_dir) / f"{step_tag}.pt.manifest.json",
            Path(self.weights_dir) / f"{step_tag}.pt.COMPLETE",
        ]
        if self.save_training_state_enabled:
            state_path = Path(self.state_dir) / step_tag
            targets.append(state_path)
            if self.seal_training_state:
                targets.append(
                    state_path.with_name(f"{state_path.name}.state-tree.json")
                )
        local_conflicts = [
            path for path in targets if path.exists() or path.is_symlink()
        ]
        local_count = torch.tensor(
            [len(local_conflicts)],
            device=self.accelerator.device,
            dtype=torch.int64,
        )
        global_count = self.accelerator.reduce(local_count, reduction="sum")
        if int(global_count.item()) != 0:
            local_detail = (
                ", ".join(str(path) for path in local_conflicts) or "none-local"
            )
            raise FileExistsError(
                "Refusing checkpoint publication because at least one rank observed "
                f"pre-existing targets for {step_tag}; local_conflicts={local_detail}"
            )

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "evaluation_records": self._evaluation_records,
            "last_step_metrics": self._last_step_metrics,
            "run_contract": self._training_state_contract(),
        }
        if self._uses_agent_count_batch_sampler:
            payload["data_schedule"] = self._data_schedule_contract(self.epoch)
        publish_exclusive_json(state_file, payload)

    def _data_schedule_contract(self, epoch: int) -> dict[str, object]:
        """Describe the exact rank-aligned schedule used for an epoch."""

        provenance_mode = getattr(self, "provenance_mode", "sha256")
        contract = {
            "agent_action_token_budget": self.train_sampler.agent_action_token_budget,
            "gradient_accumulation_steps": self.train_sampler.gradient_accumulation_steps,
            "num_processes": self.train_sampler.num_processes,
            "global_batches_per_epoch": self.train_sampler.global_batches_per_epoch,
            "optimizer_steps_per_epoch": self.train_sampler.optimizer_steps_per_epoch,
            "phase_balanced_fraction": self.train_sampler.phase_balanced_fraction,
            "original_global_batches_per_epoch": (
                self.train_sampler.original_global_batches_per_epoch
            ),
            "phase_balanced_global_batches_per_epoch": (
                self.train_sampler.phase_balanced_global_batches_per_epoch
            ),
        }
        if provenance_mode == "stat_cmp":
            contract["provenance_mode"] = provenance_mode
            contract["epoch"] = int(epoch)
            contract["exact_schedule"] = [
                {
                    "source": source,
                    "phase": phase,
                    "indices": list(indices),
                }
                for source, phase, indices in self.train_sampler.global_epoch_schedule(
                    int(epoch)
                )
            ]
            contract["schedule_record_count"] = len(contract["exact_schedule"])
        else:
            contract["fingerprint"] = self.train_sampler.schedule_fingerprint(
                int(epoch)
            )
        return contract

    def _validate_data_schedule_compatibility(
        self,
        payload: dict,
        *,
        state_file: Path,
    ) -> None:
        """Reject schedule drift before Accelerate mutates training state."""

        if not self._uses_agent_count_batch_sampler:
            return
        saved_schedule = payload.get("data_schedule")
        if saved_schedule is None:
            if self.phase_balanced_fraction:
                raise RuntimeError(
                    "B4 full-state resume lacks the required data_schedule contract "
                    f"and is refused before accelerator.load_state: {state_file}"
                )
            logger.warning(
                "Trainer state predates data-schedule contracts; resume "
                "compatibility cannot be verified: %s",
                state_file,
            )
            return
        if not isinstance(saved_schedule, dict):
            raise TypeError(f"data_schedule must be a mapping: {state_file}")
        saved_schedule = dict(saved_schedule)
        if not self.phase_balanced_fraction:
            # Pre-B4 trainer states already carried the deterministic schedule
            # fingerprint and totals.  Interpret their absent B4 fields as the
            # original-only schedule so the default 0.0 treatment remains
            # backward compatible; B4 (0.5) never takes this migration path.
            saved_schedule.setdefault("phase_balanced_fraction", 0.0)
            saved_schedule.setdefault(
                "original_global_batches_per_epoch",
                saved_schedule.get("global_batches_per_epoch"),
            )
            saved_schedule.setdefault("phase_balanced_global_batches_per_epoch", 0)
        current_schedule = self._data_schedule_contract(
            int(payload.get("epoch", 0))
        )
        mismatches = {
            key: (saved_schedule.get(key), current_value)
            for key, current_value in current_schedule.items()
            if saved_schedule.get(key) != current_value
        }
        if mismatches:
            raise RuntimeError(
                "Cannot resume with a different deterministic data schedule "
                "before accelerator.load_state: "
                f"{mismatches}"
            )

    @staticmethod
    def _seal_training_state_tree(state_path: str) -> dict:
        state_dir = Path(state_path).expanduser().resolve(strict=True)
        manifest = state_dir.parent / f"{state_dir.name}.state-tree.json"
        repository = Path(__file__).resolve().parents[2]
        generator = repository / "scripts" / "state_tree_manifest.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(generator),
                "build",
                "--state-root",
                str(state_dir),
                "--output",
                str(manifest),
                "--role",
                "accelerate_zero2_full_state",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Failed to seal Accelerate/DeepSpeed state tree: "
                f"status={completed.returncode} stderr={completed.stderr.strip()}"
            )
        try:
            summary = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"State-tree generator returned invalid JSON: {completed.stdout!r}"
            ) from error
        logger.info(
            "Sealed full training state: path=%s manifest=%s sha256=%s files=%s bytes=%s",
            state_dir,
            manifest,
            summary.get("manifest_sha256"),
            summary.get("file_count"),
            summary.get("total_bytes"),
        )
        return summary

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"
        weights_complete = (
            Path(self.weights_dir) / f"{step_tag}.pt.COMPLETE"
        )

        self._assert_checkpoint_targets_absent(step_tag=step_tag)
        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        else:
            self._wait_for_published_regular_file(
                weights_complete,
                label="weights checkpoint COMPLETE marker",
            )
        self.accelerator.wait_for_everyone()

        state_path = None
        state_manifest = None
        if self.save_training_state_enabled:
            state_path = os.path.join(self.state_dir, step_tag)
            ensure_dir(state_path)
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self.accelerator.wait_for_everyone()
            if self.seal_training_state:
                state_manifest_path = Path(state_path).with_name(
                    f"{Path(state_path).name}.state-tree.json"
                )
                if self.accelerator.is_main_process:
                    state_manifest = self._seal_training_state_tree(state_path)
                else:
                    self._wait_for_published_regular_file(
                        state_manifest_path,
                        label="sealed training state-tree manifest",
                    )
                self.accelerator.wait_for_everyone()

        return {
            "weights_path": ckpt_path,
            "state_path": state_path,
            "state_manifest": state_manifest,
        }

    def _should_save_final_checkpoint(self, *, checkpoint_saved_this_step: bool) -> bool:
        """Return whether the terminal step still needs a checkpoint write."""
        return self.save_final_checkpoint_enabled and not checkpoint_saved_this_step

    def load_training_state(self, state_dir: str):
        state_file = Path(state_dir) / "trainer_state.json"
        payload = None
        restored_last_step_metrics: dict[str, object] = {}
        restored_evaluation_records: list[dict[str, object]] = []
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise TypeError(f"Trainer state metadata must be a mapping: {state_file}")
            self._validate_training_state_contract(payload, state_file=state_file)
            (
                restored_last_step_metrics,
                restored_evaluation_records,
            ) = self._validate_resumable_terminal_evidence(
                payload, state_file=state_file
            )
            self._validate_data_schedule_compatibility(
                payload,
                state_file=state_file,
            )
        elif not self.allow_legacy_resume:
            raise RuntimeError(
                "Full-state resume is missing trainer_state.json and is refused before "
                f"model/optimizer mutation: {state_file}. Set allow_legacy_resume=true only "
                "for an explicitly audited legacy recovery."
            )

        # The contract gate above must run before Accelerate mutates the model,
        # optimizer, scheduler, scaler, or RNG state.
        self.accelerator.load_state(input_dir=state_dir)
        if payload is not None:
            self.global_step = int(payload["global_step"])
            self._last_step_metrics = restored_last_step_metrics
            self._evaluation_records = restored_evaluation_records

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self._set_train_data_epoch(self.epoch)
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

        logger.info(
            "Starting training at cumulative_step=%d with max_steps=%d and "
            "optimizer_steps_this_run=%d.",
            self.global_step,
            self.max_steps,
            self.optimizer_steps_this_run,
        )
        # Keep this deliberately short: Rich may wrap the human-readable line
        # above, while the formal one-step launcher needs a stable receipt.
        logger.warning(
            "FASTWAM_TRAINING_START initial_global_step=%d max_steps=%d optimizer_steps_this_run=%d",
            self.global_step,
            self.max_steps,
            self.optimizer_steps_this_run,
        )
        self._set_train_data_epoch(self.epoch)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        self.run_local_samples = 0
        self.run_local_agent_action_tokens = 0
        if self.formal_n4_fullmodel_gate:
            if self.n4_fullmodel_gate_phase != "save":
                raise RuntimeError("the N=4 load phase must verify state without training")
            if not torch.cuda.is_available():
                raise RuntimeError("N=4 full-model gate requires CUDA")
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)

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
                gate_sample_shapes = None
                gate_losses = None
                if self.formal_n4_fullmodel_gate:
                    gate_sample_shapes = self._n4_gate_sample_shapes(sample)
                    gate_losses = self._n4_gate_losses(loss, loss_dict)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    gate_gradients = None
                    if self.formal_n4_fullmodel_gate:
                        gate_gradients = self._n4_gate_gradient_evidence(grad_norm)
                    self.optimizer.step()
                    if (
                        self.formal_n4_fullmodel_gate
                        and self.accelerator.optimizer_step_was_skipped
                    ):
                        raise RuntimeError("N=4 gate optimizer step was skipped")
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
                    if not np.isfinite(global_loss) or not np.isfinite(global_grad_norm):
                        raise RuntimeError(
                            "training produced non-finite terminal metrics: "
                            f"loss={global_loss} grad_norm={global_grad_norm}"
                        )
                    if not all(np.isfinite(value) for value in global_loss_metrics.values()):
                        raise RuntimeError(
                            f"training produced non-finite loss components: {global_loss_metrics}"
                        )
                    self._last_step_metrics = {
                        "grad_norm": global_grad_norm,
                        "learning_rate": current_lr,
                        "loss": global_loss,
                        "loss_components": dict(sorted(global_loss_metrics.items())),
                        "step": int(self.global_step),
                    }
                    if (
                        self.accelerator.is_main_process
                        and self.optimizer_steps_this_run == 1
                        and self.global_step == self.max_steps
                    ):
                        # Keep this receipt short and independent of the rich
                        # human-readable progress line, which may be wrapped.
                        logger.warning(
                            "FASTWAM_OPTIMIZER_STEP global_step=%d max_steps=%d",
                            self.global_step,
                            self.max_steps,
                        )
                    if self.formal_n4_fullmodel_gate:
                        if gate_sample_shapes is None or gate_losses is None or gate_gradients is None:
                            raise RuntimeError("N=4 gate lost required per-step evidence")
                        self._write_n4_gate_step_proof(
                            step=self.global_step,
                            sample_shapes=gate_sample_shapes,
                            losses=gate_losses,
                            gradients=gate_gradients,
                        )

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
                            evaluation_record = {
                                key: value
                                for key, value in metrics.items()
                                if key != "video_path"
                            }
                            evaluation_record["step"] = int(self.global_step)
                            self._evaluation_records.append(evaluation_record)
                            if metrics.get("evaluation_kind") == "multi_robot_offline_loss":
                                logger.info(
                                    "[eval] step=%d kind=offline_loss total=%.4f "
                                    "action=%.4f video=%s samples=%d agent_counts=%s tasks=%s",
                                    self.global_step,
                                    metrics["val_loss"],
                                    metrics["val_loss_action"],
                                    (
                                        "n/a"
                                        if "val_loss_video" not in metrics
                                        else f"{metrics['val_loss_video']:.4f}"
                                    ),
                                    metrics["offline_samples"],
                                    metrics["offline_agent_counts"],
                                    metrics["offline_tasks"],
                                )
                                eval_payload = {
                                    "eval/val_loss": float(metrics["val_loss"]),
                                    "eval/val_loss_action": float(
                                        metrics["val_loss_action"]
                                    ),
                                    "eval/offline_samples": int(
                                        metrics["offline_samples"]
                                    ),
                                }
                                if "val_loss_video" in metrics:
                                    eval_payload["eval/val_loss_video"] = float(
                                        metrics["val_loss_video"]
                                    )
                                self._wandb_log(eval_payload)
                            else:
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

        # The loop is skipped when an exact full-state resume already starts
        # at max_steps.  Its checkpoint was sealed before that state could be
        # selected for resume, so never try to recreate the same exclusive
        # step target; the run-level terminal publisher below will strongly
        # re-read it and fail closed if it is incomplete or changed.
        if self.global_step >= self.max_steps:
            logger.info(
                "Training state already restored at max_steps=%d; "
                "skipping duplicate checkpoint publication.",
                self.max_steps,
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
        
