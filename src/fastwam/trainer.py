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
    ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT,
    ACTION_ONLY_N2_1X8_WORLD_SIZE,
    ACTION_ONLY_N2_PAID_GATE_STEP,
    ACTION_ONLY_N2_RELOAD_LOAD_ATTEMPTS_DIR,
    ACTION_ONLY_N2_RELOAD_PROOF_DIR,
    ACTION_ONLY_N2_RELOAD_PROOF_SCHEMA_VERSION,
    ACTION_ONLY_N2_RUN_PROFILES,
    ACTION_ONLY_N2_TERMINAL_CANDIDATE,
    N4_GATE_GLOBAL_TRAIN_BATCH_SIZE,
    N4_GATE_GRADIENT_ACCUMULATION_STEPS,
    N4_GATE_LOCAL_MICRO_BATCH_SIZE,
    N4_GATE_MAX_PEAK_ALLOCATED_BYTES,
    N4_GATE_MAX_PEAK_RESERVED_BYTES,
    N4_GATE_TRAIN_STEPS,
    N4_GATE_WORLD_SIZE,
    canonical_json_sha256,
    checkpoint_seal_descriptor,
    finalize_action_only_n2_paid_gate,
    next_rng_sample,
    normalize_formal_evaluation_records,
    publish_exclusive_json,
    publish_failure_marker,
    publish_action_only_n2_terminal_candidate,
    publish_action_only_n2_reload_attempt_commit,
    publish_action_only_n2_reload_proof_record,
    publish_training_terminal_seal,
    read_canonical_json,
    require_proof_attempt_id,
    require_sha256,
    resolved_unaliased_directory,
    state_fingerprints,
    validate_action_only_n2_reload_proof,
    validate_action_only_n2_terminal_reservation,
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
from .nohash_artifacts import (
    copy_exclusive_and_compare as nohash_copy_exclusive_and_compare,
    publish_exclusive_json as nohash_publish_exclusive_json,
    read_json as nohash_read_json,
    regular_file_metadata as nohash_regular_file_metadata,
)

logger = get_logger(__name__)


class Wan22Trainer:
    # Preserve the historical integrity behavior for lightweight callers and
    # tests that construct the trainer with ``__new__`` instead of ``__init__``.
    # Normal training always replaces this with the explicitly configured mode.
    artifact_integrity_mode = "sha256"

    @staticmethod
    def _validate_recovery_gate_stop_after_checkpoint_step(
        configured_step,
        *,
        artifact_integrity_mode,
        max_steps,
        save_every,
    ):
        if configured_step is None:
            return None
        if isinstance(configured_step, bool) or not isinstance(configured_step, int):
            raise ValueError(
                "`recovery_gate_stop_after_checkpoint_step` must be a positive "
                f"integer when enabled, got {configured_step!r}"
            )
        if artifact_integrity_mode != "metadata_no_hash":
            raise ValueError(
                "`recovery_gate_stop_after_checkpoint_step` is restricted to "
                "artifact_integrity_mode='metadata_no_hash'"
            )
        if configured_step <= 0:
            raise ValueError(
                "`recovery_gate_stop_after_checkpoint_step` must be positive, "
                f"got {configured_step}"
            )
        if max_steps is None:
            raise ValueError(
                "`recovery_gate_stop_after_checkpoint_step` requires an explicit "
                "max_steps"
            )
        if configured_step >= max_steps:
            raise ValueError(
                "`recovery_gate_stop_after_checkpoint_step` must be smaller than "
                f"max_steps ({max_steps}), got {configured_step}"
            )
        if save_every <= 0:
            raise ValueError(
                "`recovery_gate_stop_after_checkpoint_step` requires save_every>0"
            )
        if configured_step % save_every != 0:
            raise ValueError(
                "`recovery_gate_stop_after_checkpoint_step` must coincide with a "
                f"checkpoint step selected by save_every={save_every}, got "
                f"{configured_step}"
            )
        return configured_step

    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.artifact_integrity_mode = str(
            cfg.get("artifact_integrity_mode", "sha256")
        ).strip().lower()
        if self.artifact_integrity_mode not in {"sha256", "metadata_no_hash"}:
            raise ValueError(
                "artifact_integrity_mode must be 'sha256' or "
                f"'metadata_no_hash', got {self.artifact_integrity_mode!r}"
            )
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.recovery_gate_stop_after_checkpoint_step = (
            self._validate_recovery_gate_stop_after_checkpoint_step(
                cfg.get("recovery_gate_stop_after_checkpoint_step", None),
                artifact_integrity_mode=self.artifact_integrity_mode,
                max_steps=self.max_steps,
                save_every=self.save_every,
            )
        )
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
        
        self.resume = cfg.resume
        self.init_weights = cfg.get("init_weights", None)
        if self.resume and self.init_weights:
            raise ValueError("`resume` and `init_weights` are mutually exclusive")
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
        self.allow_legacy_resume = bool(cfg.get("allow_legacy_resume", False))
        self.save_training_state_enabled = bool(cfg.get("save_training_state", True))
        self.seal_training_state = bool(cfg.get("seal_training_state", False))
        self.save_final_checkpoint_enabled = bool(cfg.get("save_final_checkpoint", True))
        self.seal_training_run = bool(cfg.get("seal_training_run", False))
        self.terminal_rehash_weights = bool(cfg.get("terminal_rehash_weights", True))
        if self.artifact_integrity_mode == "metadata_no_hash":
            if self.seal_training_state or self.seal_training_run:
                raise ValueError(
                    "metadata_no_hash requires seal_training_state=false and "
                    "seal_training_run=false"
                )
            if self.terminal_rehash_weights:
                raise ValueError(
                    "metadata_no_hash requires terminal_rehash_weights=false"
                )
        configured_terminal_contract = cfg.get("training_terminal_contract", None)
        self.training_terminal_contract = (
            None
            if configured_terminal_contract in (None, "", "null")
            else str(configured_terminal_contract).strip()
        )
        if (
            self.artifact_integrity_mode == "metadata_no_hash"
            and self.training_terminal_contract is not None
        ):
            raise ValueError(
                "metadata_no_hash is a non-formal recovery gate and forbids "
                "training_terminal_contract"
            )
        if self.training_terminal_contract not in {
            None,
            ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT,
        }:
            raise ValueError(
                "unsupported training_terminal_contract: "
                f"{self.training_terminal_contract!r}"
            )
        configured_run_profile = cfg.get("training_run_profile", None)
        self.training_run_profile = (
            None
            if configured_run_profile in (None, "", "null")
            else str(configured_run_profile).strip()
        )
        if self.training_terminal_contract is None:
            if self.training_run_profile is not None:
                raise ValueError(
                    "training_run_profile requires an explicit "
                    "training_terminal_contract"
                )
        elif self.training_run_profile not in ACTION_ONLY_N2_RUN_PROFILES:
            raise ValueError(
                "action_only_n2_1x8_v1 requires training_run_profile in "
                f"{sorted(ACTION_ONLY_N2_RUN_PROFILES)}, got "
                f"{self.training_run_profile!r}"
            )
        configured_task_scope_receipt = cfg.get(
            "training_task_scope_receipt", None
        )
        self.training_task_scope_receipt = (
            None
            if configured_task_scope_receipt in (None, "", "null")
            else str(configured_task_scope_receipt).strip()
        )
        if (
            self.training_terminal_contract is None
            and self.training_task_scope_receipt is not None
        ):
            raise ValueError(
                "training_task_scope_receipt requires an explicit "
                "training_terminal_contract"
            )
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
        requested_n2_reload_phase = os.environ.get(
            "FASTWAM_N2_RELOAD_PROOF_PHASE", ""
        ).strip().lower()
        requested_n2_source_output = os.environ.get(
            "FASTWAM_N2_RELOAD_SOURCE_OUTPUT", ""
        ).strip()
        requested_n2_reload_attempt_id = os.environ.get(
            "FASTWAM_N2_RELOAD_PROOF_ATTEMPT_ID", ""
        ).strip()
        requested_n2_load_attempt_id = os.environ.get(
            "FASTWAM_N2_RELOAD_LOAD_ATTEMPT_ID", ""
        ).strip()
        if requested_n2_reload_phase not in {"", "load"}:
            raise ValueError(
                "FASTWAM_N2_RELOAD_PROOF_PHASE must be unset or 'load'; "
                "the paid save phase is selected by its terminal contract"
            )
        is_n2_paid_save = (
            self.training_terminal_contract == ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT
            and self.training_run_profile == "paid_gate_1step"
        )
        if is_n2_paid_save:
            if (
                requested_n2_reload_phase
                or requested_n2_source_output
                or requested_n2_load_attempt_id
            ):
                raise ValueError(
                    "N=2 paid save phase forbids fresh-load environment markers"
                )
            self.n2_reload_proof_phase = "save"
            self.n2_reload_source_output = self.output_dir
        elif requested_n2_reload_phase == "load":
            if self.training_terminal_contract is not None:
                raise ValueError(
                    "N=2 fresh reload must drop terminal publication authority"
                )
            if not requested_n2_source_output:
                raise ValueError(
                    "FASTWAM_N2_RELOAD_SOURCE_OUTPUT is required in the load phase"
                )
            self._n2_reload_load_attempt_id = require_proof_attempt_id(
                requested_n2_load_attempt_id,
                label="FASTWAM_N2_RELOAD_LOAD_ATTEMPT_ID",
            )
            self.n2_reload_proof_phase = "load"
            self.n2_reload_source_output = requested_n2_source_output
        else:
            if requested_n2_source_output or requested_n2_load_attempt_id:
                raise ValueError(
                    "N=2 fresh-load environment markers require the load phase"
                )
            self.n2_reload_proof_phase = None
            self.n2_reload_source_output = None
            self._n2_reload_load_attempt_id = None
        if self.n2_reload_proof_phase is not None:
            self._n2_reload_proof_attempt_id = require_proof_attempt_id(
                requested_n2_reload_attempt_id,
                label="FASTWAM_N2_RELOAD_PROOF_ATTEMPT_ID",
            )
            self._n2_reload_process_nonce = secrets.token_hex(16)
            self._n2_reload_process_pid = os.getpid()
            self._n2_reload_process_start_ticks = self._process_start_ticks()
            if self.n2_reload_proof_phase == "save":
                self._n2_reload_load_attempt_id = None
        else:
            if requested_n2_reload_attempt_id:
                raise ValueError(
                    "FASTWAM_N2_RELOAD_PROOF_ATTEMPT_ID is forbidden outside "
                    "the N=2 paid save/load proof phases"
                )
            self._n2_reload_proof_attempt_id = None
        self._n2_reload_pre_load_fingerprints = None
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
        dataset_contract_builder = (
            self._dataset_contract_metadata_no_hash
            if self.artifact_integrity_mode == "metadata_no_hash"
            else self._dataset_contract
        )
        train_data_contract = dataset_contract_builder(self.train_dataset)
        val_data_contract = (
            train_data_contract
            if self.val_dataset is self.train_dataset
            else dataset_contract_builder(self.val_dataset)
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

        # Reject an unauthorized or unsealed formal N=2 resume before any
        # checkpoint bytes can mutate the model.  The preflight depends only
        # on the resolved run configuration and sealed on-disk metadata; the
        # full terminal authorization remains below because it also verifies
        # the checkpoint descriptor installed by the weight loader.
        if self.training_terminal_contract is not None:
            self._preflight_action_only_n2_resume_before_load()
            if self.init_weights:
                # Authorize the immutable run/task scope before deserializing
                # any initialization tensor into the live model.  Sampler and
                # loaded-checkpoint checks remain in the full validation below.
                self._validate_action_only_n2_terminal_contract(preload=True)

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

        # Every formal N=2 authorization and immutable checkpoint contract is
        # checked before prepare/zero_grad/W&B can mutate state or publish data.
        if self.training_terminal_contract is not None:
            self._validate_action_only_n2_terminal_contract()
        elif self.n2_reload_proof_phase == "load":
            self._validate_action_only_n2_reload_load_contract(post_load=False)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_training_state_after_prepare()

        if self.formal_n4_fullmodel_gate:
            self._validate_n4_fullmodel_gate_contract()
        elif self.n2_reload_proof_phase == "load":
            self._validate_action_only_n2_reload_load_contract(post_load=True)
            self.publish_action_only_n2_reload_load_proof()

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

    def _preflight_action_only_n2_resume_before_load(self) -> None:
        """Validate a formal N=2 resume without mutating prepared state.

        In particular, a resume directory is accepted only when it is the
        canonical, sealed step-500/step-1000 directory owned by this run.  Its
        trainer contract is read through the no-symlink canonical JSON reader
        and checked before ``accelerator.load_state`` can run.  The contract
        check also restores the immutable base-checkpoint descriptor needed by
        the subsequent terminal-authorization validation.
        """

        if self.training_terminal_contract != ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT:
            raise ValueError(
                "unexpected N=2 terminal contract during resume preflight: "
                f"{self.training_terminal_contract!r}"
            )
        if self.training_run_profile == "paid_gate_1step":
            if self.resume:
                raise ValueError(
                    "N=2 paid gate must start from init_weights and cannot resume"
                )
            return
        if self.training_run_profile != "formal_1k" or not self.resume:
            return
        if self.init_weights:
            raise ValueError("N=2 formal resume must not also apply init_weights")

        resume_supplied = Path(str(self.resume)).expanduser()
        if not resume_supplied.is_absolute():
            raise ValueError("N=2 formal resume path must be absolute")
        resume_path = resolved_unaliased_directory(
            resume_supplied,
            label="N=2 formal resume state",
        )
        expected_parent = resolved_unaliased_directory(
            Path(self.output_dir) / "checkpoints" / "state",
            label="N=2 formal state parent",
        )
        if resume_path.parent != expected_parent:
            raise RuntimeError(
                "N=2 formal resume must stay inside this run's sealed state root"
            )
        match = re.fullmatch(r"step_(\d{6})", resume_path.name)
        resume_step = int(match.group(1)) if match else -1
        if resume_step not in {500, 1000}:
            raise RuntimeError(
                "N=2 formal resume requires a sealed step-500 or step-1000 state"
            )
        descriptor = checkpoint_seal_descriptor(
            self.output_dir,
            step=resume_step,
            rehash_weights=True,
            expected_checkpoint_state_kind="sparse_delta",
        )
        if (
            descriptor.get("global_step") != resume_step
            or descriptor.get("state", {}).get("root")
            != f"checkpoints/state/step_{resume_step:06d}"
        ):
            raise RuntimeError("N=2 formal resume checkpoint descriptor mismatch")

        state_file = resume_path / "trainer_state.json"
        payload, _, _ = read_canonical_json(state_file)
        if not isinstance(payload, dict):
            raise TypeError(f"Trainer state metadata must be a mapping: {state_file}")
        if payload.get("global_step") != resume_step:
            raise RuntimeError(
                "N=2 formal resume trainer step does not match the sealed directory: "
                f"directory={resume_step} metadata={payload.get('global_step')!r}"
            )
        self._validate_training_state_contract(payload, state_file=state_file)

    def _validate_action_only_n2_terminal_contract(
        self, *, preload: bool = False
    ) -> None:
        """Fail closed before training under the versioned N=2 formal contract."""

        if self.training_terminal_contract != ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT:
            raise ValueError(
                "unexpected N=2 terminal contract: "
                f"{self.training_terminal_contract!r}"
            )
        profile_scalar_contracts = {
            "paid_gate_1step": {
                "batch_size": (self.batch_size, 1),
                "max_steps": (self.max_steps, 1),
                "save_every": (self.save_every, 1),
                "eval_every": (self.eval_every, 0),
                "offline_eval_num_samples": (self.offline_eval_num_samples, 0),
            },
            "formal_1k": {
                "batch_size": (self.batch_size, 2),
                "max_steps": (self.max_steps, 1000),
                "save_every": (self.save_every, 500),
                "eval_every": (self.eval_every, 500),
                "offline_eval_num_samples": (self.offline_eval_num_samples, 32),
            },
        }
        if self.training_run_profile not in profile_scalar_contracts:
            raise ValueError(
                "N=2 action-only terminal contract lacks a supported run profile: "
                f"{self.training_run_profile!r}"
            )
        scalar_contract = {
            "gradient_accumulation_steps": (self.gradient_accumulation_steps, 4),
            "mixed_precision": (self.mixed_precision, "bf16"),
            "checkpoint_state_kind": (self.checkpoint_state_kind, "sparse_delta"),
            "trainable_scope": (self.trainable_scope, "action"),
            "agent_action_token_budget": (self.agent_action_token_budget, 128),
            "world_size": (int(self.accelerator.num_processes), 8),
            "formal_n4_fullmodel_gate": (self.formal_n4_fullmodel_gate, False),
            **profile_scalar_contracts[self.training_run_profile],
        }
        mismatches = {
            name: {"observed": observed, "expected": expected}
            for name, (observed, expected) in scalar_contract.items()
            if observed != expected
        }
        if mismatches:
            raise ValueError(
                f"N=2 action-only terminal scalar contract mismatch: {mismatches}"
            )
        if self.terminal_rehash_weights is not True:
            raise ValueError(
                "N=2 action-only terminal contracts require "
                "terminal_rehash_weights=true"
            )
        if self.training_run_profile == "paid_gate_1step":
            if self.resume:
                raise ValueError(
                    "N=2 paid gate must start from init_weights and cannot resume"
                )
            if not self.init_weights:
                raise ValueError("N=2 paid gate requires init_weights")
        elif self.resume:
            if self.init_weights:
                raise ValueError("N=2 formal resume must not also apply init_weights")
            resume_supplied = Path(str(self.resume)).expanduser()
            if not resume_supplied.is_absolute():
                raise ValueError("N=2 formal resume path must be absolute")
            resume_path = resolved_unaliased_directory(
                resume_supplied, label="N=2 formal resume state"
            )
            expected_parent = resolved_unaliased_directory(
                Path(self.output_dir) / "checkpoints" / "state",
                label="N=2 formal state parent",
            )
            if resume_path.parent != expected_parent:
                raise RuntimeError(
                    "N=2 formal resume must stay inside this run's sealed state root"
                )
            match = re.fullmatch(r"step_(\d{6})", resume_path.name)
            resume_step = int(match.group(1)) if match else -1
            if resume_step not in {500, 1000}:
                raise RuntimeError(
                    "N=2 formal resume requires a sealed step-500 or step-1000 state"
                )
            checkpoint_seal_descriptor(
                self.output_dir,
                step=resume_step,
                rehash_weights=True,
                expected_checkpoint_state_kind="sparse_delta",
            )
        elif not self.init_weights:
            raise ValueError("N=2 formal first launch requires init_weights")
        if not (
            self.save_training_state_enabled
            and self.seal_training_state
            and self.save_final_checkpoint_enabled
            and self.seal_training_run
        ):
            raise ValueError(
                "N=2 action-only terminal contract requires saved and sealed training "
                "state, final weights, and the run-level terminal seal"
            )
        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if plugin is None:
            raise RuntimeError("N=2 action-only terminal contract requires DeepSpeed")
        zero_stage = plugin.deepspeed_config.get("zero_optimization", {}).get("stage")
        if zero_stage != 2:
            raise ValueError(
                f"N=2 action-only terminal contract requires ZeRO stage 2, got {zero_stage!r}"
            )
        if not preload:
            if not self._uses_agent_count_batch_sampler:
                raise ValueError(
                    "N=2 action-only terminal contract requires the "
                    "variable-agent sampler"
                )
            observed_counts = [
                int(value) for value in self.train_sampler.observed_agent_counts
            ]
            batch_sizes = {
                int(key): int(value)
                for key, value in self.train_sampler.batch_size_by_agent_count.items()
            }
            if observed_counts != [2] or batch_sizes != {2: 2}:
                raise ValueError(
                    "N=2 terminal sampler contract mismatch: "
                    f"counts={observed_counts} batch_sizes={batch_sizes}"
                )

        model = self.accelerator.unwrap_model(self.model)
        training_mode = str(getattr(model, "training_mode", "")).strip().lower()
        if training_mode != "action_only_cache":
            raise ValueError(
                "N=2 action-only terminal contract requires "
                f"training_mode='action_only_cache', got {training_mode!r}"
            )
        effective_patched_tree = os.environ.get(
            "FASTWAM_EFFECTIVE_PATCHED_TREE", ""
        ).strip().lower()
        request_sha256 = os.environ.get("FASTWAM_REQUEST_SHA256", "").strip().lower()
        init_checkpoint_sha256 = os.environ.get(
            "FASTWAM_INIT_CHECKPOINT_SHA256", ""
        ).strip().lower()
        if not preload:
            base_descriptor = getattr(
                model, "_loaded_base_checkpoint_descriptor", None
            )
            if (
                not isinstance(base_descriptor, dict)
                or set(base_descriptor) != {"path", "role", "sha256"}
                or base_descriptor.get("role") != "base_dependency"
            ):
                raise RuntimeError(
                    "N=2 action-only terminal contract requires the exact loaded "
                    "base checkpoint descriptor"
                )
            if (
                str(base_descriptor.get("sha256", "")).strip().lower()
                != init_checkpoint_sha256
            ):
                raise RuntimeError(
                    "loaded initialization checkpoint does not match "
                    "FASTWAM_INIT_CHECKPOINT_SHA256"
                )
        if not self.training_task_scope_receipt:
            raise ValueError(
                "N=2 action-only terminal contract requires "
                "training_task_scope_receipt"
            )
        evidence = validate_action_only_n2_terminal_reservation(
            self.output_dir,
            run_id=os.environ.get("RUN_ID", ""),
            base_code_commit=self._git_commit() or "",
            effective_patched_tree=effective_patched_tree,
            request_sha256=request_sha256,
            init_checkpoint_sha256=init_checkpoint_sha256,
            world_size=int(self.accelerator.num_processes),
            formal_n4_fullmodel_gate=self.formal_n4_fullmodel_gate,
            checkpoint_state_kind=self.checkpoint_state_kind,
            trainable_scope=self.trainable_scope,
            training_mode=training_mode,
            dataset_contract=self._dataset_run_contract,
            task_scope_receipt_relative_path=self.training_task_scope_receipt,
            run_profile=self.training_run_profile,
        )
        if not preload:
            required_tasks = evidence["task_scope"]["required_tasks"]
            sampler_tasks = [
                str(value)
                for value in self.train_sampler.tasks_by_agent_count.get(2, ())
            ]
            if sampler_tasks != required_tasks:
                raise ValueError(
                    "N=2 terminal sampler task scope mismatch: "
                    f"sampler={sampler_tasks} required={required_tasks}"
                )
        for label, dataset in (("train", self.train_dataset), ("val", self.val_dataset)):
            if getattr(dataset, "load_future_video", True):
                raise ValueError(
                    f"N=2 action-only terminal {label} dataset must not load future video"
                )
        self._action_only_n2_terminal_evidence = evidence

    def _n2_reload_source_root(self) -> Path:
        if self.n2_reload_proof_phase is None or not self.n2_reload_source_output:
            raise RuntimeError("N=2 reload proof has no source output root")
        return resolved_unaliased_directory(
            self.n2_reload_source_output,
            label="N=2 reload source output",
        )

    def _n2_reload_proof_dir(self) -> Path:
        path = self._n2_reload_source_root() / ACTION_ONLY_N2_RELOAD_PROOF_DIR
        if path.exists() or path.is_symlink():
            return resolved_unaliased_directory(
                path, label="N=2 reload proof directory"
            )
        return path

    def _n2_reload_state_fingerprints(
        self, *, require_optimizer_state: bool = True
    ) -> dict[str, object]:
        return state_fingerprints(
            model=self.accelerator.unwrap_model(self.model),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            require_optimizer_state=require_optimizer_state,
            full_state=True,
        )

    def _n2_reload_sampler_cursor(self) -> dict[str, object]:
        if not self._uses_agent_count_batch_sampler:
            raise RuntimeError("N=2 reload proof requires the native agent-count sampler")
        cursor = {
            "agent_action_token_budget": int(
                self.train_sampler.agent_action_token_budget
            ),
            "batch_in_epoch": int(self.batch_in_epoch),
            "epoch": int(self.epoch),
            # The save world's sampler has not been put into resume mode yet.
            # Bind the semantic cursor restored from trainer_state.json rather
            # than the sampler's transient resume offset.
            "global_batch_offset": int(self.batch_in_epoch)
            * int(self.accelerator.num_processes),
            "global_batches_per_epoch": int(
                self.train_sampler.global_batches_per_epoch
            ),
            "global_step": int(self.global_step),
            "gradient_accumulation_steps": int(
                self.train_sampler.gradient_accumulation_steps
            ),
            "microbatches_per_process": int(
                self.train_sampler.microbatches_per_process
            ),
            "num_processes": int(self.train_sampler.num_processes),
            "optimizer_steps_per_epoch": int(
                self.train_sampler.optimizer_steps_per_epoch
            ),
            "schedule_fingerprint": self.train_sampler.schedule_fingerprint(
                self.epoch
            ),
            "uses_agent_count_batch_sampler": True,
        }
        if (
            self.n2_reload_proof_phase == "load"
            and int(self.train_sampler.resume_batch_offset)
            != int(self.batch_in_epoch)
        ):
            raise RuntimeError(
                "N=2 fresh reload sampler did not restore batch_in_epoch: "
                f"resume_offset={self.train_sampler.resume_batch_offset} "
                f"batch_in_epoch={self.batch_in_epoch}"
            )
        return cursor

    def _read_n2_reload_checkpoint_binding(
        self,
    ) -> tuple[dict[str, object], str]:
        binding_path = self._n2_reload_proof_dir() / "checkpoint-binding.json"
        binding, binding_sha256, _ = read_canonical_json(binding_path)
        _, _, candidate_sha256 = self._read_n2_terminal_candidate()
        if set(binding) != {
            "checkpoint",
            "global_step",
            "proof_attempt_id",
            "run_id",
            "schema_name",
            "schema_version",
            "terminal_arguments_sha256",
            "terminal_candidate_sha256",
            "world_size",
        }:
            raise ValueError("N=2 reload checkpoint binding fields mismatch")
        if (
            binding.get("schema_name")
            != "fastwam-action-only-n2-reload-checkpoint-binding"
            or binding.get("schema_version")
            != ACTION_ONLY_N2_RELOAD_PROOF_SCHEMA_VERSION
            or binding.get("run_id") != os.environ.get("RUN_ID", "")
            or binding.get("global_step") != ACTION_ONLY_N2_PAID_GATE_STEP
            or binding.get("world_size") != ACTION_ONLY_N2_1X8_WORLD_SIZE
            or binding.get("proof_attempt_id")
            != self._n2_reload_proof_attempt_id
            or binding.get("terminal_candidate_sha256") != candidate_sha256
        ):
            raise RuntimeError("N=2 reload checkpoint binding identity mismatch")
        require_sha256(
            binding.get("terminal_arguments_sha256", ""),
            label="N=2 reload binding terminal arguments SHA-256",
        )
        candidate, _, _ = self._read_n2_terminal_candidate()
        if binding.get("terminal_arguments_sha256") != candidate.get(
            "arguments_sha256"
        ):
            raise RuntimeError(
                "N=2 reload binding does not bind the staged terminal arguments"
            )
        checkpoint = binding.get("checkpoint")
        if not isinstance(checkpoint, dict) or checkpoint.get("global_step") != 1:
            raise RuntimeError("N=2 reload binding lacks the step-1 checkpoint")
        state = checkpoint.get("state")
        weights = checkpoint.get("weights")
        if (
            not isinstance(state, dict)
            or not isinstance(weights, dict)
            or state.get("root") != "checkpoints/state/step_000001"
            or weights.get("checkpoint")
            != "checkpoints/weights/step_000001.pt"
        ):
            raise RuntimeError("N=2 reload binding points at an unexpected checkpoint")
        return binding, binding_sha256

    def _read_n2_terminal_candidate(
        self,
    ) -> tuple[dict[str, object], dict[str, object], str]:
        candidate, candidate_sha256, _ = read_canonical_json(
            self._n2_reload_source_root() / ACTION_ONLY_N2_TERMINAL_CANDIDATE
        )
        if set(candidate) != {
            "arguments",
            "arguments_sha256",
            "run_id",
            "schema_name",
            "schema_version",
            "status",
        }:
            raise ValueError("N=2 terminal candidate fields mismatch")
        arguments = candidate.get("arguments")
        if not isinstance(arguments, dict):
            raise TypeError("N=2 terminal candidate arguments must be a mapping")
        if (
            candidate.get("schema_name")
            != "fastwam-action-only-n2-terminal-candidate"
            or candidate.get("schema_version") != 1
            or candidate.get("status") != "AWAITING_FRESH_RELOAD"
            or candidate.get("run_id") != os.environ.get("RUN_ID", "")
            or candidate.get("run_id") != arguments.get("run_id")
            or candidate.get("arguments_sha256")
            != canonical_json_sha256(arguments)
        ):
            raise RuntimeError("N=2 terminal candidate identity mismatch")
        expected = {
            "checkpoint_state_kind": "sparse_delta",
            "dataset_contract": self._dataset_run_contract,
            "dataset_contract_sha256": canonical_json_sha256(
                self._dataset_run_contract
            ),
            "evaluation_records": [],
            "expected_checkpoint_steps": [1],
            "expected_evaluation_steps": [],
            "formal_n4_fullmodel_gate": False,
            "max_steps": ACTION_ONLY_N2_PAID_GATE_STEP,
            "offline_eval_num_samples": 0,
            "run_id": os.environ.get("RUN_ID", ""),
            "run_profile": "paid_gate_1step",
            "trainable_scope": "action",
            "training_mode": "action_only_cache",
            "training_terminal_contract": ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT,
            "world_size": ACTION_ONLY_N2_1X8_WORLD_SIZE,
        }
        mismatches = {
            key: {"expected": value, "observed": arguments.get(key)}
            for key, value in expected.items()
            if arguments.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"N=2 terminal candidate contract mismatch: {mismatches}"
            )
        if arguments.get("code_commit") != (self._git_commit() or ""):
            raise RuntimeError("N=2 fresh reload code commit differs from save world")
        return candidate, arguments, candidate_sha256

    def _validate_action_only_n2_reload_load_contract(
        self, *, post_load: bool
    ) -> None:
        if self.n2_reload_proof_phase != "load":
            raise RuntimeError("N=2 reload load contract is only valid in load phase")
        scalar_contract = {
            "agent_action_token_budget": (self.agent_action_token_budget, 128),
            "batch_size": (self.batch_size, 1),
            "checkpoint_state_kind": (self.checkpoint_state_kind, "sparse_delta"),
            "eval_every": (self.eval_every, 0),
            "formal_n4_fullmodel_gate": (self.formal_n4_fullmodel_gate, False),
            "gradient_accumulation_steps": (
                self.gradient_accumulation_steps,
                4,
            ),
            "max_steps": (self.max_steps, 1),
            "mixed_precision": (self.mixed_precision, "bf16"),
            "offline_eval_num_samples": (self.offline_eval_num_samples, 0),
            "save_every": (self.save_every, 0),
            "trainable_scope": (self.trainable_scope, "action"),
            "world_size": (
                int(self.accelerator.num_processes),
                ACTION_ONLY_N2_1X8_WORLD_SIZE,
            ),
        }
        mismatches = {
            key: {"expected": expected, "observed": observed}
            for key, (observed, expected) in scalar_contract.items()
            if observed != expected
        }
        if mismatches:
            raise ValueError(f"N=2 fresh reload scalar contract mismatch: {mismatches}")
        if self.init_weights:
            raise ValueError("N=2 fresh reload must not apply init_weights")
        if any(
            (
                self.save_training_state_enabled,
                self.seal_training_state,
                self.save_final_checkpoint_enabled,
                self.seal_training_run,
            )
        ):
            raise ValueError("N=2 fresh reload must be read-only and non-sealing")
        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if plugin is None:
            raise RuntimeError("N=2 fresh reload requires DeepSpeed")
        zero_stage = plugin.deepspeed_config.get("zero_optimization", {}).get("stage")
        if zero_stage != 2:
            raise ValueError(f"N=2 fresh reload requires ZeRO stage 2, got {zero_stage!r}")
        if not self._uses_agent_count_batch_sampler:
            raise ValueError("N=2 fresh reload requires the variable-agent sampler")
        observed_counts = [int(value) for value in self.train_sampler.observed_agent_counts]
        batch_sizes = {
            int(key): int(value)
            for key, value in self.train_sampler.batch_size_by_agent_count.items()
        }
        if observed_counts != [2] or batch_sizes != {2: 2}:
            raise ValueError(
                "N=2 fresh reload sampler contract mismatch: "
                f"counts={observed_counts} batch_sizes={batch_sizes}"
            )
        model = self.accelerator.unwrap_model(self.model)
        if str(getattr(model, "training_mode", "")).strip().lower() != "action_only_cache":
            raise ValueError("N=2 fresh reload requires action_only_cache mode")
        for label, dataset in (("train", self.train_dataset), ("val", self.val_dataset)):
            if getattr(dataset, "load_future_video", True):
                raise ValueError(f"N=2 fresh reload {label} dataset loaded future video")

        source_root = self._n2_reload_source_root()
        load_output = resolved_unaliased_directory(
            self.output_dir, label="N=2 reload probe output"
        )
        if source_root == load_output:
            raise ValueError("N=2 fresh reload requires a distinct probe output root")
        expected_resume = resolved_unaliased_directory(
            source_root / "checkpoints" / "state" / "step_000001",
            label="N=2 paid reload state",
        )
        resume_path = Path(str(self.resume)).expanduser()
        if not resume_path.is_absolute():
            raise ValueError("N=2 fresh reload resume path must be absolute")
        resume_path = resolved_unaliased_directory(
            resume_path, label="N=2 fresh reload resume state"
        )
        if resume_path != expected_resume:
            raise RuntimeError(
                "N=2 fresh reload must resume the exact paid checkpoint state: "
                f"expected={expected_resume} observed={resume_path}"
            )
        _, arguments, _ = self._read_n2_terminal_candidate()
        binding, _ = self._read_n2_reload_checkpoint_binding()
        if not post_load:
            live_checkpoint = checkpoint_seal_descriptor(
                source_root,
                step=ACTION_ONLY_N2_PAID_GATE_STEP,
                rehash_weights=True,
                expected_checkpoint_state_kind="sparse_delta",
            )
            if live_checkpoint != binding["checkpoint"]:
                raise RuntimeError(
                    "N=2 fresh reload source checkpoint/state tree changed after "
                    "the save-world binding"
                )
        if post_load:
            restored_contract = {
                "batch_in_epoch": (self.batch_in_epoch, 4),
                "epoch": (self.epoch, 0),
                "global_step": (self.global_step, ACTION_ONLY_N2_PAID_GATE_STEP),
                "evaluation_records": (
                    self._evaluation_records,
                    arguments.get("evaluation_records"),
                ),
                "last_step_metrics": (
                    self._last_step_metrics,
                    arguments.get("last_step_metrics"),
                ),
            }
            restored_mismatches = {
                key: {"expected": expected, "observed": observed}
                for key, (observed, expected) in restored_contract.items()
                if observed != expected
            }
            if restored_mismatches:
                raise RuntimeError(
                    "N=2 fresh reload trainer-state mismatch: "
                    f"{restored_mismatches}"
                )
            cursor = self._n2_reload_sampler_cursor()
            if cursor["global_batch_offset"] != 32:
                raise RuntimeError("N=2 fresh reload did not restore sampler cursor 32")

    def publish_action_only_n2_reload_save_proof(self) -> None:
        if self.n2_reload_proof_phase != "save":
            raise RuntimeError("N=2 save proof is only valid in paid save phase")
        if (
            self.global_step != ACTION_ONLY_N2_PAID_GATE_STEP
            or int(self.accelerator.num_processes) != ACTION_ONLY_N2_1X8_WORLD_SIZE
            or self.accelerator.device.type != "cuda"
        ):
            raise RuntimeError("N=2 save proof requires CUDA step 1 on exactly 8 ranks")
        source_root = self._n2_reload_source_root()
        proof_dir = source_root / ACTION_ONLY_N2_RELOAD_PROOF_DIR
        binding_path = proof_dir / "checkpoint-binding.json"
        candidate, _, candidate_sha256 = self._read_n2_terminal_candidate()
        terminal_arguments_sha256 = require_sha256(
            candidate.get("arguments_sha256", ""),
            label="N=2 staged terminal arguments SHA-256",
        )
        if self.accelerator.is_main_process:
            if proof_dir.exists() or proof_dir.is_symlink():
                raise FileExistsError(f"N=2 reload proof directory already exists: {proof_dir}")
            proof_dir.mkdir(mode=0o700)
            (proof_dir / ACTION_ONLY_N2_RELOAD_LOAD_ATTEMPTS_DIR).mkdir(mode=0o700)
            checkpoint = checkpoint_seal_descriptor(
                source_root,
                step=ACTION_ONLY_N2_PAID_GATE_STEP,
                rehash_weights=True,
                expected_checkpoint_state_kind="sparse_delta",
            )
            publish_action_only_n2_reload_proof_record(
                source_root,
                relative_path=(
                    f"{ACTION_ONLY_N2_RELOAD_PROOF_DIR}/checkpoint-binding.json"
                ),
                payload={
                    "checkpoint": checkpoint,
                    "global_step": ACTION_ONLY_N2_PAID_GATE_STEP,
                    "proof_attempt_id": self._n2_reload_proof_attempt_id,
                    "run_id": os.environ.get("RUN_ID", ""),
                    "schema_name": (
                        "fastwam-action-only-n2-reload-checkpoint-binding"
                    ),
                    "schema_version": ACTION_ONLY_N2_RELOAD_PROOF_SCHEMA_VERSION,
                    "terminal_arguments_sha256": terminal_arguments_sha256,
                    "terminal_candidate_sha256": candidate_sha256,
                    "world_size": ACTION_ONLY_N2_1X8_WORLD_SIZE,
                },
            )
        else:
            self._wait_for_published_regular_file(
                binding_path, label="N=2 reload checkpoint binding"
            )
        binding, binding_sha256 = self._read_n2_reload_checkpoint_binding()
        fingerprints = self._n2_reload_state_fingerprints()
        sampler_cursor = self._n2_reload_sampler_cursor()
        rng_sample = next_rng_sample(self.accelerator.device)
        rank = int(self.accelerator.process_index)
        publish_action_only_n2_reload_proof_record(
            source_root,
            relative_path=(
                f"{ACTION_ONLY_N2_RELOAD_PROOF_DIR}/save-rank-{rank:05d}.json"
            ),
            payload={
                "checkpoint": binding["checkpoint"],
                "checkpoint_binding_sha256": binding_sha256,
                "fingerprints": fingerprints,
                "global_step": ACTION_ONLY_N2_PAID_GATE_STEP,
                "next_rng_sample": rng_sample,
                "phase": "save_after_sealed_checkpoint",
                "proof_attempt_id": self._n2_reload_proof_attempt_id,
                "process_nonce": self._n2_reload_process_nonce,
                "process_pid": self._n2_reload_process_pid,
                "process_start_ticks": self._n2_reload_process_start_ticks,
                "rank": rank,
                "run_id": os.environ.get("RUN_ID", ""),
                "sampler_cursor": sampler_cursor,
                "schema_name": "fastwam-action-only-n2-reload-save-proof",
                "schema_version": ACTION_ONLY_N2_RELOAD_PROOF_SCHEMA_VERSION,
                "terminal_arguments_sha256": binding[
                    "terminal_arguments_sha256"
                ],
                "terminal_candidate_sha256": binding[
                    "terminal_candidate_sha256"
                ],
                "world_size": ACTION_ONLY_N2_1X8_WORLD_SIZE,
            },
        )
        self.accelerator.wait_for_everyone()

    def publish_action_only_n2_reload_load_proof(self) -> None:
        if self.n2_reload_proof_phase != "load":
            raise RuntimeError("N=2 load proof is only valid in fresh load phase")
        if (
            self.global_step != ACTION_ONLY_N2_PAID_GATE_STEP
            or int(self.accelerator.num_processes) != ACTION_ONLY_N2_1X8_WORLD_SIZE
            or self.accelerator.device.type != "cuda"
            or self._n2_reload_pre_load_fingerprints is None
        ):
            raise RuntimeError(
                "N=2 load proof requires a captured pre-load state and restored CUDA step 1"
            )
        binding, binding_sha256 = self._read_n2_reload_checkpoint_binding()
        rank = int(self.accelerator.process_index)
        save_path = self._n2_reload_proof_dir() / f"save-rank-{rank:05d}.json"
        saved, _, _ = read_canonical_json(save_path)
        if (
            saved.get("rank") != rank
            or saved.get("world_size") != ACTION_ONLY_N2_1X8_WORLD_SIZE
            or saved.get("run_id") != os.environ.get("RUN_ID", "")
            or saved.get("checkpoint_binding_sha256") != binding_sha256
            or saved.get("checkpoint") != binding["checkpoint"]
            or saved.get("proof_attempt_id")
            != self._n2_reload_proof_attempt_id
            or saved.get("terminal_arguments_sha256")
            != binding["terminal_arguments_sha256"]
            or saved.get("terminal_candidate_sha256")
            != binding["terminal_candidate_sha256"]
        ):
            raise RuntimeError(f"N=2 save proof identity mismatch on rank {rank}")
        restored = self._n2_reload_state_fingerprints()
        sampler_cursor = self._n2_reload_sampler_cursor()
        observed_next_rng_sample = next_rng_sample(self.accelerator.device)
        expected_fingerprints = saved.get("fingerprints", {})
        checks = {
            "checkpoint_binding": saved.get("checkpoint") == binding["checkpoint"],
            "fresh_process": (
                saved.get("process_nonce") != self._n2_reload_process_nonce
                and (saved.get("process_pid"), saved.get("process_start_ticks"))
                != (
                    self._n2_reload_process_pid,
                    self._n2_reload_process_start_ticks,
                )
            ),
            "global_step": restored.get("global_step")
            == expected_fingerprints.get("global_step"),
            "model": restored.get("model") == expected_fingerprints.get("model"),
            "next_rng_sample": observed_next_rng_sample
            == saved.get("next_rng_sample"),
            "optimizer": restored.get("optimizer")
            == expected_fingerprints.get("optimizer"),
            "pre_load_was_distinct": any(
                self._n2_reload_pre_load_fingerprints.get(key)
                != expected_fingerprints.get(key)
                for key in ("model", "optimizer")
            ),
            "rng": restored.get("rng") == expected_fingerprints.get("rng"),
            "sampler_cursor": sampler_cursor == saved.get("sampler_cursor"),
            "scheduler": restored.get("scheduler")
            == expected_fingerprints.get("scheduler"),
            "terminal_candidate": (
                saved.get("terminal_arguments_sha256")
                == binding["terminal_arguments_sha256"]
                and saved.get("terminal_candidate_sha256")
                == binding["terminal_candidate_sha256"]
            ),
        }
        if restored != expected_fingerprints:
            checks["model"] = False
        if not all(checks.values()):
            raise RuntimeError(f"N=2 fresh reload mismatch on rank {rank}: {checks}")
        load_attempt_id = self._n2_reload_load_attempt_id
        if load_attempt_id is None:
            raise RuntimeError("N=2 fresh reload lacks a load attempt id")
        load_attempt_dir = (
            self._n2_reload_proof_dir()
            / ACTION_ONLY_N2_RELOAD_LOAD_ATTEMPTS_DIR
            / load_attempt_id
        )
        if self.accelerator.is_main_process:
            if load_attempt_dir.exists() or load_attempt_dir.is_symlink():
                raise FileExistsError(
                    "N=2 reload load-attempt directory already exists: "
                    f"{load_attempt_dir}"
                )
            load_attempt_dir.mkdir(mode=0o700)
        self.accelerator.wait_for_everyone()
        publish_action_only_n2_reload_proof_record(
            self._n2_reload_source_root(),
            relative_path=(
                f"{ACTION_ONLY_N2_RELOAD_PROOF_DIR}/"
                f"{ACTION_ONLY_N2_RELOAD_LOAD_ATTEMPTS_DIR}/{load_attempt_id}/"
                f"load-rank-{rank:05d}.json"
            ),
            payload={
                "checkpoint": binding["checkpoint"],
                "checkpoint_binding_sha256": binding_sha256,
                "checks": checks,
                "fingerprints": restored,
                "global_step": ACTION_ONLY_N2_PAID_GATE_STEP,
                "load_attempt_id": load_attempt_id,
                "next_rng_sample": observed_next_rng_sample,
                "phase": "load_fresh_process",
                "pre_load_fingerprints": self._n2_reload_pre_load_fingerprints,
                "proof_attempt_id": self._n2_reload_proof_attempt_id,
                "process_nonce": self._n2_reload_process_nonce,
                "process_pid": self._n2_reload_process_pid,
                "process_start_ticks": self._n2_reload_process_start_ticks,
                "rank": rank,
                "run_id": os.environ.get("RUN_ID", ""),
                "sampler_cursor": sampler_cursor,
                "schema_name": "fastwam-action-only-n2-reload-load-proof",
                "schema_version": ACTION_ONLY_N2_RELOAD_PROOF_SCHEMA_VERSION,
                "terminal_arguments_sha256": binding[
                    "terminal_arguments_sha256"
                ],
                "terminal_candidate_sha256": binding[
                    "terminal_candidate_sha256"
                ],
                "world_size": ACTION_ONLY_N2_1X8_WORLD_SIZE,
            },
        )
        self.accelerator.wait_for_everyone()
        source_root = self._n2_reload_source_root()
        if self.accelerator.is_main_process:
            # A failed fresh-reload attempt is attempt-local runtime evidence;
            # it must never poison the source training run with TRAINING.FAILED.
            publish_action_only_n2_reload_attempt_commit(
                source_root,
                run_id=os.environ.get("RUN_ID", ""),
                checkpoint=binding["checkpoint"],
                terminal_arguments_sha256=binding["terminal_arguments_sha256"],
                load_attempt_id=load_attempt_id,
            )
            finalize_action_only_n2_paid_gate(source_root)
        self.accelerator.wait_for_everyone()

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
            steps.extend(range(self.save_every, self.max_steps + 1, self.save_every))
        if self.save_final_checkpoint_enabled and self.max_steps not in steps:
            steps.append(self.max_steps)
        terminal_arguments = {
            "run_id": os.environ.get("RUN_ID", ""),
            "code_commit": self._git_commit() or "",
            "config_relative_path": config_relative_path,
            "config_sha256": config_sha256,
            "max_steps": self.max_steps,
            "expected_checkpoint_steps": steps,
            "expected_evaluation_steps": (
                []
                if self.eval_every <= 0
                else list(
                    range(
                        self.eval_every,
                        self.max_steps + 1,
                        self.eval_every,
                    )
                )
            ),
            "world_size": int(self.accelerator.num_processes),
            "last_step_metrics": self._last_step_metrics,
            "evaluation_records": self._evaluation_records,
            "training_mode": training_mode,
            "dataset_contract_sha256": canonical_json_sha256(
                self._dataset_run_contract
            ),
            "authorization_gate_complete_sha256": os.environ.get(
                "FASTWAM_N4_FULLMODEL_GATE_COMPLETE_SHA256", ""
            ),
            "rehash_weights": self.terminal_rehash_weights,
            "training_terminal_contract": self.training_terminal_contract,
            "formal_n4_fullmodel_gate": self.formal_n4_fullmodel_gate,
            "checkpoint_state_kind": self.checkpoint_state_kind,
            "trainable_scope": self.trainable_scope,
            "dataset_contract": self._dataset_run_contract,
            "task_scope_receipt_relative_path": (
                self.training_task_scope_receipt or ""
            ),
            "effective_patched_tree": os.environ.get(
                "FASTWAM_EFFECTIVE_PATCHED_TREE", ""
            ),
            "request_sha256": os.environ.get("FASTWAM_REQUEST_SHA256", ""),
            "init_checkpoint_sha256": os.environ.get(
                "FASTWAM_INIT_CHECKPOINT_SHA256", ""
            ),
            "offline_eval_num_samples": self.offline_eval_num_samples,
            "run_profile": self.training_run_profile or "",
        }
        self.accelerator.wait_for_everyone()
        complete_path = Path(self.output_dir) / "TRAINING.COMPLETE"
        failure_path = Path(self.output_dir) / "TRAINING.FAILED.json"
        success_path = complete_path
        success_label = "run-level training terminal seal"
        if self.n2_reload_proof_phase == "save":
            success_path = Path(self.output_dir) / ACTION_ONLY_N2_TERMINAL_CANDIDATE
            success_label = "N=2 paid-gate terminal candidate"
        if self.accelerator.is_main_process:
            try:
                if self.n2_reload_proof_phase == "save":
                    publish_action_only_n2_terminal_candidate(
                        self.output_dir,
                        terminal_arguments=terminal_arguments,
                    )
                else:
                    publish_training_terminal_seal(
                        self.output_dir,
                        **terminal_arguments,
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
                success_path=success_path,
                failure_path=failure_path,
                label=success_label,
            )
        self.accelerator.wait_for_everyone()
        if self.n2_reload_proof_phase == "save":
            # The checkpoint and its full state-tree seal already exist.  Rank
            # zero has now exclusively staged the terminal candidate, so all
            # ranks can bind the same candidate/arguments digests into the
            # checkpoint binding and their save-world proofs.
            self.publish_action_only_n2_reload_save_proof()
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
            )
            schedule_identity = (
                {
                    "integrity_mode": "metadata_no_hash",
                    "epoch": int(self.train_sampler.epoch),
                    "batch_count": len(
                        self.train_sampler.global_epoch_batches(
                            self.train_sampler.epoch
                        )
                    ),
                }
                if self.artifact_integrity_mode == "metadata_no_hash"
                else self.train_sampler.schedule_fingerprint()
            )
            logger.info(
                "Using hierarchical task/count-balanced batching: counts=%s tasks_by_count=%s "
                "batch_sizes=%s token_budget=%s global_batches=%d local_microbatches=%d "
                "optimizer_steps=%d schedule_identity=%s",
                self.train_sampler.observed_agent_counts,
                self.train_sampler.tasks_by_agent_count,
                self.train_sampler.batch_size_by_agent_count,
                self.train_sampler.agent_action_token_budget,
                self.train_sampler.global_batches_per_epoch,
                self.train_sampler.microbatches_per_process,
                self.train_sampler.optimizer_steps_per_epoch,
                schedule_identity,
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

    @classmethod
    def _dataset_contract(cls, dataset):
        """Build a stable scientific identity without hashing the full HDF5 corpus."""

        if dataset is None:
            return None
        contract = {
            "class": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
            "length": int(len(dataset)),
        }
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
            "gaussian_fallback_projection",
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
            "required_tasks",
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
            contract["source_inventory_sha256"] = cls._canonical_json_sha256(
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
            contract["window_index_sha256"] = cls._canonical_json_sha256(
                normalized_entries
            )

        stats_path = getattr(dataset, "_stats_path", None)
        if stats_path is not None:
            stats_path = Path(stats_path).expanduser().resolve()
            contract["normalization"] = {
                "path": str(stats_path),
                "sha256": cls._sha256_file(stats_path),
                "schema": getattr(dataset, "_stats_metadata", None),
            }
        return contract

    @staticmethod
    def _metadata_value(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): Wan22Trainer._metadata_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [Wan22Trainer._metadata_value(item) for item in value]
        raise TypeError(
            "metadata_no_hash contract encountered an unsupported value: "
            f"{type(value)}"
        )

    @classmethod
    def _dataset_contract_metadata_no_hash(cls, dataset):
        """Record the complete sorted dataset inventory without a digest."""

        if dataset is None:
            return None
        contract = {
            "integrity_mode": "metadata_no_hash",
            "class": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
            "length": int(len(dataset)),
        }
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
            "gaussian_fallback_projection",
            "gaussian_channels",
            "context_len",
        )
        for name in scalar_attributes:
            if hasattr(dataset, name):
                value = getattr(dataset, name)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    contract[name] = value
        for name in (
            "video_size",
            "video_indices",
            "required_agent_counts",
            "required_tasks",
            "gaussian_size",
        ):
            if hasattr(dataset, name):
                value = getattr(dataset, name)
                contract[name] = None if value is None else cls._metadata_value(value)

        for name in ("gaussian_cache_dir", "gaussian_fallback_cache_dir"):
            if hasattr(dataset, name):
                value = getattr(dataset, name)
                contract[name] = None if value is None else str(Path(value).resolve())
        preflight = getattr(dataset, "_gaussian_preflight", None)
        if preflight is not None:
            contract["gaussian_preflight"] = cls._metadata_value(preflight)

        root = getattr(dataset, "root_dir", None)
        if root is not None:
            root_path = Path(root).expanduser().resolve(strict=True)
            if not root_path.is_dir():
                raise RuntimeError(f"Dataset root is not a directory: {root_path}")
            inventory = []
            for source_path in sorted(root_path.rglob("*.h5")):
                if source_path.is_symlink():
                    raise RuntimeError(
                        f"Dataset source symlink is forbidden in metadata_no_hash: {source_path}"
                    )
                metadata = nohash_regular_file_metadata(source_path)
                inventory.append(
                    {
                        "path": source_path.relative_to(root_path).as_posix(),
                        "bytes": metadata["bytes"],
                        "mtime_ns": metadata["mtime_ns"],
                        "dev": metadata["dev"],
                        "ino": metadata["ino"],
                        "mode": metadata["mode"],
                    }
                )
            contract["root_dir"] = str(root_path)
            contract["source_inventory"] = inventory

        entries = getattr(dataset, "entries", None)
        if entries is not None:
            contract["window_index"] = [
                {
                    str(key): cls._metadata_value(value)
                    for key, value in sorted(entry.items())
                    if key != "path"
                }
                for entry in entries
            ]

        stats_path = getattr(dataset, "_stats_path", None)
        if stats_path is not None:
            metadata = nohash_regular_file_metadata(stats_path)
            contract["normalization"] = {
                "file": metadata,
                "schema": cls._metadata_value(
                    getattr(dataset, "_stats_metadata", None)
                ),
            }
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
            "init_weights",
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
            "allow_legacy_resume",
            "process_group_timeout_seconds",
            "checkpoint_io_timeout_seconds",
            "wandb",
            "hydra",
        ):
            resolved.pop(key, None)
        if getattr(self, "n2_reload_proof_phase", None) == "load":
            # The load-world config is intentionally non-sealing, but its
            # restored state must still match the exact authorization values
            # embedded in the save-world contract.  Normalize only this
            # explicitly unauthoritative probe from the staged candidate.
            _, arguments, _ = self._read_n2_terminal_candidate()
            resolved["training_terminal_contract"] = arguments[
                "training_terminal_contract"
            ]
            resolved["training_run_profile"] = arguments["run_profile"]
            resolved["training_task_scope_receipt"] = arguments[
                "task_scope_receipt_relative_path"
            ]
        return resolved

    @classmethod
    def _drop_digest_named_fields(cls, value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                lowered = str(key).lower()
                if any(token in lowered for token in ("sha", "hash", "digest", "checksum", "md5")):
                    continue
                result[str(key)] = cls._drop_digest_named_fields(item)
            return result
        if isinstance(value, list):
            return [cls._drop_digest_named_fields(item) for item in value]
        return value

    def _training_state_contract_metadata_no_hash(self) -> dict:
        model = self.accelerator.unwrap_model(self.model)
        architecture = None
        architecture_builder = getattr(model, "_multi_robot_architecture_metadata", None)
        if callable(architecture_builder):
            architecture = self._metadata_value(architecture_builder())
        trainable = [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
            }
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        config_contract = self._drop_digest_named_fields(
            self._resolved_config_contract()
        )
        base_checkpoint = None
        if self.checkpoint_state_kind == "sparse_delta":
            candidate = getattr(model, "_loaded_base_checkpoint_descriptor", None)
            if isinstance(candidate, dict):
                base_checkpoint = self._drop_digest_named_fields(
                    self._metadata_value(candidate)
                )
            else:
                base_checkpoint = getattr(
                    self, "_resume_base_checkpoint_provenance", None
                )
            is_full_state_resume = bool(self.resume) and Path(
                str(self.resume)
            ).is_dir()
            if not isinstance(base_checkpoint, dict) and not is_full_state_resume:
                raise RuntimeError(
                    "metadata_no_hash sparse checkpoint requires a base checkpoint "
                    "metadata descriptor"
                )
        return {
            "contract_version": 2,
            "integrity_mode": "metadata_no_hash",
            "state_kind": "accelerate_full_state",
            "treatment": {
                "training_mode": getattr(model, "training_mode", None),
                "trainable_scope": self.trainable_scope,
                "checkpoint_state_kind": self.checkpoint_state_kind,
                "video_gen": getattr(model, "training_mode", None) == "joint",
                "hub": None if architecture is None else architecture.get("hub_enabled"),
                "gaussian": None if architecture is None else architecture.get("enable_gaussian"),
            },
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
                "warmup_steps": int(self.max_steps * 0.05),
                "batch_size": self.batch_size,
                "agent_action_token_budget": self.agent_action_token_budget,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "world_size": int(self.accelerator.num_processes),
                "mixed_precision": self.mixed_precision,
                "max_grad_norm": self.max_grad_norm,
                "seed": self.seed,
            },
            "resolved_config": self._metadata_value(config_contract),
            "code_commit": self._git_commit(),
        }

    def _training_state_contract(self) -> dict:
        if self.artifact_integrity_mode == "metadata_no_hash":
            return self._training_state_contract_metadata_no_hash()
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
        return {
            "contract_version": 1,
            "state_kind": "accelerate_full_state",
            "treatment": treatment,
            "multi_robot_architecture": architecture,
            "trainable_parameters": trainable,
            "trainable_parameters_sha256": self._canonical_json_sha256(trainable),
            "base_checkpoint": base_checkpoint,
            "dataset": self._dataset_run_contract,
            "optimization": {
                "optimizer": "torch.optim.AdamW",
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "betas": [0.9, 0.95],
                "lr_scheduler_type": str(self.cfg.lr_scheduler_type),
                "max_steps": int(self.max_steps),
                "warmup_steps": int(self.max_steps * 0.05),
                "batch_size": self.batch_size,
                "agent_action_token_budget": self.agent_action_token_budget,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "world_size": int(self.accelerator.num_processes),
                "mixed_precision": self.mixed_precision,
                "max_grad_norm": self.max_grad_norm,
                "seed": self.seed,
            },
            "resolved_config_sha256": self._canonical_json_sha256(config_contract),
            "code_commit": self._git_commit(),
        }

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
                []
                if self.eval_every <= 0
                else list(range(self.eval_every, saved_step + 1, self.eval_every))
            )
            evaluation_contract: dict[str, object] = {}
            if (
                self.training_terminal_contract
                == ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT
            ):
                evidence = getattr(self, "_action_only_n2_terminal_evidence", None)
                if not isinstance(evidence, dict):
                    raise RuntimeError(
                        "N=2 formal resume evidence lacks the validated terminal contract"
                    )
                task_scope = evidence.get("task_scope")
                if not isinstance(task_scope, dict):
                    raise RuntimeError(
                        "N=2 formal resume evidence lacks the validated task scope"
                    )
                evaluation_contract = {
                    "expected_offline_samples": self.offline_eval_num_samples,
                    "expected_offline_agent_counts": task_scope.get(
                        "required_agent_counts", []
                    ),
                    "expected_offline_tasks": task_scope.get("required_tasks", []),
                }
            if not expected_steps:
                if evaluations:
                    raise RuntimeError(
                        "formal eval_every=0 requires evaluation_records=[] in "
                        f"trainer state: {state_file}"
                    )
                evaluations = []
            else:
                evaluations = normalize_formal_evaluation_records(
                    evaluations,
                    expected_steps=expected_steps,
                    training_mode=training_mode,
                    **evaluation_contract,
                )
        else:
            evaluations = [dict(record) for record in evaluations]
        return dict(last_metrics), evaluations

    def _restore_base_checkpoint_provenance(self, descriptor, *, state_file: Path) -> None:
        if self.artifact_integrity_mode == "metadata_no_hash":
            self._restore_base_checkpoint_metadata_no_hash(
                descriptor, state_file=state_file
            )
            return
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

    def _restore_base_checkpoint_metadata_no_hash(
        self, descriptor, *, state_file: Path
    ) -> None:
        if not isinstance(descriptor, dict):
            raise RuntimeError(
                f"Invalid metadata_no_hash base descriptor: {state_file}"
            )
        allowed = {"path", "role", "integrity_mode", "stat"}
        if set(descriptor) != allowed:
            raise RuntimeError(
                "metadata_no_hash base descriptor has unexpected fields: "
                f"{sorted(set(descriptor) - allowed)}"
            )
        raw_path = descriptor.get("path")
        expected_stat = descriptor.get("stat")
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or descriptor.get("role") != "base_dependency"
            or descriptor.get("integrity_mode") != "metadata_no_hash"
            or not isinstance(expected_stat, dict)
        ):
            raise RuntimeError(
                f"Invalid metadata_no_hash base descriptor: {state_file}"
            )
        expected_stat_keys = {"bytes", "mtime_ns", "dev", "ino", "mode"}
        if set(expected_stat) != expected_stat_keys or not all(
            isinstance(expected_stat[key], int) for key in expected_stat_keys
        ):
            raise RuntimeError(
                f"Invalid metadata_no_hash base stat descriptor: {state_file}"
            )

        verification = None
        if self.accelerator.is_main_process:
            try:
                observed = nohash_regular_file_metadata(raw_path)
                observed_stat = {
                    key: observed[key] for key in sorted(expected_stat_keys)
                }
                if observed_stat != {
                    key: expected_stat[key] for key in sorted(expected_stat_keys)
                }:
                    raise RuntimeError(
                        "Base checkpoint metadata changed before full-state resume: "
                        f"expected={expected_stat} observed={observed_stat}"
                    )
                verification = {"ok": True, "error": None}
            except Exception as error:
                verification = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            shared = [verification]
            torch.distributed.broadcast_object_list(shared, src=0)
            verification = shared[0]
        if not isinstance(verification, dict) or verification.get("ok") is not True:
            detail = None if not isinstance(verification, dict) else verification.get("error")
            raise RuntimeError(
                "metadata_no_hash base checkpoint verification failed before "
                f"accelerator.load_state: {detail}"
            )

        normalized = {
            "path": str(Path(raw_path).expanduser().resolve(strict=True)),
            "role": "base_dependency",
            "integrity_mode": "metadata_no_hash",
            "stat": {key: expected_stat[key] for key in sorted(expected_stat_keys)},
        }
        self._resume_base_checkpoint_provenance = normalized
        model = self.accelerator.unwrap_model(self.model)
        for name, value in (
            ("_loaded_base_checkpoint", normalized["path"]),
            ("_loaded_base_checkpoint_descriptor", normalized),
            ("_loaded_base_checkpoint_can_restore_sparse", True),
        ):
            if hasattr(model, name):
                setattr(model, name, value)

    def _load_weight_checkpoint_before_prepare(self):
        """Load resume or initialization weights before ZeRO master construction."""

        init_weights = getattr(self, "init_weights", None)
        if init_weights:
            init_path = Path(str(init_weights))
            if not init_path.exists():
                raise FileNotFoundError(
                    f"Initialization checkpoint not found: {init_weights}"
                )
            if not init_path.is_file():
                raise ValueError(
                    "Initialization checkpoint must be a self-contained weights file: "
                    f"{init_weights}"
                )
            load_initialization = getattr(
                self.model,
                "load_initialization_checkpoint",
                None,
            )
            if not callable(load_initialization):
                raise TypeError(
                    f"Model {type(self.model).__name__} does not support strict "
                    "cross-treatment initialization"
                )
            logger.info(
                "Loading full initialization weights before optimizer/DeepSpeed "
                "construction: %s",
                init_weights,
            )
            expected_sha256 = None
            if (
                getattr(self, "training_terminal_contract", None)
                == ACTION_ONLY_N2_1X8_TERMINAL_CONTRACT
            ):
                expected_sha256 = os.environ.get(
                    "FASTWAM_INIT_CHECKPOINT_SHA256", ""
                ).strip().lower()
                if (
                    len(expected_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in expected_sha256
                    )
                ):
                    raise ValueError(
                        "Formal N=2 initialization requires a lowercase 64-hex "
                        "FASTWAM_INIT_CHECKPOINT_SHA256"
                    )
            load_parameters = inspect.signature(load_initialization).parameters
            load_kwargs = {"expected_sha256": expected_sha256}
            if self.artifact_integrity_mode == "metadata_no_hash":
                if "checkpoint_integrity_mode" not in load_parameters:
                    raise RuntimeError(
                        f"Model {type(self.model).__name__} does not support "
                        "metadata_no_hash initialization"
                    )
                load_kwargs["checkpoint_integrity_mode"] = "metadata_no_hash"
            load_initialization(str(init_path), **load_kwargs)
            self._weight_checkpoint_loaded_before_prepare = True
            logger.warning(
                "Loaded full initialization weights only; optimizer/scheduler/step "
                "are intentionally initialized from scratch."
            )
            return

        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        if not resume_path.is_file():
            raise ValueError(
                f"Resume path must be a checkpoint file or state directory: {resume}"
            )
        logger.info(
            "Loading weight checkpoint before optimizer/DeepSpeed initialization: %s",
            resume,
        )
        self.model.load_checkpoint(str(resume_path), optimizer=None)
        self._weight_checkpoint_loaded_before_prepare = True
        logger.warning(
            "Loaded .pt weights before ZeRO master construction; "
            "optimizer/scheduler/step are intentionally not restored."
        )

    def _resume_training_state_after_prepare(self):
        """Restore prepared full state, or confirm an earlier file preload."""

        init_weights = getattr(self, "init_weights", None)
        if init_weights:
            if not self._weight_checkpoint_loaded_before_prepare:
                raise RuntimeError(
                    "Initialization checkpoint reached post-prepare without being "
                    f"loaded before optimizer construction: {init_weights}"
                )
            logger.info(
                "Initialization weights were loaded before prepare; no post-prepare "
                "reload: %s",
                init_weights,
            )
            return

        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            if getattr(self, "n2_reload_proof_phase", None) == "load":
                self._validate_action_only_n2_reload_load_contract(post_load=False)
                self._n2_reload_pre_load_fingerprints = (
                    self._n2_reload_state_fingerprints(
                        require_optimizer_state=False
                    )
                )
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
        for key in ("agent_geometry", "agent_ids", "agent_gaussian", "action_is_pad"):
            if key in batched and batched[key].shape[1] != num_agents:
                raise ValueError(f"{key} and action agent axes differ in eval sample")
        if "action_is_pad" in batched and batched["action_is_pad"].shape[2] != horizon:
            raise ValueError("action_is_pad and action horizon axes differ in eval sample")

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
        if self.artifact_integrity_mode == "metadata_no_hash":
            return self._save_weights_checkpoint_metadata_no_hash(step_tag)
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
            expected_sha256 = self._sha256_regular_file(staged)

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
            actual_bytes = destination.stat().st_size
            actual_sha256 = self._sha256_regular_file(destination)
            if (actual_bytes, actual_sha256) != (expected_bytes, expected_sha256):
                raise RuntimeError(
                    "Published weights checkpoint failed strong readback: "
                    f"expected=({expected_bytes},{expected_sha256}) "
                    f"actual=({actual_bytes},{actual_sha256}) path={destination}"
                )

            manifest = {
                "schema_name": "fastwam-weights-checkpoint",
                "schema_version": 1,
                "filename": destination.name,
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "global_step": int(self.global_step),
                "checkpoint_state_kind": self.checkpoint_state_kind,
            }
            manifest_bytes = (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            self._publish_exclusive_bytes(manifest_path, manifest_bytes)
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            complete_bytes = (
                json.dumps(
                    {
                        "schema_name": "fastwam-weights-checkpoint-complete",
                        "schema_version": 1,
                        "manifest_filename": manifest_path.name,
                        "manifest_sha256": manifest_sha256,
                        "checkpoint_sha256": actual_sha256,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            self._publish_exclusive_bytes(complete_path, complete_bytes)
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

    def _save_weights_checkpoint_metadata_no_hash(self, step_tag: str):
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
            os.environ.get("FASTWAM_WEIGHT_STAGING_DIR", "/tmp/fastwam-weight-staging")
        ).expanduser().resolve()
        staging_root.mkdir(parents=True, exist_ok=True)
        staged = staging_root / f".{step_tag}.rank0.{os.getpid()}.{time.time_ns()}.pt"
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
                if "checkpoint_integrity_mode" not in save_parameters:
                    raise RuntimeError(
                        f"Model {type(model).__name__} does not support "
                        "metadata_no_hash checkpoint publication"
                    )
                model.save_checkpoint(
                    staged,
                    optimizer=None,
                    step=self.global_step,
                    checkpoint_state_kind=self.checkpoint_state_kind,
                    checkpoint_integrity_mode="metadata_no_hash",
                )
            staged_metadata = nohash_regular_file_metadata(staged)
            destination_metadata = nohash_copy_exclusive_and_compare(
                staged, destination
            )
            if staged_metadata["bytes"] != destination_metadata["bytes"]:
                raise RuntimeError(
                    f"Checkpoint byte count changed during publication: {destination}"
                )
            manifest = {
                "schema_name": "fastwam-weights-checkpoint-metadata-no-hash",
                "schema_version": 1,
                "integrity_mode": "metadata_no_hash",
                "filename": destination.name,
                "file": destination_metadata,
                "global_step": int(self.global_step),
                "checkpoint_state_kind": self.checkpoint_state_kind,
            }
            manifest_metadata = nohash_publish_exclusive_json(
                manifest_path, manifest
            )
            nohash_publish_exclusive_json(
                complete_path,
                {
                    "schema_name": "fastwam-weights-checkpoint-complete-metadata-no-hash",
                    "schema_version": 1,
                    "integrity_mode": "metadata_no_hash",
                    "manifest_filename": manifest_path.name,
                    "manifest_file": manifest_metadata,
                    "checkpoint_filename": destination.name,
                    "checkpoint_file": destination_metadata,
                },
            )
            self._validate_weights_checkpoint_metadata_no_hash(step_tag)
            logger.info(
                "Published metadata_no_hash weights checkpoint: path=%s bytes=%d manifest=%s",
                destination,
                destination_metadata["bytes"],
                manifest_path,
            )
            return str(destination)
        except BaseException:
            if not complete_path.exists():
                manifest_path.unlink(missing_ok=True)
                destination.unlink(missing_ok=True)
            raise
        finally:
            staged.unlink(missing_ok=True)

    def _validate_weights_checkpoint_metadata_no_hash(self, step_tag: str) -> None:
        destination = Path(self.weights_dir) / f"{step_tag}.pt"
        manifest_path = destination.with_name(f"{destination.name}.manifest.json")
        complete_path = destination.with_name(f"{destination.name}.COMPLETE")
        complete, _ = nohash_read_json(complete_path)
        manifest, manifest_metadata = nohash_read_json(manifest_path)
        checkpoint_metadata = nohash_regular_file_metadata(destination)
        if not isinstance(complete, dict) or not isinstance(manifest, dict):
            raise RuntimeError(f"Invalid metadata_no_hash checkpoint metadata: {destination}")
        expected_complete = {
            "schema_name": "fastwam-weights-checkpoint-complete-metadata-no-hash",
            "schema_version": 1,
            "integrity_mode": "metadata_no_hash",
            "manifest_filename": manifest_path.name,
            "manifest_file": manifest_metadata,
            "checkpoint_filename": destination.name,
            "checkpoint_file": checkpoint_metadata,
        }
        if complete != expected_complete:
            raise RuntimeError(
                f"metadata_no_hash COMPLETE marker mismatch: {complete_path}"
            )
        if (
            manifest.get("schema_name")
            != "fastwam-weights-checkpoint-metadata-no-hash"
            or manifest.get("integrity_mode") != "metadata_no_hash"
            or manifest.get("filename") != destination.name
            or manifest.get("file") != checkpoint_metadata
            or manifest.get("global_step") != int(self.global_step)
            or manifest.get("checkpoint_state_kind") != self.checkpoint_state_kind
        ):
            raise RuntimeError(
                f"metadata_no_hash checkpoint manifest mismatch: {manifest_path}"
            )

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
        if self.artifact_integrity_mode == "metadata_no_hash":
            nohash_publish_exclusive_json(state_file, payload)
        else:
            publish_exclusive_json(state_file, payload)

    def _data_schedule_contract(self, epoch: int) -> dict:
        common = {
            "agent_action_token_budget": self.train_sampler.agent_action_token_budget,
            "gradient_accumulation_steps": self.train_sampler.gradient_accumulation_steps,
            "num_processes": self.train_sampler.num_processes,
            "global_batches_per_epoch": self.train_sampler.global_batches_per_epoch,
            "optimizer_steps_per_epoch": self.train_sampler.optimizer_steps_per_epoch,
        }
        if self.artifact_integrity_mode == "metadata_no_hash":
            return {
                "integrity_mode": "metadata_no_hash",
                "epoch": int(epoch),
                "seed": int(self.train_sampler.seed),
                "batches": self.train_sampler.global_epoch_batches(int(epoch)),
                **common,
            }
        return {
            "fingerprint": self.train_sampler.schedule_fingerprint(int(epoch)),
            **common,
        }

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
        if self.artifact_integrity_mode == "metadata_no_hash":
            self._validate_weights_checkpoint_metadata_no_hash(step_tag)
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

    def _should_pause_after_recovery_gate_checkpoint(
        self, *, checkpoint_saved_this_step: bool
    ) -> bool:
        """Return whether this completed checkpoint is the recovery-gate pause."""
        stop_step = getattr(
            self, "recovery_gate_stop_after_checkpoint_step", None
        )
        return (
            checkpoint_saved_this_step
            and stop_step is not None
            and self.global_step == stop_step
        )

    def load_training_state(self, state_dir: str):
        recovery_receipt_target = self._recovery_load_receipt_target()
        state_directory = Path(state_dir).expanduser()
        if recovery_receipt_target is not None:
            if state_directory.is_symlink() or not state_directory.is_dir():
                raise RuntimeError(
                    "metadata_no_hash recovery receipt requires a non-linked "
                    f"state directory: {state_directory}"
                )
            state_directory = state_directory.resolve(strict=True)
        state_file = Path(state_dir) / "trainer_state.json"
        payload = None
        state_file_metadata = None
        restored_last_step_metrics: dict[str, object] = {}
        restored_evaluation_records: list[dict[str, object]] = []
        try:
            if self.artifact_integrity_mode == "metadata_no_hash":
                payload, state_file_metadata = nohash_read_json(state_file)
            else:
                payload, _, _ = read_canonical_json(state_file)
        except FileNotFoundError:
            payload = None
        if payload is not None:
            if not isinstance(payload, dict):
                raise TypeError(f"Trainer state metadata must be a mapping: {state_file}")
            self._validate_training_state_contract(payload, state_file=state_file)
            if self.artifact_integrity_mode == "metadata_no_hash":
                self._validate_data_schedule_before_resume(
                    payload, state_file=state_file
                )
            (
                restored_last_step_metrics,
                restored_evaluation_records,
            ) = self._validate_resumable_terminal_evidence(
                payload, state_file=state_file
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
                if self._uses_agent_count_batch_sampler:
                    saved_schedule = payload.get("data_schedule")
                    if saved_schedule is None:
                        logger.warning(
                            "Trainer state predates data-schedule fingerprints; "
                            "resume compatibility cannot be verified."
                        )
                    else:
                        current_schedule = self._data_schedule_contract(self.epoch)
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
            if recovery_receipt_target is not None:
                if state_file_metadata is None:
                    raise RuntimeError(
                        "metadata_no_hash recovery receipt lacks source-state metadata"
                    )
                self._publish_recovery_load_receipt(
                    target=recovery_receipt_target,
                    source_state_dir=state_directory,
                    source_trainer_state_file=state_file_metadata,
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

    def _recovery_load_receipt_target(self):
        configured = os.environ.get("FASTWAM_RECOVERY_LOAD_RECEIPT", "").strip()
        if not configured:
            return None
        if self.artifact_integrity_mode != "metadata_no_hash":
            raise RuntimeError(
                "FASTWAM_RECOVERY_LOAD_RECEIPT is restricted to metadata_no_hash"
            )
        target = Path(configured).expanduser()
        if not target.is_absolute():
            raise ValueError("FASTWAM_RECOVERY_LOAD_RECEIPT must be an absolute path")
        output = Path(self.output_dir).expanduser().resolve(strict=True)
        expected = output / "recovery_load_receipt.json"
        target = target.resolve(strict=False)
        if target != expected:
            raise ValueError(
                "FASTWAM_RECOVERY_LOAD_RECEIPT must be the direct output receipt: "
                f"expected={expected} configured={target}"
            )
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"Refusing pre-existing recovery load receipt: {target}"
            )
        return target

    def _publish_recovery_load_receipt(
        self,
        *,
        target: Path,
        source_state_dir: Path,
        source_trainer_state_file: dict,
    ) -> None:
        """Collectively attest that Accelerate returned from a full-state load.

        This receipt is intentionally native to the trainer.  The external Gate2
        validator can therefore distinguish a real ``accelerator.load_state``
        return from rendered log text or a checkpoint directory that merely
        contains ``trainer_state.json``.
        """

        payload = {
            "schema_name": "fastwam-recovery-load-receipt",
            "schema_version": 1,
            "integrity_mode": "metadata_no_hash",
            "accelerator_load_state_returned": True,
            "source_state_dir": str(source_state_dir.resolve(strict=True)),
            "source_trainer_state_file": source_trainer_state_file,
            "output_dir": str(Path(self.output_dir).expanduser().resolve(strict=True)),
            "restored_global_step": int(self.global_step),
            "restored_epoch": int(self.epoch),
            "restored_batch_in_epoch": int(self.batch_in_epoch),
            "world_size": int(self.accelerator.num_processes),
        }
        if self.accelerator.is_main_process:
            nohash_publish_exclusive_json(target, payload)
        self.accelerator.wait_for_everyone()
        observed, _ = nohash_read_json(target)
        if observed != payload:
            raise RuntimeError(
                f"recovery load receipt differs across ranks: {target}"
            )

    def _validate_data_schedule_before_resume(
        self, payload: dict, *, state_file: Path
    ) -> None:
        if not self._uses_agent_count_batch_sampler:
            if payload.get("data_schedule") is not None:
                raise RuntimeError(
                    "Saved state has a dynamic data schedule but the current "
                    f"loader does not: {state_file}"
                )
            return
        if "epoch" not in payload:
            raise RuntimeError(
                f"metadata_no_hash state lacks epoch for schedule validation: {state_file}"
            )
        saved_schedule = payload.get("data_schedule")
        if not isinstance(saved_schedule, dict):
            raise RuntimeError(
                f"metadata_no_hash state lacks exact data schedule: {state_file}"
            )
        current_schedule = self._data_schedule_contract(int(payload["epoch"]))
        mismatches = self._contract_mismatches(saved_schedule, current_schedule)
        if mismatches:
            raise RuntimeError(
                "Cannot resume with a different deterministic data schedule "
                f"before accelerator.load_state: {mismatches}"
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

                    if self._should_pause_after_recovery_gate_checkpoint(
                        checkpoint_saved_this_step=checkpoint_saved_this_step
                    ):
                        self.accelerator.wait_for_everyone()
                        logger.info(
                            "[recovery-gate] rank=%d paused after checkpoint "
                            "step=%d; restart from the saved state to continue",
                            self.accelerator.process_index,
                            self.global_step,
                        )
                        self.accelerator.wait_for_everyone()
                        return

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
        
