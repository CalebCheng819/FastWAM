from collections import defaultdict
import hashlib
import json
from math import ceil, gcd, lcm
from operator import index as integer_index
from typing import Iterator, Mapping, Optional, Sequence, Sized

import torch
from torch.utils.data import Sampler


def _coerce_agent_count(raw_count, index: int) -> int:
    if isinstance(raw_count, bool):
        raise ValueError(f"Agent count at index {index} must be a positive integer, got {raw_count!r}")
    try:
        count = integer_index(raw_count)
    except TypeError as exc:
        raise ValueError(
            f"Agent count at index {index} must be a positive integer, got {raw_count!r}"
        ) from exc
    if count <= 0:
        raise ValueError(f"Agent count at index {index} must be positive, got {count}")
    return count


def resolve_agent_counts(dataset: Sized) -> Optional[tuple[int, ...]]:
    """Return per-sample agent counts when a dataset exposes that metadata.

    Variable-cardinality datasets can expose either an ``agent_counts``
    sequence (or zero-argument callable) or ``get_agent_count(index)``.  The
    returned tuple is intentionally materialized once so batching never opens
    sample payloads merely to discover their cardinality.
    """

    counts_source = getattr(dataset, "agent_counts", None)
    if counts_source is not None:
        raw_counts = counts_source() if callable(counts_source) else counts_source
        try:
            raw_counts = list(raw_counts)
        except TypeError as exc:
            raise TypeError("`dataset.agent_counts` must be an iterable of positive integers.") from exc
    else:
        get_agent_count = getattr(dataset, "get_agent_count", None)
        if not callable(get_agent_count):
            return None
        raw_counts = [get_agent_count(index) for index in range(len(dataset))]

    if len(raw_counts) != len(dataset):
        raise ValueError(
            "Agent-count metadata length must match the dataset: "
            f"counts={len(raw_counts)} dataset={len(dataset)}"
        )

    return tuple(_coerce_agent_count(raw_count, index) for index, raw_count in enumerate(raw_counts))


def _coerce_task_id(raw_task_id, index: int) -> str:
    if not isinstance(raw_task_id, (str, int)) or isinstance(raw_task_id, bool):
        raise ValueError(
            f"Task id at index {index} must be a stable string or integer, got {raw_task_id!r}"
        )
    task_id = str(raw_task_id).strip()
    if not task_id:
        raise ValueError(f"Task id at index {index} cannot be empty.")
    return task_id


def resolve_task_ids(dataset: Sized) -> Optional[tuple[str, ...]]:
    """Resolve stable per-sample task ids without loading sample payloads."""

    task_source = getattr(dataset, "task_ids", None)
    if task_source is not None:
        raw_task_ids = task_source() if callable(task_source) else task_source
        try:
            raw_task_ids = list(raw_task_ids)
        except TypeError as exc:
            raise TypeError("`dataset.task_ids` must be an iterable of stable task ids.") from exc
    else:
        get_task_id = getattr(dataset, "get_task_id", None)
        if callable(get_task_id):
            raw_task_ids = [get_task_id(index) for index in range(len(dataset))]
        else:
            entries = getattr(dataset, "entries", None)
            if entries is None or len(entries) != len(dataset):
                return None
            raw_task_ids = []
            for index, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    return None
                task_id = entry.get("task_id", entry.get("task_name"))
                if task_id is None:
                    return None
                raw_task_ids.append(task_id)

    if len(raw_task_ids) != len(dataset):
        raise ValueError(
            "Task-id metadata length must match the dataset: "
            f"task_ids={len(raw_task_ids)} dataset={len(dataset)}"
        )
    return tuple(_coerce_task_id(task_id, index) for index, task_id in enumerate(raw_task_ids))


class ResumableEpochSampler(Sampler[int]):
    """Deterministic sample sampler for ordinary fixed-shape datasets."""

    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.resume_batch_offset = 0

        if self.batch_size <= 0 or self.num_processes <= 0:
            raise ValueError("`batch_size` and `num_processes` must be positive.")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        """Backward-compatible alias for older trainer checkpoints."""

        self.set_epoch(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        batch_in_epoch = int(batch_in_epoch)
        if batch_in_epoch < 0:
            raise ValueError(f"`batch_in_epoch` must be non-negative, got {batch_in_epoch}")
        self.resume_batch_offset = batch_in_epoch

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    @property
    def resume_sample_offset(self) -> int:
        return self.resume_batch_offset * self.batch_size * self.num_processes

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.resume_sample_offset:
            indices = indices[self.resume_sample_offset :]
        return iter(indices)

    def __len__(self) -> int:
        return max(len(self.dataset) - self.resume_sample_offset, 0)


class ResumableAgentCountBatchSampler(Sampler[list[int]]):
    """Deterministic hierarchical task/count-balanced batch schedule.

    The global schedule gives every observed agent count the same number of
    batches.  Within a count, its tasks receive equal batch frequency.  Small
    strata are sampled with replacement; no dummy agent slots or mixed-count
    batches are introduced.

    When ``agent_action_token_budget`` is set, the sample batch size for count
    ``N`` is ``budget // (N * action_horizon)``.  The global batch count is
    aligned to ``num_processes * gradient_accumulation_steps`` so Accelerate's
    ``BatchSamplerShard(even_batches=False)`` gives every rank the same number
    of micro-steps and every epoch ends on an optimizer-step boundary.
    """

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        *,
        agent_counts: Optional[Sequence[int]] = None,
        task_ids: Optional[Sequence[str | int]] = None,
        action_horizon: Optional[int] = None,
        agent_action_token_budget: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.reference_batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.epoch = 0
        self.resume_batch_offset = 0
        self.drop_last = False

        if (
            self.reference_batch_size <= 0
            or self.num_processes <= 0
            or self.gradient_accumulation_steps <= 0
        ):
            raise ValueError(
                "`batch_size`, `num_processes`, and `gradient_accumulation_steps` "
                "must be positive."
            )

        resolved_counts = resolve_agent_counts(dataset) if agent_counts is None else tuple(agent_counts)
        if resolved_counts is None:
            raise TypeError(
                "Agent-count bucket batching requires `dataset.agent_counts` or "
                "`dataset.get_agent_count(index)`."
            )
        if len(resolved_counts) != len(dataset):
            raise ValueError(
                "Agent-count metadata length must match the dataset: "
                f"counts={len(resolved_counts)} dataset={len(dataset)}"
            )
        self.agent_counts = tuple(
            _coerce_agent_count(raw_count, index)
            for index, raw_count in enumerate(resolved_counts)
        )

        resolved_task_ids = resolve_task_ids(dataset) if task_ids is None else tuple(task_ids)
        if resolved_task_ids is None:
            raise TypeError(
                "Balanced variable-agent batching requires `dataset.task_ids`, "
                "`dataset.get_task_id(index)`, or entries containing task_id/task_name."
            )
        if len(resolved_task_ids) != len(dataset):
            raise ValueError(
                "Task-id metadata length must match the dataset: "
                f"task_ids={len(resolved_task_ids)} dataset={len(dataset)}"
            )
        self.task_ids = tuple(
            _coerce_task_id(task_id, index)
            for index, task_id in enumerate(resolved_task_ids)
        )
        if not self.agent_counts:
            raise ValueError("Agent-count bucket batching requires a non-empty dataset.")

        self.agent_action_token_budget = (
            None
            if agent_action_token_budget is None
            else _coerce_agent_count(agent_action_token_budget, -1)
        )
        if self.agent_action_token_budget is not None:
            if action_horizon is None:
                action_horizon = getattr(dataset, "action_horizon", None)
            if action_horizon is None:
                raise ValueError(
                    "`action_horizon` or `dataset.action_horizon` is required when "
                    "agent_action_token_budget is enabled."
                )
            self.action_horizon = _coerce_agent_count(action_horizon, -1)
        else:
            self.action_horizon = None

        observed_counts = sorted(set(self.agent_counts))
        if self.agent_action_token_budget is None:
            self.batch_size = self.reference_batch_size
            self.batch_size_by_agent_count = {
                count: self.reference_batch_size for count in observed_counts
            }
        else:
            minimum_tokens = max(observed_counts) * self.action_horizon
            if self.agent_action_token_budget < minimum_tokens:
                raise ValueError(
                    "agent_action_token_budget must fit at least one largest-count sample: "
                    f"budget={self.agent_action_token_budget} required={minimum_tokens}"
                )
            # A scalar batch_size would make Accelerate assume all yielded
            # batches have that size. None forces the required even_batches=False gate.
            self.batch_size = None
            self.batch_size_by_agent_count = {
                count: self.agent_action_token_budget // (count * self.action_horizon)
                for count in observed_counts
            }

        strata: dict[tuple[int, str], list[int]] = defaultdict(list)
        for index, (count, task_id) in enumerate(zip(self.agent_counts, self.task_ids)):
            strata[(count, task_id)].append(index)
        self._strata = {
            stratum: tuple(indices) for stratum, indices in sorted(strata.items())
        }
        self.observed_agent_counts = tuple(observed_counts)

        tasks_by_count: dict[int, list[str]] = defaultdict(list)
        natural_batches: dict[tuple[int, str], int] = {}
        for (count, task_id), indices in self._strata.items():
            tasks_by_count[count].append(task_id)
            natural_batches[(count, task_id)] = ceil(
                len(indices) / self.batch_size_by_agent_count[count]
            )
        self.tasks_by_agent_count = {
            count: tuple(sorted(task_ids)) for count, task_ids in sorted(tasks_by_count.items())
        }

        minimum_batches_per_count = max(
            len(task_ids)
            * max(natural_batches[(count, task_id)] for task_id in task_ids)
            for count, task_ids in self.tasks_by_agent_count.items()
        )
        global_step_width = self.num_processes * self.gradient_accumulation_steps
        count_alignment = global_step_width // gcd(
            len(self.observed_agent_counts), global_step_width
        )
        batches_per_count_alignment = lcm(
            count_alignment,
            *(len(task_ids) for task_ids in self.tasks_by_agent_count.values()),
        )
        self.batches_per_agent_count = (
            ceil(minimum_batches_per_count / batches_per_count_alignment)
            * batches_per_count_alignment
        )
        self.batches_per_stratum = {
            (count, task_id): self.batches_per_agent_count // len(task_ids)
            for count, task_ids in self.tasks_by_agent_count.items()
            for task_id in task_ids
        }
        self.global_batches_per_epoch = (
            len(self.observed_agent_counts) * self.batches_per_agent_count
        )
        if self.global_batches_per_epoch % global_step_width:
            raise RuntimeError("Internal error: global batch schedule is not optimizer-step aligned.")
        self.microbatches_per_process = self.global_batches_per_epoch // self.num_processes
        self.optimizer_steps_per_epoch = (
            self.microbatches_per_process // self.gradient_accumulation_steps
        )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        """Backward-compatible alias matching :class:`ResumableEpochSampler`."""

        self.set_epoch(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        batch_in_epoch = int(batch_in_epoch)
        if batch_in_epoch < 0:
            raise ValueError(f"`batch_in_epoch` must be non-negative, got {batch_in_epoch}")
        if batch_in_epoch > self.microbatches_per_process:
            raise ValueError(
                f"`batch_in_epoch`={batch_in_epoch} exceeds "
                f"microbatches_per_process={self.microbatches_per_process}"
            )
        if batch_in_epoch % self.gradient_accumulation_steps:
            raise ValueError(
                "Resume must start on an optimizer-step boundary: "
                f"batch_in_epoch={batch_in_epoch} "
                f"gradient_accumulation_steps={self.gradient_accumulation_steps}"
            )
        self.resume_batch_offset = batch_in_epoch

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    @property
    def resume_global_batch_offset(self) -> int:
        return self.resume_batch_offset * self.num_processes

    def global_epoch_batches(self, epoch: Optional[int] = None) -> list[list[int]]:
        """Materialize the deterministic global schedule before rank sharding."""

        epoch = self.epoch if epoch is None else int(epoch)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + epoch)
        batches: list[list[int]] = []

        for stratum, indices in self._strata.items():
            count, _ = stratum
            batch_size = self.batch_size_by_agent_count[count]
            sample_target = self.batches_per_stratum[stratum] * batch_size
            sampled: list[int] = []
            while len(sampled) < sample_target:
                permutation = torch.randperm(len(indices), generator=generator).tolist()
                sampled.extend(indices[position] for position in permutation)
            sampled = sampled[:sample_target]
            batches.extend(
                sampled[start : start + batch_size]
                for start in range(0, sample_target, batch_size)
            )

        if len(batches) > 1:
            batch_order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[position] for position in batch_order]
        if len(batches) != self.global_batches_per_epoch:
            raise RuntimeError(
                "Internal error: materialized schedule length mismatch: "
                f"got={len(batches)} expected={self.global_batches_per_epoch}"
            )
        return batches

    def schedule_fingerprint(self, epoch: Optional[int] = None) -> str:
        epoch = self.epoch if epoch is None else int(epoch)
        payload = {
            "seed": self.seed,
            "epoch": epoch,
            "num_processes": self.num_processes,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "agent_action_token_budget": self.agent_action_token_budget,
            "action_horizon": self.action_horizon,
            "batches": self.global_epoch_batches(epoch),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def __iter__(self) -> Iterator[list[int]]:
        batches = self.global_epoch_batches()
        if self.resume_global_batch_offset:
            batches = batches[self.resume_global_batch_offset :]
        return iter(batches)

    def __len__(self) -> int:
        return max(self.global_batches_per_epoch - self.resume_global_batch_offset, 0)
