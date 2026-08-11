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


def _coerce_phase_labels(raw_labels, index: int) -> tuple[str, ...]:
    if isinstance(raw_labels, str):
        raw_labels = (raw_labels,)
    try:
        labels = tuple(sorted({str(label).strip() for label in raw_labels}))
    except TypeError as exc:
        raise ValueError(
            f"B4 phase labels at index {index} must be an iterable of strings"
        ) from exc
    if not labels or any(not label for label in labels):
        raise ValueError(f"B4 phase labels at index {index} cannot be empty")
    return labels


def resolve_b4_phase_labels(dataset: Sized) -> Optional[tuple[tuple[str, ...], ...]]:
    """Resolve per-window target-action proxy labels without loading samples."""

    labels_source = getattr(dataset, "b4_phase_labels", None)
    if labels_source is not None:
        raw_labels = labels_source() if callable(labels_source) else labels_source
        try:
            raw_labels = list(raw_labels)
        except TypeError as exc:
            raise TypeError("`dataset.b4_phase_labels` must be iterable") from exc
    else:
        getter = getattr(dataset, "get_b4_phase_labels", None)
        if not callable(getter):
            return None
        raw_labels = [getter(index) for index in range(len(dataset))]
    if len(raw_labels) != len(dataset):
        raise ValueError(
            "B4 phase-label metadata length must match the dataset: "
            f"labels={len(raw_labels)} dataset={len(dataset)}"
        )
    return tuple(
        _coerce_phase_labels(labels, index) for index, labels in enumerate(raw_labels)
    )


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

    With ``phase_balanced_fraction=0.5``, exactly half of every rank's epoch
    retains the original task/count schedule.  The other half keeps the same
    task/count allocation but chooses, with replacement, from each window's
    multi-label target-action phase/event-proxy strata.  This never introduces
    fixed-capacity agent slots or mixes native agent counts within a batch.
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
        b4_phase_labels: Optional[Sequence[Sequence[str]]] = None,
        phase_balanced_fraction: float = 0.0,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.reference_batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.epoch = 0
        self.resume_batch_offset = 0
        self.drop_last = False
        self.phase_balanced_fraction = float(phase_balanced_fraction)

        if (
            self.reference_batch_size <= 0
            or self.num_processes <= 0
            or self.gradient_accumulation_steps <= 0
        ):
            raise ValueError(
                "`batch_size`, `num_processes`, and `gradient_accumulation_steps` "
                "must be positive."
            )
        if self.phase_balanced_fraction not in (0.0, 0.5):
            raise ValueError(
                "phase_balanced_fraction currently supports only 0.0 or the "
                "audited B4 50/50 mixture (0.5)"
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
        if self.phase_balanced_fraction:
            resolved_phase_labels = (
                resolve_b4_phase_labels(dataset)
                if b4_phase_labels is None
                else tuple(b4_phase_labels)
            )
            if resolved_phase_labels is None:
                raise TypeError(
                    "B4 phase-balanced sampling requires `dataset.b4_phase_labels` "
                    "or `dataset.get_b4_phase_labels(index)`."
                )
            if len(resolved_phase_labels) != len(dataset):
                raise ValueError(
                    "B4 phase-label metadata length must match the dataset: "
                    f"labels={len(resolved_phase_labels)} dataset={len(dataset)}"
                )
            self.b4_phase_labels = tuple(
                _coerce_phase_labels(labels, index)
                for index, labels in enumerate(resolved_phase_labels)
            )
        else:
            self.b4_phase_labels = None
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
        phase_strata: dict[tuple[int, str, str], list[int]] = defaultdict(list)
        if self.b4_phase_labels is not None:
            for index, (count, task_id, labels) in enumerate(
                zip(self.agent_counts, self.task_ids, self.b4_phase_labels)
            ):
                for label in labels:
                    phase_strata[(count, task_id, label)].append(index)
        self._phase_strata = {
            stratum: tuple(indices) for stratum, indices in sorted(phase_strata.items())
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
        # B4 keeps the baseline epoch/token budget.  Align the total schedule
        # so each (count, task) stratum can be split exactly in half and each
        # source half consists of complete world-size blocks.  This prevents
        # ranks from specializing into original-vs-phase sampling and avoids
        # silently doubling optimizer steps per epoch.
        schedule_width = global_step_width
        stratum_divisor = 1
        if self.phase_balanced_fraction:
            schedule_width = lcm(schedule_width, 2 * self.num_processes)
            stratum_divisor = 2
        count_alignment = schedule_width // gcd(
            len(self.observed_agent_counts), schedule_width
        )
        batches_per_count_alignment = lcm(
            count_alignment,
            *(
                stratum_divisor * len(task_ids)
                for task_ids in self.tasks_by_agent_count.values()
            ),
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
        self.base_global_batches_per_epoch = (
            len(self.observed_agent_counts) * self.batches_per_agent_count
        )
        if self.base_global_batches_per_epoch % global_step_width:
            raise RuntimeError("Internal error: global batch schedule is not optimizer-step aligned.")
        self.original_batches_per_stratum = {
            stratum: (
                batch_count // 2
                if self.phase_balanced_fraction
                else batch_count
            )
            for stratum, batch_count in self.batches_per_stratum.items()
        }
        self.phase_balanced_batches_per_stratum = {
            stratum: (
                batch_count - self.original_batches_per_stratum[stratum]
                if self.phase_balanced_fraction
                else 0
            )
            for stratum, batch_count in self.batches_per_stratum.items()
        }
        self.original_global_batches_per_epoch = sum(
            self.original_batches_per_stratum.values()
        )
        self.phase_balanced_global_batches_per_epoch = sum(
            self.phase_balanced_batches_per_stratum.values()
        )
        if self.phase_balanced_fraction and (
            self.original_global_batches_per_epoch
            != self.phase_balanced_global_batches_per_epoch
        ):
            raise RuntimeError("Internal error: B4 schedule is not an exact 50/50 split.")
        self.global_batches_per_epoch = self.base_global_batches_per_epoch
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

    @staticmethod
    def _sample_batches(
        indices: Sequence[int],
        *,
        batch_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> list[list[int]]:
        sample_target = int(batch_count) * int(batch_size)
        sampled: list[int] = []
        while len(sampled) < sample_target:
            permutation = torch.randperm(len(indices), generator=generator).tolist()
            sampled.extend(indices[position] for position in permutation)
        sampled = sampled[:sample_target]
        return [
            sampled[start : start + batch_size]
            for start in range(0, sample_target, batch_size)
        ]

    def _original_epoch_records(
        self, generator: torch.Generator
    ) -> list[tuple[str, Optional[str], list[int]]]:
        batches: list[list[int]] = []
        for stratum, indices in self._strata.items():
            count, _ = stratum
            batch_size = self.batch_size_by_agent_count[count]
            batches.extend(
                self._sample_batches(
                    indices,
                    batch_count=self.original_batches_per_stratum[stratum],
                    batch_size=batch_size,
                    generator=generator,
                )
            )
        if len(batches) > 1:
            batch_order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[position] for position in batch_order]
        return [("original", None, batch) for batch in batches]

    def _phase_epoch_records(
        self, generator: torch.Generator
    ) -> list[tuple[str, Optional[str], list[int]]]:
        records: list[tuple[str, Optional[str], list[int]]] = []
        for stratum in self._strata:
            count, task_id = stratum
            labels = sorted(
                label
                for phase_count, phase_task, label in self._phase_strata
                if phase_count == count and phase_task == task_id
            )
            if not labels:
                raise RuntimeError(
                    f"No B4 target-action phase labels for count/task stratum {stratum}"
                )
            label_schedule: list[str] = []
            batch_target = self.phase_balanced_batches_per_stratum[stratum]
            while len(label_schedule) < batch_target:
                order = torch.randperm(len(labels), generator=generator).tolist()
                label_schedule.extend(labels[position] for position in order)
            for label in label_schedule[:batch_target]:
                indices = self._phase_strata[(count, task_id, label)]
                batch = self._sample_batches(
                    indices,
                    batch_count=1,
                    batch_size=self.batch_size_by_agent_count[count],
                    generator=generator,
                )[0]
                records.append(("phase_balanced", label, batch))
        if len(records) > 1:
            order = torch.randperm(len(records), generator=generator).tolist()
            records = [records[position] for position in order]
        return records

    def global_epoch_schedule(
        self, epoch: Optional[int] = None
    ) -> list[tuple[str, Optional[str], list[int]]]:
        """Return auditable ``(source, selected_label, batch)`` records."""

        epoch = self.epoch if epoch is None else int(epoch)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + epoch)
        original = self._original_epoch_records(generator)
        if len(original) != self.original_global_batches_per_epoch:
            raise RuntimeError(
                "Internal error: original schedule length mismatch: "
                f"got={len(original)} expected={self.original_global_batches_per_epoch}"
            )
        if not self.phase_balanced_fraction:
            return original

        phase_balanced = self._phase_epoch_records(generator)
        if len(phase_balanced) != self.phase_balanced_global_batches_per_epoch:
            raise RuntimeError(
                "Internal error: phase-balanced schedule length mismatch: "
                "got="
                f"{len(phase_balanced)} "
                f"expected={self.phase_balanced_global_batches_per_epoch}"
            )
        # Merge complete rank-width blocks. Every rank therefore receives one
        # original and one phase-balanced microbatch per pair of blocks, rather
        # than specializing odd/even ranks into different sampling treatments.
        records: list[tuple[str, Optional[str], list[int]]] = []
        for start in range(0, len(original), self.num_processes):
            original_block = original[start : start + self.num_processes]
            phase_block = phase_balanced[start : start + self.num_processes]
            if int(torch.randint(0, 2, (1,), generator=generator).item()):
                records.extend(phase_block)
                records.extend(original_block)
            else:
                records.extend(original_block)
                records.extend(phase_block)
        if len(records) != self.global_batches_per_epoch:
            raise RuntimeError(
                "Internal error: mixed schedule length mismatch: "
                f"got={len(records)} expected={self.global_batches_per_epoch}"
            )
        return records

    def global_epoch_batches(self, epoch: Optional[int] = None) -> list[list[int]]:
        """Materialize the deterministic global schedule before rank sharding."""

        return [record[2] for record in self.global_epoch_schedule(epoch)]

    def schedule_fingerprint(self, epoch: Optional[int] = None) -> str:
        epoch = self.epoch if epoch is None else int(epoch)
        payload = {
            "seed": self.seed,
            "epoch": epoch,
            "num_processes": self.num_processes,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "agent_action_token_budget": self.agent_action_token_budget,
            "action_horizon": self.action_horizon,
            "phase_balanced_fraction": self.phase_balanced_fraction,
            "schedule": self.global_epoch_schedule(epoch),
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
