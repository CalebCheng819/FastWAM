from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import DataLoader

from fastwam.trainer import Wan22Trainer
from fastwam.utils.samplers import (
    ResumableAgentCountBatchSampler,
    ResumableEpochSampler,
    resolve_agent_counts,
    resolve_task_ids,
)


class _AgentCountsDataset(torch.utils.data.Dataset):
    def __init__(self, agent_counts, task_ids, action_horizon=4, treatment="control"):
        self.agent_counts = tuple(agent_counts)
        self.task_ids = tuple(task_ids)
        self.action_horizon = int(action_horizon)
        self.treatment = treatment

    def __len__(self):
        return len(self.agent_counts)

    def __getitem__(self, index):
        count = self.agent_counts[index]
        return {
            "agent_count": torch.tensor(count),
            "task_id": self.task_ids[index],
            "action": torch.full(
                (count, self.action_horizon, 2),
                float(index),
            ),
        }


class _GetterDataset(torch.utils.data.Dataset):
    def __init__(self, agent_counts, task_ids):
        self._counts = tuple(agent_counts)
        self._task_ids = tuple(task_ids)
        self.action_horizon = 4

    def __len__(self):
        return len(self._counts)

    def get_agent_count(self, index):
        return self._counts[index]

    def get_task_id(self, index):
        return self._task_ids[index]

    def __getitem__(self, index):
        return index


class _GenericDataset(torch.utils.data.Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index


def _imbalanced_dataset(treatment="control"):
    records = (
        [(1, "n1-a")] * 1
        + [(1, "n1-b")] * 5
        + [(2, "n2-a")] * 2
        + [(2, "n2-b")] * 7
        + [(2, "n2-c")] * 1
        + [(4, "n4-a")] * 9
    )
    return _AgentCountsDataset(
        [count for count, _ in records],
        [task_id for _, task_id in records],
        action_horizon=4,
        treatment=treatment,
    )


def _dynamic_sampler(dataset, *, seed=17, num_processes=2, grad_accum=2):
    return ResumableAgentCountBatchSampler(
        dataset,
        seed=seed,
        batch_size=1,
        num_processes=num_processes,
        agent_action_token_budget=32,
        gradient_accumulation_steps=grad_accum,
    )


def _batch_stratum(batch, dataset):
    return dataset.agent_counts[batch[0]], dataset.task_ids[batch[0]]


def test_resolve_agent_counts_and_task_ids_support_sequence_getter_and_entries():
    sequence_dataset = _AgentCountsDataset([1, 2, 4], ["a", "b", "c"])
    getter_dataset = _GetterDataset([4, 3, 2, 1], ["d", "c", "b", "a"])
    entries_dataset = _GenericDataset(2)
    entries_dataset.entries = [
        {"agent_count": 1, "task_name": "one"},
        {"agent_count": 2, "task_name": "two"},
    ]
    entries_dataset.get_agent_count = lambda index: entries_dataset.entries[index]["agent_count"]

    assert resolve_agent_counts(sequence_dataset) == (1, 2, 4)
    assert resolve_task_ids(sequence_dataset) == ("a", "b", "c")
    assert resolve_agent_counts(getter_dataset) == (4, 3, 2, 1)
    assert resolve_task_ids(getter_dataset) == ("d", "c", "b", "a")
    assert resolve_task_ids(entries_dataset) == ("one", "two")
    assert resolve_agent_counts(_GenericDataset(3)) is None
    assert resolve_task_ids(_GenericDataset(3)) is None


def test_hierarchical_balance_and_real_token_budget_batches():
    dataset = _imbalanced_dataset()
    sampler = _dynamic_sampler(dataset)
    batches = sampler.global_epoch_batches()

    assert sampler.batch_size is None
    assert sampler.batch_size_by_agent_count == {1: 8, 2: 4, 4: 2}
    assert sampler.batches_per_agent_count == 12
    assert sampler.global_batches_per_epoch == 36
    assert sampler.global_batches_per_epoch % (2 * 2) == 0

    count_frequency = Counter()
    stratum_frequency = Counter()
    for batch in batches:
        count, task_id = _batch_stratum(batch, dataset)
        assert {dataset.agent_counts[index] for index in batch} == {count}
        assert {dataset.task_ids[index] for index in batch} == {task_id}
        assert len(batch) == sampler.batch_size_by_agent_count[count]
        assert len(batch) * count * dataset.action_horizon <= 32
        count_frequency[count] += 1
        stratum_frequency[(count, task_id)] += 1

    assert count_frequency == {1: 12, 2: 12, 4: 12}
    assert stratum_frequency == {
        (1, "n1-a"): 6,
        (1, "n1-b"): 6,
        (2, "n2-a"): 4,
        (2, "n2-b"): 4,
        (2, "n2-c"): 4,
        (4, "n4-a"): 12,
    }

    collated = list(DataLoader(dataset, batch_sampler=sampler))
    assert len(collated) == len(batches)
    for batch in collated:
        assert "agent_mask" not in batch
        assert batch["action"].ndim == 4
        assert torch.unique(batch["agent_count"]).numel() == 1
        assert batch["action"].shape[1] == int(batch["agent_count"][0])


def test_treatment_arms_share_exact_schedule_and_epoch_seed_advances():
    first = _dynamic_sampler(_imbalanced_dataset(treatment="vg0-hub0"), seed=29)
    second = _dynamic_sampler(_imbalanced_dataset(treatment="vg1-hub1"), seed=29)

    assert first.global_epoch_batches() == second.global_epoch_batches()
    assert first.schedule_fingerprint() == second.schedule_fingerprint()

    epoch_zero = first.global_epoch_batches()
    first.set_epoch(1)
    second.set_epoch(1)
    assert first.global_epoch_batches() == second.global_epoch_batches()
    assert first.schedule_fingerprint() == second.schedule_fingerprint()
    assert first.global_epoch_batches() != epoch_zero


def test_two_rank_sharding_and_optimizer_step_alignment():
    dataset = _imbalanced_dataset()
    rank_zero_source = _dynamic_sampler(dataset, seed=41, num_processes=2, grad_accum=2)
    rank_one_source = _dynamic_sampler(dataset, seed=41, num_processes=2, grad_accum=2)
    global_batches = rank_zero_source.global_epoch_batches()

    with pytest.raises(ValueError, match="even_batches=False"):
        BatchSamplerShard(
            rank_zero_source,
            num_processes=2,
            process_index=0,
            split_batches=False,
            even_batches=True,
        )

    rank_zero = list(
        BatchSamplerShard(
            rank_zero_source,
            num_processes=2,
            process_index=0,
            split_batches=False,
            even_batches=False,
        )
    )
    rank_one = list(
        BatchSamplerShard(
            rank_one_source,
            num_processes=2,
            process_index=1,
            split_batches=False,
            even_batches=False,
        )
    )

    assert rank_zero == global_batches[0::2]
    assert rank_one == global_batches[1::2]
    assert len(rank_zero) == len(rank_one) == rank_zero_source.microbatches_per_process
    assert len(rank_zero) % rank_zero_source.gradient_accumulation_steps == 0
    assert rank_zero_source.optimizer_steps_per_epoch == len(rank_zero) // 2


def test_two_rank_resume_replays_exact_remaining_optimizer_steps():
    dataset = _imbalanced_dataset()
    full_rank_zero_source = _dynamic_sampler(dataset, seed=43, num_processes=2, grad_accum=2)
    full_rank_one_source = _dynamic_sampler(dataset, seed=43, num_processes=2, grad_accum=2)
    full_rank_zero = list(
        BatchSamplerShard(
            full_rank_zero_source, 2, 0, split_batches=False, even_batches=False
        )
    )
    full_rank_one = list(
        BatchSamplerShard(
            full_rank_one_source, 2, 1, split_batches=False, even_batches=False
        )
    )

    resumed_rank_zero_source = _dynamic_sampler(dataset, seed=43, num_processes=2, grad_accum=2)
    resumed_rank_one_source = _dynamic_sampler(dataset, seed=43, num_processes=2, grad_accum=2)
    resumed_rank_zero_source.set_resume_batch_offset(2)
    resumed_rank_one_source.set_resume_batch_offset(2)
    resumed_rank_zero = list(
        BatchSamplerShard(
            resumed_rank_zero_source, 2, 0, split_batches=False, even_batches=False
        )
    )
    resumed_rank_one = list(
        BatchSamplerShard(
            resumed_rank_one_source, 2, 1, split_batches=False, even_batches=False
        )
    )

    assert resumed_rank_zero == full_rank_zero[2:]
    assert resumed_rank_one == full_rank_one[2:]
    assert len(resumed_rank_zero) == len(resumed_rank_one)
    assert len(resumed_rank_zero) % 2 == 0
    with pytest.raises(ValueError, match="optimizer-step boundary"):
        resumed_rank_zero_source.set_resume_batch_offset(1)


def test_no_token_budget_preserves_fixed_batch_size():
    dataset = _imbalanced_dataset()
    sampler = ResumableAgentCountBatchSampler(
        dataset,
        seed=47,
        batch_size=3,
        num_processes=2,
        gradient_accumulation_steps=2,
    )

    assert sampler.agent_action_token_budget is None
    assert sampler.batch_size == 3
    assert sampler.batch_size_by_agent_count == {1: 3, 2: 3, 4: 3}
    assert all(len(batch) == 3 for batch in sampler.global_epoch_batches())
    assert sampler.global_batches_per_epoch % 4 == 0


def test_generic_sampler_resume_uses_current_epoch_and_advances_seed():
    dataset = _GenericDataset(20)
    sampler = ResumableEpochSampler(
        dataset,
        seed=53,
        batch_size=2,
        num_processes=2,
    )
    sampler.set_epoch(4)
    full_epoch = list(sampler)

    sampler.set_resume_batch_offset(2)
    assert sampler.resume_sample_offset == 8
    assert list(sampler) == full_epoch[8:]
    assert len(sampler) == len(dataset) - 8

    sampler.clear_resume_batch_offset()
    sampler.set_epoch(5)
    assert list(sampler) != full_epoch


def test_trainer_step_estimate_uses_aligned_optimizer_steps():
    dataset = _imbalanced_dataset()
    sampler = _dynamic_sampler(dataset, seed=61, num_processes=2, grad_accum=2)
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.max_steps = None
    trainer.train_dataset = dataset
    trainer.train_sampler = sampler
    trainer._uses_agent_count_batch_sampler = True
    trainer.accelerator = SimpleNamespace(num_processes=2)
    trainer.batch_size = 1
    trainer.gradient_accumulation_steps = 2
    trainer.num_epochs = 3

    assert trainer._estimate_total_train_steps() == sampler.optimizer_steps_per_epoch * 3


def test_performance_counts_use_real_dynamic_work():
    multi_agent = {"action": torch.zeros(4, 2, 8, 7)}
    single_agent = {"action": torch.zeros(3, 8, 7)}
    assert Wan22Trainer._sample_work_counts(multi_agent) == (4, 64)
    assert Wan22Trainer._sample_work_counts(single_agent) == (3, 24)


@pytest.mark.parametrize(
    ("save_final_checkpoint", "checkpoint_saved_this_step", "expected"),
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, True, False),
    ],
)
def test_final_checkpoint_toggle_avoids_duplicate_terminal_write(
    save_final_checkpoint, checkpoint_saved_this_step, expected
):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.save_final_checkpoint_enabled = save_final_checkpoint

    assert trainer._should_save_final_checkpoint(
        checkpoint_saved_this_step=checkpoint_saved_this_step
    ) is expected


def test_dynamic_batching_resolves_deepspeed_micro_batch_metadata():
    deepspeed_config = {"train_micro_batch_size_per_gpu": "auto"}
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.agent_action_token_budget = 128
    trainer.batch_size = 1
    trainer.accelerator = SimpleNamespace(
        state=SimpleNamespace(
            deepspeed_plugin=SimpleNamespace(deepspeed_config=deepspeed_config)
        )
    )

    trainer._configure_dynamic_deepspeed_batch_accounting()

    assert deepspeed_config["train_micro_batch_size_per_gpu"] == 1


def test_training_loss_runs_through_prepared_model_forward():
    class _PreparedModel:
        def __init__(self):
            self.called = False

        def __call__(self, sample):
            self.called = True
            return sample["loss"], {"loss_action": 1.0}

        def training_loss(self, sample):
            raise AssertionError("prepared wrappers must not be bypassed")

    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = _PreparedModel()
    loss = torch.tensor(1.0)

    result = trainer._forward_training_loss({"loss": loss})

    assert trainer.model.called
    assert result == (loss, {"loss_action": 1.0})


@pytest.mark.parametrize("invalid", [0, -1, True, "bad"])
def test_dynamic_batching_rejects_invalid_deepspeed_micro_batch_metadata(invalid):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.agent_action_token_budget = 128
    trainer.batch_size = 1
    trainer.accelerator = SimpleNamespace(
        state=SimpleNamespace(
            deepspeed_plugin=SimpleNamespace(
                deepspeed_config={"train_micro_batch_size_per_gpu": invalid}
            )
        )
    )

    with pytest.raises(ValueError, match="positive integer"):
        trainer._configure_dynamic_deepspeed_batch_accounting()


@pytest.mark.parametrize("counts", [[1, 0, 2], [True, 2], [1.5, 2], ["bad", 2]])
def test_invalid_agent_count_metadata_is_rejected(counts):
    dataset = _AgentCountsDataset(counts, ["task"] * len(counts))
    with pytest.raises(ValueError):
        resolve_agent_counts(dataset)


def test_invalid_or_missing_schedule_metadata_is_rejected():
    missing_tasks = _GenericDataset(2)
    missing_tasks.agent_counts = [1, 2]
    with pytest.raises(TypeError, match="task"):
        ResumableAgentCountBatchSampler(
            missing_tasks,
            seed=67,
            batch_size=1,
            num_processes=1,
        )

    dataset = _AgentCountsDataset([4], ["task"], action_horizon=8)
    with pytest.raises(ValueError, match="fit at least one"):
        ResumableAgentCountBatchSampler(
            dataset,
            seed=67,
            batch_size=1,
            num_processes=1,
            agent_action_token_budget=31,
        )


def test_metadata_lengths_must_match_dataset():
    dataset = _GenericDataset(3)
    dataset.agent_counts = [1, 2]
    dataset.task_ids = ["a", "b"]
    with pytest.raises(ValueError, match="length must match"):
        resolve_agent_counts(dataset)
    with pytest.raises(ValueError, match="length must match"):
        resolve_task_ids(dataset)
