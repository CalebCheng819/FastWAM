from contextlib import nullcontext

import pytest
import torch

from fastwam.trainer import Wan22Trainer


def _sample(num_agents: int, *, include_gaussian: bool = True):
    sample = {
        "video": torch.randn(3, 1, 16, 16),
        "action": torch.randn(num_agents, 5, 3),
        "agent_state": torch.randn(num_agents, 4),
        "agent_geometry": torch.randn(num_agents, 7),
        "agent_ids": torch.arange(num_agents),
        "action_is_pad": torch.zeros(num_agents, 5, dtype=torch.bool),
        "image_is_pad": torch.zeros(1, dtype=torch.bool),
        "context": torch.randn(6, 16),
        "context_mask": torch.ones(6, dtype=torch.bool),
        "prompt": f"coordinate {num_agents} robots",
        "task_name": f"synthetic-{num_agents}",
        "agent_count": num_agents,
    }
    if include_gaussian:
        sample["agent_gaussian"] = torch.randn(
            num_agents, 13, 28, 40, dtype=torch.float16
        )
    return sample


@pytest.mark.parametrize("num_agents", [1, 2, 3, 4])
def test_multi_robot_eval_batching_preserves_native_agent_fields(num_agents):
    source = _sample(num_agents)
    batched = Wan22Trainer._to_batched_multi_robot_eval_sample(source)

    assert batched["video"].shape == (1, 3, 1, 16, 16)
    assert batched["action"].shape == (1, num_agents, 5, 3)
    assert batched["agent_state"].shape == (1, num_agents, 4)
    assert batched["agent_geometry"].shape == (1, num_agents, 7)
    assert batched["agent_ids"].shape == (1, num_agents)
    assert batched["action_is_pad"].shape == (1, num_agents, 5)
    assert batched["image_is_pad"].shape == (1, 1)
    assert batched["context"].shape == (1, 6, 16)
    assert batched["context_mask"].shape == (1, 6)
    assert batched["agent_gaussian"].shape == (1, num_agents, 13, 28, 40)
    assert batched["agent_gaussian"].dtype == torch.float16
    assert batched["agent_count"].tolist() == [num_agents]
    assert batched["prompt"] == [source["prompt"]]
    assert batched["task_name"] == [source["task_name"]]


class _Dataset:
    def __init__(self, counts):
        self.agent_counts = tuple(counts)
        self.samples = [_sample(count) for count in counts]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class _OfflineLossModel(torch.nn.Module):
    def __init__(self, *, stochastic=False, video_loss=None):
        super().__init__()
        self.action_expert = torch.nn.Identity()
        self.video_expert = torch.nn.Identity()
        self.stochastic = stochastic
        self.video_loss = video_loss
        self.seen_counts = []

    def infer_action_multi(self):  # pragma: no cover - marker for dispatch
        raise NotImplementedError

    def training_loss(self, sample):
        count = int(sample["agent_count"].item())
        self.seen_counts.append(count)
        assert sample["action"].shape[1] == count
        assert sample["agent_state"].shape[1] == count
        assert sample["agent_geometry"].shape[1] == count
        assert sample["agent_ids"].shape[1] == count
        assert sample["agent_gaussian"].shape[1] == count
        loss_action = torch.tensor(float(count))
        if self.stochastic:
            loss_action = loss_action + torch.rand(())
        metrics = {"loss_action": loss_action}
        loss = loss_action
        if self.video_loss is not None:
            loss_video = torch.tensor(float(self.video_loss))
            metrics["loss_video"] = loss_video
            loss = loss + loss_video
        return loss, metrics


class _Accelerator:
    def __init__(self, *, process_index=0, num_processes=1, remote_stats=None):
        self.process_index = process_index
        self.num_processes = num_processes
        self.device = torch.device("cpu")
        self.remote_stats = remote_stats
        self.reduce_calls = 0

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return nullcontext()

    def reduce(self, tensor, *, reduction):
        self.reduce_calls += 1
        assert reduction == "sum"
        if self.remote_stats is None:
            return tensor
        return tensor + torch.as_tensor(
            self.remote_stats, device=tensor.device, dtype=tensor.dtype
        )


def _offline_trainer(dataset, model, accelerator, *, limit):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.val_dataset = dataset
    trainer.model = model
    trainer.accelerator = accelerator
    trainer.offline_eval_num_samples = limit
    trainer.seed = 42
    return trainer


def test_multi_robot_offline_eval_is_fixed_and_does_not_advance_training_rng():
    dataset = _Dataset([1, 2, 3, 4])
    model = _OfflineLossModel(stochastic=True).eval()
    trainer = _offline_trainer(dataset, model, _Accelerator(), limit=4)

    torch.manual_seed(909)
    before = torch.random.get_rng_state().clone()
    first = trainer.evaluate()
    after_first = torch.random.get_rng_state().clone()
    second = trainer.evaluate()
    after_second = torch.random.get_rng_state().clone()

    assert first == second
    assert first["evaluation_kind"] == "multi_robot_offline_loss"
    assert first["offline_samples"] == 4
    assert first["offline_agent_counts"] == [1, 2, 3, 4]
    assert first["val_loss_action"] == pytest.approx(first["val_loss"])
    assert "val_loss_video" not in first
    assert torch.equal(before, after_first)
    assert torch.equal(before, after_second)


def test_multi_robot_offline_eval_shards_samples_and_reduces_sum_and_count():
    dataset = _Dataset([1, 2, 3, 4])
    model = _OfflineLossModel().eval()
    # Rank 0 evaluates N=1 and N=3. Simulate rank 1 contributing N=2 and N=4.
    accelerator = _Accelerator(
        process_index=0,
        num_processes=2,
        remote_stats=((6.0, 2.0), (6.0, 2.0), (0.0, 0.0)),
    )
    trainer = _offline_trainer(dataset, model, accelerator, limit=4)

    metrics = trainer.evaluate()

    assert model.seen_counts == [1, 3]
    assert accelerator.reduce_calls == 1
    assert metrics["val_loss"] == pytest.approx(2.5)
    assert metrics["val_loss_action"] == pytest.approx(2.5)
    assert metrics["offline_samples"] == 4
    assert metrics["offline_agent_counts"] == [1, 2, 3, 4]


def test_cross_arm_action_loss_remains_comparable_when_joint_has_video_loss():
    dataset = _Dataset([1, 2, 3, 4])
    action_only = _offline_trainer(
        dataset,
        _OfflineLossModel().eval(),
        _Accelerator(),
        limit=4,
    ).evaluate()
    joint = _offline_trainer(
        dataset,
        _OfflineLossModel(video_loss=10.0).eval(),
        _Accelerator(),
        limit=4,
    ).evaluate()

    assert action_only["val_loss_action"] == pytest.approx(2.5)
    assert joint["val_loss_action"] == pytest.approx(2.5)
    assert action_only["val_loss"] == pytest.approx(2.5)
    assert joint["val_loss"] == pytest.approx(12.5)
    assert "val_loss_video" not in action_only
    assert joint["val_loss_video"] == pytest.approx(10.0)


def test_offline_eval_rejects_nonfinite_component_metric():
    class _BadComponentModel(_OfflineLossModel):
        def training_loss(self, sample):
            loss, _ = super().training_loss(sample)
            return loss, {"loss_action": torch.tensor(float("nan"))}

    dataset = _Dataset([2])
    trainer = _offline_trainer(
        dataset,
        _BadComponentModel().eval(),
        _Accelerator(),
        limit=1,
    )
    with pytest.raises(FloatingPointError, match="metric 'action'"):
        trainer.evaluate()


def test_offline_eval_rejects_distributed_component_count_mismatch():
    dataset = _Dataset([1, 2])
    accelerator = _Accelerator(
        process_index=0,
        num_processes=2,
        # Remote rank contributes total but (incorrectly) omits action.
        remote_stats=((2.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
    )
    trainer = _offline_trainer(
        dataset,
        _OfflineLossModel().eval(),
        accelerator,
        limit=2,
    )
    with pytest.raises(RuntimeError, match="metric presence mismatch"):
        trainer.evaluate()


def test_multi_robot_eval_subset_is_cardinality_stratified_and_seeded():
    dataset = _Dataset([1, 1, 2, 2, 3, 3, 4, 4])
    first = Wan22Trainer._select_multi_robot_eval_indices(
        dataset, limit=4, seed=17
    )
    second = Wan22Trainer._select_multi_robot_eval_indices(
        dataset, limit=4, seed=17
    )
    assert first == second
    assert sorted(dataset.agent_counts[index] for index in first) == [1, 2, 3, 4]


def test_multi_robot_eval_subset_covers_tasks_within_each_cardinality():
    dataset = _Dataset([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    dataset.task_ids = (
        "n2-a",
        "n2-a",
        "n2-b",
        "n2-c",
        "n3-a",
        "n3-a",
        "n3-b",
        "n3-b",
        "n4-a",
        "n4-a",
        "n4-a",
        "n4-a",
    )

    selected = Wan22Trainer._select_multi_robot_eval_indices(
        dataset, limit=12, seed=17
    )

    assert sorted(dataset.agent_counts[index] for index in selected) == [
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
    ]
    assert {dataset.task_ids[index] for index in selected} == {
        "n2-a",
        "n2-b",
        "n2-c",
        "n3-a",
        "n3-b",
        "n4-a",
    }


def test_trainer_epoch_update_reaches_dataset_sampler_and_prepared_loader():
    class _EpochRecorder:
        def __init__(self):
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.train_dataset = _EpochRecorder()
    trainer.train_sampler = _EpochRecorder()
    trainer.train_loader = _EpochRecorder()

    trainer._set_train_data_epoch(7)

    assert trainer.train_dataset.epochs == [7]
    assert trainer.train_sampler.epochs == [7]
    assert trainer.train_loader.epochs == [7]
