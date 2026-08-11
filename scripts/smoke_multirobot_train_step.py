#!/usr/bin/env python3
"""Run one multi-robot forward/backward/optimizer step without checkpointing."""

from __future__ import annotations

import argparse
import gc
from contextlib import nullcontext
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data._utils.collate import default_collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="robofactory_multi_robot_vg1_hub1_gau1_224_1e-4",
    )
    parser.add_argument("--scope", default=None)
    parser.add_argument("--agent-count", type=int, default=None)
    parser.add_argument("--required-agent-counts", type=int, nargs="+", default=None)
    parser.add_argument("--required-tasks", nargs="+", default=None)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--text-cache", default=None)
    parser.add_argument("--video-height", type=int, default=None)
    parser.add_argument("--video-width", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={args.task}"])

    dataset_overrides = {}
    if args.required_agent_counts is not None:
        dataset_overrides["required_agent_counts"] = args.required_agent_counts
    if args.required_tasks is not None:
        dataset_overrides["required_tasks"] = args.required_tasks
    if args.stats is not None:
        dataset_overrides["pretrained_norm_stats"] = args.stats
    if args.text_cache is not None:
        dataset_overrides["text_embedding_cache_dir"] = args.text_cache
    if (args.video_height is None) != (args.video_width is None):
        raise ValueError("Set both --video-height and --video-width, or neither.")
    if args.video_height is not None:
        dataset_overrides["video_size"] = [args.video_height, args.video_width]

    print("SMOKE phase=dataset", flush=True)
    dataset = instantiate(cfg.data.train, **dataset_overrides)
    if args.agent_count is None:
        sample_index = 0
    else:
        try:
            sample_index = next(
                index
                for index, count in enumerate(dataset.agent_counts)
                if count == args.agent_count
            )
        except StopIteration as exc:
            raise ValueError(
                f"No sample with agent_count={args.agent_count}; "
                f"observed={sorted(set(dataset.agent_counts))}"
            ) from exc
    sample = default_collate([dataset[sample_index]])
    print(
        f"SMOKE train_windows={len(dataset)} task={sample['task_name'][0]} "
        f"agent_count={int(sample['agent_count'][0])} "
        f"video_shape={tuple(sample['video'].shape)} "
        f"action_shape={tuple(sample['action'].shape)}",
        flush=True,
    )
    print("SMOKE phase=model_init", flush=True)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=args.device)
    print("SMOKE phase=load_checkpoint", flush=True)
    resume = cfg.get("resume")
    init_weights = cfg.get("init_weights")
    if resume and init_weights:
        raise ValueError("`resume` and `init_weights` are mutually exclusive")
    if not resume and not init_weights:
        raise ValueError("Smoke requires exactly one of `resume` or `init_weights`")
    if init_weights:
        model.load_initialization_checkpoint(str(init_weights))
    else:
        model.load_checkpoint(str(resume))
    scope = str(cfg.trainable_scope) if args.scope is None else args.scope
    params = model.configure_trainable_parameters(scope)
    print(
        f"SMOKE arm={args.task} scope={scope} "
        f"trainable_params={sum(p.numel() for p in params)}",
        flush=True,
    )

    optimizer = None
    if not args.forward_only:
        optimizer = torch.optim.AdamW(params, lr=float(cfg.learning_rate))
    torch.cuda.reset_peak_memory_stats()
    print("SMOKE phase=forward", flush=True)
    context = torch.no_grad() if args.forward_only else nullcontext()
    with context:
        loss, metrics = model.training_loss(sample)
    print(f"SMOKE loss={float(loss.detach())} metrics={metrics}", flush=True)
    if not torch.isfinite(loss).item():
        raise RuntimeError("Smoke forward produced a non-finite loss")
    if args.forward_only:
        print(
            f"SMOKE peak_allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}",
            flush=True,
        )
        print(
            f"SMOKE peak_reserved_gib={torch.cuda.max_memory_reserved() / 2**30:.3f}",
            flush=True,
        )
        print("SMOKE RESULT=PASS_FORWARD_ONLY", flush=True)
        del model, sample
        gc.collect()
        torch.cuda.empty_cache()
        return

    print("SMOKE phase=backward", flush=True)
    loss.backward()
    finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in params
    )
    gradient_tensors = sum(parameter.grad is not None for parameter in params)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print(
        f"SMOKE gradients_finite={finite} gradient_tensors={gradient_tensors}",
        flush=True,
    )
    print(
        f"SMOKE peak_allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}",
        flush=True,
    )
    print(
        f"SMOKE peak_reserved_gib={torch.cuda.max_memory_reserved() / 2**30:.3f}",
        flush=True,
    )
    if not finite or gradient_tensors == 0:
        raise RuntimeError("Smoke step produced missing or non-finite gradients")
    print("SMOKE RESULT=PASS", flush=True)

    del model, optimizer, sample
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
