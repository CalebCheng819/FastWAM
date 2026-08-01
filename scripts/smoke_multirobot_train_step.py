#!/usr/bin/env python3
"""Run one multi-robot forward/backward/optimizer step without checkpointing."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data._utils.collate import default_collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="robofactory_multi_robot_hub_224_1e-4")
    parser.add_argument("--scope", default="hub_io")
    args = parser.parse_args()

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={args.task}"])

    print("SMOKE phase=dataset", flush=True)
    dataset = instantiate(cfg.data.train)
    sample = default_collate([dataset[0]])
    print(
        f"SMOKE train_windows={len(dataset)} task={sample['task_name'][0]}",
        flush=True,
    )
    print("SMOKE phase=model_init", flush=True)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda:0")
    print("SMOKE phase=load_checkpoint", flush=True)
    model.load_checkpoint(str(cfg.resume))
    params = model.configure_trainable_parameters(args.scope)
    print(f"SMOKE trainable_params={sum(p.numel() for p in params)}", flush=True)

    optimizer = torch.optim.AdamW(params, lr=float(cfg.learning_rate))
    torch.cuda.reset_peak_memory_stats()
    print("SMOKE phase=forward", flush=True)
    loss, metrics = model.training_loss(sample)
    print(f"SMOKE loss={float(loss.detach())} metrics={metrics}", flush=True)
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
