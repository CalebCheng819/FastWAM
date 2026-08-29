# ITER-2026-08-29-FASTWAM-TABLE11SAFE-SCRATCH50K-24G-001

- Experiment: `FASTWAM-MR-TABLE11SAFE-VG1H1GAU1-SCRATCH50K-S42-24G-R1-20260829`
- Run: `fastwam-table11safe-vg1h1gau1-scratch50k-s42-24g-r1-20260829`
- Code treatment commit: pending publication
- Requested action: replace the illegal-data continuation with a new Table11-safe training run whose optimizer, scheduler, and global step all start at zero.

## Decision

Load only the official generic FastWAM model parameters from `libero_uncond_2cam224.pt`. Do not load the old N234 GAU1 task checkpoint and do not restore any task checkpoint's optimizer, scheduler, epoch, sampler, random state, or global step. This is a new task-training run from global step zero, while retaining the generic pretrained representation rather than randomly initializing the approximately six-billion-parameter model.

The only throughput adjustment is increasing DataLoader workers from 4 to 8. Model architecture, trainable scope, loss functions, Table11-safe data and statistics, sample order, Gaussian conditioning, text embeddings, world size 24, global batch 24, learning rate, warmup schedule, and 50,000 optimizer-update target remain unchanged.

## Audited inputs

- Dataset: `/oss-chengjuntao/robofactory/table/robofactory-table-11task-200each-h256-2g-stateful-safe-r3-20260827/tasks`
- Coverage: 11 H5 files, 11 tasks, 2,200 trajectories.
- Statistics: `/oss-chengjuntao/fastwam-assets/robofactory/table11-200each-h256-stateful-safe-r3-s42/stats/train-stats.json`
- Text embeddings: `/oss-chengjuntao/fastwam-assets/robofactory/table11-200each-h256-stateful-safe-r3-s42/text-embeds`
- Gaussian cache: `/oss-chengjuntao/fastwam-assets/robofactory/table11-200each-h256-stateful-safe-r3-s42/gaussian/compact-s42-13x28x40-fp16-meanalpha-direct-v1`, 89,977 selected frames.
- Generic pretrained model: `/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/yuanty-fastwam-139eebb6d90cdd9bdbbe465f72c6edc9ad5a518a/libero_uncond_2cam224.pt`, 12,041,735,140 bytes.
- Forbidden initialization source: the old N234 GAU1 `step_005000.pt` checkpoint and every optimizer/state artifact from the illegal-data continuation.

## Launch contract

- PAI DLC, priority 7, three workers, eight GPUs per worker.
- Per-device batch 1, accumulation 1, world size 24, global batch 24.
- Global step 0 to 50,000 with a newly constructed optimizer and scheduler.
- Full recoverable state and model weights at 5,000-step intervals through step 50,000.
- Canonical output: `/oss-chengjuntao/artifacts/fastwam-table11safe-vg1h1gau1-scratch50k-s42-24g-r1-20260829`.

## Gates

1. Freeze and publish a clean immutable source commit and source bundle.
2. Validate the exact Hydra composition, topology, source weight, and step-zero initialization contract.
3. Run exactly one 1-worker x 8-GPU, one-update real-data preflight and require a finite optimizer step plus successful provider terminal state.
4. Only after the preflight passes, re-read and precisely stop the superseded job `dlcglfwnemj3y76y`; preserve its output and do not retry it.
5. Submit exactly one priority-7, 3-worker x 8-GPU formal job with a permanent pre-create receipt and live identity reconciliation.
6. Distinguish scheduler acceptance, pod startup, actual optimizer progress, durable checkpoints, and terminal completion in all reporting.

## Current state

Planned. Source implementation and static validation are in progress. The superseded illegal-data continuation remains running until the new source and one-update real-data preflight are proven ready.
