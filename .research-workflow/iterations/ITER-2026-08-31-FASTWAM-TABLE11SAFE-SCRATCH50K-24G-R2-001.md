# ITER-2026-08-31-FASTWAM-TABLE11SAFE-SCRATCH50K-24G-R2-001

- Experiment: `FASTWAM-MR-TABLE11SAFE-VG1H1GAU1-SCRATCH50K-S42-24G-R2-20260831`
- Base revision: `3b594a9`
- Parent failed run: `fastwam-table11safe-vg1h1gau1-scratch50k-s42-24g-r1-20260829` / `dlct0cxzes6kj364`
- Failure boundary: DataLoader worker PID 8728 received `SIGBUS` at step 1630; no durable training checkpoint or terminal completion marker was published.
- Initialization: the same official generic FastWAM weight `libero_uncond_2cam224.pt`; optimizer, scheduler, and global step start from zero. The failed R1 state is not resumed.

## Frozen training semantics

- Keep VG1/H1/GAU1 architecture and action-plus-video training objectives unchanged.
- Keep the repaired Table11-safe 11-task dataset, Gaussian cache, text cache, stats, task mix, and sample ordering unchanged.
- Keep 3 workers x 8 GPUs, per-device batch 1, gradient accumulation 1, world-size/global batch 24, learning rate `1e-4`, cosine schedule, warmup 2250, and 50,000 optimizer updates unchanged.

## Authorized operational changes

- Reduce DataLoader workers per rank from 8 to 2.
- Set DataLoader `prefetch_factor=1` and `persistent_workers=false` explicitly.
- Log DataLoader multiprocessing, worker PID, file-descriptor limits, and `/dev/shm` capacity at startup for diagnosis.
- Save recoverable checkpoints every 1,000 updates instead of every 5,000 updates.
- Use new R2 run, attempt, preflight, output, latch, and Notion identities; submit with Priority 7 exactly once after all gates pass.

## Status

- Phase: `CODE_VALIDATED`
- Chronicle marker written and strictly read back before code mutation.
- Exact configuration semantic diff passed: model/defaults, data wiring, batch, learning rate, schedule, objective, world size, and 50,000-step target are unchanged.
- Validation passed: `316 passed, 3 deselected, 2 subtests passed`; the three deselections are environment-only checks requiring unavailable write permissions or `boto3`. Three additional collection modules require unavailable local `h5py`; the directly relevant controller/trainer suite passed `64/64`.
- Python compile, Bash syntax, and whitespace checks passed.
- No code commit, source publication, preflight submission, formal CreateJob, or optimizer step yet.
