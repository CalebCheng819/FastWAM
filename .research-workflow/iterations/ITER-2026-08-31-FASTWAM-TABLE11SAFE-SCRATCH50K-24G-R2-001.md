# ITER-2026-08-31-FASTWAM-TABLE11SAFE-SCRATCH50K-24G-R2-001

- Experiment: `FASTWAM-TABLE11SAFE-VG1H1GAU1-SCRATCH50K-S42-24G-R2-20260831`
- Notion page: `3cc21e77-89cc-8102-b55e-ed80c242e465`
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
- Retain the newest two complete resumable checkpoint tuples; invalidate the
  weight `COMPLETE` marker first and fail closed on links, special files,
  multiply-linked files, incomplete tuples, or concurrent tree changes. The
  one-step preflight disables retention because it does not publish checkpoints.
- Use new R2 run, attempt, preflight, output, latch, and Notion identities; submit with Priority 7 exactly once after all gates pass.

## Status

- Phase: `SOURCE_FREEZE_IN_PROGRESS`
- Chronicle marker written and strictly read back before code mutation.
- Exact configuration semantic diff passed: model/defaults, data wiring, batch, learning rate, schedule, objective, world size, and 50,000-step target are unchanged.
- Rolling-retention target tests passed (`37 passed` across retention, checkpoint
  load-order, and launcher contracts). The expanded available local suite passed
  (`320 passed`, `3 deselected`, `2 subtests passed`); compileall, launcher
  `bash -n`, and `git diff --check` also passed. The three deselections are the
  previously isolated local permission/optional-`boto3` environment cases.
- Immutable source and refreshed controller bindings are being frozen; neither
  the 1x8 preflight nor the 3x8 formal job has been submitted yet.
- Frozen runtime/source commit: `7a99d93dcc14cd8b8afeb962b589b67c79ea89e1`.
- Rotated preflight identity: `fastwam-table11safe-vg1h1gau1-scratch-preflight-s42-8g-r8-20260831`; source bundle target: `fastwam-table11safe-scratch50k-source-r9-20260831.bundle`.
- The exact Experiment ID resolves to the unique Notion page above. The page is `Planned`, with no Job ID or runtime/result fields populated.
- No preflight submission, formal CreateJob, or optimizer step has occurred yet.
