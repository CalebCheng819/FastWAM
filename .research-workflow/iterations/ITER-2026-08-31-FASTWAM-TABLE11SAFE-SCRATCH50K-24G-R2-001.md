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

- Phase: `RUNNING`
- Chronicle marker written and strictly read back before code mutation.
- Exact configuration semantic diff passed: model/defaults, data wiring, batch, learning rate, schedule, objective, world size, and 50,000-step target are unchanged.
- Rolling-retention target tests passed (`37 passed` across retention, checkpoint
  load-order, and launcher contracts). The expanded available local suite passed
  (`320 passed`, `3 deselected`, `2 subtests passed`); compileall, launcher
  `bash -n`, and `git diff --check` also passed. The three deselections are the
  previously isolated local permission/optional-`boto3` environment cases.
- Immutable source and refreshed controller bindings passed their source,
  configuration, and exactly-once launch gates.
- The first R8 preflight audit stopped before its audit/latch/ACK/output or any
  cloud mutation because the controller expected the bundle head name `HEAD`,
  while the immutable bundle deliberately exposes only
  `refs/bundles/fastwam-table11safe-scratch50k-r9`. The controller now accepts
  exactly that frozen ref at exactly the frozen runtime commit and rejects
  `HEAD`, alternate refs, alternate commits, and multiple heads; the focused
  controller suite passes (`15 passed`, `4 subtests passed`).
- Frozen runtime/source commit: `7a99d93dcc14cd8b8afeb962b589b67c79ea89e1`.
- Rotated preflight identity: `fastwam-table11safe-vg1h1gau1-scratch-preflight-s42-8g-r8-20260831`; source bundle target: `fastwam-table11safe-scratch50k-source-r9-20260831.bundle`.
- The 1-worker x 8-GPU step-0-to-step-1 preflight completed successfully as
  DLC job `dlc1baelu2qkl1uw` before formal submission.
- The formal 3-worker x 8-GPU job was submitted exactly once as
  `dlcx57qq2xf7sn7p` at Priority 7. The permanent submission receipt records
  CreateJob RequestId `01A0548C-F657-5A2B-B78C-18666BAE1072`; this identity
  must never be retried or recreated.
- Runtime source was bound to commit
  `7a99d93dcc14cd8b8afeb962b589b67c79ea89e1`; the submission controller was
  bound to commit `3823452bb4b44d9b96afb0a74ca72dcefc66aa56`.
- Live runtime gates passed on all 3 nodes / 24 ranks: distributed CUDA
  validation, staged source/config barrier, joint-safe Table11 train and val
  indexing, and the official generic `libero_uncond_2cam224.pt` weight-only
  initialization. The run then created optimizer state from scratch and
  announced `initial_global_step=0`, `max_steps=50000`, and
  `optimizer_steps_this_run=50000`.
- The effective rank-0 DataLoader contract was logged as `num_workers=2`,
  `prefetch_factor=1`, `persistent_workers=False`, and `pin_memory=True`; worker
  IDs 0 and 1 both started successfully.
- First confirmed optimizer progress: `2026-08-31T05:34:25+08:00`, epoch 0,
  step `10/50000`, total loss `2.4186`, action loss `2.1234`, video loss
  `0.2952`, gradient norm `7.3022`, and learning rate `4.40e-07`.
- At that observation the provider and all three pods were `Running`, Priority
  remained 7, pod UIDs were unchanged, `RestartTimes` and `RestartRecord` were
  empty, and no Bus error, OOM, traceback, or `ChildFailedError` was present.
- This is verified live optimizer progress, not a durable checkpoint or terminal
  success. Recoverability first becomes provable after the complete step-1000
  checkpoint tuple is published and independently read back; terminal success
  still requires provider success plus the formal completion evidence.
