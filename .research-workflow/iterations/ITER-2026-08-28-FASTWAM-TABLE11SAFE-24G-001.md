# ITER-2026-08-28-FASTWAM-TABLE11SAFE-24G-001

- Experiment: `FASTWAM-MR-TABLE11SAFE-VG1H1GAU1-CONT50K-S42-24G-R1-20260828`
- Code treatment commit: `e43858c6f339f5bffbd76cd600d65fef735adefa`
- Requested action: restart the 24-GPU GAU1 50k treatment on the newly uploaded joint-safe Table11 collection.

## Decision

Run a new, isolated experiment from the original GAU1 step-5000 weights. Do not resume the old run's optimizer or step-45000 state, because the demonstrations, action normalization statistics, and Gaussian conditioning cache all change together. Keep the original 24-GPU batch contract so that the data treatment remains the principal experimental change.

## Audited inputs

- Dataset: `/oss-chengjuntao/robofactory/table/robofactory-table-11task-200each-h256-2g-stateful-safe-r3-20260827/tasks`
- Coverage: 11 H5 files, 11 tasks, 2200 trajectories.
- Action statistics: 1,458,478 fitted actions from 1,954 training trajectories.
- Sealed Gaussian selection: 89,977 frames across N=1/2/3/4 robot cases.
- Initial weights: GAU1 cumulative step 5000, 12,047,213,728 bytes, weights-only.
- Text embeddings: reused only because the fixed 11 instructions and Qwen text model are unchanged; each cache file is checked before launch.

## Launch contract

- PAI DLC, priority 7, three workers, eight GPUs per worker.
- Per-device batch 1, accumulation 1, global batch 24.
- Cumulative global step 5000 to 50000, fresh optimizer, cosine LR 1e-4 with 2250 warmup steps.
- Full recoverable state and weight checkpoints at 10000, 15000, ..., 50000.
- Canonical outputs under `/oss-chengjuntao/artifacts/fastwam-table11safe-vg1h1gau1-cont50k-s42-24g-r1-20260828`.

## Gates

1. Complete and validate new-data Gaussian direct-compact cache.
2. Run static launcher and config tests from the immutable code bundle.
3. Run a real GPU short-run that performs an optimizer update with finite loss.
4. Create the Notion experiment record and read it back.
5. Submit exactly one DLC job, reconcile by run ID, then distinguish queued, pod-started, and optimizer-running states.

## Current state

Planned. Submission is forbidden until the Gaussian cache and real short-run gates pass. The Notion experiment record was created and read back as page `3ca21e77-89cc-81b3-95a8-ef016fe0b4e0` with the exact treatment commit and run ID.
