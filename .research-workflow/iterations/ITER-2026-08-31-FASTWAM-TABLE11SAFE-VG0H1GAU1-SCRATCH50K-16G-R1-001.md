# ITER-2026-08-31-FASTWAM-TABLE11SAFE-VG0H1GAU1-SCRATCH50K-16G-R1-001

- Experiment: `FASTWAM-TABLE11SAFE-VG0H1GAU1-SCRATCH50K-S42-16G-R1-20260831`
- Base revision: `94284303174539ec883521f344d5ac2419fa8a61`
- Code treatment commit: `ee8bd8295047877b30b8bccddd9af7567502f88a`
- Parent configuration: current joint-safe Table11 scratch-training R2.
- Initialization: official generic FastWAM `libero_uncond_2cam224.pt` model
  weights only. Optimizer, scheduler, and global step start from zero; no
  checkpoint trained on the old joint-unsafe data is resumed.

## Frozen training semantics

- Dataset: repaired joint-safe Table11 r3 trajectories and their matching
  joint-safe stats, text caches, and Gaussian assets.
- Treatment: VG0/H1/GAU1. Train only the action path with
  `training_mode=action_only_cache`, disable future-video targets and video
  loss, and retain hub plus Gaussian conditioning.
- Topology: 2 workers x 8 GPUs, world size/global batch 16, per-device batch 1,
  gradient accumulation 1. This is intentionally not sample-budget-equivalent
  to the world-24 reference.
- Schedule: 50,000 fresh optimizer updates from global step 0, learning rate
  `1e-4`, cosine decay, and the existing scratch warmup contract.
- Reliability: DataLoader workers per rank 2, prefetch factor 1,
  `persistent_workers=false`, full-state checkpointing every 1,000 updates,
  retain the newest two complete resumable checkpoint tuples.

## Status

- Phase: `CODE FROZEN; PRELAUNCH PENDING`.
- New independent worktree and branch created; no changes were made to the
  currently running joint-safe R2 worktree.
- Formal and preflight submission scripts are bound to the immutable code
  treatment commit above and still require the exact published source bundle,
  prelaunch evidence, and preflight receipt before formal `CreateJob`.
- No preflight, DLC submission, retry, stop, CreateJob, or other cloud mutation
  has been performed. Notion has not been updated for this local-only variant.
- Validation: `bash -n` and Python compilation passed. The focused new 16-GPU
  suite plus the unchanged 24-GPU joint-safe regression suite passed with
  `31 passed, 8 subtests passed`.
- Renderer structure, 2x8 topology, Priority 7, corrected Table11 paths, VG0,
  H1, GAU1, and unpublished-commit fail-closed behavior are covered by the
  focused tests. A real pinned-bundle render remains gated on publishing the
  exact immutable source bundle; no synthetic publication was created to
  bypass that gate.
