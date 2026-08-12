# FastWAM B4 PlaceFood closed-loop evaluation

Experiment ID: `FASTWAM-MR-B4-N2-PLACEFOOD-CLOSEDLOOP-EVAL-20260813`

This evaluation measures the B4 phase/gripper/contact action-finetuned checkpoint
under the same frozen PlaceFood train-layout protocol used for R5. It first runs
one complete train-split episode as a strict-load and closed-loop smoke gate,
then evaluates all remaining train and validation episodes.

## Frozen contract

- Task: `PlaceFood-rf`, two native robot agents.
- Checkpoint: B4 action-only step 2500.
- Initial state: raw state stored in the selected demonstration trajectory.
- Rollout: 300 environment steps maximum, execute horizon 5, action horizon 32,
  20 diffusion inference steps.
- Panels: the frozen R5 `train8.json` and `val8.json` panels, reused unchanged.
- Success: RoboFactory simulator success; offline action loss is not a success
  substitute.
- Provenance: Git revision, paths, timestamps, job/run IDs, file sizes, and
  schemas; no new hashes or hash chains.

Formal artifact root:

`/oss-chengjuntao/artifacts/fastwam-b4-placefood-closedloop-eval-20260813`

The train episode 00 smoke must finish with a terminal manifest, a completed
summary, at least one policy query, and at least one environment step before the
remaining 15 episodes are launched.
