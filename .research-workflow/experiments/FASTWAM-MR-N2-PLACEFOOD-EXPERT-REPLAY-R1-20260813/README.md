# FastWAM PlaceFood expert replay

Experiment ID: `FASTWAM-MR-N2-PLACEFOOD-EXPERT-REPLAY-R1-20260813`

This control experiment replays the complete stored two-arm expert action
stream on the same frozen eight-train and eight-validation PlaceFood panels used
for the R5 policy evaluation. It tests the environment, raw H5 state restore,
action ordering, controller mode, temporal alignment, and simulator success
criterion without loading FastWAM, NoPoSplat, Gaussian features, or checkpoints.

## Frozen contract

- Task: `PlaceFood-rf`, two native robot agents.
- Initial state: untouched H5 state at timestep zero.
- Actions: same-timestep H5 expert actions for both agents, no replanning.
- Horizon: at most 300 simulator steps or the available expert action count.
- Panels: the exact R5 split-seed-42 `train8.json` and `val8.json` files.
- Success: RoboFactory simulator `info.success`.
- Artifacts: summaries, traces, videos, and logs only under OSS.
- Provenance: Git revision, paths, timestamps, run IDs, sizes, and schemas; no
  new hashes or hash chains.

Formal artifact root:

`/oss-chengjuntao/artifacts/fastwam-placefood-expert-replay-20260813-r1`

Run the fixed panel on selected physical GPUs:

```bash
FASTWAM_EVAL_GPU_IDS=0,1,2 ./run_panel.sh \
  /oss-chengjuntao/artifacts/fastwam-placefood-expert-replay-20260813-r1
```

The aggregate is emitted only after all 16 episode directories satisfy the
formal expert-replay contract and terminal-state checks.
