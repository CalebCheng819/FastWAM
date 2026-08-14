# FastWAM P13 metric-Gaussian evaluation

This experiment evaluates the P13 `metric_gaussian_v5` action policy without
Policy-Lightning, NoPoSplat, or a rendered Gaussian teacher at deployment time.
The policy constructs its 13-channel, 60x80 metric geometry online from the
current calibrated RoboFactory depth observations.

## Fixed comparisons

- Offline: the same 263 PlaceFood teacher-forcing states used for P10, starting
  at timestep 5, with H1, H5, full-horizon arm/gripper errors.
- Closed loop: the fixed `val8` panel with raw initial state, action horizon 32,
  official TOPP control, 60 policy queries, and 30,000 simulator-step budget.
- Training checkpoint: P13 `step_001000.pt`, initialized from P10 step 1000.
- Training source revision: `e5f20bbf91477b82990e5c571d54305c639705c6`.
- Integrity policy: Git revision plus file path, byte count, modification time,
  completion markers, and cache metadata. No new content hashes are generated.

## Launch order

Set `P13_MODEL_ROOT` to a clean checkout of the fixed training revision and set
`P13_METRIC_CACHE_ROOT` to the exact completed metric cache used by training.
Then run:

```bash
bash run_teacher_forcing.sh
bash run_closedloop_h32.sh
```

The closed-loop launcher fails unless the offline run has terminal status
`SUCCEEDED`. Offline error is an endpoint diagnostic, not evidence of task
success; the val8 closed-loop aggregate is the behavioral result.
