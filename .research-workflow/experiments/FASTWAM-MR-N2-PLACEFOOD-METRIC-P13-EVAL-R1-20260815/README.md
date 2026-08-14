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

For unattended execution on the DSW with the verified Vulkan runtime, launch
the fail-closed supervisor instead:

```bash
python scripts/supervise_p13_training_and_eval.py \
  --training-receipt /oss-chengjuntao/artifacts/fastwam-placefood-metric-gaussian-p13-supervisor-r1-20260815/submission-receipt.json \
  --expected-training-output /oss-chengjuntao/artifacts/fastwam-placefood-metric-gaussian-p13-s42-8g-r1-20260815 \
  --checkpoint /oss-chengjuntao/artifacts/fastwam-placefood-metric-gaussian-p13-s42-8g-r1-20260815/checkpoints/weights/step_001000.pt \
  --record-root /oss-chengjuntao/artifacts/fastwam-placefood-metric-gaussian-p13-eval-supervisor-r1-20260815 \
  --lock-root /mnt/workspace/experiments/FASTWAM-P13-EVAL-SUPERVISOR-R1-20260815/runtime \
  --teacher-script .research-workflow/experiments/FASTWAM-MR-N2-PLACEFOOD-METRIC-P13-EVAL-R1-20260815/run_teacher_forcing.sh \
  --closedloop-script .research-workflow/experiments/FASTWAM-MR-N2-PLACEFOOD-METRIC-P13-EVAL-R1-20260815/run_closedloop_h32.sh \
  --teacher-output /oss-chengjuntao/artifacts/fastwam-placefood-metric-gaussian-p13-paired-tf-r1-20260815 \
  --closedloop-output /oss-chengjuntao/artifacts/fastwam-placefood-metric-gaussian-p13-official-topp-h32-val8-r1-20260815 \
  --run-eval
```

The supervisor does not allocate GPUs while waiting. It requires a successful
DLC training terminal state, a non-empty checkpoint and completion marker, and
the same validated metric cache recorded in the training receipt. It then uses
one genuinely idle GPU for teacher forcing and four genuinely idle GPUs for the
closed-loop panel. Existing incomplete or failed outputs are preserved and
cause a fail-closed stop instead of being overwritten.

The closed-loop launcher fails unless the offline run has terminal status
`SUCCEEDED`. Offline error is an endpoint diagnostic, not evidence of task
success; the val8 closed-loop aggregate is the behavioral result.
