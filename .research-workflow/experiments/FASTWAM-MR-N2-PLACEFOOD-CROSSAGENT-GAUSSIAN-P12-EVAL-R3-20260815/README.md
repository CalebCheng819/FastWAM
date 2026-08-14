# FastWAM P12 cross-agent Gaussian evaluation

This experiment evaluates P12 `cross_agent_gaussian_v4` at steps 500 and 1000.
P12 is an action-only continuation of P10 that adds a learned remote-agent gate
and cross-agent spatial Gaussian attention. Deployment computes NoPoSplat
features online from the current observations.

## Fixed comparisons

- Offline: the fixed 263 PlaceFood teacher-forcing states, beginning at
  timestep 5, with H1, H5, and full-horizon action errors.
- Closed loop: the fixed validation panel of eight episodes, raw initial state,
  action horizon 32, official TOPP control, 60 policy queries, and a 30,000
  simulator-step limit.
- Checkpoints: P12 steps 500 and 1000 from the same training run.
- Training revision: `1181a375c880a4a51df2ae78d533e16dde757465`.
- Integrity: Git revisions, paths, byte counts, modification times, and
  completion markers. No new content hashes are generated.

## Execution

`run_teacher_forcing.sh` evaluates both checkpoints and writes one comparison
record. `run_closedloop_h32.sh` evaluates the checkpoint selected by
`P12_EVAL_STEP`, which must be `000500` or `001000`.

`supervise_eval.sh` waits for both checkpoint completion markers and idle DSW
GPUs, then runs teacher forcing followed by the two closed-loop panels. It does
not allocate GPUs while waiting, does not overwrite existing output, and keeps
each checkpoint in a separate OSS directory.

Offline action error is a checkpoint diagnostic. Only the complete eight-run
closed-loop aggregates are behavioral success evidence.
