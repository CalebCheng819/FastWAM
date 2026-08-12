# FastWAM R5 PlaceFood train-layout evaluation

Experiment ID: `FASTWAM-MR-FT-ACT-N2-PLACEFOOD-TRAINLAYOUT-EVAL-R5-20260813`

This confirmatory closed-loop evaluation asks whether the R5 N2 FastWAM policy
performs better when RoboFactory starts from states recorded in the exact
training split. It compares eight train-split initial states against eight
validation-split initial states. The two panels use paired policy seeds; only
the source H5 trajectory and its raw initial environment state differ.

## Frozen contract

- Task: `PlaceFood-rf`, two native robot agents.
- Checkpoint: R5 N2 action-only step 1000.
- Initial state: raw state stored in the selected demonstration trajectory.
- Rollout: 300 environment steps maximum, execute horizon 5, action horizon 32,
  20 diffusion inference steps.
- Split: seed 42, validation proportion 0.1, exact
  `sorted_trajectory_ordinal_splitmix64_v1` assignment.
- Panels: `panels/train8.json` and `panels/val8.json`.
- Success: RoboFactory simulator success returned by the same formal evaluator;
  offline action loss is not used as a success substitute.
- Provenance: ordinary Git revision, paths, timestamps, run IDs, file sizes, and
  schemas; no new hashes or hash chains.

Formal artifact root:

`/oss-chengjuntao/artifacts/fastwam-placefood-trainlayout-eval-r5-20260813`

## Execution

Run one episode on a selected physical GPU:

```bash
./run_one.sh train 0 0 /oss-chengjuntao/artifacts/fastwam-placefood-trainlayout-eval-r5-20260813
```

The first train episode is the smoke gate. The remaining 15 episodes may start
only after its manifest is terminal, its summary is COMPLETED, the policy emits
actions, the environment advances, and the output structure is valid. A run is
never overwritten or retried under the same episode directory.

After all 16 episode directories are terminal, aggregate them exactly once:

```bash
/opt/venvs/gaudp-robofactory-py310/bin/python -B aggregate_results.py \
  --root /oss-chengjuntao/artifacts/fastwam-placefood-trainlayout-eval-r5-20260813
```

The aggregate is valid only when every train and validation episode passes the
identity, split, terminal-state, environment-step, and policy-query checks in
`aggregate_results.py`.
