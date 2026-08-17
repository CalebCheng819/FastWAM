# GAU0 PlaceFood same-panel closed-loop evaluation R25

R25 preserves the R23 provider-native graphics and exact 16-episode panel while
making the normalization-statistics provenance explicit before any reservation
is published. R24 prepare stopped before durable prepare state and before episode
0 because the checkpoint-native legacy statistics file does not contain a `dim`
field; it is a launch failure, not a zero-percent evaluation result.

The `gau1_stats` arm must use the train-split statistics contract
(`split=train`, seed 42, validation fraction 0.1). The `gau0_native_stats` arm
may use only the checkpoint-bound legacy full-dataset statistics with the exact
source, population counts, cardinalities, file count, and trajectory count
frozen by this controller. Its historical schema has exactly `count`, `max`,
`mean`, `min`, and `std` per action/state record; those four numeric arrays must
be finite and exactly 8/18 elements, respectively. Cross-use, extra fields, and
schema or numeric drift fail before durable prepare state is written.

Both arms still disable inference-time Gaussian conditioning and use the same
step-5000 checkpoint, eight frozen PlaceFood states/seeds, and evaluator settings.
A result exists only after 16 complete episodes, provider terminal success, and
the strict `COMPLETE` validator. This measures normalization-statistics
sensitivity while Gaussian conditioning is disabled; it is not a train-time
causal ablation.

The final report also carries the already-completed historical `gau1_baseline`
panel as context. That third panel used a different checkpoint/training scope and
must not be treated as one of the two matched R25 arms or as evidence for a
train-time Gaussian ablation. The primary R25 comparison is strictly
`gau1_stats` versus `gau0_native_stats` under the shared checkpoint and shared
`--no-gaussian-conditioning` evaluator contract above.
