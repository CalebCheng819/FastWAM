# GAU0 PlaceFood same-panel evaluation R13

R13 is a new isolated run after R12. It never reuses the R12 job, durable
state, local state, output root, source snapshot, or submission latch.

R12 stopped during controller prepare before source validation, worker
preflight, SDK loading, or CreateJob. Its identity wrapper rebound constants on
the R11 wrapper module, while the actual R10 implementation still resolved the
R11 source root through its own function globals. R13 loads the actual R10
implementation directly and binds every run identity in the namespace used by
`main` and `prepare`. Regression tests verify that execution namespace, not
only the outer wrapper attributes. The R12 GL/Vulkan preflight correction is
retained unchanged.

One 1x8-GPU DLC job evaluates the GAU0 checkpoint on the exact historical eight
PlaceFood validation episodes. Four evaluator processes each run two episodes
sequentially. Each episode is capped at 300 environment steps and 60 policy
queries. The `gau1_stats` arm uses the historical GAU1 evaluator statistics and
is the primary matched comparison; the `gau0_native_stats` arm repeats the same
panel with GAU0-native statistics as a sensitivity analysis. Both arms
explicitly disable Gaussian conditioning.

The historical GAU1 checkpoint scored 0/8 on this panel. Because the GAU0 and
GAU1 checkpoints also differ in training lineage and trainable scope, the final
comparison is evidence about these concrete checkpoints, not a pure causal
Gaussian ablation.
