# GAU0 PlaceFood same-panel evaluation R12

R12 is a new isolated run after R11. It never reuses the R11 job, durable
state, local state, output root, source snapshot, or submission latch.

R11 stopped during the controller's worker dependency preflight, before
prepare, SDK loading, or CreateJob. Importing DeepSpeed imported OpenCV and
prepended an absent OpenCV `lib64` path to `LD_LIBRARY_PATH`; the preflight then
mistook that mutable first entry for the GL/Vulkan shim root. R12 explicitly
freezes the verified shim root in `FASTWAM_GL_SHIM_ROOT` before heavyweight
imports and uses that immutable path for the Vulkan loader check. This is an
environment-preflight correction and does not change the scientific protocol.

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
