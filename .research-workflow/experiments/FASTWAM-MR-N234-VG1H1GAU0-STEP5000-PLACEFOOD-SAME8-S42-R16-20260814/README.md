# GAU0 PlaceFood same-panel evaluation R16

R15 created exactly one Priority-7 DLC job, `dlcmjdkwapb2pwhy`. Its frozen
source, request, latch, acknowledgement, and job identity all remained exact,
but SAPIEN crashed with exit code 139 while constructing the first environment.
It completed zero episodes and published no output. R15 is retained as a failed
record and is never retried or reused by R16.

The R15 worker used a manual GLVND frontend shim, Vulkan loader symbols,
and ctypes preloads. Serialization removed concurrent environment creation but
did not remove that graphics ABI boundary. R16 instead uses the minimal NVIDIA vendor-driver namespace
previously exercised by the successful GAU1 evaluator:
`driver-lib` followed by the CUDA runtime, the NVIDIA ICD and EGL vendor
descriptors, and no `SAPIEN_VULKAN_LIBRARY_PATH` or custom GL shim. Both the
controller and worker reject those overrides before importing the evaluator
stack. The worker then constructs and closes one real PlaceFood environment
before any episode evaluation, so a graphics initialization failure fails
closed before the formal panel begins.

R16 has new experiment, run, source, output, durable-state, local-state, and lock
namespaces. Its wrapper propagates controller failures, exports
`PYTHONDONTWRITEBYTECODE=1`, and runs the four evaluator shards sequentially.
Publication validation disables pytest's cache provider so the published source
remains an ordinary direct-byte copy of the frozen Git commit.

One 1x8-GPU DLC job evaluates the GAU0 checkpoint on the exact historical eight
PlaceFood validation episodes. Four shards each run two episodes, one shard at
a time. Each episode is capped at 300 environment steps and 60 policy queries.
The `gau1_stats` arm uses the historical GAU1 evaluator statistics and is the
primary matched comparison; the `gau0_native_stats` arm repeats the same panel
with GAU0-native statistics as a sensitivity analysis. Both arms explicitly
disable Gaussian conditioning.

The historical GAU1 checkpoint scored 0/8 on this panel. Because the GAU0 and
GAU1 checkpoints also differ in training lineage and trainable scope, the final
comparison is evidence about these concrete checkpoints, not a pure causal
Gaussian ablation.
