# GAU0 PlaceFood same-panel evaluation R14

R14 is a new isolated run after the failed R13. It never reuses the R13 job,
durable state, local state, output root, source snapshot, or submission latch.

R13 created one DLC job, but all four evaluator processes segfaulted while
constructing the SAPIEN environment and produced zero episodes. Its controller
also detected that the worker's `SAPIEN_VULKAN_LIBRARY_PATH` differed from the
frozen request, but the thin entrypoint failed to propagate the controller's
nonzero return code. R14 makes that entrypoint fail closed, binds the worker to
the same Vulkan path frozen in the request, constructs and closes a real
PlaceFood environment before evaluation, and runs the four shards sequentially
to remove concurrent SAPIEN initialization from the execution path.

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
