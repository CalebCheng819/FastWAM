# GAU0 PlaceFood same-panel evaluation R11

R11 is a new isolated run after R10. It never reuses the R10 job, durable state,
local state, output root, source snapshot, or submission latch.

R10 reached the worker but all four evaluator processes crashed before producing
an episode. SAPIEN could not find a system Vulkan loader in the controlled
runtime namespace, fell back to its bundled loader, and then segfaulted. R11
keeps the complete GLVND front-end namespace from R10 and adds the ordinary
CPFS-persisted DSW Vulkan loader as both `libvulkan.so.1` and `libvulkan.so`.
The controller and worker call `vkEnumerateInstanceVersion` before any durable
prepare state or evaluator launch, set `SAPIEN_VULKAN_LIBRARY_PATH`, and enable
Python faulthandler.

The scientific protocol is unchanged: one 1x8-GPU DLC job evaluates the GAU0
checkpoint on the exact historical eight PlaceFood validation episodes. Four
evaluator processes each run two episodes sequentially. Each episode is capped
at 300 environment steps and 60 policy queries. The `gau1_stats` arm uses the
historical GAU1 evaluator statistics and is the primary matched comparison; the
`gau0_native_stats` arm repeats the same panel with GAU0-native statistics as a
sensitivity analysis. Both arms explicitly disable Gaussian conditioning.

The historical GAU1 checkpoint scored 0/8 on this panel. Because the GAU0 and
GAU1 checkpoints also differ in training lineage and trainable scope, the final
comparison is evidence about these concrete checkpoints, not a pure causal
Gaussian ablation.
