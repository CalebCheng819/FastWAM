### ITER-2026-08-17-P13-R6-GRAPHICS-RETRY — Repair P13 Vulkan startup and resubmit once

- 时间：2026-08-17T18:30:30+08:00
- 类型：Code, configuration, and experiment-state update
- 动机：P13 cache R5 reached the dedicated 1x8 GPU worker but failed before cache construction because SAPIEN could not create a Vulkan instance. The failed job and output identity must remain immutable while the still-open lineage is repaired.
- 变更：Add an exact one-GPU SAPIEN environment-construction probe across bounded Vulkan/EGL profiles; require a pinned CPFS Vulkan loader; select and reapply only the first passing profile; add a new R6 Priority-7 request, worker, and exactly-once submitter with a permanent latch and distinct run, output, runtime, and source-bundle identities.
- Git：Runtime source commit `60de16ef0628d70f58c3349a182c2fe8be3ade2c` on `exp/placefood-metric-gaussian-p13-20260814`, pushed to the fork. R6 launch artifacts are versioned in the same branch before publication.
- 实验：`FASTWAM-MR-N2-PLACEFOOD-METRIC-P13-S42-8G-R1-20260815`; Notion page `3bc21e77-89cc-81c0-b4b2-cb2feb8698e4`; failed R5 job `dlc8sjnka1s9woh9` is preserved; planned R6 run `fastwam-p13-metric-cache-s42-8g-r6-graphics-probed-dedicated-20260817`.
- 结果：R6 graphics/submission regression suites pass (14 targeted tests); Python compile, two Bash syntax checks, request JSON validation, and `git diff --check` pass. Full discovery is environment-limited because the system interpreter lacks pytest, h5py, numpy, and torch; no discovered assertion failure was observed. R6 submission is pending immutable publication and remote audit.
- 决策：Do not retry or overwrite R5. Publish the frozen R6 source and controller files, run audit-only first, then make at most one Priority-7 CreateJob call after the permanent latch is durable. Do not submit P13 training until the versioned R6 cache has a strict COMPLETE terminal.
- 下一步：Publish R6 to its unique runtime root, verify direct bytes and live graphics inputs, submit exactly once, and monitor until the exact environment-construction probe and cache builder are running normally.
- 记录人：chengjuntao / Codex
