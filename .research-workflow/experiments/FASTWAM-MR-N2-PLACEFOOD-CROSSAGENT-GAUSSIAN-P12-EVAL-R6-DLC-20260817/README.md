# P12 R6 DLC evaluation

This run evaluates P12 checkpoints `step_000500` and `step_001000` without
using the occupied DSW GPUs.

- Offline: fixed 263-state teacher-forcing panel, both checkpoints.
- Closed loop: fixed PlaceFood-rf val8 panel, official TOPP, action horizon 32.
- Worker topology: one DLC pod with eight GPUs. Closed-loop checkpoints run in
  parallel on GPU groups `0-3` and `4-7` after the offline gate passes.
- Graphics probe mode: request the resource-group-supported 1x8 shape, then
  restrict the probe process to GPU0 while validating Vulkan/EGL and a real
  PlaceFood-rf construction before any full evaluation submission.
- Graphics: select the first runtime profile that can construct and close a
  real PlaceFood-rf environment. CPU rendering fallback is forbidden.
- Outputs: fresh OSS paths ending in `r6-dlc`; existing R2/R3/R5 evidence is
  preserved. Probe outputs go to `...-graphics-probe` controller directory.

The submit supervisor fails closed unless both checkpoint `.COMPLETE` markers
already exist. Submit the GPU0-only probe in a legal 1x8 allocation first. The
8-GPU full evaluation is rejected until the same R6 probe record contains both a `SUCCEEDED`
`graphics-probe-summary.json` and a matching `SUCCEEDED`, return-code-zero
`worker-terminal.json`. Probe and full modes each write a permanent exclusive
submission latch before their sole `CreateJob` call. `--audit-only` performs no
durable or local writes.

A successful DLC state is operational evidence only; the worker terminal
record and evaluation aggregates are required for a scientific result.
