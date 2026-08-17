# P12 R8 DLC evaluation

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
  real PlaceFood-rf environment. R8 satisfies SAPIEN's import-time EGL vendor
  contract with a private NVIDIA manifest, prefers the provider/system NVIDIA
  Vulkan and GLVND stack, and keeps the CPFS Mesa/driver mixture as the final
  fallback only. Every rejected profile writes a persistent log and prints its
  last 80 lines in the main worker log. CPU rendering fallback is forbidden.
- Outputs: fresh OSS paths ending in `r8-dlc`; existing R2/R3/R5 evidence is
  preserved. Probe outputs go to `...-graphics-probe` controller directory.

The submit supervisor fails closed unless both checkpoint `.COMPLETE` markers
already exist. Submit the GPU0-only probe in a legal 1x8 allocation first. The
8-GPU full evaluation is rejected until the same R8 probe record contains both a `SUCCEEDED`
`graphics-probe-summary.json` and a matching `SUCCEEDED`, return-code-zero
`worker-terminal.json`. Probe and full modes each write a permanent exclusive
submission latch before their sole `CreateJob` call. `--audit-only` performs no
durable or local writes.

The frozen request remains strict: Priority 7, non-spot, no oversold request,
and `EnableRDMA=false`. Post-submit `GetJob` accepts only the controlled provider
normalizations observed in the failed R7 and first R8 readbacks:
`OversoldType` absent to an empty string, `EnableRDMA` false to true, and an
empty requested `CustomEnvs` projected to a public one-to-one list of the exact
requested `Envs`. The detailed response may also omit `MountAccess` from each
otherwise exact, ordered data-source entry and omit the top-level
`JobMaxRunningTimeMinutes` and `SuccessPolicy`. A returned access mode, duration,
or success policy must remain exact. Reordered, missing, extra, or changed data
sources and duplicate, private, missing, extra, or changed environment entries
still fail closed. Every other frozen request field must remain equal, and the
accepted normalizations are persisted in the receipt.

If the sole `CreateJob` succeeds but strict post-create readback stops before a
receipt is written, `--reconcile` validates the permanent latch, recorded Job
ID, unique provider candidate, and full frozen request before completing the
durable receipt/state. Reconciliation has no CreateJob path and must never be
used to retry or replace the latched job.

A successful DLC state is operational evidence only; the worker terminal
record and evaluation aggregates are required for a scientific result.
