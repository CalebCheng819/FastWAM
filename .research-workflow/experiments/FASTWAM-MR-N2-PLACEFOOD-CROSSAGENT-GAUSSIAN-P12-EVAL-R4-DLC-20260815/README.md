# P12 R4 DLC evaluation

This run evaluates P12 checkpoints `step_000500` and `step_001000` without
using the occupied DSW GPUs.

- Offline: fixed 263-state teacher-forcing panel, both checkpoints.
- Closed loop: fixed PlaceFood-rf val8 panel, official TOPP, action horizon 32.
- Worker topology: one DLC pod with eight GPUs. Closed-loop checkpoints run in
  parallel on GPU groups `0-3` and `4-7` after the offline gate passes.
- Graphics: select the first runtime profile that can construct and close a
  real PlaceFood-rf environment. CPU rendering fallback is forbidden.
- Outputs: fresh OSS paths ending in `r4-dlc`; existing R2/R3 evidence is
  preserved.

The submit supervisor waits for both checkpoint `.COMPLETE` markers before it
requests a GPU job. A successful DLC state is operational evidence only; the
worker terminal record and evaluation aggregates are required for a scientific
result.
