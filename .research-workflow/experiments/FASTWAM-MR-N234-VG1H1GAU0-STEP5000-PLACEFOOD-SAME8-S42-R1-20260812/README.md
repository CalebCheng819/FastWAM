# GAU0 PlaceFood same-panel DLC evaluation

This frozen experiment evaluates the archived FastWAM GAU0 step-5000 checkpoint with Gaussian conditioning disabled.

- Primary matched arm: GAU0 evaluated with the historical GAU1 evaluator stats.
- Sensitivity arm: the same GAU0 checkpoint evaluated with its native training stats.
- Each arm runs the same eight PlaceFood-rf panel episodes, environment seeds, policy seeds, horizons, and inference settings as the historical GAU1 evaluation.
- The two arms are sequential; each arm uses eight isolated one-GPU evaluator processes concurrently.
- This is a matched evaluation, not an isolated causal Gaussian ablation: the GAU0 and GAU1 checkpoints differ in training lineage and trainable scope.
- Provider `Succeeded` is insufficient. Scientific completion requires the frozen validator to accept all 16 invocations, the closed output allowlist, the terminal receipt, and the `COMPLETE` marker written last.

The submission request is permanently latched before its one allowed `CreateJob` call and uses DLC Priority 7.

The fixed OSS durable-control directory may expose mount-projected mode `0777`.
The controller therefore treats it as an object-integrity boundary rather than
a confidentiality boundary: the path is frozen, it must be an ordinary
root-owned non-link directory, it must be empty before prepare, reservation
files are created with `O_EXCL|O_NOFOLLOW`, and every phase enforces a closed
child allowlist plus stable single-link file reads. The local `/run` state root
retains the stricter private-mode contract.
