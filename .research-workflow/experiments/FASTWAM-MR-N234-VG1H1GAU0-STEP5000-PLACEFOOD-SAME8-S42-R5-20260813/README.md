# GAU0 PlaceFood same-panel DLC evaluation

This frozen experiment evaluates the archived FastWAM GAU0 step-5000 checkpoint with Gaussian conditioning disabled.

- Primary matched arm: GAU0 evaluated with the historical GAU1 evaluator stats.
- Sensitivity arm: the same GAU0 checkpoint evaluated with its native training stats.
- Each arm runs the same eight PlaceFood-rf panel episodes, environment seeds, policy seeds, horizons, and inference settings as the historical GAU1 evaluation.
- The two arms are sequential; each arm uses eight isolated one-GPU evaluator processes concurrently.
- This is a matched evaluation, not an isolated causal Gaussian ablation: the GAU0 and GAU1 checkpoints differ in training lineage and trainable scope.
- Provider `Succeeded` is insufficient. Scientific completion requires the frozen validator to accept all 16 invocations, the closed output allowlist, the terminal receipt, and the `COMPLETE` marker written last.

The submission request is permanently latched before its one allowed `CreateJob` call and uses DLC Priority 7.

R5 is a new, isolated identity after R4 correctly passed the legacy checkpoint
schema gate but failed before the first evaluation episode because the frozen
worker Python environment could not import `mani_skill`. The archived GAU0 checkpoint
uses the original native-v2 full-checkpoint envelope and predates both the
`state_kind` field and `action_attention_topology` metadata. R5 accepts that
legacy envelope only when the evaluation target has Gaussian conditioning
disabled, the complete top-level key set is exact, `training_mode=joint`,
`trainable_scope=dit`, and all seven original architecture metadata fields are
present with the expected values. The existing full-checkpoint loader still
requires the entire model state key set, tensor shapes, tensor dtypes, and
strict state loading to match before evaluation begins.

The frozen source import binding introduced for R3 is retained. R5 freezes the
worker to Python 3.12, adds a separate immutable dependency layer for missing
pure-Python packages, places RoboFactory first on `PYTHONPATH`, imports
`utils.scenes` before ManiSkill task modules, and executes the same dependency
and module-provenance preflight before prepare publication, before SDK loading
in submit, and again before any evaluator process starts. The R1, R2, R3, and
R4 latches remain preserved and are neither retried nor reused.

The fixed OSS durable-control directory may expose mount-projected mode `0777`.
The controller therefore treats it as an object-integrity boundary rather than
a confidentiality boundary: the path is frozen, it must be an ordinary
root-owned non-link directory, it must be empty before prepare, reservation
files are created with `O_EXCL|O_NOFOLLOW`, and every phase enforces a closed
child allowlist plus stable single-link file reads. The local `/run` state root
retains the stricter private-mode contract.
