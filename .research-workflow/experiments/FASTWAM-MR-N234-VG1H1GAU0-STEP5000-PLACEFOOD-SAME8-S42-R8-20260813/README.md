# GAU0 PlaceFood same-panel DLC evaluation

This frozen experiment evaluates the archived FastWAM GAU0 step-5000 checkpoint with Gaussian conditioning disabled.

- Primary matched arm: GAU0 evaluated with the historical GAU1 evaluator stats.
- Sensitivity arm: the same GAU0 checkpoint evaluated with its native training stats.
- Each arm runs the same eight PlaceFood-rf panel episodes, environment seeds, policy seeds, horizons, and inference settings as the historical GAU1 evaluation.
- The two arms are sequential; each arm uses eight isolated one-GPU evaluator processes concurrently.
- This is a matched evaluation, not an isolated causal Gaussian ablation: the GAU0 and GAU1 checkpoints differ in training lineage and trainable scope.
- Provider `Succeeded` is insufficient. Scientific completion requires the frozen validator to accept all 16 invocations, the closed output allowlist, the terminal receipt, and the `COMPLETE` marker written last.

The submission request is permanently latched before its one allowed `CreateJob` call and uses DLC Priority 7.

R8 is a new, isolated identity after R7 passed its Python, source, data, and
checkpoint preflights but failed before the first evaluator process. The DLC
image did not provide the unversioned GLVND `libEGL.so.1` name, so importing the
RoboFactory scene stack left PyOpenGL's `eglQueryString` unresolved. R8 freezes
the versioned EGL frontend, GL dispatch library, and NVIDIA EGL vendor library;
it constructs a worker-private `libEGL.so.1` shim, puts both controlled library
directories on `LD_LIBRARY_PATH`, and performs a direct ctypes/PyOpenGL probe
before importing RoboFactory or starting an evaluator. The failed R7 job,
latch, run identity, and empty output prefix remain preserved and are not
retried or reused.

R5 had been a new, isolated identity after R4 correctly passed the legacy checkpoint
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

The frozen source import binding introduced for R3 is retained. R8 retains the
worker to Python 3.10, adds a separate immutable dependency layer for missing
pure-Python packages, places RoboFactory first on `PYTHONPATH`, imports
`utils.scenes` before ManiSkill task modules, and executes the same dependency
and module-provenance preflight before prepare publication, before SDK loading
in submit, and again before any evaluator process starts. The R1 through R7
latches remain preserved and are neither retried nor reused.

The fixed OSS durable-control directory may expose mount-projected mode `0777`.
The controller therefore treats it as an object-integrity boundary rather than
a confidentiality boundary: the path is frozen, it must be an ordinary
root-owned non-link directory, it must be empty before prepare, reservation
files are created with `O_EXCL|O_NOFOLLOW`, and every phase enforces a closed
child allowlist plus stable single-link file reads. The local `/run` state root
retains the stricter private-mode contract.
