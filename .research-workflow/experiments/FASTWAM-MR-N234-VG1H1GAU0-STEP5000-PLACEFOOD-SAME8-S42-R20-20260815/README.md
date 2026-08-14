# FastWAM PlaceFood-rf matched-panel GAU0 evaluation — R20

R20 preserves the R19 scientific contract and changes only the DLC graphics-runtime
bootstrap. The formal evaluator still runs the same eight frozen episodes twice with
`--no-gaussian-conditioning`: once with the historical GAU1 normalization statistics
and once with GAU0-native statistics. The historical GAU1 evaluation remains the paired
comparison baseline.

Before any formal episode, the worker now tries a closed list of graphics profiles in
isolated, timeout-bounded subprocesses. A profile is accepted only if it actually creates
and closes the frozen `PlaceFood-rf` environment on CUDA device 0. A crash or timeout is
contained to the probe process; CPU rendering is not an allowed fallback. The accepted
profile is then reused unchanged by all eight evaluator subprocesses, covering all
16 episode evaluations across the two statistics arms.

The DLC request deliberately contains no `LD_LIBRARY_PATH`, Vulkan ICD, EGL/GLX vendor,
or SAPIEN loader override. This allows the worker to begin with the GPU node's provider
runtime instead of the DSW-frozen graphics namespace that caused the R19 SAPIEN device
segmentation fault. Static imports and source/input bindings are still validated before
the adaptive environment-construction probe.

Completion still requires provider success plus the frozen terminal validator: eight
complete episodes in each GAU0 statistics arm, the exact paired panel, the expected 29
artifact files, the comparison report, terminal receipt, and `COMPLETE` written last.
