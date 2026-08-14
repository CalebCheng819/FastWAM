# FastWAM PlaceFood GAU0 same-panel evaluation R18

R18 preserves the exact R17 evaluator, checkpoint, historical eight-episode
panel, seeds, horizons, diffusion steps, serial shard order, and both statistics
arms. It changes only the worker environment contract and all run identities.

R17 created exactly one Priority-7 DLC job, `dlc7lmi2y16cjuuk`. It failed before
episode 0 after the real EGL runtime preflight passed: the base controller then
compared the worker-created private EGL shim in `LD_LIBRARY_PATH` against the
two-entry frozen request value and rejected the expected runtime difference.
R17's job, latch, ACK, output namespace, and failure evidence remain immutable.

R18 permits precisely one worker-only transformation: the one-link private EGL
shim is prepended to the frozen NVIDIA driver-lib and CUDA loader entries. It
rejects any suffix or other loader entry, validates the shim target and file
size, requires `SAPIEN_VULKAN_LIBRARY_PATH` to remain absent, and continues to
compare every other request environment field exactly. The runtime clears
inherited `PYTHONPATH` and `LD_LIBRARY_PATH` before the frozen R17 runtime builds
its deterministic namespaces.

The scientific output remains a matched comparison of no-Gaussian inference
against the existing GAU1 historical baseline on the exact same eight
PlaceFood-rf episodes. The two GAU0 arms use the GAU1 training statistics and
GAU0-native statistics respectively. This diagnoses normalization sensitivity;
it is not by itself a pure causal Gaussian ablation because the evaluated model
checkpoint was trained in the GAU1 architecture.
