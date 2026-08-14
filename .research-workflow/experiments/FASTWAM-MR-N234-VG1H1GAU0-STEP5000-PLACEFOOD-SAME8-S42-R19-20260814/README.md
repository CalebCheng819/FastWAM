# FastWAM PlaceFood GAU0 same-panel evaluation R19

R19 preserves the exact checkpoint, historical eight-episode PlaceFood-rf
panel, seeds, horizons, diffusion steps, serial four-shard order, and two
statistics arms from R18. The evaluated model is unchanged and inference keeps
`--no-gaussian-conditioning`.

R18 created exactly one Priority-7 DLC job, `dlc12z7itm259r6z`. Its worker
passed the frozen source, input, import, and narrow EGL preflights, then exited
with native status 139 while constructing the first ManiSkill/SAPIEN
environment. No episode was evaluated and no scientific output was published;
the R18 job, latch, ACK, and empty output namespace remain preserved.

R19 changes only the graphics-loader runtime and all run identities. It creates
a private, closed-allowlist GLVND namespace containing the unversioned and
runtime sonames for EGL, GL, GLES1, GLES2, GLX, and OpenGL. The loader order is
exactly the private namespace, the frozen NVIDIA GLVND frontend directory, the
frozen NVIDIA driver directory, and CUDA. Inherited loader and Python paths are
not admitted, and `SAPIEN_VULKAN_LIBRARY_PATH` remains absent. This exact
configuration was exercised read-only on the target DSW host through a real
`PlaceFood-rf` environment construction before R19 was frozen.

The scientific output is a matched inference-time comparison against the
existing GAU1 historical baseline on the same eight episodes. GAU0 is evaluated
twice: once with the GAU1 training statistics and once with GAU0-native
statistics. The comparison diagnoses inference-time Gaussian and normalization
sensitivity. It is not a train-time causal Gaussian ablation because the frozen
checkpoint was trained with Gaussian conditioning enabled.
