# FastWAM PlaceFood-rf matched-panel GAU0 evaluation — R23

R23 preserves the full R22 scientific contract and fixes the complete provider-native
graphics loader chain. R19 reached the real `PlaceFood-rf` constructor but segfaulted
after SAPIEN fell back to its builtin Vulkan loader. R22 supplied a private EGL manifest
but failed before episode 0 because the loaded EGL object did not expose
`eglQueryString`. Neither failure is an evaluation result, and both identities remain
immutable.

The R23 worker builds a private GLVND/Vulkan shim from the frozen ordinary files for
EGL, GL, GLES1/2, OpenGL, GLX, GLdispatch, the NVIDIA EGL vendor, and the Vulkan loader.
It binds the frozen Vulkan and EGL manifests, verifies every target and declared byte
size, loads the complete dispatch chain through `ctypes`, checks the GLVND frontend
`eglQueryString`, NVIDIA vendor `__egl_Main`, and
`vkEnumerateInstanceVersion`, and imports the full environment dependency set. It then
must create and close the real `PlaceFood-rf` environment on CUDA device 0. R23 forces
the shared evaluator to accept only this provider-native profile; it cannot fall back to
a different system loader or CPU rendering.

The evaluator still runs the exact same eight frozen states and seeds twice with
`--no-gaussian-conditioning`: once with historical GAU1 normalization statistics and
once with GAU0-native statistics. Completion requires 16 complete episodes, the exact
paired panel, 29 declared artifacts, the comparison report, provider success, terminal
receipt, and `COMPLETE` written last. The comparison measures inference-time Gaussian
and statistics sensitivity of the frozen checkpoint; it is not a train-time causal
ablation because that checkpoint was trained with Gaussian conditioning enabled.
