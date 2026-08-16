# FastWAM PlaceFood-rf matched-panel GAU0 evaluation — R22

R22 preserves the complete R21 scientific contract and changes only the SAPIEN EGL
bootstrap. R21 failed before evaluator startup because SAPIEN 3.0.1 called
`os.listdir("/usr/share/glvnd/egl_vendor.d")` on a directory absent from the DLC image.
No R21 episode ran and no formal output was published; that failed identity remains
immutable.

The R22 worker creates a private, ordinary-file EGL vendor manifest in its temporary
scratch tree. The manifest points to the already frozen and validated NVIDIA EGL vendor
library and is selected through `__EGL_VENDOR_LIBRARY_FILENAMES` before SAPIEN import.
This follows SAPIEN's own supported guard and does not modify the container filesystem,
inject a different driver, permit CPU rendering, or weaken any dependency or source gate.

Every candidate graphics profile must still create and close the real frozen
`PlaceFood-rf` environment on CUDA device 0 inside an isolated, timeout-bounded process.
The selected profile is then reused unchanged for all 16 formal episodes: the exact same
eight frozen states with `--no-gaussian-conditioning`, once using historical GAU1
normalization statistics and once using GAU0-native statistics.

Completion requires provider success plus the frozen terminal validator: eight complete
episodes per arm, exact paired state/seed panel, 29 declared artifacts, comparison report,
terminal receipt, and `COMPLETE` written last.
