#!/usr/bin/env bash
set -Eeuo pipefail

# The frozen R17 runtime creates the sole permitted worker-only difference: a
# private libEGL.so.1 link prepended to the two-entry request loader namespace.
unset PYTHONPATH LD_LIBRARY_PATH SAPIEN_VULKAN_LIBRARY_PATH FASTWAM_GL_SHIM_ROOT
export FASTWAM_EXPERIMENT_REL_OVERRIDE='.research-workflow/experiments/FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R18-20260814'
export FASTWAM_SCRATCH_TEMPLATE_OVERRIDE='/tmp/fastwam-gau0-placefood-r18.XXXXXXXX'

exec /bin/bash "${FASTWAM_SOURCE_ROOT}/.research-workflow/experiments/FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R17-20260814/runtime.sh"
