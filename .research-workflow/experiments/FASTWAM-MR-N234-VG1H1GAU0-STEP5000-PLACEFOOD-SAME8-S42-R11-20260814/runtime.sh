#!/usr/bin/env bash
set -Eeuo pipefail

export FASTWAM_EXPERIMENT_REL_OVERRIDE='.research-workflow/experiments/FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R11-20260814'
export FASTWAM_SCRATCH_TEMPLATE_OVERRIDE='/tmp/fastwam-gau0-placefood-r11.XXXXXXXX'

exec /bin/bash "${FASTWAM_SOURCE_ROOT}/.research-workflow/experiments/FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R10-20260814/runtime.sh" "$@"
