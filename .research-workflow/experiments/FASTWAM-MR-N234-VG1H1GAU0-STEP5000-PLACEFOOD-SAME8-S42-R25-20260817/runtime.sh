#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export FASTWAM_RUNTIME_EXPERIMENT_REL='.research-workflow/experiments/FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R25-20260817'
export FASTWAM_RUNTIME_GENERATION='R25'
exec /bin/bash "${script_dir}/../FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R23-20260817/runtime.sh"
