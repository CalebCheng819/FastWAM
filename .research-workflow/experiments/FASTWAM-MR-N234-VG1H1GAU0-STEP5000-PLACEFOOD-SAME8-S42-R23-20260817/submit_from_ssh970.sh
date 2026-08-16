#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_root="$(cd -- "${script_dir}/../../.." && pwd -P)"

export PYTHONDONTWRITEBYTECODE=1
export FASTWAM_EXPERIMENT_REL_OVERRIDE='.research-workflow/experiments/FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R23-20260817'
export FASTWAM_LOCK_NAME_OVERRIDE='gau0-placefood-same8-r23-controller.lock'
export FASTWAM_WRAPPER_ENTRYPOINT="$(readlink -f -- "${BASH_SOURCE[0]}")"

exec /bin/bash "${source_root}/.research-workflow/experiments/FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R10-20260814/submit_from_ssh970.sh" "$@"
