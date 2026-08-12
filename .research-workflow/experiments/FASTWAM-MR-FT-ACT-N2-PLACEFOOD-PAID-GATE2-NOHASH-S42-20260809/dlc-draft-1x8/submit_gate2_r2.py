#!/usr/bin/env python3
"""Independent R2 identity for the fixed Gate2 source snapshot."""

from pathlib import Path

import submit_gate2 as controller


SOURCE_EXPERIMENT_ID = (
    "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-S42-20260809"
)
controller.EXPERIMENT_ID = (
    "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-R2-S42-20260809"
)
controller.SUBMISSION_TAG_PREFIX = "fastwam-gate2-nohash-r2-s42"
controller.REAL_PREFLIGHT_REL = Path(
    ".research-workflow/experiments"
) / SOURCE_EXPERIMENT_ID / "real_data_nohash_preflight.py"
controller.ENTRYPOINT_REL = (
    Path(".research-workflow/experiments")
    / SOURCE_EXPERIMENT_ID
    / "dlc-draft-1x8/runtime.sh"
)


if __name__ == "__main__":
    controller.main()
