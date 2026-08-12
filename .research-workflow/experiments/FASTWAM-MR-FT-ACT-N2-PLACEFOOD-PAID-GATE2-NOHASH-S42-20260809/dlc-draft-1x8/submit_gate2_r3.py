#!/usr/bin/env python3
"""Independent R3 identity for an explicitly prepared Gate2 OSS snapshot."""

from pathlib import Path

import submit_gate2 as controller


SOURCE_EXPERIMENT_ID = (
    "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-S42-20260809"
)
controller.EXPERIMENT_ID = (
    "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-R3-S42-20260809"
)
controller.SUBMISSION_TAG_PREFIX = "fastwam-gate2-nohash-r3-s42"
controller.DISPLAY_NAME_PREFIX = "fw-g2-nh-r3-s42"
controller.CONTROL_ENTRYPOINT = "submit_from_ssh970_r3.sh"
controller.SOURCE_PREFIX = Path(
    "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots"
)
# The exact direct-child snapshot is supplied to ``prepare --source-root`` and
# then frozen in the durable prepared binding.  R3 deliberately has no stale
# source name baked into executable code; execute revalidates the bound path
# and complete direct-content metadata before acquiring submission authority.
controller.APPROVED_SOURCE_ROOT = None
controller.REAL_PREFLIGHT_REL = Path(
    ".research-workflow/experiments"
) / SOURCE_EXPERIMENT_ID / "real_data_nohash_preflight.py"
controller.ENTRYPOINT_REL = (
    Path(".research-workflow/experiments")
    / SOURCE_EXPERIMENT_ID
    / "dlc-draft-1x8/runtime.sh"
)
controller.STRUCTURED_EVIDENCE_REL = (
    Path(".research-workflow/experiments")
    / SOURCE_EXPERIMENT_ID
    / "dlc-draft-1x8/gate2_structured_evidence.py"
)


if __name__ == "__main__":
    controller.main()
