#!/usr/bin/env python3
"""Independent R7 identity after the terminal, permanently retired R6 attempt."""

import importlib.util
import stat
import sys
from pathlib import Path


_HERE = Path(__file__).resolve(strict=True).parent
_CONTROLLER_PATH = _HERE / "submit_gate2.py"
_controller_info = _CONTROLLER_PATH.lstat()
if not stat.S_ISREG(_controller_info.st_mode) or _CONTROLLER_PATH.is_symlink():
    raise RuntimeError("R7 controller must be an exact sibling regular file")
_controller_spec = importlib.util.spec_from_file_location(
    "fastwam_gate2_r7_controller", _CONTROLLER_PATH
)
if _controller_spec is None or _controller_spec.loader is None:
    raise RuntimeError("unable to create the exact R7 controller import spec")
controller = importlib.util.module_from_spec(_controller_spec)
sys.modules[_controller_spec.name] = controller
_controller_spec.loader.exec_module(controller)


SOURCE_EXPERIMENT_ID = (
    "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-S42-20260809"
)
controller.EXPERIMENT_ID = (
    "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-R7-S42-20260811"
)
controller.SUBMISSION_TAG_PREFIX = "fastwam-gate2-nohash-r7-s42"
controller.DISPLAY_NAME_PREFIX = "fw-g2-nh-r7-s42"
controller.CONTROL_ENTRYPOINT = "submit_from_ssh970_r7.sh"
controller.SOURCE_PREFIX = Path(
    "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots"
)
# R7 can only use the replacement publication from the current reviewed tree.
# The incomplete source-r6 publication and every earlier tree are retired.
controller.APPROVED_SOURCE_ROOT = controller.SOURCE_PREFIX / (
    "fastwam-action-n234-formal-20260811-r7"
)
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
