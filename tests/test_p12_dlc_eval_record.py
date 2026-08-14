from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = (
    "FASTWAM-MR-N2-PLACEFOOD-CROSSAGENT-GAUSSIAN-P12-EVAL-R4-DLC-20260815"
)
RECORD = ROOT / ".research-workflow" / "experiments" / EXPERIMENT_ID


def _load_submit_module():
    path = RECORD / "submit_eval.py"
    spec = importlib.util.spec_from_file_location("p12_r4_dlc_submit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_request_uses_one_eight_gpu_worker_and_fresh_outputs() -> None:
    module = _load_submit_module()
    request = module.request_body("a" * 40)
    assert request["JobSpecs"][0]["PodCount"] == 1
    assert request["JobSpecs"][0]["ResourceConfig"]["GPU"] == "8"
    assert request["Settings"]["EnableRDMA"] is False
    assert request["Envs"]["P12_INTEGRITY_MODE"] == "metadata_no_hash"
    assert request["Envs"]["P12_TF_OUTPUT_ROOT"].endswith("r4-dlc")
    assert request["Envs"]["P12_STEP500_OUTPUT_ROOT"].endswith("r4-dlc")
    assert request["Envs"]["P12_STEP1000_OUTPUT_ROOT"].endswith("r4-dlc")


def test_runtime_has_gpu_probe_and_disjoint_closed_loop_groups() -> None:
    runtime = (RECORD / "runtime.sh").read_text()
    assert "_build_environment(root, \"PlaceFood-rf\")" in runtime
    assert "no GPU graphics profile could construct" in runtime
    assert "P12_EVAL_GPUS='0 1 2 3'" in runtime
    assert "P12_EVAL_GPUS='4 5 6 7'" in runtime
    assert "P12_R4_DLC_EVAL_SCIENTIFIC_COMPLETE" in runtime


def test_submit_supervisor_has_local_exclusive_lock() -> None:
    submit = (RECORD / "submit_eval.py").read_text()
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in submit
    assert "another P12 R4 DLC submit supervisor is active" in submit
