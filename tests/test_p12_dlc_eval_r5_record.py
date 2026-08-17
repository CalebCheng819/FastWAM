from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = (
    "FASTWAM-MR-N2-PLACEFOOD-CROSSAGENT-GAUSSIAN-P12-EVAL-R5-DLC-20260816"
)
RECORD = ROOT / ".research-workflow" / "experiments" / EXPERIMENT_ID


def _load_submit_module():
    path = RECORD / "submit_eval.py"
    spec = importlib.util.spec_from_file_location("p12_r5_dlc_submit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_request_uses_one_eight_gpu_worker_and_fresh_outputs() -> None:
    module = _load_submit_module()
    request = module.request_body("a" * 40, "full_eval")
    assert request["JobSpecs"][0]["PodCount"] == 1
    assert request["JobSpecs"][0]["ResourceConfig"]["GPU"] == "8"
    assert request["Settings"]["EnableRDMA"] is False
    assert request["Envs"]["P12_INTEGRITY_MODE"] == "metadata_no_hash"
    assert request["Envs"]["P12_RUN_MODE"] == "full_eval"
    assert request["Envs"]["P12_TF_OUTPUT_ROOT"].endswith("r5-dlc")
    assert request["Envs"]["P12_STEP500_OUTPUT_ROOT"].endswith("r5-dlc")
    assert request["Envs"]["P12_STEP1000_OUTPUT_ROOT"].endswith("r5-dlc")
    assert request["DisplayName"] == module.DISPLAY_NAME
    assert request["Priority"] == 7
    assert "OversoldType" not in request["Settings"]
    assert request["JobSpecs"][0]["ElasticSpotSpecs"] == []


def test_request_uses_one_gpu_probe_worker() -> None:
    module = _load_submit_module()
    request = module.request_body("a" * 40, "graphics_probe")
    assert request["DisplayName"] == module.PROBE_DISPLAY_NAME
    assert request["JobSpecs"][0]["ResourceConfig"]["GPU"] == "1"
    assert request["JobSpecs"][0]["ResourceConfig"]["CPU"] == "16"
    assert request["JobSpecs"][0]["ResourceConfig"]["Memory"] == "120Gi"
    assert request["Envs"]["P12_RUN_MODE"] == "graphics_probe"
    assert request["Settings"]["Tags"]["topology"] == "1x1"


def test_runtime_has_gpu_probe_and_disjoint_closed_loop_groups() -> None:
    runtime = (RECORD / "runtime.sh").read_text()
    assert "_build_environment(root, \"PlaceFood-rf\")" in runtime
    assert "no GPU graphics profile could construct" in runtime
    assert "cpfs_manifest_headless" in runtime
    assert "P12_EVAL_GPUS='0 1 2 3'" in runtime
    assert "P12_EVAL_GPUS='4 5 6 7'" in runtime
    assert "P12_R5_DLC_EVAL_SCIENTIFIC_COMPLETE" in runtime


def test_submit_supervisor_has_local_exclusive_lock() -> None:
    submit = (RECORD / "submit_eval.py").read_text()
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in submit
    assert "another P12 R5 DLC submit supervisor is active" in submit


def test_submitter_latches_before_create_job_and_has_read_only_audit() -> None:
    submit = (RECORD / "submit_eval.py").read_text()
    assert submit.index("write_exclusive(latch_path, latch)") < submit.index(
        "client.create_job(request)"
    )
    assert 'mode.add_argument("--audit-only"' in submit
    assert '"create_job_called": False' in submit
    assert "def list_jobs(" in submit
    assert "ListJobs pagination mismatch" in submit


def test_full_eval_requires_matching_successful_probe_terminal(tmp_path: Path) -> None:
    module = _load_submit_module()
    (tmp_path / "graphics-probe-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "fastwam-p12-dlc-gpu-probe-v1",
                "status": "SUCCEEDED",
                "graphics_profile": "cpfs_manifest_headless",
            }
        )
    )
    (tmp_path / "worker-terminal.json").write_text(
        json.dumps(
            {
                "schema_version": "fastwam-p12-dlc-eval-worker-terminal-v1",
                "status": "SUCCEEDED",
                "return_code": 0,
                "graphics_profile": "cpfs_manifest_headless",
            }
        )
    )
    gate = module.validate_probe_terminal(tmp_path)
    assert gate["graphics_profile"] == "cpfs_manifest_headless"
    assert gate["return_code"] == 0


def test_full_eval_rejects_failed_probe_terminal(tmp_path: Path) -> None:
    module = _load_submit_module()
    (tmp_path / "graphics-probe-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "fastwam-p12-dlc-gpu-probe-v1",
                "status": "SUCCEEDED",
                "graphics_profile": "cpfs_manifest_headless",
            }
        )
    )
    (tmp_path / "worker-terminal.json").write_text(
        json.dumps(
            {
                "schema_version": "fastwam-p12-dlc-eval-worker-terminal-v1",
                "status": "FAILED",
                "return_code": 1,
                "graphics_profile": "cpfs_manifest_headless",
            }
        )
    )
    try:
        module.validate_probe_terminal(tmp_path)
    except RuntimeError as error:
        assert "not successful" in str(error)
    else:
        raise AssertionError("failed graphics probe terminal was accepted")
