from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / ".research-workflow" / "experiments" / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R1-20260812"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


controller = _load("gau0_placefood_controller", EXPERIMENT / "controller.py")
aggregate = _load("gau0_placefood_aggregate", EXPERIMENT / "aggregate_results.py")


def _provider_job(request: dict) -> dict:
    job = copy.deepcopy(request)
    job.pop("Envs")
    job.pop("JobMaxRunningTimeMinutes")
    job.pop("SuccessPolicy")
    job["CustomEnvs"] = [
        {"Key": key, "Value": value, "Visible": "public"}
        for key, value in request["Envs"].items()
    ]
    job["DataSources"] = [
        {"DataSourceId": item["DataSourceId"], "MountPath": item["MountPath"], "Uri": ""}
        for item in request["DataSources"]
    ]
    return job


def test_wrapper_freezes_control_python_link_and_resolved_targets():
    wrapper = (EXPERIMENT / "submit_from_ssh970.sh").read_text(encoding="utf-8")
    assert "CONTROL_PYTHON_LINK_TARGET='python3'" in wrapper
    assert "CONTROL_PYTHON_RESOLVED_TARGET='/usr/local/bin/python3.12'" in wrapper
    assert 'readlink -- "${CONTROL_PYTHON}"' in wrapper
    assert 'readlink -f -- "${CONTROL_PYTHON}"' in wrapper


def test_validate_python_freezes_immediate_and_final_targets(tmp_path: Path, monkeypatch):
    final = tmp_path / "python3.10-final"
    final.write_bytes(b"#!/bin/sh\nexit 0\n")
    final.chmod(0o700)
    intermediate = tmp_path / "python3.10"
    intermediate.symlink_to(final)
    entry = tmp_path / "python"
    entry.symlink_to(intermediate)

    monkeypatch.setattr(controller, "PYTHON", entry)
    monkeypatch.setattr(controller, "PYTHON_TARGET", intermediate)
    monkeypatch.setattr(controller, "PYTHON_RESOLVED_TARGET", final)
    controller.validate_python()

    entry.unlink()
    entry.symlink_to(final)
    with pytest.raises(controller.ContractError, match="symlink target changed"):
        controller.validate_python()

    entry.unlink()
    entry.symlink_to(intermediate)
    wrong_final = tmp_path / "wrong-final"
    wrong_final.write_bytes(b"#!/bin/sh\nexit 0\n")
    wrong_final.chmod(0o700)
    monkeypatch.setattr(controller, "PYTHON_RESOLVED_TARGET", wrong_final)
    with pytest.raises(controller.ContractError, match="resolved target changed"):
        controller.validate_python()


def test_runtime_env_freezes_both_worker_python_targets():
    env = controller.runtime_env("a" * 40)
    assert env["FASTWAM_PYTHON_TARGET"] == str(controller.PYTHON_TARGET)
    assert env["FASTWAM_PYTHON_RESOLVED_TARGET"] == str(controller.PYTHON_RESOLVED_TARGET)


def test_exact_job_accepts_only_priority7_frozen_projection():
    request = controller.request_body("a" * 40)
    job = _provider_job(request)
    assert request["Priority"] == 7
    assert controller.exact_job(request, job)

    changed = copy.deepcopy(job)
    changed["Priority"] = 1
    assert not controller.exact_job(request, changed)

    changed = copy.deepcopy(job)
    changed["CustomEnvs"].append({"Key": "UNDECLARED", "Value": "1", "Visible": "public"})
    assert not controller.exact_job(request, changed)

    changed = copy.deepcopy(job)
    changed["DataSources"][0]["MountPath"] = "/wrong"
    assert not controller.exact_job(request, changed)

    changed = copy.deepcopy(job)
    changed["WorkspaceId"] = int(request["WorkspaceId"])
    assert not controller.exact_job(request, changed)


@pytest.mark.parametrize("reader", [controller.stable_read, aggregate.stable_read])
def test_stable_read_rejects_symlink_and_hardlink(tmp_path: Path, reader):
    source = tmp_path / "source.json"
    source.write_bytes(b"{}\n")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(source)
    with pytest.raises((controller.ContractError, RuntimeError, OSError)):
        reader(symlink)

    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(source)
    with pytest.raises((controller.ContractError, RuntimeError)):
        reader(source)


def test_validate_arm_requires_exact_eight_no_gaussian_invocations(tmp_path: Path):
    arm = "gau1_stats"
    for index in range(8):
        shard = tmp_path / arm / f"episode-{index:02d}"
        shard.mkdir(parents=True)
        episode = {
            "task_index": index,
            "panel_index": index,
            "environment_seed": aggregate.EXPECTED_ENV_SEEDS[index],
            "policy_seed": aggregate.EXPECTED_POLICY_SEEDS[index],
            "task_name": "PlaceFood-rf",
            "status": "completed",
            "success": index == 7,
            "steps": 300,
            "policy_queries": 60,
            "action_bound_violations": index,
        }
        (shard / "episodes.jsonl").write_text(json.dumps(episode) + "\n", encoding="utf-8")
        (shard / "summary.json").write_text(
            json.dumps({"status": "PASS", "episodes_requested": 1, "episodes_completed": 1, "infrastructure_errors": 0}) + "\n",
            encoding="utf-8",
        )
        (shard / "run_manifest.json").write_text(
            json.dumps({"argv": ["--task", "PlaceFood-rf", "--no-gaussian-conditioning"]}) + "\n",
            encoding="utf-8",
        )

    result = aggregate.validate_arm(tmp_path, arm)
    assert result["episodes_completed"] == 8
    assert result["successes"] == 1
    assert result["total_steps"] == 2400
    assert result["policy_queries"] == 480
    assert result["action_bound_violations"] == sum(range(8))

    manifest = tmp_path / arm / "episode-03" / "run_manifest.json"
    manifest.write_text(json.dumps({"argv": ["--task", "PlaceFood-rf"]}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not prove GAU0 execution"):
        aggregate.validate_arm(tmp_path, arm)
