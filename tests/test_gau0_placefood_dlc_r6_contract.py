from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / ".research-workflow" / "experiments" / "FASTWAM-MR-N234-VG1H1GAU0-STEP5000-PLACEFOOD-SAME8-S42-R6-20260813"


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


def test_validate_python_freezes_portable_entry_and_abi_not_container_final_path(tmp_path: Path, monkeypatch):
    final = tmp_path / "python3.10-final"
    final.write_bytes(b"#!/bin/sh\nexit 0\n")
    final.chmod(0o700)
    intermediate = tmp_path / "python3.10"
    intermediate.symlink_to(final)
    entry = tmp_path / "python"
    entry.symlink_to(intermediate)

    monkeypatch.setattr(controller, "PYTHON", entry)
    monkeypatch.setattr(controller, "PYTHON_TARGET", intermediate)
    calls = []

    class Result:
        returncode = 0
        stdout = json.dumps({
            "version": list(controller.PYTHON_VERSION),
            "cache_tag": controller.PYTHON_CACHE_TAG,
            "soabi": controller.PYTHON_SOABI,
        })
        stderr = ""

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return Result()

    monkeypatch.setattr(controller.subprocess, "run", fake_run)
    controller.validate_python()
    assert calls[0][0][:4] == [str(entry), "-B", "-I", "-c"]

    entry.unlink()
    entry.symlink_to(final)
    with pytest.raises(controller.ContractError, match="symlink target changed"):
        controller.validate_python()

    entry.unlink()
    entry.symlink_to(intermediate)
    Result.stdout = json.dumps({
        "version": [3, 12, 12],
        "cache_tag": controller.PYTHON_CACHE_TAG,
        "soabi": controller.PYTHON_SOABI,
    })
    with pytest.raises(controller.ContractError, match="Python ABI changed"):
        controller.validate_python()


def test_runtime_env_freezes_portable_worker_python_contract():
    env = controller.runtime_env("a" * 40)
    assert env["FASTWAM_PYTHON_TARGET"] == str(controller.PYTHON_TARGET)
    assert env["FASTWAM_PYTHON_VERSION"] == ".".join(str(item) for item in controller.PYTHON_VERSION)
    assert env["FASTWAM_PYTHON_CACHE_TAG"] == controller.PYTHON_CACHE_TAG
    assert env["FASTWAM_PYTHON_SOABI"] == controller.PYTHON_SOABI
    assert "FASTWAM_PYTHON_RESOLVED_TARGET" not in env
    assert env["FASTWAM_PYTHON_EXTRA_ROOT"] == str(controller.PYTHON_EXTRA_ROOT)


def test_runtime_imports_complete_worker_stack_from_frozen_paths_and_fails_closed():
    runtime = (EXPERIMENT / "runtime.sh").read_text(encoding="utf-8")
    expected_export = 'export PYTHONPATH="${FASTWAM_ROBOFACTORY_ROOT}:${FASTWAM_SOURCE_ROOT}/src:${FASTWAM_PYTHON_EXTRA_ROOT}:${FASTWAM_SOURCE_ROOT}/experiments/robofactory${PYTHONPATH:+:${PYTHONPATH}}"'
    assert expected_export in runtime
    assert "import utils.scenes as scenes" in runtime
    assert "import mani_skill" in runtime
    assert "import fastwam_multi_robot_policy as policy" in runtime
    assert "import tasks.place_food as place_food" in runtime
    assert "EGL_PLATFORM=surfaceless" in runtime
    assert "GAU0_FROZEN_RUNTIME_IMPORT_PASS" in runtime
    assert "worker PYTHONPATH prefix mismatch" in runtime
    assert "worker module provenance mismatch" in runtime
    assert 'callable(getattr(runtime, "create_multi_robot_fastwam", None))' in runtime


def test_r6_readme_records_r5_portability_failure_without_reusing_latch():
    readme = (EXPERIMENT / "README.md").read_text(encoding="utf-8")
    assert "R4 correctly passed the legacy checkpoint" in readme
    assert "failed before the first evaluation episode" in readme
    assert "could not import `mani_skill`" in readme
    assert "original native-v2 full-checkpoint envelope" in readme
    assert "predates both the" in readme
    assert "`state_kind` field" in readme
    assert "`action_attention_topology` metadata" in readme
    assert "Gaussian conditioning\ndisabled" in readme
    assert "complete top-level key set is exact" in readme
    assert "all seven original architecture metadata fields" in readme
    assert "entire model state key set, tensor shapes, tensor dtypes" in readme
    assert "Python 3.12" in readme
    assert "before SDK loading" in readme
    assert "pinned Python resolved target changed" in readme
    assert "The R1, R2, R3, R4, and R5 latches remain preserved" in readme


def test_dependency_preflight_is_before_every_mutation_or_evaluator_boundary():
    prepare_source = inspect.getsource(controller.prepare)
    assert prepare_source.index("validate_worker_dependencies()") < prepare_source.index("write_json_exclusive(")

    submit_source = inspect.getsource(controller.submit)
    assert submit_source.index("validate_worker_dependencies()") < submit_source.index("load_sdk()")
    assert submit_source.index("validate_worker_dependencies()") < submit_source.index("write_json_exclusive(LATCH_PATH")

    worker_source = inspect.getsource(controller.worker_preflight)
    assert worker_source.index("validate_worker_dependencies()") < worker_source.index('print("GAU0_WORKER_PREFLIGHT_PASS")')


def test_worker_dependency_env_and_import_program_are_frozen(monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)

        class Result:
            returncode = 0
            stdout = "GAU0_WORKER_DEPENDENCY_PREFLIGHT_PASS\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(controller, "validate_python", lambda: None)
    monkeypatch.setattr(controller, "require_dir", lambda path: None)
    monkeypatch.setattr(controller.subprocess, "run", fake_run)
    controller.validate_worker_dependencies()

    assert captured["argv"][:3] == [str(controller.PYTHON), "-B", "-c"]
    program = captured["argv"][3]
    assert program.index("import utils.scenes as scenes") < program.index("import mani_skill")
    assert "import boto3" in program
    assert "import git" in program
    assert "worker module provenance mismatch" in program
    assert captured["cwd"] == controller.SOURCE_ROOT
    env = captured["env"]
    assert env["PYTHONPATH"] == controller.worker_pythonpath()
    assert env["MUJOCO_GL"] == "egl"
    assert env["EGL_PLATFORM"] == "surfaceless"


def _project_root_ownership(monkeypatch: pytest.MonkeyPatch, *paths: Path) -> None:
    original_lstat = Path.lstat
    projected = {str(path) for path in paths}

    def root_owned_lstat(self: Path):
        info = original_lstat(self)
        if str(self) not in projected:
            return info
        fields = list(info)
        fields[4] = 0
        fields[5] = 0
        return os.stat_result(fields)

    monkeypatch.setattr(Path, "lstat", root_owned_lstat)


def test_oss_durable_root_accepts_only_empty_root_owned_projection(tmp_path: Path, monkeypatch):
    durable = tmp_path / "durable"
    durable.mkdir(mode=0o700)
    durable.chmod(0o777)
    _project_root_ownership(monkeypatch, durable)
    controller.validate_empty_oss_durable_root(durable)
    controller.ensure_empty_oss_durable_root(durable)

    (durable / "unexpected").write_text("x", encoding="utf-8")
    with pytest.raises(controller.ContractError, match="must be empty before prepare"):
        controller.validate_empty_oss_durable_root(durable)


def test_oss_durable_root_rejects_symlink_and_unrecognized_mode(tmp_path: Path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises((controller.ContractError, RuntimeError), match="canonical"):
        controller.validate_empty_oss_durable_root(linked)

    target.chmod(0o755)
    _project_root_ownership(monkeypatch, target)
    with pytest.raises(controller.ContractError, match="durable root contract failed"):
        controller.validate_empty_oss_durable_root(target)


def test_prepare_reservation_publication_has_closed_allowlist(tmp_path: Path, monkeypatch):
    durable = tmp_path / "durable"
    durable.mkdir(mode=0o700)
    _project_root_ownership(monkeypatch, durable)
    controller.ensure_empty_oss_durable_root(durable)
    reservation = {"schema": "test", "value": 1}
    path = durable / "prepared-reservation.json"
    controller.write_json_exclusive(path, reservation)
    assert controller.load_json(path) == reservation
    controller.require_exact_children(durable, {path.name})

    (durable / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(controller.ContractError, match="allowlist mismatch"):
        controller.require_exact_children(durable, {path.name})


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


def test_create_job_tags_are_server_compatible_frozen_map():
    body = controller.request_body("a" * 40)
    expected = {
        "experiment_id": controller.EXPERIMENT_ID,
        "run_id": controller.RUN_ID,
    }
    assert body["Settings"]["Tags"] == expected
    assert isinstance(body["Settings"]["Tags"], dict)

    class FakeCreateJobRequest:
        def __init__(self):
            self._body = None

        def from_map(self, value):
            self._body = copy.deepcopy(value)
            return self

        def validate(self):
            return None

        def to_map(self):
            return copy.deepcopy(self._body)

    class FakeModels:
        CreateJobRequest = FakeCreateJobRequest

    assert controller.sdk_request(FakeModels, body).to_map() == body

    broken = copy.deepcopy(body)
    broken["Settings"]["Tags"] = [
        {"Key": "experiment_id", "Value": controller.EXPERIMENT_ID},
        {"Key": "run_id", "Value": controller.RUN_ID},
    ]
    with pytest.raises(controller.ContractError, match="must be the frozen string map"):
        controller.sdk_request(FakeModels, broken)


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
