#!/usr/bin/env python3
"""Network-free structural tests for the formal N=2/3/4 launcher."""

from __future__ import annotations

import ast
import base64
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import stat
import time
import zlib
from contextlib import ExitStack
from types import SimpleNamespace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent
CONTROLLER = ROOT / "controller.py"
RUNTIME = ROOT / "runtime.sh"
WRAPPER = ROOT / "submit_from_ssh970.sh"
README = ROOT / "README.md"


def load_controller():
    spec = importlib.util.spec_from_file_location("formal_controller", CONTROLLER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_test_request(module, member: str) -> dict:
    return module.build_request(
        member,
        source_root=(
            module.SOURCE_PREFIX
            / "fastwam-action-n234-formal-r4-20260812-r1"
        ),
        source_commit="a" * 40,
        dataset_root=module.OSS_ROOT / "dataset",
        stats_source=module.OSS_ROOT / "stats.json",
        initial_checkpoint=module.OSS_ROOT / "initial.pt",
        vae_source=module.VAE_SOURCE,
        gaussian_cache=module.GAUSSIAN_PREFIX / "primary",
        gaussian_fallback_cache=module.GAUSSIAN_PREFIX / "fallback",
        text_caches=module.DEFAULT_TEXT_CACHES,
        trusted_runtime=b"runtime-bytes",
    )


def observed_getjob_shape(request: dict) -> dict:
    """Exact identity-relevant response projection observed from PAI GetJob."""

    observed = copy.deepcopy(request)
    observed["JobId"] = "dlc-observed-fixture"
    observed["Status"] = "Running"
    observed.pop("JobMaxRunningTimeMinutes")
    observed.pop("SuccessPolicy")
    observed["CustomEnvs"] = [
        {"Key": key, "Value": value, "Visible": "public"}
        for key, value in reversed(list(request["Envs"].items()))
    ]
    observed["DataSources"] = [
        {
            "DataSourceId": source["DataSourceId"],
            "MountPath": source["MountPath"],
            "Uri": "",
        }
        for source in request["DataSources"]
    ]
    observed["Settings"]["ServerManagedSetting"] = True
    observed["JobSpecs"][0].update(
        {"AssignNodeSpec": {}, "EcsSpec": {}, "ImageConfig": {}}
    )
    observed["JobSpecs"][0]["ResourceConfig"]["GPUType"] = ""
    return observed


def assert_request(module, member: str) -> None:
    spec = module.MEMBERS[member]
    request = build_test_request(module, member)
    module.validate_request(member, request)
    assert request["JobMaxRunningTimeMinutes"] == 2160
    assert request["JobSpecs"] == [
        {
            "ElasticSpotSpecs": [],
            "Image": module.IMAGE,
            "LocalMountSpecs": [],
            "PodCount": 1,
            "ResourceConfig": {
                "CPU": "126",
                "GPU": "8",
                "Memory": "960Gi",
                "SharedMemory": "960Gi",
            },
            "RestartPolicy": "Never",
            "StartupDependencies": [],
            "Type": "Worker",
        }
    ]
    env = request["Envs"]
    assert env["FASTWAM_EXTERNAL_CONTRACT"] == "action_only_native_agents_1x8_v1"
    assert env["FASTWAM_AGENT_COUNT"] == str(spec["agent_count"])
    assert json.loads(env["FASTWAM_TASKS_JSON"]) == spec["tasks"]
    assert json.loads(env["FASTWAM_TEXT_CACHE_MAP_JSON"]) == {
        task: module.DEFAULT_TEXT_CACHES[task] for task in spec["tasks"]
    }
    assert env["FASTWAM_MAX_OSS_PUBLISH_BYTES"] == str(62 * 1024**3)
    assert env["FASTWAM_PYTHON"] == str(module.PINNED_PYTHON)
    assert env["FASTWAM_PYTHON_TARGET"] == str(module.PINNED_PYTHON_TARGET)
    assert Path(env["FASTWAM_SOURCE_ROOT"]).parent == module.SOURCE_PREFIX
    assert Path(env["FASTWAM_OSS_OUTPUT_ROOT"]).parent == module.OUTPUT_PREFIX


def suite_record(module) -> dict:
    timestamp = "2026-08-12T00:00:00Z"
    return {
        "schema": "fastwam-action-native-agents-suite-storage-reservation-v1",
        "suite_id": module.SUITE_ID,
        "external_contract": module.CONTRACT,
        "members": list(module.MEMBERS),
        "member_run_ids": {
            name: spec["run_id"] for name, spec in module.MEMBERS.items()
        },
        "per_run_publish_limit_bytes": module.PER_RUN_OSS_BUDGET_BYTES,
        "suite_reserved_bytes": len(module.MEMBERS) * module.PER_RUN_OSS_BUDGET_BYTES,
        "suite_cap_bytes": module.SUITE_OSS_BUDGET_BYTES,
        "platform_quota_snapshot": {
            "quota_bytes": 500 * 1024**3,
            "free_bytes": 242 * 1024**3,
            "evidence": "PAI console quota observation 2026-08-11",
            "observed_at": timestamp,
            "authority": "platform_quota_not_fuse_df",
        },
        "output_roots": {
            name: str(module.output_root(name)) for name in module.MEMBERS
        },
        "prepared_at": timestamp,
        "semantics": "test record",
    }


def test_controller_structure(module) -> None:
    tree = ast.parse(CONTROLLER.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_job_with_options"
    ]
    assert len(calls) == 1, "controller must contain exactly one CreateJob call site"
    text = CONTROLLER.read_text(encoding="utf-8")
    assert text.index("exclusive_write(latch_path(member), latch)") < text.index(
        "client.create_job_with_options("
    )
    assert "autoretry=False" in text and "max_attempts=1" in text
    assert str(module.SOURCE_PREFIX).endswith("fastwam-nohash-source-snapshots")
    assert module.SOURCE_INVENTORY_SCHEMA == "fastwam-formal-source-content-binding-v1"
    assert module.INPUTS_SCHEMA == "fastwam-formal-portable-input-binding-v2"
    assert module.MEMBER_RESERVATION_SCHEMA == (
        "fastwam-action-native-agents-reservation-v3"
    )
    assert module.MEMBER_RESERVATION_KEYS == {
        "schema",
        "suite_id",
        "external_contract",
        "member",
        "experiment_id",
        "run_id",
        "native_agent_count",
        "tasks",
        "masked_agent_set",
        "treatment",
        "schedule",
        "hardware",
        "storage_contract",
        "source",
        "inputs",
        "output_root",
        "request",
        "prepared_at",
        "semantics",
    }
    assert module.MEMBER_RESERVATION_SEMANTICS == (
        "external generic reservation; trainer terminal contract fields remain null; "
        "terminal success is granted only by the runtime receipt"
    )
    assert module.SUITE_ID == "FASTWAM-MR-ACTION-N234-FORMAL-R4-20260812"
    assert str(module.OUTPUT_PREFIX).endswith(
        "/fastwam-action-n234-formal-r4-20260812"
    )
    assert module.CONTROL_ANCHOR == Path("/run")
    assert module.LOCAL_CONTROL_ROOT == Path(
        "/run/fastwam-dlc-submit-state/workspace-270969"
    )
    assert module.CONTROL_LOCK_PATH == (
        module.LOCAL_CONTROL_ROOT / "action-n234-formal-r4-controller.lock"
    )
    for spec in module.MEMBERS.values():
        assert "-R4-20260812" in spec["experiment_id"]
        assert "-r4-20260812" in spec["run_id"]
        assert spec["display_name"].endswith("-r4")
    assert module.SUITE_OSS_BUDGET_BYTES == 190 * 1024**3
    assert module.PER_RUN_OSS_BUDGET_BYTES == 62 * 1024**3
    assert str(module.PINNED_PYTHON).endswith(
        "/venvs/fastwam-gaudp-py310-20260802/bin/python"
    )
    assert str(module.PINNED_PYTHON_TARGET).endswith(
        "/runtimes/uv-python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"
    )
    bootstrap = module.TRUSTED_BOOTSTRAP_COMMAND
    assert f"exec {module.PINNED_PYTHON} -B -I -S" in bootstrap
    assert f"readlink -f -- {module.PINNED_PYTHON}" in bootstrap
    assert str(module.PINNED_PYTHON_TARGET) in bootstrap
    assert "/usr/bin/python3" not in bootstrap
    first = SimpleNamespace(st_mode=0o100600, st_size=17, st_mtime_ns=23,
                            st_dev=1, st_ino=2)
    remounted = SimpleNamespace(st_mode=0o100600, st_size=17, st_mtime_ns=23,
                                st_dev=9001, st_ino=9002)
    assert module.portable_file_stat(first) == module.portable_file_stat(remounted)
    assert module.portable_file_stat(first) == {"bytes": 17}
    prepare_start = text.index("def prepare(args:")
    prepare_end = text.index("\ndef load_sdk", prepare_start)
    prepare_text = text[prepare_start:prepare_end]
    assert prepare_text.index("planned_reservations = {") < prepare_text.index(
        "os.mkdir(OUTPUT_PREFIX, 0o700)"
    )
    assert prepare_text.index("validate_existing_member_state(") < prepare_text.index(
        "os.mkdir(OUTPUT_PREFIX, 0o700)"
    )
    assert prepare_text.index("os.mkdir(OUTPUT_PREFIX, 0o700)") < prepare_text.index(
        "outcomes = ["
    )
    assert prepare_text.index("outcomes = [") < prepare_text.index(
        "exclusive_write(SUITE_STORAGE_RESERVATION_PATH, suite_reservation)"
    )
    assert prepare_text.index(
        "exclusive_write(SUITE_STORAGE_RESERVATION_PATH, suite_reservation)"
    ) < prepare_text.index("write_prepared_local_state(member)")
    live_start = text.index("def validate_reservation_live(")
    live_end = text.index("\ndef reconcile_member", live_start)
    assert "validate_complete_suite_members(suite_reservation)" in text[live_start:live_end]
    terminal_start = text.index("def validate_formal_terminal_output(")
    terminal_end = text.index("\ndef validate_reservation_live", terminal_start)
    terminal_text = text[terminal_start:terminal_end]
    for required in (
        "SCIENTIFIC_COMPLETE",
        "formal COMPLETE marker mismatch",
        "formal output filesystem differs from terminal allowlist",
        "checkpoints/weights/step_000500.pt",
        "checkpoints/weights/step_001000.pt",
        "checkpoints/state/step_001000/trainer_state.json",
        "step1000-fresh-load.json",
        "offline-eval.json",
    ):
        assert required in terminal_text
    reconcile_start = text.index("def reconcile_member(")
    reconcile_end = text.index("\ndef submit(", reconcile_start)
    assert text[reconcile_start:reconcile_end].count(
        "validate_formal_terminal_output(member)"
    ) == 2
    submit_start = reconcile_end + 1
    submit_end = text.index("\ndef reconcile(", submit_start)
    submit_text = text[submit_start:submit_end]
    assert submit_text.index("validate_reservation_live(") < submit_text.index(
        "require_n2_scientific_completion_for_downstream_submit("
    )
    assert submit_text.index(
        "require_n2_scientific_completion_for_downstream_submit("
    ) < submit_text.index("load_sdk()")
    assert submit_text.index("load_sdk()") < submit_text.index(
        "exclusive_write(latch_path(member), latch)"
    )
    prerequisite_start = text.index(
        "def require_n2_scientific_completion_for_downstream_submit("
    )
    prerequisite_end = text.index("\ndef reconcile_member(", prerequisite_start)
    prerequisite_text = text[prerequisite_start:prerequisite_end]
    assert 'if member == "n2":\n        return None' in prerequisite_text
    assert 'validate_formal_terminal_output("n2")' in prerequisite_text
    assert '"SCIENTIFIC_COMPLETE"' in prerequisite_text
    for forbidden in ("hashlib", "sha256sum", "md5sum", "blake2", "checksum"):
        assert forbidden not in text.lower()
    for retired in ("R1-20260811", "R2-20260811", "-r1-20260811", "-r2-20260811"):
        assert retired not in text

    record = suite_record(module)
    module.validate_suite_storage_reservation(record)
    bad = copy.deepcopy(record)
    bad["platform_quota_snapshot"]["evidence"] = "ossfs df output"
    try:
        module.validate_suite_storage_reservation(bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("FUSE df must never satisfy the platform-quota gate")
    bad = copy.deepcopy(record)
    bad["platform_quota_snapshot"]["observed_at"] = "not-a-timestamp"
    try:
        module.validate_suite_storage_reservation(bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("quota observation requires an explicit UTC timestamp")
    bad = copy.deepcopy(record)
    bad["platform_quota_snapshot"]["observed_at"] = "2026-08-10T00:00:00Z"
    try:
        module.validate_suite_storage_reservation(bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("stale platform quota evidence must not authorize prepare")

    request = build_test_request(module, "n2")
    observed = observed_getjob_shape(request)
    assert module.exact_job(observed, request)

    direct_mutations = (
        ("UserCommand", "changed"),
        ("Accessibility", "PUBLIC"),
        ("JobType", "changed"),
        ("Priority", 99),
    )
    for field, value in direct_mutations:
        changed = copy.deepcopy(observed)
        changed[field] = value
        assert not module.exact_job(changed, request)

    changed = copy.deepcopy(observed)
    changed["Envs"]["FASTWAM_INITIAL_CHECKPOINT"] = "changed"
    assert not module.exact_job(changed, request)
    changed = copy.deepcopy(observed)
    changed["Settings"]["Tags"]["run_id"] = "changed"
    assert not module.exact_job(changed, request)
    changed = copy.deepcopy(observed)
    changed["JobSpecs"][0]["Image"] = "changed"
    assert not module.exact_job(changed, request)
    changed = copy.deepcopy(observed)
    changed["JobSpecs"][0]["ResourceConfig"]["GPU"] = "7"
    assert not module.exact_job(changed, request)
    changed = copy.deepcopy(observed)
    changed["WorkspaceId"] = int(request["WorkspaceId"])
    assert not module.exact_job(changed, request)

    for field, value in (
        ("DataSourceId", "changed"),
        ("MountPath", "/changed"),
        ("Uri", "oss://unexpected"),
    ):
        changed = copy.deepcopy(observed)
        changed["DataSources"][0][field] = value
        assert not module.exact_job(changed, request)
    changed = copy.deepcopy(observed)
    changed["DataSources"].reverse()
    assert not module.exact_job(changed, request)


    custom_mutations = []
    changed = copy.deepcopy(observed)
    changed["CustomEnvs"].pop()
    custom_mutations.append(changed)
    changed = copy.deepcopy(observed)
    changed["CustomEnvs"][-1] = copy.deepcopy(changed["CustomEnvs"][0])
    custom_mutations.append(changed)
    changed = copy.deepcopy(observed)
    changed["CustomEnvs"][0]["Value"] = "changed"
    custom_mutations.append(changed)
    changed = copy.deepcopy(observed)
    changed["CustomEnvs"][0]["Visible"] = "private"
    custom_mutations.append(changed)
    changed = copy.deepcopy(observed)
    changed["CustomEnvs"][0]["Unexpected"] = True
    custom_mutations.append(changed)
    assert all(not module.exact_job(changed, request) for changed in custom_mutations)

    for field in ("JobMaxRunningTimeMinutes", "SuccessPolicy"):
        returned = copy.deepcopy(observed)
        returned[field] = request[field]
        assert module.exact_job(returned, request)
        returned[field] = "changed"
        assert not module.exact_job(returned, request)

        for mutation in ("removed", "changed"):
            invalid_request = copy.deepcopy(request)
            if mutation == "removed":
                invalid_request.pop(field)
            else:
                invalid_request[field] = "changed"
            try:
                module.validate_request("n2", invalid_request)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"request {field} {mutation} must fail closed")

    invalid_request = copy.deepcopy(request)
    invalid_request["CustomEnvs"] = [{"Key": "unexpected"}]
    try:
        module.validate_request("n2", invalid_request)
    except RuntimeError:
        pass
    else:
        raise AssertionError("request CustomEnvs must remain exactly empty")

    future_request = copy.deepcopy(request)
    future_request["FutureFrozenField"] = {"Required": True}
    assert not module.exact_job(observed, future_request)
    future_observed = copy.deepcopy(observed)
    future_observed["FutureFrozenField"] = {"Required": True, "ServerAdded": 1}
    assert module.exact_job(future_observed, future_request)
    future_observed["FutureFrozenField"]["Required"] = False
    assert not module.exact_job(future_observed, future_request)
    strict_scalar_request = copy.deepcopy(request)
    strict_scalar_request["FutureFrozenField"] = {"Required": 1}
    strict_scalar_observed = copy.deepcopy(observed)
    strict_scalar_observed["FutureFrozenField"] = {"Required": True}
    assert not module.exact_job(strict_scalar_observed, strict_scalar_request)


def test_r4_controller_lock_binds_fd_to_exact_path(module) -> None:
    with tempfile.TemporaryDirectory(prefix="formal-r4-controller-lock-") as name:
        root = Path(name)
        r4_lock = root / "action-n234-formal-r4-controller.lock"
        old_lock = root / "retired-controller.lock"
        r4_lock.write_bytes(b"")
        old_lock.write_bytes(b"")
        r4_metadata = os.lstat(r4_lock)
        old_metadata = os.lstat(old_lock)
        environment = {
            "FASTWAM_CONTROL_NODE": module.CONTROL_NODE,
            "FASTWAM_LOCK_FD": "9",
        }

        with (
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch.object(module, "CONTROL_LOCK_PATH", r4_lock),
            mock.patch.object(module.os, "fstat", return_value=r4_metadata),
            mock.patch.object(module.fcntl, "flock") as lock,
        ):
            module.require_controller_lock()
        lock.assert_called_once_with(
            9, module.fcntl.LOCK_EX | module.fcntl.LOCK_NB
        )

        with (
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch.object(module, "CONTROL_LOCK_PATH", r4_lock),
            mock.patch.object(module.os, "fstat", return_value=old_metadata),
            mock.patch.object(module.fcntl, "flock") as lock,
        ):
            _assert_runtime_rejected(
                module.require_controller_lock,
                "descriptor inherited from retired lock path",
            )
        lock.assert_not_called()

        bound_metadata = os.lstat(r4_lock)
        replacement = root / "replacement.lock"
        replacement.write_bytes(b"")
        os.replace(replacement, r4_lock)
        assert (bound_metadata.st_dev, bound_metadata.st_ino) != (
            os.lstat(r4_lock).st_dev,
            os.lstat(r4_lock).st_ino,
        )
        with (
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch.object(module, "CONTROL_LOCK_PATH", r4_lock),
            mock.patch.object(module.os, "fstat", return_value=bound_metadata),
            mock.patch.object(module.fcntl, "flock") as lock,
        ):
            _assert_runtime_rejected(
                module.require_controller_lock,
                "R4 lock path replaced after descriptor open",
            )
        lock.assert_not_called()

        target = root / "symlink-target.lock"
        target.write_bytes(b"")
        symlink = root / "symlink.lock"
        symlink.symlink_to(target)
        with (
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch.object(module, "CONTROL_LOCK_PATH", symlink),
            mock.patch.object(module.os, "fstat", return_value=os.stat(target)),
            mock.patch.object(module.fcntl, "flock") as lock,
        ):
            _assert_runtime_rejected(
                module.require_controller_lock,
                "symlink controller lock path",
            )
        lock.assert_not_called()


def test_controller_exclusive_writer_fails_closed(module) -> None:
    with tempfile.TemporaryDirectory(prefix="formal-r4-exclusive-writer-") as name:
        root = Path(name)

        collision = root / "collision.json"
        collision.write_bytes(b"pre-existing immutable record\n")
        try:
            module.exclusive_write(collision, {"new": "record"})
        except FileExistsError:
            pass
        else:
            raise AssertionError("O_EXCL collision must reject an existing record")
        assert collision.read_bytes() == b"pre-existing immutable record\n"

        zero_write = root / "zero-write.json"
        with mock.patch.object(module.os, "write", return_value=0):
            try:
                module.exclusive_write(zero_write, {"record": "cannot-complete"})
            except OSError as error:
                assert "zero-byte write" in str(error)
            else:
                raise AssertionError("zero-byte durable write must fail closed")
        assert zero_write.is_file()
        assert zero_write.read_bytes() == b""
        try:
            module.exclusive_write(zero_write, {"record": "retry-is-forbidden"})
        except FileExistsError:
            pass
        else:
            raise AssertionError("partial immutable record must poison the R4 identity")

        local_state = root / "local-state.json"
        local_state.write_bytes(b"old-local-state\n")
        with (
            mock.patch.object(module.os, "write", return_value=0),
            mock.patch.object(module.os, "replace") as replace,
        ):
            try:
                module.atomic_write(local_state, {"phase": "new-state"})
            except OSError as error:
                assert "zero-byte write" in str(error)
            else:
                raise AssertionError("zero-byte local-state write must fail closed")
        replace.assert_not_called()
        assert local_state.read_bytes() == b"old-local-state\n"

        unenforced = root / "unenforced-exclusive.json"
        real_open = module.os.open
        exclusive_opens = 0
        duplicate_descriptor = None

        def simulate_unenforced_exclusive(path, flags, mode=0o777, *, dir_fd=None):
            nonlocal exclusive_opens, duplicate_descriptor
            if Path(path) == unenforced and flags & os.O_EXCL:
                exclusive_opens += 1
                if exclusive_opens == 2:
                    duplicate_descriptor = real_open(os.devnull, os.O_WRONLY)
                    return duplicate_descriptor
            return real_open(path, flags, mode, dir_fd=dir_fd)

        with mock.patch.object(
            module.os, "open", side_effect=simulate_unenforced_exclusive
        ):
            _assert_runtime_rejected(
                lambda: module.exclusive_write(unenforced, {"record": "written"}),
                "durable storage that does not enforce O_EXCL",
            )
        assert exclusive_opens == 2
        assert duplicate_descriptor is not None
        try:
            os.fstat(duplicate_descriptor)
        except OSError:
            pass
        else:
            raise AssertionError("unexpected duplicate descriptor was not closed")


def _submit_with_n2_evidence(
    module,
    member: str,
    formal_result,
    *,
    fixture_mutation: str | None = None,
):
    requests = {name: build_test_request(module, name) for name in module.MEMBERS}
    shared_source = {
        "root": str(module.EXPECTED_SOURCE_ROOT),
        "git_commit": "a" * 40,
        "inventory": {
            "schema": module.SOURCE_INVENTORY_SCHEMA,
            "entries": [{"path": ".", "kind": "directory"}],
        },
    }
    reservations = {
        name: {
            "member": name,
            "source": copy.deepcopy(shared_source),
            "request": requests[name],
        }
        for name in module.MEMBERS
    }
    selected_reservation = reservations[member]
    alternate_source = str(
        module.SOURCE_PREFIX
        / f"fastwam-action-n234-formal-r4-20260812-{member}-alternate"
    )
    if fixture_mutation == "suite_path":
        selected_reservation["request"]["Envs"][
            "FASTWAM_SUITE_STORAGE_RESERVATION_PATH"
        ] = str(module.OSS_ROOT / "different-suite/reservation.json")
    elif fixture_mutation == "source_root":
        selected_reservation["source"]["root"] = alternate_source
        selected_reservation["request"]["Envs"][
            "FASTWAM_SOURCE_ROOT"
        ] = alternate_source
    elif fixture_mutation == "source_commit":
        selected_reservation["source"]["git_commit"] = "b" * 40
        selected_reservation["request"]["Envs"]["FASTWAM_CODE_COMMIT"] = "b" * 40
    elif fixture_mutation == "shared_request_env":
        selected_reservation["request"]["Envs"][
            "FASTWAM_DATASET_ROOT"
        ] = str(module.OSS_ROOT / "different-valid-dataset")
    elif fixture_mutation not in {None, "selected_vs_suite", "n2_reread_vs_suite"}:
        raise AssertionError(f"unknown scientific-gate fixture mutation: {fixture_mutation}")

    # Commit and shared-environment changes remain individually request-valid;
    # source-root drift now also violates the R4 exact frozen-source contract.
    if fixture_mutation in {"source_commit", "shared_request_env"}:
        module.validate_request(member, selected_reservation["request"])

    selected_readback = copy.deepcopy(selected_reservation)
    n2_readback = copy.deepcopy(reservations["n2"])
    if fixture_mutation == "selected_vs_suite":
        selected_readback["source"]["root"] = alternate_source
        selected_readback["request"]["Envs"]["FASTWAM_SOURCE_ROOT"] = alternate_source
    elif fixture_mutation == "n2_reread_vs_suite":
        alternate_n2_source = str(
            module.SOURCE_PREFIX
            / "fastwam-action-n234-formal-r4-20260812-n2-reread"
        )
        n2_readback["source"]["root"] = alternate_n2_source
        n2_readback["request"]["Envs"]["FASTWAM_SOURCE_ROOT"] = alternate_n2_source
    events: list[str] = []

    class Client:
        def create_job_with_options(self, request, headers, runtime_options):
            events.append("CreateJob")
            assert request == "validated-sdk-request"
            assert headers == {}
            assert runtime_options == {"autoretry": False}
            return SimpleNamespace(
                body=SimpleNamespace(to_map=lambda: {"JobId": "dlc-r4-test"})
            )

    def read_records(_path):
        index = read_records.calls
        read_records.calls += 1
        if index == 0:
            return copy.deepcopy(selected_readback), {}
        if member == "n2":
            raise AssertionError("N2 submit read an unexpected prerequisite record")
        if index == 1:
            return {"suite": "fixture"}, {}
        if index == 2:
            return copy.deepcopy(n2_readback), {}
        raise AssertionError("submit read an unexpected extra durable record")

    read_records.calls = 0

    def validate_live(selected, reservation, *, require_output_absent=True):
        events.append(f"live:{selected}")
        if selected == member:
            assert require_output_absent
        if selected == "n2" and member != "n2":
            assert not require_output_absent
        assert reservation["member"] == selected
        return copy.deepcopy(reservation["request"])

    def validate_structure(selected, reservation):
        assert reservation["member"] == selected
        return copy.deepcopy(reservation["request"])

    def terminal(selected):
        assert selected == "n2"
        events.append("n2-terminal")
        if isinstance(formal_result, BaseException):
            raise formal_result
        return copy.deepcopy(formal_result)

    def load_sdk():
        events.append("load-sdk")
        return Client(), SimpleNamespace(), SimpleNamespace()

    def list_jobs(*_args):
        events.append("list-jobs")
        return []

    def publish_latch(_path, _value):
        events.append("latch")

    def publish_local_state(_path, _value):
        events.append("local-state")

    with tempfile.TemporaryDirectory(prefix=f"formal-r4-submit-gate-{member}-") as name:
        root = Path(name)
        with ExitStack() as stack:
            read_json = stack.enter_context(
                mock.patch.object(module, "read_json", side_effect=read_records)
            )
            suite_validator = stack.enter_context(
                mock.patch.object(
                    module,
                    "validate_complete_suite_members",
                    return_value=reservations,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    module, "validate_reservation_live", side_effect=validate_live
                )
            )
            stack.enter_context(
                mock.patch.object(
                    module,
                    "validate_member_reservation_structure",
                    side_effect=validate_structure,
                )
            )
            terminal_validator = stack.enter_context(
                mock.patch.object(
                    module, "validate_formal_terminal_output", side_effect=terminal
                )
            )
            sdk_loader = stack.enter_context(
                mock.patch.object(module, "load_sdk", side_effect=load_sdk)
            )
            stack.enter_context(
                mock.patch.object(
                    module,
                    "validate_request",
                    return_value="validated-sdk-request",
                )
            )
            jobs = stack.enter_context(
                mock.patch.object(module, "list_jobs", side_effect=list_jobs)
            )
            latch = stack.enter_context(
                mock.patch.object(module, "exclusive_write", side_effect=publish_latch)
            )
            local_state = stack.enter_context(
                mock.patch.object(
                    module, "atomic_write", side_effect=publish_local_state
                )
            )
            stack.enter_context(
                mock.patch.object(
                    module,
                    "runtime_options",
                    return_value={"autoretry": False},
                )
            )
            stack.enter_context(
                mock.patch.object(
                    module,
                    "get_job",
                    return_value={"JobId": "dlc-r4-test", "Status": "Running"},
                )
            )
            stack.enter_context(mock.patch.object(module, "exact_job", return_value=True))
            stack.enter_context(
                mock.patch.object(
                    module,
                    "publish_acknowledgement",
                    return_value={"job_id": "dlc-r4-test", "job_status": "Running"},
                )
            )
            stack.enter_context(
                mock.patch.object(
                    module,
                    "acknowledgement_path",
                    return_value=root / "acknowledgement.json",
                )
            )
            stack.enter_context(
                mock.patch.object(
                    module, "latch_path", return_value=root / "latch.json"
                )
            )
            stack.enter_context(
                mock.patch.object(
                    module, "local_state_path", return_value=root / "state.json"
                )
            )
            stack.enter_context(mock.patch("builtins.print"))
            error = None
            try:
                module.submit(
                    SimpleNamespace(
                        member=member,
                        confirm_experiment_id=module.MEMBERS[member]["experiment_id"],
                    )
                )
            except (FileNotFoundError, RuntimeError) as caught:
                error = caught

    return SimpleNamespace(
        error=error,
        events=events,
        read_json=read_json,
        suite_validator=suite_validator,
        terminal_validator=terminal_validator,
        sdk_loader=sdk_loader,
        jobs=jobs,
        latch=latch,
        local_state=local_state,
    )


def test_downstream_submit_requires_n2_scientific_completion(module) -> None:
    invalid_evidence = (
        ("absent", FileNotFoundError("N2 COMPLETE is absent")),
        ("malformed", ["not", "a", "structured", "completion"]),
        ("incomplete", {"status": "NOT_COMPLETE"}),
    )
    for member in ("n3", "n4"):
        for label, evidence in invalid_evidence:
            result = _submit_with_n2_evidence(module, member, evidence)
            assert result.error is not None, f"{member} accepted {label} N2 evidence"
            assert result.events[:2] == [f"live:{member}", "live:n2"]
            assert result.events[2:] == ["n2-terminal"]
            result.terminal_validator.assert_called_once_with("n2")
            result.sdk_loader.assert_not_called()
            result.jobs.assert_not_called()
            result.latch.assert_not_called()
            result.local_state.assert_not_called()

        scientific = {
            "status": "SCIENTIFIC_COMPLETE",
            "output_root": str(module.output_root("n2")),
            "published_bytes": 123,
            "artifact_files": 9,
        }
        accepted = _submit_with_n2_evidence(module, member, scientific)
        assert accepted.error is None
        assert accepted.events.index("n2-terminal") < accepted.events.index("load-sdk")
        assert accepted.events.index("load-sdk") < accepted.events.index("latch")
        assert accepted.events.index("latch") < accepted.events.index("CreateJob")
        accepted.terminal_validator.assert_called_once_with("n2")
        accepted.latch.assert_called_once()
        assert accepted.local_state.call_count == 2


def test_downstream_gate_rejects_cross_record_drift_before_submission(module) -> None:
    scientific = {
        "status": "SCIENTIFIC_COMPLETE",
        "output_root": str(module.output_root("n2")),
        "published_bytes": 123,
        "artifact_files": 9,
    }
    mismatches = (
        (
            "suite_path",
            "N2 and downstream requests do not share one frozen basis",
        ),
        (
            "source_root",
            "N2 and downstream members do not bind the same source",
        ),
        (
            "source_commit",
            "N2 and downstream members do not bind the same source",
        ),
        (
            "shared_request_env",
            "N2 and downstream requests do not share one frozen basis",
        ),
        (
            "selected_vs_suite",
            "downstream reservation differs from this immutable suite member",
        ),
        (
            "n2_reread_vs_suite",
            "N2 reservation changed during prerequisite validation",
        ),
    )
    for member in ("n3", "n4"):
        for mutation, expected_error in mismatches:
            result = _submit_with_n2_evidence(
                module,
                member,
                scientific,
                fixture_mutation=mutation,
            )
            assert isinstance(result.error, RuntimeError), (
                f"{member} accepted scientific-gate drift: {mutation}"
            )
            assert expected_error in str(result.error)
            assert "load-sdk" not in result.events
            assert "list-jobs" not in result.events
            assert "latch" not in result.events
            assert "local-state" not in result.events
            assert "CreateJob" not in result.events
            result.terminal_validator.assert_not_called()
            result.sdk_loader.assert_not_called()
            result.jobs.assert_not_called()
            result.latch.assert_not_called()
            result.local_state.assert_not_called()


def test_n2_submit_has_no_scientific_prerequisite(module) -> None:
    result = _submit_with_n2_evidence(
        module,
        "n2",
        AssertionError("N2 must not inspect predecessor terminal evidence"),
    )
    assert result.error is None
    assert result.events[0] == "live:n2"
    assert "n2-terminal" not in result.events
    assert result.events.index("load-sdk") < result.events.index("latch")
    assert result.events.index("latch") < result.events.index("CreateJob")
    result.suite_validator.assert_not_called()
    result.terminal_validator.assert_not_called()
    result.latch.assert_called_once()


def _write_portable_fixture(root: Path, payload: bytes = b"portable-source") -> None:
    nested = root / "nested"
    nested.mkdir()
    (nested / "payload.bin").write_bytes(payload)
    (root / "empty").mkdir()


def test_source_inventory_cross_mount_portability(module) -> None:
    shared_memory = Path("/dev/shm")
    assert shared_memory.is_dir(), "/dev/shm is required for the cross-mount test"
    with tempfile.TemporaryDirectory(prefix="formal-r4-posix-", dir="/tmp") as left_name:
        with tempfile.TemporaryDirectory(
            prefix="formal-r4-shm-", dir=shared_memory
        ) as right_name:
            left = Path(left_name)
            right = Path(right_name)
            _write_portable_fixture(left)
            _write_portable_fixture(right)
            os.chmod(left / "nested/payload.bin", 0o600)
            os.chmod(right / "nested/payload.bin", 0o644)
            os.utime(left / "nested/payload.bin", ns=(1_600_000_000_000_000_000,) * 2)
            os.utime(right / "nested/payload.bin", ns=(1_700_000_000_000_000_000,) * 2)
            left_inventory = module.source_inventory(left)
            right_inventory = module.source_inventory(right)
            assert left.stat().st_dev != right.stat().st_dev
            assert left_inventory == right_inventory
            module.assert_source_inventory_matches(
                left_inventory, right_inventory, label="cross-mount mismatch"
            )
            serialized = json.dumps(left_inventory, sort_keys=True)
            for forbidden in ("mode", "mtime", "device", "inode", "ctime"):
                assert forbidden not in serialized


def test_source_inventory_ignores_mode_and_mtime(module) -> None:
    with tempfile.TemporaryDirectory(prefix="formal-r4-metadata-") as temporary:
        root = Path(temporary)
        _write_portable_fixture(root)
        first = module.source_inventory(root)
        replacement = root / "replacement.bin"
        replacement.write_bytes(b"portable-source")
        os.chmod(replacement, 0o700)
        os.utime(replacement, ns=(1_500_000_000_000_000_000,) * 2)
        os.replace(replacement, root / "nested/payload.bin")
        os.chmod(root / "nested", 0o755)
        os.utime(root / "nested", ns=(1_550_000_000_000_000_000,) * 2)
        second = module.source_inventory(root)
        assert first == second


def test_source_inventory_content_difference_is_path_only(module) -> None:
    with tempfile.TemporaryDirectory(prefix="formal-r4-content-") as temporary:
        root = Path(temporary)
        _write_portable_fixture(root, b"first-content")
        expected = module.source_inventory(root)
        (root / "nested/payload.bin").write_bytes(b"other-content")
        observed = module.source_inventory(root)
        difference = module.source_inventory_difference(expected, observed)
        assert difference == {
            "missing": [],
            "extra": [],
            "changed": ["nested/payload.bin"],
        }
        try:
            module.assert_source_inventory_matches(
                expected, observed, label="portable source mismatch"
            )
        except RuntimeError as error:
            message = str(error)
            assert "nested/payload.bin" in message
            assert base64.b64encode(b"first-content").decode("ascii") not in message
            assert base64.b64encode(b"other-content").decode("ascii") not in message
        else:
            raise AssertionError("content replacement must fail closed")


def test_source_inventory_schema_rejects_float_bool_and_noncanonical(module) -> None:
    valid = {
        "schema": module.SOURCE_INVENTORY_SCHEMA,
        "entries": [
            {"path": ".", "kind": "directory"},
            {
                "path": "payload.bin",
                "kind": "file",
                "size": 3,
                "content_b64": base64.b64encode(b"abc").decode("ascii"),
            },
        ],
    }
    module.validate_source_inventory(valid)
    mutations = []
    for invalid_size in (True, 3.0):
        mutation = copy.deepcopy(valid)
        mutation["entries"][1]["size"] = invalid_size
        mutations.append(mutation)
    mutation = copy.deepcopy(valid)
    mutation["entries"][1]["content_b64"] = "not canonical base64"
    mutations.append(mutation)
    mutation = copy.deepcopy(valid)
    mutation["entries"][1]["path"] = "../payload.bin"
    mutations.append(mutation)
    mutation = copy.deepcopy(valid)
    mutation["entries"][1]["mode"] = 0o644
    mutations.append(mutation)
    mutation = copy.deepcopy(valid)
    mutation["entries"].append(copy.deepcopy(mutation["entries"][1]))
    mutations.append(mutation)
    mutation = copy.deepcopy(valid)
    mutation["entries"] = list(reversed(mutation["entries"]))
    mutations.append(mutation)
    mutation = copy.deepcopy(valid)
    mutation["entries"][1]["path"] = "missing/payload.bin"
    mutations.append(mutation)
    for index, mutation in enumerate(mutations):
        try:
            module.validate_source_inventory(mutation)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"malformed source schema accepted at case {index}")


def test_source_inventory_rejects_symlink_and_path_race(module) -> None:
    with tempfile.TemporaryDirectory(prefix="formal-r4-link-") as temporary:
        root = Path(temporary)
        (root / "payload.bin").write_bytes(b"payload")
        (root / "payload-link").symlink_to("payload.bin")
        try:
            module.source_inventory(root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("source symlink must fail closed")

    with tempfile.TemporaryDirectory(prefix="formal-r4-race-") as temporary:
        root = Path(temporary)
        child = root / "nested"
        moved = root / "nested-moved"
        child.mkdir()
        (child / "payload.bin").write_bytes(b"payload")
        real_open = os.open
        replacement_triggered = False

        def replacing_open(path, flags, mode=0o777, *, dir_fd=None):
            nonlocal replacement_triggered
            if dir_fd is None:
                descriptor = real_open(path, flags, mode)
            else:
                descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            if path == "nested" and dir_fd is not None and not replacement_triggered:
                replacement_triggered = True
                child.rename(moved)
                child.symlink_to(moved.name, target_is_directory=True)
            return descriptor

        with mock.patch.object(module.os, "open", side_effect=replacing_open):
            try:
                module.source_inventory(root)
            except RuntimeError:
                pass
            else:
                raise AssertionError("source directory replacement must fail closed")
        assert replacement_triggered


def _assert_runtime_rejected(callback, label: str) -> None:
    try:
        callback()
    except RuntimeError:
        return
    raise AssertionError(f"malformed portable binding was accepted: {label}")


def test_request_schema_scalar_types_and_trusted_runtime_base64(module) -> None:
    """The immutable CreateJob request must reject extension and type ambiguity."""

    member = "n2"
    request = build_test_request(module, member)
    module.validate_request(member, request)
    mutations: list[tuple[str, dict]] = []

    for key in (
        "Description",
        "Envs",
        "DataSources",
        "JobSpecs",
        "Settings",
    ):
        changed = copy.deepcopy(request)
        changed.pop(key)
        mutations.append((f"missing request key {key}", changed))
    changed = copy.deepcopy(request)
    changed["Unexpected"] = True
    mutations.append(("extra request key", changed))

    nested_key_cases = (
        ("job spec", ("JobSpecs", 0), "StartupDependencies"),
        ("resource config", ("JobSpecs", 0, "ResourceConfig"), "CPU"),
        ("data source", ("DataSources", 0), "MountAccess"),
        ("settings", ("Settings",), "EnableOssAppend"),
        ("tags", ("Settings", "Tags"), "purpose"),
        ("environment", ("Envs",), "FASTWAM_VAE_SOURCE"),
    )
    for label, location, key in nested_key_cases:
        changed = copy.deepcopy(request)
        branch = changed
        for part in location:
            branch = branch[part]
        branch.pop(key)
        mutations.append((f"missing {label} key {key}", changed))
        changed = copy.deepcopy(request)
        branch = changed
        for part in location:
            branch = branch[part]
        branch["Unexpected"] = True
        mutations.append((f"extra {label} key", changed))

    scalar_cases = (
        ("bool Priority", ("Priority",), True),
        ("float max runtime", ("JobMaxRunningTimeMinutes",), 2160.0),
        ("bool PodCount", ("JobSpecs", 0, "PodCount"), True),
        (
            "integer settings boolean",
            ("Settings", "AllocateAllRDMADevices"),
            1,
        ),
    )
    for label, location, value in scalar_cases:
        changed = copy.deepcopy(request)
        branch = changed
        for part in location[:-1]:
            branch = branch[part]
        branch[location[-1]] = value
        mutations.append((label, changed))

    changed = copy.deepcopy(request)
    encoded = changed["Envs"][module.TRUSTED_RUNTIME_B64_ENV]
    changed["Envs"][module.TRUSTED_RUNTIME_B64_ENV] = (
        _noncanonical_base64_for_same_payload(encoded)
    )
    assert base64.b64decode(
        changed["Envs"][module.TRUSTED_RUNTIME_B64_ENV], validate=True
    ) == base64.b64decode(encoded, validate=True)
    mutations.append(("non-canonical trusted-runtime Base64", changed))

    for label, candidate in mutations:
        _assert_runtime_rejected(
            lambda candidate=candidate: module.validate_request(member, candidate),
            label,
        )


def _json_payload(value: dict) -> bytes:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    while len(payload) % 3 == 0:
        payload += b" "
    return payload


def _content_descriptor(path: str, payload: bytes) -> dict:
    return {
        "path": path,
        "kind": "file",
        "bytes": len(payload),
        "content_b64": base64.b64encode(payload).decode("ascii"),
    }


def _compressed_content_descriptor(path: str, payload: bytes) -> dict:
    compressed = zlib.compress(payload, level=6)
    return {
        "path": path,
        "kind": "file",
        "bytes": len(payload),
        "content_encoding": "zlib-level6-base64-v1",
        "compressed_bytes": len(compressed),
        "compressed_b64": base64.b64encode(compressed).decode("ascii"),
    }


def _decode_compressed_descriptor(module, descriptor: dict, *, label: str) -> bytes:
    return module.validate_compressed_content_file_descriptor(
        descriptor,
        expected_path=descriptor["path"],
        label=label,
    )


def _gaussian_manifest(
    cache_kind: str, *, generation: str = "aa", padding_bytes: int = 0
) -> bytes:
    if cache_kind == "compact":
        height, width, selection_mode = 28, 40, "index"
    else:
        height, width, selection_mode = 56, 80, "all"
    value = {
        "generation": generation,
        "manifest_version": 1,
        "shards": [{}],
        "total_frames": 2,
        "schema": {
            "cache_kind": cache_kind,
            "channel_count": 13,
            "height": height,
            "width": width,
        },
        "selection": {"mode": selection_mode},
    }
    if padding_bytes:
        value["padding"] = "x" * padding_bytes
    return _json_payload(value)


def _gaussian_complete(
    manifest_payload: bytes, *, legacy_manifest_field: str = "a" * 64
) -> bytes:
    manifest = json.loads(manifest_payload)
    return _json_payload(
        {
            "complete": True,
            "schema_name": "fastwam.canonical-gaussian-cache",
            "schema_version": 1,
            "manifest_version": manifest["manifest_version"],
            "manifest": "manifest.json",
            "manifest_bytes": len(manifest_payload),
            "manifest_sha256": legacy_manifest_field,
            "shard_count": len(manifest["shards"]),
            "total_frames": manifest["total_frames"],
        }
    )


def _gaussian_binding(root: str, cache_kind: str) -> dict:
    manifest = _gaussian_manifest(cache_kind)
    marker = _gaussian_complete(manifest)
    dimensions = [13, 28, 40] if cache_kind == "compact" else [13, 56, 80]
    return {
        "path": root,
        "kind": "directory",
        "manifest": _compressed_content_descriptor(
            f"{root}/manifest.json", manifest
        ),
        "completion_marker": _content_descriptor(f"{root}/COMPLETE", marker),
        "cache_kind": cache_kind,
        "dimensions": dimensions,
        "selection_mode": "index" if cache_kind == "compact" else "all",
    }


def _valid_inputs_binding(module, member: str = "n3") -> tuple[dict, dict]:
    request = build_test_request(module, member)
    envs = request["Envs"]
    dataset = envs["FASTWAM_DATASET_ROOT"]
    stats_payload = _json_payload({"source_root": dataset, "generation": "aa"})
    text_map = json.loads(envs["FASTWAM_TEXT_CACHE_MAP_JSON"])
    binding = {
        "schema": module.INPUTS_SCHEMA,
        "dataset": {"path": dataset, "kind": "directory"},
        "normalization_stats": {
            **_content_descriptor(envs["FASTWAM_STATS_SOURCE"], stats_payload),
            "declared_source_root": dataset,
            "resolved_source_root": dataset,
        },
        "initial_checkpoint": {
            "path": envs["FASTWAM_INITIAL_CHECKPOINT"],
            "kind": "file",
            "bytes": 101,
        },
        "vae": {
            "path": envs["FASTWAM_VAE_SOURCE"],
            "kind": "file",
            "bytes": 202,
        },
        "gaussian_primary": _gaussian_binding(
            envs["FASTWAM_GAUSSIAN_CACHE_DIR"], "compact"
        ),
        "gaussian_fallback": _gaussian_binding(
            envs["FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR"], "canonical"
        ),
        "text_caches": {
            task: {"path": path, "kind": "file", "bytes": 303 + index}
            for index, (task, path) in enumerate(text_map.items())
        },
    }
    module.validate_inputs_binding(member, request, binding)
    return request, binding


def test_gaussian_completion_marker_semantics(module) -> None:
    """The raw-bound marker must also satisfy the metadata-no-hash contract."""

    member = "n3"
    request, valid = _valid_inputs_binding(module, member)
    primary = valid["gaussian_primary"]
    manifest_payload = _decode_compressed_descriptor(
        module, primary["manifest"], label="test Gaussian manifest"
    )
    complete_payload = base64.b64decode(
        primary["completion_marker"]["content_b64"], validate=True
    )
    complete = json.loads(complete_payload)
    assert set(complete) == module.GAUSSIAN_COMPLETE_KEYS

    marker_mutations: list[tuple[str, dict]] = []
    for key in sorted(module.GAUSSIAN_COMPLETE_KEYS):
        changed = copy.deepcopy(complete)
        changed.pop(key)
        marker_mutations.append((f"missing COMPLETE field {key}", changed))
    changed = copy.deepcopy(complete)
    changed["unexpected"] = True
    marker_mutations.append(("extra COMPLETE field", changed))
    marker_mutations.extend(
        (
            ("integer COMPLETE boolean", {**complete, "complete": 1}),
            ("wrong COMPLETE schema name", {**complete, "schema_name": "other"}),
            ("boolean COMPLETE schema version", {**complete, "schema_version": True}),
            ("float COMPLETE manifest version", {**complete, "manifest_version": 1.0}),
            ("wrong COMPLETE manifest name", {**complete, "manifest": "other.json"}),
            ("boolean COMPLETE manifest bytes", {**complete, "manifest_bytes": True}),
            (
                "mismatched COMPLETE manifest bytes",
                {**complete, "manifest_bytes": len(manifest_payload) + 1},
            ),
            ("missing legacy manifest field", {**complete, "manifest_sha256": None}),
            (
                "uppercase legacy manifest field",
                {**complete, "manifest_sha256": "A" * 64},
            ),
            ("boolean COMPLETE shard count", {**complete, "shard_count": True}),
            ("mismatched COMPLETE shard count", {**complete, "shard_count": 2}),
            ("float COMPLETE total frames", {**complete, "total_frames": 2.0}),
            ("mismatched COMPLETE total frames", {**complete, "total_frames": 3}),
        )
    )
    for label, candidate in marker_mutations:
        changed_binding = copy.deepcopy(valid)
        payload = _json_payload(candidate)
        changed_binding["gaussian_primary"]["completion_marker"] = (
            _content_descriptor(
                primary["completion_marker"]["path"], payload
            )
        )
        _assert_runtime_rejected(
            lambda changed_binding=changed_binding: module.validate_inputs_binding(
                member, request, changed_binding
            ),
            label,
        )

    manifest = json.loads(manifest_payload)
    manifest_mutations = (
        ("boolean manifest version", "manifest_version", True),
        ("unsupported manifest version", "manifest_version", 3),
        ("empty manifest shards", "shards", []),
        ("non-list manifest shards", "shards", {}),
        ("boolean manifest total frames", "total_frames", True),
        ("non-positive manifest total frames", "total_frames", 0),
    )
    for label, key, value in manifest_mutations:
        changed_manifest = copy.deepcopy(manifest)
        changed_manifest[key] = value
        changed_manifest_payload = _json_payload(changed_manifest)
        changed_complete = copy.deepcopy(complete)
        changed_complete["manifest_bytes"] = len(changed_manifest_payload)
        changed_binding = copy.deepcopy(valid)
        changed_binding["gaussian_primary"]["manifest"] = _compressed_content_descriptor(
            primary["manifest"]["path"], changed_manifest_payload
        )
        changed_binding["gaussian_primary"]["completion_marker"] = (
            _content_descriptor(
                primary["completion_marker"]["path"],
                _json_payload(changed_complete),
            )
        )
        _assert_runtime_rejected(
            lambda changed_binding=changed_binding: module.validate_inputs_binding(
                member, request, changed_binding
            ),
            label,
        )


def _noncanonical_base64_for_same_payload(value: str) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    if value.endswith("=="):
        index = len(value) - 3
    elif value.endswith("="):
        index = len(value) - 2
    else:
        raise AssertionError("fixture payload must require base64 padding")
    replacement = alphabet[(alphabet.index(value[index]) + 1) % len(alphabet)]
    mutated = value[:index] + replacement + value[index + 1 :]
    assert base64.b64decode(mutated, validate=True) == base64.b64decode(
        value, validate=True
    )
    assert mutated != value
    return mutated


def test_portable_inputs_schema_rejects_legacy_fields_paths_and_tasks(module) -> None:
    member = "n3"
    request, valid = _valid_inputs_binding(module, member)
    mutations: list[tuple[str, dict, dict]] = []

    changed = copy.deepcopy(valid)
    changed["schema"] = "fastwam-formal-portable-input-binding-v1"
    mutations.append(("old input schema", request, changed))
    changed = copy.deepcopy(valid)
    changed["unexpected"] = True
    mutations.append(("unknown top-level field", request, changed))

    for field in ("mode", "mtime_ns", "ctime_ns", "dev", "ino", "nlink"):
        changed = copy.deepcopy(valid)
        changed["initial_checkpoint"][field] = 1
        mutations.append((f"legacy large-file field {field}", request, changed))
        changed = copy.deepcopy(valid)
        changed["normalization_stats"][field] = 1
        mutations.append((f"legacy control-file field {field}", request, changed))
        changed = copy.deepcopy(valid)
        changed["gaussian_primary"]["manifest"][field] = 1
        mutations.append((f"legacy Gaussian field {field}", request, changed))

    for key, path in (
        ("dataset", "/oss-chengjuntao/changed-dataset"),
        ("initial_checkpoint", "/oss-chengjuntao/changed-checkpoint.pt"),
        ("vae", "/oss-chengjuntao/changed-vae.pt"),
    ):
        changed = copy.deepcopy(valid)
        changed[key]["path"] = path
        mutations.append((f"changed {key} path", request, changed))
    changed = copy.deepcopy(valid)
    changed["normalization_stats"]["resolved_source_root"] = (
        "/oss-chengjuntao/changed-dataset"
    )
    mutations.append(("stats dataset mismatch", request, changed))
    changed = copy.deepcopy(valid)
    changed["gaussian_primary"]["manifest"]["path"] += ".changed"
    mutations.append(("Gaussian manifest path mismatch", request, changed))

    first_task = module.MEMBERS[member]["tasks"][0]
    changed = copy.deepcopy(valid)
    changed["text_caches"].pop(first_task)
    mutations.append(("missing member task", request, changed))
    changed = copy.deepcopy(valid)
    changed["text_caches"]["PlaceFood-rf"] = {
        "path": "/oss-chengjuntao/unexpected.pt",
        "kind": "file",
        "bytes": 1,
    }
    mutations.append(("extra member task", request, changed))
    changed_request = copy.deepcopy(request)
    changed_request["Envs"]["FASTWAM_TASKS_JSON"] = '["PlaceFood-rf"]'
    changed_request["Envs"]["FASTWAM_TEXT_CACHE_MAP_JSON"] = json.dumps(
        {"PlaceFood-rf": "/oss-chengjuntao/placefood.pt"}
    )
    mutations.append(("request task scope differs from member", changed_request, valid))

    nested_key_cases = (
        ("dataset", ("dataset",), "kind"),
        ("normalization stats", ("normalization_stats",), "resolved_source_root"),
        ("initial checkpoint", ("initial_checkpoint",), "kind"),
        ("VAE", ("vae",), "bytes"),
        ("Gaussian fallback", ("gaussian_fallback",), "selection_mode"),
        (
            "Gaussian fallback manifest",
            ("gaussian_fallback", "manifest"),
            "compressed_b64",
        ),
        (
            "Gaussian fallback completion marker",
            ("gaussian_fallback", "completion_marker"),
            "bytes",
        ),
        ("text cache", ("text_caches", first_task), "kind"),
    )
    for label, location, key in nested_key_cases:
        changed = copy.deepcopy(valid)
        descriptor = changed
        for part in location:
            descriptor = descriptor[part]
        descriptor.pop(key)
        mutations.append((f"missing {label} field {key}", request, changed))
        changed = copy.deepcopy(valid)
        descriptor = changed
        for part in location:
            descriptor = descriptor[part]
        descriptor["unexpected"] = True
        mutations.append((f"extra {label} field", request, changed))

    for label, candidate_request, candidate in mutations:
        _assert_runtime_rejected(
            lambda candidate_request=candidate_request, candidate=candidate: (
                module.validate_inputs_binding(member, candidate_request, candidate)
            ),
            label,
        )


def test_portable_inputs_reject_bool_float_negative_size_and_base64(module) -> None:
    member = "n3"
    request, valid = _valid_inputs_binding(module, member)
    first_task = module.MEMBERS[member]["tasks"][0]
    size_locations = (
        ("initial checkpoint", ("initial_checkpoint",)),
        ("VAE", ("vae",)),
        ("normalization stats", ("normalization_stats",)),
        ("Gaussian manifest", ("gaussian_primary", "manifest")),
        (
            "Gaussian completion marker",
            ("gaussian_primary", "completion_marker"),
        ),
        ("Gaussian fallback manifest", ("gaussian_fallback", "manifest")),
        (
            "Gaussian fallback completion marker",
            ("gaussian_fallback", "completion_marker"),
        ),
        ("text cache", ("text_caches", first_task)),
    )
    for label, location in size_locations:
        for invalid_size in (True, 1.0, -1):
            changed = copy.deepcopy(valid)
            descriptor = changed
            for key in location:
                descriptor = descriptor[key]
            descriptor["bytes"] = invalid_size
            _assert_runtime_rejected(
                lambda changed=changed: module.validate_inputs_binding(
                    member, request, changed
                ),
                f"{label} bytes={invalid_size!r}",
            )

    changed = copy.deepcopy(valid)
    encoded = changed["normalization_stats"]["content_b64"]
    changed["normalization_stats"]["content_b64"] = (
        _noncanonical_base64_for_same_payload(encoded)
    )
    _assert_runtime_rejected(
        lambda: module.validate_inputs_binding(member, request, changed),
        "non-canonical base64 with identical decoded bytes",
    )
    changed = copy.deepcopy(valid)
    changed["gaussian_primary"]["manifest"]["compressed_b64"] = "not base64"
    _assert_runtime_rejected(
        lambda: module.validate_inputs_binding(member, request, changed),
        "invalid Gaussian base64",
    )


def test_gaussian_manifest_reversible_descriptor_contract(module) -> None:
    assert module.MAX_CONTROL_FILE_BYTES == 16 * 1024**2
    assert module.MAX_GAUSSIAN_MANIFEST_RAW_BYTES == 64 * 1024**2
    assert module.MAX_COMPRESSED_CONTENT_BYTES == 16 * 1024**2
    assert module.GAUSSIAN_MANIFEST_CONTENT_ENCODING == "zlib-level6-base64-v1"

    path = "/oss-chengjuntao/cache/manifest.json"
    payload = _gaussian_manifest("compact")
    descriptor = module.compressed_content_file_metadata(Path(path), payload)
    assert set(descriptor) == {
        "path",
        "kind",
        "bytes",
        "content_encoding",
        "compressed_bytes",
        "compressed_b64",
    }
    assert descriptor == _compressed_content_descriptor(path, payload)
    assert _decode_compressed_descriptor(
        module, descriptor, label="valid reversible descriptor"
    ) == payload

    mutations: list[tuple[str, dict]] = []
    for key in sorted(descriptor):
        changed = copy.deepcopy(descriptor)
        changed.pop(key)
        mutations.append((f"missing compressed descriptor field {key}", changed))
    changed = copy.deepcopy(descriptor)
    changed["unexpected"] = True
    mutations.append(("extra compressed descriptor field", changed))
    for key, value in (
        ("path", f"{path}.changed"),
        ("kind", "directory"),
        ("content_encoding", "zlib-base64"),
        ("content_encoding", 1),
        ("bytes", True),
        ("bytes", 1.0),
        ("bytes", -1),
        ("bytes", module.MAX_GAUSSIAN_MANIFEST_RAW_BYTES + 1),
        ("compressed_bytes", True),
        ("compressed_bytes", 1.0),
        ("compressed_bytes", -1),
        ("compressed_bytes", module.MAX_COMPRESSED_CONTENT_BYTES + 1),
        ("compressed_b64", 1),
        ("compressed_b64", "not base64"),
    ):
        changed = copy.deepcopy(descriptor)
        changed[key] = value
        mutations.append((f"invalid compressed descriptor {key}", changed))

    changed = copy.deepcopy(descriptor)
    changed["compressed_bytes"] += 1
    mutations.append(("declared compressed length mismatch", changed))
    changed = copy.deepcopy(descriptor)
    changed["bytes"] += 1
    mutations.append(("declared decompressed length mismatch", changed))

    padded_descriptor = None
    for suffix_size in range(1, 8):
        candidate = module.compressed_content_file_metadata(
            Path(path), payload + b"x" * suffix_size
        )
        if candidate["compressed_b64"].endswith("="):
            padded_descriptor = candidate
            break
    assert padded_descriptor is not None
    changed = copy.deepcopy(padded_descriptor)
    changed["compressed_b64"] = _noncanonical_base64_for_same_payload(
        changed["compressed_b64"]
    )
    mutations.append(("non-canonical compressed Base64", changed))

    compressed = base64.b64decode(descriptor["compressed_b64"], validate=True)

    def descriptor_for_stream(stream: bytes, *, raw_bytes: int = len(payload)) -> dict:
        changed = copy.deepcopy(descriptor)
        changed["bytes"] = raw_bytes
        changed["compressed_bytes"] = len(stream)
        changed["compressed_b64"] = base64.b64encode(stream).decode("ascii")
        return changed

    mutations.extend(
        (
            ("truncated compressed stream", descriptor_for_stream(compressed[:-1])),
            ("non-EOF compressed stream", descriptor_for_stream(compressed[:2])),
            (
                "concatenated compressed streams",
                descriptor_for_stream(compressed + zlib.compress(b"second", level=6)),
            ),
            ("trailing compressed bytes", descriptor_for_stream(compressed + b"tail")),
            (
                "bounded decompression overrun with unconsumed input",
                descriptor_for_stream(zlib.compress(b"x" * 4096, level=6), raw_bytes=1),
            ),
        )
    )
    for label, candidate in mutations:
        _assert_runtime_rejected(
            lambda candidate=candidate: module.validate_compressed_content_file_descriptor(
                candidate, expected_path=path, label=label
            ),
            label,
        )

    with mock.patch.object(
        module.zlib,
        "compress",
        return_value=b"x" * (module.MAX_COMPRESSED_CONTENT_BYTES + 1),
    ) as compress:
        _assert_runtime_rejected(
            lambda: module.compressed_content_file_metadata(Path(path), payload),
            "compressed Gaussian manifest over 16 MiB",
        )
        compress.assert_called_once_with(payload, level=6)


def test_gaussian_manifest_large_raw_bounds(module) -> None:
    with tempfile.TemporaryDirectory(prefix="formal-r4-large-manifest-") as name:
        prefix = Path(name)
        cache = prefix / "large"
        cache.mkdir()
        manifest = _gaussian_manifest(
            "compact", padding_bytes=module.MAX_CONTROL_FILE_BYTES + 1
        )
        assert module.MAX_CONTROL_FILE_BYTES < len(manifest)
        assert len(manifest) <= module.MAX_GAUSSIAN_MANIFEST_RAW_BYTES
        (cache / "manifest.json").write_bytes(manifest)
        (cache / "COMPLETE").write_bytes(_gaussian_complete(manifest))
        with mock.patch.object(module, "GAUSSIAN_PREFIX", prefix):
            binding = module.validate_gaussian_root(cache, expected_kind="compact")
        assert binding["manifest"]["bytes"] == len(manifest)
        assert _decode_compressed_descriptor(
            module, binding["manifest"], label="large reversible manifest"
        ) == manifest

        oversized = prefix / "oversized"
        oversized.mkdir()
        with (oversized / "manifest.json").open("wb") as stream:
            stream.truncate(module.MAX_GAUSSIAN_MANIFEST_RAW_BYTES + 1)
        (oversized / "COMPLETE").write_bytes(b"{}")
        with mock.patch.object(module, "GAUSSIAN_PREFIX", prefix):
            _assert_runtime_rejected(
                lambda: module.validate_gaussian_root(
                    oversized, expected_kind="compact"
                ),
                "Gaussian manifest over 64 MiB raw bound",
            )


def _write_member_input_fixture(
    module, root: Path, *, member: str, declared_dataset_root: Path
) -> tuple[dict, dict[str, Path]]:
    dataset = root / "dataset"
    dataset.mkdir()
    stats = root / "stats.json"
    stats.write_bytes(
        _json_payload(
            {"source_root": str(declared_dataset_root), "generation": "aa"}
        )
    )
    checkpoint = root / "initial.pt"
    checkpoint.write_bytes(b"initial checkpoint")
    vae = root / "vae.safetensors"
    vae.write_bytes(b"vae weights")
    gaussian_root = root / "gaussian"
    primary = gaussian_root / "primary"
    fallback = gaussian_root / "fallback"
    for cache, kind in ((primary, "compact"), (fallback, "canonical")):
        cache.mkdir(parents=True)
        manifest_payload = _gaussian_manifest(kind)
        (cache / "manifest.json").write_bytes(manifest_payload)
        (cache / "COMPLETE").write_bytes(_gaussian_complete(manifest_payload))
    text_root = root / "text"
    text_root.mkdir()
    text_map = {}
    for task in module.MEMBERS[member]["tasks"]:
        path = text_root / f"{task}.pt"
        path.write_bytes(f"text cache: {task}".encode("utf-8"))
        text_map[task] = str(path)
    request = {
        "Envs": {
            "FASTWAM_DATASET_ROOT": str(dataset),
            "FASTWAM_STATS_SOURCE": str(stats),
            "FASTWAM_INITIAL_CHECKPOINT": str(checkpoint),
            "FASTWAM_VAE_SOURCE": str(vae),
            "FASTWAM_GAUSSIAN_CACHE_DIR": str(primary),
            "FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR": str(fallback),
            "FASTWAM_TASKS_JSON": json.dumps(
                module.MEMBERS[member]["tasks"], separators=(",", ":")
            ),
            "FASTWAM_TEXT_CACHE_MAP_JSON": json.dumps(
                text_map, sort_keys=True, separators=(",", ":")
            ),
        }
    }
    return request, {
        "dataset": dataset,
        "stats": stats,
        "checkpoint": checkpoint,
        "vae": vae,
        "gaussian_root": gaussian_root,
        "primary_manifest": primary / "manifest.json",
        "primary_complete": primary / "COMPLETE",
        "fallback_manifest": fallback / "manifest.json",
        "fallback_complete": fallback / "COMPLETE",
    }


def _collect_fixture_inputs(module, member: str, request: dict, gaussian_root: Path) -> dict:
    with (
        mock.patch.object(module, "validate_request", return_value=None),
        mock.patch.object(module, "OSS_ROOT", Path("/")),
        mock.patch.object(module, "GAUSSIAN_PREFIX", gaussian_root),
    ):
        return module.collect_member_inputs(member, request)


def _replace_path_prefix(value, source: Path, replacement: str = "/portable"):
    if isinstance(value, dict):
        return {
            key: _replace_path_prefix(item, source, replacement)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_path_prefix(item, source, replacement) for item in value]
    if isinstance(value, str) and (
        value == str(source) or value.startswith(f"{source}/")
    ):
        return replacement + value[len(str(source)) :]
    return value


def _all_mapping_keys(value) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for item in value.values():
            result.update(_all_mapping_keys(item))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            result.update(_all_mapping_keys(item))
        return result
    return set()


def test_member_inputs_cross_mount_mode_and_mtime_portability(module) -> None:
    member = "n3"
    shared_memory = Path("/dev/shm")
    assert shared_memory.is_dir(), "/dev/shm is required for the cross-mount test"
    with tempfile.TemporaryDirectory(prefix="formal-r4-input-left-", dir="/tmp") as left_name:
        with tempfile.TemporaryDirectory(
            prefix="formal-r4-input-right-", dir=shared_memory
        ) as right_name:
            with tempfile.TemporaryDirectory(
                prefix="formal-r4-input-alias-", dir="/tmp"
            ) as alias_name:
                left = Path(left_name)
                right = Path(right_name)
                alias = Path(alias_name) / "dataset"
                alias.symlink_to(left / "dataset", target_is_directory=True)
                left_request, left_paths = _write_member_input_fixture(
                    module, left, member=member, declared_dataset_root=alias
                )
                right_request, right_paths = _write_member_input_fixture(
                    module, right, member=member, declared_dataset_root=alias
                )
                for path in (
                    left_paths["stats"],
                    left_paths["checkpoint"],
                    left_paths["vae"],
                    left_paths["primary_manifest"],
                    left_paths["primary_complete"],
                    left_paths["fallback_manifest"],
                    left_paths["fallback_complete"],
                ):
                    os.chmod(path, 0o600)
                    os.utime(path, ns=(1_600_000_000_000_000_000,) * 2)
                for path in (
                    right_paths["stats"],
                    right_paths["checkpoint"],
                    right_paths["vae"],
                    right_paths["primary_manifest"],
                    right_paths["primary_complete"],
                    right_paths["fallback_manifest"],
                    right_paths["fallback_complete"],
                ):
                    os.chmod(path, 0o644)
                    os.utime(path, ns=(1_700_000_000_000_000_000,) * 2)
                assert left_paths["stats"].stat().st_dev != right_paths["stats"].stat().st_dev
                left_binding = _collect_fixture_inputs(
                    module, member, left_request, left_paths["gaussian_root"]
                )
                alias.unlink()
                alias.symlink_to(right / "dataset", target_is_directory=True)
                right_binding = _collect_fixture_inputs(
                    module, member, right_request, right_paths["gaussian_root"]
                )
                normalized_left = _replace_path_prefix(left_binding, left)
                normalized_right = _replace_path_prefix(right_binding, right)
                assert normalized_left == normalized_right
                assert not _all_mapping_keys(normalized_left).intersection(
                    {"mode", "mtime_ns", "ctime_ns", "dev", "ino", "nlink"}
                )


def _same_size_replace(path: Path, payload: bytes) -> None:
    assert len(payload) == path.stat().st_size
    replacement = path.with_name(f".{path.name}.replacement")
    replacement.write_bytes(payload)
    os.replace(replacement, path)


def test_same_size_stats_and_gaussian_control_replacements_are_detected(module) -> None:
    member = "n2"
    with tempfile.TemporaryDirectory(prefix="formal-r4-control-content-") as name:
        root = Path(name)
        request, paths = _write_member_input_fixture(
            module, root, member=member, declared_dataset_root=root / "dataset"
        )
        baseline = _collect_fixture_inputs(
            module, member, request, paths["gaussian_root"]
        )

        changed_stats = _json_payload(
            {"source_root": str(root / "dataset"), "generation": "bb"}
        )
        _same_size_replace(paths["stats"], changed_stats)
        after_stats = _collect_fixture_inputs(
            module, member, request, paths["gaussian_root"]
        )
        assert baseline["normalization_stats"]["bytes"] == after_stats[
            "normalization_stats"
        ]["bytes"]
        assert baseline["normalization_stats"] != after_stats["normalization_stats"]

        original_stats = _json_payload(
            {"source_root": str(root / "dataset"), "generation": "aa"}
        )
        _same_size_replace(paths["stats"], original_stats)
        restored = _collect_fixture_inputs(
            module, member, request, paths["gaussian_root"]
        )
        assert restored == baseline

        for binding_key, cache_kind, manifest_key, complete_key in (
            (
                "gaussian_primary",
                "compact",
                "primary_manifest",
                "primary_complete",
            ),
            (
                "gaussian_fallback",
                "canonical",
                "fallback_manifest",
                "fallback_complete",
            ),
        ):
            original_manifest = _gaussian_manifest(cache_kind, generation="aa")
            changed_manifest = _gaussian_manifest(cache_kind, generation="bb")
            _same_size_replace(paths[manifest_key], changed_manifest)
            after_manifest = _collect_fixture_inputs(
                module, member, request, paths["gaussian_root"]
            )
            assert baseline[binding_key]["manifest"]["bytes"] == after_manifest[
                binding_key
            ]["manifest"]["bytes"]
            assert baseline[binding_key] != after_manifest[binding_key]

            _same_size_replace(paths[manifest_key], original_manifest)
            restored = _collect_fixture_inputs(
                module, member, request, paths["gaussian_root"]
            )
            assert restored == baseline

            original_complete = _gaussian_complete(original_manifest)
            changed_complete = _gaussian_complete(
                original_manifest, legacy_manifest_field="b" * 64
            )
            _same_size_replace(paths[complete_key], changed_complete)
            after_complete = _collect_fixture_inputs(
                module, member, request, paths["gaussian_root"]
            )
            assert restored[binding_key]["completion_marker"]["bytes"] == (
                after_complete[binding_key]["completion_marker"]["bytes"]
            )
            assert restored[binding_key] != after_complete[binding_key]
            _same_size_replace(paths[complete_key], original_complete)
            assert _collect_fixture_inputs(
                module, member, request, paths["gaussian_root"]
            ) == baseline


def _valid_member_reservation(module, member: str, request: dict, inputs: dict) -> dict:
    spec = module.MEMBERS[member]
    return {
        "schema": module.MEMBER_RESERVATION_SCHEMA,
        "suite_id": module.SUITE_ID,
        "external_contract": module.CONTRACT,
        "member": member,
        "experiment_id": spec["experiment_id"],
        "run_id": spec["run_id"],
        "native_agent_count": spec["agent_count"],
        "tasks": spec["tasks"],
        "masked_agent_set": False,
        "treatment": {
            "training_mode": "action_only_cache",
            "video_generation": False,
            "hub_enabled": True,
            "gaussian_enabled": True,
            "trainable_scope": "action",
        },
        "schedule": {
            "max_steps": 1000,
            "save_every": 500,
            "eval_every": 500,
            "offline_eval_num_samples": 32,
            "seed": 42,
            "train_split_seed": 42,
            "val_split_seed": 42,
        },
        "hardware": {"workers": 1, "gpus_per_worker": 8, "total_gpus": 8},
        "storage_contract": module.expected_member_storage_contract(),
        "source": {
            "root": request["Envs"]["FASTWAM_SOURCE_ROOT"],
            "git_commit": request["Envs"]["FASTWAM_CODE_COMMIT"],
            "inventory": {
                "schema": module.SOURCE_INVENTORY_SCHEMA,
                "entries": [{"path": ".", "kind": "directory"}],
            },
        },
        "inputs": inputs,
        "output_root": str(module.output_root(member)),
        "request": request,
        "prepared_at": "2026-08-12T00:00:00Z",
        "semantics": (
            "external generic reservation; trainer terminal contract fields remain null; "
            "terminal success is granted only by the runtime receipt"
        ),
    }


def test_old_reservation_and_prepare_live_collector_contract(module) -> None:
    member = "n3"
    request, inputs = _valid_inputs_binding(module, member)
    reservation = _valid_member_reservation(module, member, request, inputs)
    with mock.patch.object(module, "validate_request", return_value=None):
        assert module.validate_member_reservation_structure(member, reservation) is request
        for old_schema in (
            "fastwam-action-native-agents-reservation-v1",
            "fastwam-action-native-agents-reservation-v2",
        ):
            old = copy.deepcopy(reservation)
            old["schema"] = old_schema
            _assert_runtime_rejected(
                lambda old=old: module.validate_member_reservation_structure(
                    member, old
                ),
                f"old reservation schema {old_schema}",
            )
        unexpected = copy.deepcopy(reservation)
        unexpected["unexpected"] = True
        _assert_runtime_rejected(
            lambda: module.validate_member_reservation_structure(member, unexpected),
            "unexpected reservation top-level field",
        )
        for key in reservation:
            missing = copy.deepcopy(reservation)
            missing.pop(key)
            _assert_runtime_rejected(
                lambda missing=missing: module.validate_member_reservation_structure(
                    member, missing
                ),
                f"missing reservation top-level field {key}",
            )

        scalar_mutations = (
            ("native agent count float", ("native_agent_count",), 3.0),
            ("treatment integer boolean", ("treatment", "video_generation"), 0),
            ("schedule float", ("schedule", "max_steps"), 1000.0),
            ("hardware bool", ("hardware", "workers"), True),
            (
                "storage float",
                ("storage_contract", "per_run_publish_limit_bytes"),
                float(module.PER_RUN_OSS_BUDGET_BYTES),
            ),
            (
                "storage integer boolean",
                ("storage_contract", "publish_step500_state"),
                0,
            ),
        )
        for label, location, value in scalar_mutations:
            changed = copy.deepcopy(reservation)
            branch = changed
            for part in location[:-1]:
                branch = branch[part]
            branch[location[-1]] = value
            _assert_runtime_rejected(
                lambda changed=changed: module.validate_member_reservation_structure(
                    member, changed
                ),
                label,
            )

        for label, key in (
            ("treatment", "trainable_scope"),
            ("schedule", "seed"),
            ("hardware", "total_gpus"),
            ("storage contract", "checkpoint_state_kind"),
            ("source", "git_commit"),
        ):
            changed = copy.deepcopy(reservation)
            branch_name = "storage_contract" if label == "storage contract" else label
            changed[branch_name].pop(key)
            _assert_runtime_rejected(
                lambda changed=changed: module.validate_member_reservation_structure(
                    member, changed
                ),
                f"missing nested {label} field",
            )
            changed = copy.deepcopy(reservation)
            changed[branch_name]["unexpected"] = True
            _assert_runtime_rejected(
                lambda changed=changed: module.validate_member_reservation_structure(
                    member, changed
                ),
                f"extra nested {label} field",
            )

        for invalid_timestamp in (None, True, 1, 1.0, "2026-08-12T00:00:00"):
            changed = copy.deepcopy(reservation)
            changed["prepared_at"] = invalid_timestamp
            _assert_runtime_rejected(
                lambda changed=changed: module.validate_member_reservation_structure(
                    member, changed
                ),
                f"invalid prepared_at {invalid_timestamp!r}",
            )
        changed = copy.deepcopy(reservation)
        changed["semantics"] += " changed"
        _assert_runtime_rejected(
            lambda: module.validate_member_reservation_structure(member, changed),
            "changed reservation semantics",
        )

    tree = ast.parse(CONTROLLER.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    for function_name in ("prepare_one", "validate_reservation_live"):
        calls = [
            node
            for node in ast.walk(functions[function_name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "collect_member_inputs"
        ]
        assert len(calls) == 1, f"{function_name} must call the shared collector once"
        assert [argument.id for argument in calls[0].args if isinstance(argument, ast.Name)] == [
            "member",
            "request",
        ]


def test_prepare_one_is_pure_and_publish_is_explicit(module) -> None:
    member = "n2"
    request, inputs = _valid_inputs_binding(module, member)
    source_entries = {
        "schema": module.SOURCE_INVENTORY_SCHEMA,
        "entries": [{"path": ".", "kind": "directory"}],
    }
    with tempfile.TemporaryDirectory(prefix="formal-r4-prepare-behavior-") as name:
        root = Path(name)
        reservation_destination = root / "prepared-reservation.json"
        output_destination = root / "output"
        with (
            mock.patch.object(module, "build_request", return_value=request),
            mock.patch.object(
                module, "collect_member_inputs", return_value=inputs
            ) as collect,
            mock.patch.object(module, "output_root", return_value=output_destination),
            mock.patch.object(
                module, "validate_member_reservation_structure", return_value=request
            ) as validate_reservation,
            mock.patch.object(module, "exclusive_write") as write_reservation,
            mock.patch.object(module, "atomic_write") as write_state,
            mock.patch.object(
                module, "publish_member_reservation"
            ) as publish_reservation,
            mock.patch.object(
                module, "write_prepared_local_state"
            ) as write_local_state,
            mock.patch.object(module, "utc_now", return_value="2026-08-12T00:00:00Z"),
        ):
            reservation = module.prepare_one(
                member,
                source=Path(request["Envs"]["FASTWAM_SOURCE_ROOT"]),
                source_commit=request["Envs"]["FASTWAM_CODE_COMMIT"],
                source_entries=source_entries,
                dataset=Path(request["Envs"]["FASTWAM_DATASET_ROOT"]),
                stats=Path(request["Envs"]["FASTWAM_STATS_SOURCE"]),
                checkpoint=Path(request["Envs"]["FASTWAM_INITIAL_CHECKPOINT"]),
                vae=Path(request["Envs"]["FASTWAM_VAE_SOURCE"]),
                primary=Path(request["Envs"]["FASTWAM_GAUSSIAN_CACHE_DIR"]),
                fallback=Path(
                    request["Envs"]["FASTWAM_GAUSSIAN_FALLBACK_CACHE_DIR"]
                ),
                text_paths=json.loads(
                    request["Envs"]["FASTWAM_TEXT_CACHE_MAP_JSON"]
                ),
                trusted_runtime=b"runtime-bytes",
            )

        collect.assert_called_once_with(member, request)
        validate_reservation.assert_called_once_with(member, reservation)
        assert set(reservation) == module.MEMBER_RESERVATION_KEYS
        assert reservation["schema"] == module.MEMBER_RESERVATION_SCHEMA
        assert reservation["request"] is request
        assert reservation["inputs"] is inputs
        assert reservation["source"]["inventory"] is source_entries
        assert reservation["prepared_at"] == "2026-08-12T00:00:00Z"
        assert reservation["semantics"] == module.MEMBER_RESERVATION_SEMANTICS
        assert not output_destination.exists()
        write_reservation.assert_not_called()
        write_state.assert_not_called()
        publish_reservation.assert_not_called()
        write_local_state.assert_not_called()

        with (
            mock.patch.object(
                module, "reservation_path", return_value=reservation_destination
            ),
            mock.patch.object(module, "latch_path", return_value=root / "latch.json"),
            mock.patch.object(
                module,
                "acknowledgement_path",
                return_value=root / "acknowledgement.json",
            ),
            mock.patch.object(
                module, "validate_member_reservation_structure", return_value=request
            ),
        ):
            result = module.publish_member_reservation(member, reservation)
            assert result == {
                "member": member,
                "status": "PREPARED",
                "path": str(reservation_destination),
            }
            persisted, _ = module.read_json(reservation_destination)
            assert persisted == reservation
            repeated = module.publish_member_reservation(member, reservation)
        assert repeated == {
            "member": member,
            "status": "ALREADY_PREPARED",
            "path": str(reservation_destination),
        }


def _prepare_test_args(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        member=None,
        source_root=str(root / "source"),
        source_commit="a" * 40,
        dataset_root=str(root / "dataset"),
        stats_source=str(root / "stats.json"),
        initial_checkpoint=str(root / "initial.pt"),
        vae_source=str(root / "vae.safetensors"),
        gaussian_cache=str(root / "gaussian/primary"),
        gaussian_fallback_cache=str(root / "gaussian/fallback"),
        platform_oss_quota_bytes=500 * 1024**3,
        platform_oss_free_bytes=242 * 1024**3,
        platform_oss_quota_evidence="PAI console quota observation 2026-08-12",
        platform_oss_observed_at="2026-08-12T00:00:00Z",
        text_cache_placefood=str(root / "text/placefood.pt"),
        text_cache_three_shoes=str(root / "text/three-shoes.pt"),
        text_cache_three_stack=str(root / "text/three-stack.pt"),
        text_cache_four_stack=str(root / "text/four-stack.pt"),
    )


def _prepare_read_only_patches(module, root: Path):
    source = root / "source"
    return (
        mock.patch.object(
            module, "canonical_expected_source_root", return_value=source
        ),
        mock.patch.object(
            module,
            "validate_source",
            return_value={
                "schema": module.SOURCE_INVENTORY_SCHEMA,
                "entries": [{"path": ".", "kind": "directory"}],
            },
        ),
        mock.patch.object(
            module, "stable_read", return_value=(b"runtime", {"bytes": 7})
        ),
        mock.patch.object(
            module,
            "canonical_oss_path",
            side_effect=lambda value, **_kwargs: Path(value),
        ),
        mock.patch.object(
            module,
            "expected_text_map",
            return_value={
                task: str(root / f"text/{index}.pt")
                for index, task in enumerate(
                    {
                        task
                        for spec in module.MEMBERS.values()
                        for task in spec["tasks"]
                    }
                )
            },
        ),
        mock.patch.object(
            module, "validate_suite_storage_reservation", return_value=None
        ),
    )


def test_prepare_member_failures_leave_no_durable_or_local_state(module) -> None:
    for failure_stage in ("collect", "validate"):
        for failing_member in module.MEMBERS:
            with tempfile.TemporaryDirectory(
                prefix=f"formal-r4-atomic-{failure_stage}-{failing_member}-"
            ) as name:
                root = Path(name)
                python_target = root / "python-target"
                python_target.write_bytes(b"runtime")
                python_target.chmod(0o700)
                python_link = root / "python"
                python_link.symlink_to(python_target)
                output_prefix = root / "durable-output-parent"
                suite_path = root / "suite-reservation.json"
                member_paths = {
                    member: root / f"{member}-reservation.json"
                    for member in module.MEMBERS
                }
                local_paths = {
                    member: root / f"{member}-state.json"
                    for member in module.MEMBERS
                }
                prepared: list[str] = []

                def build_request(member: str, **_kwargs):
                    prepared.append(member)
                    return {"member": member}

                def collect_inputs(member: str, request: dict):
                    assert request == {"member": member}
                    if failure_stage == "collect" and member == failing_member:
                        raise RuntimeError(f"collection failed for {member}")
                    return {"member": member}

                def validate_reservation(member: str, reservation: dict):
                    assert reservation["member"] == member
                    if failure_stage == "validate" and member == failing_member:
                        raise RuntimeError(f"reservation validation failed for {member}")
                    return reservation["request"]

                patches = _prepare_read_only_patches(module, root)
                with ExitStack() as stack:
                    for patcher in patches:
                        stack.enter_context(patcher)
                    stack.enter_context(
                        mock.patch.object(module, "PINNED_PYTHON", python_link)
                    )
                    stack.enter_context(
                        mock.patch.object(
                            module, "PINNED_PYTHON_TARGET", python_target
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(module, "OUTPUT_PREFIX", output_prefix)
                    )
                    stack.enter_context(
                        mock.patch.object(
                            module, "SUITE_STORAGE_RESERVATION_PATH", suite_path
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            module,
                            "reservation_path",
                            side_effect=lambda member: member_paths[member],
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            module,
                            "local_state_path",
                            side_effect=lambda member: local_paths[member],
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            module,
                            "output_root",
                            side_effect=lambda member: output_prefix / member,
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            module, "build_request", side_effect=build_request
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            module,
                            "collect_member_inputs",
                            side_effect=collect_inputs,
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            module,
                            "validate_member_reservation_structure",
                            side_effect=validate_reservation,
                        )
                    )
                    make_output = stack.enter_context(
                        mock.patch.object(module.os, "mkdir")
                    )
                    validate_existing = stack.enter_context(
                        mock.patch.object(module, "validate_existing_member_state")
                    )
                    publish_member = stack.enter_context(
                        mock.patch.object(module, "publish_member_reservation")
                    )
                    publish_suite = stack.enter_context(
                        mock.patch.object(module, "exclusive_write")
                    )
                    atomic_write = stack.enter_context(
                        mock.patch.object(module, "atomic_write")
                    )
                    write_local_state = stack.enter_context(
                        mock.patch.object(module, "write_prepared_local_state")
                    )
                    _assert_runtime_rejected(
                        lambda: module.prepare(_prepare_test_args(root)),
                        f"suite {failure_stage} failure at {failing_member}",
                    )

                expected_prefix = list(module.MEMBERS)
                assert prepared == expected_prefix[
                    : expected_prefix.index(failing_member) + 1
                ]
                make_output.assert_not_called()
                validate_existing.assert_not_called()
                publish_member.assert_not_called()
                publish_suite.assert_not_called()
                atomic_write.assert_not_called()
                write_local_state.assert_not_called()
                assert not output_prefix.exists()
                assert not suite_path.exists()
                assert all(not path.exists() for path in member_paths.values())
                assert all(not path.exists() for path in local_paths.values())


def test_prepare_phase_two_follows_all_pure_results(module) -> None:
    with tempfile.TemporaryDirectory(prefix="formal-r4-phase-two-") as name:
        root = Path(name)
        python_target = root / "python-target"
        python_target.write_bytes(b"runtime")
        python_target.chmod(0o700)
        python_link = root / "python"
        python_link.symlink_to(python_target)
        output_prefix = root / "durable-output-parent"
        suite_path = root / "suite-reservation.json"
        events: list[str] = []
        published_suite: dict[str, dict] = {}
        planned = {
            member: {"member": member, "pure": True} for member in module.MEMBERS
        }
        real_mkdir = os.mkdir

        def prepare_member(member: str, **_kwargs):
            events.append(f"pure:{member}")
            return planned[member]

        def inspect_member(member: str, reservation: dict):
            assert reservation is planned[member]
            events.append(f"inspect:{member}")
            return None

        def create_output(path: Path, mode: int):
            events.append("mkdir:output")
            real_mkdir(path, mode)

        def publish_member(member: str, reservation: dict):
            assert reservation is planned[member]
            events.append(f"publish:{member}")
            return {"member": member, "status": "PREPARED", "path": str(root / member)}

        def validate_suite(_suite: dict):
            events.append("validate:suite-members")
            return planned

        def publish_suite(path: Path, suite: dict):
            assert path == suite_path
            events.append("publish:suite")
            published_suite["value"] = copy.deepcopy(suite)

        def read_suite(path: Path):
            assert path == suite_path
            events.append("read:suite")
            return copy.deepcopy(published_suite["value"]), {}

        def write_local(member: str):
            events.append(f"local:{member}")

        patches = _prepare_read_only_patches(module, root)
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            stack.enter_context(
                mock.patch.object(module, "PINNED_PYTHON", python_link)
            )
            stack.enter_context(
                mock.patch.object(module, "PINNED_PYTHON_TARGET", python_target)
            )
            stack.enter_context(mock.patch.object(module, "OUTPUT_PREFIX", output_prefix))
            stack.enter_context(
                mock.patch.object(module, "SUITE_STORAGE_RESERVATION_PATH", suite_path)
            )
            stack.enter_context(
                mock.patch.object(module, "prepare_one", side_effect=prepare_member)
            )
            stack.enter_context(
                mock.patch.object(
                    module,
                    "validate_existing_member_state",
                    side_effect=inspect_member,
                )
            )
            stack.enter_context(
                mock.patch.object(module.os, "mkdir", side_effect=create_output)
            )
            stack.enter_context(
                mock.patch.object(
                    module, "publish_member_reservation", side_effect=publish_member
                )
            )
            stack.enter_context(
                mock.patch.object(
                    module,
                    "validate_complete_suite_members",
                    side_effect=validate_suite,
                )
            )
            stack.enter_context(
                mock.patch.object(module, "exclusive_write", side_effect=publish_suite)
            )
            stack.enter_context(
                mock.patch.object(module, "read_json", side_effect=read_suite)
            )
            stack.enter_context(
                mock.patch.object(
                    module, "write_prepared_local_state", side_effect=write_local
                )
            )
            stack.enter_context(mock.patch("builtins.print"))
            module.prepare(_prepare_test_args(root))

        assert output_prefix.is_dir()
        assert events[:3] == ["pure:n2", "pure:n3", "pure:n4"]
        assert events[3:6] == ["inspect:n2", "inspect:n3", "inspect:n4"]
        assert events.index("mkdir:output") > events.index("inspect:n4")
        assert events.index("publish:n2") > events.index("mkdir:output")
        assert events.index("publish:suite") > events.index("publish:n4")
        assert events.index("read:suite") > events.index("publish:suite")
        for member in module.MEMBERS:
            assert events.index(f"local:{member}") > events.index("read:suite")


def test_live_validation_rejects_all_same_size_control_changes(module) -> None:
    member = "n2"
    request, inputs = _valid_inputs_binding(module, member)
    reservation = _valid_member_reservation(module, member, request, inputs)
    source = reservation["source"]
    stats_payload = base64.b64decode(
        inputs["normalization_stats"]["content_b64"], validate=True
    )
    changed_payload = stats_payload.replace(b'"aa"', b'"bb"')
    assert changed_payload != stats_payload and len(changed_payload) == len(stats_payload)
    changed_stats = copy.deepcopy(inputs)
    changed_stats["normalization_stats"]["content_b64"] = base64.b64encode(
        changed_payload
    ).decode("ascii")
    module.validate_inputs_binding(member, request, changed_stats)

    changed_bindings = [
        (
            "same-size stats replacement between prepare and live validation",
            changed_stats,
        )
    ]
    for binding_key in ("gaussian_primary", "gaussian_fallback"):
        manifest_descriptor = inputs[binding_key]["manifest"]
        manifest_payload = _decode_compressed_descriptor(
            module, manifest_descriptor, label=f"{binding_key} baseline manifest"
        )
        replacement_manifest = manifest_payload.replace(b'"aa"', b'"bb"', 1)
        assert replacement_manifest != manifest_payload
        assert len(replacement_manifest) == len(manifest_payload)
        changed_manifest = copy.deepcopy(inputs)
        changed_manifest[binding_key]["manifest"] = _compressed_content_descriptor(
            manifest_descriptor["path"], replacement_manifest
        )
        module.validate_inputs_binding(member, request, changed_manifest)
        changed_bindings.append(
            (f"same-size {binding_key} manifest replacement", changed_manifest)
        )

        marker_descriptor = inputs[binding_key]["completion_marker"]
        marker_payload = base64.b64decode(
            marker_descriptor["content_b64"], validate=True
        )
        replacement_marker = marker_payload.replace(b"a" * 64, b"b" * 64, 1)
        assert replacement_marker != marker_payload
        assert len(replacement_marker) == len(marker_payload)
        changed_marker = copy.deepcopy(inputs)
        changed_marker[binding_key]["completion_marker"] = _content_descriptor(
            marker_descriptor["path"], replacement_marker
        )
        module.validate_inputs_binding(member, request, changed_marker)
        changed_bindings.append(
            (f"same-size {binding_key} COMPLETE replacement", changed_marker)
        )

    def live_validate(observed_inputs: dict):
        common_patches = (
            mock.patch.object(
                module, "read_json", return_value=(suite_record(module), {})
            ),
            mock.patch.object(
                module,
                "validate_complete_suite_members",
                return_value={member: reservation},
            ),
            mock.patch.object(
                module, "validate_member_reservation_structure", return_value=request
            ),
            mock.patch.object(
                module, "canonical_oss_path", return_value=Path("/tmp/output")
            ),
            mock.patch.object(
                module, "canonical_direct_child", return_value=Path("/tmp/source")
            ),
            mock.patch.object(
                module, "source_inventory", return_value=source["inventory"]
            ),
        )
        with (
            common_patches[0],
            common_patches[1],
            common_patches[2],
            common_patches[3],
            common_patches[4],
            common_patches[5],
            mock.patch.object(
                module, "collect_member_inputs", return_value=observed_inputs
            ) as collect,
        ):
            result = module.validate_reservation_live(
                member, reservation, require_output_absent=False
            )
            collect.assert_called_once_with(member, request)
            return result

    assert live_validate(inputs) is request
    for label, changed_inputs in changed_bindings:
        _assert_runtime_rejected(
            lambda changed_inputs=changed_inputs: live_validate(changed_inputs),
            label,
        )


def test_suite_rejects_common_input_mismatch(module) -> None:
    reservations: dict[str, dict] = {}
    for member in module.MEMBERS:
        request, inputs = _valid_inputs_binding(module, member)
        reservations[member] = _valid_member_reservation(
            module, member, request, inputs
        )

    def read_member(path: Path):
        matches = [
            reservation
            for member, reservation in reservations.items()
            if path == module.reservation_path(member)
        ]
        assert len(matches) == 1
        return matches[0], {}

    suite = suite_record(module)
    with (
        mock.patch.object(module, "read_json", side_effect=read_member),
        mock.patch.object(module, "validate_request", return_value=None),
    ):
        assert module.validate_complete_suite_members(suite) == reservations

    changed = copy.deepcopy(reservations["n4"])
    changed["inputs"]["initial_checkpoint"]["bytes"] += 1
    module.validate_inputs_binding("n4", changed["request"], changed["inputs"])
    reservations["n4"] = changed
    with (
        mock.patch.object(module, "read_json", side_effect=read_member),
        mock.patch.object(module, "validate_request", return_value=None),
    ):
        _assert_runtime_rejected(
            lambda: module.validate_complete_suite_members(suite),
            "suite common initial-checkpoint descriptor mismatch",
        )


def test_first_frozen_controller_import_does_not_mutate_source() -> None:
    with tempfile.TemporaryDirectory(prefix="formal-r4-import-") as temporary:
        source = Path(temporary) / "source"
        controller = source / CONTROLLER.name
        source.mkdir()
        shutil.copyfile(CONTROLLER, controller)
        before = {
            path.relative_to(source).as_posix()
            for path in source.rglob("*")
        }
        script = """
import importlib.util
import sys
from pathlib import Path

controller_path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("formal_first_import_test", controller_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load exact controller file")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
if Path(module.__file__).resolve(strict=True) != controller_path.resolve(strict=True):
    raise RuntimeError("controller import escaped exact sibling file")
"""
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run(
            [sys.executable, "-B", "-I", "-S", "-", str(controller)],
            input=script,
            text=True,
            env=environment,
            check=True,
        )
        after = {
            path.relative_to(source).as_posix()
            for path in source.rglob("*")
        }
        assert after == before
        assert not any(
            path.name == "__pycache__" or path.suffix == ".pyc"
            for path in source.rglob("*")
        )


def test_runtime_structure() -> None:
    text = RUNTIME.read_text(encoding="utf-8")
    assert text.count("launch_training \"") == 3
    for identity in (
        "/oss-chengjuntao/artifacts/fastwam-action-n234-formal-r4-20260812/",
        "/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/fastwam-action-n234-formal-r4-20260812-r1",
        "FASTWAM-MR-FT-ACT-N2-PLACEFOOD-1K-S42-R4-20260812",
        "FASTWAM-MR-FT-ACT-N3-POOL-1K-S42-R4-20260812",
        "FASTWAM-MR-FT-ACT-N4-STACKCUBE-1K-S42-R4-20260812",
        "fastwam-act-n2-placefood-1k-s42-r4-20260812",
        "fastwam-act-n3-pool-1k-s42-r4-20260812",
        "fastwam-act-n4-stackcube-1k-s42-r4-20260812",
    ):
        assert identity in text
    for required in (
        "+artifact_integrity_mode=metadata_no_hash",
        "+model.checkpoint_integrity_mode=metadata_no_hash",
        "+data.train.integrity_mode=metadata_no_hash",
        "+data.val.integrity_mode=metadata_no_hash",
        "data.train.gaussian_cache_verify=manifest",
        "data.val.gaussian_cache_verify=manifest",
        "+data.train.text_embedding_cache_files=${FASTWAM_TEXT_CACHE_MAP_JSON}",
        "+data.val.text_embedding_cache_files=${FASTWAM_TEXT_CACHE_MAP_JSON}",
        "training_terminal_contract=null",
        "training_run_profile=null",
        "training_task_scope_receipt=null",
        "+recovery_gate_stop_after_checkpoint_step=500",
        "checkpoint_state_kind=full",
        "seal_training_state=false",
        "seal_training_run=false",
        "terminal_rehash_weights=false",
        "save_training_state=true",
        "save_training_state=false",
        "step_000500",
        "step_001000",
        "phase3_fresh_world_load",
        'cd "${LOCAL_SOURCE}"',
        'export PYTHONPATH="${LOCAL_SOURCE}/src"',
        "fastwam import escaped staged source",
        '[[ "${PYTHONPATH}" == "${LOCAL_SOURCE}/src" ]]',
        '[[ -L "${FASTWAM_PYTHON}" && -x "${FASTWAM_PYTHON}" ]]',
        'resolved_python="$(readlink -f -- "${FASTWAM_PYTHON}")"',
        '[[ "${resolved_python}" == "${FASTWAM_PYTHON_TARGET}" ]]',
        '[[ -f "${FASTWAM_PYTHON_TARGET}" && -x "${FASTWAM_PYTHON_TARGET}" && ! -L "${FASTWAM_PYTHON_TARGET}" ]]',
        'rm -rf -- "${STEP500_STATE}"',
        "O_EXCL",
        "COMPLETE last",
        "zero-byte write while publishing durable artifact",
    ):
        assert required in text
    assert text.count("+recovery_gate_stop_after_checkpoint_step=500") == 1
    assert text.count("root.mkdir(parents=False, mode=0o700)") == 1
    assert 'prepared durable output prefix is absent' in text
    assert text.index('launch_training "${PHASE3_LOG}"') < text.index(
        "root.mkdir(parents=False, mode=0o700)"
    )
    assert "checkpoints/state/step_000500/" not in text
    assert '"checkpoints/state/step_001000/"' in text
    assert "checkpoints/weights/step_000500.pt" in text
    assert "checkpoints/weights/step_001000.pt" in text
    assert text.rfind('create_bytes(root / "COMPLETE"') > text.rfind("stream_copy(")
    publisher_start = text.index("# Build and validate the complete allowlist")
    publisher = text[publisher_start:]
    prepublisher = text[:publisher_start]
    assert 'f"{checkpoint.name}.manifest.json"' in prepublisher
    assert 'f"{checkpoint.name}.COMPLETE"' in prepublisher
    assert '"checkpoint_state_kind": "full"' in prepublisher
    assert ".pt.manifest.json" not in publisher
    assert ".pt.COMPLETE" not in publisher
    enumeration = text.split("# STATE_TREE_ENUMERATION_BEGIN\n", 1)[1].split(
        "# STATE_TREE_ENUMERATION_END", 1
    )[0]
    namespace = {"stat": stat}
    exec(enumeration, namespace)
    enumerate_state_tree = namespace["enumerate_state_tree"]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "regular").write_bytes(b"state")
        assert len(enumerate_state_tree(root)) == 1
        root_link = root.parent / f"{root.name}-root-link"
        root_link.symlink_to(root, target_is_directory=True)
        try:
            enumerate_state_tree(root_link)
        except RuntimeError:
            pass
        else:
            raise AssertionError("state tree root directory symlink must fail closed")
        root_link.unlink()
        directory = root / "directory"
        directory.mkdir()
        (root / "directory-link").symlink_to(directory, target_is_directory=True)
        try:
            enumerate_state_tree(root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("state tree directory symlink must fail closed")
        (root / "directory-link").unlink()
        os.link(root / "regular", root / "hardlink")
        try:
            enumerate_state_tree(root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("state tree hardlink must fail closed")
    assert "os.fsync" not in publisher
    assert "disk_usage" not in text
    assert "df -PB1 /tmp" in text
    for suffix in (
        "gaussian_cache_expected_manifest_sha256=null",
        "gaussian_cache_expected_selection_sha256=null",
        "gaussian_cache_expected_source_identity_sha256=null",
    ):
        assert text.count(suffix) == 2
    for forbidden in ("hashlib", "sha256sum", "md5sum", "blake2", "checksum"):
        assert forbidden not in text.lower()
    assert "sparse_delta" not in text
    assert "new pod" not in text.lower()
    for retired in ("R1-20260811", "R2-20260811", "-r1-20260811", "-r2-20260811"):
        assert retired not in text
    source_copy_validation = text.split(
        'cp -a -- "${FASTWAM_SOURCE_ROOT}/." "${LOCAL_SOURCE}/"', 1
    )[1].split("PY\n", 1)[0]
    assert "reservation, _ = module.read_json(Path(reservation_literal))" in (
        source_copy_validation
    )
    assert 'source = reservation.get("source")' in source_copy_validation
    assert 'expected = source.get("inventory")' in source_copy_validation
    assert (
        "expected = module.source_inventory(Path(source_literal))"
        not in source_copy_validation
    )
    assert "observed = module.source_inventory(Path(target_literal))" in (
        source_copy_validation
    )
    assert "source_inventory" in source_copy_validation
    assert "assert_source_inventory_matches" in source_copy_validation
    assert ".rglob(" not in source_copy_validation
    controller_import_lines = [
        line
        for line in text.splitlines()
        if line.startswith('"${FASTWAM_PYTHON}"')
        and '"${SOURCE_CONTROLLER}"' in line
    ]
    assert len(controller_import_lines) == 2
    assert all(
        line.startswith(
            '"${FASTWAM_PYTHON}" -B -I -S - "${SOURCE_CONTROLLER}"'
        )
        for line in controller_import_lines
    )
    first_controller_import = text.index(
        '"${FASTWAM_PYTHON}" -B -I -S - "${SOURCE_CONTROLLER}"'
    )
    assert text.index("export PYTHONDONTWRITEBYTECODE=1") < first_controller_import
    assert 'SOURCE_CONTROLLER="${FASTWAM_SOURCE_ROOT}/${CONTROLLER_REL}"' in text
    assert "spec_from_file_location(\"formal_worker_controller\", controller_path)" in text


def test_runtime_durable_writers_fail_closed() -> None:
    text = RUNTIME.read_text(encoding="utf-8")
    writers = text.split("def write_all(fd, payload):\n", 1)[1].split(
        "\n# This is intentionally the first mutation beneath the unique durable output.",
        1,
    )[0]
    namespace = {"os": os}
    exec("def write_all(fd, payload):\n" + writers, namespace)
    create_bytes = namespace["create_bytes"]

    with tempfile.TemporaryDirectory(prefix="formal-r4-runtime-writer-") as name:
        root = Path(name)
        collision = root / "COMPLETE"
        collision.write_bytes(b"immutable-existing-marker\n")
        try:
            create_bytes(collision, b"replacement\n")
        except FileExistsError:
            pass
        else:
            raise AssertionError("runtime O_EXCL collision was overwritten")
        assert collision.read_bytes() == b"immutable-existing-marker\n"

        zero_write = root / "zero-write.json"
        with mock.patch.object(os, "write", return_value=0):
            try:
                create_bytes(zero_write, b"must-not-complete")
            except OSError as error:
                assert "zero-byte write" in str(error)
            else:
                raise AssertionError("runtime accepted a zero-byte durable write")
        assert zero_write.is_file()
        assert zero_write.read_bytes() == b""

        target = root / "symlink-target"
        target.write_bytes(b"do-not-truncate\n")
        linked_destination = root / "linked-destination"
        linked_destination.symlink_to(target)
        try:
            create_bytes(linked_destination, b"replacement\n")
        except OSError:
            pass
        else:
            raise AssertionError("runtime durable writer followed a destination symlink")
        assert target.read_bytes() == b"do-not-truncate\n"


def test_runtime_staged_copy_uses_prepared_inventory_counterexample() -> None:
    text = RUNTIME.read_text(encoding="utf-8")
    marker = (
        '"${FASTWAM_PYTHON}" -B -I -S - "${SOURCE_CONTROLLER}" '
        '"${FASTWAM_MEMBER}" "${FASTWAM_PREPARED_RESERVATION_PATH}" '
        '"${FASTWAM_SOURCE_ROOT}" "${LOCAL_SOURCE}" <<\'PY\'\n'
    )
    assert text.count(marker) == 1
    stage_script = text.split(marker, 1)[1].split("\nPY\n", 1)[0]

    with tempfile.TemporaryDirectory(prefix="formal-r4-stage-binding-") as name:
        root = Path(name)
        controller = root / "controller.py"
        source = root / "source"
        target = root / "target"
        reservation = root / "reservation.json"
        source.mkdir()
        target.mkdir()
        (source / "payload.bin").write_bytes(b"changed")
        (target / "payload.bin").write_bytes(b"changed")
        controller.write_text(
            """import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")), {}


def source_inventory(path):
    return {"payload": (Path(path) / "payload.bin").read_text(encoding="utf-8")}


def assert_source_inventory_matches(expected, observed, *, label):
    if expected != observed:
        raise RuntimeError(label)
""",
            encoding="utf-8",
        )
        reservation.write_text(
            json.dumps(
                {
                    "member": "n2",
                    "source": {
                        "root": str(source),
                        "inventory": {"payload": "original"},
                    },
                }
            ),
            encoding="utf-8",
        )
        rejected = subprocess.run(
            [
                sys.executable,
                "-B",
                "-I",
                "-S",
                "-",
                str(controller),
                "n2",
                str(reservation),
                str(source),
                str(target),
            ],
            input=stage_script,
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected.returncode != 0
        assert "staged source portable content mismatch" in rejected.stderr

        reservation.write_text(
            json.dumps(
                {
                    "member": "n2",
                    "source": {
                        "root": str(source),
                        "inventory": {"payload": "changed"},
                    },
                }
            ),
            encoding="utf-8",
        )
        accepted = subprocess.run(
            [
                sys.executable,
                "-B",
                "-I",
                "-S",
                "-",
                str(controller),
                "n2",
                str(reservation),
                str(source),
                str(target),
            ],
            input=stage_script,
            text=True,
            capture_output=True,
            check=False,
        )
        assert accepted.returncode == 0, accepted.stderr


def _make_wrapper_fixture(
    case_root: Path,
    lock_anchor: Path,
    lock_root: Path,
    *,
    pause_before_revalidation: bool = False,
    reported_euid: int | None = None,
) -> tuple[Path, Path, Path, dict[str, str]]:
    case_root.mkdir(mode=0o700)
    os.chmod(case_root, 0o700)
    interpreter_target = Path(sys.executable).resolve(strict=True)
    interpreter_link = case_root / "control-python"
    interpreter_link.symlink_to(interpreter_target)
    marker = case_root / "controller-ran"
    controller = case_root / "controller.py"
    controller.write_text(
        """import fcntl
import os
import stat

lock_path = os.environ["FASTWAM_WRAPPER_EXPECTED_LOCK"]
marker_path = os.environ["FASTWAM_WRAPPER_CONTROLLER_MARKER"]
opened = os.fstat(9)
named = os.lstat(lock_path)
if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
    raise RuntimeError("descriptor 9 is not a single-link regular lock")
if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
    raise RuntimeError("descriptor 9 is not the named R4 lock")
if os.get_inheritable(9) is not True:
    raise RuntimeError("descriptor 9 did not survive exec")
if os.environ.get("FASTWAM_CONTROL_NODE") != "ssh970":
    raise RuntimeError("missing control-node binding")
if os.environ.get("FASTWAM_LOCK_FD") != "9":
    raise RuntimeError("missing descriptor binding")
fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
with open(marker_path, "xb") as handle:
    handle.write(b"controller-ran\\n")
""",
        encoding="utf-8",
    )

    wrapper_text = WRAPPER.read_text(encoding="utf-8")
    preflight_start = wrapper_text.index('"${CONTROL_PYTHON}" -I -c')
    lock_assignment = wrapper_text.index('LOCK_ANCHOR="', preflight_start)
    wrapper_text = (
        wrapper_text[:preflight_start]
        + '"${CONTROL_PYTHON}" -B -I -S -c \'pass\'\n'
        + wrapper_text[lock_assignment:]
    )
    substitutions = {
        "CONTROL_PYTHON=/mnt/workspace/tools/pai-control-py312/20260717-credentials1.0.10-dlc1.9.2-aiworkspace8.2.0/bin/python": (
            f"CONTROL_PYTHON={json.dumps(str(interpreter_link))}"
        ),
        "CONTROL_PYTHON_TARGET=/usr/local/bin/python3.12": (
            f"CONTROL_PYTHON_TARGET={json.dumps(str(interpreter_target))}"
        ),
        'LOCK_ANCHOR="/run"': f"LOCK_ANCHOR={json.dumps(str(lock_anchor))}",
        'LOCK_ROOT="/run/fastwam-dlc-submit-state/workspace-270969"': (
            f"LOCK_ROOT={json.dumps(str(lock_root))}"
        ),
    }
    for old, new in substitutions.items():
        assert wrapper_text.count(old) == 1
        wrapper_text = wrapper_text.replace(old, new)

    environment = dict(os.environ)
    environment.update(
        {
            "SSH_CONNECTION": "127.0.0.1 12345 127.0.0.1 970",
            "FASTWAM_WRAPPER_EXPECTED_LOCK": str(
                lock_root / "action-n234-formal-r4-controller.lock"
            ),
            "FASTWAM_WRAPPER_CONTROLLER_MARKER": str(marker),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    bootstrap_imports = ""
    if reported_euid is not None:
        environment["FASTWAM_LOCK_TEST_REPORTED_EUID"] = str(reported_euid)
        bootstrap_imports += (
            'os.geteuid = lambda: int('
            'os.environ["FASTWAM_LOCK_TEST_REPORTED_EUID"])\n'
        )
    if pause_before_revalidation:
        ready = case_root / "lock-bootstrap-ready"
        release = case_root / "lock-bootstrap-release"
        bootstrap_imports += "import time\n"
        injection = """        ready_path = os.environ["FASTWAM_LOCK_TEST_READY"]
        release_path = os.environ["FASTWAM_LOCK_TEST_RELEASE"]
        ready_fd = os.open(
            ready_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        os.close(ready_fd)
        while True:
            try:
                release_fd = os.open(release_path, os.O_RDONLY | os.O_NOFOLLOW)
            except FileNotFoundError:
                time.sleep(0.01)
                continue
            os.close(release_fd)
            break
"""
        assert wrapper_text.count("        # LOCK_TEST_REVALIDATION_POINT") == 1
        wrapper_text = wrapper_text.replace(
            "        # LOCK_TEST_REVALIDATION_POINT", injection.rstrip("\n")
        )
        environment["FASTWAM_LOCK_TEST_READY"] = str(ready)
        environment["FASTWAM_LOCK_TEST_RELEASE"] = str(release)

    assert wrapper_text.count("import sys\n") == 1
    wrapper_text = wrapper_text.replace(
        "import sys\n", f"import sys\n{bootstrap_imports}", 1
    )

    wrapper = case_root / "submit_from_ssh970.sh"
    wrapper.write_text(wrapper_text, encoding="utf-8")
    wrapper.chmod(0o700)
    return wrapper, marker, interpreter_target, environment


def test_wrapper_uses_pinned_nofollow_lock_bootstrap(module) -> None:
    wrapper_text = WRAPPER.read_text(encoding="utf-8")
    assert '[[ -n "${SSH_CONNECTION:-}" ]]' in wrapper_text
    control_python = (
        "/mnt/workspace/tools/pai-control-py312/"
        "20260717-credentials1.0.10-dlc1.9.2-aiworkspace8.2.0/bin/python"
    )
    assert f"CONTROL_PYTHON={control_python}" in wrapper_text
    assert "CONTROL_PYTHON_TARGET=/usr/local/bin/python3.12" in wrapper_text
    assert '[[ -L "${CONTROL_PYTHON}" && -x "${CONTROL_PYTHON}" ]]' in wrapper_text
    assert 'realpath -e -- "${CONTROL_PYTHON}"' in wrapper_text
    assert '== "${CONTROL_PYTHON_TARGET}"' in wrapper_text
    assert '-L "${CONTROL_PYTHON_TARGET}"' in wrapper_text
    assert "import alibabacloud_credentials,alibabacloud_pai_dlc20201203" in wrapper_text
    assert 'environment["FASTWAM_CONTROL_NODE"] = "ssh970"' in wrapper_text
    assert 'environment["FASTWAM_LOCK_FD"] = str(expected_lock_fd)' in wrapper_text
    assert 'environment["PYTHONDONTWRITEBYTECODE"] = "1"' in wrapper_text
    assert 'LOCK_ANCHOR="/run"' in wrapper_text
    assert 'LOCK_ROOT="/run/fastwam-dlc-submit-state/workspace-270969"' in wrapper_text
    assert "action-n234-formal-r4-controller.lock" in wrapper_text
    assert "action-n234-formal-controller.lock" not in wrapper_text
    assert "exec 9>" not in wrapper_text
    assert "flock -n 9" not in wrapper_text
    assert "os.O_NOFOLLOW" in wrapper_text
    assert "opened.st_uid != os.geteuid()" in wrapper_text
    assert "stat.S_IMODE(opened.st_mode) & 0o022" in wrapper_text
    assert "validate_anchor(anchor_fd)" in wrapper_text
    assert "dir_fd=parent_fd" in wrapper_text
    assert "validate_private_directory(*edge)" in wrapper_text
    assert "validate_lock(parent_fd, lock_fd)" in wrapper_text
    assert "fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)" in wrapper_text
    assert "os.dup2(lock_fd, expected_lock_fd, inheritable=True)" in wrapper_text
    assert "os.execve(" in wrapper_text
    assert '[control_python, "-B", "-I", controller, *controller_args]' in wrapper_text
    assert "/usr/bin/python3" not in wrapper_text
    expected_lock = str(module.CONTROL_LOCK_PATH)
    assert expected_lock == (
        "/run/fastwam-dlc-submit-state/workspace-270969/"
        "action-n234-formal-r4-controller.lock"
    )
    assert expected_lock in README.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="formal-r4-real-wrapper-") as base_name:
        base = Path(base_name)
        os.chmod(base, 0o700)

        positive_case = base / "positive"
        positive_anchor = positive_case / "anchor"
        positive_root = positive_anchor / "state" / "workspace"
        positive_wrapper, positive_marker, _target, positive_env = (
            _make_wrapper_fixture(positive_case, positive_anchor, positive_root)
        )
        positive_anchor.mkdir(mode=0o755)
        os.chmod(positive_anchor, 0o755)
        assert stat.S_IMODE(os.lstat(positive_anchor).st_mode) == 0o755
        assert not os.lstat(positive_anchor).st_mode & stat.S_ISVTX
        positive = subprocess.run(
            ["/bin/bash", str(positive_wrapper)],
            env=positive_env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert positive.returncode == 0, positive.stderr
        assert positive_marker.read_bytes() == b"controller-ran\n"
        lock_metadata = os.lstat(
            positive_root / "action-n234-formal-r4-controller.lock"
        )
        assert stat.S_ISREG(lock_metadata.st_mode)
        assert stat.S_IMODE(lock_metadata.st_mode) == 0o600
        assert lock_metadata.st_nlink == 1
        assert stat.S_IMODE(os.lstat(positive_anchor / "state").st_mode) == 0o700
        assert stat.S_IMODE(os.lstat(positive_root).st_mode) == 0o700

        for label, unsafe_mode in (("world-writable", 0o777), ("other-writable", 0o757)):
            unsafe_case = base / label
            unsafe_anchor = unsafe_case / "anchor"
            unsafe_root = unsafe_anchor / "state" / "workspace"
            unsafe_wrapper, unsafe_marker, _target, unsafe_env = (
                _make_wrapper_fixture(unsafe_case, unsafe_anchor, unsafe_root)
            )
            unsafe_anchor.mkdir(mode=0o700)
            os.chmod(unsafe_anchor, unsafe_mode)
            assert not os.lstat(unsafe_anchor).st_mode & stat.S_ISVTX
            rejected_unsafe = subprocess.run(
                ["/bin/bash", str(unsafe_wrapper)],
                env=unsafe_env,
                text=True,
                capture_output=True,
                check=False,
            )
            assert rejected_unsafe.returncode != 0
            assert "unsafe controller lock anchor" in rejected_unsafe.stderr
            assert not (unsafe_anchor / "state").exists()
            assert not unsafe_marker.exists()

        wrong_owner_case = base / "wrong-owner"
        wrong_owner_anchor = wrong_owner_case / "anchor"
        wrong_owner_root = wrong_owner_anchor / "state" / "workspace"
        wrong_owner_wrapper, wrong_owner_marker, _target, wrong_owner_env = (
            _make_wrapper_fixture(
                wrong_owner_case,
                wrong_owner_anchor,
                wrong_owner_root,
                reported_euid=os.geteuid() + 1,
            )
        )
        wrong_owner_anchor.mkdir(mode=0o755)
        os.chmod(wrong_owner_anchor, 0o755)
        rejected_owner = subprocess.run(
            ["/bin/bash", str(wrong_owner_wrapper)],
            env=wrong_owner_env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected_owner.returncode != 0
        assert "unsafe controller lock anchor" in rejected_owner.stderr
        assert not (wrong_owner_anchor / "state").exists()
        assert not wrong_owner_marker.exists()

        anchor_link_case = base / "anchor-symlink"
        anchor_link = anchor_link_case / "anchor"
        anchor_target = anchor_link_case / "anchor-target"
        anchor_link_root = anchor_link / "state" / "workspace"
        anchor_link_wrapper, anchor_link_marker, _target, anchor_link_env = (
            _make_wrapper_fixture(
                anchor_link_case, anchor_link, anchor_link_root
            )
        )
        anchor_target.mkdir(mode=0o755)
        os.chmod(anchor_target, 0o755)
        anchor_link.symlink_to(anchor_target, target_is_directory=True)
        rejected_anchor_link = subprocess.run(
            ["/bin/bash", str(anchor_link_wrapper)],
            env=anchor_link_env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected_anchor_link.returncode != 0
        assert not (anchor_target / "state").exists()
        assert not anchor_link_marker.exists()

        public_child_case = base / "public-child"
        public_child_anchor = public_child_case / "anchor"
        public_child_root = public_child_anchor / "state" / "workspace"
        public_child_wrapper, public_child_marker, _target, public_child_env = (
            _make_wrapper_fixture(
                public_child_case, public_child_anchor, public_child_root
            )
        )
        public_child_anchor.mkdir(mode=0o755)
        os.chmod(public_child_anchor, 0o755)
        public_state = public_child_anchor / "state"
        public_state.mkdir(mode=0o755)
        os.chmod(public_state, 0o755)
        rejected_public_child = subprocess.run(
            ["/bin/bash", str(public_child_wrapper)],
            env=public_child_env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected_public_child.returncode != 0
        assert "unsafe controller lock directory component: state" in (
            rejected_public_child.stderr
        )
        assert not public_child_root.exists()
        assert not public_child_marker.exists()

        symlink_case = base / "symlink"
        symlink_anchor = symlink_case / "anchor"
        symlink_root = symlink_anchor / "state" / "workspace"
        symlink_wrapper, symlink_marker, _target, symlink_env = (
            _make_wrapper_fixture(symlink_case, symlink_anchor, symlink_root)
        )
        symlink_anchor.mkdir(mode=0o755)
        os.chmod(symlink_anchor, 0o755)
        state = symlink_anchor / "state"
        state.mkdir(mode=0o700)
        symlink_root.mkdir(mode=0o700)
        lock_target = symlink_case / "lock-target"
        lock_target.write_bytes(b"must-not-be-truncated\n")
        lock_link = symlink_root / "action-n234-formal-r4-controller.lock"
        lock_link.symlink_to(lock_target)
        rejected_link = subprocess.run(
            ["/bin/bash", str(symlink_wrapper)],
            env=symlink_env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected_link.returncode != 0
        assert lock_target.read_bytes() == b"must-not-be-truncated\n"
        assert not symlink_marker.exists()

        hardlink_case = base / "hardlink"
        hardlink_anchor = hardlink_case / "anchor"
        hardlink_root = hardlink_anchor / "state" / "workspace"
        hardlink_wrapper, hardlink_marker, _target, hardlink_env = (
            _make_wrapper_fixture(hardlink_case, hardlink_anchor, hardlink_root)
        )
        hardlink_anchor.mkdir(mode=0o755)
        os.chmod(hardlink_anchor, 0o755)
        hardlink_state = hardlink_anchor / "state"
        hardlink_state.mkdir(mode=0o700)
        hardlink_root.mkdir(mode=0o700)
        hardlink_target = hardlink_case / "lock-target"
        hardlink_target.write_bytes(b"must-remain-single-payload\n")
        os.chmod(hardlink_target, 0o600)
        hardlink_path = hardlink_root / "action-n234-formal-r4-controller.lock"
        os.link(hardlink_target, hardlink_path)
        rejected_hardlink = subprocess.run(
            ["/bin/bash", str(hardlink_wrapper)],
            env=hardlink_env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected_hardlink.returncode != 0
        assert "unsafe R4 controller lock file" in rejected_hardlink.stderr
        assert hardlink_target.read_bytes() == b"must-remain-single-payload\n"
        assert os.lstat(hardlink_target).st_nlink == 2
        assert not hardlink_marker.exists()

        replacement_case = base / "ancestor-replacement"
        replacement_anchor = replacement_case / "anchor"
        replacement_root = replacement_anchor / "state" / "workspace"
        replacement_wrapper, replacement_marker, _target, replacement_env = (
            _make_wrapper_fixture(
                replacement_case,
                replacement_anchor,
                replacement_root,
                pause_before_revalidation=True,
            )
        )
        replacement_anchor.mkdir(mode=0o755)
        os.chmod(replacement_anchor, 0o755)
        process = subprocess.Popen(
            ["/bin/bash", str(replacement_wrapper)],
            env=replacement_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        ready = Path(replacement_env["FASTWAM_LOCK_TEST_READY"])
        deadline = time.monotonic() + 10
        while not ready.exists():
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError(
                    f"lock bootstrap exited before race injection: {stdout} {stderr}"
                )
            if time.monotonic() >= deadline:
                process.kill()
                process.communicate()
                raise AssertionError("lock bootstrap did not reach revalidation point")
            time.sleep(0.01)

        opened_state = replacement_anchor / "state-opened-by-wrapper"
        (replacement_anchor / "state").rename(opened_state)
        new_state = replacement_anchor / "state"
        new_state.mkdir(mode=0o700)
        (new_state / "workspace").mkdir(mode=0o700)
        Path(replacement_env["FASTWAM_LOCK_TEST_RELEASE"]).write_bytes(b"release\n")
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise AssertionError("lock bootstrap hung after ancestor replacement")
        assert process.returncode != 0, stdout
        assert "unsafe controller lock directory component" in stderr
        assert not replacement_marker.exists()

        concurrent_anchor = base / "concurrent-anchor"
        concurrent_root = concurrent_anchor / "state" / "workspace"
        holder_case = base / "concurrent-holder"
        holder_wrapper, holder_marker, _target, holder_env = _make_wrapper_fixture(
            holder_case,
            concurrent_anchor,
            concurrent_root,
            pause_before_revalidation=True,
        )
        waiter_case = base / "concurrent-waiter"
        waiter_wrapper, waiter_marker, _target, waiter_env = _make_wrapper_fixture(
            waiter_case, concurrent_anchor, concurrent_root
        )
        concurrent_anchor.mkdir(mode=0o755)
        os.chmod(concurrent_anchor, 0o755)
        holder = subprocess.Popen(
            ["/bin/bash", str(holder_wrapper)],
            env=holder_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        holder_ready = Path(holder_env["FASTWAM_LOCK_TEST_READY"])
        deadline = time.monotonic() + 10
        while not holder_ready.exists():
            if holder.poll() is not None:
                stdout, stderr = holder.communicate()
                raise AssertionError(
                    f"lock holder exited before contention: {stdout} {stderr}"
                )
            if time.monotonic() >= deadline:
                holder.kill()
                holder.communicate()
                raise AssertionError("lock holder did not reach contention point")
            time.sleep(0.01)
        rejected_waiter = subprocess.run(
            ["/bin/bash", str(waiter_wrapper)],
            env=waiter_env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected_waiter.returncode != 0
        assert "another formal R4 controller is active" in rejected_waiter.stderr
        assert not waiter_marker.exists()
        Path(holder_env["FASTWAM_LOCK_TEST_RELEASE"]).write_bytes(b"release\n")
        try:
            holder_stdout, holder_stderr = holder.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.communicate()
            raise AssertionError("lock holder hung after contention release")
        assert holder.returncode == 0, f"{holder_stdout} {holder_stderr}"
        assert holder_marker.read_bytes() == b"controller-ran\n"


def main() -> None:
    module = load_controller()
    assert list(module.MEMBERS) == ["n2", "n3", "n4"]
    for member in module.MEMBERS:
        assert_request(module, member)
    test_controller_structure(module)
    test_r4_controller_lock_binds_fd_to_exact_path(module)
    test_controller_exclusive_writer_fails_closed(module)
    test_downstream_submit_requires_n2_scientific_completion(module)
    test_downstream_gate_rejects_cross_record_drift_before_submission(module)
    test_n2_submit_has_no_scientific_prerequisite(module)
    test_source_inventory_cross_mount_portability(module)
    test_source_inventory_ignores_mode_and_mtime(module)
    test_source_inventory_content_difference_is_path_only(module)
    test_source_inventory_schema_rejects_float_bool_and_noncanonical(module)
    test_source_inventory_rejects_symlink_and_path_race(module)
    test_request_schema_scalar_types_and_trusted_runtime_base64(module)
    test_portable_inputs_schema_rejects_legacy_fields_paths_and_tasks(module)
    test_portable_inputs_reject_bool_float_negative_size_and_base64(module)
    test_gaussian_manifest_reversible_descriptor_contract(module)
    test_gaussian_manifest_large_raw_bounds(module)
    test_gaussian_completion_marker_semantics(module)
    test_member_inputs_cross_mount_mode_and_mtime_portability(module)
    test_same_size_stats_and_gaussian_control_replacements_are_detected(module)
    test_old_reservation_and_prepare_live_collector_contract(module)
    test_prepare_one_is_pure_and_publish_is_explicit(module)
    test_prepare_member_failures_leave_no_durable_or_local_state(module)
    test_prepare_phase_two_follows_all_pure_results(module)
    test_live_validation_rejects_all_same_size_control_changes(module)
    test_suite_rejects_common_input_mismatch(module)
    test_first_frozen_controller_import_does_not_mutate_source()
    test_runtime_structure()
    test_runtime_durable_writers_fail_closed()
    test_runtime_staged_copy_uses_prepared_inventory_counterexample()
    test_wrapper_uses_pinned_nofollow_lock_bootstrap(module)
    print("PASS: formal N=2/3/4 full-weight three-world launcher contract")


if __name__ == "__main__":
    main()
