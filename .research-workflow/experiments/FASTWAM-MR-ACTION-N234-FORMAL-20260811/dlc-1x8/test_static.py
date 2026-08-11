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
from types import SimpleNamespace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent
CONTROLLER = ROOT / "controller.py"
RUNTIME = ROOT / "runtime.sh"
WRAPPER = ROOT / "submit_from_ssh970.sh"


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
        source_root=module.SOURCE_PREFIX / "source-20260811",
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
    timestamp = "2026-08-11T06:00:00Z"
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
    assert module.INPUTS_SCHEMA == "fastwam-formal-portable-input-binding-v1"
    assert module.MEMBER_RESERVATION_SCHEMA == (
        "fastwam-action-native-agents-reservation-v2"
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
    assert module.SUITE_ID == "FASTWAM-MR-ACTION-N234-FORMAL-R2-20260811"
    assert str(module.OUTPUT_PREFIX).endswith(
        "/fastwam-action-n234-formal-r2-20260811"
    )
    for spec in module.MEMBERS.values():
        assert "-R2-20260811" in spec["experiment_id"]
        assert "-r2-20260811" in spec["run_id"]
        assert spec["display_name"].endswith("-r2")
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
    assert prepare_text.index("outcomes = [") < prepare_text.index(
        "exclusive_write(SUITE_STORAGE_RESERVATION_PATH, suite_reservation)"
    )
    assert prepare_text.index("os.mkdir(OUTPUT_PREFIX, 0o700)") < prepare_text.index(
        "outcomes = ["
    )
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
    for forbidden in ("hashlib", "sha256sum", "md5sum", "blake2", "checksum"):
        assert forbidden not in text.lower()
    assert "R1-20260811" not in text
    assert "-r1-20260811" not in text

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


def _write_portable_fixture(root: Path, payload: bytes = b"portable-source") -> None:
    nested = root / "nested"
    nested.mkdir()
    (nested / "payload.bin").write_bytes(payload)
    (root / "empty").mkdir()


def test_source_inventory_cross_mount_portability(module) -> None:
    shared_memory = Path("/dev/shm")
    assert shared_memory.is_dir(), "/dev/shm is required for the cross-mount test"
    with tempfile.TemporaryDirectory(prefix="formal-r2-posix-", dir="/tmp") as left_name:
        with tempfile.TemporaryDirectory(
            prefix="formal-r2-shm-", dir=shared_memory
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
    with tempfile.TemporaryDirectory(prefix="formal-r2-metadata-") as temporary:
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
    with tempfile.TemporaryDirectory(prefix="formal-r2-content-") as temporary:
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
    with tempfile.TemporaryDirectory(prefix="formal-r2-link-") as temporary:
        root = Path(temporary)
        (root / "payload.bin").write_bytes(b"payload")
        (root / "payload-link").symlink_to("payload.bin")
        try:
            module.source_inventory(root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("source symlink must fail closed")

    with tempfile.TemporaryDirectory(prefix="formal-r2-race-") as temporary:
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


def _gaussian_manifest(cache_kind: str, *, generation: str = "aa") -> bytes:
    if cache_kind == "compact":
        height, width, selection_mode = 28, 40, "index"
    else:
        height, width, selection_mode = 56, 80, "all"
    return _json_payload(
        {
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
    )


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
        "manifest": _content_descriptor(f"{root}/manifest.json", manifest),
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
    manifest_payload = base64.b64decode(
        primary["manifest"]["content_b64"], validate=True
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
        changed_binding["gaussian_primary"]["manifest"] = _content_descriptor(
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
    changed["schema"] = "fastwam-formal-portable-input-binding-v0"
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
            "content_b64",
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
    changed["gaussian_primary"]["manifest"]["content_b64"] = "not base64"
    _assert_runtime_rejected(
        lambda: module.validate_inputs_binding(member, request, changed),
        "invalid Gaussian base64",
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
    with tempfile.TemporaryDirectory(prefix="formal-r2-input-left-", dir="/tmp") as left_name:
        with tempfile.TemporaryDirectory(
            prefix="formal-r2-input-right-", dir=shared_memory
        ) as right_name:
            with tempfile.TemporaryDirectory(
                prefix="formal-r2-input-alias-", dir="/tmp"
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
                ):
                    os.chmod(path, 0o600)
                    os.utime(path, ns=(1_600_000_000_000_000_000,) * 2)
                for path in (
                    right_paths["stats"],
                    right_paths["checkpoint"],
                    right_paths["vae"],
                    right_paths["primary_manifest"],
                    right_paths["primary_complete"],
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
    with tempfile.TemporaryDirectory(prefix="formal-r2-control-content-") as name:
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

        changed_manifest = _gaussian_manifest("compact", generation="bb")
        _same_size_replace(paths["primary_manifest"], changed_manifest)
        after_manifest = _collect_fixture_inputs(
            module, member, request, paths["gaussian_root"]
        )
        assert baseline["gaussian_primary"]["manifest"]["bytes"] == after_manifest[
            "gaussian_primary"
        ]["manifest"]["bytes"]
        assert baseline["gaussian_primary"] != after_manifest["gaussian_primary"]

        _same_size_replace(
            paths["primary_manifest"], _gaussian_manifest("compact", generation="aa")
        )
        restored = _collect_fixture_inputs(
            module, member, request, paths["gaussian_root"]
        )
        changed_complete = _gaussian_complete(
            _gaussian_manifest("compact"), legacy_manifest_field="b" * 64
        )
        assert len(changed_complete) == paths["primary_complete"].stat().st_size
        _same_size_replace(paths["primary_complete"], changed_complete)
        after_complete = _collect_fixture_inputs(
            module, member, request, paths["gaussian_root"]
        )
        assert restored["gaussian_primary"]["completion_marker"]["bytes"] == (
            after_complete["gaussian_primary"]["completion_marker"]["bytes"]
        )
        assert restored["gaussian_primary"] != after_complete["gaussian_primary"]


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
        old = copy.deepcopy(reservation)
        old["schema"] = "fastwam-action-native-agents-reservation-v1"
        _assert_runtime_rejected(
            lambda: module.validate_member_reservation_structure(member, old),
            "old reservation v1",
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


def test_prepare_one_persists_shared_collector_result(module) -> None:
    member = "n2"
    request, inputs = _valid_inputs_binding(module, member)
    source_entries = {
        "schema": module.SOURCE_INVENTORY_SCHEMA,
        "entries": [{"path": ".", "kind": "directory"}],
    }
    with tempfile.TemporaryDirectory(prefix="formal-r2-prepare-behavior-") as name:
        root = Path(name)
        reservation_destination = root / "prepared-reservation.json"
        local_state_destination = root / "state.json"
        output_destination = root / "output"
        with (
            mock.patch.object(module, "build_request", return_value=request),
            mock.patch.object(module, "validate_request", return_value=None) as validate,
            mock.patch.object(
                module, "collect_member_inputs", return_value=inputs
            ) as collect,
            mock.patch.object(module, "output_root", return_value=output_destination),
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
                module, "local_state_path", return_value=local_state_destination
            ),
            mock.patch.object(module, "exclusive_write") as write_reservation,
            mock.patch.object(module, "atomic_write") as write_state,
            mock.patch.object(module, "utc_now", return_value="2026-08-12T00:00:00Z"),
        ):
            result = module.prepare_one(
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

        validate.assert_called_once_with(member, request, live=True)
        collect.assert_called_once_with(member, request)
        write_reservation.assert_called_once()
        destination, written = write_reservation.call_args.args
        assert destination == reservation_destination
        assert set(written) == module.MEMBER_RESERVATION_KEYS
        assert written["schema"] == module.MEMBER_RESERVATION_SCHEMA
        assert written["request"] is request
        assert written["inputs"] is inputs
        assert written["source"]["inventory"] is source_entries
        assert written["prepared_at"] == "2026-08-12T00:00:00Z"
        assert written["semantics"] == module.MEMBER_RESERVATION_SEMANTICS
        assert result == {
            "member": member,
            "status": "PREPARED",
            "path": str(reservation_destination),
        }
        write_state.assert_called_once()


def test_live_validation_recollects_and_rejects_same_size_control_change(module) -> None:
    member = "n2"
    request, inputs = _valid_inputs_binding(module, member)
    reservation = _valid_member_reservation(module, member, request, inputs)
    source = reservation["source"]
    stats_payload = base64.b64decode(
        inputs["normalization_stats"]["content_b64"], validate=True
    )
    changed_payload = stats_payload.replace(b'"aa"', b'"bb"')
    assert changed_payload != stats_payload and len(changed_payload) == len(stats_payload)
    changed_inputs = copy.deepcopy(inputs)
    changed_inputs["normalization_stats"]["content_b64"] = base64.b64encode(
        changed_payload
    ).decode("ascii")
    module.validate_inputs_binding(member, request, changed_inputs)

    common_patches = (
        mock.patch.object(module, "read_json", return_value=(suite_record(module), {})),
        mock.patch.object(
            module, "validate_complete_suite_members", return_value={member: reservation}
        ),
        mock.patch.object(
            module, "validate_member_reservation_structure", return_value=request
        ),
        mock.patch.object(module, "canonical_oss_path", return_value=Path("/tmp/output")),
        mock.patch.object(module, "canonical_direct_child", return_value=Path("/tmp/source")),
        mock.patch.object(module, "source_inventory", return_value=source["inventory"]),
    )
    with common_patches[0], common_patches[1], common_patches[2], common_patches[3], common_patches[4], common_patches[5]:
        with mock.patch.object(
            module, "collect_member_inputs", return_value=inputs
        ) as collect:
            assert module.validate_reservation_live(
                member, reservation, require_output_absent=False
            ) is request
            collect.assert_called_once_with(member, request)

    common_patches = (
        mock.patch.object(module, "read_json", return_value=(suite_record(module), {})),
        mock.patch.object(
            module, "validate_complete_suite_members", return_value={member: reservation}
        ),
        mock.patch.object(
            module, "validate_member_reservation_structure", return_value=request
        ),
        mock.patch.object(module, "canonical_oss_path", return_value=Path("/tmp/output")),
        mock.patch.object(module, "canonical_direct_child", return_value=Path("/tmp/source")),
        mock.patch.object(module, "source_inventory", return_value=source["inventory"]),
    )
    with common_patches[0], common_patches[1], common_patches[2], common_patches[3], common_patches[4], common_patches[5], mock.patch.object(
        module, "collect_member_inputs", return_value=changed_inputs
    ) as collect:
        _assert_runtime_rejected(
            lambda: module.validate_reservation_live(
                member, reservation, require_output_absent=False
            ),
            "same-size stats replacement between prepare and live validation",
        )
        collect.assert_called_once_with(member, request)


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
    with tempfile.TemporaryDirectory(prefix="formal-r2-import-") as temporary:
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
    assert "R1-20260811" not in text
    assert "-r1-20260811" not in text
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


def test_runtime_staged_copy_uses_prepared_inventory_counterexample() -> None:
    text = RUNTIME.read_text(encoding="utf-8")
    marker = (
        '"${FASTWAM_PYTHON}" -B -I -S - "${SOURCE_CONTROLLER}" '
        '"${FASTWAM_MEMBER}" "${FASTWAM_PREPARED_RESERVATION_PATH}" '
        '"${FASTWAM_SOURCE_ROOT}" "${LOCAL_SOURCE}" <<\'PY\'\n'
    )
    assert text.count(marker) == 1
    stage_script = text.split(marker, 1)[1].split("\nPY\n", 1)[0]

    with tempfile.TemporaryDirectory(prefix="formal-r2-stage-binding-") as name:
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


def main() -> None:
    module = load_controller()
    assert list(module.MEMBERS) == ["n2", "n3", "n4"]
    for member in module.MEMBERS:
        assert_request(module, member)
    test_controller_structure(module)
    test_source_inventory_cross_mount_portability(module)
    test_source_inventory_ignores_mode_and_mtime(module)
    test_source_inventory_content_difference_is_path_only(module)
    test_source_inventory_schema_rejects_float_bool_and_noncanonical(module)
    test_source_inventory_rejects_symlink_and_path_race(module)
    test_request_schema_scalar_types_and_trusted_runtime_base64(module)
    test_portable_inputs_schema_rejects_legacy_fields_paths_and_tasks(module)
    test_portable_inputs_reject_bool_float_negative_size_and_base64(module)
    test_gaussian_completion_marker_semantics(module)
    test_member_inputs_cross_mount_mode_and_mtime_portability(module)
    test_same_size_stats_and_gaussian_control_replacements_are_detected(module)
    test_old_reservation_and_prepare_live_collector_contract(module)
    test_prepare_one_persists_shared_collector_result(module)
    test_live_validation_recollects_and_rejects_same_size_control_change(module)
    test_suite_rejects_common_input_mismatch(module)
    test_first_frozen_controller_import_does_not_mutate_source()
    test_runtime_structure()
    test_runtime_staged_copy_uses_prepared_inventory_counterexample()
    wrapper = WRAPPER.read_text(encoding="utf-8")
    assert "FASTWAM_CONTROL_NODE=ssh970" in wrapper
    assert "FASTWAM_LOCK_FD=9" in wrapper
    assert '[[ -n "${SSH_CONNECTION:-}" ]]' in wrapper
    control_python = (
        "/mnt/workspace/tools/pai-control-py312/"
        "20260717-credentials1.0.10-dlc1.9.2-aiworkspace8.2.0/bin/python"
    )
    assert f"CONTROL_PYTHON={control_python}" in wrapper
    assert "CONTROL_PYTHON_TARGET=/usr/local/bin/python3.12" in wrapper
    assert '[[ -L "${CONTROL_PYTHON}" && -x "${CONTROL_PYTHON}" ]]' in wrapper
    assert 'realpath -e -- "${CONTROL_PYTHON}"' in wrapper
    assert '== "${CONTROL_PYTHON_TARGET}"' in wrapper
    assert '-L "${CONTROL_PYTHON_TARGET}"' in wrapper
    assert "import alibabacloud_credentials,alibabacloud_pai_dlc20201203" in wrapper
    assert "export PYTHONDONTWRITEBYTECODE=1" in wrapper
    assert 'exec "${CONTROL_PYTHON}" -B -I "${SCRIPT_DIR}/controller.py" "$@"' in wrapper
    assert "action-n234-formal-r2-controller.lock" in wrapper
    assert "action-n234-formal-controller.lock" not in wrapper
    assert "/usr/bin/python3" not in wrapper
    print("PASS: formal N=2/3/4 full-weight three-world launcher contract")


if __name__ == "__main__":
    main()
