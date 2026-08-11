#!/usr/bin/env python3
"""Network-free structural tests for the formal N=2/3/4 launcher."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
import sys
import tempfile
import stat
from types import SimpleNamespace
from pathlib import Path


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
    assert module.SUITE_OSS_BUDGET_BYTES == 190 * 1024**3
    assert module.PER_RUN_OSS_BUDGET_BYTES == 62 * 1024**3
    assert str(module.PINNED_PYTHON).endswith(
        "/venvs/fastwam-gaudp-py310-20260802/bin/python"
    )
    assert str(module.PINNED_PYTHON_TARGET).endswith(
        "/runtimes/uv-python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"
    )
    first = SimpleNamespace(st_mode=0o100600, st_size=17, st_mtime_ns=23,
                            st_dev=1, st_ino=2)
    remounted = SimpleNamespace(st_mode=0o100600, st_size=17, st_mtime_ns=23,
                                st_dev=9001, st_ino=9002)
    assert module.portable_file_stat(first) == module.portable_file_stat(remounted)
    assert set(module.portable_file_stat(first)) == {"mode", "bytes", "mtime_ns"}
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


def main() -> None:
    module = load_controller()
    assert list(module.MEMBERS) == ["n2", "n3", "n4"]
    for member in module.MEMBERS:
        assert_request(module, member)
    test_controller_structure(module)
    test_runtime_structure()
    wrapper = WRAPPER.read_text(encoding="utf-8")
    assert "FASTWAM_CONTROL_NODE=ssh970" in wrapper
    assert "FASTWAM_LOCK_FD=9" in wrapper
    assert '[[ -n "${SSH_CONNECTION:-}" ]]' in wrapper
    control_python = (
        "/mnt/workspace/tools/pai-control-py312/"
        "20260717-credentials1.0.10-dlc1.9.2-aiworkspace8.2.0/bin/python"
    )
    assert f"CONTROL_PYTHON={control_python}" in wrapper
    assert 'realpath -e -- "${CONTROL_PYTHON}"' in wrapper
    assert "import alibabacloud_credentials,alibabacloud_pai_dlc20201203" in wrapper
    assert "export PYTHONDONTWRITEBYTECODE=1" in wrapper
    assert 'exec "${CONTROL_PYTHON}" -B -I "${SCRIPT_DIR}/controller.py" "$@"' in wrapper
    assert "/usr/bin/python3" not in wrapper
    print("PASS: formal N=2/3/4 full-weight three-world launcher contract")


if __name__ == "__main__":
    main()
