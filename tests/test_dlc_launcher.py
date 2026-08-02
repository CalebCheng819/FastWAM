from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_zero2.sh"
CACHE_SCRIPT = REPO_ROOT / "scripts" / "dlc_local_cache.sh"
MULTI_CACHE_SCRIPT = REPO_ROOT / "scripts" / "dlc_multi_source_cache.sh"
OFFLINE_ENV_SCRIPT = REPO_ROOT / "scripts" / "bootstrap_offline_training_env.sh"
PREFLIGHT_SCRIPT = REPO_ROOT / "scripts" / "dlc_preflight.sh"
PYTHON_ENV_SCRIPT = REPO_ROOT / "scripts" / "validate_python_environment.py"
MANIFEST_SCRIPT = REPO_ROOT / "scripts" / "build_whole_file_manifest.py"
RESERVATION_SCRIPT = REPO_ROOT / "scripts" / "reserve_dlc_run.py"
PREPARE_BUNDLE_SCRIPT = REPO_ROOT / "scripts" / "prepare_local_training_bundle.py"
ZERO_SMOKE_SCRIPT = REPO_ROOT / "scripts" / "validate_zero_checkpoint_smoke.py"
STATE_TREE_SCRIPT = REPO_ROOT / "scripts" / "state_tree_manifest.py"
ZERO_ROUNDTRIP_SCRIPT = REPO_ROOT / "scripts" / "zero2_checkpoint_roundtrip.py"
ZERO_RUNNER_SCRIPT = REPO_ROOT / "scripts" / "run_zero2_checkpoint_smoke.sh"
IMAGE_DIGEST_PROBE_SCRIPT = REPO_ROOT / "scripts" / "probe_pod_image_digest.py"
RUNTIME_PROVENANCE_SCRIPT = REPO_ROOT / "src" / "fastwam" / "runtime_provenance.py"
TOPOLOGY_ENV = (
    "WORLD_SIZE",
    "RANK",
    "NPROC_PER_NODE",
    "NNODES",
    "NODE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RUN_ID",
)


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        *TOPOLOGY_ENV,
        "FASTWAM_LAUNCHER_UNIT_TEST_ALLOW_DIRTY",
        "FASTWAM_LAUNCHER_UNIT_TEST_SKIP_ENV_PREFLIGHT",
    ):
        env.pop(name, None)
    return env


def _apply_env(env: dict[str, str], updates: dict[str, str | None]) -> None:
    for name, value in updates.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value


def _run_launcher(
    *,
    nproc: int | None = 8,
    env_updates: dict[str, str | None] | None = None,
    dry_run: bool = True,
    merge_output: bool = False,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = _base_env()
    env.update(
        {
            "FASTWAM_LAUNCH_DRY_RUN": "1" if dry_run else "0",
            "RUN_ID": "launcher-test",
            # This opt-out is intentionally named and scoped to launcher unit
            # tests. Production launches have no general environment bypass.
            "FASTWAM_LAUNCHER_UNIT_TEST_SKIP_ENV_PREFLIGHT": "1",
        }
    )
    _apply_env(env, env_updates or {})
    command = ["bash", str(TRAIN_SCRIPT)]
    if nproc is not None:
        command.append(str(nproc))
    command.extend(
        extra_args
        if extra_args is not None
        else ["task=robofactory_multi_robot_vg1_hub1_224_1e-4"]
    )
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge_output else subprocess.PIPE,
        check=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_environment(source_root: Path, manifest: Path, cache_root: Path) -> dict[str, str]:
    env = _base_env()
    env.update(
        {
            "FASTWAM_LOCAL_CACHE_SOURCE_ROOT": str(source_root),
            "FASTWAM_LOCAL_CACHE_MANIFEST": str(manifest),
            "FASTWAM_LOCAL_CACHE_ROOT": str(cache_root),
            "FASTWAM_LOCAL_CACHE_ALLOW_SHARED_FS": "1",
            "FASTWAM_LOCAL_CACHE_MIN_FREE_BYTES": "0",
            "FASTWAM_LOCAL_CACHE_WAIT_TIMEOUT": "5",
        }
    )
    return env


def _one_file_cache_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source_root = tmp_path / "cpfs-source"
    cache_root = tmp_path / "node-local"
    source_root.mkdir()
    payload = source_root / "payload.bin"
    payload.write_bytes(b"verified whole-file payload")
    manifest = tmp_path / "cache.sha256"
    manifest.write_text(f"{_sha256(payload)}  payload.bin\n", encoding="utf-8")
    return source_root, cache_root, manifest, payload


def _formal_scale_environment(*, rank: int = 0) -> dict[str, str]:
    run_id = "launcher-test"
    return {
        "WORLD_SIZE": "4",
        "RANK": str(rank),
        "NPROC_PER_NODE": "8",
        "MASTER_ADDR": "10.20.30.40",
        "MASTER_PORT": "29400",
        "RUN_ID": run_id,
        "FASTWAM_CODE_COMMIT": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "FASTWAM_DLC_PREFLIGHT": "auto",
        "FASTWAM_FORMAL_OUTPUT_DIR": f"/cpfs/user/chengjuntao/runs/{run_id}",
        "FASTWAM_LOCAL_CACHE_ENABLED": "1",
        "FASTWAM_CPFS_BUNDLE_SOURCE_ROOT": "/cpfs/user/chengjuntao",
        "FASTWAM_CPFS_BUNDLE_MANIFEST": "/cpfs/user/chengjuntao/manifests/formal-bundle.sha256",
        "FASTWAM_CPFS_BUNDLE_MANIFEST_SHA256": "1" * 64,
        "FASTWAM_TRAINING_ENV_BUNDLE_MANIFEST_SHA256": "6" * 64,
        "FASTWAM_OSS_BUNDLE_SOURCE_ROOT": "/oss-chengjuntao",
        "FASTWAM_OSS_BUNDLE_MANIFEST": "/oss-chengjuntao/manifests/formal-bundle.sha256",
        "FASTWAM_OSS_BUNDLE_MANIFEST_SHA256": "5" * 64,
        "FASTWAM_LOCAL_CACHE_ROOT": "/tmp/fastwam-whole-file-cache",
        "FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH": "checkpoints/official.pt",
        "FASTWAM_LOCAL_DATASET_RELATIVE_ROOT": "datasets/robofactory_multi_robot",
        "FASTWAM_LOCAL_STATS_RELATIVE_PATH": "datasets/robofactory_multi_robot/stats.json",
        "FASTWAM_LOCAL_TEXT_EMBEDS_RELATIVE_ROOT": "datasets/robofactory_multi_robot/text_embeds",
        "FASTWAM_LOCAL_MODEL_CACHE_RELATIVE_ROOT": "model-cache",
        "FASTWAM_LOCAL_VAE_RELATIVE_PATH": (
            "model-cache/DiffSynth-Studio/Wan-Series-Converted-Safetensors/"
            "Wan2.2_VAE.safetensors"
        ),
        "FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT": "gaussian/compact",
        "FASTWAM_LOCAL_ERDMA_RELATIVE_ROOT": "erdma/56.2-1.0.3",
        "FASTWAM_GAUSSIAN_CACHE_MANIFEST_SHA256": "2" * 64,
        "FASTWAM_GAUSSIAN_CACHE_SELECTION_SHA256": "3" * 64,
        "FASTWAM_GAUSSIAN_CACHE_SOURCE_IDENTITY_SHA256": "4" * 64,
        "FASTWAM_DLC_IMAGE_REFERENCE": (
            "pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/"
            "pj4090/chengjuntao:cjt-multirobot-benchmark"
        ),
        "FASTWAM_DLC_IMAGE_DIGEST": "sha256:" + "a" * 64,
    }


def _formal_scale_args() -> list[str]:
    return [
        "task=robofactory_multi_robot_vg1_hub1_gau1_224_1e-4",
        "+scale=robofactory_multi_robot_32gpu",
    ]


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _install_timeout_mock(fake_bin: Path, command_log: Path) -> Path:
    fake_timeout = fake_bin / "timeout"
    _write_executable(
        fake_timeout,
        "#!/usr/bin/env bash\n"
        "printf 'TIMEOUT %s\\n' \"$*\" >> \"$FASTWAM_PREFLIGHT_TEST_LOG\"\n"
        "while (($#)); do\n"
        "  case \"$1\" in\n"
        "    --foreground|--signal=*|--kill-after=*) shift ;;\n"
        "    [0-9]*s) shift; break ;;\n"
        "    *) exit 97 ;;\n"
        "  esac\n"
        "done\n"
        "exec \"$@\"\n",
    )
    return fake_timeout


_PINNED_CRITICAL_VERSIONS = {
    "torch": "2.7.1+cu128",
    "torchvision": "0.22.1+cu128",
    "accelerate": "1.12.0",
    "deepspeed": "0.18.5",
    "hydra-core": "1.3.2",
    "h5py": "3.14.0",
    "numpy": "1.26.4",
    "transformers": "4.49.0",
}


def _fake_python_site(
    tmp_path: Path,
    *,
    overrides: dict[str, str] | None = None,
    missing: set[str] | None = None,
) -> Path:
    fake_site = tmp_path / "fake-site"
    fake_site.mkdir()
    versions = {**_PINNED_CRITICAL_VERSIONS, **(overrides or {})}
    missing = missing or set()
    for name, version in versions.items():
        if name in missing:
            continue
        normalized = name.replace("-", "_")
        dist_info = fake_site / f"{normalized}-{version}.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            encoding="utf-8",
        )

    fake_pip = fake_site / "pip"
    fake_pip.mkdir()
    (fake_pip / "__init__.py").write_text("", encoding="utf-8")
    (fake_pip / "__main__.py").write_text(
        "import os\n"
        "raise SystemExit(int(os.environ.get('FASTWAM_FAKE_PIP_CHECK_STATUS', '0')))\n",
        encoding="utf-8",
    )
    return fake_site


def _run_python_environment_validator(fake_site: Path, *, pip_status: int = 0) -> subprocess.CompletedProcess[str]:
    env = _base_env()
    env.update(
        {
            "PYTHONPATH": str(fake_site),
            "FASTWAM_FAKE_PIP_CHECK_STATUS": str(pip_status),
        }
    )
    return subprocess.run(
        [
            sys.executable,
            "-S",
            str(PYTHON_ENV_SCRIPT),
            "--pyproject",
            str(REPO_ROOT / "pyproject.toml"),
            "--pip-check-timeout",
            "5",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_zero2_launcher_builds_one_32_process_world_from_aliases() -> None:
    result = _run_launcher(
        env_updates={
            "NNODES": "4",
            "NODE_RANK": "2",
            "MASTER_ADDR": "10.20.30.40",
            "MASTER_PORT": "29400",
        }
    )

    assert result.returncode == 0, result.stderr
    assert "global_world_size=32" in result.stdout
    assert "--num_machines 4" in result.stdout
    assert "--machine_rank 2" in result.stdout
    assert "--main_process_ip 10.20.30.40" in result.stdout
    assert "--main_process_port 29400" in result.stdout
    assert "--num_processes 32" in result.stdout


def test_zero2_launcher_accepts_native_pai_topology_without_positional_nproc() -> None:
    result = _run_launcher(
        nproc=None,
        env_updates={
            "WORLD_SIZE": "4",
            "RANK": "2",
            "NPROC_PER_NODE": "8",
            "MASTER_ADDR": "10.20.30.40",
            "MASTER_PORT": "29400",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "nproc_per_node=8 num_machines=4 machine_rank=2 global_world_size=32" in result.stdout
    assert "--num_processes 32" in result.stdout


def test_zero2_launcher_forces_standard_and_reuses_documented_port() -> None:
    result = _run_launcher(
        env_updates={
            "WORLD_SIZE": "4",
            "RANK": "0",
            "NPROC_PER_NODE": "8",
            "MASTER_ADDR": "10.20.30.40",
            "MASTER_PORT": "29400",
        }
    )

    assert result.returncode == 0, result.stderr
    assert "--deepspeed_multinode_launcher standard" in result.stdout
    assert "--main_process_port 29400" in result.stdout
    assert "29401" not in result.stdout
    assert "29411" not in result.stdout


def test_zero2_launcher_preserves_single_node_defaults() -> None:
    result = _run_launcher()

    assert result.returncode == 0, result.stderr
    assert "global_world_size=8" in result.stdout
    assert "--num_machines 1" in result.stdout
    assert "--machine_rank 0" in result.stdout
    assert "--main_process_ip 127.0.0.1" in result.stdout
    assert "--main_process_port 29500" in result.stdout
    assert "--num_processes 8" in result.stdout


def test_zero2_launcher_rejects_invalid_topology() -> None:
    invalid_cases = [
        ({"NNODES": "4", "NODE_RANK": "4", "MASTER_ADDR": "10.0.0.1"}, "smaller than resolved node count"),
        ({"NNODES": "4", "MASTER_ADDR": "10.0.0.1"}, "RANK or compatibility NODE_RANK is required"),
        ({"NNODES": "4", "NODE_RANK": "0"}, "MASTER_ADDR is required"),
        ({"NNODES": "4", "NODE_RANK": "0", "MASTER_ADDR": "127.0.0.1"}, "reachable by every machine"),
        ({"MASTER_PORT": "70000"}, "integer in [1, 65535]"),
        ({"FASTWAM_LAUNCH_DRY_RUN": "maybe"}, "expected a boolean value"),
    ]
    for env_updates, message in invalid_cases:
        result = _run_launcher(env_updates=env_updates)
        assert result.returncode != 0
        assert message in (result.stderr or "")


def test_zero2_launcher_rejects_native_alias_and_nproc_conflicts() -> None:
    invalid_cases = [
        (
            {"WORLD_SIZE": "4", "NNODES": "3", "RANK": "0", "MASTER_ADDR": "10.0.0.1"},
            "WORLD_SIZE (4 nodes) conflicts with NNODES (3)",
            8,
        ),
        (
            {"WORLD_SIZE": "4", "RANK": "2", "NODE_RANK": "1", "MASTER_ADDR": "10.0.0.1"},
            "RANK (2) conflicts with NODE_RANK (1)",
            8,
        ),
        (
            {"WORLD_SIZE": "4", "RANK": "0", "NPROC_PER_NODE": "8", "MASTER_ADDR": "10.0.0.1"},
            "NPROC_PER_NODE (8) conflicts with positional nproc_per_node (4)",
            4,
        ),
    ]
    for env_updates, message, nproc in invalid_cases:
        result = _run_launcher(nproc=nproc, env_updates=env_updates)
        assert result.returncode != 0
        assert message in (result.stderr or "")


def test_zero2_launcher_requires_safe_explicit_multinode_run_id() -> None:
    topology = {
        "WORLD_SIZE": "4",
        "RANK": "0",
        "NPROC_PER_NODE": "8",
        "MASTER_ADDR": "10.0.0.1",
    }
    missing = _run_launcher(nproc=None, env_updates={**topology, "RUN_ID": None})
    assert missing.returncode != 0
    assert "explicit identical RUN_ID" in (missing.stderr or "")

    for unsafe in ("../escape", "run id", "-leading-hyphen", "x" * 129):
        result = _run_launcher(nproc=None, env_updates={**topology, "RUN_ID": unsafe})
        assert result.returncode != 0
        assert "RUN_ID must be 1-128 safe characters" in (result.stderr or "")


def test_formal_32gpu_scale_is_fail_closed_even_in_dry_run() -> None:
    good = _run_launcher(
        nproc=None,
        env_updates=_formal_scale_environment(rank=2),
        extra_args=_formal_scale_args(),
    )
    assert good.returncode == 0, good.stderr
    assert "global_world_size=32" in good.stdout
    assert "output_dir=/cpfs/user/chengjuntao/runs/launcher-test" in good.stdout
    assert "wandb.name=robofactory_multi_robot_vg1_hub1_gau1_224_1e-4-launcher-test" in good.stdout
    assert "resume=/tmp/fastwam-whole-file-cache/cpfs/" + "1" * 64 + "/checkpoints/official.pt" in good.stdout
    assert "data.train.root_dir=/tmp/fastwam-whole-file-cache/cpfs/" + "1" * 64 in good.stdout
    assert "checkpoint_state_kind=full" in good.stdout

    resume_state = "/cpfs/user/chengjuntao/runs/launcher-test/checkpoints/state/step_001000"
    resumed = _run_launcher(
        nproc=None,
        env_updates={
            **_formal_scale_environment(rank=1),
            "FASTWAM_FORMAL_RESUME_STATE_DIR": resume_state,
            "FASTWAM_FORMAL_RESUME_STATE_MANIFEST": (
                "/cpfs/user/chengjuntao/runs/launcher-test/manifests/step_001000.state-tree.json"
            ),
            "FASTWAM_FORMAL_RESUME_STATE_MANIFEST_SHA256": "8" * 64,
            "FASTWAM_FORMAL_RESUME_TRAINER_STATE_SHA256": "9" * 64,
        },
        extra_args=_formal_scale_args(),
    )
    assert resumed.returncode == 0, resumed.stderr
    assert f"resume={resume_state}" in resumed.stdout

    invalid_cases = [
        ({"WORLD_SIZE": "1", "RANK": "0"}, "requires DLC WORLD_SIZE=4"),
        ({"NPROC_PER_NODE": "4"}, "requires DLC WORLD_SIZE=4"),
        ({"FASTWAM_DLC_PREFLIGHT": "0"}, "cannot disable FASTWAM_DLC_PREFLIGHT"),
        ({"FASTWAM_LOCAL_CACHE_ENABLED": "0"}, "requires FASTWAM_LOCAL_CACHE_ENABLED=1"),
        ({"FASTWAM_LOCAL_CACHE_ROOT": "/tmp/a-different-root"}, "is fixed to"),
        ({"FASTWAM_FORMAL_OUTPUT_DIR": "./runs/launcher-test"}, "formal output_dir must be under"),
        ({"FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH": "../escape.pt"}, "non-escaping relative path"),
        ({"FASTWAM_CODE_COMMIT": "0" * 40}, "exact current 40-hex HEAD"),
    ]
    base = _formal_scale_environment()
    for updates, message in invalid_cases:
        result = _run_launcher(
            nproc=None,
            env_updates={**base, **updates},
            extra_args=_formal_scale_args(),
        )
        assert result.returncode != 0, updates
        assert message in (result.stderr or ""), (updates, result.stderr)

    conflict = _run_launcher(
        nproc=None,
        env_updates=base,
        extra_args=[*_formal_scale_args(), "output_dir=/cpfs/user/chengjuntao/runs/bypass"],
    )
    assert conflict.returncode != 0
    assert "owns provenance and node-local path override 'output_dir'" in (conflict.stderr or "")


def test_formal_32gpu_cli_allowlist_seals_treatment_and_schedule() -> None:
    base = _formal_scale_environment()
    forbidden = [
        "model.action_dit_config.hub_enabled=false",
        "model.action_dit_config.enable_gaussian=false",
        "model.training_mode=action_only_cache",
        "model.loss.lambda_video=0.0",
        "trainable_scope=action",
        "data.train.required_agent_counts=[2]",
        "data.val.required_agent_counts=[4]",
        "gradient_accumulation_steps=8",
        "max_steps=1",
        "eval_every=0",
        "offline_eval_num_samples=0",
        "save_every=1",
        "seed=43",
        "++model.action_dit_config.hub_enabled=false",
        "~model.action_dit_config.enable_gaussian",
        "--multirun",
    ]
    for override in forbidden:
        result = _run_launcher(
            nproc=None,
            env_updates=base,
            extra_args=[*_formal_scale_args(), override],
        )
        assert result.returncode != 0, override
        assert "formal 32-GPU CLI allowlist" in (result.stderr or ""), (
            override,
            result.stderr,
        )

    launcher_owned = [
        "checkpoint_state_kind=sparse_delta",
        "resume=/tmp/bypass.pt",
        "data.train.root_dir=/tmp/bypass",
    ]
    for override in launcher_owned:
        result = _run_launcher(
            nproc=None,
            env_updates=base,
            extra_args=[*_formal_scale_args(), override],
        )
        assert result.returncode != 0, override
        assert "owns provenance and node-local path override" in (result.stderr or "")

    task = _formal_scale_args()[0]
    duplicate_task = _run_launcher(
        nproc=None,
        env_updates=base,
        extra_args=[*_formal_scale_args(), task],
    )
    assert duplicate_task.returncode != 0
    assert "exactly one task selector" in (duplicate_task.stderr or "")

    duplicate_scale = _run_launcher(
        nproc=None,
        env_updates=base,
        extra_args=[*_formal_scale_args(), "+scale=robofactory_multi_robot_32gpu"],
    )
    assert duplicate_scale.returncode != 0
    assert "exactly one task selector" in (duplicate_scale.stderr or "")

    legacy_task = _run_launcher(
        nproc=None,
        env_updates=base,
        extra_args=[
            "task=robofactory_multi_robot_vg1_hub1_224_1e-4",
            "+scale=robofactory_multi_robot_32gpu",
        ],
    )
    assert legacy_task.returncode != 0
    assert "eight explicit" in (legacy_task.stderr or "")

    allowed = _run_launcher(
        nproc=None,
        env_updates=base,
        extra_args=_formal_scale_args(),
    )
    assert allowed.returncode == 0, allowed.stderr
    # These are launcher-owned, appended after the already validated user
    # selectors, and therefore are not mistaken for user customization.
    assert "checkpoint_state_kind=full" in allowed.stdout
    assert "output_dir=/cpfs/user/chengjuntao/runs/launcher-test" in allowed.stdout
    assert "resume=/tmp/fastwam-whole-file-cache/" in allowed.stdout


def test_formal_non_dry_rejects_test_bypasses_before_git_or_python() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        current_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        fake_git = fake_bin / "git"
        fake_python = fake_bin / "python-fastwam"
        _write_executable(
            fake_git,
            "#!/usr/bin/env bash\n"
            "printf 'GIT %s\\n' \"$*\" >> \"$FASTWAM_TEST_GIT_LOG\"\n"
            "exit 97\n",
        )
        _write_executable(
            fake_python,
            "#!/usr/bin/env bash\n"
            "printf 'PYTHON %s\\n' \"$*\" >> \"$FASTWAM_TEST_PYTHON_LOG\"\n"
            "exit 98\n",
        )

        for bypass_name in (
            "FASTWAM_LAUNCHER_UNIT_TEST_ALLOW_DIRTY",
            "FASTWAM_LAUNCHER_UNIT_TEST_SKIP_ENV_PREFLIGHT",
        ):
            case_name = bypass_name.removeprefix("FASTWAM_LAUNCHER_UNIT_TEST_").lower()
            git_log = tmp_path / f"{case_name}.git.log"
            python_log = tmp_path / f"{case_name}.python.log"
            result = _run_launcher(
                nproc=None,
                env_updates={
                    **_formal_scale_environment(),
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "FASTWAM_PYTHON": str(fake_python),
                    "FASTWAM_TEST_GIT_LOG": str(git_log),
                    "FASTWAM_TEST_PYTHON_LOG": str(python_log),
                    "FASTWAM_TEST_HEAD": current_head,
                    "FASTWAM_LAUNCHER_UNIT_TEST_ALLOW_DIRTY": "0",
                    "FASTWAM_LAUNCHER_UNIT_TEST_SKIP_ENV_PREFLIGHT": "0",
                    bypass_name: "1",
                },
                dry_run=False,
                extra_args=_formal_scale_args(),
            )

            assert result.returncode != 0
            assert f"forbids unit-test bypass {bypass_name}" in (result.stderr or "")
            # The formal bypass gate is earlier than Git identity/cleanliness,
            # reservation validation, and the exact Python/pip-check preflight.
            assert not git_log.exists()
            assert not python_log.exists()

        # Unit-test bypasses remain usable only for an explicit parameter-only
        # dry-run, which cannot reserve an output or execute Accelerate.
        dry_run = _run_launcher(
            nproc=None,
            env_updates={
                **_formal_scale_environment(),
                "FASTWAM_LAUNCHER_UNIT_TEST_ALLOW_DIRTY": "1",
                "FASTWAM_LAUNCHER_UNIT_TEST_SKIP_ENV_PREFLIGHT": "1",
            },
            dry_run=True,
            extra_args=_formal_scale_args(),
        )
        assert dry_run.returncode == 0, dry_run.stderr
        assert "[dry_run]" in dry_run.stdout


def test_formal_non_dry_requires_clean_checkout_and_exact_env_preflight() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        current_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        fake_git = fake_bin / "git"
        fake_python = fake_bin / "python-fastwam"
        _write_executable(
            fake_git,
            "#!/usr/bin/env bash\n"
            "printf 'GIT %s\\n' \"$*\" >> \"$FASTWAM_TEST_GIT_LOG\"\n"
            "if [[ \"${3-}\" == rev-parse && \"${4-}\" == --verify && \"${5-}\" == HEAD ]]; then\n"
            "  printf '%s\\n' \"$FASTWAM_TEST_HEAD\"\n"
            "  exit 0\n"
            "fi\n"
            "if [[ \"${3-}\" == status ]]; then\n"
            "  [[ \"${FASTWAM_TEST_GIT_DIRTY:-0}\" == 1 ]] && printf ' M tracked-file\\n'\n"
            "  exit 0\n"
            "fi\n"
            "exit 97\n",
        )
        _write_executable(
            fake_python,
            "#!/usr/bin/env bash\n"
            "printf 'PYTHON %s\\n' \"$*\" >> \"$FASTWAM_TEST_PYTHON_LOG\"\n"
            "case \"${1-}\" in\n"
            "  */reserve_dlc_run.py)\n"
            "    [[ \"${2-}\" == --mode && \"${3-}\" == validate ]] || exit 96\n"
            "    exit 0\n"
            "    ;;\n"
            "  */validate_python_environment.py) exit 61 ;;\n"
            "  *) exit 98 ;;\n"
            "esac\n",
        )
        common = {
            **_formal_scale_environment(),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FASTWAM_PYTHON": str(fake_python),
            "FASTWAM_TEST_HEAD": current_head,
            "FASTWAM_LAUNCHER_UNIT_TEST_ALLOW_DIRTY": "0",
            "FASTWAM_LAUNCHER_UNIT_TEST_SKIP_ENV_PREFLIGHT": "0",
        }

        dirty_git_log = tmp_path / "dirty.git.log"
        dirty_python_log = tmp_path / "dirty.python.log"
        dirty = _run_launcher(
            nproc=None,
            env_updates={
                **common,
                "FASTWAM_TEST_GIT_LOG": str(dirty_git_log),
                "FASTWAM_TEST_PYTHON_LOG": str(dirty_python_log),
                "FASTWAM_TEST_GIT_DIRTY": "1",
            },
            dry_run=False,
            extra_args=_formal_scale_args(),
        )
        assert dirty.returncode != 0
        assert "requires a clean immutable Git worktree" in (dirty.stderr or "")
        assert "status --porcelain --untracked-files=all" in dirty_git_log.read_text(
            encoding="utf-8"
        )
        assert not dirty_python_log.exists()

        clean_git_log = tmp_path / "clean.git.log"
        clean_python_log = tmp_path / "clean.python.log"
        clean = _run_launcher(
            nproc=None,
            env_updates={
                **common,
                "FASTWAM_TEST_GIT_LOG": str(clean_git_log),
                "FASTWAM_TEST_PYTHON_LOG": str(clean_python_log),
                "FASTWAM_TEST_GIT_DIRTY": "0",
            },
            dry_run=False,
            extra_args=_formal_scale_args(),
        )
        assert clean.returncode == 61
        assert "Python environment preflight failed with status=61" in (clean.stderr or "")
        python_calls = clean_python_log.read_text(encoding="utf-8").splitlines()
        assert len(python_calls) == 2, python_calls
        assert "reserve_dlc_run.py --mode validate" in python_calls[0]
        assert "--timeout 300 --resume-timeout 21600" in python_calls[0]
        assert "--training-env-bundle-manifest-sha256 " + "6" * 64 in python_calls[0]
        assert "validate_python_environment.py --pyproject" in python_calls[1]
        assert all("--mode owner" not in call for call in python_calls)


def test_formal_image_digest_is_mandatory_for_execution_but_ack_is_dry_run_only() -> None:
    unresolved = {
        **_formal_scale_environment(),
        "FASTWAM_DLC_IMAGE_DIGEST": None,
        "FASTWAM_ACK_MUTABLE_IMAGE_TAG_RISK": "1",
        "FASTWAM_LAUNCHER_UNIT_TEST_ALLOW_DIRTY": "0",
        "FASTWAM_LAUNCHER_UNIT_TEST_SKIP_ENV_PREFLIGHT": "0",
    }
    refused = _run_launcher(
        nproc=None,
        env_updates=unresolved,
        dry_run=False,
        extra_args=_formal_scale_args(),
    )
    assert refused.returncode != 0
    assert "formal non-dry-run launch requires FASTWAM_DLC_IMAGE_DIGEST" in (
        refused.stderr or ""
    )

    diagnostic = _run_launcher(
        nproc=None,
        env_updates=unresolved,
        dry_run=True,
        extra_args=_formal_scale_args(),
    )
    assert diagnostic.returncode == 0, diagnostic.stderr
    assert "warning=mutable_image_tag" in (diagnostic.stderr or "")


def test_formal_gau0_has_no_gaussian_oss_asset_dependency() -> None:
    gaussian_names = (
        "FASTWAM_OSS_BUNDLE_SOURCE_ROOT",
        "FASTWAM_OSS_BUNDLE_MANIFEST",
        "FASTWAM_OSS_BUNDLE_MANIFEST_SHA256",
        "FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT",
        "FASTWAM_GAUSSIAN_CACHE_DIR",
        "FASTWAM_GAUSSIAN_CACHE_MANIFEST_SHA256",
        "FASTWAM_GAUSSIAN_CACHE_SELECTION_SHA256",
        "FASTWAM_GAUSSIAN_CACHE_SOURCE_IDENTITY_SHA256",
    )
    gau0_env = _formal_scale_environment()
    gau0_env.update({name: None for name in gaussian_names})
    gau0_args = [
        "task=robofactory_multi_robot_vg1_hub1_gau0_224_1e-4",
        "+scale=robofactory_multi_robot_32gpu",
    ]
    passed = _run_launcher(
        nproc=None,
        env_updates=gau0_env,
        dry_run=True,
        extra_args=gau0_args,
    )
    assert passed.returncode == 0, passed.stderr
    assert "gaussian_cache_dir=" not in passed.stdout

    leaked = _run_launcher(
        nproc=None,
        env_updates=_formal_scale_environment(),
        dry_run=True,
        extra_args=gau0_args,
    )
    assert leaked.returncode != 0
    assert "GAU0 formal arms forbid irrelevant Gaussian OSS input" in (
        leaked.stderr or ""
    )

    stale_runtime_mapping = _run_launcher(
        nproc=None,
        env_updates={
            **gau0_env,
            "FASTWAM_GAUSSIAN_CACHE_DIR": "/tmp/stale-gaussian-cache",
        },
        dry_run=True,
        extra_args=gau0_args,
    )
    assert stale_runtime_mapping.returncode != 0
    assert "FASTWAM_GAUSSIAN_CACHE_DIR" in (stale_runtime_mapping.stderr or "")

    gau1_missing = _run_launcher(
        nproc=None,
        env_updates={
            **_formal_scale_environment(),
            "FASTWAM_OSS_BUNDLE_SOURCE_ROOT": None,
            "FASTWAM_OSS_BUNDLE_MANIFEST": None,
            "FASTWAM_OSS_BUNDLE_MANIFEST_SHA256": None,
        },
        dry_run=True,
        extra_args=_formal_scale_args(),
    )
    assert gau1_missing.returncode != 0
    assert "Gaussian compact bundle root and manifest must be on OSS" in (
        gau1_missing.stderr or ""
    )


def test_zero2_launcher_execs_accelerate_and_propagates_status() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        accelerate_log = tmp_path / "accelerate.log"
        fake_python = fake_bin / "python-fastwam"
        _write_executable(
            fake_python,
            "#!/usr/bin/env bash\n"
            "[[ \"$1\" == -m && \"$2\" == accelerate.commands.launch ]] || exit 96\n"
            "shift 2\n"
            "printf 'ARGS=%s\\n' \"$*\" > \"$FASTWAM_ACCELERATE_TEST_LOG\"\n"
            "printf 'WORLD_SIZE=%s\\n' \"${WORLD_SIZE-UNSET}\" >> \"$FASTWAM_ACCELERATE_TEST_LOG\"\n"
            "printf 'RANK=%s\\n' \"${RANK-UNSET}\" >> \"$FASTWAM_ACCELERATE_TEST_LOG\"\n"
            "exit 37\n",
        )
        env = {
            "WORLD_SIZE": "4",
            "RANK": "2",
            "NPROC_PER_NODE": "8",
            "MASTER_ADDR": "10.20.30.40",
            "MASTER_PORT": "29400",
            "FASTWAM_DLC_PREFLIGHT": "0",
            "FASTWAM_LOCAL_CACHE_ENABLED": "0",
            "FASTWAM_ACCELERATE_TEST_LOG": str(accelerate_log),
            "FASTWAM_PYTHON": str(fake_python),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        }
        result = _run_launcher(nproc=None, env_updates=env, dry_run=False)

        assert result.returncode == 37
        log = accelerate_log.read_text(encoding="utf-8")
        assert "--num_machines 4" in log
        assert "--machine_rank 2" in log
        assert "--num_processes 32" in log
        assert "--deepspeed_multinode_launcher standard" in log
        assert "WORLD_SIZE=UNSET" in log
        assert "RANK=UNSET" in log
        assert "reason=launcher_unit_test_override" in (result.stderr or "")


def test_python_environment_preflight_accepts_exact_pyproject_versions() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fake_site = _fake_python_site(Path(directory))
        result = _run_python_environment_validator(fake_site)

        assert result.returncode == 0, result.stderr
        assert "package=torch expected=2.7.1+cu128 actual=2.7.1+cu128 status=PASS" in result.stdout
        assert "package=accelerate expected=1.12.0 actual=1.12.0 status=PASS" in result.stdout
        assert "pip_check=PASS" in result.stdout
        assert "status=PASS critical_packages=8" in result.stdout


def test_python_environment_preflight_rejects_version_drift() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fake_site = _fake_python_site(
            Path(directory),
            overrides={"torch": "2.10.0", "accelerate": "1.14.0", "deepspeed": "0.18.9"},
        )
        result = _run_python_environment_validator(fake_site)

        assert result.returncode == 1
        assert "package=torch expected=2.7.1+cu128 actual=2.10.0 status=MISMATCH" in result.stderr
        assert "package=accelerate expected=1.12.0 actual=1.14.0 status=MISMATCH" in result.stderr
        assert "package=deepspeed expected=0.18.5 actual=0.18.9 status=MISMATCH" in result.stderr
        assert "failed closed" in result.stderr


def test_python_environment_preflight_rejects_missing_critical_package() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fake_site = _fake_python_site(Path(directory), missing={"transformers"})
        result = _run_python_environment_validator(fake_site)

        assert result.returncode == 1
        assert "package=transformers expected=4.49.0 actual=MISSING status=MISSING" in result.stderr
        assert "failed closed" in result.stderr


def test_python_environment_preflight_propagates_pip_check_failure() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fake_site = _fake_python_site(Path(directory))
        result = _run_python_environment_validator(fake_site, pip_status=23)

        assert result.returncode == 1
        assert "pip check failed with status=23" in result.stderr
        assert "failed closed" in result.stderr


def test_whole_file_manifest_generator_is_deterministic_atomic_and_safe() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        source = tmp_path / "source"
        output_dir = tmp_path / "manifests"
        (source / "z").mkdir(parents=True)
        (source / "a dir").mkdir()
        output_dir.mkdir()
        first = source / "z" / "second.bin"
        second = source / "a dir" / "first.bin"
        first.write_bytes(b"second")
        second.write_bytes(b"first")
        manifest = output_dir / "bundle.sha256"
        command = [
            sys.executable,
            str(MANIFEST_SCRIPT),
            "--source-root",
            str(source),
            "--include",
            "z",
            "--include",
            "a dir/first.bin",
            "--include",
            "z/second.bin",
            "--output",
            str(manifest),
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        summary = json.loads(result.stdout)
        expected = (
            f"{_sha256(second)}  a dir/first.bin\n"
            f"{_sha256(first)}  z/second.bin\n"
        )
        assert manifest.read_text(encoding="utf-8") == expected
        assert summary["file_count"] == 2
        assert summary["manifest_sha256"] == hashlib.sha256(expected.encode()).hexdigest()

        no_clobber = subprocess.run(command, text=True, capture_output=True, check=False)
        assert no_clobber.returncode != 0
        assert "refusing to replace existing manifest" in no_clobber.stderr

        symlink = source / "bad-link"
        symlink.symlink_to(first)
        symlink_result = subprocess.run(
            [
                sys.executable,
                str(MANIFEST_SCRIPT),
                "--source-root",
                str(source),
                "--include",
                "bad-link",
                "--output",
                str(output_dir / "symlink.sha256"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert symlink_result.returncode != 0
        assert "symlinks are forbidden" in symlink_result.stderr

        inside = subprocess.run(
            [
                sys.executable,
                str(MANIFEST_SCRIPT),
                "--source-root",
                str(source),
                "--include",
                "z",
                "--output",
                str(source / "manifest.sha256"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert inside.returncode != 0
        assert "outside the source root" in inside.stderr


def test_run_reservation_is_exclusive_and_identity_exact() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        allowed = tmp_path / "cpfs"
        source = tmp_path / "source"
        parent = allowed / "runs"
        parent.mkdir(parents=True)
        source.mkdir()
        output = parent / "formal-run"
        base = [
            sys.executable,
            str(RESERVATION_SCRIPT),
            "--output-dir",
            str(output),
            "--allowed-prefix",
            str(allowed),
            "--source-root",
            str(source),
            "--run-id",
            "formal-run",
            "--code-commit",
            "a" * 40,
            "--task",
            "robofactory_multi_robot_vg1_hub1_gau1_224_1e-4",
            "--num-machines",
            "4",
            "--nproc-per-node",
            "8",
            "--expected-global-world-size",
            "32",
            "--bundle-manifest-sha256",
            "b" * 64,
            "--cache-manifest-sha256",
            "c" * 64,
            "--cache-selection-sha256",
            "d" * 64,
            "--cache-source-identity-sha256",
            "e" * 64,
            "--checkpoint-sha256",
            "f" * 64,
            "--vae-sha256",
            "1" * 64,
            "--image-reference",
            "registry.example/fastwam:mutable-tag",
            "--image-digest-status",
            "unresolved_mutable_tag",
            "--pyproject-sha256",
            "2" * 64,
            "--output-storage",
            "cpfs",
        ]
        owner = subprocess.run([*base, "--mode", "owner"], text=True, capture_output=True, check=False)
        assert owner.returncode == 0, owner.stderr
        marker = output / ".RUN_RESERVED"
        identity = json.loads(marker.read_text(encoding="utf-8"))
        assert identity["global_world_size"] == 32
        assert identity["run_id"] == "formal-run"
        assert identity["bundle_manifest_sha256"] == "b" * 64
        assert identity["image_digest_status"] == "unresolved_mutable_tag"
        assert identity["image_digest"] is None

        waiter = subprocess.run([*base, "--mode", "wait", "--timeout", "1"], text=True, capture_output=True, check=False)
        assert waiter.returncode == 0, waiter.stderr

        state_dir = output / "checkpoints" / "state" / "step_000100"
        state_dir.mkdir(parents=True)
        trainer_state = state_dir / "trainer_state.json"
        trainer_state.write_text(
            json.dumps({"run_contract": {"contract_version": 1}}) + "\n",
            encoding="utf-8",
        )
        optimizer_state = state_dir / "optimizer.bin"
        optimizer_state.write_bytes(b"zero2 optimizer shard")
        manifest = output / "manifests" / "step_000100.state-tree.json"
        manifest.parent.mkdir()
        built_manifest = subprocess.run(
            [
                sys.executable,
                str(STATE_TREE_SCRIPT),
                "build",
                "--state-root",
                str(state_dir),
                "--output",
                str(manifest),
                "--role",
                "accelerate_zero2_full_state",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert built_manifest.returncode == 0, built_manifest.stderr
        resume_identity_args = [
            "--resume-state-dir",
            str(state_dir),
            "--resume-state-manifest",
            str(manifest),
            "--resume-state-manifest-sha256",
            _sha256(manifest),
            "--resume-trainer-state-sha256",
            _sha256(trainer_state),
        ]
        resume = subprocess.run(
            [
                *base,
                "--mode",
                "validate-existing",
                *resume_identity_args,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert resume.returncode == 0, resume.stderr
        resume_waiter = subprocess.run(
            [*base, "--mode", "wait-existing", "--timeout", "1", *resume_identity_args],
            text=True,
            capture_output=True,
            check=False,
        )
        assert resume_waiter.returncode == 0, resume_waiter.stderr

        optimizer_state.write_bytes(b"same-shape wrong optimizer shard")
        corrupt_tree = subprocess.run(
            [*base, "--mode", "validate-existing", *resume_identity_args],
            text=True,
            capture_output=True,
            check=False,
        )
        assert corrupt_tree.returncode != 0
        assert "state-tree file" in corrupt_tree.stderr
        optimizer_state.write_bytes(b"zero2 optimizer shard")

        wrong_resume_hash = subprocess.run(
            [
                *base,
                "--mode",
                "validate-existing",
                *resume_identity_args[:-1],
                "9" * 64,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert wrong_resume_hash.returncode != 0
        assert "trainer_state.json SHA-256 mismatch" in wrong_resume_hash.stderr

        reused = subprocess.run([*base, "--mode", "owner"], text=True, capture_output=True, check=False)
        assert reused.returncode != 0
        assert "never reused" in reused.stderr

        changed_task = list(base)
        changed_task[changed_task.index("--task") + 1] = "different-task"
        mismatch = subprocess.run(
            [*changed_task, "--mode", "wait", "--timeout", "1"],
            text=True,
            capture_output=True,
            check=False,
        )
        assert mismatch.returncode != 0
        assert "reservation identity mismatch" in mismatch.stderr

        source_output = source / "formal-run"
        inside_source = list(base)
        inside_source[inside_source.index("--output-dir") + 1] = str(source_output)
        inside_source[inside_source.index("--allowed-prefix") + 1] = str(tmp_path)
        refused = subprocess.run(
            [*inside_source, "--mode", "validate"], text=True, capture_output=True, check=False
        )
        assert refused.returncode != 0
        assert "must not be inside source tree" in refused.stderr


def test_zero_checkpoint_smoke_evidence_is_hash_and_filesystem_bound() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        smoke_root = root / "smoke-output"
        output_parent = root / "formal-output"
        smoke_root.mkdir()
        output_parent.mkdir()
        state_root = smoke_root / "zero2-state"
        proof_root = state_root / "smoke-proof"
        proof_root.mkdir(parents=True)
        batch_accounting = {
            "global_train_batch_size": 128,
            "gradient_accumulation_steps": 1,
            "local_micro_batch_size": 4,
            "world_size": 32,
        }
        for rank in range(32):
            for prefix in ("save", "mutated", "load"):
                (proof_root / f"{prefix}-rank-{rank:05d}.json").write_text(
                    json.dumps(
                        {
                            "batch_accounting": batch_accounting,
                            "rank": rank,
                            "phase": prefix,
                            "schema_version": 2,
                            "world_size": 32,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
        (state_root / "optimizer-shard.bin").write_bytes(b"real state fixture")
        state_manifest = smoke_root / "zero2-state-tree.json"
        built = subprocess.run(
            [
                sys.executable,
                str(STATE_TREE_SCRIPT),
                "build",
                "--state-root",
                str(state_root),
                "--output",
                str(state_manifest),
                "--role",
                "zero2_roundtrip_smoke_state",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert built.returncode == 0, built.stderr
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        package_versions = {
            "torch": "2.7.1+cu128",
            "accelerate": "1.12.0",
            "deepspeed": "0.18.5",
        }
        image_reference = "registry.example/fastwam@sha256:test"
        image_digest = "sha256:" + "7" * 64
        payload = {
            "batch_accounting": batch_accounting,
            "code_commit": commit,
            "filesystem_device": os.stat(smoke_root).st_dev,
            "image_digest": image_digest,
            "image_reference": image_reference,
            "output_root": str(smoke_root.resolve()),
            "package_versions": package_versions,
            "pyproject_sha256": _sha256(REPO_ROOT / "pyproject.toml"),
            "roundtrip": {
                "global_step": True,
                "model": True,
                "optimizer": True,
                "rng": True,
                "rng_next_sample": True,
                "scheduler": True,
                "separate_process": True,
            },
            "schema_version": 2,
            "state_tree_manifest": str(state_manifest.resolve()),
            "state_tree_manifest_sha256": _sha256(state_manifest),
            "state_tree_root": str(state_root.resolve()),
            "status": "PASS",
            "zero_stage": 2,
            "world_size": 32,
        }
        marker = smoke_root / "zero-smoke.json"
        marker.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(ZERO_SMOKE_SCRIPT),
            "--marker",
            str(marker),
            "--expected-sha256",
            _sha256(marker),
            "--output-parent",
            str(output_parent),
        ]
        env = _base_env()
        env.update(
            {
                "FASTWAM_CODE_COMMIT": commit,
                "FASTWAM_DLC_IMAGE_REFERENCE": image_reference,
                "FASTWAM_DLC_IMAGE_DIGEST": image_digest,
                "FASTWAM_ZERO_SMOKE_UNIT_TEST_ALLOW_DIRTY": "1",
                "FASTWAM_ZERO_SMOKE_UNIT_TEST_PACKAGE_VERSIONS": json.dumps(
                    package_versions, sort_keys=True
                ),
            }
        )
        passed = subprocess.run(
            command, text=True, capture_output=True, check=False, env=env
        )
        assert passed.returncode == 0, passed.stderr
        validated = json.loads(passed.stdout)
        assert validated["status"] == "PASS"
        assert validated["batch_accounting"] == batch_accounting

        bad_proof = proof_root / "load-rank-00031.json"
        bad_payload = json.loads(bad_proof.read_text(encoding="utf-8"))
        bad_payload["batch_accounting"] = {
            **batch_accounting,
            "gradient_accumulation_steps": 2,
        }
        bad_proof.write_text(json.dumps(bad_payload) + "\n", encoding="utf-8")
        state_manifest.unlink()
        rebuilt = subprocess.run(
            [
                sys.executable,
                str(STATE_TREE_SCRIPT),
                "build",
                "--state-root",
                str(state_root),
                "--output",
                str(state_manifest),
                "--role",
                "zero2_roundtrip_smoke_state",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert rebuilt.returncode == 0, rebuilt.stderr
        payload["state_tree_manifest_sha256"] = _sha256(state_manifest)
        marker.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        command[command.index("--expected-sha256") + 1] = _sha256(marker)
        wrong_batch = subprocess.run(
            command, text=True, capture_output=True, check=False, env=env
        )
        assert wrong_batch.returncode != 0
        assert "proof batch accounting mismatch at rank 31" in wrong_batch.stderr

        bad_payload["batch_accounting"] = batch_accounting
        bad_proof.write_text(json.dumps(bad_payload) + "\n", encoding="utf-8")
        state_manifest.unlink()
        rebuilt = subprocess.run(
            [
                sys.executable,
                str(STATE_TREE_SCRIPT),
                "build",
                "--state-root",
                str(state_root),
                "--output",
                str(state_manifest),
                "--role",
                "zero2_roundtrip_smoke_state",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert rebuilt.returncode == 0, rebuilt.stderr
        payload["state_tree_manifest_sha256"] = _sha256(state_manifest)
        marker.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        command[command.index("--expected-sha256") + 1] = _sha256(marker)

        (state_root / "optimizer-shard.bin").write_bytes(b"same-shape corrupted state")
        failed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        assert failed.returncode != 0
        assert "state-tree file" in failed.stderr


def test_zero_smoke_runner_clears_pai_topology_and_initializes_accelerator_first() -> None:
    shell = ZERO_RUNNER_SCRIPT.read_text(encoding="utf-8")
    assert "env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE" in shell
    assert "-u GROUP_RANK -u ROLE_RANK" in shell
    assert "-u ACCELERATE_GRADIENT_ACCUMULATION_STEPS" in shell
    source = ZERO_ROUNDTRIP_SCRIPT.read_text(encoding="utf-8")
    build_runtime = source[source.index("def _build_runtime"):source.index("def _train_step")]
    assert build_runtime.index("Accelerator(") < build_runtime.index(
        "set_seed(seed, device_specific=True)"
    )
    assert build_runtime.index("Accelerator(") < build_runtime.index(
        "_configure_smoke_deepspeed_batch_accounting(accelerator)"
    ) < build_runtime.index("accelerator.prepare(")
    train_step = source[source.index("def _train_step"):source.index("def _atomic_json")]
    assert ".reshape(SMOKE_LOCAL_MICRO_BATCH_SIZE, 16)" in train_step
    assert ".reshape(SMOKE_LOCAL_MICRO_BATCH_SIZE, 8)" in train_step


def _zero_smoke_batch_helper_namespace() -> dict[str, object]:
    source = ZERO_ROUNDTRIP_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper_names = {
        "_configure_smoke_deepspeed_batch_accounting",
        "_require_smoke_gradient_accumulation",
        "_resolved_smoke_batch_accounting",
    }
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    assert {helper.name for helper in helpers} == helper_names
    namespace: dict[str, object] = {
        "SMOKE_GRADIENT_ACCUMULATION_STEPS": 1,
        "SMOKE_LOCAL_MICRO_BATCH_SIZE": 4,
    }
    exec(
        compile(ast.Module(body=helpers, type_ignores=[]), str(ZERO_ROUNDTRIP_SCRIPT), "exec"),
        namespace,
    )
    return namespace


class _ZeroSmokePluginStub:
    def __init__(self, deepspeed_config: dict[str, object]) -> None:
        self.deepspeed_config = deepspeed_config


class _ZeroSmokeStateStub:
    def __init__(self, plugin) -> None:
        self.deepspeed_plugin = plugin


class _ZeroSmokeAcceleratorStub:
    def __init__(
        self,
        plugin,
        *,
        gradient_accumulation_steps: int = 1,
        num_processes: int = 32,
    ) -> None:
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_processes = num_processes
        self.state = _ZeroSmokeStateStub(plugin)


def test_zero_smoke_resolves_fixed_micro_batch_without_a_dataloader() -> None:
    namespace = _zero_smoke_batch_helper_namespace()
    configure = namespace["_configure_smoke_deepspeed_batch_accounting"]
    resolved = namespace["_resolved_smoke_batch_accounting"]
    plugin = _ZeroSmokePluginStub(
        {"train_micro_batch_size_per_gpu": "auto"}
    )
    accelerator = _ZeroSmokeAcceleratorStub(plugin)
    configure(accelerator)
    assert plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == 4
    plugin.deepspeed_config.update(
        {
            "gradient_accumulation_steps": 1,
            "train_batch_size": 128,
        }
    )
    assert resolved(accelerator) == {
        "global_train_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "local_micro_batch_size": 4,
        "world_size": 32,
    }


def test_zero_smoke_rejects_ambient_accumulation_override() -> None:
    configure = _zero_smoke_batch_helper_namespace()[
        "_configure_smoke_deepspeed_batch_accounting"
    ]
    accelerator = _ZeroSmokeAcceleratorStub(
        _ZeroSmokePluginStub({"train_micro_batch_size_per_gpu": "auto"}),
        gradient_accumulation_steps=8,
    )
    try:
        configure(accelerator)
    except ValueError as error:
        assert "ACCELERATE_GRADIENT_ACCUMULATION_STEPS" in str(error)
    else:
        raise AssertionError("ambient gradient accumulation override was accepted")


def test_zero_smoke_rejects_wrong_micro_batch_value() -> None:
    configure = _zero_smoke_batch_helper_namespace()[
        "_configure_smoke_deepspeed_batch_accounting"
    ]
    accelerator = _ZeroSmokeAcceleratorStub(
        _ZeroSmokePluginStub({"train_micro_batch_size_per_gpu": 2})
    )
    try:
        configure(accelerator)
    except ValueError as error:
        assert "train_micro_batch_size_per_gpu=4" in str(error)
    else:
        raise AssertionError("wrong DeepSpeed local micro batch was accepted")


def test_zero_smoke_requires_deepspeed_plugin() -> None:
    configure = _zero_smoke_batch_helper_namespace()[
        "_configure_smoke_deepspeed_batch_accounting"
    ]
    accelerator = _ZeroSmokeAcceleratorStub(None)
    try:
        configure(accelerator)
    except RuntimeError as error:
        assert "requires the DeepSpeed plugin" in str(error)
    else:
        raise AssertionError("missing DeepSpeed plugin was accepted")


def test_pod_image_digest_probe_outputs_only_normalized_identity() -> None:
    env = _base_env()
    env.update(
        {
            "FASTWAM_POD_IMAGE_ID": (
                "docker-pullable://registry.example/private/image@sha256:" + "A" * 64
            ),
            "DO_NOT_PRINT_THIS_TOKEN": "secret-token-value",
        }
    )
    result = subprocess.run(
        [sys.executable, str(IMAGE_DIGEST_PROBE_SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == [
        {
            "container": "FASTWAM_POD_IMAGE_ID",
            "digest": "sha256:" + "a" * 64,
            "source": "environment",
        }
    ]
    assert "registry.example" not in result.stdout
    assert "secret-token-value" not in result.stdout + result.stderr


def test_runtime_rank_zero_atomic_config_file_barrier() -> None:
    spec = importlib.util.spec_from_file_location("fastwam_runtime_provenance_test", RUNTIME_PROVENANCE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        payload = b"a: 1\nb: 2\n"
        observed: list[str] = []
        errors: list[BaseException] = []

        def waiter() -> None:
            try:
                observed.append(
                    module.publish_rank_zero_file(
                        path, payload, rank=1, world_size=2, timeout_seconds=2
                    )
                )
            except BaseException as error:  # pragma: no cover - asserted below
                errors.append(error)

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.15)
        owner_sha = module.publish_rank_zero_file(
            path, payload, rank=0, world_size=2, timeout_seconds=2
        )
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert not errors
        assert observed == [owner_sha]
        assert owner_sha == hashlib.sha256(payload).hexdigest()
        assert path.read_bytes() == payload
        assert len(list(path.parent.glob(".config.yaml.ready.*"))) == 1


def test_prepare_local_training_bundle_rewrites_stats_provenance_without_mutating_source() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        bundle = tmp_path / "bundle"
        dataset = bundle / "dataset"
        text_root = dataset / "text_embeds"
        gaussian = bundle / "gaussian"
        model_cache = bundle / "model-cache"
        vae = (
            model_cache
            / "DiffSynth-Studio"
            / "Wan-Series-Converted-Safetensors"
            / "Wan2.2_VAE.safetensors"
        )
        checkpoint = bundle / "checkpoint.pt"
        stats = dataset / "stats.json"
        text_root.mkdir(parents=True)
        gaussian.mkdir(parents=True)
        vae.parent.mkdir(parents=True)
        for index in range(24):
            (dataset / f"part-{index:02d}.h5").write_bytes(b"h5")
        (text_root / "prompt.pt").write_bytes(b"embedding")
        (gaussian / "manifest.json").write_text("{}", encoding="utf-8")
        checkpoint.write_bytes(b"official checkpoint fixture")
        vae.write_bytes(b"vae fixture")
        original_stats = {
            "source_root": "/cpfs/user/chengjuntao/datasets/robofactory_multi_robot",
            "files": 24,
            "trajectories": 1,
            "cardinality": {"agent_counts": [2, 3, 4], "trajectories_by_agent_count": {"2": 1}},
            "action": {"mean": [0], "std": [1]},
            "state": {"mean": [0], "std": [1]},
        }
        stats.write_text(json.dumps(original_stats), encoding="utf-8")
        derived = tmp_path / "runtime" / "stats.json"
        checkpoint_sha = _sha256(checkpoint)
        vae_sha = _sha256(vae)
        result = subprocess.run(
            [
                sys.executable,
                str(PREPARE_BUNDLE_SCRIPT),
                "--bundle-root",
                str(bundle),
                "--bundle-manifest-sha256",
                "a" * 64,
                "--dataset-root",
                str(dataset),
                "--expected-h5-files",
                "24",
                "--stats-source",
                str(stats),
                "--text-embeds-root",
                str(text_root),
                "--checkpoint",
                str(checkpoint),
                "--checkpoint-manifest-sha256",
                checkpoint_sha,
                "--expected-checkpoint-sha256",
                checkpoint_sha,
                "--model-cache-root",
                str(model_cache),
                "--vae",
                str(vae),
                "--vae-manifest-sha256",
                vae_sha,
                "--expected-vae-sha256",
                vae_sha,
                "--gaussian-root",
                str(gaussian),
                "--gaussian-bundle-root",
                str(bundle),
                "--output-stats",
                str(derived),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["dataset_h5_files"] == 24
        assert json.loads(result.stdout)["vae_sha256"] == vae_sha
        derived_payload = json.loads(derived.read_text(encoding="utf-8"))
        assert derived_payload["source_root"] == str(dataset.resolve())
        assert derived_payload["fastwam_local_derivation"]["bundle_manifest_sha256"] == "a" * 64
        assert json.loads(stats.read_text(encoding="utf-8")) == original_stats


def test_local_cache_copies_whole_files_maps_gaussian_and_reuses_ready() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        source_root = tmp_path / "cpfs-source"
        cache_root = tmp_path / "node-local"
        (source_root / "dataset").mkdir(parents=True)
        (source_root / "checkpoint").mkdir(parents=True)
        dataset = source_root / "dataset" / "episodes.hdf5"
        checkpoint = source_root / "checkpoint" / "model.pt"
        dataset.write_bytes(b"hdf5-whole-file\x00" * 101)
        checkpoint.write_bytes(b"checkpoint-whole-file\x01" * 73)
        manifest = tmp_path / "cache.sha256"
        manifest.write_text(
            f"{_sha256(dataset)}  ./dataset/episodes.hdf5\n"
            f"{_sha256(checkpoint)}  checkpoint/model.pt\n",
            encoding="utf-8",
        )
        env = _cache_environment(source_root, manifest, cache_root)
        env.update(
            {
                "FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT": "dataset",
                "FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH": "checkpoint/model.pt",
            }
        )
        command = [
            "bash",
            "-c",
            'source "$1"; fastwam_prepare_local_cache; '
            'printf "%s|%s|%s|%s\\n" "$FASTWAM_LOCAL_CACHE_DIR" '
            '"$FASTWAM_GAUSSIAN_CACHE_DIR" "$FASTWAM_LOCAL_CHECKPOINT_PATH" '
            '"$FASTWAM_LOCAL_CHECKPOINT_MANIFEST_SHA256"',
            "cache-test",
            str(CACHE_SCRIPT),
        ]

        first = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
        assert first.returncode == 0, first.stderr
        local_dir, gaussian_dir, checkpoint_path, checkpoint_sha = first.stdout.strip().split("|")
        destination = Path(local_dir)
        assert gaussian_dir == str(destination / "dataset")
        assert checkpoint_path == str(destination / "checkpoint" / "model.pt")
        assert checkpoint_sha == _sha256(checkpoint)
        assert (destination / ".FASTWAM_READY").is_file()
        assert (destination / "dataset" / "episodes.hdf5").read_bytes() == dataset.read_bytes()
        assert (destination / "checkpoint" / "model.pt").read_bytes() == checkpoint.read_bytes()

        second = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
        assert second.returncode == 0, second.stderr
        assert second.stdout.strip() == (
            f"{destination}|{destination / 'dataset'}|"
            f"{destination / 'checkpoint' / 'model.pt'}|{_sha256(checkpoint)}"
        )
        assert "action=hit" in second.stderr


def test_multi_source_cache_combines_cpfs_and_oss_mappings() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        cpfs = tmp_path / "cpfs"
        oss = tmp_path / "oss"
        cache = tmp_path / "local"
        dataset = cpfs / "dataset"
        text_root = dataset / "text"
        model_cache = cpfs / "model-cache"
        gaussian = oss / "compact"
        text_root.mkdir(parents=True)
        model_cache.mkdir(parents=True)
        gaussian.mkdir(parents=True)
        files = {
            cpfs / "checkpoint.pt": b"checkpoint",
            dataset / "part.h5": b"h5",
            dataset / "stats.json": b"{}",
            text_root / "prompt.pt": b"text",
            model_cache / "vae.safetensors": b"vae",
            gaussian / "manifest.json": b"gaussian",
        }
        for path, payload in files.items():
            path.write_bytes(payload)
        cpfs_manifest = tmp_path / "cpfs.sha256"
        oss_manifest = tmp_path / "oss.sha256"
        cpfs_manifest.write_text(
            "".join(
                f"{_sha256(path)}  {path.relative_to(cpfs).as_posix()}\n"
                for path in sorted((path for path in files if cpfs in path.parents))
            ),
            encoding="utf-8",
        )
        oss_manifest.write_text(
            f"{_sha256(gaussian / 'manifest.json')}  compact/manifest.json\n",
            encoding="utf-8",
        )
        env = _base_env()
        env.update(
            {
                "FASTWAM_LOCAL_CACHE_ROOT": str(cache),
                "FASTWAM_LOCAL_CACHE_ALLOW_SHARED_FS": "1",
                "FASTWAM_LOCAL_CACHE_MIN_FREE_BYTES": "0",
                "FASTWAM_CPFS_BUNDLE_SOURCE_ROOT": str(cpfs),
                "FASTWAM_CPFS_BUNDLE_MANIFEST": str(cpfs_manifest),
                "FASTWAM_CPFS_BUNDLE_MANIFEST_SHA256": _sha256(cpfs_manifest),
                "FASTWAM_OSS_BUNDLE_SOURCE_ROOT": str(oss),
                "FASTWAM_OSS_BUNDLE_MANIFEST": str(oss_manifest),
                "FASTWAM_OSS_BUNDLE_MANIFEST_SHA256": _sha256(oss_manifest),
                "FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH": "checkpoint.pt",
                "FASTWAM_LOCAL_DATASET_RELATIVE_ROOT": "dataset",
                "FASTWAM_LOCAL_STATS_RELATIVE_PATH": "dataset/stats.json",
                "FASTWAM_LOCAL_TEXT_EMBEDS_RELATIVE_ROOT": "dataset/text",
                "FASTWAM_LOCAL_MODEL_CACHE_RELATIVE_ROOT": "model-cache",
                "FASTWAM_LOCAL_VAE_RELATIVE_PATH": "model-cache/vae.safetensors",
                "FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT": "compact",
            }
        )
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; fastwam_prepare_multi_source_cache; '
                'printf "%s|%s|%s|%s|%s|%s\\n" "$FASTWAM_LOCAL_CPFS_CACHE_DIR" '
                '"$FASTWAM_LOCAL_OSS_CACHE_DIR" "$FASTWAM_LOCAL_CHECKPOINT_PATH" '
                '"$FASTWAM_GAUSSIAN_CACHE_DIR" "$DIFFSYNTH_MODEL_BASE_PATH" '
                '"$FASTWAM_LOCAL_STATS_MANIFEST_SHA256"',
                "multi-cache-test",
                str(MULTI_CACHE_SCRIPT),
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        cpfs_dest, oss_dest, checkpoint_path, gaussian_path, model_path, stats_sha = (
            result.stdout.strip().split("|")
        )
        assert checkpoint_path == str(Path(cpfs_dest) / "checkpoint.pt")
        assert model_path == str(Path(cpfs_dest) / "model-cache")
        assert gaussian_path == str(Path(oss_dest) / "compact")
        assert stats_sha == _sha256(dataset / "stats.json")
        assert Path(checkpoint_path).read_bytes() == b"checkpoint"
        assert (Path(gaussian_path) / "manifest.json").read_bytes() == b"gaussian"
        assert "mode=multi_source" in result.stderr

        cpfs_only_cache = tmp_path / "local-cpfs-only"
        cpfs_only_env = dict(env)
        cpfs_only_env["FASTWAM_LOCAL_CACHE_ROOT"] = str(cpfs_only_cache)
        for name in (
            "FASTWAM_OSS_BUNDLE_SOURCE_ROOT",
            "FASTWAM_OSS_BUNDLE_MANIFEST",
            "FASTWAM_OSS_BUNDLE_MANIFEST_SHA256",
            "FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT",
        ):
            cpfs_only_env.pop(name, None)
        cpfs_only = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; fastwam_prepare_multi_source_cache; '
                'printf "%s|%s|%s\\n" "$FASTWAM_LOCAL_CPFS_CACHE_DIR" '
                '"${FASTWAM_LOCAL_OSS_CACHE_DIR-}" '
                '"${FASTWAM_LOCAL_OSS_CACHE_MANIFEST_SHA256-}"',
                "multi-cache-cpfs-only-test",
                str(MULTI_CACHE_SCRIPT),
            ],
            cwd=REPO_ROOT,
            env=cpfs_only_env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert cpfs_only.returncode == 0, cpfs_only.stderr
        cpfs_only_dest, oss_cache_dir, oss_cache_sha = cpfs_only.stdout.strip().split("|")
        assert Path(cpfs_only_dest, "checkpoint.pt").read_bytes() == b"checkpoint"
        assert oss_cache_dir == ""
        assert oss_cache_sha == ""
        assert not (cpfs_only_cache / "oss").exists()
        assert "oss_sha256=none" in cpfs_only.stderr


def test_offline_environment_bootstrap_is_content_addressed_and_zstd_free() -> None:
    source = OFFLINE_ENV_SCRIPT.read_text(encoding="utf-8")
    assert "fastwam_prepare_local_cache" in source
    assert "FASTWAM_LOCAL_CACHE_REQUIRE_VERIFY_HIT=1" in source
    assert "--no-index" in source
    assert "--require-hashes" in source
    assert "FASTWAM_OFFLINE_ENV_RUNTIME_LOCK_SHA256" in source
    assert "FASTWAM_OFFLINE_ENV_CACHE_HELPER_SHA256" in source
    assert "dlc_local_cache.sh SHA-256 mismatch" in source
    assert "_fastwam_offline_env_normalize_relative_path" in source
    assert ".FASTWAM_ENV_READY" in source
    assert "git clone --no-hardlinks" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "validate_python_environment.py" in source
    assert "FASTWAM_TRAINING_ENV_BUNDLE_MANIFEST_SHA256" in source
    assert "FASTWAM_NODE_LOCAL_RANK=0" in source
    assert "mv -T" in source
    assert "python3.10" in source
    assert "python_implementation=" in source
    assert "python_platform=" in source
    assert "env -u PYTHONHOME -u PYTHONPATH" in source
    assert "PYTHONDONTWRITEBYTECODE=1" in source
    assert "PYTHONPATH=\"${checkout}/src\"" in source
    assert "fastwam.__file__" in source
    assert "accelerate.commands.launch --help" in source
    assert "${FASTWAM_REPO_ROOT}/scripts/train_zero2.sh" in source
    assert ".LOCK" in source
    assert ".FAILED" in source
    assert "trap _fastwam_" in source
    assert "zstd" not in source.lower()


def _run_offline_env_function(
    function_call: str,
    *arguments: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            f'source "$1"; {function_call}',
            "offline-env-function-test",
            str(OFFLINE_ENV_SCRIPT),
            *arguments,
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_offline_env_tmp_roots_reject_traversal_and_symlink_chain() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory, tempfile.TemporaryDirectory(
        dir="/tmp"
    ) as outside_directory:
        root = Path(directory)
        safe = root / "safe" / "child"
        accepted = _run_offline_env_function(
            '_fastwam_offline_env_prepare_tmp_root TEST_ROOT "$2"', str(safe)
        )
        assert accepted.returncode == 0, accepted.stderr
        assert accepted.stdout.strip() == str(safe)
        assert safe.is_dir()

        traversal = _run_offline_env_function(
            '_fastwam_offline_env_prepare_tmp_root TEST_ROOT "$2"',
            str(root / "safe" / ".." / "escape"),
        )
        assert traversal.returncode != 0
        assert "canonical lexical path" in traversal.stderr

        outside = Path(outside_directory)
        symlink_parent = root / "external-link"
        symlink_parent.symlink_to(outside, target_is_directory=True)
        escaped_child = outside / "must-not-be-created"
        symlink = _run_offline_env_function(
            '_fastwam_offline_env_prepare_tmp_root TEST_ROOT "$2"',
            str(symlink_parent / escaped_child.name),
        )
        assert symlink.returncode != 0
        assert "escapes /tmp" in symlink.stderr or "symlink root or parent" in symlink.stderr
        assert not escaped_child.exists()


def test_offline_env_forces_node_local_builder_rank_zero() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        bootstrap = root / "bootstrap_offline_training_env.sh"
        helper = root / "dlc_local_cache.sh"
        payload = root / "payload"
        rank_log = root / "rank.log"
        payload.mkdir()
        shutil.copy2(OFFLINE_ENV_SCRIPT, bootstrap)
        helper.write_text(
            """fastwam_prepare_local_cache() {
  printf '%s|%s|%s|%s\\n' \
    "$FASTWAM_NODE_LOCAL_RANK" "${LOCAL_RANK-unset}" \
    "${FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH-unset}" \
    "${FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT-unset}" >"$FASTWAM_TEST_RANK_LOG"
  export FASTWAM_LOCAL_CACHE_DIR="$FASTWAM_TEST_PAYLOAD"
}
""",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env.update(
            {
                "FASTWAM_NODE_LOCAL_RANK": "7",
                "LOCAL_RANK": "9",
                "FASTWAM_TEST_RANK_LOG": str(rank_log),
                "FASTWAM_TEST_PAYLOAD": str(payload),
                "FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH": "training/checkpoint.pt",
                "FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT": "training/gaussian",
            }
        )
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; _fastwam_offline_env_stage_payload '
                '"$(dirname -- "$1")" /unused/source /unused/manifest '
                f'{"a" * 64} /tmp/unused-cache 1',
                "rank-zero-test",
                str(bootstrap),
            ],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(payload)
        assert rank_log.read_text(encoding="utf-8").strip() == "0|unset|unset|unset"


def test_offline_env_binds_exact_cpython310_identity() -> None:
    def run_probe(
        root: Path, identity_line: str, *, explicit_path: bool = False
    ) -> subprocess.CompletedProcess[str]:
        fake_python = root / "python3.10"
        fake_python.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' {identity_line} \"$(realpath -e -- \"$0\")\"\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        env = os.environ.copy()
        env["PATH"] = f"{root}:{env.get('PATH', '')}"
        if explicit_path:
            env["FASTWAM_OFFLINE_ENV_BASE_PYTHON"] = str(fake_python)
        else:
            env.pop("FASTWAM_OFFLINE_ENV_BASE_PYTHON", None)
        return _run_offline_env_function(
            '_fastwam_offline_env_bind_python_identity || exit $?; '
            'printf "%s|%s|%s|%s\\n" '
            '"$FASTWAM_OFFLINE_ENV_PYTHON_VERSION" '
            '"$FASTWAM_OFFLINE_ENV_PYTHON_ABI" '
            '"$FASTWAM_OFFLINE_ENV_PYTHON_CACHE_TAG" '
            '"$FASTWAM_OFFLINE_ENV_PYTHON_IDENTITY_SHA256"',
            env=env,
        )

    with tempfile.TemporaryDirectory(dir="/tmp") as first_directory, tempfile.TemporaryDirectory(
        dir="/tmp"
    ) as second_directory:
        first = run_probe(
            Path(first_directory),
            "'3.10.14' 'cpython' 'cpython-310-x86_64-linux-gnu' 'cpython-310' 'linux-x86_64'",
            explicit_path=True,
        )
        assert first.returncode == 0, first.stderr
        version, abi, cache_tag, first_identity = first.stdout.strip().split("|")
        assert version == "3.10.14"
        assert abi == "cpython-310-x86_64-linux-gnu"
        assert cache_tag == "cpython-310"
        assert len(first_identity) == 64

        second = run_probe(
            Path(second_directory),
            "'3.10.15' 'cpython' 'cpython-310-x86_64-linux-gnu' 'cpython-310' 'linux-x86_64'",
        )
        assert second.returncode == 0, second.stderr
        second_identity = second.stdout.strip().split("|")[-1]
        assert second_identity != first_identity

        rejected = run_probe(
            Path(second_directory),
            "'3.11.9' 'cpython' 'cpython-311-x86_64-linux-gnu' 'cpython-311' 'linux-x86_64'",
        )
        assert rejected.returncode != 0
        assert "incompatible version" in rejected.stderr

        rejected_implementation = run_probe(
            Path(second_directory),
            "'3.10.14' 'pypy' 'pypy310-x86_64-linux-gnu' 'pypy310' 'linux-x86_64'",
        )
        assert rejected_implementation.returncode != 0
        assert "implementation must be CPython" in rejected_implementation.stderr


def test_offline_env_import_origin_is_exact_checkout() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        checkout = root / "checkout"
        package = checkout / "src" / "fastwam"
        poison = root / "poison" / "fastwam"
        package.mkdir(parents=True)
        poison.mkdir(parents=True)
        (package / "__init__.py").write_text("ORIGIN = 'checkout'\n", encoding="utf-8")
        (poison / "__init__.py").write_text(
            "raise RuntimeError('poison PYTHONPATH was imported')\n", encoding="utf-8"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(poison.parent)
        result = _run_offline_env_function(
            '_fastwam_offline_env_validate_import_resolution "$2" "$3"',
            sys.executable,
            str(checkout),
            env=env,
        )
        assert result.returncode == 0, result.stderr


def _create_offline_checkout_fixture(root: Path) -> tuple[Path, str]:
    repository = root / "producer"
    (repository / "scripts").mkdir(parents=True)
    (repository / "src" / "fastwam").mkdir(parents=True)
    (repository / ".gitignore").write_text("*.pyc\n__pycache__/\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text("[project]\nname='fastwam'\n", encoding="utf-8")
    (repository / "scripts" / "validate_python_environment.py").write_text(
        "raise SystemExit(0)\n", encoding="utf-8"
    )
    (repository / "scripts" / "train_zero2.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (repository / "src" / "fastwam" / "__init__.py").write_text("\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=FastWAM Test",
            "-c",
            "user.email=fastwam-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    bundle = root / "source.bundle"
    subprocess.run(
        ["git", "-C", str(repository), "bundle", "create", str(bundle), "HEAD"],
        check=True,
    )
    return bundle, commit


def _offline_checkout_process(
    *,
    bundle: Path,
    checkout_root: Path,
    identity: str,
    marker: str,
    commit: str,
    env: dict[str, str],
    start_new_session: bool = False,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            "bash",
            "-c",
            'source "$1"; _fastwam_offline_env_prepare_checkout '
            '"$2" "$3" "$4" "$5" "$6" 15 60',
            "checkout-concurrency-test",
            str(OFFLINE_ENV_SCRIPT),
            str(bundle),
            str(checkout_root),
            identity,
            marker,
            commit,
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=start_new_session,
    )


def test_offline_env_concurrent_checkout_builds_once_and_waits() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        bundle, commit = _create_offline_checkout_fixture(root)
        checkout_root = root / "checkouts"
        checkout_root.mkdir()
        fake_bin = root / "fake-bin"
        fake_bin.mkdir()
        clone_log = root / "clone.log"
        git_wrapper = fake_bin / "git"
        git_wrapper.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1-}" == clone ]]; then
  printf 'clone pid=%s\\n' "$BASHPID" >>"$FASTWAM_TEST_CLONE_LOG"
  sleep 1
fi
exec "$FASTWAM_TEST_REAL_GIT" "$@"
""",
            encoding="utf-8",
        )
        git_wrapper.chmod(0o755)
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{fake_bin}:{env.get('PATH', '')}",
                "FASTWAM_TEST_REAL_GIT": shutil.which("git") or "git",
                "FASTWAM_TEST_CLONE_LOG": str(clone_log),
            }
        )
        identity = "b" * 64
        marker = f"schema=test-checkout\ncommit={commit}"
        owner = _offline_checkout_process(
            bundle=bundle,
            checkout_root=checkout_root,
            identity=identity,
            marker=marker,
            commit=commit,
            env=env,
        )
        time.sleep(0.15)
        waiter = _offline_checkout_process(
            bundle=bundle,
            checkout_root=checkout_root,
            identity=identity,
            marker=marker,
            commit=commit,
            env=env,
        )
        owner_stdout, owner_stderr = owner.communicate(timeout=15)
        waiter_stdout, waiter_stderr = waiter.communicate(timeout=15)
        assert owner.returncode == 0, owner_stderr
        assert waiter.returncode == 0, waiter_stderr
        assert owner_stdout.strip() == waiter_stdout.strip()
        assert len(clone_log.read_text(encoding="utf-8").splitlines()) == 1
        assert "action=wait" in waiter_stderr
        assert not list(checkout_root.glob(".*.LOCK"))
        assert not list(checkout_root.glob(".*.FAILED"))
        assert not list(checkout_root.glob(".*.STAGING.*"))
        destination = Path(owner_stdout.strip())
        assert destination.is_dir()
        assert not any("STAGING" in child.name for child in destination.iterdir())


def test_offline_env_checkout_failure_wakes_waiter_and_cleans() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        bundle, commit = _create_offline_checkout_fixture(root)
        checkout_root = root / "checkouts"
        checkout_root.mkdir()
        fake_bin = root / "fake-bin"
        fake_bin.mkdir()
        clone_log = root / "clone.log"
        git_wrapper = fake_bin / "git"
        git_wrapper.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1-}" == clone ]]; then
  printf 'clone pid=%s\\n' "$BASHPID" >>"$FASTWAM_TEST_CLONE_LOG"
  sleep 1
  exit 42
fi
exec "$FASTWAM_TEST_REAL_GIT" "$@"
""",
            encoding="utf-8",
        )
        git_wrapper.chmod(0o755)
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{fake_bin}:{env.get('PATH', '')}",
                "FASTWAM_TEST_REAL_GIT": shutil.which("git") or "git",
                "FASTWAM_TEST_CLONE_LOG": str(clone_log),
            }
        )
        identity = "c" * 64
        marker = f"schema=test-checkout\ncommit={commit}"
        owner = _offline_checkout_process(
            bundle=bundle,
            checkout_root=checkout_root,
            identity=identity,
            marker=marker,
            commit=commit,
            env=env,
        )
        time.sleep(0.15)
        waiter = _offline_checkout_process(
            bundle=bundle,
            checkout_root=checkout_root,
            identity=identity,
            marker=marker,
            commit=commit,
            env=env,
        )
        _, owner_stderr = owner.communicate(timeout=15)
        _, waiter_stderr = waiter.communicate(timeout=15)
        assert owner.returncode != 0
        assert waiter.returncode != 0
        assert len(clone_log.read_text(encoding="utf-8").splitlines()) == 1
        assert "builder failed" in waiter_stderr, waiter_stderr
        assert "action=build" in owner_stderr
        assert not list(checkout_root.glob(".*.LOCK"))
        assert len(list(checkout_root.glob(".*.FAILED"))) == 1
        assert not list(checkout_root.glob(".*.STAGING.*"))
        assert not list(checkout_root.glob("source-*"))
        assert not list(checkout_root.glob(".*.READY"))


def test_offline_env_fresh_ownerless_lock_is_not_stale() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        lock_dir = Path(directory) / ".identity.LOCK"
        lock_dir.mkdir()
        result = _run_offline_env_function(
            '_fastwam_offline_env_lock_is_stale "$2" 7200; '
            'status=$?; [[ "$status" == 1 ]]',
            str(lock_dir),
        )
        assert result.returncode == 0, result.stderr
        assert lock_dir.is_dir()


def test_offline_env_checkout_hit_waits_for_active_lock() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        bundle, commit = _create_offline_checkout_fixture(root)
        checkout_root = root / "checkouts"
        checkout_root.mkdir()
        identity = "d" * 64
        marker = f"schema=test-checkout\ncommit={commit}"
        first = _offline_checkout_process(
            bundle=bundle,
            checkout_root=checkout_root,
            identity=identity,
            marker=marker,
            commit=commit,
            env=os.environ.copy(),
        )
        first_stdout, first_stderr = first.communicate(timeout=15)
        assert first.returncode == 0, first_stderr

        lock_dir = checkout_root / f".source-{identity}.LOCK"
        lock_dir.mkdir()
        (lock_dir / "owner").write_text(
            "token=active-test\npid=1\nhost=another-host\n"
            f"time={int(time.time())}\nstaging=\n",
            encoding="utf-8",
        )
        late = _offline_checkout_process(
            bundle=bundle,
            checkout_root=checkout_root,
            identity=identity,
            marker=marker,
            commit=commit,
            env=os.environ.copy(),
        )
        time.sleep(0.25)
        assert late.poll() is None, "late caller incorrectly consumed READY while LOCK existed"
        shutil.rmtree(lock_dir)
        late_stdout, late_stderr = late.communicate(timeout=15)
        assert late.returncode == 0, late_stderr
        assert late_stdout.strip() == first_stdout.strip()
        assert "action=wait" in late_stderr


def test_offline_env_prepare_venv_stdout_is_only_destination() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        venv_root = root / "venvs"
        venv_root.mkdir()
        identity = "e" * 64
        expected_destination = venv_root / f"cpython3.10-{identity}"
        command = r'''
source "$1"
_fastwam_offline_env_validate_venv() { [[ -d "$1" ]]; }
_fastwam_offline_env_validate_training_runtime() { echo "runtime validation output"; }
_fastwam_offline_env_build_venv() {
  local destination="$4"
  local lock_dir="$6"
  local lock_token="${17}"
  echo "pip and validator output"
  mkdir -p -- "$destination/bin"
  _fastwam_offline_env_release_lock "$lock_dir" "$lock_token"
}
_fastwam_offline_env_prepare_venv \
  /unused/lock /unused/wheelhouse /unused/checkout "$2" "$3" marker \
  15 60 3.10.14 cpython cpython-310-x86_64-linux-gnu cpython-310 \
  linux-x86_64 "$4" /unused/python
'''
        result = subprocess.run(
            [
                "bash",
                "-c",
                command,
                "venv-stdout-test",
                str(OFFLINE_ENV_SCRIPT),
                str(venv_root),
                identity,
                "f" * 64,
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(expected_destination)
        assert "pip and validator output" in result.stderr
        assert "runtime validation output" in result.stderr


def test_offline_env_venv_ready_is_published_after_runtime_gate() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        runtime_lock = root / "runtime.lock"
        wheelhouse = root / "wheelhouse"
        checkout = root / "checkout"
        venv_root = root / "venvs"
        fake_python = root / "python3.10"
        entered = root / "validator-entered"
        release = root / "validator-release"
        runtime_lock.write_text("", encoding="utf-8")
        wheelhouse.mkdir()
        checkout.mkdir()
        venv_root.mkdir()
        fake_python.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ " $* " == *" -m venv "* ]]; then
  destination="${@: -1}"
  mkdir -p -- "$destination/bin"
  cp -- "$0" "$destination/bin/python"
  chmod 0755 "$destination/bin/python"
  exit 0
fi
if [[ " $* " == *" -c "* ]]; then
  printf '3.10.14\\tcpython\\tcpython-310-x86_64-linux-gnu\\tcpython-310\\tlinux-x86_64\\n'
  exit 0
fi
exit 0
""",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        executable_sha = _sha256(fake_python)
        identity = "8" * 64
        marker = "schema=test-venv\npython_version=3.10.14"
        expected_destination = venv_root / f"cpython3.10-{identity}"
        command = r'''
source "$1"
_fastwam_offline_env_validate_training_runtime() {
  printf 'entered\n' >"$FASTWAM_TEST_VALIDATOR_ENTERED"
  while [[ ! -e "$FASTWAM_TEST_VALIDATOR_RELEASE" ]]; do sleep 0.05; done
  echo "runtime validation output"
}
_fastwam_offline_env_prepare_venv \
  "$2" "$3" "$4" "$5" "$6" "$7" 15 60 \
  3.10.14 cpython cpython-310-x86_64-linux-gnu cpython-310 \
  linux-x86_64 "$8" "$9"
'''
        env = os.environ.copy()
        env.update(
            {
                "FASTWAM_TEST_VALIDATOR_ENTERED": str(entered),
                "FASTWAM_TEST_VALIDATOR_RELEASE": str(release),
            }
        )
        process = subprocess.Popen(
            [
                "bash",
                "-c",
                command,
                "venv-ready-test",
                str(OFFLINE_ENV_SCRIPT),
                str(runtime_lock),
                str(wheelhouse),
                str(checkout),
                str(venv_root),
                identity,
                marker,
                executable_sha,
                str(fake_python),
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not entered.is_file():
            time.sleep(0.02)
        assert entered.is_file(), "staged runtime validator was not reached"
        assert not expected_destination.exists()
        assert not list(venv_root.glob("*/.FASTWAM_ENV_READY"))
        release.touch()
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stderr
        assert stdout.strip() == str(expected_destination)
        assert (expected_destination / ".FASTWAM_ENV_READY").read_text(
            encoding="utf-8"
        ).strip() == marker
        assert "runtime validation output" in stderr
        assert not list(venv_root.glob(".*.LOCK"))
        assert not list(venv_root.glob(".*.FAILED"))
        assert not list(venv_root.glob(".*.STAGING.*"))


def test_offline_env_concurrent_venv_builds_once_and_waits() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        runtime_lock = root / "runtime.lock"
        wheelhouse = root / "wheelhouse"
        checkout = root / "checkout"
        venv_root = root / "venvs"
        fake_python = root / "python3.10"
        build_log = root / "venv-build.log"
        runtime_lock.write_text("", encoding="utf-8")
        wheelhouse.mkdir()
        checkout.mkdir()
        venv_root.mkdir()
        fake_python.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ " $* " == *" -m venv "* ]]; then
  printf 'venv pid=%s\\n' "$BASHPID" >>"$FASTWAM_TEST_VENV_BUILD_LOG"
  sleep 1
  destination="${@: -1}"
  mkdir -p -- "$destination/bin"
  cp -- "$0" "$destination/bin/python"
  chmod 0755 "$destination/bin/python"
  exit 0
fi
if [[ " $* " == *" -c "* ]]; then
  printf '3.10.14\\tcpython\\tcpython-310-x86_64-linux-gnu\\tcpython-310\\tlinux-x86_64\\n'
  exit 0
fi
exit 0
""",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        executable_sha = _sha256(fake_python)
        identity = "7" * 64
        marker = "schema=test-concurrent-venv\npython_version=3.10.14"
        command = r'''
source "$1"
_fastwam_offline_env_validate_training_runtime() { echo "runtime validation output"; }
_fastwam_offline_env_prepare_venv \
  "$2" "$3" "$4" "$5" "$6" "$7" 15 60 \
  3.10.14 cpython cpython-310-x86_64-linux-gnu cpython-310 \
  linux-x86_64 "$8" "$9"
'''
        arguments = [
            "bash",
            "-c",
            command,
            "venv-concurrency-test",
            str(OFFLINE_ENV_SCRIPT),
            str(runtime_lock),
            str(wheelhouse),
            str(checkout),
            str(venv_root),
            identity,
            marker,
            executable_sha,
            str(fake_python),
        ]
        env = os.environ.copy()
        env["FASTWAM_TEST_VENV_BUILD_LOG"] = str(build_log)
        owner = subprocess.Popen(
            arguments,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.15)
        waiter = subprocess.Popen(
            arguments,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        owner_stdout, owner_stderr = owner.communicate(timeout=15)
        waiter_stdout, waiter_stderr = waiter.communicate(timeout=15)
        assert owner.returncode == 0, owner_stderr
        assert waiter.returncode == 0, waiter_stderr
        assert owner_stdout.strip() == waiter_stdout.strip()
        assert len(build_log.read_text(encoding="utf-8").splitlines()) == 1
        assert "action=wait" in waiter_stderr
        assert not list(venv_root.glob(".*.LOCK"))
        assert not list(venv_root.glob(".*.FAILED"))
        assert not list(venv_root.glob(".*.STAGING.*"))


def test_offline_env_venv_failure_wakes_waiter_and_cleans() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        runtime_lock = root / "runtime.lock"
        wheelhouse = root / "wheelhouse"
        checkout = root / "checkout"
        venv_root = root / "venvs"
        fake_python = root / "python3.10"
        build_log = root / "venv-build.log"
        runtime_lock.write_text("", encoding="utf-8")
        wheelhouse.mkdir()
        checkout.mkdir()
        venv_root.mkdir()
        fake_python.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ " $* " == *" -m venv "* ]]; then
  printf 'venv pid=%s\\n' "$BASHPID" >>"$FASTWAM_TEST_VENV_BUILD_LOG"
  destination="${@: -1}"
  mkdir -p -- "$destination/bin"
  sleep 1
  exit 42
fi
exit 0
""",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        identity = "6" * 64
        command = r'''
source "$1"
_fastwam_offline_env_validate_training_runtime() { return 0; }
_fastwam_offline_env_prepare_venv \
  "$2" "$3" "$4" "$5" "$6" marker 15 60 \
  3.10.14 cpython cpython-310-x86_64-linux-gnu cpython-310 \
  linux-x86_64 "$7" "$8"
'''
        arguments = [
            "bash",
            "-c",
            command,
            "venv-failure-test",
            str(OFFLINE_ENV_SCRIPT),
            str(runtime_lock),
            str(wheelhouse),
            str(checkout),
            str(venv_root),
            identity,
            _sha256(fake_python),
            str(fake_python),
        ]
        env = os.environ.copy()
        env["FASTWAM_TEST_VENV_BUILD_LOG"] = str(build_log)
        owner = subprocess.Popen(
            arguments,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.15)
        waiter = subprocess.Popen(
            arguments,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _, owner_stderr = owner.communicate(timeout=15)
        _, waiter_stderr = waiter.communicate(timeout=15)
        assert owner.returncode != 0
        assert waiter.returncode != 0
        assert "action=build" in owner_stderr
        assert "builder failed" in waiter_stderr, (
            waiter_stderr
            + "\nOWNER:\n"
            + owner_stderr
            + "\nFILES:\n"
            + "\n".join(path.name for path in venv_root.iterdir())
        )
        assert len(build_log.read_text(encoding="utf-8").splitlines()) == 1
        assert not list(venv_root.glob(".*.LOCK"))
        assert len(list(venv_root.glob(".*.FAILED"))) == 1
        assert not list(venv_root.glob(".*.STAGING.*"))
        assert not list(venv_root.glob("cpython3.10-*"))


def test_offline_env_rejects_ignored_checkout_artifacts() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        bundle, commit = _create_offline_checkout_fixture(root)
        checkout = root / "checkout"
        subprocess.run(["git", "clone", "-q", str(bundle), str(checkout)], check=True)
        ignored = checkout / "src" / "fastwam" / "shadow.pyc"
        ignored.write_bytes(b"not bytecode")
        result = _run_offline_env_function(
            '_fastwam_offline_env_validate_checkout_tree "$2" "$3"',
            str(checkout),
            commit,
        )
        assert result.returncode != 0


def test_offline_env_waiter_recovers_hard_killed_builder_and_staging() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        bundle, commit = _create_offline_checkout_fixture(root)
        checkout_root = root / "checkouts"
        checkout_root.mkdir()
        fake_bin = root / "fake-bin"
        fake_bin.mkdir()
        clone_log = root / "clone.log"
        git_wrapper = fake_bin / "git"
        git_wrapper.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1-}" == clone ]]; then
  printf 'clone pid=%s\\n' "$BASHPID" >>"$FASTWAM_TEST_CLONE_LOG"
  sleep 1
fi
exec "$FASTWAM_TEST_REAL_GIT" "$@"
""",
            encoding="utf-8",
        )
        git_wrapper.chmod(0o755)
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{fake_bin}:{env.get('PATH', '')}",
                "FASTWAM_TEST_REAL_GIT": shutil.which("git") or "git",
                "FASTWAM_TEST_CLONE_LOG": str(clone_log),
            }
        )
        identity = "9" * 64
        marker = f"schema=test-checkout\ncommit={commit}"
        owner = _offline_checkout_process(
            bundle=bundle,
            checkout_root=checkout_root,
            identity=identity,
            marker=marker,
            commit=commit,
            env=env,
            start_new_session=True,
        )
        lock_dir = checkout_root / f".source-{identity}.LOCK"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            owner_file = lock_dir / "owner"
            if owner_file.is_file() and "staging=" in owner_file.read_text(encoding="utf-8"):
                staging_line = next(
                    line
                    for line in owner_file.read_text(encoding="utf-8").splitlines()
                    if line.startswith("staging=")
                )
                if staging_line != "staging=":
                    break
            time.sleep(0.02)
        else:
            os.killpg(owner.pid, signal.SIGKILL)
            raise AssertionError("owner never published its staging identity")

        waiter = _offline_checkout_process(
            bundle=bundle,
            checkout_root=checkout_root,
            identity=identity,
            marker=marker,
            commit=commit,
            env=env,
        )
        second_waiter = _offline_checkout_process(
            bundle=bundle,
            checkout_root=checkout_root,
            identity=identity,
            marker=marker,
            commit=commit,
            env=env,
        )
        time.sleep(0.2)
        os.killpg(owner.pid, signal.SIGKILL)
        owner.communicate(timeout=5)
        waiter_stdout, waiter_stderr = waiter.communicate(timeout=15)
        second_stdout, second_stderr = second_waiter.communicate(timeout=15)
        assert owner.returncode == -signal.SIGKILL
        assert waiter.returncode == 0, waiter_stderr
        assert second_waiter.returncode == 0, second_stderr
        assert Path(waiter_stdout.strip()).is_dir()
        assert second_stdout.strip() == waiter_stdout.strip()
        combined_stderr = waiter_stderr + second_stderr
        assert "action=retry_stale" in combined_stderr
        assert "action=reap_stale" in combined_stderr
        assert "action=wait" in combined_stderr
        assert len(clone_log.read_text(encoding="utf-8").splitlines()) == 2
        assert not list(checkout_root.glob(".*.LOCK"))
        assert not list(checkout_root.glob(".*.FAILED"))
        assert not list(checkout_root.glob(".*.STAGING.*"))


def test_offline_env_post_publish_owner_kill_does_not_replace_checkout() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        bundle, commit = _create_offline_checkout_fixture(root)
        checkout_root = root / "checkouts"
        checkout_root.mkdir()
        fake_bin = root / "fake-bin"
        fake_bin.mkdir()
        clone_log = root / "clone.log"
        release_entered = root / "release-entered"
        git_wrapper = fake_bin / "git"
        git_wrapper.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1-}" == clone ]]; then
  printf 'clone pid=%s\\n' "$BASHPID" >>"$FASTWAM_TEST_CLONE_LOG"
fi
exec "$FASTWAM_TEST_REAL_GIT" "$@"
""",
            encoding="utf-8",
        )
        git_wrapper.chmod(0o755)
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{fake_bin}:{env.get('PATH', '')}",
                "FASTWAM_TEST_REAL_GIT": shutil.which("git") or "git",
                "FASTWAM_TEST_CLONE_LOG": str(clone_log),
                "FASTWAM_TEST_RELEASE_ENTERED": str(release_entered),
            }
        )
        identity = "5" * 64
        marker = f"schema=test-checkout\ncommit={commit}"
        owner_command = r'''
source "$1"
_fastwam_offline_env_release_lock() {
  printf 'entered\n' >"$FASTWAM_TEST_RELEASE_ENTERED"
  while true; do sleep 0.05; done
}
_fastwam_offline_env_prepare_checkout "$2" "$3" "$4" "$5" "$6" 15 60
'''
        owner = subprocess.Popen(
            [
                "bash",
                "-c",
                owner_command,
                "post-publish-owner-test",
                str(OFFLINE_ENV_SCRIPT),
                str(bundle),
                str(checkout_root),
                identity,
                marker,
                commit,
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and not release_entered.is_file():
            time.sleep(0.02)
        assert release_entered.is_file(), "owner did not reach post-publication release"
        destination = checkout_root / f"source-{identity}"
        ready = checkout_root / f".source-{identity}.READY"
        lock_dir = checkout_root / f".source-{identity}.LOCK"
        assert destination.is_dir() and ready.is_file() and lock_dir.is_dir()
        published_inode = destination.stat().st_ino

        first_waiter = _offline_checkout_process(
            bundle=bundle,
            checkout_root=checkout_root,
            identity=identity,
            marker=marker,
            commit=commit,
            env=env,
        )
        second_waiter = _offline_checkout_process(
            bundle=bundle,
            checkout_root=checkout_root,
            identity=identity,
            marker=marker,
            commit=commit,
            env=env,
        )
        time.sleep(0.2)
        os.killpg(owner.pid, signal.SIGKILL)
        owner.communicate(timeout=5)
        first_stdout, first_stderr = first_waiter.communicate(timeout=15)
        second_stdout, second_stderr = second_waiter.communicate(timeout=15)
        assert owner.returncode == -signal.SIGKILL
        assert first_waiter.returncode == 0, first_stderr
        assert second_waiter.returncode == 0, second_stderr
        assert first_stdout.strip() == second_stdout.strip() == str(destination)
        assert destination.stat().st_ino == published_inode
        assert len(clone_log.read_text(encoding="utf-8").splitlines()) == 1
        combined_stderr = first_stderr + second_stderr
        assert "action=recovered_hit" in combined_stderr
        assert not lock_dir.exists()
        assert not list(checkout_root.glob(".*.FAILED"))
        assert not list(checkout_root.glob(".*.STAGING.*"))


def test_offline_env_venv_rechecks_ready_after_acquiring_lock() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        venv_root = root / "venvs"
        venv_root.mkdir()
        identity = "4" * 64
        destination = venv_root / f"cpython3.10-{identity}"
        (destination / "bin").mkdir(parents=True)
        first_probe = root / "first-probe"
        build_called = root / "build-called"
        command = r'''
source "$1"
_fastwam_offline_env_validate_venv() {
  if [[ ! -e "$FASTWAM_TEST_FIRST_PROBE" ]]; then
    : >"$FASTWAM_TEST_FIRST_PROBE"
    return 1
  fi
  [[ -d "$1" ]]
}
_fastwam_offline_env_validate_training_runtime() { return 0; }
_fastwam_offline_env_build_venv() {
  : >"$FASTWAM_TEST_BUILD_CALLED"
  return 99
}
_fastwam_offline_env_prepare_venv \
  /unused/lock /unused/wheelhouse /unused/checkout "$2" "$3" marker \
  15 60 3.10.14 cpython cpython-310-x86_64-linux-gnu cpython-310 \
  linux-x86_64 "$4" /unused/python
'''
        env = os.environ.copy()
        env.update(
            {
                "FASTWAM_TEST_FIRST_PROBE": str(first_probe),
                "FASTWAM_TEST_BUILD_CALLED": str(build_called),
            }
        )
        inode = destination.stat().st_ino
        result = subprocess.run(
            [
                "bash",
                "-c",
                command,
                "venv-acquired-hit-test",
                str(OFFLINE_ENV_SCRIPT),
                str(venv_root),
                identity,
                "a" * 64,
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(destination)
        assert "action=recovered_hit" in result.stderr
        assert destination.stat().st_ino == inode
        assert not build_called.exists()
        assert not list(venv_root.glob(".*.LOCK"))


def test_local_cache_default_hit_verification_fails_closed_on_corruption() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        source_root, cache_root, manifest, _ = _one_file_cache_fixture(tmp_path)
        env = _cache_environment(source_root, manifest, cache_root)
        first = subprocess.run(["bash", str(CACHE_SCRIPT)], cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
        assert first.returncode == 0, first.stderr
        destination = Path(first.stdout.strip())
        (destination / "payload.bin").write_bytes(b"corrupted")

        second = subprocess.run(["bash", str(CACHE_SCRIPT)], cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
        assert second.returncode != 0
        assert "cached SHA-256 mismatch" in second.stderr

        env["FASTWAM_LOCAL_CACHE_REQUIRE_VERIFY_HIT"] = "1"
        env["FASTWAM_LOCAL_CACHE_VERIFY_HIT"] = "0"
        forbidden = subprocess.run(["bash", str(CACHE_SCRIPT)], cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
        assert forbidden.returncode != 0
        assert "forbids disabling" in forbidden.stderr


def test_local_cache_rejects_parent_traversal_and_manifest_symlink() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        source_root = tmp_path / "cpfs-source"
        cache_root = tmp_path / "node-local"
        source_root.mkdir()
        manifest = tmp_path / "cache.sha256"
        manifest.write_text(f"{'0' * 64}  ../escape.bin\n", encoding="utf-8")
        env = _cache_environment(source_root, manifest, cache_root)

        traversal = subprocess.run(["bash", str(CACHE_SCRIPT)], cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
        assert traversal.returncode != 0
        assert "unsafe relative path" in traversal.stderr

        payload = source_root / "payload.bin"
        payload.write_bytes(b"payload")
        manifest.write_text(f"{_sha256(payload)}  payload.bin\n", encoding="utf-8")
        manifest_link = tmp_path / "manifest-link.sha256"
        manifest_link.symlink_to(manifest)
        env["FASTWAM_LOCAL_CACHE_MANIFEST"] = str(manifest_link)
        symlink = subprocess.run(["bash", str(CACHE_SCRIPT)], cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
        assert symlink.returncode != 0
        assert "regular non-symlink file" in symlink.stderr


def test_local_cache_rejects_mapping_escape_and_manifest_identity_mismatch() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        source_root, cache_root, manifest, _ = _one_file_cache_fixture(tmp_path)
        env = _cache_environment(source_root, manifest, cache_root)
        env["FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH"] = "../payload.bin"
        escaped = subprocess.run(
            ["bash", str(CACHE_SCRIPT)], cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False
        )
        assert escaped.returncode != 0
        assert "unsafe node-local mapping relative path" in escaped.stderr

        env["FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH"] = "payload.bin"
        env["FASTWAM_LOCAL_CACHE_EXPECTED_MANIFEST_SHA256"] = "0" * 64
        mismatch = subprocess.run(
            ["bash", str(CACHE_SCRIPT)], cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False
        )
        assert mismatch.returncode != 0
        assert "bundle manifest SHA-256 mismatch" in mismatch.stderr


def test_local_cache_nonzero_rank_waits_for_rank_zero_ready() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        source_root, cache_root, manifest, _ = _one_file_cache_fixture(tmp_path)
        base_env = _cache_environment(source_root, manifest, cache_root)
        waiter_env = base_env.copy()
        waiter_env["FASTWAM_NODE_LOCAL_RANK"] = "1"
        builder_env = base_env.copy()
        builder_env["FASTWAM_NODE_LOCAL_RANK"] = "0"

        waiter = subprocess.Popen(
            ["bash", str(CACHE_SCRIPT)],
            cwd=REPO_ROOT,
            env=waiter_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.2)
        builder = subprocess.run(["bash", str(CACHE_SCRIPT)], cwd=REPO_ROOT, env=builder_env, text=True, capture_output=True, check=False)
        waiter_stdout, waiter_stderr = waiter.communicate(timeout=5)

        assert builder.returncode == 0, builder.stderr
        assert waiter.returncode == 0, waiter_stderr
        assert Path(waiter_stdout.strip()) == Path(builder.stdout.strip())
        assert "action=wait" in waiter_stderr


def test_local_cache_reaps_dead_local_owner_lock() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        source_root, cache_root, manifest, _ = _one_file_cache_fixture(tmp_path)
        env = _cache_environment(source_root, manifest, cache_root)
        cache_root.mkdir()
        manifest_sha = _sha256(manifest)
        lock = cache_root / f".{manifest_sha}.LOCK"
        lock.mkdir()
        (lock / "owner").write_text(
            f"pid=99999999\nhostname={socket.gethostname()}\nstarted_epoch=1\n",
            encoding="utf-8",
        )

        result = subprocess.run(["bash", str(CACHE_SCRIPT)], cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "action=reap_stale_lock reason=dead_local_owner" in result.stderr
        assert Path(result.stdout.strip(), "payload.bin").is_file()


def test_local_cache_recovers_after_builder_is_killed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        source_root, cache_root, manifest, _ = _one_file_cache_fixture(tmp_path)
        env = _cache_environment(source_root, manifest, cache_root)
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        real_cp = shutil.which("cp")
        assert real_cp is not None
        _write_executable(fake_bin / "cp", f"#!/usr/bin/env bash\nsleep 30\nexec {real_cp} \"$@\"\n")
        killed_env = env.copy()
        killed_env["PATH"] = f"{fake_bin}:{env['PATH']}"
        process = subprocess.Popen(
            ["bash", str(CACHE_SCRIPT)],
            cwd=REPO_ROOT,
            env=killed_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        lock = cache_root / f".{_sha256(manifest)}.LOCK"
        deadline = time.monotonic() + 5
        while not (lock / "owner").is_file() and time.monotonic() < deadline:
            if process.poll() is not None:
                break
            time.sleep(0.05)
        assert (lock / "owner").is_file(), process.communicate(timeout=1)[1]
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=2)
        assert process.returncode == -signal.SIGKILL

        recovered = subprocess.run(["bash", str(CACHE_SCRIPT)], cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
        assert recovered.returncode == 0, recovered.stderr
        assert "action=reap_stale_lock reason=dead_local_owner" in recovered.stderr
        assert Path(recovered.stdout.strip(), "payload.bin").is_file()


def test_dlc_preflight_builds_bounded_same_port_all_reduce_command() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        fake_bin = tmp_path / "bin"
        host_lib = tmp_path / "host-lib"
        shim_dir = tmp_path / "shim"
        cuda_lib = tmp_path / "cuda-lib"
        fake_bin.mkdir()
        host_lib.mkdir()
        cuda_lib.mkdir()
        (host_lib / "libcuda.so.570.test").write_bytes(b"fake cuda")
        (host_lib / "libnvidia-ml.so.570.test").write_bytes(b"fake nvml")
        _write_executable(fake_bin / "nvidia-smi", "#!/usr/bin/env bash\nprintf 'GPU 0: fake-host570\\n'\n")
        command_log = tmp_path / "commands.log"
        fake_python = fake_bin / "python"
        _write_executable(fake_python, "#!/usr/bin/env bash\nprintf 'PYTHON %s\\n' \"$*\" >> \"$FASTWAM_PREFLIGHT_TEST_LOG\"\n")
        fake_timeout = _install_timeout_mock(fake_bin, command_log)

        env = _base_env()
        env.update(
            {
                "PATH": f"{fake_bin}:{env['PATH']}",
                "FASTWAM_PYTHON": str(fake_python),
                "FASTWAM_TIMEOUT_BIN": str(fake_timeout),
                "FASTWAM_PREFLIGHT_TEST_LOG": str(command_log),
                "FASTWAM_NVIDIA_HOST570_FIX": "1",
                "FASTWAM_NVIDIA_DRIVER_VERSION": "570.test",
                "FASTWAM_NVIDIA_HOST_LIB_DIR": str(host_lib),
                "FASTWAM_NVIDIA_SHIM_DIR": str(shim_dir),
                "FASTWAM_CUDA_LIB_DIR": str(cuda_lib),
            }
        )
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; fastwam_run_dlc_preflight 8 4 2 10.20.30.40 29400 launcher-test; '
                'printf "EXPORTED_LD_LIBRARY_PATH=%s\\n" "$LD_LIBRARY_PATH"',
                "preflight-test",
                str(PREFLIGHT_SCRIPT),
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert (shim_dir / "libcuda.so.1").resolve() == host_lib / "libcuda.so.570.test"
        assert (shim_dir / "libnvidia-ml.so.1").resolve() == host_lib / "libnvidia-ml.so.570.test"
        assert f"EXPORTED_LD_LIBRARY_PATH={shim_dir}:{cuda_lib}" in result.stdout
        commands = command_log.read_text(encoding="utf-8").splitlines()
        python_commands = [line for line in commands if line.startswith("PYTHON ")]
        timeout_commands = [line for line in commands if line.startswith("TIMEOUT ")]
        assert len(python_commands) == 3
        assert len(timeout_commands) == 1
        assert "validate_python_environment.py --pyproject" in python_commands[0]
        assert "--pip-check-timeout 120" in python_commands[0]
        assert "validate_cuda_devices.py --expected 8" in python_commands[1]
        assert "--foreground --signal=TERM --kill-after=15s 240s" in timeout_commands[0]
        assert "-m torch.distributed.run --nnodes 4" in python_commands[2]
        assert "--nproc-per-node 8" in python_commands[2]
        assert "--node-rank 2" in python_commands[2]
        assert "--master-addr 10.20.30.40 --master-port 29400" in python_commands[2]
        assert "--rdzv-backend static --rdzv-id launcher-test-preflight" in python_commands[2]
        assert "--rdzv-conf timeout=180" in python_commands[2]
        assert "validate_distributed_cuda.py --expected-world-size 32" in python_commands[2]
        assert "--bandwidth-mib 256" in python_commands[2]
        assert "--bandwidth-warmup 2" in python_commands[2]
        assert "--bandwidth-iters 5" in python_commands[2]
        assert "--min-algbw-gbps 5.0" in python_commands[2]


def test_dlc_preflight_erdma_transport_parser_rejects_known_fallbacks() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        good = tmp_path / "good.log"
        no_device = tmp_path / "no-device.log"
        socket_fallback = tmp_path / "socket.log"
        missing = tmp_path / "missing.log"
        good.write_text("node: NCCL INFO NET/IB : Using [0]erdma_0:1/RoCE\n", encoding="utf-8")
        no_device.write_text("node: NCCL INFO NET/IB : No device found.\n", encoding="utf-8")
        socket_fallback.write_text("node: NCCL INFO NET/Socket : Using [0]eth0\n", encoding="utf-8")
        missing.write_text("node: NCCL INFO Bootstrap : Using eth0\n", encoding="utf-8")

        for path, status, message in (
            (good, 0, "transport=erdma status=PASS"),
            (no_device, 1, "No device found"),
            (socket_fallback, 1, "Socket transport fallback"),
            (missing, 1, "no NET/IB eRDMA transport evidence"),
        ):
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; _fastwam_validate_nccl_transport_log "$2"',
                    "transport-test",
                    str(PREFLIGHT_SCRIPT),
                    str(path),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            assert result.returncode == status, (path, result.stderr)
            assert message in (result.stdout + result.stderr)


def test_formal_erdma_hook_is_nonlogin_ordered_and_inherited() -> None:
    launcher = TRAIN_SCRIPT.read_text(encoding="utf-8")
    cache_position = launcher.index("fastwam_prepare_multi_source_cache")
    source_position = launcher.index('source "${ERDMA_BOOTSTRAP_SCRIPT}"')
    prepare_position = launcher.index("fastwam_prepare_erdma_userspace", source_position)
    collective_position = launcher.index("fastwam_run_global_allreduce_preflight", prepare_position)
    exec_position = launcher.index('exec "${ACCELERATE_COMMAND[@]}"', collective_position)
    assert cache_position < source_position < prepare_position < collective_position < exec_position
    assert 'export NCCL_IB_HCA="${NCCL_IB_HCA:-erdma}"' in launcher
    assert "bash -l" not in launcher


def test_dlc_preflight_propagates_distributed_failure_status() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        command_log = tmp_path / "commands.log"
        fake_python = fake_bin / "python"
        _write_executable(
            fake_python,
            "#!/usr/bin/env bash\n"
            "printf 'PYTHON %s\\n' \"$*\" >> \"$FASTWAM_PREFLIGHT_TEST_LOG\"\n"
            "[[ \"$*\" == *torch.distributed.run* ]] && exit 42\n"
            "exit 0\n",
        )
        fake_timeout = _install_timeout_mock(fake_bin, command_log)
        env = _base_env()
        env.update(
            {
                "PATH": f"{fake_bin}:{env['PATH']}",
                "FASTWAM_PYTHON": str(fake_python),
                "FASTWAM_TIMEOUT_BIN": str(fake_timeout),
                "FASTWAM_PREFLIGHT_TEST_LOG": str(command_log),
            }
        )
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; fastwam_run_global_allreduce_preflight 8 4 2 10.20.30.40 29400 launcher-test',
                "preflight-failure-test",
                str(PREFLIGHT_SCRIPT),
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 42
        assert "status=42" in result.stderr
        assert "status=PASS" not in result.stdout


def test_launcher_startup_order_is_local_cache_global_then_exec() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        source_root, cache_root, manifest, _ = _one_file_cache_fixture(tmp_path)
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        command_log = tmp_path / "commands.log"
        _write_executable(fake_bin / "nvidia-smi", "#!/usr/bin/env bash\nprintf '[mock_nvidia_smi]\\n'\n")
        fake_python = fake_bin / "python"
        _write_executable(
            fake_python,
            "#!/usr/bin/env bash\n"
            "if [[ \"$1\" == -m && \"$2\" == accelerate.commands.launch ]]; then\n"
            "  shift 2\n"
            "  printf '[mock_accelerate] %s\\n' \"$*\"\n"
            "else\n"
            "  printf '[mock_python] %s\\n' \"$*\"\n"
            "fi\n",
        )
        fake_timeout = _install_timeout_mock(fake_bin, command_log)
        env = _cache_environment(source_root, manifest, cache_root)
        env.update(
            {
                "PATH": f"{fake_bin}:{env['PATH']}",
                "WORLD_SIZE": "4",
                "RANK": "0",
                "NPROC_PER_NODE": "8",
                "MASTER_ADDR": "10.20.30.40",
                "MASTER_PORT": "29400",
                "RUN_ID": "startup-order-test",
                "FASTWAM_DLC_PREFLIGHT": "1",
                "FASTWAM_LOCAL_CACHE_ENABLED": "1",
                "FASTWAM_PYTHON": str(fake_python),
                "FASTWAM_TIMEOUT_BIN": str(fake_timeout),
                "FASTWAM_PREFLIGHT_TEST_LOG": str(command_log),
                "FASTWAM_NVIDIA_HOST570_FIX": "0",
                "FASTWAM_LAUNCHER_UNIT_TEST_SKIP_ENV_PREFLIGHT": None,
            }
        )
        result = _run_launcher(nproc=None, env_updates=env, dry_run=False, merge_output=True)

        assert result.returncode == 0, result.stdout
        output = result.stdout
        positions = [
            output.index("stage=python_environment"),
            output.index("[nvidia_host_fix]"),
            output.index("stage=nvidia_smi"),
            output.index("stage=torch_devices"),
            output.index("[local_cache] status=READY"),
            output.index("stage=distributed_all_reduce"),
            output.index("[mock_accelerate]"),
        ]
        assert positions == sorted(positions), output
        assert "rendezvous=10.20.30.40:29400" in output
        assert "--main_process_port 29400" in output


if __name__ == "__main__":
    tests = [
        test_zero2_launcher_builds_one_32_process_world_from_aliases,
        test_zero2_launcher_accepts_native_pai_topology_without_positional_nproc,
        test_zero2_launcher_forces_standard_and_reuses_documented_port,
        test_zero2_launcher_preserves_single_node_defaults,
        test_zero2_launcher_rejects_invalid_topology,
        test_zero2_launcher_rejects_native_alias_and_nproc_conflicts,
        test_zero2_launcher_requires_safe_explicit_multinode_run_id,
        test_formal_32gpu_scale_is_fail_closed_even_in_dry_run,
        test_formal_32gpu_cli_allowlist_seals_treatment_and_schedule,
        test_formal_non_dry_rejects_test_bypasses_before_git_or_python,
        test_formal_non_dry_requires_clean_checkout_and_exact_env_preflight,
        test_formal_image_digest_is_mandatory_for_execution_but_ack_is_dry_run_only,
        test_formal_gau0_has_no_gaussian_oss_asset_dependency,
        test_zero2_launcher_execs_accelerate_and_propagates_status,
        test_python_environment_preflight_accepts_exact_pyproject_versions,
        test_python_environment_preflight_rejects_version_drift,
        test_python_environment_preflight_rejects_missing_critical_package,
        test_python_environment_preflight_propagates_pip_check_failure,
        test_whole_file_manifest_generator_is_deterministic_atomic_and_safe,
        test_run_reservation_is_exclusive_and_identity_exact,
        test_zero_checkpoint_smoke_evidence_is_hash_and_filesystem_bound,
        test_zero_smoke_runner_clears_pai_topology_and_initializes_accelerator_first,
        test_zero_smoke_resolves_fixed_micro_batch_without_a_dataloader,
        test_zero_smoke_rejects_ambient_accumulation_override,
        test_zero_smoke_rejects_wrong_micro_batch_value,
        test_zero_smoke_requires_deepspeed_plugin,
        test_pod_image_digest_probe_outputs_only_normalized_identity,
        test_runtime_rank_zero_atomic_config_file_barrier,
        test_prepare_local_training_bundle_rewrites_stats_provenance_without_mutating_source,
        test_local_cache_copies_whole_files_maps_gaussian_and_reuses_ready,
        test_multi_source_cache_combines_cpfs_and_oss_mappings,
        test_offline_environment_bootstrap_is_content_addressed_and_zstd_free,
        test_offline_env_tmp_roots_reject_traversal_and_symlink_chain,
        test_offline_env_forces_node_local_builder_rank_zero,
        test_offline_env_binds_exact_cpython310_identity,
        test_offline_env_import_origin_is_exact_checkout,
        test_offline_env_concurrent_checkout_builds_once_and_waits,
        test_offline_env_checkout_failure_wakes_waiter_and_cleans,
        test_offline_env_fresh_ownerless_lock_is_not_stale,
        test_offline_env_checkout_hit_waits_for_active_lock,
        test_offline_env_prepare_venv_stdout_is_only_destination,
        test_offline_env_venv_ready_is_published_after_runtime_gate,
        test_offline_env_concurrent_venv_builds_once_and_waits,
        test_offline_env_venv_failure_wakes_waiter_and_cleans,
        test_offline_env_rejects_ignored_checkout_artifacts,
        test_offline_env_waiter_recovers_hard_killed_builder_and_staging,
        test_offline_env_post_publish_owner_kill_does_not_replace_checkout,
        test_offline_env_venv_rechecks_ready_after_acquiring_lock,
        test_local_cache_default_hit_verification_fails_closed_on_corruption,
        test_local_cache_rejects_parent_traversal_and_manifest_symlink,
        test_local_cache_rejects_mapping_escape_and_manifest_identity_mismatch,
        test_local_cache_nonzero_rank_waits_for_rank_zero_ready,
        test_local_cache_reaps_dead_local_owner_lock,
        test_local_cache_recovers_after_builder_is_killed,
        test_dlc_preflight_builds_bounded_same_port_all_reduce_command,
        test_dlc_preflight_erdma_transport_parser_rejects_known_fallbacks,
        test_formal_erdma_hook_is_nonlogin_ordered_and_inherited,
        test_dlc_preflight_propagates_distributed_failure_status,
        test_launcher_startup_order_is_local_cache_global_then_exec,
    ]
    for test in tests:
        test()
    print(f"DLC_LAUNCHER_TESTS=PASS tests={len(tests)}")
