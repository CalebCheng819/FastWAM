#!/usr/bin/env python3
"""Run official FastWAM LIBERO evaluation with external G0 observers.

Security boundary: the manager verifies the instrumentation commit/tree and
passes the exact validated instrumentation bytes through a fixed, fully sealed
memfd.  It also monitors both source roots with local recursive inotify until
terminal publication.  This closes ordinary same-host, same-UID pathname
rename/swap races, but is not a claim about CPFS writes from another node,
ptrace, or root.

Runner-only settings are environment variables.  Command-line arguments are
restricted to the formal ``key=value`` override allowlist, composed against
the verified official config directory, and passed to the undecorated official
evaluation function.  This is necessary because Hydra 1.3 does not propagate
the decorated task function's return value.

* ``MF_WAM_OFFICIAL_ROOT``: pristine official FastWAM checkout (required)
* ``MF_WAM_OFFICIAL_COMMIT``: expected official commit (defaults to release)
* ``MF_WAM_G0_RUN_ID``: preregistered candidate run ID (required)
* ``MF_WAM_INSTRUMENTATION_COMMIT``: required clean observer/runner commit
* ``MF_WAM_G0_PREREG_PATH`` / ``MF_WAM_G0_PREREG_SHA256`` (required)
* ``MF_WAM_G0_RUNTIME_START_PATH`` / ``MF_WAM_G0_RUNTIME_START_SHA256`` (required)
* ``MF_WAM_G0_SEED_SCHEDULE_PATH`` / ``MF_WAM_G0_SEED_SCHEDULE_SHA256`` (required)
* ``MF_WAM_G0_RESOLVED_CONFIG_PATH`` / ``MF_WAM_G0_RESOLVED_CONFIG_SHA256`` (required)

The preregistered artifact root fixes all worker-owned paths: ``results/``,
``trace_receipts/``, and ``traces/``.  No independent trace-root override is
accepted by the production runner.
"""

from __future__ import annotations

import json
import fcntl
import hashlib
import os
import stat
import subprocess
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Defense in depth for direct invocations; formal workers also receive
# PYTHONDONTWRITEBYTECODE=1 in the exact 30-key environment.
sys.dont_write_bytecode = True

FORMAL_OVERRIDE_KEYS = frozenset(
    (
        "task",
        "ckpt",
        "gpu_id",
        "seed",
        "output_dir",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
        "EVALUATION.output_dir",
        "EVALUATION.dataset_stats_path",
        "EVALUATION.num_trials",
        "EVALUATION.env_num",
        "EVALUATION.num_steps_wait",
        "EVALUATION.replan_steps",
        "EVALUATION.binarize_gripper",
        "EVALUATION.use_action_ensembler",
        "EVALUATION.visualize_future_video",
        "EVALUATION.action_horizon",
    )
)
DYNAMIC_RUNTIME_OVERLAY_KEYS = frozenset(
    ("gpu_id", "EVALUATION.task_suite_name", "EVALUATION.task_id")
)
INSTRUMENTATION_MEMFD_FD = 198
_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
_MEMFD_SEALS = (
    getattr(fcntl, "F_SEAL_SEAL", 1)
    | getattr(fcntl, "F_SEAL_SHRINK", 2)
    | getattr(fcntl, "F_SEAL_GROW", 4)
    | getattr(fcntl, "F_SEAL_WRITE", 8)
)

FIXED_WORKER_ENVIRONMENT = {
    "HOME": "/tmp",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONUTF8": "1",
    "TMPDIR": "/tmp",
    "TZ": "UTC",
}
FORMAL_ENVIRONMENT_KEYS = frozenset(
    (
        *FIXED_WORKER_ENVIRONMENT,
        "CUDA_VISIBLE_DEVICES",
        "DIFFSYNTH_DOWNLOAD_SOURCE",
        "DIFFSYNTH_MODEL_BASE_PATH",
        "DIFFSYNTH_SKIP_DOWNLOAD",
        "LOCAL_RANK",
        "MF_WAM_G0_PREREG_PATH",
        "MF_WAM_G0_PREREG_SHA256",
        "MF_WAM_G0_RESOLVED_CONFIG_PATH",
        "MF_WAM_G0_RESOLVED_CONFIG_SHA256",
        "MF_WAM_G0_RUN_ID",
        "MF_WAM_G0_RUNTIME_START_PATH",
        "MF_WAM_G0_RUNTIME_START_SHA256",
        "MF_WAM_G0_SEED_SCHEDULE_PATH",
        "MF_WAM_G0_SEED_SCHEDULE_SHA256",
        "MF_WAM_INSTRUMENTATION_COMMIT",
        "MF_WAM_OFFICIAL_COMMIT",
        "MF_WAM_OFFICIAL_ROOT",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "PYTHONHASHSEED",
        "RANK",
        "WORLD_SIZE",
    )
)


class TracedRunnerError(RuntimeError):
    """Raised when the traced worker cannot prove its launch contract."""


def _load_instrumentation_api() -> Any:
    """Load only the manager-validated bytes from the inherited sealed memfd."""

    try:
        try:
            metadata = os.fstat(INSTRUMENTATION_MEMFD_FD)
            seals = fcntl.fcntl(INSTRUMENTATION_MEMFD_FD, _F_GET_SEALS)
        except OSError as exc:
            raise TracedRunnerError(
                "sealed instrumentation memfd is absent or unreadable"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise TracedRunnerError("sealed instrumentation memfd has invalid metadata")
        if seals != _MEMFD_SEALS:
            raise TracedRunnerError("instrumentation memfd seal set is incomplete")
        raw = os.pread(INSTRUMENTATION_MEMFD_FD, metadata.st_size + 1, 0)
        terminal = os.fstat(INSTRUMENTATION_MEMFD_FD)
        if (
            len(raw) != metadata.st_size
            or terminal.st_size != metadata.st_size
            or (terminal.st_dev, terminal.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise TracedRunnerError("instrumentation memfd changed during readback")
    finally:
        try:
            os.close(INSTRUMENTATION_MEMFD_FD)
        except OSError:
            pass

    instrumentation_root = Path(__file__).resolve().parents[1]
    expected_git_dir = instrumentation_root / ".git"
    try:
        git_dir_fd = os.open(
            expected_git_dir,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise TracedRunnerError(
            "instrumentation .git must be a real in-tree directory"
        ) from exc
    try:
        info_fd = os.open(
            "info",
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=git_dir_fd,
        )
    except OSError as exc:
        os.close(git_dir_fd)
        raise TracedRunnerError(
            "instrumentation .git/info must be a real in-tree directory"
        ) from exc
    try:
        try:
            os.stat("attributes", dir_fd=info_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise TracedRunnerError(
                "instrumentation .git/info/attributes is forbidden"
            )
    finally:
        os.close(info_fd)
        os.close(git_dir_fd)
    expected_commit = os.environ.get("MF_WAM_INSTRUMENTATION_COMMIT", "")
    if len(expected_commit) != 40 or any(
        character not in "0123456789abcdef" for character in expected_commit
    ):
        raise TracedRunnerError("instrumentation commit is not an exact 40-hex identity")
    git_environment = {
        **FIXED_WORKER_ENVIRONMENT,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    base_command = [
        "/usr/bin/git",
        "-c",
        f"safe.directory={instrumentation_root}",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-C",
        str(instrumentation_root),
    ]
    try:
        local_config_keys = subprocess.run(
            [
                *base_command,
                "config",
                "--local",
                "--no-includes",
                "--name-only",
                "--list",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=git_environment,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise TracedRunnerError(
            f"cannot inspect instrumentation repository-local config: {exc}"
        ) from exc
    forbidden_config = [
        key
        for key in local_config_keys
        if key.lower().startswith(("filter.", "include.", "includeif."))
        or key.lower() == "core.attributesfile"
    ]
    if forbidden_config:
        raise TracedRunnerError(
            "instrumentation repository-local filters/includes are forbidden"
        )
    try:
        marker = subprocess.run(
            [
                *base_command,
                "ls-files",
                "-v",
                "--",
                "scripts/mf_wam_g0_instrumentation.py",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=git_environment,
        ).stdout.strip()
        git_blob = subprocess.run(
            [
                *base_command,
                "cat-file",
                "blob",
                f"{expected_commit}:scripts/mf_wam_g0_instrumentation.py",
            ],
            check=True,
            capture_output=True,
            timeout=30,
            env=git_environment,
        ).stdout
        top_level = subprocess.run(
            [*base_command, "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=git_environment,
        ).stdout.strip()
        absolute_git_dir = subprocess.run(
            [*base_command, "rev-parse", "--absolute-git-dir"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=git_environment,
        ).stdout.strip()
        common_git_dir = subprocess.run(
            [
                *base_command,
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=git_environment,
        ).stdout.strip()
        object_format = subprocess.run(
            [*base_command, "rev-parse", "--show-object-format"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=git_environment,
        ).stdout.strip()
        replacements = subprocess.run(
            [
                *base_command,
                "for-each-ref",
                "--format=%(refname)",
                "refs/replace/",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=git_environment,
        ).stdout.splitlines()
        ignored = subprocess.run(
            [
                *base_command,
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
            timeout=30,
            env=git_environment,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise TracedRunnerError(
            f"cannot bind sealed instrumentation memfd to exact Git blob: {exc}"
        ) from exc
    if marker != "H scripts/mf_wam_g0_instrumentation.py":
        raise TracedRunnerError(
            "instrumentation source has assume-unchanged/skip-worktree flags"
        )
    if raw != git_blob:
        raise TracedRunnerError(
            "sealed instrumentation memfd differs from exact commit blob"
        )
    if top_level != str(instrumentation_root):
        raise TracedRunnerError(
            "instrumentation Git top-level differs from its source root"
        )
    if absolute_git_dir != str(expected_git_dir) or common_git_dir != str(
        expected_git_dir
    ):
        raise TracedRunnerError(
            "instrumentation linked worktree/external Git directory is forbidden"
        )
    if object_format != "sha1":
        raise TracedRunnerError("instrumentation Git object format must be sha1")
    if replacements:
        raise TracedRunnerError("instrumentation Git replace refs are forbidden")
    if ignored:
        if not ignored.endswith(b"\0"):
            raise TracedRunnerError(
                "instrumentation ignored-file inventory is not NUL terminated"
            )
        raise TracedRunnerError(
            "instrumentation source root contains gitignored artifacts"
        )
    module_name = "mf_wam_g0_instrumentation"
    module = types.ModuleType(module_name)
    module.__file__ = f"/proc/self/fd/{INSTRUMENTATION_MEMFD_FD}"
    module.__package__ = ""
    module.__loader__ = None
    module.__sealed_source_sha256__ = hashlib.sha256(raw).hexdigest()
    module.__sealed_source_size_bytes__ = len(raw)
    module.__sealed_source_seals__ = seals
    sys.modules[module_name] = module
    try:
        code = compile(raw, module.__file__, "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise TracedRunnerError(f"required environment variable is absent: {name}")
    return value


def _validate_formal_process_environment() -> str:
    actual = dict(os.environ)
    if set(actual) != FORMAL_ENVIRONMENT_KEYS:
        missing = sorted(FORMAL_ENVIRONMENT_KEYS - set(actual))
        unexpected = sorted(set(actual) - FORMAL_ENVIRONMENT_KEYS)
        raise TracedRunnerError(
            "worker environment differs from the sealed minimal allowlist; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    if any(not isinstance(value, str) for value in actual.values()):
        raise TracedRunnerError("worker environment contains a non-string value")
    if any(actual[key] != value for key, value in FIXED_WORKER_ENVIRONMENT.items()):
        raise TracedRunnerError("worker fixed process environment is invalid")
    encoded = json.dumps(
        actual,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_hydra_overrides(argv: Sequence[str]) -> list[str]:
    """Accept exactly one plain ``key=value`` override per formal key."""

    overrides: list[str] = []
    observed: set[str] = set()
    for argument in argv:
        if (
            not isinstance(argument, str)
            or "=" not in argument
            or argument.startswith("-")
            or any(character in argument for character in ("\x00", "\r", "\n"))
        ):
            raise TracedRunnerError(
                f"only controlled Hydra key=value overrides are accepted: {argument!r}"
            )
        key, value = argument.split("=", 1)
        if key not in FORMAL_OVERRIDE_KEYS:
            raise TracedRunnerError(f"unsupported Hydra override key: {key!r}")
        if key in observed:
            raise TracedRunnerError(f"duplicate Hydra override key: {key!r}")
        if value == "":
            raise TracedRunnerError(f"empty Hydra override value: {key!r}")
        observed.add(key)
        overrides.append(argument)
    missing = sorted(FORMAL_OVERRIDE_KEYS - observed)
    if missing:
        raise TracedRunnerError(f"missing formal Hydra overrides: {missing!r}")
    return overrides


def _compose_official_config(official_root: Path, argv: Sequence[str]) -> Any:
    """Compose the official config without invoking Hydra's decorated main."""

    overrides = _validated_hydra_overrides(argv)
    try:
        from hydra import compose, initialize_config_dir
    except (ImportError, ModuleNotFoundError) as exc:
        raise TracedRunnerError("hydra-core is required for the traced worker") from exc

    config_dir = official_root / "configs"
    try:
        with initialize_config_dir(
            version_base="1.3",
            config_dir=str(config_dir),
            job_name="mf_wam_g0_traced_worker",
        ):
            return compose(config_name="sim_libero", overrides=overrides)
    except Exception as exc:
        raise TracedRunnerError(f"official Hydra config composition failed: {exc}") from exc


def _assert_no_locked_interpolation(node: Any, *, location: str) -> None:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - caller imports too.
        raise TracedRunnerError(
            "omegaconf is required for resolved config binding"
        ) from exc
    if isinstance(node, DictConfig):
        keys: Sequence[str | int] = list(node.keys())
    elif isinstance(node, ListConfig):
        keys = list(range(len(node)))
    else:
        return
    for key in keys:
        child_location = f"{location}.{key}"
        if OmegaConf.is_interpolation(node, key):
            raise TracedRunnerError(
                f"locked resolved config contains interpolation at {child_location}"
            )
        _assert_no_locked_interpolation(
            node._get_node(key),  # noqa: SLF001 - avoid resolving before rejection.
            location=child_location,
        )


def _validate_locked_resolved_config(
    cfg: Any,
    resolved_config_path: Path,
    argv: Sequence[str],
) -> None:
    """Bind a live worker config to one locked base plus sealed CLI overlay.

    The locked YAML is a fully resolved base config.  The formal command/status
    receipt seals the complete command.  Only GPU ID, suite, and task ID may
    differ across the 40 workers; every other resolved value (including both
    output directories) must remain exactly equal as a JSON value.
    """

    overrides = _validated_hydra_overrides(argv)
    override_values = dict(argument.split("=", 1) for argument in overrides)
    if override_values["task"] != "libero_uncond_2cam224_1e-4":
        raise TracedRunnerError("formal Hydra task config override is invalid")
    try:
        from omegaconf import OmegaConf
    except (ImportError, ModuleNotFoundError) as exc:
        raise TracedRunnerError("omegaconf is required for resolved config binding") from exc
    try:
        locked = OmegaConf.load(resolved_config_path)
        _assert_no_locked_interpolation(locked, location="locked")
        locked_value = OmegaConf.to_container(
            locked,
            resolve=True,
            throw_on_missing=True,
        )
        live_value = OmegaConf.to_container(
            cfg,
            resolve=True,
            throw_on_missing=True,
        )
        expected = OmegaConf.create(locked_value)
        missing = object()
        for key in sorted(DYNAMIC_RUNTIME_OVERLAY_KEYS):
            if OmegaConf.select(expected, key, default=missing) is missing:
                raise TracedRunnerError(
                    f"locked resolved base lacks runtime overlay path: {key}"
                )
            live_overlay_value = OmegaConf.select(cfg, key, default=missing)
            if live_overlay_value is missing:
                raise TracedRunnerError(
                    f"composed config lacks runtime overlay path: {key}"
                )
            OmegaConf.update(
                expected,
                key,
                live_overlay_value,
                merge=False,
                force_add=False,
            )
        expected_value = OmegaConf.to_container(
            expected,
            resolve=True,
            throw_on_missing=True,
        )
    except TracedRunnerError:
        raise
    except Exception as exc:
        raise TracedRunnerError(f"cannot resolve locked/live Hydra config: {exc}") from exc
    if live_value != expected_value:
        raise TracedRunnerError(
            "composed Hydra config differs from locked base plus runtime overlay"
        )


def _call_undecorated_official_eval(official_module: Any, cfg: Any) -> Mapping[str, Any]:
    decorated = getattr(official_module, "eval_single_process", None)
    undecorated = getattr(decorated, "__wrapped__", None)
    if not callable(undecorated):
        raise TracedRunnerError(
            "official eval_single_process lacks the Hydra __wrapped__ task function"
        )
    result = undecorated(cfg)
    if not isinstance(result, Mapping):
        raise TracedRunnerError(
            "official undecorated evaluator must return a result Mapping"
        )
    return result


def _main(instrumentation: Any, environment_sha256: str) -> int:
    runner_path = Path(__file__).resolve()
    instrumentation_root = runner_path.parents[1]
    official_root = Path(_required_environment("MF_WAM_OFFICIAL_ROOT"))
    expected_commit = os.environ.get(
        "MF_WAM_OFFICIAL_COMMIT", instrumentation.OFFICIAL_FASTWAM_COMMIT
    )
    instrumentation_commit = _required_environment("MF_WAM_INSTRUMENTATION_COMMIT")
    run_id = _required_environment("MF_WAM_G0_RUN_ID")
    if "MF_WAM_G0_TRACE_ROOT" in os.environ:
        raise TracedRunnerError(
            "MF_WAM_G0_TRACE_ROOT is forbidden; preregistration fixes artifact_root/traces"
        )

    official_identity = instrumentation.verify_pristine_official_root(
        official_root,
        expected_commit=expected_commit,
    )
    instrumentation_identity = instrumentation.verify_pristine_instrumentation_root(
        instrumentation_root,
        expected_commit=instrumentation_commit,
    )
    resolved_config_path = Path(
        _required_environment("MF_WAM_G0_RESOLVED_CONFIG_PATH")
    )
    upstream_bindings = instrumentation.load_upstream_artifact_bindings(
        run_id=run_id,
        preregistration_path=Path(_required_environment("MF_WAM_G0_PREREG_PATH")),
        preregistration_sha256=_required_environment("MF_WAM_G0_PREREG_SHA256"),
        runtime_start_path=Path(_required_environment("MF_WAM_G0_RUNTIME_START_PATH")),
        runtime_start_sha256=_required_environment("MF_WAM_G0_RUNTIME_START_SHA256"),
        seed_schedule_path=Path(_required_environment("MF_WAM_G0_SEED_SCHEDULE_PATH")),
        seed_schedule_sha256=_required_environment("MF_WAM_G0_SEED_SCHEDULE_SHA256"),
        resolved_config_path=resolved_config_path,
        resolved_config_sha256=_required_environment("MF_WAM_G0_RESOLVED_CONFIG_SHA256"),
    )

    # Import and compose only after the official checkout has passed its
    # Git/readback audit.  Do not call the Hydra-decorated function: Hydra 1.3's
    # wrapper intentionally discards the task function return value.
    official_module = instrumentation.import_pristine_official_eval(official_root)
    cfg = _compose_official_config(official_root, sys.argv[1:])
    _validate_locked_resolved_config(cfg, resolved_config_path, sys.argv[1:])
    tracer = instrumentation.G0TraceInstrumentation(
        official_module,
        official_root=official_root,
        official_identity=official_identity,
        run_id=run_id,
        instrumentation_identity=instrumentation_identity,
        upstream_bindings=upstream_bindings,
    ).install()
    try:
        result = _call_undecorated_official_eval(official_module, cfg)
        result_receipt = tracer.bind_official_task_result(result)
        receipt = tracer.finalize_process()
        terminal_sources = tracer.verify_terminal_source_identities()
    finally:
        tracer.restore()

    print(
        json.dumps(
            {
                "status": "PASS",
                "kind": "mf_wam_g0_traced_worker_terminal",
                "run_id": run_id,
                "process_receipt": str(receipt),
                "official_commit": expected_commit,
                "official_result_type": type(result).__name__,
                "official_result_receipt": result_receipt,
                "terminal_source_identities": terminal_sources,
                "external_prelaunch_commit_tree_gate_required": True,
                "environment_sha256": environment_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    environment_sha256 = _validate_formal_process_environment()
    instrumentation = _load_instrumentation_api()
    try:
        return _main(instrumentation, environment_sha256)
    except instrumentation.InstrumentationError as exc:
        raise TracedRunnerError(str(exc)) from exc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TracedRunnerError as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "kind": "mf_wam_g0_traced_worker_terminal",
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
