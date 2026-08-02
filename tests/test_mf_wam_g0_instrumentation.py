from __future__ import annotations

import json
import hashlib
import os
import random
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    import torch
except ModuleNotFoundError as dependency_error:  # Production FastWAM provides both.
    RUNTIME_DEPS_AVAILABLE = False
    RUNTIME_DEPS_REASON = f"G0 instrumentation tests require FastWAM runtime deps: {dependency_error}"
else:
    RUNTIME_DEPS_AVAILABLE = True
    RUNTIME_DEPS_REASON = ""
    from scripts.mf_wam_g0_instrumentation import (
        G0TraceInstrumentation,
        InstrumentationError,
        _atomic_write_json_no_replace,
        _rng_fingerprint,
        audit_module_origins,
        load_upstream_artifact_bindings,
        sha256_file,
        validate_structured_trace_payload,
        verify_process_receipt_trace_inventory,
        verify_pristine_instrumentation_root,
        verify_pristine_official_root,
    )

from scripts.run_mf_wam_g0_traced import (
    FORMAL_OVERRIDE_KEYS,
    TracedRunnerError,
    _call_undecorated_official_eval,
    _compose_official_config,
    _validate_locked_resolved_config,
    _validated_hydra_overrides,
)

try:
    import hydra
except ModuleNotFoundError:
    HYDRA_132_AVAILABLE = False
    HYDRA_132_REASON = "hydra-core 1.3.2 is not installed in this test environment"
else:
    HYDRA_132_AVAILABLE = getattr(hydra, "__version__", None) == "1.3.2"
    HYDRA_132_REASON = (
        "hydra-core 1.3.2 integration test requires the locked production version"
    )


class AttrDict(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def make_observation(step: int) -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.full((2, 2, 3), step % 255, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((2, 2, 3), (step + 1) % 255, dtype=np.uint8),
        "robot0_eef_pos": np.asarray([step / 100.0, 0.0, 1.0], dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.04, -0.04], dtype=np.float32),
    }


class FakeEnv:
    def __init__(self) -> None:
        self.step_index = 0
        self.seed_value = None
        self.action_objects: list[object] = []
        self.action_values: list[list[float]] = []
        self.result_objects: list[tuple] = []

    def seed(self, seed: int) -> None:
        self.seed_value = int(seed)

    def reset(self) -> None:
        self.step_index = 0

    def set_init_state(self, initial_state):
        del initial_state
        return make_observation(self.step_index)

    def step(self, action):
        self.action_objects.append(action)
        self.action_values.append(np.asarray(action, dtype=np.float64).tolist())
        self.step_index += 1
        result = (make_observation(self.step_index), 0.0, False, {"step": self.step_index})
        self.result_objects.append(result)
        return result


def make_cfg(output_dir: Path) -> AttrDict:
    return AttrDict(
        seed=42,
        gpu_id=0,
        EVALUATION=AttrDict(
            num_steps_wait=30,
            replan_steps=10,
            num_trials=50,
            use_action_ensembler=False,
            binarize_gripper=True,
            task_suite_name="libero_spatial",
            task_id=0,
            output_dir=str(output_dir),
        ),
    )


def make_formal_hydra_overrides(root: Path) -> list[str]:
    return [
        "task=libero_uncond_2cam224_1e-4",
        f"ckpt={root / 'checkpoint.pt'}",
        "gpu_id=3",
        "seed=42",
        f"output_dir={root / 'artifacts'}",
        "EVALUATION.task_suite_name=libero_goal",
        "EVALUATION.task_id=7",
        f"EVALUATION.output_dir={root / 'artifacts'}",
        f"EVALUATION.dataset_stats_path={root / 'dataset_stats.json'}",
        "EVALUATION.num_trials=50",
        "EVALUATION.env_num=1",
        "EVALUATION.num_steps_wait=30",
        "EVALUATION.replan_steps=10",
        "EVALUATION.binarize_gripper=true",
        "EVALUATION.use_action_ensembler=false",
        "EVALUATION.visualize_future_video=false",
        "EVALUATION.action_horizon=32",
    ]


def make_synthetic_upstream(
    artifact_root: Path,
    *,
    run_id: str,
    official_identity: dict,
    instrumentation_identity: dict,
    trial_count: int,
) -> dict:
    digest_values = {
        "preregistration_file_sha256": "a" * 64,
        "preregistration_canonical_sha256": "b" * 64,
        "runtime_start_file_sha256": "c" * 64,
        "runtime_start_canonical_sha256": "d" * 64,
        "seed_schedule_file_sha256": "e" * 64,
        "seed_schedule_canonical_sha256": "f" * 64,
        "resolved_config_sha256": "9" * 64,
    }
    source = {
        "fastwam": {"commit": official_identity["commit"]},
        "instrumentation": {"commit": instrumentation_identity["commit"]},
    }
    process = {
        "process_id": "libero_spatial/task00",
        "task_suite": "libero_spatial",
        "task_id": 0,
        "global_rank": 0,
        "global_seed": 42,
        "environment_seed": 42,
        "environment_seed_scope": "once-before-trial-0",
        "policy_seed": 42,
        "policy_seed_scope": "constant-each-replan-call",
        "python_hash_seed": 42,
        "trial_order": list(range(trial_count)),
        "initial_state_index_rule": "trial_idx",
    }
    episodes = [
        {
            "episode_id": f"libero_spatial/task00/trial{trial_idx:03d}",
            "process_id": "libero_spatial/task00",
            "task_suite": "libero_spatial",
            "task_id": 0,
            "trial_idx": trial_idx,
            "episode_ordinal": trial_idx,
            "initial_state_index": trial_idx,
        }
        for trial_idx in range(trial_count)
    ]
    identities = {
        "preregistration": {
            "path": str(artifact_root / "anchors/preregistration.json"),
            "file_sha256": "a" * 64,
            "size_bytes": 1,
            "canonical_sha256": "b" * 64,
        },
        "runtime_start": {
            "path": str(artifact_root / "anchors/runtime-start.json"),
            "file_sha256": "c" * 64,
            "size_bytes": 1,
            "canonical_sha256": "d" * 64,
        },
        "seed_schedule": {
            "path": str(artifact_root / "anchors/seed-schedule.json"),
            "file_sha256": "e" * 64,
            "size_bytes": 1,
            "canonical_sha256": "f" * 64,
        },
        "resolved_config": {
            "path": str(artifact_root / "anchors/resolved-config.yaml"),
            "file_sha256": "9" * 64,
            "size_bytes": 1,
        },
    }
    return {
        "status": "PASS",
        "run_id": run_id,
        "artifact_root": str(artifact_root),
        "digests": digest_values,
        "identities": identities,
        "documents": {
            "preregistration": {
                "source": source,
                "image": {"digest": "sha256:" + "8" * 64},
            },
            "runtime_start": {"source": source},
            "seed_schedule": {
                "task_processes": [process],
                "episodes": episodes,
            },
        },
    }


def make_fake_eval_module() -> types.ModuleType:
    module = types.ModuleType("synthetic_official_eval")
    module.__file__ = __file__
    module.created_envs = []
    module.issued_action_objects = []
    module.prediction_identity_checks = []
    module.prediction_return_objects = []
    module.last_episode_return = None

    def set_global_seed(seed: int, get_worker_init_fn: bool = False):
        del get_worker_init_fn
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_libero_env(task, resolution, seed, env_num=1):
        del resolution, env_num
        env = FakeEnv()
        env.seed(seed)
        module.created_envs.append(env)
        return env, task.language

    def denormalize_action(action, processor):
        del processor
        return action

    def predict_action_chunk(
        obs,
        task_description,
        model,
        processor,
        cfg,
        *,
        action_horizon,
        input_w,
        input_h,
        model_device,
    ):
        del obs, task_description, model, cfg, input_w, input_h, model_device
        random_component = random.random()
        numpy_component = float(np.random.random())
        torch_component = float(torch.rand(1).item())
        base = random_component + numpy_component + torch_component
        grid = np.arange(action_horizon * 7, dtype=np.float32).reshape(1, action_horizon, 7)
        normalized = grid / 1000.0 + np.float32(base)
        raw = module._denormalize_action(normalized, processor)[0]
        env_action = raw.copy()
        env_action[..., -1] = env_action[..., -1] * 2 - 1
        env_action[..., -1] *= -1
        env_action[..., -1] = np.sign(env_action[..., -1])
        result = (env_action, {"synthetic": True}, None)
        module.prediction_return_objects.append(result)
        return result

    def run_single_episode(
        env,
        initial_state,
        task_description,
        model,
        processor,
        cfg,
        episode_idx,
        *,
        action_horizon,
        input_w,
        input_h,
        model_device,
    ):
        del episode_idx
        env.reset()
        obs = env.set_init_state(initial_state)
        for _ in range(30):
            action = [0, 0, 0, 0, 0, 0, -1]
            module.issued_action_objects.append(action)
            step_result = env.step(action)
            obs = step_result[0]
        for _ in range(7):
            prediction = module._predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            module.prediction_identity_checks.append(
                prediction is module.prediction_return_objects[-1]
            )
            pending = prediction[0][:10].tolist()
            for action in pending:
                module.issued_action_objects.append(action)
                step_result = env.step(action)
                obs = step_result[0]
        result = (False, [], [], None)
        module.last_episode_return = result
        return result

    module.set_global_seed = set_global_seed
    module.get_libero_env = get_libero_env
    module._denormalize_action = denormalize_action
    module._predict_action_chunk = predict_action_chunk
    module.run_single_episode = run_single_episode
    return module


class G0TracedRunnerTest(unittest.TestCase):
    def test_traced_runner_accepts_only_complete_controlled_hydra_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            overrides = make_formal_hydra_overrides(Path(directory))
            self.assertEqual(set(item.split("=", 1)[0] for item in overrides), FORMAL_OVERRIDE_KEYS)
            self.assertEqual(_validated_hydra_overrides(overrides), overrides)

            with self.assertRaisesRegex(TracedRunnerError, "unsupported Hydra override"):
                _validated_hydra_overrides(overrides + ["hydra.run.dir=/tmp/escape"])
            with self.assertRaisesRegex(TracedRunnerError, "duplicate Hydra override"):
                _validated_hydra_overrides(overrides + ["gpu_id=4"])
            with self.assertRaisesRegex(TracedRunnerError, "missing formal Hydra overrides"):
                _validated_hydra_overrides(overrides[:-1])
            with self.assertRaisesRegex(TracedRunnerError, "only controlled Hydra"):
                _validated_hydra_overrides(["--multirun", *overrides])

    def test_traced_runner_calls_wrapped_task_and_requires_mapping_return(self) -> None:
        decorated_calls: list[None] = []
        raw_calls: list[object] = []
        sentinel_cfg = object()

        def decorated_main():
            decorated_calls.append(None)
            return None

        def raw_task(cfg):
            raw_calls.append(cfg)
            return {"total_episodes": 50}

        decorated_main.__wrapped__ = raw_task
        module = types.SimpleNamespace(eval_single_process=decorated_main)
        result = _call_undecorated_official_eval(module, sentinel_cfg)
        self.assertEqual(result, {"total_episodes": 50})
        self.assertEqual(decorated_calls, [])
        self.assertEqual(raw_calls, [sentinel_cfg])

        decorated_main.__wrapped__ = lambda cfg: None
        with self.assertRaisesRegex(TracedRunnerError, "must return a result Mapping"):
            _call_undecorated_official_eval(module, sentinel_cfg)

    @unittest.skipUnless(HYDRA_132_AVAILABLE, HYDRA_132_REASON)
    def test_hydra_132_composes_official_config_for_wrapped_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            from omegaconf import OmegaConf

            repository_root = Path(__file__).resolve().parents[1]
            overrides = make_formal_hydra_overrides(Path(directory))
            cfg = _compose_official_config(
                repository_root,
                overrides,
            )
            self.assertEqual(str(cfg.ckpt), str(Path(directory) / "checkpoint.pt"))
            self.assertEqual(int(cfg.gpu_id), 3)
            self.assertEqual(int(cfg.seed), 42)
            self.assertEqual(str(cfg.output_dir), str(Path(directory) / "artifacts"))
            self.assertEqual(cfg.EVALUATION.task_suite_name, "libero_goal")
            self.assertEqual(int(cfg.EVALUATION.task_id), 7)
            self.assertEqual(int(cfg.EVALUATION.num_trials), 50)
            self.assertEqual(int(cfg.EVALUATION.action_horizon), 32)
            locked_path = Path(directory) / "resolved.yaml"
            OmegaConf.save(config=cfg, f=locked_path, resolve=True)
            _validate_locked_resolved_config(cfg, locked_path, overrides)

            second_overrides = [
                "gpu_id=1" if item.startswith("gpu_id=")
                else "EVALUATION.task_suite_name=libero_spatial"
                if item.startswith("EVALUATION.task_suite_name=")
                else "EVALUATION.task_id=2"
                if item.startswith("EVALUATION.task_id=")
                else item
                for item in overrides
            ]
            second_cfg = _compose_official_config(repository_root, second_overrides)
            _validate_locked_resolved_config(
                second_cfg,
                locked_path,
                second_overrides,
            )
            unsafe_fixed_overrides = [
                "EVALUATION.num_trials=49"
                if item.startswith("EVALUATION.num_trials=")
                else item
                for item in overrides
            ]
            unsafe_fixed_cfg = _compose_official_config(
                repository_root,
                unsafe_fixed_overrides,
            )
            with self.assertRaisesRegex(
                TracedRunnerError,
                "differs from locked base plus runtime overlay",
            ):
                _validate_locked_resolved_config(
                    unsafe_fixed_cfg,
                    locked_path,
                    unsafe_fixed_overrides,
                )
            locked = OmegaConf.load(locked_path)
            locked.mixed_precision = "fp16"
            OmegaConf.save(config=locked, f=locked_path, resolve=True)
            with self.assertRaisesRegex(
                TracedRunnerError,
                "differs from locked base plus runtime overlay",
            ):
                _validate_locked_resolved_config(cfg, locked_path, overrides)

            interpolated = OmegaConf.create(
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            )
            interpolated.output_dir = "${oc.env:HOME}"
            OmegaConf.save(config=interpolated, f=locked_path, resolve=False)
            with self.assertRaisesRegex(
                TracedRunnerError,
                "contains interpolation",
            ):
                _validate_locked_resolved_config(cfg, locked_path, overrides)


@unittest.skipUnless(RUNTIME_DEPS_AVAILABLE, RUNTIME_DEPS_REASON)
class G0InstrumentationTest(unittest.TestCase):

    def test_upstream_artifacts_require_trusted_nofollow_cross_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id = "SYNTHETIC-G0"
            artifact_root = root / run_id
            resolved_path = root / "resolved.yaml"
            resolved_path.write_text("seed: 42\n", encoding="utf-8")
            resolved_sha = hashlib.sha256(resolved_path.read_bytes()).hexdigest()

            processes = []
            episodes = []
            for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
                for task_id in range(10):
                    process_id = f"{suite}/task{task_id:02d}"
                    processes.append({
                        "process_id": process_id,
                        "task_suite": suite,
                        "task_id": task_id,
                        "global_rank": 0,
                        "global_seed": 42,
                        "environment_seed": 42,
                        "environment_seed_scope": "once-before-trial-0",
                        "policy_seed": 42,
                        "policy_seed_scope": "constant-each-replan-call",
                        "python_hash_seed": 42,
                        "trial_order": list(range(50)),
                        "initial_state_index_rule": "trial_idx",
                    })
                    for trial_idx in range(50):
                        episodes.append({
                            "episode_id": f"{process_id}/trial{trial_idx:03d}",
                            "process_id": process_id,
                            "task_suite": suite,
                            "task_id": task_id,
                            "trial_idx": trial_idx,
                            "episode_ordinal": trial_idx,
                            "initial_state_index": trial_idx,
                        })
            schedule = {
                "schema_version": 1,
                "kind": "mf_wam_g0_task_process_seed_schedule",
                "semantics": "one-process-per-task-sequential-trials-v1",
                "seed": 42,
                "python_hash_seed": 42,
                "task_process_count": 40,
                "episode_count": 2000,
                "task_processes": processes,
                "episodes": episodes,
            }

            def canonical_sha(value) -> str:
                return hashlib.sha256(json.dumps(
                    value,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")).hexdigest()

            schedule_path = root / "seed-schedule.json"
            schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
            schedule_file_sha = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
            schedule_canonical_sha = canonical_sha(schedule)
            source = {
                "fastwam": {"commit": "0" * 40},
                "instrumentation": {"commit": "1" * 40},
            }
            prereg = {
                "kind": "mf_wam_g0_preregistration",
                "phase": "PREREGISTERED",
                "run_id": run_id,
                "source": source,
                "artifacts": {
                    "resolved_config": {
                        "sha256": resolved_sha,
                        "size_bytes": resolved_path.stat().st_size,
                    }
                },
                "seeds": {
                    "schedule_canonical_sha256": schedule_canonical_sha,
                    "seed": 42,
                    "python_hash_seed": 42,
                },
                "output": {"artifact_root": str(artifact_root), "overwrite": False},
            }
            prereg_path = root / "preregistration.json"
            prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
            prereg_file_sha = hashlib.sha256(prereg_path.read_bytes()).hexdigest()
            prereg_canonical_sha = canonical_sha(prereg)
            start = {
                "kind": "mf_wam_g0_runtime_start",
                "phase": "STARTED",
                "run_id": run_id,
                "preregistration_canonical_sha256": prereg_canonical_sha,
                "source": source,
                "bindings": {
                    "resolved_config_sha256": resolved_sha,
                    "seed_schedule_canonical_sha256": schedule_canonical_sha,
                },
            }
            start_path = root / "runtime-start.json"
            start_path.write_text(json.dumps(start), encoding="utf-8")
            start_file_sha = hashlib.sha256(start_path.read_bytes()).hexdigest()
            bindings = load_upstream_artifact_bindings(
                run_id=run_id,
                preregistration_path=prereg_path,
                preregistration_sha256=prereg_file_sha,
                runtime_start_path=start_path,
                runtime_start_sha256=start_file_sha,
                seed_schedule_path=schedule_path,
                seed_schedule_sha256=schedule_file_sha,
                resolved_config_path=resolved_path,
                resolved_config_sha256=resolved_sha,
            )
            self.assertEqual(bindings["status"], "PASS")
            self.assertEqual(bindings["digests"]["seed_schedule_canonical_sha256"], schedule_canonical_sha)

            with self.assertRaisesRegex(InstrumentationError, "digest mismatch"):
                load_upstream_artifact_bindings(
                    run_id=run_id,
                    preregistration_path=prereg_path,
                    preregistration_sha256="0" * 64,
                    runtime_start_path=start_path,
                    runtime_start_sha256=start_file_sha,
                    seed_schedule_path=schedule_path,
                    seed_schedule_sha256=schedule_file_sha,
                    resolved_config_path=resolved_path,
                    resolved_config_sha256=resolved_sha,
                )
            resolved_link = root / "resolved-link.yaml"
            resolved_link.symlink_to(resolved_path.name)
            with self.assertRaisesRegex(
                InstrumentationError,
                "without following symlinks|symlink, gitlink",
            ):
                load_upstream_artifact_bindings(
                    run_id=run_id,
                    preregistration_path=prereg_path,
                    preregistration_sha256=prereg_file_sha,
                    runtime_start_path=start_path,
                    runtime_start_sha256=start_file_sha,
                    seed_schedule_path=schedule_path,
                    seed_schedule_sha256=schedule_file_sha,
                    resolved_config_path=resolved_link,
                    resolved_config_sha256=resolved_sha,
                )

    def _run_fake_episode(
        self,
        module,
        cfg,
        *,
        traced: bool,
        trace_root: Path | None = None,
        episode_idx: int = 0,
        expected_trial_count: int = 1,
        bind_result: bool = True,
    ):
        tracer = None
        if traced:
            official_identity = {
                "role": "official_policy_and_evaluator_source",
                "commit": "0" * 40,
                "clean": True,
            }
            instrumentation_identity = {
                "role": "external_observer_and_launcher_source",
                "commit": "1" * 40,
            }
            artifact_root = Path(str(cfg.EVALUATION.output_dir))
            upstream_bindings = make_synthetic_upstream(
                artifact_root,
                run_id="SYNTHETIC-G0",
                official_identity=official_identity,
                instrumentation_identity=instrumentation_identity,
                trial_count=expected_trial_count,
            )
            tracer = G0TraceInstrumentation(
                module,
                official_root=Path("/synthetic/official"),
                official_identity=official_identity,
                trace_root=trace_root,
                run_id="SYNTHETIC-G0",
                instrumentation_identity=instrumentation_identity,
                upstream_bindings=upstream_bindings,
                enforce_module_origins=False,
                _test_expected_trial_count=expected_trial_count,
                _test_terminal_source_verifier=lambda: (
                    official_identity,
                    instrumentation_identity,
                ),
                _test_terminal_upstream_verifier=lambda: upstream_bindings["identities"],
            ).install()
        rank_environment = {
            "PYTHONHASHSEED": "42",
            "WORLD_SIZE": "1",
            "RANK": "0",
            "SLURM_PROCID": "0",
            "LOCAL_RANK": "0",
        }
        with mock.patch.dict(os.environ, rank_environment, clear=False):
            module.set_global_seed(42, get_worker_init_fn=False)
            task = types.SimpleNamespace(
                language="pick the object",
                problem_folder="libero_spatial",
                bddl_file="task0.bddl",
            )
            env, description = module.get_libero_env(task, 256, 42)
            initial_state = np.arange(10, dtype=np.float32)
            result = module.run_single_episode(
                env=env,
                initial_state=initial_state,
                task_description=description,
                model=object(),
                processor=object(),
                cfg=cfg,
                episode_idx=episode_idx,
                action_horizon=32,
                input_w=448,
                input_h=224,
                model_device="cuda",
            )
        if traced and bind_result and expected_trial_count == 1 and episode_idx == 0:
            result_payload = {
                "task_suite": cfg.EVALUATION.task_suite_name,
                "task_id": cfg.EVALUATION.task_id,
                "task_description": description,
                "successes": 0,
                "total_episodes": 1,
                "gpu_id": cfg.gpu_id,
                "success_episodes": [],
                "failure_episodes": [0],
                "start_time": "synthetic",
                "duration": 1.0,
            }
            source_path = (
                Path(str(cfg.EVALUATION.output_dir))
                / cfg.EVALUATION.task_suite_name
                / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"
            )
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(
                json.dumps(result_payload, indent=4) + "\n", encoding="utf-8"
            )
            tracer.bind_official_task_result(result_payload)
        return tracer, env, result

    def test_transparent_trace_preserves_rng_actions_and_return_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_module = make_fake_eval_module()
            baseline_cfg = make_cfg(root / "baseline")
            _, baseline_env, baseline_result = self._run_fake_episode(
                baseline_module, baseline_cfg, traced=False
            )
            baseline_rng = _rng_fingerprint()

            traced_module = make_fake_eval_module()
            traced_cfg = make_cfg(root / "candidate")
            trace_root = root / "candidate" / "traces"
            with mock.patch.dict(
                os.environ,
                {"RANK": "0", "SLURM_PROCID": "0", "LOCAL_RANK": "0"},
                clear=False,
            ):
                tracer, traced_env, traced_result = self._run_fake_episode(
                    traced_module,
                    traced_cfg,
                    traced=True,
                    trace_root=trace_root,
                )
            traced_rng = _rng_fingerprint()

            self.assertIs(traced_result, traced_module.last_episode_return)
            self.assertTrue(all(traced_module.prediction_identity_checks))
            self.assertEqual(baseline_rng, traced_rng)
            self.assertEqual(baseline_env.action_values, traced_env.action_values)
            self.assertIs(baseline_result, baseline_module.last_episode_return)
            self.assertEqual(len(traced_env.action_objects), len(traced_module.issued_action_objects))
            self.assertTrue(
                all(
                    observed is issued
                    for observed, issued in zip(
                        traced_env.action_objects,
                        traced_module.issued_action_objects,
                    )
                )
            )

            trace_path = trace_root / "libero_spatial" / "task00" / "trial000.json"
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["kind"], "mf_wam_g0_structured_trace")
            self.assertFalse(payload["metadata"]["success"])
            self.assertEqual(payload["metadata"]["record_count"], 7)
            self.assertEqual(payload["metadata"]["initial_state_index"], 0)
            seed_contract = payload["metadata"]["seed_contract"]
            self.assertEqual(seed_contract["task_seed"], 42)
            self.assertEqual(seed_contract["environment_seed"], 42)
            self.assertEqual(seed_contract["policy_seed"], 42)
            self.assertEqual(
                seed_contract["environment_seed_scope"],
                "once_per_task_process_before_trial_loop",
            )
            self.assertEqual(
                seed_contract["episode_rng_position"],
                "ordered_trial_index_in_shared_task_environment_stream",
            )
            self.assertGreater(payload["metadata"]["observer_rng_unchanged_checks"], 0)

            for replan_idx, record in enumerate(payload["records"]):
                self.assertEqual(record["env_step"], 30 + replan_idx * 10)
                self.assertEqual(np.asarray(record["state"]).shape, (8,))
                self.assertEqual(np.asarray(record["pre_state"]).shape, (8,))
                self.assertEqual(
                    np.asarray(record["proposed_raw_action_chunk"]).shape,
                    (32, 7),
                )
                self.assertEqual(
                    np.asarray(record["proposed_env_action_chunk"]).shape,
                    (32, 7),
                )
                self.assertNotIn("raw_action_chunk", record)
                self.assertEqual(np.asarray(record["executed_env_actions"]).shape, (10, 7))
                self.assertEqual(len(record["executions"]), 10)
                self.assertTrue(
                    np.array_equal(
                        np.asarray(record["executed_env_actions"]),
                        np.asarray(record["proposed_env_action_chunk"])[:10],
                    )
                )
                self.assertFalse(record["done_after_execution"])
                self.assertTrue(
                    all("post_state" in execution for execution in record["executions"])
                )
                self.assertTrue(
                    all(
                        "post_observation_sha256" in execution
                        for execution in record["executions"]
                    )
                )
            receipt_path = tracer.finalize_process()
            self.assertEqual(
                receipt_path,
                root / "candidate/trace_receipts/libero_spatial/task00.json",
            )
            receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(receipt_payload),
                {
                    "schema_version", "kind", "run_id", "process_id",
                    "task_suite", "task_id", "execution_scope", "world_size",
                    "global_rank", "local_rank", "bindings", "seeds",
                    "official_result", "episode_count", "traces", "tree_sha256",
                },
            )
            self.assertEqual(receipt_payload["kind"], "mf_wam_g0_task_trace_receipt")
            self.assertEqual(receipt_payload["process_id"], "libero_spatial/task00")
            self.assertEqual(
                receipt_payload["traces"][0]["path"],
                "traces/libero_spatial/task00/trial000.json",
            )
            self.assertEqual(receipt_payload["traces"][0]["trial_idx"], 0)
            self.assertEqual(
                receipt_payload["official_result"]["path"],
                "results/libero_spatial/task00.json",
            )
            self.assertEqual(
                receipt_payload["bindings"]["seed_schedule_canonical_sha256"],
                payload["metadata"]["upstream_digests"][
                    "seed_schedule_canonical_sha256"
                ],
            )
            self.assertTrue((root / "candidate/results/libero_spatial/task00.json").is_file())
            verified = verify_process_receipt_trace_inventory(
                receipt_path,
                output_root=root / "candidate",
                expected_trace_count=1,
            )
            self.assertEqual(verified["status"], "PASS")
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(InstrumentationError, "content mismatch"):
                verify_process_receipt_trace_inventory(
                    receipt_path,
                    output_root=root / "candidate",
                    expected_trace_count=1,
                )
            trace_path.unlink()
            outside = root / "outside-trace.json"
            outside.write_text("{}\n", encoding="utf-8")
            trace_path.symlink_to(outside)
            with self.assertRaisesRegex(InstrumentationError, "without following symlinks"):
                verify_process_receipt_trace_inventory(
                    receipt_path,
                    output_root=root / "candidate",
                    expected_trace_count=1,
                )
            tracer.restore()

    def test_trace_schema_rejects_legacy_and_unknown_record_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracer, _, _ = self._run_fake_episode(
                make_fake_eval_module(),
                make_cfg(root / "candidate"),
                traced=True,
                trace_root=root / "candidate/traces",
            )
            trace_path = root / "candidate/traces/libero_spatial/task00/trial000.json"
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertIs(validate_structured_trace_payload(payload), payload)

            legacy = json.loads(json.dumps(payload))
            legacy["records"][0]["raw_action_chunk"] = [[0.0] * 7 for _ in range(10)]
            with self.assertRaisesRegex(InstrumentationError, "unknown or missing fields"):
                validate_structured_trace_payload(legacy)

            unknown = json.loads(json.dumps(payload))
            unknown["records"][0]["unregistered_observer_field"] = True
            with self.assertRaisesRegex(InstrumentationError, "unknown or missing fields"):
                validate_structured_trace_payload(unknown)
            tracer.restore()

    def test_result_identity_mismatch_and_terminal_upstream_drift_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = make_cfg(root / "bad-result")
            tracer, _, _ = self._run_fake_episode(
                make_fake_eval_module(),
                cfg,
                traced=True,
                trace_root=root / "bad-result/traces",
                bind_result=False,
            )
            bad_result = {
                "task_suite": "libero_spatial",
                "task_id": 0,
                "task_description": "pick the object",
                "successes": 0,
                "total_episodes": 1,
                "gpu_id": 0,
                "success_episodes": [],
                "failure_episodes": [],
            }
            source = root / "bad-result/libero_spatial/gpu0_task0_results.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(bad_result), encoding="utf-8")
            with self.assertRaisesRegex(InstrumentationError, "success partition"):
                tracer.bind_official_task_result(bad_result)
            tracer.restore()

            drift_cfg = make_cfg(root / "upstream-drift")
            drift, _, _ = self._run_fake_episode(
                make_fake_eval_module(),
                drift_cfg,
                traced=True,
                trace_root=root / "upstream-drift/traces",
            )
            changed = {
                name: dict(identity)
                for name, identity in drift.upstream_identities.items()
            }
            changed["resolved_config"]["file_sha256"] = "0" * 64
            drift._test_terminal_upstream_verifier = lambda: changed
            with self.assertRaisesRegex(InstrumentationError, "upstream artifact identity drifted"):
                drift.finalize_process()
            drift.restore()

            trace_drift_cfg = make_cfg(root / "trace-drift")
            trace_drift, _, _ = self._run_fake_episode(
                make_fake_eval_module(),
                trace_drift_cfg,
                traced=True,
                trace_root=root / "trace-drift/traces",
            )
            trace_path = root / "trace-drift/traces/libero_spatial/task00/trial000.json"
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(InstrumentationError, "trace drifted"):
                trace_drift.finalize_process()
            trace_drift.restore()

    def test_atomic_json_publish_never_replaces_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            _atomic_write_json_no_replace(path, {"value": 1})
            with self.assertRaisesRegex(InstrumentationError, "refusing to replace"):
                _atomic_write_json_no_replace(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1})

    def test_atomic_json_publish_does_not_create_through_parent_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_root = root / "artifacts"
            outside = root / "outside"
            artifact_root.mkdir()
            outside.mkdir()
            (artifact_root / "traces").symlink_to(outside, target_is_directory=True)
            target = artifact_root / "traces" / "libero_spatial" / "trace.json"

            with self.assertRaisesRegex(
                InstrumentationError, "without following symlinks"
            ):
                _atomic_write_json_no_replace(target, {"value": 1})

            self.assertFalse((outside / "libero_spatial").exists())
            self.assertFalse(target.exists())

    def test_sha256_and_critical_inventory_reject_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            link = root / "critical.py"
            link.symlink_to(target.name)
            with self.assertRaisesRegex(InstrumentationError, "without following symlinks"):
                sha256_file(link)

            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "add", "target.py", "critical.py"], check=True
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=G0 Test",
                    "-c",
                    "user.email=g0@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "symlink fixture",
                ],
                check=True,
            )
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            with self.assertRaisesRegex(
                InstrumentationError,
                "without following symlinks|symlink, gitlink",
            ):
                verify_pristine_official_root(
                    root,
                    expected_commit=commit,
                    critical_paths=("critical.py",),
                )

    def test_terminal_source_drift_and_incomplete_or_duplicate_scope_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            drift_module = make_fake_eval_module()
            drift_cfg = make_cfg(root / "drift")
            tracer, _, _ = self._run_fake_episode(
                drift_module,
                drift_cfg,
                traced=True,
                trace_root=root / "drift/traces",
            )
            drifted = dict(tracer.official_identity)
            drifted["tree"] = "drifted"
            tracer._test_terminal_source_verifier = lambda: (
                drifted,
                tracer.instrumentation_identity,
            )
            with self.assertRaisesRegex(InstrumentationError, "identity drifted"):
                tracer.finalize_process()
            tracer.restore()

            incomplete_module = make_fake_eval_module()
            incomplete_cfg = make_cfg(root / "incomplete")
            incomplete, _, _ = self._run_fake_episode(
                incomplete_module,
                incomplete_cfg,
                traced=True,
                trace_root=root / "incomplete/traces",
                expected_trial_count=2,
            )
            with self.assertRaisesRegex(InstrumentationError, "scope is incomplete"):
                incomplete.finalize_process()

            with self.assertRaisesRegex(InstrumentationError, "duplicate trial trace"):
                incomplete_module.run_single_episode(
                    env=incomplete_module.created_envs[-1],
                    initial_state=np.arange(10, dtype=np.float32),
                    task_description="pick the object",
                    model=object(),
                    processor=object(),
                    cfg=incomplete_cfg,
                    episode_idx=0,
                    action_horizon=32,
                    input_w=448,
                    input_h=224,
                    model_device="cuda",
                )
            incomplete.restore()

    def test_invalid_identity_and_trace_root_escape_fail_before_rollout(self) -> None:
        module = make_fake_eval_module()
        with self.assertRaisesRegex(InstrumentationError, "run_id must match"):
            G0TraceInstrumentation(
                module,
                official_root=Path("/synthetic/official"),
                official_identity={"commit": "0" * 40},
                run_id="../escape",
                instrumentation_identity={"commit": "1" * 40},
                upstream_bindings={},
                enforce_module_origins=False,
            )

        cases = (
            ("../escape", 0, 0, "invalid formal LIBERO suite"),
            ("libero_spatial", 10, 0, "task_id must be an int"),
            ("libero_spatial", 0, 50, "trial_idx must be an int"),
        )
        for suite, task_id, trial_idx, message in cases:
            with self.subTest(suite=suite, task_id=task_id, trial_idx=trial_idx):
                with tempfile.TemporaryDirectory() as directory:
                    cfg = make_cfg(Path(directory) / "output")
                    cfg.EVALUATION.task_suite_name = suite
                    cfg.EVALUATION.task_id = task_id
                    with self.assertRaisesRegex(InstrumentationError, message):
                        self._run_fake_episode(
                            make_fake_eval_module(),
                            cfg,
                            traced=True,
                            trace_root=Path(directory) / "output/traces",
                            episode_idx=trial_idx,
                        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            artifact_root = root / "output"
            artifact_root.mkdir()
            linked_root = artifact_root / "traces"
            linked_root.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(InstrumentationError, "without following symlinks"):
                self._run_fake_episode(
                    make_fake_eval_module(),
                    make_cfg(artifact_root),
                    traced=True,
                    trace_root=linked_root,
                )

    def test_pristine_official_git_check_fails_on_dirty_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "critical.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "critical.py"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=G0 Test",
                    "-c",
                    "user.email=g0@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "fixture",
                ],
                check=True,
            )
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            receipt = verify_pristine_official_root(
                root,
                expected_commit=commit,
                critical_paths=("critical.py",),
            )
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["commit"], commit)

            (root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(InstrumentationError, "not clean"):
                verify_pristine_official_root(
                    root,
                    expected_commit=commit,
                    critical_paths=("critical.py",),
                )

    def test_instrumentation_git_identity_requires_exact_clean_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "observer.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "observer.py"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=G0 Test",
                    "-c",
                    "user.email=g0@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "observer fixture",
                ],
                check=True,
            )
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            receipt = verify_pristine_instrumentation_root(
                root,
                expected_commit=commit,
                critical_paths=("observer.py",),
            )
            self.assertEqual(receipt["commit"], commit)
            self.assertEqual(receipt["role"], "external_observer_and_launcher_source")
            with self.assertRaisesRegex(InstrumentationError, "HEAD mismatch"):
                verify_pristine_instrumentation_root(
                    root,
                    expected_commit="f" * 40,
                    critical_paths=("observer.py",),
                )

    def test_module_origin_audit_rejects_non_official_fastwam_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            official = Path(directory) / "official"
            official_eval_path = official / "experiments/libero/eval_libero_single.py"
            official_fastwam_path = official / "src/fastwam/runtime.py"
            official_eval_path.parent.mkdir(parents=True)
            official_fastwam_path.parent.mkdir(parents=True)
            official_eval_path.write_text("# eval\n", encoding="utf-8")
            official_fastwam_path.write_text("# runtime\n", encoding="utf-8")
            subprocess.run(["/usr/bin/git", "init", "-q", str(official)], check=True)
            subprocess.run(
                ["/usr/bin/git", "-C", str(official), "add", "."], check=True
            )
            subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(official),
                    "-c",
                    "user.name=G0 Test",
                    "-c",
                    "user.email=g0@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "module origins",
                ],
                check=True,
            )

            eval_module = types.ModuleType("experiments.libero.eval_libero_single")
            eval_module.__file__ = str(official_eval_path)
            fastwam_module = types.ModuleType("fastwam.runtime")
            fastwam_module.__file__ = str(official_fastwam_path)
            modules = {
                eval_module.__name__: eval_module,
                fastwam_module.__name__: fastwam_module,
            }
            receipt = audit_module_origins(official, modules=modules)
            self.assertEqual(receipt["status"], "PASS")

            evil = types.ModuleType("fastwam.models.evil")
            evil.__file__ = str(Path(directory) / "instrumentation/evil.py")
            modules[evil.__name__] = evil
            with self.assertRaisesRegex(InstrumentationError, "non-official"):
                audit_module_origins(official, modules=modules)

    def test_gitignored_native_and_filter_hidden_source_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hidden.py"
            source.write_text("MASKED\n", encoding="utf-8")
            (root / ".gitattributes").write_text(
                "hidden.py filter=mask\n", encoding="utf-8"
            )
            subprocess.run(["/usr/bin/git", "init", "-q", str(root)], check=True)
            filter_script = root / ".git/mask-filter.sh"
            filter_script.write_text(
                "#!/bin/sh\nprintf 'MASKED\\n'\n", encoding="utf-8"
            )
            filter_script.chmod(0o700)
            subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(root),
                    "config",
                    "filter.mask.clean",
                    str(filter_script),
                ],
                check=True,
            )
            subprocess.run(
                ["/usr/bin/git", "-C", str(root), "add", "."], check=True
            )
            subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=G0 Test",
                    "-c",
                    "user.email=g0@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "filtered source",
                ],
                check=True,
            )
            commit = subprocess.run(
                ["/usr/bin/git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            # Keep the tracked byte length unchanged so Git's stat cache plus
            # the clean filter can make porcelain appear clean.
            source.write_text("EVIL!!\n", encoding="utf-8")
            self.assertEqual(
                subprocess.run(
                    [
                        "/usr/bin/git",
                        "-C",
                        str(root),
                        "status",
                        "--porcelain",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout,
                "",
            )
            with self.assertRaisesRegex(
                InstrumentationError, "filters/includes|exact commit tree"
            ):
                verify_pristine_official_root(
                    root,
                    expected_commit=commit,
                    critical_paths=(".gitattributes",),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "critical.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["/usr/bin/git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["/usr/bin/git", "-C", str(root), "add", "critical.py"],
                check=True,
            )
            subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=G0 Test",
                    "-c",
                    "user.email=g0@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "ignored native",
                ],
                check=True,
            )
            commit = subprocess.run(
                ["/usr/bin/git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            (root / ".git/info/exclude").write_text("*.so\n", encoding="utf-8")
            (root / "critical.cpython-310-x86_64-linux-gnu.so").write_bytes(
                b"native shadow\n"
            )
            with self.assertRaisesRegex(
                InstrumentationError, "gitignored artifacts"
            ):
                verify_pristine_official_root(
                    root,
                    expected_commit=commit,
                    critical_paths=("critical.py",),
                )


if __name__ == "__main__":
    unittest.main()
