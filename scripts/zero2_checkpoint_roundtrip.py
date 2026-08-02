#!/usr/bin/env python3
"""Real two-process-boundary Accelerate/DeepSpeed ZeRO-2 checkpoint smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import random
import secrets
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from state_tree_manifest import canonical_bytes, publish_state_tree_manifest


PINNED_PACKAGES = ("torch", "accelerate", "deepspeed")
SMOKE_LOCAL_MICRO_BATCH_SIZE = 4
SMOKE_GRADIENT_ACCUMULATION_STEPS = 1
FORMAL_SMOKE_WORLD_SIZE = 32
FORMAL_SMOKE_GLOBAL_TRAIN_BATCH_SIZE = (
    SMOKE_LOCAL_MICRO_BATCH_SIZE
    * SMOKE_GRADIENT_ACCUMULATION_STEPS
    * FORMAL_SMOKE_WORLD_SIZE
)


class SmokeProgress:
    def __init__(self, global_step: int = 0):
        self.global_step = int(global_step)

    def state_dict(self) -> dict[str, int]:
        return {"global_step": self.global_step}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.global_step = int(state["global_step"])


class TinyStatefulModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 8),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def _update_digest(digest: "hashlib._Hash", value) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        # Flatten first: PyTorch refuses dtype-viewing a 0-D optimizer step
        # tensor directly even though its storage is valid.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode() + b"\0")
        for item in value:
            _update_digest(digest, item)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        digest.update(type(value).__name__.encode() + b"\0")
        digest.update(repr(value).encode())
        digest.update(b"\0")
    else:
        buffer = io.BytesIO()
        torch.save(value, buffer)
        digest.update(b"torch-save\0")
        digest.update(buffer.getvalue())


def _fingerprint(value) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def _rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state(torch.cuda.current_device())
    return state


def _next_rng_sample(device: torch.device) -> dict[str, object]:
    sample: dict[str, object] = {
        "numpy": np.random.random(4).tolist(),
        "python": [random.random() for _ in range(4)],
        "torch_cpu": torch.rand(4, device="cpu").tolist(),
    }
    if device.type == "cuda":
        sample["torch_cuda"] = torch.rand(4, device=device).cpu().tolist()
    return sample


def _state_fingerprints(accelerator, model, optimizer, scheduler, progress) -> dict[str, object]:
    unwrapped = accelerator.unwrap_model(model)
    return {
        "global_step": progress.global_step,
        "model": _fingerprint(unwrapped.state_dict()),
        "optimizer": _fingerprint(optimizer.state_dict()),
        "rng": _fingerprint(_rng_state()),
        "scheduler": _fingerprint(scheduler.state_dict()),
    }


def _require_smoke_gradient_accumulation(accelerator) -> None:
    """Reject environment-driven changes to the smoke accumulation contract."""

    configured = accelerator.gradient_accumulation_steps
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured != SMOKE_GRADIENT_ACCUMULATION_STEPS
    ):
        raise ValueError(
            "ZeRO-2 smoke requires resolved gradient_accumulation_steps="
            f"{SMOKE_GRADIENT_ACCUMULATION_STEPS}, got {configured!r}. "
            "Unset ACCELERATE_GRADIENT_ACCUMULATION_STEPS."
        )


def _configure_smoke_deepspeed_batch_accounting(accelerator) -> None:
    """Resolve DeepSpeed metadata when the smoke has no DataLoader.

    Accelerate cannot infer a micro-batch size from the model, optimizer and
    scheduler alone.  The smoke constructs four samples per rank explicitly in
    ``_train_step``, so bind that fixed local batch before ``prepare`` while
    leaving the shared training DeepSpeed config on ``auto``.
    """

    _require_smoke_gradient_accumulation(accelerator)
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is None:
        raise RuntimeError("formal ZeRO-2 smoke requires the DeepSpeed plugin")
    deepspeed_config = plugin.deepspeed_config
    configured = deepspeed_config.get("train_micro_batch_size_per_gpu", "auto")
    if configured == "auto":
        configured = SMOKE_LOCAL_MICRO_BATCH_SIZE
        deepspeed_config["train_micro_batch_size_per_gpu"] = configured
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured != SMOKE_LOCAL_MICRO_BATCH_SIZE
    ):
        raise ValueError(
            "ZeRO-2 smoke requires train_micro_batch_size_per_gpu="
            f"{SMOKE_LOCAL_MICRO_BATCH_SIZE}, got {configured!r}."
        )


def _resolved_smoke_batch_accounting(accelerator) -> dict[str, int]:
    """Read back the values DeepSpeed will actually use after ``prepare``."""

    _require_smoke_gradient_accumulation(accelerator)
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is None:
        raise RuntimeError("formal ZeRO-2 smoke requires the DeepSpeed plugin")
    deepspeed_config = plugin.deepspeed_config
    values = {
        "local_micro_batch_size": deepspeed_config.get(
            "train_micro_batch_size_per_gpu"
        ),
        "gradient_accumulation_steps": deepspeed_config.get(
            "gradient_accumulation_steps"
        ),
        "global_train_batch_size": deepspeed_config.get("train_batch_size"),
        "world_size": accelerator.num_processes,
    }
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"ZeRO-2 smoke requires a resolved positive integer {key}, "
                f"got {value!r}."
            )
    expected_global = (
        values["local_micro_batch_size"]
        * values["gradient_accumulation_steps"]
        * values["world_size"]
    )
    if values["local_micro_batch_size"] != SMOKE_LOCAL_MICRO_BATCH_SIZE:
        raise ValueError(
            "ZeRO-2 smoke resolved an unexpected local micro batch: "
            f"{values['local_micro_batch_size']!r}."
        )
    if values["gradient_accumulation_steps"] != SMOKE_GRADIENT_ACCUMULATION_STEPS:
        raise ValueError(
            "ZeRO-2 smoke resolved an unexpected gradient accumulation value: "
            f"{values['gradient_accumulation_steps']!r}."
        )
    if values["global_train_batch_size"] != expected_global:
        raise ValueError(
            "ZeRO-2 smoke DeepSpeed train_batch_size is inconsistent: "
            f"resolved={values['global_train_batch_size']!r} "
            f"expected={expected_global}."
        )
    return values


def _require_formal_smoke_batch_accounting(
    batch_accounting: dict[str, int],
) -> None:
    expected = {
        "global_train_batch_size": FORMAL_SMOKE_GLOBAL_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": SMOKE_GRADIENT_ACCUMULATION_STEPS,
        "local_micro_batch_size": SMOKE_LOCAL_MICRO_BATCH_SIZE,
        "world_size": FORMAL_SMOKE_WORLD_SIZE,
    }
    if batch_accounting != expected:
        raise ValueError(
            "formal ZeRO-2 smoke batch accounting mismatch: "
            f"expected={expected} observed={batch_accounting}"
        )


def _build_runtime(seed: int):
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="no")
    _configure_smoke_deepspeed_batch_accounting(accelerator)
    # device_specific=True needs AcceleratorState to exist. The target
    # Accelerate 1.12.0 raises if this is called before Accelerator().
    set_seed(seed, device_specific=True)
    model = TinyStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: max(0.1, 1.0 - 0.05 * step)
    )
    progress = SmokeProgress()
    accelerator.register_for_checkpointing(progress)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    batch_accounting = _resolved_smoke_batch_accounting(accelerator)
    return accelerator, model, optimizer, scheduler, progress, batch_accounting


def _train_step(accelerator, model, optimizer, scheduler, *, offset: float) -> None:
    optimizer.zero_grad(set_to_none=True)
    base = torch.arange(
        SMOKE_LOCAL_MICRO_BATCH_SIZE * 16,
        device=accelerator.device,
        dtype=torch.float32,
    ).reshape(SMOKE_LOCAL_MICRO_BATCH_SIZE, 16)
    target = torch.arange(
        SMOKE_LOCAL_MICRO_BATCH_SIZE * 8,
        device=accelerator.device,
        dtype=torch.float32,
    ).reshape(SMOKE_LOCAL_MICRO_BATCH_SIZE, 8)
    prediction = model(base / 63.0 + offset)
    loss = torch.nn.functional.mse_loss(prediction, target / 31.0)
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace smoke proof: {path}")
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        encoded = canonical_bytes(payload)
        written = 0
        while written < len(encoded):
            written += os.write(descriptor, encoded[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _process_start_ticks() -> int:
    # Linux /proc field 22 disambiguates a genuinely new process even if the
    # kernel rapidly reuses a PID between the save and load worlds.
    fields = Path("/proc/self/stat").read_text(encoding="utf-8").split()
    return int(fields[21])


def run_save(state_dir: Path, seed: int) -> None:
    accelerator, model, optimizer, scheduler, progress, batch_accounting = (
        _build_runtime(seed)
    )
    _require_formal_smoke_batch_accounting(batch_accounting)
    _train_step(accelerator, model, optimizer, scheduler, offset=0.0)
    progress.global_step = 1
    state_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(state_dir))
    accelerator.wait_for_everyone()

    saved = _state_fingerprints(accelerator, model, optimizer, scheduler, progress)
    next_rng = _next_rng_sample(accelerator.device)
    proof_dir = state_dir / "smoke-proof"
    proof_dir.mkdir(exist_ok=True)
    rank = accelerator.process_index
    save_proof = {
        "batch_accounting": batch_accounting,
        "fingerprints": saved,
        "next_rng_sample": next_rng,
        "phase": "save",
        "process_nonce": secrets.token_hex(16),
        "process_pid": os.getpid(),
        "process_start_ticks": _process_start_ticks(),
        "rank": rank,
        "schema_version": 2,
        "world_size": accelerator.num_processes,
    }
    _atomic_json(proof_dir / f"save-rank-{rank:05d}.json", save_proof)

    # Deliberately move every registered state family away from the saved
    # values before this process exits. The load phase runs in a fresh process.
    _train_step(accelerator, model, optimizer, scheduler, offset=1.0)
    progress.global_step = 99
    random.random()
    np.random.random()
    torch.rand(1, device=accelerator.device)
    mutated = _state_fingerprints(accelerator, model, optimizer, scheduler, progress)
    for key in ("model", "optimizer", "scheduler", "rng", "global_step"):
        if mutated[key] == saved[key]:
            raise RuntimeError(f"smoke mutation did not change {key} on rank {rank}")
    _atomic_json(
        proof_dir / f"mutated-rank-{rank:05d}.json",
        {
            "batch_accounting": batch_accounting,
            "fingerprints": mutated,
            "phase": "mutated_before_process_exit",
            "process_nonce": save_proof["process_nonce"],
            "process_pid": os.getpid(),
            "process_start_ticks": _process_start_ticks(),
            "rank": rank,
            "schema_version": 2,
            "world_size": accelerator.num_processes,
        },
    )
    accelerator.wait_for_everyone()


def run_load(state_dir: Path, seed: int) -> None:
    accelerator, model, optimizer, scheduler, progress, batch_accounting = (
        _build_runtime(seed + 100000)
    )
    _require_formal_smoke_batch_accounting(batch_accounting)
    rank = accelerator.process_index
    pre_load = _state_fingerprints(accelerator, model, optimizer, scheduler, progress)
    accelerator.load_state(str(state_dir))
    restored = _state_fingerprints(accelerator, model, optimizer, scheduler, progress)
    expected_path = state_dir / "smoke-proof" / f"save-rank-{rank:05d}.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    expected_fingerprints = expected["fingerprints"]
    checks = {
        key: restored[key] == expected_fingerprints[key]
        for key in ("model", "optimizer", "scheduler", "rng", "global_step")
    }
    checks["rng_next_sample"] = (
        _next_rng_sample(accelerator.device) == expected["next_rng_sample"]
    )
    checks["pre_load_was_distinct"] = any(
        pre_load[key] != expected_fingerprints[key]
        for key in ("model", "optimizer", "scheduler", "rng", "global_step")
    )
    if not all(checks.values()):
        raise RuntimeError(f"ZeRO-2 roundtrip mismatch on rank {rank}: {checks}")
    _atomic_json(
        state_dir / "smoke-proof" / f"load-rank-{rank:05d}.json",
        {
            "batch_accounting": batch_accounting,
            "checks": checks,
            "fingerprints": restored,
            "phase": "load_fresh_process",
            "process_nonce": secrets.token_hex(16),
            "process_pid": os.getpid(),
            "process_start_ticks": _process_start_ticks(),
            "rank": rank,
            "schema_version": 2,
            "world_size": accelerator.num_processes,
        },
    )
    accelerator.wait_for_everyone()


def _git_head(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().lower()


def finalize(output_root: Path, state_dir: Path, marker: Path, manifest: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    code_commit = os.environ.get("FASTWAM_CODE_COMMIT", "").strip().lower()
    if len(code_commit) != 40 or _git_head(repository) != code_commit:
        raise RuntimeError("FASTWAM_CODE_COMMIT must equal the smoke runner Git HEAD")
    image_reference = os.environ.get("FASTWAM_DLC_IMAGE_REFERENCE", "").strip()
    image_digest = os.environ.get("FASTWAM_DLC_IMAGE_DIGEST", "").strip().lower()
    if not image_reference or not image_digest.startswith("sha256:") or len(image_digest) != 71:
        raise RuntimeError("real OSS smoke requires an exact image reference and OCI SHA-256 digest")
    int(image_digest.split(":", 1)[1], 16)
    package_versions = {
        package: importlib.metadata.version(package) for package in PINNED_PACKAGES
    }
    proof_dir = state_dir / "smoke-proof"
    save_proofs = sorted(proof_dir.glob("save-rank-*.json"))
    load_proofs = sorted(proof_dir.glob("load-rank-*.json"))
    mutated_proofs = sorted(proof_dir.glob("mutated-rank-*.json"))
    world_size = len(save_proofs)
    if world_size != 32 or len(load_proofs) != 32 or len(mutated_proofs) != 32:
        raise RuntimeError(
            "formal OSS smoke requires exactly 32 save/mutate/load rank proofs"
        )
    all_checks = {
        "global_step": True,
        "model": True,
        "optimizer": True,
        "rng": True,
        "rng_next_sample": True,
        "scheduler": True,
        "separate_process": True,
    }
    expected_batch_accounting = {
        "global_train_batch_size": FORMAL_SMOKE_GLOBAL_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": SMOKE_GRADIENT_ACCUMULATION_STEPS,
        "local_micro_batch_size": SMOKE_LOCAL_MICRO_BATCH_SIZE,
        "world_size": FORMAL_SMOKE_WORLD_SIZE,
    }
    for rank, (save_path, mutate_path, load_path) in enumerate(
        zip(save_proofs, mutated_proofs, load_proofs, strict=True)
    ):
        saved = json.loads(save_path.read_text(encoding="utf-8"))
        mutated = json.loads(mutate_path.read_text(encoding="utf-8"))
        loaded = json.loads(load_path.read_text(encoding="utf-8"))
        if saved["rank"] != rank or mutated["rank"] != rank or loaded["rank"] != rank:
            raise RuntimeError(f"rank proof ordering mismatch at rank {rank}")
        for phase, proof in (
            ("save", saved),
            ("mutated", mutated),
            ("load", loaded),
        ):
            if proof.get("world_size") != FORMAL_SMOKE_WORLD_SIZE:
                raise RuntimeError(
                    f"{phase} proof world size mismatch at rank {rank}"
                )
            if proof.get("batch_accounting") != expected_batch_accounting:
                raise RuntimeError(
                    f"{phase} proof batch accounting mismatch at rank {rank}: "
                    f"{proof.get('batch_accounting')!r}"
                )
        for key in ("global_step", "model", "optimizer", "rng", "scheduler"):
            all_checks[key] &= bool(loaded["checks"].get(key))
        all_checks["rng_next_sample"] &= bool(loaded["checks"].get("rng_next_sample"))
        all_checks["separate_process"] &= (
            saved["process_nonce"] != loaded["process_nonce"]
            and (
                saved["process_pid"],
                saved["process_start_ticks"],
            )
            != (
                loaded["process_pid"],
                loaded["process_start_ticks"],
            )
        )
    if not all(all_checks.values()):
        raise RuntimeError(f"roundtrip proof aggregation failed: {all_checks}")
    manifest_summary = publish_state_tree_manifest(
        state_dir,
        manifest,
        role="zero2_roundtrip_smoke_state",
    )
    pyproject = repository / "pyproject.toml"
    payload = {
        "batch_accounting": expected_batch_accounting,
        "code_commit": code_commit,
        "filesystem_device": int(os.stat(output_root).st_dev),
        "image_digest": image_digest,
        "image_reference": image_reference,
        "output_root": str(output_root.resolve(strict=True)),
        "package_versions": package_versions,
        "pyproject_sha256": hashlib.sha256(pyproject.read_bytes()).hexdigest(),
        "roundtrip": all_checks,
        "schema_version": 2,
        "state_tree_manifest": str(manifest.resolve(strict=True)),
        "state_tree_manifest_sha256": manifest_summary["manifest_sha256"],
        "state_tree_root": str(state_dir.resolve(strict=True)),
        "status": "PASS",
        "world_size": world_size,
        "zero_stage": 2,
    }
    _atomic_json(marker, payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("save", "load", "finalize"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve(strict=True)
    state_dir = output_root / "zero2-state"
    if args.phase == "save":
        run_save(state_dir, args.seed)
    elif args.phase == "load":
        run_load(state_dir, args.seed)
    else:
        finalize(
            output_root,
            state_dir,
            output_root / "zero2-roundtrip-smoke.json",
            output_root / "zero2-state-tree.json",
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
        raise
