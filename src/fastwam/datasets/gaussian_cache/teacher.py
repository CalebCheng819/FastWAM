"""Optional external teacher providers for canonical Gaussian extraction.

No Policy-Lightning/NoPoSplat source is vendored here.  The provider imports an
explicit external checkout only after verifying its exact Git commit and a
caller-supplied checkpoint identity contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import torch

from .manifest import sha256_file
from .schema import correct_policy_lightning_legacy_covariance_order


class GaussianTeacher(Protocol):
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Map ``[B,V,3,H,W]`` RGB in [-1,1] to corrected ``[B,V,13,H,W]``."""

    def provenance(self) -> Mapping[str, Any]:
        """Return immutable teacher source/checkpoint provenance."""


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_checkpoint_identity(
    checkpoint_path: str | Path,
    *,
    integrity_mode: str,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> tuple[Path, str | None, int]:
    """Bind a teacher checkpoint by digest or strict regular-file metadata."""

    path = Path(checkpoint_path).expanduser()
    if integrity_mode == "sha256":
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise FileNotFoundError(
                f"External teacher checkpoint is not a regular file: {resolved}"
            )
        if not expected_sha256:
            raise ValueError("External teacher checkpoint SHA-256 is required")
        normalized = str(expected_sha256).lower()
        actual_sha256 = sha256_file(resolved)
        if actual_sha256 != normalized:
            raise ValueError(
                "External teacher checkpoint SHA-256 mismatch: "
                f"expected={normalized} actual={actual_sha256}"
            )
        return resolved, actual_sha256, resolved.stat().st_size

    if integrity_mode != "metadata_no_hash":
        raise ValueError(f"Unsupported teacher checkpoint integrity mode: {integrity_mode!r}")
    if expected_size_bytes is None or int(expected_size_bytes) <= 0:
        raise ValueError(
            "External teacher checkpoint byte size is required in metadata_no_hash mode"
        )

    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(
            f"External teacher checkpoint must be a direct regular file: {path}"
        )
    if before.st_nlink != 1:
        raise ValueError(
            "External teacher checkpoint must have exactly one hard link: "
            f"path={path} nlink={before.st_nlink}"
        )
    if before.st_size != int(expected_size_bytes):
        raise ValueError(
            "External teacher checkpoint byte-size mismatch: "
            f"expected={int(expected_size_bytes)} actual={before.st_size} path={path}"
        )
    resolved = path.resolve(strict=True)
    after = os.stat(resolved, follow_symlinks=False)
    if not stat.S_ISREG(after.st_mode):
        raise ValueError(f"External teacher checkpoint is not regular: {resolved}")
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise RuntimeError(
            f"External teacher checkpoint changed during metadata binding: {path}"
        )
    return resolved, None, before.st_size


def _resolved_config_sha256(config) -> str:
    from omegaconf import OmegaConf

    container = OmegaConf.to_container(config, resolve=True)
    payload = (json.dumps(container, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _contains_defaults(value: Any) -> bool:
    if isinstance(value, Mapping):
        return "defaults" in value or any(_contains_defaults(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_defaults(item) for item in value)
    return False


def _compose_encoder_config(repo_path: Path, config_path: Path):
    """Resolve Hydra defaults from the pinned checkout and return encoder config."""

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    raw = OmegaConf.load(config_path)
    raw_container = OmegaConf.to_container(raw, resolve=False)
    if _contains_defaults(raw_container):
        config_root = (repo_path / "config").resolve()
        try:
            relative = config_path.resolve().relative_to(config_root)
        except ValueError as exc:
            raise ValueError(
                "A config containing Hydra defaults must be inside the pinned repo config/ tree: "
                f"{config_path}"
            ) from exc
        if relative.suffix not in {".yaml", ".yml"}:
            raise ValueError(f"Hydra config must be YAML: {config_path}")
        config_name = relative.with_suffix("").as_posix()
        with initialize_config_dir(
            version_base=None,
            config_dir=str(config_root),
            job_name="fastwam_gaussian_teacher",
        ):
            composed = compose(config_name=config_name)
        composition = {
            "method": "hydra-compose",
            "config_root_relative_path": "config",
            "config_name": config_name,
        }
    else:
        composed = raw
        composition = {
            "method": "resolved-yaml",
            "config_name": config_path.name,
        }

    encoder_config = composed.encoder if "encoder" in composed else composed
    unresolved = OmegaConf.to_container(encoder_config, resolve=False)
    if _contains_defaults(unresolved) or "backbone" not in encoder_config:
        raise ValueError(
            "External encoder config has unresolved Hydra defaults/backbone; "
            "compose the pinned config before constructing the teacher"
        )
    OmegaConf.resolve(encoder_config)
    composition["composed_encoder_sha256"] = _resolved_config_sha256(encoder_config)
    return encoder_config, composition


class ExternalPolicyLightningTeacher:
    """Pinned adapter around an external Policy-Lightning checkout."""

    def __init__(
        self,
        *,
        repo_path: str | Path,
        expected_commit: str,
        checkpoint_path: str | Path,
        checkpoint_sha256: str | None,
        checkpoint_size_bytes: int | None = None,
        integrity_mode: str = "sha256",
        config_path: str | Path = "config/encoder/noposplat.yaml",
        device: str | torch.device = "cuda",
        require_clean_repo: bool = True,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.expected_commit = str(expected_commit).lower()
        self._integrity_mode = str(integrity_mode)
        self.checkpoint_path, actual_checkpoint_sha256, checkpoint_size = (
            _validate_checkpoint_identity(
                checkpoint_path,
                integrity_mode=self._integrity_mode,
                expected_sha256=checkpoint_sha256,
                expected_size_bytes=checkpoint_size_bytes,
            )
        )
        self.expected_checkpoint_sha256 = (
            None if checkpoint_sha256 is None else str(checkpoint_sha256).lower()
        )
        config = Path(config_path)
        self.config_path = (self.repo_path / config).resolve() if not config.is_absolute() else config
        self.device = torch.device(device)
        if not self.repo_path.is_dir():
            raise FileNotFoundError(f"External Policy-Lightning repo is missing: {self.repo_path}")
        actual_commit = _git(self.repo_path, "rev-parse", "HEAD").lower()
        if actual_commit != self.expected_commit:
            raise ValueError(
                f"External teacher commit mismatch: expected={self.expected_commit} actual={actual_commit}"
            )
        repository_dirty = bool(
            _git(self.repo_path, "status", "--porcelain=v1", "-uall")
        )
        if require_clean_repo and repository_dirty:
            raise ValueError(f"External teacher checkout is dirty: {self.repo_path}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"External teacher config is missing: {self.config_path}")
        try:
            self._config_relative_path = self.config_path.relative_to(self.repo_path).as_posix()
        except ValueError as exc:
            raise ValueError(
                "External teacher config must live inside the pinned repository: "
                f"{self.config_path}"
            ) from exc
        self._actual_commit = actual_commit
        self._repository_clean = not repository_dirty
        self._checkpoint_sha256 = actual_checkpoint_sha256
        self._checkpoint_size_bytes = checkpoint_size
        self._config_sha256 = sha256_file(self.config_path)
        self._remote_url = _git(self.repo_path, "remote", "get-url", "origin")
        self._encoder = self._load_encoder()

    def _load_encoder(self):
        # Policy-Lightning uses absolute imports rooted at its checkout.  Limit
        # sys.path mutation to import time and refuse an ambiguous pre-imported
        # top-level `model` package.
        existing_model = sys.modules.get("model")
        if existing_model is not None:
            existing_path = Path(getattr(existing_model, "__file__", "")).resolve()
            if self.repo_path not in existing_path.parents:
                raise RuntimeError(
                    "A different top-level 'model' package is already imported; "
                    "run canonical extraction in a fresh process"
                )
        sys.path.insert(0, str(self.repo_path))
        try:
            from model.noposplat.encoder import get_encoder
            from omegaconf import OmegaConf

            encoder_config, composition = _compose_encoder_config(
                self.repo_path,
                self.config_path,
            )
            self._config_composition = composition

            overrides: list[dict[str, str]] = []

            def force_unify(node, prefix: str = "") -> None:
                if OmegaConf.is_dict(node):
                    for key in list(node.keys()):
                        path = f"{prefix}.{key}" if prefix else str(key)
                        if str(key) == "coor_type":
                            overrides.append(
                                {
                                    "path": path,
                                    "original": str(node[key]),
                                    "value": "unify",
                                }
                            )
                            node[key] = "unify"
                        else:
                            force_unify(node[key], path)
                elif OmegaConf.is_list(node):
                    for index, value in enumerate(node):
                        force_unify(value, f"{prefix}[{index}]")

            force_unify(encoder_config, "encoder")
            if not overrides:
                raise ValueError(
                    "Pinned external teacher config has no coor_type field to override; "
                    "refusing ambiguous multi-view pairing"
                )
            self._config_overrides = overrides
            self._resolved_config_sha256 = _resolved_config_sha256(encoder_config)
            encoder = get_encoder(encoder_config)
        finally:
            try:
                sys.path.remove(str(self.repo_path))
            except ValueError:
                pass

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
            raise ValueError("External teacher checkpoint must contain a state_dict mapping")
        weights = {
            key[8:]: value
            for key, value in checkpoint["state_dict"].items()
            if str(key).startswith("encoder.")
        }
        if not weights:
            raise ValueError("External teacher checkpoint has no encoder.* weights")
        missing, unexpected = encoder.load_state_dict(weights, strict=False)
        self._missing_keys = sorted(map(str, missing))
        self._unexpected_keys = sorted(map(str, unexpected))
        self._critical_missing_keys = [
            key
            for key in self._missing_keys
            if "backbone" in key.lower() or "head" in key.lower()
        ]
        if self._critical_missing_keys:
            raise ValueError(
                "External teacher checkpoint is missing core backbone/head weights: "
                f"{self._critical_missing_keys[:32]}"
            )
        encoder.to(self.device)
        encoder.eval()
        return encoder

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"Teacher images must be [B,V,3,H,W], got {tuple(images.shape)}")
        images = images.to(device=self.device, dtype=torch.float32, non_blocking=True)
        raw = self._encoder({"image": images})
        if not isinstance(raw, torch.Tensor) or raw.ndim != 5 or raw.shape[2] != 13:
            shape = None if not isinstance(raw, torch.Tensor) else tuple(raw.shape)
            raise ValueError(f"External teacher must return [B,V,13,H,W], got {shape}")
        corrected = correct_policy_lightning_legacy_covariance_order(raw)
        if not bool(torch.isfinite(corrected).all().item()):
            raise ValueError("External teacher emitted non-finite canonical Gaussian values")
        canonical = corrected.to(device="cpu", dtype=torch.float16).contiguous()
        if not bool(torch.isfinite(canonical).all().item()):
            max_abs = float(corrected.detach().abs().max().item())
            raise OverflowError(
                "External teacher values overflow canonical FP16 without clamping: "
                f"max_abs_float32={max_abs} fp16_max={torch.finfo(torch.float16).max}"
            )
        return canonical

    def provenance(self) -> Mapping[str, Any]:
        return {
            "kind": "external-policy-lightning",
            "repository_url": self._remote_url,
            "repository_commit": self._actual_commit,
            "repository_clean": self._repository_clean,
            "config_relative_path": self._config_relative_path,
            "config_sha256": self._config_sha256,
            "config_composition": self._config_composition,
            "resolved_config_sha256": self._resolved_config_sha256,
            "config_overrides": {
                "coor_type": "unify",
                "coor_type_paths": self._config_overrides,
            },
            "checkpoint_filename": self.checkpoint_path.name,
            "checkpoint_integrity_mode": self._integrity_mode,
            "checkpoint_size_bytes": self._checkpoint_size_bytes,
            "checkpoint_sha256": self._checkpoint_sha256,
            "legacy_covariance_layout_corrected": True,
            "missing_weight_keys": self._missing_keys,
            "unexpected_weight_keys": self._unexpected_keys,
            "critical_missing_weight_keys": self._critical_missing_keys,
            "usage_scope": "research_noncommercial",
            "checkpoint_license_declared": False,
            "license_note": (
                "Policy-Lightning top level is MIT, but referenced NoPoSplat/CroCo files "
                "include CC BY-NC-SA 4.0 notices; the checkpoint repository declares no "
                "cardData/license. Cache generation and use are restricted to "
                "non-commercial research pending separate rights clearance."
            ),
        }
