#!/usr/bin/env python3
"""Validate node-local bundle mappings and derive local-root stats metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


def _inside(root: Path, candidate: Path, *, kind: str) -> Path:
    if candidate.is_symlink():
        raise ValueError(f"{kind} must not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{kind} escapes node-local bundle root: {candidate}") from error
    return resolved


def _regular_tree_files(root: Path, *, suffix: str | None = None) -> list[Path]:
    files: list[Path] = []
    for directory, names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            child = directory_path / name
            if child.is_symlink():
                raise ValueError(f"symlink directory is forbidden in local bundle: {child}")
        for name in filenames:
            child = directory_path / name
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ValueError(f"bundle member must be a regular non-symlink file: {child}")
            if suffix is None or child.suffix == suffix:
                files.append(child)
    return sorted(files, key=lambda path: os.fsencode(str(path.relative_to(root))))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_stats(output: Path, payload: bytes) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise ValueError(f"derived stats parent must not be a symlink: {output.parent}")
    temporary = output.parent / f".{output.name}.tmp.{os.getpid()}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--bundle-manifest-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-h5-files", type=int, default=24)
    parser.add_argument("--stats-source", type=Path, required=True)
    parser.add_argument("--text-embeds-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--model-cache-root", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--vae-manifest-sha256", required=True)
    parser.add_argument("--expected-vae-sha256", required=True)
    parser.add_argument("--gaussian-root", type=Path)
    parser.add_argument("--gaussian-bundle-root", type=Path)
    parser.add_argument("--output-stats", type=Path, required=True)
    args = parser.parse_args()
    try:
        bundle_root = args.bundle_root.resolve(strict=True)
        if bundle_root.is_symlink() or not bundle_root.is_dir():
            raise ValueError("bundle root must be a regular directory")
        if len(args.bundle_manifest_sha256) != 64:
            raise ValueError("bundle manifest SHA-256 must contain 64 hex characters")
        int(args.bundle_manifest_sha256, 16)
        dataset_root = _inside(bundle_root, args.dataset_root, kind="dataset root")
        stats_source = _inside(bundle_root, args.stats_source, kind="stats source")
        text_root = _inside(bundle_root, args.text_embeds_root, kind="text embeds root")
        checkpoint = _inside(bundle_root, args.checkpoint, kind="checkpoint")
        model_cache_root = _inside(bundle_root, args.model_cache_root, kind="model cache root")
        vae = _inside(bundle_root, args.vae, kind="Wan VAE")
        if not dataset_root.is_dir():
            raise ValueError(f"dataset root is not a directory: {dataset_root}")
        h5_files = _regular_tree_files(dataset_root, suffix=".h5")
        if len(h5_files) != args.expected_h5_files:
            raise ValueError(
                f"expected {args.expected_h5_files} H5 files under {dataset_root}, "
                f"observed {len(h5_files)}"
            )
        if not stats_source.is_file():
            raise ValueError(f"stats source is not a regular file: {stats_source}")
        if not text_root.is_dir() or not _regular_tree_files(text_root):
            raise ValueError(f"text embeds root is empty or not a directory: {text_root}")
        if not checkpoint.is_file():
            raise ValueError(f"checkpoint is not a regular file: {checkpoint}")
        if not model_cache_root.is_dir() or not _regular_tree_files(model_cache_root):
            raise ValueError(f"model cache root is empty or not a directory: {model_cache_root}")
        if not vae.is_file():
            raise ValueError(f"Wan VAE is not a regular file: {vae}")
        try:
            vae.relative_to(model_cache_root)
        except ValueError as error:
            raise ValueError("Wan VAE must be inside the mapped model cache root") from error
        expected_checkpoint = args.expected_checkpoint_sha256.lower()
        manifest_checkpoint = args.checkpoint_manifest_sha256.lower()
        if len(expected_checkpoint) != 64 or len(manifest_checkpoint) != 64:
            raise ValueError("checkpoint SHA-256 values must contain 64 hex characters")
        int(expected_checkpoint, 16)
        int(manifest_checkpoint, 16)
        if manifest_checkpoint != expected_checkpoint:
            raise ValueError(
                "checkpoint manifest identity mismatch: "
                f"expected={expected_checkpoint} manifest={manifest_checkpoint}"
            )
        expected_vae = args.expected_vae_sha256.lower()
        manifest_vae = args.vae_manifest_sha256.lower()
        if len(expected_vae) != 64 or len(manifest_vae) != 64:
            raise ValueError("Wan VAE SHA-256 values must contain 64 hex characters")
        int(expected_vae, 16)
        int(manifest_vae, 16)
        if manifest_vae != expected_vae:
            raise ValueError(
                "Wan VAE manifest identity mismatch: "
                f"expected={expected_vae} manifest={manifest_vae}"
            )
        if args.gaussian_root is not None:
            if args.gaussian_bundle_root is None:
                raise ValueError("--gaussian-bundle-root is required with --gaussian-root")
            gaussian_bundle_root = args.gaussian_bundle_root.resolve(strict=True)
            gaussian_root = _inside(
                gaussian_bundle_root, args.gaussian_root, kind="Gaussian root"
            )
            if not gaussian_root.is_dir() or not _regular_tree_files(gaussian_root):
                raise ValueError(f"Gaussian root is empty or not a directory: {gaussian_root}")

        payload = json.loads(stats_source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("normalization stats must be a JSON object")
        source_stats_sha256 = _sha256(stats_source)
        payload["source_root"] = str(dataset_root)
        payload["fastwam_local_derivation"] = {
            "bundle_manifest_sha256": args.bundle_manifest_sha256.lower(),
            "source_stats_sha256": source_stats_sha256,
        }
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        output_stats = args.output_stats.resolve(strict=False)
        try:
            output_stats.relative_to(bundle_root)
        except ValueError:
            pass
        else:
            raise ValueError("derived stats output must be outside the immutable bundle root")
        _atomic_stats(output_stats, encoded)
        print(
            json.dumps(
                {
                    "checkpoint_sha256": manifest_checkpoint,
                    "dataset_h5_files": len(h5_files),
                    "derived_stats": str(output_stats),
                    "source_stats_sha256": source_stats_sha256,
                    "status": "PASS",
                    "vae_sha256": manifest_vae,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
