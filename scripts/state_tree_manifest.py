#!/usr/bin/env python3
"""Build or validate a canonical, immutable whole-state tree manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path, PurePosixPath


SCHEMA_VERSION = 1
ROLES = {"accelerate_zero2_full_state", "zero2_roundtrip_smoke_state"}


def canonical_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def sha256_regular_file(path: Path, *, require_single_link: bool = True) -> tuple[str, int]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {path}")
        if require_single_link and before.st_nlink != 1:
            raise ValueError(
                f"state files must not be hard-linked: nlink={before.st_nlink} path={path}"
            )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise RuntimeError(f"file changed while hashing: {path}")
        return digest.hexdigest(), int(after.st_size)
    finally:
        os.close(descriptor)


def safe_relative(value: str) -> PurePosixPath:
    if not value or any(character in value for character in ("\x00", "\n", "\r")):
        raise ValueError(f"unsafe state-tree path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe state-tree path: {value!r}")
    return relative


def _resolved_unaliased(path: Path, *, label: str, must_exist: bool) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    absolute = Path(os.path.abspath(path))
    resolved = path.resolve(strict=must_exist)
    if absolute != resolved:
        raise ValueError(f"{label} must not traverse symlinks or aliases: {path}")
    return resolved


def scan_state_tree(root: Path) -> dict[PurePosixPath, Path]:
    root = _resolved_unaliased(root, label="state root", must_exist=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"state root must be a non-symlink directory: {root}")
    observed: dict[PurePosixPath, Path] = {}
    for current_root, directories, files in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_root)
        directories.sort(key=os.fsencode)
        files.sort(key=os.fsencode)
        for name in directories:
            path = current / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"state tree contains a symlink/special directory: {path}")
        for name in files:
            path = current / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ValueError(f"state tree contains a symlink/special file: {path}")
            relative = safe_relative(path.relative_to(root).as_posix())
            observed[relative] = path
    if not observed:
        raise ValueError(f"state tree contains no regular files: {root}")
    return observed


def build_state_tree_payload(root: Path, *, role: str) -> dict[str, object]:
    if role not in ROLES:
        raise ValueError(f"unsupported state-tree role: {role!r}")
    root = _resolved_unaliased(root, label="state root", must_exist=True)
    observed = scan_state_tree(root)
    files: list[dict[str, object]] = []
    total_bytes = 0
    for relative in sorted(observed, key=lambda item: os.fsencode(item.as_posix())):
        digest, size = sha256_regular_file(observed[relative])
        files.append({"bytes": size, "path": relative.as_posix(), "sha256": digest})
        total_bytes += size
    return {
        "files": files,
        "role": role,
        "schema_version": SCHEMA_VERSION,
        "total_bytes": total_bytes,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_state_tree_manifest(root: Path, output: Path, *, role: str) -> dict[str, object]:
    root = _resolved_unaliased(root, label="state root", must_exist=True)
    output = _resolved_unaliased(output, label="manifest output", must_exist=False)
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("state-tree manifest must be outside the sealed state root")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError(
            f"manifest parent must be an existing non-symlink directory: {output.parent}"
        )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace existing state-tree manifest: {output}")
    payload = build_state_tree_payload(root, role=role)
    encoded = canonical_bytes(payload)
    temporary = output.parent / f".{output.name}.tmp.{os.getpid()}"
    lock = output.parent / f".{output.name}.publish.lock"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        written = 0
        while written < len(encoded):
            written += os.write(descriptor, encoded[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.mkdir(lock, 0o700)
        try:
            if output.exists() or output.is_symlink():
                raise FileExistsError(
                    f"refusing to replace existing state-tree manifest: {output}"
                )
            os.replace(temporary, output)
            _fsync_directory(output.parent)
        finally:
            lock.rmdir()
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return {
        "file_count": len(payload["files"]),
        "manifest": str(output),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "role": role,
        "state_root": str(root),
        "total_bytes": payload["total_bytes"],
    }


def validate_state_tree_manifest(
    root: Path,
    manifest: Path,
    *,
    expected_manifest_sha256: str,
    expected_role: str,
) -> dict[str, object]:
    if expected_role not in ROLES:
        raise ValueError(f"unsupported state-tree role: {expected_role!r}")
    root = _resolved_unaliased(root, label="state root", must_exist=True)
    manifest = _resolved_unaliased(manifest, label="state-tree manifest", must_exist=True)
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError(f"state-tree manifest must be a regular non-symlink file: {manifest}")
    expected_manifest_sha256 = expected_manifest_sha256.lower()
    if len(expected_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_manifest_sha256
    ):
        raise ValueError("expected manifest SHA-256 must be 64 lowercase hex characters")
    encoded = manifest.read_bytes()
    actual_manifest_sha256 = hashlib.sha256(encoded).hexdigest()
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            "state-tree manifest SHA-256 mismatch: "
            f"expected={expected_manifest_sha256} actual={actual_manifest_sha256} path={manifest}"
        )
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid state-tree manifest JSON: {manifest}: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {
        "files",
        "role",
        "schema_version",
        "total_bytes",
    }:
        raise ValueError("state-tree manifest fields do not match schema v1")
    if payload["schema_version"] != SCHEMA_VERSION or payload["role"] != expected_role:
        raise ValueError(
            f"state-tree manifest role/schema mismatch: {payload.get('role')!r}"
        )
    if encoded != canonical_bytes(payload):
        raise ValueError("state-tree manifest is not canonical JSON")
    records = payload["files"]
    if not isinstance(records, list) or not records:
        raise ValueError("state-tree manifest files must be a non-empty list")
    expected: dict[PurePosixPath, tuple[int, str]] = {}
    previous: bytes | None = None
    for record in records:
        if not isinstance(record, dict) or set(record) != {"bytes", "path", "sha256"}:
            raise ValueError("state-tree manifest file record fields mismatch")
        relative = safe_relative(record["path"])
        key = os.fsencode(relative.as_posix())
        if previous is not None and key <= previous:
            raise ValueError("state-tree records must be unique and bytewise sorted")
        previous = key
        size = record["bytes"]
        digest = record["sha256"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"invalid state-tree byte size for {relative}: {size!r}")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"invalid state-tree SHA-256 for {relative}")
        expected[relative] = (size, digest)
    observed = scan_state_tree(root)
    missing = sorted(
        (path.as_posix() for path in expected.keys() - observed.keys()), key=os.fsencode
    )
    unexpected = sorted(
        (path.as_posix() for path in observed.keys() - expected.keys()), key=os.fsencode
    )
    if missing or unexpected:
        raise RuntimeError(
            "state tree does not exactly match manifest: "
            f"missing={missing[:12]} unexpected={unexpected[:12]}"
        )
    total_bytes = 0
    for relative in sorted(expected, key=lambda item: os.fsencode(item.as_posix())):
        expected_size, expected_sha256 = expected[relative]
        actual_sha256, actual_size = sha256_regular_file(observed[relative])
        if actual_size != expected_size:
            raise RuntimeError(
                "state-tree file byte-size mismatch: "
                f"expected={expected_size} actual={actual_size} path={observed[relative]}"
            )
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "state-tree file SHA-256 mismatch: "
                f"expected={expected_sha256} actual={actual_sha256} path={observed[relative]}"
            )
        total_bytes += actual_size
    if payload["total_bytes"] != total_bytes:
        raise RuntimeError(
            "state-tree total byte count mismatch: "
            f"expected={payload['total_bytes']} actual={total_bytes}"
        )
    return {
        "file_count": len(expected),
        "manifest": str(manifest),
        "manifest_sha256": actual_manifest_sha256,
        "role": expected_role,
        "state_root": str(root),
        "total_bytes": total_bytes,
        "files": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--state-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--role", choices=sorted(ROLES), required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--state-root", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--expected-manifest-sha256", required=True)
    validate.add_argument("--role", choices=sorted(ROLES), required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = publish_state_tree_manifest(args.state_root, args.output, role=args.role)
        else:
            result = validate_state_tree_manifest(
                args.state_root,
                args.manifest,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_role=args.role,
            )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
