from __future__ import annotations

import hashlib
import importlib.util
import os
import threading
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PROVENANCE = REPO_ROOT / "src" / "fastwam" / "runtime_provenance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "fastwam_runtime_provenance_atomic_test", RUNTIME_PROVENANCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ready_marker_is_hidden_until_its_payload_is_fully_fsynced(
    tmp_path: Path,
) -> None:
    module = _load_module()
    target = tmp_path / "config.yaml"
    payload = b"model: fastwam\n"
    digest = hashlib.sha256(payload).hexdigest()
    ready = tmp_path / f".config.yaml.ready.{digest}"
    marker_write_started = threading.Event()
    release_marker_write = threading.Event()
    original_write = module._write_fsynced_exclusive

    def delayed_write(path: Path, content: bytes, mode: int) -> None:
        if ".ready." not in path.name:
            original_write(path, content, mode)
            return
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        try:
            assert os.write(descriptor, content[:1]) == 1
            os.fsync(descriptor)
            marker_write_started.set()
            assert release_marker_write.wait(timeout=2)
            view = memoryview(content)[1:]
            while view:
                written = os.write(descriptor, view)
                assert written > 0
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    module._write_fsynced_exclusive = delayed_write
    results: list[str] = []
    errors: list[BaseException] = []

    def invoke(rank: int) -> None:
        try:
            results.append(
                module.publish_rank_zero_file(
                    target, payload, rank=rank, world_size=2, timeout_seconds=2
                )
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    waiter = threading.Thread(target=invoke, args=(1,))
    owner = threading.Thread(target=invoke, args=(0,))
    waiter.start()
    owner.start()
    assert marker_write_started.wait(timeout=2)
    assert not ready.exists()
    assert any(
        ".ready." in path.name and ".tmp." in path.name for path in tmp_path.iterdir()
    )
    time.sleep(0.15)
    assert not errors
    release_marker_write.set()
    owner.join(timeout=3)
    waiter.join(timeout=3)

    assert not owner.is_alive() and not waiter.is_alive()
    assert not errors
    assert results == [digest, digest]
    assert ready.read_bytes() == f"sha256={digest}\n".encode("ascii")
    assert not [path for path in tmp_path.iterdir() if ".tmp." in path.name]


def test_ready_marker_no_clobber_rejects_different_existing_payload(
    tmp_path: Path,
) -> None:
    module = _load_module()
    target = tmp_path / "config.yaml"
    payload = b"seed: 42\n"
    digest = hashlib.sha256(payload).hexdigest()
    ready = tmp_path / f".config.yaml.ready.{digest}"
    ready.write_bytes(b"partial")

    with pytest.raises(
        RuntimeError, match="no-clobber collision has different content"
    ):
        module.publish_rank_zero_file(
            target, payload, rank=0, world_size=1, timeout_seconds=1
        )

    assert ready.read_bytes() == b"partial"


def test_ready_marker_no_clobber_is_idempotent_for_identical_payload(
    tmp_path: Path,
) -> None:
    module = _load_module()
    target = tmp_path / "config.yaml"
    payload = b"seed: 42\n"
    digest = hashlib.sha256(payload).hexdigest()

    first = module.publish_rank_zero_file(
        target, payload, rank=0, world_size=1, timeout_seconds=1
    )
    second = module.publish_rank_zero_file(
        target, payload, rank=0, world_size=1, timeout_seconds=1
    )

    assert first == second == digest
    assert len(list(tmp_path.glob(".config.yaml.ready.*"))) == 1


def test_exclusive_writer_retries_short_writes(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    target = tmp_path / "payload"
    payload = b"abcdefghijklmnopqrstuvwxyz"
    original_write = module.os.write

    def short_write(descriptor: int, content) -> int:
        return original_write(descriptor, content[:3])

    monkeypatch.setattr(module.os, "write", short_write)
    module._write_fsynced_exclusive(target, payload, 0o440)

    assert target.read_bytes() == payload


@pytest.mark.parametrize("operation", ["replace", "noreplace"])
def test_atomic_publish_cleans_temporary_after_write_error(
    tmp_path: Path, monkeypatch, operation: str
) -> None:
    module = _load_module()
    target = tmp_path / operation

    def fail_after_creating_temporary(path: Path, payload: bytes, mode: int) -> None:
        del payload, mode
        path.write_bytes(b"partial")
        raise OSError("injected write failure")

    monkeypatch.setattr(
        module, "_write_fsynced_exclusive", fail_after_creating_temporary
    )
    with pytest.raises(OSError, match="injected write failure"):
        if operation == "replace":
            module._atomic_replace(target, b"payload")
        else:
            module._atomic_publish_noreplace(target, b"payload", 0o440)

    assert not target.exists()
    assert not [path for path in tmp_path.iterdir() if ".tmp." in path.name]


def test_ready_marker_reader_rejects_symlink(tmp_path: Path) -> None:
    module = _load_module()
    target = tmp_path / "config.yaml"
    payload = b"seed: 42\n"
    digest = hashlib.sha256(payload).hexdigest()
    target.write_bytes(payload)
    marker_payload = tmp_path / "marker-payload"
    marker_payload.write_text(f"sha256={digest}\n", encoding="ascii")
    ready = tmp_path / f".config.yaml.ready.{digest}"
    ready.symlink_to(marker_payload)

    with pytest.raises(RuntimeError, match="must not be a symlink"):
        module.publish_rank_zero_file(
            target, payload, rank=1, world_size=2, timeout_seconds=1
        )
