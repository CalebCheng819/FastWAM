import os

import pytest
import torch

from fastwam.datasets.gaussian_cache import (
    GaussianCacheBuilder,
    GaussianCacheSchema,
    pack_gaussian_channels,
    sha256_file,
    source_record,
)
from fastwam.datasets.gaussian_cache.transaction import (
    UnsafeCacheRestartError,
    prepare_cache_build,
    run_paired_micro_part,
)


def _frame(cache_kind: str) -> torch.Tensor:
    height, width = (28, 40) if cache_kind == "compact" else (32, 40)
    means = torch.zeros(1, 3, height, width)
    covariance = torch.zeros(1, 3, 3, height, width)
    covariance[:, 0, 0] = 1.0
    covariance[:, 1, 1] = 1.0
    covariance[:, 2, 2] = 1.0
    opacity = torch.full((1, 1, height, width), 0.2)
    return pack_gaussian_channels(means, covariance, opacity).half()


def _seal(root, source, staging, cache_kind):
    frame = _frame(cache_kind)
    builder = GaussianCacheBuilder(
        root,
        GaussianCacheSchema(frame.shape[-2], frame.shape[-1], cache_kind),
        sources=[source],
        teacher={"kind": "synthetic-test", "checkpoint_sha256": "0" * 64},
        selection={"mode": "all", "selected_key_count": 1},
        staging_dir=staging,
    )
    builder.append_stream(
        source_path="demo.h5",
        trajectory="traj_0",
        agent_name="panda-0",
        observation_count=1,
        timesteps=[0],
        frames=frame,
    )
    return builder.finish()


def _source(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "demo.h5"
    source_path.write_bytes(b"source")
    return source_record(source_path, source_root=source_root)


def _transaction_args():
    return {
        "task_id": "pytest-transaction",
        "work_plan_sha256": "a" * 64,
        "micro_part_index": 0,
        "work_identity": {
            "source_path": "demo.h5",
            "trajectory": "traj_0",
            "checkpoint_sha256": "0" * 64,
        },
    }


def test_late_interruption_restart_clears_only_task_owned_incomplete_part(tmp_path):
    source = _source(tmp_path)
    canonical = tmp_path / "canonical" / "parts" / "part-00000"
    compact = tmp_path / "compact" / "parts" / "part-00000"
    args = _transaction_args()
    assert (
        prepare_cache_build(canonical, role="canonical", **args) == "new"
    )
    canonical.mkdir()
    interrupted = canonical / "late-upload.partial"
    interrupted.write_bytes(b"partial")

    build_calls = 0

    def build_both():
        nonlocal build_calls
        build_calls += 1
        assert not interrupted.exists()
        _seal(canonical, source, tmp_path / "staging", "canonical")
        _seal(compact, source, tmp_path / "staging", "compact")

    result = run_paired_micro_part(
        canonical,
        compact,
        build_both=build_both,
        recover_compact_from_canonical=lambda: pytest.fail("unexpected recovery"),
        **args,
    )
    assert result["status"] == "built"
    assert build_calls == 1
    assert (canonical / "COMPLETE").is_file()
    assert (compact / "COMPLETE").is_file()


def test_dual_output_failure_recovers_compact_without_teacher_recompute(tmp_path):
    source = _source(tmp_path)
    canonical = tmp_path / "canonical" / "parts" / "part-00000"
    compact = tmp_path / "compact" / "parts" / "part-00000"
    args = _transaction_args()
    teacher_calls = 0

    def fail_after_canonical_seal():
        nonlocal teacher_calls
        teacher_calls += 1
        _seal(canonical, source, tmp_path / "staging", "canonical")
        compact.mkdir()
        (compact / "compact-upload.partial").write_bytes(b"partial")
        raise RuntimeError("simulated compact upload failure")

    with pytest.raises(RuntimeError, match="compact upload failure"):
        run_paired_micro_part(
            canonical,
            compact,
            build_both=fail_after_canonical_seal,
            recover_compact_from_canonical=lambda: pytest.fail("unexpected first recovery"),
            **args,
        )
    canonical_manifest_sha = sha256_file(canonical / "manifest.json")

    recovery_calls = 0

    def recover_compact():
        nonlocal recovery_calls
        recovery_calls += 1
        assert not (compact / "compact-upload.partial").exists()
        _seal(compact, source, tmp_path / "staging", "compact")

    result = run_paired_micro_part(
        canonical,
        compact,
        build_both=lambda: pytest.fail("sealed canonical must not be recomputed"),
        recover_compact_from_canonical=recover_compact,
        **args,
    )
    assert result["status"] == "compact-recovered"
    assert teacher_calls == 1
    assert recovery_calls == 1
    assert sha256_file(canonical / "manifest.json") == canonical_manifest_sha
    assert (canonical / "COMPLETE").is_file()
    assert (compact / "COMPLETE").is_file()


def test_restart_rejects_foreign_marker_and_invalid_complete_without_deleting(tmp_path):
    root = tmp_path / "canonical" / "parts" / "part-00000"
    args = _transaction_args()
    prepare_cache_build(root, role="canonical", **args)
    marker = root.with_name("part-00000.BUILDING.json")
    os.chmod(marker, 0o644)
    marker.write_text("{}\n", encoding="utf-8")
    root.mkdir()
    survivor = root / "survivor"
    survivor.write_bytes(b"keep")
    with pytest.raises(UnsafeCacheRestartError, match="identity does not match"):
        prepare_cache_build(root, role="canonical", **args)
    assert survivor.read_bytes() == b"keep"

    (root / "COMPLETE").write_text("{}\n", encoding="utf-8")
    with pytest.raises(UnsafeCacheRestartError, match="invalid COMPLETE"):
        prepare_cache_build(root, role="canonical", **args)
    assert survivor.read_bytes() == b"keep"
