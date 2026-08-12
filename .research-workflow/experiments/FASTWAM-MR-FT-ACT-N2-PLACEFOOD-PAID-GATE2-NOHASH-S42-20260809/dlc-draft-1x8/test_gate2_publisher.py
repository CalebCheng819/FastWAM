#!/usr/bin/env python3
"""Dependency-free checks for the Gate2 OSS publication boundary."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
PUBLISHER = HERE / "publish_gate2.py"
RUNTIME = HERE / "runtime.sh"


def load_publisher():
    spec = importlib.util.spec_from_file_location("gate2_publisher_under_test", PUBLISHER)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load Gate2 publisher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Gate2PublisherTest(unittest.TestCase):
    def test_runtime_uses_dedicated_publisher_without_bulk_output_copy(self) -> None:
        runtime = RUNTIME.read_text(encoding="utf-8")
        self.assertIn('"${FASTWAM_PYTHON}" "${PUBLISHER_SCRIPT}"', runtime)
        self.assertIn('--oss-output-root "${FASTWAM_OSS_OUTPUT_ROOT}"', runtime)
        self.assertNotIn('cp -a -- "${PUBLISH_ROOT}/."', runtime)

        publisher = PUBLISHER.read_text(encoding="utf-8")
        for forbidden in ("os.fsync", "os.link", "os.rename", "os.replace"):
            self.assertNotIn(forbidden, publisher)
        self.assertIn("os.O_EXCL", publisher)
        self.assertIn("temporary_probe_removed_after_readback", publisher)
        self.assertLess(
            publisher.index('output / "publication_receipt.json"'),
            publisher.index('output / "COMPLETE.json"'),
        )

    def test_tiny_tree_is_compared_and_large_probe_is_exactly_reclaimed(self) -> None:
        module = load_publisher()
        with tempfile.TemporaryDirectory(prefix="fastwam-gate2-publish-test-") as tmp:
            root = Path(tmp)
            stage = root / "stage"
            # Mirror a fresh OSS prefix: neither the shared parent nor the
            # run-owned output exists before publication.
            output = root / "missing-shared-prefix" / "output"
            (stage / "load_world/checkpoints/weights").mkdir(parents=True)
            (stage / "load_world/checkpoints/state/step_000002").mkdir(parents=True)
            (stage / "final_verify_world").mkdir(parents=True)

            for name in (
                "gate2_trainer_evidence.json",
                "real_data_nohash_preflight.json",
                "real_data_nohash_preflight.log",
                "gaussian_primary_staging.json",
                "vae_staging.json",
                "save_world.log",
                "load_world.log",
                "final_verify_world.log",
            ):
                (stage / name).write_text(f"{name}\n", encoding="utf-8")
            (stage / "load_world/recovery_load_receipt.json").write_text(
                "load\n", encoding="utf-8"
            )
            (stage / "final_verify_world/recovery_load_receipt.json").write_text(
                "verify\n", encoding="utf-8"
            )
            (stage / "load_world/checkpoints/weights/step_000002.pt").write_bytes(
                b"full-weights\n"
            )
            state = stage / "load_world/checkpoints/state/step_000002"
            (state / "trainer_state.json").write_text("{}\n", encoding="utf-8")
            (state / "model_states.pt").write_bytes(b"state-shard\n")

            previous = sys.argv
            sys.argv = [
                str(PUBLISHER),
                "--local-stage",
                str(stage),
                "--oss-output-root",
                str(output),
                "--submission-tag",
                "tiny-gate",
            ]
            try:
                module.main()
            finally:
                sys.argv = previous

            receipt = json.loads((output / "publication_receipt.json").read_text())
            complete = json.loads((output / "COMPLETE.json").read_text())
            self.assertEqual(complete["status"], "COMPLETE")
            self.assertTrue(
                receipt["temporary_large_probe_objects_removed_after_readback"]
            )
            self.assertTrue(receipt["directly_compared_bytes"] > receipt["retained_bytes"])
            self.assertFalse(
                (output / "artifacts/load_world/checkpoints/weights/step_000002.pt").exists()
            )
            self.assertFalse(
                (output / "artifacts/load_world/checkpoints/state/step_000002").exists()
            )
            self.assertTrue((output / "artifacts/gate2_trainer_evidence.json").is_file())
            self.assertTrue(
                (output / "artifacts/final_verify_world/recovery_load_receipt.json").is_file()
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
