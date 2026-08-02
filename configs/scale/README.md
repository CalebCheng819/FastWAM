# RoboFactory multi-robot scale profiles

`robofactory_multi_robot_32gpu` is a formal DLC 4-node x 8-GPU override. It is
not a generic tuning preset: `scripts/train_zero2.sh` rejects any topology other
than PAI `WORLD_SIZE=4`, `NPROC_PER_NODE=8`, and node `RANK=0..3`, including in
dry-run. The profile uses gradient accumulation 1 and a deterministic 12-window
offline validation-loss subset every 1,000 optimizer steps. This is not a robot
simulator or task-success metric.

The training sampler is task- and agent-count-balanced with replacement. Its
objective is therefore intentionally reweighted rather than empirical-risk
minimization under the raw N=2/3/4 trajectory proportions. VG0 versus VG1 is a
composite VideoGen-paradigm treatment: it jointly changes future-video targets,
`training_mode`, video loss, and trainable scope. Report it as that complete
treatment, not as a loss-coefficient-only causal ablation.

## Storage layout after the CPFS ENOSPC incident

The live CPFS block pool was observed at 600 TiB / 600 TiB with zero writable
blocks even though the user's quota view still showed nominal headroom. Do not
write a new compact cache or eRDMA package there and do not delete data as a
workaround.

- CPFS is a read-only source for the 24 RoboFactory H5 files, normalization
  stats, text embeddings, the 12,041,735,140-byte official checkpoint, and the
  1,409,401,152-byte Wan2.2 VAE.
- `/oss-chengjuntao` is a required DLC mount and is the immutable source for the
  compact Gaussian cache. The eRDMA helper independently verifies its small
  versioned OSS bundle and stages it into its own content-addressed `/tmp`
  runtime. The
  full-resolution all-timestep 13-channel FP16 corpus also remains canonical on
  OSS; it is not a per-step training hot path.
- Every node copies the CPFS training bundle to a fixed
  `/tmp/fastwam-whole-file-cache` root, verifies every whole-file SHA-256, and
  publishes READY atomically. GAU1 arms additionally copy the compact-Gaussian
  OSS bundle; GAU0 arms forbid all Gaussian OSS inputs and never stage that
  irrelevant asset. Accelerate receives only the local paths used by its arm.
- Every formal arm forces `checkpoint_state_kind=full`; VG0/action-only weights
  therefore contain a full `mot` and never serialize a node-local `/tmp` base
  dependency. Every full Accelerate state is sealed after the global save
  barrier by a canonical JSON manifest of `{path,bytes,sha256}` records.
- Formal output may be under CPFS only if the exclusive create/write/fsync/
  atomic-rename probe succeeds. `/oss-chengjuntao` output is experimental and
  is rejected unless a hash-pinned real DeepSpeed ZeRO-2 save/load roundtrip
  smoke marker passes `scripts/validate_zero_checkpoint_smoke.py`. That gate
  requires a real 32-rank, two-process-boundary save/mutate/load smoke, an
  exact OCI image digest, and a live-verified complete state-tree manifest.
  A hand-written boolean JSON marker cannot authorize OSS output.

The fixed release identities are:

- FastWAM checkpoint SHA-256:
  `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`
- Wan2.2 VAE SHA-256:
  `0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996`
- N=2/3/4 train-s42 stats (3,604 bytes) SHA-256:
  `350493b685d8db0ea4cfd66f58f49849e8cd1f65cecc269f15aff9101ac8a04d`
- eRDMA bundle SHA-256:
  `8f2c1c43d64a7745bea19bfe4cd1383344c9cf32779166f4aa67809ebf1f5fab`
- eRDMA source-manifest SHA-256:
  `f05443faa27533274ae1b322723e21ac09bd80bd5b2513638dd2619c67552215`
- eRDMA exported-environment SHA-256:
  `b581a454249ad2a27ef21dad929a0db6d963a6613340bce10a866ff40017c11c`

## Building the two training manifests

The repository generator requires explicit relative includes, rejects
traversal/symlinks/special files, byte-sorts entries, and publishes without
overwriting an existing manifest:

```bash
python scripts/build_whole_file_manifest.py \
  --source-root /cpfs/user/chengjuntao \
  --include datasets/robofactory_multi_robot \
  --include checkpoints/FastWAM/yuanty-fastwam-139eebb6d90cdd9bdbbe465f72c6edc9ad5a518a/libero_uncond_2cam224.pt \
  --include checkpoints/FastWAM/model-cache/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors \
  --output /oss-chengjuntao/manifests/robofactory-cpfs-bundle.sha256

python scripts/build_whole_file_manifest.py \
  --source-root /oss-chengjuntao \
  --include fastwam/gaussian/compact/<version> \
  --output /oss-chengjuntao/manifests/robofactory-oss-bundle.sha256
```

Review the generated manifests before launch. Their file-list SHA-256 values,
the three Gaussian semantic identities, the code commit, image provenance, and
runtime identities are sealed in rank-0's exclusive `.RUN_RESERVED` marker.
Other node launchers wait for and compare the exact marker. Fresh launches
never reuse an existing output. The only reuse path is an explicit full-state
resume inside the same previously reserved output: rank zero verifies the
complete sealed state tree and publishes an immutable resume-validation marker;
the other nodes wait for that exact marker.

## Formal environment skeleton

Each DLC node receives the same non-login-shell command and environment except
for PAI `RANK`. PAI supplies `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, and
`MASTER_PORT`.

Formal user CLI is deliberately a two-selector allowlist: exactly one of the
eight committed
`task=robofactory_multi_robot_vg{0,1}_hub{0,1}_gau{0,1}_224_1e-4` arms and
exactly `+scale=robofactory_multi_robot_32gpu`. Any other Hydra flag or override
is rejected, including direct changes to hub/Gaussian/VideoGen treatment,
losses, trainable scope, required agent counts, seed, training schedule, eval,
or checkpoint policy. This prevents the recorded task name from disagreeing
with the resolved scientific contract. It does not reject the launcher's fixed
output, resume, local-data, stats and `checkpoint_state_kind=full` overrides:
those are constructed internally only after user-argument validation and are
appended last.

`FASTWAM_LAUNCHER_UNIT_TEST_ALLOW_DIRTY` and
`FASTWAM_LAUNCHER_UNIT_TEST_SKIP_ENV_PREFLIGHT` are test-harness controls, not
formal launch controls. A formal non-dry-run rejects either truthy value before
Git identity/cleanliness checks, output reservation validation, or Python
preflight. Consequently, the clean immutable producer checkout and the exact
`pyproject.toml` package pins plus `python -m pip check` are mandatory and have
no environment-variable bypass. The controls may be used only by an explicit
parameter-only dry-run or a non-formal launcher mock; neither authorizes a DLC
reservation.

```bash
export RUN_ID=fastwam-mr-formal-s42
export NPROC_PER_NODE=8
export FASTWAM_CODE_COMMIT=<40-hex-clean-HEAD>
export FASTWAM_FORMAL_OUTPUT_DIR=<nonexistent-path-ending-in-$RUN_ID>

export FASTWAM_LOCAL_CACHE_ENABLED=1
export FASTWAM_LOCAL_CACHE_ROOT=/tmp/fastwam-whole-file-cache
export FASTWAM_LOCAL_RUNTIME_ROOT=/tmp/fastwam-local-runtime
export FASTWAM_CPFS_BUNDLE_SOURCE_ROOT=/cpfs/user/chengjuntao
export FASTWAM_CPFS_BUNDLE_MANIFEST=/oss-chengjuntao/manifests/robofactory-cpfs-bundle.sha256
export FASTWAM_CPFS_BUNDLE_MANIFEST_SHA256=<64-hex>
export FASTWAM_TRAINING_ENV_BUNDLE_MANIFEST_SHA256=<64-hex-offline-wheelhouse-SHA256SUMS>
# The following three OSS bundle variables are required for GAU1 only. Leave
# them unset for GAU0; the formal launcher rejects a contaminated baseline.
export FASTWAM_OSS_BUNDLE_SOURCE_ROOT=/oss-chengjuntao
export FASTWAM_OSS_BUNDLE_MANIFEST=/oss-chengjuntao/manifests/robofactory-oss-bundle.sha256
export FASTWAM_OSS_BUNDLE_MANIFEST_SHA256=<64-hex>

export FASTWAM_LOCAL_CHECKPOINT_RELATIVE_PATH=<relative-checkpoint-file>
export FASTWAM_LOCAL_DATASET_RELATIVE_ROOT=datasets/robofactory_multi_robot
export FASTWAM_LOCAL_STATS_RELATIVE_PATH=datasets/robofactory_multi_robot/fastwam_multi_robot_n234_train_s42_stats_v2.json
export FASTWAM_LOCAL_TEXT_EMBEDS_RELATIVE_ROOT=datasets/robofactory_multi_robot/text_embeds_cache_n234
export FASTWAM_LOCAL_MODEL_CACHE_RELATIVE_ROOT=checkpoints/FastWAM/model-cache
export FASTWAM_LOCAL_VAE_RELATIVE_PATH=checkpoints/FastWAM/model-cache/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
# GAU1 only; leave unset for GAU0.
export FASTWAM_LOCAL_GAUSSIAN_RELATIVE_ROOT=fastwam/gaussian/compact/<version>

# All three semantic identities are GAU1 only; leave unset for GAU0.
export FASTWAM_GAUSSIAN_CACHE_MANIFEST_SHA256=<64-hex>
export FASTWAM_GAUSSIAN_CACHE_SELECTION_SHA256=<64-hex>
export FASTWAM_GAUSSIAN_CACHE_SOURCE_IDENTITY_SHA256=<64-hex>

export FASTWAM_DLC_IMAGE_REFERENCE=dsw-registry-vpc.cn-beijing.cr.aliyuncs.com/pai/pytorch:2.7.1-gpu-py310-cu128-ubuntu22.04-3995b779-1764350887
export FASTWAM_DLC_IMAGE_DIGEST=sha256:a57915104bffd280400d5a2a2a0af8f5987f2752347bd58e05ca91547489f265

bash scripts/train_zero2.sh 8 \
  task=robofactory_multi_robot_vg1_hub1_gau1_224_1e-4 \
  +scale=robofactory_multi_robot_32gpu
```

`docker/prepare-erdma-userspace.sh` is sourced and called in the same non-login
launcher shell after local bundle preparation and before the global collective.
Its `LD_LIBRARY_PATH`, `IBV_CONFIG_DIR`, `IBV_DRIVERS=erdma`,
`RDMAV_DRIVERS=erdma`, and PATH exports survive into Accelerate. Formal mode
sets `NCCL_IB_HCA=erdma`, requires measured bandwidth, rejects
`NET/IB : No device found` and Socket fallback, and requires NET/IB eRDMA log
evidence.

## Exact full-state resume

`scripts/state_tree_manifest.py` is invoked automatically after each formal
Accelerate state save. To resume, keep the same `RUN_ID` and
`FASTWAM_FORMAL_OUTPUT_DIR`; select a state directory underneath that output
and provide its adjacent immutable manifest and both exact hashes:

```bash
export FASTWAM_FORMAL_RESUME_STATE_DIR="$FASTWAM_FORMAL_OUTPUT_DIR/checkpoints/state/step_001000"
export FASTWAM_FORMAL_RESUME_STATE_MANIFEST="$FASTWAM_FORMAL_OUTPUT_DIR/checkpoints/state/step_001000.state-tree.json"
export FASTWAM_FORMAL_RESUME_STATE_MANIFEST_SHA256=<64-hex>
export FASTWAM_FORMAL_RESUME_TRAINER_STATE_SHA256=<64-hex-from-manifest>
```

The formal local cache and runtime roots are fixed because dataset and
normalization absolute paths are part of the strict trainer run contract.
Changing either path causes resume refusal before `accelerator.load_state`.
New-run reservation keeps its short 300-second wait, while full-state tree
validation uses the independent `FASTWAM_RESUME_VALIDATION_TIMEOUT` (21,600
seconds by default). Formal checkpoint publication likewise uses six-hour
process-group and filesystem-marker watchdogs; rank-zero weight copies and
state-tree hashes no longer leave the other ranks blocked inside a collective.

## Real 32-rank OSS checkpoint smoke

First try to recover the actual pulled digest from a minimal diagnostic DLC
pod. The probe reads an injected image ID if present, otherwise queries only
its own pod status with the service-account token; it outputs normalized image
digests and never prints the token or API response body:

```bash
python scripts/probe_pod_image_digest.py
```

If RBAC exposes a digest, set `FASTWAM_DLC_IMAGE_DIGEST=sha256:<64-hex>` and run
the following command on all four 8-GPU DLC nodes (same environment except PAI
`RANK`). The script clears PAI's node-level `WORLD_SIZE/RANK` before each
Accelerate launch, runs save and load in separate 32-rank process worlds, and
checks model, optimizer, scheduler, RNG and global-step restoration:

```bash
export FASTWAM_ZERO_SMOKE_OUTPUT_ROOT=/oss-chengjuntao/fastwam/smoke/<new-id>
export FASTWAM_CODE_COMMIT=<clean-40-hex-HEAD>
export FASTWAM_DLC_IMAGE_REFERENCE=<exact-PAI-image-reference>
export FASTWAM_DLC_IMAGE_DIGEST=sha256:<64-hex>
bash scripts/run_zero2_checkpoint_smoke.sh
```

Use the resulting `zero2-roundtrip-smoke.json` path and its SHA-256 as
`FASTWAM_OUTPUT_ZERO_CHECKPOINT_SMOKE_MARKER` and
`FASTWAM_OUTPUT_ZERO_CHECKPOINT_SMOKE_SHA256`. Until that real smoke succeeds,
OSS formal output remains blocked. If pod-status RBAC cannot reveal an OCI
digest, the stricter OSS gate also remains blocked; do not fabricate one.

`FASTWAM_GAUSSIAN_CACHE_SELECTION_SHA256` is the SHA-256 of normalized compact
`selection.jsonl`, not the original window-record JSONL. The source identity is
SHA-256 over canonical JSON for sorted
`[{"bytes":...,"path":...,"sha256":...}]` records.
