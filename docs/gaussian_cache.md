# Canonical Gaussian cache

This package defines the data boundary between an optional external Gaussian
teacher and FastWAM. It does not vendor Policy-Lightning or NoPoSplat code.
Canonical extraction imports an explicitly supplied external checkout only
after verifying its exact Git commit, clean worktree, configuration checksum,
and checkpoint SHA-256.

Teacher calls never pass all agents as one arbitrary multiview set. For each
observation batch and each requested agent, extraction forms exactly one
ordered pair `[head_camera_global, head_camera_agent_i]`, reshapes the input to
`[B*N,2,3,240,320]`, applies an in-memory `coor_type=unify` override, and caches
only pair view 1 (the agent). The global prediction is reference-only. The
manifest seals this as `pairing=global_agent_unify_v1`, so changing agent count
or ordering cannot create an agent-agent teacher pair.

## Tensor contract

Every cached frame is little-endian FP16 with shape `[13,H,W]` and channels:

```text
mean_x mean_y mean_z
cov_xx cov_xy cov_xz cov_yx cov_yy cov_yz cov_zx cov_zy cov_zz
opacity
```

Covariance is row-major `(i,j)`: a teacher covariance
`[...,H,W,3,3]` is first permuted to `[...,3,3,H,W]` and only then flattened
to nine channels. This intentionally corrects the legacy operation that
reshaped `[H,W,3,3]` directly to `[9,H,W]` and mixed spatial and matrix axes.

Two independent cache roots use the same channel schema:

- `canonical`: full-resolution `[13,240,320]`, every selected per-agent
  observation timestep. A formal canonical corpus uses `selection=all`.
- `compact`: `[13,28,40]`, derived from a canonical cache by opacity-aware
  moment matching. The active training cache should normally contain only the
  split/index current-frame keys, not every timestep.

The dataset-facing field name is `agent_gaussian`. The reader preserves the
requested agent ordering and returns FP16 `[N,13,H,W]`:

```python
from fastwam.datasets.gaussian_cache import FrameKey, GaussianCache

cache = GaussianCache.open(cache_root, verify="manifest")
key = FrameKey("task/motionplanning/demo.h5", "traj_0", 32, "panda-1")
assert cache.contains_frame(key)
cache.preflight_keys(all_split_keys)  # fail closed before training
sample = cache.get_agents(
    key.source_path,
    key.trajectory,
    key.timestep,
    ["panda-1", "panda-0"],
)
assert sample["agent_gaussian"].shape == (2, 13, 28, 40)
```

`GaussianCache` opens shard memmaps lazily through a per-process LRU (64 open
shards by default, configurable with `max_open_shards`). Eviction and `close()`
close the underlying mmap rather than only dropping Python references, and
pickling drops all handles so DataLoader workers reopen their own read-only
maps. This bounds file descriptors even when the merged cache contains one
shard per trajectory.

## Immutable layout and provenance

```text
cache-root/
  manifest.json
  COMPLETE
  selection.jsonl                 # only for selection=index
  shards/
    shard-000000-<sha-prefix>.f16
    ...
```

Shards are raw, contiguous little-endian FP16 frames. Non-final shards have a
logical size from 1 GiB through 4 GiB; the final shard may be smaller. A shard
is written and hashed under a task-owned node-local staging directory (normally
an exact `/tmp/...` root), streamed once to its content-addressed final output key, checked by readback size and,
by formal default, full SHA-256 readback, then removed from staging. At most one shard is staged;
large shard publication never relies on OSSFS rename, which is copy+delete and
non-atomic. `manifest.json` records every
shard checksum/byte count, source HDF5 relative path/size/SHA-256, schema,
teacher commit/checkpoint/config identity, selection, and per-stream segments.
`COMPLETE` is created last and pins the manifest SHA-256. Readers reject a
missing marker or mismatched manifest before opening a shard.

Formal GAU1 training additionally pins the chosen cache in the resolved Hydra
config. The four required environment variables are
`FASTWAM_GAUSSIAN_CACHE_DIR`,
`FASTWAM_GAUSSIAN_CACHE_MANIFEST_SHA256`,
`FASTWAM_GAUSSIAN_CACHE_SELECTION_SHA256`, and
`FASTWAM_GAUSSIAN_CACHE_SOURCE_IDENTITY_SHA256`. The last value is SHA-256 of
compact canonical JSON over the manifest's path-sorted source records, keeping
only `path`, `bytes`, and `sha256` with sorted object keys. Dataset
construction checks all three identities and every indexed frame before a
formal run can enter its first optimizer step.
The selection value is the compact/merged manifest's normalized
`selection.index_sha256`, not the original window-record JSONL hash. Before
freezing the compact manifest SHA, also confirm that
`derivation.parent_manifest_sha256` names the intended canonical cache; the
compact manifest hash then pins that linkage transitively.

Distributed extraction formally makes every whole trajectory one immutable
micro-part, with weight `observation_count * num_agents`. The verified corpus
therefore has 1,587 micro-parts. DSW workers (first formal path: four GPUs) or
DLC ranks claim those micro-parts by deterministic stride/dynamic assignment;
the work plan is not reduced to 32 large caches, so one failed trajectory does
not force a large multi-trajectory recomputation. The
compatibility option `--partition-unit source` instead balances whole files by
bytes. Multiple trajectory parts may reference one source HDF5; merge dedupes
it only when relative path, byte count, and SHA-256 are identical. Each GPU
writes a fully independent immutable cache under `parts/part-XXXXX`. A manifest-v2 merge
renumbers shard IDs and references those original objects as
`parts/part-XXXXX/shards/...`; it does not copy the canonical payload. Each
part retains its own legal small final shard. The merge rejects missing part
indices, conflicting shared-source provenance, duplicate streams, schema or
teacher differences, partition-plan mismatches, and incomplete trajectory
coverage. For every assigned trajectory it additionally requires the exact
agent set, the sealed observation count for every agent, stored counts matching
the full selection or normalized sparse index, and full-cache stored weight
equal to `observation_count * num_agents`.

A frame is addressed without an unstable global row number:

```text
(source relative path, trajectory, timestep, agent_name)
```

Stream segments map arithmetic timestep runs to immutable shard offsets. This
supports both all-frame canonical streams and sparse stride-16/stride-32
current-frame projections without a multi-million-line dense index.

## Selection JSONL

`--selection all` stores every agent observation, including the final
observation at `T_action`. `--selection index` accepts deduplicated JSONL. Each
record contains exact source/trajectory/timestep identity and either one agent
or an ordered list:

```json
{"source_path":"Task-rf/motionplanning/demo.h5","trajectory":"traj_0","timestep":16,"agent_names":["panda-0","panda-1"]}
```

The normalized key set is copied to `selection.jsonl` inside the cache and its
SHA-256/count are sealed in the manifest. Sparse readers fail closed for keys
outside that set. Build the projection JSONL from the exact FastWAM train/val
index so the active compact cache need not retain all observation frames.

The coordinator also seals the raw selection path/bytes/SHA-256, the canonical
normalized full key SHA/count, the planned-scope SHA/count, and one exact
SHA/count for every trajectory micro-part. Every worker verifies the same raw
bytes and all three normalized identity levels before CUDA or teacher startup.
Compact part manifests carry the corresponding identity and the merged compact
key union must equal the planned key set exactly. Canonical caches remain
independent full caches with `selection=all`.

## Extraction and compact projection

The external teacher checkpoint must be obtained and authorized separately.
The repository contains only an adapter and provenance checks:

The formal entry point is repository-owned; it does not require an external
Python callback module. First the coordinator hashes the checkpoint and each
source HDF5 once, discovers trajectory metadata, and seals an immutable plan:

### Formal storage gate (2026-08-02)

The live CPFS mount has exhausted unique-block capacity (`600T/600T`) and
returns `ENOSPC` on real writes even when a user quota view appears to retain
space. CPFS is therefore not an allowed formal cache-build destination at this
gate. The earlier two-trajectory cross-mount E2E remains valid diagnostic
evidence, but it does not authorize a full CPFS write.

Place canonical and compact outputs in two distinct, never-reused,
immutable/versioned OSS roots, for example:

```text
/oss-chengjuntao/fastwam-gaudp/v2/canonical/<producer-commit>-<plan-sha-prefix>/
/oss-chengjuntao/fastwam-gaudp/v2/compact/<producer-commit>-<plan-sha-prefix>/
```

Do not point both roles at the same root and do not overwrite an existing
version. Before training, publish a SHA-256 whole-file bundle manifest and use
`scripts/dlc_local_cache.sh` (via `scripts/train_zero2.sh`) to atomically copy
the compact cache from OSS into a specific node-local `/tmp/.../<manifest-sha>`
directory. Training must read `FASTWAM_GAUSSIAN_CACHE_DIR` from that verified
local bundle, not issue random reads against OSS and not fall back to CPFS.

```bash
# Replace this example with a fresh unique ID for every formal build.
FASTWAM_GAUDP_CACHE_VERSION=20260802T210000Z-fastwam-cleancommit

python -m fastwam.datasets.gaussian_cache.orchestrate plan \
  --plan-root /oss-chengjuntao/fastwam-gaudp/v2/plans/${FASTWAM_GAUDP_CACHE_VERSION} \
  --dataset-root /mnt/workspace/datasets/robofactory_multi_robot \
  --checkpoint /mnt/workspace/checkpoints/noposplat.ckpt \
  --checkpoint-sha256 <official-sha256> \
  --compact-selection-jsonl /path/to/train-val-current-frames.jsonl \
  --planned-worker-count 4 \
  --teacher-repository-commit c944b4989a89c99c69d2572ea870f6a04680f5e7 \
  --teacher-repository-url https://github.com/Ziyeeee/Policy-Lightning.git \
  --teacher-config-relative-path config/encoder/noposplat.yaml \
  --teacher-config-sha256 <pinned-config-sha256> \
  --teacher-training-provenance-json requirements/noposplat-mixre10kdl3dv-512-training-provenance.json \
  --producer-repo /path/to/clean/FastWAM

torchrun --standalone --nproc-per-node 4 \
  -m fastwam.datasets.gaussian_cache.orchestrate worker \
  --platform torchrun \
  --plan-root /oss-chengjuntao/fastwam-gaudp/v2/plans/${FASTWAM_GAUDP_CACHE_VERSION} \
  --dataset-root /mnt/workspace/datasets/robofactory_multi_robot \
  --canonical-output-root /oss-chengjuntao/fastwam-gaudp/v2/canonical/${FASTWAM_GAUDP_CACHE_VERSION} \
  --compact-output-root /oss-chengjuntao/fastwam-gaudp/v2/compact/${FASTWAM_GAUDP_CACHE_VERSION} \
  --compact-selection-jsonl /path/to/train-val-current-frames.jsonl \
  --staging-dir /tmp/fastwam-gaussian-staging \
  --teacher-repo /mnt/workspace/Policy-Lightning \
  --checkpoint /mnt/workspace/checkpoints/noposplat.ckpt

python -m fastwam.datasets.gaussian_cache.orchestrate merge-validate \
  --plan-root /oss-chengjuntao/fastwam-gaudp/v2/plans/${FASTWAM_GAUDP_CACHE_VERSION} \
  --dataset-root /mnt/workspace/datasets/robofactory_multi_robot \
  --canonical-output-root /oss-chengjuntao/fastwam-gaudp/v2/canonical/${FASTWAM_GAUDP_CACHE_VERSION} \
  --compact-output-root /oss-chengjuntao/fastwam-gaudp/v2/compact/${FASTWAM_GAUDP_CACHE_VERSION}
```

The plan and every part/final manifest carry the clean FastWAM producer commit,
tree and source-snapshot SHA-256. The worker loads the selection index and
teacher once per GPU, hashes each distinct assigned source HDF5 exactly once at
startup, and uses stat-only TOCTOU checks at worker completion. The coordinator
performs the one final global source checksum pass. It passes sealed source stat
tokens to each exact micro-part extraction. `--micro-part-index` and repeatable
`--trajectory relative/source.h5::traj_name` select exact units for diagnosis.
For the official two-trajectory E2E, repeat `--trajectory` on the `plan`
command itself; this creates a two-part sealed test plan against the original
HDF5 files, so merge remains authoritative. Omitting the selectors is the only
formal full-corpus mode.

Top-level merge uses an identity-sealed `MERGE.BUILDING.json` transaction. A
retry may remove only coordinator-owned top-level `manifest.json`, `COMPLETE`
and `selection.jsonl` from the matching failed merge; it never traverses or
deletes `parts/`. If `COMPLETE` was already durably sealed before a crash,
restart verifies the immutable seal and every identity bound by the marker,
then removes only the stale marker without rewriting cache metadata.

The lower-level commands remain useful for local diagnostics:

```bash
python -m fastwam.datasets.gaussian_cache.extract canonical \
  --dataset-root /path/to/robofactory_multi_robot \
  --output-root /oss-prefix/gaussian-canonical-v1/parts/part-00000 \
  --teacher-repo /external/Policy-Lightning \
  --teacher-commit <full-commit> \
  --teacher-config config/encoder/noposplat.yaml \
  --teacher-checkpoint /authorized/noposplat.ckpt \
  --teacher-checkpoint-sha256 <sha256> \
  --selection all \
  --target-shard-gib 2 \
  --staging-dir /tmp/fastwam-gaussian-staging \
  --partition-index 0 \
  --partition-count 1587 \
  --partition-unit trajectory \
  --compact-output-root /active/gaussian-compact-v1/parts/part-00000 \
  --compact-selection-jsonl /path/to/train-val-current-frames.jsonl

# Run/claim exact micro-part indices independently, then seal both zero-copy roots.
python -m fastwam.datasets.gaussian_cache.extract merge-part-manifests \
  --parts-root /oss-prefix/gaussian-canonical-v1/parts \
  --output-root /oss-prefix/gaussian-canonical-v1

python -m fastwam.datasets.gaussian_cache.extract merge-part-manifests \
  --parts-root /active/gaussian-compact-v1/parts \
  --output-root /active/gaussian-compact-v1 \
  --canonical-root /oss-prefix/gaussian-canonical-v1

python -m fastwam.datasets.gaussian_cache.extract compact \
  --canonical-root /oss-prefix/gaussian-canonical-v1 \
  --output-root /fast-active-storage/gaussian-compact-train-v1 \
  --selection index \
  --selection-jsonl /path/to/train-current-frames.jsonl \
  --target-shard-gib 2 \
  --staging-dir /tmp/fastwam-gaussian-staging
```

The paired `--compact-output-root` path is the formal compact-cache path. It
moment-matches selected frames immediately from the canonical tensor already
in memory and causes no extra teacher forward. After both part sets finish,
merge canonical first; compact merge then pins the final canonical manifest
SHA, teacher provenance, and canonical selection in its derivation record.
The separate `compact` projection command is retained for small/local caches
and diagnostics, but is not the formal 4.68 TiB workflow: projecting later
through OSSFS would require about 146,000 random frame reads across roughly
2,397 two-GiB objects.

Uploaded-object SHA-256 readback is enabled by default. The explicit
`--no-verify-uploaded-checksum` escape hatch is diagnostic-only and is not a
formal cache-build setting; size readback remains mandatory. Staging subdirectories are
unique per task and removed only after successful upload. On failure, only the
current task's known partial is deleted; final objects already uploaded remain
unreferenced because `COMPLETE` has not been written.

Checksum accounting is deliberate: publication performs exactly one strong
post-upload SHA-256 readback per new shard (the offloader returns a verified
receipt, preventing a duplicate writer read). Immediate transaction and worker
postchecks use manifest plus byte counts. A restart strongly hashes existing
parts before skipping them, and coordinator `merge-validate` performs one final
full shard checksum pass after the zero-copy merge. It does not hash each part
again during pre-merge, merge, and validation.

Each micro-part has an adjacent task-owned `part-XXXXX.BUILDING.json` identity
and append-only `part-XXXXX.JOURNAL.jsonl`. A restart may clear an incomplete
root only when that marker exactly matches task ID, work-plan SHA, role,
micro-part index, source/trajectory identity, and checkpoint identity. A root
containing `COMPLETE` is never deleted automatically, even if verification
finds corruption. If the canonical half seals and compact publication fails,
restart verifies and preserves canonical, clears only the owned incomplete
compact root, and derives compact from canonical without another teacher
forward.

The provider loads the official checkpoint with `weights_only=True`. Empty
`encoder.*` weights, shape mismatches, or missing core backbone/head weights
fail closed; all missing/unexpected keys are recorded. Provenance also records
the source YAML SHA, Hydra-composed encoder SHA, post-override resolved config
SHA, and exact override path. Encoder fragments such as
`config/encoder/noposplat.yaml` are composed against the pinned checkout's
`config/` tree so `defaults: [backbone: croco]` cannot silently remain
unresolved. A missing composed `backbone` is a hard error. Provenance also records
`usage_scope=research_noncommercial`: although the Policy-Lightning top level
is MIT, referenced NoPoSplat/CroCo files include CC BY-NC-SA 4.0 notices, and
the checkpoint repository declares no cardData/license. The cache must not be
treated as commercially redistributable without separate rights clearance.

Compact cells use opacity as mixture weight for Gaussian moments. For means `mu_i`, covariances
`Sigma_i`, and opacity `a_i`:

```text
mu = sum(a_i mu_i) / sum(a_i)
Sigma = sum(a_i (Sigma_i + mu_i mu_i^T)) / sum(a_i) - mu mu^T
opacity = sum(a_i) / number_of_pixels_in_cell
```

The opacity is an area-normalized density, not alpha union: it does not
artificially saturate as a cell contains more pixels. Explicit per-cell pixel
counts handle the non-uniform 240x320 to 28x40 geometry. Accumulation is FP32,
covariance is re-symmetrized, empty cells are zero, and the stored result is
FP16 `[13,28,40]`. Every packed FP32 value is checked against the finite FP16
range before casting, and the complete cast tensor is checked again; a large
between-mean covariance therefore fails the build instead of sealing `inf`.
Manifests identify this exact transform as
`opacity-aware-moment-matching-cell-mean-alpha-v2`.

## Official cache-build environment

Cache extraction runs in a dedicated teacher environment, not the FastWAM
training environment. The verified boundary is Python 3.12.13, CUDA 12.8,
Policy-Lightning commit `c944b4989a89c99c69d2572ea870f6a04680f5e7`, and
the pins in `requirements/gaussian-cache-teacher.lock`. In particular,
`jaxtyping==0.3.11` is required by the NoPoSplat encoder; the normal
`fastwam-py310` environment does not provide it. The formal DSW environment is
an isolated `include-system-site-packages=false` venv at
`/tmp/fastwam-teacher-noposplat-c944b49-py312-cu128-20260802`. Its full freeze,
`pip check`, runtime identity, source hashes, and real B1/N4 plus B8/N4 forward
logs are sealed at
`/oss-chengjuntao/artifacts/fastwam-teacher-env-proof-c944b49-20260802-v1`.
The `/tmp` venv is node-local and must be reconstructed from the lock if the
DSW node is replaced; the OSS proof is provenance, not an executable venv.

## Validation

Fast validation checks schema, `COMPLETE`, manifest SHA, shard existence and
exact byte counts. Formal validation additionally hashes every shard and source
HDF5 and deterministically checks first/middle/last tensor semantics for every
shard (therefore every micro-part):

```bash
python -m fastwam.datasets.gaussian_cache.validate \
  --cache-root /path/to/cache \
  --checksums \
  --source-root /path/to/robofactory_multi_robot \
  --source-checksums \
  --semantic-mode coverage
```

`--semantic-mode sample --semantic-sample-frames N` remains a diagnostic
option, while `--semantic-mode none` must never be used for a formal seal.

For the verified N=2/3/4 corpus, all 2,577,023 per-agent observation frames at
240x320 occupy exactly 5,145,799,526,400 bytes (4.680077406 TiB) as raw 13ch
FP16, before manifest/shard overhead. Do not attempt to materialize that corpus
on a 500 GiB personal filesystem or the currently full CPFS mount. Both the
canonical corpus and the much smaller compact projection are sealed in their
own immutable OSS roots; only the compact projection is copied into node-local
`/tmp` for the training hot path.
