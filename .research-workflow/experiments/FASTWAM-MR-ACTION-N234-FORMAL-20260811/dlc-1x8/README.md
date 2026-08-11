# FASTWAM native-agent N=2/3/4 formal DLC launcher R3

This directory contains a prepare-first, fail-closed 1 Worker x 8 GPU launcher
for three action-only runs under a canary promotion policy.  N=2 is submitted
first; N=3 and N=4 are eligible only after the same suite's N=2 run has durable,
structured scientific-completion evidence.  Nothing in this directory has
been submitted by creating or testing these files.  R3 uses new experiment
IDs, run IDs, output roots, suite/member ledger paths, and a distinct controller
lock; the submitted R1 identities and durable records are never reused.

The frozen R3 suite ID is
`FASTWAM-MR-ACTION-N234-FORMAL-R3-20260812`; its immutable source is
`/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/fastwam-action-n234-formal-r3-20260812-r1`,
its output prefix is
`/oss-chengjuntao/artifacts/fastwam-action-n234-formal-r3-20260812`, and its
control lock is
`/tmp/fastwam-dlc-submit-state/workspace-270969/action-n234-formal-r3-controller.lock`.
The three experiment IDs end in `N2-PLACEFOOD-1K-S42-R3-20260812`,
`N3-POOL-1K-S42-R3-20260812`, and `N4-STACKCUBE-1K-S42-R3-20260812`;
their run IDs use the corresponding `-r3-20260812` suffix.

The frozen experiment matrix is N=2 PlaceFood-rf, N=3
ThreeRobotsPlaceShoes-rf plus ThreeRobotsStackCube-rf, and N=4
FourRobotsStackCube-rf.  Every run uses 1000 steps, save/eval every 500 steps,
seed and split seed 42, native agent counts, and no fixed-capacity masked agent
set.  The external reservation contract is
`action_only_native_agents_1x8_v1`; the trainer's terminal contract fields stay
null because training uses `metadata_no_hash`.

The source reservation uses the strict, formal-specific portable
`fastwam-formal-source-content-binding-v1` schema.  It persists only canonical
relative paths and kinds plus regular-file size and Base64 content.  Filesystem
mode, timestamps, device, inode, and other mount-local identity remain
transient race checks and are not persisted.  Preparation and the worker use
the same fd-rooted, `O_NOFOLLOW` validator, so an immutable OSS source can be
copied to local `/tmp` even when the two mounts assign different metadata.

Member reservations use `fastwam-action-native-agents-reservation-v3` and a
single shared `fastwam-formal-portable-input-binding-v2` collector during both
prepare and worker live validation.  Dataset directories persist only their
canonical path and kind.  The initial checkpoint, VAE, and task text-cache
files persist canonical path, kind, and byte size.  Small control files -- the
normalization statistics and Gaussian completion markers -- additionally
persist their exact raw Base64 content.  Each Gaussian manifest binds every
original byte reversibly as one canonical Base64-encoded zlib level-6 stream;
the raw input is bounded at 64 MiB and the compressed representation at 16
MiB, and validation accepts exactly one complete stream with no trailing or
concatenated bytes.  This is an exact byte binding, not a hash and not a
semantic-only summary.  The decompressed JSON and completion marker must also
satisfy the frozen Gaussian semantics.  Mount-local mode, timestamp, device,
inode, and link metadata are used only for same-open race checks and are never
compared across mounts.

Preparation is two-phase across the whole N=2/3/4 suite.  It first builds and
validates every member request, source/input binding, and reservation entirely
in memory.  If any member fails, it creates no output parent, member or suite
reservation, and no local `PREPARED` state.  Only after all three pure results
pass does it create the new output parent, publish and read back all member
reservations, publish the suite marker, and finally record local prepared
state.  The suite marker is the atomic authorization boundary, not a rollback
mechanism: if phase two stops after publishing only some member records, those
records cannot authorize submission and the R3 identity must not be reused.
A failed R2 identity is never resumed or reused.

This no-hash contract deliberately does not claim content identity for a
dataset directory or for a same-size replacement of a large checkpoint, VAE,
or text cache.  Those inputs therefore also rely on their run-specific,
non-overwritten OSS paths and the recorded producer/run identity.  A new
identity and source snapshot are required if that external immutability
assumption cannot be maintained.

The runtime uses three independent eight-rank process worlds in one DLC pod:
fresh training stops at step 500, a new process world resumes the real
Accelerate/ZeRO-2 state and reaches step 1000, and a third process world loads
the local final state with zero updates.  Step-500 full state is local scratch
and is removed only after the trainer-native recovery receipt proves that
`accelerator.load_state` returned.  OSS receives self-contained full weights at
steps 500 and 1000, offline-eval and recovery/log receipts, and only the final
step-1000 full state.  Before publication the runtime validates both local
trainer manifests and completion markers against the actual files, steps, and
`checkpoint_state_kind=full`; these path-bound sidecars are not copied.  The first
formal output-directory mutation occurs only after all three worlds and local
validation succeed; `COMPLETE` is created exclusively and last.

The third phase proves a fresh-process, eight-rank load from local final state;
it does not claim a new-pod or remounted-OSS load.  Durable byte identity is
established by exclusive streaming plus close/reopen comparison, and the
self-contained full weights avoid an external base-checkpoint dependency.
All three worlds import `fastwam` only from the staged immutable source's
`src` directory; a prelaunch origin check rejects an installed or editable
package from another checkout.  The worker Python contract binds both the
logical venv entry point (needed to select its site-packages) and the exact
final regular CPFS interpreter reached by `readlink -f`; retargeting either is
rejected before training.

The publish cap is 62 GiB per run.  Preparation requires authoritative platform
quota evidence for at least 190 GiB free for the three-run suite; OSS FUSE `df`
is never accepted as quota evidence.  The worker requires 200 GiB local
`/tmp` free space and the job timeout is 2160 minutes (36 hours).  All output
publication uses exclusive stream copies plus close/reopen byte comparison;
the runtime does not use hard links, renames, or directory fsync on OSS.

Run the local, network-free checks with:

```bash
./run_static_tests.sh
```

Invoke the controller only on ssh970 through `submit_from_ssh970.sh`; the
wrapper pins and preflights the known PAI-SDK Python under `/mnt/workspace`.
With no explicit command it defaults to `prepare`; preparation validates paths and
then applies the suite-wide two-phase protocol above but makes no DLC API
mutation.  `submit` requires a single member plus its exact experiment ID and
writes a permanent latch before the controller's sole non-retrying CreateJob
call.

Formal submission is deliberately staged.  Submit N=2 as the canary; N=2 has
no predecessor-completion requirement.  Do not submit N=3 or N=4 merely because
N=2 is running, has stopped, or the platform reports `Succeeded`.  Before any
SDK load, job listing, submission latch, local submission state, or CreateJob
for N=3/N=4, the controller re-reads the exact suite and N=2 reservations,
checks their shared source and request basis, validates the live N=2 reservation
without requiring its output to remain absent, and validates N=2's `COMPLETE`
plus structured terminal evidence.  Promotion requires the resulting status
to be exactly `SCIENTIFIC_COMPLETE`; absent, malformed, incomplete, stale, or
inconsistent evidence fails closed before a downstream submission can become
durable.  Once that gate passes, N=3 and N=4 may be submitted independently,
including in parallel.  This ordering is a launch-safety policy, not a claim
that the three tasks are scientifically interchangeable.

Reconciliation reports platform state separately and only reports
`SCIENTIFIC_COMPLETE` after re-reading the exact durable allowlist, COMPLETE
and terminal receipts, both native recovery receipts, offline evaluations, and
the final trainer state.  A cloud `Succeeded` status alone is not completion.

Publication is immutable and fail-closed.  If a worker fails after exclusively
creating part of its unique output root but before `COMPLETE`, that run ID is
not automatically retried, cleaned, or reused; after diagnosis, an operator
must prepare an explicitly new run ID and reservation.
