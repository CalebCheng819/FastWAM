# FASTWAM native-agent N=2/3/4 formal DLC launcher

This directory contains a prepare-first, fail-closed 1 Worker x 8 GPU launcher
for three independent action-only runs.  Nothing in this directory has been
submitted by creating or testing these files.

The frozen experiment matrix is N=2 PlaceFood-rf, N=3
ThreeRobotsPlaceShoes-rf plus ThreeRobotsStackCube-rf, and N=4
FourRobotsStackCube-rf.  Every run uses 1000 steps, save/eval every 500 steps,
seed and split seed 42, native agent counts, and no fixed-capacity masked agent
set.  The external reservation contract is
`action_only_native_agents_1x8_v1`; the trainer's terminal contract fields stay
null because training uses `metadata_no_hash`.

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
writes the immutable reservation but makes no DLC API mutation.  `submit`
requires a single member plus its exact experiment ID and writes a permanent
latch before the controller's sole non-retrying CreateJob call.

Reconciliation reports platform state separately and only reports
`SCIENTIFIC_COMPLETE` after re-reading the exact durable allowlist, COMPLETE
and terminal receipts, both native recovery receipts, offline evaluations, and
the final trainer state.  A cloud `Succeeded` status alone is not completion.

Publication is immutable and fail-closed.  If a worker fails after exclusively
creating part of its unique output root but before `COMPLETE`, that run ID is
not automatically retried, cleaned, or reused; after diagnosis, an operator
must prepare an explicitly new run ID and reservation.
