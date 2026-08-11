# Gate2 R3 DLC 1x8 design (not submitted)

This directory is a submission draft only.  Static validation does not call
PAI, create a DLC job, stop E38, or touch E38 artifacts.

## Fixed contract

- Workspace `270969`, quota `quotaksvqq2oh2pg`.
- The candidate is exactly one `Worker` pod with 8 GPUs.  There is no
  artificial workspace-wide GPU ceiling and no dependency on E38 being active.
  Live PAI resource and scheduling responses remain authoritative.
- CPFS datasource `d-a5mu77ymwjio71dkmw` is mounted `RO`.  OSS datasource
  `d-n7rly4fll0q2z6v91h` is mounted `RW`.
- Training uses task
  `robofactory_multi_robot_ft_n2_placefood_vg0_hub1_gau1_224_3e-5_nohash_gate`
  and `metadata_no_hash` for artifacts, checkpoints, train data and val data.
- `prepare` requires the final unique OSS normalization-stats path.  The worker
  stages that regular file into its private `/tmp` root, compares the source
  and staged bytes directly, validates the literal CPFS dataset `source_root`,
  and exports `FASTWAM_N234_NOHASH_STATS` only as the staged local file.
- Three independent eight-process Accelerate/DeepSpeed worlds are required.
  The save world uses `max_steps=2`,
  `recovery_gate_stop_after_checkpoint_step=1`, and
  `checkpoint_state_kind=full`.  It must save step-1 weights plus complete
  trainer state and return at the recovery pause without running step 2.  A
  separate eight-process world resumes exactly `step_000001`, performs the
  real step-2 update, runs with `save_training_state=true`, and saves full
  step-2 weights plus complete step-2 trainer state.  A third separately
  launched eight-process world resumes exactly `step_000002`, emits a second
  trainer-native recovery receipt, and exits with no training update and no
  checkpoint publication because the restored global step already equals
  `max_steps`.
- Each complete DeepSpeed ZeRO-2 state must contain a non-empty `latest` file
  whose exact content is `pytorch_model`, one non-empty model-state file,
  eight non-empty optimizer-state shards, eight non-empty rank RNG-state
  files, non-empty `scheduler.bin`, `zero_to_fp32.py`, and
  `trainer_state.json`.  The trainer state must carry the fixed dynamic data
  schedule (`seed=42`, `epoch=0`, `global_batches=1352`,
  `optimizer_steps=169`) and a batch cursor equal to the saved global step.
- Before any distributed world starts, one CPU rank-0 process runs the
  real-data no-hash preflight.  The compact primary cache is staged into the
  worker's private `/tmp` and compared directly by path, type, size, and bytes;
  the 5.1-TB canonical cache remains on OSS and is read only on a compact-cache
  miss.  The preflight must prove an actual train fallback frame read, projected
  finite `[2,13,28,40]` Gaussian data, exact N=2 selection, and zero guarded
  digest attempts.
- Mutable training output is created under node-local `/tmp`.  Only after all
  three process worlds and both fresh full-state loads pass is it copied to the unique
  `/oss-chengjuntao/artifacts/fastwam-gate2-nohash-results/<submission-tag>`.
  `gate2_trainer_evidence.json`, the local-stage `inventory.json`, and the final
  `COMPLETE.json` explicitly bind `resumed_from_step=1` and
  `final_global_step=2`.  The evidence validator requires trainer-native
  receipts written only after `accelerator.load_state` returns: one binds the
  load world to the step-1 state, and the other binds the final verification
  world to the step-2 state.  Rendered logs and wrapper exit status are
  auxiliary evidence only.  The validator also requires the save pause, exact
  full-state component sets, fixed data schedule, step-2 update and terminal
  records, an empty checkpoint container in the final verification world,
  real-data preflight, and compact-cache staging receipt.  After the local tree is copied, a
  persistent `publication_receipt.json` records both source and destination
  metadata plus direct byte comparisons for every published regular file.
  `COMPLETE.json` references that receipt and is the last OSS object: it is
  created directly at its final key with exclusive create, file `fsync`, and
  exact readback, without relying on object-store rename or directory `fsync`.
  No output is written to CPFS.

## Single-submit protocol

`submit_from_ssh970_r3.sh` is the only R3 entrypoint.  It must be run inside the
SSH session reached through `ssh root@123.57.187.96 -p 970`, and holds local
`flock` file descriptor 9 for the whole controller process.  This local lock
serializes one SSH970 instance but is not the cross-node safety boundary.  The
mutable POSIX state and lock live on the SSH970 node-local ext4 filesystem:

```text
/tmp/fastwam-dlc-submit-state/workspace-270969/
  FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-R3-S42-20260809/
    <attempt-uuid>/
      request.json
      state.json
```

The state/request writes are replace-atomic and file-plus-directory `fsync`ed.
The request file is bound by regular-file metadata and stable readback.  Since
node-local state does not survive an instance replacement, the authoritative,
append-only ledger lives on OSS:

```text
/oss-chengjuntao/artifacts/fastwam-dlc-submit-ledger/workspace-270969/
  FASTWAM-MR-FT-ACT-N2-PLACEFOOD-PAID-GATE2-NOHASH-R3-S42-20260809/
    prepared-binding.json
    submission-latch.json
    create-response.json       # only if a response was received
    acknowledgement.json       # only after exact GetJob identity passes
```

Any existing experiment latch is an unconditional fail-closed result, including
when it names the same attempt.  Only the process that has just created that
experiment-scoped record may call `CreateJob`.  Independent N=2, N=3 and N=4
experiment identities therefore do not share a workspace-wide admission claim
and may run as three concurrent 1x8 jobs.

Every OSS ledger object is created directly at its final key with exclusive
create, file `fsync`, exact readback, stable-read metadata, and a second
exclusive-create rejection check.  No ledger object is renamed, overwritten,
or deleted, and OSS directory `fsync` is not used.  `prepared-binding.json`
contains the complete frozen request and is written before local PREPARED state.
If node-local state is lost, it is reconstructed from these immutable records.

```text
PREPARED -> SUBMITTING -> SENT -> ACK
            |           |
            +---------->+-----> AMBIGUOUS
latched PREPARED/SUBMITTING/SENT/AMBIGUOUS
  --read-only reconcile--> RECONCILED or AMBIGUOUS
```

Before `SUBMITTING`, the launcher obtains two complete, paginated, workspace-
wide `ListJobs` snapshots without filtering by quota resource. Active jobs from
all workspace resources are confirmed with `GetJob`; unknown statuses are
treated as active. Both snapshots must have the same active allocation and
contain no exact Gate2 identity.  The observed active GPU count and the
projected count after adding this 8-GPU job are retained as audit telemetry
only; neither is compared with an artificial total-card ceiling.

The Python source contains exactly one mutating SDK call:
`create_job_with_options`.  Its runtime options set `autoretry=False` and
`max_attempts=1`; the DLC worker also uses `RestartPolicy=Never`.  The launcher
creates the OSS experiment-level permanent latch and persists local
`SUBMITTING` immediately before this call.  The latch is never removed.  As
soon as a CreateJob response is observed, its job/request IDs are written to
the OSS immutable response record before any response validation or `GetJob`.
Network timeout/reset, response ambiguity, ACK readback failure, or caught
interruption transitions to `AMBIGUOUS` while retaining any observed IDs.  A
process crash or SSH970 instance replacement can leave only the durable latch;
`execute` then always refuses another submission and `reconcile` reconstructs
local state from OSS.  This is an at-most-once CreateJob protocol with durable
reconciliation; there is never an automatic resubmit.  `reconcile` uses only
`ListJobs` and `GetJob`, first trying a durable response job ID and otherwise
scanning complete job identities.  Exact identity includes display name, description,
experiment/submission tags, source, data/stats/text/primary/fallback paths,
workspace/resource, output, settings, and exactly one 8-GPU worker.  ACK
requires an exact `GetJob` readback and is then persisted as an immutable OSS
acknowledgement.

## R3 sequence (do not execute until the source snapshot is frozen and reviewed)

The source snapshot is an explicitly chosen, unique direct child of
`/oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/`.  Its exact path
is supplied once to `prepare --source-root` and then frozen in the durable
prepared binding; R3 does not carry a stale snapshot name in executable code.
OSS is mounted writable for final result publication, so filesystem mode bits
are not treated as a source-integrity boundary.  Instead, `prepare` records
every source entry's path/type/size/mode/timestamp and direct file bytes, and
`execute` rejects any difference before acquiring submission authority.  The
exact runtime script bytes are also carried inside the immutable DLC request;
the fixed request bootstrap creates that script exclusively under `/tmp` and
executes it without first executing code from OSS.  The worker rereads the
prepared binding immediately before and after copying the bound snapshot, reconstructs the
same direct-content binding, and compares every staged regular file byte for
byte with the source.

```bash
./submit_from_ssh970_r3.sh prepare \
  --source-root /oss-chengjuntao/artifacts/fastwam-nohash-source-snapshots/<unique-snapshot-name> \
  --stats-source /oss-chengjuntao/artifacts/fastwam-nohash-inputs-20260809/fastwam_multi_robot_n234_train_s42_stats_cpfs_nohash_v1.json \
  --gaussian-cache /oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/noposplat-c944b498-4a35bc8c/builds/fastwam-8a035024af96-s42-20260801T230944Z/compact-s42-13x28x40-fp16-meanalpha-v2 \
  --gaussian-fallback-cache /oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/noposplat-c944b498-4a35bc8c/builds/fastwam-8a035024af96-s42-20260801T230944Z/canonical-all-13x240x320-fp16
```

`prepare` makes no cloud call and prints an attempt UUID.  After manually
reviewing its `request.json`, the only submission command is:

```bash
./submit_from_ssh970_r3.sh execute --attempt <attempt-uuid>
```

If the permanent latch exists, or the recorded phase is anything other than an
unlatched `PREPARED`, do not run `execute` again.  Use the read-only command:

```bash
./submit_from_ssh970_r3.sh reconcile --attempt <attempt-uuid>
```

No command in this draft stops, deletes, restarts, or otherwise changes E38.
