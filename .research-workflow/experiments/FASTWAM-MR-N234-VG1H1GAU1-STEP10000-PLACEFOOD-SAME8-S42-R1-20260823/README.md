# GAU1 step-10000 PlaceFood same-panel 8-GPU DLC evaluation

- Experiment: `FASTWAM-MR-N234-VG1H1GAU1-STEP10000-PLACEFOOD-SAME8-S42-R1-20260823`
- Current run: `fastwam-gau1-step10k-placefood-same8-r3-20260823` (`attempt-003`)
- Checkpoint: `step_010000.pt` (12,047,213,657 bytes)
- Topology: one DLC Worker pod with exactly 8 GPUs; one fixed-panel episode per GPU
- Task: `PlaceFood-rf`
- Panel: the frozen eight-environment-seed panel used by the earlier same8 baseline
- Policy seeds: 10000 through 10007
- Closed-loop contract: `exec_horizon=5`, `action_horizon=32`, 20 inference steps, 300 environment steps maximum
- Conditioning: GAU1 enabled; Policy-Lightning and NoPoSplat provenance is pinned
- Integrity mode: `metadata_no_hash`; regular-file identity and exact byte sizes are checked without creating a new digest
- Scheduler priority: 7 in the initial CreateJob request

Scientific completion requires all eight evaluator processes to complete and the strict aggregator to publish `COMPLETE.json`. A zero-success scientific result is still a completed evaluation; an incomplete shard, infrastructure error, or contract mismatch fails the DLC job.

## Attempt history

- `attempt-001` / `fw-gau1-s10k-placefood-same8-r1` / Job `dlcanqj2ibd02y6v`: infrastructure failure before checkpoint load. The inherited `PYTHONPATH` selected an older CPFS `fastwam.runtime` module without `create_multi_robot_fastwam`; all eight evaluator processes exited and produced no scientific result.
- `attempt-002` / `fw-gau1-s10k-placefood-same8-r2` / Job `dlcu3lziinnxixqu`: source isolation passed, but evaluation failed before checkpoint loading because the policy called the current `FastWAMMultiRobot.load_checkpoint()` with the removed `record_checkpoint_sha256` keyword. All eight evaluator processes exited with the same interface error and produced no scientific result.
- `attempt-003`: preserves the identical checkpoint, SAME8 panel, 8-GPU topology, policy seeds, and closed-loop settings. It keeps the verified isolated-source bootstrap and calls the current checkpoint loader with its supported positional path API.
