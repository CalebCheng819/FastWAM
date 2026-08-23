# GAU1 step-10000 PlaceFood same-panel 8-GPU DLC evaluation

- Experiment: `FASTWAM-MR-N234-VG1H1GAU1-STEP10000-PLACEFOOD-SAME8-S42-R1-20260823`
- Run: `fastwam-gau1-step10k-placefood-same8-r1-20260823`
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
