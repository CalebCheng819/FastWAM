# RoboFactory multi-robot B4 on H200-0254

This preparation targets one directed H-cluster node (`H200-0254`) with eight
H200 GPUs. It does not claim that a job has been submitted or that training has
produced an optimizer update.

## Frozen training contract

- task: `robofactory_multi_robot_b4_phase_gripcontact_actft_224_1e-5`
- topology: one node, eight processes, one GPU per process
- scale: `robofactory_multi_robot_8gpu_h254_b4`
- gradient accumulation: 3; effective global batch: 24
- schedule: 2,500 steps; checkpoints/evaluation at 1,250 and 2,500
- initialization: weights-only from the existing step-5,000 joint checkpoint;
  optimizer and scheduler start fresh at step zero
- target node: `gpu-l-lg-cmc-h-h200-0254.host.h.pjlab.org.cn`
- H priority: 9; charged quota group: `eailabagent_gpu`

## Data preparation

The 24 raw H5 files already live under the HDD2 `fkp-migrate` prefix and are
listed exactly in `configs/transfer/b4_h254_raw_h5_paths.txt`. Do not duplicate
them through CDK.

Use the five credential-free tasks in
`configs/transfer/b4_h254_cdk_tasks.json` for stats, six cached text embeddings,
the compact Gaussian cache, the weights-only warm start, and the Wan2.2 VAE.
Enter source and destination credentials only in the CDK web UI. Run CDK Dry
Run first, use concurrency 2, then compare status, object counts, byte totals,
and exact paths. Do not create new digest metadata.

## Gates before formal submission

1. CDK reports all five task groups complete and the H destination inventory
   matches the manifest counts and known byte totals.
2. The exact 24 raw H5 paths are readable through the approved HDD2 rclone
   config; the config remains mode 0600 and is never printed or committed.
3. A Python 3.10 environment exactly matches the package versions enforced by
   `scripts/launch_b4_h254_8gpu.sh`. The currently discovered FastWAM_yuner
   environment is only a candidate and fails that exact-version gate.
4. The committed Git bundle and launcher are published on personal GPFS.
5. `scripts/render_b4_h254_rjob.py --execute-predict` returns a successful
   scheduler prediction for H200-0254. This command always includes
   `--predict-only true` and cannot submit a formal job.
6. Only after the preceding gates pass may a separate, reviewed formal-submit
   command be prepared. Runtime acceptance still requires eight CUDA devices,
   strict input loading, and a finite optimizer-step log; Queuing or Running is
   not training success.
