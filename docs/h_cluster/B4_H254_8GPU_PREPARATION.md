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
- H priority: 9; charged quota group: `eailabagent_gpu`; private-machine mode:
  `group` (no separate `--group`); directed node tag:
  `node/gpu-l-lg-cmc-h-h200-0254.host.h.pjlab.org.cn`

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
3. Prepare the exact local CUDA 12.8 Torch wheelhouse with
   `scripts/prepare_b4_h254_torch_wheelhouse.sh`, then build the isolated
   Python 3.10 environment with `scripts/bootstrap_b4_h254_env.sh`. The
   wheelhouse reuses the already downloaded exact CUDA dependency wheels and
   fetches only the exact missing Triton, TorchVision, and TorchCodec wheels;
   it is published atomically only after wheel metadata validation. The
   environment installs the Torch stack only from that local wheelhouse and
   disables pip's HTTP cache on GPFS while keeping all build temporaries on
   GPFS, avoiding the filesystem's unsupported cache-file `mmap` path. The
   remaining pinned dependencies are fetched from the single official PyPI
   index on the login-side preparation host with bounded retries; the GPU job
   never installs packages or requires network access. The environment
   publishes only after `pip check` and the complete pinned metadata contract
   pass, at
   `/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/envs/fastwam-b4-h254-py310-20260820`.
   TorchCodec's CUDA-linked import and the exact eight-device check remain
   runtime gates on H200-0254; a login-node metadata check is not a GPU gate.
4. The committed Git bundle and launcher are published on personal GPFS.
5. `scripts/render_b4_h254_rjob.py --execute-predict` returns a successful
   scheduler prediction for H200-0254. This command always includes
   `--predict-only true` and cannot submit a formal job.
6. Only after the preceding gates pass may a separate, reviewed formal-submit
   command be prepared. Runtime acceptance still requires eight CUDA devices,
   strict input loading, and a finite optimizer-step log; Queuing or Running is
   not training success.
