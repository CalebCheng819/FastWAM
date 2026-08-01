# RoboFactory sensor controls

`robofactory_multi_robot_sensor_controls.yaml` is an audit registry, not a
Hydra task config. It defines the sensor-control comparisons allowed in formal
reporting while holding the N=2/3/4 corpus, seed, optimizer, schedule, and
official FastWAM initialization fixed.

| Control | Current status | Formal use |
| --- | --- | --- |
| global-only | Runnable | Use a `GAU0` task; the Gaussian cache is explicitly null. |
| global+agent-RGB | Planned, not runnable | Proposal only. No per-agent RGB dataset/model contract exists, so this row must not be reported as an executed baseline. |
| global+agent-Gaussian | Runnable with cache preflight | Use a `GAU1` task and set `FASTWAM_GAUSSIAN_CACHE_DIR` to an immutable, versioned compact-cache root. |

GAU1 deliberately has no default cache path. Missing environment configuration
fails during Hydra resolution; a present path is subsequently checked by the
dataset's manifest validation. This keeps task configs independent of a cache
manifest that has not yet been generated.
