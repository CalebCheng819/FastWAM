# MF-WAM validation protocol

Status: preregistered implementation protocol; formal MF-WAM training is not
authorized by this document alone.

## Decision rule

Formal training is permitted only when all four independent gates are `PASS`:

```text
G0 && G1 && G2 && G3 == PASS
```

`UNCERTAIN` is not a pass. A point estimate that clears a threshold while its
required confidence bound does not clear the threshold is `UNCERTAIN`. Missing
fields, non-finite values, incomplete episode scope, post-hoc threshold tuning,
unmatched evaluation identities, or values outside their metric's physical
domain are also `UNCERTAIN` unless they prove a predeclared failure condition.

Detection, routing, recovery, training, and final evaluation are separate
claims. They use separate Experiment IDs, Notion pages, manifests, immutable
artifact roots, and terminal receipts.

Raw in-memory evidence dictionaries never authorize formal training. G1--G3
must each reference a terminal audited-receipt JSON whose SHA-256 is checked and
whose envelope is reverse-bound to the canonical policy, CI contract, and
evidence payload. Envelope verification alone remains `UNCERTAIN`: formal
authorization additionally requires specialized auditors to read the referenced
source manifest, identity inventory, metric rows, and bootstrap draws and
independently recompute scope, pairing, and confidence intervals. That
audited-bundle authorization entry point is intentionally not implemented yet,
so the current code always returns `formal_training_allowed=false`. Missing
receipts are `UNCERTAIN`; malformed or contract-mismatched receipts are `FAIL`.

## Evidence boundary

Historical evidence is retained as baseline and negative evidence:

- Blind monitor-v3 evidence supports failure detection.
- Typed-event evidence supports offline failure-mode identifiability.
- Recurrent-context reset reduced success; action resampling did not establish
  benefit.
- An untrained online risk selector catastrophically underperformed the
  original policy.

These results do not establish executable recovery and are not silently
relabelled as a new confirmatory MF-WAM experiment.

## G0: exact FastWAM reproduction

Freeze and content-bind:

- full Git commit and clean source snapshot;
- checkpoint and normalization-stat hashes;
- data revision, manifest, and checksums;
- container image digest, Python, CUDA, PyTorch, MuJoCo, and LIBERO versions;
- task, environment, policy, and disturbance seeds;
- strict success predicate, maximum horizon, action chunk, and replan settings;
- per-episode traces and terminal summary artifacts.

Each G0 trace starts at the preregistered first replan (`env_step=30`), follows
the fixed 10-step cadence, contains at least seven replan records, and carries
the exact task, environment, and policy seeds declared for that episode. The
data manifest is not accepted from declarations alone: every listed file is
read back and checked against its recorded size and SHA-256.

Required evaluation scope is the 40 LIBERO tasks with 50 trials per task. G0 is
`PASS` only when:

- the paired 95% confidence interval for overall success-rate difference lies
  entirely inside `[-0.02, +0.02]`;
- every suite is no more than 0.03 below its bound historical baseline;
- all 2,000 expected episode identities are unique and present;
- no required trace or terminal result is missing;
- all required numeric values are finite;
- `summary.csv` and `task_success_rates.csv` agree with the episode inventory.

A historical artifact may establish the scientific code/checkpoint baseline if
its provenance passes these checks. A different deployment host still requires
its own environment, checkpoint, data, CUDA, memory, and one-episode execution
contract before it can submit later jobs.

## G1: causal failure detector

Freeze FastWAM and compare on identical traces:

- historical FAIL-Detect signal;
- current observation latent;
- observation plus predicted-action latent;
- observed-predicted latent residual;
- observation-only and action-shuffled negative controls.

Features must be causal. The split unit is the base episode, not a frame. Every
nominal, disturbance, and severity derivative of one base episode remains in
the same split. Use 60% train, 15% validation, 10% success-only conformal
calibration, and 15% locked test. Fix `alpha=0.05` before reading test outcomes.

The locked statistical contract is `MF-WAM-G1-CI-v1`, with confidence level
0.95, exactly 10,000 bootstrap resamples, and bootstrap seed 42. The audited
receipt binds the split manifest, complete base-episode identity inventory,
metric rows, and actual bootstrap draws.

G1 requires all of:

- nominal episode FPR one-sided 95% upper bound at most 0.075;
- TPR at FPR at most 0.05 with 95% lower bound at least 0.80;
- AUROC 95% lower bound at least 0.85;
- median onset-relative delay at most one action chunk;
- paired action-conditioned improvement over observation-only with 95% lower
  bound strictly above zero.

Missed failures remain visible in the delay analysis; delay is never reported
only over successful alarms.

## G2: interaction-mode routing probe

Freeze the FastWAM backbone. Start with `K=4`, top-2 lightweight action-side
adapters. Compare against an active-parameter and active-FLOP matched dense
adapter, shuffled routing, task-ID routing, and an oracle semantic-mode upper
bound.

Semantic modes are derived from simulator predicates and interactions, not task
names or absolute time. Initial labels include free-space, pre-contact, stable
grasp, transport, slip/drop, constrained contact, articulation, placement, and
recovery.

G2 requires all of:

- contact/recovery prediction-error improvement versus the matched dense
  adapter has a 95% lower bound of at least 0.05;
- every expert receives at least 5% of dispatches;
- maximum expert load is at most 50%;
- effective expert count is at least `0.6 * K`;
- cross-seed expert matching AMI is at least 0.5.

Router alignment with task ID and absolute timestep is reported as a confound,
not as positive mode evidence.

The locked statistical contract is `MF-WAM-G2-CI-v1`, with confidence level
0.95, exactly 10,000 task-and-training-seed-stratified paired bootstrap
resamples, and bootstrap seed 42. Its receipt binds all three training seeds,
the complete identity inventory, metric rows, and bootstrap draws.

## G3: frozen-backbone closed-loop recovery

Use the same recovery controller, horizon, replan policy, recovery-attempt cap,
and paired disturbance draws across arms. The controller states are:

```text
Normal -> Suspected -> Confirmed -> Recovery -> Recovered | Abort
```

Confirmation invalidates the remaining open-loop action chunk exactly once and
forces a replan. It does not reset learned context. Consecutive evidence or
hysteresis prevents one noisy score from causing repeated replans.

Compare FastWAM, fixed-time retry, FAIL-Detect plus the same controller,
action-conditioned residual detector plus the same controller, oracle onset
plus the same controller, and full dual-router sidecar.

G3 requires all of:

- disturbed strict-success paired 95% lower bound improvement at least +0.05;
- conditional-recovery paired 95% lower bound improvement at least +0.10;
- nominal strict-success non-inferiority lower bound strictly above -0.02;
- harmful false-trigger one-sided 95% upper bound at most 0.02.

The gated conditional-recovery comparison uses a common, preregistered set of
paired saved intervention states. Every arm starts from the same eligible state
and disturbance draw, so the denominator is independent of the policy's prior
trajectory. Per-arm trigger reach and exposure counts are still reported. An
online policy that never reaches a trigger remains a failed disturbed terminal
episode, but policy-dependent reachable subsets are diagnostic only and cannot
satisfy the conditional-recovery gate.

The locked statistical contract is `MF-WAM-G3-CI-v1`, with confidence level
0.95, exactly 10,000 task-and-training-seed-stratified paired bootstrap
resamples, and bootstrap seed 42. For each model and each of three independent
training seeds it requires 2,000 nominal episodes (40 tasks x 50) and 500
disturbed episodes (10 tasks x 50), hence 6,000 nominal and 1,500 disturbed
episodes per model. The receipt also binds the paired saved-state inventory.

## Disturbance and recovery data

Disturbances are semantic-event triggered:

- miss grasp before gripper closure;
- slip/drop after stable grasp and lift;
- object reset after grasp and before placement;
- transport/contact blocking or short joint stall;
- articulation regression after partial opening;
- placement-target movement or post-placement displacement;
- benign lighting, distractor, or small camera-jitter negatives.

Every row records trigger, first observable time, recipe, severity, draw,
exposure, recoverability, terminal strict success, subgoal re-achievement, and
recovery duration.

Failure-terminal traces alone are insufficient action targets. Adapter or
recovery-policy training requires trajectories that start from perturbed states
and ultimately succeed. Recovery/contact frames may be oversampled only after
the ratio is frozen from the router probe and recorded in the data manifest.

The historical E9 typed-intervention bundle is only a candidate source. Its
audit reports 821 active event rows from episodes that eventually succeeded,
with a conservative lower bound of 300 rows after a second event confirmation.
However, `target_action` is the baseline policy proposal, not a verified
executed corrective action or counterfactual expert target, and there is no
recovery-complete marker. These rows require raw-trace chronology, executed vs
proposed action equality, and post-disturbance success filtering before any may
enter a recovery-training manifest. The NPZ and raw sidecars currently reside
on H GPFS and are not present in this local checkout.

## Minimal model boundary

FastWAM MoT is a shared-attention video/action dual trunk, not a sparse MoE.
The first implementation therefore preserves the existing video and action
experts and adds only lightweight action-side adapters and router heads.

- Prior features use final first-frame video hidden states. The causal mask must
  prove that these tokens cannot see future video or action tokens.
- Posterior diagnosis is computed only after executing actions and observing the
  next state, preferably at a replan boundary.
- Adapter outputs are zero initialized, and a fixed-input fixed-seed test must
  prove base and MF action outputs are initially equivalent.
- Base-to-MF checkpoint loading has an explicit missing-key allowlist. MF-to-MF
  loading is strict. Formal evaluation cannot continue with randomly missing
  router, adapter, posterior, or controller state.

## Pilot and confirmatory budgets

Pilot work is go/no-go only. A suggested fixed pilot uses five hard tasks,
20-25 matched base seeds, nominal plus four disturbance families, and one
training seed.

Confirmatory evaluation uses:

- nominal: 40 tasks x 50 matched seeds = 2,000 episodes per model and training
  seed;
- disturbed primary: 10 LIBERO-Long tasks x 50 matched seeds = 500 episodes per
  model and training seed;
- three independent training seeds;
- task and training-seed stratified paired bootstrap with at least 10,000
  resamples, plus exact McNemar where applicable;
- Holm correction for secondary disturbance-family claims.

Checkpoint selection uses validation outcomes only. Locked test is run once
after code, thresholds, config, and checkpoint identities are frozen.

## Resource and launch contract

- One DLC program requests one worker with at most eight GPUs.
- Total MF-WAM-related concurrent allocation is at most 16 GPUs.
- Existing unrelated eight-GPU jobs are not stopped or overwritten.
- Live DLC state, quota, GPU SKU, image digest, mounts, free space, and duplicate
  submission state are read immediately before submission.
- Formal Arm A is full dual-router MF-WAM. Arm B is an expert-count,
  parameter-count, active-FLOP, data, optimizer, step, and seed matched
  state-only single-router control.
- Run paired seeds in order: A0+B0, A1+B1, A2+B2. A 5,000-step pilot precedes
  any 20,000-step confirmatory training.

## Terminal evidence

No gate or final run is terminally successful without all required artifacts:

- validated planned, running, and terminal manifests linked by the same Notion
  page ID;
- immutable code, config, data, environment, checkpoint, and threshold hashes;
- matched `summary.csv` and `task_success_rates.csv`;
- failure/recovery metrics CSV;
- per-episode controller traces and disturbance receipts;
- router load and mode-confound reports;
- final artifact checksum inventory and completion marker.

Queued, Running, a checkpoint, decreasing loss, detector recall, or an interim
episode count is not a final result.
