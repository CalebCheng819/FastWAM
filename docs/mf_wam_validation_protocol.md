# MF-WAM validation protocol

Status: staged preregistered implementation protocol. The stages below define
future evidence requirements, but the current runtime authorizer is deliberately
hard-disabled and always returns `formal_training_allowed=false`. The local G0
contract builder, schema-v2 traced worker, terminal-bundle sealer, and paired
artifact recomputation tool are evidence producers/validators only. They have
not yet produced a real terminal 2,000-episode paired bundle and cannot authorize
training.

## Decision semantics

The former rule, "G0 through G3 must all pass before any training", was
circular: G2 requires trained router/adapters and G3 requires closed-loop
evaluation of trained recovery models. It is replaced by three ordered,
non-authorizing protocol stages:

```text
S1 bounded probe / single-seed pilot
    -> S2 paired confirmatory training
    -> S3 paper-level confirmatory conclusion
```

These are protocol-level eligibility decisions, not executable launch
permissions. No stage, receipt, or point estimate can change the current
runtime result from `false`. A future runtime authorizer requires a separate
reviewed implementation and tests; it is outside this protocol revision.

`UNCERTAIN` is never a pass or a paper-level positive conclusion. A point
estimate that clears a threshold while its required confidence bound does not
clear the threshold is `UNCERTAIN`. Missing fields, non-finite values,
incomplete episode scope, post-hoc threshold tuning, unmatched evaluation
identities, or values outside their metric's physical domain are also
`UNCERTAIN` unless they prove a predeclared failure condition.

Detection, routing, recovery, training, and final evaluation are separate
claims. They use separate Experiment IDs, Notion pages, manifests, immutable
artifact roots, and terminal receipts.

Raw in-memory evidence dictionaries never satisfy a stage or a final claim. G0
must reference `specialized_g0_audit_receipt`; G1--G3 must each reference their
terminal audited-receipt JSON. Each reference binds a literal path and SHA-256,
and each envelope is reverse-bound to the canonical policy, CI contract, and
evidence payload. Envelope verification alone remains `UNCERTAIN`: stage entry
and final conclusion additionally require specialized auditors to read the
referenced source manifests, identity inventories, metric rows, traces, and
bootstrap draws and independently recompute scope, pairing, and confidence
intervals. The local G0 specialized auditor can now recompute a paired canonical
bundle and validate exact semantic external-anchor documents. The independent
trust consumer for those anchors and every executable authorization entry point
are still intentionally not implemented. Consequently, the generic evaluator
treats even a structurally valid locally produced receipt as
`STRUCTURAL_PASS_ONLY`/`UNCERTAIN`. Missing receipts are `UNCERTAIN`; malformed
or contract-mismatched receipts are `FAIL`.

## G0 evidence classes and specialized receipt boundary

The following machine-readable classifications are distinct and must never be
substituted for one another:

- `OUTCOME_PARITY_ONLY`: all raw G0 literals and confidence intervals satisfy
  the outcome-parity contract. This says only that the submitted numbers show
  parity; it does not verify their source, lineage, recomputation, or external
  anchors. The G0 scientific gate is still `UNCERTAIN`.
- `STRUCTURAL_PASS_ONLY`: the specialized receipt file, file hash, exact schema,
  reverse evidence hash, policy/CI bindings, declared artifact hashes, and
  declared anchor-lineage records are internally consistent. A self-authored or
  resealed JSON can reach this class. No referenced artifact or external anchor
  has been independently consumed, so the G0 scientific gate is still
  `UNCERTAIN`.
- `SPECIALIZED_G0_PASS`: the local specialized auditor has independently read
  and recomputed all bound raw artifacts and a separate trusted consumer has
  resolved every required external anchor under the same immutable lineage.
  Only that combined result could satisfy the scientific G0 predicate. The
  current generic evaluator cannot emit or consume this class.

Thus neither `OUTCOME_PARITY_ONLY` nor `STRUCTURAL_PASS_ONLY` is a weakened form
of `SPECIALIZED_G0_PASS`; both remain non-eligible for S1 and leave
`stage_eligibility=false` and `formal_training_allowed=false`. Supplying status
strings such as `PASS` or `SPECIALIZED_G0_PASS` inside raw evidence has no
semantic effect.

The future G0 specialized receipt contract is exact:

- evidence field: `specialized_g0_audit_receipt`, containing only `path` and
  `sha256` in the reference;
- receipt `kind=mf_wam_g0_specialized_audit_receipt`, `schema_version=1`,
  `gate_id=G0`, `ci_contract_id=MF-WAM-G0-CI-v1`, `terminal=true`, and claimed
  `scientific_status=SPECIALIZED_G0_PASS`, while
  `formal_training_allowed=false` remains mandatory;
- reverse bindings: exact `policy_id`, canonical `policy_sha256`, and the
  canonical SHA-256 of the G0 evidence excluding the receipt reference;
- locked scope: 2,000 episodes, four suites, ten tasks per suite, 50 trials per
  task, 95% confidence, 10,000 bootstrap replicates, bootstrap seed 42, and
  `outcome_parity_classification=OUTCOME_PARITY_ONLY`;
- exact artifact-digest keys: source manifest, data manifest, seed manifest,
  resolved config, checkpoint, dataset statistics, runtime environment,
  identity inventory, metric rows, trace tree, and terminal summary bundle;
- exact ordered external-anchor types: `notion_experiment_page`,
  `immutable_artifact_root`, `source_commit`, and `container_image_digest`.
  Each lineage item contains only `anchor_type`, non-empty `anchor_id`, and
  `artifact_sha256`.

The local auditor accepts each external anchor only as a strict JSON document
with exactly `schema_version=1`, `kind=mf_wam_g0_external_anchor`,
`anchor_type`, `anchor_id`, and `bindings`. The ID and the complete bindings are
derived from both validated run contracts. A caller-supplied hash or arbitrary
file is insufficient. These documents prove semantic consistency locally; they
still require preservation and resolution through an independently trusted
channel before the scientific classification can be consumed.

The current code checks only this structural envelope. It deliberately reports
`STRUCTURAL_PASS_ONLY`, `specialized_artifact_recomputation_verified=false`, and
`external_anchor_lineage_verified=false`, even when a locally resealed receipt
matches every field.

## S1: bounded probe and single-seed pilot

S1 allows only bounded evidence generation. It does not permit three-seed
confirmatory training, locked confirmatory evaluation, or a paper-level claim.

### S1 entry evidence

- `specialized_g0_audit_receipt` is produced by the local specialized G0
  auditor and independently trusted anchor resolution upgrades the combined
  result to `SPECIALIZED_G0_PASS`; neither `OUTCOME_PARITY_ONLY` nor
  `STRUCTURAL_PASS_ONLY` is sufficient.
- The G0 candidate contract is `LOCKED` and binds data inventory, seed tree,
  resolved config, checkpoint/statistics, image digest, runtime versions, and
  terminal traces.
- Official FastWAM model identity and instrumentation identity are separately
  immutable. Instrumentation must not masquerade as the official source, and a
  fixed-input fixed-seed equivalence receipt must show that tracing does not
  change base actions.
- Every GPU activity has its own preregistered Experiment ID/page and immutable
  prelaunch manifest. No active shared checkout is mutated.
- Before a model pilot, zero-initialized adapter action equivalence, prior causal
  masking, missing-key allowlisting, strict MF checkpoint loading, and finite
  forward/backward smoke receipts all pass.

### S1 allowed work and exit evidence

S1 may collect fresh nominal/disturbed traces, fit the G1 detector and negative
controls, test recovery oracle headroom on train/validation saved states, and run
one training seed for at most 5,000 pilot steps. The five hard tasks, 20--25
matched base seeds, disturbance families, pilot metrics, and go/no-go thresholds
must be frozen before outcomes are read.

S2 eligibility requires all of:

- a specialized-auditor-verified G1 `PASS` on its locked test;
- a terminal single-seed G2 pilot `GO` receipt showing no base-action drift,
  causal leakage, non-finite value, or expert collapse, plus improvement in the
  preregistered direction against its active-compute-matched control;
- a terminal recovery-headroom receipt and a single-seed G3 pilot `GO` receipt
  on paired saved states, with no preregistered nominal-harm or harmful-trigger
  stop condition;
- complete pilot identity, metric-row, controller-trace, and checksum
  inventories.

The G2/G3 pilot `GO` decisions are feasibility evidence only. They never set G2
or G3 to formal `PASS`, and pilot data may not be relabelled as confirmatory.

### S1 GPU budget and stop conditions

- Each DLC program requests one worker and at most eight GPUs.
- Total concurrent MF-WAM allocation is at most 16 GPUs and at most two
  eight-GPU programs; CPU audit work consumes no GPU allowance.
- Live quota, SKU, image, mounts, free space, unrelated jobs, and duplicate
  submissions are read immediately before every launch. Unrelated jobs are not
  stopped to make room.
- Stop before submission if capacity/quota is unverified, any identity or
  manifest is missing, or the proposed allocation would exceed either limit.
- Stop S1 if the future G0 scientific result ceases to be
  `SPECIALIZED_G0_PASS`; tracing changes base actions; causal-mask or
  checkpoint-load checks fail; G1 is `FAIL`/`UNCERTAIN`; oracle recovery has no
  preregistered headroom; the router collapses; or the recovery pilot is null or
  harmful under its locked pilot rule.

Infrastructure failure is recorded as invalid/`UNCERTAIN` and may be rerun only
with the same scientific identities. A scientific failure requires a new
protocol/Experiment ID; it is not repaired by tuning on the locked test.

## S2: paired confirmatory training

S2 begins only after the complete S1 exit bundle is verified. It authorizes a
future protocol to train the already-frozen candidate design; it does not grant
runtime permission in the current code and does not establish G2/G3 success.

### S2 entry and terminal evidence

- The implementation uses a clean, pushed, immutable commit and separately
  content-bound data, config, image, command, optimizer, step, global-batch, and
  seed manifests.
- Arm A is the full dual-router MF-WAM. Arm B is the preregistered state-only
  control. Their expert count, parameter count, active FLOPs, data, optimizer,
  steps, checkpoint schedule, and seed are matched; any additional dense
  adapter used by G2 remains a separately identified control.
- Three independent training seeds are fixed before launch. Run paired waves in
  order: `A0+B0`, `A1+B1`, `A2+B2`, with at most 20,000 confirmatory steps per
  arm after the 5,000-step pilot.
- Checkpoint selection uses validation data only. Locked-test identities remain
  unread until all six terminal training receipts and the evaluation protocol
  are frozen.
- S2 terminal evidence contains six matched terminal manifests, exact last-good
  checkpoint hashes, step/global-batch equivalence, loss/finite-value summaries,
  environment receipts, and a locked S3 evaluation preregistration.

### S2 GPU budget and stop conditions

- Each arm uses at most eight GPUs; one paired wave uses at most `8+8=16` GPUs.
- Only one A/B seed pair runs at a time. No third MF-WAM program may overlap an
  eight-plus-eight wave.
- Stop before launch if the complete S1 exit bundle, clean pushed commit, exact
  live capacity, or paired manifests are absent.
- Stop both arms at the last common durable step if either arm has non-finite
  loss/gradients, persistent OOM/NCCL failure, data/config/hash drift, an
  unreadable checkpoint, or a scientifically unmatched change. Resume only from
  identical immutable checkpoints and manifests; otherwise invalidate and rerun
  the pair.
- Do not select checkpoints, thresholds, or recovery settings using locked-test
  outcomes. Any such exposure invalidates S3.

## S3: paper-level confirmatory conclusion

S3 runs the locked evaluation and decides the final scientific claim. It is not
a retrospective license for training and does not change the hard-disabled
runtime authorizer.

### S3 required evidence

- S2 terminal evidence is complete and the locked evaluation protocol was
  frozen before test access.
- G0 and G1 remain specialized-auditor-verified `PASS` under the same immutable
  model/data/runtime lineage used by the candidate.
- G2 is recomputed over all three training seeds with the matched dense,
  shuffled-routing, task-ID-routing, and oracle semantic-mode controls.
- G3 is recomputed over the complete nominal, disturbed, and paired saved-state
  inventories, with training seed and evaluation seed represented separately.
- G0, G1, G2, and G3 are all `PASS`; each audited bundle, metric row, bootstrap
  draw, summary, task success table, controller trace, and checksum inventory is
  complete and independently recomputed.

Only this intersection supports the paper-level confirmatory conclusion:

```text
paper_level_confirmatory_conclusion =
    G0 && G1 && G2 && G3 == PASS
    && all_specialized_audited_bundles_verified
```

### S3 GPU budget and stop conditions

- Locked evaluation runs in waves of at most two programs, each at most eight
  GPUs, for a maximum concurrent MF-WAM allocation of 16 GPUs.
- Baseline/control results may be reused only when their exact model, evaluation
  seed, saved state, horizon, controller, and disturbance identities match; no
  result is duplicated merely to invent a training-seed dimension for a fixed
  control.
- Stop and withhold the positive conclusion if any gate is `FAIL` or
  `UNCERTAIN`, any required identity is unmatched, any terminal artifact is
  missing, or summaries disagree with the episode inventory.
- Technical missingness invalidates affected identities and permits only exact
  completion/rerun. A threshold failure is a terminal negative result and is not
  retuned post hoc.
- Confirmatory evaluation has no ad-hoc early stopping. A sequential/futility
  rule is allowed only if its alpha-spending contract was preregistered before
  locked-test access.

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
- exact six-file model-cache inventory, canonical
  `DIFFSYNTH_MODEL_BASE_PATH`, and `DIFFSYNTH_SKIP_DOWNLOAD=true` so a run
  cannot silently fetch or substitute weights;
- data revision, manifest, and checksums;
- container image digest, Python, CUDA, PyTorch, MuJoCo, and LIBERO versions;
- task, environment, policy, and disturbance seeds;
- strict success predicate, maximum horizon, action chunk, and replan settings;
- per-episode traces and terminal summary artifacts.

Each G0 trace starts at the preregistered first replan (`env_step=30`), follows
the fixed 10-step cadence, and contains at least seven replan records. Seed
semantics are process-scoped: one fresh process owns one task, global and
environment seeding happen once before trial 0, and the same policy seed is
used at every replan while trials run in the fixed order 0--49. There is no
invented per-episode task seed; each episode instead binds its trial index,
initial-state index/hash, complete task-process seed object, and policy seed on
every replan record. The data manifest is not accepted from declarations alone:
every listed file is read back and checked against its recorded size and
SHA-256.

Required evaluation scope is the 40 LIBERO tasks with 50 trials per task. The
G0 outcome-parity subdecision is `PASS` only when:

- the paired 95% confidence interval for overall success-rate difference lies
  entirely inside `[-0.02, +0.02]`;
- every suite is no more than 0.03 below its bound historical baseline;
- all 2,000 expected episode identities are unique and present;
- no required trace or terminal result is missing;
- all required numeric values are finite;
- `summary.csv` and `task_success_rates.csv` agree with the episode inventory.

Meeting these conditions yields at most `OUTCOME_PARITY_ONLY`; it never yields
the G0 scientific `PASS`. That higher claim still requires successful local
specialized recomputation plus the independent external-anchor trust consumer
defined above.

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
resamples, and bootstrap seed 42. For each learned model and each of three
independent training seeds it requires 2,000 nominal episodes (40 tasks x 50)
and 500 disturbed episodes (10 tasks x 50), hence 6,000 nominal and 1,500
disturbed episodes per learned model. Fixed baseline/control arms are evaluated
in the same matched evaluation-seed blocks; they do not acquire a fictitious
training-seed identity. The receipt also binds the paired saved-state inventory.

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
- Formal policy and instrumentation sources use separate, fresh, commit-addressed
  immutable ordinary clones (no linked worktree, external Git directory, or
  shared Git common directory). Active checkouts that contain ignored caches, checkpoints,
  results, bytecode, or native extensions are diagnostic-only and cannot satisfy
  the source gate. Any Git-ignored worktree entry or replacement ref fails the
  formal source check; Git system/global config, object replacement, fsmonitor,
  and the untracked cache are disabled during identity readback.
- Every formal Python entry point is launched with `python -B` and also disables
  bytecode before importing repository modules. Static verification compiles
  source in memory or in a disposable archive copy; `compileall` is never run
  inside either formal clone.
- Live DLC state, quota, GPU SKU, image digest, mounts, free space, and duplicate
  submission state are read immediately before submission.
- The worker environment fixes the approved model-cache root and disables all
  DiffSynth downloads; the runtime-start receipt and terminal audit both reread
  the same six cached files.
- S2 confirmatory Arm A is full dual-router MF-WAM. Arm B is an expert-count,
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
