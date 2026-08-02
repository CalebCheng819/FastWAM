# MF-WAM Project Chronicle

- Project ID: `MF-WAM`
- Canonical Notion page ID: `3b021e77-89cc-81b9-8238-e3cdbf44dda2`
- Canonical Notion page: https://app.notion.com/p/MF-WAM-Project-Chronicle-3b021e7789cc81b98238e3cdbf44dda2
- Parent research-note page ID: `3a521e77-89cc-8050-baac-cd4a93e35e67`
- Experiments database ID: `110f6b6f-9069-4a26-864c-f0f82fc0a215`
- Experiments data source ID: `a693d9d0-b4ed-44e8-a4e0-a153ef7deda6`
- Workspace ID: `2dc05352-6acc-48e1-a630-f42652151140`
- Upstream repository: https://github.com/yuantianyuan01/FastWAM
- Implementation repository: https://github.com/CalebCheng819/FastWAM
- Official baseline commit: `45d8e1458921d83f8ad6cf9ce993d371208dabd0`

The Notion page is the canonical append-only project history. This file binds the
local checkout to its exact remote identities and is not a substitute for the
full Chronicle.

A bounded S1 probe is eligible only after specialized-auditor-verified G0. Paired
S2 confirmatory training is eligible only after verified G1 plus the frozen
single-seed G2/G3 feasibility pilot; the paper-level positive conclusion still
requires `G0 && G1 && G2 && G3 == PASS`. Detection, router probes, sidecar
recovery, training, and final evaluation use separate globally unique
Experiment IDs and pages. The current runtime authorizer remains hard-disabled.

## 2026-08-03 — G0 evidence pipeline deployment

- Implementation commit `a36295442fb3aafe204adb4a771fd184cdb00ae7` was
  pushed to `fork/exp/mf-wam-validation`; local and remote branch readback match.
- 942 uses the same CPFS data disk but a new control instance. The official
  baseline and instrumentation were deployed as separate commit-addressed clean
  clones under `/mnt/workspace/MF-WAM/repos`; both have zero status, ignored, and
  replace-ref bytes and passed the exact-tree source gate.
- The 942 control runtime remains diagnostic-only: Hydra 1.3.4 differs from the
  locked 1.3.2, MuJoCo is absent, and no CUDA device is visible. No DLC or GPU
  job was launched; `formal_training_allowed=false` remains authoritative.
