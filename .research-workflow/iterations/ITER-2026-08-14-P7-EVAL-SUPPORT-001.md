### ITER-2026-08-14-P7-EVAL-SUPPORT-001 - P7 closed-loop evaluator support

- Time: 2026-08-14T08:05:00+08:00
- Type: Code_Config_Validation
- Status: Complete
- Scope: Add fail-closed evaluator support for the P7 task-conditioned Gaussian relation action architecture while preserving the P6 spatial-Gaussian path.
- Runtime boundary: Evaluator scripts come from this checkout; the FastWAM model package must come first from the explicit model-project checkout so P7-only modules cannot silently fall back to P6 code.
- Validation: Static compile and diff checks passed locally. In the DSW FastWAM runtime, 49 focused evaluator tests passed in 4.33 seconds, covering config resolution, CLI acceptance, runtime import ordering, P6 compatibility, and structural relation-attention checks.
- Evaluation gate: Formal P7 val8 closed-loop evaluation remains blocked until P6 val8 is terminal and P7 training plus paired offline selection complete.
- Notion: P7 preregistration page `3bb21e77-89cc-81f3-87d2-c789cb207894` already records the architecture and evaluation acceptance criteria.
- Provenance: Git revisions, paths, timestamps, experiment IDs, run IDs, and ordinary file metadata; no newly computed artifact hashes.
- Result: The evaluator can select `task_conditioned_relation_v3`, imports the model package from the explicit model-project checkout before evaluator sources, and refuses a missing or mismatched relation attention/gate/norm contract.
