# Results and verification

Evidence cutoff: **September 29, 2026, 23:02 UTC**. This is a historical project snapshot. Claims below distinguish official scores, retrospective development metrics, accepted-but-unscored submissions, and incomplete engineering work.

## Official benchmark state

| Evidence | Recorded result |
|---|---:|
| Verified official Kaggle public score | **0.947** |
| Verified reference submission | **56376695** |
| Locally qualified candidate submission | **56659462 → 0.946** |
| Image/division probe submission | **56503335 → 0.946** |
| Exact public-0.953 reproduction | **56687425 → pending at cutoff** |
| Official improvements beyond 0.947 verified at evidence cutoff | **0** |

The 0.947 submission reproduces the public Harmonic Fusion inference reference. The later 0.946 results are official adverse evidence and supersede optimism based only on retrospective development gains.

## Retrospective full-baseline integration

Thirty-seven saved candidate policies plus two controlled combinations were replayed against one frozen Harmonic development control.

| Five-movie local exact metric | Original Harmonic control | Locally qualified candidate | Change |
|---|---:|---:|---:|
| Combined local score | **0.933046** | **0.948059** | **+0.015014** |
| Edge TP | 2,391 | **2,392** | +1 |
| Edge FP | 111 | **110** | -1 |
| Edge FN | 112 | **111** | -1 |
| Division TP | 1 | **2** | +1 |
| Division FP | **1** | **1** | 0 |
| Division FN | 5 | **4** | -1 |

The candidate passed the local gate but then scored **0.946 officially**. The correct interpretation is therefore not that the candidate “won locally,” but that the project exposed a validation/generalization mismatch in a repeatedly inspected development cohort.

## Prediction-fidelity audit of the 0.946 regression

The scored-reference preview output and the failed candidate output were both verified by SHA-256 before comparison.

The comparison found:

- identical preview movie coverage;
- identical row counts;
- no added or removed preview detections;
- no shared-ID coordinate shifts;
- no frame conflicts;
- **eight association-edge edits total** across four movies.

No hidden labels were used. This audit cannot prove why the hidden/public score fell, but it substantially weakens an accidental detector/coordinate-drift explanation and supports treating the local association edits as poor generalization evidence.

## Public 0.953 reproduction

The final-day sprint audited a public saved notebook advertised at **0.953**. Its source, saved version, and complete output were bound by hash before submission.

Submission **56687425** was accepted by Kaggle from saved notebook version 1. At the latest captured evidence cutoff it remained **pending**, so this repository does not promote 0.953 to a verified personal leaderboard result.

The exact public output contained:

| Output property | Recorded value |
|---|---:|
| Rows | **238,260** |
| Nodes | **121,219** |
| Edges | **117,041** |
| Datasets | **4** |
| SHA-256 prefix | **d52a5d…** |

## Public 0.953 diversity audit

Two additional public notebooks advertised at 0.953 were checked at their scored versions:

- Kunal Desale, V10;
- Raunak Dey, V5.

Both produced the **same complete output SHA-256** as the first audited 0.953 notebook, with the same row/node/edge counts. They were not independent prediction systems for ensemble purposes. The runner correctly consumed **zero additional submission slots**.

## V1284 / source dissection

The small public refinement artifact was inspected with a weights-only CPU loader. Its tensors correspond to a compact **224→32→3** coordinate-regression MLP plus 224-dimensional normalization statistics.

Source comparison also showed that the public 0.953 lineage contains additional relinking/readmission/gap-repair behavior. Therefore the public-score difference cannot be attributed to the refinement head alone from the available evidence.

A clean head-on/head-off causal experiment did not reach a scientific result before the deadline because of environment and historical-source/version issues. Those are engineering failures, not model-quality evidence.

## HOCT frozen-detection branch

The public HOCT higher-order tracker was pinned to a specific source revision and brought to real GPU inference while the detection layer remained frozen.

The final branch reached complete neural inference on a large validation movie, but:

- the default global ILP path first encountered an unavailable Gurobi license and exceeded the bounded SCIP fallback window;
- a faster source-preserving decoder then reached inference but failed on a downstream graph-API call before native scoring.

No HOCT native metric result was established at this evidence cutoff. It is recorded as unfinished engineering work, not a scientific negative.

## Earlier research outcomes retained

| Study | Evidence | Decision |
|---|---|---|
| Harmonic Fusion reproduction | Official public score **0.947** | Verified reference |
| Association feature ablation | 557 numeric features; no tracking improvement | Retain frozen probabilities |
| Sparse-real division fine-tuning | Exact graph gain **0.0** | Close event-classification line |
| Grouped tracklet transformer | Held-out parent-F1 gains on early folds | Extend to cross-fitting |
| Cross-fitted tracklet MOE | Mean held-out parent-F1 gain about **+0.0090** across six movies | Exact confirmation only |
| Dense parental point model | Added true associations but too many false links/forks | Close point-only branch |
| Native metric reconciliation | Corrected proxy/native mismatch across frozen systems | Use native full-graph evidence |
| Harmonic fastlane | Exact linker parity; about **34.7×** component speedup | Retain optimization |

## Engineering evidence

The private project ledger records **45 completed scientific workflows** at this publication cutoff. Scientific stops and delivery/runtime failures are tracked separately so a failed hypothesis is not confused with a broken execution.

Important runners are bounded, resumable, hash-gated, and expected to emit compact diagnostic returns on both success and failure. Packaging tests and status-only reads are not counted as scientific workflows.

## Final interpretation

The strongest verified personal leaderboard result at this evidence cutoff remains **0.947**.

The project nevertheless produced a stronger research conclusion than the earlier publication snapshot: locally selected graph improvements on contaminated/repeatedly inspected cohorts were not reliable enough to predict official performance. The final-day work shifted the emphasis toward fidelity auditing, genuinely independent validation, and structurally different representations rather than more micro-adjustments to the same graph.

## Publication boundary

This public repository intentionally omits private competition data, AWS account state, model weights, exact feature recipes, tuned thresholds, working checkpoints, per-movie candidate graphs, return bundles, and executable submission logic. It is an employer-facing, semi-reproducible case study of research decisions and verified aggregate evidence.
