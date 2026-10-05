# Results and verification

Evidence cutoff: **September 29, 2026, 23:02 UTC**. Results are grouped by evidence type so retrospective diagnostics are not confused with independent scored evaluation.

## Executive evidence table

| Result | Value | Evidence type |
|---|---:|---|
| Faithfully reproduced external reference | **0.947** | Independent scored evaluation |
| Retrospective five-movie baseline | **0.933046** | Development diagnostic |
| Retrospective selected candidate | **0.948059** | Development diagnostic |
| Same candidate on independent scored evaluation | **0.946** | Independent scored evaluation |
| Association features evaluated | **557** | Research / ablation evidence |
| Prediction-fidelity difference after regression | **8 association-edge edits across 4 movies** | Diagnostic audit |
| Motion-linking component speedup | **~34.7×** | Engineering benchmark with parity checks |
| Completed scientific workflows | **45** | Project execution ledger |
| Executed evidence notebooks | **7** | Reproducibility / communication |

## What the independent regression established

The selected candidate improved a repeatedly inspected local cohort from **0.933046 to 0.948059**. Exact graph accounting showed:

| Five-movie local metric | Frozen control | Selected candidate | Change |
|---|---:|---:|---:|
| Combined local score | 0.933046 | **0.948059** | +0.015014 |
| Edge TP | 2,391 | **2,392** | +1 |
| Edge FP | 111 | **110** | -1 |
| Edge FN | 112 | **111** | -1 |
| Division TP | 1 | **2** | +1 |
| Division FP | 1 | **1** | 0 |
| Division FN | 5 | **4** | -1 |

The candidate then scored **0.946** on independent evaluation versus the unchanged **0.947** reference. The correct conclusion is not that the local experiment “won”; it is that repeated inspection and upstream provenance made the development cohort too optimistic for promotion decisions.

## Prediction-fidelity audit

To distinguish model generalization from accidental pipeline drift, the frozen-reference and candidate outputs were verified by SHA-256 and compared structurally.

The audit found:

- identical preview movie coverage;
- identical row counts;
- no added or removed preview detections;
- no shared-ID coordinate shifts;
- no frame conflicts;
- **eight association-edge edits** across four movies.

No hidden labels were used. This audit does not prove causality, but it substantially narrows the failure mode to the intended association changes rather than detector or coordinate corruption.

## Earlier research decisions retained

| Study | Evidence | Decision |
|---|---|---|
| Association feature ablation | 557 numeric features; no tracking improvement | Keep frozen probabilities |
| Sparse-real division fine-tuning | Exact graph gain 0.0 | Close event-classification line |
| Grouped tracklet transformer | Held-out parent-F1 gains on early folds | Extend to cross-fitting |
| Cross-fitted tracklet mixture-of-experts | Mean held-out parent-F1 gain about +0.0090 across six movies | Require exact graph confirmation |
| Dense parental point model | Added true associations but too many false links/forks | Close point-only branch |
| Native metric reconciliation | Corrected proxy/native mismatch | Prefer full-graph evidence |
| Motion-linking optimization | Exact linker parity; ~34.7× component speedup | Retain optimization |

## Research execution accounting

The project ledger records **45 completed scientific workflows** at the publication cutoff. A scientifically negative but correctly executed experiment counts as completed research; broken delivery/runtime incidents are tracked separately.

This distinction matters because “the model did not help” and “the experiment never completed correctly” are different engineering facts.

## Evidence hierarchy

1. **Independent scored evidence** — strongest available generalization check.
2. **Retrospective development evidence** — controlled comparison, not clean promotion evidence.
3. **Diagnostic evidence** — output fidelity, source/version, graph differences, provenance.
4. **Engineering evidence** — runtime, parity, packaging, tests, reproducibility.

## Publication boundary

This public repository intentionally omits private competition data, credentials, AWS account state, third-party weights, tuned private thresholds, large checkpoints/caches, private candidate graphs, and private submission/orchestration logic. It publishes the code, selected experiments, executed notebooks, compact evidence, and provenance needed for employer review.
