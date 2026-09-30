# Final sprint reproduction notes

## Scope

This document preserves the September 29 deadline sprint at the level needed to reproduce its evidence after AWS decommissioning.

The scientific ledger at the final publication cutoff was 45 completed scientific workflows. Engineering/runtime failures were tracked separately from scientific negative results.

## Official regression review

A fresh authenticated Kaggle read established:

- 56376695: reference, COMPLETE, 0.947;
- 56503335: image/division probe, COMPLETE, 0.946;
- 56659462: locally qualified source-preserving candidate, COMPLETE, 0.946.

The 0.946 candidate had previously improved the retrospective five-movie score from 0.9330456734 to 0.9480594131. The official regression therefore superseded the local promotion as generalization evidence.

## Fidelity audit

The saved 0.947 and 0.946 prediction outputs were bound by SHA-256 before comparison.

The preview comparison found the same detection universe and coordinates. The candidate differed through eight association-edge edits across four movies. The audit is descriptive: it cannot establish hidden-test causality without hidden labels.

The clone-level reproducer for this comparison is `scripts/reproduce_final_evidence.py compare`.

## Public 0.953 reproduction

A public notebook advertised at 0.953 was verified by saved source/output/version identity and accepted as submission 56687425. It remained pending at the captured evidence cutoff.

The audited public output has:

- 238,260 rows;
- 121,219 node rows;
- 117,041 edge rows;
- SHA-256 `d52a5da2ae5cb0d1b22499f6ca51a00838a6c32ae9a1986ec756ea72e7909e03`.

Two other public notebook versions advertised at 0.953 produced the same complete output byte-for-byte. They were therefore not independent ensemble members.

The clone-level reproducer is `scripts/reproduce_final_evidence.py duplicate`.

## V1284 source/head audit

The audited public lineage contained a compact coordinate-refinement head with a 224→32→3 MLP structure plus normalization tensors. The public notebook also contained additional relinking/recovery logic, so the full public-score difference was not attributed to the small head alone.

The pinned head SHA-256 is recorded in `reproducibility/external_assets.json`.

A clean same-source head-on/head-off metric result was not established before the deadline because of environment/historical-source issues. That absence is preserved rather than converted into a model claim.

## HOCT frozen-detection branch

The public HOCT source and released weights were pinned by commit/hash. The branch held detections fixed and attempted to replace only association logic.

Real GPU neural inference completed on the AWS L4 runtime. Global ILP decoding encountered an unavailable Gurobi license and exceeded the bounded SCIP fallback window. A later fast decoder reached complete neural inference but hit a downstream TracksData edge-construction API issue before native scoring.

No HOCT native-metric result was established. This is unfinished engineering work, not a scientific negative.

## Archived runner identities

The exact final-sprint handoff artifact SHA-256 values are preserved in `reproducibility/external_assets.json`. They allow an independently retained runner binary to be authenticated even though the Git archive uses clean source/tests rather than committing opaque handoff binaries.
