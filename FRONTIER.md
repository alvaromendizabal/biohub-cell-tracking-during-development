# Research directions

This document records the technical directions that remain interesting after the published evidence snapshot. It is intentionally framed around **validation quality, representation, and system design** rather than external ranking.

## Current technical conclusions

- A stable frozen reference is essential for controlled graph experiments.
- Retrospective metrics on repeatedly inspected embryos are useful diagnostics but weak promotion evidence.
- Full-graph behavior can contradict parent-level or local association improvements.
- Prediction diversity must be measured at the output/error level; different model names do not guarantee complementary behavior.
- Detection quality, association quality, and division topology should be evaluated separately before end-to-end integration.
- Runtime optimizations should carry parity evidence when they affect graph construction.

## Highest-value future work

### 1. Clean embryo-disjoint validation

Build a provenance-aware split where upstream pretrained components have not seen the primary validation embryos, and keep that split insulated from iterative manual tuning.

### 2. Frozen-detection higher-order association

Complete a structurally different higher-order association comparison while holding detections fixed. This isolates association quality from detector changes and tests whether richer global context provides genuinely different residual behavior.

### 3. Localization-focused detector study

Compare a temporal 3D reference with a high-resolution 2.5D/3D alternative using centroid error, close-neighbor separation, temporal consistency, and downstream graph quality.

### 4. Out-of-fold fusion only when diversity is demonstrated

Learned fusion should be attempted only after candidate systems show complementary held-out errors. Out-of-fold predictions are required so the fusion layer does not train on in-sample base-model behavior.

### 5. Division logic after detector/linker selection

Division-specific modeling should be evaluated after upstream components are frozen, with promotion based on complete graph behavior rather than isolated division recall.

## Closed or deprioritized branches

- broad feature expansion without graph improvement;
- point-only parental competition that increases false links/forks;
- repeated micro-adjustments on the same inspected development cohort;
- ensemble ideas based on nominal model diversity without prediction diversity.

## Evidence required to reopen a branch

A previously closed idea should return only with at least one of:

- new independent validation data;
- a materially different representation;
- a corrected evaluation contract;
- a measured complementary error pattern;
- a resolved engineering blocker that prevented a valid scientific result.

## Public/private boundary

The repository publishes enough first-party code and evidence for technical review while withholding private data, third-party weights, tuned private thresholds, large caches/checkpoints, and private orchestration.
