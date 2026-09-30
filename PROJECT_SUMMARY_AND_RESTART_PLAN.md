# Project summary and restart plan

## Executive summary

This project explored **3D cell detection, temporal association, and lineage reconstruction** for the Biohub Cell Tracking During Development Kaggle competition.

The strongest **verified personal official score** preserved in the project evidence is **0.947**, reproduced from the public Harmonic Fusion reference. A later source-preserving candidate improved the repeatedly inspected development cohort from **0.933046 to 0.948059**, but scored **0.946 officially**. That adverse result became the most important scientific lesson in the project: **validation quality and independence mattered more than another local graph edit**.

The final sprint therefore shifted from “add one more model” to:

- auditing prediction fidelity;
- distinguishing detector drift from association changes;
- reproducing and dissecting the public 0.953 lineage;
- checking whether apparently different public 0.953 notebooks were actually prediction-diverse;
- testing structurally different association mechanisms such as HOCT;
- hardening the AWS/Kaggle delivery path and preserving a reproducible research archive.

The repository now contains the first-party package, research modules, tests, executed evidence notebooks, compact receipts, final-sprint source snapshot, reproduction scripts, and version/hash pins needed to inspect the work from a fresh clone.

---

## The problem

A complete cell-tracking system must solve three coupled problems:

1. **Detection** — identify cells accurately in anisotropic 3D microscopy volumes.
2. **Association** — link detections across time without identity switches, missing links, or false branches.
3. **Division modeling** — identify true parent-to-two-daughter events without inventing invalid lineage structure.

The task is difficult because:

- labels are sparse;
- missing annotations are not automatically negatives;
- microscopy is anisotropic in physical space;
- tracking errors propagate through the lineage graph;
- strong local association metrics can still produce worse full-graph behavior;
- repeatedly inspected development movies can become a poor estimate of hidden-test generalization.

---

## What I built

### 1. A scored reference and evidence baseline

I reproduced the public Harmonic Fusion inference system and established the verified **0.947** official reference.

That reference became the frozen control for later experiments rather than repeatedly changing the baseline.

### 2. Representation and association research

The project evaluated a broad sequence of association ideas, including:

- 557 numeric association features;
- graph correction and source-preserving integration;
- sparse-real transfer experiments;
- grouped higher-order tracklet modeling;
- a cross-fitted tracklet mixture-of-experts ranker;
- dense parental competition;
- image-conditioned parent ranking;
- temporal reassignment;
- division-specific experiments;
- public graph/post-processing mechanisms;
- a higher-order transformer-based tracking branch.

Several systems improved intermediate parent-ranking or local graph metrics. Those improvements were not accepted as final evidence until they survived the native graph metric and, where applicable, official submission.

### 3. Native-metric reconciliation

A major project milestone was moving away from proxy-only conclusions.

Candidate systems were evaluated using organizer-native graph behavior, including:

- correct and incorrect associations;
- missed links;
- division true/false positives;
- topology validity;
- node/detection preservation;
- embryo/movie-level stability.

This closed multiple branches that looked attractive under parent-level or local metrics but damaged the complete graph.

### 4. Source-preserving integration

Rather than replacing the strong reference graph wholesale, completed candidate systems were replayed against the original Harmonic control.

The strongest retrospective candidate improved the five-movie local metric:

- **0.933046 → 0.948059**
- edge TP: **2391 → 2392**
- edge FP: **111 → 110**
- edge FN: **112 → 111**
- division TP: **1 → 2**
- division FP: **1 → 1**
- division FN: **5 → 4**

The official result was **0.946**, below the unchanged 0.947 reference.

That result is deliberately preserved rather than hidden.

### 5. Prediction-fidelity auditing

After the official regression, the project stopped adding corrections and asked a more important question:

> Did the submission accidentally change the detector or coordinate geometry, or did the intended association edits simply fail to generalize?

A hash-bound prediction audit found that the 0.946 candidate preserved the preview detection universe and coordinates. The difference was a small number of association-edge changes.

This substantially reduced the likelihood that broad detector/coordinate drift explained the regression and shifted the conclusion toward **weak generalization evidence for locally selected association edits**.

### 6. Public 0.953 lineage audit

A public saved notebook advertised at 0.953 was reproduced through a version- and output-hash-gated submission path.

Two other public notebooks advertised at 0.953 were then tested for prediction diversity.

All three audited saved outputs were byte-identical:

- **238,260 rows**
- **121,219 nodes**
- **117,041 edges**
- identical full-file SHA-256

The practical conclusion was important: three notebook names did **not** represent three independent ensemble members.

### 7. Higher-order association frontier

The final structurally different branch evaluated RoyerLab HOCT while holding detections fixed.

The public model reached real GPU inference in the AWS environment. Global optimization and later graph-integration issues prevented a completed native-metric result before the competition evidence cutoff.

That branch is recorded as **unfinished engineering work**, not as a failed scientific hypothesis.

### 8. Delivery and reproducibility engineering

The project also became a substantial ML systems exercise.

The private execution framework used:

- AWS as the canonical workspace;
- immutable artifact hashes;
- resumable checkpoints and caches;
- isolated CPU/GPU runtime layers;
- bounded runtime and cost;
- stage-level heartbeats;
- failure packaging;
- persistent notebook-output validation;
- duplicate-submission guards;
- source/output/version checks for Kaggle delivery.

One parity-checked motion-linking optimization reduced a measured component from roughly 80.5 seconds to 2.32 seconds, about **34.7×**, without changing the selected edges.

That figure is a component benchmark, not a whole-pipeline speedup claim.

---

## Verified results at the archived evidence cutoff

| Evidence | Result |
|---|---:|
| Verified official Kaggle score | **0.947** |
| Reference submission | **56376695** |
| Locally qualified candidate | **0.946 official** |
| Image/division probe | **0.946 official** |
| Strongest retrospective five-movie candidate | **0.948059 local** |
| Public 0.953 saved-notebook reproduction | accepted; **pending at archived cutoff** |
| Public 0.953 prediction diversity | **none across the three audited notebooks** |
| HOCT frozen-detection native score | **unfinished at cutoff** |

The local 0.948059 and public notebook's advertised 0.953 are not presented as verified personal leaderboard improvements.

---

## What worked

### Baseline locking

Keeping a stable 0.947 reference made it possible to tell whether new systems truly helped the full graph.

### Native graph evaluation

Several appealing proxy gains disappeared when evaluated at the graph level. Closing those branches saved time and prevented misleading conclusions.

### Conservative integration

Source-preserving integration was substantially better than combining every non-conflicting historical edit.

### Error attribution

The project progressively moved from “score went down” to identifying whether failures came from:

- detection changes;
- association changes;
- division structure;
- source/output drift;
- metric mismatch;
- delivery/runtime behavior.

### Reproducibility discipline

The final archive preserves enough first-party source, tests, executed evidence, and provenance to inspect the project without the deleted SageMaker workspace.

---

## What did not work

### Repeated local optimization on a small development cohort

The most important failure mode was **selection overfitting**.

The locally best candidate looked convincingly better on the repeatedly inspected development movies and still regressed officially.

### Proxy improvements without graph improvements

Parent-level F1, local association quality, or recovered division examples were not sufficient when the complete graph gained false edges or invalid topology.

### Near-duplicate public systems as ensemble candidates

The public 0.953 notebooks looked like separate lineages until complete saved predictions were compared. They were not independent.

### Late-stage environment friction

Several final-sprint experiments lost time to:

- dependency isolation;
- GPU/runtime mismatches;
- solver availability;
- historical asset/version drift;
- graph-library API changes.

These were engineering failures rather than scientific negatives, but they consumed deadline time.

---

# If I restarted this project today

I would change the order of operations substantially.

## Phase 1 — Build clean validation before model research

This would be the highest-priority change.

### Goal

Create a genuinely **embryo-disjoint, provenance-aware validation regime** before tuning detection or association.

### Actions

- Group all validation splits by embryo identifier.
- Audit whether any public pretrained detector or linker has seen each validation embryo.
- Separate:
  - clean unseen-embryo validation;
  - contaminated diagnostic validation;
  - retrospective error-analysis cohorts.
- Freeze a single primary validation split and a secondary stress split.
- Prevent manual inspection of the primary split during iterative development.
- Implement the organizer-native metric first and test it with synthetic edge cases.

### Promotion rule

No model or post-processing change advances based only on the repeatedly inspected historical five-movie cohort.

### Why this comes first

The official 0.946 regression showed that the project had already reached the point where better local metrics did not imply better hidden-test performance.

---

## Phase 2 — Detection and localization first

The next restart would follow a strict ordering:

**detection → linking → division**

The original project spent substantial effort on association while the public frontier suggested that localization and close-cell separation could still be material.

### Candidate detector program

I would compare at least two structurally different detector representations:

1. a TemporalUNet3D-style reference;
2. a high-resolution **2.5D heatmap detector with a pretrained 2D backbone**, depth-aware pooling, and 3D context between encoder and decoder.

### Evaluation

Measure more than recall:

- centroid localization error in physical units;
- close-neighbor separation;
- precision in valid annotated regions;
- temporal detection consistency;
- node-count calibration;
- embryo-level stability;
- downstream native graph score with the linker frozen.

### Why

A better detector can improve association and division simultaneously. Another linker cannot recover information that was never localized correctly.

---

## Phase 3 — Build genuinely independent association systems

Only after detector selection would I compare linkers.

The restart would include:

- the strongest Harmonic-style linker;
- a grouped-tracklet / candidate-set model;
- a completed frozen-detection HOCT evaluation;
- a simple motion/global-assignment control.

All linkers would receive **identical detections**.

### Required evidence

Track:

- native graph score;
- edge precision/recall;
- division behavior;
- embryo-level variance;
- prediction/residual correlation between linkers.

The last item matters because ensembling should be based on **complementary errors**, not model names.

---

## Phase 4 — Learned fusion only with OOF predictions

If detector or linker systems show complementary errors, I would test a compact learned fusion model.

Inputs could include:

- raw/image context;
- detector confidence maps;
- heterogeneous detector outputs;
- association-model scores;
- motion consistency;
- local temporal context.

The fusion model would be trained strictly on **out-of-fold predictions**.

Simple averaging would remain the control.

### Kill condition

If learned fusion does not improve embryo-held-out native metrics, stop immediately.

---

## Phase 5 — Division after detector and linker selection

Division logic would come last.

The project showed that division metrics can improve locally while the complete graph gets worse.

A restart would therefore train/evaluate division-specific logic only after the detector and linker are frozen.

Useful inputs would include:

- precise spatial geometry;
- image context around parent/daughters;
- temporal appearance changes;
- competing-parent context;
- source/linker confidence;
- local topology.

Promotion would require full-graph improvement, not isolated division recall.

---

## Phase 6 — Treat diversity as a first-class ensemble metric

Before submitting multiple systems, compare their actual prediction errors.

For every candidate pair:

- output hash;
- graph overlap;
- edge-disagreement rate;
- matched-node coordinate disagreement;
- residual correlation;
- embryo-level complementary wins.

Two systems with different architecture names but identical outputs count as one system.

This would have identified the public 0.953 duplication immediately.

---

## Phase 7 — Engineer the delivery path before the last week

The final sprint lost too much time to environment and submission mechanics.

If restarting, I would establish early:

- a tested CPU environment;
- a tested GPU environment;
- pinned third-party model/source versions;
- a saved-notebook submission template;
- output/source/version hash verification;
- an idempotent Kaggle submission guard;
- a minimal solver fallback;
- a reproducible dependency overlay;
- final inference timing on realistic test-size data.

The goal would be for the final week to contain **model decisions**, not environment debugging.

---

# A practical restart roadmap

## Week 1 — Validation and reproduction

- Reproduce the 0.947 reference.
- Implement and test the native metric.
- Build embryo-disjoint split/provenance tables.
- Freeze primary and secondary validation sets.
- Establish inference/runtime benchmarks.

**Deliverable:** trusted baseline + validation contract.

## Weeks 2–3 — Detection

- TemporalUNet3D control.
- 2.5D/high-resolution detector.
- Localization and close-neighbor diagnostics.
- Detector diversity analysis.

**Deliverable:** selected detector or detector ensemble.

## Week 4 — Association

- Harmonic/motion control.
- grouped-tracklet candidate model.
- HOCT frozen-detection comparison.
- residual-diversity analysis.

**Deliverable:** strongest stable linker plus one genuinely complementary alternative.

## Week 5 — Fusion and division

- OOF learned fusion if justified.
- division-specific image/temporal head.
- full native graph evaluation.

**Deliverable:** end-to-end candidate families.

## Week 6 — Robustness and submission

- embryo-level stability;
- controlled ablations;
- inference optimization;
- saved-notebook parity;
- submission rehearsal;
- diverse final candidate selection.

**Deliverable:** two or three structurally distinct, fully verified submission candidates.

---

## What I would not repeat

A restart would explicitly avoid:

- optimizing repeatedly against the same tiny inspected cohort;
- broad threshold sweeps after validation saturation;
- treating parent-F1 as sufficient evidence;
- combining all historical graph edits because they are individually conflict-free;
- retraining valid checkpoints without evidence they are wrong;
- assuming different public notebooks imply different predictions;
- leaving solver/runtime/submission integration until the deadline;
- using public train-movie performance as clean detector CV when upstream checkpoints saw those movies.

---

## What would remain reusable

The archived project still contains useful foundations for a restart:

- native evaluation logic;
- graph and feature engineering modules;
- grouped temporal/parental modeling work;
- candidate-set modeling components;
- fidelity-audit tooling;
- public-solution audit patterns;
- AWS execution design;
- reproducibility checks;
- saved evidence notebooks;
- exact final-sprint source snapshot;
- external asset/version/hash manifest.

A restart should reuse those foundations while replacing the **validation strategy and experimental ordering**.

---

## Bottom line

The project did not beat the 0.947 verified reference within the competition window, but it produced a much clearer understanding of why.

The main limitation was not lack of modeling ideas. It was that **evidence quality lagged behind model complexity**.

If I restarted the project, I would invest first in clean embryo-disjoint validation, then improve detection/localization, then compare genuinely independent linkers under frozen detections, and only then spend effort on division logic and learned fusion.

That sequence would give every modeling improvement a much better chance of surviving contact with unseen embryos—and would make the final leaderboard result far more predictable.
