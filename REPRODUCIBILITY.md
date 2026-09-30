# Reproducibility boundary

## Purpose

This repository is intentionally **semi-reproducible**: a reviewer should be able to understand the experiment contracts, inspect the aggregate evidence, follow why systems were promoted or rejected, verify attribution, and see the engineering controls without receiving a turnkey competition implementation.

The public repository is therefore more than a slide deck, but intentionally less than a submission package.

## What is public and reproducible

The current tree exposes:

- the verified official 0.947 reference and its attribution boundary;
- the official 0.946 regression of the locally qualified candidate;
- aggregate development metrics for major completed experiments;
- the prediction-fidelity audit separating detection/coordinate drift from association edits;
- the public-0.953 output lineage and duplicate-output audit;
- the distinction between completed scientific results and unfinished engineering branches;
- the progression from feature ablation through structured temporal models and higher-order association;
- the promotion/kill-gate philosophy;
- the presentation notebook and selected saved evidence figure;
- the public-snapshot integrity checker and CI workflow.

The public integrity entry point remains scripts/check_showcase.py.

## High-level experimental contract

The main research loop is reproducible at the level of method:

1. **Freeze a scored baseline.** Preserve reference predictions, model family, evaluator, and configuration identity.
2. **Generate candidates without retrospective labels.** Candidate systems may come from learned association, representation, graph-fusion, or source-preserving correction families.
3. **Freeze complete candidate outputs before retrospective scoring.**
4. **Score with the native full-graph evaluator.** Track aggregate score, edge true/false positives and negatives, divisions, topology, and embryo-level behavior.
5. **Promote conservatively.** Require material gain and preservation of trusted structure.
6. **Submit the prediction-time implementation, not hand-edited development graphs.**
7. **Treat official evidence as stronger than retrospective development evidence.**
8. **When an official result contradicts local validation, audit prediction fidelity before changing models.**
9. **Separate source identity from output identity.** A source-provenance mismatch does not automatically invalidate a hash-bound prediction comparison, and prediction parity does not prove source parity.
10. **Audit ensemble diversity from predictions, not notebook names.**

This sequence is sufficient to understand and audit the project logic while withholding tuned competition-specific mechanics.

## Reproducing the final-day evidence conceptually

The final-day work added three public-safe reproducibility patterns.

### 1. Regression fidelity audit

Two complete saved prediction outputs were verified by cryptographic hash before comparison. Their movie coverage, detections, frames, coordinates, and associations were compared without hidden labels.

The published conclusion is limited to what that comparison establishes: the failed candidate preserved the preview detection universe and changed only a small number of associations. It does not claim access to hidden-test causality.

### 2. Public-output lineage audit

Three public notebooks advertised at 0.953 were compared at their saved scored versions using complete submission outputs.

The outputs matched byte-for-byte, including row/node/edge counts and SHA-256. This is enough to conclude that the notebooks do not provide independent prediction diversity for ensembling, without redistributing the underlying competition output.

### 3. Higher-order association boundary

The HOCT branch was run with frozen detections and pinned public source/model provenance. GPU inference reached completion, but native metric scoring did not complete before the evidence cutoff because of decoding/integration failures.

The repository therefore records the branch as unfinished. Reproducibility includes the ability to reproduce a failure boundary, not only successful scores.

## Determinism and engineering controls

The private AWS implementation records and checks:

- source and artifact identities;
- data/cache provenance;
- software and hardware context;
- checkpoint provenance;
- bounded runtime and cost;
- stage-level heartbeats;
- checkpoint/resume receipts;
- immutable completed outputs;
- topology and schema validation;
- notebook execution and saved-output persistence;
- compact return bundles on success and failure;
- duplicate-submission guards.

The public CI checks that the showcase contains only declared files, that the presentation notebook remains markdown-only, that the selected image asset is structurally valid, that common credential/private-path markers are absent, and that the official/local evidence claims remain consistent.

## What is intentionally not public

To avoid turning the portfolio into a copyable competition solution, this repository does **not** publish:

- model weights or fitted checkpoints;
- exact feature recipes or learned tensors;
- tuned numerical thresholds;
- private training/validation caches;
- candidate graph files or per-movie edits;
- private AWS paths, account state, or credentials;
- active experiment orchestration;
- saved-notebook submission machinery;
- private inference assets;
- return bundles or packaged runners.

These omissions are deliberate and are part of the project boundary.

## How to interpret the numbers

The official **0.947** result is a scored Kaggle submission.

The local **0.948059** result is a retrospective five-movie development score and is **not directly comparable** to the official public score.

The same locally qualified policy later scored **0.946 officially**. That official regression is the stronger generalization evidence.

Submission **56687425** is an exact saved-notebook reproduction of a public system advertised at 0.953. It was accepted and still pending at the latest captured evidence cutoff. This repository therefore does not count 0.953 as a verified personal result.

## Final reproducibility frontier

If the research is continued, the next high-value work is not another local graph micro-adjustment. It is:

- clean embryo-disjoint validation for upstream detection;
- a completed frozen-detection HOCT native-metric comparison;
- a same-source localization ablation;
- a genuinely independent detector representation;
- diversity-aware ensembling based on prediction/residual differences.

The public repository preserves the methodology and evidence boundary while keeping the competition-specific implementation private.
