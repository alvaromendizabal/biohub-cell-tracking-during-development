# Reproducibility boundary

## Purpose

This repository is designed to be **semi-reproducible**: a reviewer should be able to understand the experimental contracts, verify the published evidence snapshot, follow the research progression, and see how candidate systems were promoted or rejected without receiving a copy of the active competition implementation.

The public repository is therefore more than a slide deck, but intentionally less than a turnkey submission package.

## What is public and reproducible

The current tree exposes:

- the verified official reference score and its attribution boundary;
- aggregate development metrics for major completed experiments;
- the progression from feature ablation to structured temporal models, native metric reconciliation, full-baseline integration, and deployment;
- the distinction between proxy/component metrics and full tracking-graph evaluation;
- the promotion and kill-gate philosophy;
- the presentation notebook and saved evidence figure;
- the public-snapshot integrity checker and CI workflow;
- the methodological interface between the frozen baseline, candidate graph changes, native evaluation, and saved-notebook delivery.

Run the public integrity check with inline command `python3 scripts/check_showcase.py`.

## High-level experimental contract

The main full-graph research loop is reproducible at the level of method:

1. **Freeze a scored baseline.** Preserve the reference detector, model family, graph output, and evaluation configuration.
2. **Generate candidate systems without retrospective labels.** Candidates may come from feature, learned-association, graph-fusion, or source-preserving correction families.
3. **Express historical candidates as graph changes.** Compare each candidate with its own paired source rather than transplanting an entire weaker graph.
4. **Map only compatible changes onto the strongest anchor.** Require unique same-time spatial correspondence and an unchanged local neighborhood before applying a change.
5. **Preserve unaffected source structure.** Original detections, coordinates, and unrelated associations stay intact.
6. **Freeze every complete candidate before opening development annotations.**
7. **Score with the native full-graph evaluator.** Track aggregate score, edge true/false positives and negatives, division events, topology, and embryo-level behavior.
8. **Promote conservatively.** Require material score gain and preservation of trusted true positives and topology.
9. **Reconstruct the winning policy from prediction-time artifacts.** Do not hard-code development node IDs or label-derived edits.
10. **Verify saved-notebook output before submission.** Treat notebook launch, completion, output validation, and competition submission as distinct states.

This sequence is sufficient to understand and audit the research logic while withholding the tuned competition-specific mechanics.

## Determinism and engineering controls

The private AWS implementation records and checks:

- source and artifact identities;
- data/cache provenance;
- software and hardware context;
- model/checkpoint provenance;
- bounded runtime and cost;
- stage-level heartbeats;
- checkpoint/resume receipts;
- immutable completed outputs;
- topology and schema validation;
- notebook execution and saved-output persistence;
- compact return bundles on success and failure.

The public CI checks that the showcase contains only the declared files, that its presentation notebook remains markdown-only, that the selected image asset is structurally valid, that internal/private paths and common credential markers are absent, and that the official and local evidence claims remain present.

## What is intentionally not public

To avoid turning the portfolio into a copyable competition solution, this repository does **not** publish:

- model weights or fitted checkpoints;
- exact feature recipes or learned feature tensors;
- tuned numerical thresholds;
- private training/validation caches;
- candidate graph files or per-movie edits;
- private AWS paths, account state, or credentials;
- active experiment orchestration;
- dynamic correction implementation;
- saved-notebook submission machinery;
- private inference assets.

These omissions are deliberate and are part of the project boundary, not missing documentation.

## How to interpret the numbers

The official **0.947** result is a scored Kaggle submission.

The **0.948059** result is a local retrospective five-movie score for the current candidate, compared with a **0.933046** local score for the paired original Harmonic control. Those local values are meaningful for controlled engineering comparison but are **not directly comparable** to the official public score.

The local candidate was selected after multiple prior experiments and repeated inspection of the development cohort. It should therefore be treated as deployment evidence, not as an unbiased estimate of hidden-test performance.

## Current reproducibility frontier

The remaining non-public step is remote confirmation: run the qualified prediction-time policy on unseen competition movies, validate the complete saved-notebook output, and obtain an official score.

Until that happens, the repository intentionally states **official improvements beyond 0.947: zero**.
