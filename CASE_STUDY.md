# Case study | Tracking cells under sparse supervision

## The problem

A tracking system must detect cells, preserve identity through time, and represent divisions without creating invalid lineage structure. Sparse annotations complicate evaluation: absence is not automatically a negative example, while the competition metric still penalizes wrong links, false forks, and detection-count mismatch.

## My role

I reproduced a scored external reference, implemented original temporal and graph representations, built an AWS-centered experiment and evidence workflow, reconciled proxy and native metrics, integrated completed research back into the strongest full baseline, audited prediction fidelity after an official regression, reproduced a public 0.953 saved-output lineage, and explored a higher-order association model under frozen detections.

Public architectures, notebooks, checkpoints, and organizer code remain attributed to their original authors.

## Benchmark first

The project established an official **0.947** Kaggle reference. Development metrics and diagnostic studies were kept separate from official performance.

The research moved through feature ablation, graph corrections, sparse-real transfer, grouped temporal models, mixture-of-experts ranking, dense parental competition, image-conditioned variants, native metric reconciliation, and source-preserving integration.

## A local win that failed officially

The strongest retrospective integration replayed **37 saved policies and two controlled combinations** against the original Harmonic development control.

The selected candidate improved the local exact metric from **0.933046 to 0.948059** (**+0.015014**), increased correct edges from 2,391 to 2,392, reduced false edges from 111 to 110, and recovered one additional annotated division without adding a false division.

The candidate then scored **0.946** on Kaggle, below the unchanged **0.947** reference.

That result changed the project more than another local gain would have. The development cohort had been repeatedly inspected and overlapped public pretraining, so the official regression demonstrated that the promotion evidence was not an unbiased generalization estimate.

## Diagnose the regression before adding more models

The next step was a prediction-only fidelity audit.

The baseline and failed candidate outputs were hash-bound, then compared without hidden labels. Their preview detection sets, frames, and coordinates were identical. The difference was a small set of **eight association-edge edits across four movies**.

This did not prove the causal reason for the hidden-score decline, but it ruled out a broad class of accidental detector/coordinate drift explanations. The appropriate response was therefore not another tiny association threshold sweep.

## Reproduce the public frontier carefully

The final-day sprint then audited public notebooks advertised at **0.953**.

One exact saved notebook version was verified by source and output hash and submitted as **56687425**. It was accepted and still pending at the latest captured evidence cutoff, so 0.953 is not presented as a verified personal score.

Two other public 0.953 notebook versions were tested for prediction diversity. They generated the same complete hidden-test output byte-for-byte, with identical node and edge counts. That closed the idea of treating the three notebooks as an ensemble.

A source-level audit also showed that the 0.953 lineage was not simply the 0.947 system plus one tiny coordinate head. The refinement artifact was a compact 224→32→3 MLP, while the notebook also introduced additional relinking and recovery logic. That distinction matters because it prevents falsely attributing a leaderboard gain to the easiest visible component.

## Try a structurally different association model

The next branch evaluated HOCT, a higher-order cell-tracking transformer, while holding the detection layer fixed.

The public implementation reached real GPU inference on the AWS L4 runtime. The first decoding path was blocked by an unavailable Gurobi license and a bounded SCIP timeout; a later fast decoder reached the end of neural inference but hit a graph-API integration error before native scoring.

No scientific HOCT result was established before the publication cutoff. Recording that distinction is intentional: an unfinished execution is not evidence that the model failed.

## Engineering decisions

Several engineering controls became first-class project artifacts:

- immutable hashes for scored and candidate outputs;
- separate source-identity and prediction-fidelity evidence axes;
- isolated CPU/GPU/runtime overlays rather than destructive environment mutation;
- bounded execution time and cost;
- resumable caches and explicit reuse contracts;
- compact failure returns;
- persistent notebook-output checks;
- idempotent remote submission guards.

A parity-checked motion-linking optimization also reduced a measured component from roughly 80.5 seconds to 2.32 seconds, about **34.7×**, without changing selected edges.

## Evidence limitations

The development movies are retrospective diagnostics, not clean hidden-test validation. Some public pretrained detector checkpoints were trained on all annotated train videos, so train-movie evaluation is explicitly treated as contaminated/in-sample diagnostic evidence.

The official 0.946 regression is therefore weighted more heavily than the local 0.948059 promotion result.

This repository intentionally omits private datasets, model weights, exact feature recipes, tuned thresholds, per-movie corrections, executable runners, and submission machinery.
