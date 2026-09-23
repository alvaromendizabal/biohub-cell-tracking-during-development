# Results and verification

Evidence cutoff: **September 23, 2026, 01:42 UTC**. This is a historical results
snapshot, not a live leaderboard or a promise of future performance.

## Official benchmark

| Evidence | Recorded result |
|---|---:|
| Official public score | **0.947** |
| Submission identifier | **56376695** |
| Saved notebook version | **1** |
| Public leader observed September 21, 17:51 UTC | **0.974** |
| Gap within that captured public-score comparison | **0.027** |

The scored entry reproduces the external Harmonic Fusion reference. These figures do
not establish a private-leaderboard score or a newly improved model.

## Completed research outcomes

| Study | Evidence | Decision |
|---|---|---|
| Association feature ablation | 557 numeric features; no improvement in tracking decisions | Retain frozen probabilities |
| Division-retention applicability | No eligible two-daughter proposals in eight saved graphs | Stop this configuration |
| Temporal reassignment | Five-movie evaluation; zero accepted exchanges; no metric gain | Reject |
| Exact error attribution | 2,391 correct, 111 incorrect, 112 missed edges | Prioritize measured failure mechanisms |
| Competing-parent division model | 108 added forks; unchanged measured score | Reject |
| Geometry + image continuation | Completed four-fit screen; unchanged measured score | Reject |
| Open-target assignment | Expanded candidate set; no measured tracking improvement | Reject |
| Pretrained neural decoder | Reduced local score by dropping correct links | Reject |
| Link-count-controlled neural decoder | Preserved count but created false divisions and regressed | Reject |
| Learned neural link ranking | Neural-feature ranker underperformed frozen reference | Reject |
| Distinct detector localization | Moving reference centers reduced local score | Reject |
| Detector-supported gap completion | Added nodes/bridges without scored edge or division recovery | Reject |
| Detector-informed precision pruning | +0.000166 local gain, exactly matched by equal-count control | No demonstrated detector value |
| Hand-generated synthetic division rank transfer | Four selected transactions; zero score or division-TP gain | Close this line |

The five-movie frozen baseline combined score is **0.933046**, comprising adjusted edge
Jaccard **0.918760** plus the evaluator's division contribution. These local numbers are
not interchangeable with the official 0.947 public score.

## Current training pivot

The active lane moves from post-hoc graph heuristics to **sequence-level supervised
division learning** using a public fully labelled synthetic lineage resource.

The AWS ingestion milestone verified:

- public notebook source capture;
- a complete **19-page / 3,716-file** output inventory;
- identification of **2,174 sequence NPZ outputs**;
- deterministic selection of a bounded training subset;
- project regressions and Plotly preflight.

The first bounded transfer stopped before model fitting because a Kaggle kernel-output
inventory byte count did not equal the downloaded NPZ byte count. That is treated as a
transport-integrity issue, not a modeling result. The next run verifies the actual
downloaded NPZ containers, SHA-256 hashes, and real byte budget before training.

No public-sequence model result is claimed yet.

## Software and publication evidence

The earlier research-publication checkpoint contained seven verified notebooks and
passed a 277-test synthetic suite. Pull request 1 merged at commit
`b28e74eeb335860f03850bb30a646fe9468d12b0`; both pull-request and post-merge CI
succeeded. Pull request 2 converted the default branch to this presentation-oriented
portfolio while keeping the AWS research workspace intact.

The public repository intentionally does not redistribute working datasets, weights,
private caches, exact training recipes, or the active research implementation.
