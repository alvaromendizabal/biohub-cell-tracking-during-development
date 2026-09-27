# Results and verification

Evidence cutoff: **September 27, 2026, 05:08 UTC**. This is a historical research snapshot, not a live leaderboard claim or a promise of future performance.

## Official benchmark

| Evidence | Recorded result |
|---|---:|
| Verified official Kaggle public score | **0.947** |
| Verified submission identifier | **56376695** |
| External reference | Harmonic Fusion saved-notebook inference |
| Official improvements beyond 0.947 | **0** |

Historical leaderboard snapshots captured during the project are not presented here as current standings. Local development scores below are not interchangeable with the official 0.947 public score.

## Latest native-metric reconciliation

The latest completed audit froze **18 existing graph systems**—source, pairwise, and contextual predictions for six movies—and evaluated them with the pinned native graph metric. No new neural fit or inference was run during this reconciliation.

| Native aggregate | Raw-source comparator | Pairwise model | Context model |
|---|---:|---:|---:|
| Adjusted score | **0.830087** | 0.805788 | 0.805368 |
| Edge TP | 2,079 | **2,096** | **2,096** |
| Edge FP | **185** | 289 | 290 |
| Edge FN | 251 | **234** | **234** |
| Division TP | 0 | **2** | **2** |
| Division FP | **0** | 89 | 94 |
| Division FN | 10 | **8** | **8** |

The learned models recovered two division events and 17 additional correct edges, but those gains were dominated by false associations and false forks. The point-only division branch was therefore closed.

## Selected completed research outcomes

| Study | Evidence | Decision |
|---|---|---|
| Harmonic Fusion reproduction | Official public score **0.947** | Verified reference |
| Association feature ablation | 557 numeric features; no tracking improvement | Retain frozen probabilities |
| Exact error attribution | Measured correct, incorrect, and missed links plus division failures | Prioritize concrete failure mechanisms |
| Sparse-real division fine-tuning | Event-level validation saturated; exact graph gain **0.0** | Close event-classification line |
| Grouped tracklet transformer | Held-out parent-F1 improved across folds | Continue to broader cross-fitting |
| Cross-fitted tracklet MOE | Mean held-out parent-F1 gain about **+0.0090** across six movies | Exact confirmation only; no promotion claim |
| MOE exact comparison | Best arm gained about **+0.0081** locally but missed the +0.010 gate and division floor | Close MOE line |
| Dense parental point model | Eight fixed CUDA fits; added some true associations but failed the research screen | Native reconciliation required |
| Native metric reconciliation | 18 frozen graphs; 2 division TPs but 89–94 false divisions | Close point-only division line |

## Engineering evidence

The latest reconciliation run completed in about **30 seconds** on `ml.g6e.2xlarge`, executed **61 regressions**, froze all 18 graph inputs before annotation access, reconciled native counts, executed the employer-facing notebook, and produced a validated return bundle. That run performed **zero new fits, zero new predictions, and zero cloud mutations**.

The research ledger at this cutoff records **39 completed bounded workflows, 34 hard execution failures, and 10 correctly gated scientific stops**. Passing unit tests are not counted as research workflows.

## Current research boundary

The next branch is image-conditioned temporal division-event scoring with source preservation. The objective is not to make the graph more complicated; it is to add discriminative evidence exactly where point-only models confused true mothers with nearby alternatives or produced false splits.

A new candidate must outperform geometry-only controls and then pass the unchanged full-graph promotion requirements before any Kaggle submission is prepared.

## Publication boundary

This public repository intentionally omits private competition data, AWS account state, model weights, exact feature recipes, working checkpoints, private caches, and executable submission logic. It is an employer-facing case study of research decisions and verified aggregate evidence.
