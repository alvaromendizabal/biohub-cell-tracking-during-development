# Results and verification

Evidence cutoff: **September 28, 2026, 05:10 UTC**. This is a historical research snapshot, not a live leaderboard claim or a promise of future performance.

## Official benchmark

| Evidence | Recorded result |
|---|---:|
| Verified official Kaggle public score | **0.947** |
| Verified submission identifier | **56376695** |
| External reference | Harmonic Fusion saved-notebook inference |
| Official improvements beyond 0.947 | **0** |

Historical leaderboard snapshots captured during the project are not presented as current standings. Local development scores below are **not interchangeable** with the official 0.947 public score.

## Qualified full-baseline integration

Recent work replayed compatible completed research against the original Harmonic development control rather than evaluating only weaker component baselines.

Thirty-seven saved policies plus two controlled combinations were evaluated under one frozen full-graph contract. The selected source-preserving candidate produced:

| Five-movie local exact metric | Original Harmonic control | Qualified candidate | Change |
|---|---:|---:|---:|
| Combined local score | **0.933046** | **0.948059** | **+0.015014** |
| Edge TP | 2,391 | **2,392** | +1 |
| Edge FP | 111 | **110** | -1 |
| Edge FN | 112 | **111** | -1 |
| Division TP | 1 | **2** | +1 |
| Division FP | **1** | **1** | 0 |
| Division FN | 5 | **4** | -1 |

The candidate passed the predeclared local promotion checks for score gain, true-positive preservation, embryo stability, topology, material graph change, and no additional false divisions.

This result is **retrospective development evidence selected among multiple candidate systems on repeatedly inspected movies**. It is not a clean hidden-test estimate, and **0.948059 is not a Kaggle score**.

An all-conflict-free combination of historical changes scored below the original control and lost many correct edges. The selected policy therefore remains deliberately conservative rather than aggregating every historical idea.

## Recent work since the previous publication snapshot

| Milestone | Evidence | Decision |
|---|---|---|
| Image-conditioned parent-ranking pilot | Four fixed fits completed; calibrated image policy accepted no corrections | Close fixed image-ranker branch |
| Harmonic fastlane | Exact motion-linker parity on five native graph comparisons; about **34.7×** summed component speedup | Retain optimization |
| Completed-work integration | 37 saved policies + 2 combinations tested against original Harmonic control | Promote one source-preserving candidate to deployment |
| Dynamic deployment replay | Saved winner reconstructed from prediction-time artifacts with no hard-coded development edits | Pass deployment qualification |
| Shared-encoder verification | Candidate head shown compatible with the frozen image encoder under the deployment contract | Reuse encoder features |
| Thin private notebook | Inference assets externalized; notebook reduced to a lightweight loader/orchestrator | Retain for remote delivery |
| Guarded delivery state machine | Upload, launch, output verification, and submission separated into resumable idempotent states | Prevent duplicate pushes/submissions |
| Current remote state | Private inference input creation acknowledged; notebook execution and official submission not yet verified | Continue delivery; no new score claim |

## Earlier completed research outcomes

| Study | Evidence | Decision |
|---|---|---|
| Harmonic Fusion reproduction | Official public score **0.947** | Verified reference |
| Association feature ablation | 557 numeric features; no tracking improvement | Retain frozen probabilities |
| Exact error attribution | Measured correct, incorrect, and missed links plus division failures | Prioritize concrete failure mechanisms |
| Sparse-real division fine-tuning | Event-level validation saturated; exact graph gain **0.0** | Close event-classification line |
| Grouped tracklet transformer | Held-out parent-F1 improved across folds | Continue to broader cross-fitting |
| Cross-fitted tracklet MOE | Mean held-out parent-F1 gain about **+0.0090** across six movies | Exact confirmation only |
| MOE exact comparison | Best arm gained about **+0.0081** locally but missed the strict gate | Close MOE line |
| Dense parental point model | Added some true associations but produced too many false links/forks under native scoring | Close point-only division line |
| Native metric reconciliation | Corrected a proxy/native mismatch across 18 frozen systems | Use native full-graph evidence for decisions |

## Delivery and runtime evidence

The deployment path preserves the selected model behavior while reducing repeated work.

The motion-linker optimization reproduced the original selected edges, edge attributes, and reported statistics across five native graph comparisons. Summed measured component time fell from roughly **80.5 seconds to 2.32 seconds**, about **34.7×** for that component.

That benchmark does not include image inference, remote queueing, other post-processing, or submission overhead. It must not be interpreted as a whole-notebook runtime ratio.

The current delivery design also removes repeated validation/reselection and separates the private inference assets from the small saved notebook. Remote input creation has been acknowledged, but private notebook completion and official scoring remain unverified at this evidence cutoff.

## Engineering evidence

The private research ledger currently records **43 completed bounded workflows**. Execution failures, scientific stops, and remote-service blocks are tracked separately in the private project state rather than collapsed into model-quality results.

Important runs are bounded, resumable, hash-gated, and expected to produce compact diagnostic returns on success or failure. Passing local tests are not counted as completed scientific workflows.

## Current research boundary

The next milestone is **delivery confirmation of the qualified full-baseline candidate**, not another model-training round.

The candidate must run dynamically on unseen competition movies, produce a valid complete output, pass version/input/topology checks, and receive an actual Kaggle score before any stronger performance claim is made.

If the official result does not improve the 0.947 reference, the next modeling branch should be structurally different rather than another small threshold sweep.

## Publication boundary

This public repository intentionally omits private competition data, AWS account state, model weights, exact feature recipes, tuned thresholds, working checkpoints, private caches, candidate graphs, and executable submission logic. It is an employer-facing, semi-reproducible case study of research decisions and verified aggregate evidence.
