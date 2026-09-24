# Results and verification

Evidence cutoff: **September 24, 2026, 03:10 UTC**. This is a historical research snapshot, not a live leaderboard or a promise of future performance.

## Official benchmark

| Evidence | Recorded result |
|---|---:|
| Verified official public score | **0.947** |
| Verified submission identifier | **56376695** |
| Saved notebook version | **1** |
| Second image-division probe | Submission **56503335**, pending at latest returned account check |
| Highest account-side public leader snapshot retained in project evidence | **0.974** |
| Gap from verified 0.947 to that captured leader | **0.027** |

The scored entry reproduces the external Harmonic Fusion reference. These figures do not establish a private-leaderboard score or a newly improved model.

## Frozen five-movie development reference

| Metric | Value |
|---|---:|
| Combined local score | **0.933046** |
| Adjusted edge Jaccard | **0.918760** |
| Edge TP / FP / FN | **2,391 / 111 / 112** |
| Division TP / FP / FN | **1 / 1 / 5** |

These local figures are not interchangeable with the official 0.947 public score.

## Completed research outcomes

| Study | Evidence | Decision |
|---|---|---|
| Association feature ablation | 557 numeric features; no improvement in tracking decisions | Retain frozen probabilities |
| Division-retention applicability | No eligible two-daughter proposals in saved graphs | Stop |
| Temporal reassignment | Zero accepted useful exchanges | Reject |
| Exact error attribution | 2,391 correct, 111 incorrect, 112 missed edges | Prioritize measured failure mechanisms |
| Competing-parent division model | Changed graph structure without metric gain | Reject |
| Geometry + image continuation | Full screen unchanged | Reject |
| Open-target assignment | Expanded candidates without tracking improvement | Reject |
| Neural affinity decoder | Dropped correct links | Reject |
| Count-controlled neural decoder | Preserved edge count but introduced false divisions / regression | Reject |
| Learned neural ranking | Underperformed frozen reference | Reject |
| Detector localization | Moving reference centers reduced score | Reject |
| Detector-supported gap completion | Added nodes/bridges without scored recovery | Reject |
| Detector-informed pruning | +0.000166 local gain, matched by equal-count control | No demonstrated detector value |
| Hand-generated synthetic division rank transfer | Selected plausible events; zero division-TP or score gain | Close line |
| Public synthetic graph division model | Strong synthetic discrimination; five-movie score unchanged | Close line |
| 64-sequence five-frame image model | Very high synthetic validation; five-movie score unchanged | Close line |
| Cross-embryo domain adaptation | Reduced embedding discrepancy; selected same useful events | Close line |
| 256-sequence image scale-up | Improved synthetic AP slightly; real graph score unchanged | Close line |
| Sparse-real division fine-tuning | Real held-out AP saturated at 1.0; exact five-movie score unchanged | Close line |
| Real-sparse transfer audit | Score delta **0.0**, division TP gain **0** | Close event-classification line |

## What the negative results established

The project repeatedly observed that candidate-event classification could improve without changing the final graph metric. That is now treated as a structural finding, not a reason for more threshold tuning.

The remaining high-value direction is the temporal-linking component itself.

## Active frontier: integrated multi-frame linker

The current AWS lane fine-tunes the public baseline's node-linking transformer with sparse real continuation and division edges while keeping the public visual encoder/detector fixed.

The research contract includes:

- multi-frame temporal windows;
- all available sparse real edge supervision rather than division-only events;
- resumable acquisition and immutable object caches;
- cached frozen visual features;
- held-out validation before development inference;
- source-versus-candidate component comparison;
- the same exact full-graph evaluator used in prior studies.

At this evidence cutoff, the integrated-linker run was still in resumable data acquisition and had encountered handled Kaggle rate limits. **No training or score improvement from this lane is claimed yet.**

## Software and publication evidence

The earlier research-publication checkpoint contained seven verified notebooks and passed a 277-test synthetic suite. Pull request 1 merged at commit b28e74eeb335860f03850bb30a646fe9468d12b0; both pull-request and post-merge CI succeeded.

Subsequent presentation-focused pull requests kept the repository public while separating the employer-facing case study from the active AWS research implementation.

The public repository intentionally does not redistribute working datasets, weights, private caches, exact training recipes, or the active research code.
