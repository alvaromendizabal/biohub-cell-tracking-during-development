# 3D Cell Tracking & Lineage Analysis

**Alvaro Mendizabal · Machine Learning Engineer**  
[GitHub profile](https://github.com/alvaromendizabal)

### Computer vision. Graph learning. Evidence-led ML engineering.

I built an AWS research system for tracking cells through three-dimensional time-lapse microscopy: reproducing a scored external benchmark, engineering original representations, training structured temporal models, reconciling proxy and native graph metrics, and tracing failures to the exact association and lineage decisions responsible.

**Verified Kaggle public score: 0.947.** Official submission 56376695 reproduced the public Harmonic Fusion inference reference. The original pretrained systems and public competition baselines are credited to their authors; the research orchestration, original feature work, controlled experiments, transfer studies, metric reconciliation, error analysis, and AWS execution infrastructure described here are my project contributions.

[Technical case study](CASE_STUDY.md) · [Results and evidence](RESULTS.md) · [Current research frontier](FRONTIER.md) · [Visual portfolio](notebooks/portfolio.ipynb)

## What I built and demonstrated

| Capability | Evidence from the project |
|---|---|
| Representation engineering | Evaluated 557 association features and progressed from hand-designed graph corrections to grouped temporal and dense-parental learned representations. |
| Structured prediction | Built source-preserving association, grouped tracklet, mixture-of-experts, parental-competition, and topology-gated graph systems. |
| Metric-faithful evaluation | Reconciled proxy and native tracking metrics across six movies and 18 frozen graph systems before deciding the next research direction. |
| Exact error attribution | Traced continuation and division failures to candidate coverage, wrong competing parents, false forks, and missing unique detector matches. |
| Cloud ML engineering | Established AWS-first execution with immutable caches, resumable checkpoints, artifact hashes, failure recovery, bounded compute, and persistent notebook evidence. |
| Technical judgment | Closed plausible model families when native graph evaluation showed that extra true links were outweighed by false links or false divisions. |

## Latest verified research result

The latest completed audit used the pinned native graph metric on **18 frozen graph systems**: six source graphs plus pairwise and contextual learned graphs for each movie. No new model fit or inference was performed during reconciliation.

The point-only learned models recovered **2 of 10 division events** and added **17 correct edges** versus the raw-source comparator, but they also introduced **89–94 false divisions** and roughly **104–105 additional false edges**. Their aggregate native scores (**0.8058 pairwise, 0.8054 contextual**) were below the raw-source comparator (**0.8301**).

That result closed the point-only division branch. The current frontier is **image-conditioned temporal division-event scoring with source preservation**, using independently verified supervision before any new fit. No improved official score is claimed.

## Research progression

1. **Reproduce the benchmark.** The external Harmonic Fusion reference was reproduced and scored at 0.947.
2. **Test richer associations.** Hundreds of original and public features were evaluated against frozen baseline predictions.
3. **Attribute exact failures.** Full graph evaluation exposed continuation-link errors and sparse division failures.
4. **Move from post-processing to learned graph context.** Grouped tracklet and mixture-of-experts models improved held-out parent association but did not clear the full exact promotion gate.
5. **Test dense parental competition.** Eight fixed CUDA fits increased some correct associations but also created many false links and false division events under the native metric.
6. **Reconcile the metric before proceeding.** A zero-fit native audit settled the discrepancy between proxy screens and the pinned organizer metric, then closed the point-only branch.
7. **Current frontier.** Add image evidence to division-event scoring while preserving trusted source associations.

## Evidence discipline

Development cohorts have been inspected repeatedly, so local improvements are treated as engineering evidence rather than independent proof of leaderboard generalization. Official competition claims remain limited to scored submissions. Negative experiments are retained because they narrow the search space and prevent repeated spending on non-improving directions.

## About this portfolio

This **public repository is a hiring-oriented case study**, not a training kit or an open-source competition implementation. It contains high-level methods, aggregate results, selected genuine visuals, attribution, and research decisions—not working datasets, model weights, exact feature recipes, account state, or the active AWS implementation.

[Rights](RIGHTS.md) · [Attribution](ATTRIBUTION.md)
