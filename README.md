# 3D Cell Tracking & Lineage Analysis

**Alvaro Mendizabal · Machine Learning Engineer**  
[GitHub profile](https://github.com/alvaromendizabal)

### Computer vision. Graph learning. Evidence-led ML engineering.

I built an AWS research system for tracking cells through three-dimensional time-lapse
microscopy: reproducing a scored external benchmark, engineering original representations,
evaluating structured association models, and tracing failures to the exact detection,
link, and lineage decisions responsible.

**Verified Kaggle public score: 0.947.** Official submission 56376695 reproduced the
public Harmonic Fusion inference reference. The original pretrained system is credited
to its authors; the engineering, integration studies, original feature work, controlled
experiments, and error analysis described here are my project contributions.

[Technical case study](CASE_STUDY.md) · [Results and evidence](RESULTS.md) ·
[Visual portfolio](notebooks/portfolio.ipynb)

## What I built and demonstrated

| Capability | Evidence from the project |
|---|---|
| Representation engineering | Evaluated 557 association features: 32 public-score features and 525 original features covering motion, geometry, context, and related representations. |
| Structured prediction | Implemented and tested graph-preserving temporal corrections and learned competing-parent division decisions. |
| Rigorous evaluation | Reconciled 111 false and 112 missed edges across five full development movies, with separate topology-aware division analysis. |
| Cloud ML engineering | Established AWS execution with cache reuse, checkpoint integrity, failure recovery, numerical checks, bounded computation, and persistent notebook evidence. |
| Software quality | The original research publication passed 277 synthetic tests and GitHub CI before and after its merge. These are engineering checks, not a claim of statistical generalization. |
| Technical judgment | Retained the stronger benchmark when new models did not improve the measured tracking objective; separated official scores from exploratory development metrics. |

## Selected technical evidence

![Exact error attribution](assets/exact_error_budget_0.png)

A genuine saved figure from the completed error analysis. It shows where the development
baseline fails; it is not a projected score improvement.

## Research status

The benchmark reproduction and documented studies are completed milestones. Image-informed
association is the active follow-up. The initial extractor stopped on an image-domain check
before fitting; the completed follow-up screen is recorded in RESULTS.md, separately from the verified benchmark.

The last captured public leader scored 0.974 on September 21, 2026, a gap of 0.027.
This project has not established a leaderboard-leading system or clinical readiness.
Detailed results and limitations are in [RESULTS.md](RESULTS.md), rather than hidden
behind an unsupported claim of success.

## About this portfolio

This **public repository is a hiring-oriented case study**, not a training kit or
an open-source reproduction package. The current snapshot contains high-level methods,
aggregate results, selected genuine visuals, and attribution—not model implementations,
weights, exact feature recipes, or runnable research notebooks. Research continues in AWS.
Earlier published commits are retained and remain accessible; this change does not erase
historically published code. No repository is made private and no history is rewritten.

[Rights](RIGHTS.md) · [Attribution](ATTRIBUTION.md)
