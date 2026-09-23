# 3D Cell Tracking & Lineage Analysis

**Alvaro Mendizabal · Machine Learning Engineer**  
[GitHub profile](https://github.com/alvaromendizabal)

### Computer vision. Graph learning. Evidence-led ML engineering.

I built an AWS research system for tracking cells through three-dimensional time-lapse
microscopy: reproducing a scored external benchmark, engineering original representations,
testing structured graph models, and tracing failures to the exact detection, link, and
lineage decisions responsible.

**Verified Kaggle public score: 0.947.** Official submission 56376695 reproduced the
public Harmonic Fusion inference reference. The original pretrained system is credited
to its authors; the engineering, integration studies, original feature work, controlled
experiments, error analysis, and research infrastructure described here are my project
contributions.

[Technical case study](CASE_STUDY.md) · [Results and evidence](RESULTS.md) ·
[Visual portfolio](notebooks/portfolio.ipynb)

## What I built and demonstrated

| Capability | Evidence from the project |
|---|---|
| Representation engineering | Evaluated 557 association features: 32 public-score features and 525 original features spanning motion, geometry, context, and related representations. |
| Structured prediction | Built and evaluated graph-preserving association, division, temporal, neural-affinity, and detector-consensus experiments. |
| Rigorous evaluation | Reconciled 111 false and 112 missed edges across five full development movies, with separate topology-aware division analysis. |
| Cloud ML engineering | Established AWS-first execution with cache reuse, checkpoint integrity, failure recovery, numerical checks, bounded compute, and persistent notebook evidence. |
| Software quality | The original research publication passed 277 synthetic tests and GitHub CI before and after merge. These are engineering checks, not a claim of statistical generalization. |
| Technical judgment | Rejected no-gain and regressive candidates, preserved the stronger benchmark, and separated official scores from exploratory development metrics. |

## Selected technical evidence

![Exact error attribution](assets/exact_error_budget_0.png)

A genuine saved figure from the completed error analysis. It shows where the
development baseline fails; it is not a projected score improvement.

## Research frontier

The benchmark reproduction and documented ablations are completed milestones. Recent
post-hoc correction families did not produce a validated tracking gain, so the active
research lane has moved upstream to **sequence-level supervised division learning**.

The current AWS workflow is ingesting a public fully labelled synthetic lineage resource
with 2,174 time sequences and 165,267 mitotic-parent events. Source capture and a
3,716-file output inventory are verified; training is not represented as complete until
the bounded AWS acquisition, sequence-level split, held-out validation gate, and
full five-movie evaluator all pass.

The last account-side leaderboard capture recorded a public leader score of 0.974 on
September 21, 2026, a gap of 0.027 from the verified 0.947 submission. The repository
does not claim a leaderboard-leading system or clinical readiness. See [RESULTS.md](RESULTS.md)
for the measured outcomes and limitations.

## About this portfolio

This **public repository is a hiring-oriented case study**, not a training kit or an
open-source reproduction package. The current snapshot contains high-level methods,
aggregate results, selected genuine visuals, and attribution—not model implementations,
weights, exact feature recipes, or runnable research notebooks. Research continues in AWS.

Earlier published commits remain accessible; this presentation snapshot does not rewrite
history. No repository is made private.

[Rights](RIGHTS.md) · [Attribution](ATTRIBUTION.md)
