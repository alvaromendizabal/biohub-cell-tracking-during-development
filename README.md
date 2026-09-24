# 3D Cell Tracking & Lineage Analysis

**Alvaro Mendizabal · Machine Learning Engineer**  
[GitHub profile](https://github.com/alvaromendizabal)

### Computer vision. Graph learning. Evidence-led ML engineering.

I built an AWS research system for tracking cells through three-dimensional time-lapse microscopy: reproducing a scored external benchmark, engineering original representations, testing structured graph models, training image-based division models, fine-tuning on sparse real annotations, and tracing failures to the exact detection, link, and lineage decisions responsible.

**Verified Kaggle public score: 0.947.** Official submission 56376695 reproduced the public Harmonic Fusion inference reference. The original pretrained systems and public competition baselines are credited to their authors; the research orchestration, original feature work, controlled experiments, transfer studies, error analysis, and AWS execution infrastructure described here are my project contributions.

[Technical case study](CASE_STUDY.md) · [Results and evidence](RESULTS.md) · [Current research frontier](FRONTIER.md) · [Visual portfolio](notebooks/portfolio.ipynb)

## What I built and demonstrated

| Capability | Evidence from the project |
|---|---|
| Representation engineering | Evaluated 557 association features: 32 public-score features and 525 original motion, geometry, context, and related representations. |
| Structured prediction | Built graph-preserving association, division, temporal, neural-affinity, detector-consensus, and matched-control experiments. |
| Image-model research | Trained and evaluated synthetic 3D image division models, scale-up variants, domain adaptation, and sparse-real fine-tuning. |
| Exact error attribution | Reconciled 2,391 correct, 111 incorrect, and 112 missed edges across five full development movies; division counts were 1 TP, 1 FP, and 5 FN. |
| Cloud ML engineering | Established AWS-first execution with immutable caches, resumable acquisition, checkpoint integrity, failure recovery, numerical checks, bounded compute, and persistent notebook evidence. |
| Technical judgment | Closed multiple plausible but non-improving research lines instead of promoting proxy-metric wins that failed the full tracking evaluator. |

## Selected technical evidence

![Exact error attribution](assets/exact_error_budget_0.png)

This is a genuine saved figure from the completed error analysis. It shows where the development baseline fails; it is not a projected score improvement.

## Research progression

The project deliberately moved upstream as evidence accumulated.

1. **Reproduce the benchmark.** The external Harmonic Fusion reference was reproduced and scored at 0.947.
2. **Test richer associations.** Hundreds of original and public features were evaluated against frozen baseline predictions.
3. **Attribute exact failures.** Full graph evaluation exposed continuation-link errors and sparse division failures.
4. **Test targeted correction families.** Geometry, neural-affinity, detector, and division post-processing candidates were rejected when the exact metric did not improve.
5. **Change the supervision.** Public synthetic lineage training, 3D image models, scale-up, domain adaptation, and sparse-real fine-tuning were evaluated with held-out gates and matched controls.
6. **Move to the integrated linker.** The active frontier now fine-tunes the actual multi-frame edge linker rather than applying another post-hoc graph patch.

## Current frontier

The active AWS experiment targets the core temporal-linking component directly. It uses the public Biohub baseline architecture lineage—a temporal 3D U-Net feeding a cross-attention node transformer—while keeping the public encoder/detector fixed and fine-tuning the linker on sparse real continuation and division edges.

The workflow uses multi-frame windows, resumable feature caches, held-out validation, source-vs-candidate component comparison, and the same exact full-graph evaluator used throughout the project. The run is still in progress, so **no improved score is claimed**.

See [FRONTIER.md](FRONTIER.md) for the current research boundary and promotion criteria.

## Verified negative results matter

Recent model families produced excellent proxy metrics without improving the tracked graph:

- synthetic graph-only division classification;
- five-frame synthetic image division classification;
- synthetic-to-real domain adaptation;
- 256-sequence image-model scale-up;
- sparse-real division fine-tuning.

Those results narrowed the search space. The project now treats graph-level score movement—not classifier AP alone—as the standard for promotion.

## About this portfolio

This **public repository is a hiring-oriented case study**, not a training kit or an open-source competition implementation. It contains high-level methods, aggregate results, selected genuine visuals, attribution, and research decisions—not working datasets, model weights, exact feature recipes, account state, or the active AWS implementation.

Earlier published commits remain accessible; this presentation snapshot does not rewrite history. No repository is made private.

[Rights](RIGHTS.md) · [Attribution](ATTRIBUTION.md)
