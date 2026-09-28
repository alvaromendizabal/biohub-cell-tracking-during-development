# 3D Cell Tracking & Lineage Analysis

**Alvaro Mendizabal · Machine Learning Engineer**  
[GitHub profile](https://github.com/alvaromendizabal)

### Computer vision. Graph learning. Evidence-led ML engineering.

I built an AWS research system for tracking cells through three-dimensional time-lapse microscopy: reproducing a scored external benchmark, engineering original representations, training structured temporal models, reconciling proxy and native graph metrics, integrating prior experiments back into the strongest full pipeline, and hardening the saved-notebook delivery path.

**Verified Kaggle public score: 0.947.** Official submission 56376695 reproduced the public Harmonic Fusion inference reference. The original pretrained systems and public competition baselines are credited to their authors; the research orchestration, original representation work, controlled experiments, transfer studies, source-preserving integration, metric reconciliation, error analysis, and AWS execution infrastructure described here are my project contributions.

[Technical case study](CASE_STUDY.md) · [Results and evidence](RESULTS.md) · [Research frontier](FRONTIER.md) · [Reproducibility boundary](REPRODUCIBILITY.md) · [Visual portfolio](notebooks/portfolio.ipynb)

## What I built and demonstrated

| Capability | Evidence from the project |
|---|---|
| Representation engineering | Evaluated 557 association features and progressed from hand-designed graph corrections to grouped temporal, mixture-of-experts, dense-parental, and image-conditioned representations. |
| Structured prediction | Built source-preserving association, grouped tracklet, graph-fusion, parental-competition, and topology-gated systems around a frozen strong baseline. |
| Baseline-locked integration | Replayed prior candidate families against the original Harmonic development control instead of judging them only against weaker component comparators. |
| Metric-faithful evaluation | Reconciled proxy and native tracking metrics, froze candidate graphs before annotation access, and used full-graph promotion gates. |
| Delivery engineering | Removed repeated validation/reselection work, verified exact motion-linker parity, externalized private inference assets, and added idempotent launch/submission guards. |
| Cloud ML engineering | Established AWS-first execution with immutable caches, resumable checkpoints, artifact hashes, failure recovery, bounded compute, and persistent notebook evidence. |
| Technical judgment | Closed attractive model families when extra true links were outweighed by false links or false divisions, and rejected an indiscriminate all-changes ensemble when it damaged the graph. |

## Latest verified research result

The strongest recent result comes from integrating completed research back into the **original Harmonic development control**, not into a replacement tracker.

Thirty-seven saved candidate policies and two controlled combinations were compared under the same frozen full-graph evaluator. The selected source-preserving candidate improved the local exact score from **0.933046 to 0.948059**, a gain of **+0.015014**. It increased correct edges from **2,391 to 2,392**, reduced false edges from **111 to 110**, increased correct divisions from **1 to 2**, and did not add a false division.

This is **retrospective local development evidence**, not a new Kaggle score. The development cohort has been inspected repeatedly, so the official public score remains **0.947** until a new saved-notebook submission is actually scored.

An all-conflict-free combination of historical changes performed substantially worse than the original control. The lesson was not “ensemble everything”; it was to preserve the strong source graph and apply only compatible, independently validated changes.

## Delivery performance

The deployment work also targeted notebook latency without changing the selected tracking policy:

- the optimized motion-linking component matched the original selected edges and attributes exactly while reducing summed component time by about **34.7×** in the measured AWS comparison;
- repeated validation inference and post-processing reselection were removed from the deployment path because the scored configuration is already frozen;
- the private inference head and runtime are carried as hash-checked private inputs instead of embedding a multi-megabyte payload directly in the notebook;
- notebook launch, remote completion, output verification, and submission are tracked as separate idempotent states to avoid duplicate pushes or submissions.

The 34.7× figure is a **component benchmark**, not a whole-notebook speedup claim. Complete remote Kaggle runtime remains unmeasured.

## Research progression

1. **Reproduce the benchmark.** The external Harmonic Fusion reference was reproduced and scored at 0.947.
2. **Test richer associations.** Hundreds of original and public features were evaluated against frozen predictions.
3. **Attribute exact failures.** Full graph evaluation exposed continuation-link errors, missed divisions, and detector/matching limits.
4. **Move from heuristics to learned context.** Grouped tracklet, mixture-of-experts, dense-parental, and image-conditioned variants were tested with explicit kill gates.
5. **Reconcile the evaluator.** Proxy improvements were checked against the native organizer metric; point-only models that created false forks were closed.
6. **Return to the strongest full system.** Compatible prior work was replayed against the original Harmonic control, producing the current locally qualifying source-preserving candidate.
7. **Harden delivery.** The inference path was optimized and converted into a guarded private saved-notebook workflow.
8. **Current frontier.** Finish private notebook execution and obtain the candidate's actual Kaggle score before starting another modeling branch.

## Evidence discipline

Development cohorts have been examined repeatedly, so local improvements are treated as engineering evidence rather than independent proof of leaderboard generalization. Official competition claims remain limited to scored submissions. Negative experiments remain part of the record because they narrow the search space and prevent repeated spending on non-improving directions.

## About this portfolio

This **public repository is a hiring-oriented case study and semi-reproducible research record**, not a training kit or an open-source competition implementation. It exposes the research sequence, validation contracts, aggregate results, public attribution, and presentation checks while intentionally withholding working datasets, model weights, exact feature recipes, tuned thresholds, private AWS state, candidate graph files, and executable submission logic.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the public reproducibility contract.

[Rights](RIGHTS.md) · [Attribution](ATTRIBUTION.md)
