# Biohub · 3D Cell Tracking During Development

**Sparse-supervision tracking, public-reference reproduction, and error-driven model research**  
Alvaro Mendizabal · [GitHub](https://github.com/alvaromendizabal)

[![Portfolio CI](https://github.com/alvaromendizabal/biohub-cell-tracking-during-development/actions/workflows/portfolio.yml/badge.svg)](https://github.com/alvaromendizabal/biohub-cell-tracking-during-development/actions/workflows/portfolio.yml)

## Result at a glance

| Evidence | Result | Interpretation |
|---|---:|---|
| Official public score | **0.947** | Scored reproduction of the external Harmonic Fusion reference; submission **56376695**, notebook version **1** |
| Captured public leader | **0.974** | Snapshot observed September 21, 2026; recorded gap **0.027**, not a live rank |
| Five-movie development baseline | **0.933046** | Separate local combined metric on two repeatedly examined embryos; not comparable as a leaderboard result |
| Latest image-association experiment | **Blocked before fitting** | Coordinate/image-grid bounds check stopped the run; no new model result |

This repository publishes a reviewed research snapshot through the run ending
**2026-09-22 02:05:40 UTC**. The reference reproduction and documented ablations are
complete milestones. A leaderboard-beating integrated challenger has **not** been demonstrated.
See [the score receipt](reports/official_submission.json) and [the current status](docs/STATUS.md).

## Start here

**[Open the executed portfolio overview](notebooks/00_portfolio_overview.ipynb)** for a short technical tour.
It recomputes the recorded gap from committed evidence and displays genuine Plotly
outputs from completed AWS notebooks. No private data or GPU is needed to read it.

For deeper review, follow [the notebook guide](notebooks/README.md): feature ablation,
full-video association tests, exact error attribution, and a learned competing-parent model.

## What the project demonstrates

**Representation engineering.** Project-native morphology, intensity, texture,
geometry, motion, candidate competition, and temporal feature modules accompany
synthetic invariant tests. The anchored comparison evaluated **557 numeric features**
(32 public-score features and 525 original features); more features did not establish
better tracking decisions.

**Tracking and evaluation.** The reference couples temporal 3D detection with learned
node association and structured postprocessing. Subsequent work audits the exact
sparse-annotation edge and division metric, tests graph-preserving temporal exchanges,
and learns competing-parent division transactions across embryos.

**Research judgment.** Negative results remain visible. The daughter-retention rule
was stopped when the saved candidate graphs contained no usable daughter alternatives.
The division model added 108 forks without changing the scored result and was rejected.
The latest image-feature run is documented as blocked, not represented as trained.

**Engineering.** AWS is the canonical workspace for data, checkpoints, and experiments.
This repository is a curated code-and-evidence publication. Hashes, execution receipts,
source attribution, graph invariants, and CPU-only CI make the evidence inspectable
without publishing private caches or rerunning expensive experiments.

## Architecture and evidence boundary

```text
AWS: immutable images + external weights + cached detections / learned associations
                               |
                    frozen scored reference
                               |
        feature / graph hypotheses -> grouped experiments -> exact error analysis
                               |
              curated code + executed notebooks + compact receipts
                               |
GitHub: public research snapshot       Kaggle: versioned notebook submissions
```

The public 0.947 stack and organizer architecture are credited to their original
authors. This project does not claim to have invented those models or independently
reproduced the 0.974 leader's complete training method.
[Attribution](THIRD_PARTY_NOTICES.md) · [reproduction matrix](docs/REPRODUCTION_MATRIX.md)

## Inspect or reproduce

Read the saved notebooks directly, or run the lightweight publication validator:

```bash
python scripts/verify_portfolio.py
```

To run the curated synthetic CPU suite in a separate local environment:

```bash
python -m venv .venv-review
. .venv-review/bin/activate
python -m pip install -r requirements-test.txt
python scripts/run_ci.py
```

CI checks code syntax, feature/graph invariants, saved notebook outputs, evidence
integrity, and publication exclusions. It **does not** retrain the reference,
rerun microscopy inference, repair the blocked experiment, or reproduce Kaggle scoring.
Full experimental reruns require the private data/artifact layout described in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Repository map

| Directory | Purpose |
|---|---|
| `notebooks/` | Executed technical narrative and genuine archived experiment outputs |
| `src/biohub_tracking/` | Project-native feature, model, and evaluation implementation snapshot |
| `research/` | Recent exact-error, temporal, competing-parent, and appearance experiment source |
| `tests/`, `scripts/` | Curated synthetic tests and data-free publication checks |
| `reports/` | Compact result receipts, score evidence, and source hashes |
| `docs/` | Validation, model card, experiment decisions, and reproducibility scope |

Raw microscopy, GEFF chunk stores, virtual environments, fitted weights, predictions,
credentials, and AWS resource identifiers are intentionally not mirrored here.

[Model card](docs/MODEL_CARD.md) · [experiment ledger](docs/EXPERIMENTS.md) ·
[publication scope](docs/PUBLICATION_SCOPE.md) · [limitations](docs/STATUS.md)
