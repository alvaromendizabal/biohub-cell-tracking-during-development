# Reproducibility contract

## Scope

This repository is a **semi-reproducible ML research portfolio**. A fresh clone contains the first-party Python package, tests, selected experiment modules, configs, compact evidence receipts, and executed notebooks needed to inspect the public research logic without private cloud state.

External assets remain at their lawful source rather than being mirrored into Git.

## Fresh-clone contract

A clean checkout should be able to:

1. create a Python 3.11–3.13 environment;
2. install `requirements-test.txt` and the local package;
3. run `scripts/check_showcase.py`;
4. run `scripts/verify_portfolio.py`;
5. run `scripts/run_ci.py`;
6. run `scripts/reproduce_final_evidence.py self-test`;
7. inspect the seven executed evidence notebooks without AWS access.

The GitHub Actions workflow exercises the package, integrity checks, evidence tooling, and synthetic/unit suite from a fresh checkout.

## What Git contains

- `src/biohub_tracking/` — reusable first-party package;
- `tests/` — synthetic/unit tests;
- `research/` — selected executable experiment modules;
- `configs/` — versioned feature/experiment configuration;
- `notebooks/` — seven executed evidence notebooks plus a presentation notebook;
- `reports/` — compact result/provenance/run receipts;
- `docs/` — engineering overview, model card, experiment ledger, status, and reproduction notes;
- `scripts/` — CI, integrity, and evidence utilities.

## External assets

Competition data, third-party model weights, and selected public outputs are not vendored. Where reproduction depends on them, `reproducibility/external_assets.json` records immutable source references and checksums.

Reproduction tooling fails closed on relevant hash mismatch rather than silently accepting a different asset.

## Evidence tiers

**Independent scored evidence** is the strongest available external check.

**Retrospective development evidence** supports controlled comparisons but is not treated as unbiased generalization evidence.

**Diagnostic evidence** includes source/output/fidelity audits without hidden labels.

**Engineering evidence** includes environment, hash, runtime, packaging, parity, and failure-path tests.

The repository keeps these categories separate rather than collapsing them into a single performance claim.

## Data, weights, and private infrastructure

Reproducibility does not require GitHub to become a mirror of licensed data or private infrastructure. The public repository therefore excludes:

- competition datasets;
- third-party pretrained weights;
- credentials and tokens;
- AWS account identifiers/state;
- large predictions, caches, and checkpoints;
- tuned private thresholds and private candidate graphs;
- private submission/orchestration machinery.

## Integrity philosophy

The public artifact is designed to answer three questions for another engineer:

1. **What was actually built?** — package code, models, experiments, notebooks.
2. **What evidence supports each claim?** — receipts, metrics, provenance, saved outputs.
3. **Can the public checks run from a clean environment?** — CI and local reproduction commands.
