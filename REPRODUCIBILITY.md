# Reproducibility contract

## Scope

This repository is a **reproducible research archive**. A fresh clone contains the first-party Python package, tests, experiment modules, configs, compact evidence receipts, and executed notebooks needed to inspect and rerun the public research logic.

The only non-Git inputs are assets that should remain at their original source:

- Biohub competition data, which requires Kaggle authentication and accepted competition rules;
- third-party pretrained weights;
- public notebook outputs used in the final-day fidelity/diversity audit.

Those external inputs are versioned by source reference and checksum in `reproducibility/external_assets.json`.

## Fresh-clone contract

A clean checkout should be able to:

1. create a Python 3.11–3.13 environment;
2. install `requirements-test.txt` and the local package;
3. run `scripts/check_showcase.py`;
4. run `scripts/verify_portfolio.py`;
5. run `scripts/run_ci.py`;
6. run `scripts/reproduce_final_evidence.py self-test`;
7. inspect the seven executed evidence notebooks without AWS;
8. reproduce final-day CSV fidelity/diversity checks after downloading the referenced public outputs.

The CI workflow exercises steps 2–6 from a fresh GitHub checkout.

## What Git contains

- `src/biohub_tracking/`: reusable project package;
- `tests/`: synthetic/unit tests;
- `research/`: executable experiment modules and their pinned organizer metric copies;
- `configs/`: feature/experiment configuration;
- `notebooks/`: seven executed evidence notebooks plus the presentation notebook;
- `reports/`: compact saved run/result/provenance receipts;
- `docs/`: experiment, model-card, status, reproduction, and final-sprint documentation;
- `scripts/`: CI, validation, and final-evidence reproduction utilities.

## External assets

The asset manifest records immutable identifiers for:

- organizer source commit `075fc5f5a52d11077f9dc2b074644618f26939e2`;
- HOCT source commit `8709ee9d3c4d7aae1f022b259d48dc6584237b02`;
- HOCT `general_v1` and `ctc_v0` released model SHA-256 values;
- the public 0.953 notebook versions used in the diversity audit;
- the expected public 0.953 output SHA-256;
- the scored 0.947 reference-output SHA-256;
- the failed 0.946 candidate-output SHA-256.

Reproduction scripts fail closed on hash mismatch.

## Evidence tiers

**Official evidence** is a scored Kaggle submission.

**Retrospective development evidence** is useful for controlled comparisons but is not treated as unbiased generalization evidence.

**Diagnostic evidence** includes source/output/fidelity audits without hidden labels.

**Engineering evidence** includes environment, hash, runtime, packaging, and failure-path tests.

The repository preserves these categories rather than collapsing them into a single “score.”

## Final-day evidence

The final-day public-output claims can be reproduced from CSVs with:

`python scripts/reproduce_final_evidence.py compare <baseline.csv> <candidate.csv>`

and:

`python scripts/reproduce_final_evidence.py duplicate <public1.csv> <public2.csv> <public3.csv>`

The script reports row/node/edge counts, hashes, detection/coordinate equality, and edge-set differences without requiring hidden labels.

## Data and weights

Competition data and third-party weights are intentionally not vendored. Reproducibility means that the clone contains the code, immutable identifiers, checks, and instructions to reacquire them from their lawful source—not that GitHub becomes a mirror of licensed competition data.

## AWS decommissioning

The canonical SageMaker app was already deleted before this archival milestone. The remaining project Space used a 128-GB EBS volume. Once this branch is merged and CI passes, the Git repository—not that volume—is the durable first-party source archive.

No Biohub-specific S3 bucket or top-level Biohub S3 prefix was identified in the account inventory performed during archival; unrelated project buckets must not be deleted as part of Biohub cleanup.
