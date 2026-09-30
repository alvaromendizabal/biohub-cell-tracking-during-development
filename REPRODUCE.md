# Reproduce this project from a fresh clone

This archive is designed so a reviewer can clone it, install the first-party package, run all data-free tests, inspect the executed evidence notebooks, and reproduce the final saved-output audits without the deleted SageMaker Space.

## 1. Clone and create the environment

Use Python 3.12 if available; the package supports Python 3.11–3.13.

```bash
git clone https://github.com/alvaromendizabal/biohub-cell-tracking-during-development.git
cd biohub-cell-tracking-during-development
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-test.txt
python -m pip install -e .
```

## 2. Run the clone-level validation

```bash
python scripts/check_showcase.py
python scripts/verify_portfolio.py
python scripts/run_ci.py
python scripts/reproduce_final_evidence.py self-test
```

These checks use synthetic fixtures and committed evidence only. They do not contact AWS or Kaggle and do not require a GPU.

## 3. Inspect the executed evidence notebooks

Seven executed research notebooks are committed under `notebooks/` in addition to the presentation notebook. The research notebooks retain execution counts and saved evidence figures.

## 4. Reproduce the final regression-fidelity audit

Obtain the two saved prediction CSVs identified in `reproducibility/external_assets.json`:

- the exact 0.947 reference output;
- the submitted 0.946 candidate output.

Then run:

```bash
python scripts/reproduce_final_evidence.py compare path/to/reference.csv path/to/candidate.csv
```

The script reports file hashes, row/node/edge counts, added/removed detections, shared-node coordinate drift, and edge-set differences. It does not use hidden labels.

## 5. Reproduce the public-0.953 diversity audit

Install/authenticate the Kaggle CLI yourself and accept the competition rules. Download the saved `submission.csv` output from these exact notebook versions:

- `anvithpothula/biohub-0-953-lb-original/1`;
- `kunaldesale2408/biohub-cell-tracking/10`;
- `raunakdey07/biohub-harmonic-fusion-v3/5`.

Then run:

```bash
python scripts/reproduce_final_evidence.py duplicate   path/to/anvith_submission.csv   path/to/kunal_submission.csv   path/to/raunak_submission.csv
```

The expected saved-output SHA-256 is recorded in the external-asset manifest. A mismatch should be treated as version drift, not silently accepted.

## 6. Reacquire third-party source/weights

`reproducibility/external_assets.json` pins the organizer and HOCT Git commits plus released HOCT model hashes. Clone those repositories at the listed commits and verify downloaded weights before use.

Competition datasets and public model assets remain at their original distribution points; they are not mirrored into Git.

## 7. What a fresh clone can and cannot reproduce

A fresh clone can reproduce the first-party package/tests, synthetic research logic, saved notebook evidence, publication integrity checks, final CSV fidelity audit, and public-output diversity audit.

Real-data reruns additionally require lawful access to the Biohub competition data and referenced third-party model assets. That dependency is explicit and versioned rather than hidden inside the retired AWS volume.
