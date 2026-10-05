# Reproduce this project from a fresh clone

This portfolio is designed so a reviewer can clone it, install the first-party package, run all data-free checks, inspect genuine executed evidence notebooks, and reproduce the public fidelity/diversity utilities without access to private AWS state.

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

## 2. Run the public quality gates

```bash
python scripts/check_showcase.py
python scripts/verify_portfolio.py
python scripts/run_ci.py
python scripts/reproduce_final_evidence.py self-test
```

These checks use synthetic fixtures and committed evidence only. They do not contact AWS or external scoring services and do not require a GPU.

## 3. Inspect the executed evidence notebooks

Seven executed research notebooks are committed under `notebooks/` in addition to the presentation notebook. They retain execution counts and saved evidence figures so a reviewer can inspect what actually ran.

Start with [`notebooks/00_portfolio_overview.ipynb`](notebooks/00_portfolio_overview.ipynb) or use the [`notebooks/README.md`](notebooks/README.md) reading guide.

## 4. Reproduce the regression-fidelity audit

Obtain the two saved prediction CSVs identified in `reproducibility/external_assets.json`: the frozen reference output and the independently scored candidate output.

```bash
python scripts/reproduce_final_evidence.py compare path/to/reference.csv path/to/candidate.csv
```

The script reports file hashes, row/node/edge counts, added/removed detections, shared-node coordinate drift, and edge-set differences. It does not require hidden labels.

## 5. Reproduce the prediction-diversity audit

`reproducibility/external_assets.json` also records exact public notebook/output versions used for a duplicate-prediction audit. After obtaining those outputs from their original source, run:

```bash
python scripts/reproduce_final_evidence.py duplicate path/to/output1.csv path/to/output2.csv path/to/output3.csv
```

The tool compares complete output hashes and graph counts so nominally different systems are not assumed to provide ensemble diversity without evidence.

## 6. Reacquire third-party source and weights

`reproducibility/external_assets.json` pins organizer and third-party source revisions plus released model hashes where applicable. Clone those repositories at the listed revisions and verify downloaded assets before use.

Competition datasets and third-party model assets remain at their original distribution points; they are not mirrored into Git.

## 7. What a fresh clone can and cannot reproduce

A fresh clone can reproduce the first-party package/tests, synthetic research logic, saved notebook evidence, portfolio integrity checks, and CSV fidelity/diversity tooling.

Real-data reruns additionally require lawful access to the original dataset and referenced third-party model assets. That dependency is explicit and versioned rather than hidden inside a private cloud workspace.
