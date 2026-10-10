# Run and inspect the project

A fresh clone supports the interactive lineage viewer, a CPU tracking pipeline, package tests and the recorded research evidence. The public demos use authored synthetic data and need no AWS account, private microscopy or pretrained weights.

## Three-minute browser review

[Open the public demo](https://alvaro-cell-lineage-explorer.tartmacaw2.chatgpt.site) without installation. You can also open `public-demo/index.html` directly from a checkout, or serve it locally:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000/public-demo/`. Play or scrub the cell movie and select a cell to inspect its ancestry. Adjust the link-score threshold, one/two-child limit and gap-closing option. The browser solves the candidate graph again and updates the overlays and exact-ID diagnostics. You can also compare the committed Python-solver policies. No model fitting or backend service is involved.

## Install the CPU environment

Use Python 3.12 for the hash-locked review environment.

```bash
git clone https://github.com/alvaromendizabal/biohub-cell-tracking-during-development.git
cd biohub-cell-tracking-during-development
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-test.lock
python -m pip install --no-deps --no-build-isolation -e .
```

## Generate the synthetic tracking demo

```bash
python examples/run_tracking_demo.py --output public-demo
```

The pipeline generates a deterministic twelve-frame microscopy fixture, builds geometric and appearance-based association candidates, and runs the repository's constrained graph solver under several policies. The authored sequence includes divisions, missed observations and false detections so errors remain visible. The default seed is `2026`; `--seed` creates another fixture.

`public-demo/data.json` contains the synthetic frames, candidates, selected graphs, exact-ID diagnostics and provenance. `data.js` supplies the same result to the buildless browser viewer, including when it is opened directly from disk. The frontend compares those computed policies, runs its own constrained graph optimizer for interactive settings, and supports result export. The browser implementation matches the selected edges of all five Python fixture policies; its degree-constrained optimization was also checked against exhaustive optima on 80 small graphs. Adjacent-frame optimization is followed by a separate residual-endpoint gap-closing pass; the combined procedure is not claimed to be a globally optimal temporal tracker.

These diagnostics use known synthetic identities and authored truth. They establish behavior of the public pipeline, not biological accuracy or a new external score. Regenerating demo fixtures changes public files; review the diff before including them in a release.

The smaller five-cell graph example remains available:

```bash
python examples/run_demo.py
```

Its [guide and executed notebook](examples/README.md) show how local parent choices can violate graph constraints, and how a structurally valid solution can still contain incorrect links.

## Run the quality gates

```bash
python scripts/check_showcase.py
python scripts/verify_portfolio.py
python scripts/run_ci.py
python scripts/reproduce_final_evidence.py self-test
node scripts/test_live_graph.cjs
node tests/test_public_demo_frontend.js
```

These commands use synthetic fixtures and committed evidence. They do not contact cloud services or require a GPU. [Engineering coverage](docs/ENGINEERING_OVERVIEW.md)

## Inspect the research evidence

Start with [the executed overview](notebooks/00_portfolio_overview.ipynb) or the [notebook guide](notebooks/README.md). Six archived research notebooks and one executed overview retain the measured findings. The presentation-only notebook is separate from that executed evidence.

The historical retained-system score is **0.947**; a later candidate scored **0.946** externally despite improving the reused development cohort. The new public demo does not change those results. [Detailed results](RESULTS.md)

## Run saved-output audits

`reproducibility/external_assets.json` identifies the historical retained-system and candidate CSVs, along with the public outputs used in the diversity audit. Obtain them through their authorized distribution routes, then run:

```bash
python scripts/reproduce_final_evidence.py compare path/to/reference.csv path/to/candidate.csv
python scripts/reproduce_final_evidence.py duplicate path/to/output1.csv path/to/output2.csv path/to/output3.csv
```

The tools validate structure and finite coordinates, report hashes, compare detections and coordinates, and measure edge-set differences or duplicate outputs. Compare input hashes with the pinned records separately before treating a supplied file as a historical artifact. The tools do not authenticate official scoring receipts.

## Research and asset boundary

The public package, synthetic workflows, tests, evidence checks and CSV utilities can be rerun from this clone. Historical real-data experiments additionally require lawful data/model access and their original runtime integration. Private orchestration, predictions, tuned settings and weights are intentionally absent.

Source revisions and model hashes remain in `reproducibility/external_assets.json`; upstream licensing still applies. The [archive record](docs/DECOMMISSION.md) documents the historical AWS environment. This guide performs no cloud cleanup or account operation. [Detailed execution matrix](docs/REPRODUCTION_MATRIX.md)
