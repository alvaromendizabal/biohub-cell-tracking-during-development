# Reproducibility contract

## Scope

This repository publishes a **runnable ML engineering portfolio with archived research evidence**. A fresh clone contains the first-party Python package, tests, selected experiment modules, configs, compact evidence receipts, and executed notebooks needed to inspect the public research logic without private cloud state.

External assets remain at their lawful source rather than being mirrored into Git.

## Fresh-clone contract

A clean checkout should be able to:

1. create a Python 3.12 review environment;
2. install `requirements-test.lock` with `--require-hashes`, then the local package with `--no-deps --no-build-isolation`;
3. run `scripts/check_showcase.py`;
4. run `scripts/verify_portfolio.py`;
5. run `scripts/run_ci.py`;
6. run `scripts/reproduce_final_evidence.py self-test`;
7. generate the synthetic tracking demo and open its interactive viewer;
8. inspect the seven executed evidence notebooks without AWS access.

The GitHub Actions workflow exercises the package, integrity checks, evidence tooling, and synthetic/unit suite from a fresh checkout.

## What Git contains

- `src/biohub_tracking/` — reusable first-party package;
- `tests/` — synthetic/unit tests;
- `research/` — selected executable experiment modules;
- `configs/` — versioned feature/experiment configuration;
- `notebooks/` — six archived research notebooks, one executed overview, and one presentation-only notebook;
- `reports/` — compact result/provenance/run receipts;
- `docs/` — engineering overview, model card, experiment ledger, status, and reproduction notes;
- `scripts/` — CI, integrity, and evidence utilities;
- `public-demo/` — buildless lineage viewer and authored synthetic solver outputs.

## External assets

Competition data, third-party model weights, and selected public outputs are not vendored. Where reproduction depends on them, `reproducibility/external_assets.json` records immutable source references and checksums.

The CSV evidence utility reports hashes and validates input structure. Compare those hashes with the source pins before treating an input as the historical artifact; the generic comparison command does not enforce manifest hashes or authenticate external scores.

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

## Archived execution limits


The overview reads committed summaries and images. The six research notebooks are
archived AWS executions: their original private workspace references describe the
historical environment. Their outputs remain unchanged. `portfolio.ipynb` is a
separate presentation-only notebook and is not counted as an executed experiment.

Saved outputs support inspection. They do not mean that a fresh clone can rerun
all original cells without the missing private inputs and runtime.

## Test coverage

The lightweight suite checks selected feature/graph behavior, input contracts,
publication scope, Python syntax, and notebook output integrity. Optional neural
components require additional dependencies; CI does not train them or repeat
competition inference. The numerical-recovery evidence records a finite smoke test
followed by a nonfinite full training attempt, not complete training reproduction.

## External and private inputs

Immutable references in [external_assets.json](reproducibility/external_assets.json)
identify public sources and assets. Access and redistribution terms still apply.
Private predictions, candidate graphs, exact tuned settings, cached tensors, and
operational adapters are intentionally excluded. Their recovery or availability
has not been verified by this publication update.

The original temporal replay worker depended on a private reference adapter; its
public core algorithm and tests do not replace that integration. The archived
image-appearance worker was blocked at its coordinate contract and is not a
completed biological result.


## Component reproduction matrix

| Method / component | Reproduction status in this clone | External requirement |
|---|---|---|
| Retained harmonic-fusion system, historical score 0.947 | Source/evidence contract and historical official receipt preserved | Public notebook/model assets, competition access, and original runtime integration |
| Original feature research | Source, configs, tests, and reports restored | Synthetic tests require no competition data |
| Organizer architecture adaptation | Source and finite-smoke evidence retained; full attempt was nonfinite | Optional neural dependencies and data; no complete training reproduction |
| Exact error-budget / temporal / division studies | Research modules, reports, figures, and executed notebooks restored | Saved real-data reruns require competition inputs |
| Official 0.946 regression fidelity audit | Reproducible from the two saved submission CSVs via `reproduce_final_evidence.py compare` | User-supplied/downloaded CSVs |
| Cross-system prediction-diversity audit | Reproducible from exact saved outputs via `reproduce_final_evidence.py duplicate` | Kaggle public notebook output access |
| V1284 source/head audit | Immutable public notebook/output/head identifiers recorded | Public notebook/dataset access |
| HOCT frozen-detection frontier | Upstream commit/model hashes and experimental contract recorded | HOCT weights + Biohub images/detections |
| Audited public output variants | Anvith v1, Kunal v10, Raunak v5 identifiers recorded | Kaggle output access |

The clone contains the selected public code, tests, configs, reports, and saved notebooks. It does not include every private operational adapter or reproduce full competition inference. External data and model binaries must be acquired under their source terms.
