# From detections to coherent cell lineages.

**Alvaro Mendizabal · 3D vision / graph association / scientific ML**

A microscopy-tracking project spanning detection, temporal association, division modeling, graph evaluation, and fidelity audits. The public explorer makes graph decisions inspectable on authored synthetic data.

[Project overview](../README.md) · [Printable inspection brief](index.html) · [Review configuration](project.json)

## Review the engineering, not just the headline

| Focus | Decision to inspect | Follow the proof |
|---|---|---|
| Applied AI | **Solve association as a constrained graph problem.** Expose candidate links, lineage structure, and division behavior instead of hiding decisions behind a single score. | [Code](../src/biohub_tracking/synthetic_demo.py) · [Tests](../tests/test_synthetic_demo.py) · [Evidence](../REPRODUCE.md) |
| Data science | **Screen features without inflating evidence.** Keep feature candidates, measurements, and recorded selection outcomes linked to inspectable research reports. | [Code](../src/biohub_tracking/feature_round2.py) · [Tests](../tests/test_feature_round2.py) · [Evidence](../reports/portfolio_summary.json) |
| ML engineering | **Preserve graph identity during optimization.** Audit selected links and graph fidelity so runtime changes can be separated from altered detections or coordinates. | [Code](../src/biohub_tracking/public_graph_methods.py) · [Tests](../tests/test_evidence_inputs.py) · [Evidence](../RESULTS.md) |
| Data science | **Keep external transfer failures visible.** Retain the stronger system when a locally attractive candidate fails to transfer to external evaluation. | [Code](../scripts/reproduce_final_evidence.py) · [Tests](../tests/test_publication.py) · [Evidence](../reports/official_submission.json) |

The linked tests are review entry points, not a claim that this compact command executes the full research suite. The project’s canonical reproducibility guide describes the broader environment and tests.

## One command, an inspectable result

From a checkout with Python 3.11 or newer:

```bash
python3 -S review/run.py --output /tmp/ml-review
```

Open `/tmp/ml-review/index.html`. The directory contains an accessible HTML brief, `receipt.json`, `manifest.json`, and logs for each executed command. No Python package installation, model download, cloud credentials, or external dataset is required. On Windows, substitute a writable directory outside the checkout for `/tmp/ml-review`.

To execute the browser engine checks as well, install Node.js and use a separate output directory:

```bash
python3 -S review/run.py --output /tmp/ml-review-node --with-node
python3 -S review/run.py --output /tmp/ml-review-node --with-node --check
```

The default path does not execute Node checks and never labels them as passed. CI runs both the Python and browser checks on Python 3.11 and 3.13. It retains the exact review artifacts with the workflow run.

| Check | Runtime | Inspected entry point |
|---|---|---|
| `source-integrity` | python | [`scripts/verify_portfolio.py`](../scripts/verify_portfolio.py) |
| `evidence-selftest` | python | [`scripts/reproduce_final_evidence.py`](../scripts/reproduce_final_evidence.py) |
| `graph-contract` | node | [`scripts/test_live_graph.cjs`](../scripts/test_live_graph.cjs) |
| `lineage-interface` | node | [`tests/test_public_demo_frontend.js`](../tests/test_public_demo_frontend.js) |

## What is actually reproduced

The core review checks archive integrity and the evidence utility self-tests. Optional Node checks execute the public graph engine and interface. Full synthetic microscopy regeneration remains available through the existing pinned Python environment.

A repeat invocation verifies the saved bundle rather than rerunning completed checks. `--check` performs verification only. A changed linked source file, missing evidence, changed artifact, failed command, invalid path, or mismatched check mode causes a nonzero exit. Use a fresh output directory after an intentional source change; stale evidence is never silently relabeled as current.

Each command has a default 90-second deadline, configurable with `--timeout`. The runner records UTC time, interpreter version, elapsed command times, command identities, and SHA-256 hashes for its configuration, template, invoked entry points, linked implementations, tests, and evidence. Failed runs keep diagnostic logs in a separately named failed-run directory and never produce a success receipt.

The fingerprint includes the complete public-demo HTML, CSS, JavaScript and JSON scope declared in `project.json`, as well as the curated review entry points. Additions and removals within that scope invalidate an older receipt. The receipt records the actual Python/Node runtimes and binds the rendered report plus the exact command entry points, arguments and logs. A clean Git checkout produces commit-specific source links; modified checkouts are labeled. This is a declared review scope, **not every transitive dependency or every repository file**. Checksums establish consistency, not independent authentication of historical scores. The receipt records the local bytes inspected, even when a modified checkout requires branch-relative links. The static page is explicitly an inspection brief, not an execution receipt.

## Public/private boundary

Microscopy volumes, competition-derived graphs, private model weights, exact fusion recipes, large intermediate tensors, and operational checkpoints are not added by this release.

This review adds no training result and changes no historical score, selection decision, notebook output, source attribution, or license. A synthetic demonstration, an aggregate evidence check, public reference training, and exact private-model replay are different reproducibility levels.

## Verify the review kit itself

```bash
python3 -B -S -m unittest discover -s review -p test_run.py -v
python3 -B -S -O -m unittest discover -s review -p test_run.py
python3 -S review/run.py --check-page
```

After an intentional configuration or template change, regenerate the static brief with `python3 -S review/run.py --render`, inspect it, then run the checks above. The kit needs no JavaScript or network access to render its HTML. It uses semantic headings, keyboard-visible links, a skip link, responsive layouts, and print styling.
