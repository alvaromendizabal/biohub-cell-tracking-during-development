# Engineering overview

## Purpose

This document is the fastest technical review path for engineers and hiring managers who want to understand the system design behind the portfolio without reading every experiment log.

## End-to-end responsibility

The project spans the ML lifecycle rather than a single training script:

**data/provenance → detection → candidate associations → temporal/graph models → lineage constraints → native evaluation → error attribution → evidence packaging → CI**

I owned the integration and research workflow across those layers, while clearly attributing third-party reference architectures, checkpoints, notebooks, and organizer metric code.

## Modeling surfaces

### Volumetric detection

`src/biohub_tracking/models/temporal_unet3d.py` provides a compact 3D U-Net-style temporal component using 3D convolutions, instance normalization, SiLU activations, and skip connections.

### Candidate-set modeling

`src/biohub_tracking/models/candidate_set_transformer.py` implements a Transformer encoder over candidate association sets. This representation allows candidates to be scored in context rather than independently.

### Feature and graph research

The package and `research/` modules cover geometric, temporal, graph, association, division, and error-analysis logic. A 557-feature study tested whether representation expansion improved tracking before further increasing model complexity.

## Evaluation architecture

The project deliberately separates four layers of evidence:

1. **Independent scored evidence** — external check of end-to-end behavior.
2. **Retrospective development evidence** — controlled experiments on known cohorts.
3. **Diagnostic evidence** — graph diffs, prediction fidelity, provenance, source/output identity.
4. **Engineering evidence** — tests, runtime, parity, packaging, and failure handling.

This matters because sparse cell annotations and repeatedly inspected embryos can make convenient local metrics look more reliable than they are.

## Error attribution

Instead of treating a metric change as a single opaque outcome, the project decomposes errors into:

- correct / incorrect / missed associations;
- division true positives, false positives, and misses;
- node/detection preservation;
- coordinate stability;
- invalid or excessive forks;
- movie/embryo-level stability;
- exact edge-set differences between candidate graphs.

That decomposition enabled a key regression audit: a candidate that looked stronger locally preserved detections and coordinates but changed only **eight association edges across four movies**. The diagnosis therefore focused on association generalization rather than accidental detector drift.

## Runtime and delivery engineering

The AWS-centered research workflow used:

- bounded GPU/CPU runs;
- resumable caches and checkpoints;
- immutable artifact hashes;
- isolated runtime layers;
- stage-level evidence receipts;
- failure-path packaging;
- persistent notebook outputs;
- source/output/version verification;
- duplicate-action guards.

A parity-checked motion-linking optimization reduced one measured component from about 80.5 seconds to 2.32 seconds, approximately **34.7×**, while preserving selected edges.

## Repository quality gates

The public repository is designed to fail closed on common portfolio problems:

- missing required employer-facing artifacts;
- broken internal Markdown links;
- accidental credentials or private key material;
- vendored model/checkpoint/archive binaries that should not be public;
- unexecuted evidence notebooks;
- corrupted saved PNG outputs;
- syntax-invalid Python;
- drift in core evidence contracts.

GitHub Actions installs the package from a clean checkout and runs the public integrity and synthetic/unit suites.

## What to inspect in code review

| Review goal | Suggested path |
|---|---|
| Model architecture | `src/biohub_tracking/models/` |
| Evaluation / sparse supervision | `src/biohub_tracking/annotation_eval.py`, `labeled_eval.py` |
| Feature engineering | `src/biohub_tracking/feature_*` |
| Controlled research modules | `research/` |
| Exact error analysis | `research/exact_error_budget/`, notebook 28 |
| Temporal graph work | `research/temporal_reassignment/`, notebook 27 |
| Test strategy | `tests/`, `scripts/run_ci.py` |
| Provenance / external pins | `reproducibility/external_assets.json`, `reports/source_provenance.json` |
| Portfolio integrity | `scripts/check_showcase.py`, `.github/workflows/portfolio.yml` |

## Engineering principles demonstrated

- Freeze trustworthy controls before experimentation.
- Prefer full-system metrics over proxy-only wins.
- Treat validation independence as a first-class design constraint.
- Keep negative scientific results distinct from failed executions.
- Require parity evidence for performance optimizations.
- Preserve provenance for third-party components and external assets.
- Publish enough for technical review without leaking credentials, licensed data, private infrastructure, or unnecessary tuned implementation details.
