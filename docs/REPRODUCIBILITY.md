# Reproducibility: three distinct levels

## 1. Read the results
The committed notebooks preserve original execution counts and genuine embedded
Plotly PNGs. They are readable on GitHub without AWS access. The overview reads only
committed summaries and images and is data-free; the six numbered research notebooks
are archived AWS executions. Their private workspace paths remain as historical
execution context. A provenance header was added, but the scientific outputs were
not fabricated or relabeled as a new execution.

## 2. Run lightweight checks
Use Python 3.12 and `requirements-test.txt` in a separate environment. Run
`python scripts/run_ci.py`. This validates the publication manifest, notebook PNGs,
source syntax, feature invariants and selected graph/association tests using synthetic
fixtures. It does not require credentials or network after dependencies are installed.

The original 44 source modules are preserved as a project-native implementation
snapshot. The selected feature and graph tests are the CI scope. Optional neural
architecture modules are not exercised by this lightweight suite; full project tests
require additional private-runtime dependencies. A green badge is not a claim that
all historical experiments or external library integrations passed.

## 3. Rerun the research in the canonical AWS workspace
Retain the existing `biohub-cell-tracking-during-development` workspace with its
`.venv`, `.venv-gpu`, licensed competition data, cached graph stores, frozen reference
outputs, checkpoint fingerprints, and original run manifests. No such private bytes
are included here. Research workers use `BIOHUB_PROJECT_ROOT` where supported;
individual archived notebook code records the exact original paths and environment.

The publication copy is **not a drop-in replacement** for the AWS directory. It is
a curated source/evidence view. Reconstructing the private data layout is a separate
operation requiring the competition's access permissions. Full metrics and figure
replays require the original manifests; they are not triggered by CI.

The original temporal replay worker is excluded because it depends on the full
private reference adapter. Its core algorithm and tests remain public.

The current image-appearance worker is experimental and known blocked at
`IMAGE_COORDINATE_BOUNDS`; do not rerun it at scale before repairing that contract.

## Publication safety
The publisher works in a separate `biohub-publication` Git clone, never stages the
AWS workspace, and never force-pushes. It verifies selected live source fingerprints,
creates an initial main-branch scaffold only for the inspected empty remote, opens
a feature-branch pull request, waits for the exact commit's checks, and merges only
that tested head. A concurrent unexpected change stops the workflow.
