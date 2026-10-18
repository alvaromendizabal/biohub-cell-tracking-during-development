# Case study · Reliable cell-lineage reconstruction

## Problem and ownership

Nearby detections create ambiguous links, missing frames interrupt tracks, and division creates two-child events. High local association scores do not guarantee a coherent lineage. I integrated model outputs, built fidelity checks, optimized motion linking with parity verification, and developed the public CPU pipeline and interactive explorer. The retained **0.947** system uses credited upstream architectures and pretrained artifacts. [Contribution boundary](ATTRIBUTION.md)

## The final approach

[Volumetric components](src/biohub_tracking/public_reproduction/models.py) describe the credited architecture. The [harmonic kernel](src/biohub_tracking/fusion.py) exposes the retained fusion arithmetic with caller-supplied weights and numerical parity tests. The [constrained solver](src/biohub_tracking/public_graph_methods.py) selects compatible parent/child edges rather than every locally attractive edge.

The original private orchestration, full microscopy inputs and tuned fusion recipe are not shipped as apparently runnable public code. The documented example is genuinely self-contained: it generates synthetic images, candidates and solved graph policies on CPU.

## A decision supported by evidence

A candidate improved the repeatedly inspected development cohort but scored **0.946** externally, so the **0.947** reference stayed frozen. The candidate changed eight association edges across four movies while keeping detections and shared coordinates unchanged. This isolates the changed output component without proving the cause of the regression. The original measurements remain intact. [Results](RESULTS.md)

## Engineering a reviewer can exercise

The [CSV audit](scripts/reproduce_final_evidence.py) rejects duplicate identities, invalid schemas and nonfinite coordinates. Large integer IDs are parsed without lossy float conversion. File hashes and exact node/edge differences make the comparison inspectable. [Failure-path tests](tests/test_evidence_inputs.py)

The [public pipeline](src/biohub_tracking/synthetic_demo.py) and browser share policy constraints. The browser solver is checked against exhaustive small-graph optima and saved Python policies. Gap closing is a separate pass; global optimality of the combined procedure is not claimed. The **34.7×** historical speedup applies only to motion linking with selected-edge parity, not end-to-end inference.

## Reproducibility and limitations

[Run from a clone](REPRODUCE.md), inspect the [executed five-cell notebook](examples/lineage_reasoning.ipynb), or replay browser decisions against synthetic truth. These exercises establish component contracts, not biological accuracy. The historical five-movie cohort covered two previously examined embryos and was not an untouched test. [Asset identities and source credits](reproducibility/external_assets.json) preserve the distinction between upstream models and this project's integration work.
