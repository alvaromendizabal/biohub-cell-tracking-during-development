# Attribution and provenance

## Competition organizer code

The organizer evaluation and reference architecture work comes from [RoyerLab / Kaggle Cell Tracking Competition](https://github.com/royerlab/kaggle-cell-tracking-competition). The project pinned organizer-derived evaluation code to commit 075fc5f5a52d11077f9dc2b074644618f26939e2 during the research. Copyright and licensing remain with the upstream authors under their published terms.

The public portfolio does not vendor the active organizer research tree.

## Scored Harmonic reference

The official 0.947 result is a reproduction of the public **Biohub Harmonic Fusion** inference system, including external TemporalUNet3D/SimpleNodeTransformer weights, dual-seed/forward-reverse fusion, and DeepCenter-style prior logic. Attribution belongs to the original public notebook and checkpoint authors, including the pilkwang releases.

The project's scored reproduction is publicly visible on Kaggle under the user's Biohub notebook history. The external model assets are not copied into this repository.

## Public 0.953 notebooks

The final-day audit examined public saved notebooks by:

- anvithpothula/biohub-0-953-lb-original;
- raunakdey07/biohub-harmonic-fusion-v3;
- kunaldesale2408/biohub-cell-tracking.

Their audited saved prediction outputs were byte-identical. The portfolio records the diversity result and provenance boundary but does not redistribute the saved competition outputs or claim authorship of the public systems.

## HOCT

The higher-order association experiment used [RoyerLab HOCT](https://github.com/royerlab/hoct), pinned during the deadline sprint to commit 8709ee9d3c4d7aae1f022b259d48dc6584237b02, with upstream released pretrained weights and their published checksums.

The HOCT architecture, implementation, and weights remain third-party work. The public portfolio claims only the evaluation/integration methodology built around them.

## Original research and engineering

Original project contributions include representation engineering, matched-control experiments, full-graph integration, fidelity auditing, output-diversity analysis, AWS execution, resource gating, caching/checkpoint design, notebook persistence checks, and the employer-facing evidence record.

External libraries retain their own licenses. Private source provenance, exact paths, model assets, and executable competition machinery are not redistributed.
