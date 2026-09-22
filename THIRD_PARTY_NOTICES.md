# Attribution and provenance

## Competition organizer code
The files under `research/*/pinned_metric/` are the organizer's evaluation code from
[RoyerLab / Kaggle Cell Tracking Competition](https://github.com/royerlab/kaggle-cell-tracking-competition)
at commit `075fc5f5a52d11077f9dc2b074644618f26939e2`.
Copyright 2026 Thibaut Goldsborough; [BSD 3-Clause](licenses/royerlab-BSD-3-Clause.txt).
The project-native architecture recreations in `src/biohub_tracking/public_reproduction/`
are attributed adaptations of the documented organizer architecture, not a claim of inventing it.

## Scored inference reference
The official 0.947 result is a reproduction of the public **Biohub Harmonic Fusion**
inference system, including external TemporalUNet3D/SimpleNodeTransformer weights,
dual-seed and forward/reverse fusion, and the DeepCenter prior. Attribution belongs
to the public notebook and checkpoint authors, including the `pilkwang` artifact releases.
The exact run is [our scored notebook](https://www.kaggle.com/code/alvaromendizabal/biohub-harmonic-fusion-exact-repro).
The original full notebook and its external model assets are not copied into this publication.
References to Junhao-lineage and Heng Cher Keng/Focus3D public work describe audits,
not independently completed training reproductions or newly claimed leaderboard gains.

## Original research and engineering
Feature implementation, controlled association experiments, error attribution,
checkpoint/failure handling, and the portfolio evidence are the project contributions.
External libraries retain their own licenses. `tracksdata` was pinned in the AWS
runtime to `7523bb39ed9ab1bf88512911fd6c69e46fcad188`; its repository is not vendored.
File-level source provenance and original hashes are in `reports/source_provenance.json`.
