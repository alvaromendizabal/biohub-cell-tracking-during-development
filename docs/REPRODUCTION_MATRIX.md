# Reproduction matrix

| Method / component | Reproduction status in this clone | External requirement |
|---|---|---|
| Harmonic 0.947 reference evidence | Source/evidence contract and official receipt preserved | Public notebook/model assets and competition access |
| Original feature research | Source, configs, tests, and reports restored | Synthetic tests require no competition data |
| Organizer architecture adaptation | Project-native source and smoke tests restored | Full training data/weights not vendored |
| Exact error-budget / temporal / division studies | Research modules, reports, figures, and executed notebooks restored | Saved real-data reruns require competition inputs |
| Official 0.946 regression fidelity audit | Reproducible from the two saved submission CSVs via `reproduce_final_evidence.py compare` | User-supplied/downloaded CSVs |
| Public 0.953 diversity audit | Reproducible from exact saved outputs via `reproduce_final_evidence.py duplicate` | Kaggle public notebook output access |
| V1284 source/head audit | Immutable public notebook/output/head identifiers recorded | Public notebook/dataset access |
| HOCT frozen-detection frontier | Upstream commit/model hashes and experimental contract recorded | HOCT weights + Biohub images/detections |
| Public 0.953 variants | Anvith v1, Kunal v10, Raunak v5 identifiers recorded | Kaggle output access |

The clone is self-contained for first-party code, tests, configs, reports, and executed notebooks. Licensed competition data and third-party model binaries are reacquired from the original source rather than duplicated in Git.
