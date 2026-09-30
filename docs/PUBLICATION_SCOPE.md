# Publication and archive boundary

This repository now serves two purposes:

1. an employer-facing case study at the repository root; and
2. a reproducible executable research archive under `src/`, `research/`, `tests/`, `configs/`, `reports/`, and `notebooks/`.

The earlier presentation-only publication deliberately withheld the executable tree while the competition was active. After the competition deadline, the tracked executable snapshot was restored from Git history so the project can survive AWS decommissioning.

## Included

GitHub contains reviewed first-party source, synthetic tests, executed notebooks, compact result/provenance receipts, public-source attribution, experiment configs, CI, and reproduction utilities.

## Not mirrored as Git binaries

- Kaggle competition data;
- third-party model weights;
- large predictions/caches/checkpoints;
- virtual environments;
- credentials or AWS account state.

Those inputs remain reproducible through their original source plus the immutable refs/checksums in `reproducibility/external_assets.json`.

No credential, AWS key, Kaggle token, or private key belongs in this archive.

Third-party rights and notices are preserved. Original project rights remain reserved unless a file states otherwise.
