# Publication boundary

This repository is an **employer-facing, semi-reproducible ML engineering portfolio**.

## Included

GitHub contains reviewed first-party source, synthetic/unit tests, selected experiment modules, executed notebooks, compact result/provenance receipts, experiment configs, CI, reproduction utilities, and third-party attribution.

The public artifact is intentionally sufficient for technical review without mirroring private infrastructure or licensed assets.

## Intentionally excluded

- competition datasets;
- third-party model weights;
- large predictions, caches, and checkpoints;
- virtual environments;
- credentials, tokens, and AWS account state;
- tuned private thresholds and private candidate graphs;
- private orchestration/submission machinery.

External dependencies are referenced through source/version identifiers and checksums in `reproducibility/external_assets.json` where appropriate.

## Review principle

A portfolio reviewer should be able to determine:

1. what first-party work was implemented;
2. what evidence supports the claims;
3. which components came from third parties;
4. what can be tested from a fresh clone;
5. what remains intentionally private.

No credential, AWS key, external-service token, or private key belongs in this repository.

Third-party rights and notices are preserved. Original project rights remain reserved unless a file states otherwise.
