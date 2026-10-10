# Project status

Scientific evidence cutoff: **September 29, 2026, 23:02 UTC**. Public documentation refreshed October 10, 2026; no new experiment or external score is claimed.

## Current public state

- The repository is an archived **employer-facing ML engineering portfolio**, not an active experiment workspace.
- The reusable package, selected research modules, tests, configs, and reports are committed, with **six archived research notebooks and one executed overview**, plus a presentation-only notebook.
- The original AWS workspace has been removed, as confirmed by the owner on October 6. See the [archive record](DECOMMISSION.md).
- The retained tracking system scored **0.947** in historical external evaluation.
- A retrospective candidate improved the inspected five-movie development metric to **0.948059** but scored **0.946** on independent evaluation, establishing a validation/generalization mismatch.
- The prediction-fidelity audit localized the difference to **eight association-edge edits** while preserving the compared detections and coordinates.
- The public CI path validates portfolio integrity, notebook evidence, package syntax/tests, and reproduction utilities from a fresh checkout.

## Research state

Completed branches and negative results are preserved because they changed technical decisions. Unfinished work is labeled as unfinished rather than treated as model-quality evidence.

[`../FRONTIER.md`](../FRONTIER.md) preserves the research directions recorded at the scientific cutoff. It is a historical design discussion; this release adds public demonstration and review tools without claiming new biological experiments.

## Publication boundary

The public project excludes private datasets, credentials, AWS account state, third-party weights, large checkpoints/caches, tuned private thresholds, and private orchestration/submission machinery.
