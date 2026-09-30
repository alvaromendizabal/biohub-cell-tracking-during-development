# 3D Cell Tracking & Lineage Analysis

**Alvaro Mendizabal · Machine Learning Engineer**  
[GitHub profile](https://github.com/alvaromendizabal)

### Computer vision. Graph learning. Evidence-led ML engineering.

This repository is now the **reproducible research archive** for my Biohub cell-tracking project. It contains the employer-facing case study **and** the executable Python package, synthetic/unit tests, research modules, experiment configs, compact result receipts, and seven executed evidence notebooks that were previously preserved only in repository history.

**Verified Kaggle public score: 0.947.** Submission **56376695** is the scored Harmonic Fusion reference reproduction. A later locally qualified source-preserving candidate officially scored **0.946**, demonstrating that a strong retrospective development gain did not generalize. A final exact reproduction of a public notebook advertised at 0.953 was accepted as submission **56687425** and remained **pending** at the latest captured evidence cutoff, so this repository does not count 0.953 as my verified official score.

[Reproduce from a fresh clone](REPRODUCE.md) · [Technical case study](CASE_STUDY.md) · [Results](RESULTS.md) · [Research frontier](FRONTIER.md) · [Reproducibility contract](REPRODUCIBILITY.md) · [Visual portfolio](notebooks/portfolio.ipynb)

## What is included

- installable `src/biohub_tracking` package;
- synthetic/unit tests under `tests/`;
- executable research modules under `research/`;
- versioned experiment configs under `configs/`;
- seven executed research notebooks plus the presentation notebook;
- compact saved receipts and result JSONs under `reports/`;
- the final-day fidelity/public-0.953/HOCT reproduction contract;
- an exact **99-file readable source snapshot** extracted from the September 29 final-sprint handoff artifacts, with a safe extractor/self-test;
- CI that installs the package from a fresh checkout, validates the presentation snapshot, runs the synthetic/unit suite, self-tests the final-evidence reproducer, and validates the final-sprint source snapshot.

Large competition datasets and third-party pretrained weights are **not committed as binaries**. They are reacquired from their original public/Kaggle sources using the exact versions/commits/checksums recorded in `reproducibility/external_assets.json`. Kaggle competition data still requires the user to accept the competition rules and authenticate.

## What I built and demonstrated

| Capability | Evidence from the project |
|---|---|
| Representation engineering | Evaluated 557 association features and progressed from graph corrections to grouped temporal, mixture-of-experts, image-conditioned, and higher-order association experiments. |
| Baseline-locked integration | Replayed completed candidate families against the strongest Harmonic development control instead of judging them only against weaker component comparators. |
| Metric-faithful evaluation | Reconciled proxy and organizer-native graph metrics, froze candidate graphs before retrospective annotation access, and separated diagnostic evidence from official leaderboard evidence. |
| Fidelity auditing | Compared hash-bound baseline and candidate outputs and established that the failed 0.946 candidate preserved the preview detection universe and coordinates while changing only a small number of association edges. |
| Delivery engineering | Built hash-gated notebook/output verification, idempotent submission guards, resumable state, isolated GPU/runtime overlays, and bounded failure returns. |
| Cloud ML engineering | Used AWS as the canonical research environment with immutable caches, checkpoint/reuse contracts, resource gates, and persistent notebook evidence. |
| Technical judgment | Closed locally attractive systems when full-graph or official evidence contradicted proxy gains, including a locally +0.015 candidate that regressed officially. |

## Final-day evidence changed the conclusion

Before the official submission, the strongest retrospective candidate improved the five-movie development metric from **0.933046 to 0.948059** (**+0.015014**). It added one correct edge, removed one false edge, recovered one annotated division, and passed the local promotion gate.

That candidate then scored **0.946**, below the unchanged **0.947** reference. This is the most important scientific result of the final sprint: the development cohort had been inspected too heavily to serve as an unbiased generalization estimate.

A prediction-only fidelity audit then compared the scored-reference output with the failed candidate. The candidate preserved the same preview detections and coordinates and differed through **eight association-edge edits** across four movies. Because hidden labels are unavailable, the audit is descriptive rather than causal, but it redirected the project away from further local graph micro-tuning.

## Public 0.953 reproduction and diversity audit

One exact saved notebook version advertised at **0.953** was source/output/version hash-gated and accepted by Kaggle as submission **56687425**. Its official score was still pending at the latest captured evidence cutoff.

Two other public 0.953 notebook lineages were then audited for prediction diversity. Their complete saved outputs were byte-for-byte identical to the first public 0.953 output: **238,260 rows, 121,219 nodes, 117,041 edges**, with the same SHA-256 beginning **d52a5d…**. They therefore provided no independent ensemble diversity.

A separate source audit showed that the public 0.953 lineage contains more than a small coordinate-refinement head; it also includes additional association/relinking behavior. The small refinement artifact itself was inspected safely as a **224→32→3** MLP with normalization tensors, but no claim is made that the head alone explains the public-score difference.

## Higher-order association frontier

The final structurally different branch evaluated the public HOCT higher-order tracking system while freezing detections. The pinned model/runtime reached real GPU inference, but global decoding and then a downstream graph-API integration issue prevented a native-metric result before the evidence cutoff. That branch is recorded as **unfinished engineering work, not a negative model result**.

## Delivery performance

- parity-checked motion linking reproduced selected edges/attributes while reducing measured component time by about **34.7×**;
- repeated validation/reselection was removed from the frozen delivery path;
- public/private assets were hash-checked instead of silently substituted;
- launch, output verification, and competition submission were treated as separate idempotent states.

The 34.7× figure is a **component benchmark**, not a whole-notebook speedup claim.

## Evidence discipline

The official 0.946 regression is treated as stronger evidence than the retrospective 0.948059 development score. Diagnostic cohorts that overlap public pretraining are explicitly labeled contaminated/in-sample rather than clean validation. Passing tests and local graph improvements are never presented as leaderboard improvements.

## Reproducibility and archival status

A fresh clone contains all first-party source needed to install, test, and inspect the published experiments. External competition data and third-party weights are fetched from their original sources because redistributing them in Git would be inappropriate and unnecessarily large.

See [REPRODUCE.md](REPRODUCE.md) for the exact clone-to-tests workflow and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for evidence boundaries.

The original Biohub SageMaker JupyterLab app and persistent Space were deleted after the reproducible archive merged and passed post-merge CI. A final S3 audit found no remaining Biohub objects under the project's known backup prefix; unrelated shared-bucket data was left untouched. See [Decommission record](docs/DECOMMISSION.md).

[Rights](RIGHTS.md) · [Attribution](ATTRIBUTION.md)
