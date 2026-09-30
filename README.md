# 3D Cell Tracking & Lineage Analysis

**Alvaro Mendizabal · Machine Learning Engineer**  
[GitHub profile](https://github.com/alvaromendizabal)

### Computer vision. Graph learning. Evidence-led ML engineering.

I built an AWS-centered research system for tracking cells through 3D time-lapse microscopy: reproducing scored public references, engineering temporal and graph representations, reconciling proxy and native metrics, hardening prediction-time delivery, and using negative results to identify which layers actually generalized.

**Verified Kaggle public score: 0.947.** Submission **56376695** is the scored Harmonic Fusion reference reproduction. A later locally qualified source-preserving candidate officially scored **0.946**, demonstrating that a strong retrospective development gain did not generalize. A final exact reproduction of a public notebook advertised at 0.953 was accepted as submission **56687425** and remained **pending** at the latest captured evidence cutoff, so this repository does not count 0.953 as my verified official score.

[Technical case study](CASE_STUDY.md) · [Results and evidence](RESULTS.md) · [Research frontier](FRONTIER.md) · [Reproducibility boundary](REPRODUCIBILITY.md) · [Visual portfolio](notebooks/portfolio.ipynb)

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

That candidate was then submitted and scored **0.946**, below the unchanged **0.947** reference. This is the most important scientific result of the final sprint: the development cohort had been inspected too heavily to serve as an unbiased generalization estimate.

A prediction-only fidelity audit then compared the scored-reference output with the failed candidate. The candidate preserved the same preview detections and coordinates and differed through **eight association-edge edits** across four movies. This substantially weakened the hypothesis that the official regression came from accidental detector or coordinate drift. Because hidden labels are unavailable, the audit is descriptive rather than a causal proof, but it redirected the project away from more local graph micro-tuning.

## Public 0.953 reproduction and diversity audit

The project also audited public notebooks advertised at **0.953**. One exact saved notebook version was source/output/version hash-gated and accepted by Kaggle as submission **56687425**. Its official score was still pending at the latest captured evidence cutoff.

Two other public 0.953 notebook lineages were then audited for hidden-test prediction diversity. Their complete saved outputs were byte-for-byte identical to the first public 0.953 output: **238,260 rows, 121,219 nodes, 117,041 edges**, with the same SHA-256 beginning **d52a5d…**. They therefore provided no useful independent ensemble diversity, and no duplicate submissions were burned.

A separate source audit showed that the public 0.953 lineage contains more than a small coordinate-refinement head; it also includes additional association/relinking behavior. The small refinement artifact itself was inspected safely as a **224→32→3** MLP with normalization tensors, but no claim is made that the head alone explains the public-score difference.

## Higher-order association frontier

The final structurally different branch evaluated the public HOCT higher-order tracking system while freezing detections. The pinned model/runtime reached real GPU inference, but global decoding and then a downstream graph-API integration issue prevented a native-metric result before the evidence cutoff. That branch is therefore recorded as **unfinished engineering work, not a negative model result**.

## Delivery performance

The productionization work retained earlier verified improvements:

- a parity-checked motion-linking component reproduced selected edges and attributes while reducing measured component time by about **34.7×**;
- repeated validation/reselection was removed from the frozen delivery path;
- public and private assets were hash-checked instead of silently substituted;
- remote launch, output verification, and competition submission were treated as separate idempotent states.

The 34.7× figure is a **component benchmark**, not a whole-notebook speedup claim.

## Research progression

1. **Reproduce a scored benchmark.** Established the 0.947 Harmonic Fusion reference.
2. **Test richer associations.** Evaluated hundreds of original and public association features.
3. **Attribute exact failures.** Measured edge, division, and topology failure modes.
4. **Move from heuristics to learned context.** Tested grouped tracklets, MOE ranking, dense parental competition, and image-conditioned variants.
5. **Reconcile evaluation.** Used organizer-native full-graph evidence to close proxy-improving but graph-worsening branches.
6. **Return completed work to the strongest baseline.** Produced the locally qualifying +0.015 source-preserving candidate.
7. **Test the official result.** The candidate scored 0.946, invalidating the local promotion as generalization evidence.
8. **Audit fidelity and public frontier.** Ruled out preview detector drift, reproduced the public 0.953 output lineage, and showed its public variants were prediction-identical.
9. **Explore a structurally different linker.** Brought HOCT to real GPU inference while preserving detections; final metric scoring remained unfinished at cutoff.

## Evidence discipline

The official 0.946 regression is treated as stronger evidence than the retrospective 0.948059 development score. Diagnostic cohorts that overlap public pretraining are explicitly labeled contaminated/in-sample rather than clean validation. Passing unit tests, successful packaging, and local graph improvements are never presented as leaderboard improvements.

## About this portfolio

This **public repository is a hiring-oriented case study and semi-reproducible research record**, not a turnkey competition solution. It exposes the research sequence, validation contracts, aggregate evidence, negative-result discipline, attribution, and presentation checks while intentionally withholding working datasets, model weights, exact feature recipes, tuned thresholds, private AWS state, candidate graph files, and executable submission logic.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the public reproducibility contract.

[Rights](RIGHTS.md) · [Attribution](ATTRIBUTION.md)
