# Research frontier

## Current question

The point-only division branch is closed. Native metric reconciliation showed that dense parental models sometimes recover real division events, but the extra recall arrives with too many incorrect links and false forks.

The active question is now:

**Can image-conditioned temporal evidence identify real division events while preserving the strong source graph everywhere else?**

## Why this is the next layer

The latest native audit evaluated 18 frozen graph systems across six movies. Relative to the raw-source comparator, the learned point-only models added **17 correct edges** and recovered **2 of 10** division events, but also added more than one hundred false links and **89–94 false divisions**.

Event-level probability inspection showed two distinct limitations:

- some true daughters lacked a unique detector-point match, which is a detection/matching limitation;
- in many matched cases, the true mother was available but ranked behind a nearby alternative parent, which is a representation problem.

Point geometry alone is therefore not a sufficient discriminator for the next round.

## Current model direction

The next bounded experiment will compare three frozen-policy systems:

1. the source graph;
2. a geometry-only division-event scorer;
3. an image-conditioned temporal division-event scorer.

The design preserves trusted source associations by default and asks the learned model to intervene only on independently supported division hypotheses. Supervision must be verified before any fit, and candidate graphs remain frozen before retrospective exact scoring.

## Promotion controls

A frontier result is not promoted because training loss falls or because division recall increases in isolation.

The next lane requires:

1. improvement over the geometry-only control;
2. additional correct divisions without losing existing correct edges;
3. no false-division explosion;
4. no embryo-level regression;
5. valid lineage topology;
6. material full-graph exact improvement;
7. private saved-notebook verification before any authorized Kaggle submission.

The existing full-system promotion threshold remains approximately **+0.010 exact gain** with no true-positive, embryo, or topology regression.

## What remains private

This repository does not publish model weights, active AWS code, exact feature recipes, working caches, private data, account credentials, or executable submission logic. The purpose is to show the research reasoning, validation discipline, and engineering boundary—not to publish the active competition system.

## Status

The native metric discrepancy is resolved and the point-only division line is closed. The image-conditioned branch is the next unexecuted research milestone. **No score improvement beyond the verified 0.947 official result is claimed.**
