# Case study | Tracking cells under sparse supervision

## The problem

A tracking system must detect cells, preserve identity through time, and represent divisions without creating invalid lineage structure. Sparse annotations complicate evaluation: an absent annotation is not automatically a negative example, while the competition metric still penalizes wrong links, false forks, and excessive detections.

## My role

I reproduced a scored external reference, implemented original representations and structured association models, developed an AWS-centered execution and evidence workflow, trained transfer models, reconciled proxy and native graph metrics, and used exact graph evaluation to decide which research directions to continue. Public models, checkpoints, and baseline architecture authors retain credit for their contributions.

## Benchmark first, then measure incremental value

The project established an official **0.947** Kaggle public-score reference. Development metrics, ablations, and diagnostics were kept separate from official competition performance, and learned candidates were compared against frozen source predictions rather than moving controls.

The research moved through feature ablations, graph corrections, image-based division models, sparse-real transfer, grouped tracklet models, mixture-of-experts ranking, and dense parental competition. Each branch had an explicit promotion gate and was closed when the full graph evidence did not support it.

## Why proxy metrics were not enough

The cross-fitted tracklet mixture-of-experts model improved held-out parent association across all six movies, but its exact development candidate improved by less than the predeclared full-graph promotion threshold and still left division recovery weak.

A later dense-parental model trained directly on detected-point neighborhoods. Under the earlier research screen it appeared to add correct associations without recovering divisions. A subsequent native-metric reconciliation showed the more important truth: the learned graphs did recover **2 of 10** division events and **17 additional correct edges**, but also created **89–94 false divisions** and roughly **104–105 additional false edges** versus the raw-source comparator.

The aggregate native score therefore fell from **0.8301** for the raw-source comparator to **0.8058** for the pairwise model and **0.8054** for the contextual model. This is a negative result, but it is also a decisive one: point-only parental competition is not the next system to promote.

## Reconcile the evaluator before spending more compute

The metric audit froze all **18 graph systems** before reading annotations, used the pinned native graph metric, and performed event-level attribution. It found that a prior proxy screen omitted some false-positive links involving unmatched endpoints and treated division timing more strictly than the organizer metric.

That discrepancy changed the diagnosis of the model. The issue was not simply that the learned model never found a real division. Instead, it occasionally found the event while making too many surrounding association mistakes. The next branch therefore needs better evidence for when a split is real, not another point-only threshold sweep.

## What the event audit exposed

For true division daughters with detector matches, several errors were competitive rather than missing-candidate failures: the correct mother was present but ranked behind a nearby alternative parent. A smaller set of daughters lacked a unique detector-point match altogether. This separates representation errors from detection/matching limitations and gives the next model a concrete target.

## Current frontier

The active research contract is now **image-conditioned temporal division-event scoring with source preservation**. The design keeps strong source associations by default, uses image evidence only where a division hypothesis is independently supported, and requires verified supervision before fitting.

The next model must beat geometry-only controls, improve real division recovery without losing trusted edge true positives, preserve embryo-level performance and topology, and pass the same full exact evaluation before any submission is prepared.

## Engineering decisions

Expensive predictions and graph evaluations are cached rather than regenerated. Each bounded runner validates its environment, artifacts, notebooks, and return bundle before spending compute. Failed experiments preserve diagnostics and resumable state. AWS remains the canonical research environment; the public GitHub repository contains only employer-facing evidence, aggregate results, and methodological boundaries.

## Evidence limitations

The development movies are a retrospective diagnostic cohort and have been examined repeatedly. They support controlled engineering comparisons, not independent generalization claims. The official 0.947 score remains the only improved-performance claim in this portfolio.

This document intentionally omits private datasets, model weights, exact feature recipes, execution instructions, and submission logic.
