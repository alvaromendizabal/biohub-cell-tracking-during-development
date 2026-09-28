# Case study | Tracking cells under sparse supervision

## The problem

A tracking system must detect cells, preserve identity through time, and represent divisions without creating invalid lineage structure. Sparse annotations complicate evaluation: an absent annotation is not automatically a negative example, while the competition metric still penalizes wrong links, false forks, and excessive detections.

## My role

I reproduced a scored external reference, implemented original representations and structured association models, developed an AWS-centered execution and evidence workflow, trained transfer models, reconciled proxy and native graph metrics, integrated completed research back into the strongest full baseline, and hardened the saved-notebook delivery path. Public models, checkpoints, and baseline architecture authors retain credit for their contributions.

## Benchmark first, then measure incremental value

The project established an official **0.947** Kaggle public-score reference. Development metrics, ablations, and diagnostics were kept separate from official competition performance, and learned candidates were compared against frozen source predictions rather than moving controls.

The research moved through feature ablations, graph corrections, image-based division models, sparse-real transfer, grouped tracklet models, mixture-of-experts ranking, dense parental competition, native metric reconciliation, and finally baseline-locked integration.

## Why proxy metrics were not enough

Several learned systems improved intermediate association metrics without improving the complete tracking graph.

A dense-parental model, for example, recovered real division events and additional correct links but created too many false associations and false forks under the native organizer metric. That result closed the point-only branch.

An image-conditioned reparenting pilot then tested whether frozen image features could disambiguate difficult parent choices. The fixed image policy ultimately accepted no corrections under its preservation gate. That branch also closed.

These results were useful because they narrowed the mechanism that mattered: any improvement had to preserve the already strong Harmonic graph rather than replace large portions of it.

## Return every compatible idea to the strongest baseline

The next milestone changed the experimental frame.

Instead of asking whether a candidate beat a weaker component comparator, the system replayed **37 saved policies and two controlled combinations against the original Harmonic development control**. Historical candidates were represented as complete graph changes relative to their own paired sources, then transferred only when they could be mapped uniquely and the affected original neighborhood remained compatible.

The selected source-preserving candidate improved the local exact score from **0.933046 to 0.948059** (**+0.015014**), increased correct edges from **2,391 to 2,392**, reduced false edges from **111 to 110**, and increased correct divisions from **1 to 2** without adding a false division.

The all-conflict-free combination performed substantially worse. This confirmed an important engineering principle: **conservative integration beat indiscriminate aggregation**.

The development cohort had been inspected repeatedly, so this is not presented as independent evidence of leaderboard generalization. The official public score remains 0.947 until a new saved-notebook result is scored.

## Make deployment reproducible before claiming a win

The qualifying graph policy was then reconstructed from prediction-time artifacts rather than hard-coded development edits. Deployment checks required the dynamic replay to reproduce the saved winner, and required the additional association head to remain compatible with the frozen image encoder.

The delivery path also removed unnecessary repeated work:

- a parity-checked spatial optimization reduced the measured motion-linking component time by about **34.7×**;
- validation inference and post-processing reselection were removed from the frozen deployment path;
- the additional private inference assets were externalized from the notebook;
- notebook launch, output verification, and competition submission were converted into idempotent state transitions so retries do not deliberately duplicate remote writes.

The latest remote delivery state has acknowledged creation of the private inference input. Private notebook completion and an official new score are still pending.

## Engineering decisions

Expensive predictions and graph evaluations are cached rather than regenerated. Each bounded runner validates its environment, artifacts, notebooks, and return bundle before spending compute. Failed runs preserve diagnostics and resumable state. Candidate graphs are frozen before retrospective labels are opened. AWS remains the canonical research environment; the public GitHub repository contains only employer-facing evidence, aggregate results, and methodological boundaries.

## Evidence limitations

The development movies are a retrospective diagnostic cohort and have been examined repeatedly. They support controlled engineering comparisons, not independent generalization claims.

The locally qualifying candidate was selected among multiple integrated policies. Its **0.948059 local development score is not comparable to the official 0.947 Kaggle score**. The only definitive next evidence is a completed, verified saved-notebook submission on unseen competition inputs.

This document intentionally omits private datasets, model weights, exact feature recipes, tuned thresholds, execution instructions, and submission logic.
