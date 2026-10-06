# Case study | 3D cell tracking under sparse supervision

**Status:** archived research; the original AWS workspace has been removed. [Run a small public graph example](examples/README.md) or [inspect the executed evidence](notebooks/README.md).

## Executive summary

This project combines computer vision, temporal modeling, graph reasoning, and ML systems engineering. The task is to reconstruct cell identities and division lineages through 3D microscopy sequences where annotations are sparse and errors compound across time.

My contribution covered the full research loop: reference reproduction, feature and model development, native-metric evaluation, failure analysis, AWS/GPU execution, runtime optimization, provenance controls, and publication-quality packaging.

The most important result was not a single model change. It was building an evidence process strong enough to reject locally attractive ideas when independent evidence disagreed.

## The problem

A complete tracker must solve three coupled tasks:

1. **Detection** — localize cells in anisotropic 3D volumes.
2. **Association** — connect the same cell across adjacent time points.
3. **Division modeling** — create valid one-parent/two-daughter lineage events without false forks.

Sparse labels make this harder: an unlabeled cell is not automatically a negative example, and a small association mistake can alter an entire downstream lineage graph.

## My role

I treated the project as an end-to-end ML system rather than an isolated model notebook. I:

- reproduced and froze a scored reference for controlled comparisons;
- implemented original temporal, graph, and candidate-set representations;
- evaluated 557 association features and multiple learned association families;
- reconciled proxy metrics with organizer-native graph behavior;
- built edge- and division-level error budgets;
- diagnosed output fidelity after an external regression;
- used AWS GPU environments with resumable artifacts and bounded runs;
- optimized a motion-linking component with output parity checks;
- packaged reusable code, executed notebooks, tests, provenance receipts, and CI.

Public architectures, checkpoints, notebooks, and organizer code remain attributed to their original authors.

## System design

| Layer | Engineering decision |
|---|---|
| Input | Treat physical 3D geometry and anisotropy explicitly |
| Detection | Keep detection and association evidence separable so regressions can be localized |
| Candidate generation | Preserve a strong frozen reference while evaluating new candidate relationships |
| Representation | Combine geometric, temporal, graph, and learned candidate-set features |
| Association | Evaluate both local ranking quality and complete graph effects |
| Lineage constraints | Track false links, missed links, forks, and division structure |
| Evaluation | Prefer native full-graph evidence over convenient proxy metrics |
| Delivery | Hash outputs, preserve receipts, bound runs, verify notebooks, and test from a fresh clone |

## Decision 1 — freeze a trustworthy reference

A faithfully reproduced external reference scored **0.947**. Freezing that result as a control made later comparisons interpretable and prevented baseline drift from being mistaken for research progress.

## Decision 2 — evaluate the complete graph, not just a classifier

Several ideas improved parent-ranking or local association measures without improving the complete lineage graph. The project therefore tracked exact edge true positives, false positives, false negatives, and division topology alongside model-level metrics.

Across one five-movie retrospective cohort, a selected integration changed the local graph metric from **0.933046 to 0.948059**, with edge TP 2,391→2,392, FP 111→110, FN 112→111, and one additional correctly recovered annotated division.

An independent scored evaluation of that candidate was **0.946**, below the frozen 0.947 reference. That contradiction changed the research direction: the repeatedly inspected cohort was no longer treated as an unbiased estimate of generalization.

## Decision 3 — diagnose the regression before adding complexity

I compared the frozen-reference and candidate predictions using hash-bound output audits. The candidate preserved the same preview detections and coordinates; the difference was **eight association-edge edits across four movies**.

This ruled out a broad detector/coordinate-drift explanation and focused the lesson on association generalization and validation quality.

## Decision 4 — measure prediction diversity, not model names

Multiple public notebook variants that appeared to be separate systems were checked at the complete-output level. Their saved predictions were byte-identical. The practical lesson is directly transferable to ensemble work: architectural or notebook diversity is not useful if residual behavior is the same.

## Decision 5 — optimize runtime only with parity evidence

A motion-linking component was reduced from roughly **80.5 seconds to 2.32 seconds**, about **34.7×**, while preserving selected edges under parity checks. I report this as a component benchmark, not a whole-pipeline speedup.

## What did not work

Several completed experiments were closed rather than polished into success stories:

- broad association feature expansion did not improve tracking despite **557** numeric candidates;
- sparse-real division fine-tuning produced no exact graph gain;
- dense parental competition added true associations but too many false links/forks;
- some image-conditioned work was blocked by an input-coordinate contract before a valid fit;
- a higher-order tracking branch reached GPU inference but not a complete native-metric result before the evidence cutoff.

Keeping these outcomes visible demonstrates model judgment: a negative scientific result is different from a broken execution, and neither should be rewritten as a win.

## ML systems engineering

The research workflow used AWS as the canonical environment and emphasized:

- resumable caches/checkpoints;
- immutable artifact hashes;
- isolated CPU/GPU runtime layers;
- bounded runtime and failure returns;
- persistent notebook outputs;
- source/output/version checks;
- synthetic/unit tests;
- a fresh-checkout CI gate;
- explicit separation of public artifacts from private data, weights, credentials, and orchestration.

## Transferable skills demonstrated

- **Computer vision:** volumetric data, localization, temporal context, augmentation/model components.
- **Graph ML:** association graphs, lineage constraints, topology-aware error analysis.
- **Model development:** ablations, transformers, representation research, failure-driven iteration.
- **Evaluation design:** validation contamination, native metrics, exact error budgets, independent evidence.
- **MLOps / ML systems:** AWS GPU workflows, reproducibility, artifact provenance, CI, bounded execution.
- **Technical judgment:** stopping weak branches, preserving adverse evidence, and separating scientific from engineering failures.

## Portfolio boundary

The public repository contains the reusable first-party package, tests, selected research modules, executed notebooks, compact evidence receipts, and provenance needed to inspect the work. It intentionally omits private datasets, model weights, tuned private thresholds, large caches/checkpoints, credentials, and private submission/orchestration machinery.
