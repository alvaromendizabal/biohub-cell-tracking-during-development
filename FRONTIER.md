# Research frontier

## Final verified state at publication cutoff

The competition deadline has passed, so this document records the research frontier rather than an active submission plan.

The strongest **verified personal official score** in the captured evidence remains **0.947** from submission 56376695.

The locally qualified source-preserving candidate scored **0.946** officially, despite improving the retrospective development metric from 0.933046 to 0.948059. That result closes the specific local graph-edit policy as generalization evidence.

An exact public notebook advertised at 0.953 was accepted as submission **56687425**, but it was still pending at the latest captured evidence cutoff. It is therefore tracked as an accepted reproduction attempt, not promoted to a verified personal score.

## What the official regression taught us

The post-submission fidelity audit established that the failed candidate did not alter the preview detection universe or coordinates. It made only a small number of association-edge changes.

The remaining uncertainty is therefore not “did the deployment accidentally change detections?” but whether association changes selected on an in-sample/repeatedly inspected cohort can generalize at all.

That shifts the research priority toward **independent validation and structurally new representations**.

## Public 0.953 lineage

Three public notebooks advertised at 0.953 were audited. Their complete saved prediction outputs were byte-identical, so they are one prediction lineage for diversity purposes.

The associated source audit indicates that the lineage combines coordinate refinement with additional relinking/recovery logic. The refinement head is small, but the available evidence does not isolate it as the source of the full public-score difference.

A future clean study would need same-source head-on/head-off inference with version-locked historical inputs and embryo-disjoint upstream training provenance.

## Higher-order association

HOCT remains the most interesting unfinished association branch because it is structurally different from the Harmonic linker and the project's grouped-tracklet/MOE families.

The deadline sprint established that:

- the public model can run on the project's GPU environment;
- neural inference completed on a large movie;
- global ILP decoding was operationally constrained by solver availability/runtime;
- a source-preserving fast decoder still needed one final API integration fix before native scoring.

That is enough to keep HOCT on the future research frontier, but not enough to claim a metric gain.

## Recommended next research layer

If this project is continued outside the competition deadline, the highest-value sequence is:

1. reconstruct a clean embryo-disjoint detector validation regime;
2. finish the frozen-detection HOCT comparison under the native metric;
3. run a controlled localization ablation on the public refinement mechanism;
4. test a genuinely independent high-resolution 2.5D detector representation;
5. evaluate ensemble value using prediction/residual diversity rather than notebook lineage names.

The goal is to escape the 0.947/0.953 public lineage by changing validation quality and representation, not by accumulating more local graph micro-adjustments.

## What remains private

This repository does not publish model weights, private AWS orchestration, tuned thresholds, exact feature recipes, cached candidate graphs, competition data, or executable submission logic.

## Status

**Verified official score 0.947; locally promoted candidate officially regressed to 0.946; exact public-0.953 reproduction accepted but unscored at the latest captured evidence cutoff; public 0.953 variants proved prediction-identical; HOCT native scoring unfinished.**
