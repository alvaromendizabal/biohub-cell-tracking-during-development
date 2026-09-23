# Case study | Tracking cells under sparse supervision

## The problem

A tracking system must identify cells across time, assign continuations, and represent
divisions without introducing invalid lineage relationships. Sparse annotations make
validation especially important: absence of an annotation is not automatically evidence
that a predicted cell or link is wrong.

## My role

I implemented original representation and association experiments, developed an
AWS-centered execution and evidence workflow, reproduced a scored external reference,
and used exact evaluation and error analysis to decide which research directions to
continue. Public model and checkpoint authors retain credit for the reference methods.

## Benchmark first, then test incremental value

The project established an official **0.947** public-score reference. Research candidates
were compared against frozen baseline predictions rather than against changing controls.
Development metrics, ablations, and diagnostics were kept separate from official
competition performance.

An expanded association feature set did not improve tracking decisions over the frozen
public probabilities. Subsequent graph, neural-affinity, detector-consensus, and
division-focused candidates were promoted only when they improved the full evaluator;
otherwise they were rejected.

## Use negative results to make better decisions

Several candidate families were deliberately stopped:

- a daughter-retention rule had no eligible alternatives in the saved graphs;
- temporal reassignment accepted no useful changes;
- learned competing-parent and neural decoders altered graph structure without improving
  the measured objective;
- detector localization and gap completion changed predictions but reduced or failed to
  improve the local score;
- detector-informed pruning produced the same tiny gain as a matched equal-count control;
- a hand-generated synthetic division model selected plausible events but recovered no
  scored divisions.

These are not presented as model wins. They demonstrate a measured research process:
verify applicability, compare against a fixed baseline and matched controls, preserve
stronger predictions, and stop unproductive directions.

## Attribute errors before escalating complexity

The exact error analysis separated missed detections from incorrect links between
already-detected cells. On five development movies, it reconciled **2,391 correct,
111 incorrect, and 112 missed edges**. Division counts were **one correct, one incorrect,
and five missed events**. Several missed daughter relationships competed with existing
assignments, explaining why add-only rules could not solve them.

![Reconciled development errors](assets/exact_error_budget_0.png)

## Pivot the training signal when post-processing plateaus

The recent experiments showed that repeatedly changing the output of the same pretrained
system was not enough. The active research lane therefore moves upstream to training from
a public fully labelled synthetic lineage resource containing **2,174 time sequences and
165,267 mitotic-parent events**.

The new AWS workflow uses sequence-level splits, held-out synthetic validation, a matched
geometry control, and the same full Biohub evaluator. Real Biohub annotations are not used
for model fitting or threshold selection. Training is not claimed complete until source
integrity, bounded data acquisition, public-graph validation, held-out validation, and
the five-movie comparison all pass.

This pivot illustrates a broader engineering lesson: when an inference-time correction
plateaus, change the supervision and representation rather than endlessly tune the same
decoder.

## Engineering decisions

Expensive predictions and completed graph evaluations were cached instead of repeatedly
regenerated. Failed experiments preserved diagnostic evidence. Invalid checkpoints were
excluded from promotion. Notebook plots remained embedded and readable after reopening.
Public reporting was kept separate from working datasets, weights, account settings, and
active implementation.

## Evidence limitations

The recent development studies cover five movies from two previously examined embryos,
with few annotated divisions. Pretrained-checkpoint training independence is not
established. Cross-embryo evaluation does not remove those limitations. A public-sequence
training result is not yet claimed.

The leaderboard comparison is a dated public snapshot, not a current private score.

This document intentionally omits implementation details, parameter settings, feature
recipes, and execution instructions.
