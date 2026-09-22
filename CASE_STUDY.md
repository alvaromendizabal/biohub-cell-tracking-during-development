# Case study | Tracking cells under sparse supervision

## The problem

A tracking system must identify cells across time, assign continuations, and represent
divisions without introducing invalid lineage relationships. Sparse annotations make
validation especially important: lack of an annotation is not automatically evidence
that a predicted cell or link is wrong.

## My role

I implemented original representation and association experiments, developed an
AWS-centered execution and evidence workflow, reproduced a scored external reference,
and used exact evaluation and error analysis to decide which research directions to
continue. Public model and checkpoint authors retain credit for the reference methods.

## Benchmark first, then test incremental value

The project established an official 0.947 public-score reference. Research candidates
were compared against frozen baseline predictions rather than against changing
controls. Model diagnostics and development scores were kept distinct from official
competition performance.

An expanded association feature set improved neither tracking decisions nor the
small-table tracking metric over the frozen public probabilities. This prevented a
weaker fitted model from replacing the scored baseline merely because it used more
features.

## Use negative results to make better decisions

A proposed division-retention rule was rejected after an input audit showed its saved
candidate graphs lacked the alternatives it needed. A conservative temporal
reassignment experiment completed on five full movies but accepted no changes. A
learned competing-parent experiment made graph changes without improving the scored
development outcomes and was also rejected.

These outcomes are not presented as successful model improvements. They show a
measured decision process: verify applicability, compare the actual graph, preserve
the stronger reference, and avoid promoting an unsupported change.

## Attribute the errors before escalating complexity

The exact error analysis separated missed detections from incorrect links between
already-detected cells. On five development movies, it reconciled 2,391 correct,
111 incorrect, and 112 missed edges. Division counts were one correct, one incorrect,
and five missed events. Several missed daughter relationships competed with existing
assignments, explaining why add-only rules could not solve them.

![Reconciled development errors](assets/exact_error_budget_0.png)

## Engineering decisions

Expensive predictions and completed graph evaluations were cached instead of repeatedly
regenerated. Failed experiments preserved diagnostic evidence. Invalid training
checkpoints were excluded from promotion. Notebook plots remained embedded and
readable after reopening. Public reporting was kept separate from working datasets,
weights, account settings, and implementation.

## What is finished and what is not

The verified benchmark, error-attribution study, and documented completed ablations
are finished research milestones. The image-association follow-up screen is now recorded in RESULTS.md; it remains exploratory development evidence. Independent confirmation and a new scored notebook
are still needed before claiming a stronger system.

## Evidence limitations

The recent development studies cover five movies from two previously examined
embryos, with few annotated divisions. Pretrained-checkpoint training independence
is not established. Cross-embryo evaluation does not remove those limitations.
The public leaderboard snapshot is dated and does not establish a current rank.

This document intentionally omits implementation details, parameter settings,
feature recipes, and execution instructions.
