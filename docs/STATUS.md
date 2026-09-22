# Current state and unresolved work

Snapshot basis: returned run `20260922T020508773179Z`, ending **2026-09-22 02:05:40 UTC**.

## Completed

The public Harmonic Fusion notebook reproduced the official **0.947** public result
(submission 56376695, saved version 1). Six completed AWS research notebooks are
included as evidence, alongside this publication's data-free overview. Their
negative or diagnostic results are retained without promoting them into improvements.
The five-movie baseline has 2,391 correct, 111 incorrect, and 112 missed edges, with
one correct, one incorrect, and five missed annotated divisions.

## Latest attempt: blocked, not a negative model result

The appearance-association helper ran for **31.319 seconds**. Payload integrity,
50 synthetic association tests, and the real saved Plotly preflight passed.
The first movie generated **181,638 candidate pairs**. At its first image-profile
frame the worker raised **`IMAGE_COORDINATE_BOUNDS`**. It fitted **zero** project-data
models, made no submissions, and did not complete Notebook 30.

The check requires finite coordinates and bounds relative to the image grid.
The returned traceback does not establish whether the failed conjunct was nonfiniteness,
a coordinate convention mismatch, or an actual out-of-grid detection. It does not
contain the offending coordinates. Clipping coordinates without diagnosing units,
transforms, and bounds would conceal the problem rather than fix it.

Evidence: [run receipt](../reports/appearance_association/run_receipt.json),
[worker failure](../reports/appearance_association/worker_failure.json), and
[fit counter](../reports/appearance_association/fit_progress.json).

## Next research gate, after publication

Inspect the failing movie's coordinate ranges and image shape, axis ordering,
physical/voxel scaling, crop transforms, and boundary convention. Preserve a tiny
non-sensitive synthetic regression case for the actual cause; then rerun only the
first-frame extraction before allowing fitting. No correction has been experimentally
validated in this publication step.

## Competition scope

The last captured leader score is **0.974**, observed September 21, 2026. The recorded
gap is **0.027**. There is no new official challenger score. The public/private metric
and observation dates must remain separate from local development measurements.

A finished **publication snapshot** is not a claim that all research goals are complete.
