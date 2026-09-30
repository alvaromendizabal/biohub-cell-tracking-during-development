<!-- Archived contemporaneous execution record. Historical paths, costs, and states are evidence, not current instructions. -->

# Biohub metric reconciliation — inspection and test evidence

## Scope and limitations

This record distinguishes the user's completed AWS experiment, local replay of actual returned graphs with explicit external-dependency substitutes, packaged orchestration tests, and the native AWS task that has NOT yet executed. No new project model was trained in this response. No AWS compute, infrastructure change, Git write, download of project data, or Kaggle submission was initiated. The artifact is an audit of existing predictions, not a score-improvement claim.

## Latest returned AWS experiment

Source: biohub_dense_parental_return.zip; results/run_receipt.json, result.json, execution_ledger.json, data_receipt.json, and frozen-model/prediction receipts. Run ID: 20260927T042847Z_76d1da.

The run completed all seven stages, eight CUDA fits, 32 fixed epochs per fit, twelve learned prediction graphs, and Notebook 62. Elapsed time: 73.933 seconds. Recorded command-time estimate: $0.057503 at $2.80/hour. No exact competition metric was executed in this run; its result explicitly says exact_metric_run=false.

The original research screen reported:

| System | Correct edges | Screen false-positive edges | Missed edges | Screen edge F1 | Exact-frame division TP |
|---|---:|---:|---:|---:|---:|
| Raw source | 2079 | 2 | 251 | 0.942643392 | 0/10 |
| Pairwise | 2096 | 3 | 234 | 0.946489049 | 0/10 |
| Context | 2096 | 3 | 234 | 0.946489049 | 0/10 |

The screen's division-FP counts were 0, 2, 2 respectively; unassessed predicted divisions were 0, 89, 94. The prior decision was close_fixed_dense_parental_pilot_no_threshold_sweep. That rejection remains in force. These are six-movie detector-transfer results, not the complete Harmonic source's leaderboard score.

Recorded cumulative tally: 82 unique returns; 38 completed workflows; 34 hard execution failures; 10 separately retained historical gated stops; zero official improvements beyond 0.947. These historical totals are carried forward from the ledger, not reconstructed as a complete fresh audit. A completed negative experiment counts as a completed workflow; local test runs do not increment this ledger.

## Discovered measurement mismatch

The previous screen is not equivalent to the organizer metric. It omits some false-positive links whose endpoint evidence makes them evaluable under the organizer rules. It also requires an exact-frame split, whereas the pinned division implementation evaluates a bounded local temporal window with directed topology and one-to-one matching of events.

The audit reproduced both mismatches. Organizer documentation inspected on this date:
https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md

The production package bundles the metrics.py and division_metrics.py source already pinned in the earlier exact bridge; those source files were verified byte-for-byte unchanged. It does not replace native scoring with the local test adapter.

## Actual returned graph replay — LOCAL DIAGNOSTIC ONLY

Input: six existing raw detector graphs and twelve saved learned graphs, across 137,009 detector nodes. All returned coordinates were integral; no rounding of fractional coordinates was needed in the local adapter. The unchanged pinned division-scoring algorithm ran through an explicit test-only graph and optimal-matching adapter. Native tracksdata/Polars were unavailable locally. Edge counts were independently checked against the original graph structures using the same returned one-to-one point matches.

**These counts are local diagnostic evidence, NOT native-backend-verified competition scores.** The native backend may produce different matching choices and must be checked on AWS.

| System | Edge TP | Edge FP | Edge FN | Division TP | Division FP | Division FN | Edge Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|
| source | 2079 | 185 | 251 | 0 | 0 | 10 | 0.826640159 |
| pairwise | 2096 | 289 | 234 | 2 | 89 | 8 | 0.800305460 |
| context | 2096 | 290 | 234 | 2 | 94 | 8 | 0.800000000 |

The replay finds two time-shifted divisions in each learned family, accompanied by substantially more wrong links and false division events. This does not rescue the rejected model. Nine of ten exact-frame event decisions were identical between pairwise and context models. The audit is post-hoc and cannot create an independent confirmation set.

The returned graph caches do not contain all organizer coarse total-node estimates. No adjusted combined score was computed locally. The new AWS run searches existing organizer metadata and withholds that score unless the required estimates are available for every sample. Predicted-node count is not substituted for a missing true-node estimate.

## Delivered workflow

1. Standard-library launch, archive/hash validation, supported hardware and budget discovery, existing scientific interpreter selection, native dependency compatibility check, and native perfect-split smoke.
2. Run all 61 targeted regressions and execute persistent Plotly preflight before complete graph scoring.
3. Bind to the completed experiment receipts, freeze all eighteen existing source/candidate graph files, then reopen annotation graph content. Models, thresholds, predictions, coordinates, and topology remain unchanged.
4. Score eighteen movie/system combinations with the archived native metric implementation. Cross-check edge counts independently using the native match results and preserve event-level division evidence.
5. Check existing saved probability arrays for supplemental attribution when present and valid. The no-match probability class is index zero. Missing/corrupt supplemental arrays are explicitly blocked diagnostics, not fabricated values and not a trigger for inference.
6. Apply per-comparison input/code/result hashes for safe reuse. Search organizer metadata for valid total-node estimates. Preserve the prior rejection and prohibit post-hoc automatic promotion.
7. Execute Notebook 63, update the run ledger, and attempt one return ZIP on completion, failure, or budget stop.

No fitting, detector execution, neural inference, package installation, network project-data download, Kaggle action, or infrastructure change is performed. Independent CPU metric workers are bounded by host size and available RAM: one on xlarge, two on 2xlarge, four on 4xlarge, and up to six on 8xlarge/16xlarge. Each worker uses one math thread.

## Test evidence

### Exact final archive

Artifact: biohub_metric_reconciliation.pyz

SHA-256: 9104512ca436378d2e9cac4b89a6485cea05b8f96afaac4a468c63df346f20db

Size: 54,147 bytes. Python source files: 15, including the outer launcher.

The final downloadable archive itself passed 61 tests, zero errors, zero failures, zero skipped tests. Every Python file compiled and parsed under Python 3.12 grammar. Both archive layers passed CRC validation; payload member hashes passed. The pinned scientific metric sources were byte-identical to the earlier exact bridge. The final execution command passed bash syntax checking.

Tests cover sparse-endpoint false positives, exact/early/late division windows, distinct daughter branches, local merges, event matching, ground-truth masks, malformed graphs, nonfinite coordinates, invalid paths, metadata absence/conflicts, graph freezing, live input revalidation before cache reuse, corruption, probability-array schema, native API adapter behavior, original noncontiguous IDs, rejection preservation, ledger idempotency, owned-process timeouts, all five supported resource plans, and invalid budget arguments.

### Actual-data orchestration

All eighteen actual returned graph systems were frozen and evaluated through the disclosed matching adapter. An identical rerun reused all eighteen completed comparisons. Corrupting one result caused only that comparison to be recomputed. No new fit or prediction was made. The archive's native bridge was separately exercised with explicit native-API fixtures; these tests do not constitute execution of the real installed tracksdata package.

### Packaged entry point and failure paths

The final full-workflow test used the unchanged production launcher and final 61-test suite. Local hardware, unavailable native dependency probe, graph/matching backend, and pure-metric import boundaries were replaced explicitly. The full seven-stage workflow and return packaging completed. Native AWS scoring is not inferred from this result.

| Case | Exit status | State | Failure class | Return ZIP CRC passed |
|---|---:|---|---|---|
| budget_stop | 124 | stopped | BUDGET | True |
| cli_nan | 2 | argument rejected | none | not applicable to argument rejection |
| cli_seconds | 2 | argument rejected | none | not applicable to argument rejection |
| cli_instance | 2 | argument rejected | none | not applicable to argument rejection |
| corrupt_payload | 1 | failed | HANDOFF | True |
| dependency_failure | 1 | failed | DEPENDENCY | True |
| final_complete_workflow | 0 | complete | none | True |
| missing_interpreter | 1 | failed | ENVIRONMENT | True |
| missing_workspace | 1 | failed | PATH | True |
| native_failure | 1 | failed | METRIC | True |
| real_environment_failure | 1 | failed | ENVIRONMENT | True |
| regression_failure | 1 | failed | TEST | True |
| identical_full_rerun | 0 | complete | none | True |

The missing-workspace, missing-interpreter, real local no-GPU environment failure, corrupted payload, and invalid CLI checks did not claim a native scientific success. CLI argument rejection occurs before a project run and does not promise a diagnostic ZIP. Runtime failures preserve the real nonzero exit status and attempt packaging.

### Executed notebooks and offline browser tests

Three notebooks executed using the final report code: smoke, actual-data local diagnostic report, and failed/empty report. All nine saved charts were reopened and rendered at 1080 by 620 pixels in an offline Chromium browser. There were zero browser errors and zero external requests. Native Plotly outputs and self-contained HTML/JavaScript were preserved, execution counts were ordered, no notebook error outputs were present, and final completion sentinels passed. The actual-data report explicitly labels the local matching adapter rather than claiming native verification. The production runtime does not require a server-side browser or Kaleido.

### Still environment-dependent

The real tracksdata/Polars execution on the original AWS workspace, native matching and its tie behavior, availability/validity of all organizer total-node estimates, actual wall time, and the SageMaker browser frontend have not been validated by local substitutes. Earlier successful exact-return dependency records list tracksdata 0.1.0rc10 and Polars 1.44.2 in the GPU interpreter; the new run checks the current interpreter rather than assuming those versions remain intact. There is no separate assistant AWS account. The connected space reports RemoteAccess=DISABLED.

## AWS inspection and spending assumptions

Authenticated AWS API reads at 2026-09-27T04:36:32Z reported Biohub InService on ml.g6e.2xlarge in us-west-2. A final status read at 2026-09-27T04:59:03Z again reported InService on ml.g6e.2xlarge. No cloud mutation occurred. The following rates are AmazonSageMaker studio-jupyterlab prices from GetProducts, not generic EC2 prices; publication date was 2026-09-27T02:52:30Z.

| Instance | USD/hour | Planned maximum under 900 seconds and $0.75 |
|---|---:|---|
| ml.g6e.xlarge | 2.61 | 15 minutes |
| ml.g6e.2xlarge | 2.80 | 15 minutes |
| ml.g6e.4xlarge | 3.76 | 11 minutes 58 seconds |
| ml.g6e.8xlarge | 5.66 | 7 minutes 57 seconds |
| ml.g6e.16xlarge | 9.47 | 4 minutes 45 seconds |

All five have one L40S GPU with 48 GB of memory; official hardware reference: https://aws.amazon.com/ec2/instance-types/g6e/ . This audit uses CPU scoring, not GPU compute. Physical GPU information is used only for supported-instance discovery.

Planning range on current 2xlarge: 2–8 minutes, about $0.09–$0.37 command-time compute. Native full-run duration remains unmeasured. The supplied 15-minute allowance corresponds to $0.70 on this instance; the separate requested budget is $0.75. Major scoring stages reserve 120 seconds for reporting/cleanup and general stages retain a 45-second packaging reserve. Require at least 8 GiB available system RAM and 3 GiB free disk.

The budget is not an AWS account spending cap. Startup, idle time, storage, unrelated resources, and unusual cleanup overruns are excluded. The runner does not stop the SageMaker app. Stop the app after preserving the return ZIP to end its instance-compute billing; persistent storage may continue to cost money. Full historical/account-wide charges were not measured here.

## Decision and expected return

Expected return: /home/sagemaker-user/biohub_metric_reconciliation_return.zip

The runner ends with SUCCESS, FAILED, or STOPPED, the decision/failure class, and the return path. Hash-valid completed comparisons are resumable. A native failure is a blocked execution, not evidence against the model. A native-confirmed negative closes this point-only division route and supplies event-level requirements for a materially different, image-conditioned joint division model with source preservation. No such new model is trained by this handoff. Any later candidate still needs independent full-graph evidence under the unchanged +0.010, no true-positive/embryo loss, topology and historical-floor requirements before private notebook output verification and an explicitly authorized Kaggle submission.