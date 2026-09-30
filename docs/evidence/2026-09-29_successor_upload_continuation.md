<!-- Archived contemporaneous execution record. Historical paths, costs, and states are evidence, not current instructions. -->

# Biohub successor — continue the acknowledged final-input upload

## Current returned evidence

Inspected the mounted `/mnt/data/biohub_successor_final_submit_return.zip` programmatically. ZIP CRC passed. Source SHA-256: 478d3a3013fa0b040c12b98904ebfbbf73a72a0756e8cb7df8f91508f15785a4.

Run 20260929T045803Z_306d53 completed all eight stages in 20.848 seconds. Recorded command-time cost: USD 0.021775 on ml.g6e.4xlarge at USD 3.76/hour. Its decision is existing_final_input_visibility_pending.

The API acknowledged creation of private inference input alvaromendizabal/biohub-successor-58c949ac214f3e32. Private visibility was requested, not independently verified. The subsequent GetDatasetMetadata read returned HTTP 403, permission datasets.get denied. The owned inventory did not show the new target. The runner preserved the acknowledged creation and returned without a second creation, a final inference launch, or a competition submission.

This evidence does NOT prove that processing delay caused the denial. Current readiness, permissions, and visibility require a fresh authenticated check. The exact returned state has no acknowledged push version and no successor competition submission.

The first accepted submission 56659462 remained PENDING, with no public score, at the returned observation during 2026-09-29T04:58Z (September 28, about 9:58 p.m. Los Angeles). This is a historical returned observation, not a current-score assertion.

No model fitting, neural inference, graph evaluation, infrastructure mutation, or Git write occurred in this delivery invocation. Existing qualification was reused.

## Retained scientific state

Original Harmonic development control: 0.9330456734289274.
First accepted candidate, local score: 0.9480594130753609.
Qualified successor, local score: 0.95317334233363.
Successor edge TP/FP/FN: 2392/109/111. Division TP/FP/FN: 2/0/4.

The successor's +0.005113929258269145 incremental local gain comes from one branch removal on one repeatedly inspected development movie. It is not an independent holdout result, a new Kaggle score, or a guarantee of a stronger official score. The recorded best official score remains 0.947 until a scored improvement is observed.

Scientific ledger carried forward: 45 completed workflows, 38 historical hard failures, 10 historical scientific stops, one separately retained remote-service block, 94 research returns. Separate successor-deployment ledger: one successful qualification and one earlier failed deployment. This upload-stage result adds zero scientific workflows; it is a completed delivery phase with a readiness/access gate outstanding.

## Unchanged handoff

File: biohub_successor_final_submit.pyz
SHA-256: ebec9ea37a198c828d2ce797dab7c55fe7d4d2406efab70e3517520e563fb99f
Bytes: 42561

No production code or artifact bytes were modified in this response. No new variant of the inference system or model was created. The same package and command resumes the acknowledged upload and existing write-ahead state. There is no need to rerun training, detector/U-Net inference, the 39-candidate consolidation, successor research, or five-movie deployment qualification.

Existing inference content identity: 58c949ac214f3e32de96c884aee605c70e3364988c7da327ea0165ffd345591e.
Final loader/notebook SHA-256: 81ed844a239305aa18cd8e1fab2f523ba88723b1a1e33eab1a35d94c77389849.
Required private inference files total 5298909 bytes. Their actual blob bytes were not included in the current return; production checks verify them on AWS before any new remote operation.

## Tests executed in this response

### Exact downloadable archive

The exact archive's self-test passed all 54 included regressions, with zero failures/errors/skips. Both ZIP layers passed CRC checks, the payload SHA-256 matched, all 13 member hashes matched, and all 13 Python sources compiled and passed Python 3.12 grammar checks. The exact supplied shell command passed bash syntax checking. CLI help, a forbidden unrelated launch option, and an actual missing-workspace run were also checked. The missing-workspace run returned nonzero status and a valid PATH diagnostic ZIP.

### Additional continuation tests: 8 passed

Seven tests used the actual returned remote-state dictionary and transport-proof dictionary unchanged for the early read/gate paths: repeated unresolved visibility, private input processing, public input rejection, permission denial, first-submission official regression, missing authorization, and conflicting candidate identity. No recreation was permitted after acknowledged input creation.

The eighth test exercised processing, ready/launch, running, complete/submission, and already-submitted lifecycle transitions. It used the real returned state's structure with explicit fixture replacement of the missing model-asset byte identities. Across those phases: zero dataset creations, one input download, one push, one output download, and one submission. All remote calls were simulated.

### Actual completed qualification

The production qualify function re-read 49 bound files extracted from the actual native replay return, checked their hashes, and reopened all ten actual/expected complete graph pairs. It reproduced the recorded qualification and asset identities without running any model or graph postprocessor.

### Exact-package orchestration from acknowledged-upload state

The unchanged zipapp entry point completed four lifecycle invocations: launch from the recorded-creation state, running, completed output/submission, and already-submitted. All eight stages completed in each invocation, including the real qualification verifier, subprocess dependencies and regressions, notebook execution, and return packaging.

Explicit substitutes: host/GPU hardware discovery, omitted model/runtime asset bytes, and live Kaggle endpoints. The scientific qualification files were actual returned files. Candidate asset identity fields were adjusted only for those synthetic missing blobs in the local fixture. No such replacements are present in the downloadable production archive.

Simulated totals: {"asset_download": 1, "output": 1, "push": 1, "source": 1, "submit": 1}. Dataset creations: zero. No live Kaggle mutation occurred.

The local harness initially pointed a Python symlink at the wrong environment and was corrected to use the installed test environment. Multi-invocation testing crossed the tool watchdog; it was split into bounded executions and the remaining phases completed. These were local test-harness issues, not user AWS experiment results. They did not require changing production code. The CLI test was also corrected to test a genuinely unsupported option rather than argparse's accepted abbreviation of an existing flag.

### Notebook persistence

The original smoke and report functions executed using the actual latest returned JSON summary. Three saved charts were reopened and rendered in offline system Chromium in their sandboxed iframe contexts at 1080 by 620. JavaScript errors: zero. Network requests: zero. The actual stage-timing chart was visually inspected. Initial browser selection was corrected from an unavailable bundled executable to installed system Chromium. This is local saved-output evidence, not verification of the user's exact SageMaker frontend.

## Fresh AWS diagnosis

Authenticated read-only AWS check at 2026-09-29T16:22:11Z:

- Biohub JupyterLab application: Failed; configured instance ml.g6e.4xlarge.
- FailureReason: EC2InsufficientCapacityError; the requested 4xlarge instance was temporarily unavailable in the configured availability zones.
- Persistent Biohub space: InService; 128 GiB EBS configuration still present.
- RemoteAccess: DISABLED.
- No app, space, instance, quota, or storage was created, resized, stopped, or deleted by the assistant.

The appropriate user action is to reopen the existing space on an alternate supported size, preferably ml.g6e.2xlarge, retaining its storage and environment. This is not a guarantee that 2xlarge has immediate capacity. AWS's documented remedies are retrying later or choosing another instance size/type. No deletion of the persistent space is needed.

## Cost and limits

Authenticated SageMaker JupyterLab us-west-2 price snapshot returned on 2026-09-29 (publication 2026-09-28T21:57:12Z):

| Instance | USD/hour | Requested-command time ceiling |
|---|---:|---:|
| ml.g6e.xlarge | 2.61 | 600 seconds |
| ml.g6e.2xlarge | 2.80 | 600 seconds |
| ml.g6e.4xlarge | 3.76 | 478 seconds |
| ml.g6e.8xlarge | 5.66 | 318 seconds |
| ml.g6e.16xlarge | 9.47 | 190 seconds |

The same command requests 600 seconds maximum and USD 0.50 of command-compute allowance, shortened for the detected instance. Ten minutes on 2xlarge corresponds to about USD 0.467. Planning range for an active continuation is 1–5 minutes: USD 0.047–0.233 on 2xlarge or USD 0.063–0.313 on 4xlarge. Real live-service duration is not measured here. An unresolved-status path may finish sooner.

Required available system RAM: 8 GiB. Required free disk: 3 GiB. No model GPU computation is performed by this AWS delivery command. The final remote inference has its separate 9,000-second limit; it is not the predicted duration or included in the AWS command allowance.

Startup, idle time, storage, transfers, unrelated resources, and unusual cleanup overruns are excluded. This is not an account-wide spending cap. The runner does not stop the SageMaker app.

## Remaining live checks

AWS instance capacity; current authenticated Kaggle permission/readiness/private visibility; first submission's current official score; live input hash roundtrip; final inference completion and output validity; successor competition acknowledgment and score. None is replaced by a local fixture result.

The original first submission remains untouched. The same --submit-final flag authorizes only the successor's final inference-input/required code-submission lifecycle. AWS remains the research and report environment. A known first-candidate official score below 0.947 blocks further successor delivery for review. An identical/no-correction successor output is blocked. A persistent input-access error must not be bypassed.

## Current public references

- Official competition overview and rules: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/ — final deadline September 29, 2026, 23:59 UTC; offline notebook submission required.
- AWS troubleshooting: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-troubleshooting.html — insufficient-capacity remedies.
- AWS space configuration: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide-configure-space.html — change instance type for the existing space.
- AWS persistent-space explanation: https://aws.amazon.com/blogs/machine-learning/boost-productivity-on-amazon-sagemaker-studio-introducing-jupyterlab-spaces-and-generative-ai-tools/ — EBS storage persists independently until the space is deleted.

Return path remains /home/sagemaker-user/biohub_successor_final_submit_return.zip.

## Final live-state update before delivery

At 2026-09-29T16:32:33.377895Z (9:32 a.m. Los Angeles), the same Biohub application was Pending on ml.g6e.8xlarge, rather than Failed on 4xlarge. This newer read supersedes the earlier current-state description. The assistant made zero infrastructure mutations. Do not interrupt an in-progress startup merely to follow the earlier 2xlarge preference; use the existing terminal after it becomes InService. Capacity success remains unverified while status is Pending.

On this now-requested 8xlarge, the identical command automatically limits planned runtime to 318 seconds at USD 5.66/hour (about USD 0.4999). A 1–5 minute active invocation corresponds to approximately USD 0.094–0.472, excluding startup/idle/storage and other account costs. The runner has no model training or AWS inference to benefit from the larger GPU-host size.