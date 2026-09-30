<!-- Archived contemporaneous execution record. Historical paths, costs, and states are evidence, not current instructions. -->

# Biohub: qualified Harmonic delivery-resume evidence

## Inspected source and actual AWS result

Source: the mounted user upload `biohub_harmonic_winner_deploy_return.zip`; programmatic inspection was necessary because Files returned no parsed ZIP content. The uploaded run is `20260928T030216Z_c11643` (runner `biohub-harmonic-winner-deploy-20260928-v1`).

The AWS run failed only at the final private-kernel launch. Its first ten stages passed, including five dynamic graph replays, five complete native graph replays, exact shared-encoder equality, notebook construction, and executed Notebook 67. The native replay stage took 273.647 seconds. Total elapsed time was 313.495 seconds; recorded command-compute cost was $0.243829.

The launch logged HTTP 400 from Kaggle SaveKernel. The old wrapper did not preserve a detailed response body. The exact server-side rejection reason is therefore not established. The old notebook had 3,153,363 bytes and embedded a 2,340,859-byte model head inside a base64 ZIP. Removing this large inline payload addresses a concrete transport risk; it is not proof that size caused the HTTP 400.

No confirmed private launch, official competition submission, new fit, or new score resulted from that run. One attempted push must not be confused with zero attempted Kaggle actions simply because an outer counter was not incremented after failure.

## Scientific result preserved

The selected policy remains `trajectory__source_division_strict`, based on the original scored Harmonic system, not a weaker component baseline.

| Five-movie retrospective native metric | Original control | Selected candidate |
|---|---:|---:|
| Local exact score | 0.9330456734289274 | 0.9480594130753609 |
| Edge TP / FP / FN | 2391 / 111 / 112 | 2392 / 110 / 111 |
| Division TP / FP / FN | 1 / 1 / 5 | 2 / 1 / 4 |

Local score gain: 0.015013739646433488. The existing improvement, true-positive, embryo, material-change, topology, and false-division gates passed. The official scored reference remains 0.947, submission 56376695; no new official improvement is established. The development cohort has been inspected repeatedly; its local gain is not an independent estimate of unseen-test performance.

Current research ledger carried forward: 42 completed workflows; 36 hard execution failures; 10 separately retained historical gated stops; 88 unique research returns; zero official improvements beyond 0.947. Local tests and simulated API requests are not new AWS research experiments.

## Transport-only change

The new runner verifies all archived qualification-file hashes and the ten completed graph-output hashes. It does not repeat graph replay, model training, or AWS inference.

The exact existing runtime ZIP is placed in a private input as `winner_payload.bin`. The old notebook's large base64 assignment is replaced by a local mounted-file read. Removing that one transport assignment from the before/after syntax trees produces identical ASTs. All 11 embedded module/head member hashes remain identical. The original ten inference blocks are retained unchanged.

The resulting notebook is 2,860 bytes. The private input contains three files totalling 2,405,929 bytes: the unchanged runtime ZIP, extracted unchanged inference entry, and a hash manifest. It does not contain training images, labels, predictions, credentials, or the AWS workspace. Private dataset identity is derived from content hashes. The runtime refuses missing, altered, or ambiguous inputs.

The existing pilot head safely loaded on CPU with `weights_only=True`: 60 tensor keys, all finite. Its SHA-256 remains `f37e73e8d22c5d5fbcbe2375334258e200e1da899d17c7e401fbe9a74e3da8af`. No head inference or fitting was performed in this check.

The private artifact is checked for visibility and byte-for-byte upload roundtrip before notebook launch. A new upload/version is not silently created after an ambiguous mutation. Current CLI/API exceptions preserve a sanitized response body and status. A subsequent failed remote run collects the small available timing/manifest diagnostics without submitting it.

## Complete lifecycle and authorization

`--launch-private` explicitly authorizes creation of the one immutable private inference asset and a private saved-notebook launch. `--submit-when-ready` explicitly authorizes one saved-notebook competition submission only after the recorded kernel version completes and source/output verification passes. These actions execute from the AWS terminal, not from this assistant's tools.

The same command resumes after private-asset processing or remote inference. It does not keep the AWS instance polling throughout the remote run. Remote kernel/source collisions, ambiguous unconfirmed push/submission outcomes, failed inference, invalid outputs, and exhausted work budgets stop safely instead of creating duplicate remote actions. An ambiguous source-matching remote version without a confirmed version number is reported for reconciliation, not guessed.

Output verification checks the pinned saved source, private asset, frozen tight55 configuration, all ten completed inference blocks, per-movie dynamic evidence, nonempty changes, complete CSV row/movie coverage, finite nonnegative coordinates, unique IDs/edges, consecutive-frame links, one-parent/two-child degree limits, and agreement with the notebook's own output hash/counts. It rejects a CSV identical to the old scored baseline. No direct CSV-file upload is used for the competition: the official code-submission API is bound to the validated saved-notebook version.

## Actually executed tests

- Final downloadable zipapp: 65 passing targeted regressions, zero failures/errors/skips.
- Exact final archive: CRCs for both ZIP layers, all nine payload-member SHA-256 values, all eight Python sources compiled, and Python 3.12 grammar checked.
- Eight final artifact/CLI checks: archive/compile, help, invalid cost argument, standard-library-only missing-workspace failure, actual local missing-GPU failure, corrupted payload rejection, concurrent-run isolation, and supplied Bash syntax.
- Complete eight-stage packaged workflow with the actual returned qualification files and explicit local hardware/Kaggle API fixtures. Transport preparation, file hashing, original-code extraction, reporting, and packaging executed. The fixture progressed through launch, running, verified output/submission, and a repeated completed invocation. Across that lifecycle: exactly one simulated asset creation, one simulated push, one simulated submission, and no duplicate output download.
- Full packaged HTTP-400 failure path with a simulated API response. The real bridge captured the response body/status, redacted the fixture secret, stopped with nonzero status, executed its failure report, and produced a diagnostic return ZIP without model/input/CSV binaries.
- Actual returned selected graph geometry was serialized with the original writer's nonnegative rounded-coordinate convention and independently validated: five movies, 142,388 nodes, 137,018 edges, 279,406 CSV rows. Validation took approximately 2.1 seconds locally. The timing, transport, and delivery envelope records in this test were explicit fixtures, not new inference or quality measurements.
- The thin loader verified all actual private-asset file bytes and completed its transport receipt with the heavy inference entry explicitly mocked. A corrupted asset stopped before the entry call.
- The report and smoke notebooks executed, and their saved Jupyter trust signatures were verified. All four saved plots reopened and rendered offline in Chromium at 1080 by 620, without browser errors or external requests. The stage-time chart was visually inspected.

Local testing caught an error-redaction coverage defect before release; it was fixed and retested. Initial fixture setup needed the actual installed Python environment rather than a bare interpreter alias. The large CSV fixture was corrected to use the original nonnegative rounding convention rather than introducing an invalid alternative serializer. An externally interrupted report test produced a bounded-stop bundle and resumed; it was not treated as a completed cloud run.

## What was not executed here

No live private Kaggle asset upload, kernel push, remote inference, competition submission, or official score retrieval was performed by the assistant. Kaggle SDK/API behavior in the full workflow used explicit protocol fixtures, not an authenticated cloud execution. The exact current SDK method signatures are inspected in the existing AWS environment before mutations. Real credentials, permissions, quota, dataset processing, CUDA/hidden-test inference, native library behavior, and full notebook runtime remain environment-dependent.

There is no assistant-owned AWS account available. The read-only connection reports this SageMaker space has remote access disabled. No AWS infrastructure or GitHub writes occurred.

## Current AWS and cost bounds

Read-only check: 2026-09-28T03:47:10.058143+00:00. Biohub JupyterLab: InService, ml.g6e.2xlarge. Remote access: DISABLED. No cloud mutations.

Authenticated AWS Pricing returned these us-west-2 SageMaker JupyterLab hourly USD rates, publication 2026-09-27T02:52:30Z:

| Instance | USD/hour | Maximum planned seconds with 600-second / $0.50 request |
|---|---:|---:|
| ml.g6e.xlarge | 2.61 | 600 |
| ml.g6e.2xlarge | 2.80 | 600 |
| ml.g6e.4xlarge | 3.76 | 478 |
| ml.g6e.8xlarge | 5.66 | 318 |
| ml.g6e.16xlarge | 9.47 | 190 |

Planning allowance on 2xlarge: 2–5 minutes ($0.093–$0.233), not a measured cloud completion forecast. Ten minutes is approximately $0.467. The runner is CPU/transport-only, requires 8 GiB available system RAM and 3 GiB free disk, and reserves time for reporting/cleanup. Remote Kaggle inference has a separate 9,000-second safety timeout, not a runtime forecast. The command budget does not cap startup, idle, storage, unrelated resources, or extraordinary cleanup overruns. The runner does not stop the SageMaker app.

## Artifact

Filename: biohub_harmonic_delivery_resume.pyz

SHA-256: 46c257ff6702cf16102a85b28a5d24f13f787481a62dad96da9ab8621302b302

Size: 33,456 bytes.

Expected return: /home/sagemaker-user/biohub_harmonic_delivery_resume_return.zip