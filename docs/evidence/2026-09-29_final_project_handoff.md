<!-- Archived final project handoff. Historical paths/state are evidence, not current instructions. -->

# START OF HANDOFF PROMPT

You are taking over my **Biohub Cell Tracking During Development** Kaggle competition project on **September 29, 2026**, the final day of the competition. Read this entire handoff before proposing, generating, or running anything.

Continue from the exact current state. **Do not restart the project, repeat closed experiments, retrain models whose checkpoints are already valid, recreate the 0.947 baseline, or make me reconstruct project history manually.**

---

# 1. Core objective

The objective is to beat the strongest legitimate comparable Biohub Cell Tracking score **before the deadline today** while preserving rule compliance, validation rigor, AWS reproducibility, cost control, and a strong employer-facing project record.

The one verified official score that has survived all experiments is:

- **0.947 public Kaggle score**
- submission **56376695**
- description: `biohub-notebook-harmonic-v1-a69c78229c65 | saved notebook version 1; unchanged reference; no development corrections`

This is the **Harmonic Fusion reference** and it remains the official floor.

Do not confuse local retrospective improvements, unit tests, replay parity, or notebook construction with a real leaderboard improvement.

The recent target we had been chasing was around **0.970**, but before stating a current top score or leaderboard gap, inspect the current Kaggle leaderboard/public information and verify it.

---

# 2. Non-negotiable execution rules

## AWS is canonical

All development, research, training, validation, inference preparation, notebooks, checkpoints, model artifacts, experiment outputs, diagnostics, caches, and reproducibility state remain canonical in AWS/SageMaker.

Canonical project root:

`/home/sagemaker-user/biohub-cell-tracking-during-development`

Canonical AWS region:

`us-west-2`

Biohub SageMaker space:

`biohub-cell-tracking-during-development-dev`

## Kaggle is submission-only

Kaggle is **not** the development environment.

Use Kaggle only where this code competition technically requires a notebook/kernel as the final inference/submission vehicle. Treat that notebook as a **thin final delivery wrapper around artifacts and logic built and validated in AWS**.

Do not move research, training, tuning, model selection, validation, or canonical notebooks into Kaggle.

Any Kaggle submission must be initiated from the AWS SageMaker terminal and its result recorded back in AWS.

## User execution model

I execute project work by default.

Use connected AWS/GitHub/web/Kaggle read-only inspection aggressively when it helps diagnosis. Do not ask me to manually provide information already exposed by connected services or attached return bundles.

Do not make cost-incurring or state-changing AWS/Kaggle/GitHub changes unless explicitly authorized for that milestone.

## One handoff artifact + one command

For each bounded execution milestone:

- provide one self-contained `.pyz` when technically appropriate;
- I upload it to `/home/sagemaker-user`;
- provide **exactly one executable code block** containing only the real terminal command I should run;
- implementation complexity belongs inside the `.pyz`, not in many manual files/commands;
- every substantial runner must self-test, preflight, gate, checkpoint, emit heartbeats, stop on failure, and attempt one compact return ZIP.

Never use the phrase **“you can”** in project responses. Give direct instructions.

Test everything reasonably testable before handoff and clearly separate:

- what ChatGPT actually executed;
- what was statically inspected;
- what was mocked/fixture-based;
- what still depends on AWS/Kaggle/GPU/data.

## Resumability

Never recompute expensive completed stages solely because a downstream step failed.

Persist and reuse valid checkpoints, inference results, graph evaluations, model exports, and qualification receipts.

## Cost discipline

Every execution response must state:

- live/assumed instance;
- hourly rate;
- expected runtime range;
- expected direct command-time cost;
- hard runtime/cost ceiling;
- required RAM/GPU/disk;
- exact stop conditions.

---

# 3. Current live AWS state — latest verified check

Fresh authenticated read-only AWS inspection at approximately **2026-09-29 10:48 a.m. Los Angeles time**:

- Biohub JupyterLab app: **InService**
- current instance: **`ml.g6.4xlarge`**
- SageMaker image: GPU distribution 4.5.0
- remote access: disabled
- no infrastructure mutation was performed by ChatGPT

Fresh authenticated SageMaker JupyterLab price in `us-west-2`:

- **`ml.g6.4xlarge` = $1.654/hour**

This is now the correct hardware/cost basis unless a new live AWS read shows otherwise.

Important historical correction:

- I originally preferred `ml.g6e.2xlarge`;
- g6e capacity repeatedly failed;
- an older deployment runner also incorrectly hard-gated on one 48 GB L40S and rejected the actual L4;
- that L40S-only gate was a delivery bug, not a scientific failure;
- I am now on **g6.4xlarge because g6e capacity was unavailable**.

Do not carry forward stale g6e assumptions.

---

# 4. Deadline

Competition deadline:

- **September 29, 2026, 23:59 UTC**
- **4:59 p.m. Los Angeles time**

At the time of this handoff it is roughly late morning in Los Angeles, so time is critical.

Do not open another long training direction unless there is decisive evidence that it can beat the baseline within the remaining window.

---

# 5. Official Kaggle submission state — current source of truth

The latest attached return is:

`biohub_deadline_recovery_return(1).zip`

It performed a **fresh authenticated Kaggle submission read** and is the latest source of truth for official results.

## Submission 56376695 — preserve this baseline

- status: **COMPLETE**
- public score: **0.947**
- description: unchanged Harmonic reference
- this is the current official best
- **preserve this submission for final judging**

## Submission 56659462 — first locally qualified dynamic candidate

- status: **COMPLETE**
- public score: **0.946**
- description: `biohub-strict-winner-2f7ac937d458 | original 0.947 + qualified strict dynamic policy; saved version 1`
- this is an **official regression** relative to 0.947

## Submission 56503335 — image/division probe

- status: **COMPLETE**
- public score: **0.946**
- also worse than 0.947

Therefore:

- official best is still **0.947**;
- official improvements beyond 0.947: **0**;
- do not pretend the local 0.948/0.953 results established leaderboard progress;
- do not auto-submit the dependent successor just because it passed local retrospective gates.

The latest deadline-recovery decision is:

`confirmed_official_regression_review`

and explicitly says:

- preserve baseline submission **56376695**;
- `successor_automatic_launch_allowed = false`;
- `new_submission_allowed = false` under that audit state;
- reported adverse official evidence supersedes retrospective local qualification.

---

# 6. Latest deadline-recovery run

Run ID:

`20260929T171044Z_fdbfc124`

Runner:

`biohub-g6-deadline-review-20260929-1`

Result:

- status: **inconclusive**
- decision: **confirmed_official_regression_review**
- elapsed: **36.78 seconds**
- recorded command-time cost: **$0.016898**
- new fits: **0**
- new inference: **0**
- Kaggle writes: **0**
- scientific workflows added: **0**

Stages that passed:

1. CPU resource budget
2. packaged regressions
3. freshness/current-state check
4. fresh authenticated Kaggle score read
5. baseline-prediction fidelity audit
6. executed AWS evidence notebook

Its package self-tests:

- **46 tests passed**
- zero failures/errors

AWS report notebook:

- saved-output validation passed
- Plotly persisted after reopen
- 1080 × 620 figure dimensions
- no notebook errors

---

# 7. Critical new blocker discovered by the deadline audit

The recovery audit verified both prediction CSV files by hash:

## Original 0.947 preview CSV

Path:

`/home/sagemaker-user/biohub-cell-tracking-during-development/outputs/public_frontier_reproduction/exact_0947/our_exact_output/submission.csv`

SHA-256:

`a69c78229c6556d06d5fe9ff050074254b4a326e83249efbf578b843aec7a848`

Hash matched the recorded reference.

## Submitted candidate preview CSV

Path:

`/home/sagemaker-user/biohub-cell-tracking-during-development/outputs/competitive_gap_closure/harmonic_submission_finish_v2/verified_output_v1/submission.csv`

SHA-256:

`0e36b2eec19bebc17189331160b5f76ceb595f1db5a7fc6cdc66c6a81da599c8`

Hash matched its recorded submission artifact.

## Baseline executable-source mismatch

The audit found the local baseline notebook/code identity no longer matches the recorded reference:

- observed code SHA-256: `55b2b8847bab51b42047f703f71274d058a45ca5f7f9fc357a5c567361ad75da`
- expected recorded baseline source SHA-256: `93de46c8b61ac237349d444fc849c153b5273abab3e1353cb061997986abf523`

Audit status:

`blocked_missing_or_changed_input`

Issue:

`baseline_notebook: original executable code differs from recorded reference`

This means the latest deadline audit could **not** finish a clean source-level baseline-fidelity comparison.

Do not silently “fix” this by overwriting files.

The next chat should determine whether:

1. the 0.947 CSV itself is still a trustworthy immutable reference and the changed notebook is merely an unrelated later copy;
2. the candidate submission altered only intended associations while preserving detections/coordinates;
3. there was unintended baseline/detection drift in the final submission path;
4. the official 0.946 regression is primarily a generalization failure of the local correction or an implementation-fidelity problem.

This is now more important than another small local metric sweep.

---

# 8. Important historical deployment state

Before the fresh deadline recovery, the successor delivery path had historical state:

- final successor private input creation acknowledged:
  `alvaromendizabal/biohub-successor-58c949ac214f3e32`
- historical loader/notebook SHA:
  `81ed844a239305aa18cd8e1fab2f523ba88723b1a1e33eab1a35d94c77389849`
- no acknowledged successor final notebook push in that persistent state
- no acknowledged successor competition submission

Treat those as **historical events**, not instructions to replay them.

The deadline recovery did no remote writes.

---

# 9. Public baseline and solution audit history

## Harmonic Fusion reproduction

The public Harmonic Fusion saved-notebook system was reproduced and officially scored **0.947**.

This remains the official reference system and floor.

## Duplicate 0.947 audit

A second apparent 0.947 public lineage was audited and found to be effectively the same system/output, so it provides no useful independent ensemble diversity.

Closed.

## Lower public lineages / organizer mechanisms

We inspected lower-scoring public systems around 0.941/0.944 and organizer/public mechanisms including:

- TemporalUNet3D
- SimpleNodeTransformer
- DeepCenter
- TTA
- ILP/global assignment
- learned motion linking
- division handling
- gap repair
- graph postprocessing

These informed later experiments but did not independently beat 0.947.

---

# 10. Major experiment history

## A. Synthetic lineage/division experiments

Reproduced public synthetic lineage/division approaches and tested extra division logic.

Some local division recovery occurred, but no robust exact full-graph promotion.

Closed as a standalone route.

## B. Real sparse fine-tuning

Adapted on real sparse annotations.

Real validation AP saturated at 1.0 for source and adapted models.

Exact graph transfer:

- score delta: `0.0`
- division TP gain: `0`

Valid negative result. Closed.

## C. Integrated three-frame temporal linker

Added real sparse all-edge supervision and a three-frame temporal linker while freezing detector/U-Net.

This produced the first meaningful complementary graph behavior and the “pilot” linker/checkpoint used later.

## D. Dense affinity relay

Captured pre-threshold dense affinities from source and pilot linkers and fed them into global assignment.

Recovered useful division signal but lost trusted source edges.

Exact full-graph promotion failed.

## E. Source-preserving division grafting

Best local arm reached roughly **0.945073578** on the five-movie exact development metric.

Recovered two division TPs but edge TP dropped below the stronger historical floor.

Not promoted.

## F. Cross-graph division fusion

Initial implementation incorrectly assumed graph identifiers aligned.

Diagnosed shifted/refined identifiers and built deterministic same-frame physical registration using anisotropic micrometre coordinates and one-to-one ambiguity gates.

Corrected system generated material fusion variants, but exact score did not improve.

Historical-anchor transaction grafting closed.

## G. Grouped higher-order tracklet transformer

Moved away from final-graph hand-written transactions.

Jointly ranked candidate parents using:

- source/pilot affinities
- displacement
- predecessor/successor context
- motion/acceleration
- candidate rank and margin
- topology
- disagreement
- no-match context

First two held-out folds improved parent F1:

- Fold A: about `+0.011019`
- Fold B: about `+0.008322`
- mean gain: about `+0.009670`

This narrowly missed the +0.010 screening gate.

Valid scientific stop, not execution failure.

## H. Cross-fitted Tracklet Mixture of Experts

Completed OOF coverage across six real movies.

Compared:

- precision gate
- linear blend
- logistic candidate ranker
- histogram-gradient-boosted candidate ranker

Selected family:

`histgb_ranker`

Six-movie OOF parent-F1 gain:

`+0.008985834375263568`

Precision delta:

`+0.000801829205206217`

All six held-out movies improved.

Embryo gains:

- `44b6`: `+0.007642892551320557`
- `6bba`: `+0.010519685837789394`

This still missed the strict +0.010 OOF line by roughly 0.001014.

No automatic promotion.

## I. Exact MOE bridge / ID alignment failure

A zero-fit exact confirmation bridge was built to evaluate the trained MOE on five development movies.

It originally failed with:

`TRACKLET_ALIGNMENT_NODE_TIME moe_apply:44b6_12dfb391`

Root cause:

runtime graph node IDs and dense-sidecar detector IDs did not preserve identical ID→frame semantics.

We corrected this class of issue using same-frame coordinate registration and one-to-one ambiguity control instead of trusting integer IDs.

## J. Native metric / MOE topology evaluation

The complete MOE system and native metric were tested.

Standalone MOE exact result did not justify replacing Harmonic, but its graph later became useful as evidence for one conservative branch suppression when anchored onto the stronger full system.

## K. Image-conditioned reparenting/division probe

Built a fixed image-conditioned parent-ranking/division probe.

Local preservation gates accepted no useful corrections.

Its eventual official submission **56503335 scored 0.946**, confirming no official improvement.

Closed.

## L. Full Harmonic integration of completed work

This was a major correction in experimental framing.

Instead of comparing candidate systems only against weaker component baselines, we replayed **37 saved policies plus two controlled combinations against the original Harmonic development control**.

Original local Harmonic control:

`0.9330456734289274`

Selected locally qualified candidate:

`0.9480594130753609`

Local gain:

`+0.015013739646433488`

Counts:

- edge TP: 2391 → 2392
- edge FP: 111 → 110
- edge FN: 112 → 111
- division TP: 1 → 2
- division FP: 1 → 1
- division FN: 5 → 4

The all-conflict-free combination was substantially worse and lost many correct edges.

Lesson:

**conservative source-preserving integration beat indiscriminate aggregation.**

This candidate was later submitted as **56659462 and officially scored 0.946**, so the local gain did **not** generalize to the public leaderboard.

This is now important adverse evidence.

## M. Harmonic fastlane / runtime optimization

A parity-checked motion-linking optimization reduced measured component runtime from about 80.5 seconds to 2.32 seconds, roughly **34.7×** for that component.

Do not claim 34.7× whole-notebook speedup.

Also removed repeated validation/reselection and reused compatible encoder features.

## N. Qualified successor research

After candidate one, four fixed source-preserving successor policies were tested against both:

- original Harmonic control;
- first qualified candidate.

Three tied the first candidate.

One, `moe_fork_suppression`, improved local exact score to:

`0.95317334233363`

versus first candidate local:

`0.9480594130753609`

Incremental local gain:

`+0.005113929258269145`

Counts:

- edge TP: 2392
- edge FP: 109
- edge FN: 111
- division TP: 2
- division FP: 0
- division FN: 4

But this incremental gain comes from **one branch removal on one development movie**.

Leaving out that changed movie removes the incremental advantage.

This is weak generalization evidence and should not override the official regression of its parent candidate.

## O. Successor dynamic deployment qualification

The successor deployment initially failed because replay used a compact graph snapshot instead of native GEFF ordering/confidence inputs.

That omitted original edge-confidence values and changed node iteration order.

The native input path was corrected.

After correction:

- all five movie replays passed;
- complete MOE donor graphs matched expected graphs;
- complete successor graphs matched expected graphs;
- all recorded model-probability and ranker comparisons were exact (0 error);
- final successor inference assets were staged in AWS;
- no new model fitting was performed.

A later successor final-input upload was acknowledged, but final inference/submission remained paused.

## P. Official regression review / deadline recovery

Fresh authenticated Kaggle read confirmed:

- 0.947 baseline = COMPLETE
- candidate one = 0.946 COMPLETE
- image probe = 0.946 COMPLETE

Automatic successor launch was stopped.

The audit then found the recorded baseline executable code hash mismatch described earlier.

This is the **current technical decision point**.

---

# 11. Critical reusable model artifacts — do not retrain casually

Grouped Tracklet MOE output root:

`/home/sagemaker-user/biohub-cell-tracking-during-development/outputs/competitive_gap_closure/grouped_tracklet_moe`

Final selected stacker:

`.../grouped_tracklet_moe/final_stacker/stacker.pkl`

SHA-256:

`ef98b89d5a5a703a08d8cb38d90f4f4757dd5bc41466e264bf7ec302ce6f4aba`

OOF records:

`.../grouped_tracklet_moe/oof/oof_records.pt`

records:

`2174`

SHA-256:

`778c85b950d3309aa382bf463463a221a342b3741328e7cfc147db3c74fa907c`

Four neural fold checkpoints:

### Fold A

`.../grouped_tracklet_transformer/folds/fold_a/tracklet_model.pt`

SHA:

`7de326601c346fe4c1b060004c61e77af3ca7f27262f66bfe7d3aac0bb131a2e`

### Fold B

`.../grouped_tracklet_transformer/folds/fold_b/tracklet_model.pt`

SHA:

`8ab3339bbb4d18bb2a18f25fb09ac4112d4c6106fc08108f563ab7612268127b`

### Fold C

`.../grouped_tracklet_moe/oof_base/folds/fold_c_aaf8/tracklet_model.pt`

SHA:

`690d3c55368cec347b0c1c51dcfcd07fed47503b45cc8ba2bbdb8a9db76670ca`

### Fold D

`.../grouped_tracklet_moe/oof_base/folds/fold_d_d2f/tracklet_model.pt`

SHA:

`4a420370fd6ee5a9dcfdec0cdda3bee5aaf7f527431fa48b5658b2b5f79f79b0`

Do not retrain these checkpoints unless evidence shows they are invalid.

---

# 12. Important AWS output roots

Original exact 0.947 reproduction/output:

`outputs/public_frontier_reproduction/exact_0947`

Candidate-one integration/deployment roots used during the recent work include:

- `outputs/competitive_gap_closure/harmonic_strict_winner_deploy_v1`
- `outputs/competitive_gap_closure/harmonic_delivery_resume_v1`
- `outputs/competitive_gap_closure/harmonic_submission_finish_v2`

Successor research/deployment roots include:

- `outputs/competitive_gap_closure/harmonic_successor_v1`
- `outputs/competitive_gap_closure/harmonic_successor_deploy_v1`
- `outputs/competitive_gap_closure/successor_final_submission_v1`

Do not overwrite these. Inspect actual current files before relying on historical path names.

---

# 13. Latest official score evidence from the deadline recovery

Fresh authenticated score inventory at:

`2026-09-29T17:10:58Z`

Contained exactly these relevant records:

1. `56376695` — Harmonic reference — **0.947 COMPLETE**
2. `56503335` — image/division probe — **0.946 COMPLETE**
3. `56659462` — locally qualified strict dynamic candidate — **0.946 COMPLETE**

Therefore:

**Official improvements beyond 0.947 = 0.**

Another candidate must be justified against this adverse evidence.

---

# 14. Current running tally

Use the latest verified scientific ledger:

- **45 completed scientific workflows**
- **38 historical hard execution failures**
- **10 correctly gated scientific stops**
- **1 separately tracked remote-service block**
- **94 research return bundles**

Keep delivery/infrastructure failures separate from scientific outcomes.

Relevant recent delivery/deployment events:

- one successful successor deployment qualification;
- one earlier failed successor deployment attempt;
- one L40S-only environment rejection on the L4, which was a delivery-environment bug, not a scientific result;
- latest deadline-recovery run was **inconclusive**, not a scientific workflow.

Do not inflate the scientific success count with unit-test suites, packaging tests, or status-only submission reads.

---

# 15. GitHub / employer-facing portfolio state

The public repository was updated after the qualified Harmonic integration work:

Repository:

`https://github.com/alvaromendizabal/biohub-cell-tracking-during-development`

PR #7 was squash-merged to `main`.

Merge commit:

`2f151b95c31e79b5d0a98171a0b094b463689bea`

Public portfolio now documents:

- official 0.947 reference;
- local 0.948059 source-preserving integration as retrospective evidence;
- 34.7× motion-linker component optimization;
- deployment engineering;
- semi-reproducibility boundary;
- negative-result discipline;
- private competition implementation intentionally withheld.

Files updated included:

- `README.md`
- `RESULTS.md`
- `CASE_STUDY.md`
- `FRONTIER.md`
- `REPRODUCIBILITY.md`
- `ATTRIBUTION.md`
- `notebooks/portfolio.ipynb`
- `scripts/check_showcase.py`
- `PUBLICATION_MANIFEST.json`

GitHub CI passed on the PR and after merge.

Do not push another Git milestone unless I explicitly ask.

---

# 16. What is closed / what not to repeat

Do not spend time on:

- reproducing Harmonic 0.947 again;
- blending the duplicate 0.947 lineage;
- standalone synthetic-division tuning;
- standalone real-sparse fine-tuning;
- historical-anchor graph-transaction grafting;
- point-only dense parental model line;
- image-conditioned reparenting/division probe;
- indiscriminate combination of all historical graph edits;
- retraining the four MOE fold models or stacker without evidence they are invalid;
- tiny postprocessing threshold sweeps on the same repeatedly inspected five movies;
- treating local 0.948/0.953 as evidence that a new submission is automatically better;
- creating more Kaggle development notebooks;
- rerunning completed successor deployment qualification;
- replaying the acknowledged successor input upload solely because a later stage failed.

---

# 17. Why the project failed to beat 0.947 so far

This must be stated plainly in future reasoning.

We produced real engineering progress, richer temporal representations, a functioning MOE, exact graph registration, native metric reconciliation, source-preserving integration, and deployment parity.

But **the selection evidence was too dependent on a tiny repeatedly inspected development cohort**.

The candidate one local improvement included one recovered annotated division and looked strong under local exact scoring, yet officially regressed to 0.946.

The successor’s additional local improvement is even narrower: one removed false branch on one development movie.

Therefore the main problem now is **generalization/evidence quality**, not a lack of another micro-adjustment.

There is also an unresolved **baseline/source-fidelity question** because the audit found the current baseline executable code hash differs from the recorded 0.947 reference code hash.

Until that is explained, another dependent submission risks wasting the remaining deadline and another slot.

---

# 18. Immediate next task for the new chat

Do not give me a generic roadmap.

First inspect:

1. the latest attached `biohub_deadline_recovery_return(1).zip`;
2. live AWS read-only state;
3. current Kaggle scores/leaderboard read-only if available;
4. the exact baseline notebook/source artifacts and hashes in AWS;
5. the 0.947 preview CSV and 0.946 candidate preview CSV;
6. any complete cached output from the successor, without launching or submitting it.

Then answer these questions:

## A. Is the 0.947 baseline source actually corrupted/drifted, or is the audit comparing the wrong notebook copy?

Locate the true executable source associated with submission 56376695.

Establish whether the expected code hash `93de46...` refers to the actual scored baseline or an obsolete local snapshot.

Do not overwrite anything while diagnosing.

## B. What changed from the 0.947 predictions to the 0.946 submission?

Using existing prediction CSVs only, compare:

- movie coverage;
- node count;
- coordinates;
- temporal frames;
- edge count;
- division count;
- candidate-one edits;
- whether nodes/detections are geometrically identical after deterministic same-time registration;
- whether differences are only intended association changes or include unintended detection/coordinate drift.

This comparison must be prediction-only; do not use hidden labels.

## C. Is there evidence that the 0.946 official regression came from an implementation-fidelity bug?

If yes, identify the smallest correction that restores exact 0.947 baseline behavior before layering any new correction.

Require byte/hash/graph parity with the known scored baseline wherever possible.

If no—if detections and baseline behavior are faithful and only the intended association corrections changed—treat the regression as a **generalization failure** of the local correction.

## D. Should the 0.953 successor be submitted today?

Do not answer “yes” merely because 0.953 > 0.948 locally.

The successor depends on the same development cohort and its incremental gain comes from one movie.

Only advance it if the new evidence meaningfully addresses the 0.946 official failure, e.g.:

- a discovered fidelity bug is fixed and baseline parity is re-established;
- or independent/non-overfit evidence clearly supports the successor mechanism;
- and final prediction-time execution is verified without altering the baseline.

Otherwise preserve 0.947 and do not burn a final submission on a candidate whose evidence is not stronger than the failed first candidate.

---

# 19. If a new execution milestone is justified

The next runner should be the **cheapest decisive AWS-only experiment** that resolves the baseline-fidelity/generalization question.

Requirements:

- CPU-only unless GPU is genuinely necessary;
- no new training by default;
- use existing preview/submission artifacts and graph caches;
- self-gate before expensive work;
- compare exact hashes and graph invariants;
- checkpoint/resume;
- produce one compact return ZIP on pass/fail;
- execute an employer-facing AWS notebook with persistent Plotly only if it materially documents the result;
- zero Kaggle writes unless I explicitly authorize a final submission after evidence passes;
- no AWS infrastructure changes without explicit authorization.

Promotion condition for another submission must be based on evidence addressing the failed official result—not another retrospective development score alone.

Kill condition:

If baseline fidelity is correct and no independent evidence supports the successor beyond the same inspected cohort, preserve the official 0.947 and stop submitting speculative variants.

---

# 20. Current cost basis

Current live instance:

`ml.g6.4xlarge`

Authenticated SageMaker JupyterLab us-west-2 price:

**$1.654/hour**

Use this rate until a new live AWS read says otherwise.

For quick CPU audits:

- 1 minute ≈ $0.0276
- 3 minutes ≈ $0.0827
- 5 minutes ≈ $0.1378
- 10 minutes ≈ $0.2757
- 20 minutes ≈ $0.5513

These are active instance-time estimates, not account-wide billing caps. Startup, idle time, EBS, transfers, and unrelated services are separate.

---

# 21. Required response structure in the new chat

Every substantial project response should include:

## Project status

- Where we are
- Current best
- Target/gap
- What changed
- Where we are going

## Running tally

- successful scientific workflows
- hard execution failures
- valid scientific stops
- delivery/service failures separately where useful

## This milestone

- objective
- new capability or decisive diagnostic
- why it should move the metric or resolve uncertainty
- promotion threshold
- kill/stop threshold

## Test evidence

- what ChatGPT actually tested
- what was mocked/static
- what remains AWS/Kaggle/environment-dependent

## Execution budget

- expected runtime
- hardware assumptions
- estimated cost
- hard stop

## Handoff

- one `.pyz`
- exactly one executable code block

## Return

- exact return ZIP path

## Next decision

- result that moves us forward
- result that kills the direction
- roadmap consequence

Do not make me repeatedly ask where we are, whether something worked, what it costs, or what comes next.

---

# 22. Most important behavioral instructions for the next assistant

- **Do not tell me an improvement is guaranteed.**
- **Do not call local 0.953 “better” in the official sense.** It is only locally stronger.
- **Do not ignore the official 0.946 regression.** It supersedes prior optimism about candidate one.
- **Do not submit more variants merely to increase submission count.**
- **Do not retrain existing valid models without evidence.**
- **Do not move development to Kaggle.**
- **Do not overwrite AWS artifacts while diagnosing source drift.**
- **Do not spend another round on a threshold sweep that cannot resolve the official failure.**
- **Do inspect current AWS and Kaggle state before giving instructions.**
- **Do preserve submission 56376695 / score 0.947.**
- **Do treat the baseline executable-code hash mismatch as unresolved until proved otherwise.**
- **Do prioritize the fastest decisive audit before the deadline.**
