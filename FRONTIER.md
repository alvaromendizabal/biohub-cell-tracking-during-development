# Research frontier

## Current question

The current research question is no longer whether to train another point-only or image-conditioned reparenting model.

A source-preserving candidate has already cleared the local full-graph promotion gate against the original Harmonic development control.

The active question is now:

**Does the locally qualifying full-baseline candidate reproduce its prediction-time behavior on unseen competition movies and improve the verified 0.947 public score?**

## Why this is the right next layer

The selected candidate improved the retrospective five-movie exact score from **0.933046 to 0.948059**, while increasing correct edges and correct divisions and reducing false edges without adding a false division.

That result is stronger than the earlier proxy improvements because it was measured against the original full Harmonic control. It is still not independent evidence: the development movies have been inspected repeatedly and the candidate was selected among multiple policies.

The only useful next evidence is therefore **deployment on unseen test movies followed by an actual saved-notebook score**.

## Current deployment direction

The deployment system keeps the scored Harmonic detector, image models, association logic, post-processing family, and CSV contract as the anchor.

The qualifying correction policy is reconstructed dynamically from prediction-time artifacts. No development movie IDs, annotations, or hard-coded graph edits are embedded into the test-time logic.

The additional learned association head reuses compatible frozen image features instead of running a second image encoder. Delivery also preserves the previously verified faster motion-linking implementation and skips repeated validation/reselection work that is no longer necessary once the baseline configuration is frozen.

Remote delivery is stateful and idempotent:

1. verify the qualified local deployment contract;
2. verify the private inference assets;
3. launch one private saved notebook;
4. return while remote inference runs;
5. retrieve and validate the completed pinned output;
6. submit that verified notebook version once.

At the current evidence cutoff, private inference-input creation has been acknowledged. Private notebook completion and official submission remain pending.

## Promotion controls

The candidate has already passed the retrospective local promotion gate. No stronger official claim is made until the saved notebook:

1. runs on unseen competition inputs;
2. produces complete, valid tracking output;
3. preserves the expected inference/configuration identities;
4. passes topology and output-integrity checks;
5. is submitted as the verified pinned notebook version;
6. receives an official Kaggle score.

## If the official score does not improve

A non-improving official result would close this specific source-preserving integration candidate despite its retrospective gain.

The next branch should then be structurally different—such as a more deeply joint image/linker representation or independent supervision—not another small threshold sweep on the same development cohort.

## What remains private

This repository does not publish model weights, active AWS code, tuned thresholds, exact feature recipes, working caches, candidate graph files, private data, account credentials, or executable submission logic.

The public goal is to expose enough methodology, evidence, attribution, and validation structure to assess the engineering work without publishing the competition-specific implementation.

## Status

**Local full-baseline candidate qualified; official score unchanged at 0.947; private deployment in progress.**
