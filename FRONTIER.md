# Research frontier

## Current question

The project has reached a clear boundary: increasingly capable post-hoc division classifiers changed candidate probabilities and rankings but did not improve the exact tracking graph.

The active question is therefore:

**Can the temporal linker itself be improved using sparse real continuation and division supervision, while preserving the strong pretrained detector and visual encoder?**

## Why this is the next layer

The frozen development reference contains:

- 2,391 correct edges;
- 111 incorrect edges;
- 112 missed edges;
- 1 correct division;
- 1 false division;
- 5 missed divisions.

That error budget is much broader than a rare-event division classifier can address. A linker trained over all annotated temporal edges can, in principle, affect both continuation mistakes and division structure.

## Current model direction

The public baseline architecture uses:

- a temporal 3D U-Net for image representation and detection;
- node features indexed from the image representation;
- positional features;
- a bidirectional cross-attention transformer;
- pairwise edge scoring between adjacent frames.

The active AWS experiment keeps the public visual encoder and detector fixed, caches their outputs, and fine-tunes the edge linker over multi-frame windows using sparse real edge supervision.

## Research controls

A frontier result is not promoted because training loss falls.

The current lane requires:

1. a held-out validation improvement;
2. comparison against the unchanged source linker;
3. exact full-graph evaluation on the fixed retrospective development cohort;
4. preservation of correct edges and lineage structure;
5. an independent official confirmation step before any stronger performance claim.

## What is deliberately not public here

This repository does not publish:

- model weights;
- active AWS code;
- exact data-acquisition recipes;
- private caches;
- exact feature recipes;
- account credentials or competition state;
- executable submission logic.

The purpose of this document is to show the research reasoning and engineering boundary, not to publish an active competition system.

## Status

As of the current evidence cutoff, the integrated-linker run is still acquiring and caching the real sparse training material in AWS. Rate-limit interruptions are handled by resumable object-level checkpoints.

**No score improvement is claimed until training, held-out validation, full development evaluation, and official confirmation are complete.**
