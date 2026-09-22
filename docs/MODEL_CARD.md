# Model and system card

## Intended use
Research on 3D time-lapse cell detections and lineage reconstruction under sparse
annotations. This is a competition/research system, not a validated clinical tool.

## Frozen scored system
External TemporalUNet3D detection/feature extraction and SimpleNodeTransformer
association, dual-seed and forward/reverse harmonic fusion, structured association,
motion relinking, gap repair, pruning, division repair, and DeepCenter assistance.
The exact scored notebook is `alvaromendizabal/biohub-harmonic-fusion-exact-repro`,
version 1, submission 56376695, official public score **0.947**.

The primary public checkpoint fingerprint recorded in project evidence is
`12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`;
DeepCenter is `8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0`.
No weights are distributed in this repository.

## Evaluation
The pinned organizer evaluator matches detections in physical space, evaluates
edges under sparse annotation, applies the documented node-count adjustment, and
scores divisions using local topology. Local five-movie comparisons use the same
saved frozen predictions and the same evaluator. See the exact `pinned_metric`
source, its upstream citation in the notices, and the individual experiment receipts.

The five-movie development score **0.933046** and the small-edge-table **0.975**
diagnostic are not substitutes for the **0.947** official public score. Repeated
inspection, only two embryos, few division events, and unverified checkpoint training
independence limit generalization claims. There is no independent untouched validation
claim for the recent exploratory corrections.

## Research candidates
Temporal reassignment accepted no exchanges. Competing-parent division learning
added 108 forks without a scored benefit. Neither replaced the official reference.
The image-appearance classifier has not fitted project data due to a coordinate-bound
failure. Its synthetic tests establish invariants, not real-data predictive performance.

## Training limitations
The organizer smoke generated a checkpoint, but the separate full three-epoch attempt
became numerically invalid. The invalid checkpoint was quarantined. The finite smoke
checkpoint is not the scored reference, and exact optimizer-state resume was not available.
Independent reproduction of the leader's complete training procedure is not claimed.
