# Current state and archival status

Evidence cutoff: **September 29, 2026, 23:02 UTC**.

## Verified competition state

- Harmonic reference reproduction: **0.947**, submission 56376695.
- Locally qualified source-preserving candidate: **0.946**, submission 56659462.
- Image/division probe: **0.946**, submission 56503335.
- Exact public notebook advertised at 0.953: submission **56687425**, accepted and still pending at the captured cutoff.

The official regression is treated as stronger evidence than the local 0.948059 retrospective development score.

## Final-day audits

- baseline/candidate preview detections and coordinates were identical; eight association-edge edits differed across four movies;
- three audited public 0.953 notebook versions produced byte-identical complete prediction outputs;
- V1284 source/head inspection established a compact 224→32→3 head but did not isolate its causal leaderboard contribution;
- HOCT reached real GPU neural inference with frozen detections but did not reach a completed native-metric result before cutoff.

## Reproducible repository state

The executable September 22 research snapshot has been restored from Git history onto the archival branch: package source, research modules, configs, reports, tests, and seven executed evidence notebooks.

The root docs and final-evidence utilities capture the September 23–29 research conclusions and reproduce the saved-output fidelity/diversity claims from downloaded CSV artifacts.

## AWS decommissioned after archive verification

The Biohub JupyterLab app was already **Deleted** before archival. The remaining SageMaker Space, previously configured with a **128-GB EBS volume**, was deleted only after PR #9 merged and the post-merge fresh-clone CI passed.

Deletion was then verified two ways: the Space no longer appeared in `ListSpaces`, and `DescribeSpace` returned `ResourceNotFound`.

A targeted final S3 audit of the known Biohub backup prefix found **0 objects / 0 bytes**. The shared SageMaker bucket was not deleted because it contains unrelated projects.

See [DECOMMISSION.md](DECOMMISSION.md).
