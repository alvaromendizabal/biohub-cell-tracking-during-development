# Biohub AWS decommission record

Date: September 30, 2026.

## Archive gate

AWS deletion occurred only after the Git repository became the durable first-party archive.

PR #9 restored and published:

- the executable research package, configs, tests, reports, figures, and seven executed notebooks;
- late-stage execution evidence;
- immutable external asset/source pins;
- the 99-file exact readable final-sprint source snapshot;
- fresh-clone install/test/reproduction instructions.

The PR-level fresh-clone workflow passed. The squash merge commit is:

`df116766665a104d4a397cfdb39cffd2ddf4cf19`

The post-merge `main` workflow also passed before deletion began.

## SageMaker cleanup

Before cleanup:

- the Biohub JupyterLab application was already in `Deleted` state;
- the persistent Biohub SageMaker Space was still `InService`;
- prior inventory showed a 128-GB EBS volume associated with that Space.

The Space was deleted with the SageMaker API only after the archive gate passed.

Deletion verification:

- `ListSpaces` returned no matching Biohub Space;
- `DescribeSpace` returned `ResourceNotFound`.

This removes the persistent project Space rather than merely stopping compute.

## S3 cleanup

Historical project source showed that optional compact backups used a `biohub-cell-tracking/` prefix in the account's shared SageMaker bucket.

A final targeted S3 listing of that exact prefix returned:

- objects: **0**
- bytes: **0**

There was therefore no Biohub S3 object left to delete.

The shared SageMaker bucket itself was intentionally preserved because it contains unrelated projects. Other project-specific buckets in the account were also left untouched.

## Durable archive boundary

The Git repository is now the durable first-party source of truth for the completed Biohub project.

Large competition data and third-party model binaries remain reproducible from their original distribution points using the immutable versions/checksums in `reproducibility/external_assets.json`; they were not copied into Git or another private cloud bucket solely for archival.

AWS billing consoles may lag resource deletion, but there is no remaining Biohub JupyterLab app, SageMaker Space, or Biohub object under the known S3 backup prefix according to the final API verification.
