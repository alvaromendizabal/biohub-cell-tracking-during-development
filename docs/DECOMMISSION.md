# Archived AWS workspace and durable public evidence

Updated October 6, 2026.

The project owner confirmed that the Biohub SageMaker workspace no longer exists.
This repository is the retained public source and evidence archive. Public tests
and the synthetic example do not require that workspace.

## Historical cleanup record

The [September 30 archival follow-up](https://github.com/alvaromendizabal/biohub-cell-tracking-during-development/pull/10)
reported that the remaining project Space was removed after the
[archive publication](https://github.com/alvaromendizabal/biohub-cell-tracking-during-development/pull/9)
merged and its checks passed. That record describes verification through both a
space listing and a not-found response from the resource lookup. It also reports
an empty project backup location and preservation of unrelated shared resources.

This page incorporates that historical record and the owner's October 6
confirmation. The current documentation refresh did not query or modify AWS and
does not independently attest to current billing or account-wide storage.

## What remains reproducible

The public package, selected experiment modules, authored fixtures, tests, compact
reports, and saved notebook evidence remain in Git. The direct test dependencies
and external source references are recorded. A reviewer can run the documented
public checks without an AWS account or competition data.

The public archive intentionally excludes private predictions, full orchestration,
raw microscopy, large checkpoints, and exact tuned settings. Reacquiring public
third-party data or weights alone does not reconstruct that private layer. No
claim is made that all historical private artifacts remain recoverable.

[Fresh-clone instructions](../REPRODUCE.md) · [Reproduction matrix](REPRODUCTION_MATRIX.md)
