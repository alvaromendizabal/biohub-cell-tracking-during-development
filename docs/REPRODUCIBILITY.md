# Reproduction scope after AWS archival

The original Biohub AWS workspace has been removed. The public package and its
checks run from a clean clone; historical private execution is not restored by
installing this repository. Start with the [fresh-clone guide](../REPRODUCE.md).

| Level | Available from Git | What the result establishes |
|---|---|---|
| Read historical evidence | Six archived research notebooks and one executed overview with saved PNGs | What ran at the recorded evidence cutoff |
| Run public software checks | Package, hash-locked test dependencies, synthetic tests and CSV utilities | Software behavior on authored fixtures |
| Run the tracking demo | Generated synthetic microscopy, Python constrained solving and a live browser optimizer | End-to-end synthetic tracking behavior and exact-ID graph diagnostics |
| Run the five-cell example | Authored cells/candidate scores and the constrained solver | A small inspectable graph decision; no model inference |
| Repeat a saved-output audit | Comparison utility plus separately acquired CSVs | Differences in supplied outputs; authenticity needs matching source hashes |
| Reconstruct complete research | Requires external data/weights and private runtime integration/manifests | Outside the public fresh-clone guarantee |

## Historical notebooks

The overview reads committed summaries and images. The six research notebooks are
archived AWS executions: their original private workspace references describe the
historical environment. Their outputs remain unchanged. `portfolio.ipynb` is a
separate presentation-only notebook and is not counted as an executed experiment.

Saved outputs support inspection. They do not mean that a fresh clone can rerun
all original cells without the missing private inputs and runtime.

## Test coverage

The lightweight suite checks selected feature/graph behavior, input contracts,
publication scope, Python syntax, and notebook output integrity. Optional neural
components require additional dependencies; CI does not train them or repeat
competition inference. The numerical-recovery evidence records a finite smoke test
followed by a nonfinite full training attempt, not complete training reproduction.

## External and private inputs

Immutable references in [external_assets.json](../reproducibility/external_assets.json)
identify public sources and assets. Access and redistribution terms still apply.
Private predictions, candidate graphs, exact tuned settings, cached tensors, and
operational adapters are intentionally excluded. Their recovery or availability
has not been verified by this publication update.

The original temporal replay worker depended on a private reference adapter; its
public core algorithm and tests do not replace that integration. The archived
image-appearance worker was blocked at its coordinate contract and is not a
completed biological result.

See the [archive record](DECOMMISSION.md) and [publication boundary](PUBLICATION_SCOPE.md).
