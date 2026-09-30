# Publication boundary

**AWS retains:** immutable images, GEFF stores, full data inventories, predictions,
weights, virtual environments, failed/partial state, private account configuration,
and the original working tree and Git index.

**GitHub receives:** reviewed source modules, selected synthetic tests, executed
research notebooks, genuine result charts, compact metrics and status receipts,
source attribution, and reproducibility/validation documentation.

The handoff contains a fixed file allowlist with hashes. It does not use `git add .`
in the AWS research directory and does not mirror arbitrary local files. Git operations
occur only in the separate publication clone. Symlinks, credentials, model binaries,
raw data stores, and oversized files are rejected. The original AWS index and working
source files remain unchanged.

No automatic scientific rerun, competition submission, infrastructure change, or
new model fit is part of this publication. Existing account authorization is reused;
if terminal GitHub authorization is absent, the helper opens the CLI browser/device
flow rather than asking for a token in chat. A missing CLI may be installed privately
from the pinned, checksum-verified official distribution.

No blanket new open-source license is applied. Third-party rights and notices are
preserved. Original project rights remain reserved unless a file states otherwise.
