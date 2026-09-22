# Temporal Reassignment

Recorded research source from the corresponding return archive; see
[the experiment ledger](../../docs/EXPERIMENTS.md) and [source provenance](../../reports/source_provenance.json).
The `pinned_metric` directory is organizer code covered by
[BSD 3-Clause](../../licenses/royerlab-BSD-3-Clause.txt).
Synthetic tests do not establish real-data predictive performance. Worker execution
requires the private AWS workspace documented in [REPRODUCIBILITY.md](../../docs/REPRODUCIBILITY.md).

The original temporal replay worker is intentionally excluded: it depends on the full
private reference replay/cache adapter. The core temporal algorithm, candidate tests,
metric reference, and report source are included. Use the original AWS orchestrator
for full-data replay; this folder is not a standalone inference pipeline.
