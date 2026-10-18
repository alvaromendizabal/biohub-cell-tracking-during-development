# Source-linked public review

```bash
python3 -B -S review/run.py --output /tmp/biohub-review --with-node
python3 -B -S review/run.py --output /tmp/biohub-review --with-node --check
```

Python 3.11+ is required; the optional Node checks require Node.js 22 or 24. The bundle records source hashes and actual check outcomes. Reuse verifies existing output rather than silently overwriting it.

The review checks original aggregate evidence, saved-output fidelity and optionally the actual graph engine and browser interface. It does not run private inference or produce a new competitive result. [Run the CPU pipeline](../REPRODUCE.md) · [Case study](../CASE_STUDY.md) · [Fusion contract](../src/biohub_tracking/fusion.py)

`project.json` binds review cards to maintained source and evidence. `run.py --render` regenerates the inspection page; `--check-page` verifies it. Changed inputs, changed outputs, missing checks and invalid paths are rejected.
