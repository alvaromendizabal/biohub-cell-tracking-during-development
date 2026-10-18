# Engineering overview

The maintained release connects [credited volumetric components](../src/biohub_tracking/public_reproduction/models.py), a [numerically tested fusion kernel](../src/biohub_tracking/fusion.py), [constrained association](../src/biohub_tracking/public_graph_methods.py), [CSV fidelity auditing](../scripts/reproduce_final_evidence.py), and a [runnable synthetic pipeline](../src/biohub_tracking/synthetic_demo.py).

CI installs hash-locked CPU dependencies, runs every retained test, regenerates the microscope fixture twice and exercises the actual browser handlers. The graph engine is compared with Python policies and exhaustive small-graph optima. Gap closing is a separate pass, not a claim of global temporal optimality.

The [executed example](../examples/lineage_reasoning.ipynb) shows why valid topology alone does not establish correct identities. Synthetic exact-ID metrics are not organizer-native or biological performance measurements. The retained **0.947** external score and **34.7×** motion-linking component benchmark remain historical evidence, unchanged by this release.

[Reproduction commands](../REPRODUCE.md) · [Final results](../RESULTS.md) · [Contribution boundary](../ATTRIBUTION.md)
