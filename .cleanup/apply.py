"""One-time reviewed publication transformation; removed before merge."""
from pathlib import Path
import json
import re
import shutil
import subprocess

ROOT = Path.cwd()
BASE = "6c62b00da3ece53ca249462ea854e474fc8ddd48"
if subprocess.check_output(["git", "rev-parse", "origin/main"], text=True).strip() != BASE:
    raise RuntimeError("Main advanced; reconcile before changing the publication")

def write(name, text):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def delete(name):
    path = ROOT / name
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()

def replace(name, old, new):
    path = ROOT / name
    text = path.read_text()
    if old not in text:
        raise ValueError("Expected source not found: " + name)
    path.write_text(text.replace(old, new))

for name in (
    "research", "configs", "notebooks", "assets", "FRONTIER.md", "REPRODUCIBILITY.md",
    "CHANGELOG.md", "docs/EXPERIMENTS.md", "docs/STATUS.md", "docs/evidence",
    "reports/anchored_association", "reports/appearance_association",
    "reports/competing_parent_events", "reports/division_retention", "reports/exact_error_budget",
    "reports/numerical_recovery", "reports/temporal_reassignment",
):
    delete(name)
allowed = {
    "__init__.py", "public_graph_methods.py", "synthetic_demo.py",
    "public_reproduction/__init__.py", "public_reproduction/models.py",
    "public_reproduction/losses.py", "public_reproduction/utils.py",
    "public_reproduction/augmentations.py",
}
for path in (ROOT / "src/biohub_tracking").rglob("*.py"):
    if path.relative_to(ROOT / "src/biohub_tracking").as_posix() not in allowed:
        path.unlink()
for name in (
    "tests/feature_fixtures.py", "tests/test_feature_round1.py", "tests/test_feature_round2.py",
    "tests/test_pilot_contract.py", "tests/test_pilot_features.py",
):
    delete(name)
for path in sorted(ROOT.rglob("*"), reverse=True):
    if ".git" not in path.parts and path.is_dir() and not any(path.iterdir()):
        path.rmdir()

write("src/biohub_tracking/fusion.py", '''"""Numerical contract extracted from the retained harmonic integration.

The caller supplies the reverse-evidence weight; no private tuning is encoded.
The reciprocal form matches (1+w)/(1/forward+w/reverse).
"""
from __future__ import annotations

import math

import numpy as np


def weighted_harmonic_probability(
    forward: np.ndarray, reverse: np.ndarray, *, weight: float
) -> np.ndarray:
    """Fuse aligned probabilities; reject malformed evidence before arithmetic."""
    pf = np.asarray(forward, dtype=np.float64)
    pr = np.asarray(reverse, dtype=np.float64)
    w = float(weight)
    if pf.shape != pr.shape or pf.size == 0:
        raise ValueError("probabilities must be nonempty and identically shaped")
    if not math.isfinite(w) or w < 0:
        raise ValueError("weight must be finite and non-negative")
    if not np.isfinite(pf).all() or not np.isfinite(pr).all():
        raise ValueError("probabilities must be finite")
    if ((pf < 0) | (pf > 1) | (pr < 0) | (pr > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    pf = np.clip(pf, 1e-12, 1.0)
    pr = np.clip(pr, 1e-12, 1.0)
    forward_mass = 1.0 / (1.0 + w)
    reverse_mass = w / (1.0 + w)
    return 1.0 / (forward_mass / pf + reverse_mass / pr)
''')
write("tests/test_fusion.py", '''"""Numerical parity and invalid-input checks; no private data or fitted model."""
import numpy as np
import pytest

from biohub_tracking.fusion import weighted_harmonic_probability


@pytest.mark.parametrize("weight", [0.0, 0.1, 1.0, 5.0, 100.0])
def test_matches_original_reciprocal_form(weight):
    forward = np.array([0.0, 0.01, 0.2, 0.9, 1.0])
    reverse = np.array([0.8, 0.4, 0.2, 0.1, 1.0])
    expected = (1 + weight) / (1 / np.clip(forward, 1e-12, 1) + weight / np.clip(reverse, 1e-12, 1))
    actual = weighted_harmonic_probability(forward, reverse, weight=weight)
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    assert np.isfinite(actual).all()


def test_identity_and_low_reverse_support():
    p = np.array([0.01, 0.5, 1.0])
    np.testing.assert_allclose(weighted_harmonic_probability(p, p, weight=0.5), p)
    assert weighted_harmonic_probability(np.array([0.9]), np.array([0.1]), weight=1)[0] < 0.5


@pytest.mark.parametrize("weight", [-1, np.nan, np.inf])
def test_invalid_weight_rejected(weight):
    with pytest.raises(ValueError):
        weighted_harmonic_probability(np.array([0.5]), np.array([0.5]), weight=weight)


@pytest.mark.parametrize("forward,reverse", [([], []), ([0.5], [0.5, 0.5]), ([np.nan], [0.5]), ([1.01], [0.5]), ([0.5], [-0.1])])
def test_invalid_probabilities_rejected(forward, reverse):
    with pytest.raises(ValueError):
        weighted_harmonic_probability(np.array(forward), np.array(reverse), weight=0.5)


def test_large_finite_weight_stays_finite():
    actual = weighted_harmonic_probability(np.array([0.8]), np.array([0.2]), weight=1e300)
    np.testing.assert_allclose(actual, [0.2])
''')
write("README.md", '''# 3D Cell Tracking & Lineage Reconstruction

**Alvaro Mendizabal · Computer vision · Constrained optimization · ML systems**

[![Portfolio quality](https://github.com/alvaromendizabal/biohub-cell-tracking-during-development/actions/workflows/portfolio.yml/badge.svg)](https://github.com/alvaromendizabal/biohub-cell-tracking-during-development/actions/workflows/portfolio.yml)

![Temporal detections, association graphs and verifiable lineages](docs/assets/hero.svg)

I integrated a volumetric tracking system, built graph-level fidelity checks, and made association decisions inspectable through a reproducible CPU pipeline and interactive lineage explorer.

**Retained historical score: 0.947.** The Harmonic Fusion system uses credited upstream architectures and pretrained artifacts. My contribution is integration, evaluation, optimization and inspection tooling—not independent authorship of the pretrained system. [Attribution](ATTRIBUTION.md) · [Original scoring receipt](reports/official_submission.json)

[**Open the lineage explorer ↗**](https://alvaro-cell-lineage-explorer.tartmacaw2.chatgpt.site) · [Case study](CASE_STUDY.md) · [Run locally](REPRODUCE.md) · [Executed graph example](examples/lineage_reasoning.ipynb)

## From detections to a coherent lineage

![Detection, temporal association and lineage validation](docs/assets/architecture.svg)

| Component | Inspect the implementation | Responsibility |
|---|---|---|
| Volumetric model interfaces | [Model components](src/biohub_tracking/public_reproduction/models.py) | Credited detection and association architectures |
| Bidirectional fusion | [Harmonic kernel](src/biohub_tracking/fusion.py) | Retained reciprocal arithmetic with caller-supplied weights and malformed-input checks |
| Constrained association | [Graph solver](src/biohub_tracking/public_graph_methods.py) | One parent per child and bounded children per parent |
| Output fidelity | [CSV comparison](scripts/reproduce_final_evidence.py) | Exact identities; edge changes separated from coordinate drift |
| Runnable example | [Synthetic pipeline](src/biohub_tracking/synthetic_demo.py) | Images, candidates, solved graphs and exact-ID diagnostics on CPU |

## Run and inspect

With Python 3.12:

```bash
python -m pip install --require-hashes -r requirements-test.lock
python -m pip install --no-deps --no-build-isolation -e .
python examples/run_tracking_demo.py --output public-demo
python scripts/run_ci.py
```

Open `public-demo/index.html`. Scrub through the twelve-frame movie, select a cell, and inspect its ancestry. Change the threshold, child limit or gap policy: the browser recomputes the graph. Candidate explanations and reference-policy comparisons show exactly which links changed and why.

This is **authored synthetic microscopy**, not private competition data or a new biological measurement. [Complete setup and replay checks](REPRODUCE.md)

## Evidence that determined the release

| Evidence | Recorded result | Scope |
|---|---:|---|
| Retained system | **0.947** | Historical external scoring receipt |
| Later candidate | **0.946** | Rejected after external regression |
| Output-fidelity audit | **8 changed edges / 4 movies** | Same detections and shared coordinates |
| Motion-linking optimization | **34.7×** | Component benchmark with selected-edge parity; not total latency |

[Results and limitations](RESULTS.md) · [Unmodified aggregate record](reports/portfolio_summary.json)

## Quality and publication boundary

CI tests malformed identities, finite coordinates, corrupt fixtures, constrained assignment, numerical fusion parity and graph-change accounting. It regenerates the fixture twice, compares its bytes with the published version, and tests the actual browser handlers against Python policies and exhaustive small-graph optima.

The maintained tree contains model components, public fusion arithmetic, the graph solver, runnable examples, tests and compact final evidence. Abandoned experiment implementations and private-runtime stubs are not presented as public capabilities. Private volumes, model weights, tuned fusion settings and full inference orchestration are not distributed. Public reproduction demonstrates component behavior; it does not recreate the historical private competitive score.

[Engineering overview](docs/ENGINEERING_OVERVIEW.md) · [Model card](docs/MODEL_CARD.md) · [License notices](THIRD_PARTY_NOTICES.md) · [Publication scope](docs/PUBLICATION_SCOPE.md)
''')
write("CASE_STUDY.md", '''# Case study · Reliable cell-lineage reconstruction

## Problem and ownership

Nearby detections create ambiguous links, missing frames interrupt tracks, and division creates two-child events. High local association scores do not guarantee a coherent lineage. I integrated model outputs, built fidelity checks, optimized motion linking with parity verification, and developed the public CPU pipeline and interactive explorer. The retained **0.947** system uses credited upstream architectures and pretrained artifacts. [Contribution boundary](ATTRIBUTION.md)

## The final approach

[Volumetric components](src/biohub_tracking/public_reproduction/models.py) describe the credited architecture. The [harmonic kernel](src/biohub_tracking/fusion.py) exposes the retained fusion arithmetic with caller-supplied weights and numerical parity tests. The [constrained solver](src/biohub_tracking/public_graph_methods.py) selects compatible parent/child edges rather than every locally attractive edge.

The original private orchestration, full microscopy inputs and tuned fusion recipe are not shipped as apparently runnable public code. The documented example is genuinely self-contained: it generates synthetic images, candidates and solved graph policies on CPU.

## A decision supported by evidence

A candidate improved the repeatedly inspected development cohort but scored **0.946** externally, so the **0.947** reference stayed frozen. The candidate changed eight association edges across four movies while keeping detections and shared coordinates unchanged. This isolates the changed output component without proving the cause of the regression. The original measurements remain intact. [Results](RESULTS.md)

## Engineering a reviewer can exercise

The [CSV audit](scripts/reproduce_final_evidence.py) rejects duplicate identities, invalid schemas and nonfinite coordinates. Large integer IDs are parsed without lossy float conversion. File hashes and exact node/edge differences make the comparison inspectable. [Failure-path tests](tests/test_evidence_inputs.py)

The [public pipeline](src/biohub_tracking/synthetic_demo.py) and browser share policy constraints. The browser solver is checked against exhaustive small-graph optima and saved Python policies. Gap closing is a separate pass; global optimality of the combined procedure is not claimed. The **34.7×** historical speedup applies only to motion linking with selected-edge parity, not end-to-end inference.

## Reproducibility and limitations

[Run from a clone](REPRODUCE.md), inspect the [executed five-cell notebook](examples/lineage_reasoning.ipynb), or replay browser decisions against synthetic truth. These exercises establish component contracts, not biological accuracy. The historical five-movie cohort covered two previously examined embryos and was not an untouched test. [Asset identities and source credits](reproducibility/external_assets.json) preserve the distinction between upstream models and this project's integration work.
''')
write("RESULTS.md", '''# Final results and evidence scope

The original records have a **September 29, 2026** research cutoff. Publication cleanup does not retrain models or alter measured results.

| Evidence | Recorded value | Meaning |
|---|---:|---|
| Retained Harmonic Fusion system | **0.947** | Historical external result with credited upstream models |
| Rejected later candidate | **0.946** | External regression; not a promoted improvement |
| Reused development comparison | **0.933046 → 0.948059** | Five movies, two previously inspected embryos; not independent confirmation |
| Fidelity audit | **8 edge edits across 4 movies** | Same detections and shared coordinates |
| Motion-linking benchmark | **34.7×** | One optimized component with parity checks |

[Original scoring receipt](reports/official_submission.json) · [Unmodified aggregate record](reports/portfolio_summary.json) · [Attribution](ATTRIBUTION.md)

The candidate's local gain did not transfer externally, so the release stayed frozen. Sparse divisions and repeated inspection limit conclusions from the development cohort. The fidelity audit narrows the changed component; it is not a causal explanation of the regression.

Historical activity counts in the original record describe the old workspace, not the number of notebooks or alternatives shipped today. The public examples reproduce graph constraints, fusion arithmetic and exact-ID diagnostics on authored synthetic inputs. They do not authenticate an external scoring service or recreate the private competitive system. [Run the checks](REPRODUCE.md)
''')
write("ATTRIBUTION.md", '''# Attribution and contribution boundary

The retained **Biohub Harmonic Fusion** system has a historical recorded score of **0.947**. Its architectures, pretrained weights and upstream components remain credited to **pilkwang** artifact releases and **RoyerLab** competition work. This repository does not claim independent authorship of that pretrained system.

Alvaro Mendizabal's maintained contributions are integration and evaluation, output-fidelity auditing, graph constraints, runtime/parity verification, reproducibility safeguards, the public fusion contract and tests, and the authored synthetic lineage explorer. The public kernel exposes the retained reciprocal arithmetic, not private tuning or model weights.

[Original asset identities](reproducibility/external_assets.json) · [Results](RESULTS.md) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [RoyerLab license](licenses/royerlab-BSD-3-Clause.txt)

Retiring an experiment does not remove its authors' rights. Existing source credits and license notices remain in force.
''')
write("docs/ENGINEERING_OVERVIEW.md", '''# Engineering overview

The maintained release connects [credited volumetric components](../src/biohub_tracking/public_reproduction/models.py), a [numerically tested fusion kernel](../src/biohub_tracking/fusion.py), [constrained association](../src/biohub_tracking/public_graph_methods.py), [CSV fidelity auditing](../scripts/reproduce_final_evidence.py), and a [runnable synthetic pipeline](../src/biohub_tracking/synthetic_demo.py).

CI installs hash-locked CPU dependencies, runs every retained test, regenerates the microscope fixture twice and exercises the actual browser handlers. The graph engine is compared with Python policies and exhaustive small-graph optima. Gap closing is a separate pass, not a claim of global temporal optimality.

The [executed example](../examples/lineage_reasoning.ipynb) shows why valid topology alone does not establish correct identities. Synthetic exact-ID metrics are not organizer-native or biological performance measurements. The retained **0.947** external score and **34.7×** motion-linking component benchmark remain historical evidence, unchanged by this release.

[Reproduction commands](../REPRODUCE.md) · [Final results](../RESULTS.md) · [Contribution boundary](../ATTRIBUTION.md)
''')
write("review/README.md", '''# Source-linked public review

```bash
python3 -B -S review/run.py --output /tmp/biohub-review --with-node
python3 -B -S review/run.py --output /tmp/biohub-review --with-node --check
```

Python 3.11+ is required; the optional Node checks require Node.js 22 or 24. The bundle records source hashes and actual check outcomes. Reuse verifies existing output rather than silently overwriting it.

The review checks original aggregate evidence, saved-output fidelity and optionally the actual graph engine and browser interface. It does not run private inference or produce a new competitive result. [Run the CPU pipeline](../REPRODUCE.md) · [Case study](../CASE_STUDY.md) · [Fusion contract](../src/biohub_tracking/fusion.py)

`project.json` binds review cards to maintained source and evidence. `run.py --render` regenerates the inspection page; `--check-page` verifies it. Changed inputs, changed outputs, missing checks and invalid paths are rejected.
''')
path = ROOT / "REPRODUCE.md"
text = path.read_text()
start = text.index("## Inspect the research evidence")
end = text.index("## Run saved-output audits")
text = text[:start] + '''## Inspect the final evidence

The [executed solver example](examples/lineage_reasoning.ipynb) is the maintained notebook. The historical **0.947** retained score and rejected **0.946** candidate remain in the original compact reports. Old experiment notebooks are not part of the runnable public project. [Results](RESULTS.md)

''' + text[end:]
text = text.replace("[Detailed execution matrix](REPRODUCIBILITY.md#component-reproduction-matrix)", "[Implementation and validation](docs/ENGINEERING_OVERVIEW.md)")
path.write_text(text)
for name in ("THIRD_PARTY_NOTICES.md", "docs/MODEL_CARD.md", "docs/PUBLICATION_SCOPE.md", "docs/DECOMMISSION.md", "examples/README.md"):
    path = ROOT / name
    text = path.read_text().replace("REPRODUCIBILITY.md", "REPRODUCE.md")
    text = text.replace("../notebooks/README.md", "../examples/README.md").replace("notebooks/README.md", "examples/README.md")
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: m.group(0) if m.group(2).startswith(("http", "#")) or (path.parent / m.group(2).split("#")[0]).exists() else m.group(1), text)
    path.write_text(text)
manifest = json.loads((ROOT / "PUBLICATION_MANIFEST.json").read_text())
manifest["release_id"] = "biohub-final-project-20261018"
manifest["scope"] = "Final tracking components, public solver, synthetic examples and compact original evidence"
manifest["required_files"] = [p for p in manifest["required_files"] if (ROOT / p).is_file()] + ["examples/lineage_reasoning.ipynb"]
manifest["required_directories"] = [p for p in manifest["required_directories"] if (ROOT / p).is_dir()]
write("PUBLICATION_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
receipt = json.loads((ROOT / "reports/publication_redaction.json").read_text())
receipt["changed_files"] = [v for v in receipt["changed_files"] if (ROOT / v["path"]).is_file()]
receipt["notebook_outputs_unchanged"] = {}
write("reports/publication_redaction.json", json.dumps(receipt, indent=2) + "\n")
replace("scripts/verify_portfolio.py", 'require(png >= 1, "Missing saved PNG " + str(path))', 'require(png >= 1 or any("image/svg+xml" in out.get("data", {}) for cell in cells for out in cell.get("outputs", [])), "Missing saved figure " + str(path))')
path = ROOT / "scripts/verify_portfolio.py"
text = path.read_text()
start = text.index("    evidence_notebooks = [")
end = text.index("    notebooks = [check_notebook", start)
text = text[:start] + '''    evidence_notebooks = [root / "examples/lineage_reasoning.ipynb"]
    require(not (root / "research").exists(), "Retired experiment workspace returned")
    require(not (root / "src/biohub_tracking/models").exists(), "Unused model prototypes returned")
''' + text[end:]
path.write_text(text)
path = ROOT / "scripts/check_showcase.py"
text = path.read_text()
start = text.index("PRIMARY_DOCS = [")
end = text.index("FORBIDDEN_FRAMING", start)
text = text[:start] + '''PRIMARY_DOCS = [
    "README.md", "CASE_STUDY.md", "RESULTS.md", "REPRODUCE.md",
    "docs/ENGINEERING_OVERVIEW.md", "docs/PUBLICATION_SCOPE.md", "examples/README.md",
]
''' + text[end:]
text = text.replace('("0.947", "0.946", "557", "45", "34.7")', '("0.947", "0.946", "34.7")')
start = text.index('    image = (root / "assets/')
end = text.index("    return {", start)
text = text[:start] + text[end:]
text = text.replace('"hero_image": [width, height],', '"hero_image": "docs/assets/hero.svg",')
path.write_text(text)
path = ROOT / "scripts/run_ci.py"
text = path.read_text()
start = text.index(" for d in sorted(")
end = text.index(" for cmd,cwd in commands:", start)
path.write_text(text[:start] + text[end:])
replace("tests/test_publication.py", "import sys,json,copy", "import sys,json,copy,base64")
replace("tests/test_publication.py", "next((ROOT/'assets').glob('*.png')).read_bytes()", "base64.b64decode(json.loads((ROOT/'public-demo/data.json').read_text())['frames'][0]['image_data_uri'].split(',',1)[1])")
replace("tests/test_publication.py", "ROOT/'notebooks/00_portfolio_overview.ipynb'", "ROOT/'examples/lineage_reasoning.ipynb'")
project = json.loads((ROOT / "review/project.json").read_text())
project["signals"][1] = {"value": "0.947", "label": "Retained historical scored system"}
project["cards"] = [c for c in project["cards"] if c["title"] != "Screen features without inflating evidence."]
project["scope"] = "The public review checks retained source, original final receipts and saved-output fidelity. Optional Node checks exercise the actual graph engine and UI. The hash-locked CPU environment regenerates authored microscopy."
write("review/project.json", json.dumps(project, indent=2) + "\n")
print("Final-system curation applied. Original scientific aggregate records were not changed.")
