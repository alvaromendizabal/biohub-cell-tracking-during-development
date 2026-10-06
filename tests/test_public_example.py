"""Public example tests exercise selection, graph contracts, and honest scope."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public_lineage_example", ROOT / "examples/run_demo.py"
)
assert SPEC and SPEC.loader
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def test_example_uses_constraints_without_confusing_validity_with_accuracy():
    result = demo.run_demo()
    independent = result["scenarios"]["independent"]
    constrained = result["scenarios"]["constrained"]
    wrong = result["scenarios"]["valid_but_wrong"]
    assert independent["issues"] == ["more than two children"]
    assert independent["toy_edge_jaccard"] == 0.5
    assert constrained["valid_topology"]
    assert constrained["edges"] == result["expected_edges"]
    assert constrained["toy_edge_jaccard"] == 1.0
    assert wrong["valid_topology"]
    assert wrong["false_positive"] == wrong["false_negative"] == 2
    assert wrong["toy_edge_jaccard"] == 0.2


@pytest.mark.parametrize(
    "edges,issue",
    [
        ([[1, 3], [1, 3]], "duplicate edges"),
        ([[1, 3], [2, 3]], "more than one parent"),
        ([[1, 3], [1, 4], [1, 5]], "more than two children"),
        ([[3, 1]], "edge does not connect consecutive frames"),
        ([[1, 6]], "unknown endpoint"),
    ],
)
def test_example_rejects_malformed_graphs(edges, issue):
    fixture = json.loads((ROOT / "examples/fixture.json").read_text())
    assert issue in demo.graph_issues(fixture["nodes"], edges)


def test_cli_does_not_depend_on_working_directory(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples/run_demo.py")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(completed.stdout)
    assert "no inference or official evaluation" in result["scope"]
    assert result["scenarios"]["constrained"]["valid_topology"]


def test_saved_example_notebook_is_executed_with_visible_graph():
    notebook = json.loads((ROOT / "examples/lineage_reasoning.ipynb").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert [cell["execution_count"] for cell in cells] == list(range(1, len(cells) + 1))
    outputs = [output for cell in cells for output in cell["outputs"]]
    assert not any(output["output_type"] == "error" for output in outputs)
    assert any("image/svg+xml" in output.get("data", {}) for output in outputs)
    assert any("text/html" in output.get("data", {}) for output in outputs)
    saved_svg = next(
        output["data"]["image/svg+xml"]
        for output in outputs
        if "image/svg+xml" in output.get("data", {})
    )
    saved_svg = "".join(saved_svg) if isinstance(saved_svg, list) else saved_svg
    computed_svg = demo.render_svg(demo.run_demo())
    assert saved_svg == computed_svg
    assert (ROOT / "examples/lineage_demo.svg").read_text() == computed_svg
