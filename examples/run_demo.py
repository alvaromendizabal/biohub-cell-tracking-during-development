"""A CPU-only lineage selection example using the public package's MILP solver.

All cells and scores are authored synthetic values. Exact-ID edge agreement is
only a teaching diagnostic; it is not the organizer's microscopy graph metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from biohub_tracking.public_graph_methods import solve_constrained_edges


def graph_issues(nodes: list[dict], edges: list[list[int]]) -> list[str]:
    """Check the simple topology contract for this authored, consecutive-frame toy."""
    times = {node["node_id"]: node["t"] for node in nodes}
    issues = []
    pairs = [tuple(edge) for edge in edges]
    if len(set(pairs)) != len(pairs):
        issues.append("duplicate edges")
    outgoing = Counter(source for source, _ in pairs)
    incoming = Counter(target for _, target in pairs)
    if any(count > 2 for count in outgoing.values()):
        issues.append("more than two children")
    if any(count > 1 for count in incoming.values()):
        issues.append("more than one parent")
    for source, target in pairs:
        if source not in times or target not in times:
            issues.append("unknown endpoint")
        elif times[target] != times[source] + 1:
            issues.append("edge does not connect consecutive frames")
    return sorted(set(issues))


def edge_diagnostic(edges: list[list[int]], truth: list[list[int]]) -> dict:
    """Exact-ID agreement; real evaluation additionally needs spatial matching."""
    predicted, expected = set(map(tuple, edges)), set(map(tuple, truth))
    tp, fp, fn = (
        len(predicted & expected),
        len(predicted - expected),
        len(expected - predicted),
    )
    denominator = tp + fp + fn
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "toy_edge_jaccard": tp / denominator if denominator else None,
    }


def run_demo(fixture_path: Path | None = None) -> dict:
    fixture = json.loads((fixture_path or ROOT / "examples/fixture.json").read_text())
    candidates = pd.DataFrame(fixture["candidates"])
    independent = candidates.loc[candidates.groupby("target_id")["edge_score"].idxmax()]
    constrained = solve_constrained_edges(candidates)
    misleading = solve_constrained_edges(candidates, score_col="misleading_score")
    tables = {
        "independent": independent,
        "constrained": constrained[constrained.selected],
        "valid_but_wrong": misleading[misleading.selected],
    }
    scenarios = {}
    for name, table in tables.items():
        edges = sorted(table[["source_id", "target_id"]].astype(int).values.tolist())
        issues = graph_issues(fixture["nodes"], edges)
        scenarios[name] = {
            "edges": edges,
            "valid_topology": not issues,
            "issues": issues,
            **edge_diagnostic(edges, fixture["expected_edges"]),
        }
    return {
        "scope": "authored synthetic example; no inference or official evaluation",
        "solver": "biohub_tracking.public_graph_methods.solve_constrained_edges",
        "nodes": fixture["nodes"],
        "expected_edges": fixture["expected_edges"],
        "scenarios": scenarios,
    }


def render_svg(result: dict) -> str:
    """Render the actual selected edges, with text labels and error colors."""
    labels = {node["node_id"]: node["label"] for node in result["nodes"]}
    expected = set(map(tuple, result["expected_edges"]))
    points = {1: (64, 166), 2: (64, 266), 3: (276, 116), 4: (276, 216), 5: (276, 316)}
    titles = [
        "Independent choices",
        "Constrained selection",
        "Valid graph, wrong answer",
    ]
    subtitle = [
        "Three children violate the contract",
        "A division and a continuation",
        "Constraints do not establish accuracy",
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1140" height="490" viewBox="0 0 1140 490" role="img" aria-labelledby="title desc">',
        '<title id="title">Why lineage tracking needs constraints and evaluation</title>',
        '<desc id="desc">Five invented cells show an invalid independent selection, a correct constrained graph, and an incorrect graph that still satisfies structural checks.</desc>',
        '<rect width="1140" height="490" fill="#f5f7fa"/>',
        '<g font-family="Arial, sans-serif" fill="#183041">',
        '<text x="26" y="32" font-size="21" font-weight="700">A small graph exposes a large modeling lesson</text>',
        '<text x="26" y="55" font-size="13">Authored cells and scores · Existing public MILP solver · No microscopy, inference, or official score</text>',
    ]
    for index, (name, scenario) in enumerate(result["scenarios"].items()):
        x = 20 + index * 375
        parts.append(f'<g transform="translate({x},74)">')
        parts.append(
            '<rect width="350" height="365" rx="12" fill="#ffffff" stroke="#d8e1e8"/>'
        )
        parts.append(
            f'<text x="16" y="28" font-size="17" font-weight="700">{titles[index]}</text>'
        )
        parts.append(f'<text x="16" y="49" font-size="12">{subtitle[index]}</text>')
        parts.append(
            '<text x="35" y="82" font-size="12" fill="#647789">Frame 0</text><text x="248" y="82" font-size="12" fill="#647789">Frame 1</text>'
        )
        for source, target in scenario["edges"]:
            sx, sy = points[source]
            tx, ty = points[target]
            color = "#17734d" if (source, target) in expected else "#be4048"
            parts.append(
                f'<line x1="{sx + 20}" y1="{sy}" x2="{tx - 20}" y2="{ty}" stroke="{color}" stroke-width="3"/>'
            )
        for node_id, (nx, ny) in points.items():
            color = "#be4048" if name == "independent" and node_id == 1 else "#183041"
            parts.append(
                f'<circle cx="{nx}" cy="{ny}" r="20" fill="#eef3f7" stroke="{color}" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{nx}" y="{ny + 5}" text-anchor="middle" font-size="16" font-weight="700">{escape(labels[node_id])}</text>'
            )
        diagnostic = scenario["toy_edge_jaccard"]
        parts.append(
            f'<text x="16" y="351" font-size="12">Exact-ID toy edge agreement: {diagnostic:.2f}</text></g>'
        )
    parts.append(
        '<text x="26" y="467" font-size="13">Green = authored true link. Red = authored wrong link. Agreement here is not the organizer’s graph score.</text></g></svg>'
    )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--svg", type=Path, help="Optionally save the generated graph illustration."
    )
    args = parser.parse_args()
    result = run_demo()
    if args.svg:
        args.svg.write_text(render_svg(result), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
