import base64
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from biohub_tracking import synthetic_demo as demo


@pytest.fixture(scope="module")
def result():
    return demo.run_demo()


def test_real_milp_is_used_and_all_selected_graphs_are_checked(monkeypatch):
    original = demo.public_graph_methods.solve_constrained_edges
    calls = []
    def recording(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)
    monkeypatch.setattr(demo.public_graph_methods, "solve_constrained_edges", recording)
    result = demo.run_demo()
    assert len(calls) >= 5 and max(calls) > 50
    for solution in result["solutions"]:
        expected = demo.graph_metrics(result["nodes"], solution["edges"], result["truth_edges"])
        assert solution["metrics"] == expected
        if solution["id"] != "independent":
            assert solution["checks"]["valid_topology"]
            assert solution["checks"]["max_in_degree"] <= 1
            assert solution["checks"]["max_out_degree"] <= solution["parameters"]["max_children"]


def test_substantive_sequence_has_division_missing_and_false_observations(result):
    assert result["summary"]["frames"] == 12
    assert result["summary"]["truth_nodes"] > 100
    assert result["summary"]["truth_divisions"] == 2
    assert result["summary"]["missed_detections"] == 2
    assert result["summary"]["false_positive_detections"] == 2
    assert len(result["solutions"]) == 6
    assert result["provenance"]["evidence_type"] == "SYNTHETIC_ONLY"
    assert result["provenance"]["training_performed"] is False
    assert result["provenance"]["official_score"] is None


def test_pngs_contain_actual_distinct_generated_pixels(result):
    images = []
    for frame in result["frames"]:
        raw = base64.b64decode(frame["image_data_uri"].split(",", 1)[1])
        assert raw.startswith(b"\x89PNG\r\n\x1a\n")
        position, compressed = 8, []
        while position < len(raw):
            length = struct.unpack(">I", raw[position:position + 4])[0]
            kind, payload = raw[position + 4:position + 8], raw[position + 8:position + 8 + length]
            checksum = struct.unpack(">I", raw[position + 8 + length:position + 12 + length])[0]
            assert zlib.crc32(kind + payload) & 0xffffffff == checksum
            if kind == b"IHDR":
                assert struct.unpack(">IIBBBBB", payload) == (160, 128, 8, 0, 0, 0, 0)
            elif kind == b"IDAT":
                compressed.append(payload)
            position += length + 12
        pixels = zlib.decompress(b"".join(compressed))
        assert len(pixels) == 128 * 161
        assert all(pixels[row * 161] == 0 for row in range(128))
        images.append(demo.sha(pixels))
    assert len(set(images)) == 12


def test_candidate_generation_cannot_read_truth_annotations(result):
    altered = copy.deepcopy(result["nodes"])
    for index, node in enumerate(altered):
        node["track_id"] = "unrelated-" + str(index)
        node["is_false_positive"] = not node["is_false_positive"]
        node["observed"] = False
    assert demo.build_candidates(altered) == result["candidates"]
    assert all(edge["dt"] in (1, 2) and np.isfinite(edge["score"]) for edge in result["candidates"])


def test_division_and_gap_policies_have_measurable_effects(result):
    solutions = {solution["id"]: solution for solution in result["solutions"]}
    assert solutions["no_gap"]["metrics"]["gap_edges"] == 0
    assert solutions["balanced"]["metrics"]["gap_edges"] > 0
    assert solutions["no_divisions"]["metrics"]["division_events"] == 0
    assert solutions["no_divisions"]["metrics"]["division_false_negative"] == 2
    assert solutions["balanced"]["metrics"]["division_true_positive"] > 0
    assert solutions["conservative"]["metrics"]["selected_edges"] < solutions["permissive"]["metrics"]["selected_edges"]


def test_deterministic_regeneration_and_browser_json_match(tmp_path, result):
    regenerated = demo.write_demo(tmp_path)
    assert regenerated == result
    raw = (tmp_path / "data.json").read_bytes()
    script = (tmp_path / "data.js").read_text()
    assert json.loads(script.removeprefix("window.BIOHUB_DEMO_DATA = ").removesuffix(";\n")) == json.loads(raw)
    demo.write_demo(tmp_path)
    assert (tmp_path / "data.json").read_bytes() == raw


@pytest.mark.parametrize("kind,issue", [
    ("duplicate", "duplicate edges"), ("parents", "multiple parents"),
    ("children", "child capacity exceeded"), ("backward", "invalid time direction or gap"),
    ("missing", "unknown endpoint"), ("uncandidate", "selected edge is not a candidate")])
def test_graph_checker_rejects_structural_failures(kind, issue):
    nodes = [{"id": i, "t": 0 if i < 2 else 1} for i in range(5)]
    pairs = {"duplicate": [(0, 2), (0, 2)], "parents": [(0, 2), (1, 2)],
             "children": [(0, 2), (0, 3), (0, 4)], "backward": [(2, 0)],
             "missing": [(0, 99)], "uncandidate": [(0, 3)]}[kind]
    edges = [{"source": s, "target": t} for s, t in pairs]
    candidates = edges if kind != "uncandidate" else []
    assert issue in demo.graph_checks(nodes, edges, candidates)["issues"]


def test_valid_topology_does_not_imply_correct_lineage():
    nodes = [{"id": i, "t": 0 if i < 2 else 1} for i in range(4)]
    truth = [{"source": 0, "target": 2}, {"source": 1, "target": 3}]
    wrong = [{"source": 0, "target": 3}, {"source": 1, "target": 2}]
    assert demo.graph_checks(nodes, wrong, truth + wrong)["valid_topology"]
    metrics = demo.graph_metrics(nodes, wrong, truth)
    assert metrics["edge_f1"] == 0 and metrics["false_positive"] == metrics["false_negative"] == 2


def test_candidate_schema_rejects_nonfinite_and_duplicate_nodes(result):
    nodes = copy.deepcopy(result["nodes"])
    nodes[0]["x"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        demo.build_candidates(nodes)
    with pytest.raises(ValueError, match="Unique"):
        demo.build_candidates(result["nodes"] + result["nodes"][:1])


def test_cli_from_an_unrelated_directory(tmp_path):
    completed = subprocess.run([sys.executable, str(ROOT / "examples/run_tracking_demo.py"),
                                "--output", str(tmp_path / "fixture")], cwd=tmp_path,
                               capture_output=True, text=True, check=True)
    summary = json.loads(completed.stdout)
    assert summary["evidence_type"] == "SYNTHETIC_ONLY"
    assert summary["summary"]["frames"] == 12
    assert (tmp_path / "fixture/data.json").is_file() and (tmp_path / "fixture/data.js").is_file()
