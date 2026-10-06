"""Input identity must survive saved-output fidelity audits unchanged."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reproduce_final_evidence.py"
SPEC = importlib.util.spec_from_file_location("biohub_evidence_inputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)

HEADER = ["id", "dataset", "row_type", "node_id", "source_id", "target_id", "t", "z", "y", "x"]


def write_csv(path, rows, header=HEADER):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return path


def node(nid="1", dataset="movie", **changes):
    return dict(dataset=dataset, row_type="node", node_id=nid, t="0", z="1", y="2", x="3", **changes)


def test_exact_large_id_and_nullable_integer_serialization(tmp_path):
    first, second = 2**53, 2**53 + 1
    path = write_csv(tmp_path / "large.csv", [
        node(str(first)),
        node(f"{second}.0"),
        dict(dataset="movie", row_type="edge", source_id=f"{first}.0", target_id=str(second)),
    ])
    parsed = evidence.load_submission(path)
    assert set(parsed["nodes"]) == {("movie", first), ("movie", second)}
    assert parsed["edges"] == {"movie": {(first, second)}}
    assert evidence.compare(path, path)["shared_nodes"] == 2
    assert evidence.duplicate([path, path])["all_sha256_identical"]


def test_identity_is_scoped_to_dataset(tmp_path):
    path = write_csv(tmp_path / "scoped.csv", [node(dataset="a"), node(dataset="b")])
    assert len(evidence.load_submission(path)["nodes"]) == 2


@pytest.mark.parametrize("rows", [
    [node(), node("1.0")],
    [dict(dataset="movie", row_type="edge", source_id="1", target_id="2"),
     dict(dataset="movie", row_type="edge", source_id="1.0", target_id="2.0")],
])
def test_duplicate_identities_rejected(tmp_path, rows):
    with pytest.raises(ValueError, match="duplicate .* identity"):
        evidence.load_submission(write_csv(tmp_path / "duplicate.csv", rows))


@pytest.mark.parametrize("column,value", [
    ("node_id", "1.5"), ("node_id", "nan"), ("node_id", "Infinity"),
    ("t", "0.1"), ("z", "NaN"), ("x", "inf"), ("y", "1e9999"),
    ("x", ""), ("t", ""), ("dataset", "  "),
])
def test_malformed_node_values_rejected(tmp_path, column, value):
    row = node()
    row[column] = value
    with pytest.raises(ValueError, match="line 2"):
        evidence.load_submission(write_csv(tmp_path / "invalid.csv", [row]))


@pytest.mark.parametrize("row", [
    dict(dataset="movie", row_type="node", node_id="1"),
    dict(dataset="movie", row_type="edge", source_id="1"),
])
def test_required_row_type_columns_rejected(tmp_path, row):
    with pytest.raises(ValueError, match="missing columns"):
        evidence.load_submission(write_csv(tmp_path / "schema.csv", [row], list(row)))


@pytest.mark.parametrize("text,expected", [
    ("dataset,row_type,node_id,node_id\nmovie,node,1,2\n", "duplicate column"),
    ("dataset,row_type\nmovie,node,extra\n", "row length"),
    ("dataset,row_type\nmovie\n", "row length"),
    ("dataset,row_type\nmovie,unknown\n", "unexpected row_type"),
])
def test_invalid_csv_structure_rejected(tmp_path, text, expected):
    path = tmp_path / "malformed.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=expected):
        evidence.load_submission(path)
