#!/usr/bin/env python3
"""Reproduce final Biohub saved-output fidelity/diversity evidence.

This script is intentionally data-light: it reads submission CSV files supplied by
the user, never contacts AWS/Kaggle, and never uses hidden labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _int_or_none(value: str | None):
    if value is None or value == "":
        return None
    # Pandas can serialize nullable integer columns as "123.0". Accept that
    # representation without routing identities through lossy binary floats.
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"invalid integer value {value!r}") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"expected a finite integer, got {value!r}")
    return int(number)


def _float_or_none(value: str | None):
    if value is None or value == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"expected a finite coordinate, got {value!r}")
    return number


def load_submission(path: Path):
    nodes = {}
    edges = {}
    rows = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"dataset", "row_type"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing required columns {sorted(required)}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: duplicate column names")
        row_columns = {
            "node": {"node_id", "t", "z", "y", "x"},
            "edge": {"source_id", "target_id"},
        }
        for row in reader:
            rows += 1
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("row length does not match the CSV header")
                ds = row["dataset"]
                if not ds.strip():
                    raise ValueError("row without dataset")
                rt = row["row_type"].lower()
                if rt not in row_columns:
                    raise ValueError(f"unexpected row_type={rt!r}")
                missing = row_columns[rt] - row.keys()
                if missing:
                    raise ValueError(f"{rt} row missing columns {sorted(missing)}")
                empty = [name for name in sorted(row_columns[rt]) if not row[name].strip()]
                if empty:
                    raise ValueError(f"{rt} row missing values {empty}")
                if rt == "node":
                    nid = _int_or_none(row["node_id"])
                    t = _int_or_none(row["t"])
                    z = _float_or_none(row["z"])
                    y = _float_or_none(row["y"])
                    x = _float_or_none(row["x"])
                    if (ds, nid) in nodes:
                        raise ValueError(f"duplicate node identity {(ds, nid)!r}")
                    nodes[(ds, nid)] = (t, z, y, x)
                else:
                    src = _int_or_none(row["source_id"])
                    tgt = _int_or_none(row["target_id"])
                    dataset_edges = edges.setdefault(ds, set())
                    if (src, tgt) in dataset_edges:
                        raise ValueError(f"duplicate edge identity {(ds, src, tgt)!r}")
                    dataset_edges.add((src, tgt))
            except ValueError as exc:
                raise ValueError(f"{path}: line {reader.line_num}: {exc}") from exc
    return {"rows": rows, "nodes": nodes, "edges": edges}


def stats(path: Path):
    data = load_submission(path)
    edge_count = sum(len(v) for v in data["edges"].values())
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": data["rows"],
        "nodes": len(data["nodes"]),
        "edges": edge_count,
        "datasets": sorted({k[0] for k in data["nodes"]} | set(data["edges"])),
    }, data


def compare(a: Path, b: Path):
    sa, da = stats(a)
    sb, db = stats(b)
    ka, kb = set(da["nodes"]), set(db["nodes"])
    shared = ka & kb
    coordinate_shifted = [k for k in shared if da["nodes"][k] != db["nodes"][k]]
    edge_added = {}
    edge_removed = {}
    for ds in sorted(set(da["edges"]) | set(db["edges"])):
        ea = da["edges"].get(ds, set())
        eb = db["edges"].get(ds, set())
        add = sorted(eb - ea)
        rem = sorted(ea - eb)
        if add:
            edge_added[ds] = add
        if rem:
            edge_removed[ds] = rem
    return {
        "a": sa,
        "b": sb,
        "nodes_added": len(kb - ka),
        "nodes_removed": len(ka - kb),
        "shared_nodes": len(shared),
        "shared_nodes_with_coordinate_change": len(coordinate_shifted),
        "coordinate_change_examples": coordinate_shifted[:20],
        "edges_added_total": sum(len(v) for v in edge_added.values()),
        "edges_removed_total": sum(len(v) for v in edge_removed.values()),
        "edges_added_by_dataset": {k: len(v) for k, v in edge_added.items()},
        "edges_removed_by_dataset": {k: len(v) for k, v in edge_removed.items()},
        "edge_added_examples": {k: v[:20] for k, v in edge_added.items()},
        "edge_removed_examples": {k: v[:20] for k, v in edge_removed.items()},
    }


def duplicate(paths: list[Path]):
    rows = []
    for p in paths:
        s, _ = stats(p)
        rows.append(s)
    return {
        "files": rows,
        "all_sha256_identical": len({r["sha256"] for r in rows}) == 1,
        "all_shape_identical": len({(r["rows"], r["nodes"], r["edges"], tuple(r["datasets"])) for r in rows}) == 1,
    }


def self_test():
    header = ["id", "dataset", "row_type", "node_id", "source_id", "target_id", "t", "z", "y", "x"]
    rows_a = [
        [0, "m", "node", 1, "", "", 0, 1, 2, 3],
        [1, "m", "node", 2, "", "", 1, 1, 3, 3],
        [2, "m", "edge", "", 1, 2, "", "", "", ""],
    ]
    rows_b = [
        [0, "m", "node", 1, "", "", 0, 1, 2, 3],
        [1, "m", "node", 2, "", "", 1, 1, 3, 3],
        [2, "m", "edge", "", 2, 1, "", "", "", ""],
    ]
    with tempfile.TemporaryDirectory() as td:
        a = Path(td) / "a.csv"
        b = Path(td) / "b.csv"
        for path, rows in [(a, rows_a), (b, rows_b)]:
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)
        result = compare(a, b)
        assert result["nodes_added"] == 0
        assert result["nodes_removed"] == 0
        assert result["shared_nodes_with_coordinate_change"] == 0
        assert result["edges_added_total"] == 1
        assert result["edges_removed_total"] == 1
        dup = duplicate([a, a])
        assert dup["all_sha256_identical"] and dup["all_shape_identical"]
    print("FINAL_EVIDENCE_REPRO_SELF_TEST_PASSED")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("compare")
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p = sub.add_parser("duplicate")
    p.add_argument("files", nargs="+", type=Path)
    sub.add_parser("self-test")
    args = ap.parse_args()
    if args.command == "self-test":
        self_test()
        return
    if args.command == "compare":
        out = compare(args.baseline, args.candidate)
    else:
        out = duplicate(args.files)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
