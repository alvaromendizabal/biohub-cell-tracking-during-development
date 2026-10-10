"""Authored synthetic microscopy and lineage-association engineering demo.

The images, cells, detections and scoring parameters are generated demonstration
values. This module uses the repository's existing degree-constrained MILP
solver; it does not train a detector, use private assets or implement an official
competition scorer. Ground truth is used only after association for diagnostics.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import zlib

import numpy as np
import pandas as pd
import scipy

from . import public_graph_methods

SCHEMA = "biohub-synthetic-lineage-demo-v1"


@dataclass(frozen=True)
class DemoConfig:
    seed: int = 2026
    frame_count: int = 12
    width: int = 160
    height: int = 128
    candidate_radius: float = 13.0
    distance_scale: float = 4.2
    appearance_weight: float = 2.0
    gap_penalty: float = 0.8
    observation_noise: float = 0.35

    def validate(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (self.frame_count, self.width, self.height) != (12, 160, 128):
            raise ValueError("This authored sequence uses a fixed 12-frame, 160-by-128 field.")
        for name in ("candidate_radius", "distance_scale", "appearance_weight", "gap_penalty", "observation_noise"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _png_gray(pixels: np.ndarray) -> bytes:
    """Encode real generated pixels as PNG using only the standard library."""
    if pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise ValueError("PNG input must be a two-dimensional uint8 array")
    height, width = pixels.shape
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    raw = b"".join(b"\0" + row.tobytes() for row in pixels)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def render_frame(cells: list[dict], artifacts: list[dict], config: DemoConfig, rng) -> np.ndarray:
    """A noisy XY projection of synthetic fluorescent Gaussian cell profiles."""
    yy, xx = np.mgrid[:config.height, :config.width]
    image = rng.normal(9.0, 1.7, (config.height, config.width))
    for cell in [*cells, *artifacts]:
        sigma = cell["radius"] * 0.65
        image += 220 * cell["intensity"] * np.exp(-((xx - cell["x"])**2 + (yy - cell["y"])**2) / (2 * sigma**2))
    return np.rint(np.clip(image, 0, 255)).astype(np.uint8)


def generate_sequence(config: DemoConfig = DemoConfig()) -> dict:
    config.validate()
    rng = np.random.default_rng(config.seed)
    image_rng = np.random.default_rng(config.seed + 1)
    initial = [
        ((22, 24, 2), (.8, .35, .05)), ((66, 42, 3), (1.5, .15, 0)),
        ((96, 47, 3), (-1.5, -.15, 0)), ((135, 24, 5), (-.25, .55, -.04)),
        ((24, 94, 4), (.4, -.5, 0)), ((61, 91, 2), (.35, -.1, .03)),
        ((104, 98, 4), (.4, -.6, -.02)), ((137, 85, 1), (-.5, .3, .01)),
    ]
    active = {}
    for index, (position, velocity) in enumerate(initial):
        active[f"cell-{index:02d}"] = {"position": np.asarray(position, float),
            "velocity": np.asarray(velocity, float), "radius": float(rng.uniform(3.5, 4.3)),
            "intensity": float(rng.uniform(.58, .92)), "previous": None}
    divisions = {4: "cell-00", 7: "cell-05"}
    missed = {(6, "cell-02"), (8, "cell-06")}
    artifacts = {3: (114., 52.), 9: (32., 86.)}
    frames, truth, observations, events = [], [], [], []
    for t in range(config.frame_count):
        if t:
            for state in active.values():
                state["position"] += state["velocity"]
        divided = divisions.get(t)
        if divided is not None:
            parent = active.pop(divided)
            for suffix, sign in (("a", -1), ("b", 1)):
                active[divided + suffix] = {
                    "position": parent["position"] + sign * np.asarray([3.1, 2.1, .35]),
                    "velocity": parent["velocity"] + sign * np.asarray([.28, .18, .015]),
                    "radius": parent["radius"] * .78, "intensity": parent["intensity"] * .96,
                    "previous": parent["previous"]}
        cells, detections, daughter_ids = [], [], []
        for track_id, state in sorted(active.items()):
            node_id = len(truth)
            x, y, z = state["position"]
            observed = (t, track_id) not in missed
            cell = {"id": node_id, "track_id": track_id, "t": t, "x": float(x), "y": float(y),
                    "z": float(z), "radius": state["radius"], "intensity": state["intensity"], "observed": observed}
            truth.append({**cell, "parent": state["previous"]})
            state["previous"] = node_id
            cells.append(cell)
            if divided is not None and track_id in (divided + "a", divided + "b"):
                daughter_ids.append(node_id)
            if observed:
                perturbation = rng.normal(0, config.observation_noise, 3)
                detection = {**cell, "x": float(x + perturbation[0]), "y": float(y + perturbation[1]),
                    "z": float(z + perturbation[2]), "intensity": float(np.clip(state["intensity"] + rng.normal(0, .025), .1, 1)),
                    "is_false_positive": False}
                detections.append(detection)
            else:
                events.append({"type": "missed_detection", "frame": t, "node_ids": [node_id],
                    "track_ids": [track_id], "description": "A visible synthetic cell has no observation; a two-frame link can reconnect its track."})
        if divided is not None:
            parent_id = truth[daughter_ids[0]]["parent"]
            events.append({"type": "division", "frame": t, "node_ids": [parent_id, *daughter_ids],
                "track_ids": [divided, divided + "a", divided + "b"], "description": "One authored parent gives rise to two daughter tracks."})
        frame_artifacts = []
        if t in artifacts:
            x, y = artifacts[t]
            artifact = {"id": 50000 + t, "track_id": f"artifact-{t}", "t": t,
                "x": x, "y": y, "z": 3., "radius": 2.6, "intensity": .54,
                "observed": True, "is_false_positive": True}
            frame_artifacts.append(artifact)
            detections.append(artifact)
            events.append({"type": "false_positive", "frame": t, "node_ids": [artifact["id"]],
                "track_ids": [artifact["track_id"]], "description": "A synthetic fluorescent artifact is deliberately included among the observations."})
        png = _png_gray(render_frame(cells, frame_artifacts, config, image_rng))
        observations.extend(detections)
        frames.append({"index": t, "cells": cells, "detections": detections,
                       "image_data_uri": "data:image/png;base64," + base64.b64encode(png).decode()})
    # Observable reference: contract only the deliberately absent detections.
    # This exact-ID diagnostic is separate from spatial matching/official scoring.
    observed_ids = {node["id"] for node in observations if not node["is_false_positive"]}
    expected = []
    for node in truth:
        if node["id"] not in observed_ids:
            continue
        parent_id = node["parent"]
        while parent_id is not None and parent_id not in observed_ids:
            parent_id = truth[parent_id]["parent"]
        if parent_id is not None:
            parent = truth[parent_id]
            kind = ("division" if parent["track_id"] != node["track_id"] else
                    "gap" if node["t"] - parent["t"] > 1 else "continuation")
            expected.append({"source": parent_id, "target": node["id"], "kind": kind})
    return {"frames": frames, "nodes": observations, "truth_edges": expected,
            "events": sorted(events, key=lambda event: (event["frame"], event["type"])), "truth_nodes": len(truth)}


def build_candidates(nodes: list[dict], config: DemoConfig = DemoConfig()) -> list[dict]:
    """Only observed position/time/intensity enter this untrained public heuristic."""
    config.validate()
    ids = [node["id"] for node in nodes]
    if len(ids) != len(set(ids)) or any(type(node_id) is not int for node_id in ids):
        raise ValueError("Unique integer detection identifiers are required")
    for node in nodes:
        if type(node["t"]) is not int or not 0 <= node["t"] < config.frame_count:
            raise ValueError("Detection time is outside the synthetic sequence")
        if not all(math.isfinite(node[key]) for key in ("x", "y", "z", "intensity")):
            raise ValueError("Detection features must be finite")
    rows = []
    for source in sorted(nodes, key=lambda node: node["id"]):
        for target in sorted(nodes, key=lambda node: node["id"]):
            dt = target["t"] - source["t"]
            if dt not in (1, 2):
                continue
            distance = math.sqrt(sum((target[key] - source[key])**2 for key in ("x", "y", "z")))
            if distance > config.candidate_radius * math.sqrt(dt):
                continue
            appearance = abs(target["intensity"] - source["intensity"])
            score = (2.4 - distance / (config.distance_scale * math.sqrt(dt))
                     - config.appearance_weight * appearance - config.gap_penalty * (dt - 1))
            rows.append({"source": source["id"], "target": target["id"], "dt": dt,
                         "distance": distance, "appearance_difference": appearance, "score": score})
    return rows


def graph_checks(nodes: list[dict], edges: list[dict], candidates: list[dict], max_gap=2, max_children=2) -> dict:
    times = {node["id"]: node["t"] for node in nodes}
    pairs = [(edge["source"], edge["target"]) for edge in edges]
    outgoing, incoming = Counter(source for source, _ in pairs), Counter(target for _, target in pairs)
    issues = []
    if len(pairs) != len(set(pairs)):
        issues.append("duplicate edges")
    if any(count > 1 for count in incoming.values()):
        issues.append("multiple parents")
    if any(count > max_children for count in outgoing.values()):
        issues.append("child capacity exceeded")
    known = all(source in times and target in times for source, target in pairs)
    forward = known and all(1 <= times[target] - times[source] <= max_gap for source, target in pairs)
    if not known:
        issues.append("unknown endpoint")
    elif not forward:
        issues.append("invalid time direction or gap")
    children = defaultdict(list)
    for source, target in pairs:
        children[source].append(target)
    if known and any(len(targets) == 2 and len({times[target] for target in targets}) > 1 for targets in children.values()):
        issues.append("division children occupy different frames")
    allowed = {(edge["source"], edge["target"]) for edge in candidates}
    are_candidates = all(pair in allowed for pair in pairs)
    if not are_candidates:
        issues.append("selected edge is not a candidate")
    return {"valid_topology": not issues, "issues": sorted(set(issues)),
            "max_in_degree": max(incoming.values(), default=0),
            "max_out_degree": max(outgoing.values(), default=0),
            "forward_time_only": forward, "all_edges_are_candidates": are_candidates}


def _divisions(pairs):
    children = defaultdict(set)
    for source, target in pairs:
        children[source].add(target)
    return {(source, tuple(sorted(targets))) for source, targets in children.items() if len(targets) == 2}


def _agreement(predicted, expected):
    tp, fp, fn = len(predicted & expected), len(predicted - expected), len(expected - predicted)
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "jaccard": tp / (tp + fp + fn) if tp + fp + fn else 0.}


def graph_metrics(nodes: list[dict], edges: list[dict], truth_edges: list[dict]) -> dict:
    predicted = {(edge["source"], edge["target"]) for edge in edges}
    expected = {(edge["source"], edge["target"]) for edge in truth_edges}
    edge_score, division_score = _agreement(predicted, expected), _agreement(_divisions(predicted), _divisions(expected))
    adjacency = {node["id"]: set() for node in nodes}
    for source, target in predicted:
        if source not in adjacency or target not in adjacency:
            raise ValueError("Cannot evaluate edges with unknown endpoints")
        adjacency[source].add(target)
        adjacency[target].add(source)
    seen, components = set(), 0
    for node_id in adjacency:
        if node_id in seen:
            continue
        components += 1
        pending = [node_id]
        while pending:
            node = pending.pop()
            if node not in seen:
                seen.add(node)
                pending.extend(adjacency[node] - seen)
    times = {node["id"]: node["t"] for node in nodes}
    return {**{key: edge_score[key] for key in ("true_positive", "false_positive", "false_negative")},
            **{"edge_" + key: edge_score[key] for key in ("precision", "recall", "f1", "jaccard")},
            **{"division_" + key: division_score[key] for key in ("true_positive", "false_positive", "false_negative", "precision", "recall", "f1")},
            "selected_edges": len(predicted), "gap_edges": sum(times[target] - times[source] > 1 for source, target in predicted),
            "division_events": len(_divisions(predicted)), "components": components}


POLICIES = (
    ("balanced", "Constrained + gap closing", "Degree-constrained adjacent links, then residual two-frame gap closing.", 2, 2, 0.),
    ("no_gap", "Constrained · no gap closing", "Adjacent-frame associations only; missing observations break tracks.", 1, 2, 0.),
    ("no_divisions", "One child per cell", "A one-child capacity prevents the graph from representing cell division.", 2, 1, 0.),
    ("conservative", "Conservative association", "A higher utility cutoff leaves uncertain observations unlinked.", 2, 2, 1.2),
    ("permissive", "Permissive association", "A lower utility cutoff admits weak associations, including possible artifacts.", 2, 2, -.5),
    ("independent", "Independent best parent", "Targets choose their best candidate independently; parent capacities are unchecked.", 2, 2, 0.),
)


def solve_policy(candidates: list[dict], policy) -> list[dict]:
    name, _, _, max_gap, max_children, cutoff = policy
    eligible = [row for row in candidates if row["dt"] <= max_gap and row["score"] > cutoff]
    if name == "independent":
        best = {}
        for row in eligible:
            if row["target"] not in best or row["score"] > best[row["target"]]["score"]:
                best[row["target"]] = row
        selected = list(best.values())
    else:
        def constrained(rows, capacity):
            if not rows:
                return []
            table = pd.DataFrame([{"source_id": row["source"], "target_id": row["target"], "edge_score": row["score"] - cutoff}
                                  for row in rows])
            solved = public_graph_methods.solve_constrained_edges(table, max_out_degree=capacity)
            return [row for row, selected in zip(rows, solved["selected"].to_numpy()) if selected]
        selected = constrained([row for row in eligible if row["dt"] == 1], max_children)
        # A gap joins two unlinked endpoints only. It cannot create a division
        # spread across frames or compete against an already assigned parent.
        outgoing = {row["source"] for row in selected}
        incoming = {row["target"] for row in selected}
        selected += constrained([row for row in eligible if row["dt"] == 2 and row["source"] not in outgoing
                                 and row["target"] not in incoming], 1)
    counts = Counter(row["source"] for row in selected)
    return [{"source": row["source"], "target": row["target"], "score": row["score"],
             "kind": "division" if counts[row["source"]] == 2 else "gap" if row["dt"] > 1 else "continuation"}
            for row in sorted(selected, key=lambda row: (row["source"], row["target"]))]


def run_demo(config: DemoConfig = DemoConfig()) -> dict:
    movie = generate_sequence(config)
    candidates = build_candidates(movie["nodes"], config)
    solutions = []
    for policy in POLICIES:
        name, label, description, gap, children, cutoff = policy
        edges = solve_policy(candidates, policy)
        checks = graph_checks(movie["nodes"], edges, candidates, gap, children)
        if name != "independent" and not checks["valid_topology"]:
            raise RuntimeError("Constrained synthetic graph failed its topology contract")
        solutions.append({"id": name, "label": label, "description": description,
            "parameters": {"max_gap": gap, "max_children": children, "minimum_score": cutoff},
            "edges": edges, "metrics": graph_metrics(movie["nodes"], edges, movie["truth_edges"]), "checks": checks})
    truth_nodes = movie.pop("truth_nodes")
    result = {"schema": SCHEMA, "config": asdict(config), **movie, "candidates": candidates,
              "solutions": solutions, "default_solution": "balanced"}
    result["summary"] = {"frames": config.frame_count, "truth_nodes": truth_nodes,
        "observed_true_nodes": sum(not node["is_false_positive"] for node in movie["nodes"]),
        "missed_detections": sum(event["type"] == "missed_detection" for event in movie["events"]),
        "false_positive_detections": sum(node["is_false_positive"] for node in movie["nodes"]),
        "detections": len(movie["nodes"]), "candidates": len(candidates), "truth_edges": len(movie["truth_edges"]),
        "truth_divisions": len(_divisions({(edge["source"], edge["target"]) for edge in movie["truth_edges"]}))}
    result["provenance"] = {
        "evidence_type": "SYNTHETIC_ONLY", "generator_seed": config.seed,
        "authored_scope": "Synthetic image sequence, perturbed observations, candidate-generation pipeline, solver integration, gap-closing logic, graph validation and interactive evidence fixture.",
        "image_scope": "Generated Gaussian fluorescence XY projections with noise. Observations are synthetic perturbed centroids, not learned detector outputs. z is an authored synthetic coordinate.",
        "annotation_scope": "track_id, observed and is_false_positive are synthetic reference annotations, never predicted track identities or association inputs. Predicted lineages are defined by selected edges.",
        "metric_scope": "Exact-ID edge and parent/daughter-pair agreement on the observable synthetic reference, contracting deliberately missing detections. These diagnostics are not official competition scores or biological validation.",
        "score_scope": "Public geometry/appearance utility with illustrative fixed parameters; not a calibrated probability, trained association model or private recipe.",
        "solver": "biohub_tracking.public_graph_methods.solve_constrained_edges",
        "solver_scope": "Existing SciPy MILP enforces parent/child degrees; this demo adds a separate residual-endpoint gap-closing pass. No new optimization algorithm or global two-pass optimality claim.",
        "source_files": {"src/biohub_tracking/synthetic_demo.py": sha(Path(__file__).read_bytes()),
                         "src/biohub_tracking/public_graph_methods.py": sha(Path(public_graph_methods.__file__).read_bytes())},
        "dependencies": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "private_assets_used": False, "training_performed": False, "official_score": None,
        "reproducibility_scope": "Deterministic generated fixture and solver choices in the recorded numerical environment; no cross-version bitwise guarantee."}
    # JSON validation is also the frontend boundary: no NaN/Infinity or opaque arrays.
    canonical(result)
    return result


def _atomic_write(path: Path, raw: bytes):
    if path.is_symlink():
        raise ValueError("Refusing a symlink output")
    fd, temporary = tempfile.mkstemp(prefix=".tracking-demo-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_demo(output: Path, config: DemoConfig = DemoConfig()) -> dict:
    output = Path(output)
    if output.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    result = run_demo(config)
    output.mkdir(parents=True, exist_ok=True)
    raw = canonical(result)
    _atomic_write(output / "data.json", raw + b"\n")
    _atomic_write(output / "data.js", b"window.BIOHUB_DEMO_DATA = " + raw + b";\n")
    return result
