from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .annotation_eval import _pilot_features_for_annotation, match_annotation_nodes
from .feature_io import atomic, write_json
from .multiwindow_eval import _strict_gt_pair_opportunities


def classify_candidate_bottleneck(planned_negatives, negatives_with_matched_endpoints, represented_negative_pairs):
    if planned_negatives <= 0:
        return "no_negative_opportunities_in_selected_windows"
    if negatives_with_matched_endpoints <= 0:
        return "detector_or_gt_matching_coverage"
    if represented_negative_pairs <= 0:
        return "candidate_pair_generator_pruning"
    if represented_negative_pairs < negatives_with_matched_endpoints:
        return "partial_candidate_pair_generator_coverage"
    return "negative_candidate_coverage_available"


def diagnose_public_frontier_candidate_coverage(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = json.loads((root / "configs/public_frontier_association.json").read_text())
    plan = json.loads((root / "outputs/public_frontier_assoc/plan.json").read_text())
    acquire = json.loads((root / "outputs/public_frontier_assoc/acquire_receipt.json").read_text())

    artifact_map = {
        (x["sample_id"], x["window_id"]): root / x["path"]
        for x in acquire["window_artifacts"]
    }

    rows = []
    union_planned_positive = set()
    union_planned_negative = set()
    union_matched_positive = set()
    union_matched_negative = set()
    union_represented_positive = set()
    union_represented_negative = set()
    gt_cache = {}

    for row in plan["windows"]:
        sid = row["sample_id"]
        wid = row["window_id"]

        if sid not in gt_cache:
            base = root / "outputs/multisample_scout/samples" / sid
            gt_cache[sid] = (
                pd.read_csv(base / "gt_nodes.csv"),
                pd.read_csv(base / "gt_edges.csv"),
            )
        gt_nodes, gt_edges = gt_cache[sid]

        with np.load(artifact_map[(sid, wid)], allow_pickle=False) as z:
            images = z["images"]
            frames = z["frames"]
            offsets = z["offsets"]
            spacing = z["spacing_um"]

        nodes, pairs, _, _ = _pilot_features_for_annotation(
            root, images, frames, offsets, spacing
        )

        plan_like = {
            "frames": [int(v) for v in frames],
            "roi_offsets_zyx": [int(v) for v in offsets],
            "roi_shape_zyx": list(images.shape[1:]),
            "spacing_um": [float(v) for v in spacing],
        }
        matches, _, node_summary = match_annotation_nodes(
            nodes, gt_nodes, plan_like
        )

        opp = _strict_gt_pair_opportunities(
            gt_nodes,
            gt_edges,
            frames=[int(v) for v in frames],
            offsets=np.asarray(offsets, dtype=int),
            roi_shape=np.asarray(images.shape[1:], dtype=int),
            spacing_um=np.asarray(spacing, dtype=float),
            descriptor_margin_um=cfg["descriptor_margin_um"],
            pair_radius_um=cfg["pair_radius_um"],
        )

        positive = {tuple(v) for v in opp["positive_pairs"]}
        negative = {tuple(v) for v in opp["negative_pairs"]}

        matched = matches[matches["matched"]].copy()
        candidate_to_gt = {
            int(r.candidate_id): int(r.gt_id)
            for r in matched.itertuples()
        }
        matched_gt_ids = set(candidate_to_gt.values())

        pair_gt = set()
        for p in pairs.itertuples():
            s = int(p.source_id)
            t = int(p.target_id)
            if s in candidate_to_gt and t in candidate_to_gt:
                pair_gt.add((candidate_to_gt[s], candidate_to_gt[t]))

        positive_matched = {
            x for x in positive
            if x[0] in matched_gt_ids and x[1] in matched_gt_ids
        }
        negative_matched = {
            x for x in negative
            if x[0] in matched_gt_ids and x[1] in matched_gt_ids
        }
        positive_repr = positive_matched & pair_gt
        negative_repr = negative_matched & pair_gt

        tag = lambda pairs_: {(sid, int(a), int(b)) for a, b in pairs_}
        union_planned_positive |= tag(positive)
        union_planned_negative |= tag(negative)
        union_matched_positive |= tag(positive_matched)
        union_matched_negative |= tag(negative_matched)
        union_represented_positive |= tag(positive_repr)
        union_represented_negative |= tag(negative_repr)

        rows.append({
            "sample_id": sid,
            "window_id": wid,
            "candidate_nodes": int(len(nodes)),
            "candidate_pairs": int(len(pairs)),
            "annotated_gt_nodes": int(node_summary["annotated_gt_nodes_in_roi"]),
            "matched_gt_nodes": int(node_summary["matched_gt_nodes"]),
            "annotated_node_recall": node_summary["annotated_node_recall"],
            "planned_positive_pairs": len(positive),
            "planned_negative_pairs": len(negative),
            "positive_pairs_with_matched_endpoints": len(positive_matched),
            "negative_pairs_with_matched_endpoints": len(negative_matched),
            "represented_positive_pairs": len(positive_repr),
            "represented_negative_pairs": len(negative_repr),
        })

    table = pd.DataFrame(rows)
    out = root / "outputs/public_frontier_assoc"
    atomic(out / "candidate_coverage_diagnostic.csv", table.to_csv(index=False))

    bottleneck = classify_candidate_bottleneck(
        len(union_planned_negative),
        len(union_matched_negative),
        len(union_represented_negative),
    )

    receipt = {
        "status": "completed",
        "planned_positive_pairs": len(union_planned_positive),
        "planned_negative_pairs": len(union_planned_negative),
        "positive_pairs_with_matched_endpoints": len(union_matched_positive),
        "negative_pairs_with_matched_endpoints": len(union_matched_negative),
        "represented_positive_pairs": len(union_represented_positive),
        "represented_negative_pairs": len(union_represented_negative),
        "bottleneck": bottleneck,
        "training_fits": 0,
        "network_requests": 0,
        "next_decision": (
            "densify_candidate_pair_graph"
            if bottleneck in {
                "candidate_pair_generator_pruning",
                "partial_candidate_pair_generator_coverage",
            }
            else (
                "improve_detector_or_matching"
                if bottleneck == "detector_or_gt_matching_coverage"
                else "association_modeling"
            )
        ),
    }
    write_json(out / "candidate_coverage_receipt.json", receipt)
    return receipt
