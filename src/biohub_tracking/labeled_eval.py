from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata


def _latest_feature_table(root: Path, stage: str) -> Path:
    receipt = json.loads((root / "outputs" / "features" / f"{stage}_receipt.json").read_text())
    path = root / receipt["table"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_node_matches(root: Path, *, max_distance_um: float, scale_zyx_um: tuple[float, float, float]) -> tuple[pd.DataFrame, dict]:
    pilot_nodes = pd.read_csv(root / "outputs" / "pilot" / "node_features.csv")
    gt = pd.read_csv(root / "outputs" / "labeled" / "gt_nodes.csv")
    sample_receipt = json.loads((root / "outputs" / "pilot" / "sample_receipt.json").read_text())

    frames = [int(v) for v in sample_receipt["frames"]]
    oz, oy, ox = [float(v) for v in sample_receipt["roi_offsets_zyx"]]
    _, rz, ry, rx = [int(v) for v in sample_receipt["roi_shape_tzyx"]]

    gt_roi = gt[
        gt["t"].isin(frames)
        & gt["z"].between(oz, oz + rz - 1)
        & gt["y"].between(oy, oy + ry - 1)
        & gt["x"].between(ox, ox + rx - 1)
    ].copy()

    candidates = pilot_nodes[["node_id", "t", "z", "y", "x"]].copy()
    candidates["z_full"] = candidates["z"] + oz
    candidates["y_full"] = candidates["y"] + oy
    candidates["x_full"] = candidates["x"] + ox

    rows = []
    scale = np.asarray(scale_zyx_um, dtype=float)
    for t in frames:
        c = candidates[candidates["t"] == t].reset_index(drop=True)
        g = gt_roi[gt_roi["t"] == t].reset_index(drop=True)

        frame_rows = {
            int(r.node_id): {
                "candidate_id": int(r.node_id),
                "t": int(t),
                "candidate_z": float(r.z_full),
                "candidate_y": float(r.y_full),
                "candidate_x": float(r.x_full),
                "gt_id": np.nan,
                "distance_um": np.nan,
                "matched": False,
            }
            for r in c.itertuples()
        }

        if len(c) and len(g):
            cp = c[["z_full","y_full","x_full"]].to_numpy(float)
            gp = g[["z","y","x"]].to_numpy(float)
            delta = (cp[:, None, :] - gp[None, :, :]) * scale[None, None, :]
            dist = np.sqrt(np.square(delta).sum(axis=2))
            ri, ci = linear_sum_assignment(dist)
            for a, b in zip(ri, ci, strict=True):
                d = float(dist[a, b])
                if d <= max_distance_um:
                    cand_id = int(c.iloc[a]["node_id"])
                    frame_rows[cand_id]["gt_id"] = int(g.iloc[b]["gt_id"])
                    frame_rows[cand_id]["distance_um"] = d
                    frame_rows[cand_id]["matched"] = True

        rows.extend(frame_rows.values())

    matches = pd.DataFrame(rows).sort_values(["t","candidate_id"]).reset_index(drop=True)
    matched_gt = set(matches.loc[matches["matched"], "gt_id"].dropna().astype(int))
    summary = {
        "candidate_nodes": int(len(matches)),
        "annotated_gt_nodes_in_roi": int(len(gt_roi)),
        "matched_candidates": int(matches["matched"].sum()),
        "matched_gt_nodes": int(len(matched_gt)),
        "annotated_node_recall": (
            float(len(matched_gt) / len(gt_roi)) if len(gt_roi) else float("nan")
        ),
        "max_distance_um": float(max_distance_um),
    }
    return matches, summary


def build_labeled_pairs(root: Path, matches: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pair = pd.read_csv(_latest_feature_table(root, "round2"))
    gt_edges = pd.read_csv(root / "outputs" / "labeled" / "gt_edges.csv")

    match_map = {
        int(r.candidate_id): int(r.gt_id)
        for r in matches[matches["matched"]].itertuples()
    }
    edge_set = set(
        zip(
            gt_edges["source_gt_id"].astype(int),
            gt_edges["target_gt_id"].astype(int),
            strict=True,
        )
    )
    gt_out = set(gt_edges["source_gt_id"].astype(int))
    gt_in = set(gt_edges["target_gt_id"].astype(int))

    labels = []
    reasons = []
    src_gt = []
    tgt_gt = []

    for r in pair.itertuples(index=False):
        s = match_map.get(int(r.source_id))
        t = match_map.get(int(r.target_id))
        src_gt.append(s if s is not None else np.nan)
        tgt_gt.append(t if t is not None else np.nan)

        if s is not None and t is not None and (s, t) in edge_set:
            labels.append(1.0)
            reasons.append("annotated_edge")
        elif s is not None and t is not None and (s in gt_out or t in gt_in):
            labels.append(0.0)
            reasons.append("annotated_conflict")
        else:
            labels.append(np.nan)
            reasons.append("ignored_sparse_supervision")

    out = pair.copy()
    out.insert(4, "source_gt_id", src_gt)
    out.insert(5, "target_gt_id", tgt_gt)
    out.insert(6, "edge_label", labels)
    out.insert(7, "label_reason", reasons)

    counts = out["label_reason"].value_counts(dropna=False).to_dict()
    summary = {
        "candidate_pairs": int(len(out)),
        "positive_pairs": int((out["edge_label"] == 1).sum()),
        "negative_pairs": int((out["edge_label"] == 0).sum()),
        "ignored_pairs": int(out["edge_label"].isna().sum()),
        "label_reason_counts": {str(k): int(v) for k, v in counts.items()},
        "supervision": "strict sparse edge labels; ignored rows are not treated as negatives",
    }
    return out, summary


def add_round1_pair_pool(root: Path, labeled_pairs: pd.DataFrame) -> pd.DataFrame:
    """Join Round-1 node descriptors and construct pairwise absolute differences.

    The difference block is built as one DataFrame and concatenated once. This avoids
    pandas block fragmentation from inserting ~128 columns one at a time.
    """
    node = pd.read_csv(_latest_feature_table(root, "round1"))
    features = [c for c in node.columns if c.startswith("r1_")]

    src = node[["node_id"] + features].rename(
        columns={"node_id": "source_id", **{c: f"src__{c}" for c in features}}
    )
    tgt = node[["node_id"] + features].rename(
        columns={"node_id": "target_id", **{c: f"tgt__{c}" for c in features}}
    )

    merged = labeled_pairs.merge(
        src,
        on="source_id",
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        tgt,
        on="target_id",
        how="left",
        validate="many_to_one",
    )

    absdiff = pd.DataFrame(
        {
            f"absdiff__{c}": (
                merged[f"src__{c}"] - merged[f"tgt__{c}"]
            ).abs()
            for c in features
        },
        index=merged.index,
    )
    return pd.concat([merged, absdiff], axis=1).copy()



def _auc_from_values(values: np.ndarray, labels: np.ndarray) -> float:
    pos = values[labels == 1]
    neg = values[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]), method="average")
    pos_rank_sum = ranks[:n_pos].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def screen_features(table: pd.DataFrame, *, min_per_class: int = 3) -> pd.DataFrame:
    labeled = table[table["edge_label"].isin([0.0, 1.0])].copy()
    exclude = {
        "source_id","target_id","t_source","t_target","source_gt_id","target_gt_id",
        "edge_label","label_reason",
    }

    rows = []
    for col in labeled.columns:
        if col in exclude or not pd.api.types.is_numeric_dtype(labeled[col]):
            continue
        part = labeled[[col, "edge_label"]].replace([np.inf, -np.inf], np.nan).dropna()
        n_pos = int((part["edge_label"] == 1).sum())
        n_neg = int((part["edge_label"] == 0).sum())
        if n_pos < min_per_class or n_neg < min_per_class:
            continue
        vals = part[col].to_numpy(float)
        labs = part["edge_label"].to_numpy(int)
        if np.unique(vals).size < 2:
            continue
        auc = _auc_from_values(vals, labs)
        rows.append({
            "feature": col,
            "n": len(part),
            "positive_n": n_pos,
            "negative_n": n_neg,
            "auc": auc,
            "signal_strength": abs(auc - 0.5) * 2,
            "positive_median": float(np.median(vals[labs == 1])),
            "negative_median": float(np.median(vals[labs == 0])),
            "scope": "single-sample sparse-label diagnostic only",
        })
    if not rows:
        return pd.DataFrame(columns=[
            "feature","n","positive_n","negative_n","auc","signal_strength",
            "positive_median","negative_median","scope"
        ])
    return pd.DataFrame(rows).sort_values(
        ["signal_strength","n"], ascending=[False,False]
    ).reset_index(drop=True)


def family_from_feature(name: str) -> str:
    name = str(name)
    if name.startswith("r2_"):
        return name.split("__", 1)[0].replace("r2_", "round2/")
    for prefix in ("src__r1_", "tgt__r1_", "absdiff__r1_"):
        if name.startswith(prefix):
            core = name[len(prefix):].split("__", 1)[0]
            pool = prefix.split("__",1)[0]
            return f"round1/{core}/{pool}"
    return "other"
