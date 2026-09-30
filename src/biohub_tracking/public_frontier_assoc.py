from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit as scipy_logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .annotation_eval import (
    _pilot_features_for_annotation,
    _pool_round1,
    _round_tables,
    chunk_keys_for_roi,
    label_annotation_pairs,
    match_annotation_nodes,
)
from .feature_io import atomic, write_json
from .multiwindow_eval import enumerate_windows, select_diverse_windows
from .multisample_scout import _download_one, read_json, reset_rate_limit_budget
from .public_graph_methods import solve_constrained_edges
from .public_solution_features import (
    add_candidate_competition_context,
    add_kinematic_context,
    agreement_gated_fusion,
    dual_seed_logit_blend,
)


LEAKAGE_COLUMNS = {
    "edge_label",
    "label_reason",
    "source_gt_id",
    "target_gt_id",
    "sample_id",
    "window_id",
    "source_id",
    "target_id",
}


def _sample_cfg(cfg: dict, sample_id: str) -> dict:
    matches = [x for x in cfg["development_samples"] if x["sample_id"] == sample_id]
    if len(matches) != 1:
        raise KeyError(f"Expected one development sample config for {sample_id}.")
    return matches[0]


def _network_policy(root: Path) -> dict:
    scout_cfg = read_json(root / "configs/multisample_scout.json")
    return {
        "request_interval_seconds": float(
            scout_cfg.get("request_interval_seconds", 4.0)
        ),
        "retry_delays_seconds": tuple(
            scout_cfg.get("retry_delays_seconds", [15.0, 30.0, 60.0])
        ),
        "max_cumulative_retry_sleep_seconds": float(
            scout_cfg.get("max_cumulative_retry_sleep_seconds", 105.0)
        ),
    }


def _download(root: Path, remote: str) -> Path:
    destination = root / "data/multisample_scout"
    policy = _network_policy(root)
    return _download_one(
        "biohub-cell-tracking-during-development",
        remote,
        destination,
        request_interval_seconds=policy["request_interval_seconds"],
        retry_delays_seconds=policy["retry_delays_seconds"],
        max_cumulative_retry_sleep_seconds=policy[
            "max_cumulative_retry_sleep_seconds"
        ],
    )


def _ensure_image_metadata(root: Path, sample_id: str) -> tuple[dict, dict]:
    prefix = f"train/{sample_id}.zarr/"
    root_meta = _download(root, prefix + "zarr.json")
    array_meta = _download(root, prefix + "0/zarr.json")
    return read_json(root_meta), read_json(array_meta)


def _sample_gt(root: Path, sample_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_dir = root / "outputs/multisample_scout/samples" / sample_id
    nodes = pd.read_csv(sample_dir / "gt_nodes.csv")
    edges = pd.read_csv(sample_dir / "gt_edges.csv")
    return nodes, edges


def prepare_public_frontier_plan(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/public_frontier_association.json")
    reset_rate_limit_budget()

    out = root / "outputs/public_frontier_assoc"
    out.mkdir(parents=True, exist_ok=True)

    all_windows = []
    unique_chunk_refs = set()
    total_decoded = 0

    for sample in cfg["development_samples"]:
        sid = sample["sample_id"]
        _, array_meta = _ensure_image_metadata(root, sid)
        shape = array_meta["shape"]
        gt_nodes, gt_edges = _sample_gt(root, sid)

        candidates = enumerate_windows(
            gt_nodes,
            gt_edges,
            shape_tzyx=shape,
            roi_shape=cfg["roi_shape_zyx"],
            spacing_um=cfg["spacing_um"],
            frame_count=cfg["frame_count"],
            descriptor_margin_um=cfg["descriptor_margin_um"],
            pair_radius_um=cfg["pair_radius_um"],
        )
        selected, selection_summary = select_diverse_windows(
            candidates,
            max_windows=sample["max_windows"],
            target_unique_positive_links=sample["target_positive_links"],
            target_unique_negative_links=sample["target_negative_links"],
        )

        for i, row in enumerate(selected, start=1):
            row = dict(row)
            row["sample_id"] = sid
            row["role"] = sample["role"]
            row["window_id"] = f"{sid}__w{i:02d}"
            keys = chunk_keys_for_roi(
                array_meta,
                row["frames"],
                row["roi_offsets_zyx"],
                cfg["roi_shape_zyx"],
            )
            row["chunk_keys"] = keys
            row["chunk_file_count"] = len(keys)
            decoded = (
                len(row["frames"])
                * int(np.prod(cfg["roi_shape_zyx"]))
                * np.dtype(array_meta["data_type"]).itemsize
            )
            row["selected_decoded_bytes"] = int(decoded)
            total_decoded += int(decoded)
            for key in keys:
                unique_chunk_refs.add((sid, key))
            all_windows.append(row)

        write_json(
            out / f"{sid}_selection_summary.json",
            {
                "sample_id": sid,
                **selection_summary,
            },
        )

    if len(unique_chunk_refs) > int(cfg["max_unique_image_chunks"]):
        raise RuntimeError(
            f"Planned {len(unique_chunk_refs)} unique image chunks above cap "
            f"{cfg['max_unique_image_chunks']}."
        )
    if total_decoded > int(cfg["max_total_decoded_roi_bytes"]):
        raise RuntimeError("Decoded ROI plan exceeds configured cap.")

    table = pd.DataFrame([
        {
            "sample_id": row["sample_id"],
            "window_id": row["window_id"],
            "frames": "-".join(str(v) for v in row["frames"]),
            "roi_z": row["roi_offsets_zyx"][0],
            "roi_y": row["roi_offsets_zyx"][1],
            "roi_x": row["roi_offsets_zyx"][2],
            "safe_gt_nodes": row["safe_gt_nodes"],
            "safe_gt_edges": row["safe_gt_edges"],
            "potential_positive_links": row["potential_positive_links"],
            "potential_negative_links": row["potential_negative_links"],
            "chunk_file_count": row["chunk_file_count"],
        }
        for row in all_windows
    ])
    atomic(out / "window_plan.csv", table.to_csv(index=False))

    plan = {
        "status": "planned",
        "windows": all_windows,
        "unique_image_chunks": len(unique_chunk_refs),
        "total_decoded_roi_bytes": int(total_decoded),
        "training_fits": 0,
        "official_score": None,
        "validation_scope": cfg["validation_scope"],
    }
    write_json(out / "plan.json", plan)
    return plan


def acquire_public_frontier_windows(root: Path) -> dict:
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r"crc32c usage is deprecated since numcodecs v0\.16\.4.*",
        category=DeprecationWarning,
        module=r"numcodecs(\..*)?",
    )
    import zarr

    root = Path(root).resolve()
    cfg = read_json(root / "configs/public_frontier_association.json")
    plan = read_json(root / "outputs/public_frontier_assoc/plan.json")
    reset_rate_limit_budget()

    cache_root = root / "data/public_frontier_windows"
    cache_root.mkdir(parents=True, exist_ok=True)

    remote_refs = []
    for row in plan["windows"]:
        sid = row["sample_id"]
        for key in row["chunk_keys"]:
            remote_refs.append((sid, key))
    remote_refs = list(dict.fromkeys(remote_refs))

    downloaded = 0
    reused = 0
    for sid, key in remote_refs:
        target = (
            root / "data/multisample_scout/train" / f"{sid}.zarr" / key
        )
        existed = target.is_file() and target.stat().st_size > 0
        _download(root, f"train/{sid}.zarr/{key}")
        if existed:
            reused += 1
        else:
            downloaded += 1

    artifacts = []
    total_cache = 0
    for row in plan["windows"]:
        sid = row["sample_id"]
        store = root / "data/multisample_scout/train" / f"{sid}.zarr"
        arr = zarr.open_array(store / "0", mode="r")

        offsets = np.asarray(row["roi_offsets_zyx"], dtype=int)
        rz, ry, rx = [int(v) for v in cfg["roi_shape_zyx"]]
        slices = (
            slice(int(offsets[0]), int(offsets[0] + rz)),
            slice(int(offsets[1]), int(offsets[1] + ry)),
            slice(int(offsets[2]), int(offsets[2] + rx)),
        )

        rois = []
        for t in row["frames"]:
            roi = np.asarray(arr[int(t), slices[0], slices[1], slices[2]])
            if list(roi.shape) != cfg["roi_shape_zyx"]:
                raise ValueError(
                    f"{row['window_id']}: unexpected ROI shape {roi.shape}."
                )
            if not np.isfinite(roi).all() or roi.max() <= roi.min():
                raise ValueError(f"{row['window_id']}: invalid/flat ROI.")
            rois.append(np.array(roi, copy=True))

        sample_dir = cache_root / sid
        sample_dir.mkdir(parents=True, exist_ok=True)
        path = sample_dir / f"{row['window_id']}.npz"
        tmp = path.with_name(path.name + ".partial")
        with tmp.open("wb") as f:
            np.savez_compressed(
                f,
                images=np.stack(rois),
                frames=np.asarray(row["frames"], dtype=int),
                offsets=offsets,
                spacing_um=np.asarray(cfg["spacing_um"], dtype=float),
            )
        os.replace(tmp, path)
        artifacts.append({
            "sample_id": sid,
            "window_id": row["window_id"],
            "path": str(path.relative_to(root)),
        })

    total_cache = sum(
        p.stat().st_size
        for p in cache_root.rglob("*")
        if p.is_file()
    )
    if total_cache > int(cfg["max_snapshot_bytes"]):
        raise RuntimeError("Public-frontier window cache exceeded cap.")

    receipt = {
        "status": "completed",
        "downloaded_chunks": downloaded,
        "reused_chunks": reused,
        "unique_chunk_refs": len(remote_refs),
        "window_artifacts": artifacts,
        "window_cache_bytes": int(total_cache),
        "training_fits": 0,
    }
    write_json(
        root / "outputs/public_frontier_assoc/acquire_receipt.json",
        receipt,
    )
    return receipt


def _evaluate_window(
    root: Path,
    row: dict,
    cache_path: Path,
) -> tuple[pd.DataFrame, dict]:
    with np.load(cache_path, allow_pickle=False) as z:
        images = z["images"]
        frames = z["frames"]
        offsets = z["offsets"]
        spacing = z["spacing_um"]

    nodes, pairs, _, _ = _pilot_features_for_annotation(
        root, images, frames, offsets, spacing
    )
    r1, r2, _ = _round_tables(
        root, images, frames, offsets, spacing, nodes, pairs
    )
    gt_nodes, gt_edges = _sample_gt(root, row["sample_id"])

    plan_like = {
        "frames": [int(v) for v in frames],
        "roi_offsets_zyx": [int(v) for v in offsets],
        "roi_shape_zyx": list(images.shape[1:]),
        "spacing_um": [float(v) for v in spacing],
    }
    matches, gt_roi, node_summary = match_annotation_nodes(
        nodes, gt_nodes, plan_like
    )
    labeled = label_annotation_pairs(r2, matches, gt_edges)

    # Public-solution-inspired CPU context, reimplemented locally.
    context = add_candidate_competition_context(
        labeled,
        score_col="r2_geometry__distance",
        higher_is_better=False,
    )
    context = add_kinematic_context(
        context,
        nodes,
        scale_zyx_um=tuple(float(v) for v in spacing),
    )
    pooled = _pool_round1(r1, context)
    pooled["sample_id"] = row["sample_id"]
    pooled["window_id"] = row["window_id"]

    gt_ids = set(gt_roi["gt_id"].astype(int))
    gt_edges_roi = gt_edges[
        gt_edges["source_gt_id"].astype(int).isin(gt_ids)
        & gt_edges["target_gt_id"].astype(int).isin(gt_ids)
    ]
    represented = set(
        zip(
            labeled.loc[labeled["edge_label"] == 1, "source_gt_id"].astype(int),
            labeled.loc[labeled["edge_label"] == 1, "target_gt_id"].astype(int),
            strict=True,
        )
    )
    edge_recall = (
        float(len(represented) / len(gt_edges_roi))
        if len(gt_edges_roi)
        else None
    )

    summary = {
        "sample_id": row["sample_id"],
        "window_id": row["window_id"],
        "candidate_nodes": int(len(nodes)),
        "candidate_pairs": int(len(labeled)),
        "annotated_gt_nodes": int(node_summary["annotated_gt_nodes_in_roi"]),
        "matched_gt_nodes": int(node_summary["matched_gt_nodes"]),
        "annotated_node_recall": node_summary["annotated_node_recall"],
        "gt_edges_in_roi": int(len(gt_edges_roi)),
        "represented_gt_edges": int(len(represented)),
        "candidate_edge_recall": edge_recall,
        "positive_pairs": int((labeled["edge_label"] == 1).sum()),
        "negative_pairs": int((labeled["edge_label"] == 0).sum()),
        "ignored_pairs": int(labeled["edge_label"].isna().sum()),
    }
    return pooled, summary


def build_public_frontier_dataset(root: Path) -> dict:
    root = Path(root).resolve()
    plan = read_json(root / "outputs/public_frontier_assoc/plan.json")
    acquire = read_json(
        root / "outputs/public_frontier_assoc/acquire_receipt.json"
    )
    artifact_map = {
        (x["sample_id"], x["window_id"]): root / x["path"]
        for x in acquire["window_artifacts"]
    }

    pooled_parts = []
    summaries = []
    for row in plan["windows"]:
        pooled, summary = _evaluate_window(
            root,
            row,
            artifact_map[(row["sample_id"], row["window_id"])],
        )
        pooled_parts.append(pooled)
        summaries.append(summary)

    all_rows = pd.concat(pooled_parts, ignore_index=True)
    strict = all_rows[all_rows["edge_label"].isin([0.0, 1.0])].copy()

    before = int(len(strict))
    strict = strict.drop_duplicates(
        subset=[
            "sample_id",
            "source_gt_id",
            "target_gt_id",
            "edge_label",
        ],
        keep="first",
    ).reset_index(drop=True)

    out = root / "outputs/public_frontier_assoc"
    atomic(out / "window_summary.csv", pd.DataFrame(summaries).to_csv(index=False))
    atomic(out / "candidate_dataset.csv", all_rows.to_csv(index=False))
    atomic(out / "strict_labeled_dataset.csv", strict.to_csv(index=False))

    r2_cols = [c for c in strict.columns if c.startswith("r2_")]
    r1_base = set()
    for c in strict.columns:
        for prefix in ("src__r1_", "tgt__r1_", "absdiff__r1_"):
            if c.startswith(prefix):
                r1_base.add(c.split(prefix, 1)[1])

    publicctx_cols = [c for c in strict.columns if c.startswith("publicctx_")]

    sample_stats = (
        strict.groupby("sample_id", as_index=False)
        .agg(
            total=("edge_label", "size"),
            positive=("edge_label", "sum"),
        )
    )
    sample_stats["positive"] = sample_stats["positive"].astype(int)
    sample_stats["negative"] = (
        sample_stats["total"] - sample_stats["positive"]
    ).astype(int)
    atomic(out / "strict_label_counts.csv", sample_stats.to_csv(index=False))

    receipt = {
        "status": "completed",
        "candidate_rows": int(len(all_rows)),
        "strict_rows_before_dedup": before,
        "strict_rows_after_dedup": int(len(strict)),
        "positive_rows": int((strict["edge_label"] == 1).sum()),
        "negative_rows": int((strict["edge_label"] == 0).sum()),
        "round1_base_descriptors_present": int(len(r1_base)),
        "round2_descriptors_present": int(len(r2_cols)),
        "public_context_features": int(len(publicctx_cols)),
        "training_fits": 0,
        "official_score": None,
    }
    if receipt["round1_base_descriptors_present"] != 128:
        raise ValueError(
            "Not all 128 Round-1 base descriptors are represented in pair expansions."
        )
    if receipt["round2_descriptors_present"] != 128:
        raise ValueError("Not all 128 Round-2 descriptors are present.")
    write_json(out / "dataset_receipt.json", receipt)
    return receipt


def _feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in LEAKAGE_COLUMNS:
            continue
        if c in {"t_source", "t_target"}:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    if not cols:
        raise ValueError("No numeric model features found.")
    return cols


def _clean_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = df[features].copy()
    return x.replace([np.inf, -np.inf], np.nan)


def _sample_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    n = len(y)
    counts = np.bincount(y, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both classes are required.")
    return np.where(
        y == 1,
        n / (2.0 * counts[1]),
        n / (2.0 * counts[0]),
    ).astype(float)


def _binary_metrics(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
    }


def _jaccard_for_threshold(y, p, threshold: float) -> float:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(p, dtype=float) >= float(threshold)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    den = tp + fp + fn
    return float(tp / den) if den else 0.0


def _choose_threshold(y, p, grid) -> float:
    scored = [
        (_jaccard_for_threshold(y, p, t), float(t))
        for t in grid
    ]
    scored.sort(key=lambda x: (x[0], -abs(x[1] - 0.5)), reverse=True)
    return scored[0][1]


def _local_metrics(y, p, threshold) -> dict:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(p, dtype=float) >= float(threshold)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        pred.astype(int),
        average="binary",
        zero_division=0,
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "labeled_edge_jaccard": _jaccard_for_threshold(y, p, threshold),
    }


def _fit_models(X_train, y_train, cfg):
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                C=float(cfg["models"]["logistic_full"]["C"]),
                max_iter=int(cfg["models"]["logistic_full"]["max_iter"]),
                class_weight="balanced",
                solver="lbfgs",
                random_state=20260916,
            ),
        ),
    ])
    logistic.fit(X_train, y_train)

    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    Xh = imputer.fit_transform(X_train)
    hgb = HistGradientBoostingClassifier(
        learning_rate=float(
            cfg["models"]["hist_gradient_boosting_full"]["learning_rate"]
        ),
        max_iter=int(
            cfg["models"]["hist_gradient_boosting_full"]["max_iter"]
        ),
        max_leaf_nodes=int(
            cfg["models"]["hist_gradient_boosting_full"]["max_leaf_nodes"]
        ),
        l2_regularization=float(
            cfg["models"]["hist_gradient_boosting_full"]["l2_regularization"]
        ),
        random_state=20260916,
    )
    hgb.fit(Xh, y_train, sample_weight=_sample_weights(y_train))
    return logistic, (imputer, hgb)


def _predict_hgb(bundle, X):
    imputer, model = bundle
    return model.predict_proba(imputer.transform(X))[:, 1]


def _decode_one_window(group: pd.DataFrame, score_col: str, threshold: float):
    g = group.copy()
    p = np.clip(g[score_col].to_numpy(float), 1e-6, 1 - 1e-6)
    cutoff = float(np.clip(threshold, 1e-6, 1 - 1e-6))
    g["milp_score"] = scipy_logit(p) - float(scipy_logit(cutoff))
    solved = solve_constrained_edges(
        g[["source_id", "target_id", "milp_score"]].copy(),
        score_col="milp_score",
        max_out_degree=2,
    )
    return solved["selected"].to_numpy(bool)


def _decoded_labeled_metrics(frame: pd.DataFrame, score_col: str, threshold: float):
    selected = np.zeros(len(frame), dtype=bool)
    for _, idx in frame.groupby("window_id").groups.items():
        positions = np.asarray(list(idx), dtype=int)
        selected[positions] = _decode_one_window(
            frame.loc[positions],
            score_col,
            threshold,
        )

    strict = frame["edge_label"].isin([0.0, 1.0]).to_numpy()
    y = frame.loc[strict, "edge_label"].astype(int).to_numpy()
    pred = selected[strict]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        pred.astype(int),
        average="binary",
        zero_division=0,
    )
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    den = tp + fp + fn
    return {
        "decoded_precision": float(precision),
        "decoded_recall": float(recall),
        "decoded_f1": float(f1),
        "decoded_labeled_edge_jaccard": float(tp / den) if den else 0.0,
        "decoded_selected_edges": int(pred.sum()),
    }


def run_public_frontier_ablation(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/public_frontier_association.json")
    out = root / "outputs/public_frontier_assoc"

    strict = pd.read_csv(out / "strict_labeled_dataset.csv")
    candidates = pd.read_csv(out / "candidate_dataset.csv")
    features = _feature_columns(strict)

    class_gate = (
        strict.groupby("sample_id")["edge_label"]
        .agg(["count", "sum"])
        .reset_index()
    )
    class_gate["positive"] = class_gate["sum"].astype(int)
    class_gate["negative"] = (
        class_gate["count"] - class_gate["positive"]
    ).astype(int)
    if class_gate["positive"].min() <= 0 or class_gate["negative"].min() <= 0:
        receipt = {
            "status": "blocked_pretraining",
            "training_fits": 0,
            "reason": "each sample-disjoint fold requires positive and negative strict links",
            "sample_class_counts": class_gate[
                ["sample_id", "positive", "negative"]
            ].to_dict(orient="records"),
            "next_decision": "run_candidate_coverage_diagnostic",
            "official_score": None,
        }
        write_json(out / "ablation_receipt.json", receipt)
        return receipt

    fits = 0
    metric_rows = []
    coefficient_rows = []
    prediction_parts = []

    samples = [x["sample_id"] for x in cfg["development_samples"]]
    for valid_sid in samples:
        train_sid = [x for x in samples if x != valid_sid][0]
        train = strict[strict["sample_id"] == train_sid].copy()
        valid = strict[strict["sample_id"] == valid_sid].copy()
        valid_all = candidates[candidates["sample_id"] == valid_sid].copy()

        y_train = train["edge_label"].astype(int).to_numpy()
        y_valid = valid["edge_label"].astype(int).to_numpy()
        if len(np.unique(y_train)) != 2 or len(np.unique(y_valid)) != 2:
            raise ValueError(
                f"Both classes required in sample-disjoint fold: "
                f"train={train_sid} valid={valid_sid}."
            )

        X_train = _clean_X(train, features)
        X_valid = _clean_X(valid, features)
        X_valid_all = _clean_X(valid_all, features)

        logistic, hgb_bundle = _fit_models(X_train, y_train, cfg)
        fits += 2

        p_train_log = logistic.predict_proba(X_train)[:, 1]
        p_valid_log = logistic.predict_proba(X_valid)[:, 1]
        p_all_log = logistic.predict_proba(X_valid_all)[:, 1]

        p_train_hgb = _predict_hgb(hgb_bundle, X_train)
        p_valid_hgb = _predict_hgb(hgb_bundle, X_valid)
        p_all_hgb = _predict_hgb(hgb_bundle, X_valid_all)

        variants = {
            "logistic_full": (p_train_log, p_valid_log, p_all_log),
            "hist_gradient_boosting_full": (
                p_train_hgb,
                p_valid_hgb,
                p_all_hgb,
            ),
            "logit_blend_50_50": (
                dual_seed_logit_blend(p_train_log, p_train_hgb, weight_a=0.5),
                dual_seed_logit_blend(p_valid_log, p_valid_hgb, weight_a=0.5),
                dual_seed_logit_blend(p_all_log, p_all_hgb, weight_a=0.5),
            ),
            "agreement_gate_20pct": (
                agreement_gated_fusion(
                    p_train_log,
                    p_train_hgb,
                    logit_weight=0.5,
                    max_abs_prob_disagreement=0.20,
                ),
                agreement_gated_fusion(
                    p_valid_log,
                    p_valid_hgb,
                    logit_weight=0.5,
                    max_abs_prob_disagreement=0.20,
                ),
                agreement_gated_fusion(
                    p_all_log,
                    p_all_hgb,
                    logit_weight=0.5,
                    max_abs_prob_disagreement=0.20,
                ),
            ),
        }

        fold_all = valid_all.copy().reset_index(drop=True)
        for variant, (p_train, p_valid, p_all) in variants.items():
            threshold = _choose_threshold(
                y_train, p_train, cfg["threshold_grid"]
            )
            metrics = {
                **_binary_metrics(y_valid, p_valid),
                **_local_metrics(y_valid, p_valid, threshold),
            }
            score_col = f"score__{variant}"
            fold_all[score_col] = p_all
            decoded = _decoded_labeled_metrics(
                fold_all,
                score_col,
                threshold,
            )
            metric_rows.append({
                "train_sample": train_sid,
                "valid_sample": valid_sid,
                "variant": variant,
                **metrics,
                **decoded,
            })

        # Coefficients are diagnostics only; no feature cap is applied to training.
        logistic_model = logistic.named_steps["model"]
        imputer = logistic.named_steps["imputer"]
        transformed_names = list(imputer.get_feature_names_out(features))
        scaler = logistic.named_steps["scaler"]
        # scaler keeps the same transformed feature ordering
        for name, coef in zip(
            transformed_names,
            logistic_model.coef_[0],
            strict=False,
        ):
            coefficient_rows.append({
                "train_sample": train_sid,
                "valid_sample": valid_sid,
                "feature": str(name),
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
            })

        prediction_parts.append(fold_all)

    if fits > int(cfg["training_fits_cap"]):
        raise RuntimeError(
            f"Training fit count {fits} exceeded cap {cfg['training_fits_cap']}."
        )

    metrics = pd.DataFrame(metric_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)

    atomic(out / "model_metrics.csv", metrics.to_csv(index=False))
    atomic(
        out / "logistic_coefficients.csv",
        coefficients.to_csv(index=False),
    )
    atomic(
        out / "validation_candidate_scores.csv",
        predictions.to_csv(index=False),
    )

    aggregate = (
        metrics.groupby("variant", as_index=False)
        .agg(
            folds=("valid_sample", "size"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_average_precision=("average_precision", "mean"),
            mean_brier=("brier", "mean"),
            mean_local_jaccard=("labeled_edge_jaccard", "mean"),
            mean_decoded_jaccard=("decoded_labeled_edge_jaccard", "mean"),
            mean_decoded_f1=("decoded_f1", "mean"),
        )
        .sort_values(
            ["mean_decoded_jaccard", "mean_average_precision"],
            ascending=False,
        )
    )
    atomic(out / "model_aggregate.csv", aggregate.to_csv(index=False))

    best = aggregate.iloc[0].to_dict()
    gate = cfg["public_frontier_gate"]
    cpu_gate = (
        float(best["mean_average_precision"])
        >= float(gate["cpu_gate_min_mean_ap"])
        and float(best["mean_decoded_jaccard"])
        >= float(gate["cpu_gate_min_mean_labeled_edge_jaccard"])
    )

    receipt = {
        "status": "completed",
        "training_fits": fits,
        "feature_columns_used": len(features),
        "all_round1_base_descriptors_considered": True,
        "all_round2_descriptors_considered": True,
        "best_variant": str(best["variant"]),
        "best_mean_average_precision": float(best["mean_average_precision"]),
        "best_mean_decoded_labeled_edge_jaccard": float(
            best["mean_decoded_jaccard"]
        ),
        "cpu_gate_passed": bool(cpu_gate),
        "next_decision": (
            "gpu_public_reference_reproduction"
            if cpu_gate
            else "inspect_cpu_ablation_before_gpu"
        ),
        "official_score": None,
        "validation_scope": cfg["validation_scope"],
        "harmonic_forward_reverse_status": (
            "source-ready only; true harmonic fusion requires genuine forward/reverse "
            "learned logits from the GPU public-reference lane"
        ),
    }
    write_json(out / "ablation_receipt.json", receipt)
    return receipt
