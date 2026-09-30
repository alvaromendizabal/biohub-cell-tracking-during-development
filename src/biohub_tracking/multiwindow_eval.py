from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .annotation_eval import (
    _edge_stats,
    _pilot_features_for_annotation,
    _pool_round1,
    _roi_offsets,
    _round_tables,
    _safe_mask,
    chunk_keys_for_roi,
    label_annotation_pairs,
    match_annotation_nodes,
    read_json,
    safe_json_number,
    sha256_file,
)
from .feature_io import atomic, write_json
from .geff_snapshot import _download_one
from .labeled_eval import family_from_feature, screen_features

FEATURE_SCREEN_COLUMNS = [
    "feature",
    "n",
    "positive_n",
    "negative_n",
    "auc",
    "signal_strength",
    "positive_median",
    "negative_median",
    "scope",
    "family",
]

FAMILY_SCREEN_COLUMNS = [
    "family",
    "features_screened",
    "median_signal",
    "max_signal",
]


def _strict_gt_pair_opportunities(
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    *,
    frames,
    offsets,
    roi_shape,
    spacing_um,
    descriptor_margin_um,
    pair_radius_um,
):
    spacing = np.asarray(spacing_um, dtype=float)
    margin = np.ceil(float(descriptor_margin_um) / spacing).astype(int)

    safe_nodes = _safe_mask(
        gt_nodes,
        [int(v) for v in frames],
        np.asarray(offsets, dtype=int),
        np.asarray(roi_shape, dtype=int),
        margin,
    )
    internal_edges, division_sources = _edge_stats(safe_nodes, gt_edges)

    if safe_nodes.empty:
        return {
            "safe_nodes": safe_nodes,
            "internal_edges": internal_edges,
            "division_sources": division_sources,
            "positive_pairs": set(),
            "negative_pairs": set(),
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

    positive = set()
    negative = set()

    for t in frames[:-1]:
        src = safe_nodes[safe_nodes["t"] == int(t)]
        tgt = safe_nodes[safe_nodes["t"] == int(t) + 1]
        if src.empty or tgt.empty:
            continue

        sp = src[["z", "y", "x"]].to_numpy(float)
        tp = tgt[["z", "y", "x"]].to_numpy(float)
        dist = np.linalg.norm(
            (sp[:, None, :] - tp[None, :, :]) * spacing[None, None, :],
            axis=2,
        )

        for i, srow in enumerate(src.itertuples()):
            sid = int(srow.gt_id)
            for j, trow in enumerate(tgt.itertuples()):
                if float(dist[i, j]) > float(pair_radius_um):
                    continue
                tid = int(trow.gt_id)
                pair = (sid, tid)
                if pair in edge_set:
                    positive.add(pair)
                elif sid in gt_out or tid in gt_in:
                    negative.add(pair)

    return {
        "safe_nodes": safe_nodes,
        "internal_edges": internal_edges,
        "division_sources": division_sources,
        "positive_pairs": positive,
        "negative_pairs": negative,
    }


def enumerate_windows(
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    *,
    shape_tzyx,
    roi_shape,
    spacing_um,
    frame_count,
    descriptor_margin_um,
    pair_radius_um,
):
    shape_tzyx = tuple(int(v) for v in shape_tzyx)
    spatial_shape = np.asarray(shape_tzyx[1:], dtype=int)
    roi_shape = np.asarray(roi_shape, dtype=int)
    spacing = np.asarray(spacing_um, dtype=float)

    rows = []
    max_start = shape_tzyx[0] - int(frame_count)
    for start in range(max_start + 1):
        frames = list(range(start, start + int(frame_count)))
        frame_nodes = gt_nodes[gt_nodes["t"].isin(frames)]
        if frame_nodes.empty:
            continue

        centers = frame_nodes[["z", "y", "x"]].to_numpy(float).tolist()
        centers.append(
            frame_nodes[["z", "y", "x"]].median().to_numpy(float).tolist()
        )

        seen_offsets = set()
        for center in centers:
            offsets = _roi_offsets(center, spatial_shape, roi_shape)
            offset_key = tuple(int(v) for v in offsets)
            if offset_key in seen_offsets:
                continue
            seen_offsets.add(offset_key)

            stats = _strict_gt_pair_opportunities(
                gt_nodes,
                gt_edges,
                frames=frames,
                offsets=offsets,
                roi_shape=roi_shape,
                spacing_um=spacing,
                descriptor_margin_um=descriptor_margin_um,
                pair_radius_um=pair_radius_um,
            )
            safe = stats["safe_nodes"]
            if safe.empty:
                continue

            span = (
                safe[["z", "y", "x"]].max()
                - safe[["z", "y", "x"]].min()
            ).to_numpy(float)
            compactness = float(np.linalg.norm(span * spacing))

            rows.append({
                "start_frame": int(start),
                "frames": [int(v) for v in frames],
                "roi_offsets_zyx": [int(v) for v in offsets],
                "safe_gt_nodes": int(len(safe)),
                "safe_gt_edges": int(len(stats["internal_edges"])),
                "safe_division_sources": int(len(stats["division_sources"])),
                "potential_positive_links": int(len(stats["positive_pairs"])),
                "potential_negative_links": int(len(stats["negative_pairs"])),
                "positive_pairs": sorted(
                    [list(x) for x in stats["positive_pairs"]]
                ),
                "negative_pairs": sorted(
                    [list(x) for x in stats["negative_pairs"]]
                ),
                "safe_gt_ids": sorted(int(v) for v in safe["gt_id"]),
                "physical_span_norm_um": compactness,
            })

    if not rows:
        raise ValueError("No labeled multi-window candidates were found.")

    rows.sort(
        key=lambda r: (
            r["potential_negative_links"] > 0,
            r["potential_negative_links"],
            r["potential_positive_links"],
            r["safe_gt_edges"],
            r["safe_gt_nodes"],
            r["safe_division_sources"],
            -r["physical_span_norm_um"],
            -r["start_frame"],
        ),
        reverse=True,
    )
    return rows


def select_diverse_windows(
    candidates,
    *,
    max_windows,
    target_unique_positive_links,
    target_unique_negative_links,
):
    selected = []
    used_keys = set()
    covered_pos = set()
    covered_neg = set()
    covered_gt = set()

    remaining = list(candidates)

    while remaining and len(selected) < int(max_windows):
        best_idx = None
        best_score = None

        for i, row in enumerate(remaining):
            key = (
                tuple(row["frames"]),
                tuple(row["roi_offsets_zyx"]),
            )
            if key in used_keys:
                continue

            pos = {tuple(v) for v in row["positive_pairs"]}
            neg = {tuple(v) for v in row["negative_pairs"]}
            gt = set(int(v) for v in row["safe_gt_ids"])

            marginal_pos = len(pos - covered_pos)
            marginal_neg = len(neg - covered_neg)
            marginal_gt = len(gt - covered_gt)

            if not (marginal_pos or marginal_neg or marginal_gt):
                continue

            temporal_overlap = 0
            for chosen in selected:
                temporal_overlap += len(
                    set(row["frames"]) & set(chosen["frames"])
                )

            score = (
                marginal_neg,
                marginal_pos,
                marginal_gt,
                row["safe_gt_edges"],
                row["safe_division_sources"],
                -temporal_overlap,
                -row["physical_span_norm_um"],
            )
            if best_score is None or score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break

        row = dict(remaining.pop(best_idx))
        row["window_id"] = f"window_{len(selected)+1:02d}"
        selected.append(row)
        used_keys.add(
            (tuple(row["frames"]), tuple(row["roi_offsets_zyx"]))
        )
        covered_pos.update(tuple(v) for v in row["positive_pairs"])
        covered_neg.update(tuple(v) for v in row["negative_pairs"])
        covered_gt.update(int(v) for v in row["safe_gt_ids"])

        if (
            len(covered_pos) >= int(target_unique_positive_links)
            and len(covered_neg) >= int(target_unique_negative_links)
        ):
            break

    if not selected:
        raise ValueError("Greedy multi-window selection returned no windows.")

    summary = {
        "selected_windows": len(selected),
        "planned_unique_positive_links": len(covered_pos),
        "planned_unique_negative_links": len(covered_neg),
        "planned_unique_gt_nodes": len(covered_gt),
        "target_unique_positive_links": int(target_unique_positive_links),
        "target_unique_negative_links": int(target_unique_negative_links),
    }
    return selected, summary


def prepare_multiwindow_plan(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/multiwindow_eval.json")
    pilot_cfg = read_json(root / "configs/pilot.json")
    gt_nodes = pd.read_csv(root / "outputs/labeled/gt_nodes.csv")
    gt_edges = pd.read_csv(root / "outputs/labeled/gt_edges.csv")

    source_store = (
        root
        / "data/pilot_snapshot/train"
        / f"{cfg['sample_id']}.zarr"
    )
    root_meta_path = source_store / "zarr.json"
    array_meta_path = source_store / "0/zarr.json"
    if not root_meta_path.is_file() or not array_meta_path.is_file():
        raise FileNotFoundError(
            "Reviewed pilot Zarr metadata is required."
        )

    array_meta = read_json(array_meta_path)
    shape = array_meta["shape"]
    if shape != [100, 64, 256, 256]:
        raise ValueError(f"Unexpected reviewed image shape: {shape}")

    candidates = enumerate_windows(
        gt_nodes,
        gt_edges,
        shape_tzyx=shape,
        roi_shape=cfg["roi_shape_zyx"],
        spacing_um=cfg["spacing_um"],
        frame_count=cfg["frame_count"],
        descriptor_margin_um=cfg["descriptor_margin_um"],
        pair_radius_um=float(
            pilot_cfg.get("pair_radius_um", cfg["pair_radius_um"])
        ),
    )
    selected, selection_summary = select_diverse_windows(
        candidates,
        max_windows=cfg["max_windows"],
        target_unique_positive_links=cfg["target_unique_positive_links"],
        target_unique_negative_links=cfg["target_unique_negative_links"],
    )

    all_keys = set()
    total_decoded = 0
    for row in selected:
        keys = chunk_keys_for_roi(
            array_meta,
            row["frames"],
            row["roi_offsets_zyx"],
            cfg["roi_shape_zyx"],
        )
        row["chunk_keys"] = keys
        row["chunk_file_count"] = len(keys)
        all_keys.update(keys)
        decoded = (
            len(row["frames"])
            * int(np.prod(cfg["roi_shape_zyx"]))
            * np.dtype(array_meta["data_type"]).itemsize
        )
        row["selected_decoded_bytes"] = int(decoded)
        total_decoded += int(decoded)

    if len(all_keys) > cfg["max_chunk_files"]:
        raise RuntimeError(
            f"Planned {len(all_keys)} unique chunks, above cap "
            f"{cfg['max_chunk_files']}."
        )
    if total_decoded > cfg["max_selected_decoded_bytes"]:
        raise RuntimeError(
            f"Planned decoded ROI bytes {total_decoded} exceed cap "
            f"{cfg['max_selected_decoded_bytes']}."
        )

    out = root / "outputs/multiwindow_eval"
    out.mkdir(parents=True, exist_ok=True)

    ranking = pd.DataFrame([
        {
            "rank": i + 1,
            "start_frame": row["start_frame"],
            "frames": "-".join(str(v) for v in row["frames"]),
            "roi_z": row["roi_offsets_zyx"][0],
            "roi_y": row["roi_offsets_zyx"][1],
            "roi_x": row["roi_offsets_zyx"][2],
            "safe_gt_nodes": row["safe_gt_nodes"],
            "safe_gt_edges": row["safe_gt_edges"],
            "potential_positive_links": row["potential_positive_links"],
            "potential_negative_links": row["potential_negative_links"],
            "physical_span_norm_um": row["physical_span_norm_um"],
        }
        for i, row in enumerate(candidates[:100])
    ])
    atomic(out / "window_ranking.csv", ranking.to_csv(index=False))

    selected_table = pd.DataFrame([
        {
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
        for row in selected
    ])
    atomic(out / "selected_windows.csv", selected_table.to_csv(index=False))

    plan = {
        "status": "planned",
        "sample_id": cfg["sample_id"],
        "windows": selected,
        "selection_summary": selection_summary,
        "unique_chunk_keys": sorted(all_keys),
        "unique_chunk_file_count": len(all_keys),
        "total_selected_decoded_bytes": int(total_decoded),
        "training_fits": 0,
        "official_score": None,
        "evaluation": cfg["evaluation_scope"],
    }
    write_json(out / "plan.json", plan)

    snapshot = (
        root
        / "data/multiwindow_snapshot/train"
        / f"{cfg['sample_id']}.zarr"
    )
    snapshot.mkdir(parents=True, exist_ok=True)
    for src, dst in (
        (root_meta_path, snapshot / "zarr.json"),
        (array_meta_path, snapshot / "0/zarr.json"),
    ):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if sha256_file(dst) != sha256_file(src):
                raise ValueError(f"Metadata copy differs: {dst}")
        else:
            shutil.copy2(src, dst)

    write_json(
        out / "prepare_receipt.json",
        {
            "status": "completed",
            "selected_windows": len(selected),
            "unique_chunk_file_count": len(all_keys),
            "total_selected_decoded_bytes": int(total_decoded),
            "planned_unique_positive_links": selection_summary[
                "planned_unique_positive_links"
            ],
            "planned_unique_negative_links": selection_summary[
                "planned_unique_negative_links"
            ],
            "training_fits": 0,
            "network_requests": 0,
        },
    )
    return plan


def acquire_multiwindow_rois(root: Path) -> dict:
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r"crc32c usage is deprecated since numcodecs v0\.16\.4.*",
        category=DeprecationWarning,
        module=r"numcodecs(\..*)?",
    )
    import zarr

    root = Path(root).resolve()
    cfg = read_json(root / "configs/multiwindow_eval.json")
    plan = read_json(root / "outputs/multiwindow_eval/plan.json")

    destination = root / "data/multiwindow_snapshot"
    prefix = f"train/{cfg['sample_id']}.zarr/"
    downloaded = []
    reused = []

    for key in plan["unique_chunk_keys"]:
        target = destination / (prefix + key)
        existed = target.is_file()
        path = _download_one(
            competition=cfg["competition"],
            remote=prefix + key,
            destination=destination,
            timeout_seconds=60,
        )
        if path.stat().st_size > cfg["max_download_file_bytes"]:
            raise RuntimeError(f"Chunk exceeded file cap: {key}")
        (reused if existed else downloaded).append(path)

    total = sum(
        p.stat().st_size
        for p in destination.rglob("*")
        if p.is_file()
    )
    if total > cfg["max_snapshot_bytes"]:
        raise RuntimeError("Multi-window snapshot exceeds cache cap.")

    store = (
        destination
        / "train"
        / f"{cfg['sample_id']}.zarr"
    )
    arr = zarr.open_array(store / "0", mode="r")
    windows_dir = destination / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    quality_rows = []
    window_artifacts = []
    for row in plan["windows"]:
        offsets = np.asarray(row["roi_offsets_zyx"], dtype=int)
        rz, ry, rx = [int(v) for v in cfg["roi_shape_zyx"]]
        slices = (
            slice(int(offsets[0]), int(offsets[0] + rz)),
            slice(int(offsets[1]), int(offsets[1] + ry)),
            slice(int(offsets[2]), int(offsets[2] + rx)),
        )

        rois = []
        for t in row["frames"]:
            roi = np.asarray(
                arr[int(t), slices[0], slices[1], slices[2]]
            )
            if list(roi.shape) != cfg["roi_shape_zyx"]:
                raise ValueError(
                    f"Unexpected ROI shape for {row['window_id']} frame {t}"
                )
            if not np.isfinite(roi).all() or roi.max() <= roi.min():
                raise ValueError(
                    f"Invalid/flat ROI for {row['window_id']} frame {t}"
                )
            rois.append(np.array(roi, copy=True))
            q = np.quantile(roi, [0.01, 0.5, 0.998])
            quality_rows.append({
                "window_id": row["window_id"],
                "t": int(t),
                "roi_min": float(roi.min()),
                "roi_max": float(roi.max()),
                "roi_q01": float(q[0]),
                "roi_median": float(q[1]),
                "roi_q998": float(q[2]),
                "roi_mean": float(roi.mean()),
                "roi_std": float(roi.std()),
            })

        target = windows_dir / f"{row['window_id']}.npz"
        tmp = target.with_name(target.name + ".partial")
        with tmp.open("wb") as f:
            np.savez_compressed(
                f,
                images=np.stack(rois),
                frames=np.asarray(row["frames"], dtype=int),
                offsets=offsets,
                spacing_um=np.asarray(cfg["spacing_um"], dtype=float),
            )
        os.replace(tmp, target)
        window_artifacts.append({
            "window_id": row["window_id"],
            "path": str(target.relative_to(root)),
            "sha256": sha256_file(target),
        })

    out = root / "outputs/multiwindow_eval"
    atomic(
        out / "image_quality.csv",
        pd.DataFrame(quality_rows).to_csv(index=False),
    )
    receipt = {
        "status": "completed",
        "downloaded_chunks": len(downloaded),
        "reused_chunks": len(reused),
        "unique_chunk_count": len(plan["unique_chunk_keys"]),
        "snapshot_bytes": int(total),
        "windows": window_artifacts,
        "training_fits": 0,
    }
    write_json(out / "acquire_receipt.json", receipt)
    return receipt


def _evaluate_one_window(
    root,
    window,
    cache_path,
    gt_nodes,
    gt_edges,
):
    with np.load(cache_path, allow_pickle=False) as z:
        images = z["images"]
        frames = z["frames"]
        offsets = z["offsets"]
        spacing = z["spacing_um"]

    nodes, pairs, detector, motion = _pilot_features_for_annotation(
        root, images, frames, offsets, spacing
    )
    r1, r2, divisions = _round_tables(
        root, images, frames, offsets, spacing, nodes, pairs
    )

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

    gt_ids = set(gt_roi["gt_id"].astype(int))
    gt_edges_roi = gt_edges[
        gt_edges["source_gt_id"].astype(int).isin(gt_ids)
        & gt_edges["target_gt_id"].astype(int).isin(gt_ids)
    ]
    represented = set(
        zip(
            labeled.loc[
                labeled["edge_label"] == 1, "source_gt_id"
            ].astype(int),
            labeled.loc[
                labeled["edge_label"] == 1, "target_gt_id"
            ].astype(int),
            strict=True,
        )
    )
    edge_recall = (
        float(len(represented) / len(gt_edges_roi))
        if len(gt_edges_roi)
        else None
    )

    pos = int((labeled["edge_label"] == 1).sum())
    neg = int((labeled["edge_label"] == 0).sum())
    ign = int(labeled["edge_label"].isna().sum())

    pooled = _pool_round1(r1, labeled)
    pooled["window_id"] = window["window_id"]

    summary = {
        "window_id": window["window_id"],
        "frames": "-".join(str(v) for v in window["frames"]),
        "annotated_gt_nodes": node_summary["annotated_gt_nodes_in_roi"],
        "matched_gt_nodes": node_summary["matched_gt_nodes"],
        "annotated_node_recall": node_summary["annotated_node_recall"],
        "gt_edges_in_roi": int(len(gt_edges_roi)),
        "represented_gt_edges": int(len(represented)),
        "candidate_edge_recall": edge_recall,
        "positive_pairs": pos,
        "negative_pairs": neg,
        "ignored_pairs": ign,
        "candidate_nodes": int(len(nodes)),
        "candidate_pairs": int(len(labeled)),
    }

    return {
        "summary": summary,
        "nodes": nodes,
        "r1": r1,
        "r2": r2,
        "matches": matches,
        "gt_roi": gt_roi,
        "gt_edges_roi": gt_edges_roi,
        "labeled": labeled,
        "pooled": pooled,
        "detector": detector,
        "motion": motion,
        "divisions": divisions,
        "represented": represented,
    }


def evaluate_multiwindow(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/multiwindow_eval.json")
    plan = read_json(root / "outputs/multiwindow_eval/plan.json")
    acquire = read_json(
        root / "outputs/multiwindow_eval/acquire_receipt.json"
    )
    gt_nodes = pd.read_csv(root / "outputs/labeled/gt_nodes.csv")
    gt_edges = pd.read_csv(root / "outputs/labeled/gt_edges.csv")

    out = root / "outputs/multiwindow_eval"
    windows_out = out / "windows"
    windows_out.mkdir(parents=True, exist_ok=True)

    summaries = []
    pooled_parts = []
    labeled_parts = []

    union_gt_nodes = set()
    union_matched_gt = set()
    union_gt_edges = set()
    union_represented_edges = set()

    artifact_map = {
        x["window_id"]: root / x["path"]
        for x in acquire["windows"]
    }

    for window in plan["windows"]:
        window_id = window["window_id"]
        result = _evaluate_one_window(
            root,
            window,
            artifact_map[window_id],
            gt_nodes,
            gt_edges,
        )
        summaries.append(result["summary"])

        labeled = result["labeled"].copy()
        labeled["window_id"] = window_id
        labeled_parts.append(labeled)

        pooled_parts.append(result["pooled"])

        union_gt_nodes.update(
            int(v) for v in result["gt_roi"]["gt_id"]
        )
        union_matched_gt.update(
            result["matches"]
            .loc[result["matches"]["matched"], "gt_id"]
            .dropna()
            .astype(int)
            .tolist()
        )
        union_gt_edges.update(
            zip(
                result["gt_edges_roi"]["source_gt_id"].astype(int),
                result["gt_edges_roi"]["target_gt_id"].astype(int),
                strict=True,
            )
        )
        union_represented_edges.update(result["represented"])

        wdir = windows_out / window_id
        wdir.mkdir(parents=True, exist_ok=True)
        for name, table in (
            ("node_matches.csv", result["matches"]),
            ("gt_nodes_in_roi.csv", result["gt_roi"]),
            ("labeled_pairs.csv", labeled),
            ("detector_diagnostics.csv", result["detector"]),
            ("motion_diagnostics.csv", result["motion"]),
        ):
            atomic(wdir / name, table.to_csv(index=False))

    summary_df = pd.DataFrame(summaries)
    atomic(out / "window_summary.csv", summary_df.to_csv(index=False))

    all_labeled = pd.concat(labeled_parts, ignore_index=True)
    atomic(out / "all_candidate_pairs.csv", all_labeled.to_csv(index=False))

    all_pooled = pd.concat(pooled_parts, ignore_index=True)

    labeled_only = all_pooled[
        all_pooled["edge_label"].isin([0.0, 1.0])
    ].copy()
    before_dedup = int(len(labeled_only))
    labeled_only = labeled_only.drop_duplicates(
        subset=["source_gt_id", "target_gt_id", "edge_label"],
        keep="first",
    ).reset_index(drop=True)

    pos = int((labeled_only["edge_label"] == 1).sum())
    neg = int((labeled_only["edge_label"] == 0).sum())
    ignored = int(all_labeled["edge_label"].isna().sum())

    screen = pd.DataFrame(columns=FEATURE_SCREEN_COLUMNS)
    family = pd.DataFrame(columns=FAMILY_SCREEN_COLUMNS)

    if (
        pos >= cfg["min_positive_links_for_screen"]
        and neg >= cfg["min_negative_links_for_screen"]
    ):
        screen = screen_features(
            labeled_only,
            min_per_class=min(
                cfg["min_positive_links_for_screen"],
                cfg["min_negative_links_for_screen"],
            ),
        )
        if len(screen):
            screen["family"] = screen["feature"].map(family_from_feature)
            screen = screen.reindex(columns=FEATURE_SCREEN_COLUMNS)
            family = (
                screen.groupby("family", as_index=False)
                .agg(
                    features_screened=("feature", "size"),
                    median_signal=("signal_strength", "median"),
                    max_signal=("signal_strength", "max"),
                )
                .sort_values(
                    ["median_signal", "max_signal"],
                    ascending=False,
                )
                .reindex(columns=FAMILY_SCREEN_COLUMNS)
            )

    atomic(out / "feature_screen.csv", screen.to_csv(index=False))
    atomic(out / "family_screen.csv", family.to_csv(index=False))

    node_recall = (
        float(len(union_matched_gt) / len(union_gt_nodes))
        if union_gt_nodes
        else None
    )
    edge_recall = (
        float(len(union_represented_edges) / len(union_gt_edges))
        if union_gt_edges
        else None
    )

    if node_recall is None or node_recall < 0.8:
        next_decision = "detector_research"
    elif edge_recall is None or edge_recall < 0.8:
        next_decision = "candidate_graph_research"
    elif pos >= 3 and neg >= 3:
        next_decision = "association_modeling"
    else:
        next_decision = "expand_to_additional_labeled_samples"

    receipt = {
        "stage": "multiwindow_sparse_supervision",
        "status": "completed",
        "sample_id": cfg["sample_id"],
        "selected_windows": len(plan["windows"]),
        "unique_gt_nodes_evaluated": len(union_gt_nodes),
        "unique_gt_nodes_matched": len(union_matched_gt),
        "aggregate_annotated_node_recall": safe_json_number(
            node_recall
        ) if node_recall is not None else None,
        "unique_gt_edges_evaluated": len(union_gt_edges),
        "unique_gt_edges_represented": len(union_represented_edges),
        "aggregate_candidate_edge_recall": safe_json_number(
            edge_recall
        ) if edge_recall is not None else None,
        "candidate_pairs_total": int(len(all_labeled)),
        "ignored_candidate_pairs": ignored,
        "labeled_rows_before_gt_pair_dedup": before_dedup,
        "unique_positive_links": pos,
        "unique_negative_links": neg,
        "round1_descriptors": 128,
        "round2_descriptors": 128,
        "features_screened": int(len(screen)),
        "feature_families_screened": int(len(family)),
        "training_fits": 0,
        "official_score": None,
        "leaderboard_score": None,
        "evaluation": cfg["evaluation_scope"],
        "next_decision": next_decision,
    }
    write_json(out / "evaluation_receipt.json", receipt)
    return receipt
