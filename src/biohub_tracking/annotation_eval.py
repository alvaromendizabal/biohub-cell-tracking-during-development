from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .feature_io import atomic, write_json
from .geff_snapshot import _download_one
from .pilot_features import (
    candidate_features,
    normalize_volume,
    pair_features,
    phase_shift,
    physical_log,
    select_candidates,
)
from .feature_round1 import frame_features
from .feature_round2 import make_context, pair_frame_features
from .feature_audit import audit
from .labeled_eval import screen_features, family_from_feature


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

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_json_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _roi_offsets(center_zyx, shape_zyx, roi_shape):
    offsets = []
    for center, n, r in zip(center_zyx, shape_zyx, roi_shape, strict=True):
        lo = int(round(float(center) - r / 2))
        lo = max(0, min(lo, int(n) - int(r)))
        offsets.append(lo)
    return np.asarray(offsets, dtype=int)


def _safe_mask(nodes, frames, offsets, roi_shape, margin_vox):
    subset = nodes[nodes["t"].isin(frames)].copy()
    if subset.empty:
        return subset
    lo = offsets + margin_vox
    hi = offsets + np.asarray(roi_shape) - 1 - margin_vox
    mask = (
        subset["z"].between(lo[0], hi[0])
        & subset["y"].between(lo[1], hi[1])
        & subset["x"].between(lo[2], hi[2])
    )
    return subset[mask].copy()


def _edge_stats(covered_nodes, edges):
    ids = set(covered_nodes["gt_id"].astype(int))
    internal = edges[
        edges["source_gt_id"].astype(int).isin(ids)
        & edges["target_gt_id"].astype(int).isin(ids)
    ].copy()
    outdegree = internal.groupby("source_gt_id").size()
    division_sources = sorted(int(i) for i, n in outdegree.items() if int(n) >= 2)
    return internal, division_sources


def choose_annotation_plan(
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    *,
    shape_tzyx,
    roi_shape,
    spacing_um,
    frame_count=4,
    descriptor_margin_um=6.0,
):
    shape_tzyx = tuple(int(v) for v in shape_tzyx)
    roi_shape = np.asarray(roi_shape, dtype=int)
    spacing = np.asarray(spacing_um, dtype=float)
    spatial_shape = np.asarray(shape_tzyx[1:], dtype=int)
    margin = np.ceil(float(descriptor_margin_um) / spacing).astype(int)

    if np.any(roi_shape + 2 * margin > spatial_shape + 2 * margin):
        raise ValueError("ROI shape exceeds image spatial shape.")
    if frame_count < 2 or frame_count > shape_tzyx[0]:
        raise ValueError("Invalid annotation frame count.")
    if gt_nodes.empty or gt_edges.empty:
        raise ValueError("Sparse GT nodes/edges are required.")

    candidates = []
    max_start = shape_tzyx[0] - frame_count
    for start in range(max_start + 1):
        frames = list(range(start, start + frame_count))
        window_nodes = gt_nodes[gt_nodes["t"].isin(frames)]
        if window_nodes.empty:
            continue

        # Candidate centers are actual annotated nodes plus the median annotation center.
        centers = window_nodes[["z", "y", "x"]].to_numpy(float).tolist()
        centers.append(window_nodes[["z", "y", "x"]].median().to_numpy(float).tolist())

        seen_offsets = set()
        for center in centers:
            offsets = _roi_offsets(center, spatial_shape, roi_shape)
            key = tuple(int(v) for v in offsets)
            if key in seen_offsets:
                continue
            seen_offsets.add(key)

            safe_nodes = _safe_mask(
                gt_nodes, frames, offsets, roi_shape, margin
            )
            if safe_nodes.empty:
                continue

            internal_edges, division_sources = _edge_stats(safe_nodes, gt_edges)
            span = safe_nodes[["z", "y", "x"]].max() - safe_nodes[["z", "y", "x"]].min()
            physical_span = span.to_numpy(float) * spacing
            compactness = float(np.linalg.norm(physical_span))

            score = (
                int(len(internal_edges)),
                int(len(division_sources)),
                int(len(safe_nodes)),
                -compactness,
                -int(start),
            )
            candidates.append({
                "start_frame": int(start),
                "frames": frames,
                "roi_offsets_zyx": [int(v) for v in offsets],
                "safe_gt_nodes": int(len(safe_nodes)),
                "safe_gt_edges": int(len(internal_edges)),
                "safe_division_sources": int(len(division_sources)),
                "safe_division_ids": division_sources,
                "safe_gt_ids": sorted(int(v) for v in safe_nodes["gt_id"]),
                "physical_span_norm_um": compactness,
                "_score": score,
            })

    if not candidates:
        raise ValueError("No annotation-centered ROI candidate contains safe GT nodes.")

    candidates.sort(key=lambda r: r["_score"], reverse=True)
    best = dict(candidates[0])
    best.pop("_score", None)
    ranking = []
    for rank, row in enumerate(candidates[:100], start=1):
        x = dict(row)
        x.pop("_score", None)
        x["rank"] = rank
        ranking.append(x)
    return best, pd.DataFrame(ranking)


def chunk_keys_for_roi(array_meta, frames, offsets, roi_shape):
    shape = np.asarray(array_meta["shape"], dtype=int)
    chunks = np.asarray(
        array_meta["chunk_grid"]["configuration"]["chunk_shape"], dtype=int
    )
    if shape.shape != (4,) or chunks.shape != (4,):
        raise ValueError("Expected four-dimensional regular Zarr array.")
    encoding = array_meta.get("chunk_key_encoding", {})
    name = encoding.get("name")
    sep = encoding.get("configuration", {}).get(
        "separator", "/" if name == "default" else "."
    )
    if name not in ("default", "v2"):
        raise ValueError(f"Unsupported chunk key encoding: {name!r}")

    z0, y0, x0 = [int(v) for v in offsets]
    rz, ry, rx = [int(v) for v in roi_shape]
    starts = np.asarray([z0, y0, x0], dtype=int)
    ends = starts + np.asarray([rz, ry, rx], dtype=int) - 1

    spatial_ranges = [
        range(int(starts[i] // chunks[i + 1]), int(ends[i] // chunks[i + 1]) + 1)
        for i in range(3)
    ]
    keys = []
    for t in frames:
        ti = int(t // chunks[0])
        for zi in spatial_ranges[0]:
            for yi in spatial_ranges[1]:
                for xi in spatial_ranges[2]:
                    idx = [ti, int(zi), int(yi), int(xi)]
                    encoded = sep.join(str(v) for v in idx)
                    if name == "default":
                        encoded = "c" + sep + encoded
                    keys.append("0/" + encoded)
    return sorted(set(keys))


def prepare_plan(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/annotation_eval.json")
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
            "Existing pilot Zarr metadata is required; do not redownload blindly."
        )

    root_meta = read_json(root_meta_path)
    array_meta = read_json(array_meta_path)
    shape = array_meta.get("shape")
    if shape != [100, 64, 256, 256]:
        raise ValueError(f"Unexpected reviewed image shape: {shape}")

    best, ranking = choose_annotation_plan(
        gt_nodes,
        gt_edges,
        shape_tzyx=shape,
        roi_shape=cfg["roi_shape_zyx"],
        spacing_um=cfg["spacing_um"],
        frame_count=cfg["frame_count"],
        descriptor_margin_um=cfg["descriptor_margin_um"],
    )
    keys = chunk_keys_for_roi(
        array_meta,
        best["frames"],
        best["roi_offsets_zyx"],
        cfg["roi_shape_zyx"],
    )
    if len(keys) > cfg["max_chunk_files"]:
        raise RuntimeError(
            f"Planned {len(keys)} chunks, above cap {cfg['max_chunk_files']}."
        )

    decoded = (
        len(best["frames"])
        * int(np.prod(cfg["roi_shape_zyx"]))
        * np.dtype(array_meta["data_type"]).itemsize
    )
    if decoded > cfg["max_selected_decoded_bytes"]:
        raise RuntimeError("Selected annotation ROI exceeds decoded-byte cap.")

    best.update({
        "status": "planned",
        "sample_id": cfg["sample_id"],
        "roi_shape_zyx": cfg["roi_shape_zyx"],
        "spacing_um": cfg["spacing_um"],
        "descriptor_margin_um": cfg["descriptor_margin_um"],
        "chunk_keys": keys,
        "chunk_file_count": len(keys),
        "selected_decoded_bytes": int(decoded),
        "image_shape_tzyx": shape,
        "array_dtype": array_meta["data_type"],
        "training_fits": 0,
        "official_score": None,
        "evaluation": cfg["evaluation_scope"],
    })

    out = root / "outputs/annotation_eval"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "plan.json", best)
    atomic(out / "window_ranking.csv", ranking.to_csv(index=False))

    snapshot = (
        root
        / "data/annotation_snapshot/train"
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
                raise ValueError(f"Annotation metadata copy differs: {dst}")
        else:
            shutil.copy2(src, dst)

    write_json(
        out / "prepare_receipt.json",
        {
            "status": "completed",
            "plan_sha256": sha256_file(out / "plan.json"),
            "window_ranking_sha256": sha256_file(out / "window_ranking.csv"),
            "source_metadata": {
                str(root_meta_path.relative_to(root)): sha256_file(root_meta_path),
                str(array_meta_path.relative_to(root)): sha256_file(array_meta_path),
            },
            "training_fits": 0,
            "network_requests": 0,
        },
    )
    return best


def acquire_roi(root: Path) -> dict:
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r"crc32c usage is deprecated since numcodecs v0\.16\.4.*",
        category=DeprecationWarning,
        module=r"numcodecs(\..*)?",
    )
    import zarr

    root = Path(root).resolve()
    cfg = read_json(root / "configs/annotation_eval.json")
    plan = read_json(root / "outputs/annotation_eval/plan.json")
    if plan.get("status") != "planned":
        raise ValueError("Annotation plan is not ready.")

    destination = root / "data/annotation_snapshot"
    prefix = f"train/{cfg['sample_id']}.zarr/"
    downloaded = []
    reused = []
    for key in plan["chunk_keys"]:
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
        raise RuntimeError("Annotation snapshot exceeds cache cap.")

    store = (
        destination
        / "train"
        / f"{cfg['sample_id']}.zarr"
    )
    for key in plan["chunk_keys"]:
        p = store / key
        if not p.is_file():
            raise FileNotFoundError(f"Missing planned annotation chunk: {key}")

    arr = zarr.open_array(store / "0", mode="r")
    offsets = np.asarray(plan["roi_offsets_zyx"], dtype=int)
    rz, ry, rx = [int(v) for v in plan["roi_shape_zyx"]]
    slices = (
        slice(int(offsets[0]), int(offsets[0] + rz)),
        slice(int(offsets[1]), int(offsets[1] + ry)),
        slice(int(offsets[2]), int(offsets[2] + rx)),
    )

    rois = []
    qc = []
    for t in plan["frames"]:
        roi = np.asarray(arr[int(t), slices[0], slices[1], slices[2]])
        if list(roi.shape) != plan["roi_shape_zyx"]:
            raise ValueError(f"Unexpected annotation ROI shape for frame {t}.")
        if not np.isfinite(roi).all() or roi.max() <= roi.min():
            raise ValueError(f"Invalid/flat annotation ROI for frame {t}.")
        rois.append(np.array(roi, copy=True))
        q = np.quantile(roi, [0.01, 0.5, 0.998])
        qc.append({
            "t": int(t),
            "roi_min": float(roi.min()),
            "roi_max": float(roi.max()),
            "roi_q01": float(q[0]),
            "roi_median": float(q[1]),
            "roi_q998": float(q[2]),
            "roi_mean": float(roi.mean()),
            "roi_std": float(roi.std()),
        })

    target = destination / "selected_roi.npz"
    tmp = target.with_name(target.name + ".partial")
    with tmp.open("wb") as f:
        np.savez_compressed(
            f,
            images=np.stack(rois),
            frames=np.asarray(plan["frames"], dtype=int),
            offsets=offsets,
            spacing_um=np.asarray(plan["spacing_um"], dtype=float),
        )
    os.replace(tmp, target)

    out = root / "outputs/annotation_eval"
    atomic(out / "image_quality.csv", pd.DataFrame(qc).to_csv(index=False))
    receipt = {
        "status": "completed",
        "selected_roi": str(target.relative_to(root)),
        "selected_roi_sha256": sha256_file(target),
        "downloaded_chunks": len(downloaded),
        "reused_chunks": len(reused),
        "chunk_count": len(plan["chunk_keys"]),
        "snapshot_bytes": int(total),
        "frames": plan["frames"],
        "roi_offsets_zyx": plan["roi_offsets_zyx"],
        "roi_shape_zyx": plan["roi_shape_zyx"],
        "training_fits": 0,
    }
    write_json(out / "acquire_receipt.json", receipt)
    return receipt


def _pilot_features_for_annotation(root, images, frames, offsets, spacing):
    pilot_cfg = read_json(root / "configs/pilot.json")
    cfg = dict(pilot_cfg)
    cfg["frames"] = [int(v) for v in frames]
    cfg["sigma_um"] = list(read_json(root / "configs/annotation_eval.json")["detector_sigmas_um"])

    all_nodes = []
    detector_rows = []
    for i, t in enumerate(frames):
        normal, norm_stats = normalize_volume(images[i])
        responses = [physical_log(normal, spacing, s) for s in cfg["sigma_um"]]
        multi = np.maximum.reduce(responses)
        coords, diag = select_candidates(multi, spacing, cfg)
        nodes = candidate_features(
            images[i],
            coords,
            responses,
            spacing,
            cfg["sigma_um"],
            int(t),
            offsets,
            cfg,
        )
        if nodes.empty:
            raise ValueError(
                f"No detector candidates in annotation frame {int(t)}; "
                "do not lower thresholds before reviewing this ROI."
            )
        all_nodes.append(nodes)
        detector_rows.append({
            "t": int(t),
            "candidate_count": int(len(nodes)),
            "candidate_threshold": float(diag["threshold"]),
            "raw_peak_count": int(diag["raw_peak_count"]),
            "candidate_cap_reached": bool(diag["candidate_cap_reached"]),
            **{f"norm_{k}": v for k, v in norm_stats.items()},
        })

    pair_tables = []
    motion_rows = []
    for i in range(len(frames) - 1):
        shift, phase_diag = phase_shift(images[i], images[i + 1])
        drift = shift * spacing
        pair = pair_features(all_nodes[i], all_nodes[i + 1], drift, cfg)
        pair_tables.append(pair)
        motion_rows.append({
            "t_source": int(frames[i]),
            "t_target": int(frames[i + 1]),
            "drift_z_um": float(drift[0]),
            "drift_y_um": float(drift[1]),
            "drift_x_um": float(drift[2]),
            "drift_norm_um": float(np.linalg.norm(drift)),
            **phase_diag,
        })

    nodes = pd.concat(all_nodes, ignore_index=True)
    pairs = pd.concat(pair_tables, ignore_index=True)
    if pairs.empty:
        raise ValueError("Annotation ROI produced no adjacent candidate pairs.")
    return nodes, pairs, pd.DataFrame(detector_rows), pd.DataFrame(motion_rows)


def _round_tables(root, images, frames, offsets, spacing, nodes, pairs):
    cfg = read_json(root / "configs/feature_rounds.json")
    r1_parts = []
    for i, t in enumerate(frames):
        selected = nodes[nodes["t"] == int(t)].copy()
        r1_parts.append(
            frame_features(
                images[i],
                selected,
                spacing,
                offsets,
                cfg["round1"]["families"],
                emit=lambda **_: None,
            )
        )
    r1 = pd.concat(r1_parts, ignore_index=True)
    r1_cols = [c for c in r1.columns if c.startswith("r1_")]
    if len(r1_cols) != 128:
        raise ValueError(f"Expected 128 Round-1 descriptors, found {len(r1_cols)}.")

    context = make_context(images, frames, offsets, spacing, nodes, r1)
    frame_map = {int(t): nodes[nodes["t"] == int(t)].copy() for t in frames}
    r2_parts = []
    divisions = []
    for t in frames[:-1]:
        selected = pairs[pairs["t_source"] == int(t)].copy()
        table, div = pair_frame_features(
            selected,
            pairs,
            context,
            frame_map,
            cfg["round2"]["families"],
            cfg["round2"]["max_siblings_per_edge"],
            emit=lambda **_: None,
        )
        r2_parts.append(table)
        divisions.append(div)
    r2 = pd.concat(r2_parts, ignore_index=True)
    r2_cols = [c for c in r2.columns if c.startswith("r2_")]
    if len(r2_cols) != 128:
        raise ValueError(f"Expected 128 Round-2 descriptors, found {len(r2_cols)}.")

    # Avoid pandas empty/all-NA concat dtype inference deprecation.
    division_parts_clean = []
    for div in divisions:
        if div is None or div.empty:
            continue
        cleaned = div.dropna(axis=1, how="all").copy()
        if cleaned.empty or len(cleaned.columns) == 0:
            continue
        division_parts_clean.append(cleaned)

    division = (
        pd.concat(division_parts_clean, ignore_index=True)
        if division_parts_clean
        else pd.DataFrame()
    )
    return r1, r2, division


def match_annotation_nodes(nodes, gt_nodes, plan):
    spacing = np.asarray(plan["spacing_um"], dtype=float)
    offsets = np.asarray(plan["roi_offsets_zyx"], dtype=float)
    roi_shape = np.asarray(plan["roi_shape_zyx"], dtype=float)
    frames = [int(v) for v in plan["frames"]]

    gt = gt_nodes[
        gt_nodes["t"].isin(frames)
        & gt_nodes["z"].between(offsets[0], offsets[0] + roi_shape[0] - 1)
        & gt_nodes["y"].between(offsets[1], offsets[1] + roi_shape[1] - 1)
        & gt_nodes["x"].between(offsets[2], offsets[2] + roi_shape[2] - 1)
    ].copy()

    rows = []
    for t in frames:
        c = nodes[nodes["t"] == t].reset_index(drop=True)
        g = gt[gt["t"] == t].reset_index(drop=True)
        frame = {
            int(r.node_id): {
                "candidate_id": int(r.node_id),
                "t": int(t),
                "gt_id": np.nan,
                "distance_um": np.nan,
                "matched": False,
            }
            for r in c.itertuples()
        }
        if len(c) and len(g):
            cp = c[["z", "y", "x"]].to_numpy(float)
            gp = g[["z", "y", "x"]].to_numpy(float)
            dist = np.linalg.norm(
                (cp[:, None, :] - gp[None, :, :]) * spacing[None, None, :],
                axis=2,
            )
            ri, ci = linear_sum_assignment(dist)
            for a, b in zip(ri, ci, strict=True):
                d = float(dist[a, b])
                if d <= 7.0:
                    cid = int(c.iloc[a]["node_id"])
                    frame[cid]["gt_id"] = int(g.iloc[b]["gt_id"])
                    frame[cid]["distance_um"] = d
                    frame[cid]["matched"] = True
        rows.extend(frame.values())

    result = pd.DataFrame(rows)
    matched_gt = set(result.loc[result["matched"], "gt_id"].dropna().astype(int))
    summary = {
        "candidate_nodes": int(len(result)),
        "annotated_gt_nodes_in_roi": int(len(gt)),
        "matched_candidates": int(result["matched"].sum()),
        "matched_gt_nodes": int(len(matched_gt)),
        "annotated_node_recall": (
            float(len(matched_gt) / len(gt)) if len(gt) else None
        ),
    }
    return result, gt, summary


def label_annotation_pairs(pairs, matches, gt_edges):
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

    labels, reasons, src_gt, tgt_gt = [], [], [], []
    for r in pairs.itertuples(index=False):
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

    out = pairs.copy()
    out.insert(4, "source_gt_id", src_gt)
    out.insert(5, "target_gt_id", tgt_gt)
    out.insert(6, "edge_label", labels)
    out.insert(7, "label_reason", reasons)
    return out


def _pool_round1(r1, labeled_pairs):
    features = [c for c in r1.columns if c.startswith("r1_")]
    src = r1[["node_id"] + features].rename(
        columns={"node_id": "source_id", **{c: f"src__{c}" for c in features}}
    )
    tgt = r1[["node_id"] + features].rename(
        columns={"node_id": "target_id", **{c: f"tgt__{c}" for c in features}}
    )
    merged = labeled_pairs.merge(src, on="source_id", how="left", validate="many_to_one")
    merged = merged.merge(tgt, on="target_id", how="left", validate="many_to_one")
    diff = pd.DataFrame(
        {
            f"absdiff__{c}": (
                merged[f"src__{c}"] - merged[f"tgt__{c}"]
            ).abs()
            for c in features
        },
        index=merged.index,
    )
    return pd.concat([merged, diff], axis=1).copy()


def run_annotation_evaluation(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/annotation_eval.json")
    plan = read_json(root / "outputs/annotation_eval/plan.json")
    acquire = read_json(root / "outputs/annotation_eval/acquire_receipt.json")
    cache = root / acquire["selected_roi"]

    with np.load(cache, allow_pickle=False) as z:
        images = z["images"]
        frames = z["frames"]
        offsets = z["offsets"]
        spacing = z["spacing_um"]

    if frames.tolist() != plan["frames"]:
        raise ValueError("Annotation cache frames differ from the reviewed plan.")
    if list(images.shape) != [len(frames), *plan["roi_shape_zyx"]]:
        raise ValueError("Annotation cache shape mismatch.")

    nodes, pairs, detector, motion = _pilot_features_for_annotation(
        root, images, frames, offsets, spacing
    )
    r1, r2, divisions = _round_tables(
        root, images, frames, offsets, spacing, nodes, pairs
    )

    gt_nodes = pd.read_csv(root / "outputs/labeled/gt_nodes.csv")
    gt_edges = pd.read_csv(root / "outputs/labeled/gt_edges.csv")
    matches, gt_roi, node_summary = match_annotation_nodes(nodes, gt_nodes, plan)
    labeled_pairs = label_annotation_pairs(r2, matches, gt_edges)

    gt_ids = set(gt_roi["gt_id"].astype(int))
    gt_edges_roi = gt_edges[
        gt_edges["source_gt_id"].astype(int).isin(gt_ids)
        & gt_edges["target_gt_id"].astype(int).isin(gt_ids)
    ]
    matched_map = {
        int(r.candidate_id): int(r.gt_id)
        for r in matches[matches["matched"]].itertuples()
    }
    represented = set()
    for r in labeled_pairs[labeled_pairs["edge_label"] == 1].itertuples():
        represented.add((int(r.source_gt_id), int(r.target_gt_id)))
    edge_recall = (
        float(len(represented) / len(gt_edges_roi))
        if len(gt_edges_roi)
        else None
    )

    pos = int((labeled_pairs["edge_label"] == 1).sum())
    neg = int((labeled_pairs["edge_label"] == 0).sum())
    ign = int(labeled_pairs["edge_label"].isna().sum())

    pooled = None
    screen = pd.DataFrame(columns=FEATURE_SCREEN_COLUMNS)
    family = pd.DataFrame(columns=FAMILY_SCREEN_COLUMNS)
    if (
        pos >= cfg["min_positive_links_for_screen"]
        and neg >= cfg["min_negative_links_for_screen"]
    ):
        pooled = _pool_round1(r1, labeled_pairs)
        screen = screen_features(
            pooled,
            min_per_class=min(
                cfg["min_positive_links_for_screen"],
                cfg["min_negative_links_for_screen"],
            ),
        )
        if len(screen):
            screen["family"] = screen["feature"].map(family_from_feature)
            family = (
                screen.groupby("family", as_index=False)
                .agg(
                    features_screened=("feature", "size"),
                    median_signal=("signal_strength", "median"),
                    max_signal=("signal_strength", "max"),
                )
                .sort_values(["median_signal", "max_signal"], ascending=False)
            )

    out = root / "outputs/annotation_eval"
    tables = {
        "pilot_node_features.csv": nodes,
        "pilot_pair_features.csv": pairs,
        "detector_diagnostics.csv": detector,
        "motion_diagnostics.csv": motion,
        "round1_features.csv": r1,
        "round2_features.csv": r2,
        "division_hypotheses.csv": divisions,
        "node_matches.csv": matches,
        "gt_nodes_in_roi.csv": gt_roi,
        "labeled_pairs.csv": labeled_pairs,
        "feature_screen.csv": screen,
        "family_screen.csv": family,
    }
    for name, table in tables.items():
        atomic(out / name, table.to_csv(index=False))

    r1q, r1red = audit(r1, "r1_")
    r2q, r2red = audit(r2, "r2_")
    atomic(out / "round1_quality.csv", r1q.to_csv(index=False))
    atomic(out / "round1_redundancy.csv", r1red.to_csv(index=False))
    atomic(out / "round2_quality.csv", r2q.to_csv(index=False))
    atomic(out / "round2_redundancy.csv", r2red.to_csv(index=False))

    if node_summary["annotated_node_recall"] is None:
        next_decision = "expand_annotation_window"
    elif node_summary["annotated_node_recall"] < 0.8:
        next_decision = "detector_research"
    elif edge_recall is not None and edge_recall < 0.8:
        next_decision = "candidate_graph_research"
    elif pos >= 3 and neg >= 3:
        next_decision = "association_modeling"
    else:
        next_decision = "expand_annotation_windows_for_supervision"

    receipt = {
        "stage": "annotation_centered_feature_evaluation",
        "status": "completed",
        "sample_id": cfg["sample_id"],
        "frames": [int(v) for v in frames],
        "roi_offsets_zyx": [int(v) for v in offsets],
        "roi_shape_zyx": plan["roi_shape_zyx"],
        "expected_safe_gt_nodes": plan["safe_gt_nodes"],
        "expected_safe_gt_edges": plan["safe_gt_edges"],
        "node_summary": node_summary,
        "gt_edges_in_roi": int(len(gt_edges_roi)),
        "candidate_gt_edges_represented": int(len(represented)),
        "candidate_edge_recall": safe_json_number(edge_recall) if edge_recall is not None else None,
        "pair_summary": {
            "candidate_pairs": int(len(labeled_pairs)),
            "positive_pairs": pos,
            "negative_pairs": neg,
            "ignored_pairs": ign,
        },
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
