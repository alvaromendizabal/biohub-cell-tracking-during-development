from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

from .annotation_eval import (
    _pool_round1,
    _round_tables,
    label_annotation_pairs,
    match_annotation_nodes,
)
from .feature_io import atomic, write_json
from .pilot_features import (
    candidate_features,
    normalize_volume,
    pair_features,
    phase_shift,
    physical_log,
)
from .public_frontier_assoc import (
    _binary_metrics,
    _choose_threshold,
    _clean_X,
    _decoded_labeled_metrics,
    _feature_columns,
    _fit_models,
    _local_metrics,
    _predict_hgb,
)
from .public_reproduction.models import (
    UNetNodeTransformerReproduction,
)
from .public_reproduction.utils import (
    detect_local_maxima,
    pool_kernel_from_um,
)
from .public_solution_features import (
    add_candidate_competition_context,
    add_kinematic_context,
    agreement_gated_fusion,
    dual_seed_logit_blend,
)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _find_quantiles(obj):
    if isinstance(obj, dict):
        q = obj.get("quantiles")
        if isinstance(q, dict) and "0.001" in q and "0.999" in q:
            return float(q["0.001"]), float(q["0.999"])
        for value in obj.values():
            found = _find_quantiles(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_quantiles(value)
            if found is not None:
                return found
    return None


def _feature_context_bounds(
    analysis_offsets,
    analysis_shape,
    image_shape,
    spacing,
    halo_um,
):
    analysis_offsets = np.asarray(analysis_offsets, dtype=int)
    analysis_shape = np.asarray(analysis_shape, dtype=int)
    image_shape = np.asarray(image_shape, dtype=int)
    spacing = np.asarray(spacing, dtype=float)

    halo_vox = np.ceil(float(halo_um) / spacing).astype(int)
    lo = np.maximum(analysis_offsets - halo_vox, 0)
    hi = np.minimum(
        analysis_offsets + analysis_shape + halo_vox,
        image_shape,
    )
    if np.any(hi <= lo):
        raise ValueError(
            f"Invalid feature-context bounds: lo={lo.tolist()} hi={hi.tolist()}"
        )
    return lo, hi, halo_vox


def _descriptor_valid_mask(
    local_coords,
    image_shape,
    spacing,
    patch_radius_um,
):
    coords = np.asarray(local_coords, dtype=int)
    shape = np.asarray(image_shape, dtype=int)
    spacing = np.asarray(spacing, dtype=float)
    if coords.size == 0:
        return np.zeros(0, dtype=bool)

    radii = np.ceil(float(patch_radius_um) / spacing).astype(int)
    return np.all(
        (coords - radii[None, :] >= 0)
        & (coords + radii[None, :] < shape[None, :]),
        axis=1,
    )


def _analysis_boundary_distance_um(
    global_coords,
    analysis_offsets,
    analysis_shape,
    spacing,
):
    coords = np.asarray(global_coords, dtype=int)
    offsets = np.asarray(analysis_offsets, dtype=int)
    shape = np.asarray(analysis_shape, dtype=int)
    spacing = np.asarray(spacing, dtype=float)

    local = coords - offsets[None, :]
    distance_vox = np.minimum(
        local,
        shape[None, :] - 1 - local,
    )
    return np.min(distance_vox * spacing[None, :], axis=1)


def _load_feature_context(
    root,
    *,
    sample_id,
    frames,
    analysis_offsets,
    analysis_shape,
    spacing,
    halo_um,
):
    zarr_root = (
        Path(root)
        / "data/multisample_scout/train"
        / f"{sample_id}.zarr"
    )
    arr = zarr.open_array(zarr_root / "0", mode="r")
    full_shape = np.asarray(arr.shape[1:], dtype=int)

    lo, hi, halo_vox = _feature_context_bounds(
        analysis_offsets,
        analysis_shape,
        full_shape,
        spacing,
        halo_um,
    )
    slices = tuple(
        slice(int(lo[i]), int(hi[i]))
        for i in range(3)
    )
    images = np.stack(
        [
            np.asarray(arr[int(t), slices[0], slices[1], slices[2]])
            for t in frames
        ]
    )

    return images, lo.astype(int), {
        "feature_context_offsets_zyx": [int(v) for v in lo],
        "feature_context_shape_zyx": [
            int(v) for v in images.shape[1:]
        ],
        "feature_context_halo_vox_zyx": [
            int(v) for v in halo_vox
        ],
        "feature_context_halo_um": float(halo_um),
        "full_image_shape_zyx": [int(v) for v in full_shape],
    }


def load_public_model(root: Path):
    cfg = read_json(root / "configs/public_detector_integration.json")
    model_cfg = read_json(root / cfg["model_config"])
    model = UNetNodeTransformerReproduction(
        unet_out_channels=int(model_cfg.get("unet_out_channels", 32)),
        unet_layers=tuple(model_cfg.get("unet_layers", [32, 64, 128])),
        pos_feat_dim=32,
    )
    state = torch.load(
        root / cfg["checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Public checkpoint mismatch: missing={missing[:20]} "
            f"unexpected={unexpected[:20]}"
        )
    model.eval()
    return model, model_cfg


def _encode_public_pair(model, frames_tensor: torch.Tensor):
    """Organizer-style 4-view XY detector TTA; features remain original-view."""
    imgs = frames_tensor.unsqueeze(0)
    with torch.no_grad():
        unet_out, det_logits = model.encode(imgs)
        for dims in [(-1,), (-2,), (-2, -1)]:
            flipped = imgs.flip(dims)
            _, det_flip = model.encode(flipped)
            for i in range(len(det_logits)):
                det_logits[i] = det_logits[i] + det_flip[i].flip(dims)
        det_logits = [x / 4.0 for x in det_logits]
    return unet_out, det_logits


def public_window_detections(
    root: Path,
    *,
    sample_id: str,
    window: dict,
    model,
    model_cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    cfg = read_json(root / "configs/public_detector_integration.json")
    det_cfg = cfg["detector"]
    graph_cfg = cfg["candidate_graph"]

    zarr_root = root / "data/multisample_scout/train" / f"{sample_id}.zarr"
    arr = zarr.open_array(zarr_root / "0", mode="r")
    quantiles = _find_quantiles(read_json(zarr_root / "zarr.json"))
    if quantiles is None:
        raise RuntimeError(f"Missing image quantiles for {sample_id}.")
    q_low, q_high = quantiles

    downsample = np.asarray(
        model_cfg.get("downsample", det_cfg["downsample_zyx"]),
        dtype=int,
    )
    spacing = np.asarray([1.625, 0.40625, 0.40625], dtype=float)
    ds_spacing = spacing * downsample
    pool_kernel = pool_kernel_from_um(
        float(det_cfg["pool_kernel_um"]),
        ds_spacing,
    )

    frames = [int(v) for v in window["frames"]]
    offsets = np.asarray(window["roi_offsets_zyx"], dtype=int)
    roi_shape = np.asarray([32, 128, 128], dtype=int)
    upper = offsets + roi_shape

    seen = set()
    rows = []
    full_counts = {}

    for i in range(len(frames) - 1):
        ts = frames[i : i + 2]
        raw = []
        for t in ts:
            full = np.asarray(arr[t], dtype=np.float32)
            ds = full[
                :: downsample[0],
                :: downsample[1],
                :: downsample[2],
            ]
            norm = np.clip(
                (ds - q_low) / (q_high - q_low + 1e-6),
                0,
                None,
            )
            raw.append(torch.from_numpy(norm))

        _, det_logits = _encode_public_pair(
            model, torch.stack(raw)
        )

        for local_idx, t in enumerate(ts):
            if t in seen:
                continue
            seen.add(t)

            logits = det_logits[local_idx][0, 0]
            coords_ds = detect_local_maxima(
                logits,
                probability_threshold=float(det_cfg["threshold"]),
                pool_kernel=pool_kernel,
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            full_counts[t] = int(len(coords_ds))

            coords_global = coords_ds * downsample[None, :]
            mask = np.all(
                (coords_global >= offsets[None, :])
                & (coords_global < upper[None, :]),
                axis=1,
            )
            roi_global = coords_global[mask]
            roi_ds = coords_ds[mask].astype(int)

            if (
                len(roi_global)
                > int(
                    graph_cfg[
                        "max_roi_detections_per_frame_fail_only"
                    ]
                )
            ):
                raise RuntimeError(
                    f"{sample_id} {window['window_id']} t={t}: "
                    f"{len(roi_global)} ROI detections above fail-only ceiling."
                )

            for global_coord, ds_coord in zip(
                roi_global, roi_ds, strict=True
            ):
                rows.append(
                    {
                        "t": int(t),
                        "local_z": int(global_coord[0] - offsets[0]),
                        "local_y": int(global_coord[1] - offsets[1]),
                        "local_x": int(global_coord[2] - offsets[2]),
                        "z": int(global_coord[0]),
                        "y": int(global_coord[1]),
                        "x": int(global_coord[2]),
                        "public_detector_probability": float(
                            probs[tuple(ds_coord)]
                        ),
                    }
                )

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError(
            f"{sample_id} {window['window_id']}: public detector produced no ROI detections."
        )

    return table, {
        "sample_id": sample_id,
        "window_id": window["window_id"],
        "frames": frames,
        "roi_detection_rows": int(len(table)),
        "full_frame_detection_counts": full_counts,
    }


def _build_window_features(
    root: Path,
    *,
    sample_id: str,
    window: dict,
    cache_path: Path,
    model,
    model_cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    cfg = read_json(root / "configs/public_detector_integration.json")
    graph_cfg = cfg["candidate_graph"]
    context_cfg = cfg["feature_context"]

    with np.load(cache_path, allow_pickle=False) as z:
        analysis_images = z["images"]
        frames = z["frames"].astype(int)
        offsets = z["offsets"].astype(int)
        spacing = z["spacing_um"].astype(float)

    analysis_shape = np.asarray(analysis_images.shape[1:], dtype=int)

    detections, det_summary = public_window_detections(
        root,
        sample_id=sample_id,
        window=window,
        model=model,
        model_cfg=model_cfg,
    )

    feature_images, feature_offsets, feature_context = _load_feature_context(
        root,
        sample_id=sample_id,
        frames=frames,
        analysis_offsets=offsets,
        analysis_shape=analysis_shape,
        spacing=spacing,
        halo_um=float(context_cfg["halo_um"]),
    )

    pilot_cfg = read_json(root / "configs/pilot.json")
    annotation_cfg = read_json(root / "configs/annotation_eval.json")
    node_cfg = dict(pilot_cfg)
    node_cfg["frames"] = [int(v) for v in frames]
    node_cfg["sigma_um"] = list(annotation_cfg["detector_sigmas_um"])
    node_cfg["max_candidates_per_frame"] = int(
        graph_cfg["node_id_stride"]
    )

    node_parts = []
    detector_rows = []
    descriptor_invalid_total = 0

    for i, t in enumerate(frames):
        frame_det = detections[detections["t"] == int(t)].copy()

        global_coords = frame_det[
            ["z", "y", "x"]
        ].to_numpy(int)
        feature_local_coords = (
            global_coords - feature_offsets[None, :]
        )

        valid = _descriptor_valid_mask(
            feature_local_coords,
            feature_images[i].shape,
            spacing,
            float(context_cfg["patch_radius_um"]),
        )
        invalid_count = int((~valid).sum())
        descriptor_invalid_total += invalid_count

        frame_det = frame_det.loc[valid].copy().reset_index(drop=True)
        local_coords = feature_local_coords[valid]

        if frame_det.empty:
            raise RuntimeError(
                f"{sample_id} {window['window_id']} t={t}: "
                "all public detections are descriptor-invalid at the true image boundary."
            )

        normal, norm_stats = normalize_volume(feature_images[i])
        responses = [
            physical_log(normal, spacing, s)
            for s in node_cfg["sigma_um"]
        ]
        nodes = candidate_features(
            feature_images[i],
            local_coords,
            responses,
            spacing,
            node_cfg["sigma_um"],
            int(t),
            feature_offsets,
            node_cfg,
        )
        if nodes.empty:
            raise RuntimeError(
                f"{sample_id} {window['window_id']} t={t}: no feature-valid ROI nodes."
            )

        analysis_boundary = _analysis_boundary_distance_um(
            nodes[["z", "y", "x"]].to_numpy(int),
            offsets,
            analysis_shape,
            spacing,
        )
        nodes["roi_boundary_distance_um"] = analysis_boundary

        prob_lookup = {
            (int(r.z), int(r.y), int(r.x)):
                float(r.public_detector_probability)
            for r in frame_det.itertuples()
        }
        nodes["public_detector_probability"] = [
            prob_lookup[(int(r.z), int(r.y), int(r.x))]
            for r in nodes.itertuples()
        ]

        node_parts.append(nodes)
        detector_rows.append(
            {
                "t": int(t),
                "roi_candidate_count_before_descriptor_gate": int(len(valid)),
                "descriptor_invalid_candidate_count": invalid_count,
                "roi_candidate_count": int(len(nodes)),
                **{f"norm_{k}": v for k, v in norm_stats.items()},
            }
        )

    nodes = pd.concat(node_parts, ignore_index=True)

    pair_parts = []
    motion_rows = []
    for i in range(len(frames) - 1):
        source = nodes[nodes["t"] == int(frames[i])].copy()
        target = nodes[nodes["t"] == int(frames[i + 1])].copy()

        shift, phase_diag = phase_shift(
            analysis_images[i],
            analysis_images[i + 1],
        )
        drift = shift * spacing

        dense_cfg = dict(pilot_cfg)
        dense_cfg["pair_radius_um"] = float(
            graph_cfg["pair_radius_um"]
        )
        dense_cfg["pair_top_k"] = max(int(len(target)), 1)

        pair = pair_features(source, target, drift, dense_cfg)
        if len(pair) > int(
            graph_cfg["max_pairs_per_transition_fail_only"]
        ):
            raise RuntimeError(
                f"{sample_id} {window['window_id']} "
                f"{frames[i]}->{frames[i+1]} has {len(pair)} pairs above "
                "the fail-only ceiling."
            )
        pair_parts.append(pair)
        motion_rows.append(
            {
                "t_source": int(frames[i]),
                "t_target": int(frames[i + 1]),
                "pair_count": int(len(pair)),
                "drift_z_um": float(drift[0]),
                "drift_y_um": float(drift[1]),
                "drift_x_um": float(drift[2]),
                **phase_diag,
            }
        )

    pairs = pd.concat(pair_parts, ignore_index=True)
    if pairs.empty:
        raise RuntimeError(
            f"{sample_id} {window['window_id']}: no radius-gated pairs."
        )

    r1, r2, _ = _round_tables(
        root,
        feature_images,
        frames,
        feature_offsets,
        spacing,
        nodes,
        pairs,
    )

    boundary_map = (
        nodes.set_index("node_id")["roi_boundary_distance_um"]
        .to_dict()
    )
    border_col = "r1_quality__border_margin_um"
    if border_col not in r1.columns:
        raise RuntimeError(
            f"Expected Round-1 border feature {border_col!r}."
        )
    r1[border_col] = r1["node_id"].map(boundary_map)

    gt_base = root / "outputs/multisample_scout/samples" / sample_id
    gt_nodes = pd.read_csv(gt_base / "gt_nodes.csv")
    gt_edges = pd.read_csv(gt_base / "gt_edges.csv")
    plan_like = {
        "frames": [int(v) for v in frames],
        "roi_offsets_zyx": [int(v) for v in offsets],
        "roi_shape_zyx": [int(v) for v in analysis_shape],
        "spacing_um": [float(v) for v in spacing],
    }

    matches, gt_roi, node_summary = match_annotation_nodes(
        nodes, gt_nodes, plan_like
    )
    labeled = label_annotation_pairs(r2, matches, gt_edges)

    probs = nodes[
        ["node_id", "public_detector_probability"]
    ].copy()
    src = probs.rename(
        columns={
            "node_id": "source_id",
            "public_detector_probability":
                "publicctx_source_detector_probability",
        }
    )
    tgt = probs.rename(
        columns={
            "node_id": "target_id",
            "public_detector_probability":
                "publicctx_target_detector_probability",
        }
    )
    labeled = labeled.merge(
        src, on="source_id", how="left", validate="many_to_one"
    ).merge(
        tgt, on="target_id", how="left", validate="many_to_one"
    )
    labeled["publicctx_detector_probability_min"] = labeled[
        [
            "publicctx_source_detector_probability",
            "publicctx_target_detector_probability",
        ]
    ].min(axis=1)
    labeled["publicctx_detector_probability_mean"] = labeled[
        [
            "publicctx_source_detector_probability",
            "publicctx_target_detector_probability",
        ]
    ].mean(axis=1)
    labeled["publicctx_detector_probability_absdiff"] = (
        labeled["publicctx_source_detector_probability"]
        - labeled["publicctx_target_detector_probability"]
    ).abs()

    labeled = add_candidate_competition_context(
        labeled,
        score_col="r2_geometry__distance",
        higher_is_better=False,
    )
    labeled = add_kinematic_context(
        labeled,
        nodes,
        scale_zyx_um=tuple(float(v) for v in spacing),
    )

    pooled = _pool_round1(r1, labeled)
    pooled["sample_id"] = sample_id
    pooled["window_id"] = window["window_id"]

    strict = pooled[pooled["edge_label"].isin([0.0, 1.0])].copy()

    gt_ids = set(gt_roi["gt_id"].astype(int))
    gt_edges_roi = gt_edges[
        gt_edges["source_gt_id"].astype(int).isin(gt_ids)
        & gt_edges["target_gt_id"].astype(int).isin(gt_ids)
    ]
    represented_pos = set(
        zip(
            strict.loc[strict["edge_label"] == 1, "source_gt_id"].astype(int),
            strict.loc[strict["edge_label"] == 1, "target_gt_id"].astype(int),
            strict=True,
        )
    )

    summary = {
        **det_summary,
        **feature_context,
        **node_summary,
        "descriptor_invalid_candidates": int(descriptor_invalid_total),
        "candidate_pairs": int(len(pairs)),
        "strict_positive_rows": int((strict["edge_label"] == 1).sum()),
        "strict_negative_rows": int((strict["edge_label"] == 0).sum()),
        "ignored_rows": int(pooled["edge_label"].isna().sum()),
        "gt_edges_in_roi": int(len(gt_edges_roi)),
        "represented_gt_edges": int(len(represented_pos)),
        "candidate_edge_recall": (
            float(len(represented_pos) / len(gt_edges_roi))
            if len(gt_edges_roi)
            else None
        ),
        "round1_base_descriptors": int(
            len(
                {
                    c.split("src__r1_", 1)[1]
                    for c in pooled.columns
                    if c.startswith("src__r1_")
                }
            )
        ),
        "round2_descriptors": int(
            len([c for c in pooled.columns if c.startswith("r2_")])
        ),
    }
    return pooled, summary

def build_public_detector_dataset(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/public_detector_integration.json")
    context = read_json(root / cfg["source_context_receipt"])
    if (
        float(context["best_full_context_recall"]) < 0.80
        or int(context["max_negative_pairs_with_matched_endpoints"]) <= 0
    ):
        raise RuntimeError("Full-context public detector gate has not passed.")

    plan = read_json(root / cfg["source_plan"])
    acquire = read_json(root / cfg["source_acquisition"])
    artifact_map = {
        (x["sample_id"], x["window_id"]): root / x["path"]
        for x in acquire["window_artifacts"]
    }

    windows = [
        w for w in plan["windows"]
        if w["sample_id"] in set(cfg["samples"])
    ]
    windows.sort(
        key=lambda w: (
            w["sample_id"],
            -int(w.get("potential_negative_links", 0)),
            -int(w.get("potential_positive_links", 0)),
            w["window_id"],
        )
    )

    out = root / "outputs/public_detector_integration"
    win_dir = out / "windows"
    win_dir.mkdir(parents=True, exist_ok=True)

    model, model_cfg = load_public_model(root)

    parts = []
    summaries = []

    for index, window in enumerate(windows, start=1):
        sid = window["sample_id"]
        wid = window["window_id"]
        dataset_path = win_dir / f"{wid}__dataset.csv"
        summary_path = win_dir / f"{wid}__summary.json"

        if dataset_path.is_file() and summary_path.is_file():
            part = pd.read_csv(dataset_path)
            summary = read_json(summary_path)
            if summary.get("status") != "completed":
                raise RuntimeError(
                    f"Existing window receipt is not completed: {summary_path}"
                )
            print(
                f"PUBLIC_DETECTOR_WINDOW_REUSED {index}/{len(windows)} "
                f"{wid}",
                flush=True,
            )
        else:
            print(
                f"PUBLIC_DETECTOR_WINDOW_STARTED {index}/{len(windows)} "
                f"{wid}",
                flush=True,
            )
            part, summary = _build_window_features(
                root,
                sample_id=sid,
                window=window,
                cache_path=artifact_map[(sid, wid)],
                model=model,
                model_cfg=model_cfg,
            )
            summary = {"status": "completed", **summary}
            atomic(dataset_path, part.to_csv(index=False))
            write_json(summary_path, summary)
            print(
                f"PUBLIC_DETECTOR_WINDOW_COMPLETE {index}/{len(windows)} "
                f"{wid} pos={summary['strict_positive_rows']} "
                f"neg={summary['strict_negative_rows']}",
                flush=True,
            )

        parts.append(part)
        summaries.append(summary)

    dataset = pd.concat(parts, ignore_index=True)
    strict = dataset[dataset["edge_label"].isin([0.0, 1.0])].copy()
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

    summary_df = pd.DataFrame(summaries)
    counts = (
        strict.groupby("sample_id", as_index=False)
        .agg(
            total=("edge_label", "size"),
            positive=("edge_label", "sum"),
        )
    )
    counts["positive"] = counts["positive"].astype(int)
    counts["negative"] = (
        counts["total"] - counts["positive"]
    ).astype(int)

    r1_base = {
        c.split("src__r1_", 1)[1]
        for c in strict.columns
        if c.startswith("src__r1_")
    }
    r2_cols = [c for c in strict.columns if c.startswith("r2_")]

    atomic(out / "window_summary.csv", summary_df.to_csv(index=False))
    atomic(out / "candidate_dataset.csv", dataset.to_csv(index=False))
    atomic(out / "strict_labeled_dataset.csv", strict.to_csv(index=False))
    atomic(out / "strict_label_counts.csv", counts.to_csv(index=False))

    if len(r1_base) != int(
        cfg["feature_contract"]["round1_base_descriptors"]
    ):
        raise RuntimeError(f"Expected 128 Round-1 descriptors, got {len(r1_base)}.")
    if len(r2_cols) != int(
        cfg["feature_contract"]["round2_descriptors"]
    ):
        raise RuntimeError(f"Expected 128 Round-2 descriptors, got {len(r2_cols)}.")

    gate = cfg["model_gate"]
    by_sample = counts.set_index("sample_id")
    ready = all(
        sid in by_sample.index
        and int(by_sample.loc[sid, "positive"])
        >= int(gate["min_positive_per_sample"])
        and int(by_sample.loc[sid, "negative"])
        >= int(gate["min_negative_per_sample"])
        for sid in cfg["samples"]
    )

    receipt = {
        "status": "completed",
        "windows": int(len(windows)),
        "candidate_rows": int(len(dataset)),
        "strict_rows_before_dedup": before,
        "strict_rows_after_dedup": int(len(strict)),
        "positive_rows": int((strict["edge_label"] == 1).sum()),
        "negative_rows": int((strict["edge_label"] == 0).sum()),
        "round1_base_descriptors": int(len(r1_base)),
        "round2_descriptors": int(len(r2_cols)),
        "numeric_features_available": int(
            len(_feature_columns(strict))
        ),
        "sample_counts": counts.to_dict(orient="records"),
        "training_fits": 0,
        "network_requests": 0,
        "model_gate_passed": bool(ready),
        "next_decision": (
            "run_bounded_association_ablation"
            if ready
            else "inspect_strict_label_coverage"
        ),
        "official_score": None,
    }
    write_json(out / "dataset_receipt.json", receipt)
    return receipt


def run_integration_ablation(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/public_detector_integration.json")
    out = root / "outputs/public_detector_integration"
    receipt = read_json(out / "dataset_receipt.json")

    if not receipt["model_gate_passed"]:
        blocked = {
            "status": "blocked_pretraining",
            "training_fits": 0,
            "reason": "strict class gate not passed for both development samples",
            "next_decision": receipt["next_decision"],
            "official_score": None,
        }
        write_json(out / "ablation_receipt.json", blocked)
        return blocked

    strict = pd.read_csv(out / "strict_labeled_dataset.csv")
    candidates = pd.read_csv(out / "candidate_dataset.csv")
    features = _feature_columns(strict)
    samples = list(cfg["samples"])

    fit_count = 0
    metric_rows = []
    coef_rows = []

    for valid_sid in samples:
        train_sid = [s for s in samples if s != valid_sid][0]
        train = strict[strict["sample_id"] == train_sid].copy()
        valid = strict[strict["sample_id"] == valid_sid].copy()
        valid_all = candidates[
            candidates["sample_id"] == valid_sid
        ].copy()

        y_train = train["edge_label"].astype(int).to_numpy()
        y_valid = valid["edge_label"].astype(int).to_numpy()
        if len(np.unique(y_train)) != 2 or len(np.unique(y_valid)) != 2:
            raise RuntimeError(
                f"Both classes required: train={train_sid} valid={valid_sid}."
            )

        X_train = _clean_X(train, features)
        X_valid = _clean_X(valid, features)
        X_valid_all = _clean_X(valid_all, features)

        logistic, hgb = _fit_models(
            X_train,
            y_train,
            {
                "models": {
                    "logistic_full": {
                        "C": 1.0,
                        "max_iter": 2000,
                    },
                    "hist_gradient_boosting_full": {
                        "learning_rate": 0.05,
                        "max_iter": 150,
                        "max_leaf_nodes": 31,
                        "l2_regularization": 1.0,
                    },
                }
            },
        )
        fit_count += 2

        p_train_log = logistic.predict_proba(X_train)[:, 1]
        p_valid_log = logistic.predict_proba(X_valid)[:, 1]
        p_all_log = logistic.predict_proba(X_valid_all)[:, 1]

        p_train_hgb = _predict_hgb(hgb, X_train)
        p_valid_hgb = _predict_hgb(hgb, X_valid)
        p_all_hgb = _predict_hgb(hgb, X_valid_all)

        variants = {
            "logistic_full": (p_train_log, p_valid_log, p_all_log),
            "hist_gradient_boosting_full": (
                p_train_hgb, p_valid_hgb, p_all_hgb
            ),
            "logit_blend_50_50": (
                dual_seed_logit_blend(
                    p_train_log, p_train_hgb, weight_a=0.5
                ),
                dual_seed_logit_blend(
                    p_valid_log, p_valid_hgb, weight_a=0.5
                ),
                dual_seed_logit_blend(
                    p_all_log, p_all_hgb, weight_a=0.5
                ),
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

        all_frame = valid_all.copy().reset_index(drop=True)
        threshold_grid = [
            0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
            0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
        ]

        for name, (pt, pv, pa) in variants.items():
            threshold = _choose_threshold(
                y_train, pt, threshold_grid
            )
            score_col = f"score__{name}"
            all_frame[score_col] = pa
            metric_rows.append(
                {
                    "train_sample": train_sid,
                    "valid_sample": valid_sid,
                    "variant": name,
                    **_binary_metrics(y_valid, pv),
                    **_local_metrics(y_valid, pv, threshold),
                    **_decoded_labeled_metrics(
                        all_frame, score_col, threshold
                    ),
                }
            )

        model = logistic.named_steps["model"]
        imputer = logistic.named_steps["imputer"]
        names = list(imputer.get_feature_names_out(features))
        for name, coef in zip(names, model.coef_[0], strict=False):
            coef_rows.append(
                {
                    "train_sample": train_sid,
                    "valid_sample": valid_sid,
                    "feature": str(name),
                    "coefficient": float(coef),
                    "abs_coefficient": float(abs(coef)),
                }
            )

    if fit_count > int(cfg["model_gate"]["max_model_fits"]):
        raise RuntimeError("Model fit cap exceeded.")

    metrics = pd.DataFrame(metric_rows)
    coefficients = pd.DataFrame(coef_rows)
    aggregate = (
        metrics.groupby("variant", as_index=False)
        .agg(
            folds=("valid_sample", "size"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_average_precision=("average_precision", "mean"),
            mean_brier=("brier", "mean"),
            mean_local_jaccard=("labeled_edge_jaccard", "mean"),
            mean_decoded_jaccard=(
                "decoded_labeled_edge_jaccard", "mean"
            ),
            mean_decoded_f1=("decoded_f1", "mean"),
        )
        .sort_values(
            ["mean_decoded_jaccard", "mean_average_precision"],
            ascending=False,
        )
    )

    atomic(out / "model_metrics.csv", metrics.to_csv(index=False))
    atomic(out / "model_aggregate.csv", aggregate.to_csv(index=False))
    atomic(
        out / "logistic_coefficients.csv",
        coefficients.to_csv(index=False),
    )

    best = aggregate.iloc[0].to_dict()
    final = {
        "status": "completed",
        "training_fits": fit_count,
        "feature_columns_used": int(len(features)),
        "best_variant": str(best["variant"]),
        "best_mean_average_precision": float(
            best["mean_average_precision"]
        ),
        "best_mean_decoded_labeled_edge_jaccard": float(
            best["mean_decoded_jaccard"]
        ),
        "validation": cfg["model_gate"]["validation"],
        "official_score": None,
        "next_decision": "add_public_node_transformer_edge_logits",
    }
    write_json(out / "ablation_receipt.json", final)
    return final
