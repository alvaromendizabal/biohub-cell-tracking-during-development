from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

from .annotation_eval import match_annotation_nodes
from .feature_io import atomic, write_json
from .public_detector_integration import (
    _descriptor_valid_mask,
    _feature_context_bounds,
    _find_quantiles,
    load_public_model,
    read_json,
)
from .public_reproduction.utils import (
    detect_local_maxima,
    pool_kernel_from_um,
)
from .secondary_reverse_harmonic import (
    _load_model_at,
    verify_secondary_checkpoint,
)


def _encode_detector_tta4(model, pair_tensor: torch.Tensor):
    """Match the existing project primary detector's 4-view XY TTA."""
    imgs = pair_tensor.unsqueeze(0)
    with torch.no_grad():
        _, det_logits = model.encode(imgs)
        for dims in [(-1,), (-2,), (-2, -1)]:
            flipped = imgs.flip(dims)
            _, det_flip = model.encode(flipped)
            for f in range(len(det_logits)):
                det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
        return [x / 4.0 for x in det_logits]


def align_secondary_detector(
    primary: torch.Tensor,
    secondary: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Public mean/std alignment before detector-field blending."""
    p = primary.float()
    s = secondary.float()

    p_mean = p.mean()
    s_mean = s.mean()
    p_scale = p.std(unbiased=False).clamp_min(1e-4)
    s_scale = s.std(unbiased=False).clamp_min(1e-4)
    ratio = (p_scale / s_scale).clamp(0.5, 2.0)
    aligned = (s - s_mean) * ratio + p_mean

    return aligned.to(primary.dtype), {
        "primary_mean": float(p_mean),
        "secondary_mean": float(s_mean),
        "primary_std": float(p_scale),
        "secondary_std": float(s_scale),
        "scale_ratio": float(ratio),
    }


def fuse_detector_fields(
    primary: torch.Tensor,
    secondary: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    aligned, _ = align_secondary_detector(primary, secondary)
    a = float(alpha)
    if not 0.0 <= a < 1.0:
        raise ValueError("alpha must be in [0,1).")
    return (1.0 - a) * primary + a * aligned


def _candidate_pairs(
    nodes: pd.DataFrame,
    *,
    frames: list[int],
    spacing: np.ndarray,
    radius_um: float,
) -> pd.DataFrame:
    rows = []
    for ts, tt in zip(frames[:-1], frames[1:], strict=True):
        src = nodes[nodes["t"] == ts].reset_index(drop=True)
        tgt = nodes[nodes["t"] == tt].reset_index(drop=True)
        if src.empty or tgt.empty:
            continue
        sp = src[["z", "y", "x"]].to_numpy(float) * spacing[None, :]
        tp = tgt[["z", "y", "x"]].to_numpy(float) * spacing[None, :]
        dist = np.linalg.norm(
            sp[:, None, :] - tp[None, :, :],
            axis=2,
        )
        ii, jj = np.where(dist <= float(radius_um))
        for i, j in zip(ii, jj, strict=True):
            rows.append(
                {
                    "source_id": int(src.iloc[i]["node_id"]),
                    "target_id": int(tgt.iloc[j]["node_id"]),
                    "t_source": int(ts),
                    "t_target": int(tt),
                    "distance_um": float(dist[i, j]),
                }
            )
    return pd.DataFrame(rows)


def _evaluate_variant(
    *,
    nodes: pd.DataFrame,
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    window: dict,
    spacing: np.ndarray,
    radius_um: float,
) -> dict:
    frames = [int(v) for v in window["frames"]]
    plan_like = {
        "frames": frames,
        "roi_offsets_zyx": [
            int(v) for v in window["roi_offsets_zyx"]
        ],
        "roi_shape_zyx": [
            int(v)
            for v in window.get("roi_shape_zyx", [32, 128, 128])
        ],
        "spacing_um": [float(v) for v in spacing],
    }

    matches, gt_roi, node_summary = match_annotation_nodes(
        nodes,
        gt_nodes,
        plan_like,
    )

    pairs = _candidate_pairs(
        nodes,
        frames=frames,
        spacing=spacing,
        radius_um=radius_um,
    )

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

    represented = set()
    strict_negative = 0
    for r in pairs.itertuples(index=False):
        s = match_map.get(int(r.source_id))
        t = match_map.get(int(r.target_id))
        if s is None or t is None:
            continue
        if (s, t) in edge_set:
            represented.add((s, t))
        elif s in gt_out or t in gt_in:
            strict_negative += 1

    gt_ids = set(gt_roi["gt_id"].astype(int))
    gt_edges_roi = gt_edges[
        gt_edges["source_gt_id"].astype(int).isin(gt_ids)
        & gt_edges["target_gt_id"].astype(int).isin(gt_ids)
    ]

    matched_dist = matches.loc[
        matches["matched"], "distance_um"
    ].dropna().to_numpy(float)

    return {
        **node_summary,
        "candidate_pairs": int(len(pairs)),
        "strict_negative_pairs": int(strict_negative),
        "gt_edges_in_roi": int(len(gt_edges_roi)),
        "represented_gt_edges": int(len(represented)),
        "candidate_edge_recall": (
            float(len(represented) / len(gt_edges_roi))
            if len(gt_edges_roi)
            else None
        ),
        "mean_localization_error_um": (
            float(np.mean(matched_dist))
            if len(matched_dist)
            else None
        ),
        "median_localization_error_um": (
            float(np.median(matched_dist))
            if len(matched_dist)
            else None
        ),
        "p95_localization_error_um": (
            float(np.quantile(matched_dist, 0.95))
            if len(matched_dist)
            else None
        ),
    }


def run_secondary_detector_fusion_audit(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/secondary_detector_fusion_audit.json")
    secondary_receipt = verify_secondary_checkpoint(root)

    harmonic = read_json(
        root / "outputs/secondary_reverse_harmonic/ablation_receipt.json"
    )
    if harmonic.get("status") != "completed":
        raise RuntimeError("Harmonic milestone is not complete.")

    primary_model, model_cfg = load_public_model(root)
    secondary_model, secondary_cfg = _load_model_at(
        root,
        cfg["secondary_checkpoint"],
    )
    if model_cfg != secondary_cfg:
        raise RuntimeError("Primary/secondary model config mismatch.")

    plan = read_json(root / cfg["source_plan"])
    windows = [
        w for w in plan["windows"]
        if str(w["sample_id"]) in set(cfg["samples"])
    ]
    windows.sort(
        key=lambda w: (str(w["sample_id"]), str(w["window_id"]))
    )
    if len(windows) != 7:
        raise RuntimeError(f"Expected 7 windows, got {len(windows)}.")

    det_cfg = cfg["detector"]
    match_cfg = cfg["matching"]
    integration_cfg = read_json(
        root / "configs/public_detector_integration.json"
    )
    context_cfg = integration_cfg["feature_context"]
    downsample = np.asarray(
        model_cfg.get("downsample", det_cfg["downsample_zyx"]),
        dtype=int,
    )
    spacing = np.asarray([1.625, 0.40625, 0.40625], dtype=float)
    pool_kernel = pool_kernel_from_um(
        float(det_cfg["pool_kernel_um"]),
        spacing * downsample,
    )
    weights = [float(v) for v in cfg["weights"]]

    rows = []
    alignment_rows = []

    for wi, window in enumerate(windows, start=1):
        sid = str(window["sample_id"])
        wid = str(window["window_id"])
        zarr_root = (
            root / "data/multisample_scout/train" / f"{sid}.zarr"
        )
        arr = zarr.open_array(zarr_root / "0", mode="r")
        quantiles = _find_quantiles(read_json(zarr_root / "zarr.json"))
        if quantiles is None:
            raise RuntimeError(f"Missing quantiles for {sid}.")
        q_low, q_high = quantiles

        frames = [int(v) for v in window["frames"]]
        offsets = np.asarray(window["roi_offsets_zyx"], dtype=int)
        roi_shape = np.asarray(
            window.get("roi_shape_zyx", [32, 128, 128]),
            dtype=int,
        )
        upper = offsets + roi_shape
        image_shape = np.asarray(arr.shape[1:], dtype=int)
        feature_lo, feature_hi, _ = _feature_context_bounds(
            offsets,
            roi_shape,
            image_shape,
            spacing,
            float(context_cfg["halo_um"]),
        )
        feature_shape = feature_hi - feature_lo

        # Canonical temporal first-encounter state, one logit field per frame.
        seen = set()
        frame_fields = {}
        for ts, tt in zip(frames[:-1], frames[1:], strict=True):
            raw = []
            for t in (ts, tt):
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
            pair = torch.stack(raw)

            pdet = _encode_detector_tta4(primary_model, pair)
            sdet = _encode_detector_tta4(secondary_model, pair)

            for local_idx, t in enumerate((ts, tt)):
                if t in seen:
                    continue
                seen.add(t)
                aligned, diag = align_secondary_detector(
                    pdet[local_idx][0, 0],
                    sdet[local_idx][0, 0],
                )
                frame_fields[t] = {
                    "primary": pdet[local_idx][0, 0].cpu(),
                    "secondary_aligned": aligned.cpu(),
                }
                alignment_rows.append(
                    {
                        "sample_id": sid,
                        "window_id": wid,
                        "t": int(t),
                        **diag,
                    }
                )

        gt_base = root / "outputs/multisample_scout/samples" / sid
        gt_nodes = pd.read_csv(gt_base / "gt_nodes.csv")
        gt_edges = pd.read_csv(gt_base / "gt_edges.csv")

        for alpha in weights:
            node_rows = []
            frame_counts = []
            descriptor_invalid_total = 0
            for t in frames:
                p = frame_fields[t]["primary"]
                s = frame_fields[t]["secondary_aligned"]
                fused = (1.0 - alpha) * p + alpha * s
                coords_ds = detect_local_maxima(
                    fused,
                    probability_threshold=float(det_cfg["threshold"]),
                    pool_kernel=pool_kernel,
                ).astype(np.float32)
                coords_global = (
                    coords_ds * downsample[None, :]
                ).astype(np.int32)
                roi_mask = np.all(
                    (coords_global >= offsets[None, :])
                    & (coords_global < upper[None, :]),
                    axis=1,
                )
                roi = coords_global[roi_mask]

                feature_local = roi - feature_lo[None, :]
                descriptor_valid = _descriptor_valid_mask(
                    feature_local,
                    feature_shape,
                    spacing,
                    float(context_cfg["patch_radius_um"]),
                )
                invalid_count = int((~descriptor_valid).sum())
                roi = roi[descriptor_valid]

                frame_counts.append(int(len(roi)))
                descriptor_invalid_total += invalid_count

                for local_idx, coord in enumerate(roi):
                    node_rows.append(
                        {
                            "node_id": int(t * 100000 + local_idx),
                            "t": int(t),
                            "z": int(coord[0]),
                            "y": int(coord[1]),
                            "x": int(coord[2]),
                        }
                    )

            nodes = pd.DataFrame(node_rows)
            if nodes.empty:
                raise RuntimeError(
                    f"{wid} alpha={alpha}: zero ROI detections."
                )
            metrics = _evaluate_variant(
                nodes=nodes,
                gt_nodes=gt_nodes,
                gt_edges=gt_edges,
                window=window,
                spacing=spacing,
                radius_um=float(match_cfg["candidate_radius_um"]),
            )
            rows.append(
                {
                    "sample_id": sid,
                    "window_id": wid,
                    "secondary_detection_weight": float(alpha),
                    "roi_detections": int(len(nodes)),
                    "descriptor_invalid_candidates": int(
                        descriptor_invalid_total
                    ),
                    "mean_roi_detections_per_frame": float(
                        np.mean(frame_counts)
                    ),
                    "max_roi_detections_per_frame": int(
                        max(frame_counts)
                    ),
                    **metrics,
                }
            )

        print(
            "SECONDARY_DETECTOR_AUDIT_WINDOW_COMPLETE",
            f"{wi}/{len(windows)}",
            wid,
            flush=True,
        )

    per_window = pd.DataFrame(rows)
    alignment = pd.DataFrame(alignment_rows)

    # Primary-only parity against the completed detector-integration summaries.
    parity_rows = []
    for window in windows:
        wid = str(window["window_id"])
        saved = read_json(
            root
            / cfg["baseline_windows"]
            / f"{wid}__summary.json"
        )
        audit = per_window[
            (per_window["window_id"] == wid)
            & (per_window["secondary_detection_weight"] == 0.0)
        ].iloc[0]
        nr_err = abs(
            float(audit["annotated_node_recall"])
            - float(saved["annotated_node_recall"])
        )
        er_err = abs(
            float(audit["candidate_edge_recall"])
            - float(saved["candidate_edge_recall"])
        )
        node_count_err = abs(
            int(audit["candidate_nodes"])
            - int(saved["candidate_nodes"])
        )
        pair_count_err = abs(
            int(audit["candidate_pairs"])
            - int(saved["candidate_pairs"])
        )
        descriptor_invalid_err = abs(
            int(audit["descriptor_invalid_candidates"])
            - int(saved["descriptor_invalid_candidates"])
        )
        parity_rows.append(
            {
                "window_id": wid,
                "node_recall_abs_error": nr_err,
                "edge_recall_abs_error": er_err,
                "candidate_nodes_abs_error": node_count_err,
                "candidate_pairs_abs_error": pair_count_err,
                "descriptor_invalid_abs_error": descriptor_invalid_err,
                "passed": bool(
                    nr_err <= 1e-12
                    and er_err <= 1e-12
                    and node_count_err == 0
                    and pair_count_err == 0
                    and descriptor_invalid_err == 0
                ),
            }
        )

    parity = pd.DataFrame(parity_rows)
    if not bool(parity["passed"].all()):
        raise RuntimeError(
            "Primary detector audit does not reproduce the saved baseline: "
            + parity[~parity["passed"]].to_json(orient="records")
        )

    aggregate_rows = []
    for alpha, grp in per_window.groupby(
        "secondary_detection_weight",
        sort=True,
    ):
        gt_nodes_total = int(grp["annotated_gt_nodes_in_roi"].sum())
        matched_nodes_total = int(grp["matched_gt_nodes"].sum())
        gt_edges_total = int(grp["gt_edges_in_roi"].sum())
        represented_total = int(grp["represented_gt_edges"].sum())
        aggregate_rows.append(
            {
                "secondary_detection_weight": float(alpha),
                "windows": int(len(grp)),
                "annotated_gt_nodes": gt_nodes_total,
                "matched_gt_nodes": matched_nodes_total,
                "micro_node_recall": (
                    matched_nodes_total / gt_nodes_total
                    if gt_nodes_total else np.nan
                ),
                "gt_edges_in_roi": gt_edges_total,
                "represented_gt_edges": represented_total,
                "micro_candidate_edge_recall": (
                    represented_total / gt_edges_total
                    if gt_edges_total else np.nan
                ),
                "mean_window_node_recall": float(
                    grp["annotated_node_recall"].mean()
                ),
                "mean_window_edge_recall": float(
                    grp["candidate_edge_recall"].mean()
                ),
                "total_roi_detections": int(grp["roi_detections"].sum()),
                "total_candidate_pairs": int(grp["candidate_pairs"].sum()),
                "total_strict_negative_pairs": int(
                    grp["strict_negative_pairs"].sum()
                ),
                "mean_median_localization_error_um": float(
                    grp["median_localization_error_um"].mean()
                ),
            }
        )

    aggregate = pd.DataFrame(aggregate_rows)
    baseline = aggregate[
        aggregate["secondary_detection_weight"] == 0.0
    ].iloc[0]
    aggregate["delta_micro_node_recall"] = (
        aggregate["micro_node_recall"]
        - float(baseline["micro_node_recall"])
    )
    aggregate["delta_micro_candidate_edge_recall"] = (
        aggregate["micro_candidate_edge_recall"]
        - float(baseline["micro_candidate_edge_recall"])
    )
    aggregate["delta_total_candidate_pairs"] = (
        aggregate["total_candidate_pairs"]
        - int(baseline["total_candidate_pairs"])
    )

    # Window-level deltas vs primary-only.
    baseline_window = per_window[
        per_window["secondary_detection_weight"] == 0.0
    ][
        ["window_id", "annotated_node_recall", "candidate_edge_recall"]
    ].rename(
        columns={
            "annotated_node_recall": "baseline_node_recall",
            "candidate_edge_recall": "baseline_edge_recall",
        }
    )
    per_window = per_window.merge(
        baseline_window,
        on="window_id",
        how="left",
        validate="many_to_one",
    )
    per_window["delta_node_recall"] = (
        per_window["annotated_node_recall"]
        - per_window["baseline_node_recall"]
    )
    per_window["delta_edge_recall"] = (
        per_window["candidate_edge_recall"]
        - per_window["baseline_edge_recall"]
    )

    eligible = aggregate[
        aggregate["micro_node_recall"]
        >= float(baseline["micro_node_recall"]) - 1e-12
    ].copy()
    eligible = eligible.sort_values(
        [
            "micro_candidate_edge_recall",
            "micro_node_recall",
            "total_candidate_pairs",
            "total_roi_detections",
            "mean_median_localization_error_um",
        ],
        ascending=[False, False, True, True, True],
    )
    selected = eligible.iloc[0]

    improved = bool(
        float(selected["micro_candidate_edge_recall"])
        > float(baseline["micro_candidate_edge_recall"]) + 1e-12
        or float(selected["micro_node_recall"])
        > float(baseline["micro_node_recall"]) + 1e-12
    )
    selected_weight = float(
        selected["secondary_detection_weight"]
    )

    out = root / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    atomic(out / "per_window.csv", per_window.to_csv(index=False))
    atomic(out / "aggregate.csv", aggregate.to_csv(index=False))
    atomic(out / "alignment_diagnostics.csv", alignment.to_csv(index=False))
    atomic(out / "primary_parity.csv", parity.to_csv(index=False))

    receipt = {
        "status": "completed",
        "windows": int(len(windows)),
        "weights": weights,
        "primary_parity_passed": True,
        "primary_micro_node_recall": float(
            baseline["micro_node_recall"]
        ),
        "primary_micro_candidate_edge_recall": float(
            baseline["micro_candidate_edge_recall"]
        ),
        "selected_weight": selected_weight,
        "selected_micro_node_recall": float(
            selected["micro_node_recall"]
        ),
        "selected_micro_candidate_edge_recall": float(
            selected["micro_candidate_edge_recall"]
        ),
        "selected_total_candidate_pairs": int(
            selected["total_candidate_pairs"]
        ),
        "detector_fusion_improved_recall": improved,
        "descriptor_rebuild_performed": False,
        "feature_recompute": False,
        "training_fits": 0,
        "network_requests": 0,
        "secondary_checkpoint": secondary_receipt,
        "next_decision": (
            "consider_descriptor_rebuild_for_selected_detector"
            if improved and selected_weight != 0.0
            else "keep_primary_detector_and_evaluate_d4_or_deepcenter"
        ),
    }
    write_json(out / "audit_receipt.json", receipt)
    return receipt
