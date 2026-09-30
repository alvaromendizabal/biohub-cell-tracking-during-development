from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

from biohub_tracking.multiwindow_eval import _strict_gt_pair_opportunities
from biohub_tracking.public_reproduction.models import (
    UNetNodeTransformerReproduction,
)
from biohub_tracking.public_reproduction.utils import (
    detect_local_maxima,
    pool_kernel_from_um,
)


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _recursive_pairs(obj, prefix=""):
    rows = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (dict, list)):
                rows.extend(_recursive_pairs(value, path))
            else:
                rows.append({"path": path, "value": value})
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            path = f"{prefix}[{i}]"
            if isinstance(value, (dict, list)):
                rows.extend(_recursive_pairs(value, path))
            else:
                rows.append({"path": path, "value": value})
    return rows


def summarize_artifact_provenance(root: Path) -> dict:
    root = Path(root).resolve()
    config_path = root / "artifacts/public_reference/primary/config.json"
    manifest_path = (
        root / "artifacts/public_reference/primary/ARTIFACT_MANIFEST.json"
    )
    checkpoint_path = (
        root
        / "artifacts/public_reference/primary/edge_predictor_best.pth"
    )

    for path in (config_path, manifest_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = _read_json(config_path)
    manifest = _read_json(manifest_path)

    config_rows = pd.DataFrame(_recursive_pairs(config))
    manifest_rows = pd.DataFrame(_recursive_pairs(manifest))

    out = root / "outputs/public_checkpoint_context_audit"
    out.mkdir(parents=True, exist_ok=True)
    config_rows.to_csv(out / "primary_config_flat.csv", index=False)
    manifest_rows.to_csv(out / "artifact_manifest_flat.csv", index=False)

    searchable = json.dumps(manifest, sort_keys=True).lower()
    epoch_mentions = sorted(
        {
            int(token)
            for token in searchable.replace('"', " ").replace(":", " ").split()
            if token.isdigit() and int(token) in {50, 100, 125, 200, 250, 300, 400}
        }
    )

    receipt = {
        "status": "completed",
        "config_path": str(config_path.relative_to(root)),
        "manifest_path": str(manifest_path.relative_to(root)),
        "checkpoint_path": str(checkpoint_path.relative_to(root)),
        "manifest_name": manifest.get("name") or manifest.get("artifact_name"),
        "epoch_mentions": epoch_mentions,
        "config": config,
        "execution_policy": (
            "Public artifact/config are provenance references; executable model "
            "remains src/biohub_tracking/public_reproduction/."
        ),
    }
    (out / "provenance_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


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


def _greedy_match_ids(
    detected_global,
    gt_frame: pd.DataFrame,
    spacing,
    radius_um=7.0,
):
    if len(gt_frame) == 0 or len(detected_global) == 0:
        return set()

    d = np.asarray(detected_global, dtype=float)
    g = gt_frame[["z", "y", "x"]].to_numpy(float)
    ids = gt_frame["gt_id"].astype(int).to_numpy()
    spacing = np.asarray(spacing, dtype=float)

    dist = np.linalg.norm(
        (d[:, None, :] - g[None, :, :]) * spacing[None, None, :],
        axis=2,
    )

    triples = []
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            if dist[i, j] <= radius_um:
                triples.append((float(dist[i, j]), i, j))
    triples.sort()

    used_d = set()
    used_g = set()
    matched_ids = set()
    for _, i, j in triples:
        if i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        matched_ids.add(int(ids[j]))
    return matched_ids


def _load_model(root: Path):
    checkpoint = (
        root
        / "artifacts/public_reference/primary/edge_predictor_best.pth"
    )
    config = _read_json(
        root / "artifacts/public_reference/primary/config.json"
    )

    out_channels = int(config.get("unet_out_channels", 32))
    layers = tuple(config.get("unet_layers", [32, 64, 128]))
    model = UNetNodeTransformerReproduction(
        unet_out_channels=out_channels,
        unet_layers=layers,
        pos_feat_dim=32,
    )

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/config mismatch: "
            f"missing={missing[:20]} unexpected={unexpected[:20]}"
        )
    model.eval()
    return model, config


def _flip_xy_average(model, frame):
    base = model.detect(frame)
    flipped_frame = torch.flip(frame, dims=(-1,))
    flipped_logits = model.detect(flipped_frame)
    return 0.5 * (base + torch.flip(flipped_logits, dims=(-1,)))


def run_full_context_audit(root: Path, max_windows: int = 2) -> dict:
    root = Path(root).resolve()

    diag = pd.read_csv(
        root
        / "outputs/public_frontier_assoc/candidate_coverage_diagnostic.csv"
    )
    selected = (
        diag.sort_values(
            ["planned_negative_pairs", "annotated_gt_nodes"],
            ascending=False,
        )
        .head(int(max_windows))
        .copy()
    )

    plan = _read_json(root / "outputs/public_frontier_assoc/plan.json")
    plan_map = {
        (x["sample_id"], x["window_id"]): x
        for x in plan["windows"]
    }

    prior = pd.read_csv(
        root
        / "outputs/public_detector_reproduction/primary_detector_eval.csv"
    )

    model, config = _load_model(root)

    downsample = np.asarray(
        config.get("downsample", [1, 4, 4]),
        dtype=int,
    )
    if downsample.shape != (3,):
        raise ValueError(f"Unexpected downsample config: {downsample}")

    original_spacing = np.asarray([1.625, 0.40625, 0.40625])
    ds_spacing = original_spacing * downsample

    thresholds = [0.95, 0.96875, 0.985, 0.99]
    pool_kernel = pool_kernel_from_um(3.0, ds_spacing)

    gt_cache = {}
    rows = []
    endpoint_rows = []

    for item in selected.itertuples(index=False):
        sid = item.sample_id
        wid = item.window_id
        p = plan_map[(sid, wid)]

        if sid not in gt_cache:
            base = root / "outputs/multisample_scout/samples" / sid
            gt_cache[sid] = (
                pd.read_csv(base / "gt_nodes.csv"),
                pd.read_csv(base / "gt_edges.csv"),
            )
        gt_nodes, gt_edges = gt_cache[sid]

        zarr_root = (
            root / "data/multisample_scout/train" / f"{sid}.zarr"
        )
        array_path = zarr_root / "0"
        if not array_path.exists():
            raise FileNotFoundError(array_path)

        root_meta = _read_json(zarr_root / "zarr.json")
        quantiles = _find_quantiles(root_meta)
        if quantiles is None:
            raise RuntimeError(
                f"Missing normalization quantiles for {sid}."
            )
        q_low, q_high = quantiles

        arr = zarr.open_array(array_path, mode="r")
        frames = [int(v) for v in p["frames"]]
        offsets = np.asarray(p["roi_offsets_zyx"], dtype=int)
        roi_shape = np.asarray([32, 128, 128], dtype=int)

        opp = _strict_gt_pair_opportunities(
            gt_nodes,
            gt_edges,
            frames=frames,
            offsets=offsets,
            roi_shape=roi_shape,
            spacing_um=original_spacing,
            descriptor_margin_um=6.0,
            pair_radius_um=15.0,
        )
        negative_pairs = {tuple(v) for v in opp["negative_pairs"]}

        # Only score GT nodes inside the same development ROI so this is a
        # controlled crop-vs-full-context comparison.
        lo = offsets
        hi = offsets + roi_shape
        gt_roi = gt_nodes[
            gt_nodes["t"].astype(int).isin(frames)
            & (gt_nodes["z"] >= lo[0])
            & (gt_nodes["z"] < hi[0])
            & (gt_nodes["y"] >= lo[1])
            & (gt_nodes["y"] < hi[1])
            & (gt_nodes["x"] >= lo[2])
            & (gt_nodes["x"] < hi[2])
        ].copy()

        for mode in ("identity", "flip_xy"):
            for threshold in thresholds:
                matched_ids = set()
                total_detected = 0

                for t in frames:
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
                    frame = torch.from_numpy(norm)

                    with torch.no_grad():
                        if mode == "identity":
                            logits = model.detect(frame)
                        else:
                            logits = _flip_xy_average(model, frame)

                    coords_ds = detect_local_maxima(
                        logits,
                        probability_threshold=threshold,
                        pool_kernel=pool_kernel,
                    )
                    coords_global = coords_ds * downsample[None, :]
                    total_detected += len(coords_global)

                    gt_frame = gt_roi[
                        gt_roi["t"].astype(int) == t
                    ].copy()
                    matched_ids |= _greedy_match_ids(
                        coords_global,
                        gt_frame,
                        original_spacing,
                        radius_um=7.0,
                    )

                gt_ids = set(gt_roi["gt_id"].astype(int))
                matched = len(matched_ids)
                recall = matched / len(gt_ids) if gt_ids else np.nan

                neg_matched = {
                    pair
                    for pair in negative_pairs
                    if pair[0] in matched_ids and pair[1] in matched_ids
                }

                rows.append(
                    {
                        "sample_id": sid,
                        "window_id": wid,
                        "context": "full_frame",
                        "tta_mode": mode,
                        "threshold": float(threshold),
                        "annotated_gt_nodes": len(gt_ids),
                        "matched_gt_nodes": matched,
                        "annotated_node_recall": float(recall),
                        "detected_nodes": int(total_detected),
                        "detections_per_frame": (
                            float(total_detected / len(frames))
                        ),
                        "planned_negative_pairs": len(negative_pairs),
                        "negative_pairs_with_matched_endpoints": len(
                            neg_matched
                        ),
                    }
                )

                endpoint_rows.append(
                    {
                        "sample_id": sid,
                        "window_id": wid,
                        "tta_mode": mode,
                        "threshold": float(threshold),
                        "planned_negative_pairs": len(negative_pairs),
                        "negative_pairs_with_matched_endpoints": len(
                            neg_matched
                        ),
                        "endpoint_coverage": (
                            float(len(neg_matched) / len(negative_pairs))
                            if negative_pairs
                            else np.nan
                        ),
                    }
                )

    full_df = pd.DataFrame(rows)
    endpoint_df = pd.DataFrame(endpoint_rows)

    prior_subset = prior[
        prior["window_id"].isin(selected["window_id"])
    ].copy()
    prior_subset["context"] = "cropped_roi"
    keep = [
        "sample_id",
        "window_id",
        "context",
        "tta_mode",
        "threshold",
        "annotated_gt_nodes",
        "matched_gt_nodes",
        "annotated_node_recall",
        "detected_nodes",
        "detections_per_frame",
        "planned_negative_pairs",
    ]
    prior_subset = prior_subset[keep]

    compare = pd.concat(
        [prior_subset, full_df[keep]],
        ignore_index=True,
    )

    out = root / "outputs/public_checkpoint_context_audit"
    out.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out / "full_context_detector_eval.csv", index=False)
    endpoint_df.to_csv(
        out / "negative_endpoint_coverage.csv",
        index=False,
    )
    compare.to_csv(out / "context_comparison.csv", index=False)

    best_full = full_df.loc[
        full_df["annotated_node_recall"].idxmax()
    ].to_dict()
    best_crop = prior_subset.loc[
        prior_subset["annotated_node_recall"].idxmax()
    ].to_dict()

    max_endpoint = int(
        endpoint_df["negative_pairs_with_matched_endpoints"].max()
    )

    receipt = {
        "status": "completed",
        "windows_evaluated": int(selected.shape[0]),
        "network_requests": 0,
        "model_fits": 0,
        "best_cropped_recall": float(
            best_crop["annotated_node_recall"]
        ),
        "best_full_context_recall": float(
            best_full["annotated_node_recall"]
        ),
        "full_minus_crop_recall": float(
            best_full["annotated_node_recall"]
            - best_crop["annotated_node_recall"]
        ),
        "max_negative_pairs_with_matched_endpoints": max_endpoint,
        "next_decision": (
            "integrate_public_detector_with_full_context"
            if (
                best_full["annotated_node_recall"] >= 0.80
                and max_endpoint > 0
            )
            else "launch_public_training_reproduction_ladder"
        ),
    }
    (out / "context_audit_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt
