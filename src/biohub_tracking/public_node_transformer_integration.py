from __future__ import annotations

import ast
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

from .feature_io import atomic, write_json
from .public_detector_integration import (
    _descriptor_valid_mask,
    _encode_public_pair,
    _feature_context_bounds,
    _find_quantiles,
    load_public_model,
    read_json,
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
from .public_reproduction.utils import (
    detect_local_maxima,
    pool_kernel_from_um,
    positional_features,
)
from .public_solution_features import (
    agreement_gated_fusion,
    dual_seed_logit_blend,
)


def _node_id_map_for_frame(
    *,
    t: int,
    full_coords_global: np.ndarray,
    roi_offsets: np.ndarray,
    roi_shape: np.ndarray,
    image_shape: np.ndarray,
    spacing: np.ndarray,
    halo_um: float,
    patch_radius_um: float,
    node_id_stride: int,
) -> tuple[dict[int, int], dict]:
    """Map canonical integration node IDs to full-frame detector indices.

    The public detector first produces *all* full-frame detections. The
    integration lane then keeps detections inside the canonical ROI and drops
    only detections that cannot support a complete descriptor patch at the
    true image boundary. candidate_features enumerates those retained
    detections in order and assigns t*stride + index.
    """
    coords = np.asarray(full_coords_global, dtype=int)
    offsets = np.asarray(roi_offsets, dtype=int)
    shape = np.asarray(roi_shape, dtype=int)
    image_shape = np.asarray(image_shape, dtype=int)
    spacing = np.asarray(spacing, dtype=float)

    upper = offsets + shape
    in_roi = np.all(
        (coords >= offsets[None, :])
        & (coords < upper[None, :]),
        axis=1,
    )
    full_indices = np.flatnonzero(in_roi)
    roi_coords = coords[in_roi]

    feature_lo, feature_hi, halo_vox = _feature_context_bounds(
        offsets,
        shape,
        image_shape,
        spacing,
        halo_um,
    )
    feature_shape = feature_hi - feature_lo
    feature_local = roi_coords - feature_lo[None, :]
    valid = _descriptor_valid_mask(
        feature_local,
        feature_shape,
        spacing,
        patch_radius_um,
    )
    retained_full_indices = full_indices[valid]

    mapping = {
        int(t * node_id_stride + local_index): int(full_index)
        for local_index, full_index in enumerate(retained_full_indices)
    }

    return mapping, {
        "full_detections": int(len(coords)),
        "roi_detections_before_descriptor_gate": int(len(roi_coords)),
        "descriptor_invalid_detections": int((~valid).sum()),
        "integration_nodes": int(len(retained_full_indices)),
        "feature_context_offsets_zyx": [int(v) for v in feature_lo],
        "feature_context_shape_zyx": [int(v) for v in feature_shape],
        "feature_context_halo_vox_zyx": [int(v) for v in halo_vox],
    }


def _rank_desc(values: np.ndarray) -> np.ndarray:
    """1-based competition rank, descending; deterministic for ties."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(-values, kind="stable")
    rank = np.empty(len(values), dtype=np.int32)
    rank[order] = np.arange(1, len(values) + 1, dtype=np.int32)
    return rank


def _best_other(values: np.ndarray, index: int) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return float("nan")
    if index == int(np.argmax(values)):
        tmp = values.copy()
        tmp[index] = -np.inf
        return float(np.max(tmp))
    return float(np.max(values))


def _edge_feature_rows(
    *,
    raw: np.ndarray,
    candidate_rows: pd.DataFrame,
    source_map: dict[int, int],
    target_map: dict[int, int],
) -> pd.DataFrame:
    """Extract public learned-edge features for saved candidate rows."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"Expected 2-D edge matrix, got {raw.shape}.")

    # Exact public inference semantics: normalize over candidate SOURCES for
    # each target. This is dim=0 in the upstream implementation.
    shifted = raw - np.max(raw, axis=0, keepdims=True)
    exp = np.exp(shifted)
    soft = exp / np.sum(exp, axis=0, keepdims=True)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(raw, -60, 60)))

    col_sums = soft.sum(axis=0)
    max_softmax_error = float(np.max(np.abs(col_sums - 1.0)))

    target_ranks = np.stack(
        [_rank_desc(soft[:, j]) for j in range(soft.shape[1])],
        axis=1,
    )
    source_ranks = np.stack(
        [_rank_desc(soft[i, :]) for i in range(soft.shape[0])],
        axis=0,
    )

    target_best = np.max(soft, axis=0)
    source_best = np.max(soft, axis=1)

    eps = 1e-12
    target_entropy = -np.sum(
        soft * np.log(np.clip(soft, eps, 1.0)),
        axis=0,
    )

    rows = []
    missing = []
    for row in candidate_rows.itertuples(index=False):
        sid = int(row.source_id)
        tid = int(row.target_id)
        if sid not in source_map or tid not in target_map:
            missing.append((sid, tid))
            continue

        i = source_map[sid]
        j = target_map[tid]
        prob = float(soft[i, j])

        target_other = _best_other(soft[:, j], i)
        source_other = _best_other(soft[i, :], j)

        rows.append(
            {
                "sample_id": str(row.sample_id),
                "window_id": str(row.window_id),
                "source_id": sid,
                "target_id": tid,
                "t_source": int(row.t_source),
                "t_target": int(row.t_target),
                "publicedge_primary_logit": float(raw[i, j]),
                "publicedge_primary_softmax_prob": prob,
                "publicedge_primary_log_softmax_prob": float(
                    math.log(max(prob, eps))
                ),
                "publicedge_primary_sigmoid_prob": float(sigmoid[i, j]),
                "publicedge_target_source_rank": int(target_ranks[i, j]),
                "publicedge_target_best_prob": float(target_best[j]),
                "publicedge_target_margin_vs_best_other": (
                    float(prob - target_other)
                    if np.isfinite(target_other)
                    else float("nan")
                ),
                "publicedge_target_entropy": float(target_entropy[j]),
                "publicedge_source_target_rank": int(source_ranks[i, j]),
                "publicedge_source_best_prob": float(source_best[i]),
                "publicedge_source_margin_vs_best_other": (
                    float(prob - source_other)
                    if np.isfinite(source_other)
                    else float("nan")
                ),
                "publicedge_is_best_parent": bool(target_ranks[i, j] == 1),
                "publicedge_is_best_child": bool(source_ranks[i, j] == 1),
                "publicedge_mutual_best": bool(
                    target_ranks[i, j] == 1
                    and source_ranks[i, j] == 1
                ),
            }
        )

    if missing:
        raise RuntimeError(
            "Node-ID join coverage failure. Missing candidate pairs: "
            + repr(missing[:20])
        )

    result = pd.DataFrame(rows)
    if len(result) != len(candidate_rows):
        raise RuntimeError(
            f"Expected {len(candidate_rows)} scored rows, got {len(result)}."
        )

    result.attrs["softmax_dim0_max_abs_sum_error"] = max_softmax_error
    return result


def _load_window_plan(root: Path) -> dict[str, dict]:
    cfg = read_json(root / "configs/public_node_transformer_integration.json")
    plan = read_json(root / cfg["plan"])
    return {str(w["window_id"]): w for w in plan["windows"]}


def _parse_full_counts(value):
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return {}


def audit_saved_join_schema(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/public_node_transformer_integration.json")
    base = root / cfg["base_output"]
    win_dir = base / "windows"
    paths = sorted(win_dir.glob("*__dataset.csv"))
    if not paths:
        raise RuntimeError(f"No saved window datasets in {win_dir}")

    required = set(cfg["join_contract"]["keys"])
    rows = []
    for path in paths:
        df = pd.read_csv(path, nrows=100)
        missing = sorted(required - set(df.columns))
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "columns": int(len(df.columns)),
                "missing_join_columns": missing,
                "sample_id": (
                    str(df["sample_id"].iloc[0])
                    if len(df) and "sample_id" in df
                    else None
                ),
                "window_id": (
                    str(df["window_id"].iloc[0])
                    if len(df) and "window_id" in df
                    else None
                ),
            }
        )
        if missing:
            raise RuntimeError(
                f"{path.name} missing join columns: {missing}"
            )

    out = root / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "completed",
        "window_datasets": len(paths),
        "join_keys": list(cfg["join_contract"]["keys"]),
        "all_required_columns_present": True,
        "network_requests": 0,
        "model_fits": 0,
        "rows": rows,
    }
    write_json(out / "schema_audit_receipt.json", receipt)
    return receipt



def _verify_cached_transition_replay(
    *,
    receipt: dict,
    frame_state: dict,
    t_src: int,
    t_tgt: int,
) -> dict:
    """Verify cached score metadata against replayed canonical node state."""
    actual = {
        "full_source_nodes": int(len(frame_state[t_src]["coords_ds"])),
        "full_target_nodes": int(len(frame_state[t_tgt]["coords_ds"])),
        "source_integration_nodes": int(len(frame_state[t_src]["node_map"])),
        "target_integration_nodes": int(len(frame_state[t_tgt]["node_map"])),
    }
    mismatches = {
        key: {
            "cached": int(receipt.get(key, -1)),
            "replayed": value,
        }
        for key, value in actual.items()
        if int(receipt.get(key, -1)) != value
    }
    if mismatches:
        raise RuntimeError(
            "Cached transition is inconsistent with replayed canonical "
            "first-encounter detector state: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return actual

@torch.no_grad()
def _score_window(
    root: Path,
    *,
    window: dict,
    max_new_transitions: int | None,
    processed_counter: list[int],
) -> list[dict]:
    cfg = read_json(root / "configs/public_node_transformer_integration.json")
    base_cfg = read_json(root / "configs/public_detector_integration.json")
    out = root / cfg["output"]
    score_dir = out / "transition_scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    sample_id = str(window["sample_id"])
    window_id = str(window["window_id"])
    base_window_path = (
        root
        / cfg["base_output"]
        / "windows"
        / f"{window_id}__dataset.csv"
    )
    if not base_window_path.is_file():
        raise FileNotFoundError(base_window_path)

    base_rows = pd.read_csv(base_window_path)
    expected_keys = cfg["join_contract"]["keys"]
    for key in expected_keys:
        if key not in base_rows.columns:
            raise RuntimeError(
                f"{base_window_path.name} missing key {key!r}."
            )

    zarr_root = (
        root / "data/multisample_scout/train" / f"{sample_id}.zarr"
    )
    arr = zarr.open_array(zarr_root / "0", mode="r")
    image_shape = np.asarray(arr.shape[1:], dtype=int)
    quantiles = _find_quantiles(read_json(zarr_root / "zarr.json"))
    if quantiles is None:
        raise RuntimeError(f"Missing image quantiles for {sample_id}.")
    q_low, q_high = quantiles

    model, model_cfg = load_public_model(root)
    model.eval()

    downsample = np.asarray(
        model_cfg.get(
            "downsample",
            cfg["public_inference"]["downsample_zyx"],
        ),
        dtype=int,
    )
    ds_shape = tuple(
        int(math.ceil(int(image_shape[i]) / int(downsample[i])))
        for i in range(3)
    )
    spacing = np.asarray([1.625, 0.40625, 0.40625], dtype=float)
    ds_spacing = spacing * downsample
    pool_kernel = pool_kernel_from_um(
        float(cfg["public_inference"]["pool_kernel_um"]),
        ds_spacing,
    )

    offsets = np.asarray(window["roi_offsets_zyx"], dtype=int)
    roi_shape = np.asarray([32, 128, 128], dtype=int)
    frames = [int(v) for v in window["frames"]]

    halo_um = float(base_cfg["feature_context"]["halo_um"])
    patch_radius_um = float(
        base_cfg["feature_context"]["patch_radius_um"]
    )
    stride = int(cfg["join_contract"]["node_id_stride"])

    seen = set()
    frame_state = {}
    receipts = []

    for pair_index in range(len(frames) - 1):
        t_src = int(frames[pair_index])
        t_tgt = int(frames[pair_index + 1])
        transition_key = (
            f"{window_id}__t{t_src:03d}_t{t_tgt:03d}"
        )
        csv_path = score_dir / f"{transition_key}.csv"
        json_path = score_dir / f"{transition_key}.json"

        cached_receipt = None
        if csv_path.is_file() and json_path.is_file():
            cached_receipt = read_json(json_path)
            if cached_receipt.get("status") != "completed":
                raise RuntimeError(
                    f"Incomplete transition receipt: {json_path}"
                )

        if (
            cached_receipt is None
            and max_new_transitions is not None
            and processed_counter[0] >= max_new_transitions
        ):
            break

        started = time.perf_counter()

        raw_frames = []
        for t in (t_src, t_tgt):
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
            raw_frames.append(torch.from_numpy(norm))

        pair_tensor = torch.stack(raw_frames)
        unet_out, det_logits = _encode_public_pair(
            model,
            pair_tensor,
        )

        for local_index, t in enumerate((t_src, t_tgt)):
            if t in seen:
                continue
            seen.add(t)

            logits = det_logits[local_index][0, 0]
            coords_ds = detect_local_maxima(
                logits,
                probability_threshold=float(
                    cfg["public_inference"]["detector_threshold"]
                ),
                pool_kernel=pool_kernel,
            ).astype(np.float32)
            coords_global = (
                coords_ds * downsample[None, :]
            ).astype(np.int32)

            node_map, node_summary = _node_id_map_for_frame(
                t=t,
                full_coords_global=coords_global,
                roi_offsets=offsets,
                roi_shape=roi_shape,
                image_shape=image_shape,
                spacing=spacing,
                halo_um=halo_um,
                patch_radius_um=patch_radius_um,
                node_id_stride=stride,
            )
            frame_state[t] = {
                "coords_ds": coords_ds,
                "coords_global": coords_global,
                "node_map": node_map,
                "summary": node_summary,
            }

        if t_src not in frame_state or t_tgt not in frame_state:
            raise RuntimeError(
                f"Missing frame detection state for {t_src}->{t_tgt}."
            )

        # Resume invariant: cached edge scores are reusable only after the
        # upstream sliding-window detector state has been replayed. This keeps
        # a frame first detected as the target of t-1->t identical when it is
        # later reused as the source of t->t+1.
        if cached_receipt is not None:
            replay = _verify_cached_transition_replay(
                receipt=cached_receipt,
                frame_state=frame_state,
                t_src=t_src,
                t_tgt=t_tgt,
            )
            receipts.append(cached_receipt)
            elapsed = time.perf_counter() - started
            print(
                "PUBLIC_EDGE_TRANSITION_REPLAYED",
                transition_key,
                f"full_nodes={replay['full_source_nodes']}x"
                f"{replay['full_target_nodes']}",
                f"elapsed={elapsed:.3f}s",
                flush=True,
            )
            del unet_out
            continue

        c_src = frame_state[t_src]["coords_ds"]
        c_tgt = frame_state[t_tgt]["coords_ds"]
        n_src = len(c_src)
        n_tgt = len(c_tgt)
        if n_src == 0 or n_tgt == 0:
            raise RuntimeError(
                f"Empty full-frame detection set for {t_src}->{t_tgt}."
            )

        p_coords_src = (
            torch.from_numpy(c_src.astype(np.float32))
            .unsqueeze(0)
        )
        p_coords_tgt = (
            torch.from_numpy(c_tgt.astype(np.float32))
            .unsqueeze(0)
        )
        p_mask_src = torch.ones(
            1, n_src, dtype=torch.bool
        )
        p_mask_tgt = torch.ones(
            1, n_tgt, dtype=torch.bool
        )

        c_src_rel = np.column_stack(
            [
                np.zeros(n_src, dtype=np.float32),
                c_src.astype(np.float32),
            ]
        )
        c_tgt_rel = np.column_stack(
            [
                np.ones(n_tgt, dtype=np.float32),
                c_tgt.astype(np.float32),
            ]
        )
        window_shape = (2,) + tuple(ds_shape)
        p_pos_src = torch.from_numpy(
            positional_features(
                c_src_rel,
                window_shape,
                per_axis_dim=int(
                    cfg["public_inference"][
                        "positional_embedding_per_axis"
                    ]
                ),
            )
        ).unsqueeze(0)
        p_pos_tgt = torch.from_numpy(
            positional_features(
                c_tgt_rel,
                window_shape,
                per_axis_dim=int(
                    cfg["public_inference"][
                        "positional_embedding_per_axis"
                    ]
                ),
            )
        ).unsqueeze(0)

        # Exact project-native equivalent of upstream _index_features +
        # predict_edges. Detection TTA only chooses node coordinates; the
        # transformer consumes ORIGINAL-VIEW UNet features.
        unet_feat_src = model.index_features(
            unet_out[:, 0],
            p_coords_src,
            p_mask_src,
        )
        unet_feat_tgt = model.index_features(
            unet_out[:, 1],
            p_coords_tgt,
            p_mask_tgt,
        )

        feat_src = torch.cat(
            [unet_feat_src, p_pos_src],
            dim=-1,
        )
        feat_tgt = torch.cat(
            [unet_feat_tgt, p_pos_tgt],
            dim=-1,
        )
        ds_tensor = torch.from_numpy(
            downsample.astype(np.float32)
        )
        raw_logits = model.transformer(
            feat_src,
            feat_tgt,
            p_coords_src * ds_tensor,
            p_coords_tgt * ds_tensor,
            p_mask_src,
            p_mask_tgt,
        )[0].cpu().numpy()

        transition_rows = base_rows[
            (base_rows["t_source"].astype(int) == t_src)
            & (base_rows["t_target"].astype(int) == t_tgt)
        ].copy()
        if transition_rows.empty:
            raise RuntimeError(
                f"No saved candidate rows for {transition_key}."
            )

        scored = _edge_feature_rows(
            raw=raw_logits,
            candidate_rows=transition_rows,
            source_map=frame_state[t_src]["node_map"],
            target_map=frame_state[t_tgt]["node_map"],
        )

        elapsed = time.perf_counter() - started
        max_err = float(
            scored.attrs["softmax_dim0_max_abs_sum_error"]
        )
        if max_err > 1e-5:
            raise RuntimeError(
                f"Source-axis softmax normalization error: {max_err}"
            )

        atomic(csv_path, scored.to_csv(index=False))
        receipt = {
            "status": "completed",
            "sample_id": sample_id,
            "window_id": window_id,
            "t_source": t_src,
            "t_target": t_tgt,
            "candidate_rows": int(len(scored)),
            "full_source_nodes": int(n_src),
            "full_target_nodes": int(n_tgt),
            "source_integration_nodes": int(
                len(frame_state[t_src]["node_map"])
            ),
            "target_integration_nodes": int(
                len(frame_state[t_tgt]["node_map"])
            ),
            "join_coverage": 1.0,
            "softmax_dim": 0,
            "softmax_dim0_max_abs_sum_error": max_err,
            "elapsed_seconds": float(elapsed),
            "network_requests": 0,
            "model_fits": 0,
        }
        write_json(json_path, receipt)
        receipts.append(receipt)
        processed_counter[0] += 1

        print(
            "PUBLIC_EDGE_TRANSITION_COMPLETE",
            transition_key,
            f"rows={len(scored)}",
            f"full_nodes={n_src}x{n_tgt}",
            f"elapsed={elapsed:.3f}s",
            flush=True,
        )

    return receipts


def score_public_node_transformer_edges(
    root: Path,
    *,
    max_new_transitions: int | None = None,
    smoke_only: bool = False,
) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/public_node_transformer_integration.json")
    base = root / cfg["base_output"]
    out = root / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)

    base_receipt = read_json(base / "dataset_receipt.json")
    ablation = read_json(base / "ablation_receipt.json")
    if not base_receipt.get("model_gate_passed"):
        raise RuntimeError("Base public-detector model gate did not pass.")
    if (
        ablation.get("next_decision")
        != "add_public_node_transformer_edge_logits"
    ):
        raise RuntimeError(
            "Base ablation is not ready for public edge-logit integration."
        )

    plan_map = _load_window_plan(root)
    window_paths = sorted(
        (base / "windows").glob("*__dataset.csv")
    )
    window_ids = [p.name.replace("__dataset.csv", "") for p in window_paths]

    if smoke_only:
        smoke_id = str(cfg["runtime"]["smoke_window_id"])
        if smoke_id not in plan_map:
            raise RuntimeError(f"Smoke window missing: {smoke_id}")
        smoke_ts = int(cfg["runtime"]["smoke_t_source"])
        smoke_tt = int(cfg["runtime"]["smoke_t_target"])
        smoke_key = f"{smoke_id}__t{smoke_ts:03d}_t{smoke_tt:03d}"
        smoke_json = out / "transition_scores" / f"{smoke_key}.json"
        smoke_csv = out / "transition_scores" / f"{smoke_key}.csv"
        if smoke_json.is_file() and smoke_csv.is_file():
            smoke_transition = read_json(smoke_json)
            if smoke_transition.get("status") == "completed":
                receipt = {
                    "status": "completed",
                    "scope": "worst_case_transition_smoke",
                    "transition": smoke_transition,
                    "next_decision": "run_full_public_edge_scoring",
                    "cache_reused": True,
                }
                write_json(out / "smoke_receipt.json", receipt)
                return receipt
        windows = [plan_map[smoke_id]]
        max_new_transitions = 1
    else:
        windows = [plan_map[wid] for wid in window_ids]

    processed = [0]
    all_receipts = []
    for window in windows:
        all_receipts.extend(
            _score_window(
                root,
                window=window,
                max_new_transitions=max_new_transitions,
                processed_counter=processed,
            )
        )

    if smoke_only:
        smoke_t_source = int(cfg["runtime"]["smoke_t_source"])
        smoke_t_target = int(cfg["runtime"]["smoke_t_target"])
        smoke = [
            x for x in all_receipts
            if int(x["t_source"]) == smoke_t_source
            and int(x["t_target"]) == smoke_t_target
        ]
        if len(smoke) != 1:
            raise RuntimeError(
                "Smoke run did not complete the configured worst-case transition."
            )
        receipt = {
            "status": "completed",
            "scope": "worst_case_transition_smoke",
            "transition": smoke[0],
            "next_decision": "run_full_public_edge_scoring",
        }
        write_json(out / "smoke_receipt.json", receipt)
        return receipt

    expected_transitions = []
    for wid in window_ids:
        w = plan_map[wid]
        frames = [int(v) for v in w["frames"]]
        for i in range(len(frames) - 1):
            expected_transitions.append(
                (wid, frames[i], frames[i + 1])
            )

    score_dir = out / "transition_scores"
    completed = []
    for wid, ts, tt in expected_transitions:
        key = f"{wid}__t{ts:03d}_t{tt:03d}"
        csv_path = score_dir / f"{key}.csv"
        json_path = score_dir / f"{key}.json"
        if csv_path.is_file() and json_path.is_file():
            r = read_json(json_path)
            if r.get("status") == "completed":
                completed.append((wid, ts, tt))

    if len(completed) != len(expected_transitions):
        partial = {
            "status": "partial",
            "expected_transitions": len(expected_transitions),
            "completed_transitions": len(completed),
            "new_transitions_this_run": int(processed[0]),
            "next_decision": "resume_public_edge_scoring",
        }
        write_json(out / "score_receipt.json", partial)
        return partial

    candidate = pd.read_csv(base / "candidate_dataset.csv")
    score_parts = [
        pd.read_csv(path)
        for path in sorted(score_dir.glob("*.csv"))
    ]
    scores = pd.concat(score_parts, ignore_index=True)

    join_keys = list(cfg["join_contract"]["keys"])
    if scores.duplicated(join_keys).any():
        raise RuntimeError("Duplicate learned-edge score keys detected.")

    augmented = candidate.merge(
        scores,
        on=join_keys,
        how="left",
        validate="many_to_one",
    )

    public_cols = list(cfg["public_edge_features"])
    missing_mask = augmented[public_cols].isna().all(axis=1)
    missing_rows = int(missing_mask.sum())
    join_coverage = float(1.0 - missing_rows / max(len(augmented), 1))
    if join_coverage != float(
        cfg["join_contract"]["required_join_coverage"]
    ):
        raise RuntimeError(
            f"Public-edge join coverage {join_coverage:.6f} != 1.0"
        )

    strict = augmented[
        augmented["edge_label"].isin([0.0, 1.0])
    ].copy()
    strict = strict.drop_duplicates(
        subset=[
            "sample_id",
            "source_gt_id",
            "target_gt_id",
            "edge_label",
        ],
        keep="first",
    ).reset_index(drop=True)

    feature_cols = _feature_columns(strict)
    if len(feature_cols) < int(
        cfg["hybrid_model_gate"]["baseline_numeric_features"]
    ):
        raise RuntimeError(
            "Augmented feature set lost original numeric features."
        )

    atomic(
        out / "candidate_dataset_with_public_edges.csv",
        augmented.to_csv(index=False),
    )
    atomic(
        out / "strict_labeled_with_public_edges.csv",
        strict.to_csv(index=False),
    )

    transition_receipts = [
        read_json(path)
        for path in sorted(score_dir.glob("*.json"))
    ]
    receipt = {
        "status": "completed",
        "expected_transitions": len(expected_transitions),
        "completed_transitions": len(transition_receipts),
        "candidate_rows": int(len(augmented)),
        "strict_rows": int(len(strict)),
        "join_coverage": join_coverage,
        "baseline_numeric_features": int(
            cfg["hybrid_model_gate"]["baseline_numeric_features"]
        ),
        "augmented_numeric_features": int(len(feature_cols)),
        "public_edge_features": public_cols,
        "network_requests": 0,
        "model_fits": 0,
        "reference_checkpoint_independence": cfg[
            "hybrid_model_gate"
        ]["reference_checkpoint_independence"],
        "next_decision": "run_public_edge_hybrid_ablation",
    }
    write_json(out / "score_receipt.json", receipt)
    return receipt


def run_public_edge_hybrid_ablation(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/public_node_transformer_integration.json")
    base = root / cfg["base_output"]
    out = root / cfg["output"]

    score_receipt = read_json(out / "score_receipt.json")
    if score_receipt.get("status") != "completed":
        raise RuntimeError("Public learned-edge scoring is not complete.")

    strict = pd.read_csv(out / "strict_labeled_with_public_edges.csv")
    candidates = pd.read_csv(
        out / "candidate_dataset_with_public_edges.csv"
    )
    features = _feature_columns(strict)
    samples = list(cfg["samples"])

    if len(features) < int(
        cfg["hybrid_model_gate"]["baseline_numeric_features"]
    ):
        raise RuntimeError("Original feature representation was not preserved.")

    fit_count = 0
    metric_rows = []
    coef_rows = []

    threshold_grid = list(cfg["direct_score_threshold_grid"])

    for valid_sid in samples:
        train_sid = [s for s in samples if s != valid_sid][0]
        train = strict[strict["sample_id"] == train_sid].copy()
        valid = strict[strict["sample_id"] == valid_sid].copy()
        valid_all = candidates[
            candidates["sample_id"] == valid_sid
        ].copy()

        y_train = train["edge_label"].astype(int).to_numpy()
        y_valid = valid["edge_label"].astype(int).to_numpy()

        direct_col = "publicedge_primary_softmax_prob"
        p_train_direct = train[direct_col].to_numpy(float)
        p_valid_direct = valid[direct_col].to_numpy(float)
        p_all_direct = valid_all[direct_col].to_numpy(float)

        tuned_threshold = _choose_threshold(
            y_train,
            p_train_direct,
            threshold_grid,
        )
        direct_frame = valid_all.copy().reset_index(drop=True)
        direct_frame["score__public_primary_softmax"] = p_all_direct

        for name, threshold in [
            ("public_primary_softmax_default_0p5", 0.5),
            ("public_primary_softmax_train_tuned", tuned_threshold),
        ]:
            metric_rows.append(
                {
                    "train_sample": train_sid,
                    "valid_sample": valid_sid,
                    "variant": name,
                    "source": "released_public_checkpoint_direct",
                    "training_fits": 0,
                    **_binary_metrics(y_valid, p_valid_direct),
                    **_local_metrics(
                        y_valid,
                        p_valid_direct,
                        threshold,
                    ),
                    **_decoded_labeled_metrics(
                        direct_frame,
                        "score__public_primary_softmax",
                        threshold,
                    ),
                }
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
            "hybrid_logistic_full": (
                p_train_log,
                p_valid_log,
                p_all_log,
            ),
            "hybrid_hist_gradient_boosting_full": (
                p_train_hgb,
                p_valid_hgb,
                p_all_hgb,
            ),
            "hybrid_logit_blend_50_50": (
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
            "hybrid_agreement_gate_20pct": (
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

        for name, (pt, pv, pa) in variants.items():
            threshold = _choose_threshold(
                y_train,
                pt,
                [
                    0.10, 0.15, 0.20, 0.25, 0.30,
                    0.35, 0.40, 0.45, 0.50, 0.55,
                    0.60, 0.65, 0.70, 0.75, 0.80,
                    0.85, 0.90,
                ],
            )
            frame = valid_all.copy().reset_index(drop=True)
            score_col = f"score__{name}"
            frame[score_col] = pa
            metric_rows.append(
                {
                    "train_sample": train_sid,
                    "valid_sample": valid_sid,
                    "variant": name,
                    "source": "hybrid_525_plus_public_edge_features",
                    "training_fits": 1 if "logistic" in name or "hist_" in name else 0,
                    **_binary_metrics(y_valid, pv),
                    **_local_metrics(y_valid, pv, threshold),
                    **_decoded_labeled_metrics(
                        frame,
                        score_col,
                        threshold,
                    ),
                }
            )

        model = logistic.named_steps["model"]
        imputer = logistic.named_steps["imputer"]
        names = list(imputer.get_feature_names_out(features))
        for name, coef in zip(
            names,
            model.coef_[0],
            strict=False,
        ):
            coef_rows.append(
                {
                    "train_sample": train_sid,
                    "valid_sample": valid_sid,
                    "feature": str(name),
                    "coefficient": float(coef),
                    "abs_coefficient": float(abs(coef)),
                    "is_publicedge_feature": bool(
                        str(name).startswith("publicedge_")
                    ),
                }
            )

    if fit_count != int(
        cfg["hybrid_model_gate"]["max_model_fits"]
    ):
        raise RuntimeError(
            f"Expected exactly 4 fitted models, got {fit_count}."
        )

    metrics = pd.DataFrame(metric_rows)
    aggregate = (
        metrics.groupby(
            ["variant", "source"],
            as_index=False,
        )
        .agg(
            folds=("valid_sample", "size"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_average_precision=("average_precision", "mean"),
            mean_brier=("brier", "mean"),
            mean_local_jaccard=("labeled_edge_jaccard", "mean"),
            mean_decoded_jaccard=(
                "decoded_labeled_edge_jaccard",
                "mean",
            ),
            mean_decoded_f1=("decoded_f1", "mean"),
        )
        .sort_values(
            [
                "mean_decoded_jaccard",
                "mean_average_precision",
            ],
            ascending=False,
        )
    )

    baseline = pd.read_csv(base / "model_aggregate.csv").copy()
    baseline["source"] = "saved_prior_baseline"
    baseline["variant"] = "baseline::" + baseline["variant"].astype(str)

    comparison = pd.concat(
        [baseline, aggregate],
        ignore_index=True,
        sort=False,
    ).sort_values(
        [
            "mean_decoded_jaccard",
            "mean_average_precision",
        ],
        ascending=False,
    )

    coefficients = pd.DataFrame(coef_rows)
    public_coefficients = coefficients[
        coefficients["is_publicedge_feature"]
    ].copy()

    atomic(out / "hybrid_model_metrics.csv", metrics.to_csv(index=False))
    atomic(out / "hybrid_model_aggregate.csv", aggregate.to_csv(index=False))
    atomic(out / "baseline_vs_public_edge.csv", comparison.to_csv(index=False))
    atomic(
        out / "hybrid_logistic_coefficients.csv",
        coefficients.to_csv(index=False),
    )
    atomic(
        out / "public_edge_coefficients.csv",
        public_coefficients.to_csv(index=False),
    )

    baseline_best = float(
        baseline["mean_decoded_jaccard"].max()
    )
    hybrid_rows = aggregate[
        aggregate["source"]
        == "hybrid_525_plus_public_edge_features"
    ]
    hybrid_best_row = hybrid_rows.iloc[
        hybrid_rows["mean_decoded_jaccard"].argmax()
    ]
    hybrid_best = float(
        hybrid_best_row["mean_decoded_jaccard"]
    )
    delta = hybrid_best - baseline_best

    final = {
        "status": "completed",
        "training_fits": fit_count,
        "baseline_numeric_features": int(
            cfg["hybrid_model_gate"]["baseline_numeric_features"]
        ),
        "augmented_numeric_features": int(len(features)),
        "public_edge_feature_count": int(
            len(cfg["public_edge_features"])
        ),
        "baseline_best_mean_decoded_jaccard": baseline_best,
        "hybrid_best_variant": str(
            hybrid_best_row["variant"]
        ),
        "hybrid_best_mean_decoded_jaccard": hybrid_best,
        "hybrid_minus_baseline_decoded_jaccard": float(delta),
        "reference_checkpoint_independence": (
            cfg["hybrid_model_gate"][
                "reference_checkpoint_independence"
            ]
        ),
        "validation": cfg["hybrid_model_gate"]["validation"],
        "official_score": None,
        "next_decision": "add_secondary_seed_and_reverse_harmonic",
    }
    write_json(out / "ablation_receipt.json", final)
    return final
