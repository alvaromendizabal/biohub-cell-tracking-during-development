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
from .public_node_transformer_integration import (
    _load_window_plan,
    _node_id_map_for_frame,
)
from .public_reproduction.models import (
    UNetNodeTransformerReproduction,
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_secondary_checkpoint(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/secondary_reverse_harmonic.json")
    path = root / cfg["secondary_checkpoint"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Secondary checkpoint missing: {path}. "
            "Fetch it with scripts/fetch_public_reference_artifacts.py "
            "--keys secondary."
        )
    actual = _sha256(path)
    expected = str(cfg["secondary_expected_sha256"])
    if actual != expected:
        raise RuntimeError(
            f"Secondary SHA256 mismatch: {actual} != {expected}"
        )
    return {
        "status": "completed",
        "path": str(path.relative_to(root)),
        "sha256": actual,
        "bytes": int(path.stat().st_size),
    }


def _load_model_at(root: Path, checkpoint_rel: str):
    cfg = read_json(root / "configs/secondary_reverse_harmonic.json")
    model_cfg = read_json(root / cfg["model_config"])
    model = UNetNodeTransformerReproduction(
        unet_out_channels=int(model_cfg.get("unet_out_channels", 32)),
        unet_layers=tuple(model_cfg.get("unet_layers", [32, 64, 128])),
        pos_feat_dim=32,
    )
    state = torch.load(
        root / checkpoint_rel,
        map_location="cpu",
        weights_only=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint architecture mismatch: "
            f"missing={missing[:20]} unexpected={unexpected[:20]}"
        )
    model.eval()
    return model, model_cfg


def softmax_dim0(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    shifted = raw - np.max(raw, axis=0, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=0, keepdims=True)
    return exp / np.clip(denom, 1e-300, None)


def weighted_harmonic_probability(
    p_forward: np.ndarray,
    p_reverse: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Forward-dominant weighted harmonic probability.

    H_w = (1+w)/(1/p_f + w/p_r)

    Increasing w increases the penalty from low reverse-time support while
    preserving H_w(p,p)=p.
    """
    pf = np.clip(np.asarray(p_forward, dtype=np.float64), 1e-12, 1.0)
    pr = np.clip(np.asarray(p_reverse, dtype=np.float64), 1e-12, 1.0)
    w = float(weight)
    if w < 0:
        raise ValueError("weight must be non-negative")
    return (1.0 + w) / ((1.0 / pf) + (w / pr))


@torch.no_grad()
def _predict_edge_raw(
    *,
    model,
    unet_out: torch.Tensor,
    source_coords_ds: np.ndarray,
    target_coords_ds: np.ndarray,
    downsample: np.ndarray,
    source_time_index: int,
    target_time_index: int,
    image_shape_ds: tuple[int, int, int],
) -> np.ndarray:
    src = np.asarray(source_coords_ds, dtype=np.float32)
    tgt = np.asarray(target_coords_ds, dtype=np.float32)
    n_src, n_tgt = len(src), len(tgt)
    if n_src == 0 or n_tgt == 0:
        raise RuntimeError("Cannot score an empty node set.")

    p_src = torch.from_numpy(src).unsqueeze(0)
    p_tgt = torch.from_numpy(tgt).unsqueeze(0)
    m_src = torch.ones(1, n_src, dtype=torch.bool)
    m_tgt = torch.ones(1, n_tgt, dtype=torch.bool)

    src4 = np.column_stack(
        [
            np.full(n_src, source_time_index, dtype=np.float32),
            src,
        ]
    )
    tgt4 = np.column_stack(
        [
            np.full(n_tgt, target_time_index, dtype=np.float32),
            tgt,
        ]
    )
    window_shape = (2,) + tuple(int(v) for v in image_shape_ds)
    pos_src = torch.from_numpy(
        positional_features(
            src4,
            window_shape,
            per_axis_dim=8,
        )
    ).unsqueeze(0)
    pos_tgt = torch.from_numpy(
        positional_features(
            tgt4,
            window_shape,
            per_axis_dim=8,
        )
    ).unsqueeze(0)

    feat_src = model.index_features(
        unet_out[:, source_time_index],
        p_src,
        m_src,
    )
    feat_tgt = model.index_features(
        unet_out[:, target_time_index],
        p_tgt,
        m_tgt,
    )
    feat_src = torch.cat([feat_src, pos_src], dim=-1)
    feat_tgt = torch.cat([feat_tgt, pos_tgt], dim=-1)

    ds = torch.from_numpy(
        np.asarray(downsample, dtype=np.float32)
    )
    raw = model.transformer(
        feat_src,
        feat_tgt,
        p_src * ds,
        p_tgt * ds,
        m_src,
        m_tgt,
    )[0]
    return raw.detach().cpu().numpy()


def _extract_candidate_features(
    *,
    rows: pd.DataFrame,
    source_map: dict[int, int],
    target_map: dict[int, int],
    primary_forward_raw: np.ndarray,
    secondary_forward_raw: np.ndarray,
    primary_reverse_raw: np.ndarray,
    secondary_reverse_raw: np.ndarray,
    secondary_weight: float,
    reverse_weights: list[float],
) -> tuple[pd.DataFrame, dict]:
    pf = softmax_dim0(primary_forward_raw)
    sf = softmax_dim0(secondary_forward_raw)

    # Reverse matrices are (forward_target, forward_source).
    pr_matrix = softmax_dim0(primary_reverse_raw)
    sr_matrix = softmax_dim0(secondary_reverse_raw)

    dual_f_raw = (
        np.asarray(primary_forward_raw, dtype=np.float64)
        + float(secondary_weight)
        * np.asarray(secondary_forward_raw, dtype=np.float64)
    )
    dual_r_raw = (
        np.asarray(primary_reverse_raw, dtype=np.float64)
        + float(secondary_weight)
        * np.asarray(secondary_reverse_raw, dtype=np.float64)
    )
    dual_f = softmax_dim0(dual_f_raw)
    dual_r_matrix = softmax_dim0(dual_r_raw)

    sec_sigmoid = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                np.asarray(secondary_forward_raw, dtype=np.float64),
                -60,
                60,
            )
        )
    )

    result = []
    for row in rows.itertuples(index=False):
        sid = int(row.source_id)
        tid = int(row.target_id)
        if sid not in source_map or tid not in target_map:
            raise RuntimeError(
                f"Candidate node missing from canonical maps: {(sid, tid)}"
            )
        i = int(source_map[sid])
        j = int(target_map[tid])

        p_primary_f = float(pf[i, j])
        p_secondary_f = float(sf[i, j])
        p_primary_r = float(pr_matrix[j, i])
        p_secondary_r = float(sr_matrix[j, i])
        p_dual_f = float(dual_f[i, j])
        p_dual_r = float(dual_r_matrix[j, i])

        rec = {
            "sample_id": str(row.sample_id),
            "window_id": str(row.window_id),
            "source_id": sid,
            "target_id": tid,
            "t_source": int(row.t_source),
            "t_target": int(row.t_target),
            "harmonic_secondary_forward_logit": float(
                secondary_forward_raw[i, j]
            ),
            "harmonic_secondary_forward_softmax_prob": p_secondary_f,
            "harmonic_secondary_forward_sigmoid_prob": float(
                sec_sigmoid[i, j]
            ),
            "harmonic_primary_reverse_logit": float(
                primary_reverse_raw[j, i]
            ),
            "harmonic_primary_reverse_softmax_prob": p_primary_r,
            "harmonic_secondary_reverse_logit": float(
                secondary_reverse_raw[j, i]
            ),
            "harmonic_secondary_reverse_softmax_prob": p_secondary_r,
            "harmonic_dual_forward_w015_prob": p_dual_f,
            "harmonic_dual_reverse_w015_prob": p_dual_r,
            "harmonic_primary_forward_reverse_absdiff": float(
                abs(p_primary_f - p_primary_r)
            ),
            "harmonic_dual_forward_reverse_absdiff": float(
                abs(p_dual_f - p_dual_r)
            ),
            "harmonic_primary_secondary_forward_absdiff": float(
                abs(p_primary_f - p_secondary_f)
            ),
            "harmonic_reverse_support_ratio_primary": float(
                p_primary_r / max(p_primary_f, 1e-12)
            ),
            "harmonic_reverse_support_ratio_dual": float(
                p_dual_r / max(p_dual_f, 1e-12)
            ),
        }
        for w in reverse_weights:
            tag = f"{int(round(float(w) * 100)):03d}"
            rec[f"harmonic_primary_bidir_w{tag}_prob"] = float(
                weighted_harmonic_probability(
                    p_primary_f,
                    p_primary_r,
                    float(w),
                )
            )
            rec[f"harmonic_dual_bidir_w{tag}_prob"] = float(
                weighted_harmonic_probability(
                    p_dual_f,
                    p_dual_r,
                    float(w),
                )
            )
        result.append(rec)

    out = pd.DataFrame(result)
    audit = {
        "primary_forward_softmax_max_error": float(
            np.max(np.abs(pf.sum(axis=0) - 1.0))
        ),
        "secondary_forward_softmax_max_error": float(
            np.max(np.abs(sf.sum(axis=0) - 1.0))
        ),
        "primary_reverse_softmax_max_error": float(
            np.max(np.abs(pr_matrix.sum(axis=0) - 1.0))
        ),
        "secondary_reverse_softmax_max_error": float(
            np.max(np.abs(sr_matrix.sum(axis=0) - 1.0))
        ),
        "dual_forward_softmax_max_error": float(
            np.max(np.abs(dual_f.sum(axis=0) - 1.0))
        ),
        "dual_reverse_softmax_max_error": float(
            np.max(np.abs(dual_r_matrix.sum(axis=0) - 1.0))
        ),
    }
    return out, audit


@torch.no_grad()
def score_secondary_reverse_harmonic(
    root: Path,
    *,
    smoke_only: bool = False,
    max_new_transitions: int | None = None,
) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/secondary_reverse_harmonic.json")
    secondary_receipt = verify_secondary_checkpoint(root)

    base_receipt = read_json(
        root / cfg["base_output"] / "ablation_receipt.json"
    )
    if (
        base_receipt.get("next_decision")
        != "add_secondary_seed_and_reverse_harmonic"
    ):
        raise RuntimeError(
            "Primary NodeTransformer lane is not ready for harmonic integration."
        )

    primary_scores = pd.read_csv(
        root
        / cfg["base_output"]
        / "candidate_dataset_with_public_edges.csv"
    )
    plan_map = _load_window_plan(root)

    window_paths = sorted(
        (
            root / cfg["detector_output"] / "windows"
        ).glob("*__dataset.csv")
    )
    window_ids = [
        p.name.removesuffix("__dataset.csv")
        for p in window_paths
    ]

    if smoke_only:
        window_ids = [str(cfg["runtime"]["smoke_window_id"])]

    primary_model, model_cfg = load_public_model(root)
    secondary_model, secondary_cfg = _load_model_at(
        root,
        cfg["secondary_checkpoint"],
    )
    if model_cfg != secondary_cfg:
        # Same config object is used to construct both models, but preserve
        # explicit check in case the loader contract changes later.
        raise RuntimeError("Primary/secondary architecture config mismatch.")

    downsample = np.asarray(
        model_cfg.get(
            "downsample",
            cfg["fixed_detector"]["downsample_zyx"],
        ),
        dtype=int,
    )
    spacing = np.asarray([1.625, 0.40625, 0.40625], dtype=float)
    ds_spacing = spacing * downsample
    pool_kernel = pool_kernel_from_um(
        float(cfg["fixed_detector"]["pool_kernel_um"]),
        ds_spacing,
    )

    detector_cfg = read_json(
        root / "configs/public_detector_integration.json"
    )
    halo_um = float(detector_cfg["feature_context"]["halo_um"])
    patch_radius_um = float(
        detector_cfg["feature_context"]["patch_radius_um"]
    )
    node_stride = int(
        detector_cfg["candidate_graph"]["node_id_stride"]
    )

    out_dir = root / cfg["output"]
    score_dir = out_dir / "transition_scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    new_count = 0
    receipts = []

    for wid in window_ids:
        window = plan_map[wid]
        sample_id = str(window["sample_id"])
        frames = [int(v) for v in window["frames"]]
        offsets = np.asarray(window["roi_offsets_zyx"], dtype=int)
        roi_shape = np.asarray([32, 128, 128], dtype=int)

        zarr_root = (
            root / "data/multisample_scout/train" / f"{sample_id}.zarr"
        )
        arr = zarr.open_array(zarr_root / "0", mode="r")
        image_shape = np.asarray(arr.shape[1:], dtype=int)
        image_shape_ds = tuple(
            int(math.ceil(int(image_shape[i]) / int(downsample[i])))
            for i in range(3)
        )
        quantiles = _find_quantiles(read_json(zarr_root / "zarr.json"))
        if quantiles is None:
            raise RuntimeError(f"Missing image quantiles for {sample_id}")
        q_low, q_high = quantiles

        seen_frames = set()
        frame_state = {}

        for pair_index in range(len(frames) - 1):
            ts = int(frames[pair_index])
            tt = int(frames[pair_index + 1])
            stem = f"{wid}__t{ts:03d}_t{tt:03d}"
            csv_path = score_dir / f"{stem}.csv"
            json_path = score_dir / f"{stem}.json"

            cached = (
                csv_path.is_file()
                and json_path.is_file()
                and read_json(json_path).get("status") == "completed"
            )

            if (
                not cached
                and max_new_transitions is not None
                and new_count >= int(max_new_transitions)
            ):
                break

            started = time.perf_counter()
            normalized = []
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
                normalized.append(torch.from_numpy(norm))
            pair_tensor = torch.stack(normalized)

            # Primary forward encode includes detector TTA and returns the
            # original-view UNet features used for edge inference.
            primary_f_unet, det_logits = _encode_public_pair(
                primary_model,
                pair_tensor,
            )

            # Reconstruct canonical first-encounter detector state exactly.
            for local_idx, t in enumerate((ts, tt)):
                if t in seen_frames:
                    continue
                seen_frames.add(t)
                det = det_logits[local_idx][0, 0]
                coords_ds = detect_local_maxima(
                    det,
                    probability_threshold=float(
                        cfg["fixed_detector"]["threshold"]
                    ),
                    pool_kernel=pool_kernel,
                ).astype(np.float32)
                coords_global = (
                    coords_ds * downsample[None, :]
                ).astype(np.int32)
                node_map, summary = _node_id_map_for_frame(
                    t=t,
                    full_coords_global=coords_global,
                    roi_offsets=offsets,
                    roi_shape=roi_shape,
                    image_shape=image_shape,
                    spacing=spacing,
                    halo_um=halo_um,
                    patch_radius_um=patch_radius_um,
                    node_id_stride=node_stride,
                )
                frame_state[t] = {
                    "coords_ds": coords_ds,
                    "node_map": node_map,
                    "summary": summary,
                }

            if cached:
                rec = read_json(json_path)
                # Validate replayed node counts against cached metadata.
                actual = {
                    "full_source_nodes": len(frame_state[ts]["coords_ds"]),
                    "full_target_nodes": len(frame_state[tt]["coords_ds"]),
                }
                for k, v in actual.items():
                    if int(rec.get(k, -1)) != int(v):
                        raise RuntimeError(
                            f"Cached harmonic transition drift {stem}: "
                            f"{k} cached={rec.get(k)} replayed={v}"
                        )
                receipts.append(rec)
                print(
                    "HARMONIC_TRANSITION_REPLAYED",
                    stem,
                    f"elapsed={time.perf_counter()-started:.3f}s",
                    flush=True,
                )
                continue

            src_coords = frame_state[ts]["coords_ds"]
            tgt_coords = frame_state[tt]["coords_ds"]

            # Secondary forward original-view encoding on the same fixed nodes.
            secondary_f_unet, _ = secondary_model.encode(
                pair_tensor.unsqueeze(0)
            )

            # True reverse-time temporal context; no reverse detector is used.
            reverse_tensor = torch.stack(
                [pair_tensor[1], pair_tensor[0]]
            )
            primary_r_unet, _ = primary_model.encode(
                reverse_tensor.unsqueeze(0)
            )
            secondary_r_unet, _ = secondary_model.encode(
                reverse_tensor.unsqueeze(0)
            )

            primary_forward_raw = _predict_edge_raw(
                model=primary_model,
                unet_out=primary_f_unet,
                source_coords_ds=src_coords,
                target_coords_ds=tgt_coords,
                downsample=downsample,
                source_time_index=0,
                target_time_index=1,
                image_shape_ds=image_shape_ds,
            )
            secondary_forward_raw = _predict_edge_raw(
                model=secondary_model,
                unet_out=secondary_f_unet,
                source_coords_ds=src_coords,
                target_coords_ds=tgt_coords,
                downsample=downsample,
                source_time_index=0,
                target_time_index=1,
                image_shape_ds=image_shape_ds,
            )
            primary_reverse_raw = _predict_edge_raw(
                model=primary_model,
                unet_out=primary_r_unet,
                source_coords_ds=tgt_coords,
                target_coords_ds=src_coords,
                downsample=downsample,
                source_time_index=0,
                target_time_index=1,
                image_shape_ds=image_shape_ds,
            )
            secondary_reverse_raw = _predict_edge_raw(
                model=secondary_model,
                unet_out=secondary_r_unet,
                source_coords_ds=tgt_coords,
                target_coords_ds=src_coords,
                downsample=downsample,
                source_time_index=0,
                target_time_index=1,
                image_shape_ds=image_shape_ds,
            )

            transition_rows = primary_scores[
                (primary_scores["window_id"].astype(str) == wid)
                & (primary_scores["t_source"].astype(int) == ts)
                & (primary_scores["t_target"].astype(int) == tt)
            ].copy()
            if transition_rows.empty:
                raise RuntimeError(f"No base rows for {stem}")

            features, audit = _extract_candidate_features(
                rows=transition_rows,
                source_map=frame_state[ts]["node_map"],
                target_map=frame_state[tt]["node_map"],
                primary_forward_raw=primary_forward_raw,
                secondary_forward_raw=secondary_forward_raw,
                primary_reverse_raw=primary_reverse_raw,
                secondary_reverse_raw=secondary_reverse_raw,
                secondary_weight=float(
                    cfg["association"]["secondary_edge_logit_weight"]
                ),
                reverse_weights=[
                    float(v)
                    for v in cfg["association"]["reverse_weights"]
                ],
            )

            # Critical parity gate: newly recomputed primary forward scores
            # must exactly match the completed prior lane.
            parity = transition_rows[
                [
                    "sample_id",
                    "window_id",
                    "source_id",
                    "target_id",
                    "t_source",
                    "t_target",
                    "publicedge_primary_softmax_prob",
                ]
            ].merge(
                features,
                on=[
                    "sample_id",
                    "window_id",
                    "source_id",
                    "target_id",
                    "t_source",
                    "t_target",
                ],
                how="inner",
                validate="one_to_one",
            )
            pf = softmax_dim0(primary_forward_raw)
            recomputed = []
            for row in transition_rows.itertuples(index=False):
                i = frame_state[ts]["node_map"][int(row.source_id)]
                j = frame_state[tt]["node_map"][int(row.target_id)]
                recomputed.append(float(pf[i, j]))
            parity_error = float(
                np.max(
                    np.abs(
                        np.asarray(recomputed)
                        - transition_rows[
                            "publicedge_primary_softmax_prob"
                        ].to_numpy(float)
                    )
                )
            )
            tolerance = float(
                cfg["association"]["primary_forward_parity_tolerance"]
            )
            if parity_error > tolerance:
                raise RuntimeError(
                    f"Primary forward parity failed for {stem}: "
                    f"{parity_error} > {tolerance}"
                )

            for key, value in audit.items():
                if value > 1e-5:
                    raise RuntimeError(
                        f"Softmax normalization failed {stem} {key}={value}"
                    )

            atomic(csv_path, features.to_csv(index=False))
            rec = {
                "status": "completed",
                "sample_id": sample_id,
                "window_id": wid,
                "t_source": ts,
                "t_target": tt,
                "candidate_rows": int(len(features)),
                "full_source_nodes": int(len(src_coords)),
                "full_target_nodes": int(len(tgt_coords)),
                "join_coverage": 1.0,
                "primary_forward_parity_max_abs_error": parity_error,
                "secondary_edge_logit_weight": float(
                    cfg["association"]["secondary_edge_logit_weight"]
                ),
                "reverse_weights": list(
                    cfg["association"]["reverse_weights"]
                ),
                **audit,
                "elapsed_seconds": float(time.perf_counter() - started),
                "network_requests": 0,
                "model_fits": 0,
            }
            write_json(json_path, rec)
            receipts.append(rec)
            new_count += 1
            print(
                "HARMONIC_TRANSITION_COMPLETE",
                stem,
                f"rows={len(features)}",
                f"nodes={len(src_coords)}x{len(tgt_coords)}",
                f"parity={parity_error:.3e}",
                f"elapsed={rec['elapsed_seconds']:.3f}s",
                flush=True,
            )

            if smoke_only:
                receipt = {
                    "status": "completed",
                    "scope": "secondary_reverse_harmonic_smoke",
                    "secondary_checkpoint": secondary_receipt,
                    "transition": rec,
                    "next_decision": "run_full_harmonic_scoring",
                }
                write_json(out_dir / "smoke_receipt.json", receipt)
                return receipt

    expected = len(window_ids) * 3
    completed_files = []
    for p in sorted(score_dir.glob("*.json")):
        r = read_json(p)
        if r.get("status") == "completed":
            completed_files.append(r)

    if smoke_only:
        raise RuntimeError("Configured smoke transition was not scored.")

    if len(completed_files) != expected:
        receipt = {
            "status": "partial",
            "expected_transitions": expected,
            "completed_transitions": len(completed_files),
            "new_transitions_this_run": int(new_count),
            "next_decision": "resume_harmonic_scoring",
        }
        write_json(out_dir / "score_receipt.json", receipt)
        return receipt

    parts = [
        pd.read_csv(p)
        for p in sorted(score_dir.glob("*.csv"))
    ]
    scores = pd.concat(parts, ignore_index=True)

    join_keys = [
        "sample_id",
        "window_id",
        "source_id",
        "target_id",
        "t_source",
        "t_target",
    ]
    if scores.duplicated(join_keys).any():
        raise RuntimeError("Duplicate harmonic score keys.")

    augmented = primary_scores.merge(
        scores,
        on=join_keys,
        how="left",
        validate="one_to_one",
    )
    new_features = list(cfg["new_features"])
    missing = augmented[new_features].isna().all(axis=1)
    coverage = float(1.0 - missing.sum() / max(len(augmented), 1))
    if coverage != 1.0:
        raise RuntimeError(f"Harmonic join coverage {coverage} != 1.0")

    strict = augmented[
        augmented["edge_label"].isin([0.0, 1.0])
    ].drop_duplicates(
        subset=[
            "sample_id",
            "source_gt_id",
            "target_gt_id",
            "edge_label",
        ],
        keep="first",
    ).reset_index(drop=True)

    feature_cols = _feature_columns(strict)
    base_features = _feature_columns(
        primary_scores[
            primary_scores["edge_label"].isin([0.0, 1.0])
        ].copy()
    )

    atomic(
        out_dir / "candidate_dataset_with_harmonic.csv",
        augmented.to_csv(index=False),
    )
    atomic(
        out_dir / "strict_labeled_with_harmonic.csv",
        strict.to_csv(index=False),
    )

    max_parity = max(
        float(r["primary_forward_parity_max_abs_error"])
        for r in completed_files
    )
    receipt = {
        "status": "completed",
        "expected_transitions": expected,
        "completed_transitions": len(completed_files),
        "candidate_rows": int(len(augmented)),
        "strict_rows": int(len(strict)),
        "join_coverage": coverage,
        "base_numeric_features": int(len(base_features)),
        "augmented_numeric_features": int(len(feature_cols)),
        "new_feature_count": int(len(new_features)),
        "primary_forward_parity_max_abs_error": max_parity,
        "secondary_checkpoint": secondary_receipt,
        "network_requests": 0,
        "model_fits": 0,
        "next_decision": "run_harmonic_ablation",
    }
    write_json(out_dir / "score_receipt.json", receipt)
    return receipt


def run_harmonic_ablation(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/secondary_reverse_harmonic.json")
    out = root / cfg["output"]
    score_receipt = read_json(out / "score_receipt.json")
    if score_receipt.get("status") != "completed":
        raise RuntimeError("Harmonic scoring is not complete.")

    strict = pd.read_csv(out / "strict_labeled_with_harmonic.csv")
    candidates = pd.read_csv(out / "candidate_dataset_with_harmonic.csv")
    features = _feature_columns(strict)

    base_ablation = read_json(
        root / cfg["base_output"] / "ablation_receipt.json"
    )
    baseline_best = float(
        base_ablation["hybrid_best_mean_decoded_jaccard"]
    )

    samples = list(cfg["samples"])
    metrics = []
    coefficients = []
    fit_count = 0

    direct_cols = {
        "secondary_forward": "harmonic_secondary_forward_softmax_prob",
        "dual_seed_forward_w015": "harmonic_dual_forward_w015_prob",
        "primary_bidir_w015": "harmonic_primary_bidir_w015_prob",
        "primary_bidir_w030": "harmonic_primary_bidir_w030_prob",
        "dual_bidir_w015": "harmonic_dual_bidir_w015_prob",
        "dual_bidir_w030": "harmonic_dual_bidir_w030_prob",
    }

    direct_threshold_grid = [
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
        0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
        0.40, 0.45, 0.50, 0.55, 0.60, 0.70,
        0.80, 0.90,
    ]

    for valid_sid in samples:
        train_sid = [s for s in samples if s != valid_sid][0]
        train = strict[strict["sample_id"] == train_sid].copy()
        valid = strict[strict["sample_id"] == valid_sid].copy()
        valid_all = candidates[
            candidates["sample_id"] == valid_sid
        ].copy()

        y_train = train["edge_label"].astype(int).to_numpy()
        y_valid = valid["edge_label"].astype(int).to_numpy()

        for variant, col in direct_cols.items():
            pt = train[col].to_numpy(float)
            pv = valid[col].to_numpy(float)
            pa = valid_all[col].to_numpy(float)
            threshold = _choose_threshold(
                y_train,
                pt,
                direct_threshold_grid,
            )
            frame = valid_all.copy().reset_index(drop=True)
            score_col = f"score__{variant}"
            frame[score_col] = pa
            metrics.append(
                {
                    "train_sample": train_sid,
                    "valid_sample": valid_sid,
                    "variant": variant,
                    "source": "no_fit_reference_fusion",
                    "training_fits": 0,
                    **_binary_metrics(y_valid, pv),
                    **_local_metrics(y_valid, pv, threshold),
                    **_decoded_labeled_metrics(
                        frame, score_col, threshold
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

        ptl = logistic.predict_proba(X_train)[:, 1]
        pvl = logistic.predict_proba(X_valid)[:, 1]
        pal = logistic.predict_proba(X_valid_all)[:, 1]
        pth = _predict_hgb(hgb, X_train)
        pvh = _predict_hgb(hgb, X_valid)
        pah = _predict_hgb(hgb, X_valid_all)

        variants = {
            "harmonic_hybrid_logistic_full": (ptl, pvl, pal),
            "harmonic_hybrid_hgb_full": (pth, pvh, pah),
            "harmonic_hybrid_logit_blend_50_50": (
                dual_seed_logit_blend(ptl, pth, weight_a=0.5),
                dual_seed_logit_blend(pvl, pvh, weight_a=0.5),
                dual_seed_logit_blend(pal, pah, weight_a=0.5),
            ),
            "harmonic_hybrid_agreement_gate_20pct": (
                agreement_gated_fusion(
                    ptl, pth, logit_weight=0.5,
                    max_abs_prob_disagreement=0.20,
                ),
                agreement_gated_fusion(
                    pvl, pvh, logit_weight=0.5,
                    max_abs_prob_disagreement=0.20,
                ),
                agreement_gated_fusion(
                    pal, pah, logit_weight=0.5,
                    max_abs_prob_disagreement=0.20,
                ),
            ),
        }
        threshold_grid = [
            0.10, 0.15, 0.20, 0.25, 0.30,
            0.35, 0.40, 0.45, 0.50, 0.55,
            0.60, 0.65, 0.70, 0.75, 0.80,
            0.85, 0.90,
        ]
        for name, (pt, pv, pa) in variants.items():
            threshold = _choose_threshold(
                y_train, pt, threshold_grid
            )
            frame = valid_all.copy().reset_index(drop=True)
            score_col = f"score__{name}"
            frame[score_col] = pa
            metrics.append(
                {
                    "train_sample": train_sid,
                    "valid_sample": valid_sid,
                    "variant": name,
                    "source": "all_features_plus_secondary_reverse",
                    "training_fits": (
                        1 if name in {
                            "harmonic_hybrid_logistic_full",
                            "harmonic_hybrid_hgb_full",
                        } else 0
                    ),
                    **_binary_metrics(y_valid, pv),
                    **_local_metrics(y_valid, pv, threshold),
                    **_decoded_labeled_metrics(
                        frame, score_col, threshold
                    ),
                }
            )

        model = logistic.named_steps["model"]
        imputer = logistic.named_steps["imputer"]
        names = list(imputer.get_feature_names_out(features))
        for name, coef in zip(names, model.coef_[0], strict=False):
            coefficients.append(
                {
                    "train_sample": train_sid,
                    "valid_sample": valid_sid,
                    "feature": str(name),
                    "coefficient": float(coef),
                    "abs_coefficient": float(abs(coef)),
                    "is_harmonic_feature": bool(
                        str(name).startswith("harmonic_")
                    ),
                }
            )

    if fit_count != int(cfg["runtime"]["max_model_fits"]):
        raise RuntimeError(
            f"Expected 4 model fits, got {fit_count}."
        )

    metrics_df = pd.DataFrame(metrics)
    aggregate = (
        metrics_df.groupby(["variant", "source"], as_index=False)
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

    base_compare = pd.read_csv(
        root / cfg["base_output"] / "baseline_vs_public_edge.csv"
    ).copy()
    base_compare["stage"] = "through_primary_node_transformer"
    current = aggregate.copy()
    current["stage"] = "secondary_reverse_harmonic"
    comparison = pd.concat(
        [base_compare, current],
        ignore_index=True,
        sort=False,
    ).sort_values(
        ["mean_decoded_jaccard", "mean_average_precision"],
        ascending=False,
    )

    coef = pd.DataFrame(coefficients)
    harmonic_coef = coef[
        coef["is_harmonic_feature"]
    ].copy()

    atomic(out / "harmonic_model_metrics.csv", metrics_df.to_csv(index=False))
    atomic(out / "harmonic_model_aggregate.csv", aggregate.to_csv(index=False))
    atomic(out / "all_stage_comparison.csv", comparison.to_csv(index=False))
    atomic(out / "harmonic_logistic_coefficients.csv", coef.to_csv(index=False))
    atomic(out / "harmonic_feature_coefficients.csv", harmonic_coef.to_csv(index=False))

    fitted = aggregate[
        aggregate["source"] == "all_features_plus_secondary_reverse"
    ]
    best = fitted.sort_values(
        ["mean_decoded_jaccard", "mean_average_precision"],
        ascending=False,
    ).iloc[0]
    best_j = float(best["mean_decoded_jaccard"])

    direct = aggregate[
        aggregate["source"] == "no_fit_reference_fusion"
    ].sort_values(
        ["mean_decoded_jaccard", "mean_average_precision"],
        ascending=False,
    )
    best_direct = direct.iloc[0]

    receipt = {
        "status": "completed",
        "training_fits": fit_count,
        "base_numeric_features": int(
            score_receipt["base_numeric_features"]
        ),
        "augmented_numeric_features": int(len(features)),
        "new_harmonic_feature_count": int(
            score_receipt["new_feature_count"]
        ),
        "prior_best_mean_decoded_jaccard": baseline_best,
        "best_hybrid_variant": str(best["variant"]),
        "best_hybrid_mean_decoded_jaccard": best_j,
        "hybrid_delta_vs_prior": float(best_j - baseline_best),
        "best_no_fit_fusion_variant": str(best_direct["variant"]),
        "best_no_fit_fusion_mean_decoded_jaccard": float(
            best_direct["mean_decoded_jaccard"]
        ),
        "reference_checkpoint_independence": (
            cfg["validation"]["reference_checkpoint_independence"]
        ),
        "validation": cfg["validation"]["scheme"],
        "official_score": None,
        "next_decision": "evaluate_secondary_detection_d4_deepcenter_or_independent_retrain",
    }
    write_json(out / "ablation_receipt.json", receipt)
    return receipt
