
from __future__ import annotations

import numpy as np
import pandas as pd


def _clip_prob(p, eps=1e-6):
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def logit(p, eps=1e-6):
    p = _clip_prob(p, eps)
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def dual_seed_logit_blend(p_a, p_b, weight_a=0.5):
    w = float(weight_a)
    if not 0.0 <= w <= 1.0:
        raise ValueError("weight_a must be in [0,1]")
    return sigmoid(w * logit(p_a) + (1.0 - w) * logit(p_b))


def weighted_harmonic_fusion(p_forward, p_reverse, reverse_weight=0.2, eps=1e-6):
    """Weighted harmonic fusion for bidirectional association probabilities."""
    w = float(reverse_weight)
    if not 0.0 <= w <= 1.0:
        raise ValueError("reverse_weight must be in [0,1]")
    f = np.clip(np.asarray(p_forward, dtype=float), eps, 1.0)
    r = np.clip(np.asarray(p_reverse, dtype=float), eps, 1.0)
    return 1.0 / ((1.0 - w) / f + w / r)


def agreement_gated_fusion(
    primary,
    secondary,
    *,
    logit_weight=0.5,
    max_abs_prob_disagreement=0.20,
):
    """Blend when seeds agree; preserve primary probability on large disagreement."""
    a = np.asarray(primary, dtype=float)
    b = np.asarray(secondary, dtype=float)
    blend = dual_seed_logit_blend(a, b, weight_a=logit_weight)
    agree = np.abs(a - b) <= float(max_abs_prob_disagreement)
    return np.where(agree, blend, a)


def add_candidate_competition_context(
    pairs: pd.DataFrame,
    *,
    score_col: str,
    source_col="source_id",
    target_col="target_id",
    higher_is_better=True,
):
    df = pairs.copy()
    ascending = not higher_is_better
    df["publicctx_out_rank"] = df.groupby(source_col)[score_col].rank(
        method="first", ascending=ascending
    )
    df["publicctx_in_rank"] = df.groupby(target_col)[score_col].rank(
        method="first", ascending=ascending
    )
    df["publicctx_mutual_best"] = (
        (df["publicctx_out_rank"] == 1) & (df["publicctx_in_rank"] == 1)
    ).astype(float)

    best = df.groupby(source_col)[score_col].transform(
        "max" if higher_is_better else "min"
    )
    df["publicctx_gap_from_best"] = (
        best - df[score_col] if higher_is_better else df[score_col] - best
    )
    return df


def add_kinematic_context(
    pairs: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    scale_zyx_um=(1.625, 0.40625, 0.40625),
):
    scale = np.asarray(scale_zyx_um, dtype=float)
    nmap = nodes.set_index("node_id")
    by_t = {int(t): g.copy() for t, g in nodes.groupby("t")}
    out = pairs.copy()

    accel, turn, speed_ratio, lookahead = [], [], [], []
    for row in out.itertuples(index=False):
        s = nmap.loc[int(row.source_id)]
        t = nmap.loc[int(row.target_id)]

        sxyz = np.array([s["z"], s["y"], s["x"]], float)
        txyz = np.array([t["z"], t["y"], t["x"]], float)
        cur = (txyz - sxyz) * scale
        cur_speed = np.linalg.norm(cur)

        prev_nodes = by_t.get(int(s["t"]) - 1)
        if prev_nodes is None or prev_nodes.empty:
            accel.append(np.nan)
            turn.append(np.nan)
            speed_ratio.append(np.nan)
        else:
            q = prev_nodes[["z", "y", "x"]].to_numpy(float)
            d = np.linalg.norm((q - sxyz) * scale, axis=1)
            prev = q[int(np.argmin(d))]
            pv = (sxyz - prev) * scale
            ps = np.linalg.norm(pv)
            accel.append(float(np.linalg.norm(cur - pv)))
            turn.append(
                float(np.dot(cur, pv) / (cur_speed * ps))
                if cur_speed * ps > 1e-9
                else np.nan
            )
            speed_ratio.append(float(cur_speed / ps) if ps > 1e-9 else np.nan)

        fut = by_t.get(int(t["t"]) + 1)
        if fut is None or fut.empty:
            lookahead.append(np.nan)
        else:
            expected = txyz + (txyz - sxyz)
            fq = fut[["z", "y", "x"]].to_numpy(float)
            resid = np.linalg.norm((fq - expected) * scale, axis=1)
            lookahead.append(float(np.min(resid)))

    out["publicctx_accel_residual_um"] = accel
    out["publicctx_turn_cosine"] = turn
    out["publicctx_speed_ratio"] = speed_ratio
    out["publicctx_forward_lookahead_residual_um"] = lookahead
    return out
