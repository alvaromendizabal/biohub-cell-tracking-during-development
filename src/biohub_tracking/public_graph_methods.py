
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def prune_short_components(nodes, edges, min_nodes=6):
    adj = defaultdict(set)
    for s, t in edges[["source_id", "target_id"]].itertuples(index=False, name=None):
        adj[int(s)].add(int(t))
        adj[int(t)].add(int(s))

    keep, seen = set(), set()
    for n in nodes.node_id.astype(int):
        if n in seen:
            continue
        stack, comp = [n], []
        seen.add(n)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(comp) >= int(min_nodes):
            keep.update(comp)

    return (
        nodes[nodes.node_id.isin(keep)].copy(),
        edges[
            edges.source_id.isin(keep) & edges.target_id.isin(keep)
        ].copy(),
    )


def solve_constrained_edges(candidates, score_col="edge_score", max_out_degree=2):
    """Degree-constrained global edge selection with SciPy MILP."""
    df = candidates.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        return df.assign(selected=False)

    rows, upper = [], []
    for s in sorted(df.source_id.astype(int).unique()):
        rows.append(np.flatnonzero(df.source_id.to_numpy() == s))
        upper.append(float(max_out_degree))
    for t in sorted(df.target_id.astype(int).unique()):
        rows.append(np.flatnonzero(df.target_id.to_numpy() == t))
        upper.append(1.0)

    A = lil_matrix((len(rows), n), dtype=float)
    for i, idx in enumerate(rows):
        A[i, idx] = 1.0

    result = milp(
        c=-df[score_col].to_numpy(float),
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(
            A.tocsr(),
            -np.inf * np.ones(len(upper)),
            np.asarray(upper),
        ),
        options={"time_limit": 30.0},
    )
    if result.x is None:
        raise RuntimeError(
            f"MILP failed status={result.status}: {result.message}"
        )
    df["selected"] = result.x > 0.5
    return df
