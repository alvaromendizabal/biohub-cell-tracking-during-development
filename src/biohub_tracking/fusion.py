"""Numerical contract extracted from the retained harmonic integration.

The caller supplies the reverse-evidence weight; no private tuning is encoded.
The reciprocal form matches (1+w)/(1/forward+w/reverse).
"""
from __future__ import annotations

import math

import numpy as np


def weighted_harmonic_probability(
    forward: np.ndarray, reverse: np.ndarray, *, weight: float
) -> np.ndarray:
    """Fuse aligned probabilities; reject malformed evidence before arithmetic."""
    pf = np.asarray(forward, dtype=np.float64)
    pr = np.asarray(reverse, dtype=np.float64)
    w = float(weight)
    if pf.shape != pr.shape or pf.size == 0:
        raise ValueError("probabilities must be nonempty and identically shaped")
    if not math.isfinite(w) or w < 0:
        raise ValueError("weight must be finite and non-negative")
    if not np.isfinite(pf).all() or not np.isfinite(pr).all():
        raise ValueError("probabilities must be finite")
    if ((pf < 0) | (pf > 1) | (pr < 0) | (pr > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    pf = np.clip(pf, 1e-12, 1.0)
    pr = np.clip(pr, 1e-12, 1.0)
    forward_mass = 1.0 / (1.0 + w)
    reverse_mass = w / (1.0 + w)
    return 1.0 / (forward_mass / pf + reverse_mass / pr)
