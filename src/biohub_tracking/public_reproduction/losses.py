from __future__ import annotations

import torch
import torch.nn.functional as F


def sparse_edge_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Public sparse supervision semantics: ignore unlabeled rows/columns."""
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    if not mask.any():
        return torch.zeros((), device=logits.device, requires_grad=True)

    probs = torch.softmax(logits, dim=0)
    bce = F.binary_cross_entropy(probs, target, reduction="none")
    p_t = probs * target + (1 - probs) * (1 - target)
    focal = ((1 - p_t) ** 2) * bce
    return focal[mask].mean()


def detection_loss(
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    neg_weight: float = 0.01,
) -> torch.Tensor:
    """Count-normalized detector BCE matching the public training semantics."""
    b = det_logits.shape[0]
    spatial = det_logits.shape[2:]
    logits = det_logits[:, 0]
    target = torch.zeros_like(logits)

    for bi in range(b):
        n = int(mask[bi].sum().item())
        if n == 0:
            continue
        c = coords[bi, :n]
        z = c[:, 0].long().clamp(0, spatial[0] - 1)
        y = c[:, 1].long().clamp(0, spatial[1] - 1)
        x = c[:, 2].long().clamp(0, spatial[2] - 1)
        target[bi, z, y, x] = 1.0

    n_pos = target.reshape(b, -1).sum(1).clamp(min=1)
    n_neg = (target.numel() // b - n_pos).clamp(min=1)
    shape = (b,) + (1,) * len(spatial)
    w_pos = (1.0 / n_pos).reshape(shape)
    w_neg = (neg_weight / n_neg).reshape(shape)
    weight = torch.where(target == 1.0, w_pos, w_neg)

    return (
        F.binary_cross_entropy_with_logits(
            logits,
            target,
            weight=weight,
            reduction="sum",
        )
        / b
    )
