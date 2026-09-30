
from __future__ import annotations


def d4_xy_views(x):
    """Eight D4 views over the final two spatial axes."""
    import torch

    views = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(-2, -1))
        views.append((f"rot{k}", r))
        views.append((f"rot{k}_flip", torch.flip(r, dims=(-1,))))
    return views


def undo_d4_xy(name, x):
    import torch

    if name.endswith("_flip"):
        x = torch.flip(x, dims=(-1,))
        name = name[:-5]
    k = int(name.replace("rot", ""))
    return torch.rot90(x, -k, dims=(-2, -1))
