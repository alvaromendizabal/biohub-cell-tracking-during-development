from __future__ import annotations

import numpy as np
import torch


def brightness_shift(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    rng: np.random.Generator,
    shift_range: float = 0.1,
):
    shift = float(rng.uniform(-shift_range, shift_range))
    return imgs + shift, coords, masks


def independent_axis_flips(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    rng: np.random.Generator,
):
    flip_mask = rng.random(3) < 0.5
    dims = [1 + i for i, flag in enumerate(flip_mask) if flag]
    if not dims:
        return imgs, coords, masks

    imgs = imgs.flip(dims=dims)
    coords = coords.clone()
    shape = imgs.shape[1:]
    for axis in range(3):
        if flip_mask[axis]:
            coords[masks, axis] = shape[axis] - coords[masks, axis] - 1
    return imgs, coords, masks
