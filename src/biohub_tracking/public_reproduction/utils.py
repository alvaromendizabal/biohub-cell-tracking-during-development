from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("utf-8")
    return hashlib.sha1(header + payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def positional_features(
    coords_tzyx: np.ndarray,
    image_shape,
    per_axis_dim: int = 8,
) -> np.ndarray:
    coords_tzyx = np.asarray(coords_tzyx, dtype=np.float32)
    image_shape = np.asarray(image_shape, dtype=np.float32)
    norms = coords_tzyx / np.maximum(image_shape, 1)

    freqs = (2 ** np.arange(per_axis_dim // 2, dtype=np.float32)) * np.pi
    parts = []
    for axis in range(4):
        angles = norms[:, axis, None] * freqs[None, :]
        parts.extend([np.sin(angles), np.cos(angles)])
    return np.concatenate(parts, axis=1).astype(np.float32)


def pool_kernel_from_um(um: float, voxel_size):
    kernel = []
    for spacing in voxel_size:
        k = max(1, round(float(um) / float(spacing)))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def detect_local_maxima(
    logits_zyx: torch.Tensor,
    probability_threshold: float,
    pool_kernel,
) -> np.ndarray:
    logits = logits_zyx[None, None]
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(
        logits,
        kernel_size=pool_kernel,
        stride=1,
        padding=pad,
    )
    peaks = (
        (logits == pooled)
        & (torch.sigmoid(logits) > float(probability_threshold))
    )
    return torch.nonzero(peaks[0, 0]).cpu().numpy().astype(np.float32)


def d4_transform_xy(x: torch.Tensor, index: int) -> torch.Tensor:
    if not (0 <= index < 8):
        raise ValueError(index)
    reflected = index >= 4
    rotations = index % 4
    out = x
    if reflected:
        out = torch.flip(out, dims=(-1,))
    if rotations:
        out = torch.rot90(out, rotations, dims=(-2, -1))
    return out


def d4_inverse_xy(x: torch.Tensor, index: int) -> torch.Tensor:
    reflected = index >= 4
    rotations = index % 4
    out = x
    if rotations:
        out = torch.rot90(out, -rotations, dims=(-2, -1))
    if reflected:
        out = torch.flip(out, dims=(-1,))
    return out
