from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class TemporalAttention(nn.Module):
    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels, n_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape[:3]
        spatial = x.shape[3:]
        s = math.prod(spatial)
        h = (
            x.reshape(b, t, c, s)
            .permute(0, 3, 1, 2)
            .reshape(b * s, t, c)
        )
        h = self.norm(h)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = (
            h.reshape(b, s, t, c)
            .permute(0, 2, 3, 1)
            .reshape(b, t, c, *spatial)
        )
        return x + h


class TemporalUNet3DReproduction(nn.Module):
    """Project-native reproduction of the public organizer TemporalUNet3D."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        layers: Sequence[int] = (32, 64, 128),
        gradient_checkpointing: bool = True,
        skip_fullres_temporal: bool = True,
    ):
        super().__init__()
        layers = list(layers)
        self.gradient_checkpointing = gradient_checkpointing

        self.encoder_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()
        prev = in_channels
        for i, width in enumerate(layers):
            self.encoder_blocks.append(conv_block(prev, width))
            if skip_fullres_temporal and i == 0:
                self.temporal_blocks.append(nn.Identity())
            else:
                self.temporal_blocks.append(TemporalAttention(width, n_heads=4))
            prev = width

        self.pool = nn.MaxPool3d(2, 2)
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(layers) - 1, 0, -1):
            self.upsamples.append(
                nn.Upsample(
                    scale_factor=2,
                    mode="trilinear",
                    align_corners=False,
                )
            )
            self.decoder_blocks.append(
                conv_block(layers[i] + layers[i - 1], layers[i - 1])
            )
        self.head = nn.Conv3d(layers[0], out_channels, kernel_size=1)

    def _run(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return grad_checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        x = x.reshape(b * t, *x.shape[2:])
        skips = []

        for i, (block, temporal) in enumerate(
            zip(self.encoder_blocks, self.temporal_blocks, strict=True)
        ):
            if i > 0:
                x = self.pool(x)
            x = self._run(block, x)
            shaped = x.reshape(b, t, *x.shape[1:])
            x = temporal(shaped).reshape(b * t, *x.shape[1:])
            if i < len(self.encoder_blocks) - 1:
                skips.append(x)

        for up, block, skip in zip(
            self.upsamples,
            self.decoder_blocks,
            skips[::-1],
            strict=True,
        ):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            x = torch.cat([x, skip], dim=1)
            x = self._run(block, x)

        x = self.head(x)
        return x.reshape(b, t, *x.shape[1:])


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            batch_first=True,
            dropout=dropout,
        )
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_padding_mask = ~kv_mask if kv_mask is not None else None
        norm_q = self.norm1(q)
        norm_kv = self.norm1(kv)
        attn_out, _ = self.cross_attn(
            norm_q,
            norm_kv,
            norm_kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        q = q + attn_out
        q = q + self.mlp(self.norm2(q))
        return q


class SimpleNodeTransformerReproduction(nn.Module):
    """Project-native reproduction of the public node-transformer edge model."""

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.3,
        pair_chunk_size: int | None = 32,
    ):
        super().__init__()
        self.pair_chunk_size = pair_chunk_size
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    hidden_dim,
                    n_heads,
                    mlp_ratio,
                    dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        unbatched = feat_t.ndim == 2
        if unbatched:
            feat_t = feat_t.unsqueeze(0)
            feat_t1 = feat_t1.unsqueeze(0)
            coords_t = coords_t.unsqueeze(0)
            coords_t1 = coords_t1.unsqueeze(0)

        q = self.norm_in(self.proj(feat_t))
        k = self.norm_in(self.proj(feat_t1))

        for block in self.blocks:
            if torch.is_grad_enabled():
                q = grad_checkpoint(
                    lambda qq, kk: block(
                        qq, kk, kv_mask=mask_t1
                    ),
                    q,
                    k,
                    use_reentrant=False,
                )
                k = grad_checkpoint(
                    lambda kk, qq: block(
                        kk, qq, kv_mask=mask_t
                    ),
                    k,
                    q,
                    use_reentrant=False,
                )
            else:
                q = block(q, k, kv_mask=mask_t1)
                k = block(k, q, kv_mask=mask_t)

        q = self.norm_out(q)
        k = self.norm_out(k)

        n_t = q.shape[1]
        chunk = self.pair_chunk_size or n_t
        outputs = []
        for i in range(0, n_t, chunk):
            qc = q[:, i : i + chunk]
            cc = coords_t[:, i : i + chunk]
            n1 = k.shape[1]
            qe = qc.unsqueeze(2).expand(-1, -1, n1, -1)
            ke = k.unsqueeze(1).expand(-1, qc.shape[1], -1, -1)
            rel = (
                cc.unsqueeze(2) - coords_t1.unsqueeze(1)
            ) / 100.0
            pair_input = torch.cat([qe, ke, rel], dim=-1)
            outputs.append(self.pair_mlp(pair_input).squeeze(-1))

        logits = torch.cat(outputs, dim=1)
        if unbatched:
            logits = logits.squeeze(0)
        return logits


class UNetNodeTransformerReproduction(nn.Module):
    def __init__(
        self,
        unet_out_channels: int = 32,
        unet_layers=(32, 64, 128),
        pos_feat_dim: int = 32,
    ):
        super().__init__()
        self.unet = TemporalUNet3DReproduction(
            in_channels=1,
            out_channels=unet_out_channels,
            layers=unet_layers,
        )
        self.detect_head = nn.Conv3d(
            unet_out_channels, 1, kernel_size=1
        )
        self.transformer = SimpleNodeTransformerReproduction(
            feat_dim=unet_out_channels + pos_feat_dim,
            hidden_dim=128,
            n_heads=4,
            n_blocks=4,
            dropout=0.3,
            pair_chunk_size=32,
        )

    def encode(self, imgs: torch.Tensor):
        # imgs: (B, W, Z, Y, X)
        unet_out = self.unet(imgs.unsqueeze(2))
        det_logits = [
            self.detect_head(unet_out[:, i])
            for i in range(unet_out.shape[1])
        ]
        return unet_out, det_logits

    def detect(self, frame: torch.Tensor) -> torch.Tensor:
        pair = (
            torch.stack([frame, frame], dim=0)
            .unsqueeze(0)
            .unsqueeze(2)
        )
        out = self.unet(pair)
        return self.detect_head(out[0, 0:1])[0, 0]

    def index_features(
        self,
        feat_maps: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        b, c = feat_maps.shape[:2]
        spatial = feat_maps.shape[2:]
        out = torch.zeros(
            b,
            coords.shape[1],
            c,
            device=feat_maps.device,
            dtype=feat_maps.dtype,
        )
        for bi in range(b):
            n = int(mask[bi].sum().item())
            if n == 0:
                continue
            z = coords[bi, :n, 0].long().clamp(0, spatial[0] - 1)
            y = coords[bi, :n, 1].long().clamp(0, spatial[1] - 1)
            x = coords[bi, :n, 2].long().clamp(0, spatial[2] - 1)
            out[bi, :n] = feat_maps[bi, :, z, y, x].T
        return out
