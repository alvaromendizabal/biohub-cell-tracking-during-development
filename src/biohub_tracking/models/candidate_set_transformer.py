
from __future__ import annotations


def build_candidate_set_transformer(
    feature_dim,
    hidden_dim=160,
    n_heads=5,
    n_layers=3,
    dropout=0.1,
):
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x, padding_mask=None):
            h = self.input(x)
            h = self.encoder(h, src_key_padding_mask=padding_mask)
            return self.head(h).squeeze(-1)

    return Model()
