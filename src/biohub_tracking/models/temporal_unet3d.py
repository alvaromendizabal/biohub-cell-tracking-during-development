
from __future__ import annotations


def build_temporal_unet3d(in_channels=3, base_channels=24, out_channels=1):
    import torch
    import torch.nn as nn

    def block(ci, co):
        return nn.Sequential(
            nn.Conv3d(ci, co, 3, padding=1),
            nn.InstanceNorm3d(co, affine=True),
            nn.SiLU(),
            nn.Conv3d(co, co, 3, padding=1),
            nn.InstanceNorm3d(co, affine=True),
            nn.SiLU(),
        )

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            b = base_channels
            self.e1 = block(in_channels, b)
            self.e2 = block(b, b * 2)
            self.e3 = block(b * 2, b * 4)
            self.pool = nn.MaxPool3d(2)
            self.u2 = nn.ConvTranspose3d(b * 4, b * 2, 2, 2)
            self.d2 = block(b * 4, b * 2)
            self.u1 = nn.ConvTranspose3d(b * 2, b, 2, 2)
            self.d1 = block(b * 2, b)
            self.head = nn.Conv3d(b, out_channels, 1)

        def forward(self, x):
            x1 = self.e1(x)
            x2 = self.e2(self.pool(x1))
            x3 = self.e3(self.pool(x2))
            y2 = self.d2(torch.cat([self.u2(x3), x2], 1))
            y1 = self.d1(torch.cat([self.u1(y2), x1], 1))
            return self.head(y1)

    return Net()
