"""Registered capacity and global-query controls; v3 classes stay unchanged."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .detail_route import UNetDeepLociDetailRoute
from .unet import UNetDeepBlink


class UNetDeepLociGridControl(UNetDeepBlink):
    """G0: an active capacity-matched residual branch using grid features only."""
    requires_valid_mask = True

    def __init__(self, cell_size: int = 4, **kwargs):
        if cell_size != 4:
            raise ValueError("G0 requires cell_size=4.")
        super().__init__(cell_size=cell_size, **kwargs)
        channels = self.logit_block.out_channels
        self.grid_projection = nn.Sequential(nn.Conv2d(channels, 16, 1), nn.ReLU(),
                                             nn.Conv2d(16, 10, 1))
        self.residual = nn.Sequential(nn.Conv2d(channels + 10, 32, 1), nn.ReLU(),
                                      nn.Conv2d(32, 3, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, inputs, valid_mask=None):
        if valid_mask is None:
            raise ValueError("G0 requires an image-geometry mask.")
        grid = self._forward_features(inputs)
        valid = F.avg_pool2d(valid_mask.float(), 4) > 0
        residual = self.residual(torch.cat([grid, self.grid_projection(grid)], 1)) * valid
        return torch.sigmoid(self.logit_conv(grid) + residual).permute(0, 2, 3, 1).contiguous()


class UNetDeepLociGlobalQuery(UNetDeepLociDetailRoute):
    """Q-global: R1 with one geometry-masked global query per image."""

    def __init__(self, cell_size: int = 4, **kwargs):
        super().__init__(cell_size=cell_size, selection="attention", **kwargs)

    @staticmethod
    def global_grid(grid, valid_mask):
        coverage = F.avg_pool2d(valid_mask.float(), 4).to(grid)
        return (grid * coverage).sum((-2, -1), keepdim=True) / coverage.sum(
            (-2, -1), keepdim=True).clamp_min(1e-12)

    def _detail_summary(self, detail, grid, valid_mask):
        shared_grid = self.global_grid(grid, valid_mask).expand_as(grid)
        return super()._detail_summary(detail, shared_grid, valid_mask)
