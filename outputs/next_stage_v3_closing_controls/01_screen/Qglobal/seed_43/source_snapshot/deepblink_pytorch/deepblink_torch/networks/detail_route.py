"""Round-two detail route and its position-preserving projection control."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from .unet import UNetDeepBlink


def phase_features(tensor: torch.Tensor, cell_size: int = 4) -> torch.Tensor:
    """Return B, phase(row-major), channel, grid_y, grid_x without pooling."""
    packed = F.pixel_unshuffle(tensor, cell_size)
    b, _, h, w = packed.shape
    return packed.reshape(b, tensor.shape[1], cell_size**2, h, w).transpose(1, 2)


class UNetDeepLociDetailRoute(UNetDeepBlink):
    requires_valid_mask = True

    def __init__(self, cell_size: int = 4, selection: str = "attention", **kwargs):
        if cell_size != 4 or selection not in ("attention", "projection"):
            raise ValueError("Detail route requires grid 4 and a registered selection.")
        super().__init__(cell_size=cell_size, **kwargs)
        self.selection = selection
        self.detail_projection = nn.Conv2d(self.first_decoder_channels, 8, 1)
        if selection == "attention":
            self.query = nn.Conv2d(self.logit_block.out_channels, 8, 1)
            self.key = nn.Conv2d(8, 8, 1)
            self.value = nn.Conv2d(8, 8, 1)
            self.position = nn.Parameter(torch.zeros(16, 8))
        else:
            self.plain_projection = nn.Conv2d(16 * 8, 10, 1)
        self.residual = nn.Sequential(
            nn.Conv2d(self.logit_block.out_channels + 10, 32, 1),
            nn.ReLU(), nn.Conv2d(32, 3, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        self.register_buffer("phase_centers", torch.stack([yy.flatten(), xx.flatten()], 1).float().add(.5).div(4), persistent=False)

    def _detail_summary(self, detail, grid, valid_mask):
        valid = phase_features(valid_mask.float())[:, :, 0] > 0.5
        if self.selection == "projection":
            packed = F.pixel_unshuffle(detail * valid_mask, 4)
            return self.plain_projection(packed), valid.any(1, keepdim=True)
        keys = phase_features(self.key(detail))
        values = phase_features(self.value(detail))
        query = self.query(grid).unsqueeze(1)
        logits = ((keys + self.position[None, :, :, None, None]) * query).sum(2) / math.sqrt(8)
        weights = torch.softmax(logits.float().masked_fill(~valid, -1e4), dim=1) * valid
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
        spatial = self.phase_centers[None, :, :, None, None].to(values)
        summary = torch.cat([(weights.unsqueeze(2) * values).sum(1),
                             (weights.unsqueeze(2) * spatial).sum(1)], dim=1)
        return summary.to(grid.dtype), valid.any(1, keepdim=True)

    def forward(self, inputs: torch.Tensor, valid_mask: torch.Tensor | None = None):
        if valid_mask is None:
            raise ValueError("Detail route requires an image-geometry mask.")
        full, bottom = self._forward_first_decoder(inputs)
        grid = self._forward_grid_features(full, bottom)
        detail = self.detail_projection(full)
        summary, valid = self._detail_summary(detail, grid, valid_mask)
        residual = self.residual(torch.cat([grid, summary], 1)) * valid
        result = torch.sigmoid(self.logit_conv(grid) + residual)
        return result.permute(0, 2, 3, 1).contiguous()
