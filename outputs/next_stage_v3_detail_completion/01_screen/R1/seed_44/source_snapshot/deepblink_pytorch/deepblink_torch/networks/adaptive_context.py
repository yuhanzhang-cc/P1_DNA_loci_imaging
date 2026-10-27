"""Coverage-aware context scale selection and fixed-fusion control."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .unet import UNetDeepBlink


class ExcludedCenterContext(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5**.5)
        mask = torch.ones(1, 1, kernel_size, kernel_size)
        center = kernel_size // 2
        mask[..., center-1:center+2, center-1:center+2] = 0
        self.register_buffer("support", mask)
        self.pointwise = nn.Conv2d(channels, channels, 1)

    def forward(self, feature, valid):
        padding = self.kernel_size // 2
        coverage = F.conv2d(valid.float(), self.support.float(), padding=padding) / self.support.sum()
        result = F.conv2d(feature * valid, self.weight * self.support,
                          padding=padding, groups=feature.shape[1])
        result = result / coverage.to(result.dtype).clamp_min(1e-6)
        result = F.leaky_relu(self.pointwise(result), .01) * (coverage > 0)
        return result, coverage


class UNetDeepLociAdaptiveContext(UNetDeepBlink):
    requires_valid_mask = True

    def __init__(self, cell_size: int = 4, adaptive: bool = True, **kwargs):
        if cell_size != 4:
            raise ValueError("Registered context model uses grid 4.")
        super().__init__(cell_size=cell_size, **kwargs)
        self.adaptive = adaptive
        self.context_projection = nn.Conv2d(self.logit_block.out_channels, 16, 1)
        self.contexts = nn.ModuleList([ExcludedCenterContext(16, k) for k in (5, 9, 17)])
        selector_inputs = self.logit_block.out_channels + 16*3 + 3
        # Both alternatives have exactly the same number of active parameters.
        self.selector_or_projection = nn.Sequential(nn.Conv2d(selector_inputs, 16, 1), nn.ReLU(), nn.Conv2d(16, 3, 1))
        self.residual = nn.Sequential(nn.Conv2d(self.logit_block.out_channels + 16 + 3, 32, 1),
                                      nn.ReLU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, inputs, valid_mask=None):
        if valid_mask is None:
            raise ValueError("Context model requires an image-geometry mask.")
        grid = self._forward_features(inputs)
        valid = F.interpolate(valid_mask.float(), size=grid.shape[-2:], mode="area")
        feature = self.context_projection(grid)
        outputs = [layer(feature, valid) for layer in self.contexts]
        contexts = torch.stack([x[0] for x in outputs], dim=1)
        coverage = torch.cat([x[1] for x in outputs], dim=1).to(grid.dtype)
        selection_input = torch.cat([grid, *[x[0] for x in outputs], coverage], dim=1)
        mapped = self.selector_or_projection(selection_input)
        if self.adaptive:
            weights = torch.softmax(mapped.float(), 1).to(grid.dtype)
            fused = (contexts * weights.unsqueeze(2)).sum(1)
        else:
            # Equal scale weights, with the same active projection capacity used
            # as an ordinary post-fusion feature correction (no scale choice).
            feature_groups = torch.arange(16, device=mapped.device) % 3
            fused = contexts.mean(1) + mapped[:, feature_groups]
        raw = self.logit_conv(grid)
        delta = 2 * torch.tanh(self.residual(torch.cat([grid, fused, coverage], 1))) * (valid > 0)
        result = torch.cat([torch.sigmoid(raw[:, :1] + delta), torch.sigmoid(raw[:, 1:])], 1)
        return result.permute(0, 2, 3, 1).contiguous()
