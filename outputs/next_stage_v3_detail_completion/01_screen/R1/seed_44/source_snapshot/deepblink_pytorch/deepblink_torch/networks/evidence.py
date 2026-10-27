"""Experimental point-position-conditioned image evidence, not a validated method.

Geometry masks describe real pixels only. No background segmentation labels,
PSF calibration, or truth positions are needed at inference.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .unet import UNetDeepBlink


class RadialEvidence(nn.Module):
    """Read center/ring evidence at predicted subpixel positions.

    Three rings (1.5, 3, 6 original pixels), eight samples each, yield contrast,
    normalized contrast, angular variation, and valid coverage (12 channels).
    Bilinear sampling is differentiable with respect to the predicted offsets.
    """
    def __init__(self, cell_size=4):
        super().__init__()
        self.cell_size = cell_size
        directions = [(math.sin(i*math.pi/4), math.cos(i*math.pi/4)) for i in range(8)]
        points = [(0., 0.)] + [(r*y, r*x) for r in (1.5, 3., 6.) for y, x in directions]
        self.register_buffer("sampling_offsets_yx", torch.tensor(points, dtype=torch.float32))

    def forward(self, image, offsets, valid_mask):
        if valid_mask.shape != image.shape:
            raise ValueError("valid_mask must have image shape (B,1,H,W).")
        batch, _, height, width = image.shape
        _, _, gh, gw = offsets.shape
        with torch.autocast(device_type=image.device.type, enabled=False):
            image, offsets, valid_mask = image.float(), offsets.float(), valid_mask.float()
            yy, xx = torch.meshgrid(torch.arange(gh, device=image.device),
                                    torch.arange(gw, device=image.device), indexing="ij")
            cell_origin = torch.stack([yy, xx], -1).float() * self.cell_size
            center = cell_origin[None] + offsets.permute(0, 2, 3, 1) * self.cell_size
            positions = center[..., None, :] + self.sampling_offsets_yx
            grid_y = (positions[..., 0] + .5) * (2. / height) - 1
            grid_x = (positions[..., 1] + .5) * (2. / width) - 1
            grid = torch.stack([grid_x, grid_y], -1).reshape(batch, gh, gw*25, 2)
            sampled = F.grid_sample(torch.cat([image * valid_mask, valid_mask], 1), grid,
                                    mode="bilinear", padding_mode="zeros", align_corners=False)
            sampled = sampled.reshape(batch, 2, gh, gw, 25)
            coverage = sampled[:, 1].clamp(0., 1.)
            values = sampled[:, 0] / coverage.clamp_min(1e-6)
            center_value = values[..., 0:1]
            ring_values = values[..., 1:].reshape(batch, gh, gw, 3, 8)
            ring_weights = coverage[..., 1:].reshape(batch, gh, gw, 3, 8)
            weight_sum = ring_weights.sum(-1).clamp_min(1e-6)
            mean = (ring_values * ring_weights).sum(-1) / weight_sum
            variance = ((ring_values-mean[..., None]).square() * ring_weights).sum(-1) / weight_sum
            std = (variance + 1e-4).sqrt()
            contrast = center_value - mean
            # Bounded evidence protects training from near-uniform patches.
            evidence = torch.cat([contrast.clamp(-10, 10),
                                  (contrast / std.clamp_min(.1)).clamp(-10, 10),
                                  std.clamp(max=10), ring_weights.mean(-1)], -1)
            center_valid = (coverage[..., :1] > 1e-6).float()
            return (evidence * center_valid).permute(0, 3, 1, 2).contiguous()


class UNetDeepLociEvidence(UNetDeepBlink):
    """Keep the backbone; correct occupancy using offset-conditioned evidence.

    Zero-initialized residual starts from the exact inherited head. No new
    objective is added. The sampling geometry is fixed, so an unconstrained
    learned 'background image' cannot copy the input to satisfy a loss.
    """
    requires_valid_mask = True

    def __init__(self, cell_size=4, **kwargs):
        super().__init__(cell_size=cell_size, **kwargs)
        channels = self.logit_block.out_channels
        self.evidence_reader = RadialEvidence(cell_size)
        self.feature_projection = nn.Sequential(nn.Conv2d(channels, 16, 1), nn.LeakyReLU())
        self.evidence_fusion = nn.Sequential(nn.Conv2d(28, 32, 1), nn.LeakyReLU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.evidence_fusion[-1].weight)
        nn.init.zeros_(self.evidence_fusion[-1].bias)

    def forward(self, inputs, valid_mask=None):
        if valid_mask is None:
            raise ValueError("Evidence model requires an explicit geometric valid_mask.")
        features = self._forward_features(inputs)
        logits = self.logit_conv(features)
        offsets = torch.sigmoid(logits[:, 1:])
        evidence = self.evidence_reader(inputs, offsets, valid_mask)
        fused = torch.cat([self.feature_projection(features), evidence.to(features.dtype)], 1)
        correction = self.evidence_fusion(fused)
        occupancy = torch.sigmoid(logits[:, :1] + correction)
        return torch.cat([occupancy, offsets], 1).permute(0, 2, 3, 1).contiguous()


class UNetDeepLociLocal(UNetDeepBlink):
    """Simple 3x3 occupancy control; offsets keep the original features."""
    def __init__(self, cell_size=4, **kwargs):
        super().__init__(cell_size=cell_size, **kwargs)
        channels = self.logit_block.out_channels
        del self.logit_conv
        self.occupancy_head = nn.Sequential(nn.Conv2d(channels, 16, 3, padding=1),
                                            nn.ReLU(), nn.Conv2d(16, 1, 1))
        self.offset_head = nn.Conv2d(channels, 2, 1)
        for module in self.modules():
            if module in (self.occupancy_head[0], self.occupancy_head[2], self.offset_head):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu" if module is self.occupancy_head[0] else "sigmoid")
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        features = self._forward_features(inputs)
        output = torch.cat([torch.sigmoid(self.occupancy_head(features)),
                            torch.sigmoid(self.offset_head(features))], 1)
        return output.permute(0, 2, 3, 1).contiguous()
