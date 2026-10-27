"""D2 hypothesis: local feature contrast instead of fixed intensity evidence."""
import torch
from torch import nn
from torch.nn import functional as F
from .evidence import RadialEvidence, UNetDeepLociEvidence


class LearnedRadialEvidence(RadialEvidence):
    def __init__(self, cell_size=4):
        super().__init__(cell_size)
        self.embedding = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.LeakyReLU(),
                                       nn.Conv2d(8, 8, 3, padding=1), nn.LeakyReLU())

    def read_features(self, fields, offsets, valid_mask):
        batch, channels, height, width = fields.shape
        _, _, gh, gw = offsets.shape
        if valid_mask.shape != (batch, 1, height, width):
            raise ValueError("Expected a geometric mask for the feature field.")
        with torch.autocast(device_type=fields.device.type, enabled=False):
            fields, offsets, valid_mask = fields.float(), offsets.float(), valid_mask.float()
            yy, xx = torch.meshgrid(torch.arange(gh, device=fields.device),
                                    torch.arange(gw, device=fields.device), indexing="ij")
            origin = torch.stack([yy, xx], -1).float() * self.cell_size
            center = origin[None] + offsets.permute(0, 2, 3, 1) * self.cell_size
            positions = center[..., None, :] + self.sampling_offsets_yx
            grid_y = (positions[..., 0] + .5) * (2. / height) - 1
            grid_x = (positions[..., 1] + .5) * (2. / width) - 1
            grid = torch.stack([grid_x, grid_y], -1).reshape(batch, gh, gw*25, 2)
            sampled = F.grid_sample(torch.cat([fields*valid_mask, valid_mask], 1), grid,
                                    mode="bilinear", padding_mode="zeros", align_corners=False)
            sampled = sampled.reshape(batch, channels+1, gh, gw, 25)
            coverage = sampled[:, -1:].clamp(0., 1.)
            values = sampled[:, :-1] / coverage.clamp_min(1e-6)
            center_values = values[..., :1]
            rings = values[..., 1:].reshape(batch, channels, gh, gw, 3, 8)
            weights = coverage[..., 1:].reshape(batch, 1, gh, gw, 3, 8)
            mean = (rings*weights).sum(-1) / weights.sum(-1).clamp_min(1e-6)
            variance = ((rings-mean[..., None]).square()*weights).sum(-1) / weights.sum(-1).clamp_min(1e-6)
            contrasts = (center_values-mean).clamp(-10, 10)
            variation = (variance+1e-4).sqrt().clamp(max=10)
            # 8 channels x 3 rings for each term, plus 3 geometric coverages.
            contrasts = contrasts.permute(0, 1, 4, 2, 3).reshape(batch, channels*3, gh, gw)
            variation = variation.permute(0, 1, 4, 2, 3).reshape(batch, channels*3, gh, gw)
            coverages = weights.mean(-1)[:, 0].permute(0, 3, 1, 2)
            evidence = torch.cat([contrasts, variation, coverages], 1)
            return evidence * (coverage[..., 0] > 1e-6).float()

    def forward(self, image, offsets, valid_mask):
        return self.read_features(self.embedding(image), offsets, valid_mask)


class UNetDeepLociFeatureEvidence(UNetDeepLociEvidence):
    """Same offset conditioning and main objective, learned evidence field."""
    def __init__(self, cell_size=4, **kwargs):
        super().__init__(cell_size=cell_size, **kwargs)
        self.evidence_reader = LearnedRadialEvidence(cell_size)
        self.evidence_fusion = nn.Sequential(nn.Conv2d(67, 32, 1), nn.LeakyReLU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.evidence_fusion[-1].weight)
        nn.init.zeros_(self.evidence_fusion[-1].bias)
