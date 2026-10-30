"""A: diagnostic geometry-aware SE pooling; no novelty claim."""
import torch
from torch import nn
from torch.nn import functional as F
from ._networks import SqueezeBlock
from .unet import UNetDeepBlink


class GeometricSqueezeBlock(nn.Module):
    """Reuse all parameters and their names without consuming extra RNG."""
    def __init__(self, source):
        super().__init__()
        self.pool, self.fc1, self.fc2 = source.pool, source.fc1, source.fc2
        self.activation, self.gate = source.activation, source.gate
        self.valid_mask = None

    def weights_for(self, features, valid_mask):
        mask = F.interpolate(valid_mask.float(), size=features.shape[-2:], mode="area")
        pooled = (features*mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1.)
        return self.gate(self.fc2(self.activation(self.fc1(pooled))))[..., None, None]

    def forward(self, features):
        if self.valid_mask is None:
            raise ValueError("Masked SE requires an explicit geometric mask.")
        return features * self.weights_for(features, self.valid_mask)


class UNetDeepBlinkMaskedSE(UNetDeepBlink):
    requires_valid_mask = True

    def __init__(self, cell_size=4, **kwargs):
        super().__init__(cell_size=cell_size, **kwargs)
        for name, module in list(self.named_modules()):
            if isinstance(module, SqueezeBlock):
                parent_name, child_name = name.rsplit(".", 1)
                setattr(self.get_submodule(parent_name), child_name, GeometricSqueezeBlock(module))

    def forward(self, inputs, valid_mask=None):
        if valid_mask is None or valid_mask.shape != inputs.shape:
            raise ValueError("Masked model requires a valid_mask matching inputs.")
        blocks = [module for module in self.modules() if isinstance(module, GeometricSqueezeBlock)]
        try:
            for block in blocks:
                block.valid_mask = valid_mask
            return super().forward(inputs)
        finally:
            # Do not retain previous geometry or expose it in the checkpoint.
            for block in blocks:
                block.valid_mask = None
