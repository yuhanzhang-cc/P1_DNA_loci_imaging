"""Full-resolution heatmap/offset reference; occupancy output is a logit."""
from __future__ import annotations

import torch
from torch import nn
from .unet import UNetDeepBlink


class UNetDeepLociHeatmap(UNetDeepBlink):
    output_representation = "dense_heatmap_offset_logits"

    def __init__(self, cell_size: int = 1, **kwargs):
        if cell_size != 1:
            raise ValueError("Dense heatmap uses one-pixel anchors.")
        super().__init__(cell_size=4, **kwargs)
        del self.encoders_v2, self.logit_block, self.logit_conv
        self.cell_size = 1
        self.dense_head = nn.Conv2d(self.first_decoder_channels, 3, 1)
        nn.init.kaiming_normal_(self.dense_head.weight, nonlinearity="sigmoid")
        nn.init.zeros_(self.dense_head.bias)

    def forward(self, inputs):
        full, _ = self._forward_first_decoder(inputs)
        logits = self.dense_head(full)
        output = torch.cat([logits[:, :1], torch.sigmoid(logits[:, 1:])], dim=1)
        return output.permute(0, 2, 3, 1).contiguous()
