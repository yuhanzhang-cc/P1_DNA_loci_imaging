"""PyTorch implementation of the deepBlink U-Net variant.

This mirrors `deepblink/networks/unet.py` while using PyTorch internals.
Internally, features are channel-first `(B, C, H, W)`, but the public output is
returned as channel-last `(B, H_grid, W_grid, 3)` to preserve deepBlink's
original label/layout convention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ._networks import OPTIONS_CONV
from ._networks import conv_block
from ._networks import inception_block
from ._networks import residual_block
from ._networks import squeeze_block
from ._networks import upconv_block


class _DeepBlinkBlock(nn.Module):
    """One deepBlink block: chosen backbone block followed by SE."""

    def __init__(
        self,
        in_channels: int,
        filters: int,
        block: str,
        l2: float,
    ) -> None:
        super().__init__()

        opts_conv = dict(OPTIONS_CONV)

        if block == "convolutional":
            self.core = conv_block(
                in_channels=in_channels,
                filters=filters,
                n_convs=3,
                opts_conv=opts_conv,
            )
            out_channels = filters
        elif block == "inception":
            self.core = inception_block(
                in_channels=in_channels,
                filters=filters,
                l2_regularizer=l2,
            )
            out_channels = filters + filters + (filters // 2) + (filters // 2)
        elif block == "residual":
            self.core = residual_block(
                in_channels=in_channels,
                filters=filters,
                opts_conv=opts_conv,
            )
            out_channels = filters
        else:
            raise ValueError(
                "block must be one of {'convolutional', 'inception', 'residual'}, "
                f"but got {block!r}."
            )

        self.squeeze = squeeze_block(channels=out_channels)
        self.out_channels = out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.core(inputs)
        x = self.squeeze(x)
        return x


class _Encoder(nn.Module):
    """deepBlink encoder block: block -> Dropout2d -> MaxPool2d."""

    def __init__(
        self,
        in_channels: int,
        filters: int,
        block: str,
        l2: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block = _DeepBlinkBlock(
            in_channels=in_channels,
            filters=filters,
            block=block,
            l2=l2,
        )
        self.dropout = nn.Dropout2d(p=dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    @property
    def out_channels(self) -> int:
        return self.block.out_channels

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.block(inputs)
        skip = self.dropout(x)
        x = self.pool(skip)
        return x, skip


class _Decoder(nn.Module):
    """deepBlink decoder block: block -> upsample -> concat skip."""

    def __init__(
        self,
        in_channels: int,
        filters: int,
        block: str,
        l2: float,
    ) -> None:
        super().__init__()
        self.block = _DeepBlinkBlock(
            in_channels=in_channels,
            filters=filters,
            block=block,
            l2=l2,
        )
        self.up = upconv_block()

    @property
    def out_channels(self) -> int:
        return self.block.out_channels

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(inputs)
        x = self.up(x, skip)
        return x


class UNetDeepBlink(nn.Module):
    """deepBlink U-Net model.

    Parameters
    ----------
    dropout:
        Percentage of dropout before each max-pooling step.
    cell_size:
        Size of one cell in the prediction matrix. Must be a power of 2.
    filters:
        Log2 number of filters in the first block.
    ndown:
        Number of downsampling steps in the first encoder / decoder.
    l2:
        Kept for API compatibility with TensorFlow deepBlink. In PyTorch this is
        typically handled as optimizer weight decay.
    block:
        Type of block to use. One of:
        - ``'convolutional'``
        - ``'inception'``
        - ``'residual'``

    Returns
    -------
    torch.Tensor
        Channel-last prediction tensor of shape ``(B, H_grid, W_grid, 3)``.
    """

    def __init__(
        self,
        dropout: float = 0.3,
        cell_size: int = 4,
        filters: int = 5,
        ndown: int = 2,
        l2: float = 1e-6,
        block: str = "convolutional",
    ) -> None:
        super().__init__()

        if not math.log(cell_size, 2).is_integer():
            raise ValueError(f"cell_size must be a power of 2, but is {cell_size}.")

        self.dropout = dropout
        self.cell_size = cell_size
        self.filters = filters
        self.ndown = ndown
        self.l2 = l2
        self.block_name = block

        self.input_channels = 1

        encoder_modules = []
        skip_channels = []
        current_channels = self.input_channels

        for n in range(ndown):
            block_filters = 2 ** (filters + n)
            encoder = _Encoder(
                in_channels=current_channels,
                filters=block_filters,
                block=block,
                l2=l2,
                dropout=dropout,
            )
            encoder_modules.append(encoder)
            current_channels = encoder.out_channels
            skip_channels.append(encoder.out_channels)

        self.encoders = nn.ModuleList(encoder_modules)
        self.skip_bottom_channels = current_channels

        decoder_modules = []
        for n, skip_ch in enumerate(reversed(skip_channels)):
            block_filters = 2 ** (filters + (ndown - n))
            decoder = _Decoder(
                in_channels=current_channels,
                filters=block_filters,
                block=block,
                l2=l2,
            )
            decoder_modules.append(decoder)
            current_channels = decoder.out_channels + skip_ch

        self.decoders = nn.ModuleList(decoder_modules)

        encoder_v2_modules = []
        ndown_cell = int(math.log(cell_size, 2))
        self.ndown_cell = ndown_cell

        for n in range(ndown_cell):
            block_filters = 2 ** (filters + n)
            encoder = _Encoder(
                in_channels=current_channels,
                filters=block_filters,
                block=block,
                l2=l2,
                dropout=dropout,
            )
            encoder_v2_modules.append(encoder)
            current_channels = encoder.out_channels

        self.encoders_v2 = nn.ModuleList(encoder_v2_modules)

        if ndown == 2 and cell_size == 4:
            current_channels = current_channels + self.skip_bottom_channels

        self.logit_block = _DeepBlinkBlock(
            in_channels=current_channels,
            filters=2 ** (filters + ndown_cell),
            block=block,
            l2=l2,
        )

        self.logit_conv = nn.Conv2d(
            in_channels=self.logit_block.out_channels,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        nn.init.kaiming_normal_(self.logit_conv.weight, nonlinearity="sigmoid")
        if self.logit_conv.bias is not None:
            nn.init.zeros_(self.logit_conv.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Parameters
        ----------
        inputs:
            Input tensor with shape ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Prediction tensor with shape ``(B, H_grid, W_grid, 3)``.
        """
        x = inputs
        skip_layers = []

        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_layers.append(skip)
        skip_bottom = x

        for decoder, skip in zip(self.decoders, reversed(skip_layers)):
            x = decoder(x, skip)

        for encoder in self.encoders_v2:
            x, _ = encoder(x)

        if self.ndown == 2 and self.cell_size == 4:
            x = torch.cat([skip_bottom, x], dim=1)

        x = self.logit_block(x)
        x = self.logit_conv(x)
        x = torch.sigmoid(x)

        return x.permute(0, 2, 3, 1).contiguous()
