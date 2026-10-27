"""PyTorch network building blocks for deepBlink.

This module mirrors the TensorFlow implementation in
`deepblink/networks/_networks.py` as closely as practical in PyTorch.

Notes
-----
- TensorFlow `Conv2D(..., padding="same", kernel_size=3)` is mapped to
  `nn.Conv2d(..., kernel_size=3, padding=1)`.
- TensorFlow `Activation(tf.nn.leaky_relu)` is mapped to `nn.LeakyReLU()`.
- TensorFlow SE block uses channel-last tensors; here we operate internally on
  channel-first tensors `(B, C, H, W)`.
- `l2_regularizer` / `opts_conv` compatibility arguments are retained in the
  signatures where relevant, but actual L2 regularization is expected to be
  handled by the optimizer via weight decay in PyTorch.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


REG = 1e-6
OPTIONS_CONV = {
    "kernel_size": 3,
    "padding": 1,
    "bias": True,
}


def _init_conv(conv: nn.Conv2d) -> None:
    """Match TensorFlow `he_normal` as closely as possible."""
    nn.init.kaiming_normal_(conv.weight, nonlinearity="leaky_relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def _make_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    padding: int = 1,
    bias: bool = True,
) -> nn.Conv2d:
    """Create and initialize one convolution layer."""
    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        bias=bias,
    )
    _init_conv(conv)
    return conv


class ConvBlock(nn.Module):
    """Convolutional block with repeated conv + LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        filters: int,
        n_convs: int = 2,
        opts_conv: Optional[dict] = None,
    ) -> None:
        super().__init__()
        opts = dict(OPTIONS_CONV if opts_conv is None else opts_conv)

        layers = []
        current_in = in_channels
        for _ in range(n_convs):
            conv = _make_conv(
                in_channels=current_in,
                out_channels=filters,
                kernel_size=opts.get("kernel_size", 3),
                padding=opts.get("padding", 1),
                bias=opts.get("bias", True),
            )
            layers.append(conv)
            layers.append(nn.LeakyReLU(inplace=True))
            current_in = filters

        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class InceptionBlock(nn.Module):
    """Inception block matching deepBlink TensorFlow behavior."""

    def __init__(
        self,
        in_channels: int,
        filters: int,
        efficient: bool = True,
        l2_regularizer: float = 1e-6,
    ) -> None:
        super().__init__()
        del l2_regularizer

        self.efficient = efficient
        self.activation = nn.LeakyReLU(inplace=True)

        self.conv1 = _make_conv(in_channels, filters, kernel_size=1, padding=0)

        if efficient:
            self.conv3_reduce = _make_conv(in_channels, filters, kernel_size=1, padding=0)
            self.conv3 = _make_conv(filters, filters, kernel_size=3, padding=1)

            self.conv5_reduce = _make_conv(in_channels, filters, kernel_size=1, padding=0)
            self.conv5 = _make_conv(filters, filters // 2, kernel_size=5, padding=2)

            self.pool_proj = _make_conv(in_channels, filters // 2, kernel_size=1, padding=0)
        else:
            self.conv3 = _make_conv(in_channels, filters, kernel_size=3, padding=1)
            self.conv5 = _make_conv(in_channels, filters // 2, kernel_size=5, padding=2)
            self.pool_proj = None

        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        conv1 = self.activation(self.conv1(inputs))

        if self.efficient:
            conv3 = self.activation(self.conv3_reduce(inputs))
            conv3 = self.activation(self.conv3(conv3))

            conv5 = self.activation(self.conv5_reduce(inputs))
            conv5 = self.activation(self.conv5(conv5))
        else:
            conv3 = self.activation(self.conv3(inputs))
            conv5 = self.activation(self.conv5(inputs))

        pool = self.pool(inputs)
        if self.efficient:
            pool = self.activation(self.pool_proj(pool))

        return torch.cat([conv1, conv3, conv5, pool], dim=1)


class UpConvBlock(nn.Module):
    """Upsampling block with skip concatenation."""

    def __init__(self) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(inputs)
        return torch.cat([skip, x], dim=1)


class ResidualBlock(nn.Module):
    """Simple residual block matching the TensorFlow implementation."""

    def __init__(
        self,
        in_channels: int,
        filters: int,
        opts_conv: Optional[dict] = None,
    ) -> None:
        super().__init__()
        opts = dict(OPTIONS_CONV if opts_conv is None else opts_conv)

        self.conv1 = _make_conv(
            in_channels, filters, opts.get("kernel_size", 3), opts.get("padding", 1), opts.get("bias", True)
        )
        self.conv2 = _make_conv(
            filters, filters, opts.get("kernel_size", 3), opts.get("padding", 1), opts.get("bias", True)
        )
        self.conv3 = _make_conv(
            filters, filters, opts.get("kernel_size", 3), opts.get("padding", 1), opts.get("bias", True)
        )
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(inputs))
        skip = x

        x = self.activation(self.conv2(x))
        x = self.conv3(x)

        x = x + skip
        x = self.activation(x)
        return x


class SqueezeBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, ratio: int = 8) -> None:
        super().__init__()
        hidden = max(channels // ratio, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.activation = nn.LeakyReLU(inplace=True)
        self.gate = nn.Sigmoid()

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="leaky_relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="sigmoid")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        x1 = self.pool(x).view(b, c)
        x1 = self.activation(self.fc1(x1))
        x1 = self.gate(self.fc2(x1)).view(b, c, 1, 1)
        return x * x1


def conv_block(
    in_channels: int,
    filters: int,
    n_convs: int = 2,
    opts_conv: Optional[dict] = None,
) -> nn.Module:
    """Factory matching the original function name."""
    return ConvBlock(
        in_channels=in_channels,
        filters=filters,
        n_convs=n_convs,
        opts_conv=opts_conv,
    )


def inception_block(
    in_channels: int,
    filters: int,
    efficient: bool = True,
    l2_regularizer: float = 1e-6,
) -> nn.Module:
    """Factory matching the original function name."""
    return InceptionBlock(
        in_channels=in_channels,
        filters=filters,
        efficient=efficient,
        l2_regularizer=l2_regularizer,
    )


def upconv_block() -> nn.Module:
    """Factory matching the original function name."""
    return UpConvBlock()


def residual_block(
    in_channels: int,
    filters: int,
    opts_conv: Optional[dict] = None,
) -> nn.Module:
    """Factory matching the original function name."""
    return ResidualBlock(
        in_channels=in_channels,
        filters=filters,
        opts_conv=opts_conv,
    )


def squeeze_block(channels: int, ratio: int = 8) -> nn.Module:
    """Factory matching the original function name."""
    return SqueezeBlock(channels=channels, ratio=ratio)
