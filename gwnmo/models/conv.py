"""
Based on: https://pytorch.org/blog/FX-feature-extraction-torchvision/
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    Applies `num_layers` 3x3 convolutions each followed by ReLU then downsamples
    via 2x2 max pool.
    """

    def __init__(self, num_layers, in_channels, out_channels):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else out_channels,
                          out_channels, 3, padding=1),
                nn.ReLU()
            )
            for i in range(num_layers)
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        return x


class CNNFeatEx(nn.Module):
    """
    Applies several ConvBlocks each doubling the number of channels, and
    halving the feature map size, before taking a global average and classifying.
    """

    def __init__(self, in_channels, num_blocks):
        super().__init__()
        first_channels = 64
        self.blocks = nn.ModuleList([
            ConvBlock(2 if i == 0 else 3,
                      in_channels=(in_channels if i == 0 else first_channels*(2**(i-1))),
                      out_channels=first_channels*(2**i))
            for i in range(num_blocks)
        ])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        return x
