import torch
from torch import nn
from torch.nn import functional as F


class SimpleCNN(nn.Module):
    """Simple CNN for image feature extraction"""

    def __init__(self, in_channels: int, depth: int):
        super(SimpleCNN, self).__init__()
        self.seq = nn.Sequential()
        self._gen_arch(in_channels, depth)

    def _gen_arch(self, in_channels: int, depth: int):
        channels: int = in_channels
        mul: int = 4**depth
        for _ in range(depth):
            new_channels: int = channels*mul
            self.seq.append(nn.Conv2d(channels, new_channels, 3, 1))

            mul = min(1, int(mul/4))
            channels = new_channels

    def forward(self, x: torch.Tensor):
        x = self.seq(x)
        x = F.max_pool2d(x, 2)
        return torch.flatten(x)
