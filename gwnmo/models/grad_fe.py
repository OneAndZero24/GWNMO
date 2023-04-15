from functools import reduce
from operator import mul

import torch
from torch import nn


class GradFeatEx(nn.Module):
    """Gradient feature extractor"""

    def __init__(self, grad_size: torch.Size, depth: int):
        super(GradFeatEx, self).__init__()
        self.seq = nn.Sequential()
        self._gen_arch(grad_size, depth)

    def _gen_arch(self, grad_size: torch.Size, depth: int):
        self.size: int = reduce(mul, grad_size)
        c_size = self.size
        for _ in range(depth):
            new_size: int = int(c_size/4)
            self.seq.append(nn.Linear(c_size, new_size))
            self.seq.append(nn.ReLU())
            c_size = new_size

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x)
        return self.seq(x)
