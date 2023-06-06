import torch
from torch import nn
import torch.nn.functional as F


class Target(nn.Module):
    """
    Target network to be trained.
    Works on data processed by feature extractor (output of `FeatEx`)
    """

    def __init__(self):
        super(Target, self).__init__()

        self.seq = nn.Sequential()
        self.seq.append(nn.Linear(512, 10))

    def forward(self, x: torch.Tensor):
        x = self.seq(x)
        return F.log_softmax(x, dim=1)