import torch
from torch import nn
import torch.nn.functional as F


class Body(nn.Module):
    """
    Specifically for `GWNMOFS`.  

    Works on data processed by feature extractor. 
    """

    def __init__(self):
        super(Body, self).__init__()

        self.seq = nn.Sequential()
        self.seq.append(nn.BatchNorm1d(512))
        self.seq.append(nn.Linear(512, 1024))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.BatchNorm1d(1024))
        self.seq.append(nn.Linear(1024, 8192))
        self.seq.append(nn.ReLU())

    def forward(self, x: torch.Tensor):
        return self.seq(x)