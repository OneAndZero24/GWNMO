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
        self.seq.append(nn.BatchNorm1d(12544))
        self.seq.append(nn.Linear(12544, 2048))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.BatchNorm1d(2048))
        self.seq.append(nn.Linear(2048, 512))
        self.seq.append(nn.ReLU())

    def forward(self, x: torch.Tensor):
        return self.seq(x)