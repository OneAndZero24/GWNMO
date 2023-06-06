import torch
from torch import nn


class MetaOptimizer(nn.Module):
    """
    Gradient weighting network in `GWNMO`
    """

    def __init__(self):
        super(MetaOptimizer, self).__init__()
        self.seq = nn.Sequential()
        self.seq.append(nn.Linear(26644, 128))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.Linear(128, 32))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.Linear(32, 5130))
        self.seq.append(nn.ReLU())

    def forward(self, params, grad, x_embd):
        x = torch.cat([params, grad, x_embd.flatten()])
        return self.seq(x)