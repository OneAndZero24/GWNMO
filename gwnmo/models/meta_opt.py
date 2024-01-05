import torch
from torch import nn

CLASSIC_10CLASS_WEIGHTS_SIZE = 33482
CLASSIC_10CLASS_IN_SIZE = 83348

class MetaOptimizer(nn.Module):
    """
    Gradient weighting network in `GWNMO`
    """

    def __init__(self, insize=CLASSIC_10CLASS_IN_SIZE, outsize=CLASSIC_10CLASS_WEIGHTS_SIZE):
        """
        insize - concat(param_vals, grad, x_embd)
        outsize - grad
        """

        super(MetaOptimizer, self).__init__()
        self.seq = nn.Sequential()
        self.seq.append(nn.BatchNorm1d(insize))
        self.seq.append(nn.Linear(insize, 128))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.BatchNorm1d(128))
        self.seq.append(nn.Linear(128, 32))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.BatchNorm1d(32))
        self.seq.append(nn.Linear(32, outsize))
        self.seq.append(nn.ReLU())

    def forward(self, params, grad, x_embd):
        x = torch.cat([params, grad, x_embd.flatten()]).reshape([x.shape, 1])
        return self.seq(x)