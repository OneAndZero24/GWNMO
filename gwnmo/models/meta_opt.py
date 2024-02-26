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
        self.seq.append(nn.Linear(insize, 1024))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.Linear(1024, 4096))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.Linear(4096, 1024))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.Linear(1024, outsize))
        self.seq.append(nn.ReLU())

    def forward(self, params, grad, x_embd):
        x = torch.cat([params, grad, x_embd.flatten()])
        return self.seq(x)