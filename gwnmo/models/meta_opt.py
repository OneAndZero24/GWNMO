import torch
from torch import nn

CLASSIC_10CLASS_WEIGHTS_SIZE = 33482

class MetaOptimizer(nn.Module):
    """
    Gradient weighting network in `GWNMO`
    """

    def __init__(self, size=CLASSIC_10CLASS_WEIGHTS_SIZE):
        """
        insize - concat(param_vals, grad, x_embd)
        outsize - grad
        """

        super(MetaOptimizer, self).__init__()
        self.embd = nn.Sequential()
        self.embd.append(nn.BatchNorm1d(128))
        self.embd.append(nn.Linear(128, 64))
        self.embd.append(nn.ReLU())

        self.maing = nn.Sequential()
        self.maing.append(nn.Linear(size, 1024))
        self.maing.append(nn.ReLU())
        self.maing.append(nn.Linear(1024, 512))
        self.maing.append(nn.ReLU())
        
        self.mainp = nn.Sequential()
        self.mainp.append(nn.Linear(size, 1024))
        self.mainp.append(nn.ReLU())
        self.mainp.append(nn.Linear(1024, 512))
        self.mainp.append(nn.ReLU())

        self.exit = nn.Sequential()
        self.exit.append(nn.Linear(1344, 512))
        self.exit.append(nn.ReLU())
        self.exit.append(nn.Linear(512, 2048))
        self.exit.append(nn.ReLU())
        self.exit.append(nn.Linear(2048, size))

    def forward(self, params, grad, x_embd):
        e = self.embd(x_embd).flatten()
        g = self.maing(grad)
        p = self.mainp(params)
        return self.exit(torch.cat([g, p, e]))