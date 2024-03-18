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
        self.embd.append(nn.BatchNorm1d(64))
        self.embd.append(nn.Linear(64, 512))
        self.embd.append(nn.ReLU())

        self.maing = nn.Sequential()
        self.maing.append(nn.Linear(size, 2048))
        self.maing.append(nn.ReLU())
        self.maing.append(nn.Dropout(0.1))
        self.maing.append(nn.Linear(2048, 512))
        self.maing.append(nn.ReLU())
        
        self.mainp = nn.Sequential()
        self.mainp.append(nn.Linear(size, 2048))
        self.mainp.append(nn.ReLU())
        self.mainp.append(nn.Dropout(0.1))
        self.mainp.append(nn.Linear(2048, 512))
        self.mainp.append(nn.ReLU())

        self.exit = nn.Sequential()
        self.exit.append(nn.Linear(3584, 1024))
        self.exit.append(nn.ReLU())
        self.exit.append(nn.Dropout(0.1))
        self.exit.append(nn.Linear(1024, 1024))
        self.exit.append(nn.ReLU())
        self.exit.append(nn.Linear(1024, size))

    def forward(self, params, grad, x_embd):
        e = self.embd(x_embd).flatten()
        g = self.maing(grad)
        p = self.mainp(params)
        return self.exit(torch.cat([g, p, e]))