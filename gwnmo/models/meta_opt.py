import torch
from torch import nn

CLASSIC_10CLASS_WEIGHTS_SIZE = 33482

class FSMetaOptimizer(nn.Module):
    """
    Gradient weighting network in `GWNMOFS`
    """

    def __init__(self, size=CLASSIC_10CLASS_WEIGHTS_SIZE):
        """
        insize - concat(param_vals, grad, x_embd)
        outsize - grad
        """

        super(FSMetaOptimizer, self).__init__()
        self.embd = nn.Sequential()
        self.embd.append(nn.BatchNorm1d(64))
        self.embd.append(nn.Linear(64, 8))
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
        self.exit.append(nn.Linear(1024+5*8, 512))
        self.exit.append(nn.ReLU())
        self.exit.append(nn.Linear(512, 1024))
        self.exit.append(nn.ReLU())
        self.exit.append(nn.Linear(1024, size))

    def forward(self, params, grad, x_embd):
        e = self.embd(x_embd).flatten()
        g = self.maing(grad)
        p = self.mainp(params)
        return self.exit(torch.cat([g, p, e]))
    

class MetaOptimizer(nn.Module):
    """
    Gradient weighting network in `GWNMO`
    """
    def __init__(self):
        super(MetaOptimizer, self).__init__()
        self.seq = nn.Sequential()
        self.seq.append(nn.Linear(83348, 128))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.Linear(128, 32))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.Linear(32, 33482))
        self.seq.append(nn.ReLU())

    def forward(self, params, grad, x_embd):
        x = torch.cat([params, grad, x_embd.flatten()])
        return self.seq(x)