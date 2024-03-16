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
        self.embd.append(nn.BatchNorm1d(512))
        self.embd.append(nn.Linear(512, 1024))
        self.embd.append(nn.ReLU())
        self.embd.append(nn.Dropout(0.1))
        self.embd.append(nn.BatchNorm1d(1024))
        self.embd.append(nn.Linear(1024, 256))
        self.embd.append(nn.ReLU())

        self.main = nn.Sequential()
        self.main.append(nn.Linear(size*2, 2048))
        self.main.append(nn.ReLU())
        self.main.append(nn.Dropout(0.1))
        self.main.append(nn.Linear(2048, 1024))
        self.main.append(nn.ReLU())
        self.main.append(nn.Dropout(0.1))
        self.main.append(nn.Linear(1024, 2048))
        self.main.append(nn.ReLU())
        
        self.exit = nn.Sequential()
        self.exit.append(nn.Linear(3328, 8128))
        self.exit.append(nn.ReLU())
        self.exit.append(nn.Dropout(0.1))
        self.exit.append(nn.Linear(8128, 8128))
        self.exit.append(nn.ReLU())
        self.exit.append(nn.Dropout(0.1))
        self.exit.append(nn.Linear(8128, size))

    def forward(self, params, grad, x_embd):
        e = self.embd(x_embd).flatten()
        x = self.main(torch.cat([params, grad]))
        return self.exit(torch.cat([x, e]))