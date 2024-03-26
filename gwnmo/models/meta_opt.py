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
        self.embd.append(nn.Linear(512, 64))
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
        self.exit.append(nn.Linear(1344, 1024))
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
    

class AttentionMetaOptimizer(nn.Module):
    """
    Gradient weighting network in `GWNMO`
    """

    def __init__(self, size=CLASSIC_10CLASS_WEIGHTS_SIZE):
        """
        insize - concat(param_vals, grad, x_embd)
        outsize - grad
        """

        super(AttentionMetaOptimizer, self).__init__()
        self.attng = nn.MultiheadAttention(1024, 8)
        self.attnp = nn.MultiheadAttention(1024, 8)

        self.embd = nn.Sequential()
        self.embd.append(nn.BatchNorm1d(1024))
        self.embd.append(nn.Linear(1024, 256))
        self.embd.append(nn.ReLU())
        self.embd.append(nn.BatchNorm1d(256))
        self.embd.append(nn.Linear(256, 8))
        self.embd.append(nn.ReLU())

        self.q = nn.Sequential()

        self.k = nn.Sequential()

        self.exit = nn.Sequential()
        self.exit.append(nn.Linear((1036*24)+2*10+8*5, 4096))
        self.exit.append(nn.ReLU())
        self.exit.append(nn.Linear(4096, 2048))
        self.exit.append(nn.ReLU())
        self.exit.append(nn.Linear(2048, 1024))
        self.exit.append(nn.ReLU())
        self.exit.append(nn.Linear(1024, size))

    def forward(self, params, grad, x_embd, l_grad, l_data):
        # [12, 1024]
        values = grad
        x_embd = self.embd(x_embd).flatten()
        tmp = torch.Tensor([in list(zip(params, grad))]) # TODO networks for q, k (what to choose for q, k)
        xg, wg = self.attng(grad, grad, grad) # TODO pass one by one dont flatten before exit FC
        xp, wp = self.attnp(params, params, params)
        # TODO transpose w after attn
        g = torch.cat([xg.T, wg.T]).flatten()
        p = torch.cat([xp.T, wp.T]).flatten()
        return self.exit(torch.cat([g, p, x_embd, l_grad, l_data]))