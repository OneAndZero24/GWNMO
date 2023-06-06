import torch
from torch import nn

import learn2learn as l2l

from module_abc import ModuleABC

from models.target import Target
from models.feat_ex import FeatEx


class HypergradTransform(nn.Module):
    """
    Hypergradient-style per-parameter learning rates
    """

    def __init__(self, param, lr: int = 0.01):
        super(HypergradTransform, self).__init__()

        self.lr = lr * torch.ones_like(param, requires_grad=True)
        self.lr = nn.Parameter(self.lr)

    def forward(self, grad):
        return self.lr * grad


class HyperGrad(ModuleABC):
    """
    Hypergradient method based on:
    https://github.com/learnables/learn2learn/blob/master/examples/optimization/hypergrad_mnist.py
    """

    def __init__(self, lr: float = 0.01, gm: float = 0.001):
        super(HyperGrad, self).__init__()

        self.state = None   # For repetitions

        self.FE = FeatEx
        self.target = Target()
        self.loss = nn.NLLLoss()
        self.lr = lr

        self.opt = l2l.optim.LearnableOptimizer(
            model=self.target, 
            transform=HypergradTransform,
            lr = gm
        )

    @property
    def target(self):
        return self.target

    def reset_target(self):
        """
        Reinitalizes target model
        """

        self.target = Target()

    def get_state(self, opt):
        """
        Given optimizer returns it's state
        """

        return opt.transforms

    def set_state(self, state):
        """
        Sets optimizer state in order to save it
        """

        self.state = state

    def training_step(self, batch, batch_idx):
        """
        Training step in most simple flow
        Returns `preds` & `err`
        """

        x, y = batch
        x_embd = torch.reshape(self.FE(x), (-1, 512))
        preds = self.target(x_embd)
        err = self.loss(preds, y)
        return (x_embd, preds, err)

    def configure_optimizers(self):
        """
        Sets-up & return proper optimizers (meta and normal)
        """

        metaopt = torch.optim.Adam(self.opt.parameters(), lr=self.lr)
        return [self.opt, metaopt]