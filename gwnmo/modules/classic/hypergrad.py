import torch
from torch import nn

import learn2learn as l2l

from modules.module_abc import ModuleABC

from utils import device, normalize_weighting
from models.target import Target
from models.feature_extractor import FeatureExtractor


def gen_hypergrad_transform(normalize: bool = False):
    """
    Generates Hypergrad Transform class
    """

    def nonorm(self):
        def forward(grad):
            return self.lr * grad
        return forward

    def norm(self):
        def forward(grad):
            return normalize_weighting(self.lr, grad)
        return forward

    choice = norm if normalize else nonorm

    class HypergradTransform(nn.Module):
        """
        Hypergradient-style per-parameter learning rates
        """

        def __init__(self, param, lr: int = 0.01):
            super(HypergradTransform, self).__init__()

            self.lr = lr * torch.ones_like(param, requires_grad=True)
            self.lr = nn.Parameter(self.lr)
            self.forward = choice(self)

    return HypergradTransform

class HyperGrad(ModuleABC):
    """
    Hypergradient method based on:
    https://github.com/learnables/learn2learn/blob/master/examples/optimization/hypergrad_mnist.py
    """

    def __init__(self, lr: float = 0.01, gm: float = 0.001, normalize: bool = False):
        super(HyperGrad, self).__init__()

        self.state = None   # For repetitions

        self.FE = FeatureExtractor().to(device)
        self._target = Target().to(device)
        self.loss = nn.NLLLoss()

        self.lr = lr
        self.gamma = gm
        self.normalize = normalize

    @property
    def target(self):
        return self._target

    def reset_target(self):
        """
        Reinitalizes target model
        """

        self._target = Target().to(device)

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
        preds = self._target(x_embd)
        err = self.loss(preds, y)
        return (x_embd, preds, err)

    def configure_optimizers(self):
        """
        Sets-up & return proper optimizers (meta and normal)
        """
        opt = l2l.optim.LearnableOptimizer(
            model=self._target, 
            transform=gen_hypergrad_transform(self.normalize),
            lr = self.gamma
        ).to(device)
        metaopt = torch.optim.Adam(opt.parameters(), lr=self.lr)
        return [opt, metaopt]