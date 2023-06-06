import torch
from torch import nn

from module_abc import ModuleABC

from models.target import Target
from models.feat_ex import FeatEx
from models.meta_opt import MetaOptimizer
from core import GWNMO as GWNMOopt

class GWNMO(ModuleABC):
    """
    GWNMO (Gradient Weighting by Neural Meta Optimizer)
    Optimizer weights optimizee's gradient using a neural network.
    """

    def __init__(self, lr: float = 0.01, gm: float = 0.001, normalize: bool = True):
        super(GWNMO, self).__init__()

        self.MO = MetaOptimizer()

        self.FE = FeatEx
        self.target = Target()
        self.loss = nn.NLLLoss()
        self.lr = lr

        self.opt = GWNMOopt(
            model=self.target, 
            transform=self.MO, 
            gamma=gm, 
            normalize=normalize
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

        return opt

    def set_state(self, state):
        """
        Sets optimizer state in order to save it
        """

        pass

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