from typing import Any
import torch
from torch import nn

from modules.fewshot.fsmodule_abc import FSModuleABC

from utils import device, split_batch
from models.target import Target
from models.feat_ex import FeatEx
from models.meta_opt import MetaOptimizer
from core import GWNMO as GWNMOopt

class GWNMOFS(FSModuleABC):
    """
    GWNMOFS - Few Shot training algorithm based on GWNMO.
    """

    def __init__(self, lr1: float = 0.01, lr2: float = 0.01, gm: float = 0.001, normalize: bool = True, ways: int = 1, shots: int = 5):
        super(GWNMOFS, self).__init__()

        self.MO = MetaOptimizer().to(device)
        self.MO.train()

        self.FE = FeatEx().to(device)
        self._target = Target().to(device)
        self.loss = nn.NLLLoss()

        self.lr1 = lr1
        self.lr2 = lr2
        self.gamma = gm
        self.normalize = normalize
        self.ways = ways
        self.shots = shots

        self.opt = GWNMOopt(
            model=self._target, 
            transform=self.MO, 
            gamma=self.gamma, 
            normalize=self.normalize
        ).to(device)

    @property
    def target(self):
        return self._target
    
    def reset_target(self):
        """
        Reinitializes target model
        """

        self._target = Target().to(device)

    def get_state(self, opt):
        """
        Given optimizer returns it's state
        """

        return opt.transform
    
    def set_state(self, state):
        """
        Sets optimizer state in order to save it
        """

        self.MO = state
        self.MO.train()

    def clone(self):
        """
        Create detached clone of target & wrap in self clone
        """

        # TODO implement
    
    def adapt(self, adapt_X_embd, adapt_y, eval_X_embd):
        """
        Single GWNMOFS adaptation step
        """

        c = self.clone()

        c.opt.zero_grad()

        preds = c.target(adapt_X_embd)
        err = c.loss(preds, adapt_y)
        err.backward()

        updates = c.opt.step(adapt_X_embd)

        return (updates, c.target(eval_X_embd))

    def training_step(self, batch, batch_idx):
        """
        Single training step
        Returns `updates` & `preds`
        """

        split = split_batch(batch, self.ways, self.shots)
        adapt_X, adapt_y = split['adapt']
        eval_X, eval_y = split['eval']

        adapt_X_embd = torch.reshape(self.FE(adapt_X), (-1, 512))
        eval_X_embd = torch.reshape(self.FE(eval_X), (-1, 512))

        updates, preds = self.adapt(adapt_X_embd, adapt_y, eval_X_embd)

        # TODO return err too 
        # but GWNMOopt has _target params

        return (updates, preds)
    
    def configure_optimizers(self) -> list:
        """
        Sets-up & returns proper optimizers alpha & start
        """

        adam1 = torch.optim.Adam(self.opt.parameters(), lr=self.lr1)
        adam2 = torch.optim.Adam(self.target.parameters(), lr=self.lr2)

        return [adam1, adam2]

        
    
