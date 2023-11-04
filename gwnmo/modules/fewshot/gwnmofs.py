import torch
from torch import nn

from learn2learn.utils import clone_module
from learn2learn.data.utils import partition_task

from modules.fewshot.fsmodule_abc import FSModuleABC

from datasets import OMNIGLOT_CLASSES
from utils import device 
from models.target import ScallableTarget
from models.feat_ex import FeatEx
from models.meta_opt import MetaOptimizer
from core import GWNMO as GWNMOopt

class GWNMOFS(FSModuleABC):
    """
    GWNMOFS - Few Shot training algorithm based on GWNMO.
    """

    def __init__(self, lr1: float = 0.01, lr2: float = 0.01, gm: float = 0.001,
                normalize: bool = True, adaptation_steps: int = 1, ways: int = 1,
                shots: int = 5, target: nn.Module = ScallableTarget(OMNIGLOT_CLASSES)):
        super(GWNMOFS, self).__init__()

        self.MO = MetaOptimizer(insize=1672878, outsize=832599).to(device)
        self.MO.train()

        self.FE = FeatEx().to(device)
        self._target = target.to(device)
        self.loss = nn.NLLLoss()

        self.lr1 = lr1
        self.lr2 = lr2
        self.gamma = gm
        self.normalize = normalize
        self.adaptation_steps = adaptation_steps
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

        self._target = ScallableTarget(OMNIGLOT_CLASSES).to(device)

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

        return GWNMOFS(self.lr1, self.lr2, self.gamma, self.normalize, self.ways, self.shots, clone_module(self.target))
    
    def adapt(self, adapt_X_embd, adapt_y, eval_X_embd):
        """
        Single GWNMOFS adaptation step
        """

        preds = self.target(adapt_X_embd)
        err = self.loss(preds, adapt_y)
        err.backward(retain_graph=True)

        self.opt.step(adapt_X_embd)

        return self.target(eval_X_embd)

    def training_step(self, batch, batch_idx):
        """
        Single training step
        Returns `preds` & `err`
        """

        X, y = batch
        (adapt_X, adapt_y), (eval_X, eval_y) = partition_task(X, y, shots=self.shots)

        adapt_X_embd = torch.reshape(self.FE(adapt_X), (-1, 512))
        eval_X_embd = torch.reshape(self.FE(eval_X), (-1, 512))

        c = self.clone()
        c.opt.zero_grad()
        for i in range(self.adaptation_steps):
            preds = c.adapt(adapt_X_embd, adapt_y, eval_X_embd)

        err = self.loss(preds, eval_y)

        return (eval_X_embd, preds, err)
    
    def configure_optimizers(self) -> list:
        """
        Sets-up & returns proper optimizers alpha & start
        """

        self.opt = GWNMOopt(
            model=self.target, 
            transform=self.MO, 
            gamma=self.gamma, 
            normalize=self.normalize
        ).to(device)

        adam1 = torch.optim.Adam(self.opt.parameters(), lr=self.lr1)
        adam2 = torch.optim.Adam(self.target.parameters(), lr=self.lr2)

        return [adam1, adam2]
