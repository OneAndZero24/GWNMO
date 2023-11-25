import torch
from torch import nn

from learn2learn.utils import clone_module, detach_module
from learn2learn.data.utils import partition_task

from modules.fewshot.fsmodule_abc import FSModuleABC

from utils import device 
from models.target import ScallableTarget
from models.feature_extractor import FeatureExtractor, TrainableFeatureExtractor
from models.meta_opt import MetaOptimizer
from core import GWNMO as GWNMOopt

class GWNMOFS(FSModuleABC):
    """
    GWNMOFS - Few Shot training algorithm based on GWNMO.
    """

    def __init__(self, 
                 lr: float = 0.01, 
                 gm: float = 0.001,
                 normalize: bool = True, 
                 adaptation_steps: int = 1, 
                 ways: int = 5,
                 shots: int = 1, 
                 query: int = 50,
                 trainable_fe: bool = False, 
                 feature_extractor_backbone = None
        ):
        super(GWNMOFS, self).__init__()

        self.MO = MetaOptimizer().to(device)
        self.MO.train()

        
        if not trainable_fe:
            self.FE = FeatureExtractor().to(device)
        else:
            self.FE = TrainableFeatureExtractor(backbone_name=feature_extractor_backbone, flatten=True).to(device)
        self.loss = nn.NLLLoss()

        self.lr = lr
        self.gamma = gm
        self.normalize = normalize
        self.adaptation_steps = adaptation_steps
        self.ways = ways
        self.shots = shots
        self.query = query

        self.reset_target()

        self.MO = MetaOptimizer(insize=348170, outsize=172805).to(device)
        self.MO.train()

        self.loss = nn.NLLLoss()

    @property
    def target(self):
        return self._target
    
    def reset_target(self):
        """
        Reinitializes target model
        """

        self._target = ScallableTarget(self.ways).to(device)

    def get_state(self, opt):
        """
        Given optimizer returns it's state
        """

        return self.MO
    
    def set_state(self, state):
        """
        Sets optimizer state in order to save it
        """

        self.MO = state
        self.MO.train()

    def adapt(self, adapt_X_embd, adapt_y, eval_X_embd):
        """
        Single GWNMOFS adaptation step
        """

        clone = clone_module(self.target)
        detach_module(clone, keep_requires_grad=True)

        self.opt = GWNMOopt(
            model=clone, 
            transform=self.MO, 
            gamma=self.gamma, 
            normalize=self.normalize
        ).to(device)      

        for i in range(self.adaptation_steps):
            self.opt.zero_grad()

            preds = clone(adapt_X_embd)
            err = self.loss(preds, adapt_y)
            err.backward(retain_graph=True)

            self.opt.step(adapt_X_embd)

        return clone(eval_X_embd)

    def training_step(self, batch, batch_idx):
        """
        Single training step
        Returns `evl_y`, `preds` & `err`
        """

        X, y = batch
        (adapt_X, adapt_y), (eval_X, eval_y) = partition_task(X, y, shots=self.shots)

        adapt_X, adapt_y, eval_X, eval_y = adapt_X.to(device), adapt_y.to(device), eval_X.to(device), eval_y.to(device)

        adapt_X_embd = torch.reshape(self.FE(adapt_X), (-1, 512))
        eval_X_embd = torch.reshape(self.FE(eval_X), (-1, 512))

        preds = self.adapt(adapt_X_embd, adapt_y, eval_X_embd)

        err = self.loss(preds, eval_y)

        return (eval_y, preds, err)
    
    def configure_optimizers(self) -> list:
        """
        Sets-up & returns proper optimizers alpha & start
        """

        adam = torch.optim.Adam(self.opt.parameters(), lr=self.lr)

        return adam
