import torch
from torch import nn

from learn2learn.utils import clone_module, detach_module

from modules.fewshot.fsmodule_abc import FSModuleABC

from utils import device, map_classes
from models.target import SmolTarget
from models.body import Body
from models.feature_extractor import FeatureExtractor, TrainableFeatureExtractor
from models.meta_opt import FSMetaOptimizer
from core import GWNMO as GWNMOopt


OMNIGLOT_CONV4P_WEIGHTS = 778

class GWNMOFS(FSModuleABC):
    """
    GWNMOFS - Few Shot training algorithm based on GWNMO.
    """

    def __init__(self, 
        lr1: float = 0.01,
        lr2: float = 0.01,  
        fe_lr: float = 0.01, 
        gm: float = 0.001,
        normalize: bool = True, 
        adaptation_steps: int = 1, 
        ways: int = 5,
        shots: int = 1, 
        query: int = 10,
        trainable_fe: bool = False, 
        feature_extractor_backbone = None,
        second_order: bool = False,
        mo_size: int = OMNIGLOT_CONV4P_WEIGHTS
    ):
        super(GWNMOFS, self).__init__()
        
        self.second_order = second_order
        self.trainable_fe = trainable_fe
        if not trainable_fe:
            self.FE = FeatureExtractor().to(device)
        else:
            self.FE = TrainableFeatureExtractor(backbone_name=feature_extractor_backbone, flatten=True).to(device)
        self.loss = nn.NLLLoss()

        self.body = nn.Identity().to(device)

        self._weighting = True

        self.lr1 = lr1
        self.lr2 = lr2
        self.fe_lr = fe_lr
        self.gamma = gm
        self.normalize = normalize
        self.adaptation_steps = adaptation_steps
        self.ways = ways
        self.shots = shots
        self.query = query

        self.reset_target()

        self.MO = FSMetaOptimizer(size=mo_size).to(device)
        self.MO.train()

        self.opt = GWNMOopt(
            model=self.target, 
            transform=self.MO, 
            gamma=self.gamma, 
            normalize=self.normalize,
            weighting=self._weighting
        ).to(device)  
        
    @property
    def target(self):
        return self._target
    
    def toggle_weighting(self, state):
        self._weighting = state

    def reset_target(self):
        """
        Reinitializes target model
        """

        self._target = SmolTarget().to(device)

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
        clone.train()
        for param in clone.parameters():
            param.retain_grad()

        self.opt.set_state(clone)    
        self.opt.zero_grad()
        for i in range(self.adaptation_steps):
            adapt_X_embd = self.body(adapt_X_embd)

            preds = clone(adapt_X_embd)
            err = self.loss(preds, adapt_y)
            err.backward(retain_graph=True, create_graph=self.second_order)

            self.opt.step(adapt_X_embd)

        return clone(self.body(eval_X_embd))

    def training_step(self, batch, batch_idx):
        """
        Single training step
        Returns `evl_y`, `preds` & `err`
        """

        X, y = batch
        y = map_classes(self.ways, y)
        adapt_X = X[:,:self.shots,:,:,:].contiguous().view( self.ways* self.shots, *X.size()[2:]) #support data 
        eval_X = X[:,self.shots:,:,:,:].contiguous().view( self.ways* self.query, *X.size()[2:]) #query data
        
        adapt_y = y[:,:self.shots].reshape((-1))
        eval_y = y[:,self.shots:].reshape((-1))

        adapt_X, adapt_y, eval_X, eval_y = adapt_X.to(device), adapt_y.to(device), eval_X.to(device), eval_y.to(device)

        detach_module(self.FE)
        FEclone = clone_module(self.FE)
        adapt_X_embd = FEclone(adapt_X)
        eval_X_embd = FEclone(eval_X)

        self.FE = FEclone

        if not self.trainable_fe:
            adapt_X_embd = torch.reshape(adapt_X_embd, (-1, 512))
            eval_X_embd = torch.reshape(eval_X_embd, (-1, 512))
        else:
            adapt_X_embd = torch.flatten(adapt_X_embd, -3)
            eval_X_embd = torch.flatten(eval_X_embd, -3)

        detach_module(self.target, keep_requires_grad=True)

        preds = self.adapt(adapt_X_embd, adapt_y, eval_X_embd)

        err = self.loss(preds, eval_y)

        return (eval_y, preds, err)
    
    def configure_optimizers(self) -> list:
        """
        Sets-up & returns proper optimizer
        """

        adam = torch.optim.Adam([
            {'params': self.opt.parameters(), 'lr': self.lr1},
            {'params': self.target.parameters(), 'lr': self.lr2},
            {'params': self.body.parameters(), 'lr': self.lr2},
            {'params': self.FE.parameters(), 'lr': self.fe_lr},
        ])

        return adam
