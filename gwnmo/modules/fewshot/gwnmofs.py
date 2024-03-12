import torch
from torch import nn

from learn2learn.utils import clone_module, detach_module

from modules.fewshot.fsmodule_abc import FSModuleABC

from utils import device, map_classes
from models.target import Target
from models.body import Body
from models.feature_extractor import FeatureExtractor, TrainableFeatureExtractor
from models.meta_opt import MetaOptimizer
from core import GWNMO as GWNMOopt


OMNIGLOT_RESNET18_IN = 87316
OMNIGLOT_RESNET18_OUT = 12298

class GWNMOFS(FSModuleABC):
    """
    GWNMOFS - Few Shot training algorithm based on GWNMO.
    """

    def __init__(self, 
        lr1: float = 0.01,
        lr2: float = 0.01,  
        gm: float = 0.001,
        normalize: bool = True, 
        adaptation_steps: int = 1, 
        ways: int = 5,
        shots: int = 1, 
        query: int = 10,
        trainable_fe: bool = False, 
        feature_extractor_backbone = None,
        mo_insize: int = OMNIGLOT_RESNET18_IN,
        mo_outsize: int = OMNIGLOT_RESNET18_OUT
    ):
        super(GWNMOFS, self).__init__()
        
        if not trainable_fe:
            self.FE = FeatureExtractor().to('cuda:0')
        else:
            self.FE = TrainableFeatureExtractor(backbone_name=feature_extractor_backbone, flatten=True).to('cuda:0')
        self.loss = nn.NLLLoss()

        self.body = Body().to('cuda:0')

        self._weighting = True

        self.lr1 = lr1
        self.lr2 = lr2
        self.gamma = gm
        self.normalize = normalize
        self.adaptation_steps = adaptation_steps
        self.ways = ways
        self.shots = shots
        self.query = query

        self.reset_target()

        self.MO = MetaOptimizer(insize=mo_insize, outsize=mo_outsize).to('cuda:1') 
        self.MO.train()

        self.opt = GWNMOopt(
            model=self.target, 
            transform=self.MO, 
            gamma=self.gamma, 
            normalize=self.normalize,
            weighting=self._weighting
        ) 
        
    @property
    def target(self):
        return self._target
    
    def toggle_weighting(self, state):
        self._weighting = state

    def reset_target(self):
        """
        Reinitializes target model
        """

        self._target = Target().to('cuda:1')

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

        for i in range(self.adaptation_steps):
            self.opt.zero_grad()

            preds = clone(self.body(adapt_X_embd))
            err = self.loss(preds, adapt_y)
            err.backward(retain_graph=True)

            self.opt.step(adapt_X_embd.detach())

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

        adapt_X, adapt_y, eval_X, eval_y = adapt_X.to('cuda:0'), adapt_y.to('cuda:0'), eval_X.to('cuda:0'), eval_y.to('cuda:1')

        adapt_X_embd = torch.flatten(self.FE(adapt_X), -3).to('cuda:0')
        eval_X_embd = torch.flatten(self.FE(eval_X), -3).to('cuda:1')

        detach_module(self.target, keep_requires_grad=True)

        # self.FE.to('cuda:1') ?
        # self.body.to('cuda:1') ?
        preds = self.adapt(adapt_X_embd, adapt_y, eval_X_embd).to('cuda:1')
        # self.FE.to('cuda:0') ?
        # self.body.to('cuda:0') ?

        err = self.loss(preds, eval_y)

        return (eval_y, preds, err)
    
    def configure_optimizers(self) -> list:
        """
        Sets-up & returns proper optimizer
        """

        adam = torch.optim.Adam([
            {'params': self.opt.parameters(), 'lr': self.lr1},
            {'params': self.target.parameters(), 'lr': self.lr2},
        ])

        return adam
