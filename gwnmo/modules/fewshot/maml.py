import torch
from torch import nn, optim

import learn2learn as l2l
from learn2learn.utils import clone_module, detach_module

from modules.fewshot.fsmodule_abc import FSModuleABC

from utils import logger, device, map_classes
from models.target import SmolTarget
from models.body import Body
from models.feature_extractor import FeatureExtractor, TrainableFeatureExtractor

class MAML(FSModuleABC):
    """
    MAML learn2learn implementation
    """

    def __init__(self,
        lr1: float = 0.01,
        lr2: float = 0.01,
        fe_lr: float = 0.01, 
        adaptation_steps: int = 1,
        ways: int = 5,
        shots: int = 1,
        query: int = 10,
        trainable_fe: bool = False, 
        feature_extractor_backbone = None,
        second_order: bool = False,
    ):
        super(MAML, self).__init__()

        self.trainable_fe = trainable_fe
        if not trainable_fe:
            self.FE = FeatureExtractor().to(device)
        else:
            self.FE = TrainableFeatureExtractor(backbone_name=feature_extractor_backbone, flatten=True).to(device)
        self.loss = nn.NLLLoss()

        self.body = nn.Identity().to(device)

        self.lr1 = lr1
        self.lr2 = lr2
        self.fe_lr = fe_lr
        self.adaptation_steps = adaptation_steps
        self.ways = ways
        self.shots = shots
        self.query = query
        
        self.reset_target()

        self.opt = l2l.algorithms.MAML(self.target, lr=lr1, first_order=(not second_order))

    @property
    def target(self):
        return self._target

    def reset_target(self):
        self._target = SmolTarget().to(device)

    def get_state(self):
        return self.opt

    def set_state(self, state):
        self.opt = state
        self.opt.train()

    def adapt(self, adapt_X_embd, adapt_y, eval_X_embd):
        learner = self.opt.clone()
        learner.zero_grad()
        learner.train()

        for i in range(self.adaptation_steps):
            preds = learner(self.body(adapt_X_embd))
            err = self.loss(preds, adapt_y)
            learner.adapt(err)

        return learner(self.body(eval_X_embd))

    def training_step(self, batch, batch_idx):
        X, y = batch
        y = map_classes(self.ways, y)
        adapt_X = X[:,:self.shots,:,:,:].contiguous().view( self.ways* self.shots, *X.size()[2:]) #support data 
        eval_X = X[:,self.shots:,:,:,:].contiguous().view( self.ways* self.query, *X.size()[2:]) #query data

        adapt_y = y[:,:self.shots].reshape((-1))
        eval_y = y[:,self.shots:].reshape((-1))

        check = (len(set(torch.unique(torch.flatten(eval_y)).tolist()).difference(set(torch.unique(torch.flatten(adapt_y)).tolist()))) == 0)
        if not check:
            logger.get().print_to_term('debug', f'Dataset Fckd')

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

        preds = self.adapt(adapt_X_embd, adapt_y, eval_X_embd)

        err = self.loss(preds, eval_y)

        return (eval_y, preds, err)

    def configure_optimizers(self) -> list:
        adam = torch.optim.Adam([
            {'params': self.opt.parameters(), 'lr': self.lr2},
            {'params': self.body.parameters(), 'lr': self.lr2},
            {'params': self.FE.parameters(), 'lr': self.fe_lr},
        ])

        return adam
