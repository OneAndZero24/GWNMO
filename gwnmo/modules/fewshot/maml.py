import torch
from torch import nn, optim

import learn2learn as l2l

from modules.fewshot.fsmodule_abc import FSModuleABC

from utils import logger, device, map_classes
from models.target import ScallableTarget, Target
from models.feature_extractor import FeatureExtractor, TrainableFeatureExtractor, SimpleFeatureExtractor

class MAML(FSModuleABC):
    """
    MAML learn2learn implementation
    """

    def __init__(self,
        lr1: float = 0.01,
        lr2: float = 0.01,
        adaptation_steps: int = 1,
        ways: int = 5,
        shots: int = 1,
        query: int = 10,
        trainable_fe: bool = False, 
        feature_extractor_backbone = None
    ):
        super(MAML, self).__init__()

        if not trainable_fe:
            self.FE = FeatureExtractor().to(device)
        else:
            self.FE = TrainableFeatureExtractor(backbone_name=feature_extractor_backbone, flatten=True).to(device)
        #self.FE = SimpleFeatureExtractor().to(device)
        self.loss = nn.NLLLoss()

        self.lr1 = lr1
        self.lr2 = lr2
        self.adaptation_steps = adaptation_steps
        self.ways = ways
        self.shots = shots
        self.query = query
        
        self.reset_target()

        self.opt = l2l.algorithms.MAML(self.target, lr=lr1)

    @property
    def target(self):
        return self._target

    def reset_target(self):
        self._target = Target().to(device)

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
            preds = learner(adapt_X_embd)
            err = self.loss(preds, adapt_y)
            learner.adapt(err)

        return learner(eval_X_embd)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y = map_classes(self.ways, y)
        adapt_X = X[:,:self.shots,:,:,:].contiguous().view( self.ways* self.shots, *X.size()[2:]) #support data 
        eval_X = X[:,self.shots:,:,:,:].contiguous().view( self.ways* self.query, *X.size()[2:]) #query data

        adapt_y = y[:,:self.shots].reshape((-1))
        eval_y = y[:,self.shots:].reshape((-1))

        check = (len(set(torch.unique(torch.flatten(eval_y)).tolist()).difference(set(torch.unique(torch.flatten(adapt_y)).tolist()))) == 0)
        if not check:
            logger.get().print_to_term('debug', f'Dataset Fuckd')

        # logger.get().print_to_term('debug', f'Adapt X: {adapt_X.shape}')
        # logger.get().print_to_term('debug', f'Adapt y: {adapt_y.shape}')

        # logger.get().print_to_term('debug', f'Eval X: {eval_X.shape}')
        # logger.get().print_to_term('debug', f'Eval y: {eval_y.shape}')

        adapt_X, adapt_y, eval_X, eval_y = adapt_X.to(device), adapt_y.to(device), eval_X.to(device), eval_y.to(device)

        adapt_X_embd = torch.reshape(self.FE(adapt_X), (-1, 512))
        eval_X_embd = torch.reshape(self.FE(eval_X), (-1, 512))

        # logger.get().print_to_term('debug', f'Adapt X EMBD: {adapt_X_embd.shape}')
        # logger.get().print_to_term('debug', f'Eval X EMBD: {eval_X_embd.shape}')

        preds = self.adapt(adapt_X_embd, adapt_y, eval_X_embd)

        err = self.loss(preds, eval_y)

        return (eval_y, preds, err)

    def configure_optimizers(self) -> list:
        return optim.Adam(self.opt.parameters(), self.lr2)
