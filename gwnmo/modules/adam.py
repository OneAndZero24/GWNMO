import torch
from torch import nn

from modules.module_abc import ModuleABC

from models.target import Target
from models.feat_ex import FeatEx


class Adam(ModuleABC):
    """
    Baseline - Target optimized by adam
    """

    def __init__(self, lr: float = 0.01):
        super(Adam, self).__init__()

        self.state = None   # For repetitions

        self.FE = FeatEx
        self.target = Target()
        self.loss = nn.NLLLoss()

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

        return opt.state_dict()

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
        Sets-up & return proper optimizer
        """

        opt = torch.optim.Adam(self.target.parameters(), lr=self.lr)
        if self.state is not None:
            opt.load_state_dict(self.state)
        return [opt]