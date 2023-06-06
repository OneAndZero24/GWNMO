from abc import ABCMeta, abstractmethod

import pytorch_lightning as pl


class ModuleABC(pl.LightningModule, metaclass = ABCMeta):
    """
    Training module interface
    """

    @property
    @abstractmethod
    def target(self):
        ...

    @abstractmethod
    def reset_target(self):
        ...
    
    @abstractmethod
    def get_state(self, opt):
        ...

    @abstractmethod
    def set_state(self, state):
        ...
    
    @abstractmethod
    def training_step(self, batch, batch_idx):
        ...

    @abstractmethod
    def configure_optimizers(self) -> list:
        ...
