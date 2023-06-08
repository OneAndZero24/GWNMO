import neptune

import torch.nn as nn

from torchsummary import summary

from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only

from __init__ import __version__


class NeptuneLogger(Logger):
    """
    Custom, simplistic neptune logger for Lightning due to `neptune` vs `neptune-client` problems.
    """

    def __init__(self, args):
        super(NeptuneLogger, self).__init__()
        self.run = neptune.init_run()
        tags = [ k+': '+str(v) for (k, v) in list(vars(args).items()) ]
        tags.append(__version__)
        self.run["sys/tags"].add(tags)

    @property
    def name(self):
        return "NeptuneLogger"

    @property
    def version(self):
        """
        Return the experiment version str
        """

        return __version__

    @rank_zero_only
    def log_hyperparams(self, params):
        self.run["params"].append(params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for key, value in metrics.items():
            self.run[key].append(value)

    def log_model_summary(self, module):
        """
        Returns `torchsummary` of all `nn.Module` objects in module.
        """

        for attr, value in module.__dict__.items():
            if type(value) is nn.Module:
                self.run['model_summary'] = summary(value)