import time
import neptune

from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only

from __init__ import __version__


class NeptuneLogger(Logger):
    """
    Custom, simplistic neptune logger for Lightning due to `neptune` vs `neptune-client` problems.
    """

    def __init__(self, online: bool):
        super(NeptuneLogger, self).__init__()
        self.output = self.print_to_term
        if online:
            self.run = neptune.init_run()
            self.output = self.push_to_neptune

    def print_to_term(self, key, value):
        clock = time.strftime("%H:%M:%S", time.localtime(time.time()))
        print(f'[{clock}] {key}: {value}')

    def push_to_neptune(self, key, value):
        self.run[key].append(value)

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
        self.output('params', params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for key, value in metrics.items():
            self.output(key, value)

    def tag(self, args):
        """
        Converts `args` Namespace to run tags
        """

        tags = [ k+': '+str(v) for (k, v) in list(vars(args).items()) ]
        tags.append(__version__)
        self.output('sys/tags', tags)

    def log_model_summary(self, module):
        """
        Returns `torchsummary` of all `nn.Module` objects in module.
        """

        for attr, value in module.__dict__['_modules'].items():
            self.output('model_summary', str(value))