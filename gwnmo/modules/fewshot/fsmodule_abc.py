from abc import ABCMeta, abstractmethod

from modules.module_abc import ModuleABC


class FSModuleABC(ModuleABC, metaclass = ABCMeta):
    """
    Few-Shot training module interface
    """

    @abstractmethod
    def adapt(self, adapt_X_embd, adapt_y, eval_X_embd):
        ...
        