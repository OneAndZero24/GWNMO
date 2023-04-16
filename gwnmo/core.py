from functools import reduce
from operator import mul
import warnings

import learn2learn as l2l
import numpy as np
import torch

from gwnmo.utils import log, device


class GWNMO(torch.nn.Module):
    """Gradient Weighting by Neural Meta Optimizer implementation based on learn2learn LearnableOptimizer"""

    def __init__(self, model, transform):
        super(GWNMO, self).__init__()
        assert isinstance(model, torch.nn.Module), \
            'model should inherit from nn.Module.'

        # Keep pointer to model, but don't include in self._modules,
        # self._children, or self._parameters
        self.info = {
            'model': model,
        }

        self.transform = transform()

    def step(self, x_embd, closure=None):
        model = self.info['model']
        # Ignore warnings as torch 1.5+ warns about accessing .grad of non-leaf
        # variables.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            params = list(model.parameters())
                  
            log.info('Core algo sanity check')
            log.info(f'Params: {len(params)}')

            grad_lengths: list[torch.Size] = [ param.grad.detach().shape for param in params if hasattr(param, 'grad') and param.grad is not None ]

            log.info(f'Gradients: {grad_lengths}')

            grad: torch.Tensor = torch.cat([ param.grad.detach().flatten() for param in params if hasattr(param, 'grad') and param.grad is not None ]).to(device)
            grad.requires_grad = False

            log.info(f'Gradient: {grad.shape}')

            updates: torch.Tensor = -grad*self.transform(grad, x_embd)

            start = 0
            for i in range(len(params)):
                params[i].detach_()
                params[i].requires_grad = False
                size = grad_lengths[i]
                f_size = reduce(mul, grad_lengths[i])
                params[i].update = updates[start:start+f_size].reshape(size) # type: ignore
                start += f_size

            l2l.update_module(model, updates=None)

            for param in model.parameters():
                param.retain_grad()

    def zero_grad(self):
        """Only reset target parameters."""
        model = self.info['model']
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                # Do not reset in-place:
                # it breaks the computation graph of step().
                p.grad = torch.zeros_like(p.data).to(device)