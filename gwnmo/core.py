from functools import reduce
from operator import mul
import warnings

import learn2learn as l2l
import torch

from utils import device, normalize_weighting


class GWNMO(torch.nn.Module):
    """
    Gradient Weighting by Neural Meta Optimizer implementation based on learn2learn LearnableOptimizer
    """

    def __init__(self, model, transform, gamma=0.01, normalize=True, weighting=True):
        super(GWNMO, self).__init__()
        self.gamma = gamma
        self.normalize = normalize
        assert isinstance(model, torch.nn.Module), \
            'model should inherit from nn.Module.'

        # Keep pointer to model, but don't include in self._modules,
        # self._children, or self._parameters
        self.info = {
            'model': model,
        }

        self._weighting = weighting
        self.transform = transform

    def set_state(self, model):
        self.info = {
            'model': model
        }

    def step(self, x_embd, closure=None):
        model = self.info['model']
        # Ignore warnings as torch 1.5+ warns about accessing .grad of non-leaf
        # variables.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            params = list(model.parameters())

            for p in params:
                if hasattr(p, 'grad') and p.grad is not None:
                    p.grad.detach_()
                    p.data.detach_()

            grad_lengths: list[torch.Size] = [ param.grad.shape for param in params if hasattr(param, 'grad') and param.grad is not None ]

            grad: torch.Tensor = torch.cat([ param.grad.flatten() for param in params if hasattr(param, 'grad') and param.grad is not None ]).to(device)
            grad.requires_grad = False

            param_vals: torch.Tensor = torch.cat([ param.data.flatten() for param in params if hasattr(param, 'grad') and param.grad is not None ])
            param_vals.requires_grad = False

            h: torch.Tensor = self.transform(param_vals, grad, x_embd)

            updates: torch.Tensor
            if self.normalize:
                updates = -self.gamma*normalize_weighting(h, grad)
            else:
                updates = -self.gamma*h*grad

            if not self._weighting:
                ones = torch.ones_like(updates)
                torch.where((updates-ones).bool(), updates, ones)

            start = 0
            for i in range(len(params)):
                if hasattr(params[i], 'grad') and params[i].grad is not None:
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
