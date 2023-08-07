from functools import reduce
from operator import mul
import warnings

import learn2learn as l2l
import torch

from utils import device

from torchviz import make_dot


class GWNMO(torch.nn.Module):
    """
    Gradient Weighting by Neural Meta Optimizer implementation based on learn2learn LearnableOptimizer
    """

    def __init__(self, model, transform, gamma=0.01, normalize=True):
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

        self.transform = transform

    def step(self, x_embd, err, closure=None):
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

            param_vals: torch.Tensor = torch.cat([ param.data.flatten() for param in params ])
            param_vals.requires_grad = False

            print(f'{list(self.transform.parameters())}') # DEBUG

            h: torch.Tensor = self.transform(param_vals, grad, x_embd)
            temp : torch.Tensor = torch.clamp(h, min=0, max= 1)
            selected: torch.Tensor = temp*grad

            updates: torch.Tensor
            if self.normalize:
                updates = -self.gamma*selected*(torch.linalg.norm(grad)/torch.linalg.norm(selected))
            else:
                updates = -self.gamma*h*grad
            #updates.detach()

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
                #param.requires_grad = True
                param.retain_grad()

    def zero_grad(self):
        """Only reset target parameters."""
        model = self.info['model']
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                # Do not reset in-place:
                # it breaks the computation graph of step().
                p.grad = torch.zeros_like(p.data).to(device)
