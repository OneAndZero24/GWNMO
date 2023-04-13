import torch
import learn2learn as l2l
import warnings


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

            params = model.parameters()

            grad = torch.Tensor([ param.grad.detach() for param in params if hasattr(param, 'grad') and param.grad is not None ])
            grad.requires_grad = False

            updates = -grad*self.transform(grad, x_embd)

            for param, update in zip(params, updates):
                param.detach_()
                param.requires_grad = False
                param.update = update

            l2l.update_module(model)

            for param in model.parameters():
                param.retain_grad()

    def zero_grad(self):
        """Only reset target parameters."""
        model = self.info['model']
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                # Do not reset in-place:
                # it breaks the computation graph of step().
                p.grad = torch.zeros_like(p.data)