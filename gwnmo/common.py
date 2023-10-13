import torch

def normalize(x, grad):
    """
    Permforms magical normalization described in PDF given meta optimizer's network output and gradient
    """

    temp : torch.Tensor = torch.clamp(x, min=0, max=1)
    selected: torch.Tensor = temp*grad

    return selected*(torch.linalg.norm(grad)/torch.linalg.norm(selected))