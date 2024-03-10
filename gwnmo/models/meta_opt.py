import torch
from torch import nn

from fairscale.experimental.nn.offload import OffloadModel

from utils import device

CLASSIC_10CLASS_WEIGHTS_SIZE = 33482
CLASSIC_10CLASS_IN_SIZE = 83348

class MetaOptimizer(nn.Module):
    """
    Gradient weighting network in `GWNMO`
    """

    def __init__(self, insize=CLASSIC_10CLASS_IN_SIZE, outsize=CLASSIC_10CLASS_WEIGHTS_SIZE):
        """
        insize - concat(param_vals, grad, x_embd)
        outsize - grad
        """

        super(MetaOptimizer, self).__init__()
        self.seq = nn.Sequential()
        self.seq.append(nn.Linear(insize, 4096))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.Linear(4096, outsize))
        self.seq.append(nn.ReLU())
        self.model = OffloadModel(
            model=self.seq,
            device=device,
            offload_device=torch.device("cpu"),
            num_slices=3,
            num_microbatches=1
        )

    def forward(self, params, grad, x_embd):
        x = torch.cat([params, grad, x_embd.flatten()])
        return self.model(x)