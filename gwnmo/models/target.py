import torch
from torch import nn
import torch.nn.functional as F


class Target(nn.Module):
    """
    Target network to be trained.
    Works on data processed by feature extractor (output of `FeatureExtractor`)
    """

    def __init__(self):
        super(Target, self).__init__()

        self.seq = nn.Sequential()
        self.seq.append(nn.Linear(512, 64))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.Linear(64, 10))

    def forward(self, x: torch.Tensor):
        x = self.seq(x)
        return F.log_softmax(x, dim=1)
    

class ScallableTarget(nn.Module):
    """
    Target network that scales it's output size based on number of classes.
    """

    def __init__(self, classes):
        super(Target, self).__init__()

        self.seq = nn.Sequential()
        for i in range(6, 10):
            p = 2**i
            if classes > p:
                self.seq.append
                break
            else:
                # TODO
                pass
        self.seq.append(nn.Linear(512, 64))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.Linear(64, 10))

    def forward(self, x: torch.Tensor):
        x = self.seq(x)
        return F.log_softmax(x, dim=1)