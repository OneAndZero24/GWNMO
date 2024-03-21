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
        self.seq.append(nn.BatchNorm1d(512))
        self.seq.append(nn.Linear(512, 4096))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.BatchNorm1d(4096))
        self.seq.append(nn.Linear(4096, 1024))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.BatchNorm1d(1024))
        self.seq.append(nn.Linear(1024, 10))

    def forward(self, x: torch.Tensor):
        x = self.seq(x)
        return F.log_softmax(x, dim=1)
    

class ScallableTarget(nn.Module):
    """
    Target network that scales it's output size based on number of classes.
    """

    def __init__(self, classes: int):
        super(ScallableTarget, self).__init__()

        self.seq = nn.Sequential()
        for i in range(9, 5, -1):
            p = 2**i
            if classes > p:
                self.seq.append(nn.BatchNorm1d(p))
                self.seq.append(nn.Linear(p, classes))
                break
            else:
                if i-1 > 5:
                    self.seq.append(nn.BatchNorm1d(p))
                    self.seq.append(nn.Linear(p, p//2))
                    self.seq.append(nn.ReLU())
                else:
                    self.seq.append(nn.BatchNorm1d(p))
                    self.seq.append(nn.Linear(p, classes))

    def forward(self, x: torch.Tensor):
        x = self.seq(x)
        return F.log_softmax(x, dim=1)
    

class SmolTarget(nn.Module):
    """
    Target network to be trained.
    Specifically for `GWNMOFS`.  

    Works on data processed by feature extractor & body
    """

    def __init__(self):
        super(SmolTarget, self).__init__()

        self.seq = nn.Sequential()
        self.seq.append(nn.BatchNorm1d(512))
        self.seq.append(nn.Linear(512, 10))

    def forward(self, x: torch.Tensor):
        x = self.seq(x)
        return F.log_softmax(x, dim=1)