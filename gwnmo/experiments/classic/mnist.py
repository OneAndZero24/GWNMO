"""
Based on: https://github.com/learnables/learn2learn/blob/master/examples/optimization/hypergrad_mnist.py
"""

from typing import Callable

import torch
import torchvision as tv
import learn2learn as l2l
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from gwnmo.core import GWNMO
from gwnmo.models.grad_fe import GradFeatEx
from gwnmo.models.simple_cnn import SimpleCNN
from gwnmo.utils import log, accuracy, run, device


def _setup_dataset():
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(
        tv.datasets.MNIST('~/data', train=True, download=True,
                          transform=tv.transforms.Compose([
                              tv.transforms.ToTensor(),
                              tv.transforms.Normalize((0.1307,), (0.3081,))
                          ])),
        batch_size=1, shuffle=True, **kwargs)
    test_loader = DataLoader(
        tv.datasets.MNIST('~/data', train=False,
                          transform=tv.transforms.Compose([
                              tv.transforms.ToTensor(),
                              tv.transforms.Normalize((0.1307,), (0.3081,))
                          ])),
        batch_size=1, shuffle=False, **kwargs)
    return (train_loader, test_loader)

def _test(target: nn.Module, test_loader: DataLoader):
    target.eval()
    test_accuracy = 0.0
    with torch.no_grad():
        for _, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            preds = target(X)
            test_accuracy += accuracy(preds, y)
        test_accuracy /= len(test_loader)
    log.info(f'Accuracy: {test_accuracy}')
    run["accuracy"].append(test_accuracy)

def _loop(epochs: int, train_loader: DataLoader, test_loader: DataLoader,
          target: nn.Module, metaopt, 
          opt: torch.optim.Optimizer, step: Callable):
    for epoch in range(epochs):
        log.info(f'Epoch: {epoch}')
        target.train()
        for _, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            step(metaopt, opt)(X, y)

        _test(target, test_loader)


class Target(nn.Module):
    """Target network"""

    def __init__(self):
        super(Target, self).__init__()
        self.fe = SimpleCNN(1, 2)

    def _gen_seq(self, size: int):
        self.seq = nn.Sequential()
        self.seq.append(nn.Linear(size, 10))

    def forward(self, x: torch.Tensor):
        x = self.fe(x)

        if not hasattr(self, 'seq'):
            self._gen_seq(x.shape[0])
        x = self.seq(x)
        x = torch.reshape(x, [1, 10])
        return F.log_softmax(x, dim=1)


##---GWNMO---##

class MetaOptimizer(nn.Module):
    """Gradient weighting network"""

    def __init__(self):
        super(MetaOptimizer, self).__init__()

    def _gen_fe(self, size: int):
        self.fe = GradFeatEx(size, 2)

    def _gen_seq(self, size: int):
        self.seq = nn.Sequential()
        self.seq.append(nn.Linear(size, 128))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.Linear(128, 32))
        self.seq.append(nn.ReLU())

    def _gen_exit(self, size: int):
        self.exit = nn.Sequential()
        self.exit.append(nn.Linear(32, size))
        self.exit.append(nn.ReLU())

    def forward(self, grad, x_embd):
        if not hasattr(self, 'fe'):
            self._gen_fe(grad.shape[0])
        grad_embd = self.fe(grad)

        x = torch.cat([grad_embd, x_embd])

        if not hasattr(self, 'seq'):
            self._gen_seq(x.shape[0])
        x = self.seq(x)

        if not hasattr(self, 'exit'):
            self._gen_exit(grad.shape[0])
        x = self.exit(x)

        return x

def gwnmo(epochs: int, mlr:int):
    target = Target()
    target.to(device)

    metaopt = GWNMO(
        model=target, transform=MetaOptimizer)
    metaopt.to(device)

    loss = torch.nn.NLLLoss()

    train_loader, test_loader = _setup_dataset()

    #Init metaopt parameters
    (X, y) = next(iter(train_loader))
    X, y = X.to(device), y.to(device)
    metaopt.zero_grad()
    err = loss(target(X), y)
    x_embd = target.fe(X).detach()
    err.backward()
    metaopt.step(x_embd)

    opt = torch.optim.Adam(metaopt.parameters(), lr=mlr)

    def _step(metaopt, opt: torch.optim.Optimizer):
        def f(X: torch.Tensor, y: torch.Tensor):
            metaopt.zero_grad()
            opt.zero_grad()
            err = loss(target(X), y)
            x_embd = target.fe(X).detach()
            err.backward()
            opt.step()  # Update metaopt parameters
            metaopt.step(x_embd)  # Update model parameters

    _loop(epochs, train_loader, test_loader, target, metaopt, opt, _step)


##---ADAM---###

def adam(epochs: int, lr: int):
    target = Target()
    target.to(device)

    opt = torch.optim.Adam(target.parameters(), lr=lr)
    loss = torch.nn.NLLLoss()

    train_loader, test_loader = _setup_dataset()

    def _step(_, opt: torch.optim.Optimizer):
        def f(X: torch.Tensor, y: torch.Tensor):
            opt.zero_grad()
            err = loss(target(X), y)
            err.backward()
            opt.step()

    _loop(epochs, train_loader, test_loader, target, None, opt, _step)

##---HyperGrad---###

class HypergradTransform(torch.nn.Module):
    """Hypergradient-style per-parameter learning rates"""

    def __init__(self, param, lr=0.01):
        super(HypergradTransform, self).__init__()
        self.lr = lr * torch.ones_like(param, requires_grad=True)
        self.lr = torch.nn.Parameter(self.lr)

    def forward(self, grad):
        return self.lr * grad


def hypergrad(epochs: int, mlr:int):
    target = Target()
    target.to(device)

    metaopt = l2l.optim.LearnableOptimizer(
        model=target, 
        transform=HypergradTransform
    )
    metaopt.to(device)

    opt = torch.optim.Adam(metaopt.parameters(), lr=mlr)
    loss = torch.nn.NLLLoss()

    train_loader, test_loader = _setup_dataset()

    def _step(metaopt, opt: torch.optim.Optimizer):
        def f(X: torch.Tensor, y: torch.Tensor):
            metaopt.zero_grad()
            opt.zero_grad()
            err = loss(target(X), y)
            err.backward()
            opt.step()
            metaopt.step()

    _loop(epochs, train_loader, test_loader, target, metaopt, opt, _step)
