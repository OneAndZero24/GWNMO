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

from core import GWNMO
from models.grad_fe import GradFeatEx
from utils import log, accuracy, device, run

def _setup_dataset(batch_size: int = 32):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        tv.models.ResNet18_Weights.DEFAULT.transforms(antialias=True), 
    ])
    train_loader = DataLoader(
        tv.datasets.MNIST('/shared/sets/datasets', train=True, download=True, transform=transforms),
        batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = DataLoader(
        tv.datasets.MNIST('/shared/sets/datasets', train=False, transform=transforms),
        batch_size=batch_size, shuffle=False, **kwargs
    )
    return (train_loader, test_loader)

def _test(target: nn.Module, test_loader: DataLoader):
    target.eval()
    loss = torch.nn.NLLLoss()
    test_error = 0.0
    test_accuracy = 0.0
    with torch.no_grad():
        for _, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            preds = target(torch.reshape(FE(X), (-1, 512)))
            test_error += loss(preds, y)
            test_accuracy += accuracy(preds, y)
        test_error /= len(test_loader)
        test_accuracy /= len(test_loader)
    log.info(f'Test Accuracy: {test_accuracy}')
    log.info(f'Test Loss: {test_error}')
    run["accuracy"].append(test_accuracy)
    run["loss"].append(test_error)

def _loop(epochs: int, train_loader: DataLoader, test_loader: DataLoader,
          target: nn.Module, metaopt, 
          opt: torch.optim.Optimizer, step: Callable):
    f_step = step(metaopt, opt)  
    for epoch in range(epochs):
        log.info(f'Epoch: {epoch}')
        target.train()
        for _, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            f_step(X, y)

        _test(target, test_loader)

def _twostep_loop(epochs: int, train_loader: DataLoader, test_loader: DataLoader,
          target: nn.Module, metaopt, 
          opt: torch.optim.Optimizer, step: Callable):
    f_step = step(metaopt, opt)  
    for epoch in range(epochs):
        log.info(f'Epoch: {epoch}')
        target.train()
        enum = enumerate(train_loader)
        _, (lastX, lasty) = next(enum)
        lastX.to(device)
        lasty.to(device)
        for _, (X, y) in enum:
            X, y = X.to(device), y.to(device)
            f_step(lastX, lasty, X, y)
            lastX = X
            lasty = y

        _test(target, test_loader)


# Pretrained Feature Extractor - Resnet18
FE = nn.Sequential(*list(tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT).children())[:-1])
for param in FE.parameters():
    param.requires_grad = False
FE.to(device)

class Target(nn.Module):
    """Target network, takes x after FE"""

    def __init__(self):
        super(Target, self).__init__()

        self.seq = nn.Sequential()
        self.seq.append(nn.Linear(512, 10))

    def forward(self, x: torch.Tensor):
        x = self.seq(x)
        return F.log_softmax(x, dim=1)


##---GWNMO---##
class MetaOptimizer(nn.Module):
    """Gradient weighting network"""

    def __init__(self):
        super(MetaOptimizer, self).__init__()
        self.seq = nn.Sequential()
        self.seq.append(nn.Linear(26644, 128))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.Linear(128, 32))
        self.seq.append(nn.ReLU())
        self.seq.append(nn.Linear(32, 5130))
        self.seq.append(nn.ReLU())

    def forward(self, params, grad, x_embd):
        x = torch.cat([params, grad, x_embd.flatten()])
        return self.seq(x)

def gwnmo(epochs: int, mlr:float, gm:float, reps: int = 1, twostep: bool = False):
    run["sys/tags"].add(['gwnmo', f'lr={mlr}', f'gm={gm}', f'reps={reps}'])
    transform = None

    for I in range(reps):
        target = Target()
        target.to(device)

        metaopt = GWNMO(
            model=target, transform=MetaOptimizer, gamma=gm)
        if transform is not None:
            metaopt.transform = transform
        metaopt.to(device)

        loss = torch.nn.NLLLoss()

        train_loader, test_loader = _setup_dataset()

        opt = torch.optim.Adam(metaopt.parameters(), lr=mlr)

        if twostep:
            def _twostep_step(metaopt, opt: torch.optim.Optimizer):
                def f(lastX: torch.Tensor, lasty: torch.Tensor, X: torch.Tensor, y: torch.Tensor):
                    metaopt.zero_grad()
                    x_embd = torch.reshape(FE(lastX), (-1, 512))
                    err = loss(target(x_embd), lasty)
                    err.backward()
                    metaopt.step(x_embd)  # Update model parameters

                    opt.zero_grad()
                    x_embd = torch.reshape(FE(X), (-1, 512))
                    err = loss(target(x_embd), y)
                    err.backward()
                    opt.step()  # Update metaopt parameters
                return f

            _twostep_loop(epochs, train_loader, test_loader, target, metaopt, opt, _twostep_step)
        else:
            def _step(metaopt, opt: torch.optim.Optimizer):
                def f(X: torch.Tensor, y: torch.Tensor):
                    metaopt.zero_grad()
                    opt.zero_grad()
                    x_embd = torch.reshape(FE(X), (-1, 512))
                    err = loss(target(x_embd), y)
                    err.backward()
                    opt.step()  # Update metaopt parameters
                    metaopt.step(x_embd)  # Update model parameters
                return f

            _loop(epochs, train_loader, test_loader, target, metaopt, opt, _step)
        transform = metaopt.transform


##---ADAM---###

def adam(epochs: int, lr: int, reps: int = 1):
    run["sys/tags"].add(['adam', f'lr={lr}', f'reps={reps}'])
    state = None

    for I in range(reps):
        target = Target()
        target.to(device)

        opt = torch.optim.Adam(target.parameters(), lr=lr)
        if state is not None:
            opt.load_state_dict(state)

        loss = torch.nn.NLLLoss()

        train_loader, test_loader = _setup_dataset()

        def _step(_, opt: torch.optim.Optimizer):
            def f(X: torch.Tensor, y: torch.Tensor):
                opt.zero_grad()
                x_embd = torch.reshape(FE(X), (-1, 512))
                err = loss(target(x_embd), y)
                err.backward()
                opt.step()
            return f

        _loop(epochs, train_loader, test_loader, target, None, opt, _step)
        state = opt.state_dict()

##---HyperGrad---###

class HypergradTransform(torch.nn.Module):
    """Hypergradient-style per-parameter learning rates"""

    def __init__(self, param, lr=0.01):
        super(HypergradTransform, self).__init__()
        self.lr = lr * torch.ones_like(param, requires_grad=True)
        self.lr = torch.nn.Parameter(self.lr)

    def forward(self, grad):
        return self.lr * grad


def hypergrad(epochs: int, mlr:int, gm:float, reps: int = 1, twostep: bool = False):
    run["sys/tags"].add(['hypergrad', f'lr={mlr}', f'gm={gm}', f'reps={reps}'])
    transforms = None

    for I in range(reps):
        target = Target()
        target.to(device)

        metaopt = l2l.optim.LearnableOptimizer(
            model=target, 
            transform=HypergradTransform,
            lr = gm
        )
        if transforms is not None:
            metaopt.transforms = transforms
        metaopt.to(device)

        opt = torch.optim.Adam(metaopt.parameters(), lr=mlr)
        loss = torch.nn.NLLLoss()

        train_loader, test_loader = _setup_dataset()

        if twostep:
            def _twostep_step(metaopt, opt: torch.optim.Optimizer):
                def f(lastX: torch.Tensor, lasty: torch.Tensor, X: torch.Tensor, y: torch.Tensor):
                    metaopt.zero_grad()
                    x_embd = torch.reshape(FE(lastX), (-1, 512))
                    err = loss(target(x_embd), lasty)
                    err.backward()
                    metaopt.step(x_embd)  # Update model parameters

                    opt.zero_grad()
                    x_embd = torch.reshape(FE(X), (-1, 512))
                    err = loss(target(x_embd), y)
                    err.backward()
                    opt.step()  # Update metaopt parameters
                return f

            _twostep_loop(epochs, train_loader, test_loader, target, metaopt, opt, _twostep_step)
        else:
            def _step(metaopt, opt: torch.optim.Optimizer):
                def f(X: torch.Tensor, y: torch.Tensor):
                    metaopt.zero_grad()
                    opt.zero_grad()
                    x_embd = torch.reshape(FE(X), (-1, 512))
                    err = loss(target(x_embd), y)
                    err.backward()
                    opt.step()  # Update metaopt parameters
                    metaopt.step()   # Update model parameters
                return f

            _loop(epochs, train_loader, test_loader, target, metaopt, opt, _step)
        transforms = metaopt.transforms