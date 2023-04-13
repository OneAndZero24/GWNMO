"""
Based on: https://github.com/learnables/learn2learn/blob/master/examples/optimization/hypergrad_mnist.py
"""

import learn2learn as l2l
import neptune
import torch
import torchvision as tv
from rich.progress import Progress
from torch.nn import functional as F
from torch.utils.data import DataLoader

from gwnmo.utils import log
from gwnmo.core import GWNMO

def accuracy(predictions, targets):
    """Returns mean accuracy over a mini-batch"""
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class FeatEx(torch.nn.Module):
    """Image feature extractor"""

    def __init__(self):
        super(FeatEx, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        return torch.flatten(x, 1)


class GradFeatEx(torch.nn.Module):
    """Gradient feature extractor"""

    def __init__(self, size):
        super(GradFeatEx, self).__init__()
        self.fc1 = torch.nn.Linear(size, 1024)
        self.fc2 = torch.nn.Linear(1024, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


class Target(torch.nn.Module):
    """Target network"""

    def __init__(self):
        super(Target, self).__init__()
        self.fe = FeatEx()
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.fe(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MetaOptimizer(torch.nn.Module):
    """Gradient weighting network"""

    def __init__(self):
        super(MetaOptimizer, self).__init__()
        grad_shape = len(list(Target().parameters()))
        log.debug(f'Gradient Shape in Meta Optimizer: {grad_shape}')
        self.fe = GradFeatEx(grad_shape)
        self.fc1 = torch.nn.Linear(9344, 2048)
        self.fc2 = torch.nn.Linear(2048, 256)
        self.fc3 = torch.nn.Linear(256, grad_shape)

    def forward(self, grad, x_embd):
        grad_embd = self.fe(grad)
        x = self.fc1(torch.cat([grad_embd, x_embd]))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x * grad


def main():
    run = neptune.init_run()
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = Target()
    target.to(device)

    metaopt = GWNMO(
        model=target, transform=MetaOptimizer)
    metaopt.to(device)

    opt = torch.optim.Adam(metaopt.parameters(), lr=3e-4)
    loss = torch.nn.NLLLoss()

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(
        tv.datasets.MNIST('~/data', train=True, download=True,
                          transform=tv.transforms.Compose([
                              tv.transforms.ToTensor(),
                              tv.transforms.Normalize((0.1307,), (0.3081,))
                          ])),
        batch_size=32, shuffle=True, **kwargs)
    test_loader = DataLoader(
        tv.datasets.MNIST('~/data', train=False,
                          transform=tv.transforms.Compose([
                              tv.transforms.ToTensor(),
                              tv.transforms.Normalize((0.1307,), (0.3081,))
                          ])),
        batch_size=128, shuffle=False, **kwargs)

    for epoch in range(100):
        log.info(f'Epoch: {epoch}')
        target.train()
        for _, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            metaopt.zero_grad()
            opt.zero_grad()
            err = loss(target(X), y)
            x_embd = target.fe(X).detach()
            err.backward()
            opt.step()  # Update metaopt parameters
            metaopt.step(x_embd)  # Update model parameters

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


if __name__ == '__main__':
    main()
