import os
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights


_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
DATASET_DIR = os.getenv('DATASET_DIR')
if DATASET_DIR is None:
    DATASET_DIR = '/shared/sets/datasets'

def setup_MNIST(batch_size: int = 32):
    """
    Returns properly setup MNIST Dataloader:
    `(train_loader, test_loader)`
    """

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ResNet18_Weights.DEFAULT.transforms(antialias=True), 
    ])
    train_loader = DataLoader(
        datasets.MNIST(DATASET_DIR, train=True, download=True, transform=trans),
        batch_size=batch_size, shuffle=True, **_kwargs
    )
    test_loader = DataLoader(
        datasets.MNIST(DATASET_DIR, train=False, transform=trans),
        batch_size=batch_size, shuffle=False, **_kwargs
    )

    return (train_loader, test_loader)

def setup_FMNIST(batch_size: int = 32):
    """
    Returns properly setup FashionMNIST Dataloader:
    `(train_loader, test_loader)`
    """

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ResNet18_Weights.DEFAULT.transforms(antialias=True), 
    ])
    train_loader = DataLoader(
        datasets.FashionMNIST(DATASET_DIR, train=True, download=True, transform=trans),
        batch_size=batch_size, shuffle=True, **_kwargs
    )
    test_loader = DataLoader(
        datasets.FashionMNIST(DATASET_DIR, train=False, transform=trans),
        batch_size=batch_size, shuffle=False, **_kwargs
    )

    return (train_loader, test_loader)

def setup_CIFAR10(batch_size: int = 32):
    """
    Returns properly setup CIFAR Dataloader:
    `(train_loader, test_loader)`
    """

    trans = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        ResNet18_Weights.DEFAULT.transforms(antialias=True), 
    ])
    train_loader = DataLoader(
        datasets.CIFAR10(DATASET_DIR, train=True, download=True, transform=trans),
        batch_size=batch_size, shuffle=True, **_kwargs
    )
    test_loader = DataLoader(
        datasets.CIFAR10(DATASET_DIR, train=False, transform=trans),
        batch_size=batch_size, shuffle=False, **_kwargs
    )

    return (train_loader, test_loader)

def setup_SVHN(batch_size: int = 32):
    """
    Returns properly setup SVHN Dataloader:
    `(train_loader, test_loader)`
    """

    trans = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        ResNet18_Weights.DEFAULT.transforms(antialias=True), 
    ])
    train_loader = DataLoader(
        datasets.SVHN(DATASET_DIR, split='train', download=True, transform=trans),
        batch_size=batch_size, shuffle=True, **_kwargs
    )
    test_loader = DataLoader(
        datasets.SVHN(DATASET_DIR, split='test', transform=trans),
        batch_size=batch_size, shuffle=False, **_kwargs
    )

    return (train_loader, test_loader)