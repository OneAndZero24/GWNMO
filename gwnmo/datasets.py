import os
import random
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels

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
        batch_size=batch_size, shuffle=True, drop_last=True, **_kwargs
    )
    test_loader = DataLoader(
        datasets.MNIST(DATASET_DIR, train=False, download=True, transform=trans),
        batch_size=batch_size, shuffle=False, drop_last=True, **_kwargs
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
        batch_size=batch_size, shuffle=True, drop_last=True, **_kwargs
    )
    test_loader = DataLoader(
        datasets.FashionMNIST(DATASET_DIR, train=False, download=True, transform=trans),
        batch_size=batch_size, shuffle=False, drop_last=True, **_kwargs
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
        batch_size=batch_size, shuffle=True, drop_last=True, **_kwargs
    )
    test_loader = DataLoader(
        datasets.CIFAR10(DATASET_DIR, train=False, download=True, transform=trans),
        batch_size=batch_size, shuffle=False, drop_last=True, **_kwargs
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
        batch_size=batch_size, shuffle=True, drop_last=True, **_kwargs
    )
    test_loader = DataLoader(
        datasets.SVHN(DATASET_DIR, split='test', download=True, transform=trans),
        batch_size=batch_size, shuffle=False, drop_last=True, **_kwargs
    )

    return (train_loader, test_loader)

OMNIGLOT_CLASSES = 1623

def setup_FS_Omniglot(device, ways: int, shots: int, query: int):
    """
    Returns properly setup Omniglot Taskset:
    `(taskset.train, taskset.test)`
    """

    query = query // ways # ideally query should be multiplicity of ways

    trans = transforms.Compose([       
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ResNet18_Weights.DEFAULT.transforms(antialias=True) 
    ]) 
    dataset = l2l.vision.datasets.FullOmniglot(root=DATASET_DIR, transform=trans, download=True)

    classes = list(range(OMNIGLOT_CLASSES))
    random.shuffle(classes)
    train_dataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[:1100])
    test_dataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[1100:])

    train_fs_trans = [
        NWays(train_dataset, ways), 
        KShots(train_dataset, shots + query),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    test_fs_trans = [
        NWays(test_dataset, ways),
        KShots(test_dataset, shots + query),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]

    train_tasksets = l2l.data.TaskDataset(dataset=train_dataset, task_transforms=train_fs_trans, num_tasks=-1)
    test_tasksets = l2l.data.TaskDataset(dataset=test_dataset, task_transforms=test_fs_trans, num_tasks=-1)

    return (train_tasksets, test_tasksets)