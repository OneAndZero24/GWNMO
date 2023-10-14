import os
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

def setup_FS_Omniglot(device, ways: int, shots: int):
    """
    Returns properly setup Omniglot Taskset:
    `(taskset.train, taskset.test)`
    """

    trans = transforms.Compose([        
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.ToTensor(),
        ResNet18_Weights.DEFAULT.transforms(antialias=True) 
    ]) 
    dataset = l2l.vision.datasets.FullOmniglot(root=DATASET_DIR, transform=trans, download=True)

    if device is not None:
        metadataset = l2l.data.OnDeviceDataset(dataset, device=device)
    metadataset = l2l.data.MetaDataset(dataset)

    fs_trans = [
        NWays(metadataset, ways),
        KShots(metadataset, shots),
        LoadData(metadataset),
        ConsecutiveLabels(metadataset),
    ]
    tasksets = l2l.data.TaskDataset(dataset=metadataset, task_transforms=fs_trans, num_tasks=-1)

    # TODO
    # classes = list(range(1623))
    # random.shuffle(classes)
    # train_dataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[:1100])
    # validation_datatset = l2l.data.FilteredMetaDataset(dataset, labels=classes[1100:1200])
    # test_dataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[1200:])

    # train_transforms = [
    #     l2l.data.transforms.FusedNWaysKShots(dataset,
    #                                          n=train_ways,
    #                                          k=train_samples),
    #     l2l.data.transforms.LoadData(dataset),
    #     l2l.data.transforms.RemapLabels(dataset),
    #     l2l.data.transforms.ConsecutiveLabels(dataset),
    #     l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    # ]
    # test_transforms = [
    #     l2l.data.transforms.FusedNWaysKShots(dataset,
    #                                          n=test_ways,
    #                                          k=test_samples),
    #     l2l.data.transforms.LoadData(dataset),
    #     l2l.data.transforms.RemapLabels(dataset),
    #     l2l.data.transforms.ConsecutiveLabels(dataset),
    #     l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    # ]

    # _datasets = (train_dataset, validation_datatset, test_dataset)
    # _transforms = (train_transforms, validation_transforms, test_transforms)
    # return _datasets, _transforms

    return (tasksets.train, tasksets.test)