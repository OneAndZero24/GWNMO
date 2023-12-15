import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from abc import abstractmethod
from PIL import Image
import json

IDENTITY = lambda x:x

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

def setup_FS_Omniglot(device, ways: int, shots: int, query: int, tasks: int):
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

    train_tasksets = l2l.data.TaskDataset(dataset=train_dataset, task_transforms=train_fs_trans, num_tasks=tasks)
    test_tasksets = l2l.data.TaskDataset(dataset=test_dataset, task_transforms=test_fs_trans, num_tasks=tasks)

    return (train_tasksets, test_tasksets)


# HELPERS

# Dataset helpers 

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=IDENTITY):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        image_path = os.path.join( self.sub_meta[i])
        img = Image.open(image_path, mode="r").convert("L")
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class SetDataset:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  pin_memory = False)

        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform)
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

# Data Managers
class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_eposide = 100):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.transform = transforms.Compose([       
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ResNet18_Weights.DEFAULT.transforms(antialias=True) 
        ])  

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        dataset = SetDataset( data_file , self.batch_size, self.transform )
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  

        data_loader_params = dict(batch_sampler = sampler, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


def experimental_setup_FS_Omniglot(device, ways, shots, query, tasks):
    image_size = 224
    path_base = os.environ('OMNIGLOT_PATH')
    train_file = os.path.join(path_base, 'base.json')
    test_file = os.path.join(path_base, 'val.json')


    train_data_manager = SetDataManager(image_size, **dict(n_way=ways, n_support=shots, n_query=query))
    test_data_manager = SetDataManager(image_size, **dict(n_way=ways, n_support=shots, n_query=query))

    train_loader = train_data_manager.get_data_loader(train_file, aug=False)
    test_loader = test_data_manager.get_data_loader(test_file, aug=False)

    return train_loader, test_loader
