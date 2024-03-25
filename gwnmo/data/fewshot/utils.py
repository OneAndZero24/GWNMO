import os
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from typing import Any

IDENTITY = lambda x:x

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

class DataManager:
    def __init__(self, image_size: int, n_way: int, n_support: int, n_query: int, transform: Any, root: str, n_episode = 100, download = True, dataset_class = None):        
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.transform = transform 
        self.download = download
        self.root = root
        self.dataset_class = dataset_class

    def get_data_loader(self, type): #parameters that would change on train/val set
        dataset = self.dataset_class(root=self.root, download=self.download, type=type, transform=self.transform, batch_size=self.batch_size)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)  
        self.download = False # download only once

        data_loader_params = dict(batch_sampler = sampler, pin_memory=True)
        data_loader = DataLoader(dataset, **data_loader_params)
        return data_loader