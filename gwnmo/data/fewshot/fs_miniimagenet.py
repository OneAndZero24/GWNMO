from torchvision.datasets import VisionDataset
import os
from typing import Callable, Optional, Union, Literal, Any
from os.path import join
from .utils import SubDataset
from torch.utils.data import DataLoader
import numpy as np
import json
import requests
import re
from os import listdir
import random
from .utils import DataManager
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights

# Since access to miniImagenet is restricted we use dataset from our server.
# This directory should not be modified so we define another SAVE_DIR path to store {train|val|test}.json files there.
DATASET_DIR = os.getenv('DATASET_DIR')
SAVE_DIR = os.getenv('SAVE_DIR') # This should be treated as root of dataset

class FSMiniImagenet(VisionDataset):
    folder = 'miniImagenet-fewshot'

    train_url = 'https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/train.csv'
    val_url = 'https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/val.csv'
    test_url = 'https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/test.csv'

    def __init__(self, 
                 root: str,
                 type: Union[Literal['base'], Literal['val'], Literal['novel']] = 'base',
                 batch_size: int = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download = True,
                 ):
        super().__init__(os.path.join(root, self.folder), transform=transform, target_transform=target_transform)

        # `download` is a bit misused name here. In fact, this part of code prepares {train|val|test}.json files and downloads only csv files.
        # Access to miniImageNet is restricted so we do not expose download here.
        if download:
            self._save_csv_files()
            self._write_miniImagenet_filelist(cross=False)
            self._write_miniImagenet_filelist(cross=True)

        self.type = type
        assert batch_size is not None
        self.batch_size = batch_size

        # setup dataset
        self.data_file = join(self.root, self.type) + '.json'

        with open(self.data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()
        self.sub_meta = {}

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(
            batch_size = batch_size,
            shuffle = True,
            pin_memory = False
        )

        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform)
            self.sub_dataloader.append(DataLoader(sub_dataset, **sub_data_loader_params))

    def _save_csv_files(self):
        text_data_dict = {
            'train.csv': requests.get(self.train_url),
            'test.csv': requests.get(self.test_url),
            'val.csv': requests.get(self.val_url),
        }

        for filename in text_data_dict.keys():
            with open(os.path.join(self.root, filename), 'w') as f:
                f.write(text_data_dict[filename].text)
    
    def _write_miniImagenet_filelist(self, cross=False):
        datasets = {
            'base': 'train',
            'val': 'val',
            'novel': 'test',
        }

        filelists = {
            'base': {},
            'val': {},
            'novel': {},
        }

        flattened_filelists = {
            'base': [],
            'val': [],
            'novel': [], 
        }

        flattened_labels = {
            'base': [],
            'val': [],
            'novel': [],  
        }

        directory_list = []

        for dataset in datasets.keys():
            with open(join(self.root,datasets[dataset]) + '.csv', 'r') as lines:
                for i, line in enumerate(lines):
                    if i == 0:
                        continue
                    fid, _ , label = re.split(',|\.', line)
                    label = label.replace('\n','')

                    if not label in filelists[dataset]:
                        directory_list.append(label)
                        filelists[dataset][label] = []
                        filenames = listdir(join(DATASET_DIR, label))
                        filenames_numbers = [int(re.split('_|\.', fname)[1]) for fname in filenames]
                        sorted_filenames = list(zip( *sorted(  zip(filenames, filenames_numbers), key = lambda file: file[1] )))[0]

                    fid = int(fid[-5:])-1
                    fname = join(DATASET_DIR,label, sorted_filenames[fid])
                    filelists[dataset][label].append(fname)

            for cl, (key, filelist) in enumerate(filelists[dataset].items()):
                random.shuffle(filelists)
                flattened_filelists += filelists
                flattened_labels += np.repeat(cl, len(filelist)).tolist() 

        if cross:
            with open(join(self.root,'all.json'), 'w') as f:
                json.dump({
                    'label_names': directory_list,
                    'image_names': [image for _, filesList in flattened_filelists.items() for image in filesList],
                    'image_labels': [label for _, labelsList in flattened_labels.items() for label in labelsList],
                }, f) 
        else:
            for dataset in datasets.keys():
                with open(join(self.root,dataset) + '.json', 'w') as f:
                    json.dump({
                    'label_names': directory_list,
                    'image_names': flattened_filelists[dataset],
                    'image_labels': flattened_labels[dataset],
                }, f)
                    
# tasks here is equivalent of episodes
def experimental_setup_FS_MiniImagenet(device, ways, shots, query, tasks):
    image_size = 224
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ResNet18_Weights.DEFAULT.transforms(antialias=True) 
    ]) 
    
    miniimagenet_data_manager = DataManager(
        image_size=image_size, 
        n_way=ways, 
        n_support=shots, 
        n_query=query, 
        n_episode=tasks, 
        download=True, 
        transform=transform, 
        root=SAVE_DIR,
        dataset_class=FSMiniImagenet,
    )

    train_loader = miniimagenet_data_manager.get_data_loader(type='base')
    test_loader = miniimagenet_data_manager.get_data_loader(type='val')
    
    return train_loader, test_loader