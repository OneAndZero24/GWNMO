from torchvision.datasets import VisionDataset
import os
from typing import Callable, Optional, Union, Literal, Any
import requests
import zipfile
import io
import shutil
import glob
from os.path import isfile, isdir, join
from os import listdir
from PIL import Image
import random
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
# TO DO: Add method for checking integrity with md5 hashes

DATASET_DIR = os.getenv('DATASET_DIR')
if DATASET_DIR is None:
    DATASET_DIR = '/shared/sets/datasets'

IDENTITY = lambda x:x

class FSOmniglotSubDataset:
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

class FSOmniglot(VisionDataset):
    folder = 'omniglot-fewshot'

    train_url = 'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/train.txt'
    val_url = 'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/val.txt'
    test_url = 'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/test.txt'

    background_url = 'https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip?raw=true'
    evaluation_url = 'https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip?raw=true'

    def __init__(self, 
                 root: str, 
                 type: Union[Literal['base'], Literal['val'], Literal['novel']] = 'base',
                 batch_size: int = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ):
        super().__init__(os.path.join(root, self.folder), transform=transform, target_transform=target_transform)
        self.images_path = os.path.join(self.root, 'images')

        if download:
            self.download()
            # prepare dataset
            self._rotate_images()
            self._write_omniglot_filelist()
            self._write_cross_char_base_filelist()

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
            sub_dataset = FSOmniglotSubDataset(self.sub_meta[cl], cl, transform = transform)
            self.sub_dataloader.append(DataLoader(sub_dataset, **sub_data_loader_params))
        
    def _get_and_extract_zip(self, url):
        images_path = os.path.join(self.root, 'images')

        request = requests.get(url, stream=True)
        zip = zipfile.ZipFile(io.BytesIO(request.content))

        zip.extractall(images_path) 

    def _delete_parent_folder(self, name):
        file_paths = glob.glob(os.path.join(self.images_path, name) + '/*')

        for path in file_paths:
            shutil.move(path, self.images_path)

        os.rmdir(os.path.join(self.images_path, name))

    def _save_text_files(self):
        
        text_data_dict = {
            'train.txt': requests.get(self.train_url),
            'test.txt': requests.get(self.test_url),
            'val.txt': requests.get(self.val_url),
        }

        for filename in text_data_dict.keys():
            with open(os.path.join(self.root, filename), 'w') as f:
                f.write(text_data_dict[filename].text)

    def _get_language_directories(self):
        language_directories = [join(self.images_path, dir) for dir in listdir(self.images_path) if isdir(join(self.images_path, dir))]
        language_directories.sort()
        return language_directories

    def _rotate_images(self):
        language_directories = self._get_language_directories()

        for dir in language_directories:
            character_directories = [join(dir, character_directory) for character_directory in listdir(dir) if isdir(join(dir, character_directory))] 
            for character_directory in character_directories:
                image_list = [img for img in listdir(character_directory) if (isfile(join(character_directory,img)) and img[0] != '.')]

                for deg in [0, 90, 180, 270]:
                    rotation_string = "rot%03d"%deg
                    rotation_dir_path = join(character_directory, rotation_string)

                    if not os.path.exists(rotation_dir_path):
                        os.makedirs(rotation_dir_path)
                    for img in image_list:
                        rot_img = Image.open(join(character_directory,img)).rotate(deg)
                        rot_img.save(join(rotation_dir_path,img))

    def _write_omniglot_filelist(self):
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
            with open(join(self.root,datasets[dataset]) + '.txt', 'r') as lines:
                for line in lines:
                    label = line.replace('\n', '')
                    directory_list.append(label)
                    filelists[dataset][label] = [join(self.images_path, label, f) for f in listdir(join(self.images_path, label))]
            for cl, (_, filelist) in enumerate(filelists[dataset].items()):
                random.shuffle(filelist)
                flattened_filelists[dataset] += filelist
                flattened_labels[dataset] += np.repeat(cl, len(filelist)).tolist() 
        
        for dataset in datasets.keys():
            dataset_metadata = {
                'label_names': directory_list,
                'image_names': flattened_filelists[dataset],
                'image_labels': flattened_labels[dataset],
            }

            with open(join(self.root,dataset) + '.json', 'w') as f:
                json.dump(dataset_metadata, f)


    def _write_cross_char_base_filelist(self):
        language_directories = self._get_language_directories()

        filelists = {}
        directory_list = []

        for dir in language_directories:
            if dir == 'Latin':
                continue
            character_directories = [join(dir, character_directory) for character_directory in listdir(dir) if isdir(join(dir, character_directory))] 
            for character_directory in character_directories:
                label = character_directory[len(self.images_path):]
                directory_list.append(label)
                filelists[label] = [join(character_directory, img) for img in listdir(character_directory) if (isfile(join(character_directory,img)) and img[-3:] == 'png')]
        
        flattened_labels = []
        flattened_filelists = []

        for cl, (_, filelist) in enumerate(filelists.items()):
            random.shuffle(filelist)
            flattened_filelists += filelists
            flattened_labels += np.repeat(cl, len(filelist)).tolist()

        dataset_metadata = {
            'label_names': directory_list,
            'image_names': flattened_filelists,
            'image_labels': flattened_labels,
        }

        with open(join(self.root,'noLatin.json'), 'w') as f:
            json.dump(dataset_metadata, f)

    def download(self):

        # assure that root directory and images directory exist
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        
            if not os.path.exists(self.images_path):
                os.mkdir(self.images_path)

            self._get_and_extract_zip(self.background_url)
            self._get_and_extract_zip(self.evaluation_url)

            self._delete_parent_folder('images_background')
            self._delete_parent_folder('images_evaluation')

            self._save_text_files()

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)
    

# HELPER CLASSES
    
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
    
class FSOmniglotDataManager:
    def __init__(self, image_size: int, n_way: int, n_support: int, n_query: int, transform: Any, root: str, n_episode = 100, download = True):        
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.transform = transform 
        self.download = download
        self.root = root

    def get_data_loader(self, type): #parameters that would change on train/val set
        dataset = FSOmniglot(root=self.root, download=self.download, type=type, transform=self.transform, batch_size=self.batch_size)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)  
        self.download = False # download only once

        data_loader_params = dict(batch_sampler = sampler, pin_memory=True)
        data_loader = DataLoader(dataset, **data_loader_params)
        return data_loader

# tasks here is equivalent of episodes
def experimental_setup_FS_Omniglot(device, ways, shots, query, tasks):
    image_size = 224
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ResNet18_Weights.DEFAULT.transforms(antialias=True) 
    ]) 
    
    omniglot_data_manager = FSOmniglotDataManager(
        image_size=image_size, 
        n_way=ways, 
        n_support=shots, 
        n_query=query, 
        n_episode=tasks, 
        download=True, 
        transform=transform, 
        root=DATASET_DIR
    )

    train_loader = omniglot_data_manager.get_data_loader(type='base')
    test_loader = omniglot_data_manager.get_data_loader(type='val')
    
    return train_loader, test_loader