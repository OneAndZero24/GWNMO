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
from .utils import DataManager, SubDataset
# TO DO: Add method for checking integrity with md5 hashes
DATASET_DIR = os.getenv('DATASET_DIR')

class FSOmniglot(VisionDataset):
    folder = 'fs-omniglot'

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
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform)
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

# tasks here is equivalent of episodes
def experimental_setup_FS_Omniglot(device, ways, shots, query, tasks):
    image_size = 224
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ResNet18_Weights.DEFAULT.transforms(antialias=True) 
    ]) 
    
    omniglot_data_manager = DataManager(
        image_size=image_size, 
        n_way=ways, 
        n_support=shots, 
        n_query=query, 
        n_episode=tasks, 
        download=True, 
        transform=transform, 
        root=DATASET_DIR,
        dataset_class=FSOmniglot,
    )

    train_loader = omniglot_data_manager.get_data_loader(type='base')
    test_loader = omniglot_data_manager.get_data_loader(type='val')
    
    return train_loader, test_loader