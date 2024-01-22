import argparse
import numpy as np

import torch

from __init__ import __version__
from neptune_logger import NeptuneLogger
import datasets
from models.feature_extractor import feature_extractors
import logging


# Singleton boilerplate
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def normalize_weighting(x, grad):
    """
    Permforms magical normalization described in PDF given meta optimizer's network output and gradient
    """

    temp : torch.Tensor = torch.clamp(x, min=0, max=1)
    selected: torch.Tensor = temp*grad

    return selected*(torch.linalg.norm(grad)/torch.linalg.norm(selected))


def map_classes(ways, y):
    flattened_y = torch.flatten(y)
    classes_list = flattened_y.tolist()

    assert len(set(classes_list)) == ways, f"Invalid number of classes {set(classes_list)}"

    mapping_dict = dict()
    for idx, val in enumerate(set(classes_list)):
        mapping_dict[val] = idx

    return y.apply_(lambda x: mapping_dict[x])


def accuracy(predictions, targets):
    """
    Returns accuracy on batch
    """
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def _setup_arg_parser():
    """
    Minimallistic `argparse` setup/handler
    """

    parser = argparse.ArgumentParser(prog='gwnmo')

    parser.add_argument('epochs', type=int, help='Number of epochs to train for')
    parser.add_argument('--offline', action='store_true', help='No logging to neptune')
    parser.add_argument('--lr', type=float, default=0.01, required=False, help='Learning rate')
    parser.add_argument("--gamma", type=float, default=0.01, required=False, help='Gamma')
    parser.add_argument('--nonorm', action='store_true', help='Normalization in GWNMO/HG in fs & classic')

    subparsers = parser.add_subparsers(title='mode', dest='mode', help='Learning paradigm selection', required=True)
    
    parser_classic = subparsers.add_parser('classic')
    parser_classic.add_argument('--reps', type=int, default=1, required=False, help='Repetitions of training with persistent optimizer state but changing optimizee')
    parser_classic.add_argument("--dataset", choices=['mnist', 'fmnist', 'cifar10', 'svhn'], required=True, default='mnist', help='Dataset selection')
    parser_classic.add_argument("--module", choices=['adam', 'gwnmo', 'hypergrad'], required=True, default='gwnmo', help='Module selection')

    parser_fs = subparsers.add_parser('fewshot')
    parser_fs.add_argument("--dataset", choices=['omniglot'], required=True, default='omniglot', help='Dataset selection')
    parser_fs.add_argument("--module", choices=['gwnmofs', 'maml'], required=True, default='gwnmofs', help='Module selection')
    parser_fs.add_argument('--lr2', type=float, default=0.01, required=False, help='Secondary learning rate')
    parser_fs.add_argument('--ways', type=int, default=5, required=False, help='Number of classes in task')
    parser_fs.add_argument('--shots', type=int, default=1, required=False, help='Number of class examples')
    parser_fs.add_argument('--query', type=int, default=10, required=False, help='Number of samples in query set (ideally multiplicity of n_way)')
    parser_fs.add_argument('--steps', type=int, default=1, required=False, help='Number of adaptation steps')
    parser_fs.add_argument('--tasks', type=int, default=100, required=False, help='Number of tasksets in epoch')
    parser_fs.add_argument('--no_weighting', type=int, default=-1, required=False, help='Number of epochs for wich GWNMOFS weighting will be set to one')
    parser_fs.add_argument('--trainable_fe', type=bool, default=False, required=False, help='Tells if model should use trainable backbone')
    parser_fs.add_argument('--backbone_type', type=str, default="ResNet18", required=False, choices=sorted(feature_extractors.keys()), help='Specifies trainable backbone (used if trainable_fe == True)')

    return parser

parser = _setup_arg_parser()    # Global argument parser


class LoggerHandler(metaclass=Singleton):
    def __init__(self):
        self.logger = NeptuneLogger(False)

    def toggle_online(self):
        self.logger = NeptuneLogger(True)

    def get(self):
        return self.logger

logger = LoggerHandler()


def _setup_torch():
    """
    Sets `torch` up, returns device
    """
    
    torch.autograd.set_detect_anomaly(True)
    
    torch.manual_seed(1)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    logger.get().print_to_term("torch_device", device_name)
    return torch.device(device_name)

device = _setup_torch()


from modules.classic.adam import Adam
from modules.classic.gwnmo import GWNMO
from modules.classic.hypergrad import HyperGrad

from modules.fewshot.gwnmofs import GWNMOFS
from modules.fewshot.maml import MAML

# Maps "selector" arguments to their options handlers
map2cmd = {
    "module": {
        "gwnmo": GWNMO,
        "hypergrad": HyperGrad,
        "adam": Adam,
        "gwnmofs": GWNMOFS,
        "maml": MAML
    },
    "dataset": {
        "mnist": datasets.setup_MNIST,
        "fmnist": datasets.setup_FMNIST,
        "cifar10": datasets.setup_CIFAR10,
        "svhn": datasets.setup_SVHN,
        "omniglot": datasets.experimental_setup_FS_Omniglot
    }
}