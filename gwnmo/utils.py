import os
import argparse

import torch

from __init__ import __version__
from datasets import *


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

    parser = argparse.ArgumentParser()

    parser.add_argument('epochs', type=int, help='Number of epochs to train for')
    parser.add_argument('--reps', type=int, default=1, required=False, help='Repetitions of training with persistent optimizer state but changing optimizee')
    parser.add_argument('--twostep', action='store_true', help='Optional meta-learning paradigm where one batch is used for fitting and later for validation')
    parser.add_argument("--dataset", choices=['mnist', 'fmnist', 'cifar10', 'svhn'], required=True, default='mnist', help='Dataset selection')

    parser.add_argument("--module", choices=['adam', 'gwnmo', 'hypergrad'], required=True, default='gwnmo', help='Module selection')
    parser.add_argument('--lr', type=float, default=0.01, required=False, help='Learning rate')
    parser.add_argument("--gamma", type=float, default=0.01, required=False, help='Gamma')
    parser.add_argument('--nonorm', action='store_true', help='Normalization in GWNMO')
    parser.add_argument('--noneptune', action='store_true', help='Disable neptune')

    return parser

parser = _setup_arg_parser()    # Global argument parser


logger = None


def _setup_torch():
    """
    Sets `torch` up, returns device
    """

    torch.manual_seed(1)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = _setup_torch()

from modules.adam import Adam
from modules.gwnmo import GWNMO
from modules.hypergrad import HyperGrad

# Maps "selector" arguments to their options handlers
map2cmd = {
    "module": {
        "gwnmo": GWNMO,
        "hypergrad": HyperGrad,
        "adam": Adam,
    },
    "dataset": {
        "mnist": setup_MNIST,
        "fmnist": setup_FMNIST,
        "cifar10": setup_CIFAR10,
        "svhn": setup_SVHN,
    }
}