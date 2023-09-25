import argparse
import numpy as np

import torch

from __init__ import __version__
from neptune_logger import NeptuneLogger
from datasets import *


def accuracy(predictions, targets):
    """
    Returns accuracy on batch
    """
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def split_batch(batch, ways: int, shots: int):
    """
    Splits batch into adapt and eval parts.
    Returns dict of tuples (data, labels) indexed by 'adapt', 'eval'.
    """

    X, y = batch
    X, y = X.to(device), y.to(device)

    adapt_indices = np.zeros(X.size(0), dtype=bool)
    adapt_indices[np.arange(shots*ways) * 2] = True
    eval_indices = torch.from_numpy(~adapt_indices)
    adapt_indices = torch.from_numpy(adapt_indices)
    adapt_X, adapt_y = X[adapt_indices], y[adapt_indices]
    eval_X, eval_y = X[eval_indices], y[eval_indices]

    return {
        'adapt': (adapt_X, adapt_y),
        'eval': (eval_X, eval_y)
    }


def _setup_arg_parser():
    """
    Minimallistic `argparse` setup/handler
    """

    parser = argparse.ArgumentParser(prog='gwnmo')

    parser.add_argument('epochs', type=int, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, required=False, help='Learning rate')
    parser.add_argument("--gamma", type=float, default=0.01, required=False, help='Gamma')
    parser.add_argument('--nonorm', action='store_true', help='Normalization in GWNMO')

    subparsers = parser.add_subparsers(title='mode', dest='mode', help='Learning paradigm selection', required=True)
    
    parser_classic = subparsers.add_parser('classic')
    parser_classic.add_argument('--reps', type=int, default=1, required=False, help='Repetitions of training with persistent optimizer state but changing optimizee')
    parser_classic.add_argument("--dataset", choices=['mnist', 'fmnist', 'cifar10', 'svhn'], required=True, default='mnist', help='Dataset selection')
    parser_classic.add_argument("--module", choices=['adam', 'gwnmo', 'hypergrad'], required=True, default='gwnmo', help='Module selection')

    parser_fs = subparsers.add_parser('fewshot')
    parser_fs.add_argument("--dataset", choices=['omniglot'], required=True, default='omniglot', help='Dataset selection')
    parser_fs.add_argument("--module", choices=['gwnmofs', 'metasgd', 'maml'], required=True, default='gwnmofs', help='Module selection')
    parser_fs.add_argument('--lr2', type=float, default=0.01, required=False, help='Secondary learning rate')
    parser_fs.add_argument('--ways', type=int, default=15, required=False, help='Number of classes in task')
    parser_fs.add_argument('--shots', type=int, default=1, required=False, help='Number of class examples')
    parser_fs.add_argument('--steps', type=int, default=1, required=False, help='Number of adaptation steps.')

    return parser

parser = _setup_arg_parser()    # Global argument parser


logger = NeptuneLogger()


def _setup_torch():
    """
    Sets `torch` up, returns device
    """

    torch.manual_seed(1)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = _setup_torch()


from modules.classic.adam import Adam
from modules.classic.gwnmo import GWNMO
from modules.classic.hypergrad import HyperGrad

from modules.fewshot.gwnmofs import GWNMOFS

# Maps "selector" arguments to their options handlers
map2cmd = {
    "module": {
        "gwnmo": GWNMO,
        "hypergrad": HyperGrad,
        "adam": Adam,
        "gwnmofs": GWNMOFS
    },
    "dataset": {
        "mnist": setup_MNIST,
        "fmnist": setup_FMNIST,
        "cifar10": setup_CIFAR10,
        "svhn": setup_SVHN,
        "omniglot": setup_FS_Omniglot
    }
}