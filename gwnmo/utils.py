import argparse
import logging

import neptune
import torch
from rich.logging import RichHandler


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def _setup_logger(log_level: str):
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True,
                              tracebacks_suppress=[torch])]
    )

    return logging.getLogger("rich")

log = _setup_logger("INFO")


def _setup_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs', type=int, help='Number of epochs to train for')
    parser.add_argument("--exp", choices=['mnist.gwnmo', 'mnist.adam', 'mnist.hypergrad'], required=True, default="M", help='Experiment selection')
    parser.add_argument('--lr', type=float, default=0.01, required=False, help='Meta optimizer learning rate')
    parser.add_argument("--gamma", type=float, default=0.01, required=False, help='Gamma for metopts')
    parser.add_argument('--reps', type=int, default=1, required=False, help='External loop')
    parser.add_argument('--twostep', type=bool, action='store_true', help='Optional learning method for HG methods')
    return parser

parser = _setup_arg_parser()


run = neptune.init_run()


def _setup_torch():
    torch.manual_seed(1)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = _setup_torch()