import argparse
import logging

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


log = _setup_logger("DEBUG")


def _setup_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs', type=int, help='Number of epochs to train for')
    return parser


parser = _setup_arg_parser()