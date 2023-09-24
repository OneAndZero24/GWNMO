import torch
import learn2learn as l2l

from modules.fewshot.fsmodule_abc import FSModuleABC
from utils import device, logger, accuracy


def test(module: FSModuleABC, test_loader, epoch: int):
    """
    Test target on whole test
    """

    # TODO implement


def train(dataset_gen, epochs: int, reps: int, adapt_steps: int, 
          batch_size: int, ways: int, shots: int, module: FSModuleABC):
    """
    FewShot training flow
    """

    # TODO loop
    