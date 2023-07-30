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
    for batch in batches:
        loss1 = 0
        loss2 = 0
        for item in batch:
            support, query = item
            c = target.clone()
            l = c.forward(support) # before adaptation

            gwnmo.step(l) # adapt

            loss1 += c.forward(query) # after adaptation

            params = c.get_params()
            loss2 = target.clone().set_params(params).forward(query)

        loss1 /= len(batch)
        loss2 /= len(batch)
        adam1.step(loss1) # adjust GWNMO
        adam2.step(loss2) # adjust target

# TODO rewrite all no detach