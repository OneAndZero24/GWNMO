import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from modules.module_abc import ModuleABC
from utils import logger, device, accuracy

def test(module: ModuleABC, test_loader: DataLoader):
    """
    Test target on whole test
    """

    module.target.eval()

    test_error = 0.0
    test_accuracy = 0.0

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            _, preds, err = module.training_step(X, y, i)
            test_error += err
            test_accuracy += accuracy(preds, y)
        test_error /= len(test_loader)
        test_accuracy /= len(test_loader)
    logger.experiment["accuracy"].log(test_accuracy)
    logger.experiment["loss"].log(test_error)


def train(dataset, epochs: int, reps: int, module: ModuleABC):
    """
    Normal training flow.
    - Gets batch
    - Puts data through target
    - Adjusts target
    - Adjusts weighting (meta-optimizer)
    """

    train_loader, test_loader = dataset

    state = None
    for _ in range(reps):
        opts = module.configure_optimizers()

        module.reset_target()
        module.target.to(device)

        if state is not None:
            module.set_state(state)

        if len(opts) > 1:
            opts[0].to(device)
            
        for _ in range(epochs):
            module.target.train()
            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                
                for opt in opts.reverse():
                    opt.zero_grad()

                x_embd, _, err = module.training_step((X, y), i)
                err.backward()

                if len(opts) > 1:
                    opts[0].step(x_embd)
                opts[-1].step()
            
            test(module, test_loader)

        state = module.get_state(opts[0])

def train_twostep(dataset, epochs: int, reps: int, module: ModuleABC):
    """
    Two batch meta-learning flow.
    Note: Supports only `gwnmo` & `hypergrad`
    - Gets two batches
    - Puts first through target
    - Adjusts target
    - Puts second through target
    - Adjusts weighting (meta-optimizer) based on second loss
    """

    train_loader, test_loader = dataset

    state = None
    for _ in range(reps):
        opts = module.configure_optimizers()

        module.reset_target()
        module.target.to(device)

        if state is not None:
            module.set_state(state)

        opts[0].to(device)
        opt = opts[0]
        metaopt = opts[1]
            
        for _ in range(epochs):
            module.target.train()
            batches = enumerate(train_loader)

            lasti, (lastX, lasty) = next(batches)
            lastX, lasty = lastX.to(device), lasty.to(device)

            for i, (X, y) in enumerate(batches):
                X, y = X.to(device), y.to(device)
                
                # BATCH 1
                opt.zero_grad()

                x_embd, _, err = module.training_step((lastX, lasty), lasti)
                err.backward()

                opt.step(x_embd)

                # BATCH 2
                metaopt.zero_grad()

                x_embd, _, err = module.training_step((X, y), i)
                err.backward()

                metaopt.step()

                lastX, lasty = X, y
            
            test(module, test_loader)

