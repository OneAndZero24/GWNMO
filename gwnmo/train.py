import torch

import lightning as L

from modules.module_abc import ModuleABC
from utils import logger, accuracy

def _setup_fabric():
    """
    Sets-up & returns lightning fabric
    """

    fabric = L.Fabric(loggers=logger)
    fabric.launch()
    return fabric

fabric = _setup_fabric()


def test(module: ModuleABC, test_loader):
    """
    Test target on whole test
    """

    module.target.eval()

    test_error = 0.0
    test_accuracy = 0.0

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            _, preds, err = module.training_step(X, y, i)
            test_error += err
            test_accuracy += accuracy(preds, y)
        test_error /= len(test_loader)
        test_accuracy /= len(test_loader)
    if logger is not None:
        logger.log_metrics({
            "accuracy": test_accuracy,
            "loss": test_error
            })


def train(dataset, epochs: int, reps: int, module: ModuleABC):
    """
    Normal training flow.
    - Gets batch
    - Puts data through target
    - Adjusts target
    - Adjusts weighting (meta-optimizer)
    """

    train_loader, test_loader = dataset
    train_loader = fabric.setup_dataloaders(train_loader)
    test_loader = fabric.setup_dataloaders(test_loader)

    state = None
    for _ in range(reps):
        opts = module.configure_optimizers()

        opt = opts[0]
        metaopt = None
        if len(opts) > 1:
            metaopt = opts[1]

        if state is not None:
            module.set_state(state)

        module.reset_target()

        module.target, opt = fabric.setup(module.target, opt)
        if metaopt is not None:
            opt, metaopt = fabric.setup(opt, metaopt)
            
        for _ in range(epochs):
            module.target.train()
            for i, (X, y) in enumerate(train_loader):

                if metaopt is not None:
                    metaopt.zero_grad()
                opt.zero_grad()

                x_embd, _, err = module.training_step((X, y), i)
                fabric.backward(err)

                if metaopt is not None:
                    opt.step(x_embd)
                else:
                    opt.step()
                metaopt.step()
            
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
    train_loader = fabric.setup_dataloaders(train_loader)
    test_loader = fabric.setup_dataloaders(test_loader)

    state = None
    for _ in range(reps):
        opts = module.configure_optimizers()

        opt = opts[0]
        metaopt = opts[1]

        if state is not None:
            module.set_state(state)

        module.reset_target()

        module.target, opt = fabric.setup(module.target, opt)
        opt, metaopt = fabric.setup(opt, metaopt)
            
        for _ in range(epochs):
            module.target.train()
            batches = enumerate(train_loader)

            lasti, (lastX, lasty) = next(batches)

            for i, (X, y) in enumerate(batches):
                
                # BATCH 1
                opt.zero_grad()

                x_embd, _, err = module.training_step((lastX, lasty), lasti)
                fabric.backward(err)

                opt.step(x_embd)

                # BATCH 2
                metaopt.zero_grad()

                x_embd, _, err = module.training_step((X, y), i)
                fabric.backward(err)

                metaopt.step()

                lastX, lasty = X, y
            
            test(module, test_loader)

