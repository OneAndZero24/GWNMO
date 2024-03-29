import torch

from modules.fewshot.fsmodule_abc import FSModuleABC
from utils import logger, accuracy


def test(module: FSModuleABC, test_loader, epoch: int):
    """
    Test target on whole test
    """
    module.eval()

    test_error = 0.0
    test_accuracy = 0.0

    for i, batch in enumerate(test_loader):
        y, preds, err = module.training_step(batch, i)
        test_error += err
        test_accuracy += accuracy(preds, y)
    test_error /= len(test_loader)
    test_accuracy /= len(test_loader)
    logger.get().log_metrics({
        "test/accuracy": test_accuracy,
        "test/loss": test_error
        }, epoch)


def train(dataset, epochs: int, module: FSModuleABC, no_weighting: int = -1, second_order: bool = False):
    """
    FewShot training flow
    - Gets batch
    - Puts data through target
    - Adapts via weighting
    - Adjusts weighting and target
    """

    train_loader, test_loader = dataset

    if hasattr(module, 'toggle_weighting'):
        module.toggle_weighting(False)

    module.reset_target()
    for I in range(epochs):
        if I > no_weighting:
            if hasattr(module, 'toggle_weighting'):
                module.toggle_weighting(True)
        module.train()

        train_error = 0.0
        train_accuracy = 0.0

        opt = module.configure_optimizers()

        for i, batch in enumerate(train_loader):
            opt.zero_grad()
            y, preds, err = module.training_step(batch, i)

            train_error += err
            train_accuracy += accuracy(preds, y)

            err.backward(retain_graph=True, create_graph=second_order)
            for p in module.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(1.0/len(train_loader))
            opt.step()

        train_error /= len(train_loader)
        train_accuracy /= len(train_loader)
        logger.get().log_metrics({
        "train/accuracy": train_accuracy,
        "train/loss": train_error
        }, I)

        test(module, test_loader, I)

    