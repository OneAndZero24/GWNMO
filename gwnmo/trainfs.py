import torch

from modules.fewshot.fsmodule_abc import FSModuleABC
from utils import device, logger, accuracy


def test(module: FSModuleABC, test_loader, epoch: int):
    """
    Test target on whole test
    """
    module.target.eval()

    for param in module.target.parameters():
        param.requires_grad = False

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


def train(dataset, epochs: int, module: FSModuleABC):
    """
    FewShot training flow
    - Gets batch
    - Puts data through target
    - Adapts via weighting
    - Adjusts weighting and target
    """

    train_loader, test_loader = dataset

    module.reset_target()
    for I in range(epochs):
        module.target.train()

        train_error = 0.0
        train_accuracy = 0.0

        for i, batch in enumerate(train_loader):
            y, preds, err = module.training_step(batch, i)

            opt = module.configure_optimizers()
            opt.zero_grad()

            train_error += err
            train_accuracy += accuracy(preds, y)

            err.backward(retain_graph=True)
            
            opt.step()

        train_error /= len(train_loader)
        train_accuracy /= len(train_loader)
        logger.get().log_metrics({
        "train/accuracy": train_accuracy,
        "train/loss": train_error
        }, I)

        test(module, test_loader, I)

    