import torch
import learn2learn as l2l

from modules.fewshot.fsmodule_abc import FSModuleABC
from utils import device, logger, accuracy


def test(module: FSModuleABC, test_loader, epoch: int):
    """
    Test target on whole test
    """

    module.target.eval()

    test_error = 0.0
    test_accuracy = 0.0

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            _, preds, err = module.training_step((X, y), i)
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
    opts = module.configure_optimizers()

    for I in range(epochs):
        module.target.train()

        train_error = 0.0

        for i, batch in enumerate(train_loader):
            for opt in opts:
                opt.zero_grad()

            _, preds, err = module.training_step(batch, i)

            train_error += err

            err.backward(retain_graph=True)

            for opt in opts:
                opt.step()

        train_error /= len(train_loader)
        logger.get().log_metrics({
        "train/loss": train_error
        }, I)

        test(module, test_loader, I)

    