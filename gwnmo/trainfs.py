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
    logger.log_metrics({
        "accuracy": test_accuracy,
        "loss": test_error
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
        for i, batch in enumerate(train_loader):
            for opt in opts:
                opt.zero_grad()

                x_embd, _, err = module.training_step(batch, i)
                err.backward(retain_graph=True)

            for opt in opts:
                opt.step()

        test(module, test_loader, I)

    