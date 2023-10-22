import torch

from modules.module_abc import ModuleABC
from utils import device, logger, accuracy


def test(module: ModuleABC, test_loader, epoch: int):
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


def train(dataset, epochs: int, reps: int, module: ModuleABC):
    """
    Training flow
    - Gets batch
    - Puts data through target
    - Adjusts weighting (meta-optimizer)
    - Adjusts target on next batch
    """

    train_loader, test_loader = dataset

    state = None
    for _ in range(reps):
        module.reset_target()

        if state is not None:
            module.set_state(state)

        opts = module.configure_optimizers()

        for I in range(epochs):
            module.target.train()

            train_error = 0.0
            train_accuracy = 0.0

            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)

                for opt in reversed(opts):
                    opt.zero_grad()

                x_embd, preds, err = module.training_step((X, y), i)

                train_error += err
                train_accuracy += accuracy(preds, y)

                err.backward(retain_graph=True)

                if (i > 0) and (len(opts) > 1):
                    opts[-1].step()     # ADJUST WEIGTING NETWORK

                if len(opts) > 1:
                    opts[0].step(x_embd)    # ADJUST TARGET WITH WEIGHTED GRAD

            train_error /= len(train_loader)
            train_accuracy /= len(train_loader)
            logger.get().log_metrics({
            "train/accuracy": train_accuracy,
            "train/loss": train_error
            }, I)

            test(module, test_loader, I)

        state = module.get_state(opts[0])
