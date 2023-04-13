import logging

import torch
from rich.logging import RichHandler


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
