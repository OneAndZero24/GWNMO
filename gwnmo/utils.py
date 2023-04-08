import logging

import torch
from rich.logging import RichHandler


def _setup_logger(log_level: str):
    return logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True,
                              tracebacks_suppress=[torch])]
    )


log = _setup_logger("DEBUG")
