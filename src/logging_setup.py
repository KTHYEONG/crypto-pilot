from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_FORMAT = "%(asctime)s [%(levelname)s] [%(tag)s] %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
