from __future__ import annotations

import logging
import sys

from src.common.config import BASE_DIR

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_FORMAT = "%(asctime)s [%(levelname)s] [%(tag)s] %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


class _TagDefaultFormatter(logging.Formatter):
    """Formatter that injects a default 'tag' value when extra={'tag':...} is absent."""

    def __init__(self, fmt: str, datefmt: str | None = None) -> None:
        super().__init__(fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "tag"):
            record.tag = "SYS"
        return super().format(record)


def setup_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = _TagDefaultFormatter(_FORMAT, datefmt=_DATE_FMT)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
