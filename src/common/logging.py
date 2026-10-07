from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Final

from src.common.paths import BASE_DIR

LOG_DIR: Path = Path(os.environ["CRYPTO_PILOT_LOG_DIR"]) if os.environ.get("CRYPTO_PILOT_LOG_DIR") else BASE_DIR / "logs"
"""Operator process-log root, resolved once at import (``CRYPTO_PILOT_LOG_DIR`` overrides ``BASE_DIR / "logs"``).

Resolution only: nothing is created at import. Directories are created by the explicit boundary setup that
attaches a handler into them, so importing any ``src`` module never touches the filesystem.
"""

_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] [%(tag)s] %(message)s"
_DATE_FMT: Final[str] = "%Y-%m-%d %H:%M:%S"
_NO_TAG_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(message)s"
_KNOWN_TAGS: Final[frozenset[str]] = frozenset({"SYS", "DATA", "ALGO", "PORTFOLIO", "RISK", "EXEC", "EVAL"})
_TAG_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"^\[([A-Z]{2,10})\]")


class _TagDefaultFormatter(logging.Formatter):
    """Formatter that injects a default 'tag' value when extra={'tag':...} is absent."""

    def __init__(self, fmt: str, datefmt: str | None = None) -> None:
        super().__init__(fmt, datefmt=datefmt)
        self._plain_formatter = logging.Formatter(_NO_TAG_FORMAT, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "tag"):
            record.tag = "SYS"
        message = record.getMessage()
        match = _TAG_PREFIX_RE.match(message.lstrip())
        if match is not None and match.group(1) in _KNOWN_TAGS:
            return self._plain_formatter.format(record)
        return super().format(record)


def setup_logger(name: str, *, log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Attach stdout and ``<log_dir>/<name>.log`` handlers to logger ``name``.

    Only an application boundary (CLI handler, daemon entrypoint) calls this; it is the single place a log-file
    destination is chosen, so library imports and library object construction never open operator log files.
    Re-invocation replaces the logger's handlers and closes the replaced ones, so repeated setup in one process
    neither leaks file descriptors nor duplicates lines.

    Args:
        name: Logger name; also the log-file stem.
        log_dir: Destination directory, created if absent. Operator callers pass ``LOG_DIR``.
        level: Threshold applied to the logger and both handlers.
    Returns:
        The configured logger, with ``propagate`` disabled.
    Raises:
        TypeError: ``log_dir`` is not a ``pathlib.Path``; raised before any directory or file is created and
            before the logger's existing handlers are touched.
        OSError: The directory or log file cannot be created or opened.
    """
    if not isinstance(log_dir, Path):
        raise TypeError(f"log_dir must be a pathlib.Path, got {type(log_dir).__name__}")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(level)

    formatter = _TagDefaultFormatter(_FORMAT, datefmt=_DATE_FMT)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
