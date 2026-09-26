"""Stdlib logging setup for the capture process."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_configured_for: set[str] = set()


def configure_capture_logging(log_dir: Path, slot: str) -> Path | None:
    """Root logging to stdout plus ``RotatingFileHandler(log_dir/"capture_<slot>.log", 10 MiB x 5)``.

    The rotating file survives container recreation (docker json logs do not). Failure to open
    the file falls back to stdout only with a warning. The call is idempotent. Log size is
    bounded by rotation, so no unbounded residue accumulates.
    """
    root = logging.getLogger()
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    if not any(isinstance(item, logging.StreamHandler) for item in root.handlers):
        root.addHandler(logging.StreamHandler())
    marker = f"{log_dir}:{slot}"
    if marker in _configured_for:
        return log_dir / f"capture_{slot}.log"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"capture_{slot}.log"
        for item in root.handlers:
            if isinstance(item, RotatingFileHandler) and getattr(item, "baseFilename", "") == str(path):
                _configured_for.add(marker)
                return path
        handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "[SYS] stage=capture component=logsetup status=FILE_FALLBACK error=%s",
            type(exc).__name__,
        )
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
    _configured_for.add(marker)
    return path
