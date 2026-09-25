"""Atomic parquet read-modify-write helpers with quarantine for undecodable partitions."""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pyarrow.lib import ArrowException

QUARANTINE_DIRNAME: str = "_quarantine"

_logger = logging.getLogger("ParquetIO")



def is_undecodable_parquet_error(exc: BaseException) -> bool:
    """Whether ``exc`` means the parquet bytes themselves are unreadable (not the filesystem).

    pyarrow reports footer/metadata damage as ``ArrowException`` subclasses and page-level
    decompression damage as ``OSError`` without an errno; genuine OS failures (permissions,
    descriptor exhaustion, device I/O) always carry an errno and are therefore excluded, because
    moving a healthy file aside on a transient OS error would split a partition for no reason.

    Args:
        exc: Exception raised by ``pandas.read_parquet``.

    Returns:
        True only for ``pyarrow.lib.ArrowException`` instances or ``OSError`` instances whose
        ``errno`` is ``None``.
    """
    if isinstance(exc, ArrowException):
        return True
    if isinstance(exc, OSError):
        return exc.errno is None
    return False


def write_parquet_atomic(frame: pd.DataFrame, path: Path, *, compression: str) -> Path:
    """Replace ``path`` with ``frame`` so that readers never observe a partially written file.

    The frame is written to a process- and thread-unique sibling whose name starts with ``.`` and
    ends with ``.tmp`` (excluded by the backup filter), flushed to stable storage with ``fsync``, and
    then moved over ``path`` with ``os.replace``, which is atomic within one POSIX filesystem. A crash
    at any point leaves either the previous complete ``path`` or the new complete ``path`` plus, at
    worst, an ignored temp file.

    Args:
        frame: Complete partition content to persist (index is not written).
        path: Final partition path; its parent directory must already exist.
        compression: Parquet codec passed through to the writer (e.g. ``"zstd"``, ``"snappy"``).

    Returns:
        ``path``.

    Raises:
        Exception: Any error from serialization, ``fsync`` or ``os.replace`` is re-raised after the
            temp file is removed; ``path`` is left byte-for-byte unchanged.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        frame.to_parquet(tmp, index=False, compression=compression)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return path


def read_parquet_or_quarantine(path: Path, *, stage: str) -> pd.DataFrame | None:
    """Load an existing partition for read-modify-write, moving undecodable bytes aside.

    Appenders must never replace an unreadable partition with the current batch: that silently
    destroys every earlier row. An undecodable file is instead moved (never copied, never deleted)
    to ``<path.parent>/_quarantine/<path.name>.<UTC %Y%m%dT%H%M%S%fZ>.<pid>.corrupt`` and reported at
    ERROR level with its traceback, after which the caller starts a fresh partition. The
    ``.corrupt`` suffix keeps quarantined files out of every ``*.parquet`` glob while the backup
    still preserves them as evidence.

    Args:
        path: Partition to read.
        stage: Log correlation id of the calling writer (e.g. ``"live_fills"``, ``"liquidations"``).

    Returns:
        The full partition, or ``None`` when ``path`` does not exist or was just quarantined.

    Raises:
        OSError: OS-level read failures (errno set), propagated unchanged with ``path`` untouched.
        Exception: Any other non-decode failure, propagated unchanged with ``path`` untouched.
    """
    path = Path(path)
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        return None
    except Exception as exc:
        if not is_undecodable_parquet_error(exc):
            raise
        quarantine_dir = path.parent / QUARANTINE_DIRNAME
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")  # noqa: UP017
        target = quarantine_dir / f"{path.name}.{stamp}.{os.getpid()}.corrupt"
        counter = 0
        while target.exists():
            counter += 1
            target = quarantine_dir / f"{path.name}.{stamp}.{os.getpid()}.{counter}.corrupt"
        os.replace(path, target)
        _logger.error(
            "[DATA] stage=%s status=QUARANTINED path=%s moved_to=%s error=%s",
            stage,
            path,
            target,
            exc,
            exc_info=True,
            extra={"tag": "DATA"},
        )
        return None
