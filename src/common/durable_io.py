"""Crash-durable whole-file replacement for small state artifacts.

``os.replace`` alone is atomic against process crashes but not against power loss:
an un-fsynced temp file can surface as an empty or torn file, and an un-fsynced
directory can lose the rename. Readers of live state fail closed on such files, so
a power loss would otherwise become a manual-intervention outage.
"""

from __future__ import annotations

import contextlib
import errno
import os
import threading
from collections.abc import Callable
from pathlib import Path


def fsync_directory(directory: Path) -> None:
    """Flush ``directory``'s entries (completed renames/creations) to stable storage.

    A directory that cannot be opened (platform without directory handles, restricted
    mount) or a filesystem that rejects directory fsync (``EINVAL``, ``ENOTSUP``) is
    skipped: the file contents are already durable and nothing more can be done.

    Raises:
        OSError: any other fsync failure (e.g. ``EIO``), because the rename's
            durability is then unknown.
    """
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
            return
        raise
    finally:
        os.close(fd)


def durable_replace(path: Path, write: Callable[[Path], None]) -> None:
    """Replace ``path`` with content produced by ``write`` so that after a crash or power
    loss readers observe either the previous complete file or the new complete file.

    ``write`` receives a hidden sibling temp path (``.<name>.<pid>.<thread>.tmp``,
    excluded from backups and from ``*.json``/``*.parquet`` globs) and must leave the
    complete content in it. The temp file is fsynced, moved over ``path`` with
    ``os.replace`` and the parent directory is fsynced.

    Args:
        path: Destination; its parent directory must already exist.
        write: Writes the complete content to the given temp path.

    Raises:
        BaseException: anything raised by ``write``, the file fsync or ``os.replace`` is
            re-raised unchanged after the temp file is removed; ``path`` is untouched.
        OSError: a directory fsync failure after the replace (see ``fsync_directory``);
            ``path`` already holds the new content.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        write(tmp)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    fsync_directory(path.parent)


def durable_write_text(path: Path, text: str) -> None:
    """``durable_replace`` for UTF-8 text."""

    def _write(tmp: Path) -> None:
        tmp.write_text(text, encoding="utf-8")

    durable_replace(path, _write)


def durable_write_bytes(path: Path, data: bytes) -> None:
    """``durable_replace`` for raw bytes."""

    def _write(tmp: Path) -> None:
        tmp.write_bytes(data)

    durable_replace(path, _write)
