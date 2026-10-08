"""Hermetic storage-root guard for the pytest session."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest


def escaped_file_handlers(temp_root: Path, *extra_roots: Path) -> list[tuple[str, Path]]:
    """List file handlers attached anywhere in the logging tree whose file escapes ``temp_root``.

    Scans the root logger and every registered ``logging.Logger`` (placeholders skipped) for
    ``logging.FileHandler`` instances (``RotatingFileHandler`` included) and resolves each ``baseFilename``.
    The OS null device (pytest's own log-file sink when no ``log_file`` is configured) is skipped:
    it discards writes and is never operator storage.

    Args:
        temp_root: This process's partitioned pytest temp root.
        extra_roots: Further hermetic roots, e.g. the session ``basetemp``: under xdist the controller owns
            ``basetemp``, so worker ``tmp_path`` directories live outside the worker's own temp root.
    Returns:
        ``(logger_name, resolved_path)`` pairs outside ``temp_root.resolve()``, sorted by logger name then path;
        the root logger is reported as ``"root"``. Empty when hermetic.
    """
    resolved_roots = tuple(root.resolve() for root in (temp_root, *extra_roots))
    null_device = Path(os.devnull).resolve()
    offenders: list[tuple[str, Path]] = []
    loggers: list[tuple[str, logging.Logger]] = [("root", logging.getLogger())]
    manager = logging.Logger.manager
    for name, obj in list(manager.loggerDict.items()):
        if isinstance(obj, logging.Logger):
            loggers.append((name, obj))
    for logger_name, logger in loggers:
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                resolved = Path(handler.baseFilename).resolve()
                if resolved == null_device:
                    continue
                if not any(resolved.is_relative_to(root) for root in resolved_roots):
                    offenders.append((logger_name, resolved))
    offenders.sort(key=lambda item: (item[0], str(item[1])))
    return offenders


def assert_storage_roots_hermetic(
    roots: Mapping[str, Path],
    temp_root: Path,
    preimported_src_modules: Sequence[str],
) -> None:
    """Fail the session unless every import-time storage root is inside this run's temp root.

    ``BACKTESTS_DIR`` and ``LOG_DIR`` are pinned when ``src.common.paths`` /
    ``src.common.logging`` are first imported. If anything imported them before
    ``tests/conftest.py`` set the redirect variables (a ``-p`` plugin, a coverage
    source that imports packages, a wrapper script), tests would write the
    operator registry and logs while the redirect looks configured.

    Args:
        roots: Label -> resolved-at-import root (e.g. ``{"BACKTESTS_DIR": ...}``).
        temp_root: This process's partitioned pytest temp root.
        extra_roots: Further hermetic roots, e.g. the session ``basetemp``: under xdist the controller owns
            ``basetemp``, so worker ``tmp_path`` directories live outside the worker's own temp root.
        preimported_src_modules: ``src`` modules already in ``sys.modules`` when
            conftest started, reported verbatim to locate the early importer.
    Raises:
        pytest.UsageError: Naming every offending label with its resolved path,
            the expected temp root, and ``preimported_src_modules`` (or ``none``).
    """
    resolved_temp = temp_root.resolve()
    offenders: list[tuple[str, Path]] = []
    for label, root in roots.items():
        resolved_root = root.resolve()
        if not resolved_root.is_relative_to(resolved_temp):
            offenders.append((label, resolved_root))
    if not offenders:
        return None
    names = list(preimported_src_modules)
    if names:
        shown = ", ".join(names[:20])
        if len(names) > 20:
            shown += f", ... (+{len(names) - 20} more)"
        preimported = f"{len(names)} module(s): {shown}"
    else:
        preimported = "none"
    details = "; ".join(f"{label}={path} expected under {resolved_temp}" for label, path in offenders)
    raise pytest.UsageError(
        f"hermetic storage roots escaped temp root ({len(offenders)} root(s): {details}). "
        "Ensure CRYPTO_PILOT_BACKTESTS_DIR/CRYPTO_PILOT_LOG_DIR are set before "
        f"importing src; preimported src modules: {preimported}."
    )
