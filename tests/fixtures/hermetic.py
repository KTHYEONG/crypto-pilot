"""Hermetic storage-root guard for the pytest session."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest


def assert_storage_roots_hermetic(
    roots: Mapping[str, Path],
    temp_root: Path,
    preimported_src_modules: Sequence[str],
) -> None:
    """Fail the session unless every import-time storage root is inside this run's temp root.

    ``BACKTESTS_DIR`` and ``LOG_DIR`` are frozen when ``src.common.paths`` /
    ``src.common.logging`` are first imported. If anything imported them before
    ``tests/conftest.py`` set the redirect variables (a ``-p`` plugin, a coverage
    source that imports packages, a wrapper script), tests would write the
    operator registry and logs while the redirect looks configured.

    Args:
        roots: Label -> resolved-at-import root (e.g. ``{"BACKTESTS_DIR": ...}``).
        temp_root: This process's partitioned pytest temp root.
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
