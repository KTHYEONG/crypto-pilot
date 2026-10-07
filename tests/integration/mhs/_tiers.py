"""Heavy-tier classification and xdist fork-worker cap for MHS integration tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

from tests.integration.mhs._report_cache import CollectedItem

MARKET_LAKE_MARKER: Final[str] = "requires_market_lake"


def market_lake_skip_reason(lake_root: Path) -> str | None:
    """Skip reason for lake-dependent acceptance tests, or None when the lake is usable.

    Absence is an environment fact, not a data anomaly: only a lake root whose ``1h``
    directory holds no ``*.parquet`` (including a missing root) is skipped. A partially
    populated lake returns None so the test runs and fails loudly.

    Args:
        lake_root: The default MHS lake root (``FUTURES_DATA_DIR / "ohlcv"``).
    Returns:
        ``"requires_market_lake: no 1h parquet under <lake_root>; run on a host with the MHS lake"``
        when absent, else None.
    """
    if not list((lake_root / "1h").glob("*.parquet")):
        return f"requires_market_lake: no 1h parquet under {lake_root}; run on a host with the MHS lake"
    return None


HEAVY_DIAGNOSTIC_FIXTURES: Final[frozenset[str]] = frozenset({
    "canonical_report_run",
    "report",
    "calibrated_report",
    "annualization_report",
    "touch_report",
    "late_market_report",
    "fold_safe_report",
    "fold_safe_baseline_report",
    "refactor_report",
    "matrix_market",
    "fold_parity_request",
})
XDIST_FORK_WORKER_CAP: Final[int] = 1


def fork_worker_cap(environ: Mapping[str, str]) -> int | None:
    """Fork-pool worker cap for MHS diagnostics run inside the test process.

    Under pytest-xdist every worker already occupies a core, so nested fork pools only
    oversubscribe the host; a single fork worker keeps the production fork/COW code path
    while bounding the process tree. Outside xdist the production CPU bound applies.

    Args:
        environ: Process environment (``os.environ`` in fixtures).
    Returns:
        ``XDIST_FORK_WORKER_CAP`` when ``PYTEST_XDIST_WORKER`` is present, else ``None``.
    """
    if "PYTEST_XDIST_WORKER" in environ:
        return XDIST_FORK_WORKER_CAP
    return None


def _has_mark(item: CollectedItem, name: str) -> bool:
    return any(True for _ in item.iter_markers(name=name))


def heavy_tier_violations(items: Iterable[CollectedItem], suite_root: Path) -> list[str]:
    """List MHS integration items whose heavy-tier marking disagrees with their fixtures.

    Only items under ``suite_root`` are checked. Rules, one line per violation
    (``"<nodeid>: <rule>"``):
    - fixture closure intersects ``HEAVY_DIAGNOSTIC_FIXTURES``, not ``slow``, lacks ``e2e_heavy``;
    - carries both ``e2e_heavy`` and ``slow`` (tiers must stay disjoint);
    - carries ``e2e_heavy`` without any heavy fixture (stale tier mark).

    Returns:
        Violation lines in item order; empty when consistent.
    """
    suite_root = Path(suite_root)
    violations: list[str] = []
    for item in items:
        path = Path(item.path)
        if path != suite_root and suite_root not in path.parents:
            continue
        fixtures = set(item.fixturenames)
        uses_heavy = bool(fixtures & HEAVY_DIAGNOSTIC_FIXTURES)
        is_slow = _has_mark(item, "slow")
        is_heavy = _has_mark(item, "e2e_heavy")
        if is_heavy and is_slow:
            violations.append(f"{item.nodeid}: e2e_heavy and slow marks must stay disjoint")
        if is_heavy and not uses_heavy:
            violations.append(f"{item.nodeid}: e2e_heavy mark without heavy diagnostic fixture")
        if uses_heavy and not is_slow and not is_heavy:
            violations.append(f"{item.nodeid}: heavy diagnostic fixture requires e2e_heavy mark")
    return violations
