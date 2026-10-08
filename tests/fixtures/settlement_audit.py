"""Explicit settlement-registry supply for synthetic fixture lakes.

Every replay entry point audits its census against a settlement registry; a
synthetic fixture archive stops at the fixture end, which is a collection
horizon rather than a delisting, so one truncation record per 3m archive
explains it. Tests that replay a fixture lake through a real entry point must
call :func:`patch_settlement_registry_for_fixture`; tests that only build fold
targets never reach an audit gate and need nothing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def fixture_truncation_registry(root: str | Path):
    """Build a truncation-only registry explaining a fixture lake's end."""
    from src.core.instrument_settlements import (
        DataTruncationRecord,
        assemble_instrument_settlement_registry,
    )
    from src.core.settlement_evidence import measure_symbol_tail

    verified = pd.Timestamp("2026-07-01T00:00:00Z")
    records = [
        DataTruncationRecord(
            symbol=path.stem,
            data_end=measure_symbol_tail(path).last_bar + pd.Timedelta(minutes=3),
            evidence="synthetic fixture lake ends here; collection horizon, not a delisting",
            verified_at=verified,
        )
        for path in sorted((Path(root) / "3m").glob("*.parquet"))
    ]
    return assemble_instrument_settlement_registry([], records)


def patch_settlement_registry_for_fixture(monkeypatch: pytest.MonkeyPatch, root: str | Path) -> None:
    """Supply the fixture's explicit truncation registry at every audit entry point."""
    registry = fixture_truncation_registry(root)
    monkeypatch.setattr(
        "src.lab.mhs.pipeline.stages.panel.settlement_registry_for_root", lambda _root: registry,
    )
    monkeypatch.setattr(
        "src.engine.strategy_backtest.settlement_registry_for_root", lambda _root: registry,
    )
    monkeypatch.setattr(
        "src.lab.mhs.backtest.inventory.settlement_registry_for_root", lambda _root: registry,
    )


def settlement_registry_for_test_root(ohlcv_root: str | Path):
    """Lazy registry resolver for test sessions over synthetic lakes.

    The canonical lake resolves to the committed registry (production
    behaviour); any other root with a ``3m`` directory gets an explicit
    truncation registry explaining its collection horizon, while roots
    without one get the empty registry (symbols without archives are
    skipped by the audit and owned by existing missing-source checks).
    Unreadable archives fall back to the empty registry so the audit
    itself raises the precise error.
    """
    from src.common.errors import DataIntegrityError
    from src.common.paths import FUTURES_DATA_DIR
    from src.core.instrument_settlements import (
        EMPTY_SETTLEMENT_REGISTRY,
        load_instrument_settlement_registry,
    )

    resolved = Path(ohlcv_root).resolve()
    if resolved == (FUTURES_DATA_DIR / "ohlcv").resolve():
        return load_instrument_settlement_registry()
    if not (resolved / "3m").is_dir():
        return EMPTY_SETTLEMENT_REGISTRY
    try:
        return fixture_truncation_registry(resolved)
    except DataIntegrityError:
        return EMPTY_SETTLEMENT_REGISTRY
