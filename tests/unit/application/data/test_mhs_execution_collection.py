"""SCENARIO_MHS_EXECUTION_DATA_COVERAGE_GATE_*: pre-flight execution cache
coverage gate contract (docs/specs/mhs_execution_data_coverage_gate.md §3.1).

``assert_execution_data_coverage`` must fail closed with the deficient symbol
list and its status when any symbol is MISSING or GAPPED, stay a no-op when
every symbol is PRESENT, and reuse the existing ``_coverage`` helper only
(local Parquet metadata reads, never ``DataCollector``/network). The ``root``
override must be backward compatible: calling ``_coverage`` without ``root``
still resolves under ``FUTURES_DATA_DIR / 'ohlcv'``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.application.data import mhs_execution_collection
from src.common.errors import DataIntegrityError

_START = "2023-01-01T00:00:00Z"
_END = "2023-01-01T23:55:00Z"


def _epoch_ms(idx: pd.DatetimeIndex) -> pd.Series:
    return (idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")


_FREQ_BY_INTERVAL = {"1m": "1min", "3m": "3min", "5m": "5min"}


def _write_cache(
    root: Path, symbol: str, interval: str = "5m", start: str = _START,
    end: str = _END, drop_slice: slice | None = None,
) -> None:
    """Write one symbol's ``interval`` Parquet covering [start, end];
    ``drop_slice`` removes a contiguous interior span of bars to produce an
    internal hole."""
    idx = pd.date_range(start, end, freq=_FREQ_BY_INTERVAL[interval], tz="UTC")
    if drop_slice is not None:
        idx = idx.delete(range(*drop_slice.indices(len(idx))))
    (root / interval).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": _epoch_ms(idx)}).to_parquet(root / interval / f"{symbol}.parquet")


def test_mhs_execution_data_coverage_gate_missing_symbol_raises(tmp_path) -> None:
    # SCENARIO_MHS_EXECUTION_DATA_COVERAGE_GATE_MISSING_SYMBOL_RAISES: only
    # BTCUSDT has a parquet file; MISSINGUSDT has none. The gate must raise
    # DataIntegrityError listing MISSINGUSDT with its MISSING status.
    root = tmp_path / "cache"
    _write_cache(root, "BTCUSDT")
    with pytest.raises(DataIntegrityError) as exc_info:
        mhs_execution_collection.assert_execution_data_coverage(
            ["BTCUSDT", "MISSINGUSDT"], "5m", _START, _END, root=str(root),
        )
    message = str(exc_info.value)
    assert "MISSINGUSDT" in message
    assert "MISSING" in message


def test_mhs_execution_data_coverage_gate_all_present_noop(tmp_path) -> None:
    # SCENARIO_MHS_EXECUTION_DATA_COVERAGE_GATE_ALL_PRESENT_NOOP: every symbol
    # PRESENT with full [start, end] internal coverage never raises.
    root = tmp_path / "cache"
    for symbol in ("BTCUSDT", "ETHUSDT"):
        _write_cache(root, symbol)
    assert mhs_execution_collection.assert_execution_data_coverage(
        ["BTCUSDT", "ETHUSDT"], "5m", _START, _END, root=str(root),
    ) is None


def test_mhs_execution_data_coverage_gate_gapped_symbol_raises(tmp_path) -> None:
    # SCENARIO_MHS_EXECUTION_DATA_COVERAGE_GATE_GAPPED_SYMBOL_RAISES: a file
    # with an interior hole inside [start, end] (missing_internal_bars > 0)
    # raises DataIntegrityError naming the symbol with its GAPPED status.
    root = tmp_path / "cache"
    _write_cache(root, "BTCUSDT")
    _write_cache(root, "GAPPEDUSDT", drop_slice=slice(120, 132))
    with pytest.raises(DataIntegrityError) as exc_info:
        mhs_execution_collection.assert_execution_data_coverage(
            ["BTCUSDT", "GAPPEDUSDT"], "5m", _START, _END, root=str(root),
        )
    message = str(exc_info.value)
    assert "GAPPEDUSDT" in message
    assert "GAPPED" in message
    assert "BTCUSDT" not in message


def test_mhs_coverage_root_override_backward_compatible(tmp_path, monkeypatch) -> None:
    # SCENARIO_MHS_COVERAGE_ROOT_OVERRIDE_BACKWARD_COMPATIBLE: calling
    # ``_coverage`` without ``root`` (the existing two call sites' calling
    # convention) still resolves under ``FUTURES_DATA_DIR / 'ohlcv'``.
    root = tmp_path / "canonical"
    (root / "ohlcv" / "5m").mkdir(parents=True)
    idx = pd.date_range(_START, _END, freq="5min", tz="UTC")
    pd.DataFrame({"timestamp": _epoch_ms(idx)}).to_parquet(root / "ohlcv" / "5m" / "BTCUSDT.parquet")
    monkeypatch.setattr(mhs_execution_collection, "FUTURES_DATA_DIR", root)
    result = mhs_execution_collection._coverage("BTCUSDT", "5m", _START, _END)
    assert result["status"] == "PRESENT"
    assert result["rows"] == 288


def test_mhs_coverage_step_mapping_3m_present(tmp_path) -> None:
    # SCENARIO_MHS_COVERAGE_STEP_MAPPING_3M: ``_coverage`` with timeframe
    # ``'3m'`` resolves the internal-gap step to exactly 3 minutes -- a
    # 3-minute-bar parquet covering [start, end] reports PRESENT with zero
    # missing internal bars (not silently computed via the old 5-minute step).
    root = tmp_path / "cache"
    _write_cache(root, "BTCUSDT", interval="3m")
    result = mhs_execution_collection._coverage("BTCUSDT", "3m", _START, _END, root=str(root))
    assert result["status"] == "PRESENT"
    assert result["missing_internal_bars"] == 0
    idx = pd.date_range(_START, _END, freq="3min", tz="UTC")
    assert result["rows"] == len(idx)


def test_mhs_coverage_step_mapping_3m_gapped(tmp_path) -> None:
    # SCENARIO_MHS_COVERAGE_STEP_MAPPING_3M (GAPPED): removing one 3-minute bar
    # from the middle reports GAPPED with exactly one missing internal bar --
    # proving the 3m gap is detected at 3-minute resolution.
    root = tmp_path / "cache"
    _write_cache(root, "BTCUSDT", interval="3m", drop_slice=slice(240, 241))
    result = mhs_execution_collection._coverage("BTCUSDT", "3m", _START, _END, root=str(root))
    assert result["status"] == "GAPPED"
    assert result["missing_internal_bars"] == 1


def test_mhs_execution_plan_3m_default(tmp_path, monkeypatch) -> None:
    # SCENARIO_MHS_EXECUTION_PLAN_3M_DEFAULT: ``build_mhs_execution_plan``
    # without a timeframe kwarg plans the new native 3m interval (the only
    # interval physically present under data/futures/ohlcv/); an out-of-contract
    # ``'7m'`` still raises ValueError.
    idx = pd.date_range("2025-01-01", periods=2200, freq="1h", tz="UTC")
    quote = pd.DataFrame({f"S{i:02d}": float(i + 1) for i in range(16)}, index=idx)
    close = pd.DataFrame(
        {symbol: 100.0 + (i + 1) * pd.Series(range(len(idx)), index=idx)
         for i, symbol in enumerate(quote.columns)},
    )
    monkeypatch.setattr(
        mhs_execution_collection, "load_base_panel",
        lambda *args, **kwargs: {"close": close, "quote_vol": quote},
    )
    monkeypatch.setattr(mhs_execution_collection, "funding_path", lambda symbol: tmp_path / f"{symbol}.parquet")
    for symbol in quote.columns:
        (tmp_path / f"{symbol}.parquet").touch()

    plan = mhs_execution_collection.build_mhs_execution_plan("2025-01-01", "2025-03-30", execution_universe_size=8)
    assert plan.timeframe == "3m"
    with pytest.raises(ValueError, match="'1m', '3m' or '5m'"):
        mhs_execution_collection.build_mhs_execution_plan("2025-01-01", "2025-03-30", timeframe="7m")
