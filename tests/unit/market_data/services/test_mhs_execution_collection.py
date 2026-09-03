"""SCENARIO_MHS_EXECUTION_DATA_COVERAGE_GATE_*: pre-flight execution cache coverage gate contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.market_data.services.futures_collection as fc
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


def _write_mark_cache(
    root: Path, symbol: str, hourly_idx: pd.DatetimeIndex, closes: list[float | None],
) -> None:
    """Write one symbol's ``1h`` markPriceKlines Parquet with ``datetime`` and
    ``close`` columns (``None`` entries become NaN)."""
    d = root / "markPriceKlines" / "1h"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": _epoch_ms(hourly_idx),
            "datetime": hourly_idx,
            "close": closes,
        }
    ).to_parquet(d / f"{symbol}.parquet")


def _pin_mark_path(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Route ``_mark_price_path`` to ``root``. ``mhs_execution_collection``
    resolves the path through ``_futures_collection._mark_price_path`` at call
    time and ``_futures_collection`` is the same module object as ``fc``, so
    patching ``fc`` reaches the gate."""
    monkeypatch.setattr(
        fc, "_mark_price_path",
        lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet",
    )


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


def test_roster_membership_intervals_contiguous_runs() -> None:
    # SCENARIO_ROSTER_MEMBERSHIP_INTERVALS_CONTIGUOUS_RUNS: a mask column with
    # pattern [F,T,T,F,T,F] over an hourly index yields exactly two intervals --
    # (idx[1], idx[2]) and (idx[4], idx[4]) -- leave-and-re-enter is NOT
    # collapsed into one span.
    idx = pd.date_range("2021-01-01", periods=6, freq="1h", tz="UTC")
    mask = pd.DataFrame({"A": [False, True, True, False, True, False]}, index=idx)
    intervals = mhs_execution_collection.roster_membership_intervals(mask)
    assert intervals["A"] == (
        (idx[1], idx[2]),
        (idx[4], idx[4]),
    )


def test_roster_membership_intervals_omits_never_member() -> None:
    # SCENARIO_ROSTER_MEMBERSHIP_INTERVALS_OMITS_NEVER_MEMBER: an all-False
    # symbol is absent from the mapping entirely (not mapped to an empty tuple);
    # an all-True column yields exactly one interval spanning (index[0],
    # index[-1]).
    idx = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
    mask = pd.DataFrame(
        {"NEVER": [False, False, False, False], "ALWAYS": [True, True, True, True]},
        index=idx,
    )
    intervals = mhs_execution_collection.roster_membership_intervals(mask)
    assert "NEVER" not in intervals
    assert intervals["ALWAYS"] == ((idx[0], idx[-1]),)


def test_relevant_execution_coverage_ignores_gap_outside_membership(tmp_path) -> None:
    # SCENARIO_RELEVANT_EXECUTION_COVERAGE_IGNORES_GAP_OUTSIDE_MEMBERSHIP: a
    # symbol whose 3m parquet has an internal gap entirely OUTSIDE its roster
    # membership interval passes -- reproducing the measured 36/36
    # false-positive case the full-scope gate wrongly blocked.
    root = tmp_path / "cache"
    # 3m bars with an interior hole at ~12:00-13:57 (bars 240..279).
    _write_cache(root, "GAPUSDT", interval="3m", drop_slice=slice(240, 280))
    hourly = pd.date_range(_START, _END, freq="1h", tz="UTC")
    # Roster covers only hours 00:00..08:00 -- the gap at noon is irrelevant.
    mask = pd.DataFrame(
        {"GAPUSDT": [True] * 9 + [False] * (len(hourly) - 9)}, index=hourly,
    )
    assert mhs_execution_collection.assert_relevant_execution_data_coverage(
        mask, "3m", root=str(root),
    ) is None


def test_relevant_execution_coverage_raises_on_gap_inside_membership(tmp_path) -> None:
    # SCENARIO_RELEVANT_EXECUTION_COVERAGE_RAISES_ON_GAP_INSIDE_MEMBERSHIP: a
    # gap inside the roster interval fails closed naming the symbol and the
    # offending interval.
    root = tmp_path / "cache"
    # 3m bars with an interior hole at ~02:00-02:33 (bars 40..51).
    _write_cache(root, "GAPUSDT", interval="3m", drop_slice=slice(40, 52))
    hourly = pd.date_range(_START, _END, freq="1h", tz="UTC")
    mask = pd.DataFrame({"GAPUSDT": [True] * 21 + [False] * (len(hourly) - 21)}, index=hourly)
    with pytest.raises(DataIntegrityError) as exc_info:
        mhs_execution_collection.assert_relevant_execution_data_coverage(
            mask, "3m", root=str(root),
        )
    message = str(exc_info.value)
    assert "GAPUSDT" in message
    assert "GAPPED" in message


def test_relevant_mark_coverage_raises_when_roster_precedes_mark(tmp_path, monkeypatch) -> None:
    # SCENARIO_RELEVANT_MARK_COVERAGE_RAISES_WHEN_ROSTER_PRECEDES_MARK: a
    # symbol in the roster from 2021-01-30 but whose mark parquet starts
    # 2022-04-01 (the measured VETUSDT case) raises naming the symbol -- the
    # failure the current pipeline only discovers 91 seconds into the replay.
    root = tmp_path / "cache"
    mark_idx = pd.date_range("2022-04-01", periods=72, freq="1h", tz="UTC")
    _write_mark_cache(root, "VETUSDT", mark_idx, [100.0] * len(mark_idx))
    _pin_mark_path(monkeypatch, root)
    idx = pd.date_range("2021-01-30", periods=100, freq="1h", tz="UTC")
    mask = pd.DataFrame({"VETUSDT": [True] * len(idx)}, index=idx)
    with pytest.raises(DataIntegrityError) as exc_info:
        mhs_execution_collection.assert_relevant_mark_price_coverage(mask)
    assert "VETUSDT" in str(exc_info.value)


def test_relevant_mark_coverage_rejects_nonpositive_mark(tmp_path, monkeypatch) -> None:
    # SCENARIO_RELEVANT_MARK_COVERAGE_REJECTS_NONPOSITIVE_MARK: rows existing
    # across the whole interval but carrying close<=0 or NaN are treated as NOT
    # covered, matching _cached_mark_panel's finite-and-strictly-positive
    # filter rather than counting row presence alone.
    root = tmp_path / "cache"
    idx = pd.date_range("2021-01-01", periods=48, freq="1h", tz="UTC")
    closes = [100.0] * len(idx)
    closes[10] = 0.0
    closes[20] = None
    _write_mark_cache(root, "A", idx, closes)
    _pin_mark_path(monkeypatch, root)
    # Roster from hour 1 so the fixture's own marks cover every interval hour.
    mask = pd.DataFrame({"A": [False] + [True] * (len(idx) - 1)}, index=idx)
    with pytest.raises(DataIntegrityError) as exc_info:
        mhs_execution_collection.assert_relevant_mark_price_coverage(mask)
    assert "A" in str(exc_info.value)


def test_relevant_mark_coverage_applies_one_hour_availability_shift(tmp_path, monkeypatch) -> None:
    # SCENARIO_RELEVANT_MARK_COVERAGE_APPLIES_ONE_HOUR_AVAILABILITY_SHIFT: a
    # mark row is available only from timestamp + 1h. With marks covering
    # [idx[0], idx[-1]] and the roster spanning the same window, the last row
    # (== interval end) cannot cover the final interval hour and the first hour
    # needs a prior-day mark, so the gate raises -- a no-shift gate would pass.
    # The control (roster from hour 1) passes because each interval hour is
    # covered by the PRIOR hour's mark.
    root = tmp_path / "cache"
    idx = pd.date_range("2021-01-01", periods=48, freq="1h", tz="UTC")
    _write_mark_cache(root, "A", idx, [100.0] * len(idx))
    _pin_mark_path(monkeypatch, root)
    all_mask = pd.DataFrame({"A": [True] * len(idx)}, index=idx)
    with pytest.raises(DataIntegrityError) as exc_info:
        mhs_execution_collection.assert_relevant_mark_price_coverage(all_mask)
    assert "A" in str(exc_info.value)
    shifted_mask = pd.DataFrame({"A": [False] + [True] * (len(idx) - 1)}, index=idx)
    assert mhs_execution_collection.assert_relevant_mark_price_coverage(shifted_mask) is None


def test_relevant_mark_coverage_stale_carry_allowance(tmp_path, monkeypatch) -> None:
    # SCENARIO_RELEVANT_MARK_COVERAGE_STALE_CARRY_ALLOWANCE: a 6-hour internal
    # hole raises with stale_hours=0 but passes with stale_hours=24, matching
    # the cache_required vs cache_required_stale_carry replay modes.
    root = tmp_path / "cache"
    idx = pd.date_range("2021-01-01", periods=48, freq="1h", tz="UTC")
    closes = [100.0] * len(idx)
    for h in range(20, 26):
        closes[h] = None
    _write_mark_cache(root, "A", idx, closes)
    _pin_mark_path(monkeypatch, root)
    mask = pd.DataFrame({"A": [False] + [True] * (len(idx) - 1)}, index=idx)
    with pytest.raises(DataIntegrityError):
        mhs_execution_collection.assert_relevant_mark_price_coverage(mask, stale_hours=0)
    assert mhs_execution_collection.assert_relevant_mark_price_coverage(
        mask, stale_hours=24,
    ) is None
