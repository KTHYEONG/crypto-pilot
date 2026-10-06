"""SCENARIO_MHS_EXECUTION_DATA_COVERAGE_GATE_*: pre-flight execution cache coverage gate contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.market_data.services import mhs_execution as mhs_execution_collection
from src.common.errors import DataIntegrityError

_START = "2023-01-01T00:00:00Z"
_END = "2023-01-01T23:57:00Z"


def _epoch_ms(idx: pd.DatetimeIndex) -> pd.Series:
    return (idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")


_FREQ_BY_INTERVAL = {"1m": "1min", "3m": "3min", "5m": "5min"}


def _write_cache(
    root: Path, symbol: str, interval: str = "3m", start: str = _START,
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


def test_mhs_coverage_root_override_backward_compatible(tmp_path, monkeypatch) -> None:
    # SCENARIO_MHS_COVERAGE_ROOT_OVERRIDE_BACKWARD_COMPATIBLE: calling
    # ``_coverage`` without ``root`` (the existing two call sites' calling
    # convention) still resolves under ``FUTURES_DATA_DIR / 'ohlcv'``.
    root = tmp_path / "canonical"
    (root / "ohlcv" / "3m").mkdir(parents=True)
    idx = pd.date_range(_START, _END, freq="3min", tz="UTC")
    pd.DataFrame({"timestamp": _epoch_ms(idx)}).to_parquet(root / "ohlcv" / "3m" / "BTCUSDT.parquet")
    monkeypatch.setattr(mhs_execution_collection, "FUTURES_DATA_DIR", root)
    result = mhs_execution_collection._coverage("BTCUSDT", "3m", _START, _END)
    assert result["status"] == "PRESENT"
    assert result["rows"] == 480


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


def test_mhs_execution_plan_reads_warmup_so_short_windows_plan(tmp_path, monkeypatch) -> None:
    idx = pd.date_range("2026-03-01", periods=4700, freq="1h", tz="UTC")
    quote = pd.DataFrame({f"S{i:02d}": float(i + 1) for i in range(16)}, index=idx)
    close = pd.DataFrame(
        {symbol: 100.0 + (i + 1) * pd.Series(range(len(idx)), index=idx)
         for i, symbol in enumerate(quote.columns)},
    )
    captured: dict[str, pd.Timestamp] = {}

    def _panel(root, interval, columns, start, end, **kwargs):
        captured["start"] = start
        captured["end"] = end
        return {"close": close, "quote_vol": quote}

    monkeypatch.setattr(mhs_execution_collection, "load_base_panel", _panel)
    monkeypatch.setattr(mhs_execution_collection, "funding_path", lambda symbol: tmp_path / f"{symbol}.parquet")
    for symbol in quote.columns:
        (tmp_path / f"{symbol}.parquet").touch()

    # When planning a 75-day forward window
    plan = mhs_execution_collection.build_mhs_execution_plan("2026-07-01", "2026-09-14", execution_universe_size=8)

    # Then the panel read starts one warmup before the window and symbols are selected
    assert captured["start"] == pd.Timestamp("2026-07-01", tz="UTC") - pd.Timedelta(
        hours=mhs_execution_collection.MHS_EXECUTION_PLAN_WARMUP_HOURS
    )
    assert captured["end"] == pd.Timestamp("2026-09-14", tz="UTC")
    assert len(plan.symbols) > 0
    assert plan.start == pd.Timestamp("2026-07-01", tz="UTC").isoformat()
