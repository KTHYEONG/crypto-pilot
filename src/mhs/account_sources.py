"""Account-scale replay inputs assembled from one frozen candidate source load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.account_ledger import AccountMarkPanels
from src.mhs.frozen_research_candidate import FrozenMhsCandidate
from src.mhs.frozen_research_run import FrozenSourceContext
from src.mhs.resources import assert_mhs_stage_allocation

_MARKS_COLUMNS = ("open", "high", "low", "close")


def _read_held_marks(symbol: str, root: str, grid: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Read one held symbol's 3m bars onto the replay grid, forward-filling the last mark."""
    path = Path(root) / "3m" / f"{symbol}.parquet"
    if not path.exists():
        raise DataIntegrityError(f"account marks miss 3m source for {symbol}")
    frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns or any(col not in frame.columns for col in _MARKS_COLUMNS):
        raise DataIntegrityError(f"account marks 3m source malformed for {symbol}")
    stamps = pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms", utc=True)
    ordered = frame.set_index(pd.DatetimeIndex(stamps)).sort_index()
    windowed = ordered.loc[grid[0] : grid[-1]]
    return {
        column: windowed[column].reindex(grid).ffill().astype("float32")
        for column in ("close", "high", "low")
    }


def _cumulative_funding_on_grid(series: pd.Series | None, grid: pd.DatetimeIndex) -> np.ndarray:
    """Settle each funding print on the first 3m bar at or after its timestamp, then cumulate."""
    step = np.zeros(len(grid))
    if series is not None and len(series):
        stamps = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True))
        positions = grid.searchsorted(stamps, side="left")
        live = positions < len(grid)
        np.add.at(step, positions[live], series.to_numpy(dtype="float64")[live])
    return np.cumsum(step)


def causal_adv_sigma(
    daily_quote_volume: pd.DataFrame, daily_close: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Causal ADV and daily sigma on the daily grid.

    ADV is the 30-day median daily quote volume shifted one day and sigma is the
    21-day std of daily log returns shifted one day, so neither includes the day
    it is labelled on.
    """
    adv = daily_quote_volume.rolling(30, min_periods=1).median().shift(1)
    closes = daily_close
    with np.errstate(divide="ignore", invalid="ignore"):
        log_returns = np.log(closes / closes.shift(1))
    daily_sigma = log_returns.rolling(21, min_periods=1).std().shift(1)
    return adv, daily_sigma


def assemble_account_inputs(
    candidate: FrozenMhsCandidate, context: FrozenSourceContext,
) -> tuple[pd.DataFrame, AccountMarkPanels, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Unit weights (held columns only), 3m marks, funding cumulative on the 3m grid, and
    causal ADV (30-day median daily quote volume, shifted one day) and daily sigma (21-day std of
    daily log returns, shifted one day) for the candidate's held symbols."""
    weights = candidate.target_weights
    held = [column for column in weights.columns if bool((weights[column] != 0).any())]
    entries = weights.index
    grid = pd.date_range(entries[0], entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    assert_mhs_stage_allocation(
        stage="account_marks", estimated_bytes=len(grid) * len(held) * 12,
        budget=context.budget, replay=False, initial_swap_bytes=None,
    )
    columns: dict[str, dict[str, pd.Series]] = {
        symbol: _read_held_marks(symbol, context.root, grid) for symbol in held
    }
    marks = AccountMarkPanels(
        close=pd.DataFrame({symbol: columns[symbol]["close"] for symbol in held}, index=grid),
        high=pd.DataFrame({symbol: columns[symbol]["high"] for symbol in held}, index=grid),
        low=pd.DataFrame({symbol: columns[symbol]["low"] for symbol in held}, index=grid),
    )
    funding_grid = pd.DataFrame(
        {symbol: _cumulative_funding_on_grid(context.funding_by_symbol.get(symbol), grid) for symbol in held},
        index=grid,
    )
    # 원장은 진입일 단위로 펀딩 증분을 정산하므로 누적값을 각 진입 시각에서 표본화한다.
    funding_cum = funding_grid.reindex(entries)
    adv_full, sigma_full = causal_adv_sigma(context.daily_quote_volume[held], context.daily_close[held])
    adv = adv_full.reindex(entries)
    daily_sigma = sigma_full.reindex(entries)
    return weights[held], marks, funding_cum, adv, daily_sigma
