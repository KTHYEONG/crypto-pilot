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
    if windowed.empty:
        raise DataIntegrityError(f"account marks miss 3m source for {symbol}")
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
) -> tuple[pd.DataFrame, AccountMarkPanels, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Unit weights (held columns only), 3m marks, funding cumulative at each entry's anchor,
    causal ADV and daily sigma, and the per-entry submit anchors for the candidate's held symbols.

    The anchor of an entry is the last 3m bar whose label is at or before the entry's
    ``signal_available_at`` -- the bar whose close is the last price observable when the order
    is submitted on the next bar. This is the canonical inventory ledger's ``submit_bar``
    anchor and the live submission time; anchoring at the entry label instead delays every
    rebalance by the release-to-entry gap (one hour for the frozen book) and understates CAGR.

    The 3m grid starts at the first anchor so that every anchor is a grid bar. Funding is
    cumulated on the grid (each print settles on the first bar at or after its timestamp) and
    sampled at the anchors, so the step between consecutive entries is the funding paid on the
    inventory held between their anchors. The returned funding, ADV and sigma frames stay
    indexed by the entry labels of ``candidate.target_weights``. ADV and sigma for an entry
    are taken at its decision day, so they only use daily bars completed before the 23:00
    release (identical to the live frozen book).

    Returns:
        ``(unit_weights, marks, funding_cum, adv, daily_sigma, anchor_times)`` where
        ``anchor_times[i]`` is the anchor bar label of ``unit_weights.index[i]``.

    Raises:
        DataIntegrityError: A held symbol has no or malformed 3m source, an entry's release
            precedes the first grid bar, or two entries share one anchor bar.
    """
    weights = candidate.target_weights
    held = [column for column in weights.columns if bool((weights[column] != 0).any())]
    entries = weights.index
    releases = pd.DatetimeIndex(pd.to_datetime(candidate.signal_available_at, utc=True))
    anchor_times = releases.floor("3min").tz_convert("UTC")
    if bool(anchor_times.duplicated().any()):
        raise DataIntegrityError("two entries share one anchor bar")
    grid = pd.date_range(anchor_times[0], entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    if bool((releases < grid[0]).any()):
        raise DataIntegrityError("entry release precedes the first grid bar")
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
    sampled = funding_grid.reindex(anchor_times)
    sampled.index = entries
    funding_cum = sampled
    adv_full, sigma_full = causal_adv_sigma(context.daily_quote_volume[held], context.daily_close[held])
    decision_idx = entries - pd.Timedelta(days=1)
    adv = adv_full.reindex(decision_idx)
    adv.index = entries
    daily_sigma = sigma_full.reindex(decision_idx)
    daily_sigma.index = entries
    return weights[held], marks, funding_cum, adv, daily_sigma, anchor_times
