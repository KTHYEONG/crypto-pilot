"""Common index alignment and normalized trade concatenation for sleeve blends.

Shared by the fixed equal-weight and directional sleeve executions. This module
has no dependency on any other ``sleeve_blend`` execution module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.research.baseline.backtest import BacktestResult

_EMPTY_TRADE_COLUMNS = (
    "symbol",
    "entry_bar",
    "exit_bar",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "qty",
    "reason",
    "pnl",
    "return_pct",
    "funding_pnl",
    "side",
)


def _common_index(sleeve_results: dict[str, BacktestResult]) -> pd.DatetimeIndex:
    common = sorted(
        set.intersection(*(set(res.equity.index) for res in sleeve_results.values()))
    )
    if len(common) < 2:
        raise DataIntegrityError(
            f"sleeve equity curves share fewer than 2 common bars across "
            f"{sorted(sleeve_results)}"
        )
    return pd.DatetimeIndex(common)


def _equal_weight_blend(
    sleeve_results: dict[str, BacktestResult],
    common: pd.DatetimeIndex,
) -> pd.Series:
    """Equal-capital-weight blend of the per-sleeve equity curves.

    Each sleeve's equity is normalized to its common-index start value (equal
    capital weight) and the normalized curves are averaged pointwise; this is
    exactly the value of an equal-weight portfolio holding every sleeve from
    the common index start.
    """
    normalized: list[pd.Series] = []
    for res in sleeve_results.values():
        segment = res.equity.loc[common]
        normalized.append(segment / segment.iloc[0])
    blend = sum(normalized) / len(normalized)
    return pd.Series(blend, index=common, name="equity", dtype=np.float64)


def _concat_sleeve_trades(
    sleeve_results: dict[str, BacktestResult],
) -> pd.DataFrame:
    """Concatenate per-sleeve trades, tagged with ``symbol`` and wall-clock times.

    ``entry_time``/``exit_time`` are resolved from each sleeve's own equity
    index so holdout attribution never depends on a relative ``exit_bar`` whose
    meaning differs across sleeves.
    """
    frames: list[pd.DataFrame] = []
    for symbol, res in sleeve_results.items():
        trades = res.trades.copy()
        if len(trades) > 0:
            trades["symbol"] = symbol
            trades["entry_time"] = res.equity.index[
                trades["entry_bar"].astype(int).to_numpy()
            ].to_numpy()
            trades["exit_time"] = res.equity.index[
                trades["exit_bar"].astype(int).to_numpy()
            ].to_numpy()
        frames.append(trades)
    if not frames:
        return pd.DataFrame(columns=list(_EMPTY_TRADE_COLUMNS))
    return pd.concat(frames, ignore_index=True)
