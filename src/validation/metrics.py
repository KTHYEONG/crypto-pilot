from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.logging_setup import setup_logger

_logger = setup_logger("Metrics")


@dataclass
class Metrics:
    cagr: float
    mdd: float
    sharpe: float
    sortino: float
    calmar: float
    profit_factor: float
    expectancy: float
    win_rate: float
    payoff_ratio: float
    trade_count: int
    exposure: float
    turnover: float
    trades_per_year: dict[str, int]


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    *,
    bars_per_year: int = 2190,
) -> Metrics:
    if len(equity) < 2 or equity.iloc[-1] <= 0:
        return _empty_metrics()

    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 86400)
    if years <= 0:
        return _empty_metrics()

    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1

    running_max = equity.cummax()
    dd = equity / running_max - 1
    mdd = float(dd.min())

    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return _empty_metrics()

    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret * np.sqrt(bars_per_year)) if std_ret > 0 else 0.0

    neg_returns = returns[returns < 0]
    neg_std = neg_returns.std()
    sortino = (mean_ret / neg_std * np.sqrt(bars_per_year)) if neg_std > 0 else 0.0

    calmar = cagr / abs(mdd) if mdd != 0 else 0.0

    trade_count = len(trades)
    if trade_count == 0:
        return Metrics(
            cagr=cagr, mdd=mdd, sharpe=sharpe, sortino=sortino,
            calmar=calmar, profit_factor=0.0, expectancy=0.0,
            win_rate=0.0, payoff_ratio=0.0, trade_count=0,
            exposure=0.0, turnover=0.0, trades_per_year={},
        )

    winning = trades[trades["pnl"] > 0]
    losing = trades[trades["pnl"] < 0]

    win_rate = len(winning) / trade_count if trade_count > 0 else 0.0

    total_win = float(winning["pnl"].sum()) if len(winning) > 0 else 0.0
    total_loss = float(losing["pnl"].sum()) if len(losing) > 0 else 0.0

    profit_factor = total_win / abs(total_loss) if abs(total_loss) > 0 else float("inf")

    avg_win = total_win / len(winning) if len(winning) > 0 else 0.0
    avg_loss = total_loss / len(losing) if len(losing) > 0 else 0.0
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss) if trade_count > 0 else 0.0

    in_position = equity.diff() != 0
    in_position.iloc[0] = False
    exposure = float(in_position.sum() / len(equity)) if len(equity) > 0 else 0.0

    if "entry_bar" in trades.columns and len(trades) > 0:
        avg_bars = trades["entry_bar"].diff().dropna().mean()
        turnover = 1.0 / avg_bars if avg_bars and avg_bars > 0 else 0.0
    else:
        turnover = 0.0

    trades_per_year: dict[str, int] = {}
    if "entry_bar" in trades.columns:
        for _, tr in trades.iterrows():
            ts = equity.index[int(tr["entry_bar"])] if int(tr["entry_bar"]) < len(equity) else None
            if ts is not None:
                year = str(ts.year)
                trades_per_year[year] = trades_per_year.get(year, 0) + 1

    _logger.info(
        "cagr=%.4f mdd=%.4f sharpe=%.3f trades=%d pf=%.3f",
        cagr, mdd, sharpe, trade_count, profit_factor,
        extra={"tag": "EVAL"},
    )
    return Metrics(
        cagr=cagr, mdd=mdd, sharpe=sharpe, sortino=sortino,
        calmar=calmar, profit_factor=profit_factor,
        expectancy=expectancy, win_rate=win_rate,
        payoff_ratio=payoff_ratio, trade_count=trade_count,
        exposure=exposure, turnover=turnover,
        trades_per_year=trades_per_year,
    )


def _empty_metrics() -> Metrics:
    return Metrics(
        cagr=0.0, mdd=0.0, sharpe=0.0, sortino=0.0,
        calmar=0.0, profit_factor=0.0, expectancy=0.0,
        win_rate=0.0, payoff_ratio=0.0, trade_count=0,
        exposure=0.0, turnover=0.0, trades_per_year={},
    )
