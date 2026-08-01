"""Fixed equal-weight MDD-budget sleeve blend execution.

Calibrates one static leverage scalar from the base-cost blend and applies it
under stress without re-calibrating. Depends on ``common`` only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import ohlcv_path
from src.common.errors import DataIntegrityError
from src.market_data.storage.loaders import load_ohlcv_4h
from src.research.baseline.backtest import BacktestResult, run_backtest
from src.research.contracts import CostModel, StrategySpec
from src.research.evaluation.reliability import ReliabilityGateConfig
from src.research.sleeve_blend.common import (
    _common_index,
    _concat_sleeve_trades,
    _equal_weight_blend,
)
from src.research.sleeve_blend.contracts import FixedSleevePortfolioSpec


def _run_blend(
    *,
    symbols: tuple[str, ...],
    start: str | None,
    end: str | pd.Timestamp | None,
    costs: CostModel,
    initial_equity: float,
    signal_delay_bars: int,
    lev: float | None,
    mdd_budget_fraction: float | None = None,
) -> tuple[pd.Series, pd.DataFrame, float]:
    if lev is not None:
        budget = 0.85
    elif mdd_budget_fraction is not None:
        budget = mdd_budget_fraction
    else:
        raise ValueError("mdd_budget_fraction is required when lev is not fixed")
    spec = FixedSleevePortfolioSpec(symbols=symbols, mdd_budget_fraction=budget)
    if initial_equity <= 0:
        raise ValueError(f"initial_equity must be > 0, got {initial_equity}")
    if signal_delay_bars < 0:
        raise ValueError(f"signal_delay_bars must be >= 0, got {signal_delay_bars}")

    sleeve_results: dict[str, BacktestResult] = {}
    for symbol in spec.symbols:
        df = load_ohlcv_4h(ohlcv_path(symbol, "1h"), start=start, end=end)
        sleeve_results[symbol] = run_backtest(
            df, StrategySpec(symbol=symbol), costs, signal_delay_bars=signal_delay_bars,
        )

    common = _common_index(sleeve_results)
    blend = _equal_weight_blend(sleeve_results, common)

    blended_mdd = float((blend / blend.cummax() - 1.0).min())
    if lev is None:
        if blended_mdd >= 0:
            raise DataIntegrityError(
                "unlevered blended MDD must be < 0 for leverage calibration, got "
                f"{blended_mdd:.6f}"
            )
        target_mdd = ReliabilityGateConfig().mdd_floor * spec.mdd_budget_fraction
        lev = target_mdd / blended_mdd

    returns = blend.pct_change().fillna(0.0)
    equity = (1.0 + returns * lev).cumprod() * initial_equity
    equity = pd.Series(equity, index=common, name="equity", dtype=np.float64)
    trades = _concat_sleeve_trades(sleeve_results)
    return equity, trades, lev


def run_fixed_sleeve_portfolio(
    symbols: tuple[str, ...],
    start: str | None,
    end: str | pd.Timestamp | None,
    costs: CostModel,
    mdd_budget_fraction: float,
    initial_equity: float = 10_000.0,
    signal_delay_bars: int = 0,
) -> BacktestResult:
    """Run one equal-weight fixed-sleeve Donchian blend with MDD-budget leverage.

    Each sleeve is executed with the frozen, unmodified ``run_backtest`` under
    the frozen ``StrategySpec``/``CostModel`` (no changes to any existing
    contract). The per-sleeve equity curves are equal-capital-weight blended on
    their common index, and one static leverage scalar is calibrated as
    ``mdd_floor * mdd_budget_fraction`` divided by the blend's realized
    unlevered MDD, then applied to the blended return stream. Calibration
    divides by realized MDD, so a non-negative unlevered blended MDD raises
    ``DataIntegrityError`` (fail closed rather than silently skipping leverage).
    """
    equity, trades, _lev = _run_blend(
        symbols=symbols, start=start, end=end, costs=costs,
        initial_equity=initial_equity, signal_delay_bars=signal_delay_bars,
        lev=None, mdd_budget_fraction=mdd_budget_fraction,
    )
    return BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame())


def run_fixed_sleeve_portfolio_calibrated(
    symbols: tuple[str, ...],
    start: str | None,
    end: str | pd.Timestamp | None,
    costs: CostModel,
    mdd_budget_fraction: float,
    initial_equity: float = 10_000.0,
    signal_delay_bars: int = 0,
) -> tuple[BacktestResult, float]:
    """Calibrate the blend leverage from the base-cost run and return it.

    The application service uses this for the base-cost run so the stress re-run
    can apply the *same* frozen scalar instead of re-calibrating around stressed
    costs. ``lev`` is the calibration as in ``run_fixed_sleeve_portfolio``.
    """
    equity, trades, lev = _run_blend(
        symbols=symbols, start=start, end=end, costs=costs,
        initial_equity=initial_equity, signal_delay_bars=signal_delay_bars,
        lev=None, mdd_budget_fraction=mdd_budget_fraction,
    )
    return BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame()), lev


def run_fixed_sleeve_portfolio_with_leverage(
    symbols: tuple[str, ...],
    start: str | None,
    end: str | pd.Timestamp | None,
    costs: CostModel,
    lev: float,
    initial_equity: float = 10_000.0,
    signal_delay_bars: int = 0,
) -> BacktestResult:
    """Apply a pre-frozen leverage scalar to the blend without re-calibrating.

    Used by the stress re-run so stress genuinely tests robustness under the
    base-cost calibration rather than re-fitting leverage around stressed costs.
    """
    if lev <= 0:
        raise ValueError(f"lev must be > 0, got {lev}")
    equity, trades, _lev = _run_blend(
        symbols=symbols, start=start, end=end, costs=costs,
        initial_equity=initial_equity, signal_delay_bars=signal_delay_bars,
        lev=lev,
    )
    return BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame())


__all__ = [
    "run_fixed_sleeve_portfolio",
    "run_fixed_sleeve_portfolio_calibrated",
    "run_fixed_sleeve_portfolio_with_leverage",
]
