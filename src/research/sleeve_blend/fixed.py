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
from src.research.sleeve_blend.contracts import (
    CausalLeverageSpec,
    FixedSleevePortfolioSpec,
)


def build_causal_leverage_schedule(
    unit_equity: pd.Series,
    risk_spec: CausalLeverageSpec,
) -> pd.Series:
    """Build the ex-ante causal gross-leverage schedule over a unit-leverage ledger.

    Each schedule row at bar ``t`` sees only completed unit-leverage marks
    strictly before ``t``: the trailing realized drawdown is the worst
    ``mark / running-peak - 1`` within the trailing ``lookback_days`` window of
    prior marks, and the row is
    ``abs(mdd_floor) * risk_budget_fraction / abs(trailing_mdd)`` bounded by the
    source-controlled hard ``max_gross_leverage`` cap. Before a complete
    lookback exists the row is zero exposure. The computation is fully
    vectorized (no ``pd.apply``) and is deterministic for identical input.
    """
    if not isinstance(unit_equity.index, pd.DatetimeIndex) or len(unit_equity) < 2:
        raise ValueError(
            "unit_equity must be a DatetimeIndex series with at least 2 points"
        )
    if not unit_equity.index.is_monotonic_increasing:
        raise ValueError("unit_equity index must be monotonic increasing")
    values = unit_equity.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("unit_equity must contain only finite values")
    if (values <= 0).any():
        raise ValueError("unit_equity must be strictly positive")

    bar_period = unit_equity.index[1] - unit_equity.index[0]
    lookback_bars = max(
        1,
        round(pd.Timedelta(days=risk_spec.lookback_days) / bar_period),
    )

    # At bar t the schedule may use marks 0..t-1 only.
    shifted = np.concatenate([[np.nan], values[:-1]])
    running_peak = np.maximum.accumulate(np.where(np.isnan(shifted), -np.inf, shifted))
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown_prefix = shifted / running_peak - 1.0
    drawdown = pd.Series(drawdown_prefix, index=unit_equity.index, dtype=np.float64)
    trailing_mdd = drawdown.rolling(
        lookback_bars, min_periods=lookback_bars
    ).min()

    target_mdd = abs(ReliabilityGateConfig().mdd_floor) * risk_spec.risk_budget_fraction
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = target_mdd / trailing_mdd.abs()
    schedule = raw.clip(upper=risk_spec.max_gross_leverage).fillna(0.0)
    schedule = schedule.where(
        trailing_mdd.isna() | (trailing_mdd < 0.0), risk_spec.max_gross_leverage
    )
    schedule = schedule.fillna(0.0)
    return pd.Series(schedule, index=unit_equity.index, name="leverage", dtype=np.float64)


def apply_leverage_schedule(
    unit_equity: pd.Series,
    schedule: pd.Series,
    initial_equity: float = 10_000.0,
) -> pd.Series:
    """Apply a frozen per-bar leverage schedule to a unit-leverage equity ledger.

    The schedule is reused verbatim (aligned by timestamp, missing rows default
    to zero exposure) and never re-fitted, so a stressed replay of the same
    returns stream shares the base allocation exactly.
    """
    if initial_equity <= 0:
        raise ValueError(f"initial_equity must be > 0, got {initial_equity}")
    aligned = schedule.reindex(unit_equity.index).fillna(0.0).to_numpy(dtype=np.float64)
    returns = unit_equity.pct_change().fillna(0.0).to_numpy(dtype=np.float64)
    equity = (1.0 + returns * aligned).cumprod() * initial_equity
    return pd.Series(equity, index=unit_equity.index, name="equity", dtype=np.float64)


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


def run_fixed_sleeve_portfolio_with_schedule(
    symbols: tuple[str, ...],
    start: str | None,
    end: str | pd.Timestamp | None,
    costs: CostModel,
    schedule: pd.Series,
    initial_equity: float = 10_000.0,
    signal_delay_bars: int = 0,
) -> BacktestResult:
    """Run the equal-weight fixed-sleeve blend under a frozen causal schedule.

    Executes the unlevered blend exactly as ``run_fixed_sleeve_portfolio``
    would at leverage 1.0 and applies ``schedule`` verbatim via
    :func:`apply_leverage_schedule`; the schedule is never re-fitted around the
    executed costs or delay.
    """
    equity, trades, _lev = _run_blend(
        symbols=symbols, start=start, end=end, costs=costs,
        initial_equity=initial_equity, signal_delay_bars=signal_delay_bars,
        lev=1.0,
    )
    scheduled = apply_leverage_schedule(equity, schedule, initial_equity=initial_equity)
    return BacktestResult(equity=scheduled, trades=trades, signals=pd.DataFrame())


__all__ = [
    "apply_leverage_schedule",
    "build_causal_leverage_schedule",
    "run_fixed_sleeve_portfolio",
    "run_fixed_sleeve_portfolio_calibrated",
    "run_fixed_sleeve_portfolio_with_leverage",
    "run_fixed_sleeve_portfolio_with_schedule",
]
