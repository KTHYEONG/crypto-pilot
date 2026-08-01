"""Funding input load and directional sleeve execution.

Runs the funding-gated long/short sleeve with causal risk weights, and the
fixed-weight stress re-run. Depends on ``common`` and ``weights``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import funding_path, ohlcv_path
from src.common.errors import DataIntegrityError
from src.market_data.storage.loaders import load_funding_rates, load_ohlcv_4h
from src.research.baseline.backtest import (
    BacktestResult,
    _run_directional_engine,
    run_directional_backtest,
)
from src.research.contracts import CostModel, StrategySpec
from src.research.sleeve_blend.common import _common_index, _concat_sleeve_trades
from src.research.sleeve_blend.contracts import DirectionalSleeveSpec
from src.research.sleeve_blend.weights import (
    _LONG_SUFFIX,
    _SHORT_SUFFIX,
    _causal_weight_series,
    component_labels,
)


def _load_directional_sleeve_data(
    symbols: tuple[str, ...],
    start: str | None,
    end: str | pd.Timestamp | None,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Load each symbol's 4h bars and its funding clipped to the bar window.

    Funding events before the first bar or after the last bar are outside the
    executed window and are excluded exactly like the baseline evaluation does;
    a symbol with no bars or no funding is a fail-closed integrity error.
    """
    data: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for symbol in symbols:
        df = load_ohlcv_4h(ohlcv_path(symbol, "1h"), start=start, end=end)
        if len(df) < 2:
            raise DataIntegrityError(f"no 4h bars for {symbol} in the window")
        funding = load_funding_rates(funding_path(symbol))
        bar_period = df.index[1] - df.index[0]
        window_end = df.index[-1] + bar_period
        funding = funding[
            (funding.index >= df.index[0]) & (funding.index < window_end)
        ]
        if len(funding) == 0:
            raise DataIntegrityError(f"no funding events for {symbol} in the window")
        data[symbol] = (df, funding)
    return data


def _run_directional_sleeve_core(
    *,
    symbols: tuple[str, ...],
    start: str | None,
    end: str | pd.Timestamp | None,
    costs: CostModel,
    initial_equity: float,
    signal_delay_bars: int,
    fixed_weights: pd.DataFrame | None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Directional sleeve core: per-symbol ledgers + causal risk-weight blend.

    Returns ``(equity, trades, weights)`` where ``weights`` is the per-bar
    risk-budget series actually applied (computed causally when
    ``fixed_weights`` is ``None``, otherwise reused verbatim for the stress
    run).
    """
    spec = DirectionalSleeveSpec(symbols=symbols)
    if initial_equity <= 0:
        raise ValueError(f"initial_equity must be > 0, got {initial_equity}")
    if signal_delay_bars < 0:
        raise ValueError(f"signal_delay_bars must be >= 0, got {signal_delay_bars}")
    active_components = tuple(
        comp for s in spec.symbols for comp in component_labels(s)
    )

    data = _load_directional_sleeve_data(symbols, start, end)
    sleeve_results: dict[str, BacktestResult] = {}
    component_equities: dict[str, pd.Series] = {}
    for symbol, (df, funding) in data.items():
        strategy = StrategySpec(symbol=symbol)
        sleeve_results[symbol] = run_directional_backtest(
            df, strategy, costs, funding,
            initial_equity=initial_equity, signal_delay_bars=signal_delay_bars,
        )
        for direction, suffix in (("long", _LONG_SUFFIX), ("short", _SHORT_SUFFIX)):
            component_equities[f"{symbol}{suffix}"] = _run_directional_engine(
                df, strategy, costs, funding, side=direction,
                initial_equity=initial_equity, signal_delay_bars=signal_delay_bars,
            ).equity

    common = _common_index(sleeve_results)
    component_returns = pd.DataFrame({
        comp: component_equities[comp].loc[common].pct_change()
        for comp in active_components
    })

    if fixed_weights is not None:
        weights = fixed_weights.reindex(common).fillna(0.0)
        weights = weights.loc[:, [c for c in weights.columns if c in active_components]]
    else:
        weights = _causal_weight_series(
            component_returns,
            active_components,
            history_days=spec.history_days,
            max_symbol_weight=spec.max_symbol_weight,
        )

    returns = pd.Series(0.0, index=common, dtype=np.float64)
    for bar in common[1:]:
        returns.loc[bar] = float((weights.loc[bar] * component_returns.loc[bar]).sum())
    equity = (1.0 + returns).cumprod() * initial_equity
    equity = pd.Series(equity, index=common, name="equity", dtype=np.float64)
    trades = _concat_sleeve_trades(sleeve_results)
    return equity, trades, weights


def run_directional_sleeve_portfolio(
    symbols: tuple[str, ...],
    start: str | None,
    end: str | pd.Timestamp | None,
    costs: CostModel,
    initial_equity: float = 10_000.0,
) -> BacktestResult:
    """Run the unlevered funding-gated directional sleeve blend (leverage 1.0)."""
    equity, trades, _weights = _run_directional_sleeve_core(
        symbols=symbols, start=start, end=end, costs=costs,
        initial_equity=initial_equity, signal_delay_bars=0, fixed_weights=None,
    )
    return BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame())


def run_directional_sleeve_portfolio_with_weights(
    symbols: tuple[str, ...],
    start: str | None,
    end: str | pd.Timestamp | None,
    costs: CostModel,
    initial_equity: float = 10_000.0,
) -> tuple[BacktestResult, pd.DataFrame]:
    """Run the base directional sleeve and return its causal weight series.

    The application service uses the returned weights for the stress re-run so
    stress reuses the base allocation instead of re-fitting weights around
    stressed costs.
    """
    equity, trades, weights = _run_directional_sleeve_core(
        symbols=symbols, start=start, end=end, costs=costs,
        initial_equity=initial_equity, signal_delay_bars=0, fixed_weights=None,
    )
    return (
        BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame()),
        weights,
    )


def run_directional_sleeve_portfolio_fixed_weights(
    symbols: tuple[str, ...],
    start: str | None,
    end: str | pd.Timestamp | None,
    costs: CostModel,
    weights: pd.DataFrame,
    initial_equity: float = 10_000.0,
    signal_delay_bars: int = 1,
) -> BacktestResult:
    """Re-run the directional sleeve under a frozen base allocation.

    Used by the stress run: the component backtests execute under stressed
    costs and a one-bar delay while the causal weight series from the base run
    is reused verbatim (never re-calibrated).
    """
    equity, trades, _weights = _run_directional_sleeve_core(
        symbols=symbols, start=start, end=end, costs=costs,
        initial_equity=initial_equity, signal_delay_bars=signal_delay_bars,
        fixed_weights=weights,
    )
    return BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame())


__all__ = [
    "run_directional_sleeve_portfolio",
    "run_directional_sleeve_portfolio_fixed_weights",
    "run_directional_sleeve_portfolio_with_weights",
]
