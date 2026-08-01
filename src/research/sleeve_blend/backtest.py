from __future__ import annotations

from dataclasses import fields
from typing import cast

import numpy as np
import pandas as pd

from src.common.config import funding_path, ohlcv_path
from src.common.errors import DataIntegrityError
from src.common.logging import setup_logger
from src.market_data.storage.loaders import load_funding_rates, load_ohlcv_4h
from src.research.baseline.backtest import (
    BacktestResult,
    _run_directional_engine,
    run_backtest,
    run_directional_backtest,
)
from src.research.contracts import CostModel, StrategySpec
from src.research.evaluation.reliability import ReliabilityGateConfig
from src.research.sleeve_blend.contracts import (
    DirectionalSleeveSpec,
    FixedSleevePortfolioSpec,
)

_logger = setup_logger("SleeveBlendBacktest")

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


_LONG_SUFFIX = ":long"
_SHORT_SUFFIX = ":short"


def symbol_of_component(component: str) -> str:
    """Map a directional component label back to its symbol.

    Component labels encode the direction as ``"<SYMBOL>:long"`` /
    ``"<SYMBOL>:short"``; any other shape is a malformed contract input.
    """
    if component.endswith(_LONG_SUFFIX):
        return component[: -len(_LONG_SUFFIX)]
    if component.endswith(_SHORT_SUFFIX):
        return component[: -len(_SHORT_SUFFIX)]
    raise ValueError(f"malformed component label: {component}")


def component_labels(symbol: str) -> tuple[str, str]:
    """Long/short component labels for a directional sleeve symbol."""
    return (f"{symbol}{_LONG_SUFFIX}", f"{symbol}{_SHORT_SUFFIX}")


def _zero_weights(active_components: tuple[str, ...]) -> pd.Series:
    return pd.Series(0.0, index=list(active_components), dtype=np.float64)


def _cap_symbol_weights_np(
    weights: np.ndarray,
    symbol_ids: np.ndarray,
    n_symbols: int,
    max_symbol_weight: float,
) -> np.ndarray:
    """Numpy water-fill cap of per-component weights by symbol.

    Symbols whose aggregate free weight exceeds the cap are pinned at the cap
    and the remaining budget is split proportionally among the rest, so no
    symbol ever exceeds the cap (deterministic and convergent). When the cap is
    infeasible for the symbol count, the leftover budget is left unallocated
    rather than pushing any symbol over the cap.
    """
    agg = np.bincount(symbol_ids, weights=weights, minlength=n_symbols)
    order = np.argsort(-agg)
    final = np.zeros(n_symbols, dtype=np.float64)
    budget = 1.0
    i = 0
    while i < n_symbols:
        rem = order[i:]
        remaining_sum = float(agg[rem].sum())
        if remaining_sum <= 0:
            break
        count = 0
        for s in rem:
            if budget * agg[s] / remaining_sum <= max_symbol_weight:
                break
            count += 1
        if count == 0:
            final[rem] = budget * agg[rem] / remaining_sum
            break
        final[order[i : i + count]] = max_symbol_weight
        budget -= max_symbol_weight * count
        i += count

    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.divide(
            final[symbol_ids],
            np.where(agg[symbol_ids] > 0, agg[symbol_ids], 1.0),
        )
    return cast(np.ndarray, np.multiply(weights, scale))


def _symbol_ids_for(components: tuple[str, ...]) -> tuple[np.ndarray, int]:
    symbols = sorted({symbol_of_component(c) for c in components})
    lookup = {s: i for i, s in enumerate(symbols)}
    return (
        np.asarray([lookup[symbol_of_component(c)] for c in components], dtype=np.intp),
        len(symbols),
    )


def _cap_symbol_weights(
    weights: pd.Series,
    max_symbol_weight: float,
) -> pd.Series:
    """Cap each symbol's aggregated long+short weight and renormalize.

    Delegates to the numpy water-fill (see ``_cap_symbol_weights_np``) and
    returns a like-indexed Series, preserving the public contract surface.
    """
    symbol_ids, n_symbols = _symbol_ids_for(tuple(weights.index))
    capped = _cap_symbol_weights_np(
        weights.to_numpy(dtype=np.float64),
        symbol_ids,
        n_symbols,
        max_symbol_weight,
    )
    return pd.Series(capped, index=weights.index, dtype=np.float64)


def compute_causal_risk_weights(
    completed_returns: pd.DataFrame,
    active_components: tuple[str, ...],
    as_of: pd.Timestamp,
    history_days: int = 30,
    max_symbol_weight: float = 0.25,
) -> pd.Series:
    """Causal inverse-volatility risk weights over the trailing completed month.

    Uses strictly earlier marked returns (never the ``as_of`` bar or later
    data), weighting active components by ``1 / std`` of their completed
    30-day window. A symbol's aggregated long+short weight is capped at
    ``max_symbol_weight`` and the remainder is renormalized. An insufficient
    (non-full-month) history, fewer than two completed bars, or all
    zero/non-finite volatilities returns an all-zero weight vector so the
    candidate stays in cash.
    """
    if not isinstance(completed_returns.index, pd.DatetimeIndex):
        raise ValueError("completed_returns must have a DatetimeIndex")
    if not completed_returns.index.is_monotonic_increasing:
        raise ValueError("completed_returns index must be monotonic increasing")
    if len(active_components) == 0:
        raise ValueError("active_components must be non-empty")
    missing = [c for c in active_components if c not in completed_returns.columns]
    if missing:
        raise ValueError(f"active_components missing from returns: {missing}")
    if not isinstance(as_of, pd.Timestamp):
        raise ValueError("as_of must be a pd.Timestamp")
    if as_of.tzinfo is not None and completed_returns.index.tz is None:
        raise ValueError("as_of is tz-aware while returns index is tz-naive")
    if as_of.tzinfo is None and completed_returns.index.tz is not None:
        raise ValueError("as_of is tz-naive while returns index is tz-aware")
    if history_days < 1:
        raise ValueError(f"history_days must be >= 1, got {history_days}")
    if not 0.0 < max_symbol_weight <= 1.0:
        raise ValueError(
            f"max_symbol_weight must be in (0, 1], got {max_symbol_weight}"
        )

    window = completed_returns.loc[
        (completed_returns.index > as_of - pd.Timedelta(days=history_days))
        & (completed_returns.index < as_of)
    ]
    if len(window) < 2:
        return _zero_weights(active_components)
    if (as_of - window.index[0]).days < history_days - 1:
        return _zero_weights(active_components)

    vol = window[list(active_components)].std()
    valid = vol.notna() & (vol > 0)
    if not bool(valid.any()):
        return _zero_weights(active_components)

    weights = _zero_weights(active_components)
    weights.loc[valid.index[valid]] = 1.0 / vol[valid]
    weights = weights / float(weights.sum())
    return _cap_symbol_weights(weights, max_symbol_weight)


def _causal_weight_series(
    component_returns: pd.DataFrame,
    active_components: tuple[str, ...],
    history_days: int,
    max_symbol_weight: float,
) -> pd.DataFrame:
    """Vectorized causal inverse-vol weight series over the completed month.

    Mirrors ``compute_causal_risk_weights`` row by row (strictly-earlier
    completed returns, a full ``history_days`` window, inverse-vol among active
    components, per-symbol 0.25 cap + renormalization) using cumulative sums so
    a long sealed observation window computes in linear time.
    """
    idx = component_returns.index
    cols = list(active_components)
    x = component_returns[cols].to_numpy(dtype=np.float64)
    x_filled = np.where(np.isnan(x), 0.0, x)
    cum = np.cumsum(x_filled, axis=0)
    cum_sq = np.cumsum(x_filled * x_filled, axis=0)

    starts = np.asarray(
        idx.searchsorted(idx - pd.Timedelta(days=history_days), side="right")
    )
    n = len(idx)
    bar_pos = np.arange(n)
    counts = bar_pos - starts
    pos_mask = (bar_pos == 0)[:, None]
    cum_up_to_prev = np.where(pos_mask, 0.0, cum[bar_pos - 1])
    cum_sq_up_to_prev = np.where(pos_mask, 0.0, cum_sq[bar_pos - 1])
    start_mask = (starts == 0)[:, None]
    prev = np.where(start_mask, 0.0, cum[starts - 1])
    prev_sq = np.where(start_mask, 0.0, cum_sq[starts - 1])
    sums = cum_up_to_prev - prev
    sumsq = cum_sq_up_to_prev - prev_sq
    counts_col = np.maximum(counts, 1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sums / counts_col
        var = (sumsq - counts_col * mean * mean) / np.maximum(counts - 1, 1)[:, None]
    std = np.sqrt(np.clip(var, 0.0, None))

    span_days = (idx - pd.DatetimeIndex(idx[starts])).days.astype(np.float64)
    insufficient = (counts < 2) | (span_days < history_days - 1)
    std[insufficient] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / std
    valid = np.isfinite(inv)
    weights = np.zeros((n, len(cols)), dtype=np.float64)
    weights[valid] = inv[valid]
    row_sums = weights.sum(axis=1)
    active = row_sums > 0
    weights[active] = weights[active] / row_sums[active, None]

    symbol_ids, n_symbols = _symbol_ids_for(active_components)
    for i in np.flatnonzero(active):
        weights[i] = _cap_symbol_weights_np(
            weights[i], symbol_ids, n_symbols, max_symbol_weight
        )
    return pd.DataFrame(weights, index=idx, columns=cols)


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


def _check_contract() -> None:
    """Executable assertions locking the frozen sleeve-blend surface."""
    from inspect import signature  # noqa: PLC0415

    spec = FixedSleevePortfolioSpec(symbols=("BTCUSDT", "ETHUSDT"))
    assert spec.mdd_budget_fraction == 0.85
    assert {f.name for f in fields(FixedSleevePortfolioSpec)} == {
        "symbols", "mdd_budget_fraction",
    }
    params = signature(run_fixed_sleeve_portfolio).parameters
    assert list(params) == [
        "symbols", "start", "end", "costs", "mdd_budget_fraction",
        "initial_equity", "signal_delay_bars",
    ]
    assert run_fixed_sleeve_portfolio.__name__ == "run_fixed_sleeve_portfolio"
    assert np.isfinite(spec.mdd_budget_fraction)


_check_contract()
