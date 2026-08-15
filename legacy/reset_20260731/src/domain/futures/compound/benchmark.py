from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import L2BenchmarkConfig
from src.domain.futures.compound.contracts import CausalityError, L2BenchmarkSeries

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class DailyMarketReturns:
    timestamps_ns: NDArray[np.int64]
    returns_2d: NDArray[np.float64]
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.timestamps_ns.ndim != 1:
            raise ValueError("timestamps_ns must be 1-D")
        if self.returns_2d.ndim != 2:
            raise ValueError("returns_2d must be 2-D")
        if self.timestamps_ns.shape[0] != self.returns_2d.shape[0]:
            raise ValueError("timestamps_ns length must match returns_2d rows")
        if self.returns_2d.shape[1] != len(self.symbols):
            raise ValueError("returns_2d columns must match symbols length")
        if len(np.unique(self.timestamps_ns)) != len(self.timestamps_ns):
            raise ValueError("timestamps_ns must be unique")
        if not np.all(np.isfinite(self.returns_2d)):
            raise ValueError("returns_2d must be finite")


def aggregate_1h_close_to_daily_last(
    timestamps_ns: NDArray[np.int64], close_2d: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    ns_per_day = 24 * 3600 * 10**9
    day_start_ns = timestamps_ns - (timestamps_ns % ns_per_day)
    unique_days, inverse, _counts = np.unique(day_start_ns, return_inverse=True, return_counts=True)
    n_days = len(unique_days)
    n_syms = close_2d.shape[1]
    daily_close = np.empty((n_days, n_syms), dtype=np.float64)
    for i in range(n_days):
        mask = inverse == i
        daily_close[i] = close_2d[mask][-1]
    return unique_days, daily_close


def build_daily_market_returns(
    *, timestamps_ns: NDArray[np.int64], close_2d: NDArray[np.float64], symbols: tuple[str, ...],
) -> DailyMarketReturns:
    n_days = timestamps_ns.shape[0]
    n_syms = len(symbols)
    if n_days < 2:
        raise ValueError("need at least 2 daily timestamps to compute returns")
    returns_2d = np.empty((n_days, n_syms), dtype=np.float64)
    returns_2d[0, :] = 0.0
    prev = close_2d[:-1]
    curr = close_2d[1:]
    mask = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
    returns_2d[1:, :] = np.where(mask, curr / prev - 1.0, 0.0)
    return DailyMarketReturns(
        timestamps_ns=timestamps_ns,
        returns_2d=returns_2d,
        symbols=symbols,
    )


def _causal_volatility_scale(
    basket_returns: NDArray[np.float64],
    lookback: int,
    target_ann_vol: float,
) -> NDArray[np.float64]:
    n = len(basket_returns)
    scale = np.ones(n, dtype=np.float64)
    for d in range(n):
        if d < lookback:
            continue
        window = basket_returns[max(0, d - lookback):d]
        valid = window[np.isfinite(window)]
        if len(valid) < 10:
            continue
        realized_vol = float(np.std(valid, ddof=1)) * math.sqrt(365.25)
        if realized_vol > 1e-12:
            scale[d] = min(target_ann_vol / realized_vol, 3.0)
    return scale


def causal_beta_series(
    strategy_daily_1d: NDArray[np.float64],
    benchmark_daily_1d: NDArray[np.float64],
    *, lookback_days: int, min_obs: int, beta_clip: tuple[float, float],
) -> NDArray[np.float64]:
    if strategy_daily_1d.shape != benchmark_daily_1d.shape:
        raise ValueError("strategy and benchmark must have same length")
    n = len(strategy_daily_1d)
    beta = np.zeros(n, dtype=np.float64)
    for d in range(n):
        if d < lookback_days:
            continue
        start = max(0, d - lookback_days)
        s_win = np.log1p(strategy_daily_1d[start:d])
        b_win = np.log1p(benchmark_daily_1d[start:d])
        valid = np.isfinite(s_win) & np.isfinite(b_win)
        if np.sum(valid) < min_obs:
            continue
        s_valid = s_win[valid]
        b_valid = b_win[valid]
        b_var = np.var(b_valid, ddof=1)
        if b_var < 1e-12:
            continue
        cov = np.cov(b_valid, s_valid, ddof=1)[0, 1]
        raw_beta = cov / b_var
        beta[d] = np.clip(raw_beta, beta_clip[0], beta_clip[1])
        if raw_beta != beta[d]:
            _logger.info("[EVAL] beta_clipped= day=%d raw=%.4f clipped=%.4f", d, raw_beta, beta[d])
    return beta


def assert_contemporaneous_alignment(
    strategy_daily_1d: NDArray[np.float64],
    benchmark_daily_1d: NDArray[np.float64],
    *, max_lag: int = 1,
) -> None:
    if strategy_daily_1d.shape != benchmark_daily_1d.shape:
        raise ValueError("strategy and benchmark must have same length")
    n = len(strategy_daily_1d)
    if n < max_lag + 2:
        return
    s = np.log1p(strategy_daily_1d)
    b = np.log1p(benchmark_daily_1d)
    valid = np.isfinite(s) & np.isfinite(b)
    s = s[valid]
    b = b[valid]
    if len(s) < max_lag + 2:
        return
    s_var = float(np.var(s, ddof=1))
    b_var = float(np.var(b, ddof=1))
    if s_var < 1e-12 or b_var < 1e-12:
        return
    lag0 = np.corrcoef(s, b)[0, 1]
    lags: list[float] = []
    for lag in range(1, max_lag + 1):
        r = np.corrcoef(s[lag:], b[:-lag])[0, 1]
        lags.append(r)
    max_lag_corr = max(abs(r) for r in lags) if lags else 0.0
    _logger.info(
        "[DATA] alignment lag0=%.4f lag±1=%s status=%s",
        lag0, [f"{r:.4f}" for r in lags],
        "ok" if abs(lag0) >= max_lag_corr else "shift_detected",
    )
    if max_lag_corr > abs(lag0) + 0.1:
        raise CausalityError(
            f"contemporaneous correlation {lag0:.4f} < max lag-{max_lag} correlation {max_lag_corr:.4f}: "
            f"series are not contemporaneously aligned"
        )


def build_causal_l2_benchmark(
    *, daily_market_returns: DailyMarketReturns,
    window_timestamps_ns: NDArray[np.int64], config: L2BenchmarkConfig,
) -> L2BenchmarkSeries:
    if config.mode == "cash_collateral":
        raise NotImplementedError("cash_collateral benchmark mode not yet implemented")

    daily_ts = daily_market_returns.timestamps_ns

    if not np.all(np.diff(window_timestamps_ns) > 0):
        raise CausalityError("window_timestamps_ns must be strictly increasing")

    start_idx = int(np.searchsorted(daily_ts, window_timestamps_ns[0], side="left"))
    end_idx = int(np.searchsorted(daily_ts, window_timestamps_ns[-1], side="right"))
    if start_idx >= len(daily_ts) or end_idx > len(daily_ts):
        raise CausalityError("window days outside daily timestamp range")
    if start_idx >= end_idx:
        raise CausalityError("empty aligned window")
    aligned_n = end_idx - start_idx
    window_n = len(window_timestamps_ns)
    if aligned_n != window_n:
        raise CausalityError(
            f"window length mismatch: daily={aligned_n} vs window={window_n} "
        )

    warmup_start = max(0, start_idx - config.volatility_lookback_days)
    all_sym_returns = daily_market_returns.returns_2d
    warmup_n = end_idx - warmup_start
    warmup_basket = np.zeros(warmup_n, dtype=np.float64)
    for sym, w in zip(config.crypto_symbols, config.crypto_weights, strict=True):
        try:
            sym_pos = daily_market_returns.symbols.index(sym)
        except ValueError as err:
            raise ValueError(f"missing benchmark symbol: {sym}") from err
        warmup_basket += w * all_sym_returns[warmup_start:end_idx, sym_pos]

    full_scale = _causal_volatility_scale(warmup_basket, config.volatility_lookback_days, config.target_ann_vol)
    causal_scale = full_scale[-window_n:]
    aligned_basket = warmup_basket[-window_n:]
    scaled_returns = aligned_basket * causal_scale

    benchmark_id = f"{config.mode}_{'_'.join(config.crypto_symbols)}_{config.volatility_lookback_days}d"
    return L2BenchmarkSeries(
        benchmark_id=benchmark_id,
        timestamps_ns=window_timestamps_ns,
        daily_returns_1d=scaled_returns,
        causal_scale_1d=causal_scale,
    )
