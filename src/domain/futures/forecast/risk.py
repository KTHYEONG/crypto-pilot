"""RiskForecast builder — Ledoit-Wolf cov + BTC beta (strict causal) + residual var."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.domain.futures.forecast.contracts import RiskForecast

_logger = logging.getLogger(__name__)


def _rolling_residual_variance(
    sym_ret: np.ndarray,  # [T, N]
    btc_ret: np.ndarray,  # [T]
    beta_2d: np.ndarray,  # [T, N]
    lookback: int,
) -> np.ndarray:
    """Compute per-bar rolling residual variance (strict causal).

    Args:
        sym_ret: Symbol returns [T, N].
        btc_ret: BTC returns [T].
        beta_2d: Rolling beta [T, N].
        lookback: Rolling window in bars.

    Returns:
        Residual variance [T, N], non-negative.

    """
    t, n = sym_ret.shape
    lb = max(3, int(lookback))
    residual_ret = sym_ret - beta_2d * btc_ret[:, np.newaxis]
    residual_var = np.zeros((t, n), dtype=np.float64)
    for i in range(1, t):
        start = max(0, i - lb)
        window = residual_ret[start:i, :]
        if window.shape[0] >= 2:
            residual_var[i, :] = np.nanvar(window, axis=0, ddof=1)
    return np.maximum(residual_var, 0.0)


def _rolling_beta_strict_causal(
    sym_ret: np.ndarray,  # [T, N]
    btc_ret: np.ndarray,  # [T]
    lookback: int,
) -> tuple[np.ndarray, str]:
    """Compute BTC beta per bar using strict causal window [0:t] (excludes bar t).

    Args:
        sym_ret: Symbol returns [T, N].
        btc_ret: BTC index returns [T].
        lookback: Rolling window in bars.

    Returns:
        Tuple of (beta [T, N], beta_source string).

    """
    t, n = sym_ret.shape
    lb = max(3, int(lookback))
    beta = np.zeros((t, n), dtype=np.float64)
    for i in range(1, t):
        start = max(0, i - lb)
        x = btc_ret[start:i]
        if x.size < 2:
            continue
        var_x = float(np.var(x, ddof=1))
        if var_x < 1e-14:
            continue
        x_m = float(np.mean(x))
        for j in range(n):
            y = sym_ret[start:i, j]
            cov_xy = float(np.mean((x - x_m) * (y - float(np.mean(y)))))
            beta[i, j] = float(np.nan_to_num(cov_xy / var_x, nan=0.0))
    return beta, "trailing_btc"


def build_risk_forecast(
    close_2d: np.ndarray,
    symbols: list[str],
    tf: str,
    cfg: dict[str, Any],
    *,
    btc_index: int | None = None,
    lookback: int = 60,
    min_obs: int = 20,
    vol_lookback: int = 20,
) -> RiskForecast:
    """Build typed RiskForecast: LW covariance + BTC beta + residual variance + vol.

    All computations are strictly causal: bar t uses only data from [0, t).

    Args:
        close_2d: Close prices [T, N].
        symbols: Symbol names [N].
        tf: Timeframe string (unused in computation, for metadata).
        cfg: Config dict.
        btc_index: Column index of BTC in close_2d. If None, uses symbol search.
        lookback: Rolling window in bars for covariance and beta.
        min_obs: Minimum observations for Ledoit-Wolf (fallback to identity).
        vol_lookback: Rolling window for vol and residual variance.

    Returns:
        Typed RiskForecast.

    """
    from src.domain.futures.portfolio.portfolio_constructor import precompute_rolling_covariances

    c = np.asarray(close_2d, dtype=np.float64)
    t, n = c.shape

    # Symbol returns (simple, causal) — shape [T, N]
    sym_ret = np.zeros((t, n), dtype=np.float64)
    sym_ret[1:] = (c[1:] - c[:-1]) / np.maximum(np.abs(c[:-1]), 1e-12)
    sym_ret = np.nan_to_num(sym_ret, nan=0.0)

    # Forecast vol (rolling std, causal) — shape [T, N]
    lb_v = max(3, int(vol_lookback))
    forecast_vol_2d = np.zeros((t, n), dtype=np.float64)
    for i in range(1, t):
        start = max(0, i - lb_v)
        window = sym_ret[start:i, :]
        if window.shape[0] >= 2:
            forecast_vol_2d[i, :] = np.nanstd(window, axis=0, ddof=1)

    # Covariance (existing function, strict causal) — shape [T, N, N]
    covariance_3d = precompute_rolling_covariances(c, lookback=lookback, min_obs=min_obs)

    # BTC index detection
    beta_source = "unavailable"
    beta_2d = np.zeros((t, n), dtype=np.float64)
    btc_ret = np.zeros(t, dtype=np.float64)

    btc_col = btc_index
    if btc_col is None:
        for idx, sym in enumerate(symbols):
            if "BTC" in sym.upper():
                btc_col = idx
                break

    if btc_col is not None and 0 <= int(btc_col) < n:
        btc_ret = sym_ret[:, int(btc_col)]
        beta_2d, beta_source = _rolling_beta_strict_causal(sym_ret, btc_ret, lookback)
    else:
        _logger.debug("RiskForecast: BTC column not found, beta=0 (unavailable)")

    # Residual variance — shape [T, N]
    residual_var_2d = _rolling_residual_variance(sym_ret, btc_ret, beta_2d, lb_v)

    return RiskForecast(
        covariance_3d=covariance_3d,
        beta_2d=beta_2d,
        residual_var_2d=residual_var_2d,
        forecast_vol_2d=forecast_vol_2d,
        beta_source=beta_source,
        source="lw_btc_residual_v1",
    )
