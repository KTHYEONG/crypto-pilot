"""EWMA-variance Kelly scaling (RiskMetrics-style) with fat-tail nu clamp; replaces MLE t-GARCH for speed."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

_DEGENERATE_RET_STD: float = 1e-8
# RiskMetrics-style decay (lambda); variance_t = lambda * var_{t-1} + (1-lambda) * r_{t-1}^2
_EWMA_LAMBDA: float = 0.94
# Shrink nu estimate toward nu_fallback after clip (reduces kurtosis noise)
_NU_SHRINKAGE: float = 0.5
_NU_CLIP_LOW: float = 3.5
_NU_CLIP_HIGH: float = 10.0

_GARCH_CACHE_MAXSIZE: int = 64
_garch_cache_lock: threading.Lock = threading.Lock()
_garch_cache: OrderedDict[tuple[int, int, float, int, int], pd.Series] = OrderedDict()


def _garch_close_fingerprint(close: pd.Series) -> int:
    """Lightweight invalidation when OHLCV rows change but length stays equal (aligned with spot signal cache)."""
    c = close.astype(np.float64).to_numpy()
    n = len(c)
    if n == 0:
        return 0
    head = c[: min(5, n)]
    tail = c[max(0, n - 5) :]
    h = hash((tuple(head.tolist()), tuple(tail.tolist()), n))
    return int(h & ((1 << 63) - 1))


def _t_kelly_fraction_vec(mu: np.ndarray, sigma2: np.ndarray, nu: float) -> np.ndarray:
    """Vectorized t-Kelly for EWMA path."""
    nu_c = float(np.clip(nu, 3.0, 30.0))
    if nu_c <= 2.01:
        return np.full_like(mu, 0.25, dtype=np.float64)
    num = mu * (nu_c - 2.0)
    den = sigma2 * (nu_c - 1.0)
    ok = (sigma2 > 1e-18) & (np.abs(den) >= 1e-18)
    f = np.where(ok, num / den, 0.25)
    return np.clip(f, -2.0, 2.0)


def _kelly_to_multiplier_vec(k: np.ndarray) -> np.ndarray:
    return np.clip(0.2 + 0.4 * np.minimum(1.0, 1.0 / (1.0 + np.abs(k))), 0.15, 1.0)


def _riskmetrics_ewma_variance(log_ret: np.ndarray, lam: float) -> np.ndarray:
    """Recursive EWMA variance: h_t = lam * h_{t-1} + (1-lam) * r_{t-1}^2."""
    n = len(log_ret)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    warm = min(30, max(2, n))
    v0 = float(np.var(log_ret[:warm])) + 1e-12
    out[0] = max(v0, 1e-18)
    one_m_lam = 1.0 - lam
    for i in range(1, n):
        out[i] = lam * out[i - 1] + one_m_lam * (log_ret[i - 1] ** 2)
    return out


def _nu_from_full_window_excess_kurtosis(
    log_ret: np.ndarray,
    w: int,
    nu_fallback: float,
) -> float:
    """One-shot excess kurtosis on post-warmup returns; clip + shrink toward nu_fallback."""
    seg = log_ret[int(w) :] if len(log_ret) > int(w) else log_ret
    if seg.size < 50:
        return float(np.clip(nu_fallback, _NU_CLIP_LOW, _NU_CLIP_HIGH))
    x = seg - float(np.mean(seg))
    m2 = float(np.mean(x * x))
    if m2 < 1e-18:
        return float(np.clip(nu_fallback, _NU_CLIP_LOW, _NU_CLIP_HIGH))
    m4 = float(np.mean(x * x * x * x))
    excess_kurt = m4 / (m2 * m2) - 3.0
    if excess_kurt > 1e-6:
        nu_est = 4.0 + 6.0 / excess_kurt
    else:
        nu_est = float(nu_fallback)
    nu_est = float(np.clip(nu_est, _NU_CLIP_LOW, _NU_CLIP_HIGH))
    nu_final = _NU_SHRINKAGE * nu_est + (1.0 - _NU_SHRINKAGE) * float(nu_fallback)
    return float(np.clip(nu_final, _NU_CLIP_LOW, _NU_CLIP_HIGH))


def _compute_garch_kelly_series_core(
    close: pd.Series,
    *,
    window: int,
    retrain_freq: int,
    nu_fallback: float,
) -> pd.Series:
    """Vectorized EWMA Kelly series (no MLE; retrain_freq accepted for API compatibility, ignored)."""
    _ = retrain_freq
    n = len(close)
    out = np.full(n, 0.5, dtype=np.float64)
    log_ret = np.log(close.astype(np.float64).clip(lower=1e-12))
    log_ret = log_ret.diff().fillna(0.0).to_numpy(dtype=np.float64)

    w = max(120, int(window))
    if n <= w:
        return pd.Series(out, index=close.index)

    if float(np.std(log_ret[w:])) < _DEGENERATE_RET_STD:
        _logger.debug("Degenerate returns after warmup; Kelly series stays neutral.")
        return pd.Series(out, index=close.index)

    nu = _nu_from_full_window_excess_kurtosis(log_ret, w, nu_fallback)
    sigma2 = _riskmetrics_ewma_variance(log_ret, _EWMA_LAMBDA)
    mu_short = (
        pd.Series(log_ret).rolling(30, min_periods=1).mean().to_numpy(dtype=np.float64)
    )

    var_seg = np.maximum(sigma2[w:], 1e-12)
    mu_seg = mu_short[w:]
    k_vec = _t_kelly_fraction_vec(mu_seg, var_seg, nu)
    out[w:] = _kelly_to_multiplier_vec(k_vec)

    return pd.Series(out, index=close.index)


def compute_garch_kelly_series(
    close: pd.Series,
    *,
    window: int,
    retrain_freq: int = 24,
    nu_fallback: float = 5.0,
) -> pd.Series:
    """
    Per-bar fractional-Kelly multiplier in (0, 1] from EWMA variance and fixed nu (kurtosis shrink).
    Retrain frequency is ignored (kept for Optuna search-space compatibility).
    """
    n = len(close)
    w = max(120, int(window))
    rf = max(1, int(retrain_freq))
    fp = _garch_close_fingerprint(close)
    cache_key: tuple[int, int, float, int, int] = (
        w,
        rf,
        float(nu_fallback),
        n,
        fp,
    )

    with _garch_cache_lock:
        if cache_key in _garch_cache:
            _garch_cache.move_to_end(cache_key)
            return _garch_cache[cache_key].copy()

    series = _compute_garch_kelly_series_core(
        close,
        window=window,
        retrain_freq=retrain_freq,
        nu_fallback=nu_fallback,
    )

    with _garch_cache_lock:
        while len(_garch_cache) >= _GARCH_CACHE_MAXSIZE:
            _garch_cache.popitem(last=False)
        _garch_cache[cache_key] = series.copy()

    return series
