"""Portfolio weights: Ledoit–Wolf covariance, fractional Kelly → vol target → constrained net.

Expected return ``mu`` is in **simple return per bar** (same units as bar-to-bar returns).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def cov_lookback_bars(tf: str, opt_cfg: dict[str, Any]) -> int:
    """~30 calendar days in bars for the bar width of *tf*."""
    key = str(tf).strip().lower()
    by_tf = opt_cfg.get("FUTURES_PORTFOLIO_COV_LOOKBACK_BY_TF") or {}
    if isinstance(by_tf, dict) and key in by_tf:
        return max(5, int(by_tf[key]))
    return max(5, int(opt_cfg.get("FUTURES_PORTFOLIO_COV_LOOKBACK", 180)))


def rolling_ledoit_wolf_cov(
    returns_hist: np.ndarray, *, min_obs: int = 20
) -> np.ndarray:
    """returns_hist shape (T, N). PSD covariance of last row's distribution estimate."""
    r = np.asarray(returns_hist, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] < 2:
        raise ValueError("returns_hist must be 2-D with T>=2")
    if r.shape[0] < min_obs:
        v = np.var(r, axis=0, ddof=1)
        v = np.where(np.isfinite(v) & (v > 1e-18), v, 1e-8)
        return np.diag(v)
    lw = LedoitWolf().fit(r)
    return np.asarray(lw.covariance_, dtype=np.float64)


def mu_from_cross_section_signals(
    xs_long_prev: np.ndarray, xs_short_prev: np.ndarray
) -> np.ndarray:
    """Legacy CS z-score of (long − short); prefer :func:`mu_net_from_composer_channels`."""
    xl = np.asarray(xs_long_prev, dtype=np.float64).ravel()
    xs_ = np.asarray(xs_short_prev, dtype=np.float64).ravel()
    d = xl - xs_
    m = float(np.nanmean(d))
    sd = float(np.nanstd(d, ddof=1))
    if not np.isfinite(sd) or sd < 1e-9:
        return np.zeros_like(d, dtype=np.float64)
    z = (d - m) / sd
    return z * 5e-4


def mu_net_from_composer_channels(
    mu_long_prev: np.ndarray, mu_short_prev: np.ndarray
) -> np.ndarray:
    """Per-symbol net μ from signal-composer outputs (simple return per bar scale)."""
    sl = np.asarray(mu_long_prev, dtype=np.float64).ravel()
    ss = np.asarray(mu_short_prev, dtype=np.float64).ravel()
    return sl - ss


def _vol_ann_from_per_bar_sigma(sigma_port_bar: float, bars_per_year: float) -> float:
    return float(sigma_port_bar * math.sqrt(max(bars_per_year, 1e-9)))


def _kelly_raw(
    mu: np.ndarray, sigma_diag: np.ndarray, *, f_kelly_max: float, eps: float = 1e-12
) -> np.ndarray:
    var = np.maximum(sigma_diag**2, eps)
    f = mu / var
    return np.clip(f, -abs(f_kelly_max), abs(f_kelly_max))


def _apply_ls_balance(w: np.ndarray, *, lo: float = 0.5, hi: float = 2.0) -> np.ndarray:
    """Rescale long / short legs so gross long / gross short ∈ [lo, hi] ratio."""
    out = np.asarray(w, dtype=np.float64).copy()
    for _ in range(24):
        long_m = float(np.sum(out[out > 0]))
        short_m = float(np.sum(np.abs(out[out < 0])))
        if short_m < 1e-12 and long_m < 1e-12:
            return out
        if short_m < 1e-12 or long_m < 1e-12:
            return out  # 단방향 포트폴리오는 그대로 통과 (regime-driven direction 허용)
        r = long_m / short_m
        if r < lo:
            scale_l = min(hi / max(r, 1e-12), 10.0)
            out = np.where(out > 0, out * scale_l, out)
        elif r > hi:
            scale_s = min(r / lo, 10.0)
            out = np.where(out < 0, out * scale_s, out)
        else:
            break
    return out


def _project_l1_linf(
    w_pre: np.ndarray, *, gross_cap: float, per_symbol_cap: float
) -> np.ndarray:
    """L1 gross + per-symbol caps via SLSQP (small n)."""
    n = int(w_pre.size)
    w0 = np.asarray(w_pre, dtype=np.float64).ravel()
    cap = float(per_symbol_cap) * float(gross_cap)

    def objective(x: np.ndarray) -> float:
        return float(np.sum((x - w0) ** 2))

    cons = (
        {"type": "ineq", "fun": lambda x: float(gross_cap) - float(np.sum(np.abs(x)))},
    )
    bounds = [(-cap, cap) for _ in range(n)]
    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 400, "ftol": 1e-12},
    )
    x = np.asarray(res.x, dtype=np.float64)
    if not res.success:
        x = np.clip(w0, -cap, cap)
        g1 = float(np.sum(np.abs(x)))
        if g1 > gross_cap + 1e-9 and g1 > 0:
            x *= gross_cap / g1
    return x


def solve_constrained_weights(
    mu: np.ndarray,
    Sigma: np.ndarray,
    *,
    kappa: float,
    f_kelly_max: float,
    sigma_target_ann: float,
    bars_per_year: float,
    gross_cap: float,
    per_symbol_cap: float,
    current_dd: float,
    kelly_sigma_diag: np.ndarray | None = None,
) -> np.ndarray:
    """Return signed portfolio weights (fractions of equity, before leverage)."""
    mu_v = np.asarray(mu, dtype=np.float64).ravel()
    n = int(mu_v.size)
    if kelly_sigma_diag is not None:
        sig = np.asarray(kelly_sigma_diag, dtype=np.float64).ravel()
        if sig.size != n:
            raise ValueError("kelly_sigma_diag size must match mu")
        sig = np.maximum(sig, 1e-12)
    else:
        sig = np.sqrt(np.clip(np.diag(np.asarray(Sigma, dtype=np.float64)), 1e-12, None))
    w_raw = kappa * _kelly_raw(mu_v, sig, f_kelly_max=f_kelly_max)

    Sigma = np.asarray(Sigma, dtype=np.float64)
    if Sigma.shape != (n, n):
        Sigma = np.diag(np.diag(Sigma))

    port_var_bar = float(w_raw @ Sigma @ w_raw)
    sigma_pb = math.sqrt(max(port_var_bar, 1e-18))
    ann = _vol_ann_from_per_bar_sigma(sigma_pb, bars_per_year)
    if ann > 1e-12:
        w_pre = w_raw * (float(sigma_target_ann) / ann)
    else:
        w_pre = w_raw * 0.0

    w_c = _project_l1_linf(w_pre, gross_cap=gross_cap, per_symbol_cap=per_symbol_cap)
    # L/S balance 강제 제거: regime 신호(HMM)가 방향성을 결정하므로 단방향 포트폴리오 허용

    dd = float(max(0.0, current_dd))
    dd_scale = float(np.clip(1.0 - max(0.0, dd - 0.05) / 0.10, 0.3, 1.0))
    return w_c * dd_scale


def precompute_rebalance_weights(
    close_2d: np.ndarray,
    xs_long: np.ndarray,
    xs_short: np.ndarray,
    *,
    rebalance_bars: int,
    lookback: int,
    bars_per_year: float,
    kappa: float,
    f_kelly_max: float,
    sigma_target_ann: float,
    gross_cap: float,
    per_symbol_cap: float,
    current_dd: float = 0.0,
    min_obs: int = 20,
    composer_sigma_2d: np.ndarray | None = None,
) -> np.ndarray:
    """Sparse target weights: nonzero when ``i > 0`` and ``i % rebalance_bars == 0``."""
    c = np.asarray(close_2d, dtype=np.float64)
    xl = np.asarray(xs_long, dtype=np.float64)
    xs_ = np.asarray(xs_short, dtype=np.float64)
    n_bars, n_syms = c.shape
    out = np.zeros((n_bars, n_syms), dtype=np.float64)
    rb = max(1, int(rebalance_bars))
    lb = max(5, int(lookback))

    for i in range(n_bars):
        if i == 0:
            continue
        if (i % rb) != 0:
            continue
        start_i = max(0, i - lb)
        hist = c[start_i:i, :]
        if hist.shape[0] < 2:
            continue
        rr = np.diff(hist, axis=0) / np.maximum(hist[:-1, :], 1e-12)
        rr = np.nan_to_num(rr, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = rolling_ledoit_wolf_cov(rr, min_obs=min_obs)
        sl = np.nan_to_num(xl[i - 1, :], nan=0.0, posinf=0.0, neginf=0.0)
        ss = np.nan_to_num(xs_[i - 1, :], nan=0.0, posinf=0.0, neginf=0.0)
        mu = mu_net_from_composer_channels(sl, ss)
        ks_diag = None
        if composer_sigma_2d is not None:
            ks_row = np.asarray(composer_sigma_2d[i - 1, :], dtype=np.float64).ravel()
            ks_diag = np.maximum(ks_row, 1e-12)
        out[i, :] = solve_constrained_weights(
            mu,
            sigma,
            kappa=float(kappa),
            f_kelly_max=float(f_kelly_max),
            sigma_target_ann=float(sigma_target_ann),
            bars_per_year=float(bars_per_year),
            gross_cap=float(gross_cap),
            per_symbol_cap=float(per_symbol_cap),
            current_dd=float(current_dd),
            kelly_sigma_diag=ks_diag,
        )

    return out


def portfolio_weight_params_from_optuna(
    params: dict[str, Any], opt_cfg: dict[str, Any]
) -> dict[str, Any]:
    pol = opt_cfg.get("FUTURES_PORTFOLIO_POLICY", {})
    tf = str(params.get("TIMEFRAME", "4h"))
    return {
        "lookback": cov_lookback_bars(tf, opt_cfg),
        "kappa": float(
            params.get("PORTFOLIO_KAPPA", opt_cfg.get("FUTURES_PORTFOLIO_KAPPA", 0.35))
        ),
        "f_kelly_max": float(
            params.get(
                "PORTFOLIO_F_KELLY_MAX",
                opt_cfg.get("FUTURES_PORTFOLIO_F_KELLY_MAX", 2.0),
            )
        ),
        "sigma_target_ann": float(
            params.get(
                "TARGET_ANN_VOL",
                pol.get("target_ann_vol", 0.45),
            )
        ),
        "gross_cap": float(
            params.get("MAX_EXPOSURE", pol.get("gross_exposure_cap", 1.2))
        ),
        "per_symbol_cap": float(
            params.get("MAX_EXPOSURE_PER_COIN", pol.get("per_symbol_cap", 0.25))
        ),
    }
