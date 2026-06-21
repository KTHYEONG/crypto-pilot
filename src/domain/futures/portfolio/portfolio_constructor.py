"""Portfolio weights: Ledoit-Wolf covariance, fractional Kelly -> vol target -> constrained net.

Expected return ``mu`` is in **simple return per bar** (same units as bar-to-bar returns).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf

from src.domain.futures.portfolio.portfolio_optimizer import PortfolioPolicyInputs


def cov_lookback_bars(tf: str, opt_cfg: dict[str, Any]) -> int:
    """~30 calendar days in bars for the bar width of *tf*."""
    key = str(tf).strip().lower()
    by_tf = opt_cfg.get("FUTURES_PORTFOLIO_COV_LOOKBACK_BY_TF") or {}
    if isinstance(by_tf, dict) and key in by_tf:
        return max(5, int(by_tf[key]))
    return max(5, int(opt_cfg.get("FUTURES_PORTFOLIO_COV_LOOKBACK", 180)))


def rolling_ledoit_wolf_cov(returns_hist: np.ndarray, *, min_obs: int = 20) -> np.ndarray:
    """returns_hist shape (T, N). PSD covariance of last row's distribution estimate."""
    r = np.asarray(returns_hist, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("returns_hist must be 2-D")
    if r.shape[0] < 2:
        return np.eye(r.shape[1]) * 1e-8
    if r.shape[0] < min_obs:
        v = np.var(r, axis=0, ddof=1)
        v = np.where(np.isfinite(v) & (v > 1e-18), v, 1e-8)
        return np.diag(v)
    lw = LedoitWolf().fit(r)
    return np.asarray(lw.covariance_, dtype=np.float64)


def mu_from_cross_section_signals(
    xs_long_prev: np.ndarray, xs_short_prev: np.ndarray
) -> np.ndarray:
    """Legacy CS z-score of (long - short); prefer :func:`mu_net_from_composer_channels`."""
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


KELLY_FRACTION: float = 0.25  # Fractional Kelly — 변경 금지


def _kelly_raw(
    mu: np.ndarray, sigma_diag: np.ndarray, *, f_kelly_max: float, eps: float = 1e-12
) -> np.ndarray:
    var = np.maximum(sigma_diag**2, eps)
    f = mu / var
    return np.asarray(np.clip(f, -abs(f_kelly_max), abs(f_kelly_max)), dtype=np.float64)


def _kelly_scaled(
    mu: np.ndarray,
    sigma_diag: np.ndarray,
    *,
    f_kelly_max: float,
) -> np.ndarray:
    """Fractional Kelly weight = _kelly_raw * KELLY_FRACTION.

    KELLY_FRACTION은 모듈 상수(0.25)이며 외부 주입 불가.
    """
    raw = _kelly_raw(mu, sigma_diag, f_kelly_max=f_kelly_max)
    return raw * KELLY_FRACTION


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
    return np.asarray(out, dtype=np.float64)


@numba.njit(cache=True)  # type: ignore[untyped-decorator]
def _project_l1_linf_numba(
    w_pre: np.ndarray, gross_cap: float, per_symbol_cap: float
) -> np.ndarray:
    """Fast L1 gross + per-symbol caps via iterative scaling and clipping (Numba)."""
    out = w_pre.copy()
    n = out.size
    if n == 0:
        return out

    cap = float(per_symbol_cap) * float(gross_cap)

    # 1. Immediate per-symbol clipping
    for i in range(n):
        if out[i] > cap:
            out[i] = cap
        elif out[i] < -cap:
            out[i] = -cap

    # 2. Iterative Gross Cap adjustment (max 10 iterations)
    for _ in range(10):
        g1 = 0.0
        for i in range(n):
            g1 += abs(out[i])

        if g1 <= gross_cap + 1e-9:
            break

        # Scale down to meet gross cap
        scale = gross_cap / max(g1, 1e-12)
        for i in range(n):
            out[i] *= scale

    return out


def _project_l1_linf(w_pre: np.ndarray, *, gross_cap: float, per_symbol_cap: float) -> np.ndarray:
    """Fast L1 gross + per-symbol caps via iterative scaling and clipping."""
    return np.asarray(
        _project_l1_linf_numba(w_pre, float(gross_cap), float(per_symbol_cap)),
        dtype=np.float64,
    )


def solve_constrained_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    kappa: float,
    f_kelly_max: float,
    sigma_target_ann: float,
    bars_per_year: float,
    gross_cap: float,
    per_symbol_cap: float,
    current_dd: float,
    kelly_sigma_diag: np.ndarray | None = None,
    bl_shrinkage_var_mult: float = 0.20,
    bl_shrinkage_omega_mult: float = 0.10,
) -> np.ndarray:
    """Return signed portfolio weights (fractions of equity, before leverage)."""
    mu_v = np.asarray(mu, dtype=np.float64).ravel()
    n = int(mu_v.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    if kelly_sigma_diag is not None:
        sig = np.asarray(kelly_sigma_diag, dtype=np.float64).ravel()
        if sig.size != n:
            raise ValueError("kelly_sigma_diag size must match mu")
        sig = np.maximum(sig, 1e-12)
    else:
        sig = np.sqrt(np.clip(np.diag(np.asarray(sigma, dtype=np.float64)), 1e-12, None))

    sigma_mat = np.asarray(sigma, dtype=np.float64)
    if sigma_mat.shape != (n, n):
        sigma_mat = np.diag(np.diag(sigma_mat))

    # Black-Litterman 사후 기대수익률 산출
    tau = 1.0
    omega_diag = np.maximum(sig ** 2, 1e-12)
    try:
        # Adaptive Diagonal Shrinkage (20% mean variance)
        mean_var = float(np.mean(np.diag(sigma_mat))) * bl_shrinkage_var_mult
        inv_tau_sigma = np.linalg.pinv(tau * sigma_mat + np.eye(n) * (mean_var + 1e-6))
        inv_omega = np.diag(1.0 / omega_diag)
        mean_inv_omega = float(np.mean(1.0 / omega_diag)) * bl_shrinkage_omega_mult
        bl_cov = np.linalg.pinv(inv_tau_sigma + inv_omega + np.eye(n) * (mean_inv_omega + 1e-6))
        mu_bl = bl_cov @ (inv_omega @ mu_v)
    except Exception:
        mu_bl = mu_v

    w_raw = kappa * _kelly_raw(mu_bl, sig, f_kelly_max=f_kelly_max)

    port_var_bar = float(w_raw @ sigma_mat @ w_raw)
    sigma_pb = math.sqrt(max(port_var_bar, 1e-18))
    ann = _vol_ann_from_per_bar_sigma(sigma_pb, bars_per_year)
    w_pre = w_raw * (float(sigma_target_ann) / ann) if ann > 1e-12 else w_raw * 0.0

    w_c = _project_l1_linf(w_pre, gross_cap=gross_cap, per_symbol_cap=per_symbol_cap)

    _ = current_dd
    return w_c


def precompute_rolling_covariances(
    close_2d: np.ndarray, lookback: int, min_obs: int = 20
) -> np.ndarray:
    """Precompute rolling Ledoit-Wolf covariance matrices for all bars.

    Called once during optimization precompute phase to eliminate 1M+ redundant
    calls during trials. Optimized via threading parallel dispatch.
    """
    c = np.asarray(close_2d, dtype=np.float64)
    n_bars, n_syms = c.shape
    out = np.zeros((n_bars, n_syms, n_syms), dtype=np.float64)
    lb = max(5, int(lookback))

    from joblib import Parallel, delayed

    def _compute_single_cov(i: int) -> tuple[int, np.ndarray]:
        start_i = max(0, i - lb)
        hist = c[start_i:i, :]
        if hist.shape[0] < 3:
            cov = np.zeros((n_syms, n_syms), dtype=np.float64)
            for j in range(n_syms):
                cov[j, j] = 1e-6
            return i, cov

        rr = np.diff(hist, axis=0) / np.maximum(hist[:-1, :], 1e-12)
        rr = np.nan_to_num(rr, nan=0.0, posinf=0.0, neginf=0.0)
        return i, rolling_ledoit_wolf_cov(rr, min_obs=min_obs)

    # Scikit-learn algorithms run GIL-free inside C routines;
    # we leverage threading backend to bypass serialization costs
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(_compute_single_cov)(i) for i in range(1, n_bars)
    )

    for i, cov in results:
        out[i] = cov

    return out


@numba.njit(cache=True)  # type: ignore[untyped-decorator]
def _solve_constrained_weights_numba(
    mu: np.ndarray,
    sigma: np.ndarray,
    kappa: float,
    f_kelly_max: float,
    sigma_target_ann: float,
    bars_per_year: float,
    gross_cap: float,
    per_symbol_cap: float,
    current_dd: float,
    kelly_sigma_diag: np.ndarray | None = None,
    bl_shrinkage_var_mult: float = 0.20,
    bl_shrinkage_omega_mult: float = 0.10,
) -> np.ndarray:
    n = mu.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    mu_mod = mu.copy()
    dyn_gross_cap = gross_cap

    if kelly_sigma_diag is not None:
        sig = kelly_sigma_diag
    else:
        sig = np.zeros(n, dtype=np.float64)
        for i in range(n):
            sig[i] = math.sqrt(max(sigma[i, i], 1e-12))

    # Black-Litterman 사후 기대수익률 산출 (Numba)
    tau = 1.0
    mean_var = 0.0
    for i in range(n):
        mean_var += sigma[i, i]
    mean_var = (mean_var / n) * bl_shrinkage_var_mult

    tau_sigma = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            tau_sigma[i, j] = tau * sigma[i, j]
        tau_sigma[i, i] += mean_var + 1e-6

    inv_tau_sigma = np.linalg.inv(tau_sigma)
    inv_omega_diag = np.zeros(n, dtype=np.float64)
    mean_inv_omega = 0.0
    for i in range(n):
        val = 1.0 / max(sig[i] ** 2, 1e-12)
        inv_omega_diag[i] = val
        mean_inv_omega += val
    mean_inv_omega = (mean_inv_omega / n) * bl_shrinkage_omega_mult

    a_mat = inv_tau_sigma.copy()
    for i in range(n):
        a_mat[i, i] += inv_omega_diag[i] + mean_inv_omega + 1e-6

    bl_cov = np.linalg.inv(a_mat)
    rhs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        rhs[i] = inv_omega_diag[i] * mu_mod[i]

    mu_bl = bl_cov @ rhs

    # _kelly_raw logic
    w_raw = np.zeros(n, dtype=np.float64)
    f_max = abs(f_kelly_max)
    for i in range(n):
        var = max(sig[i] ** 2, 1e-12)
        f = mu_bl[i] / var
        if f > f_max:
            f = f_max
        elif f < -f_max:
            f = -f_max
        w_raw[i] = kappa * f

    # Sigma interaction
    port_var_bar = 0.0
    for i in range(n):
        for j in range(n):
            port_var_bar += w_raw[i] * sigma[i, j] * w_raw[j]

    sigma_pb = math.sqrt(max(port_var_bar, 1e-18))
    ann = sigma_pb * math.sqrt(max(bars_per_year, 1e-9))

    w_pre = w_raw * (sigma_target_ann / ann) if ann > 1e-12 else w_raw * 0.0

    w_c = _project_l1_linf_numba(w_pre, dyn_gross_cap, per_symbol_cap)

    _ = current_dd
    return np.asarray(w_c, dtype=np.float64)


@numba.njit(cache=True)  # type: ignore[untyped-decorator]
def _precompute_loop_numba(
    n_bars: int,
    n_syms: int,
    rb: int,
    mu_2d: np.ndarray,
    sigma_3d: np.ndarray,
    kappa: float,
    f_kelly_max: float,
    sigma_target_ann: float,
    bars_per_year: float,
    gross_cap: float,
    per_symbol_cap: float,
    current_dd: float,
    ks_diag_2d: np.ndarray | None = None,
    bl_shrinkage_var_mult: float = 0.20,
    bl_shrinkage_omega_mult: float = 0.10,
) -> np.ndarray:
    out = np.zeros((n_bars, n_syms), dtype=np.float64)
    for i in range(1, n_bars):
        if (i % rb) != 0:
            continue

        mu = mu_2d[i - 1]
        sigma = sigma_3d[i]
        ks_diag = ks_diag_2d[i - 1] if ks_diag_2d is not None else None

        out[i, :] = _solve_constrained_weights_numba(
            mu,
            sigma,
            kappa,
            f_kelly_max,
            sigma_target_ann,
            bars_per_year,
            gross_cap,
            per_symbol_cap,
            current_dd,
            kelly_sigma_diag=ks_diag,
            bl_shrinkage_var_mult=bl_shrinkage_var_mult,
            bl_shrinkage_omega_mult=bl_shrinkage_omega_mult,
        )
    return out


@dataclass(frozen=True)
class RiskSnapshot:
    """Factor-lite risk snapshot for rebalance projection.

    Attributes:
        covariance_3d: Per-bar covariance cube with shape [T, N, N].
        beta_2d: Optional per-bar beta matrix with shape [T, N].
        residual_var_2d: Optional per-bar residual variance matrix with shape [T, N].

    """

    covariance_3d: np.ndarray
    beta_2d: np.ndarray | None = None
    residual_var_2d: np.ndarray | None = None


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
    sigma_3d: np.ndarray | None = None,
    risk_snapshot: RiskSnapshot | None = None,
    btc_beta_2d: np.ndarray | None = None,
    policy_inputs: PortfolioPolicyInputs | None = None,
    use_residual_var_for_kelly: bool = False,
    bl_shrinkage_var_mult: float = 0.20,
    bl_shrinkage_omega_mult: float = 0.10,
) -> np.ndarray:
    """Sparse target weights: precomputed or rolling LW covariance."""
    c = np.asarray(close_2d, dtype=np.float64)
    xl = np.asarray(xs_long, dtype=np.float64)
    xs_ = np.asarray(xs_short, dtype=np.float64)
    n_bars, n_syms = c.shape
    rb = max(1, int(rebalance_bars))

    # Prepare inputs for Numba loop
    mu_2d = xl - xs_
    if policy_inputs is not None:
        mu_long = policy_inputs.mu_long_2d
        mu_short = policy_inputs.mu_short_2d
        if mu_long is not None and mu_short is not None:
            mu_long_2d = np.asarray(mu_long, dtype=np.float64)
            mu_short_2d = np.asarray(mu_short, dtype=np.float64)
            if mu_long_2d.shape == mu_2d.shape and mu_short_2d.shape == mu_2d.shape:
                mu_2d = mu_long_2d - mu_short_2d
    mu_2d = np.nan_to_num(mu_2d, nan=0.0, posinf=0.0, neginf=0.0)

    sigma_input = sigma_3d
    if policy_inputs is not None and policy_inputs.risk_sigma_3d is not None:
        sigma_input = np.asarray(policy_inputs.risk_sigma_3d, dtype=np.float64)
    if policy_inputs is not None and btc_beta_2d is None and policy_inputs.risk_beta_2d is not None:
        btc_beta_2d = np.asarray(policy_inputs.risk_beta_2d, dtype=np.float64)
    if risk_snapshot is not None:
        sigma_input = np.asarray(risk_snapshot.covariance_3d, dtype=np.float64)
        if btc_beta_2d is None and risk_snapshot.beta_2d is not None:
            btc_beta_2d = np.asarray(risk_snapshot.beta_2d, dtype=np.float64)

    residual_var_2d: np.ndarray | None = None
    if policy_inputs is not None and policy_inputs.risk_residual_var_2d is not None:
        residual_var_2d = np.asarray(policy_inputs.risk_residual_var_2d, dtype=np.float64)
    if (
        residual_var_2d is None
        and risk_snapshot is not None
        and risk_snapshot.residual_var_2d is not None
    ):
        residual_var_2d = np.asarray(risk_snapshot.residual_var_2d, dtype=np.float64)

    ks_diag_2d: np.ndarray | None = None
    if (
        use_residual_var_for_kelly
        and residual_var_2d is not None
        and residual_var_2d.shape == mu_2d.shape
    ):
        ks_diag_2d = np.sqrt(np.maximum(residual_var_2d, 1e-12))
    elif composer_sigma_2d is not None:
        ks_diag_2d = np.asarray(composer_sigma_2d, dtype=np.float64)
        ks_diag_2d = np.maximum(ks_diag_2d, 1e-12)

    # If sigma_input is provided, we can use the high-speed Numba loop
    if sigma_input is not None:
        out = np.asarray(
            _precompute_loop_numba(
                n_bars,
                n_syms,
                rb,
                mu_2d,
                sigma_input,
                float(kappa),
                float(f_kelly_max),
                float(sigma_target_ann),
                float(bars_per_year),
                float(gross_cap),
                float(per_symbol_cap),
                float(current_dd),
                ks_diag_2d=ks_diag_2d,
                bl_shrinkage_var_mult=bl_shrinkage_var_mult,
                bl_shrinkage_omega_mult=bl_shrinkage_omega_mult,
            ),
            dtype=np.float64,
        )
    else:
        # Fallback to slower Python loop with rolling LW
        out = np.zeros((n_bars, n_syms), dtype=np.float64)
        lb = max(5, int(lookback))
        for i in range(1, n_bars):
            if (i % rb) != 0:
                continue
            start_i = max(0, i - lb)
            hist = c[start_i:i, :]
            if hist.shape[0] < 2:
                continue
            rr = np.diff(hist, axis=0) / np.maximum(hist[:-1, :], 1e-12)
            rr = np.nan_to_num(rr, nan=0.0, posinf=0.0, neginf=0.0)
            sigma = rolling_ledoit_wolf_cov(rr, min_obs=min_obs)

            ks_diag = None
            if ks_diag_2d is not None and ks_diag_2d.shape == mu_2d.shape:
                ks_diag = np.asarray(ks_diag_2d[i - 1, :], dtype=np.float64).ravel()
                ks_diag = np.maximum(ks_diag, 1e-12)

            out[i, :] = np.asarray(
                _solve_constrained_weights_numba(
                    mu_2d[i - 1],
                    sigma,
                    float(kappa),
                    float(f_kelly_max),
                    float(sigma_target_ann),
                    float(bars_per_year),
                    float(gross_cap),
                    float(per_symbol_cap),
                    float(current_dd),
                    kelly_sigma_diag=ks_diag,
                    bl_shrinkage_var_mult=bl_shrinkage_var_mult,
                    bl_shrinkage_omega_mult=bl_shrinkage_omega_mult,
                ),
                dtype=np.float64,
            )

    # project_all_caps 및 quantize_weights 통합 후처리
    caps = PortfolioCaps(
        gross=float(gross_cap),
        per_symbol=float(per_symbol_cap),
        target_ann_vol=float(sigma_target_ann),
    )

    lb = max(5, int(lookback))
    for i in range(1, n_bars):
        if (i % rb) != 0:
            continue
        w = out[i, :]
        if np.all(w == 0.0):
            continue

        if sigma_input is not None:
            sigma = sigma_input[i]
        else:
            start_i = max(0, i - lb)
            hist = c[start_i:i, :]
            if hist.shape[0] < 2:
                sigma = np.eye(n_syms) * 1e-4
            else:
                rr = np.diff(hist, axis=0) / np.maximum(hist[:-1, :], 1e-12)
                rr = np.nan_to_num(rr, nan=0.0, posinf=0.0, neginf=0.0)
                sigma = rolling_ledoit_wolf_cov(rr, min_obs=min_obs)

        sigma_port = float(np.sqrt(max(0.0, float(w @ sigma @ w))))

        if btc_beta_2d is not None and i < int(np.asarray(btc_beta_2d).shape[0]):
            btc_beta = np.asarray(btc_beta_2d[i], dtype=np.float64).ravel()
        else:
            btc_beta = np.zeros(n_syms, dtype=np.float64)

        w_proj = project_all_caps(
            w=w,
            btc_beta=btc_beta,
            sigma_port=sigma_port,
            bars_per_year=bars_per_year,
            caps=caps,
        )

        out[i, :] = np.asarray(w_proj, dtype=np.float64)

    return np.asarray(out, dtype=np.float64)


def portfolio_weight_params_from_optuna(
    params: dict[str, Any], opt_cfg: dict[str, Any]
) -> dict[str, Any]:
    pol = opt_cfg.get("FUTURES_PORTFOLIO_POLICY", {})
    tf = str(params.get("TIMEFRAME", "4h"))
    return {
        "lookback": cov_lookback_bars(tf, opt_cfg),
        "kappa": float(params.get("PORTFOLIO_KAPPA", opt_cfg.get("FUTURES_PORTFOLIO_KAPPA", 0.35))),
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
        "gross_cap": float(params.get("MAX_EXPOSURE", pol.get("gross_exposure_cap", 1.2))),
        "per_symbol_cap": float(
            params.get("MAX_EXPOSURE_PER_COIN", pol.get("per_symbol_cap", 0.25))
        ),
    }


# ---------------------------------------------------------------------------
# Phase 5: 5-cap 투영 및 minNotional 양자화
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioCaps:
    """5종 포트폴리오 cap 제약.

    Attributes:
        gross: Σ|w_i| ≤ gross.
        per_symbol: |w_i| ≤ per_symbol (각 심볼 개별 cap).
        net: |Σw_i| ≤ net.
        beta: |w @ btc_beta| ≤ beta.
        target_ann_vol: 연율화 변동성 target (vol scaling 기준).

    """

    gross: float = 3.0
    per_symbol: float = 0.10
    net: float = 0.30
    beta: float = 0.50
    target_ann_vol: float = 0.20


def project_all_caps(
    w: np.ndarray,
    btc_beta: np.ndarray,
    sigma_port: float,
    bars_per_year: float,
    caps: PortfolioCaps | None = None,
    *,
    support_mask: np.ndarray | None = None,
    allow_vol_upscale: bool = False,
) -> np.ndarray:
    """5-cap 투영: gross, per_symbol, net, beta, vol_target.

    각 cap을 순서대로 적용하는 iterative projection.

    Args:
        w: 초기 weight vector, shape [N].
        btc_beta: 심볼별 BTC beta, shape [N].
        sigma_port: 1-bar realized 포트폴리오 vol.
        bars_per_year: 연율화 factor.
        caps: PortfolioCaps 제약.
        support_mask: 활성 포지션 마스크 [N]. None이면 |w|>0 기준 자동 결정.
        allow_vol_upscale: True이면 Cap5에서 vol 양방향 정규화(확대 허용).
            False(기본)이면 기존 동작(축소만 허용).

    Returns:
        투영된 weight vector, shape [N].

    """
    if caps is None:
        caps = PortfolioCaps()
    out = np.asarray(w, dtype=np.float64).copy()
    n = int(out.size)
    if n == 0:
        return out
    support = (
        np.abs(out) > 1e-12
        if support_mask is None
        else np.asarray(support_mask, dtype=bool).ravel()
    )
    if support.size != n:
        support = np.abs(out) > 1e-12
    out = np.where(support, out, 0.0)

    beta_arr = np.asarray(btc_beta, dtype=np.float64).ravel()
    if beta_arr.size != n:
        beta_arr = np.zeros(n, dtype=np.float64)

    # Cap 1: per_symbol clipping
    out = np.clip(out, -caps.per_symbol, caps.per_symbol)

    # Cap 2: gross cap (L1 norm)
    for _ in range(20):
        gross = float(np.sum(np.abs(out)))
        if gross <= caps.gross + 1e-9:
            break
        out = out * (caps.gross / gross)
        # per_symbol 재적용
        out = np.clip(out, -caps.per_symbol, caps.per_symbol)

    # Cap 3: net cap (signed sum) — 기존 support와 부호를 보존한 채 축소
    net = float(np.sum(out))
    if net > caps.net + 1e-9:
        pos_mask = support & (out > 0.0)
        pos_total = float(np.sum(out[pos_mask]))
        excess = net - caps.net
        if pos_total > 1e-12:
            scale = max((pos_total - excess) / pos_total, 0.0)
            out[pos_mask] = out[pos_mask] * scale
        out = np.where(support, out, 0.0)
        out = np.clip(out, -caps.per_symbol, caps.per_symbol)
    elif net < -caps.net - 1e-9:
        neg_mask = support & (out < 0.0)
        neg_total = float(np.sum(np.abs(out[neg_mask])))
        excess = abs(net) - caps.net
        if neg_total > 1e-12:
            scale = max((neg_total - excess) / neg_total, 0.0)
            out[neg_mask] = out[neg_mask] * scale
        out = np.where(support, out, 0.0)
        out = np.clip(out, -caps.per_symbol, caps.per_symbol)

    # Cap 4: beta cap
    beta_exp = float(np.dot(out, beta_arr))
    if abs(beta_exp) > caps.beta + 1e-9:
        # beta 초과분을 beta 방향으로 축소
        scale = caps.beta / abs(beta_exp)
        out = out * scale
        out = np.where(support, out, 0.0)

    # Cap 5: vol target scaling
    ann_vol = float(sigma_port) * math.sqrt(max(float(bars_per_year), 1e-9))
    if ann_vol > 1e-12 and caps.target_ann_vol is not None and caps.target_ann_vol > 1e-12:
        vol_scale = caps.target_ann_vol / ann_vol
        out = out * vol_scale if allow_vol_upscale else out * min(vol_scale, 1.0)
        out = np.where(support, out, 0.0)

    # 최종 per_symbol 재확인
    out = np.clip(out, -caps.per_symbol, caps.per_symbol)
    out = np.where(support, out, 0.0)

    return out


def quantize_weights(
    w: np.ndarray,
    equity: float,
    prices: np.ndarray,
    step_sizes: np.ndarray,
    min_notional: float = 20.0,
) -> np.ndarray:
    """MinNotional 및 step_size 기반 weight 양자화.

    Args:
        w: 비중 벡터, shape [N].
        equity: 현재 계좌 자산 (USDT).
        prices: 심볼별 현재가, shape [N].
        step_sizes: 심볼별 step size (exchangeInfo), shape [N].
        min_notional: 최소 notional (기본 20.0 USDT).

    Returns:
        양자화된 비중 벡터, shape [N].
        qty = floor(w * equity / (price * step_size)) * step_size
        notional < min_notional → qty = 0
        return qty * price / equity

    """
    w_arr = np.asarray(w, dtype=np.float64)
    p_arr = np.asarray(prices, dtype=np.float64)
    s_arr = np.asarray(step_sizes, dtype=np.float64)
    eq = float(equity)

    # step_size 양자화
    raw_qty = w_arr * eq / np.where(p_arr * s_arr > 1e-15, p_arr * s_arr, 1e-15)
    qty = np.floor(np.abs(raw_qty)) * s_arr * np.sign(w_arr)

    # minNotional 필터
    notional = np.abs(qty) * p_arr
    qty = np.where(notional < min_notional, 0.0, qty)

    # 비중으로 재변환
    return qty * p_arr / eq


# ---------------------------------------------------------------------------
# Phase 6: 신규 아키텍처 — Diagonal Kelly (독립 경로)
# ---------------------------------------------------------------------------


def diagonal_kelly_weights(
    mu_bps: NDArray[np.float64],
    sigma: NDArray[np.float64],
    *,
    kelly_fraction: float,
    vol_target: float | None,
    caps: PortfolioCaps,
    prev_w: NDArray[np.float64],
    no_trade_band: float,
    btc_beta: NDArray[np.float64] | None = None,
    bars_per_year: float = 2190.0,
    support_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """신규 아키텍처용 Diagonal Kelly 사이징.

    기존 LW/BL/full-cov 경로와 독립적인 신규 함수. 기존 solve_constrained_weights와 무관.

    처리 순서:
    1. Diagonal Kelly: w_raw_i = kelly_fraction * mu_bps_i / max(sigma_i^2, VOL_FLOOR^2)
    2. vol_target 스케일링 (optional): 포트폴리오 sigma 기준 비례 축소
    3. No-trade band: |w_i - prev_w_i| < no_trade_band → w_i = prev_w_i (회전율 억제)
    4. project_all_caps: per_symbol / beta / gross / net / vol_target cap 적용

    Note: Friction filter removed. mu_bps is already net-of-cost per caller contract
    (signed_net_bps_per_bar in awf_sim.py subtracts execution cost before calling here).
    Re-applying would double-count execution cost and zero valid signals.

    Args:
        mu_bps: 심볼별 기대수익 [N], 단위: bps. SymbolSignal.raw_mu (Layer1 출력).
            per-bar NET edge (비용 이미 차감된 값).
        sigma: per-bar sigma [N]. >= VOL_FLOOR 보장 권장.
        kelly_fraction: 분수 Kelly 계수 (0,1].
        vol_target: 연율화 포트폴리오 변동성 목표 (None이면 미적용).
        caps: PortfolioCaps 제약 (5종 cap).
        prev_w: 이전 bar 비중 [N] (no-trade band 기준).
        no_trade_band: Δw < band이면 rebalance 생략 (절대값, 예: 0.01=1%).
        btc_beta: 심볼별 BTC beta [N] (project_all_caps에 전달, None이면 0 벡터).
        bars_per_year: 연율화 factor (4h=2190, 1h=8760).

    Returns:
        최종 비중 벡터 [N], float64.

    """
    # 순환 import 방지 — 함수 내부 로컬 import
    from src.domain.futures.strategy.cs_rank import VOL_FLOOR

    n = mu_bps.size
    mu = np.asarray(mu_bps, dtype=np.float64).ravel()
    sig = np.asarray(sigma, dtype=np.float64).ravel()
    p_w = np.asarray(prev_w, dtype=np.float64).ravel()
    beta = (
        np.zeros(n, dtype=np.float64)
        if btc_beta is None
        else np.asarray(btc_beta, dtype=np.float64).ravel()
    )
    support = (
        np.abs(mu) > 1e-12
        if support_mask is None
        else np.asarray(support_mask, dtype=bool).ravel()
    )
    if support.size != n:
        support = np.abs(mu) > 1e-12

    # Step 1 (friction filter) removed: mu_bps is already net-of-cost per caller contract.
    # See awf_sim.py caller — signed_net_bps_per_bar already subtracts hurdle*safety_mult.
    # Re-applying the filter double-counts execution cost and zeroes valid signals.

    # 2. Diagonal Kelly (mu를 per-bar return으로 변환: bps → fraction)
    mu_ret = mu * 1e-4  # bps → per-bar simple return
    sig_clipped = np.maximum(sig, float(VOL_FLOOR))
    var = sig_clipped**2
    w_raw: NDArray[np.float64] = kelly_fraction * mu_ret / var
    w_raw = np.where(support, w_raw, 0.0)

    # 3. vol_target 스케일링: caps.target_ann_vol override (project_all_caps에 위임)
    import dataclasses

    effective_caps = (
        dataclasses.replace(caps, target_ann_vol=vol_target)
        if vol_target is not None
        else caps
    )

    # 포트폴리오 실현 vol 추정: per_symbol 클립 후 계산
    # (pre-cap w_raw는 극단값이므로 vol_target scaling을 왜곡함)
    w_clipped_est = np.clip(w_raw, -caps.per_symbol, caps.per_symbol)
    sigma_port = float(np.sqrt(float(np.dot(w_clipped_est**2, var))))

    w_capped: NDArray[np.float64] = project_all_caps(
        w_raw, beta, sigma_port, bars_per_year, effective_caps,
        support_mask=support,
        allow_vol_upscale=True,   # L2 unit-vol 정규화 양방향
    )

    # 4. No-trade band: |Δw_i| < no_trade_band → 이전 비중 유지
    delta = np.abs(w_capped - p_w)
    w_final: NDArray[np.float64] = np.where(delta >= no_trade_band, w_capped, p_w)
    w_final = np.where(support, w_final, 0.0)

    return np.asarray(w_final, dtype=np.float64)
