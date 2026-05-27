"""Portfolio weights: Ledoit-Wolf covariance, fractional Kelly -> vol target -> constrained net.

Expected return ``mu`` is in **simple return per bar** (same units as bar-to-bar returns).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
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
        sig = np.sqrt(np.clip(np.diag(np.asarray(sigma, dtype=np.float64)), 1e-12, None))
    w_raw = kappa * _kelly_raw(mu_v, sig, f_kelly_max=f_kelly_max)

    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.shape != (n, n):
        sigma = np.diag(np.diag(sigma))

    port_var_bar = float(w_raw @ sigma @ w_raw)
    sigma_pb = math.sqrt(max(port_var_bar, 1e-18))
    ann = _vol_ann_from_per_bar_sigma(sigma_pb, bars_per_year)
    if ann > 1e-12:
        w_pre = w_raw * (float(sigma_target_ann) / ann)
    else:
        w_pre = w_raw * 0.0

    w_c = _project_l1_linf(w_pre, gross_cap=gross_cap, per_symbol_cap=per_symbol_cap)

    _ = current_dd
    return w_c


def precompute_rolling_covariances(
    close_2d: np.ndarray, lookback: int, min_obs: int = 20
) -> np.ndarray:
    """Precompute rolling Ledoit-Wolf covariance matrices for all bars.

    Called once during optimization precompute phase to eliminate 1M+ redundant
    calls during trials.
    """
    c = np.asarray(close_2d, dtype=np.float64)
    n_bars, n_syms = c.shape
    out = np.zeros((n_bars, n_syms, n_syms), dtype=np.float64)
    lb = max(5, int(lookback))

    # LW fit is relatively heavy, but we only do this once per bar for the entire optimization.
    for i in range(1, n_bars):
        start_i = max(0, i - lb)
        hist = c[start_i:i, :]
        if hist.shape[0] < 3:
            # Identity matrix fallback for insufficient history
            for j in range(n_syms):
                out[i, j, j] = 1e-6
            continue

        rr = np.diff(hist, axis=0) / np.maximum(hist[:-1, :], 1e-12)
        rr = np.nan_to_num(rr, nan=0.0, posinf=0.0, neginf=0.0)
        out[i] = rolling_ledoit_wolf_cov(rr, min_obs=min_obs)

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
) -> np.ndarray:
    n = mu.size
    mu_mod = mu.copy()
    dyn_gross_cap = gross_cap

    if kelly_sigma_diag is not None:
        sig = kelly_sigma_diag
    else:
        sig = np.zeros(n, dtype=np.float64)
        for i in range(n):
            sig[i] = math.sqrt(max(sigma[i, i], 1e-12))

    # _kelly_raw logic  # [N] → scalar Kelly fraction per symbol
    w_raw = np.zeros(n, dtype=np.float64)
    f_max = abs(f_kelly_max)
    for i in range(n):
        var = max(sig[i] ** 2, 1e-12)
        f = mu_mod[i] / var
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

    if ann > 1e-12:
        w_pre = w_raw * (sigma_target_ann / ann)
    else:
        w_pre = w_raw * 0.0

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
) -> np.ndarray:
    """5-cap 투영: gross, per_symbol, net, beta, vol_target.

    각 cap을 순서대로 적용하는 iterative projection.

    Args:
        w: 초기 weight vector, shape [N].
        btc_beta: 심볼별 BTC beta, shape [N].
        sigma_port: 1-bar realized 포트폴리오 vol.
        bars_per_year: 연율화 factor.
        caps: PortfolioCaps 제약.

    Returns:
        투영된 weight vector, shape [N].

    """
    if caps is None:
        caps = PortfolioCaps()
    out = np.asarray(w, dtype=np.float64).copy()
    n = int(out.size)
    if n == 0:
        return out

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

    # Cap 3: net cap (signed sum)
    net = float(np.sum(out))
    if abs(net) > caps.net + 1e-9:
        # net 초과분을 비례 축소
        excess = abs(net) - caps.net
        if abs(net) > 1e-12:
            correction = np.sign(net) * excess / n
            out = out - correction
        # per_symbol 재적용
        out = np.clip(out, -caps.per_symbol, caps.per_symbol)

    # Cap 4: beta cap
    beta_exp = float(np.dot(out, beta_arr))
    if abs(beta_exp) > caps.beta + 1e-9:
        # beta 초과분을 beta 방향으로 축소
        scale = caps.beta / abs(beta_exp)
        out = out * scale

    # Cap 5: vol target scaling
    ann_vol = float(sigma_port) * math.sqrt(max(float(bars_per_year), 1e-9))
    if ann_vol > 1e-12:
        vol_scale = caps.target_ann_vol / ann_vol
        out = out * min(vol_scale, 1.0)  # 축소만 허용 (확대 금지)

    # 최종 per_symbol 재확인
    out = np.clip(out, -caps.per_symbol, caps.per_symbol)

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
