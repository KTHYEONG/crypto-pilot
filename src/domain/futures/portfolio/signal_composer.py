"""Linear alpha + regime signal composition for portfolio weights (no CS rank / HMM multiply)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.opt_config import OPT_FUTURES_CONFIG


def hours_per_bar_tf(tf: str) -> float:
    t = str(tf).strip().lower()
    if t.endswith("h"):
        return float(t.replace("h", "") or 4)
    if t.endswith("d"):
        return float(t.replace("d", "") or 1) * 24.0
    if t.endswith("m"):
        return float(t.replace("m", "") or 1) / 60.0
    return 4.0


def composer_sigma_lookback_bars(tf: str, opt_cfg: dict[str, Any] | None = None) -> int:
    """~8 calendar days of bars for simple per-bar return std (σ_t,i)."""
    cfg = opt_cfg or OPT_FUTURES_CONFIG
    by_tf = cfg.get("FUTURES_COMPOSER_SIGMA_LOOKBACK_BY_TF")
    key = str(tf).strip().lower()
    if isinstance(by_tf, dict) and key in by_tf:
        return max(3, int(by_tf[key]))
    days = float(cfg.get("FUTURES_COMPOSER_SIGMA_CALENDAR_DAYS", 8.0))
    hpb = hours_per_bar_tf(tf)
    return max(3, int(days * 24.0 / max(hpb, 1e-9)))


def rolling_per_bar_return_std(close_1d: np.ndarray, window: int) -> np.ndarray:
    """Rolling std of simple returns r_t = (c_t - c_{t-1}) / |c_{t-1}| (causal)."""
    c = np.asarray(close_1d, dtype=np.float64).ravel()
    n = c.size
    out = np.zeros(n, dtype=np.float64)
    if n < 2:
        return out
    r = np.zeros(n, dtype=np.float64)
    r[1:] = (c[1:] - c[:-1]) / np.maximum(np.abs(c[:-1]), 1e-12)
    rw = max(2, int(window))
    s = pd.Series(r).rolling(rw, min_periods=2).std(ddof=1)
    v = s.to_numpy(dtype=np.float64)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(v, 1e-12)


def apply_linear_signal_composer_scores(
    df: Any,
    alpha_long: np.ndarray,
    alpha_short: np.ndarray,
    params: dict[str, Any],
    *,
    opt_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose long/short expected edge with optional posterior-aware regime gates."""
    cfg = opt_config or OPT_FUTURES_CONFIG

    if not isinstance(df, pd.DataFrame):
        raise TypeError("apply_linear_signal_composer_scores expects a DataFrame")

    n = len(df)
    beta_a = float(params.get("BETA_ALPHA", cfg.get("FUTURES_DEFAULT_BETA_ALPHA", 1.0)))
    b_bull = float(params.get("BETA_REGIME_BULL", cfg.get("FUTURES_DEFAULT_BETA_REGIME_BULL", 1.0)))
    b_bear = float(params.get("BETA_REGIME_BEAR", cfg.get("FUTURES_DEFAULT_BETA_REGIME_BEAR", 1.0)))
    b_crisis = float(params.get("BETA_REGIME_CRISIS", cfg.get("FUTURES_DEFAULT_BETA_REGIME_CRISIS", 1.0)))
    b_chop = float(params.get("BETA_REGIME_CHOP", cfg.get("FUTURES_DEFAULT_BETA_REGIME_CHOP", 0.25)))
    ev_h = float(params.get("EV_HURDLE_BPS", cfg.get("FUTURES_DEFAULT_EV_HURDLE_BPS", 5.0)))

    from config.settings import SLIPPAGE_RATE, TAKER_FEE_RATE

    slip = float(SLIPPAGE_RATE) * float(params.get("SLIPPAGE_BPS_BUFFER_MULT", 1.0))
    fee = float(TAKER_FEE_RATE)
    fund_bar = float(cfg.get("FUTURES_COMPOSER_FUNDING_BAR_FRAC", 1e-5))
    buf_mult = float(cfg.get("FUTURES_FRICTION_BUFFER_MULT", 1.5))
    friction = buf_mult * (fee + slip + fund_bar)

    pbull = np.zeros(n, dtype=np.float64)
    if "hmm_prob_bull_calm" in df.columns and "hmm_prob_bull_vol_up" in df.columns:
        pbull = (
            df["hmm_prob_bull_calm"].astype(np.float64).fillna(0.0).to_numpy()
            + df["hmm_prob_bull_vol_up"].astype(np.float64).fillna(0.0).to_numpy()
        )
    p_bear = (
        df["hmm_prob_bear_trend"].astype(np.float64).fillna(0.0).to_numpy()
        if "hmm_prob_bear_trend" in df.columns
        else np.zeros(n, dtype=np.float64)
    )
    p_chop = (
        df["hmm_prob_chop"].astype(np.float64).fillna(0.0).to_numpy()
        if "hmm_prob_chop" in df.columns
        else np.zeros(n, dtype=np.float64)
    )
    p_crisis = (
        df["hmm_prob_crisis"].astype(np.float64).fillna(0.0).to_numpy()
        if "hmm_prob_crisis" in df.columns
        else np.zeros(n, dtype=np.float64)
    )
    # Optional recovery column (rare); omitted from tmp table when absent.
    precov = (
        df["hmm_prob_recovery"].astype(np.float64).fillna(0.0).to_numpy()
        if "hmm_prob_recovery" in df.columns
        else np.zeros(n, dtype=np.float64)
    )
    b_rec = float(params.get("BETA_REGIME_RECOVERY", cfg.get("FUTURES_DEFAULT_BETA_REGIME_RECOVERY", 0.0)))

    regime = (
        b_bull * pbull
        + b_bear * p_bear
        + b_chop * p_chop
        + b_crisis * p_crisis
        + b_rec * precov
    )
    mu_l = beta_a * np.asarray(alpha_long, dtype=np.float64) + regime - friction
    mu_s = beta_a * np.asarray(alpha_short, dtype=np.float64) + regime - friction

    regime_policy_enabled = bool(
        params.get(
            "REGIME_POLICY_ENABLED",
            cfg.get("FUTURES_REGIME_POLICY_ENABLED", False),
        )
    )
    if regime_policy_enabled:
        probs = np.column_stack([pbull, p_bear, p_chop, p_crisis, precov])
        psum = np.maximum(probs.sum(axis=1), 1e-12)
        probs = probs / psum[:, None]
        logk = np.log(float(probs.shape[1]))
        ent = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1) / max(logk, 1e-12)
        ent = np.clip(ent, 0.0, 1.0)
        conf = 1.0 - ent

        ent_mult = float(
            params.get(
                "REGIME_CONFIDENCE_ENTROPY_MULT",
                cfg.get("FUTURES_REGIME_CONFIDENCE_ENTROPY_MULT", 0.50),
            )
        )
        min_mult = float(params.get("REGIME_MULT_MIN", cfg.get("FUTURES_REGIME_MULT_MIN", 0.10)))
        max_mult = float(params.get("REGIME_MULT_MAX", cfg.get("FUTURES_REGIME_MULT_MAX", 1.50)))
        conf_scale = np.clip(1.0 - ent_mult * (1.0 - conf), min_mult, max_mult)

        l_bull_w = float(params.get("REGIME_LONG_BULL_W", cfg.get("FUTURES_REGIME_LONG_BULL_W", 0.35)))
        l_bear_p = float(params.get("REGIME_LONG_BEAR_PENALTY", cfg.get("FUTURES_REGIME_LONG_BEAR_PENALTY", 0.35)))
        l_chop_p = float(params.get("REGIME_LONG_CHOP_PENALTY", cfg.get("FUTURES_REGIME_LONG_CHOP_PENALTY", 0.55)))
        l_crisis_p = float(params.get("REGIME_LONG_CRISIS_PENALTY", cfg.get("FUTURES_REGIME_LONG_CRISIS_PENALTY", 0.90)))
        s_bear_w = float(params.get("REGIME_SHORT_BEAR_W", cfg.get("FUTURES_REGIME_SHORT_BEAR_W", 0.45)))
        s_bull_p = float(params.get("REGIME_SHORT_BULL_PENALTY", cfg.get("FUTURES_REGIME_SHORT_BULL_PENALTY", 0.25)))
        s_chop_p = float(params.get("REGIME_SHORT_CHOP_PENALTY", cfg.get("FUTURES_REGIME_SHORT_CHOP_PENALTY", 0.45)))
        s_crisis_w = float(params.get("REGIME_SHORT_CRISIS_W", cfg.get("FUTURES_REGIME_SHORT_CRISIS_W", 0.15)))

        long_mult = 1.0 + (l_bull_w * pbull) - (l_bear_p * p_bear) - (l_chop_p * p_chop) - (l_crisis_p * p_crisis)
        short_mult = 1.0 + (s_bear_w * p_bear) - (s_bull_p * pbull) - (s_chop_p * p_chop) + (s_crisis_w * p_crisis)
        long_mult = np.clip(long_mult * conf_scale, min_mult, max_mult)
        short_mult = np.clip(short_mult * conf_scale, min_mult, max_mult)

        mu_l = mu_l * long_mult
        mu_s = mu_s * short_mult

        ev_chop = float(params.get("REGIME_EV_CHOP_ADD_BPS", cfg.get("FUTURES_REGIME_EV_CHOP_ADD_BPS", 8.0)))
        ev_crisis = float(params.get("REGIME_EV_CRISIS_ADD_BPS", cfg.get("FUTURES_REGIME_EV_CRISIS_ADD_BPS", 12.0)))
        ev_entropy = float(params.get("REGIME_EV_ENTROPY_ADD_BPS", cfg.get("FUTURES_REGIME_EV_ENTROPY_ADD_BPS", 6.0)))
        ev_h = ev_h + (ev_chop * p_chop) + (ev_crisis * p_crisis) + (ev_entropy * ent)

    hurdle_frac = np.asarray(ev_h, dtype=np.float64) / 10000.0
    xs_l = np.where(mu_l >= hurdle_frac, mu_l, 0.0)
    xs_s = np.where(mu_s >= hurdle_frac, mu_s, 0.0)
    return xs_l, xs_s
