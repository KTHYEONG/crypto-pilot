"""Linear alpha + regime signal composition for portfolio weights (no CS rank / HMM multiply)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG


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
    """Get ~8 calendar days of bars for simple per-bar return std (sigma_t,i)."""
    cfg = opt_cfg or OPT_FUTURES_CONFIG
    by_tf = cfg.get("FUTURES_COMPOSER_SIGMA_LOOKBACK_BY_TF")
    key = str(tf).strip().lower()
    if isinstance(by_tf, dict) and key in by_tf:
        return max(3, int(by_tf[key]))
    days = float(cfg.get("FUTURES_COMPOSER_SIGMA_CALENDAR_DAYS", 8.0))
    hpb = hours_per_bar_tf(tf)
    return max(3, int(days * 24.0 / max(hpb, 1e-9)))


def rolling_per_bar_return_std(close_1d: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling std of simple returns r_t = (c_t - c_{t-1}) / |c_{t-1}| (causal)."""
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
    return np.maximum(v, 1e-12)  # type: ignore[no-any-return]


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

    beta_a = float(params.get("BETA_ALPHA", cfg.get("FUTURES_DEFAULT_BETA_ALPHA", 1.0)))
    ev_h = float(params.get("EV_HURDLE_BPS", cfg.get("FUTURES_DEFAULT_EV_HURDLE_BPS", 10.0)))

    from src.core.settings import round_trip_cost_bps

    # Taker 진입 + Taker 청산 + 2x 슬리피지 = 14bps (execution_sim 과 동일 기준)
    friction = round_trip_cost_bps() / 10000.0

    mu_l = beta_a * np.asarray(alpha_long, dtype=np.float64) - friction
    mu_s = beta_a * np.asarray(alpha_short, dtype=np.float64) - friction

    hurdle_frac = np.asarray(ev_h, dtype=np.float64) / 10000.0
    xs_l = np.where(mu_l >= hurdle_frac, mu_l, 0.0)
    xs_s = np.where(mu_s >= hurdle_frac, mu_s, 0.0)
    return xs_l, xs_s
