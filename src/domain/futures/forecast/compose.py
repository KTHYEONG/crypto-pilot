"""Single SSOT for alpha - cost - hurdle composition (compose_mu)."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.domain.futures.forecast.contracts import AlphaForecast, CostForecast


def compose_mu(
    alpha: AlphaForecast,
    cost: CostForecast,
    params: dict[str, Any],
    *,
    holding_bars: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compose expected returns after cost and hurdle gate.

    Args:
        alpha: Typed alpha forecast.
        cost: Typed cost forecast.
        params: Trial/strategy params dict (must contain BETA_ALPHA, EV_HURDLE_BPS).
        holding_bars: Expected holding duration in bars. Only used when
            COST_GATE_AMORTIZE=True to amortize round-trip cost per bar.

    Returns:
        Tuple of (xs_long_2d, xs_short_2d, mu_long_2d, mu_short_2d).

    """
    from src.domain.futures.optimization.opt_config import (
        OPT_FUTURES_CONFIG,
        default_ev_hurdle_bps,
    )

    beta_a = float(
        params.get("BETA_ALPHA", OPT_FUTURES_CONFIG.get("FUTURES_DEFAULT_BETA_ALPHA", 1.0))
    )
    ev_h = float(params.get("EV_HURDLE_BPS", default_ev_hurdle_bps(OPT_FUTURES_CONFIG)))
    hurdle = ev_h / 10000.0

    cost_frac = np.asarray(cost.execution_cost_fraction_2d, dtype=np.float64)
    if params.get("COST_GATE_AMORTIZE", False) and holding_bars and int(holding_bars) > 1:
        cost_frac = cost_frac / float(int(holding_bars))

    mu_long = beta_a * np.asarray(alpha.alpha_long_2d, dtype=np.float64) - cost_frac
    mu_short = beta_a * np.asarray(alpha.alpha_short_2d, dtype=np.float64) - cost_frac
    admission_mode = str(params.get("POST_COST_ADMISSION_MODE", "ev_gate"))
    if admission_mode == "rank_then_ev_gate":
        rank_l = getattr(alpha, "rank_score_long_2d", None)
        rank_s = getattr(alpha, "rank_score_short_2d", None)
        if rank_l is not None and rank_s is not None:
            top_k = max(1, int(params.get("RANK_PORTFOLIO_TOP_K", 4)))
            min_spread = float(params.get("RANK_PORTFOLIO_MIN_SCORE_SPREAD_BPS", 0.0)) / 10000.0
            rank_l2d = np.asarray(rank_l, dtype=np.float64)
            rank_s2d = np.asarray(rank_s, dtype=np.float64)
            long_mask = np.zeros_like(mu_long, dtype=bool)
            short_mask = np.zeros_like(mu_short, dtype=bool)
            for t in range(mu_long.shape[0]):
                lrow = rank_l2d[t]
                srow = rank_s2d[t]
                lfinite = np.isfinite(lrow) & (lrow > 0.0)
                sfinite = np.isfinite(srow) & (srow < 0.0)
                if np.count_nonzero(lfinite) > 0:
                    idx = np.flatnonzero(lfinite)
                    l_sorted = idx[np.argsort(lrow[idx])[::-1]]
                    pick = l_sorted[:top_k]
                    if pick.size > 0 and (lrow[pick[0]] - lrow[pick[-1]] >= min_spread):
                        long_mask[t, pick] = True
                if np.count_nonzero(sfinite) > 0:
                    idx = np.flatnonzero(sfinite)
                    s_sorted = idx[np.argsort(srow[idx])]
                    pick = s_sorted[:top_k]
                    if pick.size > 0 and (srow[pick[-1]] - srow[pick[0]] >= min_spread):
                        short_mask[t, pick] = True
            mu_long = np.where(long_mask, mu_long, -np.inf)
            mu_short = np.where(short_mask, mu_short, -np.inf)
    xs_long = np.where(mu_long >= hurdle, mu_long, 0.0)
    xs_short = np.where(mu_short >= hurdle, mu_short, 0.0)
    return xs_long, xs_short, mu_long, mu_short
