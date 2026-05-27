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
    xs_long = np.where(mu_long >= hurdle, mu_long, 0.0)
    xs_short = np.where(mu_short >= hurdle, mu_short, 0.0)
    return xs_long, xs_short, mu_long, mu_short
