"""Rolling t-GARCH(1,1) Kelly scaling with fat-tail nu clamp and fallbacks."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

try:
    from arch import arch_model
except ImportError:
    arch_model = None  # type: ignore[misc, assignment]

_DEGENERATE_RET_STD: float = 1e-8
_GARCH_MAXITER: int = 100


def _t_kelly_fraction(mu: float, sigma2: float, nu: float) -> float:
    """t-distribution adjusted Kelly: f_t = mu * (nu - 2) / (sigma^2 * (nu - 1)), nu > 2."""
    nu_c = float(np.clip(nu, 3.0, 30.0))
    if sigma2 <= 1e-18 or nu_c <= 2.01:
        return 0.25
    num = mu * (nu_c - 2.0)
    den = sigma2 * (nu_c - 1.0)
    if abs(den) < 1e-18:
        return 0.25
    f = num / den
    return float(np.clip(f, -2.0, 2.0))


def _kelly_to_multiplier(k: float) -> float:
    return float(np.clip(0.2 + 0.4 * min(1.0, 1.0 / (1.0 + abs(k))), 0.15, 1.0))


def compute_garch_kelly_series(
    close: pd.Series,
    *,
    window: int,
    retrain_freq: int = 24,
    nu_fallback: float = 5.0,
) -> pd.Series:
    """
    Per-bar fractional-Kelly multiplier in (0, 1] derived from rolling t-GARCH.
    Retrain every `retrain_freq` bars; forward-fill between retrains.
    """
    n = len(close)
    out = np.full(n, 0.5, dtype=np.float64)
    log_ret = np.log(close.astype(np.float64).clip(lower=1e-12)).diff().fillna(0.0).to_numpy()

    w = max(120, int(window))
    rf = max(1, int(retrain_freq))

    last_nu = nu_fallback

    if arch_model is None:
        _logger.warning("arch not installed; GARCH Kelly uses rolling-vol fallback.")
        for t in range(n):
            seg = log_ret[max(0, t - 30) : t]
            mu = float(np.mean(seg)) if seg.size > 0 else 0.0
            rv = float(np.std(seg) + 1e-12) if seg.size > 1 else 0.02
            k = _t_kelly_fraction(mu, rv * rv, last_nu)
            out[t] = _kelly_to_multiplier(k)
        return pd.Series(out, index=close.index)

    last_k_mult = 0.5
    last_starting: Optional[np.ndarray] = None

    for t in range(w, n):
        if (t - w) % rf != 0:
            out[t] = last_k_mult
            continue

        y = log_ret[t - w : t] * 100.0
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.std(y)) < _DEGENERATE_RET_STD:
            seg = log_ret[t - 30 : t]
            mu = float(np.mean(seg)) if seg.size > 0 else 0.0
            rv = float(np.std(seg) + 1e-12) if seg.size > 1 else 0.02
            k = _t_kelly_fraction(mu, rv * rv, nu_fallback)
            last_k_mult = _kelly_to_multiplier(k)
            out[t] = last_k_mult
            continue

        try:
            am = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
            n_param = len(am._all_parameter_names())
            fit_kw: dict[str, object] = {
                "disp": "off",
                "show_warning": False,
                "options": {"maxiter": int(_GARCH_MAXITER)},
            }
            if last_starting is not None and last_starting.shape[0] == n_param:
                fit_kw["starting_values"] = last_starting
            res = am.fit(**fit_kw)
            last_starting = np.asarray(res.params.values, dtype=np.float64)
            cv = res.conditional_volatility
            sig = float(cv.iloc[-1]) / 100.0
            var = max(sig * sig, 1e-12)
            nu = float(res.params.get("nu", nu_fallback))
            nu = float(np.clip(nu, 3.0, 30.0))
            mu = float(res.params.get("mu", float(np.mean(y)))) / 100.0
            k = _t_kelly_fraction(mu, var, nu)
            last_nu = nu
            last_k_mult = _kelly_to_multiplier(k)
            out[t] = last_k_mult
        except Exception as exc:
            _logger.debug("GARCH fit failed at t=%s: %s", t, exc)
            seg = log_ret[t - 30 : t]
            mu = float(np.mean(seg)) if seg.size > 0 else 0.0
            rv = float(np.std(seg) + 1e-12) if seg.size > 1 else 0.02
            k = _t_kelly_fraction(mu, rv * rv, nu_fallback)
            last_k_mult = _kelly_to_multiplier(k)
            out[t] = last_k_mult

    return pd.Series(out, index=close.index)
