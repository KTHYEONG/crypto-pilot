"""Parameter perturbation and stability checks for spot optimization."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

_SPOT_FLOAT_PERTURB_KEYS: frozenset[str] = frozenset(
    {
        "RISK_PER_TRADE",
        "MAX_POSITION_PCT",
        "VOL_Z_THRESHOLD",
    }
)
_SPOT_INT_PERTURB_KEYS: frozenset[str] = frozenset(
    {
        "MACRO_EMA_PERIOD",
        "FAST_EMA_PERIOD",
        "MOMENTUM_PERIOD",
        "ATR_PERIOD",
    }
)


def perturb_params_spot(
    params: Dict[str, Any],
    *,
    scale: float,
    direction: int,
) -> Dict[str, Any]:
    """
    direction: +1 or -1. scale: fractional change (e.g. 0.15 for ±15% on floats).
    """
    out = copy.deepcopy(params)
    sign = 1.0 if direction >= 0 else -1.0
    for k in _SPOT_FLOAT_PERTURB_KEYS:
        if k not in out:
            continue
        v = float(out[k])
        out[k] = max(1e-9, v * (1.0 + sign * scale))
    for k in _SPOT_INT_PERTURB_KEYS:
        if k not in out:
            continue
        v = int(out[k])
        delta = max(1, int(round(abs(v) * scale)))
        out[k] = max(1, v + int(sign * delta))
    if "MACRO_EMA_PERIOD" in out:
        out["BTC_REGIME_SMA_PERIOD"] = int(out["MACRO_EMA_PERIOD"])
    return out


def stability_ratio_ok(
    base_score: float,
    perturbed_scores: List[float],
    *,
    max_rel_drop: float = 0.25,
) -> Tuple[bool, float]:
    """Pass if no perturbed run drops more than max_rel_drop relative to base_score."""
    if not perturbed_scores:
        return True, 1.0
    worst = min(perturbed_scores)
    if abs(base_score) < 1e-9:
        ok = worst >= -1e-6
        return ok, 0.0 if not ok else 1.0
    ratio = worst / base_score if base_score > 0 else worst - base_score
    rel = (base_score - worst) / abs(base_score) if base_score != 0 else 0.0
    ok = rel <= max_rel_drop
    return ok, float(ratio)
