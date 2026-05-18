"""Gate metric evaluators for AlphaFactoryV1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class GateMetrics:
    """Gate metric bundle for AlphaFactoryV1 diagnostics and filtering."""

    oos_net_ic: float
    fold_positive_ratio: float
    cost_adjusted_expectancy: float
    crisis_long_suppression: float
    coverage: float


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if int(mask.sum()) < 3:
        return 0.0
    xr = _rankdata(x[mask])
    yr = _rankdata(y[mask])
    x_std = float(np.std(xr))
    y_std = float(np.std(yr))
    if x_std <= _EPS or y_std <= _EPS:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def calculate_oos_net_ic(alpha_net: np.ndarray, forward_returns: np.ndarray) -> float:
    """OOS net IC using Spearman rank correlation."""
    return _spearman_ic(
        np.asarray(alpha_net, dtype=np.float64),
        np.asarray(forward_returns, dtype=np.float64),
    )


def calculate_fold_positive_ratio(fold_ics: np.ndarray) -> float:
    """Fraction of folds with positive IC."""
    arr = np.asarray(fold_ics, dtype=np.float64)
    valid = ~np.isnan(arr)
    if int(valid.sum()) == 0:
        return 0.0
    return float(np.mean(arr[valid] > 0.0))


def calculate_cost_adjusted_expectancy(
    alpha_net: np.ndarray,
    forward_returns: np.ndarray,
    turnover_hint: np.ndarray,
    cost_per_turnover: float,
) -> float:
    """Compute expected return after transaction cost proxy adjustment."""
    an = np.asarray(alpha_net, dtype=np.float64)
    fr = np.asarray(forward_returns, dtype=np.float64)
    th = np.asarray(turnover_hint, dtype=np.float64)
    n = min(len(an), len(fr), len(th))
    if n == 0:
        return 0.0
    gross = an[:n] * fr[:n]
    net = gross - float(cost_per_turnover) * np.clip(th[:n], 0.0, None)
    mask = ~np.isnan(net)
    if int(mask.sum()) == 0:
        return 0.0
    return float(np.mean(net[mask]))


def calculate_crisis_long_suppression(
    alpha_long: np.ndarray,
    crisis_mask: np.ndarray,
) -> float:
    """How much long exposure is suppressed in crisis bars (higher is better)."""
    al = np.clip(np.asarray(alpha_long, dtype=np.float64), 0.0, 1.0)
    cm = np.asarray(crisis_mask, dtype=bool)
    n = min(len(al), len(cm))
    if n == 0:
        return 0.0
    base = float(np.mean(al[:n]))
    crisis_sel = al[:n][cm[:n]]
    if crisis_sel.size == 0 or base <= _EPS:
        return 0.0
    crisis_mean = float(np.mean(crisis_sel))
    return float(np.clip((base - crisis_mean) / (base + _EPS), 0.0, 1.0))


def calculate_coverage(alpha_net: np.ndarray) -> float:
    """Non-NaN coverage ratio for alpha_net."""
    an = np.asarray(alpha_net, dtype=np.float64)
    if an.size == 0:
        return 0.0
    return float(np.mean(~np.isnan(an)))


def build_gate_metrics(
    *,
    alpha_long: np.ndarray,
    alpha_net: np.ndarray,
    forward_returns: np.ndarray,
    fold_ics: np.ndarray,
    turnover_hint: np.ndarray,
    crisis_mask: np.ndarray,
    cost_per_turnover: float,
) -> GateMetrics:
    """Compute the complete gate metric set required by AlphaFactoryV1."""
    return GateMetrics(
        oos_net_ic=calculate_oos_net_ic(alpha_net=alpha_net, forward_returns=forward_returns),
        fold_positive_ratio=calculate_fold_positive_ratio(fold_ics=fold_ics),
        cost_adjusted_expectancy=calculate_cost_adjusted_expectancy(
            alpha_net=alpha_net,
            forward_returns=forward_returns,
            turnover_hint=turnover_hint,
            cost_per_turnover=cost_per_turnover,
        ),
        crisis_long_suppression=calculate_crisis_long_suppression(
            alpha_long=alpha_long,
            crisis_mask=crisis_mask,
        ),
        coverage=calculate_coverage(alpha_net=alpha_net),
    )
