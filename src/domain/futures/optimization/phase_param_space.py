from __future__ import annotations

from typing import Any

import optuna

V43_SIGNAL_PARAM_KEYS: tuple[str, ...] = (
    "BETA_REGIME_BEAR",
    "BETA_REGIME_CHOP",
    "K_LONG",
    "K_SHORT",
    "REBALANCE_BARS",
    "EV_HURDLE_BPS",
)

V43_RISK_PARAM_KEYS: tuple[str, ...] = (
    "PORTFOLIO_KAPPA",
    "TARGET_ANN_VOL",
    "MAX_EXPOSURE",
    "MAX_EXPOSURE_PER_COIN",
)

V43_CORE_PARAM_KEYS: tuple[str, ...] = V43_SIGNAL_PARAM_KEYS + V43_RISK_PARAM_KEYS

V43_FIXED_DEFAULTS: dict[str, Any] = {
    "BETA_REGIME_BULL": 1.0,
    "BETA_REGIME_CRISIS": 0.0,
    "SLIPPAGE_BPS_BUFFER_MULT": 1.5,
    "CRISIS_OVERRIDE_THRESHOLD": 0.70,
    "CRISIS_GAMMA": 0.20,
    "CRISIS_EXIT_BARS": 3,
    # Engine compatibility default (removed from optimization).
    "MIN_SCORE_PERCENTILE": 0.55,
}

_SIGNAL_DEFAULT_RANGES: dict[str, tuple[Any, Any, bool]] = {
    "BETA_REGIME_BEAR": (0.0, 1.5, False),
    "BETA_REGIME_CHOP": (0.0, 1.0, False),
    "K_LONG": (1, 8, False),
    "K_SHORT": (0, 5, False),
    "REBALANCE_BARS": (1, 24, False),
    "EV_HURDLE_BPS": (5.0, 100.0, True),
}

_RISK_DEFAULT_RANGES: dict[str, tuple[Any, Any, bool]] = {
    "PORTFOLIO_KAPPA": (0.05, 0.50, True),
    "TARGET_ANN_VOL": (0.05, 0.40, True),
    "MAX_EXPOSURE": (0.50, 3.00, False),
    "MAX_EXPOSURE_PER_COIN": (0.05, 0.40, True),
}


def _merge_ranges(
    default_ranges: dict[str, tuple[Any, Any, bool]], ranges: dict[str, tuple[Any, Any]] | None
) -> dict[str, tuple[Any, Any, bool]]:
    merged = dict(default_ranges)
    if not ranges:
        return merged
    for key, value in ranges.items():
        if key in merged and isinstance(value, tuple) and len(value) == 2:
            low, high = value
            _, _, is_log = merged[key]
            merged[key] = (low, high, is_log)
    return merged


def _suggest_group(
    trial: optuna.trial.Trial,
    default_ranges: dict[str, tuple[Any, Any, bool]],
    ranges: dict[str, tuple[Any, Any]] | None,
    fixed: dict[str, Any] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    merged_ranges = _merge_ranges(default_ranges, ranges)
    fixed = fixed or {}
    for key, (low, high, is_log) in merged_ranges.items():
        if key in fixed:
            out[key] = fixed[key]
            continue
        if isinstance(low, int) and isinstance(high, int):
            out[key] = int(trial.suggest_int(key, int(low), int(high)))
        else:
            out[key] = float(trial.suggest_float(key, float(low), float(high), log=is_log))
    return out


def suggest_signal_params(
    trial: optuna.trial.Trial,
    ranges: dict[str, tuple[Any, Any]] | None = None,
    fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _suggest_group(trial, _SIGNAL_DEFAULT_RANGES, ranges, fixed)


def suggest_risk_params(
    trial: optuna.trial.Trial,
    ranges: dict[str, tuple[Any, Any]] | None = None,
    fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _suggest_group(trial, _RISK_DEFAULT_RANGES, ranges, fixed)


def suggest_joint_params(
    trial: optuna.trial.Trial,
    ranges: dict[str, tuple[Any, Any]] | None = None,
    fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    signal = suggest_signal_params(trial, ranges=ranges, fixed=fixed)
    risk = suggest_risk_params(trial, ranges=ranges, fixed=fixed)
    return {**signal, **risk}
