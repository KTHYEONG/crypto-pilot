from __future__ import annotations

import optuna
from typing import Any, Dict


def suggest_params_spot(trial: optuna.Trial, space: Dict[str, Any], tf: str) -> Dict[str, Any]:
    """
    Spot optimization parameters: search space from `space`, execution guards applied after.
    """
    params: Dict[str, Any] = {"TIMEFRAME": tf}

    def _suggest(param_name: str) -> None:
        if param_name not in space:
            return
        spec = space[param_name]
        t = spec["type"]
        if t == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, spec["choices"])
        elif t == "int":
            params[param_name] = trial.suggest_int(
                param_name, spec["low"], spec["high"], step=spec.get("step", 1)
            )
        elif t == "float":
            params[param_name] = trial.suggest_float(
                param_name, spec["low"], spec["high"], step=spec.get("step")
            )

    _suggest("MACRO_EMA_PERIOD")
    _suggest("FAST_EMA_PERIOD")
    _suggest("ADX_PERIOD")
    _suggest("ADX_THRESHOLD")
    _suggest("KC_MULT")
    _suggest("MOMENTUM_PERIOD")
    _suggest("VOL_Z_THRESHOLD")
    _suggest("ATR_PERIOD")
    _suggest("LONG_ATR_MULT")
    _suggest("LONG_TRAIL_MULT")
    _suggest("RISK_PER_TRADE")
    _suggest("MAX_POSITION_PCT")
    _suggest("TIME_STOP_BARS")

    params["LONG_TP_MULT"] = 0
    params["LEVERAGE"] = 1
    params["USE_COMPOUNDING"] = True

    long_atr = float(params.get("LONG_ATR_MULT", 2.0))
    long_trail = float(params.get("LONG_TRAIL_MULT", long_atr))
    if long_trail < long_atr:
        params["LONG_TRAIL_MULT"] = long_atr

    macro = int(params.get("MACRO_EMA_PERIOD", 200))
    params["BTC_REGIME_SMA_PERIOD"] = macro

    return params
