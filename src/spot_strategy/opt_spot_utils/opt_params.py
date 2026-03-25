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

    for param_name in sorted(space.keys()):
        _suggest(param_name)

    params["LEVERAGE"] = 1
    params["USE_COMPOUNDING"] = True

    long_atr = float(params.get("LONG_ATR_MULT", 2.0))
    long_trail = float(params.get("LONG_TRAIL_MULT", long_atr))
    if long_trail < long_atr:
        params["LONG_TRAIL_MULT"] = long_atr

    long_tp = float(params.get("LONG_TP_MULT", 5.0))
    if long_tp < long_atr * 1.2:
        params["LONG_TP_MULT"] = long_atr * 1.2

    atr_s = int(params.get("ATR_RATIO_PERIOD", 14))
    atr_l = int(params.get("ATR_RATIO_LONG_PERIOD", 42))
    if atr_l <= atr_s:
        params["ATR_RATIO_LONG_PERIOD"] = atr_s * 3

    hmm_w = int(params.get("HMM_TRAIN_WINDOW", 360))
    if hmm_w < 120:
        params["HMM_TRAIN_WINDOW"] = 120

    g_w = int(params.get("GARCH_WINDOW", 360))
    if g_w < 120:
        params["GARCH_WINDOW"] = 120

    g_rf = int(params.get("GARCH_RETRAIN_FREQ", 24))
    h_rf = int(params.get("HMM_RETRAIN_FREQ", 24))
    if g_rf < 1:
        params["GARCH_RETRAIN_FREQ"] = 1
    if h_rf < 1:
        params["HMM_RETRAIN_FREQ"] = 1

    return params
