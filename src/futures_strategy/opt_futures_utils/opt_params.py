from __future__ import annotations
import optuna
from typing import Dict, Any

def suggest_params_futures(trial: optuna.Trial, space: Dict[str, Any], tf: str) -> Dict[str, Any]:
    """
    Suggests parameters for optimization based on the RSM-VT architecture.
    """
    params: Dict[str, Any] = {"TIMEFRAME": tf}
    
    def _suggest(param_name: str) -> None:
        if param_name not in space: return
        spec = space[param_name]
        t = spec["type"]
        if t == "categorical": params[param_name] = trial.suggest_categorical(param_name, spec["choices"])
        elif t == "int": params[param_name] = trial.suggest_int(param_name, spec["low"], spec["high"], step=spec.get("step", 1))
        elif t == "float": params[param_name] = trial.suggest_float(param_name, spec["low"], spec["high"], step=spec.get("step"))

    # --- 1. Macro Trend Filter ---
    _suggest("MACRO_EMA_PERIOD")
    
    # --- 2. Squeeze Parameters ---
    _suggest("KC_MULT")
    
    # --- 3. Momentum Breakout Trigger ---
    _suggest("MOMENTUM_PERIOD")
    
    # --- 4. Exits (Asymmetric Hard Stop vs Fat-Tail Trail) ---
    _suggest("ATR_PERIOD")
    _suggest("LONG_ATR_MULT")
    _suggest("LONG_TRAIL_MULT")
    _suggest("SHORT_ATR_MULT")
    _suggest("SHORT_TP_MULT")
    
    # --- 5. Portfolio Risk Sizing ---
    _suggest("RISK_PER_TRADE")

    # [INSTITUTIONAL] Leverage & Execution Config
    params["LEVERAGE"] = 20
    params["USE_COMPOUNDING"] = True

    return params
