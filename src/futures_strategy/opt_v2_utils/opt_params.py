from __future__ import annotations
import optuna
from typing import Dict, Any

def suggest_params_v2(trial: optuna.Trial, space: Dict[str, Any], tf: str) -> Dict[str, Any]:
    """
    Suggests parameters for optimization based on the updated TSMOM+ATR regime.
    """
    params: Dict[str, Any] = {"TIMEFRAME": tf}
    
    def _suggest(param_name: str) -> None:
        if param_name not in space: return
        spec = space[param_name]
        t = spec["type"]
        if t == "categorical": params[param_name] = trial.suggest_categorical(param_name, spec["choices"])
        elif t == "int": params[param_name] = trial.suggest_int(param_name, spec["low"], spec["high"], step=spec.get("step", 1))
        elif t == "float": params[param_name] = trial.suggest_float(param_name, spec["low"], spec["high"], step=spec.get("step"))

    # 4 Core Parameters
    _suggest("TSMOM_ENTRY_THRESHOLD")
    _suggest("TSMOM_WEIGHT_DECAY")
    _suggest("ATR_WINDOW")
    _suggest("ATR_MULTIPLIER")
    _suggest("ATR_PRC_WINDOW")
    
    # [NEW] Asymmetry & Sizing Parameters
    _suggest("VELOCITY_K")
    _suggest("RISK_PER_TRADE")

    # Hardcoded constraints for pure alpha discovery (no compounding/leverage noise)
    params["LEVERAGE"] = 1           # 1x leverage
    params["STOP_LOSS_TYPE"] = "ATR"
    params["EXIT_TYPE"] = "ATR"

    return params
