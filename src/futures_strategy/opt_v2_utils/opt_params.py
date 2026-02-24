"""
최적화 과정에서 탐색할 파라미터의 범위(Search Space)를 정의하고, Optuna trial에 파라미터를 제안함.
선택된 전략 유형에 따라 불필요한 파라미터를 제거(Pruning)하여 최적화 탐색 효율을 극대화함.
"""
import optuna
from typing import Dict, Any

def suggest_params_v2(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    
    for k, spec in space.items():
        if spec["type"] == "categorical":
            params[k] = trial.suggest_categorical(k, spec["choices"])
        elif spec["type"] == "int":
            log: bool = spec.get("log", False)
            step: int = spec.get("step", 1)
            params[k] = trial.suggest_int(k, spec["low"], spec["high"], step=step, log=log)
        elif spec["type"] == "float":
            log: bool = spec.get("log", False)
            step_f: Any = spec.get("step", None)
            params[k] = trial.suggest_float(k, spec["low"], spec["high"], step=step_f, log=log)

    tf: str = str(params.get("TIMEFRAME", "4h"))
    if "MAX_HOLDING_BARS_4H" in params and "MAX_HOLDING_BARS_1D" in params:
        if tf == "4h":
            params["MAX_HOLDING_BARS"] = params["MAX_HOLDING_BARS_4H"]
        else:
            params["MAX_HOLDING_BARS"] = params["MAX_HOLDING_BARS_1D"]
        params.pop("MAX_HOLDING_BARS_4H", None)
        params.pop("MAX_HOLDING_BARS_1D", None)

    entry: str = str(params.get("ENTRY_TYPE", ""))
    if entry != "BOLLINGER":
        params.pop("BB_STD", None)
    if entry != "KELTNER":
        params.pop("KELTNER_ATR_MULT", None)

    trend: str = str(params.get("TREND_FILTER_TYPE", ""))
    if trend != "SUPERTREND":
        params.pop("SUPERTREND_MULT", None)
        params.pop("SUPERTREND_PERIOD", None)
    if trend != "DMI":
        params.pop("DMI_PERIOD", None)
    if trend != "VWAP":
        params.pop("VWAP_STD_MULT", None)

    strength: str = str(params.get("STRENGTH_FILTER_TYPE", ""))
    if strength == "NONE":
        params.pop("STRENGTH_FILTER_PERIOD", None)
    if strength != "ADX":
        params.pop("ADX_THRESHOLD", None)
    if strength != "NATR":
        params.pop("NATR_THRESHOLD", None)
    if strength != "ER":
        params.pop("ER_THRESHOLD", None)

    if not params.get("USE_VOLUME_FILTER", False):
        params.pop("VOLUME_MA_PERIOD", None)
        params.pop("VOLUME_Z_THRESHOLD", None)

    if not params.get("USE_DYNAMIC_RISK", False):
        params.pop("STRONG_REGIME_HURST", None)
        params.pop("STRONG_REGIME_NATR", None)
        params.pop("STRONG_REGIME_MULTIPLIER", None)
        params.pop("WEAK_REGIME_HURST", None)
        params.pop("WEAK_REGIME_MULTIPLIER", None)
        params.pop("PANIC_REGIME_NATR", None)
        params.pop("PANIC_REGIME_MULTIPLIER", None)

    return params
