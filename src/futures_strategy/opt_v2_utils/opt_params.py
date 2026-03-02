from __future__ import annotations
import optuna
from typing import Dict, Any

def suggest_params_v2(trial: optuna.Trial, space: Dict[str, Any], tf: str) -> Dict[str, Any]:
    """
    Suggests parameters for optimization based on a timeframe-specific search space.
    """
    params: Dict[str, Any] = {"TIMEFRAME": tf}
    
    def _suggest(param_name: str) -> None:
        if param_name not in space: return
        spec = space[param_name]
        t = spec["type"]
        if t == "categorical": params[param_name] = trial.suggest_categorical(param_name, spec["choices"])
        elif t == "int": params[param_name] = trial.suggest_int(param_name, spec["low"], spec["high"], step=spec.get("step", 1))
        elif t == "float": params[param_name] = trial.suggest_float(param_name, spec["low"], spec["high"], step=spec.get("step"))

    # 1. Structural Configurations
    for cat_param in ["STOP_LOSS_TYPE", "USE_TAKE_PROFIT", "EXIT_TYPE"]: _suggest(cat_param)

    # 2. Strategic Factor Weights
    for w in ["W_BREAKOUT", "W_TREND", "W_VOLUME", "W_MEAN_REVERSION"]: _suggest(w)
    
    # Priority 1: Adaptive Thresholds
    _suggest("THRESHOLD_LOOKBACK")
    _suggest("THRESHOLD_QUANTILE")

    # 3. Core & Shared Signal Parameters
    params["ATR_PERIOD"] = 14
    
    # [INSTITUTIONAL] 타임프레임별 리스크 최적화 (Kelly-Optimum)
    # 4H(추세추종)는 MDD 15% 수준을 타겟으로 3.5% 리스크 부여, 1H(역추세 단타)는 연쇄 손절 붕괴를 막기 위해 1.5%로 축소
    if tf == "4h":
        params["RISK_PER_TRADE"] = 0.035
    else:
        params["RISK_PER_TRADE"] = 0.015
        
    _suggest("TIME_EXIT_PROFIT_THRESHOLD")
    _suggest("LEVERAGE")
    _suggest("ENTRY_PERIOD")
    _suggest("MAX_HOLDING_BARS")
    _suggest("BB_STD")
    _suggest("KC_MULT")
    _suggest("VOL_WINDOW")

    # 4. Timeframe Specific Specialized Indicators
    if tf == "1h":
        _suggest("VWAP_STD_MULT")
        _suggest("STOCH_RSI_PERIOD")
        _suggest("STOCH_RSI_EXTREME")
        _suggest("CMF_PERIOD")
    elif tf == "4h":
        _suggest("MACRO_SMA_PERIOD")

    # 5. Exit & Stop Loss Specifics
    _suggest("RSI_EXIT_THRESHOLD")
    _suggest("ENABLE_TREND_EXIT")
    
    exit_type = params.get("EXIT_TYPE")
    if exit_type == "ATR":
        _suggest("ATR_MULTIPLIER")
        _suggest("TRAILING_ACTIVATION_ATR")
    elif exit_type == "PARABOLIC_SAR":
        _suggest("PSAR_STEP")
        _suggest("PSAR_MAX")

    stop_loss_type = params.get("STOP_LOSS_TYPE")
    if stop_loss_type == "FIXED": _suggest("STOP_LOSS_PCT")
    elif stop_loss_type == "ATR": _suggest("ATR_STOP_LOSS_MULT")

    if params.get("USE_TAKE_PROFIT"): _suggest("TAKE_PROFIT_ATR_MULT")

    # 6. Constraints & Pruning
    if params.get("STOP_LOSS_TYPE") == "ATR" and params.get("USE_TAKE_PROFIT"):
        if params.get("TAKE_PROFIT_ATR_MULT", 0) / params.get("ATR_STOP_LOSS_MULT", 1) < 1.2:
            params["_INVALID_CONSTRAINT"] = True

    # [NEW] 역추세(1H)는 시계열 평균 회귀의 성질상 이익을 짧게 끊어내야 하므로 익절(TP) 강제
    if tf == "1h" and not params.get("USE_TAKE_PROFIT", False):
        params["_INVALID_CONSTRAINT"] = True

    return params
