from __future__ import annotations

from typing import Any, Dict

import optuna

from config.opt_config import ENGINE_PARAM_SPACE_FUTURES


def _suggest_one(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
    t = spec["type"]
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if t == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if t == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], step=spec.get("step"))
    raise ValueError(f"Unknown param type: {t}")


def build_full_discovery_space_futures() -> Dict[str, Any]:
    """Union of signal / regime / sizing plugin spaces + engine keys."""
    from src.futures_strategy.regimes import FUTURES_REGIME_REGISTRY
    from src.futures_strategy.signals import FUTURES_SIGNAL_REGISTRY
    from src.futures_strategy.sizing import FUTURES_SIZING_REGISTRY

    out: Dict[str, Any] = {
        "SIGNAL_TYPE": {
            "type": "categorical",
            "choices": tuple(sorted(FUTURES_SIGNAL_REGISTRY.keys())),
        },
        "REGIME_TYPE": {
            "type": "categorical",
            "choices": tuple(sorted(FUTURES_REGIME_REGISTRY.keys())),
        },
        "SIZING_METHOD": {
            "type": "categorical",
            "choices": tuple(sorted(FUTURES_SIZING_REGISTRY.keys())),
        },
    }
    for name in sorted(FUTURES_SIGNAL_REGISTRY.keys()):
        inst = FUTURES_SIGNAL_REGISTRY[name]
        for k, spec in inst.param_space.items():
            out.setdefault(k, dict(spec))
    for name in sorted(FUTURES_REGIME_REGISTRY.keys()):
        inst = FUTURES_REGIME_REGISTRY[name]
        for k, spec in inst.param_space.items():
            out.setdefault(k, dict(spec))
    for name in sorted(FUTURES_SIZING_REGISTRY.keys()):
        inst = FUTURES_SIZING_REGISTRY[name]
        for k, spec in inst.param_space.items():
            out.setdefault(k, dict(spec))
    for k, spec in ENGINE_PARAM_SPACE_FUTURES.items():
        out[k] = dict(spec)
    return out


def build_combined_param_space_futures(signal: str, regime: str, sizing: str) -> Dict[str, Any]:
    """Locked combo space for combination screener / Phase C deep search."""
    from src.futures_strategy.regimes import FUTURES_REGIME_REGISTRY
    from src.futures_strategy.signals import FUTURES_SIGNAL_REGISTRY
    from src.futures_strategy.sizing import FUTURES_SIZING_REGISTRY

    st = str(signal).upper()
    rt = str(regime).upper()
    sm = str(sizing).lower()

    space: Dict[str, Any] = {}
    space["SIGNAL_TYPE"] = {"type": "categorical", "choices": (st,)}
    space["REGIME_TYPE"] = {"type": "categorical", "choices": (rt,)}
    space["SIZING_METHOD"] = {"type": "categorical", "choices": (sm,)}

    sig = FUTURES_SIGNAL_REGISTRY[st]
    reg = FUTURES_REGIME_REGISTRY[rt]
    siz = FUTURES_SIZING_REGISTRY[sm]
    for k, v in sig.param_space.items():
        space[k] = dict(v)
    for k, v in reg.param_space.items():
        space[k] = dict(v)
    for k, v in siz.param_space.items():
        space[k] = dict(v)
    for k, v in ENGINE_PARAM_SPACE_FUTURES.items():
        space[k] = dict(v)
    return space


def suggest_params_futures(trial: optuna.Trial, space: Dict[str, Any], tf: str) -> Dict[str, Any]:
    """Flat search space; regime branch selects which params affect signals at runtime."""
    params: Dict[str, Any] = {"TIMEFRAME": tf}
    for name, spec in space.items():
        params[name] = _suggest_one(trial, name, spec)

    params["LEVERAGE"] = 20
    params["USE_COMPOUNDING"] = True

    return params
