from __future__ import annotations

from typing import Any, Dict, Mapping

import optuna

from config.opt_config import ENGINE_PARAM_SPACE, SPOT_SHARED_PARAM_SPACE


def build_full_discovery_space() -> Dict[str, Any]:
    """Union of signal / regime / sizing plugin spaces + engine + shared keys."""
    from src.spot_strategy.regimes import REGIME_REGISTRY
    from src.spot_strategy.signals import SIGNAL_REGISTRY
    from src.spot_strategy.sizing import SIZING_REGISTRY

    out: Dict[str, Any] = {
        "SIGNAL_TYPE": {
            "type": "categorical",
            "choices": tuple(sorted(SIGNAL_REGISTRY.keys())),
        },
        "REGIME_TYPE": {
            "type": "categorical",
            "choices": tuple(sorted(REGIME_REGISTRY.keys())),
        },
        "SIZING_METHOD": {
            "type": "categorical",
            "choices": tuple(sorted(SIZING_REGISTRY.keys())),
        },
    }
    for name in sorted(SIGNAL_REGISTRY.keys()):
        inst = SIGNAL_REGISTRY[name]
        for k, spec in inst.param_space.items():
            out.setdefault(k, dict(spec))
    for name in sorted(REGIME_REGISTRY.keys()):
        inst = REGIME_REGISTRY[name]
        for k, spec in inst.param_space.items():
            out.setdefault(k, dict(spec))
    for name in sorted(SIZING_REGISTRY.keys()):
        inst = SIZING_REGISTRY[name]
        for k, spec in inst.param_space.items():
            out.setdefault(k, dict(spec))
    for k, spec in ENGINE_PARAM_SPACE.items():
        out[k] = dict(spec)
    for k, spec in SPOT_SHARED_PARAM_SPACE.items():
        out.setdefault(k, dict(spec))
    return out


def build_combined_param_space(signal: str, regime: str, sizing: str) -> Dict[str, Any]:
    from src.spot_strategy.regimes import REGIME_REGISTRY
    from src.spot_strategy.signals import SIGNAL_REGISTRY
    from src.spot_strategy.sizing import SIZING_REGISTRY

    space: Dict[str, Any] = {}
    space["SIGNAL_TYPE"] = {"type": "categorical", "choices": (signal,)}
    space["REGIME_TYPE"] = {"type": "categorical", "choices": (regime,)}
    space["SIZING_METHOD"] = {"type": "categorical", "choices": (sizing,)}
    sig = SIGNAL_REGISTRY[signal]
    reg = REGIME_REGISTRY[regime]
    siz = SIZING_REGISTRY[sizing]
    for k, v in sig.param_space.items():
        space[k] = dict(v)
    for k, v in reg.param_space.items():
        space[k] = dict(v)
    for k, v in siz.param_space.items():
        space[k] = dict(v)
    for k, v in ENGINE_PARAM_SPACE.items():
        space[k] = dict(v)
    for k, v in SPOT_SHARED_PARAM_SPACE.items():
        space.setdefault(k, dict(v))
    return space


_CORE_ORDER: tuple[str, ...] = (
    "SIZING_METHOD",
    "RISK_PER_TRADE",
    "MAX_EXPOSURE",
    "KELLY_FRACTION",
    "MAX_CAP_PER_COIN",
    "MAX_CAP_LIQUID_MAJOR",
    "MAX_CAP_TRENDING_ALT",
    "ATR_PERIOD",
)


def _iter_param_names_define_by_run(space: Mapping[str, Any], signal_type: str) -> list[str]:
    st = str(signal_type).upper()
    out: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name in space and name not in seen:
            out.append(name)
            seen.add(name)

    _add("SIGNAL_TYPE")
    _add("REGIME_TYPE")
    for name in _CORE_ORDER:
        _add(name)
    from src.spot_strategy.regimes import REGIME_REGISTRY
    from src.spot_strategy.signals import SIGNAL_REGISTRY
    from src.spot_strategy.sizing import SIZING_REGISTRY

    if st not in SIGNAL_REGISTRY:
        raise KeyError(f"Unknown SIGNAL_TYPE {st!r}")
    for k in sorted(SIGNAL_REGISTRY[st].param_space.keys()):
        _add(k)
    for rname in sorted(REGIME_REGISTRY.keys()):
        for k in sorted(REGIME_REGISTRY[rname].param_space.keys()):
            _add(k)
    for zname in sorted(SIZING_REGISTRY.keys()):
        for k in sorted(SIZING_REGISTRY[zname].param_space.keys()):
            _add(k)
    for k in sorted(ENGINE_PARAM_SPACE.keys()):
        _add(k)
    for k in sorted(SPOT_SHARED_PARAM_SPACE.keys()):
        _add(k)
    for name in sorted(space.keys()):
        if name not in seen:
            _add(name)
    return out


def _suggest_one(
    trial: optuna.Trial,
    space: Mapping[str, Any],
    params: Dict[str, Any],
    param_name: str,
) -> None:
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


def suggest_params_spot(
    trial: optuna.Trial,
    space: Dict[str, Any],
    tf: str,
    *,
    locked: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {**dict(locked or {}), "TIMEFRAME": tf}

    if "SIGNAL_TYPE" not in space:
        raise ValueError("space must contain SIGNAL_TYPE for spot optimization.")
    if "REGIME_TYPE" not in space:
        raise ValueError("space must contain REGIME_TYPE for spot optimization.")

    if "SIGNAL_TYPE" not in params:
        _suggest_one(trial, space, params, "SIGNAL_TYPE")
    signal_type = str(params["SIGNAL_TYPE"]).upper()
    params["SIGNAL_TYPE"] = signal_type

    if "REGIME_TYPE" not in params:
        _suggest_one(trial, space, params, "REGIME_TYPE")
    params["REGIME_TYPE"] = str(params["REGIME_TYPE"]).upper()

    if "SIZING_METHOD" in space and "SIZING_METHOD" not in params:
        _suggest_one(trial, space, params, "SIZING_METHOD")
    if "SIZING_METHOD" in params:
        params["SIZING_METHOD"] = str(params["SIZING_METHOD"]).lower()

    for name in _iter_param_names_define_by_run(space, signal_type):
        if name in params:
            continue
        if name in ("SIGNAL_TYPE", "REGIME_TYPE"):
            continue
        _suggest_one(trial, space, params, name)

    params["LEVERAGE"] = 1
    params["USE_COMPOUNDING"] = True

    if "EMA_SLOW_PERIOD" in params:
        params["EMA_SLOW_PERIOD"] = int(max(100, params["EMA_SLOW_PERIOD"]))
    if "RSI_PERIOD" in params:
        params["RSI_PERIOD"] = int(max(2, params["RSI_PERIOD"]))

    return params


def spot_define_by_run_param_names(space: Mapping[str, Any], signal_type: str) -> list[str]:
    return _iter_param_names_define_by_run(space, signal_type)
