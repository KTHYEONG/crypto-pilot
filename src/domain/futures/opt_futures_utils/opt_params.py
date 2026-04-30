from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol

import optuna

from config.opt_config import ENGINE_PARAM_SPACE_FUTURES, OPT_FUTURES_CONFIG


def _suggest_one(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    t = spec["type"]
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if t == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if t == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], step=spec.get("step"))
    raise ValueError(f"Unknown param type: {t}")


def build_full_discovery_space_futures() -> dict[str, Any]:
    """Union of signal / regime / sizing plugin spaces + engine keys."""
    from src.domain.futures.regimes import FUTURES_REGIME_REGISTRY
    from src.domain.futures.signals import FUTURES_SIGNAL_REGISTRY
    from src.domain.futures.sizing import FUTURES_SIZING_REGISTRY

    out: dict[str, Any] = {
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
        sig_inst = FUTURES_SIGNAL_REGISTRY[name]
        for k, spec in sig_inst.param_space.items():
            out.setdefault(k, dict(spec))
    for name in sorted(FUTURES_REGIME_REGISTRY.keys()):
        reg_inst = FUTURES_REGIME_REGISTRY[name]
        for k, spec in reg_inst.param_space.items():
            out.setdefault(k, dict(spec))
    for name in sorted(FUTURES_SIZING_REGISTRY.keys()):
        siz_inst = FUTURES_SIZING_REGISTRY[name]
        for k, spec in siz_inst.param_space.items():
            out.setdefault(k, dict(spec))
    for k, spec in ENGINE_PARAM_SPACE_FUTURES.items():
        out[k] = dict(spec)
    return out


def build_combined_param_space_futures(signal: str, regime: str, sizing: str) -> dict[str, Any]:
    """Locked combo space for combination screener / Phase C deep search."""
    from src.domain.futures.regimes import FUTURES_REGIME_REGISTRY
    from src.domain.futures.signals import FUTURES_SIGNAL_REGISTRY
    from src.domain.futures.sizing import FUTURES_SIZING_REGISTRY

    st = str(signal).upper()
    rt = str(regime).upper()
    sm = str(sizing).lower()

    space: dict[str, Any] = {}
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


class _Stage1ComboLike(Protocol):
    signal: str
    regime: str
    sizing: str


def build_multi_combo_param_space_futures(tops: Sequence[_Stage1ComboLike]) -> dict[str, Any]:
    """Union param space for multiple Stage1 combos — allows TPE to choose signal/regime/sizing."""
    from src.domain.futures.regimes import FUTURES_REGIME_REGISTRY
    from src.domain.futures.signals import FUTURES_SIGNAL_REGISTRY
    from src.domain.futures.sizing import FUTURES_SIZING_REGISTRY

    if not tops:
        raise ValueError(
            "build_multi_combo_param_space_futures requires at least one Stage1 combo."
        )

    all_sigs = list(dict.fromkeys(t.signal for t in tops))
    all_regs = list(dict.fromkeys(t.regime for t in tops))
    all_sizs = list(dict.fromkeys(t.sizing for t in tops))

    space: dict[str, Any] = {
        "SIGNAL_TYPE": {"type": "categorical", "choices": tuple(all_sigs)},
        "REGIME_TYPE": {"type": "categorical", "choices": tuple(all_regs)},
        "SIZING_METHOD": {"type": "categorical", "choices": tuple(all_sizs)},
    }
    for sig in all_sigs:
        for k, v in FUTURES_SIGNAL_REGISTRY[sig].param_space.items():
            space.setdefault(k, dict(v))
    for reg in all_regs:
        for k, v in FUTURES_REGIME_REGISTRY[reg].param_space.items():
            space.setdefault(k, dict(v))
    for siz in all_sizs:
        for k, v in FUTURES_SIZING_REGISTRY[siz].param_space.items():
            space.setdefault(k, dict(v))
    for k, v in ENGINE_PARAM_SPACE_FUTURES.items():
        space[k] = dict(v)
    return space


def suggest_params_futures(trial: optuna.Trial, space: dict[str, Any], tf: str) -> dict[str, Any]:
    """Flat search space; regime branch selects which params affect signals at runtime."""
    params: dict[str, Any] = {"TIMEFRAME": tf}
    for name, spec in space.items():
        if name.startswith("_"):
            continue  # Skip metadata (Institutional Quant Phase C results)
        params[name] = _suggest_one(trial, name, spec)

    _lev_default = int(OPT_FUTURES_CONFIG.get("FUTURES_DISCOVERY_LEVERAGE", 8))
    params["LEVERAGE"] = int(os.getenv("FUTURES_DISCOVERY_LEVERAGE", str(_lev_default)))
    params["USE_COMPOUNDING"] = True
    params["LONG_MOD_FLOOR"] = trial.suggest_float("LONG_MOD_FLOOR", 0.60, 0.85)

    return params
