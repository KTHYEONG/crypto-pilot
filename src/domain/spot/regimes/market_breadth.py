"""
Market-breadth regime with hysteresis (Schmitt-style) and BTC circuit breaker.

Returns per-bar strength in [0, 1]: risk-off / BTC lock => 0; risk-on strength scales
with smoothed breadth signal so regime_state can be OFF / SOFT / FULL (strategies_spot).
Causal only.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Mapping

import numpy as np
import pandas as pd

from src.domain.spot.regimes.registry import register_regime


def compute_market_breadth_regime(
    close_map: Mapping[str, np.ndarray],
    btc_close: np.ndarray,
    w_signal: int,
    k_accel: int,
    mb_floor: float,
    c_hyst: float,
    epsilon_min: float,
    k_cool_down: int,
    btc_drop_threshold: float = -0.15,
) -> np.ndarray:
    if not close_map:
        raise ValueError("close_map must be non-empty.")
    sym_list = list(close_map.keys())
    ref = np.asarray(close_map[sym_list[0]], dtype=np.float64).ravel()
    n = int(ref.size)
    if n < 3:
        return np.zeros(n, dtype=np.float64)

    w_sig = max(2, int(w_signal))
    k_acc = max(1, int(k_accel))
    inner_span = max(2, min(k_acc, n))
    floor = float(np.clip(mb_floor, 0.01, 0.99))
    c_h = float(max(c_hyst, 1e-6))
    eps_min = float(max(epsilon_min, 1e-6))
    k_cd = max(1, int(k_cool_down))

    stacks: list[np.ndarray] = []
    for sym in sym_list:
        c = np.asarray(close_map[sym], dtype=np.float64).ravel()
        if c.size != n:
            raise ValueError(f"Length mismatch for {sym}: expected {n}, got {c.size}.")
        stacks.append(c)
    mat = np.stack(stacks, axis=0)

    ema_each = np.empty_like(mat)
    for i in range(mat.shape[0]):
        ema_each[i] = (
            pd.Series(mat[i]).ewm(span=w_sig, adjust=False).mean().to_numpy(dtype=np.float64)
        )
    bull = (mat > ema_each).astype(np.float64)
    mb_t = np.mean(bull, axis=0)

    mb_s = pd.Series(mb_t)
    mb_inner = mb_s.ewm(span=inner_span, adjust=False).mean()
    signal_t = mb_inner.ewm(span=w_sig, adjust=False).mean().to_numpy(dtype=np.float64)
    roll_std = (
        mb_s.rolling(window=w_sig, min_periods=max(2, min(w_sig, n)))
        .std()
        .to_numpy(dtype=np.float64)
    )
    roll_std = np.nan_to_num(roll_std, nan=0.0, posinf=0.0, neginf=0.0)
    eps_t = np.maximum(c_h * roll_std, eps_min)

    btc = np.asarray(btc_close, dtype=np.float64).ravel()
    if btc.size != n:
        raise ValueError(f"btc_close length mismatch: expected {n}, got {btc.size}.")
    prev = np.roll(btc, k_cd)
    prev[:k_cd] = btc[0]
    safe_prev = np.where(np.abs(prev) > 1e-12, prev, 1.0)
    btc_ret_k = (btc / safe_prev) - 1.0
    btc_lock = (btc_ret_k <= float(btc_drop_threshold)).astype(np.bool_)

    out = np.zeros(n, dtype=np.float64)
    state = 0
    for t in range(n):
        if bool(btc_lock[t]):
            state = 0
            out[t] = 0.0
            continue
        s = float(signal_t[t])
        e = float(eps_t[t])
        if s >= floor + e:
            state = 1
        elif s <= floor - e:
            state = 0
        strength = float(state) * float(np.clip(s, 0.0, 1.0))
        out[t] = strength
    return out


@register_regime
class MarketBreadthRegime:
    name: ClassVar[str] = "MARKET_BREADTH"
    param_space: ClassVar[Dict[str, Any]] = {
        "W_SIGNAL": {"type": "int", "low": 5, "high": 20, "step": 5},
        "K_ACCEL": {"type": "int", "low": 1, "high": 6, "step": 1},
        "MB_FLOOR": {"type": "float", "low": 0.40, "high": 0.70, "step": 0.05},
        "C_HYST": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.5},
        "EPSILON_MIN": {"type": "float", "low": 0.01, "high": 0.05, "step": 0.01},
        "K_COOL_DOWN": {"type": "int", "low": 3, "high": 12, "step": 1},
    }

    def compute(self, data_maps: Dict[str, Dict[str, Any]], params: Dict[str, Any]) -> np.ndarray:
        tf = str(params.get("TIMEFRAME", "4h"))
        symbols = sorted(
            s for s in data_maps if tf in data_maps[s] and data_maps[s][tf] is not None
        )
        if not symbols:
            raise ValueError("market_breadth: empty data_maps")
        ref = data_maps[symbols[0]][tf]
        n = len(ref)
        close_map: Dict[str, np.ndarray] = {}
        for s in symbols:
            df = data_maps[s][tf]
            if len(df) != n:
                raise ValueError(f"market_breadth: length mismatch for {s}")
            close_map[s] = df["close"].to_numpy(dtype=np.float64)
        btc_sym = "KRW-BTC" if "KRW-BTC" in symbols else symbols[0]
        btc_close = data_maps[btc_sym][tf]["close"].to_numpy(dtype=np.float64)
        mb = compute_market_breadth_regime(
            close_map,
            btc_close,
            int(params.get("W_SIGNAL", 10)),
            int(params.get("K_ACCEL", 2)),
            float(params.get("MB_FLOOR", 0.2)),
            float(params.get("C_HYST", 1.0)),
            float(params.get("EPSILON_MIN", 0.02)),
            int(params.get("K_COOL_DOWN", 6)),
        )
        return mb.astype(np.float64)
