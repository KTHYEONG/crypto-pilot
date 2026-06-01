from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.domain.futures.strategy.config import StrategyConfig


def _build_rule_signals(close_arr: np.ndarray) -> np.ndarray:
    """Build simple directional rule signal from close returns."""
    ret = np.zeros(close_arr.shape[0], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        if close_arr.shape[0] > 1:
            ret[1:] = close_arr[1:] / np.maximum(close_arr[:-1], 1e-12) - 1.0
    signal = np.sign(ret)
    signal[~np.isfinite(signal)] = 0.0
    return signal


def _signals_to_events(signal: np.ndarray) -> np.ndarray:
    """Convert dense rule signal series to event markers."""
    events = np.zeros(signal.shape[0], dtype=bool)
    if signal.shape[0] == 0:
        return events
    events[0] = bool(signal[0] != 0.0)
    if signal.shape[0] > 1:
        events[1:] = signal[1:] != signal[:-1]
    return events


def build_candidate_strategy_output(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """Build candidate/rule-baseline alpha panel from simple rule events.

    Args:
        data_maps: Symbol/timeframe data map.
        symbols: Target symbol list.
        tf: Timeframe key.
        cfg: Strategy config.

    Returns:
        MultiIndex(alpha datetime,symbol) DataFrame with alpha_long/alpha_short.
    """
    rows: list[pd.DataFrame] = []
    event_count_by_symbol: dict[str, int] = {}
    for sym in symbols:
        symbol_map = data_maps.get(sym)
        if symbol_map is None or tf not in symbol_map:
            continue
        frame = symbol_map[tf]
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        if "datetime" not in frame.columns or "close" not in frame.columns:
            continue

        close_arr = frame["close"].to_numpy(dtype=np.float64, copy=False)
        signal = _build_rule_signals(close_arr)
        events = _signals_to_events(signal)
        event_count_by_symbol[sym] = int(np.count_nonzero(events))

        alpha_long = np.clip(signal, 0.0, np.inf).astype(np.float64, copy=False)
        alpha_short = np.clip(-signal, 0.0, np.inf).astype(np.float64, copy=False)

        sym_panel = pd.DataFrame(
            {
                "datetime": frame["datetime"].to_numpy(copy=False),
                "symbol": sym,
                "alpha_long": alpha_long,
                "alpha_short": alpha_short,
            }
        )
        rows.append(sym_panel)

    if not rows:
        panel = pd.DataFrame(columns=["alpha_long", "alpha_short"])
        panel.index = pd.MultiIndex.from_arrays(
            [pd.Index([], dtype="datetime64[ns]"), pd.Index([], dtype="object")],
            names=["datetime", "symbol"],
        )
        return panel

    panel = (
        pd.concat(rows, axis=0, ignore_index=True)
        .set_index(["datetime", "symbol"])
        .sort_index()
    )
    panel = panel[["alpha_long", "alpha_short"]]
    panel.attrs["strategy_name"] = cfg.name
    panel.attrs["candidate_metadata"] = {
        "pipeline": "rule_event_baseline",
        "mode": cfg.name,
        "symbol_count": len(event_count_by_symbol),
        "event_count_by_symbol": event_count_by_symbol,
    }
    panel.attrs["candidate_rule_report"] = {
        "signal_type": "sign_close_return",
        "events_total": int(sum(event_count_by_symbol.values())),
        "event_count_by_symbol": event_count_by_symbol,
    }
    return panel
