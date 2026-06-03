from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig

_BPS_SCALE = 1e4
_ATR_PERIOD = 14
_EXIT_POLICY_VERSION = "candidate_label_atr_v1"
_logger = logging.getLogger(__name__)


def _compute_atr_2d(aligned: AlignedMarketData, period: int = _ATR_PERIOD) -> np.ndarray:
    high = aligned.high_2d
    low = aligned.low_2d
    close = aligned.close_2d
    prev_close = np.vstack([close[:1], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    out = np.full_like(tr, np.nan, dtype=np.float64)
    for t in range(tr.shape[0]):
        start = max(0, t - period + 1)
        window = tr[start : t + 1]
        with np.errstate(invalid="ignore"):
            out[t] = np.nanmean(window, axis=0)
    return out


def _find_symbol_index(symbols: tuple[str, ...], symbol: str) -> int:
    for idx, value in enumerate(symbols):
        if value == symbol:
            return idx
    raise KeyError(f"unknown symbol: {symbol}")


def label_candidate_events(
    *,
    events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
) -> pd.DataFrame:
    """Attach leak-free forward outcomes to candidate events.

    Time Complexity: O(E * H), where E is number of events and H is holding horizon.
    Space Complexity: O(T * N) for ATR cache.
    """
    if events.empty:
        return events.copy()

    required = {"symbol", "side", "entry_idx", "expected_holding_bars", "stop_atr_mult", "take_profit_atr_mult"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"missing required event columns: {sorted(missing)}")

    atr_2d = _compute_atr_2d(aligned)
    out = events.copy()
    t_len = aligned.close_2d.shape[0]

    gross_list: list[float] = []
    cost_list: list[float] = []
    edge_list: list[float] = []
    barrier_label_list: list[int] = []
    profitable_label_list: list[int] = []
    tte_list: list[int] = []
    mae_list: list[float] = []
    mfe_list: list[float] = []
    rv_list: list[float] = []
    exit_reason_list: list[str] = []
    exit_idx_list: list[int] = []
    exit_policy_version_list: list[str] = []
    same_bar_collision_list: list[int] = []

    for row in out.itertuples(index=False):
        symbol = str(row.symbol)
        side = int(row.side)
        entry_idx = int(row.entry_idx)
        horizon = max(int(row.expected_holding_bars), 1)
        stop_mult = float(row.stop_atr_mult)
        tp_mult = float(row.take_profit_atr_mult)

        sym_idx = _find_symbol_index(aligned.symbols, symbol)
        decision_idx = entry_idx - 1
        if decision_idx < 0 or entry_idx >= t_len:
            gross_list.append(np.nan)
            cost_list.append(np.nan)
            edge_list.append(np.nan)
            barrier_label_list.append(0)
            profitable_label_list.append(0)
            tte_list.append(0)
            mae_list.append(np.nan)
            mfe_list.append(np.nan)
            rv_list.append(np.nan)
            exit_reason_list.append("invalid")
            exit_idx_list.append(-1)
            exit_policy_version_list.append(_EXIT_POLICY_VERSION)
            same_bar_collision_list.append(0)
            continue

        entry_px = float(aligned.open_2d[entry_idx, sym_idx])
        if not np.isfinite(entry_px) or entry_px <= 0.0:
            gross_list.append(np.nan)
            cost_list.append(np.nan)
            edge_list.append(np.nan)
            barrier_label_list.append(0)
            profitable_label_list.append(0)
            tte_list.append(0)
            mae_list.append(np.nan)
            mfe_list.append(np.nan)
            rv_list.append(np.nan)
            exit_reason_list.append("invalid")
            exit_idx_list.append(-1)
            exit_policy_version_list.append(_EXIT_POLICY_VERSION)
            same_bar_collision_list.append(0)
            continue

        exit_limit = min(entry_idx + horizon - 1, t_len - 1)
        high_path = aligned.high_2d[entry_idx : exit_limit + 1, sym_idx]
        low_path = aligned.low_2d[entry_idx : exit_limit + 1, sym_idx]
        close_path = aligned.close_2d[entry_idx : exit_limit + 1, sym_idx]

        atr = float(atr_2d[decision_idx, sym_idx])
        atr = atr if np.isfinite(atr) and atr > 0.0 else entry_px * 0.01
        tp_thr = (tp_mult * atr) / entry_px
        sl_thr = (stop_mult * atr) / entry_px

        if side > 0:
            fav = (high_path / entry_px) - 1.0
            adv = (low_path / entry_px) - 1.0
        else:
            fav = (entry_px / np.maximum(low_path, 1e-12)) - 1.0
            adv = (entry_px / np.maximum(high_path, 1e-12)) - 1.0

        tp_hits = np.flatnonzero(np.isfinite(fav) & (fav >= tp_thr))
        sl_hits = np.flatnonzero(np.isfinite(adv) & (adv <= -sl_thr))

        tp_i = int(tp_hits[0]) if tp_hits.size > 0 else math.inf
        sl_i = int(sl_hits[0]) if sl_hits.size > 0 else math.inf
        same_bar_collision = int(np.isfinite(tp_i) and np.isfinite(sl_i) and tp_i == sl_i)
        if np.isfinite(sl_i) and (not np.isfinite(tp_i) or sl_i <= tp_i):
            exit_off = int(sl_i)
            barrier_label = 0
            exit_reason = "stop_loss"
        elif np.isfinite(tp_i):
            exit_off = int(tp_i)
            barrier_label = 1
            exit_reason = "take_profit"
        else:
            exit_off = int(close_path.shape[0] - 1)
            barrier_label = 0
            exit_reason = "time_exit"

        exit_px = float(close_path[exit_off])
        if side > 0:
            gross_ret_bps = ((exit_px / entry_px) - 1.0) * _BPS_SCALE
            path_ret = (close_path / entry_px) - 1.0
        else:
            gross_ret_bps = ((entry_px / max(exit_px, 1e-12)) - 1.0) * _BPS_SCALE
            path_ret = (entry_px / np.maximum(close_path, 1e-12)) - 1.0
        path_ret = path_ret[: exit_off + 1]

        ex_ante_cost_bps = float(getattr(row, "cost_floor_bps", np.nan))
        if not np.isfinite(ex_ante_cost_bps):
            if aligned.execution_cost_bps_2d is not None:
                ex_ante_cost_bps = float(aligned.execution_cost_bps_2d[decision_idx, sym_idx])
            else:
                ex_ante_cost_bps = 0.0
        hurdle_bps = float(getattr(row, "hurdle_bps", 0.0))
        edge_after_hurdle_bps = gross_ret_bps - ex_ante_cost_bps - hurdle_bps

        barrier_first_label = 1 if barrier_label == 1 and edge_after_hurdle_bps > 0.0 else 0
        profitable_after_hurdle_label = 1 if edge_after_hurdle_bps > 0.0 else 0

        valid_path = path_ret[np.isfinite(path_ret)]
        mae_bps = float(np.min(valid_path) * _BPS_SCALE) if valid_path.size > 0 else np.nan
        mfe_bps = float(np.max(valid_path) * _BPS_SCALE) if valid_path.size > 0 else np.nan
        rv_bps = (
            float(np.std(np.diff(np.log(np.maximum(close_path[: exit_off + 1], 1e-12)))) * _BPS_SCALE)
            if exit_off >= 1
            else 0.0
        )

        gross_list.append(float(gross_ret_bps))
        cost_list.append(float(ex_ante_cost_bps))
        edge_list.append(float(edge_after_hurdle_bps))
        barrier_label_list.append(int(barrier_first_label))
        profitable_label_list.append(int(profitable_after_hurdle_label))
        tte_list.append(int(exit_off + 1))
        mae_list.append(float(mae_bps))
        mfe_list.append(float(mfe_bps))
        rv_list.append(float(rv_bps))
        exit_reason_list.append(exit_reason)
        exit_idx_list.append(int(entry_idx + exit_off))
        exit_policy_version_list.append(_EXIT_POLICY_VERSION)
        same_bar_collision_list.append(int(same_bar_collision))

    out["barrier_first_label"] = np.asarray(barrier_label_list, dtype=np.int8)
    out["profitable_after_hurdle_label"] = np.asarray(profitable_label_list, dtype=np.int8)
    out["gross_fwd_bps"] = np.asarray(gross_list, dtype=np.float64)
    out["ex_ante_cost_bps"] = np.asarray(cost_list, dtype=np.float64)
    out["edge_after_hurdle_bps"] = np.asarray(edge_list, dtype=np.float64)
    out["triple_barrier_label"] = np.asarray(barrier_label_list, dtype=np.int8)
    out["time_to_exit_bars"] = np.asarray(tte_list, dtype=np.int32)
    out["mae_bps"] = np.asarray(mae_list, dtype=np.float64)
    out["mfe_bps"] = np.asarray(mfe_list, dtype=np.float64)
    out["realized_vol_bps"] = np.asarray(rv_list, dtype=np.float64)
    out["exit_reason"] = np.asarray(exit_reason_list, dtype=object)
    out["exit_idx"] = np.asarray(exit_idx_list, dtype=np.int32)
    out["exit_policy_version"] = np.asarray(exit_policy_version_list, dtype=object)
    out["same_bar_collision"] = np.asarray(same_bar_collision_list, dtype=np.int8)

    _barrier_labels = np.asarray(barrier_label_list, dtype=np.int8)
    _profitable_labels = np.asarray(profitable_label_list, dtype=np.int8)
    _edge = np.asarray(edge_list, dtype=np.float64)
    _finite_edge = _edge[np.isfinite(_edge)]
    _barrier_label1_rate = float(_barrier_labels.mean()) if len(_barrier_labels) > 0 else 0.0
    _gate_label1_rate = float(_profitable_labels.mean()) if len(_profitable_labels) > 0 else 0.0
    _logger.debug(
        "[DIAG][LABEL] events=%d barrier_label1_rate=%.3f gate_label1_rate=%.3f "
        "mean_edge=%.1f median_edge=%.1f "
        "pct_edge_pos=%.3f p10_edge=%.1f p90_edge=%.1f",
        len(_barrier_labels),
        _barrier_label1_rate,
        _gate_label1_rate,
        float(np.mean(_finite_edge)) if len(_finite_edge) > 0 else float("nan"),
        float(np.median(_finite_edge)) if len(_finite_edge) > 0 else float("nan"),
        float((_finite_edge > 0).mean()) if len(_finite_edge) > 0 else 0.0,
        float(np.percentile(_finite_edge, 10)) if len(_finite_edge) > 0 else float("nan"),
        float(np.percentile(_finite_edge, 90)) if len(_finite_edge) > 0 else float("nan"),
    )
    return out
