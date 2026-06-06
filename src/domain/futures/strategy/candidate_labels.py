from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.execution_cost import ExecutionCostModel

_BPS_SCALE = 1e4
_ATR_PERIOD = 14
_ATR_FALLBACK_FRACTION = 0.01  # L-4: fallback when ATR unavailable; was inline magic number
_EXIT_POLICY_VERSION = "candidate_label_atr_v2"
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


def _invalid_label_payload() -> dict[str, float | int | str]:
    return {
        "gross_event_bps": float("nan"),
        "execution_cost_bps": float("nan"),
        "realized_funding_bps": float("nan"),
        "net_event_bps": float("nan"),
        "triple_barrier_label": 0,
        "barrier_first_label": 0,
        "profitable_after_hurdle_label": 0,
        "time_to_exit_bars": 0,
        "mae_bps": float("nan"),
        "mfe_bps": float("nan"),
        "realized_vol_bps": float("nan"),
        "sl_thr_bps": float("nan"),
        "exit_reason": "invalid",
        "exit_idx": -1,
        "same_bar_collision": 0,
        "net_return_r": float("nan"),
        "mae_r": float("nan"),
        "mfe_r": float("nan"),
        "risk_unit_bps": float("nan"),
    }


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
    cost_model = ExecutionCostModel(
        maker_fee_bps=float(getattr(cfg, "maker_fee_bps", 2.0)),
        taker_fee_bps=float(getattr(cfg, "taker_fee_bps", 5.0)),
        maker_ratio=float(getattr(cfg, "maker_ratio", 0.75)),
        slippage_bps=float(getattr(cfg, "slippage_bps", 1.0)),
        impact_coeff_bps=float(getattr(cfg, "impact_coeff_bps", 0.0)),
        stress_multiplier=float(getattr(cfg, "cost_stress_multiplier", 1.5)),
    )
    taker_round_trip_bps = cost_model.taker_round_trip_bps()

    gross_list: list[float] = []
    cost_list: list[float] = []
    funding_cost_list: list[float] = []
    edge_list: list[float] = []
    raw_barrier_label_list: list[int] = []  # L-1: raw triple-barrier result (TP=1, SL/time=0)
    barrier_label_list: list[int] = []      # L-1: cost-conditioned label (TP AND edge>0)
    profitable_label_list: list[int] = []
    tte_list: list[int] = []
    mae_list: list[float] = []
    mfe_list: list[float] = []
    rv_list: list[float] = []
    sl_thr_bps_list: list[float] = []  # RC3: realizable stop loss in bps (for q10 clipping)
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
            invalid = _invalid_label_payload()
            gross_list.append(float(invalid["gross_event_bps"]))
            cost_list.append(float(invalid["execution_cost_bps"]))
            funding_cost_list.append(float(invalid["realized_funding_bps"]))
            edge_list.append(float(invalid["net_event_bps"]))
            raw_barrier_label_list.append(int(invalid["triple_barrier_label"]))
            barrier_label_list.append(int(invalid["barrier_first_label"]))
            profitable_label_list.append(int(invalid["profitable_after_hurdle_label"]))
            tte_list.append(int(invalid["time_to_exit_bars"]))
            mae_list.append(float(invalid["mae_bps"]))
            mfe_list.append(float(invalid["mfe_bps"]))
            rv_list.append(float(invalid["realized_vol_bps"]))
            sl_thr_bps_list.append(float(invalid["sl_thr_bps"]))
            exit_reason_list.append(str(invalid["exit_reason"]))
            exit_idx_list.append(int(invalid["exit_idx"]))
            exit_policy_version_list.append(_EXIT_POLICY_VERSION)
            same_bar_collision_list.append(int(invalid["same_bar_collision"]))
            continue

        entry_px = float(aligned.open_2d[entry_idx, sym_idx])
        if not np.isfinite(entry_px) or entry_px <= 0.0:
            invalid = _invalid_label_payload()
            gross_list.append(float(invalid["gross_event_bps"]))
            cost_list.append(float(invalid["execution_cost_bps"]))
            funding_cost_list.append(float(invalid["realized_funding_bps"]))
            edge_list.append(float(invalid["net_event_bps"]))
            raw_barrier_label_list.append(int(invalid["triple_barrier_label"]))
            barrier_label_list.append(int(invalid["barrier_first_label"]))
            profitable_label_list.append(int(invalid["profitable_after_hurdle_label"]))
            tte_list.append(int(invalid["time_to_exit_bars"]))
            mae_list.append(float(invalid["mae_bps"]))
            mfe_list.append(float(invalid["mfe_bps"]))
            rv_list.append(float(invalid["realized_vol_bps"]))
            sl_thr_bps_list.append(float(invalid["sl_thr_bps"]))
            exit_reason_list.append(str(invalid["exit_reason"]))
            exit_idx_list.append(int(invalid["exit_idx"]))
            exit_policy_version_list.append(_EXIT_POLICY_VERSION)
            same_bar_collision_list.append(int(invalid["same_bar_collision"]))
            continue

        exit_limit = min(entry_idx + horizon - 1, t_len - 1)
        high_path = aligned.high_2d[entry_idx : exit_limit + 1, sym_idx]
        low_path = aligned.low_2d[entry_idx : exit_limit + 1, sym_idx]
        close_path = aligned.close_2d[entry_idx : exit_limit + 1, sym_idx]
        next_open_idx = entry_idx + horizon

        atr = float(atr_2d[decision_idx, sym_idx])
        atr = atr if np.isfinite(atr) and atr > 0.0 else entry_px * _ATR_FALLBACK_FRACTION  # L-4
        tp_thr = (tp_mult * atr) / entry_px
        sl_thr = (stop_mult * atr) / entry_px

        if side > 0:
            fav = (high_path / entry_px) - 1.0
            adv = (low_path / entry_px) - 1.0
        else:
            fav = (entry_px / np.maximum(low_path, 1e-12)) - 1.0
            adv = (entry_px / np.maximum(high_path, 1e-12)) - 1.0

        # L-3: engine_aligned → skip first min_holding_bars when scanning barriers
        min_hold = max(0, int(getattr(row, "min_holding_bars", 0))) if cfg.exit_policy_mode == "engine_aligned" else 0
        scan_from = min_hold

        fav_scan = fav[scan_from:]
        adv_scan = adv[scan_from:]
        tp_hits_rel = np.flatnonzero(np.isfinite(fav_scan) & (fav_scan >= tp_thr))
        sl_hits_rel = np.flatnonzero(np.isfinite(adv_scan) & (adv_scan <= -sl_thr))
        tp_hits = tp_hits_rel + scan_from if tp_hits_rel.size > 0 else tp_hits_rel
        sl_hits = sl_hits_rel + scan_from if sl_hits_rel.size > 0 else sl_hits_rel

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

        # L-2: use barrier price as exit_px (not close of exit bar)
        # Prevents SL optimism (low triggers but close rebounds) and TP pessimism.
        if exit_reason == "take_profit":
            exit_px = entry_px * (1.0 + side * tp_thr)
        elif exit_reason == "stop_loss":
            exit_px = entry_px * (1.0 - side * sl_thr)
        else:  # time_exit: next decision open to match engine lifecycle
            if next_open_idx >= t_len:
                invalid = _invalid_label_payload()
                gross_list.append(float(invalid["gross_event_bps"]))
                cost_list.append(float(invalid["execution_cost_bps"]))
                funding_cost_list.append(float(invalid["realized_funding_bps"]))
                edge_list.append(float(invalid["net_event_bps"]))
                raw_barrier_label_list.append(int(invalid["triple_barrier_label"]))
                barrier_label_list.append(int(invalid["barrier_first_label"]))
                profitable_label_list.append(int(invalid["profitable_after_hurdle_label"]))
                tte_list.append(int(invalid["time_to_exit_bars"]))
                mae_list.append(float(invalid["mae_bps"]))
                mfe_list.append(float(invalid["mfe_bps"]))
                rv_list.append(float(invalid["realized_vol_bps"]))
                sl_thr_bps_list.append(float(invalid["sl_thr_bps"]))
                exit_reason_list.append(str(invalid["exit_reason"]))
                exit_idx_list.append(int(invalid["exit_idx"]))
                exit_policy_version_list.append(_EXIT_POLICY_VERSION)
                same_bar_collision_list.append(int(invalid["same_bar_collision"]))
                continue
            exit_px = float(aligned.open_2d[next_open_idx, sym_idx])
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
                ex_ante_cost_bps = taker_round_trip_bps
        ex_ante_cost_bps = max(ex_ante_cost_bps, taker_round_trip_bps)
        funding_stop = min((next_open_idx if exit_reason == "time_exit" else entry_idx + exit_off + 1), t_len)
        realized_funding_bps = 0.0
        if funding_stop > entry_idx:
            funding_path = aligned.funding_2d[entry_idx:funding_stop, sym_idx]
            finite_funding = funding_path[np.isfinite(funding_path)]
            if finite_funding.size > 0:
                realized_funding_bps = float(np.sum(finite_funding) * side * _BPS_SCALE)
        hurdle_bps = float(getattr(row, "hurdle_bps", 0.0))
        edge_after_hurdle_bps = gross_ret_bps - ex_ante_cost_bps - realized_funding_bps - hurdle_bps

        # L-1: separate raw barrier result from cost-conditioned label
        raw_barrier = int(barrier_label)  # 1=TP reached first, 0=SL or time
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

        # RC3: realizable stop loss in bps — used to clip paper-MAE in q10 target
        sl_thr_bps = sl_thr * _BPS_SCALE  # fractional stop → bps (always positive)

        gross_list.append(float(gross_ret_bps))
        cost_list.append(float(ex_ante_cost_bps))
        funding_cost_list.append(float(realized_funding_bps))
        edge_list.append(float(edge_after_hurdle_bps))
        raw_barrier_label_list.append(raw_barrier)
        barrier_label_list.append(int(barrier_first_label))
        profitable_label_list.append(int(profitable_after_hurdle_label))
        tte_list.append(int(horizon if exit_reason == "time_exit" else exit_off + 1))
        mae_list.append(float(mae_bps))
        sl_thr_bps_list.append(float(sl_thr_bps))
        mfe_list.append(float(mfe_bps))
        rv_list.append(float(rv_bps))
        exit_reason_list.append(exit_reason)
        exit_idx_list.append(int(next_open_idx if exit_reason == "time_exit" else entry_idx + exit_off))
        exit_policy_version_list.append(_EXIT_POLICY_VERSION)
        same_bar_collision_list.append(int(same_bar_collision))

    out["barrier_first_label"] = np.asarray(barrier_label_list, dtype=np.int8)
    out["profitable_after_hurdle_label"] = np.asarray(profitable_label_list, dtype=np.int8)
    out["event_id"] = np.arange(len(out), dtype=np.int64)
    out["gross_event_bps"] = np.asarray(gross_list, dtype=np.float64)
    out["gross_fwd_bps"] = np.asarray(gross_list, dtype=np.float64)
    out["gross_direction_label"] = (np.asarray(gross_list, dtype=np.float64) > 0.0).astype(np.int8)
    out["execution_cost_bps"] = np.asarray(cost_list, dtype=np.float64)
    out["realized_funding_bps"] = np.asarray(funding_cost_list, dtype=np.float64)
    out["net_event_bps"] = np.asarray(edge_list, dtype=np.float64)
    out["ex_ante_cost_bps"] = np.asarray(cost_list, dtype=np.float64)
    out["edge_after_hurdle_bps"] = np.asarray(edge_list, dtype=np.float64)
    # L-1: triple_barrier_label = raw result (TP reached first=1, else=0)
    #      barrier_first_label  = cost-conditioned (TP AND net_edge>0)
    out["triple_barrier_label"] = np.asarray(raw_barrier_label_list, dtype=np.int8)
    out["time_to_exit_bars"] = np.asarray(tte_list, dtype=np.int32)
    out["mae_bps"] = np.asarray(mae_list, dtype=np.float64)
    out["mfe_bps"] = np.asarray(mfe_list, dtype=np.float64)
    out["sl_thr_bps"] = np.asarray(sl_thr_bps_list, dtype=np.float64)  # RC3: stop-loss threshold in bps
    out["realized_vol_bps"] = np.asarray(rv_list, dtype=np.float64)
    out["exit_reason"] = np.asarray(exit_reason_list, dtype=object)
    out["exit_idx"] = np.asarray(exit_idx_list, dtype=np.int32)
    out["exit_policy_version"] = np.asarray(exit_policy_version_list, dtype=object)
    out["same_bar_collision"] = np.asarray(same_bar_collision_list, dtype=np.int8)
    min_risk_unit_bps = float(getattr(cfg, "min_risk_unit_bps", 25.0))
    risk_unit_bps = np.maximum(
        pd.to_numeric(out["sl_thr_bps"], errors="coerce").to_numpy(dtype=np.float64),
        min_risk_unit_bps,
    )
    net_event_bps = pd.to_numeric(out["net_event_bps"], errors="coerce").to_numpy(dtype=np.float64)
    mae_bps_arr = pd.to_numeric(out["mae_bps"], errors="coerce").to_numpy(dtype=np.float64)
    mfe_bps_arr = pd.to_numeric(out["mfe_bps"], errors="coerce").to_numpy(dtype=np.float64)
    out["risk_unit_bps"] = risk_unit_bps
    out["net_return_r"] = net_event_bps / _BPS_SCALE
    out["mae_r"] = mae_bps_arr / _BPS_SCALE
    out["mfe_r"] = mfe_bps_arr / _BPS_SCALE

    _barrier_labels = np.asarray(raw_barrier_label_list, dtype=np.int8)
    _profitable_labels = np.asarray(profitable_label_list, dtype=np.int8)
    _edge = np.asarray(edge_list, dtype=np.float64)
    _finite_edge = _edge[np.isfinite(_edge)]
    _gross_arr = np.asarray(gross_list, dtype=np.float64)
    _gross_direction_labels = (_gross_arr > 0.0).astype(np.int8)
    _barrier_label1_rate = float(_barrier_labels.mean()) if len(_barrier_labels) > 0 else 0.0
    _gate_label1_rate = float(_profitable_labels.mean()) if len(_profitable_labels) > 0 else 0.0
    _gross_dir_rate = float(_gross_direction_labels.mean()) if len(_gross_direction_labels) > 0 else 0.0
    _logger.debug(
        "[DIAG][LABEL] events=%d barrier_label1_rate=%.3f gate_label1_rate=%.3f "
        "gross_dir_label1_rate=%.3f mean_edge=%.1f median_edge=%.1f "
        "pct_edge_pos=%.3f p10_edge=%.1f p90_edge=%.1f",
        len(_barrier_labels),
        _barrier_label1_rate,
        _gate_label1_rate,
        _gross_dir_rate,
        float(np.mean(_finite_edge)) if len(_finite_edge) > 0 else float("nan"),
        float(np.median(_finite_edge)) if len(_finite_edge) > 0 else float("nan"),
        float((_finite_edge > 0).mean()) if len(_finite_edge) > 0 else 0.0,
        float(np.percentile(_finite_edge, 10)) if len(_finite_edge) > 0 else float("nan"),
        float(np.percentile(_finite_edge, 90)) if len(_finite_edge) > 0 else float("nan"),
    )
    return out
