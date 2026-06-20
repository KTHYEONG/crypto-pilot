from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from numba import njit

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.execution_cost import ExecutionCostModel

_BPS_SCALE = 1e4
_ATR_PERIOD = 14
_ATR_FALLBACK_FRACTION = 0.01  # L-4: fallback when ATR unavailable; was inline magic number
_EXIT_POLICY_VERSION = "candidate_label_atr_v2"
_BARRIER_EXIT_REASON_MAP: dict[int, str] = {
    0: "invalid",
    1: "stop_loss",
    2: "take_profit",
    3: "time_exit",
}
_logger = logging.getLogger(__name__)


def _rolling_mean_2d(values: np.ndarray, window: int) -> np.ndarray:
    return np.asarray(
        pd.DataFrame(values).rolling(window=window, min_periods=1).mean().to_numpy(),
        dtype=np.float64,
    )


def _rolling_var_2d(values: np.ndarray, window: int) -> np.ndarray:
    return np.asarray(
        pd.DataFrame(values)
        .rolling(window=window, min_periods=2)
        .var(ddof=0)
        .fillna(0.0)
        .to_numpy(),
        dtype=np.float64,
    )


def _compute_yang_zhang_vol_2d(aligned: AlignedMarketData, period: int = _ATR_PERIOD) -> np.ndarray:
    open_ = np.maximum(aligned.open_2d, 1e-12)
    high = np.maximum(aligned.high_2d, 1e-12)
    low = np.maximum(aligned.low_2d, 1e-12)
    close = np.maximum(aligned.close_2d, 1e-12)
    prev_close = np.vstack([close[:1], close[:-1]])

    log_oo = np.log(open_ / prev_close)
    log_cc = np.log(close / open_)
    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)
    rs = log_ho * (log_ho - log_cc) + log_lo * (log_lo - log_cc)

    k = 0.34 / (1.34 + (period + 1.0) / max(period - 1.0, 1.0))
    sigma2_oo = _rolling_var_2d(log_oo, period)
    sigma2_cc = _rolling_var_2d(log_cc, period)
    sigma2_rs = _rolling_mean_2d(rs, period)
    yz_var = np.maximum(sigma2_oo + k * sigma2_cc + (1.0 - k) * sigma2_rs, 0.0)
    return close * np.sqrt(yz_var)


def _scan_barriers_vectorized(
    *,
    high: np.ndarray,
    low: np.ndarray,
    entry_price: float,
    side: int,
    tp_thr: float,
    sl_thr: float,
    scan_from: int,
) -> tuple[int, int, str, int]:
    """Return exit offset, barrier label, exit reason, and same-bar collision flag."""
    if side > 0:
        fav = (high / entry_price) - 1.0
        adv = (low / entry_price) - 1.0
    else:
        fav = (entry_price / np.maximum(low, 1e-12)) - 1.0
        adv = (entry_price / np.maximum(high, 1e-12)) - 1.0

    fav_scan = fav[scan_from:]
    adv_scan = adv[scan_from:]
    tp_hit_mask = np.isfinite(fav_scan) & (fav_scan >= tp_thr)
    sl_hit_mask = np.isfinite(adv_scan) & (adv_scan <= -sl_thr)
    tp_i = int(np.argmax(tp_hit_mask)) + scan_from if bool(tp_hit_mask.any()) else math.inf
    sl_i = int(np.argmax(sl_hit_mask)) + scan_from if bool(sl_hit_mask.any()) else math.inf

    same_bar_collision = int(np.isfinite(tp_i) and np.isfinite(sl_i) and tp_i == sl_i)
    if np.isfinite(sl_i) and (not np.isfinite(tp_i) or sl_i <= tp_i):
        return int(sl_i), 0, "stop_loss", same_bar_collision
    if np.isfinite(tp_i):
        return int(tp_i), 1, "take_profit", same_bar_collision
    return int(high.shape[0] - 1), 0, "time_exit", same_bar_collision


def _find_symbol_index(symbols: tuple[str, ...], symbol: str) -> int:
    for idx, value in enumerate(symbols):
        if value == symbol:
            return idx
    raise KeyError(f"unknown symbol: {symbol}")


def _invalid_label_payload() -> dict[str, float | int | str]:
    return {
        "gross_event_bps": float("nan"),
        "gross_return_bps": float("nan"),
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
        "gross_return_r": float("nan"),
    }


@njit(cache=True)  # type: ignore[untyped-decorator]
def _label_events_kernel(
    n_events: int,
    entry_idx_arr: np.ndarray,
    side_arr: np.ndarray,
    horizon_arr: np.ndarray,
    stop_mult_arr: np.ndarray,
    tp_mult_arr: np.ndarray,
    min_hold_arr: np.ndarray,
    cost_floor_arr: np.ndarray,
    hurdle_arr: np.ndarray,
    sym_idx_arr: np.ndarray,
    open_2d: np.ndarray,
    high_2d: np.ndarray,
    low_2d: np.ndarray,
    close_2d: np.ndarray,
    funding_2d: np.ndarray,
    atr_2d: np.ndarray,
    cost_2d: np.ndarray,
    has_cost_2d: bool,
    taker_round_trip_bps: float,
    atr_fallback_fraction: float,
    bps_scale: float,
) -> tuple:  # type: ignore[type-arg]
    """Numba-JIT triple-barrier labeling kernel; processes all E events in a single compiled loop."""
    t_len = open_2d.shape[0]
    inf_int = t_len + 2  # sentinel for "no barrier hit"

    out_gross = np.full(n_events, np.nan)
    out_cost = np.full(n_events, np.nan)
    out_funding = np.full(n_events, np.nan)
    out_edge = np.full(n_events, np.nan)
    out_raw_barrier = np.zeros(n_events, dtype=np.int8)
    out_barrier_first = np.zeros(n_events, dtype=np.int8)
    out_profitable = np.zeros(n_events, dtype=np.int8)
    out_tte = np.zeros(n_events, dtype=np.int32)
    out_mae = np.full(n_events, np.nan)
    out_mfe = np.full(n_events, np.nan)
    out_rv = np.full(n_events, np.nan)
    out_sl_thr_bps = np.full(n_events, np.nan)
    out_exit_code = np.zeros(n_events, dtype=np.int8)  # 0=invalid,1=sl,2=tp,3=time
    out_exit_idx = np.full(n_events, -1, dtype=np.int32)
    out_same_bar = np.zeros(n_events, dtype=np.int8)

    for i in range(n_events):
        entry_i = int(entry_idx_arr[i])
        side_i = int(side_arr[i])
        horizon_i = int(horizon_arr[i])
        stop_mult_i = stop_mult_arr[i]
        tp_mult_i = tp_mult_arr[i]
        scan_from = int(min_hold_arr[i])
        hurdle_i = hurdle_arr[i]
        sym_i = int(sym_idx_arr[i])

        decision_i = entry_i - 1
        next_open_i = entry_i + horizon_i
        exit_limit = min(entry_i + horizon_i - 1, t_len - 1)

        # invalid: out of bounds
        if decision_i < 0 or entry_i >= t_len:
            continue  # out_exit_code[i]=0 (invalid)

        entry_px = open_2d[entry_i, sym_i]
        if not np.isfinite(entry_px) or entry_px <= 0.0:
            continue

        # ATR for barrier levels
        atr_i = atr_2d[decision_i, sym_i]
        if not np.isfinite(atr_i) or atr_i <= 0.0:
            atr_i = entry_px * atr_fallback_fraction
        tp_thr = (tp_mult_i * atr_i) / entry_px
        sl_thr = (stop_mult_i * atr_i) / entry_px
        out_sl_thr_bps[i] = sl_thr * bps_scale

        path_len = exit_limit - entry_i + 1

        # barrier scan: find first TP and SL hit offsets
        tp_off = inf_int
        sl_off = inf_int
        for off in range(scan_from, path_len):
            bar_i = entry_i + off
            h = high_2d[bar_i, sym_i]
            lo = low_2d[bar_i, sym_i]
            if not np.isfinite(h) or not np.isfinite(lo):
                continue
            if side_i > 0:
                fav_off = h / entry_px - 1.0
                adv_off = lo / entry_px - 1.0
            else:
                fav_off = entry_px / max(lo, 1e-12) - 1.0
                adv_off = entry_px / max(h, 1e-12) - 1.0
            if fav_off >= tp_thr and tp_off == inf_int:
                tp_off = off
            if adv_off <= -sl_thr and sl_off == inf_int:
                sl_off = off
            if tp_off != inf_int and sl_off != inf_int:
                break

        same_bar_i = np.int8(1) if (tp_off < inf_int and sl_off < inf_int and tp_off == sl_off) else np.int8(0)

        # determine exit
        if sl_off < inf_int and (tp_off >= inf_int or sl_off <= tp_off):
            exit_off = sl_off
            barrier_label = 0
            exit_code = 1  # stop_loss
        elif tp_off < inf_int:
            exit_off = tp_off
            barrier_label = 1
            exit_code = 2  # take_profit
        else:
            exit_off = path_len - 1
            barrier_label = 0
            exit_code = 3  # time_exit

        # time_exit: next open out of bounds → invalid
        # out_sl_thr_bps[i] was set above; revert to invalid payload (NaN init)
        if exit_code == 3 and next_open_i >= t_len:
            out_sl_thr_bps[i] = np.nan
            continue

        # exit price
        if exit_code == 2:
            exit_px = entry_px * (1.0 + side_i * tp_thr)
        elif exit_code == 1:
            exit_px = entry_px * (1.0 - side_i * sl_thr)
        else:
            exit_px = open_2d[next_open_i, sym_i]

        # gross return in bps
        if side_i > 0:
            gross_bps = (exit_px / entry_px - 1.0) * bps_scale
        else:
            gross_bps = (entry_px / max(exit_px, 1e-12) - 1.0) * bps_scale

        # execution cost: cost_floor → cost_2d → taker_round_trip_bps
        cost_grid = cost_2d[decision_i, sym_i] if has_cost_2d else taker_round_trip_bps
        cost_floor_i = cost_floor_arr[i]
        ex_ante_cost = cost_floor_i if np.isfinite(cost_floor_i) else cost_grid
        ex_ante_cost = max(ex_ante_cost, taker_round_trip_bps)

        # realized funding
        funding_stop = next_open_i if exit_code == 3 else min(entry_i + exit_off + 1, t_len)
        realized_funding = 0.0
        for fi in range(entry_i, funding_stop):
            fv = funding_2d[fi, sym_i]
            if np.isfinite(fv):
                realized_funding += fv * side_i * bps_scale

        edge = gross_bps - ex_ante_cost - realized_funding - hurdle_i

        # mae/mfe on path returns
        path_end = exit_off + 1
        close_path = close_2d[entry_i : entry_i + path_end, sym_i]
        path_r = close_path / entry_px - 1.0 if side_i > 0 else entry_px / np.maximum(close_path, 1e-12) - 1.0
        finite_mask = np.isfinite(path_r)
        if finite_mask.any():
            valid_r = path_r[finite_mask]
            out_mae[i] = np.min(valid_r) * bps_scale
            out_mfe[i] = np.max(valid_r) * bps_scale

        # realized vol: std(diff(log(close_path)))
        if exit_off >= 1:
            log_c = np.log(np.maximum(close_path, 1e-12))
            diffs = np.diff(log_c)
            out_rv[i] = np.std(diffs) * bps_scale
        else:
            out_rv[i] = 0.0

        ei = next_open_i if exit_code == 3 else entry_i + exit_off

        out_gross[i] = gross_bps
        out_cost[i] = ex_ante_cost
        out_funding[i] = realized_funding
        out_edge[i] = edge
        out_raw_barrier[i] = np.int8(barrier_label)
        out_barrier_first[i] = np.int8(1) if (barrier_label == 1 and edge > 0.0) else np.int8(0)
        out_profitable[i] = np.int8(1) if edge > 0.0 else np.int8(0)
        out_tte[i] = np.int32(horizon_i if exit_code == 3 else exit_off + 1)
        out_exit_code[i] = np.int8(exit_code)
        out_exit_idx[i] = np.int32(ei)
        out_same_bar[i] = same_bar_i

    return (
        out_gross, out_cost, out_funding, out_edge,
        out_raw_barrier, out_barrier_first, out_profitable,
        out_tte, out_mae, out_mfe, out_rv, out_sl_thr_bps,
        out_exit_code, out_exit_idx, out_same_bar,
    )


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

    atr_2d = _compute_yang_zhang_vol_2d(aligned)
    out = events.copy()
    # O(1) symbol lookup: build once before the per-event pre-extraction
    _sym_to_idx: dict[str, int] = {sym: idx for idx, sym in enumerate(aligned.symbols)}
    cost_model = ExecutionCostModel(
        maker_fee_bps=float(getattr(cfg, "maker_fee_bps", 2.0)),
        taker_fee_bps=float(getattr(cfg, "taker_fee_bps", 5.0)),
        maker_ratio=float(getattr(cfg, "maker_ratio", 0.75)),
        slippage_bps=float(getattr(cfg, "slippage_bps", 1.0)),
        impact_coeff_bps=float(getattr(cfg, "impact_coeff_bps", 0.0)),
        stress_multiplier=float(getattr(cfg, "cost_stress_multiplier", 1.5)),
    )
    taker_round_trip_bps = cost_model.taker_round_trip_bps()

    n = len(out)

    # Pre-extract event arrays for numba kernel
    sym_idx_arr = np.array([_sym_to_idx.get(str(s), -1) for s in out["symbol"]], dtype=np.int64)
    unknown_pos = np.where(sym_idx_arr == -1)[0]
    if unknown_pos.size > 0:
        raise KeyError(f"unknown symbol: {out['symbol'].iloc[unknown_pos[0]]}")

    entry_idx_arr = out["entry_idx"].to_numpy(dtype=np.int64)
    side_arr = out["side"].to_numpy(dtype=np.int64)
    horizon_arr = np.maximum(
        pd.to_numeric(out["expected_holding_bars"], errors="coerce").fillna(1).to_numpy(dtype=np.int64), 1
    )
    stop_mult_arr = out["stop_atr_mult"].to_numpy(dtype=np.float64)
    tp_mult_arr = out["take_profit_atr_mult"].to_numpy(dtype=np.float64)

    if cfg.exit_policy_mode == "engine_aligned" and "min_holding_bars" in out.columns:
        min_hold_arr = np.maximum(
            pd.to_numeric(out["min_holding_bars"], errors="coerce").fillna(0).to_numpy(dtype=np.int64), 0
        )
    else:
        min_hold_arr = np.zeros(n, dtype=np.int64)

    cost_floor_arr = (
        pd.to_numeric(out["cost_floor_bps"], errors="coerce").to_numpy(dtype=np.float64)
        if "cost_floor_bps" in out.columns
        else np.full(n, np.nan)
    )
    hurdle_arr = (
        pd.to_numeric(out["hurdle_bps"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if "hurdle_bps" in out.columns
        else np.zeros(n, dtype=np.float64)
    )

    open_2d_c = np.ascontiguousarray(aligned.open_2d, dtype=np.float64)
    high_2d_c = np.ascontiguousarray(aligned.high_2d, dtype=np.float64)
    low_2d_c = np.ascontiguousarray(aligned.low_2d, dtype=np.float64)
    close_2d_c = np.ascontiguousarray(aligned.close_2d, dtype=np.float64)
    funding_2d_c = np.ascontiguousarray(aligned.funding_2d, dtype=np.float64)
    atr_2d_c = np.ascontiguousarray(atr_2d, dtype=np.float64)

    has_cost_2d = aligned.execution_cost_bps_2d is not None
    cost_2d_c = (
        np.ascontiguousarray(aligned.execution_cost_bps_2d, dtype=np.float64)
        if has_cost_2d
        else np.zeros_like(close_2d_c)
    )

    (
        gross_arr, cost_arr, funding_arr, edge_arr,
        raw_barrier_arr, barrier_first_arr, profitable_arr,
        tte_arr, mae_arr, mfe_arr, rv_arr, sl_thr_bps_arr,
        exit_code_arr, exit_idx_arr, same_bar_arr,
    ) = _label_events_kernel(
        n,
        entry_idx_arr, side_arr, horizon_arr,
        stop_mult_arr, tp_mult_arr, min_hold_arr,
        cost_floor_arr, hurdle_arr, sym_idx_arr,
        open_2d_c, high_2d_c, low_2d_c, close_2d_c, funding_2d_c,
        atr_2d_c, cost_2d_c, has_cost_2d,
        taker_round_trip_bps,
        float(_ATR_FALLBACK_FRACTION),
        float(_BPS_SCALE),
    )

    exit_reason_arr = np.array(
        [_BARRIER_EXIT_REASON_MAP[int(c)] for c in exit_code_arr], dtype=object
    )

    # Column assignments — same schema as the original itertuples version
    out["barrier_first_label"] = barrier_first_arr.astype(np.int8)
    out["profitable_after_hurdle_label"] = profitable_arr.astype(np.int8)
    out["event_id"] = np.arange(len(out), dtype=np.int64)
    out["gross_event_bps"] = gross_arr
    out["gross_return_bps"] = gross_arr.copy()
    out["gross_fwd_bps"] = gross_arr.copy()
    out["gross_direction_label"] = (gross_arr > 0.0).astype(np.int8)
    out["execution_cost_bps"] = cost_arr
    out["realized_funding_bps"] = funding_arr
    out["net_event_bps"] = edge_arr
    out["ex_ante_cost_bps"] = cost_arr.copy()
    out["edge_after_hurdle_bps"] = edge_arr.copy()
    out["triple_barrier_label"] = raw_barrier_arr.astype(np.int8)
    out["time_to_exit_bars"] = tte_arr.astype(np.int32)
    out["mae_bps"] = mae_arr
    out["mfe_bps"] = mfe_arr
    out["sl_thr_bps"] = sl_thr_bps_arr
    out["realized_vol_bps"] = rv_arr
    out["exit_reason"] = exit_reason_arr
    out["exit_idx"] = exit_idx_arr.astype(np.int32)
    out["exit_policy_version"] = np.full(len(out), _EXIT_POLICY_VERSION, dtype=object)
    out["same_bar_collision"] = same_bar_arr.astype(np.int8)

    # Risk-adjusted returns
    min_risk_unit_bps = float(getattr(cfg, "min_risk_unit_bps", 25.0))
    risk_unit_bps = np.maximum(sl_thr_bps_arr, min_risk_unit_bps)
    out["risk_unit_bps"] = risk_unit_bps
    safe_risk_unit = np.maximum(risk_unit_bps, 1.0)
    out["gross_return_r"] = gross_arr / safe_risk_unit
    out["net_return_r"] = edge_arr / safe_risk_unit
    out["mae_r"] = mae_arr / safe_risk_unit
    out["mfe_r"] = mfe_arr / safe_risk_unit

    # Diagnostic log (same metrics as original, derived from kernel output arrays)
    _finite_edge = edge_arr[np.isfinite(edge_arr)]
    _gross_dir = (gross_arr > 0.0).astype(np.int8)
    _logger.debug(
        "[DIAG][LABEL] events=%d barrier_label1_rate=%.3f gate_label1_rate=%.3f "
        "gross_dir_label1_rate=%.3f mean_edge=%.1f median_edge=%.1f "
        "pct_edge_pos=%.3f p10_edge=%.1f p90_edge=%.1f",
        n,
        float(raw_barrier_arr.mean()) if n > 0 else 0.0,
        float(profitable_arr.mean()) if n > 0 else 0.0,
        float(_gross_dir.mean()) if n > 0 else 0.0,
        float(np.mean(_finite_edge)) if _finite_edge.size > 0 else float("nan"),
        float(np.median(_finite_edge)) if _finite_edge.size > 0 else float("nan"),
        float((_finite_edge > 0).mean()) if _finite_edge.size > 0 else 0.0,
        float(np.percentile(_finite_edge, 10)) if _finite_edge.size > 0 else float("nan"),
        float(np.percentile(_finite_edge, 90)) if _finite_edge.size > 0 else float("nan"),
    )
    return out
