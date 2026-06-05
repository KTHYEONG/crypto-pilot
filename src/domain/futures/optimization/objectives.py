from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import optuna
from numpy.typing import NDArray

from src.core.settings import (
    FUTURES_INITIAL_BALANCE,
    MAKER_FEE_RATE,
    SLIPPAGE_RATE,
    TAKER_FEE_RATE,
)
from src.domain.futures.backtest.engine import (
    backtest_target_weights_intrabar_numba,
    backtest_target_weights_numba,
    hours_per_bar_from_timeframe,
    max_hold_bars_from_time_barrier,
)
from src.domain.futures.backtest.preparation import prepare_backtest_inputs
from src.domain.futures.optimization.common import (
    _safe_float_or_none,
    _weight_stage_diag,
)
from src.domain.futures.optimization.evaluator import (
    _log_tw_from_ret_pct,
    calc_cvar5_loss_pct_from_equity,
    calc_gate1_dsr_from_path_log_tw,
    calc_max_underwater_days_from_equity,
    calc_mdd_from_equity,
    compute_v3_score,
)
from src.domain.futures.optimization.ml_context import (
    merge_membership_constraints_into_aligned,
    precompute_ml_optimization_context,
    rerun_precompute_for_ctx,
)
from src.domain.futures.optimization.observability.trial_observability import set_trial_event_attrs
from src.domain.futures.optimization.opt_config import (
    OPT_FUTURES_CONFIG,
    default_ev_hurdle_bps,
)
from src.domain.futures.portfolio.friction_model import CostSnapshot

if TYPE_CHECKING:
    from src.domain.futures.optimization.ml_context import MLPhaseDContext

_logger = logging.getLogger(__name__)


def _build_strategy_compose_diag(
    *,
    alpha_long: np.ndarray,
    alpha_short: np.ndarray,
    xs_long: np.ndarray,
    xs_short: np.ndarray,
    params: dict[str, Any],
    cost_snapshot: CostSnapshot,
    holding_bars: int | None = None,
) -> dict[str, float]:
    from src.domain.futures.optimization.diag_utils import preservation_ratio

    n_bars = alpha_long.shape[0]
    beta_a = float(
        params.get("BETA_ALPHA", OPT_FUTURES_CONFIG.get("FUTURES_DEFAULT_BETA_ALPHA", 1.0))
    )
    ev_h = float(
        params.get("EV_HURDLE_BPS", default_ev_hurdle_bps(OPT_FUTURES_CONFIG))
    )
    friction_2d = np.asarray(cost_snapshot.execution_cost_fraction_2d, dtype=np.float64)
    friction_bps_2d = np.asarray(cost_snapshot.execution_cost_bps_2d, dtype=np.float64)

    # Delegate cost blend to ExecutionCostModel (SSOT)
    from src.domain.futures.strategy.execution_cost import ExecutionCostModel
    _cost_model = ExecutionCostModel(
        maker_fee_bps=float(params.get("MAKER_FEE_BPS", 2.0)),
        taker_fee_bps=float(params.get("TAKER_FEE_BPS", 5.0)),
        maker_ratio=float(params.get("MAKER_RATIO", 0.20)),
        slippage_bps=float(params.get("SLIPPAGE_BPS", 2.0)),
    )
    effective_rt = _cost_model.round_trip_bps() / 10000.0
    taker_fee = float(params.get("TAKER_FEE_BPS", 5.0))
    slippage = float(params.get("SLIPPAGE_BPS", 2.0))
    baseline_rt = (taker_fee + slippage) * 2.0 / 10000.0
    if baseline_rt > 1e-12:
        scale = effective_rt / baseline_rt
        friction_2d = friction_2d * scale
        friction_bps_2d = friction_bps_2d * scale

    effective_friction_2d = friction_2d
    if params.get("COST_GATE_AMORTIZE", False) and holding_bars and int(holding_bars) > 1:
        effective_friction_2d = friction_2d / float(int(holding_bars))
    effective_friction_bps_2d = effective_friction_2d * 10000.0
    threshold_bps = effective_friction_bps_2d + ev_h
    alpha_p95_bps = max(
        _safe_pct(alpha_long, 95) * 10000.0,
        _safe_pct(alpha_short, 95) * 10000.0,
    )
    _logger.info(
        "[ML-COST-WALL] alpha_p95=%.2fbps friction=%.1fbps "
        "hurdle_bps=%.1fbps floor=%.1fbps signal_clears_floor=%s",
        alpha_p95_bps,
        float(np.nanmean(effective_friction_bps_2d)),
        ev_h,
        float(np.nanmean(threshold_bps)),
        str(alpha_p95_bps >= float(np.nanmean(threshold_bps))),
    )

    mu_l_pre = beta_a * alpha_long - effective_friction_2d
    mu_s_pre = beta_a * alpha_short - effective_friction_2d

    cost_source = str(cost_snapshot.execution_cost_bps_source)
    return {
        "bars": float(n_bars),
        "alpha_long_nz_ratio": _nonzero_ratio(alpha_long),
        "alpha_short_nz_ratio": _nonzero_ratio(alpha_short),
        "alpha_long_p50": _safe_pct(alpha_long, 50),
        "alpha_long_p95": _safe_pct(alpha_long, 95),
        "alpha_long_p99": _safe_pct(alpha_long, 99),
        "alpha_short_p50": _safe_pct(alpha_short, 50),
        "alpha_short_p95": _safe_pct(alpha_short, 95),
        "alpha_short_p99": _safe_pct(alpha_short, 99),
        "friction_bps": float(np.nanmean(friction_bps_2d)),
        "raw_friction_bps": float(np.nanmean(friction_bps_2d)),
        "effective_friction_bps": float(np.nanmean(effective_friction_bps_2d)),
        "ev_hurdle_bps": float(ev_h),
        "effective_threshold_bps": float(np.nanmean(threshold_bps)),
        "cost_gate_amortized": float(
            1.0
            if params.get("COST_GATE_AMORTIZE", False)
            and holding_bars
            and int(holding_bars) > 1
            else 0.0
        ),
        "holding_bars": float(int(holding_bars) if holding_bars is not None else 1),
        "execution_cost_source_universe_static": (
            1.0 if cost_source == "universe_static" else 0.0
        ),
        "execution_cost_source_parametric_dynamic": (
            1.0 if cost_source == "parametric_dynamic" else 0.0
        ),
        "execution_cost_source_fallback_global": (
            1.0 if cost_source == "fallback_global" else 0.0
        ),
        "mu_pre_hurdle_p95_long": _safe_pct(mu_l_pre, 95),
        "mu_pre_hurdle_p95_short": _safe_pct(mu_s_pre, 95),
        "xs_long_nz_ratio": _nonzero_ratio(xs_long),
        "xs_short_nz_ratio": _nonzero_ratio(xs_short),
        "xs_long_preservation_ratio": preservation_ratio(alpha_long, xs_long),
        "xs_short_preservation_ratio": preservation_ratio(alpha_short, xs_short),
        "mu_long_mean": float(np.mean(mu_l_pre)) if mu_l_pre.size > 0 else 0.0,
        "mu_short_mean": float(np.mean(mu_s_pre)) if mu_s_pre.size > 0 else 0.0,
        "threshold_bps": float(np.nanmean(threshold_bps)),
    }

def _safe_pct(arr: np.ndarray, q: float) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.nanpercentile(finite, q))

def _nonzero_ratio(arr: np.ndarray, eps: float = 1e-12) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(arr) > eps) / arr.size)


def _pf_and_ev_cost_from_trades(all_trades: np.ndarray) -> tuple[float, float]:
    """PF = gross_win / |gross_loss|; EV/cost = |sum(pnl)| / sum(entry_fee + funding_fee)."""
    if all_trades.size == 0:
        return 1.0, 0.0
    pnl: np.ndarray = all_trades[:, 6].astype(np.float64, copy=False)
    gross_win = float(np.sum(pnl[pnl > 0.0]))
    gross_loss = float(np.sum(np.abs(pnl[pnl < 0.0])))
    avg_pf = gross_win / max(abs(gross_loss), 1e-9) if gross_loss != 0.0 else 1.0
    net_pnl = float(np.sum(pnl))
    fees = all_trades[:, 8].astype(np.float64, copy=False) + all_trades[:, 9].astype(
        np.float64, copy=False
    )
    total_fee = float(np.sum(fees))
    ev_cost_ratio = abs(net_pnl) / max(total_fee, 1e-9)
    return avg_pf, ev_cost_ratio


def _funding_drag_ratio_from_trades(all_trades: np.ndarray) -> tuple[float, str]:
    """Funding drag ratio using a conservative gross-PnL basis."""
    if all_trades.size == 0:
        return 0.0, "funding_fee_abs_over_gross_pnl_abs"
    pnl: NDArray[np.float64] = all_trades[:, 6].astype(np.float64, copy=False)
    funding_fee: NDArray[np.float64] = all_trades[:, 9].astype(np.float64, copy=False)
    gross_pnl_abs = float(np.sum(np.abs(pnl)))
    funding_abs = float(np.sum(np.abs(funding_fee)))
    ratio = float(funding_abs / max(gross_pnl_abs, 1e-9))
    return ratio, "funding_fee_abs_over_gross_pnl_abs"


def _base_engine_params(ml: dict[str, Any], tf: str) -> dict[str, Any]:
    ann_vol = float(ml.get("TARGET_ANN_VOL", 0.20))
    kelly_lambda = float(ml.get("KELLY_LAMBDA", 0.20))
    lev = float(OPT_FUTURES_CONFIG.get("FUTURES_DISCOVERY_LEVERAGE", 5))
    cfg = OPT_FUTURES_CONFIG
    atm = float(ml.get("ATR_MULT", cfg.get("FUTURES_ATR_STOP_MULT", 2.5)))
    atr_period_fixed = int(cfg.get("FUTURES_ATR_PERIOD_FIXED", 30))

    return {
        "TIMEFRAME": tf,
        "SIGNAL_TYPE": "ML_CALIB_PROB",
        "REGIME_TYPE": "EMA_ATR",
        "SIZING_METHOD": "profit_factor_kelly",
        "USE_CS_RANK_ENGINE": False,
        "K_LONG": int(ml.get("K_LONG", 2)),
        "K_SHORT": int(ml.get("K_SHORT", 2)),
        "REBALANCE_BARS": int(ml.get("REBALANCE_BARS", 1)),
        "REBALANCE_TURNOVER_THRESHOLD": float(ml.get("REBALANCE_TURNOVER_THRESHOLD", 0.15)),
        "MIN_SCORE_PERCENTILE": float(ml.get("MIN_SCORE_PERCENTILE", 0.50)),
        "CRISIS_GAMMA": float(ml.get("CRISIS_GAMMA", 1.0)),
        "TRAIL_MULT": float(ml.get("TRAIL_MULT", atm)),
        "ATR_MULT": atm,
        "ATR_PERIOD": int(ml.get("ATR_PERIOD", atr_period_fixed)),
        "SHORT_TP_MULT": float(ml.get("SHORT_TP_MULT", 2.0)),
        "LONG_SCALE_ATR_MULT": float(ml.get("LONG_SCALE_ATR_MULT", 3.0)),
        "RISK_PER_TRADE": kelly_lambda,
        "MAX_EXPOSURE_PER_COIN": float(ml.get("MAX_EXPOSURE_PER_COIN", 1.0)),
        "MAX_EXPOSURE": float(ml.get("MAX_EXPOSURE", 1.0)),
        "DD_SCALING_THRESHOLD": float(ml.get("DD_SCALING_THRESHOLD", 0.0)),
        "CS_Z_SCORE_THRESHOLD": float(ml.get("CS_Z_SCORE_THRESHOLD", 1.0)),
        "LONG_CS_Z_ENTRY": float(ml.get("LONG_CS_Z_ENTRY", ml.get("CS_Z_SCORE_THRESHOLD", 1.0))),
        "SHORT_CS_Z_ENTRY": float(ml.get("SHORT_CS_Z_ENTRY", ml.get("CS_Z_SCORE_THRESHOLD", 1.0))),
        "HYSTERESIS_GAP": float(ml.get("HYSTERESIS_GAP", 0.3)),
        "DYNAMIC_RA_CRISIS_COEF": float(ml.get("DYNAMIC_RA_CRISIS_COEF", 3.0)),
        "DYNAMIC_RA_BEAR_COEF": float(ml.get("DYNAMIC_RA_BEAR_COEF", 1.5)),
        "NORM_VAR_CONSTANT": float(ml.get("NORM_VAR_CONSTANT", 0.5)),
        "CRISIS_LONG_Z_BOOST": float(ml.get("CRISIS_LONG_Z_BOOST", 0.0)),
        "CRISIS_LONG_MAG_SUPPRESS": float(
            ml.get(
                "CRISIS_LONG_MAG_SUPPRESS",
                cfg.get("FUTURES_CRISIS_LONG_MAG_SUPPRESS", 1.0),
            )
        ),
        "TARGET_ANN_VOL": ann_vol,
        "KELLY_LAMBDA": kelly_lambda,
        "USE_COMPOUNDING": True,
        "LEVERAGE": int(lev),
        "BETA_ALPHA": float(ml.get("BETA_ALPHA", cfg.get("FUTURES_DEFAULT_BETA_ALPHA", 1.0))),
        "EV_HURDLE_BPS": float(
            ml.get("EV_HURDLE_BPS", default_ev_hurdle_bps(cfg))
        ),
        "COST_GATE_AMORTIZE": bool(
            ml.get("COST_GATE_AMORTIZE", cfg.get("COST_GATE_AMORTIZE", False))
        ),
        "KELLY_USE_RESIDUAL_VAR": bool(
            ml.get("KELLY_USE_RESIDUAL_VAR", cfg.get("KELLY_USE_RESIDUAL_VAR", False))
        ),
        "COST_FORECAST_DYNAMIC": bool(
            ml.get("COST_FORECAST_DYNAMIC", cfg.get("COST_FORECAST_DYNAMIC", False))
        ),
        "SLIPPAGE_BPS_BUFFER_MULT": float(
            ml.get("SLIPPAGE_BPS_BUFFER_MULT", cfg.get("SLIPPAGE_BPS_BUFFER_MULT", 1.0))
        ),
        "TIME_BARRIER_H": float(
            ml.get("TIME_BARRIER_H", cfg.get("FUTURES_DEFAULT_TIME_BARRIER_H", 0.0))
        ),
        "PORTFOLIO_KAPPA": float(
            ml.get("PORTFOLIO_KAPPA", cfg.get("FUTURES_PORTFOLIO_KAPPA", 0.35))
        ),
        "FUTURES_EXECUTION_MODE": str(
            ml.get("FUTURES_EXECUTION_MODE")
            or OPT_FUTURES_CONFIG.get("FUTURES_EXECUTION_MODE", "coarse")
        ),
        "STRATEGY_MODE": bool(ml.get("STRATEGY_MODE", False)),
        "POST_COST_ADMISSION_MODE": str(
            ml.get("POST_COST_ADMISSION_MODE", ml.get("post_cost_admission_mode", "ev_gate"))
        ),
        "RANK_PORTFOLIO_TOP_K": int(
            ml.get("RANK_PORTFOLIO_TOP_K", ml.get("rank_portfolio_top_k", 4))
        ),
        "RANK_PORTFOLIO_MIN_SCORE_SPREAD_BPS": float(
            ml.get(
                "RANK_PORTFOLIO_MIN_SCORE_SPREAD_BPS",
                ml.get("rank_portfolio_min_score_spread_bps", 0.0),
            )
        ),
        "MAKER_RATIO": float(
            ml.get("MAKER_RATIO", ml.get("maker_ratio", 0.20))
        ),
        "MAKER_FEE_BPS": float(
            ml.get("MAKER_FEE_BPS", ml.get("maker_fee_bps", 2.0))
        ),
        "TAKER_FEE_BPS": float(
            ml.get("TAKER_FEE_BPS", ml.get("taker_fee_bps", 5.0))
        ),
        "SLIPPAGE_BPS": float(
            ml.get("SLIPPAGE_BPS", ml.get("slippage_bps", 2.0))
        ),
    }


def _cached_kill_fund_lev(
    aligned: dict[str, Any], params: dict[str, Any]
) -> tuple[Any, Any, Any]:
    if "kill_signal_cached" not in aligned:
        zkill = aligned.get("effective_kill_signal")
        if zkill is None:
            zkill = aligned.get("kill_signal")
        if zkill is None:
            zkill = np.zeros_like(aligned["close"])
        aligned["kill_signal_cached"] = zkill
    zkill = aligned["kill_signal_cached"]
    if "funding_rate_sum_cached" not in aligned:
        zfund = aligned.get("funding_rate_sum")
        if zfund is None:
            zfund = np.zeros_like(aligned["close"])
        aligned["funding_rate_sum_cached"] = zfund
    zfund = aligned["funding_rate_sum_cached"]
    if "dyn_leverage_cached" not in aligned:
        lev_blk = aligned.get("dyn_leverage")
        if lev_blk is None or lev_blk.shape != aligned["close"].shape:
            lev_blk = np.full_like(aligned["close"], float(params["LEVERAGE"]), dtype=np.float64)
        else:
            lev_blk = np.maximum(lev_blk.astype(np.float64, copy=False), 0.0)
        aligned["dyn_leverage_cached"] = lev_blk
    lev_blk = aligned["dyn_leverage_cached"]
    return zkill, zfund, lev_blk


def _run_portfolio_numba_block(
    params: dict[str, Any],
    aligned: dict[str, Any],
    estimated_b: float = 1.05,
    *,
    trial_number: int | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    _ = estimated_b
    orig_aligned = aligned

    # 1. 먼저 캐시 확인 및 준비 (prepare 비용은 공유 캐시로 절감)
    t0_align = time.perf_counter()
    if "_prepared_cache" in orig_aligned:
        prepared = orig_aligned["_prepared_cache"]
        aligned = cast(dict[str, Any], prepared.aligned_data)
    else:
        prepared = prepare_backtest_inputs(aligned, params)
        aligned = cast(dict[str, Any], prepared.aligned_data)
        merge_membership_constraints_into_aligned(aligned, persist_stats=True)
        orig_aligned["_prepared_cache"] = prepared
    t_prep_align = time.perf_counter() - t0_align

    t0_compose = time.perf_counter()
    tw_from_strategy = aligned.get("target_weights")
    if tw_from_strategy is None or np.asarray(tw_from_strategy).shape != np.asarray(aligned["close"]).shape:
        raise RuntimeError(
            "target_weights required: pre-merge candidate output before calling objectives"
        )
    t_compose = time.perf_counter() - t0_compose

    t0_const = time.perf_counter()
    zkill, zfund, lev_blk = _cached_kill_fund_lev(aligned, params)
    t_prep_constraint = time.perf_counter() - t0_const
    t_prep = t_prep_align + t_prep_constraint

    # 캐싱 기록
    aligned["_prof_compose"] = t_compose
    aligned["_prof_prep"] = t_prep
    aligned["_prof_prep_align"] = t_prep_align
    aligned["_prof_prep_constraint"] = t_prep_constraint
    cfg_block = OPT_FUTURES_CONFIG
    reb_b = max(1, int(params["REBALANCE_BARS"]))

    hpb = hours_per_bar_from_timeframe(str(params.get("TIMEFRAME", "4h")))
    tw_blk = np.asarray(tw_from_strategy, dtype=np.float64)
    entry_block = aligned.get("entry_block_mask")
    if entry_block is not None:
        entry_block_2d = np.asarray(entry_block, dtype=np.float64)
        if entry_block_2d.shape == tw_blk.shape:
            tw_blk = np.where(entry_block_2d > 0.0, 0.0, tw_blk)
            symbol_names = aligned.get("symbol_names")
            if (
                isinstance(symbol_names, np.ndarray)
                and symbol_names.ndim == 1
                and int(symbol_names.shape[0]) == tw_blk.shape[1]
                and not bool(aligned.get("_membership_mask_logged", False))
            ):
                active_mask = np.where(np.abs(tw_blk) > 1e-12, 1.0, 0.0)
                for s_idx, symbol in enumerate(symbol_names):
                    sym_block = entry_block_2d[:, s_idx] > 0.0
                    if not np.any(sym_block):
                        continue
                    _logger.info(
                        "[MEMBERSHIP-MASK] symbol=%s active_ratio=%.4f warm_ratio=%.4f "
                        "forced_exit_count=%d blocked_entry_count=%d",
                        str(symbol),
                        float(np.mean((entry_block_2d[:, s_idx] <= 0.0).astype(np.float64))),
                        float(np.mean((entry_block_2d[:, s_idx] <= 0.0).astype(np.float64))),
                        int(np.count_nonzero(np.asarray(zkill)[:, s_idx] > 0.0)),
                        int(np.count_nonzero(sym_block & (active_mask[:, s_idx] > 0.0))),
                    )
                aligned["_membership_mask_logged"] = True
    aligned["target_weights"] = tw_blk
    _path_diag = aligned.get("_strategy_signal_path_diag")
    if isinstance(_path_diag, dict):
        _path_diag["merge_nz"] = float(
            np.count_nonzero(np.abs(tw_blk) > 1e-12) / max(tw_blk.size, 1)
        )
    if bool(params.get("STRATEGY_MODE", False)):
        should_log_weight = (trial_number is not None and int(trial_number) < 5) or (
            trial_number is None
        )
        if should_log_weight:
            _logger.info(
                " [WEIGHT-STAGE-DIAG] trial=%s %s",
                trial_number if trial_number is not None else "replay",
                _weight_stage_diag(
                    tw_blk,
                    per_symbol_cap=_safe_float_or_none(params.get("MAX_EXPOSURE_PER_COIN")),
                ),
            )
    use_simple_atr_i = (
        1 if bool(cfg_block.get("FUTURES_SIMPLE_ATR_STOP", True)) else 0
    )
    mx_hold = max_hold_bars_from_time_barrier(params)
    sborr = float(cfg_block.get("FUTURES_SHORT_BORROW_DAILY", 0.0))
    buf_mult = float(
        params.get(
            "SLIPPAGE_BPS_BUFFER_MULT",
            OPT_FUTURES_CONFIG.get("SLIPPAGE_BPS_BUFFER_MULT", 1.0),
        )
    )
    slip_eff = float(SLIPPAGE_RATE) * max(buf_mult, 1e-9)
    atr_m = float(params["ATR_MULT"])
    trail_m = float(params["TRAIL_MULT"])

    use_intrabar = (
        prepared.execution_mode == "intrabar_1m"
        and prepared.exec_bar_start_1m_idx is not None
        and prepared.exec_bar_end_1m_idx is not None
        and aligned.get("exec_open_1m") is not None
        and aligned.get("exec_high_1m") is not None
        and aligned.get("exec_low_1m") is not None
        and aligned.get("exec_close_1m") is not None
    )
    t0 = time.perf_counter()
    out_tw: tuple[NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64]]
    if use_intrabar:
        exec_start = prepared.exec_bar_start_1m_idx
        exec_end = prepared.exec_bar_end_1m_idx
        out_tw = cast(
            tuple[NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64]],
            backtest_target_weights_intrabar_numba(
            aligned["close"],
            aligned["high"],
            aligned["low"],
            aligned["open"],
            tw_blk,
            lev_blk,
            aligned["atr"],
            zkill,
            np.asarray(aligned["exec_open_1m"], dtype=np.float64),
            np.asarray(aligned["exec_high_1m"], dtype=np.float64),
            np.asarray(aligned["exec_low_1m"], dtype=np.float64),
            np.asarray(aligned["exec_close_1m"], dtype=np.float64),
            exec_start,
            exec_end,
            float(FUTURES_INITIAL_BALANCE),
            MAKER_FEE_RATE,
            TAKER_FEE_RATE,
            slip_eff,
            reb_b,
            mx_hold,
            sborr,
            atr_m,
            trail_m,
            use_simple_atr_i,
            int(params.get("MAX_CONCURRENT_POSITIONS", 2)),
            float(params.get("MAX_EXPOSURE", 0.8)),
            float(params["MAX_EXPOSURE_PER_COIN"]),
            float(params["DD_SCALING_THRESHOLD"]),
            funding_event_mask_1m=aligned.get("funding_event_mask_1m"),
            funding_rate_1m=aligned.get("funding_rate_event_1m"),
            volume_1m_2d=aligned.get("exec_volume_1m"),
            ),
        )
    else:
        out_tw = cast(
            tuple[NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64]],
            backtest_target_weights_numba(
            aligned["close"],
            aligned["high"],
            aligned["low"],
            aligned["open"],
            zfund,
            zkill,
            tw_blk,
            float(FUTURES_INITIAL_BALANCE),
            lev_blk,
            MAKER_FEE_RATE,
            TAKER_FEE_RATE,
            slip_eff,
            reb_b,
            mx_hold,
            sborr,
            hpb,
            aligned["atr"],
            atr_m,
            trail_m,
            use_simple_atr_i,
            int(params.get("MAX_CONCURRENT_POSITIONS", 2)),
            float(params.get("MAX_EXPOSURE", 0.8)),
            float(params["MAX_EXPOSURE_PER_COIN"]),
            float(params["DD_SCALING_THRESHOLD"]),
            volume_2d=aligned.get("volume"),
            ),
        )
    t_exec = time.perf_counter() - t0
    orig_aligned["_prof_compose"] = t_compose
    orig_aligned["_prof_prep"] = t_prep
    orig_aligned["_prof_exec"] = t_exec
    return out_tw


def _awf_gate_stat_ref_bars(awf_slices: list[dict[str, Any]]) -> int:
    tot = 0
    for leg in awf_slices:
        lr = leg.get("leg_range")
        if isinstance(lr, (tuple, list)) and len(lr) >= 2:
            tot += max(0, int(lr[1]) - int(lr[0]))
    return max(tot, 1)


def replay_robust_awf_for_trial_params(
    ctx: MLPhaseDContext, raw_optuna_params: dict[str, Any]
) -> tuple[float | tuple[float, float], dict[str, Any]]:
    """Replay AWF legs with fixed tuned Optuna param dict."""
    from src.domain.futures.optimization.samplers import build_ml_phase_d_params
    if ctx.awf_leg_slices is None:
        rerun_precompute_for_ctx(ctx)
    merged_full = build_ml_phase_d_params(raw_optuna_params, ctx.tf)
    return _evaluate_awf_phase_d_aggregate(ctx, merged_full, trial=None)


def _evaluate_awf_phase_d_aggregate(
    ctx: MLPhaseDContext,
    ml_bundle: dict[str, Any],
    trial: optuna.Trial | None,
) -> tuple[float | tuple[float, float], dict[str, Any]]:
    """Core AWF leg loop + robust objective."""
    t_compose_tot = 0.0
    t_prep_tot = 0.0
    t_prep_align_tot = 0.0
    t_prep_constraint_tot = 0.0
    t_exec_tot = 0.0
    t_metrics_tot = 0.0
    t_metrics_pure_tot = 0.0
    t_metrics_db_io_tot = 0.0

    cfg = OPT_FUTURES_CONFIG
    awf_slices = ctx.awf_leg_slices or []
    mai = ctx.multi_alignment_info
    if not awf_slices or mai is None:
        diag = {"empty": True, "robust_val": (-1e9)}
        fail = 1e9
        ns = bool(cfg.get("FUTURES_ML_ALPHA_NSGA2_ENABLED", False))
        return ((fail, fail) if ns else fail), diag

    params = (
        dict(ml_bundle)
        if ml_bundle.get("TIMEFRAME")
        else _base_engine_params(ml_bundle, ctx.tf)
    )
    params["STRATEGY_MODE"] = True
    params["ESTIMATED_B"] = ctx.estimated_b

    n_trials_eff = int(cfg.get("total_trials", 400))
    if ctx.effective_total_trials is not None:
        n_trials_eff = max(int(ctx.effective_total_trials), 1)
    gate_stat_ref = _awf_gate_stat_ref_bars(awf_slices)

    liq_mdd_thr = float(cfg.get("FUTURES_MAX_MDD", 25.0))
    n_syms_ctx = max(1, len(ctx.symbols))

    leg_log_tw: list[float] = []
    leg_mdds: list[float] = []
    all_trades_chunks: list[np.ndarray] = []
    leg_trade_counts: list[float] = []
    leg_long_counts: list[int] = []
    leg_short_counts: list[int] = []
    leg_l_pf: list[float] = []
    leg_s_pf: list[float] = []
    leg_exposures: list[float] = []
    leg_crisis_mean: list[float] = []
    chop_trade_counts: list[int] = []
    chop_loss_notional: list[float] = []
    total_loss_notional: list[float] = []
    leg_flip_proxy: list[float] = []
    leg_mdd_duration_days: list[float] = []
    leg_cvar_pct: list[float] = []
    first_leg_done = False

    for leg_idx, leg in enumerate(awf_slices):
        aligned = leg.get("data")
        leg_range: tuple[int, int] = leg["leg_range"]
        if not aligned:
            leg_log_tw.append(-10.0)
            leg_mdds.append(100.0)
            leg_trade_counts.append(0.0)
            leg_exposures.append(0.0)
            leg_crisis_mean.append(0.0)
            continue

        b_trades_raw, b_bal, b_equity, _b_diag = _run_portfolio_numba_block(
            params,
            aligned,
            ctx.estimated_b,
            trial_number=(trial.number if trial is not None else None),
        )

        t_compose_tot += aligned.get("_prof_compose", 0.0)
        t_prep_tot += aligned.get("_prof_prep", 0.0)
        t_prep_align_tot += aligned.get("_prof_prep_align", 0.0)
        t_prep_constraint_tot += aligned.get("_prof_prep_constraint", 0.0)
        t_exec_tot += aligned.get("_prof_exec", 0.0)

        t0_met = time.perf_counter()

        n_tr = int(b_trades_raw.shape[0])
        all_trades_chunks.append(b_trades_raw)

        # [LEG] diagnostic log — gated to first 5 trials
        _trial_num_leg = trial.number if trial is not None else None
        if _trial_num_leg is not None and _trial_num_leg < 5:
            _b_bars_leg = max(1, leg_range[1] - leg_range[0])
            _n_long_leg = int(np.sum(b_trades_raw[:, 3] == 1.0)) if n_tr > 0 else 0
            _n_short_leg = int(np.sum(b_trades_raw[:, 3] == -1.0)) if n_tr > 0 else 0
            _path_diag_leg = aligned.get("_strategy_signal_path_diag")
            _alpha_nz = (
                float(_path_diag_leg.get("alpha_nz", 0.0))
                if isinstance(_path_diag_leg, dict)
                else 0.0
            )
            _merge_nz = (
                float(_path_diag_leg.get("merge_nz", 0.0))
                if isinstance(_path_diag_leg, dict)
                else 0.0
            )
            _logger.info(
                "[STRAT-PATH] trial=%d leg=%d range=(%d,%d) bars=%d"
                " alpha_nz=%.4f merge_nz=%.4f trades=%d long=%d short=%d",
                _trial_num_leg,
                leg_idx,
                leg_range[0],
                leg_range[1],
                _b_bars_leg,
                _alpha_nz,
                _merge_nz,
                n_tr,
                _n_long_leg,
                _n_short_leg,
            )

        if not first_leg_done:
            first_leg_done = True
            if n_tr == 0:
                if trial is not None:
                    set_trial_event_attrs(
                        trial,
                        status="pruned",
                        reason="zero_trades_first_leg",
                        stage="awf_leg_eval",
                        step=int(leg_idx),
                        metrics={"n_trades": n_tr},
                    )
                    raise optuna.TrialPruned()
                diag = {"pruned": True, "robust_val": (-1e9)}
                return 1e9, diag

        mdd = float(calc_mdd_from_equity(b_equity)) if b_equity.size > 0 else 100.0
        mdd_duration_days = (
            float(
                calc_max_underwater_days_from_equity(
                    b_equity, hours_per_bar_from_timeframe(ctx.tf)
                )
            )
            if b_equity.size > 1
            else 0.0
        )
        cvar_pct = (
            float(calc_cvar5_loss_pct_from_equity(b_equity))
            if b_equity.size > 1
            else 0.0
        )
        log_ret = _log_tw_from_ret_pct(float((b_bal / FUTURES_INITIAL_BALANCE - 1.0) * 100.0))

        if _trial_num_leg is not None and _trial_num_leg < 5:
            _prune_suffix = (
                " -> PRUNE(first_leg_log_ret_too_low)"
                if (leg_idx == 0 and log_ret < -0.1 and trial is not None)
                else ""
            )
            _logger.info(
                "[LEG] trial=%d leg=%d log_ret=%.3f mdd=%.1f%%%s",
                _trial_num_leg,
                leg_idx,
                log_ret,
                mdd,
                _prune_suffix,
            )

        if leg_idx == 0 and trial is not None and log_ret < -0.1:
            set_trial_event_attrs(
                trial,
                status="pruned",
                reason="first_leg_log_ret_too_low",
                stage="awf_leg_eval",
                step=int(leg_idx),
                metrics={"log_ret": log_ret},
            )
            raise optuna.TrialPruned()

        if mdd >= liq_mdd_thr:
            log_ret -= (mdd - liq_mdd_thr) * 3.0

        b_bars = max(1, leg_range[1] - leg_range[0])
        b_exposure = 0.0
        n_long, n_short = 0, 0
        if n_tr > 0:
            holding_bars = float(np.sum(b_trades_raw[:, 2] - b_trades_raw[:, 1]))
            b_exposure = holding_bars / float(b_bars * n_syms_ctx)
            n_long = int(np.sum(b_trades_raw[:, 3] == 1.0))
            n_short = int(np.sum(b_trades_raw[:, 3] == -1.0))

        if n_tr > 0 and b_trades_raw.size > 0:
            _pnl_arr: NDArray[np.float64] = b_trades_raw[:, 6].astype(np.float64, copy=False)
            _dir_arr = b_trades_raw[:, 3]
            _l_pnl = _pnl_arr[_dir_arr == 1.0]
            _s_pnl = _pnl_arr[_dir_arr == -1.0]
            _l_win = float(np.sum(_l_pnl[_l_pnl > 0.0]))
            _l_loss = float(np.sum(np.abs(_l_pnl[_l_pnl < 0.0])))
            _s_win = float(np.sum(_s_pnl[_s_pnl > 0.0]))
            _s_loss = float(np.sum(np.abs(_s_pnl[_s_pnl < 0.0])))
            _lpf = _l_win / max(_l_loss, 1e-9) if _l_loss > 0 else (1.5 if _l_win > 0 else 1.0)
            _spf = _s_win / max(_s_loss, 1e-9) if _s_loss > 0 else (1.5 if _s_win > 0 else 1.0)
        else:
            _lpf, _spf = 1.0, 1.0
        leg_l_pf.append(_lpf)
        leg_s_pf.append(_spf)

        chop_tr = 0
        chop_loss = 0.0
        tot_loss = 0.0
        flip_proxy = 0.0
        if n_tr > 0 and b_trades_raw.size > 0:
            try:
                _pnl = np.asarray(b_trades_raw[:, 6], dtype=np.float64)
                tot_loss = float(np.sum(np.clip(-_pnl, 0.0, None)))
                if b_trades_raw.shape[0] >= 2:
                    _side = np.asarray(b_trades_raw[:, 3], dtype=np.float64)
                    flip_proxy = float(np.mean(np.abs(np.diff(_side)) > 0.0))
            except Exception:
                chop_tr, chop_loss, tot_loss, flip_proxy = 0, 0.0, 0.0, 0.0
        chop_trade_counts.append(int(chop_tr))
        chop_loss_notional.append(float(chop_loss))
        total_loss_notional.append(float(tot_loss))
        leg_flip_proxy.append(float(flip_proxy))

        leg_crisis_mean.append(0.0)

        leg_log_tw.append(log_ret)
        leg_mdds.append(mdd)
        leg_mdd_duration_days.append(mdd_duration_days)
        leg_cvar_pct.append(cvar_pct)
        leg_trade_counts.append(float(n_tr))
        leg_long_counts.append(n_long)
        leg_short_counts.append(n_short)
        leg_exposures.append(b_exposure)

        # Metrics Pure Calculation 시간 누적
        t_met_calc_done = time.perf_counter()
        t_metrics_pure_tot += (t_met_calc_done - t0_met)

        if leg_idx >= 2 and trial is not None:
            t0_gate = time.perf_counter()
            cum_log_tw = float(np.sum(leg_log_tw))
            max_leg_mdd = float(np.max(leg_mdds))
            if (not np.isfinite(cum_log_tw)) or (not np.isfinite(max_leg_mdd)):
                raise optuna.TrialPruned()
            if abs(cum_log_tw) > 100.0:
                raise optuna.TrialPruned()
            if cum_log_tw < -0.25 or max_leg_mdd > liq_mdd_thr:
                t_metrics_pure_tot += (time.perf_counter() - t0_gate)
                t_metrics_tot = t_metrics_pure_tot + t_metrics_db_io_tot
                set_trial_event_attrs(
                    trial,
                    status="pruned",
                    reason="hard_risk_gate_violation",
                    stage="awf_leg_eval",
                    step=int(leg_idx),
                    metrics={"cum_log_tw": cum_log_tw, "max_mdd": max_leg_mdd},
                )
                if trial is not None:
                    trial.set_user_attr("prof_compose", float(t_compose_tot))
                    trial.set_user_attr("prof_prep", float(t_prep_tot))
                    trial.set_user_attr("prof_prep_align", float(t_prep_align_tot))
                    trial.set_user_attr("prof_prep_constraint", float(t_prep_constraint_tot))
                    trial.set_user_attr("prof_exec", float(t_exec_tot))
                    trial.set_user_attr("prof_metrics", float(t_metrics_tot))
                    trial.set_user_attr("prof_metrics_pure", float(t_metrics_pure_tot))
                    trial.set_user_attr("prof_metrics_db_io", float(t_metrics_db_io_tot))
                raise optuna.TrialPruned()
            t_metrics_pure_tot += (time.perf_counter() - t0_gate)

            if (
                trial is not None
                and len(trial.study.directions) == 1
                and bool(cfg.get("FUTURES_PRUNING_ENABLED", False))
            ):
                # Apply 2-step stride to avoid heavy SQLite WAL DB lock contention.
                # Always report on the last leg to guarantee pruning sanity.
                # Keep stride=1 for small slices (<=3) for test compatibility.
                n_slices = len(awf_slices)
                should_check = True
                if n_slices > 3:
                    is_last_leg = (leg_idx == n_slices - 1)
                    should_check = (leg_idx % 2 == 0) or is_last_leg

                if should_check:
                    t0_db = time.perf_counter()
                    trial.report(float(np.mean(leg_log_tw)), step=leg_idx)
                    is_pruned = trial.should_prune()
                    t_metrics_db_io_tot += (time.perf_counter() - t0_db)

                    if is_pruned:
                        t_metrics_tot = t_metrics_pure_tot + t_metrics_db_io_tot
                        set_trial_event_attrs(
                            trial,
                            status="pruned",
                            reason="trial_should_prune",
                            stage="awf_intermediate_report",
                            step=int(leg_idx),
                            metrics={"mean_leg_log_tw": float(np.mean(leg_log_tw))},
                        )
                        if trial is not None:
                            trial.set_user_attr("prof_compose", float(t_compose_tot))
                            trial.set_user_attr("prof_prep", float(t_prep_tot))
                            trial.set_user_attr("prof_prep_align", float(t_prep_align_tot))
                            trial.set_user_attr(
                                "prof_prep_constraint",
                                float(t_prep_constraint_tot),
                            )
                            trial.set_user_attr("prof_exec", float(t_exec_tot))
                            trial.set_user_attr("prof_metrics", float(t_metrics_tot))
                            trial.set_user_attr("prof_metrics_pure", float(t_metrics_pure_tot))
                            trial.set_user_attr("prof_metrics_db_io", float(t_metrics_db_io_tot))
                        raise optuna.TrialPruned()
        else:
            if trial is not None:
                t_metrics_pure_tot += (time.perf_counter() - t0_met)

        t_metrics_tot = t_metrics_pure_tot + t_metrics_db_io_tot

    leg_arr = np.asarray(leg_log_tw, dtype=np.float64)
    n_legs_done = leg_arr.size
    all_trades = (
        np.vstack(all_trades_chunks) if all_trades_chunks
        else np.zeros((0, 10), dtype=np.float64)
    )

    avg_trades_agg = float(np.mean(leg_trade_counts)) if leg_trade_counts else 0.0
    worst_mdd_legs = float(max(leg_mdds, default=100.0))
    total_long = sum(leg_long_counts)
    total_short = sum(leg_short_counts)
    total_dir = total_long + total_short
    minority = float(min(total_long, total_short) / total_dir) if total_dir > 0 else 0.0

    k_legs_n = float(max(n_legs_done, 1))
    mu_log = float(np.mean(leg_arr)) if leg_arr.size > 0 else -10.0
    worst_leg = float(np.min(leg_arr)) if leg_arr.size > 0 else -10.0
    awf_pos_frac = float(np.sum(leg_arr > 0.0)) / k_legs_n
    dsr_awf = calc_gate1_dsr_from_path_log_tw(
        leg_arr,
        ctx.tf,
        float(gate_stat_ref),
        float(n_trials_eff),
    )

    sig_awf_diag = float(np.std(leg_arr, ddof=1)) if leg_arr.size >= 2 else 0.0
    _awf_pf_agg, ev_cost_ratio = _pf_and_ev_cost_from_trades(all_trades)
    funding_drag_ratio, _funding_drag_basis = _funding_drag_ratio_from_trades(all_trades)
    mdd_duration_days = float(max(leg_mdd_duration_days, default=0.0))
    cvar_pct = float(max(leg_cvar_pct, default=0.0))
    turnover_cost_ratio = float(np.clip(1.0 / max(ev_cost_ratio, 1e-9), 0.0, 1e6))

    robust_val = compute_v3_score(
        leg_log_tw=leg_arr,
        worst_mdd=worst_mdd_legs / 100.0,
        cvar_5=cvar_pct / 100.0,
        excess_turnover=turnover_cost_ratio,
        funding_drag=funding_drag_ratio,
        aum_impact_penalty=0.0,
    )

    total_trades_agg = float(np.sum(leg_trade_counts)) if leg_trade_counts else 0.0
    total_chop_trades = float(np.sum(chop_trade_counts)) if chop_trade_counts else 0.0
    chop_trade_share = float(total_chop_trades / max(total_trades_agg, 1.0))
    loss_total = float(np.sum(total_loss_notional)) if total_loss_notional else 0.0
    loss_chop = float(np.sum(chop_loss_notional)) if chop_loss_notional else 0.0
    chop_loss_share = float(loss_chop / max(loss_total, 1e-9)) if loss_total > 0.0 else 0.0
    step4_enabled = bool(cfg.get("FUTURES_STEP4_DEPLOYABILITY_ENABLED", False))
    if step4_enabled:
        chop_trade_w4 = float(cfg.get("FUTURES_STEP4_OBJ_CHOP_TRADE_W", 0.10))
        turnover_w4 = float(cfg.get("FUTURES_STEP4_OBJ_TURNOVER_W", 0.10))
        chop_trade_ref = float(
            cfg.get(
                "FUTURES_STEP2_CHOP_TRADE_SHARE_MAX",
                cfg.get("FUTURES_STEP4_CHOP_TRADE_SHARE_MAX", 0.70),
            )
        )
        turnover_ref = float(cfg.get("FUTURES_STEP4_TURNOVER_COST_RATIO_MAX", 0.35))
        excess_trade4 = max(0.0, chop_trade_share - chop_trade_ref)
        excess_turnover4 = max(0.0, turnover_cost_ratio - turnover_ref)
        robust_val -= chop_trade_w4 * excess_trade4 + turnover_w4 * excess_turnover4

    if leg_arr.size >= 2:
        _tw_legs = np.exp(leg_arr)
        _tw_mean = float(np.mean(_tw_legs))
        _erg_dev_pct = (
            float(np.max(np.abs(_tw_legs - _tw_mean)) / max(_tw_mean, 1e-9) * 100.0)
            if _tw_mean > 1e-9
            else 0.0
        )
        _erg_dev_floor = float(cfg.get("FUTURES_AWF_ERG_DEV_FLOOR", 1.5))
        _erg_dev_w = float(cfg.get("FUTURES_AWF_ERG_DEV_W", 0.001))
        robust_val -= _erg_dev_w * max(0.0, _erg_dev_pct - _erg_dev_floor)

    obj = -robust_val
    k_cfg = int(cfg.get("FUTURES_AWF_K_LEGS", 6))
    if n_legs_done < k_cfg:
        obj += 20.0 * (k_cfg - n_legs_done)

    diag_res: dict[str, Any] = {
        "objective": float(obj),
        "robust_val": float(robust_val),
        "awf_robust_score": float(robust_val),
        "awf_leg_log_tw": [float(x) for x in leg_log_tw],
        "awf_leg_trade_counts": [float(x) for x in leg_trade_counts],
        "awf_pos_frac": awf_pos_frac,
        "gate1_dsr": dsr_awf,
        "awf_mu_log": mu_log,
        "awf_sigma_log": sig_awf_diag,
        "awf_worst_leg_log_tw": float(worst_leg),
        "awf_worst_mdd_pct": float(worst_mdd_legs),
        "avg_trades": avg_trades_agg,
        "awf_turnover_cost_ratio": float(turnover_cost_ratio),
        "awf_funding_drag_ratio": float(funding_drag_ratio),
        "mdd_duration": float(mdd_duration_days),
        "cvar": float(cvar_pct),
        "minority_side_ratio": float(minority),
        "n_trades": float(avg_trades_agg),
    }

    if trial is not None:
        trial.set_user_attr("awf_leg_log_tw", [float(x) for x in leg_log_tw])
        trial.set_user_attr("awf_mu_log", mu_log)
        trial.set_user_attr("awf_sigma_log", sig_awf_diag)
        trial.set_user_attr("awf_robust_score", float(robust_val))
        trial.set_user_attr("awf_pos_frac", awf_pos_frac)
        trial.set_user_attr("gate1_dsr", dsr_awf)
        trial.set_user_attr("awf_worst_leg_log_tw", float(worst_leg))
        trial.set_user_attr("awf_worst_mdd_pct", float(worst_mdd_legs))
        trial.set_user_attr("avg_trades", avg_trades_agg)
        trial.set_user_attr("awf_turnover_cost_ratio", float(turnover_cost_ratio))
        trial.set_user_attr("awf_funding_drag_ratio", float(funding_drag_ratio))
        trial.set_user_attr("mdd_duration", float(mdd_duration_days))
        trial.set_user_attr("cvar", float(cvar_pct))
        trial.set_user_attr("minority_side_ratio", float(minority))
        trial.set_user_attr("n_trades", float(avg_trades_agg))
        trial.set_user_attr(
            "phase2_cost_forecast_dynamic",
            float(
                1.0
                if bool(
                    params.get(
                        "COST_FORECAST_DYNAMIC",
                        cfg.get("COST_FORECAST_DYNAMIC", False),
                    )
                )
                else 0.0
            ),
        )
        trial.set_user_attr(
            "phase2_cost_gate_amortize",
            float(1.0 if bool(params.get("COST_GATE_AMORTIZE", False)) else 0.0),
        )
        trial.set_user_attr(
            "phase2_kelly_use_residual_var",
            float(
                1.0
                if bool(
                    params.get(
                        "KELLY_USE_RESIDUAL_VAR",
                        cfg.get("KELLY_USE_RESIDUAL_VAR", False),
                    )
                )
                else 0.0
            ),
        )
        trial.set_user_attr("prof_compose", float(t_compose_tot))
        trial.set_user_attr("prof_prep", float(t_prep_tot))
        trial.set_user_attr("prof_prep_align", float(t_prep_align_tot))
        trial.set_user_attr("prof_prep_constraint", float(t_prep_constraint_tot))
        trial.set_user_attr("prof_exec", float(t_exec_tot))
        trial.set_user_attr("prof_metrics", float(t_metrics_tot))
        trial.set_user_attr("prof_metrics_pure", float(t_metrics_pure_tot))
        trial.set_user_attr("prof_metrics_db_io", float(t_metrics_db_io_tot))

    ns2 = bool(cfg.get("FUTURES_ML_ALPHA_NSGA2_ENABLED", False))
    if ns2:
        obj1 = -float(robust_val)
        tail_mdd_w = float(cfg.get("FUTURES_AWF_OBJ_PSI_DD", 0.5))
        chop_trade_w = float(cfg.get("FUTURES_STEP2_OBJ_CHOP_LOSS_W", 0.25))
        obj2 = (
            -float(worst_leg)
            + tail_mdd_w * float(worst_mdd_legs)
            + chop_trade_w * chop_loss_share
        )
        return (obj1, obj2), diag_res
    return float(obj), diag_res


def _evaluate_is_phase_d(
    ctx: MLPhaseDContext,
    ml_bundle: dict[str, Any],
    trial: optuna.Trial | None,
) -> tuple[tuple[float, float], dict[str, Any]]:
    """Single-IS backtest for decoupled optimization."""
    cfg = OPT_FUTURES_CONFIG
    aligned = ctx.is_slice
    mai = ctx.multi_alignment_info

    if aligned is None or mai is None:
        fail = 1e9
        return (fail, fail), {"empty": True}

    params = ml_bundle if ml_bundle.get("TIMEFRAME") else _base_engine_params(ml_bundle, ctx.tf)
    params["ESTIMATED_B"] = ctx.estimated_b

    b_trades_raw, b_bal, b_equity, _b_diag = _run_portfolio_numba_block(
        params, aligned, ctx.estimated_b
    )

    n_tr = int(b_trades_raw.shape[0])
    if n_tr == 0:
        if trial is not None:
            raise optuna.TrialPruned()
        return (1e9, 1e9), {"pruned": True}

    is_ret_pct = (b_bal / FUTURES_INITIAL_BALANCE - 1.0) * 100.0
    if trial is not None and is_ret_pct < -20.0:
        raise optuna.TrialPruned()

    is_mdd = float(calc_mdd_from_equity(b_equity))

    k_chunks = 10
    n_bars = b_equity.size
    chunk_size = max(1, n_bars // k_chunks)
    leg_log_tw = []
    for i in range(k_chunks):
        s = i * chunk_size
        e = (i + 1) * chunk_size if i < k_chunks - 1 else n_bars
        if e > s + 1:
            chunk_ret = (b_equity[e-1] / b_equity[s] - 1.0) * 100.0
            leg_log_tw.append(_log_tw_from_ret_pct(chunk_ret))

    leg_arr = np.asarray(leg_log_tw, dtype=np.float64)
    is_cvar = float(calc_cvar5_loss_pct_from_equity(b_equity)) if b_equity.size > 0 else 0.0
    robust_val = compute_v3_score(
        leg_log_tw=leg_arr,
        worst_mdd=is_mdd / 100.0,
        cvar_5=is_cvar / 100.0,
        excess_turnover=0.0,
        funding_drag=0.0,
        aum_impact_penalty=0.0,
    )

    n_trials_eff = int(cfg.get("total_trials", 1500))
    if ctx.effective_total_trials is not None:
        n_trials_eff = max(int(ctx.effective_total_trials), 1)

    is_dsr = calc_gate1_dsr_from_path_log_tw(
        leg_arr, ctx.tf, float(n_bars), float(n_trials_eff)
    )

    obj1 = -float(robust_val)
    obj2 = -float(np.min(leg_arr)) if leg_arr.size > 0 else 1e9

    if trial is not None:
        trial.set_user_attr("IS_MDD", is_mdd)
        trial.set_user_attr("IS_DSR", is_dsr)
        trial.set_user_attr("IS_RET_PCT", float(is_ret_pct))
        trial.set_user_attr("IS_ROBUST_SCORE", float(robust_val))
        trial.set_user_attr("avg_trades", float(n_tr))

    diag = {
        "is_ret_pct": is_ret_pct,
        "is_mdd": is_mdd,
        "is_dsr": is_dsr,
        "robust_val": robust_val,
        "obj1": obj1,
        "obj2": obj2,
    }
    return (obj1, obj2), diag


def objective_ml_phase_d(trial: optuna.Trial, ctx: MLPhaseDContext) -> tuple[float, float] | float:
    """Joint NSGA-II Portfolio Optimization \u2014 AWF-based objectives (T2)."""
    from src.domain.futures.optimization.samplers import _suggest_ml_joint_nsga2
    if hasattr(ctx, "registry") and ctx.registry is not None:
        ctx.registry.validate()
    if ctx.awf_leg_slices is None:
        precompute_ml_optimization_context(ctx)
    if ctx.run_id:
        trial.set_user_attr("run_id", str(ctx.run_id))
    try:
        merged = _suggest_ml_joint_nsga2(trial, ctx)
        result, _ = _evaluate_awf_phase_d_aggregate(ctx, merged, trial=trial)
        if isinstance(result, tuple):
            return result
        return float(result)
    except optuna.TrialPruned:
        raise
    except Exception:
        set_trial_event_attrs(
            trial,
            status="failed",
            reason="objective_exception",
            stage="objective_ml_phase_d",
        )
        raise


def select_best_trial_by_holdout_log_ret(
    trials: list[optuna.trial.FrozenTrial]
) -> optuna.trial.FrozenTrial:
    """Select the best trial from a list based on a multi-metric scoring system."""
    if not trials:
        raise ValueError("empty trials")

    def _score(
        t: optuna.trial.FrozenTrial
    ) -> tuple[float, float, float, float, float, float, float]:
        holdout = float(np.clip(t.user_attrs.get("ml_holdout_log_ret", 0.0), -2.0, 2.0))
        robust = float(
            t.user_attrs.get("awf_robust_score", t.user_attrs.get("awf_contract_reward", -1e9))
        )
        is_cpcv = float(
            np.clip(
                t.user_attrs.get(
                    "awf_mean_log_tw", t.user_attrs.get("ml_mean_log_growth_cpcv", -2.0)
                ),
                -2.0,
                2.0,
            )
        )
        p10_cpcv = float(
            np.clip(
                t.user_attrs.get(
                    "awf_worst_leg_log_tw", t.user_attrs.get("ml_p10_log_growth_cpcv", -2.0)
                ),
                -2.0,
                2.0,
            )
        )
        worst_mdd = float(
            t.user_attrs.get("awf_worst_mdd_pct", t.user_attrs.get("ml_worst_mdd_cpcv", 999.0))
        )
        dsr = float(t.user_attrs.get("gate1_dsr", 0.0))
        path_std = float(np.clip(t.user_attrs.get("ml_std_log_growth_cpcv", 1.0), 0.0, 2.0))
        if is_cpcv < 0:
            holdout = holdout - abs(is_cpcv) * 2.0
        return (robust, dsr, is_cpcv, p10_cpcv, -path_std, -worst_mdd, holdout)

    return max(trials, key=_score)


def topsis_select_best(pareto_trials: list[optuna.trial.FrozenTrial]) -> optuna.trial.FrozenTrial:
    """Select the best trial from a Pareto front using the TOPSIS method."""
    if not pareto_trials:
        raise ValueError("empty pareto_trials")
    if len(pareto_trials) == 1:
        return pareto_trials[0]
    def _safe_float(v: Any, default: float) -> float:
        try:
            x = float(v)
            return x if np.isfinite(x) else default
        except Exception:
            return default

    feats: list[list[float]] = []
    for t in pareto_trials:
        ua = t.user_attrs
        robust = _safe_float(
            ua.get("awf_robust_score", ua.get("awf_contract_reward", np.nan)), np.nan
        )
        if not np.isfinite(robust):
            v0 = float(t.values[0]) if t.values else np.nan
            robust = -v0 if np.isfinite(v0) else -1e9

        mu_log = _safe_float(
            ua.get(
                "awf_mu_log",
                ua.get("awf_mean_log_tw", ua.get("ml_mean_log_growth_cpcv", np.nan))
            ),
            -2.0,
        )
        worst_leg = _safe_float(
            ua.get("awf_worst_leg_log_tw", ua.get("ml_p10_log_growth_cpcv", np.nan)),
            -2.0,
        )
        pos_frac = _safe_float(ua.get("awf_pos_frac", np.nan), 0.0)
        pos_frac = float(np.clip(pos_frac, 0.0, 1.0))
        mdd = _safe_float(
            ua.get("awf_worst_mdd_pct", ua.get("ml_worst_mdd_cpcv", np.nan)),
            999.0,
        )
        feats.append([robust, mu_log, worst_leg, pos_frac, -mdd])

    x = np.asarray(feats, dtype=np.float64)
    xmin = np.min(x, axis=0)
    xmax = np.max(x, axis=0)
    span = xmax - xmin
    norm = np.where(span > 1e-12, (x - xmin) / span, 0.5)

    weights = np.asarray([0.40, 0.20, 0.20, 0.10, 0.10], dtype=np.float64)
    score = np.sum(norm * weights, axis=1)

    best_idx = max(
        range(len(pareto_trials)),
        key=lambda i: (
            float(score[i]),
            float(x[i, 0]),
            float(x[i, 2]),
            float(x[i, 1]),
            float(x[i, 3]),
            float(x[i, 4]),
            -int(pareto_trials[i].number),
        ),
    )
    return pareto_trials[int(best_idx)]
