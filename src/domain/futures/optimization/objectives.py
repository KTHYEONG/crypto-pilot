from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import pandas as pd

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
from src.domain.futures.optimization.evaluator import (
    _log_tw_from_ret_pct,
    calc_cvar5_loss_pct_from_equity,
    calc_gate1_dsr_from_path_log_tw,
    calc_max_underwater_days_from_equity,
    calc_mdd_from_equity,
    compute_v3_score,
)
from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG
from src.domain.futures.optimization.observability.trial_observability import set_trial_event_attrs
from src.domain.futures.portfolio.portfolio_constructor import (
    portfolio_weight_params_from_optuna,
    precompute_rebalance_weights,
)
from src.domain.futures.portfolio.signal_composer import apply_linear_signal_composer_scores
from src.domain.futures.optimization.ml_context import (
    precompute_ml_optimization_context,
    merge_membership_constraints_into_aligned,
    rerun_precompute_for_ctx,
)

from src.domain.futures.optimization.common import (
    _diag_to_dict,
    _weight_stage_diag,
    _safe_float_or_none,
)

if TYPE_CHECKING:
    from src.domain.futures.optimization.ml_context import MLPhaseDContext

_logger = logging.getLogger(__name__)


def _build_strategy_compose_diag(
    *,
    alpha_long: np.ndarray,
    alpha_short: np.ndarray,
    xs_long: np.ndarray,
    xs_short: np.ndarray,
    hmm_probs: dict[str, np.ndarray],
    params: dict[str, Any],
) -> dict[str, float]:
    n_bars = alpha_long.shape[0]
    beta_a = float(
        params.get("BETA_ALPHA", OPT_FUTURES_CONFIG.get("FUTURES_DEFAULT_BETA_ALPHA", 1.0))
    )
    b_bull = float(
        params.get(
            "BETA_REGIME_BULL",
            OPT_FUTURES_CONFIG.get("FUTURES_DEFAULT_BETA_REGIME_BULL", 1.0),
        )
    )
    b_bear = float(
        params.get(
            "BETA_REGIME_BEAR",
            OPT_FUTURES_CONFIG.get("FUTURES_DEFAULT_BETA_REGIME_BEAR", 1.0),
        )
    )
    b_crisis = float(
        params.get(
            "BETA_REGIME_CRISIS",
            OPT_FUTURES_CONFIG.get("FUTURES_DEFAULT_BETA_REGIME_CRISIS", 1.0),
        )
    )
    b_chop = float(
        params.get(
            "BETA_REGIME_CHOP",
            OPT_FUTURES_CONFIG.get("FUTURES_DEFAULT_BETA_REGIME_CHOP", 0.25),
        )
    )
    b_rec = float(
        params.get(
            "BETA_REGIME_RECOVERY",
            OPT_FUTURES_CONFIG.get("FUTURES_DEFAULT_BETA_REGIME_RECOVERY", 0.0),
        )
    )
    ev_h = float(
        params.get("EV_HURDLE_BPS", OPT_FUTURES_CONFIG.get("FUTURES_DEFAULT_EV_HURDLE_BPS", 5.0))
    )
    from src.core.settings import SLIPPAGE_RATE, TAKER_FEE_RATE
    slip = float(SLIPPAGE_RATE) * float(params.get("SLIPPAGE_BPS_BUFFER_MULT", 1.0))
    fee = float(TAKER_FEE_RATE)
    fund_bar = float(OPT_FUTURES_CONFIG.get("FUTURES_COMPOSER_FUNDING_BAR_FRAC", 1e-5))
    buf_mult = float(OPT_FUTURES_CONFIG.get("FUTURES_FRICTION_BUFFER_MULT", 1.5))
    friction = buf_mult * (fee + slip + fund_bar)
    friction_bps = friction * 10000.0
    threshold_bps = friction_bps + ev_h

    pbull = hmm_probs["hmm_prob_bull_calm"] + hmm_probs["hmm_prob_bull_vol_up"]
    regime = (
        b_bull * pbull
        + b_bear * hmm_probs["hmm_prob_bear_trend"]
        + b_chop * hmm_probs["hmm_prob_chop"]
        + b_crisis * hmm_probs["hmm_prob_crisis"]
        + b_rec * hmm_probs["hmm_prob_recovery"]
    )
    regime = np.broadcast_to(regime[:, None], alpha_long.shape)
    mu_l_pre = (beta_a * alpha_long) + regime - friction
    mu_s_pre = (beta_a * alpha_short) + regime - friction

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
        "friction_bps": float(friction_bps),
        "ev_hurdle_bps": float(ev_h),
        "effective_threshold_bps": float(threshold_bps),
        "mu_pre_hurdle_p95_long": _safe_pct(mu_l_pre, 95),
        "mu_pre_hurdle_p95_short": _safe_pct(mu_s_pre, 95),
        "xs_long_nz_ratio": _nonzero_ratio(xs_long),
        "xs_short_nz_ratio": _nonzero_ratio(xs_short),
        "mu_long_mean": float(np.mean(mu_l_pre)) if mu_l_pre.size > 0 else 0.0,
        "mu_short_mean": float(np.mean(mu_s_pre)) if mu_s_pre.size > 0 else 0.0,
        "threshold_bps": threshold_bps,
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
    pnl = all_trades[:, 6].astype(np.float64, copy=False)
    funding_fee = all_trades[:, 9].astype(np.float64, copy=False)
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
        "BETA_REGIME_BULL": float(
            ml.get("BETA_REGIME_BULL", cfg.get("FUTURES_DEFAULT_BETA_REGIME_BULL", 1.0))
        ),
        "BETA_REGIME_BEAR": float(
            ml.get("BETA_REGIME_BEAR", cfg.get("FUTURES_DEFAULT_BETA_REGIME_BEAR", 0.25))
        ),
        "BETA_REGIME_CRISIS": float(
            ml.get("BETA_REGIME_CRISIS", cfg.get("FUTURES_DEFAULT_BETA_REGIME_CRISIS", -0.5))
        ),
        "BETA_REGIME_CHOP": float(
            ml.get("BETA_REGIME_CHOP", cfg.get("FUTURES_DEFAULT_BETA_REGIME_CHOP", 0.25))
        ),
        "EV_HURDLE_BPS": float(
            ml.get("EV_HURDLE_BPS", cfg.get("FUTURES_DEFAULT_EV_HURDLE_BPS", 5.0))
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
        "FUTURES_EXECUTION_MODE": str(ml.get("FUTURES_EXECUTION_MODE", "coarse")),
        "STRATEGY_MODE": bool(ml.get("STRATEGY_MODE", False)),
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


def _compose_strategy_scores_inplace(aligned: dict[str, Any], params: dict[str, Any]) -> None:
    """Build xs_score from alpha_long/alpha_short for strategy-mode trials."""
    alpha_l = aligned.get("alpha_long")
    alpha_s = aligned.get("alpha_short")
    if alpha_l is None or alpha_s is None:
        raise RuntimeError("strategy mode requires aligned alpha_long/alpha_short")
    alpha_l_2d = np.asarray(alpha_l, dtype=np.float64)
    alpha_s_2d = np.asarray(alpha_s, dtype=np.float64)
    if alpha_l_2d.ndim != 2 or alpha_s_2d.ndim != 2 or alpha_l_2d.shape != alpha_s_2d.shape:
        raise RuntimeError("strategy mode requires 2D alpha_long/alpha_short with matching shape")

    n_bars, n_syms = alpha_l_2d.shape
    xs_l = np.zeros((n_bars, n_syms), dtype=np.float64)
    xs_s = np.zeros((n_bars, n_syms), dtype=np.float64)
    hmm_cols = (
        "hmm_prob_bull_calm",
        "hmm_prob_bull_vol_up",
        "hmm_prob_bear_trend",
        "hmm_prob_chop",
        "hmm_prob_crisis",
    )
    hmm_prob_map: dict[str, np.ndarray] = {}
    for hmm_col in hmm_cols:
        hmm_2d = aligned.get(hmm_col)
        if hmm_2d is None:
            hmm_prob_map[hmm_col] = np.zeros((n_bars,), dtype=np.float64)
            continue
        hmm_arr = np.asarray(hmm_2d, dtype=np.float64)
        if hmm_arr.ndim != 2 or hmm_arr.shape != alpha_l_2d.shape:
            raise RuntimeError(f"strategy mode requires aligned {hmm_col} with alpha shape")
        hmm_prob_map[hmm_col] = np.mean(hmm_arr, axis=1)
    hmm_prob_map["hmm_prob_recovery"] = np.zeros((n_bars,), dtype=np.float64)

    for col_idx in range(n_syms):
        composer_df = pd.DataFrame(index=np.arange(n_bars))
        for hmm_col in hmm_cols:
            hmm_2d = aligned.get(hmm_col)
            if hmm_2d is None:
                composer_df[hmm_col] = np.zeros(n_bars, dtype=np.float64)
                continue
            hmm_arr = np.asarray(hmm_2d, dtype=np.float64)
            composer_df[hmm_col] = hmm_arr[:, col_idx]
        xl, xs = apply_linear_signal_composer_scores(
            composer_df,
            alpha_l_2d[:, col_idx],
            alpha_s_2d[:, col_idx],
            params,
            opt_config=OPT_FUTURES_CONFIG,
        )
        xs_l[:, col_idx] = xl
        xs_s[:, col_idx] = xs

    aligned["xs_score_long"] = np.ascontiguousarray(xs_l)
    aligned["xs_score_short"] = np.ascontiguousarray(xs_s)
    aligned["_strategy_compose_diag"] = _build_strategy_compose_diag(
        alpha_long=alpha_l_2d,
        alpha_short=alpha_s_2d,
        xs_long=xs_l,
        xs_short=xs_s,
        hmm_probs=hmm_prob_map,
        params=params,
    )


def _run_portfolio_numba_block(
    params: dict[str, Any],
    aligned: dict[str, Any],
    estimated_b: float = 1.05,
    *,
    trial_number: int | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    _ = estimated_b
    if bool(params.get("STRATEGY_MODE", False)):
        _compose_strategy_scores_inplace(aligned, params)
        if "xs_score_long" not in aligned or "xs_score_short" not in aligned:
            raise RuntimeError("strategy mode failed to generate xs_score_long/xs_score_short")
    prepared = prepare_backtest_inputs(aligned, params)
    aligned = prepared.aligned_data
    merge_membership_constraints_into_aligned(aligned, persist_stats=True)
    zkill, zfund, lev_blk = _cached_kill_fund_lev(aligned, params)
    cfg_block = OPT_FUTURES_CONFIG
    reb_b = max(1, int(params["REBALANCE_BARS"]))
    pwp = portfolio_weight_params_from_optuna(params, cfg_block)

    _hmm_cols_t3 = [
        "hmm_prob_bull_calm", "hmm_prob_bull_vol_up", "hmm_prob_bear_trend",
        "hmm_prob_chop", "hmm_prob_crisis",
    ]
    _hmm_t3 = [aligned.get(c) for c in _hmm_cols_t3]
    if all(a is not None for a in _hmm_t3):
        def _to_1d(a: Any) -> np.ndarray:
            arr = np.asarray(a, dtype=np.float64)
            return arr[:, 0] if arr.ndim == 2 else arr
        _p5 = np.stack([_to_1d(a) for a in _hmm_t3], axis=1)
        _log5 = np.log(5.0)
        _ent = -np.sum(_p5 * np.log(np.clip(_p5, 1e-12, 1.0)), axis=1)
        _h_norm = float(np.mean(_ent) / _log5)
        _mean_crisis = float(np.mean(np.clip(_p5[:, 4], 0.0, 1.0)))
        _kelly_disc = max(0.1, (1.0 - _h_norm) * (1.0 - _mean_crisis))
        pwp["f_kelly_max"] = float(pwp["f_kelly_max"]) * _kelly_disc
    pwp["f_kelly_max"] = min(float(pwp["f_kelly_max"]), float(params.get("KELLY_IC_UPPER", 0.5)))

    hpb = hours_per_bar_from_timeframe(str(params.get("TIMEFRAME", "4h")))
    bars_py = (365.0 * 24.0) / max(hpb, 1e-9)
    close_np = np.asarray(aligned["close"], dtype=np.float64)
    xl = np.asarray(
        aligned.get("xs_score_long", np.zeros_like(close_np)), dtype=np.float64
    )
    xs = np.asarray(
        aligned.get("xs_score_short", np.zeros_like(close_np)), dtype=np.float64
    )
    sigma_3d = aligned.get("sigma_3d")
    hmm_probs_2d = None
    _hmm_cols_pw = [
        "hmm_prob_bull_calm",
        "hmm_prob_bull_vol_up",
        "hmm_prob_bear_trend",
        "hmm_prob_chop",
        "hmm_prob_crisis",
    ]
    _hmm_blocks_pw = [aligned.get(c) for c in _hmm_cols_pw]
    if all(b is not None for b in _hmm_blocks_pw):
        _cols_pw = []
        for b in _hmm_blocks_pw:
            arr = np.asarray(b, dtype=np.float64)
            _cols_pw.append(arr[:, 0] if arr.ndim == 2 else arr)
        hmm_probs_2d = np.stack(_cols_pw, axis=1)
    tw_blk = np.asarray(
        precompute_rebalance_weights(
            close_np,
            xl,
            xs,
            rebalance_bars=reb_b,
            lookback=int(pwp["lookback"]),
            bars_per_year=bars_py,
            kappa=float(pwp["kappa"]),
            f_kelly_max=float(pwp["f_kelly_max"]),
            sigma_target_ann=float(pwp["sigma_target_ann"]),
            gross_cap=float(pwp["gross_cap"]),
            per_symbol_cap=float(pwp["per_symbol_cap"]),
            current_dd=0.0,
            composer_sigma_2d=(
                np.asarray(aligned["composer_sigma_bar"], dtype=np.float64)
                if aligned.get("composer_sigma_bar") is not None
                else None
            ),
            sigma_3d=sigma_3d,
            hmm_probs_2d=hmm_probs_2d,
            regime_policy_enabled=bool(pwp.get("regime_policy_enabled", False)),
            chop_gross_damp=float(pwp.get("chop_gross_damp", 0.50)),
            crisis_gross_damp=float(pwp.get("crisis_gross_damp", 0.80)),
            entropy_gross_damp=float(pwp.get("entropy_gross_damp", 0.35)),
            bear_gross_damp=float(pwp.get("bear_gross_damp", 0.10)),
            gross_floor_mult=float(pwp.get("gross_floor_mult", 0.15)),
            crisis_long_suppress_thr=float(pwp.get("crisis_long_suppress_thr", 0.60)),
            crisis_long_suppress_mult=float(pwp.get("crisis_long_suppress_mult", 0.10)),
        ),
        dtype=np.float64,
    )
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
    if use_intrabar:
        exec_start = prepared.exec_bar_start_1m_idx
        exec_end = prepared.exec_bar_end_1m_idx
        out_tw = backtest_target_weights_intrabar_numba(
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
        )
    else:
        out_tw = backtest_target_weights_numba(
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
            aligned["atr"],
            atr_m,
            trail_m,
            use_simple_atr_i,
            int(params.get("MAX_CONCURRENT_POSITIONS", 2)),
            float(params.get("MAX_EXPOSURE", 0.8)),
            float(params["MAX_EXPOSURE_PER_COIN"]),
            float(params["DD_SCALING_THRESHOLD"]),
            volume_2d=aligned.get("volume"),
        )
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
    cfg = OPT_FUTURES_CONFIG
    awf_slices = ctx.awf_leg_slices or []
    mai = ctx.multi_alignment_info
    if not awf_slices or mai is None:
        diag = {"empty": True, "robust_val": (-1e9)}
        fail = 1e9
        ns = bool(cfg.get("FUTURES_ML_ALPHA_NSGA2_ENABLED", False))
        return ((fail, fail) if ns else fail), diag

    if ml_bundle.get("TIMEFRAME"):
        params = dict(ml_bundle)
    else:
        params = _base_engine_params(ml_bundle, ctx.tf)
    if ctx.strategy_mode:
        params["STRATEGY_MODE"] = True
    params["ESTIMATED_B"] = ctx.estimated_b
    params["KELLY_IC_UPPER"] = ctx.kelly_ic_upper

    n_trials_eff = int(cfg.get("total_trials", 400))
    if ctx.effective_total_trials is not None:
        n_trials_eff = max(int(ctx.effective_total_trials), 1)
    gate_stat_ref = _awf_gate_stat_ref_bars(awf_slices)

    if trial is not None and trial.number < 10 and awf_slices:
        ad0 = awf_slices[0].get("data") or {}
        xl = ad0.get("xs_score_long")
        hy = ad0.get("hmm_prob_crisis")
        if xl is not None and hy is not None and getattr(xl, "size", 0) > 0:
            disp = float(np.nanstd(np.asarray(xl, dtype=np.float64)))
            cclip = np.clip(np.asarray(hy, dtype=np.float64), 0.0, 1.0)
            gamma = float(params.get("CRISIS_GAMMA", params.get("CRISIS_GATE_PROB", 1.0)))
            soft_m = float(np.mean((1.0 - cclip) ** gamma))
            thr = float(cfg.get("FUTURES_HMM_CRISIS_THRESHOLD", 0.6))
            rej_r = float(np.mean(np.max(hy, axis=1) > thr))
            trial.set_user_attr("xs_score_dispersion_mean", disp)
            trial.set_user_attr("crisis_soft_weight_mean", soft_m)
            trial.set_user_attr("crisis_gate_rejection_rate", rej_r)

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

        n_tr = int(b_trades_raw.shape[0])
        all_trades_chunks.append(b_trades_raw)

        if not first_leg_done:
            first_leg_done = True
            if n_tr == 0:
                diag_dict = _diag_to_dict(_b_diag)
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

        if leg_idx == 0 and trial is not None:
            if log_ret < -0.1:
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
            _pnl_arr = b_trades_raw[:, 6].astype(np.float64, copy=False)
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
                _sym_idx = np.asarray(b_trades_raw[:, 0], dtype=np.int64)
                _entry_idx = np.asarray(b_trades_raw[:, 1], dtype=np.int64)
                _pnl = np.asarray(b_trades_raw[:, 6], dtype=np.float64)
                _chop_2d = aligned.get("hmm_prob_chop")
                if _chop_2d is not None:
                    _chop_np = np.asarray(_chop_2d, dtype=np.float64)
                    if _chop_np.ndim == 2 and _chop_np.size > 0:
                        rb, cb = _chop_np.shape
                        _r = np.clip(_entry_idx, 0, max(rb - 1, 0))
                        _c = np.clip(_sym_idx, 0, max(cb - 1, 0))
                        _p_chop = _chop_np[_r, _c]
                        _is_chop = _p_chop >= 0.50
                        chop_tr = int(np.sum(_is_chop))
                        if np.any(_is_chop):
                            _chop_pnl = _pnl[_is_chop]
                            chop_loss = float(np.sum(np.clip(-_chop_pnl, 0.0, None)))
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

        _hy_arr = aligned.get("hmm_prob_crisis") if aligned else None
        if _hy_arr is not None:
            try:
                _hy_np = np.asarray(_hy_arr, dtype=np.float64)
                if _hy_np.ndim > 1:
                    _hy_np = _hy_np[:, 0]
                leg_crisis_mean.append(float(np.nanmean(_hy_np)))
            except Exception:
                leg_crisis_mean.append(0.0)
        else:
            leg_crisis_mean.append(0.0)

        leg_log_tw.append(log_ret)
        leg_mdds.append(mdd)
        leg_mdd_duration_days.append(mdd_duration_days)
        leg_cvar_pct.append(cvar_pct)
        leg_trade_counts.append(float(n_tr))
        leg_long_counts.append(n_long)
        leg_short_counts.append(n_short)
        leg_exposures.append(b_exposure)

        if leg_idx >= 1 and trial is not None:
            cum_log_tw = float(np.sum(leg_log_tw))
            max_leg_mdd = float(np.max(leg_mdds))
            if (not np.isfinite(cum_log_tw)) or (not np.isfinite(max_leg_mdd)):
                break
            if abs(cum_log_tw) > 100.0:
                break
            if cum_log_tw < -0.25 or max_leg_mdd > liq_mdd_thr:
                break

            if trial is not None and len(trial.study.directions) == 1:
                trial.report(float(np.mean(leg_log_tw)), step=leg_idx)
                if trial.should_prune():
                    set_trial_event_attrs(
                        trial,
                        status="pruned",
                        reason="trial_should_prune",
                        stage="awf_intermediate_report",
                        step=int(leg_idx),
                        metrics={"mean_leg_log_tw": float(np.mean(leg_log_tw))},
                    )
                    raise optuna.TrialPruned()

    leg_arr = np.asarray(leg_log_tw, dtype=np.float64)
    n_legs_done = leg_arr.size
    all_trades = (
        np.vstack(all_trades_chunks) if all_trades_chunks
        else np.zeros((0, 10), dtype=np.float64)
    )

    avg_trades_agg = float(np.mean(leg_trade_counts)) if leg_trade_counts else 0.0
    worst_mdd_legs = float(max(leg_mdds, default=100.0))
    avg_exposure = float(np.mean(leg_exposures)) if leg_exposures else 0.0
    total_long = sum(leg_long_counts)
    total_short = sum(leg_short_counts)
    total_dir = total_long + total_short
    minority = float(min(total_long, total_short) / total_dir) if total_dir > 0 else 0.0

    l_pf_agg, s_pf_agg = 1.0, 1.0
    if all_trades.size > 0:
        pnl_arr = all_trades[:, 6].astype(np.float64, copy=False)
        dir_arr = all_trades[:, 3]
        l_mask = dir_arr == 1.0
        s_mask = dir_arr == -1.0
        l_pnl = pnl_arr[l_mask]
        l_win = float(np.sum(l_pnl[l_pnl > 0.0]))
        l_loss = float(np.sum(np.abs(l_pnl[l_pnl < 0.0])))
        l_pf_agg = l_win / max(l_loss, 1e-9) if l_loss > 0 else 1.0
        s_pnl = pnl_arr[s_mask]
        s_win = float(np.sum(s_pnl[s_pnl > 0.0]))
        s_loss = float(np.sum(np.abs(s_pnl[s_pnl < 0.0])))
        s_pf_agg = s_win / max(s_loss, 1e-9) if s_loss > 0 else 1.0

    k_legs_n = float(max(n_legs_done, 1))
    mu_log = float(np.mean(leg_arr)) if leg_arr.size > 0 else -10.0
    worst_leg = float(np.min(leg_arr)) if leg_arr.size > 0 else -10.0
    med_leg = float(np.median(leg_arr)) if leg_arr.size > 0 else -10.0
    awf_pos_frac = float(np.sum(leg_arr > 0.0)) / k_legs_n
    dsr_awf = calc_gate1_dsr_from_path_log_tw(
        leg_arr,
        ctx.tf,
        float(gate_stat_ref),
        float(n_trials_eff),
    )

    sig_awf_diag = float(np.std(leg_arr, ddof=1)) if leg_arr.size >= 2 else 0.0
    leg_l_pf_mean = float(np.mean(leg_l_pf)) if leg_l_pf else l_pf_agg
    leg_s_pf_mean = float(np.mean(leg_s_pf)) if leg_s_pf else s_pf_agg
    _awf_pf_agg, ev_cost_ratio = _pf_and_ev_cost_from_trades(all_trades)
    funding_drag_ratio, funding_drag_basis = _funding_drag_ratio_from_trades(all_trades)
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
    flip_rate_proxy = float(np.mean(leg_flip_proxy)) if leg_flip_proxy else 0.0

    step2_enabled = bool(cfg.get("FUTURES_STEP2_REGIME_DEPLOY_ENABLED", False))
    if step2_enabled:
        chop_loss_w = float(cfg.get("FUTURES_STEP2_OBJ_CHOP_LOSS_W", 0.25))
        chop_trade_w = float(cfg.get("FUTURES_STEP2_OBJ_CHOP_TRADE_W", 0.15))
        flip_w = float(cfg.get("FUTURES_STEP2_OBJ_FLIP_W", 0.10))
        loss_thr = float(cfg.get("FUTURES_STEP2_CHOP_LOSS_SHARE_MAX", 0.60))
        trade_thr = float(cfg.get("FUTURES_STEP2_CHOP_TRADE_SHARE_MAX", 0.70))
        flip_thr = float(cfg.get("FUTURES_STEP2_FLIP_RATE_PROXY_MAX", 0.75))
        excess_loss = max(0.0, chop_loss_share - loss_thr)
        excess_trade = max(0.0, chop_trade_share - trade_thr)
        excess_flip = max(0.0, flip_rate_proxy - flip_thr)
        robust_val -= chop_loss_w * excess_loss + chop_trade_w * excess_trade + flip_w * excess_flip

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
