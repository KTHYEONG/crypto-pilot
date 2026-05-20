from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from .data_utils import (
    _dataframe_to_symbol_arrays_extended,
    _slice_symbol_arrays_view,
    _span_days_ref_slice,
)
from .signal_cache import (
    _ARRAYS_CACHE_MAXSIZE,
    _arrays_cache,
    _build_signal_cache_key,
    _cache_lock,
    _dataset_fingerprint_from_df,
    _SignalCacheKey,
    get_or_compute_signals,
)

try:
    from filelock import FileLock
except ImportError:
    FileLock = None  # type: ignore[misc, assignment]

from config.opt_config import OPT_SPOT_CONFIG, get_spot_effective_independent_trials
from config.settings import SPOT_INITIAL_BALANCE
from src.domain.spot.opt_spot_utils.cv_utils import (
    CPCVPath,
    build_cpcv_test_paths_with_fallback,
)
from src.domain.spot.opt_spot_utils.exit_family_prior import exit_family_prior_penalty
from src.domain.spot.opt_spot_utils.metrics import (
    calc_mdd_from_equity,
    calc_sortino_from_equity,
    calc_tail_ratio_from_equity,
    compute_dsr_from_path_values,
    cvar_loss_pct_from_simple_returns,
    mean_of_worst_quartile,
    portfolio_cagr_pct_from_equity,
    probabilistic_sharpe_ratio,
)
from src.domain.spot.opt_spot_utils.opt_params import suggest_params_spot
from src.domain.spot.portfolio_shared_cash import run_shared_cash_multi_symbol
from src.domain.spot.strategies_spot import UltimateSpotStrategy

_logger: logging.Logger = logging.getLogger("opt_spot")

# Optuna TPE `constraints_func`: each value <= 0 means satisfied (Gardner-style soft constraints).
SPOT_OBJECTIVE_CONSTRAINT_DIM: int = 9


def spot_frozen_trial_constraints(trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
    raw = trial.user_attrs.get("spot_constraint_values")
    if isinstance(raw, (list, tuple)) and len(raw) == SPOT_OBJECTIVE_CONSTRAINT_DIM:
        return tuple(float(x) for x in raw)
    return tuple(1.0 for _ in range(SPOT_OBJECTIVE_CONSTRAINT_DIM))


def compute_embargo_bars(tf: str, longest_indicator_period: int = 150) -> int:
    fixed_min: dict[str, int] = {"4h": 24}
    ratio_map: dict[str, float] = {"4h": 0.05}
    ratio: float = ratio_map.get(tf, 0.03)
    return max(fixed_min.get(tf, 2), int(longest_indicator_period * ratio))


EMBARGO_BARS: dict[str, int] = {
    "4h": compute_embargo_bars("4h"),
}


_SPOT_OBJECTIVE_CAGR_WEIGHT: float = 0.0  # Abandoning additive CAGR
_SPOT_OBJECTIVE_MIN_TRADES_HARD: float = 10.0  # Min trades per path to even consider
_SPOT_OBJECTIVE_MIN_TRADES_SOFT: float = 40.0  # Target trades for statistical robustness
_SPOT_OBJECTIVE_LOG_TWR_WEIGHT: float = 1.0
_SPOT_OBJECTIVE_PATH_CV_PENALTY: float = 0.75  # Path CV penalty (Generalized Kelly λ≈0.75)


def _compute_path_gmgr_high_moments(equity: np.ndarray) -> float:
    """Geometric mean growth proxy with high-moment correction on log returns:
    r_mean - σ²/2 + S·σ³/6 - K·σ⁴/24. Returns winsorized at [p1, p99].
    """
    eq = np.asarray(equity, dtype=np.float64).ravel()
    if eq.size < 3:
        return -1.0
    eq = np.maximum(eq, 1e-12)
    r = np.diff(np.log(eq))
    r = r[np.isfinite(r)]
    if r.size < 5:
        return float(np.mean(r)) if r.size > 0 else -1.0
    lo, hi = np.percentile(r, [1.0, 99.0])
    rw = np.clip(r, lo, hi)
    mu = float(np.mean(rw))
    sd = float(np.std(rw, ddof=1))
    if sd < 1e-12:
        return mu
    m3 = float(np.mean((rw - mu) ** 3))
    m4 = float(np.mean((rw - mu) ** 4))
    skew = m3 / (sd**3)
    ex_kurt = m4 / (sd**4) - 3.0
    return float(mu - (sd**2) / 2.0 + skew * (sd**3) / 6.0 - ex_kurt * (sd**4) / 24.0)


def _compute_ulcer_index(equity: np.ndarray) -> float:
    """Ulcer Index: sqrt(mean(D_i^2)), D_i = (peak_i - eq_i) / peak_i * 100."""
    eq = np.asarray(equity, dtype=np.float64).ravel()
    if eq.size < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    safe = np.where(peak > 1e-12, peak, 1.0)
    drawdown_pct = (peak - eq) / safe * 100.0
    return float(np.sqrt(np.mean(drawdown_pct * drawdown_pct)))


def _cpcv_path_compound_raw_log_tw(
    segments: list[tuple[int, int]],
    *,
    prebuilt_full_arrays: dict[str, dict[str, np.ndarray]],
    symbols: list[str],
    params: dict[str, Any],
    is_off: int,
    ref_df: pd.DataFrame,
    max_slots: int,
    warmup_bars: int,
    concurrency_penalty_scale: float = 1.0,
) -> float:
    """Sum of raw log terminal-wealth ratios per segment (matches objective_spot CPCV path metric)."""
    seg_raw_log_tw: list[float] = []
    running_balance = float(SPOT_INITIAL_BALANCE)
    for test_start, test_end in segments:
        abs_start = is_off + int(test_start)
        abs_end = is_off + int(test_end)
        slice_start = max(0, abs_start - 1)
        slice_end = min(len(ref_df), abs_end)
        if slice_end - slice_start < 5:
            continue
        symbol_arrays: dict[str, dict[str, np.ndarray]] = {}
        rank_scores: dict[str, np.ndarray] = {}
        for sym in symbols:
            symbol_arrays[sym] = _slice_symbol_arrays_view(
                prebuilt_full_arrays[sym], slice_start, slice_end
            )
            rs = symbol_arrays[sym].get("slot_rank_score")
            if rs is not None:
                rank_scores[sym] = rs
        execution_start_idx = max(1, abs_start - slice_start)
        segment_initial = max(running_balance, 1e-9)
        result = run_shared_cash_multi_symbol(
            symbol_arrays,
            symbols,
            params,
            initial_balance=segment_initial,
            max_concurrent_positions=max_slots,
            rank_scores=rank_scores if rank_scores else None,
            warmup_bars=warmup_bars,
            execution_start_idx=execution_start_idx,
            allow_python_fallback=False,
            concurrency_penalty_scale=float(concurrency_penalty_scale),
        )
        eq = result.equity_curve
        if eq.size == 0:
            twr = 1.0
        else:
            twr = max(float(result.final_balance / segment_initial), 1e-9)
        raw_log_tw = float(np.log(twr))
        seg_raw_log_tw.append(raw_log_tw)
        running_balance = max(float(result.final_balance), 1e-9)
    if not seg_raw_log_tw:
        return float("nan")
    return float(np.sum(seg_raw_log_tw))


def objective_spot(
    trial: optuna.Trial,
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf_target: str,
    *,
    space: dict[str, dict[str, Any]],
    mode: str = "single",
    project_root: str | None = None,
    prebuilt_cpcv_bundle: tuple[list[CPCVPath], int, int] | None = None,
    signal_disk_cache_root: Path | None = None,
) -> float:
    params: dict[str, Any] = suggest_params_spot(trial, space, tf_target)
    tf: str = tf_target
    strategy: UltimateSpotStrategy = UltimateSpotStrategy(name="OptSpot", params=params)

    ref_sym = symbols[0]
    is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
    ref_df = data_maps[ref_sym][tf]
    if ref_df is None or ref_df.empty:
        raise optuna.TrialPruned()
    ref_len = len(ref_df) - is_off
    if ref_len < 200:
        raise optuna.TrialPruned()

    embargo = int(EMBARGO_BARS.get(tf, 0))
    if prebuilt_cpcv_bundle is not None:
        cpcv_paths, nb_cpcv, k_cpcv = prebuilt_cpcv_bundle
    else:
        cpcv_paths, nb_cpcv, k_cpcv = build_cpcv_test_paths_with_fallback(ref_len, embargo=embargo)
    if not cpcv_paths:
        raise optuna.TrialPruned()
    n_independent_paths = max(2, nb_cpcv // k_cpcv)

    cache_root: Path | None = signal_disk_cache_root
    if cache_root is None and project_root is not None:
        cache_root = Path(project_root) / ".spot_signal_cache"

    strategy._portfolio_eval_ctx = {"data_maps": data_maps, "symbols": list(symbols), "tf": tf}
    try:
        full_signal_dfs: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            target_df_full: pd.DataFrame | None = data_maps.get(sym, {}).get(tf)
            if target_df_full is None or target_df_full.empty:
                continue
            fp = _dataset_fingerprint_from_df(target_df_full)
            cache_key: _SignalCacheKey = _build_signal_cache_key(
                params, sym, tf, len(target_df_full), fp
            )
            full_signal_dfs[sym] = get_or_compute_signals(
                cache_key, target_df_full, strategy, disk_cache_root=cache_root
            )

        if len(full_signal_dfs) != len(symbols):
            raise optuna.TrialPruned()

        prebuilt_full_arrays: dict[str, dict[str, np.ndarray]] = {}
        for sym in symbols:
            target_df_full = data_maps.get(sym, {}).get(tf)
            if target_df_full is None or target_df_full.empty:
                raise optuna.TrialPruned()
            fp = _dataset_fingerprint_from_df(target_df_full)
            sig_key = _build_signal_cache_key(params, sym, tf, len(target_df_full), fp)
            with _cache_lock:
                if sig_key in _arrays_cache:
                    _arrays_cache.move_to_end(sig_key)
                    prebuilt_full_arrays[sym] = _arrays_cache[sig_key]
                    continue
            arrs = _dataframe_to_symbol_arrays_extended(full_signal_dfs[sym])
            with _cache_lock:
                while len(_arrays_cache) >= _ARRAYS_CACHE_MAXSIZE:
                    _arrays_cache.popitem(last=False)
                _arrays_cache[sig_key] = arrs
            prebuilt_full_arrays[sym] = arrs

        max_slots = int(OPT_SPOT_CONFIG.get("SPOT_MAX_CONCURRENT_POSITIONS", 3))
        min_seg_trades = int(OPT_SPOT_CONFIG.get("SPOT_MIN_TRADES_PER_CPCV_SEGMENT", 4))
        # Signals are pre-computed on full IS; per-segment re-warmup would skip most of each test block.
        warmup_bars = 0

        cfg = OPT_SPOT_CONFIG
        seg_fail_pen = float(cfg.get("SPOT_SEGMENT_TRADE_FAIL_PENALTY", 2.0))
        sortino_eps = float(cfg.get("SPOT_OBJECTIVE_SORTINO_EPS", 1e-6))
        sortino_ratio_cap = float(cfg.get("SPOT_OBJECTIVE_SORTINO_RATIO_CAP", 1.0e6))
        path_sortino_clip = float(cfg.get("SPOT_OBJECTIVE_PATH_SORTINO_CLIP", 500.0))

        path_compound_log_tw: list[float] = []
        path_compound_raw_log_tw: list[float] = []
        path_compound_tw_ratio: list[float] = []
        path_sortino_vals: list[float] = []
        path_worst_mdd: list[float] = []
        path_max_cvar: list[float] = []
        path_trades: list[int] = []
        path_tail_ratios: list[float] = []
        path_pfs: list[float] = []
        path_gmgr: list[float] = []
        path_ui: list[float] = []
        path_calmars: list[float] = []
        path_cagrs: list[float] = []
        path_regime_rates: list[float] = []
        total_sym_pnl = np.zeros(len(symbols), dtype=np.float64)

        for path_idx, path in enumerate(cpcv_paths):
            seg_log_tw: list[float] = []
            seg_raw_log_tw: list[float] = []
            seg_tw_ratio: list[float] = []
            seg_mdds: list[float] = []
            seg_cvars: list[float] = []
            seg_pfs: list[float] = []
            path_total_trades = 0
            path_regime_on = 0
            path_regime_len = 0
            running_balance = float(SPOT_INITIAL_BALANCE)
            path_eq_chunks: list[np.ndarray] = []
            span_path_days = 0.0
            for test_start, test_end in path:
                abs_start = is_off + int(test_start)
                abs_end = is_off + int(test_end)
                slice_start = max(0, abs_start - 1)
                slice_end = min(len(ref_df), abs_end)
                if slice_end - slice_start < 5:
                    continue
                span_path_days += _span_days_ref_slice(ref_df, abs_start, abs_end)
                symbol_arrays: dict[str, dict[str, np.ndarray]] = {}
                rank_scores: dict[str, np.ndarray] = {}
                for sym in symbols:
                    symbol_arrays[sym] = _slice_symbol_arrays_view(
                        prebuilt_full_arrays[sym], slice_start, slice_end
                    )
                    rs = symbol_arrays[sym].get("slot_rank_score")
                    if rs is not None:
                        rank_scores[sym] = rs

                ref_seg = symbol_arrays.get(ref_sym)
                if ref_seg is not None:
                    rs_arr = ref_seg.get("regime_state")
                    if rs_arr is not None and rs_arr.size > 0:
                        path_regime_on += int(np.sum(rs_arr > 0.1))
                        path_regime_len += int(rs_arr.size)

                execution_start_idx = max(1, abs_start - slice_start)
                segment_initial = max(running_balance, 1e-9)
                try:
                    result = run_shared_cash_multi_symbol(
                        symbol_arrays,
                        symbols,
                        params,
                        initial_balance=segment_initial,
                        max_concurrent_positions=max_slots,
                        rank_scores=rank_scores if rank_scores else None,
                        warmup_bars=warmup_bars,
                        execution_start_idx=execution_start_idx,
                        allow_python_fallback=False,
                        concurrency_penalty_scale=1.0,
                    )
                except Exception as exc:
                    _logger.warning(
                        "Shared-cash Numba path failed (allow_python_fallback=False): %s",
                        exc,
                        exc_info=True,
                    )
                    raise
                eq = result.equity_curve
                path_total_trades += int(result.total_trades)
                if hasattr(result, "per_symbol_pnl"):
                    total_sym_pnl += result.per_symbol_pnl
                if eq.size == 0:
                    twr = 1.0
                else:
                    twr = max(float(result.final_balance / segment_initial), 1e-9)
                raw_log_tw = float(np.log(twr))
                log_tw = raw_log_tw
                if eq.size > 0:
                    eq0 = float(eq[0]) if float(eq[0]) > 1e-12 else segment_initial
                    scale = float(segment_initial) / eq0
                    scaled_eq = eq.astype(np.float64, copy=False) * scale
                    if not path_eq_chunks:
                        path_eq_chunks.append(scaled_eq)
                    else:
                        path_eq_chunks.append(scaled_eq[1:])
                if eq.size > 1:
                    mdd_seg = float(calc_mdd_from_equity(eq))
                    seg_mdds.append(mdd_seg)
                    seg_cvars.append(float(cvar_loss_pct_from_simple_returns(eq)))
                else:
                    seg_mdds.append(0.0)
                    seg_cvars.append(0.0)

                pnl = result.pnl_array
                pos_pnl = float(np.sum(pnl[pnl > 0.0]))
                neg_pnl = float(np.abs(np.sum(pnl[pnl < 0.0])))
                # Asymptotic Cap for zero-loss singularity to avoid penalizing perfect segments
                if neg_pnl > 1e-12:
                    seg_pf = pos_pnl / neg_pnl
                elif pos_pnl > 1e-12:
                    seg_pf = 3.0  # Theoretical cap for stable alpha
                else:
                    seg_pf = 1.0  # Neutral for zero trades
                seg_pfs.append(seg_pf)

                if int(result.total_trades) < min_seg_trades:
                    log_tw -= seg_fail_pen

                seg_raw_log_tw.append(raw_log_tw)
                seg_log_tw.append(log_tw)
                seg_tw_ratio.append(twr)
                running_balance = max(float(result.final_balance), 1e-9)

            if not seg_log_tw:
                raise optuna.TrialPruned()
            if path_regime_len > 0:
                path_regime_rates.append(path_regime_on / float(path_regime_len))
            path_compound_log_tw.append(float(np.sum(seg_log_tw)))
            path_compound_raw_log_tw.append(float(np.sum(seg_raw_log_tw)))
            path_compound_tw_ratio.append(float(np.prod(seg_tw_ratio)) if seg_tw_ratio else 1.0)
            path_worst_mdd.append(float(np.max(seg_mdds)) if seg_mdds else 0.0)

            # HARD HURDLE: If any path exceeds MDD limit, prune immediately.
            path_mdd_limit = float(OPT_SPOT_CONFIG.get("SPOT_HOLDOUT_MDD_LIMIT_PCT", 20.0))
            if path_worst_mdd[-1] > path_mdd_limit:
                raise optuna.TrialPruned()

            # HARD HURDLE: Min trades per path for statistical relevance.
            if path_total_trades < _SPOT_OBJECTIVE_MIN_TRADES_HARD:
                raise optuna.TrialPruned()

            path_trades.append(path_total_trades)
            path_pfs.append(float(np.mean(seg_pfs)) if seg_pfs else 1.0)

            path_eq = (
                np.concatenate(path_eq_chunks) if path_eq_chunks else np.array([], dtype=np.float64)
            )
            path_level_tail = (
                float(calc_tail_ratio_from_equity(path_eq)) if path_eq.size >= 2 else 1.0
            )
            path_tail_ratios.append(path_level_tail)
            span_for_sortino = max(span_path_days, 1.0)
            raw_ps = (
                float(calc_sortino_from_equity(path_eq, span_for_sortino))
                if path_eq.size >= 2
                else 0.0
            )
            if not np.isfinite(raw_ps):
                raw_ps = 0.0
            path_sortino = float(np.clip(raw_ps, -path_sortino_clip, path_sortino_clip))
            path_sortino_vals.append(path_sortino)

            if path_eq.size >= 2:
                path_gmgr.append(_compute_path_gmgr_high_moments(path_eq))
                path_ui.append(_compute_ulcer_index(path_eq))
                span_c = max(span_path_days, 1.0)
                pc_cagr = float(portfolio_cagr_pct_from_equity(path_eq, span_c))
                pc_mdd = float(calc_mdd_from_equity(path_eq))
                path_calmars.append(pc_cagr / max(pc_mdd, 0.01))
                path_cagrs.append(pc_cagr)
            else:
                path_gmgr.append(-1.0)
                path_ui.append(0.0)
                path_calmars.append(0.0)
                path_cagrs.append(0.0)

            # Intermediate pruning: mean CPCV path log-TWR so far (enables MedianPruner / PatientPruner).
            if path_idx >= 4 and path_compound_raw_log_tw:
                intermediate = float(np.mean(path_compound_raw_log_tw))
                trial.report(intermediate, step=path_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mean_regime_on_rate = float(np.mean(path_regime_rates)) if path_regime_rates else 1.0
        cpcv_mean_path_cagr_pct = float(np.mean(path_cagrs)) if path_cagrs else 0.0
        trial.set_user_attr("cpcv_mean_path_cagr_pct", float(cpcv_mean_path_cagr_pct))

        mean_log_tw = float(np.mean(path_compound_raw_log_tw))
        mean_penalized_log_tw = float(np.mean(path_compound_log_tw))
        cvar25_log = float(mean_of_worst_quartile(path_compound_raw_log_tw))

        worst_path_tw = (
            float(np.percentile(np.asarray(path_compound_tw_ratio, dtype=np.float64), 10.0))
            if path_compound_tw_ratio
            else 1.0
        )
        mean_path_sortino = float(np.mean(path_sortino_vals)) if path_sortino_vals else 0.0
        std_path_sortino = (
            float(np.std(path_sortino_vals, ddof=1)) if len(path_sortino_vals) > 1 else 0.0
        )
        sortino_ratio = mean_path_sortino / (std_path_sortino + sortino_eps)
        sortino_ratio = float(np.clip(sortino_ratio, -sortino_ratio_cap, sortino_ratio_cap))

        gmgr_arr = np.asarray(path_gmgr, dtype=np.float64)
        ui_arr = np.asarray(path_ui, dtype=np.float64)
        p10_gmgr = float(np.percentile(gmgr_arr, 10.0)) if gmgr_arr.size else -1.0
        cagr_arr = np.asarray(path_cagrs, dtype=np.float64)
        p10_cagr = float(np.percentile(cagr_arr, 10.0)) if cagr_arr.size else -100.0
        max_ui = float(np.max(ui_arr)) if ui_arr.size else 0.0
        n_trades_mean = float(np.mean(path_trades)) if path_trades else 0.0

        path_arr = np.asarray(path_compound_raw_log_tw, dtype=np.float64)
        n_paths_ct = int(path_arr.size)
        mu_paths = float(np.mean(path_arr)) if path_arr.size > 0 else -10.0
        sd_paths = float(np.std(path_arr, ddof=1)) if path_arr.size > 1 else 10.0
        if n_paths_ct >= 2:
            gate1_sqn = (
                math.sqrt(float(n_paths_ct)) * mu_paths / sd_paths if sd_paths > 1e-12 else 0.0
            )
            cv_paths = sd_paths / (abs(mu_paths) + 1e-6)
        else:
            gate1_sqn = 0.0
            cv_paths = 10.0

        # DSR (Deflated Sharpe Ratio) for path log-TWR stability
        n_done = trial.number + 1
        n_startup = int(OPT_SPOT_CONFIG.get("tpe_n_startup_trials", 100))
        n_eff = get_spot_effective_independent_trials(n_done, n_startup)
        path_vals = [float(x) for x in path_compound_raw_log_tw]
        dsr_val = float(compute_dsr_from_path_values(path_vals, n_eff))

        if n_paths_ct >= 2 and sd_paths > 1e-12:
            sr_est = mu_paths / sd_paths
            psr_val = float(probabilistic_sharpe_ratio(sr_est, n_obs=n_paths_ct))
        else:
            psr_val = 0.0

        if n_paths_ct >= 4 and dsr_val < -1.0:
            raise optuna.TrialPruned()

        min_path_tw_ratio = (
            float(np.min(np.asarray(path_compound_tw_ratio, dtype=np.float64)))
            if path_compound_tw_ratio
            else 0.0
        )
        psr_floor = float(cfg.get("SPOT_CONSTRAINT_PSR_FLOOR", 0.02))
        dsr_floor = max(
            float(cfg.get("SPOT_CONSTRAINT_DSR_FLOOR", 0.0)),
            float(cfg.get("SPOT_DISCOVERY_DSR_MIN", 0.35)),
        )
        min_tw_req = float(cfg.get("SPOT_CONSTRAINT_MIN_PATH_TW_RATIO", 0.88))
        min_mean_tr_req = float(cfg.get("SPOT_CONSTRAINT_MIN_MEAN_TRADES", 12.0))
        mean_path_pf_gate = float(np.mean(path_pfs)) if path_pfs else 1.0
        mean_tail_gate = float(np.mean(path_tail_ratios)) if path_tail_ratios else 0.0
        worst25_cal_gate = (
            float(np.percentile(np.asarray(path_calmars, dtype=np.float64), 25.0))
            if path_calmars
            else 0.0
        )
        min_mean_pf_req = float(cfg.get("SPOT_CONSTRAINT_MIN_MEAN_PF", 1.0))
        min_mean_tail_req = float(cfg.get("SPOT_CONSTRAINT_MIN_MEAN_PATH_TAIL", 1.0))
        min_w25_cal_req = float(cfg.get("SPOT_CONSTRAINT_MIN_WORST25_CALMAR", 0.0))
        min_p10_cagr_req = float(cfg.get("SPOT_MIN_P10_GMGR_CAGR_PCT", 5.0))
        min_regime_floor = float(cfg.get("SPOT_MIN_REGIME_ON_RATE", 0.15))
        if n_paths_ct >= 3:
            c_w25 = float(min_w25_cal_req - worst25_cal_gate) if min_w25_cal_req > 1e-12 else 0.0
            c_reg = float(min_regime_floor - mean_regime_on_rate) if path_regime_rates else 0.0
            constraint_vec = (
                float(psr_floor - psr_val),
                float(dsr_floor - dsr_val),
                float(min_p10_cagr_req - p10_cagr),
                float(min_tw_req - min_path_tw_ratio),
                float(min_mean_tr_req - n_trades_mean),
                float(min_mean_pf_req - mean_path_pf_gate),
                float(min_mean_tail_req - mean_tail_gate),
                c_w25,
                c_reg,
            )
        else:
            constraint_vec = tuple(0.0 for _ in range(SPOT_OBJECTIVE_CONSTRAINT_DIM))

        trial.set_user_attr("spot_constraint_values", list(constraint_vec))
        spot_feasible = all(c <= 0.0 for c in constraint_vec)
        trial.set_user_attr("spot_constraints_feasible", spot_feasible)

        infeasible_pen = float(cfg.get("SPOT_OBJECTIVE_INFEASIBLE_RETURN", -1e9))
        if n_paths_ct >= 3 and not spot_feasible:
            trial.set_user_attr("objective_final", float(infeasible_pen))
            return float(infeasible_pen)

        # Kelly-CVaR: E[log W] with coherent tail penalty (replaces CRRA CEQ + z_conf LPM).
        path_log_tw_arr = np.asarray(path_compound_raw_log_tw, dtype=np.float64)
        path_returns = path_log_tw_arr

        mean_log_tw_k = float(np.mean(path_log_tw_arr)) if path_log_tw_arr.size > 0 else 0.0
        w_mean = float(cfg.get("SPOT_OBJECTIVE_W_MEAN_LOG_TW", 0.7))
        w_mean = float(np.clip(w_mean, 0.0, 1.0))
        w_p10 = 1.0 - w_mean
        p10_log_tw_path = (
            float(np.percentile(path_log_tw_arr, 10.0))
            if path_log_tw_arr.size >= 10
            else mean_log_tw_k
        )
        kelly_obj = w_mean * mean_log_tw_k + w_p10 * p10_log_tw_path

        cvar_alpha = float(cfg.get("SPOT_CPCV_CVAR_ALPHA", 0.10))
        cvar_thr = float(cfg.get("SPOT_CPCV_CVAR_THRESHOLD", 0.05))
        cvar_weight = float(cfg.get("SPOT_CPCV_CVAR_WEIGHT", 0.80))
        sorted_rtns = np.sort(path_log_tw_arr)
        n_paths_log = int(sorted_rtns.size)
        if n_paths_log > 0:
            k_worst = max(2, int(n_paths_log * cvar_alpha))
            k_worst = min(k_worst, n_paths_log)
            cvar_val = float(-np.mean(sorted_rtns[:k_worst]))
        else:
            cvar_val = 0.0
        cvar_pen = max(0.0, cvar_val - cvar_thr) * cvar_weight

        stat_lcb = kelly_obj - cvar_pen

        # Refined Penalties (Minimal & Targetted)
        concentration_pen = max(0.0, cv_paths - 1.5) * 0.15

        total_abs_pnl = float(np.sum(np.abs(total_sym_pnl))) + 1e-9
        sym_shares = total_sym_pnl / total_abs_pnl
        hhi = float(np.sum(sym_shares**2))
        n_sym = max(1, len(total_sym_pnl))
        hhi_equal = 1.0 / float(n_sym)
        hhi_penalty = max(0.0, hhi - 3.0 * hhi_equal) * 0.50

        trade_count_pen = max(0.0, _SPOT_OBJECTIVE_MIN_TRADES_SOFT - n_trades_mean) * 0.15

        _tail_ratio_target = float(
            cfg.get("SPOT_OBJECTIVE_TAIL_RATIO_TARGET", cfg.get("SPOT_GATE1_TAIL_RATIO_MIN", 1.1))
        )
        w_tail_obj = float(cfg.get("SPOT_OBJECTIVE_W_TAIL_RATIO", 0.60))
        tail_shortfall = max(0.0, _tail_ratio_target - mean_tail_gate)
        tail_bonus = math.log(max(mean_tail_gate, 0.5)) * w_tail_obj
        tail_penalty = tail_shortfall * 0.80
        tail_ratio_reward = tail_bonus - tail_penalty

        # Metrics for Optuna attributes and transparency
        p25_log = float(np.percentile(path_returns, 25.0)) if path_returns.size else -10.0
        p10_log = float(np.percentile(path_returns, 10.0)) if path_returns.size else -10.0
        total_soft_penalty = trade_count_pen + hhi_penalty

        temporal_decay_pen = 0.0
        n_p_raw = len(path_compound_raw_log_tw)
        temporal_decay_weight = float(
            cfg.get("SPOT_CPCV_TEMPORAL_LAMBDA", cfg.get("SPOT_TEMPORAL_DECAY_PEN_WEIGHT", 0.20))
        )
        if n_p_raw >= 4 and temporal_decay_weight > 0.0:
            k_recent = max(2, n_p_raw // 4)
            tw_raw_arr = np.asarray(path_compound_raw_log_tw, dtype=np.float64)
            all_mean_tw = float(np.mean(tw_raw_arr))
            recent_mean_tw = float(np.mean(tw_raw_arr[-k_recent:]))
            temporal_decay = max(0.0, all_mean_tw - recent_mean_tw)
            temporal_decay_pen = temporal_decay * temporal_decay_weight

        objective_final = float(
            stat_lcb  # Core Risk-Adjusted Growth
            + tail_ratio_reward  # CPCV path tail ratio (soft reward toward gate)
            - concentration_pen  # Path stability
            - hhi_penalty  # Asset diversity
            - trade_count_pen  # Statistical significance
            - temporal_decay_pen  # Recent CPCV paths must not collapse vs earlier paths
        )

        fwd_weight = float(cfg.get("SPOT_RECENT_IS_GATE_WEIGHT", 0.25))
        fwd_pen = 0.0
        recent_is_log_tw = float("nan")
        if fwd_weight > 0.0:
            n_is_bars = int(ref_len)
            recent_start = is_off + int(n_is_bars * 0.75)
            recent_end = is_off + n_is_bars
            slice_rs = max(0, recent_start - 1)
            slice_re = min(len(ref_df), recent_end)
            if slice_re - slice_rs >= 5:
                symbol_arrays_fwd: dict[str, dict[str, np.ndarray]] = {}
                rank_scores_fwd: dict[str, np.ndarray] = {}
                for sym_fwd in symbols:
                    symbol_arrays_fwd[sym_fwd] = _slice_symbol_arrays_view(
                        prebuilt_full_arrays[sym_fwd], slice_rs, slice_re
                    )
                    rs_f = symbol_arrays_fwd[sym_fwd].get("slot_rank_score")
                    if rs_f is not None:
                        rank_scores_fwd[sym_fwd] = rs_f
                exec_fwd = max(1, recent_start - slice_rs)
                result_fwd = run_shared_cash_multi_symbol(
                    symbol_arrays_fwd,
                    symbols,
                    params,
                    initial_balance=float(SPOT_INITIAL_BALANCE),
                    max_concurrent_positions=max_slots,
                    rank_scores=rank_scores_fwd if rank_scores_fwd else None,
                    warmup_bars=warmup_bars,
                    execution_start_idx=exec_fwd,
                    allow_python_fallback=False,
                    concurrency_penalty_scale=1.0,
                )
                recent_is_log_tw = float(
                    np.log(max(result_fwd.final_balance / float(SPOT_INITIAL_BALANCE), 1e-9))
                )
                fwd_pen = max(0.0, -recent_is_log_tw) * fwd_weight
                objective_final = float(objective_final - fwd_pen)

        prior_scale = float(cfg.get("SPOT_EXIT_FAMILY_PRIOR_SCALE", 1.0))
        prior_pen = (
            float(
                exit_family_prior_penalty(
                    str(params.get("SIGNAL_TYPE", "ADX_BREAKOUT")),
                    str(params.get("EXIT_FAMILY", "BALANCED")),
                )
            )
            * prior_scale
        )
        objective_final = float(objective_final - prior_pen)

        worst_seg_mdd = float(np.max(path_worst_mdd)) if path_worst_mdd else 0.0
        mean_path_return_pct = (math.exp(mean_log_tw) - 1.0) * 100.0
        mean_path_calmar = float(np.mean(path_calmars)) if path_calmars else 0.0

        trial.set_user_attr("objective_final", objective_final)
        trial.set_user_attr("growth_score", float(objective_final))
        trial.set_user_attr("temporal_decay_pen", float(temporal_decay_pen))
        trial.set_user_attr("exit_family_prior_penalty", float(prior_pen))
        trial.set_user_attr("p10_gmgr", float(p10_gmgr))
        trial.set_user_attr("cpcv_p25_log_tw", float(p25_log))
        trial.set_user_attr("cpcv_path_cv", float(cv_paths))
        trial.set_user_attr("kelly_obj", float(kelly_obj))
        trial.set_user_attr("cvar_val", float(cvar_val))
        trial.set_user_attr("objective_ceq_growth", float(kelly_obj))
        trial.set_user_attr("recent_is_log_tw", float(recent_is_log_tw))
        trial.set_user_attr("recent_is_gate_penalty", float(fwd_pen))
        trial.set_user_attr("objective_geometric_mean_log", float(mu_paths))
        trial.set_user_attr("objective_mean_path_calmar", float(mean_path_calmar))
        trial.set_user_attr("max_ulcer_index", float(max_ui))
        trial.set_user_attr("objective_soft_penalty", float(total_soft_penalty))
        trial.set_user_attr("dsr_paths", dsr_val)
        trial.set_user_attr("gate1_sqn", float(gate1_sqn))
        trial.set_user_attr("psr_paths", float(psr_val))
        trial.set_user_attr("gate1_path_sortino", float(mean_path_sortino))
        trial.set_user_attr("cpcv_path_tail_ratio", float(mean_tail_gate))
        trial.set_user_attr("cpcv_mean_path_return_pct", float(mean_path_return_pct))
        trial.set_user_attr("cpcv_worst_segment_mdd_pct", float(worst_seg_mdd))
        trial.set_user_attr(
            "cpcv_path_oos_log_tw",
            [float(x) for x in path_compound_raw_log_tw],
        )

        return float(objective_final)
    finally:
        strategy._portfolio_eval_ctx = None
