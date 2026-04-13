"""
Futures Optuna objective: CPCV paths, Kelly-CVaR scalar, disk+memory signal cache.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import (
    FUTURES_INITIAL_BALANCE,
    SLIPPAGE_RATE,
    TRADING_FEE_RATE,
)
from src.domain.futures.engine_multi_futures import (
    PortfolioBacktestEngineFast,
)
from src.domain.futures.engine_single_futures import BacktestEngineFast
from src.domain.futures.opt_futures_utils.cv_utils import (
    CPCVPath,
    cpcv_complement_segments,
)
from src.domain.futures.opt_futures_utils.metrics import (
    _log_tw_from_ret_pct,
    calc_cvar5_loss_pct_from_equity,
    calc_max_underwater_days_from_equity,
    calc_mdd_from_equity,
    calc_ulcer_index_from_equity,
    compute_pbo_from_cpcv_paths,
)
from src.domain.futures.strategies_futures import (
    FuturesPipelineStrategy,
    UltimateStrategy,
)

from .data_utils import (
    _build_aligned_2d_from_prebuilt,
    _dataframe_to_symbol_arrays,
    _segment_with_context,
    align_data_for_2d_engine,
)
from .signal_cache import (
    get_tiered_signals,
)

_logger: logging.Logger = logging.getLogger("opt_futures")


def evaluate_symbol_fold(
    strategy: FuturesPipelineStrategy,
    params: Dict[str, Any],
    symbol: str,
    tf: str,
    target_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    merge_idx: int,
    _val_range: Optional[Tuple[int, int]] = None,
    test_start: int = 0,
    test_end: int = 0,
    precomputed_signal_df: Optional[pd.DataFrame] = None,
    execution_start_idx: int = 0,
) -> Tuple[float, float, float, int, float, float, int, int, np.ndarray, float, float]:
    """
    Returns (cagr, ret_pct, mdd, n_trades, win_rate, pf, long_c, short_c, eq_curve, fpaid, gross).
    """
    if precomputed_signal_df is not None:
        sig_oos: pd.DataFrame = precomputed_signal_df
    else:
        # [OPTIMIZATION] Use tiered signals for caching support
        full_signal: pd.DataFrame = get_tiered_signals(
            params, symbol, tf, target_df, cast(UltimateStrategy, strategy)
        )
        sig_oos, execution_start_idx = _segment_with_context(full_signal, test_start, test_end)

    warmup_bars: int = 0
    sig_oos.attrs = {"warmup_bars": warmup_bars}

    # Timeframe hours calculation (fixed for ruff)
    _ = 1.0
    if tf.endswith("h"):
        try:
            _ = float(tf.replace("h", ""))
        except ValueError:
            pass
    elif tf.endswith("d"):
        try:
            _ = float(tf.replace("d", "")) * 24.0
        except ValueError:
            pass

    # [FIX] BacktestEngineFast constructor arguments fixed to match definition
    engine: BacktestEngineFast = BacktestEngineFast(
        hourly_df=sig_oos,
        daily_df=daily_df,
        strategy=strategy,
        initial_balance=10000.0,
        merge_index_map=merge_idx,
        execution_start_idx=execution_start_idx,
    )
    res = engine.run()
    eq_curve = res.get("equity_curve", np.array([10000.0]))
    final_bal = res.get("final_balance", 10000.0)
    trades_df = res.get("trades_df", pd.DataFrame())

    ret_pct = (final_bal / 10000.0 - 1.0) * 100.0
    mdd = calc_mdd_from_equity(eq_curve)
    n_tr = len(trades_df)

    if n_tr > 0:
        win_rate = (trades_df["pnl"] > 0).mean() * 100.0
        gross_p = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        gross_l = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
        pf = gross_p / max(gross_l, 1e-9)
        long_c = (trades_df["side"] == "BUY").sum()
        short_c = (trades_df["side"] == "SELL").sum()
        fpaid = trades_df["funding_fee"].sum() if "funding_fee" in trades_df.columns else 0.0
        gross_pnl = trades_df["pnl"].sum()
    else:
        win_rate, pf, long_c, short_c, fpaid, gross_pnl = 0.0, 1.0, 0, 0, 0.0, 0.0

    days = (len(sig_oos) * (4 if tf == "4h" else 1)) / 24.0
    cagr = ((final_bal / 10000.0) ** (365.0 / max(days, 1.0)) - 1.0) * 100.0

    return (
        float(cagr),
        float(ret_pct),
        float(mdd),
        int(n_tr),
        float(win_rate),
        float(pf),
        int(long_c),
        int(short_c),
        eq_curve,
        float(fpaid),
        float(gross_pnl),
    )


def run_oos_margin_shared_portfolio(
    symbols: List[str],
    tf: str,
    params: Dict[str, Any],
    oos_data_maps: Dict[str, Dict[str, Any]],
    cache_root: Optional[Path] = None,
    return_signal_dfs: bool = False,
    oos_end_idx: Optional[int] = None,
) -> Dict[str, Any]:
    strategy = FuturesPipelineStrategy(name="OOS_Portfolio", params=params)
    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    seg_dfs: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        full_df = oos_data_maps[sym][tf]
        oos_start = int(oos_data_maps[sym][f"oos_start_idx_{tf}"])
        # [OPTIMIZATION] Use tiered caching
        full_sig = get_tiered_signals(params, sym, tf, full_df, strategy)
        end_cap = int(oos_end_idx) if oos_end_idx is not None else len(full_df)
        seg, _ = _segment_with_context(full_sig, oos_start, end_cap)
        full_signal_dfs[sym] = full_sig
        seg_dfs[sym] = seg

    aligned_data, _ = align_data_for_2d_engine(seg_dfs, symbols)
    if not aligned_data:
        return {
            "cagr_pct": -100.0, "mdd_pct": 100.0, "profit_factor": 0.0,
            "total_trades": 0, "moic": 0.0, "terminal_wealth_ratio": 0.0,
            "cvar_pct": 100.0, "hw_recovery_days": 999.0, "win_rate_pct": 0.0,
            "oos_long_short_minority_pct": 0.0, "long_trades": 0, "short_trades": 0,
            "equity_curve": np.array([FUTURES_INITIAL_BALANCE]),
        }

    engine = PortfolioBacktestEngineFast(
        aligned_data=aligned_data,
        symbol_names=symbols,
        strategy_params=params,
        initial_balance=float(FUTURES_INITIAL_BALANCE),
        fee_rate=TRADING_FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
    )
    trades_df, equity_curve, final_balance = engine.run()

    # Metrics
    hours_per_bar = int(tf.replace("h", "")) if tf.endswith("h") else 4
    n_days = (len(equity_curve) * hours_per_bar) / 24.0
    cagr = ((final_balance / FUTURES_INITIAL_BALANCE) ** (365.0 / max(n_days, 1.0)) - 1.0) * 100.0
    mdd = calc_mdd_from_equity(equity_curve)
    moic = final_balance / FUTURES_INITIAL_BALANCE
    cvar = calc_cvar5_loss_pct_from_equity(equity_curve)
    hw_days = calc_max_underwater_days_from_equity(equity_curve, float(hours_per_bar))
    ulcer = calc_ulcer_index_from_equity(equity_curve)

    if not trades_df.empty:
        gains = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        losses = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
        pf_val = gains / max(losses, 1e-9)
        wr = (trades_df["pnl"] > 0).mean() * 100.0
        lt = int((trades_df["side"] == "LONG").sum())
        st = int((trades_df["side"] == "SHORT").sum())

        long_trades = trades_df[trades_df["side"] == "LONG"]
        l_gains = long_trades[long_trades["pnl"] > 0]["pnl"].sum()
        l_losses_sum = abs(long_trades[long_trades["pnl"] < 0]["pnl"].sum())
        l_pf = l_gains / max(l_losses_sum, 1e-9)

        short_trades = trades_df[trades_df["side"] == "SHORT"]
        s_gains = short_trades[short_trades["pnl"] > 0]["pnl"].sum()
        s_losses_sum = abs(short_trades[short_trades["pnl"] < 0]["pnl"].sum())
        s_pf = s_gains / max(s_losses_sum, 1e-9)

        short_wr = float((short_trades["pnl"] > 0).mean() * 100.0) if st > 0 else 0.0
        minority_pct = float(min(lt, st) / max(lt + st, 1) * 100.0)

        # EV/cost ratio: avg net PnL per trade / round-trip transaction cost
        avg_net_pnl = float(trades_df["pnl"].mean())
        round_trip_cost = (
            (TRADING_FEE_RATE * 2.0 + SLIPPAGE_RATE * 2.0) * float(FUTURES_INITIAL_BALANCE)
        )
        ev_ratio = avg_net_pnl / max(round_trip_cost, 1e-9)
    else:
        pf_val, wr, lt, st, l_pf, s_pf = 1.0, 0.0, 0, 0, 1.0, 1.0
        short_wr, minority_pct, ev_ratio = 0.0, 0.0, 0.0

    calmar = cagr / abs(mdd) if abs(mdd) > 1e-6 else 0.0

    out = {
        "cagr_pct": cagr, "mdd_pct": mdd, "profit_factor": pf_val, "total_trades": len(trades_df),
        "moic": moic, "terminal_wealth_ratio": moic, "win_rate_pct": wr, "long_trades": lt,
        "short_trades": st, "long_pf": l_pf, "short_pf": s_pf, "equity_curve": equity_curve,
        "cvar_pct": cvar, "hw_recovery_days": hw_days, "oos_long_short_minority_pct": minority_pct,
        "calmar_ratio": calmar, "ulcer_index": ulcer, "short_win_rate_pct": short_wr,
        "ev_cost_ratio": ev_ratio,
        "trades_df": trades_df,
    }
    if return_signal_dfs:
        out["full_signal_dfs"] = full_signal_dfs
    return out


def compute_regime_conditional_oos_metrics(
    full_sigs: Dict[str, pd.DataFrame],
    equity_curve: np.ndarray,
    oos_start: int,
    symbols: List[str],
) -> Dict[str, Any]:
    """Advisory regime metrics for diagnostics."""
    out: Dict[str, Any] = {}
    ref_sig = full_sigs.get(symbols[0])
    if ref_sig is None:
        return out

    oos_sig = ref_sig.iloc[oos_start:].copy()
    if "regime_risk_mult" not in oos_sig.columns:
        return out

    # Simple logic: split equity curve by regime
    reg = oos_sig["regime_risk_mult"].to_numpy()
    rets = np.diff(equity_curve) / equity_curve[:-1]
    
    for r_val in np.unique(reg):
        mask = reg[:-1] == r_val
        if not mask.any():
            continue
        r_rets = rets[mask]
        bc = int(mask.sum())
        ret_pct = float(np.prod(1 + r_rets) - 1) * 100.0
        mdd_c = float(calc_mdd_from_equity(np.cumprod(1 + r_rets)))
        avg_br = float(np.mean(r_rets))
        out[str(r_val)] = {
            "bar_count": bc,
            "return_pct": ret_pct,
            "mdd_pct": mdd_c,
            "avg_bar_return": avg_br,
        }
    return out


def run_cpcv_complement_evaluation(
    params: Dict[str, Any],
    symbols: List[str],
    tf: str,
    data_maps: Dict[str, Dict[str, Any]],
    cpcv_paths: List[CPCVPath],
    all_block_ranges: List[Tuple[int, int]],
    *,
    oos_path_scores: Sequence[float],
    signal_disk_cache_root: Optional[Path] = None,
    project_root: Optional[str] = None,
    concurrency_penalty_scale: float = 1.0,
) -> Tuple[float, float]:
    """
    Evaluate CPCV complement (train) segments for each path on fixed params;
    compare to stored OOS path scores. Returns (pbo, spearman_rho).
    """
    oos_list = [float(x) for x in oos_path_scores]
    if len(cpcv_paths) != len(oos_list) or not cpcv_paths or not all_block_ranges:
        return (0.5, 0.0)

    ref_sym = symbols[0]
    is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
    ref_df = data_maps[ref_sym][tf]
    if ref_df is None or ref_df.empty:
        return (0.5, 0.0)

    p = dict(params)
    strategy = UltimateStrategy(name="PBOComplement", params=p)

    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        target_df_full = data_maps.get(sym, {}).get(tf)
        if target_df_full is None or target_df_full.empty:
            continue
        full_signal_dfs[sym] = get_tiered_signals(p, sym, tf, target_df_full, strategy)

    if len(full_signal_dfs) != len(symbols):
        return (0.5, 0.0)

    prebuilt_full_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for sym in symbols:
        prebuilt_full_arrays[sym] = _dataframe_to_symbol_arrays(full_signal_dfs[sym])

    unique_comp_blocks = set()
    for path in cpcv_paths:
        comp = cpcv_complement_segments(path, all_block_ranges)
        for b in comp:
            unique_comp_blocks.add(tuple(b))

    # Use same MDD threshold as CPCV objective (FUTURES_MAX_MDD=35%) for consistent scoring.
    liq_mdd_thr = float(OPT_FUTURES_CONFIG.get("FUTURES_MAX_MDD", 35.0))
    block_metrics: Dict[Tuple[int, int], float] = {}
    for b_range in sorted(list(unique_comp_blocks), key=lambda x: x[0]):
        b_start, b_end = cast(Tuple[int, int], b_range)
        adj_s, adj_e = b_start + is_off, b_end + is_off
        aligned_data = _build_aligned_2d_from_prebuilt(
            prebuilt_full_arrays, symbols, adj_s, adj_e
        )
        if aligned_data is None:
            block_metrics[(b_start, b_end)] = -10.0
            continue

        try:
            engine = PortfolioBacktestEngineFast(
                aligned_data=aligned_data, symbol_names=symbols, strategy_params=p,
                initial_balance=float(FUTURES_INITIAL_BALANCE),
                fee_rate=TRADING_FEE_RATE, slippage_rate=SLIPPAGE_RATE,
            )
            _, equity_curve, final_balance = engine.run()
            ret_pct = float((final_balance / FUTURES_INITIAL_BALANCE - 1.0) * 100.0)
            raw_log = _log_tw_from_ret_pct(ret_pct)
            mdd_seg = float(calc_mdd_from_equity(equity_curve)) if equity_curve.size > 0 else 0.0
            if mdd_seg >= liq_mdd_thr:
                raw_log -= 1e9
            block_metrics[(b_start, b_end)] = raw_log
        except Exception:
            block_metrics[(b_start, b_end)] = -10.0

    is_scores: List[float] = []
    for path in cpcv_paths:
        comp = cpcv_complement_segments(path, all_block_ranges)
        p_score = sum(block_metrics.get(cast(Tuple[int, int], tuple(b)), -10.0) for b in comp)
        is_scores.append(float(p_score))

    if any(not np.isfinite(x) for x in is_scores):
        return (0.5, 0.0)
    return compute_pbo_from_cpcv_paths(is_scores, oos_list)
