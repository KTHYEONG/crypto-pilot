import importlib
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

# Project Root Setup
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config.opt_config
from config.opt_config import (
    FUTURES_ANCHOR_SYMBOLS,
    FUTURES_SCREENER_CONFIG,
    OPT_FUTURES_CONFIG,
    get_quarterly_window,
)
from config.settings import FUTURES_DATA_DIR
from src.domain.futures.ml_pipeline.features.engineering import HMM_SEMANTIC_PROB_COLUMNS
from src.domain.futures.data_loader import DataCollector
from src.domain.futures.ml_pipeline.pipeline_runner import run_ml_pipeline_for_universe
from src.execution.opt_main_futures import _load_futures_data_maps_for_symbols
from src.domain.futures.optimization.screener import (
    screen_futures_universe,
    screen_symbol_refinement_futures,
)

warnings.filterwarnings("ignore")

# Logger Setup
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("test_universe_to_hmm")

def audit_hmm_logic_changes(ml_out, symbols, data_maps, tf):
    """Deep audit of HMM regime classification with Log-Wealth & Tail Capture metrics."""
    _logger.info("\n" + "╔" + "═" * 83 + "╗")
    _logger.info(f"║ [INSTITUTIONAL HMM AUDIT] Log-Wealth & Ergodicity (v16) TF: {tf:<8} {' ':<4} ║")
    _logger.info("╠" + "═" * 83 + "╣")

    # Try to get data from symbols first, then fallback to market_probs
    mff = None
    sym = None
    for s in symbols:
        if s in ml_out.meta_feature_frame_by_symbol and s in data_maps:
            sym = s
            mff = ml_out.meta_feature_frame_by_symbol[s]
            break
    
    if mff is None and not ml_out.market_probs.empty:
        _logger.info("║ [INFO] No per-symbol fusion found. Using Market Probs + BTC proxy.           ║")
        mff = ml_out.market_probs
        # Use BTC or first available symbol for close prices
        for s in ["BTC/USDT", "BTCUSDT"] + symbols:
            if s in data_maps:
                sym = s
                break

    if mff is None or sym is None:
        _logger.error(f"║ No common data for HMM audit. {' ':<48} ║")
        _logger.info("╚" + "═" * 83 + "╝")
        return

    df = data_maps[sym][tf].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime")
    # Calculate returns on the source 4h/1h data directly
    df["ret"] = df["close"].pct_change().fillna(0.0)
    
    # Ensure returns are available
    if "datetime" not in mff.columns:
        mff = mff.reset_index()
    
    mff["datetime"] = pd.to_datetime(mff["datetime"], utc=True)
    
    # Merge HMM probs with the enriched OHLCV (which already has 'ret')
    merged = pd.merge(mff, df[["datetime", "close", "ret"]], on="datetime", how="inner")
    
    if merged.empty:
        _logger.error(f"║ Merge produced empty frame. Check datetime alignment. {' ':<25} ║")
        _logger.info("╚" + "═" * 83 + "╝")
        return

    regime_cols = [c for c in HMM_SEMANTIC_PROB_COLUMNS if c in merged.columns]
    if not regime_cols:
        _logger.error("║ Missing HMM probability columns. {' ':<46} ║")
        return

    merged["regime"] = merged[regime_cols].idxmax(axis=1)
    semantic_cols = [c for c in HMM_SEMANTIC_PROB_COLUMNS if c in merged.columns]

    # v9.0 Regime Names
    regime_display_names = {
        "hmm_prob_bull_calm": "BULL_CALM",
        "hmm_prob_bull_vol_up": "BULL_VOL_UP",
        "hmm_prob_bear_trend": "BEAR_TREND",
        "hmm_prob_chop": "CHOP",
        "hmm_prob_crisis": "CRISIS",
    }

    # 1. Log-Wealth Dispersion (g = mu - 0.5 * sigma^2)
    _logger.info(f"║ [A] LOG-WEALTH DISPERSION (Regime Purity) | Base: {tf:<23} ║")
    _logger.info("╟──────────────┬──────────┬──────────┬──────────┬──────────┬──────────────────────╢")
    _logger.info("║ REGIME       │ TIME %   │ MU (%)   │ SIG (%)  │ G_log(%) │ BEHAVIOR             ║")
    _logger.info("╟──────────────┼──────────┼──────────┼──────────┼──────────┼──────────────────────╢")

    for col in semantic_cols:
        reg_name = regime_display_names.get(col, col.replace("hmm_prob_", "").upper())
        mask = merged["regime"] == col
        if mask.any():
            r = merged.loc[mask, "ret"]
            mu = float(r.mean() * 100.0)
            sig = float(r.std() * 100.0)
            # Geometric Growth Approximation in % units: mu - 0.5 * (sig^2 / 100)
            g = mu - 0.5 * (sig**2 / 100.0)
            time_pct = float(mask.mean() * 100.0)

            behavior = "UNSTABLE"
            if g > 0.02: behavior = "WEALTH_EXP"
            elif g < -0.10: behavior = "TAIL_DEFENSE"
            elif abs(g) < 0.02: behavior = "NOISE_LOCKED"
            
            # v9.2.1: Institutional Verdicts
            v_tag = ""
            if reg_name == "BEAR_TREND":
                v_tag = " [PASS]" if mu < -0.2 else " [FAIL]" if mu >= 0 else " [WARN]"
            elif reg_name == "CRISIS":
                v_tag = " [PREVENTIVE]" if mu > 0.5 else " [LAGGING]"

            _logger.info(f"║ {reg_name:<12} │ {time_pct:>8.1f}% │ {mu:>8.3f} │ {sig:>8.3f} │ {g:>8.3f} │ {behavior:<14}{v_tag:<6} ║")
        else:
            _logger.info(f"║ {reg_name:<12} │ {0.0:>8.1f}% │ {'-':>8} │ {'-':>8} │ {'-':>8} │ {'-':<20} ║")
    
    # 2. Left-Tail Capture Ratio (Worst 5% Isolation)
    _logger.info("╟──────────────┴──────────┴──────────┴──────────┴──────────┴──────────────────────╢")
    _logger.info("║ [B] LEFT-TAIL CAPTURE (Tail-Risk Isolation) {' ':<39} ║")
    
    q05_thr = float(merged["ret"].quantile(0.05))
    worst_bars = merged[merged["ret"] <= q05_thr]
    if not worst_bars.empty:
        tail_dist = worst_bars["regime"].value_counts(normalize=True) * 100
        # v9.2.1: CRISIS and BEAR_TREND are the risk regimes
        crisis_capture = float(tail_dist.get("hmm_prob_crisis", 0.0) + tail_dist.get("hmm_prob_bear_trend", 0.0))
        
        # v9.2.1: Target lowered to 40% as primary protection is now Lead-Lag
        status = "PASS" if crisis_capture >= 40 else "FAIL"
        _logger.info(f"║ Worst 5% Events: {len(worst_bars):<4} | CRISIS/BEAR Capture: {crisis_capture:>5.1f}% | Verdict: {status:<10}    ║")
    
    # 3. Switching Friction & Stability
    _logger.info("╟─────────────────────────────────────────────────────────────────────────────────╢")
    _logger.info("║ [C] REGIME STABILITY & FRICTION {' ':<46} ║")

    _hard_labels = list(HMM_SEMANTIC_PROB_COLUMNS)
    if "hmm_hard_state" in merged.columns:
        hard_regime = merged["hmm_hard_state"].astype(int).map(
            {i: _hard_labels[i] for i in range(len(_hard_labels))}
        ).fillna(merged["regime"])
    else:
        hard_regime = merged["regime"]

    transitions = int((hard_regime != hard_regime.shift(1)).sum())
    avg_duration = float(len(merged) / max(1, transitions))
    friction_est = transitions * 0.00025 * 100.0  # 2.5bps per switch estimate

    # v9.2.1: Stability target 24 bars
    stab_status = "PASS" if avg_duration >= 24 else "FAIL"
    _logger.info(f"║ Total Switches: {transitions:<4} | Avg Duration: {avg_duration:>6.1f} bars | Status: {stab_status:<10}       ║")

    # [D] Lead-Lag Tail Capture: CRISIS 진입 후 8 bars 내 tail event 발생률
    _logger.info("╟─────────────────────────────────────────────────────────────────────────────────╢")
    _logger.info("║ [D] LEAD-LAG TAIL CAPTURE (Preventive Power, N=8 bars ahead)                   ║")

    is_crisis_arr = (merged["regime"] == "hmm_prob_crisis").to_numpy()
    is_tail_arr   = (merged["ret"] <= q05_thr).to_numpy()
    crisis_enters = is_crisis_arr & np.concatenate([[False], ~is_crisis_arr[:-1]])
    n_total = len(merged)
    fwd_tail = np.array([
        bool(np.any(is_tail_arr[i : min(i + 9, n_total)]))
        for i in range(n_total)
    ])
    n_crisis_entries = int(crisis_enters.sum())
    if n_crisis_entries > 0:
        ll_capture = float(fwd_tail[crisis_enters].mean() * 100.0)
        ll_status = "PASS" if ll_capture >= 40 else "ACCEPTABLE" if ll_capture >= 25 else "FAIL"
        _logger.info(f"║ CRISIS Entries: {n_crisis_entries:<4} | Followed by Tail (≤8 bars): {ll_capture:>5.1f}% | Verdict: {ll_status:<10} ║")
    else:
        _logger.info("║ CRISIS Entries: 0    | No entries to measure Lead-Lag.                         ║")

    # [E] Regime IC: Spearman(p_crisis, forward_4bar_return)
    _logger.info("╟─────────────────────────────────────────────────────────────────────────────────╢")
    _logger.info("║ [E] REGIME IC — Spearman(p_crisis, fwd_4bar_ret)                               ║")

    crisis_col = next((c for c in regime_cols if "crisis" in c), None)
    if crisis_col and crisis_col in merged.columns:
        fwd_ret = merged["ret"].shift(-4)
        valid_mask = fwd_ret.notna() & merged[crisis_col].notna()
        if valid_mask.sum() > 50:
            ic_val, ic_pval = spearmanr(
                merged.loc[valid_mask, crisis_col],
                fwd_ret[valid_mask]
            )
            ic_status = "PASS" if ic_val < -0.05 else "ACCEPTABLE" if ic_val < 0.0 else "FAIL"
            _logger.info(f"║ Spearman IC: {ic_val:>+.4f} (p={ic_pval:.3f}) | Target: <-0.05 | Verdict: {ic_status:<10}    ║")
        else:
            _logger.info("║ Insufficient data for IC calculation.                                           ║")
    else:
        _logger.info("║ hmm_prob_crisis column not found.                                               ║")

    # [F] CRISIS Entry Lead Time: tail event 기준 CRISIS가 몇 bars 전에 활성화되었는지
    _logger.info("╟─────────────────────────────────────────────────────────────────────────────────╢")
    _logger.info("║ [F] CRISIS LEAD TIME (bars before tail event CRISIS was active)                 ║")

    tail_positions = np.where(is_tail_arr)[0]
    lead_times: list[int] = []
    for pos in tail_positions:
        lookback = max(0, pos - 20)
        window = is_crisis_arr[lookback : pos + 1]
        if window[-1]:  # CRISIS active at tail event bar
            run = 0
            for k in range(len(window) - 1, -1, -1):
                if window[k]:
                    run += 1
                else:
                    break
            lead_times.append(run - 1)  # bars BEFORE the tail event
        # If not in CRISIS at tail event bar, skip (don't penalize here)

    if lead_times:
        avg_lead = float(np.mean(lead_times))
        pct_concurrent = float(np.mean(np.array(lead_times) == 0) * 100.0)
        pct_leading    = float(np.mean(np.array(lead_times) > 0) * 100.0)
        lead_status = "PASS" if avg_lead >= 1.0 else "ACCEPTABLE" if avg_lead >= 0 else "FAIL"
        _logger.info(f"║ Crisis-covered tail events: {len(lead_times)}/{len(tail_positions)} ({len(lead_times)/max(1,len(tail_positions))*100:.0f}%)                          ║")
        _logger.info(f"║ Avg Lead Time: {avg_lead:>5.1f} bars | Concurrent: {pct_concurrent:>5.1f}% | Leading: {pct_leading:>5.1f}% | {lead_status:<6} ║")
    else:
        _logger.info("║ CRISIS was never active at any tail event bar (0 coverage).                    ║")

    _logger.info("╚" + "═" * 83 + "╝")

def audit_oos_and_ic(ml_out, symbols, is_data_maps, oos_data_maps, tf):
    """OOS Holdout Audit for HMM Tail Capture."""
    _logger.info("\n" + "╔" + "═" * 83 + "╗")
    _logger.info(f"║ [OOS AUDIT] HMM Tail-Risk Generalization Check | TF: {tf:<10} {' ':<14} ║")
    _logger.info("╠" + "═" * 83 + "╣")

    mff = None
    sym = None
    for s in symbols:
        if s in ml_out.meta_feature_frame_by_symbol and s in is_data_maps and s in oos_data_maps:
            sym = s
            mff = ml_out.meta_feature_frame_by_symbol[s]
            break
    
    if mff is None and not ml_out.market_probs.empty:
        _logger.info("║ [INFO] No per-symbol fusion found. Using Market Probs + BTC proxy.           ║")
        mff = ml_out.market_probs
        for s in ["BTC/USDT", "BTCUSDT"] + symbols:
            if s in is_data_maps and s in oos_data_maps:
                sym = s
                break

    if mff is None or sym is None:
        _logger.info("║ No symbol with full IS+OOS data for OOS audit.                             ║")
        _logger.info("╚" + "═" * 83 + "╝")
        return
    if "datetime" not in mff.columns:
        mff = mff.reset_index()
    mff["datetime"] = pd.to_datetime(mff["datetime"], utc=True)

    regime_cols = [c for c in HMM_SEMANTIC_PROB_COLUMNS if c in mff.columns]
    if not regime_cols:
        _logger.info("║ Missing HMM columns for OOS audit.                                         ║")
        _logger.info("╚" + "═" * 83 + "╝")
        return

    # OOS Tail Capture
    oos_df = oos_data_maps[sym][tf].copy()
    oos_df["datetime"] = pd.to_datetime(oos_df["datetime"], utc=True)
    merged_oos = pd.merge(mff, oos_df[["datetime", "close"]], on="datetime", how="inner")

    if len(merged_oos) > 100:
        merged_oos = merged_oos.sort_values("datetime").reset_index(drop=True)
        merged_oos["ret"] = merged_oos["close"].pct_change().fillna(0.0)
        merged_oos["regime"] = merged_oos[regime_cols].idxmax(axis=1)
        q05 = float(merged_oos["ret"].quantile(0.05))
        worst_oos = merged_oos[merged_oos["ret"] <= q05]
        if not worst_oos.empty:
            tail_dist = worst_oos["regime"].value_counts(normalize=True) * 100
            capture = float(tail_dist.get("hmm_prob_crisis", 0.0) + tail_dist.get("hmm_prob_bear_trend", 0.0))
            verdict = "PASS" if capture >= 50 else "ACCEPTABLE" if capture >= 40 else "FAIL"
            _logger.info(f"║ [A] OOS Tail Capture ({len(oos_df)} {tf:<4} bars): CRISIS+BEAR = {capture:5.1f}% | Verdict: {verdict:<8} ║")
        else:
            _logger.info("║ [A] OOS: No tail events found.                                              ║")

        # [B] OOS Regime IC: Spearman(p_crisis, fwd_4bar_return)
        _logger.info("╟─────────────────────────────────────────────────────────────────────────────────╢")
        crisis_col_oos = next((c for c in regime_cols if "crisis" in c), None)
        if crisis_col_oos and crisis_col_oos in merged_oos.columns:
            fwd_ret_oos = merged_oos["ret"].shift(-4)
            valid_mask = fwd_ret_oos.notna() & merged_oos[crisis_col_oos].notna()
            if valid_mask.sum() > 50:
                ic_val, ic_pval = spearmanr(
                    merged_oos.loc[valid_mask, crisis_col_oos],
                    fwd_ret_oos[valid_mask]
                )
                ic_status = "PASS" if ic_val < -0.05 else "ACCEPTABLE" if ic_val < 0.0 else "FAIL"
                _logger.info(f"║ [B] OOS Regime IC: {ic_val:>+.4f} (p={ic_pval:.3f}) | Target: <-0.05 | Verdict: {ic_status:<10}      ║")
            else:
                _logger.info("║ [B] OOS IC: Insufficient overlap for IC calculation.                        ║")
        else:
            _logger.info("║ [B] OOS IC: hmm_prob_crisis column not found.                               ║")

        # [C] OOS Lead-Lag Tail Capture (CRISIS entries → tail within 8 bars)
        _logger.info("╟─────────────────────────────────────────────────────────────────────────────────╢")
        is_crisis_oos = (merged_oos["regime"] == "hmm_prob_crisis").to_numpy()
        is_tail_oos   = (merged_oos["ret"] <= q05).to_numpy()
        crisis_enters_oos = is_crisis_oos & np.concatenate([[False], ~is_crisis_oos[:-1]])
        n_oos = len(merged_oos)
        fwd_tail_oos = np.array([
            bool(np.any(is_tail_oos[i : min(i + 9, n_oos)]))
            for i in range(n_oos)
        ])
        n_entries_oos = int(crisis_enters_oos.sum())
        if n_entries_oos > 0:
            ll_oos = float(fwd_tail_oos[crisis_enters_oos].mean() * 100.0)
            ll_status_oos = "PASS" if ll_oos >= 40 else "ACCEPTABLE" if ll_oos >= 25 else "FAIL"
            _logger.info(f"║ [C] OOS Lead-Lag Capture: {n_entries_oos} entries, {ll_oos:>5.1f}% → tail ≤8 bars | {ll_status_oos:<10}    ║")
        else:
            _logger.info("║ [C] OOS Lead-Lag: No CRISIS entries found in OOS period.                    ║")
    else:
        _logger.info("║ [A] OOS: Insufficient data overlap for holdout audit.                       ║")

    _logger.info("╚" + "═" * 83 + "╝")


def test_universe_gp_hmm_flow(tf="4h"):
    _logger.info("=" * 85)
    _logger.info(f" [TEST] Universe -> HMM Regime Flow (Integrated) TF: {tf}")
    _logger.info("=" * 85)
    
    # 1. Window Setup (Aligned with opt_futures.py)
    fetch_start_date, start_date, is_end_date, end_date = get_quarterly_window(None)
    collector = DataCollector()
    
    _logger.info(f"Window: {fetch_start_date} ~ {end_date} (IS End: {is_end_date})")

    # 2. Universe Discovery & Filtering (Aligned with opt_futures.py)
    _logger.info("\n[STEP] Universe Discovery & Filtering...")
    broad_candidates, _ = screen_futures_universe(
        collector,
        [],
        tf,
        FUTURES_SCREENER_CONFIG,
        fetch_start_date,
        is_end_date,
        data_dir=FUTURES_DATA_DIR,
    )
    
    if not broad_candidates:
        _logger.error("No broad candidates found. Aborting.")
        return

    data_maps_broad, _, valid_broad = _load_futures_data_maps_for_symbols(
        broad_candidates,
        tf,
        fetch_start_date,
        start_date,
        is_end_date,
        end_date,
        skip_metrics=True,
    )

    success = screen_symbol_refinement_futures(
        broad_candidates=list(broad_candidates),
        winning_signal_type="CS_RANK",
        is_end_date=is_end_date,
        tf=tf,
        symbol_dfs_4h={s: data_maps_broad[s][tf] for s in valid_broad},
        daily_dfs={s: data_maps_broad[s]["1d"] for s in valid_broad},
        phase_b_params=None,
        anchor_symbols=FUTURES_ANCHOR_SYMBOLS,
    )
    if not success:
        _logger.error("Universe refinement failed.")
        return
    
    # Reload config to get the updated FUTURES_SYMBOLS
    importlib.reload(config.opt_config)
    from config.opt_config import FUTURES_MACRO_INDEX_SYMBOLS, FUTURES_SYMBOLS
    final_symbols = list(set(FUTURES_SYMBOLS + FUTURES_ANCHOR_SYMBOLS + FUTURES_MACRO_INDEX_SYMBOLS))
    _logger.info(f"Verified ML Universe: {len(final_symbols)} symbols (Anchors + Macro Index included)")

    # 3. Clear HMM Cache to force fresh training
    from config.settings import FUTURES_CACHE_DIR
    import os
    if FUTURES_CACHE_DIR.exists():
        for f in os.listdir(FUTURES_CACHE_DIR):
            if "HMM" in f and f.endswith(".parquet"):
                try:
                    os.remove(FUTURES_CACHE_DIR / f)
                except Exception:
                    pass

    # 4. Data Loading for ML (Ensuring all symbols including anchors are loaded)
    data_maps, oos_data_maps, valid_ml_symbols = _load_futures_data_maps_for_symbols(
        final_symbols, tf, fetch_start_date, start_date, is_end_date, end_date
    )

    # 5. ML Pipeline (HMM Focused)
    cfg = dict(OPT_FUTURES_CONFIG)
    # Ensure consistency with opt_main_futures.py defaults
    # cfg["FUTURES_ML_ALPHA_USE_TBM_WEIGHT"] = True (from opt_config)
    
    import time
    start_time = time.time()
    
    _logger.info("\nExecuting ML Pipeline (Integrated HMM Audit Mode)...")
    ml_out = run_ml_pipeline_for_universe(
        valid_ml_symbols,
        tf,
        fetch_start_date, 
        end_date,
        cfg,
        workers=4,
        n_jobs=4,
        is_end_date=is_end_date,
        is_start_date=start_date,
        gp_only=False,
        hmm_only=True
    )
    
    elapsed_time = time.time() - start_time
    _logger.info(f"\n[TIME] ML Pipeline (JAX HMM) took: {elapsed_time:.2f} seconds")
    
    # 6. Verification & Deep Audit
    # We use data_maps (IS only) for audit consistency
    audit_hmm_logic_changes(ml_out, valid_ml_symbols, data_maps, tf)

    # 7. OOS + IC Audit
    audit_oos_and_ic(ml_out, valid_ml_symbols, data_maps, oos_data_maps, tf)

    _logger.info("=" * 85)
    _logger.info(" [RESULT] Integrated Universe -> HMM test completed.")
    _logger.info("=" * 85)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", type=str, default="4h", choices=["1h", "4h"])
    args = parser.parse_known_args()[0]
    test_universe_gp_hmm_flow(args.tf)

