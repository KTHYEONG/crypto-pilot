import importlib
import logging
import os
import sys
import warnings
from pathlib import Path

# 프로젝트 루트 설정
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config.opt_config
from config.opt_config import (
    FUTURES_ANCHOR_SYMBOLS,
    FUTURES_MACRO_INDEX_SYMBOLS,
    FUTURES_SCREENER_CONFIG,
    OPT_FUTURES_CONFIG,
    get_quarterly_window,
)
from config.settings import FUTURES_DATA_DIR
from src.domain.futures.data_loader import DataCollector
from src.domain.futures.ml_pipeline.pipeline_runner import run_ml_pipeline_for_universe
from src.domain.futures.optimization.opt_data_utils import (
    load_futures_data_maps_for_symbols,
)
from src.execution.opt_main_futures import (
    _resolve_futures_parallel_policy,
)

warnings.filterwarnings("ignore")

# Force Linux 'fork' method for memory efficiency (CoW)
if sys.platform != "win32":
    try:
        import multiprocessing
        multiprocessing.set_start_method("fork", force=True)
    except (RuntimeError, ImportError):
        pass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

# 로그 설정
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("test_full_universe_gp")

def run_universe_to_gp_test(tf="4h"):
    _logger.info("=" * 70)
    _logger.info(f" [PHASE 1] Universe Filtering (Broad + Refinement) EXEC TF: {tf}")
    _logger.info("=" * 70)
    
    # 1. 분기 윈도우 설정
    res = get_quarterly_window()
    fetch_start, start, is_end, end = res
    collector = DataCollector()
    
    # [Institutional Quant] Alpha Dual-TF Logic Alignment
    # Always screen universe and train models on 4h to avoid noise/fees, even if execution is 1h.
    ml_train_tf = "4h" if tf == "1h" else tf
    _logger.info(f" [ARCH] Alpha Dual-TF Mode: Exec={tf} | Training={ml_train_tf}")

    # 2. Broad Screening (Always use 4h for quality anchoring if target is 1h/4h)
    from src.domain.futures.optimization.screener import (
        screen_futures_universe,
        screen_symbol_refinement_futures,
    )
    
    _logger.info(f"Window: {fetch_start} ~ {is_end}")
    
    broad_candidates, _ = screen_futures_universe(
        collector, [], ml_train_tf, FUTURES_SCREENER_CONFIG, fetch_start, is_end, data_dir=FUTURES_DATA_DIR
    )
    
    if not broad_candidates:
        _logger.error("No broad candidates found. Test aborted.")
        return

    _logger.info(f"Broad Candidates: {len(broad_candidates)} symbols.")

    # 3. Data Loading for Refinement
    data_maps_broad, _, valid_broad = load_futures_data_maps_for_symbols(
        list(broad_candidates), ml_train_tf, fetch_start, start, is_end, end, skip_metrics=True
    )
    
    # 4. Refinement (Winning Signal Type: CS_RANK)
    success = screen_symbol_refinement_futures(
        broad_candidates=list(broad_candidates),
        winning_signal_type="CS_RANK",
        is_end_date=is_end,
        tf=ml_train_tf,
        symbol_dfs_4h={s: data_maps_broad[s][ml_train_tf] for s in valid_broad},
        daily_dfs={s: data_maps_broad[s]["1d"] for s in valid_broad},
        phase_b_params=None,
        anchor_symbols=FUTURES_ANCHOR_SYMBOLS,
    )
    
    if not success:
        _logger.error("Refinement failed. No symbols passed.")
        return
        
    # 설정 리로드
    importlib.reload(config.opt_config)
    final_symbols = config.opt_config.FUTURES_SYMBOLS
    
    # [3-Tier Universe] Ensure Anchors and Macro Index symbols are always loaded for systemic HMM
    load_symbols = list(set(final_symbols + FUTURES_ANCHOR_SYMBOLS + FUTURES_MACRO_INDEX_SYMBOLS))
    
    _logger.info(f"\nFinal Selected Universe ({len(final_symbols)}): {final_symbols}")
    _logger.info(f"3-Tier Load Universe ({len(load_symbols)} symbols for Pipeline)")

    # [Data Integrity] Re-load data for the 3-tier universe (matching production logic)
    data_maps, _, valid_symbols = load_futures_data_maps_for_symbols(
        load_symbols, tf, fetch_start, start, is_end, end, skip_metrics=False
    )
    
    if not valid_symbols:
        _logger.error("No valid symbols loaded for 3-tier universe. Test aborted.")
        return

    _logger.info("\n" + "=" * 70)
    _logger.info(f" [PHASE 2] ML Pipeline: Alpha Mining on {ml_train_tf}")
    _logger.info("=" * 70)
    
    # 5. ML Pipeline 실행 (Production Settings 일치)
    cfg = dict(OPT_FUTURES_CONFIG)
    cfg["FUTURES_USE_META_LABELER"] = False  # Speed up test by skipping Meta-Labeler
    ml_n_jobs = _resolve_futures_parallel_policy(len(valid_symbols))
    
    _logger.info(f"\nExecuting ML Pipeline (Dual-TF Mode: {ml_train_tf} training)...")
    ml_out = run_ml_pipeline_for_universe(
        valid_symbols,
        ml_train_tf,
        fetch_start_date=fetch_start,
        end=end,
        cfg=cfg,
        workers=ml_n_jobs,
        n_jobs=ml_n_jobs,
        is_end_date=is_end,
        is_start_date=start,
        preloaded_data_maps=data_maps
    )
    
    # 6. 결과 보고
    best_fitness = ml_out.alpha_panel.attrs.get("best_fitness", 0.0)
    filter_meta = ml_out.alpha_panel.attrs.get("alpha_component_filter", {})
    
    _logger.info("\n" + "-" * 70)
    _logger.info(" [ALPHA IC AUDIT - 3.5bps Friction Environment]")
    _logger.info("-" * 70)
    _logger.info(f" IS Mean IC:                     {filter_meta.get('primary_is_mu', 0.0):>8.4f}")
    _logger.info(f" OOS Mean IC:                    {filter_meta.get('primary_oos_mu', 0.0):>8.4f}")
    _logger.info(f" IS Best Fitness (Composite):    {best_fitness:>8.4f}")
    _logger.info(f" Components Tried:               {filter_meta.get('n_components', 0):>8.0f}")
    _logger.info(f" Components Surviving:            {filter_meta.get('n_surviving', 0):>8.0f}")
    _logger.info(f" Alpha Half-Life (Bars):         {filter_meta.get('primary_half_life', 0.0):>8.2f}")
    neu_p = bool(filter_meta.get("neutralize_primary", 0))
    _logger.info(f" Primary Alpha Neutralized:       {neu_p}")
    _logger.info("-" * 70)
    
    if filter_meta.get('primary_is_mu', 0.0) > 0.02 and filter_meta.get('primary_oos_mu', 0.0) > 0.01:
        _logger.info(" [RESULT] SOTA Alpha performance detected. Track A is healthy.")
    else:
        _logger.warning(" [RESULT] Alpha strength below SOTA targets. Check features.")
    _logger.info("=" * 70)

if __name__ == "__main__":
    run_universe_to_gp_test("4h")
