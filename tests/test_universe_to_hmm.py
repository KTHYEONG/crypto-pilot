import importlib
import logging
import sys
import warnings
from pathlib import Path

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
from src.domain.futures.data_collector import DataCollector
from src.domain.futures.ml_pipeline.ml_pipeline_runner import run_ml_pipeline_for_universe
from src.execution.opt_main_futures import _load_futures_data_maps_for_symbols

warnings.filterwarnings("ignore")

# Logger Setup
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("test_universe_to_hmm")

def test_universe_gp_hmm_flow():
    _logger.info("=" * 85)
    _logger.info(" [TEST] Universe -> GP Alpha -> HMM Regime Flow")
    _logger.info("=" * 85)
    
    # 1. Window Setup
    res = get_quarterly_window()
    fetch_start, start, is_end, end = res
    tf = "1h"
    collector = DataCollector()
    
    _logger.info(f"Window: {fetch_start} ~ {is_end}")

    # 2. Universe Filtering (Broad)
    from src.domain.futures.opt_futures_utils.universe_screener_futures import (
        screen_futures_universe,
        screen_symbol_refinement_futures,
    )
    
    broad_candidates, _ = screen_futures_universe(
        collector, [], tf, FUTURES_SCREENER_CONFIG, fetch_start, is_end, data_dir=FUTURES_DATA_DIR
    )
    
    if not broad_candidates:
        _logger.error("No broad candidates found. Aborting.")
        return

    _logger.info(f"Broad Candidates: {len(broad_candidates)} symbols.")

    # 3. Data Loading for Refinement
    # [Efficiency] Limit to 10 symbols for faster test if many
    test_symbols = list(broad_candidates)[:10]
    data_maps_broad, _, valid_broad = _load_futures_data_maps_for_symbols(
        test_symbols, tf, fetch_start, start, is_end, end, skip_metrics=True
    )
    
    # 4. Refinement
    success = screen_symbol_refinement_futures(
        broad_candidates=valid_broad,
        winning_signal_type="CS_RANK",
        is_end_date=is_end,
        tf=tf,
        symbol_dfs_4h={s: data_maps_broad[s][tf] for s in valid_broad},
        daily_dfs={s: data_maps_broad[s]["1d"] for s in valid_broad},
        phase_b_params=None,
        anchor_symbols=FUTURES_ANCHOR_SYMBOLS,
    )
    
    if not success:
        _logger.error("Refinement failed. No symbols passed.")
        return
        
    importlib.reload(config.opt_config)
    final_symbols = config.opt_config.FUTURES_SYMBOLS
    _logger.info(f"Final Selected Universe: {final_symbols}")

    # 5. ML Pipeline (GP + HMM)
    cfg = dict(OPT_FUTURES_CONFIG)
    # Reduced generations for faster test execution
    cfg["FUTURES_ML_GP_GENERATIONS"] = 3
    cfg["FUTURES_ML_GP_POPULATION"] = 300
    
    _logger.info("\nExecuting ML Pipeline (GP Mining + HMM Regime Inference)...")
    ml_out = run_ml_pipeline_for_universe(
        final_symbols,
        tf,
        fetch_start,
        end,
        cfg,
        workers=4,
        n_jobs=4,
        is_end_date=is_end,
        is_start_date=start,
        gp_only=False  # Run HMM as well
    )
    
    # 6. Verification
    _logger.info("\n" + "-" * 85)
    _logger.info(" [VERIFICATION] ML Pipeline Output")
    _logger.info("-" * 85)
    
    # Verify GP
    best_fitness = ml_out.alpha_panel.attrs.get("best_fitness", 0.0)
    _logger.info(f" GP Best Fitness: {best_fitness:.6f}")
    
    # Verify HMM results in Meta Feature Frames
    for sym, mff in ml_out.meta_feature_frame_by_symbol.items():
        _logger.info(f" Symbol: {sym}")
        cols = mff.columns.tolist()
        has_gp = "gp_alpha_00" in cols
        has_modulator = "hmm_modulator" in cols
        hmm_probs = [c for c in cols if c.startswith("hmm_prob_") or c in ["bull_trend", "bear_trend", "sideways", "crisis"]]
        
        _logger.info(f"  - Has GP Alpha: {has_gp}")
        _logger.info(f"  - Has HMM Modulator: {has_modulator}")
        _logger.info(f"  - HMM Probability Columns: {len(hmm_probs)}")
        
        if has_modulator:
            mean_mod = mff["hmm_modulator"].mean()
            _logger.info(f"  - Mean HMM Modulator: {mean_mod:.4f}")
            
        break # Just check the first symbol
        
    _logger.info("=" * 85)
    _logger.info(" [RESULT] Universe -> GP -> HMM Flow test completed.")
    _logger.info("=" * 85)

if __name__ == "__main__":
    test_universe_gp_hmm_flow()
