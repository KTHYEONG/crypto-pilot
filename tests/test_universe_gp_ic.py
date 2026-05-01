import importlib
import logging
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
    FUTURES_SCREENER_CONFIG,
    OPT_FUTURES_CONFIG,
    get_quarterly_window,
)
from config.settings import FUTURES_DATA_DIR
from src.domain.futures.data_collector import DataCollector
from src.domain.futures.ml_pipeline.ml_pipeline_runner import run_ml_pipeline_for_universe
from src.execution.opt_main_futures import _load_futures_data_maps_for_symbols

warnings.filterwarnings("ignore")

# 로그 설정
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("test_full_universe_gp")

def run_universe_to_gp_test():
    _logger.info("=" * 70)
    _logger.info(" [PHASE 1] Universe Filtering (Broad + Refinement)")
    _logger.info("=" * 70)
    
    # 1. 분기 윈도우 설정
    res = get_quarterly_window()
    fetch_start, start, is_end, end = res
    tf = "1h"
    collector = DataCollector()
    
    # 2. Broad Screening
    from src.domain.futures.opt_futures_utils.universe_screener_futures import (
        screen_futures_universe,
        screen_symbol_refinement_futures,
    )
    
    _logger.info(f"Window: {fetch_start} ~ {is_end}")
    
    broad_candidates, _ = screen_futures_universe(
        collector, [], tf, FUTURES_SCREENER_CONFIG, fetch_start, is_end, data_dir=FUTURES_DATA_DIR
    )
    
    if not broad_candidates:
        _logger.error("No broad candidates found. Test aborted.")
        return

    _logger.info(f"Broad Candidates: {len(broad_candidates)} symbols.")

    # 3. Data Loading for Refinement
    data_maps_broad, _, valid_broad = _load_futures_data_maps_for_symbols(
        list(broad_candidates), tf, fetch_start, start, is_end, end, skip_metrics=True
    )
    
    # 4. Refinement (Winning Signal Type: CS_RANK)
    success = screen_symbol_refinement_futures(
        broad_candidates=list(broad_candidates),
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
        
    # 설정 리로드
    importlib.reload(config.opt_config)
    final_symbols = config.opt_config.FUTURES_SYMBOLS
    
    _logger.info(f"\nFinal Selected Universe ({len(final_symbols)}): {final_symbols}")
    
    _logger.info("\n" + "=" * 70)
    _logger.info(" [PHASE 2] GP Alpha Mining & IC Evaluation")
    _logger.info("=" * 70)
    
    # 5. ML Pipeline 실행
    cfg = dict(OPT_FUTURES_CONFIG)
    cfg["FUTURES_ML_GP_GENERATIONS"] = 5
    cfg["FUTURES_ML_GP_POPULATION"] = 500
    
    ml_out = run_ml_pipeline_for_universe(
        final_symbols,
        tf,
        fetch_start,
        end,
        cfg,
        workers=4,
        n_jobs=4,
        is_end_date=is_end,
        is_start_date=start
    )
    
    # 6. 결과 보고
    best_fitness = ml_out.alpha_panel.attrs.get("best_fitness", 0.0)
    filter_meta = ml_out.alpha_panel.attrs.get("alpha_component_filter", {})
    
    _logger.info("\n" + "-" * 70)
    _logger.info(" [GP IC VALIDATION RESULTS]")
    _logger.info("-" * 70)
    _logger.info(f" IS Best Fitness (Composite ICIR): {best_fitness:.6f}")
    _logger.info(f" GP Alpha Components Tried:      {filter_meta.get('n_components', 0)}")
    _logger.info(f" GP Alpha Components Surviving:   {filter_meta.get('n_surviving', 0)}")
    neu_p = bool(filter_meta.get("neutralize_primary", 0))
    _logger.info(f" Primary Alpha Neutralized:       {neu_p}")
    _logger.info("-" * 70)
    
    if best_fitness > 0.01: # ICIR 기반이므로 0.01 이상이면 유의미
        _logger.info(" [RESULT] Reasonable IC/Fitness detected. Track A is healthy.")
    else:
        _logger.warning(" [RESULT] Low Fitness detected. Check market volatility or features.")
    _logger.info("=" * 70)

if __name__ == "__main__":
    run_universe_to_gp_test()
