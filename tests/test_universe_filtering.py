
import sys
from pathlib import Path
import pandas as pd
import pytest

# Project Root Setup
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.data_loader import DataCollector
from src.domain.futures.optimization.screener import (
    screen_futures_universe,
    screen_symbol_refinement_futures,
)
from config.opt_config import (
    FUTURES_SCREENER_CONFIG,
    FUTURES_ANCHOR_SYMBOLS,
    get_quarterly_window
)
from config.settings import FUTURES_DATA_DIR
from src.execution.opt_main_futures import _load_futures_data_maps_for_symbols

def test_futures_universe_filtering():
    """
    유니버스 필터링 파이프라인(Phase A & B) 통합 테스트
    """
    # 1. 테스트용 기간 설정 (최근 분기)
    tf = "1h"
    # reference_date를 고정하여 테스트 재현성 확보
    fetch_start, start, is_end, end = get_quarterly_window("2024-01-01")
    collector = DataCollector()

    print(f"\n[Phase A] Screening Broad Universe (TF: {tf}, IS_END: {is_end})...")
    
    # 2. Phase A: Broad Screening 실행
    broad_candidates, _ = screen_futures_universe(
        collector,
        [], # blacklist
        tf,
        FUTURES_SCREENER_CONFIG,
        fetch_start,
        is_end,
        data_dir=FUTURES_DATA_DIR,
    )

    assert isinstance(broad_candidates, (list, set)), "Broad candidates should be a list or set"
    assert len(broad_candidates) > 0, "Should find at least one candidate symbol"
    print(f"✅ Phase A Complete: Found {len(broad_candidates)} symbols.")

    # 3. 데이터 로딩 (Refinement를 위해 필요한 데이터만 로드)
    # 테스트 속도를 위해 상위 5개만 샘플링하여 진행
    test_symbols = list(broad_candidates)[:10]
    print(f"[Data Load] Loading data for refinement check (Sample size: {len(test_symbols)})...")
    
    data_maps_broad, _, valid_broad = _load_futures_data_maps_for_symbols(
        test_symbols, tf, fetch_start, start, is_end, end
    )
    
    assert len(valid_broad) > 0, "Should have valid symbols after data loading"

    # 4. Phase B: Refinement 실행
    print(f"[Phase B] Refining symbols (Target TF: {tf})...")
    success = screen_symbol_refinement_futures(
        broad_candidates=list(valid_broad),
        winning_signal_type="CS_RANK",
        is_end_date=is_end,
        symbol_dfs_4h={s: data_maps_broad[s][tf] for s in valid_broad},
        daily_dfs={s: data_maps_broad[s]["1d"] for s in valid_broad},
        phase_b_params=None,
        anchor_symbols=FUTURES_ANCHOR_SYMBOLS,
    )

    assert success is True, "Phase B Refinement process failed"
    print("✅ Phase B Complete: Universe Refinement successful.")

if __name__ == "__main__":
    test_futures_universe_filtering()
