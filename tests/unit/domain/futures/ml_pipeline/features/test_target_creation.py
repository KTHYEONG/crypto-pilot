import pandas as pd
import numpy as np
import sys
from pathlib import Path
import pytest

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[6])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.execution.opt_main_futures import load_futures_data_maps_for_symbols, get_quarterly_window
from src.domain.futures.ml_pipeline.features.cross_sectional import CrossSectionalPipelineUtils
from config.opt_config import OPT_FUTURES_CONFIG, FUTURES_ANCHOR_SYMBOLS, FUTURES_MACRO_INDEX_SYMBOLS

@pytest.mark.skipif(not Path("data").exists(), reason="Data directory not found")
def test_target_creation_logic():
    """Tests the multi-horizon rank target creation logic."""
    tf = "4h"
    reference_date = "2024-01-01" # Use a past date for stability
    fetch_start, start, is_end, end = get_quarterly_window(reference_date)
    symbols = ["BTC/USDT", "ETH/USDT"]
    load_symbols = list(set(symbols + FUTURES_ANCHOR_SYMBOLS + FUTURES_MACRO_INDEX_SYMBOLS))

    try:
        data_maps, oos_data_maps, valid_symbols = load_futures_data_maps_for_symbols(load_symbols, tf, fetch_start, start, is_end, end)
    except Exception as e:
        pytest.skip(f"Failed to load data: {e}")

    if not valid_symbols:
        pytest.skip("No valid symbols loaded")

    h_utils = CrossSectionalPipelineUtils()
    panel_df = h_utils.build_panel_df(data_maps, tf=tf)

    raw_h = OPT_FUTURES_CONFIG.get("FUTURES_ML_ALPHA_HORIZONS", (3, 6, 12, 24))
    horizons = tuple(int(x) for x in raw_h)
    _ic_hl = float(OPT_FUTURES_CONFIG.get("FUTURES_ML_IC_HALF_LIFE", 2.3))
    _h_weights = tuple(float(np.exp(-h / _ic_hl)) for h in horizons)

    target = h_utils.create_multi_horizon_rank_targets(panel_df, horizons=horizons, weights=_h_weights)

    assert isinstance(target, pd.Series)
    assert not target.empty
    assert target.isna().sum() < len(target)
