from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = str(Path(__file__).resolve().parents[4])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.backtest_engine import MultiSymbolEngine
from config.settings import FUTURES_DATA_DIR

def test_backtest_engine_multi_symbol_mock():
    """Tests the MultiSymbolEngine with mock data."""
    n_bars = 100
    n_syms = 2
    symbols = ["BTC/USDT", "ETH/USDT"]
    
    # Mock aligned data
    aligned_data = {
        "close": np.ones((n_bars, n_syms)) * 100.0,
        "high": np.ones((n_bars, n_syms)) * 101.0,
        "low": np.ones((n_bars, n_syms)) * 99.0,
        "open": np.ones((n_bars, n_syms)) * 100.0,
        "atr": np.ones((n_bars, n_syms)) * 2.0,
        "funding_rate_sum": np.zeros((n_bars, n_syms)),
        "kill_signal": np.zeros((n_bars, n_syms)),
        "xs_score_long": np.ones((n_bars, n_syms)) * 0.5,
        "xs_score_short": np.zeros((n_bars, n_syms)),
        "hmm_prob_crisis": np.zeros((n_bars, n_syms)),
        "hmm_modulator_long": np.ones((n_bars, n_syms)),
        "hmm_modulator_short": np.ones((n_bars, n_syms)),
        # Additional required columns for alignment/engine
        "entry_upper": np.zeros((n_bars, n_syms)),
        "entry_lower": np.ones((n_bars, n_syms)) * 999999.0,
        "trend_direction": np.ones((n_bars, n_syms)),
        "strength_filter": np.ones((n_bars, n_syms)),
        "slot_rank_score": np.ones((n_bars, n_syms)),
        "ml_calib_prob": np.zeros((n_bars, n_syms)),
    }
    
    strategy_params = {
        "K_LONG": 1,
        "K_SHORT": 1,
        "REBALANCE_BARS": 6,
        "ATR_MULT": 3.0,
        "TRAIL_MULT": 3.0,
        "MAX_EXPOSURE_PER_COIN": 1.5,
        "MAX_EXPOSURE": 0.8,
        "LEVERAGE": 5.0,
    }
    
    engine = MultiSymbolEngine(
        aligned_data=aligned_data,
        symbol_names=symbols,
        strategy_params=strategy_params,
        initial_balance=10000.0
    )
    
    trades_df, equity, final_bal, diag = engine.run()
    
    assert isinstance(trades_df, pd.DataFrame)
    assert isinstance(equity, np.ndarray)
    assert isinstance(final_bal, float)
    assert isinstance(diag, (dict, np.ndarray))
    assert len(equity) == n_bars

@pytest.mark.skipif(not FUTURES_DATA_DIR.exists(), reason="Data directory not found")
def test_backtest_engine_real_data_structure():
    """Tests the MultiSymbolEngine with real data structure (if available)."""
    files = list(FUTURES_DATA_DIR.glob('*_1h.parquet'))
    if not files:
        pytest.skip("No parquet data files found for test")
        
    df = pd.read_parquet(files[0]).iloc[-200:]
    n_bars = len(df)
    symbols = ["TEST/USDT"]
    
    # Mock the required columns for the engine
    aligned_data = {
        "close": df["close"].to_numpy().reshape(-1, 1),
        "high": df["high"].to_numpy().reshape(-1, 1),
        "low": df["low"].to_numpy().reshape(-1, 1),
        "open": df["open"].to_numpy().reshape(-1, 1),
        "atr": (df["close"] * 0.01).to_numpy().reshape(-1, 1),
        "funding_rate_sum": np.zeros((n_bars, 1)),
        "kill_signal": np.zeros((n_bars, 1)),
        "xs_score_long": np.ones((n_bars, 1)) * 0.5,
        "xs_score_short": np.zeros((n_bars, 1)),
        "hmm_prob_crisis": np.zeros((n_bars, 1)),
        "hmm_modulator_long": np.ones((n_bars, 1)),
        "hmm_modulator_short": np.ones((n_bars, 1)),
        "entry_upper": np.zeros((n_bars, 1)),
        "entry_lower": np.ones((n_bars, 1)) * 999999.0,
        "trend_direction": np.ones((n_bars, 1)),
        "strength_filter": np.ones((n_bars, 1)),
        "slot_rank_score": np.ones((n_bars, 1)),
        "ml_calib_prob": np.zeros((n_bars, 1)),
    }
    
    engine = MultiSymbolEngine(
        aligned_data=aligned_data,
        symbol_names=symbols,
        strategy_params={"REBALANCE_BARS": 6},
        initial_balance=1000.0
    )
    
    trades_df, equity, final_bal, diag = engine.run()
    assert isinstance(trades_df, pd.DataFrame)
    assert final_bal > 0
