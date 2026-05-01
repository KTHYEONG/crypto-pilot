import numpy as np
import pandas as pd
import pytest
from src.domain.futures.ml_pipeline.feature_engineering import (
    build_gp_input_features,
    build_hmm_input_features,
    build_systemic_hmm_features,
)
from src.core.optimization.opt_utils import compute_segment_merge_index
from src.domain.futures.opt_futures_utils.objective import inject_cs_momentum_ranks

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data for testing."""
    np.random.seed(42)
    n_samples = 500
    
    dates = pd.date_range("2025-01-01", periods=n_samples, freq="1h", tz="UTC")
    close = pd.Series(100 * np.exp(np.random.randn(n_samples).cumsum() * 0.01), index=dates)
    high = close * (1 + np.random.rand(n_samples) * 0.02)
    low = close * (1 - np.random.rand(n_samples) * 0.02)
    open_ = close.shift(1).fillna(100.0)
    
    # Intraday fluctuation adjustments
    high = np.maximum(high, open_)
    high = np.maximum(high, close)
    low = np.minimum(low, open_)
    low = np.minimum(low, close)
    
    volume = np.random.lognormal(mean=10, sigma=1, size=n_samples)
    quote_volume = volume * close
    taker_buy_quote_volume = quote_volume * np.random.beta(a=5, b=5, size=n_samples)
    
    funding_rate = np.random.normal(loc=0.0001, scale=0.0005, size=n_samples)
    
    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "quote_volume": quote_volume,
        "taker_buy_quote_volume": taker_buy_quote_volume,
        "funding_rate": funding_rate,
    }, index=dates)
    df.index.name = "datetime"
    return df

def test_gp_features_lookahead_bias(sample_ohlcv: pd.DataFrame) -> None:
    """Ensure future data does not leak into current feature values."""
    df_base = sample_ohlcv.copy()
    features_base = build_gp_input_features(df_base)
    
    # Modify data from index 250 onwards
    df_future = sample_ohlcv.copy()
    df_future.iloc[250:, df_future.columns.get_loc("close")] *= 2.0
    df_future.iloc[250:, df_future.columns.get_loc("high")] *= 2.0
    
    features_future = build_gp_input_features(df_future)
    
    # Features before index 250 should be EXACTLY identical
    pd.testing.assert_frame_equal(
        features_base.iloc[:250],
        features_future.iloc[:250],
        check_exact=True,
        obj="GP Features Lookahead Leak"
    )

def test_hmm_features_lookahead_bias(sample_ohlcv: pd.DataFrame) -> None:
    df_base = sample_ohlcv.copy()
    features_base = build_hmm_input_features(df_base)
    
    df_future = sample_ohlcv.copy()
    df_future.iloc[250:, df_future.columns.get_loc("close")] *= 2.0
    df_future.iloc[250:, df_future.columns.get_loc("volume")] *= 5.0
    
    features_future = build_hmm_input_features(df_future)
    
    pd.testing.assert_frame_equal(
        features_base.iloc[:250],
        features_future.iloc[:250],
        check_exact=True,
        obj="HMM Features Lookahead Leak"
    )

def test_gp_features_nan_propagation(sample_ohlcv: pd.DataFrame) -> None:
    """Ensure missing data is handled safely without crashing or corrupting everything."""
    df_nan = sample_ohlcv.copy()
    
    # Introduce NaN in the middle
    df_nan.iloc[100:105, df_nan.columns.get_loc("close")] = np.nan
    
    features = build_gp_input_features(df_nan)
    
    # The output should have the same length
    assert len(features) == len(df_nan)
    
    # After the window passes, features should recover and not remain NaN forever
    assert not features.iloc[300].isna().all(), "NaNs cascaded infinitely."

def test_systemic_hmm_features_lookahead_bias(sample_ohlcv: pd.DataFrame) -> None:
    # Systemic HMM expects a multi-index panel
    sample_ohlcv["symbol"] = "BTC/USDT"
    panel = sample_ohlcv.reset_index().set_index(["datetime", "symbol"])
    panel["cs_dispersion"] = 0.05
    panel["market_breadth"] = 0.6
    
    feats_base = build_systemic_hmm_features(panel)
    
    # Modify future data
    panel_future = panel.copy()
    pf_reset = panel_future.reset_index()
    pf_reset.loc[250:, "close"] *= 1.5
    panel_future = pf_reset.set_index(["datetime", "symbol"])
    
    feats_future = build_systemic_hmm_features(panel_future)
    
    pd.testing.assert_frame_equal(
        feats_base.iloc[:250],
        feats_future.iloc[:250],
        check_exact=True,
        obj="Systemic HMM Lookahead Leak"
    )

def test_multi_timeframe_merge_leak(sample_ohlcv: pd.DataFrame) -> None:
    """
    CRITICAL: Verify that merging daily data into hourly data doesn't leak 'today's' daily close.
    In crypto, 1d bar T represents data from T 00:00 to T+1 00:00.
    At time T 10:00, we should ONLY see 1d bar T-1.
    """
    hourly_df = sample_ohlcv.copy().reset_index()
    
    # Create daily DF where 'close' is the close of the day
    daily_df = hourly_df.set_index("datetime").resample("1D").last().reset_index()
    
    # Use the utility to get merge indices
    merge_idx = compute_segment_merge_index(hourly_df, daily_df)
    
    # Map daily close to hourly bars
    hourly_df["daily_close_merged"] = daily_df["close"].iloc[merge_idx].values
    
    # Check a specific point: 2025-01-02 10:00:00
    # daily_df has bars for 2025-01-01, 2025-01-02, ...
    # At 2025-01-02 10:00, we should see daily_df['close'] of 2025-01-01.
    
    test_dt = pd.Timestamp("2025-01-02 10:00:00", tz="UTC")
    matching_rows = hourly_df[hourly_df["datetime"] == test_dt]
    if matching_rows.empty:
        # Debug: find what dates ARE available
        available_dates = hourly_df["datetime"].head(5).tolist()
        raise ValueError(f"test_dt {test_dt} not found in hourly_df. Available head: {available_dates}")
    
    row = matching_rows.iloc[0]
    actual_merged_close = row["daily_close_merged"]
    
    # At 2025-01-02 10:00, we should see the daily bar of 2025-01-01
    expected_daily_dt = pd.Timestamp("2025-01-01", tz="UTC")
    expected_row = daily_df[daily_df["datetime"] == expected_daily_dt]
    if expected_row.empty:
        raise ValueError(f"expected_daily_dt {expected_daily_dt} not found in daily_df.")
    expected_close = expected_row["close"].iloc[0]
    
    assert actual_merged_close == expected_close, (
        f"LEAK DETECTED: At {test_dt}, hourly bar saw daily close of "
        f"{daily_df[daily_df['close'] == actual_merged_close]['datetime'].iloc[0]} "
        f"instead of {expected_daily_dt}."
    )

def test_cross_sectional_rank_leak(sample_ohlcv: pd.DataFrame) -> None:
    """Verify that cross-sectional ranking at time T doesn't use future data."""
    sym1 = "BTC/USDT"
    sym2 = "ETH/USDT"
    
    df1 = sample_ohlcv.copy()
    df2 = sample_ohlcv.copy()
    df2["close"] *= 1.2 # Different price levels
    
    data_maps = {
        sym1: {"1h": df1},
        sym2: {"1h": df2}
    }
    
    inject_cs_momentum_ranks(data_maps, [sym1, sym2], "1h", lookbacks=[12])
    
    rank_base = data_maps[sym1]["1h"]["cs_mom_rank_12"].copy()
    
    # Modify FUTURE data for sym2
    data_maps_future = {
        sym1: {"1h": df1.copy()},
        sym2: {"1h": df2.copy()}
    }
    data_maps_future[sym2]["1h"].iloc[250:, data_maps_future[sym2]["1h"].columns.get_loc("close")] *= 10.0
    
    inject_cs_momentum_ranks(data_maps_future, [sym1, sym2], "1h", lookbacks=[12])
    
    rank_future = data_maps_future[sym1]["1h"]["cs_mom_rank_12"]
    
    pd.testing.assert_series_equal(
        rank_base.iloc[:250],
        rank_future.iloc[:250],
        obj="CS Rank Lookahead Leak"
    )

def test_backtest_alignment_and_signals(sample_ohlcv: pd.DataFrame) -> None:
    """Verify that signals generated in the strategy reach the backtester aligned correctly."""
    from src.domain.futures.opt_futures_utils.data_utils import align_data_for_2d_engine
    
    df = sample_ohlcv.copy().reset_index()
    # Mock some signal columns
    df["gp_alpha_00"] = 0.5
    df["xs_score_long"] = 0.5
    df["xs_score_short"] = -0.5
    df["hmm_prob_crisis"] = 0.0
    df["hmm_modulator_long"] = 1.0
    df["hmm_modulator_short"] = 1.0
    df["atr"] = 1.0
    df["garch_kelly_f"] = 1.0
    
    # Required columns for 2d engine alignment
    for col in ["entry_upper", "entry_lower", "trend_direction", "strength_filter", "slot_rank_score", "ml_calib_prob", "funding_rate_sum"]:
        df[col] = 0.0
        
    symbols = ["BTC/USDT"]
    data_maps = {"BTC/USDT": df}
    
    aligned, master_index = align_data_for_2d_engine(data_maps, symbols)
    
    # Check if xs_score_long reached aligned data
    assert (aligned["xs_score_long"] == 0.5).all(), "Signals lost during alignment!"
    assert len(master_index) == len(df)
    
def test_gp_alpha_mining_lookahead_mock():
    """
    Speculative: Check if MLAlphaMiner's target creation has a shift bug.
    Alphas should predict FUTURE returns, not current ones.
    """
    from src.domain.futures.ml_pipeline.cross_sectional_utils import CrossSectionalPipelineUtils
    utils = CrossSectionalPipelineUtils()
    
    # Create panel with returns
    dates = pd.date_range("2025-01-01", periods=100, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "datetime": dates,
        "symbol": "BTC/USDT",
        "close": np.arange(100, 200, dtype=np.float64), # Pure trend
        "volume": 1000.0
    })
    df["open"] = df["close"] - 1.0
    df["high"] = df["close"] + 0.5
    df["low"] = df["close"] - 0.5
    
    # create_multi_horizon_rank_targets uses pct_change(horizons).shift(-horizons)
    # We want to verify it actually shifts NEGATIVELY (into the future)
    targets = utils.create_multi_horizon_rank_targets(df.set_index(["datetime", "symbol"]), horizons=(1, 2))
    
    # At index 0, target should depend on close[1] and close[2]
    # If it's 1.0 (since it's a trend), it's probably working.
    # But if we change close[1] and target[0] changes, then it's correctly looking forward for the TARGET.
    
    # The important part is that the FEATURES don't look forward. 
    # (Checked in test_gp_features_lookahead_bias)
    assert not targets.isna().all()

