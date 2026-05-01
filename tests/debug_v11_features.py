import pandas as pd
import numpy as np
from pathlib import Path
from src.domain.futures.data_collector import DataCollector
from src.domain.futures.ml_pipeline.feature_engineering import build_gp_input_features
from config.settings import FUTURES_DATA_DIR

def diagnostic():
    print("Starting v11 Data & Feature Pipeline Diagnostic...")
    symbol = "BTC/USDT"
    tf = "1h"
    start = "2024-01-01"
    end = "2024-02-01"
    
    collector = DataCollector()
    print(f"1. Fetching data for {symbol}...")
    collector.ensure_funding_data(symbol, start, end)
    collector.ensure_metrics_data(symbol, start, end)
    
    df_raw = collector.collect_and_save(symbol, tf, start, end)
    
    from src.domain.futures.funding_utils import merge_funding_into_ohlcv
    from src.domain.futures.metrics_utils import merge_metrics_into_ohlcv
    
    df = merge_funding_into_ohlcv(symbol, df_raw, Path(FUTURES_DATA_DIR))
    df = merge_metrics_into_ohlcv(symbol, df, Path(FUTURES_DATA_DIR))
    
    print("\n2. Data Merge Integrity Check:")
    metrics_cols = [
        "sum_open_interest", "long_short_ratio", 
        "top_trader_long_short_ratio", "taker_buy_sell_vol_value"
    ]
    for col in metrics_cols:
        presence = "OK" if col in df.columns else "MISSING"
        nan_pct = df[col].isna().mean() * 100 if col in df.columns else 100
        print(f"  - {col:<30}: {presence} (NaN: {nan_pct:.2f}%)")
        if col in df.columns and df[col].std() == 0:
            print(f"    [WARNING] {col} has ZERO variance!")

    print("\n3. Feature Calculation (v11) Check:")
    v11_features = build_gp_input_features(df)
    
    new_v11_names = [
        "oi_momentum_24h", "top_trader_lsr_z_24h", "lsr_spread_12h", 
        "cvd_divergence_24h", "absorption_ratio_12h"
    ]
    
    for f in new_v11_names:
        if f in v11_features.columns:
            std = v11_features[f].std()
            nan_pct = v11_features[f].isna().mean() * 100
            print(f"  - {f:<25}: OK (std: {std:.6f}, NaN: {nan_pct:.2f}%)")
            if std < 1e-9:
                print(f"    [CRITICAL] Feature {f} is DEAD (zero variance)!")
        else:
            print(f"  - {f:<25}: MISSING from output!")

    print("\nDiagnostic Complete.")

if __name__ == "__main__":
    diagnostic()
