import pandas as pd
from pathlib import Path
from config.settings import FUTURES_DATA_DIR
from src.domain.futures.optimization.data_aligner import _dataframe_to_symbol_arrays
from src.domain.futures.backtest_engine import calculate_position_size

sym = 'BTC_USDT'
tf = '1h'
cache_path = FUTURES_DATA_DIR / f'{sym}_{tf}.parquet'

if cache_path.exists():
    df = pd.read_parquet(cache_path)
    print('Loaded', sym, len(df), 'rows')
    # just mock the missing columns to test calculation
    if 'atr' not in df.columns:
        df['atr'] = df['close'] * 0.01
    
    fill_price = df['close'].iloc[-1]
    asset_atr_pct = df['atr'].iloc[-1] / fill_price
    
    qty = calculate_position_size(
        fill_price=fill_price,
        asset_atr_pct=asset_atr_pct,
        current_equity_for_risk=10000.0,
        available_margin=10000.0,
        risk_per_trade=0.02,
        leverage=5.0,
        sf=1.0,
        gk=1.0
    )
    print('Test qty:', qty)
else:
    print('Cache not found', cache_path)
