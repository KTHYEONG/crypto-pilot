import argparse
import pandas as pd
import os
import sys
import optuna
import logging
import sqlite3
import numpy as np
from pathlib import Path
import threading
import time

# Project Root Setup
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    sys.path.append(os.getcwd())

from config.settings import (
    DATA_DIR,
    BACKTEST_START_DATE,
    BACKTEST_END_DATE,
    TRAIN_CUTOFF_DATE,
)
from config.optimization_config_modes import GET_SEARCH_SPACE, BASE_SEARCH_SPACE
from config.optimization_config_ultimate import (
    COMMON_SEARCH_SPACE,
)  # Keep for potential shared usage
from src.data.collector import DataCollector
from src.futures_strategy.strategies_futures import UltimateStrategy
from src.futures_strategy.engine_fast_futures import (
    BacktestEngineFast,
    backtest_loop_numba,
)
from src.optimization.opt_utils import suggest_params, calculate_score


def load_all_timeframes(symbol, start_date, end_date, timeframes):
    """Load all necessary timeframe data into memory"""
    data_map = {}
    collector = DataCollector()

    # Daily Data (Required for Indicators)
    # Even for SCALP mode, daily context is often useful (e.g., trend alignment)
    daily_file = DATA_DIR / f"{symbol.replace('/', '_')}_1d_{start_date}_{end_date}.csv"
    if not daily_file.exists():
        try:
            collector.collect_and_save(symbol, "1d", start_date, end_date)
        except Exception as e:
            print(f"❌ Error: Failed to download {symbol}-1d data: {e}")
            sys.exit(1)

    try:
        df = pd.read_csv(daily_file)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["date_key"] = df["datetime"].dt.strftime("%Y-%m-%d")
        data_map["1d"] = df
    except Exception as e:
        print(f"❌ Error: Failed to load {daily_file}: {e}")
        sys.exit(1)

    for tf in timeframes:
        tf_file = DATA_DIR / f"{symbol.replace('/', '_')}_{tf}_{start_date}_{end_date}.csv"
        
        if not tf_file.exists():
            print(f"Downloading {tf} data...")
            try:
                collector.collect_and_save(symbol, tf, start_date, end_date)
            except Exception as e:
                print(f"❌ Error: Failed to download {symbol}-{tf} data: {e}")
                sys.exit(1)

        try:
            df = pd.read_csv(tf_file)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["date_key"] = df["datetime"].dt.strftime("%Y-%m-%d")
            data_map[tf] = df
        except Exception as e:
            print(f"❌ Error: Failed to load {tf_file}: {e}")
            sys.exit(1)

    return data_map


def compute_merge_indices(data_maps):
    """
    Pre-compute merge index mappings for all symbols and timeframes.
    
    This eliminates the need for pd.merge on every trial by creating a lookup table
    that maps each hourly bar to its corresponding daily bar index.
    
    Returns:
        dict: {symbol: {timeframe: merge_index_array}}
    """
    merge_indices = {}
    
    for symbol, data_map in data_maps.items():
        merge_indices[symbol] = {}
        
        # Get daily date_key mapping
        daily_df = data_map['1d']
        daily_date_keys = daily_df['date_key'].values
        
        # Create a lookup dict: date_key -> daily_index
        date_to_daily_idx = {date_key: idx for idx, date_key in enumerate(daily_date_keys)}
        
        # For each timeframe, create the merge index
        for tf, tf_df in data_map.items():
            if tf == '1d':
                continue  # Skip daily itself
            
            # Map each hourly bar's date_key to its daily index
            hourly_date_keys = tf_df['date_key'].values
            merge_index = np.array([
                date_to_daily_idx.get(date_key, -1) 
                for date_key in hourly_date_keys
            ], dtype=np.int32)
            
            # Handle missing mappings (shouldn't happen with proper data)
            # Replace -1 with 0 and issue warning if found
            if np.any(merge_index == -1):
                print(f"⚠️  Warning: {symbol}-{tf} has unmapped date_keys. Using fallback index 0.")
                merge_index[merge_index == -1] = 0
            
            merge_indices[symbol][tf] = merge_index
    
    return merge_indices









def objective(
    trial, strategy_cls, strategy_name, data_maps, search_space, common_search_space, merge_indices=None
):
    """
    Multi-symbol objective function.
    
    Args:
        merge_indices: Pre-computed merge index mappings (optional, for optimization)
    """
    import gc
    
    # 1. Generate Params
    strategy_params = suggest_params(trial, search_space)
    # Merge common search space if it has unique keys, or ignore if fully handled by search_space
    # In this new design, GET_SEARCH_SPACE returns almost everything.
    common_params = suggest_params(trial, common_search_space)
    full_params = {**strategy_params, **common_params}

    # [VALIDATION] Enforce Logical Constraints
    if full_params.get("TREND_FILTER_TYPE") == "MACD":
        if full_params.get("MACD_FAST", 12) >= full_params.get("MACD_SLOW", 26):
            return -10000  # Invalid trial penalty

    # 2. Select Timeframe
    selected_tf = full_params.get("TIMEFRAME", "1h")

    # 3. Run backtest for EACH symbol
    symbol_scores = []
    symbol_results = {}

    for symbol, data_map in data_maps.items():
        # Ensure timeframe exists
        if selected_tf not in data_map:
            return -10000

        hourly_df = data_map[selected_tf]  # Read-only reference (engine will copy)
        daily_df = data_map["1d"]  # Read-only reference (engine will copy)

        # Create Strategy
        strategy = strategy_cls(f"{strategy_name}_{symbol}", full_params)

        # Engine Execution
        engine = BacktestEngineFast(hourly_df, daily_df, strategy, initial_balance=600)
        
        # [OPTIMIZATION] Inject pre-computed merge index if available
        if merge_indices and symbol in merge_indices and selected_tf in merge_indices[symbol]:
            engine._merge_index_map = merge_indices[symbol][selected_tf]

        # Inject Leverage/Risk
        engine.leverage = full_params.get("LEVERAGE", 1)
        engine.risk_per_trade = full_params.get("RISK_PER_TRADE", 0.02)

        try:
            result = engine.run()
        except Exception as e:
            # [DEBUG] Log backtest failure
            print(f"⚠️ Backtest failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            
            # [MEMORY] Cleanup on error
            del engine, strategy
            gc.collect()
            return -10000

        # Extract Metrics
        ret = result["total_return_pct"]
        mdd = result["mdd_pct"]
        trades = result["total_trades"]
        win_rate = result["win_rate"]

        # Calculate Profit Factor (PF)
        trades_df = result["trades_df"]
        pf = 0.0
        if not trades_df.empty and "pnl" in trades_df.columns:
            gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
            gross_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
            pf = (
                gross_profit / gross_loss
                if gross_loss > 0
                else (gross_profit if gross_profit > 0 else 0.0)
            )

        symbol_results[symbol] = {
            "return": ret,
            "mdd": mdd,
            "trades": trades,
            "win_rate": win_rate,
            "pf": pf,
        }

        # Calculate Score
        # Extract mode from strategy_name (e.g., Ultimate_DAY)
        mode_str = strategy_name.split("_")[-1]
        score = calculate_score(ret, mdd, trades_df, mode=mode_str, market_type="futures", timeframe=selected_tf)

        # [SPEED UP] Short-circuit: If any symbol performs poorly, fail the trial immediately
        # One bad apple spoils the bunch (Harmonic Mean logic)
        # [ADJUSTED] Threshold lowered from 5 to -50 for Simple Interest scale
        if score < -50:
            # [MEMORY] Cleanup before early return
            del engine, result, strategy, trades_df
            gc.collect()
            return -10000

        symbol_scores.append(score)

        # Set individual attrs
        trial.set_user_attr(f"ret_{symbol.replace('/', '_')}", float(ret))
        trial.set_user_attr(f"mdd_{symbol.replace('/', '_')}", float(mdd))
        trial.set_user_attr(f"pf_{symbol.replace('/', '_')}", float(pf))
        
        # [MEMORY] Explicit cleanup per symbol
        del engine, result, strategy, trades_df

    # 4. Combine scores using HARMONIC MEAN
    offset = 200  # [ADJUSTED] Reduced from 6000 for Simple Interest scale
    shifted_scores = [s + offset for s in symbol_scores]

    if any(s <= 0 for s in shifted_scores):
        final_score = -10000
    else:
        harmonic_mean = len(shifted_scores) / sum(1 / s for s in shifted_scores)
        final_score = harmonic_mean - offset

    # Record Average Attributes
    avg_ret = np.mean([r["return"] for r in symbol_results.values()])
    avg_mdd = np.mean([r["mdd"] for r in symbol_results.values()])
    avg_pf = np.mean([r["pf"] for r in symbol_results.values()])

    trial.set_user_attr("return_avg", avg_ret)
    trial.set_user_attr("mdd_avg", avg_mdd)
    trial.set_user_attr("pf_avg", avg_pf)
    
    # [MEMORY] Force garbage collection after processing all symbols
    gc.collect()

    return final_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of optimization trials (default: auto-set by mode)",
    )
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        default="UNIFIED",
        choices=["SCALP", "DAY", "SWING", "UNIFIED", "ALL"],
        help="Trading Mode: SCALP, DAY, SWING, or UNIFIED (recommended - auto-selects best timeframe)",
    )
    args = parser.parse_args()

    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",")]
    mode = args.mode.upper()

    # Auto-set trials based on mode if not specified
    # Formula: Parameters × 150 (TPE rule of thumb)
    # All modes have 3 timeframes, adjusted by backtest speed and trade frequency
    # [UPDATED] Trials adjusted based on search space complexity and data volume analysis
    MODE_TRIALS_MAP = {
        "SCALP": 3600,  # 3 timeframes (5m,15m,30m), high data volume but narrow param range
        "DAY": 4200,    # 3 timeframes (1h,2h,4h), balanced - most commonly used mode
        "SWING": 5000,  # 3 timeframes (4h,1d,3d), wide param range + low data volume (overfitting risk)
        "UNIFIED": 8000, # Increased for 55+ parameters and more timeframes (15m, 30m)
        "ALL": 8000,     # Alias for UNIFIED
    }

    if args.trials is None:
        trials = MODE_TRIALS_MAP.get(mode, 2500)
        print(f"ℹ️  Auto-setting trials for {mode} mode: {trials}")
    else:
        trials = args.trials
        print(f"ℹ️  Using custom trials: {trials}")

    # Adjust Logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("src.futures_strategy.engine_fast_futures").setLevel(
        logging.WARNING
    )

    # Get Search Space & Timeframes
    try:
        search_space = GET_SEARCH_SPACE(mode, market_type="futures")
        timeframes = search_space["TIMEFRAME"]["choices"]
    except Exception as e:
        print(f"❌ Error: Failed to load search space for mode '{mode}'")
        print(f"   Details: {e}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"🚀 MODE: {mode} OPRIMIZATION")
    print(f"⏰ Target Timeframes: {timeframes}")
    # print(f"🔍 Search Space Size: {len(search_space)} parameters")
    print(f"{'='*70}\n")

    data_maps = {}
    print(f"📡 Loading data for symbols: {', '.join(symbols)}")

    # [CRITICAL] Ensure '1d' is loaded for HTF Trend Filter, even if not optimizing on it
    loading_timeframes = list(set(timeframes + ['1d']))
    
    for symbol in symbols:
        print(f"Loading {symbol}...")
        data_maps[symbol] = load_all_timeframes(
            symbol, BACKTEST_START_DATE, BACKTEST_END_DATE, loading_timeframes
        )
        
        # [VALIDATION] Ensure all required data is loaded
        if not data_maps[symbol] or '1d' not in data_maps[symbol]:
            print(f"❌ Error: Failed to load data for {symbol}")
            sys.exit(1)
        for tf in timeframes:
            if tf not in data_maps[symbol] or data_maps[symbol][tf].empty:
                print(f"❌ Error: Failed to load {symbol}-{tf} data")
                sys.exit(1)
    print(f"✅ Data loaded successfully for all symbols")

    # [CRITICAL] Slice Data for Optimization (Train Set) with Warmup Buffer
    print(f"✂️  Trimming Data for Optimization (Train Period: ~ {TRAIN_CUTOFF_DATE})")
    cutoff_ts = pd.Timestamp(TRAIN_CUTOFF_DATE)
    
    # [WARMUP OPTIMIZATION] Include buffer period before cutoff for indicator calculation
    # This prevents NaN values at the start of training period
    WARMUP_BUFFER_BARS = {
        '5m': 500,   # ~42 hours of data for warmup
        '15m': 400,  # ~100 hours
        '30m': 350,  # ~220 hours
        '1h': 300,   # ~12.5 days
        '2h': 250,   # ~20 days
        '4h': 200,   # ~33 days
        '1d': 150,   # ~5 months
        '3d': 100,   # ~10 months
    }

    for sym in data_maps:
        for tf in data_maps[sym]:
            df = data_maps[sym][tf]
            original_len = len(df)
            
            # Find cutoff index (end of training period)
            cutoff_mask = df["datetime"] < cutoff_ts
            train_end_idx = cutoff_mask.sum()
            
            if train_end_idx == 0:
                print(f"⚠️  Warning: {sym}-{tf} has no data before cutoff date.")
                continue
            
            # Desired warmup period
            desired_warmup = WARMUP_BUFFER_BARS.get(tf, 200)
            
            # Slice from start to cutoff (entire training period)
            data_maps[sym][tf] = df.iloc[:train_end_idx].copy()
            
            # Set warmup: first N bars are for indicator warmup, trading starts after
            data_maps[sym][tf].attrs['warmup_bars'] = min(desired_warmup, train_end_idx)
            
            new_len = len(data_maps[sym][tf])
            # if tf == timeframes[0]:
            #     print(f"  [{sym}] Train Size: {new_len} (Original: {original_len}, Warmup: {desired_warmup})")

    # [OPTIMIZATION] Pre-compute merge indices to eliminate pd.merge overhead
    print(f"🔗 Pre-computing merge indices for fast data alignment...")
    merge_indices = compute_merge_indices(data_maps)
    print(f"✅ Merge indices computed for {len(merge_indices)} symbols")

    # DB Setup (MySQL)
    from dotenv import load_dotenv
    from urllib.parse import quote_plus

    load_dotenv()

    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "trading_optuna")

    if not all([db_user, db_pass, db_name]):
        print("❌ Error: Missing DB credentials in .env (DB_USER, DB_PASS, DB_NAME)")
        sys.exit(1)

    study_name = f"futures_{mode.lower()}_strategy"
    # [CRITICAL] Encode password to handle special characters like '@'
    safe_pass = quote_plus(db_pass)
    storage_url = f"mysql+pymysql://{db_user}:{safe_pass}@{db_host}:{db_port}/{db_name}"

    # [Clean Start] Intead of deleting file, delete study from DB
    print(f"🔄 Preparing study: {study_name}")
    
    # [VALIDATION] Test DB connection before proceeding
    try:
        test_storage = optuna.storages.RDBStorage(url=storage_url)
        print(f"✅ DB connection successful")
    except Exception as e:
        print(f"❌ Error: Failed to connect to MySQL database")
        print(f"   Details: {e}")
        print(f"   Please check your .env credentials and ensure MySQL is running")
        sys.exit(1)
    
    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
        print(f"🗑️  Deleted old study: {study_name}")
    except Exception:
        pass  # Study might not exist

    # [Performance] Optimize for parallel MySQL access
    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={
            "pool_size": max(30, args.jobs * 2),  # Scale with jobs
            "max_overflow": 10,  # Allow burst connections
            "pool_recycle": 3600,
            "pool_pre_ping": True,  # Validate connections
        },
    )

    # [Performance] Use ConstantLiar for parallel efficiency
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=2000,  # 25% of 8000 trials for thorough random exploration
        multivariate=True,  # Consider param dependencies
        constant_liar=True,  # Avoid duplicate proposals
        warn_independent_sampling=False,
    )

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize", sampler=sampler
    )

    print(f"\n{'='*70}")
    print(f"🔥 STARTING OPTIMIZATION for {study_name}")
    print(f"🛢️  Storage: MySQL ({db_host}/{db_name})")
    print(f"📈 Total Trials: {trials}")
    print(f"💻 Parallel Jobs: {args.jobs}")
    print(f"{'='*70}\n")

    # [Performance] Numba JIT Warmup
    print("🔥 Warming up Numba JIT...", end="", flush=True)
    dummy_len = 10
    _dummy_arr = np.ones(dummy_len, dtype=np.float64)
    _dummy_int = np.zeros(dummy_len, dtype=np.int64)
    _dummy_ts = np.zeros(dummy_len, dtype=np.int64)  # Timestamps
    try:
        backtest_loop_numba(
            _dummy_arr,
            _dummy_arr,
            _dummy_arr,  # OHLC (close, high, low)
            _dummy_arr,  # Open Prices [NEW]
            _dummy_arr,  # Volume Ratio
            _dummy_arr,
            _dummy_arr,  # Entry Upper/Lower
            _dummy_int,
            _dummy_int,
            _dummy_arr,  # Trend, Strength, ATR
            _dummy_arr,  # Parabolic SAR
            _dummy_arr,  # RSI
            _dummy_arr,  # [NEW] Hurst
            _dummy_arr,  # [NEW] NATR
            10000.0,
            1.0,
            0.001,
            0.001,  # Bal, Lev, Fee, Slip
            0,  # Exit Type (0=Trailing, 1=SAR)
            0,
            0.01,
            1.5,  # SL Type, Pct, Mult
            3.0,  # ATR Mult
            0.02,  # Risk
            False,
            1.0,  # Vol Filter
            False,
            3.0,  # TP
            _dummy_ts,
            0.0001,
            8,  # Funding params
            1000,
            0.0,  # Max Hold, Trailing Act
            0.5,  # Time-based Exit Profit Threshold
            80.0, # RSI Exit Threshold
            False, # [NEW] Use Dynamic Risk
            0.6,   # Strong Hurst
            1.5,   # Strong NATR
            1.5,   # Strong Multiplier
            0.55,  # Weak Hurst
            0.5,   # Weak Multiplier
            4.0,   # Panic NATR
            0.25,  # Panic Multiplier
            0,     # Warmup bars
            False, # [NEW] use_compounding
            1000000.0 # [NEW] max_capital_usage
        )
        print(" Done!")
    except Exception as e:
        print(f"\n⚠️  Warning: Numba warmup failed: {e}")
        print(f"   First trial will be slower due to JIT compilation")

    try:
        study.optimize(
            lambda trial: objective(
                trial, UltimateStrategy, f"Ultimate_{mode}", data_maps, search_space, {}, merge_indices  # common_search_space={} (handled by GET_SEARCH_SPACE)
            ),
            n_trials=trials,
            n_jobs=args.jobs,
            show_progress_bar=True,
        )

    except KeyboardInterrupt:
        print("\n🛑 Optimization Interrupted by User")
        print(f"💾 Progress saved: {len(study.trials)} trials completed")
    except Exception as e:
        print(f"\n❌ Optimization failed with error: {e}")
        print(f"💾 Progress saved: {len(study.trials)} trials completed before failure")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"✅ {mode} Optimization Complete!")

    if len(study.trials) > 0:
        print(f"🏆 Best Score: {study.best_value:.2f}")
        print(f"✨ Best Params: {study.best_params}")
        
        # [NEW] Detailed Report for TRAIN Period
        print(f"\n{'='*70}")
        print(f"📊 TRAIN PERIOD PERFORMANCE (Best Strategy)")
        print(f"{'='*70}")
        
        best_params = study.best_params
        # Merge fixed params if any (handled inside objective but good to be safe)
        
        selected_tf = best_params.get('TIMEFRAME', '1h')
        
        for symbol in symbols:
            if selected_tf not in data_maps[symbol]:
                continue
                
            hourly_df = data_maps[symbol][selected_tf]
            daily_df = data_maps[symbol]['1d']
            
            # Re-create Strategy & Engine
            strategy = UltimateStrategy(f"Best_{symbol}", best_params)
            engine = BacktestEngineFast(hourly_df, daily_df, strategy, initial_balance=600)
            
            # Inject pre-computed merge index
            if merge_indices and symbol in merge_indices and selected_tf in merge_indices[symbol]:
                engine._merge_index_map = merge_indices[symbol][selected_tf]
                
            engine.leverage = best_params.get('LEVERAGE', 1)
            engine.risk_per_trade = best_params.get('RISK_PER_TRADE', 0.02)
            
            try:
                res = engine.run()
                
                ret = res['total_return_pct']
                mdd = res['mdd_pct']
                cnt = res['total_trades']
                win = res['win_rate']
                
                trades_df = res['trades_df']
                pf = 0.0
                if not trades_df.empty and 'pnl' in trades_df.columns:
                    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
                    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
                    pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
                
                print(f"   - {symbol:<9} : Return {ret:>7.2f}% | MDD {mdd:>6.2f}% | Trades {cnt:>3} | Win {win:>5.1f}% | PF {pf:.2f}")
                
            except Exception as e:
                print(f"   - {symbol:<9} : Error calculating performance: {e}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
