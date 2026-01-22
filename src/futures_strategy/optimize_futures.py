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


def load_all_timeframes(symbol, start_date, end_date, timeframes):
    """Load all necessary timeframe data into memory"""
    data_map = {}
    collector = DataCollector()

    # Daily Data (Required for Indicators)
    # Even for SCALP mode, daily context is often useful (e.g., trend alignment)
    daily_file = DATA_DIR / f"{symbol.replace('/', '_')}_1d_{start_date}_{end_date}.csv"
    if not daily_file.exists():
        collector.collect_and_save(symbol, "1d", start_date, end_date)

    df = pd.read_csv(daily_file)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    # [OPTIMIZATION] Pre-calculate date_key for merging
    df["date_key"] = df["datetime"].dt.strftime("%Y-%m-%d")
    data_map["1d"] = df

    for tf in timeframes:
        tf_file = (
            DATA_DIR / f"{symbol.replace('/', '_')}_{tf}_{start_date}_{end_date}.csv"
        )
        if not tf_file.exists():
            print(f"Downloading {tf} data...")
            collector.collect_and_save(symbol, tf, start_date, end_date)

        df = pd.read_csv(tf_file)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        # [OPTIMIZATION] Pre-calculate date_key
        df["date_key"] = df["datetime"].dt.strftime("%Y-%m-%d")
        data_map[tf] = df

    return data_map


class CheckpointCallback:
    """
    Hybrid Storage Callback for Futures:
    Saves in-memory study to SQLite database every `interval` trials.
    """

    def __init__(self, db_url, interval=100):
        self.db_url = db_url
        self.interval = interval
        self._lock = threading.Lock()

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        # Save every interval trials
        if trial.number > 0 and trial.number % self.interval == 0:
            with self._lock:  # Prevent parallel checkpointing
                db_path = self.db_url.replace("sqlite:///", "")
                temp_path = db_path + ".tmp"

                try:
                    # Cleanup old temp
                    for f in [temp_path, temp_path + "-wal", temp_path + "-shm"]:
                        if os.path.exists(f):
                            try:
                                os.remove(f)
                            except:
                                pass

                    # Copy to temp storage
                    temp_storage = optuna.storages.RDBStorage(
                        url=f"sqlite:///{temp_path}"
                    )
                    optuna.copy_study(
                        from_study_name=study.study_name,
                        from_storage=study._storage,
                        to_storage=temp_storage,
                        to_study_name=study.study_name,
                    )

                    # [CRITICAL] Dispose engine to release file handle on Windows
                    if hasattr(temp_storage, "_engine"):
                        temp_storage._engine.dispose()
                    del temp_storage

                    time.sleep(1.0)  # Grace period for OS

                    # Move temp to actual (simulating atomic overwrite)
                    if os.path.exists(db_path):
                        try:
                            os.remove(db_path)
                            for ext in ["-wal", "-shm"]:
                                if os.path.exists(db_path + ext):
                                    os.remove(db_path + ext)
                        except PermissionError:
                            return

                    os.rename(temp_path, db_path)
                    print(
                        f"💾 Checkpoint: Progress saved to {db_path} (Trial {trial.number})"
                    )
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        print(f"⚠️ Checkpoint failed: {e}")


def suggest_params(trial, search_space):
    """
    Generate trial parameters from search space with conditional dependency pruning.
    Only suggests parameters that are actually used by the selected strategy configuration.

    Efficiency Gain: 60~70% reduction in search space by skipping irrelevant parameters.
    """
    params = {}

    # === Phase 1: Core Strategy Selection ===
    for key in [
        "ENTRY_TYPE",
        "TREND_FILTER_TYPE",
        "STRENGTH_FILTER_TYPE",
        "EXIT_TYPE",
        "STOP_LOSS_TYPE",
        "USE_TAKE_PROFIT",
        "USE_VOLUME_FILTER",
        "TIMEFRAME",
    ]:
        if key in search_space:
            spec = search_space[key]
            if spec["type"] == "categorical":
                params[key] = trial.suggest_categorical(key, spec["choices"])

    # === Phase 2: Entry-Type Dependent Parameters ===
    entry_type = params.get("ENTRY_TYPE", "DONCHIAN")

    if entry_type == "BOLLINGER":
        if "BB_STD" in search_space:
            spec = search_space["BB_STD"]
            params["BB_STD"] = trial.suggest_float(
                "BB_STD", spec["low"], spec["high"], step=spec.get("step")
            )

    elif entry_type == "KELTNER":
        if "KELTNER_ATR_MULT" in search_space:
            spec = search_space["KELTNER_ATR_MULT"]
            params["KELTNER_ATR_MULT"] = trial.suggest_float(
                "KELTNER_ATR_MULT", spec["low"], spec["high"], step=spec.get("step")
            )

    elif entry_type == "CCI":
        if "CCI_THRESHOLD" in search_space:
            spec = search_space["CCI_THRESHOLD"]
            params["CCI_THRESHOLD"] = trial.suggest_int(
                "CCI_THRESHOLD", spec["low"], spec["high"], step=spec.get("step")
            )

    # === Phase 3: Trend-Filter Dependent Parameters ===
    trend_filter = params.get("TREND_FILTER_TYPE", "EMA")

    if trend_filter == "SUPERTREND":
        for key in ["SUPERTREND_MULT", "SUPERTREND_PERIOD"]:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get("log", False)
                if spec["type"] == "float":
                    params[key] = (
                        trial.suggest_float(key, spec["low"], spec["high"], log=use_log)
                        if use_log
                        else trial.suggest_float(
                            key, spec["low"], spec["high"], step=spec.get("step")
                        )
                    )
                elif spec["type"] == "int":
                    params[key] = (
                        trial.suggest_int(key, spec["low"], spec["high"], log=use_log)
                        if use_log
                        else trial.suggest_int(
                            key, spec["low"], spec["high"], step=spec.get("step")
                        )
                    )

    elif trend_filter == "MACD":
        for key in ["MACD_FAST", "MACD_SLOW", "MACD_SIGNAL"]:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get("log", False)
                params[key] = (
                    trial.suggest_int(key, spec["low"], spec["high"], log=use_log)
                    if use_log
                    else trial.suggest_int(
                        key, spec["low"], spec["high"], step=spec.get("step")
                    )
                )

    elif trend_filter == "ICHIMOKU":
        for key in ["ICHIMOKU_TENKAN", "ICHIMOKU_KIJUN", "ICHIMOKU_SENKOU_B"]:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get("log", False)
                params[key] = (
                    trial.suggest_int(key, spec["low"], spec["high"], log=use_log)
                    if use_log
                    else trial.suggest_int(
                        key, spec["low"], spec["high"], step=spec.get("step")
                    )
                )

    elif trend_filter == "VWAP":
        if "VWAP_STD_MULT" in search_space:
            spec = search_space["VWAP_STD_MULT"]
            params["VWAP_STD_MULT"] = trial.suggest_float(
                "VWAP_STD_MULT", spec["low"], spec["high"], step=spec.get("step")
            )

    # SMA, EMA, HMA, DEMA, TEMA는 공통 MA_PERIOD 사용 (아래에서 처리)

    # === Phase 4: Strength-Filter Dependent Parameters ===
    strength_filter = params.get("STRENGTH_FILTER_TYPE", "NONE")

    if strength_filter in ["ADX", "VHF", "MFI", "RSI", "STOCHASTIC", "STOCH_RSI"]:
        # STRENGTH_FILTER_PERIOD는 공통 사용
        if "STRENGTH_FILTER_PERIOD" in search_space:
            spec = search_space["STRENGTH_FILTER_PERIOD"]
            use_log = spec.get("log", False)
            params["STRENGTH_FILTER_PERIOD"] = (
                trial.suggest_int(
                    "STRENGTH_FILTER_PERIOD", spec["low"], spec["high"], log=use_log
                )
                if use_log
                else trial.suggest_int(
                    "STRENGTH_FILTER_PERIOD",
                    spec["low"],
                    spec["high"],
                    step=spec.get("step"),
                )
            )

    if strength_filter == "VHF":
        if "VHF_THRESHOLD" in search_space:
            spec = search_space["VHF_THRESHOLD"]
            params["VHF_THRESHOLD"] = trial.suggest_float(
                "VHF_THRESHOLD", spec["low"], spec["high"], step=spec.get("step")
            )

    elif strength_filter == "MFI":
        if "MFI_THRESHOLD" in search_space:
            spec = search_space["MFI_THRESHOLD"]
            params["MFI_THRESHOLD"] = trial.suggest_int(
                "MFI_THRESHOLD", spec["low"], spec["high"], step=spec.get("step")
            )

    elif strength_filter == "RSI":
        for key in ["RSI_OVERBOUGHT", "RSI_OVERSOLD"]:
            if key in search_space:
                spec = search_space[key]
                params[key] = trial.suggest_int(
                    key, spec["low"], spec["high"], step=spec.get("step")
                )

    elif strength_filter == "STOCHASTIC":
        for key in ["STOCH_OVERBOUGHT", "STOCH_OVERSOLD"]:
            if key in search_space:
                spec = search_space[key]
                params[key] = trial.suggest_int(
                    key, spec["low"], spec["high"], step=spec.get("step")
                )

    elif strength_filter == "STOCH_RSI":
        for key in ["STOCH_RSI_OVERBOUGHT", "STOCH_RSI_OVERSOLD"]:
            if key in search_space:
                spec = search_space[key]
                params[key] = trial.suggest_int(
                    key, spec["low"], spec["high"], step=spec.get("step")
                )

    elif strength_filter == "CMF":
        for key in ["CMF_PERIOD", "CMF_THRESHOLD"]:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get("log", False)
                if spec["type"] == "int":
                    params[key] = (
                        trial.suggest_int(key, spec["low"], spec["high"], log=use_log)
                        if use_log
                        else trial.suggest_int(
                            key, spec["low"], spec["high"], step=spec.get("step")
                        )
                    )
                else:
                    params[key] = trial.suggest_float(
                        key, spec["low"], spec["high"], step=spec.get("step")
                    )

    elif strength_filter == "HURST":
        for key in ["HURST_PERIOD", "HURST_TREND_THRESHOLD", "HURST_RANDOM_THRESHOLD"]:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get("log", False)
                if spec["type"] == "int":
                    params[key] = (
                        trial.suggest_int(key, spec["low"], spec["high"], log=use_log)
                        if use_log
                        else trial.suggest_int(
                            key, spec["low"], spec["high"], step=spec.get("step")
                        )
                    )
                else:
                    params[key] = trial.suggest_float(
                        key, spec["low"], spec["high"], step=spec.get("step")
                    )

    # ADX는 threshold만 필요 (아래에서 처리)

    # === Phase 5: Exit-Type Dependent Parameters ===
    exit_type = params.get("EXIT_TYPE", "ATR")

    if exit_type == "PARABOLIC_SAR":
        if "SAR_STEP" in search_space:
            spec = search_space["SAR_STEP"]
            params["SAR_STEP"] = trial.suggest_float(
                "SAR_STEP", spec["low"], spec["high"], step=spec.get("step")
            )

    # === Phase 6: Common Parameters (Always Used) ===
    # [CRITICAL] Handle STOP_LOSS_TYPE logical conflict first
    stop_loss_type = params.get("STOP_LOSS_TYPE", "FIXED")

    if stop_loss_type == "FIXED":
        # FIXED: Use percentage-based stop loss
        if "STOP_LOSS_PCT" in search_space:
            spec = search_space["STOP_LOSS_PCT"]
            params["STOP_LOSS_PCT"] = trial.suggest_float(
                "STOP_LOSS_PCT", spec["low"], spec["high"], step=spec.get("step")
            )
        # ATR_STOP_LOSS_MULT is NOT suggested (would be ignored by engine)

    elif stop_loss_type == "ATR":
        # ATR: Use ATR-based stop loss
        if "ATR_STOP_LOSS_MULT" in search_space:
            spec = search_space["ATR_STOP_LOSS_MULT"]
            use_log = spec.get("log", False)
            if use_log:
                params["ATR_STOP_LOSS_MULT"] = trial.suggest_float(
                    "ATR_STOP_LOSS_MULT", spec["low"], spec["high"], log=True
                )
            else:
                params["ATR_STOP_LOSS_MULT"] = trial.suggest_float(
                    "ATR_STOP_LOSS_MULT",
                    spec["low"],
                    spec["high"],
                    step=spec.get("step"),
                )
        # STOP_LOSS_PCT is NOT suggested (would be ignored by engine)

    # [CRITICAL] Handle USE_TAKE_PROFIT logical conflict
    use_take_profit = params.get("USE_TAKE_PROFIT", False)

    if use_take_profit:
        # TP enabled: suggest TP parameters
        if "TAKE_PROFIT_ATR_MULT" in search_space:
            spec = search_space["TAKE_PROFIT_ATR_MULT"]
            use_log = spec.get("log", False)
            if use_log:
                params["TAKE_PROFIT_ATR_MULT"] = trial.suggest_float(
                    "TAKE_PROFIT_ATR_MULT", spec["low"], spec["high"], log=True
                )
            else:
                params["TAKE_PROFIT_ATR_MULT"] = trial.suggest_float(
                    "TAKE_PROFIT_ATR_MULT",
                    spec["low"],
                    spec["high"],
                    step=spec.get("step"),
                )
    # else: TAKE_PROFIT_ATR_MULT is NOT suggested (would be ignored)

    # [CRITICAL] Handle USE_VOLUME_FILTER logical conflict
    use_volume_filter = params.get("USE_VOLUME_FILTER", False)

    if use_volume_filter:
        # Volume filter enabled: suggest volume parameters
        for key in ["VOLUME_THRESHOLD_MULT", "VOLUME_MA_PERIOD"]:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get("log", False)
                if spec["type"] == "float":
                    if use_log:
                        params[key] = trial.suggest_float(
                            key, spec["low"], spec["high"], log=True
                        )
                    else:
                        params[key] = trial.suggest_float(
                            key, spec["low"], spec["high"], step=spec.get("step")
                        )
                elif spec["type"] == "int":
                    if use_log:
                        params[key] = trial.suggest_int(
                            key, spec["low"], spec["high"], log=True
                        )
                    else:
                        params[key] = trial.suggest_int(
                            key, spec["low"], spec["high"], step=spec.get("step")
                        )
    # else: VOLUME params are NOT suggested

    # Other common parameters (no conflicts)
    common_keys = [
        "ENTRY_PERIOD",
        "MA_PERIOD",
        "ATR_PERIOD",
        "ATR_MULTIPLIER",  # Trailing stop multiplier (always used)
        "ADX_THRESHOLD",
        "RISK_PER_TRADE",
        "LEVERAGE",
        "MAX_HOLDING_BARS",
        "TRAILING_ACTIVATION_ATR",
        "RISK_PER_TRADE_SPOT",  # Spot 전용
    ]

    for key in common_keys:
        if key in search_space:
            spec = search_space[key]
            use_log = spec.get("log", False)

            if spec["type"] == "float":
                if use_log:
                    params[key] = trial.suggest_float(
                        key, spec["low"], spec["high"], log=True
                    )
                else:
                    params[key] = trial.suggest_float(
                        key, spec["low"], spec["high"], step=spec.get("step")
                    )
            elif spec["type"] == "int":
                if use_log:
                    params[key] = trial.suggest_int(
                        key, spec["low"], spec["high"], log=True
                    )
                else:
                    params[key] = trial.suggest_int(
                        key, spec["low"], spec["high"], step=spec.get("step")
                    )

    return params


def calculate_score(ret, mdd, trades_df, mode="DAY"):
    """
    SQN Hybrid Objective v3 (Futures-Optimized)

    Futures 특화 조정사항:
    - Calmar 기대치 상향: 레버리지를 통한 높은 수익률 추구
    - MDD 허용치 확대: 레버리지 변동성을 감안한 현실적 기준
    - Profit Factor 강화: 펀딩비와 높은 수수료 극복을 위한 더 높은 PF 요구
    - SQN 기준 상향: 레버리지 리스크를 상쇄할 높은 일관성 필요
    """
    import numpy as np

    if trades_df.empty:
        return -10000

    N = len(trades_df)

    # 1. Individual Trade Returns (%)
    if "pnl_pct" not in trades_df.columns:
        raise ValueError("trades_df must contain 'pnl_pct' column for SQN calculation.")

    returns = trades_df["pnl_pct"].values

    r_avg = np.mean(returns)
    r_std = np.std(returns) if len(returns) > 1 else 100.0
    if r_std == 0:
        r_std = 0.001

    # --- Helper: Soft Sigmoid Normalization ---
    def soft_sigmoid(x, center, steepness):
        z = -steepness * (x - center)
        z = np.clip(z, -500, 500)  # Overflow prevention
        return 1 / (1 + np.exp(z))

    # --- Component 1: SQN (Consistency) ---
    # [FUTURES] Higher bar: 3.0 (vs Spot 2.5)
    # 레버리지 사용 시 더 높은 일관성 요구
    sqn = (np.sqrt(N) * r_avg) / r_std
    sqn_score = soft_sigmoid(sqn, center=3.0, steepness=0.5)

    # --- Component 2: Calmar Ratio (ROI Efficiency) ---
    # [FUTURES] Higher expectation: 5.0 (vs Spot 2.5)
    # 레버리지를 통해 더 높은 수익률 추구 (예: 5배 레버시 250% ROI with 50% MDD = Calmar 5.0)
    abs_mdd = abs(mdd) if mdd != 0 else 0.01
    calmar = ret / abs_mdd
    calmar_score = soft_sigmoid(calmar, center=5.0, steepness=0.4)

    # --- Component 3: Profit Factor ---
    # [FUTURES] Higher bar: 1.8 (vs Spot 1.3)
    # 펀딩비(0.01%) + 수수료(0.05% 양방향) = 매 거래당 0.11% 추가 비용 극복 필요
    pos_sum = np.sum(returns[returns > 0])
    neg_sum = abs(np.sum(returns[returns < 0]))
    pf = pos_sum / neg_sum if neg_sum > 0 else 3.0
    pf_score = soft_sigmoid(pf, center=1.8, steepness=1.0)

    # --- Component 4: Smooth MDD Penalty ---
    # [FUTURES] More tolerant: -30% center (vs Spot -20%)
    # 레버리지 3~5배 사용 시 -30% MDD는 실질적으로 감당 가능한 수준
    # Steepness 0.25 (vs Spot 0.25): 완만한 곡선으로 공격적 전략 허용
    mdd_penalty = soft_sigmoid(-abs_mdd, center=-30, steepness=0.25)

    # --- Component 5: Soft Trade Count Penalty ---
    MIN_TRADES_MAP = {"SCALP": 500, "DAY": 80, "SWING": 30, "ALL": 100}
    min_trades = MIN_TRADES_MAP.get(mode.upper(), 100)
    trade_penalty = soft_sigmoid(N, center=min_trades, steepness=0.1)

    # Hard floor for statistical significance
    if N < 10:
        return -10000

    # --- Final Score: Multiplicative ---
    # 모든 요소가 균형있게 우수해야 높은 점수 획득
    final_score = (
        sqn_score * calmar_score * pf_score * mdd_penalty * trade_penalty * 1000
    )

    return final_score


def objective(
    trial, strategy_cls, strategy_name, data_maps, search_space, common_search_space
):
    """
    Multi-symbol objective function.
    """
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
        engine = BacktestEngineFast(hourly_df, daily_df, strategy, initial_balance=750)

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
        score = calculate_score(ret, mdd, trades_df, mode=mode_str)

        # [SPEED UP] Short-circuit: If any symbol performs poorly, fail the trial immediately
        # One bad apple spoils the bunch (Harmonic Mean logic)
        if score < 5:  # Low score threshold for soft penalty
            return -10000

        symbol_scores.append(score)

        # Set individual attrs
        trial.set_user_attr(f"ret_{symbol.replace('/', '_')}", float(ret))
        trial.set_user_attr(f"mdd_{symbol.replace('/', '_')}", float(mdd))
        trial.set_user_attr(f"pf_{symbol.replace('/', '_')}", float(pf))

    # 4. Combine scores using HARMONIC MEAN
    offset = 6000
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
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument(
        "--mode",
        type=str,
        default="DAY",
        choices=["SCALP", "DAY", "SWING", "ALL"],
        help="Trading Mode: SCALP, DAY, SWING",
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
        "SCALP": 3300,  # 3 timeframes (5m,15m,30m), high data volume but narrow param range
        "DAY": 3800,  # 3 timeframes (1h,2h,4h), balanced - most commonly used mode
        "SWING": 4700,  # 3 timeframes (4h,1d,3d), wide param range + low data volume (overfitting risk)
        "ALL": 4000,  # Catch-all (highest complexity)
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
    search_space = GET_SEARCH_SPACE(mode, market_type="futures")
    timeframes = search_space["TIMEFRAME"]["choices"]

    print(f"\n{'='*70}")
    print(f"🚀 MODE: {mode} OPRIMIZATION")
    print(f"⏰ Target Timeframes: {timeframes}")
    # print(f"🔍 Search Space Size: {len(search_space)} parameters")
    print(f"{'='*70}\n")

    data_maps = {}
    print(f"📡 Loading data for symbols: {', '.join(symbols)}")

    for symbol in symbols:
        print(f"Loading {symbol}...")
        data_maps[symbol] = load_all_timeframes(
            symbol, BACKTEST_START_DATE, BACKTEST_END_DATE, timeframes
        )

    # [CRITICAL] Slice Data for Optimization (Train Set)
    print(f"✂️  Trimming Data for Optimization (Train Period: ~ {TRAIN_CUTOFF_DATE})")
    cutoff_ts = pd.Timestamp(TRAIN_CUTOFF_DATE)

    for sym in data_maps:
        for tf in data_maps[sym]:
            original_len = len(data_maps[sym][tf])
            data_maps[sym][tf] = data_maps[sym][tf][
                data_maps[sym][tf]["datetime"] < cutoff_ts
            ].copy()
            new_len = len(data_maps[sym][tf])
            # if tf == timeframes[0]:
            #     print(f"  [{sym}] Train Size: {new_len} (Original: {original_len})")

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
        n_startup_trials=100,  # Random exploration first
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
            _dummy_arr,  # Volume Ratio
            _dummy_arr,
            _dummy_arr,  # Entry Upper/Lower
            _dummy_int,
            _dummy_int,
            _dummy_arr,  # Trend, Strength, ATR
            _dummy_arr,  # Parabolic SAR (New)
            10000.0,
            1.0,
            0.001,
            0.001,  # Bal, Lev, Fee, Slip
            0,  # Exit Type (New: 0=Trailing, 1=SAR)
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
        )
        print(" Done!")
    except Exception as e:
        print(f" Skipped ({e})")

    try:
        study.optimize(
            lambda trial: objective(
                trial, UltimateStrategy, f"Ultimate_{mode}", data_maps, search_space, {}
            ),
            n_trials=trials,
            n_jobs=args.jobs,
            show_progress_bar=True,
        )

    except KeyboardInterrupt:
        print("\n🛑 Optimization Interrupted by User")

    print(f"\n{'='*70}")
    print(f"✅ {mode} Optimization Complete!")

    if len(study.trials) > 0:
        print(f"🏆 Best Score: {study.best_value:.2f}")
        print(f"✨ Best Params: {study.best_params}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
