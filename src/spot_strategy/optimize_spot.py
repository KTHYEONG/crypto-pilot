import argparse
import gc
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import quote_plus

try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    sys.path.append(os.getcwd())

from config.optimization_config_modes import GET_SEARCH_SPACE
from config.settings import BACKTEST_END_DATE, TRAIN_CUTOFF_DATE
from src.optimization.opt_utils import calculate_score, suggest_params
from src.spot_strategy.engine_fast_spot import BacktestEngineFastSpot, backtest_loop_spot_numba
from src.spot_strategy.upbit_client import UpbitClient
from src.strategy.strategies import UltimateStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpotOptimizer")

SPOT_START_DATE = "2018-01-01"
SPOT_INITIAL_BALANCE = 1_000_000.0
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
os.makedirs(DATA_DIR, exist_ok=True)

DAILY_BUFFER_DAYS = 200
WARMUP_BUFFER_BARS = {"5m": 500, "15m": 420, "30m": 350, "1h": 300, "2h": 250, "4h": 200, "1d": 150, "1w": 80}
AWFO_DEFAULTS = {
    "enabled_modes": {"UNIFIED", "ALL"},
    "folds": 3,
    "min_trades_per_fold": 35,
    "min_test_bars": {"5m": 1600, "15m": 1200, "30m": 900, "1h": 600, "2h": 420, "4h": 240, "1d": 120, "1w": 60},
    "embargo_bars": {"5m": 40, "15m": 32, "30m": 24, "1h": 24, "2h": 16, "4h": 12, "1d": 5, "1w": 2},
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


SPOT_ROBUST = {
    "w_avg": _env_float("SPOT_FOLD_W_AVG", 0.30),
    "w_p25": _env_float("SPOT_FOLD_W_P25", 0.45),
    "w_worst": _env_float("SPOT_FOLD_W_WORST", 0.25),
    "cons_target": _env_float("SPOT_FOLD_CONSISTENCY_TARGET", 0.55),
    "cons_penalty": _env_float("SPOT_FOLD_CONSISTENCY_PENALTY", 55.0),
    "cost_stress_per_trade": _env_float("SPOT_COST_STRESS_PER_TRADE_PCT", 0.015),
    "cost_stress_w": _env_float("SPOT_COST_STRESS_WEIGHT", 0.06),
    "ret_p25_w": _env_float("SPOT_FOLD_RET_P25_WEIGHT", 0.20),
    "ret_p25_clip": _env_float("SPOT_FOLD_RET_P25_CLIP", 60.0),
}


def load_all_timeframes(symbols: List[str], start_date: str, end_date: str, timeframes: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    if not access or not secret:
        print("[ERROR] Missing UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY in .env")
        sys.exit(1)
    client = UpbitClient(access, secret)
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
    symbols_data: Dict[str, Dict[str, pd.DataFrame]] = {s: {} for s in symbols}

    for symbol in symbols:
        for tf in timeframes:
            fp = os.path.join(
                DATA_DIR,
                f"{symbol.replace('/', '_')}_{tf}_{start_date.replace('-', '')}_{end_date.replace('-', '')}_spot.csv",
            )
            df: Optional[pd.DataFrame] = None
            if os.path.exists(fp):
                try:
                    df = pd.read_csv(fp)
                    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.sort_values("datetime", inplace=True)
                    df.reset_index(drop=True, inplace=True)
                except Exception:
                    df = None

            if df is None or df.empty:
                print(f"[INFO] Downloading {symbol}-{tf}...")
                df = client.fetch_ohlcv(symbol, tf, since=start_ts, end=end_ts)
                if df is None or df.empty:
                    print(f"[ERROR] Empty data: {symbol}-{tf}")
                    sys.exit(1)
                df.sort_values("timestamp", inplace=True)
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.reset_index(drop=True, inplace=True)
                try:
                    df.to_csv(fp, index=False)
                except Exception:
                    pass
            symbols_data[symbol][tf] = df
    return symbols_data


def compute_segment_merge_index(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> np.ndarray:
    hourly_days = pd.to_datetime(hourly_df["datetime"]).dt.normalize().values.astype("datetime64[ns]")
    daily_days = pd.to_datetime(daily_df["datetime"]).dt.normalize().values.astype("datetime64[ns]")
    if len(daily_days) == 0:
        return np.zeros(len(hourly_days), dtype=np.int32)
    pos = np.searchsorted(daily_days, hourly_days, side="right") - 1
    return np.clip(pos, 0, len(daily_days) - 1).astype(np.int32)


def compute_merge_indices(data_maps: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, np.ndarray]]:
    merge_indices: Dict[str, Dict[str, np.ndarray]] = {}
    for symbol, tf_map in data_maps.items():
        merge_indices[symbol] = {}
        daily_df = tf_map.get("1d")
        if daily_df is None or daily_df.empty:
            continue
        for tf, tf_df in tf_map.items():
            if tf == "1d":
                continue
            merge_indices[symbol][tf] = compute_segment_merge_index(tf_df, daily_df)
    return merge_indices


def build_anchored_splits(n_bars: int, n_folds: int, embargo_bars: int = 0, min_test_bars: int = 120) -> List[Tuple[int, int]]:
    if n_folds < 1 or n_bars < (n_folds + 1):
        return []
    block = n_bars // (n_folds + 1)
    if block < 2:
        return []
    splits: List[Tuple[int, int]] = []
    for i in range(1, n_folds + 1):
        test_start = (block * i) + max(embargo_bars, 0)
        test_end = (block * (i + 1)) if i < n_folds else n_bars
        if test_start < test_end and (test_end - test_start) >= min_test_bars:
            splits.append((int(test_start), int(test_end)))
    return splits


def build_awfo_plan(data_maps: Dict[str, Dict[str, pd.DataFrame]], timeframes: List[str], folds: int, min_trades: int) -> Dict:
    plan = {"enabled": True, "splits": {}, "min_trades_per_fold": int(min_trades)}
    for sym, tf_map in data_maps.items():
        plan["splits"][sym] = {}
        for tf in timeframes:
            if tf not in tf_map or tf_map[tf].empty:
                plan["splits"][sym][tf] = []
                continue
            plan["splits"][sym][tf] = build_anchored_splits(
                n_bars=len(tf_map[tf]),
                n_folds=int(folds),
                embargo_bars=AWFO_DEFAULTS["embargo_bars"].get(tf, 0),
                min_test_bars=AWFO_DEFAULTS["min_test_bars"].get(tf, 120),
            )
    return plan


def build_awfo_runtime_cache(data_maps: Dict[str, Dict[str, pd.DataFrame]], timeframes: List[str], awfo_plan: Dict) -> Dict[str, Dict[str, List[Dict]]]:
    if not awfo_plan or not awfo_plan.get("enabled", False):
        return {}
    cached: Dict[str, Dict[str, List[Dict]]] = {}
    splits_by_symbol = awfo_plan.get("splits", {})
    for symbol, tf_map in data_maps.items():
        cached[symbol] = {}
        daily_df = tf_map.get("1d")
        if daily_df is None or daily_df.empty:
            for tf in timeframes:
                cached[symbol][tf] = []
            continue
        for tf in timeframes:
            hourly_df = tf_map.get(tf)
            if hourly_df is None or hourly_df.empty:
                cached[symbol][tf] = []
                continue
            folds_ctx: List[Dict] = []
            for test_start, test_end in splits_by_symbol.get(symbol, {}).get(tf, []):
                seg_start = max(0, test_start - WARMUP_BUFFER_BARS.get(tf, 200))
                segment_hourly = hourly_df.iloc[seg_start:test_end].copy()
                if len(segment_hourly) < 100:
                    continue
                segment_hourly.attrs["warmup_bars"] = int(test_start - seg_start)
                actual_start_time = pd.Timestamp(hourly_df.iloc[test_start]["datetime"])
                actual_end_time = pd.Timestamp(hourly_df.iloc[test_end - 1]["datetime"])
                end_time = pd.Timestamp(segment_hourly["datetime"].iloc[-1])
                daily_start = pd.Timestamp(segment_hourly["datetime"].iloc[0]) - pd.Timedelta(days=DAILY_BUFFER_DAYS)
                segment_daily = daily_df[(daily_df["datetime"] >= daily_start) & (daily_df["datetime"] <= end_time)].copy()
                if segment_daily.empty:
                    continue
                folds_ctx.append(
                    {
                        "hourly": segment_hourly,
                        "daily": segment_daily,
                        "merge_index": compute_segment_merge_index(segment_hourly, segment_daily),
                        "actual_start_time": actual_start_time,
                        "actual_end_time": actual_end_time,
                    }
                )
            cached[symbol][tf] = folds_ctx
    return cached


def _filter_trades_for_window(trades_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        return df[(df["entry_time"] >= start) & (df["entry_time"] <= end)].copy()
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        return df[(df["exit_time"] >= start) & (df["exit_time"] <= end)].copy()
    return pd.DataFrame()


def calculate_oos_mdd_pct(pnl_series: pd.Series, initial_balance: float) -> float:
    if pnl_series is None or len(pnl_series) == 0:
        return 0.0
    equity = float(initial_balance) + pnl_series.cumsum().values
    run_max = np.maximum.accumulate(equity)
    run_max[run_max == 0] = 1e-9
    dd = (equity - run_max) / run_max * 100.0
    return float(np.min(dd)) if len(dd) else 0.0


def objective(
    trial: optuna.trial.Trial,
    symbols_data: Dict[str, Dict[str, pd.DataFrame]],
    search_space: Dict,
    mode: str = "DAY",
    merge_indices: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    awfo_plan: Optional[Dict] = None,
) -> float:
    params = suggest_params(trial, search_space)
    tf = params.get("TIMEFRAME", "1h")
    if params.get("TREND_FILTER_TYPE") == "MACD" and params.get("MACD_FAST", 12) >= params.get("MACD_SLOW", 26):
        return -10000.0

    awfo_enabled = bool(awfo_plan and awfo_plan.get("enabled"))
    awfo_cache = awfo_plan.get("cache", {}) if awfo_enabled else {}
    awfo_min_trades = awfo_plan.get("min_trades_per_fold", 35) if awfo_enabled else None
    symbol_scores: List[float] = []
    symbol_results: Dict[str, Dict[str, float]] = {}
    report_step = 0

    def fallback(sym: str, reason: str) -> None:
        print(f"[WARN] Spot fallback: {sym} ({reason})")
        symbol_scores.append(-180.0)
        symbol_results[sym] = {"return": -20.0, "mdd": -55.0, "pf": 0.0}

    for symbol, data_map in symbols_data.items():
        key = symbol.replace("/", "_").replace("-", "_")
        if tf not in data_map or "1d" not in data_map:
            fallback(symbol, "missing timeframe")
            trial.set_user_attr(f"ret_{key}", float(symbol_results[symbol]["return"]))
            trial.set_user_attr(f"mdd_{key}", float(symbol_results[symbol]["mdd"]))
            trial.set_user_attr(f"pf_{key}", float(symbol_results[symbol]["pf"]))
            continue

        if awfo_enabled:
            fold_ctxs = awfo_cache.get(symbol, {}).get(tf, [])
            if len(fold_ctxs) < 2:
                fallback(symbol, "insufficient awfo folds")
                trial.set_user_attr(f"ret_{key}", float(symbol_results[symbol]["return"]))
                trial.set_user_attr(f"mdd_{key}", float(symbol_results[symbol]["mdd"]))
                trial.set_user_attr(f"pf_{key}", float(symbol_results[symbol]["pf"]))
                continue

            fold_scores: List[float] = []
            fold_returns: List[float] = []
            fold_mdds: List[float] = []
            fold_pfs: List[float] = []
            fold_stress: List[float] = []
            invalid = 0
            for idx, ctx in enumerate(fold_ctxs):
                try:
                    strategy = UltimateStrategy(f"Opt_{symbol}_F{idx + 1}", params)
                    engine = BacktestEngineFastSpot(
                        ctx["hourly"],
                        ctx["daily"],
                        strategy,
                        backtest_loop_spot_numba,
                        initial_balance=SPOT_INITIAL_BALANCE,
                        fee_rate=0.0005,
                        slippage_rate=0.0003,
                        merge_index_map=ctx["merge_index"],
                    )
                    engine.risk_per_trade = params.get("RISK_PER_TRADE_SPOT", 0.99)
                    result = engine.run()
                except Exception:
                    invalid += 1
                    gc.collect()
                    continue

                oos_trades = _filter_trades_for_window(
                    result.get("trades_df", pd.DataFrame()),
                    pd.Timestamp(ctx["actual_start_time"]),
                    pd.Timestamp(ctx["actual_end_time"]),
                )
                if oos_trades.empty or "pnl" not in oos_trades.columns:
                    invalid += 1
                else:
                    fold_ret = float(oos_trades["pnl"].sum() / SPOT_INITIAL_BALANCE * 100.0)
                    fold_mdd = calculate_oos_mdd_pct(oos_trades["pnl"], SPOT_INITIAL_BALANCE)
                    fold_score = calculate_score(
                        fold_ret,
                        fold_mdd,
                        oos_trades,
                        mode=mode,
                        market_type="spot",
                        timeframe=tf,
                        min_trades_override=awfo_min_trades,
                    )
                    if np.isfinite(fold_score) and fold_score > -9000:
                        fold_scores.append(float(fold_score))
                        fold_returns.append(float(fold_ret))
                        fold_mdds.append(float(fold_mdd))
                        gross_profit = float(oos_trades[oos_trades["pnl"] > 0]["pnl"].sum())
                        gross_loss = abs(float(oos_trades[oos_trades["pnl"] < 0]["pnl"].sum()))
                        fold_pfs.append(gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0))
                        fold_stress.append(fold_ret - (len(oos_trades) * SPOT_ROBUST["cost_stress_per_trade"]))
                    else:
                        invalid += 1

                report_step += 1
                trial.report(float(np.percentile(fold_scores, 25)) if fold_scores else -220.0, report_step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            required = max(2, int(np.ceil(len(fold_ctxs) * 0.6)))
            if len(fold_scores) < required:
                fallback(symbol, f"valid_folds={len(fold_scores)} < required={required}")
                trial.set_user_attr(f"ret_{key}", float(symbol_results[symbol]["return"]))
                trial.set_user_attr(f"mdd_{key}", float(symbol_results[symbol]["mdd"]))
                trial.set_user_attr(f"pf_{key}", float(symbol_results[symbol]["pf"]))
                continue

            avg = float(np.mean(fold_scores))
            p25 = float(np.percentile(fold_scores, 25))
            worst = float(np.min(fold_scores))
            p25_ret = float(np.percentile(fold_returns, 25)) if fold_returns else -100.0
            p25_stress = float(np.percentile(fold_stress, 25)) if fold_stress else p25_ret
            consistency = float(np.mean(np.array(fold_scores) > 0))
            score = (SPOT_ROBUST["w_avg"] * avg) + (SPOT_ROBUST["w_p25"] * p25) + (SPOT_ROBUST["w_worst"] * worst)
            score += SPOT_ROBUST["ret_p25_w"] * np.clip(p25_ret, -SPOT_ROBUST["ret_p25_clip"], SPOT_ROBUST["ret_p25_clip"])
            score += SPOT_ROBUST["cost_stress_w"] * np.clip(p25_stress, -60.0, 60.0)
            if consistency < SPOT_ROBUST["cons_target"]:
                score -= (SPOT_ROBUST["cons_target"] - consistency) * SPOT_ROBUST["cons_penalty"]
            score -= invalid * 12.0

            symbol_scores.append(float(score))
            symbol_results[symbol] = {
                "return": float(np.mean(fold_returns)),
                "mdd": float(np.mean(fold_mdds)),
                "pf": float(np.mean(fold_pfs)) if fold_pfs else 0.0,
            }
        else:
            try:
                strategy = UltimateStrategy(f"Opt_{symbol}", params)
                current_merge = merge_indices.get(symbol, {}).get(tf) if merge_indices else None
                engine = BacktestEngineFastSpot(
                    data_map[tf],
                    data_map["1d"],
                    strategy,
                    backtest_loop_spot_numba,
                    initial_balance=SPOT_INITIAL_BALANCE,
                    fee_rate=0.0005,
                    slippage_rate=0.0003,
                    merge_index_map=current_merge,
                )
                engine.risk_per_trade = params.get("RISK_PER_TRADE_SPOT", 0.99)
                result = engine.run()
            except Exception:
                fallback(symbol, "single run failed")
                trial.set_user_attr(f"ret_{key}", float(symbol_results[symbol]["return"]))
                trial.set_user_attr(f"mdd_{key}", float(symbol_results[symbol]["mdd"]))
                trial.set_user_attr(f"pf_{key}", float(symbol_results[symbol]["pf"]))
                continue

            trades_df = result.get("trades_df", pd.DataFrame())
            ret = float(result.get("total_return_pct", 0.0))
            mdd = float(result.get("mdd_pct", 0.0))
            pf = 0.0
            if not trades_df.empty and "pnl" in trades_df.columns:
                gp = float(trades_df[trades_df["pnl"] > 0]["pnl"].sum())
                gl = abs(float(trades_df[trades_df["pnl"] < 0]["pnl"].sum()))
                pf = gp / gl if gl > 0 else (gp if gp > 0 else 0.0)
            score = calculate_score(ret, mdd, trades_df, mode=mode, market_type="spot", timeframe=tf)
            symbol_scores.append(float(score if np.isfinite(score) and score > -9000 else -220.0))
            symbol_results[symbol] = {"return": ret, "mdd": mdd, "pf": pf}

        trial.set_user_attr(f"ret_{key}", float(symbol_results[symbol]["return"]))
        trial.set_user_attr(f"mdd_{key}", float(symbol_results[symbol]["mdd"]))
        trial.set_user_attr(f"pf_{key}", float(symbol_results[symbol]["pf"]))

    if not symbol_scores:
        return -10000.0
    mdd_abs = [abs(float(v["mdd"])) for v in symbol_results.values()]
    if mdd_abs and (max(mdd_abs) > 70.0 or float(np.mean(mdd_abs)) > 55.0):
        return -10000.0
    shifted = np.array(symbol_scores, dtype=np.float64) + 220.0
    if np.any(shifted <= 1e-9):
        return -10000.0
    hm = float(len(shifted) / np.sum(1.0 / shifted)) - 220.0
    p25 = float(np.percentile(symbol_scores, 25))
    ret_values = [float(v["return"]) for v in symbol_results.values()]
    final_score = (0.65 * hm) + (0.35 * p25)
    final_score += 0.05 * float(np.percentile(ret_values, 25))
    final_score -= 0.04 * float(np.std(ret_values))
    trial.set_user_attr("score_avg", float(final_score))
    trial.set_user_attr("score_p25", float(p25))
    return float(final_score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--symbols", type=str, default="KRW-BTC,KRW-ETH")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--mode", type=str, default="UNIFIED", choices=["SCALP", "DAY", "SWING", "UNIFIED", "ALL"])
    args = parser.parse_args()

    mode = args.mode.upper()
    symbols = [s.strip() for s in args.symbols.split(",")]
    awfo_enabled = mode in AWFO_DEFAULTS["enabled_modes"]
    trials = args.trials if args.trials is not None else {"SCALP": 3600, "DAY": 4200, "SWING": 5000, "UNIFIED": 8000, "ALL": 8000}.get(mode, 2500)
    print(f"[INFO] mode={mode}, trials={trials}, awfo={'ON' if awfo_enabled else 'OFF'}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    search_space = GET_SEARCH_SPACE(mode, market_type="spot")
    timeframes = search_space["TIMEFRAME"]["choices"]
    load_tfs = sorted(set(timeframes + ["1d"]))
    symbols_data = load_all_timeframes(symbols, SPOT_START_DATE, BACKTEST_END_DATE, load_tfs)

    cutoff_ts = pd.Timestamp(TRAIN_CUTOFF_DATE)
    train_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sym, tf_map in symbols_data.items():
        train_data[sym] = {}
        for tf, df in tf_map.items():
            end_idx = int((df["datetime"] < cutoff_ts).sum())
            if end_idx <= 0:
                continue
            sliced = df.iloc[:end_idx].copy()
            sliced.attrs["warmup_bars"] = min(WARMUP_BUFFER_BARS.get(tf, 200), len(sliced))
            train_data[sym][tf] = sliced

    merge_indices = compute_merge_indices(train_data)
    awfo_plan = {"enabled": False, "splits": {}, "min_trades_per_fold": None, "cache": {}}
    if awfo_enabled:
        awfo_plan = build_awfo_plan(train_data, timeframes, AWFO_DEFAULTS["folds"], AWFO_DEFAULTS["min_trades_per_fold"])
        awfo_plan["cache"] = build_awfo_runtime_cache(train_data, timeframes, awfo_plan)

    load_dotenv()
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "trading_optuna")
    if not all([db_user, db_pass, db_name]):
        print("[ERROR] Missing DB credentials in .env")
        sys.exit(1)

    study_name = f"spot_{mode.lower()}_strategy"
    storage_url = f"mysql+pymysql://{db_user}:{quote_plus(db_pass)}@{db_host}:{db_port}/{db_name}"
    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
    except Exception:
        pass

    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={"pool_size": max(30, args.jobs * 2), "max_overflow": 10, "pool_recycle": 3600, "pool_pre_ping": True},
    )
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=max(200, int(trials * 0.18)),
        multivariate=True,
        constant_liar=True,
        warn_independent_sampling=False,
    )
    study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize", sampler=sampler)

    try:
        backtest_loop_spot_numba(
            np.ones(10, dtype=np.float64), np.ones(10, dtype=np.float64), np.ones(10, dtype=np.float64), np.ones(10, dtype=np.float64),
            np.ones(10, dtype=np.float64), np.zeros(10, dtype=np.int64), np.zeros(10, dtype=np.int64), np.ones(10, dtype=np.float64),
            np.ones(10, dtype=np.float64), np.ones(10, dtype=np.float64), np.ones(10, dtype=np.float64), np.ones(10, dtype=np.float64),
            np.ones(10, dtype=np.float64), 10_000.0, 0.001, 0.001, 0, 0, 0.01, 1.5, 3.0, 0.99, False, 1.0, False, 3.0, 1000, 0.0,
            0.6, 4.5, 94.0, 1.3, 0.15, 0.45, 0.6, 90.0, 0.1, 0, False, 1_000_000.0
        )
    except Exception:
        pass

    study.optimize(
        lambda t: objective(t, train_data, search_space, mode, merge_indices, awfo_plan),
        n_trials=trials,
        n_jobs=args.jobs,
        show_progress_bar=True,
    )

    print(f"[DONE] best_score={study.best_value:.2f}")
    print(f"[DONE] best_params={study.best_params}")


if __name__ == "__main__":
    main()
