
import argparse
import pandas as pd
import sys
import os
import logging
import json
from pathlib import Path
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# Project Root Setup
project_root = os.getcwd()
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    if project_root not in sys.path:
        sys.path.append(project_root)

from config.settings import (
    FUTURES_BACKTEST_START_DATE,
    FUTURES_BACKTEST_END_DATE,
    FUTURES_TRAIN_CUTOFF_DATE,
    FUTURES_INITIAL_BALANCE,
    DATA_DIR,
)
from src.common.utils import setup_logger
from src.futures_strategy.data_collector import DataCollector
from src.futures_strategy.strategies_futures import UltimateStrategy
from src.futures_strategy.funding_utils import merge_funding_into_ohlcv

# Setup Logging
logger = setup_logger("FuturesVerifier", write_file=False)
# Reduce engine logs to keep verify output readable.
logging.getLogger("src.futures_strategy.engine_fast_futures").setLevel(logging.WARNING)

LOG_WIDTH = 80
FUTURES_VERIFY_LOG_PATH = Path(project_root) / "logs" / "futures_verify_comparison.jsonl"
FUTURES_COMPARISON_METRICS: Tuple[str, ...] = (
    "core_avg_ret",
    "core_avg_mdd_abs",
    "core_avg_pf",
    "core_min_trades",
    "core_total_trades",
    "core_wfa_consistency",
    "core_mc_worst_mdd_95",
    "alt_median_ret",
    "alt_p25_ret",
    "alt_pos_rate",
    "alt_worst_mdd_abs",
    "dispersion",
    "avg_ret",
    "train_score",
    "selection_rank_score",
)
FUTURES_LOWER_IS_BETTER_METRICS = {
    "core_avg_mdd_abs",
    "core_mc_worst_mdd_95",
    "alt_worst_mdd_abs",
    "dispersion",
}


def _log_header(title: str, subtitle: Optional[str] = None) -> None:
    print("\n" + "=" * LOG_WIDTH)
    print(title)
    if subtitle:
        print(subtitle)
    print("=" * LOG_WIDTH)


def _log_subheader(title: str) -> None:
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def _safe_float(value: Any) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(converted):
        return None
    return float(converted)


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _extract_numeric_metrics(raw_metrics: Any) -> Dict[str, float]:
    if not isinstance(raw_metrics, dict):
        return {}
    metrics: Dict[str, float] = {}
    for key, raw_value in raw_metrics.items():
        converted = _safe_float(raw_value)
        if converted is not None:
            metrics[str(key)] = converted
    return metrics


def _extract_summary_metrics(
    summary: Optional[Dict[str, Any]],
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    summary_dict = summary if isinstance(summary, dict) else {}
    for key in FUTURES_COMPARISON_METRICS:
        converted = _safe_float(summary_dict.get(key))
        if converted is not None:
            metrics[key] = converted
    if extra_metrics:
        metrics.update(_extract_numeric_metrics(extra_metrics))
    return metrics


def _read_latest_comparison_entry(log_path: Path) -> Optional[Dict[str, Any]]:
    if not log_path.exists():
        return None
    try:
        latest_line: Optional[str] = None
        with log_path.open("r", encoding="utf-8") as file_handle:
            for raw_line in file_handle:
                line = raw_line.strip()
                if line:
                    latest_line = line
        if not latest_line:
            return None
        loaded = json.loads(latest_line)
        return loaded if isinstance(loaded, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"[WARN] Failed to read previous futures comparison log: {exc}")
        return None


def _build_metric_comparison(
    previous_metrics: Dict[str, float],
    current_metrics: Dict[str, float],
) -> Dict[str, Any]:
    metric_deltas: Dict[str, Dict[str, float]] = {}
    improved_metrics: List[str] = []
    degraded_metrics: List[str] = []

    for metric in sorted(set(previous_metrics) | set(current_metrics)):
        prev_value = previous_metrics.get(metric)
        curr_value = current_metrics.get(metric)
        if prev_value is None or curr_value is None:
            continue
        delta = float(curr_value - prev_value)
        metric_deltas[metric] = {
            "previous": float(prev_value),
            "current": float(curr_value),
            "delta": delta,
        }
        if np.isclose(delta, 0.0):
            continue
        lower_is_better = metric in FUTURES_LOWER_IS_BETTER_METRICS
        is_improved = (delta < 0.0) if lower_is_better else (delta > 0.0)
        if is_improved:
            improved_metrics.append(metric)
        else:
            degraded_metrics.append(metric)

    return {
        "metric_deltas": metric_deltas,
        "improved_metrics": improved_metrics,
        "degraded_metrics": degraded_metrics,
    }


def _record_verification_comparison(
    *,
    log_path: Path,
    run_type: str,
    current_best: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> None:
    previous_entry = _read_latest_comparison_entry(log_path)
    previous_best = previous_entry.get("current_best") if isinstance(previous_entry, dict) else None
    previous_metrics = _extract_numeric_metrics(
        previous_best.get("metrics") if isinstance(previous_best, dict) else None
    )
    current_metrics = _extract_numeric_metrics(current_best.get("metrics"))
    comparison = _build_metric_comparison(previous_metrics, current_metrics)

    entry = {
        "logged_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "strategy_type": "futures",
        "run_type": str(run_type),
        "context": context or {},
        "current_best": current_best,
        "previous_latest": previous_best,
        "comparison": comparison,
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(_sanitize_for_json(entry), ensure_ascii=False) + "\n")
        print(f"[INFO] Comparison log appended: {log_path}")
    except OSError as exc:
        logger.warning(f"[WARN] Failed to write futures comparison log: {exc}")


# Selection policy for long-term stable operation + high return.
SELECTION_POLICY_VERSION = "SELECTION_POLICY_V3"
# Timeframe-specific min trades for selection gate (lower TF = more bars → higher min; 4h needs fewer).
FUTURES_TF_SELECTION_MIN_TRADES: Dict[str, int] = {
    "1h": 30,
    "4h": 15,
}
FUTURES_SELECTION_MIN_TRADES_DEFAULT = 30

SELECTION_POLICY = {
    "gates": {
        "core_min_return": 0.0,
        "core_min_trades": 30,
        "core_max_avg_mdd_abs": 25.0,
        # With n_splits=5, 40% explicitly allows 2/5 positive WFA segments (reliable threshold).
        "core_min_wfa_consistency": 40.0,
        "core_max_mc_worst_mdd_95_abs": 45.0,
        "alt_min_pos_rate": 0.50,
        "alt_max_worst_mdd_abs": 50.0,
        "alt_min_p25_return": -20.0,
    },
    # Core-first: 70% core, 25% alt, 5% div (align with spot; favor return over uniformity).
    "weights": {
        "core_total": 0.70,
        "alt_total": 0.25,
        "div_total": 0.05,
        "core_return": 0.35,
        "core_pf": 0.15,
        "core_wfa": 0.20,
        "core_mdd": 0.15,
        "core_mc": 0.15,
        "alt_median": 0.35,
        "alt_p25": 0.25,
        "alt_pos": 0.25,
        "alt_mdd": 0.15,
    },
}


def futures_selection_min_trades_for_timeframe(timeframe: Optional[str]) -> int:
    """Return selection gate min trades for the given timeframe; default if unknown."""
    if not timeframe:
        return FUTURES_SELECTION_MIN_TRADES_DEFAULT
    tf = str(timeframe).strip().lower()
    return int(FUTURES_TF_SELECTION_MIN_TRADES.get(tf, FUTURES_SELECTION_MIN_TRADES_DEFAULT))


# Bonus sweep anti-overfitting defaults:
# Rank on selection OOS window, then run a separate holdout sanity check.
BONUS_SWEEP_HOLDOUT_RATIO = 0.30
BONUS_SWEEP_MIN_HOLDOUT_DAYS = 120
COST_STRESS_MULTIPLIERS = (1.5, 2.0)

HOLDOUT_SANITY_GATES = {
    "core_min_return": 0.0,
    "core_min_trades": 30,
    "core_max_avg_mdd_abs": 35.0,
}


def _clip01(x):
    return float(np.clip(float(x), 0.0, 1.0))


def _log_scaled_positive(x, cap):
    if x <= 0:
        return 0.0
    return _clip01(np.log1p(float(x)) / np.log1p(float(cap)))


def _calculate_mdd_pct_from_pnl(pnl_series, initial_balance):
    if pnl_series is None or len(pnl_series) == 0:
        return 0.0
    equity = float(initial_balance) + pd.Series(pnl_series).cumsum().values
    running_max = np.maximum.accumulate(equity)
    running_max[running_max == 0] = 1e-9
    drawdown = (equity - running_max) / running_max * 100.0
    return float(np.min(drawdown)) if len(drawdown) else 0.0


def _to_inclusive_end_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    """Convert date-like timestamp to inclusive end-of-day timestamp."""
    if value == value.normalize():
        return value + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    return value


def resolve_eval_window(
    eval_start_time: Optional[pd.Timestamp] = None,
    eval_end_time: Optional[pd.Timestamp] = None,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    eval_start = pd.Timestamp(eval_start_time) if eval_start_time is not None else pd.Timestamp(FUTURES_TRAIN_CUTOFF_DATE)
    raw_end = pd.Timestamp(eval_end_time) if eval_end_time is not None else pd.Timestamp(FUTURES_BACKTEST_END_DATE)
    eval_end = _to_inclusive_end_timestamp(raw_end)
    if eval_end < eval_start:
        raise ValueError(f"Invalid eval window: start={eval_start}, end={eval_end}")
    return eval_start, eval_end


def split_bonus_oos_windows() -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Split OOS into selection + holdout windows to reduce meta-overfitting in bonus sweep.
    """
    oos_start, oos_end = resolve_eval_window(pd.Timestamp(FUTURES_TRAIN_CUTOFF_DATE), pd.Timestamp(FUTURES_BACKTEST_END_DATE))
    total_days = max(1, int((oos_end.normalize() - oos_start.normalize()).days) + 1)
    holdout_days = max(BONUS_SWEEP_MIN_HOLDOUT_DAYS, int(total_days * BONUS_SWEEP_HOLDOUT_RATIO))
    if holdout_days >= total_days:
        holdout_days = max(30, total_days // 3)
    holdout_start = oos_end.normalize() - pd.Timedelta(days=holdout_days - 1)
    selection_end = holdout_start - pd.Timedelta(milliseconds=1)
    if selection_end <= oos_start:
        selection_end = oos_start + pd.Timedelta(days=max(30, total_days // 2))
        holdout_start = selection_end + pd.Timedelta(milliseconds=1)
    return oos_start, selection_end, holdout_start, oos_end


def compute_dynamic_holdout_min_trades(holdout_start: pd.Timestamp, holdout_end: pd.Timestamp) -> int:
    holdout_days = max(1, int((holdout_end.normalize() - holdout_start.normalize()).days) + 1)
    dynamic_floor = int(round(holdout_days * 0.18))
    return max(20, dynamic_floor)


def score_holdout_candidate(summary: Optional[Dict]) -> float:
    """
    Score holdout-passed candidates by OOS return quality + robustness.
    Used to choose winner among passers (avoids order-dependent selection).
    """
    if not summary:
        return -1.0
    core_ret = float(summary.get("core_avg_ret", 0.0))
    core_pf = float(summary.get("core_avg_pf", 0.0))
    core_mdd_abs = float(summary.get("core_avg_mdd_abs", 100.0))
    core_trades = float(summary.get("core_min_trades", summary.get("core_avg_trades", 0)))

    ret_score = _clip01((core_ret + 2.0) / 12.0)
    pf_score = _clip01((core_pf - 1.0) / 0.8)
    mdd_score = 1.0 - _clip01(core_mdd_abs / 15.0)
    activity_score = _clip01(core_trades / 12.0)

    return float(
        0.45 * pf_score
        + 0.30 * ret_score
        + 0.15 * mdd_score
        + 0.10 * activity_score
    )


def evaluate_holdout_sanity(summary, min_trades_gate: Optional[int] = None):
    if not summary:
        return False, ["holdout_no_summary"]
    trades_gate = int(
        min_trades_gate
        if min_trades_gate is not None
        else HOLDOUT_SANITY_GATES["core_min_trades"]
    )
    reasons = []
    if float(summary.get("core_avg_ret", 0.0)) <= HOLDOUT_SANITY_GATES["core_min_return"]:
        reasons.append("holdout_core_return_low")
    if int(summary.get("core_min_trades", 0)) < trades_gate:
        reasons.append("holdout_core_trades_low")
    if float(summary.get("core_avg_mdd_abs", 0.0)) > HOLDOUT_SANITY_GATES["core_max_avg_mdd_abs"]:
        reasons.append("holdout_core_mdd_too_high")
    return len(reasons) == 0, reasons


def run_cost_stress_verification(
    best_params: dict,
    symbols: list,
    primary_symbols: list,
    eval_start_time: pd.Timestamp,
    eval_end_time: pd.Timestamp,
    multipliers: Tuple[float, ...] = COST_STRESS_MULTIPLIERS,
):
    """Run additional durability checks with stressed fee/slippage assumptions."""
    stress_summaries = {}
    for mult in multipliers:
        print("\n" + "-" * LOG_WIDTH)
        print(f"Cost Stress Verification x{mult:.1f} (fee/slippage)")
        print(f"Window: {eval_start_time} ~ {eval_end_time}")
        print("-" * LOG_WIDTH)
        stress_results = []
        for symbol in symbols:
            result = verify_single_symbol_futures(
                symbol,
                best_params,
                primary_symbols,
                eval_start_time=eval_start_time,
                eval_end_time=eval_end_time,
                run_robustness_checks=False,
                cost_mult=float(mult),
            )
            if result:
                stress_results.append(result)

        calculate_mode_performance(stress_results)
        stress_summary = summarize_profile(stress_results, timeframe=best_params.get("TIMEFRAME"))
        stress_summaries[float(mult)] = stress_summary
        if stress_summary:
            print(
                f"Cost Stress x{mult:.1f} Summary: "
                f"core_ret={stress_summary['core_avg_ret']:.2f}% | "
                f"core_mdd={stress_summary['core_avg_mdd_abs']:.2f}% | "
                f"core_pf={stress_summary['core_avg_pf']:.2f} | "
                f"core_min_trades={stress_summary['core_min_trades']}"
            )
        else:
            print(f"[WARN] Cost Stress x{mult:.1f} summary unavailable.")
    return stress_summaries


def build_rolling_oos_windows(
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    window_days: int = 120,
    step_days: int = 30,
    max_windows: int = 6,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Build forward rolling OOS windows within [eval_start, eval_end]."""
    start = pd.Timestamp(eval_start).normalize()
    end = pd.Timestamp(eval_end)
    window_days = int(max(30, window_days))
    step_days = int(max(7, step_days))
    max_windows = int(max(1, max_windows))

    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor <= end and len(windows) < max_windows:
        w_start = cursor
        w_end = _to_inclusive_end_timestamp(cursor + pd.Timedelta(days=window_days - 1))
        if w_start > end:
            break
        if w_end > end:
            w_end = end
        if w_end >= w_start:
            windows.append((w_start, w_end))
        cursor = cursor + pd.Timedelta(days=step_days)
        if w_end >= end:
            break
    return windows


def run_rolling_oos_verification(
    best_params: dict,
    symbols: list,
    primary_symbols: list,
    eval_start_time: pd.Timestamp,
    eval_end_time: pd.Timestamp,
    window_days: int = 120,
    step_days: int = 30,
    max_windows: int = 6,
):
    """Run rolling OOS checks for additional temporal robustness diagnostics."""
    windows = build_rolling_oos_windows(
        eval_start=eval_start_time,
        eval_end=eval_end_time,
        window_days=window_days,
        step_days=step_days,
        max_windows=max_windows,
    )
    if not windows:
        print("[WARN] Rolling OOS skipped: no valid windows.")
        return []

    roll_summaries: List[Dict] = []
    print("\n" + "-" * LOG_WIDTH)
    print(
        "Rolling OOS Verification "
        f"(window_days={window_days}, step_days={step_days}, max_windows={max_windows})"
    )
    print("-" * LOG_WIDTH)

    for idx, (w_start, w_end) in enumerate(windows, start=1):
        print(f"[ROLL-{idx}] Window: {w_start} ~ {w_end}")
        roll_results = []
        for symbol in symbols:
            result = verify_single_symbol_futures(
                symbol,
                best_params,
                primary_symbols,
                eval_start_time=w_start,
                eval_end_time=w_end,
                run_robustness_checks=False,
            )
            if result:
                roll_results.append(result)

        calculate_mode_performance(roll_results)
        roll_summary = summarize_profile(roll_results, timeframe=best_params.get("TIMEFRAME"))
        if not roll_summary:
            print(f"[ROLL-{idx}] Summary unavailable.")
            continue
        print(
            f"[ROLL-{idx}] Summary: core_ret={roll_summary['core_avg_ret']:.2f}% | "
            f"core_mdd={roll_summary['core_avg_mdd_abs']:.2f}% | "
            f"core_pf={roll_summary['core_avg_pf']:.2f} | "
            f"core_min_trades={roll_summary['core_min_trades']}"
        )
        roll_summary["window_start"] = str(w_start)
        roll_summary["window_end"] = str(w_end)
        roll_summaries.append(roll_summary)

    if roll_summaries:
        core_ret = np.array([float(s["core_avg_ret"]) for s in roll_summaries], dtype=np.float64)
        core_mdd = np.array([float(s["core_avg_mdd_abs"]) for s in roll_summaries], dtype=np.float64)
        core_pf = np.array([float(s["core_avg_pf"]) for s in roll_summaries], dtype=np.float64)
        pass_rate = float(np.mean(core_ret > 0.0) * 100.0)
        print(
            "[ROLL] Aggregate: "
            f"windows={len(roll_summaries)} | pass_rate(core_ret>0)={pass_rate:.1f}% | "
            f"ret_median={float(np.median(core_ret)):.2f}% | "
            f"ret_p25={float(np.percentile(core_ret, 25)):.2f}% | "
            f"mdd_worst={float(np.max(core_mdd)):.2f}% | "
            f"pf_median={float(np.median(core_pf)):.2f}"
        )
    else:
        print("[WARN] Rolling OOS produced no usable summaries.")
    return roll_summaries

def load_data(symbol, start_date, end_date, timeframe):
    """Load Data Helper"""
    collector = DataCollector()

    # Daily Data (Parquet range cache + incremental fetch)
    daily_df = collector.ensure_data(symbol, "1d", start_date, end_date)
    daily_df['datetime'] = pd.to_datetime(daily_df['timestamp'], unit='ms')

    # Timeframe Data (Parquet range cache + incremental fetch)
    hourly_df = collector.ensure_data(symbol, timeframe, start_date, end_date)
    hourly_df['datetime'] = pd.to_datetime(hourly_df['timestamp'], unit='ms')
    hourly_df = merge_funding_into_ohlcv(symbol, hourly_df, DATA_DIR)

    # Engine performs signal generation/merge internally.
    return hourly_df, daily_df


def _slice_eval_with_warmup(hourly_df, daily_df, eval_start_ts, eval_end_ts, warmup_days=60):
    warmup_start = pd.Timestamp(eval_start_ts) - pd.Timedelta(days=warmup_days)
    eval_end_ts = pd.Timestamp(eval_end_ts)
    test_hourly = hourly_df[
        (hourly_df['datetime'] >= warmup_start) & (hourly_df['datetime'] <= eval_end_ts)
    ].copy()
    test_daily = daily_df[
        (daily_df['datetime'] >= warmup_start) & (daily_df['datetime'] <= eval_end_ts)
    ].copy()
    return test_hourly, test_daily


def _filter_trades_for_window(trades_df, eval_start_ts, eval_end_ts):
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    trades_df = trades_df.copy()
    eval_start_ts = pd.Timestamp(eval_start_ts)
    eval_end_ts = pd.Timestamp(eval_end_ts)

    if 'entry_time' in trades_df.columns:
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        return trades_df[
            (trades_df['entry_time'] >= eval_start_ts)
            & (trades_df['entry_time'] <= eval_end_ts)
        ].copy()

    if 'exit_time' in trades_df.columns:
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        return trades_df[
            (trades_df['exit_time'] >= eval_start_ts)
            & (trades_df['exit_time'] <= eval_end_ts)
        ].copy()

    return pd.DataFrame()


def _diagnose_zero_trade_window(
    merged_df: pd.DataFrame,
    params: dict,
    eval_start_ts: pd.Timestamp,
    eval_end_ts: pd.Timestamp,
) -> None:
    """
    Print step-by-step signal attrition diagnostics for zero-trade windows.
    Uses the same core entry conditions as engine signal detection.
    """
    if merged_df is None or merged_df.empty:
        print("   [DIAG] merged_df is empty.")
        return

    df = merged_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    eval_mask = (df["datetime"] >= pd.Timestamp(eval_start_ts)) & (df["datetime"] <= pd.Timestamp(eval_end_ts))
    eval_df = df.loc[eval_mask].copy()

    if eval_df.empty:
        print("   [DIAG] No bars in eval window (data coverage issue).")
        print(f"   [DIAG] merged range: {df['datetime'].min()} ~ {df['datetime'].max()}")
        return

    # Required engine-side inputs
    close = eval_df["close"].astype(float)
    entry_upper = eval_df.get("daily_entry_upper", pd.Series(np.nan, index=eval_df.index)).astype(float)
    entry_lower = eval_df.get("daily_entry_lower", pd.Series(np.nan, index=eval_df.index)).astype(float)
    trend_dir = eval_df.get("daily_trend_direction", pd.Series(0, index=eval_df.index)).fillna(0).astype(int)
    strength = eval_df.get("daily_strength_filter", pd.Series(0, index=eval_df.index)).fillna(0).astype(int)
    volume_ratio = eval_df.get("daily_volume_ratio", pd.Series(np.nan, index=eval_df.index)).astype(float)

    use_volume_filter = bool(params.get("USE_VOLUME_FILTER", False))
    vol_threshold = float(params.get("VOLUME_THRESHOLD_MULT", 1.0))

    total = len(eval_df)
    has_band = ~(entry_upper.isna() | entry_lower.isna())
    pass_strength = strength != 0
    pass_volume = (volume_ratio >= vol_threshold) if use_volume_filter else pd.Series(True, index=eval_df.index)

    valid = has_band & pass_strength & pass_volume
    trend_long = trend_dir == 1
    trend_short = trend_dir == -1
    breakout_long = close > entry_upper
    breakout_short = close < entry_lower

    long_signal = valid & trend_long & breakout_long
    short_signal = valid & trend_short & breakout_short

    print("   [DIAG] Zero-trade diagnostics")
    print(f"   [DIAG] Bars in eval window: {total} ({eval_df['datetime'].min()} ~ {eval_df['datetime'].max()})")
    print(f"   [DIAG] After band-NaN check: {int(has_band.sum())}/{total}")
    print(f"   [DIAG] After strength filter: {int((has_band & pass_strength).sum())}/{total}")
    if use_volume_filter:
        print(f"   [DIAG] After volume filter: {int((has_band & pass_strength & pass_volume).sum())}/{total} (threshold={vol_threshold:.4f})")
    else:
        print("   [DIAG] Volume filter disabled")

    print(
        f"   [DIAG] Trend distribution: long={int(trend_long.sum())}, "
        f"short={int(trend_short.sum())}, neutral={int((~trend_long & ~trend_short).sum())}"
    )
    print(
        f"   [DIAG] Breakout counts (pre-trend): long={int((valid & breakout_long).sum())}, "
        f"short={int((valid & breakout_short).sum())}"
    )
    print(
        f"   [DIAG] Final signal candidates: long={int(long_signal.sum())}, "
        f"short={int(short_signal.sum())}, total={int((long_signal | short_signal).sum())}"
    )


def detailed_backtest_futures(hourly_df, daily_df, params):
    """
    Detailed Backtest for Futures (Long/Short)
    Engine path only (same execution model as optimize/verify).
    Uses cutoff-anchored OOS evaluation and WFA.
    """
    logger.info(f"--- Starting Detailed Backtest (Futures) ---")

    from src.futures_strategy.engine_fast_futures import BacktestEngineFast
    from src.futures_strategy.walk_forward_futures import FuturesWalkForwardAnalyzer
    from src.futures_strategy.monte_carlo_futures import FuturesMonteCarloSimulator

    eval_start_ts, eval_end_ts = resolve_eval_window()
    test_hourly, test_daily = _slice_eval_with_warmup(hourly_df, daily_df, eval_start_ts, eval_end_ts)

    if test_hourly.empty:
        print("No data available for detailed backtest window.")
        return

    strategy = UltimateStrategy("Verify_Detailed", params)
    engine = BacktestEngineFast(test_hourly, test_daily, strategy, initial_balance=FUTURES_INITIAL_BALANCE)
    engine.leverage = params.get('LEVERAGE', 1)
    engine.risk_per_trade = params.get('RISK_PER_TRADE', 0.02)
    engine.funding_events_per_bar = 3 if params.get('TIMEFRAME') == '1d' else 1
    res = engine.run()

    oos_trades = _filter_trades_for_window(res.get('trades_df', pd.DataFrame()), eval_start_ts, eval_end_ts)
    trades_log = oos_trades['pnl'].tolist() if not oos_trades.empty else []
    total_pnl = float(oos_trades['pnl'].sum()) if trades_log else 0.0
    ret_pct = (total_pnl / FUTURES_INITIAL_BALANCE) * 100.0 if trades_log else 0.0
    mdd = _calculate_mdd_pct_from_pnl(oos_trades['pnl'], FUTURES_INITIAL_BALANCE) if trades_log else 0.0
    final_val = FUTURES_INITIAL_BALANCE + total_pnl

    print("\n" + "=" * 50)
    print("BACKTEST RESULT")
    print("=" * 50)
    print(f"Eval Window  : {eval_start_ts} ~ {eval_end_ts}")
    print(f"Final Balance: {final_val:,.0f} (Initial: {FUTURES_INITIAL_BALANCE:,.0f})")
    print(f"Total Return : {ret_pct:.2f}%")
    print(f"Trade Count  : {len(trades_log)}")

    if trades_log:
        wins = [p for p in trades_log if p > 0]
        losses = [p for p in trades_log if p < 0]
        win_rate = (len(wins) / len(trades_log)) * 100.0
        gross_profit = float(sum(wins))
        gross_loss = abs(float(sum(losses)))
        pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        print(f"Win Rate      : {win_rate:.2f}%")
        print(f"Profit Factor : {pf:.2f}")
        print(f"Max drawdown  : {mdd:.2f}%")
    else:
        print("No Trades Executed.")

    print("=" * 50)

    print(f"\nRunning Walk-Forward Analysis (5 Splits, cutoff-anchored)...")
    wfa = FuturesWalkForwardAnalyzer(test_hourly, test_daily, params, eval_start_time=eval_start_ts)
    wfa_results = wfa.run(n_splits=5)

    print(f"{'=' * 50}")
    print(f"WALK FORWARD ANALYSIS RESULT")
    print(f"{'=' * 50}")
    if wfa_results.empty:
        print("Not enough data to run Walk-Forward Analysis.")
    else:
        print(wfa_results.to_markdown(index=False, floatfmt=".2f"))
        avg_wfa_ret = wfa_results['Return'].mean()
        print(f"\nAverage Return per Split: {avg_wfa_ret:.2f}%")
        consistency = len(wfa_results[wfa_results['Return'] > 0]) / len(wfa_results) * 100
        print(f"Consistency (Positive Segments): {consistency:.0f}%")

    print(f"\nRunning Monte Carlo Simulation (10,000 runs)...")
    if trades_log:
        roi_log = [(pnl / FUTURES_INITIAL_BALANCE) * 100.0 for pnl in trades_log]
        mc = FuturesMonteCarloSimulator(roi_log)
        mc_res = mc.run(n_simulations=10000, initial_balance=FUTURES_INITIAL_BALANCE)
        print(f"{'=' * 50}")
        print(f"MONTE CARLO SIMULATION RESULT (95% Confidence)")
        print(f"{'=' * 50}")
        print(f"Probability of Profit : {mc_res['prob_profit']:.2f}%")
        print(f"Expected Return       : {mc_res['mean_return_pct']:.2f}% (Median: {mc_res['median_return_pct']:.2f}%)")
        print(f"Worst Case MDD (5%)   : {mc_res['worst_case_mdd']:.2f}%")
        print(f"Return Range (95%)    : {mc_res['lower_bound_95']:.2f}% ~ {mc_res['upper_bound_95']:.2f}%")
        print("=" * 50)
    else:
        print("Not enough trades for Monte Carlo.")

    return


def load_best_params_from_mysql(mode, storage_url):
    """Load best params from MySQL with robust study-name fallback."""
    target_name = f"futures_{mode.lower()}_strategy"

    def _norm(s):
        return str(s).strip().lower()

    def _read_best(study_name):
        try:
            st = optuna.load_study(study_name=study_name, storage=storage_url)
            return st, st.best_params, st.best_value
        except KeyError:
            return None, None, None
        except ValueError:
            # Study exists but no complete trial.
            return None, None, None
        except Exception as e:
            logger.warning(f"Study load failed for '{study_name}': {e}")
            return None, None, None

    # 1) Exact name first.
    try:
        study, best_params, best_value = _read_best(target_name)
        if study is not None:
            return target_name, best_params, best_value
    except Exception:
        pass

    # 2) Fallback: normalize + fuzzy candidate selection.
    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage_url)
    except Exception as e:
        logger.warning(f"Failed to list studies for fallback: {e}")
        return None, None, None

    if not summaries:
        return None, None, None

    by_norm = {_norm(s.study_name): s.study_name for s in summaries}
    resolved_name = by_norm.get(_norm(target_name))

    if resolved_name is not None:
        study, best_params, best_value = _read_best(resolved_name)
        if study is not None:
            return resolved_name, best_params, best_value
        # If normalized match has no complete trial, keep searching.
        resolved_name = None

    if resolved_name is None:
        needle = f"futures_{mode.lower()}"
        candidates = []
        for s in summaries:
            name_norm = _norm(s.study_name)
            if needle not in name_norm:
                continue
            if "__s1_" in name_norm or "__s2_" in name_norm:
                continue
            candidates.append(s)
        if candidates:
            # Prefer studies with more trials, but require at least one complete trial.
            candidates.sort(key=lambda x: int(x.n_trials or 0), reverse=True)
            for c in candidates:
                study, best_params, best_value = _read_best(c.study_name)
                if study is None:
                    continue
                resolved_name = c.study_name
                logger.warning(
                    f"Study name fallback used: expected '{target_name}', resolved '{resolved_name}'"
                )
                return resolved_name, best_params, best_value

    if resolved_name is None:
        return None, None, None

    study, best_params, best_value = _read_best(resolved_name)
    if study is not None:
        return resolved_name, best_params, best_value
    return None, None, None


def deploy_best_to_local(source_storage_url, source_study_name, mode_label, target_db="futures_strategy.db"):
    """Deploy best trial from source study to local sqlite for bot consumption."""
    try:
        if os.path.exists(target_db):
            os.remove(target_db)

        target_storage = f"sqlite:///{target_db}"
        src_study = optuna.load_study(study_name=source_study_name, storage=source_storage_url)
        best_trial = src_study.best_trial

        optuna.create_study(
            study_name="futures_strategy",
            storage=target_storage,
            direction="maximize",
            load_if_exists=True,
        )
        study_dest = optuna.load_study(study_name="futures_strategy", storage=target_storage)
        frozen_trial = optuna.trial.create_trial(
            params=best_trial.params,
            distributions=best_trial.distributions,
            value=best_trial.value,
        )
        study_dest.add_trial(frozen_trial)
        print(f"[OK] Deployed strategy: {mode_label}")
        print(f"     Target DB: {target_db}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save strategy: {e}")
        return False

def verify_single_symbol_futures(
    symbol,
    best_params,
    primary_symbols,
    eval_start_time=None,
    eval_end_time=None,
    run_robustness_checks=True,
    cost_mult: float = 1.0,
):
    """Single-symbol OOS verification using the same engine execution model."""
    try:
        tf = best_params.get('TIMEFRAME', '1h') # Data Loading
        hourly_df, daily_df = load_data(symbol, FUTURES_BACKTEST_START_DATE, FUTURES_BACKTEST_END_DATE, tf)
    except Exception as e:
        print(f"   Data load error for {symbol}: {e}")
        return None

    eval_start_ts, eval_end_ts = resolve_eval_window(eval_start_time, eval_end_time)
    test_hourly, test_daily = _slice_eval_with_warmup(hourly_df, daily_df, eval_start_ts, eval_end_ts)

    if test_hourly.empty:
        print(f"   No data for {symbol} in eval window. Skipping verification.")
        return None

    from src.futures_strategy.engine_fast_futures import BacktestEngineFast

    strategy = UltimateStrategy(f"Verify_{symbol}", best_params)
    engine = BacktestEngineFast(
        test_hourly, test_daily, strategy, initial_balance=FUTURES_INITIAL_BALANCE
    )
    engine.leverage = best_params.get('LEVERAGE', 1)
    engine.risk_per_trade = best_params.get('RISK_PER_TRADE', 0.02)
    engine.funding_events_per_bar = 3 if best_params.get('TIMEFRAME') == '1d' else 1
    safe_cost_mult = max(0.0, float(cost_mult))
    engine.fee_rate = float(engine.fee_rate) * safe_cost_mult
    engine.slippage_rate = float(engine.slippage_rate) * safe_cost_mult
    # Keep pre-run merged view for no-trade diagnostics (engine frees memory after run).
    merged_df_for_diag = engine.merged_df.copy(deep=False) if getattr(engine, "merged_df", None) is not None else None
    res = engine.run()

    oos_trades = _filter_trades_for_window(res.get('trades_df', pd.DataFrame()), eval_start_ts, eval_end_ts)
    all_trades_count = len(res.get('trades_df', pd.DataFrame()))

    trade_count = len(oos_trades)
    if trade_count > 0:
        oos_pnl = float(oos_trades['pnl'].sum())
        ret_pct = (oos_pnl / FUTURES_INITIAL_BALANCE) * 100.0
        mdd = _calculate_mdd_pct_from_pnl(oos_trades['pnl'], FUTURES_INITIAL_BALANCE)

        win_trades = oos_trades[oos_trades['pnl'] > 0]
        win_rate = (len(win_trades) / trade_count) * 100.0

        pos_pnl = float(win_trades['pnl'].sum())
        neg_pnl = abs(float(oos_trades[oos_trades['pnl'] < 0]['pnl'].sum()))
        pf = pos_pnl / neg_pnl if neg_pnl > 0 else (pos_pnl if pos_pnl > 0 else 0.0)
    else:
        ret_pct = 0.0
        mdd = 0.0
        win_rate = 0.0
        pf = 0.0
        if all_trades_count > 0:
            print(
                f"   [DIAG] Trades exist outside eval window: total={all_trades_count}, "
                f"in-window={trade_count}."
            )
        _diagnose_zero_trade_window(
            merged_df=merged_df_for_diag,
            params=best_params,
            eval_start_ts=eval_start_ts,
            eval_end_ts=eval_end_ts,
        )

    is_primary = symbol in primary_symbols
    indicator = "PRIMARY" if is_primary else "REFERENCE"
    cost_tag = f" x{safe_cost_mult:.1f}" if abs(safe_cost_mult - 1.0) > 1e-9 else ""
    print(
        f"   - {symbol} [{indicator}{cost_tag}]: Return {ret_pct:.2f}% | MDD {mdd:.2f}% "
        f"| Trades {trade_count} | Win {win_rate:.1f}% | PF {pf:.2f}"
    )

    result = {
        'symbol': symbol,
        'return': ret_pct,
        'mdd': mdd,
        'trades': trade_count,
        'win_rate': win_rate,
        'pf': pf,
        'is_primary': is_primary,
        'wfa_results': None,
        'mc_results': None,
        'trades_log': oos_trades['pnl'].tolist() if not oos_trades.empty else [],
        'eval_start': str(eval_start_ts),
        'eval_end': str(eval_end_ts),
    }

    if run_robustness_checks and trade_count >= 10:
        roi_log = [(pnl / FUTURES_INITIAL_BALANCE) * 100.0 for pnl in result['trades_log']]
        print(f"      Running detailed analysis for {symbol}...")

        try:
            from src.futures_strategy.walk_forward_futures import FuturesWalkForwardAnalyzer

            wfa = FuturesWalkForwardAnalyzer(
                test_hourly,
                test_daily,
                best_params,
                eval_start_time=eval_start_ts,
            )
            wfa_results = wfa.run(n_splits=5)

            if not wfa_results.empty:
                avg_wfa_ret = float(wfa_results['Return'].mean())
                consistency = (len(wfa_results[wfa_results['Return'] > 0]) / len(wfa_results)) * 100.0
                result['wfa_results'] = {
                    'avg_return': avg_wfa_ret,
                    'consistency': consistency,
                    'splits': len(wfa_results),
                }
                print(f"         WFA: Avg {avg_wfa_ret:.1f}% | Consistency {consistency:.0f}%")
        except Exception as e:
            logger.warning(f"      WFA failed for {symbol}: {e}")

        try:
            from src.futures_strategy.monte_carlo_futures import FuturesMonteCarloSimulator

            mc = FuturesMonteCarloSimulator(roi_log)
            mc_res = mc.run(
                n_simulations=10000, initial_balance=FUTURES_INITIAL_BALANCE
            )

            result['mc_results'] = {
                'prob_profit': mc_res['prob_profit'],
                'mean_return': mc_res['mean_return_pct'],
                'worst_mdd_95': mc_res['worst_case_mdd'],
                'lower_bound_95': mc_res['lower_bound_95'],
            }
            print(
                f"         MC: Profit Prob {mc_res['prob_profit']:.1f}% "
                f"| Worst MDD(95%) {mc_res['worst_case_mdd']:.1f}%"
            )
        except Exception as e:
            logger.warning(f"      MC failed for {symbol}: {e}")

    return result


def calculate_mode_performance(all_results):
    """Print summary for PRIMARY/REFERENCE and return primary average return."""
    primary_results = [r for r in all_results if r['is_primary']]
    if not primary_results:
        return None
    
    avg_ret = sum(r['return'] for r in primary_results) / len(primary_results)
    print(f"\n   [Summary]")
    print(f"   - PRIMARY Avg Return (BTC/ETH): {avg_ret:.2f}%")
    
    ref_results = [r for r in all_results if not r['is_primary']]
    if ref_results:
        ref_avg = sum(r['return'] for r in ref_results) / len(ref_results)
        print(f"   - REFERENCE Avg Return (Alts): {ref_avg:.2f}%")
    
    # [?곸꽭 遺꾩꽍 ?붿빟] WFA & MC 寃곌낵 (PRIMARY)
    primary_wfa = [r for r in primary_results if r.get('wfa_results')]
    primary_mc = [r for r in primary_results if r.get('mc_results')]
    
    if primary_wfa:
        print(f"\n   [WFA] PRIMARY")
        for r in primary_wfa:
            wfa = r['wfa_results']
            print(f"      {r['symbol']}: Avg {wfa['avg_return']:.1f}% | Consistency {wfa['consistency']:.0f}% ({wfa['splits']} splits)")
    
    if primary_mc:
        print(f"\n   [MC] PRIMARY")
        for r in primary_mc:
            mc = r['mc_results']
            print(f"      {r['symbol']}: Profit Prob {mc['prob_profit']:.1f}% | Worst MDD(95%) {mc['worst_mdd_95']:.1f}%")
    
    # [?곸꽭 遺꾩꽍 ?붿빟] WFA & MC 寃곌낵 (REFERENCE)
    ref_wfa = [r for r in ref_results if r.get('wfa_results')]
    ref_mc = [r for r in ref_results if r.get('mc_results')]
    
    if ref_wfa:
        print(f"\n   [WFA] REFERENCE")
        for r in ref_wfa:
            wfa = r['wfa_results']
            print(f"      {r['symbol']}: Avg {wfa['avg_return']:.1f}% | Consistency {wfa['consistency']:.0f}% ({wfa['splits']} splits)")
    
    if ref_mc:
        print(f"\n   [MC] REFERENCE")
        for r in ref_mc:
            mc = r['mc_results']
            print(f"      {r['symbol']}: Profit Prob {mc['prob_profit']:.1f}% | Worst MDD(95%) {mc['worst_mdd_95']:.1f}%")
    
    return avg_ret


def summarize_profile(results: List[Dict[str, Any]], timeframe: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not results:
        return None
    primary = [r for r in results if r.get("is_primary")]
    if not primary:
        return None

    core_returns = np.array([float(r["return"]) for r in primary], dtype=np.float64)
    core_mdd_abs = np.array([abs(float(r["mdd"])) for r in primary], dtype=np.float64)
    core_pf = np.array([float(r["pf"]) for r in primary], dtype=np.float64)

    core_trades = np.array([int(r["trades"]) for r in primary], dtype=np.int64)
    wfa_consist = []
    mc_worst_abs = []
    for r in primary:
        if r.get("wfa_results"):
            wfa_consist.append(r["wfa_results"]["consistency"])
        if r.get("mc_results"):
            # MC worst MDD is usually negative in output; convert to absolute risk.
            mc_worst_abs.append(abs(float(r["mc_results"]["worst_mdd_95"])))

    core_wfa_consistency = float(np.mean(wfa_consist)) if wfa_consist else 0.0
    core_mc_worst_mdd_abs = float(np.mean(mc_worst_abs)) if mc_worst_abs else 0.0

    alts = [r for r in results if not r.get("is_primary")]
    alt_returns = np.array([float(r["return"]) for r in alts], dtype=np.float64) if alts else np.array([], dtype=np.float64)
    alt_mdd_abs = np.array([abs(float(r["mdd"])) for r in alts], dtype=np.float64) if alts else np.array([], dtype=np.float64)
    alt_trades = np.array([int(r["trades"]) for r in alts], dtype=np.int64) if alts else np.array([], dtype=np.int64)
    alt_median_ret = float(np.median(alt_returns)) if alt_returns.size else 0.0
    alt_p25_ret = float(np.percentile(alt_returns, 25)) if alt_returns.size else 0.0
    alt_pos_rate = float(np.mean(alt_returns > 0)) if alt_returns.size else 0.0
    alt_worst_mdd = float(np.max(alt_mdd_abs)) if alt_mdd_abs.size else 0.0

    all_returns = np.array([float(r["return"]) for r in results], dtype=np.float64)
    mean_abs = max(abs(float(np.mean(all_returns))), 1e-9)
    dispersion = float(np.std(all_returns) / mean_abs)

    gates = SELECTION_POLICY["gates"]
    gate_core_min_trades = futures_selection_min_trades_for_timeframe(timeframe)
    # Hard gates: survival first.
    gate_reasons = []
    if np.any(core_returns <= gates["core_min_return"]):
        gate_reasons.append("core_negative_return")
    if int(np.min(core_trades)) < int(gate_core_min_trades):
        gate_reasons.append("core_trade_count_low")
    if float(np.mean(core_mdd_abs)) > gates["core_max_avg_mdd_abs"]:
        gate_reasons.append("core_mdd_too_high")
    if core_wfa_consistency < gates["core_min_wfa_consistency"]:
        gate_reasons.append("core_wfa_low")
    if core_mc_worst_mdd_abs > gates["core_max_mc_worst_mdd_95_abs"]:
        gate_reasons.append("core_mc_mdd_too_high")
    if alt_returns.size:
        if alt_pos_rate < gates["alt_min_pos_rate"]:
            gate_reasons.append("alt_pos_rate_low")
        if alt_worst_mdd > gates["alt_max_worst_mdd_abs"]:
            gate_reasons.append("alt_worst_mdd_too_high")
        if alt_p25_ret < gates["alt_min_p25_return"]:
            gate_reasons.append("alt_tail_return_too_low")

    gates_passed = len(gate_reasons) == 0

    return {
        "policy_version": SELECTION_POLICY_VERSION,
        "core_avg_ret": float(np.mean(core_returns)),
        "core_avg_mdd_abs": float(np.mean(core_mdd_abs)),
        "core_avg_pf": float(np.mean(core_pf)),
        "core_min_trades": int(np.min(core_trades)),
        "gate_core_min_trades": gate_core_min_trades,
        "core_avg_trades": float(np.mean(core_trades)),
        "core_wfa_consistency": core_wfa_consistency,
        "core_mc_worst_mdd_95": core_mc_worst_mdd_abs,
        "alt_median_ret": alt_median_ret,
        "alt_p25_ret": alt_p25_ret,
        "alt_pos_rate": alt_pos_rate,
        "alt_worst_mdd_abs": alt_worst_mdd,
        "alt_avg_trades": float(np.mean(alt_trades)) if alt_trades.size else 0.0,
        "dispersion": dispersion,
        "gates_passed": gates_passed,
        "gate_reasons": gate_reasons,
    }


def rank_profiles(profile_summaries):
    w = SELECTION_POLICY["weights"]
    ranked = []
    for key, s in profile_summaries.items():
        if not s:
            continue
        if not s.get("gates_passed", False):
            continue

        core_return_score = _log_scaled_positive(s["core_avg_ret"], cap=3000.0)
        core_pf_score = _log_scaled_positive(s["core_avg_pf"], cap=20.0)
        core_wfa_score = _clip01(s["core_wfa_consistency"] / 100.0)
        core_mdd_score = 1.0 - _clip01(s["core_avg_mdd_abs"] / 25.0)
        core_mc_score = 1.0 - _clip01(s["core_mc_worst_mdd_95"] / 60.0)

        core_score = (
            (w["core_return"] * core_return_score)
            + (w["core_pf"] * core_pf_score)
            + (w["core_wfa"] * core_wfa_score)
            + (w["core_mdd"] * core_mdd_score)
            + (w["core_mc"] * core_mc_score)
        )

        alt_median_score = _log_scaled_positive(s["alt_median_ret"], cap=800.0)
        alt_p25_score = _clip01((s["alt_p25_ret"] + 30.0) / 130.0)
        alt_pos_score = _clip01(s["alt_pos_rate"])
        alt_mdd_score = 1.0 - _clip01(s["alt_worst_mdd_abs"] / 70.0)
        alt_score = (
            (w["alt_median"] * alt_median_score)
            + (w["alt_p25"] * alt_p25_score)
            + (w["alt_pos"] * alt_pos_score)
            + (w["alt_mdd"] * alt_mdd_score)
        )

        diversification_score = 1.0 - _clip01(s["dispersion"] / 3.0)
        base_score = (
            (w["core_total"] * core_score)
            + (w["alt_total"] * alt_score)
            + (w["div_total"] * diversification_score)
        )
        # Trade frequency soft penalty:
        # keep low-trade strategies viable, but gently prefer richer sample sizes.
        min_trades = int(s.get("core_min_trades", 0))
        trade_penalty = 1.0 / (1.0 + np.exp(-0.3 * (float(min_trades) - 25.0)))
        score = float(base_score * trade_penalty)
        ranked.append((key, score, s))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

if __name__ == "__main__":
    import argparse
    import optuna
    import shutil
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT",
                        help="Comma-separated list of symbols to verify")
    parser.add_argument("--alt", type=int, default=0, choices=[0, 1],
                        help="Include altcoins for validation (default: 0). Adds SOL, XRP, DOGE, BNB")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify current deployed strategy (futures_strategy.db) without saving. Skips MySQL and deployment.")
    parser.add_argument("--bonus-sweep", dest="bonus_sweep", action="store_true",
                        help="Verify A/B/C bonus DBs and pick a winner automatically (default: enabled).")
    parser.add_argument("--no-bonus-sweep", dest="bonus_sweep", action="store_false",
                        help="Disable A/B/C bonus sweep and verify only DB_NAME from .env.")
    parser.add_argument("--rolling-oos", dest="rolling_oos", action="store_true",
                        help="Run additional rolling OOS verification for final winner (default: enabled).")
    parser.add_argument("--no-rolling-oos", dest="rolling_oos", action="store_false",
                        help="Disable rolling OOS verification.")
    parser.add_argument("--roll-window-days", type=int, default=120,
                        help="Rolling OOS window length in days (default: 120).")
    parser.add_argument("--roll-step-days", type=int, default=30,
                        help="Rolling OOS step in days (default: 30).")
    parser.add_argument("--roll-max-windows", type=int, default=6,
                        help="Maximum number of rolling OOS windows (default: 6).")
    parser.set_defaults(bonus_sweep=True)
    parser.set_defaults(rolling_oos=True)
    args = parser.parse_args()
    
    # Build symbol list
    base_symbols = [s.strip() for s in args.symbols.split(',')]
    if args.alt == 1:
        alt_symbols = ['SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'BNB/USDT']
        for alt in alt_symbols:
            if alt not in base_symbols:
                base_symbols.append(alt)
        print(f"[INFO] Altcoin validation enabled: {', '.join(alt_symbols)}")
    
    symbols = base_symbols
    PRIMARY_SYMBOLS = ['BTC/USDT', 'ETH/USDT']
    
    # UNIFIED-only verification
    MODES = ['UNIFIED']
    print("[INFO] Verification mode: UNIFIED only")
    
    
    # Dry-run: verify deployed local strategy only.
    if args.dry_run:
        _log_header("DRY-RUN: Verify Current Deployed Strategy", "Source: futures_strategy.db")
        
        target_db = "futures_strategy.db"
        if not os.path.exists(target_db):
            print(f"[ERROR] {target_db} not found. Deploy a strategy first.")
            sys.exit(1)
        
        try:
            # Load current local strategy
            local_storage = f"sqlite:///{target_db}"
            study = optuna.load_study(study_name="futures_strategy", storage=local_storage)
            best_params = study.best_params
            train_score = study.best_value
            
            print(f"[INFO] Loaded current strategy (Train Score: {train_score:.4f})")
            print(f"   Timeframe: {best_params.get('TIMEFRAME')}")
            print(f"   Leverage: {best_params.get('LEVERAGE', 1)}x")
            
            # OOS verification
            all_results = []
            for symbol in symbols:
                result = verify_single_symbol_futures(symbol, best_params, PRIMARY_SYMBOLS)
                if result:
                    all_results.append(result)
            
            avg_ret = calculate_mode_performance(all_results)
            dry_run_summary = summarize_profile(all_results, timeframe=best_params.get("TIMEFRAME"))
            _record_verification_comparison(
                log_path=FUTURES_VERIFY_LOG_PATH,
                run_type="dry_run",
                current_best={
                    "label": "deployed_local_strategy",
                    "source_db_name": "futures_strategy.db",
                    "study_name": "futures_strategy",
                    "timeframe": str(best_params.get("TIMEFRAME", "1h")),
                    "leverage": _safe_float(best_params.get("LEVERAGE")),
                    "metrics": _extract_summary_metrics(
                        dry_run_summary,
                        {
                            "avg_ret": avg_ret,
                            "train_score": train_score,
                        },
                    ),
                    "summary": dry_run_summary,
                },
                context={
                    "symbols": list(symbols),
                    "deployed": False,
                },
            )
            
            _log_header("DRY-RUN COMPLETE", "No changes saved")
            if avg_ret is not None:
                print(f"[INFO] Current strategy OOS performance: {avg_ret:.2f}%")
            
            sys.exit(0)
            
        except Exception as e:
            print(f"[ERROR] Failed to load deployed strategy: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Normal mode: verify from MySQL and deploy best.
    _log_header(
        "INTEGRATED STRATEGY VERIFICATION (Futures)",
        f"Searching optimized strategies: {MODES}",
    )
    
    results = []
    
    # MySQL setup
    from dotenv import load_dotenv
    from urllib.parse import quote_plus
    load_dotenv()
    
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "trading_optuna")
    
    if not all([db_user, db_pass, db_name]):
        print("[ERROR] Missing DB credentials in .env")
        sys.exit(1)
        
    safe_pass = quote_plus(db_pass)
    storage_url = f"mysql+pymysql://{db_user}:{safe_pass}@{db_host}:{db_port}/{db_name}"

    bonus_dbs = {
        "A": "trading_optuna_bonus_a",
        "B": "trading_optuna_bonus_b",
        "C": "trading_optuna_bonus_c",
    }
    
    best_overall_score = -float('inf')
    best_mode = None
    best_study_name = None
    if args.bonus_sweep:
        oos_start, selection_end, holdout_start, holdout_end = split_bonus_oos_windows()

        _log_header(
            "BONUS SWEEP VERIFICATION (A/B/C)",
            f"Selection window (rank): {oos_start} ~ {selection_end}\n"
            f"Holdout window (sanity): {holdout_start} ~ {holdout_end}",
        )

        profile_results = {}
        unified_source = {}
        for key, db in bonus_dbs.items():
            _log_subheader(f"[{key}] DB: {db}")
            profile_url = f"mysql+pymysql://{db_user}:{safe_pass}@{db_host}:{db_port}/{db}"
            profile_summary = {}

            for mode in MODES:
                print(f"\nVerifying {mode} Mode Strategy (from MySQL)...")
                study_name, best_params, train_score = load_best_params_from_mysql(mode, profile_url)
                if study_name is None:
                    print(f"[WARN] {mode} strategy not found in MySQL. Skipping.")
                    try:
                        names = optuna.study.get_all_study_names(storage=profile_url)
                        preview = ", ".join(names[:8]) if names else "(none)"
                        print(f"      available studies: {preview}")
                    except Exception as e:
                        print(f"      study list read failed: {e}")
                    continue

                print(f"   Loaded Params (Train Score: {train_score:.4f})")
                print(f"   Timeframe: {best_params.get('TIMEFRAME')}")

                all_results = []
                for symbol in symbols:
                    result = verify_single_symbol_futures(
                        symbol,
                        best_params,
                        PRIMARY_SYMBOLS,
                        eval_start_time=oos_start,
                        eval_end_time=selection_end,
                        run_robustness_checks=True,
                    )
                    if result:
                        all_results.append(result)

                avg_ret = calculate_mode_performance(all_results)
                summary = summarize_profile(all_results, timeframe=best_params.get("TIMEFRAME"))
                profile_summary[mode] = {
                    "avg_ret": avg_ret,
                    "summary": summary,
                }
                if mode == "UNIFIED":
                    unified_source[key] = {
                        "storage_url": profile_url,
                        "study_name": study_name,
                        "db_name": db,
                        "best_params": best_params,
                        "train_score": train_score,
                    }

            profile_results[key] = profile_summary

        unified_summaries = {
            k: (v.get("UNIFIED", {}) or {}).get("summary") for k, v in profile_results.items()
        }
        ranked = rank_profiles(unified_summaries)

        _log_header(
            "BONUS SWEEP WINNER (UNIFIED) - SELECTION WINDOW",
            f"Policy: {SELECTION_POLICY_VERSION}",
        )
        for k, s in unified_summaries.items():
            if not s:
                print(f"- {k}: no summary")
                continue
            gate_state = "PASS" if s.get("gates_passed") else f"FAIL({','.join(s.get('gate_reasons', []))})"
            print(
                f"- {k}: {gate_state} | core_ret={s['core_avg_ret']:.2f}% | "
                f"core_mdd={s['core_avg_mdd_abs']:.2f}% | core_pf={s['core_avg_pf']:.2f} | core_min_trades={s['core_min_trades']} | "
                f"core_wfa={s['core_wfa_consistency']:.1f}% | core_mc_mdd95={s['core_mc_worst_mdd_95']:.2f}% | "
                f"alt_med={s['alt_median_ret']:.2f}% | alt_p25={s['alt_p25_ret']:.2f}% | "
                f"alt_pos={s['alt_pos_rate']*100:.1f}% | dispersion={s['dispersion']:.2f}"
            )

        if not ranked:
            print("[ERROR] No valid summaries to rank.")
            sys.exit(1)

        winner, score, s = ranked[0]
        print(f"Winner (selection): {winner} (score={score:.2f})")
        print(
            f"   core_ret={s['core_avg_ret']:.2f}% | core_mdd={s['core_avg_mdd_abs']:.2f}% | "
            f"core_pf={s['core_avg_pf']:.2f} | core_wfa={s['core_wfa_consistency']:.1f}% | "
            f"core_mc_mdd95={s['core_mc_worst_mdd_95']:.2f}% | "
            f"alt_med={s['alt_median_ret']:.2f}% | alt_p25={s['alt_p25_ret']:.2f}% | "
            f"alt_pos={s['alt_pos_rate']*100:.1f}%"
        )

        holdout_min_trades_gate = compute_dynamic_holdout_min_trades(holdout_start, holdout_end)
        print(
            f"[INFO] Holdout dynamic trade gate: min_trades>={holdout_min_trades_gate} "
            f"(window: {holdout_start} ~ {holdout_end})"
        )

        holdout_evals = {}
        for candidate_key, _, _ in ranked:
            candidate_src = unified_source.get(candidate_key)
            if not candidate_src or not candidate_src.get("best_params"):
                holdout_evals[candidate_key] = {
                    "passed": False,
                    "reasons": ["holdout_not_evaluated"],
                    "summary": None,
                    "source": candidate_src,
                }
                continue

            print("\n" + "-" * LOG_WIDTH)
            print(f"Holdout Verification for Candidate [{candidate_key}] (used for final selection)")
            print(f"Window: {holdout_start} ~ {holdout_end}")
            print("-" * LOG_WIDTH)

            holdout_results = []
            for symbol in symbols:
                result = verify_single_symbol_futures(
                    symbol,
                    candidate_src["best_params"],
                    PRIMARY_SYMBOLS,
                    eval_start_time=holdout_start,
                    eval_end_time=holdout_end,
                    run_robustness_checks=False,
                )
                if result:
                    holdout_results.append(result)

            calculate_mode_performance(holdout_results)
            holdout_summary = summarize_profile(
                holdout_results,
                timeframe=candidate_src.get("best_params", {}).get("TIMEFRAME"),
            )
            holdout_passed, holdout_reasons = evaluate_holdout_sanity(
                holdout_summary,
                min_trades_gate=holdout_min_trades_gate,
            )

            if holdout_summary:
                print(
                    f"Holdout Summary: core_ret={holdout_summary['core_avg_ret']:.2f}% | "
                    f"core_mdd={holdout_summary['core_avg_mdd_abs']:.2f}% | "
                    f"core_pf={holdout_summary['core_avg_pf']:.2f} | "
                    f"core_min_trades={holdout_summary['core_min_trades']}"
                )
            print(
                "Holdout Gate: PASS"
                if holdout_passed
                else f"Holdout Gate: FAIL({','.join(holdout_reasons)})"
            )

            holdout_evals[candidate_key] = {
                "passed": bool(holdout_passed),
                "reasons": holdout_reasons,
                "summary": holdout_summary,
                "source": candidate_src,
            }

        # Among holdout-passed candidates, choose by holdout score (OOS performance), not selection order.
        deployment_candidate = None
        passed_with_scores: List[Tuple[str, float, int]] = []
        for rank_idx, (candidate_key, _, _) in enumerate(ranked):
            ev = holdout_evals.get(candidate_key)
            if not ev or not ev.get("passed"):
                continue
            summary = ev.get("summary")
            ho_score = score_holdout_candidate(summary) if summary else -1.0
            passed_with_scores.append((candidate_key, ho_score, rank_idx))
        if passed_with_scores:
            passed_with_scores.sort(key=lambda x: (-x[1], x[2]))
            deployment_candidate = passed_with_scores[0][0]

        if deployment_candidate:
            winner = deployment_candidate
            winner_src = holdout_evals[winner]["source"]
            winner_ho_score = next((s for k, s, _ in passed_with_scores if k == winner), -1.0)
            print(f"[INFO] Final winner after holdout gate: {winner} (holdout_score={winner_ho_score:.4f})")
            holdout_passed = True
        else:
            winner_src = unified_source.get(winner)
            holdout_passed = False
            print("[WARN] No holdout-passed candidate found. Deployment skipped.")

        deployment_attempted = False
        deployment_ok = False
        if winner_src and holdout_passed:
            # Additional durability check: keep base selection/holdout unchanged,
            # and run stressed-cost scenarios for practical execution robustness.
            run_cost_stress_verification(
                best_params=winner_src["best_params"],
                symbols=symbols,
                primary_symbols=PRIMARY_SYMBOLS,
                eval_start_time=holdout_start,
                eval_end_time=holdout_end,
            )
            if args.rolling_oos:
                run_rolling_oos_verification(
                    best_params=winner_src["best_params"],
                    symbols=symbols,
                    primary_symbols=PRIMARY_SYMBOLS,
                    eval_start_time=oos_start,
                    eval_end_time=holdout_end,
                    window_days=args.roll_window_days,
                    step_days=args.roll_step_days,
                    max_windows=args.roll_max_windows,
                )
            print("-" * LOG_WIDTH)
            print(
                f"Saving Winner ({winner}) from MySQL "
                f"[{winner_src['db_name']}/{winner_src['study_name']}] to 'futures_strategy.db'..."
            )
            deployment_attempted = True
            deployment_ok = deploy_best_to_local(
                source_storage_url=winner_src["storage_url"],
                source_study_name=winner_src["study_name"],
                mode_label=f"UNIFIED/{winner}",
            )
        else:
            print("[WARN] Skip deployment due to failed/missing holdout sanity check.")
        selected_eval = holdout_evals.get(winner, {}) if isinstance(holdout_evals, dict) else {}
        selected_holdout_summary = (
            selected_eval.get("summary") if isinstance(selected_eval, dict) else None
        )
        selected_selection_summary = unified_summaries.get(winner)
        selected_summary = selected_holdout_summary or selected_selection_summary
        selected_source = (
            selected_eval.get("source") if isinstance(selected_eval, dict) else None
        ) or winner_src or unified_source.get(winner)
        selected_rank_score = next(
            (float(rank_score) for key, rank_score, _ in ranked if key == winner),
            None,
        )
        _record_verification_comparison(
            log_path=FUTURES_VERIFY_LOG_PATH,
            run_type="bonus_sweep",
            current_best={
                "label": str(winner),
                "selected_mode": "UNIFIED",
                "study_name": selected_source.get("study_name") if isinstance(selected_source, dict) else None,
                "source_db_name": selected_source.get("db_name") if isinstance(selected_source, dict) else None,
                "timeframe": str(
                    (selected_source.get("best_params", {}) if isinstance(selected_source, dict) else {}).get(
                        "TIMEFRAME",
                        "1h",
                    )
                ),
                "leverage": _safe_float(
                    (selected_source.get("best_params", {}) if isinstance(selected_source, dict) else {}).get(
                        "LEVERAGE"
                    )
                ),
                "holdout_gate_state": "PASS" if holdout_passed else "FAIL",
                "holdout_gate_reasons": (
                    selected_eval.get("reasons", [])
                    if isinstance(selected_eval, dict)
                    else ["holdout_not_evaluated"]
                ),
                "metrics": _extract_summary_metrics(
                    selected_summary,
                    {
                        "selection_rank_score": selected_rank_score,
                        "train_score": selected_source.get("train_score")
                        if isinstance(selected_source, dict)
                        else None,
                    },
                ),
                "selection_summary": selected_selection_summary,
                "holdout_summary": selected_holdout_summary,
            },
            context={
                "symbols": list(symbols),
                "selection_window_start": str(oos_start),
                "selection_window_end": str(selection_end),
                "holdout_window_start": str(holdout_start),
                "holdout_window_end": str(holdout_end),
                "holdout_passed": bool(holdout_passed),
                "deployment_attempted": bool(deployment_attempted),
                "deployment_ok": bool(deployment_ok),
            },
        )
        if deployment_attempted and not deployment_ok:
            sys.exit(1)

        print("=" * LOG_WIDTH)
        sys.exit(0)
    else:
        for mode in MODES:
            print(f"\nVerifying {mode} mode strategy (from MySQL)...")
            
            try:
                study_name, best_params, train_score = load_best_params_from_mysql(mode, storage_url)
                if study_name is None:
                    print(f"[WARN] {mode} strategy not found in MySQL. Skipping.")
                    continue
                
                print(f"[INFO] Loaded params (Train Score: {train_score:.4f})")
                print(f"   Timeframe: {best_params.get('TIMEFRAME')}")
                
                # OOS verification
                all_results = []
                for symbol in symbols:
                    result = verify_single_symbol_futures(symbol, best_params, PRIMARY_SYMBOLS)
                    if result:
                        all_results.append(result)
                
                avg_ret = calculate_mode_performance(all_results)
                if avg_ret is None:
                    continue
                
                summary = summarize_profile(all_results, timeframe=best_params.get("TIMEFRAME"))
                results.append({
                    'mode': mode,
                    'study_name': study_name,
                    'return': avg_ret,
                    'score': train_score,
                    'all_results': all_results,
                    'summary': summary,
                    'best_params': best_params,
                })
                
                if avg_ret > best_overall_score:
                    best_overall_score = avg_ret
                    best_mode = mode
                    best_study_name = study_name
                
            except Exception as e:
                print(f"[ERROR] Failed to process {mode}: {e}")
                import traceback
                traceback.print_exc()

    _log_header("FINAL RESULTS")
    
    if not results:
        print("[ERROR] No valid strategies found/verified.")
        sys.exit(1)
        
    results.sort(key=lambda x: x['return'], reverse=True)
    
    for res in results:
        mark = "*" if res['mode'] == best_mode else " "
        print(f"{mark} {res['mode']:<6} | OOS Return: {res['return']:>7.2f}% | Train Score: {res['score']:.4f}")
        
    print("-" * LOG_WIDTH)
    
    target_db = "futures_strategy.db"
    
    selected_result = next((res for res in results if res['mode'] == best_mode), results[0])
    deployment_attempted = False
    deployment_ok = False
    if best_study_name:
        print(f"[INFO] Saving best strategy ({best_mode}) from MySQL to '{target_db}'...")
        print("       Migrating best params to standard 'futures_strategy' study...")
        deployment_attempted = True
        deployment_ok = deploy_best_to_local(
            source_storage_url=storage_url,
            source_study_name=best_study_name,
            mode_label=str(best_mode),
            target_db=target_db,
        )
    else:
        print("[WARN] No best strategy selected.")
    _record_verification_comparison(
        log_path=FUTURES_VERIFY_LOG_PATH,
        run_type="single_mode",
        current_best={
            "label": str(selected_result.get("mode")),
            "selected_mode": str(selected_result.get("mode")),
            "study_name": selected_result.get("study_name"),
            "source_db_name": db_name,
            "timeframe": str(selected_result.get("best_params", {}).get("TIMEFRAME", "1h")),
            "leverage": _safe_float(selected_result.get("best_params", {}).get("LEVERAGE")),
            "holdout_gate_state": "SKIPPED",
            "holdout_gate_reasons": ["bonus_sweep_disabled"],
            "metrics": _extract_summary_metrics(
                selected_result.get("summary"),
                {
                    "avg_ret": selected_result.get("return"),
                    "train_score": selected_result.get("score"),
                },
            ),
            "summary": selected_result.get("summary"),
        },
        context={
            "symbols": list(symbols),
            "deployment_attempted": bool(deployment_attempted),
            "deployment_ok": bool(deployment_ok),
        },
    )
    if deployment_attempted and not deployment_ok:
        sys.exit(1)



