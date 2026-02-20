import argparse
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import quote_plus

project_root = os.getcwd()
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    if project_root not in sys.path:
        sys.path.append(project_root)

from config.settings import (
    SPOT_BACKTEST_START_DATE,
    SPOT_BACKTEST_END_DATE,
    SPOT_TRAIN_CUTOFF_DATE,
    DATA_DIR,
)
from src.common.utils import setup_logger
from src.spot_strategy.engine_fast_spot import BacktestEngineFastSpot, backtest_loop_spot_numba
from src.spot_strategy.monte_carlo_spot import SpotMonteCarloSimulator
from src.spot_strategy.upbit_client import UpbitClient
from src.spot_strategy.walk_forward_spot import SpotWalkForwardAnalyzer
from src.strategy.strategies import UltimateStrategy

logger = setup_logger("SpotVerifier", write_file=False)
logger.setLevel(logging.WARNING)

LOG_WIDTH = 80
SPOT_VERIFY_LOG_PATH = Path(project_root) / "logs" / "spot_verify_comparison.jsonl"
SPOT_COMPARISON_METRICS: Tuple[str, ...] = (
    "core_avg_ret",
    "core_avg_excess_ret",
    "core_avg_mdd_abs",
    "core_avg_pf",
    "core_avg_pf_clipped_shrunk",
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
SPOT_LOWER_IS_BETTER_METRICS = {
    "core_avg_mdd_abs",
    "core_mc_worst_mdd_95",
    "alt_worst_mdd_abs",
    "dispersion",
}
SPOT_INITIAL_BALANCE = 1_000_000.0
SPOT_BASE_FEE = 0.0005
SPOT_BASE_SLIPPAGE = 0.0003
WARMUP_DAYS = 60
COST_STRESS_MULTIPLIERS = (1.5, 2.0)
BONUS_SWEEP_HOLDOUT_RATIO = 0.30
BONUS_SWEEP_MIN_HOLDOUT_DAYS = 120
# Selection window dynamically tied to SPOT_TRAIN_CUTOFF_DATE.
# Selection covers the period leading up to the cutoff (In-Sample or Look-back).
# Holdout (OOS) covers the period AFTER the cutoff.
_train_cutoff_ts = pd.Timestamp(SPOT_TRAIN_CUTOFF_DATE)
BONUS_SELECTION_FIXED_START_DATE = str((_train_cutoff_ts - pd.DateOffset(months=18)).date())
BONUS_SELECTION_FIXED_END_DATE = str((_train_cutoff_ts - pd.Timedelta(seconds=1)).date())
# OOS is enabled only when unseen post-selection data is sufficiently long.
# For spot (BTC/ETH primary, 4h often sparse), 6 months is the minimum practical floor.
# Enable holdout sanity checks once at least ~3 months of unseen data are available.
HOLDOUT_ACTIVATION_MIN_DAYS = 90
GAP_FILL_MAX_RANGES = int(os.getenv("SPOT_GAP_FILL_MAX_RANGES", "3"))

# Process-local cache to avoid repeated disk I/O + repeated gap backfills across
# holdout/cost-stress/rolling windows in the same run.
_SPOT_DATA_CACHE: Dict[Tuple[str, str, str, str], Tuple[pd.DataFrame, pd.DataFrame]] = {}
_SPOT_GAP_FILLED_KEYS: set = set()
SHOW_ZERO_TRADE_DIAG = False
SELECTION_POLICY_VERSION = "SPOT_SELECTION_POLICY_V11_SIMPLE"
# Timeframe-specific min trades for selection gate (aligned with futures: lower TF = more bars → higher min).
SPOT_TF_SELECTION_MIN_TRADES: Dict[str, int] = {
    "30m": 12,
    "1h": 10,
    "4h": 8,
}
SPOT_SELECTION_MIN_TRADES_DEFAULT = 10

SELECTION_POLICY = {
    "gates": {
        "core_min_return": 0.0,
        "core_min_trades": 10,
        "core_max_avg_mdd_abs": 25.0,
        "core_min_wfa_consistency": 40.0,
        "core_max_mc_worst_mdd_95_abs": 45.0,
        "alt_min_pos_rate": 0.50,
        "alt_max_worst_mdd_abs": 50.0,
        "alt_min_p25_return": -20.0,
    },
    # Core-first: 70% core (return/risk), 25% alt, 5% div so selection favors return over uniformity.
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
    "robust": {
        "pf_clip_per_symbol": 6.0,
    },
}

HOLDOUT_SANITY_GATES = {
    "core_min_return": 0.0,
    "core_min_trades": 24,  # long-only: lower than futures fallback (30)
    "core_max_avg_mdd_abs": 35.0,
}
# TF-specific holdout min_trades: (coef, floor) per TF; longer TF -> fewer bars -> lower gate.
SPOT_HOLDOUT_TF_COEF_FLOOR: Dict[str, Tuple[float, int]] = {
    "30m": (0.10, 24),
    "1h": (0.065, 20),
    "4h": (0.035, 16),
}
SPOT_HOLDOUT_DEFAULT_COEF, SPOT_HOLDOUT_DEFAULT_FLOOR = 0.10, 24
# PF shrinkage reference trades (for _shrink_pf_by_trades; scoring only, not gate).
SPOT_PF_SHRINK_REF_TRADES = 30.0
ROLLING_SANITY_GATES = {
    "min_windows": 2,
    "min_pf_pass_rate": 0.60,
    "min_ret_pass_rate": 0.50,
    "min_median_pf": 1.00,
    "min_median_ret": 0.0,
    "max_worst_mdd_abs": 18.0,
}
def assert_strict_oos_window(selection_end: pd.Timestamp, gate_start: pd.Timestamp, gate_end: pd.Timestamp) -> None:
    sel_end = pd.Timestamp(selection_end)
    g_start = pd.Timestamp(gate_start)
    g_end = pd.Timestamp(gate_end)
    if g_start <= sel_end:
        raise ValueError(
            f"Invalid holdout gate window overlap: selection_end={sel_end}, gate_start={g_start}"
        )
    if g_end < g_start:
        raise ValueError(f"Invalid holdout gate window: start={g_start}, end={g_end}")


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
    for key in SPOT_COMPARISON_METRICS:
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
        logger.warning(f"[WARN] Failed to read previous spot comparison log: {exc}")
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
        lower_is_better = metric in SPOT_LOWER_IS_BETTER_METRICS
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
        "strategy_type": "spot",
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
        logger.warning(f"[WARN] Failed to write spot comparison log: {exc}")


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _shrink_pf_by_trades(pf: float, trades: float, ref_trades: float = SPOT_PF_SHRINK_REF_TRADES) -> float:
    p = max(0.0, float(pf))
    n = max(0.0, float(trades))
    ref = max(1.0, float(ref_trades))
    weight = min(1.0, float(np.sqrt(n / ref)))
    return float(1.0 + (p - 1.0) * weight)


def _log_scaled_positive(x: float, cap: float) -> float:
    if x <= 0:
        return 0.0
    return _clip01(np.log1p(float(x)) / np.log1p(float(cap)))


def _to_inclusive_end_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    if value == value.normalize():
        return value + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    return value


def resolve_eval_window(
    eval_start_time: Optional[pd.Timestamp] = None,
    eval_end_time: Optional[pd.Timestamp] = None,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    eval_start = pd.Timestamp(eval_start_time) if eval_start_time is not None else pd.Timestamp(SPOT_TRAIN_CUTOFF_DATE)
    raw_end = pd.Timestamp(eval_end_time) if eval_end_time is not None else pd.Timestamp(SPOT_BACKTEST_END_DATE)
    eval_end = _to_inclusive_end_timestamp(raw_end)
    if eval_end < eval_start:
        raise ValueError(f"Invalid eval window: start={eval_start}, end={eval_end}")
    return eval_start, eval_end


def split_bonus_oos_windows() -> Tuple[pd.Timestamp, pd.Timestamp, Optional[pd.Timestamp], Optional[pd.Timestamp], bool, int]:
    data_start, data_end = resolve_eval_window(pd.Timestamp(SPOT_TRAIN_CUTOFF_DATE), pd.Timestamp(SPOT_BACKTEST_END_DATE))
    sel_fixed_start = pd.Timestamp(BONUS_SELECTION_FIXED_START_DATE)
    sel_fixed_end = _to_inclusive_end_timestamp(pd.Timestamp(BONUS_SELECTION_FIXED_END_DATE))

    # Selection is defined as the period BEFORE the cutoff (look-back from cutoff).
    # Since optimize used data up to cutoff, selection here acts as a high-fidelity
    # in-sample cross-validation window.
    selection_start = sel_fixed_start
    selection_end = sel_fixed_end
    if selection_end < selection_start:
        raise ValueError(
            f"Invalid fixed selection window after clipping: "
            f"selection_start={selection_start}, selection_end={selection_end}, "
            f"data_start={data_start}, data_end={data_end}"
        )

    holdout_start: Optional[pd.Timestamp] = None
    holdout_end: Optional[pd.Timestamp] = None
    holdout_days = 0
    holdout_enabled = False

    candidate_holdout_start = selection_end + pd.Timedelta(milliseconds=1)
    if candidate_holdout_start <= data_end:
        holdout_start = candidate_holdout_start
        holdout_end = data_end
        holdout_days = max(0, int((holdout_end.normalize() - holdout_start.normalize()).days) + 1)
        holdout_enabled = holdout_days >= int(max(1, HOLDOUT_ACTIVATION_MIN_DAYS))

    return selection_start, selection_end, holdout_start, holdout_end, holdout_enabled, holdout_days


def spot_selection_min_trades_for_timeframe(timeframe: Optional[str]) -> int:
    """Return selection gate min trades for the given timeframe; default if unknown (aligned with futures)."""
    if not timeframe:
        return SPOT_SELECTION_MIN_TRADES_DEFAULT
    tf = str(timeframe).strip().lower()
    return int(SPOT_TF_SELECTION_MIN_TRADES.get(tf, SPOT_SELECTION_MIN_TRADES_DEFAULT))


def compute_dynamic_holdout_min_trades(
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
    timeframe: Optional[str] = None,
) -> int:
    """Spot: TF-aware holdout gate (long-only; longer TF -> lower coef/floor). Same formula shape as futures."""
    holdout_days = max(1, int((holdout_end.normalize() - holdout_start.normalize()).days) + 1)
    tf = str(timeframe).strip().lower() if timeframe else ""
    coef, floor = SPOT_HOLDOUT_TF_COEF_FLOOR.get(tf, (SPOT_HOLDOUT_DEFAULT_COEF, SPOT_HOLDOUT_DEFAULT_FLOOR))
    dynamic_floor = int(round(holdout_days * coef))
    return max(floor, dynamic_floor)


def evaluate_holdout_sanity(
    summary: Optional[Dict],
    min_trades_gate: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    """Three checks only (aligned with futures): return, min_trades, mdd."""
    if not summary:
        return False, ["holdout_no_summary"]
    trades_gate = int(
        min_trades_gate
        if min_trades_gate is not None
        else HOLDOUT_SANITY_GATES["core_min_trades"]
    )
    reasons: List[str] = []
    if float(summary.get("core_avg_ret", 0.0)) <= HOLDOUT_SANITY_GATES["core_min_return"]:
        reasons.append("holdout_core_return_low")
    if int(summary.get("core_min_trades", 0)) < trades_gate:
        reasons.append("holdout_core_trades_low")
    if float(summary.get("core_avg_mdd_abs", 0.0)) > HOLDOUT_SANITY_GATES["core_max_avg_mdd_abs"]:
        reasons.append("holdout_core_mdd_too_high")
    return len(reasons) == 0, reasons


def classify_holdout_outcome(
    summary: Optional[Dict],
    passed: bool,
    reasons: List[str],
    min_trades_gate: int,
) -> Tuple[str, List[str]]:
    """
    Split holdout outcomes into PASS / INACTIVE / FAIL for readability.
    INACTIVE means "no meaningful activity" rather than "logic is broken".
    Deployment policy remains unchanged: only PASS is deployable.
    """
    if passed:
        return "PASS", []
    if not summary:
        return "FAIL", list(reasons)
    core_total_trades = int(summary.get("core_total_trades", summary.get("core_min_trades", 0)))
    if core_total_trades <= 0:
        return "INACTIVE", ["holdout_no_trades"]
    return "FAIL", list(reasons)


def score_holdout_candidate(summary: Optional[Dict]) -> float:
    """
    Score holdout-passed candidates by absolute return quality + robustness.
    Keeps deployment decision from being order-dependent when multiple pass.
    """
    if not summary:
        return -1.0
    core_ret = float(summary.get("core_avg_ret", 0.0))
    core_pf_clip = float(
        summary.get(
            "core_avg_pf_clipped_shrunk",
            summary.get("core_avg_pf_shrunk", summary.get("core_avg_pf_clipped", summary.get("core_avg_pf", 0.0))),
        )
    )
    core_mdd_abs = float(summary.get("core_avg_mdd_abs", 100.0))
    core_total_trades = float(summary.get("core_total_trades", summary.get("core_min_trades", 0)))

    ret_score = _clip01((core_ret + 2.0) / 12.0)
    pf_score = _clip01((core_pf_clip - 1.0) / 0.8)
    mdd_score = 1.0 - _clip01(core_mdd_abs / 15.0)
    activity_score = _clip01(core_total_trades / 12.0)

    return float(
        0.45 * pf_score
        + 0.30 * ret_score
        + 0.15 * mdd_score
        + 0.10 * activity_score
    )


def _load_study_safely(study_name: str, storage: str):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"The distribution is specified by .*",
        )
        return optuna.load_study(study_name=study_name, storage=storage)


def _calculate_mdd_pct_from_pnl(pnl_series: pd.Series, initial_balance: float) -> float:
    if pnl_series is None or len(pnl_series) == 0:
        return 0.0
    equity = float(initial_balance) + pd.Series(pnl_series).cumsum().values
    run_max = np.maximum.accumulate(equity)
    run_max[run_max == 0] = 1e-9
    dd = (equity - run_max) / run_max * 100.0
    return float(np.min(dd)) if len(dd) else 0.0


def _slice_eval_with_warmup(
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    eval_start_ts: pd.Timestamp,
    eval_end_ts: pd.Timestamp,
    warmup_days: int = WARMUP_DAYS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    warmup_start = pd.Timestamp(eval_start_ts) - pd.Timedelta(days=warmup_days)
    eval_end_ts = pd.Timestamp(eval_end_ts)
    test_hourly = hourly_df[(hourly_df["datetime"] >= warmup_start) & (hourly_df["datetime"] <= eval_end_ts)].copy()
    test_daily = daily_df[(daily_df["datetime"] >= warmup_start) & (daily_df["datetime"] <= eval_end_ts)].copy()
    return test_hourly, test_daily


def _filter_trades_for_window(trades_df: pd.DataFrame, eval_start_ts: pd.Timestamp, eval_end_ts: pd.Timestamp) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    eval_start_ts = pd.Timestamp(eval_start_ts)
    eval_end_ts = pd.Timestamp(eval_end_ts)
    has_entry = "entry_time" in df.columns
    has_exit = "exit_time" in df.columns
    if has_entry:
        df["entry_time"] = pd.to_datetime(df["entry_time"])
    if has_exit:
        df["exit_time"] = pd.to_datetime(df["exit_time"])
    if has_entry and has_exit:
        return df[(df["entry_time"] <= eval_end_ts) & (df["exit_time"] >= eval_start_ts)].copy()
    if has_entry:
        return df[(df["entry_time"] >= eval_start_ts) & (df["entry_time"] <= eval_end_ts)].copy()
    if has_exit:
        return df[(df["exit_time"] >= eval_start_ts) & (df["exit_time"] <= eval_end_ts)].copy()
    return pd.DataFrame()


def _calculate_benchmark_return_pct(
    hourly_df: pd.DataFrame,
    eval_start_ts: pd.Timestamp,
    eval_end_ts: pd.Timestamp,
) -> float:
    if hourly_df is None or hourly_df.empty or "close" not in hourly_df.columns:
        return 0.0
    seg = hourly_df[
        (hourly_df["datetime"] >= pd.Timestamp(eval_start_ts))
        & (hourly_df["datetime"] <= pd.Timestamp(eval_end_ts))
    ]
    if seg.empty:
        return 0.0
    start_px = float(pd.to_numeric(seg["close"], errors="coerce").iloc[0])
    end_px = float(pd.to_numeric(seg["close"], errors="coerce").iloc[-1])
    if (not np.isfinite(start_px)) or (not np.isfinite(end_px)) or start_px <= 0.0:
        return 0.0
    return float(((end_px / start_px) - 1.0) * 100.0)


def compute_segment_merge_index(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> np.ndarray:
    hourly_days = pd.to_datetime(hourly_df["datetime"]).dt.normalize().values.astype("datetime64[ns]")
    daily_days = pd.to_datetime(daily_df["datetime"]).dt.normalize().values.astype("datetime64[ns]")
    if len(daily_days) == 0:
        return np.zeros(len(hourly_days), dtype=np.int32)
    pos = np.searchsorted(daily_days, hourly_days, side="right") - 1
    return np.clip(pos, 0, len(daily_days) - 1).astype(np.int32)


def _safe_spot_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(symbol).strip())


def _spot_symbol_to_cache_token(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if "-" in s:
        parts = [p for p in s.split("-") if p]
        if len(parts) == 2:
            quote, base = parts[0], parts[1]
            return f"{base}_{quote}"
    if "/" in s:
        parts = [p for p in s.split("/") if p]
        if len(parts) == 2:
            base, quote = parts[0], parts[1]
            return f"{base}_{quote}"
    return _safe_spot_symbol(s)


def _spot_single_cache_path(symbol: str, timeframe: str) -> str:
    safe_symbol = _spot_symbol_to_cache_token(symbol)
    safe_tf = re.sub(r"[^A-Za-z0-9]+", "_", str(timeframe).strip())
    return os.path.join(DATA_DIR, f"{safe_symbol}_{safe_tf}.parquet")


def _seed_spot_cache_from_legacy(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    safe_symbol = symbol.replace("/", "_")
    pattern_parquet = re.compile(
        rf"^{re.escape(safe_symbol)}_{re.escape(timeframe)}_\d{{8}}_\d{{8}}_spot\.parquet$"
    )
    pattern_csv = re.compile(
        rf"^{re.escape(safe_symbol)}_{re.escape(timeframe)}_\d{{8}}_\d{{8}}_spot\.csv$"
    )
    frames: List[pd.DataFrame] = []
    try:
        for name in os.listdir(DATA_DIR):
            fp = os.path.join(DATA_DIR, name)
            if not os.path.isfile(fp):
                continue
            if pattern_parquet.match(name):
                try:
                    part = pd.read_parquet(fp)
                    if not part.empty:
                        frames.append(part)
                except Exception:
                    continue
            elif pattern_csv.match(name):
                try:
                    part = pd.read_csv(fp)
                    if not part.empty:
                        frames.append(part)
                except Exception:
                    continue
    except Exception:
        return None

    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    if "timestamp" not in merged.columns:
        return None
    merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    merged["datetime"] = pd.to_datetime(merged["timestamp"], unit="ms")
    return merged


def compute_segment_merge_index(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> np.ndarray:
    hourly_days = pd.to_datetime(hourly_df["datetime"]).dt.normalize().values.astype("datetime64[ns]")
    daily_days = pd.to_datetime(daily_df["datetime"]).dt.normalize().values.astype("datetime64[ns]")
    if len(daily_days) == 0:
        return np.zeros(len(hourly_days), dtype=np.int32)
    pos = np.searchsorted(daily_days, hourly_days, side="right") - 1
    return np.clip(pos, 0, len(daily_days) - 1).astype(np.int32)


def _spot_timeframe_to_ms(timeframe: str) -> Optional[int]:
    tf = str(timeframe).strip().lower()
    fixed = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "10m": 600_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
        "1w": 604_800_000,
    }
    if tf in fixed:
        return fixed[tf]
    m = re.fullmatch(r"(\d+)m", tf)
    if m:
        return int(m.group(1)) * 60_000
    h = re.fullmatch(r"(\d+)h", tf)
    if h:
        return int(h.group(1)) * 3_600_000
    return None


def _find_internal_gap_ranges(df: pd.DataFrame, start_ts: int, end_ts: int, step_ms: int) -> List[Tuple[int, int]]:
    if df is None or df.empty or "timestamp" not in df.columns or step_ms <= 0:
        return []
    ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna().astype("int64")
    ts = ts[(ts >= int(start_ts)) & (ts <= int(end_ts))]
    if ts.empty:
        return []
    arr = np.sort(ts.unique())
    if arr.size < 2:
        return []
    diffs = np.diff(arr)
    gap_idx = np.where(diffs > step_ms)[0]
    ranges: List[Tuple[int, int]] = []
    for idx in gap_idx:
        gap_start = int(arr[idx] + step_ms)
        gap_end = int(arr[idx + 1] - step_ms)
        if gap_start <= gap_end:
            ranges.append((gap_start, gap_end))
    return ranges


def _fill_internal_gaps_spot(
    client: UpbitClient,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    start_ts: int,
    end_ts: int,
    max_ranges: int = GAP_FILL_MAX_RANGES,
) -> pd.DataFrame:
    step_ms = _spot_timeframe_to_ms(timeframe)
    if step_ms is None or df is None or df.empty:
        return df
    gaps = _find_internal_gap_ranges(df, start_ts, end_ts, step_ms)
    if not gaps:
        return df

    selected = gaps[: max(1, int(max_ranges))]
    logger.info(
        f"[INFO] Detected {len(gaps)} internal gap(s) for {symbol}-{timeframe}; "
        f"filling {len(selected)} range(s)."
    )
    parts: List[pd.DataFrame] = [df]
    added_rows = 0
    for gap_start, gap_end in selected:
        fetched = client.fetch_ohlcv(symbol, timeframe, since=int(gap_start), end=int(gap_end))
        if fetched is None or fetched.empty:
            continue
        fetched = fetched.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        fetched["datetime"] = pd.to_datetime(fetched["timestamp"], unit="ms")
        added_rows += len(fetched)
        parts.append(fetched)
    if len(parts) == 1:
        return df
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    logger.info(f"[INFO] Gap fill merged {added_rows} row(s) for {symbol}-{timeframe}.")
    return merged


def load_data_spot(symbol: str, timeframe: str, start_date: str, end_date: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    cache_key = (str(symbol), str(timeframe), str(start_date), str(end_date))
    if cache_key in _SPOT_DATA_CACHE:
        h, d = _SPOT_DATA_CACHE[cache_key]
        return h.copy(deep=False), d.copy(deep=False)

    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
    start_dt = pd.Timestamp(start_date)
    end_dt_inclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)

    results: Dict[str, Optional[pd.DataFrame]] = {"1d": None, timeframe: None}
    client: Optional[UpbitClient] = None

    for tf in ("1d", timeframe):
        parquet_fp = _spot_single_cache_path(symbol, tf)
        df: Optional[pd.DataFrame] = None

        if os.path.exists(parquet_fp):
            try:
                df = pd.read_parquet(parquet_fp)
                if "timestamp" in df.columns:
                    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                else:
                    df = None
            except Exception:
                df = None

        if df is None or df.empty:
            df = _seed_spot_cache_from_legacy(symbol, tf)
            if df is not None and not df.empty:
                try:
                    df.to_parquet(parquet_fp, index=False)
                except Exception:
                    pass

        need_fetch = False
        fetch_since_ts = start_ts
        if df is None or df.empty:
            need_fetch = True
        else:
            cached_start = int(df["timestamp"].min())
            cached_end = int(df["timestamp"].max())
            # Spot symbols can have listing dates later than requested start.
            # Do not refetch backward-only gaps repeatedly; fetch only when right edge is missing.
            if cached_end < end_ts:
                need_fetch = True
                # Incremental fetch from the current cache tail (with 1ms overlap-safe boundary).
                fetch_since_ts = max(start_ts, cached_end + 1)

        if need_fetch:
            if client is None:
                access = os.getenv("UPBIT_ACCESS_KEY")
                secret = os.getenv("UPBIT_SECRET_KEY")
                if not access or not secret:
                    print(f"[ERROR] Missing Upbit API keys while cache missing: {symbol}-{tf}")
                    return None, None
                client = UpbitClient(access, secret)

            logger.info(f"[INFO] Downloading {symbol}-{tf}...")
            fetched = client.fetch_ohlcv(symbol, tf, since=fetch_since_ts, end=end_ts)
            if fetched is None or fetched.empty:
                print(f"[ERROR] Failed to fetch {symbol}-{tf}")
                return None, None
            fetched = fetched.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            fetched["datetime"] = pd.to_datetime(fetched["timestamp"], unit="ms")

            if df is not None and not df.empty:
                df = pd.concat([df, fetched], ignore_index=True)
                df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            else:
                df = fetched

            try:
                df.to_parquet(parquet_fp, index=False)
            except Exception:
                try:
                    fallback_csv = parquet_fp.replace(".parquet", ".csv")
                    df.to_csv(fallback_csv, index=False)
                except Exception:
                    pass

        # Internal gap backfill (middle missing candles): run once per process/symbol/tf.
        gap_fill_key = (str(symbol), str(tf))
        if df is not None and not df.empty and gap_fill_key not in _SPOT_GAP_FILLED_KEYS:
            step_ms = _spot_timeframe_to_ms(tf)
            if step_ms is not None:
                gaps = _find_internal_gap_ranges(df, start_ts, end_ts, step_ms)
                if gaps:
                    if client is None:
                        access = os.getenv("UPBIT_ACCESS_KEY")
                        secret = os.getenv("UPBIT_SECRET_KEY")
                        if access and secret:
                            client = UpbitClient(access, secret)
                        else:
                            logger.warning(
                                f"[WARN] Internal gaps detected for {symbol}-{tf} but API keys are missing; "
                                "skipping gap backfill."
                            )
                    if client is not None:
                        df = _fill_internal_gaps_spot(
                            client=client,
                            symbol=symbol,
                            timeframe=tf,
                            df=df,
                            start_ts=start_ts,
                            end_ts=end_ts,
                        )
                        try:
                            df.to_parquet(parquet_fp, index=False)
                        except Exception:
                            pass
                _SPOT_GAP_FILLED_KEYS.add(gap_fill_key)

        if df is None or df.empty:
            return None, None

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt_inclusive)].copy()
        df.reset_index(drop=True, inplace=True)
        results[tf] = df

    hourly = results.get(timeframe)
    daily = results.get("1d")
    if hourly is not None and daily is not None:
        _SPOT_DATA_CACHE[cache_key] = (hourly.copy(deep=False), daily.copy(deep=False))
    return hourly, daily


def load_best_params_from_mysql(
    mode: str,
    storage_url: str,
) -> Tuple[Optional[str], Optional[Dict], Optional[float]]:
    base_name = f"spot_{mode.lower()}_strategy"
    target_candidates = [base_name]

    def _norm(s: str) -> str:
        return str(s).strip().lower()

    def _read(study_name: str):
        try:
            st = _load_study_safely(study_name=study_name, storage=storage_url)
            return st, st.best_params, st.best_value
        except KeyError:
            return None, None, None
        except ValueError:
            return None, None, None
        except Exception:
            return None, None, None

    # 1) Exact name sequence
    for target_name in target_candidates:
        st, params, val = _read(target_name)
        if st is not None:
            return target_name, params, val

    # 2) Normalized / fuzzy fallback
    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage_url)
    except Exception:
        return None, None, None

    if not summaries:
        return None, None, None

    by_norm = {_norm(s.study_name): s.study_name for s in summaries}
    for target_name in target_candidates:
        resolved = by_norm.get(_norm(target_name))
        if resolved:
            st, params, val = _read(resolved)
            if st is not None:
                return resolved, params, val

    needle = f"spot_{mode.lower()}"
    candidates = []
    for s in summaries:
        name_norm = _norm(s.study_name)
        if needle not in name_norm:
            continue
        if "__s1_" in name_norm or "__s2_" in name_norm:
            continue
        candidates.append(s)
    if candidates:
        candidates.sort(key=lambda x: int(x.n_trials or 0), reverse=True)
        for c in candidates:
            st, params, val = _read(c.study_name)
            if st is None:
                continue
            logger.warning(
                f"Study name fallback used: expected one of {target_candidates}, resolved '{c.study_name}'"
            )
            return c.study_name, params, val

    return None, None, None


def _diagnose_zero_trade_window_spot(
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    params: Dict,
    eval_start_ts: pd.Timestamp,
    eval_end_ts: pd.Timestamp,
) -> None:
    if not SHOW_ZERO_TRADE_DIAG:
        return
    try:
        strategy = UltimateStrategy("Diag_Spot", params)
        ddf = strategy.generate_signals(daily_df.copy())
        hdf = strategy.generate_signals(hourly_df.copy())

        if "date_key" not in ddf.columns:
            ddf["date_key"] = pd.to_datetime(ddf["datetime"]).dt.strftime("%Y-%m-%d")
        if "date_key" not in hdf.columns:
            hdf["date_key"] = pd.to_datetime(hdf["datetime"]).dt.strftime("%Y-%m-%d")

        exclude_cols = {"date_key", "datetime", "date", "open", "high", "low", "close", "volume"}
        daily_cols = [c for c in ddf.columns if c not in exclude_cols]
        shifted_daily = ddf[daily_cols].shift(1)
        shifted_daily.columns = [f"daily_{c}" for c in daily_cols]
        shifted_daily["date_key"] = ddf["date_key"]
        merged = pd.merge(hdf, shifted_daily, on="date_key", how="left")

        if "daily_trend_direction" in merged.columns:
            h_trend = merged["trend_direction"].fillna(0).values
            d_trend = merged["daily_trend_direction"].fillna(0).values
            trend_gate_mode = str(params.get("TREND_GATE_MODE", "STRICT")).strip().upper()
            if trend_gate_mode == "OFF":
                merged["trend_direction"] = np.where(h_trend == 1, 1, 0)
            elif trend_gate_mode == "SOFT":
                merged["trend_direction"] = np.where((h_trend == 1) | (d_trend == 1), 1, 0)
            else:
                merged["trend_direction"] = np.where((h_trend == 1) & (d_trend == 1), 1, 0)

        for col in [
            "entry_upper",
            "trend_direction",
            "strength_filter",
            "volume_ratio",
            "atr",
            "parabolic_sar",
            "hurst",
            "natr",
            "rsi",
        ]:
            if col in merged.columns:
                merged[col] = merged[col].shift(1)

        merged["datetime"] = pd.to_datetime(merged["datetime"])
        eval_df = merged[
            (merged["datetime"] >= pd.Timestamp(eval_start_ts))
            & (merged["datetime"] <= pd.Timestamp(eval_end_ts))
        ].copy()
        if eval_df.empty:
            print("   [DIAG] No bars in eval window (data coverage issue).")
            print(f"   [DIAG] merged range: {merged['datetime'].min()} ~ {merged['datetime'].max()}")
            return

        close = eval_df["close"].astype(float)
        entry_upper = eval_df.get("entry_upper", pd.Series(np.nan, index=eval_df.index)).astype(float)
        trend_dir = eval_df.get("trend_direction", pd.Series(0, index=eval_df.index)).fillna(0).astype(int)
        strength = eval_df.get("strength_filter", pd.Series(0, index=eval_df.index)).fillna(0).astype(int)
        volume_ratio = eval_df.get("volume_ratio", pd.Series(np.nan, index=eval_df.index)).astype(float)
        rsi = eval_df.get("rsi", pd.Series(np.nan, index=eval_df.index)).astype(float)
        natr = eval_df.get("natr", pd.Series(np.nan, index=eval_df.index)).astype(float)

        use_volume_filter = bool(params.get("USE_VOLUME_FILTER", False))
        vol_threshold = float(params.get("VOLUME_THRESHOLD_MULT", 1.0))
        rsi_entry_max_raw = params.get("RSI_ENTRY_MAX", 100.0)
        rsi_entry_max = 100.0 if rsi_entry_max_raw is None else float(rsi_entry_max_raw)
        natr_entry_min = float(params.get("NATR_ENTRY_MIN", 0.0))

        total = len(eval_df)
        has_upper = ~entry_upper.isna()
        pass_strength = strength != 0
        pass_volume = (volume_ratio >= vol_threshold) if use_volume_filter else pd.Series(True, index=eval_df.index)
        pass_rsi = rsi < rsi_entry_max
        pass_natr = natr >= natr_entry_min
        pass_trend = trend_dir == 1
        pass_breakout = close > entry_upper

        valid = has_upper & pass_strength & pass_volume & pass_rsi & pass_natr
        signal = valid & pass_trend & pass_breakout

        print("   [DIAG] Zero-trade diagnostics (spot)")
        print(f"   [DIAG] Bars in eval window: {total} ({eval_df['datetime'].min()} ~ {eval_df['datetime'].max()})")
        print(f"   [DIAG] After entry-upper check: {int(has_upper.sum())}/{total}")
        print(f"   [DIAG] After strength filter: {int((has_upper & pass_strength).sum())}/{total}")
        if use_volume_filter:
            print(
                f"   [DIAG] After volume filter: "
                f"{int((has_upper & pass_strength & pass_volume).sum())}/{total} "
                f"(threshold={vol_threshold:.4f})"
            )
        else:
            print("   [DIAG] Volume filter disabled")
        print(f"   [DIAG] After RSI/NATR filters: {int(valid.sum())}/{total}")
        print(f"   [DIAG] Trend-up bars: {int(pass_trend.sum())}/{total}")
        print(f"   [DIAG] Breakout bars (pre-trend): {int((valid & pass_breakout).sum())}/{total}")
        print(f"   [DIAG] Final signal candidates: {int(signal.sum())}/{total}")
    except Exception as e:
        print(f"   [DIAG] failed to run zero-trade diagnostics: {e}")


def verify_single_symbol_spot(
    symbol: str,
    best_params: Dict,
    primary_symbols: List[str],
    eval_start_time: Optional[pd.Timestamp] = None,
    eval_end_time: Optional[pd.Timestamp] = None,
    run_robustness_checks: bool = True,
    cost_mult: float = 1.0,
) -> Optional[Dict]:
    tf = best_params.get("TIMEFRAME", "1h")
    hourly_df, daily_df = load_data_spot(symbol, tf, SPOT_BACKTEST_START_DATE, SPOT_BACKTEST_END_DATE)
    if hourly_df is None or daily_df is None:
        print(f"[WARN] Data missing for {symbol}. Skipping.")
        return None

    eval_start_ts, eval_end_ts = resolve_eval_window(eval_start_time, eval_end_time)
    test_hourly, test_daily = _slice_eval_with_warmup(hourly_df, daily_df, eval_start_ts, eval_end_ts)
    if test_hourly.empty:
        print(f"[WARN] No data for {symbol} in eval window. Skipping.")
        return None

    safe_cost_mult = max(0.0, float(cost_mult))
    strategy = UltimateStrategy(f"Verify_{symbol}", best_params)
    
    # [CRITICAL] Generate indicators before passing to engine
    test_daily = strategy.generate_signals(test_daily)
    test_hourly = strategy.generate_signals(test_hourly)

    current_merge = compute_segment_merge_index(test_hourly, test_daily)
    engine = BacktestEngineFastSpot(
        test_hourly,
        test_daily,
        strategy,
        backtest_loop_spot_numba,
        initial_balance=SPOT_INITIAL_BALANCE,
        fee_rate=SPOT_BASE_FEE * safe_cost_mult,
        slippage_rate=SPOT_BASE_SLIPPAGE * safe_cost_mult,
        merge_index_map=current_merge,
    )
    engine.risk_per_trade = best_params.get("RISK_PER_TRADE_SPOT", 0.99)
    res = engine.run()

    oos_trades = _filter_trades_for_window(res.get("trades_df", pd.DataFrame()), eval_start_ts, eval_end_ts)
    all_trades_count = len(res.get("trades_df", pd.DataFrame()))
    trade_count = len(oos_trades)
    if trade_count > 0:
        total_pnl = float(oos_trades["pnl"].sum())
        ret_pct = (total_pnl / SPOT_INITIAL_BALANCE) * 100.0
        mdd = _calculate_mdd_pct_from_pnl(oos_trades["pnl"], SPOT_INITIAL_BALANCE)
        win_rate = float(np.mean(oos_trades["pnl"] > 0) * 100.0)
        gross_profit = float(oos_trades[oos_trades["pnl"] > 0]["pnl"].sum())
        gross_loss = abs(float(oos_trades[oos_trades["pnl"] < 0]["pnl"].sum()))
        pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
    else:
        ret_pct = 0.0
        mdd = 0.0
        win_rate = 0.0
        pf = 0.0
        if SHOW_ZERO_TRADE_DIAG and all_trades_count > 0:
            print(
                f"   [DIAG] Trades exist outside eval window: total={all_trades_count}, "
                f"in-window={trade_count}."
            )
        _diagnose_zero_trade_window_spot(
            hourly_df=test_hourly,
            daily_df=test_daily,
            params=best_params,
            eval_start_ts=eval_start_ts,
            eval_end_ts=eval_end_ts,
        )

    benchmark_ret_pct = _calculate_benchmark_return_pct(test_hourly, eval_start_ts, eval_end_ts)
    excess_ret_pct = float(ret_pct - benchmark_ret_pct)

    is_primary = symbol in primary_symbols
    indicator = "PRIMARY" if is_primary else "REFERENCE"
    cost_tag = f" x{safe_cost_mult:.1f}" if abs(safe_cost_mult - 1.0) > 1e-9 else ""
    print(
        f"   - {symbol} [{indicator}{cost_tag}]: Return {ret_pct:.2f}% | "
        f"MDD {mdd:.2f}% | Trades {trade_count} | Win {win_rate:.1f}% | PF {pf:.2f}"
    )

    result = {
        "symbol": symbol,
        "timeframe": str(tf),
        "return": ret_pct,
        "mdd": mdd,
        "trades": trade_count,
        "win_rate": win_rate,
        "pf": pf,
        "bh_return": float(benchmark_ret_pct),
        "excess_return": float(excess_ret_pct),
        "is_primary": is_primary,
        "wfa_results": None,
        "mc_results": None,
        "trades_log": oos_trades["pnl_pct"].tolist() if not oos_trades.empty and "pnl_pct" in oos_trades.columns else [],
        "eval_start": str(eval_start_ts),
        "eval_end": str(eval_end_ts),
    }

    if run_robustness_checks and trade_count >= 10:
        print(f"      Running detailed analysis for {symbol}...")
        try:
            wfa = SpotWalkForwardAnalyzer(test_hourly, test_daily, best_params)
            wfa_results = wfa.run(n_splits=5)
            if not wfa_results.empty:
                avg_wfa_ret = float(wfa_results["Return"].mean())
                consistency = (len(wfa_results[wfa_results["Return"] > 0]) / len(wfa_results)) * 100.0
                result["wfa_results"] = {
                    "avg_return": avg_wfa_ret,
                    "consistency": consistency,
                    "splits": len(wfa_results),
                }
                print(f"         WFA: Avg {avg_wfa_ret:.1f}% | Consistency {consistency:.0f}%")
        except Exception as e:
            logger.warning(f"WFA failed for {symbol}: {e}")

        try:
            mc = SpotMonteCarloSimulator(result["trades_log"])
            mc_res = mc.run(n_simulations=10000, initial_balance=SPOT_INITIAL_BALANCE)
            result["mc_results"] = {
                "prob_profit": mc_res["prob_profit"],
                "mean_return": mc_res["mean_return_pct"],
                "worst_mdd_95": mc_res["worst_case_mdd"],
                "lower_bound_95": mc_res.get("lower_bound_95", 0.0),
            }
            print(
                f"         MC: Profit Prob {mc_res['prob_profit']:.1f}% "
                f"| Worst MDD(95%) {mc_res['worst_case_mdd']:.1f}%"
            )
        except Exception as e:
            logger.warning(f"MC failed for {symbol}: {e}")

    return result


def calculate_mode_performance(all_results: List[Dict]) -> Optional[float]:
    primary_results = [r for r in all_results if r["is_primary"]]
    if not primary_results:
        return None
    avg_ret = float(np.mean([r["return"] for r in primary_results]))
    print(f"\n   [Summary]")
    print(f"   - PRIMARY Avg Return (BTC/ETH): {avg_ret:.2f}%")
    ref_results = [r for r in all_results if not r["is_primary"]]
    if ref_results:
        ref_avg = float(np.mean([r["return"] for r in ref_results]))
        print(f"   - REFERENCE Avg Return (Alts): {ref_avg:.2f}%")
    primary_wfa = [r for r in primary_results if r.get("wfa_results")]
    primary_mc = [r for r in primary_results if r.get("mc_results")]
    if primary_wfa:
        print(f"\n   [WFA] PRIMARY")
        for r in primary_wfa:
            wfa = r["wfa_results"]
            print(
                f"      {r['symbol']}: Avg {wfa['avg_return']:.1f}% | "
                f"Consistency {wfa['consistency']:.0f}% ({wfa['splits']} splits)"
            )
    if primary_mc:
        print(f"\n   [MC] PRIMARY")
        for r in primary_mc:
            mc = r["mc_results"]
            print(
                f"      {r['symbol']}: Profit Prob {mc['prob_profit']:.1f}% | "
                f"Worst MDD(95%) {mc['worst_mdd_95']:.1f}%"
            )
    ref_wfa = [r for r in ref_results if r.get("wfa_results")]
    ref_mc = [r for r in ref_results if r.get("mc_results")]
    if ref_wfa:
        print(f"\n   [WFA] REFERENCE")
        for r in ref_wfa:
            wfa = r["wfa_results"]
            print(
                f"      {r['symbol']}: Avg {wfa['avg_return']:.1f}% | "
                f"Consistency {wfa['consistency']:.0f}% ({wfa['splits']} splits)"
            )
    if ref_mc:
        print(f"\n   [MC] REFERENCE")
        for r in ref_mc:
            mc = r["mc_results"]
            print(
                f"      {r['symbol']}: Profit Prob {mc['prob_profit']:.1f}% | "
                f"Worst MDD(95%) {mc['worst_mdd_95']:.1f}%"
            )
    return avg_ret


def summarize_profile(results: List[Dict]) -> Optional[Dict]:
    if not results:
        return None
    primary = [r for r in results if r.get("is_primary")]
    if not primary:
        return None

    core_returns = np.array([float(r["return"]) for r in primary], dtype=np.float64)
    core_bh_returns = np.array([float(r.get("bh_return", 0.0)) for r in primary], dtype=np.float64)
    core_excess_returns = np.array([float(r.get("excess_return", 0.0)) for r in primary], dtype=np.float64)
    core_mdd_abs = np.array([abs(float(r["mdd"])) for r in primary], dtype=np.float64)
    core_pf = np.array([float(r["pf"]) for r in primary], dtype=np.float64)
    pf_clip = float(SELECTION_POLICY.get("robust", {}).get("pf_clip_per_symbol", 6.0))
    core_pf_clipped = np.clip(core_pf, 0.0, pf_clip)
    core_trades = np.array([int(r["trades"]) for r in primary], dtype=np.int64)
    core_total_trades = int(np.sum(core_trades))

    wfa_cons = [r["wfa_results"]["consistency"] for r in primary if r.get("wfa_results")]
    mc_worst_abs = [abs(float(r["mc_results"]["worst_mdd_95"])) for r in primary if r.get("mc_results")]
    core_wfa_consistency = float(np.mean(wfa_cons)) if wfa_cons else 0.0
    core_mc_worst_mdd_abs = float(np.mean(mc_worst_abs)) if mc_worst_abs else 0.0

    alts = [r for r in results if not r.get("is_primary")]
    alt_returns = np.array([float(r["return"]) for r in alts], dtype=np.float64) if alts else np.array([], dtype=np.float64)
    alt_mdd_abs = np.array([abs(float(r["mdd"])) for r in alts], dtype=np.float64) if alts else np.array([], dtype=np.float64)
    alt_median_ret = float(np.median(alt_returns)) if alt_returns.size else 0.0
    alt_p25_ret = float(np.percentile(alt_returns, 25)) if alt_returns.size else 0.0
    alt_pos_rate = float(np.mean(alt_returns > 0)) if alt_returns.size else 0.0
    alt_worst_mdd = float(np.max(alt_mdd_abs)) if alt_mdd_abs.size else 0.0

    all_returns = np.array([float(r["return"]) for r in results], dtype=np.float64)
    mean_abs = max(abs(float(np.mean(all_returns))), 1e-9)
    dispersion = float(np.std(all_returns) / mean_abs)
    eval_starts = [pd.Timestamp(r["eval_start"]) for r in primary if r.get("eval_start")]
    eval_ends = [pd.Timestamp(r["eval_end"]) for r in primary if r.get("eval_end")]
    eval_days = 0
    if eval_starts and eval_ends:
        eval_days = max(1, int((max(eval_ends).normalize() - min(eval_starts).normalize()).days) + 1)
    core_avg_trades = float(np.mean(core_trades))
    core_trades_per_30d = (core_avg_trades / float(eval_days)) * 30.0 if eval_days > 0 else 0.0
    core_avg_pf_raw = float(np.mean(core_pf))
    core_avg_pf_clipped = float(np.mean(core_pf_clipped))
    core_avg_pf_shrunk = _shrink_pf_by_trades(core_avg_pf_raw, core_total_trades)
    core_avg_pf_clipped_shrunk = _shrink_pf_by_trades(core_avg_pf_clipped, core_total_trades)
    core_reliability = _clip01(min(1.0, core_total_trades / 60.0) * min(1.0, float(np.min(core_trades)) / 12.0))

    gates = SELECTION_POLICY["gates"]
    primary_timeframes = [str(r.get("timeframe", "")).strip().lower() for r in primary if r.get("timeframe")]
    core_timeframe = max(set(primary_timeframes), key=primary_timeframes.count) if primary_timeframes else "default"
    gate_core_min_trades = spot_selection_min_trades_for_timeframe(core_timeframe)
    core_avg_bh_ret = float(np.mean(core_bh_returns))
    core_avg_excess_ret = float(np.mean(core_excess_returns))
    core_up_excess = core_excess_returns[core_bh_returns >= 5.0]
    core_down_excess = core_excess_returns[core_bh_returns <= -5.0]
    core_up_excess_ret = float(np.mean(core_up_excess)) if core_up_excess.size else 0.0
    core_down_excess_ret = float(np.mean(core_down_excess)) if core_down_excess.size else 0.0
    core_up_sample_count = int(core_up_excess.size)
    core_down_sample_count = int(core_down_excess.size)

    # Hard gates (aligned with futures): return, trades, mdd, wfa, mc, alt 3.
    reasons: List[str] = []
    if np.any(core_returns <= gates["core_min_return"]):
        reasons.append("core_negative_return")
    if int(np.min(core_trades)) < int(gate_core_min_trades):
        reasons.append("core_trade_count_low")
    if float(np.mean(core_mdd_abs)) > gates["core_max_avg_mdd_abs"]:
        reasons.append("core_mdd_too_high")
    if core_wfa_consistency < gates["core_min_wfa_consistency"]:
        reasons.append("core_wfa_low")
    if core_mc_worst_mdd_abs > gates["core_max_mc_worst_mdd_95_abs"]:
        reasons.append("core_mc_mdd_too_high")
    if alt_returns.size > 0:
        if alt_pos_rate < gates["alt_min_pos_rate"]:
            reasons.append("alt_pos_rate_low")
        if alt_worst_mdd > gates["alt_max_worst_mdd_abs"]:
            reasons.append("alt_worst_mdd_too_high")
        if alt_p25_ret < gates["alt_min_p25_return"]:
            reasons.append("alt_tail_return_too_low")

    return {
        "policy_version": SELECTION_POLICY_VERSION,
        "core_avg_ret": float(np.mean(core_returns)),
        "core_avg_bh_ret": core_avg_bh_ret,
        "core_avg_excess_ret": core_avg_excess_ret,
        "core_avg_mdd_abs": float(np.mean(core_mdd_abs)),
        "core_avg_pf": float(core_avg_pf_raw),
        "core_avg_pf_shrunk": float(core_avg_pf_shrunk),
        "core_avg_pf_clipped": float(core_avg_pf_clipped),
        "core_avg_pf_clipped_shrunk": float(core_avg_pf_clipped_shrunk),
        "core_timeframe": str(core_timeframe),
        "core_min_trades": int(np.min(core_trades)),
        "gate_core_min_trades": int(gate_core_min_trades),
        "core_total_trades": int(core_total_trades),
        "core_avg_trades": core_avg_trades,
        "core_trades_per_30d": float(core_trades_per_30d),
        "core_reliability": float(core_reliability),
        "core_up_excess_ret": float(core_up_excess_ret),
        "core_down_excess_ret": float(core_down_excess_ret),
        "core_up_sample_count": int(core_up_sample_count),
        "core_down_sample_count": int(core_down_sample_count),
        "eval_days": int(eval_days),
        "core_wfa_consistency": core_wfa_consistency,
        "core_mc_worst_mdd_95": core_mc_worst_mdd_abs,
        "alt_median_ret": alt_median_ret,
        "alt_p25_ret": alt_p25_ret,
        "alt_pos_rate": alt_pos_rate,
        "alt_worst_mdd_abs": alt_worst_mdd,
        "dispersion": dispersion,
        "gates_passed": len(reasons) == 0,
        "gate_reasons": reasons,
    }


def _simple_profile_score(summary: Dict, gate_failed_penalty: float = 0.0) -> float:
    """
    Multi-axis weighted score: core (70%) + alt (25%) + diversification (5%).
    Core-first so selection favors return over cross-symbol uniformity.
    WFA/MC sub-scores default to neutral (0.5) when data is absent to avoid penalizing
    strategies that ran without alt symbols or with too few trades for robustness checks.
    """
    w = SELECTION_POLICY["weights"]

    # --- Core axis ---
    core_ret = float(summary.get("core_avg_ret", 0.0))
    core_pf_clip = float(
        summary.get(
            "core_avg_pf_clipped_shrunk",
            summary.get("core_avg_pf_shrunk", summary.get("core_avg_pf_clipped", summary.get("core_avg_pf", 0.0))),
        )
    )
    core_mdd_abs = float(summary.get("core_avg_mdd_abs", 100.0))
    core_wfa = float(summary.get("core_wfa_consistency", 0.0))
    core_mc_mdd = float(summary.get("core_mc_worst_mdd_95", 0.0))
    # WFA/MC: use neutral 0.5 when not available (core_wfa_consistency == 0 means no WFA ran)
    core_wfa_score = _clip01(core_wfa / 100.0) if core_wfa > 0.0 else 0.5
    core_mc_score = (1.0 - _clip01(core_mc_mdd / 40.0)) if core_mc_mdd > 0.0 else 0.5
    core_ret_score = _log_scaled_positive(core_ret, cap=50.0)
    core_pf_score = _clip01((core_pf_clip - 1.0) / 1.5)
    core_mdd_score = 1.0 - _clip01(core_mdd_abs / 20.0)

    core_score = (
        w["core_return"] * core_ret_score
        + w["core_pf"] * core_pf_score
        + w["core_wfa"] * core_wfa_score
        + w["core_mdd"] * core_mdd_score
        + w["core_mc"] * core_mc_score
    )

    # --- Alt axis (neutral 0.5 when no alt symbols present) ---
    alt_median = float(summary.get("alt_median_ret", 0.0))
    alt_p25 = float(summary.get("alt_p25_ret", 0.0))
    alt_pos = float(summary.get("alt_pos_rate", 0.0))
    alt_mdd = float(summary.get("alt_worst_mdd_abs", 0.0))
    has_alts = bool(summary.get("alt_pos_rate", None) is not None and alt_pos > 0.0 or alt_median != 0.0)
    if has_alts:
        alt_median_score = _log_scaled_positive(alt_median, cap=30.0)
        alt_p25_score = _clip01((alt_p25 + 15.0) / 40.0)
        alt_pos_score = _clip01(alt_pos)
        alt_mdd_score = 1.0 - _clip01(alt_mdd / 35.0)
        alt_score = (
            w["alt_median"] * alt_median_score
            + w["alt_p25"] * alt_p25_score
            + w["alt_pos"] * alt_pos_score
            + w["alt_mdd"] * alt_mdd_score
        )
    else:
        alt_score = 0.5

    # --- Diversification axis ---
    dispersion = float(summary.get("dispersion", 0.0))
    div_score = 1.0 - _clip01(dispersion / 3.0)

    score = (
        w["core_total"] * core_score
        + w["alt_total"] * alt_score
        + w["div_total"] * div_score
    )

    # Trade-count soft penalty: prefer richer sample sizes without hard-blocking low-trade strategies
    min_trades = int(summary.get("core_min_trades", 0))
    trade_penalty = float(1.0 / (1.0 + np.exp(-0.2 * (float(min_trades) - 15.0))))
    score = float(score * trade_penalty)

    if not summary.get("gates_passed", False):
        score -= float(gate_failed_penalty)
    return float(score)


def rank_profiles(profile_summaries: Dict[str, Dict]) -> List[Tuple[str, float, Dict]]:
    ranked: List[Tuple[str, float, Dict]] = []
    for key, s in profile_summaries.items():
        if not s or not s.get("gates_passed", False):
            continue
        score = _simple_profile_score(s)
        ranked.append((key, score, s))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def rank_profiles_soft(profile_summaries: Dict[str, Dict]) -> List[Tuple[str, float, Dict]]:
    """
    Soft ranking for fallback holdout checks:
    include gate-failed profiles with penalty instead of dropping them.
    """
    ranked: List[Tuple[str, float, Dict]] = []
    for key, s in profile_summaries.items():
        if not s:
            continue
        base_score = _simple_profile_score(s, gate_failed_penalty=0.10)
        ranked.append((key, base_score, s))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

def build_rolling_oos_windows(
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    window_days: int = 120,
    step_days: int = 30,
    max_windows: int = 6,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
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
    best_params: Dict,
    symbols: List[str],
    primary_symbols: List[str],
    eval_start_time: pd.Timestamp,
    eval_end_time: pd.Timestamp,
    window_days: int = 120,
    step_days: int = 30,
    max_windows: int = 6,
) -> List[Dict]:
    windows = build_rolling_oos_windows(eval_start_time, eval_end_time, window_days, step_days, max_windows)
    if not windows:
        print("[WARN] Rolling OOS skipped: no valid windows.")
        return []

    print("\n" + "-" * LOG_WIDTH)
    print(
        "Rolling OOS Verification "
        f"(window_days={window_days}, step_days={step_days}, max_windows={max_windows})"
    )
    print("-" * LOG_WIDTH)
    roll_summaries: List[Dict] = []
    for idx, (w_start, w_end) in enumerate(windows, start=1):
        print(f"[ROLL-{idx}] Window: {w_start} ~ {w_end}")
        roll_results: List[Dict] = []
        for symbol in symbols:
            r = verify_single_symbol_spot(
                symbol,
                best_params,
                primary_symbols,
                eval_start_time=w_start,
                eval_end_time=w_end,
                run_robustness_checks=False,
            )
            if r:
                roll_results.append(r)
        calculate_mode_performance(roll_results)
        s = summarize_profile(roll_results)
        if not s:
            print(f"[ROLL-{idx}] Summary unavailable.")
            continue
        print(
            f"[ROLL-{idx}] Summary: core_ret={s['core_avg_ret']:.2f}% | "
            f"core_mdd={s['core_avg_mdd_abs']:.2f}% | "
            f"core_pf={s['core_avg_pf']:.2f} | "
            f"core_min_trades={s['core_min_trades']}"
        )
        s["window_start"] = str(w_start)
        s["window_end"] = str(w_end)
        roll_summaries.append(s)

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


def evaluate_rolling_sanity(roll_summaries: List[Dict]) -> Tuple[bool, List[str], Dict[str, float]]:
    if not roll_summaries:
        return False, ["rolling_no_summary"], {}
    core_ret = np.array([float(s.get("core_avg_ret", 0.0)) for s in roll_summaries], dtype=np.float64)
    core_pf = np.array([float(s.get("core_avg_pf", 0.0)) for s in roll_summaries], dtype=np.float64)
    core_mdd = np.array([float(s.get("core_avg_mdd_abs", 0.0)) for s in roll_summaries], dtype=np.float64)
    windows = int(len(roll_summaries))
    pf_pass_rate = float(np.mean(core_pf >= 1.0))
    ret_pass_rate = float(np.mean(core_ret >= 0.0))
    median_pf = float(np.median(core_pf))
    median_ret = float(np.median(core_ret))
    worst_mdd_abs = float(np.max(core_mdd))

    g = ROLLING_SANITY_GATES
    reasons: List[str] = []
    if windows < int(g["min_windows"]):
        reasons.append("rolling_windows_low")
    if pf_pass_rate < float(g["min_pf_pass_rate"]):
        reasons.append("rolling_pf_pass_rate_low")
    if ret_pass_rate < float(g["min_ret_pass_rate"]):
        reasons.append("rolling_ret_pass_rate_low")
    if median_pf < float(g["min_median_pf"]):
        reasons.append("rolling_median_pf_low")
    if median_ret < float(g["min_median_ret"]):
        reasons.append("rolling_median_ret_low")
    if worst_mdd_abs > float(g["max_worst_mdd_abs"]):
        reasons.append("rolling_worst_mdd_high")

    summary = {
        "windows": float(windows),
        "pf_pass_rate": float(pf_pass_rate),
        "ret_pass_rate": float(ret_pass_rate),
        "median_pf": float(median_pf),
        "median_ret": float(median_ret),
        "worst_mdd_abs": float(worst_mdd_abs),
    }
    return len(reasons) == 0, reasons, summary


def run_cost_stress_verification(
    best_params: Dict,
    symbols: List[str],
    primary_symbols: List[str],
    eval_start_time: pd.Timestamp,
    eval_end_time: pd.Timestamp,
    multipliers: Tuple[float, ...] = COST_STRESS_MULTIPLIERS,
) -> Dict[float, Optional[Dict]]:
    stress_summaries: Dict[float, Optional[Dict]] = {}
    for mult in multipliers:
        print("\n" + "-" * LOG_WIDTH)
        print(f"Cost Stress Verification x{mult:.1f} (fee/slippage)")
        print(f"Window: {eval_start_time} ~ {eval_end_time}")
        print("-" * LOG_WIDTH)
        stress_results: List[Dict] = []
        for symbol in symbols:
            r = verify_single_symbol_spot(
                symbol,
                best_params,
                primary_symbols,
                eval_start_time=eval_start_time,
                eval_end_time=eval_end_time,
                run_robustness_checks=False,
                cost_mult=float(mult),
            )
            if r:
                stress_results.append(r)
        calculate_mode_performance(stress_results)
        s = summarize_profile(stress_results)
        stress_summaries[float(mult)] = s
        if s:
            print(
                f"Cost Stress x{mult:.1f} Summary: "
                f"core_ret={s['core_avg_ret']:.2f}% | "
                f"core_mdd={s['core_avg_mdd_abs']:.2f}% | "
                f"core_pf={s['core_avg_pf']:.2f} | "
                f"core_min_trades={s['core_min_trades']}"
            )
        else:
            print(f"[WARN] Cost Stress x{mult:.1f} summary unavailable.")
    return stress_summaries


def deploy_best_to_local(
    source_storage_url: str,
    source_study_name: str,
    mode_label: str,
    target_db: str = "spot_strategy.db",
    deploy_metadata: Optional[Dict] = None,
) -> bool:
    try:
        if os.path.exists(target_db):
            os.remove(target_db)
        target_storage = f"sqlite:///{target_db}"
        src_study = _load_study_safely(study_name=source_study_name, storage=source_storage_url)
        best_trial = src_study.best_trial
        optuna.create_study(study_name="spot_strategy", storage=target_storage, direction="maximize", load_if_exists=True)
        study_dest = _load_study_safely(study_name="spot_strategy", storage=target_storage)
        user_attrs = dict(getattr(best_trial, "user_attrs", {}) or {})
        if deploy_metadata:
            user_attrs.update(deploy_metadata)
        frozen_trial = optuna.trial.create_trial(
            params=best_trial.params,
            distributions=best_trial.distributions,
            value=best_trial.value,
            user_attrs=user_attrs,
        )
        study_dest.add_trial(frozen_trial)
        print(f"[OK] Deployed strategy: {mode_label}")
        print(f"     Target DB: {target_db}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to deploy strategy: {e}")
        return False


def _run_mode_eval(
    mode: str,
    storage_url: str,
    symbols: List[str],
    primary_symbols: List[str],
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> Optional[Dict]:
    study_name, best_params, train_score = load_best_params_from_mysql(mode, storage_url)
    if study_name is None or best_params is None:
        print(f"[WARN] {mode} strategy not found in MySQL. Skipping.")
        return None
    print(f"   Loaded Params (Train Score: {float(train_score):.4f})")
    print(f"   Timeframe: {best_params.get('TIMEFRAME', '1h')}")
    all_results: List[Dict] = []
    for symbol in symbols:
        r = verify_single_symbol_spot(
            symbol,
            best_params,
            primary_symbols,
            eval_start_time=eval_start,
            eval_end_time=eval_end,
            run_robustness_checks=True,
        )
        if r:
            all_results.append(r)
    calculate_mode_performance(all_results)
    profile = summarize_profile(all_results)
    if profile:
        reasons = profile.get("gate_reasons", [])
        gate_state = "PASS" if profile.get("gates_passed") else f"FAIL({','.join(reasons)})"
        print(
            f"[PROFILE] {mode}: gate={gate_state} | core_ret={profile['core_avg_ret']:.2f}% | "
            f"core_mdd={profile['core_avg_mdd_abs']:.2f}% | "
            f"core_pf={profile['core_avg_pf']:.2f}(clip:{profile.get('core_avg_pf_clipped', profile['core_avg_pf']):.2f},"
            f"shrink:{profile.get('core_avg_pf_clipped_shrunk', profile.get('core_avg_pf_shrunk', profile['core_avg_pf'])):.2f}) | "
            f"core_min_trades={profile['core_min_trades']} | rel={profile.get('core_reliability', 0.0):.2f} | "
            f"reg_n(up/down)={profile.get('core_up_sample_count', 0)}/{profile.get('core_down_sample_count', 0)}"
        )
    return {
        "mode": mode,
        "study_name": study_name,
        "best_params": best_params,
        "train_score": float(train_score),
        "results": all_results,
        "profile": profile,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="KRW-BTC,KRW-ETH")
    parser.add_argument("--alt", type=int, default=0, choices=[0, 1], help="Include altcoins: SOL, XRP, DOGE, ADA")
    parser.add_argument("--dry-run", action="store_true", help="Verify currently deployed spot_strategy.db only")
    parser.add_argument(
        "--bonus-sweep",
        dest="bonus_sweep",
        action="store_true",
        help="Verify A/B/C spot bonus DBs and pick winner automatically (default: enabled).",
    )
    parser.add_argument(
        "--no-bonus-sweep",
        dest="bonus_sweep",
        action="store_false",
        help="Disable A/B/C bonus sweep and verify only DB_NAME from .env.",
    )
    parser.add_argument("--rolling-oos", dest="rolling_oos", action="store_true", help="Enable rolling OOS (default)")
    parser.add_argument("--no-rolling-oos", dest="rolling_oos", action="store_false", help="Disable rolling OOS")
    parser.add_argument("--roll-window-days", type=int, default=90)
    parser.add_argument("--roll-step-days", type=int, default=30)
    parser.add_argument("--roll-max-windows", type=int, default=2)
    parser.add_argument(
        "--show-diag",
        action="store_true",
        help="Show zero-trade diagnostics (default: off).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose verifier logs (default: warning only).",
    )
    parser.set_defaults(bonus_sweep=True)
    parser.set_defaults(rolling_oos=True)
    args = parser.parse_args()
    SHOW_ZERO_TRADE_DIAG = bool(args.show_diag)
    logger.setLevel(logging.INFO if args.verbose else logging.WARNING)

    base_symbols = [s.strip() for s in args.symbols.split(",")]
    if args.alt == 1:
        for alt in ["KRW-SOL", "KRW-XRP", "KRW-DOGE", "KRW-ADA"]:
            if alt not in base_symbols:
                base_symbols.append(alt)
    symbols = base_symbols
    PRIMARY_SYMBOLS = ["KRW-BTC", "KRW-ETH"]
    MODES = ["UNIFIED"]

    if args.dry_run:
        _log_header("DRY-RUN: Verify Deployed Spot Strategy", "Source: spot_strategy.db")
        target_db = "spot_strategy.db"
        if not os.path.exists(target_db):
            print(f"[ERROR] {target_db} not found.")
            sys.exit(1)
        try:
            local_storage = f"sqlite:///{target_db}"
            study = _load_study_safely(study_name="spot_strategy", storage=local_storage)
            best_params = study.best_params
            print(f"[INFO] Loaded current strategy (Train Score: {study.best_value:.4f})")
            print(f"   Timeframe: {best_params.get('TIMEFRAME', '1h')}")
            eval_start, eval_end = resolve_eval_window()
            all_results: List[Dict] = []
            for symbol in symbols:
                r = verify_single_symbol_spot(symbol, best_params, PRIMARY_SYMBOLS, eval_start, eval_end)
                if r:
                    all_results.append(r)
            avg_ret = calculate_mode_performance(all_results)
            dry_run_profile = summarize_profile(all_results)
            _record_verification_comparison(
                log_path=SPOT_VERIFY_LOG_PATH,
                run_type="dry_run",
                current_best={
                    "label": "deployed_local_strategy",
                    "source_db_name": "spot_strategy.db",
                    "study_name": "spot_strategy",
                    "timeframe": str(best_params.get("TIMEFRAME", "1h")),
                    "metrics": _extract_summary_metrics(
                        dry_run_profile,
                        {
                            "avg_ret": avg_ret,
                            "train_score": study.best_value,
                        },
                    ),
                    "summary": dry_run_profile,
                },
                context={
                    "symbols": list(symbols),
                    "eval_start": str(eval_start),
                    "eval_end": str(eval_end),
                    "deployed": False,
                },
            )
            _log_header("DRY-RUN COMPLETE", "No changes saved")
            if avg_ret is not None:
                print(f"[INFO] Current strategy OOS performance: {avg_ret:.2f}%")
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] Dry-run failed: {e}")
            sys.exit(1)

    _log_header(
        "INTEGRATED STRATEGY VERIFICATION (Spot)",
        f"Searching optimized strategies: {MODES}",
    )
    load_dotenv()
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "trading_optuna")
    if not all([db_user, db_pass, db_name]):
        print("[ERROR] Missing DB credentials in .env")
        sys.exit(1)
    storage_url = f"mysql+pymysql://{db_user}:{quote_plus(db_pass)}@{db_host}:{db_port}/{db_name}"

    oos_start, selection_end, holdout_start, holdout_end, holdout_enabled, holdout_days = split_bonus_oos_windows()

    if args.bonus_sweep:
        bonus_dbs = {
            "A": "trading_optuna_spot_bonus_a",
            "B": "trading_optuna_spot_bonus_b",
            "C": "trading_optuna_spot_bonus_c",
        }
        rank_mode = "UNIFIED" if "UNIFIED" in MODES else MODES[0]

        holdout_subtitle = (
            f"Holdout window (sanity): {holdout_start} ~ {holdout_end}"
            if holdout_enabled and holdout_start is not None and holdout_end is not None
            else f"Holdout window (sanity): DISABLED (available={holdout_days}d, require>={HOLDOUT_ACTIVATION_MIN_DAYS}d)"
        )
        _log_header(
            "BONUS SWEEP VERIFICATION (A/B/C)",
            f"Selection window (rank): {oos_start} ~ {selection_end}\n{holdout_subtitle}",
        )

        profile_results: Dict[str, Dict] = {}
        mode_sources: Dict[str, Dict] = {}
        for key, bonus_db in bonus_dbs.items():
            _log_subheader(f"[{key}] DB: {bonus_db}")
            profile_url = f"mysql+pymysql://{db_user}:{quote_plus(db_pass)}@{db_host}:{db_port}/{bonus_db}"
            summary_by_mode: Dict[str, Dict] = {}
            source_by_mode: Dict[str, Dict] = {}

            for mode in MODES:
                print(f"\nVerifying {mode} Mode Strategy (from MySQL)...")
                ev = _run_mode_eval(
                    mode,
                    profile_url,
                    symbols,
                    PRIMARY_SYMBOLS,
                    oos_start,
                    selection_end,
                )
                if not ev:
                    continue
                summary_by_mode[mode] = {
                    "avg_ret": float(ev.get("profile", {}).get("core_avg_ret", 0.0))
                    if ev.get("profile")
                    else 0.0,
                    "summary": ev.get("profile"),
                }
                source_by_mode[mode] = {
                    "storage_url": profile_url,
                    "study_name": ev["study_name"],
                    "db_name": bonus_db,
                    "best_params": ev["best_params"],
                    "train_score": ev.get("train_score"),
                    "mode": mode,
                }

            profile_results[key] = summary_by_mode
            if rank_mode in source_by_mode:
                mode_sources[key] = source_by_mode[rank_mode]

        rank_summaries = {
            k: (v.get(rank_mode, {}) or {}).get("summary")
            for k, v in profile_results.items()
        }
        ranked = rank_profiles(rank_summaries)
        ranking_mode = "strict"
        if not ranked:
            ranked = rank_profiles_soft(rank_summaries)
            ranking_mode = "soft"

        _log_header(
            f"BONUS SWEEP WINNER ({rank_mode}) - SELECTION WINDOW",
            f"Policy: {SELECTION_POLICY_VERSION}",
        )
        for k, s in rank_summaries.items():
            if not s:
                print(f"- {k}: no summary")
                continue
            gate_state = "PASS" if s.get("gates_passed") else f"FAIL({','.join(s.get('gate_reasons', []))})"
            print(
                f"- {k}: {gate_state} | core_ret={s['core_avg_ret']:.2f}% | "
                f"core_mdd={s['core_avg_mdd_abs']:.2f}% | "
                f"core_pf={s.get('core_avg_pf_clipped_shrunk', s.get('core_avg_pf_shrunk', s['core_avg_pf'])):.2f}(raw:{s['core_avg_pf']:.2f}) | "
                f"core_min_trades={s['core_min_trades']} | core_wfa={s['core_wfa_consistency']:.1f}% | "
                f"core_mc_mdd95={s['core_mc_worst_mdd_95']:.2f}% | alt_med={s['alt_median_ret']:.2f}% | "
                f"alt_p25={s['alt_p25_ret']:.2f}% | alt_pos={s['alt_pos_rate']*100:.1f}% | dispersion={s['dispersion']:.2f}"
            )

        if not ranked:
            print("[ERROR] No valid summaries to rank.")
            sys.exit(1)
        if ranking_mode == "soft":
            print("[WARN] No strict gate-passed candidate. Using soft ranking fallback for holdout evaluation.")
        winner, score, s = ranked[0]
        print(f"Winner (selection): {winner} (score={score:.2f})")
        print(
            f"   core_ret={s['core_avg_ret']:.2f}% | core_mdd={s['core_avg_mdd_abs']:.2f}% | "
            f"core_pf={s.get('core_avg_pf_clipped_shrunk', s.get('core_avg_pf_shrunk', s['core_avg_pf'])):.2f}(raw:{s['core_avg_pf']:.2f}) | "
            f"core_wfa={s['core_wfa_consistency']:.1f}% | "
            f"core_mc_mdd95={s['core_mc_worst_mdd_95']:.2f}% | "
            f"alt_med={s['alt_median_ret']:.2f}% | alt_p25={s['alt_p25_ret']:.2f}% | "
            f"alt_pos={s['alt_pos_rate']*100:.1f}%"
        )

        selected_holdout_min_trades = int(HOLDOUT_SANITY_GATES["core_min_trades"])
        holdout_evals: Dict[str, Dict] = {}
        holdout_passed = not holdout_enabled
        winner_src = mode_sources.get(winner)

        if holdout_enabled and holdout_start is not None and holdout_end is not None:
            print(
                f"[INFO] Holdout dynamic trade gate (window: {holdout_start} ~ {holdout_end})"
            )
            # Gate window must be strictly OOS vs selection window.
            assert_strict_oos_window(selection_end, holdout_start, holdout_end)

            holdout_order = [k for k, _, _ in ranked]
            for candidate_key in holdout_order:
                candidate_src = mode_sources.get(candidate_key)
                if not candidate_src:
                    holdout_evals[candidate_key] = {
                        "passed": False,
                        "reasons": ["holdout_not_evaluated"],
                        "summary": None,
                        "source": None,
                    }
                    continue

                candidate_params = candidate_src["best_params"]
                candidate_study_name = candidate_src["study_name"]
                candidate_tf = candidate_params.get("TIMEFRAME")

                candidate_min_trades_gate = int(
                    compute_dynamic_holdout_min_trades(holdout_start, holdout_end, timeframe=candidate_tf)
                )

                print("\n" + "-" * LOG_WIDTH)
                print(f"Holdout Verification for Candidate [{candidate_key}] (used for final selection)")
                print(f"Window: {holdout_start} ~ {holdout_end}")
                print(f"Trade Gate: min_trades>={candidate_min_trades_gate}")
                print("-" * LOG_WIDTH)
                gate_results: List[Dict] = []
                for symbol in symbols:
                    r = verify_single_symbol_spot(
                        symbol,
                        candidate_params,
                        PRIMARY_SYMBOLS,
                        eval_start_time=holdout_start,
                        eval_end_time=holdout_end,
                        run_robustness_checks=False,
                    )
                    if r:
                        gate_results.append(r)
                calculate_mode_performance(gate_results)
                holdout_summary = summarize_profile(gate_results)
                passed, reasons = evaluate_holdout_sanity(
                    holdout_summary,
                    min_trades_gate=candidate_min_trades_gate,
                )
                gate_state, gate_reasons_display = classify_holdout_outcome(
                    holdout_summary,
                    bool(passed),
                    reasons,
                    int(candidate_min_trades_gate),
                )
                if holdout_summary:
                    print(
                        f"Holdout Summary: core_ret={holdout_summary['core_avg_ret']:.2f}% | "
                        f"core_excess={holdout_summary.get('core_avg_excess_ret', 0.0):.2f}% | "
                        f"core_mdd={holdout_summary['core_avg_mdd_abs']:.2f}% | "
                        f"core_pf={holdout_summary['core_avg_pf']:.2f}(shrink:{holdout_summary.get('core_avg_pf_clipped_shrunk', holdout_summary.get('core_avg_pf_shrunk', holdout_summary['core_avg_pf'])):.2f}) | "
                        f"core_min_trades={holdout_summary['core_min_trades']} | "
                        f"core_total_trades={holdout_summary.get('core_total_trades', holdout_summary['core_min_trades'])}"
                    )
                if gate_state == "PASS":
                    print("Holdout Gate: PASS")
                else:
                    print(f"Holdout Gate: {gate_state}({','.join(gate_reasons_display)})")
                holdout_source = dict(candidate_src)
                holdout_source["best_params"] = candidate_params
                holdout_source["study_name"] = candidate_study_name
                holdout_evals[candidate_key] = {
                    "passed": bool(passed),
                    "reasons": reasons,
                    "gate_state": str(gate_state),
                    "gate_reasons_display": list(gate_reasons_display),
                    "summary": holdout_summary,
                    "source": holdout_source,
                    "min_trades_gate": int(candidate_min_trades_gate),
                    "holdout_gate_window": (str(holdout_start), str(holdout_end)),
                }

            # Among holdout-passed candidates, choose by holdout score (OOS performance), not selection order.
            final_candidate_key: Optional[str] = None
            passed_with_scores: List[Tuple[str, float, int]] = []
            for rank_idx, (candidate_key, _, _) in enumerate(ranked):
                ev = holdout_evals.get(candidate_key)
                if not ev or not ev.get("passed"):
                    continue
                summary = ev.get("summary")
                ho_score = score_holdout_candidate(summary) if summary else -1.0
                passed_with_scores.append((candidate_key, ho_score, rank_idx))
            if passed_with_scores:
                # Max holdout score; tiebreaker: first in selection rank (lower rank_idx).
                passed_with_scores.sort(key=lambda x: (-x[1], x[2]))
                final_candidate_key = passed_with_scores[0][0]

            holdout_passed = final_candidate_key is not None
            if holdout_passed:
                winner = str(final_candidate_key)
                winner_src = holdout_evals[winner]["source"]
                selected_holdout_min_trades = int(
                    holdout_evals[winner].get("min_trades_gate", HOLDOUT_SANITY_GATES["core_min_trades"])
                )
                winner_ho_score = next((s for k, s, _ in passed_with_scores if k == winner), -1.0)
                print(f"[INFO] Final winner after holdout gate: {winner} (holdout_score={winner_ho_score:.4f})")
            else:
                inactive_count = int(
                    sum(1 for ev in holdout_evals.values() if str(ev.get("gate_state", "")).upper() == "INACTIVE")
                )
                if inactive_count > 0:
                    print(
                        f"[INFO] Holdout result: INACTIVE candidates={inactive_count}/{len(holdout_evals)} "
                        "(insufficient activity in gate window)."
                    )
                print("[WARN] No holdout-passed candidate found. Deployment skipped.")
        else:
            print(
                f"[INFO] Holdout disabled: unseen window={holdout_days}d "
                f"(require>={HOLDOUT_ACTIVATION_MIN_DAYS}d). Using selection winner for deployment candidate."
            )

        deployment_attempted = False
        deployment_ok = False
        if winner_src and holdout_passed:
            best_params = winner_src["best_params"]
            rolling_summary = []
            rolling_passed = True
            rolling_reasons: List[str] = []
            rolling_sanity_summary: Dict[str, float] = {}
            stress_summary: Dict[str, float] = {}
            if holdout_enabled and holdout_start is not None and holdout_end is not None:
                stress_summary = run_cost_stress_verification(
                    best_params=best_params,
                    symbols=symbols,
                    primary_symbols=PRIMARY_SYMBOLS,
                    eval_start_time=holdout_start,
                    eval_end_time=holdout_end,
                )
                if args.rolling_oos:
                    rolling_summary = run_rolling_oos_verification(
                        best_params=best_params,
                        symbols=symbols,
                        primary_symbols=PRIMARY_SYMBOLS,
                        eval_start_time=oos_start,
                        eval_end_time=holdout_end,
                        window_days=args.roll_window_days,
                        step_days=args.roll_step_days,
                        max_windows=args.roll_max_windows,
                    )
                    rolling_passed, rolling_reasons, rolling_sanity_summary = evaluate_rolling_sanity(rolling_summary)
                    rolling_gate_state = (
                        "PASS" if rolling_passed else f"FAIL({','.join(rolling_reasons)})"
                    )
                    print(
                        "[ROLL-GATE] "
                        f"windows={int(rolling_sanity_summary.get('windows', 0))} | "
                        f"pf_pass_rate={rolling_sanity_summary.get('pf_pass_rate', 0.0)*100:.1f}% | "
                        f"ret_pass_rate={rolling_sanity_summary.get('ret_pass_rate', 0.0)*100:.1f}% | "
                        f"median_pf={rolling_sanity_summary.get('median_pf', 0.0):.2f} | "
                        f"median_ret={rolling_sanity_summary.get('median_ret', 0.0):.2f}% | "
                        f"worst_mdd={rolling_sanity_summary.get('worst_mdd_abs', 0.0):.2f}% | "
                        f"gate={rolling_gate_state}"
                    )
            else:
                print(
                    f"[INFO] OOS gates deferred: unseen window={holdout_days}d "
                    f"(require>={HOLDOUT_ACTIVATION_MIN_DAYS}d)."
                )
            if not rolling_passed:
                print("[WARN] Rolling OOS sanity gate failed — proceeding to deployment with caution.")

            print("-" * LOG_WIDTH)
            print(
                f"Saving Winner ({winner}) from MySQL "
                f"[{winner_src['db_name']}/{winner_src['study_name']}] to 'spot_strategy.db'..."
            )
            deployment_attempted = True
            deployment_ok = deploy_best_to_local(
                source_storage_url=winner_src["storage_url"],
                source_study_name=winner_src["study_name"],
                mode_label=f"{winner_src['mode']}/{winner}",
                target_db="spot_strategy.db",
                deploy_metadata={
                    "policy_version": SELECTION_POLICY_VERSION,
                    "selected_mode": winner_src["mode"],
                    "selected_profile": winner,
                    "source_db_name": winner_src["db_name"],
                    "selection_window_start": str(oos_start),
                    "selection_window_end": str(selection_end),
                    "holdout_window_start": str(holdout_start) if holdout_start is not None else "N/A",
                    "holdout_window_end": str(holdout_end) if holdout_end is not None else "N/A",
                    "holdout_enabled": bool(holdout_enabled),
                    "holdout_activation_min_days": int(HOLDOUT_ACTIVATION_MIN_DAYS),
                    "holdout_available_days": int(holdout_days),
                    "holdout_dynamic_min_trades": int(selected_holdout_min_trades),
                    "cost_stress_multipliers": ",".join(str(x) for x in COST_STRESS_MULTIPLIERS),
                    "cost_stress_summary": str(stress_summary),
                    "rolling_oos_enabled": bool(args.rolling_oos and holdout_enabled),
                    "rolling_oos_summary": str(rolling_summary),
                    "rolling_oos_gate_passed": bool(rolling_passed),
                    "rolling_oos_gate_reasons": ",".join(rolling_reasons),
                    "rolling_oos_gate_metrics": str(rolling_sanity_summary),
                },
            )
        else:
            print("[WARN] Skip deployment due to failed/missing holdout sanity check.")
        selected_eval = holdout_evals.get(winner, {}) if isinstance(holdout_evals, dict) else {}
        selected_holdout_summary = (
            selected_eval.get("summary") if isinstance(selected_eval, dict) else None
        )
        selected_selection_summary = rank_summaries.get(winner)
        selected_summary = selected_holdout_summary or selected_selection_summary
        selected_source = (
            selected_eval.get("source") if isinstance(selected_eval, dict) else None
        ) or winner_src or mode_sources.get(winner)
        selected_rank_score = next(
            (float(rank_score) for key, rank_score, _ in ranked if key == winner),
            None,
        )
        _record_verification_comparison(
            log_path=SPOT_VERIFY_LOG_PATH,
            run_type="bonus_sweep",
            current_best={
                "label": str(winner),
                "selected_mode": selected_source.get("mode") if isinstance(selected_source, dict) else rank_mode,
                "study_name": selected_source.get("study_name") if isinstance(selected_source, dict) else None,
                "source_db_name": selected_source.get("db_name") if isinstance(selected_source, dict) else None,
                "timeframe": str(
                    (selected_source.get("best_params", {}) if isinstance(selected_source, dict) else {}).get(
                        "TIMEFRAME",
                        "1h",
                    )
                ),
                "holdout_gate_state": (
                    selected_eval.get("gate_state")
                    if isinstance(selected_eval, dict)
                    else ("SKIPPED" if not holdout_enabled else "NOT_EVALUATED")
                ),
                "holdout_gate_reasons": (
                    selected_eval.get("gate_reasons_display", selected_eval.get("reasons", []))
                    if isinstance(selected_eval, dict)
                    else []
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
                "holdout_window_start": str(holdout_start) if holdout_start is not None else None,
                "holdout_window_end": str(holdout_end) if holdout_end is not None else None,
                "holdout_enabled": bool(holdout_enabled),
                "holdout_passed": bool(holdout_passed),
                "deployment_attempted": bool(deployment_attempted),
                "deployment_ok": bool(deployment_ok),
            },
        )
        if deployment_attempted and not deployment_ok:
            sys.exit(1)
        print("=" * LOG_WIDTH)
        sys.exit(0)

    candidates: Dict[str, Dict] = {}
    for mode in MODES:
        print(f"\nVerifying {mode} mode strategy (from MySQL)...")
        ev = _run_mode_eval(
            mode,
            storage_url,
            symbols,
            PRIMARY_SYMBOLS,
            oos_start,
            selection_end,
        )
        if ev:
            candidates[mode] = ev

    if not candidates:
        print("[ERROR] No valid candidates found in MySQL.")
        sys.exit(1)

    profile_summaries = {k: v["profile"] for k, v in candidates.items() if v.get("profile")}
    ranked = rank_profiles(profile_summaries)
    print("\n" + "=" * LOG_WIDTH)
    print("Selection Ranking")
    print("=" * LOG_WIDTH)
    if ranked:
        for i, (mode, score, prof) in enumerate(ranked, start=1):
            print(
                f"{i}. {mode:<7} | rank_score={score:.4f} | core_ret={prof['core_avg_ret']:.2f}% | "
                f"core_mdd={prof['core_avg_mdd_abs']:.2f}% | core_pf={prof['core_avg_pf']:.2f} | "
                f"core_min_trades={prof['core_min_trades']}"
            )
    else:
        print("[WARN] No gate-passed candidate in selection window.")

    holdout_passed_mode: Optional[str] = None
    holdout_passed_candidate: Optional[Dict] = None
    last_holdout_gate_state = "FAIL"
    selected_holdout_min_trades = int(HOLDOUT_SANITY_GATES["core_min_trades"])
    if holdout_enabled and holdout_start is not None and holdout_end is not None:
        print("[INFO] Holdout dynamic trade gate")
        assert_strict_oos_window(selection_end, holdout_start, holdout_end)

        check_order = [r[0] for r in ranked] if ranked else list(candidates.keys())
        for mode in check_order:
            c = candidates[mode]
            holdout_params = c["best_params"]
            holdout_study_name = c["study_name"]
            candidate_tf = holdout_params.get("TIMEFRAME")
            candidate_min_trades_gate = int(
                compute_dynamic_holdout_min_trades(holdout_start, holdout_end, timeframe=candidate_tf)
            )

            print("\n" + "-" * LOG_WIDTH)
            print(f"Holdout Verification for Candidate [{mode}] (used for final selection)")
            print(f"Window: {holdout_start} ~ {holdout_end}")
            print(f"Trade Gate: min_trades>={candidate_min_trades_gate}")
            print("-" * LOG_WIDTH)
            holdout_results: List[Dict] = []
            for symbol in symbols:
                r = verify_single_symbol_spot(
                    symbol,
                    holdout_params,
                    PRIMARY_SYMBOLS,
                    eval_start_time=holdout_start,
                    eval_end_time=holdout_end,
                    run_robustness_checks=False,
                )
                if r:
                    holdout_results.append(r)
            calculate_mode_performance(holdout_results)
            holdout_summary = summarize_profile(holdout_results)
            passed, reasons = evaluate_holdout_sanity(
                holdout_summary,
                min_trades_gate=candidate_min_trades_gate,
            )
            gate_state, gate_reasons_display = classify_holdout_outcome(
                holdout_summary,
                bool(passed),
                reasons,
                int(candidate_min_trades_gate),
            )
            last_holdout_gate_state = str(gate_state)
            if holdout_summary:
                print(
                    f"Holdout Summary: core_ret={holdout_summary['core_avg_ret']:.2f}% | "
                    f"core_excess={holdout_summary.get('core_avg_excess_ret', 0.0):.2f}% | "
                    f"core_mdd={holdout_summary['core_avg_mdd_abs']:.2f}% | "
                    f"core_pf={holdout_summary['core_avg_pf']:.2f}(shrink:{holdout_summary.get('core_avg_pf_clipped_shrunk', holdout_summary.get('core_avg_pf_shrunk', holdout_summary['core_avg_pf'])):.2f}) | "
                    f"core_min_trades={holdout_summary['core_min_trades']} | "
                    f"core_total_trades={holdout_summary.get('core_total_trades', holdout_summary['core_min_trades'])}"
                )
            if gate_state == "PASS":
                print("Holdout Gate: PASS")
            else:
                print(f"Holdout Gate: {gate_state}({','.join(gate_reasons_display)})")
            if passed:
                holdout_passed_mode = mode
                holdout_passed_candidate = dict(c)
                holdout_passed_candidate["best_params"] = holdout_params
                holdout_passed_candidate["study_name"] = holdout_study_name
                selected_holdout_min_trades = int(candidate_min_trades_gate)
                break

        if not holdout_passed_candidate or not holdout_passed_mode:
            # Keep deployment criteria strict (PASS only), but explain inactivity distinctly.
            if last_holdout_gate_state == "INACTIVE":
                print("[INFO] Holdout result classified as INACTIVE (low/no activity in gate window).")
            print("[WARN] No holdout-passed candidate found. Deployment skipped.")
            print("[WARN] Skip deployment due to failed/missing holdout sanity check.")
            fallback_mode = str(check_order[0]) if check_order else None
            fallback_candidate = candidates.get(fallback_mode) if fallback_mode is not None else None
            fallback_rank_score = next(
                (float(rank_score) for mode_key, rank_score, _ in ranked if mode_key == fallback_mode),
                None,
            )
            _record_verification_comparison(
                log_path=SPOT_VERIFY_LOG_PATH,
                run_type="single_mode",
                current_best={
                    "label": fallback_mode,
                    "selected_mode": fallback_mode,
                    "study_name": fallback_candidate.get("study_name") if isinstance(fallback_candidate, dict) else None,
                    "source_db_name": db_name,
                    "timeframe": str(
                        (fallback_candidate.get("best_params", {}) if isinstance(fallback_candidate, dict) else {}).get(
                            "TIMEFRAME",
                            "1h",
                        )
                    ),
                    "holdout_gate_state": str(last_holdout_gate_state),
                    "holdout_gate_reasons": ["holdout_failed"],
                    "metrics": _extract_summary_metrics(
                        fallback_candidate.get("profile") if isinstance(fallback_candidate, dict) else None,
                        {
                            "selection_rank_score": fallback_rank_score,
                            "train_score": fallback_candidate.get("train_score")
                            if isinstance(fallback_candidate, dict)
                            else None,
                        },
                    ),
                    "summary": fallback_candidate.get("profile") if isinstance(fallback_candidate, dict) else None,
                },
                context={
                    "symbols": list(symbols),
                    "selection_window_start": str(oos_start),
                    "selection_window_end": str(selection_end),
                    "holdout_window_start": str(holdout_start),
                    "holdout_window_end": str(holdout_end),
                    "holdout_enabled": True,
                    "holdout_passed": False,
                    "deployment_attempted": False,
                    "deployment_ok": False,
                },
            )
            sys.exit(0)
    else:
        check_order = [r[0] for r in ranked] if ranked else list(candidates.keys())
        if not check_order:
            print("[ERROR] No selection candidate available for deployment.")
            sys.exit(1)
        top_mode = str(check_order[0])
        holdout_passed_mode = top_mode
        holdout_passed_candidate = dict(candidates[top_mode])
        print(
            f"[INFO] Holdout disabled: unseen window={holdout_days}d "
            f"(require>={HOLDOUT_ACTIVATION_MIN_DAYS}d). Using selection winner [{top_mode}]."
        )

    print(f"[INFO] Final winner after holdout gate: {holdout_passed_mode}")
    best_params = holdout_passed_candidate["best_params"]
    rolling_summary = []
    rolling_passed = True
    rolling_reasons: List[str] = []
    rolling_sanity_summary: Dict[str, float] = {}
    stress_summary: Dict[str, float] = {}
    if holdout_enabled and holdout_start is not None and holdout_end is not None:
        stress_summary = run_cost_stress_verification(
            best_params=best_params,
            symbols=symbols,
            primary_symbols=PRIMARY_SYMBOLS,
            eval_start_time=holdout_start,
            eval_end_time=holdout_end,
        )
        if args.rolling_oos:
            rolling_summary = run_rolling_oos_verification(
                best_params=best_params,
                symbols=symbols,
                primary_symbols=PRIMARY_SYMBOLS,
                eval_start_time=oos_start,
                eval_end_time=holdout_end,
                window_days=args.roll_window_days,
                step_days=args.roll_step_days,
                max_windows=args.roll_max_windows,
            )
            rolling_passed, rolling_reasons, rolling_sanity_summary = evaluate_rolling_sanity(rolling_summary)
            rolling_gate_state = (
                "PASS" if rolling_passed else f"FAIL({','.join(rolling_reasons)})"
            )
            print(
                "[ROLL-GATE] "
                f"windows={int(rolling_sanity_summary.get('windows', 0))} | "
                f"pf_pass_rate={rolling_sanity_summary.get('pf_pass_rate', 0.0)*100:.1f}% | "
                f"ret_pass_rate={rolling_sanity_summary.get('ret_pass_rate', 0.0)*100:.1f}% | "
                f"median_pf={rolling_sanity_summary.get('median_pf', 0.0):.2f} | "
                f"median_ret={rolling_sanity_summary.get('median_ret', 0.0):.2f}% | "
                f"worst_mdd={rolling_sanity_summary.get('worst_mdd_abs', 0.0):.2f}% | "
                f"gate={rolling_gate_state}"
            )
    else:
        print(
            f"[INFO] OOS gates deferred: unseen window={holdout_days}d "
            f"(require>={HOLDOUT_ACTIVATION_MIN_DAYS}d)."
        )
    if not rolling_passed:
        print("[WARN] Rolling OOS sanity gate failed — proceeding to deployment with caution.")

    print("-" * LOG_WIDTH)
    print(
        f"[INFO] Saving best strategy ({holdout_passed_mode}) from MySQL to 'spot_strategy.db'..."
    )
    ok = deploy_best_to_local(
        source_storage_url=storage_url,
        source_study_name=holdout_passed_candidate["study_name"],
        mode_label=holdout_passed_mode,
        target_db="spot_strategy.db",
        deploy_metadata={
            "policy_version": SELECTION_POLICY_VERSION,
            "selected_mode": holdout_passed_mode,
            "selection_window_start": str(oos_start),
            "selection_window_end": str(selection_end),
            "holdout_window_start": str(holdout_start) if holdout_start is not None else "N/A",
            "holdout_window_end": str(holdout_end) if holdout_end is not None else "N/A",
            "holdout_enabled": bool(holdout_enabled),
            "holdout_activation_min_days": int(HOLDOUT_ACTIVATION_MIN_DAYS),
            "holdout_available_days": int(holdout_days),
            "holdout_dynamic_min_trades": int(selected_holdout_min_trades),
            "cost_stress_multipliers": ",".join(str(x) for x in COST_STRESS_MULTIPLIERS),
            "cost_stress_summary": str(stress_summary),
            "rolling_oos_enabled": bool(args.rolling_oos and holdout_enabled),
            "rolling_oos_summary": str(rolling_summary),
            "rolling_oos_gate_passed": bool(rolling_passed),
            "rolling_oos_gate_reasons": ",".join(rolling_reasons),
            "rolling_oos_gate_metrics": str(rolling_sanity_summary),
        },
    )
    selected_rank_score = next(
        (float(rank_score) for mode_key, rank_score, _ in ranked if mode_key == holdout_passed_mode),
        None,
    )
    _record_verification_comparison(
        log_path=SPOT_VERIFY_LOG_PATH,
        run_type="single_mode",
        current_best={
            "label": str(holdout_passed_mode),
            "selected_mode": str(holdout_passed_mode),
            "study_name": holdout_passed_candidate.get("study_name"),
            "source_db_name": db_name,
            "timeframe": str(holdout_passed_candidate.get("best_params", {}).get("TIMEFRAME", "1h")),
            "holdout_gate_state": "PASS" if holdout_enabled else "SKIPPED",
            "holdout_gate_reasons": [] if holdout_enabled else ["holdout_disabled"],
            "metrics": _extract_summary_metrics(
                holdout_passed_candidate.get("profile"),
                {
                    "selection_rank_score": selected_rank_score,
                    "train_score": holdout_passed_candidate.get("train_score"),
                },
            ),
            "summary": holdout_passed_candidate.get("profile"),
        },
        context={
            "symbols": list(symbols),
            "selection_window_start": str(oos_start),
            "selection_window_end": str(selection_end),
            "holdout_window_start": str(holdout_start) if holdout_start is not None else None,
            "holdout_window_end": str(holdout_end) if holdout_end is not None else None,
            "holdout_enabled": bool(holdout_enabled),
            "holdout_passed": True,
            "deployment_attempted": True,
            "deployment_ok": bool(ok),
        },
    )
    if not ok:
        sys.exit(1)
