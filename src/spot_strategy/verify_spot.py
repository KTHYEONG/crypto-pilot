import argparse
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from statistics import NormalDist
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

from config.optimization_config_modes import GET_SPOT_TRADE_GATE_POLICY
from config.settings import BACKTEST_END_DATE, BACKTEST_START_DATE, DATA_DIR, TRAIN_CUTOFF_DATE
from src.common.utils import setup_logger
from src.spot_strategy.engine_fast_spot import BacktestEngineFastSpot, backtest_loop_spot_numba
from src.spot_strategy.monte_carlo_spot import SpotMonteCarloSimulator
from src.spot_strategy.upbit_client import UpbitClient
from src.spot_strategy.walk_forward_spot import SpotWalkForwardAnalyzer
from src.strategy.strategies import UltimateStrategy

logger = setup_logger("SpotVerifier")
logger.setLevel(logging.WARNING)

LOG_WIDTH = 80
SPOT_INITIAL_BALANCE = 1_000_000.0
SPOT_BASE_FEE = 0.0005
SPOT_BASE_SLIPPAGE = 0.0003
WARMUP_DAYS = 60
COST_STRESS_MULTIPLIERS = (1.5, 2.0)
BONUS_SWEEP_HOLDOUT_RATIO = 0.30
BONUS_SWEEP_MIN_HOLDOUT_DAYS = 120
BONUS_SELECTION_FIXED_START_DATE = "2025-01-01"
BONUS_SELECTION_FIXED_END_DATE = "2026-02-10"
# OOS is enabled only when unseen post-selection data is sufficiently long.
# For spot (BTC/ETH primary, 4h often sparse), 6 months is the minimum practical floor.
HOLDOUT_ACTIVATION_MIN_DAYS = 180
GAP_FILL_MAX_RANGES = int(os.getenv("SPOT_GAP_FILL_MAX_RANGES", "3"))

# Process-local cache to avoid repeated disk I/O + repeated gap backfills across
# holdout/cost-stress/rolling windows in the same run.
_SPOT_DATA_CACHE: Dict[Tuple[str, str, str, str], Tuple[pd.DataFrame, pd.DataFrame]] = {}
_SPOT_GAP_FILLED_KEYS: set = set()
SHOW_ZERO_TRADE_DIAG = False
SELECTION_POLICY_VERSION = "SPOT_SELECTION_POLICY_V7_PARITY"
SELECTION_POLICY = {
    "gates": {
        # Simple is best: deploy gate uses only core profitability, risk, and activity.
        "core_min_avg_return": 0.0,
        "core_min_pf_clipped": 1.05,
        "core_max_avg_mdd_abs": 15.0,
        "core_min_trades": 10,
        "core_min_total_trades": 20,
    },
    "robust": {
        "pf_clip_per_symbol": 6.0,
    },
}

SPOT_GATE_POLICY = GET_SPOT_TRADE_GATE_POLICY()
SPOT_GATE_STAT = dict(SPOT_GATE_POLICY.get("statistical", {}))
SPOT_GATE_SELECTION = dict(SPOT_GATE_POLICY.get("selection", {}))
SPOT_GATE_HOLDOUT = dict(SPOT_GATE_POLICY.get("holdout", {}))
SPOT_GATE_HOLDOUT_SANITY = dict(SPOT_GATE_HOLDOUT.get("sanity_gates", {}))

SPOT_TF_SELECTION_TRADE_GATES = dict(SPOT_GATE_SELECTION.get("tf_min_gates", {}))
SPOT_TF_HOLDOUT_TRADES_PER_30D = dict(SPOT_GATE_HOLDOUT.get("trades_per_30d", {}))
SPOT_TF_HOLDOUT_MIN_TRADES_FLOOR = dict(SPOT_GATE_HOLDOUT.get("min_trades_floor", {}))
SPOT_TF_HOLDOUT_MIN_TRADES_CAP = dict(SPOT_GATE_HOLDOUT.get("min_trades_cap", {}))
SPOT_SELECTION_USE_STAT_MIN_TRADES = bool(SPOT_GATE_SELECTION.get("use_stat_min_trades", True))
SPOT_HOLDOUT_USE_STAT_MIN_TRADES = bool(SPOT_GATE_HOLDOUT.get("use_stat_min_trades", True))
SPOT_STAT_TRADE_CONFIDENCE = float(SPOT_GATE_STAT.get("confidence", 0.80))
SPOT_STAT_TRADE_MARGIN_ERROR = float(SPOT_GATE_STAT.get("margin_error", 0.25))
SPOT_STAT_TRADE_REFERENCE_DAYS = int(SPOT_GATE_STAT.get("reference_days", 120))
SPOT_STAT_TRADE_DAY_SCALE_EXP = float(SPOT_GATE_STAT.get("day_scale_exp", 0.5))
SPOT_STAT_TRADE_MIN_DAY_SCALE = float(SPOT_GATE_STAT.get("min_day_scale", 0.6))
SPOT_SELECTION_STAT_MIN_TRADES_CAP = int(SPOT_GATE_SELECTION.get("stat_min_trades_cap", 60))
SPOT_SELECTION_TOTAL_TRADES_MULTIPLIER = float(SPOT_GATE_SELECTION.get("total_trades_multiplier", 1.0))
SPOT_PF_SHRINK_REF_TRADES = float(SPOT_GATE_SELECTION.get("pf_shrink_ref_trades", 30.0))
HOLDOUT_SANITY_GATES = {
    "core_min_return": float(SPOT_GATE_HOLDOUT_SANITY.get("core_min_return", 0.0)),
    "core_min_excess_ret": float(SPOT_GATE_HOLDOUT_SANITY.get("core_min_excess_ret", 2.0)),
    "core_dual_return_mode": bool(SPOT_GATE_HOLDOUT_SANITY.get("core_dual_return_mode", True)),
    "core_min_pf": float(SPOT_GATE_HOLDOUT_SANITY.get("core_min_pf", 1.05)),
    "core_min_trades": int(SPOT_GATE_HOLDOUT_SANITY.get("core_min_trades", 8)),
    "core_min_symbol_trades": int(SPOT_GATE_HOLDOUT_SANITY.get("core_min_symbol_trades", 6)),
    "core_min_total_trades": int(SPOT_GATE_HOLDOUT_SANITY.get("core_min_total_trades", 14)),
    "core_total_trades_gate_mult": float(SPOT_GATE_HOLDOUT_SANITY.get("core_total_trades_gate_mult", 1.5)),
    "low_activity_neutralize_ratio": float(SPOT_GATE_HOLDOUT_SANITY.get("low_activity_neutralize_ratio", 0.75)),
    "low_activity_severe_min_return": float(SPOT_GATE_HOLDOUT_SANITY.get("low_activity_severe_min_return", -1.0)),
    "low_activity_severe_min_pf": float(SPOT_GATE_HOLDOUT_SANITY.get("low_activity_severe_min_pf", 0.90)),
    "core_max_avg_mdd_abs": float(SPOT_GATE_HOLDOUT_SANITY.get("core_max_avg_mdd_abs", 2.5)),
}
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
    eval_start = pd.Timestamp(eval_start_time) if eval_start_time is not None else pd.Timestamp(TRAIN_CUTOFF_DATE)
    raw_end = pd.Timestamp(eval_end_time) if eval_end_time is not None else pd.Timestamp(BACKTEST_END_DATE)
    eval_end = _to_inclusive_end_timestamp(raw_end)
    if eval_end < eval_start:
        raise ValueError(f"Invalid eval window: start={eval_start}, end={eval_end}")
    return eval_start, eval_end


def split_bonus_oos_windows() -> Tuple[pd.Timestamp, pd.Timestamp, Optional[pd.Timestamp], Optional[pd.Timestamp], bool, int]:
    data_start, data_end = resolve_eval_window(pd.Timestamp(TRAIN_CUTOFF_DATE), pd.Timestamp(BACKTEST_END_DATE))
    sel_fixed_start = pd.Timestamp(BONUS_SELECTION_FIXED_START_DATE)
    sel_fixed_end = _to_inclusive_end_timestamp(pd.Timestamp(BONUS_SELECTION_FIXED_END_DATE))

    selection_start = max(data_start, sel_fixed_start)
    selection_end = min(data_end, sel_fixed_end)
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


def _normalize_trade_gate_tf(timeframe: Optional[str]) -> str:
    tf = str(timeframe or "").strip().lower()
    if tf in SPOT_TF_HOLDOUT_TRADES_PER_30D:
        return tf
    if tf in SPOT_TF_SELECTION_TRADE_GATES:
        return tf
    return "default"


def _compute_statistical_trade_min_trades(window_days: int) -> int:
    days = max(1, int(window_days))
    confidence = float(np.clip(SPOT_STAT_TRADE_CONFIDENCE, 0.50, 0.99))
    margin_error = float(np.clip(SPOT_STAT_TRADE_MARGIN_ERROR, 0.05, 0.45))
    ref_days = max(1, int(SPOT_STAT_TRADE_REFERENCE_DAYS))
    scale_exp = float(np.clip(SPOT_STAT_TRADE_DAY_SCALE_EXP, 0.25, 1.00))
    min_day_scale = float(np.clip(SPOT_STAT_TRADE_MIN_DAY_SCALE, 0.25, 1.00))

    z = float(NormalDist().inv_cdf(0.5 + (confidence / 2.0)))
    base_samples = int(np.ceil((z * z * 0.25) / (margin_error * margin_error)))
    day_ratio = max(1e-9, float(days) / float(ref_days))
    day_scale = max(min_day_scale, float(day_ratio ** scale_exp))
    scaled_samples = int(np.ceil(float(base_samples) * day_scale))
    return int(max(1, scaled_samples))


def selection_trade_gates_for_timeframe(
    timeframe: Optional[str],
    eval_days: Optional[int] = None,
    primary_symbol_count: int = 2,
) -> Tuple[int, int]:
    tf_key = _normalize_trade_gate_tf(timeframe)
    cfg = SPOT_TF_SELECTION_TRADE_GATES.get(tf_key)
    if cfg is None:
        gates = SELECTION_POLICY["gates"]
        min_trades = int(gates["core_min_trades"])
        min_total_trades = int(gates["core_min_total_trades"])
    else:
        min_trades = int(max(1, cfg.get("core_min_trades", SELECTION_POLICY["gates"]["core_min_trades"])))
        min_total_trades = int(max(min_trades, cfg.get("core_min_total_trades", SELECTION_POLICY["gates"]["core_min_total_trades"])))

    if SPOT_SELECTION_USE_STAT_MIN_TRADES and eval_days is not None and int(eval_days) > 0:
        stat_min_trades = _compute_statistical_trade_min_trades(int(eval_days))
        stat_min_trades = int(min(max(1, stat_min_trades), max(1, SPOT_SELECTION_STAT_MIN_TRADES_CAP)))
        min_trades = int(max(min_trades, stat_min_trades))
        dynamic_total_gate = int(
            np.ceil(
                float(min_trades)
                * max(1, int(primary_symbol_count))
                * max(0.5, float(SPOT_SELECTION_TOTAL_TRADES_MULTIPLIER))
            )
        )
        min_total_trades = int(max(min_total_trades, dynamic_total_gate))
    return min_trades, min_total_trades


def compute_dynamic_holdout_min_trades(
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
    timeframe: Optional[str] = None,
) -> int:
    holdout_days = max(1, int((holdout_end.normalize() - holdout_start.normalize()).days) + 1)
    tf_key = _normalize_trade_gate_tf(timeframe)
    per_30d = float(SPOT_TF_HOLDOUT_TRADES_PER_30D.get(tf_key, SPOT_TF_HOLDOUT_TRADES_PER_30D["default"]))
    floor = int(SPOT_TF_HOLDOUT_MIN_TRADES_FLOOR.get(tf_key, SPOT_TF_HOLDOUT_MIN_TRADES_FLOOR["default"]))
    cap = int(SPOT_TF_HOLDOUT_MIN_TRADES_CAP.get(tf_key, SPOT_TF_HOLDOUT_MIN_TRADES_CAP["default"]))
    scaled = int(round((holdout_days / 30.0) * max(0.1, per_30d)))
    activity_gate = int(max(1, min(max(floor, scaled), max(floor, cap))))
    if not SPOT_HOLDOUT_USE_STAT_MIN_TRADES:
        return activity_gate
    stat_gate = int(min(_compute_statistical_trade_min_trades(holdout_days), max(floor, cap)))
    return int(max(activity_gate, stat_gate))


def _resolve_holdout_trade_gates(trades_gate: int) -> Tuple[int, int]:
    symbol_min_gate = int(max(1, int(HOLDOUT_SANITY_GATES.get("core_min_symbol_trades", trades_gate))))
    total_mult = float(np.clip(float(HOLDOUT_SANITY_GATES.get("core_total_trades_gate_mult", 1.5)), 1.0, 3.0))
    policy_total_gate = int(max(1, int(HOLDOUT_SANITY_GATES.get("core_min_total_trades", symbol_min_gate * 2))))
    dynamic_total_gate = int(np.ceil(float(max(1, trades_gate)) * total_mult))
    total_gate = int(max(policy_total_gate, dynamic_total_gate))
    return symbol_min_gate, total_gate


def evaluate_holdout_sanity(
    summary: Optional[Dict],
    min_trades_gate: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    if not summary:
        return False, ["holdout_no_summary"]
    trades_gate = int(min_trades_gate if min_trades_gate is not None else HOLDOUT_SANITY_GATES["core_min_trades"])
    symbol_min_gate, total_gate = _resolve_holdout_trade_gates(trades_gate)
    reasons: List[str] = []
    core_avg_ret = float(summary.get("core_avg_ret", 0.0))
    core_excess_ret = float(summary.get("core_avg_excess_ret", 0.0))
    core_symbol_min_trades = int(summary.get("core_min_trades", 0))
    core_total_trades = int(summary.get("core_total_trades", core_symbol_min_trades))
    low_activity_ratio = float(core_total_trades) / float(max(1, total_gate))
    neutralize_ratio = float(np.clip(float(HOLDOUT_SANITY_GATES.get("low_activity_neutralize_ratio", 0.75)), 0.40, 1.00))
    is_low_activity = (core_symbol_min_trades < symbol_min_gate) or (core_total_trades < total_gate)
    low_activity_neutral = bool(is_low_activity and (low_activity_ratio < neutralize_ratio))
    severe_min_return = float(HOLDOUT_SANITY_GATES.get("low_activity_severe_min_return", -1.0))
    severe_min_pf = float(HOLDOUT_SANITY_GATES.get("low_activity_severe_min_pf", 0.90))

    dual_return_mode = bool(HOLDOUT_SANITY_GATES.get("core_dual_return_mode", True))
    return_ok = False
    if dual_return_mode:
        return_ok = bool(
            core_avg_ret >= float(HOLDOUT_SANITY_GATES["core_min_return"])
            or core_excess_ret >= float(HOLDOUT_SANITY_GATES.get("core_min_excess_ret", 0.0))
        )
    else:
        return_ok = core_avg_ret >= float(HOLDOUT_SANITY_GATES["core_min_return"])
    if not return_ok and ((not low_activity_neutral) or (core_avg_ret < severe_min_return)):
        reasons.append("holdout_core_return_low")
    if core_symbol_min_trades < symbol_min_gate:
        reasons.append("holdout_core_trades_low")
    if core_total_trades < total_gate:
        reasons.append("holdout_core_total_trades_low")
    if float(summary.get("core_avg_mdd_abs", 0.0)) > float(HOLDOUT_SANITY_GATES["core_max_avg_mdd_abs"]):
        reasons.append("holdout_core_mdd_too_high")
    core_pf_shrunk = float(
        summary.get(
            "core_avg_pf_clipped_shrunk",
            summary.get("core_avg_pf_shrunk", summary.get("core_avg_pf_clipped", summary.get("core_avg_pf", 0.0))),
        )
    )
    if core_pf_shrunk < float(HOLDOUT_SANITY_GATES.get("core_min_pf", 1.0)) and (
        (not low_activity_neutral) or (core_pf_shrunk < severe_min_pf)
    ):
        reasons.append("holdout_core_pf_low")
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
    core_ret = float(summary.get("core_avg_ret", 0.0))
    core_pf_shrunk = float(
        summary.get(
            "core_avg_pf_clipped_shrunk",
            summary.get("core_avg_pf_shrunk", summary.get("core_avg_pf_clipped", summary.get("core_avg_pf", 0.0))),
        )
    )
    reason_set = set(reasons)
    trade_reason_set = {"holdout_core_trades_low", "holdout_core_total_trades_low"}
    neutral_reason_set = {"holdout_core_return_low", "holdout_core_pf_low"} | trade_reason_set

    if core_total_trades <= 0:
        return "INACTIVE", ["holdout_no_trades"]
    if reason_set and reason_set.issubset(trade_reason_set):
        return "INACTIVE", ["holdout_low_activity"]
    if (
        core_total_trades < int(max(1, min_trades_gate))
        and reason_set
        and reason_set.issubset(neutral_reason_set)
        and abs(core_ret) <= 0.35
        and core_pf_shrunk >= 0.95
    ):
        return "INACTIVE", ["holdout_low_activity_near_flat"]
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
    hourly_df, daily_df = load_data_spot(symbol, tf, BACKTEST_START_DATE, BACKTEST_END_DATE)
    if hourly_df is None or daily_df is None:
        print(f"[WARN] Data missing for {symbol}. Skipping.")
        return None

    eval_start_ts, eval_end_ts = resolve_eval_window(eval_start_time, eval_end_time)
    test_hourly, test_daily = _slice_eval_with_warmup(hourly_df, daily_df, eval_start_ts, eval_end_ts)
    if test_hourly.empty:
        print(f"[WARN] No data for {symbol} in eval window. Skipping.")
        return None

    current_params = best_params.copy()

    safe_cost_mult = max(0.0, float(cost_mult))
    current_merge = compute_segment_merge_index(test_hourly, test_daily)
    strategy = UltimateStrategy(f"Verify_{symbol}", current_params)
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
    engine.risk_per_trade = current_params.get("RISK_PER_TRADE_SPOT", 0.99)
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
            params=current_params,
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
            wfa = SpotWalkForwardAnalyzer(test_hourly, test_daily, current_params)
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
    gate_core_min_trades, gate_core_min_total_trades = selection_trade_gates_for_timeframe(
        core_timeframe,
        eval_days=eval_days,
        primary_symbol_count=len(primary),
    )
    up_bh_threshold = 5.0
    down_bh_threshold = -5.0
    core_up_excess = core_excess_returns[core_bh_returns >= up_bh_threshold]
    core_down_excess = core_excess_returns[core_bh_returns <= down_bh_threshold]
    core_up_excess_ret = float(np.mean(core_up_excess)) if core_up_excess.size else 0.0
    core_down_excess_ret = float(np.mean(core_down_excess)) if core_down_excess.size else 0.0
    core_up_sample_count = int(core_up_excess.size)
    core_down_sample_count = int(core_down_excess.size)

    reasons: List[str] = []
    if float(np.mean(core_returns)) < float(gates["core_min_avg_return"]):
        reasons.append("core_avg_return_low")
    if int(np.min(core_trades)) < int(gate_core_min_trades):
        reasons.append("core_trade_count_low")
    if int(core_total_trades) < int(gate_core_min_total_trades):
        reasons.append("core_total_trades_low")
    if float(core_avg_pf_clipped_shrunk) < float(gates["core_min_pf_clipped"]):
        reasons.append("core_pf_low")
    if float(np.mean(core_mdd_abs)) > gates["core_max_avg_mdd_abs"]:
        reasons.append("core_mdd_too_high")

    return {
        "policy_version": SELECTION_POLICY_VERSION,
        "core_avg_ret": float(np.mean(core_returns)),
        "core_avg_bh_ret": float(np.mean(core_bh_returns)),
        "core_avg_excess_ret": float(np.mean(core_excess_returns)),
        "core_avg_mdd_abs": float(np.mean(core_mdd_abs)),
        "core_avg_pf": float(core_avg_pf_raw),
        "core_avg_pf_shrunk": float(core_avg_pf_shrunk),
        "core_avg_pf_clipped": float(core_avg_pf_clipped),
        "core_avg_pf_clipped_shrunk": float(core_avg_pf_clipped_shrunk),
        "core_timeframe": str(core_timeframe),
        "core_min_trades": int(np.min(core_trades)),
        "gate_core_min_trades": int(gate_core_min_trades),
        "gate_core_min_total_trades": int(gate_core_min_total_trades),
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

    score = (
        0.45 * pf_score
        + 0.30 * ret_score
        + 0.15 * mdd_score
        + 0.10 * activity_score
    )
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
                f"[INFO] Holdout dynamic trade gate: timeframe-aware "
                f"(window: {holdout_start} ~ {holdout_end})"
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

                candidate_tf = str(candidate_params.get("TIMEFRAME", "1h")).strip().lower()
                candidate_min_trades_gate = int(
                    compute_dynamic_holdout_min_trades(
                        holdout_start,
                        holdout_end,
                        timeframe=candidate_tf,
                    )
                )
                candidate_symbol_gate, candidate_total_gate = _resolve_holdout_trade_gates(candidate_min_trades_gate)

                print("\n" + "-" * LOG_WIDTH)
                print(f"Holdout Verification for Candidate [{candidate_key}] (used for final selection)")
                print(f"Window: {holdout_start} ~ {holdout_end}")
                print(
                    f"Trade Gate: min_symbol_trades>={candidate_symbol_gate} | "
                    f"core_total_trades>={candidate_total_gate} (tf={candidate_tf})"
                )
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

            final_candidate_key: Optional[str] = None
            for candidate_key, _, _ in ranked:
                ev = holdout_evals.get(candidate_key)
                if ev and ev.get("passed"):
                    final_candidate_key = candidate_key
                    break

            holdout_passed = final_candidate_key is not None
            if holdout_passed:
                winner = str(final_candidate_key)
                winner_src = holdout_evals[winner]["source"]
                selected_holdout_min_trades = int(
                    holdout_evals[winner].get("min_trades_gate", HOLDOUT_SANITY_GATES["core_min_trades"])
                )
                print(f"[INFO] Final winner after holdout gate: {winner}")
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
                print("[WARN] Skip deployment due to failed rolling OOS sanity gate.")
                print("=" * LOG_WIDTH)
                sys.exit(0)

            print("-" * LOG_WIDTH)
            print(
                f"Saving Winner ({winner}) from MySQL "
                f"[{winner_src['db_name']}/{winner_src['study_name']}] to 'spot_strategy.db'..."
            )
            ok = deploy_best_to_local(
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
            if not ok:
                sys.exit(1)
        else:
            print("[WARN] Skip deployment due to failed/missing holdout sanity check.")
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
        print("[INFO] Holdout dynamic trade gate: timeframe-aware")
        assert_strict_oos_window(selection_end, holdout_start, holdout_end)

        check_order = [r[0] for r in ranked] if ranked else list(candidates.keys())
        for mode in check_order:
            c = candidates[mode]
            holdout_params = c["best_params"]
            holdout_study_name = c["study_name"]
            candidate_tf = str(holdout_params.get("TIMEFRAME", "1h")).strip().lower()
            candidate_min_trades_gate = int(
                compute_dynamic_holdout_min_trades(
                    holdout_start,
                    holdout_end,
                    timeframe=candidate_tf,
                )
            )
            candidate_symbol_gate, candidate_total_gate = _resolve_holdout_trade_gates(candidate_min_trades_gate)

            print("\n" + "-" * LOG_WIDTH)
            print(f"Holdout Verification for Candidate [{mode}] (used for final selection)")
            print(f"Window: {holdout_start} ~ {holdout_end}")
            print(
                f"Trade Gate: min_symbol_trades>={candidate_symbol_gate} | "
                f"core_total_trades>={candidate_total_gate} (tf={candidate_tf})"
            )
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
        print("[WARN] Skip deployment due to failed rolling OOS sanity gate.")
        sys.exit(0)

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
    if not ok:
        sys.exit(1)
