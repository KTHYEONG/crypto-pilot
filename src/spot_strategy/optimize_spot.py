import argparse
import gc
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url

try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    sys.path.append(os.getcwd())

from config.optimization_config_modes import GET_SEARCH_SPACE, GET_SPOT_TRADE_GATE_POLICY
from config.settings import BACKTEST_END_DATE, TRAIN_CUTOFF_DATE
from src.optimization.opt_utils import calculate_score, suggest_params
from src.spot_strategy.engine_fast_spot import BacktestEngineFastSpot, backtest_loop_spot_numba
from src.spot_strategy.upbit_client import UpbitClient
from src.strategy.strategies import UltimateStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpotOptimizer")
SHOW_OPTUNA_PROGRESS_BAR = os.getenv("SPOT_SHOW_OPTUNA_PROGRESS_BAR", "0") == "1"

# Use a more recent optimization window to reduce regime drift and stale-history overfit.
SPOT_START_DATE = os.getenv("SPOT_START_DATE", "2020-01-01")
SPOT_INITIAL_BALANCE = 1_000_000.0
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
os.makedirs(DATA_DIR, exist_ok=True)
GAP_FILL_MAX_RANGES = int(os.getenv("SPOT_GAP_FILL_MAX_RANGES", "3"))
GAP_FILL_ENABLED = os.getenv("SPOT_GAP_FILL_ENABLE", "1") == "1"

DAILY_BUFFER_DAYS = 200
WARMUP_BUFFER_BARS = {"5m": 500, "15m": 420, "30m": 350, "1h": 300, "2h": 250, "4h": 200, "1d": 150, "1w": 80}
AWFO_DEFAULTS = {
    "enabled_modes": {"UNIFIED", "ALL"},
    "folds": 3,
    "min_trades_per_fold": 16,
    "min_test_bars": {"5m": 1800, "15m": 1000, "30m": 800, "1h": 560, "2h": 420, "4h": 260, "1d": 140, "1w": 70},
    "embargo_bars": {"5m": 40, "15m": 32, "30m": 24, "1h": 24, "2h": 16, "4h": 12, "1d": 5, "1w": 2},
}

STRUCTURE_PARAM_KEYS = [
    "ENTRY_TYPE",
    "TREND_FILTER_TYPE",
    "TREND_GATE_MODE",
    "STRENGTH_FILTER_TYPE",
    "EXIT_TYPE",
    "STOP_LOSS_TYPE",
    "USE_TAKE_PROFIT",
    "USE_VOLUME_FILTER",
    "TIMEFRAME",
    "USE_DYNAMIC_RISK",
    "ENABLE_SCALE_OUT",
    "ENABLE_BREAKEVEN",
    "ENABLE_PYRAMIDING",
]

# Keep Stage1 simple: explore only core market-structure dimensions.
# Complex execution toggles are deferred to Stage2.
SPOT_STAGE1_STRUCTURE_KEYS = tuple(
    k.strip()
    for k in os.getenv(
        "SPOT_STAGE1_STRUCTURE_KEYS",
        ",".join(
            [
                "ENTRY_TYPE",
                "TREND_FILTER_TYPE",
                "STRENGTH_FILTER_TYPE",
                "EXIT_TYPE",
                "STOP_LOSS_TYPE",
                "USE_TAKE_PROFIT",
                "USE_VOLUME_FILTER",
                "TIMEFRAME",
                "USE_DYNAMIC_RISK",
                "TREND_GATE_MODE",
            ]
        ),
    ).split(",")
    if k.strip()
)

# Spot (long-only, no leverage) tuned defaults.
SPOT_TWO_STAGE_UNIFIED_DEFAULTS = {
    "stage1_total_trials": 2000,
    "stage1_fidelity_steps": [
        {"name": "low", "ratio": 0.45, "symbols": 2, "data_ratio": 0.65, "folds": 2, "min_trades": 12, "startup_ratio": 0.32},
        {"name": "mid", "ratio": 0.33, "symbols": 2, "data_ratio": 0.82, "folds": 3, "min_trades": 14, "startup_ratio": 0.27},
        {"name": "high", "ratio": 0.22, "symbols": 2, "data_ratio": 1.00, "folds": 3, "min_trades": 16, "startup_ratio": 0.22},
    ],
    "promotion_ratio": 0.35,
    "stage2_top_structures": 4,
    "stage2_min_trials_per_structure": 120,
    "stage2_folds": 3,
    "stage2_min_trades": 16,
    "stage2_startup_ratio": 0.18,
    "stage2_refine_ratio": 0.45,
    "stage2_refine_top_quantile": 0.22,
    "stage2_refine_min_width_ratio": 0.22,
    "stage2_refine_min_samples": 28,
    "stage2_refine_step_span": 5,
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


SPOT_GATE_POLICY = GET_SPOT_TRADE_GATE_POLICY()
SPOT_GATE_STAT = dict(SPOT_GATE_POLICY.get("statistical", {}))
SPOT_GATE_HOLDOUT = dict(SPOT_GATE_POLICY.get("holdout", {}))
SPOT_GATE_HOLDOUT_SANITY = dict(SPOT_GATE_HOLDOUT.get("sanity_gates", {}))
HOLDOUT_SANITY_GATES = {
    "core_min_return": float(SPOT_GATE_HOLDOUT_SANITY.get("core_min_return", 0.0)),
    "core_min_trades": int(SPOT_GATE_HOLDOUT_SANITY.get("core_min_trades", 8)),
    "core_max_avg_mdd_abs": float(SPOT_GATE_HOLDOUT_SANITY.get("core_max_avg_mdd_abs", 30.0)),
}
HOLDOUT_MIN_SYMBOL_COVERAGE = float(SPOT_GATE_HOLDOUT_SANITY.get("min_symbol_coverage", 0.67))
SPOT_TF_HOLDOUT_TRADES_PER_30D = dict(SPOT_GATE_HOLDOUT.get("trades_per_30d", {}))
SPOT_TF_HOLDOUT_MIN_TRADES_FLOOR = dict(SPOT_GATE_HOLDOUT.get("min_trades_floor", {}))
SPOT_TF_HOLDOUT_MIN_TRADES_CAP = dict(SPOT_GATE_HOLDOUT.get("min_trades_cap", {}))
SPOT_HOLDOUT_USE_STAT_MIN_TRADES = bool(SPOT_GATE_HOLDOUT.get("use_stat_min_trades", True))
SPOT_STAT_TRADE_CONFIDENCE = float(SPOT_GATE_STAT.get("confidence", 0.80))
SPOT_STAT_TRADE_MARGIN_ERROR = float(SPOT_GATE_STAT.get("margin_error", 0.25))
SPOT_STAT_TRADE_REFERENCE_DAYS = int(SPOT_GATE_STAT.get("reference_days", 120))
SPOT_STAT_TRADE_DAY_SCALE_EXP = float(SPOT_GATE_STAT.get("day_scale_exp", 0.5))
SPOT_STAT_TRADE_MIN_DAY_SCALE = float(SPOT_GATE_STAT.get("min_day_scale", 0.6))



def ensure_database_exists(db_url: str) -> None:
    try:
        url = make_url(db_url)
        db_name = url.database
        if not db_name:
            return

        # Connect to 'mysql' system database to check/create target DB
        admin_url = url.set(database="mysql")
        engine = create_engine(admin_url)
        
        with engine.connect() as conn:
            # Use backticks for safety against special characters in db_name
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
            # Explicit commit is required for DDL in some configurations/drivers
            conn.commit()
    except Exception as e:
        print(f"[WARN] Failed to ensure database exists: {e}")


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
        "2h": 7_200_000,
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


def _parse_seed_list(seed_arg: Optional[str]) -> List[int]:
    if seed_arg is None:
        return []
    raw = [s.strip() for s in str(seed_arg).split(",") if s.strip()]
    seeds: List[int] = []
    for x in raw:
        try:
            seeds.append(int(x))
        except ValueError:
            continue
    uniq: List[int] = []
    seen = set()
    for s in seeds:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _allocate_seed_trials(total_trials: int, seeds: Sequence[int], min_trials_per_seed: int = 80) -> List[Tuple[int, int]]:
    total_trials = int(max(1, total_trials))
    if not seeds:
        return [(13, total_trials)]
    seeds = list(seeds)
    max_seed_count = max(1, total_trials // max(1, int(min_trials_per_seed)))
    active = seeds[:max_seed_count]
    base = total_trials // len(active)
    rem = total_trials % len(active)
    out: List[Tuple[int, int]] = []
    for i, seed in enumerate(active):
        n = base + (1 if i < rem else 0)
        if n > 0:
            out.append((int(seed), int(n)))
    return out or [(int(seeds[0]), total_trials)]


def _study_complete_trials(study: optuna.study.Study) -> List[optuna.trial.FrozenTrial]:
    completed = [tr for tr in study.trials if tr.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda tr: float(tr.value), reverse=True)
    return completed


def _robust_value_from_trials(completed: Sequence[optuna.trial.FrozenTrial]) -> float:
    if not completed:
        return -float("inf")
    vals = np.array([float(tr.value) for tr in completed], dtype=np.float64)
    top = vals[: max(3, min(12, len(vals)))]
    return (0.65 * float(np.mean(top))) + (0.35 * float(np.percentile(top, 25)))


def _robust_value_from_study(study: optuna.study.Study) -> float:
    return _robust_value_from_trials(_study_complete_trials(study))


def _align_to_step(value: float, base: float, step: float, mode: str = "round") -> float:
    if step <= 0:
        return float(value)
    pos = (value - base) / step
    if mode == "ceil":
        snapped = np.ceil(pos) * step + base
    elif mode == "floor":
        snapped = np.floor(pos) * step + base
    else:
        snapped = np.round(pos) * step + base
    s = f"{float(step):.12f}".rstrip("0").rstrip(".")
    digits = len(s.split(".")[1]) if "." in s else 0
    return float(round(float(snapped), digits))


def _sanitize_step_bounds(low: float, high: float, base: float, step: float) -> Tuple[float, float]:
    if step <= 0:
        return float(low), float(high)
    safe_step = max(float(step), 1e-12)
    n_low = int(np.ceil(((float(low) - float(base)) / safe_step) - 1e-10))
    n_high = int(np.floor(((float(high) - float(base)) / safe_step) + 1e-10))
    if n_high < n_low:
        return float(low), float(high)
    s = f"{float(step):.12f}".rstrip("0").rstrip(".")
    digits = len(s.split(".")[1]) if "." in s else 0
    aligned_low = float(round(float(base) + (n_low * safe_step), digits))
    aligned_high = float(round(float(base) + (n_high * safe_step), digits))
    return aligned_low, aligned_high


def build_adaptive_numeric_space(
    base_space: Dict[str, dict],
    completed_trials: Sequence[optuna.trial.FrozenTrial],
    top_quantile: float = 0.24,
    min_width_ratio: float = 0.25,
    min_samples: int = 24,
    min_step_span: int = 4,
) -> Dict[str, dict]:
    if not completed_trials:
        return base_space

    narrowed: Dict[str, dict] = {k: v.copy() for k, v in base_space.items()}
    top_quantile = float(np.clip(top_quantile, 0.05, 0.50))
    min_width_ratio = float(np.clip(min_width_ratio, 0.05, 0.85))
    min_samples = int(max(10, min_samples))
    min_step_span = int(max(2, min_step_span))

    top_n = max(1, int(len(completed_trials) * top_quantile))
    elite = list(completed_trials[:top_n])

    for key, spec in narrowed.items():
        if spec.get("type") not in {"float", "int"}:
            continue
        if "low" not in spec or "high" not in spec:
            continue

        vals: List[float] = []
        for tr in elite:
            if key in tr.params:
                try:
                    vals.append(float(tr.params[key]))
                except (TypeError, ValueError):
                    continue
        if len(vals) < min_samples:
            continue

        base_low = float(spec["low"])
        base_high = float(spec["high"])
        if base_high <= base_low:
            continue

        q20, q50, q80 = np.percentile(vals, [20, 50, 80])
        elite_span = max(float(q80 - q20), 1e-12)
        base_span = base_high - base_low
        min_span = base_span * min_width_ratio
        target_span = max(min_span, elite_span * 1.6)

        if spec.get("type") == "int":
            target_span = max(target_span, float(min_step_span))
        elif "step" in spec:
            step = float(spec["step"])
            target_span = max(target_span, step * float(min_step_span))

        new_low = max(base_low, float(q50) - (target_span / 2.0))
        new_high = min(base_high, float(q50) + (target_span / 2.0))
        if new_high <= new_low:
            continue

        if "step" in spec:
            step = float(spec["step"])
            new_low = _align_to_step(new_low, base_low, step, mode="ceil")
            new_high = _align_to_step(new_high, base_low, step, mode="floor")
            new_low, new_high = _sanitize_step_bounds(new_low, new_high, base_low, step)
            if new_high <= new_low:
                continue
            span_steps = int(round((new_high - new_low) / max(step, 1e-12)))
            if span_steps < min_step_span:
                continue

        if spec.get("type") == "int":
            new_low = int(np.floor(new_low))
            new_high = int(np.ceil(new_high))
            if new_high <= new_low:
                continue
            if (new_high - new_low) < min_step_span:
                continue
            spec["low"] = int(max(int(base_low), new_low))
            spec["high"] = int(min(int(base_high), new_high))
        else:
            spec["low"] = float(max(base_low, new_low))
            spec["high"] = float(min(base_high, new_high))
            if "step" in spec:
                step = float(spec["step"])
                lo, hi = _sanitize_step_bounds(spec["low"], spec["high"], base_low, step)
                if hi <= lo:
                    continue
                spec["low"] = lo
                spec["high"] = hi

    return narrowed


def subset_data_maps(
    base_data_maps: Dict[str, Dict[str, pd.DataFrame]],
    ordered_symbols: Sequence[str],
    n_symbols: int,
    data_ratio: float,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    n_symbols = max(1, min(int(n_symbols), len(ordered_symbols)))
    ratio = float(max(0.10, min(data_ratio, 1.0)))
    selected_symbols = list(ordered_symbols[:n_symbols])

    subset: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sym in selected_symbols:
        subset[sym] = {}
        for tf, df in base_data_maps[sym].items():
            if ratio >= 0.999:
                sliced = df.copy()
            else:
                take_n = max(200, int(len(df) * ratio))
                sliced = df.iloc[-take_n:].copy()
            warm = int(getattr(df, "attrs", {}).get("warmup_bars", 0))
            sliced.attrs["warmup_bars"] = min(warm, len(sliced))
            subset[sym][tf] = sliced
    return subset


def build_stage1_search_space(full_search_space: Dict[str, dict]) -> Dict[str, dict]:
    stage1: Dict[str, dict] = {}
    stage1_keys = [k for k in SPOT_STAGE1_STRUCTURE_KEYS if k in STRUCTURE_PARAM_KEYS]
    for key in stage1_keys:
        if key not in full_search_space:
            continue
        spec = full_search_space[key].copy()
        if spec.get("type") == "categorical":
            stage1[key] = spec
    return stage1


def freeze_structure_in_space(full_search_space: Dict[str, dict], structure_params: Dict[str, object]) -> Dict[str, dict]:
    stage2: Dict[str, dict] = {}
    for key, spec in full_search_space.items():
        stage2[key] = spec.copy()
        if key in structure_params and stage2[key].get("type") == "categorical":
            stage2[key]["choices"] = [structure_params[key]]
    return stage2


def extract_structure_signature(params: Dict[str, object]) -> Dict[str, object]:
    sig: Dict[str, object] = {}
    for key in STRUCTURE_PARAM_KEYS:
        if key in params:
            sig[key] = params[key]
    return sig


def restrict_stage1_space_by_candidates(stage1_space: Dict[str, dict], candidate_structures: Sequence[Dict[str, object]]) -> Dict[str, dict]:
    if not candidate_structures:
        return stage1_space
    restricted: Dict[str, dict] = {}
    for key, spec in stage1_space.items():
        restricted[key] = spec.copy()
        if spec.get("type") == "categorical":
            values = sorted({c[key] for c in candidate_structures if key in c})
            if values:
                restricted[key]["choices"] = values
    return restricted


def _default_structure_signature(stage1_space: Dict[str, dict]) -> Dict[str, object]:
    sig: Dict[str, object] = {}
    for key, spec in stage1_space.items():
        choices = list(spec.get("choices", []))
        if choices:
            sig[key] = choices[0]
    return sig


def run_seeded_studies(
    study_name_prefix: str,
    storage_url: str,
    storage: optuna.storages.RDBStorage,
    n_trials: int,
    n_jobs: int,
    startup_ratio: float,
    objective_fn,
    seeds: Sequence[int],
    min_trials_per_seed: Optional[int] = None,
) -> List[Tuple[int, int, optuna.study.Study, int]]:
    seed_min_trials = int(
        max(
            1,
            min_trials_per_seed
            if min_trials_per_seed is not None
            else _env_int("SPOT_SEED_MIN_TRIALS", 90),
        )
    )
    allocations = _allocate_seed_trials(
        total_trials=int(max(1, n_trials)),
        seeds=seeds,
        min_trials_per_seed=seed_min_trials,
    )
    seed_runs: List[Tuple[int, int, optuna.study.Study, int]] = []
    for seed, seed_trials in allocations:
        seed_study_name = f"{study_name_prefix}__seed_{seed}_{int(time.time()*1000)}"
        try:
            optuna.delete_study(study_name=seed_study_name, storage=storage_url)
        except Exception:
            pass

        startup = max(20, int(seed_trials * max(0.10, startup_ratio)))
        sampler = optuna.samplers.TPESampler(
            seed=int(seed),
            n_startup_trials=startup,
            multivariate=True,
            constant_liar=True,
            warn_independent_sampling=False,
        )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=max(12, startup // 2),
            n_warmup_steps=2,
            interval_steps=1,
        )
        study = optuna.create_study(
            study_name=seed_study_name,
            storage=storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        no_progress = os.environ.get("OPTUNA_NO_PROGRESS", "0") == "1"
        callbacks = []
        if no_progress:
            def status_callback(study, trial):
                completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                try:
                    best_val = study.best_value
                except ValueError:
                    best_val = -float("inf")
                print(f"[STATUS] {seed_study_name} | Trial {completed}/{seed_trials} | Best: {best_val:.4f}", flush=True)
            callbacks.append(status_callback)

        study.optimize(
            objective_fn,
            n_trials=seed_trials,
            n_jobs=n_jobs,
            show_progress_bar=SHOW_OPTUNA_PROGRESS_BAR and not no_progress,
            callbacks=callbacks,
        )
        seed_runs.append((seed, seed_trials, study, startup))
    return seed_runs


def _publish_best_trial_alias(
    storage_url: str,
    alias_study_name: str,
    source_study_name: str,
    source_trial: optuna.trial.FrozenTrial,
    metadata: Optional[Dict] = None,
) -> None:
    try:
        optuna.delete_study(study_name=alias_study_name, storage=storage_url)
    except Exception:
        pass
    optuna.create_study(
        study_name=alias_study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
    )
    alias = optuna.load_study(study_name=alias_study_name, storage=storage_url)
    user_attrs = dict(getattr(source_trial, "user_attrs", {}) or {})
    if metadata:
        user_attrs.update(metadata)
    effective_params = apply_spot_tf_param_limits(dict(source_trial.params))
    frozen = optuna.trial.create_trial(
        params=effective_params,
        distributions=source_trial.distributions,
        value=float(source_trial.value),
        user_attrs=user_attrs,
    )
    alias.add_trial(frozen)
    print(f"[INFO] Published alias study '{alias_study_name}' from source '{source_study_name}'")


def _trial_attr_float(trial: optuna.trial.FrozenTrial, key: str, default: float = 0.0) -> float:
    try:
        return float(getattr(trial, "user_attrs", {}).get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _select_representative_trial(completed: Sequence[optuna.trial.FrozenTrial]) -> optuna.trial.FrozenTrial:
    if not completed:
        raise ValueError("completed trials is empty")
    pool_n = max(5, min(24, len(completed)))
    pool = list(completed[:pool_n])
    best_trial = pool[0]
    best_score = -float("inf")
    for tr in pool:
        value = float(tr.value)
        score_p25 = _trial_attr_float(tr, "score_p25", value)
        ret_p25 = _trial_attr_float(tr, "return_p25", _trial_attr_float(tr, "return_avg", 0.0))
        excess_p25 = _trial_attr_float(tr, "excess_return_p25", ret_p25)
        mdd_avg = _trial_attr_float(tr, "mdd_avg_abs", 0.0)
        mdd_max = _trial_attr_float(tr, "mdd_max_abs", mdd_avg)
        robust = (0.50 * value) + (0.25 * score_p25) + (0.15 * ret_p25) + (0.10 * excess_p25)
        robust -= max(0.0, mdd_avg - 20.0) * 0.90
        robust -= max(0.0, mdd_max - 32.0) * 1.20
        if robust > best_score:
            best_score = robust
            best_trial = tr
    return best_trial


SPOT_ROBUST = {
    "w_avg": _env_float("SPOT_FOLD_W_AVG", 0.30),
    "w_p25": _env_float("SPOT_FOLD_W_P25", 0.40),
    "w_worst": _env_float("SPOT_FOLD_W_WORST", 0.30),
    "cons_target": _env_float("SPOT_FOLD_CONSISTENCY_TARGET", 0.50),
    "cons_penalty": _env_float("SPOT_FOLD_CONSISTENCY_PENALTY", 55.0),
    "cost_stress_per_trade": _env_float("SPOT_COST_STRESS_PER_TRADE_PCT", 0.015),
    "cost_stress_w": _env_float("SPOT_COST_STRESS_WEIGHT", 0.08),
    "ret_p25_w": _env_float("SPOT_FOLD_RET_P25_WEIGHT", 0.20),
    "ret_p25_clip": _env_float("SPOT_FOLD_RET_P25_CLIP", 80.0),
    # Recent-regime alignment: keep some recency signal but avoid overfitting to one phase.
    "recent_score_w": _env_float("SPOT_RECENT_FOLD_SCORE_WEIGHT", 0.00),
    "recent_ret_w": _env_float("SPOT_RECENT_FOLD_RET_WEIGHT", 0.00),
    # Trade-density control: penalize low-activity candidates even if return is high.
    "trade_target_min": _env_int("SPOT_FOLD_TRADE_TARGET_MIN", 12),
    "trade_shortfall_penalty": _env_float("SPOT_FOLD_TRADE_SHORTFALL_PENALTY", 50.0),
    "trade_min_shortfall_penalty": _env_float("SPOT_FOLD_TRADE_MIN_SHORTFALL_PENALTY", 80.0),
    # Penalize single-fold collapses to improve downside robustness in long-only regimes.
    "single_fold_loss_threshold": _env_float("SPOT_SINGLE_FOLD_LOSS_THRESHOLD", -18.0),
    "single_fold_loss_penalty": _env_float("SPOT_SINGLE_FOLD_LOSS_PENALTY", 95.0),
    # Cross-symbol blend (futures-inspired, spot-tailored; no long/short balance term).
    "cross_offset": _env_float("SPOT_CROSS_OFFSET", 240.0),
    "cross_hm_weight": _env_float("SPOT_CROSS_HM_WEIGHT", 0.35),
    "cross_gm_weight": _env_float("SPOT_CROSS_GM_WEIGHT", 0.65),
    "cross_log_dispersion_scale": _env_float("SPOT_CROSS_LOG_DISPERSION_SCALE", 0.22),
    "cross_eff_ret_weight": _env_float("SPOT_CROSS_EFF_RET_WEIGHT", 11.0),
    "cross_eff_pf_weight": _env_float("SPOT_CROSS_EFF_PF_WEIGHT", 4.0),
    "cross_ret_p25_weight": _env_float("SPOT_CROSS_RET_P25_WEIGHT", 10.0),
    "cross_pf_p25_weight": _env_float("SPOT_CROSS_PF_P25_WEIGHT", 4.0),
    "cross_score_p25_weight": _env_float("SPOT_CROSS_SCORE_P25_WEIGHT", 0.08),
    "cross_stress_p25_weight": _env_float("SPOT_CROSS_STRESS_P25_WEIGHT", 0.10),
    "excess_ret_p25_w": _env_float("SPOT_EXCESS_RET_P25_WEIGHT", 0.00),
    "bear_excess_w": _env_float("SPOT_BEAR_EXCESS_WEIGHT", 0.00),
    "cross_excess_p25_weight": _env_float("SPOT_CROSS_EXCESS_P25_WEIGHT", 0.00),
    "cross_excess_avg_weight": _env_float("SPOT_CROSS_EXCESS_AVG_WEIGHT", 0.00),
    # PRIMARY guardrails: avoid selecting "less bad" but negative expectancy candidates.
    "primary_require_symbols": _env_int("SPOT_PRIMARY_REQUIRE_SYMBOLS", 1),
    "primary_avg_pf_floor": _env_float("SPOT_PRIMARY_AVG_PF_FLOOR", 1.00),
    "primary_p25_pf_floor": _env_float("SPOT_PRIMARY_P25_PF_FLOOR", 0.90),
    "primary_avg_ret_floor": _env_float("SPOT_PRIMARY_AVG_RET_FLOOR", -0.30),
    "primary_p25_ret_floor": _env_float("SPOT_PRIMARY_P25_RET_FLOOR", -1.20),
    "primary_avg_pf_penalty_mult": _env_float("SPOT_PRIMARY_AVG_PF_PENALTY_MULT", 50.0),
    "primary_p25_pf_penalty_mult": _env_float("SPOT_PRIMARY_P25_PF_PENALTY_MULT", 35.0),
    "primary_avg_ret_penalty_mult": _env_float("SPOT_PRIMARY_AVG_RET_PENALTY_MULT", 8.0),
    "primary_p25_ret_penalty_mult": _env_float("SPOT_PRIMARY_P25_RET_PENALTY_MULT", 6.0),
    "primary_final_pf_penalty_mult": _env_float("SPOT_PRIMARY_FINAL_PF_PENALTY_MULT", 80.0),
    "primary_final_ret_penalty_mult": _env_float("SPOT_PRIMARY_FINAL_RET_PENALTY_MULT", 12.0),
    "primary_hard_avg_pf_floor": _env_float("SPOT_PRIMARY_HARD_AVG_PF_FLOOR", 0.95),
    "primary_hard_p25_pf_floor": _env_float("SPOT_PRIMARY_HARD_P25_PF_FLOOR", 0.75),
    "primary_hard_avg_ret_floor": _env_float("SPOT_PRIMARY_HARD_AVG_RET_FLOOR", -1.50),
    "primary_hard_p25_ret_floor": _env_float("SPOT_PRIMARY_HARD_P25_RET_FLOOR", -3.00),
    "soft_mdd_avg_center": _env_float("SPOT_SOFT_MDD_AVG_CENTER", 24.0),
    "soft_mdd_avg_mult": _env_float("SPOT_SOFT_MDD_AVG_MULT", 1.4),
    "soft_mdd_max_center": _env_float("SPOT_SOFT_MDD_MAX_CENTER", 36.0),
    "soft_mdd_max_mult": _env_float("SPOT_SOFT_MDD_MAX_MULT", 2.0),
    "collapsed_symbol_penalty": _env_float("SPOT_COLLAPSED_SYMBOL_PENALTY", 90.0),
    # Complexity control: discourage fragile over-engineered parameter sets.
    "complexity_feature_penalty": _env_float("SPOT_COMPLEXITY_FEATURE_PENALTY", 0.0),
    "complexity_scale_out_ratio_center": _env_float("SPOT_COMPLEXITY_SCALE_OUT_RATIO_CENTER", 0.50),
    "complexity_scale_out_ratio_mult": _env_float("SPOT_COMPLEXITY_SCALE_OUT_RATIO_MULT", 0.0),
    "complexity_pyramid_add_penalty": _env_float("SPOT_COMPLEXITY_PYRAMID_ADD_PENALTY", 0.0),
    "complexity_pyramid_risk_center": _env_float("SPOT_COMPLEXITY_PYRAMID_RISK_CENTER", 0.30),
    "complexity_pyramid_risk_mult": _env_float("SPOT_COMPLEXITY_PYRAMID_RISK_MULT", 0.0),
    "trade_density_shortfall_penalty": _env_float("SPOT_TRADE_DENSITY_SHORTFALL_PENALTY", 40.0),
    "trade_density_min_shortfall_penalty": _env_float("SPOT_TRADE_DENSITY_MIN_SHORTFALL_PENALTY", 65.0),
    # P0: prevent low-sample PF explosions and inactive winners.
    "pf_clip_per_symbol": _env_float("SPOT_PF_CLIP_PER_SYMBOL", 6.0),
    "cross_pooled_pf_weight": _env_float("SPOT_CROSS_POOLED_PF_WEIGHT", 5.5),
    "primary_pooled_pf_floor": _env_float("SPOT_PRIMARY_POOLED_PF_FLOOR", 1.00),
    "primary_pooled_pf_hard_floor": _env_float("SPOT_PRIMARY_POOLED_PF_HARD_FLOOR", 0.90),
    "primary_pooled_pf_penalty_mult": _env_float("SPOT_PRIMARY_POOLED_PF_PENALTY_MULT", 75.0),
    "primary_total_trade_target": _env_int("SPOT_PRIMARY_TOTAL_TRADE_TARGET", 44),
    "primary_total_trade_hard_floor": _env_int("SPOT_PRIMARY_TOTAL_TRADE_HARD_FLOOR", 20),
    "primary_total_trade_penalty_mult": _env_float("SPOT_PRIMARY_TOTAL_TRADE_PENALTY_MULT", 150.0),
    "all_total_trade_target": _env_int("SPOT_ALL_TOTAL_TRADE_TARGET", 96),
    "all_total_trade_penalty_mult": _env_float("SPOT_ALL_TOTAL_TRADE_PENALTY_MULT", 60.0),
    # Core objective alignment (P0): absolute return/PF/MDD/trade sufficiency.
    "core_weight_ret": _env_float("SPOT_CORE_WEIGHT_RET", 6.5),
    "core_weight_pooled_pf": _env_float("SPOT_CORE_WEIGHT_POOLED_PF", 70.0),
    "core_weight_pf_p25": _env_float("SPOT_CORE_WEIGHT_PF_P25", 36.0),
    "core_mdd_penalty_mult": _env_float("SPOT_CORE_MDD_PENALTY_MULT", 2.8),
    "core_score_mix": _env_float("SPOT_CORE_SCORE_MIX", 0.85),
    # Risk-budget objective: maximize return while staying near a practical MDD budget.
    "risk_budget_mix": _env_float("SPOT_RISK_BUDGET_MIX", 0.30),
    "risk_budget_target_mdd": _env_float("SPOT_RISK_BUDGET_TARGET_MDD", 5.5),
    "risk_budget_target_band": _env_float("SPOT_RISK_BUDGET_TARGET_BAND", 0.35),
    "risk_budget_low_util_floor": _env_float("SPOT_RISK_BUDGET_LOW_UTIL_FLOOR", 0.55),
    "risk_budget_ret_weight": _env_float("SPOT_RISK_BUDGET_RET_WEIGHT", 14.0),
    "risk_budget_pf_weight": _env_float("SPOT_RISK_BUDGET_PF_WEIGHT", 12.0),
    "risk_budget_center_bonus": _env_float("SPOT_RISK_BUDGET_CENTER_BONUS", 6.0),
    "risk_budget_under_penalty_mult": _env_float("SPOT_RISK_BUDGET_UNDER_PENALTY_MULT", 14.0),
    "risk_budget_over_penalty_mult": _env_float("SPOT_RISK_BUDGET_OVER_PENALTY_MULT", 90.0),
    "risk_budget_hard_mdd_cap": _env_float("SPOT_RISK_BUDGET_HARD_MDD_CAP", 12.0),
    "core_pf_activity_ref_primary_trades": _env_float("SPOT_CORE_PF_ACTIVITY_REF_PRIMARY_TRADES", 36.0),
    "core_pf_activity_ref_all_trades": _env_float("SPOT_CORE_PF_ACTIVITY_REF_ALL_TRADES", 80.0),
    "core_pf_p25_activity_ref_min_trades": _env_float("SPOT_CORE_PF_P25_ACTIVITY_REF_MIN_TRADES", 18.0),
    "core_activity_floor": _env_float("SPOT_CORE_ACTIVITY_FLOOR", 0.05),
    # Low-sample/fake-PF suppression.
    "loss_trade_floor_all": _env_int("SPOT_LOSS_TRADE_FLOOR_ALL", 20),
    "loss_trade_floor_primary": _env_int("SPOT_LOSS_TRADE_FLOOR_PRIMARY", 12),
    "loss_trade_shortfall_penalty_all": _env_float("SPOT_LOSS_TRADE_SHORTFALL_PENALTY_ALL", 140.0),
    "loss_trade_shortfall_penalty_primary": _env_float("SPOT_LOSS_TRADE_SHORTFALL_PENALTY_PRIMARY", 190.0),
    "single_win_share_cap": _env_float("SPOT_SINGLE_WIN_SHARE_CAP", 0.55),
    "single_win_share_penalty_mult": _env_float("SPOT_SINGLE_WIN_SHARE_PENALTY_MULT", 200.0),
    # Regime-balance control: avoid candidates trained only on one-sided (mostly up) folds.
    "regime_up_bh_threshold": _env_float("SPOT_REGIME_UP_BH_THRESHOLD", 5.0),
    "regime_down_bh_threshold": _env_float("SPOT_REGIME_DOWN_BH_THRESHOLD", -5.0),
    "regime_min_up_samples": _env_int("SPOT_REGIME_MIN_UP_SAMPLES", 0),
    "regime_min_down_samples": _env_int("SPOT_REGIME_MIN_DOWN_SAMPLES", 0),
    "regime_missing_penalty": _env_float("SPOT_REGIME_MISSING_PENALTY", 0.0),
    "regime_imbalance_penalty": _env_float("SPOT_REGIME_IMBALANCE_PENALTY", 0.0),
    "regime_imbalance_tolerance": _env_float("SPOT_REGIME_IMBALANCE_TOLERANCE", 0.60),
}

SPOT_PRIMARY_SYMBOLS = tuple(
    s.strip().upper()
    for s in os.getenv("SPOT_PRIMARY_SYMBOLS", "KRW-BTC,KRW-ETH").split(",")
    if s.strip()
)

SPOT_TF_TRADE_DENSITY_TARGET_PER30D = {
    "15m": 10.0,
    "30m": 8.5,
    "1h": 5.0,
    "2h": 3.6,
    "4h": 2.8,
    "1d": 1.4,
    "default": 3.4,
}

SPOT_TF_PARAM_LIMITS = {
    # All bounds are step-compatible to avoid Optuna warning noise from post-adjustment.
    "30m": {
        "ENTRY_PERIOD": {"type": "int", "low": 8, "high": 72, "step": 1},
        "MA_PERIOD": {"type": "int", "low": 8, "high": 96, "step": 1},
        "ATR_PERIOD": {"type": "int", "low": 7, "high": 24, "step": 1},
        "STOP_LOSS_PCT": {"type": "float", "low": 0.008, "high": 0.030, "step": 0.002},
        "ATR_STOP_LOSS_MULT": {"type": "float", "low": 1.2, "high": 3.2, "step": 0.2},
        "TAKE_PROFIT_ATR_MULT": {"type": "float", "low": 1.6, "high": 8.5},
        "ATR_MULTIPLIER": {"type": "float", "low": 1.5, "high": 4.0, "step": 0.3},
        "MAX_HOLDING_BARS": {"type": "int", "low": 24, "high": 220, "step": 1},
        "TRAILING_ACTIVATION_ATR": {"type": "float", "low": 0.5, "high": 2.5, "step": 0.5},
        "TIME_EXIT_PROFIT_THRESHOLD": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.1},
        "RSI_EXIT_THRESHOLD": {"type": "int", "low": 80, "high": 98, "step": 1},
        "ADX_THRESHOLD": {"type": "int", "low": 10, "high": 24, "step": 1},
        "VOLUME_THRESHOLD_MULT": {"type": "float", "low": 0.80, "high": 1.20},
        "VOLUME_MA_PERIOD": {"type": "int", "low": 8, "high": 24, "step": 1},
        "RSI_ENTRY_MAX": {"type": "int", "low": 70, "high": 92, "step": 2},
        "NATR_ENTRY_MIN": {"type": "float", "low": 0.0, "high": 0.6, "step": 0.1},
        "STRENGTH_FILTER_PERIOD": {"type": "int", "low": 8, "high": 24, "step": 1},
        "KELTNER_ATR_MULT": {"type": "float", "low": 1.0, "high": 2.4, "step": 0.1},
        "BB_STD": {"type": "float", "low": 1.6, "high": 3.0, "step": 0.1},
        "SUPERTREND_MULT": {"type": "float", "low": 1.0, "high": 2.4},
        "SUPERTREND_PERIOD": {"type": "int", "low": 5, "high": 20, "step": 1},
        "ICHIMOKU_TENKAN": {"type": "int", "low": 7, "high": 18, "step": 1},
        "ICHIMOKU_KIJUN": {"type": "int", "low": 18, "high": 36, "step": 1},
        "ICHIMOKU_SENKOU_B": {"type": "int", "low": 34, "high": 68, "step": 1},
        "SAR_STEP": {"type": "float", "low": 0.005, "high": 0.030, "step": 0.005},
    },
    "1h": {
        "ENTRY_PERIOD": {"type": "int", "low": 10, "high": 96, "step": 1},
        "MA_PERIOD": {"type": "int", "low": 10, "high": 130, "step": 1},
        "ATR_PERIOD": {"type": "int", "low": 8, "high": 28, "step": 1},
        "STOP_LOSS_PCT": {"type": "float", "low": 0.008, "high": 0.034, "step": 0.002},
        "ATR_STOP_LOSS_MULT": {"type": "float", "low": 1.2, "high": 3.6, "step": 0.2},
        "TAKE_PROFIT_ATR_MULT": {"type": "float", "low": 1.6, "high": 10.0},
        "ATR_MULTIPLIER": {"type": "float", "low": 1.6, "high": 4.4, "step": 0.3},
        "MAX_HOLDING_BARS": {"type": "int", "low": 18, "high": 260, "step": 1},
        "TRAILING_ACTIVATION_ATR": {"type": "float", "low": 0.5, "high": 3.0, "step": 0.5},
        "TIME_EXIT_PROFIT_THRESHOLD": {"type": "float", "low": 0.0, "high": 1.2, "step": 0.1},
        "RSI_EXIT_THRESHOLD": {"type": "int", "low": 80, "high": 98, "step": 1},
        "ADX_THRESHOLD": {"type": "int", "low": 10, "high": 26, "step": 1},
        "VOLUME_THRESHOLD_MULT": {"type": "float", "low": 0.80, "high": 1.25},
        "VOLUME_MA_PERIOD": {"type": "int", "low": 8, "high": 28, "step": 1},
        "RSI_ENTRY_MAX": {"type": "int", "low": 72, "high": 94, "step": 2},
        "NATR_ENTRY_MIN": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
        "STRENGTH_FILTER_PERIOD": {"type": "int", "low": 8, "high": 28, "step": 1},
        "KELTNER_ATR_MULT": {"type": "float", "low": 1.0, "high": 2.8, "step": 0.1},
        "BB_STD": {"type": "float", "low": 1.6, "high": 3.2, "step": 0.1},
        "SUPERTREND_MULT": {"type": "float", "low": 1.0, "high": 2.8},
        "SUPERTREND_PERIOD": {"type": "int", "low": 6, "high": 24, "step": 1},
        "ICHIMOKU_TENKAN": {"type": "int", "low": 7, "high": 24, "step": 1},
        "ICHIMOKU_KIJUN": {"type": "int", "low": 20, "high": 42, "step": 1},
        "ICHIMOKU_SENKOU_B": {"type": "int", "low": 40, "high": 78, "step": 1},
        "SAR_STEP": {"type": "float", "low": 0.005, "high": 0.040, "step": 0.005},
    },
    "4h": {
        "ENTRY_PERIOD": {"type": "int", "low": 14, "high": 140, "step": 1},
        "MA_PERIOD": {"type": "int", "low": 16, "high": 170, "step": 1},
        "ATR_PERIOD": {"type": "int", "low": 10, "high": 28, "step": 1},
        "STOP_LOSS_PCT": {"type": "float", "low": 0.010, "high": 0.036, "step": 0.002},
        "ATR_STOP_LOSS_MULT": {"type": "float", "low": 1.4, "high": 4.0, "step": 0.2},
        "TAKE_PROFIT_ATR_MULT": {"type": "float", "low": 1.8, "high": 12.0},
        "ATR_MULTIPLIER": {"type": "float", "low": 1.8, "high": 4.8, "step": 0.3},
        "MAX_HOLDING_BARS": {"type": "int", "low": 8, "high": 180, "step": 1},
        "TRAILING_ACTIVATION_ATR": {"type": "float", "low": 0.5, "high": 3.5, "step": 0.5},
        "TIME_EXIT_PROFIT_THRESHOLD": {"type": "float", "low": 0.0, "high": 1.4, "step": 0.1},
        "RSI_EXIT_THRESHOLD": {"type": "int", "low": 82, "high": 98, "step": 1},
        "ADX_THRESHOLD": {"type": "int", "low": 12, "high": 26, "step": 1},
        "VOLUME_THRESHOLD_MULT": {"type": "float", "low": 0.85, "high": 1.30},
        "VOLUME_MA_PERIOD": {"type": "int", "low": 10, "high": 30, "step": 1},
        "RSI_ENTRY_MAX": {"type": "int", "low": 74, "high": 94, "step": 2},
        "NATR_ENTRY_MIN": {"type": "float", "low": 0.0, "high": 0.4, "step": 0.1},
        "STRENGTH_FILTER_PERIOD": {"type": "int", "low": 10, "high": 28, "step": 1},
        "KELTNER_ATR_MULT": {"type": "float", "low": 1.1, "high": 2.8, "step": 0.1},
        "BB_STD": {"type": "float", "low": 1.7, "high": 3.2, "step": 0.1},
        "SUPERTREND_MULT": {"type": "float", "low": 1.1, "high": 2.8},
        "SUPERTREND_PERIOD": {"type": "int", "low": 8, "high": 24, "step": 1},
        "ICHIMOKU_TENKAN": {"type": "int", "low": 9, "high": 24, "step": 1},
        "ICHIMOKU_KIJUN": {"type": "int", "low": 24, "high": 42, "step": 1},
        "ICHIMOKU_SENKOU_B": {"type": "int", "low": 40, "high": 78, "step": 1},
        "SAR_STEP": {"type": "float", "low": 0.005, "high": 0.040, "step": 0.005},
    },
}


def _clip_tf_int(value: object, low: int, high: int, step: int = 1) -> int:
    x = int(round(float(value)))
    x = max(int(low), min(int(high), x))
    s = max(1, int(step))
    x = int(low + (round((x - int(low)) / s) * s))
    return max(int(low), min(int(high), int(x)))


def _clip_tf_float(value: object, low: float, high: float, step: Optional[float] = None) -> float:
    x = float(value)
    x = max(float(low), min(float(high), x))
    if step is None or float(step) <= 0.0:
        return float(x)
    return float(
        max(
            float(low),
            min(
                float(high),
                _align_to_step(float(x), float(low), float(step), mode="round"),
            ),
        )
    )


def apply_spot_tf_param_limits(params: Dict[str, object]) -> Dict[str, object]:
    tf = str(params.get("TIMEFRAME", "")).strip().lower()
    limits = SPOT_TF_PARAM_LIMITS.get(tf)
    if not limits:
        return params
    out = dict(params)
    for key, rule in limits.items():
        if key not in out:
            continue
        v = out.get(key)
        if v is None or isinstance(v, bool):
            continue
        typ = str(rule.get("type", "")).strip().lower()
        if typ == "int":
            out[key] = int(
                _clip_tf_int(
                    v,
                    int(rule["low"]),
                    int(rule["high"]),
                    int(rule.get("step", 1)),
                )
            )
        elif typ == "float":
            out[key] = float(
                _clip_tf_float(
                    v,
                    float(rule["low"]),
                    float(rule["high"]),
                    float(rule["step"]) if rule.get("step") is not None else None,
                )
            )
    return out


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
            parquet_fp = _spot_single_cache_path(symbol, tf)
            df: Optional[pd.DataFrame] = None
            if os.path.exists(parquet_fp):
                try:
                    df = pd.read_parquet(parquet_fp)
                    if "timestamp" in df.columns:
                        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                        df.sort_values("timestamp", inplace=True)
                        df.drop_duplicates(subset=["timestamp"], inplace=True)
                        df.reset_index(drop=True, inplace=True)
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
            fetch_start = start_ts
            fetch_end = end_ts
            if df is None or df.empty:
                need_fetch = True
            else:
                cached_start = int(df["timestamp"].min())
                cached_end = int(df["timestamp"].max())
                # Spot symbols may list after requested start_date.
                # Avoid repeated backward refetch for pre-listing gaps; only fetch missing tail.
                if cached_end < end_ts:
                    need_fetch = True
                    fetch_start = max(start_ts, cached_end + 1)

            if need_fetch:
                print(f"[INFO] Downloading {symbol}-{tf}...")
                fetched = client.fetch_ohlcv(symbol, tf, since=fetch_start, end=fetch_end)
                if fetched is None or fetched.empty:
                    print(f"[ERROR] Empty data: {symbol}-{tf}")
                    sys.exit(1)
                fetched.sort_values("timestamp", inplace=True)
                fetched["datetime"] = pd.to_datetime(fetched["timestamp"], unit="ms")
                fetched.reset_index(drop=True, inplace=True)
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

            # Internal gap backfill (middle missing candles)
            if GAP_FILL_ENABLED and df is not None and not df.empty:
                step_ms = _spot_timeframe_to_ms(tf)
                if step_ms is not None:
                    gaps = _find_internal_gap_ranges(df, start_ts, end_ts, step_ms)
                    if gaps:
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
                            try:
                                fallback_csv = parquet_fp.replace(".parquet", ".csv")
                                df.to_csv(fallback_csv, index=False)
                            except Exception:
                                pass

            if df is None or df.empty:
                print(f"[ERROR] Empty cache after fetch: {symbol}-{tf}")
                sys.exit(1)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df[(df["datetime"] >= pd.Timestamp(start_date)) & (df["datetime"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1))].copy()
            df.reset_index(drop=True, inplace=True)
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
                        "oos_start_idx": int(test_start - seg_start),
                        "oos_end_idx": int((test_end - 1) - seg_start),
                    }
                )
            cached[symbol][tf] = folds_ctx
    return cached


def _filter_trades_for_window(trades_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    has_entry = "entry_time" in df.columns
    has_exit = "exit_time" in df.columns
    if has_entry:
        df["entry_time"] = pd.to_datetime(df["entry_time"])
    if has_exit:
        df["exit_time"] = pd.to_datetime(df["exit_time"])
    if has_entry and has_exit:
        return df[(df["entry_time"] <= end) & (df["exit_time"] >= start)].copy()
    if has_entry:
        return df[(df["entry_time"] >= start) & (df["entry_time"] <= end)].copy()
    if has_exit:
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


def calculate_oos_metrics_from_equity(
    equity_curve: Optional[Sequence[float]],
    start_idx: int,
    end_idx: int,
    initial_balance: float,
) -> Optional[Tuple[float, float]]:
    if equity_curve is None:
        return None
    eq = np.asarray(equity_curve, dtype=np.float64)
    if eq.size == 0:
        return None
    s = int(np.clip(int(start_idx), 0, eq.size - 1))
    e = int(np.clip(int(end_idx), s, eq.size - 1))
    start_equity = float(eq[s - 1]) if s > 0 else float(initial_balance)
    end_equity = float(eq[e])
    ret_pct = float((end_equity - start_equity) / float(initial_balance) * 100.0)
    seg = eq[s : e + 1]
    if seg.size == 0:
        return ret_pct, 0.0
    run_max = np.maximum.accumulate(seg)
    run_max[run_max <= 0.0] = 1e-9
    dd = (seg - run_max) / run_max * 100.0
    mdd_pct = float(np.min(dd)) if dd.size else 0.0
    return ret_pct, mdd_pct


def calculate_oos_benchmark_return_pct(
    hourly_df: Optional[pd.DataFrame],
    start_idx: int,
    end_idx: int,
) -> float:
    if hourly_df is None or hourly_df.empty or "close" not in hourly_df.columns:
        return 0.0
    close = pd.to_numeric(hourly_df["close"], errors="coerce").to_numpy(dtype=np.float64)
    if close.size == 0:
        return 0.0
    s = int(np.clip(int(start_idx), 0, close.size - 1))
    e = int(np.clip(int(end_idx), s, close.size - 1))
    start_px = float(close[s])
    end_px = float(close[e])
    if (not np.isfinite(start_px)) or (not np.isfinite(end_px)) or start_px <= 0.0:
        return 0.0
    return float(((end_px / start_px) - 1.0) * 100.0)


def _normalize_holdout_tf_key(timeframe: Optional[str]) -> str:
    tf = str(timeframe or "").strip().lower()
    return tf if tf in SPOT_TF_HOLDOUT_TRADES_PER_30D else "default"


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


def compute_dynamic_holdout_min_trades(
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
    timeframe: Optional[str] = None,
) -> int:
    days = max(1, int((holdout_end.normalize() - holdout_start.normalize()).days) + 1)
    tf_key = _normalize_holdout_tf_key(timeframe)
    per_30d = float(SPOT_TF_HOLDOUT_TRADES_PER_30D.get(tf_key, SPOT_TF_HOLDOUT_TRADES_PER_30D["default"]))
    floor = int(SPOT_TF_HOLDOUT_MIN_TRADES_FLOOR.get(tf_key, SPOT_TF_HOLDOUT_MIN_TRADES_FLOOR["default"]))
    cap = int(SPOT_TF_HOLDOUT_MIN_TRADES_CAP.get(tf_key, SPOT_TF_HOLDOUT_MIN_TRADES_CAP["default"]))
    scaled = int(round((days / 30.0) * max(0.1, per_30d)))
    activity_gate = int(max(1, min(max(floor, scaled), max(floor, cap))))
    if not SPOT_HOLDOUT_USE_STAT_MIN_TRADES:
        return activity_gate
    stat_gate = int(min(_compute_statistical_trade_min_trades(days), max(floor, cap)))
    return int(max(activity_gate, stat_gate))


def evaluate_holdout_sanity(summary: Optional[Dict], min_trades_gate: Optional[int] = None) -> Tuple[bool, List[str]]:
    if not summary:
        return False, ["holdout_no_summary"]
    trades_gate = int(min_trades_gate if min_trades_gate is not None else HOLDOUT_SANITY_GATES["core_min_trades"])
    reasons: List[str] = []
    symbols_total = int(summary.get("symbols_total", 0))
    symbols_evaluated = int(summary.get("symbols_evaluated", 0))
    coverage = float(symbols_evaluated) / float(max(1, symbols_total))
    if float(summary.get("core_avg_ret", 0.0)) <= float(HOLDOUT_SANITY_GATES["core_min_return"]):
        reasons.append("holdout_core_return_low")
    core_trade_count = int(
        summary.get(
            "core_primary_total_trades",
            summary.get("core_total_trades", summary.get("core_min_trades", 0)),
        )
    )
    if core_trade_count < trades_gate:
        reasons.append("holdout_core_trades_low")
    if float(summary.get("core_avg_mdd_abs", 0.0)) > float(HOLDOUT_SANITY_GATES["core_max_avg_mdd_abs"]):
        reasons.append("holdout_core_mdd_too_high")
    if coverage < float(HOLDOUT_MIN_SYMBOL_COVERAGE):
        reasons.append("holdout_symbol_coverage_low")
    return len(reasons) == 0, reasons


def classify_holdout_outcome(
    summary: Optional[Dict],
    passed: bool,
    reasons: List[str],
    min_trades_gate: int,
) -> Tuple[str, List[str]]:
    if passed:
        return "PASS", []
    if not summary:
        return "FAIL", list(reasons)

    core_total_trades = int(summary.get("core_total_trades", summary.get("core_min_trades", 0)))
    core_ret = float(summary.get("core_avg_ret", 0.0))
    reason_set = set(reasons)
    trade_reason_set = {"holdout_core_trades_low", "holdout_core_total_trades_low"}
    neutral_reason_set = {"holdout_core_return_low"} | trade_reason_set

    if core_total_trades <= 0:
        return "INACTIVE", ["holdout_no_trades"]
    if reason_set and reason_set.issubset(trade_reason_set):
        return "INACTIVE", ["holdout_low_activity"]
    if (
        core_total_trades < int(max(1, min_trades_gate))
        and reason_set
        and reason_set.issubset(neutral_reason_set)
        and abs(core_ret) <= 0.20
    ):
        return "INACTIVE", ["holdout_low_activity_near_flat"]
    return "FAIL", list(reasons)


def run_holdout_gate(
    best_params: Dict,
    symbols_data: Dict[str, Dict[str, pd.DataFrame]],
    mode: str,
    cutoff_ts: pd.Timestamp,
) -> Tuple[bool, Dict[str, float], List[str]]:
    tf = str(best_params.get("TIMEFRAME", "1h"))
    primary_symbol_set = set(SPOT_PRIMARY_SYMBOLS)
    returns: List[float] = []
    mdd_abs_vals: List[float] = []
    pfs: List[float] = []
    trades_min: List[int] = []
    trades_total: List[int] = []
    primary_total_trades = 0
    holdout_end_candidates: List[pd.Timestamp] = []

    for symbol, data_map in symbols_data.items():
        hourly_df = data_map.get(tf)
        daily_df = data_map.get("1d")
        if hourly_df is None or hourly_df.empty or daily_df is None or daily_df.empty:
            continue

        dt = pd.to_datetime(hourly_df["datetime"])
        mask = (dt >= cutoff_ts).to_numpy(dtype=bool)
        if not np.any(mask):
            continue
        first_oos_idx = int(np.argmax(mask))
        seg_start = max(0, first_oos_idx - WARMUP_BUFFER_BARS.get(tf, 200))
        segment_hourly = hourly_df.iloc[seg_start:].copy()
        if segment_hourly.empty:
            continue
        oos_start_idx = int(first_oos_idx - seg_start)
        oos_end_idx = int(len(segment_hourly) - 1)
        segment_hourly.attrs["warmup_bars"] = oos_start_idx

        seg_end_time = pd.Timestamp(segment_hourly["datetime"].iloc[-1])
        holdout_end_candidates.append(seg_end_time)
        daily_start = pd.Timestamp(segment_hourly["datetime"].iloc[0]) - pd.Timedelta(days=DAILY_BUFFER_DAYS)
        segment_daily = daily_df[(daily_df["datetime"] >= daily_start) & (daily_df["datetime"] <= seg_end_time)].copy()
        if segment_daily.empty:
            continue

        try:
            strategy = UltimateStrategy(f"Holdout_{symbol}", best_params)
            engine = BacktestEngineFastSpot(
                segment_hourly,
                segment_daily,
                strategy,
                backtest_loop_spot_numba,
                initial_balance=SPOT_INITIAL_BALANCE,
                fee_rate=0.0005,
                slippage_rate=0.0003,
                merge_index_map=compute_segment_merge_index(segment_hourly, segment_daily),
            )
            engine.risk_per_trade = best_params.get("RISK_PER_TRADE_SPOT", 0.99)
            result = engine.run()
        except Exception:
            continue

        oos_metrics = calculate_oos_metrics_from_equity(
            result.get("equity_curve"),
            oos_start_idx,
            oos_end_idx,
            SPOT_INITIAL_BALANCE,
        )
        if oos_metrics is None:
            continue
        ret_pct, mdd_pct = oos_metrics

        oos_trades = _filter_trades_for_window(
            result.get("trades_df", pd.DataFrame()),
            pd.Timestamp(cutoff_ts),
            seg_end_time,
        )
        trade_count = int(len(oos_trades))

        pf = 0.0
        if "pnl" in oos_trades.columns:
            gross_profit = float(oos_trades[oos_trades["pnl"] > 0]["pnl"].sum())
            gross_loss = abs(float(oos_trades[oos_trades["pnl"] < 0]["pnl"].sum()))
            pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

        _ = calculate_score(ret_pct, mdd_pct, oos_trades, mode=mode, market_type="spot", timeframe=tf)
        returns.append(float(ret_pct))
        mdd_abs_vals.append(abs(float(mdd_pct)))
        pfs.append(float(pf))
        trades_min.append(trade_count)
        trades_total.append(trade_count)
        if str(symbol).strip().upper() in primary_symbol_set:
            primary_total_trades += int(trade_count)

    if not returns:
        return False, {}, ["holdout_no_valid_symbols"]

    holdout_end_ts = max(holdout_end_candidates) if holdout_end_candidates else pd.Timestamp(BACKTEST_END_DATE)
    dynamic_min_trades = compute_dynamic_holdout_min_trades(
        pd.Timestamp(cutoff_ts),
        holdout_end_ts,
        timeframe=tf,
    )
    summary = {
        "core_avg_ret": float(np.mean(returns)),
        "core_avg_mdd_abs": float(np.mean(mdd_abs_vals)),
        "core_avg_pf": float(np.mean(pfs)) if pfs else 0.0,
        "core_min_trades": int(np.min(np.asarray(trades_min, dtype=np.int64))),
        "core_total_trades": int(np.sum(np.asarray(trades_total, dtype=np.int64))) if trades_total else 0,
        "core_primary_total_trades": int(primary_total_trades),
        "symbols_evaluated": int(len(returns)),
        "symbols_total": int(len(symbols_data)),
        "dynamic_min_trades_gate": int(dynamic_min_trades),
        "timeframe": str(tf),
    }
    passed, reasons = evaluate_holdout_sanity(summary, min_trades_gate=dynamic_min_trades)
    return passed, summary, reasons


def objective(
    trial: optuna.trial.Trial,
    symbols_data: Dict[str, Dict[str, pd.DataFrame]],
    search_space: Dict,
    mode: str = "UNIFIED",
    merge_indices: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    awfo_plan: Optional[Dict] = None,
) -> float:
    params = suggest_params(trial, search_space)
    params = apply_spot_tf_param_limits(params)
    tf = params.get("TIMEFRAME", "1h")
    if params.get("TREND_FILTER_TYPE") == "MACD" and params.get("MACD_FAST", 12) >= params.get("MACD_SLOW", 26):
        return -10000.0

    awfo_enabled = bool(awfo_plan and awfo_plan.get("enabled"))
    awfo_cache = awfo_plan.get("cache", {}) if awfo_enabled else {}
    awfo_min_trades = awfo_plan.get("min_trades_per_fold", AWFO_DEFAULTS["min_trades_per_fold"]) if awfo_enabled else None
    symbol_scores: List[float] = []
    symbol_results: Dict[str, Dict[str, float]] = {}
    report_step = 0
    fallback_count = 0
    primary_symbol_set = set(SPOT_PRIMARY_SYMBOLS)

    def fallback(sym: str, reason: str) -> None:
        nonlocal fallback_count
        print(f"[WARN] Spot fallback: {sym} ({reason})")
        fallback_count += 1
        symbol_scores.append(-260.0)
        symbol_results[sym] = {
            "return": -35.0,
            "mdd": -70.0,
            "pf": 0.0,
            "stress_ret": -40.0,
            "bh_ret": 0.0,
            "excess_ret": -35.0,
            "trades_total": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "loss_trades": 0,
            "win_trades": 0,
            "max_win_pnl": 0.0,
        }

    for symbol, data_map in symbols_data.items():
        is_primary_symbol = str(symbol).strip().upper() in primary_symbol_set
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
            fold_trade_counts: List[int] = []
            fold_trade_density: List[float] = []
            fold_bh_returns: List[float] = []
            fold_excess_returns: List[float] = []
            fold_bear_excess: List[float] = []
            fold_gross_profit_sum = 0.0
            fold_gross_loss_sum = 0.0
            fold_loss_trades_sum = 0
            fold_win_trades_sum = 0
            fold_max_win_pnl = 0.0
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
                oos_metrics = calculate_oos_metrics_from_equity(
                    result.get("equity_curve"),
                    int(ctx.get("oos_start_idx", 0)),
                    int(ctx.get("oos_end_idx", len(ctx["hourly"]) - 1)),
                    SPOT_INITIAL_BALANCE,
                )
                if oos_metrics is None or oos_trades.empty or "pnl" not in oos_trades.columns:
                    invalid += 1
                else:
                    fold_ret, fold_mdd = oos_metrics
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
                        trade_count = int(len(oos_trades))
                        fold_trade_counts.append(trade_count)
                        fold_days = max(
                            1,
                            int(
                                (
                                    pd.Timestamp(ctx["actual_end_time"]).normalize()
                                    - pd.Timestamp(ctx["actual_start_time"]).normalize()
                                ).days
                            )
                            + 1,
                        )
                        fold_trade_density.append((float(trade_count) / float(fold_days)) * 30.0)
                        bh_ret = calculate_oos_benchmark_return_pct(
                            ctx.get("hourly"),
                            int(ctx.get("oos_start_idx", 0)),
                            int(ctx.get("oos_end_idx", len(ctx["hourly"]) - 1)),
                        )
                        fold_bh_returns.append(float(bh_ret))
                        fold_excess = float(fold_ret - bh_ret)
                        fold_excess_returns.append(fold_excess)
                        if bh_ret < 0.0:
                            fold_bear_excess.append(fold_excess)
                        gross_profit = float(oos_trades[oos_trades["pnl"] > 0]["pnl"].sum())
                        gross_loss = abs(float(oos_trades[oos_trades["pnl"] < 0]["pnl"].sum()))
                        win_trades = int((oos_trades["pnl"] > 0).sum())
                        loss_trades = int((oos_trades["pnl"] < 0).sum())
                        fold_win_trades_sum += win_trades
                        fold_loss_trades_sum += loss_trades
                        if win_trades > 0:
                            fold_max_win_pnl = max(
                                fold_max_win_pnl,
                                float(oos_trades.loc[oos_trades["pnl"] > 0, "pnl"].max()),
                            )
                        fold_pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
                        fold_pfs.append(float(np.clip(fold_pf, 0.0, float(SPOT_ROBUST["pf_clip_per_symbol"]))))
                        fold_gross_profit_sum += gross_profit
                        fold_gross_loss_sum += gross_loss
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
            p25_excess = float(np.percentile(fold_excess_returns, 25)) if fold_excess_returns else p25_ret
            bear_excess = float(np.mean(fold_bear_excess)) if fold_bear_excess else p25_excess
            consistency = float(np.mean(np.array(fold_scores) > 0))
            score = (SPOT_ROBUST["w_avg"] * avg) + (SPOT_ROBUST["w_p25"] * p25) + (SPOT_ROBUST["w_worst"] * worst)
            score += SPOT_ROBUST["ret_p25_w"] * np.clip(p25_ret, -SPOT_ROBUST["ret_p25_clip"], SPOT_ROBUST["ret_p25_clip"])
            score += SPOT_ROBUST["cost_stress_w"] * np.clip(p25_stress, -60.0, 60.0)
            score += SPOT_ROBUST["excess_ret_p25_w"] * np.clip(p25_excess, -70.0, 90.0)
            score += SPOT_ROBUST["bear_excess_w"] * np.clip(bear_excess, -60.0, 80.0)
            # Recency-weighted fold score/return (later folds are closer to holdout regime).
            if len(fold_scores) >= 2:
                w = np.arange(1, len(fold_scores) + 1, dtype=np.float64)
                w = w / np.sum(w)
                recent_score = float(np.dot(np.asarray(fold_scores, dtype=np.float64), w))
                recent_ret = float(np.dot(np.asarray(fold_returns, dtype=np.float64), w))
                score += SPOT_ROBUST["recent_score_w"] * recent_score
                score += SPOT_ROBUST["recent_ret_w"] * np.clip(recent_ret, -80.0, 120.0)
            if consistency < SPOT_ROBUST["cons_target"]:
                score -= (SPOT_ROBUST["cons_target"] - consistency) * SPOT_ROBUST["cons_penalty"]
            if fold_returns and min(fold_returns) < SPOT_ROBUST["single_fold_loss_threshold"]:
                score -= SPOT_ROBUST["single_fold_loss_penalty"]
            # Trade-density penalty to avoid "few-trade overfit winners" that fail holdout gate.
            if fold_trade_counts:
                target = max(1.0, float(max(int(awfo_min_trades or 0), int(SPOT_ROBUST["trade_target_min"]))))
                avg_trades = float(np.mean(fold_trade_counts))
                min_trades = float(np.min(fold_trade_counts))
                avg_shortfall = max(0.0, (target - avg_trades) / target)
                min_shortfall = max(0.0, (target - min_trades) / target)
                score -= avg_shortfall * SPOT_ROBUST["trade_shortfall_penalty"]
                score -= min_shortfall * SPOT_ROBUST["trade_min_shortfall_penalty"]
            if fold_trade_density:
                density_target = float(
                    SPOT_TF_TRADE_DENSITY_TARGET_PER30D.get(
                        str(tf),
                        SPOT_TF_TRADE_DENSITY_TARGET_PER30D["default"],
                    )
                )
                density_target = max(0.5, density_target)
                avg_density = float(np.mean(fold_trade_density))
                min_density = float(np.min(fold_trade_density))
                avg_density_shortfall = max(0.0, (density_target - avg_density) / density_target)
                min_density_floor = max(0.5, density_target * 0.7)
                min_density_shortfall = max(0.0, (min_density_floor - min_density) / min_density_floor)
                score -= avg_density_shortfall * float(SPOT_ROBUST["trade_density_shortfall_penalty"])
                score -= min_density_shortfall * float(SPOT_ROBUST["trade_density_min_shortfall_penalty"])
            if fold_bh_returns:
                up_thr = float(SPOT_ROBUST["regime_up_bh_threshold"])
                down_thr = float(SPOT_ROBUST["regime_down_bh_threshold"])
                up_count = int(np.sum(np.asarray(fold_bh_returns, dtype=np.float64) >= up_thr))
                down_count = int(np.sum(np.asarray(fold_bh_returns, dtype=np.float64) <= down_thr))
                min_up = int(SPOT_ROBUST["regime_min_up_samples"])
                min_down = int(SPOT_ROBUST["regime_min_down_samples"])
                missing_up = max(0, min_up - up_count)
                missing_down = max(0, min_down - down_count)
                if (missing_up + missing_down) > 0:
                    score -= float(SPOT_ROBUST["regime_missing_penalty"]) * float(missing_up + missing_down)
                regime_total = max(1, up_count + down_count)
                if up_count > 0 and down_count > 0:
                    imbalance = abs(float(up_count - down_count)) / float(regime_total)
                    tol = float(SPOT_ROBUST["regime_imbalance_tolerance"])
                    if imbalance > tol:
                        score -= (imbalance - tol) * float(SPOT_ROBUST["regime_imbalance_penalty"]) * 10.0
            if is_primary_symbol and fold_pfs and fold_returns:
                avg_pf_sym = float(np.mean(fold_pfs))
                p25_pf_sym = float(np.percentile(np.asarray(fold_pfs, dtype=np.float64), 25))
                avg_ret_sym = float(np.mean(fold_returns))
                p25_ret_sym = float(np.percentile(np.asarray(fold_returns, dtype=np.float64), 25))
                score -= max(0.0, float(SPOT_ROBUST["primary_avg_pf_floor"]) - avg_pf_sym) * float(
                    SPOT_ROBUST["primary_avg_pf_penalty_mult"]
                )
                score -= max(0.0, float(SPOT_ROBUST["primary_p25_pf_floor"]) - p25_pf_sym) * float(
                    SPOT_ROBUST["primary_p25_pf_penalty_mult"]
                )
                score -= max(0.0, float(SPOT_ROBUST["primary_avg_ret_floor"]) - avg_ret_sym) * float(
                    SPOT_ROBUST["primary_avg_ret_penalty_mult"]
                )
                score -= max(0.0, float(SPOT_ROBUST["primary_p25_ret_floor"]) - p25_ret_sym) * float(
                    SPOT_ROBUST["primary_p25_ret_penalty_mult"]
                )
            score -= invalid * 12.0

            symbol_scores.append(float(score))
            symbol_results[symbol] = {
                "return": float(np.mean(fold_returns)),
                "mdd": float(np.mean(fold_mdds)),
                "pf": float(np.mean(fold_pfs)) if fold_pfs else 0.0,
                "stress_ret": float(p25_stress),
                "bh_ret": float(np.mean(fold_bh_returns)) if fold_bh_returns else 0.0,
                "excess_ret": float(np.mean(fold_excess_returns)) if fold_excess_returns else float(np.mean(fold_returns)),
                "trades_total": int(np.sum(np.asarray(fold_trade_counts, dtype=np.int64))) if fold_trade_counts else 0,
                "gross_profit": float(fold_gross_profit_sum),
                "gross_loss": float(fold_gross_loss_sum),
                "loss_trades": int(fold_loss_trades_sum),
                "win_trades": int(fold_win_trades_sum),
                "max_win_pnl": float(fold_max_win_pnl),
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
            gp = 0.0
            gl = 0.0
            win_trades = 0
            loss_trades = 0
            max_win_pnl = 0.0
            if not trades_df.empty and "pnl" in trades_df.columns:
                gp = float(trades_df[trades_df["pnl"] > 0]["pnl"].sum())
                gl = abs(float(trades_df[trades_df["pnl"] < 0]["pnl"].sum()))
                win_trades = int((trades_df["pnl"] > 0).sum())
                loss_trades = int((trades_df["pnl"] < 0).sum())
                if win_trades > 0:
                    max_win_pnl = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].max())
                pf = gp / gl if gl > 0 else (gp if gp > 0 else 0.0)
            pf = float(np.clip(pf, 0.0, float(SPOT_ROBUST["pf_clip_per_symbol"])))
            score = calculate_score(ret, mdd, trades_df, mode=mode, market_type="spot", timeframe=tf)
            if is_primary_symbol:
                score -= max(0.0, float(SPOT_ROBUST["primary_avg_pf_floor"]) - float(pf)) * float(
                    SPOT_ROBUST["primary_avg_pf_penalty_mult"]
                )
                score -= max(0.0, float(SPOT_ROBUST["primary_avg_ret_floor"]) - float(ret)) * float(
                    SPOT_ROBUST["primary_avg_ret_penalty_mult"]
                )
            symbol_scores.append(float(score if np.isfinite(score) and score > -9000 else -220.0))
            bh_ret = calculate_oos_benchmark_return_pct(data_map.get(tf), 0, len(data_map.get(tf, pd.DataFrame())) - 1)
            symbol_results[symbol] = {
                "return": ret,
                "mdd": mdd,
                "pf": pf,
                "stress_ret": ret - (float(result.get("total_trades", 0)) * SPOT_ROBUST["cost_stress_per_trade"]),
                "bh_ret": float(bh_ret),
                "excess_ret": float(ret - bh_ret),
                "trades_total": int(len(trades_df)),
                "gross_profit": float(gp),
                "gross_loss": float(gl),
                "loss_trades": int(loss_trades),
                "win_trades": int(win_trades),
                "max_win_pnl": float(max_win_pnl),
            }

        trial.set_user_attr(f"ret_{key}", float(symbol_results[symbol]["return"]))
        trial.set_user_attr(f"mdd_{key}", float(symbol_results[symbol]["mdd"]))
        trial.set_user_attr(f"pf_{key}", float(symbol_results[symbol]["pf"]))

    if not symbol_scores:
        return -10000.0
    min_success_ratio = float(np.clip(_env_float("SPOT_MIN_SYMBOL_SUCCESS_RATIO", 0.67), 0.5, 1.0))
    success_ratio = float(len(symbol_scores) - fallback_count) / float(max(1, len(symbol_scores)))
    if success_ratio < min_success_ratio:
        return -10000.0
    mdd_abs = [abs(float(v["mdd"])) for v in symbol_results.values()]
    if mdd_abs and (max(mdd_abs) > 70.0 or float(np.mean(mdd_abs)) > 55.0):
        return -10000.0

    offset = float(SPOT_ROBUST["cross_offset"])
    raw_shifted = np.array(symbol_scores, dtype=np.float64) + offset
    collapsed_symbol_count = int(np.sum(raw_shifted <= 1e-9))
    if len(symbol_scores) <= 2 and collapsed_symbol_count > 0:
        return -10000.0
    shifted = np.clip(raw_shifted, 1e-6, None)

    hm_shifted = float(len(shifted) / np.sum(1.0 / shifted))
    gm_shifted = float(np.exp(np.mean(np.log(shifted))))
    log_dispersion = float(np.std(np.log(shifted)))
    dispersion_penalty = float(
        np.exp(-float(SPOT_ROBUST["cross_log_dispersion_scale"]) * log_dispersion)
    )

    blended_shifted = (
        float(SPOT_ROBUST["cross_hm_weight"]) * hm_shifted
        + float(SPOT_ROBUST["cross_gm_weight"]) * gm_shifted
    )

    ret_values = np.array([float(v["return"]) for v in symbol_results.values()], dtype=np.float64)
    pf_values = np.array([float(v["pf"]) for v in symbol_results.values()], dtype=np.float64)
    stress_ret_values = np.array(
        [float(v.get("stress_ret", v["return"])) for v in symbol_results.values()],
        dtype=np.float64,
    )
    excess_ret_values = np.array(
        [float(v.get("excess_ret", v["return"])) for v in symbol_results.values()],
        dtype=np.float64,
    )

    ret_p25 = float(np.percentile(ret_values, 25))
    excess_p25 = float(np.percentile(excess_ret_values, 25))
    excess_avg = float(np.mean(excess_ret_values))
    pf_p25 = float(np.percentile(pf_values, 25))
    stress_ret_p25 = float(np.percentile(stress_ret_values, 25))
    score_p25 = float(np.percentile(symbol_scores, 25))
    avg_ret = float(np.mean(ret_values))
    avg_pf = float(np.mean(pf_values))
    avg_mdd_abs = float(np.mean(mdd_abs)) if mdd_abs else 0.0
    max_mdd_abs = float(np.max(mdd_abs)) if mdd_abs else 0.0
    primary_ret_values = np.array(
        [float(v["return"]) for s, v in symbol_results.items() if str(s).strip().upper() in primary_symbol_set],
        dtype=np.float64,
    )
    primary_pf_values = np.array(
        [float(v["pf"]) for s, v in symbol_results.items() if str(s).strip().upper() in primary_symbol_set],
        dtype=np.float64,
    )
    all_trade_counts = np.array([int(v.get("trades_total", 0)) for v in symbol_results.values()], dtype=np.int64)
    total_trades_all = int(np.sum(all_trade_counts))
    all_min_trades_obs = int(np.min(all_trade_counts)) if all_trade_counts.size else 0
    total_loss_trades_all = int(np.sum(np.array([int(v.get("loss_trades", 0)) for v in symbol_results.values()], dtype=np.int64)))
    total_win_trades_all = int(np.sum(np.array([int(v.get("win_trades", 0)) for v in symbol_results.values()], dtype=np.int64)))
    primary_trade_counts = np.array(
        [
            int(v.get("trades_total", 0))
            for s, v in symbol_results.items()
            if str(s).strip().upper() in primary_symbol_set
        ],
        dtype=np.int64,
    )
    primary_total_trades = int(np.sum(primary_trade_counts)) if primary_trade_counts.size else 0
    primary_min_trades_obs = int(np.min(primary_trade_counts)) if primary_trade_counts.size else 0
    primary_loss_trades = int(
        np.sum(
            np.array(
                [
                    int(v.get("loss_trades", 0))
                    for s, v in symbol_results.items()
                    if str(s).strip().upper() in primary_symbol_set
                ],
                dtype=np.int64,
            )
        )
    )
    primary_win_trades = int(
        np.sum(
            np.array(
                [
                    int(v.get("win_trades", 0))
                    for s, v in symbol_results.items()
                    if str(s).strip().upper() in primary_symbol_set
                ],
                dtype=np.int64,
            )
        )
    )
    max_single_win_pnl = float(
        np.max(np.array([float(v.get("max_win_pnl", 0.0)) for v in symbol_results.values()], dtype=np.float64))
    ) if symbol_results else 0.0
    gross_profit_all = float(np.sum(np.array([float(v.get("gross_profit", 0.0)) for v in symbol_results.values()], dtype=np.float64)))
    gross_loss_all = float(np.sum(np.array([float(v.get("gross_loss", 0.0)) for v in symbol_results.values()], dtype=np.float64)))
    pooled_pf_all = float(
        gross_profit_all / gross_loss_all
        if gross_loss_all > 0.0
        else (gross_profit_all if gross_profit_all > 0.0 else 0.0)
    )
    primary_gross_profit = float(
        np.sum(
            np.array(
                [
                    float(v.get("gross_profit", 0.0))
                    for s, v in symbol_results.items()
                    if str(s).strip().upper() in primary_symbol_set
                ],
                dtype=np.float64,
            )
        )
    )
    primary_gross_loss = float(
        np.sum(
            np.array(
                [
                    float(v.get("gross_loss", 0.0))
                    for s, v in symbol_results.items()
                    if str(s).strip().upper() in primary_symbol_set
                ],
                dtype=np.float64,
            )
        )
    )
    primary_pooled_pf = float(
        primary_gross_profit / primary_gross_loss
        if primary_gross_loss > 0.0
        else (primary_gross_profit if primary_gross_profit > 0.0 else 0.0)
    )
    if int(SPOT_ROBUST["primary_require_symbols"]) == 1 and primary_ret_values.size == 0:
        return -10000.0
    if primary_ret_values.size > 0 and primary_pf_values.size > 0:
        primary_avg_pf_hard = float(np.mean(primary_pf_values))
        primary_p25_pf_hard = float(np.percentile(primary_pf_values, 25))
        primary_avg_ret_hard = float(np.mean(primary_ret_values))
        primary_p25_ret_hard = float(np.percentile(primary_ret_values, 25))
        if primary_avg_pf_hard < float(SPOT_ROBUST["primary_hard_avg_pf_floor"]):
            return -10000.0
        if primary_p25_pf_hard < float(SPOT_ROBUST["primary_hard_p25_pf_floor"]):
            return -10000.0
        if primary_avg_ret_hard < float(SPOT_ROBUST["primary_hard_avg_ret_floor"]):
            return -10000.0
        if primary_p25_ret_hard < float(SPOT_ROBUST["primary_hard_p25_ret_floor"]):
            return -10000.0
    if primary_total_trades < int(SPOT_ROBUST["primary_total_trade_hard_floor"]):
        return -10000.0
    if primary_pooled_pf < float(SPOT_ROBUST["primary_pooled_pf_hard_floor"]):
        return -10000.0

    efficiency_boost = (
        float(SPOT_ROBUST["cross_eff_ret_weight"]) * np.clip(np.arcsinh(avg_ret / 40.0), -2.8, 2.8)
        + float(SPOT_ROBUST["cross_eff_pf_weight"]) * np.clip(np.arcsinh((avg_pf - 1.0) / 0.8), -2.4, 2.4)
    )

    soft_mdd_penalty = (
        max(0.0, avg_mdd_abs - float(SPOT_ROBUST["soft_mdd_avg_center"])) * float(SPOT_ROBUST["soft_mdd_avg_mult"])
        + max(0.0, max_mdd_abs - float(SPOT_ROBUST["soft_mdd_max_center"])) * float(SPOT_ROBUST["soft_mdd_max_mult"])
    )

    feature_scale_out = bool(params.get("ENABLE_SCALE_OUT", False))
    feature_breakeven = bool(params.get("ENABLE_BREAKEVEN", False))
    feature_pyramiding = bool(params.get("ENABLE_PYRAMIDING", False))
    feature_dynamic_risk = bool(params.get("USE_DYNAMIC_RISK", False))
    active_feature_count = int(feature_scale_out) + int(feature_breakeven) + int(feature_pyramiding) + int(feature_dynamic_risk)
    complexity_penalty = active_feature_count * float(SPOT_ROBUST["complexity_feature_penalty"])
    if feature_scale_out:
        scale_out_ratio = float(params.get("SCALE_OUT_RATIO", 0.0))
        complexity_penalty += (
            max(0.0, scale_out_ratio - float(SPOT_ROBUST["complexity_scale_out_ratio_center"]))
            * float(SPOT_ROBUST["complexity_scale_out_ratio_mult"])
        )
    if feature_pyramiding:
        pyramid_max_adds = int(max(1, params.get("PYRAMID_MAX_ADDS", 1)))
        pyramid_risk_ratio = float(params.get("PYRAMID_RISK_RATIO", 0.0))
        complexity_penalty += max(0, pyramid_max_adds - 1) * float(SPOT_ROBUST["complexity_pyramid_add_penalty"])
        complexity_penalty += (
            max(0.0, pyramid_risk_ratio - float(SPOT_ROBUST["complexity_pyramid_risk_center"]))
            * float(SPOT_ROBUST["complexity_pyramid_risk_mult"])
        )

    legacy_score = (blended_shifted * dispersion_penalty) - offset
    legacy_score += efficiency_boost
    legacy_score += (
        float(SPOT_ROBUST["cross_ret_p25_weight"])
        * np.clip(np.arcsinh(ret_p25 / 35.0), -2.4, 2.4)
    )
    legacy_score += (
        float(SPOT_ROBUST["cross_pf_p25_weight"])
        * np.clip(np.arcsinh((pf_p25 - 1.0) / 0.8), -2.0, 2.0)
    )
    legacy_score += float(SPOT_ROBUST["cross_score_p25_weight"]) * score_p25
    legacy_score += (
        float(SPOT_ROBUST["cross_stress_p25_weight"])
        * np.clip(stress_ret_p25, -80.0, 80.0)
    )
    legacy_score += (
        float(SPOT_ROBUST["cross_excess_p25_weight"])
        * np.clip(np.arcsinh(excess_p25 / 30.0), -2.4, 2.4)
    )
    legacy_score += (
        float(SPOT_ROBUST["cross_excess_avg_weight"])
        * np.clip(np.arcsinh(excess_avg / 35.0), -2.4, 2.4)
    )
    legacy_score += (
        float(SPOT_ROBUST["cross_pooled_pf_weight"])
        * np.clip(np.arcsinh((min(float(SPOT_ROBUST["pf_clip_per_symbol"]), pooled_pf_all) - 1.0) / 0.8), -2.2, 2.2)
    )
    primary_guard_penalty = 0.0
    primary_avg_ret = 0.0
    primary_avg_pf = 0.0
    primary_p25_ret = 0.0
    primary_p25_pf = 0.0
    if primary_ret_values.size > 0 and primary_pf_values.size > 0:
        primary_avg_ret = float(np.mean(primary_ret_values))
        primary_avg_pf = float(np.mean(primary_pf_values))
        primary_p25_ret = float(np.percentile(primary_ret_values, 25))
        primary_p25_pf = float(np.percentile(primary_pf_values, 25))
        primary_guard_penalty += max(
            0.0, float(SPOT_ROBUST["primary_avg_pf_floor"]) - primary_avg_pf
        ) * float(SPOT_ROBUST["primary_final_pf_penalty_mult"])
        primary_guard_penalty += max(
            0.0, float(SPOT_ROBUST["primary_p25_pf_floor"]) - primary_p25_pf
        ) * float(SPOT_ROBUST["primary_p25_pf_penalty_mult"])
        primary_guard_penalty += max(
            0.0, float(SPOT_ROBUST["primary_avg_ret_floor"]) - primary_avg_ret
        ) * float(SPOT_ROBUST["primary_final_ret_penalty_mult"])
        primary_guard_penalty += max(
            0.0, float(SPOT_ROBUST["primary_p25_ret_floor"]) - primary_p25_ret
        ) * float(SPOT_ROBUST["primary_p25_ret_penalty_mult"])
    primary_guard_penalty += max(
        0.0, float(SPOT_ROBUST["primary_pooled_pf_floor"]) - primary_pooled_pf
    ) * float(SPOT_ROBUST["primary_pooled_pf_penalty_mult"])
    primary_trade_target = max(1, int(SPOT_ROBUST["primary_total_trade_target"]))
    primary_trade_shortfall = max(0.0, (float(primary_trade_target) - float(primary_total_trades)) / float(primary_trade_target))
    primary_guard_penalty += primary_trade_shortfall * float(SPOT_ROBUST["primary_total_trade_penalty_mult"])
    all_trade_target = max(1, int(SPOT_ROBUST["all_total_trade_target"]))
    all_trade_shortfall = max(0.0, (float(all_trade_target) - float(total_trades_all)) / float(all_trade_target))
    primary_guard_penalty += all_trade_shortfall * float(SPOT_ROBUST["all_total_trade_penalty_mult"])
    legacy_score -= soft_mdd_penalty
    legacy_score -= primary_guard_penalty
    legacy_score -= complexity_penalty
    legacy_score -= collapsed_symbol_count * float(SPOT_ROBUST["collapsed_symbol_penalty"])
    core_ret_ref = float(np.mean(primary_ret_values)) if primary_ret_values.size > 0 else float(avg_ret)
    core_pf_ref = float(primary_pooled_pf) if primary_ret_values.size > 0 else float(pooled_pf_all)
    core_pf_ref = float(min(float(SPOT_ROBUST["pf_clip_per_symbol"]), core_pf_ref))
    core_pf_p25_ref = float(primary_p25_pf) if primary_pf_values.size > 0 else float(pf_p25)
    core_activity_floor = float(np.clip(float(SPOT_ROBUST["core_activity_floor"]), 0.0, 1.0))
    core_pf_ref_primary_trades = max(1.0, float(SPOT_ROBUST["core_pf_activity_ref_primary_trades"]))
    core_pf_ref_all_trades = max(1.0, float(SPOT_ROBUST["core_pf_activity_ref_all_trades"]))
    core_pf_p25_ref_min_trades = max(1.0, float(SPOT_ROBUST["core_pf_p25_activity_ref_min_trades"]))

    if primary_ret_values.size > 0:
        pooled_activity_raw = float(primary_total_trades) / core_pf_ref_primary_trades
        min_trade_activity_raw = float(primary_min_trades_obs) / core_pf_p25_ref_min_trades
    else:
        pooled_activity_raw = float(total_trades_all) / core_pf_ref_all_trades
        min_trade_activity_raw = float(all_min_trades_obs) / core_pf_p25_ref_min_trades
    pooled_activity = float(np.clip(max(core_activity_floor, pooled_activity_raw), 0.0, 1.0))
    min_trade_activity = float(np.clip(max(core_activity_floor, min_trade_activity_raw), 0.0, 1.0))

    pooled_pf_edge = float(core_pf_ref - 1.0)
    pf_p25_edge = float(core_pf_p25_ref - 1.0)
    pooled_pf_edge_eff = pooled_pf_edge * pooled_activity if pooled_pf_edge > 0.0 else pooled_pf_edge
    pf_p25_edge_eff = pf_p25_edge * min_trade_activity if pf_p25_edge > 0.0 else pf_p25_edge

    core_score = (
        (float(SPOT_ROBUST["core_weight_ret"]) * core_ret_ref)
        + (float(SPOT_ROBUST["core_weight_pooled_pf"]) * pooled_pf_edge_eff)
        + (float(SPOT_ROBUST["core_weight_pf_p25"]) * pf_p25_edge_eff)
        - (float(SPOT_ROBUST["core_mdd_penalty_mult"]) * float(avg_mdd_abs))
    )
    risk_budget_target_mdd = max(1e-6, float(SPOT_ROBUST["risk_budget_target_mdd"]))
    risk_budget_target_band = max(0.10, float(SPOT_ROBUST["risk_budget_target_band"]))
    risk_budget_low_util_floor = float(np.clip(float(SPOT_ROBUST["risk_budget_low_util_floor"]), 0.20, 1.20))
    risk_budget_mdd_util = float(avg_mdd_abs) / risk_budget_target_mdd
    risk_budget_ret_term = (
        float(SPOT_ROBUST["risk_budget_ret_weight"])
        * np.clip(np.arcsinh(core_ret_ref / 22.0), -2.8, 2.8)
    )
    risk_budget_pf_term = (
        float(SPOT_ROBUST["risk_budget_pf_weight"])
        * np.clip(np.arcsinh((core_pf_ref - 1.0) / 0.9), -2.5, 2.5)
    )
    risk_budget_center_bonus = max(
        0.0,
        1.0 - (abs(risk_budget_mdd_util - 1.0) / max(1e-6, risk_budget_target_band)),
    )
    risk_budget_under_shortfall = max(0.0, risk_budget_low_util_floor - risk_budget_mdd_util)
    risk_budget_over_excess = max(0.0, risk_budget_mdd_util - (1.0 + risk_budget_target_band))
    risk_budget_score = (
        risk_budget_ret_term
        + risk_budget_pf_term
        + (risk_budget_center_bonus * float(SPOT_ROBUST["risk_budget_center_bonus"]))
        - (risk_budget_under_shortfall * float(SPOT_ROBUST["risk_budget_under_penalty_mult"]))
        - (risk_budget_over_excess * float(SPOT_ROBUST["risk_budget_over_penalty_mult"]))
    )
    if float(avg_mdd_abs) > float(SPOT_ROBUST["risk_budget_hard_mdd_cap"]):
        return -10000.0
    fake_pf_penalty = 0.0
    loss_floor_all = max(1, int(SPOT_ROBUST["loss_trade_floor_all"]))
    loss_floor_primary = max(1, int(SPOT_ROBUST["loss_trade_floor_primary"]))
    loss_shortfall_all = max(0.0, (float(loss_floor_all) - float(total_loss_trades_all)) / float(loss_floor_all))
    loss_shortfall_primary = max(
        0.0, (float(loss_floor_primary) - float(primary_loss_trades)) / float(loss_floor_primary)
    )
    fake_pf_penalty += loss_shortfall_all * float(SPOT_ROBUST["loss_trade_shortfall_penalty_all"])
    fake_pf_penalty += loss_shortfall_primary * float(SPOT_ROBUST["loss_trade_shortfall_penalty_primary"])
    single_win_share = (max_single_win_pnl / gross_profit_all) if gross_profit_all > 0.0 else 0.0
    single_win_cap = float(SPOT_ROBUST["single_win_share_cap"])
    if single_win_share > single_win_cap:
        fake_pf_penalty += (single_win_share - single_win_cap) * float(SPOT_ROBUST["single_win_share_penalty_mult"])
    mix = float(np.clip(float(SPOT_ROBUST["core_score_mix"]), 0.0, 1.0))
    base_score = (mix * core_score) + ((1.0 - mix) * legacy_score)
    risk_budget_mix = float(np.clip(float(SPOT_ROBUST["risk_budget_mix"]), 0.0, 1.0))
    final_score = ((1.0 - risk_budget_mix) * base_score) + (risk_budget_mix * risk_budget_score) - fake_pf_penalty
    if not np.isfinite(final_score):
        return -10000.0

    trial.set_user_attr("score_avg", float(final_score))
    trial.set_user_attr("score_p25", float(score_p25))
    trial.set_user_attr("return_avg", float(avg_ret))
    trial.set_user_attr("return_p25", float(ret_p25))
    trial.set_user_attr("stress_return_p25", float(stress_ret_p25))
    trial.set_user_attr("excess_return_p25", float(excess_p25))
    trial.set_user_attr("excess_return_avg", float(excess_avg))
    trial.set_user_attr("pf_avg", float(avg_pf))
    trial.set_user_attr("pf_pooled", float(pooled_pf_all))
    trial.set_user_attr("pf_p25", float(pf_p25))
    trial.set_user_attr("mdd_avg_abs", float(avg_mdd_abs))
    trial.set_user_attr("mdd_max_abs", float(max_mdd_abs))
    trial.set_user_attr("complexity_penalty", float(complexity_penalty))
    trial.set_user_attr("primary_guard_penalty", float(primary_guard_penalty))
    trial.set_user_attr("primary_avg_ret", float(primary_avg_ret))
    trial.set_user_attr("primary_avg_pf", float(primary_avg_pf))
    trial.set_user_attr("primary_pooled_pf", float(primary_pooled_pf))
    trial.set_user_attr("primary_total_trades", int(primary_total_trades))
    trial.set_user_attr("all_total_trades", int(total_trades_all))
    trial.set_user_attr("primary_p25_ret", float(primary_p25_ret))
    trial.set_user_attr("primary_p25_pf", float(primary_p25_pf))
    trial.set_user_attr("total_loss_trades_all", int(total_loss_trades_all))
    trial.set_user_attr("total_win_trades_all", int(total_win_trades_all))
    trial.set_user_attr("primary_loss_trades", int(primary_loss_trades))
    trial.set_user_attr("primary_win_trades", int(primary_win_trades))
    trial.set_user_attr("single_win_share", float(single_win_share))
    trial.set_user_attr("fake_pf_penalty", float(fake_pf_penalty))
    trial.set_user_attr("legacy_score", float(legacy_score))
    trial.set_user_attr("core_score", float(core_score))
    trial.set_user_attr("risk_budget_score", float(risk_budget_score))
    trial.set_user_attr("risk_budget_mdd_util", float(risk_budget_mdd_util))
    trial.set_user_attr("risk_budget_target_mdd", float(risk_budget_target_mdd))
    trial.set_user_attr("risk_budget_mix", float(risk_budget_mix))
    trial.set_user_attr("core_pf_activity", float(pooled_activity))
    trial.set_user_attr("core_pf_p25_activity", float(min_trade_activity))
    trial.set_user_attr("core_primary_min_trades_obs", int(primary_min_trades_obs))
    trial.set_user_attr("active_mgmt_features", int(active_feature_count))
    trial.set_user_attr("fallback_count", int(fallback_count))
    trial.set_user_attr("symbol_success_ratio", float(success_ratio))
    return float(final_score)


def _run_default_bonus_sweep() -> int:
    sweep_script = Path(__file__).resolve().with_name("optimize_spot_bonus_sweep.py")
    cmd = [sys.executable, str(sweep_script)]
    print("[INFO] No arguments provided. Running spot bonus sweep orchestrator.")
    return int(subprocess.run(cmd).returncode)


def main(argv: Optional[List[str]] = None) -> int:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if not effective_argv and os.getenv("SPOT_SWEEP_CHILD", "0") != "1":
        return _run_default_bonus_sweep()

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--symbols", type=str, default="KRW-BTC,KRW-ETH")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--mode", type=str, default="UNIFIED", choices=["UNIFIED", "ALL"])
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated sampler seeds, e.g. 13,37,73")
    parser.add_argument(
        "--prepare-data-only",
        action="store_true",
        help="Download/refresh spot OHLCV cache only, then exit without optimization.",
    )
    args = parser.parse_args(effective_argv)

    mode = args.mode.upper()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("[ERROR] No valid symbols provided. Use --symbols with at least one market, e.g. KRW-BTC,KRW-ETH")
        return 1
    awfo_enabled = mode in AWFO_DEFAULTS["enabled_modes"]
    is_two_stage_mode = mode in {"UNIFIED", "ALL"}
    trials = args.trials if args.trials is not None else {"UNIFIED": 6000, "ALL": 6000}.get(mode, 2500)
    seed_arg = args.seeds if args.seeds is not None else os.getenv("OPTUNA_SEEDS")
    if seed_arg is None:
        seed_arg = "13,37,73" if is_two_stage_mode else "13"
    seed_list = _parse_seed_list(seed_arg) or [13]
    seed_min_trials_global = _env_int("SPOT_SEED_MIN_TRIALS", 90)
    seed_alloc = _allocate_seed_trials(trials, seed_list, min_trials_per_seed=seed_min_trials_global)
    spot_growth_coef = _env_float("SPOT_GROWTH_BONUS_COEF", 18.0)
    spot_risk_coef = _env_float("SPOT_RISK_DRAG_COEF", 10.0)
    spot_tail_coef = _env_float("SPOT_TAIL_DRAG_COEF", 10.0)
    profile_key = os.getenv("SPOT_BONUS_PROFILE", "BASE").strip() or "BASE"
    print(
        f"[INFO] mode={mode}, trials={trials}, awfo={'ON' if awfo_enabled else 'OFF'}, two_stage={'ON' if is_two_stage_mode else 'OFF'}, "
        f"seeds={seed_list}, alloc={seed_alloc}, profile={profile_key}, "
        f"bonus(g={spot_growth_coef},r={spot_risk_coef},t={spot_tail_coef})"
    )
    print(f"[INFO] optimization symbols={symbols}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    search_space = GET_SEARCH_SPACE(mode, market_type="spot")
    timeframes = search_space["TIMEFRAME"]["choices"]
    load_tfs = sorted(set(timeframes + ["1d"]))
    symbols_data = load_all_timeframes(symbols, SPOT_START_DATE, BACKTEST_END_DATE, load_tfs)
    if args.prepare_data_only:
        print(
            f"[INFO] Data preload completed for {len(symbols)} symbols x {len(load_tfs)} timeframes. "
            f"(gap_fill={'ON' if GAP_FILL_ENABLED else 'OFF'})"
        )
        return 0

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
        return 1

    study_name = f"spot_{mode.lower()}_strategy"
    storage_url = f"mysql+pymysql://{db_user}:{quote_plus(db_pass)}@{db_host}:{db_port}/{db_name}"
    
    ensure_database_exists(storage_url)

    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
    except Exception:
        pass

    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={"pool_size": max(30, args.jobs * 2), "max_overflow": 10, "pool_recycle": 3600, "pool_pre_ping": True},
    )
    try:
        backtest_loop_spot_numba(
            np.ones(10, dtype=np.float64), np.ones(10, dtype=np.float64), np.ones(10, dtype=np.float64), np.ones(10, dtype=np.float64),
            np.ones(10, dtype=np.float64), np.zeros(10, dtype=np.int64), np.zeros(10, dtype=np.int64), np.ones(10, dtype=np.float64),
            np.ones(10, dtype=np.float64), np.ones(10, dtype=np.float64), np.ones(10, dtype=np.float64), np.ones(10, dtype=np.float64),
            np.ones(10, dtype=np.float64), 10_000.0, 0.001, 0.001, 0, 0, 0.01, 1.5, 3.0, 0.99, False, 1.0, False, 3.0, 1000, 0.0,
            1.4, True, 0.6, 1.0, 4.5, 94.0, 1.3, 0.15, 0.45, 0.6, 90.0, 0.1,
            False, 1.2, 0.5, True, 0.001, False, 1.8, 1.0, 0.3, 1,
            0, False, 1_000_000.0
        )
    except Exception:
        pass

    final_best_trial: Optional[optuna.trial.FrozenTrial] = None
    final_source_name = ""
    final_robust = -float("inf")

    best_candidate_study: Optional[optuna.study.Study] = None
    best_candidate_label = ""
    best_candidate_value = -float("inf")

    try:
        if is_two_stage_mode:
            print("[2STAGE] Spot 2-Stage + Multi-Fidelity + Adaptive Refine enabled")
            cfg = SPOT_TWO_STAGE_UNIFIED_DEFAULTS.copy()
            scale = max(0.45, trials / 2800.0)
            stage1_total = int(cfg["stage1_total_trials"] * scale)
            stage1_total = max(260, min(stage1_total, max(360, int(trials * 0.52))))
            stage2_total_budget = max(180, trials - stage1_total)

            promoted_structures: List[Dict[str, object]] = []
            stage1_space_base = build_stage1_search_space(search_space)
            top_pool_limit = max(10, cfg["stage2_top_structures"] * 3)

            print(
                f"[2STAGE] Stage1 trials={stage1_total}, Stage2 budget={stage2_total_budget}, "
                f"target_structures={cfg['stage2_top_structures']}"
            )

            for fidelity in cfg["stage1_fidelity_steps"]:
                step_trials = int(stage1_total * fidelity["ratio"])
                if step_trials < 40:
                    continue

                step_symbols = max(1, min(int(fidelity["symbols"]), len(symbols)))
                step_data = subset_data_maps(
                    train_data,
                    symbols,
                    n_symbols=step_symbols,
                    data_ratio=float(fidelity["data_ratio"]),
                )
                step_merge_indices = compute_merge_indices(step_data)
                step_awfo = build_awfo_plan(
                    step_data,
                    timeframes,
                    folds=int(fidelity["folds"]),
                    min_trades=int(fidelity["min_trades"]),
                )
                step_awfo["cache"] = build_awfo_runtime_cache(step_data, timeframes, step_awfo)

                step_space = restrict_stage1_space_by_candidates(stage1_space_base, promoted_structures)
                step_study_name = f"{study_name}__s1_{fidelity['name']}_{int(time.time())}"

                print(
                    f"[STAGE1] {fidelity['name'].upper()} | trials={step_trials}, symbols={step_symbols}, "
                    f"data_ratio={fidelity['data_ratio']}, folds={fidelity['folds']}, seeds={seed_list}"
                )
                seed_runs = run_seeded_studies(
                    study_name_prefix=step_study_name,
                    storage_url=storage_url,
                    storage=storage,
                    n_trials=step_trials,
                    n_jobs=args.jobs,
                    startup_ratio=float(fidelity["startup_ratio"]),
                    objective_fn=lambda t, _data=step_data, _space=step_space, _merge=step_merge_indices, _awfo=step_awfo: objective(
                        t,
                        _data,
                        _space,
                        mode,
                        _merge,
                        _awfo,
                    ),
                    seeds=seed_list,
                    min_trials_per_seed=_env_int("SPOT_STAGE1_SEED_MIN_TRIALS", seed_min_trials_global),
                )

                completed: List[optuna.trial.FrozenTrial] = []
                for seed, seed_trials, seed_study, step_startup in seed_runs:
                    seed_best = float(seed_study.best_value) if len(seed_study.trials) > 0 else -float("inf")
                    seed_robust = _robust_value_from_study(seed_study)
                    print(
                        f"   [INFO] Stage1-{fidelity['name']} seed={seed} | trials={seed_trials} | startup={step_startup} "
                        f"| best={seed_best:.2f} | robust={seed_robust:.2f}"
                    )
                    if np.isfinite(seed_robust) and seed_robust > best_candidate_value:
                        best_candidate_value = float(seed_robust)
                        best_candidate_study = seed_study
                        best_candidate_label = f"{step_study_name}:seed{seed}"
                    completed.extend(_study_complete_trials(seed_study))

                if not completed:
                    continue
                completed.sort(key=lambda tr: float(tr.value), reverse=True)
                top_k = max(cfg["stage2_top_structures"], int(len(completed) * cfg["promotion_ratio"]))
                top_k = min(top_k, top_pool_limit, len(completed))

                promoted: List[Dict[str, object]] = []
                seen = set()
                for tr in completed[:top_k]:
                    sig = extract_structure_signature(tr.params)
                    sig_key = tuple((k, sig.get(k)) for k in STRUCTURE_PARAM_KEYS)
                    if sig_key in seen or not sig:
                        continue
                    seen.add(sig_key)
                    promoted.append(sig)
                promoted_structures = promoted
                print(f"   [PROMOTE] structures={len(promoted_structures)} (top={top_k})")

            if not promoted_structures:
                fallback_sig = _default_structure_signature(stage1_space_base)
                if fallback_sig:
                    promoted_structures = [fallback_sig]
                else:
                    promoted_structures = [{}]

            promoted_structures = promoted_structures[: cfg["stage2_top_structures"]]
            min_stage2_trials = int(max(60, cfg["stage2_min_trials_per_structure"]))
            max_structures_by_budget = max(1, int(stage2_total_budget // max(1, min_stage2_trials)))
            effective_structure_count = max(
                1,
                min(len(promoted_structures), int(cfg["stage2_top_structures"]), max_structures_by_budget),
            )
            promoted_structures = promoted_structures[:effective_structure_count]
            per_structure_trials = max(
                min_stage2_trials,
                int(stage2_total_budget / max(1, len(promoted_structures))),
            )
            stage2_awfo = build_awfo_plan(
                train_data,
                timeframes,
                folds=cfg["stage2_folds"],
                min_trades=cfg["stage2_min_trades"],
            )
            stage2_awfo["cache"] = build_awfo_runtime_cache(train_data, timeframes, stage2_awfo)

            best_stage2_study: Optional[optuna.study.Study] = None
            best_stage2_label = ""
            best_stage2_value = -float("inf")

            for i, struct_sig in enumerate(promoted_structures, start=1):
                stage2_space = freeze_structure_in_space(search_space, struct_sig)
                step_study_name = f"{study_name}__s2_{i}_{int(time.time())}"
                print(
                    f"[STAGE2] {i}/{len(promoted_structures)} | total_trials={per_structure_trials} | "
                    f"structure={struct_sig}"
                )

                pass1_trials = max(35, int(per_structure_trials * cfg["stage2_refine_ratio"]))
                pass2_trials = max(0, per_structure_trials - pass1_trials)

                stage2_completed: List[optuna.trial.FrozenTrial] = []
                struct_best_study: Optional[optuna.study.Study] = None
                struct_best_robust = -float("inf")
                struct_best_label = ""

                p1_runs = run_seeded_studies(
                    study_name_prefix=f"{step_study_name}_p1",
                    storage_url=storage_url,
                    storage=storage,
                    n_trials=pass1_trials,
                    n_jobs=args.jobs,
                    startup_ratio=float(cfg["stage2_startup_ratio"]),
                    objective_fn=lambda t, _space=stage2_space: objective(
                        t,
                        train_data,
                        _space,
                        mode,
                        merge_indices,
                        stage2_awfo,
                    ),
                    seeds=seed_list,
                    min_trials_per_seed=_env_int("SPOT_STAGE2_SEED_MIN_TRIALS", 30),
                )
                for seed, seed_trials, seed_study, seed_startup in p1_runs:
                    seed_best = float(seed_study.best_value) if len(seed_study.trials) > 0 else -float("inf")
                    seed_robust = _robust_value_from_study(seed_study)
                    print(
                        f"   [INFO] Stage2-{i} pass1 seed={seed} | trials={seed_trials} | startup={seed_startup} "
                        f"| best={seed_best:.2f} | robust={seed_robust:.2f}"
                    )
                    stage2_completed.extend(_study_complete_trials(seed_study))
                    if np.isfinite(seed_robust) and seed_robust > struct_best_robust:
                        struct_best_robust = float(seed_robust)
                        struct_best_study = seed_study
                        struct_best_label = f"{step_study_name}_p1:seed{seed}"

                if stage2_completed and pass2_trials >= 25:
                    refined_space = build_adaptive_numeric_space(
                        stage2_space,
                        stage2_completed,
                        top_quantile=cfg["stage2_refine_top_quantile"],
                        min_width_ratio=cfg["stage2_refine_min_width_ratio"],
                        min_samples=cfg["stage2_refine_min_samples"],
                        min_step_span=cfg["stage2_refine_step_span"],
                    )
                    p2_runs = run_seeded_studies(
                        study_name_prefix=f"{step_study_name}_p2",
                        storage_url=storage_url,
                        storage=storage,
                        n_trials=pass2_trials,
                        n_jobs=args.jobs,
                        startup_ratio=float(cfg["stage2_startup_ratio"]),
                        objective_fn=lambda t, _space=refined_space: objective(
                            t,
                            train_data,
                            _space,
                            mode,
                            merge_indices,
                            stage2_awfo,
                        ),
                        seeds=seed_list,
                        min_trials_per_seed=_env_int("SPOT_STAGE2_SEED_MIN_TRIALS", 30),
                    )
                    for seed, seed_trials, seed_study, seed_startup in p2_runs:
                        seed_best = float(seed_study.best_value) if len(seed_study.trials) > 0 else -float("inf")
                        seed_robust = _robust_value_from_study(seed_study)
                        print(
                            f"   [INFO] Stage2-{i} pass2 seed={seed} | trials={seed_trials} | startup={seed_startup} "
                            f"| best={seed_best:.2f} | robust={seed_robust:.2f}"
                        )
                        stage2_completed.extend(_study_complete_trials(seed_study))
                        if np.isfinite(seed_robust) and seed_robust > struct_best_robust:
                            struct_best_robust = float(seed_robust)
                            struct_best_study = seed_study
                            struct_best_label = f"{step_study_name}_p2:seed{seed}"

                if not stage2_completed or struct_best_study is None:
                    continue

                structure_robust = _robust_value_from_trials(stage2_completed)
                print(f"   [INFO] Stage2-{i} summary | robust={structure_robust:.2f} | complete_trials={len(stage2_completed)}")
                if np.isfinite(structure_robust) and structure_robust > best_candidate_value:
                    best_candidate_value = float(structure_robust)
                    best_candidate_study = struct_best_study
                    best_candidate_label = f"{step_study_name}:robust"

                if structure_robust > best_stage2_value:
                    best_stage2_value = float(structure_robust)
                    best_stage2_study = struct_best_study
                    best_stage2_label = struct_best_label

            if best_stage2_study is None:
                raise RuntimeError("2-stage optimization failed to produce complete stage2 studies.")

            completed_final = _study_complete_trials(best_stage2_study)
            if not completed_final:
                raise RuntimeError("2-stage winner has no completed trials.")
            final_best_trial = _select_representative_trial(completed_final)
            final_source_name = best_stage2_label or "stage2_winner"
            final_robust = float(best_stage2_value)

        else:
            startup_ratio = 0.22 if awfo_enabled else 0.20
            seed_runs = run_seeded_studies(
                study_name_prefix=f"{study_name}__single",
                storage_url=storage_url,
                storage=storage,
                n_trials=trials,
                n_jobs=args.jobs,
                startup_ratio=startup_ratio,
                objective_fn=lambda t: objective(
                    t,
                    train_data,
                    search_space,
                    mode,
                    merge_indices,
                    awfo_plan,
                ),
                seeds=seed_list,
                min_trials_per_seed=_env_int("SPOT_SINGLE_SEED_MIN_TRIALS", seed_min_trials_global),
            )
            single_best_study: Optional[optuna.study.Study] = None
            single_best_seed: Optional[int] = None
            single_best_robust = -float("inf")
            for seed, seed_trials, seed_study, seed_startup in seed_runs:
                seed_best = float(seed_study.best_value) if len(seed_study.trials) > 0 else -float("inf")
                seed_robust = _robust_value_from_study(seed_study)
                print(
                    f"[INFO] single seed={seed} | trials={seed_trials} | startup={seed_startup} "
                    f"| best={seed_best:.2f} | robust={seed_robust:.2f}"
                )
                if np.isfinite(seed_robust) and seed_robust > single_best_robust:
                    single_best_robust = float(seed_robust)
                    single_best_study = seed_study
                    single_best_seed = seed
            if single_best_study is None:
                raise RuntimeError("single-stage optimization failed to produce any completed seed study.")

            completed_final = _study_complete_trials(single_best_study)
            if not completed_final:
                raise RuntimeError("single-stage winner has no completed trials.")
            final_best_trial = _select_representative_trial(completed_final)
            final_source_name = f"single_seed_{single_best_seed}"
            final_robust = float(single_best_robust)

    except Exception as e:
        print(f"[ERROR] optimization failed: {e}")
        if best_candidate_study is not None:
            completed_fallback = _study_complete_trials(best_candidate_study)
            if completed_fallback:
                final_best_trial = _select_representative_trial(completed_fallback)
                final_source_name = f"fallback:{best_candidate_label}"
                final_robust = _robust_value_from_trials(completed_fallback)
                print(f"[FALLBACK] publishing best intermediate result from {best_candidate_label}")
        if final_best_trial is None:
            return 1

    if final_best_trial is None:
        print("[ERROR] No completed trial to publish.")
        return 1

    final_best_params = apply_spot_tf_param_limits(dict(final_best_trial.params))

    holdout_gate_enabled = os.getenv("SPOT_ENABLE_HOLDOUT_GATE", "0") == "1"
    holdout_gate_enforced = os.getenv("SPOT_ENFORCE_HOLDOUT_GATE", "0") == "1"
    holdout_summary: Dict[str, float] = {}
    holdout_reasons: List[str] = []
    holdout_passed = True
    holdout_gate_state = "PASS"
    holdout_gate_display_reasons: List[str] = []
    if holdout_gate_enabled:
        try:
            holdout_passed, holdout_summary, holdout_reasons = run_holdout_gate(
                best_params=final_best_params,
                symbols_data=symbols_data,
                mode=mode,
                cutoff_ts=cutoff_ts,
            )
            min_trades_gate = int(
                holdout_summary.get(
                    "dynamic_min_trades_gate",
                    HOLDOUT_SANITY_GATES["core_min_trades"],
                )
            )
            holdout_gate_state, holdout_gate_display_reasons = classify_holdout_outcome(
                holdout_summary,
                bool(holdout_passed),
                holdout_reasons,
                int(min_trades_gate),
            )
            gate_repr = (
                "PASS"
                if holdout_gate_state == "PASS"
                else f"{holdout_gate_state}({','.join(holdout_gate_display_reasons)})"
            )
            print(
                "[HOLDOUT] "
                f"window={TRAIN_CUTOFF_DATE}~{BACKTEST_END_DATE} | "
                f"ret={holdout_summary.get('core_avg_ret', 0.0):.2f}% | "
                f"mdd_abs={holdout_summary.get('core_avg_mdd_abs', 0.0):.2f}% | "
                f"min_trades={int(holdout_summary.get('core_min_trades', 0))} | "
                f"symbols={int(holdout_summary.get('symbols_evaluated', 0))}/{int(holdout_summary.get('symbols_total', 0))} | "
                f"gate={gate_repr}"
            )
        except Exception as e:
            holdout_passed = False
            holdout_reasons = [f"holdout_eval_error:{e}"]
            holdout_gate_state = "FAIL"
            holdout_gate_display_reasons = list(holdout_reasons)
            print(f"[WARN] holdout gate evaluation failed (non-blocking): {e}")
        if not holdout_passed and holdout_gate_enforced:
            print("[WARN] Holdout gate failed and enforcement is enabled. Publishing is blocked.")
            return 1
    else:
        print("[INFO] Holdout gate skipped in optimize stage (verify stage should enforce holdout).")

    optimizer_version = "SPOT_AWFO_2STAGE_MF_ADAPT_V1" if is_two_stage_mode else "SPOT_AWFO_MULTI_SEED_V2"
    _publish_best_trial_alias(
        storage_url=storage_url,
        alias_study_name=study_name,
        source_study_name=final_source_name,
        source_trial=final_best_trial,
        metadata={
            "optimizer_version": optimizer_version,
            "seed_source": final_source_name,
            "seed_robust_value": float(final_robust),
            "seed_list": ",".join(str(s) for s in seed_list),
            "awfo_enabled": bool(awfo_enabled),
            "two_stage_enabled": bool(is_two_stage_mode),
            "bonus_profile": profile_key,
            "spot_growth_bonus_coef": float(spot_growth_coef),
            "spot_risk_drag_coef": float(spot_risk_coef),
            "spot_tail_drag_coef": float(spot_tail_coef),
            "db_name": str(db_name),
            "holdout_gate_enabled": bool(holdout_gate_enabled),
            "holdout_gate_enforced": bool(holdout_gate_enforced),
            "holdout_gate_passed": bool(holdout_passed),
            "holdout_gate_state": str(holdout_gate_state),
            "holdout_gate_reasons": ",".join(holdout_reasons),
            "holdout_gate_display_reasons": ",".join(holdout_gate_display_reasons),
            "holdout_core_avg_ret": float(holdout_summary.get("core_avg_ret", 0.0)),
            "holdout_core_avg_mdd_abs": float(holdout_summary.get("core_avg_mdd_abs", 0.0)),
            "holdout_core_avg_pf": float(holdout_summary.get("core_avg_pf", 0.0)),
            "holdout_core_min_trades": int(holdout_summary.get("core_min_trades", 0)),
            "holdout_dynamic_min_trades_gate": int(holdout_summary.get("dynamic_min_trades_gate", 0)),
            "holdout_symbols_evaluated": int(holdout_summary.get("symbols_evaluated", 0)),
            "holdout_symbols_total": int(holdout_summary.get("symbols_total", 0)),
            "holdout_min_symbol_coverage": float(HOLDOUT_MIN_SYMBOL_COVERAGE),
        },
    )

    print(f"[DONE] best_score={float(final_best_trial.value):.2f}")
    print(f"[DONE] best_params={final_best_params}")
    final_attrs = dict(getattr(final_best_trial, "user_attrs", {}) or {})
    total_trades_all = int(final_attrs.get("all_total_trades", 0) or 0)
    total_win_trades_all = int(final_attrs.get("total_win_trades_all", 0) or 0)
    win_rate_all = (float(total_win_trades_all) / float(total_trades_all) * 100.0) if total_trades_all > 0 else 0.0
    snapshot = {
        "profile": str(profile_key),
        "mode": str(mode),
        "score": float(final_best_trial.value),
        "ret": float(final_attrs.get("return_avg", 0.0)),
        "mdd": float(final_attrs.get("mdd_avg_abs", 0.0)),
        "pf": float(final_attrs.get("pf_avg", 0.0)),
        "trades": int(total_trades_all),
        "win_rate": float(win_rate_all),
        "study": str(study_name),
        "source": str(final_source_name),
    }
    print(f"[SNAPSHOT] {json.dumps(snapshot, ensure_ascii=True, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
