import argparse
import gc
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
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
GAP_FILL_MAX_RANGES = int(os.getenv("SPOT_GAP_FILL_MAX_RANGES", "8"))

DAILY_BUFFER_DAYS = 200
WARMUP_BUFFER_BARS = {"5m": 500, "15m": 420, "30m": 350, "1h": 300, "2h": 250, "4h": 200, "1d": 150, "1w": 80}
AWFO_DEFAULTS = {
    "enabled_modes": {"UNIFIED", "ALL"},
    "folds": 3,
    "min_trades_per_fold": 35,
    "min_test_bars": {"5m": 1600, "15m": 1200, "30m": 900, "1h": 600, "2h": 420, "4h": 240, "1d": 120, "1w": 60},
    "embargo_bars": {"5m": 40, "15m": 32, "30m": 24, "1h": 24, "2h": 16, "4h": 12, "1d": 5, "1w": 2},
}

STRUCTURE_PARAM_KEYS = [
    "ENTRY_TYPE",
    "TREND_FILTER_TYPE",
    "STRENGTH_FILTER_TYPE",
    "EXIT_TYPE",
    "STOP_LOSS_TYPE",
    "USE_TAKE_PROFIT",
    "USE_VOLUME_FILTER",
    "TIMEFRAME",
    "USE_DYNAMIC_RISK",
]

# Spot (long-only, no leverage) tuned defaults.
SPOT_TWO_STAGE_UNIFIED_DEFAULTS = {
    "stage1_total_trials": 1400,
    "stage1_fidelity_steps": [
        {"name": "low", "ratio": 0.50, "symbols": 1, "data_ratio": 0.45, "folds": 2, "min_trades": 24, "startup_ratio": 0.34},
        {"name": "mid", "ratio": 0.32, "symbols": 2, "data_ratio": 0.72, "folds": 3, "min_trades": 30, "startup_ratio": 0.27},
        {"name": "high", "ratio": 0.18, "symbols": 3, "data_ratio": 1.00, "folds": 3, "min_trades": 35, "startup_ratio": 0.22},
    ],
    "promotion_ratio": 0.35,
    "stage2_top_structures": 7,
    "stage2_trials_per_structure": 170,
    "stage2_folds": 3,
    "stage2_min_trades": 30,
    "stage2_startup_ratio": 0.18,
    "stage2_refine_ratio": 0.45,
    "stage2_refine_top_quantile": 0.24,
    "stage2_refine_min_width_ratio": 0.25,
    "stage2_refine_min_samples": 28,
    "stage2_refine_step_span": 4,
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
    return float(snapped)


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
            if new_high <= new_low:
                continue
            span_steps = int(np.floor((new_high - new_low) / max(step, 1e-12)))
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
    for key in STRUCTURE_PARAM_KEYS:
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
) -> List[Tuple[int, int, optuna.study.Study, int]]:
    allocations = _allocate_seed_trials(
        total_trials=int(max(1, n_trials)),
        seeds=seeds,
        min_trials_per_seed=_env_int("SPOT_SEED_MIN_TRIALS", 70),
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
        study.optimize(
            objective_fn,
            n_trials=seed_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
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
    frozen = optuna.trial.create_trial(
        params=source_trial.params,
        distributions=source_trial.distributions,
        value=float(source_trial.value),
        user_attrs=user_attrs,
    )
    alias.add_trial(frozen)
    print(f"[INFO] Published alias study '{alias_study_name}' from source '{source_study_name}'")


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
    # Recent-regime alignment: emphasize latest folds to reduce holdout collapse.
    "recent_score_w": _env_float("SPOT_RECENT_FOLD_SCORE_WEIGHT", 0.16),
    "recent_ret_w": _env_float("SPOT_RECENT_FOLD_RET_WEIGHT", 0.06),
    # Trade-density control: penalize low-activity candidates even if return is high.
    "trade_target_min": _env_int("SPOT_FOLD_TRADE_TARGET_MIN", 22),
    "trade_shortfall_penalty": _env_float("SPOT_FOLD_TRADE_SHORTFALL_PENALTY", 85.0),
    "trade_min_shortfall_penalty": _env_float("SPOT_FOLD_TRADE_MIN_SHORTFALL_PENALTY", 110.0),
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
            if df is not None and not df.empty:
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
            fold_trade_counts: List[int] = []
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
                        fold_trade_counts.append(int(len(oos_trades)))
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
            # Trade-density penalty to avoid "few-trade overfit winners" that fail holdout gate.
            if fold_trade_counts:
                target = max(1.0, float(max(int(awfo_min_trades or 0), int(SPOT_ROBUST["trade_target_min"]))))
                avg_trades = float(np.mean(fold_trade_counts))
                min_trades = float(np.min(fold_trade_counts))
                avg_shortfall = max(0.0, (target - avg_trades) / target)
                min_shortfall = max(0.0, (target - min_trades) / target)
                score -= avg_shortfall * SPOT_ROBUST["trade_shortfall_penalty"]
                score -= min_shortfall * SPOT_ROBUST["trade_min_shortfall_penalty"]
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
    parser.add_argument("--mode", type=str, default="UNIFIED", choices=["SCALP", "DAY", "SWING", "UNIFIED", "ALL"])
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated sampler seeds, e.g. 13,37,73")
    args = parser.parse_args(effective_argv)

    mode = args.mode.upper()
    symbols = [s.strip() for s in args.symbols.split(",")]
    awfo_enabled = mode in AWFO_DEFAULTS["enabled_modes"]
    is_two_stage_mode = mode in {"UNIFIED", "ALL"}
    trials = args.trials if args.trials is not None else {"SCALP": 3600, "DAY": 4200, "SWING": 5000, "UNIFIED": 5600, "ALL": 5600}.get(mode, 2500)
    seed_arg = args.seeds if args.seeds is not None else os.getenv("OPTUNA_SEEDS")
    if seed_arg is None:
        seed_arg = "13,37,73" if is_two_stage_mode else "13"
    seed_list = _parse_seed_list(seed_arg) or [13]
    seed_alloc = _allocate_seed_trials(trials, seed_list, min_trials_per_seed=_env_int("SPOT_SEED_MIN_TRIALS", 70))
    spot_growth_coef = _env_float("SPOT_GROWTH_BONUS_COEF", 18.0)
    spot_risk_coef = _env_float("SPOT_RISK_DRAG_COEF", 10.0)
    spot_tail_coef = _env_float("SPOT_TAIL_DRAG_COEF", 10.0)
    profile_key = os.getenv("SPOT_BONUS_PROFILE", "BASE").strip() or "BASE"
    print(
        f"[INFO] mode={mode}, trials={trials}, awfo={'ON' if awfo_enabled else 'OFF'}, two_stage={'ON' if is_two_stage_mode else 'OFF'}, "
        f"seeds={seed_list}, alloc={seed_alloc}, profile={profile_key}, "
        f"bonus(g={spot_growth_coef},r={spot_risk_coef},t={spot_tail_coef})"
    )

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
            1.4, True, 0.6, 1.0, 4.5, 94.0, 1.3, 0.15, 0.45, 0.6, 90.0, 0.1, 0, False, 1_000_000.0
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
            stage1_total = max(260, min(stage1_total, max(360, int(trials * 0.58))))
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
            per_structure_trials = max(
                60,
                min(
                    cfg["stage2_trials_per_structure"],
                    int(stage2_total_budget / max(1, len(promoted_structures))),
                ),
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
            final_best_trial = completed_final[0]
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
            final_best_trial = completed_final[0]
            final_source_name = f"single_seed_{single_best_seed}"
            final_robust = float(single_best_robust)

    except Exception as e:
        print(f"[ERROR] optimization failed: {e}")
        if best_candidate_study is not None:
            completed_fallback = _study_complete_trials(best_candidate_study)
            if completed_fallback:
                final_best_trial = completed_fallback[0]
                final_source_name = f"fallback:{best_candidate_label}"
                final_robust = _robust_value_from_trials(completed_fallback)
                print(f"[FALLBACK] publishing best intermediate result from {best_candidate_label}")
        if final_best_trial is None:
            return 1

    if final_best_trial is None:
        print("[ERROR] No completed trial to publish.")
        return 1

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
        },
    )

    print(f"[DONE] best_score={float(final_best_trial.value):.2f}")
    print(f"[DONE] best_params={final_best_trial.params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
