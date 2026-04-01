from __future__ import annotations

import joblib
import hashlib
import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import optuna
import numpy as np
import pandas as pd
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from filelock import FileLock
except ImportError:
    FileLock = None  # type: ignore[misc, assignment]

from config.opt_config import OPT_SPOT_CONFIG, get_spot_effective_independent_trials
from config.settings import SPOT_INITIAL_BALANCE
from src.spot_strategy.strategies_spot import UltimateSpotStrategy
from src.spot_strategy.engine_spot import BacktestEngineFastSpot
from src.spot_strategy.portfolio_shared_cash import run_shared_cash_multi_symbol
from src.spot_strategy.opt_spot_utils.metrics import (
    calc_profit_factor_from_pnl,
    calc_mdd_from_equity,
    portfolio_cagr_pct_from_equity,
    calc_sortino_from_equity,
    calc_tail_ratio_from_equity,
    cvar_loss_pct_from_simple_returns,
    compute_dsr_from_path_values,
    max_underwater_bars_from_equity,
    mean_of_worst_quartile,
    probabilistic_sharpe_ratio,
)
from src.spot_strategy.opt_spot_utils.cv_utils import (
    CPCVPath,
    build_cpcv_test_paths_with_fallback,
)
from src.spot_strategy.opt_spot_utils.opt_params import suggest_params_spot

_logger: logging.Logger = logging.getLogger("opt_spot")

SymbolFoldResult = Tuple[str, float, float, float, float, float, float, float, np.ndarray, float]

_MAX_SYMBOL_WORKERS: int = max(1, int(os.getenv("OPT_SPOT_SYMBOL_WORKERS", "1")))

def compute_embargo_bars(tf: str, longest_indicator_period: int = 150) -> int:
    fixed_min: Dict[str, int] = {"4h": 24}
    ratio_map: Dict[str, float] = {"4h": 0.05}
    ratio: float = ratio_map.get(tf, 0.03)
    return max(fixed_min.get(tf, 2), int(longest_indicator_period * ratio))

EMBARGO_BARS: Dict[str, int] = {
    "4h": compute_embargo_bars("4h"),
}

SIGNAL_CACHE_PARAM_KEYS: frozenset[str] = frozenset([
    "SIGNAL_TYPE",
    "REGIME_TYPE",
    "SIZING_METHOD",
    "ATR_PERIOD",
    "EMA_FAST_PERIOD",
    "EMA_SLOW_PERIOD",
    "RSI_PERIOD",
    "RSI_LOW_THRESH",
    "KC_PERIOD",
    "KC_MULT",
    "MOMENTUM_PERIOD",
    "TP_MEAN_PERIOD",
    "KELLY_WINDOW",
    "W_SIGNAL",
    "K_ACCEL",
    "MB_FLOOR",
    "C_HYST",
    "EPSILON_MIN",
    "K_COOL_DOWN",
    "EMA_ATR_REGIME_SLOW",
    "ATR_REGIME_PERIOD",
    "VOL_PCT_WINDOW",
    "VOL_QUANTILE",
    "ST_ATR_PERIOD",
    "ST_MULT",
    "TQ_EMA_FAST",
    "TQ_EMA_SLOW",
    "TQ_ADX_PERIOD",
    "TQ_ADX_THRESHOLD",
    "PFK_WINDOW",
    "PFK_MIN_F",
])

_SIGNAL_CACHE_SCHEMA_VERSION: int = 11

_SPOT_OBJECTIVE_CAGR_WEIGHT: float = 0.0  # Abandoning additive CAGR
_SPOT_OBJECTIVE_MIN_TRADES_HARD: float = 10.0 # Min trades per path to even consider
_SPOT_OBJECTIVE_MIN_TRADES_SOFT: float = 25.0 # Target trades for statistical robustness
_SPOT_OBJECTIVE_TAIL_RATIO_WEIGHT: float = 0.15 # Increased importance of asymmetry
_SPOT_OBJECTIVE_LOG_TWR_WEIGHT: float = 1.0
_SPOT_OBJECTIVE_PATH_CV_PENALTY: float = 0.75  # Kelly drag 균형점: G∞=μ-σ²/2 → σ직접 패널티화의 합리적 상한 λ=0.75 (λ=1.0은 고수익 고변동 경로 과억압)

_SignalCacheKey = Tuple[Tuple[Tuple[str, Any], ...], str, str, int, int, int]
_SIGNAL_CACHE_MAXSIZE: int = 256
_ARRAYS_CACHE_MAXSIZE: int = 256
_cache_lock: threading.Lock = threading.Lock()
_signal_cache: OrderedDict[_SignalCacheKey, pd.DataFrame] = OrderedDict()
_arrays_cache: OrderedDict[_SignalCacheKey, Dict[str, np.ndarray]] = OrderedDict()
_numpy_cache_warning_emitted: bool = False

_DISK_CACHE_READ_EXCEPTIONS: tuple[type[Exception], ...] = (
    OSError,
    EOFError,
    ValueError,
    AttributeError,
    ImportError,
    ModuleNotFoundError,
)


def _segment_with_context(
    full_signal_df: pd.DataFrame,
    exec_start_idx: int,
    exec_end_idx: int,
) -> Tuple[pd.DataFrame, int]:
    slice_start = max(0, int(exec_start_idx) - 1)
    slice_end = max(slice_start, int(exec_end_idx))
    segment = full_signal_df.iloc[slice_start:slice_end].copy()
    execution_start_idx = int(exec_start_idx) - slice_start
    if execution_start_idx == 0 and len(segment) > 1:
        execution_start_idx = 1
    return segment, execution_start_idx


def _evaluate_symbol_for_fold_parallel(
    sym: str,
    *,
    params: Dict[str, Any],
    strategy: UltimateSpotStrategy,
    tf: str,
    data_maps: Dict[str, Dict[str, Any]],
    full_signal_dfs: Dict[str, pd.DataFrame],
    test_start: int,
    test_end: int,
) -> Optional[SymbolFoldResult]:
    target_df: Optional[pd.DataFrame] = data_maps.get(sym, {}).get(tf)
    daily_df: Optional[pd.DataFrame] = data_maps.get(sym, {}).get("1d")
    full_merge_idx: Optional[np.ndarray] = data_maps.get(sym, {}).get(f"merge_idx_{tf}")
    if target_df is None or daily_df is None or full_merge_idx is None:
        return None

    is_start_idx: int = int(data_maps[sym].get(f"is_start_idx_{tf}", 0))
    adj_test_start: int = test_start + is_start_idx
    adj_test_end: int = test_end + is_start_idx

    if sym not in full_signal_dfs:
        return None

    segment, execution_start_idx = _segment_with_context(
        full_signal_dfs[sym], adj_test_start, adj_test_end
    )

    try:
        (
            score,
            ret_pct,
            mdd_pct,
            trades_count,
            win_rate,
            pf,
            long_count,
            equity_curve,
            tail_ratio,
        ) = evaluate_symbol_fold(
            strategy,
            params,
            sym,
            tf,
            target_df,
            daily_df,
            full_merge_idx,
            None,
            adj_test_start,
            adj_test_end,
            precomputed_signal_df=segment,
            execution_start_idx=execution_start_idx,
        )
    except Exception as exc:
        _logger.warning("Symbol-level evaluation error for %s: %s", sym, exc, exc_info=True)
        return None

    if equity_curve.size == 0:
        span_days_sym: float = 0.0
    elif "datetime" in segment.columns and len(segment) > execution_start_idx:
        span_seconds: float = float(
            (
                segment["datetime"].iloc[-1]
                - segment["datetime"].iloc[execution_start_idx]
            ).total_seconds()
        )
        span_days_sym = max(span_seconds / 86400.0, 1.0)
    else:
        span_days_sym = 0.0

    return (
        sym,
        score,
        ret_pct,
        mdd_pct,
        float(trades_count),
        win_rate,
        pf,
        tail_ratio,
        equity_curve,
        span_days_sym,
    )


def _dataset_fingerprint_from_df(df: pd.DataFrame) -> int:
    """Lightweight cache invalidation when OHLCV rows change but length stays equal."""
    if "close" not in df.columns or df.empty:
        return 0
    c = df["close"].to_numpy(dtype=np.float64)
    n = len(c)
    head = c[: min(5, n)]
    tail = c[max(0, n - 5) :]
    h = hash((tuple(head.tolist()), tuple(tail.tolist()), n))
    return int(h & ((1 << 63) - 1))


def _signal_disk_cache_path(cache_key: _SignalCacheKey, root: Path) -> Path:
    # cache_key = (params_tuple, sym, tf, data_len, fingerprint, version)
    params_tuple, sym, tf, data_len, fingerprint, version = cache_key
    
    # SIGNAL_TYPE 추출 (폴더 구조용)
    sig_type = "default"
    for k, v in params_tuple:
        if k == "SIGNAL_TYPE":
            sig_type = str(v).lower()
            break
            
    # 파라미터 제원 + 데이터 길이/버전/지문까지 포함하여 해시 충돌 방지
    hash_payload = (params_tuple, data_len, fingerprint, version)
    digest = hashlib.sha256(repr(hash_payload).encode("utf-8")).hexdigest()
    
    # 구조: root / symbol / tf / sig_type / hash.joblib
    folder = root / sym / tf / sig_type
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{digest}.joblib"


def _build_signal_cache_key(
    params: Dict[str, Any],
    sym: str,
    tf: str,
    data_len: int,
    fingerprint: int,
) -> _SignalCacheKey:
    signal_items: List[Tuple[str, Any]] = sorted(
        (k, params[k]) for k in SIGNAL_CACHE_PARAM_KEYS if k in params
    )
    return (
        tuple(signal_items),
        sym,
        tf,
        data_len,
        int(fingerprint),
        _SIGNAL_CACHE_SCHEMA_VERSION,
    )


def get_or_compute_signals(
    cache_key: _SignalCacheKey,
    target_df: pd.DataFrame,
    strategy: UltimateSpotStrategy,
    *,
    disk_cache_root: Optional[Path] = None,
) -> pd.DataFrame:
    with _cache_lock:
        if cache_key in _signal_cache:
            _signal_cache.move_to_end(cache_key)
            return _signal_cache[cache_key]

    # 1. 디스크 캐시 확인
    if disk_cache_root is not None:
        cache_path = _signal_disk_cache_path(cache_key, disk_cache_root)
        if cache_path.exists():
            try:
                # joblib.load와 mmap_mode='r'을 사용하여 RAM 사용량 최소화 (DDR5/NVMe 최적)
                full_df = joblib.load(cache_path, mmap_mode='r')
                
                # 메모리 캐시에도 저장 (LRU)
                with _cache_lock:
                    while len(_signal_cache) >= _SIGNAL_CACHE_MAXSIZE:
                        _signal_cache.popitem(last=False)
                    _signal_cache[cache_key] = full_df
                return full_df
            except _DISK_CACHE_READ_EXCEPTIONS:
                _logger.warning("Failed to read signal cache with joblib at %s, recomputing...", cache_path)

    # 2. 신규 계산
    full_df: pd.DataFrame = strategy.generate_signals(target_df.copy(deep=True))
    
    # 3. 디스크 캐시 저장
    if disk_cache_root is not None:
        cache_path = _signal_disk_cache_path(cache_key, disk_cache_root)
        try:
            # 병렬 워커 간 충돌 방지를 위한 임시 파일 저장 후 교체 방식
            tmp_path = cache_path.with_suffix(f".tmp.{os.getpid()}")
            # mmap 효율을 위해 압축 없이 저장 (DDR5/NVMe 환경 최적)
            joblib.dump(full_df, tmp_path)
            tmp_path.replace(cache_path)
        except Exception as e:
            _logger.warning("Failed to write signal cache with joblib at %s: %s", cache_path, e)

    # 4. 메모리 캐시 저장
    with _cache_lock:
        if cache_key in _signal_cache:
            _signal_cache.move_to_end(cache_key)
            return _signal_cache[cache_key]
        while len(_signal_cache) >= _SIGNAL_CACHE_MAXSIZE:
            _signal_cache.popitem(last=False)
        _signal_cache[cache_key] = full_df
        return full_df

def evaluate_symbol_fold(
    strategy: UltimateSpotStrategy,
    params: Dict[str, Any],
    symbol: str,
    tf: str,
    target_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    full_merge_idx: np.ndarray,
    precomputed_daily_df: Optional[pd.DataFrame],
    test_start: int,
    test_end: int,
    precomputed_signal_df: Optional[pd.DataFrame] = None,
    execution_start_idx: int = 0,
) -> Tuple[float, float, float, int, float, float, int, np.ndarray, float]:
    if precomputed_signal_df is not None:
        sig_oos: pd.DataFrame = precomputed_signal_df
    else:
        full_signal: pd.DataFrame = strategy.generate_signals(target_df.copy(deep=True))
        sig_oos, execution_start_idx = _segment_with_context(full_signal, test_start, test_end)

    warmup_bars: int = 0
    sig_oos.attrs = {"warmup_bars": warmup_bars}
    
    engine: BacktestEngineFastSpot = BacktestEngineFastSpot(
        hourly_df=sig_oos,
        daily_df=daily_df,
        strategy=strategy,
        initial_balance=SPOT_INITIAL_BALANCE,
        merge_index_map=None,
        precomputed_daily_df=None,
        warmup_bars=warmup_bars,
        execution_start_idx=execution_start_idx,
    )

    params_fixed = params.copy()
    params_fixed["USE_COMPOUNDING"] = True
    engine.strategy.params = params_fixed

    try:
        result: Dict[str, Any] = engine.run()
        trades_df: pd.DataFrame = result.get("trades_df", pd.DataFrame())
    except Exception as e:
        _logger.warning("Backtest engine error: %s", e, exc_info=True)
        return -10.0, 0.0, 0.0, 0, 0.0, 0.0, 0, np.array([]), 0.0

    if trades_df is None or trades_df.empty:
        return -10.0, 0.0, 0.0, 0, 0.0, 0.0, 0, np.array([]), 0.0
        
    long_count: int = len(trades_df[trades_df["side"] == "LONG"])

    equity_curve = result.get("equity_curve", np.array([]))
    mdd_pct: float = abs(float(result.get("mdd_pct", 0.0)))
    ret_pct: float = float(result.get("total_return_pct", 0.0))

    span_days: float = (
        float(
            (
                sig_oos["datetime"].iloc[-1]
                - sig_oos["datetime"].iloc[min(execution_start_idx, len(sig_oos) - 1)]
            ).total_seconds() / 86400.0
        )
        if "datetime" in sig_oos.columns and not sig_oos.empty
        else 1.0
    )
    span_days = max(span_days, 1.0)
    
    true_pnl = trades_df["pnl"]
    win_rate: float = float((len(trades_df[true_pnl > 0]) / len(trades_df)) * 100) if len(trades_df) > 0 else 0.0
    pf = calc_profit_factor_from_pnl(true_pnl)
    
    total_ret_ratio = 1.0 + (ret_pct / 100.0)
    cagr = ((total_ret_ratio ** (365.0 / span_days)) - 1.0) * 100.0 if total_ret_ratio > 0 else -100.0

    tail_ratio = calc_tail_ratio_from_equity(equity_curve) if equity_curve.size > 1 else 0.0

    return cagr, ret_pct, mdd_pct, len(trades_df), win_rate, pf, long_count, equity_curve, tail_ratio

def _merge_spot_fixed_signal_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Student-t HMM + FRAMA/EvR are fixed in UltimateSpotStrategy; no Optuna toggles."""
    return dict(params)


def _dataframe_to_symbol_arrays(sig_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    # volume: required for shared-cash Numba ADV anchor (concurrency slippage scaling).
    required = ("open", "high", "low", "close", "volume", "atr", "long_entry_signal", "entry_upper")
    for c in required:
        if c not in sig_df.columns:
            raise ValueError(f"Missing column {c} for shared-cash segment.")
    out: Dict[str, np.ndarray] = {}
    for c in required:
        out[c] = sig_df[c].to_numpy(dtype=np.float64)
    if "regime_risk_mult" in sig_df.columns:
        out["regime_risk_mult"] = sig_df["regime_risk_mult"].to_numpy(dtype=np.float64)
    if "garch_kelly_f" in sig_df.columns:
        out["garch_kelly_f"] = sig_df["garch_kelly_f"].to_numpy(dtype=np.float64)
    if "kill_signal" in sig_df.columns:
        out["kill_signal"] = sig_df["kill_signal"].to_numpy(dtype=np.float64)
    if "bb_upper" in sig_df.columns:
        out["bb_upper"] = sig_df["bb_upper"].to_numpy(dtype=np.float64)
    if "trail_tighten_flag" in sig_df.columns:
        out["trail_tighten_flag"] = sig_df["trail_tighten_flag"].to_numpy(dtype=np.float64)
    return out


def _dataframe_to_symbol_arrays_extended(sig_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Full IS arrays for shared-cash + optional slot_rank_score (views used in CPCV slices)."""
    base = _dataframe_to_symbol_arrays(sig_df)
    if "slot_rank_score" in sig_df.columns:
        base["slot_rank_score"] = sig_df["slot_rank_score"].to_numpy(dtype=np.float64)
    return base


def _slice_symbol_arrays_view(
    full: Dict[str, np.ndarray],
    slice_start: int,
    slice_end: int,
) -> Dict[str, np.ndarray]:
    return {k: v[slice_start:slice_end] for k, v in full.items()}


def _segment_span_days(sig_df: pd.DataFrame, execution_start_idx: int) -> float:
    if "datetime" not in sig_df.columns or sig_df.empty:
        return 1.0
    i0 = min(max(0, int(execution_start_idx)), len(sig_df) - 1)
    span_seconds = float(
        (sig_df["datetime"].iloc[-1] - sig_df["datetime"].iloc[i0]).total_seconds()
    )
    return max(span_seconds / 86400.0, 1.0)


def _span_days_ref_slice(ref_df: pd.DataFrame, abs_start: int, abs_end: int) -> float:
    """Calendar span (days) for CPCV segment [abs_start, abs_end) on aligned reference OHLCV."""
    if ref_df.empty:
        return 1.0
    if "datetime" not in ref_df.columns:
        return max(float(abs_end - abs_start), 1.0) * (4.0 / 24.0)
    i0 = int(np.clip(abs_start, 0, len(ref_df) - 1))
    i1 = int(np.clip(abs_end - 1, 0, len(ref_df) - 1))
    if i1 < i0:
        return 1.0 / 24.0
    dt0 = ref_df["datetime"].iloc[i0]
    dt1 = ref_df["datetime"].iloc[i1]
    sec = float((dt1 - dt0).total_seconds())
    return max(sec / 86400.0, 1.0 / 24.0)


def _compute_path_gmgr_high_moments(equity: np.ndarray) -> float:
    """
    Geometric mean growth proxy with high-moment correction on log returns:
    r_mean - σ²/2 + S·σ³/6 - K·σ⁴/24. Returns winsorized at [p1, p99].
    """
    eq = np.asarray(equity, dtype=np.float64).ravel()
    if eq.size < 3:
        return -1.0
    eq = np.maximum(eq, 1e-12)
    r = np.diff(np.log(eq))
    r = r[np.isfinite(r)]
    if r.size < 5:
        return float(np.mean(r)) if r.size > 0 else -1.0
    lo, hi = np.percentile(r, [1.0, 99.0])
    rw = np.clip(r, lo, hi)
    mu = float(np.mean(rw))
    sd = float(np.std(rw, ddof=1))
    if sd < 1e-12:
        return mu
    m3 = float(np.mean((rw - mu) ** 3))
    m4 = float(np.mean((rw - mu) ** 4))
    skew = m3 / (sd**3)
    ex_kurt = m4 / (sd**4) - 3.0
    return float(mu - (sd**2) / 2.0 + skew * (sd**3) / 6.0 - ex_kurt * (sd**4) / 24.0)


def _compute_ulcer_index(equity: np.ndarray) -> float:
    """Ulcer Index: sqrt(mean(D_i^2)), D_i = (peak_i - eq_i) / peak_i * 100."""
    eq = np.asarray(equity, dtype=np.float64).ravel()
    if eq.size < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    safe = np.where(peak > 1e-12, peak, 1.0)
    drawdown_pct = (peak - eq) / safe * 100.0
    return float(np.sqrt(np.mean(drawdown_pct * drawdown_pct)))


def objective_spot(
    trial: optuna.Trial,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf_target: str,
    *,
    space: Dict[str, Dict[str, Any]],
    mode: str = "single",
    project_root: Optional[str] = None,
    prebuilt_cpcv_bundle: Optional[Tuple[List[CPCVPath], int, int]] = None,
    signal_disk_cache_root: Optional[Path] = None,
) -> float:
    params: Dict[str, Any] = suggest_params_spot(trial, space, tf_target)
    tf: str = tf_target
    strategy: UltimateSpotStrategy = UltimateSpotStrategy(name="OptSpot", params=params)

    ref_sym = symbols[0]
    is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
    ref_df = data_maps[ref_sym][tf]
    if ref_df is None or ref_df.empty:
        raise optuna.TrialPruned()
    ref_len = len(ref_df) - is_off
    if ref_len < 200:
        raise optuna.TrialPruned()

    embargo = int(EMBARGO_BARS.get(tf, 0))
    if prebuilt_cpcv_bundle is not None:
        cpcv_paths, nb_cpcv, k_cpcv = prebuilt_cpcv_bundle
    else:
        cpcv_paths, nb_cpcv, k_cpcv = build_cpcv_test_paths_with_fallback(ref_len, embargo=embargo)
    if not cpcv_paths:
        raise optuna.TrialPruned()
    n_independent_paths = max(2, nb_cpcv // k_cpcv)

    cache_root: Optional[Path] = signal_disk_cache_root
    if cache_root is None and project_root is not None:
        cache_root = Path(project_root) / ".spot_signal_cache"

    strategy._portfolio_eval_ctx = {"data_maps": data_maps, "symbols": list(symbols), "tf": tf}
    try:
        full_signal_dfs: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            target_df_full: Optional[pd.DataFrame] = data_maps.get(sym, {}).get(tf)
            if target_df_full is None or target_df_full.empty:
                continue
            fp = _dataset_fingerprint_from_df(target_df_full)
            cache_key: _SignalCacheKey = _build_signal_cache_key(
                params, sym, tf, len(target_df_full), fp
            )
            full_signal_dfs[sym] = get_or_compute_signals(
                cache_key, target_df_full, strategy, disk_cache_root=cache_root
            )

        if len(full_signal_dfs) != len(symbols):
            raise optuna.TrialPruned()

        prebuilt_full_arrays: Dict[str, Dict[str, np.ndarray]] = {}
        for sym in symbols:
            target_df_full = data_maps.get(sym, {}).get(tf)
            if target_df_full is None or target_df_full.empty:
                raise optuna.TrialPruned()
            fp = _dataset_fingerprint_from_df(target_df_full)
            sig_key = _build_signal_cache_key(params, sym, tf, len(target_df_full), fp)
            with _cache_lock:
                if sig_key in _arrays_cache:
                    _arrays_cache.move_to_end(sig_key)
                    prebuilt_full_arrays[sym] = _arrays_cache[sig_key]
                    continue
            arrs = _dataframe_to_symbol_arrays_extended(full_signal_dfs[sym])
            with _cache_lock:
                while len(_arrays_cache) >= _ARRAYS_CACHE_MAXSIZE:
                    _arrays_cache.popitem(last=False)
                _arrays_cache[sig_key] = arrs
            prebuilt_full_arrays[sym] = arrs

        max_slots = int(OPT_SPOT_CONFIG.get("SPOT_MAX_CONCURRENT_POSITIONS", 3))
        min_seg_trades = int(OPT_SPOT_CONFIG.get("SPOT_MIN_TRADES_PER_CPCV_SEGMENT", 4))
        # Signals are pre-computed on full IS; per-segment re-warmup would skip most of each test block.
        warmup_bars = 0

        cfg = OPT_SPOT_CONFIG
        seg_fail_pen = float(cfg.get("SPOT_SEGMENT_TRADE_FAIL_PENALTY", 2.0))
        sortino_eps = float(cfg.get("SPOT_OBJECTIVE_SORTINO_EPS", 1e-6))
        sortino_ratio_cap = float(cfg.get("SPOT_OBJECTIVE_SORTINO_RATIO_CAP", 1.0e6))
        path_sortino_clip = float(cfg.get("SPOT_OBJECTIVE_PATH_SORTINO_CLIP", 500.0))
        mini_window_pct = float(cfg.get("SPOT_OBJECTIVE_MINI_WINDOW_PCT", 0.33))
        lam_ui = float(cfg.get("SPOT_OBJECTIVE_LAMBDA_UI", 0.05))
        w_trade_pen = float(cfg.get("SPOT_OBJECTIVE_W_TRADE", 0.005))
        w_sqn_pen = float(cfg.get("SPOT_OBJECTIVE_W_SQN", 0.02))

        path_compound_log_tw: List[float] = []
        path_compound_raw_log_tw: List[float] = []
        path_compound_tw_ratio: List[float] = []
        path_sortino_vals: List[float] = []
        path_worst_mdd: List[float] = []
        path_max_cvar: List[float] = []
        path_trades: List[int] = []
        path_tail_ratios: List[float] = []
        path_pfs: List[float] = []
        path_gmgr: List[float] = []
        path_ui: List[float] = []
        path_calmars: List[float] = []
        path_cagrs: List[float] = []

        total_sym_trades = np.zeros(len(symbols), dtype=np.int32)
        total_sym_pnl = np.zeros(len(symbols), dtype=np.float64)

        for path_idx, path in enumerate(cpcv_paths):
            seg_log_tw: List[float] = []
            seg_raw_log_tw: List[float] = []
            seg_tw_ratio: List[float] = []
            seg_mdds: List[float] = []
            seg_cvars: List[float] = []
            seg_tails: List[float] = []
            seg_pfs: List[float] = []
            path_total_trades = 0
            running_balance = float(SPOT_INITIAL_BALANCE)
            path_eq_chunks: List[np.ndarray] = []
            span_path_days = 0.0
            for test_start, test_end in path:
                abs_start = is_off + int(test_start)
                abs_end = is_off + int(test_end)
                slice_start = max(0, abs_start - 1)
                slice_end = min(len(ref_df), abs_end)
                if slice_end - slice_start < 5:
                    continue
                span_path_days += _span_days_ref_slice(ref_df, abs_start, abs_end)
                symbol_arrays: Dict[str, Dict[str, np.ndarray]] = {}
                rank_scores: Dict[str, np.ndarray] = {}
                for sym in symbols:
                    symbol_arrays[sym] = _slice_symbol_arrays_view(
                        prebuilt_full_arrays[sym], slice_start, slice_end
                    )
                    rs = symbol_arrays[sym].get("slot_rank_score")
                    if rs is not None:
                        rank_scores[sym] = rs

                execution_start_idx = max(1, abs_start - slice_start)
                segment_initial = max(running_balance, 1e-9)
                try:
                    result = run_shared_cash_multi_symbol(
                        symbol_arrays,
                        symbols,
                        params,
                        initial_balance=segment_initial,
                        max_concurrent_positions=max_slots,
                        rank_scores=rank_scores if rank_scores else None,
                        warmup_bars=warmup_bars,
                        execution_start_idx=execution_start_idx,
                        allow_python_fallback=False,
                        concurrency_penalty_scale=0.5,
                    )
                except Exception as exc:
                    _logger.warning(
                        "Shared-cash Numba path failed (allow_python_fallback=False): %s",
                        exc,
                        exc_info=True,
                    )
                    raise
                eq = result.equity_curve
                path_total_trades += int(result.total_trades)
                if hasattr(result, "per_symbol_trades"):
                    total_sym_trades += result.per_symbol_trades
                    total_sym_pnl += result.per_symbol_pnl
                if eq.size == 0:
                    twr = 1.0
                else:
                    twr = max(float(result.final_balance / segment_initial), 1e-9)
                raw_log_tw = float(np.log(twr))
                log_tw = raw_log_tw
                if eq.size > 0:
                    eq0 = float(eq[0]) if float(eq[0]) > 1e-12 else segment_initial
                    scale = float(segment_initial) / eq0
                    scaled_eq = eq.astype(np.float64, copy=False) * scale
                    if not path_eq_chunks:
                        path_eq_chunks.append(scaled_eq)
                    else:
                        path_eq_chunks.append(scaled_eq[1:])
                if eq.size > 1:
                    mdd_seg = float(calc_mdd_from_equity(eq))
                    seg_mdds.append(mdd_seg)
                    seg_cvars.append(float(cvar_loss_pct_from_simple_returns(eq)))
                    seg_tails.append(float(calc_tail_ratio_from_equity(eq)))
                else:
                    seg_mdds.append(0.0)
                    seg_cvars.append(0.0)
                    seg_tails.append(1.0)
                    
                pnl = result.pnl_array
                pos_pnl = float(np.sum(pnl[pnl > 0.0]))
                neg_pnl = float(np.abs(np.sum(pnl[pnl < 0.0])))
                # Asymptotic Cap for zero-loss singularity to avoid penalizing perfect segments
                if neg_pnl > 1e-12:
                    seg_pf = pos_pnl / neg_pnl
                elif pos_pnl > 1e-12:
                    seg_pf = 3.0  # Theoretical cap for stable alpha
                else:
                    seg_pf = 1.0  # Neutral for zero trades
                seg_pfs.append(seg_pf)

                if int(result.total_trades) < min_seg_trades:
                    log_tw -= seg_fail_pen

                seg_raw_log_tw.append(raw_log_tw)
                seg_log_tw.append(log_tw)
                seg_tw_ratio.append(twr)
                running_balance = max(float(result.final_balance), 1e-9)

            if not seg_log_tw:
                raise optuna.TrialPruned()
            path_compound_log_tw.append(float(np.sum(seg_log_tw)))
            path_compound_raw_log_tw.append(float(np.sum(seg_raw_log_tw)))
            path_compound_tw_ratio.append(float(np.prod(seg_tw_ratio)) if seg_tw_ratio else 1.0)
            path_worst_mdd.append(float(np.max(seg_mdds)) if seg_mdds else 0.0)
            
            # HARD HURDLE: If any path exceeds MDD limit, prune immediately.
            path_mdd_limit = float(OPT_SPOT_CONFIG.get("SPOT_HOLDOUT_MDD_LIMIT_PCT", 20.0))
            if path_worst_mdd[-1] > path_mdd_limit:
                 raise optuna.TrialPruned()
            
            # HARD HURDLE: Min trades per path for statistical relevance.
            if path_total_trades < _SPOT_OBJECTIVE_MIN_TRADES_HARD:
                 raise optuna.TrialPruned()

            path_trades.append(path_total_trades)
            path_tail_ratios.append(float(np.mean(seg_tails)) if seg_tails else 1.0)
            path_pfs.append(float(np.mean(seg_pfs)) if seg_pfs else 1.0)

            path_eq = np.concatenate(path_eq_chunks) if path_eq_chunks else np.array([], dtype=np.float64)
            span_for_sortino = max(span_path_days, 1.0)
            raw_ps = float(calc_sortino_from_equity(path_eq, span_for_sortino)) if path_eq.size >= 2 else 0.0
            if not np.isfinite(raw_ps):
                raw_ps = 0.0
            path_sortino = float(np.clip(raw_ps, -path_sortino_clip, path_sortino_clip))
            path_sortino_vals.append(path_sortino)

            if path_eq.size >= 2:
                path_gmgr.append(_compute_path_gmgr_high_moments(path_eq))
                path_ui.append(_compute_ulcer_index(path_eq))
                span_c = max(span_path_days, 1.0)
                pc_cagr = float(portfolio_cagr_pct_from_equity(path_eq, span_c))
                pc_mdd = float(calc_mdd_from_equity(path_eq))
                path_calmars.append(pc_cagr / max(pc_mdd, 0.01))
                path_cagrs.append(pc_cagr)
            else:
                path_gmgr.append(-1.0)
                path_ui.append(0.0)
                path_calmars.append(0.0)
                path_cagrs.append(0.0)

            g_arr = np.asarray(path_gmgr, dtype=np.float64)
            u_arr = np.asarray(path_ui, dtype=np.float64)
            run_max_ui = float(np.max(u_arr)) if u_arr.size else 0.0
            path_arr_partial = np.asarray(path_compound_raw_log_tw, dtype=np.float64)
            n_p = int(path_arr_partial.size)
            if n_p >= 2:
                mu_p = float(np.mean(path_arr_partial))
                sd_p = float(np.std(path_arr_partial, ddof=1))
                sqn_p = math.sqrt(float(n_p)) * mu_p / sd_p if sd_p > 1e-12 else 0.0
            else:
                sqn_p = 0.0
            n_tm = float(np.mean(path_trades)) if path_trades else 0.0
            p_pf = float(np.mean(path_pfs)) if path_pfs else 1.0
            
            # Penalize: trades < target, SQN < target, Turnover > 100, PF < 1.4
            soft_p = (
                max(0.0, _SPOT_OBJECTIVE_MIN_TRADES_SOFT - n_tm) * w_trade_pen
                + max(0.0, 1.6 - sqn_p) * w_sqn_pen
                + max(0.0, n_tm - 100.0) * 0.005
                + max(0.0, 1.4 - p_pf) * 0.1
            )
            run_geo = float(np.mean(path_compound_raw_log_tw)) if path_compound_raw_log_tw else 0.0
            run_calmar = float(np.mean(path_calmars)) if path_calmars else 0.0
            run_cagr = float(np.mean(path_cagrs)) if path_cagrs else 0.0
            run_tail = float(np.mean(path_tail_ratios)) if path_tail_ratios else 0.0
            run_tail_bonus = math.log(max(run_tail, 0.1)) * _SPOT_OBJECTIVE_TAIL_RATIO_WEIGHT
            run_cagr_floor_pen = max(0.0, 30.0 - run_cagr) * 0.003
            mu_p = float(np.mean(path_arr_partial)) if path_arr_partial.size > 0 else 0.0
            sd_p = float(np.std(path_arr_partial)) if path_arr_partial.size > 1 else 0.0
            
            # Simple intermediate objective for pruning: Mean Log-TWR - Variance Penalty
            interm = mu_p - _SPOT_OBJECTIVE_PATH_CV_PENALTY * sd_p + run_tail_bonus - lam_ui * run_max_ui - (soft_p * 2.0)
            trial.report(interm, step=path_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_log_tw = float(np.mean(path_compound_raw_log_tw))
        mean_penalized_log_tw = float(np.mean(path_compound_log_tw))
        cvar25_log = float(mean_of_worst_quartile(path_compound_raw_log_tw))

        worst_path_tw = (
            float(np.percentile(np.asarray(path_compound_tw_ratio, dtype=np.float64), 10.0))
            if path_compound_tw_ratio
            else 1.0
        )
        mean_path_sortino = float(np.mean(path_sortino_vals)) if path_sortino_vals else 0.0
        std_path_sortino = (
            float(np.std(path_sortino_vals, ddof=1)) if len(path_sortino_vals) > 1 else 0.0
        )
        sortino_ratio = mean_path_sortino / (std_path_sortino + sortino_eps)
        sortino_ratio = float(np.clip(sortino_ratio, -sortino_ratio_cap, sortino_ratio_cap))

        gmgr_arr = np.asarray(path_gmgr, dtype=np.float64)
        ui_arr = np.asarray(path_ui, dtype=np.float64)
        p10_gmgr = float(np.percentile(gmgr_arr, 10.0)) if gmgr_arr.size else -1.0
        max_ui = float(np.max(ui_arr)) if ui_arr.size else 0.0
        n_trades_mean = float(np.mean(path_trades)) if path_trades else 0.0

        path_arr = np.asarray(path_compound_raw_log_tw, dtype=np.float64)
        n_paths_ct = int(path_arr.size)
        if n_paths_ct >= 2:
            mu_paths = float(np.mean(path_arr))
            sd_paths = float(np.std(path_arr, ddof=1))
            gate1_sqn = math.sqrt(float(n_paths_ct)) * mu_paths / sd_paths if sd_paths > 1e-12 else 0.0
            cv_paths = sd_paths / (abs(mu_paths) + 1e-6)
        else:
            gate1_sqn = 0.0
            cv_paths = 10.0

        # DSR (Deflated Sharpe Ratio) for path log-TWR stability
        n_done = trial.number + 1
        n_startup = int(OPT_SPOT_CONFIG.get("tpe_n_startup_trials", 100))
        n_eff = get_spot_effective_independent_trials(n_done, n_startup)
        path_vals = [float(x) for x in path_compound_raw_log_tw]
        dsr_val = float(compute_dsr_from_path_values(path_vals, n_eff))

        if n_paths_ct >= 4 and dsr_val < -1.0:
            raise optuna.TrialPruned()

        concentration_pen = max(0.0, cv_paths - 2.0) * 0.02
        # Soft Penalties: SQN < 1.6, low mean CPCV trades (CLT), path CV concentration
        # NEW MULTIPLICATIVE/ROBUST OBJECTIVE: Log-TWR Mean - Path Variance Penalty
        mu_paths = float(np.mean(path_arr)) if path_arr.size > 0 else -10.0
        sd_paths = float(np.std(path_arr)) if path_arr.size > 1 else 10.0
        
        # Path Consistency Coefficient (CV)
        path_cv = sd_paths / (abs(mu_paths) + 1e-6)
        
        # Base Wealth Maximization (Geometric Mean in log space)
        raw_geometric_mean = mu_paths
        
        # Soft Cap for Extreme IS Returns (Restored for Power Law extraction)
        # Sweet Spot: 100% CAGR ≈ 0.693 log return. Encourages aggressive compounding.
        cap_threshold = 0.693
        if raw_geometric_mean > cap_threshold:
            geometric_mean_log = cap_threshold + (raw_geometric_mean - cap_threshold) * 0.3
        else:
            geometric_mean_log = raw_geometric_mean
        
        # Penalize inconsistency between paths
        consistency_penalty = _SPOT_OBJECTIVE_PATH_CV_PENALTY * sd_paths
        
        # Tail Ratio and Trade count bonuses/penalties refined
        mean_tail_ratio = float(np.mean(path_tail_ratios)) if path_tail_ratios else 0.0
        tail_bonus = math.log(max(mean_tail_ratio, 0.1)) * _SPOT_OBJECTIVE_TAIL_RATIO_WEIGHT
        
        mean_path_pf = float(np.mean(path_pfs)) if path_pfs else 1.0
        
        # Breadth Penalty (Generalization without forced equality)
        # Ensures underlying alpha works on at least 3~4 assets, allowing Power Law scaling.
        positive_coins = np.sum(total_sym_pnl > 0.0)
        breadth_penalty = max(0.0, 4.0 - float(positive_coins)) * 0.15
        
        soft_penalty = (
            max(0.0, _SPOT_OBJECTIVE_MIN_TRADES_SOFT - n_trades_mean) * (w_trade_pen * 10.0)
            + max(0.0, 2.0 - gate1_sqn) * (w_sqn_pen * 2.0)
            + max(0.0, 1.35 - mean_path_pf) * 0.1     # Recalibrated: 0.1 PF drop ≈ 1% log-drag
            + breadth_penalty
        )

        w_calmar = float(cfg.get("SPOT_OBJECTIVE_W_CALMAR", 0.08))
        worst25_calmar = float(np.percentile(path_calmars, 25)) if path_calmars else 0.0
        calmar_term = math.log(max(worst25_calmar + 1.0, 0.01)) * w_calmar

        objective_final = float(
            geometric_mean_log
            - consistency_penalty
            + tail_bonus
            + calmar_term
            - lam_ui * max_ui
            - soft_penalty
        )

        # PSR: P(SR > 0) on CPCV path log-TWR values
        if n_paths_ct >= 2 and sd_paths > 1e-12:
            sr_est = mu_paths / sd_paths
            psr_val = probabilistic_sharpe_ratio(sr_est, n_obs=n_paths_ct)
        else:
            psr_val = 0.0

        worst_seg_mdd = float(np.max(path_worst_mdd)) if path_worst_mdd else 0.0
        mean_path_return_pct = (math.exp(mean_log_tw) - 1.0) * 100.0
        mean_path_calmar = float(np.mean(path_calmars)) if path_calmars else 0.0

        trial.set_user_attr("growth_score", float(objective_final))
        trial.set_user_attr("p10_gmgr", float(p10_gmgr))
        trial.set_user_attr("objective_geometric_mean_log", float(geometric_mean_log))
        trial.set_user_attr("objective_mean_path_calmar", float(mean_path_calmar))
        trial.set_user_attr("max_ulcer_index", float(max_ui))
        trial.set_user_attr("objective_soft_penalty", float(soft_penalty))
        trial.set_user_attr("dsr_paths", dsr_val)
        trial.set_user_attr("objective_final", objective_final)
        trial.set_user_attr("gate1_sqn", float(gate1_sqn))
        trial.set_user_attr("psr_paths", float(psr_val))
        trial.set_user_attr("gate1_path_sortino", float(mean_path_sortino))
        trial.set_user_attr("cpcv_path_tail_ratio", float(mean_tail_ratio))
        trial.set_user_attr("cpcv_mean_path_return_pct", float(mean_path_return_pct))
        trial.set_user_attr("cpcv_worst_segment_mdd_pct", float(worst_seg_mdd))

        return float(objective_final)
    finally:
        strategy._portfolio_eval_ctx = None


def _compute_signal_stats(sig_df: pd.DataFrame) -> Dict[str, float]:
    """OOS holdout: quantify signal vs regime gating on the execution window."""
    n = len(sig_df)
    if n == 0:
        return {}
    if "long_entry_signal" not in sig_df.columns:
        return {}
    les = sig_df["long_entry_signal"]
    signal_rate = float((les > 0).sum() / n)
    if "regime_risk_mult" in sig_df.columns:
        rrm = sig_df["regime_risk_mult"]
        regime_rate = float((rrm > 0.0).sum() / n)
        joint_rate = float(((les > 0) & (rrm > 0.0)).sum() / n)
    else:
        regime_rate = float("nan")
        joint_rate = float("nan")
    return {
        "signal_fire_rate": signal_rate,
        "regime_on_rate": regime_rate,
        "joint_entry_eligible_rate": joint_rate,
    }


def run_holdout_shared_cash_portfolio(
    params: Dict[str, Any],
    symbols: List[str],
    tf: str,
    oos_data_maps: Dict[str, Dict[str, Any]],
    *,
    signal_disk_cache_root: Optional[Path] = None,
    return_signal_dfs: bool = False,
    concurrency_penalty_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    OOS holdout: single shared-cash run from oos_start_idx to end for all symbols.
    """
    p = dict(params)
    strategy: UltimateSpotStrategy = UltimateSpotStrategy(name="HoldoutSpot", params=p)
    strategy._portfolio_eval_ctx = {"data_maps": oos_data_maps, "symbols": list(symbols), "tf": tf}
    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    try:
        for sym in symbols:
            df_full: Optional[pd.DataFrame] = oos_data_maps.get(sym, {}).get(tf)
            if df_full is None or df_full.empty:
                continue
            fp = _dataset_fingerprint_from_df(df_full)
            cache_key: _SignalCacheKey = _build_signal_cache_key(p, sym, tf, len(df_full), fp)
            full_signal_dfs[sym] = get_or_compute_signals(
                cache_key,
                df_full,
                strategy,
                disk_cache_root=signal_disk_cache_root,
            )
    finally:
        strategy._portfolio_eval_ctx = None
    if len(full_signal_dfs) != len(symbols):
        failed: Dict[str, Any] = {
            "portfolio_cagr_pct": -100.0,
            "mdd_pct": 100.0,
            "cvar_pct": 100.0,
            "tail_ratio": 0.0,
            "long_trades": 0.0,
            "min_path_tw": 0.0,
            "dd_bars": 0.0,
            "final_balance": 0.0,
            "moic": 0.0,
            "equity_curve": np.array([]),
            "oos_signal_stats_by_symbol": {},
        }
        if return_signal_dfs:
            failed["full_signal_dfs"] = {}
        return failed

    ref_sym = symbols[0]
    oos_start = int(oos_data_maps[ref_sym].get(f"oos_start_idx_{tf}", 0))
    ref_df = full_signal_dfs[ref_sym]
    slice_start = max(0, oos_start - 1)
    slice_end = len(ref_df)
    _logger.info(
        "Holdout OOS debug: oos_start=%d, slice_start=%d, slice_end=%d, "
        "exec_start=%d, seg_len=%d, n_symbols=%d",
        oos_start, slice_start, slice_end, max(1, oos_start - slice_start),
        slice_end - slice_start, len(symbols)
    )
    if slice_end - slice_start < 5:
        _logger.warning("Holdout OOS segment too short (len < 5). Returning FAIL.")
        failed: Dict[str, Any] = {
            "portfolio_cagr_pct": -100.0,
            "mdd_pct": 100.0,
            "cvar_pct": 100.0,
            "tail_ratio": 0.0,
            "long_trades": 0.0,
            "min_path_tw": 0.0,
            "dd_bars": 0.0,
            "final_balance": 0.0,
            "moic": 0.0,
            "equity_curve": np.array([]),
            "oos_signal_stats_by_symbol": {},
        }
        if return_signal_dfs:
            failed["full_signal_dfs"] = full_signal_dfs
        return failed

    symbol_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    rank_scores: Dict[str, np.ndarray] = {}
    for sym in symbols:
        seg = full_signal_dfs[sym].iloc[slice_start:slice_end]
        symbol_arrays[sym] = _dataframe_to_symbol_arrays(seg)
        if "slot_rank_score" in seg.columns:
            rank_scores[sym] = seg["slot_rank_score"].to_numpy(dtype=np.float64)

    execution_start_idx = max(1, oos_start - slice_start)
    holdout_warmup_bars = 0
    initial_balance = float(SPOT_INITIAL_BALANCE)
    max_slots = int(OPT_SPOT_CONFIG.get("SPOT_MAX_CONCURRENT_POSITIONS", 3))
    res = run_shared_cash_multi_symbol(
        symbol_arrays,
        symbols,
        p,
        initial_balance=initial_balance,
        max_concurrent_positions=max_slots,
        rank_scores=rank_scores if rank_scores else None,
        warmup_bars=holdout_warmup_bars,
        execution_start_idx=execution_start_idx,
        allow_python_fallback=False,
        concurrency_penalty_scale=float(concurrency_penalty_scale),
    )
    eq = res.equity_curve
    _logger.info(
        "Holdout OOS result: final_balance=%.2f, total_trades=%d, "
        "eq_first=%.2f, eq_last=%.2f",
        res.final_balance, res.total_trades,
        float(eq[0]) if eq.size > 0 else -1.0,
        float(eq[-1]) if eq.size > 0 else -1.0,
    )
    span_days = _segment_span_days(
        full_signal_dfs[ref_sym].iloc[slice_start:slice_end],
        max(holdout_warmup_bars, execution_start_idx),
    )
    cagr = float(portfolio_cagr_pct_from_equity(eq, span_days)) if eq.size > 1 else -100.0
    mdd = float(calc_mdd_from_equity(eq)) if eq.size > 1 else 100.0
    cvar_pct = float(cvar_loss_pct_from_simple_returns(eq)) if eq.size > 1 else 100.0
    pnl = res.pnl_array
    pos_pnl = float(np.sum(pnl[pnl > 0.0]))
    neg_pnl = float(np.abs(np.sum(pnl[pnl < 0.0])))
    pf = pos_pnl / neg_pnl if neg_pnl > 1e-12 else 10.0
    calmar = abs(cagr) / abs(mdd) if abs(mdd) > 1e-6 else 0.0
    win_rate = float(np.sum(pnl > 0.0) / len(pnl)) * 100.0 if len(pnl) > 0 else 0.0

    twr = max(float(res.final_balance / initial_balance), 1e-9)
    tail_r = float(calc_tail_ratio_from_equity(eq)) if eq.size > 1 else 0.0
    dd_bars = float(max_underwater_bars_from_equity(eq)) if eq.size > 1 else 0.0
    final_bal = float(res.final_balance)
    moic = final_bal / initial_balance if initial_balance > 0 else 0.0
    oos_signal_stats_by_symbol: Dict[str, Dict[str, float]] = {}
    for sym in symbols:
        seg = full_signal_dfs[sym].iloc[slice_start:slice_end]
        diag = seg.iloc[execution_start_idx:]
        oos_signal_stats_by_symbol[sym] = _compute_signal_stats(diag)

    out: Dict[str, Any] = {
        "portfolio_cagr_pct": cagr,
        "mdd_pct": mdd,
        "cvar_pct": cvar_pct,
        "tail_ratio": tail_r,
        "long_trades": float(res.total_trades),
        "min_path_tw": twr,
        "dd_bars": dd_bars,
        "final_balance": final_bal,
        "moic": float(moic),
        "equity_curve": eq,
        "oos_signal_stats_by_symbol": oos_signal_stats_by_symbol,
        "profit_factor": pf,
        "calmar_ratio": calmar,
        "win_rate_pct": win_rate,
        "span_days": span_days,
        "per_symbol_trades": res.per_symbol_trades,
        "per_symbol_wins": res.per_symbol_wins,
        "per_symbol_pnl": res.per_symbol_pnl,
    }
    if return_signal_dfs:
        out["full_signal_dfs"] = full_signal_dfs
    return out
