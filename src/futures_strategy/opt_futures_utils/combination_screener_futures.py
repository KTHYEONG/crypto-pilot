"""Phase B: (signal × regime × sizing) screening with quick CPCV + balance gate."""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import optuna
from optuna.storages import InMemoryStorage
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from config.opt_config import OPT_FUTURES_CONFIG
from src.futures_strategy.opt_futures_utils.cv_utils import build_cpcv_test_paths_with_fallback
from src.futures_strategy.opt_futures_utils.evaluator import EMBARGO_BARS, objective_futures
from src.futures_strategy.opt_futures_utils.opt_params import build_combined_param_space_futures
from src.futures_strategy.regimes import FUTURES_REGIME_REGISTRY
from src.futures_strategy.signals import FUTURES_SIGNAL_REGISTRY
from src.futures_strategy.sizing import FUTURES_SIZING_REGISTRY
from src.futures_strategy.strategies_futures import UltimateStrategy

_logger = logging.getLogger("combination_screener_futures")


@dataclass
class CombinationScoreFutures:
    signal: str
    regime: str
    sizing: str
    p10_gmgr: float
    ls_ratio: float
    mean_signal_rate: float
    disqualified: bool
    reason: str = ""


@dataclass(frozen=True)
class CombinationScreeningResult:
    """Phase B output: ranked combos + whether Phase 1 had no edge (strict floor in Phase D)."""

    combos: List[CombinationScoreFutures]
    phase1_no_edge: bool

def _combo_search_dim(space: Dict[str, Any]) -> int:
    dim = 0
    for k, spec in space.items():
        if k in ("SIGNAL_TYPE", "REGIME_TYPE", "SIZING_METHOD"):
            continue
        if not isinstance(spec, dict) or "type" not in spec:
            continue
        dim += 1
    return dim

def _auto_combo_trials(viable_count: int, median_dim: float, cfg: Dict[str, Any]) -> tuple[int, int]:
    import math
    v = max(1, int(viable_count))
    d = max(1.0, float(median_dim))
    p1_floor = int(cfg.get("combo_phase1_trials", 10))
    p2_floor = int(cfg.get("combo_phase2_trials", 28))
    p1_cap = max(p1_floor, int(cfg.get("FUTURES_COMBO_PHASE1_MAX", 18)))
    p2_cap = max(p2_floor, int(cfg.get("FUTURES_COMBO_QUICK_TRIALS_MAX", 40)))
    
    phase1 = int(round(7.0 + 0.75 * math.log2(v + 1.0) + 0.20 * math.sqrt(d)))
    phase2 = int(round(19.0 + 1.25 * math.log2(v + 1.0) + 0.45 * math.sqrt(d)))
    
    phase1 = int(np.clip(phase1, p1_floor, p1_cap))
    phase2 = int(np.clip(phase2, p2_floor, p2_cap))
    if phase2 < phase1 + 12:
        phase2 = min(p2_cap, phase1 + 12)
    return phase1, phase2


def _ambiguity_phase2_boost(
    phase1_scores: Sequence[float],
    top_k: int,
    cfg: Dict[str, Any],
) -> int:
    arr = np.asarray([float(x) for x in phase1_scores if np.isfinite(float(x))], dtype=np.float64)
    if arr.size < max(6, top_k + 3):
        return 0
    arr.sort()
    std = float(np.std(arr))
    if std <= 1e-12:
        return int(cfg.get("FUTURES_COMBO_PHASE2_AMBIGUITY_BOOST", 4))
    edge_idx = max(1, min(arr.size - 1, arr.size - top_k))
    gap = float(arr[edge_idx] - arr[edge_idx - 1])
    ratio = abs(gap) / std
    thr = float(cfg.get("FUTURES_COMBO_AMBIGUITY_STD_RATIO", 0.15))
    if ratio < thr:
        return int(cfg.get("FUTURES_COMBO_PHASE2_AMBIGUITY_BOOST", 4))
    return 0


def _warmup_numba(
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf: str,
    project_root: str = "",
    signal_cache_dir: str = "",
) -> None:
    from src.futures_strategy.regimes import FUTURES_REGIME_REGISTRY
    from src.futures_strategy.signals import FUTURES_SIGNAL_REGISTRY
    from src.futures_strategy.sizing import FUTURES_SIZING_REGISTRY
    from src.futures_strategy.opt_futures_utils.evaluator import objective_futures

    sig = sorted(FUTURES_SIGNAL_REGISTRY.keys())[0]
    reg = sorted(FUTURES_REGIME_REGISTRY.keys())[0]
    siz = sorted(FUTURES_SIZING_REGISTRY.keys())[0]
    space = build_combined_param_space_futures(sig, reg, siz)
    try:
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(n_startup_trials=1, seed=0),
            storage=InMemoryStorage(),
        )

        def _warm_obj(trial: optuna.Trial) -> float:
            return objective_futures(
                trial,
                data_maps,
                list(symbols),
                tf,
                space=space,
                mode="multi",
                project_root=project_root,
                prebuilt_cpcv_bundle=None,
                signal_disk_cache_root=Path(signal_cache_dir) if signal_cache_dir else None,
            )

        study.optimize(
            _warm_obj,
            n_trials=1,
            n_jobs=1,
            show_progress_bar=False,
            catch=(Exception,),
        )
    except Exception as e:
        _logger.warning("Numba warmup failed, proceeding anyway: %s", e)



def _p25_path_consistency_score(metrics: Dict[str, float]) -> float:
    p25 = float(metrics.get("cpcv_p25_log_tw", -1e9))
    if p25 <= -1e8:
        return p25
    cv = float(metrics.get("cpcv_path_cv", 10.0))
    return float(p25 * (1.0 / (1.0 + max(0.0, cv))))


def _mid_params_from_space(space: Dict[str, Any], tf: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"TIMEFRAME": tf, "LEVERAGE": 20, "USE_COMPOUNDING": True}
    for k, spec in space.items():
        if not isinstance(spec, dict) or "type" not in spec:
            continue
        t = spec["type"]
        if t == "categorical":
            ch = spec.get("choices", ())
            out[k] = ch[0] if ch else None
        elif t == "int":
            lo = int(spec["low"])
            hi = int(spec["high"])
            step = int(spec.get("step", 1))
            mid = lo + ((hi - lo) // 2 // step) * step
            out[k] = int(np.clip(mid, lo, hi))
        elif t == "float":
            out[k] = float((float(spec["low"]) + float(spec["high"])) / 2.0)
    return out


def _phase0_ls_ratio(
    data_maps: Dict[str, Dict[str, Any]],
    symbols: Sequence[str],
    tf: str,
    combo: Tuple[str, str, str],
) -> Tuple[float, float, str]:
    sig, reg, siz = combo
    space = build_combined_param_space_futures(sig, reg, siz)
    probe = _mid_params_from_space(space, tf)
    ref = symbols[0]
    target_df = data_maps[ref][tf]
    is_off = int(data_maps[ref].get(f"is_start_idx_{tf}", 0))
    df = target_df.iloc[is_off:].copy()
    if df.empty or len(df) < 50:
        return 0.0, 0.0, "short_history"
    strat = UltimateStrategy(name="probe_ls", params=probe)
    try:
        out = strat.generate_signals(df)
    except Exception as exc:
        _logger.warning("Phase0 signal probe failed: %s", exc)
        return 0.0, 0.0, "signal_error"
    long_b = (out["trend_direction"].to_numpy(dtype=np.float64) > 0).sum()
    short_b = (out["trend_direction"].to_numpy(dtype=np.float64) < 0).sum()
    tot = int(long_b + short_b)
    if tot == 0:
        return 0.0, 0.0, "no_signals"
    ls_ratio = float(min(long_b, short_b) / max(tot, 1))
    mean_signal_rate = float(tot / len(df))
    return ls_ratio, mean_signal_rate, ""


def _metrics_from_best_study(study: optuna.Study) -> Dict[str, float]:
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    if not completed:
        return {
            "objective_final": -1e9,
            "cpcv_p25_log_tw": -1e9,
            "cpcv_path_cv": 10.0,
            "p10_gmgr": -1e9,
        }
    completed.sort(key=lambda tr: float(tr.value or -1e9), reverse=True)
    best = completed[0]
    ua = best.user_attrs
    return {
        "objective_final": float(best.value or -1e9),
        "cpcv_p25_log_tw": float(ua.get("cpcv_p25_log_tw", -1e9)),
        "cpcv_path_cv": float(ua.get("cpcv_path_cv", 10.0)),
        "p10_gmgr": float(ua.get("p10_gmgr", -1e9)),
    }


def run_quick_cpcv_for_combo_futures(
    combo: Tuple[str, str, str],
    *,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf: str,
    n_trials: int,
    project_root: str,
    signal_cache_dir: str,
) -> float:
    sig, reg, siz = combo
    space = build_combined_param_space_futures(sig, reg, siz)
    ref_sym = symbols[0]
    is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
    ref_df = data_maps[ref_sym][tf]
    prebuilt = None
    if ref_df is not None and not ref_df.empty:
        ref_len = len(ref_df) - is_off
        if ref_len >= 200:
            prebuilt = build_cpcv_test_paths_with_fallback(ref_len, embargo=int(EMBARGO_BARS.get(tf, 0)))
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(n_startup_trials=min(15, max(5, n_trials // 3)), seed=42),
        storage=InMemoryStorage(),
    )
    cache_root = Path(signal_cache_dir) if signal_cache_dir else None

    def _obj(trial: optuna.Trial) -> float:
        return objective_futures(
            trial,
            data_maps,
            list(symbols),
            tf,
            space=space,
            mode="multi",
            project_root=project_root,
            prebuilt_cpcv_bundle=prebuilt,
            signal_disk_cache_root=cache_root,
        )

    study.optimize(_obj, n_trials=max(5, int(n_trials)), n_jobs=1, show_progress_bar=False)
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    if not completed:
        return -1e9
    completed.sort(key=lambda tr: float(tr.value or -1e9), reverse=True)
    ktop = max(1, len(completed) // 5)
    top_trials = completed[:ktop]
    return float(
        np.mean([float(t.user_attrs.get("cpcv_mean_path_return_pct", -1e9)) for t in top_trials])
    )


def run_phase2_metrics_futures(
    combo: Tuple[str, str, str],
    *,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf: str,
    n_trials: int,
    project_root: str,
    signal_cache_dir: str,
) -> Dict[str, float]:
    sig, reg, siz = combo
    space = build_combined_param_space_futures(sig, reg, siz)
    ref_sym = symbols[0]
    is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
    ref_df = data_maps[ref_sym][tf]
    prebuilt = None
    if ref_df is not None and not ref_df.empty:
        ref_len = len(ref_df) - is_off
        if ref_len >= 200:
            prebuilt = build_cpcv_test_paths_with_fallback(ref_len, embargo=int(EMBARGO_BARS.get(tf, 0)))
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(n_startup_trials=min(20, max(8, n_trials // 4)), seed=43),
        storage=InMemoryStorage(),
    )
    cache_root = Path(signal_cache_dir) if signal_cache_dir else None

    def _obj(trial: optuna.Trial) -> float:
        return objective_futures(
            trial,
            data_maps,
            list(symbols),
            tf,
            space=space,
            mode="multi",
            project_root=project_root,
            prebuilt_cpcv_bundle=prebuilt,
            signal_disk_cache_root=cache_root,
        )

    study.optimize(_obj, n_trials=max(8, int(n_trials)), n_jobs=1, show_progress_bar=False)
    return _metrics_from_best_study(study)


def run_combination_screening_futures(
    *,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf: str,
    project_root: str,
    signal_cache_dir: str = "",
) -> CombinationScreeningResult:
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    from tqdm import tqdm
    import multiprocessing as mp
    import sys
    import os

    cfg = OPT_FUTURES_CONFIG
    p1 = int(cfg.get("combo_phase1_trials", 30))
    p2 = int(cfg.get("combo_phase2_trials", 80))
    top_k = int(cfg.get("combo_top_k", 3))
    ls_min = 0.15
    n_workers = int(cfg.get("n_jobs", 3))
    n_workers = max(1, min(n_workers, os.cpu_count() or 2))

    combos = list(
        itertools.product(
            sorted(FUTURES_SIGNAL_REGISTRY.keys()),
            sorted(FUTURES_REGIME_REGISTRY.keys()),
            sorted(FUTURES_SIZING_REGISTRY.keys()),
        )
    )
    scored: List[CombinationScoreFutures] = []
    viable_items: List[Tuple[Tuple[str, str, str], float]] = []

    _logger.info("Phase 0: Screening %d combinations for ls_ratio >= %.2f...", len(combos), ls_min)

    min_rate_cfg = float(cfg.get("COMBO_MIN_SIGNAL_RATE", 0.005))

    for sig, reg, siz in combos:
        ls_ratio, mean_signal_rate, reason = _phase0_ls_ratio(data_maps, symbols, tf, (sig, reg, siz))
        if mean_signal_rate < min_rate_cfg:
            scored.append(
                CombinationScoreFutures(
                    signal=sig,
                    regime=reg,
                    sizing=siz,
                    p10_gmgr=-1e9,
                    ls_ratio=ls_ratio,
                    mean_signal_rate=mean_signal_rate,
                    disqualified=True,
                    reason="signal_sparse",
                )
            )
        elif ls_ratio < ls_min:
            scored.append(
                CombinationScoreFutures(
                    signal=sig,
                    regime=reg,
                    sizing=siz,
                    p10_gmgr=-1e9,
                    ls_ratio=ls_ratio,
                    mean_signal_rate=mean_signal_rate,
                    disqualified=True,
                    reason=reason or "ls_ratio",
                )
            )
        else:
            viable_items.append(((sig, reg, siz), ls_ratio, mean_signal_rate))

    _logger.info("Phase 0 complete: %d viable combinations", len(viable_items))

    if not viable_items:
        _logger.warning("No viable futures combos after Phase 0; returning defaults.")
        return CombinationScreeningResult(
            combos=[
                CombinationScoreFutures(
                    "RSM_VT", "EMA_ATR", "vol_target", -1.0, 0.0, 0.0, True, "fallback"
                )
            ],
            phase1_no_edge=False,
        )

    viable_dims: List[int] = []
    for (sig, reg, siz), _, _ in viable_items:
        combo_space = build_combined_param_space_futures(sig, reg, siz)
        viable_dims.append(_combo_search_dim(combo_space))

    median_dim = float(np.median(np.asarray(viable_dims, dtype=np.float64))) if viable_dims else 1.0
    p1_dyn, p2_dyn = _auto_combo_trials(len(viable_items), median_dim, cfg)
    _logger.info(
        "Auto trials scale: phase1=%d, phase2=%d (viable=%d, median_dim=%.1f)",
        p1_dyn, p2_dyn, len(viable_items), median_dim
    )
    p1 = p1_dyn
    p2 = p2_dyn

    _logger.info("Warming up Numba JIT before process pool...")
    _warmup_numba(data_maps, symbols, tf, project_root=project_root, signal_cache_dir=signal_cache_dir)

    viable_combos = [item[0] for item in viable_items]

    def get_mp_ctx():
        if sys.platform == "win32":
            return mp.get_context("spawn")
        try:
            return mp.get_context("fork")
        except ValueError:
            return mp.get_context("spawn")

    mp_ctx = get_mp_ctx()
    
    _logger.info("Phase 1: Quick CPCV (%d trials × %d combos, %d workers)...", p1, len(viable_combos), n_workers)
    
    phase1_fn = partial(
        run_quick_cpcv_for_combo_futures,
        data_maps=data_maps,
        symbols=symbols,
        tf=tf,
        n_trials=p1,
        project_root=project_root,
        signal_cache_dir=signal_cache_dir,
    )
    
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as pool:
        phase1_results = list(
            tqdm(
                pool.map(phase1_fn, viable_combos, chunksize=1),
                total=len(viable_combos),
                desc="[Futures Phase 1]",
                unit="combo",
            )
        )

    prune_thr = float(cfg.get("FUTURES_COMBO_PRUNE_THR", -0.05))
    surviving_items: List[Tuple[Tuple[str, str, str], float, float]] = []
    
    for (combo, ls_ratio, mean_signal_rate), score in zip(viable_items, phase1_results):
        if score > prune_thr:
            surviving_items.append((combo, ls_ratio, mean_signal_rate))
        else:
            scored.append(
                CombinationScoreFutures(
                    signal=combo[0],
                    regime=combo[1],
                    sizing=combo[2],
                    p10_gmgr=score,
                    ls_ratio=ls_ratio,
                    mean_signal_rate=mean_signal_rate,
                    disqualified=True,
                    reason="phase1_pruned",
                )
            )

    _logger.info("Phase 1 complete: %d surviving / %d viable", len(surviving_items), len(viable_items))

    phase1_no_edge = False
    if not surviving_items:
        best_score = max(phase1_results) if phase1_results else -1e9
        _logger.warning(
            "Phase 1: all %d combos pruned (best=%.4f, thr=%.4f). "
            "IS period may lack exploitable edge.",
            len(viable_items),
            best_score,
            prune_thr,
        )
        relaxed_thr = 0.0
        surviving_items = [
            item for item, score in zip(viable_items, phase1_results) if score > relaxed_thr
        ]
        if not surviving_items:
            surviving_items = [
                x
                for _, x in sorted(
                    zip(phase1_results, viable_items), reverse=True, key=lambda pair: pair[0]
                )[:3]
            ]
            phase1_no_edge = True
            _logger.error(
                "No edge found after relaxed threshold. Using best-3 combos; Phase D applies strict objective floor."
            )

    phase2_boost = _ambiguity_phase2_boost(phase1_results, top_k=max(1, top_k), cfg=cfg)
    if phase2_boost > 0:
        p2_cap = int(cfg.get("FUTURES_COMBO_QUICK_TRIALS_MAX", 40))
        p2 = min(p2_cap, p2 + phase2_boost)
        _logger.info("Phase 2 Ambiguity Boost applied (+%d trials) -> %d trials/combo", phase2_boost, p2)

    survivor_combos = [s[0] for s in surviving_items]

    _logger.info("Phase 2: Full CPCV (%d trials × %d combos, %d workers)...", p2, len(survivor_combos), n_workers)
    
    phase2_fn = partial(
        run_phase2_metrics_futures,
        data_maps=data_maps,
        symbols=symbols,
        tf=tf,
        n_trials=p2,
        project_root=project_root,
        signal_cache_dir=signal_cache_dir,
    )

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as pool:
        phase2_results = list(
            tqdm(
                pool.map(phase2_fn, survivor_combos, chunksize=1),
                total=len(survivor_combos),
                desc="[Futures Phase 2]",
                unit="combo",
            )
        )

    finals: List[CombinationScoreFutures] = []
    for item, m in zip(surviving_items, phase2_results):
        combo, ls_ratio, mean_signal_rate = item
        score = _p25_path_consistency_score(m)
        finals.append(
            CombinationScoreFutures(
                signal=combo[0],
                regime=combo[1],
                sizing=combo[2],
                p10_gmgr=score,
                ls_ratio=ls_ratio,
                mean_signal_rate=mean_signal_rate,
                disqualified=False,
                reason="",
            )
        )

    finals.sort(key=lambda x: x.p10_gmgr, reverse=True)
    
    for s in scored:
        finals.append(s)
        
    valid_finals = [s for s in finals if not s.disqualified]
    return CombinationScreeningResult(combos=valid_finals[:top_k], phase1_no_edge=phase1_no_edge)

