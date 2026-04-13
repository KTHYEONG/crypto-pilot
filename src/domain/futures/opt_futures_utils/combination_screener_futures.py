"""Phase B: (signal x regime x sizing) screening with quick CPCV + balance gate."""

from __future__ import annotations

import itertools
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.storages import InMemoryStorage
from optuna.trial import TrialState

from config.opt_config import OPT_FUTURES_CONFIG
from src.domain.futures.opt_futures_utils.cv_utils import build_cpcv_test_paths_with_fallback
from src.domain.futures.opt_futures_utils.objective import EMBARGO_BARS, objective_futures
from src.domain.futures.opt_futures_utils.opt_params import build_combined_param_space_futures
from src.domain.futures.regimes import FUTURES_REGIME_REGISTRY
from src.domain.futures.signals import FUTURES_SIGNAL_REGISTRY
from src.domain.futures.sizing import FUTURES_SIZING_REGISTRY
from src.domain.futures.strategies_futures import UltimateStrategy

_logger = logging.getLogger("combination_screener_futures")


@dataclass
class CombinationScoreFutures:
    signal: str
    regime: str
    sizing: str
    p10_gmgr: float
    ls_ratio: float
    mean_signal_rate: float
    disqualified: bool = False
    reason: str = ""
    best_params: Dict[str, Any] = field(default_factory=dict)


def build_probe_params_futures(combo: "CombinationScoreFutures", tf: str) -> Dict[str, Any]:
    """Return search-space midpoint params for a Phase-B combo (signal-agnostic for Phase C)."""
    return _build_probe_params_futures(combo.signal, combo.regime, combo.sizing, tf)


def _build_probe_params_futures(sig: str, reg: str, siz: str, tf: str) -> Dict[str, Any]:
    combo_space = build_combined_param_space_futures(sig, reg, siz)
    probe_params: Dict[str, Any] = {
        "SIGNAL_TYPE": sig,
        "REGIME_TYPE": reg,
        "SIZING_METHOD": siz,
        "TIMEFRAME": tf,
        "LEVERAGE": 20,
        "USE_COMPOUNDING": True,
    }
    for k, spec in combo_space.items():
        if k in ("SIGNAL_TYPE", "REGIME_TYPE", "SIZING_METHOD", "TIMEFRAME", "LEVERAGE", "USE_COMPOUNDING"):
            continue
        if not isinstance(spec, dict) or "type" not in spec:
            continue
        if spec["type"] == "categorical":
            probe_params[k] = spec["choices"][0]
        else:
            probe_params[k] = _mid_value_from_spec(spec)
    return probe_params


def _mid_value_from_spec(spec: Dict[str, Any]) -> int | float:
    t = spec["type"]
    if t == "int":
        lo = int(spec["low"])
        hi = int(spec["high"])
        step = int(spec.get("step", 1))
        mid = (lo + hi) // 2
        if step > 1:
            mid = lo + ((mid - lo) // step) * step
        return int(np.clip(mid, lo, hi))
    if t == "float":
        return float((float(spec["low"]) + float(spec["high"])) / 2.0)
    choices = spec.get("choices", ())
    if not choices:
        raise ValueError("categorical spec missing choices")
    return choices[len(choices) // 2]


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


def _auto_combo_trials(
    viable_count: int,
    median_dim: float,
    cfg: Dict[str, Any],
) -> tuple[int, int]:
    """
    Practical auto-sizing for Stage-1 quick CPCV.
    Keep current config as floors, then scale mildly by viable combo count and
    effective combo dimension so runtime stays bounded as the search space grows.
    """
    import math

    v = max(1, int(viable_count))
    d = max(1.0, float(median_dim))
    p1_floor = int(cfg.get("combo_phase1_trials", 10))
    p2_floor = int(cfg.get("combo_phase2_trials", 28))
    p1_cap = max(p1_floor, int(cfg.get("FUTURES_COMBO_PHASE1_MAX", 30)))
    p2_cap = max(p2_floor, int(cfg.get("FUTURES_COMBO_QUICK_TRIALS_MAX", 60)))

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
    """
    If the prune boundary is crowded, add a small Phase-2 budget bump.
    This preserves fast default behavior while spending a bit more only when
    combo ranking is genuinely ambiguous.
    """
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
    """Trigger Numba JIT in the parent before fork so children inherit compiled code."""
    from src.domain.futures.regimes import FUTURES_REGIME_REGISTRY
    from src.domain.futures.signals import FUTURES_SIGNAL_REGISTRY
    from src.domain.futures.sizing import FUTURES_SIZING_REGISTRY

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
    """Stage-1 growth rank: p25 log-TWR scaled by inverse path CV (tmp.md: p25 / (1 + path_cv))."""
    p25 = float(metrics.get("cpcv_p25_log_tw", -1e9))
    if p25 <= -1e8:
        return p25
    cv = float(metrics.get("cpcv_path_cv", 10.0))
    return float(p25 * (1.0 / (1.0 + max(0.0, cv))))


def _phase0_ls_ratio(
    data_maps: Dict[str, Dict[str, Any]],
    symbols: Sequence[str],
    tf: str,
    combo: Tuple[str, str, str],
) -> Tuple[float, float, str]:
    sig, reg, siz = combo
    space = build_combined_param_space_futures(sig, reg, siz)
    probe = _build_probe_params_futures(sig, reg, siz, tf)
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
        import traceback
        _logger.warning("Phase0 signal probe failed: %s\n%s", exc, traceback.format_exc())
        return 0.0, 0.0, "signal_error"
    long_b = (out["trend_direction"].to_numpy(dtype=np.float64) > 0).sum()
    short_b = (out["trend_direction"].to_numpy(dtype=np.float64) < 0).sum()
    tot = int(long_b + short_b)
    if tot == 0:
        return 0.0, 0.0, "no_signals"
    ls_ratio = float(min(long_b, short_b) / max(tot, 1))
    mean_signal_rate = float(tot / len(df))
    return ls_ratio, mean_signal_rate, ""


def _metrics_from_best_study(study: optuna.Study) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Extract screening metrics from a completed Phase-2 study.

    Key-name mapping (objective_futures sets → screener reads):
      growth_score          → objective_final
      gate1_p10_gmgr        → p10_gmgr
      gate1_dsr             → dsr_paths
      psr_paths             → psr_paths  (unchanged)
      cpcv_path_oos_log_tw  → p25/mean/cv computed inline
    """
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    if not completed:
        return {
            "objective_final": -1e9, "cpcv_mean_path_return_pct": -1e9,
            "cpcv_p25_log_tw": -1e9, "cpcv_path_cv": 10.0,
            "psr_paths": -1.0, "dsr_paths": -1.0, "p10_gmgr": -1e9,
        }, {}
    completed.sort(key=lambda tr: float(tr.value or -1e9), reverse=True)
    best = completed[0]
    ua = best.user_attrs

    # Reconstruct path-level statistics from the stored log-TW list.
    raw_log_tw: List[float] = [float(x) for x in ua.get("cpcv_path_oos_log_tw", [])]
    if raw_log_tw:
        arr = np.asarray(raw_log_tw, dtype=np.float64)
        p25_log_tw = float(np.percentile(arr, 25.0)) if arr.size >= 4 else float(np.mean(arr))
        mean_ret_pct = float(np.mean(np.expm1(arr)) * 100.0)
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        path_cv = sd / (abs(mu) + 1e-6)
    else:
        p25_log_tw, mean_ret_pct, path_cv = -1e9, -1e9, 10.0

    metrics: Dict[str, float] = {
        # objective_futures stores final score as "growth_score"
        "objective_final": float(best.value or -1e9),
        "cpcv_mean_path_return_pct": mean_ret_pct,
        "cpcv_p25_log_tw": p25_log_tw,
        "cpcv_path_cv": path_cv,
        # objective_futures uses "psr_paths" (identical name)
        "psr_paths": float(ua.get("psr_paths", 0.0)),
        # objective_futures stores DSR as "gate1_dsr"
        "dsr_paths": float(ua.get("gate1_dsr", -1.0)),
        # objective_futures stores p10 GMGR as "gate1_p10_gmgr"
        "p10_gmgr": float(ua.get("gate1_p10_gmgr", -1e9)),
    }
    return metrics, best.params


def _phase1_worker(
    combo: tuple[str, str, str],
    *,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf: str,
    n_trials: int,
    project_root: str,
    signal_cache_dir: str,
) -> float:
    """Quick Phase-1 CPCV; with fork, data_maps is shared read-only via CoW."""
    from src.domain.futures.opt_futures_utils.objective import compute_multi_alignment_info

    sig, reg, siz = combo
    space = build_combined_param_space_futures(sig, reg, siz)
    embargo = int(EMBARGO_BARS.get(tf, 0))
    alignment_info = compute_multi_alignment_info(data_maps, symbols, tf, embargo)
    prebuilt = alignment_info["cpcv_bundle"] if alignment_info else None

    if not prebuilt:
        ref_sym = symbols[0]
        is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
        ref_df = data_maps[ref_sym][tf]
        if ref_df is not None and not ref_df.empty:
            ref_len = len(ref_df) - is_off
            if ref_len >= 200:
                prebuilt = build_cpcv_test_paths_with_fallback(ref_len, embargo=embargo)

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
            multi_alignment_info=alignment_info,
            signal_disk_cache_root=cache_root,
        )

    study.optimize(_obj, n_trials=max(5, int(n_trials)), n_jobs=1, show_progress_bar=False)
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    if not completed:
        return -1e9
    completed.sort(key=lambda tr: float(tr.value or -1e9), reverse=True)
    ktop = max(1, len(completed) // 5)
    top_trials = completed[:ktop]
    # Use "gate1_p10_gmgr" (the key objective_futures actually sets).
    # Threshold FUTURES_COMBO_PRUNE_THR=-0.05 is in the same decimal-return units.
    return float(
        np.mean([float(t.user_attrs.get("gate1_p10_gmgr", -1e9)) for t in top_trials])
    )


def _phase2_worker(
    combo: tuple[str, str, str],
    *,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf: str,
    n_trials: int,
    project_root: str,
    signal_cache_dir: str,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Full-trial CPCV screen for surviving combinations."""
    from src.domain.futures.opt_futures_utils.objective import compute_multi_alignment_info

    sig, reg, siz = combo
    space = build_combined_param_space_futures(sig, reg, siz)
    embargo = int(EMBARGO_BARS.get(tf, 0))
    alignment_info = compute_multi_alignment_info(data_maps, symbols, tf, embargo)
    prebuilt = alignment_info["cpcv_bundle"] if alignment_info else None

    if not prebuilt:
        ref_sym = symbols[0]
        is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
        ref_df = data_maps[ref_sym][tf]
        if ref_df is not None and not ref_df.empty:
            ref_len = len(ref_df) - is_off
            if ref_len >= 200:
                prebuilt = build_cpcv_test_paths_with_fallback(ref_len, embargo=embargo)

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
            multi_alignment_info=alignment_info,
            signal_disk_cache_root=cache_root,
        )

    study.optimize(_obj, n_trials=max(8, int(n_trials)), n_jobs=1, show_progress_bar=False)
    return _metrics_from_best_study(study)


def _process_pool_context():
    """Prefer fork on Linux (WSL2); fall back to spawn where fork is unavailable."""
    import multiprocessing as mp

    if sys.platform == "win32":
        return mp.get_context("spawn")
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context("spawn")


def run_combination_screening_futures(
    *,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf: str,
    project_root: str,
    signal_cache_dir: str = "",
) -> CombinationScreeningResult:
    import os
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    from tqdm import tqdm

    from config.opt_config import FUTURES_ANCHOR_SYMBOLS

    cfg = OPT_FUTURES_CONFIG
    ls_min = 0.15
    prune_thr = float(cfg.get("FUTURES_COMBO_PRUNE_THR", -0.05))
    top_k = int(cfg.get("combo_top_k", 3))
    bucket_each = 2
    n_workers_cfg = int(cfg.get("n_jobs", 3))
    worker_cap = max(1, (os.cpu_count() or 2) - 1)
    n_workers = max(1, min(n_workers_cfg, worker_cap))

    # Use anchors for Phase B screening
    screen_symbols = [s for s in FUTURES_ANCHOR_SYMBOLS if s in symbols]
    if not screen_symbols:
        _logger.warning("No anchor symbols found in data_maps for Phase B; falling back to all symbols.")
        screen_symbols = symbols

    combos = list(
        itertools.product(
            sorted(FUTURES_SIGNAL_REGISTRY.keys()),
            sorted(FUTURES_REGIME_REGISTRY.keys()),
            sorted(FUTURES_SIZING_REGISTRY.keys()),
        )
    )

    scores: List[CombinationScoreFutures] = []
    viable_items: List[Tuple[Tuple[str, str, str], float, float]] = []
    viable_dims: List[int] = []

    _logger.info("Phase 0: Screening %d combinations for ls_ratio >= %.2f...", len(combos), ls_min)

    min_rate_cfg = float(cfg.get("COMBO_MIN_SIGNAL_RATE", 0.005))

    for sig, reg, siz in combos:
        ls_ratio, mean_signal_rate, reason = _phase0_ls_ratio(
            data_maps, screen_symbols, tf, (sig, reg, siz)
        )
        if mean_signal_rate < min_rate_cfg:
            scores.append(
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
            scores.append(
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
            combo_space = build_combined_param_space_futures(sig, reg, siz)
            viable_dims.append(_combo_search_dim(combo_space))

    _logger.info("Phase 0 complete: %d viable combinations", len(viable_items))

    if not viable_items:
        _logger.error("Stage1: all combinations disqualified after Phase 0.")
        return CombinationScreeningResult(combos=[], phase1_no_edge=True)

    median_dim = float(np.median(np.asarray(viable_dims, dtype=np.float64))) if viable_dims else 1.0
    n_phase1, n_quick = _auto_combo_trials(len(viable_items), median_dim, cfg)
    _logger.info(
        "Stage1 auto trials: phase1=%d, phase2=%d, viable=%d, median_dim=%.1f",
        n_phase1,
        n_quick,
        len(viable_items),
        median_dim,
    )

    _logger.info("Warming up Numba JIT before process pool...")
    _warmup_numba(
        data_maps, screen_symbols, tf, project_root=project_root, signal_cache_dir=signal_cache_dir
    )

    mp_ctx = _process_pool_context()

    phase1_fn = partial(
        _phase1_worker,
        data_maps=data_maps,
        symbols=screen_symbols,
        tf=tf,
        n_trials=n_phase1,
        project_root=project_root,
        signal_cache_dir=signal_cache_dir,
    )

    _logger.info(
        "Phase 1: quick screen (%d trials × %d combos, %d workers)...",
        n_phase1,
        len(viable_items),
        n_workers,
    )
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as pool:
        phase1_results = list(
            tqdm(
                pool.map(phase1_fn, [it[0] for it in viable_items], chunksize=2),
                total=len(viable_items),
                desc="[Combo Screen Phase-1]",
                unit="combo",
            )
        )

    surviving_items: List[Tuple[Tuple[str, str, str], float, float]] = []
    for (combo, ls_ratio, mean_signal_rate), score in zip(viable_items, phase1_results, strict=True):
        if score > prune_thr:
            surviving_items.append((combo, ls_ratio, mean_signal_rate))
        else:
            scores.append(
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
        _logger.warning("Phase 1: all pruned. Relaxing threshold, keeping top-5.")
        surviving_items = [
            item for _, item in sorted(zip(phase1_results, viable_items, strict=True), reverse=True, key=lambda x: x[0])[:5]
        ]
        phase1_no_edge = True

    phase2_boost = _ambiguity_phase2_boost(phase1_results, top_k=max(1, top_k), cfg=cfg)
    if phase2_boost > 0:
        p2_cap = int(cfg.get("FUTURES_COMBO_QUICK_TRIALS_MAX", 60))
        n_quick = min(p2_cap, n_quick + phase2_boost)
        _logger.info("Phase 1 boundary ambiguous: boosting phase2 trials by +%d -> %d", phase2_boost, n_quick)

    phase2_fn = partial(
        _phase2_worker,
        data_maps=data_maps,
        symbols=screen_symbols,
        tf=tf,
        n_trials=n_quick,
        project_root=project_root,
        signal_cache_dir=signal_cache_dir,
    )

    _logger.info(
        "Phase 2: full screen (%d trials × %d combos, %d workers)...",
        n_quick,
        len(surviving_items),
        n_workers,
    )
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as pool:
        phase2_results = list(
            tqdm(
                pool.map(phase2_fn, [it[0] for it in surviving_items], chunksize=1),
                total=len(surviving_items),
                desc="[Combo Screen Phase-2]",
                unit="combo",
            )
        )

    scored_rows: List[Tuple[Tuple[str, str, str], Dict[str, float], Dict[str, Any], float, float]] = []
    for (combo, ls_ratio, mean_signal_rate), (m, best_params) in zip(surviving_items, phase2_results, strict=True):
        scored_rows.append((combo, m, best_params, ls_ratio, mean_signal_rate))

    def _robust_key(m: Dict[str, float]) -> float:
        return float(m.get("psr_paths", 0.0)) + float(max(0.0, m.get("dsr_paths", 0.0)))

    by_growth = sorted(scored_rows, key=lambda x: _p25_path_consistency_score(x[1]), reverse=True)[: max(1, bucket_each)]
    by_robust = sorted(scored_rows, key=lambda x: _robust_key(x[1]), reverse=True)[: max(1, bucket_each)]
    by_balance = sorted(scored_rows, key=lambda x: x[1].get("objective_final", -1e9), reverse=True)[: max(1, bucket_each)]

    seen: set[tuple[str, str, str]] = set()
    bucket_order: List[tuple[str, str, str]] = []
    for bucket in (by_growth, by_robust, by_balance):
        for combo, _, _, _, _ in bucket:
            if combo not in seen:
                seen.add(combo)
                bucket_order.append(combo)

    for combo in bucket_order:
        row = next(r for r in scored_rows if r[0] == combo)
        metrics, best_params, ls_ratio, sig_rate = row[1], row[2], row[3], row[4]
        score = _p25_path_consistency_score(metrics)
        scores.append(
            CombinationScoreFutures(
                signal=combo[0],
                regime=combo[1],
                sizing=combo[2],
                p10_gmgr=score,
                ls_ratio=ls_ratio,
                mean_signal_rate=sig_rate,
                disqualified=False,
                best_params=best_params,
            )
        )

    qualified = [s for s in scores if not s.disqualified]
    qualified.sort(key=lambda x: x.p10_gmgr, reverse=True)

    # ── Rich Screening Report ─────────────────────────────────────────────────
    sep = "=" * 72
    thin = "-" * 72

    _logger.info("\n%s", sep)
    _logger.info(" [STAGE 1. FUTURES COMBINATION SCREENING REPORT]")
    _logger.info("%s", sep)

    if qualified:
        _logger.info("Qualified Combinations  (sort: p25_log_tw / (1 + path_cv))")
        hdr = (
            f"  {'Signal':<18} {'Regime':<18} {'Sizing':<16}"
            f" {'Rate':>6} {'L/S':>5} {'p25/CV':>8}"
        )
        _logger.info("%s", hdr)
        _logger.info("  %s", thin)
        for c in qualified[:10]:
            _logger.info(
                "  %-18s %-18s %-16s %5.1f%% %4.2f %8.4f",
                c.signal, c.regime, c.sizing,
                c.mean_signal_rate * 100.0,
                c.ls_ratio,
                c.p10_gmgr,
            )
    else:
        _logger.warning("  (no qualified combinations)")

    # Disqualification breakdown
    reason_counts = Counter(s.reason for s in scores if s.disqualified)
    n_disq = sum(reason_counts.values())
    _logger.info("%s", thin)
    _logger.info("Exclusion Summary  (total=%d disqualified)", n_disq)
    _label = {
        "signal_sparse": f"Signal Sparse (rate < {min_rate_cfg * 100:.1f}%)",
        "ls_ratio":      "L/S Imbalance (ls_ratio < 0.15)",
        "short_history": "Insufficient History",
        "signal_error":  "Signal Generation Error",
        "no_signals":    "No Signals Produced",
        "phase1_pruned": f"Phase-1 Pruned (p10_gmgr <= {prune_thr:.2f})",
    }
    for reason, cnt in reason_counts.most_common():
        _logger.info("  - %-42s : %d", _label.get(reason, reason), cnt)
    if phase1_no_edge:
        _logger.warning(
            "  Phase-1 no-edge: all combos pruned; top-5 relaxed for Phase-2."
        )

    _logger.info("%s", sep)
    if qualified:
        best = qualified[0]
        _logger.info(
            "Winning Combo -> %s | %s | %s  (p25_score=%.4f, ls=%.2f, rate=%.1f%%)",
            best.signal, best.regime, best.sizing,
            best.p10_gmgr, best.ls_ratio, best.mean_signal_rate * 100.0,
        )
    _logger.info("%s\n", sep)
    # ─────────────────────────────────────────────────────────────────────────

    n_out = max(top_k, len(bucket_order))
    return CombinationScreeningResult(
        combos=qualified[:n_out], phase1_no_edge=phase1_no_edge
    )
